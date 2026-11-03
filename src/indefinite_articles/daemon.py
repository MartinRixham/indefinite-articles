"""Long-running service: fetch feeds, poll replies, send the daily digest."""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime

from . import embed, extract, feeds, inbox, mailer, ranker
from .db import DB
from .llm import LLM, LLMError

log = logging.getLogger(__name__)


def prefilter(cfg, db: DB, llm: LLM, stop=None):
    """Embed new articles and update pre-filter scores (skipped if no embedding model)."""
    if not cfg.llm.embedding_model:
        return
    try:
        embed.embed_new(cfg, db, llm, stop=stop)
        embed.update_prefilter(cfg, db, llm, ranker.ensure_profile(cfg, db))
    except LLMError as e:
        log.warning("embedding pre-filter unavailable, using newest articles instead: %s", e)


def refresh(cfg, db: DB, llm: LLM, stop=None) -> int:
    """Fetch feeds, pre-filter, fetch full text for the shortlist, and LLM-score it."""
    n = feeds.update_all(cfg, db, stop=stop)
    prefilter(cfg, db, llm, stop)
    extract.enrich_candidates(cfg, db, stop=stop)
    try:
        ranker.score_candidates(cfg, db, llm, stop=stop)
    except LLMError as e:
        log.warning("scoring skipped: %s", e)
    return n


def send_digest(cfg, db: DB, llm: LLM, dry_run: bool = False, stop=None):
    """Score, select and send today's digest. Returns the EmailMessage or None."""
    note = ""
    prefilter(cfg, db, llm, stop)
    extract.enrich_candidates(cfg, db, stop=stop)
    try:
        llm.ping()
        ranker.score_candidates(cfg, db, llm, stop=stop)
    except LLMError as e:
        log.warning("LLM unavailable for digest, using heuristic ranking: %s", e)
        note = "(The local LLM was unavailable, so today's list was ranked by simple heuristics.)"
    items = ranker.select(cfg, db)
    if not items:
        log.warning("no candidate articles for today's digest")
        return None
    token = mailer.new_token()
    msg = mailer.build_digest(cfg, items, token, note)
    if dry_run:
        return msg
    mailer.send(cfg, msg)
    db.record_digest(token, msg["Message-ID"], datetime.now().date().isoformat(),
                     [(a["id"], exp) for a, exp in items])
    return msg


class Service:
    TICK = 20  # seconds

    def __init__(self, cfg):
        self.cfg = cfg
        self.db = DB(cfg.database)
        self.llm = LLM(cfg.llm)
        self.stop = threading.Event()
        self.last_fetch = 0.0
        self.last_poll = 0.0
        self.last_prune = 0.0
        self.next_digest_attempt = 0.0

    def _digest_due(self) -> bool:
        now = datetime.now()
        h, m = self.cfg.digest_hour_minute
        today = now.date().isoformat()
        if self.db.get_state("last_digest_date") == today:
            return False
        return (now.hour, now.minute) >= (h, m)

    def _run_job(self, name, fn):
        try:
            fn()
        except Exception:
            log.exception("%s failed", name)

    def fetch_and_score(self):
        refresh(self.cfg, self.db, self.llm, stop=self.stop)

    def poll_and_learn(self):
        used = inbox.poll(self.cfg, self.db, self.llm)
        if used:
            log.info("processed %d replies", used)
        # Also retries pending feedback if an earlier update failed.
        ranker.update_profile(self.cfg, self.db, self.llm)

    def digest(self):
        today = datetime.now().date().isoformat()
        msg = send_digest(self.cfg, self.db, self.llm, stop=self.stop)
        # Mark the day done even if there was nothing to send, so we don't retry every tick.
        self.db.set_state("last_digest_date", today)
        if msg is not None:
            log.info("digest sent: %s", msg["Subject"])

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop.set())
        ranker.ensure_profile(self.cfg, self.db)
        log.info("indefinite-articles service started: %d feeds, digest at %s, model %s at %s",
                 len(self.cfg.feeds), self.cfg.digest_time, self.cfg.llm.model, self.cfg.llm.base_url)
        while not self.stop.is_set():
            now = time.monotonic()
            if not self.last_fetch or now - self.last_fetch >= self.cfg.fetch_interval_minutes * 60:
                self.last_fetch = now
                self._run_job("feed fetch", self.fetch_and_score)
            if self.stop.is_set():
                break
            if not self.last_poll or now - self.last_poll >= self.cfg.poll_interval_minutes * 60:
                self.last_poll = now
                self._run_job("inbox poll", self.poll_and_learn)
            if self.stop.is_set():
                break
            if now >= self.next_digest_attempt and self._digest_due():
                # If sending fails (e.g. SMTP down) retry in 15 minutes, not every tick.
                self.next_digest_attempt = now + 900
                self._run_job("digest", self.digest)
            if now - self.last_prune >= 86400:
                self.last_prune = now
                self._run_job("prune", self.db.prune)
            self.stop.wait(self.TICK)
        log.info("indefinite-articles service stopping")
        self.db.close()
