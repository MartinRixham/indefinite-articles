"""Long-running service: fetch feeds, poll replies, send the daily digest."""

import logging
import signal
import threading
import time
from datetime import datetime, timedelta

from . import __version__, embed, extract, feeds, inbox, mailer, ranker
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


def _ago(ts) -> str:
    if not ts:
        return "never"
    mins = int((time.time() - ts) / 60)
    return f"{mins}m ago" if mins < 120 else f"{mins // 60}h ago"


def _classifier_status(db: DB) -> str:
    likes, _, dislikes = db.get_state("classifier_info", "0:0").partition(":")
    if db.get_state("classifier", ""):
        return f"classifier: trained on {likes} likes / {dislikes} dislikes"
    return (f"classifier: not trained yet ({likes} likes / {dislikes} dislikes with embeddings;"
            f" needs {embed.MIN_PER_CLASS} of each)")


def debug_report(cfg, db: DB, items, info: dict) -> list[str]:
    """Plain-text diagnostics appended to the digest: pipeline health and why each item was picked."""
    sel = info.get("select", {})
    st = db.stats()
    profile = db.current_profile()
    fb, comments = db.feedback_since(profile["created_at"]) if profile else ([], [])
    states = db.feed_states()
    never = sum(f.url not in states for f in cfg.feeds)
    last_ok = max((s["last_success"] or 0 for s in states.values()), default=0)
    new_24h = db.conn.execute("SELECT COUNT(*) FROM articles WHERE fetched_at >= ?",
                              (time.time() - 86400,)).fetchone()[0]
    t = info.get("timings", {})
    lines = [
        f"version {__version__}, built {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"llm: {cfg.llm.model} at {cfg.llm.base_url}",
        f"embeddings: {cfg.llm.embedding_model or 'disabled'}, {st['embedded']} stored",
        f"feeds: {len(cfg.feeds)} configured, {st['feeds_failing']} failing, {never} never fetched,"
        f" last success {_ago(last_ok)}",
        f"articles: {new_24h} new in 24h, {st['articles']} in db,"
        f" full text {st['full_text']} ok / {st['full_text_failed']} failed",
        f"selection: {sel.get('candidates', 0)} candidates, {sel.get('llm_scored', 0)} scored"
        f" against current profile, {sel.get('stale_scored', 0)} with stale scores",
        f"profile: v{profile['id'] if profile else '-'} ({profile['note'] if profile else ''}),"
        f" {len(fb)} ratings + {len(comments)} comments pending",
        f"feedback total: +{st['likes']} / -{st['dislikes']}, {st['digests']} digests sent",
        _classifier_status(db),
        "timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in t.items()),
        "",
        "#   rank  final =   llm + adj  + src   text      source",
    ]
    scores, rank = sel.get("scores", {}), sel.get("rank", {})
    for pos, (a, exploratory) in enumerate(items, start=1):
        s = scores.get(a["id"])
        if s is None:
            continue
        llm_s = f"{s['llm']:5.1f}" if s["llm"] is not None else "    -"
        lines.append(
            f"{pos:<3} {rank.get(a['id'], 0):>4}  {s['final']:5.2f} = {llm_s} {s['base']:+5.2f}"
            f" {s['bonus']:+5.2f}   {(a['content_status'] or '-')[:8]:8}  {a['source'][:30]}"
            + ("  [explore]" if exploratory else "")
        )
    lines.append("(adj: embedding agreement, or fallback rank when the LLM did not score;"
                 " src: source like/dislike bonus)")
    return lines


def send_digest(cfg, db: DB, dry_run: bool = False):
    """Select and send today's digest from the scores the hourly runs left behind.

    It never calls the LLM, so the email goes out on time however slow the LLM is.
    Articles not scored yet stay candidates for later digests. Returns the
    EmailMessage or None.
    """
    note = ""
    info: dict = {"timings": {}, "select": {}}

    def timed(name, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            info["timings"][name] = time.monotonic() - t0

    items = timed("select", ranker.select, cfg, db, debug=info["select"])
    sel = info["select"]
    if items and not sel.get("llm_scored") and not sel.get("stale_scored"):
        note = "(The local LLM has not scored any of these yet, so today's list was ranked by simple heuristics.)"
    if not items:
        log.warning("no candidate articles for today's digest")
        return None
    debug = None
    if cfg.email.debug_info:
        try:
            debug = debug_report(cfg, db, items, info)
        except Exception:  # diagnostics must never stop the digest going out
            log.exception("building debug report failed")
    token = mailer.new_token()
    msg = mailer.build_digest(cfg, items, token, note, debug)
    if dry_run:
        return msg
    mailer.send(cfg, msg)
    db.record_digest(token, msg["Message-ID"], datetime.now().date().isoformat(),
                     [(a["id"], exp) for a, exp in items])
    return msg


class _StopAt:
    """Stop flag that is also set once a wall-clock deadline passes.

    Only reads the clock, so the feed fetcher's worker threads can check it too.
    """

    def __init__(self, stop: threading.Event, deadline: float):
        self.stop, self.deadline = stop, deadline

    def is_set(self) -> bool:
        return self.stop.is_set() or time.time() >= self.deadline


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

    def _next_digest_time(self) -> float:
        """When the next digest is due (epoch seconds), at least a moment from now."""
        h, m = self.cfg.digest_hour_minute
        now = datetime.now()
        at = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if at <= now:
            at += timedelta(days=1)
        return at.timestamp()

    def _run_job(self, name, fn):
        try:
            fn()
        except Exception:
            log.exception("%s failed", name)

    def fetch_and_score(self):
        # Give way to the digest: stop at digest time and leave the rest for the next run.
        refresh(self.cfg, self.db, self.llm, stop=_StopAt(self.stop, self._next_digest_time()))

    def poll_and_learn(self):
        used = inbox.poll(self.cfg, self.db, self.llm)
        if used:
            log.info("processed %d replies", used)
        # Also retries pending feedback if an earlier update failed.
        ranker.update_profile(self.cfg, self.db, self.llm)

    def digest(self):
        today = datetime.now().date().isoformat()
        msg = send_digest(self.cfg, self.db)
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
            # The digest goes first, so other jobs never delay it.
            if now >= self.next_digest_attempt and self._digest_due():
                # If sending fails (e.g. SMTP down) retry in 15 minutes, not every tick.
                self.next_digest_attempt = now + 900
                self._run_job("digest", self.digest)
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
            if now - self.last_prune >= 86400:
                self.last_prune = now
                self._run_job("prune", self.db.prune)
            self.stop.wait(self.TICK)
        log.info("indefinite-articles service stopping")
        self.db.close()
