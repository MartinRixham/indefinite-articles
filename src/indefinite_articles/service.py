"""Long-running service: fetch feeds, poll replies, send the daily digest."""

import logging
import signal
import threading
import time
from collections.abc import Callable
from datetime import datetime

from .config import Config
from .daemon import rank, send_digest
from .fetching import feeds
from .language_model.llm import LLM, LLMError
from .mail import inbox
from .ranking import ranker
from .storage.database import Database

log = logging.getLogger(__name__)


class Service:
    """Runs each job on its own thread, so a slow LLM never holds up the others.

    - fetch:  every fetch_interval_minutes, fetch feeds
    - rank:   whenever fetch or a profile change asks for it, embed and pre-filter, then
              download full text for the shortlist and LLM-score it (in that order, so
              scoring never works from half-finished embeddings). The first run waits
              for the first fetch, so the articles it brings are embedded before any
              scoring.
    - inbox:  every poll_interval_minutes, read replies and rewrite the profile
    - main:   sends the digest at digest_time (no LLM) and prunes old data

    Each thread has its own database connection (SQLite in WAL mode).
    """

    TICK: float = 20  # seconds

    def __init__(self, config: Config):
        self.config = config
        self.database = Database(config.database)
        self.llm = LLM(config.llm, deadline=config.next_digest_time)
        self.stop = threading.Event()
        self.rank_wanted = threading.Event()
        self.last_prune = 0.0
        self.next_digest_attempt = 0.0
        # What each worker thread is doing, for the digest's debug report. Each entry is
        # replaced whole, so the main thread never sees one half-updated.
        self.jobs: dict[str, dict] = {name: {} for name in ("feed fetch", "ranking", "inbox poll")}

    def _digest_due(self) -> bool:
        now = datetime.now()
        if self.database.get_state("last_digest_date") == now.date().isoformat():
            return False
        return self.config.digest_hour_minute.has_passed(now)

    def _run_job(self, name: str, function: Callable[..., object], *args: object):
        tracked = name in self.jobs  # the worker jobs; never adds keys from another thread
        started = time.time()
        if tracked:
            self.jobs[name] = {**self.jobs[name], "running_since": started}
        result, error = None, None
        try:
            result = function(*args)
        except Exception as failure:
            log.exception("%s failed", name)
            error = str(failure)[:150] or type(failure).__name__
        if tracked:
            end = time.time()
            self.jobs[name] = {"running_since": None, "last_end": end, "last_seconds": end - started,
                               "last_result": result, "last_error": error}

    # -- jobs (each gets the database connection of the thread it runs on) ---------
    def _fetch(self, database: Database) -> str:
        try:
            new = feeds.update_all(self.config, database, stop=self.stop)
        finally:
            self.rank_wanted.set()  # even after a failure: ranking's first run waits for this
        return f"{new} new articles"

    def _rank(self, database: Database) -> str:
        try:
            return f"scored {rank(self.config, database, self.llm, stop=self.stop)}"
        except LLMError as error:
            log.warning("scoring skipped: %s", error)
            return f"skipped: {error}"

    def _poll_and_learn(self, database: Database) -> str:
        used = inbox.poll(self.config, database, self.llm)
        if used:
            log.info("processed %d replies", used)
        # Also retries pending feedback if an earlier update failed.
        if ranker.update_profile(self.config, database, self.llm):
            self.rank_wanted.set()  # pre-filter and rescore the shortlist against the new profile
            return f"{used} replies, profile updated"
        return f"{used} replies"

    def _digest(self):
        today = datetime.now().date().isoformat()
        message = send_digest(self.config, self.database,
                          status={"jobs": {key: dict(value) for key, value in self.jobs.items()}, "llm": self.llm})
        # Mark the day done even if there was nothing to send, so we don't retry every tick.
        self.database.set_state("last_digest_date", today)
        if message is not None:
            log.info("digest sent: %s", message["Subject"])

    # -- threads -----------------------------------------------------------------
    def _with_database(self, loop: Callable[..., None], *args: object):
        database = Database(self.config.database)
        try:
            loop(database, *args)
        finally:
            database.close()

    def _every(self, database: Database, name: str, minutes: int, job: Callable[[Database], str]):
        """Run `job` now and then every `minutes`, measured from the start of each run."""
        while not self.stop.is_set():
            started = time.monotonic()
            self._run_job(name, job, database)
            self.stop.wait(max(0.0, minutes * 60 - (time.monotonic() - started)))

    def _on_demand(self, database: Database, name: str, job: Callable[[Database], str]):
        """Run `job` whenever rank_wanted is set. It isn't set at start: the first
        fetch sets it, so the first run sees the articles that fetch brings."""
        while True:
            self.rank_wanted.wait()
            if self.stop.is_set():
                return
            # Clear first: a request made while the job runs triggers another run.
            self.rank_wanted.clear()
            self._run_job(name, job, database)

    def _shutdown(self, *_: object):
        self.stop.set()
        self.rank_wanted.set()  # wake the ranking thread so it can see the stop

    def run(self) -> None:
        for signature in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signature, self._shutdown)
        ranker.ensure_profile(self.config, self.database)
        log.info("indefinite-articles service started: %d feeds, digest at %s, model %s at %s",
                 len(self.config.feeds), self.config.digest_time, self.config.llm.model, self.config.llm.base_url)
        # Daemon threads: one stuck in a slow LLM request must not stop the service exiting.
        threads = [
            threading.Thread(target=self._with_database, name="fetch", daemon=True,
                             args=(self._every, "feed fetch", self.config.fetch_interval_minutes, self._fetch)),
            threading.Thread(target=self._with_database, name="rank", daemon=True,
                             args=(self._on_demand, "ranking", self._rank)),
            threading.Thread(target=self._with_database, name="inbox", daemon=True,
                             args=(self._every, "inbox poll", self.config.poll_interval_minutes,
                                   self._poll_and_learn)),
        ]
        for thread in threads:
            thread.start()
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= self.next_digest_attempt and self._digest_due():
                # If sending fails (e.g. SMTP down) retry in 15 minutes, not every tick.
                self.next_digest_attempt = now + 900
                self._run_job("digest", self._digest)
            if now - self.last_prune >= 86400:
                self.last_prune = now
                self._run_job("prune", self.database.prune)
            self.stop.wait(self.TICK)
        log.info("indefinite-articles service stopping")
        for thread in threads:
            thread.join(timeout=5)
        self.database.close()
