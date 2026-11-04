"""Long-running service: fetch feeds, poll replies, send the daily digest."""

import logging
import signal
import threading
import time
from collections.abc import Callable
from datetime import datetime

from . import inbox, ranker
from .config import Config
from .daemon import gather, send_digest
from .database import Database
from .llm import LLM, LLMError

log = logging.getLogger(__name__)


class Service:
    """Runs each job on its own thread, so a slow LLM never holds up the others.

    - fetch:  every fetch_interval_minutes, fetch feeds, pre-filter, download text
    - score:  LLM-scores the shortlist whenever fetch or a profile change asks for it
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
        self.score_wanted = threading.Event()
        self.last_prune = 0.0
        self.next_digest_attempt = 0.0
        # What each worker thread is doing, for the digest's debug report. Each entry is
        # replaced whole, so the main thread never sees one half-updated.
        self.jobs: dict[str, dict] = {name: {} for name in ("feed fetch", "scoring", "inbox poll")}

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
        new = gather(self.config, database, self.llm, stop=self.stop)
        self.score_wanted.set()
        return f"{new} new articles"

    def _score(self, database: Database) -> str:
        try:
            return f"scored {ranker.score_candidates(self.config, database, self.llm, stop=self.stop)}"
        except LLMError as error:
            log.warning("scoring skipped: %s", error)
            return f"skipped: {error}"

    def _poll_and_learn(self, database: Database) -> str:
        used = inbox.poll(self.config, database, self.llm)
        if used:
            log.info("processed %d replies", used)
        # Also retries pending feedback if an earlier update failed.
        if ranker.update_profile(self.config, database, self.llm):
            self.score_wanted.set()  # rescore the shortlist against the new profile
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
        """Run `job` whenever score_wanted is set (also once at start)."""
        self.score_wanted.set()
        while True:
            self.score_wanted.wait()
            if self.stop.is_set():
                return
            # Clear first: a request made while the job runs triggers another run.
            self.score_wanted.clear()
            self._run_job(name, job, database)

    def _shutdown(self, *_: object):
        self.stop.set()
        self.score_wanted.set()  # wake the scoring thread so it can see the stop

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
            threading.Thread(target=self._with_database, name="score", daemon=True,
                             args=(self._on_demand, "scoring", self._score)),
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
