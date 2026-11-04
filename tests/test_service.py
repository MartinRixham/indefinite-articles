import shutil
import signal
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from unittest import mock

from indefinite_articles import daemon
from indefinite_articles.config import Config
from indefinite_articles.fetching import extract, feeds
from indefinite_articles.language_model.llm import LLMError
from indefinite_articles.mail import inbox
from indefinite_articles.ranking import ranker
from indefinite_articles.service import Service
from indefinite_articles.storage.database import Database


class ServiceTests(unittest.TestCase):
    """Drives the service through run() and stops it the way SIGTERM does."""

    def setUp(self) -> None:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        self.config = Config()
        self.config.database = str(directory / "service.db")
        self.first_scoring = threading.Event()
        self.score = mock.Mock(return_value=0)
        self.score.side_effect = lambda *args, **kwargs: (self.first_scoring.set(), self.score.return_value)[1]
        # Quiet defaults: fetching waits until the service stops, the inbox is empty and
        # today's digest has already gone out.
        for patch in (mock.patch.object(feeds, "update_all", self._fetch_nothing),
                      mock.patch.object(daemon, "prefilter", lambda *args: None),
                      mock.patch.object(extract, "enrich_candidates", lambda *args, **kwargs: 0),
                      mock.patch.object(ranker, "score_candidates", self.score),
                      mock.patch.object(inbox, "poll", return_value=0),
                      mock.patch.object(ranker, "update_profile", return_value=False)):
            patch.start()
            self.addCleanup(patch.stop)
        self._set_last_digest_date(datetime.now().date().isoformat())

    @staticmethod
    def _fetch_nothing(*args: object, stop: threading.Event, **kwargs: object) -> int:
        stop.wait()
        return 0

    def _set_last_digest_date(self, value: str) -> None:
        database = Database(self.config.database)
        database.set_state("last_digest_date", value)
        database.connection.commit()
        database.close()

    def _run_service(self, until: Callable[[dict[str, dict]], bool]) -> dict[str, dict]:
        """Run the service until `until(service.jobs)` holds, then stop it; returns its jobs."""
        services: list[Service] = []

        def run() -> None:  # the service must be built on the thread that runs it
            service = Service(self.config)
            service.TICK = 0.01
            services.append(service)
            service.run()

        with mock.patch.object(signal, "signal") as register_handler:
            thread = threading.Thread(target=run)
            thread.start()
            try:
                deadline = time.monotonic() + 5
                while not (services and until(services[0].jobs)):
                    self.assertLess(time.monotonic(), deadline, f"timed out; jobs: {services and services[0].jobs}")
                    time.sleep(0.01)
            finally:
                for call in register_handler.call_args_list[:1]:  # as if sent SIGTERM
                    call.args[1](signal.SIGTERM, None)
                thread.join(10)
        self.assertFalse(thread.is_alive())
        return services[0].jobs

    def test_fetch_asks_for_scoring(self) -> None:
        def fetch_after_first_scoring(*args: object, **kwargs: object) -> int:
            self.first_scoring.wait(5)
            return 4

        with mock.patch.object(feeds, "update_all", fetch_after_first_scoring):
            jobs = self._run_service(lambda jobs: self.score.call_count >= 2)
        self.assertEqual(jobs["feed fetch"]["last_result"], "4 new articles")

    def test_score_reports_skips(self) -> None:
        self.score.return_value = 7
        jobs = self._run_service(lambda jobs: "last_end" in jobs["scoring"])
        self.assertEqual(jobs["scoring"]["last_result"], "scored 7")
        self.score.side_effect = LLMError("down")
        with self.assertLogs("indefinite_articles.service", "WARNING"):
            jobs = self._run_service(lambda jobs: "last_end" in jobs["scoring"])
        self.assertEqual(jobs["scoring"]["last_result"], "skipped: down")

    def test_poll_and_learn(self) -> None:
        with mock.patch.object(inbox, "poll", return_value=2):
            jobs = self._run_service(lambda jobs: "last_end" in jobs["inbox poll"] and "last_end" in jobs["scoring"])
        self.assertEqual(jobs["inbox poll"]["last_result"], "2 replies")
        self.assertEqual(self.score.call_count, 1)  # only the run at startup

    def test_profile_update_asks_for_scoring(self) -> None:
        def update_after_first_scoring(*args: object) -> bool:
            self.first_scoring.wait(5)
            return True

        with mock.patch.object(ranker, "update_profile", update_after_first_scoring):
            jobs = self._run_service(lambda jobs: self.score.call_count >= 2)
        self.assertEqual(jobs["inbox poll"]["last_result"], "0 replies, profile updated")

    def test_digest_marks_the_day_even_when_empty(self) -> None:
        self.config.digest_time = "00:00"
        self._set_last_digest_date("")
        today = datetime.now().date().isoformat()

        def digest_done(jobs: dict[str, dict]) -> bool:
            database = Database(self.config.database)
            try:
                return database.get_state("last_digest_date") == today
            finally:
                database.close()

        with self.assertLogs("indefinite_articles.daemon", "WARNING"):
            self._run_service(digest_done)


if __name__ == "__main__":
    unittest.main()
