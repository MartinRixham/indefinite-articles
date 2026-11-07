import contextlib
import io
import shutil
import tempfile
import unittest
from email.message import EmailMessage
from pathlib import Path
from unittest import mock

from indefinite_articles import cli, daemon
from indefinite_articles.config import Config
from indefinite_articles.fetching import extract, feeds
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.fetching.feed import Feed
from indefinite_articles.language_model.llm import LLM, LLMError, LLMUnavailable
from indefinite_articles.mail import mailer
from indefinite_articles.ranking import embed, ranker
from indefinite_articles.service import Service
from indefinite_articles.storage.article import Article
from indefinite_articles.storage.database import Database


def config_with_articles(count: int = 3) -> tuple[Config, Database]:
    config = Config()
    config.email.sender, config.email.to = "bot@example.org", "you@example.com"
    config.fetch_full_articles = False
    config.articles_per_digest = 2
    config.exploration = 0
    database = Database(":memory:")
    for index in range(count):
        database.add_article(Entry(f"https://e/{index}", f"Article {index}", "", None), f"S{index}")
    database.connection.commit()
    return config, database


class DaemonFunctionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.database = config_with_articles()
        self.addCleanup(self.database.close)
        self.llm = mock.Mock(spec=LLM)

    def test_prefilter_skipped_without_embedding_model(self):
        self.config.llm.embedding_model = ""
        with mock.patch.object(embed, "embed_new", side_effect=AssertionError("embedded")):
            daemon.prefilter(self.config, self.database, self.llm)

    def test_prefilter_tolerates_llm_failure(self):
        self.llm.embed.side_effect = LLMError("down")
        with self.assertLogs("indefinite_articles.daemon", "WARNING"):
            daemon.prefilter(self.config, self.database, self.llm)

    def test_stalled_embedding_stops_the_round_before_scoring(self):
        self.llm.embed.side_effect = LLMUnavailable("stopped responding")
        with mock.patch.object(ranker, "score_candidates", side_effect=AssertionError("scored")), \
             self.assertRaises(LLMUnavailable):
            daemon.rank(self.config, self.database, self.llm)

    def test_refresh_runs_the_pipeline_and_tolerates_scoring_failure(self):
        calls = []

        def score(*args: object, **kwargs: object) -> int:
            calls.append("score")
            raise LLMError("down")

        with mock.patch.object(feeds, "update_all", lambda *args, **kwargs: calls.append("fetch") or 5), \
             mock.patch.object(daemon, "prefilter", lambda *args: calls.append("prefilter")), \
             mock.patch.object(extract, "enrich_candidates", lambda *args, **kwargs: calls.append("extract")), \
             mock.patch.object(ranker, "score_candidates", score), \
             self.assertLogs("indefinite_articles.daemon", "WARNING"):
            self.assertEqual(daemon.refresh(self.config, self.database, self.llm), 5)
        self.assertEqual(calls, ["fetch", "prefilter", "extract", "score"])

    def test_send_digest_nothing_to_send(self):
        _, empty = config_with_articles(0)
        self.addCleanup(empty.close)
        with self.assertLogs("indefinite_articles.daemon", "WARNING"):
            self.assertIsNone(daemon.send_digest(self.config, empty))

    def test_send_digest_records_only_when_sent(self):
        sent = []
        with mock.patch.object(mailer, "send", lambda config, message: sent.append(message)):
            daemon.send_digest(self.config, self.database, dry_run=True)
            self.assertEqual((sent, self.database.statistics()["digests"]), ([], 0))
            message = daemon.send_digest(self.config, self.database)
        assert message is not None
        self.assertEqual(sent, [message])
        digest = self.database.find_digest(message["X-Indefinite-Articles-Digest"])
        assert digest is not None
        self.assertEqual(digest["message_id"], message["Message-ID"])
        self.assertEqual(len(self.database.digest_items(digest["id"])), 2)
        self.assertEqual(len(self.database.pool(3)), 1)  # sent articles are not offered again

    def test_send_digest_survives_broken_debug_report(self):
        with mock.patch.object(self.database, "statistics", side_effect=KeyError("oops")), \
             self.assertLogs("indefinite_articles.daemon", "ERROR"):
            message = daemon.send_digest(self.config, self.database, dry_run=True)
        assert message is not None
        self.assertNotIn("-- debug --", mailer.part_text(message))

    def test_debug_report(self):
        self.config.feeds = [Feed("https://feed/a"), Feed("https://feed/b")]
        self.database.feed_ok("https://feed/a", "A", feeds.CacheValidators())
        ranker.ensure_profile(self.config, self.database)
        status = {"jobs": {"ranking": {"running_since": 1.0}, "inbox poll": {}}, "llm": LLM(self.config.llm)}
        message = daemon.send_digest(self.config, self.database, dry_run=True, status=status)
        assert message is not None
        text = mailer.part_text(message).split("-- debug --\n", 1)[1]
        lines = text.splitlines()
        self.assertIn("feeds: 2 configured, 0 failing, 1 never fetched", text)
        self.assertIn("selection: 3 candidates, 0 scored", text)
        self.assertIn("timings: select ", text)
        self.assertIn("  inbox poll: idle, not run yet", text)
        self.assertIn("llm stall limits: none learned yet", text)
        self.assertIn("classifier: not trained yet", text)
        self.assertEqual(sum(line[:1].isdigit() for line in lines), self.config.articles_per_digest)  # one row per item


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.config_path = self.directory / "config.toml"
        self.config_path.write_text(
            f'[general]\ndatabase = "{self.directory}/data.db"\nfetch_full_articles = false\n'
            '[profile]\ninterests = "Rust"\n'
            '[email]\nto = "you@example.com"\nfrom = "bot@example.org"\n'
            '[[feeds]]\nurl = "https://feed/a"\nname = "Feed A"\ncategory = "Tech"\n'
            '[[feeds]]\nurl = "https://feed/b"\n')

    def _run(self, *argv: str) -> tuple[int, str, str]:
        out, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(error), \
             mock.patch("logging.basicConfig"):
            code = cli.main(["-c", str(self.config_path), *argv])
        return code, out.getvalue(), error.getvalue()

    def _database(self) -> Database:
        database = Database(f"{self.directory}/data.db")
        self.addCleanup(database.close)
        return database

    def test_bad_config(self):
        self.config_path.write_text("[general\n")
        code, _, error = self._run("status")
        self.assertEqual(code, 2)
        self.assertIn("indefinite-articles:", error)

    def test_version(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--version"])

    def test_profile_commands(self):
        self.assertEqual(self._run("profile"), (0, "Rust\n", ""))
        self.assertEqual(self._run("profile", "--set", "Gardening")[1], "profile replaced\n")
        self.assertEqual(self._run("profile")[1], "Gardening\n")
        history = self._run("profile", "--history")[1]
        self.assertLess(history.index("version 1 (initial (from config))"), history.index("version 2 (set manually)"))

    def test_status(self):
        code, out, _ = self._run("status")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"articles\s+0")
        self.assertRegex(out, r"last_digest\s+never")

    def test_feeds(self):
        database = self._database()
        database.feed_failed("https://feed/b", "HTTPError: 404", 3600)
        database.connection.commit()
        out = self._run("feeds")[1]
        self.assertIn("Feed A", out)
        self.assertIn("never fetched", out)
        self.assertIn("FAILING x1: HTTPError: 404", out)
        self.assertIn("-- 2 feeds shown, 2 configured", out)
        out = self._run("feeds", "--broken")[1]
        self.assertNotIn("Feed A", out)
        self.assertIn("https://feed/b", out)
        self.assertEqual(self._run("feeds", "--categories")[1], "   1  (none)\n   1  Tech\n   2  total\n")

    def test_digest_dry_run(self):
        code, _, error = self._run("digest", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("no candidate articles", error)
        database = self._database()
        database.add_article(Entry("https://e/1", "Something", "", None), "S")
        database.connection.commit()
        with mock.patch.object(mailer, "send", side_effect=AssertionError("sent")):
            code, out, _ = self._run("digest", "-n")
        self.assertEqual(code, 0)
        self.assertIn("1. Something", out)

    def test_feedback_from_file(self):
        database = self._database()
        database.add_article(Entry("https://e/1", "Something", "", None), "S")
        database.record_digest("abcd1234", "<d@x>", "2026-11-03", [ranker.DigestPick(Article(1, "https://e/1", "Something"), False)])
        message = EmailMessage()
        message["From"] = "you@example.com"
        message["Subject"] = "Re: [rec:abcd1234]"
        message.set_content("+1")
        path = self.directory / "reply.eml"
        path.write_bytes(bytes(message))
        with mock.patch.object(ranker, "update_profile", return_value=False):
            code, out, _ = self._run("feedback", str(path))
            self.assertEqual((code, out.splitlines()[0]), (0, "ok"))
            self.assertIn("liked #1: Something", out)
            self.assertEqual(self._run("feedback", str(path))[:2], (1, "already-processed\n"))

    def test_run_refuses_incomplete_config(self):
        self.config_path.write_text(f'[general]\ndatabase = "{self.directory}/data.db"\n')
        with mock.patch.object(Service, "run", side_effect=AssertionError("started")), \
             self.assertLogs(level="ERROR"):
            self.assertEqual(self._run("run")[0], 1)

    def test_run_refuses_cpu_only(self):
        self.config_path.write_text(self.config_path.read_text() + '[email.smtp]\nhost = "smtp.example.org"\n')
        # Never let a test ask a real local Ollama to download a model.
        patcher = mock.patch.object(LLM, "pull_missing_models")
        patcher.start()
        self.addCleanup(patcher.stop)
        with mock.patch.object(Service, "run", side_effect=AssertionError("started")), \
             mock.patch.object(LLM, "ping", return_value="OK"), \
             mock.patch.object(LLM, "on_cpu_only", return_value=True), \
             self.assertLogs(level="WARNING"):
            self.assertEqual(self._run("run")[0], 78)
        with mock.patch.object(Service, "run", side_effect=AssertionError("started")), \
             mock.patch.object(LLM, "ping", side_effect=LLMError("down")), \
             self.assertLogs(level="WARNING"):
            self.assertEqual(self._run("run")[0], 1)
        # Like the real run(), the stand-in closes the service's database when it returns.
        with mock.patch.object(Service, "run", autospec=True,
                               side_effect=lambda service: service.database.close()) as run, \
             mock.patch.object(LLM, "ping", return_value="OK"), \
             mock.patch.object(LLM, "on_cpu_only", return_value=False), \
             self.assertLogs(level="WARNING"):
            self.assertEqual(self._run("run")[0], 0)
        run.assert_called_once()

    def test_run_pulls_missing_models(self):
        self.config_path.write_text(self.config_path.read_text() + '[email.smtp]\nhost = "smtp.example.org"\n')
        with mock.patch.object(Service, "run", side_effect=AssertionError("started")), \
             mock.patch.object(LLM, "pull_missing_models", side_effect=LLMError("no space left")) as pull, \
             mock.patch.object(LLM, "ping", side_effect=AssertionError("pinged")), \
             self.assertLogs(level="ERROR") as logs:
            self.assertEqual(self._run("run")[0], 1)
        pull.assert_called_once()
        self.assertIn("cannot start: no space left", "\n".join(logs.output))
        self.config_path.write_text(self.config_path.read_text() + '[llm]\npull_models = false\n')
        with mock.patch.object(Service, "run", side_effect=AssertionError("started")), \
             mock.patch.object(LLM, "pull_missing_models", side_effect=AssertionError("pulled")), \
             mock.patch.object(LLM, "ping", side_effect=LLMError("down")), \
             self.assertLogs(level="WARNING"):
            self.assertEqual(self._run("run")[0], 1)

    def test_check(self):
        with mock.patch.object(LLM, "ping", return_value="OK"), \
             mock.patch.object(LLM, "embed", return_value=[[0.0] * 768]), \
             mock.patch.object(feeds, "fetch", return_value=b"<rss><channel><item><title>T</title>"
                                                             b"<link>https://e/1</link></item></channel></rss>"):
            code, out, _ = self._run("check", "--feeds", "1")
        self.assertEqual(code, 0)
        self.assertIn("[warn] config: email.smtp.host is not set", out)
        self.assertIn("embeddings qwen3-embedding:0.6b: 768 dimensions", out)
        self.assertIn("[ok]   feed Feed A: 1 entries", out)
        with mock.patch.object(LLM, "ping", side_effect=LLMError("down")), \
             mock.patch.object(LLM, "embed", side_effect=LLMError("down")), \
             mock.patch.object(mailer, "send", side_effect=OSError("refused")):
            code, out, _ = self._run("check", "--feeds", "0", "--send-test")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] LLM: down", out)
        self.assertIn("[FAIL] SMTP: refused", out)


if __name__ == "__main__":
    unittest.main()
