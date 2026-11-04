import signal
import threading
import time
import unittest
from datetime import date
from email.message import EmailMessage
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from indefinite_articles import daemon, extract, feeds, inbox, mailer, ranker, replies
from indefinite_articles.article import Article
from indefinite_articles.config import Config, TimeOfDay, load as load_config
from indefinite_articles.database import Database
from indefinite_articles.llm import LLM, LLMError

from fake_llm import FakeLLM

ROOT = Path(__file__).resolve().parent.parent

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Example RSS</title>
<item><title>Rust &amp; the borrow checker</title><link>https://ex.com/rust</link>
<description>&lt;p&gt;A &lt;b&gt;deep&lt;/b&gt; dive.&lt;/p&gt;</description>
<pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>Celebrity gossip</title><guid>https://ex.com/gossip</guid></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Example Atom</title>
<entry><title>Linux 7.3 released</title>
<link rel="alternate" href="https://ex.org/linux"/><link rel="replies" href="https://ex.org/c"/>
<updated>2026-11-02T08:00:00Z</updated><summary>Kernel news.</summary></entry>
</feed>"""


def send_digest(config: Config, database: Database, **kwargs: Any) -> EmailMessage:
    message = daemon.send_digest(config, database, **kwargs)
    assert message is not None, "no digest built"
    return message


def current_text(database: Database) -> str:
    row = database.current_profile()
    assert row is not None
    return row["text"]


def row_count(database: Database, table: str) -> int:
    return database.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class FeedTests(unittest.TestCase):
    def test_rss(self):
        parsed = feeds.parse_feed(RSS)
        title, entries = parsed.title, parsed.entries
        self.assertEqual(title, "Example RSS")
        self.assertEqual([entry.url for entry in entries], ["https://ex.com/rust", "https://ex.com/gossip"])
        self.assertEqual(entries[0].title, "Rust & the borrow checker")
        self.assertEqual(entries[0].summary, "A deep dive.")
        self.assertIsNotNone(entries[0].published)

    def test_atom(self):
        parsed = feeds.parse_feed(ATOM)
        title, entries = parsed.title, parsed.entries
        self.assertEqual(title, "Example Atom")
        self.assertEqual(entries[0].url, "https://ex.org/linux")
        self.assertEqual(entries[0].summary, "Kernel news.")


class ReplyParsingTests(unittest.TestCase):
    ITEMS = [Article(number, f"https://e/{number}", f"Article {number}", source="S") for number in range(1, 11)]

    def test_compact(self):
        interpretation = replies.understand(None, "+1 +4 -2, 3-", self.ITEMS[:5])
        self.assertEqual(interpretation, replies.Interpretation(
            {1: replies.Rating(1), 4: replies.Rating(1), 2: replies.Rating(-1), 3: replies.Rating(-1)}, []))

    def test_lines(self):
        text = "3 like - more on compilers please\n5: no, not crypto\n2 hmm\nGreat list today"
        interpretation = replies.understand(None, text, self.ITEMS[:5])
        ratings, general = interpretation.ratings, interpretation.general
        self.assertEqual(ratings[3], replies.Rating(1, "more on compilers please"))
        self.assertEqual(ratings[5], replies.Rating(-1, "not crypto"))
        self.assertEqual(ratings[2], replies.Rating(0, "hmm"))  # unresolved: left for the LLM
        self.assertEqual(general, ["Great list today"])

    def test_out_of_range_is_general(self):
        interpretation = replies.understand(None, "2026 was a good year", self.ITEMS)
        self.assertEqual(interpretation, replies.Interpretation({}, ["2026 was a good year"]))

    def test_strip_quoted(self):
        text = ("+1 -2\nthanks\n\nOn Tue, 3 Nov 2026 at 07:00, Indefinite Articles <r@x.com>\nwrote:\n"
                "> 1. Something\n> digest-id: rec:abcd1234\n")
        self.assertEqual(replies.strip_quoted(text), "+1 -2\nthanks")
        self.assertEqual(replies.strip_quoted("ok\n-- \nMartin"), "ok")


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.config = load_config(ROOT / "dist" / "indefinite-articles.toml")
        self.config.database = ":memory:"
        self.config.llm.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.config.articles_per_digest = 3
        self.config.exploration = 1
        self.config.fetch_full_articles = False
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)
        self.llm = LLM(self.config.llm)

    def _load_articles(self, score: bool = True):
        for data in (RSS, ATOM):
            parsed = feeds.parse_feed(data)
            title, entries = parsed.title, parsed.entries
            for entry in entries:
                self.database.add_article(entry, title)
        self.database.connection.commit()
        if score:
            self._score(self.database)

    def _score(self, database: Database):
        """What the hourly run does after fetching: pre-filter and LLM-score the shortlist."""
        daemon.prefilter(self.config, database, self.llm)
        ranker.score_candidates(self.config, database, self.llm)

    def test_example_config_loads(self):
        self.assertEqual(self.config.digest_hour_minute, TimeOfDay(7, 0))
        self.assertGreater(len(self.config.feeds), 400)  # dist/feeds.d/default.opml
        self.assertEqual(len({feed.url for feed in self.config.feeds}), len(self.config.feeds))
        self.assertTrue(all(feed.category for feed in self.config.feeds))
        self.assertEqual(self.config.email.allowed_senders, ["you@example.com"])

    def test_digest_and_feedback_cycle(self):
        self._load_articles()
        sent = []
        with mock.patch.object(mailer, "send", lambda config, sent_message: sent.append(sent_message)):
            message = send_digest(self.config, self.database)
        self.assertEqual(len(sent), 1)
        body = mailer.part_text(message)
        token = message["X-Indefinite-Articles-Digest"]
        self.assertIn(f"[rec:{token}]", message["Subject"])
        # Highly scored articles come first.
        items = self.database.digest_items(1)
        self.assertEqual(len(items), 3)
        self.assertIn(items[0].title, ("Rust & the borrow checker", "Linux 7.3 released"))
        self.assertIn("1. ", body)

        # A digest that we sent ourselves must never be read back as feedback.
        message.replace_header("From", "you@example.com")
        self.assertEqual(
            inbox.handle_message(
                self.config,
                self.database,
                self.llm,
                message,
                message["Message-ID"],
            ).status,
            "ignored:own",
        )

        reply = EmailMessage()
        reply["From"] = "Me <you@example.com>"
        reply["Subject"] = "Re: " + message["Subject"]
        reply["Message-ID"] = "<reply1@example.com>"
        reply["In-Reply-To"] = message["Message-ID"]
        reply.set_content("+1\n2 the gossip one was dull\n\nOn Tue someone wrote:\n> " + body)
        result = inbox.handle_message(self.config, self.database, self.llm, reply, reply["Message-ID"])
        self.assertEqual(result.status, "ok")
        feedback = {row["title"]: row["rating"] for row in self.database.connection.execute(
            "SELECT a.title, f.rating FROM feedback f JOIN articles a ON a.id=f.article_id")}
        self.assertEqual(feedback[items[0].title], 1)
        self.assertEqual(feedback[items[1].title], -1)   # resolved by the LLM
        self.assertTrue(any("less gossip" in line for line in result.summary))

        # Unknown sender is ignored.
        reply.replace_header("From", "stranger@evil.example")
        self.assertEqual(
            inbox.handle_message(
                self.config,
                self.database,
                self.llm,
                reply,
                "<reply2@example.com>",
            ).status,
            "ignored:sender",
        )

        self.assertTrue(ranker.update_profile(self.config, self.database, self.llm))
        self.assertIn("Rust", current_text(self.database))
        self.assertFalse(ranker.update_profile(self.config, self.database, self.llm))  # nothing pending

    def _digest_and_reply(self, database_path: str):
        """Send a digest from a file-backed DB and build a reply to it."""
        database = Database(database_path)
        for data in (RSS, ATOM):
            parsed = feeds.parse_feed(data)
            title, entries = parsed.title, parsed.entries
            for entry in entries:
                database.add_article(entry, title)
        database.connection.commit()
        self._score(database)
        with mock.patch.object(mailer, "send", lambda config, sent_message: None):
            message = send_digest(self.config, database)
        reply = EmailMessage()
        reply["From"] = "you@example.com"
        reply["Subject"] = "Re: " + message["Subject"]
        reply["Message-ID"] = "<crash-test@example.com>"
        reply.set_content("+1 -2\nmore systems programming please")
        return database, reply

    def test_crash_mid_reply_records_nothing_then_retry_counts_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/r.db"
            database, reply = self._digest_and_reply(path)
            reply_id = reply["Message-ID"]
            # Crash after the ratings are staged but before the comment and marker are saved.
            with mock.patch.object(Database, "add_general_comment", side_effect=OSError("power cut")):
                with self.assertRaises(OSError):
                    inbox.handle_message(self.config, database, self.llm, reply, reply_id)
            database.close()

            database = Database(path)  # "reboot"
            self.assertEqual(row_count(database, "feedback"), 0)
            self.assertEqual(row_count(database, "processed_messages"), 0)
            # The reply is still unprocessed, so the next poll handles it, exactly once.
            self.assertEqual(inbox.handle_message(self.config, database, self.llm, reply, reply_id).status, "ok")
            self.assertEqual(inbox.handle_message(self.config, database, self.llm, reply, reply_id).status,
                             "already-processed")
            database.close()

            database = Database(path)  # "reboot" again: everything persisted
            self.assertEqual(row_count(database, "feedback"), 2)
            self.assertEqual(row_count(database, "general_comments"), 1)
            self.assertEqual(row_count(database, "processed_messages"), 1)
            database.close()

    def test_failed_message_does_not_leak_into_next_commit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            database, reply = self._digest_and_reply(f"{directory}/r.db")
            with mock.patch.object(Database, "add_general_comment", side_effect=ValueError):
                with self.assertRaises(ValueError):
                    inbox.handle_message(self.config, database, self.llm, reply, "<a@x>")
            # An unrelated later commit must not persist the failed message's ratings.
            database.mark_message("<other@x>", "ignored:sender")
            self.assertEqual(database.connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 0)
            database.close()

    def test_debug_info_appended(self):
        self._load_articles()
        message = send_digest(self.config, self.database, dry_run=True)
        body = mailer.part_text(message)
        self.assertIn("-- debug --", body)
        self.assertIn("selection:", body)
        self.assertLess(body.index("digest-id:"), body.index("-- debug --"))
        self.assertIn("debug</summary>", mailer.part_text(message, "html"))
        # Replies quoting the debug table must not be read as ratings.
        from indefinite_articles import replies
        self.assertNotIn("selection:", replies.strip_quoted("+1\n\n" + body))

        self.config.email.debug_info = False
        message = send_digest(self.config, self.database, dry_run=True)
        self.assertNotIn("-- debug --", mailer.part_text(message))

    def test_on_cpu_only(self):
        self.assertFalse(self.llm.on_cpu_only())
        FakeLLM.size_vram = 0
        try:
            self.assertTrue(self.llm.on_cpu_only())
        finally:
            FakeLLM.size_vram = 1

    def test_unscored_falls_back(self):
        self._load_articles(score=False)
        message = send_digest(self.config, self.database, dry_run=True)
        self.assertIn("simple heuristics", mailer.part_text(message))

    def test_digest_never_calls_llm(self):
        self._load_articles()
        with mock.patch.object(FakeLLM, "do_GET", autospec=True, side_effect=FakeLLM.do_GET) as get, \
             mock.patch.object(FakeLLM, "do_POST", autospec=True, side_effect=FakeLLM.do_POST) as post:
            message = send_digest(self.config, self.database, dry_run=True)
        get.assert_not_called()
        post.assert_not_called()
        self.assertNotIn("simple heuristics", mailer.part_text(message))

    def _fake_speed(self, delay: float = 0.0, gap: float = 0.0, hang_after: int | None = None):
        FakeLLM.delay, FakeLLM.gap, FakeLLM.hang_after = delay, gap, hang_after
        self.addCleanup(setattr, FakeLLM, "delay", 0.0)
        self.addCleanup(setattr, FakeLLM, "gap", 0.0)
        self.addCleanup(setattr, FakeLLM, "hang_after", None)

    def test_streamed_chat(self):
        self.assertEqual(self.llm.chat("s", "Are you there?"), "OK")
        self.assertTrue(FakeLLM.requests[-1]["stream"])
        text = self.llm.chat("maintain a concise profile", "x")
        self.assertEqual(text, "- Likes Rust and systems programming\n- Dislikes gossip")

    def test_no_stall_limit_until_something_succeeded(self):
        self._fake_speed(delay=0.5)
        with mock.patch("indefinite_articles.llm.MIN_STALL", 0.1):
            self.assertEqual(self.llm.ping(), "OK")  # no history: waits as long as it takes

    def test_slow_but_progressing_is_not_cut_off(self):
        with mock.patch("indefinite_articles.llm.MIN_STALL", 0.3):
            self.llm.chat("maintain a concise profile", "x")  # learn: fast
            self._fake_speed(gap=0.1)  # 15 tokens: 1.5s in all, but never 0.3s silent
            self.assertIn("Dislikes gossip", self.llm.chat("maintain a concise profile", "x"))

    def test_stalled_stream_is_cut_off_and_limit_doubles(self):
        from indefinite_articles.llm import LLMUnavailable
        with mock.patch("indefinite_articles.llm.MIN_STALL", 0.3):
            self.llm.chat("maintain a concise profile", "x")  # learn: fast
            self._fake_speed(hang_after=3)
            started = time.monotonic()
            with self.assertRaisesRegex(LLMUnavailable, "stopped responding"), \
                 self.assertLogs("indefinite_articles.llm", "WARNING") as logs:
                self.llm.chat("maintain a concise profile", "x")
            self.assertLess(time.monotonic() - started, 2)
            self.assertIn("next time waiting up to 0.6s", logs.output[0])  # the 0.3s limit doubled
            started = time.monotonic()
            with self.assertRaises(LLMUnavailable), \
                 self.assertLogs("indefinite_articles.llm", "WARNING") as logs:
                self.llm.chat("maintain a concise profile", "x")
            self.assertGreaterEqual(time.monotonic() - started, 0.6)  # it did wait the doubled limit
            self.assertIn("next time waiting up to 1.2s", logs.output[0])

    def test_stalled_scoring_stops_the_round(self):
        self._load_articles(score=False)
        with mock.patch("indefinite_articles.llm.MIN_STALL", 0.3):
            self.llm.ping()  # learn: fast
            self._fake_speed(delay=3)  # never sends the first token in time
            before = len(FakeLLM.requests)
            self.assertEqual(ranker.score_candidates(self.config, self.database, self.llm), 0)
            self.assertEqual(len(FakeLLM.requests) - before, 1)  # gave up after the first article

    def test_llm_deadline(self):
        llm = LLM(self.config.llm, deadline=lambda: 2e9)
        self.assertEqual(llm.ping().strip(), self.llm.ping().strip())
        llm.deadline = lambda: 0
        with self.assertRaisesRegex(LLMError, "next digest"):
            llm.ping()
        self.assertGreater(self.config.next_digest_time(), time.time())
        self.assertLessEqual(self.config.next_digest_time(), time.time() + 86400)

    def _stuck_scoring(self, config: Config, database: Database, llm: LLM,
                       stop: threading.Event | None = None):
        self.scoring.set()
        self.release.wait(10)  # like an LLM request that takes ages
        return 0

    def _run_service(self):  # the service must be built on the thread that runs it
        service = daemon.Service(self.config)
        service.TICK = 0.05
        service.run()

    def test_slow_scoring_blocks_nothing_else(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            self.config.database = f"{directory}/s.db"
            self.config.email.imap.host = ""
            self.config.email.smtp.host = "smtp.invalid"
            self.config.digest_time = "00:00"  # due as soon as today's isn't marked as sent
            database = Database(self.config.database)
            for data in (RSS, ATOM):
                parsed = feeds.parse_feed(data)
                title, entries = parsed.title, parsed.entries
                for entry in entries:
                    database.add_article(entry, title)
            database.set_state("last_digest_date", date.today().isoformat())  # held back for now
            database.connection.commit()
            database.close()

            self.release, self.scoring, fetched, sent = (threading.Event() for _ in range(4))
            sent_messages = []

            with mock.patch.object(ranker, "score_candidates", self._stuck_scoring), \
                 mock.patch.object(feeds, "update_all", lambda *args, **kwargs: fetched.set() or 0), \
                 mock.patch.object(daemon, "prefilter", lambda *args: None), \
                 mock.patch.object(extract, "enrich_candidates", lambda *args, **kwargs: 0), \
                 mock.patch.object(mailer, "send",
                                   lambda config, sent_message: (sent_messages.append(sent_message), sent.set())), \
                 mock.patch.object(signal, "signal") as register_handler:
                thread = threading.Thread(target=self._run_service)
                thread.start()
                try:
                    self.assertTrue(self.scoring.wait(5))
                    # Scoring is stuck, yet feeds are fetched and the digest goes out.
                    self.assertTrue(fetched.wait(5))
                    database = Database(self.config.database)
                    database.set_state("last_digest_date", "")  # now let the digest become due
                    database.connection.commit()
                    database.close()
                    self.assertTrue(sent.wait(5))
                    body = mailer.part_text(sent_messages[0])
                    self.assertRegex(body, r"backlog: shortlist 3, 3 never scored")
                    self.assertRegex(body, r"scoring: running for \d+s")
                    self.assertIn("llm stall limits:", body)
                finally:
                    for call in register_handler.call_args_list[:1]:  # as if sent SIGTERM
                        call.args[1](signal.SIGTERM, None)
                    self.release.set()
                    thread.join(10)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
