import json
import re
import threading
import unittest
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from indefinite_articles import config, daemon, feeds, inbox, mailer, ranker, replies
from indefinite_articles.db import DB
from indefinite_articles.llm import LLM, extract_json

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


def fake_embedding(text, dim=64):
    """Bag-of-words hashed vector: texts sharing words are similar, like real embeddings."""
    import zlib
    v = [0.0] * dim
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[zlib.crc32(w.encode()) % dim] += 1.0
    return v


class FakeLLM(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint: scores titles containing 'Rust'/'Linux' highly."""
    requests = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.requests.append(body)
        if self.path.endswith("/embeddings"):
            return self._json({"data": [{"index": i, "embedding": fake_embedding(t)}
                                        for i, t in enumerate(body["input"])]})
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if "Rate each candidate" in user:
            ratings = []
            for idx, title in re.findall(r"^\[(\d+)\] (.+)$", user, re.M):
                good = any(w in title for w in ("Rust", "Linux"))
                ratings.append({"id": int(idx), "score": 9 if good else 2, "reason": "test"})
            content = "```json\n" + json.dumps({"ratings": ratings}) + "\n```"
        elif "interpret a reader" in system:
            content = json.dumps({"ratings": [{"item": 2, "rating": "dislike", "comment": "boring"}],
                                  "general_comment": "less gossip"})
        elif "maintain a concise profile" in system:
            content = "- Likes Rust and systems programming\n- Dislikes gossip"
        else:
            content = "OK"
        self._json({"choices": [{"message": {"role": "assistant", "content": content}}]})

    def _json(self, obj):
        out = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class FeedTests(unittest.TestCase):
    def test_rss(self):
        title, entries = feeds.parse_feed(RSS)
        self.assertEqual(title, "Example RSS")
        self.assertEqual([e.url for e in entries], ["https://ex.com/rust", "https://ex.com/gossip"])
        self.assertEqual(entries[0].title, "Rust & the borrow checker")
        self.assertEqual(entries[0].summary, "A deep dive.")
        self.assertIsNotNone(entries[0].published)

    def test_atom(self):
        title, entries = feeds.parse_feed(ATOM)
        self.assertEqual(title, "Example Atom")
        self.assertEqual(entries[0].url, "https://ex.org/linux")
        self.assertEqual(entries[0].summary, "Kernel news.")


class ReplyParsingTests(unittest.TestCase):
    def test_compact(self):
        r, g, u = replies.parse_structured("+1 +4 -2, 3-", 5)
        self.assertEqual(r, {1: (1, ""), 4: (1, ""), 2: (-1, ""), 3: (-1, "")})
        self.assertEqual((g, u), ([], 0))

    def test_lines(self):
        text = "3 like - more on compilers please\n5: no, not crypto\n2 hmm\nGreat list today"
        r, g, u = replies.parse_structured(text, 5)
        self.assertEqual(r[3], (1, "more on compilers please"))
        self.assertEqual(r[5], (-1, "not crypto"))
        self.assertEqual(r[2], (0, "hmm"))
        self.assertEqual(u, 1)
        self.assertEqual(g, ["Great list today"])

    def test_out_of_range_is_general(self):
        r, g, _ = replies.parse_structured("2026 was a good year", 10)
        self.assertEqual(r, {})

    def test_strip_quoted(self):
        text = ("+1 -2\nthanks\n\nOn Tue, 3 Nov 2026 at 07:00, Indefinite Articles <r@x.com>\nwrote:\n"
                "> 1. Something\n> digest-id: rec:abcd1234\n")
        self.assertEqual(replies.strip_quoted(text), "+1 -2\nthanks")
        self.assertEqual(replies.strip_quoted("ok\n-- \nMartin"), "ok")

    def test_extract_json(self):
        self.assertEqual(extract_json('Sure! {"a": 1} hope that helps'), {"a": 1})


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.cfg = config.load(ROOT / "dist" / "indefinite-articles.toml")
        self.cfg.database = ":memory:"
        self.cfg.llm.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.cfg.articles_per_digest = 3
        self.cfg.exploration = 1
        self.cfg.fetch_full_articles = False
        self.db = DB(":memory:")
        self.llm = LLM(self.cfg.llm)

    def _load_articles(self):
        for data in (RSS, ATOM):
            title, entries = feeds.parse_feed(data)
            for e in entries:
                self.db.add_article(e.url, e.title, feeds.title_key(e.title), e.summary, title, None)
        self.db.conn.commit()

    def test_example_config_loads(self):
        self.assertEqual(self.cfg.digest_hour_minute, (7, 0))
        self.assertGreater(len(self.cfg.feeds), 400)  # dist/feeds.d/default.opml
        self.assertEqual(len({f.url for f in self.cfg.feeds}), len(self.cfg.feeds))
        self.assertTrue(all(f.category for f in self.cfg.feeds))
        self.assertEqual(self.cfg.email.allowed_senders, ["you@example.com"])

    def test_digest_and_feedback_cycle(self):
        self._load_articles()
        sent = []
        with mock.patch.object(mailer, "send", lambda cfg, m: sent.append(m)):
            msg = daemon.send_digest(self.cfg, self.db, self.llm)
        self.assertEqual(len(sent), 1)
        body = msg.get_body(preferencelist=("plain",)).get_content()
        token = msg["X-Indefinite-Articles-Digest"]
        self.assertIn(f"[rec:{token}]", msg["Subject"])
        # Highly scored articles come first.
        items = self.db.digest_items(1)
        self.assertEqual(len(items), 3)
        self.assertIn(items[0]["title"], ("Rust & the borrow checker", "Linux 7.3 released"))
        self.assertIn("1. ", body)

        # A digest that we sent ourselves must never be read back as feedback.
        msg.replace_header("From", "you@example.com")
        self.assertEqual(inbox.process_message(self.cfg, self.db, self.llm, msg)[0], "ignored:own")

        reply = EmailMessage()
        reply["From"] = "Me <you@example.com>"
        reply["Subject"] = "Re: " + msg["Subject"]
        reply["Message-ID"] = "<reply1@example.com>"
        reply["In-Reply-To"] = msg["Message-ID"]
        reply.set_content("+1\n2 the gossip one was dull\n\nOn Tue someone wrote:\n> " + body)
        status, summary = inbox.handle_message(self.cfg, self.db, self.llm, reply, reply["Message-ID"])
        self.assertEqual(status, "ok")
        fb = {r["title"]: r["rating"] for r in self.db.conn.execute(
            "SELECT a.title, f.rating FROM feedback f JOIN articles a ON a.id=f.article_id")}
        self.assertEqual(fb[items[0]["title"]], 1)
        self.assertEqual(fb[items[1]["title"]], -1)   # resolved by the LLM
        self.assertTrue(any("less gossip" in s for s in summary))

        # Unknown sender is ignored.
        reply.replace_header("From", "stranger@evil.example")
        self.assertEqual(inbox.process_message(self.cfg, self.db, self.llm, reply)[0], "ignored:sender")

        self.assertTrue(ranker.update_profile(self.cfg, self.db, self.llm))
        self.assertIn("Rust", self.db.current_profile()["text"])
        self.assertFalse(ranker.update_profile(self.cfg, self.db, self.llm))  # nothing pending

    def _digest_and_reply(self, db_path):
        """Send a digest from a file-backed DB and build a reply to it."""
        db = DB(db_path)
        for data in (RSS, ATOM):
            title, entries = feeds.parse_feed(data)
            for e in entries:
                db.add_article(e.url, e.title, feeds.title_key(e.title), e.summary, title, None)
        db.conn.commit()
        with mock.patch.object(mailer, "send", lambda cfg, m: None):
            msg = daemon.send_digest(self.cfg, db, self.llm)
        reply = EmailMessage()
        reply["From"] = "you@example.com"
        reply["Subject"] = "Re: " + msg["Subject"]
        reply["Message-ID"] = "<crash-test@example.com>"
        reply.set_content("+1 -2\nmore systems programming please")
        return db, reply

    def test_crash_mid_reply_records_nothing_then_retry_counts_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/r.db"
            db, reply = self._digest_and_reply(path)
            mid = reply["Message-ID"]
            # Crash after the ratings are staged but before the comment and marker are saved.
            with mock.patch.object(DB, "add_general_comment", side_effect=OSError("power cut")):
                with self.assertRaises(OSError):
                    inbox.handle_message(self.cfg, db, self.llm, reply, mid)
            db.close()

            db = DB(path)  # "reboot"
            count = lambda sql: db.conn.execute(sql).fetchone()[0]
            self.assertEqual(count("SELECT COUNT(*) FROM feedback"), 0)
            self.assertEqual(count("SELECT COUNT(*) FROM processed_messages"), 0)
            # The reply is still unprocessed, so the next poll handles it, exactly once.
            self.assertEqual(inbox.handle_message(self.cfg, db, self.llm, reply, mid)[0], "ok")
            self.assertEqual(inbox.handle_message(self.cfg, db, self.llm, reply, mid)[0],
                             "already-processed")
            db.close()

            db = DB(path)  # "reboot" again: everything persisted
            count = lambda sql: db.conn.execute(sql).fetchone()[0]
            self.assertEqual(count("SELECT COUNT(*) FROM feedback"), 2)
            self.assertEqual(count("SELECT COUNT(*) FROM general_comments"), 1)
            self.assertEqual(count("SELECT COUNT(*) FROM processed_messages"), 1)
            db.close()

    def test_failed_message_does_not_leak_into_next_commit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            db, reply = self._digest_and_reply(f"{d}/r.db")
            with mock.patch.object(DB, "add_general_comment", side_effect=ValueError):
                with self.assertRaises(ValueError):
                    inbox.handle_message(self.cfg, db, self.llm, reply, "<a@x>")
            # An unrelated later commit must not persist the failed message's ratings.
            db.mark_message("<other@x>", "ignored:sender")
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 0)
            db.close()

    def test_debug_info_appended(self):
        self._load_articles()
        msg = daemon.send_digest(self.cfg, self.db, self.llm, dry_run=True)
        body = msg.get_body(preferencelist=("plain",)).get_content()
        self.assertIn("-- debug --", body)
        self.assertIn("selection:", body)
        self.assertLess(body.index("digest-id:"), body.index("-- debug --"))
        self.assertIn("debug</summary>", msg.get_body(preferencelist=("html",)).get_content())
        # Replies quoting the debug table must not be read as ratings.
        from indefinite_articles import replies
        self.assertNotIn("selection:", replies.strip_quoted("+1\n\n" + body))

        self.cfg.email.debug_info = False
        msg = daemon.send_digest(self.cfg, self.db, self.llm, dry_run=True)
        self.assertNotIn("-- debug --", msg.get_body(preferencelist=("plain",)).get_content())

    def test_llm_down_falls_back(self):
        self._load_articles()
        self.cfg.llm.base_url = "http://127.0.0.1:1/v1"
        msg = daemon.send_digest(self.cfg, self.db, LLM(self.cfg.llm), dry_run=True)
        self.assertIn("unavailable", msg.get_body(preferencelist=("plain",)).get_content())


if __name__ == "__main__":
    unittest.main()
