import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from indefinite_articles.config import load as load_config
from indefinite_articles.fetching import extract
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.storage.database import Database

ROOT = Path(__file__).resolve().parents[2]


META_ONLY = '<html><head><meta property="og:description" content="Only meta here."></head><body><p>x</p></body></html>'


PAGE = """<!doctype html><html><head><title>T</title>
<meta name="description" content="Meta description of the page.">
<style>.x{color:red}</style><script>var junk = 1;</script></head><body>
<header><nav><a href="/">Home</a> <a href="/about">About us and our very long menu item list here</a></nav></header>
<div class="sidebar"><p>Popular: ten things you will not believe about something or other today.</p></div>
<main><article class="post-content share-enabled">
<h1>The borrow checker, explained</h1>
<p>Rust's borrow checker enforces that references never outlive the data they point to, and that
mutable access is exclusive.</p>
<div class="share-buttons"><p>Share this on social media with all your friends right now please!</p></div>
<p>This article walks through lifetimes with &amp; examples, including non-lexical lifetimes.</p>
<p>Short</p>
</article>
<section class="comments"><p>Great post! I have been looking for exactly this for years, thank you.</p></section>
</main><footer><p>Copyright 2026 Example Corp. All rights reserved. Terms and privacy.</p></footer>
</body></html>"""


class Page(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.end_headers()
            return
        if self.path.endswith(".pdf"):
            body, content_type = b"%PDF-1.4", "application/pdf"
        else:
            body, content_type = (META_ONLY if self.path == "/meta" else PAGE).encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Page)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_extract_main_content(self):
        text = extract.fetch_page(self.base + "/a", 2000, allow_private=True)
        self.assertIn("borrow checker enforces", text)
        self.assertIn("lifetimes with & examples", text)
        for junk in ("Popular", "Share this", "Great post", "Copyright", "junk", "About us"):
            self.assertNotIn(junk, text)

    def test_meta_description_fallback(self):
        self.assertEqual(extract.fetch_page(self.base + "/meta", 2000, allow_private=True), "Only meta here.")

    def test_truncation(self):
        self.assertLessEqual(len(extract.fetch_page(self.base + "/a", 80, allow_private=True)), 80)

    def test_private_addresses_blocked_by_default(self):
        with self.assertRaises(extract.BlockedAddress):
            extract.fetch_page(self.base + "/a", 1000)
        with self.assertRaises(extract.BlockedAddress):
            extract.fetch_page("file:///etc/passwd", 1000, allow_private=True)

    def test_redirect_to_private_blocked(self):
        # Even when the first host is allowed, redirects are re-checked. The address check
        # (port None) is told the test server is public; connecting still uses the real lookup.
        real_getaddrinfo = socket.getaddrinfo

        def getaddrinfo(host: str, port: object, *args: object, **kwargs: object) -> list:
            if host == "127.0.0.1" and port is None:
                return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))]
            return real_getaddrinfo(host, port, *args, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(socket, "getaddrinfo", getaddrinfo):
            self.assertIn("Rust", extract.fetch_page(self.base + "/a", 1000))  # the server counts as public
            with self.assertRaises(extract.BlockedAddress):
                extract.fetch_page(self.base + "/redirect", 1000)

    def test_non_html_rejected(self):
        with self.assertRaises(ValueError):
            extract.fetch_page(self.base + "/paper.pdf", 1000, allow_private=True)

    def test_enrich_candidates(self):
        config = load_config(ROOT / "dist" / "indefinite-articles.toml")
        config.allow_private_addresses = True
        database = Database(":memory:")
        self.addCleanup(database.close)
        database.add_article(Entry(self.base + "/a", "Borrow checker", "Feed blurb.", None), "S")
        database.add_article(Entry(self.base + "/b.pdf", "A paper", "Abstract text.", None), "S")
        database.connection.commit()
        self.assertEqual(extract.enrich_candidates(config, database), 1)
        rows = {row.title: row for row in database.candidates(3, 10)}
        self.assertEqual(rows["Borrow checker"].content_status, "ok")
        self.assertIn("borrow checker enforces", rows["Borrow checker"].content or "")
        self.assertTrue((rows["A paper"].content_status or "").startswith("failed"))
        self.assertEqual(rows["A paper"].content, "")
        self.assertEqual(extract.enrich_candidates(config, database), 0)  # not retried

    def test_audio_only_episode_not_downloaded(self) -> None:
        config = load_config(ROOT / "dist" / "indefinite-articles.toml")
        config.allow_private_addresses = True
        database = Database(":memory:")
        self.addCleanup(database.close)
        audio = self.base + "/a"  # would extract fine as a page, so a fetch would show
        database.add_article(Entry(audio, "Episode", "Show notes.", None, audio_url=audio), "Pod")
        database.connection.commit()
        self.assertEqual(extract.enrich_candidates(config, database), 0)
        episode = database.candidates(3, 10, podcasts=True)[0]
        self.assertEqual((episode.content_status, episode.content), ("empty", ""))

    def test_migration_from_old_schema(self):
        import sqlite3, tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "old.db")
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE articles (id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, title TEXT NOT NULL, "
                "title_key TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', "
                "published REAL, fetched_at REAL NOT NULL, score REAL, reason TEXT, "
                "score_profile_id INTEGER, sent_digest_id INTEGER)"
            )
            connection.execute("CREATE TABLE feeds (url TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '')")
            connection.execute("CREATE TABLE digests (id INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, "
                               "message_id TEXT NOT NULL, sent_at REAL NOT NULL, local_date TEXT NOT NULL)")
            connection.execute("CREATE TABLE digest_items (digest_id INTEGER NOT NULL, position INTEGER NOT NULL, "
                               "article_id INTEGER NOT NULL, exploratory INTEGER NOT NULL DEFAULT 0, "
                               "PRIMARY KEY (digest_id, position))")
            connection.execute("CREATE TABLE processed_messages (message_id TEXT PRIMARY KEY, "
                               "processed_at REAL NOT NULL, status TEXT NOT NULL)")
            connection.commit(); connection.close()
            with Database(path) as database:
                columns = {row["name"] for row in database.connection.execute("PRAGMA table_info(articles)")}
                feed_columns = {row["name"] for row in database.connection.execute("PRAGMA table_info(feeds)")}
                # The dropped NOT NULL columns would make these inserts fail.
                database.record_digest("abcd1234", "<digest@x>", [])
                database.mark_message("<reply@x>")
                remaining = {table: {row["name"] for row in database.connection.execute(f"PRAGMA table_info({table})")}
                             for table in ("digests", "digest_items", "processed_messages")}
            self.assertTrue({"content", "content_status", "audio_url", "duration", "feed_url", "apple_url"} <= columns)
            self.assertTrue({"apple_id", "apple_checked_at"} <= feed_columns)
            self.assertEqual(remaining, {"digests": {"id", "token", "message_id"},
                                         "digest_items": {"digest_id", "position", "article_id"},
                                         "processed_messages": {"message_id", "processed_at"}})


if __name__ == "__main__":
    unittest.main()
