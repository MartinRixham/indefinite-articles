import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from indefinite_articles import extract, feeds, ranker
from indefinite_articles.config import load as load_config
from indefinite_articles.database import Database

ROOT = Path(__file__).resolve().parent.parent


class StubLLM:
    """Answers every chat_json request with `reply` and records the prompts."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.prompts: list[str] = []

    def chat_json(self, system, user):
        self.prompts.append(user)
        return self.reply

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
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.end_headers()
            return
        body, content_type = (b"%PDF-1.4", "application/pdf") if self.path.endswith(".pdf") else (PAGE.encode(), "text/html; charset=utf-8")
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
        text = extract.extract_text(PAGE)
        self.assertIn("borrow checker enforces", text)
        self.assertIn("lifetimes with & examples", text)
        for junk in ("Popular", "Share this", "Great post", "Copyright", "junk", "About us"):
            self.assertNotIn(junk, text)

    def test_meta_description_fallback(self):
        html = '<html><head><meta property="og:description" content="Only meta here."></head><body><p>x</p></body></html>'
        self.assertEqual(extract.extract_text(html), "Only meta here.")

    def test_truncation(self):
        self.assertLessEqual(len(extract.extract_text(PAGE, max_characters=80)), 80)

    def test_private_addresses_blocked_by_default(self):
        with self.assertRaises(extract.BlockedAddress):
            extract.fetch_page(self.base + "/a", 1000)
        with self.assertRaises(extract.BlockedAddress):
            extract.fetch_page("file:///etc/passwd", 1000, allow_private=True)

    def test_redirect_to_private_blocked(self):
        # Even when the first host is allowed, redirects are re-checked.
        original = extract._check_host
        extract._check_host = lambda host, allow: original(host, allow and host == "127.0.0.1")
        try:
            with self.assertRaises(extract.BlockedAddress):
                extract.fetch_page(self.base + "/redirect", 1000, allow_private=True)
        finally:
            extract._check_host = original

    def test_non_html_rejected(self):
        with self.assertRaises(ValueError):
            extract.fetch_page(self.base + "/paper.pdf", 1000, allow_private=True)

    def test_enrich_candidates_and_prompt_text(self):
        config = load_config(ROOT / "dist" / "indefinite-articles.toml")
        config.allow_private_addresses = True
        database = Database(":memory:")
        self.addCleanup(database.close)
        database.add_article(self.base + "/a", "Borrow checker", "borrowchecker", "Feed blurb.", "S", None)
        database.add_article(self.base + "/b.pdf", "A paper", "apaper", "Abstract text.", "S", None)
        database.connection.commit()
        self.assertEqual(extract.enrich_candidates(config, database), 1)
        rows = {row["title"]: row for row in database.candidates(3, 10)}
        self.assertEqual(rows["Borrow checker"]["content_status"], "ok")
        self.assertIn("borrow checker enforces", ranker.article_text(rows["Borrow checker"], 1500))
        self.assertTrue(rows["A paper"]["content_status"].startswith("failed"))
        self.assertEqual(ranker.article_text(rows["A paper"], 1500), "Abstract text.")
        self.assertEqual(extract.enrich_candidates(config, database), 0)  # not retried

    def test_score_one_shares_prefix_and_reads_score(self):
        llm = StubLLM({"reason": "fits", "score": 12})
        prompts = llm.prompts
        prefix = ranker.score_prefix("Likes Rust.", "(examples)")
        rows = [{"title": f"Article {index}", "content": "x" * 3000, "summary": "", "source": "S",
                 "category": ""} for index in range(3)]
        results = [ranker.score_one(llm, prefix, article, 500) for article in rows]
        self.assertEqual(results[0], (10.0, "fits"))  # clamped to 0-10
        self.assertTrue(all(prompt.startswith(prefix) for prompt in prompts))
        self.assertTrue(all(len(prompt) < len(prefix) + 600 for prompt in prompts))
        self.assertIn("Title: Article 2", prompts[2])

    def test_score_one_rejects_missing_score(self):
        row = {"title": "t", "content": "", "summary": "s", "source": "S", "category": ""}
        with self.assertRaises(ranker.LLMError):
            ranker.score_one(StubLLM({"reason": "no number"}), "p", row)

    def test_migration_from_old_schema(self):
        import sqlite3, tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "old.db")
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, title TEXT NOT NULL, "
                      "title_key TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', "
                      "published REAL, fetched_at REAL NOT NULL, score REAL, reason TEXT, "
                      "score_profile_id INTEGER, sent_digest_id INTEGER)")
            connection.commit(); connection.close()
            with Database(path) as database:
                columns = {row["name"] for row in database.connection.execute("PRAGMA table_info(articles)")}
            self.assertTrue({"content", "content_status"} <= columns)


if __name__ == "__main__":
    unittest.main()
