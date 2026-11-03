import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from indefinite_articles import config, extract, feeds, ranker
from indefinite_articles.db import DB

ROOT = Path(__file__).resolve().parent.parent

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
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.end_headers()
            return
        body, ctype = (b"%PDF-1.4", "application/pdf") if self.path.endswith(".pdf") else (PAGE.encode(), "text/html; charset=utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
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
        self.assertLessEqual(len(extract.extract_text(PAGE, max_chars=80)), 80)

    def test_private_addresses_blocked_by_default(self):
        with self.assertRaises(extract.BlockedAddress):
            extract.fetch_page(self.base + "/a", 1000)
        with self.assertRaises(extract.BlockedAddress):
            extract.fetch_page("file:///etc/passwd", 1000, allow_private=True)

    def test_redirect_to_private_blocked(self):
        # Even when the first host is allowed, redirects are re-checked.
        orig = extract._check_host
        extract._check_host = lambda host, allow: orig(host, allow and host == "127.0.0.1")
        try:
            with self.assertRaises(extract.BlockedAddress):
                extract.fetch_page(self.base + "/redirect", 1000, allow_private=True)
        finally:
            extract._check_host = orig

    def test_non_html_rejected(self):
        with self.assertRaises(ValueError):
            extract.fetch_page(self.base + "/paper.pdf", 1000, allow_private=True)

    def test_enrich_candidates_and_prompt_text(self):
        cfg = config.load(ROOT / "dist" / "indefinite-articles.toml")
        cfg.allow_private_addresses = True
        db = DB(":memory:")
        db.add_article(self.base + "/a", "Borrow checker", "borrowchecker", "Feed blurb.", "S", None)
        db.add_article(self.base + "/b.pdf", "A paper", "apaper", "Abstract text.", "S", None)
        db.conn.commit()
        self.assertEqual(extract.enrich_candidates(cfg, db), 1)
        rows = {r["title"]: r for r in db.candidates(3, 10)}
        self.assertEqual(rows["Borrow checker"]["content_status"], "ok")
        self.assertIn("borrow checker enforces", ranker.article_text(rows["Borrow checker"], 1500))
        self.assertTrue(rows["A paper"]["content_status"].startswith("failed"))
        self.assertEqual(ranker.article_text(rows["A paper"], 1500), "Abstract text.")
        self.assertEqual(extract.enrich_candidates(cfg, db), 0)  # not retried

    def test_score_one_shares_prefix_and_reads_score(self):
        prompts = []

        class StubLLM:
            def chat_json(self, system, user):
                prompts.append(user)
                return {"reason": "fits", "score": 12}

        prefix = ranker.score_prefix("Likes Rust.", "(examples)")
        rows = [{"title": f"Article {i}", "content": "x" * 3000, "summary": "", "source": "S",
                 "category": ""} for i in range(3)]
        results = [ranker.score_one(StubLLM(), prefix, a, 500) for a in rows]
        self.assertEqual(results[0], (10.0, "fits"))  # clamped to 0-10
        self.assertTrue(all(p.startswith(prefix) for p in prompts))
        self.assertTrue(all(len(p) < len(prefix) + 600 for p in prompts))
        self.assertIn("Title: Article 2", prompts[2])

    def test_score_one_rejects_missing_score(self):
        class StubLLM:
            def chat_json(self, system, user):
                return {"reason": "no number"}

        row = {"title": "t", "content": "", "summary": "s", "source": "S", "category": ""}
        with self.assertRaises(ranker.LLMError):
            ranker.score_one(StubLLM(), "p", row)

    def test_migration_from_old_schema(self):
        import sqlite3, tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.db")
            c = sqlite3.connect(path)
            c.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, title TEXT NOT NULL, "
                      "title_key TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', "
                      "published REAL, fetched_at REAL NOT NULL, score REAL, reason TEXT, "
                      "score_profile_id INTEGER, sent_digest_id INTEGER)")
            c.commit(); c.close()
            db = DB(path)
            cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(articles)")}
            self.assertTrue({"content", "content_status"} <= cols)


if __name__ == "__main__":
    unittest.main()
