import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from indefinite_articles import config, embed, feeds, ranker
from indefinite_articles.config import Config, Feed
from indefinite_articles.db import DB
from indefinite_articles.llm import LLM, LLMError

ROOT = Path(__file__).resolve().parent.parent

FEED = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
<item><title>Post one</title><link>https://ex.com/1</link></item></channel></rss>"""


class FeedServer(BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FeedServer.hits.append((self.path, self.headers.get("If-None-Match")))
        if self.path == "/broken":
            self.send_response(500)
            self.end_headers()
            return
        if self.headers.get("If-None-Match") == '"v1"':
            self.send_response(304)
            self.end_headers()
            return
        body = FEED.replace(b"Post one", self.path.encode()).replace(b"/1<", self.path.encode() + b"<")
        self.send_response(200)
        self.send_header("ETag", '"v1"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FeedListTests(unittest.TestCase):
    def test_opml(self):
        opml = b"""<?xml version="1.0"?><opml version="2.0"><body>
        <outline text="Science"><outline text="Quanta" xmlUrl="https://q/feed" maxItems="5"/>
          <outline text="Sub"><outline title="Deep" xmlUrl="https://d/feed"/></outline></outline>
        <outline text="Loose" xmlUrl="https://l/feed"/></body></opml>"""
        fs = config.parse_opml(opml)
        self.assertEqual([(f.name, f.category, f.max_items) for f in fs],
                         [("Quanta", "Science", 5), ("Deep", "Sub", 40), ("Loose", "", 40)])

    def test_txt(self):
        fs = config.parse_feed_list("# Tech\nhttps://a/feed  A blog\n\nhttps://b/feed\n# News\nhttps://c/feed")
        self.assertEqual([(f.url, f.name, f.category) for f in fs],
                         [("https://a/feed", "A blog", "Tech"), ("https://b/feed", "", "Tech"),
                          ("https://c/feed", "", "News")])

    def test_exclude_categories_and_dedupe(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(f"{d}/feeds.d")
            Path(f"{d}/feeds.d/a.txt").write_text("# Sports\nhttps://s/feed\n# Tech\nhttps://t/feed\n")
            Path(f"{d}/r.toml").write_text(
                '[general]\nexclude_categories = ["sports"]\n[[feeds]]\nurl = "https://t/feed"\nname = "dup"\n')
            cfg = config.load(f"{d}/r.toml")
            self.assertEqual([(f.url, f.name) for f in cfg.feeds], [("https://t/feed", "dup")])


class LenientParseTests(unittest.TestCase):
    def test_html_entities_and_leading_junk(self):
        data = b"\n\n<?xml version='1.0'?><rss><channel><title>X&nbsp;Y</title><item><title>A &mdash; B</title><link>https://e/1</link></item></channel></rss>"
        title, entries = feeds.parse_feed(data)
        self.assertEqual(title, "X\xa0Y".replace("\xa0", " ").strip() or title)
        self.assertEqual(entries[0].title, "A — B")

    def test_broken_markup_regex_fallback(self):
        data = b"<rss><channel><title>T</title><item><title>Good</title><link>https://e/1</link><description><![CDATA[<p>hi</p>]]></description></item><br></channel></rss>"
        title, entries = feeds.parse_feed(data)
        self.assertEqual((entries[0].title, entries[0].summary), ("Good", "hi"))

    def test_html_page_rejected(self):
        with self.assertRaises(ValueError):
            feeds.parse_feed(b"<!DOCTYPE html><html><body>nope</body></html>")

    def test_gzip(self):
        import gzip
        self.assertEqual(feeds._decompress(gzip.compress(FEED), "gzip"), FEED)


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FeedServer)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_conditional_get_and_backoff(self):
        cfg = Config()
        cfg.feeds = [Feed(url=f"{self.base}/a", category="C"), Feed(url=f"{self.base}/b"),
                     Feed(url=f"{self.base}/broken")]
        db = DB(":memory:")
        FeedServer.hits.clear()
        self.assertEqual(feeds.update_all(cfg, db), 2)
        row = db.conn.execute("SELECT category FROM articles WHERE url LIKE '%/a'").fetchone()
        self.assertEqual(row["category"], "C")
        # Second run: unchanged feeds answer 304 and add nothing; broken feed retried once.
        self.assertEqual(feeds.update_all(cfg, db), 0)
        self.assertIn(("/a", '"v1"'), FeedServer.hits)
        st = db.feed_states()
        self.assertEqual(st[f"{self.base}/broken"]["failures"], 2)
        # ...and then it backs off instead of being fetched every cycle.
        self.assertGreater(st[f"{self.base}/broken"]["next_attempt"], time.time() + 3000)
        FeedServer.hits.clear()
        feeds.update_all(cfg, db)
        self.assertNotIn("/broken", [p for p, _ in FeedServer.hits])


class PrefilterTests(unittest.TestCase):
    def _cfg(self):
        cfg = Config()
        cfg.max_candidates = 20
        cfg.interests = "rust compilers systems programming"
        return cfg

    def _db(self, n_per_topic=30):
        db = DB(":memory:")
        topics = {"Rust Weekly": "rust compilers borrow checker systems programming",
                  "Celeb News": "celebrity gossip red carpet fashion",
                  "Garden": "tomatoes compost garden soil"}
        for src, words in topics.items():
            for i in range(n_per_topic):
                db.add_article(f"https://{src}/{i}", f"{words} {i}", f"{src}{i}", words, src, None)
        db.conn.commit()
        return db

    def _llm(self):
        from test_indefinite_articles import fake_embedding
        llm = mock.Mock(spec=LLM)
        llm.embed.side_effect = lambda texts: [fake_embedding(t) for t in texts]
        return llm

    def test_shortlist_prefers_matches_with_source_cap_and_exploration(self):
        cfg, db, llm = self._cfg(), self._db(), self._llm()
        embed.embed_new(cfg, db, llm)
        embed.update_prefilter(cfg, db, llm, ranker.ensure_profile(cfg, db))
        sl = ranker.shortlist(cfg, db)
        self.assertEqual(len(sl), 20)
        by_src = {}
        for a in sl:
            by_src[a["source"]] = by_src.get(a["source"], 0) + 1
        top = [a["source"] for a in sl[:3]]
        self.assertEqual(top, ["Rust Weekly"] * 3)
        # The cap spreads the first picks across sources; with only 3 sources the list is
        # then topped up from the best remaining matches, i.e. mostly Rust.
        first9 = [a["source"] for a in sl[:9]]
        self.assertEqual(sorted(set(first9)), ["Celeb News", "Garden", "Rust Weekly"])
        self.assertEqual(first9.count("Rust Weekly"), 3)
        self.assertGreater(by_src["Rust Weekly"], 10)
        self.assertEqual(ranker.shortlist(cfg, db)[-2:], sl[-2:])  # random picks stable

    def test_feedback_changes_ranking(self):
        cfg, db, llm = self._cfg(), self._db(), self._llm()
        cfg.interests = "anything"
        embed.embed_new(cfg, db, llm)
        embed.update_prefilter(cfg, db, llm, ranker.ensure_profile(cfg, db))
        garden = db.conn.execute("SELECT id FROM articles WHERE source='Garden' LIMIT 1").fetchone()["id"]
        db.conn.execute("UPDATE articles SET sent_digest_id=1 WHERE id=?", (garden,))
        db.add_feedback(garden, 1, "")
        db.conn.commit()
        self.assertGreater(embed.update_prefilter(cfg, db, llm, ranker.ensure_profile(cfg, db)), 0)
        best = max(db.pool(3), key=lambda a: a["prefilter"])
        self.assertEqual(best["source"], "Garden")
        self.assertEqual(embed.update_prefilter(cfg, db, llm, ranker.ensure_profile(cfg, db)), 0)

    def test_model_change_resets_embeddings(self):
        cfg, db, llm = self._cfg(), self._db(2), self._llm()
        embed.embed_new(cfg, db, llm)
        cfg.llm.embedding_model = "other-model"
        self.assertEqual(embed.embed_new(cfg, db, llm), 6)

    def test_numpy_and_python_scores_agree(self):
        if embed.np is None:
            self.skipTest("numpy not installed")
        from test_indefinite_articles import fake_embedding
        vecs = [embed.pack(fake_embedding(t)) for t in ("rust code", "garden soil", "rust garden")]
        wanted = [embed.pack(fake_embedding("rust compilers")), embed.pack(fake_embedding("tomatoes"))]
        unwanted = [embed.pack(fake_embedding("garden"))]
        likes = [embed.pack(fake_embedding("soil compost")), embed.pack(fake_embedding("code"))]
        dis = [embed.pack(fake_embedding("gossip"))]
        for args in ((wanted, unwanted, likes, dis), (wanted, [], [], [])):
            a = embed._scores_numpy(vecs, *args)
            b = embed._scores_python(vecs, *args)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=5)

    def test_split_profile(self):
        likes, avoid = embed.split_profile(
            "- Likes Rust and compilers, but not marketing posts\n"
            "- Dislikes celebrity gossip. Enjoys long science features.\n"
            "* Fewer product announcements")
        self.assertEqual(likes, ["Likes Rust and compilers", "Enjoys long science features"])
        self.assertEqual(avoid, ["not marketing posts", "Dislikes celebrity gossip",
                                 "Fewer product announcements"])

    def test_profile_dislikes_push_articles_down(self):
        # A profile naming a topic only as a dislike must not rank that topic first.
        cfg, db, llm = self._cfg(), self._db(), self._llm()
        cfg.interests = "I like tomatoes compost garden soil. I dislike celebrity gossip red carpet fashion."
        llm.chat_json.side_effect = LLMError("down")  # keyword split
        embed.embed_new(cfg, db, llm)
        embed.update_prefilter(cfg, db, llm, ranker.ensure_profile(cfg, db))
        ranked = sorted(db.pool(3), key=lambda a: a["prefilter"], reverse=True)
        self.assertEqual(ranked[0]["source"], "Garden")
        self.assertEqual(ranked[-1]["source"], "Celeb News")

    def test_profile_phrases_prefers_llm_split(self):
        llm = self._llm()
        llm.chat_json.return_value = {"likes": ["rust compilers"], "avoid": ["celebrity gossip"]}
        self.assertEqual(embed.profile_phrases(llm, "whatever"),
                         (["rust compilers"], ["celebrity gossip"]))
        llm.chat_json.return_value = {"oops": 1}  # unusable answer: keyword split
        self.assertEqual(embed.profile_phrases(llm, "Likes rust. Avoids gossip."),
                         (["Likes rust"], ["Avoids gossip"]))


if __name__ == "__main__":
    unittest.main()
