import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from indefinite_articles.config import Config, load as load_config
from indefinite_articles.fetching import feeds
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.fetching.feed import Feed
from indefinite_articles.fetching.parsed_feed import ParsedFeed
from indefinite_articles.language_model.llm import LLM, LLMError
from indefinite_articles.ranking import embed, ranker
from indefinite_articles.storage.article import Article
from indefinite_articles.storage.database import Database

ROOT = Path(__file__).resolve().parent.parent

FEED = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
<item><title>Post one</title><link>https://ex.com/1</link></item></channel></rss>"""


class FeedServer(BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, format: str, *args: object):
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


def load_feed_files(files: dict[str, str]) -> list[Feed]:
    """The feeds of a config whose feeds directory holds `files` (name to contents)."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "feeds.d").mkdir()
        for name, contents in files.items():
            (Path(directory) / "feeds.d" / name).write_text(contents)
        (Path(directory) / "config.toml").write_text("")
        return load_config(Path(directory) / "config.toml").feeds


class FeedListTests(unittest.TestCase):
    def test_opml(self):
        opml = """<?xml version="1.0"?><opml version="2.0"><body>
        <outline text="Science"><outline text="Quanta" xmlUrl="https://q/feed" maxItems="5"/>
          <outline text="Sub"><outline title="Deep" xmlUrl="https://d/feed"/></outline></outline>
        <outline text="Loose" xmlUrl="https://l/feed"/></body></opml>"""
        feed_list = load_feed_files({"feeds.opml": opml})
        self.assertEqual([(feed.name, feed.category, feed.max_items) for feed in feed_list],
                         [("Quanta", "Science", 5), ("Deep", "Sub", 40), ("Loose", "", 40)])

    def test_txt(self):
        feed_list = load_feed_files({
            "feeds.txt": "# Tech\nhttps://a/feed  A blog\n\nhttps://b/feed\n# News\nhttps://c/feed",
        })
        self.assertEqual([(feed.url, feed.name, feed.category) for feed in feed_list],
                         [("https://a/feed", "A blog", "Tech"), ("https://b/feed", "", "Tech"),
                          ("https://c/feed", "", "News")])

    def test_exclude_categories_and_dedupe(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as directory:
            os.mkdir(f"{directory}/feeds.d")
            Path(f"{directory}/feeds.d/a.txt").write_text("# Sports\nhttps://s/feed\n# Tech\nhttps://t/feed\n")
            Path(f"{directory}/r.toml").write_text(
                '[general]\nexclude_categories = ["sports"]\n[[feeds]]\nurl = "https://t/feed"\nname = "dup"\n')
            config = load_config(f"{directory}/r.toml")
            self.assertEqual([(feed.url, feed.name) for feed in config.feeds], [("https://t/feed", "dup")])


class LenientParseTests(unittest.TestCase):
    def test_html_entities_and_leading_junk(self):
        data = (
            b"\n\n<?xml version='1.0'?><rss><channel><title>X&nbsp;Y</title>"
            b"<item><title>A &mdash; B</title><link>https://e/1</link></item></channel></rss>"
        )
        parsed = ParsedFeed.parse(data)
        title, entries = parsed.title, parsed.entries
        self.assertEqual(title, "X\xa0Y".replace("\xa0", " ").strip() or title)
        self.assertEqual(entries[0].title, "A — B")

    def test_broken_markup_regex_fallback(self):
        data = (
            b"<rss><channel><title>T</title><item><title>Good</title><link>https://e/1</link>"
            b"<description><![CDATA[<p>hi</p>]]></description></item><br></channel></rss>"
        )
        parsed = ParsedFeed.parse(data)
        title, entries = parsed.title, parsed.entries
        self.assertEqual((entries[0].title, entries[0].summary), ("Good", "hi"))

    def test_html_page_rejected(self):
        with self.assertRaises(ValueError):
            ParsedFeed.parse(b"<!DOCTYPE html><html><body>nope</body></html>")


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
        config = Config()
        config.feeds = [Feed(url=f"{self.base}/a", category="C"), Feed(url=f"{self.base}/b"),
                     Feed(url=f"{self.base}/broken")]
        database = Database(":memory:")
        self.addCleanup(database.close)
        FeedServer.hits.clear()
        self.assertEqual(feeds.update_all(config, database), 2)
        row = database.connection.execute("SELECT category FROM articles WHERE url LIKE '%/a'").fetchone()
        self.assertEqual(row["category"], "C")
        # Second run: unchanged feeds answer 304 and add nothing; broken feed retried once.
        self.assertEqual(feeds.update_all(config, database), 0)
        self.assertIn(("/a", '"v1"'), FeedServer.hits)
        state = database.feed_states()
        self.assertEqual(state[f"{self.base}/broken"]["failures"], 2)
        # ...and then it backs off instead of being fetched every cycle.
        self.assertGreater(state[f"{self.base}/broken"]["next_attempt"], time.time() + 3000)
        FeedServer.hits.clear()
        feeds.update_all(config, database)
        self.assertNotIn("/broken", [path for path, _ in FeedServer.hits])


def prefilter_score(article: Article) -> float:
    """The article's pre-filter score, which these tests expect every article to have."""
    assert article.prefilter is not None, article.title
    return article.prefilter


class PrefilterTests(unittest.TestCase):
    def _config(self):
        config = Config()
        config.max_candidates = 20
        config.interests = "rust compilers systems programming"
        return config

    def _database(self, per_topic: int = 30):
        database = Database(":memory:")
        self.addCleanup(database.close)
        topics = {"Rust Weekly": "rust compilers borrow checker systems programming",
                  "Celeb News": "celebrity gossip red carpet fashion",
                  "Garden": "tomatoes compost garden soil"}
        for source, words in topics.items():
            for index in range(per_topic):
                database.add_article(Entry(f"https://{source}/{index}", f"{words} {index}", words, None),
                                     source)
        database.connection.commit()
        return database

    def _llm(self):
        from fake_llm import fake_embedding
        llm = mock.Mock(spec=LLM)
        llm.embed.side_effect = lambda texts, purpose="texts": [fake_embedding(text) for text in texts]
        return llm

    def test_shortlist_prefers_matches_with_source_cap_and_exploration(self):
        config, database, llm = self._config(), self._database(), self._llm()
        embed.embed_new(config, database, llm)
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        shortlisted = ranker.shortlist(config, database)
        self.assertEqual(len(shortlisted), 20)
        by_source = {}
        for article in shortlisted:
            by_source[article.source] = by_source.get(article.source, 0) + 1
        top = [article.source for article in shortlisted[:3]]
        self.assertEqual(top, ["Rust Weekly"] * 3)
        # The cap spreads the first picks across sources; with only 3 sources the list is
        # then topped up from the best remaining matches, i.e. mostly Rust.
        first9 = [article.source for article in shortlisted[:9]]
        self.assertEqual(sorted(set(first9)), ["Celeb News", "Garden", "Rust Weekly"])
        self.assertEqual(first9.count("Rust Weekly"), 3)
        self.assertGreater(by_source["Rust Weekly"], 10)
        self.assertEqual(ranker.shortlist(config, database)[-2:], shortlisted[-2:])  # random picks stable

    def test_feedback_changes_ranking(self):
        config, database, llm = self._config(), self._database(), self._llm()
        config.interests = "anything"
        embed.embed_new(config, database, llm)
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        garden = database.connection.execute("SELECT id FROM articles WHERE source='Garden' LIMIT 1").fetchone()["id"]
        database.connection.execute("UPDATE articles SET sent_digest_id=1 WHERE id=?", (garden,))
        database.add_feedback(garden, 1, "")
        database.connection.commit()
        self.assertGreater(embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database)), 0)
        best = max(database.pool(3), key=prefilter_score)
        self.assertEqual(best.source, "Garden")
        self.assertEqual(embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database)), 0)

    def test_model_change_resets_embeddings(self):
        config, database, llm = self._config(), self._database(2), self._llm()
        embed.embed_new(config, database, llm)
        config.llm.embedding_model = "other-model"
        self.assertEqual(embed.embed_new(config, database, llm), 6)

    def _rate(self, database: Database, source: str, rating: int, count: int = 6):
        ids = [row["id"] for row in database.connection.execute(
            "SELECT id FROM articles WHERE source=? LIMIT ?", (source, count))]
        for article_id in ids:
            database.connection.execute("UPDATE articles SET sent_digest_id=1 WHERE id=?", (article_id,))
            database.add_feedback(article_id, rating, "")
        database.connection.commit()

    def test_classifier_learns_from_ratings(self):
        config, database, llm = self._config(), self._database(), self._llm()
        config.interests = "anything"
        embed.embed_new(config, database, llm)
        self._rate(database, "Garden", 1)
        self._rate(database, "Celeb News", -1, count=4)  # too few dislikes: nearest-neighbour scoring
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        self.assertEqual(database.get_state("classifier"), "")
        self._rate(database, "Celeb News", -1)
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        self.assertNotEqual(database.get_state("classifier"), "")
        self.assertEqual(database.get_state("classifier_info"), "6:6")
        ranked = sorted(database.pool(3), key=prefilter_score, reverse=True)
        self.assertEqual(ranked[0].source, "Garden")
        self.assertEqual(ranked[-1].source, "Celeb News")

    def test_profile_dislikes_push_articles_down(self):
        # A profile naming a topic only as a dislike must not rank that topic first.
        config, database, llm = self._config(), self._database(), self._llm()
        config.interests = "I like tomatoes compost garden soil. I dislike celebrity gossip red carpet fashion."
        llm.chat_json.side_effect = LLMError("down")  # keyword split
        embed.embed_new(config, database, llm)
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        ranked = sorted(database.pool(3), key=prefilter_score, reverse=True)
        self.assertEqual(ranked[0].source, "Garden")
        self.assertEqual(ranked[-1].source, "Celeb News")

    def test_profile_phrases_prefers_llm_split(self):
        config, database, llm = self._config(), self._database(0), self._llm()

        def phrases_for(profile_text: str) -> dict:
            database.add_profile(profile_text)
            profile = database.current_profile()
            assert profile is not None
            embed.update_prefilter(config, database, llm, profile)
            return json.loads(database.get_state("profile_phrases", ""))

        llm.chat_json.return_value = {"likes": ["rust compilers"], "avoid": ["celebrity gossip"]}
        self.assertEqual(phrases_for("whatever"), {"wanted": ["rust compilers"], "unwanted": ["celebrity gossip"]})
        llm.chat_json.return_value = {"oops": 1}  # unusable answer: keyword split
        self.assertEqual(
            phrases_for("Likes rust. Avoids gossip."),
            {"wanted": ["Likes rust"], "unwanted": ["Avoids gossip"]},
        )


if __name__ == "__main__":
    unittest.main()
