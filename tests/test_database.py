import time
import unittest

from indefinite_articles.database import Database, PendingFeedback, SourceStatistics
from indefinite_articles.extract import ArticleContent
from indefinite_articles.entry import Entry
from indefinite_articles.feeds import CacheValidators
from indefinite_articles.ranker import ArticleScore, DigestPick

DAY = 86400


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)

    def _article(self, url: str, title: str | None = None, source: str = "S", published: float | None = None,
                 category: str = "") -> int:
        title = title or url
        self.assertTrue(self.database.add_article(Entry(url, title, "summary", published), source, category))
        self.database.connection.commit()
        return self.database.connection.execute("SELECT id FROM articles WHERE url=?", (url,)).fetchone()["id"]

    def _sent(self, article_id: int):
        self.database.connection.execute("UPDATE articles SET sent_digest_id=1 WHERE id=?", (article_id,))

    def _age(self, article_id: int, days: float):
        self.database.connection.execute("UPDATE articles SET fetched_at=? WHERE id=?",
                                         (time.time() - days * DAY, article_id))

    # -- state and profile -----------------------------------------------------
    def test_state(self):
        self.assertIsNone(self.database.get_state("key"))
        self.assertEqual(self.database.get_state("key", "fallback"), "fallback")
        self.database.set_state("key", "one")
        self.database.set_state("key", "two")
        self.assertEqual(self.database.get_state("key"), "two")

    def test_profiles(self):
        self.assertIsNone(self.database.current_profile())
        first = self.database.add_profile("first", note="initial")
        second = self.database.add_profile("second")
        self.assertGreater(second, first)
        profile = self.database.current_profile()
        assert profile is not None
        self.assertEqual((profile["id"], profile["text"], profile["note"]), (second, "second", ""))

    # -- articles --------------------------------------------------------------
    def test_add_article_deduplicates_by_url_and_title_key(self):
        self.assertTrue(self.database.add_article(Entry("https://a", "A", "", None), "S"))
        self.assertFalse(self.database.add_article(Entry("https://a", "Other", "", None), "S"))
        self.assertFalse(self.database.add_article(Entry("https://b", "A!", "", None), "T"))
        # An empty title key never matches another article.
        self.assertTrue(self.database.add_article(Entry("https://c", "?", "", None), "S"))
        self.assertTrue(self.database.add_article(Entry("https://d", "!", "", None), "S"))
        self.assertEqual(self.database.statistics()["articles"], 3)

    def test_candidates_balance_sources_and_skip_old_or_sent(self):
        for index in range(20):
            self._article(f"https://busy/{index}", source="Busy")
        for index in range(2):
            self._article(f"https://quiet/{index}", source="Quiet")
        self._article("https://old", source="Quiet", published=time.time() - 10 * DAY)
        self._sent(self._article("https://sent", source="Quiet"))
        candidates = self.database.candidates(3, 10)
        sources = [article.source for article in candidates]
        self.assertEqual(sources.count("Busy"), 5)
        self.assertEqual(sources.count("Quiet"), 2)
        self.assertNotIn("https://old", [article.url for article in candidates])

    def test_pool_is_newest_first(self):
        self._article("https://older", published=time.time() - DAY)
        self._article("https://newer", published=time.time() - 60)
        self._article("https://ancient", published=time.time() - 30 * DAY)
        self._sent(self._article("https://sent"))
        self.assertEqual([article.url for article in self.database.pool(3)], ["https://newer", "https://older"])
        self.assertEqual(len(self.database.pool(3, limit=1)), 1)

    def test_set_content_and_score(self):
        article_id = self._article("https://a")
        self.database.set_content(ArticleContent(article_id, "full text", "ok"))
        self.database.set_score(article_id, ArticleScore(7.5, "fits"), 3)
        row = self.database.pool(3)[0]
        self.assertEqual((row.content, row.content_status), ("full text", "ok"))
        self.assertEqual((row.score, row.reason, row.score_profile_id), (7.5, "fits", 3))

    def test_articles_per_source(self):
        self._article("https://a/1", source="A")
        self._article("https://a/2", source="A")
        self._age(self._article("https://b/1", source="B"), 10)
        self.assertEqual(self.database.articles_per_source(7), {"A": 2})

    # -- embeddings and pre-filter --------------------------------------------
    def test_unembedded(self):
        fresh = self._article("https://fresh")
        embedded = self._article("https://embedded")
        self.database.set_embedding(embedded, b"vector")
        sent = self._article("https://sent")
        self._sent(sent)
        rated = self._article("https://rated", published=time.time() - 100 * DAY)
        self._sent(rated)
        self.database.add_feedback(rated, 1, "")
        ids = [row.id for row in self.database.unembedded(3, 100)]
        self.assertEqual(sorted(ids), sorted([fresh, rated]))
        self.assertEqual(len(self.database.unembedded(3, 1)), 1)

    def test_rated_and_training_vectors(self):
        liked, disliked, noted = (self._article(f"https://{name}") for name in ("liked", "disliked", "noted"))
        for article_id, vector in ((liked, b"like"), (disliked, b"dislike"), (noted, b"noted")):
            self.database.set_embedding(article_id, vector)
        self.database.add_feedback(liked, 1, "")
        self.database.add_feedback(disliked, -1, "")
        self.database.add_feedback(noted, 0, "just a comment")
        self.assertEqual(self.database.rated_vectors(1, 10), [b"like"])
        self.assertEqual(self.database.rated_vectors(-1, 10), [b"dislike"])
        training = {bytes(row["vec"]): row["rating"] for row in self.database.training_vectors(10)}
        self.assertEqual(training, {b"like": 1, b"dislike": -1})

    def test_prefilter_todo_and_set_prefilter(self):
        first, second = self._article("https://1"), self._article("https://2")
        self._article("https://no-vector")
        for article_id in (first, second):
            self.database.set_embedding(article_id, b"v")
        self.assertEqual(sorted(row["id"] for row in self.database.prefilter_todo(3, 1)), [first, second])
        self.database.set_prefilter({first: 0.5}, 1)
        self.assertEqual([row["id"] for row in self.database.prefilter_todo(3, 1)], [second])
        self.assertEqual(len(self.database.prefilter_todo(3, 2)), 2)  # a new version redoes everything

    def test_feedback_signature_changes_with_feedback(self):
        article_id = self._article("https://a")
        before = self.database.feedback_signature()
        self.database.add_feedback(article_id, 1, "")
        self.assertNotEqual(self.database.feedback_signature(), before)

    # -- feeds -------------------------------------------------------------------
    def test_feed_failures_back_off_and_success_resets(self):
        url = "https://feed"
        self.database.feed_failed(url, "timeout", 3600)
        state = self.database.feed_states()[url]
        self.assertEqual(state["failures"], 1)
        self.assertLessEqual(state["next_attempt"], time.time())  # retried next cycle
        self.database.feed_failed(url, "x" * 1000, 3600)
        state = self.database.feed_states()[url]
        self.assertEqual(state["failures"], 2)
        self.assertAlmostEqual(state["next_attempt"] - state["last_attempt"], 7200)
        self.assertEqual(len(state["last_error"]), 300)
        for _ in range(20):
            self.database.feed_failed(url, "down", 3600)
        state = self.database.feed_states()[url]
        self.assertAlmostEqual(state["next_attempt"] - state["last_attempt"], DAY)  # capped

        self.database.feed_ok(url, "Title", CacheValidators('"etag"', "yesterday"))
        self.database.feed_ok(url, "", CacheValidators('"etag2"', None))  # a 304 keeps the title
        state = self.database.feed_states()[url]
        self.assertEqual((state["title"], state["etag"], state["last_modified"]), ("Title", '"etag2"', None))
        self.assertEqual((state["failures"], state["last_error"], state["next_attempt"]), (0, "", 0))

    # -- feedback ------------------------------------------------------------------
    def test_feedback_is_one_per_article(self):
        article_id = self._article("https://a", source="Blog")
        self.database.add_feedback(article_id, 1, "great")
        self.database.add_feedback(article_id, -1, "on reflection, no")
        statistics = self.database.statistics()
        self.assertEqual((statistics["likes"], statistics["dislikes"]), (0, 1))
        self.assertEqual([tuple(row) for row in self.database.rated_examples(-1, 5)],
                         [("https://a", "Blog", "on reflection, no")])
        self.assertEqual(self.database.rated_examples(1, 5), [])

    def test_source_statistics(self):
        for index, rating in enumerate((1, 1, -1, 0)):
            self.database.add_feedback(self._article(f"https://a/{index}", source="A"), rating, "")
        self.database.add_feedback(self._article("https://b", source="B"), -1, "")
        self.assertEqual(self.database.source_statistics(), {"A": SourceStatistics(2, 1), "B": SourceStatistics(0, 1)})

    def test_feedback_since(self):
        article_id = self._article("https://a")
        self.database.add_feedback(article_id, 1, "nice")
        self.database.add_general_comment(None, "more science")
        pending = self.database.feedback_since(0)
        self.assertEqual([(row["title"], row["rating"], row["comment"]) for row in pending.feedback],
                         [("https://a", 1, "nice")])
        self.assertEqual([row["comment"] for row in pending.comments], ["more science"])
        self.assertEqual(self.database.feedback_since(time.time() + 1), PendingFeedback([], []))

    def test_processed_messages(self):
        self.assertFalse(self.database.message_processed("<a@x>"))
        self.database.mark_message("<a@x>", "ok")
        self.assertTrue(self.database.message_processed("<a@x>"))
        self.database.mark_message("<b@x>", "ok", commit=False)
        self.database.connection.rollback()
        self.assertFalse(self.database.message_processed("<b@x>"))

    # -- digests -------------------------------------------------------------------
    def test_record_and_find_digest(self):
        first, second = self._article("https://1"), self._article("https://2")
        articles = {article.id: article for article in self.database.pool(3)}
        digest_id = self.database.record_digest("abcd1234", "<digest@x>", "2026-11-03",
                                                [DigestPick(articles[second], False), DigestPick(articles[first], True)])
        self.assertEqual(self.database.digest_items(digest_id), [articles[second], articles[first]])
        exploratory = self.database.connection.execute(
            "SELECT exploratory FROM digest_items WHERE digest_id=? ORDER BY position", (digest_id,)).fetchall()
        self.assertEqual([row["exploratory"] for row in exploratory], [0, 1])
        self.assertEqual(self.database.pool(3), [])  # both now sent
        found = self.database.find_digest("abcd1234")
        assert found is not None
        self.assertEqual(found["id"], digest_id)
        found = self.database.find_digest("deadbeef", ["<other@x>", " <digest@x> "])
        assert found is not None
        self.assertEqual(found["local_date"], "2026-11-03")
        self.assertIsNone(self.database.find_digest(None, ["<other@x>"]))
        self.assertIsNone(self.database.find_digest())

    # -- maintenance -----------------------------------------------------------------
    def test_prune_keeps_sent_articles(self):
        old = self._article("https://old")
        self.database.set_embedding(old, b"v")
        self._age(old, 40)
        old_sent = self._article("https://old-sent")
        self._sent(old_sent)
        self._age(old_sent, 40)
        recent = self._article("https://recent")
        self.database.mark_message("<old@x>", "ok")
        self.database.connection.execute("UPDATE processed_messages SET processed_at=?", (time.time() - 40 * DAY,))
        self.database.mark_message("<new@x>", "ok")
        self.database.prune(30)
        ids = {row["id"] for row in self.database.connection.execute("SELECT id FROM articles")}
        self.assertEqual(ids, {old_sent, recent})
        self.assertEqual(self.database.statistics()["embedded"], 0)
        self.assertFalse(self.database.message_processed("<old@x>"))
        self.assertTrue(self.database.message_processed("<new@x>"))

    def test_statistics(self):
        article_id = self._article("https://a")
        self.database.set_content(ArticleContent(article_id, "text", "ok"))
        self.database.add_feedback(article_id, 1, "")
        self.database.feed_failed("https://feed", "down", 60)
        statistics = self.database.statistics()
        self.assertEqual(statistics["articles"], 1)
        self.assertEqual(statistics["full_text"], 1)
        self.assertEqual(statistics["likes"], 1)
        self.assertEqual(statistics["feeds_failing"], 1)
        self.assertEqual(statistics["digests"], 0)

    def test_file_database_creates_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            database = Database(f"{directory}/nested/dir/data.db")
            database.set_state("key", "value")
            database.close()
            reopened = Database(f"{directory}/nested/dir/data.db")
            self.assertEqual(reopened.get_state("key"), "value")
            reopened.close()


if __name__ == "__main__":
    unittest.main()
