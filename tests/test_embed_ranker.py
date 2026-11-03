import random
import threading
import time
import unittest
from unittest import mock

import numpy as np

from indefinite_articles import embed, ranker
from indefinite_articles.feeds import Entry
from indefinite_articles.classifier import Classifier
from indefinite_articles.config import Config
from indefinite_articles.database import Database
from indefinite_articles.llm import LLM, LLMError, LLMUnavailable


def unit(*values: float) -> bytes:
    return embed.pack(list(values))


def article(**fields: object) -> dict:
    row = {"id": 1, "title": "Title", "summary": "", "content": None, "source": "Source", "category": ""}
    row.update(fields)
    return row


class EmbedTests(unittest.TestCase):
    def test_pack_normalises_to_float32(self):
        vector = np.frombuffer(embed.pack([3, 4]), dtype=np.float32)
        self.assertEqual(vector.tolist(), [0.6000000238418579, 0.800000011920929])
        self.assertEqual(np.frombuffer(embed.pack([0, 0]), dtype=np.float32).tolist(), [0.0, 0.0])

    def test_article_text(self):
        self.assertEqual(embed.article_text(article(summary=None)), "Title. ")
        self.assertEqual(len(embed.article_text(article(summary="x" * 1000))), len("Title. ") + 600)

    def test_score_and_profile_match(self):
        self.assertEqual(embed.score([], [unit(1, 0)], [], [], []), [])
        matches = embed.profile_match([unit(1, 0), unit(0, 1), unit(1, 1)], [unit(1, 0)], [unit(0, 1)])
        self.assertAlmostEqual(matches[0], 1.0, places=5)
        self.assertAlmostEqual(matches[1], -0.5, places=5)
        self.assertEqual(sorted(range(3), key=lambda index: -matches[index]), [0, 2, 1])

    def test_score_uses_likes_and_dislikes(self):
        vectors = [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]
        scores = embed.score(vectors, [unit(1, 1, 1)], [], likes=[unit(0, 1, 0)], dislikes=[unit(0, 0, 1)])
        self.assertEqual(max(range(3), key=lambda index: scores[index]), 1)
        self.assertEqual(min(range(3), key=lambda index: scores[index]), 2)

    def test_score_with_classifier(self):
        classifier = Classifier(np.array([10.0, -10.0]), 0.0)
        vectors = [unit(1, 0), unit(0, 1)]
        probabilities = classifier.predict(vectors)
        scores = embed.score_with_classifier(vectors, [unit(1, 1)], [], classifier)
        matches = embed.profile_match(vectors, [unit(1, 1)], [])
        self.assertAlmostEqual(scores[0] - matches[0], embed.CLASSIFIER_WEIGHT * (probabilities[0] - 0.5))
        self.assertGreater(scores[0], scores[1])

    def test_fit_from_database(self):
        database = Database(":memory:")
        self.addCleanup(database.close)

        def rate(index: int, vector: bytes, rating: int):
            database.add_article(Entry(f"https://e/{index}", f"t{index}", "", None), "S")
            database.set_embedding(index, vector)
            database.add_feedback(index, rating, "")

        for index in range(1, 6):
            rate(index, unit(1, 0.1 * index), 1)
        for index in range(6, 10):
            rate(index, unit(0.1 * index, 1), -1)
        rate(10, unit(1, 0, 0), -1)  # another model's vector: ignored
        fit = embed.fit_from_database(database, len(unit(1, 0)))
        self.assertIsNone(fit.classifier)
        self.assertEqual((fit.like_count, fit.dislike_count), (5, 4))
        rate(11, unit(0, 1), -1)
        fit = embed.fit_from_database(database, len(unit(1, 0)))
        assert fit.classifier is not None
        self.assertEqual(fit.dislike_count, 5)
        liked, disliked = fit.classifier.predict([unit(1, 0), unit(0, 1)])
        self.assertGreater(liked, disliked)

    def test_embed_new_stops_when_asked(self):
        config, database = Config(), Database(":memory:")
        self.addCleanup(database.close)
        database.add_article(Entry("https://e/1", "t", "", None), "S")
        llm = mock.Mock(spec=LLM)
        stop = threading.Event()
        stop.set()
        self.assertEqual(embed.embed_new(config, database, llm, stop=stop), 0)
        llm.embed.assert_not_called()
        self.assertEqual(database.get_state("embedding_model"), config.llm.embedding_model)

    def test_dislike_only_profile_still_wants_something(self):
        config, database = Config(interests="Dislikes celebrity gossip"), Database(":memory:")
        self.addCleanup(database.close)
        llm = mock.Mock(spec=LLM)
        llm.chat_json.return_value = {"likes": [], "avoid": ["celebrity gossip"]}  # unusable: keyword split
        llm.embed.side_effect = lambda texts: [[1.0, float(index)] for index, _ in enumerate(texts)]
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        self.assertEqual([call.args[0] for call in llm.embed.call_args_list],
                         [[embed.DEFAULT_WANTED], ["Dislikes celebrity gossip"]])
        # Nothing changed: no second LLM call.
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
        self.assertEqual(llm.chat_json.call_count, 1)


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.articles_per_digest = 4
        self.config.exploration = 1
        self.config.max_per_source = 2
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)

    def _add(self, url: str, source: str = "S", score: float | None = None, profile_id: int | None = None,
             prefilter: float | None = None, title: str | None = None):
        self.database.add_article(Entry(url, title or url, "summary", time.time()), source)
        self.database.connection.execute(
            "UPDATE articles SET score=?, score_profile_id=?, prefilter=? WHERE url=?",
            (score, profile_id, prefilter, url))
        self.database.connection.commit()

    def test_ensure_profile(self):
        self.assertEqual(ranker.ensure_profile(self.config, self.database)["text"], ranker.DEFAULT_PROFILE)
        self.config.interests = "Rust"
        self.assertEqual(ranker.ensure_profile(self.config, self.database)["text"], ranker.DEFAULT_PROFILE)
        other = Database(":memory:")
        self.addCleanup(other.close)
        profile = ranker.ensure_profile(self.config, other)
        self.assertEqual((profile["text"], profile["note"]), ("Rust", "initial (from config)"))

    def test_article_text(self):
        self.assertEqual(ranker.article_text(article(content="Full   page\ntext", summary="Sum"), 100),
                         "Full page text")
        self.assertEqual(ranker.article_text(article(content="Short", summary="A longer summary"), 100),
                         "A longer summary")
        self.assertEqual(ranker.article_text(article(summary=None), 100), "")
        self.assertEqual(ranker.article_text(article(summary="one two three four"), 10), "one two…")

    def test_score_prefix(self):
        prefix = ranker.score_prefix("PROFILE", "EXAMPLES")
        self.assertTrue(prefix.startswith("Reader profile:\nPROFILE\n\nEXAMPLES"))
        self.assertIn('"score"', prefix)
        self.assertEqual(prefix, ranker.score_prefix("PROFILE", "EXAMPLES"))

    def test_score_one_answer_shapes(self):
        llm = mock.Mock()
        for answer, expected in (({"ratings": [{"score": "6", "reason": "r"}]}, ranker.ArticleScore(6.0, "r")),
                                 ([{"score": -3}], ranker.ArticleScore(0.0, "")),
                                 ({"score": 4, "reason": "x" * 300}, ranker.ArticleScore(4.0, "x" * 200))):
            llm.chat_json.return_value = answer
            with self.subTest(answer=answer):
                self.assertEqual(ranker.score_one(llm, "", article()), expected)
        llm.chat_json.return_value = "seven"
        with self.assertRaises(LLMError):
            ranker.score_one(llm, "", article())
        llm.chat_json.return_value = {"score": 1}
        ranker.score_one(llm, "", article(category="Tech"))
        self.assertIn("Source: Source (Tech)", llm.chat_json.call_args[0][1])

    def test_shortlist_without_prefilter_uses_newest(self):
        for index in range(5):
            self._add(f"https://e/{index}")
        self.config.max_candidates = 3
        self.assertEqual(len(ranker.shortlist(self.config, self.database)), 3)

    def test_score_candidates(self):
        for index in range(3):
            self._add(f"https://e/{index}")
        llm = mock.Mock(spec=LLM)
        llm.chat_json.side_effect = [{"score": 8}, LLMError("garbled"), {"score": 3}]
        with self.assertLogs("indefinite_articles.ranker", "WARNING"):
            self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 2)
        # Only the failed article is left; an unavailable LLM stops the round.
        llm.chat_json.side_effect = LLMUnavailable("down")
        with self.assertLogs("indefinite_articles.ranker", "WARNING"):
            self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 0)
        self.assertEqual(llm.chat_json.call_count, 4)
        # A new profile makes everything due again.
        self.database.add_profile("new profile")
        llm.chat_json.side_effect = None
        llm.chat_json.return_value = {"score": 5}
        stop = threading.Event()
        stop.set()
        self.assertEqual(ranker.score_candidates(self.config, self.database, llm, stop=stop), 0)
        self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 3)
        self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 0)

    def test_select_empty(self):
        self.assertEqual(ranker.select(self.config, self.database), [])

    def test_select_ranks_caps_and_explores(self):
        profile_id = ranker.ensure_profile(self.config, self.database)["id"]
        self._add("https://a/1", "A", 9.0, profile_id)
        self._add("https://a/2", "A", 8.5, profile_id)
        self._add("https://a/3", "A", 8.0, profile_id)  # over the per-source cap
        self._add("https://b/1", "B", 6.0, profile_id)
        self._add("https://c/1", "C", 1.0, profile_id)
        self._add("https://d/1", "D", 0.5, profile_id)
        debug: dict = {}
        chosen = ranker.select(self.config, self.database, random.Random(1), debug)
        self.assertEqual([(pick.article["url"], pick.exploratory) for pick in chosen[:3]],
                         [("https://a/1", False), ("https://a/2", False), ("https://b/1", False)])
        self.assertEqual(len(chosen), 4)
        self.assertTrue(chosen[3].exploratory)
        self.assertIn(chosen[3].article["url"], ("https://c/1", "https://d/1"))
        self.assertEqual((debug["candidates"], debug["llm_scored"], debug["stale_scored"]), (6, 6, 0))
        self.assertEqual(debug["rank"][chosen[0].article["id"]], 1)
        self.assertIsInstance(debug["scores"][chosen[0].article["id"]], ranker.ScoreBreakdown)

    def test_select_fills_past_cap_with_few_sources(self):
        for index in range(5):
            self._add(f"https://a/{index}", "A")
        chosen = ranker.select(self.config, self.database, random.Random(1))
        self.assertEqual(len(chosen), 4)
        self.assertEqual(sum(pick.exploratory for pick in chosen), 0)  # cap filled, exploration impossible

    def test_select_source_ratings_move_unscored_articles(self):
        self._add("https://liked/new", "Liked")
        self._add("https://disliked/new", "Disliked")
        for index in range(3):
            self._add(f"https://liked/{index}", "Liked")
            self._add(f"https://disliked/{index}", "Disliked")
        rows = self.database.connection.execute("SELECT id, source FROM articles WHERE url LIKE '%/_'")
        for row in rows.fetchall():
            self.database.connection.execute("UPDATE articles SET sent_digest_id=1 WHERE id=?", (row["id"],))
            self.database.add_feedback(row["id"], 1 if row["source"] == "Liked" else -1, "")
        self.database.connection.commit()
        self.config.articles_per_digest, self.config.exploration = 1, 0
        chosen = ranker.select(self.config, self.database, random.Random(1))
        self.assertEqual(chosen[0].article["url"], "https://liked/new")

    def test_update_profile(self):
        llm = mock.Mock(spec=LLM)
        self.assertFalse(ranker.update_profile(self.config, self.database, llm))  # nothing pending
        llm.chat.assert_not_called()
        self._add("https://e/1", title="Liked one")
        self.database.add_feedback(1, 1, "more please")
        self.database.add_general_comment(None, "less news")
        self.database.connection.commit()

        llm.chat.side_effect = LLMError("down")
        with self.assertLogs("indefinite_articles.ranker", "WARNING"):
            self.assertFalse(ranker.update_profile(self.config, self.database, llm))
        llm.chat.side_effect = None
        llm.chat.return_value = "too short"
        with self.assertLogs("indefinite_articles.ranker", "WARNING"):
            self.assertFalse(ranker.update_profile(self.config, self.database, llm))
        llm.chat.return_value = "  - Likes careful explanations of things\n"
        self.assertTrue(ranker.update_profile(self.config, self.database, llm))
        prompt = llm.chat.call_args[0][1]
        self.assertIn('- Liked one (S) — "more please"', prompt)
        self.assertIn('- "less news"', prompt)
        profile = self.database.current_profile()
        assert profile is not None
        self.assertEqual(profile["text"], "- Likes careful explanations of things")
        self.assertEqual(profile["note"], "1 likes, 0 dislikes, 1 comments")


if __name__ == "__main__":
    unittest.main()
