import random
import threading
import time
import unittest
from collections.abc import Sequence
from unittest import mock

import numpy as np

from indefinite_articles import embed, ranker
from indefinite_articles.feeds import Entry
from indefinite_articles.classifier import Classifier
from indefinite_articles.config import Config
from indefinite_articles.database import Database
from indefinite_articles.extract import ArticleContent
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

    def setUp(self):
        self.config, self.database = Config(), Database(":memory:")
        self.addCleanup(self.database.close)
        self.vectors: dict[str, list[float]] = {"wanted": [1.0, 0.0], "unwanted": [0.0, 1.0]}
        self.llm = mock.Mock(spec=LLM)
        self.llm.chat_json.return_value = {"likes": ["wanted"], "avoid": ["unwanted"]}
        self.llm.embed.side_effect = lambda texts: [self.vectors[text] for text in texts]

    def _article(self, title: str, vector: Sequence[float], rating: int = 0):
        """Add an embedded article; a rated one has been sent in a digest."""
        self.database.add_article(Entry(f"https://e/{title}", title, "", None), "S")
        article_id = self.database.connection.execute("SELECT id FROM articles WHERE title=?", (title,)).fetchone()[0]
        self.database.set_embedding(article_id, embed.pack(list(vector)))
        if rating:
            self.database.connection.execute("UPDATE articles SET sent_digest_id=1 WHERE id=?", (article_id,))
            self.database.add_feedback(article_id, rating, "")
        self.database.connection.commit()

    def _prefilter(self) -> dict[str, float]:
        """Run the pre-filter; returns each unsent article's score by title."""
        embed.update_prefilter(self.config, self.database, self.llm, ranker.ensure_profile(self.config, self.database))
        return {row["title"]: row["prefilter"] for row in self.database.pool(3)}

    def test_embedded_text_is_title_and_start_of_summary(self):
        self.database.add_article(Entry("https://e/1", "Short", "", None), "S")
        self.database.add_article(Entry("https://e/2", "Long", "x" * 1000, None), "S")
        self.llm.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        self.assertEqual(embed.embed_new(self.config, self.database, self.llm), 2)
        texts = [text for call in self.llm.embed.call_args_list for text in call.args[0]]
        self.assertEqual(sorted(texts), ["Long. " + "x" * 600, "Short. "])

    def test_profile_match_without_ratings(self):
        profile = ranker.ensure_profile(self.config, self.database)
        self.assertEqual(embed.update_prefilter(self.config, self.database, self.llm, profile), 0)  # nothing embedded
        self._article("wanted topic", [1, 0])
        self._article("unwanted topic", [0, 1])
        self._article("both", [1, 1])
        scores = self._prefilter()
        self.assertAlmostEqual(scores["wanted topic"], 1.0, places=5)
        self.assertAlmostEqual(scores["unwanted topic"], -0.5, places=5)
        self.assertEqual(sorted(scores, key=lambda title: -scores[title]), ["wanted topic", "both", "unwanted topic"])

    def test_likes_and_dislikes_move_scores(self):
        self.vectors["wanted"] = [1.0, 1.0, 1.0]
        self.llm.chat_json.return_value = {"likes": ["wanted"], "avoid": []}
        self._article("liked", [0, 1, 0], rating=1)
        self._article("disliked", [0, 0, 1], rating=-1)
        for title, vector in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
            self._article(title, vector)
        scores = self._prefilter()
        self.assertGreater(scores["y"], scores["x"])  # like the liked article
        self.assertGreater(scores["x"], scores["z"])  # like the disliked article

    def test_classifier_needs_enough_ratings(self):
        for index in range(1, 6):
            self._article(f"liked {index}", [1, 0.1 * index], rating=1)
        for index in range(6, 10):
            self._article(f"disliked {index}", [0.1 * index, 1], rating=-1)
        self._article("other model", [1, 0, 0], rating=-1)  # another model's vector: ignored
        self._prefilter()
        self.assertEqual(self.database.get_state("classifier_info"), "5:4")
        self.assertIsNone(Classifier.load(self.database))
        self._article("disliked 10", [0, 1], rating=-1)
        self._prefilter()
        self.assertEqual(self.database.get_state("classifier_info"), "5:5")
        classifier = Classifier.load(self.database)
        assert classifier is not None
        liked, disliked = classifier.predict([embed.pack([1, 0]), embed.pack([0, 1])])
        self.assertGreater(liked, disliked)

    def test_classifier_blended_with_profile_match(self):
        self.vectors["wanted"] = [1.0, 1.0]
        self.llm.chat_json.return_value = {"likes": ["wanted"], "avoid": []}
        for index in range(1, 6):
            self._article(f"liked {index}", [1, 0.1 * index], rating=1)
            self._article(f"disliked {index}", [0.1 * index, 1], rating=-1)
        self._article("x", [1, 0])
        self._article("y", [0, 1])
        scores = self._prefilter()
        classifier = Classifier.load(self.database)
        assert classifier is not None
        probabilities = classifier.predict([embed.pack([1, 0])])
        match = float(np.dot(np.frombuffer(embed.pack([1, 0]), dtype=np.float32),
                             np.frombuffer(embed.pack([1, 1]), dtype=np.float32)))
        self.assertAlmostEqual(scores["x"], match + embed.CLASSIFIER_WEIGHT * (probabilities[0] - 0.5), places=5)
        self.assertGreater(scores["x"], scores["y"])

    def test_embed_new_stops_when_asked(self):
        config, database = self.config, self.database
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
        phrases = embed._profile_phrases(config, database, llm, ranker.ensure_profile(config, database))
        self.assertEqual(phrases.wanted, [embed.DEFAULT_WANTED])
        self.assertEqual(phrases.unwanted, ["Dislikes celebrity gossip"])
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
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

    def _add_described(self, title: str, summary: str, content: str | None = None, category: str = ""):
        self.database.add_article(Entry(f"https://e/{title}", title, summary, time.time()), "S", category)
        if content is not None:
            article_id = self.database.connection.execute(
                "SELECT id FROM articles WHERE title=?",
                (title,),
            ).fetchone()[0]
            self.database.set_content(ArticleContent(article_id, content, "ok"))
        self.database.connection.commit()

    def _score_all(self, *answers: object) -> list[str]:
        """Score the candidates, the LLM giving `answers` in turn; returns the prompts it was sent."""
        llm = mock.Mock(spec=LLM)
        llm.chat_json.side_effect = answers
        ranker.score_candidates(self.config, self.database, llm)
        return [call.args[1] for call in llm.chat_json.call_args_list]

    def test_prompts_share_prefix_and_describe_each_article(self):
        self.config.article_chars = 20
        self._add_described("Fetched", "Sum", content="Full   page\ntext")
        self._add_described("Fallback", "A longer summary", content="Short")
        self._add_described("Long", "one two three four five six", category="Tech")
        prompts = self._score_all(*[{"reason": "fits", "score": 12}] * 3)
        prefixes = {prompt.split("Article to rate:")[0] for prompt in prompts}
        self.assertEqual(len(prefixes), 1)  # the LLM server can reuse its KV cache
        prefix = prefixes.pop()
        self.assertTrue(prefix.startswith(f"Reader profile:\n{ranker.DEFAULT_PROFILE}\n\nArticles the reader"))
        self.assertIn('"score"', prefix)
        by_title = {prompt.split("Title: ")[1].split("\n")[0]: prompt for prompt in prompts}
        self.assertTrue(by_title["Fetched"].endswith("\nSource: S\nFull page text"))
        self.assertTrue(by_title["Fallback"].endswith("\nA longer summary"))  # summary longer than the page
        self.assertTrue(by_title["Long"].endswith("\nSource: S (Tech)\none two three four…"))
        rows = self.database.connection.execute("SELECT score, reason FROM articles").fetchall()
        self.assertEqual({tuple(row) for row in rows}, {(10.0, "fits")})  # clamped to 0-10

    def test_score_answer_shapes(self):
        self._add("https://e/1")
        for answer, expected in (({"ratings": [{"score": "6", "reason": "r"}]}, (6.0, "r")),
                                 ([{"score": -3}], (0.0, "")),
                                 ({"score": 4, "reason": "x" * 300}, (4.0, "x" * 200))):
            self.database.add_profile(f"profile for {answer}")  # makes the article due again
            with self.subTest(answer=answer):
                self._score_all(answer)
                row = self.database.connection.execute("SELECT score, reason FROM articles").fetchone()
                self.assertEqual(tuple(row), expected)
        for answer in ("seven", {"reason": "no number"}):
            self.database.add_profile(f"profile for {answer}")
            with self.subTest(answer=answer), self.assertLogs("indefinite_articles.ranker", "WARNING"):
                self._score_all(answer)
                self.assertEqual(self.database.connection.execute("SELECT score FROM articles").fetchone()[0], 4.0)

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
