import random
import threading
import time
import unittest
from collections.abc import Sequence
from unittest import mock

import numpy as np

from indefinite_articles.config import Config
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.fetching.extract import ArticleContent
from indefinite_articles.language_model.assistant import MAX_PROFILE_CHARS
from indefinite_articles.language_model.llm import LLM, LLMError, LLMUnavailable
from indefinite_articles.ranking import embed, ranker
from indefinite_articles.ranking.classifier import Classifier
from indefinite_articles.storage.article import EMBEDDING_SUMMARY_CHARACTERS
from indefinite_articles.storage.database import Database


def unit(*values: float) -> bytes:
    return embed.pack(list(values))


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
        self.llm.embed.side_effect = lambda texts, purpose="texts": [self.vectors[text] for text in texts]

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
        """Run the pre-filter; returns each unsent article's score by title (unscored ones are left out)."""
        embed.update_prefilter(self.config, self.database, self.llm, ranker.ensure_profile(self.config, self.database))
        return {row.title: row.prefilter for row in self.database.pool(3) if row.prefilter is not None}

    def test_embedded_text_is_title_and_start_of_summary(self):
        self.database.add_article(Entry("https://e/1", "Short", "", None), "S")
        self.database.add_article(Entry("https://e/2", "Long", "x" * (EMBEDDING_SUMMARY_CHARACTERS + 100), None), "S")
        self.llm.embed.side_effect = lambda texts, purpose="texts": [[1.0, 0.0] for _ in texts]
        self.assertEqual(embed.embed_new(self.config, self.database, self.llm), 2)
        texts = [text for call in self.llm.embed.call_args_list for text in call.args[0]]
        self.assertEqual(sorted(texts), ["Long. " + "x" * EMBEDDING_SUMMARY_CHARACTERS, "Short. "])

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
        self.assertEqual(database.get_state("embedding_setup"), embed._embedding_setup(config))

    def test_new_embedding_setup_embeds_again(self) -> None:
        self.llm.embed.side_effect = lambda texts, purpose="texts": [[1.0, 0.0] for _ in texts]
        self._article("old rated", [1, 0], rating=1)
        self.database.connection.execute("UPDATE articles SET published=0, fetched_at=0")  # too old to embed now
        self._article("recent", [1, 0])
        embed.embed_new(self.config, self.database, self.llm)  # first run: no setup recorded yet
        self.llm.embed.reset_mock()
        self.assertEqual(embed.embed_new(self.config, self.database, self.llm), 0)  # same setup: nothing to do
        self.config.llm.embedding_prefix = "something else: "
        self.assertEqual(embed.embed_new(self.config, self.database, self.llm), 2)
        self.assertEqual(sorted(self.llm.embed.call_args.args[0]), ["old rated. ", "recent. "])

    def test_dislike_only_profile_still_wants_something(self):
        config, database = Config(interests="Dislikes celebrity gossip"), Database(":memory:")
        self.addCleanup(database.close)
        llm = mock.Mock(spec=LLM)
        llm.chat_json.return_value = {"likes": [], "avoid": ["celebrity gossip"]}  # unusable: keyword split
        llm.embed.side_effect = lambda texts, purpose="texts": [[1.0, float(index)] for index, _ in enumerate(texts)]
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
            with self.subTest(answer=answer), self.assertLogs("indefinite_articles.ranking.ranker", "WARNING"):
                self._score_all(answer)
                self.assertEqual(self.database.connection.execute("SELECT score FROM articles").fetchone()[0], 4.0)

    def test_shortlist_without_prefilter_uses_newest(self):
        for index in range(5):
            self._add(f"https://e/{index}")
        self.config.max_candidates = 3
        self.assertEqual(len(ranker.shortlist(self.config, self.database)), 3)

    def _add_episode(self, url: str, source: str = "Pod", score: float | None = None,
                     profile_id: int | None = None, prefilter: float | None = None) -> None:
        self.database.add_article(Entry(url, url, "show notes", time.time(), audio_url=url + ".mp3"), source)
        self.database.connection.execute(
            "UPDATE articles SET score=?, score_profile_id=?, prefilter=? WHERE url=?",
            (score, profile_id, prefilter, url))
        self.database.connection.commit()

    def test_podcast_episodes_have_their_own_shortlist(self) -> None:
        for index in range(5):
            self._add(f"https://e/{index}", prefilter=0.9)
            self._add_episode(f"https://pod/{index}", prefilter=0.1)
        self.config.max_candidates, self.config.max_podcast_candidates = 3, 2
        short = ranker.shortlist(self.config, self.database)
        self.assertEqual([article.is_podcast for article in short], [False] * 3 + [True] * 2)
        self.config.podcasts_per_digest = 0  # no room in the digest: no LLM time spent on them
        self.assertFalse(any(article.is_podcast for article in ranker.shortlist(self.config, self.database)))

    def test_podcast_shortlist_without_prefilter_uses_newest(self) -> None:
        for index in range(3):
            self._add(f"https://e/{index}")
            self._add_episode(f"https://pod/{index}")
        self.config.max_candidates, self.config.max_podcast_candidates = 2, 1
        short = ranker.shortlist(self.config, self.database)
        self.assertEqual([article.is_podcast for article in short], [False, False, True])

    def test_select_lists_podcast_episodes_after_articles(self) -> None:
        profile_id = ranker.ensure_profile(self.config, self.database)["id"]
        for index, score in enumerate((9.0, 8.0, 7.0, 6.0, 5.0)):
            self._add(f"https://a/{index}", f"A{index}", score, profile_id)
        for index, score in enumerate((9.5, 3.0, 2.0)):
            self._add_episode(f"https://pod/{index}", f"P{index}", score, profile_id)
        self.config.podcasts_per_digest, self.config.podcast_exploration = 2, 1
        chosen = ranker.select(self.config, self.database, random.Random(1))
        self.assertEqual([pick.article.is_podcast for pick in chosen], [False] * 4 + [True] * 2)
        self.assertEqual([pick.article.url for pick in chosen[:3]], ["https://a/0", "https://a/1", "https://a/2"])
        self.assertEqual((chosen[4].article.url, chosen[4].exploratory), ("https://pod/0", False))
        self.assertTrue(chosen[5].exploratory)
        self.config.podcasts_per_digest = 0
        chosen = ranker.select(self.config, self.database, random.Random(1))
        self.assertEqual(len(chosen), 4)
        self.assertFalse(any(pick.article.is_podcast for pick in chosen))

    def test_profile_change_rescores_top_podcast_episodes_too(self) -> None:
        old_profile = ranker.ensure_profile(self.config, self.database)["id"]
        for index in range(3):
            self._add(f"https://e/{index}", f"S{index}", 9.0, old_profile)
        self._add_episode("https://pod/0", "Pod", 2.0, old_profile)
        self.database.add_profile("new profile")
        self.config.rescore_top = 1
        llm = mock.Mock(spec=LLM)
        llm.chat_json.return_value = {"score": 5}
        self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 2)
        titles = sorted(call.args[1].split("Title: ")[1].split("\n")[0] for call in llm.chat_json.call_args_list)
        self.assertEqual(len(titles), 2)
        self.assertIn("https://pod/0", titles)

    def test_score_candidates(self):
        for index in range(3):
            self._add(f"https://e/{index}")
        llm = mock.Mock(spec=LLM)
        llm.chat_json.side_effect = [{"score": 8}, LLMError("garbled"), {"score": 3}]
        with self.assertLogs("indefinite_articles.ranking.ranker", "WARNING"):
            self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 2)
        # Only the failed article is left; an unavailable LLM stops the round.
        llm.chat_json.side_effect = LLMUnavailable("down")
        with self.assertLogs("indefinite_articles.ranking.ranker", "WARNING"):
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

    def test_profile_change_rescores_only_the_top(self):
        old_profile = ranker.ensure_profile(self.config, self.database)["id"]
        for index, score in enumerate((2.0, 9.0, 5.0, 7.0, 1.0)):
            self._add(f"https://e/{index}", f"S{index}", score, old_profile)
        self._add("https://e/new")
        self.database.add_profile("new profile")
        self.config.rescore_top = 2
        llm = mock.Mock(spec=LLM)
        llm.chat_json.return_value = {"score": 4}
        # The unscored article first, then the two best stale ones, best first.
        self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 3)
        titles = [call.args[1].split("Title: ")[1].split("\n")[0] for call in llm.chat_json.call_args_list]
        self.assertEqual(titles, ["https://e/new", "https://e/1", "https://e/3"])
        # Rescored to 4, they drop below the stale 5.0, which is now in the top 2.
        llm.chat_json.reset_mock()
        self.assertEqual(ranker.score_candidates(self.config, self.database, llm), 1)
        titles = [call.args[1].split("Title: ")[1].split("\n")[0] for call in llm.chat_json.call_args_list]
        self.assertEqual(titles, ["https://e/2"])
        rows = self.database.connection.execute("SELECT url, score FROM articles WHERE score_profile_id=?",
                                                (old_profile,)).fetchall()
        self.assertEqual(sorted(tuple(row) for row in rows), [("https://e/0", 2.0), ("https://e/4", 1.0)])

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
        self.assertEqual([(pick.article.url, pick.exploratory) for pick in chosen[:3]],
                         [("https://a/1", False), ("https://a/2", False), ("https://b/1", False)])
        self.assertEqual(len(chosen), 4)
        self.assertTrue(chosen[3].exploratory)
        self.assertIn(chosen[3].article.url, ("https://c/1", "https://d/1"))
        self.assertEqual((debug["candidates"], debug["llm_scored"], debug["stale_scored"]), (6, 6, 0))
        self.assertEqual(debug["rank"][chosen[0].article.id], 1)
        self.assertIsInstance(debug["scores"][chosen[0].article.id], ranker.ScoreBreakdown)

    def test_select_avoids_overlapping_articles(self) -> None:
        profile_id = ranker.ensure_profile(self.config, self.database)["id"]
        stories = [("https://a/1", 9.0, [1.0, 0.0, 0.0, 0.0]),
                   ("https://b/1", 8.5, [0.99, 0.1, 0.0, 0.0]),  # the same story as a/1
                   ("https://c/1", 7.0, [0.0, 1.0, 0.0, 0.0]),
                   ("https://d/1", 6.0, [0.0, 0.0, 1.0, 0.0]),
                   ("https://e/1", 1.0, [0.0, 0.0, 0.0, 1.0])]
        for url, score, vector in stories:
            self._add(url, url, score, profile_id)
            article_id = self.database.connection.execute("SELECT id FROM articles WHERE url=?", (url,)).fetchone()[0]
            self.database.set_embedding(article_id, embed.pack(vector))
        self._add("https://f/1", "F", 0.5, profile_id)  # no embedding: never penalised
        self.config.articles_per_digest, self.config.exploration = 3, 0
        debug: dict = {}
        chosen = ranker.select(self.config, self.database, random.Random(1), debug)
        self.assertEqual([pick.article.url for pick in chosen], ["https://a/1", "https://c/1", "https://d/1"])
        self.assertEqual(debug["overlap"][chosen[0].article.id], 0.0)
        self.config.diversity = 0.0
        chosen = ranker.select(self.config, self.database, random.Random(1))
        self.assertEqual([pick.article.url for pick in chosen], ["https://a/1", "https://b/1", "https://c/1"])

    def test_select_centres_episodes_on_their_own(self) -> None:
        # Episodes share a direction (dimension 2) that articles lack. Centred on the mean of
        # every candidate, these three episodes on different subjects would look alike.
        profile_id = ranker.ensure_profile(self.config, self.database)["id"]
        article_vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
        episode_vectors = [[0.4, 0.0, 1.0, 0.0], [0.0, 0.4, 1.0, 0.0], [0.0, 0.0, 1.0, 0.4]]
        for index, vector in enumerate(article_vectors):
            self._add(f"https://a{index}/1", f"A{index}", 9.0 - index, profile_id)
            self._set_vector(f"https://a{index}/1", vector)
        for index, vector in enumerate(episode_vectors):
            self._add_episode(f"https://pod{index}/1", f"Pod{index}", 9.0 - index, profile_id)
            self._set_vector(f"https://pod{index}/1", vector)
        self.config.articles_per_digest, self.config.exploration = 2, 0
        self.config.podcasts_per_digest, self.config.podcast_exploration = 2, 0
        debug: dict = {}
        chosen = ranker.select(self.config, self.database, random.Random(1), debug)
        episodes = [pick.article for pick in chosen if pick.article.is_podcast]
        self.assertEqual([episode.url for episode in episodes], ["https://pod0/1", "https://pod1/1"])
        self.assertEqual(debug["overlap"][episodes[1].id], 0.0)

    def _set_vector(self, url: str, vector: list[float]) -> None:
        article_id = self.database.connection.execute("SELECT id FROM articles WHERE url=?", (url,)).fetchone()[0]
        self.database.set_embedding(article_id, embed.pack(vector))

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
        self.assertEqual(chosen[0].article.url, "https://liked/new")

    def test_update_profile(self):
        llm = mock.Mock(spec=LLM)
        self.assertFalse(ranker.update_profile(self.config, self.database, llm))  # nothing pending
        llm.chat.assert_not_called()
        self._add("https://e/1", title="Liked one")
        self.database.add_feedback(1, 1, "more please")
        self.database.add_general_comment(None, "less news")
        self.database.connection.commit()

        llm.chat.side_effect = LLMError("down")
        with self.assertLogs("indefinite_articles.ranking.ranker", "WARNING"):
            self.assertFalse(ranker.update_profile(self.config, self.database, llm))
        llm.chat.side_effect = None
        llm.chat.return_value = "too short"
        with self.assertLogs("indefinite_articles.ranking.ranker", "WARNING"):
            self.assertFalse(ranker.update_profile(self.config, self.database, llm))
        llm.chat.return_value = "  - Likes careful explanations of things\n"
        self.assertTrue(ranker.update_profile(self.config, self.database, llm))
        prompt = llm.chat.call_args[0][1]
        self.assertIn('- Liked one (S) — "more please"', prompt)
        self.assertIn('- "less news"', prompt)
        word_count = len((self.config.interests or ranker.DEFAULT_PROFILE).split())
        self.assertIn(f"The reader's profile before this feedback ({word_count} words):", prompt)
        profile = self.database.current_profile()
        assert profile is not None
        self.assertEqual(profile["text"], "- Likes careful explanations of things")
        self.assertEqual(profile["note"], "1 likes, 0 dislikes, 1 comments")

    def test_update_profile_drops_lines_from_the_top_when_too_long(self) -> None:
        llm = mock.Mock(spec=LLM)
        ranker.ensure_profile(self.config, self.database)
        self._add("https://e/1", title="Liked one")
        self.database.add_feedback(1, 1, "")
        self.database.connection.commit()
        lines = [f"- Old interest number {index:04d}" for index in range(400)] + ["- Newest interest"]
        llm.chat.return_value = "\n".join(lines)
        with self.assertLogs("indefinite_articles.ranking.ranker", "WARNING"):
            self.assertTrue(ranker.update_profile(self.config, self.database, llm))
        profile = self.database.current_profile()
        assert profile is not None
        text = profile["text"]
        self.assertLessEqual(len(text), MAX_PROFILE_CHARS)
        self.assertGreater(len(text), MAX_PROFILE_CHARS - len(lines[0]) - 1)
        self.assertTrue(text.endswith("- Newest interest"))
        self.assertTrue(text.startswith("- Old interest number "))  # whole lines only
        self.assertNotIn("number 0000", text)


if __name__ == "__main__":
    unittest.main()
