import random
import unittest
from collections.abc import Sequence

import numpy as np

from indefinite_articles.config import Config
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.ranking import ranker
from indefinite_articles.ranking.classifier import Classifier
from indefinite_articles.ranking.rating_queue import CLOSE, RANDOM, UNSURE, rating_queue
from indefinite_articles.ranking.vectors import pack
from indefinite_articles.storage.database import Database


class RatingQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.database = Config(), Database(":memory:")
        self.addCleanup(self.database.close)
        self.config.max_candidates = 2
        self.config.podcasts_per_digest = 0

    def _article(self, title: str, source: str = "S", prefilter: float | None = None,
                 vector: Sequence[float] = (1.0, 0.0)) -> int:
        self.database.add_article(Entry(f"https://e/{title}", title, "", None), source)
        article_id = self.database.connection.execute("SELECT id FROM articles WHERE title=?", (title,)).fetchone()[0]
        self.database.set_embedding(article_id, pack(list(vector)))
        if prefilter is not None:
            self.database.set_prefilter({article_id: prefilter}, 1)
        self.database.connection.commit()
        return article_id

    def _train(self) -> None:
        # Likes point along the first axis and dislikes along the second.
        Classifier(np.array([4.0, -4.0], dtype=np.float32), 0.0).save(self.database)

    def test_takes_turns_and_shows_each_article_once(self) -> None:
        self._train()
        for index in range(8):
            self._article(f"A{index}", source=f"S{index % 3}", prefilter=index / 10,
                          vector=(1.0, index / 4))
        items = list(rating_queue(self.config, self.database, random.Random(1)))
        self.assertEqual([item.kind for item in items[:3]], [CLOSE, UNSURE, RANDOM])
        ids = [item.article.id for item in items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {article.id for article in self.database.pool(3)})

    def test_close_matches_are_best_first_and_skip_the_shortlist(self) -> None:
        for index in range(6):
            self._article(f"A{index}", source=f"S{index}", prefilter=index / 10)
        shortlisted = {article.id for article in ranker.shortlist(self.config, self.database)}
        close = [item.article for item in rating_queue(self.config, self.database, random.Random(1))
                 if item.kind == CLOSE]
        self.assertTrue(close)
        self.assertFalse(shortlisted & {article.id for article in close})
        scores = [article.prefilter or 0.0 for article in close]
        self.assertEqual(scores, sorted(scores, reverse=True))
        best_left = max(article.prefilter or 0.0 for article in self.database.pool(3)
                        if article.id not in shortlisted)
        self.assertEqual(close[0].prefilter, best_left)

    def test_unsure_picks_need_a_classifier_and_come_nearest_even_first(self) -> None:
        self._article("liked", vector=(1.0, 0.0))
        undecided = self._article("undecided", vector=(1.0, 1.0))
        self._article("disliked", vector=(0.0, 1.0))
        kinds = {item.kind for item in rating_queue(self.config, self.database, random.Random(1))}
        self.assertNotIn(UNSURE, kinds)
        self._train()
        unsure = [item.article.id for item in rating_queue(self.config, self.database, random.Random(1))
                  if item.kind == UNSURE]
        self.assertEqual(unsure[0], undecided)

    def test_random_picks_choose_the_feed_first(self) -> None:
        for index in range(20):
            self._article(f"busy {index}", source="Busy")
        quiet = self._article("quiet", source="Quiet")
        items = list(rating_queue(self.config, self.database, random.Random(3)))
        self.assertEqual({item.kind for item in items}, {RANDOM})
        self.assertIn(quiet, [item.article.id for item in items[:4]])
