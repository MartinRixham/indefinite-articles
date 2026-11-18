import random
import unittest

from indefinite_articles.config import Config
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.ranking.rating_queue import rating_queue
from indefinite_articles.storage.database import Database


class RatingQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.database = Config(), Database(":memory:")
        self.addCleanup(self.database.close)

    def _article(self, title: str, source: str = "S", prefilter: float | None = None) -> int:
        self.database.add_article(Entry(f"https://e/{title}", title, "", None), source)
        article_id = self.database.connection.execute("SELECT id FROM articles WHERE title=?", (title,)).fetchone()[0]
        if prefilter is not None:
            self.database.set_prefilter({article_id: prefilter}, 1)
        self.database.connection.commit()
        return article_id

    def test_shows_each_article_in_the_pool_once(self) -> None:
        for index in range(8):
            self._article(f"A{index}", source=f"S{index % 3}", prefilter=index / 10)
        ids = [article.id for article in rating_queue(self.config, self.database, random.Random(1))]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {article.id for article in self.database.pool(3)})

    def test_ignores_the_prefilter(self) -> None:
        for index in range(10):
            self._article(f"A{index}", source=f"S{index}", prefilter=index / 10)
        orders = {tuple(article.id for article in rating_queue(self.config, self.database, random.Random(seed)))
                  for seed in range(5)}
        best_first = tuple(sorted((article.id for article in self.database.pool(3)),
                                  key=lambda article_id: -article_id))
        self.assertGreater(len(orders), 1)
        self.assertNotIn(best_first, orders)

    def test_chooses_the_feed_first(self) -> None:
        for index in range(20):
            self._article(f"busy {index}", source="Busy")
        quiet = self._article("quiet", source="Quiet")
        articles = list(rating_queue(self.config, self.database, random.Random(3)))
        self.assertIn(quiet, [article.id for article in articles[:4]])
