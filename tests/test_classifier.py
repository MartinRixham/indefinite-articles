import unittest

import numpy as np

from indefinite_articles import embed
from indefinite_articles.classifier import Classifier
from indefinite_articles.database import Database


def unit(*values: float) -> bytes:
    return embed.pack(list(values))


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)

    def test_predict_nothing(self):
        self.assertEqual(Classifier(np.array([1.0, 0.0]), 0.0).predict([]), [])

    def test_predict_follows_weights_and_bias(self):
        vectors = [unit(1, 0), unit(0, 1)]
        liked, disliked = Classifier(np.array([10.0, -10.0]), 0.0).predict(vectors)
        self.assertGreater(liked, 0.99)
        self.assertLess(disliked, 0.01)
        self.assertEqual(Classifier(np.array([0.0, 0.0]), 0.0).predict(vectors), [0.5, 0.5])
        biased = Classifier(np.array([0.0, 0.0]), 2.0).predict(vectors)
        self.assertAlmostEqual(biased[0], 1 / (1 + np.exp(-2.0)))

    def test_extreme_logits_are_clipped(self):
        probability = Classifier(np.array([1e6, 0.0]), 0.0).predict([unit(1, 0)])[0]
        self.assertTrue(np.isfinite(probability))
        self.assertLessEqual(probability, 1.0)

    def test_load_without_saved_classifier(self):
        self.assertIsNone(Classifier.load(self.database))
        self.database.set_state("classifier", "")  # what update_prefilter stores when untrained
        self.assertIsNone(Classifier.load(self.database))

    def test_save_and_load_round_trip(self):
        vectors = [unit(1, 0), unit(0.6, 0.8), unit(0, 1)]
        original = Classifier(np.array([3.0, -1.5]), 0.25)
        original.save(self.database)
        loaded = Classifier.load(self.database)
        assert loaded is not None
        self.assertEqual(loaded.weights.dtype, np.float32)
        np.testing.assert_allclose(loaded.weights, [3.0, -1.5])
        self.assertEqual(loaded.bias, 0.25)
        np.testing.assert_allclose(loaded.predict(vectors), original.predict(vectors), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
