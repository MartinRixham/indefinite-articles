"""The reader's like/dislike classifier over article embeddings."""

import base64

import numpy as np


class Classifier:
    """Logistic-regression weights and bias for uncentred article vectors."""

    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias

    def predict(self, vectors: list[bytes]) -> list[float]:
        """Probability that the reader likes each article (vectors as packed float32 bytes)."""
        if not vectors:
            return []
        matrix = np.frombuffer(b"".join(vectors), dtype=np.float32).reshape(len(vectors), -1)
        logits = np.clip(matrix.astype(np.float64) @ np.asarray(self.weights) + self.bias, -30, 30)
        return (1.0 / (1.0 + np.exp(-logits))).tolist()

    def save(self, database):
        """Store in the state table, as float32 weights."""
        weights = np.asarray(self.weights, dtype=np.float32).tobytes()
        database.set_state("classifier", base64.b64encode(weights).decode())
        database.set_state("classifier_bias", repr(self.bias))

    @classmethod
    def load(cls, database) -> "Classifier | None":
        """The classifier saved by `save`, or None if there is none."""
        encoded = database.get_state("classifier", "")
        if not encoded:
            return None
        return cls(np.frombuffer(base64.b64decode(encoded), dtype=np.float32),
                   float(database.get_state("classifier_bias", "0")))
