"""Embedding vectors as stored in the database."""

from collections.abc import Sequence

import numpy as np


def pack(vector: Sequence[float]) -> bytes:
    """Unit-length float32 bytes, as stored in the embeddings table."""
    array = np.asarray(vector, dtype=np.float32)
    return (array / (np.linalg.norm(array) or 1.0)).astype(np.float32).tobytes()
