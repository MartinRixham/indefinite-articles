"""Embedding vectors as stored in the database."""

from collections.abc import Sequence

import numpy as np


def pack(vector: Sequence[float]) -> bytes:
    """Unit-length float32 bytes, as stored in the embeddings table."""
    array = np.asarray(vector, dtype=np.float32)
    return (array / (np.linalg.norm(array) or 1.0)).astype(np.float32).tobytes()


def unpack(vectors: Sequence[bytes]) -> np.ndarray:
    """Vectors made by `pack` as the rows of a float32 matrix; `vectors` must not be empty."""
    return np.frombuffer(b"".join(vectors), dtype=np.float32).reshape(len(vectors), -1)


def max_similarities(matrix: np.ndarray, references: Sequence[bytes]) -> np.ndarray:
    """For each row of `matrix`, its cosine similarity to the closest of `references` (0 if none)."""
    if not references:
        return np.zeros(len(matrix), dtype=np.float32)
    return (matrix @ unpack(references).T).max(axis=1)
