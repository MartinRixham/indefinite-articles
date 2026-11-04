"""The reader profile as short phrases, for comparing with article embeddings."""

from .llm import LLM
from .vectors import pack


class ProfilePhrases:
    """What the reader profile says the reader wants and doesn't want, as short phrases."""

    def __init__(self, wanted: list[str], unwanted: list[str], llm: LLM):
        self.wanted = wanted
        self.unwanted = unwanted
        self._llm = llm
        self._wanted_vectors: list[bytes] | None = None
        self._unwanted_vectors: list[bytes] | None = None

    def wanted_vectors(self) -> list[bytes]:
        """The wanted phrases embedded, as made by `pack`; embedded on first use."""
        if self._wanted_vectors is None:
            self._wanted_vectors = self._embed(self.wanted)
        return self._wanted_vectors

    def unwanted_vectors(self) -> list[bytes]:
        """The unwanted phrases embedded, as made by `pack`; embedded on first use."""
        if self._unwanted_vectors is None:
            self._unwanted_vectors = self._embed(self.unwanted)
        return self._unwanted_vectors

    def dimension(self) -> int:
        """Length in bytes of each packed vector; embeds the wanted phrases on first use."""
        return len(self.wanted_vectors()[0])

    def _embed(self, phrases: list[str]) -> list[bytes]:
        return [pack(vector) for vector in self._llm.embed(phrases)] if phrases else []
