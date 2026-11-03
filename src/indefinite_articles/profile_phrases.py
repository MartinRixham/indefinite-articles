"""The reader profile as short phrases, for comparing with article embeddings."""

from .llm import LLM
from .vectors import pack


class ProfilePhrases:
    """What the reader profile says the reader wants and doesn't want, as short phrases."""

    def __init__(self, wanted: list[str], unwanted: list[str]):
        self.wanted = wanted
        self.unwanted = unwanted

    def wanted_vectors(self, llm: LLM) -> list[bytes]:
        """The wanted phrases embedded, as made by `pack`."""
        return self._embed(llm, self.wanted)

    def unwanted_vectors(self, llm: LLM) -> list[bytes]:
        """The unwanted phrases embedded, as made by `pack`."""
        return self._embed(llm, self.unwanted)

    @staticmethod
    def _embed(llm: LLM, phrases: list[str]) -> list[bytes]:
        return [pack(vector) for vector in llm.embed(phrases)] if phrases else []
