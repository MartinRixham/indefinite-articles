import unittest
from unittest import mock

from indefinite_articles.llm import LLM
from indefinite_articles.profile_phrases import ProfilePhrases
from indefinite_articles.vectors import pack


class ProfilePhrasesTests(unittest.TestCase):
    def setUp(self):
        self.llm = mock.Mock(spec=LLM)
        self.llm.embed.side_effect = lambda texts: [[3.0, float(index)] for index, _ in enumerate(texts)]

    def test_keeps_phrases(self):
        phrases = ProfilePhrases(["rust compilers"], ["celebrity gossip"])
        self.assertEqual(phrases.wanted, ["rust compilers"])
        self.assertEqual(phrases.unwanted, ["celebrity gossip"])

    def test_wanted_vectors_are_packed_embeddings(self):
        phrases = ProfilePhrases(["rust compilers", "gardening"], ["celebrity gossip"])
        self.assertEqual(phrases.wanted_vectors(self.llm), [pack([3.0, 0.0]), pack([3.0, 1.0])])
        self.llm.embed.assert_called_once_with(["rust compilers", "gardening"])

    def test_unwanted_vectors_are_packed_embeddings(self):
        phrases = ProfilePhrases(["rust compilers"], ["celebrity gossip"])
        self.assertEqual(phrases.unwanted_vectors(self.llm), [pack([3.0, 0.0])])
        self.llm.embed.assert_called_once_with(["celebrity gossip"])

    def test_no_phrases_needs_no_llm(self):
        phrases = ProfilePhrases([], [])
        self.assertEqual(phrases.wanted_vectors(self.llm), [])
        self.assertEqual(phrases.unwanted_vectors(self.llm), [])
        self.llm.embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
