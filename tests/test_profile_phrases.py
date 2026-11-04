import unittest
from unittest import mock

from indefinite_articles.llm import LLM
from indefinite_articles.profile_phrases import ProfilePhrases
from indefinite_articles.vectors import pack


class ProfilePhrasesTests(unittest.TestCase):
    def setUp(self):
        self.llm = mock.Mock(spec=LLM)
        self.llm.embed.side_effect = lambda texts: [[3.0, float(index)] for index, _ in enumerate(texts)]

    def test_keeps_phrases_without_embedding(self):
        phrases = ProfilePhrases(["rust compilers"], ["celebrity gossip"], self.llm)
        self.assertEqual(phrases.wanted, ["rust compilers"])
        self.assertEqual(phrases.unwanted, ["celebrity gossip"])
        self.llm.embed.assert_not_called()

    def test_wanted_vectors_are_packed_embeddings(self):
        phrases = ProfilePhrases(["rust compilers", "gardening"], ["celebrity gossip"], self.llm)
        self.assertEqual(phrases.wanted_vectors(), [pack([3.0, 0.0]), pack([3.0, 1.0])])
        self.llm.embed.assert_called_once_with(["rust compilers", "gardening"])

    def test_unwanted_vectors_are_packed_embeddings(self):
        phrases = ProfilePhrases(["rust compilers"], ["celebrity gossip"], self.llm)
        self.assertEqual(phrases.unwanted_vectors(), [pack([3.0, 0.0])])
        self.llm.embed.assert_called_once_with(["celebrity gossip"])

    def test_vectors_are_embedded_once(self):
        phrases = ProfilePhrases(["rust compilers"], ["celebrity gossip"], self.llm)
        phrases.wanted_vectors()
        phrases.wanted_vectors()
        phrases.unwanted_vectors()
        phrases.unwanted_vectors()
        self.assertEqual(self.llm.embed.call_count, 2)

    def test_no_phrases_needs_no_llm(self):
        phrases = ProfilePhrases([], [], self.llm)
        self.assertEqual(phrases.wanted_vectors(), [])
        self.assertEqual(phrases.unwanted_vectors(), [])
        self.llm.embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
