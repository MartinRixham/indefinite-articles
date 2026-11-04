import contextlib
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

from fake_llm import FakeLLM

from indefinite_articles.config import Config
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.language_model.assistant import MAX_PHRASES, ArticleScore, Assistant
from indefinite_articles.language_model.llm import LLM, LLMError, LLMUnavailable
from indefinite_articles.mail.interpretation import Interpretation, Rating
from indefinite_articles.storage.article import Article
from indefinite_articles.storage.database import Database, PendingFeedback

ITEMS = [Article(1, "https://e/1", "Rust", source="Blog"), Article(2, "https://e/2", "Gossip", source="Tabloid"),
         Article(3, "https://e/3", "Gardens", source="Weekly")]


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        config = Config()
        config.llm.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.assistant = Assistant(LLM(config.llm))
        FakeLLM.requests = []

    def _answering(self, answer: str) -> contextlib.AbstractContextManager[object]:
        """While active, the fake LLM gives `answer` to every chat request."""
        return mock.patch.object(FakeLLM, "answer", answer)

    def _last_prompt(self) -> str:
        """The user message of the last request the fake LLM received."""
        return FakeLLM.requests[-1]["messages"][1]["content"]

    def test_score_prefix_is_shared_and_comes_first(self) -> None:
        prefix = Assistant.score_prefix("Likes Rust", "Liked: compilers")
        self.assertTrue(prefix.startswith("Reader profile:\nLikes Rust\n\nLiked: compilers\n\n"))
        self.assertIn('"score"', prefix)
        self.assertNotIn("Article to rate:", prefix)

    def test_score_article(self) -> None:
        prefix = Assistant.score_prefix("Likes Rust", "(none yet)")
        article = Article(1, "https://e/rust", "Rust lifetimes", summary="one two three four five", source="Blog",
                          category="Tech")
        self.assertEqual(self.assistant.score_article(prefix, article, per_article=20), ArticleScore(9.0, "test"))
        prompt = self._last_prompt()
        self.assertTrue(prompt.startswith(prefix + "Article to rate:\n"))  # KV cache can be reused
        self.assertTrue(prompt.endswith("Title: Rust lifetimes\nSource: Blog (Tech)\none two three four…"))
        self.assertEqual(self.assistant.score_article(prefix, ITEMS[1]), ArticleScore(2.0, "test"))

    def test_score_article_answer_shapes(self) -> None:
        for answer, expected in (('{"ratings": [{"score": "6", "reason": "r"}]}', ArticleScore(6.0, "r")),
                                 ('[{"score": -3}]', ArticleScore(0.0, "")),
                                 ('{"score": 12, "reason": "' + "x" * 300 + '"}', ArticleScore(10.0, "x" * 200))):
            with self.subTest(answer=answer), self._answering(answer):
                self.assertEqual(self.assistant.score_article("", ITEMS[0]), expected)
        for answer in ('"seven"', '{"reason": "no number"}', '{"score": "high"}', "not JSON"):
            with self.subTest(answer=answer), self._answering(answer), self.assertRaises(LLMError):
                self.assistant.score_article("", ITEMS[0])

    def test_rewrite_profile(self) -> None:
        database = Database(":memory:")
        self.addCleanup(database.close)
        for title, rating, comment in (("Liked one", 1, "more please"), ("Disliked one", -1, ""),
                                       ("Noted one", 0, "hmm")):
            database.add_article(Entry(f"https://e/{title}", title, "", None), "S")
            article_id = database.connection.execute("SELECT id FROM articles WHERE title=?", (title,)).fetchone()[0]
            database.add_feedback(article_id, rating, comment)
        database.add_general_comment(None, "less news")
        database.connection.commit()
        text = self.assistant.rewrite_profile("Likes everything", database.feedback_since(0))
        self.assertEqual(text, "- Likes Rust and systems programming\n- Dislikes gossip")
        prompt = self._last_prompt()
        self.assertTrue(prompt.startswith("Current reader profile:\nLikes everything\n\n"))
        self.assertIn('Liked:\n- Liked one (S) — "more please"\n\n', prompt)
        self.assertIn("Disliked:\n- Disliked one (S)\n\n", prompt)
        self.assertIn('like/dislike):\n- Noted one (S) — "hmm"\n\n', prompt)
        self.assertIn('General comments from the reader:\n- "less news"\n\n', prompt)

    def test_rewrite_profile_without_feedback_and_short_answer(self) -> None:
        with self._answering("  - Likes careful explanations of things\n"):
            self.assertEqual(self.assistant.rewrite_profile("Old", PendingFeedback([], [])),
                             "- Likes careful explanations of things")
        self.assertIn("Liked:\n(none)\n\n", self._last_prompt())
        self.assertIn("General comments from the reader:\n(none)\n\n", self._last_prompt())
        with self._answering("too short"), self.assertRaises(LLMError):
            self.assistant.rewrite_profile("Old", PendingFeedback([], []))

    def test_interpret_reply(self) -> None:
        interpretation = self.assistant.interpret_reply("the second one was boring", ITEMS)
        self.assertEqual(interpretation, Interpretation({2: Rating(-1, "boring")}, ["less gossip"]))
        prompt = self._last_prompt()
        self.assertIn("1. Rust (Blog)\n2. Gossip (Tabloid)\n3. Gardens (Weekly)\n", prompt)
        self.assertIn('"""\nthe second one was boring\n"""', prompt)
        # The fake rates item 2, which this digest doesn't have.
        self.assertEqual(self.assistant.interpret_reply("x", ITEMS[:1]), Interpretation({}, ["less gossip"]))

    def test_interpret_reply_answer_shapes(self) -> None:
        answer = ('{"ratings": [{"item": "1", "rating": "LIKE", "comment": "' + "x" * 600 + '"}, '
                  '{"item": 3, "rating": "meh"}, {"item": "two"}, {"rating": "like"}, "junk"], '
                  '"general_comment": "  "}')
        with self._answering(answer):
            self.assertEqual(self.assistant.interpret_reply("x", ITEMS),
                             Interpretation({1: Rating(1, "x" * 500), 3: Rating(0)}, []))
        with self._answering('{"ratings": null}'):
            self.assertEqual(self.assistant.interpret_reply("x", ITEMS), Interpretation({}, []))
        with self._answering('["not", "a", "dict"]'), self.assertRaises(LLMError):
            self.assistant.interpret_reply("x", ITEMS)

    def test_profile_phrases(self) -> None:
        self.assertEqual(self.assistant.profile_phrases("Likes Rust. Hates gossip."),
                         (["Rust and systems programming"], ["celebrity gossip"]))
        self.assertIn("Reader profile:\nLikes Rust. Hates gossip.\n\n", self._last_prompt())
        many = ", ".join(f'"  phrase {index}  "' for index in range(20))
        with self._answering('{"likes": [' + many + ', ""], "avoid": "not a list"}'):
            likes, avoid = self.assistant.profile_phrases("x")
        self.assertEqual(likes, [f"phrase {index}" for index in range(MAX_PHRASES)])
        self.assertEqual(avoid, [])
        for answer in ('{"likes": [], "avoid": ["gossip"]}', '{"oops": 1}', '["a list"]', "not JSON"):
            with self.subTest(answer=answer), self._answering(answer), self.assertRaises(LLMError):
                self.assistant.profile_phrases("x")

    def test_unavailable_llm(self) -> None:
        config = Config()
        config.llm.base_url = "http://127.0.0.1:9/v1"  # nothing listens on the discard port
        assistant = Assistant(LLM(config.llm))
        with self.assertRaises(LLMUnavailable):
            assistant.score_article("", ITEMS[0])


if __name__ == "__main__":
    unittest.main()
