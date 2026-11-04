import unittest
from email.message import EmailMessage
from unittest import mock

from indefinite_articles import replies
from indefinite_articles.llm import LLM, LLMError

ITEMS = [{"title": "Rust", "source": "Blog"}, {"title": "Gossip", "source": "Tabloid"},
         {"title": "Gardens", "source": "Weekly"}]


def stub_llm(answer: object = None, error: Exception | None = None):
    llm = mock.Mock(spec=LLM)
    if error is not None:
        llm.chat_json.side_effect = error
    else:
        llm.chat_json.return_value = answer
    return llm


class BodyTextTests(unittest.TestCase):
    def test_plain_part_preferred(self):
        message = EmailMessage()
        message.set_content("+1 -2\n")
        message.add_alternative("<p>ignored</p>", subtype="html")
        self.assertEqual(replies.body_text(message), "+1 -2\n")

    def test_html_only_drops_quotes_and_keeps_lines(self):
        message = EmailMessage()
        message.set_content("<div>1 like</div><div>2&nbsp;no<br>thanks</div>"
                            "<blockquote>3 like (quoted)</blockquote>", subtype="html")
        lines = [line for line in replies.body_text(message).splitlines() if line]
        self.assertEqual(lines, ["1 like", "2 no", "thanks"])

    def test_no_text_part(self):
        message = EmailMessage()
        message.set_content(b"\x00\x01", maintype="application", subtype="octet-stream")
        self.assertEqual(replies.body_text(message), "")


class StripQuotedTests(unittest.TestCase):
    def test_on_line_without_wrote_is_kept(self):
        self.assertEqual(replies.strip_quoted("On reflection +1\nmore please"), "On reflection +1\nmore please")

    def test_outlook_and_signature_headers(self):
        self.assertEqual(replies.strip_quoted("+3\r\n-----Original Message-----\r\n1. x"), "+3")
        self.assertEqual(replies.strip_quoted("+3\nSent from my phone\n"), "+3")
        self.assertEqual(replies.strip_quoted("+3\nFrom: Bot <bot@x>\n"), "+3")
        self.assertEqual(replies.strip_quoted("+3\ndigest-id: rec:abcd1234"), "+3")


class StructuredReplyTests(unittest.TestCase):
    def test_compact_ignores_out_of_range(self):
        llm = stub_llm({})
        self.assertEqual(replies.understand(llm, "+1 -9; 2+", ITEMS),
                         replies.Interpretation({1: replies.Rating(1), 2: replies.Rating(1)}, []))
        llm.chat_json.assert_not_called()  # nothing unresolved, no general comment

    def test_item_line_formats(self):
        ratings = replies.understand(None, "#1) 👍 lovely\n2. nope.\n3, boring!", ITEMS).ratings
        self.assertEqual(ratings, {1: replies.Rating(1, "lovely"), 2: replies.Rating(-1), 3: replies.Rating(-1)})


class LLMInterpretationTests(unittest.TestCase):
    def test_ratings_and_general_comment(self):
        llm = stub_llm({"ratings": [{"item": "1", "rating": "Like", "comment": "x" * 600},
                                    {"item": 2, "rating": "dislike"},
                                    {"item": 3, "rating": "unsure", "comment": None},
                                    {"item": 9, "rating": "like"}, {"rating": "like"}, {"item": "two"}],
                        "general_comment": "  more science  "})
        interpretation = replies.understand(llm, "free text", ITEMS)
        ratings, general = interpretation.ratings, interpretation.general
        self.assertEqual(ratings[1], replies.Rating(1, "x" * 500))
        self.assertEqual(ratings[2], replies.Rating(-1))
        self.assertEqual(ratings[3], replies.Rating(0))
        self.assertEqual(sorted(ratings), [1, 2, 3])
        self.assertEqual(general, ["more science"])
        prompt = llm.chat_json.call_args[0][1]
        self.assertIn("2. Gossip (Tabloid)", prompt)
        self.assertIn("free text", prompt)

    def test_empty_answer(self):
        self.assertEqual(replies.understand(stub_llm({"ratings": None}), "x", ITEMS),
                         replies.Interpretation({}, []))

    def test_non_object_answer_falls_back(self):
        with self.assertLogs("indefinite_articles.replies", "WARNING"):
            interpretation = replies.understand(stub_llm(["not", "a", "dict"]), "+1\nx", ITEMS)
        self.assertEqual(interpretation, replies.Interpretation({1: replies.Rating(1)}, ["x"]))


class UnderstandTests(unittest.TestCase):
    def test_structured_reply_needs_no_llm(self):
        llm = stub_llm({})
        self.assertEqual(replies.understand(llm, "+1 -3", ITEMS),
                         replies.Interpretation({1: replies.Rating(1), 3: replies.Rating(-1)}, []))
        llm.chat_json.assert_not_called()

    def test_without_llm_keeps_structured_part(self):
        interpretation = replies.understand(None, "1 hmm\n+2\nlove the variety", ITEMS)
        ratings, general = interpretation.ratings, interpretation.general
        self.assertEqual(ratings, {1: replies.Rating(0, "hmm"), 2: replies.Rating(1)})
        self.assertEqual(general, ["love the variety"])

    def test_llm_fills_gaps_but_explicit_ratings_win(self):
        llm = stub_llm({"ratings": [{"item": 1, "rating": "like", "comment": "hmm, actually yes"},
                                    {"item": 2, "rating": "dislike"},
                                    {"item": 3, "rating": "like"}],
                        "general_comment": "variety is good"})
        interpretation = replies.understand(llm, "1 hmm\n+2\nthe garden one was great", ITEMS)
        ratings, general = interpretation.ratings, interpretation.general
        self.assertEqual(ratings, {1: replies.Rating(1, "hmm, actually yes"), 2: replies.Rating(1), 3: replies.Rating(1)})
        self.assertEqual(general, ["variety is good"])

    def test_llm_failure_falls_back(self):
        with self.assertLogs("indefinite_articles.replies", "WARNING"):
            interpretation = replies.understand(stub_llm(error=LLMError("down")), "+1\nnice list", ITEMS)
            ratings, general = interpretation.ratings, interpretation.general
        self.assertEqual(ratings, {1: replies.Rating(1)})
        self.assertEqual(general, ["nice list"])


if __name__ == "__main__":
    unittest.main()
