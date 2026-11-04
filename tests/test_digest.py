import unittest
from datetime import date

from indefinite_articles import inbox, mailer
from indefinite_articles.article import Article
from indefinite_articles.config import Config
from indefinite_articles.digest import TEMPLATES, TOKEN_PREFIX, Digest
from indefinite_articles.ranker import DigestPick


def make_items() -> list[DigestPick]:
    return [
        DigestPick(Article(1, "https://ex.com/1?a=1&b=2", "Rust & <b>", summary="A summary.", source="Blog",
                           reason="fits"), False),
        DigestPick(Article(2, "https://ex.com/2", "Other", source="News"), True),
    ]


class DigestDefaultsTests(unittest.TestCase):
    def test_token_is_random_hex_that_replies_can_find(self) -> None:
        tokens = {Digest([]).token for _ in range(20)}
        self.assertEqual(len(tokens), 20)
        self.assertTrue(all(len(token) == 8 and int(token, 16) >= 0 for token in tokens))
        self.assertTrue(all(inbox.TOKEN_RE.fullmatch(TOKEN_PREFIX + token) for token in tokens))

    def test_day_is_today(self) -> None:
        self.assertEqual(Digest([]).day, date.today())

    def test_note_and_debug_are_empty(self) -> None:
        digest = Digest(make_items())
        self.assertEqual(digest.note, "")
        self.assertIsNone(digest.debug)


class DigestPropertyTests(unittest.TestCase):
    def test_digest_id(self) -> None:
        self.assertEqual(Digest([], token="abcd1234").digest_id, "rec:abcd1234")

    def test_subject(self) -> None:
        digest = Digest([], token="abcd1234", day=date(2026, 11, 4))
        self.assertEqual(digest.subject, "Your reading list for Wednesday 04 November 2026 [rec:abcd1234]")


class DigestTextBodyTests(unittest.TestCase):
    def test_items(self) -> None:
        text = Digest(make_items(), token="abcd1234").text_body()
        self.assertEqual(text.split("\n--\n")[0],
                         "Today's 2 recommendations\n\n"
                         "1. Rust & <b>\n   Blog\n   https://ex.com/1?a=1&b=2\n   Why: fits\n   A summary.\n\n"
                         "2. Other\n   News  · something different\n   https://ex.com/2\n")

    def test_footer_has_instructions_and_digest_id(self) -> None:
        text = Digest(make_items(), token="abcd1234").text_body()
        self.assertIn((TEMPLATES / "instructions.txt").read_text().strip(), text)
        self.assertTrue(text.endswith("digest-id: rec:abcd1234"))

    def test_no_runs_of_blank_lines(self) -> None:
        self.assertNotIn("\n\n\n", Digest(make_items()).text_body())

    def test_note(self) -> None:
        self.assertIn("Heads <up>", Digest(make_items(), note="Heads <up>").text_body())

    def test_empty_reason_left_out_and_long_blurb_shortened(self) -> None:
        items = [DigestPick(Article(1, "https://e/1", "T", source="S", reason="", content="word " * 100), False)]
        text = Digest(items).text_body()
        self.assertNotIn("Why:", text)
        self.assertIn("…", text)

    def test_debug(self) -> None:
        text = Digest(make_items(), debug=["line one", "line <two>"]).text_body()
        self.assertIn("-- debug --\nline one\nline <two>", text)

    def test_no_debug_section_without_debug_lines(self) -> None:
        self.assertNotIn("-- debug --", Digest(make_items()).text_body())
        self.assertNotIn("-- debug --", Digest(make_items(), debug=[]).text_body())


class DigestHtmlBodyTests(unittest.TestCase):
    def test_escaped_and_optional_lines_dropped(self) -> None:
        body = Digest(make_items(), note="Heads <up>", token="abcd1234").html_body()
        self.assertIn("Rust &amp; &lt;b&gt;", body)
        self.assertIn('href="https://ex.com/1?a=1&amp;b=2"', body)
        self.assertIn("Heads &lt;up&gt;", body)
        self.assertIn("rec:abcd1234", body)
        self.assertEqual(body.count("Why:"), 1)
        self.assertEqual(body.count("something different"), 1)
        self.assertNotIn("$", body)

    def test_debug_escaped(self) -> None:
        body = Digest(make_items(), debug=["line <two>"]).html_body()
        self.assertIn("debug</summary>", body)
        self.assertIn("line &lt;two&gt;", body)

    def test_no_debug_section_without_debug_lines(self) -> None:
        self.assertNotIn("debug</summary>", Digest(make_items()).html_body())


class DigestMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        self.config.email.sender, self.config.email.to = "Bot <bot@example.org>", "you@example.com"

    def test_headers(self) -> None:
        digest = Digest(make_items(), token="0123abcd")
        message = digest.message(self.config)
        self.assertEqual(message["Subject"], digest.subject)
        self.assertEqual(message["X-Indefinite-Articles-Digest"], "0123abcd")
        self.assertEqual(message["From"], "Bot <bot@example.org>")
        self.assertEqual(message["To"], "you@example.com")
        self.assertTrue(message["Message-ID"])

    def test_bodies(self) -> None:
        digest = Digest(make_items(), note="Heads <up>", debug=["line <two>"])
        message = digest.message(self.config)
        self.assertEqual(mailer.part_text(message).rstrip("\n"), digest.text_body())
        self.assertEqual(mailer.part_text(message, "html").rstrip("\n"), digest.html_body())


if __name__ == "__main__":
    unittest.main()
