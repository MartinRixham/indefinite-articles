import unittest

from indefinite_articles.fetching.text_extractor import TextExtractor


class TextExtractorTests(unittest.TestCase):
    def test_strips_tags_scripts_and_entities(self) -> None:
        self.assertEqual(TextExtractor.extract("<p>One&amp;two</p><script>evil()</script><div>three</div>"),
                         "One&two three")

    def test_empty(self) -> None:
        self.assertEqual(TextExtractor.extract(""), "")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(TextExtractor.extract("  a\n\n\tb  "), "a b")
