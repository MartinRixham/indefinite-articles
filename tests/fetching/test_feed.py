import unittest
import xml.etree.ElementTree as ElementTree

from indefinite_articles.fetching.feed import Feed


class FeedTests(unittest.TestCase):
    def test_from_table_defaults(self) -> None:
        feed = Feed.from_table({"url": "https://example.com/feed"})
        self.assertEqual((feed.name, feed.category, feed.max_items), ("", "", 40))

    def test_from_table_converts_max_items(self) -> None:
        feed = Feed.from_table({"url": "https://example.com/feed", "name": "Blog", "category": "Tech",
                                "max_items": "7"})
        self.assertEqual((feed.name, feed.category, feed.max_items), ("Blog", "Tech", 7))

    def test_host(self) -> None:
        self.assertEqual(Feed("https://Blog.Example.com:8443/rss?x=1").host, "blog.example.com")
        self.assertEqual(Feed("not a url").host, "")

    def test_from_outline(self) -> None:
        outline = ElementTree.fromstring('<outline text=" Blog " xmlUrl=" https://example.com/feed " maxItems="5"/>')
        feed = Feed.from_outline(outline, "Tech")
        self.assertEqual((feed.url, feed.name, feed.category, feed.max_items),
                         ("https://example.com/feed", "Blog", "Tech", 5))

    def test_from_outline_prefers_title_and_defaults_bad_max_items(self) -> None:
        outline = ElementTree.fromstring('<outline title="Title" text="Text" xmlUrl="https://a" maxItems="many"/>')
        feed = Feed.from_outline(outline, "")
        self.assertEqual((feed.name, feed.max_items), ("Title", 40))

    def test_from_line(self) -> None:
        feed = Feed.from_line("https://example.com/feed  Some Blog ", "Tech")
        self.assertEqual((feed.url, feed.name, feed.category), ("https://example.com/feed", "Some Blog", "Tech"))
        self.assertEqual(Feed.from_line("https://example.com/feed", "").name, "")

    def test_source_prefers_name_then_title_then_url(self) -> None:
        self.assertEqual(Feed("https://feed", name="Named").source("Title"), "Named")
        self.assertEqual(Feed("https://feed").source("Title"), "Title")
        self.assertEqual(Feed("https://feed").source(""), "https://feed")

    def test_repr(self) -> None:
        self.assertEqual(repr(Feed("https://example.com/feed", name="Blog")), "Feed('https://example.com/feed')")


if __name__ == "__main__":
    unittest.main()
