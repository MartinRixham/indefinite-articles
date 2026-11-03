import gzip
import threading
import time
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from indefinite_articles import feeds
from indefinite_articles.config import Config, Feed
from indefinite_articles.database import Database


def rss(*items: str, title: str = "Channel") -> bytes:
    return (f'<?xml version="1.0"?><rss version="2.0"><channel><title>{title}</title>'
            + "".join(items) + "</channel></rss>").encode()


def item(title: str, url: str, published: str = "") -> str:
    date = f"<pubDate>{published}</pubDate>" if published else ""
    return f"<item><title>{title}</title><link>{url}</link>{date}</item>"


class Server(BaseHTTPRequestHandler):
    """/plain, /gzip and /not-modified (304 whenever a validator is sent)."""
    body = rss(item("One", "https://example.com/1"))
    requests: list = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        Server.requests.append((self.path, dict(self.headers)))
        if self.path == "/always-304":
            self.send_response(304)
            self.end_headers()
            return
        if self.path == "/not-modified" and self.headers.get("If-Modified-Since"):
            self.send_response(304)
            self.end_headers()
            return
        body = gzip.compress(Server.body) if self.path == "/gzip" else Server.body
        self.send_response(200)
        if self.path == "/gzip":
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Last-Modified", "Mon, 02 Nov 2026 10:00:00 GMT")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HTMLToTextTests(unittest.TestCase):
    def test_strips_tags_scripts_and_entities(self):
        self.assertEqual(feeds.html_to_text("<p>One&amp;two</p><script>evil()</script><div>three</div>"),
                         "One&two three")

    def test_empty(self):
        self.assertEqual(feeds.html_to_text(""), "")

    def test_collapses_whitespace(self):
        self.assertEqual(feeds.html_to_text("  a\n\n\tb  "), "a b")


class TitleKeyTests(unittest.TestCase):
    def test_normalises(self):
        self.assertEqual(feeds.title_key("Rust 1.90: What's New?"), "rust190whatsnew")
        self.assertEqual(feeds.title_key("Rust 1.90 — what’s new"), feeds.title_key("RUST 1.90: WHATS NEW"))

    def test_length_capped(self):
        self.assertEqual(len(feeds.title_key("word " * 100)), 120)

    def test_entry_property(self):
        self.assertEqual(feeds.Entry("u", "Hello, World", "", None).title_key, "helloworld")


class ParseFeedTests(unittest.TestCase):
    def test_rss_1_rdf(self):
        data = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel><title>RDF feed</title></channel>
<item><title>First</title><link>https://example.com/first</link><dc:date>2026-11-02T08:00:00Z</dc:date></item>
</rdf:RDF>"""
        title, entries = feeds.parse_feed(data)
        self.assertEqual(title, "RDF feed")
        self.assertEqual([entry.url for entry in entries], ["https://example.com/first"])
        self.assertEqual(entries[0].published, datetime(2026, 11, 2, 8, tzinfo=timezone.utc).timestamp())

    def test_skips_entries_without_url_or_title(self):
        data = rss(item("", "https://example.com/untitled"),
                   '<item><title>No link</title><guid isPermaLink="false">id-1</guid></item>',
                   item("Kept", "https://example.com/kept"))
        self.assertEqual([entry.title for entry in feeds.parse_feed(data)[1]], ["Kept"])

    def test_long_summary_truncated_at_word(self):
        summary = "word " * 300
        data = rss(f"<item><title>T</title><link>https://e/1</link><description>{summary}</description></item>")
        result = feeds.parse_feed(data)[1][0].summary
        self.assertLessEqual(len(result), feeds.SUMMARY_CHARACTERS)
        self.assertTrue(result.endswith("word…"))

    def test_dates(self):
        data = rss(item("A", "https://e/a", "Mon, 02 Nov 2026 10:00:00 GMT"),
                   item("B", "https://e/b", "not a date"))
        first, second = feeds.parse_feed(data)[1]
        self.assertEqual(first.published, datetime(2026, 11, 2, 10, tzinfo=timezone.utc).timestamp())
        self.assertIsNone(second.published)

    def test_byte_order_mark_and_atom_content(self):
        data = (b'\xef\xbb\xbf<feed xmlns="http://www.w3.org/2005/Atom"><title>A</title>'
                b'<entry><title type="html">&lt;b&gt;Bold&lt;/b&gt;</title><link href="https://e/1"/>'
                b'<content type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml"><p>Body</p></div></content>'
                b'<published>2026-11-02T08:00:00+00:00</published></entry></feed>')
        title, entries = feeds.parse_feed(data)
        self.assertEqual((title, entries[0].title, entries[0].summary), ("A", "Bold", "Body"))

    def test_unparseable(self):
        with self.assertRaises(Exception):
            feeds.parse_feed(b"this is not xml at all")


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Server.requests.clear()

    def test_fetch_conditional_sends_validators(self):
        data, metadata = feeds.fetch_conditional(self.base + "/not-modified")
        self.assertEqual(data, Server.body)
        self.assertEqual(metadata["last_modified"], "Mon, 02 Nov 2026 10:00:00 GMT")
        data, metadata = feeds.fetch_conditional(self.base + "/not-modified", '"tag"', "yesterday")
        self.assertIsNone(data)
        self.assertEqual(metadata, {"etag": '"tag"', "last_modified": "yesterday"})
        headers = Server.requests[-1][1]
        self.assertEqual((headers["If-None-Match"], headers["If-Modified-Since"]), ('"tag"', "yesterday"))
        self.assertTrue(headers["User-Agent"].startswith("indefinite-articles/"))

    def test_fetch_decompresses_gzip(self):
        self.assertEqual(feeds.fetch(self.base + "/gzip"), Server.body)

    def test_fetch_rejects_unexpected_not_modified(self):
        with self.assertRaisesRegex(ValueError, "304"):
            feeds.fetch(self.base + "/always-304")

    def test_update_all_keeps_newest_items_and_names_source(self):
        original = Server.body
        Server.body = rss(item("Old", "https://e/old", "Mon, 02 Nov 2026 08:00:00 GMT"),
                          item("New", "https://e/new", "Mon, 02 Nov 2026 10:00:00 GMT"),
                          item("Middle", "https://e/middle", "Mon, 02 Nov 2026 09:00:00 GMT"),
                          title="Feed title")
        self.addCleanup(setattr, Server, "body", original)
        config = Config()
        config.feeds = [Feed(url=self.base + "/plain", max_items=2),
                        Feed(url=self.base + "/gzip", name="Named")]
        database = Database(":memory:")
        self.addCleanup(database.close)
        # The first feed keeps its 2 newest; the second adds only the story not already stored.
        self.assertEqual(feeds.update_all(config, database), 3)
        rows = database.connection.execute("SELECT title, source FROM articles").fetchall()
        self.assertEqual({row["title"]: row["source"] for row in rows},
                         {"New": "Feed title", "Middle": "Feed title", "Old": "Named"})
        self.assertEqual(database.feed_states()[self.base + "/plain"]["title"], "Feed title")

    def test_update_all_respects_stop_and_next_attempt(self):
        config = Config()
        config.feeds = [Feed(url=self.base + "/plain")]
        database = Database(":memory:")
        self.addCleanup(database.close)
        stop = threading.Event()
        stop.set()
        self.assertEqual(feeds.update_all(config, database, stop=stop), 0)
        self.assertEqual(Server.requests, [])
        database.connection.execute("INSERT INTO feeds(url, next_attempt) VALUES(?, ?)",
                                    (self.base + "/plain", time.time() + 3600))
        self.assertEqual(feeds.update_all(config, database), 0)
        self.assertEqual(Server.requests, [])


if __name__ == "__main__":
    unittest.main()
