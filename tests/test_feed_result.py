import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from indefinite_articles.config import Config
from indefinite_articles.database import Database
from indefinite_articles.feed import Feed
from indefinite_articles.feed_result import FeedResult
from indefinite_articles.feeds import CacheValidators, Entry, FetchResult, ParsedFeed

FEED_BODY = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>Channel</title>'
             b"<item><title>One</title><link>https://example.com/1</link></item></channel></rss>")
LAST_MODIFIED = "Mon, 02 Nov 2026 10:00:00 GMT"


class Server(BaseHTTPRequestHandler):
    """/feed answers 304 whenever a validator is sent; /broken serves something that isn't a feed."""
    requests: list[dict[str, str]] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        Server.requests.append(dict(self.headers))
        if self.headers.get("If-None-Match") or self.headers.get("If-Modified-Since"):
            self.send_response(304)
            self.end_headers()
            return
        body = FEED_BODY if self.path == "/feed" else b"this is not a feed"
        self.send_response(200)
        self.send_header("ETag", '"version-1"')
        self.send_header("Last-Modified", LAST_MODIFIED)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def entry(title: str, published: float | None = None) -> Entry:
    return Entry(f"https://example.com/{title.lower()}", title, "", published)


def fetched() -> FetchResult:
    return FetchResult(b"", CacheValidators('"version-2"', LAST_MODIFIED))


class FetchTests(unittest.TestCase):
    server: ThreadingHTTPServer
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        Server.requests.clear()
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)

    def _state(self, url: str) -> None:
        self.database.feed_ok(url, "Channel", CacheValidators('"version-1"', LAST_MODIFIED))

    def test_ok_parses_the_feed_and_keeps_its_validators(self) -> None:
        feed = Feed(url=self.base + "/feed")
        result = FeedResult.fetch(feed, None)
        self.assertEqual(result.status, "ok")
        self.assertIs(result.feed, feed)
        assert result.parsed is not None and result.fetched is not None
        self.assertEqual(result.parsed.title, "Channel")
        self.assertEqual([entry.title for entry in result.parsed.entries], ["One"])
        self.assertEqual((result.fetched.validators.etag, result.fetched.validators.last_modified),
                         ('"version-1"', LAST_MODIFIED))
        self.assertEqual(result.error, "")
        self.assertNotIn("If-None-Match", Server.requests[-1])

    def test_sends_stored_validators_and_reports_not_modified(self) -> None:
        url = self.base + "/feed"
        self._state(url)
        result = FeedResult.fetch(Feed(url=url), self.database.feed_states()[url])
        self.assertEqual(result.status, "not-modified")
        self.assertIsNone(result.parsed)
        self.assertIsNotNone(result.fetched)
        headers = Server.requests[-1]
        self.assertEqual((headers["If-None-Match"], headers["If-Modified-Since"]), ('"version-1"', LAST_MODIFIED))

    def test_unparseable_feed_is_an_error_not_an_exception(self) -> None:
        result = FeedResult.fetch(Feed(url=self.base + "/broken"), None)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.parsed)
        self.assertIsNone(result.fetched)
        self.assertRegex(result.error, r"^\w+: ")

    def test_unreachable_server_is_an_error(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        port = server.server_port
        server.server_close()  # nothing listens on this port now
        result = FeedResult.fetch(Feed(url=f"http://127.0.0.1:{port}/feed"), None)
        self.assertEqual(result.status, "error")
        self.assertTrue(result.error.startswith("URLError: "), result.error)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)

    def _state(self, url: str) -> dict:
        return dict(self.database.feed_states()[url])

    def _sources(self) -> dict[str, str]:
        rows = self.database.connection.execute("SELECT title, source, category FROM articles")
        return {row["title"]: f"{row['source']}/{row['category']}" for row in rows}

    def test_ok_stores_the_newest_entries_and_returns_how_many_were_new(self) -> None:
        feed = Feed(url="https://feed", category="science", max_items=2)
        parsed = ParsedFeed("Channel", [entry("Old", 1.0), entry("New", 3.0), entry("Middle", 2.0)])
        result = FeedResult(feed, "ok", parsed, fetched())
        self.assertEqual(result.store(self.config, self.database), 2)
        self.assertEqual(self._sources(), {"New": "Channel/science", "Middle": "Channel/science"})
        state = self._state("https://feed")
        self.assertEqual((state["title"], state["etag"], state["failures"]), ("Channel", '"version-2"', 0))
        # Storing the same entries again adds nothing.
        self.assertEqual(result.store(self.config, self.database), 0)

    def test_source_is_the_feed_name_then_its_title_then_its_url(self) -> None:
        cases = [(Feed(url="https://named", name="Named"), "Title", "A"),
                 (Feed(url="https://titled"), "Title", "B"),
                 (Feed(url="https://untitled"), "", "C")]
        for feed, title, story in cases:
            FeedResult(feed, "ok", ParsedFeed(title, [entry(story)]), fetched()).store(self.config, self.database)
        self.assertEqual(self._sources(), {"A": "Named/", "B": "Title/", "C": "https://untitled/"})

    def test_not_modified_updates_validators_and_keeps_the_title(self) -> None:
        feed = Feed(url="https://feed")
        self.database.feed_ok(feed.url, "Channel", CacheValidators('"version-1"', "yesterday"))
        self.database.feed_failed(feed.url, "earlier failure", 60)
        self.assertEqual(FeedResult(feed, "not-modified", fetched=fetched()).store(self.config, self.database), 0)
        state = self._state(feed.url)
        self.assertEqual((state["title"], state["etag"], state["last_modified"], state["failures"]),
                         ("Channel", '"version-2"', LAST_MODIFIED, 0))

    def test_error_records_the_failure(self) -> None:
        feed = Feed(url="https://feed")
        result = FeedResult(feed, "error", error="URLError: refused")
        self.assertEqual(result.store(self.config, self.database), 0)
        self.assertEqual(result.store(self.config, self.database), 0)
        state = self._state(feed.url)
        self.assertEqual((state["failures"], state["last_error"]), (2, "URLError: refused"))
        self.assertGreater(state["next_attempt"], state["last_attempt"])  # backing off after two failures
        self.assertEqual(self._sources(), {})

    def test_does_not_commit(self) -> None:
        result = FeedResult(Feed(url="https://feed"), "ok", ParsedFeed("Channel", [entry("One")]), fetched())
        result.store(self.config, self.database)
        self.assertTrue(self.database.connection.in_transaction)


class ReprTests(unittest.TestCase):
    def test_shows_url_status_and_error(self) -> None:
        self.assertEqual(repr(FeedResult(Feed(url="https://feed"), "error", error="boom")),
                         "FeedResult('https://feed', 'error', error='boom')")
        self.assertEqual(repr(FeedResult(Feed(url="https://feed"), "not-modified")),
                         "FeedResult('https://feed', 'not-modified')")


if __name__ == "__main__":
    unittest.main()
