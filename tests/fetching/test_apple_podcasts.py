import json
import threading
import time
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from indefinite_articles.config import Config
from indefinite_articles.fetching import apple_podcasts
from indefinite_articles.fetching.entry import Entry
from indefinite_articles.fetching.feeds import CacheValidators
from indefinite_articles.storage.article import Article
from indefinite_articles.storage.database import Database

FEED = "https://pod.example.com/feed.xml"


def show_result(collection_id: int, name: str, feed_url: str) -> dict:
    return {"wrapperType": "track", "kind": "podcast", "collectionId": collection_id, "collectionName": name,
            "feedUrl": feed_url}


def episode_result(track_id: int, title: str, audio_url: str) -> dict:
    return {"wrapperType": "podcastEpisode", "kind": "podcast-episode", "trackId": track_id, "trackName": title,
            "episodeUrl": audio_url,
            "trackViewUrl": f"https://podcasts.apple.com/us/podcast/x/id1?i={track_id}&uo=4"}


class FakeApple(BaseHTTPRequestHandler):
    """Answers /search with `shows_by_term[term]` (or else `shows`) and /lookup?id=N with `episodes[N]`;
    records each request."""
    shows: list[dict] = []
    shows_by_term: dict[str, list[dict]] = {}
    episodes: dict[str, list[dict]] = {}
    requests: list[str] = []
    status = 200

    def do_GET(self) -> None:
        FakeApple.requests.append(self.path)
        parts = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(parts.query))
        if FakeApple.status != 200:
            self.send_error(FakeApple.status)
            return
        if parts.path == "/search":
            results = FakeApple.shows_by_term.get(query.get("term", ""), FakeApple.shows)
        else:
            results = [{"kind": "podcast", "collectionId": int(query["id"])}] + FakeApple.episodes.get(query["id"], [])
        body = json.dumps({"resultCount": len(results), "results": results}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class AddLinksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApple)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.patches = [mock.patch.object(apple_podcasts, "SEARCH_URL", base + "/search"),
                       mock.patch.object(apple_podcasts, "LOOKUP_URL", base + "/lookup"),
                       mock.patch.object(apple_podcasts, "REQUEST_INTERVAL", 0)]
        for patch in cls.patches:
            patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for patch in cls.patches:
            patch.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        FakeApple.shows, FakeApple.episodes, FakeApple.requests, FakeApple.status = [], {}, [], 200
        FakeApple.shows_by_term = {}
        self.config = Config()
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)
        self.database.feed_ok(FEED, "The Show", CacheValidators())
        self._episode("Episode One", "https://cdn.example.com/1.mp3")
        self._episode("Episode Two", "https://cdn.example.com/2.mp3")

    def _episode(self, title: str, audio_url: str) -> None:
        self.database.add_article(Entry(f"https://pod.example.com/{title}", title, "notes", time.time(),
                                        audio_url=audio_url, duration=60.0), "The Show", "Podcasts", FEED)
        self.database.connection.commit()

    def _links(self) -> dict[str, str | None]:
        return {article.title: article.apple_url for article in self.database.feed_episodes(FEED, 10)}

    def _apple_id(self) -> str | None:
        return self.database.feed_states()[FEED]["apple_id"]

    def test_matched_by_feed_url_and_episodes_by_audio_or_title(self) -> None:
        FakeApple.shows = [show_result(7, "Another show", "https://other.example.com/feed"),
                           show_result(42, "The Show (renamed)", "http://www.pod.example.com/feed.xml/")]
        FakeApple.episodes = {"42": [episode_result(1001, "Episode 1", "https://cdn.example.com/1.mp3"),
                                     episode_result(1002, "Episode Two!", "https://tracking.example.com/2.mp3")]}
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 2)
        self.assertEqual(self._links(), {
            "Episode One": "https://podcasts.apple.com/us/podcast/x/id1?i=1001",
            "Episode Two": "https://podcasts.apple.com/us/podcast/x/id1?i=1002",
        })
        self.assertEqual(self._apple_id(), "42")
        self.assertIn("term=The+Show", FakeApple.requests[0])

    def test_feed_missing_from_title_search_is_searched_for_by_opml_name(self) -> None:
        self.database.feed_ok(FEED, "The Documentary Podcast", CacheValidators())
        FakeApple.shows_by_term = {
            "The Documentary Podcast": [show_result(5, "The Documentary Podcast", "https://elsewhere.example.com/a")],
            "The Show": [show_result(42, "The Documentary Podcast", "https://pod.example.com/feed.xml")],
        }
        FakeApple.episodes = {"42": [episode_result(1001, "Episode One", "")]}
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 1)
        self.assertEqual(self._apple_id(), "42")
        self.assertEqual([urllib.parse.urlsplit(path).path for path in FakeApple.requests],
                         ["/search", "/search", "/lookup"])

    def test_known_show_is_not_searched_again_and_missing_episodes_are_retried(self) -> None:
        self.database.set_feed_apple_id(FEED, "42")
        FakeApple.episodes = {"42": [episode_result(1001, "Episode One", "")]}
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 1)
        self.assertEqual(self._links()["Episode Two"], None)  # not on Apple yet
        FakeApple.episodes["42"].append(episode_result(1002, "Episode Two", ""))
        FakeApple.requests = []
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 1)
        self.assertEqual([urllib.parse.urlsplit(path).path for path in FakeApple.requests], ["/lookup"])
        self.assertTrue(all(self._links().values()))

    def test_same_name_show_needs_shared_episodes(self) -> None:
        FakeApple.shows = [show_result(5, "The Show", "https://elsewhere.example.com/a"),
                           show_result(6, "the show", "https://elsewhere.example.com/b")]
        FakeApple.episodes = {"5": [episode_result(1, "Unrelated", "https://x/1.mp3")],
                              "6": [episode_result(2, "Episode One", "https://x/2.mp3")]}
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 1)
        self.assertEqual(self._apple_id(), "6")

    def test_show_whose_name_contains_the_feed_title_needs_shared_episodes(self) -> None:
        FakeApple.shows = [show_result(5, "The Show Daily", "https://elsewhere.example.com/a"),
                           show_result(6, "The Show: The Podcast", "https://podcast.example.com/feed.xml"),
                           show_result(7, "Another show", "https://elsewhere.example.com/c")]
        FakeApple.episodes = {"5": [episode_result(1, "Unrelated", "https://x/1.mp3")],
                              "6": [episode_result(2, "Something else", "https://cdn.example.com/2.mp3")]}
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 1)
        self.assertEqual(self._apple_id(), "6")
        self.assertNotIn("/lookup?id=7", "".join(FakeApple.requests))

    def test_feed_not_on_apple_is_checked_again_only_after_a_week(self) -> None:
        FakeApple.shows = [show_result(5, "The Show", "https://elsewhere.example.com/a")]
        FakeApple.episodes = {"5": [episode_result(1, "Unrelated", "https://x/1.mp3")]}
        with self.assertLogs("indefinite_articles.fetching.apple_podcasts", "INFO"):
            self.assertEqual(apple_podcasts.add_links(self.config, self.database), 0)
        self.assertEqual(self._apple_id(), "")
        FakeApple.requests = []
        apple_podcasts.add_links(self.config, self.database)
        self.assertEqual(FakeApple.requests, [])
        self.database.connection.execute("UPDATE feeds SET apple_checked_at=?",
                                         (time.time() - apple_podcasts.RECHECK_SECONDS - 1,))
        apple_podcasts.add_links(self.config, self.database)
        self.assertEqual(len(FakeApple.requests), 2)  # search, then the same-name show's episodes

    def test_rate_limit_stops_the_round_without_recording_anything(self) -> None:
        FakeApple.status = 429
        with self.assertLogs("indefinite_articles.fetching.apple_podcasts", "WARNING"):
            self.assertEqual(apple_podcasts.add_links(self.config, self.database), 0)
        self.assertIsNone(self._apple_id())

    def test_server_error_is_retried_next_time(self) -> None:
        FakeApple.status = 500
        self.assertEqual(apple_podcasts.add_links(self.config, self.database), 0)
        self.assertIsNone(self._apple_id())

    def test_disabled(self) -> None:
        for setting, value in (("apple_podcasts_links", False), ("podcasts_per_digest", 0)):
            with self.subTest(setting=setting):
                config = Config()
                setattr(config, setting, value)
                self.assertEqual(apple_podcasts.add_links(config, self.database), 0)
        self.assertEqual(FakeApple.requests, [])


class HelperTests(unittest.TestCase):
    def test_episode_link_drops_tracking_parameter(self) -> None:
        self.assertEqual(apple_podcasts._episode_link("https://podcasts.apple.com/us/podcast/x/id1?i=99&uo=4"),
                         "https://podcasts.apple.com/us/podcast/x/id1?i=99")

    def test_normal_feed_url(self) -> None:
        self.assertEqual(apple_podcasts._normal_feed_url("http://WWW.Example.com/feed/"),
                         apple_podcasts._normal_feed_url("https://example.com/feed"))
        self.assertNotEqual(apple_podcasts._normal_feed_url("https://example.com/feed?a=1"),
                            apple_podcasts._normal_feed_url("https://example.com/feed?a=2"))

    def test_link_for_ignores_empty_audio_and_titles(self) -> None:
        show = apple_podcasts.AppleShow("1", [apple_podcasts.AppleEpisode("", "", "https://apple/1")])
        self.assertEqual(show.link_for(Article(1, "https://e/1", "?!")), "")


if __name__ == "__main__":
    unittest.main()
