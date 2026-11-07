"""Apple Podcasts links for shortlisted podcast episodes, from Apple's public iTunes Search API.

A feed is matched to an Apple Podcasts show by searching for its title. A result with the
same feed URL is taken; otherwise a result with the same name, or else one whose name contains
the feed's title (a show's website feed is often called "Latent Space" while Apple lists
"Latent Space: The AI Engineer Podcast"), is taken only if some of the feed's episodes are
also among that show's episodes, because unrelated shows share names.
The show's id is kept in the `feeds` table, so the search runs once per feed (and again
after `RECHECK_SECONDS` for feeds Apple didn't have). Each episode is then found among
the show's newest episodes by its audio URL or title. Apple lists new episodes some hours
after they come out, so an episode not found yet is tried again on the next fetch.
"""

import json
import logging
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

from ..config import Config
from ..storage.article import Article, title_key
from ..storage.database import Database
from .feeds import USER_AGENT

log = logging.getLogger(__name__)

SEARCH_URL = "https://itunes.apple.com/search"
LOOKUP_URL = "https://itunes.apple.com/lookup"
REQUEST_INTERVAL = 3.5          # seconds between requests; Apple allows about 20 a minute
RECHECK_SECONDS = 7 * 86400     # before looking again for a feed Apple didn't have
NAME_MATCHES_TRIED = 2          # similarly named shows checked for shared episodes
VALIDATION_EPISODES = 20        # of the feed's stored episodes, looked for in a same-name show
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
TIMEOUT = 20


class AppleRefused(Exception):
    """Apple turned a request down (rate limit or block): stop asking for now."""


@dataclass(frozen=True)
class AppleEpisode:
    title_key: str
    audio_url: str
    link: str


@dataclass
class AppleShow:
    """A show on Apple Podcasts and its newest episodes."""
    collection_id: str
    episodes: list[AppleEpisode]

    def link_for(self, article: Article) -> str:
        """The Apple Podcasts link of the article's episode, or '' if Apple doesn't list it."""
        key = title_key(article.title)
        for episode in self.episodes:
            if (article.audio_url and episode.audio_url == article.audio_url) or (key and episode.title_key == key):
                return episode.link
        return ""

    def shares_episodes_with(self, articles: Iterable[Article]) -> bool:
        return any(self.link_for(article) for article in articles)


def _normal_feed_url(url: str) -> str:
    """A feed URL without its scheme, `www.` or trailing slash, lowercased, for comparing."""
    parts = urllib.parse.urlsplit(url.strip().lower())
    query = f"?{parts.query}" if parts.query else ""
    return parts.netloc.removeprefix("www.") + parts.path.rstrip("/") + query


def _episode_link(track_view_url: str) -> str:
    """Apple's episode page URL without its `uo` tracking parameter."""
    parts = urllib.parse.urlsplit(track_view_url)
    query = urllib.parse.urlencode([(key, value) for key, value in urllib.parse.parse_qsl(parts.query)
                                    if key != "uo"])
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


class _Client:
    """Requests to the iTunes Search API, spaced out to stay within Apple's rate limit."""

    def __init__(self, stop: threading.Event | None):
        self._stop = stop
        self._last_request = -REQUEST_INTERVAL

    def get(self, url: str, parameters: dict[str, str]) -> list[dict]:
        """The `results` of a request; raises `AppleRefused` on HTTP 403 or 429."""
        wait = self._last_request + REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            if self._stop is not None:
                self._stop.wait(wait)
            else:
                time.sleep(wait)
        request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(parameters)}",
                                         headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                data = response.read(MAX_RESPONSE_BYTES)
        except urllib.error.HTTPError as error:
            error.close()
            if error.code in (403, 429):
                raise AppleRefused(f"HTTP {error.code}") from None
            raise
        finally:
            self._last_request = time.monotonic()
        answer = json.loads(data)
        results = answer.get("results") if isinstance(answer, dict) else None
        return results if isinstance(results, list) else []

    def show(self, collection_id: str) -> AppleShow:
        results = self.get(LOOKUP_URL, {"id": collection_id, "media": "podcast", "entity": "podcastEpisode",
                                        "limit": "200"})
        episodes = [AppleEpisode(title_key(str(result.get("trackName") or "")), str(result.get("episodeUrl") or ""),
                                 _episode_link(str(result["trackViewUrl"])))
                    for result in results
                    if result.get("kind") == "podcast-episode" and result.get("trackViewUrl")]
        return AppleShow(collection_id, episodes)


def _find_show(client: _Client, database: Database, feed_url: str, state: sqlite3.Row,
               fallback_title: str) -> AppleShow | None:
    """The feed's show on Apple Podcasts, searched for by title; records the result in `feeds`."""
    title = state["title"] or fallback_title
    results = client.get(SEARCH_URL, {"media": "podcast", "entity": "podcast", "limit": "10", "term": title})
    for result in results:
        if result.get("collectionId") and _normal_feed_url(str(result.get("feedUrl") or "")) == \
                _normal_feed_url(feed_url):
            show = client.show(str(result["collectionId"]))
            database.set_feed_apple_id(feed_url, show.collection_id)
            return show
    wanted = title.strip().casefold()
    same_name = []
    similar_name = []
    for result in results:
        name = str(result.get("collectionName") or "").strip().casefold()
        if not result.get("collectionId") or not wanted:
            continue
        if name == wanted:
            same_name.append(result)
        elif wanted in name:
            similar_name.append(result)
    known = database.feed_episodes(feed_url, VALIDATION_EPISODES)
    for result in (same_name + similar_name)[:NAME_MATCHES_TRIED]:
        show = client.show(str(result["collectionId"]))
        if show.shares_episodes_with(known):
            database.set_feed_apple_id(feed_url, show.collection_id)
            return show
    database.set_feed_apple_id(feed_url, "")
    log.info("podcast feed %s not found on Apple Podcasts", feed_url)
    return None


def _show_for(client: _Client, database: Database, feed_url: str, state: sqlite3.Row | None,
              fallback_title: str) -> AppleShow | None:
    if state is None:
        return None  # the feed's fetch was never recorded: nothing to search with
    apple_id = state["apple_id"]
    if apple_id:
        return client.show(apple_id)
    if apple_id == "" and time.time() - (state["apple_checked_at"] or 0) < RECHECK_SECONDS:
        return None
    return _find_show(client, database, feed_url, state, fallback_title)


def add_links(config: Config, database: Database, stop: threading.Event | None = None) -> int:
    """Find Apple Podcasts links for shortlisted episodes that don't have one; return how many were found."""
    if not config.apple_podcasts_links or config.podcasts_per_digest <= 0:
        return 0
    from ..ranking.ranker import shortlist
    by_feed: dict[str, list[Article]] = {}
    for article in shortlist(config, database):
        if article.is_podcast and not article.apple_url and article.feed_url:
            by_feed.setdefault(article.feed_url, []).append(article)
    if not by_feed:
        return 0
    client = _Client(stop)
    states = database.feed_states()
    found = 0
    for feed_url, episodes in by_feed.items():
        if stop is not None and stop.is_set():
            break
        try:
            show = _show_for(client, database, feed_url, states.get(feed_url), episodes[0].source)
        except AppleRefused as error:
            log.warning("Apple Podcasts refused a lookup (%s); trying again on the next fetch", error)
            break
        except (OSError, ValueError) as error:  # network trouble or a garbled answer: try next time
            log.debug("Apple Podcasts lookup for %s failed: %s", feed_url, error)
            continue
        finally:
            database.connection.commit()
        if show is None:
            continue
        for article in episodes:
            link = show.link_for(article)
            if link:
                database.set_apple_url(article.id, link)
                found += 1
        database.connection.commit()
    log.info("Apple Podcasts links: found %d for %d shortlisted episodes", found,
             sum(len(episodes) for episodes in by_feed.values()))
    return found
