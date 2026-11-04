"""Fetch RSS 2.0 / RSS 1.0 (RDF) / Atom feeds using only the standard library."""

import logging
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import __version__
from ..config import Config
from ..storage.database import Database
from .feed import Feed
from .feed_result import FeedResult

log = logging.getLogger(__name__)

USER_AGENT = f"indefinite-articles/{__version__} (+feed reader)"
MAX_FEED_BYTES = 10 * 1024 * 1024


@dataclass
class CacheValidators:
    """The ETag and Last-Modified headers a server sent, for the next conditional GET."""
    etag: str | None = None
    last_modified: str | None = None

    def request_headers(self) -> dict[str, str]:
        """The headers that make a GET conditional on these validators."""
        headers = {}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        return headers


@dataclass
class FetchResult:
    """A fetched feed body (None when the server says it hasn't changed) and its validators."""
    data: bytes | None
    validators: CacheValidators


def _decompress(data: bytes, encoding: str) -> bytes:
    if encoding not in ("gzip", "x-gzip", "deflate"):
        return data
    decompressor = zlib.decompressobj(wbits=47 if "gzip" in encoding else 15)  # 47: auto gzip/zlib header
    out = decompressor.decompress(data, MAX_FEED_BYTES + 1)
    if len(out) > MAX_FEED_BYTES:
        raise ValueError("feed too large")
    return out


def fetch_conditional(url: str, validators: CacheValidators | None = None,
                      timeout: int = 30) -> FetchResult:
    """GET a feed. `data` is None when the server says it hasn't changed."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*;q=0.5",
        "Accept-Encoding": "gzip",
    }
    validators = validators or CacheValidators()
    headers.update(validators.request_headers())
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(MAX_FEED_BYTES + 1)
            new_validators = CacheValidators(response.headers.get("ETag"), response.headers.get("Last-Modified"))
            encoding = (response.headers.get("Content-Encoding") or "").lower()
    except urllib.error.HTTPError as error:
        if error.code == 304:
            return FetchResult(None, validators)
        raise
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("feed too large")
    return FetchResult(_decompress(data, encoding), new_validators)


def fetch(url: str, timeout: int = 30) -> bytes:
    data = fetch_conditional(url, timeout=timeout).data
    if data is None:  # 304 without If-None-Match/If-Modified-Since: a broken server
        raise ValueError(f"unexpected 304 Not Modified from {url}")
    return data


def _fetch_group(feeds: list[Feed], states: dict[str, sqlite3.Row], stop: threading.Event | None) -> list[FeedResult]:
    """Fetch feeds that share a host one after another, to be polite to that host."""
    results: list[FeedResult] = []
    for feed in feeds:
        if stop is not None and stop.is_set():
            break
        results.append(FeedResult.fetch(feed, states.get(feed.url)))
    return results


def update_all(config: Config, database: Database, stop: threading.Event | None = None) -> int:
    """Fetch all due feeds; return number of new articles stored."""
    now = time.time()
    states = database.feed_states()
    due = [feed for feed in config.feeds
           if feed.url not in states or (states[feed.url]["next_attempt"] or 0) <= now]
    groups: dict[str, list[Feed]] = {}
    for feed in due:
        groups.setdefault(feed.host, []).append(feed)

    new = 0
    statuses: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, config.fetch_concurrency)) as pool:
        for results in pool.map(lambda group: _fetch_group(group, states, stop), groups.values()):
            for result in results:
                new += result.store(config, database)
                statuses[result.status] += 1
            database.connection.commit()
    log.info("feeds: %d due, %d updated, %d unchanged, %d failed; %d new articles",
             len(due), statuses["ok"], statuses["not-modified"], statuses["error"], new)
    return new
