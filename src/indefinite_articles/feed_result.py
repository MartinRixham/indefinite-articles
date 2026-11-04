"""The outcome of fetching one feed, and storing it."""

import logging
import sqlite3
from typing import TYPE_CHECKING

from .config import Config
from .database import Database
from .feed import Feed

if TYPE_CHECKING:
    from .feeds import FetchResult, ParsedFeed

log = logging.getLogger(__name__)


class FeedResult:
    """The outcome of fetching one feed: "ok", "not-modified" or "error".

    `parsed` is set when the status is "ok", `fetched` when it is "ok" or "not-modified",
    and `error` describes what went wrong when it is "error".
    """

    def __init__(
        self,
        feed: Feed,
        status: str,
        parsed: "ParsedFeed | None" = None,
        fetched: "FetchResult | None" = None,
        error: str = "",
    ):
        self.feed = feed
        self.status = status
        self.parsed = parsed
        self.fetched = fetched
        self.error = error

    def __repr__(self) -> str:
        detail = f", error={self.error!r}" if self.error else ""
        return f"FeedResult({self.feed.url!r}, {self.status!r}{detail})"

    @classmethod
    def fetch(cls, feed: Feed, state: sqlite3.Row | None) -> "FeedResult":
        """Fetch `feed`, conditionally on the validators stored in its `state`; never raises."""
        from .feeds import CacheValidators, fetch_conditional, parse_feed  # feeds imports this module
        try:
            validators = CacheValidators(state["etag"], state["last_modified"]) if state else None
            fetched = fetch_conditional(feed.url, validators)
            if fetched.data is None:
                return cls(feed, "not-modified", fetched=fetched)
            return cls(feed, "ok", parse_feed(fetched.data), fetched)
        except Exception as error:
            return cls(feed, "error", error=f"{type(error).__name__}: {error}")

    def store(self, config: Config, database: Database) -> int:
        """Record this outcome and the feed's newest articles (not committed); return how many were new."""
        feed = self.feed
        if self.status == "error":
            database.feed_failed(feed.url, self.error, config.fetch_interval_minutes * 60)
            log.debug("feed %s failed: %s", feed.url, self.error)
            return 0
        assert self.fetched is not None
        if self.status == "not-modified":
            database.feed_ok(feed.url, "", self.fetched.validators)
            return 0
        assert self.parsed is not None
        title, entries = self.parsed.title, self.parsed.entries
        source = feed.source(title)
        entries.sort(key=lambda entry: entry.published or 0, reverse=True)
        added = sum(database.add_article(entry, source, feed.category) for entry in entries[: feed.max_items])
        database.feed_ok(feed.url, title, self.fetched.validators)
        return added
