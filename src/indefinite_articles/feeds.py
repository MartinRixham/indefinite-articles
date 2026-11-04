"""Fetch and parse RSS 2.0 / RSS 1.0 (RDF) / Atom feeds using only the standard library."""

import email.utils
import html
import html.entities
import logging
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser

from . import __version__
from .article import title_key
from .config import Config, Feed
from .database import Database
from .feed_result import FeedResult

log = logging.getLogger(__name__)

USER_AGENT = f"indefinite-articles/{__version__} (+feed reader)"
MAX_FEED_BYTES = 10 * 1024 * 1024
SUMMARY_CHARACTERS = 600


@dataclass
class Entry:
    url: str
    title: str
    summary: str
    published: float | None

    @property
    def title_key(self) -> str:
        return title_key(self.title)


@dataclass
class ParsedFeed:
    title: str
    entries: list[Entry]


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


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "br", "div", "li"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    if not markup:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", markup)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _child(element: ElementTree.Element, *names: str) -> ElementTree.Element | None:
    for child in element:
        if _local(child.tag) in names:
            return child
    return None


def _text(element: ElementTree.Element, *names: str) -> str:
    child = _child(element, *names)
    if child is None:
        return ""
    # Atom content may be xhtml with child elements
    return "".join(child.itertext()).strip()


def _parse_date(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return email.utils.parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError, IndexError):
        pass
    try:
        parsed_date = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.astimezone()
        return parsed_date.timestamp()
    except ValueError:
        return None


def _summarise(markup: str) -> str:
    text = html_to_text(markup)
    return text if len(text) <= SUMMARY_CHARACTERS else text[: SUMMARY_CHARACTERS - 1].rsplit(" ", 1)[0] + "…"


_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


def _fix_entities(text: str) -> str:
    """Turn HTML-only named entities (&nbsp; &mdash; ...) into numeric ones XML accepts."""
    return re.sub(r"&([A-Za-z][A-Za-z0-9]*);", _numeric_entity, text)


def _numeric_entity(match: re.Match) -> str:
    name = match.group(1)
    if name in _XML_ENTITIES:
        return match.group(0)
    codepoint = html.entities.name2codepoint.get(name)
    return f"&#{codepoint};" if codepoint else "&amp;" + name + ";"


def _tag_text(block: str, *names: str) -> str:
    """Text of the first of the named tags found in a block of feed XML."""
    for name in names:
        match = re.search(rf"<{name}(?:\s[^>]*)?>(.*?)</{name}>", block, re.S | re.I)
        if match:
            value = match.group(1).strip()
            cdata = re.fullmatch(r"<!\[CDATA\[(.*)\]\]>", value, re.S)
            return cdata.group(1) if cdata else html.unescape(value)
    return ""


def _lenient_entries(text: str) -> ParsedFeed:
    """Last resort for badly broken feeds: pull fields out with regular expressions."""
    entries = []
    for block in re.findall(r"<(?:item|entry)(?:\s[^>]*)?>(.*?)</(?:item|entry)>", text, re.S | re.I):
        url = _tag_text(block, "link")
        if not url:
            match = re.search(r'<link[^>]*href=["\']([^"\']+)', block, re.I)
            url = match.group(1) if match else ""
        title = html_to_text(_tag_text(block, "title"))
        if url and title:
            entries.append(Entry(url.strip(), title,
                                 _summarise(_tag_text(block, "description", "summary", "content")),
                                 _parse_date(_tag_text(block, "pubDate", "published", "updated", "dc:date"))))
    channel_title = re.search(r"<title(?:\s[^>]*)?>(.*?)</title>", text, re.S | re.I)
    return ParsedFeed(html_to_text(channel_title.group(1)) if channel_title else "", entries)


def parse_feed(data: bytes) -> ParsedFeed:
    """Return the feed title and entries. Tolerates common breakage in real-world feeds."""
    data = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    head = data[:200].lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise ValueError("got an HTML page, not a feed")
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as first:
        text = data.decode("utf-8", errors="replace")
        text = re.sub(r"^.*?(?=<\?xml)", "", text, count=1, flags=re.S)  # junk before declaration
        text = re.sub(r"^<\?xml[^>]*\?>", "", text)  # (encoding is now utf-8 text)
        try:
            root = ElementTree.fromstring(_fix_entities(text))
        except ElementTree.ParseError:
            parsed = _lenient_entries(text)
            if not parsed.entries:
                raise first
            return parsed
    kind = _local(root.tag)
    entries: list[Entry] = []
    feed_title = ""

    if kind == "feed":  # Atom
        feed_title = _text(root, "title")
        for entry in root:
            if _local(entry.tag) != "entry":
                continue
            url = ""
            for link in entry:
                if _local(link.tag) == "link" and link.get("rel", "alternate") == "alternate":
                    url = link.get("href", "")
                    break
            title = html_to_text(_text(entry, "title"))
            summary = _text(entry, "summary") or _text(entry, "content")
            date = _text(entry, "published") or _text(entry, "updated")
            if url and title:
                entries.append(Entry(url, title, _summarise(summary), _parse_date(date)))
    else:  # RSS 2.0 (<rss><channel><item>) or RSS 1.0 (<rdf:RDF><item>)
        channel = _child(root, "channel")
        if channel is not None:
            feed_title = _text(channel, "title")
        items = [element for element in root.iter() if _local(element.tag) == "item"]
        for item in items:
            url = _text(item, "link")
            if not url:
                guid = _child(item, "guid")
                if guid is not None and guid.get("isPermaLink", "true") != "false":
                    url = (guid.text or "").strip()
            title = html_to_text(_text(item, "title"))
            summary = _text(item, "description") or _text(item, "encoded", "summary")
            date = _text(item, "pubDate", "date", "published", "updated")
            if url and title:
                entries.append(Entry(url, title, _summarise(summary), _parse_date(date)))
    return ParsedFeed(html_to_text(feed_title), entries)


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
