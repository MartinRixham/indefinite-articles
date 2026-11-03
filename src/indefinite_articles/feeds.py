"""Fetch and parse RSS 2.0 / RSS 1.0 (RDF) / Atom feeds using only the standard library."""

from __future__ import annotations

import email.utils
import html
import html.entities
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser

from . import __version__

log = logging.getLogger(__name__)

USER_AGENT = f"indefinite-articles/{__version__} (+feed reader)"
MAX_FEED_BYTES = 10 * 1024 * 1024
SUMMARY_CHARS = 600


@dataclass
class Entry:
    url: str
    title: str
    summary: str
    published: float | None


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "br", "div", "li"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(s: str) -> str:
    if not s:
        return ""
    p = _TextExtractor()
    try:
        p.feed(s)
        p.close()
        text = "".join(p.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def title_key(title: str) -> str:
    """Normalised title used to de-duplicate the same story from different feeds."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())[:120]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _child(el, *names):
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _text(el, *names) -> str:
    c = _child(el, *names)
    if c is None:
        return ""
    # Atom content may be xhtml with child elements
    return "".join(c.itertext()).strip()


def _parse_date(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return email.utils.parsedate_to_datetime(s).timestamp()
    except (TypeError, ValueError, IndexError):
        pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return dt.timestamp()
    except ValueError:
        return None


def _summarise(s: str) -> str:
    t = html_to_text(s)
    return t if len(t) <= SUMMARY_CHARS else t[: SUMMARY_CHARS - 1].rsplit(" ", 1)[0] + "…"


_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


def _fix_entities(text: str) -> str:
    """Turn HTML-only named entities (&nbsp; &mdash; ...) into numeric ones XML accepts."""
    def sub(m):
        name = m.group(1)
        if name in _XML_ENTITIES:
            return m.group(0)
        cp = html.entities.name2codepoint.get(name)
        return f"&#{cp};" if cp else "&amp;" + name + ";"
    return re.sub(r"&([A-Za-z][A-Za-z0-9]*);", sub, text)


def _lenient_entries(text: str) -> tuple[str, list[Entry]]:
    """Last resort for badly broken feeds: pull fields out with regular expressions."""
    def field(block, *names):
        for n in names:
            m = re.search(rf"<{n}(?:\s[^>]*)?>(.*?)</{n}>", block, re.S | re.I)
            if m:
                v = m.group(1).strip()
                cdata = re.fullmatch(r"<!\[CDATA\[(.*)\]\]>", v, re.S)
                return cdata.group(1) if cdata else html.unescape(v)
        return ""

    entries = []
    for block in re.findall(r"<(?:item|entry)(?:\s[^>]*)?>(.*?)</(?:item|entry)>", text, re.S | re.I):
        url = field(block, "link")
        if not url:
            m = re.search(r'<link[^>]*href=["\']([^"\']+)', block, re.I)
            url = m.group(1) if m else ""
        title = html_to_text(field(block, "title"))
        if url and title:
            entries.append(Entry(url.strip(), title,
                                 _summarise(field(block, "description", "summary", "content")),
                                 _parse_date(field(block, "pubDate", "published", "updated", "dc:date"))))
    chan = re.search(r"<title(?:\s[^>]*)?>(.*?)</title>", text, re.S | re.I)
    return html_to_text(chan.group(1)) if chan else "", entries


def parse_feed(data: bytes) -> tuple[str, list[Entry]]:
    """Return (feed title, entries). Tolerates common breakage in real-world feeds."""
    data = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    head = data[:200].lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise ValueError("got an HTML page, not a feed")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as first:
        text = data.decode("utf-8", errors="replace")
        text = re.sub(r"^.*?(?=<\?xml)", "", text, count=1, flags=re.S)  # junk before declaration
        text = re.sub(r"^<\?xml[^>]*\?>", "", text)  # (encoding is now utf-8 text)
        try:
            root = ET.fromstring(_fix_entities(text))
        except ET.ParseError:
            title, entries = _lenient_entries(text)
            if not entries:
                raise first
            return title, entries
    kind = _local(root.tag)
    entries: list[Entry] = []
    feed_title = ""

    if kind == "feed":  # Atom
        feed_title = _text(root, "title")
        for e in root:
            if _local(e.tag) != "entry":
                continue
            url = ""
            for link in e:
                if _local(link.tag) == "link" and link.get("rel", "alternate") == "alternate":
                    url = link.get("href", "")
                    break
            title = html_to_text(_text(e, "title"))
            summary = _text(e, "summary") or _text(e, "content")
            date = _text(e, "published") or _text(e, "updated")
            if url and title:
                entries.append(Entry(url, title, _summarise(summary), _parse_date(date)))
    else:  # RSS 2.0 (<rss><channel><item>) or RSS 1.0 (<rdf:RDF><item>)
        channel = _child(root, "channel")
        if channel is not None:
            feed_title = _text(channel, "title")
        items = [el for el in root.iter() if _local(el.tag) == "item"]
        for it in items:
            url = _text(it, "link")
            if not url:
                guid = _child(it, "guid")
                if guid is not None and guid.get("isPermaLink", "true") != "false":
                    url = (guid.text or "").strip()
            title = html_to_text(_text(it, "title"))
            summary = _text(it, "description") or _text(it, "encoded", "summary")
            date = _text(it, "pubDate", "date", "published", "updated")
            if url and title:
                entries.append(Entry(url, title, _summarise(summary), _parse_date(date)))
    return html_to_text(feed_title), entries


def _decompress(data: bytes, encoding: str) -> bytes:
    if encoding not in ("gzip", "x-gzip", "deflate"):
        return data
    d = zlib.decompressobj(wbits=47 if "gzip" in encoding else 15)  # 47: auto gzip/zlib header
    out = d.decompress(data, MAX_FEED_BYTES + 1)
    if len(out) > MAX_FEED_BYTES:
        raise ValueError("feed too large")
    return out


def fetch_conditional(url: str, etag: str | None = None, last_modified: str | None = None,
                      timeout: int = 30) -> tuple[bytes | None, dict]:
    """GET a feed. Returns (None, meta) when the server says it hasn't changed."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*;q=0.5",
        "Accept-Encoding": "gzip",
    }
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(MAX_FEED_BYTES + 1)
            meta = {"etag": resp.headers.get("ETag"), "last_modified": resp.headers.get("Last-Modified")}
            encoding = (resp.headers.get("Content-Encoding") or "").lower()
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return None, {"etag": etag, "last_modified": last_modified}
        raise
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("feed too large")
    return _decompress(data, encoding), meta


def fetch(url: str, timeout: int = 30) -> bytes:
    return fetch_conditional(url, timeout=timeout)[0]


def _fetch_group(feeds, states, stop):
    """Fetch feeds that share a host one after another, to be polite to that host."""
    results = []
    for f in feeds:
        if stop is not None and stop.is_set():
            break
        st = states.get(f.url)
        try:
            data, meta = fetch_conditional(f.url, st["etag"] if st else None,
                                           st["last_modified"] if st else None)
            if data is None:
                results.append((f, "not-modified", None, meta))
            else:
                results.append((f, "ok", parse_feed(data), meta))
        except Exception as e:
            results.append((f, "error", None, f"{type(e).__name__}: {e}"))
    return results


def update_all(cfg, db, stop=None) -> int:
    """Fetch all due feeds; return number of new articles stored."""
    now = time.time()
    states = db.feed_states()
    due = [f for f in cfg.feeds
           if f.url not in states or (states[f.url]["next_attempt"] or 0) <= now]
    groups: dict[str, list] = {}
    for f in due:
        groups.setdefault(urllib.parse.urlsplit(f.url).hostname or "", []).append(f)

    new = changed = unchanged = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.fetch_concurrency)) as pool:
        for results in pool.map(lambda g: _fetch_group(g, states, stop), groups.values()):
            for f, status, parsed, meta in results:
                if status == "error":
                    failed += 1
                    db.feed_failed(f.url, meta, cfg.fetch_interval_minutes * 60)
                    log.debug("feed %s failed: %s", f.url, meta)
                    continue
                if status == "not-modified":
                    unchanged += 1
                    db.feed_ok(f.url, "", meta["etag"], meta["last_modified"])
                    continue
                title, entries = parsed
                source = f.name or title or f.url
                entries.sort(key=lambda en: en.published or 0, reverse=True)
                n = 0
                for en in entries[: f.max_items]:
                    if db.add_article(en.url, en.title, title_key(en.title), en.summary,
                                      source, en.published, f.category):
                        n += 1
                db.feed_ok(f.url, title, meta["etag"], meta["last_modified"])
                changed += 1
                new += n
            db.conn.commit()
    log.info("feeds: %d due, %d updated, %d unchanged, %d failed; %d new articles",
             len(due), changed, unchanged, failed, new)
    return new
