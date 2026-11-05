"""Parse RSS 2.0 / RSS 1.0 (RDF) / Atom feeds using only the standard library."""

import email.utils
import html
import html.entities
import re
import xml.etree.ElementTree as ElementTree
from datetime import datetime

from .entry import Entry
from .text_extractor import TextExtractor

SUMMARY_CHARACTERS = 2000  # enough for the embedding pre-filter to see more than the opening lines

_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
PODCAST_FEED_SHARE = 0.8  # of a feed's entries with audio, for it to count as a podcast
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac")


class ParsedFeed:
    """A feed's title and its entries."""

    def __init__(self, title: str, entries: list[Entry]):
        self.title = title
        self.entries = entries

    def __repr__(self) -> str:
        return f"ParsedFeed({self.title!r}, {len(self.entries)} entries)"

    @classmethod
    def parse(cls, data: bytes) -> "ParsedFeed":
        """Parse a feed body. Tolerates common breakage in real-world feeds.

        Entries that are podcast episodes keep their `audio_url` (see `_mark_podcast_episodes`).
        """
        parsed = cls._parse_any(data)
        _mark_podcast_episodes(parsed.entries)
        return parsed

    @classmethod
    def _parse_any(cls, data: bytes) -> "ParsedFeed":
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
                parsed = cls._parse_leniently(text)
                if not parsed.entries:
                    raise first
                return parsed
        if _local(root.tag) == "feed":
            return cls._parse_atom(root)
        return cls._parse_rss(root)

    @classmethod
    def _parse_atom(cls, root: ElementTree.Element) -> "ParsedFeed":
        entries: list[Entry] = []
        for entry in root:
            if _local(entry.tag) != "entry":
                continue
            url = audio_url = ""
            for link in entry:
                if _local(link.tag) != "link":
                    continue
                relation = link.get("rel", "alternate")
                if relation == "alternate" and not url:
                    url = link.get("href", "")
                elif relation == "enclosure" and not audio_url and _is_audio(link.get("type"), link.get("href", "")):
                    audio_url = link.get("href", "").strip()
            url = url or audio_url
            title = TextExtractor.extract(_text(entry, "title"))
            date = _text(entry, "published") or _text(entry, "updated")
            if url and title:
                summary = _summarise(_text(entry, "summary"), _text(entry, "content"))
                entries.append(Entry(url, title, summary, _parse_date(date), audio_url,
                                     _parse_duration(_text(entry, "duration")) if audio_url else None))
        return cls(TextExtractor.extract(_text(root, "title")), entries)

    @classmethod
    def _parse_rss(cls, root: ElementTree.Element) -> "ParsedFeed":
        """RSS 2.0 (<rss><channel><item>) or RSS 1.0 (<rdf:RDF><item>)."""
        channel = _child(root, "channel")
        feed_title = _text(channel, "title") if channel is not None else ""
        entries: list[Entry] = []
        for item in root.iter():
            if _local(item.tag) != "item":
                continue
            url = _text(item, "link")
            if not url:
                guid = _child(item, "guid")
                if guid is not None and guid.get("isPermaLink", "true") != "false":
                    url = (guid.text or "").strip()
                    if not url.startswith(("http://", "https://")):
                        url = ""  # podcast feeds often use an opaque id without isPermaLink="false"
            audio_url = _rss_audio_url(item)
            url = url or audio_url
            title = TextExtractor.extract(_text(item, "title"))
            date = _text(item, "pubDate", "date", "published", "updated")
            if url and title:
                summary = _summarise(_text(item, "description"), _text(item, "encoded"), _text(item, "summary"))
                entries.append(Entry(url, title, summary, _parse_date(date), audio_url,
                                     _parse_duration(_text(item, "duration")) if audio_url else None))
        return cls(TextExtractor.extract(feed_title), entries)

    @classmethod
    def _parse_leniently(cls, text: str) -> "ParsedFeed":
        """Last resort for badly broken feeds: pull fields out with regular expressions."""
        entries = []
        for block in re.findall(r"<(?:item|entry)(?:\s[^>]*)?>(.*?)</(?:item|entry)>", text, re.S | re.I):
            url = _tag_text(block, "link")
            if not url:
                match = re.search(r'<link[^>]*href=["\']([^"\']+)', block, re.I)
                url = match.group(1) if match else ""
            audio_url = _lenient_audio_url(block)
            url = url.strip() or audio_url
            title = TextExtractor.extract(_tag_text(block, "title"))
            if url and title:
                entries.append(Entry(url, title,
                                     _summarise(*(_tag_text(block, name) for name in
                                                  ("description", "summary", "content", "content:encoded"))),
                                     _parse_date(_tag_text(block, "pubDate", "published", "updated", "dc:date")),
                                     audio_url,
                                     _parse_duration(_tag_text(block, "itunes:duration")) if audio_url else None))
        channel_title = re.search(r"<title(?:\s[^>]*)?>(.*?)</title>", text, re.S | re.I)
        return cls(TextExtractor.extract(channel_title.group(1)) if channel_title else "", entries)


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


def _mark_podcast_episodes(entries: list[Entry]) -> None:
    """Clear `audio_url` on entries with audio that aren't podcast episodes.

    Many article feeds attach audio to some posts: a read-aloud version of the article, or a
    clip it embeds. Podcast episodes are told apart by an `itunes:duration`, or by coming from
    a feed in which most entries have audio (some podcast hosts leave the duration out).
    """
    podcast_feed = sum(bool(entry.audio_url) for entry in entries) >= PODCAST_FEED_SHARE * max(1, len(entries))
    for entry in entries:
        if entry.audio_url and entry.duration is None and not podcast_feed:
            entry.audio_url = ""


def _is_audio(media_type: str | None, url: str) -> bool:
    """Whether an enclosure is audio: by its MIME type, or by the file extension if it has none."""
    if media_type:
        return media_type.strip().lower().startswith("audio/")
    path = url.split("?", 1)[0].lower()
    return path.endswith(AUDIO_EXTENSIONS)


def _rss_audio_url(item: ElementTree.Element) -> str:
    """The URL of an RSS item's first audio `<enclosure>`, or ''."""
    for child in item:
        if _local(child.tag) == "enclosure" and _is_audio(child.get("type"), child.get("url", "")):
            return child.get("url", "").strip()
    return ""


def _lenient_audio_url(block: str) -> str:
    """The first audio enclosure (RSS `<enclosure url>` or Atom `<link rel="enclosure" href>`) in feed XML."""
    for tag_name, attributes_text in re.findall(r"<(enclosure|link)(\s[^>]*)>", block, re.I):
        attributes = {name.lower(): html.unescape(value)
                      for name, _, value in re.findall(r'([\w:-]+)\s*=\s*(["\'])(.*?)\2', attributes_text)}
        if tag_name.lower() == "link" and attributes.get("rel") != "enclosure":
            continue
        address = (attributes.get("url") or attributes.get("href") or "").strip()
        if address and _is_audio(attributes.get("type"), address):
            return address
    return ""


def _parse_duration(text: str) -> float | None:
    """Seconds in an `itunes:duration`: plain seconds, MM:SS or HH:MM:SS."""
    parts = (text or "").strip().split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


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


def _summarise(*markups: str) -> str:
    """Plain text of the longest of an entry's text fields, cut at a word boundary.

    Many feeds have a one-line description next to the whole article (RSS
    `content:encoded`, Atom `content`); the longer one tells the pre-filter more.
    """
    text = max((TextExtractor.extract(markup) for markup in markups if markup), key=len, default="")
    return text if len(text) <= SUMMARY_CHARACTERS else text[: SUMMARY_CHARACTERS - 1].rsplit(" ", 1)[0] + "…"


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
