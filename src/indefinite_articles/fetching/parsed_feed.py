"""Parse RSS 2.0 / RSS 1.0 (RDF) / Atom feeds using only the standard library."""

import email.utils
import html
import html.entities
import re
import xml.etree.ElementTree as ElementTree
from datetime import datetime

from .entry import Entry
from .text_extractor import TextExtractor

SUMMARY_CHARACTERS = 600

_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


class ParsedFeed:
    """A feed's title and its entries."""

    def __init__(self, title: str, entries: list[Entry]):
        self.title = title
        self.entries = entries

    def __repr__(self) -> str:
        return f"ParsedFeed({self.title!r}, {len(self.entries)} entries)"

    @classmethod
    def parse(cls, data: bytes) -> "ParsedFeed":
        """Parse a feed body. Tolerates common breakage in real-world feeds."""
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
            url = ""
            for link in entry:
                if _local(link.tag) == "link" and link.get("rel", "alternate") == "alternate":
                    url = link.get("href", "")
                    break
            title = TextExtractor.extract(_text(entry, "title"))
            summary = _text(entry, "summary") or _text(entry, "content")
            date = _text(entry, "published") or _text(entry, "updated")
            if url and title:
                entries.append(Entry(url, title, _summarise(summary), _parse_date(date)))
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
            title = TextExtractor.extract(_text(item, "title"))
            summary = _text(item, "description") or _text(item, "encoded", "summary")
            date = _text(item, "pubDate", "date", "published", "updated")
            if url and title:
                entries.append(Entry(url, title, _summarise(summary), _parse_date(date)))
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
            title = TextExtractor.extract(_tag_text(block, "title"))
            if url and title:
                entries.append(Entry(url.strip(), title,
                                     _summarise(_tag_text(block, "description", "summary", "content")),
                                     _parse_date(_tag_text(block, "pubDate", "published", "updated", "dc:date"))))
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
    text = TextExtractor.extract(markup)
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
