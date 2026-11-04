"""One configured feed: where to fetch it and how to label its articles."""

import urllib.parse
import xml.etree.ElementTree as ElementTree

DEFAULT_MAX_ITEMS = 40  # newest entries kept per fetch


class Feed:
    """A feed from the config file or the feeds directory.

    `name` overrides the feed's own title as the source of its articles, `category`
    is shown next to the source, and `max_items` limits the newest entries kept per fetch.
    """

    def __init__(self, url: str, name: str = "", category: str = "", max_items: int = DEFAULT_MAX_ITEMS):
        self.url = url
        self.name = name
        self.category = category
        self.max_items = max_items

    def __repr__(self) -> str:
        return f"Feed({self.url!r})"

    @classmethod
    def from_table(cls, table: dict) -> "Feed":
        """A `[[feeds]]` entry from the config file."""
        return cls(url=table["url"], name=table.get("name", ""), category=table.get("category", ""),
                   max_items=int(table.get("max_items", DEFAULT_MAX_ITEMS)))

    @classmethod
    def from_outline(cls, outline: ElementTree.Element, category: str) -> "Feed":
        """An OPML `<outline>` with an `xmlUrl`; an unreadable `maxItems` falls back to the default."""
        try:
            max_items = int(outline.get("maxItems", DEFAULT_MAX_ITEMS))
        except ValueError:
            max_items = DEFAULT_MAX_ITEMS
        return cls(url=outline.get("xmlUrl", "").strip(),
                   name=(outline.get("title") or outline.get("text") or "").strip(),
                   category=category, max_items=max_items)

    @classmethod
    def from_line(cls, line: str, category: str) -> "Feed":
        """A `URL [name]` line from a plain feed list."""
        url, _, name = line.strip().partition(" ")
        return cls(url=url, name=name.strip(), category=category)

    def source(self, title: str) -> str:
        """The source name its articles are stored under: its name, else the feed's own `title`, else its URL."""
        return self.name or title or self.url

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.url).hostname or ""
