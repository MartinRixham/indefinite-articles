"""Turn HTML into plain text."""

import html
import re
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """Collects the text of an HTML fragment, leaving out scripts and styles."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    @classmethod
    def extract(cls, markup: str) -> str:
        """The text of `markup`, with entities decoded and whitespace collapsed."""
        if not markup:
            return ""
        parser = cls()
        try:
            parser.feed(markup)
            parser.close()
            text = "".join(parser.parts)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", markup)
        return _tidy(text)

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


def _tidy(text: str) -> str:
    """Decode entities and collapse runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
