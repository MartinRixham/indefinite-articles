"""An entry read from a feed, before it is stored as an article."""

from dataclasses import dataclass

from .article import title_key


@dataclass
class Entry:
    url: str
    title: str
    summary: str
    published: float | None

    @property
    def title_key(self) -> str:
        return title_key(self.title)
