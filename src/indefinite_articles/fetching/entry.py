"""An entry read from a feed, before it is stored as an article."""

from dataclasses import dataclass

from ..storage.article import title_key


@dataclass
class Entry:
    url: str
    title: str
    summary: str
    published: float | None
    audio_url: str = ""          # set for podcast episodes: the audio enclosure
    duration: float | None = None  # podcast episode length in seconds, when the feed gives it

    @property
    def title_key(self) -> str:
        return title_key(self.title)
