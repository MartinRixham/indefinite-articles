"""An article stored in the database, and what the rest of the pipeline needs to know about it."""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, fields

EMBEDDING_SUMMARY_CHARACTERS = 2000  # of the summary, embedded after the title
BLURB_CHARACTERS = 280              # shown under each digest item
TITLE_KEY_CHARACTERS = 120


def title_key(title: str) -> str:
    """Normalised title used to de-duplicate the same story from different feeds."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())[:TITLE_KEY_CHARACTERS]


def _shorten(text: str, limit: int) -> str:
    """`text` with its whitespace collapsed, cut at a word boundary to at most `limit` characters."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


@dataclass(frozen=True)
class Article:
    """One row of the `articles` table.

    `audio_url` (and `duration`, in seconds, when the feed gives it) are set for podcast
    episodes, and `apple_url` once the episode has been found on Apple Podcasts. `feed_url`
    is the feed the article came from ('' for articles stored before it was recorded). `content` and `content_status` are set once the full page text has been fetched
    (or has failed); `score`, `reason` and `score_profile_id` once the LLM has scored
    it against a reader profile; `prefilter` once the embedding pre-filter has run.
    """
    id: int
    url: str
    title: str
    summary: str = ""
    source: str = ""
    category: str = ""
    published: float | None = None
    fetched_at: float = 0.0
    content: str | None = None
    content_status: str | None = None
    score: float | None = None
    reason: str | None = None
    score_profile_id: int | None = None
    prefilter: float | None = None
    audio_url: str | None = None
    duration: float | None = None
    feed_url: str = ""
    apple_url: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Article":
        """The article in a row with (at least) every column of `articles`; other columns are ignored."""
        return cls(**{name: row[name] for name in _FIELD_NAMES})

    @property
    def source_label(self) -> str:
        """The source, followed by its category in brackets when it has one."""
        return f"{self.source} ({self.category})" if self.category else self.source

    @property
    def is_podcast(self) -> bool:
        """Whether this is a podcast episode (its feed entry had an audio enclosure)."""
        return bool(self.audio_url)

    @property
    def length_label(self) -> str:
        """A podcast episode's length, e.g. "45 min" or "1 h 20 min"; '' when unknown."""
        if not self.duration:
            return ""
        minutes = max(1, round(self.duration / 60))
        hours, minutes = divmod(minutes, 60)
        if not hours:
            return f"{minutes} min"
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"

    @property
    def awaiting_full_text(self) -> bool:
        """True until the full page text has been fetched, or fetching it has failed."""
        return self.content_status is None

    def scored_against(self, profile_id: int | None) -> bool:
        """Whether the LLM has scored this article against the given reader profile."""
        return self.score is not None and self.score_profile_id == profile_id

    def has_stale_score(self, profile_id: int | None) -> bool:
        """Whether the article's LLM score was made against a profile other than the given one."""
        return self.score is not None and self.score_profile_id != profile_id

    def description(self, limit: int) -> str:
        """The best available text about the article, for the LLM: the page text if it was
        fetched and is no shorter than the feed summary, else the summary."""
        if self.content and len(self.content) >= len(self.summary):
            return _shorten(self.content, limit)
        return _shorten(self.summary, limit)

    def blurb(self, limit: int = BLURB_CHARACTERS) -> str:
        """A short description for the digest: the feed summary, or else the start of the page text."""
        return _shorten(self.summary or self.content or "", limit)

    def template_values(self, escape: Callable[[str], str]) -> dict[str, str]:
        """The digest item placeholders that describe the article itself, passed through `escape`."""
        return {
            "title": escape(self.title),
            "source": escape(self.source),
            "url": escape(self.url),
            "reason": escape(self.reason or ""),
            "blurb": escape(self.blurb()),
            "length": escape(self.length_label),
            # Left out when the episode has no page of its own, so `url` already is the audio.
            "audio_url": escape(self.audio_url) if self.audio_url and self.audio_url != self.url else "",
            "apple_url": escape(self.apple_url or ""),
        }

    def embedding_text(self) -> str:
        """What the embedding pre-filter embeds: the title and the start of the summary."""
        return f"{self.title}. {self.summary[:EMBEDDING_SUMMARY_CHARACTERS]}"


_FIELD_NAMES = tuple(field.name for field in fields(Article))
