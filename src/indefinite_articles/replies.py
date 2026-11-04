"""Understanding the reader's email replies."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from email.message import EmailMessage

from .article import Article
from .assistant import Assistant
from .feeds import html_to_text
from .interpretation import Interpretation, Rating
from .llm import LLM, LLMError

log = logging.getLogger(__name__)

LIKE_WORDS = {"+", "y", "yes", "like", "liked", "love", "loved", "good", "great", "more",
              "👍", "❤️", "❤", "interesting", "read", "keep"}
DISLIKE_WORDS = {"-", "n", "no", "nope", "dislike", "disliked", "bad", "meh", "less", "boring",
                 "👎", "skip", "nah"}

_QUOTE_HEADER = re.compile(
    r"^(On\s.+wrote:|On\s.+|-{2,}\s*Original Message\s*-{2,}|_{5,}|From:\s.+|Sent from my .+)$",
    re.IGNORECASE,
)
_COMPACT_TOKEN = re.compile(r"([+-])\s?(\d{1,2})|(\d{1,2})\s?([+-])")
_ITEM_LINE = re.compile(r"^\s*#?(\d{1,2})\s*[.):,]?\s*(.*)$")


@dataclass
class Sentiment:
    """A like/dislike word found at the start of a line (None if there was none), and the rest."""
    rating: int | None
    comment: str


@dataclass
class StructuredReply:
    """Ratings and general comments from the documented formats, and the number of item
    lines whose sentiment was unclear."""
    ratings: dict[int, Rating]
    general: list[str]
    unresolved: int


def body_text(message: EmailMessage) -> str:
    part = message.get_body(preferencelist=("plain",))
    if part is not None:
        return part.get_content()
    part = message.get_body(preferencelist=("html",))
    if part is not None:
        markup = re.sub(r"(?is)<blockquote.*?</blockquote>", "", part.get_content())
        markup = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", markup)
        return "\n".join(html_to_text(line) for line in markup.splitlines())
    return ""


def strip_quoted(text: str) -> str:
    """Return only the newly written part of a reply."""
    out = []
    lines = text.replace("\r\n", "\n").split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(">") or stripped == "--" or stripped == "-- ":
            break
        if _QUOTE_HEADER.match(stripped):
            # "On <date>, X wrote:" may be wrapped; only stop if it looks like a header.
            lookahead = " ".join(following.strip() for following in lines[index : index + 3])
            if stripped.lower().startswith("on ") and "wrote" not in lookahead.lower():
                out.append(line)
                continue
            break
        if stripped.startswith("digest-id:"):
            break
        out.append(line)
    return "\n".join(out).strip()


def _sentiment(rest: str) -> Sentiment:
    """Look at the start of `rest` for a like/dislike word."""
    rest = rest.strip()
    match = re.match(r"^([+-]|[^\W\d_]+|👍|👎|❤️?)\s*[:,.!\-–—]*\s*(.*)$", rest, re.DOTALL)
    if match:
        word = match.group(1).lower()
        if word in LIKE_WORDS:
            return Sentiment(1, match.group(2).strip())
        if word in DISLIKE_WORDS:
            return Sentiment(-1, match.group(2).strip())
    return Sentiment(None, rest)


def _parse_structured(text: str, item_count: int) -> StructuredReply:
    """Parse ratings written in the documented formats."""
    ratings: dict[int, Rating] = {}
    general: list[str] = []
    unresolved = 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # Compact form: "+1 +4 -2", "1+ 3-", "+1, -2"
        stripped = _COMPACT_TOKEN.sub("", line)
        if _COMPACT_TOKEN.search(line) and not re.sub(r"[\s,;]", "", stripped):
            for match in _COMPACT_TOKEN.finditer(line):
                sign, number = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
                item_number = int(number)
                if 1 <= item_number <= item_count:
                    ratings[item_number] = Rating(1 if sign == "+" else -1)
            continue
        match = _ITEM_LINE.match(line)
        if match and 1 <= int(match.group(1)) <= item_count:
            item_number = int(match.group(1))
            sentiment = _sentiment(match.group(2))
            if sentiment.rating is None:
                unresolved += 1
                ratings[item_number] = Rating(0, sentiment.comment)
            else:
                ratings[item_number] = Rating(sentiment.rating, sentiment.comment)
            continue
        general.append(line)
    return StructuredReply(ratings, general, unresolved)


def understand(llm: LLM | None, text: str, items: Sequence[Article]) -> Interpretation:
    """Combine the structured parser with LLM interpretation for free-form replies."""
    parsed = _parse_structured(text, len(items))
    structured = Interpretation(parsed.ratings, parsed.general)
    if llm is None or not (parsed.unresolved or parsed.general):
        return structured
    try:
        # Explicit structured ratings win; the LLM fills in the rest.
        return structured.filled_in_by(Assistant(llm).interpret_reply(text, items))
    except LLMError as error:
        log.warning("could not interpret free-form reply with LLM: %s", error)
        return structured
