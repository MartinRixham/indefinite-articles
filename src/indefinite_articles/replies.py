"""Understanding the reader's email replies."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from email.message import EmailMessage

from .database import ArticleRow
from .feeds import html_to_text
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


@dataclass(frozen=True)
class Rating:
    """The reader's verdict on one article: 1 liked, -1 disliked, 0 just a comment."""
    rating: int
    comment: str = ""

    @property
    def label(self) -> str:
        return {1: "👍 liked", -1: "👎 disliked", 0: "💬 noted"}[self.rating]


@dataclass
class Sentiment:
    """A like/dislike word found at the start of a line (None if there was none), and the rest."""
    rating: int | None
    comment: str


@dataclass
class Interpretation:
    """What a reply says: ratings by item number, and comments not about one article."""
    ratings: dict[int, Rating] = field(default_factory=dict)
    general: list[str] = field(default_factory=list)

    def filled_in_by(self, other: "Interpretation") -> "Interpretation":
        """These ratings, plus `other`'s for items these leave out or only comment on;
        `other`'s general comments."""
        ratings = dict(self.ratings)
        for item_number, rating in other.ratings.items():
            if item_number not in ratings or ratings[item_number].rating == 0:
                ratings[item_number] = rating
        return Interpretation(ratings, other.general)


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


def parse_structured(text: str, item_count: int) -> StructuredReply:
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


INTERPRET_SYSTEM = (
    "You interpret a reader's email reply to a numbered list of recommended articles. "
    "Work out which articles they liked or disliked and what general preferences they expressed."
)


def interpret_with_llm(llm: LLM, text: str, items: Sequence[ArticleRow]) -> Interpretation:
    listing = "\n".join(f"{item_number}. {article['title']} ({article['source']})"
                        for item_number, article in enumerate(items, start=1))
    user = (
        f"The reader was sent these articles:\n{listing}\n\n"
        f"Their reply:\n\"\"\"\n{text[:4000]}\n\"\"\"\n\n"
        "Respond with JSON only, in the form "
        '{"ratings": [{"item": 1, "rating": "like" or "dislike" or "neutral", '
        '"comment": "what they said about it, or empty"}], '
        '"general_comment": "any preference not tied to one article, or empty"}. '
        "Only include articles the reader actually referred to."
    )
    data = llm.chat_json(INTERPRET_SYSTEM, user)
    if not isinstance(data, dict):
        raise LLMError("unexpected interpretation format")
    ratings: dict[int, Rating] = {}
    for entry in data.get("ratings") or []:
        try:
            item_number = int(entry["item"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= item_number <= len(items):
            continue
        rating = {"like": 1, "dislike": -1}.get(str(entry.get("rating", "")).lower(), 0)
        ratings[item_number] = Rating(rating, str(entry.get("comment") or "")[:500])
    general = str(data.get("general_comment") or "").strip()
    return Interpretation(ratings, [general] if general else [])


def understand(llm: LLM | None, text: str, items: Sequence[ArticleRow]) -> Interpretation:
    """Combine the structured parser with LLM interpretation for free-form replies."""
    parsed = parse_structured(text, len(items))
    structured = Interpretation(parsed.ratings, parsed.general)
    if llm is None or not (parsed.unresolved or parsed.general):
        return structured
    try:
        # Explicit structured ratings win; the LLM fills in the rest.
        return structured.filled_in_by(interpret_with_llm(llm, text, items))
    except LLMError as error:
        log.warning("could not interpret free-form reply with LLM: %s", error)
        return structured
