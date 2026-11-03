"""Understanding the reader's email replies."""

from __future__ import annotations

import logging
import re
from email.message import EmailMessage

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


def body_text(msg: EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain",))
    if part is not None:
        return part.get_content()
    part = msg.get_body(preferencelist=("html",))
    if part is not None:
        h = re.sub(r"(?is)<blockquote.*?</blockquote>", "", part.get_content())
        h = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", h)
        return "\n".join(html_to_text(line) for line in h.splitlines())
    return ""


def strip_quoted(text: str) -> str:
    """Return only the newly written part of a reply."""
    out = []
    lines = text.replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(">") or s == "--" or s == "-- ":
            break
        if _QUOTE_HEADER.match(s):
            # "On <date>, X wrote:" may be wrapped; only stop if it looks like a header.
            nxt = " ".join(l.strip() for l in lines[i : i + 3])
            if s.lower().startswith("on ") and "wrote" not in nxt.lower():
                out.append(line)
                continue
            break
        if s.startswith("digest-id:"):
            break
        out.append(line)
    return "\n".join(out).strip()


def _sentiment(rest: str) -> tuple[int | None, str]:
    """Look at the start of `rest` for a like/dislike word. Returns (rating, remaining comment)."""
    rest = rest.strip()
    m = re.match(r"^([+-]|[^\W\d_]+|👍|👎|❤️?)\s*[:,.!\-–—]*\s*(.*)$", rest, re.DOTALL)
    if m:
        word = m.group(1).lower()
        if word in LIKE_WORDS:
            return 1, m.group(2).strip()
        if word in DISLIKE_WORDS:
            return -1, m.group(2).strip()
    return None, rest


def parse_structured(text: str, n_items: int):
    """Parse ratings written in the documented formats.

    Returns (ratings, general_comments, unresolved) where ratings maps item number to
    (rating, comment), and unresolved counts item lines whose sentiment was unclear.
    """
    ratings: dict[int, tuple[int, str]] = {}
    general: list[str] = []
    unresolved = 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # Compact form: "+1 +4 -2", "1+ 3-", "+1, -2"
        stripped = _COMPACT_TOKEN.sub("", line)
        if _COMPACT_TOKEN.search(line) and not re.sub(r"[\s,;]", "", stripped):
            for m in _COMPACT_TOKEN.finditer(line):
                sign, num = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
                n = int(num)
                if 1 <= n <= n_items:
                    ratings[n] = (1 if sign == "+" else -1, "")
            continue
        m = _ITEM_LINE.match(line)
        if m and 1 <= int(m.group(1)) <= n_items:
            n = int(m.group(1))
            rating, comment = _sentiment(m.group(2))
            if rating is None:
                unresolved += 1
                ratings[n] = (0, comment)
            else:
                ratings[n] = (rating, comment)
            continue
        general.append(line)
    return ratings, general, unresolved


INTERPRET_SYSTEM = (
    "You interpret a reader's email reply to a numbered list of recommended articles. "
    "Work out which articles they liked or disliked and what general preferences they expressed."
)


def interpret_with_llm(llm: LLM, text: str, items):
    listing = "\n".join(f"{i}. {a['title']} ({a['source']})" for i, a in enumerate(items, start=1))
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
    ratings: dict[int, tuple[int, str]] = {}
    for r in data.get("ratings") or []:
        try:
            n = int(r["item"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= n <= len(items):
            continue
        val = {"like": 1, "dislike": -1}.get(str(r.get("rating", "")).lower(), 0)
        ratings[n] = (val, str(r.get("comment") or "")[:500])
    general = str(data.get("general_comment") or "").strip()
    return ratings, ([general] if general else [])


def understand(llm: LLM | None, text: str, items):
    """Combine the structured parser with LLM interpretation for free-form replies."""
    ratings, general, unresolved = parse_structured(text, len(items))
    needs_llm = unresolved > 0 or (general and llm is not None)
    if needs_llm and llm is not None:
        try:
            l_ratings, l_general = interpret_with_llm(llm, text, items)
            # Explicit structured ratings win; the LLM fills in the rest.
            for n, v in l_ratings.items():
                if n not in ratings or ratings[n][0] == 0:
                    ratings[n] = v
            general = l_general
        except LLMError as e:
            log.warning("could not interpret free-form reply with LLM: %s", e)
    return ratings, general
