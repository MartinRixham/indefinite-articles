"""The questions the service asks the LLM, as domain operations.

`LLM` is the transport (streaming, stall limits, JSON extraction); `Assistant` owns the
prompts and turns the answers into domain values. Each method raises `LLMError` when the
LLM is unavailable or its answer can't be used, and leaves the fallback to the caller.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..mail.interpretation import Interpretation, Rating
from ..storage.article import Article
from ..storage.database import PendingFeedback
from .llm import LLM, LLMError

MAX_PHRASES = 12  # per side, from the reader profile

SCORE_SYSTEM = (
    "You are a personal reading assistant. You predict how much one specific reader will "
    "want to read an article, based on what you know about their tastes. Be discerning: "
    "use the full 0-10 range, and reserve 8+ for articles that clearly match their interests."
)

SCALE = (
    "Score on this fixed scale, the same for every article:\n"
    "0-1: the reader would actively avoid it (a topic or style they dislike)\n"
    "2-3: unlikely to interest them\n"
    "4-5: loosely related to their interests, or thin and generic\n"
    "6-7: a good match they would probably read\n"
    "8-9: a strong match on both topic and depth; among the best articles of the day\n"
    "10: exceptional; exactly what they want most\n"
    "Most articles are not for this reader, so most scores should be 5 or below."
)

PROFILE_SYSTEM = (
    "You maintain a concise profile of one reader's reading preferences, used to choose "
    "articles for them. You update it from their feedback."
)

INTERPRET_SYSTEM = (
    "You interpret a reader's email reply to a numbered list of recommended articles. "
    "Work out which articles they liked or disliked and what general preferences they expressed."
)

PREFERENCES_SYSTEM = "You turn a reader's profile into short phrases for a search engine."
PREFERENCES_PROMPT = (
    "Reader profile:\n{profile}\n\n"
    "List the kinds of article this reader wants, and separately the kinds they want to "
    "avoid, as short positive descriptions of article topics, angles or styles, e.g. "
    '"in-depth articles on Rust compiler internals" or "celebrity gossip". Never use '
    'negations such as "no", "not" or "avoid" inside a phrase: put unwanted things in '
    "the avoid list instead. At most {max_phrases} phrases per list.\n"
    'Respond with JSON only: {{"likes": ["..."], "avoid": ["..."]}}'
)


@dataclass
class ArticleScore:
    """The LLM's 0-10 score for one article, and its reason."""
    score: float
    reason: str


def _format_feedback(rows: Iterable[sqlite3.Row]) -> str:
    return "\n".join(
        f"- {row['title']} ({row['source']})" + (f' — "{row["comment"]}"' if row["comment"] else "")
        for row in rows
    ) or "(none)"


def _phrase_list(data: dict, key: str) -> list[str]:
    items = data.get(key)
    if not isinstance(items, list):
        return []
    return [str(item).strip()[:200] for item in items if str(item).strip()][:MAX_PHRASES]


class Assistant:
    """Asks the LLM about articles, replies and the reader profile."""

    def __init__(self, llm: LLM):
        self._llm = llm

    @staticmethod
    def score_prefix(profile_text: str, examples: str) -> str:
        """The part of the scoring prompt that is the same for every article.

        It comes first, so Ollama and llama-server can reuse its KV cache and only
        process the article itself on each request.
        """
        return (
            f"Reader profile:\n{profile_text}\n\n{examples}\n\n{SCALE}\n\n"
            'Respond with JSON only, in the form {"reason": "why, in at most 20 words", '
            '"score": 7}. Write the reason first, then the score.\n\n'
        )

    def score_article(self, prefix: str, article: Article, per_article: int = 1500) -> ArticleScore:
        """Score a single article from 0 to 10 for the reader described in `prefix`
        (made by `score_prefix`), showing the LLM at most `per_article` characters of it."""
        user = (
            prefix + "Article to rate:\n"
            f"Title: {article.title}\nSource: {article.source_label}\n{article.description(per_article)}"
        )
        data = self._llm.chat_json(SCORE_SYSTEM, user)
        if isinstance(data, dict) and isinstance(data.get("ratings"), list) and data["ratings"]:
            data = data["ratings"][0]  # some models keep the old batch shape
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            raise LLMError(f"no score in LLM output: {str(data)[:200]}")
        try:
            score = max(0.0, min(10.0, float(data["score"])))
        except (KeyError, TypeError, ValueError):
            raise LLMError(f"no score in LLM output: {str(data)[:200]}") from None
        return ArticleScore(score, str(data.get("reason", ""))[:200])

    def rewrite_profile(self, profile_text: str, pending: PendingFeedback) -> str:
        """The reader profile rewritten to take in the feedback received since it was made."""
        liked = [row for row in pending.feedback if row["rating"] > 0]
        disliked = [row for row in pending.feedback if row["rating"] < 0]
        neutral = [row for row in pending.feedback if row["rating"] == 0]
        user = (
            f"Current reader profile:\n{profile_text}\n\n"
            f"New feedback from the reader.\nLiked:\n{_format_feedback(liked)}\n\n"
            f"Disliked:\n{_format_feedback(disliked)}\n\n"
            f"Comments on specific articles (no explicit like/dislike):\n{_format_feedback(neutral)}\n\n"
            "General comments from the reader:\n"
            + ("\n".join(f'- "{comment["comment"]}"' for comment in pending.comments) or "(none)")
            + "\n\nRewrite the profile to incorporate this feedback. Keep established preferences "
            "unless the new feedback contradicts them; treat explicit comments as strong signals "
            "and single likes/dislikes as weak signals. Describe topics, angles, depth and styles "
            "the reader likes and dislikes, and any sources they favour or avoid. "
            "Write at most 250 words as a bullet list. Output only the profile."
        )
        text = self._llm.chat(PROFILE_SYSTEM, user).strip()
        if len(text) < 20:
            raise LLMError("profile update returned implausibly short text")
        return text

    def interpret_reply(self, text: str, items: Sequence[Article]) -> Interpretation:
        """What a free-form reply to a digest of `items` says, item numbers counting from 1."""
        listing = "\n".join(f"{item_number}. {article.title} ({article.source})"
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
        data = self._llm.chat_json(INTERPRET_SYSTEM, user)
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

    def profile_phrases(self, profile_text: str) -> tuple[list[str], list[str]]:
        """Short phrases for what the profile says the reader wants and wants to avoid,
        at most `MAX_PHRASES` of each. Raises `LLMError` if there are no wanted phrases."""
        prompt = PREFERENCES_PROMPT.format(profile=profile_text[:4000], max_phrases=MAX_PHRASES)
        data = self._llm.chat_json(PREFERENCES_SYSTEM, prompt)
        if not isinstance(data, dict):
            raise LLMError(f"no phrases in LLM output: {str(data)[:200]}")
        likes, avoid = _phrase_list(data, "likes"), _phrase_list(data, "avoid")
        if not likes:
            raise LLMError(f"no wanted phrases in LLM output: {str(data)[:200]}")
        return likes, avoid
