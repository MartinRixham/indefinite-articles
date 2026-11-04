"""Scoring articles against the learned reader profile, and learning from feedback."""

import bisect
import hashlib
import logging
import random
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from .article import Article
from .config import Config
from .database import Database, SourceStatistics
from .llm import LLM, LLMError, LLMUnavailable

log = logging.getLogger(__name__)

DEFAULT_PROFILE = (
    "No preferences learned yet. Prefer substantive, well-written, informative articles "
    "over clickbait, press releases, and thin news items."
)

SCORE_SYSTEM = (
    "You are a personal reading assistant. You predict how much one specific reader will "
    "want to read an article, based on what you know about their tastes. Be discerning: "
    "use the full 0-10 range, and reserve 8+ for articles that clearly match their interests."
)

PROFILE_SYSTEM = (
    "You maintain a concise profile of one reader's reading preferences, used to choose "
    "articles for them. You update it from their feedback."
)


def ensure_profile(config: Config, database: Database) -> sqlite3.Row:
    row = database.current_profile()
    if row is None:
        text = config.interests or DEFAULT_PROFILE
        database.add_profile(text, note="initial (from config)")
        row = database.current_profile()
        assert row is not None  # just inserted
    return row


def _format_examples(rows: Iterable[sqlite3.Row]) -> str:
    out = []
    for row in rows:
        line = f"- {row['title']} ({row['source']})"
        if row["comment"]:
            line += f' — reader said: "{row["comment"][:150]}"'
        out.append(line)
    return "\n".join(out) or "(none yet)"


def _examples(database: Database) -> str:
    return (
        "Articles the reader recently LIKED:\n" + _format_examples(database.rated_examples(1, 15)) +
        "\n\nArticles the reader recently DISLIKED:\n" + _format_examples(database.rated_examples(-1, 15))
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


def _score_prefix(profile_text: str, examples: str) -> str:
    """The part of the scoring prompt that is the same for every article.

    It comes first, so Ollama and llama-server can reuse its KV cache and only
    process the article itself on each request.
    """
    return (
        f"Reader profile:\n{profile_text}\n\n{examples}\n\n{SCALE}\n\n"
        'Respond with JSON only, in the form {"reason": "why, in at most 20 words", '
        '"score": 7}. Write the reason first, then the score.\n\n'
    )


class JSONChat(Protocol):
    def chat_json(self, system: str, user: str) -> Any: ...


@dataclass
class ArticleScore:
    """The LLM's 0-10 score for one article, and its reason."""
    score: float
    reason: str


@dataclass
class DigestPick:
    """An article chosen for the digest; exploratory picks are random, to keep learning."""
    article: Article
    exploratory: bool

    def template_values(self, position: int, escape: Callable[[str], str], exploratory_marker: str) -> dict[str, str]:
        """The `$placeholders` of the digest's `item` template section for this pick."""
        return {
            "position": str(position),
            "exploratory": exploratory_marker if self.exploratory else "",
            **self.article.template_values(escape),
        }

    def debug_line(self, position: int, score: "ScoreBreakdown", rank: int) -> str:
        """This pick's row in the digest's debug table."""
        article = self.article
        llm_score = f"{score.llm:5.1f}" if score.llm is not None else "    -"
        return (
            f"{position:<3} {rank:>4}  {score.final:5.2f} = {llm_score} {score.base:+5.2f}"
            f" {score.bonus:+5.2f}   {(article.content_status or '-')[:8]:8}  {article.source[:30]}"
            + ("  [explore]" if self.exploratory else "")
        )


@dataclass
class ScoreBreakdown:
    """How an article's final ranking score was made up.

    `llm` is the LLM's score (None when it hasn't scored the article); `base` is the
    embedding agreement added to it, or the fallback score from the pre-filter rank;
    `bonus` comes from the reader's ratings of the source.
    """
    llm: float | None
    base: float
    bonus: float
    final: float


def _score_one(llm: JSONChat, prefix: str, article: Article, per_article: int = 1500) -> ArticleScore:
    """Score a single article from 0 to 10 for the reader described in `prefix`."""
    user = (
        prefix + "Article to rate:\n"
        f"Title: {article.title}\nSource: {article.source_label}\n{article.description(per_article)}"
    )
    data = llm.chat_json(SCORE_SYSTEM, user)
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


def shortlist(config: Config, database: Database) -> list[Article]:
    """The articles worth spending LLM time on today.

    Uses the embedding pre-filter when available: the best matches (with a per-source
    cap so no single feed dominates) plus ~10% random picks for exploration. Without
    embeddings, falls back to the newest articles balanced across sources.
    """
    pool = database.pool(config.max_article_age_days)
    scored = [article for article in pool if article.prefilter is not None]
    limit = config.max_candidates
    if not scored:
        return database.candidates(config.max_article_age_days, limit)
    random_count = max(1, limit // 10)
    cap = max(3, limit // 15)
    ranked = sorted(scored, key=lambda article: article.prefilter or 0.0, reverse=True)
    out: list[Article] = []
    per_source: dict[str, int] = {}
    for article in ranked:
        if len(out) >= limit - random_count:
            break
        if per_source.get(article.source, 0) < cap:
            per_source[article.source] = per_source.get(article.source, 0) + 1
            out.append(article)
    chosen = {article.id for article in out}
    # With only a few feeds the cap can't fill the list: top up ignoring it.
    for article in ranked:
        if len(out) >= limit - random_count:
            break
        if article.id not in chosen:
            out.append(article)
            chosen.add(article.id)
    # Random picks that stay the same all day, so they aren't re-chosen every hour.
    today = date.today().isoformat()
    rest = sorted((article for article in pool if article.id not in chosen),
                  key=lambda article: hashlib.sha1(f"{today}:{article.id}".encode()).digest())
    out += rest[:random_count]
    return out


def score_candidates(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> int:
    """Score shortlisted articles that are unscored or were scored against an older profile."""
    profile = ensure_profile(config, database)
    todo = [article for article in shortlist(config, database) if not article.scored_against(profile["id"])]
    if not todo:
        return 0
    # Unscored first, so a profile change never leaves fresh articles without any score.
    todo.sort(key=lambda article: article.score is not None)
    prefix = _score_prefix(profile["text"], _examples(database))
    # Room left for the article after the shared prefix.
    per_article = min(config.article_chars, max(1000, config.llm.max_prompt_chars - len(prefix) - 300))
    done = 0
    for article in todo:
        if stop is not None and stop.is_set():
            break
        try:
            result = _score_one(llm, prefix, article, per_article)
        except LLMError as error:
            log.warning("scoring article %d failed: %s", article.id, error)
            if isinstance(error, LLMUnavailable):
                break  # leave the rest for the next round
            continue
        database.set_score(article.id, result, profile["id"])
        database.connection.commit()
        done += 1
    log.info("scored %d/%d candidate articles", done, len(todo))
    return done


def _breakdown(article: Article, statistics: dict[str, SourceStatistics],
               prefilter_ranks: list[float]) -> ScoreBreakdown:
    bonus = statistics.get(article.source, SourceStatistics()).bonus
    # Where the embedding pre-filter ranks the article among the candidates, 0..1.
    rank = None
    if article.prefilter is not None and prefilter_ranks:
        rank = bisect.bisect_left(prefilter_ranks, article.prefilter) / len(prefilter_ranks)
    if article.score is None:
        # Not scored by the LLM (e.g. it was down): use the pre-filter rank, mapped to 2..6.
        base = 4.0 if rank is None else 2.0 + 4.0 * rank
        return ScoreBreakdown(None, base, bonus, base + bonus)
    # The LLM score, nudged (by up to ±2) by the embedding rank, so that a single
    # over-enthusiastic LLM score can't put an unrelated article at the top.
    agree = 0.0 if rank is None else 4.0 * rank - 2.0
    return ScoreBreakdown(article.score, agree, bonus, article.score + agree + bonus)


def _add_capped(
    chosen: list[DigestPick],
    articles: Iterable[Article],
    limit: int,
    exploratory: bool,
    max_per_source: int,
) -> None:
    """Append articles to `chosen` until it has `limit`, at most `max_per_source` per source."""
    per_source = Counter(pick.article.source for pick in chosen)
    for article in articles:
        if len(chosen) >= limit:
            break
        if per_source[article.source] >= max_per_source:
            continue
        per_source[article.source] += 1
        chosen.append(DigestPick(article, exploratory))


def select(config: Config, database: Database, randomizer: random.Random | None = None,
           debug: dict | None = None) -> list[DigestPick]:
    """Choose the digest: top-ranked articles plus a few exploratory picks.

    If `debug` is a dict it is
    filled with candidate counts and each candidate's score breakdown.
    """
    randomizer = randomizer or random.Random()
    candidates = shortlist(config, database)
    if not candidates:
        return []
    # Also anything the LLM already scored against the current profile.
    profile = ensure_profile(config, database)
    ids = {article.id for article in candidates}
    candidates += [article for article in database.pool(config.max_article_age_days)
                   if article.id not in ids and article.scored_against(profile["id"])]
    statistics = database.source_statistics()
    prefilter_ranks = sorted(article.prefilter for article in candidates if article.prefilter is not None)
    parts = {article.id: _breakdown(article, statistics, prefilter_ranks) for article in candidates}
    ranked = sorted(candidates, key=lambda article: parts[article.id].final, reverse=True)
    if debug is not None:
        debug["candidates"] = len(candidates)
        debug["llm_scored"] = sum(article.scored_against(profile["id"]) for article in candidates)
        debug["stale_scored"] = sum(article.has_stale_score(profile["id"]) for article in candidates)
        debug["scores"] = parts
        debug["rank"] = {article.id: rank for rank, article in enumerate(ranked, start=1)}
    limit = config.articles_per_digest
    explore_count = min(config.exploration, max(0, limit - 1))
    chosen: list[DigestPick] = []
    _add_capped(chosen, ranked, limit - explore_count, False, config.max_per_source)

    # Exploration: random picks from the rest, biased towards less-seen sources,
    # so the model keeps learning about things outside its current beliefs.
    picked = {pick.article.id for pick in chosen}
    rest = [article for article in ranked if article.id not in picked]
    randomizer.shuffle(rest)
    rest.sort(key=lambda article: statistics.get(article.source, SourceStatistics()).total)
    _add_capped(chosen, rest, limit, True, config.max_per_source)

    # Too few sources to honour the cap: fill remaining slots ignoring it.
    picked = {pick.article.id for pick in chosen}
    for article in ranked:
        if len(chosen) >= limit:
            break
        if article.id not in picked:
            chosen.append(DigestPick(article, False))
    return chosen


def _format_feedback(rows: Iterable[sqlite3.Row]) -> str:
    return "\n".join(
        f"- {row['title']} ({row['source']})" + (f' — "{row["comment"]}"' if row["comment"] else "")
        for row in rows
    ) or "(none)"


def update_profile(config: Config, database: Database, llm: LLM) -> bool:
    """Rewrite the reader profile from feedback received since the current profile was made.

    Returns True if the profile changed. Pending feedback stays pending if the LLM
    is unavailable, so this is safe to call repeatedly.
    """
    profile = ensure_profile(config, database)
    pending = database.feedback_since(profile["created_at"])
    feedback, comments = pending.feedback, pending.comments
    if not feedback and not comments:
        return False
    liked = [row for row in feedback if row["rating"] > 0]
    disliked = [row for row in feedback if row["rating"] < 0]
    neutral = [row for row in feedback if row["rating"] == 0]
    user = (
        f"Current reader profile:\n{profile['text']}\n\n"
        f"New feedback from the reader.\nLiked:\n{_format_feedback(liked)}\n\n"
        f"Disliked:\n{_format_feedback(disliked)}\n\n"
        f"Comments on specific articles (no explicit like/dislike):\n{_format_feedback(neutral)}\n\n"
        "General comments from the reader:\n"
        + ("\n".join(f'- "{comment["comment"]}"' for comment in comments) or "(none)")
        + "\n\nRewrite the profile to incorporate this feedback. Keep established preferences "
        "unless the new feedback contradicts them; treat explicit comments as strong signals "
        "and single likes/dislikes as weak signals. Describe topics, angles, depth and styles "
        "the reader likes and dislikes, and any sources they favour or avoid. "
        "Write at most 250 words as a bullet list. Output only the profile."
    )
    try:
        text = llm.chat(PROFILE_SYSTEM, user).strip()
    except LLMError as error:
        log.warning("profile update failed (will retry later): %s", error)
        return False
    if len(text) < 20:
        log.warning("profile update returned implausibly short text; keeping old profile")
        return False
    database.add_profile(
        text[:4000],
        note=f"{len(liked)} likes, {len(disliked)} dislikes, {len(neutral) + len(comments)} comments",
    )
    log.info("reader profile updated")
    return True
