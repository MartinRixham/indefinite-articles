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

import numpy as np

from ..config import Config
from ..language_model.assistant import MAX_PROFILE_CHARS, ArticleScore, Assistant
from ..language_model.llm import LLM, LLMError, LLMUnavailable
from ..storage.article import Article
from ..storage.database import Database, SourceStatistics
from .vectors import unpack

log = logging.getLogger(__name__)

DEFAULT_PROFILE = (
    "No preferences learned yet. Prefer substantive, well-written, informative articles "
    "over clickbait, press releases, and thin news items."
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

    def debug_line(self, position: int, score: "ScoreBreakdown", rank: int, overlap: float = 0.0) -> str:
        """This pick's row in the digest's debug table; `overlap` is the points it lost for
        resembling an article picked before it."""
        article = self.article
        llm_score = f"{score.llm:5.1f}" if score.llm is not None else "    -"
        return (
            f"{position:<3} {rank:>4}  {score.final:5.2f} = {llm_score} {score.base:+5.2f}"
            f" {score.bonus:+5.2f}  {-overlap:+5.2f}   {(article.content_status or '-')[:8]:8}  {article.source[:30]}"
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


def _split(articles: Iterable[Article]) -> tuple[list[Article], list[Article]]:
    """The articles that aren't podcast episodes, and those that are, each in their original order."""
    articles = list(articles)
    return ([article for article in articles if not article.is_podcast],
            [article for article in articles if article.is_podcast])


def shortlist(config: Config, database: Database) -> list[Article]:
    """The articles worth spending LLM time on today: up to `max_candidates` articles,
    then (if the digest has room for any) up to `max_podcast_candidates` podcast episodes.

    Podcast episodes get their own shortlist so that the far more numerous articles
    can't crowd them out of LLM scoring.
    """
    articles, episodes = _split(database.pool(config.max_article_age_days))
    out = _shortlist_of(config, database, articles, config.max_candidates, podcasts=False)
    if config.podcasts_per_digest > 0:
        out += _shortlist_of(config, database, episodes, config.max_podcast_candidates, podcasts=True)
    return out


def _shortlist_of(config: Config, database: Database, pool: list[Article], limit: int,
                  podcasts: bool) -> list[Article]:
    """At most `limit` of `pool` (which holds only articles, or only podcast episodes).

    Uses the embedding pre-filter when available: the best matches (with a per-source
    cap so no single feed dominates) plus ~10% random picks for exploration. Without
    embeddings, falls back to the newest articles balanced across sources.
    """
    if limit <= 0:
        return []
    scored = [article for article in pool if article.prefilter is not None]
    if not scored:
        return database.candidates(config.max_article_age_days, limit, podcasts)
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
    """LLM-score the shortlisted articles that have no score yet, then refresh stale scores.

    A score made against an older profile is still a useful estimate, so after a profile
    change only the stale articles among the top `rescore_top` of the digest ranking (and
    likewise among the top podcast episodes) are rescored, best first: they are the ones
    that decide what goes into the digest.
    Rescoring everything would cost one LLM request per shortlisted article per reply.
    """
    profile = ensure_profile(config, database)
    candidates = _digest_candidates(config, database, profile["id"])
    unscored = [article for article in candidates if article.score is None]
    ranked, _ = _rank(candidates, database.source_statistics())
    stale = [article for group in _split(ranked) for article in group[:config.rescore_top]
             if article.has_stale_score(profile["id"])]
    todo = unscored + stale
    if not todo:
        return 0
    assistant = Assistant(llm)
    prefix = assistant.score_prefix(profile["text"], _examples(database))
    # Room left for the article after the shared prefix.
    per_article = min(config.article_chars, max(1000, config.llm.max_prompt_chars - len(prefix) - 300))
    done = rescored = 0
    for article in todo:
        if stop is not None and stop.is_set():
            break
        try:
            result = assistant.score_article(prefix, article, per_article)
        except LLMError as error:
            log.warning("scoring article %d failed: %s", article.id, error)
            if isinstance(error, LLMUnavailable):
                break  # leave the rest for the next round
            continue
        database.set_score(article.id, result, profile["id"])
        database.connection.commit()
        done += 1
        rescored += article.score is not None
    log.info("scored %d/%d new candidate articles, rescored %d/%d stale ones in the top %d",
             done - rescored, len(unscored), rescored, len(stale), config.rescore_top)
    return done


def _digest_candidates(config: Config, database: Database, profile_id: int) -> list[Article]:
    """The shortlist, plus anything the LLM already scored against the current profile."""
    candidates = shortlist(config, database)
    if not candidates:
        return []
    ids = {article.id for article in candidates}
    return candidates + [article for article in database.pool(config.max_article_age_days)
                         if article.id not in ids and article.scored_against(profile_id)]


def _rank(candidates: list[Article], statistics: dict[str, SourceStatistics],
          ) -> tuple[list[Article], dict[int, "ScoreBreakdown"]]:
    """The candidates best first, and each one's score breakdown by article id."""
    prefilter_ranks = sorted(article.prefilter for article in candidates if article.prefilter is not None)
    parts = {article.id: _breakdown(article, statistics, prefilter_ranks) for article in candidates}
    return sorted(candidates, key=lambda article: parts[article.id].final, reverse=True), parts


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


def _add_diverse(chosen: list[DigestPick], ranked: list[Article], limit: int, max_per_source: int,
                 diversity: _Diversity) -> None:
    """Append top picks to `chosen` until it has `limit`, at most `max_per_source` per source:
    each time the best of `ranked` by its score less its overlap penalty (ties go to the
    one ranked higher, so with no penalties this is simply the top of `ranked`)."""
    per_source = Counter(pick.article.source for pick in chosen)
    remaining = list(ranked)
    while len(chosen) < limit:
        best: Article | None = None
        best_value = best_penalty = 0.0
        for article in remaining:
            if per_source[article.source] >= max_per_source:
                continue
            penalty = diversity.penalty(article, chosen)
            value = diversity.scores[article.id] - penalty
            if best is None or value > best_value:
                best, best_value, best_penalty = article, value, penalty
        if best is None:
            return
        remaining.remove(best)
        per_source[best.source] += 1
        diversity.penalties[best.id] = best_penalty
        chosen.append(DigestPick(best, False))


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
    """Choose the digest: top-ranked articles plus a few exploratory picks, then the
    podcast episodes, chosen the same way.

    If `debug` is a dict it is
    filled with candidate counts and each candidate's score breakdown.
    """
    randomizer = randomizer or random.Random()
    profile = ensure_profile(config, database)
    candidates = _digest_candidates(config, database, profile["id"])
    if not candidates:
        return []
    statistics = database.source_statistics()
    ranked, parts = _rank(candidates, statistics)
    overlap = _Overlap(database.vectors_of(article.id for article in candidates))
    penalties: dict[int, float] = {}
    if debug is not None:
        debug["candidates"] = len(candidates)
        debug["llm_scored"] = sum(article.scored_against(profile["id"]) for article in candidates)
        debug["stale_scored"] = sum(article.has_stale_score(profile["id"]) for article in candidates)
        debug["scores"] = parts
        debug["rank"] = {article.id: rank for rank, article in enumerate(ranked, start=1)}
        debug["overlap"] = penalties
    articles, episodes = _split(ranked)
    diversity = _Diversity({article_id: part.final for article_id, part in parts.items()},
                           overlap, config.diversity, penalties)
    chosen = _choose(articles, config.articles_per_digest, config.exploration, config.max_per_source,
                     statistics, randomizer, diversity)
    if config.podcasts_per_digest > 0:
        chosen += _choose(episodes, config.podcasts_per_digest, config.podcast_exploration,
                          config.max_per_source, statistics, randomizer, diversity)
    return chosen


class _Overlap:
    """How much candidates overlap in subject, from their embeddings.

    Embeddings of news articles all point in roughly the same direction, so the cosine
    similarity is taken after subtracting the candidates' mean vector: unrelated
    articles then come out near 0, and two reports of the same story near 1.
    """

    def __init__(self, vectors: dict[int, bytes]):
        # Ignore vectors from a different model (e.g. after changing embedding_model).
        dimension = Counter(len(vector) for vector in vectors.values()).most_common(1)
        ids = [article_id for article_id, vector in vectors.items() if dimension and len(vector) == dimension[0][0]]
        self._rows = {article_id: row for row, article_id in enumerate(ids)}
        if len(ids) < 2:
            self._matrix = np.zeros((len(ids), 1), dtype=np.float32)
            return
        matrix = unpack([vectors[article_id] for article_id in ids])
        matrix = matrix - matrix.mean(axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.where(norms > 0, norms, 1.0)

    def closest(self, article: Article, others: Iterable[Article]) -> float:
        """The similarity of `article` to the most similar of `others`, 0..1 (0 if there are
        none, or no embedding to compare)."""
        row = self._rows.get(article.id)
        other_rows = [self._rows[other.id] for other in others if other.id in self._rows]
        if row is None or not other_rows:
            return 0.0
        return max(0.0, float((self._matrix[other_rows] @ self._matrix[row]).max()))


@dataclass
class _Diversity:
    """What the top picks of a digest are chosen by: each article's final ranking score,
    less `weight` points times its overlap with the closest article already picked.
    The points taken off each pick are recorded in `penalties`."""
    scores: dict[int, float]
    overlap: _Overlap
    weight: float
    penalties: dict[int, float]

    def penalty(self, article: Article, chosen: list[DigestPick]) -> float:
        if self.weight <= 0:
            return 0.0
        return self.weight * self.overlap.closest(article, (pick.article for pick in chosen))


def _choose(ranked: list[Article], limit: int, exploration: int, max_per_source: int,
            statistics: dict[str, SourceStatistics], randomizer: random.Random,
            diversity: _Diversity) -> list[DigestPick]:
    """`limit` picks from `ranked` (best first): the top ones, at most `max_per_source` from
    each source and favouring ones that don't overlap those already picked, then
    `exploration` random ones from the rest."""
    explore_count = min(exploration, max(0, limit - 1))
    chosen: list[DigestPick] = []
    _add_diverse(chosen, ranked, limit - explore_count, max_per_source, diversity)

    # Exploration: random picks from the rest, biased towards less-seen sources,
    # so the model keeps learning about things outside its current beliefs.
    picked = {pick.article.id for pick in chosen}
    rest = [article for article in ranked if article.id not in picked]
    randomizer.shuffle(rest)
    rest.sort(key=lambda article: statistics.get(article.source, SourceStatistics()).total)
    _add_capped(chosen, rest, limit, True, max_per_source)

    # Too few sources to honour the cap: fill remaining slots ignoring it.
    picked = {pick.article.id for pick in chosen}
    for article in ranked:
        if len(chosen) >= limit:
            break
        if article.id not in picked:
            chosen.append(DigestPick(article, False))
    return chosen


def _fit_profile(text: str) -> str:
    """`text` cut to `MAX_PROFILE_CHARS` by dropping whole lines from the top, since
    the profile writer appends what it has just learned at the bottom."""
    if len(text) <= MAX_PROFILE_CHARS:
        return text
    lines = text.splitlines()
    while len(lines) > 1 and len("\n".join(lines)) > MAX_PROFILE_CHARS:
        lines.pop(0)
    fitted = "\n".join(lines).strip()[-MAX_PROFILE_CHARS:]
    log.warning("reader profile too long (%d characters); dropped %d from the top",
                len(text), len(text) - len(fitted))
    return fitted


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
    try:
        text = Assistant(llm).rewrite_profile(profile["text"], pending)
    except LLMError as error:
        log.warning("profile update failed (will retry later): %s", error)
        return False
    like_count = sum(1 for row in feedback if row["rating"] > 0)
    dislike_count = sum(1 for row in feedback if row["rating"] < 0)
    comment_count = len(feedback) - like_count - dislike_count + len(comments)
    database.add_profile(
        _fit_profile(text),
        note=f"{like_count} likes, {dislike_count} dislikes, {comment_count} comments",
    )
    log.info("reader profile updated")
    return True
