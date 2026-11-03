"""Scoring articles against the learned reader profile, and learning from feedback."""

from __future__ import annotations

import bisect
import hashlib
import logging
import random
from datetime import date

from .llm import LLM, LLMError

log = logging.getLogger(__name__)

DEFAULT_PROFILE = (
    "No preferences learned yet. Prefer substantive, well-written, informative articles "
    "over clickbait, press releases, and thin news items."
)

SCORE_SYSTEM = (
    "You are a personal reading assistant. You predict how much one specific reader will "
    "want to read each article, based on what you know about their tastes. Be discerning: "
    "use the full 0-10 range, and reserve 8+ for articles that clearly match their interests."
)

PROFILE_SYSTEM = (
    "You maintain a concise profile of one reader's reading preferences, used to choose "
    "articles for them. You update it from their feedback."
)


def ensure_profile(cfg, db):
    row = db.current_profile()
    if row is None:
        text = cfg.interests or DEFAULT_PROFILE
        db.add_profile(text, note="initial (from config)")
        row = db.current_profile()
    return row


def _examples(db) -> str:
    def fmt(rows):
        out = []
        for r in rows:
            s = f"- {r['title']} ({r['source']})"
            if r["comment"]:
                s += f' — reader said: "{r["comment"][:150]}"'
            out.append(s)
        return "\n".join(out) or "(none yet)"

    return (
        "Articles the reader recently LIKED:\n" + fmt(db.rated_examples(1, 15)) +
        "\n\nArticles the reader recently DISLIKED:\n" + fmt(db.rated_examples(-1, 15))
    )


def article_text(a, limit: int) -> str:
    """Best available description of an article: page text if fetched, else the feed summary."""
    text = a["content"] or a["summary"] or ""
    if a["content"] and a["summary"] and len(a["content"]) < len(a["summary"]):
        text = a["summary"]
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _batches(todo, max_items: int, budget: int, per_article: int):
    """Group articles so each prompt stays within the model's context window."""
    batch, used = [], 0
    for a in todo:
        cost = len(a["title"]) + len(article_text(a, per_article)) + 40
        if batch and (len(batch) >= max_items or used + cost > budget):
            yield batch
            batch, used = [], 0
        batch.append(a)
        used += cost
    if batch:
        yield batch


def score_batch(llm: LLM, profile_text: str, examples: str, articles,
                per_article: int = 1500) -> dict[int, tuple[float, str]]:
    lines = []
    for i, a in enumerate(articles, start=1):
        text = article_text(a, per_article)
        src = f"{a['source']} ({a['category']})" if a["category"] else a["source"]
        lines.append(f"[{i}] {a['title']}\n    Source: {src}\n    {text}")
    user = (
        f"Reader profile:\n{profile_text}\n\n{examples}\n\n"
        "Rate each candidate article below from 0 (reader would not want it) to 10 "
        "(reader would definitely want to read it).\n"
        'Respond with JSON only, in the form {"ratings": [{"id": 1, "score": 7, '
        '"reason": "why, in at most 15 words"}, ...]} with one entry per candidate.\n\n'
        "Candidates:\n" + "\n\n".join(lines)
    )
    data = llm.chat_json(SCORE_SYSTEM, user)
    ratings = data.get("ratings", data) if isinstance(data, dict) else data
    out: dict[int, tuple[float, str]] = {}
    if not isinstance(ratings, list):
        raise LLMError("ratings missing from LLM output")
    for r in ratings:
        try:
            idx = int(r["id"])
            score = max(0.0, min(10.0, float(r["score"])))
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= idx <= len(articles):
            out[articles[idx - 1]["id"]] = (score, str(r.get("reason", ""))[:200])
    return out


def shortlist(cfg, db) -> list:
    """The articles worth spending LLM time on today.

    Uses the embedding pre-filter when available: the best matches (with a per-source
    cap so no single feed dominates) plus ~10% random picks for exploration. Without
    embeddings, falls back to the newest articles balanced across sources.
    """
    pool = db.pool(cfg.max_article_age_days)
    scored = [a for a in pool if a["prefilter"] is not None]
    n = cfg.max_candidates
    if not scored:
        return db.candidates(cfg.max_article_age_days, n)
    n_random = max(1, n // 10)
    cap = max(3, n // 15)
    ranked = sorted(scored, key=lambda a: a["prefilter"], reverse=True)
    out, per_source = [], {}
    for a in ranked:
        if len(out) >= n - n_random:
            break
        if per_source.get(a["source"], 0) < cap:
            per_source[a["source"]] = per_source.get(a["source"], 0) + 1
            out.append(a)
    chosen = {a["id"] for a in out}
    # With only a few feeds the cap can't fill the list: top up ignoring it.
    for a in ranked:
        if len(out) >= n - n_random:
            break
        if a["id"] not in chosen:
            out.append(a)
            chosen.add(a["id"])
    # Random picks that stay the same all day, so they aren't re-chosen every hour.
    today = date.today().isoformat()
    rest = sorted((a for a in pool if a["id"] not in chosen),
                  key=lambda a: hashlib.sha1(f"{today}:{a['id']}".encode()).digest())
    out += rest[:n_random]
    return out


def score_candidates(cfg, db, llm: LLM, stop=None) -> int:
    """Score shortlisted articles that are unscored or were scored against an older profile."""
    profile = ensure_profile(cfg, db)
    todo = [a for a in shortlist(cfg, db) if a["score_profile_id"] != profile["id"]]
    if not todo:
        return 0
    # Unscored first, so a profile change never leaves fresh articles without any score.
    todo.sort(key=lambda a: a["score"] is not None)
    examples = _examples(db)
    done = 0
    # Room left for article text after the instructions, profile and examples.
    budget = max(2000, cfg.llm.max_prompt_chars - len(profile["text"]) - len(examples) - 800)
    per_article = min(cfg.article_chars, budget)
    for batch in _batches(todo, max(1, cfg.llm.batch_size), budget, per_article):
        if stop is not None and stop.is_set():
            break
        try:
            results = score_batch(llm, profile["text"], examples, batch, per_article)
        except LLMError as e:
            log.warning("scoring batch failed: %s", e)
            if "cannot reach" in str(e):
                break
            continue
        for aid, (score, reason) in results.items():
            db.set_score(aid, score, reason, profile["id"])
        db.conn.commit()
        done += len(results)
    log.info("scored %d/%d candidate articles", done, len(todo))
    return done


def _source_bonus(stats: dict, source: str) -> float:
    likes, dislikes = stats.get(source, (0, 0))
    # Beta(2,2) prior -> smoothed like rate in (0,1); bonus in [-1.5, 1.5]
    rate = (likes + 2) / (likes + dislikes + 4)
    return 3.0 * (rate - 0.5)


def select(cfg, db, rng: random.Random | None = None):
    """Choose the digest: top-ranked articles plus a few exploratory picks.

    Returns a list of (article_row, exploratory: bool).
    """
    rng = rng or random.Random()
    cands = shortlist(cfg, db)
    if not cands:
        return []
    # Also anything the LLM already scored against the current profile.
    profile = ensure_profile(cfg, db)
    ids = {a["id"] for a in cands}
    cands += [a for a in db.pool(cfg.max_article_age_days)
              if a["id"] not in ids and a["score_profile_id"] == profile["id"]]
    stats = db.source_stats()
    pre = sorted(a["prefilter"] for a in cands if a["prefilter"] is not None)

    def fallback(a):
        # Not scored by the LLM (e.g. it was down): use the pre-filter rank, mapped to 2..6.
        if a["prefilter"] is None or not pre:
            return 4.0
        return 2.0 + 4.0 * bisect.bisect_left(pre, a["prefilter"]) / len(pre)

    def final(a):
        if a["score"] is None:
            return fallback(a) + _source_bonus(stats, a["source"])
        # The LLM score, nudged (by up to ±2) by the embedding rank, so that a single
        # over-enthusiastic LLM score can't put an unrelated article at the top.
        agree = 0.0
        if a["prefilter"] is not None and pre:
            agree = 4.0 * bisect.bisect_left(pre, a["prefilter"]) / len(pre) - 2.0
        return a["score"] + agree + _source_bonus(stats, a["source"])

    ranked = sorted(cands, key=final, reverse=True)
    n = cfg.articles_per_digest
    n_explore = min(cfg.exploration, max(0, n - 1))
    per_source: dict[str, int] = {}
    chosen = []

    def take(a, exploratory):
        if per_source.get(a["source"], 0) >= cfg.max_per_source:
            return False
        per_source[a["source"]] = per_source.get(a["source"], 0) + 1
        chosen.append((a, exploratory))
        return True

    for a in ranked:
        if len(chosen) >= n - n_explore:
            break
        take(a, False)

    # Exploration: random picks from the rest, biased towards less-seen sources,
    # so the model keeps learning about things outside its current beliefs.
    picked = {a["id"] for a, _ in chosen}
    rest = [a for a in ranked if a["id"] not in picked]
    rng.shuffle(rest)
    rest.sort(key=lambda a: sum(stats.get(a["source"], (0, 0))))
    for a in rest:
        if len(chosen) >= n:
            break
        take(a, True)

    # Too few sources to honour the cap: fill remaining slots ignoring it.
    picked = {a["id"] for a, _ in chosen}
    for a in ranked:
        if len(chosen) >= n:
            break
        if a["id"] not in picked:
            chosen.append((a, False))
    return chosen


def update_profile(cfg, db, llm: LLM) -> bool:
    """Rewrite the reader profile from feedback received since the current profile was made.

    Returns True if the profile changed. Pending feedback stays pending if the LLM
    is unavailable, so this is safe to call repeatedly.
    """
    profile = ensure_profile(cfg, db)
    fb, comments = db.feedback_since(profile["created_at"])
    if not fb and not comments:
        return False
    liked = [r for r in fb if r["rating"] > 0]
    disliked = [r for r in fb if r["rating"] < 0]
    neutral = [r for r in fb if r["rating"] == 0]

    def fmt(items):
        return "\n".join(
            f"- {r['title']} ({r['source']})" + (f' — "{r["comment"]}"' if r["comment"] else "")
            for r in items
        ) or "(none)"

    user = (
        f"Current reader profile:\n{profile['text']}\n\n"
        f"New feedback from the reader.\nLiked:\n{fmt(liked)}\n\nDisliked:\n{fmt(disliked)}\n\n"
        f"Comments on specific articles (no explicit like/dislike):\n{fmt(neutral)}\n\n"
        "General comments from the reader:\n"
        + ("\n".join(f'- "{c["comment"]}"' for c in comments) or "(none)")
        + "\n\nRewrite the profile to incorporate this feedback. Keep established preferences "
        "unless the new feedback contradicts them; treat explicit comments as strong signals "
        "and single likes/dislikes as weak signals. Describe topics, angles, depth and styles "
        "the reader likes and dislikes, and any sources they favour or avoid. "
        "Write at most 250 words as a bullet list. Output only the profile."
    )
    try:
        text = llm.chat(PROFILE_SYSTEM, user).strip()
    except LLMError as e:
        log.warning("profile update failed (will retry later): %s", e)
        return False
    if len(text) < 20:
        log.warning("profile update returned implausibly short text; keeping old profile")
        return False
    db.add_profile(
        text[:4000],
        note=f"{len(liked)} likes, {len(disliked)} dislikes, {len(neutral) + len(comments)} comments",
    )
    log.info("reader profile updated")
    return True
