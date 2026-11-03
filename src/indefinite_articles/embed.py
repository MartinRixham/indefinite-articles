"""Cheap first-pass ranking with embeddings.

With hundreds of feeds there are thousands of new articles a day, far more than a
local LLM can read. Every article is embedded (a few milliseconds each with a small
embedding model) and compared with the reader profile and with articles the reader
liked and disliked. Only the best matches, plus a few random ones, go on to full-text
fetching and LLM scoring.
"""

from __future__ import annotations

import base64
import logging
import math
import re
from array import array

from .llm import LLM, LLMError

try:  # optional: ~50x faster scoring for large pools
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

log = logging.getLogger(__name__)

N_LIKES = 20      # most recent liked/disliked articles compared against
N_DISLIKES = 20
MAX_PHRASES = 12  # per side, from the reader profile
DEFAULT_WANTED = "substantive, well-written, informative articles"

PREFS_SYSTEM = "You turn a reader's profile into short phrases for a search engine."
PREFS_PROMPT = (
    "Reader profile:\n{profile}\n\n"
    "List the kinds of article this reader wants, and separately the kinds they want to "
    "avoid, as short positive descriptions of article topics, angles or styles, e.g. "
    '"in-depth articles on Rust compiler internals" or "celebrity gossip". Never use '
    'negations such as "no", "not" or "avoid" inside a phrase: put unwanted things in '
    "the avoid list instead. At most {n} phrases per list.\n"
    'Respond with JSON only: {{"likes": ["..."], "avoid": ["..."]}}'
)

# Words that mark a profile sentence as describing what the reader does NOT want.
_NEGATIVE = re.compile(
    r"\b(dislik\w*|avoid\w*|hates?|not|no|never|less|fewer|don't|doesn't|isn't|skip\w*|"
    r"boring|uninterest\w*|tired of|without|irrelevant|rather than|instead of)\b", re.I)
_CLAUSES = re.compile(r"\n+|(?<=[.!?;])\s+|,?\s+\b(?:but|except|however|whereas|although)\b\s*",
                      re.I)


def _normalise(v) -> array:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return array("f", (x / norm for x in v))


def pack(v) -> bytes:
    return _normalise(v).tobytes()


def unpack(b: bytes) -> array:
    a = array("f")
    a.frombytes(b)
    return a


def article_text(a) -> str:
    return f"{a['title']}. {(a['summary'] or '')[:600]}"


def split_profile(text: str) -> tuple[list[str], list[str]]:
    """Split a free-text profile into (wanted, unwanted) phrases, without an LLM.

    Embeddings ignore negation, so "dislikes product announcements" sits close to
    product announcements. Each clause is classified by its wording instead, and the
    two sides are embedded separately.
    """
    likes, avoid = [], []
    for clause in _CLAUSES.split(text):
        clause = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", clause or "").strip(" .;:")
        if len(clause) < 3:
            continue
        (avoid if _NEGATIVE.search(clause) else likes).append(clause[:200])
    return likes[:MAX_PHRASES], avoid[:MAX_PHRASES]


def profile_phrases(llm: LLM, text: str) -> tuple[list[str], list[str]]:
    """(wanted, unwanted) phrases from the profile: the LLM's split, else `split_profile`."""
    try:
        data = llm.chat_json(PREFS_SYSTEM, PREFS_PROMPT.format(profile=text[:4000], n=MAX_PHRASES))
    except LLMError as e:
        log.info("profile split by LLM failed, using keyword split: %s", e)
        data = None
    if isinstance(data, dict):
        def clean(key):
            items = data.get(key)
            if not isinstance(items, list):
                return []
            return [str(x).strip()[:200] for x in items if str(x).strip()][:MAX_PHRASES]
        likes, avoid = clean("likes"), clean("avoid")
        if likes:
            return likes, avoid
    return split_profile(text)


def embed_new(cfg, db, llm: LLM, stop=None, limit: int = 20000) -> int:
    if db.get_state("embedding_model") != cfg.llm.embedding_model:
        # Vectors from different models aren't comparable: start again.
        db.conn.execute("DELETE FROM embeddings")
        db.conn.execute("UPDATE articles SET prefilter=NULL, prefilter_version=NULL")
        db.set_state("embedding_model", cfg.llm.embedding_model)
    rows = db.unembedded(cfg.max_article_age_days, limit)
    done = 0
    bs = max(1, cfg.llm.embedding_batch)
    for i in range(0, len(rows), bs):
        if stop is not None and stop.is_set():
            break
        batch = rows[i : i + bs]
        vecs = llm.embed([article_text(a) for a in batch])
        for a, v in zip(batch, vecs):
            db.set_embedding(a["id"], pack(v))
        db.conn.commit()
        done += len(batch)
    if done:
        log.info("embedded %d articles", done)
    return done


def _max_sims_numpy(m, refs):
    if not refs:
        return np.zeros(len(m), dtype=np.float32)
    return (m @ np.frombuffer(b"".join(refs), dtype=np.float32).reshape(len(refs), -1).T).max(axis=1)


def _scores_numpy(vecs, wanted, unwanted, likes, dislikes):
    m = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1)
    p = _max_sims_numpy(m, wanted) - 0.5 * _max_sims_numpy(m, unwanted)
    if likes:
        sims = m @ np.frombuffer(b"".join(likes), dtype=np.float32).reshape(len(likes), -1).T
        k = min(2, sims.shape[1])
        liked = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    else:
        liked = p
    disliked = _max_sims_numpy(m, dislikes)
    return (0.4 * p + 0.6 * liked - 0.4 * disliked).tolist()


def _scores_python(vecs, wanted, unwanted, likes, dislikes):
    wv = [unpack(b) for b in wanted]
    uv = [unpack(b) for b in unwanted]
    lv = [unpack(b) for b in likes]
    dv = [unpack(b) for b in dislikes]
    dot = lambda a, b: math.fsum(map(float.__mul__, a, b))
    out = []
    for b in vecs:
        v = unpack(b)
        p = (max((dot(v, x) for x in wv), default=0.0)
             - 0.5 * max((dot(v, x) for x in uv), default=0.0))
        if lv:
            sims = sorted((dot(v, x) for x in lv), reverse=True)[:2]
            liked = sum(sims) / len(sims)
        else:
            liked = p
        disliked = max((dot(v, x) for x in dv), default=0.0)
        out.append(0.4 * p + 0.6 * liked - 0.4 * disliked)
    return out


def score(vecs, wanted, unwanted, likes, dislikes):
    """Pre-filter score of each article vector.

    `wanted`/`unwanted` are the profile's phrases (an article is judged by its closest
    phrase on each side); `likes`/`dislikes` are articles the reader rated.
    """
    if not vecs:
        return []
    if np is not None:
        return _scores_numpy(vecs, wanted, unwanted, likes, dislikes)
    return _scores_python(vecs, wanted, unwanted, likes, dislikes)


def _join(vs: list[bytes]) -> str:
    return base64.b64encode(b"".join(vs)).decode()


def _split(s: str, dim: int) -> list[bytes]:
    b = base64.b64decode(s)
    return [b[i : i + dim] for i in range(0, len(b), dim)]


def update_prefilter(cfg, db, llm: LLM, profile) -> int:
    """(Re)compute the pre-filter score of every embedded candidate article.

    Everything is rescored when the profile or the feedback changes; otherwise only
    newly embedded articles are scored.
    """
    sig = f"{profile['id']}:{db.feedback_signature()}:{cfg.llm.embedding_model}"
    version = int(db.get_state("prefilter_version", "0"))
    dim = int(db.get_state("profile_dim", "0"))
    if db.get_state("prefilter_sig") != sig or not dim:
        phrases_for = f"{profile['id']}:{cfg.llm.embedding_model}"
        if db.get_state("profile_phrases_for") != phrases_for:
            # Only when the profile itself changes: this may cost an LLM call.
            wanted, unwanted = profile_phrases(llm, profile["text"])
            if not wanted:  # e.g. a profile that only lists dislikes
                wanted = [DEFAULT_WANTED]
            log.info("profile phrases: wanted %s; unwanted %s", wanted, unwanted)
            vecs = [pack(v) for v in llm.embed(wanted + unwanted)]
            dim = len(vecs[0])
            db.set_state("profile_wanted", _join(vecs[: len(wanted)]))
            db.set_state("profile_unwanted", _join(vecs[len(wanted):]))
            db.set_state("profile_dim", str(dim))
            db.set_state("profile_phrases_for", phrases_for)
            db.conn.commit()
        version += 1
        db.set_state("prefilter_version", str(version))
        db.set_state("prefilter_sig", sig)
    todo = db.prefilter_todo(cfg.max_article_age_days, version)
    if not todo:
        return 0
    wanted = _split(db.get_state("profile_wanted", ""), dim)
    unwanted = _split(db.get_state("profile_unwanted", ""), dim)
    # Ignore vectors from a different model (e.g. after changing embedding_model).
    todo = [r for r in todo if len(r["vec"]) == dim]
    likes = [v for v in db.rated_vectors(1, N_LIKES) if len(v) == dim]
    dislikes = [v for v in db.rated_vectors(-1, N_DISLIKES) if len(v) == dim]
    scores = score([r["vec"] for r in todo], wanted, unwanted, likes, dislikes)
    db.set_prefilter([(s, version, r["id"]) for s, r in zip(scores, todo)])
    log.info("pre-filter scored %d articles (%d likes, %d dislikes as reference)",
             len(todo), len(likes), len(dislikes))
    return len(todo)
