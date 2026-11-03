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
from array import array

from .llm import LLM

try:  # optional: ~50x faster scoring for large pools
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

log = logging.getLogger(__name__)

N_LIKES = 20      # most recent liked/disliked articles compared against
N_DISLIKES = 20


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


def _scores_numpy(vecs, profile, likes, dislikes):
    m = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1)
    p = m @ np.frombuffer(profile, dtype=np.float32)
    if likes:
        sims = m @ np.frombuffer(b"".join(likes), dtype=np.float32).reshape(len(likes), -1).T
        k = min(2, sims.shape[1])
        liked = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    else:
        liked = p
    if dislikes:
        sims = m @ np.frombuffer(b"".join(dislikes), dtype=np.float32).reshape(len(dislikes), -1).T
        disliked = sims.max(axis=1)
    else:
        disliked = np.zeros(len(vecs), dtype=np.float32)
    return (0.4 * p + 0.6 * liked - 0.4 * disliked).tolist()


def _scores_python(vecs, profile, likes, dislikes):
    pv = unpack(profile)
    lv = [unpack(b) for b in likes]
    dv = [unpack(b) for b in dislikes]
    dot = lambda a, b: math.fsum(map(float.__mul__, a, b))
    out = []
    for b in vecs:
        v = unpack(b)
        p = dot(v, pv)
        if lv:
            sims = sorted((dot(v, x) for x in lv), reverse=True)[:2]
            liked = sum(sims) / len(sims)
        else:
            liked = p
        disliked = max((dot(v, x) for x in dv), default=0.0)
        out.append(0.4 * p + 0.6 * liked - 0.4 * disliked)
    return out


def score(vecs, profile, likes, dislikes):
    if not vecs:
        return []
    if np is not None:
        return _scores_numpy(vecs, profile, likes, dislikes)
    return _scores_python(vecs, profile, likes, dislikes)


def update_prefilter(cfg, db, llm: LLM, profile) -> int:
    """(Re)compute the pre-filter score of every embedded candidate article.

    Everything is rescored when the profile or the feedback changes; otherwise only
    newly embedded articles are scored.
    """
    sig = f"{profile['id']}:{db.feedback_signature()}:{cfg.llm.embedding_model}"
    version = int(db.get_state("prefilter_version", "0"))
    profile_vec = db.get_state("profile_vec")
    if db.get_state("prefilter_sig") != sig or not profile_vec:
        profile_vec = base64.b64encode(pack(llm.embed([profile["text"][:3000]])[0])).decode()
        version += 1
        db.set_state("profile_vec", profile_vec)
        db.set_state("prefilter_version", str(version))
        db.set_state("prefilter_sig", sig)
    todo = db.prefilter_todo(cfg.max_article_age_days, version)
    if not todo:
        return 0
    profile_b = base64.b64decode(profile_vec)
    dim = len(profile_b)
    # Ignore vectors from a different model (e.g. after changing embedding_model).
    todo = [r for r in todo if len(r["vec"]) == dim]
    likes = [v for v in db.rated_vectors(1, N_LIKES) if len(v) == dim]
    dislikes = [v for v in db.rated_vectors(-1, N_DISLIKES) if len(v) == dim]
    scores = score([r["vec"] for r in todo], profile_b, likes, dislikes)
    db.set_prefilter([(s, version, r["id"]) for s, r in zip(scores, todo)])
    log.info("pre-filter scored %d articles (%d likes, %d dislikes as reference)",
             len(todo), len(likes), len(dislikes))
    return len(todo)
