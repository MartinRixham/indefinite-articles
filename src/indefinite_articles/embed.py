"""Cheap first-pass ranking with embeddings.

With hundreds of feeds there are thousands of new articles a day, far more than a
local LLM can read. Every article is embedded (a few milliseconds each with a small
embedding model) and compared with the reader profile. Once the reader has rated
enough articles, a small logistic-regression classifier trained on the embeddings of
every rated article adds what it has learned; before that, articles are compared with
the most recent likes and dislikes. Only the best matches, plus a few random ones, go
on to full-text fetching and LLM scoring.
"""

import base64
import logging
import re
import time

import numpy as np

from .llm import LLM, LLMError

log = logging.getLogger(__name__)

N_LIKES = 20      # most recent liked/disliked articles compared against, until the
N_DISLIKES = 20   #   classifier has enough ratings to take over
MIN_PER_CLASS = 5           # likes and dislikes each needed to train the classifier
MAX_TRAIN = 5000            # most recent ratings used for training
EPOCHS = 100                # full-batch gradient descent steps
L2 = 1e-3                   # weight decay; also keeps probabilities modest with little data
HALF_LIFE_DAYS = 365        # older ratings count less, as tastes drift
CLASSIFIER_WEIGHT = 0.5     # weight of (probability - 0.5) next to the profile match
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


def pack(v) -> bytes:
    """Unit-length float32 bytes, as stored in the embeddings table."""
    a = np.asarray(v, dtype=np.float32)
    return (a / (np.linalg.norm(a) or 1.0)).astype(np.float32).tobytes()


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


def _max_sims(m, refs):
    if not refs:
        return np.zeros(len(m), dtype=np.float32)
    return (m @ np.frombuffer(b"".join(refs), dtype=np.float32).reshape(len(refs), -1).T).max(axis=1)


def _scores(vecs, wanted, unwanted, likes, dislikes):
    m = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1)
    p = _max_sims(m, wanted) - 0.5 * _max_sims(m, unwanted)
    if likes:
        sims = m @ np.frombuffer(b"".join(likes), dtype=np.float32).reshape(len(likes), -1).T
        k = min(2, sims.shape[1])
        liked = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    else:
        liked = p
    disliked = _max_sims(m, dislikes)
    return (0.4 * p + 0.6 * liked - 0.4 * disliked).tolist()


def score(vecs, wanted, unwanted, likes, dislikes):
    """Pre-filter score of each article vector, used before the classifier is trained.

    `wanted`/`unwanted` are the profile's phrases (an article is judged by its closest
    phrase on each side); `likes`/`dislikes` are articles the reader rated.
    """
    if not vecs:
        return []
    return _scores(vecs, wanted, unwanted, likes, dislikes)


# -- classifier ----------------------------------------------------------------

def _sample_weights(labels, ages_days) -> list[float]:
    """Each class gets half the total weight (likes usually outnumber dislikes), and
    within a class older ratings count less."""
    decay = [0.5 ** (max(0.0, a) / HALF_LIFE_DAYS) for a in ages_days]
    tot = {c: sum(d for d, y in zip(decay, labels) if y == c) for c in (0, 1)}
    return [0.5 * d / tot[y] for d, y in zip(decay, labels)]


def _train(vecs, labels, weights):
    x = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1).astype(np.float64)
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(weights, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    lr = 1.0 / (0.25 * (float((xc * xc).sum(axis=1).max()) + 1.0) + L2)
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(EPOCHS):
        z = np.clip(xc @ w + b, -30, 30)
        g = s * (1.0 / (1.0 + np.exp(-z)) - y)
        w -= lr * (xc.T @ g + L2 * w)
        b -= lr * float(g.sum())
    return w, b - float(w @ mu)


def train_classifier(vecs, labels, ages_days):
    """L2-regularised logistic regression from article vectors to like (1) / dislike (0).

    Features are centred (embeddings all point in roughly the same direction) and the
    two classes weighted equally. Returns (weights, bias) for the uncentred vectors,
    so prediction is a single dot product.
    """
    return _train(vecs, labels, _sample_weights(labels, ages_days))


def predict(vecs, w, b) -> list[float]:
    """Probability that the reader likes each article."""
    if not vecs:
        return []
    m = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(vecs), -1)
    z = np.clip(m.astype(np.float64) @ np.asarray(w) + b, -30, 30)
    return (1.0 / (1.0 + np.exp(-z))).tolist()


def profile_match(vecs, wanted, unwanted) -> list[float]:
    """How well each article matches the profile's phrases (closest phrase on each side).

    With no rated articles, `score` reduces to exactly this match.
    """
    return score(vecs, wanted, unwanted, [], [])


def score_with_classifier(vecs, wanted, unwanted, w, b) -> list[float]:
    p = profile_match(vecs, wanted, unwanted)
    return [x + CLASSIFIER_WEIGHT * (q - 0.5) for x, q in zip(p, predict(vecs, w, b))]


def fit_from_db(db, dim: int):
    """Train on the reader's ratings. Returns (weights, bias, n_likes, n_dislikes);
    weights is None when there are too few of either."""
    rows = [r for r in db.training_vectors(MAX_TRAIN) if len(r["vec"]) == dim]
    labels = [1 if r["rating"] > 0 else 0 for r in rows]
    n_like = sum(labels)
    n_dislike = len(labels) - n_like
    if n_like < MIN_PER_CLASS or n_dislike < MIN_PER_CLASS:
        return None, 0.0, n_like, n_dislike
    now = time.time()
    w, b = train_classifier([r["vec"] for r in rows], labels,
                            [(now - r["received_at"]) / 86400 for r in rows])
    return w, b, n_like, n_dislike


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
        w, b, n_like, n_dislike = fit_from_db(db, dim)
        if w is None:
            db.set_state("classifier", "")
            log.info("classifier needs %d likes and %d dislikes (have %d and %d); "
                     "comparing with recent ratings instead",
                     MIN_PER_CLASS, MIN_PER_CLASS, n_like, n_dislike)
        else:
            db.set_state("classifier", _join([np.asarray(w, dtype=np.float32).tobytes()]))
            db.set_state("classifier_bias", repr(b))
            log.info("classifier trained on %d likes and %d dislikes", n_like, n_dislike)
        db.set_state("classifier_info", f"{n_like}:{n_dislike}")
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
    vecs = [r["vec"] for r in todo]
    model = db.get_state("classifier", "")
    if model:
        w = np.frombuffer(base64.b64decode(model), dtype=np.float32)
        scores = score_with_classifier(vecs, wanted, unwanted, w,
                                       float(db.get_state("classifier_bias", "0")))
        log.info("pre-filter scored %d articles with the classifier", len(todo))
    else:
        likes = [v for v in db.rated_vectors(1, N_LIKES) if len(v) == dim]
        dislikes = [v for v in db.rated_vectors(-1, N_DISLIKES) if len(v) == dim]
        scores = score(vecs, wanted, unwanted, likes, dislikes)
        log.info("pre-filter scored %d articles (%d likes, %d dislikes as reference)",
                 len(todo), len(likes), len(dislikes))
    db.set_prefilter([(s, version, r["id"]) for s, r in zip(scores, todo)])
    return len(todo)
