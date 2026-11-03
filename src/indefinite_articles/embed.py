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
from dataclasses import dataclass

import numpy as np

from .classifier import Classifier
from .llm import LLM, LLMError

log = logging.getLogger(__name__)

REFERENCE_LIKES = 20      # most recent liked/disliked articles compared against, until the
REFERENCE_DISLIKES = 20   #   classifier has enough ratings to take over
MIN_PER_CLASS = 5           # likes and dislikes each needed to train the classifier
MAX_TRAIN = 5000            # most recent ratings used for training
EPOCHS = 100                # full-batch gradient descent steps
L2 = 1e-3                   # weight decay; also keeps probabilities modest with little data
HALF_LIFE_DAYS = 365        # older ratings count less, as tastes drift
CLASSIFIER_WEIGHT = 0.5     # weight of (probability - 0.5) next to the profile match
MAX_PHRASES = 12  # per side, from the reader profile
DEFAULT_WANTED = "substantive, well-written, informative articles"

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

# Words that mark a profile sentence as describing what the reader does NOT want.
_NEGATIVE = re.compile(
    r"\b(dislik\w*|avoid\w*|hates?|not|no|never|less|fewer|don't|doesn't|isn't|skip\w*|"
    r"boring|uninterest\w*|tired of|without|irrelevant|rather than|instead of)\b", re.I)
_CLAUSES = re.compile(r"\n+|(?<=[.!?;])\s+|,?\s+\b(?:but|except|however|whereas|although)\b\s*",
                      re.I)


@dataclass
class ProfilePhrases:
    """What the reader profile says the reader wants and doesn't want, as short phrases."""
    wanted: list[str]
    unwanted: list[str]


@dataclass
class ClassifierFit:
    """The trained classifier (None when there are too few ratings of either kind)."""
    classifier: Classifier | None
    like_count: int
    dislike_count: int


def pack(vector) -> bytes:
    """Unit-length float32 bytes, as stored in the embeddings table."""
    array = np.asarray(vector, dtype=np.float32)
    return (array / (np.linalg.norm(array) or 1.0)).astype(np.float32).tobytes()


def article_text(article) -> str:
    return f"{article['title']}. {(article['summary'] or '')[:600]}"


def split_profile(text: str) -> ProfilePhrases:
    """Split a free-text profile into wanted and unwanted phrases, without an LLM.

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
    return ProfilePhrases(likes[:MAX_PHRASES], avoid[:MAX_PHRASES])


def profile_phrases(llm: LLM, text: str) -> ProfilePhrases:
    """Wanted and unwanted phrases from the profile: the LLM's split, else `split_profile`."""
    try:
        prompt = PREFERENCES_PROMPT.format(profile=text[:4000], max_phrases=MAX_PHRASES)
        data = llm.chat_json(PREFERENCES_SYSTEM, prompt)
    except LLMError as error:
        log.info("profile split by LLM failed, using keyword split: %s", error)
        data = None
    if isinstance(data, dict):
        likes, avoid = _phrase_list(data, "likes"), _phrase_list(data, "avoid")
        if likes:
            return ProfilePhrases(likes, avoid)
    return split_profile(text)


def _phrase_list(data: dict, key: str) -> list[str]:
    items = data.get(key)
    if not isinstance(items, list):
        return []
    return [str(item).strip()[:200] for item in items if str(item).strip()][:MAX_PHRASES]


def embed_new(config, database, llm: LLM, stop=None, limit: int = 20000) -> int:
    if database.get_state("embedding_model") != config.llm.embedding_model:
        # Vectors from different models aren't comparable: start again.
        database.connection.execute("DELETE FROM embeddings")
        database.connection.execute("UPDATE articles SET prefilter=NULL, prefilter_version=NULL")
        database.set_state("embedding_model", config.llm.embedding_model)
    rows = database.unembedded(config.max_article_age_days, limit)
    done = 0
    batch_size = max(1, config.llm.embedding_batch)
    for start in range(0, len(rows), batch_size):
        if stop is not None and stop.is_set():
            break
        batch = rows[start : start + batch_size]
        vectors = llm.embed([article_text(article) for article in batch])
        for article, vector in zip(batch, vectors):
            database.set_embedding(article["id"], pack(vector))
        database.connection.commit()
        done += len(batch)
    if done:
        log.info("embedded %d articles", done)
    return done


def _max_similarities(matrix, refs):
    if not refs:
        return np.zeros(len(matrix), dtype=np.float32)
    return (matrix @ np.frombuffer(b"".join(refs), dtype=np.float32).reshape(len(refs), -1).T).max(axis=1)


def _scores(vectors, wanted, unwanted, likes, dislikes):
    matrix = np.frombuffer(b"".join(vectors), dtype=np.float32).reshape(len(vectors), -1)
    matched = _max_similarities(matrix, wanted) - 0.5 * _max_similarities(matrix, unwanted)
    if likes:
        similarities = matrix @ np.frombuffer(b"".join(likes), dtype=np.float32).reshape(len(likes), -1).T
        top = min(2, similarities.shape[1])
        liked = np.sort(similarities, axis=1)[:, -top:].mean(axis=1)
    else:
        liked = matched
    disliked = _max_similarities(matrix, dislikes)
    return (0.4 * matched + 0.6 * liked - 0.4 * disliked).tolist()


def score(vectors, wanted, unwanted, likes, dislikes):
    """Pre-filter score of each article vector, used before the classifier is trained.

    `wanted`/`unwanted` are the profile's phrases (an article is judged by its closest
    phrase on each side); `likes`/`dislikes` are articles the reader rated.
    """
    if not vectors:
        return []
    return _scores(vectors, wanted, unwanted, likes, dislikes)


# -- classifier ----------------------------------------------------------------

def _sample_weights(labels, ages_days) -> list[float]:
    """Each class gets half the total weight (likes usually outnumber dislikes), and
    within a class older ratings count less."""
    decay = [0.5 ** (max(0.0, age) / HALF_LIFE_DAYS) for age in ages_days]
    totals = {label_class: sum(weight for weight, label in zip(decay, labels) if label == label_class)
              for label_class in (0, 1)}
    return [0.5 * weight / totals[label] for weight, label in zip(decay, labels)]


def _train(vectors, labels, weights) -> Classifier:
    features = np.frombuffer(b"".join(vectors), dtype=np.float32).reshape(len(vectors), -1).astype(np.float64)
    target = np.asarray(labels, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    mean = features.mean(axis=0)
    centred = features - mean
    learning_rate = 1.0 / (0.25 * (float((centred * centred).sum(axis=1).max()) + 1.0) + L2)
    coefficients, bias = np.zeros(features.shape[1]), 0.0
    for _ in range(EPOCHS):
        logits = np.clip(centred @ coefficients + bias, -30, 30)
        gradient = sample_weights * (1.0 / (1.0 + np.exp(-logits)) - target)
        coefficients -= learning_rate * (centred.T @ gradient + L2 * coefficients)
        bias -= learning_rate * float(gradient.sum())
    return Classifier(coefficients, bias - float(coefficients @ mean))


def train_classifier(vectors, labels, ages_days) -> Classifier:
    """L2-regularised logistic regression from article vectors to like (1) / dislike (0).

    Features are centred (embeddings all point in roughly the same direction) and the
    two classes weighted equally. The weights and bias are for the uncentred vectors,
    so prediction is a single dot product.
    """
    return _train(vectors, labels, _sample_weights(labels, ages_days))


def profile_match(vectors, wanted, unwanted) -> list[float]:
    """How well each article matches the profile's phrases (closest phrase on each side).

    With no rated articles, `score` reduces to exactly this match.
    """
    return score(vectors, wanted, unwanted, [], [])


def score_with_classifier(vectors, wanted, unwanted, classifier: Classifier) -> list[float]:
    matches = profile_match(vectors, wanted, unwanted)
    probabilities = classifier.predict(vectors)
    return [match + CLASSIFIER_WEIGHT * (probability - 0.5) for match, probability in zip(matches, probabilities)]


def fit_from_database(database, dimension: int) -> ClassifierFit:
    """Train on the reader's ratings; no classifier when there are too few of either kind."""
    rows = [row for row in database.training_vectors(MAX_TRAIN) if len(row["vec"]) == dimension]
    labels = [1 if row["rating"] > 0 else 0 for row in rows]
    like_count = sum(labels)
    dislike_count = len(labels) - like_count
    if like_count < MIN_PER_CLASS or dislike_count < MIN_PER_CLASS:
        return ClassifierFit(None, like_count, dislike_count)
    now = time.time()
    classifier = train_classifier([row["vec"] for row in rows], labels,
                                  [(now - row["received_at"]) / 86400 for row in rows])
    return ClassifierFit(classifier, like_count, dislike_count)


def _join(vectors: list[bytes]) -> str:
    return base64.b64encode(b"".join(vectors)).decode()


def _split(encoded: str, dimension: int) -> list[bytes]:
    raw = base64.b64decode(encoded)
    return [raw[start : start + dimension] for start in range(0, len(raw), dimension)]


def update_prefilter(config, database, llm: LLM, profile) -> int:
    """(Re)compute the pre-filter score of every embedded candidate article.

    Everything is rescored when the profile or the feedback changes; otherwise only
    newly embedded articles are scored.
    """
    signature = f"{profile['id']}:{database.feedback_signature()}:{config.llm.embedding_model}"
    version = int(database.get_state("prefilter_version", "0"))
    dimension = int(database.get_state("profile_dim", "0"))
    if database.get_state("prefilter_sig") != signature or not dimension:
        phrases_for = f"{profile['id']}:{config.llm.embedding_model}"
        if database.get_state("profile_phrases_for") != phrases_for:
            # Only when the profile itself changes: this may cost an LLM call.
            phrases = profile_phrases(llm, profile["text"])
            wanted, unwanted = phrases.wanted, phrases.unwanted
            if not wanted:  # e.g. a profile that only lists dislikes
                wanted = [DEFAULT_WANTED]
            log.info("profile phrases: wanted %s; unwanted %s", wanted, unwanted)
            vectors = [pack(vector) for vector in llm.embed(wanted + unwanted)]
            dimension = len(vectors[0])
            database.set_state("profile_wanted", _join(vectors[: len(wanted)]))
            database.set_state("profile_unwanted", _join(vectors[len(wanted):]))
            database.set_state("profile_dim", str(dimension))
            database.set_state("profile_phrases_for", phrases_for)
            database.connection.commit()
        fit = fit_from_database(database, dimension)
        if fit.classifier is None:
            database.set_state("classifier", "")
            log.info("classifier needs %d likes and %d dislikes (have %d and %d); "
                     "comparing with recent ratings instead",
                     MIN_PER_CLASS, MIN_PER_CLASS, fit.like_count, fit.dislike_count)
        else:
            fit.classifier.save(database)
            log.info("classifier trained on %d likes and %d dislikes", fit.like_count, fit.dislike_count)
        database.set_state("classifier_info", f"{fit.like_count}:{fit.dislike_count}")
        version += 1
        database.set_state("prefilter_version", str(version))
        database.set_state("prefilter_sig", signature)
    todo = database.prefilter_todo(config.max_article_age_days, version)
    if not todo:
        return 0
    wanted = _split(database.get_state("profile_wanted", ""), dimension)
    unwanted = _split(database.get_state("profile_unwanted", ""), dimension)
    # Ignore vectors from a different model (e.g. after changing embedding_model).
    todo = [row for row in todo if len(row["vec"]) == dimension]
    vectors = [row["vec"] for row in todo]
    classifier = Classifier.load(database)
    if classifier is not None:
        scores = score_with_classifier(vectors, wanted, unwanted, classifier)
        log.info("pre-filter scored %d articles with the classifier", len(todo))
    else:
        likes = [vector for vector in database.rated_vectors(1, REFERENCE_LIKES) if len(vector) == dimension]
        dislikes = [vector for vector in database.rated_vectors(-1, REFERENCE_DISLIKES) if len(vector) == dimension]
        scores = score(vectors, wanted, unwanted, likes, dislikes)
        log.info("pre-filter scored %d articles (%d likes, %d dislikes as reference)",
                 len(todo), len(likes), len(dislikes))
    database.set_prefilter([(prefilter_score, version, row["id"]) for prefilter_score, row in zip(scores, todo)])
    return len(todo)
