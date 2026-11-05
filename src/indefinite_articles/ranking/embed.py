"""Cheap first-pass ranking with embeddings.

With hundreds of feeds there are thousands of new articles a day, far more than a
local LLM can read. Every article is embedded (a few milliseconds each with a small
embedding model) and compared with the reader profile. Once the reader has rated
enough articles, a small logistic-regression classifier trained on the embeddings of
every rated article adds what it has learned; before that, articles are compared with
the most recent likes and dislikes. Only the best matches, plus a few random ones, go
on to full-text fetching and LLM scoring.
"""

import json
import sqlite3
import threading
import logging
import re
import time
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..language_model.assistant import MAX_PHRASES, Assistant
from ..language_model.llm import LLM, LLMError
from ..storage.database import Database
from .classifier import Classifier
from .profile_phrases import ProfilePhrases
from .vectors import max_similarities, pack, unpack

log = logging.getLogger(__name__)

REFERENCE_LIKES = 20      # most recent liked/disliked articles compared against, until the
REFERENCE_DISLIKES = 20   #   classifier has enough ratings to take over
MIN_PER_CLASS = 5           # likes and dislikes each needed to train the classifier
MAX_TRAIN = 5000            # most recent ratings used for training
EPOCHS = 100                # full-batch gradient descent steps
L2 = 1e-3                   # weight decay; also keeps probabilities modest with little data
HALF_LIFE_DAYS = 365        # older ratings count less, as tastes drift
CLASSIFIER_WEIGHT = 0.5     # weight of (probability - 0.5) next to the profile match
DEFAULT_WANTED = "substantive, well-written, informative articles"

# Words that mark a profile sentence as describing what the reader does NOT want.
_NEGATIVE = re.compile(
    r"\b(dislik\w*|avoid\w*|hates?|not|no|never|less|fewer|don't|doesn't|isn't|skip\w*|"
    r"boring|uninterest\w*|tired of|without|irrelevant|rather than|instead of)\b", re.I)
_CLAUSES = re.compile(r"\n+|(?<=[.!?;])\s+|,?\s+\b(?:but|except|however|whereas|although)\b\s*",
                      re.I)


@dataclass
class ClassifierFit:
    """The trained classifier (None when there are too few ratings of either kind)."""
    classifier: Classifier | None
    like_count: int
    dislike_count: int


def _split_profile(text: str) -> tuple[list[str], list[str]]:
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
    return likes[:MAX_PHRASES], avoid[:MAX_PHRASES]


def _split_phrases(llm: LLM, text: str) -> tuple[list[str], list[str]]:
    """Wanted and unwanted phrases from the profile: the LLM's split, else `_split_profile`."""
    try:
        return Assistant(llm).profile_phrases(text)
    except LLMError as error:
        log.info("profile split by LLM failed, using keyword split: %s", error)
    wanted, unwanted = _split_profile(text)
    wanted = wanted or [DEFAULT_WANTED]  # e.g. a profile that only lists dislikes
    return wanted, unwanted


def _embedding_setup(config: Config) -> str:
    """What the stored vectors depend on: the embedding model and the prefix it is given."""
    return f"{config.llm.embedding_model}:{config.llm.embedding_prefix!r}"


def embed_new(
    config: Config,
    database: Database,
    llm: LLM,
    stop: threading.Event | None = None,
    limit: int = 20000,
) -> int:
    if database.get_state("embedding_setup") != _embedding_setup(config):
        # Vectors from different models or prefixes aren't comparable: start again.
        # unembedded() then embeds the rated articles again too, whatever their age.
        database.connection.execute("DELETE FROM embeddings")
        database.connection.execute("UPDATE articles SET prefilter=NULL, prefilter_version=NULL")
        database.set_state("embedding_setup", _embedding_setup(config))
    rows = database.unembedded(config.max_article_age_days, limit)
    done = 0
    batch_size = max(1, config.llm.embedding_batch)
    for start in range(0, len(rows), batch_size):
        if stop is not None and stop.is_set():
            break
        batch = rows[start : start + batch_size]
        vectors = llm.embed([article.embedding_text() for article in batch])
        for article, vector in zip(batch, vectors):
            database.set_embedding(article.id, pack(vector))
        database.connection.commit()
        done += len(batch)
    if done:
        log.info("embedded %d articles", done)
    return done


def _score(vectors: list[bytes], phrases: ProfilePhrases, likes: list[bytes],
           dislikes: list[bytes]) -> list[float]:
    """Pre-filter score of each article vector, used before the classifier is trained.

    `phrases` are the profile's phrases (an article is judged by its closest phrase on
    each side); `likes`/`dislikes` are articles the reader rated.
    """
    if not vectors:
        return []
    matrix = unpack(vectors)
    matched = phrases.match(matrix)
    if likes:
        similarities = matrix @ unpack(likes).T
        top = min(2, similarities.shape[1])
        liked = np.sort(similarities, axis=1)[:, -top:].mean(axis=1)
    else:
        liked = matched
    disliked = max_similarities(matrix, dislikes)
    return (0.4 * matched + 0.6 * liked - 0.4 * disliked).tolist()


# -- classifier ----------------------------------------------------------------

def _sample_weights(labels: list[int], ages_days: list[float]) -> list[float]:
    """Each class gets half the total weight (likes usually outnumber dislikes), and
    within a class older ratings count less."""
    decay = [0.5 ** (max(0.0, age) / HALF_LIFE_DAYS) for age in ages_days]
    totals = {label_class: sum(weight for weight, label in zip(decay, labels) if label == label_class)
              for label_class in (0, 1)}
    return [0.5 * weight / totals[label] for weight, label in zip(decay, labels)]


def _train(vectors: list[bytes], labels: list[int], weights: list[float]) -> Classifier:
    features = unpack(vectors).astype(np.float64)
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


def _train_classifier(vectors: list[bytes], labels: list[int], ages_days: list[float]) -> Classifier:
    """L2-regularised logistic regression from article vectors to like (1) / dislike (0).

    Features are centred (embeddings all point in roughly the same direction) and the
    two classes weighted equally. The weights and bias are for the uncentred vectors,
    so prediction is a single dot product.
    """
    return _train(vectors, labels, _sample_weights(labels, ages_days))


def _score_with_classifier(vectors: list[bytes], phrases: ProfilePhrases,
                          classifier: Classifier) -> list[float]:
    if not vectors:
        return []
    matches = phrases.match(unpack(vectors)).tolist()
    probabilities = classifier.predict(vectors)
    return [match + CLASSIFIER_WEIGHT * (probability - 0.5) for match, probability in zip(matches, probabilities)]


def _fit_from_database(database: Database, dimension: int) -> ClassifierFit:
    """Train on the reader's ratings; no classifier when there are too few of either kind."""
    rows = [row for row in database.training_vectors(MAX_TRAIN) if len(row["vec"]) == dimension]
    labels = [1 if row["rating"] > 0 else 0 for row in rows]
    like_count = sum(labels)
    dislike_count = len(labels) - like_count
    if like_count < MIN_PER_CLASS or dislike_count < MIN_PER_CLASS:
        return ClassifierFit(None, like_count, dislike_count)
    now = time.time()
    classifier = _train_classifier([row["vec"] for row in rows], labels,
                                  [(now - row["received_at"]) / 86400 for row in rows])
    return ClassifierFit(classifier, like_count, dislike_count)


def _profile_phrases(config: Config, database: Database, llm: LLM, profile: sqlite3.Row) -> ProfilePhrases:
    """The profile's phrases, worked out again (maybe with an LLM call) only when the profile changes."""
    phrases_for = f"{profile['id']}:{config.llm.embedding_model}"
    stored = database.get_state("profile_phrases")
    if stored and database.get_state("profile_phrases_for") == phrases_for:
        data = json.loads(stored)
        wanted, unwanted = data["wanted"], data["unwanted"]
    else:
        wanted, unwanted = _split_phrases(llm, profile["text"])
        log.info("profile phrases: wanted %s; unwanted %s", wanted, unwanted)
        database.set_state("profile_phrases", json.dumps({"wanted": wanted, "unwanted": unwanted}))
        database.set_state("profile_phrases_for", phrases_for)
        database.connection.commit()
    return ProfilePhrases(wanted, unwanted, llm)


def update_prefilter(config: Config, database: Database, llm: LLM, profile: sqlite3.Row) -> int:
    """(Re)compute the pre-filter score of every embedded candidate article.

    Everything is rescored when the profile or the feedback changes; otherwise only
    newly embedded articles are scored.
    """
    phrases = _profile_phrases(config, database, llm, profile)
    signature = f"{profile['id']}:{database.feedback_signature()}:{_embedding_setup(config)}"
    version = int(database.get_state("prefilter_version", "0"))
    changed = database.get_state("prefilter_sig") != signature
    if not changed and not database.prefilter_todo(config.max_article_age_days, version):
        return 0
    dimension = phrases.dimension()
    if changed:
        fit = _fit_from_database(database, dimension)
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
    # Ignore vectors from a different model (e.g. after changing embedding_model).
    todo = [row for row in todo if len(row["vec"]) == dimension]
    vectors = [row["vec"] for row in todo]
    classifier = Classifier.load(database)
    if classifier is not None:
        scores = _score_with_classifier(vectors, phrases, classifier)
        log.info("pre-filter scored %d articles with the classifier", len(todo))
    else:
        likes = [vector for vector in database.rated_vectors(1, REFERENCE_LIKES) if len(vector) == dimension]
        dislikes = [vector for vector in database.rated_vectors(-1, REFERENCE_DISLIKES) if len(vector) == dimension]
        scores = _score(vectors, phrases, likes, dislikes)
        log.info("pre-filter scored %d articles (%d likes, %d dislikes as reference)",
                 len(todo), len(likes), len(dislikes))
    database.set_prefilter({row["id"]: prefilter_score for prefilter_score, row in zip(scores, todo)}, version)
    return len(todo)
