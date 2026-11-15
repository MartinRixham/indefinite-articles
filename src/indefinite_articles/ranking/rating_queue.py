"""Articles to show the reader for rating at the terminal (`indefinite-articles rate`).

The digest only shows what already matches the profile, so the reader can't tell it about
anything else they would like. The rating queue mixes three kinds of article from the
whole pool, taking turns:

- close: the best pre-filter matches that missed today's shortlist, so the LLM never
  saw them;
- unsure: those the classifier can least decide about (probability nearest 0.5), where
  a rating teaches it most; only once it has been trained;
- random: a random feed, then a random article from it, so busy feeds don't dominate
  and subjects the profile doesn't mention get a look in.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass

from ..config import Config
from ..storage.article import Article
from ..storage.database import Database
from . import ranker
from .classifier import Classifier

CLOSE = "close match"
UNSURE = "unsure"
RANDOM = "random"


@dataclass(frozen=True)
class QueueItem:
    """An article to rate, and which kind of pick it is (`CLOSE`, `UNSURE` or `RANDOM`)."""
    article: Article
    kind: str


def _close(pool: list[Article], shortlisted: set[int]) -> Iterator[Article]:
    ranked = sorted((article for article in pool if article.prefilter is not None and article.id not in shortlisted),
                    key=lambda article: article.prefilter or 0.0, reverse=True)
    yield from ranked


def _unsure(database: Database, pool: list[Article]) -> Iterator[Article]:
    classifier = Classifier.load(database)
    if classifier is None:
        return
    vectors = database.vectors_of(article.id for article in pool)
    dimension = len(classifier.weights) * 4  # float32 bytes
    embedded = [article for article in pool if len(vectors.get(article.id, b"")) == dimension]
    probabilities = classifier.predict([vectors[article.id] for article in embedded])
    yield from (article for _, article in sorted(zip(probabilities, embedded),
                                                 key=lambda pair: abs(pair[0] - 0.5)))


def _random(pool: list[Article], randomizer: random.Random) -> Iterator[Article]:
    by_source: dict[str, list[Article]] = {}
    for article in pool:
        by_source.setdefault(article.source, []).append(article)
    for articles in by_source.values():
        randomizer.shuffle(articles)
    while by_source:
        source = randomizer.choice(sorted(by_source))
        yield by_source[source].pop()
        if not by_source[source]:
            del by_source[source]


def rating_queue(config: Config, database: Database, randomizer: random.Random | None = None) -> Iterator[QueueItem]:
    """Unsent, unrated recent articles in the order to show them, each once, taking turns
    between the close, unsure and random kinds (skipping any that has run out)."""
    randomizer = randomizer or random.Random()
    pool = database.pool(config.max_article_age_days)
    shortlisted = {article.id for article in ranker.shortlist(config, database)}
    streams = [(CLOSE, _close(pool, shortlisted)), (UNSURE, _unsure(database, pool)),
               (RANDOM, _random(pool, randomizer))]
    seen: set[int] = set()
    while streams:
        for stream in list(streams):
            kind, articles = stream
            article = next((article for article in articles if article.id not in seen), None)
            if article is None:
                streams.remove(stream)
                continue
            seen.add(article.id)
            yield QueueItem(article, kind)
