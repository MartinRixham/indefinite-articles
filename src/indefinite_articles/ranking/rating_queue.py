"""Articles to show the reader for rating at the terminal (`indefinite-articles rate`).

The digest only shows what already matches the profile, so the reader can't tell it about
anything else they would like. The rating queue therefore ignores the profile and the
classifier: it picks a random feed, then a random article from it, so busy feeds don't
dominate and every subject the feeds cover gets a look in.
"""

import random
from collections.abc import Collection, Iterator

from ..config import Config
from ..storage.article import Article
from ..storage.database import Database


def rating_queue(config: Config, database: Database, randomizer: random.Random | None = None,
                 categories: Collection[str] = ()) -> Iterator[Article]:
    """Unsent, unrated recent articles in random order, each once: a random feed first,
    then a random article from it.

    With `categories`, only articles from feeds in one of those categories (ignoring case)."""
    randomizer = randomizer or random.Random()
    wanted = {category.lower() for category in categories}
    by_source: dict[str, list[Article]] = {}
    for article in database.pool(config.max_article_age_days):
        if wanted and article.category.lower() not in wanted:
            continue
        by_source.setdefault(article.source, []).append(article)
    for articles in by_source.values():
        randomizer.shuffle(articles)
    while by_source:
        source = randomizer.choice(sorted(by_source))
        yield by_source[source].pop()
        if not by_source[source]:
            del by_source[source]
