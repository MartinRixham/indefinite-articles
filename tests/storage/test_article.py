import dataclasses
import unittest

from indefinite_articles.fetching.entry import Entry
from indefinite_articles.storage.article import (BLURB_CHARACTERS, EMBEDDING_SUMMARY_CHARACTERS, TITLE_KEY_CHARACTERS,
                                                 Article, title_key)
from indefinite_articles.storage.database import Database


def make_article(**fields: object) -> Article:
    """An article with an id, url and title, and any other fields given."""
    return dataclasses.replace(Article(1, "https://example.com/1", "Title"), **fields)


class TitleKeyTests(unittest.TestCase):
    def test_keeps_only_lowercase_letters_and_digits(self) -> None:
        self.assertEqual(title_key("Rust 1.90: What's New?"), "rust190whatsnew")

    def test_same_story_with_different_punctuation_and_case(self) -> None:
        self.assertEqual(title_key("Rust 1.90 — what’s new"), title_key("RUST 1.90: WHATS NEW"))

    def test_shortened(self) -> None:
        self.assertEqual(len(title_key("word " * 100)), TITLE_KEY_CHARACTERS)

    def test_empty_without_letters_or_digits(self) -> None:
        self.assertEqual(title_key("?!"), "")


class DefaultsTests(unittest.TestCase):
    def test_new_article_has_no_text_score_or_prefilter(self) -> None:
        article = Article(1, "https://example.com/1", "Title")
        self.assertEqual((article.summary, article.source, article.category), ("", "", ""))
        self.assertIsNone(article.published)
        self.assertEqual(article.fetched_at, 0.0)
        for name in ("content", "content_status", "score", "reason", "score_profile_id", "prefilter"):
            self.assertIsNone(getattr(article, name), name)

    def test_frozen(self) -> None:
        article = make_article()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(article, "title", "Other")

    def test_equal_by_value(self) -> None:
        self.assertEqual(make_article(score=7.0), make_article(score=7.0))
        self.assertNotEqual(make_article(score=7.0), make_article(score=6.0))
        self.assertEqual(len({make_article(), make_article()}), 1)


class FromRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)
        self.database.add_article(Entry("https://example.com/1", "Title", "Summary.", 1000.0), "Source", "tech")
        self.database.connection.execute(
            "UPDATE articles SET content='Page text', content_status='ok', score=7.5, reason='fits',"
            " score_profile_id=3, prefilter=0.25")

    def test_every_field_is_an_articles_column(self) -> None:
        columns = {row["name"] for row in self.database.connection.execute("PRAGMA table_info(articles)")}
        self.assertLessEqual({field.name for field in dataclasses.fields(Article)}, columns)

    def test_reads_every_field(self) -> None:
        row = self.database.connection.execute("SELECT * FROM articles").fetchone()
        article = Article.from_row(row)
        self.assertEqual(article, Article(
            id=row["id"], url="https://example.com/1", title="Title", summary="Summary.", source="Source",
            category="tech", published=1000.0, fetched_at=row["fetched_at"], content="Page text",
            content_status="ok", score=7.5, reason="fits", score_profile_id=3, prefilter=0.25))

    def test_ignores_other_columns(self) -> None:
        row = self.database.connection.execute("SELECT *, 42 AS position FROM articles").fetchone()
        self.assertEqual(Article.from_row(row).title, "Title")

    def test_missing_column_is_an_error(self) -> None:
        row = self.database.connection.execute("SELECT id, url, title FROM articles").fetchone()
        with self.assertRaises(IndexError):
            Article.from_row(row)


class SourceLabelTests(unittest.TestCase):
    def test_source_and_category(self) -> None:
        self.assertEqual(make_article(source="Blog", category="tech").source_label, "Blog (tech)")

    def test_source_alone_without_category(self) -> None:
        self.assertEqual(make_article(source="Blog").source_label, "Blog")


class AwaitingFullTextTests(unittest.TestCase):
    def test_until_fetch_is_tried(self) -> None:
        self.assertTrue(make_article().awaiting_full_text)

    def test_not_after_success_or_failure(self) -> None:
        for status in ("ok", "empty", "failed: timeout", "blocked: private address"):
            self.assertFalse(make_article(content_status=status).awaiting_full_text, status)


class ScoreStateTests(unittest.TestCase):
    def test_unscored(self) -> None:
        article = make_article()
        self.assertFalse(article.scored_against(1))
        self.assertFalse(article.scored_against(None))
        self.assertFalse(article.has_stale_score(1))

    def test_scored_against_current_profile(self) -> None:
        article = make_article(score=7.0, score_profile_id=1)
        self.assertTrue(article.scored_against(1))
        self.assertFalse(article.has_stale_score(1))

    def test_scored_against_older_profile(self) -> None:
        article = make_article(score=7.0, score_profile_id=1)
        self.assertFalse(article.scored_against(2))
        self.assertTrue(article.has_stale_score(2))

    def test_any_score_is_stale_without_a_profile(self) -> None:
        self.assertTrue(make_article(score=7.0, score_profile_id=1).has_stale_score(None))

    def test_zero_is_a_score(self) -> None:
        self.assertTrue(make_article(score=0.0, score_profile_id=1).scored_against(1))


class DescriptionTests(unittest.TestCase):
    def test_page_text_preferred(self) -> None:
        article = make_article(summary="Short summary.", content="The longer page text of the article.")
        self.assertEqual(article.description(1000), "The longer page text of the article.")

    def test_summary_when_page_text_is_shorter(self) -> None:
        # e.g. a cookie banner instead of the article
        article = make_article(summary="A proper summary of the article.", content="Accept cookies")
        self.assertEqual(article.description(1000), "A proper summary of the article.")

    def test_summary_without_page_text(self) -> None:
        for content in (None, ""):
            self.assertEqual(make_article(summary="Summary.", content=content).description(1000), "Summary.")

    def test_page_text_without_summary(self) -> None:
        self.assertEqual(make_article(content="Page text.").description(1000), "Page text.")

    def test_empty(self) -> None:
        self.assertEqual(make_article().description(1000), "")

    def test_whitespace_collapsed(self) -> None:
        self.assertEqual(make_article(content="  one\n\ttwo   three ").description(1000), "one two three")

    def test_shortened_at_a_word_boundary(self) -> None:
        description = make_article(content="alpha beta gamma delta").description(15)
        self.assertEqual(description, "alpha beta…")
        self.assertLessEqual(len(description), 15)

    def test_exactly_at_limit_is_kept_whole(self) -> None:
        self.assertEqual(make_article(content="alpha beta").description(10), "alpha beta")


class BlurbTests(unittest.TestCase):
    def test_summary_preferred(self) -> None:
        article = make_article(summary="The summary.", content="The much longer page text of the article.")
        self.assertEqual(article.blurb(), "The summary.")

    def test_page_text_without_summary(self) -> None:
        self.assertEqual(make_article(content="Page  text.").blurb(), "Page text.")

    def test_empty(self) -> None:
        self.assertEqual(make_article().blurb(), "")

    def test_default_limit(self) -> None:
        blurb = make_article(summary="word " * 100).blurb()
        self.assertTrue(blurb.endswith("…"))
        self.assertLessEqual(len(blurb), BLURB_CHARACTERS)


class TemplateValuesTests(unittest.TestCase):
    def test_escapes_every_value(self) -> None:
        article = make_article(title="A & B", source="<Blog>", reason="Why", summary="Short summary.")
        self.assertEqual(article.template_values(str.upper), {
            "title": "A & B", "source": "<BLOG>", "url": "HTTPS://EXAMPLE.COM/1",
            "reason": "WHY", "blurb": "SHORT SUMMARY.", "length": "", "audio_url": "",
        })

    def test_missing_reason_and_blurb_are_empty(self) -> None:
        self.assertEqual(make_article().template_values(str),
                         {"title": "Title", "source": "", "url": "https://example.com/1", "reason": "", "blurb": "",
                          "length": "", "audio_url": ""})

    def test_podcast_episode_length_and_audio(self) -> None:
        values = make_article(audio_url="https://cdn.example.com/1.mp3", duration=2700.0).template_values(str)
        self.assertEqual((values["length"], values["audio_url"]), ("45 min", "https://cdn.example.com/1.mp3"))

    def test_audio_left_out_when_it_is_the_link(self) -> None:
        values = make_article(url="https://cdn.example.com/1.mp3", audio_url="https://cdn.example.com/1.mp3",
                              ).template_values(str)
        self.assertEqual(values["audio_url"], "")


class PodcastTests(unittest.TestCase):
    def test_is_podcast(self) -> None:
        self.assertFalse(make_article().is_podcast)
        self.assertTrue(make_article(audio_url="https://cdn.example.com/1.mp3").is_podcast)

    def test_length_label(self) -> None:
        for seconds, expected in ((None, ""), (20.0, "1 min"), (2700.0, "45 min"), (3600.0, "1 h"),
                                  (4830.0, "1 h 20 min")):
            with self.subTest(seconds=seconds):
                self.assertEqual(make_article(duration=seconds).length_label, expected)

    def test_given_limit(self) -> None:
        self.assertEqual(make_article(summary="alpha beta gamma").blurb(12), "alpha beta…")


class EmbeddingTextTests(unittest.TestCase):
    def test_title_and_summary(self) -> None:
        self.assertEqual(make_article(title="Title", summary="Summary.").embedding_text(), "Title. Summary.")

    def test_title_alone(self) -> None:
        self.assertEqual(make_article(title="Title").embedding_text(), "Title. ")

    def test_only_start_of_summary(self) -> None:
        text = make_article(title="Title", summary="x" * (EMBEDDING_SUMMARY_CHARACTERS + 100)).embedding_text()
        self.assertEqual(text, "Title. " + "x" * EMBEDDING_SUMMARY_CHARACTERS)

    def test_page_text_not_used(self) -> None:
        self.assertEqual(make_article(title="Title", content="Page text").embedding_text(), "Title. ")


if __name__ == "__main__":
    unittest.main()
