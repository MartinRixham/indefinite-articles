import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from unittest import mock

from indefinite_articles.config import (Config, ConfigError, EmailConfig, LLMConfig, ServerConfig, TimeOfDay,
                                        load)
from indefinite_articles.fetching.feed import Feed


class SectionTests(unittest.TestCase):
    def test_llm_section_converts_types_and_strips_slash(self):
        llm = LLMConfig.from_section({"base_url": "http://host:8080/v1/", "max_prompt_chars": "9000",
                                      "require_gpu": 0, "pull_models": False}, Path("."))
        self.assertEqual(llm.base_url, "http://host:8080/v1")
        self.assertEqual(llm.max_prompt_chars, 9000)
        self.assertIs(llm.require_gpu, False)
        self.assertIs(llm.pull_models, False)

    def test_api_key_file_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "key").write_text("secret-key\n")
            llm = LLMConfig.from_section({"api_key": "ignored", "api_key_file": "key"}, Path(directory))
        self.assertEqual(llm.api_key, "secret-key")

    def test_server_default_security_depends_on_port(self):
        self.assertEqual(ServerConfig.from_section({}, 993, Path(".")).security, "ssl")
        self.assertEqual(ServerConfig.from_section({}, 587, Path(".")).security, "starttls")
        self.assertEqual(ServerConfig.from_section({"security": "none"}, 465, Path(".")).security, "none")

    def test_server_rejects_unknown_security(self):
        with self.assertRaises(ConfigError):
            ServerConfig.from_section({"security": "tls"}, 587, Path("."))

    def test_server_password_file_absolute(self):
        with tempfile.TemporaryDirectory() as directory:
            password_path = Path(directory, "password")
            password_path.write_text("  hunter2  \n")
            server = ServerConfig.from_section({"host": "mail", "port": "2525", "username": "me",
                                                "password_file": str(password_path)}, 587, Path("/nowhere"))
        self.assertEqual((server.host, server.port, server.username, server.password),
                         ("mail", 2525, "me", "hunter2"))

    def test_email_allowed_senders_default_to_recipient(self):
        email = EmailConfig.from_section({"to": "Me@Example.com", "from": "bot@example.com"}, Path("."))
        self.assertEqual(email.allowed_senders, ["me@example.com"])
        self.assertEqual(email.sender, "bot@example.com")
        self.assertEqual(email.imap_folder, "INBOX")

    def test_email_explicit_senders_and_imap_folders(self):
        email = EmailConfig.from_section({"to": "me@example.com", "allowed_senders": ["A@x.org"],
                                          "imap": {"folder": "Replies", "processed_folder": "Done"},
                                          "send_acknowledgement": False, "debug_info": False}, Path("."))
        self.assertEqual(email.allowed_senders, ["a@x.org"])
        self.assertEqual((email.imap_folder, email.processed_folder), ("Replies", "Done"))
        self.assertFalse(email.send_acknowledgement)
        self.assertFalse(email.debug_info)


class EmailConfigTests(unittest.TestCase):
    def setUp(self):
        self.email = EmailConfig(to="you@example.com", sender="Bot <bot@mail.example.org>",
                                 allowed_senders=["you@example.com"])

    def test_new_message_headers(self):
        message = self.email.new_message("digest")
        self.assertEqual(message["From"], "Bot <bot@mail.example.org>")
        self.assertEqual(message["To"], "you@example.com")
        self.assertIsNotNone(message["Date"])
        self.assertTrue(message["Message-ID"].endswith("@mail.example.org>"))
        self.assertIn("digest", message["Message-ID"])
        self.assertNotEqual(message["Message-ID"], self.email.new_message("digest")["Message-ID"])

    def test_new_message_without_domain(self):
        self.email.sender = ""
        self.assertTrue(self.email.new_message("ack")["Message-ID"].endswith("@localhost>"))

    def test_is_allowed_sender(self):
        message = EmailMessage()
        message["From"] = "You <YOU@example.com>"
        self.assertTrue(self.email.is_allowed_sender(message))
        message.replace_header("From", "someone@example.com")
        self.assertFalse(self.email.is_allowed_sender(message))
        self.assertFalse(self.email.is_allowed_sender(EmailMessage()))


class ConfigTests(unittest.TestCase):
    def test_digest_hour_minute(self):
        config = Config(digest_time=" 7:05 ")
        self.assertEqual(config.digest_hour_minute, TimeOfDay(7, 5))

    def test_time_of_day(self):
        seven = TimeOfDay(7, 5)
        self.assertFalse(seven.has_passed(datetime(2026, 11, 3, 7, 4, 59)))
        self.assertTrue(seven.has_passed(datetime(2026, 11, 3, 7, 5)))
        self.assertEqual(seven.next_after(datetime(2026, 11, 3, 7, 4, 59)), datetime(2026, 11, 3, 7, 5))
        self.assertEqual(seven.next_after(datetime(2026, 11, 3, 7, 5)), datetime(2026, 11, 4, 7, 5))

    def test_digest_hour_minute_rejects_bad_times(self):
        for value in ("24:00", "07:60", "7am", "", "07:5"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                Config(digest_time=value).digest_hour_minute

    def test_next_digest_time_is_tomorrow_once_passed(self):
        now = time.localtime()
        config = Config(digest_time=f"{now.tm_hour}:{now.tm_min:02d}")
        ahead = config.next_digest_time() - time.time()
        self.assertGreater(ahead, 82000)  # about a day (allowing for a DST change)
        self.assertLess(ahead, 90100)

    def test_problems(self):
        self.assertEqual(len(Config().problems()), 4)
        config = Config(feeds=[Feed("https://example.com/feed")])
        config.email = EmailConfig(to="a@x", sender="b@x")
        config.email.smtp.host = config.email.imap.host = "mail"
        self.assertEqual(config.problems(), [])
        config.email.imap.host = ""
        self.assertEqual(len(config.problems()), 1)
        self.assertIn("imap", config.problems()[0])


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def _write(self, text: str) -> Path:
        path = self.directory / "config.toml"
        path.write_text(text)
        return path

    def test_missing_file(self):
        with self.assertRaisesRegex(ConfigError, "not found"):
            load(self.directory / "missing.toml")

    def test_invalid_toml(self):
        with self.assertRaises(ConfigError):
            load(self._write("[general\n"))

    def test_invalid_digest_time(self):
        with self.assertRaises(ConfigError):
            load(self._write('[general]\ndigest_time = "noon"\n'))

    def test_environment_variable_names_config(self):
        path = self._write('[general]\narticles_per_digest = 4\n')
        with mock.patch.dict(os.environ, {"INDEFINITE_ARTICLES_CONFIG": str(path)}):
            self.assertEqual(load().articles_per_digest, 4)

    def test_settings_profile_and_feeds(self):
        feeds_directory = self.directory / "elsewhere"
        feeds_directory.mkdir()
        (feeds_directory / "list.txt").write_text("# News\nhttps://n/feed\n")
        config = load(self._write(
            '[general]\nmax_candidates = 20\nexclude_categories = ["Sports"]\n'
            f'feeds_dir = "{feeds_directory}"\n'
            '[profile]\ninterests = """\n  Rust and compilers\n"""\n'
            '[llm]\nmodel = "small"\n'
            '[[feeds]]\nurl = "https://kept/feed"\n'
            '[[feeds]]\nurl = "https://table/feed"\ncategory = "sports"\n'
            '[[feeds]]\nname = "no url"\n'))
        self.assertEqual(config.max_candidates, 20)
        self.assertEqual(config.exclude_categories, ["sports"])
        self.assertEqual(config.interests, "Rust and compilers")
        self.assertEqual(config.llm.model, "small")
        self.assertEqual([feed.url for feed in config.feeds], ["https://kept/feed", "https://n/feed"])

    def test_config_file_overrides_tuning_file(self):
        (self.directory / "tuning.toml").write_text(
            '[general]\nmax_candidates = 20\nrescore_top = 5\n[llm]\nmodel = "tuned"\ntemperature = 0.5\n')
        config = load(self._write('[general]\nmax_candidates = 40\n[llm]\nmodel = "mine"\n'))
        self.assertEqual((config.max_candidates, config.rescore_top), (40, 5))
        self.assertEqual((config.llm.model, config.llm.temperature), ("mine", 0.5))

    def test_invalid_tuning_file(self):
        (self.directory / "tuning.toml").write_text("[llm\n")
        with self.assertRaisesRegex(ConfigError, "tuning.toml"):
            load(self._write(""))

    def test_plain_string_feeds(self):
        config = load(self._write('feeds = ["https://plain/feed"]\n'))
        self.assertEqual([(feed.url, feed.name) for feed in config.feeds], [("https://plain/feed", "")])


class FeedListTests(unittest.TestCase):
    """Feed lists in the feeds directory next to the config file."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.feeds_directory = self.directory / "feeds.d"
        self.feeds_directory.mkdir()

    def _feeds(self) -> list[Feed]:
        path = self.directory / "config.toml"
        path.write_text("")
        return load(path).feeds

    def test_opml_without_body(self):
        (self.feeds_directory / "empty.opml").write_bytes(b"<opml><head/></opml>")
        self.assertEqual(self._feeds(), [])

    def test_opml_bad_max_items_uses_default(self):
        (self.feeds_directory / "list.opml").write_bytes(
            b'<opml><body><outline text=" A " xmlUrl=" https://a/feed " maxItems="many"/>'
            b'<other xmlUrl="https://ignored"/></body></opml>')
        self.assertEqual([(feed.url, feed.name, feed.max_items) for feed in self._feeds()],
                         [("https://a/feed", "A", 40)])

    def test_feed_list_ignores_blank_lines(self):
        (self.feeds_directory / "list.txt").write_text("\n  \nhttps://a/feed\n")
        self.assertEqual([feed.url for feed in self._feeds()], ["https://a/feed"])

    def test_feed_directory(self):
        shutil.rmtree(self.feeds_directory)
        self.assertEqual(self._feeds(), [])  # no feeds directory
        self.feeds_directory.mkdir()
        (self.feeds_directory / "b.txt").write_text("https://b/feed\n")
        (self.feeds_directory / "a.opml").write_bytes(b'<opml><body><outline xmlUrl="https://a/feed"/></body></opml>')
        (self.feeds_directory / "notes.md").write_text("https://ignored/feed\n")
        self.assertEqual([feed.url for feed in self._feeds()], ["https://a/feed", "https://b/feed"])
        (self.feeds_directory / "c.xml").write_text("<opml><body>")
        with self.assertRaisesRegex(ConfigError, "c.xml"):
            self._feeds()


if __name__ == "__main__":
    unittest.main()
