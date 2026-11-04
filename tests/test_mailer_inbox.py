import unittest
from email.message import EmailMessage
from unittest import mock

from indefinite_articles import inbox, mailer
from indefinite_articles.article import Article
from indefinite_articles.config import Config
from indefinite_articles.database import Database, SourceStatistics
from indefinite_articles.feeds import Entry
from indefinite_articles.llm import LLM
from indefinite_articles.ranker import DigestPick


def make_config() -> Config:
    config = Config()
    config.email.sender, config.email.to = "Bot <bot@example.org>", "you@example.com"
    config.email.allowed_senders = ["you@example.com"]
    config.email.smtp.host = "smtp.example.org"
    config.email.smtp.port = 465
    config.email.imap.host = "imap.example.org"
    return config


def reply(text: str, subject: str = "Re: list", **headers: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = "you@example.com"
    message["Subject"] = subject
    for name, value in headers.items():
        message[name.replace("_", "-")] = value
    message.set_content(text)
    return message


class MailerTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_acknowledgement_threads_the_reply(self):
        original = reply("+1", subject="Re: Your reading list", Message_ID="<r1@x>", References="<d1@x>")
        message = mailer.build_acknowledgement(self.config, original, ["👍 liked #1: T"])
        self.assertEqual(message["Subject"], "Re: Your reading list")
        self.assertEqual(message["In-Reply-To"], "<r1@x>")
        self.assertEqual(message["References"], "<d1@x> <r1@x>")
        self.assertEqual(message["Auto-Submitted"], "auto-replied")
        self.assertEqual(message["X-Indefinite-Articles-Ack"], "1")
        self.assertIn("👍 liked #1: T", mailer.part_text(message))

    def test_acknowledgement_without_subject_or_id(self):
        original = EmailMessage()
        message = mailer.build_acknowledgement(self.config, original, [])
        self.assertEqual(message["Subject"], "Re: your feedback")
        self.assertIsNone(message["In-Reply-To"])

    def test_part_text_missing_part(self):
        message = EmailMessage()
        message.set_content("plain only")
        with self.assertRaises(ValueError):
            mailer.part_text(message, "html")

    def test_send_ssl(self):
        self.config.email.smtp.username, self.config.email.smtp.password = "user", "pass"
        message = reply("hello")
        with mock.patch("smtplib.SMTP_SSL") as smtp_ssl:
            mailer.send(self.config, message)
        connection = smtp_ssl.return_value
        self.assertEqual(smtp_ssl.call_args[0], ("smtp.example.org", 465))
        connection.starttls.assert_not_called()
        connection.login.assert_called_once_with("user", "pass")
        connection.send_message.assert_called_once_with(message)
        connection.quit.assert_called_once()

    def test_send_starttls_without_login(self):
        self.config.email.smtp.security = "starttls"
        with mock.patch("smtplib.SMTP") as smtp:
            mailer.send(self.config, reply("hello"))
        connection = smtp.return_value
        connection.starttls.assert_called_once()
        connection.login.assert_not_called()

    def test_send_quits_even_on_failure(self):
        import smtplib
        self.config.email.smtp.security = "none"
        with mock.patch("smtplib.SMTP") as smtp:
            connection = smtp.return_value
            connection.send_message.side_effect = smtplib.SMTPRecipientsRefused({})
            connection.quit.side_effect = smtplib.SMTPServerDisconnected()
            with self.assertRaises(smtplib.SMTPRecipientsRefused):
                mailer.send(self.config, reply("hello"))
        connection.starttls.assert_not_called()
        connection.quit.assert_called_once()


class FakeIMAP:
    """Enough of imaplib.IMAP4 for inbox.poll: a folder of raw messages keyed by uid."""

    def __init__(self, messages: dict[str, bytes], select_status: str = "OK"):
        self.messages = messages
        self.select_status = select_status
        self.flags: dict[str, set[str]] = {uid: set() for uid in messages}
        self.copied: list[tuple[str, str]] = []
        self.searches: list[tuple] = []
        self.expunged = 0
        self.logged_out = False

    def select(self, folder: str):
        return self.select_status, [b""]

    def uid(self, command: str, *args: str):
        if command == "SEARCH":
            self.searches.append(args)
            unseen = [uid for uid in self.messages if r"\Seen" not in self.flags[uid]]
            return "OK", [" ".join(unseen).encode()]
        if command == "FETCH":
            return "OK", [(b"header", self.messages[args[0]])]
        if command == "STORE":
            self.flags[args[0]].add(args[2].strip("()"))
            return "OK", []
        if command == "COPY":
            self.copied.append((args[0], args[1]))
            return "OK", []
        raise AssertionError(command)

    def expunge(self):
        self.expunged += 1

    def logout(self):
        self.logged_out = True


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.database = Database(":memory:")
        self.addCleanup(self.database.close)
        for index in (1, 2):
            self.database.add_article(Entry(f"https://e/{index}", f"Article {index}", "", None), "S")
        self.digest_id = self.database.record_digest("abcd1234", "<digest@example.org>", "2026-11-03",
                                                     [DigestPick(Article(1, "https://e/1", "Article 1"), False),
                                                      DigestPick(Article(2, "https://e/2", "Article 2"), True)])

    def test_message_id(self):
        self.assertEqual(inbox.message_id(reply("x", Message_ID=" <a@x> "), "fallback"), "<a@x>")
        self.assertEqual(inbox.message_id(reply("x"), "fallback"), "fallback")

    def test_finds_digest_by_subject_body_or_references(self):
        for index, message in enumerate((reply("+1", subject="Re: list [rec:abcd1234]"),
                                         reply("+1\n> digest-id: rec:abcd1234"),
                                         reply("+1", In_Reply_To="<digest@example.org>"),
                                         reply("+1", References="<older@x> <digest@example.org>"))):
            with self.subTest(message=message["Subject"]):
                result = inbox.handle_message(self.config, self.database, None, message, f"<found{index}@x>")
                self.assertEqual(result, inbox.ReplyResult("ok", ["👍 liked #1: Article 1"]))
        self.assertEqual(inbox.handle_message(self.config, self.database, None, reply("+1"), "<lost@x>"),
                         inbox.ReplyResult("ignored:no-digest"))

    def test_handle_message_statuses(self):
        cases = {
            "ignored:no-digest": reply("+1"),
            "ignored:own": reply("+1 rec:abcd1234", Auto_Submitted="auto-replied"),
            "empty": reply("> quoted only\n> rec:abcd1234"),
            "nothing-understood": reply("+9", subject="Re: [rec:abcd1234]"),  # no item 9
        }
        for expected, message in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(inbox.handle_message(self.config, self.database, None, message, f"<{expected}@x>"),
                                 inbox.ReplyResult(expected))

    def test_handle_message_records_feedback(self):
        message = reply("+1\n2 meh, too long\nmore like the first", subject="Re: [rec:abcd1234]")
        result = inbox.handle_message(self.config, self.database, None, message, "<r1@x>")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, ["👍 liked #1: Article 1", '👎 disliked #2: Article 2 — "too long"',
                                   '💬 general: "more like the first"'])
        self.assertEqual(self.database.source_statistics(), {"S": SourceStatistics(1, 1)})

    def test_poll_without_imap_host(self):
        self.config.email.imap.host = ""
        with mock.patch.object(inbox, "connect", side_effect=AssertionError("connected")):
            self.assertEqual(inbox.poll(self.config, self.database, None), 0)

    def _poll(self, imap: FakeIMAP, sent: list):
        with mock.patch.object(inbox, "connect", return_value=imap), \
             mock.patch.object(mailer, "send", lambda config, message: sent.append(message)):
            return inbox.poll(self.config, self.database, None)

    def test_poll_handles_replies_once_and_leaves_other_mail(self):
        self.config.email.processed_folder = "Processed"
        imap = FakeIMAP({
            "10": bytes(reply("+1 -2", subject="Re: [rec:abcd1234]", Message_ID="<r1@x>")),
            "11": bytes(reply("unrelated mail", Message_ID="<other@x>")),
            "12": bytes(reply("+9", subject="Re: [rec:abcd1234]")),  # no Message-ID, no item 9
        })
        sent: list[EmailMessage] = []
        self.assertEqual(self._poll(imap, sent), 2)
        self.assertTrue(imap.logged_out)
        self.assertEqual(imap.searches[0][-1], '"you@example.com"')
        self.assertEqual(imap.flags["10"], {r"\Seen", r"\Deleted"})
        self.assertEqual(imap.flags["11"], set())  # unrelated mail stays unread
        self.assertEqual(imap.copied, [("10", "Processed"), ("12", "Processed")])
        self.assertEqual(imap.expunged, 2)
        self.assertEqual(len(sent), 2)
        self.assertIn("liked #1", mailer.part_text(sent[0]))
        self.assertIn("couldn't find any ratings", mailer.part_text(sent[1]))
        self.assertTrue(self.database.message_processed("uid-12"))

        # Marked seen elsewhere failed? The processed marker still stops a second count.
        for uid in imap.flags:
            imap.flags[uid].clear()
        self.assertEqual(self._poll(imap, sent), 0)
        self.assertEqual(len(sent), 2)

    def test_poll_survives_a_failing_message_and_acknowledgement(self):
        self.config.email.allowed_senders = ["you@example.com", "also@example.com"]
        imap = FakeIMAP({"1": bytes(reply("+1", subject="Re: [rec:abcd1234]", Message_ID="<r1@x>")),
                         "2": bytes(reply("+2", subject="Re: [rec:abcd1234]", Message_ID="<r2@x>"))})
        original = inbox.handle_message

        def fail_first(config: Config, database: Database, llm: LLM | None, message: EmailMessage,
                       reply_id: str):
            if reply_id == "<r1@x>":
                raise RuntimeError("boom")
            return original(config, database, llm, message, reply_id)

        with mock.patch.object(inbox, "connect", return_value=imap), \
             mock.patch.object(inbox, "handle_message", fail_first), \
             mock.patch.object(mailer, "send", side_effect=OSError("smtp down")), \
             self.assertLogs("indefinite_articles.inbox") as logs:
            self.assertEqual(inbox.poll(self.config, self.database, None), 1)
        self.assertEqual(len(imap.searches), 2)  # one search per allowed sender, uids deduplicated
        self.assertEqual(imap.flags["1"], set())
        self.assertEqual(imap.flags["2"], {r"\Seen"})
        self.assertTrue(any("could not send acknowledgement" in line for line in logs.output))

    def test_poll_unknown_folder(self):
        imap = FakeIMAP({}, select_status="NO")
        with mock.patch.object(inbox, "connect", return_value=imap):
            with self.assertRaisesRegex(RuntimeError, "INBOX"):
                inbox.poll(self.config, self.database, None)
        self.assertTrue(imap.logged_out)

    def test_connect_security_modes(self):
        self.config.email.imap.username, self.config.email.imap.password = "user", "pass"
        with mock.patch("imaplib.IMAP4_SSL") as imap_ssl:
            self.assertIs(inbox.connect(self.config), imap_ssl.return_value)
        imap_ssl.return_value.login.assert_called_once_with("user", "pass")
        for security, starttls_calls in (("starttls", 1), ("none", 0)):
            self.config.email.imap.security = security
            with self.subTest(security=security), mock.patch("imaplib.IMAP4") as imap_plain:
                inbox.connect(self.config)
                self.assertEqual(imap_plain.return_value.starttls.call_count, starttls_calls)


if __name__ == "__main__":
    unittest.main()
