import unittest
from email.message import EmailMessage
from unittest import mock

from indefinite_articles import inbox, mailer, replies
from indefinite_articles.config import Config
from indefinite_articles.database import Database, SourceStatistics
from indefinite_articles.ranker import DigestPick


def make_config() -> Config:
    config = Config()
    config.email.sender, config.email.to = "Bot <bot@example.org>", "you@example.com"
    config.email.allowed_senders = ["you@example.com"]
    config.email.smtp.host = "smtp.example.org"
    config.email.smtp.port = 465
    config.email.imap.host = "imap.example.org"
    return config


def reply(text: str, subject: str = "Re: list", **headers) -> EmailMessage:
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

    def test_new_token(self):
        tokens = {mailer.new_token() for _ in range(20)}
        self.assertEqual(len(tokens), 20)
        self.assertTrue(all(len(token) == 8 and int(token, 16) >= 0 for token in tokens))
        self.assertTrue(all(inbox.TOKEN_RE.fullmatch(mailer.TOKEN_PREFIX + token) for token in tokens))

    def test_instructions_not_empty(self):
        self.assertIn("+1", mailer.instructions())

    def test_build_digest_headers_and_debug(self):
        items = [DigestPick({"title": "T", "source": "S", "url": "https://e/1", "reason": "", "summary": None,
                             "content": "word " * 100}, False)]
        message = mailer.build_digest(self.config, items, "0123abcd", debug=["line one", "line <two>"])
        self.assertTrue(message["Subject"].endswith("[rec:0123abcd]"))
        self.assertEqual(message["X-Indefinite-Articles-Digest"], "0123abcd")
        self.assertEqual(message["To"], "you@example.com")
        text = mailer.part_text(message)
        self.assertIn("-- debug --\nline one\nline <two>", text)
        self.assertNotIn("Why:", text)  # empty reason line left out
        self.assertIn("…", text)  # long content blurb shortened
        self.assertIn("line &lt;two&gt;", mailer.part_text(message, "html"))

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

    def __init__(self, messages: dict[str, bytes], select_status="OK"):
        self.messages = messages
        self.select_status = select_status
        self.flags: dict[str, set[str]] = {uid: set() for uid in messages}
        self.copied: list[tuple[str, str]] = []
        self.searches: list[tuple] = []
        self.expunged = 0
        self.logged_out = False

    def select(self, folder):
        return self.select_status, [b""]

    def uid(self, command, *args):
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
            self.copied.append(args)
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
            self.database.add_article(f"https://e/{index}", f"Article {index}", f"article{index}", "", "S", None)
        self.digest_id = self.database.record_digest("abcd1234", "<digest@example.org>", "2026-11-03",
                                                     [(1, False), (2, True)])

    def test_message_id(self):
        self.assertEqual(inbox.message_id(reply("x", Message_ID=" <a@x> "), "fallback"), "<a@x>")
        self.assertEqual(inbox.message_id(reply("x"), "fallback"), "fallback")

    def test_find_digest_by_subject_body_or_references(self):
        for message in (reply("+1", subject="Re: list [rec:abcd1234]"),
                        reply("+1\n> digest-id: rec:abcd1234"),
                        reply("+1", In_Reply_To="<digest@example.org>"),
                        reply("+1", References="<older@x> <digest@example.org>")):
            with self.subTest(message=message["Subject"]):
                found = inbox.find_digest(self.database, message, replies.body_text(message))
                assert found is not None
                self.assertEqual(found["id"], self.digest_id)
        self.assertIsNone(inbox.find_digest(self.database, reply("+1"), "+1"))

    def test_process_message_statuses(self):
        cases = {
            "ignored:no-digest": reply("+1"),
            "ignored:own": reply("+1 rec:abcd1234", Auto_Submitted="auto-replied"),
            "empty": reply("> quoted only\n> rec:abcd1234"),
            "nothing-understood": reply("+9", subject="Re: [rec:abcd1234]"),  # no item 9
        }
        for expected, message in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(inbox.process_message(self.config, self.database, None, message),
                                 inbox.ReplyResult(expected))

    def test_process_message_records_feedback(self):
        message = reply("+1\n2 meh, too long\nmore like the first", subject="Re: [rec:abcd1234]")
        result = inbox.process_message(self.config, self.database, None, message)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, ["👍 liked #1: Article 1", '👎 disliked #2: Article 2 — "too long"',
                                   '💬 general: "more like the first"'])
        self.database.connection.commit()
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

        def fail_first(config, database, llm, message, reply_id):
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
