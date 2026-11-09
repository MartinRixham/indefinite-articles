"""Reading replies from an IMAP mailbox and recording feedback."""

import email
import email.policy
import imaplib
import logging
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from email.message import EmailMessage
from email.utils import parsedate_to_datetime

from ..config import Config
from ..language_model.llm import LLM, LLMUnavailable
from ..language_model.sent_request import SentRequest
from ..storage.database import Database
from . import mailer, replies
from .digest import TOKEN_PREFIX

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(re.escape(TOKEN_PREFIX) + r"([0-9a-f]{8})")

# How long a reply waits for an unavailable LLM before only its structured ratings are
# used. Well inside the 14 days poll() searches, so a reply can't drop out of the search.
LLM_WAIT_DAYS = 3


@dataclass
class ReplyResult:
    """What became of one reply ("ok", "ignored:sender", ...) and a line per item understood."""
    status: str
    summary: list[str] = field(default_factory=list)

    @property
    def ignored(self) -> bool:
        """Not a reply to a digest: leave the message untouched."""
        return self.status.startswith("ignored")


def connect(config: Config) -> imaplib.IMAP4:
    server = config.email.imap
    context = ssl.create_default_context()
    if server.security == "ssl":
        connection = imaplib.IMAP4_SSL(server.host, server.port, ssl_context=context, timeout=60)
    else:
        connection = imaplib.IMAP4(server.host, server.port, timeout=60)
        if server.security == "starttls":
            connection.starttls(ssl_context=context)
    connection.login(server.username, server.password)
    return connection


def _find_digest(database: Database, message: EmailMessage, text: str):
    match = TOKEN_RE.search(message.get("Subject", "") or "") or TOKEN_RE.search(text)
    refs = (message.get("In-Reply-To", "") or "").split() + (message.get("References", "") or "").split()
    return database.find_digest(match.group(1) if match else None, refs)


def _process_message(config: Config, database: Database, llm: LLM | None, message: EmailMessage) -> ReplyResult:
    """Stage feedback from one reply in the open transaction (not committed).

    Use handle_message(), which commits it together with the processed-message marker.
    """
    if not config.email.is_allowed_sender(message):
        return ReplyResult("ignored:sender")
    # Never treat our own digests/acknowledgements as replies (e.g. when to == from).
    if message.get("X-Indefinite-Articles-Digest") or message.get("X-Indefinite-Articles-Ack") or (
        (message.get("Auto-Submitted") or "no").lower() != "no"
    ):
        return ReplyResult("ignored:own")
    full = replies.body_text(message)
    digest = _find_digest(database, message, full)
    if digest is None:
        return ReplyResult("ignored:no-digest")
    items = database.digest_items(digest["id"])
    text = replies.strip_quoted(full)
    if not text:
        return ReplyResult("empty")
    interpretation = replies.understand(llm, text, items)

    summary = []
    for item_number in sorted(interpretation.ratings):
        rating = interpretation.ratings[item_number]
        article = items[item_number - 1]
        database.add_feedback(article.id, rating.rating, rating.comment)
        summary.append(f"{rating.label} #{item_number}: {article.title}"
                       + (f' — "{rating.comment}"' if rating.comment else ""))
    for comment in interpretation.general:
        database.add_general_comment(digest["id"], comment)
        summary.append(f'💬 general: "{comment}"')
    return ReplyResult("ok" if summary else "nothing-understood", summary)


def message_id(message: EmailMessage, fallback: str) -> str:
    return (message.get("Message-ID") or fallback).strip()


def handle_message(
    config: Config,
    database: Database,
    llm: LLM | None,
    message: EmailMessage,
    reply_id: str,
) -> ReplyResult:
    """Process a reply exactly once: its feedback and the processed marker are committed
    in one transaction, so a crash can never record one without the other."""
    if database.message_processed(reply_id):
        return ReplyResult("already-processed")
    try:
        result = _process_message(config, database, llm, message)
        database.mark_message(reply_id, result.status, commit=False)
        database.connection.commit()
    except BaseException:
        database.connection.rollback()
        raise
    return result


def _waited_too_long(message: EmailMessage) -> bool:
    """Whether the reply was sent more than LLM_WAIT_DAYS ago (or has no usable Date)."""
    try:
        sent = parsedate_to_datetime(message.get("Date", "") or "").timestamp()
    except (TypeError, ValueError):
        return True
    return time.time() - sent > LLM_WAIT_DAYS * 86400


class _UnavailableLLM(LLM):
    """Stands in for the LLM once it has been unavailable during a poll: replies that
    need it then wait for the next poll without another slow attempt, and replies
    that don't are still handled."""

    def _send(self, path: str, body: dict, kind: str, native: bool = False) -> SentRequest:
        raise LLMUnavailable("the LLM was unavailable earlier in this poll")


def _handle_when_possible(config: Config, database: Database, llm: LLM | None, message: EmailMessage,
                          reply_id: str) -> ReplyResult:
    """handle_message(). If the reply needs the LLM and it is unavailable, this raises
    LLMUnavailable and the reply is left unread and unprocessed for the next poll,
    until it has waited LLM_WAIT_DAYS; then it is handled without the LLM."""
    try:
        return handle_message(config, database, llm, message, reply_id)
    except LLMUnavailable as error:
        if not _waited_too_long(message):
            log.warning("reply %s left until the LLM is available: %s", reply_id, error)
            raise
        log.warning("reply %s waited %d days for the LLM (%s); using only its structured ratings",
                    reply_id, LLM_WAIT_DAYS, error)
        return handle_message(config, database, None, message, reply_id)


def poll(config: Config, database: Database, llm: LLM | None) -> int:
    """Fetch unread replies, record feedback. Returns number of replies used."""
    if not config.email.imap.host:
        return 0
    used = 0
    connection = connect(config)
    try:
        imap_status, _ = connection.select(config.email.imap_folder)
        if imap_status != "OK":
            raise RuntimeError(f"cannot select IMAP folder {config.email.imap_folder!r}")
        since = (date.today() - timedelta(days=14)).strftime("%d-%b-%Y")
        uids: list[str] = []
        for sender in config.email.allowed_senders:
            imap_status, data = connection.uid("SEARCH", "UNSEEN", "SINCE", since, "FROM", f'"{sender}"')
            if imap_status == "OK" and data and data[0]:
                uids += data[0].decode().split()
        for uid in dict.fromkeys(uids):
            imap_status, data = connection.uid("FETCH", uid, "(BODY.PEEK[])")
            if imap_status != "OK" or not data or not isinstance(data[0], tuple):
                continue
            message = email.message_from_bytes(data[0][1], policy=email.policy.default)
            reply_id = message_id(message, f"uid-{uid}")
            try:
                result = _handle_when_possible(config, database, llm, message, reply_id)
            except LLMUnavailable:
                if llm is not None:
                    llm = _UnavailableLLM(llm.config)
                continue
            except Exception:
                log.exception("failed to process message %s", reply_id)
                continue
            if result.status == "already-processed":
                continue
            log.info("reply %s: %s (%d items)", reply_id, result.status, len(result.summary))
            if result.ignored:
                continue
            used += 1
            connection.uid("STORE", uid, "+FLAGS", r"(\Seen)")
            if config.email.processed_folder:
                imap_status, _ = connection.uid("COPY", uid, config.email.processed_folder)
                if imap_status == "OK":
                    connection.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
                    connection.expunge()
            if config.email.send_acknowledgement:
                lines = result.summary or ["Sorry, I couldn't find any ratings in your reply. "
                                    "Try something like: +1 +3 -2"]
                try:
                    mailer.send(config, mailer.build_acknowledgement(config, message, lines))
                except Exception as error:
                    log.warning("could not send acknowledgement: %s", error)
    finally:
        try:
            connection.logout()
        except Exception:
            pass
    return used
