"""Reading replies from an IMAP mailbox and recording feedback."""

import email
import email.policy
import imaplib
import logging
import re
import ssl
from datetime import date, timedelta

from . import mailer, replies
from .llm import LLM

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(re.escape(mailer.TOKEN_PREFIX) + r"([0-9a-f]{8})")


def connect(config) -> imaplib.IMAP4:
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


def find_digest(database, message, text: str):
    match = TOKEN_RE.search(message.get("Subject", "") or "") or TOKEN_RE.search(text)
    refs = (message.get("In-Reply-To", "") or "").split() + (message.get("References", "") or "").split()
    return database.find_digest(match.group(1) if match else None, refs)


def process_message(config, database, llm: LLM | None, message) -> tuple[str, list[str]]:
    """Stage feedback from one reply in the open transaction (not committed).

    Use handle_message(), which commits it together with the processed-message marker.
    Returns (status, summary lines).
    """
    if not config.email.is_allowed_sender(message):
        return "ignored:sender", []
    # Never treat our own digests/acknowledgements as replies (e.g. when to == from).
    if message.get("X-Indefinite-Articles-Digest") or message.get("X-Indefinite-Articles-Ack") or (
        (message.get("Auto-Submitted") or "no").lower() != "no"
    ):
        return "ignored:own", []
    full = replies.body_text(message)
    digest = find_digest(database, message, full)
    if digest is None:
        return "ignored:no-digest", []
    items = database.digest_items(digest["id"])
    text = replies.strip_quoted(full)
    if not text:
        return "empty", []
    ratings, general = replies.understand(llm, text, items)

    summary = []
    for item_number in sorted(ratings):
        rating, comment = ratings[item_number]
        article = items[item_number - 1]
        database.add_feedback(article["id"], rating, comment)
        label = {1: "👍 liked", -1: "👎 disliked", 0: "💬 noted"}[rating]
        summary.append(f"{label} #{item_number}: {article['title']}" + (f' — "{comment}"' if comment else ""))
    for comment in general:
        database.add_general_comment(digest["id"], comment)
        summary.append(f'💬 general: "{comment}"')
    return ("ok" if summary else "nothing-understood"), summary


def message_id(message, fallback: str) -> str:
    return (message.get("Message-ID") or fallback).strip()


def handle_message(config, database, llm: LLM | None, message, reply_id: str) -> tuple[str, list[str]]:
    """Process a reply exactly once: its feedback and the processed marker are committed
    in one transaction, so a crash can never record one without the other."""
    if database.message_processed(reply_id):
        return "already-processed", []
    try:
        status, summary = process_message(config, database, llm, message)
        database.mark_message(reply_id, status, commit=False)
        database.connection.commit()
    except BaseException:
        database.connection.rollback()
        raise
    return status, summary


def poll(config, database, llm: LLM | None) -> int:
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
                status, summary = handle_message(config, database, llm, message, reply_id)
            except Exception:
                log.exception("failed to process message %s", reply_id)
                continue
            if status == "already-processed":
                continue
            log.info("reply %s: %s (%d items)", reply_id, status, len(summary))
            if status.startswith("ignored"):
                continue  # leave unrelated mail untouched
            used += 1
            connection.uid("STORE", uid, "+FLAGS", r"(\Seen)")
            if config.email.processed_folder:
                imap_status, _ = connection.uid("COPY", uid, config.email.processed_folder)
                if imap_status == "OK":
                    connection.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
                    connection.expunge()
            if config.email.send_acknowledgement:
                lines = summary or ["Sorry, I couldn't find any ratings in your reply. "
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
