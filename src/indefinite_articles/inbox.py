"""Reading replies from an IMAP mailbox and recording feedback."""

import email
import email.policy
import imaplib
import logging
import re
import ssl
from datetime import date, timedelta
from email.utils import getaddresses

from . import mailer, replies
from .llm import LLM

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(re.escape(mailer.TOKEN_PREFIX) + r"([0-9a-f]{8})")


def connect(cfg) -> imaplib.IMAP4:
    s = cfg.email.imap
    ctx = ssl.create_default_context()
    if s.security == "ssl":
        conn = imaplib.IMAP4_SSL(s.host, s.port, ssl_context=ctx, timeout=60)
    else:
        conn = imaplib.IMAP4(s.host, s.port, timeout=60)
        if s.security == "starttls":
            conn.starttls(ssl_context=ctx)
    conn.login(s.username, s.password)
    return conn


def _sender_allowed(cfg, msg) -> bool:
    addrs = [a.lower() for _, a in getaddresses(msg.get_all("From", []))]
    return any(a in cfg.email.allowed_senders for a in addrs)


def find_digest(db, msg, text: str):
    m = TOKEN_RE.search(msg.get("Subject", "") or "") or TOKEN_RE.search(text)
    refs = (msg.get("In-Reply-To", "") or "").split() + (msg.get("References", "") or "").split()
    return db.find_digest(m.group(1) if m else None, refs)


def process_message(cfg, db, llm: LLM | None, msg) -> tuple[str, list[str]]:
    """Stage feedback from one reply in the open transaction (not committed).

    Use handle_message(), which commits it together with the processed-message marker.
    Returns (status, summary lines).
    """
    if not _sender_allowed(cfg, msg):
        return "ignored:sender", []
    # Never treat our own digests/acknowledgements as replies (e.g. when to == from).
    if msg.get("X-Indefinite-Articles-Digest") or msg.get("X-Indefinite-Articles-Ack") or (
        (msg.get("Auto-Submitted") or "no").lower() != "no"
    ):
        return "ignored:own", []
    full = replies.body_text(msg)
    digest = find_digest(db, msg, full)
    if digest is None:
        return "ignored:no-digest", []
    items = db.digest_items(digest["id"])
    text = replies.strip_quoted(full)
    if not text:
        return "empty", []
    ratings, general = replies.understand(llm, text, items)

    summary = []
    for n in sorted(ratings):
        rating, comment = ratings[n]
        a = items[n - 1]
        db.add_feedback(a["id"], rating, comment)
        label = {1: "👍 liked", -1: "👎 disliked", 0: "💬 noted"}[rating]
        summary.append(f"{label} #{n}: {a['title']}" + (f' — "{comment}"' if comment else ""))
    for c in general:
        db.add_general_comment(digest["id"], c)
        summary.append(f'💬 general: "{c}"')
    return ("ok" if summary else "nothing-understood"), summary


def message_id(msg, fallback: str) -> str:
    return (msg.get("Message-ID") or fallback).strip()


def handle_message(cfg, db, llm: LLM | None, msg, mid: str) -> tuple[str, list[str]]:
    """Process a reply exactly once: its feedback and the processed marker are committed
    in one transaction, so a crash can never record one without the other."""
    if db.message_processed(mid):
        return "already-processed", []
    try:
        status, summary = process_message(cfg, db, llm, msg)
        db.mark_message(mid, status, commit=False)
        db.conn.commit()
    except BaseException:
        db.conn.rollback()
        raise
    return status, summary


def poll(cfg, db, llm: LLM | None) -> int:
    """Fetch unread replies, record feedback. Returns number of replies used."""
    if not cfg.email.imap.host:
        return 0
    used = 0
    conn = connect(cfg)
    try:
        typ, _ = conn.select(cfg.email.imap_folder)
        if typ != "OK":
            raise RuntimeError(f"cannot select IMAP folder {cfg.email.imap_folder!r}")
        since = (date.today() - timedelta(days=14)).strftime("%d-%b-%Y")
        uids: list[bytes] = []
        for sender in cfg.email.allowed_senders:
            typ, data = conn.uid("SEARCH", None, "UNSEEN", "SINCE", since, "FROM", f'"{sender}"')
            if typ == "OK" and data and data[0]:
                uids += data[0].split()
        for uid in dict.fromkeys(uids):
            typ, data = conn.uid("FETCH", uid, "(BODY.PEEK[])")
            if typ != "OK" or not data or not isinstance(data[0], tuple):
                continue
            msg = email.message_from_bytes(data[0][1], policy=email.policy.default)
            mid = message_id(msg, f"uid-{uid.decode()}")
            try:
                status, summary = handle_message(cfg, db, llm, msg, mid)
            except Exception:
                log.exception("failed to process message %s", mid)
                continue
            if status == "already-processed":
                continue
            log.info("reply %s: %s (%d items)", mid, status, len(summary))
            if status.startswith("ignored"):
                continue  # leave unrelated mail untouched
            used += 1
            conn.uid("STORE", uid, "+FLAGS", r"(\Seen)")
            if cfg.email.processed_folder:
                typ, _ = conn.uid("COPY", uid, cfg.email.processed_folder)
                if typ == "OK":
                    conn.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
                    conn.expunge()
            if cfg.email.send_acknowledgement:
                lines = summary or ["Sorry, I couldn't find any ratings in your reply. "
                                    "Try something like: +1 +3 -2"]
                try:
                    mailer.send(cfg, mailer.build_ack(cfg, msg, lines))
                except Exception as e:
                    log.warning("could not send acknowledgement: %s", e)
    finally:
        try:
            conn.logout()
        except Exception:
            pass
    return used
