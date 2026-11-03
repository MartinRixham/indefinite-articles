"""Composing and sending emails over SMTP."""

import html
import logging
import secrets
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

log = logging.getLogger(__name__)

TOKEN_PREFIX = "rec:"

INSTRUCTIONS = """\
Reply to this email to teach me what you like. Any of these work:
  +1 +4 -2          (like 1 and 4, dislike 2)
  3 like - more deep dives on compilers please
  5 no, I don't care about crypto
...or just write in plain words, e.g. "loved the Rust one, less politics".
Items you don't mention are ignored."""


def new_token() -> str:
    return secrets.token_hex(4)


def _domain(addr: str) -> str:
    return addr.rsplit("@", 1)[-1].strip(">") if "@" in addr else "localhost"


def _blurb(a, limit=280) -> str:
    text = " ".join((a["summary"] or a["content"] or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_digest(cfg, items, token: str, note: str = "",
                 debug: list[str] | None = None) -> EmailMessage:
    """items: list of (article_row, exploratory). debug: lines appended after the footer."""
    today = datetime.now().strftime("%A %d %B %Y")
    msg = EmailMessage()
    msg["Subject"] = f"Your reading list for {today} [{TOKEN_PREFIX}{token}]"
    msg["From"] = cfg.email.sender
    msg["To"] = cfg.email.to
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid("digest", domain=_domain(cfg.email.sender))
    msg["X-Indefinite-Articles-Digest"] = token

    lines = [f"Today's {len(items)} recommendations", ""]
    if note:
        lines += [note, ""]
    for pos, (a, exploratory) in enumerate(items, start=1):
        lines.append(f"{pos}. {a['title']}")
        lines.append(f"   {a['source']}{'  · something different' if exploratory else ''}")
        lines.append(f"   {a['url']}")
        if a["reason"]:
            lines.append(f"   Why: {a['reason']}")
        if _blurb(a):
            lines.append(f"   {_blurb(a)}")
        lines.append("")
    lines += ["--", INSTRUCTIONS, "", f"digest-id: {TOKEN_PREFIX}{token}"]
    if debug:
        lines += ["", "-- debug --", *debug]
    msg.set_content("\n".join(lines))

    e = html.escape
    parts = [
        '<html><body style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
        'max-width:680px;margin:auto;color:#222;line-height:1.45">',
        f"<h2 style='font-weight:600'>Today's {len(items)} recommendations</h2>",
    ]
    if note:
        parts.append(f"<p style='color:#a60'>{e(note)}</p>")
    for pos, (a, exploratory) in enumerate(items, start=1):
        tag = (" <span style='font-size:12px;background:#eef;padding:1px 6px;border-radius:8px'>"
               "something different</span>") if exploratory else ""
        parts.append(
            "<div style='margin:0 0 22px'>"
            f"<div style='font-size:17px'><b>{pos}.</b> "
            f"<a href='{e(a['url'], quote=True)}' style='color:#1a56c8;text-decoration:none'>"
            f"{e(a['title'])}</a></div>"
            f"<div style='font-size:13px;color:#666'>{e(a['source'])}{tag}</div>"
            + (f"<div style='font-size:13px;color:#2a7a2a'>Why: {e(a['reason'])}</div>" if a["reason"] else "")
            + (f"<div style='font-size:14px;color:#444;margin-top:4px'>{e(_blurb(a))}</div>" if _blurb(a) else "")
            + "</div>"
        )
    parts.append(
        "<hr style='border:none;border-top:1px solid #ddd'>"
        f"<pre style='font-size:13px;color:#555;white-space:pre-wrap'>{e(INSTRUCTIONS)}</pre>"
        f"<p style='font-size:11px;color:#aaa'>digest-id: {TOKEN_PREFIX}{token}</p>"
    )
    if debug:
        parts.append(
            "<details><summary style='font-size:11px;color:#aaa'>debug</summary>"
            f"<pre style='font-size:11px;color:#888;white-space:pre-wrap'>{e(chr(10).join(debug))}</pre>"
            "</details>"
        )
    parts.append("</body></html>")
    msg.add_alternative("".join(parts), subtype="html")
    return msg


def build_ack(cfg, original, summary_lines: list[str]) -> EmailMessage:
    msg = EmailMessage()
    subj = original.get("Subject", "") or "your feedback"
    msg["Subject"] = subj if subj.lower().startswith("re:") else f"Re: {subj}"
    msg["From"] = cfg.email.sender
    msg["To"] = cfg.email.to
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid("ack", domain=_domain(cfg.email.sender))
    msg["Auto-Submitted"] = "auto-replied"
    msg["X-Indefinite-Articles-Ack"] = "1"
    if original.get("Message-ID"):
        msg["In-Reply-To"] = original["Message-ID"]
        msg["References"] = (original.get("References", "") + " " + original["Message-ID"]).strip()
    msg.set_content("Thanks — here's what I understood:\n\n" + "\n".join(summary_lines) +
                    "\n\nI'll use this for tomorrow's recommendations.")
    return msg


def send(cfg, msg: EmailMessage):
    s = cfg.email.smtp
    ctx = ssl.create_default_context()
    if s.security == "ssl":
        conn = smtplib.SMTP_SSL(s.host, s.port, context=ctx, timeout=60)
    else:
        conn = smtplib.SMTP(s.host, s.port, timeout=60)
    try:
        conn.ehlo()
        if s.security == "starttls":
            conn.starttls(context=ctx)
            conn.ehlo()
        if s.username:
            conn.login(s.username, s.password)
        conn.send_message(msg)
    finally:
        try:
            conn.quit()
        except smtplib.SMTPException:
            pass
    log.info("sent email %r to %s", msg["Subject"], msg["To"])
