"""Composing and sending emails over SMTP."""

import functools
import html
import logging
import re
import secrets
import smtplib
import ssl
import string
from collections.abc import Callable
from datetime import datetime
from importlib import resources
from email.message import EmailMessage

from .config import Config
from .database import ArticleRow
from .ranker import DigestPick

log = logging.getLogger(__name__)

TOKEN_PREFIX = "rec:"

TEMPLATES = resources.files(__package__) / "templates"
SECTION_MARKER = "@@ "

def new_token() -> str:
    return secrets.token_hex(4)


def _blurb(article: ArticleRow, limit: int = 280) -> str:
    text = " ".join((article["summary"] or article["content"] or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


@functools.cache
def _sections(name: str) -> dict[str, list[str]]:
    """The `@@ name` sections of a template file, as lists of lines.

    Text before the first section is documentation; trailing blank lines are dropped.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in (TEMPLATES / name).read_text().splitlines():
        if line.startswith(SECTION_MARKER):
            current = sections.setdefault(line.removeprefix(SECTION_MARKER).strip(), [])
        elif current is not None:
            current.append(line)
    for lines in sections.values():
        while lines and not lines[-1].strip():
            lines.pop()
    return sections


def _render(lines: list[str], values: dict[str, str]) -> str:
    """Fill in `$name` placeholders, leaving out lines whose placeholders are all empty."""
    out = []
    for line in lines:
        template = string.Template(line)
        names = template.get_identifiers()
        if names and not any(values[name] for name in names):
            continue
        out.append(template.substitute(values))
    return "\n".join(out)


def _instructions() -> str:
    """How to reply with feedback, as shown at the end of every digest."""
    return (TEMPLATES / "instructions.txt").read_text().strip()


def _digest_part(template_name: str, items: list[DigestPick], token: str, note: str, debug: list[str] | None,
                 escape: Callable[[str], str], item_separator: str) -> str:
    sections = _sections(template_name)
    rendered = []
    for position, pick in enumerate(items, start=1):
        article = pick.article
        rendered.append(_render(sections["item"], {
            "position": str(position),
            "title": escape(article["title"]),
            "source": escape(article["source"]),
            "exploratory": _render(sections["exploratory"], {}) if pick.exploratory else "",
            "url": escape(article["url"]),
            "reason": escape(article["reason"] or ""),
            "blurb": escape(_blurb(article)),
        }))
    return _render(sections["page"], {
        "count": str(len(items)),
        "note": escape(note),
        "items": item_separator.join(rendered),
        "instructions": escape(_instructions()),
        "digest_id": escape(TOKEN_PREFIX + token),
        "debug": _render(sections["debug"], {"lines": escape("\n".join(debug))}) if debug else "",
    })


def build_digest(config: Config, items: list[DigestPick], token: str, note: str = "",
                 debug: list[str] | None = None) -> EmailMessage:
    """items: the `ranker.DigestPick`s to send. debug: lines appended after the footer.

    The layout comes from templates/digest.txt and templates/digest.html.
    """
    today = datetime.now().strftime("%A %d %B %Y")
    message = config.email.new_message("digest")
    message["Subject"] = f"Your reading list for {today} [{TOKEN_PREFIX}{token}]"
    message["X-Indefinite-Articles-Digest"] = token

    text = _digest_part("digest.txt", items, token, note, debug, str, "\n\n")
    message.set_content(re.sub(r"\n{3,}", "\n\n", text).strip())
    message.add_alternative(_digest_part("digest.html", items, token, note, debug, html.escape, "\n"),
                            subtype="html")
    return message


def build_acknowledgement(config: Config, original: EmailMessage, summary_lines: list[str]) -> EmailMessage:
    message = config.email.new_message("ack")
    subject = original.get("Subject", "") or "your feedback"
    message["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    message["Auto-Submitted"] = "auto-replied"
    message["X-Indefinite-Articles-Ack"] = "1"
    if original.get("Message-ID"):
        message["In-Reply-To"] = original["Message-ID"]
        message["References"] = (original.get("References", "") + " " + original["Message-ID"]).strip()
    message.set_content("Thanks — here's what I understood:\n\n" + "\n".join(summary_lines) +
                    "\n\nI'll use this for tomorrow's recommendations.")
    return message


def part_text(message: EmailMessage, subtype: str = "plain") -> str:
    """The text/<subtype> body of a message we built, which always has one."""
    part = message.get_body(preferencelist=(subtype,))
    if part is None:
        raise ValueError(f"message has no text/{subtype} part")
    return part.get_content()


def send(config: Config, message: EmailMessage) -> None:
    server = config.email.smtp
    context = ssl.create_default_context()
    if server.security == "ssl":
        connection = smtplib.SMTP_SSL(server.host, server.port, context=context, timeout=60)
    else:
        connection = smtplib.SMTP(server.host, server.port, timeout=60)
    try:
        connection.ehlo()
        if server.security == "starttls":
            connection.starttls(context=context)
            connection.ehlo()
        if server.username:
            connection.login(server.username, server.password)
        connection.send_message(message)
    finally:
        try:
            connection.quit()
        except smtplib.SMTPException:
            pass
    log.info("sent email %r to %s", message["Subject"], message["To"])
