"""Composing and sending emails over SMTP."""

import logging
import smtplib
import ssl
from email.message import EmailMessage

from .config import Config

log = logging.getLogger(__name__)


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
