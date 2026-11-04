"""The daily digest: the chosen articles, rendered into an email from the templates."""

import functools
import html
import re
import secrets
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from email.message import EmailMessage
from importlib import resources

from .config import Config
from .ranker import DigestPick

TOKEN_PREFIX = "rec:"

TEMPLATES = resources.files(__package__) / "templates"
SECTION_MARKER = "@@ "


def _new_token() -> str:
    return secrets.token_hex(4)


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


@dataclass
class Digest:
    """One day's digest: the picks and everything else the email shows.

    items: the `ranker.DigestPick`s to send. note: a remark shown above them.
    debug: lines appended after the footer. token: identifies the digest in replies.
    The layout comes from templates/digest.txt and templates/digest.html.
    """
    items: list[DigestPick]
    note: str = ""
    debug: list[str] | None = None
    token: str = field(default_factory=_new_token)
    day: date = field(default_factory=date.today)

    @property
    def digest_id(self) -> str:
        """The marker that ties a reply to this digest, as shown in the subject and footer."""
        return TOKEN_PREFIX + self.token

    @property
    def subject(self) -> str:
        return f"Your reading list for {self.day.strftime('%A %d %B %Y')} [{self.digest_id}]"

    def text_body(self) -> str:
        text = self._render("digest.txt", str, "\n\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def html_body(self) -> str:
        return self._render("digest.html", html.escape, "\n")

    def message(self, config: Config) -> EmailMessage:
        message = config.email.new_message("digest")
        message["Subject"] = self.subject
        message["X-Indefinite-Articles-Digest"] = self.token
        message.set_content(self.text_body())
        message.add_alternative(self.html_body(), subtype="html")
        return message

    def _render(self, template_name: str, escape: Callable[[str], str], item_separator: str) -> str:
        sections = _sections(template_name)
        rendered = []
        for position, pick in enumerate(self.items, start=1):
            article = pick.article
            rendered.append(_render(sections["item"], {
                "position": str(position),
                "title": escape(article.title),
                "source": escape(article.source),
                "exploratory": _render(sections["exploratory"], {}) if pick.exploratory else "",
                "url": escape(article.url),
                "reason": escape(article.reason or ""),
                "blurb": escape(article.blurb()),
            }))
        debug = self.debug
        return _render(sections["page"], {
            "count": str(len(self.items)),
            "note": escape(self.note),
            "items": item_separator.join(rendered),
            "instructions": escape(_instructions()),
            "digest_id": escape(self.digest_id),
            "debug": _render(sections["debug"], {"lines": escape("\n".join(debug))}) if debug else "",
        })
