"""Configuration loading (TOML)."""

import os
import re
import tomllib
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formatdate, getaddresses, make_msgid
from pathlib import Path

from .fetching.feed import Feed

DEFAULT_CONFIG_PATH = "/etc/indefinite-articles/indefinite-articles.toml"
DEFAULTS_FILE_NAME = "defaults.toml"  # optional, next to the config file; the config file overrides it


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class TimeOfDay:
    hour: int
    minute: int

    def has_passed(self, now: datetime) -> bool:
        """True from this time of day until midnight."""
        return (now.hour, now.minute) >= (self.hour, self.minute)

    def next_after(self, now: datetime) -> datetime:
        """The next occurrence of this time of day, strictly after `now`."""
        at = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
        return at if at > now else at + timedelta(days=1)


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:14b"
    api_key: str = ""
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_prefix: str = "Instruct: Identify the topic of the given news article\nQuery: "
    embedding_batch: int = 64
    max_prompt_chars: int = 24000
    temperature: float = 0.2
    require_gpu: bool = True  # refuse to run if Ollama loads the model on CPU only
    pull_models: bool = True  # have Ollama download missing models when the service starts

    @classmethod
    def from_section(cls, section: dict, base: Path) -> "LLMConfig":
        llm = cls()
        _copy_settings(llm, section, ("base_url", "model", "api_key", "max_prompt_chars", "temperature",
                                      "embedding_model", "embedding_prefix", "embedding_batch", "require_gpu",
                                      "pull_models"))
        if section.get("api_key_file"):
            llm.api_key = _secret(section, "api_key", base)
        llm.base_url = llm.base_url.rstrip("/")
        return llm


@dataclass
class ServerConfig:
    host: str = ""
    port: int = 0
    security: str = "ssl"  # ssl | starttls | none
    username: str = ""
    password: str = ""

    @classmethod
    def from_section(cls, section: dict, default_port: int, base: Path) -> "ServerConfig":
        security = section.get("security", "ssl" if default_port in (465, 993) else "starttls")
        if security not in ("ssl", "starttls", "none"):
            raise ConfigError(f"security must be ssl, starttls or none (got {security!r})")
        return cls(
            host=section.get("host", ""),
            port=int(section.get("port", default_port)),
            security=security,
            username=section.get("username", ""),
            password=_secret(section, "password", base),
        )


@dataclass
class EmailConfig:
    to: str = ""
    sender: str = ""
    smtp: ServerConfig = field(default_factory=ServerConfig)
    imap: ServerConfig = field(default_factory=ServerConfig)
    imap_folder: str = "INBOX"
    processed_folder: str = ""
    allowed_senders: list[str] = field(default_factory=list)
    send_acknowledgement: bool = True
    debug_info: bool = True  # append pipeline/scoring diagnostics to each digest

    @classmethod
    def from_section(cls, section: dict, base: Path) -> "EmailConfig":
        imap = section.get("imap", {})
        to = section.get("to", "")
        return cls(
            to=to,
            sender=section.get("from", ""),
            smtp=ServerConfig.from_section(section.get("smtp", {}), 587, base),
            imap=ServerConfig.from_section(imap, 993, base),
            imap_folder=imap.get("folder", "INBOX"),
            processed_folder=imap.get("processed_folder", ""),
            allowed_senders=[sender.lower() for sender in section.get("allowed_senders", [])]
                            or ([to.lower()] if to else []),
            send_acknowledgement=bool(section.get("send_acknowledgement", True)),
            debug_info=bool(section.get("debug_info", True)),
        )

    def new_message(self, kind: str) -> EmailMessage:
        """An empty message from `sender` to `to`, with Date and Message-ID set."""
        domain = self.sender.rsplit("@", 1)[-1].strip(">") if "@" in self.sender else "localhost"
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = self.to
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid(kind, domain=domain)
        return message

    def is_allowed_sender(self, message: EmailMessage) -> bool:
        addresses = [address.lower() for _, address in getaddresses(message.get_all("From", []))]
        return any(address in self.allowed_senders for address in addresses)


@dataclass
class Config:
    database: str = "/var/lib/indefinite-articles/indefinite-articles.db"
    digest_time: str = "07:00"
    articles_per_digest: int = 10
    podcasts_per_digest: int = 3   # podcast episodes, listed after the articles
    exploration: int = 2
    podcast_exploration: int = 1
    max_per_source: int = 3
    max_article_age_days: int = 3
    max_candidates: int = 150      # articles per day that get full-text + LLM scoring
    max_podcast_candidates: int = 30  # podcast episodes per day that get LLM scoring
    apple_podcasts_links: bool = True  # look up shortlisted episodes on Apple Podcasts
    rescore_top: int = 30          # after a profile change, rescore only this many top-ranked articles
    feeds_dir: str = "feeds.d"     # relative to the config file
    exclude_categories: list[str] = field(default_factory=list)
    fetch_concurrency: int = 16
    fetch_interval_minutes: int = 60
    poll_interval_minutes: int = 5
    fetch_full_articles: bool = True
    article_chars: int = 4000
    fetch_workers: int = 4
    allow_private_addresses: bool = False
    interests: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    feeds: list[Feed] = field(default_factory=list)

    @property
    def digest_hour_minute(self) -> TimeOfDay:
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", self.digest_time.strip())
        if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
            raise ConfigError(f"digest_time must be HH:MM, got {self.digest_time!r}")
        return TimeOfDay(int(match.group(1)), int(match.group(2)))

    def next_digest_time(self) -> float:
        """When the next digest is due (epoch seconds), always in the future.

        Long-running work (LLM requests, the hourly job) must be finished by then.
        """
        return self.digest_hour_minute.next_after(datetime.now()).timestamp()

    def problems(self) -> list[str]:
        """Human-readable problems that would stop the daemon working."""
        problems = []
        if not self.feeds:
            problems.append("no feeds configured (add files to feeds.d/ or [[feeds]] entries)")
        if not self.email.to or not self.email.sender:
            problems.append("email.to and email.from must be set")
        if not self.email.smtp.host:
            problems.append("email.smtp.host is not set")
        if not self.email.imap.host:
            problems.append("email.imap.host is not set (replies will not be read)")
        return problems


def _copy_settings(target: object, section: dict, keys: Iterable[str]) -> None:
    """Copy the given keys from a TOML section, converted to the type of the default."""
    for key in keys:
        if key in section:
            setattr(target, key, type(getattr(target, key))(section[key]))


def _secret(section: dict, key: str, base: Path) -> str:
    """Read `key` directly, or from `key_file` (relative paths resolve next to the config)."""
    if section.get(f"{key}_file"):
        secret_path = Path(section[f"{key}_file"])
        if not secret_path.is_absolute():
            secret_path = base / secret_path
        return secret_path.read_text().strip()
    return str(section.get(key, ""))


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from None


def _merge(defaults: dict, overrides: dict) -> dict:
    """`overrides` on top of `defaults`; tables are merged key by key, anything else is replaced."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load(path: str | os.PathLike | None = None) -> Config:
    """Load the config file, on top of defaults.toml in the same directory if there is one.

    defaults.toml holds the packaged default settings. Users who leave it unedited get the
    updated values with each upgrade; any setting can be overridden in the config file.
    """
    path = Path(path or os.environ.get("INDEFINITE_ARTICLES_CONFIG", DEFAULT_CONFIG_PATH))
    raw = _read_toml(path)
    base = path.parent
    defaults_path = base / DEFAULTS_FILE_NAME
    if defaults_path.is_file():
        raw = _merge(_read_toml(defaults_path), raw)

    general = raw.get("general", {})
    config = Config()
    _copy_settings(config, general, (
        "database", "digest_time", "articles_per_digest", "podcasts_per_digest", "exploration",
        "podcast_exploration", "max_per_source", "max_article_age_days", "max_candidates",
        "max_podcast_candidates", "apple_podcasts_links", "rescore_top", "fetch_interval_minutes",
        "poll_interval_minutes", "fetch_full_articles", "article_chars", "fetch_workers",
        "allow_private_addresses", "feeds_dir", "fetch_concurrency",
    ))
    config.exclude_categories = [category.lower() for category in general.get("exclude_categories", [])]
    config.interests = raw.get("profile", {}).get("interests", "").strip()

    config.llm = LLMConfig.from_section(raw.get("llm", {}), base)
    config.email = EmailConfig.from_section(raw.get("email", {}), base)

    feeds: list[Feed] = []
    for feed in raw.get("feeds", []):
        if isinstance(feed, str):
            feeds.append(Feed(url=feed))
        elif feed.get("url"):
            feeds.append(Feed.from_table(feed))
    feeds_path = Path(config.feeds_dir)
    if not feeds_path.is_absolute():
        feeds_path = base / feeds_path
    feeds += _load_feed_dir(feeds_path)
    seen = set()
    for feed in feeds:
        if feed.url in seen or feed.category.lower() in config.exclude_categories:
            continue
        seen.add(feed.url)
        config.feeds.append(feed)

    config.digest_hour_minute  # validate
    return config


def _parse_opml(data: bytes) -> list[Feed]:
    """Feeds from an OPML file; the enclosing outline's title becomes the category."""
    body = ElementTree.fromstring(data).find("body")
    return list(_opml_feeds(body, "")) if body is not None else []


def _opml_feeds(element: ElementTree.Element, category: str) -> Iterator[Feed]:
    for outline in element:
        if outline.tag != "outline":
            continue
        if outline.get("xmlUrl"):
            yield Feed.from_outline(outline, category)
        else:
            label = outline.get("title") or outline.get("text") or ""
            yield from _opml_feeds(outline, label.strip() or category)


def _parse_feed_list(text: str) -> list[Feed]:
    """Plain list: one feed per line, `URL [name]`; `# Category` lines set the category."""
    out, category = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            category = line.lstrip("#").strip()
            continue
        out.append(Feed.from_line(line, category))
    return out


def _load_feed_dir(path: Path) -> list[Feed]:
    if not path.is_dir():
        return []
    feeds: list[Feed] = []
    for file_path in sorted(path.iterdir()):
        try:
            if file_path.suffix.lower() in (".opml", ".xml"):
                feeds += _parse_opml(file_path.read_bytes())
            elif file_path.suffix.lower() == ".txt":
                feeds += _parse_feed_list(file_path.read_text())
        except (OSError, ElementTree.ParseError, UnicodeDecodeError) as error:
            raise ConfigError(f"{file_path}: {error}") from None
    return feeds

