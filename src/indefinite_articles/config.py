"""Configuration loading (TOML)."""

import os
import re
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = "/etc/indefinite-articles/indefinite-articles.toml"


class ConfigError(Exception):
    pass


@dataclass
class Feed:
    url: str
    name: str = ""
    category: str = ""
    max_items: int = 40  # newest entries kept per fetch


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:14b"
    api_key: str = ""
    timeout_seconds: int = 600
    embedding_model: str = "nomic-embed-text"
    embedding_prefix: str = "clustering: "
    embedding_batch: int = 64
    max_prompt_chars: int = 12000
    temperature: float = 0.2
    require_gpu: bool = True  # refuse to run if Ollama loads the model on CPU only


@dataclass
class ServerConfig:
    host: str = ""
    port: int = 0
    security: str = "ssl"  # ssl | starttls | none
    username: str = ""
    password: str = ""


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


@dataclass
class Config:
    database: str = "/var/lib/indefinite-articles/indefinite-articles.db"
    digest_time: str = "07:00"
    articles_per_digest: int = 10
    exploration: int = 2
    max_per_source: int = 3
    max_article_age_days: int = 3
    max_candidates: int = 150      # articles per day that get full-text + LLM scoring
    feeds_dir: str = "feeds.d"     # relative to the config file
    exclude_categories: list[str] = field(default_factory=list)
    fetch_concurrency: int = 16
    fetch_interval_minutes: int = 60
    poll_interval_minutes: int = 5
    fetch_full_articles: bool = True
    article_chars: int = 1500
    fetch_workers: int = 4
    allow_private_addresses: bool = False
    interests: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    feeds: list[Feed] = field(default_factory=list)

    @property
    def digest_hour_minute(self) -> tuple[int, int]:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", self.digest_time.strip())
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            raise ConfigError(f"digest_time must be HH:MM, got {self.digest_time!r}")
        return int(m.group(1)), int(m.group(2))


def _secret(section: dict, key: str, base: Path) -> str:
    """Read `key` directly, or from `key_file` (relative paths resolve next to the config)."""
    if section.get(f"{key}_file"):
        p = Path(section[f"{key}_file"])
        if not p.is_absolute():
            p = base / p
        return p.read_text().strip()
    return str(section.get(key, ""))


def _server(section: dict, default_port: int, base: Path) -> ServerConfig:
    sec = section.get("security", "ssl" if default_port in (465, 993) else "starttls")
    if sec not in ("ssl", "starttls", "none"):
        raise ConfigError(f"security must be ssl, starttls or none (got {sec!r})")
    return ServerConfig(
        host=section.get("host", ""),
        port=int(section.get("port", default_port)),
        security=sec,
        username=section.get("username", ""),
        password=_secret(section, "password", base),
    )


def load(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("INDEFINITE_ARTICLES_CONFIG", DEFAULT_CONFIG_PATH))
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    base = path.parent

    g = raw.get("general", {})
    cfg = Config()
    for k in (
        "database", "digest_time", "articles_per_digest", "exploration", "max_per_source",
        "max_article_age_days", "max_candidates", "fetch_interval_minutes",
        "poll_interval_minutes", "fetch_full_articles", "article_chars", "fetch_workers",
        "allow_private_addresses", "feeds_dir", "fetch_concurrency",
    ):
        if k in g:
            setattr(cfg, k, type(getattr(cfg, k))(g[k]))
    cfg.exclude_categories = [c.lower() for c in g.get("exclude_categories", [])]
    cfg.interests = raw.get("profile", {}).get("interests", "").strip()

    l = raw.get("llm", {})
    for k in ("base_url", "model", "api_key", "timeout_seconds",
              "max_prompt_chars", "temperature", "embedding_model", "embedding_prefix",
              "embedding_batch", "require_gpu"):
        if k in l:
            setattr(cfg.llm, k, type(getattr(cfg.llm, k))(l[k]))
    if l.get("api_key_file"):
        cfg.llm.api_key = _secret(l, "api_key", base)
    cfg.llm.base_url = cfg.llm.base_url.rstrip("/")

    e = raw.get("email", {})
    cfg.email.to = e.get("to", "")
    cfg.email.sender = e.get("from", "")
    cfg.email.smtp = _server(e.get("smtp", {}), 587, base)
    imap = e.get("imap", {})
    cfg.email.imap = _server(imap, 993, base)
    cfg.email.imap_folder = imap.get("folder", "INBOX")
    cfg.email.processed_folder = imap.get("processed_folder", "")
    cfg.email.allowed_senders = [s.lower() for s in e.get("allowed_senders", [])] or (
        [cfg.email.to.lower()] if cfg.email.to else []
    )
    cfg.email.send_acknowledgement = bool(e.get("send_acknowledgement", True))
    cfg.email.debug_info = bool(e.get("debug_info", True))

    feeds: list[Feed] = []
    for f in raw.get("feeds", []):
        if isinstance(f, str):
            feeds.append(Feed(url=f))
        elif f.get("url"):
            feeds.append(Feed(url=f["url"], name=f.get("name", ""),
                              category=f.get("category", ""),
                              max_items=int(f.get("max_items", 40))))
    fdir = Path(cfg.feeds_dir)
    if not fdir.is_absolute():
        fdir = base / fdir
    feeds += load_feed_dir(fdir)
    seen = set()
    for f in feeds:
        if f.url in seen or f.category.lower() in cfg.exclude_categories:
            continue
        seen.add(f.url)
        cfg.feeds.append(f)

    cfg.digest_hour_minute  # validate
    return cfg


def parse_opml(data: bytes) -> list[Feed]:
    """Feeds from an OPML file; the enclosing outline's title becomes the category."""
    out: list[Feed] = []

    def walk(el, category):
        for o in el:
            if o.tag != "outline":
                continue
            label = o.get("title") or o.get("text") or ""
            if o.get("xmlUrl"):
                try:
                    max_items = int(o.get("maxItems", 40))
                except ValueError:
                    max_items = 40
                out.append(Feed(url=o.get("xmlUrl").strip(), name=label.strip(),
                                category=category, max_items=max_items))
            else:
                walk(o, label.strip() or category)

    body = ET.fromstring(data).find("body")
    if body is not None:
        walk(body, "")
    return out


def parse_feed_list(text: str) -> list[Feed]:
    """Plain list: one feed per line, `URL [name]`; `# Category` lines set the category."""
    out, category = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            category = line.lstrip("#").strip()
            continue
        url, _, name = line.partition(" ")
        out.append(Feed(url=url, name=name.strip(), category=category))
    return out


def load_feed_dir(path: Path) -> list[Feed]:
    if not path.is_dir():
        return []
    feeds: list[Feed] = []
    for p in sorted(path.iterdir()):
        try:
            if p.suffix.lower() in (".opml", ".xml"):
                feeds += parse_opml(p.read_bytes())
            elif p.suffix.lower() == ".txt":
                feeds += parse_feed_list(p.read_text())
        except (OSError, ET.ParseError, UnicodeDecodeError) as e:
            raise ConfigError(f"{p}: {e}") from None
    return feeds


def validate_for_daemon(cfg: Config) -> list[str]:
    """Return a list of human-readable problems that would stop the daemon working."""
    problems = []
    if not cfg.feeds:
        problems.append("no feeds configured (add files to feeds.d/ or [[feeds]] entries)")
    if not cfg.email.to or not cfg.email.sender:
        problems.append("email.to and email.from must be set")
    if not cfg.email.smtp.host:
        problems.append("email.smtp.host is not set")
    if not cfg.email.imap.host:
        problems.append("email.imap.host is not set (replies will not be read)")
    return problems
