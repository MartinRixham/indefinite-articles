"""Fetch article pages and extract their main text, so the LLM judges more than a headline."""

import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from html.parser import HTMLParser

from .feeds import USER_AGENT

log = logging.getLogger(__name__)

MAX_PAGE_BYTES = 3 * 1024 * 1024
TIMEOUT = 20

# Never text: always skipped.
SKIP_TAGS = {"script", "style", "noscript", "svg", "form", "button", "iframe", "template",
             "select", "figure"}
# Usually page chrome, but some sites wrap their whole page in one: dropped only if the
# article still has enough text without them.
NOISE_TAGS = {"nav", "header", "footer", "aside"}
MIN_GOOD_CHARACTERS = 300
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "td", "dd"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}
NOISE_RE = re.compile(r"comment|share|social|related|newsletter|subscribe|promo|advert|"
                      r"cookie|banner|sidebar|popup|modal|breadcrumb|footer|menu", re.I)
CONTENT_RE = re.compile(r"content|article|post|body|entry|main|story|text", re.I)


def _is_noise(attributes: str) -> bool:
    """True for containers like class="share-buttons", but not class="post-content share-on"."""
    tokens = attributes.split()
    return any(NOISE_RE.search(token) for token in tokens) and not any(CONTENT_RE.search(token) for token in tokens)


class BlockedAddress(Exception):
    pass


class _Extractor(HTMLParser):
    """Collects text blocks, noting whether each is inside <article>/<main>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, skipping, noisy)
        # (text, container level: 2 article, 1 main, 0 other, inside a noisy container)
        self.blocks: list[tuple[str, int, bool]] = []
        self.buffer: list[str] = []
        self.in_article = 0
        self.in_main = 0
        self.metadata: dict[str, str] = {}
        self.title = ""
        self._in_title = False

    @property
    def skipping(self):
        return bool(self.stack) and self.stack[-1][1]

    @property
    def noisy(self):
        return bool(self.stack) and self.stack[-1][2]

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
        self.buffer = []
        if text:
            level = 2 if self.in_article else 1 if self.in_main else 0
            self.blocks.append((text, level, self.noisy))

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = attributes.get("content")
            if key in ("description", "og:description", "twitter:description") and content:
                self.metadata.setdefault("description", content.strip())
            return
        if tag == "title":
            self._in_title = True
        if tag in VOID_TAGS:
            if tag == "br":
                self.buffer.append(" ")
            return
        classes = f"{attributes.get('class', '')} {attributes.get('id', '')} {attributes.get('role', '')}"
        skip = self.skipping or tag in SKIP_TAGS
        noisy = self.noisy or tag in NOISE_TAGS or (
            tag in ("div", "section", "ul", "ol") and _is_noise(classes))
        if tag in BLOCK_TAGS or noisy != self.noisy:
            self._flush()
        self.stack.append((tag, skip, noisy))
        if tag == "article":
            self.in_article += 1
        elif tag == "main" or attributes.get("role") == "main":
            self.in_main += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in VOID_TAGS:
            return
        # Pop to the matching tag (tolerates unclosed elements).
        for depth in range(len(self.stack) - 1, -1, -1):
            if self.stack[depth][0] == tag:
                was_noisy = self.noisy
                for open_tag, _, _ in self.stack[depth:]:
                    if open_tag == "article":
                        self.in_article -= 1
                    elif open_tag == "main":
                        self.in_main = max(0, self.in_main - 1)
                if was_noisy != (bool(self.stack[:depth]) and self.stack[depth - 1][2]):
                    self._flush()  # don't merge text across a noisy/clean boundary
                del self.stack[depth:]
                break
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skipping:
            self.buffer.append(data)

    def close(self):
        super().close()
        self._flush()


def extract_text(html: str, max_characters: int = 2000) -> str:
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as error:  # malformed markup: use whatever was collected
        log.debug("html parse error: %s", error)
    text = _main_text([block for block in parser.blocks if not block[2]])
    if len(text) < MIN_GOOD_CHARACTERS:
        # Filtering removed most of the page: it was probably wrapped in a "footer" etc.
        everything = _main_text(parser.blocks)
        if len(everything) >= 2 * len(text):
            text = everything
    if len(text) < 200 and parser.metadata.get("description"):
        text = (parser.metadata["description"] + "\n" + text).strip()
    if len(text) > max_characters:
        text = text[: max_characters - 1].rsplit(" ", 1)[0] + "…"
    return text


def _main_text(blocks) -> str:
    """Prefer <article>, then <main>; keep reasonably long blocks (real paragraphs)."""
    best = max((level for _, level, _ in blocks), default=0)
    return "\n".join(text for text, level, _ in blocks if level == best and (
        len(text) >= 60 or text.endswith((".", "?", "!", ":"))))


def _check_host(host: str, allow_private: bool):
    if allow_private:
        return
    try:
        address_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        raise urllib.error.URLError(error) from None
    for info in address_infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise BlockedAddress(f"{host} resolves to non-public address {ip}")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme not in ("http", "https"):
            raise BlockedAddress(f"redirect to {parts.scheme} URL")
        _check_host(parts.hostname or "", self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(url: str, max_characters: int, allow_private: bool = False) -> str:
    """Download `url` and return its extracted main text ('' if none)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedAddress(f"unsupported URL {url!r}")
    # Feed content is third-party: don't let it point us at services on the LAN or localhost.
    _check_host(parts.hostname, allow_private)
    opener = urllib.request.build_opener(_SafeRedirect(allow_private))
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        "Accept-Language": "en;q=1.0, *;q=0.5",
    })
    with opener.open(request, timeout=TIMEOUT) as response:
        content_type = response.headers.get_content_type()
        if content_type not in ("text/html", "application/xhtml+xml"):
            raise ValueError(f"not HTML ({content_type})")
        raw = response.read(MAX_PAGE_BYTES)
        charset = response.headers.get_content_charset()
    if not charset:
        match = re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', raw[:4096], re.I)
        charset = match.group(1).decode() if match else "utf-8"
    try:
        html = raw.decode(charset, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")
    return extract_text(html, max_characters)


def _fetch_article(config, stop, article) -> tuple[int, str | None, str]:
    """(article id, text, content status) for one article."""
    if stop is not None and stop.is_set():
        return article["id"], None, "skipped"
    try:
        text = fetch_page(article["url"], config.article_chars, config.allow_private_addresses)
        return article["id"], text, "ok" if text else "empty"
    except BlockedAddress as error:
        return article["id"], "", f"blocked: {error}"
    except Exception as error:
        return article["id"], "", f"failed: {str(error)[:150]}"


def enrich_candidates(config, database, stop=None) -> int:
    """Fetch page text for candidate articles that haven't been tried yet."""
    if not config.fetch_full_articles:
        return 0
    from .ranker import shortlist
    todo = [article for article in shortlist(config, database) if article["content_status"] is None]
    if not todo:
        return 0
    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, config.fetch_workers)) as pool:
        for article_id, text, status in pool.map(partial(_fetch_article, config, stop), todo):
            if status == "skipped":
                continue
            database.set_content(article_id, text or "", status)
            database.connection.commit()  # don't hold the write lock while waiting for more pages
            if status == "ok":
                ok += 1
            else:
                log.debug("article %d: %s", article_id, status)
    database.connection.commit()
    log.info("fetched full text for %d/%d articles", ok, len(todo))
    return ok
