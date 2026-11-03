"""Fetch article pages and extract their main text, so the LLM judges more than a headline."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
MIN_GOOD_CHARS = 300
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "td", "dd"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}
NOISE_RE = re.compile(r"comment|share|social|related|newsletter|subscribe|promo|advert|"
                      r"cookie|banner|sidebar|popup|modal|breadcrumb|footer|menu", re.I)
CONTENT_RE = re.compile(r"content|article|post|body|entry|main|story|text", re.I)


def _is_noise(attr: str) -> bool:
    """True for containers like class="share-buttons", but not class="post-content share-on"."""
    tokens = attr.split()
    return any(NOISE_RE.search(t) for t in tokens) and not any(CONTENT_RE.search(t) for t in tokens)


class BlockedAddress(Exception):
    pass


class _Extractor(HTMLParser):
    """Collects text blocks, noting whether each is inside <article>/<main>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, skipping, noisy)
        # (text, container level: 2 article, 1 main, 0 other, inside a noisy container)
        self.blocks: list[tuple[str, int, bool]] = []
        self.buf: list[str] = []
        self.in_article = 0
        self.in_main = 0
        self.meta: dict[str, str] = {}
        self.title = ""
        self._in_title = False

    @property
    def skipping(self):
        return bool(self.stack) and self.stack[-1][1]

    @property
    def noisy(self):
        return bool(self.stack) and self.stack[-1][2]

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if text:
            level = 2 if self.in_article else 1 if self.in_main else 0
            self.blocks.append((text, level, self.noisy))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key in ("description", "og:description", "twitter:description") and a.get("content"):
                self.meta.setdefault("description", a["content"].strip())
            return
        if tag == "title":
            self._in_title = True
        if tag in VOID_TAGS:
            if tag == "br":
                self.buf.append(" ")
            return
        cls = f"{a.get('class', '')} {a.get('id', '')} {a.get('role', '')}"
        skip = self.skipping or tag in SKIP_TAGS
        noisy = self.noisy or tag in NOISE_TAGS or (
            tag in ("div", "section", "ul", "ol") and _is_noise(cls))
        if tag in BLOCK_TAGS or noisy != self.noisy:
            self._flush()
        self.stack.append((tag, skip, noisy))
        if tag == "article":
            self.in_article += 1
        elif tag == "main" or a.get("role") == "main":
            self.in_main += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in VOID_TAGS:
            return
        # Pop to the matching tag (tolerates unclosed elements).
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                was_noisy = self.noisy
                for t, _, _ in self.stack[i:]:
                    if t == "article":
                        self.in_article -= 1
                    elif t == "main":
                        self.in_main = max(0, self.in_main - 1)
                if was_noisy != (bool(self.stack[:i]) and self.stack[i - 1][2]):
                    self._flush()  # don't merge text across a noisy/clean boundary
                del self.stack[i:]
                break
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skipping:
            self.buf.append(data)

    def close(self):
        super().close()
        self._flush()


def extract_text(html: str, max_chars: int = 2000) -> str:
    p = _Extractor()
    try:
        p.feed(html)
        p.close()
    except Exception as e:  # malformed markup: use whatever was collected
        log.debug("html parse error: %s", e)

    def pick(blocks):
        # Prefer <article>, then <main>; keep reasonably long blocks (real paragraphs).
        best = max((lvl for _, lvl, _ in blocks), default=0)
        return "\n".join(t for t, lvl, _ in blocks if lvl == best and (
            len(t) >= 60 or t.endswith((".", "?", "!", ":"))))

    text = pick([b for b in p.blocks if not b[2]])
    if len(text) < MIN_GOOD_CHARS:
        # Filtering removed most of the page: it was probably wrapped in a "footer" etc.
        everything = pick(p.blocks)
        if len(everything) >= 2 * len(text):
            text = everything
    if len(text) < 200 and p.meta.get("description"):
        text = (p.meta["description"] + "\n" + text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return text


def _check_host(host: str, allow_private: bool):
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise urllib.error.URLError(e) from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise BlockedAddress(f"{host} resolves to non-public address {ip}")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(newurl)
        if u.scheme not in ("http", "https"):
            raise BlockedAddress(f"redirect to {u.scheme} URL")
        _check_host(u.hostname or "", self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(url: str, max_chars: int, allow_private: bool = False) -> str:
    """Download `url` and return its extracted main text ('' if none)."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise BlockedAddress(f"unsupported URL {url!r}")
    # Feed content is third-party: don't let it point us at services on the LAN or localhost.
    _check_host(u.hostname, allow_private)
    opener = urllib.request.build_opener(_SafeRedirect(allow_private))
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        "Accept-Language": "en;q=1.0, *;q=0.5",
    })
    with opener.open(req, timeout=TIMEOUT) as resp:
        ctype = resp.headers.get_content_type()
        if ctype not in ("text/html", "application/xhtml+xml"):
            raise ValueError(f"not HTML ({ctype})")
        raw = resp.read(MAX_PAGE_BYTES)
        charset = resp.headers.get_content_charset()
    if not charset:
        m = re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', raw[:4096], re.I)
        charset = m.group(1).decode() if m else "utf-8"
    try:
        html = raw.decode(charset, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")
    return extract_text(html, max_chars)


def enrich_candidates(cfg, db, stop=None) -> int:
    """Fetch page text for candidate articles that haven't been tried yet."""
    if not cfg.fetch_full_articles:
        return 0
    from .ranker import shortlist
    todo = [a for a in shortlist(cfg, db) if a["content_status"] is None]
    if not todo:
        return 0

    def work(a):
        if stop is not None and stop.is_set():
            return a["id"], None, "skipped"
        try:
            text = fetch_page(a["url"], cfg.article_chars, cfg.allow_private_addresses)
            return a["id"], text, "ok" if text else "empty"
        except BlockedAddress as e:
            return a["id"], "", f"blocked: {e}"
        except Exception as e:
            return a["id"], "", f"failed: {str(e)[:150]}"

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.fetch_workers)) as pool:
        for aid, text, status in pool.map(work, todo):
            if status == "skipped":
                continue
            db.set_content(aid, text or "", status)
            if status == "ok":
                ok += 1
            else:
                log.debug("article %d: %s", aid, status)
    db.conn.commit()
    log.info("fetched full text for %d/%d articles", ok, len(todo))
    return ok
