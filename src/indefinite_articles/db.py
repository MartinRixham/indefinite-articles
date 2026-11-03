"""SQLite storage."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    title_key TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    published REAL,
    fetched_at REAL NOT NULL,
    score REAL,
    reason TEXT,
    score_profile_id INTEGER,
    sent_digest_id INTEGER,
    content TEXT,
    content_status TEXT,
    category TEXT NOT NULL DEFAULT '',
    prefilter REAL,
    prefilter_version INTEGER
);
CREATE INDEX IF NOT EXISTS articles_title_key ON articles(title_key);
CREATE INDEX IF NOT EXISTS articles_fetched ON articles(fetched_at);

CREATE TABLE IF NOT EXISTS embeddings (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS feeds (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    etag TEXT,
    last_modified TEXT,
    last_attempt REAL,
    last_success REAL,
    failures INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    next_attempt REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    message_id TEXT NOT NULL,
    sent_at REAL NOT NULL,
    local_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS digest_items (
    digest_id INTEGER NOT NULL REFERENCES digests(id),
    position INTEGER NOT NULL,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    exploratory INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (digest_id, position)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    rating INTEGER NOT NULL,          -- +1 like, -1 dislike, 0 comment only
    comment TEXT NOT NULL DEFAULT '',
    received_at REAL NOT NULL,
    UNIQUE (article_id)
);

CREATE TABLE IF NOT EXISTS general_comments (
    id INTEGER PRIMARY KEY,
    digest_id INTEGER,
    comment TEXT NOT NULL,
    received_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS processed_messages (
    message_id TEXT PRIMARY KEY,
    processed_at REAL NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class DB:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _migrate(self):
        """Add columns introduced after 0.1.0 to existing databases."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(articles)")}
        if not cols:
            return  # new database
        for name, decl in (("content", "TEXT"), ("content_status", "TEXT"),
                           ("category", "TEXT NOT NULL DEFAULT ''"), ("prefilter", "REAL"),
                           ("prefilter_version", "INTEGER")):
            if name not in cols:
                self.conn.execute(f"ALTER TABLE articles ADD COLUMN {name} {decl}")

    def close(self):
        self.conn.close()

    # -- state ---------------------------------------------------------------
    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    # -- profile -------------------------------------------------------------
    def current_profile(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM profiles ORDER BY id DESC LIMIT 1").fetchone()

    def add_profile(self, text: str, note: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO profiles(text, created_at, note) VALUES(?, ?, ?)",
            (text, time.time(), note),
        )
        self.conn.commit()
        return cur.lastrowid

    # -- articles ------------------------------------------------------------
    def add_article(self, url, title, title_key, summary, source, published, category="") -> bool:
        if self.conn.execute(
            "SELECT 1 FROM articles WHERE url=? OR (title_key=? AND title_key != '')",
            (url, title_key),
        ).fetchone():
            return False
        self.conn.execute(
            "INSERT INTO articles(url, title, title_key, summary, source, published, fetched_at,"
            " category) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (url, title, title_key, summary, source, published, time.time(), category),
        )
        return True

    def candidates(self, max_age_days: int, limit: int) -> list[sqlite3.Row]:
        """Unsent, recent articles, balanced across sources so a busy feed can't crowd out others."""
        cutoff = time.time() - max_age_days * 86400
        n_sources = self.conn.execute(
            "SELECT COUNT(DISTINCT source) FROM articles WHERE sent_digest_id IS NULL "
            "AND COALESCE(published, fetched_at) >= ?", (cutoff,)
        ).fetchone()[0] or 1
        per_source = max(5, -(-limit // n_sources))
        return self.conn.execute(
            "SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY source "
            "  ORDER BY COALESCE(published, fetched_at) DESC) AS rn "
            "  FROM articles WHERE sent_digest_id IS NULL AND COALESCE(published, fetched_at) >= ?) "
            "WHERE rn <= ? ORDER BY rn, COALESCE(published, fetched_at) DESC LIMIT ?",
            (cutoff, per_source, limit),
        ).fetchall()

    def pool(self, max_age_days: int, limit: int = 50000) -> list[sqlite3.Row]:
        """All unsent recent articles."""
        cutoff = time.time() - max_age_days * 86400
        return self.conn.execute(
            "SELECT * FROM articles WHERE sent_digest_id IS NULL "
            "AND COALESCE(published, fetched_at) >= ? "
            "ORDER BY COALESCE(published, fetched_at) DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()

    # -- embeddings ----------------------------------------------------------
    def unembedded(self, max_age_days: int, limit: int) -> list[sqlite3.Row]:
        """Recent unsent articles, and any rated article, that have no embedding yet."""
        cutoff = time.time() - max_age_days * 86400
        return self.conn.execute(
            "SELECT a.id, a.title, a.summary FROM articles a "
            "LEFT JOIN embeddings e ON e.article_id = a.id WHERE e.article_id IS NULL AND ("
            " (a.sent_digest_id IS NULL AND COALESCE(a.published, a.fetched_at) >= ?)"
            " OR a.id IN (SELECT article_id FROM feedback WHERE rating != 0)) "
            "ORDER BY a.id DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()

    def set_embedding(self, article_id: int, vec: bytes):
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(article_id, vec) VALUES(?, ?)", (article_id, vec)
        )

    def rated_vectors(self, rating: int, limit: int) -> list[bytes]:
        return [r[0] for r in self.conn.execute(
            "SELECT e.vec FROM feedback f JOIN embeddings e ON e.article_id = f.article_id "
            "WHERE f.rating = ? ORDER BY f.received_at DESC LIMIT ?", (rating, limit))]

    def prefilter_todo(self, max_age_days: int, version: int):
        cutoff = time.time() - max_age_days * 86400
        return self.conn.execute(
            "SELECT a.id, e.vec FROM articles a JOIN embeddings e ON e.article_id = a.id "
            "WHERE a.sent_digest_id IS NULL AND COALESCE(a.published, a.fetched_at) >= ? "
            "AND (a.prefilter_version IS NULL OR a.prefilter_version != ?)",
            (cutoff, version),
        ).fetchall()

    def set_prefilter(self, rows: list[tuple[float, int, int]]):
        """rows: (score, version, article_id)"""
        self.conn.executemany(
            "UPDATE articles SET prefilter=?, prefilter_version=? WHERE id=?", rows
        )
        self.conn.commit()

    def feedback_signature(self) -> str:
        n, ts = self.conn.execute("SELECT COUNT(*), MAX(received_at) FROM feedback").fetchone()
        return f"{n}:{ts}"

    # -- feed fetch state ----------------------------------------------------
    def feed_states(self) -> dict[str, sqlite3.Row]:
        return {r["url"]: r for r in self.conn.execute("SELECT * FROM feeds")}

    def feed_ok(self, url: str, title: str, etag, last_modified):
        now = time.time()
        self.conn.execute(
            "INSERT INTO feeds(url, title, etag, last_modified, last_attempt, last_success,"
            " failures, last_error, next_attempt) VALUES(?, ?, ?, ?, ?, ?, 0, '', 0) "
            "ON CONFLICT(url) DO UPDATE SET title=CASE WHEN excluded.title != '' "
            " THEN excluded.title ELSE feeds.title END, etag=excluded.etag,"
            " last_modified=excluded.last_modified, last_attempt=excluded.last_attempt,"
            " last_success=excluded.last_success, failures=0, last_error='', next_attempt=0",
            (url, title, etag, last_modified, now, now),
        )

    def feed_failed(self, url: str, error: str, base_interval: float):
        now = time.time()
        row = self.conn.execute("SELECT failures FROM feeds WHERE url=?", (url,)).fetchone()
        failures = (row["failures"] if row else 0) + 1
        # Retry transient errors next cycle, then back off exponentially up to a day.
        delay = 0 if failures < 2 else min(86400, base_interval * 2 ** min(failures - 1, 8))
        self.conn.execute(
            "INSERT INTO feeds(url, last_attempt, failures, last_error, next_attempt)"
            " VALUES(?, ?, ?, ?, ?) ON CONFLICT(url) DO UPDATE SET"
            " last_attempt=excluded.last_attempt, failures=excluded.failures,"
            " last_error=excluded.last_error, next_attempt=excluded.next_attempt",
            (url, now, failures, error[:300], now + delay),
        )

    def articles_per_source(self, days: int) -> dict[str, int]:
        cutoff = time.time() - days * 86400
        return {r[0]: r[1] for r in self.conn.execute(
            "SELECT source, COUNT(*) FROM articles WHERE fetched_at >= ? GROUP BY source",
            (cutoff,))}

    def set_content(self, article_id: int, content: str, status: str):
        self.conn.execute(
            "UPDATE articles SET content=?, content_status=? WHERE id=?",
            (content, status, article_id),
        )

    def set_score(self, article_id: int, score: float, reason: str, profile_id: int):
        self.conn.execute(
            "UPDATE articles SET score=?, reason=?, score_profile_id=? WHERE id=?",
            (score, reason, profile_id, article_id),
        )

    def rated_examples(self, rating: int, limit: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT a.title, a.source, f.comment FROM feedback f "
            "JOIN articles a ON a.id = f.article_id WHERE f.rating = ? "
            "ORDER BY f.received_at DESC LIMIT ?",
            (rating, limit),
        ).fetchall()

    def source_stats(self) -> dict[str, tuple[int, int]]:
        """source -> (likes, dislikes)"""
        rows = self.conn.execute(
            "SELECT a.source, SUM(f.rating > 0) AS likes, SUM(f.rating < 0) AS dislikes "
            "FROM feedback f JOIN articles a ON a.id = f.article_id GROUP BY a.source"
        ).fetchall()
        return {r["source"]: (r["likes"] or 0, r["dislikes"] or 0) for r in rows}

    def prune(self, keep_days: int = 30):
        """Drop old articles that were never sent (sent/rated ones are training data)."""
        cutoff = time.time() - keep_days * 86400
        # Embeddings first: databases from before 0.3 lack the cascade on older rows.
        self.conn.execute(
            "DELETE FROM embeddings WHERE article_id IN (SELECT id FROM articles "
            "WHERE sent_digest_id IS NULL AND fetched_at < ?)", (cutoff,)
        )
        self.conn.execute(
            "DELETE FROM articles WHERE sent_digest_id IS NULL AND fetched_at < ?", (cutoff,)
        )
        self.conn.execute(
            "DELETE FROM processed_messages WHERE processed_at < ?", (cutoff,)
        )
        self.conn.commit()

    # -- digests -------------------------------------------------------------
    def record_digest(self, token, message_id, local_date, items: list[tuple[int, bool]]) -> int:
        cur = self.conn.execute(
            "INSERT INTO digests(token, message_id, sent_at, local_date) VALUES(?, ?, ?, ?)",
            (token, message_id, time.time(), local_date),
        )
        did = cur.lastrowid
        for pos, (aid, exploratory) in enumerate(items, start=1):
            self.conn.execute(
                "INSERT INTO digest_items(digest_id, position, article_id, exploratory)"
                " VALUES(?, ?, ?, ?)",
                (did, pos, aid, int(exploratory)),
            )
            self.conn.execute("UPDATE articles SET sent_digest_id=? WHERE id=?", (did, aid))
        self.conn.commit()
        return did

    def find_digest(self, token: str | None = None, message_ids: list[str] = ()) -> sqlite3.Row | None:
        if token:
            row = self.conn.execute("SELECT * FROM digests WHERE token=?", (token,)).fetchone()
            if row:
                return row
        for mid in message_ids:
            row = self.conn.execute(
                "SELECT * FROM digests WHERE message_id=?", (mid.strip(),)
            ).fetchone()
            if row:
                return row
        return None

    def digest_items(self, digest_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT di.position, di.exploratory, a.* FROM digest_items di "
            "JOIN articles a ON a.id = di.article_id WHERE di.digest_id=? ORDER BY di.position",
            (digest_id,),
        ).fetchall()

    # -- feedback ------------------------------------------------------------
    def add_feedback(self, article_id: int, rating: int, comment: str):
        self.conn.execute(
            "INSERT INTO feedback(article_id, rating, comment, received_at) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(article_id) DO UPDATE SET rating=excluded.rating, "
            "comment=excluded.comment, received_at=excluded.received_at",
            (article_id, rating, comment, time.time()),
        )

    def feedback_since(self, ts: float):
        fb = self.conn.execute(
            "SELECT a.title, a.source, f.rating, f.comment FROM feedback f "
            "JOIN articles a ON a.id = f.article_id WHERE f.received_at > ? "
            "ORDER BY f.received_at",
            (ts,),
        ).fetchall()
        comments = self.conn.execute(
            "SELECT comment FROM general_comments WHERE received_at > ? ORDER BY received_at",
            (ts,),
        ).fetchall()
        return fb, comments

    def add_general_comment(self, digest_id: int | None, comment: str):
        self.conn.execute(
            "INSERT INTO general_comments(digest_id, comment, received_at) VALUES(?, ?, ?)",
            (digest_id, comment, time.time()),
        )

    def message_processed(self, message_id: str) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM processed_messages WHERE message_id=?", (message_id,)
        ).fetchone())

    def mark_message(self, message_id: str, status: str, commit: bool = True):
        self.conn.execute(
            "INSERT OR REPLACE INTO processed_messages(message_id, processed_at, status)"
            " VALUES(?, ?, ?)",
            (message_id, time.time(), status),
        )
        if commit:
            self.conn.commit()

    def stats(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]
        return {
            "articles": q("SELECT COUNT(*) FROM articles"),
            "unsent_scored": q("SELECT COUNT(*) FROM articles WHERE sent_digest_id IS NULL AND score IS NOT NULL"),
            "embedded": q("SELECT COUNT(*) FROM embeddings"),
            "feeds_failing": q("SELECT COUNT(*) FROM feeds WHERE failures > 0"),
            "full_text": q("SELECT COUNT(*) FROM articles WHERE content_status = 'ok'"),
            "full_text_failed": q("SELECT COUNT(*) FROM articles WHERE content_status NOT IN ('ok', 'empty')"),
            "digests": q("SELECT COUNT(*) FROM digests"),
            "likes": q("SELECT COUNT(*) FROM feedback WHERE rating > 0"),
            "dislikes": q("SELECT COUNT(*) FROM feedback WHERE rating < 0"),
            "profile_versions": q("SELECT COUNT(*) FROM profiles"),
        }
