"""Command line interface."""

from __future__ import annotations

import argparse
import email
import email.policy
import logging
import sys

from . import __version__, config, extract, feeds, inbox, mailer, ranker
from .db import DB
from .llm import LLM, LLMError


def _setup_logging(verbose: bool):
    # Under systemd, journald adds timestamps itself.
    import os
    fmt = "%(levelname)s %(name)s: %(message)s"
    if not os.environ.get("JOURNAL_STREAM"):
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt)


def cmd_run(cfg, args):
    from .daemon import Service
    problems = config.validate_for_daemon(cfg)
    for p in problems:
        logging.warning("config: %s", p)
    if not cfg.feeds or not cfg.email.smtp.host:
        logging.error("cannot start: fix %s", args.config or config.DEFAULT_CONFIG_PATH)
        return 1
    if cfg.llm.require_gpu:
        llm = LLM(cfg.llm)
        try:
            llm.ping()  # makes Ollama load the model
        except LLMError as e:
            logging.error("cannot start: %s", e)
            return 1  # systemd retries
        if llm.on_cpu_only():
            logging.error("cannot start: %s is running on the CPU (no compatible GPU);"
                          " set require_gpu = false in [llm] to allow this", cfg.llm.model)
            return 78  # RestartPreventExitStatus in the unit file
    Service(cfg).run()
    return 0


def cmd_fetch(cfg, args):
    from .daemon import prefilter, refresh
    db = DB(cfg.database)
    llm = LLM(cfg.llm)
    if args.no_score:
        print(f"{feeds.update_all(cfg, db)} new articles")
        prefilter(cfg, db, llm)
    else:
        print(f"{refresh(cfg, db, llm)} new articles")
    return 0


def cmd_feeds(cfg, args):
    """List configured feeds with their health."""
    import time
    db = DB(cfg.database)
    states = db.feed_states()
    counts = db.articles_per_source(7)
    stats = db.source_stats()
    if args.categories:
        cats: dict[str, int] = {}
        for f in cfg.feeds:
            cats[f.category or "(none)"] = cats.get(f.category or "(none)", 0) + 1
        for c, n in sorted(cats.items()):
            print(f"{n:4}  {c}")
        print(f"{len(cfg.feeds):4}  total")
        return 0
    now = time.time()
    shown = 0
    for f in sorted(cfg.feeds, key=lambda f: (f.category, f.name or f.url)):
        st = states.get(f.url)
        source = f.name or (st["title"] if st else "") or f.url
        if st is None:
            health = "never fetched"
        elif st["failures"]:
            health = f"FAILING x{st['failures']}: {st['last_error'][:60]}"
        else:
            health = f"ok {int((now - st['last_success']) / 60)}m ago"
        if args.broken and (st is None or not st["failures"]):
            continue
        likes, dislikes = stats.get(source, (0, 0))
        print(f"{f.category[:18]:18}  {source[:34]:34}  {counts.get(source, 0):4}/wk  "
              f"+{likes}/-{dislikes}  {health}")
        if args.broken:
            print(f"{'':18}  {f.url}")
        shown += 1
    print(f"-- {shown} feeds shown, {len(cfg.feeds)} configured")
    return 0


def cmd_digest(cfg, args):
    from .daemon import send_digest
    db = DB(cfg.database)
    msg = send_digest(cfg, db, LLM(cfg.llm), dry_run=args.dry_run)
    if msg is None:
        print("no candidate articles; run `indefinite-articles fetch` first", file=sys.stderr)
        return 1
    if args.dry_run:
        print(msg.get_body(preferencelist=("plain",)).get_content())
    else:
        print(f"sent: {msg['Subject']}")
    return 0


def cmd_poll(cfg, args):
    db = DB(cfg.database)
    llm = LLM(cfg.llm)
    print(f"processed {inbox.poll(cfg, db, llm)} replies")
    if ranker.update_profile(cfg, db, llm):
        print("profile updated")
    return 0


def cmd_feedback(cfg, args):
    """Process a reply saved as an .eml file (or stdin) without IMAP."""
    db = DB(cfg.database)
    llm = LLM(cfg.llm)
    raw = open(args.file, "rb").read() if args.file != "-" else sys.stdin.buffer.read()
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    import hashlib
    # Same exactly-once bookkeeping as IMAP polling, so a later poll won't count it again.
    mid = inbox.message_id(msg, "file-" + hashlib.sha256(raw).hexdigest()[:16])
    status, summary = inbox.handle_message(cfg, db, llm, msg, mid)
    print(status)
    for line in summary:
        print(" ", line)
    if ranker.update_profile(cfg, db, llm):
        print("profile updated")
    return 0 if status == "ok" else 1


def cmd_profile(cfg, args):
    db = DB(cfg.database)
    if args.set:
        db.add_profile(args.set, note="set manually")
        print("profile replaced")
        return 0
    if args.history:
        for r in db.conn.execute("SELECT * FROM profiles ORDER BY id"):
            print(f"--- version {r['id']} ({r['note']})")
            print(r["text"], "\n")
        return 0
    row = ranker.ensure_profile(cfg, db)
    print(row["text"])
    return 0


def cmd_extract(cfg, args):
    """Show what the LLM would see for a URL."""
    print(extract.fetch_page(args.url, cfg.article_chars, cfg.allow_private_addresses))
    return 0


def cmd_status(cfg, args):
    db = DB(cfg.database)
    for k, v in db.stats().items():
        print(f"{k:18} {v}")
    print(f"{'last_digest':18} {db.get_state('last_digest_date', 'never')}")
    return 0


def cmd_check(cfg, args):
    ok = True
    for p in config.validate_for_daemon(cfg):
        print(f"[warn] config: {p}")
    try:
        print(f"[ok]   LLM {cfg.llm.model} at {cfg.llm.base_url}: {LLM(cfg.llm).ping().strip()[:40]!r}")
    except LLMError as e:
        ok = False
        print(f"[FAIL] LLM: {e}")
    if cfg.llm.embedding_model:
        try:
            dim = len(LLM(cfg.llm).embed(["test"])[0])
            print(f"[ok]   embeddings {cfg.llm.embedding_model}: {dim} dimensions")
        except LLMError as e:
            print(f"[warn] embeddings {cfg.llm.embedding_model}: {e}\n"
                  f"       (without it only the newest articles are considered; "
                  f"try: ollama pull {cfg.llm.embedding_model})")
    print(f"[info] {len(cfg.feeds)} feeds configured")
    for f in cfg.feeds[: args.feeds]:
        try:
            title, entries = feeds.parse_feed(feeds.fetch(f.url))
            print(f"[ok]   feed {f.name or title}: {len(entries)} entries")
        except Exception as e:
            ok = False
            print(f"[FAIL] feed {f.url}: {e}")
    if cfg.email.imap.host:
        try:
            conn = inbox.connect(cfg)
            typ, _ = conn.select(cfg.email.imap_folder, readonly=True)
            conn.logout()
            print(f"[{'ok' if typ == 'OK' else 'FAIL'}]   IMAP {cfg.email.imap.host} folder {cfg.email.imap_folder}")
            ok &= typ == "OK"
        except Exception as e:
            ok = False
            print(f"[FAIL] IMAP: {e}")
    if args.send_test:
        from email.message import EmailMessage
        m = EmailMessage()
        m["Subject"], m["From"], m["To"] = "indefinite-articles: test email", cfg.email.sender, cfg.email.to
        m.set_content("If you can read this, SMTP is configured correctly.")
        try:
            mailer.send(cfg, m)
            print(f"[ok]   SMTP test email sent to {cfg.email.to}")
        except Exception as e:
            ok = False
            print(f"[FAIL] SMTP: {e}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="indefinite-articles",
        description="Daily article recommendations by email, trained on your replies with a local LLM.",
    )
    ap.add_argument("-c", "--config", help=f"config file (default {config.DEFAULT_CONFIG_PATH})")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"indefinite-articles {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="run the service (used by systemd)").set_defaults(fn=cmd_run)
    p = sub.add_parser("feeds", help="list feeds and their health")
    p.add_argument("--broken", action="store_true", help="only failing feeds")
    p.add_argument("--categories", action="store_true", help="count feeds per category")
    p.set_defaults(fn=cmd_feeds)
    p = sub.add_parser("fetch", help="fetch feeds now and score new articles")
    p.add_argument("--no-score", action="store_true")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("digest", help="build and send a digest now")
    p.add_argument("-n", "--dry-run", action="store_true", help="print instead of sending")
    p.set_defaults(fn=cmd_digest)
    sub.add_parser("poll", help="read replies from IMAP now").set_defaults(fn=cmd_poll)
    p = sub.add_parser("feedback", help="process a reply from an .eml file (- for stdin)")
    p.add_argument("file")
    p.set_defaults(fn=cmd_feedback)
    p = sub.add_parser("profile", help="show the learned reader profile")
    p.add_argument("--history", action="store_true")
    p.add_argument("--set", metavar="TEXT", help="replace the profile")
    p.set_defaults(fn=cmd_profile)
    sub.add_parser("status", help="show database statistics").set_defaults(fn=cmd_status)
    p = sub.add_parser("extract", help="show the article text extracted from a URL")
    p.add_argument("url")
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("check", help="test LLM, feeds, IMAP and (optionally) SMTP")
    p.add_argument("--send-test", action="store_true", help="also send a test email")
    p.add_argument("--feeds", type=int, default=3, help="number of feeds to test")
    p.set_defaults(fn=cmd_check)

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        cfg = config.load(args.config)
    except (config.ConfigError, OSError) as e:
        print(f"indefinite-articles: {e}", file=sys.stderr)
        return 2
    try:
        return args.fn(cfg, args)
    except KeyboardInterrupt:
        return 130
