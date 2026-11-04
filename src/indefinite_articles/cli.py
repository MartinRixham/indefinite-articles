"""Command line interface."""

import argparse
import email
import email.policy
import logging
import sys
from collections.abc import Sequence

from . import __version__, extract, feeds, inbox, mailer, ranker
from .config import DEFAULT_CONFIG_PATH, Config, ConfigError, load as load_config
from .database import Database, SourceStatistics
from .llm import LLM, LLMError
from .parsed_feed import ParsedFeed


def _setup_logging(verbose: bool):
    # Under systemd, journald adds timestamps itself.
    import os
    log_format = "%(levelname)s %(name)s [%(threadName)s]: %(message)s"
    if not os.environ.get("JOURNAL_STREAM"):
        log_format = "%(asctime)s " + log_format
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=log_format)


def _command_run(config: Config, args: argparse.Namespace):
    from .service import Service
    problems = config.problems()
    for problem in problems:
        logging.warning("config: %s", problem)
    if not config.feeds or not config.email.smtp.host:
        logging.error("cannot start: fix %s", args.config or DEFAULT_CONFIG_PATH)
        return 1
    if config.llm.require_gpu:
        llm = LLM(config.llm, deadline=config.next_digest_time)
        try:
            llm.ping()  # makes Ollama load the model
        except LLMError as error:
            logging.error("cannot start: %s", error)
            return 1  # systemd retries
        if llm.on_cpu_only():
            logging.error("cannot start: %s is running on the CPU (no compatible GPU);"
                          " set require_gpu = false in [llm] to allow this", config.llm.model)
            return 78  # RestartPreventExitStatus in the unit file
    Service(config).run()
    return 0


def _command_fetch(config: Config, args: argparse.Namespace):
    from .daemon import prefilter, refresh
    with Database(config.database) as database:
        llm = LLM(config.llm)
        if args.no_score:
            print(f"{feeds.update_all(config, database)} new articles")
            prefilter(config, database, llm)
        else:
            print(f"{refresh(config, database, llm)} new articles")
        return 0


def _command_feeds(config: Config, args: argparse.Namespace):
    """List configured feeds with their health."""
    import time
    with Database(config.database) as database:
        states = database.feed_states()
        counts = database.articles_per_source(7)
        statistics = database.source_statistics()
        if args.categories:
            categories: dict[str, int] = {}
            for feed in config.feeds:
                categories[feed.category or "(none)"] = categories.get(feed.category or "(none)", 0) + 1
            for category, count in sorted(categories.items()):
                print(f"{count:4}  {category}")
            print(f"{len(config.feeds):4}  total")
            return 0
        now = time.time()
        shown = 0
        for feed in sorted(config.feeds, key=lambda feed: (feed.category, feed.name or feed.url)):
            state = states.get(feed.url)
            source = feed.source(state["title"] if state else "")
            if state is None:
                health = "never fetched"
            elif state["failures"]:
                health = f"FAILING x{state['failures']}: {state['last_error'][:60]}"
            else:
                health = f"ok {int((now - state['last_success']) / 60)}m ago"
            if args.broken and (state is None or not state["failures"]):
                continue
            votes = statistics.get(source, SourceStatistics())
            print(f"{feed.category[:18]:18}  {source[:34]:34}  {counts.get(source, 0):4}/wk  "
                  f"+{votes.likes}/-{votes.dislikes}  {health}")
            if args.broken:
                print(f"{'':18}  {feed.url}")
            shown += 1
        print(f"-- {shown} feeds shown, {len(config.feeds)} configured")
        return 0


def _command_digest(config: Config, args: argparse.Namespace):
    from .daemon import send_digest
    with Database(config.database) as database:
        message = send_digest(config, database, dry_run=args.dry_run)
        if message is None:
            print("no candidate articles; run `indefinite-articles fetch` first", file=sys.stderr)
            return 1
        if args.dry_run:
            print(mailer.part_text(message))
        else:
            print(f"sent: {message['Subject']}")
        return 0


def _command_poll(config: Config, args: argparse.Namespace):
    with Database(config.database) as database:
        llm = LLM(config.llm)
        print(f"processed {inbox.poll(config, database, llm)} replies")
        if ranker.update_profile(config, database, llm):
            print("profile updated")
        return 0


def _command_feedback(config: Config, args: argparse.Namespace):
    """Process a reply saved as an .eml file (or stdin) without IMAP."""
    with Database(config.database) as database:
        llm = LLM(config.llm)
        raw = open(args.file, "rb").read() if args.file != "-" else sys.stdin.buffer.read()
        message = email.message_from_bytes(raw, policy=email.policy.default)
        import hashlib
        # Same exactly-once bookkeeping as IMAP polling, so a later poll won't count it again.
        reply_id = inbox.message_id(message, "file-" + hashlib.sha256(raw).hexdigest()[:16])
        result = inbox.handle_message(config, database, llm, message, reply_id)
        print(result.status)
        for line in result.summary:
            print(" ", line)
        if ranker.update_profile(config, database, llm):
            print("profile updated")
        return 0 if result.status == "ok" else 1


def _command_profile(config: Config, args: argparse.Namespace):
    with Database(config.database) as database:
        if args.set:
            database.add_profile(args.set, note="set manually")
            print("profile replaced")
            return 0
        if args.history:
            for row in database.connection.execute("SELECT * FROM profiles ORDER BY id"):
                print(f"--- version {row['id']} ({row['note']})")
                print(row["text"], "\n")
            return 0
        row = ranker.ensure_profile(config, database)
        print(row["text"])
        return 0


def _command_extract(config: Config, args: argparse.Namespace):
    """Show what the LLM would see for a URL."""
    print(extract.fetch_page(args.url, config.article_chars, config.allow_private_addresses))
    return 0


def _command_status(config: Config, args: argparse.Namespace):
    with Database(config.database) as database:
        for key, value in database.statistics().items():
            print(f"{key:18} {value}")
        print(f"{'last_digest':18} {database.get_state('last_digest_date', 'never')}")
        return 0


def _command_check(config: Config, args: argparse.Namespace):
    ok = True
    for problem in config.problems():
        print(f"[warn] config: {problem}")
    try:
        print(f"[ok]   LLM {config.llm.model} at {config.llm.base_url}: {LLM(config.llm).ping().strip()[:40]!r}")
    except LLMError as error:
        ok = False
        print(f"[FAIL] LLM: {error}")
    if config.llm.embedding_model:
        try:
            dimension = len(LLM(config.llm).embed(["test"])[0])
            print(f"[ok]   embeddings {config.llm.embedding_model}: {dimension} dimensions")
        except LLMError as error:
            print(f"[warn] embeddings {config.llm.embedding_model}: {error}\n"
                  f"       (without it only the newest articles are considered; "
                  f"try: ollama pull {config.llm.embedding_model})")
    print(f"[info] {len(config.feeds)} feeds configured")
    for feed in config.feeds[: args.feeds]:
        try:
            parsed = ParsedFeed.parse(feeds.fetch(feed.url))
            print(f"[ok]   feed {feed.source(parsed.title)}: {len(parsed.entries)} entries")
        except Exception as error:
            ok = False
            print(f"[FAIL] feed {feed.url}: {error}")
    if config.email.imap.host:
        try:
            connection = inbox.connect(config)
            status, _ = connection.select(config.email.imap_folder, readonly=True)
            connection.logout()
            print(f"[{'ok' if status == 'OK' else 'FAIL'}]   IMAP {config.email.imap.host}"
                  f" folder {config.email.imap_folder}")
            ok &= status == "OK"
        except Exception as error:
            ok = False
            print(f"[FAIL] IMAP: {error}")
    if args.send_test:
        from email.message import EmailMessage
        test_message = EmailMessage()
        test_message["Subject"] = "indefinite-articles: test email"
        test_message["From"], test_message["To"] = config.email.sender, config.email.to
        test_message.set_content("If you can read this, SMTP is configured correctly.")
        try:
            mailer.send(config, test_message)
            print(f"[ok]   SMTP test email sent to {config.email.to}")
        except Exception as error:
            ok = False
            print(f"[FAIL] SMTP: {error}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        prog="indefinite-articles",
        description="Daily article recommendations by email, trained on your replies with a local LLM.",
    )
    arg_parser.add_argument("-c", "--config", help=f"config file (default {DEFAULT_CONFIG_PATH})")
    arg_parser.add_argument("-v", "--verbose", action="store_true")
    arg_parser.add_argument("--version", action="version", version=f"indefinite-articles {__version__}")
    subparsers = arg_parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("run", help="run the service (used by systemd)").set_defaults(handler=_command_run)
    parser = subparsers.add_parser("feeds", help="list feeds and their health")
    parser.add_argument("--broken", action="store_true", help="only failing feeds")
    parser.add_argument("--categories", action="store_true", help="count feeds per category")
    parser.set_defaults(handler=_command_feeds)
    parser = subparsers.add_parser("fetch", help="fetch feeds now and score new articles")
    parser.add_argument("--no-score", action="store_true")
    parser.set_defaults(handler=_command_fetch)
    parser = subparsers.add_parser("digest", help="build and send a digest now")
    parser.add_argument("-n", "--dry-run", action="store_true", help="print instead of sending")
    parser.set_defaults(handler=_command_digest)
    subparsers.add_parser("poll", help="read replies from IMAP now").set_defaults(handler=_command_poll)
    parser = subparsers.add_parser("feedback", help="process a reply from an .eml file (- for stdin)")
    parser.add_argument("file")
    parser.set_defaults(handler=_command_feedback)
    parser = subparsers.add_parser("profile", help="show the learned reader profile")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--set", metavar="TEXT", help="replace the profile")
    parser.set_defaults(handler=_command_profile)
    subparsers.add_parser("status", help="show database statistics").set_defaults(handler=_command_status)
    parser = subparsers.add_parser("extract", help="show the article text extracted from a URL")
    parser.add_argument("url")
    parser.set_defaults(handler=_command_extract)
    parser = subparsers.add_parser("check", help="test LLM, feeds, IMAP and (optionally) SMTP")
    parser.add_argument("--send-test", action="store_true", help="also send a test email")
    parser.add_argument("--feeds", type=int, default=3, help="number of feeds to test")
    parser.set_defaults(handler=_command_check)

    args = arg_parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as error:
        print(f"indefinite-articles: {error}", file=sys.stderr)
        return 2
    try:
        return args.handler(config, args)
    except KeyboardInterrupt:
        return 130
