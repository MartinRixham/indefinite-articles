"""Pipeline steps the service and the CLI run: rank the articles and send the daily digest."""

import logging
import threading
import time
from datetime import datetime
from email.message import EmailMessage

from . import __version__
from .config import Config
from .fetching import apple_podcasts, extract, feeds
from .language_model.llm import LLM, LLMError, stall_limit
from .mail import mailer
from .mail.digest import Digest
from .ranking import embed, ranker
from .storage.database import Database, PendingFeedback

log = logging.getLogger(__name__)


def prefilter(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> None:
    """Embed new articles and update pre-filter scores (skipped if no embedding model).

    Raises LLMError if embedding fails. The caller should then stop and try again
    later rather than go on to scoring: scoring a shortlist picked without the new
    articles' embeddings wastes the LLM, and on a slow server it holds up the
    embeddings for hours.
    """
    if not config.llm.embedding_model:
        return
    try:
        embed.embed_new(config, database, llm, stop=stop)
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
    except LLMError as error:
        raise type(error)(f"embedding failed: {error}") from error


def rank(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> int:
    """Embed and pre-filter, fetch full text (and Apple Podcasts links) for the shortlist,
    then LLM-score it; returns the number of articles scored.

    The steps run in this order so that scoring always works from a finished pre-filter
    (after a change of embedding model, that means every article embedded again).
    Raises LLMError if embedding or scoring can't run; if embedding fails, nothing is
    scored.
    """
    prefilter(config, database, llm, stop)
    extract.enrich_candidates(config, database, stop=stop)
    apple_podcasts.add_links(config, database, stop=stop)
    return ranker.score_candidates(config, database, llm, stop=stop)


def refresh(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> int:
    """Fetch feeds, then rank(); returns the number of new articles."""
    new = feeds.update_all(config, database, stop=stop)
    try:
        rank(config, database, llm, stop)
    except LLMError as error:
        log.warning("scoring skipped: %s", error)
    return new


def _ago(timestamp: float | None) -> str:
    if not timestamp:
        return "never"
    minutes = int((time.time() - timestamp) / 60)
    return f"{minutes}m ago" if minutes < 120 else f"{minutes // 60}h ago"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 7200:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _backlog(config: Config, database: Database) -> str:
    """Work still waiting at publish time."""
    profile = database.current_profile()
    profile_id = profile["id"] if profile else None
    short = ranker.shortlist(config, database)
    unscored = sum(article.score is None for article in short)
    stale = sum(article.has_stale_score(profile_id) for article in short)
    no_text = sum(article.awaiting_full_text for article in short) if config.fetch_full_articles else 0
    unembedded = len(database.unembedded(config.max_article_age_days, 50000)) if config.llm.embedding_model else 0
    return (f"backlog: shortlist {len(short)}, {unscored} never scored, {stale} scored against"
            f" an older profile, {no_text} awaiting full text; {unembedded} articles not embedded")


def _job_status(name: str, job: dict) -> str:
    now = time.time()
    if job.get("running_since"):
        state = f"running for {_duration(now - job['running_since'])}"
    else:
        state = "idle"
    if job.get("last_end"):
        state += f", last run finished {_ago(job['last_end'])} after {_duration(job['last_seconds'])}"
        if job.get("last_error"):
            state += f": FAILED {job['last_error']}"
        elif job.get("last_result") is not None:
            state += f": {job['last_result']}"
    elif not job.get("running_since"):
        state += ", not run yet"
    return f"  {name}: {state}"


def _llm_limits(llm: LLM) -> str:
    parts = [f"{kind} {_duration(stall_limit(seconds))} (slowest {seconds:.1f}s)"
             for kind, seconds in sorted(llm.slowest_waits().items())]
    return "llm stall limits: " + (", ".join(parts) if parts else "none learned yet")


def _classifier_status(database: Database) -> str:
    likes, _, dislikes = database.get_state("classifier_info", "0:0").partition(":")
    if database.get_state("classifier", ""):
        return f"classifier: trained on {likes} likes / {dislikes} dislikes"
    return (f"classifier: not trained yet ({likes} likes / {dislikes} dislikes with embeddings;"
            f" needs {embed.MIN_PER_CLASS} of each)")


def _debug_report(config: Config, database: Database, items: list[ranker.DigestPick], info: dict) -> list[str]:
    """Plain-text diagnostics appended to the digest: pipeline health and why each item was picked."""
    selection = info.get("select", {})
    statistics = database.statistics()
    profile = database.current_profile()
    pending = database.feedback_since(profile["created_at"]) if profile else PendingFeedback([], [])
    states = database.feed_states()
    never = sum(feed.url not in states for feed in config.feeds)
    last_ok = max((state["last_success"] or 0 for state in states.values()), default=0)
    new_24h = database.connection.execute("SELECT COUNT(*) FROM articles WHERE fetched_at >= ?",
                              (time.time() - 86400,)).fetchone()[0]
    timings = info.get("timings", {})
    lines = [
        f"version {__version__}, built {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"llm: {config.llm.model} at {config.llm.base_url}",
        f"embeddings: {config.llm.embedding_model or 'disabled'}, {statistics['embedded']} stored",
        f"feeds: {len(config.feeds)} configured, {statistics['feeds_failing']} failing, {never} never fetched,"
        f" last success {_ago(last_ok)}",
        f"articles: {new_24h} new in 24h, {statistics['articles']} in db"
        f" ({statistics['podcast_episodes']} podcast episodes),"
        f" full text {statistics['full_text']} ok / {statistics['full_text_failed']} failed",
        f"selection: {selection.get('candidates', 0)} candidates, {selection.get('llm_scored', 0)} scored"
        f" against current profile, {selection.get('stale_scored', 0)} with stale scores",
        _backlog(config, database),
        f"profile: v{profile['id'] if profile else '-'} ({profile['note'] if profile else ''}),"
        f" {len(pending.feedback)} ratings + {len(pending.comments)} comments pending",
        f"feedback total: +{statistics['likes']} / -{statistics['dislikes']}, {statistics['digests']} digests sent",
        _classifier_status(database),
        "timings: " + ", ".join(f"{key} {seconds:.1f}s" for key, seconds in timings.items()),
    ]
    jobs = info.get("jobs")
    if jobs is not None:
        lines.append("jobs at publish time:")
        lines += [_job_status(name, job) for name, job in jobs.items()]
    if info.get("llm") is not None:
        lines.append(_llm_limits(info["llm"]))
    lines += [
        "",
        "#   rank  final =   llm + adj  + src    ovl   text      source",
    ]
    scores, rank = selection.get("scores", {}), selection.get("rank", {})
    overlap = selection.get("overlap", {})
    for position, pick in enumerate(items, start=1):
        score = scores.get(pick.article.id)
        if score is not None:
            lines.append(pick.debug_line(position, score, rank.get(pick.article.id, 0),
                                         overlap.get(pick.article.id, 0.0)))
    lines.append("(adj: embedding agreement, or fallback rank when the LLM did not score;"
                 " src: source like/dislike bonus; ovl: taken off for resembling an earlier pick"
                 " or sharing its source)")
    return lines


def send_digest(config: Config, database: Database, dry_run: bool = False,
                status: dict | None = None) -> EmailMessage | None:
    """Select and send today's digest from the scores the hourly runs left behind.

    It never calls the LLM, so the email goes out on time however slow the LLM is.
    Articles not scored yet stay candidates for later digests. Returns the
    EmailMessage or None. `status` (from the service) adds the state of its jobs and
    LLM client to the debug report.
    """
    note = ""
    info: dict = {"timings": {}, "select": {}, **(status or {})}
    started = time.monotonic()
    items = ranker.select(config, database, debug=info["select"])
    info["timings"]["select"] = time.monotonic() - started
    selection = info["select"]
    if items and not selection.get("llm_scored") and not selection.get("stale_scored"):
        note = "(The local LLM has not scored any of these yet, so today's list was ranked by simple heuristics.)"
    if not items:
        log.warning("no candidate articles for today's digest")
        return None
    debug = None
    if config.email.debug_info:
        try:
            debug = _debug_report(config, database, items, info)
        except Exception:  # diagnostics must never stop the digest going out
            log.exception("building debug report failed")
    digest = Digest(items, note, debug)
    message = digest.message(config)
    if dry_run:
        return message
    mailer.send(config, message)
    database.record_digest(digest.token, message["Message-ID"], digest.items)
    return message
