"""Long-running service: fetch feeds, poll replies, send the daily digest."""

import logging
import signal
import threading
import time
from collections.abc import Callable
from datetime import datetime
from email.message import EmailMessage

from . import __version__, embed, extract, feeds, inbox, mailer, ranker
from .config import Config
from .database import Database, PendingFeedback
from .digest import Digest
from .llm import LLM, LLMError

log = logging.getLogger(__name__)


def prefilter(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> None:
    """Embed new articles and update pre-filter scores (skipped if no embedding model)."""
    if not config.llm.embedding_model:
        return
    try:
        embed.embed_new(config, database, llm, stop=stop)
        embed.update_prefilter(config, database, llm, ranker.ensure_profile(config, database))
    except LLMError as error:
        log.warning("embedding pre-filter unavailable, using newest articles instead: %s", error)


def _gather(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> int:
    """Fetch feeds, pre-filter, and fetch full text for the shortlist."""
    new = feeds.update_all(config, database, stop=stop)
    prefilter(config, database, llm, stop)
    extract.enrich_candidates(config, database, stop=stop)
    return new


def refresh(config: Config, database: Database, llm: LLM, stop: threading.Event | None = None) -> int:
    """gather(), then LLM-score the shortlist."""
    new = _gather(config, database, llm, stop)
    try:
        ranker.score_candidates(config, database, llm, stop=stop)
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
    unscored = sum(article["score"] is None for article in short)
    stale = sum(article["score"] is not None and article["score_profile_id"] != profile_id for article in short)
    no_text = sum(article["content_status"] is None for article in short) if config.fetch_full_articles else 0
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
    from .llm import MIN_STALL, STALL_FACTOR
    with llm._lock:
        slowest = dict(llm._slowest)
    parts = [f"{kind} {_duration(max(MIN_STALL, STALL_FACTOR * seconds))} (slowest {seconds:.1f}s)"
             for kind, seconds in sorted(slowest.items())]
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
        f"articles: {new_24h} new in 24h, {statistics['articles']} in db,"
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
        "#   rank  final =   llm + adj  + src   text      source",
    ]
    scores, rank = selection.get("scores", {}), selection.get("rank", {})
    for position, pick in enumerate(items, start=1):
        article = pick.article
        score = scores.get(article["id"])
        if score is None:
            continue
        llm_score = f"{score.llm:5.1f}" if score.llm is not None else "    -"
        lines.append(
            f"{position:<3} {rank.get(article['id'], 0):>4}  {score.final:5.2f} = {llm_score} {score.base:+5.2f}"
            f" {score.bonus:+5.2f}   {(article['content_status'] or '-')[:8]:8}  {article['source'][:30]}"
            + ("  [explore]" if pick.exploratory else "")
        )
    lines.append("(adj: embedding agreement, or fallback rank when the LLM did not score;"
                 " src: source like/dislike bonus)")
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
    database.record_digest(digest.token, message["Message-ID"], digest.day.isoformat(), digest.items)
    return message


class Service:
    """Runs each job on its own thread, so a slow LLM never holds up the others.

    - fetch:  every fetch_interval_minutes, fetch feeds, pre-filter, download text
    - score:  LLM-scores the shortlist whenever fetch or a profile change asks for it
    - inbox:  every poll_interval_minutes, read replies and rewrite the profile
    - main:   sends the digest at digest_time (no LLM) and prunes old data

    Each thread has its own database connection (SQLite in WAL mode).
    """

    TICK: float = 20  # seconds

    def __init__(self, config: Config):
        self.config = config
        self.database = Database(config.database)
        self.llm = LLM(config.llm, deadline=config.next_digest_time)
        self.stop = threading.Event()
        self.score_wanted = threading.Event()
        self.last_prune = 0.0
        self.next_digest_attempt = 0.0
        # What each worker thread is doing, for the digest's debug report. Each entry is
        # replaced whole, so the main thread never sees one half-updated.
        self.jobs: dict[str, dict] = {name: {} for name in ("feed fetch", "scoring", "inbox poll")}

    def _digest_due(self) -> bool:
        now = datetime.now()
        if self.database.get_state("last_digest_date") == now.date().isoformat():
            return False
        return self.config.digest_hour_minute.has_passed(now)

    def _run_job(self, name: str, function: Callable[..., object], *args: object):
        tracked = name in self.jobs  # the worker jobs; never adds keys from another thread
        started = time.time()
        if tracked:
            self.jobs[name] = {**self.jobs[name], "running_since": started}
        result, error = None, None
        try:
            result = function(*args)
        except Exception as failure:
            log.exception("%s failed", name)
            error = str(failure)[:150] or type(failure).__name__
        if tracked:
            end = time.time()
            self.jobs[name] = {"running_since": None, "last_end": end, "last_seconds": end - started,
                               "last_result": result, "last_error": error}

    # -- jobs (each gets the database connection of the thread it runs on) ---------
    def _fetch(self, database: Database) -> str:
        new = _gather(self.config, database, self.llm, stop=self.stop)
        self.score_wanted.set()
        return f"{new} new articles"

    def _score(self, database: Database) -> str:
        try:
            return f"scored {ranker.score_candidates(self.config, database, self.llm, stop=self.stop)}"
        except LLMError as error:
            log.warning("scoring skipped: %s", error)
            return f"skipped: {error}"

    def _poll_and_learn(self, database: Database) -> str:
        used = inbox.poll(self.config, database, self.llm)
        if used:
            log.info("processed %d replies", used)
        # Also retries pending feedback if an earlier update failed.
        if ranker.update_profile(self.config, database, self.llm):
            self.score_wanted.set()  # rescore the shortlist against the new profile
            return f"{used} replies, profile updated"
        return f"{used} replies"

    def _digest(self):
        today = datetime.now().date().isoformat()
        message = send_digest(self.config, self.database,
                          status={"jobs": {key: dict(value) for key, value in self.jobs.items()}, "llm": self.llm})
        # Mark the day done even if there was nothing to send, so we don't retry every tick.
        self.database.set_state("last_digest_date", today)
        if message is not None:
            log.info("digest sent: %s", message["Subject"])

    # -- threads -----------------------------------------------------------------
    def _with_database(self, loop: Callable[..., None], *args: object):
        database = Database(self.config.database)
        try:
            loop(database, *args)
        finally:
            database.close()

    def _every(self, database: Database, name: str, minutes: int, job: Callable[[Database], str]):
        """Run `job` now and then every `minutes`, measured from the start of each run."""
        while not self.stop.is_set():
            started = time.monotonic()
            self._run_job(name, job, database)
            self.stop.wait(max(0.0, minutes * 60 - (time.monotonic() - started)))

    def _on_demand(self, database: Database, name: str, job: Callable[[Database], str]):
        """Run `job` whenever score_wanted is set (also once at start)."""
        self.score_wanted.set()
        while True:
            self.score_wanted.wait()
            if self.stop.is_set():
                return
            # Clear first: a request made while the job runs triggers another run.
            self.score_wanted.clear()
            self._run_job(name, job, database)

    def _shutdown(self, *_: object):
        self.stop.set()
        self.score_wanted.set()  # wake the scoring thread so it can see the stop

    def run(self) -> None:
        for signature in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signature, self._shutdown)
        ranker.ensure_profile(self.config, self.database)
        log.info("indefinite-articles service started: %d feeds, digest at %s, model %s at %s",
                 len(self.config.feeds), self.config.digest_time, self.config.llm.model, self.config.llm.base_url)
        # Daemon threads: one stuck in a slow LLM request must not stop the service exiting.
        threads = [
            threading.Thread(target=self._with_database, name="fetch", daemon=True,
                             args=(self._every, "feed fetch", self.config.fetch_interval_minutes, self._fetch)),
            threading.Thread(target=self._with_database, name="score", daemon=True,
                             args=(self._on_demand, "scoring", self._score)),
            threading.Thread(target=self._with_database, name="inbox", daemon=True,
                             args=(self._every, "inbox poll", self.config.poll_interval_minutes,
                                   self._poll_and_learn)),
        ]
        for thread in threads:
            thread.start()
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= self.next_digest_attempt and self._digest_due():
                # If sending fails (e.g. SMTP down) retry in 15 minutes, not every tick.
                self.next_digest_attempt = now + 900
                self._run_job("digest", self._digest)
            if now - self.last_prune >= 86400:
                self.last_prune = now
                self._run_job("prune", self.database.prune)
            self.stop.wait(self.TICK)
        log.info("indefinite-articles service stopping")
        for thread in threads:
            thread.join(timeout=5)
        self.database.close()
