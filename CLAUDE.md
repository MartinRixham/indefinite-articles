# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Fedora service (Python 3.14 on Fedora 44, 3.15 on Fedora 45; stdlib + numpy only) that fetches ~500 RSS/Atom feeds, ranks the articles with a local LLM (Ollama or llama.cpp, via the OpenAI-compatible `/v1` API), emails a daily digest, and learns from the user's email replies. See README.md for the user-facing description and config.

## Commands

```sh
make test                                                     # all tests (what CI runs)
PYTHONPATH=src:tests python3 -m unittest test_indefinite_articles.EndToEndTests.test_digest_and_feedback_cycle   # one test
PYTHONPATH=src python3 -m indefinite_articles -c dist/indefinite-articles.toml <command>   # run the CLI from source
npx pyright src tests                                         # type check; keep it at 0 errors
make container-rpm                                            # build the RPM in a Fedora container (podman)
```

There is no virtualenv: numpy comes from the system Python. Tests need no network or LLM. `tests/fake_llm.py` holds `FakeLLM`, an in-process HTTP server (started by `test_indefinite_articles.py` and `language_model/test_assistant.py`) that answers chat, embedding and `/api/ps` requests based on the prompt text. `FakeLLM.do_POST` recognizes each request by a marker string in its prompt: `Article to rate:` (scoring), `interpret a reader` (reply parsing), `maintain a concise profile` (profile update) and `short phrases for a search engine` (profile phrases); all four prompts live in `language_model/assistant.py`. If you reword one of those prompts, update the fake too. Setting `FakeLLM.answer` makes it give that reply to every chat request; `tests/language_model/test_assistant.py` uses it to test unusable answers. The tests load the shipped config `dist/indefinite-articles.toml` (with `:memory:` as the database) on top of `dist/tuning.toml`, so both files must stay valid.

## Architecture

Layout of `src/indefinite_articles/`: `cli`, `config`, `daemon` (the pipeline steps the CLI and the service share) and `service` sit at the top; the rest is in subpackages: `fetching/` (feeds, entries, full-text extraction), `ranking/` (embeddings, classifier, LLM scoring, profile), `language_model/` (the `LLM` client and `Assistant`), `mail/` (digest and its templates, SMTP, IMAP, reply parsing) and `storage/` (`Database`, `Article`). `tests/` mirrors this layout (each test subdirectory has an `__init__.py` so `unittest discover` finds it); `tests/fake_llm.py` stays at the top and is imported as `fake_llm`.

Pipeline:

1. **Fetch.** `feeds.update_all` fetches feeds with conditional GET (ETag/Last-Modified) and exponential backoff for failing feeds, then stores new articles. Articles are deduplicated by `title_key`.
2. **Pre-filter.** `ranking/embed.py` embeds every new article and scores it by cosine similarity against phrases from the reader profile and against recent likes and dislikes. Once there are `MIN_PER_CLASS` likes and dislikes, a numpy logistic-regression classifier (`classifier.Classifier`, trained in `embed.py`) is blended in. The result is stored in `articles.prefilter`.
3. **Extract.** `extract.enrich_candidates` downloads the full text for the shortlist only. It refuses private addresses unless the config allows them.
4. **Score.** `ranker.score_candidates` LLM-scores each shortlisted article in its own request (`Assistant.score_article`). The prompt starts with an identical prefix (`Assistant.score_prefix`: profile + examples) so the LLM server can reuse its KV cache. Scores are tied to `score_profile_id`, so a profile change triggers rescoring.
5. **Digest.** `ranker.select`, then `digest.Digest`, which fills in `mail/templates/digest.txt`, `mail/templates/digest.html` and `mail/templates/instructions.txt` (package data, listed in `pyproject.toml`). Template files are made of `@@ name` sections with `string.Template` `$placeholders`; a line whose placeholders are all empty is left out. The digest **never calls the LLM**. If nothing has been scored, it falls back to heuristics (source ratings and recency) and the email says so.
6. **Replies.** `inbox.poll` reads replies over IMAP. `replies._parse_structured` handles `+1 -2` style ratings; lines it can't parse go to the LLM (`Assistant.interpret_reply`). `ranker.update_profile` rewrites the reader profile (`Assistant.rewrite_profile`). Each reply is processed exactly once: its feedback and the "processed" marker are committed in the same transaction.

`service.Service` runs fetch, score and inbox polling on separate daemon threads, so a slow LLM can't delay the digest. The main thread sends the digest and prunes old data. **Each thread has its own `Database` connection** (SQLite in WAL mode), so never share a `Database` across threads. `score_wanted` (a `threading.Event`) wakes the scoring thread after a fetch or a profile update.

`llm.LLM` streams responses and learns the slowest normal response time for each request kind, then cuts off a stream that stalls (`STALL_FACTOR`, `MIN_STALL`). Requests are also cut off when the next digest is due (`deadline`). Raise `LLMUnavailable`/`LLMError` and let callers degrade gracefully. The service refuses to start if Ollama is running the model on the CPU only (`require_gpu`); in that case `cli` exits with code 78, which the systemd unit treats as "don't restart".

`assistant.Assistant` wraps an `LLM` and holds every chat prompt. Its methods (`score_article`, `rewrite_profile`, `interpret_reply`, `profile_phrases`) return domain values and raise `LLMError` when the answer can't be used; callers keep their fallbacks. The pipeline functions still take an `LLM` and build an `Assistant` from it, and embeddings go straight to `LLM.embed`.

Schema changes: `database.SCHEMA` uses `CREATE TABLE IF NOT EXISTS`. New `articles` columns must also be added to `Database._migrate` (an `ALTER TABLE ADD COLUMN` list), and `test_migration_from_old_schema` (in `tests/fetching/test_extract.py`) covers this.

## Packaging

- `dist/` holds the files installed to `/etc` and systemd: the config (`indefinite-articles.toml` for the user's email and preferences; `tuning.toml` for everything else, loaded first by `config.load` and overridden by the main file, so users who leave it unedited get tuning updates on upgrade), `feeds.d/default.opml`, the unit file, sysusers and the Ollama iGPU drop-in.
- `packaging/indefinite-articles.spec` builds the RPM. The `Containerfile` installs that RPM, so container paths match the RPM's.
- The version lives in `src/indefinite_articles/__init__.py` (`__version__`); `pyproject.toml` (via `tool.setuptools.dynamic`) and the Makefile read it from there. The spec file's `Version:` and its newest `%changelog` entry must match it; `tests/test_version.py` checks this, so a release means editing `__init__.py` and the spec (`Version:` plus a new changelog entry).
- CI (`.github/workflows/build.yml`) runs the tests, then builds the RPM on Fedora 44 and 45, then publishes the container image to ghcr.

## Code style

No single-character identifiers and no abbreviations, anywhere in the code or tests. This covers variables, parameters, attributes, functions, classes, modules, constants, loop and comprehension variables, lambda arguments, `except ... as` names and format placeholders. Write `config`, not `cfg`; likewise `database`, `message`, `connection`, `error`, `vectors`, `article`, `row`, `match`, `key`/`value`, `weights`/`bias`, `like_count` (not `n_like`) and `seconds` (not `secs`).

Exceptions:
- `_` for a value that is deliberately ignored.
- Python conventions: `self`, `cls`, `args`, `kwargs`, `argv`, and `np` for numpy.
- Acronyms that are the normal name for the thing: `llm`, `url`, `html`, `json`, `sql`, `imap`, `smtp`, `ssl`, `ip`, `id`/`ids`, `uid`, `etag`, `guid`, `opml`, `rss`, `L2`.
- Names fixed from outside: parameters of overridden library methods (e.g. `attrs` in `handle_starttag`), stdlib and third-party attributes and keyword arguments, TOML config keys (e.g. `max_prompt_chars`, `article_chars`), CLI flags and database column names.

Every function parameter has a type annotation, in `src/` and `tests/` alike. This includes `*args`/`**kwargs`, overridden library methods (copy the typeshed signature) and nested helper functions. Only `self`, `cls` and lambda parameters are exempt, because Python has no syntax for annotating a lambda parameter. Every public function and method (any name not starting with `_`, plus dunder methods such as `__eq__`) declares a return type, in `src/` and `tests/` alike; that includes `-> None` on test methods, `setUp` and overridden library methods. The exception is `__init__`, which doesn't need `-> None`. Use `Config`, `Database`, `threading.Event | None` for `stop`, `EmailMessage` for messages, and `article.Article` for a stored article (the `Database` methods that return articles build it with `Article.from_row`; tests construct it directly).

Commit messages are short, sentence-style summaries ending with a period (e.g. "Feed fetch, scoring and digest on separate threads.").
