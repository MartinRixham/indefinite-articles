# Indefinite Articles

A small Fedora service that emails you a daily list of articles and podcast episodes. It
picks them with a **local LLM** and learns what you like from **your replies to the email**.

How it works:

1. Every hour it fetches about 500 RSS/Atom feeds across 31 categories, plus any you add,
   and saves new articles to SQLite. That's typically a few thousand articles a day.
2. A small embedding model gives every article a quick first-pass rank against your
   reader profile and the articles you liked and disliked. Once you have rated at least
   5 of each, a small classifier trained on every rating you have given (newer ones count
   more) does the like/dislike part. Only the best ~150 (plus some random picks, so it
   keeps learning) go on to the next steps.
3. For those, it downloads the page and extracts the article text. A local LLM
   (Ollama or llama.cpp) then reads each one and scores it from 0 to 10, using your
   *reader profile* plus your latest likes and dislikes.
4. At `digest_time` it emails you the top articles. Two "something different" picks are
   added so it keeps finding out what else you might like. Below them come the top
   podcast episodes (`podcasts_per_digest`, default 3), numbered on from the articles so
   you rate them the same way.
5. Every 5 minutes it checks the mailbox over IMAP for your replies. It records your likes,
   dislikes and comments, and the LLM rewrites your reader profile. It can reply
   to confirm what it understood.

It targets Fedora 44 (Python 3.14) and Fedora 45 (Python 3.15), and needs only Python, numpy and a local LLM server.

## Build the RPM

```sh
make container-rpm        # builds in a clean Fedora container with podman, no root needed
# or, on a Fedora machine with build tools installed:
sudo dnf install rpm-build python3-devel pyproject-rpm-macros systemd-rpm-macros python3-numpy
make rpm
```

The packages are written to `build/rpmbuild/RPMS/noarch/` and `build/rpmbuild/SRPMS/`.

## Install and set up

```sh
sudo dnf install ./indefinite-articles-*.noarch.rpm

# 1. A local LLM (Ollama is in the Fedora repos)
sudo dnf install ollama ollama-rocm     # ollama-rocm: AMD GPUs, including integrated Radeon
sudo systemctl enable --now ollama
ollama pull qwen2.5:14b                 # ~9 GB; or qwen2.5:32b, gemma2:27b, llama3.1:8b ...
ollama pull nomic-embed-text            # small (274 MB) model for the first-pass ranking
                                        # (the service also downloads missing models
                                        #  when it starts: pull_models in tuning.toml)

# 2. Configure email and your starting interests
sudoedit /etc/indefinite-articles/indefinite-articles.toml

# 3. Test everything, then start the service
sudo -u indefinite-articles indefinite-articles check --send-test
sudo systemctl enable --now indefinite-articles
journalctl -u indefinite-articles -f
```

Tuning settings (ranking, fetching, the LLM model and endpoint) are in
`/etc/indefinite-articles/tuning.toml`. Leave that file unedited and each upgrade brings the
latest tuning. To change one of its settings, copy it into `indefinite-articles.toml` under
the same `[section]` header; settings there override `tuning.toml`. (If you edit
`tuning.toml` itself, RPM keeps your copy and saves upgrades as `tuning.toml.rpmnew`.)

Use a dedicated mailbox (e.g. `indefinite-articles@yourdomain`) as `from`. It sends the digests and
receives your replies. For Gmail or Fastmail, use an app password. Only replies **from**
`email.to` (or `allowed_senders`) are read. Other mail in the mailbox is left alone.

## Run in a container

Each push to `main` publishes `ghcr.io/martinrixham/indefinite-articles:edge`. Each `v*`
tag publishes `:latest`, `:X.Y.Z` and `:X.Y`. Images are built for amd64 and arm64. The
image installs the same RPM, so the paths match those above. It runs as the
unprivileged `indefinite-articles` user (UID 999).

```sh
# Config: start from the packaged one. It must be readable by UID 999 inside the
# container, so keep passwords out of it and use podman secrets instead.
podman run --rm --entrypoint cat ghcr.io/martinrixham/indefinite-articles \
  /etc/indefinite-articles/indefinite-articles.toml > indefinite-articles.toml
chmod 644 indefinite-articles.toml
#   set password_file = "/run/secrets/smtp-password" under [email.smtp]
#   and password_file = "/run/secrets/imap-password" under [email.imap]
printf '%s' 'app-password' | podman secret create smtp-password -
printf '%s' 'app-password' | podman secret create imap-password -

podman run -d --name indefinite-articles \
  --network=host \
  -e TZ=Europe/London \
  -v indefinite-articles:/var/lib/indefinite-articles \
  -v ./indefinite-articles.toml:/etc/indefinite-articles/indefinite-articles.toml:ro,Z \
  --secret smtp-password,uid=999,mode=0400 \
  --secret imap-password,uid=999,mode=0400 \
  ghcr.io/martinrixham/indefinite-articles

podman exec indefinite-articles indefinite-articles check --send-test
podman logs -f indefinite-articles
```

* Only `indefinite-articles.toml` is mounted, so `tuning.toml` comes from the image and
  is updated with it. Put overrides (such as `[llm] base_url`) in your
  `indefinite-articles.toml`.
* `--network=host` lets the default `base_url` (`http://127.0.0.1:11434/v1`) reach Ollama
  on the host. Without it, point `base_url` at `http://host.containers.internal:11434/v1`
  and make Ollama listen on more than localhost (`OLLAMA_HOST=0.0.0.0`).
* `digest_time` uses the container's time zone, so set `TZ`.
* The container doesn't need the GPU, but the host's Ollama must be using it, or the
  service exits at startup (`require_gpu`). For integrated Radeon GPUs, install
  `ollama-rocm` on the host and add `Environment=OLLAMA_IGPU_ENABLE=1` with
  `sudo systemctl edit ollama`. Add `Environment=OLLAMA_CONTEXT_LENGTH=8192` there too, so
  scoring prompts fit (see "Context size" below). The RPM does both for you; the image can't.
* The other commands work the same way: `podman exec indefinite-articles indefinite-articles status`.

To run it under systemd, save this as
`~/.config/containers/systemd/indefinite-articles.container` (Quadlet), then run
`systemctl --user daemon-reload && systemctl --user start indefinite-articles`.
Run `loginctl enable-linger` once so it keeps running after you log out.

```ini
[Container]
Image=ghcr.io/martinrixham/indefinite-articles:latest
Network=host
Environment=TZ=Europe/London
Volume=indefinite-articles:/var/lib/indefinite-articles
Volume=%h/indefinite-articles.toml:/etc/indefinite-articles/indefinite-articles.toml:ro,Z
Secret=smtp-password,uid=999,mode=0400
Secret=imap-password,uid=999,mode=0400
AutoUpdate=registry

[Service]
Restart=on-failure
# Wait for Ollama to come up instead of giving up after 5 quick retries.
RestartSec=30s
# 78: the model is running on the CPU only (no usable GPU); don't keep retrying.
RestartPreventExitStatus=78

[Install]
WantedBy=default.target
```

## Feeds

All feeds come from files in `/etc/indefinite-articles/feeds.d/`:

* `default.opml` ships with about 500 feeds. Every one was checked to be working and
  active. They cover tech, programming, Linux, security, AI, research papers (arXiv),
  science, space, medicine, climate, economics, business, world news, politics, long-form
  essays, history, philosophy, books, design, cities, food, sports, gaming and more.
  It is a config file, so your edits survive upgrades.
* The `Podcasts` category has about 50 podcasts (science, history, economics, tech,
  programming, security, interviews). Any feed works as a podcast: an entry is treated
  as an episode if it has an audio enclosure and either an `itunes:duration` or a feed
  where most entries have audio. So read-aloud versions of articles stay articles, and an
  OPML export from your podcast app works too. Episodes have their own shortlist
  (`max_podcast_candidates`), so the far more numerous articles can't crowd them out. The
  LLM judges an episode by its show notes (from the feed, or the episode's web page).
  Set `podcasts_per_digest = 0` for no podcasts.
* To add your own, put any OPML export from another feed reader in that directory, or a
  `.txt` file with one feed URL per line (`# Category` lines set the category).
* To skip whole categories, set
  `exclude_categories = ["Sports", "Gaming"]` in `indefinite-articles.toml`.

You don't need to trim the list for relevance. Everything gets the cheap first-pass rank,
and it learns from your replies which topics and sources to show you. A broad list mostly
gives it more to choose from.

`sudo -u indefinite-articles indefinite-articles feeds --categories` counts feeds per category.
`indefinite-articles feeds --broken` lists feeds that are failing; those are retried with
exponential backoff, up to once a day.

## Replying

Any of these work. You can mix them:

```
+1 +4 -2
3 like - more deep dives on compilers please
5 no, I don't care about crypto
Loved the Rust one. Less US politics generally.
```

Lines it can't parse go to the LLM, which works out what you meant. Quoted text from the
original email is ignored.

## Commands

Run these as the service user so they can use its database: `sudo -u indefinite-articles indefinite-articles …`

| command | what it does |
|---|---|
| `indefinite-articles check [--send-test]` | test the LLM, feeds, IMAP and SMTP |
| `indefinite-articles fetch` | fetch feeds and score new articles now |
| `indefinite-articles digest [--dry-run]` | send (or just print) a digest now |
| `indefinite-articles poll` | read replies now |
| `indefinite-articles feedback reply.eml` | process a reply saved to a file |
| `indefinite-articles profile [--history] [--set TEXT]` | show, review or replace the learned profile |
| `indefinite-articles status` | database statistics |
| `indefinite-articles feeds [--broken] [--categories]` | feed health: articles per week, your likes, errors |
| `indefinite-articles extract URL` | show the article text the LLM would see for a page |

## Notes

* **Hardware.** Embedding a few thousand articles a day takes a few minutes of CPU in
  total. Each article is scored in its own request, so every score uses the same scale.
  The profile and examples come first in every prompt and stay the same between requests,
  so Ollama and llama-server reuse their cached work and only process the article itself:
  a 14B model on CPU takes roughly 10–30 s per article. With about 150 articles a day that
  runs in the background after each hourly fetch, so the digest goes out quickly. When a
  reply changes your profile, only the `rescore_top` (default 30) best-ranked articles are
  scored again; the rest keep their earlier score. With no GPU, lower `max_candidates` or use a smaller model (`llama3.1:8b` or `qwen2.5:3b`).
* **No embedding model?** Set `embedding_model = ""` and it falls back to the newest
  articles, balanced across sources. With hundreds of feeds, that misses most of the
  good articles.
* **Article text.** Pages are downloaded with 4 parallel workers, once per article.
  If a page fails (paywall, HTTP 403, PDF, or a page built by JavaScript), the feed's
  summary is used instead. Links that point at localhost or private network addresses
  are refused unless `allow_private_addresses = true`. Set `fetch_full_articles = false`
  to judge on feed summaries only.
* All the settings named here are in `tuning.toml`; override them in `indefinite-articles.toml`.
* **Context size.** Scoring requests stay under `llm.max_prompt_chars` (about 24,000
  characters, roughly 7,000 tokens), because Ollama silently cuts off a prompt longer than
  its context. Ollama's default context is 4,096 tokens; the package's drop-in for
  `ollama.service` raises it to 8,192 so these prompts fit. With more RAM, run
  `sudo systemctl edit ollama`, add `Environment=OLLAMA_CONTEXT_LENGTH=16384`, then raise
  `max_prompt_chars` to about 50000.
* **If the LLM is down,** the digest still goes out. It is ranked by your past ratings of
  each source and by how recent each article is, and the email says so.
* **Data** is stored in `/var/lib/indefinite-articles/indefinite-articles.db`. Delete it to start over.
* **Security.** The sender check uses the `From:` header, which can be forged. The worst
  someone could do with a forged reply is skew your recommendations. If that matters,
  use a private mailbox address.
