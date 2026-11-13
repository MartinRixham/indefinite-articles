Name:           indefinite-articles
Version:        0.4.7
Release:        1%{?dist}
Summary:        Daily article recommendations by email, trained on your replies with a local LLM

License:        MIT
URL:            https://example.com/indefinite-articles
Source0:        %{name}-%{version}.tar.gz

BuildArch:      noarch
# The 'indefinite-articles' user is created by rpm from the sysusers.d file (Fedora 42+).
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros
BuildRequires:  systemd-rpm-macros

# Any OpenAI-compatible local server works; Ollama is the default configuration.
Recommends:     (ollama or llama-cpp)
# GPU support for AMD (incl. integrated Radeon); the service won't start on CPU only.
Recommends:     (ollama-rocm if ollama)

%description
indefinite-articles is a service that reads RSS/Atom feeds, ranks new articles
with a locally hosted large language model, and emails you a daily digest.
Reply to the email (e.g. "+1 +4 -2", or in plain words) and it learns your
preferences.

%prep
%autosetup -p1

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files -l indefinite_articles
install -Dpm 0644 dist/indefinite-articles.service %{buildroot}%{_unitdir}/indefinite-articles.service
install -Dpm 0644 dist/ollama-igpu.conf %{buildroot}%{_unitdir}/ollama.service.d/50-indefinite-articles-igpu.conf
install -Dpm 0644 dist/indefinite-articles.sysusers %{buildroot}%{_sysusersdir}/indefinite-articles.conf
install -Dpm 0640 dist/indefinite-articles.toml %{buildroot}%{_sysconfdir}/indefinite-articles/indefinite-articles.toml
install -Dpm 0644 dist/defaults.toml %{buildroot}%{_sysconfdir}/indefinite-articles/defaults.toml
install -Dpm 0644 dist/feeds.d/default.opml %{buildroot}%{_sysconfdir}/indefinite-articles/feeds.d/default.opml
install -dm 0750 %{buildroot}%{_sharedstatedir}/indefinite-articles

%check
%pyproject_check_import
%{py3_test_envvars} %{python3} -m unittest discover -s tests -v

%post
%systemd_post indefinite-articles.service

%preun
%systemd_preun indefinite-articles.service

%postun
%systemd_postun_with_restart indefinite-articles.service

%files -f %{pyproject_files}
%doc README.md
%{_bindir}/indefinite-articles
%{_unitdir}/indefinite-articles.service
%dir %{_unitdir}/ollama.service.d
%{_unitdir}/ollama.service.d/50-indefinite-articles-igpu.conf
%{_sysusersdir}/indefinite-articles.conf
%dir %attr(0750,root,indefinite-articles) %{_sysconfdir}/indefinite-articles
%config(noreplace) %attr(0640,root,indefinite-articles) %{_sysconfdir}/indefinite-articles/indefinite-articles.toml
%config(noreplace) %{_sysconfdir}/indefinite-articles/defaults.toml
%dir %{_sysconfdir}/indefinite-articles/feeds.d
%config(noreplace) %{_sysconfdir}/indefinite-articles/feeds.d/default.opml
%dir %attr(0750,indefinite-articles,indefinite-articles) %{_sharedstatedir}/indefinite-articles

%changelog
* Fri Nov 13 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.7
- Exploration picks articles unlike the recommendations
- Deduplication treats articles and podcast episodes separately
- More science feeds in the default OPML

* Thu Nov 12 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.6
- Digest picks are penalized for overlapping with articles already picked
- Apple Podcasts links are found for episodes from the last day too

* Wed Nov 11 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.5
- Replies are also read from Gmail's spam folder
- Reply processing is retried when the LLM is unavailable
- Feed fetches are lined up with the digest send time
- Profile rewriting copes better near the word limit and avoids examples
- Paywalled feeds removed from the default OPML
- Better Apple Podcasts link lookup
- Database cleanup

* Sat Nov 07 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.4
- LLM timeouts are learned per kind of request, so long scoring prompts and
  large embedding batches are no longer cut off early
- Ranking waits for the first fetch and doesn't score if embedding fails

* Sat Nov 07 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.3
- Embedding and scoring run in order on one ranking thread
- Reply interpretation is less inclined to overinterpret
- More international feeds in the default OPML
- The service requires ollama.service
- Better Apple Podcasts link lookup

* Fri Nov 06 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.2
- Pull missing models from Ollama on start-up (pull_models)
- Podcast episode recommendations, with their own shortlist and digest quota
- Apple Podcasts links for recommended episodes
- Newer default LLM models

* Thu Nov 05 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.1
- Fetch, scoring and inbox polling run on separate threads; scoring stops when
  the digest is due, so a slow LLM can't delay it
- Score each article in its own request with a shared prompt prefix (KV cache reuse)
- Pre-filter learns from likes and dislikes separately, plus a logistic-regression
  classifier once there is enough feedback; numpy is now required
- Refuse to start when Ollama runs the model on the CPU only; recommend
  ollama-rocm and ship an Ollama drop-in for integrated AMD GPUs
- Default model is qwen2.5:14b
- After a profile change, only rescore the top articles (new rescore_top option)
- Debug information in the daily email
- Shipped config no longer hard codes reading preferences
- Build for Fedora 44 and 45; publish a container image

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.0
- Rename project from recommend to indefinite-articles: package, command,
  systemd unit, service user, /etc and /var/lib paths and the Python module
  (indefinite_articles)

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.3.1
- Record reply feedback and the processed-message marker in one transaction,
  so a crash can neither lose nor double-count a reply

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.3.0
- Ship ~500 checked feeds in 31 categories (/etc/recommend/feeds.d/default.opml)
- Load feeds from OPML/text files in feeds.d; exclude_categories option
- Embedding pre-filter (nomic-embed-text) ranks every article; LLM scores the best 150
- Parallel per-host fetching, conditional GET, gzip, backoff for failing feeds
- Tolerate malformed feeds; new `recommend feeds` command

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.2.0
- Fetch each candidate article's page so the LLM ranks on the article text
- Size scoring batches to fit the model's context window (llm.max_prompt_chars)
- New `recommend extract URL` command

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.1.0
- Initial package
