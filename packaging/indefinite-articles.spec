Name:           indefinite-articles
Version:        0.4.0
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
# Speeds up the embedding pre-filter ~50x; pure Python is used without it.
Recommends:     python3-numpy

%description
indefinite-articles is a service that reads RSS/Atom feeds, ranks new articles with a
locally hosted large language model, and emails you a daily digest. Reply to the
email (e.g. "+1 +4 -2", or in plain words) and it learns your preferences.

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
install -Dpm 0644 dist/indefinite-articles.sysusers %{buildroot}%{_sysusersdir}/indefinite-articles.conf
install -Dpm 0640 dist/indefinite-articles.toml %{buildroot}%{_sysconfdir}/indefinite-articles/indefinite-articles.toml
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
%{_sysusersdir}/indefinite-articles.conf
%dir %attr(0750,root,indefinite-articles) %{_sysconfdir}/indefinite-articles
%config(noreplace) %attr(0640,root,indefinite-articles) %{_sysconfdir}/indefinite-articles/indefinite-articles.toml
%dir %{_sysconfdir}/indefinite-articles/feeds.d
%config(noreplace) %{_sysconfdir}/indefinite-articles/feeds.d/default.opml
%dir %attr(0750,indefinite-articles,indefinite-articles) %{_sharedstatedir}/indefinite-articles

%changelog
* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.4.0-1
- Rename project from recommend to indefinite-articles: package, command,
  systemd unit, service user, /etc and /var/lib paths and the Python module
  (indefinite_articles)

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.3.1-1
- Record reply feedback and the processed-message marker in one transaction,
  so a crash can neither lose nor double-count a reply

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.3.0-1
- Ship ~500 checked feeds in 31 categories (/etc/recommend/feeds.d/default.opml)
- Load feeds from OPML/text files in feeds.d; exclude_categories option
- Embedding pre-filter (nomic-embed-text) ranks every article; LLM scores the best 150
- Parallel per-host fetching, conditional GET, gzip, backoff for failing feeds
- Tolerate malformed feeds; new `recommend feeds` command

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.2.0-1
- Fetch each candidate article's page so the LLM ranks on the article text
- Size scoring batches to fit the model's context window (llm.max_prompt_chars)
- New `recommend extract URL` command

* Tue Nov 03 2026 Martin Rixham <martin.rixham@gmail.com> - 0.1.0-1
- Initial package
