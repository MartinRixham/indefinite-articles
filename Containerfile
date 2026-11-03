# Container image for indefinite-articles.
#
#   podman build -t indefinite-articles .
#
# The image installs the same RPM that `make rpm` produces, so the service user,
# /etc/indefinite-articles (with the default feeds.d) and /var/lib/indefinite-articles
# are exactly as on a Fedora host. See "Run in a container" in README.md.

ARG FEDORA=44

# The RPM is noarch: build it once on the native platform, even for multi-arch images.
FROM --platform=$BUILDPLATFORM registry.fedoraproject.org/fedora:${FEDORA} AS build
ENV LANG=C.UTF-8
RUN dnf -y -q install rpm-build python3-devel pyproject-rpm-macros systemd-rpm-macros \
        python3-setuptools make tar \
    && dnf clean all
WORKDIR /src
COPY . .
RUN make rpm

FROM registry.fedoraproject.org/fedora-minimal:${FEDORA}
ARG FEDORA
LABEL org.opencontainers.image.title="indefinite-articles" \
      org.opencontainers.image.description="Daily article recommendations by email, trained on your replies with a local LLM" \
      org.opencontainers.image.licenses="MIT"

RUN --mount=type=bind,from=build,source=/src/build/rpmbuild/RPMS/noarch,target=/rpms \
    dnf -y install --setopt=install_weak_deps=False \
        /rpms/indefinite-articles-*.noarch.rpm python3-numpy tzdata \
    && dnf clean all \
    && rm -rf /var/cache/* /var/log/dnf*

ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    INDEFINITE_ARTICLES_CONFIG=/etc/indefinite-articles/indefinite-articles.toml

USER indefinite-articles
WORKDIR /var/lib/indefinite-articles
VOLUME /var/lib/indefinite-articles

ENTRYPOINT ["/usr/bin/indefinite-articles"]
CMD ["run"]
