NAME    := indefinite-articles
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
TOPDIR  := $(CURDIR)/build/rpmbuild
TARBALL := $(TOPDIR)/SOURCES/$(NAME)-$(VERSION).tar.gz
FILES   := pyproject.toml README.md LICENSE Makefile src tests dist packaging

.PHONY: all test tarball srpm rpm container-rpm clean

all: rpm

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

tarball:
	mkdir -p $(TOPDIR)/SOURCES
	tar --exclude=__pycache__ --transform 's,^,$(NAME)-$(VERSION)/,' -czf $(TARBALL) $(FILES)

srpm: tarball
	rpmbuild --define "_topdir $(TOPDIR)" -bs packaging/$(NAME).spec

# Needs: sudo dnf install rpm-build python3-devel pyproject-rpm-macros systemd-rpm-macros python3-numpy
rpm: tarball
	rpmbuild --define "_topdir $(TOPDIR)" -ba packaging/$(NAME).spec
	@find $(TOPDIR)/RPMS $(TOPDIR)/SRPMS -name '*.rpm'

# Build inside a clean Fedora container (no root needed on the host).
# Pass FEDORA=45 to build for Fedora 45.
FEDORA ?= 44
container-rpm:
	podman run --rm -v $(CURDIR):/src:Z -w /src registry.fedoraproject.org/fedora:$(FEDORA) \
	  bash -c 'dnf -y -q install rpm-build python3-devel pyproject-rpm-macros systemd-rpm-macros python3-numpy make tar python3-setuptools \
	           && make rpm'

clean:
	rm -rf build
