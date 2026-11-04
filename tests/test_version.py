import re
import unittest
from pathlib import Path

from indefinite_articles import __version__

SPEC = Path(__file__).resolve().parent.parent / "packaging" / "indefinite-articles.spec"


class VersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = SPEC.read_text(encoding="utf-8")

    def test_spec_version_matches_package(self) -> None:
        match = re.search(r"^Version:\s*(\S+)", self.spec, re.MULTILINE)
        assert match is not None
        self.assertEqual(match.group(1), __version__)

    def test_newest_changelog_entry_matches_package(self) -> None:
        changelog = self.spec.split("%changelog", 1)[1]
        match = re.search(r"^\*.* - (\S+)-\d+$", changelog, re.MULTILINE)
        assert match is not None
        self.assertEqual(match.group(1), __version__)


if __name__ == "__main__":
    unittest.main()
