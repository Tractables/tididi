"""Refuse inconsistent release artifacts before publication."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github/scripts"))
from check_release import check

FILES = ("Cargo.toml", "Cargo.lock", "CITATION.cff", "bindings/c/Cargo.toml",
         "bindings/c/Cargo.lock", "bindings/c/CMakeLists.txt",
         "bindings/python/Cargo.toml", "bindings/python/Cargo.lock",
         "bindings/python/pyproject.toml")


class ReleaseTests(unittest.TestCase):
    def test_checkout_and_tag(self):
        version = check(ROOT)
        self.assertEqual(check(ROOT, "v" + version), version)
        with self.assertRaisesRegex(ValueError, "release tag"):
            check(ROOT, "v999.0.0")

    def test_each_distribution_detects_stale_metadata(self):
        version = check(ROOT)
        for relative in FILES:
            with self.subTest(file=relative), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                for name in FILES:
                    destination = source / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, destination)
                changed = source / relative
                text = changed.read_text()
                if relative.endswith("pyproject.toml"):
                    text = text.replace('dynamic = ["version"]', 'version = "999.0.0"')
                else:
                    text = text.replace(version, "999.0.0")
                changed.write_text(text)
                with self.assertRaises(ValueError):
                    check(source)
