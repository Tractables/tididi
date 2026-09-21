"""Regressions for missing backlinks and stale cross-language reviews."""
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = runpy.run_path(str(ROOT / "scripts/check_scenarios.py"))
check = CHECKER["check"]


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "target"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch, prefix="scenario-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.files = ["README.md", "src/lib.rs", "src/guide.rs", "bindings/python/README.rst"]
        markers = ["<!-- scenario: docs/scenarios.md#lesson -->", "// scenario: docs/scenarios.md#lesson",
                   "// scenario: docs/scenarios.md#lesson", ".. scenario: docs/scenarios.md#lesson"]
        for name, marker in zip(self.files, markers):
            self.write(name, marker + "\n\n//! A lesson.\n")
        instances = "".join(f"- [Example](../{name}) <!-- reviewed: {'0' * 64} -->\n" for name in self.files)
        self.write("docs/scenarios.md", "# Scenarios\n\n## Lesson\n\nCount five models.\n\n### Instances\n\n" + instances)
        self.assertEqual(check(self.root, "lesson"), [])

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def edit(self, name, old, new):
        self.write(name, (self.root / name).read_text(encoding="utf-8").replace(old, new))

    def test_instance_change_requires_review_and_updates_record(self):
        before = (self.root / "docs/scenarios.md").read_text()
        self.edit("README.md", "A lesson", "An improved lesson")
        self.assertIn("review needed for README.md", "\n".join(check(self.root)))
        self.assertEqual(check(self.root, "lesson"), [])
        self.assertNotEqual(before, (self.root / "docs/scenarios.md").read_text())

    def test_scenario_change_requires_review_of_all_instances(self):
        self.edit("docs/scenarios.md", "five", "six")
        errors = "\n".join(check(self.root))
        for path in self.files:
            self.assertIn(path, errors)
        self.assertEqual(check(self.root, "lesson"), [])

    def test_new_unregistered_lesson_is_discovered(self):
        self.write("bindings/python/examples/10_new.py", '"""A new tutorial."""\n')
        self.assertIn("10_new.py: missing hidden scenario reference", "\n".join(check(self.root)))
        self.write("docs/figures/new.svg", "<svg/>\n")
        self.assertIn("new.svg: missing hidden scenario reference", "\n".join(check(self.root)))

    def test_backlink_is_required_in_both_directions(self):
        self.edit("docs/scenarios.md", "- [Example](../README.md)", "- [Example](../other.md)")
        self.write("other.md", "An unlinked instance.\n")
        errors = "\n".join(check(self.root))
        self.assertIn("README.md: not listed", errors)
        self.assertIn("other.md: missing reference", errors)

    def test_unknown_and_duplicate_references(self):
        self.edit("README.md", "#lesson", "#missing")
        self.assertIn("unknown scenario missing", "\n".join(check(self.root)))
        self.edit("README.md", "#missing", "#lesson")
        self.write("README.md", (self.root / "README.md").read_text() * 2)
        self.assertIn("duplicate scenario reference", "\n".join(check(self.root)))

    def test_missing_file_blocks_review_without_writing(self):
        (self.root / "README.md").unlink()
        before = (self.root / "docs/scenarios.md").read_bytes()
        self.assertIn("missing instance", "\n".join(check(self.root, "lesson")))
        self.assertEqual(before, (self.root / "docs/scenarios.md").read_bytes())

    def test_duplicate_instances_and_escaping_paths_are_rejected(self):
        original = (self.root / "docs/scenarios.md").read_text()
        self.write("docs/scenarios.md", original + original.splitlines()[-1] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate instance"):
            check(self.root)
        self.write("docs/scenarios.md", original.replace("../README.md", "../../outside.md"))
        with self.assertRaisesRegex(ValueError, "escapes the repository"):
            check(self.root)

    def test_shared_program_requires_both_scenario_reviews(self):
        record = (self.root / "docs/scenarios.md").read_text()
        self.write("docs/scenarios.md", record + "\n## Another lesson\n\nExplain reuse.\n\n### Instances\n\n"
                   + f"- [Shared example](../README.md) <!-- reviewed: {'0' * 64} -->\n")
        self.write("README.md", (self.root / "README.md").read_text()
                   + "<!-- scenario: docs/scenarios.md#another-lesson -->\n")
        check(self.root, "lesson")
        self.assertEqual(check(self.root, "another-lesson"), [])
        self.edit("README.md", "A lesson", "New teaching")
        self.assertEqual(len(check(self.root)), 2)
        self.assertEqual(len(check(self.root, "lesson")), 1)
        self.assertEqual(check(self.root, "another-lesson"), [])

    def test_rust_runtime_edits_do_not_invalidate_doc_review(self):
        path = self.root / "src/lib.rs"
        path.write_text(path.read_text() + "pub fn function() {}\n")
        self.assertEqual(check(self.root), [])
        self.edit("src/lib.rs", "A lesson", "New documentation")
        self.assertIn("src/lib.rs", "\n".join(check(self.root)))

    def test_windows_newlines_and_generated_pages_do_not_invalidate_review(self):
        for name in self.files + ["docs/scenarios.md"]:
            path = self.root / name
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.write("bindings/python/docs/tutorials/generated.rst", "Generated output")
        self.write("bindings/python/docs/_build/html/sources/page.rst", "Generated output")
        self.assertEqual(check(self.root), [])


if __name__ == "__main__":
    unittest.main()
