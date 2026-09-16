"""Check local rustdoc links and the rendered, source-verified walkthroughs."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import sys


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.images = []
        self.redirect = None
        self.excerpts = 0
        self.highlights = 0
        self.text = path.read_text(encoding="utf-8")
        self.feed(self.text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img" and "src" in attrs:
            self.images.append(attrs["src"])
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            match = re.search(r"url=(.*)", attrs.get("content", ""), re.I)
            if match:
                self.redirect = match[1]
        if tag == "pre" and "tested-example" in attrs.get("class", "").split():
            self.excerpts += 1
        if tag == "span" and attrs.get("class") in {"kw", "string", "number"}:
            self.highlights += 1


def check(root, source):
    root = root.resolve()
    pages = {path.resolve(): Page(path) for path in root.rglob("*.html")}
    errors = []
    checked = 0
    if root / "tididi/index.html" not in pages:
        return ["Missing crate documentation; run cargo doc --no-deps first."]

    def verify(origin, href):
        nonlocal checked
        url = urlsplit(href)
        if url.netloc == "tractables.github.io" and url.path.startswith("/tididi/tididi/"):
            target = root / unquote(url.path.removeprefix("/tididi/"))
        elif url.netloc == "raw.githubusercontent.com" and url.path.startswith("/Tractables/tididi/main/"):
            target = source / unquote(url.path.removeprefix("/Tractables/tididi/main/"))
        elif url.scheme or url.netloc:
            return
        else:
            target = (origin.parent / unquote(url.path)).resolve() if url.path else origin
        checked += 1
        if target.is_dir():
            target /= "index.html"
        if not target.is_file():
            errors.append(f"{origin.relative_to(root)}: missing target {href}")
            return
        seen = set()
        while target in pages and pages[target].redirect:
            if target in seen:
                errors.append(f"{origin.relative_to(root)}: redirect cycle {href}")
                return
            seen.add(target)
            target = (target.parent / pages[target].redirect).resolve()
        if not target.is_file():
            errors.append(f"{origin.relative_to(root)}: missing redirect target {href}")
        elif target in pages and url.fragment:
            fragment = unquote(url.fragment)
            ids = pages[target].ids
            line_range = re.fullmatch(r"\d+-\d+", fragment)
            if fragment not in ids and not (line_range and all(part in ids for part in fragment.split("-"))):
                errors.append(f"{origin.relative_to(root)}: missing anchor {href}")

    for path, page in pages.items():
        # Rustdoc's root index may include crates whose docs were not requested.
        if path.parent == root:
            continue
        for href in page.links + page.images:
            verify(path, href)

    walkthroughs = sorted((source / "docs/examples").glob("*.md"))
    for markdown in walkthroughs:
        path = root / "tididi/guide/examples" / markdown.stem / "index.html"
        if path not in pages:
            errors.append(f"Missing walkthrough: {markdown.stem}")
            continue
        page = pages[path]
        expected = markdown.read_text(encoding="utf-8").count("```rust,ignore,{class=tested-example}")
        if not expected or page.excerpts != expected:
            errors.append(f"{markdown.stem}: expected {expected} checked excerpts, found {page.excerpts}")
        if not page.highlights:
            errors.append(f"{markdown.stem}: missing syntax highlighting")
        rule = ".example-wrap.ignore:has(> pre.tested-example) > .tooltip { display: none; }"
        if rule not in page.text:
            errors.append(f"{markdown.stem}: missing scoped checked-excerpt styling")
    print(f"Checked {checked} local links and {len(walkthroughs)} rendered walkthroughs.")
    return errors


def test_checker():
    """Exercise the checker with broken links and damaged rendered excerpts."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)
        root = source / "target/doc"
        (root / "tididi/guide/examples/demo").mkdir(parents=True)
        (source / "docs/examples").mkdir(parents=True)
        (source / "docs/examples/demo.md").write_text("```rust,ignore,{class=tested-example}\nlet x = 1;\n```\n")
        index = root / "tididi/index.html"
        index.write_text('<a href="guide/examples/demo/index.html#example">Example</a>')
        page = root / "tididi/guide/examples/demo/index.html"
        html = ('<style>.example-wrap.ignore:has(> pre.tested-example) > .tooltip { display: none; }</style>'
                '<h2 id="example">Example</h2><pre class="tested-example"><span class="kw">let</span> x = 1;</pre>')
        page.write_text(html)
        assert not check(root, source)
        for damaged, reason in [
            (html.replace('id="example"', 'id="renamed"'), "missing anchor"),
            (html + '<img src="missing.svg">', "missing target"),
            (html.replace('class="tested-example"', 'class="unmarked"'), "checked excerpts"),
            (html.replace('class="kw"', 'class="plain"'), "syntax highlighting"),
            (html.replace('display: none;', 'display: block;'), "styling"),
        ]:
            page.write_text(damaged)
            assert any(reason in error for error in check(root, source)), reason
        page.write_text(html)
        index.write_text('<a href="missing.html">Missing page</a>')
        assert any("missing target" in error for error in check(root, source))
    print("Documentation checker regression cases passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        test_checker()
        sys.exit(0)
    source = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else source / "target/doc"
    errors = check(root, source)
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(bool(errors))
