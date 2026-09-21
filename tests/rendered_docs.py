"""Check local rustdoc links and the rendered, source-verified walkthroughs."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import json
import sys
import runpy


BUNDLER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_docs.py"))


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


def check(root, source, bundled=True):
    root = root.resolve()
    pages = {path.resolve(): Page(path) for path in root.rglob("*.html")}
    errors = []
    checked = 0
    tag = BUNDLER["release_tag"](source)
    if root / "tididi/index.html" not in pages:
        return ["Missing crate documentation; run cargo doc --no-deps first."]

    def verify(origin, href):
        nonlocal checked
        url = urlsplit(href)
        if url.netloc == "tractables.github.io" and url.path.startswith("/tididi/tididi/"):
            target = root / unquote(url.path.removeprefix("/tididi/"))
        elif ((url.netloc == "raw.githubusercontent.com" and url.path.startswith("/Tractables/tididi/"))
              or (url.netloc == "github.com" and url.path.startswith("/Tractables/tididi/blob/"))):
            prefix = f"/Tractables/tididi/{'blob/' if url.netloc == 'github.com' else ''}{tag}/"
            if not url.path.startswith(prefix):
                errors.append(f"{origin.relative_to(root)}: asset must use release tag {tag}: {href}")
            elif bundled:
                errors.append(f"{origin.relative_to(root)}: unbundled asset {href}; run scripts/prepare_docs.py")
            return
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
            if url.fragment not in ids and fragment not in ids and not (line_range and all(part in ids for part in fragment.split("-"))):
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
    navigation = source / "docs/example-navigation.js"
    if navigation.is_file():
        index = pages.get(root / "tididi/guide/examples/index.html")
        match = re.search(r"const examples = (\[.*?\]);", index.text) if index else None
        if not match:
            errors.append("Missing numbered example navigation")
        else:
            entries = json.loads(match[1].replace(",]", "]"))
            names = [name for name, _ in entries]
            if sorted(names) != sorted(path.stem for path in walkthroughs):
                errors.append("Example navigation must name each walkthrough once")
            listing = re.search(r"<ol>(.*?)</ol>", index.text, re.S)
            links = re.findall(r'href="([^"/]+)/index.html"', listing[1]) if listing else []
            if links != names:
                errors.append("Example index and sidebar have different reading orders")
            script = navigation.read_text(encoding="utf-8").strip()
            for name in [None, *names]:
                path = root / "tididi/guide/examples"
                path = path / name if name else path
                page = pages.get(path / "index.html")
                if not page or match[0] not in page.text or script not in page.text:
                    errors.append(f"{name or 'index'}: missing shared example navigation")
    print(f"Checked {checked} local links and {len(walkthroughs)} rendered walkthroughs.")
    return errors


def test_checker():
    """Exercise the checker with broken links and damaged rendered excerpts."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)
        (source / "Cargo.toml").write_text('[package]\nversion = "0.1.0"\n')
        root = source / "target/doc"
        (root / "tididi/guide/examples/demo").mkdir(parents=True)
        (source / "docs/examples").mkdir(parents=True)
        (source / "docs/examples/demo.md").write_text("```rust,ignore,{class=tested-example}\nlet x = 1;\n```\n")
        index = root / "tididi/index.html"
        index.write_text('<a href="guide/examples/demo/index.html#example">Example</a>'
                         '<a href="guide/examples/demo/index.html#impl%3CT%3E">Encoded ID</a>'
                         '<a href="guide/examples/demo/index.html#method%3CT%3E">Decoded ID</a>')
        page = root / "tididi/guide/examples/demo/index.html"
        html = ('<style>.example-wrap.ignore:has(> pre.tested-example) > .tooltip { display: none; }</style>'
                '<i id="impl%3CT%3E"></i><i id="method&lt;T&gt;"></i>'
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
        for prefix in ["https://github.com/Tractables/tididi/blob/",
                       "https://raw.githubusercontent.com/Tractables/tididi/"]:
            pinned = f'<a href="{prefix}v0.1.0/examples/demo.rs">source</a>'
            page.write_text(html + pinned)
            assert not check(root, source, bundled=False)
            assert any("unbundled asset" in error for error in check(root, source))
            for wrong in ["main", "v0.2.0"]:
                page.write_text(html + pinned.replace("v0.1.0", wrong))
                assert any("release tag" in error for error in check(root, source, bundled=False))
        page.write_text(html)
        index.write_text('<a href="missing.html">Missing page</a>')
        assert any("missing target" in error for error in check(root, source))
    test_bundler()
    print("Documentation checker regression cases passed.")


def test_bundler():
    """Bundling is repeatable and refuses pages built from different source."""
    import tempfile
    prepare = BUNDLER["prepare"]
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)
        (source / "Cargo.toml").write_text('[package]\nversion = "0.1.0"\n')
        root = source / "target/doc"
        page = root / "tididi/guide/examples/demo/index.html"
        page.parent.mkdir(parents=True)
        (source / "examples").mkdir()
        (source / "docs/examples").mkdir(parents=True)
        url = "https://github.com/Tractables/tididi/blob/v0.1.0/examples/demo.rs"
        (source / "examples/demo.rs").write_text("fn main() { println!(\"example\"); }\n")
        (source / "docs/examples/demo.md").write_text(
            f"[complete program]({url})\n```rust,ignore,{{class=tested-example}}\nfn main() {{ println!(\"example\"); }}\n```\n")
        (source / "docs/graph.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        page.write_text(f'<a href="{url}">complete program</a>'
                       '<pre class="tested-example"><code>fn main() { println!(&quot;example&quot;); }</code></pre>'
                       '<img src="https://raw.githubusercontent.com/Tractables/tididi/v0.1.0/docs/graph.svg">')
        try:
            prepare(source, root, "v0.2.0")
        except ValueError as error:
            assert "differs from package version" in str(error)
        else:
            raise AssertionError("mismatched release tags must be rejected")
        prepare(source, root, "v0.1.0")
        first = page.read_text()
        assert "github.com" not in first and "githubusercontent.com" not in first
        assert (root / "examples/demo.rs").read_text() == (source / "examples/demo.rs").read_text()
        assert (root / "assets/docs/graph.svg").read_text() == (source / "docs/graph.svg").read_text()
        prepare(source, root)
        assert page.read_text() == first
        page.write_text(first.replace("println!", "print!"))
        try:
            prepare(source, root)
        except ValueError as error:
            assert "stale excerpts" in str(error)
        else:
            raise AssertionError("stale rendered pages must not receive new programs")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        test_checker()
        sys.exit(0)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("target/doc"))
    parser.add_argument("--unbundled", action="store_true", help="Allow version-pinned remote assets from plain cargo doc.")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    errors = check(args.root, source, bundled=not args.unbundled)
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(bool(errors))
