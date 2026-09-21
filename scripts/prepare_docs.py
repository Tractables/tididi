#!/usr/bin/env python3
"""Bundle matching example programs and images with rendered rustdoc.

Run after cargo doc --no-deps:
    python3 scripts/prepare_docs.py target/doc
"""

import argparse
import html
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil


class Excerpts(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.blocks = []
        self.active = False
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        if tag == "pre" and "tested-example" in dict(attrs).get("class", "").split():
            self.blocks.append("")
            self.active = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.blocks[-1] += data


def normalized(text):
    return "\n".join(line.strip() for line in text.strip().splitlines())


def release_tag(source):
    manifest = (source / "Cargo.toml").read_text(encoding="utf-8")
    return "v" + re.search(r'^version = "([^"]+)"', manifest, re.M)[1]


def prepare(source, preview, tag=None):
    expected_tag = release_tag(source)
    if tag and tag != expected_tag:
        raise ValueError(f"release tag {tag} differs from package version {expected_tag}")
    programs = {}
    pages = {}
    for markdown in sorted((source / "docs/examples").glob("*.md")):
        text = markdown.read_text(encoding="utf-8")
        match = re.search(r"complete program\]\((https://github\.com/Tractables/tididi/blob/([^/]+)/examples/([\w]+\.rs))\)", text)
        if not match:
            raise ValueError(f"{markdown}: missing complete-program link")
        url, linked_tag, filename = match.groups()
        if linked_tag != expected_tag:
            raise ValueError(f"{markdown}: expected a {expected_tag} complete-program link")
        program = (source / "examples" / filename).read_text(encoding="utf-8")
        page = preview / "tididi/guide/examples" / markdown.stem / "index.html"
        rendered = page.read_text(encoding="utf-8")
        expected = re.findall(r"```rust,ignore,\{class=tested-example\}\n(.*?)\n```", text, re.S)
        if not expected or list(map(normalized, Excerpts(rendered).blocks)) != list(map(normalized, expected)):
            raise ValueError(f"{page}: stale excerpts; rebuild rustdoc before refreshing")
        for excerpt in expected:
            if "\n" + normalized(excerpt) + "\n" not in "\n" + normalized(program) + "\n":
                raise ValueError(f"{markdown}: excerpt differs from {filename}")
        target = preview / "examples" / (filename + ".html")
        local = os.path.relpath(target, page.parent)
        if f'href="{url}"' not in rendered and f'href="{local}"' not in rendered:
            raise ValueError(f"{page}: missing complete-program link")
        pages[page] = rendered.replace(f'href="{url}"', f'href="{local}"')
        programs[filename] = program

    # Validate every page before changing any links.
    destination = preview / "examples"
    destination.mkdir(exist_ok=True)
    for filename, program in programs.items():
        shutil.copyfile(source / "examples" / filename, destination / filename)
        (destination / (filename + ".html")).write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{filename} — tididi</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ max-width: 72rem; margin: 2rem auto; padding: 0 1.5rem; font: 16px/1.6 system-ui, sans-serif; }}
a {{ color: light-dark(#16649c, #8ccaff); }}
pre {{ padding: 1.25rem; overflow-x: auto; background: light-dark(#f5f7f9, #202328); border-radius: .4rem; }}
code {{ font: 14px/1.6 ui-monospace, monospace; }}
</style></head><body>
<nav><a href="../tididi/guide/examples/index.html">Examples</a> · <a href="{filename}" download>Download Rust source</a></nav>
<h1>{filename}</h1>
<p>Run with <code>cargo run --example {Path(filename).stem}</code>.</p>
<pre><code>{html.escape(program)}</code></pre>
</body></html>
''', encoding="utf-8")
    for page, text in pages.items():
        page.write_text(text, encoding="utf-8")

    images = re.compile(r'https://raw\.githubusercontent\.com/Tractables/tididi/([^/]+)/([^"<>\s]+\.(?:svg|png))')
    for page in preview.rglob("*.html"):
        original = page.read_text(encoding="utf-8")

        def localize(match):
            if match[1] != expected_tag:
                raise ValueError(f"{page}: expected a {expected_tag} image link")
            relative = Path(match[2])
            target = preview / "assets" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
            return os.path.relpath(target, page.parent)

        updated = images.sub(localize, original)
        if updated != original:
            page.write_text(updated, encoding="utf-8")
    # Also refresh assets when rerunning against an already localized preview.
    for asset in (preview / "assets").rglob("*"):
        if asset.is_file():
            shutil.copyfile(source / asset.relative_to(preview / "assets"), asset)
    (preview / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=tididi/">'
        '<title>tididi documentation</title><a href="tididi/">Open the documentation</a>',
        encoding="utf-8")
    print(f"Bundled {len(programs)} programs for {len(pages)} walkthroughs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("preview", type=Path, nargs="?", default=Path("target/doc"))
    parser.add_argument("--release-tag", help="Check that the release tag matches Cargo.toml.")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.preview.resolve(), args.release_tag)
