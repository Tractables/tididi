"""Run the shipped examples as independent consumers of a cargo package."""

import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def check_output(markdown, stdout, name):
    """Check displayed output against complete stdout lines, in document order."""
    blocks = list(re.finditer(r"^```([^\n]*)\n(.*?)^```[ \t]*$", markdown, re.M | re.S))
    actual = "\n" + stdout.replace("\r\n", "\n")
    cursor = 0
    checked = 0
    for block in blocks:
        language, body = block.groups()
        if language.startswith("rust") and re.search(r"\b(?:print|println)!", body):
            if not markdown[block.end():].startswith("\n\nOutput:\n\n```text\n"):
                raise ValueError(f"{name}: printing excerpt needs an Output block immediately below it")
        if language != "text" or not markdown[:block.start()].endswith("\nOutput:\n\n"):
            continue
        if not body.strip():
            raise ValueError(f"{name}: empty Output block")
        expected = "\n" + body
        position = actual.find(expected, cursor)
        if position < 0:
            raise ValueError(f"{name}: documented output absent or out of order:\n{body}Actual output:\n{stdout}")
        cursor = position + len(expected) - 1
        checked += 1
    return checked


def test_output_checker():
    code = '```rust,ignore,{class=tested-example}\nprintln!("Count: 3");\n```'
    first = code + '\n\nOutput:\n\n```text\nCount: 3\n```'
    second = code + '\n\nOutput:\n\n```text\nDone\n```'
    markdown = first + '\n\n' + second
    assert check_output(markdown, "Setup\r\nCount: 3\r\nDone\r\n", "demo") == 2
    assert check_output('```text\na diagram\n```', "", "diagram") == 0
    for document, actual in [
        (code, "Count: 3\n"),
        (first.replace("Output:", "Result:"), "Count: 3\n"),
        (first, "Count: 4\n"),
        (first, "Count: 30\n"),
        (first, "Prefix Count: 3\n"),
        (first.replace("\nCount: 3\n", "\n\n"), "Count: 3\n"),
        (markdown, "Done\nCount: 3\n"),
    ]:
        try:
            check_output(document, actual, "demo")
        except ValueError:
            continue
        raise AssertionError(f"accepted incorrect documented output: {document!r}, {actual!r}")
    print("Example output checker regression cases passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--archive", type=Path, help="crate archive (defaults to cargo package output)")
    mode.add_argument("--self-test", action="store_true", help="test the output checker without building examples")
    args = parser.parse_args()
    if args.self_test:
        test_output_checker()
        return
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version=1"], cwd=root,
    ))
    package = next(p for p in metadata["packages"]
                   if Path(p["manifest_path"]).resolve() == root / "Cargo.toml")
    target = Path(metadata["target_directory"])
    name = f'{package["name"]}-{package["version"]}'
    archive = args.archive or target / "package" / f"{name}.crate"
    examples = sorted((root / "examples").glob("*.rs"))
    if not examples:
        raise RuntimeError("no examples found")
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="packaged-examples-", dir=target) as temporary:
        work = Path(temporary)
        with tarfile.open(archive) as contents:
            contents.extractall(work, filter="data")
        library = work / name
        for source in examples:
            shipped = library / "examples" / source.name
            if not shipped.is_file() or shipped.read_bytes() != source.read_bytes():
                raise RuntimeError(f"package is missing the current example: {source.name}")

        for source in examples:
            app = work / source.stem
            (app / "src").mkdir(parents=True)
            shutil.copyfile(library / "examples" / source.name, app / "src/main.rs")
            # Only add dependencies that the standalone program imports directly.
            extra = {
                "probabilistic_query": '\nnum-rational = "0.4"\nnum-traits = "0.2"',
                "marginalize_components": '\nnum-rational = "0.4"',
            }.get(source.stem, "")
            (app / "Cargo.toml").write_text(
                f'[package]\nname = "packaged-{source.stem}"\nversion = "0.0.0"\nedition = "2024"\n'
                f'[workspace]\n[dependencies]\ntididi = {{ path = "../{name}" }}{extra}\n'
            )
            print(f"Running packaged example: {source.stem}", flush=True)
            result = subprocess.run(
                ["cargo", "run", "--offline", "--manifest-path", str(app / "Cargo.toml"),
                 "--target-dir", str(work / "target")],
                cwd=app, check=True, stdout=subprocess.PIPE, text=True,
            )
            print(result.stdout, end="", flush=True)
            for page in sorted((root / "docs/examples").glob("*.md")):
                markdown = page.read_text(encoding="utf-8")
                if f"/examples/{source.name})" in markdown:
                    checked = check_output(markdown, result.stdout, page.name)
                    if not checked:
                        raise ValueError(f"{page.name}: no documented output checked")
                    print(f"Checked {checked} output blocks in {page.name}.", flush=True)
        print(f"Passed {len(examples)} standalone packaged examples.", flush=True)


if __name__ == "__main__":
    main()
