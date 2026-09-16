"""Run the shipped examples as independent consumers of a cargo package."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="crate archive (defaults to cargo package output)")
    args = parser.parse_args()
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
            # This example names rational numbers and their numeric traits directly.
            extra = '\nnum-rational = "0.4"\nnum-traits = "0.2"' if source.stem == "probabilistic_query" else ""
            (app / "Cargo.toml").write_text(
                f'[package]\nname = "packaged-{source.stem}"\nversion = "0.0.0"\nedition = "2024"\n'
                f'[workspace]\n[dependencies]\ntididi = {{ path = "../{name}" }}{extra}\n'
            )
            print(f"Running packaged example: {source.stem}", flush=True)
            subprocess.run(
                ["cargo", "run", "--offline", "--manifest-path", str(app / "Cargo.toml"),
                 "--target-dir", str(work / "target")],
                cwd=app, check=True,
            )
        print(f"Passed {len(examples)} standalone packaged examples.", flush=True)


if __name__ == "__main__":
    main()
