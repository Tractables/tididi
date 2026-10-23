#!/usr/bin/env python3
"""Check that the language packages and release metadata describe one version."""

import argparse
from pathlib import Path
import re

from prepare_docs import release_tag


def one(text, pattern, source):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        raise ValueError(f"{source}: expected one version declaration")
    return matches[0]


def check(source, tag=""):
    expected_tag = release_tag(source)
    version = expected_tag[1:]
    if tag and tag != expected_tag:
        raise ValueError(f"release tag {tag} differs from {expected_tag}")
    for relative in ("Cargo.toml", "bindings/c/Cargo.toml", "bindings/python/Cargo.toml"):
        text = (source / relative).read_text(encoding="utf-8")
        found = one(text, r'^version\s*=\s*"([^"]+)"\s*$', relative)
        if found != version:
            raise ValueError(f"{relative}: version {found}, expected {version}")
    citation = one((source / "CITATION.cff").read_text(encoding="utf-8"),
                   r'^version:\s*["\']?([^"\'\s]+)["\']?\s*$', "CITATION.cff")
    if citation != version:
        raise ValueError(f"CITATION.cff: version {citation}, expected {version}")
    cmake = one((source / "bindings/c/CMakeLists.txt").read_text(encoding="utf-8"),
                r'^project\(tididi_c VERSION ([^ ]+) LANGUAGES C CXX\)$', "CMakeLists.txt")
    # CMake package versions are numeric; Cargo/Python retain the prerelease suffix.
    numeric = version.split("-", 1)[0].split("+", 1)[0]
    if cmake != numeric:
        raise ValueError(f"CMakeLists.txt: version {cmake}, expected {numeric}")
    manifest = (source / "Cargo.toml").read_text(encoding="utf-8")
    documentation = one(manifest, r'^documentation\s*=\s*"([^"]+)"\s*$', "Cargo.toml")
    if documentation != f"https://docs.rs/tididi/{version}/tididi/":
        raise ValueError("Cargo.toml: documentation URL differs from the package version")
    pyproject = (source / "bindings/python/pyproject.toml").read_text(encoding="utf-8")
    if not re.search(r'^dynamic\s*=\s*\["version"\]\s*$', pyproject, re.M):
        raise ValueError("pyproject.toml: version must come from the binding Cargo manifest")
    for relative in ("Cargo.lock", "bindings/c/Cargo.lock", "bindings/python/Cargo.lock"):
        text = (source / relative).read_text(encoding="utf-8")
        for package in text.split("[[package]]")[1:]:
            name = one(package, r'^name = "([^"]+)"$', relative)
            if name in {"tididi", "tididi-c", "tididi-python"}:
                locked = one(package, r'^version = "([^"]+)"$', relative)
                if locked != version:
                    raise ValueError(f"{relative}: {name} is {locked}, expected {version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag", default="")
    args = parser.parse_args()
    try:
        print(f"Release metadata agrees: {check(Path(__file__).resolve().parents[2], args.release_tag)}")
    except ValueError as error:
        parser.exit(1, f"{error}\n")
