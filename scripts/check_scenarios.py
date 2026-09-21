"""Check teaching material against its shared scenario and recorded review."""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
RECORD = Path("docs/scenarios.md")
REFERENCE = re.compile(
    r"^(?:<!-- |\.\. |// |# )scenario: docs/scenarios\.md#([a-z0-9-]+)(?: -->)?$",
    re.MULTILINE,
)
INSTANCE = re.compile(r"^- \[([^\]]+)\]\(([^)]+)\) <!-- reviewed: ([a-f0-9]{64}) -->$")


@dataclass
class Instance:
    path: Path
    reviewed: str
    line: int


@dataclass
class Scenario:
    name: str
    script: str
    instances: list[Instance]


def local_path(root, parent, target):
    path = (parent / target).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"instance path escapes the repository: {target}")
    return path.relative_to(root.resolve())


def read_scenarios(root):
    text = (root / RECORD).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    scenarios = {}
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        title = lines[start][3:].strip()
        name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if name in scenarios:
            raise ValueError(f"duplicate scenario: {name}")
        body = lines[start:end]
        try:
            index = next(i for i, line in enumerate(body) if line.strip() == "### Instances")
        except StopIteration:
            raise ValueError(f"{name}: missing Instances list") from None
        script = "".join(body[:index]).strip()
        instances = []
        for offset in range(index + 1, len(body)):
            line = body[offset].strip()
            if not line:
                continue
            match = INSTANCE.fullmatch(line)
            if not match:
                raise ValueError(f"{name}: invalid instance entry on line {start + offset + 1}")
            _, target, reviewed = match.groups()
            path = local_path(root, (root / RECORD).parent, target)
            if any(item.path == path for item in instances):
                raise ValueError(f"{name}: duplicate instance {path}")
            instances.append(Instance(path, reviewed, start + offset))
        if not instances:
            raise ValueError(f"{name}: no instances")
        scenarios[name] = Scenario(name, script, instances)
    if not scenarios:
        raise ValueError("no scenarios found")
    return lines, scenarios


def teaching_files(root):
    """Discover new standalone lessons without requiring a registry edit first."""
    paths = {Path(p) for p in ["README.md", "src/lib.rs", "src/guide.rs", "bindings/python/README.rst"]}
    patterns = ["docs/**/*.md", "docs/**/*.svg", "examples/**/*.rs",
                "bindings/python/docs/**/*.rst", "bindings/python/examples/**/*.py",
                "bindings/python/examples/**/*.rst", "bindings/c/README.rst",
                "bindings/c/docs/**/*.rst", "bindings/c/examples/**/*.c", "bindings/c/examples/**/*.h"]
    for pattern in patterns:
        for path in root.glob(pattern):
            relative = path.relative_to(root)
            if any(part in {"_build", "_static", "_generated", "tutorials", "__pycache__"} for part in relative.parts):
                continue
            if relative == RECORD or path.name in {"logo.svg", "sg_execution_times.rst"}:
                continue
            paths.add(relative)
    # Item-level Rust documentation can also participate in a shared scenario.
    for path in (root / "src").rglob("*.rs"):
        if REFERENCE.search(path.read_text(encoding="utf-8")):
            paths.add(path.relative_to(root))
    return paths


def fingerprint(root, scenario, instance):
    text = (root / instance.path).read_text(encoding="utf-8")
    if instance.path.parts[0] == "src" and instance.path != Path("src/guide.rs"):
        # Runtime edits should not require reviewing an unchanged explanation.
        text = "\n".join(line.strip() for line in text.splitlines()
                         if line.lstrip().startswith(("//!", "///")))
    return hashlib.sha256((scenario.script + "\0" + text).encode("utf-8")).hexdigest()


def inspect(root, scenarios):
    errors = []
    declared = {}
    for name, scenario in scenarios.items():
        for instance in scenario.instances:
            declared.setdefault(instance.path, set()).add(name)
    for path in sorted(teaching_files(root) | declared.keys()):
        if not (root / path).is_file():
            errors.append(f"{path}: missing instance; update its scenario and references")
            continue
        references = REFERENCE.findall((root / path).read_text(encoding="utf-8"))
        if not references:
            errors.append(f"{path}: missing hidden scenario reference")
        if len(set(references)) != len(references):
            errors.append(f"{path}: duplicate scenario reference")
        for name in set(references):
            if name not in scenarios:
                errors.append(f"{path}: unknown scenario {name}")
            elif name not in declared.get(path, set()):
                errors.append(f"{path}: not listed in {RECORD}#{name}")
        for name in declared.get(path, set()) - set(references):
            errors.append(f"{path}: missing reference to {RECORD}#{name}")
    return errors


def check(root, review=None):
    lines, scenarios = read_scenarios(root)
    if review is not None and review not in scenarios:
        return [f"unknown scenario: {review}"]
    errors = inspect(root, scenarios)
    if errors:
        return errors  # Never stamp an incomplete or broken set of references.
    if review is not None:
        scenario = scenarios[review]
        for instance in scenario.instances:
            actual = fingerprint(root, scenario, instance)
            lines[instance.line] = lines[instance.line].replace(instance.reviewed, actual)
            instance.reviewed = actual
        (root / RECORD).write_text("".join(lines), encoding="utf-8")
    for name, scenario in scenarios.items():
        stale = [str(item.path) for item in scenario.instances
                 if item.reviewed != fingerprint(root, scenario, item)]
        if stale:
            errors.append(f"{RECORD}#{name}: review needed for {', '.join(stale)}. "
                          f"Update the scenario and check all its instances, then run "
                          f"python3 scripts/check_scenarios.py --review {name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", metavar="SCENARIO", help="record a completed review of one scenario and all its instances")
    args = parser.parse_args()
    try:
        errors = check(ROOT, args.review)
    except (OSError, ValueError) as error:
        errors = [str(error)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Scenario references and reviews are current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
