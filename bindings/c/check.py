"""Build and test the C package; optionally regenerate its header or HTML guide."""
import argparse
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def documentation(outputs):
    """Include the compiled programs and capture their real output by section."""
    generated = ROOT / "docs/_generated"
    generated.mkdir(exist_ok=True)
    for source in sorted((ROOT / "examples").glob("*.c")):
        output = outputs[source.stem]
        sections = re.split(r"^=== ([a-z_]+) ===\n", output, flags=re.M)
        if sections[0].strip() or len(sections) < 3:
            raise ValueError(f"{source.name}: output must have named sections")
        seen = set()
        for name, text in zip(sections[1::2], sections[2::2]):
            if name in seen:
                raise ValueError(f"{source.name}: repeated output section {name}")
            seen.add(name)
            (generated / f"{source.stem}-{name}.txt").write_text(text, encoding="utf-8")
    header = (ROOT / "include/tididi.h").read_text(encoding="utf-8")
    functions = header.split('extern "C" {', 1)[1]
    entries = re.findall(r"/\*\*\n(.*?)\*/\s*([^;]+\btididi_\w+\([^;]*\));", functions, re.S)
    declared = re.findall(r"\btididi_\w+\s*\([^;]*\);", functions)
    if len(entries) != len(declared):
        raise ValueError("every exported function needs a documented declaration")
    entries = {re.search(r"\b(tididi_\w+)\(", signature)[1]: (comment, signature)
               for comment, signature in entries}
    groups = [("domain", "Vtrees and limits"), ("circuit", "Construct and transform circuits"),
              ("query", "Query and inspect"), ("counter", "Observe choices"),
              ("evaluation", "Evaluate weights and costs"), ("storage", "Save and export"),
              ("lib", "Errors and strings")]
    reference = []
    for module, title in groups:
        reference.extend([title + "\n", "~" * len(title) + "\n\n"])
        source = (ROOT / "src" / (module + ".rs")).read_text(encoding="utf-8")
        names = re.findall(r'pub (?:unsafe )?extern "C" fn (tididi_\w+)\(', source)
        for name in names:
            comment, signature = entries.pop(name)
            signature = " ".join(signature.split())
            reference.append(f".. c:function:: {signature}\n\n")
            for line in comment.splitlines():
                reference.append("   " + re.sub(r"^\s*\* ?", "", line).rstrip() + "\n")
            reference.append("\n")
    if entries:
        raise ValueError(f"functions missing from the reference groups: {sorted(entries)}")
    (generated / "functions.rst").write_text("".join(reference), encoding="utf-8")
    run(sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
        ROOT / "docs", BUILD / "docs")


def check_installation(package):
    """Build and run a consumer using a moved installation, without build artifacts."""
    consumer = BUILD / "installed-consumer"
    consumer.mkdir(exist_ok=True)
    (consumer / "main.cpp").write_bytes((ROOT / "tests/cpp.cpp").read_bytes())
    (consumer / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(installed_consumer LANGUAGES CXX)
find_package(tididi CONFIG REQUIRED)
add_executable(consumer main.cpp)
set_target_properties(consumer PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
target_link_libraries(consumer PRIVATE tididi::tididi)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(consumer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
''', encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="installed-check-", dir=BUILD) as temporary:
        build = Path(temporary) / "consumer-build"
        relocated = Path(temporary) / "prefix"
        hidden = Path(temporary) / "cargo-hidden"
        package.rename(relocated)
        try:
            (BUILD / "cargo").rename(hidden)
            try:
                run("cmake", "-S", consumer, "-B", build, f"-DCMAKE_PREFIX_PATH={relocated}", "-DCMAKE_BUILD_TYPE=Release")
                run("cmake", "--build", build, "--config", "Release", "--parallel", "8")
                environment = dict(os.environ)
                environment["PATH"] = str(relocated / "bin") + os.pathsep + environment.get("PATH", "")
                run(build / "bin" / ("consumer.exe" if os.name == "nt" else "consumer"), env=environment)
            finally:
                hidden.rename(BUILD / "cargo")
        finally:
            relocated.rename(package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-header", action="store_true", help="replace the checked-in header with generated declarations")
    parser.add_argument("--docs", action="store_true", help="also build the executable guide (requires docs/requirements.txt)")
    args = parser.parse_args()
    os.environ.setdefault("CARGO_BUILD_JOBS", "8")
    result = run("cargo", "build", "--release", "--locked", "--manifest-path", ROOT / "Cargo.toml",
                 "--target-dir", BUILD / "cargo", "--message-format=json", capture_output=True, text=True)
    headers = []
    for line in result.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") == "build-script-executed":
            header = Path(item["out_dir"]) / "tididi.h"
            if header.is_file():
                headers.append(header)
    if len(headers) != 1:
        raise ValueError(f"expected one generated tididi.h; found {len(headers)}")
    header = headers[0].read_text(encoding="utf-8")
    checked_in = ROOT / "include/tididi.h"
    if args.write_header:
        checked_in.write_text(header, encoding="utf-8")
    elif checked_in.read_text(encoding="utf-8") != header:
        raise ValueError("tididi.h is stale; run python bindings/c/check.py --write-header")
    run("cmake", "-S", ROOT, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_LIBDIR=lib")
    run("cmake", "--build", BUILD, "--config", "Release", "--parallel", "8")
    run("ctest", "--test-dir", BUILD, "-C", "Release", "--output-on-failure")
    package = BUILD / "package"
    run("cmake", "--install", BUILD, "--config", "Release", "--prefix", package)
    check_installation(package)
    suffix = ".exe" if os.name == "nt" else ""
    outputs = {source.stem: run(BUILD / "bin" / (source.stem + suffix), capture_output=True, text=True).stdout
               for source in sorted((ROOT / "examples").glob("*.c"))}
    runpy.run_path(str(ROOT / "tests/tutorials.py"))["verify"](outputs)
    if args.docs:
        documentation(outputs)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise
