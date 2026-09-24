# Contributing

## Checks

```sh
cargo test --all-targets
cargo test --doc
cargo clippy --all-targets -- -D warnings
cargo clippy --release --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
python3 tests/rendered_docs.py --self-test
python3 tests/rendered_docs.py target/doc --unbundled
python3 scripts/prepare_docs.py target/doc
python3 tests/rendered_docs.py target/doc
python3 tests/package_examples.py --self-test
python3 scripts/check_scenarios.py
python3 -m unittest discover -s tests -p test_scenarios.py
```

The README snippet matches the tested crate example; `docs/` is compiled into
rustdoc. Keep walkthrough excerpts in sync with their runnable programs in
`examples/`. The `tested-example` code
block class is reserved for excerpts verified by `tests/example_docs.rs`.
Follow printing excerpts with `Output:` and a `text` block;
`tests/package_examples.py` checks those blocks against the programs’ output.

Changes to diagram operations also need the randomized differential suite,
with and without debug assertions:

```sh
cargo test --test differential -- --ignored
cargo test --release --test differential -- --ignored
```

Set `TIDIDI_FUZZ_SEED` to replay a run and `TIDIDI_FUZZ_SECONDS` to set its duration.

Changes to packaging, examples or their displayed output also need `cargo package`
followed by `python3 tests/package_examples.py`, which runs each example as a
standalone consumer of the crate archive.

## Shared teaching scenarios

Hidden `scenario:` comments link teaching material to its script and complete
instance list in [docs/scenarios.md](docs/scenarios.md). When changing a lesson,
update that script as needed and review its Rust, Python, C and figure instances.
Then run `python3 scripts/check_scenarios.py --review <scenario-id>` to record
that review. CI rejects missing links, unregistered lessons and stale reviews;
it does not establish semantic agreement. Keep running the example checks.

## Python bindings

The independent package in `bindings/python` calls this crate through its public
API. Its [README](bindings/python/README.rst) gives the build and test commands.
Sphinx Gallery executes the narrative scripts in `bindings/python/examples` to
produce the tutorials and their output; edit those scripts, not generated pages.
Keep Python signatures in `tididi/__init__.pyi` aligned with the extension.
CI tests wheels on Linux, macOS, and Windows, and checks a source-archive install.

## C bindings

`bindings/c` calls the public Rust API through an opaque-handle interface. Run
`python bindings/c/check.py --docs` to verify the generated header, C/C++ consumers,
installed CMake package and executable guide. Its [README](bindings/c/README.rst)
lists prerequisites. Edit function contracts in `bindings/c/src`; the header and
API reference are generated from them. Tutorials include the compiled C sources
and captured output. CI checks Linux, macOS and Windows.

## Repository conventions

- Do not run rustfmt over the crate; it is not rustfmt-formatted.
- Tests that build diagrams must call `test_helpers::assert_canonical`:
  a correct query result alone does not establish a valid representation.
- Keep algorithms single-threaded. Callers control parallelism, limits and
  memory probes; the library reads no environment variables and has no
  process-wide state or native dependencies. Consumers need no Cargo features;
  `testing` exposes the generators and oracles in `test_helpers` and keeps its
  invariant checkers in a release build, as stated in the
  [architecture reference](docs/architecture.md#invariants).
- Return errors for invalid input and refused work. Named operations are
  checked; do not add panicking twins or `try_` aliases.
- Put each operation's contract on its default entry point. Batch methods
  link to that contract and explain their limit behavior. Guides introduce
  tasks and link to the items instead of repeating their contracts. Introduce
  Boolean expressions with `!`, `&` and `|`; introduce checked calls when
  discussing error handling. Name the shared vtree `vtree`.
- Public docs describe current behavior; keep benchmark results, development
  history and implementation debates out of them.
- Put unit tests in a nearby `tests/` directory: for a standalone `foo.rs`,
  `tests/foo.rs` (or `tests/foo/` for several files) declared with a `#[path]`
  module attribute. Implementation directories hold implementation code;
  integration tests go in the crate-root `tests/`. Use fixed seeds and no
  timing assertions. `tests/comment_lint.rs` checks comments and test
  placement; fix a violation rather than adding an exemption.
- Keep the vtree text format and shared constructors/accessors compatible
  with [vitri](https://github.com/Tractables/vitri).
- Commit messages must be suitable for publication: no tool footers, session
  links or machine-specific details.

[Architecture](docs/architecture.md) documents module responsibilities and
the representation invariants; add a row to its module table for a new public
module, and reference its invariants from code rather than restating them.

## Releasing

Use a `v<version>` tag matching `Cargo.toml`. Update the versioned documentation
URL in that manifest and the example/figure links in `docs/` when bumping the
version; the documentation checks catch mismatches. Publish the crate to
crates.io for versioned API docs on docs.rs. Publishing the GitHub release
runs the checks and attaches a bundled HTML archive; it does not publish the
crate. The archive is kept when the development Pages site changes.

## Cross-language behavior

`tests/fixtures/conformance*.txt` contains operation traces, stateful query
sessions and independent answers shared by the Rust, Python and C test suites. Add semantic regressions
in `tests/conformance_cases.py`, regenerate with `--write`, and run all three
suites. Each interpreter checks every assignment, cached evidence counts,
support and implied literals. Session traces also check ownership transfers, invalid observations, resource
refusals and retries, with each language checking its own handle lifecycle. `python3 tests/conformance_cases.py` checks fixture freshness.
