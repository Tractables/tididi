# Contributing

## Checks

```sh
cargo test --all-targets
cargo test --doc
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
python3 tests/rendered_docs.py --self-test
python3 tests/rendered_docs.py target/doc --unbundled
python3 scripts/prepare_docs.py target/doc
python3 tests/rendered_docs.py target/doc
python3 tests/package_examples.py --self-test
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

## Python bindings

The independent package in `bindings/python` calls this crate through its public
API. Its [README](bindings/python/README.rst) gives the build and test commands.
Sphinx Gallery executes the narrative scripts in `bindings/python/examples` to
produce the tutorials and their output; edit those scripts, not generated pages.
Keep Python signatures in `tididi/__init__.pyi` aligned with the extension.
CI tests wheels on Linux, macOS, and Windows, and checks a source-archive install.

## Repository conventions

- Do not run rustfmt over the crate; it is not rustfmt-formatted.
- Tests that build diagrams must call `test_helpers::assert_canonical`:
  a correct query result alone does not establish a valid representation.
- Keep algorithms single-threaded. Callers control parallelism, limits and
  memory probes; the library reads no environment variables and has no
  process-wide state or native dependencies. Consumers need no Cargo features;
  `testing` enables generators, oracles and invariant checks for tests.
- Return errors for invalid input and refused work. Named operations are
  checked; do not add panicking twins or `try_` aliases.
- Put each operation's contract on its default entry point. Batch methods
  link to that contract and explain their limit behavior. Introduce Boolean
  expressions with `!`, `&` and `|`; introduce checked calls when discussing
  error handling. Name the shared vtree `vtree`.
- Keep the vtree text format and shared constructors/accessors compatible
  with [vitri](https://github.com/Tractables/vitri).

[Architecture](docs/architecture.md) documents module responsibilities and
the representation invariants.

## Releasing

Use a `v<version>` tag matching `Cargo.toml`. Update the versioned documentation
URL in that manifest and the example/figure links in `docs/` when bumping the
version; the documentation checks catch mismatches. Publish the crate to
crates.io for versioned API docs on docs.rs. Publishing the GitHub release
runs the checks and attaches a bundled HTML archive; it does not publish the
crate. The archive is kept when the development Pages site changes.
