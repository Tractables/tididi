# Contributing

## Checks

```sh
cargo test --all-targets
cargo test --doc
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
python3 tests/rendered_docs.py --self-test
python3 tests/rendered_docs.py target/doc
python3 tests/package_examples.py --self-test
```

The README and `docs/` are compiled into rustdoc. Keep walkthrough excerpts
in sync with their runnable programs in `examples/`. The `tested-example` code
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

## Repository conventions

- Do not run rustfmt over the crate; it is not rustfmt-formatted.
- Tests that build diagrams must call `test_helpers::assert_canonical`:
  a correct query result alone does not establish a valid representation.
- Keep algorithms single-threaded. Callers control parallelism, limits and
  memory probes; the library reads no environment variables and has no
  process-wide state, Cargo features or native dependencies.
- Return errors for invalid input and refused work. Named operations are
  checked; do not add panicking twins or `try_` aliases.
- Put each operation's contract on its default entry point. Batch methods
  link to that contract and explain their limit behavior. Examples use
  `literal`, `and`, `or` and related free functions, and name the shared vtree
  `vtree`.
- Keep the vtree text format and shared constructors/accessors compatible
  with [vitri](https://github.com/Tractables/vitri).

[Architecture](docs/architecture.md) documents module responsibilities and
the representation invariants.
