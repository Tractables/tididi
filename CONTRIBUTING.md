# Contributing

## Checks

Before opening a pull request, run:

```sh
cargo test --all-targets && cargo test --doc
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo fmt --check
```

CI runs the same commands on the toolchain pinned in `rust-toolchain.toml` and
builds on the MSRV declared in `Cargo.toml`.

## Code

- The crate has no cargo features, no `build.rs`, and no C or C++
  dependencies. It reads no environment variables: runtime configuration
  arrives as installed data (`tdd::config`, `tdd::mem_pressure`). Keep it that
  way — a new knob is a field on an existing config type, not a feature flag or
  an env read.
- The library spawns no threads. Callers run many instances in parallel, so a
  global mutable cache or a thread pool is not an option.
- Invalid caller input returns an error that names the input. Library code
  panics only on internal invariants.
- Prefer extending an existing type, table, or helper over standing up a
  parallel one. Two code paths that do the same job diverge.
- Public items carry rustdoc that says what is guaranteed, including the vtree
  and canonicity preconditions an operation assumes.

## Tests

- Unit tests that need a module's private items live beside it as
  `<module>_tests.rs` or `<module>/tests.rs`; end-to-end tests live in
  `tests/`. Production files contain no `#[cfg(test)]` code other than the
  `mod tests;` line.
- A test name states the fact being checked, one fact per test.
- Fixed seeds; no wall-clock timing, sleeps, or external binaries.
- Fixtures are small and generated in-tree.
- A bug fix comes with a regression test that fails on the parent commit, in
  the same commit.

## Docs and commits

- `README.md` and `docs/` state what exists, what to watch out for, and what is
  guaranteed. No numbers that go stale (test counts, runtimes, diagram sizes).
- When you change behaviour, change its documentation in the same commit.
- Commit subjects are imperative and describe the behaviour changed, e.g.
  "Reject a conditioning literal outside the vtree".
