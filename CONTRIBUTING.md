# Contributing

## Checks

Before opening a pull request, run:

```sh
cargo test --all-targets && cargo test --doc
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo fmt --check
```

CI runs the same commands on the toolchain pinned in `rust-toolchain.toml`,
which is also the `rust-version` declared in `Cargo.toml`.

## Code

- The crate has no cargo features, no `build.rs`, and no C or C++
  dependencies. It reads no environment variables and holds no process-wide
  state: limits and memory probes are installed on an `Engine` the caller
  owns, through `LimitSet`. A new knob is an axis on that builder or a field
  on an existing options type, not a feature flag or an environment read.
- The library spawns no threads. Callers run many instances in parallel, so
  a global mutable cache or a thread pool is not an option.
- Invalid caller input returns an error that names the input (`VtreeError`,
  `TddBuildError`, `ApplyError`). Library code panics only on internal
  invariants.
- Prefer extending an existing type, table, or helper over standing up a
  parallel one. Two code paths that do the same job diverge.
- Public items carry rustdoc that says what is guaranteed, including the
  vtree and canonicity preconditions an operation assumes. Items that exist
  only for a downstream driver or for tests are `#[doc(hidden)]`.
- The `vtree` module mirrors the vtree module of the `vitri` crate by hand:
  the `.vtree` text format and every constructor and accessor the two share
  keep the same name and behaviour, and a change to a shared item is ported
  to the other side in the same change.

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
- `tests/readme_example.rs` is the README example; a change to one is a
  change to the other.

## Docs and commits

- `README.md` and `docs/` state what exists and what is guaranteed. One fact
  per sentence; no numbers that go stale (test counts, runtimes, diagram
  sizes).
- Every identifier named in a guide exists in `src/`.
- When you change behaviour, change its documentation in the same commit.
- Commit subjects are imperative and describe the behaviour changed, e.g.
  "Reject a conditioning literal outside the vtree".
