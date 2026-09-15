# Contributing

## Checks

Before opening a pull request, run:

```sh
cargo test --all-targets && cargo test --doc
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

The crate is not rustfmt-formatted, and a pull request must not
reformat it.

CI runs the same commands on the toolchain pinned in `rust-toolchain.toml`,
and builds and tests once more on the `rust-version` declared in
`Cargo.toml`, which is the oldest toolchain the crate supports.

A change that alters the public surface is also compared against the last
released version, so that the version bump it needs is known before it lands:

```sh
cargo install cargo-semver-checks --locked
cargo semver-checks --baseline-rev <the previous release tag>
```

Any revision in the repository works as the baseline, so the check is
available before the first release is tagged: pass the revision the surface
was last agreed at. CI runs it on every pull request and skips it while the
repository carries no tag. The check does not see a change to the return type
of an inherent method — a method that starts returning a `Result` where it
returned an `Option` is a breaking change it reports as none — so a changed
signature is judged by hand as well.

## Code

- `docs/architecture.md` is the reference: the model, the numbered invariant
  list every checker and comment cites, and what each module owns and may not
  touch. Read it before adding a module or an invariant.
- The crate has no cargo features, no `build.rs`, and builds no C or C++
  code. Its own sources carry no platform-specific path, so the file it
  writes on one target is the file it writes on every other. It reads no
  environment variables and holds no process-wide state: limits and memory
  probes are installed on an `Engine` the caller owns, through `LimitConfig`. A
  new knob is an axis on that builder or a field on an existing options type,
  not a feature flag or an environment read.
- The library spawns no threads. Callers run many instances in parallel, so
  a global mutable cache or a thread pool is not an option.
- Invalid caller input returns an error that names the input (`VtreeError`,
  `TddBuildError`, `OperationError`). Library code panics only on internal
  invariants.
- Prefer extending an existing type, table, or helper over standing up a
  parallel one. Two code paths that do the same job diverge.
- Every public type implements `Debug`, and the crate lints for it. Error
  enums also implement `Display` and `std::error::Error`. An enum or options
  struct a caller reads rather than exhausts carries `#[non_exhaustive]`, so a
  new variant or field is an additive release; every such options struct keeps
  a `Default` a caller can start from. `OperationError` is the exception: callers
  mint it, so its variants are the whole set.
- Public items carry rustdoc that says what is guaranteed, including the
  vtree and canonicity preconditions an operation assumes. Items that exist
  only for a downstream driver or for tests are `#[doc(hidden)]`. Five rules
  bound what a comment may say:
  1. Every function gets one contract sentence — what it does or returns, and
     the precondition a caller must hold. A trait-impl method may inherit it
     from the trait.
  2. A `# Soundness` block is allowed only where correctness turns on a fact
     that is not visible in the body. State the invariant relied on and what
     breaking it costs, in at most ten lines.
  3. Performance rationale is one sentence whose claim can be checked against
     the code. No measurements, no comparison with an earlier version.
  4. Emphasis is carried by sentence structure, not by capitalization: no
     all-caps words in prose, and no naming of benchmark instances, external
     tools, or commits.
  5. An argument that does not fit in ten lines belongs in the owning module's
     `//!` doc if it is about that module's data, or in
     `docs/architecture.md` if it is a crate-level invariant — never on a
     helper. A comment never explains the code by contrast with a version that
     is gone.
- The `vtree` module and the [`vitri`](https://github.com/Tractables/vitri) crate share the `.vtree` text format and
  the names and behaviour of every constructor and accessor they have in
  common; a change to a shared item is ported to the other side in the same
  change.

## Tests

- End-to-end tests live in the crate-root `tests/`; every other test file
  lives in a `tests/` directory inside the module it tests, listed by that
  directory's `mod.rs`. A production file carries only the
  `#[cfg(test)] mod tests;` declaration, which `tests/comment_lint.rs`
  enforces; a helper that exists for tests lives with the tests, or in
  `test_helpers`, and one that needs the module's private state is an `impl`
  block in that module's `tests/support.rs`, which sees the state as any
  child module does.
- Test-only fields and conditional hook calls may use `#[cfg(test)]` in their
  owning production type or operation; hook implementations live in `tests/`.
- A test name states the fact being checked, one fact per test.
- Fixed seeds; no wall-clock timing, sleeps, or external binaries.
- Fixtures are small and generated in-tree.
- A bug fix comes with a regression test that fails on the parent commit, in
  the same commit.

## Docs and commits

- `README.md` and `docs/` state what exists and what is guaranteed. One fact
  per sentence; no numbers that go stale (test counts, runtimes, diagram
  sizes).
- `README.md` and every file of `docs/` is included in the crate
  documentation, so their code fences are doctests and the items they name
  are intra-doc links. A guide that drifts from the API fails the build.
  `docs/api-guide.md` is the doctested surface a change to the API keeps true.
- Introductory examples compose diagrams with imported `and`, `or`, `xor`,
  `ite` and `and_exists` functions, and use diagram methods for queries and
  unary transformations. They return `Result` so failures can propagate with
  `?`. Show operators as optional shorthand and explicit batch engines when
  teaching resource control. Put each operation's full contract on its default
  entry point; the batch method links to it and explains its limit behavior.
- When you change behaviour, change its documentation in the same commit.
- Commit subjects are imperative and describe the behaviour changed, e.g.
  "Reject a conditioning literal outside the vtree".
