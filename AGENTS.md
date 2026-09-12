# Agent Instructions

The source repository of `tididi`, a Rust library for Tree Decision Diagrams:
one crate, no workspace members, every tracked file public.
[`CONTRIBUTING.md`](CONTRIBUTING.md) binds in full; this is its short form.

## Checks

Run all four before reporting a change done. `--all-targets` is not optional;
the lib target alone misses the integration tests and the examples.

```sh
cargo test --all-targets
cargo test --doc
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

The crate is not rustfmt-formatted, and a pull request must not
reformat it.

The README and every file of `docs/` are part of the crate documentation, so
their code fences are doctests and the items they name are intra-doc links.
`docs/api-guide.md` is the surface a change keeps true: it is doctested, so a
guide that drifts from the API fails the build.

## Comments

`CONTRIBUTING.md` states five rules in full: one contract sentence per
function; a `# Soundness` block only where correctness turns on a fact
invisible in the body; performance rationale in one checkable sentence;
emphasis by sentence structure, so no all-caps words in prose and no naming of
benchmark instances, external tools, or commits; a long argument moved to the
owning module's `//!` doc or to `docs/architecture.md`. No measurements, and
never an explanation by contrast with a version that is gone.
`tests/comment_lint.rs` enforces the mechanical part over every non-test file
under `src/`; its prose allowlists are empty and stay empty.

## Tests

A test that builds a diagram calls `assert_canonical` on it. Canonicity is the
central invariant, and a query answer can be right while the structure is not;
a change to minimization, fingerprinting, or the node tables needs a test that
pins the canonical form. A bug fix comes with a regression test that fails on
the parent commit. Fixed seeds, no wall-clock timing, no external binaries.

`tests/differential.rs` is the randomized differential suite, and it is the
intended first stop after any change to `apply`, `reduce`, `marginal`, `io` or
`value`. It draws a small formula and a vtree and holds every answer — the
model count, three operation orders against each other, each operation's truth
table, the marginalized count, the text round trip, both weighted arithmetics,
and a budget too small for the work — to enumeration, so it decides cases the
fixed corpus has none of. It is ignored by default because it runs until its
time is up:

```sh
cargo test --release --test differential -- --ignored
```

`TIDIDI_FUZZ_SECONDS` sets how long a run draws for and `TIDIDI_FUZZ_SEED` the
stream it draws from; with the seed unset the run takes one from the clock, so
two unattended runs cover different cases. The seed is printed before the first
case, and a failure prints it again with the claim that broke, the formula, the
vtree, and a line that replays the case. Run it once more with
debug assertions on, which is where the invariant checkers are compiled and
where the structural half of the suite decides anything. A test may read the
environment; the library still may not.

## Invariants and modules

`docs/architecture.md` is the reference: the data model, the numbered invariant
list every checker and comment cites, and a table saying what each module owns.
`comment_lint` requires a table row for every `pub mod` in `src/lib.rs`, so a
new public module updates the table in the same commit.

## Constraints

No cargo features, no `build.rs`, no environment reads, no process-wide state,
no threads, no C or C++ dependencies. Runtime configuration is installed data:
limits and memory probes arrive on an `Engine` the caller owns, through
`LimitConfig`. Vtree heuristics and CNF handling are out of scope.

## Commits

Messages are publication-grade: an imperative subject naming the behaviour
changed, a body saying what was wrong and what stands now. No tool footers, no
co-author trailers, no session links. Documentation changes with behaviour.
