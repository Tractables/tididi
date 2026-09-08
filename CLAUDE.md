# Agent Instructions

This is the source repository of `tididi`, a Rust library for Tree Decision
Diagrams. It is a single crate (`Cargo.toml`, `src/`, `tests/`, `examples/`,
`docs/`) with no workspace members. Every tracked file and commit here is
public.

[`CONTRIBUTING.md`](CONTRIBUTING.md) binds in full: the check set, the code
and test rules, the doc and commit conventions. On top of it:

- Run the whole check set before reporting a change done, with
  `--all-targets`: the lib target alone misses the integration tests, the
  examples, and the README example in `tests/readme_example.rs`.
- For a bug fix, confirm the new regression test fails on the unfixed parent
  commit before writing the fix.
- Vtree construction heuristics and CNF handling are out of scope. Do not
  add a CNF parser, a solver driver, or a benchmark harness to this crate.
- The crate's constraints are load-bearing: no cargo features, no
  `build.rs`, no environment reads, no process-wide state, no threads, no
  C/C++ dependencies. A change that needs one of those needs a design
  discussion first.
- Canonicity is the crate's central invariant. A change to minimization,
  fingerprinting, or the node tables needs a test that pins the canonical
  form, not just the query answer.
- The `vtree` module is a hand-maintained mirror of the `vitri` crate's
  vtree module; port a change to one side to the other.
- User-facing behaviour lives in `README.md` and `docs/`. When you change
  behaviour, change its documentation in the same commit, and keep every
  identifier the guides name resolvable in `src/`.
