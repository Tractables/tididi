# Agent Instructions — tididi

This is the source repository for `tididi`, a Rust library for Tree Decision
Diagrams: construction, apply operations, canonical minimization, vtree
restructuring, and queries (SAT, model counting, weighted counting, semiring
evaluation). It is a single flat crate — `Cargo.toml`, `src/`, `tests/`,
`docs/` — with no workspace members. Every tracked file and commit here is
public.

**[`CONTRIBUTING.md`](CONTRIBUTING.md) binds in full** — the gate set, the
code and test rules, the doc and commit conventions. Operational notes on top
of it:

- Run the whole gate set before reporting a change done, and run tests with
  `--all-targets` — the lib target alone misses the integration tests and the
  README usage example in `tests/readme_example.rs`.
- For a bug fix, confirm the new regression test fails on the unfixed parent
  commit before writing the fix.
- Vtree *construction* heuristics and DIMACS-CNF compilation are out of scope
  here; they live in companion projects. Do not add a CNF parser, a solver
  driver, or a benchmark harness to this crate.
- The crate's constraints are load-bearing, not stylistic: no cargo features,
  no `build.rs`, no environment reads, no threads, no C/C++ dependencies.
  A change that needs one of those needs a design discussion first.
- Canonicity is the crate's central invariant. A change to minimization,
  fingerprinting, or the node tables needs a test that pins the canonical form,
  not just the query answer.
- User-facing behaviour lives in `README.md` and `docs/`; when you change
  behaviour, change its documentation in the same commit.
