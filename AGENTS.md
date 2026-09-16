# Agent instructions

- Run the [contribution checks](CONTRIBUTING.md#checks) before reporting work
  complete; include `--all-targets` and the separate doctest run.
- Do not reformat the crate with rustfmt.
- Public docs describe current behavior. Keep benchmark results, development
  history and implementation debates out of them.
- Document contracts on the API items. Guides introduce tasks and link to
  those items instead of repeating their specifications.
- Put unit tests in a nearby `tests/` directory; for a standalone `foo.rs`,
  use `tests/foo.rs` (or `tests/foo/` for several files) with a `#[path]` module
  declaration. Implementation directories must contain implementation code. Integration tests go in
  the crate-root `tests/`. Check constructed diagrams with
  `test_helpers::assert_canonical`; use fixed seeds and no timing assertions.
- Update the module table in `docs/architecture.md` when adding a public module.
  Keep representation invariants there and reference them from code.
- `tests/comment_lint.rs` checks source comments and test placement. Fix
  violations rather than adding prose allowlists.
- Commit messages must be suitable for publication: no tool footers, session
  links or machine-specific details.
