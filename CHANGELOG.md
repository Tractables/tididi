# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

### Added

- `Tdd`, a Tree Decision Diagram over an `Arc<Vtree>`, with base diagrams
  (`Tdd::one`, `Tdd::zero`, `Tdd::clause`) and vtrees built from leaves and
  joins, from a balanced or linear shape, or read from the `.vtree` text
  format.
- Boolean combination through the `&`, `|` and `!` operators and their
  fallible `Engine` forms, a clause-stream conjunction, and a batched
  conjunction into a large accumulator.
- Transformations: conditioning, existential quantification, restriction to a
  care set, and grafting one diagram's vtree region onto another.
- `minimize`, which reduces a diagram to the canonical form for its vtree, so
  two diagrams of one function over one vtree are identical; rotation search
  restructures a diagram toward a smaller vtree.
- Queries: model counting, an incremental counter, and semiring evaluation
  over user-supplied algebras, including rational weights and signed
  logarithms. A level whose structure is no longer needed can be summed out
  into per-node counts to bound memory.
- A public stored encoding, documented for direct traversal, with a binary
  file format, a builder for constructing diagrams from outside the crate,
  and DOT rendering for diagrams and vtrees.
- A format version in the `.tdd` problem line (`p tdd 1 …`): a file written by
  version n loads in every reader whose own version is n or greater, and a
  reader refuses a newer file naming both versions.
- Caller-owned resource control: an `Engine` carrying a `LimitSet` of node,
  memory and deadline limits, returning an error rather than aborting. The
  crate has no cargo features, no build script, reads no environment
  variables, and spawns no threads.

### Changed

- `Engine::project_var`, `Engine::project_vars`, `Engine::condition_var` and
  `Engine::condition_vars` return `ApplyError::VariableNotInVtree` when the
  request names a variable the operand's vtree does not carry. That input used
  to abort the process, which no caller could catch. The free functions of the
  same names stay infallible and document the panic.
- `save_tdd` and `load_tdd` take any `AsRef<Path>`, so a `PathBuf` goes in as
  it stands rather than through `to_str().unwrap()`.

### Fixed

- A `&[i32]` of DIMACS literals and a `&[Literal]` build a clause, as the guide
  says they do: `Literal` now converts from a reference as well as a value, so
  a clause read off a file goes into `Tdd::clause` and `Engine::cube` without a
  conversion pass.
- A clause naming one variable in both polarities is the tautology it spells.
  `Tdd::clause`, `Engine::clause`, `apply_and_clause` and `Engine::and_clause`
  read a clause's literals as a set, so such a clause builds ⊤ and conjoining
  it is the identity; each used to answer a different function, silently.
- Conditioning returns a canonical diagram. A node whose every pair belonged to
  the cofactor that was conditioned away is dropped and its falsity propagated
  to the parents that named it, so no node left in the result computes ⊥. The
  model count was already right; the structure was not, and re-conjoining such
  a result revived models the conditioning had removed.
- A signed-logarithm weighted evaluation answers a number. Two magnitudes of
  opposite sign that differ by less than the `f64` spacing cancel to zero, and
  that zero is now written canonically, so a later addition no longer yields
  `NaN`. Multiplication answers zero when a factor is zero or the product
  underflows.
