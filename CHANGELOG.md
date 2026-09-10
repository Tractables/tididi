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
- Caller-owned resource control: an `Engine` carrying a `LimitSet` of node,
  memory and deadline limits, returning an error rather than aborting. The
  crate has no cargo features, no build script, reads no environment
  variables, and spawns no threads.
