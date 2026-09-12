# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

### Added

- `Tdd`, a Tree Decision Diagram over an `Arc<Vtree>`, with base diagrams
  (`Tdd::one`, `Tdd::zero`, `Tdd::clause`, `Engine::cube`) and vtrees built
  from leaves and joins, from a balanced, linear or random shape, from a node
  list, or read from the `.vtree` text format.
- Boolean combination through the `&` and `|` operators and their fallible
  `Engine` forms, negation through `!`, and a clause-stream conjunction.
- Transformations: conditioning, existential quantification, restriction to a
  care set, and grafting one diagram's vtree region onto another.
- `minimize`, which reduces a diagram to the canonical form for its vtree, so
  two diagrams of one function over one vtree are identical up to the order
  of nodes within a level; rotation search rotates the vtree under a diagram
  toward any objective over the levels a rotation rewrites.
- Queries: model counting, an incremental counter, satisfiability and implied
  literals of a minimized diagram, and semiring evaluation over user-supplied
  algebras, including rational weights and signed logarithms. A level whose
  structure is no longer needed can be summed out into per-node counts, or
  into per-node weights held in a `WeightStore`, to bound memory.
- Examples: clauses to a count, a DIMACS file to a count with a saved diagram,
  a projection and a weighted count, and one statistic read off the stored
  encoding.
- A public stored encoding, documented for direct traversal, with a text
  file format, a builder for constructing diagrams from outside the crate,
  and DOT rendering for diagrams and vtrees.
- A format version in the `.tdd` problem line (`p tdd 1 …`): a file written by
  version n loads in every reader whose own version is n or greater, and a
  reader refuses a newer file naming both versions.
- Caller-owned resource control: an `Engine` carrying a `LimitSet` of
  byte-budget, output-cap, deadline and schedule limits, returning an error
  rather than aborting; the
  byte budget is best effort and may be overrun by up to the size of the
  diagram an operation builds. The
  crate has no cargo features, no build script, reads no environment
  variables, and spawns no threads.
- `Engine::project_var`, `Engine::project_vars`, `Engine::condition_var` and
  `Engine::condition_vars` return `ApplyError::VariableNotInVtree` for a
  variable the vtree does not carry; the free functions of the same names
  panic.
