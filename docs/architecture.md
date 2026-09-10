# Architecture

## Model

A diagram ([`Tdd`]) is an `Arc<Vtree>`, one [`TddLevel`] per vtree node, and an
output reference naming a node at the vtree root. A level is a leaf level
(implicit: it stores nothing), a structural level (its nodes are lists of
`(left, right)` pairs), or a marginal level (its structure has been summed out
and it stores one value per node instead). Every operation reads and writes
levels; nothing outside `diagram/` decodes a reference by hand.

## Glossary

| Term | Meaning |
|---|---|
| **diagram** | A [`Tdd`] value. The word used in prose; [`Tdd`] appears only as a type. |
| **level** | One vtree node's storage in a diagram ([`TddLevel`]). |
| **node** | One function at a level, addressed by [`NodeIdx`]. |
| **pair** | One `(left, right)` element of a node's decomposition. |
| **width** | The number of nodes at a level. |
| **marginal level** | A level whose structure was summed out into per-node values. |
| **marginalize** | Replace a level's structure by per-node values. |
| **value** | The per-node payload of a marginal level: a count, or a weight. |
| **value slot** | An index into a marginal level's value store; nodes may share one. |
| **cell** | A position in the apply product grid. That use only. |
| **column** | A per-node value array for one level during a bottom-up fold. |
| **fold** | One bottom-up pass computing a value per node. |
| **frontier** | The levels a fold currently holds columns for. |
| **output** | The result diagram, or its root reference ([`Tdd::output`]). |
| **limit** | A bound an operation runs under, installed on the engine. |

## Invariants

The numbered list. Every checker and every comment cites these numbers.

| # | Statement | Established by | Transiently broken by | Decided by |
|---|---|---|---|---|
| 1 | Determinism: distinct nodes at one level compute disjoint functions. | apply's emit | — | `check::check_canonicity` |
| 2 | No node computes ⊥; ⊥ is the output sentinel only. | apply's emit | — | `check::check_no_false_nodes` |
| 3 | Canonicity: no two nodes at one level are content-equal. | [`reduce::minimize`] | any apply or marginalize | `check::check_canonicity` |
| 4 | Reachability: every stored node is reachable from the output. | [`reduce::minimize`] | conditioning, restriction | `check::check_minimize_soundness` |
| 5 | Marginality is permanent and downward-closed: a marginal level never becomes structural, and every descendant of a marginal level is marginal. | [`marginal::marginalize`] | — | [`reduce`]'s demarginalization guard |
| 6 | Every reference into a marginal child decodes through [`SideView`]; no site outside `diagram/` reads the raw bits. | the marginal-reference encoding | — | review |
| 7 | Inline discipline: no value slot referenced from a structural parent holds an inline-eligible value. | the reference tagger, then the slot prune | apply's emit, before tagging | `check::marginal::check_inline_discipline` |
| 8 | Pair-fusion saturation: within a parent node, no two pairs share a structural-side child. | [`marginalize`]'s fusion sweep | a later twin merge | `check::marginal::check_no_fusion_redexes` |
| 9 | Twin canonicality: no two nodes at one level have equal pair multisets. | twin contraction | pair fusion | `check::marginal::check_no_twins` |
| 10 | Value-slot uniqueness: at a marginal level all stored values are pairwise distinct. | mint-time dedup, then the slot prune | apply's emit | `check::marginal::check_slot_count_uniqueness` |
| 11 | Weighted leaf column pin: a weight-marginal leaf's three slots are an immutable, label-ordered cache of `WeightStore::leaf_val`. No pass compacts, erases, reorders or appends to the column, and every reader re-derives it through `marginal::leaf_column_vals`. | `marginal::marginalize_leaf_weighted` | — | `marginal::debug_check_leaf_columns_pinned` |

Invariants 3, 4, 8, 9 and 10 are post-pass properties, not properties of every
intermediate state; each row says which pass establishes it.

## Modules

| Module | Owns | May not touch |
|---|---|---|
| [`vtree`] | The variable tree, its orders, its text format, rotation and graft of the tree itself. | Diagram storage. |
| [`diagram`] | Levels, nodes, pairs, the reference encodings, the level pool, weights. | Any operation's algorithm. |
| [`build`] | Constants, literals and clauses as diagrams. | Reduction. |
| [`apply`] | Conjunction, disjunction, negation, conditioning, projection, restriction. | Reference decoding by hand; reduction policy. |
| [`marginal`] | Summing levels out, the schedule that orders it, and the epilogue restoring invariants 7, 8 and 10. | The reduction passes' internals. |
| [`reduce`] | Canonical form: pruning, twin contraction, pair fusion, slot pruning. | Apply. |
| [`restructure`] | Rotation search and graft over a compiled diagram. | The counting fold. |
| [`query`] | Model counting, satisfiability, algebra evaluation, size metrics. | Mutation of a diagram. |
| [`io`] | The `.tdd` text format, both directions, and Graphviz rendering. | Anything but reading a finished diagram. |
| [`engine`] | The session: limits, memory probes, meters, scratch pools. | The diagram's contents. |
| [`operators`] | The `&`, `\|` and `!` impls for [`Tdd`]. | Anything beyond delegating to [`apply`]. |
| `value_fold` | The one bottom-up walk and the two value domains folded over it. Internal to the crate. | Which levels to fold. |
| [`error`] | The error types. | — |
| [`guide`] | The prose guides of `docs/`, included as documentation so their examples and their identifiers are checked by the build. | Any behaviour; it holds no code. |
| `check` | The invariant checkers, one per numbered invariant, compiled only under `cfg(test)` or `debug_assertions`. The debug-facing module. | Repair; a checker reports and never rewrites. |
| `compiler_seam` | Clause-spine marking and mid-compile clustering, for a driver that builds a diagram clause by clause. The driver-facing module, outside the compatibility promise. | The documented modules' jobs; it holds hooks, not operations. |

`check` and `compiler_seam` are `#[doc(hidden)]`: the first is debug-only
validation, the second the seam a clause-by-clause driver compiles against.

## One conjunction

[`Engine::and(f, g)`] plans the operation, walks the vtree bottom-up (children
before parents), and at each level runs the product-grid kernel over the two
operands' nodes, emitting the pairs that survive. The emit establishes
invariants 1 and 2, so the result needs no dedup pass. Marginal sides are
tagged after the walk, which is what invariant 7 is stated against. Reduction
is a separate call.

## Extension points: public

Implementable from outside the crate, against the published API:

- A new read-only value domain: implement [`EvalAlgebra`].
- A new rotation objective: implement [`RotationObjective`].
- A new stopping rule the caller decides: the schedule hook on [`LimitSet`],
  answered at every poll the running operation reaches.

## Internal seams

For a contributor working inside the crate. None of these is a published
extension point, and none is reachable from outside:

- A new apply shape: a variant of `ApplyPlan`.
- A new reduction rule: add it beside the rule it resembles — twin
  contraction in `reduce/contract/`, content twins in
  `reduce/content_twins.rs`, pair fusion in `reduce/contract/pair_fusion/`,
  leaf twins in `reduce/contract/contract_leaf.rs` — mark its dirty levels,
  and add a checker for the invariant it claims.
- A new marginalizable value domain: a [`WeightStore`] plus a `ValueDomain`.
- A new fold: implement `ValueDomain` and use the shared walk.
- A new order for the contraction pass to visit dirty levels in: a walk
  beside the ones in `reduce/contract/strategies.rs`.
- A new limit: a field on [`LimitSet`] and the poll site that reads it.

## Oracles

| Oracle | Path | When |
|---|---|---|
| Fast invariants | `check::check_all_fast` | After any operation, on any size. |
| Deep invariants | `check::check_all_deep` | On small structural diagrams; it minimizes. |
| Marginal invariants | `check::marginal` | After marginalize or a reduction pass. |
| Brute-force count | the test helpers | Small formulas, to confirm a count. |
| Round trip | [`io`] | To confirm a diagram survives text. |
| Differential fold | [`query::count`] | Fast and exact counts must agree. |

## Constraints

No cargo features, no `build.rs`, no environment reads, no threads, no
process-wide state, no C or C++ code built.

[`Engine::and(f, g)`]: crate::Engine::and
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`LimitSet`]: crate::engine::LimitSet
[`NodeIdx`]: crate::diagram::NodeIdx
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`SideView`]: crate::diagram::SideView
[`Tdd`]: crate::Tdd
[`Tdd::output`]: crate::Tdd::output
[`TddLevel`]: crate::diagram::TddLevel
[`WeightStore`]: crate::diagram::WeightStore
[`apply`]: crate::apply
[`build`]: crate::build
[`diagram`]: crate::diagram
[`engine`]: crate::engine
[`error`]: crate::error
[`guide`]: crate::guide
[`io`]: crate::io
[`marginal`]: crate::marginal
[`marginal::marginalize`]: crate::marginal::marginalize
[`marginalize`]: crate::marginal::marginalize
[`operators`]: crate::operators
[`query`]: crate::query
[`query::count`]: crate::query::count
[`reduce`]: crate::reduce
[`reduce::minimize`]: crate::reduce::minimize
[`restructure`]: crate::restructure
[`vtree`]: crate::vtree
