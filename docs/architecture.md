<!-- scenario: docs/scenarios.md#architecture -->

# Architecture

This page is for people changing the implementation. It explains how a
diagram is laid out in memory, what happens when an operation runs, and which
properties every pass must preserve. Read the [data model] first: it explains
vtrees, levels, nodes and pairs as mathematical objects. For using the library,
start with the [API overview].

## How a diagram is stored

A [`Tdd`] value holds:

- **its vtree**, an `Arc<Vtree>` shared with every diagram it may be combined
  with. Binary operations compare the allocation, not just the shape, so two
  separately built but identical vtrees are incompatible;
- **its levels**, one [`TddLevel`] per vtree node, stored at that node's index;
- **its output**, a reference to the node that denotes the whole function, or
  a sentinel for the constant false function ([`Tdd::is_zero`]);
- **dirty worklists**, which record the levels reduction still has to revisit.
  They do not affect the function the diagram denotes;
- **weights**, optionally, for diagrams whose summed-out levels carry weighted
  values instead of counts.

A structural level stores its nodes in arenas. A node is addressed by a
[`NodeIdx`] and lists its pairs; a pair holds two child references, one into
the left child's level and one into the right child's, each an
[`EncodedChildRef`] word. A leaf level stores nothing: its functions `x`,
`¬x` and `true` follow from the vtree, so a reference names one of them
directly. Vtree node indices stay fixed when the vtree is rotated, so array
order is not a traversal order; walk levels with [`Vtree::bottomup`] to visit
children before parents.

A *marginal level* stores a value per node instead of pairs, and a reference
into it may carry that value inline, so code outside the `diagram` module
decodes references through [`ChildDecoder`]; see
[Marginal levels](#marginal-levels).

Vtree constructors and grafts all finish through one checked constructor,
which derives parent links and returns the node permutation that grafting uses
to place levels. `TopoOrder` owns and validates the traversal order and its
inverse, and `Vtree::validate` checks the links and variable tables.

## How an operation runs

Follow one conjunction, [`and(f, g)`]:

1. **Borrow an engine.** Every vtree carries an `Arc<Context>`, separate from
   its shape. A public function borrows an *engine* from that context: the
   engine holds reusable scratch buffers and the limits for this call. Kernels
   pass the same engine down to nested work, so limits and metering cover the
   whole operation.
2. **Check the operands.** Both must share the vtree allocation, and the
   operation must accept their kinds of level.
3. **Build the result bottom-up.** At each structural level, the conjunction
   forms a *product grid*: each *cell* pairs a node of `f` with a node of `g`,
   and its pairs are the products of the two nodes' pairs, using the products
   already built at the child levels. Cells that are false are dropped; the
   rest become nodes of the result. `Products` holds this intermediate state,
   as a dense grid or as sparse lists, together with the metadata that makes
   each form readable. An `Assembly` owns the unfinished output levels and
   their weights and returns the arenas to the pool if the operation fails.
4. **Recycle.** The operands' level arenas, which the conjunction consumed,
   return to the engine's pools for the next operation.
5. **Reduce.** The result already has the right function and count, but it
   may hold nodes the output no longer reaches and *twins*: nodes that every
   parent uses with exactly the same siblings. A full reduction prunes the
   unreachable nodes and then contracts inner and leaf twins, as described in
   the next section.

[`and_exists`] can go further when a whole subtree is quantified away: that
subtree contributes only whether it is satisfiable, so the level can be decided
by an early-exiting test per product cell and written as the single `true`
node. This is what the quantification sweep would have reached there, and the
sweep's regroup by owner set restores the level-wide disjointness that the
collapse breaks. It is `Quantification::FusedSubtrees` rather than the
default, because it walks the whole product grid at that level and so gives up
any cheaper route the level had.

Edits that change levels after construction go through `diagram/tdd/edit.rs`,
which keeps a level rewrite or a renumbering of slots together with the
updates of the references into it and with the reduction work it creates.

## Reduction and canonical form

Each edit records which levels it affected (`Tdd::invalidate`); reduction
drains those worklists. `reduce::driver` decides the order of passes: it
prunes unreachable nodes, contracts twins, and for marginal diagrams also
merges duplicate nodes and removes unused value slots. Each pass reports what
it changed (leaf rewrites, merged nodes, merged value levels), and the driver
chooses the follow-up passes from that report. Contraction repeats among
siblings until nothing changes, and restores its pending work if it is
refused.

Twin contraction differs from removing duplicate nodes. Twins are nodes that
their parents use in the same way, though their own functions may differ; the
diagram can replace them by their union. Duplicates have identical pair lists.
A minimized structural diagram has neither, and is then canonical for its
vtree.

`LevelStorage` remembers the output for which a completed structural
minimization established canonical form. Taking mutable access to a level or
reseating the vtree forgets it; rolling back a rotation restores it together
with the original storage; builders start without it. Empty worklists alone do
not prove canonical form. Boolean queries such as equivalence borrow an input
that carries this record and minimize a private copy of one that does not.

## Memory, scratch and limits

Operations reuse buffers from pools held by the engine instead of allocating
afresh each time. A pooled conjunction workspace owns the `Products` and the sweep's
other temporary buffers; algorithms borrow them, and guards return them on
success or error. Sparse-level scratch uses independent checkouts, so a nested
call never borrows storage that an outer call is still using.

Every pool reports how much capacity it keeps to one engine-wide allowance,
which includes nested buffers and recycled diagram levels. Checking a buffer
out removes its claim; returning it keeps the buffer only if it still fits.
This bound on idle memory is separate from the memory budget an operation is
charged for its growth ([`LimitConfig`]), and it ignores allocator overhead.

The context's idle pool hands out a boxed engine and holds no lock while the
engine computes. Nested or concurrent checkouts get separate engines, and only
one idle engine is kept. A finished checkout clears its configuration and
callbacks; one that ends by panicking discards its scratch. Vtree clones and
projections keep their context. A graft keeps a context that all its source
vtrees share and otherwise starts a fresh one. Rotation keeps only the context
handle, so it does not force a copy of the vtree. A vtree read from text
starts with a fresh context.

Building a diagram from a table of rows has three separate steps: variable
layout, row normalization and assembly of the result. The engine caches the
last validated layout, keyed by vtree allocation and column order, with only a
weak reference to the vtree; it never keeps the rows or the diagram.

When a limit refuses work, nothing is rolled back implicitly. An operation
that consumes its operands returns an error without them, so a caller who may
retry passes clones. A pass that edits in place documents which of its
completed edits remain valid. Code that reserves memory before mutating must
keep both contracts.

## Marginal levels

[`Tdd::marginalize_levels`] replaces the structure of a subtree by one value
per node: a model count, or a weighted value when the diagram has a
[`WeightStore`]. Nodes of a marginal level point to *value slots*, and nodes
with equal values share a slot. A reference from a structural parent into a
marginal child may carry its value inline instead of a slot index; the
`diagram` module owns that encoding and every other module decodes through
[`ChildDecoder`]. `MarginalStorage` keeps count and weighted payload changes
consistent with the slot metadata without changing the compact level layout.

Marginal diagrams have extra reduction passes. *Pair fusion* merges a node's
pairs that share the same structural child into one pair carrying the summed
value; twin contraction and slot pruning then restore uniqueness. The
invariants table below lists what each pass establishes.

## Queries and counting

Counts can outgrow a machine word. `CountVec` owns count columns and
`CountRef` borrows them; both decode stored values through `CountRead`, which
hides the side table that holds overflowing values. Query folds and streaming
conjunction keep their fixed-width loops and fall back to `IntFold::sum_exact`
when a product, sum or child value no longer fits. The exact fold receives
decoded values, so it does not need to know whether a value came from a query
column, marginal storage or an inline reference.

A counter or evaluator keeps one *column* of values per level and recomputes
only what an observation changes. `query::cache::QueryCache` holds the columns
and observations of both `ModelCounter` and `Evaluator` and refreshes only the
ancestors of changed observations. Each kind of query supplies its fold, how
to read the output and its answer for the false diagram through
`query::cache::CachedQuery`; counting keeps its native-integer overflow path.
Query state holds no reference into its circuit: `Counter<D>` and
`Evaluation<S, D>` store either a borrow of the `Tdd` or the owned `Tdd`.
`ModelCounter` and `Evaluator` are the borrowing forms; the owning forms can be
moved and stored, and the Python and C bindings use them. An evaluator owns its
algebra, so replacing the algebra invalidates every column.

## Placing diagrams on other vtrees

`restructure::EmbeddingPlan` keeps a source vtree, a destination vtree and the
validated correspondence between their levels, so reusing it skips validation,
and composing plans places a diagram directly without intermediate copies.
Embedding assembles its result through `restructure::placement::CopyPlacement`
and grafting through `MovePlacement`. Both own the destination levels and
weights, keep local node indices while transferring levels, and repair the
references that new joins introduce; each operation still checks shape and
weight compatibility itself.

`Engine::embed_over` moves weight-marginal levels with their values through
`MovePlacement`. A free variable under such a level's image scales its values
by that variable's weight for `true`; an internal level the destination adds
underneath is marginal without values; and a level placed over a marginal
child is contracted at the end, because its nodes differ only by value slot.

Composition counts how many consumers each intermediate circuit has.
`SharedCircuit` keeps its storage until the last consumer takes it and clones
it, fallibly, for earlier ones. Substitution releases child columns once their
parent is built and materializes only the leaf labels the source refers to;
it does not substitute into the replacement functions.

## Modules

| Module | Responsibility |
|---|---|
| [`vtree`] | Variable grouping, traversal orders, construction and vtree edits. |
| [`diagram`] | Diagram storage, child references, weights and level pools. |
| [`limits`] | Resource limits, cancellation, memory hooks and measurements. |
| `value` | Count arithmetic, value domains, column storage and the shared value traversal. |
| `build` | Constants, literals, cubes and sets of models. |
| [`apply`] | Boolean composition, conditioning, projection and restriction. |
| `marginal` | Summing levels into values and restoring marginal invariants. |
| [`maintain`] | Adding and removing one assignment at a time, in place. |
| [`reduce`] | Reachability pruning, twin contraction, pair fusion and value-slot pruning. |
| [`restructure`] | Vtree search, grafting, embedding, variable expansion and moves to a given vtree, with the corresponding diagram edits. |
| [`query`] | Counting, satisfiability, evaluation and traversal of borrowed diagrams. |
| [`cnf`] | Clauses that define a diagram's nodes, for an external SAT solver, and the probe that asks it which nodes are impossible. |
| [`execution`] | Shared context checkouts, scratch pools and each batch's limits. |
| [`io`] | Diagram persistence and Graphviz output. |
| [`guide`] | Markdown guides included in rustdoc and doctests. |
| `test_helpers` | Formula generators, independent oracles and invariant checkers. |

Operations live with their algorithms, including their methods on `Engine`.
Storage modules expose the representation through accessors, and operations
use those accessors rather than decoding references themselves. Queries leave
the diagrams they borrow unchanged.

## Invariants

Each row states a property the code relies on, the pass that makes it true,
the passes during which it may briefly be false, and the checker that tests
it. The numbers are referred to from source comments. These properties assume
a valid TDD: the builder and the reader validate storage, but they do not
prove that an arbitrary circuit's nodes are disjoint (row 1).

| # | Statement | Established by | Transiently broken by | Decided by |
|---|---|---|---|---|
| 1 | Structural determinism: distinct nodes at one level compute disjoint functions; each child pair belongs to at most one node. | apply's emit; projection's owner-set regroup | [`and_exists`]'s subtree collapse, until that regroup runs | `test_helpers::check::check_determinism` |
| 2 | Every live node in a structural diagram is satisfiable; ⊥ is the output sentinel only. | checked builder; apply's emit; conditioning's falsity sweep | conditioning's leaf rewrite, within one call | `test_helpers::check::check_no_false_nodes` |
| 3 | Content uniqueness, structural diagrams: no two stored nodes at one level have equal pair multisets. | [`reduce::minimize`] | any apply or marginalization | `test_helpers::check::check_canonicity` |
| 4 | Reachability: every live stored node is reachable from the output. | [`reduce::minimize`] | conditioning, restriction | none |
| 5 | Marginality is permanent and downward-closed: a marginal level never becomes structural, and its descendants are marginal or leaves whose contribution is absorbed. | [`marginal::marginalize_levels`] | — | [`reduce`]'s demarginalization guard |
| 6 | Every reference into a marginal child decodes through [`ChildDecoder`]; no site outside `diagram/` reads the raw bits. | the marginal-reference encoding | — | review |
| 7 | Inline discipline: no value slot referenced from a structural parent holds an inline-eligible value. | the reference tagger, then the slot prune | apply's emit, before tagging | `test_helpers::check::marginal::check_inline_discipline` |
| 8 | Pair-fusion saturation: no eligible same-structural-child group remains (exact arithmetic; weighted leaf sums must fit the pinned column). | [`marginalize_levels`]'s fusion sweep | a later twin merge | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 9 | Twin canonicality, marginal diagrams: no two nodes at one level have equal pair multisets. | twin contraction | pair fusion | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 10 | Value-slot uniqueness: at a marginal level all stored values are pairwise distinct. | mint-time dedup, then the slot prune | apply's emit | `test_helpers::check::marginal::check_slot_count_uniqueness` |
| 11 | Weighted leaf column pin: a weight-marginal leaf's three slots are an immutable, label-ordered cache of `WeightStore::leaf_val`. No pass compacts, erases, reorders or appends to the column, and every reader re-derives it through `diagram::leaf_column_vals`. | `marginal::marginalize_leaf_weighted` | — | `test_helpers::check::marginal::check_leaf_columns_pinned` |
| 12 | Slot reachability: at a boundary marginal level other than the output's, every value slot is referenced from the structural parent. | [`reduce`]'s slot prune | pair fusion and twin contraction, until the prune runs | `test_helpers::check::marginal::check_no_orphan_slots` |

Rows 3, 4, 8, 9, 10 and 12 hold after a pass rather than throughout. Canonical
structural form also requires that no twins remain, which no row above
covers: rows 3 and 9 forbid duplicate pair lists, while twins are nodes their
parents use identically.

## Testing

| Oracle | Path | When |
|---|---|---|
| Fast invariants | `test_helpers::check::check_all_fast` | After any operation, on any size. |
| Minimize round-trip | `test_helpers::check_minimize_soundness` | On small structural diagrams; it minimizes. |
| Marginal invariants | `test_helpers::check::marginal` | After marginalization or a reduction pass. |
| Brute-force count | `test_helpers::brute_force_count` | Small formulas, to confirm a count. |
| Round trip | [`io`] | To confirm a diagram survives text. |
| Differential fold | [`query`] | Fast and exact counts must agree. |

`test_helpers::check` is compiled in every debug build, because the
`debug_assert` sites in `reduce`, `marginal` and `restructure` call it, and
under the `testing` feature in any profile, which keeps the invariant checks
active in release integration tests. The generators and oracles in
`test_helpers`, and the synthetic byte charge `Limits::charge_in_flight`,
compile only under that feature. Run the differential suite in both profiles.

## Extending the implementation

Callers can supply an [`EvalAlgebra`] (or a [`ColumnAlgebra`], which owns the
per-level storage its values live in), a [`RotationObjective`], or a stopping
callback through [`LimitConfig`].

Inside the crate, add a reduction rule beside the related pass, mark the
levels it affects dirty, and extend the invariant checkers. Query folds
implement `query::fold::LevelFold`; value arithmetic implements `ValueDomain`,
with `marginal::transition::MarginalDomain` handling storage transitions. A new
resource limit needs a setting in `LimitConfig` and a check at the right poll
or allocation point.

## Constraints

The crate is pure Rust, with no build script or native dependencies. It reads
no environment variables and owns no threads or process-wide state. Consumers
need no Cargo features; the one optional feature, `testing`, exposes the
generators and oracles in `test_helpers` and keeps its invariant checkers in a
release build, as described under [Testing](#testing).

## Glossary

| Term | Meaning |
|---|---|
| **diagram** | A [`Tdd`] value, owning level storage and an output reference. |
| **level** | One vtree node's storage in a diagram ([`TddLevel`]). |
| **node** | One function at a level, addressed by [`NodeIdx`]. |
| **pair** | One `(left, right)` element of a node's decomposition, holding two [`EncodedChildRef`] words decoded through [`ChildDecoder`]. |
| **slot count** | The number of stored node or value entries at a level; structural leaves store no slots. |
| **marginal level** | A level whose structure was summed out into per-node values. |
| **marginalize** | Replace a level's structure by per-node values. |
| **value** | The per-node payload of a marginal level: a count, or a weight. |
| **value slot** | An index into a marginal level's value store; nodes may share one. |
| **twin** | A node that every parent uses with the same siblings as another node; reduction merges twins into their union. |
| **algebra** | The domain a fold computes in: the zero, the leaf values, the sum and the product an [`EvalAlgebra`] supplies. |
| **arithmetic** | Which numeric representation a store's values use ([`Arithmetic`]): exact rationals, or the signed log domain. |
| **cell** | A position in the conjunction's product grid: one node of each operand. |
| **column** | A per-node value array for one level during a bottom-up fold. |
| **fold** | One bottom-up pass computing a value per node. |
| **frontier** | The levels a fold currently holds columns for. |
| **output** | The result diagram, or its root reference ([`Tdd::output`]). |
| **limit** | A bound an operation runs under, installed on the engine. |

[`and(f, g)`]: crate::and
[`and_exists`]: crate::and_exists
[`Arithmetic`]: crate::diagram::Arithmetic
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`ColumnAlgebra`]: crate::diagram::ColumnAlgebra
[`LimitConfig`]: crate::limits::LimitConfig
[`NodeIdx`]: crate::diagram::NodeIdx
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`ChildDecoder`]: crate::diagram::ChildDecoder
[`EncodedChildRef`]: crate::diagram::EncodedChildRef
[`Tdd`]: crate::Tdd
[`Tdd::output`]: crate::Tdd::output
[`Tdd::is_zero`]: crate::Tdd::is_zero
[`Tdd::marginalize_levels`]: crate::Tdd::marginalize_levels
[`TddLevel`]: crate::diagram::TddLevel
[`Vtree::bottomup`]: crate::Vtree::bottomup
[`WeightStore`]: crate::diagram::WeightStore
[`apply`]: crate::apply
[`cnf`]: crate::cnf
[`diagram`]: crate::diagram
[`execution`]: crate::execution
[`guide`]: crate::guide
[`io`]: crate::io
[`limits`]: crate::limits
[`maintain`]: crate::maintain
[`marginal::marginalize_levels`]: crate::Tdd::marginalize_levels
[`marginalize_levels`]: crate::Tdd::marginalize_levels
[`query`]: crate::query
[`reduce`]: crate::reduce
[`reduce::minimize`]: crate::Tdd::minimize
[`restructure`]: crate::restructure
[`vtree`]: crate::vtree

[API overview]: crate::guide::api
[data model]: crate::guide::model
