<!-- scenario: docs/scenarios.md#architecture -->

# Architecture

This reference describes the storage, passes, and invariants used by the
implementation. For library use, start with the [API overview]; for the meaning
of levels and pairs, read the [data model].

## Storage and operations

A diagram owns its level arenas and shares a vtree. Operations consume or
borrow those diagrams according to their signatures and use an engine for
scratch and limits. The diagram's dirty worklists record which levels need
reduction after an edit; they do not contribute to its Boolean meaning.

`LevelStorage` records the output for which a completed structural minimization
established canonical form. Mutable level access or vtree reseating forgets that
guarantee; rotation rollback restores it with the original storage. Builders
start without it. Boolean queries borrow certified inputs and otherwise minimize
a private copy. Empty worklists alone do not establish canonical form.

`diagram::Assembly` owns unfinished output levels and their weights, returning
the arenas to the pool if construction fails. Coordinated edits in
`diagram/tdd/edit.rs` keep a level rewrite or slot renumbering together with
its reference updates and reduction work. `MarginalStorage` coordinates count
and weighted payload changes with their slot metadata without changing the
compact level layout. Conjunction's `Products` owns dense grids, sparse lists,
conversions and the metadata that makes each representation readable. A pooled
conjunction workspace owns these products and the sweep's other temporary buffers;
algorithms borrow them, and guards return them on success or error. Sparse-level
scratch uses independent checkouts, so nested calls do not borrow active storage.

Relation construction separates variable layout, row normalization and atom
assembly. The engine caches the last validated layout by vtree allocation and
column order, holding only a weak vtree reference. Rows and constructed diagrams
are never retained by that cache.

Structural levels hold pair lists. Marginal levels hold counts or fixed
weighted values, and their references may carry an inline value instead of
a node index. The `diagram` module owns that encoding; other modules read it
through `ChildDecoder`.

Vtree constructors and grafts finish through one checked node-list constructor,
which derives parent links and returns the node permutation used by graft layouts.
`TopoOrder` owns and validates the traversal order, inverse positions and exact
leaf/internal subsequences; `Vtree::validate` checks links and variable tables.

## Glossary

| Term | Meaning |
|---|---|
| **diagram** | A [`Tdd`] value, owning level storage and an output reference. |
| **level** | One vtree node's storage in a diagram ([`TddLevel`]). |
| **node** | One function at a level, addressed by [`NodeIdx`]. |
| **pair** | One `(left, right)` element of a node's decomposition, holding two [`EncodedChildRef`] words decoded through [`ChildDecoder`]. |
| **slot count** | The number of stored node or value entries at a level, including tombstones; structural leaves store no slots. |
| **marginal level** | A level whose structure was summed out into per-node values. |
| **marginalize** | Replace a level's structure by per-node values. |
| **value** | The per-node payload of a marginal level: a count, or a weight. |
| **value slot** | An index into a marginal level's value store; nodes may share one. |
| **algebra** | The domain a fold computes in: the zero, the leaf values, the sum and the product an [`EvalAlgebra`] supplies. |
| **arithmetic** | Which numeric representation a store's values use ([`Arithmetic`]): exact rationals, or the signed log domain. |
| **cell** | A position in the apply product grid. |
| **column** | A per-node value array for one level during a bottom-up fold. |
| **fold** | One bottom-up pass computing a value per node. |
| **frontier** | The levels a fold currently holds columns for. |
| **output** | The result diagram, or its root reference ([`Tdd::output`]). |
| **limit** | A bound an operation runs under, installed on the engine. |

## Invariants

The numbers below are used by the invariant checkers and source comments.
Structural determinism assumes valid TDD input; storage validation in the
builder and reader does not establish it for an arbitrary circuit.

| # | Statement | Established by | Transiently broken by | Decided by |
|---|---|---|---|---|
| 1 | Structural determinism: distinct nodes at one level compute disjoint functions; each child pair belongs to at most one node. | apply's emit; projection's owner-set regroup | [`and_exists`]'s subtree collapse, until that regroup runs | `test_helpers::check::check_determinism` |
| 2 | Every live node in a structural diagram is satisfiable; ⊥ is the output sentinel only. | checked builder; apply's emit; conditioning's falsity sweep | conditioning's leaf rewrite, within one call | `test_helpers::check::check_no_false_nodes` |
| 3 | Content uniqueness, structural diagrams: no two stored nodes at one level have equal pair multisets. | [`reduce::minimize`] | any apply or marginalization | `test_helpers::check::check_canonicity` |
| 4 | Reachability: every live stored node is reachable from the output. | [`reduce::minimize`] | conditioning, restriction | `test_helpers::check_minimize_soundness` |
| 5 | Marginality is permanent and downward-closed: a marginal level never becomes structural, and its descendants are marginal or leaves whose contribution is absorbed. | [`marginal::marginalize_levels`] | — | [`reduce`]'s demarginalization guard |
| 6 | Every reference into a marginal child decodes through [`ChildDecoder`]; no site outside `diagram/` reads the raw bits. | the marginal-reference encoding | — | review |
| 7 | Inline discipline: no value slot referenced from a structural parent holds an inline-eligible value. | the reference tagger, then the slot prune | apply's emit, before tagging | `test_helpers::check::marginal::check_inline_discipline` |
| 8 | Pair-fusion saturation: no eligible same-structural-child group remains (exact arithmetic; weighted leaf sums must fit the pinned column). | [`marginalize_levels`]'s fusion sweep | a later twin merge | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 9 | Twin canonicality, marginal diagrams: no two nodes at one level have equal pair multisets. | twin contraction | pair fusion | `test_helpers::check::marginal::check_marginal_canonical_form` |
| 10 | Value-slot uniqueness: at a marginal level all stored values are pairwise distinct. | mint-time dedup, then the slot prune | apply's emit | `test_helpers::check::marginal::check_slot_count_uniqueness` |
| 11 | Weighted leaf column pin: a weight-marginal leaf's three slots are an immutable, label-ordered cache of `WeightStore::leaf_val`. No pass compacts, erases, reorders or appends to the column, and every reader re-derives it through `diagram::leaf_column_vals`. | `marginal::marginalize_leaf_weighted` | — | `test_helpers::check::marginal::check_leaf_columns_pinned` |

Invariants 3, 4, 8, 9 and 10 are post-pass properties; each row identifies the
pass that establishes it. Canonical structural form additionally requires
that no context twins remain: these are nodes with identical uses by their
parents, rather than identical pair lists. The distinction matters because
twin contraction unions their functions, while content deduplication merges
identical representations.

## Modules

| Module | Responsibility |
|---|---|
| [`vtree`] | Variable grouping, traversal orders, construction and vtree edits. |
| [`diagram`] | Diagram storage, child references, weights and level pools. |
| [`limits`] | Resource limits, cancellation, memory hooks, measurements and scratch pools. |
| `value` | Count arithmetic, value domains, column storage and the shared value traversal. |
| `build` | Constants, literals, cubes and sets of models. |
| [`apply`] | Boolean composition, conditioning, projection and restriction. |
| `marginal` | Summing levels into values and restoring marginal invariants. |
| [`maintain`] | Adding and removing one assignment at a time, in place. |
| [`reduce`] | Reachability pruning, twin contraction, pair fusion and value-slot pruning. |
| [`restructure`] | Vtree search, grafting and embedding with the corresponding diagram edits. |
| [`query`] | Counting, satisfiability, evaluation and traversal of borrowed diagrams. |
| [`execution`] | Shared context checkouts and each batch's scratch and limits. |
| [`io`] | Diagram persistence and Graphviz output. |
| [`guide`] | Markdown guides included in rustdoc and doctests. |
| `test_helpers` | Formula generators, independent oracles and invariant checkers. |

Operation implementations live with their algorithms, including methods on
`Engine`. Storage modules expose the representation; operations use those
accessors rather than decoding references themselves. Queries leave borrowed
diagrams unchanged.

`test_helpers::check` is compiled under `cfg(test)`, `debug_assertions`, or
the `testing` feature. The feature keeps invariant checks active in release
integration tests without enabling debug assertions in the kernels. Run the
differential suite in both profiles.

## Counting boundaries

`CountVec` owns count columns; `CountRef` borrows them. Both decode stored
values through `CountRead`, which hides the overflow side table from arithmetic.
Query folds and streaming apply retain their specialized fixed-width loops,
then use `IntFold::sum_exact` when a product, sum, or child value exceeds the
fast representation. The exact fold receives decoded values from its readers,
so it handles arithmetic without knowing whether a value came from a query
column, marginal storage, or an inline reference.

`query::cache::Observations` tracks observations and schedules affected
ancestors for both `ModelCounter` and `Evaluator`. The shared refresh walk
uses their respective folds; counting retains its native-integer overflow
path. Query state has no reference to its circuit: `Counter<D>` and
`Evaluation<S, D>` store either a borrow or the owned `Tdd` alongside it.
The borrowed aliases are `ModelCounter` and `Evaluator`; owned queries can be
moved and stored without self-reference. Python and C use those owned forms.
An evaluator owns its algebra so replacing it can invalidate all columns.

`restructure::EmbeddingPlan` retains its source and destination vtrees and the
validated level correspondence. Reusing it skips variable and shape validation;
composition builds a direct placement without intermediate circuit copies.
Embedding and grafting assemble through `restructure::placement::Placement`,
which owns destination levels and weights, preserves local indices during
transfer, and repairs references introduced by new joins. Shape and weight
compatibility checks remain in the respective operations.

`reduce::driver` owns pass ordering, content-twin rescan policy and marginal
boundary cleanup. Kernels report leaf rewrites, merged nodes and merged value
levels; the driver chooses their follow-up passes. Contraction keeps its local
sibling fixed point and restores pending work on refusal.

Composition counts the consumers of each intermediate circuit. `SharedCircuit`
keeps its storage until the last consumer takes it, cloning fallibly for earlier
uses. Substitution releases child columns after their parent is built and only
materializes leaf labels referenced by the source. Replacement functions are
not recursively substituted.

## One conjunction

[`and(f, g)`] borrows an engine from the shared context, validates operand
compatibility, and walks levels bottom-up. At each structural level it combines operand
nodes in a product grid and emits surviving child pairs. Consumed level
arenas return to the engine's pools for reuse.

A conjunction that is also quantifying an existential
([`and_exists`]) can decline to build a subtree every leaf of which is
quantified: that subtree contributes one satisfiability bit, so the level is
decided by an early-exiting test per product cell and written as the single ⊤
node. The result is then the product with those subtrees replaced by ⊤, which
is what the quantification sweep would have reached there; the sweep's
owner-set regroup re-establishes the level-wide disjointness the collapse
breaks. It is `Quantification::FusedSubtrees` and not the default, because it
reaches the level by a walk of the whole product grid and so gives up whatever
cheaper route that level had.

The result has the correct function and count but may retain unreachable
nodes and context twins. A full reduction first prunes, then contracts inner
and leaf twins. For marginal diagrams it also runs eligible content-twin and
value-slot cleanup. Edits register their effects through `Tdd::invalidate`;
reduction drains the corresponding dirty worklists.

Resource refusal is not an implicit rollback of a whole operation. Consuming
operations return an error without their operands; in-place passes document
which completed edits remain valid. Reserve-before-mutation boundaries must
preserve those contracts.

## Extending the implementation

Callers can supply an [`EvalAlgebra`], a [`RotationObjective`], or a stopping
callback through [`LimitConfig`].

Inside the crate, add reduction rules beside the related pass, mark affected
levels dirty, and extend the invariant checkers. Query folds implement
`query::fold::LevelFold`; value arithmetic implements `ValueDomain`, with
`marginal::transition::MarginalDomain` handling storage transitions. A new
resource limit needs a setting in `LimitConfig` and a check at the appropriate
poll or allocation boundary.

## Oracles

| Oracle | Path | When |
|---|---|---|
| Fast invariants | `test_helpers::check::check_all_fast` | After any operation, on any size. |
| Minimize round-trip | `test_helpers::check_minimize_soundness` | On small structural diagrams; it minimizes. |
| Marginal invariants | `test_helpers::check::marginal` | After marginalization or a reduction pass. |
| Brute-force count | `test_helpers::brute_force_count` | Small formulas, to confirm a count. |
| Round trip | [`io`] | To confirm a diagram survives text. |
| Differential fold | [`query`] | Fast and exact counts must agree. |

## Shared execution state

Each vtree carries an `Arc<Context>` separately from its shape and traversal
tables. Public Boolean functions and diagram methods clone that context
handle, borrow an engine from its idle pool, and call the operation kernels. Kernels pass the
borrowed engine through nested work so one operation retains its limits and
metering. The pool moves a boxed engine and holds no mutex during computation.
Nested or concurrent checkouts use separate engines; only one idle engine is
retained. A completed checkout clears configuration and callback references,
and an unwinding checkout discards its scratch.

Vtree clones and projections retain their context. Grafts retain a context
shared by all source vtrees; otherwise they start fresh. Binary compatibility
still compares vtree allocations. A rotation wrapper retains only the context
handle so it does not force extra copy-on-write clones of the vtree. Serialized
vtrees contain shape alone and receive fresh execution state when loaded.

## Constraints

The crate is pure Rust, with no build script. It reads no environment variables
and owns no threads or process-wide state. Its one optional feature, `testing`,
adds the `test_helpers` module — oracles, generators and the invariant checkers
listed below — which a release build otherwise does not compile.

[`and(f, g)`]: crate::and
[`and_exists`]: crate::and_exists
[`Arithmetic`]: crate::diagram::Arithmetic
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`OperationError`]: crate::OperationError
[`LimitConfig`]: crate::limits::LimitConfig
[`NodeIdx`]: crate::diagram::NodeIdx
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`ChildDecoder`]: crate::diagram::ChildDecoder
[`Tdd`]: crate::Tdd
[`Tdd::output`]: crate::Tdd::output
[`TddLevel`]: crate::diagram::TddLevel
[`WeightStore`]: crate::diagram::WeightStore
[`apply`]: crate::apply
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

[`EncodedChildRef`]: crate::diagram::EncodedChildRef

[API overview]: crate::guide::api
[data model]: crate::guide::model
