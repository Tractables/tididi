<!-- scenario: docs/scenarios.md#api-overview -->

# API overview

A [`Tdd`](crate::Tdd) represents a Boolean function. Start with the
[configuration example](crate::guide::examples::configurations) to build a
circuit and query it; use this page to find related operations. Each API item
has its own examples and contract.

## Build and transform circuits

Create a [`Vtree`](crate::Vtree) and share one `Arc<Vtree>` among circuits you
intend to combine. [`literal`](crate::literal) builds an atom;
[`Tdd::clause`](crate::Tdd::clause) joins literals with OR, and
[`Tdd::cube`](crate::Tdd::cube) joins them with AND;
[`Tdd::one`](crate::Tdd::one) and [`Tdd::zero`](crate::Tdd::zero) are the
constants. Integer literals are signed and start at 1; a
[`VarId`](crate::vtree::VarId), used to quantify and rename, carries the same
number without the sign. [`Tdd::from_models`](crate::Tdd::from_models) builds
the canonical circuit for a whole table of assignments at once, from bit-packed
rows over a chosen list of variables.

Combine circuits with [`and`](crate::and), [`or`](crate::or),
[`xor`](crate::xor) and [`Tdd::negate`](crate::Tdd::negate), or use
[`ite`](crate::ite) to choose between two branches.
[`Tdd::and_clause`](crate::Tdd::and_clause) adds a clause to an existing
circuit and [`Tdd::or_cube`](crate::Tdd::or_cube) adds a conjunction of
literals to its solutions, neither building a second circuit.
These transformations consume their operands; [`Tdd`](crate::Tdd) explains
when to clone a circuit you want to keep.

When a table behind a circuit gains or loses a row,
[`Tdd::maintain`](crate::Tdd::maintain) edits the circuit in place rather
than rebuilding it: see the [`maintain`](crate::maintain) module for what
one assignment costs and when the edit applies.

## Query solutions

| Question | Operation |
|---|---|
| Is there a solution? | [`Tdd::is_sat`](crate::Tdd::is_sat) |
| What is one complete solution? | [`Tdd::satisfying_assignment`](crate::Tdd::satisfying_assignment) |
| How many assignments satisfy the function? | [`Tdd::model_count`](crate::Tdd::model_count) |
| How many distinct choices are possible for selected variables? | [`Tdd::projected_model_count`](crate::Tdd::projected_model_count) |
| Which choices are forced? | [`Tdd::implied_literals`](crate::Tdd::implied_literals) |
| Which variables affect the function? | [`Tdd::support`](crate::Tdd::support) |
| Do two functions agree? Does one imply the other? | [`Tdd::equivalent`](crate::Tdd::equivalent), [`Tdd::implies`](crate::Tdd::implies) |

For changing observations, [`Tdd::counter`](crate::Tdd::counter) keeps counting
state between calls to [`observe`](crate::query::ModelCounter::observe);
[`Tdd::into_counter`](crate::Tdd::into_counter) moves the circuit into that state
when the counter needs to own it.
The [configuration example](crate::guide::examples::configurations) uses this
for a user's changing selections.

## Condition, quantify and rename

The [counting tutorial](crate::guide::examples::counting) compares observation,
substitution and projection on one model.

Conjoin an observation to retain the assignments consistent with it;
[`Tdd::condition`](crate::Tdd::condition) substitutes its values into the
function and documents how that affects counting.

[`Tdd::exists_vars`](crate::Tdd::exists_vars) eliminates variables by keeping
assignments that have a satisfying extension; [`and_exists`](crate::and_exists)
combines this with conjunction.
[`Tdd::rename_vars`](crate::Tdd::rename_vars) renames variables simultaneously,
while [`Tdd::substitute`](crate::Tdd::substitute) replaces them with functions.
The [reachability example](crate::guide::examples::reachability) uses
quantification and renaming to compute successor states.

## Evaluate probabilities and costs

Supply literal weights to [`Tdd::evaluate`](crate::Tdd::evaluate) to compute
weighted sums, as in the [probability example](crate::guide::examples::probability).
Implement [`EvalAlgebra`](crate::diagram::EvalAlgebra) to calculate another
quantity, such as the [minimum configuration cost](crate::guide::examples::optimization).
For weights attached to the diagram, use [`Tdd::set_weights`](crate::Tdd::set_weights)
and [`Tdd::weighted_value`](crate::Tdd::weighted_value).

## Save and inspect circuits

The [persistence example](crate::guide::examples::persistence) saves diagrams
with [`write_tdd`](crate::io::write_tdd), restores them with
[`read_tdd`](crate::io::read_tdd), and stores the vtree through
[`Vtree::to_text`](crate::Vtree::to_text) and [`Vtree::from_text`](crate::Vtree::from_text).

Render Graphviz text with [`tdd_to_dot`](crate::io::tdd_to_dot) or
[`vtree_to_dot`](crate::io::vtree_to_dot); [`Tdd::vtree_to_dot`](crate::Tdd::vtree_to_dot)
annotates the vtree with circuit sizes.
For direct traversal, follow the [custom-statistic example](crate::guide::examples::statistics).

## Choose a representation

The [vtree example](crate::guide::examples::vtrees) compares storage for two
variable groupings. Use [`Vtree::join`](crate::Vtree::join) to specify groups
or [`Vtree::linear`](crate::Vtree::linear) for a variable order.
[`Tdd::minimize`](crate::Tdd::minimize) removes redundancy under the current vtree
and says which operations already return minimized results;
[`Tdd::pair_count`](crate::Tdd::pair_count) measures the storage, and
[`Tdd::rotation_search`](crate::Tdd::rotation_search) explores other shapes;
[`Engine::rotation_search_with`](crate::Engine::rotation_search_with) runs the
same search under another acceptance policy, and
[`Engine::rotation_multistart`](crate::Engine::rotation_multistart) restarts it
from perturbed copies.

For more specialized control:

| Purpose | API |
|---|---|
| Select reduction passes | [`Tdd::reduce`](crate::Tdd::reduce), [`ReductionPlan`](crate::reduce::ReductionPlan) |
| Simplify within a care set | [`Tdd::restrict_to_care`](crate::Tdd::restrict_to_care); [worked example](crate::guide::examples::care) |
| Remove internal nodes chosen by a predicate | [`Tdd::filter_nodes`](crate::Tdd::filter_nodes), [`FilterOutcome`](crate::apply::FilterOutcome) |
| Replace structure with counts or fixed weighted values | [`Tdd::marginalize_levels`](crate::Tdd::marginalize_levels); [worked example](crate::guide::examples::marginalization) |
| Combine disjoint variable domains | [`Tdd::graft`](crate::Tdd::graft), [`Tdd::graft_over`](crate::Tdd::graft_over) |
| Place a circuit on a larger vtree under a renaming | [`Tdd::embed`](crate::Tdd::embed); [reusable components](crate::guide::examples::composition) |
| Assemble levels and pairs directly | [`TddBuilder`](crate::diagram::TddBuilder), started with [`Tdd::builder`](crate::Tdd::builder) |

The [data model](crate::guide::model) explains levels, pairs and determinism;
the [architecture reference](crate::guide::architecture) describes the implementation.

## Limit work and release scratch

[`Context::with_limits`](crate::Context::with_limits) supplies an engine with a
[`LimitConfig`](crate::limits::LimitConfig) for a batch; use that engine throughout
the bounded work. [`Context::run`](crate::Context::run) reuses scratch without
installing limits. The [execution example](crate::guide::examples::execution)
shows how to handle a refused allocation and bound repeated queries.

[`Context::bind`](crate::Context::bind) lets vtrees share scratch;
[`Context::clear_scratch`](crate::Context::clear_scratch) releases idle buffers.
