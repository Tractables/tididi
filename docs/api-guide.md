<!-- scenario: docs/scenarios.md#api-overview -->

# API overview

A [`Tdd`](crate::Tdd) represents a Boolean function. Start with the
[configuration example](crate::guide::examples::configurations) to build and
query a circuit; use this page to find an operation. Each item documents its
own contract and examples.

## Build and transform circuits

Create a [`Vtree`](crate::Vtree) and share one `Arc<Vtree>` among circuits you
intend to combine. [`literal`](crate::literal) builds an atom,
[`Tdd::clause`](crate::Tdd::clause) an OR of literals,
[`Tdd::cube`](crate::Tdd::cube) an AND of literals, and
[`Tdd::one`](crate::Tdd::one) and [`Tdd::zero`](crate::Tdd::zero) the
constants. A [`VarId`](crate::vtree::VarId) names a variable without a sign,
for quantification and renaming.

To build a circuit from a table of assignments, use
[`Tdd::from_models`](crate::Tdd::from_models) for bit-packed rows,
[`Tdd::from_cubes`](crate::Tdd::from_cubes) for rows that leave variables free,
or [`Tdd::from_columns`](crate::Tdd::from_columns) for columns of codes, the
shape [`Tdd::model_columns`](crate::Tdd::model_columns) writes.
[`Tdd::maintain`](crate::Tdd::maintain) inserts and removes rows in place; the
[`maintain`](crate::maintain) module gives the cost.

Combine circuits with [`and`](crate::and), [`or`](crate::or),
[`xor`](crate::xor), [`ite`](crate::ite) and [`Tdd::negate`](crate::Tdd::negate).
[`and_not`](crate::and_not) removes one circuit's solutions from another
without building a complement. [`Tdd::and_clause`](crate::Tdd::and_clause)
conjoins a clause and [`Tdd::or_cube`](crate::Tdd::or_cube) disjoins a cube
without building a second circuit. [`add`](crate::add) sums two count diagrams,
such as results of [`Tdd::marginalize_levels`](crate::Tdd::marginalize_levels).
These operations consume their operands; [`Tdd`](crate::Tdd) explains when to
clone.

## Query solutions

| Question | Operation |
|---|---|
| Is there a solution? | [`Tdd::is_sat`](crate::Tdd::is_sat) |
| What is one complete solution? | [`Tdd::satisfying_assignment`](crate::Tdd::satisfying_assignment) |
| How many assignments satisfy the function? | [`Tdd::model_count`](crate::Tdd::model_count) |
| How many satisfy a conjunction, without building it? | [`Engine::and_model_count`](crate::Engine::and_model_count) |
| What is the sum of two marginalized diagrams' products of counts? | [`Engine::marginal_product_count`](crate::Engine::marginal_product_count) |
| How many distinct choices are possible for selected variables? | [`Tdd::projected_model_count`](crate::Tdd::projected_model_count) |
| Which assignments of one subtree have at least `m` distinct extensions in its sibling? | [`Engine::at_least_distinct`](crate::Engine::at_least_distinct) |
| Which choices are forced? | [`Tdd::implied_literals`](crate::Tdd::implied_literals) |
| Which variables affect the function? | [`Tdd::support`](crate::Tdd::support) |
| Do two functions agree? Does one imply the other? | [`Tdd::equivalent`](crate::Tdd::equivalent), [`Tdd::implies`](crate::Tdd::implies) |

For changing observations, [`Tdd::counter`](crate::Tdd::counter) keeps counting
state between calls to [`observe`](crate::query::ModelCounter::observe), and
[`Tdd::into_counter`](crate::Tdd::into_counter) gives the counter ownership of
the circuit. [`count_table`](crate::query::ModelCounter::count_table) lists the
count under each assignment of a few variables;
[`Counter::prepare`](crate::query::Counter::prepare) builds compact storage for
repeated counts.

## Condition, quantify and rename

The [counting tutorial](crate::guide::examples::counting) compares observation,
substitution and projection. Conjoin an observation to keep the assignments
consistent with it; [`Tdd::condition`](crate::Tdd::condition) substitutes its
values into the function instead.

[`Tdd::exists_vars`](crate::Tdd::exists_vars) eliminates variables,
[`Tdd::forall_vars`](crate::Tdd::forall_vars) requires every value of them, and
[`and_exists`](crate::and_exists) conjoins and eliminates in one operation.
[`Tdd::project_to_vars`](crate::Tdd::project_to_vars) also removes the
eliminated variables from the vtree, and
[`Engine::project_to_subtree`](crate::Engine::project_to_subtree) keeps one
subtree's variables. [`Tdd::rename_vars`](crate::Tdd::rename_vars) renames
variables simultaneously; [`Tdd::substitute`](crate::Tdd::substitute) replaces
them with functions. The [reachability example](crate::guide::examples::reachability)
uses quantification and renaming to compute successor states.

## Evaluate probabilities and costs

[`Tdd::evaluate`](crate::Tdd::evaluate) computes weighted sums from literal
weights, as in the [probability example](crate::guide::examples::probability).
Implement [`EvalAlgebra`](crate::diagram::EvalAlgebra) for another quantity,
such as the [minimum configuration cost](crate::guide::examples::optimization),
or [`ColumnAlgebra`](crate::diagram::ColumnAlgebra) for an algebra that keeps
its own storage, run through [`Tdd::evaluate_columns`](crate::Tdd::evaluate_columns).
[`Tdd::evaluate_at`](crate::Tdd::evaluate_at) and
[`Tdd::evaluate_columns_at`](crate::Tdd::evaluate_columns_at) give the values of
one level's nodes. An algebra that values counts
([`EvalAlgebra::count`](crate::diagram::EvalAlgebra::count)) also evaluates
diagrams with summed-out levels, such as
[`Engine::and_marginalizing`](crate::Engine::and_marginalizing)'s result.
[`Tdd::set_weights`](crate::Tdd::set_weights) attaches weights to a diagram and
[`Tdd::weighted_value`](crate::Tdd::weighted_value) reads its value.

## Save and inspect circuits

The [persistence example](crate::guide::examples::persistence) saves diagrams
with [`write_tdd`](crate::io::write_tdd), restores them with
[`read_tdd`](crate::io::read_tdd), and stores the vtree through
[`Vtree::to_text`](crate::Vtree::to_text) and [`Vtree::from_text`](crate::Vtree::from_text).
[`write_tdd_binary`](crate::io::write_tdd_binary) and
[`read_tdd_binary`](crate::io::read_tdd_binary) are the compact binary
counterparts. [`Tdd::attach_level_counts`](crate::Tdd::attach_level_counts)
keeps every level's model counts with a diagram for later counts and the binary
format to reuse.

[`tdd_to_dot`](crate::io::tdd_to_dot) and [`vtree_to_dot`](crate::io::vtree_to_dot)
render Graphviz text; [`Tdd::level_sizes_to_dot`](crate::Tdd::level_sizes_to_dot)
annotates the vtree with circuit sizes. The
[custom-statistic example](crate::guide::examples::statistics) traverses nodes
and pairs directly.

## Choose a representation

The [vtree example](crate::guide::examples::vtrees) compares storage under two
variable groupings. [`Vtree::join`](crate::Vtree::join) specifies groups and
[`Vtree::linear`](crate::Vtree::linear) a variable order.
[`Tdd::minimize`](crate::Tdd::minimize) removes redundancy under the current
vtree, and [`Tdd::pair_count`](crate::Tdd::pair_count) measures storage.
[`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for a smaller
vtree; [`Engine::rotation_search_with`](crate::Engine::rotation_search_with)
changes its acceptance rule,
[`Engine::rotation_multistart`](crate::Engine::rotation_multistart) restarts it
from perturbed copies, and [`Engine::pool_search`](crate::Engine::pool_search)
searches one vtree shared by several diagrams.

For more specialized control:

| Purpose | API |
|---|---|
| Select reduction passes | [`Tdd::reduce`](crate::Tdd::reduce), [`ReductionPlan`](crate::reduce::ReductionPlan) |
| Simplify within a care set | [`Tdd::restrict_to_care`](crate::Tdd::restrict_to_care); [worked example](crate::guide::examples::care) |
| Remove internal nodes a predicate rejects | [`Tdd::filter_nodes`](crate::Tdd::filter_nodes), [`FilterOutcome`](crate::apply::FilterOutcome) |
| Encode nodes as clauses for a SAT solver | [`Tdd::encode_cnf`](crate::Tdd::encode_cnf), [`ClauseSink`](crate::cnf::ClauseSink) |
| Find the encoded nodes the solver proves impossible | [`Tdd::probe_nodes`](crate::Tdd::probe_nodes), [`SatOracle`](crate::cnf::SatOracle) |
| Replace structure with counts or weighted values | [`Tdd::marginalize_levels`](crate::Tdd::marginalize_levels); [worked example](crate::guide::examples::marginalization) |
| Conjoin diagrams over disjoint variables | [`Tdd::graft`](crate::Tdd::graft), [`Tdd::graft_over`](crate::Tdd::graft_over) |
| Place a circuit on a larger vtree under a renaming | [`Tdd::embed`](crate::Tdd::embed); [reusable components](crate::guide::examples::composition) |
| Move a circuit onto another vtree over the same variables | [`Tdd::restructure_to`](crate::Tdd::restructure_to) |
| Expand each variable into a class of new variables | [`Tdd::expand_variables`](crate::Tdd::expand_variables), [`Vtree::expand_leaves`](crate::Vtree::expand_leaves) |
| Build a vtree from its leaf order and the depths between leaves | [`Vtree::from_in_order`](crate::Vtree::from_in_order) |
| Place a diagram with summed-out levels | [`Engine::embed_over`](crate::Engine::embed_over) |
| Conjoin or place by moving levels, getting the operands back on refusal | [`Engine::and_restoring`](crate::Engine::and_restoring), [`Engine::embed_moving`](crate::Engine::embed_moving) |
| Conjoin two diagrams placed on a larger vtree | [`Engine::and_onto`](crate::Engine::and_onto) |
| Place or conjoin onto a vtree with mirrored subtrees | [`Engine::embed_moving_mirrored`](crate::Engine::embed_moving_mirrored), [`Engine::and_onto_mirrored`](crate::Engine::and_onto_mirrored) |
| Conjoin a small constraint into a large diagram's own levels | [`Engine::and_in_place`](crate::Engine::and_in_place), [`Engine::and_in_place_loose`](crate::Engine::and_in_place_loose), [`Engine::and_cube`](crate::Engine::and_cube) |
| Conjoin with, or sum against, a large diagram through an index of its pairs | [`PairIndex`](crate::diagram::PairIndex), [`Engine::and_indexed`](crate::Engine::and_indexed), [`Engine::sum_indexed`](crate::Engine::sum_indexed) |
| Pair each node of a level with a code | [`Engine::tag_level`](crate::Engine::tag_level) |
| Pair each point with a running sum, maximum or rank of keyed values | [`Engine::prefix_fold`](crate::Engine::prefix_fold), [`FoldMonoid`](crate::apply::FoldMonoid), [`FoldRange`](crate::apply::FoldRange) |
| Number each level's nodes in top-down walk order | [`Tdd::renumber_top_down`](crate::Tdd::renumber_top_down) |
| Assemble levels and pairs directly | [`TddBuilder`](crate::diagram::TddBuilder), started with [`Tdd::builder`](crate::Tdd::builder) |

The [data model](crate::guide::model) explains levels, pairs and determinism;
the [architecture reference](crate::guide::architecture) describes the implementation.

## Limit work and release scratch

[`Context::with_limits`](crate::Context::with_limits) runs a batch under a
[`LimitConfig`](crate::limits::LimitConfig) through the engine it supplies;
[`Context::run`](crate::Context::run) reuses scratch without limits. The
[execution example](crate::guide::examples::execution) handles a refused
allocation and bounds repeated queries.
[`Engine::try_clone`](crate::Engine::try_clone) copies a diagram under the
engine's limits, refusing a copy the allocator cannot serve.
[`LimitConfig::with_sparse_route`](crate::limits::LimitConfig::with_sparse_route)
sets when conjunction levels take the sparse route, and
[`LimitConfig::with_deadline_at_most`](crate::limits::LimitConfig::with_deadline_at_most)
adds a deadline to the bounds already set.

[`Context::bind`](crate::Context::bind) lets vtrees share scratch;
[`Context::clear_scratch`](crate::Context::clear_scratch) releases idle buffers.
