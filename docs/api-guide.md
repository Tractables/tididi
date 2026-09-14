# Using tididi

Start with a vtree and an engine, build a function, then borrow the resulting
diagram to ask questions about it. The sections below follow that sequence;
the linked API items provide the contracts and examples.

## Build a Boolean function

Choose [`Vtree::balanced`](crate::Vtree::balanced) for a first experiment,
[`Vtree::linear`](crate::Vtree::linear) for a variable order, or
[`Vtree::join`](crate::Vtree::join) to assemble your own grouping.
The [`Vtree`](crate::Vtree) introduction explains how variables are numbered
and how operands share a tree.

Build an atom with [`Engine::literal`](crate::Engine::literal), a disjunction
of literals with [`Engine::clause`](crate::Engine::clause), or a conjunction
of literals with [`Engine::cube`](crate::Engine::cube).
Compose functions with [`Engine::and`](crate::Engine::and),
[`Engine::or`](crate::Engine::or), [`Engine::negate`](crate::Engine::negate),
[`Engine::xor`](crate::Engine::xor), or [`Engine::ite`](crate::Engine::ite).
For a sequence of constraints, [`Engine::and_clause`](crate::Engine::and_clause)
adds a clause to an existing diagram.
The ownership example on [`Tdd`](crate::Tdd) shows how to retain an operand
for more than one transformation.

## Ask questions about a function

[`Engine::is_sat`](crate::Engine::is_sat) answers whether any assignment satisfies the function.
[`Engine::model_count`](crate::Engine::model_count) counts satisfying assignments,
including choices for free variables.
To obtain an assignment itself, use
[`Engine::satisfying_assignment`](crate::Engine::satisfying_assignment).
Test semantic equality with [`Engine::equivalent`](crate::Engine::equivalent),
or whether one function entails another with [`Engine::implies`](crate::Engine::implies).

For information about individual variables, [`Engine::support`](crate::Engine::support)
finds those that can affect the answer, while
[`Engine::implied_literals`](crate::Engine::implied_literals) finds literals true in every model.

## Change assignments or variables

An observation may be used to derive a new function or to answer another
query about the original one.
[`Engine::condition`](crate::Engine::condition) substitutes a fixed assignment
and explains how the resulting cofactor is counted.
If you want repeated counts under observations without rewriting the diagram,
use [`ModelCounter`](crate::query::ModelCounter) with
[`PinSemantics::Evidence`](crate::query::PinSemantics::Evidence).

Use [`Engine::exists_vars`](crate::Engine::exists_vars) to retain assignments
that have at least one satisfying extension, or
[`Engine::and_exists`](crate::Engine::and_exists) to compute successor states
from a transition relation.
[`Engine::rename_vars`](crate::Engine::rename_vars) then handles variable
identification, swaps, and current/next-state renaming.
For replacements that are whole functions, use
[`Engine::substitute`](crate::Engine::substitute).
[`Engine::restrict_to_care`](crate::Engine::restrict_to_care) simplifies a
function where only assignments in a care set matter.

## Evaluate probabilities or repeated observations

[`Engine::evaluate`](crate::Engine::evaluate) shows the dependency setup and
use of [`RationalWeights`](crate::diagram::RationalWeights) to evaluate a
structural diagram under independent-variable probabilities.
Implement [`EvalAlgebra`](crate::diagram::EvalAlgebra) to compute a different
quantity, such as the fewest true variables in a model.

For fixed weights that travel with a diagram, attach a
[`WeightStore`](crate::diagram::WeightStore) through
[`Tdd::set_weights`](crate::Tdd::set_weights) and read its result with
[`Engine::weighted_value`](crate::Engine::weighted_value).
The complete [probabilistic query example](https://github.com/Tractables/tididi/blob/main/examples/probabilistic_query.rs)
shows repeated evaluation and conditional-probability normalization.

## Control resources and diagram size

Configure limits through [`LimitConfig`](crate::limits::LimitConfig), install
them for a block with [`Limits::scope`](crate::limits::Limits::scope), and
handle refusals as [`OperationError`](crate::OperationError).
[`Limits::work_since`](crate::limits::Limits::work_since) measures the work
performed by a group of operations.

Use [`try_minimize`](crate::reduce::try_minimize) for canonical form under
the current vtree, or [`minimize`](crate::reduce::minimize) for its convenience form.
[`try_reduce`](crate::reduce::try_reduce) and [`ReductionPlan`](crate::reduce::ReductionPlan)
let advanced callers select individual passes.
[`Engine::rotation_search`](crate::Engine::rotation_search) searches different
vtree shapes when the current representation is too large.
When only counts or fixed weighted values are still needed,
[`marginalize_levels`](crate::marginal::marginalize_levels) can release structure
permanently.

## Save or inspect a diagram

A saved function needs its variable tree to recover the same interpretation.
[`write_tdd`](crate::io::write_tdd) and [`read_tdd`](crate::io::read_tdd) save
and restore a structural diagram, with its tree stored separately by
[`Vtree::to_text`](crate::Vtree::to_text) and [`Vtree::from_text`](crate::Vtree::from_text).
Render Graphviz text with [`tdd_to_dot`](crate::io::tdd_to_dot) or
[`vtree_to_dot`](crate::io::vtree_to_dot).

The [`diagram`](crate::diagram) module introduces traversal before showing how
to decode marginal values.
Use [`TddBuilder`](crate::diagram::TddBuilder) to assemble nodes yourself,
or [`Tdd::graft`](crate::Tdd::graft) and [`Tdd::graft_over`](crate::Tdd::graft_over)
to combine diagrams with disjoint variable domains.

## Read further

The [data model](https://docs.rs/tididi/latest/tididi/guide/model/index.html)
introduces levels, pairs, and determinism; the
[architecture reference](https://docs.rs/tididi/latest/tididi/guide/architecture/index.html)
explains the passes and invariants for contributors.
