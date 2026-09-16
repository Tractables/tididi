# API overview

A [`Tdd`](crate::Tdd) is a circuit representing a Boolean function. The operations
below construct and transform these functions, query their satisfying assignments,
and evaluate them under weights. Worked examples show how the operations fit
together in applications; each linked API item gives its full contract.

## Build and transform circuits

### Construct a function

Start with [`Vtree::balanced`](crate::Vtree::balanced), sharing one `Arc<Vtree>`
among functions you intend to combine.
Build atoms with [`literal`](crate::literal), disjunctions of literals
with [`Tdd::clause`](crate::Tdd::clause), and conjunctions with
[`Tdd::cube`](crate::Tdd::cube); [`Tdd::and_clause`](crate::Tdd::and_clause) adds a clause to an existing circuit.
Compose diagrams with [`and`](crate::and), [`or`](crate::or) and
[`Tdd::negate`](crate::Tdd::negate); [`Tdd`](crate::Tdd) explains ownership and copying operands.

### Change a function

Conjoining an observation retains the assignments consistent with it;
[`Tdd::condition`](crate::Tdd::condition) instead substitutes values and explains
how the resulting cofactor is counted.
[`xor`](crate::xor) computes exclusive OR, and [`ite`](crate::ite)
selects between two branches.
[`Tdd::exists_vars`](crate::Tdd::exists_vars) keeps assignments that have a satisfying extension.
[`and_exists`](crate::and_exists) combines conjunction and existential quantification into one operation.
[`Tdd::rename_vars`](crate::Tdd::rename_vars) handles simultaneous renaming and variable identification.
The [reachability walkthrough](crate::guide::examples::reachability) combines these operations into a state-space search.

## Query satisfying assignments

[`Tdd::is_sat`](crate::Tdd::is_sat) tests whether any assignment satisfies a structural diagram.
[`Tdd::model_count`](crate::Tdd::model_count) counts assignments over all vtree variables, including free ones.
[`Tdd::projected_model_count`](crate::Tdd::projected_model_count) counts distinct assignments to selected variables that have a satisfying extension.
[`Tdd::satisfying_assignment`](crate::Tdd::satisfying_assignment) returns one complete assignment, or `None`.
[`Tdd::equivalent`](crate::Tdd::equivalent) compares represented functions, and
[`Tdd::implies`](crate::Tdd::implies) tests entailment.
[`Tdd::support`](crate::Tdd::support) finds relevant variables; [`Tdd::implied_literals`](crate::Tdd::implied_literals) finds literals true in every model.

The [configuration walkthrough](crate::guide::examples::configurations) combines
construction, counting, and evidence queries in a small constraint model.
[`Tdd::counter`](crate::Tdd::counter) retains counting state across evidence updates;
[`ModelCounter::observe`](crate::query::ModelCounter::observe) accepts signed-literal observations.

## Evaluate weighted models

[`Tdd::evaluate`](crate::Tdd::evaluate) evaluates a structural diagram under literal weights;
the [probability walkthrough](crate::guide::examples::probability) combines Boolean events
and weighted evaluation to compute conditional probabilities.
Implement [`EvalAlgebra`](crate::diagram::EvalAlgebra) to evaluate another quantity, such as the [minimum cost of a configuration](crate::guide::examples::optimization).
For fixed attached weights, use [`Tdd::set_weights`](crate::Tdd::set_weights) and [`Tdd::weighted_value`](crate::Tdd::weighted_value).

## Save and restore circuits

[`write_tdd`](crate::io::write_tdd) and [`read_tdd`](crate::io::read_tdd) save and restore diagram structure,
with the vtree stored separately through [`Vtree::to_text`](crate::Vtree::to_text) and [`Vtree::from_text`](crate::Vtree::from_text).
The [persistence walkthrough](crate::guide::examples::persistence) restores two diagrams onto one shared vtree and combines them.

## Control execution

[`Context::with_limits`](crate::Context::with_limits) lends a batch engine with a
[`LimitConfig`](crate::limits::LimitConfig) installed; use that engine throughout the bounded work.
[`Context::run`](crate::Context::run) lends the same reusable workspace without initial limits.
The [execution walkthrough](crate::guide::examples::execution) handles a refusal and explains batch boundaries.
[`Context::bind`](crate::Context::bind) lets several vtrees share one workspace;
[`Context::clear_scratch`](crate::Context::clear_scratch) releases idle buffers.

## Specialize the representation

### Choose decompositions and algorithms

Use [`Vtree::linear`](crate::Vtree::linear) for a variable order or [`Vtree::join`](crate::Vtree::join)
for explicit grouping; the [vtree walkthrough](crate::guide::examples::vtrees) compares two groupings of the same function.
[`Tdd::minimize`](crate::Tdd::minimize) removes redundancy under the current vtree;
ordinary structural counting and witness queries need no explicit minimization.
[`Tdd::rotation_search`](crate::Tdd::rotation_search) searches alternative vtree shapes.
[`Tdd::exists_vars_with_strategy`](crate::Tdd::exists_vars_with_strategy) selects a quantification rewrite explicitly.
[`Tdd::reduce`](crate::Tdd::reduce) accepts a [`ReductionPlan`](crate::reduce::ReductionPlan) for selecting individual passes.

### Simplify or replace parts of a function

[`Tdd::substitute`](crate::Tdd::substitute) replaces variables with whole functions.
[`Tdd::restrict_to_care`](crate::Tdd::restrict_to_care) simplifies a function within a care set.
[`Tdd::marginalize_levels`](crate::Tdd::marginalize_levels) permanently replaces structure with counts or fixed weighted values.

### Inspect or assemble storage

The [custom-statistic walkthrough](crate::guide::examples::statistics) introduces traversal of levels, nodes and pairs.
[`tdd_to_dot`](crate::io::tdd_to_dot) and [`vtree_to_dot`](crate::io::vtree_to_dot) render Graphviz text; [`Tdd::vtree_to_dot`](crate::Tdd::vtree_to_dot) adds circuit-size annotations to its own vtree.
[`TddBuilder`](crate::diagram::TddBuilder) assembles storage directly; its contract includes determinism obligations.
[`Tdd::graft`](crate::Tdd::graft) and [`Tdd::graft_over`](crate::Tdd::graft_over) combine disjoint variable domains.
The [data model](crate::guide::model) explains the representation; the
[architecture reference](crate::guide::architecture) describes implementation responsibilities.
