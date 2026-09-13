# Task guide

Choose a task below; each linked API item contains its contract and short code examples.

## Build a Boolean function

- Choose a variable tree with [`Vtree::balanced`](crate::Vtree::balanced), [`Vtree::linear`](crate::Vtree::linear), or [`Vtree::join`](crate::Vtree::join).
- Create literals with [`Engine::literal`](crate::Engine::literal), disjunctions of literals with [`Engine::clause`](crate::Engine::clause), and conjunctions of literals with [`Engine::cube`](crate::Engine::cube).
- Combine functions with [`Engine::and`](crate::Engine::and), [`Engine::or`](crate::Engine::or), [`Engine::negate`](crate::Engine::negate), [`Engine::xor`](crate::Engine::xor), and [`Engine::ite`](crate::Engine::ite).
- Add a clause to an existing function with [`Engine::and_clause`](crate::Engine::and_clause).
- Keep a function for several transformations using the ownership example on [`Tdd`](crate::Tdd).

## Ask questions about a function

- Count satisfying assignments with [`Engine::model_count`](crate::Engine::model_count).
- Obtain a satisfying assignment with [`Engine::satisfying_assignment`](crate::Engine::satisfying_assignment).
- Compare functions with [`Engine::equivalent`](crate::Engine::equivalent) or test entailment with [`Engine::implies`](crate::Engine::implies).
- Find the variables a function depends on with [`Engine::support`](crate::Engine::support).
- Find literals true in every model with [`implied_literals`](crate::query::implied_literals).
- Check satisfiability of a minimized diagram with [`is_sat_minimized`](crate::query::is_sat_minimized).
- Choose reduction passes with [`try_reduce`](crate::reduce::try_reduce) and [`ReductionPlan`](crate::reduce::ReductionPlan).

## Change assignments or variables

- Compute a cofactor for an assignment with [`Engine::condition`](crate::Engine::condition).
- Existentially quantify variables with [`Engine::exists_vars`](crate::Engine::exists_vars).
- Conjoin and existentially quantify with [`Engine::and_exists`](crate::Engine::and_exists).
- Swap or rename variables with [`Engine::rename_vars`](crate::Engine::rename_vars).
- Substitute Boolean functions for variables with [`Engine::substitute`](crate::Engine::substitute).
- Simplify a function under a care set with [`Engine::restrict_to_care`](crate::Engine::restrict_to_care).

## Evaluate probabilities or repeated observations

- Reuse a compiled diagram under changing evidence with [`ModelCounter::try_new`](crate::query::ModelCounter::try_new), [`ModelCounter::set_pin`](crate::query::ModelCounter::set_pin), and [`ModelCounter::try_model_count`](crate::query::ModelCounter::try_model_count), choosing the counting convention through [`PinSemantics`](crate::query::PinSemantics).
- Evaluate weighted probabilities with [`Engine::evaluate`](crate::Engine::evaluate) and [`RationalWeights`](crate::diagram::RationalWeights).
- Supply your own arithmetic by implementing [`EvalAlgebra`](crate::diagram::EvalAlgebra).
- Read the value of a diagram carrying a weight store with [`Engine::weighted_value`](crate::Engine::weighted_value).

## Control resources and diagram size

- Set deadlines and memory budgets through [`LimitConfig`](crate::limits::LimitConfig) and [`Engine::limits`](crate::Engine::limits).
- Handle refused operations through [`OperationError`](crate::OperationError).
- Measure operation work through [`Limits::work_since`](crate::limits::Limits::work_since).
- Minimize under the current vtree with [`minimize`](crate::reduce::minimize) or its checked form [`try_minimize`](crate::reduce::try_minimize).
- Search for a better vtree with [`Engine::rotation_search`](crate::Engine::rotation_search).
- Release structure while preserving counts or fixed weights with [`marginalize_levels`](crate::marginal::marginalize_levels).

## Save or inspect a diagram

- Save and restore a diagram using [`write_tdd`](crate::io::write_tdd) and [`read_tdd`](crate::io::read_tdd), together with [`Vtree::to_text`](crate::Vtree::to_text) and [`Vtree::from_text`](crate::Vtree::from_text).
- Draw a diagram with [`tdd_to_dot`](crate::io::tdd_to_dot) or its vtree with [`vtree_to_dot`](crate::io::vtree_to_dot).
- Inspect nodes and pairs through the traversal examples in [`diagram`](crate::diagram).
- Assemble a diagram from nodes using [`TddBuilder`](crate::diagram::TddBuilder).
- Combine disjoint variable domains with [`Tdd::graft`](crate::Tdd::graft) or [`Tdd::graft_over`](crate::Tdd::graft_over).

## Understand the representation

Read the [TDD model](https://docs.rs/tididi/latest/tididi/guide/model/index.html) for vtrees and determinism, and the [architecture reference](https://docs.rs/tididi/latest/tididi/guide/architecture/index.html) when extending the implementation.
