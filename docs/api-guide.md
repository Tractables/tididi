# API guide

The first section is the whole library in three snippets, and the table after
it places every operation by cost. The sections below those stand on their
own: read the one for the capability you need. Item documentation is on
[docs.rs](https://docs.rs/tididi); the data model is in [`docs/tdd.md`](https://docs.rs/tididi/latest/tididi/guide/model/index.html).

Every diagram is tied to a vtree, shared as an `Arc<Vtree>`. The operands of
a binary operation must share the same [`Arc`].

## First five minutes

Build a diagram one clause at a time, reduce it to the canonical form for its
vtree, and count its models.

```rust
use std::sync::Arc;
use num_bigint::BigUint;
use tididi::Tdd;
use tididi::apply::apply_and_clause;
use tididi::reduce::minimize;
use tididi::vtree::Vtree;

let vtree = Arc::new(Vtree::balanced(4));   // x1..x4
let mut f = Tdd::one(&vtree);
for clause in [[1, -2], [2, 3], [-3, 4]] {  // DIMACS literals
    let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
    f = apply_and_clause(f, &lits);
}
minimize(&mut f);
assert_eq!(f.model_count(), BigUint::from(5u32));
```

Conjoin two diagrams under a caller's limits. An operation that runs past a
limit returns [`OperationError`] rather than a partial answer, and an owned entry
point spends both operands on the way out.

```rust
# use std::sync::Arc;
# use tididi::vtree::Vtree;
use std::time::Instant;
use tididi::{OperationError, Engine, Tdd};
use tididi::limits::LimitConfig;

let vtree = Arc::new(Vtree::balanced(4));
let engine = Engine::new();
engine.limits().install(LimitConfig::none().with_deadline(Some(Instant::now())));

let f = Tdd::clause(&vtree, [1, -2]);
let g = Tdd::clause(&vtree, [2, 3]);
match engine.and(f, g) {
    Ok(h) => assert!(!h.is_zero()),          // it finished before a poll
    Err(OperationError::Stopped) => {}          // the deadline had already passed
    Err(e) => panic!("unexpected refusal: {e}"),
}
```

Sum a vtree level out. The diagram keeps its count and loses the storage below
that level, which is what bounds the memory a count needs.

```rust
# use std::sync::Arc;
# use tididi::vtree::Vtree;
use tididi::{Engine, Tdd};
use tididi::marginal::marginalize_levels;

let vtree = Arc::new(Vtree::balanced(4));
let engine = Engine::new();
let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let before = f.model_count();

// The root's left child: its own children are leaves, so it may go first.
let (left, _right) = vtree.children(vtree.root());
marginalize_levels(&engine, &mut f, &[left]).unwrap();

assert!(f.has_marginal_level());
assert_eq!(f.model_count(), before);
```

## Operations by cost

Cost is in the diagram's own terms: what the operation walks, folds over, or
rebuilds. Refusal is available only on a caller's [`Engine`] — the free
functions run on an engine with nothing armed, so no limit cuts one short —
except where the table says otherwise.

| Operation | Cost | Refuses under limits | Canonical form |
|---|---|---|---|
| [`Tdd::one`], [`Tdd::zero`], [`Tdd::clause`], [`engine.cube`] | linear in the vtree | no | canonical |
| [`Tdd::graft`] | linear in the parts, and runs no apply | no | canonical when the parts are |
| [`engine.and`] and `&` | product of the operand levels | on an engine | count-correct; canonical after [`minimize`] |
| [`engine.or`] and `\|` | three fills and a product | on an engine | canonical |
| [`apply_and_clause`], [`engine.and_clause`] | walks the accumulator bottom-up | on an engine | count-correct; canonical after [`minimize`] |
| [`negate`] | fills every level, then complements | no | canonical |
| [`condition_var`], [`condition_vars`] | walks the diagram, and never grows it | on an engine | canonical |
| [`exists_var`], [`exists_vars`] | disjoins the cofactors of the variable, or rewrites the levels above it | on an engine | canonical |
| [`restrict_to_care`] | walks the diagram, and never grows it | on an engine | not canonical; run [`minimize`] |
| [`minimize`], [`try_reduce`] | walks the diagram | [`try_reduce`] only | establishes it |
| [`marginalize_levels`] | walks the levels named, and frees the storage below them | yes | preserved |
| [`Tdd::model_count`], [`engine.model_count`], [`evaluate`] | folds over the diagram | [`engine.model_count`] only | unchanged |
| [`ModelCounter`] | folds over the diagram, then over the levels between the changed leaves and the root | no | unchanged |
| [`engine.rotation_search`] | rebuilds the levels each pivot touches | yes | canonical |
| [`save_tdd`], [`load_tdd`], [`tdd_to_dot`] | walks the diagram | no | unchanged |
| [`Tdd::pair_count`], [`node_count()`], [`max_width()`] | walks the diagram | no | unchanged |

## Building

### Diagrams and vtrees

[`Tdd`] is the diagram: one [`TddLevel`] per vtree node, an `output` node, and
the `Arc<Vtree>`. [`Vtree`] is a binary tree whose leaves are variables.
Variables are [`VarId(0..n)`], 0-based; [`Literal`] pairs a [`VarId`] with a
polarity ([`Literal::pos`], [`Literal::neg`], or [`Literal::from(i32)`] with the
1-based DIMACS sign convention).

```rust
use std::sync::Arc;
use tididi::Literal;
use tididi::vtree::{VarId, Vtree};

let vtree = Arc::new(Vtree::balanced(4)); // x1..x4, balanced shape
let x1 = Literal::pos(VarId(0));          // the literal x1, 0-based
assert_eq!(x1, Literal::from(1));         // the same literal, DIMACS-signed
```

Constructors: [`Vtree::leaf(var)`]; [`Vtree::join(&l, &r)`] for a new root over
two vtrees with disjoint variables; [`Vtree::balanced(n)`] and
[`Vtree::balanced_over(&order)`]; [`Vtree::linear(n)`], [`Vtree::reverse_linear(n)`] and
[`Vtree::linear_from_order(&order)`] for a right-linear chain, which is an OBDD
variable order; [`Vtree::random(n, seed)`]; [`Vtree::from_nodes`] for a
hand-built node list; `Vtree::graft(&subtrees, &spine_vars)` (see
[Graft](#graft)) and `Vtree::graft_over` for parts in their own id spaces;
and `project_to_vars` for the vtree induced on a subset of the variables.
[`Vtree::from_text`] parses the
`.vtree` text format and `to_text` writes it. A vtree may
skip variable ids: [`num_vars()`] is the id space and [`num_leaves()`] the
variables carried. [`validate()`] checks the invariants of a hand-built tree.
Construction and parsing errors are [`VtreeError`] ([`Text`],
[`OverlappingVariable`], [`Invalid`]).

Read a tree with [`root()`], [`node()`], [`num_nodes()`], [`children()`],
[`sibling()`], [`leaf_of()`] (`None` for a variable no leaf carries),
[`leaf_var()`], [`lca()`], and the traversal orders [`bottomup()`],
[`leaf_bottomup()`], [`internal_bottomup()`]. [`same_tree()`] compares shape and
variables; node numbering is not identity.

### Base diagrams

```rust
# use std::sync::Arc;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::{Literal, Tdd};

let top = Tdd::one(&vtree);          // ⊤
let bot = Tdd::zero(&vtree);         // ⊥: the ZERO sentinel, no nodes
let c = Tdd::clause(&vtree, [1, -2]);    // x1 ∨ ¬x2

// The same clause from the two shapes a reader arrives with.
let from_file: Vec<i32> = vec![1, -2];
let lits: Vec<Literal> = from_file.iter().map(Literal::from).collect();
assert_eq!(Tdd::clause(&vtree, &from_file).model_count(), c.model_count());
assert_eq!(Tdd::clause(&vtree, &lits).model_count(), c.model_count());
```

[`Tdd::clause`] accepts anything convertible to [`Literal`], so a `&[i32]` of
DIMACS literals and a `&[Literal]` both work, and builds the canonical diagram
of the clause directly. [`engine.cube`] is the conjunction of literals rather
than their disjunction; a variable no literal mentions is free in both.

### Graft

[`Tdd::graft(parts, &spine_vars)`] is the conjunction of diagrams over
pairwise-disjoint variable sets, each on its own vtree, built structurally on
[`Vtree::graft`] of their vtrees without running an apply; `spine_vars` are
variables no part mentions, and the error is [`VtreeError`].

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
let b = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(3)]));
let fg = Tdd::graft(vec![Tdd::clause(&a, [1, 2]), Tdd::clause(&b, [3, -4])], &[VarId(4)]).unwrap();
assert_eq!(fg.model_count(), 18u32.into()); // 3 · 3 · 2
```

## Combining

### Boolean combination

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::apply::negate;

let conj = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let disj = Tdd::clause(&vtree, [1]) | Tdd::clause(&vtree, [2]);
let neg = !Tdd::clause(&vtree, [1, 2]);
let same = negate(Tdd::clause(&vtree, [1, 2]));  // what `!` forwards to
assert_eq!(neg.model_count(), same.model_count());
```

`&` and `|` are conjunction and disjunction, `!` forwards to [`negate`], and
all three consume their operands; `|` and `!` return canonical diagrams, and
`&` returns a count-correct diagram that [`minimize`] reduces.
[`engine.and(f, g)`] and [`engine.or(f, g)`] are the same operations on a
caller's engine, returning [`OperationError`] under a limit
([Limits and refusal](#limits-and-refusal)), and
[`engine.and_marginalizing(f, g, &targets)`] names the vtree levels to emit as
marginal ([Marginal levels](#marginal-levels)).

[`apply_and_clause(acc, &lits)`] conjoins one clause into an accumulator
without building the clause as a diagram, and [`engine.and_clause(acc, &lits)`]
is the fallible form; the accumulator is count-correct after every clause and
canonical after [`minimize`].

### Conditioning

[`condition_var`], [`condition_vars`] and [`Engine::condition`](crate::Engine::condition)
compute cofactors for one variable, a shared value, or a mixed assignment.

### Quantification

[`exists_var(&f, x, how)`] returns `∃x. f`; the result keeps the vtree, so
the forgotten variable still ranges over both values in [`Tdd::model_count`],
and summing a vtree *level* out is [`marginalize_levels`] instead
([Marginal levels](#marginal-levels)). [`exists_vars(&f, &vars, how)`]
forgets a set; a variable the vtree does not carry is
[`OperationError::VariableNotInVtree`] on the engine forms and a panic on the free
ones. `how` is [`QuantificationStrategy::Automatic`], the cofactor form `f|x=⊤ ∨ f|x=⊥`
where that is sound and an in-place rewrite where a marginal level makes it
not, or [`QuantificationStrategy::Structural`], the in-place rewrite outright, slower but
never cloning the diagram.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::apply::{exists_var, QuantificationStrategy};

let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
let g = exists_var(&f, VarId(1), QuantificationStrategy::Automatic);     // ∃x2: x1, with x2 free, twice the models
assert_eq!(g.model_count(), f.model_count() * 2u32);
```

### Restrict-to-care

[`restrict_to_care(f, care)`] prunes `f` to the pairs and nodes that produce a model
under `care`, returning `g` with `g ∧ care == f ∧ care` and `g` no larger than
`f` as [`RestrictionOutcome::Unchanged`], [`RestrictionOutcome::Shrunk(g)`] with a
non-canonical `g`, or [`RestrictionOutcome::Unsatisfiable`] carrying false;
[`into_tdd()`] extracts the diagram. [`engine.restrict_to_care`] is the same operation under the
caller's limits.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::{VarId, Vtree};
# let vtree = Arc::new(Vtree::balanced(4));
use tididi::apply::restrict_to_care;

let f = Tdd::clause(&vtree, [1, 2]);
let care = Tdd::clause(&vtree, [1]);
let g = restrict_to_care(f.clone(), care.clone()).into_tdd();
assert_eq!((g & care.clone()).model_count(), (f & care).model_count());
```

### Reduction

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# fn main() -> Result<(), tididi::OperationError> {
# let vtree = Arc::new(Vtree::balanced(4));
# let mut t = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
use tididi::engine::Engine;
use tididi::reduce::{minimize, try_reduce, ReductionPlan};

let engine = Engine::new();
minimize(&mut t);
let opts = ReductionPlan::Prune;
try_reduce(&engine, &mut t, opts)?;
# Ok(())
# }
```

[`minimize`] prunes unreachable nodes and contracts twins until the diagram is
the canonical form for its vtree ([`docs/tdd.md`](https://docs.rs/tididi/latest/tididi/guide/model/index.html)); [`Tdd::clause`],
[`Tdd::graft`], `|`, [`negate`], conditioning and projection return canonical
diagrams, while a conjunction (`&`, [`engine.and`], [`apply_and_clause`]), a
[`restrict_to_care`] result and a hand-built diagram need it.
[`try_minimize`](crate::reduce::try_minimize) runs full minimization under the engine's limits.
[`try_reduce`] runs the passes selected by [`ReductionPlan`], whose full pass accepts a [`ContentTwinPolicy`](crate::reduce::ContentTwinPolicy).

Options and report types grow fields, and the enums grow variants, without a
breaking release: use their constructors or `Default`, match with a wildcard arm, and destructure with a
trailing `..`. [`OperationError`] is the exception: callers mint it, so its
variants are the whole set.

## Limits and refusal

An [`Engine`] is the session an operation runs in: it owns the limits the
operation runs under and every buffer the operation reuses between calls.
Nothing is per-thread, and nothing is armed over an operation the caller did
not arm it over:

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
use std::time::{Duration, Instant};
use tididi::engine::Engine;
use tididi::limits::{LimitConfig, MemoryHooks, StopDecision, StopRules, StopAt};
use tididi::OperationError;

let engine = Engine::new();
let _prior = engine.limits().install(
    LimitConfig::none()
        .with_deadline(Some(Instant::now() + Duration::from_secs(30)))
        .with_memory_budget_bytes(Some(4 << 30))        // bytes one operation may grow its storage by
        .with_output_node_cap(Some(50_000_000)) // output nodes one conjunction may build
        .with_stop_callback(Some(tididi::limits::StopCallback::new(|_meters, _now| StopDecision::Continue)))
        .with_memory_hooks(MemoryHooks::NONE)
        .with_conjunction_progress(true),
);
match engine.and(f, g) {
    Ok(h) => { /* ... */ }
    Err(OperationError::OverBudget | OperationError::Stopped | OperationError::OutputCap) => { /* cut short */ }
    Err(OperationError::LevelNotInVtree(_) | OperationError::MarginalLevel(_)) => { /* unavailable structure */ }
    Err(OperationError::VariableNotInVtree(v)) => unreachable!("a conjunction names no variable: {v:?}"),
    Err(OperationError::VtreeMismatch | OperationError::RootMismatch | OperationError::IncompatibleWeights) => { /* incompatible operands */ }
}
```

[`LimitConfig`] is cloneable and shares its captured callbacks; [`engine.limits().install(set)`] arms
one, replacing every axis, and returns what was armed before;
[`engine.limits().scope(set)`] arms one for a lexical scope and [`edit`] changes
one axis of the armed set for a scope, both restoring the prior set on drop;
[`set_budget`] replaces the byte budget alone. The arming verbs
are [`with_memory_budget_bytes`], [`with_output_node_cap`], [`with_stop_rules`], [`with_stop_callback`], [`with_memory_hooks`]
([`MemoryHooks::new`](crate::limits::MemoryHooks::new) captures the caller's probes)
and [`with_conjunction_progress`], read back by [`memory_budget_bytes`], [`output_node_cap`],
[`stop_rules`], [`stop_callback`], [`memory_hooks`] and [`conjunction_progress_enabled`];
[`LimitConfig::without_stop_rules()`] clears the whole stop axis.

[`StopRules`] carries two bounds in one axis: [`unconditional`] is unconditional, and
[`after_pairs`] is `(pairs, at)`, in force once the operation has built that many
output pairs. Each bound falls at an instant ([`StopAt::Time`]) or at a reading
of the engine's own work clock ([`StopAt::WorkUnits`]), which is reproducible across
machines; [`LimitConfig::with_deadline(Some(t))`] is the common case, [`StopRules::by_time(t)`]
sets that unconditional time bound, and [`StopRules::after_pairs`] arms
the size-conditional one.

The stop callback is asked on every poll, handed the meters and the clock
reading, and answers [`StopDecision::Continue`], [`StopDecision::Stop`], or
[`StopDecision::ReplaceRules(stop)`], a commitment that replaces the stop the operation
was running under.

[`engine.reset()`] releases every buffer the engine retains, keeping the armed
limits; it is sound only between operations.

[`OperationError`] reports resource refusals and invalid operands.

[`engine.limits().meters()`] snapshots the meters ([`OperationMetrics`]:
[`in_flight_bytes`], [`pairs_in_flight`], [`work_units`], [`refused_reserve_bytes`], and
[`conjunction`] as a [`ConjunctionProgress`]); [`engine.limits().armed()`] reads back what is
armed; [`reset_meters()`] zeroes the per-operation meters
at the start of an independent compile; [`work_units()`], [`mark()`] and
[`work_since(mark)`] read the engine's work clock, which is never reset. The
free functions run on a transient engine with nothing armed; the forms that
run under the caller's limits and can refuse are [`engine.and`],
[`engine.or`], [`Engine::negate`](crate::Engine::negate), [`engine.and_clause`], [`engine.and_marginalizing`],
[`engine.exists_var`], [`engine.exists_vars`], [`engine.condition_var`],
[`engine.condition_vars`], [`Engine::condition`](crate::Engine::condition), [`engine.restrict_to_care`], [`engine.model_count`],
[`engine.rotation_search`], [`marginalize_levels`] and [`try_reduce`];
[`engine.one`], [`engine.zero`], [`engine.clause`] and [`engine.cube`] only
reuse the engine's buffers.

## Counting and semirings

### Model counting

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
let n = f.model_count();   // BigUint
# assert_eq!(n, tididi::Engine::new().model_count(&f).unwrap());
```

The count is over all variables of the vtree, a variable the function does
not mention contributing a factor of two, and [`engine.model_count(&f)`] is
the same count under the engine's limits. [`node_counts_u128(&f)`] returns
every node's count as a `u128`, saturating at `u128::MAX`.

[`ModelCounter`] borrows a diagram and refreshes changed pins when its count is read, using [`KeepAllColumns`] or [`KeepFrontier`].
[`PinSemantics::Evidence`] counts a pinned variable once; [`PinSemantics::Cofactor`] leaves its factor of two.

### Weighted and algebraic evaluation

[`evaluate(&f, &algebra)`] folds an [`EvalAlgebra`] over an explicit diagram.
[`RationalWeights::from_literals`](crate::diagram::RationalWeights::from_literals) and [`RationalWeights::from_polarities`](crate::diagram::RationalWeights::from_polarities) accept [`LiteralWeights`](crate::diagram::LiteralWeights) with named negative and positive fields.
[`RationalWeights::unit(n)`] gives every literal weight one.

```rust
# use std::sync::Arc;
# use num_rational::BigRational;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
# let half = BigRational::new(1.into(), 2.into());
# let weights: Vec<LiteralWeights<BigRational>> =
#     (0..4).map(|_| LiteralWeights { negative: half.clone(), positive: half.clone() }).collect();
use tididi::diagram::{LiteralWeights, RationalWeights};
use tididi::query::evaluate;

let algebra = RationalWeights::from_literals(&weights);
let wmc = evaluate(&f, &algebra);
```

[`SignedLog`] is a signed log-domain value with `mul`, `add_assign`, and
[`from_rational`]. [`WeightValue`] is the per-node value a [`WeightStore`] holds;
it is `#[non_exhaustive]`, so build values with [`WeightValue::exact`] and read
them with [`as_rational`], [`into_rational`], [`into_rational_opt`], or
[`as_log`].

### Shape and status

[`Tdd::pair_count()`] is the total pair count, the size measure of the paper;
[`pair_count_at_most(cap)`], [`node_count()`], [`max_width()`], [`slot_count_at(t)`],
[`reference_slot_count(t)`], [`is_zero()`], [`has_marginal_level()`],
[`retired_marginal_slots()`] and [`reachable_nodes()`] read the diagram's
shape; [`is_sat_minimized(&f)`] and [`implied_literals(&f)`] read the
satisfiability and the backbone of a minimized diagram.

## Marginal levels

This section is about vtree levels, not variables: summing a *variable* out is
existential quantification, which is [Quantification](#quantification) above.

[`marginalize_levels(engine, &mut f, &levels)`] sums the named vtree levels out of the
diagram, each becoming a marginal level holding one value per node instead of
pairs; `levels` is sorted bottom-up, a level going only once its children are
marginal or are leaves. A marginal level is permanent, so sum a level out only
once every clause over its variables is in; the errors are
[`OperationError::Stopped`], after which the levels summed out before the cut
keep their values, and [`OverBudget`].

Summing out a vtree leaf inlines the leaf's fixed count into its parent's
references by default; [`Engine::set_leaf_marginalize_inlines`] turns that off
for a caller that still needs to read the leaf's labels afterwards.

[`Tdd::has_marginal_level`] reports whether any level is marginal.

With a [`WeightStore`] attached, the same operation stores each node's
semiring value in the store instead of a count:

```rust
# use std::sync::Arc;
# use num_rational::BigRational;
# use tididi::{Engine, Tdd};
# use tididi::vtree::{Vtree, VtreeNode};
# let vtree = Arc::new(Vtree::balanced(4));
# let engine = Engine::new();
# let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
# let half = BigRational::new(1.into(), 2.into());
# let weights: Vec<LiteralWeights<BigRational>> =
#     (0..4).map(|_| LiteralWeights { negative: half.clone(), positive: half.clone() }).collect();
# // The root's left child: an internal level whose own children are leaves,
# // so it is the first level that may be summed out.
# let VtreeNode::Internal { left, .. } = *vtree.node(vtree.root()) else { unreachable!() };
# let levels = [left];
use tididi::diagram::{LiteralWeights, RationalWeights};
use tididi::diagram::{Arithmetic, WeightStore};
use tididi::marginal::marginalize_levels;
use tididi::query::weighted_value;

let algebra = RationalWeights::from_literals(&weights); // named polarities per variable
f.set_weights(WeightStore::new(algebra, Arithmetic::ExactRational)).unwrap();
marginalize_levels(&engine, &mut f, &levels).unwrap();
let total = weighted_value(&f);                    // Option<WeightValue>
# assert!(total.is_some());
```

[`Arithmetic::ExactRational`] folds in [`BigRational`] and
[`Arithmetic::SignedLog`] in the bounded-precision [`SignedLog`] domain; attach
the store before the first marginalization. [`Tdd::weights`] reads the store,
[`Tdd::take_weights`] detaches it (an error while a weight-marginal level
still holds values in it), [`WeightStore::level(t.idx())`] reads a marginal
level's values, and [`weighted_value`] or [`Engine::weighted_value`](crate::Engine::weighted_value) reads the diagram's value.

## Traversal contract

The stored encoding is the traversal contract. The [`diagram`] module
documentation states it: how each kind of level is read, and the invariants a
reader may rely on.

```rust
# use std::sync::Arc;
# use tididi::Tdd;
# use tididi::vtree::Vtree;
# let vtree = Arc::new(Vtree::balanced(4));
# let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
use tididi::diagram::{ChildRef, ValueRef};

for (t, left, right) in f.vtree().internal_bottomup() {
    let level = f.level(t);
    if level.is_marginal() { continue; }
    let (lv, rv) = (f.level(left).child_decoder(), f.level(right).child_decoder());
    for (i, pairs) in level.internal_inputs_iter() {
        for p in pairs {
            let l = lv.child(p.left);   // Node(index), or Value(Slot | Inline)
            let r = rv.child(p.right);
        }
    }
}
```

`README.md` lists the three programs under `examples/`, among them
`examples/statistic.rs`, a statistic read straight off the stored encoding.
[`Tdd::builder(&eng, &vtree)`] opens a [`TddBuilder`], which appends nodes
level by level, bottom-up ([`push`] or lazy [`intern`]), and hands back the diagram from [`finish(output)`];
[`TddBuildError`] names what it checks.

## Persistence

[`save_tdd(&f, path)`] writes the `.tdd` text format, which records the
diagram and not its vtree, and [`load_tdd(path, &vtree)`] reads it back
against the vtree it belongs to; [`write_tdd`] and [`read_tdd`] are the same
two over any `Write` or `BufRead`. [`tdd_to_dot(&f)`] and
[`vtree_to_dot(&vtree, Some(&f))`] render Graphviz DOT. All but the last
return `Result<_, IoError>`, where [`IoError`] is an underlying
[`std::io::Error`] or a [`Format`] message naming what the file or diagram
violated; both formats are structural, so a diagram with a marginal level is
refused as [`Format`].
[`Vtree::to_text()`] writes the `.vtree` format and [`Vtree::from_text()`]
reads it; [`Display`] and [`FromStr`] are the same two. `docs/vtree_example.svg`
and `docs/tdd_example.svg` in the repository are renders of one diagram.

## Restructuring

[`engine.rotation_search(&mut t, &mut objective, &config)`] rotates the vtree
under a compiled diagram to a local minimum of any [`RotationObjective`]
([`delta(before, after) -> i64`], negative to accept), within the bounds
[`RotationSearchConfig`] sets, and returns
[`RotationSearchStats { probes, accepts, sweeps }`]. The model count is
preserved, and the search returns [`Err(OperationError::Stopped)`] when the armed
stop falls, leaving the diagram canonical and count-correct.

```rust
# use std::sync::Arc;
# use tididi::{Engine, Tdd};
# use tididi::vtree::Vtree;
# fn main() -> Result<(), tididi::OperationError> {
# let engine = Engine::new();
# let vtree = Arc::new(Vtree::balanced(4));
# let mut t = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
use tididi::restructure::search::{RotationObjective, RotationSearchConfig};
use tididi::diagram::TddLevel;

struct MinPeak;
impl RotationObjective for MinPeak {
    fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
        a.0.slot_count().max(a.1.slot_count()) as i64 - b.0.slot_count().max(b.1.slot_count()) as i64
    }
}
let stats = engine.rotation_search(&mut t, &mut MinPeak, &RotationSearchConfig::default())?;
# Ok(()) }
```

[`OperationError::Stopped`]: crate::OperationError::Stopped
[`OperationError::VariableNotInVtree`]: crate::OperationError::VariableNotInVtree
[`OperationError`]: crate::OperationError
[`OperationMetrics`]: crate::limits::OperationMetrics
[`Arc`]: std::sync::Arc
[`Arithmetic::ExactRational`]: crate::diagram::Arithmetic::ExactRational
[`Arithmetic::SignedLog`]: crate::diagram::Arithmetic::SignedLog
[`BigRational`]: num_rational::BigRational
[`ContentTwinSchedule`]: crate::reduce::ContentTwinSchedule
[`Stopped`]: crate::OperationError::Stopped
[`Display`]: std::fmt::Display
[`Engine`]: crate::Engine
[`Err(OperationError::Stopped)`]: crate::OperationError::Stopped
[`EvalAlgebra`]: crate::diagram::EvalAlgebra
[`Format`]: crate::io::IoError::Format
[`FromStr`]: std::str::FromStr
[`ModelCounter::new(eng, &f, n_pins, convention)`]: crate::query::ModelCounter::new
[`ModelCounter`]: crate::query::ModelCounter
[`Invalid`]: crate::vtree::VtreeError::Invalid
[`IoError`]: crate::io::IoError
[`KeepAllColumns`]: crate::query::KeepAllColumns
[`KeepFrontier`]: crate::query::KeepFrontier
[`LimitConfig::with_deadline(Some(t))`]: crate::limits::LimitConfig::with_deadline
[`LimitConfig::without_stop_rules()`]: crate::limits::LimitConfig::without_stop_rules
[`LimitConfig`]: crate::limits::LimitConfig
[`Literal::from(i32)`]: crate::Literal
[`Literal::neg`]: crate::Literal::neg
[`Literal::pos`]: crate::Literal::pos
[`Literal`]: crate::Literal
[`MemoryHooks::NONE`]: crate::limits::MemoryHooks::NONE
[`MemoryHooks`]: crate::limits::MemoryHooks
[`ConjunctionProgress`]: crate::limits::ConjunctionProgress
[`ReductionPlan`]: crate::reduce::ReductionPlan
[`OutputCap`]: crate::OperationError::OutputCap
[`OverBudget`]: crate::OperationError::OverBudget
[`OverlappingVariable`]: crate::vtree::VtreeError::OverlappingVariable
[`QuantificationStrategy::Automatic`]: crate::apply::QuantificationStrategy::Automatic
[`QuantificationStrategy::Structural`]: crate::apply::QuantificationStrategy::Structural
[`RationalWeights::unit(n)`]: crate::diagram::RationalWeights::unit
[`RestrictionOutcome::Shrunk(g)`]: crate::apply::RestrictionOutcome::Shrunk
[`RestrictionOutcome::Unchanged`]: crate::apply::RestrictionOutcome::Unchanged
[`RestrictionOutcome::Unsatisfiable`]: crate::apply::RestrictionOutcome::Unsatisfiable
[`RotationObjective`]: crate::restructure::search::RotationObjective
[`RotationSearchConfig`]: crate::restructure::search::RotationSearchConfig
[`RotationSearchStats { probes, accepts, sweeps }`]: crate::restructure::search::RotationSearchStats
[`StopDecision::Continue`]: crate::limits::StopDecision::Continue
[`StopDecision::ReplaceRules(stop)`]: crate::limits::StopDecision::ReplaceRules
[`StopDecision::Stop`]: crate::limits::StopDecision::Stop
[`PinSemantics::Evidence`]: crate::query::PinSemantics::Evidence
[`PinSemantics::Cofactor`]: crate::query::PinSemantics::Cofactor
[`SignedLog`]: crate::diagram::SignedLog
[`StopRules::by_time(t)`]: crate::limits::StopRules::by_time
[`StopAt::Time`]: crate::limits::StopAt::Time
[`StopAt::WorkUnits`]: crate::limits::StopAt::WorkUnits
[`StopRules`]: crate::limits::StopRules
[`Tdd::builder(&eng, &vtree)`]: crate::Tdd::builder
[`Tdd::clause`]: crate::Tdd::clause
[`Tdd::graft(parts, &spine_vars)`]: crate::Tdd::graft
[`Tdd::graft`]: crate::Tdd::graft
[`Engine::set_leaf_marginalize_inlines`]: crate::Engine::set_leaf_marginalize_inlines
[`Tdd::has_marginal_level`]: crate::Tdd::has_marginal_level
[`Tdd::model_count`]: crate::Tdd::model_count
[`Tdd::one`]: crate::Tdd::one
[`Tdd::pair_count()`]: crate::Tdd::pair_count
[`Tdd::pair_count`]: crate::Tdd::pair_count
[`Tdd::take_weights`]: crate::Tdd::take_weights
[`Tdd::weights`]: crate::Tdd::weights
[`Tdd::zero`]: crate::Tdd::zero
[`TddBuildError`]: crate::diagram::TddBuildError
[`TddBuilder`]: crate::diagram::TddBuilder
[`TddLevel`]: crate::diagram::TddLevel
[`Tdd`]: crate::Tdd
[`Text`]: crate::vtree::VtreeError::Text
[`VarId(0..n)`]: crate::vtree::VarId
[`VarId`]: crate::vtree::VarId
[`Vtree::balanced(n)`]: crate::Vtree::balanced
[`Vtree::balanced_over(&order)`]: crate::Vtree::balanced_over
[`Vtree::from_text()`]: crate::Vtree::from_text
[`Vtree::from_text`]: crate::Vtree::from_text
[`Vtree::graft`]: crate::Vtree::graft
[`Vtree::join(&l, &r)`]: crate::Vtree::join
[`Vtree::leaf(var)`]: crate::Vtree::leaf
[`Vtree::linear(n)`]: crate::Vtree::linear
[`Vtree::reverse_linear(n)`]: crate::Vtree::reverse_linear
[`Vtree::linear_from_order(&order)`]: crate::Vtree::linear_from_order
[`Vtree::random(n, seed)`]: crate::Vtree::random
[`Vtree::to_text()`]: crate::Vtree::to_text
[`VtreeError`]: crate::vtree::VtreeError
[`Vtree`]: crate::Vtree
[`WeightStore`]: crate::diagram::WeightStore
[`WeightValue::exact`]: crate::diagram::WeightValue::exact
[`WeightValue`]: crate::diagram::WeightValue
[`after_pairs`]: crate::limits::StopRules::after_pairs
[`apply_and_clause(acc, &lits)`]: crate::apply::apply_and_clause
[`apply_and_clause`]: crate::apply::apply_and_clause
[`as_log`]: crate::diagram::WeightValue::as_log
[`as_rational`]: crate::diagram::WeightValue::as_rational
[`bottomup()`]: crate::Vtree::bottomup
[`with_memory_budget_bytes`]: crate::limits::LimitConfig::with_memory_budget_bytes
[`memory_budget_bytes`]: crate::limits::LimitConfig::memory_budget_bytes
[`children()`]: crate::Vtree::children
[`condition_var(&f, x, value)`]: crate::apply::condition_var
[`condition_var`]: crate::apply::condition_var
[`condition_vars(&f, &vars, value)`]: crate::apply::condition_vars
[`condition_vars`]: crate::apply::condition_vars
[`delta(before, after) -> i64`]: crate::restructure::search::RotationObjective::delta
[`diagram`]: crate::diagram
[`reference_slot_count(t)`]: crate::Tdd::reference_slot_count
[`engine.and(f, g)`]: crate::Engine::and
[`engine.and_clause(acc, &lits)`]: crate::Engine::and_clause
[`engine.and_clause`]: crate::Engine::and_clause
[`engine.and_marginalizing(f, g, &targets)`]: crate::Engine::and_marginalizing
[`engine.and`]: crate::Engine::and
[`engine.clause`]: crate::Engine::clause
[`engine.condition_var`]: crate::Engine::condition_var
[`engine.cube`]: crate::Engine::cube
[`engine.limits().armed()`]: crate::limits::Limits::armed
[`engine.limits().install(set)`]: crate::limits::Limits::install
[`engine.limits().meters()`]: crate::limits::Limits::meters
[`engine.limits().scope(set)`]: crate::limits::Limits::scope
[`edit`]: crate::limits::Limits::edit
[`set_budget`]: crate::limits::Limits::set_budget
[`work_units()`]: crate::limits::Limits::work_units
[`mark()`]: crate::limits::Limits::mark
[`work_since(mark)`]: crate::limits::Limits::work_since
[`StopRules::after_pairs`]: crate::limits::StopRules::after_pairs
[`engine.exists_vars`]: crate::Engine::exists_vars
[`engine.condition_vars`]: crate::Engine::condition_vars
[`engine.and_marginalizing`]: crate::Engine::and_marginalizing
[`node_counts_u128(&f)`]: crate::query::node_counts_u128
[`reachable_nodes()`]: crate::Tdd::reachable_nodes
[`push`]: crate::diagram::TddBuilder::push
[`intern`]: crate::diagram::TddBuilder::intern
[`write_tdd`]: crate::io::write_tdd
[`read_tdd`]: crate::io::read_tdd
[`Vtree::from_nodes`]: crate::Vtree::from_nodes
[`num_nodes()`]: crate::Vtree::num_nodes
[`WeightStore::level(t.idx())`]: crate::diagram::WeightStore::level
[`engine.model_count(&f)`]: crate::Engine::model_count
[`engine.model_count`]: crate::Engine::model_count
[`engine.one`]: crate::Engine::one
[`engine.or(f, g)`]: crate::Engine::or
[`engine.or`]: crate::Engine::or
[`engine.exists_var`]: crate::Engine::exists_var
[`engine.reset()`]: crate::Engine::reset
[`engine.restrict_to_care`]: crate::Engine::restrict_to_care
[`engine.rotation_search(&mut t, &mut objective, &config)`]: crate::Engine::rotation_search
[`engine.rotation_search`]: crate::Engine::rotation_search
[`engine.zero`]: crate::Engine::zero
[`evaluate(&f, &algebra)`]: crate::query::evaluate()
[`evaluate`]: crate::query::evaluate()
[`finish(output)`]: crate::diagram::TddBuilder::finish
[`from_rational`]: crate::diagram::SignedLog::from_rational
[`has_marginal_level()`]: crate::Tdd::has_marginal_level
[`implied_literals(&f)`]: crate::query::implied_literals
[`in_flight_bytes`]: crate::limits::OperationMetrics::in_flight_bytes
[`internal_bottomup()`]: crate::Vtree::internal_bottomup
[`into_rational_opt`]: crate::diagram::WeightValue::into_rational_opt
[`into_rational`]: crate::diagram::WeightValue::into_rational
[`into_tdd()`]: crate::apply::RestrictionOutcome::into_tdd
[`is_sat_minimized(&f)`]: crate::query::is_sat_minimized
[`is_zero()`]: crate::Tdd::is_zero
[`lca()`]: crate::Vtree::lca
[`leaf_bottomup()`]: crate::Vtree::leaf_bottomup
[`leaf_of()`]: crate::Vtree::leaf_of
[`leaf_var()`]: crate::Vtree::leaf_var
[`load_tdd(path, &vtree)`]: crate::io::load_tdd
[`load_tdd`]: crate::io::load_tdd
[`marginalize_levels(engine, &mut f, &levels)`]: crate::marginal::marginalize_levels
[`marginalize_levels`]: crate::marginal::marginalize_levels
[`max_width()`]: crate::Tdd::max_width
[`with_memory_hooks`]: crate::limits::LimitConfig::with_memory_hooks
[`memory_hooks`]: crate::limits::LimitConfig::memory_hooks
[`conjunction`]: crate::limits::OperationMetrics::conjunction
[`minimize`]: crate::reduce::minimize
[`negate`]: crate::apply::negate()
[`node()`]: crate::Vtree::node
[`node_count()`]: crate::Tdd::node_count
[`num_leaves()`]: crate::Vtree::num_leaves
[`num_vars()`]: crate::Vtree::num_vars
[`with_output_node_cap`]: crate::limits::LimitConfig::with_output_node_cap
[`output_node_cap`]: crate::limits::LimitConfig::output_node_cap
[`model_count(&f)`]: crate::query::ModelCounter::model_count
[`pairs_in_flight`]: crate::limits::OperationMetrics::pairs_in_flight
[`exists_var(&f, x, how)`]: crate::apply::exists_var
[`exists_var`]: crate::apply::exists_var
[`exists_vars(&f, &vars, how)`]: crate::apply::exists_vars
[`exists_vars`]: crate::apply::exists_vars
[`refused_reserve_bytes`]: crate::limits::OperationMetrics::refused_reserve_bytes
[`reset_meters()`]: crate::limits::Limits::reset_meters
[`restrict_to_care(f, care)`]: crate::apply::restrict_to_care()
[`restrict_to_care`]: crate::apply::restrict_to_care()
[`engine.restrict_to_care`]: crate::Engine::restrict_to_care
[`retired_marginal_slots()`]: crate::Tdd::retired_marginal_slots
[`root()`]: crate::Vtree::root
[`same_tree()`]: crate::Vtree::same_tree
[`save_tdd(&f, path)`]: crate::io::save_tdd
[`save_tdd`]: crate::io::save_tdd
[`with_stop_callback`]: crate::limits::LimitConfig::with_stop_callback
[`stop_callback`]: crate::limits::LimitConfig::stop_callback
[`set_pin(var, Some(value))`]: crate::query::ModelCounter::set_pin
[`sibling()`]: crate::Vtree::sibling
[`pair_count_at_most(cap)`]: crate::Tdd::pair_count_at_most
[`std::error::Error`]: std::error::Error
[`std::io::Error`]: std::io::Error
[`with_stop_rules`]: crate::limits::LimitConfig::with_stop_rules
[`stop_rules`]: crate::limits::LimitConfig::stop_rules
[`tdd_to_dot(&f)`]: crate::io::tdd_to_dot
[`tdd_to_dot`]: crate::io::tdd_to_dot
[`try_reduce`]: crate::reduce::try_reduce
[`validate()`]: crate::Vtree::validate
[`vtree_to_dot(&vtree, Some(&f))`]: crate::io::vtree_to_dot
[`unconditional`]: crate::limits::StopRules::unconditional
[`with_conjunction_progress`]: crate::limits::LimitConfig::with_conjunction_progress
[`conjunction_progress_enabled`]: crate::limits::LimitConfig::conjunction_progress_enabled
[`weighted_value`]: crate::query::weighted_value
[`slot_count_at(t)`]: crate::Tdd::slot_count_at
[`work_units`]: crate::limits::OperationMetrics::work_units
