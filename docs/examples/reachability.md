# Explore reachable states

Starting at A, which nodes can we reach by following these arrows?

```text
start
  │
  ▼
  A ───▶ B ◀───▶ C       D
                       isolated
```

We will describe the graph as a Boolean formula, then repeatedly compute
successors until no new states appear. The result should be {A, B, C}; we
will also check that D is unreachable.

Run it with `cargo run --example symbolic_reachability`.

## Describe a state with Boolean indicators

Give each node an indicator: `a` means “we are at A”, `b` means “we are at B”,
and so on. Exactly one is true in a state. Write the four state formulas as:

```text
A(x) =  a ∧ ¬b ∧ ¬c ∧ ¬d
B(x) = ¬a ∧  b ∧ ¬c ∧ ¬d
C(x) = ¬a ∧ ¬b ∧  c ∧ ¬d
D(x) = ¬a ∧ ¬b ∧ ¬c ∧  d
```

Here `x` stands for the four indicators `(a, b, c, d)`. Each formula fixes
all four, so it excludes assignments that name several nodes or no node.

A **set of states** is a disjunction of these formulas. For example,
`A(x) ∨ B(x)` has two satisfying assignments, one for A and one for B.
It does not set both `a` and `b` to true. The indicators describe one possible
current state; the circuit collects all the states reachable so far.

## Write the transition relation

Use a second set of indicators `x′ = (a′, b′, c′, d′)` for the next state.
The graph has three edges, so its transition relation is:

```text
T(x, x′) = (A(x) ∧ B(x′))
         ∨ (B(x) ∧ C(x′))
         ∨ (C(x) ∧ B(x′))
```

For example, `A(x) ∧ B(x′)` says “we are at A now and at B next”. The relation
is true exactly for the three allowed moves. No term enters or leaves D.

## Build those formulas

Use integer literals 1 through 4 for the current indicators and 5 through 8
for the next ones:

| Indicator | Current literal | Next literal |
|---|---|---|
| a | 1 | 5 |
| b | 2 | 6 |
| c | 3 | 7 |
| d | 4 | 8 |

A negative literal means negation. [`Tdd::cube`](crate::Tdd::cube) conjoins
its literals, so `[1, -2, -3, -4]` is precisely `A(x)` above.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{and, and_exists, or, OperationError, Tdd, Vtree};
```

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(8));
// Indicators a,b,c,d use literals 1..4; next-state indicators use 5..8.
let at_a = Tdd::cube(&vtree, [1, -2, -3, -4])?;
let at_b = Tdd::cube(&vtree, [-1, 2, -3, -4])?;
let at_c = Tdd::cube(&vtree, [-1, -2, 3, -4])?;
let at_d = Tdd::cube(&vtree, [-1, -2, -3, 4])?;
let next_b = Tdd::cube(&vtree, [-5, 6, -7, -8])?;
let next_c = Tdd::cube(&vtree, [-5, -6, 7, -8])?;
```

Translate the three terms of `T` directly:

```rust,ignore,{class=tested-example}
let a_to_b = and(at_a.clone(), next_b.clone())?;
let b_to_c = and(at_b.clone(), next_c)?;
let c_to_b = and(at_c.clone(), next_b)?;
let transition = or(a_to_b, or(b_to_c, c_to_b)?)?;
```

Initially only A has been reached: `R₀(x) = A(x)`. We also name the current
variables and the mapping from next to current indicators for the search.
A `VarId` carries the same number as the integer literal, without a sign.

```rust,ignore,{class=tested-example}
let mut reached = at_a.clone();
let current = [VarId(1), VarId(2), VarId(3), VarId(4)];
let next_to_current = [
    (VarId(5), VarId(1)), (VarId(6), VarId(2)),
    (VarId(7), VarId(3)), (VarId(8), VarId(4)),
];
let mut iterations = 0;
```

## Compute successor states

A next state is a successor if **some** reached state has an edge to it:

```text
S(x′) = ∃a,b,c,d. (R(x) ∧ T(x, x′))
```

The existential quantifier removes the source indicators, keeping the
possible destinations. Starting with `R = A`, only the first edge is possible,
so this formula gives `S(x′) = B(x′)`.

Conjoin the reached set with the relation, then quantify the current variables:

```rust,ignore,{class=tested-example}
let possible_steps = and(reached.clone(), transition.clone())?;
let successors = possible_steps.exists_vars(&current)?;
```

The result uses next-state variables. Rename them so it can serve as a
current-state set in another step: `B(x′)` becomes `B(x)`.

```rust,ignore,{class=tested-example}
let successors = successors.rename_vars(&next_to_current)?;
```

[`and_exists`](crate::and_exists) combines the first two operations. Use it
to write an image helper for the search:

```rust,ignore,{class=tested-example}
fn image(
    states: Tdd,
    transition: Tdd,
    current: &[VarId],
    next_to_current: &[(VarId, VarId)],
) -> Result<Tdd, OperationError> {
    and_exists(states, transition, current)?.rename_vars(next_to_current)
}
```

## Repeat until the set stops growing

Add each image to the states already reached. Stop when
[`equivalent`](crate::Tdd::equivalent) says the set has not changed:

```rust,ignore,{class=tested-example}
loop {
    let successors = image(reached.clone(), transition.clone(), &current, &next_to_current)?;
    let enlarged = or(reached.clone(), successors)?;
    iterations += 1;

    // Count distinct current states, regardless of next-state assignments.
    let state_count = enlarged.projected_model_count(&current)?;
    println!("Iteration {iterations}: {state_count} reachable states");
    if enlarged.equivalent(&reached)? {
        break;
    }
    reached = enlarged;
    assert!(
        iterations < 4,
        "a four-state system must converge within four images"
    );
}
```

[`projected_model_count`](crate::Tdd::projected_model_count) counts only the
current-state assignments. Ordinary counting would also count the free
next-state indicators.

| Image step | Successors | Accumulated states |
|---|---|---|
| 1 | {B} | {A, B} |
| 2 | {B, C} | {A, B, C} |
| 3 | {B, C} | {A, B, C}: fixed point |

## Check a safety property

The final circuit represents `A(x) ∨ B(x) ∨ C(x)`. To check that D is
unreachable, ask whether every reached state satisfies `¬D(x)`:

```rust,ignore,{class=tested-example}
let safe = at_d.negate()?;
assert!(reached.implies(&safe)?);
println!("D is unreachable");
```

## Find a reachable target

Intersect the reached set with `C(x)` to find a reachable assignment at C:

```rust,ignore,{class=tested-example}
let reachable_target = and(reached, at_c)?;
let witness = reachable_target
    .satisfying_assignment()?
    .expect("C is reachable");
```

The current-state part of the witness is `a=false, b=false, c=true, d=false`.
It identifies C. Recovering a path to C would also require retaining
predecessor information during the search.

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/symbolic_reachability.rs)
also checks the complete reachable set against `A(x) ∨ B(x) ∨ C(x)` and verifies
the witness indicators.
