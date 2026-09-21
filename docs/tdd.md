# Tree Decision Diagrams

A Tree Decision Diagram represents a Boolean function by decomposing it along
a vtree. Its nodes describe sets of satisfying assignments, which can be
counted or evaluated under weights. The [API overview] connects this
representation to library operations.

## Start with the vtree

A **vtree** is a binary tree with one Boolean variable at each leaf. Each
internal node splits its variables into a left group and a right group. For
example, a balanced vtree over four variables groups `{x1, x2}` on one side
of the root and `{x3, x4}` on the other.

[`Vtree`] determines both this decomposition and the universe of assignments:
a variable is still part of that universe when a function does not depend on
it.
The [grouping walkthrough](crate::guide::examples::vtrees) compares two vtrees
for the same function and explains their different storage requirements.

## From variables to a function

A [`Tdd`] has one **level** per vtree node. A leaf level provides the functions
`x`, `¬x`, and `true`; an internal level contains nodes made of **pairs**.
Each pair refers to a function from the left child level and one from the
right child level, and denotes their conjunction. A node denotes the
disjunction of its pairs. The output node denotes the whole function.

For example, with `x` and `y` at the two leaves, a node with the pairs
`(x, ¬y)` and `(¬x, y)` represents exclusive disjunction. Each pair describes
one way to satisfy the node. At a larger level, either child can itself be a
node containing several pairs.

![The vtree splits x and y; the TDD output joins the pairs (x, not y) and (not x, y).](https://raw.githubusercontent.com/Tractables/tididi/v0.1.0/docs/tdd-basics.svg)

The yellow dots are pairs, not additional TDD nodes: each selects one child
from the `x` level and one from the `y` level. The blue output node takes
the disjunction of those two conjunctions.

## Why the counts add up

In a structural TDD, a given pair of child nodes belongs to at most one node
at its parent level. Distinct pairs may share either child, but not both.
A leaf level always offers `x`, `¬x` and `true`; the rule at a leaf is that
the level as a whole refers either to `true` or to the literals, never both.
The rule is level-global, not per-parent: two parent nodes at the same level
may not disagree, because `x ∧ true` is not `⊥` and the two would overlap.
Together these syntactic rules make the functions of distinct nodes at a
level disjoint.

The factors in a pair use disjoint variable sets, and the pairs of a node
describe disjoint assignments. Counting therefore multiplies the two child
counts for each pair, then adds the results. A literal contributes one;
a free leaf contributes two. The exclusive-disjunction example has count
`1 × 1 + 1 × 1 = 2`.

## Minimization and equality

[`minimize`] removes unreachable nodes and merges twins: nodes used with
exactly the same siblings in every parent context. Their functions need not
be equal; their union can replace them because the rest of the diagram
uses them identically. For example, leaf twins `x` and `¬x` merge into `true`.

The resulting structural TDD is minimal and canonical for the fixed vtree,
up to node numbering and pair order. This is the minimization result of
[the TDD paper, Section 5]. Changing the vtree can give a different size;
[`Tdd::rotation_search`] searches such changes, while
[`Tdd::equivalent`] compares functions without relying on node identifiers.

## When only a value is needed

[`marginalize_levels`] replaces a subtree's structure with per-node counts,
or with weighted values if the diagram has an attached [`WeightStore`].
This preserves the chosen evaluation but discards the assignments behind it:
keep the structure if you will need to recover assignments or change weights
later.

Marginal levels can share equal values and carry pair multiplicities. Their
storage follows the separate rules in the [`diagram`] reference.

## Reading the stored representation

[`Tdd::output`] identifies the output, with [`Tdd::is_zero`] recognizing the
constant-false sentinel before any level is indexed.
The [`diagram`] module documents the leaf, structural, and marginal encodings
and gives traversal examples.
Use [`Vtree::bottomup`] for child-before-parent order: rotations preserve node
indices, so array order is not a traversal order.
[`TddBuilder`] checks storage when assembling a diagram by hand; its author
must also establish the determinism contract documented on that type.

[API overview]: crate::guide::api
[the TDD paper, Section 5]: https://arxiv.org/html/2604.05537v1#S5
[`Vtree`]: crate::Vtree
[`Tdd`]: crate::Tdd
[`minimize`]: crate::Tdd::minimize
[`Tdd::rotation_search`]: crate::Tdd::rotation_search
[`Tdd::equivalent`]: crate::Tdd::equivalent
[`marginalize_levels`]: crate::Tdd::marginalize_levels
[`WeightStore`]: crate::diagram::WeightStore
[`Tdd::output`]: crate::Tdd::output
[`Tdd::is_zero`]: crate::Tdd::is_zero
[`diagram`]: crate::diagram
[`Vtree::bottomup`]: crate::Vtree::bottomup
[`TddBuilder`]: crate::diagram::TddBuilder
