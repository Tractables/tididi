# Tree Decision Diagrams

A Tree Decision Diagram (TDD) is a canonical representation of a Boolean
function decomposed along a vtree, a binary tree over the variables. TDDs are
introduced and analyzed in Capelli, Choi, Mengel, Muñoz and Van den Broeck,
*A Canonical Generalization of OBDD* (<https://arxiv.org/abs/2604.05537>).
This document describes the data structure as the `tididi` crate stores it,
in the vocabulary of the [`Tdd`], [`TddLevel`], and [`InputPair`] types. The
operations are in [`docs/api-guide.md`](https://docs.rs/tididi/latest/tididi/guide/api/index.html).

## Vtree

A vtree is a rooted binary tree whose leaves are the Boolean variables. An
internal node `t` with children `t_L` and `t_R` partitions the variables
under `t` into `vars(t_L)` and `vars(t_R)`. A TDD node at `t` denotes a
function over `vars(t)` as a disjunction of products `f_left(vars(t_L)) ∧
f_right(vars(t_R))`.

A right-linear vtree, where every internal node's left child is a leaf, is a
variable order, and a TDD over it is an OBDD. A general vtree can group
related variables in one subtree and keep unrelated ones apart.

A vtree keeps its nodes in an array, and [`Vtree::bottomup()`], not the array
order, is the order to walk them in. A freshly built tree happens to number
every child below its parent, but rotating a tree relinks its nodes without
moving them, so a reader that iterates [`0..num_nodes()`] is wrong on any
rotated vtree. A vtree may leave variable ids unused: [`num_vars()`] is the id
space and [`num_leaves()`] the variables carried.

## Levels, nodes, and pairs

A [`Tdd`] stores one [`TddLevel`] per vtree node, read with [`Tdd::level(t)`] or
[`Tdd::levels()`]. A level is in one of three states.

- A leaf level stores nothing. Its three nodes are implicit and referenced
  by index from parent pairs: [`ONE_LEAF_IDX`] (the constant ⊤), [`POS_LEAF_IDX`]
  (the literal `x`), and [`NEG_LEAF_IDX`] (the literal `¬x`); [`LeafLabel`]
  names them. The constant-false atom is never stored.
- A structural level stores nodes in slots ([`TddLevel::nodes`]). Each node is a
  set of input pairs ([`InputPair { left, right }`]), where `left` indexes a node
  of the left child level and `right` a node of the right child level. A pair
  `(a, b)` denotes the rectangle `models(a) × models(b)`; a node denotes the
  union of its pairs' rectangles. Read a node's pairs through
  [`TddLevel::pairs_of`], which resolves both storage forms; the bit layout of a
  node word is documented on [`TddNodeData`].
- A marginal level has dropped its structure and keeps one model count per
  node in [`TddLevel::marginal_counts`] (see [Marginal levels](#marginal-levels)).

The `output` node ([`TddNodeId { vtree, local }`]) at the vtree root denotes
the whole function. The constant-false function is the one exception: it is
the [`ZERO`] sentinel in `output.local` alone ([`Tdd::is_zero`]), and no level
stores a node that computes false. Every counting, satisfiability, and
semiring path can therefore assume that every stored node is satisfiable.

This stored encoding is the public traversal contract, and it is read-only:
the invariants above are what every operation assumes without checking, so a
level cannot be edited from outside. The [`diagram`] module documentation
states what a reader may rely on and carries the worked walk against it;
`examples/statistic.rs` reads one statistic off the same encoding, and
[`TddBuilder`] is the one way to assemble a diagram by hand, checking the same
invariants as it goes.

## Semantics

Each node denotes a Boolean function over the variables of its vtree subtree.
A leaf atom denotes `x`, `¬x`, or ⊤. An internal node with pairs `{(a_i,
b_i)}` denotes `⋁_i (f_{a_i} ∧ f_{b_i})`. The left and right factors range
over the disjoint sets `vars(t_L)` and `vars(t_R)`, so each product is
decomposable.

## Determinism

At every vtree level the distinct nodes are pairwise mutually exclusive as
functions of their subtree variables: for distinct nodes `i, j` at level `t`,
`f_i ∧ f_j ≡ 0`. The nodes at `t` partition the assignment space of `vars(t)`
by the function's cofactor outside `t`. Within one node's pair list the
disjunction is therefore a disjoint union, and the model count of a node is
the sum over its pairs of the product of the child counts.

TDDs are not strongly deterministic: there is no exhaustiveness guarantee and
no per-node sibling structure. The guarantee is the global one that distinct
nodes at a level compute disjoint functions. Rewrites that merge two pairs
with a shared side into one pair over a disjunction are not valid, because
the union of two partition cells is not a partition cell.

## Canonical reduced form

[`minimize`] reduces a diagram in two passes.

1. Prune removes nodes not reachable from `output`: a top-down mark, then a
   bottom-up compaction with a monotone index remap.
2. Twin contraction merges nodes at one level that compute the same function.

   | Rule | Before → After | Fires when | Sound because |
   |---|---|---|---|
   | Leaf twin contraction | `(Pos_x, S), (Neg_x, S)` → `(One_x, S)` | a parent pairs both polarities of `x` with the same partner `S` | `x ∨ ¬x = ⊤`, so the two pairs cover `S` regardless of `x` |
   | Internal twin merge | two nodes with identical parent contexts → one node, pair lists unioned | two same-level nodes are referenced from identical `(parent, sibling)` contexts | identical contexts imply identical functions; determinism keeps the union disjoint |

Contraction propagates sideways to a sibling and downward to descendants,
never upward, so one parents-before-children sweep reaches the fixpoint.

The result is the canonical reduced form: no false nodes, no unreachable
nodes, no two nodes at a level computing the same function, and the leaf
atoms in their fixed order. It is a smooth form: every vtree level is present,
whether or not the function depends on it.

## Canonicity

For a fixed vtree the minimized TDD is canonical: two TDDs computing the same
function over the same vtree reduce to the identical diagram, up to the
order in which same-level nodes are listed. Apply produces canonical output
by construction, since its compacting product never emits two nodes
computing the same function.

## Size guarantee

A CNF of treewidth `k` admits a TDD of size linear in the formula and
exponential only in `k`, under a vtree derived from a width-`k` tree
decomposition. The bound is an upper bound on size: a low-treewidth formula
is guaranteed a compact TDD, and the bound says nothing about other formulas.

## Marginal levels

`docs/marginal_example.svg` in the repository shows one small diagram before
and after a level is summed out. When only a count is needed, a level whose
structure can no longer change may be summed out: its nodes and pairs are
discarded and replaced by one model count per node in
[`TddLevel::marginal_counts`] (`u128`, with an overflow sentinel whose exact
value lives in [`TddLevel::marginal_counts_big`]). With a [`WeightStore`]
attached the level is weight-marginal instead
([`TddLevel::is_weight_marginal`]) and its per-node semiring values live in
the store. A marginal node keeps only its value, so two marginal nodes with
equal values are interchangeable. A pair whose child level is marginal
refers to the child either by table index or by the count itself held inline
in the pair. Build the child's [`SideView`]
([`TddLevel::side_view`]) once and decode every side of that level through it;
it yields a [`ChildRef`], either a node of a structural child or a [`ValueRef`] —
[`Slot`] or [`Inline`] — of a marginal one. [`SideView`] is the supported way
to read such a side; the bit layout behind it is not public.

The set of marginal levels is downward-closed in the vtree: below a marginal
level every level is marginal or a leaf. A level is made marginal only after
all its descendants are, which is what lets the storage below it be freed.
Marginalization changes the representation, not the function denoted, and it
is sound because the disjointness that justifies summing counts is
established before any structure is discarded and never violated afterward.

Both value domains — the integer counts stored in the level and the semiring
values kept in an attached [`WeightStore`] — are summed out by one pass over
the vtree, which differs between them only in what a node's value is and where
the finished values are kept. A vtree leaf is summed out by lookup alone,
since its value is fixed by its label.

[`0..num_nodes()`]: crate::Vtree::num_nodes
[`ChildRef`]: crate::diagram::ChildRef
[`Inline`]: crate::diagram::ValueRef::Inline
[`InputPair`]: crate::diagram::InputPair
[`InputPair { left, right }`]: crate::diagram::InputPair
[`LeafLabel`]: crate::diagram::LeafLabel
[`NEG_LEAF_IDX`]: crate::diagram::NEG_LEAF_IDX
[`ONE_LEAF_IDX`]: crate::diagram::ONE_LEAF_IDX
[`POS_LEAF_IDX`]: crate::diagram::POS_LEAF_IDX
[`SideView`]: crate::diagram::SideView
[`Slot`]: crate::diagram::ValueRef::Slot
[`Tdd`]: crate::Tdd
[`Tdd::is_zero`]: crate::Tdd::is_zero
[`Tdd::level(t)`]: crate::Tdd::level
[`Tdd::levels()`]: crate::Tdd::levels
[`TddBuilder`]: crate::diagram::TddBuilder
[`TddLevel`]: crate::diagram::TddLevel
[`TddLevel::is_weight_marginal`]: crate::diagram::TddLevel::is_weight_marginal
[`TddLevel::marginal_counts`]: crate::diagram::TddLevel::marginal_counts
[`TddLevel::marginal_counts_big`]: crate::diagram::TddLevel::marginal_counts_big
[`TddLevel::nodes`]: crate::diagram::TddLevel::nodes
[`TddLevel::pairs_of`]: crate::diagram::TddLevel::pairs_of
[`TddLevel::side_view`]: crate::diagram::TddLevel::side_view
[`TddNodeData`]: crate::diagram::TddNodeData
[`TddNodeId { vtree, local }`]: crate::diagram::TddNodeId
[`ValueRef`]: crate::diagram::ValueRef
[`Vtree::bottomup()`]: crate::Vtree::bottomup
[`WeightStore`]: crate::diagram::WeightStore
[`ZERO`]: crate::diagram::ZERO
[`diagram`]: crate::diagram
[`minimize`]: crate::reduce::minimize
[`num_leaves()`]: crate::Vtree::num_leaves
[`num_vars()`]: crate::Vtree::num_vars
