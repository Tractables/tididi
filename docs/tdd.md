# Tree Decision Diagrams

A **Tree Decision Diagram (TDD)** is a canonical, tractable representation of a
Boolean function. It generalizes the OBDD by replacing the linear variable order
with a **vtree** — a binary tree over the variables — so that the function is
decomposed along a tree rather than a chain. TDDs are introduced and analyzed in
Capelli, Choi, Mengel, Muñoz & Van den Broeck, *A Canonical Generalization of
OBDD* (<https://arxiv.org/abs/2604.05537>); this document describes the data
structure as implemented in the `tididi` crate. For the API, see
[api-guide.md](api-guide.md).

## Vtree

A vtree is a rooted binary tree whose leaves are the Boolean variables and whose
internal nodes carry no variable of their own. Each internal vtree node `t` with
children `t_L` and `t_R` induces a partition of the variables under `t` into
`vars(t_L)` and `vars(t_R)`. This partition is what the TDD decomposes along: a
TDD node at `t` expresses a subfunction over `vars(t)` as a disjunction of
products `f_left(vars(t_L)) ∧ f_right(vars(t_R))`.

The vtree plays the role that a variable order plays for OBDDs, but with more
freedom: a variable order is a *total* order, whereas a vtree is a *tree*, which
can group related variables into a subtree and keep unrelated ones apart. A
**right-linear vtree** — where every internal node's left child is a single
leaf — corresponds exactly to a linear variable order, and a TDD over a
right-linear vtree is an OBDD. OBDDs are therefore the special case of TDDs whose
vtree is a chain.

Vtree nodes are stored leaves-first in bottom-up level order, so every edge has
`child.idx() < parent.idx()`. This gives O(1) bottom-up traversal and O(depth)
lowest-common-ancestor queries. The crate ships several constructors
(`Vtree::balanced`, `Vtree::linear`, `Vtree::random`, and the SDD `.vtree` file
format), plus in-place left/right rotations. Quality-driven vtree *construction*
(treewidth- and partition-based heuristics) lives in
[vitri](https://github.com/Tractables/vitri), and CNF compilation in the
`tididi-cnf` solver that drives the two. Neither is part of this crate.

## Node and level structure

A TDD is stored as one **level** per vtree node, indexed by vtree position:

- **Leaf levels** store nothing. The three possible atoms at a variable `x` are
  *implicit*, referenced by index from parent pairs: `One` (the constant ⊤, index
  0), `Pos` (the literal `x`, index 1), and `Neg` (the literal `¬x`, index 2).
  The constant-false atom `Zero` is never stored — see the output paragraph below.

- **Internal levels** store a list of **t-nodes**. Each t-node is a set of
  **input pairs** `(left, right)`, where `left` indexes a node in the left child
  level and `right` indexes a node in the right child level. A pair `(a, b)`
  denotes the rectangle of assignments `models(a) × models(b)`; a t-node denotes
  the union of its pairs' rectangles.

A distinguished **output** node at the vtree root denotes the whole function. The
constant-false function is the sole exception: it is represented by a virtual
sentinel index in the output pointer alone, and no level ever stores a
false-computing node. Every counting, SAT, and semiring-evaluation path can
therefore assume every stored node is satisfiable, with no zero-node special case.

## Semantics

Each node denotes a Boolean function over the variables in its vtree subtree:

- A leaf atom denotes `x` (`Pos`), `¬x` (`Neg`), or `⊤` (`One`).
- An internal t-node with pairs `{(a_i, b_i)}` denotes
  `⋁_i (f_{a_i} ∧ f_{b_i})` — a disjunction of decomposable conjunctions. The
  left and right factors range over disjoint variable sets (`vars(t_L)`,
  `vars(t_R)`), so each product is **decomposable**.

Because every product splits along the same vtree partition and the disjuncts are
pairwise disjoint (see determinism), the model count of a node is a simple sum of
products of child counts — the basis for tractable counting, SAT, and weighted
counting.

## Determinism

At every vtree level, the distinct t-nodes are **pairwise mutually exclusive** as
functions of their subtree variables: for two distinct nodes `i, j` at level `t`,
`f_i ∧ f_j ≡ 0`. Equivalently, the nodes at `t` form a *partition* of the
inside-`t` assignment space by their outside-`t` cofactor of the function. Within
a single node's pair list this makes the disjunction a disjoint union, so counts
add without inclusion-exclusion.

TDDs are deterministic in this sense but **not strongly deterministic**: unlike an
SDD, the crate makes no exhaustiveness guarantee and imposes no per-node
sibling-mutex structure. The mutual exclusivity is the *global* property that
distinct nodes at a level compute disjoint functions — not an SDD-style claim that
the "primes" within one node are both mutex and cover ⊤. Rewrite rules valid for
SDDs (e.g. merging two same-right pairs into `(α_1 ∨ α_2, β)`) are therefore
**not** valid for TDDs, because the union of two partition cells is not itself a
partition cell.

## Canonical reduced form

`minimize` reduces a TDD to canonical form by two passes:

1. **Prune** removes unreachable nodes (a top-down mark from the output, then a
   bottom-up compaction with a monotone index remap).

2. **Twin contraction** merges nodes at the same level that compute the same
   function. Two flavors fire:

   | Rule | Before → After | Fires when | Sound because |
   |---|---|---|---|
   | Leaf twin contraction | `(Pos_x, S), (Neg_x, S)` → `(One_x, S)` | a parent pairs both polarities of `x` with the *same* partner `S` | `x ∨ ¬x = ⊤`, so the two pairs cover `S` regardless of `x` |
   | Internal twin merge | two nodes with identical parent-context multisets → one node (pair lists unioned) | two same-level nodes are referenced from identical `(parent, sibling)` contexts | identical contexts ⇒ identical functions; determinism keeps the unioned pair list disjoint |

Contraction propagates only sideways (to a sibling) and downward (to descendants),
never upward, so a single parents-before-children sweep reaches the fixpoint.

The result is the **canonical smooth reduced form**: no false nodes, no
unreachable nodes, no two nodes at a level computing the same function, and
canonical leaf ordering. A level can also be recognized as *semantically
redundant* when all its pairs share one child side and the other side's node
counts sum to full coverage of that subtree (the function then depends only on the
shared child) — this is the TDD analogue of OBDD variable elimination and the
measurable non-smooth reduction, distinct from the smooth canonical form that
`minimize` produces.

## Canonicity

For a **fixed vtree**, the minimized TDD is canonical: two TDDs computing the same
Boolean function over the same vtree reduce to the identical diagram (up to the
order in which same-level nodes are listed). This is the canonicity theorem of the
paper and is what makes equivalence checking, and the deduplication-free apply,
possible: apply's compacting product never emits two nodes computing the same
function (distinct product cells yield *disjoint* pair sets), so its output is
canonical by construction and the crate has no compress/dedup pass at all.
Canonicity is verified in the test suite by probabilistic
polynomial-identity testing (Schwartz–Zippel over a large prime field), which
gives distinct signatures to distinct functions with overwhelming probability.

## Size guarantee

TDD size is governed by the vtree, and a good vtree can be read off the CNF's
structure. In particular, a CNF of treewidth `k` admits a TDD of size **fixed-
parameter tractable in `k`** (linear in the formula, exponential only in `k`),
for a vtree derived from a width-`k` tree decomposition — the paper's FPT-size
result. Treewidth here is an *upper bound* on representation size: a low-treewidth
formula is guaranteed a compact TDD. High-treewidth formulas are not excluded —
the bound simply says nothing about them, and many compile compactly in practice.

## Marginal levels (counting mode)

When only a model count (or weighted count) is needed, a level whose Boolean
structure can no longer change may be flipped from *structural* to **marginal**: its
stored nodes and pairs are discarded and replaced by a per-node vector of model
counts (arbitrary precision, with a `u128` fast path). A marginal node keeps only
its count — the anonymous identity that downstream counting reads — so two marginal
nodes with equal count are interchangeable.

The set of marginal levels is **downward-closed** in the vtree: if a level is
marginal, so is every descendant (or leaf). This holds by construction — a level is
made marginal only after all of its descendants are — and it is what lets the
structural arenas below a marginal level be freed. Marginalization is a
counting-mode optimization on the *representation*; it does not change which
function the diagram denotes, and it is sound precisely because the atom-level
disjointness that justifies summing counts is established before any structure is
discarded and is never violated afterward.

---

See [api-guide.md](api-guide.md) for how to build and query TDDs, and the paper
(<https://arxiv.org/abs/2604.05537>) for the formal development.
