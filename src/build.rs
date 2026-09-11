//! Constants, literals and clauses as diagrams.
//!
//! These are the leaves of every compilation: everything else is built by
//! combining them with [`crate::apply`] and reducing with [`crate::reduce`].
//!
//! Entry points: [`Tdd::clause`] builds a minimal, canonical diagram for one
//! clause directly, without a raw build and a reduction afterwards;
//! [`Engine::cube`] does the same for a conjunction of literals; [`Tdd::one`] and
//! [`Tdd::zero`] are the two constants. Each has an [`Engine`]
//! form that runs under the caller's limits.

use crate::limits::pool::Pool;
use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{Vtree, VtreeIdx};
use crate::engine::Engine;

use crate::diagram::{self, *};

/// No node has been built for this level yet.
const UNSET: u32 = u32::MAX;

/// Every buffer one engine's clause builds reuse between calls.
///
/// See [`Pool`] for the checkout pattern.
#[derive(Default)]
pub(crate) struct BuildScratch {
    /// Per-level index of the clause-satisfied node (c_t), or `UNSET`.
    clause_idx: Pool<Vec<u32>>,
    /// Per-level index of the complement node (d_t), or `UNSET`.
    complement_idx: Pool<Vec<u32>>,
    /// Per-level flag: true if the clause node (c_t) is absent (subtree irrelevant).
    irrelevant: Pool<Vec<bool>>,
    /// Post-order (children-before-parents) list of the vtree's internal nodes,
    /// rebuilt per call. Pooled because a fresh `Vec` would re-grow from zero
    /// on every clause build, which on a vtree with hundreds of thousands of
    /// levels is a full doubling ladder of allocations and copies.
    internal_postorder: Pool<Vec<(VtreeIdx, VtreeIdx, VtreeIdx)>>,
    /// Work stack for the post-order walk above.
    postorder_stack: Pool<Vec<VtreeIdx>>,
}

impl BuildScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.clause_idx.drain();
        self.complement_idx.drain();
        self.irrelevant.drain();
        self.internal_postorder.drain();
        self.postorder_stack.drain();
    }
}

/// Build a diagram computing the constant-false function (no assignment satisfies it).
/// Output points to the `ZERO` sentinel (`u32::MAX`) — no actual nodes are created.
pub(crate) fn constant_zero(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let levels = diagram::take_levels(eng, vtree.num_nodes());
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: ZERO },
    )
}

/// Build a diagram computing the constant-true function (all assignments satisfy it).
/// Width 1 at every internal vtree level (only one node at index 0).
/// Leaf levels are marginal (no stored nodes); One is at index 0 (`ONE_LEAF_IDX`).
pub(crate) fn constant_one(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let mut levels = diagram::take_levels(eng, vtree.num_nodes());

    // Internal levels: each has one node pairing the child's "true" node.
    // Leaf children reference One (`ONE_LEAF_IDX`); internal children reference
    // their single node at index 0.
    for (t, left, right) in vtree.internal_bottomup() {
        let left_child_idx = if vtree.node(left).is_leaf() {
            ONE_LEAF_IDX
        } else {
            NodeIdx(0)
        };
        let right_child_idx = if vtree.node(right).is_leaf() {
            ONE_LEAF_IDX
        } else {
            NodeIdx(0)
        };
        let pair = InputPair { left: left_child_idx, right: right_child_idx };
        levels[t.idx()].push_internal_node(&[pair]);
    }

    // Output: One (for single-variable vtrees) or the sole internal node (index 0).
    let out_local = if vtree.node(vtree.root()).is_leaf() {
        ONE_LEAF_IDX
    } else {
        NodeIdx(0)
    };
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: out_local },
    )
}

/// Build a minimal, canonical diagram representing a single clause.
///
/// At each vtree level, two nodes are maintained bottom-up:
/// - **`c_t`** (clause node): "at least one literal in t's subtree satisfies
///   the clause"
/// - **`d_t`** (complement node): "no literal in t's subtree satisfies the
///   clause" — only created at levels strictly below the LCA of clause vars
///
/// `d_t` is needed only when an ancestor's `c_t` uses both-relevant pairs
/// (requiring d from each child). This only happens at or below the LCA.
/// At the LCA itself and above, only one child is relevant, so `d_t` is never
/// referenced. By finding the LCA first and skipping `d_t` at/above it,
/// unreachable nodes are eliminated by construction — no prune pass needed.
///
/// Irrelevant subtrees (no clause variables) get a single One node instead.
///
/// The result satisfies all diagram invariants: no false nodes, no unreachable
/// nodes, canonical (no duplicates, no redundant pairs).
pub(crate) fn clause_to_tdd(eng: &Engine, vtree: &Arc<Vtree>, clause: &[Literal]) -> Tdd {
    // A variable named in both polarities satisfies the disjunction whatever
    // its value, and the construction below keeps one `c_t`/`d_t` column per
    // variable, which cannot say that.
    if crate::diagram::is_tautological(clause) {
        return constant_one(eng, vtree);
    }
    let num_nodes = vtree.num_nodes();
    let mut levels = diagram::take_levels(eng, num_nodes);
    let mut scratch = ClauseScratch::take(eng.build(), num_nodes);

    seed_leaf_levels(
        vtree,
        clause,
        &mut scratch.clause_idx,
        &mut scratch.complement_idx,
        &mut scratch.irrelevant,
    );
    collect_internal_postorder(
        vtree,
        &mut scratch.internal_postorder,
        &mut scratch.postorder_stack,
    );
    let lca_postorder_pos =
        mark_irrelevant_and_find_lca(&scratch.internal_postorder, &mut scratch.irrelevant);
    build_internal_levels(
        &mut levels,
        &scratch.internal_postorder,
        lca_postorder_pos,
        &mut scratch.clause_idx,
        &mut scratch.complement_idx,
        &scratch.irrelevant,
    );

    // If the root's entire subtree is irrelevant (no clause variables at all),
    // c_t was never created — return `ZERO`. In practice this path is unreachable:
    // clause_scope() panics on empty clauses, and all clause variables must
    // exist in the vtree. Kept as a defensive fallback.
    let root_idx = vtree.root().idx();
    let out_local = if scratch.irrelevant[root_idx] {
        ZERO
    } else {
        clause_satisfied_idx(
            root_idx,
            &scratch.irrelevant,
            &scratch.clause_idx,
            &scratch.complement_idx,
        )
    };

    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: out_local },
    )
}

/// The pooled scratch buffers one [`Tdd::clause`] call works in. Checked out
/// together and returned by `Drop`, so no exit from the build can skip the
/// return.
///
/// Per-level tracking for the bottom-up construction:
///   `clause_idx[t]`     — local index of c_t (clause-satisfied node), `UNSET`
///   `complement_idx[t]` — local index of d_t (complement node) or identity node
///                         at irrelevant levels (`UNSET`)
///   `irrelevant[t]`     — true if c_t is absent (no clause vars in this subtree)
struct ClauseScratch<'a> {
    pool: &'a BuildScratch,
    clause_idx: Vec<u32>,
    complement_idx: Vec<u32>,
    irrelevant: Vec<bool>,
    internal_postorder: Vec<(VtreeIdx, VtreeIdx, VtreeIdx)>,
    postorder_stack: Vec<VtreeIdx>,
}

impl<'a> ClauseScratch<'a> {
    /// Take the buffers from their pools, sized for `num_nodes` levels. The
    /// `resize` extends capacity if a previous call left the buffer shorter,
    /// then `fill` resets the values that call left behind.
    fn take(pool: &'a BuildScratch, num_nodes: usize) -> ClauseScratch<'a> {
        let mut clause_idx = pool.clause_idx.take();
        if clause_idx.len() < num_nodes { clause_idx.resize(num_nodes, UNSET); }
        clause_idx[..num_nodes].fill(UNSET);

        let mut complement_idx = pool.complement_idx.take();
        if complement_idx.len() < num_nodes { complement_idx.resize(num_nodes, UNSET); }
        complement_idx[..num_nodes].fill(UNSET);

        let mut irrelevant = pool.irrelevant.take();
        if irrelevant.len() < num_nodes { irrelevant.resize(num_nodes, false); }
        irrelevant[..num_nodes].fill(false);

        ClauseScratch {
            pool,
            clause_idx,
            complement_idx,
            irrelevant,
            internal_postorder: pool.internal_postorder.take(),
            postorder_stack: pool.postorder_stack.take(),
        }
    }
}

impl Drop for ClauseScratch<'_> {
    fn drop(&mut self) {
        self.pool.internal_postorder.put(std::mem::take(&mut self.internal_postorder));
        self.pool.postorder_stack.put(std::mem::take(&mut self.postorder_stack));
        self.pool.clause_idx.put(std::mem::take(&mut self.clause_idx));
        self.pool.complement_idx.put(std::mem::take(&mut self.complement_idx));
        self.pool.irrelevant.put(std::mem::take(&mut self.irrelevant));
    }
}

/// Get the clause-satisfied node (`c_t`) index for a child level.
///
/// At relevant levels, `c_t` is stored directly in `clause_idx`. At
/// irrelevant levels (no clause vars in subtree), the only node is the
/// one identity, stored in `complement_idx`.
#[inline]
fn clause_satisfied_idx(t: usize, irrelevant: &[bool], clause_idx: &[u32], complement_idx: &[u32]) -> NodeIdx {
    if irrelevant[t] {
        NodeIdx(complement_idx[t]) // One (identity) at this irrelevant level
    } else {
        NodeIdx(clause_idx[t])
    }
}

/// Set the implicit c_t/d_t indices at every leaf level (One=0, Pos=1, Neg=2).
/// Leaf levels are marginal, so no nodes are created here.
///
/// Two passes, not a search per leaf. Every leaf gets the irrelevant/One seed
/// first; then the clause's own literals — the only leaves that differ —
/// overwrite theirs, addressed through `var_to_leaf` in O(1). The version
/// this replaced ran `clause.iter().find(|l| l.var == var)` once per leaf,
/// i.e. O(#leaves × clause length) per clause; on a vtree whose leaf count
/// runs far ahead of any one clause's support that search dominated the
/// build. Same leaf set as before (`leaf_bottomup`), so no assumption about
/// where leaves sit in the index space is introduced.
fn seed_leaf_levels(
    vtree: &Vtree,
    clause: &[Literal],
    clause_idx: &mut [u32],
    complement_idx: &mut [u32],
    irrelevant: &mut [bool],
) {
    for (t, _var) in vtree.leaf_bottomup() {
        let t_idx = t.idx();
        complement_idx[t_idx] = ONE_LEAF_IDX.0;
        irrelevant[t_idx] = true;
    }
    for lit in clause {
        let t_idx = vtree.leaf_of(lit.var).expect("the vtree carries this variable").idx();
        // First literal on a variable wins, exactly as the `find` this replaced
        // did — a clause carrying both polarities of one variable must not have
        // its seed rewritten by the second occurrence. `clause_idx` is still
        // `UNSET` at every leaf (the seed above writes only the other two
        // arrays), so "untouched" is exactly "not yet claimed".
        if clause_idx[t_idx] != UNSET {
            continue;
        }
        //   positive literal → c_t=Pos(1), d_t=Neg(2)
        //   negative literal → c_t=Neg(2), d_t=Pos(1)
        if lit.positive {
            clause_idx[t_idx] = POS_LEAF_IDX.0;
            complement_idx[t_idx] = NEG_LEAF_IDX.0;
        } else {
            clause_idx[t_idx] = NEG_LEAF_IDX.0;
            complement_idx[t_idx] = POS_LEAF_IDX.0;
        }
        irrelevant[t_idx] = false;
    }
}

/// Fill `out` with the vtree's internal nodes in post-order (children before
/// parents) via DFS, which is robust against stale topo ordering after a vtree
/// rotation. `stack` is the pooled DFS stack.
fn collect_internal_postorder(
    vtree: &Vtree,
    out: &mut Vec<(VtreeIdx, VtreeIdx, VtreeIdx)>,
    stack: &mut Vec<VtreeIdx>,
) {
    out.clear();
    stack.clear();
    stack.push(vtree.root());
    while let Some(idx) = stack.pop() {
        if let crate::vtree::VtreeNode::Internal { left, right, .. } = *vtree.node(idx) {
            out.push((idx, left, right));
            stack.push(right);
            stack.push(left);
        }
    }
    out.reverse();
}

/// Propagate irrelevant flags bottom-up and return the post-order position of
/// the LCA (the highest node where both children have clause variables). Using
/// post-order position, not vtree index, since rotation can make child indices
/// exceed their parent's.
fn mark_irrelevant_and_find_lca(
    internal_postorder: &[(VtreeIdx, VtreeIdx, VtreeIdx)],
    irrelevant: &mut [bool],
) -> Option<usize> {
    let mut lca_postorder_pos: Option<usize> = None;
    for (pos, &(t, left, right)) in internal_postorder.iter().enumerate() {
        let left_idx = left.idx();
        let right_idx = right.idx();
        if irrelevant[left_idx] && irrelevant[right_idx] {
            irrelevant[t.idx()] = true;
        } else if !irrelevant[left_idx] && !irrelevant[right_idx] {
            lca_postorder_pos = Some(pos);
        }
    }
    lca_postorder_pos
}

/// The bottom-up construction: build `c_t` (and, strictly below the LCA, `d_t`)
/// at every internal level, recording their local indices.
fn build_internal_levels(
    levels: &mut [TddLevel],
    internal_postorder: &[(VtreeIdx, VtreeIdx, VtreeIdx)],
    lca_postorder_pos: Option<usize>,
    clause_idx: &mut [u32],
    complement_idx: &mut [u32],
    irrelevant: &[bool],
) {
    for (pos, &(t, left, right)) in internal_postorder.iter().enumerate() {
        let t_idx = t.idx();
        let left_idx = left.idx();
        let right_idx = right.idx();
        let left_c = clause_satisfied_idx(left_idx, irrelevant, clause_idx, complement_idx);
        let right_c = clause_satisfied_idx(right_idx, irrelevant, clause_idx, complement_idx);

        let level = &mut levels[t_idx];

        if irrelevant[left_idx] && irrelevant[right_idx] {
            // Both subtrees irrelevant: only one node (identity).
            let left_d = NodeIdx(complement_idx[left_idx]);
            let right_d = NodeIdx(complement_idx[right_idx]);
            let pair = InputPair { left: left_d, right: right_d };
            let one = level.push_internal_node(&[pair]);
            complement_idx[t_idx] = one.0;
        } else {
            // At least one subtree has clause variables.
            // complement_idx is always valid for irrelevant children (identity
            // node) and for relevant children below the LCA (d_t node).
            let left_d = NodeIdx(complement_idx[left_idx]);
            let right_d = NodeIdx(complement_idx[right_idx]);

            let c_pairs: Vec<InputPair> = match (irrelevant[left_idx], irrelevant[right_idx]) {
                (true, false) => vec![InputPair { left: left_d, right: right_c }],
                (false, true) => vec![InputPair { left: left_c, right: right_d }],
                _ => vec![
                    InputPair { left: left_c, right: right_c },
                    InputPair { left: left_c, right: right_d },
                    InputPair { left: left_d, right: right_c },
                ],
            };
            // No sort: pair lists are unordered sets (see `InputPair`) and
            // twin contraction is order-independent, so the clause node's pair
            // order is never consumed.
            level.push_internal_node(&c_pairs);
            clause_idx[t_idx] = 0;

            // d_t: "neither subtree satisfies" = d_left ∧ d_right.
            // Only needed below the LCA — at the LCA and above, no ancestor
            // references this level's d_t. Uses postorder position (not vtree
            // index) for the comparison since rotation can invert index order.
            if lca_postorder_pos.is_some_and(|lca| pos < lca) {
                let d_pair = InputPair { left: left_d, right: right_d };
                let d = level.push_internal_node(&[d_pair]);
                complement_idx[t_idx] = d.0;
            }
        }
    }
}

/// Build the cube diagram: one width-1 node per internal vtree node, whose
/// pair names the leaf label the cube assigns to each side's subtree.
///
/// Bottom-up, so the pair a node writes names children that already exist.
fn cube_to_tdd(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    literals: impl IntoIterator<Item = impl Into<Literal>>,
) -> Tdd {
    let mut label = vec![ONE_LEAF_IDX; vtree.num_nodes()];
    for lit in literals {
        let lit: Literal = lit.into();
        let leaf = vtree
            .leaf_of(lit.var)
            .expect("the cube names a variable this vtree has no leaf for");
        assert_eq!(
            label[leaf.idx()], ONE_LEAF_IDX,
            "the cube names variable {:?} twice",
            lit.var,
        );
        label[leaf.idx()] = if lit.positive { POS_LEAF_IDX } else { NEG_LEAF_IDX };
    }
    let mut b = Tdd::build(eng, vtree);
    for (t, left, right) in vtree.internal_bottomup() {
        label[t.idx()] = b.push(t, &[InputPair {
            left: label[left.idx()],
            right: label[right.idx()],
        }]);
    }
    let root = vtree.root();
    b.finish(TddNodeId { vtree: root, local: label[root.idx()] })
        .expect("a cube names one node per internal level and seats the root on it")
}

impl Tdd {
    /// Build a canonical diagram for a single clause from DIMACS-style literals.
    ///
    /// Sugar over [`Engine::clause`](crate::engine::Engine::clause), built on a
    /// transient engine. Each item is converted with [`Into<Literal>`], so plain
    /// integers use the 1-based DIMACS sign convention (`1` → `x1`, `-2` → `¬x2`;
    /// see [`Literal`]).
    ///
    /// The literals are a set: a variable repeated in one polarity builds the
    /// clause the deduplicated literals spell, and a variable in both
    /// polarities builds ⊤.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, -2]); // x1 ∨ ¬x2
    /// # let _ = f;
    /// ```
    pub fn clause(vtree: &Arc<Vtree>, literals: impl IntoIterator<Item = impl Into<Literal>>) -> Tdd {
        Engine::new().clause(vtree, literals)
    }

    /// The constant-true function over `vtree`: every assignment satisfies it.
    ///
    /// Width 1 at every internal vtree level; the leaf levels are marginal.
    pub fn one(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().one(vtree)
    }

    /// The constant-false function over `vtree`: no assignment satisfies it.
    ///
    /// The output points at the `ZERO` sentinel, so no nodes are created.
    pub fn zero(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().zero(vtree)
    }
}

/// The construction entry points on a caller's engine, where the per-level
/// buffers stay warm between calls.
impl crate::engine::Engine {
    /// A diagram for one clause over `vtree`, built in this engine's pools.
    ///
    /// The engine-owned form of [`Tdd::clause`]; identical result, and the
    /// per-level buffers stay warm for the next clause. The literals are a set,
    /// as in [`Tdd::clause`].
    #[must_use]
    pub fn clause(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl Into<Literal>>,
    ) -> Tdd {
        let clause: Vec<Literal> = literals.into_iter().map(Into::into).collect();
        crate::build::clause_to_tdd(self, vtree, &clause)
    }

    /// The constant-true function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn one(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_one(self, vtree)
    }

    /// The constant-false function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn zero(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_zero(self, vtree)
    }

    /// The conjunction of `literals` over `vtree`: one width-1 node per
    /// internal vtree node, so the whole diagram is one path.
    ///
    /// A variable no literal mentions is free — the cube says nothing about
    /// it, so both of its values satisfy the result. Each item is converted
    /// with [`Into<Literal>`], so plain integers use the 1-based DIMACS sign
    /// convention.
    ///
    /// # Panics
    ///
    /// If `literals` names a variable twice, or names one `vtree` has no leaf
    /// for.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Engine;
    /// use tididi::vtree::Vtree;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = eng.cube(&vtree, [1, -2]); // x1 ∧ ¬x2, with x3 free
    /// assert_eq!(f.model_count(), 2u32.into());
    /// ```
    #[must_use]
    pub fn cube(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl Into<Literal>>,
    ) -> Tdd {
        crate::build::cube_to_tdd(self, vtree, literals)
    }
}

#[cfg(test)]
mod tests;
