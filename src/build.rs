//! TDD construction: building TDDs from clauses and constants.
//!
//! `Tdd::clause` builds a minimal, canonical TDD for a single clause directly
//! (without a raw build + minimize round-trip). `Tdd::one` and `Tdd::zero`
//! create the trivial TDDs for the constant-true and constant-false functions.

use std::cell::Cell;
use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{Vtree, VtreeIdx};
use crate::engine::Engine;

use crate::diagram::{self, *};
use super::utils::{pool_put, pool_take};

/// Every buffer one engine's clause builds reuse between calls.
///
/// See `utils::pool_take` for the `Cell` checkout pattern.
#[derive(Default)]
pub(crate) struct BuildScratch {
    /// Per-level index of the clause-satisfied node (c_t), or u32::MAX if unset.
    clause_idx: Cell<Vec<u32>>,
    /// Per-level index of the complement node (d_t), or u32::MAX if unset.
    complement_idx: Cell<Vec<u32>>,
    /// Per-level flag: true if the clause node (c_t) is absent (subtree irrelevant).
    irrelevant: Cell<Vec<bool>>,
    /// Post-order (children-before-parents) list of the vtree's internal nodes,
    /// rebuilt per call. Pooled: on a vtree with hundreds of thousands of
    /// levels the fresh `Vec` this replaced re-grew from zero — a full doubling
    /// ladder of allocations and copies — on every single clause build.
    internal_postorder: Cell<Vec<(VtreeIdx, VtreeIdx, VtreeIdx)>>,
    /// Work stack for the post-order walk above.
    postorder_stack: Cell<Vec<VtreeIdx>>,
}

impl BuildScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.clause_idx.take();
        self.complement_idx.take();
        self.irrelevant.take();
        self.internal_postorder.take();
        self.postorder_stack.take();
    }
}

/// Build a TDD computing the constant-false function (no assignment satisfies it).
/// Output points to the ZERO sentinel (`u32::MAX`) — no actual nodes are created.
pub(crate) fn constant_zero(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let levels = diagram::take_levels(eng, vtree.num_nodes());
    Tdd::with_levels(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: ZERO },
    )
}

/// Build a TDD computing the constant-true function (all assignments satisfy it).
/// Width 1 at every internal vtree level (only ONE node at index 0).
/// Leaf levels are marginal (no stored nodes); One is at index 0 (`ONE_LEAF_IDX`).
pub(crate) fn constant_one(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let mut levels = diagram::take_levels(eng, vtree.num_nodes());

    // Internal levels: each has one node pairing the child's "true" node.
    // Leaf children reference One (ONE_LEAF_IDX); internal children reference
    // their single node at index 0.
    for (t, left, right) in vtree.internal_bottomup() {
        let left_child_idx = if vtree.node(left).is_leaf() {
            ONE_LEAF_IDX
        } else {
            LocalNodeIdx(0)
        };
        let right_child_idx = if vtree.node(right).is_leaf() {
            ONE_LEAF_IDX
        } else {
            LocalNodeIdx(0)
        };
        let pair = InputPair { left: left_child_idx, right: right_child_idx };
        levels[t.idx()].push_internal_node(&[pair]);
    }

    // Output: One (for single-variable vtrees) or the sole internal node (index 0).
    let out_local = if vtree.node(vtree.root()).is_leaf() {
        ONE_LEAF_IDX
    } else {
        LocalNodeIdx(0)
    };
    Tdd::with_levels(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: out_local },
    )
}

/// Build a minimal, canonical TDD representing a single clause.
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
/// The result satisfies all TDD invariants: no false nodes, no unreachable
/// nodes, canonical (no duplicates, no redundant pairs).
pub(crate) fn clause_to_tdd(eng: &Engine, vtree: &Arc<Vtree>, clause: &[Literal]) -> Tdd {
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
    // c_t was never created — return ZERO. In practice this path is unreachable:
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

    Tdd::with_levels(
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
///   `clause_idx[t]`     — local index of c_t (clause-satisfied node), `u32::MAX` = unset
///   `complement_idx[t]` — local index of d_t (complement node) or identity node
///                         at irrelevant levels (`u32::MAX` = unset)
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
        let mut clause_idx = pool_take(&pool.clause_idx);
        if clause_idx.len() < num_nodes { clause_idx.resize(num_nodes, u32::MAX); }
        clause_idx[..num_nodes].fill(u32::MAX);

        let mut complement_idx = pool_take(&pool.complement_idx);
        if complement_idx.len() < num_nodes { complement_idx.resize(num_nodes, u32::MAX); }
        complement_idx[..num_nodes].fill(u32::MAX);

        let mut irrelevant = pool_take(&pool.irrelevant);
        if irrelevant.len() < num_nodes { irrelevant.resize(num_nodes, false); }
        irrelevant[..num_nodes].fill(false);

        ClauseScratch {
            pool,
            clause_idx,
            complement_idx,
            irrelevant,
            internal_postorder: pool_take(&pool.internal_postorder),
            postorder_stack: pool_take(&pool.postorder_stack),
        }
    }
}

impl Drop for ClauseScratch<'_> {
    fn drop(&mut self) {
        pool_put(&self.pool.internal_postorder, std::mem::take(&mut self.internal_postorder));
        pool_put(&self.pool.postorder_stack, std::mem::take(&mut self.postorder_stack));
        pool_put(&self.pool.clause_idx, std::mem::take(&mut self.clause_idx));
        pool_put(&self.pool.complement_idx, std::mem::take(&mut self.complement_idx));
        pool_put(&self.pool.irrelevant, std::mem::take(&mut self.irrelevant));
    }
}

/// Get the clause-satisfied node (`c_t`) index for a child level.
///
/// At relevant levels, `c_t` is stored directly in `clause_idx`. At
/// irrelevant levels (no clause vars in subtree), the only node is the
/// One identity, stored in `complement_idx`.
#[inline]
fn clause_satisfied_idx(t: usize, irrelevant: &[bool], clause_idx: &[u32], complement_idx: &[u32]) -> LocalNodeIdx {
    if irrelevant[t] {
        LocalNodeIdx(complement_idx[t]) // One (identity) at this irrelevant level
    } else {
        LocalNodeIdx(clause_idx[t])
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
        // `u32::MAX` at every leaf (the seed above writes only the other two
        // arrays), so "untouched" is exactly "not yet claimed".
        if clause_idx[t_idx] != u32::MAX {
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
        let li = left.idx();
        let ri = right.idx();
        if irrelevant[li] && irrelevant[ri] {
            irrelevant[t.idx()] = true;
        } else if !irrelevant[li] && !irrelevant[ri] {
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
        let li = left.idx();
        let ri = right.idx();
        let left_c = clause_satisfied_idx(li, irrelevant, clause_idx, complement_idx);
        let right_c = clause_satisfied_idx(ri, irrelevant, clause_idx, complement_idx);

        let level = &mut levels[t_idx];

        if irrelevant[li] && irrelevant[ri] {
            // Both subtrees irrelevant: only one node (identity).
            let left_d = LocalNodeIdx(complement_idx[li]);
            let right_d = LocalNodeIdx(complement_idx[ri]);
            let pair = InputPair { left: left_d, right: right_d };
            let one = level.push_internal_node(&[pair]);
            complement_idx[t_idx] = one.0;
        } else {
            // At least one subtree has clause variables.
            // complement_idx is always valid for irrelevant children (identity
            // node) and for relevant children below the LCA (d_t node).
            let left_d = LocalNodeIdx(complement_idx[li]);
            let right_d = LocalNodeIdx(complement_idx[ri]);

            let c_pairs: Vec<InputPair> = match (irrelevant[li], irrelevant[ri]) {
                (true, false) => vec![InputPair { left: left_d, right: right_c }],
                (false, true) => vec![InputPair { left: left_c, right: right_d }],
                _ => vec![
                    InputPair { left: left_c, right: right_c },
                    InputPair { left: left_c, right: right_d },
                    InputPair { left: left_d, right: right_c },
                ],
            };
            // No sort: pair lists are unordered sets and twin contraction is
            // order-independent, so the clause node's pair order is never
            // consumed (verified: packed benchmark TDD-size tests stay canonical
            // without this sort). See the NOTE in tdd/types.rs.
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

impl Tdd {
    /// Build a canonical TDD for a single clause from DIMACS-style literals.
    ///
    /// Ergonomic sugar over [`Tdd::clause`]: each item is converted with
    /// [`Into<Literal>`], so plain integers use the 1-based DIMACS sign
    /// convention (`1` → `x1`, `-2` → `¬x2`; see
    /// [`Literal`](crate::diagram::Literal)). Delegates to `Tdd::clause` — the
    /// free function remains the primary API.
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
    pub fn clause(vtree: &Arc<Vtree>, lits: impl IntoIterator<Item = impl Into<Literal>>) -> Tdd {
        Engine::new().clause(vtree, lits)
    }

    /// The constant-true function over `vtree`: every assignment satisfies it.
    ///
    /// Width 1 at every internal vtree level; the leaf levels are marginal.
    pub fn one(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().one(vtree)
    }

    /// The constant-false function over `vtree`: no assignment satisfies it.
    ///
    /// The output points at the ZERO sentinel, so no nodes are created.
    pub fn zero(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().zero(vtree)
    }

    /// Exact unweighted model count of this TDD, as an arbitrary-precision integer.
    ///
    /// Inherent-method sugar over the free function
    /// [`query::model_count`](crate::query::model_count), which remains the
    /// primary API.
    pub fn model_count(&self) -> num_bigint::BigUint {
        crate::query::model_count(self)
    }
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;

