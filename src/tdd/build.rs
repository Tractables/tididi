//! TDD construction: building TDDs from clauses and constants.
//!
//! `clause_to_tdd` builds a minimal, canonical TDD for a single clause directly
//! (without a raw build + minimize round-trip). `constant_one` and `constant_zero`
//! create the trivial TDDs for the constant-true and constant-false functions.

use std::cell::Cell;
use std::sync::Arc;

use crate::vtree::{Literal, Vtree, VtreeIdx};

use super::types::{self, *};
use super::utils::{pool_put, pool_take};

// Thread-local scratch buffers for clause_to_tdd (reused across calls).
// See types.rs for explanation of the Cell::take()/Cell::set() pooling pattern.
thread_local! {
    /// Per-level index of the clause-satisfied node (c_t), or u32::MAX if unset.
    static SCRATCH_CLAUSE_IDX: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    /// Per-level index of the complement node (d_t), or u32::MAX if unset.
    static SCRATCH_COMPLEMENT_IDX: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    /// Per-level flag: true if the clause node (c_t) is absent (subtree irrelevant).
    static SCRATCH_IRRELEVANT: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Post-order (children-before-parents) list of the vtree's internal nodes,
    /// rebuilt per call. Pooled: on a vtree with hundreds of thousands of
    /// levels the fresh `Vec` this replaced re-grew from zero — a full doubling
    /// ladder of allocations and copies — on every single clause build.
    static SCRATCH_INTERNAL_POSTORDER: Cell<Vec<(VtreeIdx, VtreeIdx, VtreeIdx)>> =
        const { Cell::new(Vec::new()) };
    /// DFS stack for the post-order walk above; pooled for the same reason.
    static SCRATCH_POSTORDER_STACK: Cell<Vec<VtreeIdx>> = const { Cell::new(Vec::new()) };
}

/// Build a TDD computing the constant-false function (no assignment satisfies it).
/// Output points to the ZERO sentinel (`u32::MAX`) — no actual nodes are created.
pub fn constant_zero(vtree: &Arc<Vtree>) -> Tdd {
    let levels = types::take_levels(vtree.num_nodes());
    Tdd::with_levels(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: ZERO },
    )
}

/// Build a TDD computing the constant-true function (all assignments satisfy it).
/// Width 1 at every internal vtree level (only ONE node at index 0).
/// Leaf levels are marginal (no stored nodes); One is at index 0 (`ONE_LEAF_IDX`).
pub fn constant_one(vtree: &Arc<Vtree>) -> Tdd {
    let mut levels = types::take_levels(vtree.num_nodes());

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
pub fn clause_to_tdd(vtree: &Arc<Vtree>, clause: &[Literal]) -> Tdd {
    let num_nodes = vtree.num_nodes();
    let mut levels = types::take_levels(num_nodes);

    // Per-level tracking for the bottom-up construction:
    //   clause_idx[t]     — local index of c_t (clause-satisfied node), u32::MAX = unset
    //   complement_idx[t] — local index of d_t (complement node) or identity node
    //                       at irrelevant levels (u32::MAX = unset)
    //   irrelevant[t]     — true if c_t is absent (no clause vars in this subtree)
    //
    // These are pooled scratch buffers: resize() extends capacity if needed (the
    // buffer may be smaller from a previous call with fewer vars), then fill()
    // resets leftover values from the previous call.
    let mut clause_idx = pool_take(&SCRATCH_CLAUSE_IDX);
    if clause_idx.len() < num_nodes { clause_idx.resize(num_nodes, u32::MAX); }
    clause_idx[..num_nodes].fill(u32::MAX);

    let mut complement_idx = pool_take(&SCRATCH_COMPLEMENT_IDX);
    if complement_idx.len() < num_nodes { complement_idx.resize(num_nodes, u32::MAX); }
    complement_idx[..num_nodes].fill(u32::MAX);

    let mut irrelevant = pool_take(&SCRATCH_IRRELEVANT);
    if irrelevant.len() < num_nodes { irrelevant.resize(num_nodes, false); }
    irrelevant[..num_nodes].fill(false);

    // Leaf levels are marginal (no stored nodes). At each leaf vtree level,
    // only set clause_idx/complement_idx and irrelevant flags using the
    // canonical implicit indices: One=0, Pos=1, Neg=2.

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

    // Leaf levels: no nodes created. Set implicit indices for c_t/d_t using
    // the new ordering (One=0, Pos=1, Neg=2).
    //
    // Two passes, not a search per leaf. Every leaf gets the irrelevant/One seed
    // first; then the clause's own literals — the only leaves that differ —
    // overwrite theirs, addressed through `var_to_leaf` in O(1). The version
    // this replaced ran `clause.iter().find(|l| l.var == var)` once per leaf,
    // i.e. O(#leaves × clause length) per clause; on a vtree whose leaf count
    // runs far ahead of any one clause's support that search dominated the
    // build. Same leaf set as before (`leaf_bottomup`), so no assumption about
    // where leaves sit in the index space is introduced.
    for (t, _var) in vtree.leaf_bottomup() {
        let t_idx = t.idx();
        complement_idx[t_idx] = ONE_LEAF_IDX.0;
        irrelevant[t_idx] = true;
    }
    for lit in clause {
        let t_idx = vtree.leaf_of(lit.var).idx();
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

    // Compute a correct post-order (children before parents) traversal via DFS.
    // This is robust against stale topo ordering after vtree rotation.
    let mut internal_postorder = pool_take(&SCRATCH_INTERNAL_POSTORDER);
    let mut postorder_stack = pool_take(&SCRATCH_POSTORDER_STACK);
    {
        internal_postorder.clear();
        postorder_stack.clear();
        postorder_stack.push(vtree.root());
        while let Some(idx) = postorder_stack.pop() {
            if let crate::vtree::VtreeNode::Internal { left, right, .. } = *vtree.node(idx) {
                internal_postorder.push((idx, left, right));
                postorder_stack.push(right);
                postorder_stack.push(left);
            }
        }
        internal_postorder.reverse();
    }

    // Propagate irrelevant flags bottom-up and find the LCA (highest node
    // in the tree where both children have clause variables). Using postorder
    // position (not vtree index) since rotation can make child indices > parent.
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

    // ── Bottom-up construction ────────────────────────────────────────────
    for (pos, &(t, left, right)) in internal_postorder.iter().enumerate() {
        let t_idx = t.idx();
        let li = left.idx();
        let ri = right.idx();
        let left_c = clause_satisfied_idx(li, &irrelevant, &clause_idx, &complement_idx);
        let right_c = clause_satisfied_idx(ri, &irrelevant, &clause_idx, &complement_idx);

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

    // If the root's entire subtree is irrelevant (no clause variables at all),
    // c_t was never created — return ZERO. In practice this path is unreachable:
    // clause_scope() panics on empty clauses, and all clause variables must
    // exist in the vtree. Kept as a defensive fallback.
    let root_idx = vtree.root().idx();
    let out_local = if irrelevant[root_idx] {
        ZERO
    } else {
        clause_satisfied_idx(root_idx, &irrelevant, &clause_idx, &complement_idx)
    };
    // Return scratch buffers.
    pool_put(&SCRATCH_INTERNAL_POSTORDER, internal_postorder);
    pool_put(&SCRATCH_POSTORDER_STACK, postorder_stack);
    pool_put(&SCRATCH_CLAUSE_IDX, clause_idx);
    pool_put(&SCRATCH_COMPLEMENT_IDX, complement_idx);
    pool_put(&SCRATCH_IRRELEVANT, irrelevant);

    Tdd::with_levels(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: out_local },
    )
}

impl Tdd {
    /// Build a canonical TDD for a single clause from DIMACS-style literals.
    ///
    /// Ergonomic sugar over [`clause_to_tdd`]: each item is converted with
    /// [`Into<Literal>`], so plain integers use the 1-based DIMACS sign
    /// convention (`1` → `x1`, `-2` → `¬x2`; see
    /// [`Literal`](crate::vtree::Literal)). Delegates to `clause_to_tdd` — the
    /// free function remains the primary API.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::tdd::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, -2]); // x1 ∨ ¬x2
    /// # let _ = f;
    /// ```
    pub fn clause(vtree: &Arc<Vtree>, lits: impl IntoIterator<Item = impl Into<Literal>>) -> Tdd {
        let clause: Vec<Literal> = lits.into_iter().map(Into::into).collect();
        clause_to_tdd(vtree, &clause)
    }

    /// Exact unweighted model count of this TDD, as an arbitrary-precision integer.
    ///
    /// Inherent-method sugar over the free function
    /// [`query::model_count`](crate::tdd::query::model_count), which remains the
    /// primary API.
    pub fn model_count(&self) -> num_bigint::BigUint {
        crate::tdd::query::model_count(self)
    }
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;
