//! The clause spine: which vtree levels a clause touches, and their map layout.

use super::*;
use crate::apply::scoped_flags::ScopedFlags;

/// Conjoin a TDD with a single clause directly, without constructing the
/// clause's TDD.
///
/// Equivalent to `apply_and(acc, clause_to_tdd(vtree, clause))` followed by
/// pruning, but faster: avoids the intermediate TDD allocation and prune pass
/// by computing the clause's contribution on-the-fly during the conjunction.
///
/// ## Virtual node model
///
/// At each vtree level, the clause implicitly defines two virtual nodes:
/// - **`c_t`** (clause node): "at least one literal in this subtree satisfies
///   the clause"
/// - **`d_t`** (complement): "no literal in this subtree satisfies the clause"
///
/// Instead of materializing these as TDD nodes, we maintain one flat
/// interleaved map (`cd_map`) indexed by `[level_base[t] + acc_node_index]`:
///   - `cd_map[base + i][0]` = output index for `acc[i] ∧ c_t` (DEAD if zero)
///   - `cd_map[base + i][1]` = output index for `acc[i] ∧ d_t` (DEAD if zero)
///
/// The final output is the conjunction of the accumulator's output with `c_t` at
/// the root level.
///
/// ## MergeScope-only traversal
///
/// The set of vtree levels that interact with the clause is exactly the union
/// of root-to-leaf paths for the clause's variables — a level is "relevant"
/// iff its subtree contains a clause variable iff it is an ancestor (inclusive)
/// of some clause-variable leaf. This "spine" (the Steiner tree of the clause's
/// leaves) is discovered directly via leaf lookup + parent-pointer walk and
/// processed by a post-order DFS, so the per-clause cost is O(spine) — we never
/// sweep the full vtree/TDD. The pooled flag/offset arrays stay sized to
/// `num_nodes` for O(1) indexing, but only spine entries are written and reset.
/// Walk root-paths from every clause literal, marking visited nodes in `visited`.
///
/// The dedup terminates early as soon as a node is already marked, so the
/// ancestor-closed union of root-paths is computed in O(sum of path lengths)
/// with no revisits. `visited` must already be sized to cover all vtree node
/// indices and have the relevant range zeroed by the caller.
///
/// `newly_marked`, when supplied, collects exactly the nodes this call flipped
/// from false to true — i.e. the clause's own spine minus whatever `visited`
/// already carried. That is what lets a caller accumulate the UNION of several
/// clauses' spines across calls without a second walk or a full-vtree scan:
/// the batch builder folds clauses into one diagram and needs the set of levels
/// those folds can have touched (the downstream driver's batch-build step). The
/// clause-apply path itself passes `None` — it recovers the same set from its
/// own post-order spine list.
#[inline(always)]
pub fn mark_clause_levels(
    vtree: &crate::vtree::Vtree,
    clause: &[Literal],
    visited: &mut [bool],
    mut newly_marked: Option<&mut Vec<VtreeIdx>>,
) {
    for lit in clause {
        let mut cur = vtree.leaf_of(lit.var).expect("the vtree carries this variable");
        loop {
            if visited[cur.idx()] { break; }
            visited[cur.idx()] = true;
            if let Some(out) = newly_marked.as_deref_mut() { out.push(cur); }
            match vtree.node(cur).parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
}

/// Phase 1 of `conjoin_clause_into`: build the clause spine.
///
/// Marks every ancestor (inclusive) of each clause-variable leaf in `on_spine`,
/// then collects spine internal nodes in post-order (bottom-up) into
/// `spine_internal` via a DFS over the marked subtree.
///
/// Total work is O(spine), not O(clause-len × height): the walk stops as soon
/// as it meets an already-marked node.
pub(super) fn build_clause_spine(
    vtree: &crate::vtree::Vtree,
    clause: &[Literal],
    on_spine: &mut ScopedFlags<'_>,
    spine_internal: &mut Vec<VtreeIdx>,
    dfs_stack: &mut Vec<(VtreeIdx, bool)>,
) {
    on_spine.mark(|flags, marked| mark_clause_levels(vtree, clause, flags, Some(marked)));

    // Post-order DFS over the marked subtree (rooted at the vtree root, which is
    // always relevant — it is an ancestor of every leaf) collects the spine's
    // INTERNAL nodes bottom-up. The marked set is ancestor-closed, so it is a
    // connected subtree containing the root; descending only into marked
    // children keeps the DFS O(spine).
    spine_internal.clear();
    dfs_stack.clear();
    let root = vtree.root();
    if on_spine[root.idx()] && !vtree.node(root).is_leaf() {
        dfs_stack.push((root, false));
    }
    while let Some((t, processed)) = dfs_stack.pop() {
        if processed {
            spine_internal.push(t);
        } else {
            dfs_stack.push((t, true));
            let (l, r) = vtree.children(t);
            if on_spine[l.idx()] && !vtree.node(l).is_leaf() { dfs_stack.push((l, false)); }
            if on_spine[r.idx()] && !vtree.node(r).is_leaf() { dfs_stack.push((r, false)); }
        }
    }
    // `spine_internal` is now bottom-up (children precede parents).
}

/// Phase 2 of `conjoin_clause_into`: propagate `need_dt` top-down over the spine.
///
/// A level needs the complement conjunction (acc × `d_t`) iff its parent does, OR
/// both siblings are relevant (the both-relevant `c_t` spawns (`d_L,c_R`) and
/// (`c_L,d_R`), each consuming a `d_t` from one side). Iterating `spine_internal`
/// in reverse (top-down) and writing only marked children keeps `need_dt` clean
/// for irrelevant nodes (which are never read).
#[inline(always)]
pub(super) fn propagate_need_dt(
    vtree: &crate::vtree::Vtree,
    spine_internal: &[VtreeIdx],
    on_spine: &[bool],
    need_dt: &mut ScopedFlags<'_>,
) {
    for &t in spine_internal.iter().rev() {
        let (l, r) = vtree.children(t);
        let both = on_spine[l.idx()] && on_spine[r.idx()];
        let inherited = need_dt[t.idx()] || both;
        if inherited {
            if on_spine[l.idx()] { need_dt.set(l); }
            if on_spine[r.idx()] { need_dt.set(r); }
        }
    }
}

/// Lay out the `cd_map` blocks for the spine: one `LEAF_WIDTH` block per clause
/// leaf and one width-sized block per spine internal level, in that order.
/// Returns the total number of map entries.
///
/// # Panics
///
/// Panics if a spine internal level is marginal: its pair structure has been
/// replaced by per-node counts, so the clause cannot be conjoined into it. That
/// is a caller-side ordering error — every clause over a scope must be applied
/// before the scope is marginalized.
pub(super) fn plan_cd_map_bases(
    vtree: &Vtree,
    clause: &[Literal],
    spine_internal: &[VtreeIdx],
    levels: &[TddLevel],
    level_base: &mut [usize],
) -> usize {
    let mut total = 0usize;
    for lit in clause {
        let ti = vtree.leaf_of(lit.var).expect("the vtree carries this variable").idx();
        level_base[ti] = total;
        total += LEAF_WIDTH;
    }
    for &t in spine_internal {
        let ti = t.idx();
        if levels[ti].is_marginal() {
            panic!(
                "conjoin_clause_into: clause literal under marginal vtree subtree \
                 (vtree t={ti}, marginal-count width={}). The clause references a \
                 variable whose scope has already been marginalized in the accumulator \
                 — callers must marginalize a subtree only after every clause touching \
                 its variables has been applied.",
                levels[ti].width(),
            );
        }
        level_base[ti] = total;
        total += levels[ti].width();
    }
    total
}

/// Fill the `cd_map` blocks of the clause's own leaves.
///
/// A leaf level is marginal, so no node is created: under the leaf encoding
/// (One, Pos, Neg) the literal picks the `c_t` column and its complement the
/// `d_t` one, and the conjunction table gives both directly.
pub(super) fn fill_leaf_maps(
    vtree: &Vtree,
    clause: &[Literal],
    level_base: &[usize],
    need_dt: &[bool],
    cd_map: &mut [[u32; 2]],
) {
    for lit in clause {
        let t = vtree.leaf_of(lit.var).expect("the vtree carries this variable");
        let base = level_base[t.idx()];
        let compute_dt = need_dt[t.idx()];
        let (clause_idx, compl_idx) = if lit.positive {
            (POS_LEAF_IDX.0 as usize, NEG_LEAF_IDX.0 as usize)
        } else {
            (NEG_LEAF_IDX.0 as usize, POS_LEAF_IDX.0 as usize)
        };
        for i in 0..LEAF_WIDTH {
            let dt = if compute_dt { CONJOIN_GRID[i][compl_idx] } else { DEAD };
            cd_map[base + i] = [CONJOIN_GRID[i][clause_idx], dt];
        }
    }
}

