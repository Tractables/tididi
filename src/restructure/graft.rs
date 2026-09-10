//! Graft: the diagram of a conjunction of diagrams over pairwise-disjoint variable
//! sets, built structurally — no `apply_and` — on the vtree
//! [`Vtree::graft`] joins their vtrees under.
//!
//! Each part arrives on its own `Arc<Vtree>`. The grafted vtree hangs the
//! parts and the spine variables down one right-linear chain; the diagram mirrors
//! that chain by
//!
//!   1. moving each part's levels whole into their grafted positions (pair
//!      contents are `NodeIdx` into the child level, which survives
//!      whole-level relocation), and
//!   2. building one width-1 level per chain join whose single pair points at
//!      the running chain root on the left and the newly hung piece's root
//!      reference on the right (`output.local` for a part, `ONE_LEAF_IDX` for
//!      a spine variable).
//!
//! The result is canonical when the parts are: chain levels are width-1, so
//! there are no twins to contract.

use crate::engine::Engine;
use std::sync::Arc;

use crate::vtree::{GraftLayout, VarId, Vtree, VtreeError, VtreeIdx};

use crate::diagram::{
    return_levels, take_levels, InputPair, NodeIdx, PoolSlot, Tdd, TddLevel, TddNodeId,
    WeightStore, ONE_LEAF_IDX,
};

impl Tdd {
    /// The conjunction of `parts` — diagrams over pairwise-disjoint variable sets,
    /// each on its own vtree — as one diagram on
    /// [`Vtree::graft`]`(part vtrees, spine_vars)`.
    ///
    /// Built structurally in `O(total nodes)`: no apply runs, and the result
    /// is canonical when every part is. `spine_vars` are variables no part
    /// mentions; the result is unconstrained in them, so each doubles the
    /// model count (the count ranges over every variable the vtree carries).
    /// The parts' vtrees are copied into the grafted vtree and the parts'
    /// levels are moved, which is why `parts` is taken by value. The result is
    /// structural: a part's weight store is dropped, because the parts share
    /// one variable space here and no caller has said which semiring the
    /// conjunction is over. [`Tdd::graft_over`] is the weighted entry.
    ///
    /// # Errors
    ///
    /// [`VtreeError::OverlappingVariable`] if two parts (or a part and a
    /// spine variable) carry the same variable; [`VtreeError::Invalid`] on no
    /// part and no spine variable.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    /// let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
    /// let b = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(3)]));
    /// let f = Tdd::clause(&a, [1, 2]);   // x1 ∨ x2: 3 models
    /// let g = Tdd::clause(&b, [3, -4]);  // x3 ∨ ¬x4: 3 models
    /// let fg = Tdd::graft(vec![f, g], &[VarId(4)]).unwrap();
    /// assert_eq!(fg.model_count(), 18u32.into()); // 3 · 3 · 2 (x5 is free)
    ///
    /// // A spine variable one of the parts already carries is refused.
    /// let h = Tdd::clause(&a, [1, 2]);
    /// let k = Tdd::clause(&b, [3, -4]);
    /// match Tdd::graft(vec![h, k], &[VarId(0)]) {
    ///     Ok(_) => unreachable!("var 0 is already in the first part"),
    ///     Err(e) => assert!(matches!(e, VtreeError::OverlappingVariable(VarId(0)))),
    /// }
    /// ```
    pub fn graft(parts: Vec<Tdd>, spine_vars: &[VarId]) -> Result<Tdd, VtreeError> {
        let num_vars = parts
            .iter()
            .map(|t| t.vtree.num_vars())
            .chain(spine_vars.iter().map(|v| v.0 + 1))
            .max()
            .unwrap_or(0);
        graft_impl(&Engine::new(), parts, |_, v| v, spine_vars, num_vars, None).map(|(tdd, _)| tdd)
    }

    /// [`Tdd::graft`] for parts compiled in their own local variable spaces.
    ///
    /// Each `(tdd, local_to_global)` is renamed through its map on the way in,
    /// and the id space is `num_vars`, which must hold every renamed id. The
    /// [`GraftLayout`] comes back so a caller holding per-part side tables keyed
    /// by vtree index can relocate them.
    ///
    /// `into` is the merged diagram's weight store: each weighted part's
    /// per-level values move into it under the level's grafted index, so the
    /// result is a weighted diagram the ordinary readers and the reduction
    /// passes can take as they find it. It is a parameter rather than something
    /// derived from the parts because a part compiled in its own variable space
    /// carries a semiring in that space; only the caller knows the semiring the
    /// conjunction is over. Pass `None` for a structural graft.
    ///
    /// # Panics
    ///
    /// Panics if the renamed variable sets and `free_vars` are not pairwise
    /// disjoint, or if there is nothing to graft.
    pub fn graft_over(
        eng: &Engine,
        parts: Vec<(Tdd, Vec<VarId>)>,
        free_vars: &[VarId],
        num_vars: u32,
        into: Option<WeightStore>,
    ) -> (Tdd, GraftLayout) {
        let (parts, maps): (Vec<Tdd>, Vec<Vec<VarId>>) = parts.into_iter().unzip();
        graft_impl(eng, parts, |k, local| maps[k][local.idx()], free_vars, num_vars, into)
            .expect("component variable sets partition the formula's variables")
    }
}


/// The one graft: [`Tdd::graft`] with the identity rename, [`graft_over`]
/// with the per-part maps.
fn graft_impl(
    eng: &Engine,
    mut parts: Vec<Tdd>,
    rename: impl Fn(usize, VarId) -> VarId,
    spine_vars: &[VarId],
    num_vars: u32,
    into: Option<WeightStore>,
) -> Result<(Tdd, GraftLayout), VtreeError> {
    let n_parts = parts.len();
    let vtrees: Vec<&Vtree> = parts.iter().map(|t| &*t.vtree).collect();
    let (grafted_vtree, layout) = Vtree::graft_over(&vtrees, rename, spine_vars, num_vars)?;
    let grafted_arc: Arc<Vtree> = Arc::new(grafted_vtree);

    // Move each part's internal levels into their grafted positions, and its
    // weighted values with them: the store is keyed by level, so a level that
    // moves takes its column along or its parents' refs read another node's
    // weight.
    let mut levels = take_levels(eng, grafted_arc.num_nodes());
    let mut merged = into;
    for (k, tdd) in parts.iter_mut().enumerate() {
        let comp_to_full_k = &layout.comp_to_full[k];
        let mut part_ws = merged.as_ref().and_then(|_| tdd.take_weights());
        // Indexes `comp_to_full_k` and the component vtree at the same position.
        #[allow(clippy::needless_range_loop)]
        for c_idx in 0..tdd.vtree.num_nodes() {
            let f_idx = comp_to_full_k[c_idx];
            if let (Some(merged), Some(part_ws)) = (merged.as_mut(), part_ws.as_mut())
                && let Some(values) = part_ws.take_level(c_idx)
            {
                merged.set_level(f_idx.idx(), values);
            }
            if tdd.vtree.node(VtreeIdx(c_idx as u32)).is_leaf() {
                // A non-marginal leaf carries no state — the fresh `TddLevel::new()`
                // already at the grafted position is its correct representation, so
                // skip the move. A leaf-marginalized leaf, however, carries its
                // `is_marginal()` flag (with an empty store): the signal every reader
                // uses to decode the parent's inline leaf-count refs (bit-30). If the
                // graft drops it, the parent's relocated inline refs outlive the
                // child's marginal flag, and prune / model-count misread the inline
                // count as a node index, reading out of bounds. Relocate marginal
                // leaves so the invariant "parent inlined leaf ⟺ leaf level
                // is_marginal" survives.
                if tdd.levels[c_idx].is_marginal() {
                    levels[f_idx.idx()] =
                        std::mem::take(&mut tdd.levels[c_idx]);
                }
                continue;
            }
            levels[f_idx.idx()] = std::mem::take(&mut tdd.levels[c_idx]);
        }
    }

    // Each piece's "true" reference, as a NodeIdx into the piece's root
    // level: a part contributes its `output.local`; a spine variable
    // contributes `ONE_LEAF_IDX` (the implicit leaf-level constant-true label).
    let piece_ref = |i: usize| -> NodeIdx {
        if i < n_parts {
            parts[i].output.local
        } else {
            ONE_LEAF_IDX
        }
    };

    // Chain join j has left = running chain root (piece 0 when j == 0) and
    // right = piece j+1; a chain level is width-1, so its node sits at 0.
    for (j, &chain_idx) in layout.chain_internals.iter().enumerate() {
        let left = if j == 0 { piece_ref(0) } else { NodeIdx(0) };
        let right = piece_ref(j + 1);
        levels[chain_idx.idx()].push_internal_node(&[InputPair { left, right }]);
    }

    // The output is the last chain join when there is one; otherwise the sole
    // piece's root (a part's output, or a lone spine leaf: constant true).
    let output_local = if layout.chain_internals.is_empty() {
        piece_ref(0)
    } else {
        NodeIdx(0)
    };
    let output = TddNodeId {
        vtree: grafted_arc.root(),
        local: output_local,
    };

    let mut grafted = Tdd::from_levels_unchecked(grafted_arc, levels, output);
    if let Some(merged) = merged {
        grafted.set_weights(merged);
    }
    Ok((grafted, layout))
}

impl Tdd {
    /// Replace the subtree under `t`'s right child with `other`'s, and pair the
    /// two roots at `t`.
    ///
    /// The neighbour of [`graft`](Self::graft) for the one case that needs no
    /// new vtree: `self` and `other` are already decomposed along the same
    /// vtree, each carries one node at `t`, and the variables they actually
    /// constrain lie in opposite subtrees of `t` — `self` on the left, `other`
    /// on the right. Their conjunction is then the single pair naming both
    /// roots, and every level below moves across untouched, because a pair side
    /// names a node of its own child level and a whole-level move does not
    /// renumber those.
    ///
    /// `other`'s weight store is absorbed into `self`'s, and its levels go back
    /// to the engine's pool.
    ///
    /// # Panics
    ///
    /// If either diagram's level at `t` does not hold exactly one stored node.
    pub(crate) fn splice_subtree(&mut self, eng: &Engine, mut other: Tdd, t: VtreeIdx) {
        assert_eq!(
            self.levels[t.idx()].width(), 1,
            "splice_subtree: the left diagram has width {} at the merge point",
            self.levels[t.idx()].width(),
        );
        assert_eq!(
            other.levels[t.idx()].width(), 1,
            "splice_subtree: the right diagram has width {} at the merge point",
            other.levels[t.idx()].width(),
        );
        let left_ptr = self.levels[t.idx()].nodes()[0].inline_pair().left;
        let right_ptr = other.levels[t.idx()].nodes()[0].inline_pair().right;

        // A whole subtree of foreign levels arrives at once, so nothing the
        // diagram had cached about its widest level still bounds it.
        self.forget_stats();
        let right_child = self.vtree.children(t).1;
        swap_subtree_levels(
            &mut self.levels,
            &mut other.levels,
            &self.vtree,
            right_child,
        );

        self.levels[t.idx()].clear();
        self.levels[t.idx()].push_internal_node(&[InputPair { left: left_ptr, right: right_ptr }]);

        return_levels(eng, PoolSlot::First, std::mem::take(&mut other.levels));

        if let Some(rw) = other.take_weights() {
            match self.take_weights() {
                Some(mut lw) => {
                    lw.absorb(rw);
                    self.set_weights(lw);
                }
                None => self.set_weights(rw),
            }
        }
    }
}

/// Move every level of the subtree rooted at `subtree_root` from `src` into
/// `dst`, and `dst`'s into `src`.
fn swap_subtree_levels(
    dst: &mut [TddLevel],
    src: &mut [TddLevel],
    vtree: &Vtree,
    subtree_root: VtreeIdx,
) {
    let mut stack = vec![subtree_root];
    while let Some(idx) = stack.pop() {
        std::mem::swap(&mut dst[idx.idx()], &mut src[idx.idx()]);
        if !vtree.node(idx).is_leaf() {
            let (left, right) = vtree.children(idx);
            stack.push(left);
            stack.push(right);
        }
    }
}

#[cfg(test)]
#[path = "graft_tests.rs"]
mod tests;
