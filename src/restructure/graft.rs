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
//! Joining canonical parts introduces no structural twins. Marginal roots
//! acquire parent references, which are tagged and pruned to retain canonical form.

mod error;
pub use error::GraftError;

use crate::Engine;
use std::sync::Arc;

use crate::vtree::{GraftLayout, VarId, Vtree, VtreeIdx};

use crate::diagram::{
    return_levels, take_levels, ChildPair, NodeIdx, PoolSlot, Tdd, TddLevel, TddNodeId,
    TddBuildError, WeightStore, ONE_LEAF_IDX,
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
    /// levels are moved, which is why `parts` is taken by value; a part's
    /// integer marginal levels move with it. Structural parts may carry weights,
    /// which this unweighted entry discards. Parts with computed weight columns
    /// require [`Tdd::graft_over`] and a compatible destination store. Runs on a
    /// the first part's context with no limits armed, or a fresh context when
    /// there are no parts. The result follows [`Vtree::graft`]'s context policy.
    ///
    /// # Errors
    ///
    /// [`GraftError::Vtree`] if variable sets overlap or there are no parts or
    /// spine variables; [`GraftError::VariableOutOfRange`] if a variable cannot
    /// fit in the id space. [`GraftError::PartWeights`] if a marginal part needs
    /// its weight store. Cleanup errors are reported as [`GraftError::Operation`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    /// use tididi::restructure::GraftError;
    /// let a = Arc::new(Vtree::balanced_over(&[VarId(0), VarId(1)]));
    /// let b = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(3)]));
    /// let f = Tdd::clause(&a, [1, 2])?;   // x1 ∨ x2: 3 models
    /// let g = Tdd::clause(&b, [3, -4])?;  // x3 ∨ ¬x4: 3 models
    /// let fg = Tdd::graft(vec![f, g], &[VarId(4)]).unwrap();
    /// assert_eq!(fg.model_count()?, 18u32.into()); // 3 · 3 · 2 (x5 is free)
    ///
    /// // A spine variable one of the parts already carries is refused.
    /// let h = Tdd::clause(&a, [1, 2])?;
    /// let k = Tdd::clause(&b, [3, -4])?;
    /// match Tdd::graft(vec![h, k], &[VarId(0)]) {
    ///     Ok(_) => unreachable!("var 0 is already in the first part"),
    ///     Err(e) => assert!(matches!(e, GraftError::Vtree(VtreeError::OverlappingVariable(VarId(0))))),
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn graft(parts: Vec<Tdd>, spine_vars: &[VarId]) -> Result<Tdd, GraftError> {
        for &variable in spine_vars {
            if variable.0 == u32::MAX {
                return Err(GraftError::VariableOutOfRange { variable, num_vars: u32::MAX });
            }
        }
        let num_vars = parts
            .iter()
            .map(|t| t.vtree.num_vars())
            .chain(spine_vars.iter().map(|v| v.0 + 1))
            .max()
            .unwrap_or(0);
        let context = parts.first().map(|part| Arc::clone(part.context())).unwrap_or_default();
        context.run(|eng| graft_impl(eng, parts, |_, v| v, spine_vars, num_vars, None))
            .map(|(tdd, _)| tdd)
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
    /// result retains their interpretation. The destination table must cover
    /// every renamed and free variable; a marginalized part must use the same
    /// arithmetic and literal weights under its rename. Structural parts can
    /// be reweighted. Existing destination columns are discarded.
    ///
    /// Pass `None` for an unweighted graft as in [`Tdd::graft`]. Stored integer
    /// counts cannot be reweighted, and stored weights cannot be discarded.
    /// `parts` is consumed; `eng` supplies level storage. Attaching a marginal
    /// root below a new parent runs a prune under the engine's limits; the
    /// structural assembly does not consult them.
    ///
    /// # Errors
    ///
    /// As [`Tdd::graft`], plus [`GraftError::MissingVariableMapping`] for an
    /// incomplete map and [`GraftError::DestinationWeights`] for missing
    /// destination weights. [`GraftError::PartWeights`] identifies a part whose
    /// stored values cannot use the destination. Inputs are checked before
    /// moving levels or taking the false-result shortcut.
    pub fn graft_over(
        eng: &Engine,
        parts: Vec<(Tdd, Vec<VarId>)>,
        free_vars: &[VarId],
        num_vars: u32,
        into: Option<WeightStore>,
    ) -> Result<(Tdd, GraftLayout), GraftError> {
        for (part, (tdd, map)) in parts.iter().enumerate() {
            for (_, variable) in tdd.vtree.leaf_bottomup() {
                if map.get(variable.idx()).is_none() {
                    return Err(GraftError::MissingVariableMapping { part, variable });
                }
            }
        }
        let (parts, maps): (Vec<Tdd>, Vec<Vec<VarId>>) = parts.into_iter().unzip();
        graft_impl(eng, parts, |k, local| maps[k][local.idx()], free_vars, num_vars, into)
    }
}


/// The one graft: [`Tdd::graft`] with the identity rename, [`graft_over`](Tdd::graft_over)
/// with the per-part maps.
fn graft_impl(
    eng: &Engine,
    mut parts: Vec<Tdd>,
    rename: impl Fn(usize, VarId) -> VarId,
    spine_vars: &[VarId],
    num_vars: u32,
    into: Option<WeightStore>,
) -> Result<(Tdd, GraftLayout), GraftError> {
    let n_parts = parts.len();
    let rename = &rename;
    for variable in parts.iter().enumerate()
        .flat_map(|(k, part)| part.vtree.leaf_bottomup().map(move |(_, local)| rename(k, local)))
        .chain(spine_vars.iter().copied())
    {
        if variable.0 >= num_vars {
            return Err(GraftError::VariableOutOfRange { variable, num_vars });
        }
    }
    let vtrees: Vec<&Vtree> = parts.iter().map(|t| &*t.vtree).collect();
    let (grafted_vtree, layout) = Vtree::graft_over(&vtrees, rename, spine_vars, num_vars)?;
    let grafted_arc: Arc<Vtree> = Arc::new(grafted_vtree);

    if let Some(destination) = &into {
        destination.check_variables(grafted_arc.leaf_bottomup().map(|(_, var)| var))
            .map_err(GraftError::DestinationWeights)?;
    }
    for (part, tdd) in parts.iter().enumerate() {
        check_part_weights(tdd, into.as_ref(), |local| rename(part, local))
            .map_err(|source| GraftError::PartWeights { part, source })?;
    }
    let repair_boundary = !layout.chain_internals.is_empty()
        && parts.iter().any(|part| part.levels[part.output.vtree.idx()].is_marginal());
    let into = into.map(|store| store.empty_like());

    // A ⊥ part makes the conjunction ⊥; the chain below would name the `ZERO`
    // sentinel as a child.
    if parts.iter().any(Tdd::is_zero) {
        let mut result = crate::build::constant_zero(eng, &grafted_arc);
        result.weights = into;
        return Ok((result, layout));
    }

    // Move each part's internal levels into their grafted positions, and its
    // weighted values with them: the store is keyed by level, so a level that
    // moves takes its column along or its parents' refs read another node's
    // weight.
    let mut levels = take_levels(eng, grafted_arc.num_nodes());
    let mut merged = into;
    for (k, tdd) in parts.iter_mut().enumerate() {
        let comp_to_full_k = &layout.comp_to_full[k];
        let mut part_ws = merged.as_ref().and_then(|_| tdd.detach_weights());
        // Indexes `comp_to_full_k` and the component vtree at the same position.
        #[allow(clippy::needless_range_loop)]
        for c_idx in 0..tdd.vtree.num_nodes() {
            let f_idx = comp_to_full_k[c_idx];
            if tdd.levels[c_idx].is_weight_marginal()
                && let (Some(merged), Some(part_ws)) = (merged.as_mut(), part_ws.as_mut())
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
        levels[chain_idx.idx()].push_internal_node(&[ChildPair::new(left, right)]);
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
        grafted.set_weights(merged).map_err(GraftError::DestinationWeights)?;
    }
    if repair_boundary {
        for (leaf, _) in grafted.vtree.leaf_bottomup() {
            if grafted.levels[leaf.idx()].is_weight_marginal() {
                crate::marginal::canonicalize_apply_leaf_refs(
                    &[leaf.idx()], &grafted.vtree, &mut grafted.levels, grafted.weights.as_ref());
            }
        }
        crate::diagram::tag_all_marginal_side_slots(&mut grafted, None);
        eng.reduce(&mut grafted, crate::reduce::ReductionPlan::Prune)?;
    }
    Ok((grafted, layout))
}

/// Require a destination interpretation for every marginal value that a part will carry across.
fn check_part_weights(part: &Tdd, destination: Option<&WeightStore>, rename: impl Fn(VarId) -> VarId) -> Result<(), TddBuildError> {
    for (index, level) in part.levels.iter().enumerate() {
        let level_index = VtreeIdx(index as u32);
        if destination.is_none() && level.is_weight_marginal() {
            return Err(TddBuildError::WeightedLevelWithoutStore { level: level_index });
        }
        if destination.is_some() && level.is_marginal() && !level.is_weight_marginal() {
            return Err(TddBuildError::CountLevelWithWeights { level: level_index });
        }
    }
    if part.has_marginal_level()
        && let (Some(source), Some(destination)) = (part.weights.as_ref(), destination)
        && !source.compatible_after_rename(destination, part.vtree.leaf_bottomup().map(|(_, var)| (var, rename(var))))
    {
        return Err(TddBuildError::IncompatibleWeights);
    }
    Ok(())
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
    /// # Safety
    ///
    /// Both operands must share the same vtree allocation and compatible weight
    /// configuration. The merge level `t` must be internal, with exactly one
    /// structural, single-pair node in each operand. The left operand must be
    /// independent of `t`'s right subtree and the right operand independent of
    /// its left subtree; their wrapper pairs must represent those free sides
    /// as true. Any levels above `t` retained from `self` must still reference
    /// the merged node correctly. The result must satisfy the storage and
    /// determinism invariants of [`TddBuilder`](crate::diagram::TddBuilder).
    /// Invalid references can cause out-of-bounds reads in later operations.
    ///
    /// # Panics
    ///
    /// If either diagram's level at `t` does not hold exactly one stored node.
    pub unsafe fn splice_subtree_unchecked(&mut self, eng: &Engine, mut other: Tdd, t: VtreeIdx) {
        assert_eq!(
            self.levels[t.idx()].slot_count(), 1,
            "splice_subtree: the left diagram has width {} at the merge point",
            self.levels[t.idx()].slot_count(),
        );
        assert_eq!(
            other.levels[t.idx()].slot_count(), 1,
            "splice_subtree: the right diagram has width {} at the merge point",
            other.levels[t.idx()].slot_count(),
        );
        let left_ptr = self.levels[t.idx()].nodes()[0].inline_pair().left;
        let right_ptr = other.levels[t.idx()].nodes()[0].inline_pair().right;

        let right_child = self.vtree.children(t).1;
        swap_subtree_levels(
            &mut self.levels,
            &mut other.levels,
            &self.vtree,
            right_child,
        );

        self.levels[t.idx()].clear();
        self.levels[t.idx()].push_internal_node(&[ChildPair::new(left_ptr, right_ptr)]);

        return_levels(eng, PoolSlot::First, std::mem::take(&mut other.levels));

        if let Some(rw) = other.detach_weights() {
            match self.detach_weights() {
                Some(mut lw) => {
                    lw.absorb(rw);
                    self.weights = Some(lw);
                }
                None => self.weights = Some(rw),
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
mod tests;
