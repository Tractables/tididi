//! Assemble a conjunction over disjoint variable sets on a grafted vtree.
//!
//! Move each part's levels to their grafted indices, then join part roots and
//! free-variable leaves with one pair per new level. Whole-level moves preserve
//! local node indices. Canonical parts introduce no structural twins; marginal
//! roots are tagged and pruned after acquiring their new parent references.

use super::{GraftError, placement::MovePlacement};

use crate::Engine;
use std::sync::Arc;

use crate::vtree::{GraftLayout, VarId, Vtree, VtreeIdx};

use crate::diagram::{NodeIdx, Tdd, TddBuildError, WeightStore, ONE_LEAF_IDX};

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
    /// require [`Tdd::graft_over`] and a compatible destination store. Runs on
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
    /// let a = Arc::new(Vtree::balanced_over(&[VarId(1), VarId(2)]).unwrap());
    /// let b = Arc::new(Vtree::balanced_over(&[VarId(3), VarId(4)]).unwrap());
    /// let f = Tdd::clause(&a, [1, 2])?;   // x1 ∨ x2: 3 models
    /// let g = Tdd::clause(&b, [3, -4])?;  // x3 ∨ ¬x4: 3 models
    /// let fg = Tdd::graft(vec![f, g], &[VarId(5)]).unwrap();
    /// assert_eq!(fg.model_count()?, 18u32.into()); // 3 · 3 · 2 (x5 is free)
    ///
    /// // A spine variable one of the parts already carries is refused.
    /// let h = Tdd::clause(&a, [1, 2])?;
    /// let k = Tdd::clause(&b, [3, -4])?;
    /// match Tdd::graft(vec![h, k], &[VarId(1)]) {
    ///     Ok(_) => unreachable!("variable 1 is already in the first part"),
    ///     Err(e) => assert!(matches!(e, GraftError::Vtree(VtreeError::OverlappingVariable(VarId(1))))),
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn graft(parts: Vec<Tdd>, spine_vars: &[VarId]) -> Result<Tdd, GraftError> {
        let num_vars = crate::vtree::graft::graft_id_space(parts.iter().map(|t| &*t.vtree), spine_vars);
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
        if variable.0 == 0 || variable.0 > num_vars {
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
    let into = into.map(|store| store.empty_like());

    // A ⊥ part makes the conjunction ⊥; the chain below would name the `ZERO`
    // sentinel as a child.
    if parts.iter().any(Tdd::is_zero) {
        let mut result = crate::build::constant_zero(eng, &grafted_arc);
        result.weights = into;
        return Ok((result, layout));
    }

    let mut placement = MovePlacement::new(eng, &grafted_arc, into)?;
    for (part, map) in parts.iter_mut().zip(&layout.comp_to_full) {
        placement.move_part(part, map);
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
        placement.join(chain_idx, left, right);
    }

    // The output is the last chain join when there is one; otherwise the sole
    // piece's root (a part's output, or a lone spine leaf: constant true).
    let output_local = if layout.chain_internals.is_empty() {
        piece_ref(0)
    } else {
        NodeIdx(0)
    };
    Ok((placement.finish(output_local)?, layout))
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

#[cfg(test)]
#[path = "tests/graft/mod.rs"]
mod tests;
