//! Assemble a conjunction over disjoint variable sets on a grafted vtree.
//!
//! Move each part's levels to their grafted indices, then join part roots and
//! free-variable leaves with one pair per new level. Whole-level moves preserve
//! local node indices. Canonical parts introduce no structural twins.

use super::{ExpandError, placement::MovePlacement};

use crate::Engine;
use std::sync::Arc;

use crate::vtree::{VarId, Vtree};

use crate::diagram::{NodeIdx, Tdd, ONE_LEAF_IDX};

/// The conjunction of `parts`, structural unweighted diagrams over pairwise
/// disjoint variables, and of nothing over `spine_vars`, on [`Vtree::graft`]
/// of their vtrees, built from the parts' levels with one pair per new level.
pub(crate) fn graft(eng: &Engine, mut parts: Vec<Tdd>, spine_vars: &[VarId]) -> Result<Tdd, ExpandError> {
    let _op = eng.limits().enter()?;
    let n_parts = parts.len();
    let num_vars = crate::vtree::graft::graft_id_space(parts.iter().map(|t| &*t.vtree), spine_vars);
    let vtrees: Vec<&Vtree> = parts.iter().map(|t| &*t.vtree).collect();
    let (grafted_vtree, layout) = Vtree::graft_over(&vtrees, |_, v| v, spine_vars, num_vars)?;
    let grafted_arc: Arc<Vtree> = Arc::new(grafted_vtree);

    // A ⊥ part makes the conjunction ⊥; the chain below would name the `ZERO`
    // sentinel as a child.
    if parts.iter().any(Tdd::is_zero) {
        return Ok(crate::build::constant_zero_padded(eng, &grafted_arc));
    }

    let canonical = parts.iter().all(Tdd::is_canonical);
    let mut placement = MovePlacement::new(eng, &grafted_arc)?;
    for (k, part) in parts.iter_mut().enumerate() {
        placement.move_part(part, layout.part(k).as_slice());
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
        placement.join(chain_idx, left, right)?;
    }

    // The output is the last chain join when there is one; otherwise the sole
    // piece's root (a part's output, or a lone spine leaf: constant true).
    let output_local = if layout.chain_internals.is_empty() {
        piece_ref(0)
    } else {
        NodeIdx(0)
    };
    let mut result = placement.finish(output_local)?;
    if canonical {
        result.certify_built();
    }
    Ok(result)
}
