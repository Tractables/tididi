//! The rotation-locality check: after an accepted rotation, twin contraction
//! has nothing to do.

use crate::diagram::Tdd;
use crate::engine::Engine;
use crate::reduce::contract::contract_all_twins;
use crate::reduce::contract::contract_leaf::contract_leaf_twins;
use crate::vtree::VtreeIdx;

/// Run twin contraction and leaf-twin contraction after a rotation at `w_idx`
/// and panic if either changed a level; requires a diagram that was canonical
/// before the rotation.
///
/// Rotation locality: `restructure_inner_search` leaves every level
/// outside `{v_idx, w_idx}` bit-identical, the outer level at `v_idx` inherits
/// canonicity from its pre-rotation self by parent-context bijection, and each
/// node of the new inner level at `w_idx` is minted one per distinct
/// fingerprint, which is one per parent context — so the contraction finds no
/// twins anywhere. Leaf-twin contraction is rotation-invariant: a leaf's mode
/// is a property of the function, and every leaf was already in its mode.
///
/// A diagram with a marginal level is exempt: the bounded restructure keeps
/// the child multiset there without Boolean dedup, so the inner level
/// genuinely has twins, and contracting them would shrink levels outside
/// `{v_idx, w_idx}`, which the size predictor and the reject-path restore
/// assume untouched. Release builds rely on the preserved multiset, so the
/// check mirrors them and does nothing.
///
/// Drains the contract worklist as the contraction does; the caller clears
/// the worklists afterwards either way.
pub(crate) fn debug_assert_rotation_locality(eng: &Engine, tdd: &mut Tdd, w_idx: VtreeIdx) {
    if tdd.levels.iter().any(|l| l.is_marginal()) {
        return;
    }
    let widths: Vec<usize> = tdd.levels.iter().map(|l| l.width()).collect();
    contract_all_twins(eng, tdd).expect("rotation-locality check: an allocation was refused");
    for (i, level) in tdd.levels.iter().enumerate() {
        assert_eq!(
            level.width(),
            widths[i],
            "rotation locality: twin contraction after the rotation at {} merged nodes at level {i}",
            w_idx.0,
        );
    }
    let fired =
        contract_leaf_twins(eng, tdd).expect("rotation-locality check: an allocation was refused");
    assert!(
        !fired,
        "rotation locality: leaf-twin contraction fired after the rotation at {}",
        w_idx.0,
    );
}
