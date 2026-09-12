//! Phase 2: turning each fusion plan's summed value into a marginal-side ref.

use rustc_hash::FxHashMap;

use crate::engine::Engine;
use crate::limits::OperationError;
use crate::diagram::{ValueRef, Tdd};
use crate::vtree::VtreeIdx;

use crate::value::slots::SlotValues;

use super::PlanEntry;

/// Phase 2: encode each plan's fused value as a marginal-side ref; fill
/// `plan.new_ref`.
///
/// Returns `true` if at least one plan emitted an inline ref (the parent
/// level's marginal-side inline marker must then be raised in Phase 3).
///
/// Slot identity is value-keyed: a plan whose value matches another plan in
/// the sweep — or, where the domain seeds the map, an existing slot — shares
/// that slot rather than minting one. Sound because pair lists are multisets:
/// each shared-slot pair occurrence carries one plan's contribution.
///
/// Never a pinned leaf: the caller resolves that boundary by lookup instead.
#[inline(always)]
pub(super) fn allocate_fusion_slots<D: SlotValues>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    plans: &mut [PlanEntry<D::Value>],
) -> Result<bool, OperationError> {
    let mut any_inline = false;
    let mut by_value: FxHashMap<D::Key, u32> = FxHashMap::default();
    D::seed(tdd, v, &mut by_value);
    for plan in plans.iter_mut() {
        if let Some(raw) = D::inline_ref(&plan.value) {
            plan.new_ref = raw;
            any_inline = true;
            continue;
        }
        let key = D::key(&plan.value);
        if let Some(&existing) = by_value.get(&key) {
            plan.new_ref = ValueRef::slot_raw(existing);
            continue;
        }
        let s = D::push_slot(eng, tdd, v, plan.value.clone())?;
        by_value.insert(key, s);
        plan.new_ref = ValueRef::slot_raw(s);
    }
    Ok(any_inline)
}
