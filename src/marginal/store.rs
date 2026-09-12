//! Reading and writing the per-node marginal count / weight stores.

use crate::value::slots::{compact_slots, count_key_at, rekey_big, truncate_with_slack};
use crate::diagram::{BigSide, TddLevel};
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx};

/// Free the per-node store of `parent`'s marginal children; call it at the
/// moment `parent` becomes marginal.
///
/// A marginal parent carries no pair lists, so nothing reads its children's
/// stores again. The integer store is emptied and the weighted column cleared
/// through `ws`; either way the child's `width()` then reports 0 while it stays
/// marginal. A weight-marginal leaf's column is left alone (invariant 11).
pub(crate) fn free_subsumed_marginal_children(
    levels: &mut [TddLevel],
    vtree: &Vtree,
    parent: VtreeIdx,
    mut ws: Option<&mut WeightStore>,
) {
    if vtree.node(parent).is_leaf() {
        return;
    }
    let (l, r) = vtree.children(parent);
    for c in [l.idx(), r.idx()] {
        let lvl = &mut levels[c];
        if !lvl.is_marginal() {
            continue; // a structural child here is a leaf (`assert_can_make_marginal`)
        }
        // Integer-marginal child: empty the count store but keep `Some` so the
        // level stays marginal; drop any big-overflow side table.
        if let Some(v) = lvl.marginal_counts_mut() {
            if !v.is_empty() {
                *v = Vec::new();
            }
            lvl.clear_marginal_big();
        }
        // Weight-marginal child: zero the slot carrier and drop the column; the
        // flag stays set. A leaf's column is pinned (invariant 11, decided by
        // `test_helpers::check::marginal::check_leaf_columns_pinned`).
        if lvl.is_weight_marginal() && lvl.weight_width() != 0 && !vtree.node(VtreeIdx(c as u32)).is_leaf() {
            lvl.set_weight_width(0);
            if let Some(ws) = ws.as_deref_mut() {
                ws.set_level(c, Vec::new());
            }
        }
    }
}


/// Compact a freshly built marginal store so that each count value occupies at
/// most one slot (invariant 10), returning the deduped store and a slot remap,
/// `remap[old] = new`.
///
/// Duplicates merge onto the first slot holding their value, so the returned
/// column may be shorter than the input; with no duplicates both come back
/// unchanged. Refs are not remapped here: the caller must redirect every
/// parent-side ref into the old store through `remap` (the marginalize pass
/// does so with `remap_refs_into`). An emit-born store does not pass through
/// here; its invariant 10 is established by `prune_value_slots`.
///
/// The count column is compacted in place, so no second full-length column is
/// resident at the peak; the overflow table is rekeyed into a fresh
/// [`BigSide`], which costs only the surviving overflow entries.
pub(crate) fn dedup_fresh_store(
    mut counts: Vec<u128>,
    big: Option<BigSide>,
) -> (Vec<u128>, Option<BigSide>, Vec<u32>) {
    let n = counts.len();
    // Written for every `i`, so the remap is final as it is written and the
    // overflow table can be rekeyed in one drain once it is complete.
    let mut remap: Vec<u32> = vec![0; n];
    let (new_len, _) = compact_slots(
        &mut counts,
        0..n,
        |counts, i| count_key_at(counts, big.as_ref(), i),
        |counts, dst, src| counts[dst] = counts[src],
        &mut remap,
    );

    if new_len == n {
        // No duplicates: `remap` is the identity, and so would be the rekey.
        return (counts, big, remap);
    }

    let new_big = rekey_big(big, &remap);
    truncate_with_slack(&mut counts, new_len);
    (counts, new_big, remap)
}

