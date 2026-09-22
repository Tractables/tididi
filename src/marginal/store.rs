//! Reading and writing the per-node marginal count / weight stores.

use crate::value::slots::{compact_count_slots, truncate_with_slack};
use crate::diagram::{CountOverflow, TddLevel};
use crate::diagram::{WeightStore, WeightValue};
use crate::value::CountVec;
use crate::vtree::{Vtree, VtreeIdx};

/// Free the per-node store of `parent`'s marginal children; call it at the
/// moment `parent` becomes marginal.
///
/// A marginal parent carries no pair lists, so nothing reads its children's
/// stores again. The integer store is emptied and the weighted column cleared
/// through `ws`; either way the child's `slot_count()` then reports 0 while it stays
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
/// parent-side ref into the old store through `remap` (the marginalization pass
/// does so with `remap_refs_into`). An emit-born store does not pass through
/// here; its invariant 10 is established by `prune_value_slots`.
///
/// The count column is compacted in place, so no second full-length column is
/// resident at the peak; the overflow table is rekeyed into a fresh
/// [`CountOverflow`], which costs only the surviving overflow entries.
pub(crate) fn dedup_fresh_store(
    mut counts: Vec<u128>,
    mut big: Option<CountOverflow>,
) -> (Vec<u128>, Option<CountOverflow>, Vec<u32>) {
    let n = counts.len();
    // Written for every `i`, so the remap is final as it is written and the
    // overflow table can be rekeyed in one drain once it is complete.
    let mut remap: Vec<u32> = vec![0; n];
    let (new_len, _) = compact_count_slots(&mut counts, &mut big, 0..n, &mut remap);

    if new_len == n {
        // No duplicates: `remap` is the identity, and so would be the rekey.
        return (counts, big, remap);
    }

    truncate_with_slack(&mut counts, new_len);
    (counts, big, remap)
}


/// Commit a streamed integer column as level `left_idx`'s marginal store.
///
/// `CountVec` and `TddLevel` hold the same slot-keyed overflow table, so this
/// is a move. It does not dedup values: invariant 10 for an emit-born store is
/// established at the slot prune, after the tagger has inlined small counts,
/// because slots shared at birth would let twin merges produce duplicate pairs
/// that pair fusion then sums into new slots.
pub(crate) fn install_int_column(
    levels: &mut [TddLevel],
    left_idx: usize,
    col: CountVec,
) {
    let (fast, big) = col.into_parts();
    levels[left_idx].become_marginal(fast, big);
}

/// Commit a streamed weighted column as level `left_idx`'s marginal store:
/// the integer commit's mirror, except the payload goes to the shared
/// [`WeightStore`] and the level keeps only the slot count.
pub(crate) fn install_weight_column(
    levels: &mut [TddLevel],
    left_idx: usize,
    col: Vec<WeightValue>,
    ws: &mut WeightStore,
) {
    let slots = col.len() as u32;
    levels[left_idx].become_marginal_weighted(slots);
    ws.set_level(left_idx, col);
}
