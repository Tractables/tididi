//! Reading and writing the per-node marginal count / weight stores.

use crate::value::{CountRead, CountVec, COUNT_OVERFLOW};
use crate::limits::ReservePolicy;
use crate::value::slots::{compact_slots, count_key_at, rekey_big, truncate_with_slack};
use crate::diagram::WeightVal;
use crate::diagram::{BigSide, LeafLabel, MarginalSide, TddLevel, ValueRef};
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use super::column::LevelColumns;
use crate::diagram::leaf_count;

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


/// Resolve one child ref to a count read, against a level slice and the
/// per-batch `computed` scratch; a `Big` read borrows the `BigUint`.
///
/// At a marginal level the ref decodes by its own bits: bit 30 set is an
/// inline count (at most 2^30−1, so never an overflow sentinel); bit 30 clear
/// is a bare slot index into the store (a pre-tag mid-batch ref is a bare node
/// index, and that is its slot index). The decode keys off the ref rather than
/// the level's `marginal_inlined_*` flag because a parent level rebuilt from
/// scratch can lose the flag while its pairs still carry inline refs.
#[inline]
pub(crate) fn read_count<'a, R: ReservePolicy>(
    level_idx: usize,
    node_idx: usize,
    vtree: &Vtree,
    levels: &'a [TddLevel],
    computed: &'a [Option<CountVec<R>>],
) -> CountRead<'a> {
    if let Some(ic) = levels[level_idx].marginal_counts() {
        let raw = node_idx as u32;
        if MarginalSide(raw).is_zero_sentinel() {
            return CountRead::Fast(0); // ZERO sentinel — never decode (mirrors emit_or_tag)
        }
        return match ValueRef::from_raw(MarginalSide(raw)) {
            ValueRef::Inline(v) => CountRead::Fast(v as u128),
            // A marginal leaf keeps an empty store under the inline path (all
            // counts live inline at the parent), so a bare slot ref here is a
            // leaf-label index with a fixed count — decode it directly rather
            // than indexing the (empty) store. Reached by paths that leave a
            // leaf-side ref bare (e.g. projection) instead of inlining it.
            ValueRef::Slot(s) if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() => {
                CountRead::Fast(leaf_count(LeafLabel::from_idx(s as usize)))
            }
            ValueRef::Slot(s) => {
                let v = ic[s as usize];
                if v != COUNT_OVERFLOW {
                    return CountRead::Fast(v);
                }
                if let Some(bv) = levels[level_idx]
                    .marginal_counts_big()
                    .and_then(|ib| ib.get(s as usize))
                {
                    return CountRead::Big(bv);
                }
                unreachable!(
                    "big count not available for level {} node {}",
                    level_idx, node_idx
                );
            }
        };
    }
    // Check pre-computed buffer (non-marginal level: plain index).
    if let Some(counts) = &computed[level_idx] {
        return counts.get(node_idx);
    }
    // Leaf level: fixed counts.
    if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() {
        return CountRead::Fast(leaf_count(LeafLabel::from_idx(node_idx)));
    }
    unreachable!("counts not available for level {}", level_idx);
}


/// Weighted analogue of [`read_count`]: resolve a child node's semiring value
/// for the weighted marginalization cascade. Reads, in order:
///   0. a leaf level by label, never through the store column: a leaf-side ref
///      is a bare `LeafLabel` index whether the leaf is structural or
///      weight-marginal (the pinned column is installed in label order by
///      [`marginalize_leaf_weighted`](super::leaf::marginalize_leaf_weighted)),
///      and the shared store may hold a column at this index that belongs to
///      another diagram;
///   1. the [`WeightStore`] column of a level this diagram has already
///      weight-marginalized; marginal-side refs are bare slots in weighted mode;
///   2. the per-batch `computed_weights` buffer for a level computed earlier in
///      this batch but not yet stored.
///
/// Store-slot and per-batch reads borrow; only the `ZERO` sentinel and leaf
/// bases materialize an owned value.
pub(crate) fn read_weight<'a>(
    level_idx: usize,
    node_idx: usize,
    vtree: &Vtree,
    cols: &LevelColumns<'a>,
    computed_weights: &'a [Option<Vec<WeightVal>>],
) -> std::borrow::Cow<'a, WeightVal> {
    let ws = cols.store();
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) {
        let raw = node_idx as u32;
        if MarginalSide(raw).is_zero_sentinel() {
            // `ZERO` sentinel — mirrors `read_count`. Leaf levels only ever
            // carry Pos/Neg/One, but the bit is tested before every decode.
            return std::borrow::Cow::Owned(ws.wzero());
        }
        let label_idx = match ValueRef::from_raw(MarginalSide(raw)) {
            ValueRef::Inline(_) => unreachable!("weighted marginal-side refs are bare slots"),
            ValueRef::Slot(s) => s as usize,
        };
        let v = ws.leaf_val(var, LeafLabel::from_idx(label_idx));
        debug_assert!(
            leaf_column_slot_agrees(ws, level_idx, label_idx, &v),
            "weight-marginal leaf {level_idx}: pinned column disagrees with leaf_val \
             at label slot {label_idx}"
        );
        return std::borrow::Cow::Owned(v);
    }
    // Internal level: `cols.get` is gated on this diagram's own marginality,
    // since the shared store can hold a column at this index installed by
    // another live diagram while this one's level is still structural, and a
    // structural level's node indices are not slots of that column.
    if let Some(values) = cols.get(level_idx) {
            let raw = node_idx as u32;
            if MarginalSide(raw).is_zero_sentinel() {
                // `ZERO` sentinel — mirrors `read_count`
                return std::borrow::Cow::Owned(ws.wzero());
            }
            let slot = match ValueRef::from_raw(MarginalSide(raw)) {
                ValueRef::Inline(_) => unreachable!("weighted marginal-side refs are bare slots"),
                ValueRef::Slot(s) => s as usize,
            };
            return std::borrow::Cow::Borrowed(&values[slot]);
        }
    if let Some(w) = &computed_weights[level_idx] {
        return std::borrow::Cow::Borrowed(&w[node_idx]);
    }
    unreachable!("weighted value not available for level {}", level_idx);
}

/// Debug-only companion to the leaf branch of [`read_weight`]: true unless a
/// pinned leaf column is installed at `level_idx` and its slot `label_idx`
/// differs from `expect`, or the column has other than three slots.
#[cfg(debug_assertions)]
fn leaf_column_slot_agrees(
    ws: &WeightStore,
    level_idx: usize,
    label_idx: usize,
    expect: &WeightVal,
) -> bool {
    use crate::diagram::semiring::weight_key;
    let Some(col) = ws.level(level_idx) else { return true };
    if col.len() != crate::diagram::LEAF_WIDTH {
        return false; // some pass compacted, erased or appended to the column
    }
    weight_key(&col[label_idx]) == weight_key(expect)
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn leaf_column_slot_agrees(
    _ws: &WeightStore,
    _level_idx: usize,
    _label_idx: usize,
    _expect: &WeightVal,
) -> bool {
    true
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

