//! Borrowed access to stored and transient value columns.

use crate::diagram::{EncodedChildRef, ChildDecoder, LeafLabel, TddLevel, ValueRef, WeightStore, WeightValue, leaf_count};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use super::{CountRead, CountVec};

/// The weighted columns one diagram may read from a shared store.
pub(crate) struct LevelColumns<'a> {
    store: &'a WeightStore,
    owner: &'a [TddLevel],
}

impl<'a> LevelColumns<'a> {
    /// Pair `store` with the level slice of the diagram that reads it.
    pub(crate) fn new(store: &'a WeightStore, owner: &'a [TddLevel]) -> Self {
        LevelColumns { store, owner }
    }

    /// The store itself, for the reads that are not level columns — the
    /// semiring zero and the leaf bases.
    pub(crate) fn store(&self) -> &'a WeightStore {
        self.store
    }

    /// Level `t`'s column, or `None` when this diagram's level is structural
    /// and its node indices are therefore not slots of any column.
    pub(crate) fn get(&self, t: usize) -> Option<&'a [WeightValue]> {
        column_of(self.store, &self.owner[t], t)
    }
}

/// [`LevelColumns::get`] for a caller that holds the one level rather than the
/// whole slice — the apply, whose output levels are split apart for the level
/// it is building.
pub(crate) fn column_of<'a>(
    store: &'a WeightStore,
    level: &TddLevel,
    t: usize,
) -> Option<&'a [WeightValue]> {
    level.is_weight_marginal().then(|| store.level(t)).flatten()
}

/// Resolve one child ref to a count read, against a level slice and the
/// per-batch `computed` scratch; a `Big` read borrows the `BigUint`.
///
/// Marginal leaves can retain bare label references with an empty count
/// column; their counts still come from the fixed leaf labels.
#[inline]
pub(crate) fn read_count<'a>(
    level_idx: usize,
    side: EncodedChildRef,
    vtree: &Vtree,
    levels: &'a [TddLevel],
    computed: &'a [Option<CountVec>],
) -> CountRead<'a> {
    if let Some(ic) = levels[level_idx].marginal_counts() {
        if side.is_reserved() {
            return CountRead::Fast(0); // ZERO sentinel — never decode (mirrors emit_or_tag)
        }
        return match ChildDecoder::marginal().value(side) {
            ValueRef::Inline(v) => CountRead::Fast(v as u128),
            // A marginal leaf keeps an empty store under the inline path (all
            // counts live inline at the parent), so a bare slot ref here is a
            // leaf-label index with a fixed count — decode it directly rather
            // than indexing the (empty) store. Reached by paths that leave a
            // leaf-side ref bare (e.g. projection) instead of inlining it.
            ValueRef::Slot(s) if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() => {
                CountRead::Fast(leaf_count(LeafLabel::from_idx(s as usize)))
            }
            ValueRef::Slot(s) => CountRead::from_slot(ic, levels[level_idx].marginal_counts_big(), s as usize),
        };
    }
    // Check pre-computed buffer (non-marginal level: plain index).
    if let Some(counts) = &computed[level_idx] {
        return counts.get(ChildDecoder::structural().node(side).idx());
    }
    // Leaf level: fixed counts.
    if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() {
        return CountRead::Fast(leaf_count(LeafLabel::from_idx(ChildDecoder::structural().node(side).idx())));
    }
    unreachable!("counts not available for level {}", level_idx);
}


/// Weighted analogue of [`read_count`]: resolve a child node's semiring value
/// for the weighted marginalization cascade. Reads, in order:
///   0. a leaf level by label, never through the store column: a leaf-side ref
///      is a bare `LeafLabel` index whether the leaf is structural or
///      weight-marginal (the pinned column is installed in label order by
///      `diagram::leaf_column_vals`),
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
    side: EncodedChildRef,
    vtree: &Vtree,
    cols: &LevelColumns<'a>,
    computed_weights: &'a [Option<Vec<WeightValue>>],
) -> std::borrow::Cow<'a, WeightValue> {
    let ws = cols.store();
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) {
        if side.is_reserved() {
            // `ZERO` sentinel — mirrors `read_count`. Leaf levels only ever
            // carry Pos/Neg/One, but the bit is tested before every decode.
            return std::borrow::Cow::Owned(ws.wzero());
        }
        let label_idx = ChildDecoder::marginal().index(side);
        let v = ws.leaf_val(var, LeafLabel::from_idx(label_idx));
        #[cfg(debug_assertions)]
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
        if side.is_reserved() {
            return std::borrow::Cow::Owned(ws.wzero());
        }
        return std::borrow::Cow::Borrowed(&values[ChildDecoder::marginal().index(side)]);
    }
    if let Some(w) = &computed_weights[level_idx] {
        return std::borrow::Cow::Borrowed(&w[ChildDecoder::structural().node(side).idx()]);
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
    expect: &WeightValue,
) -> bool {
    use crate::diagram::weight_key;
    let Some(col) = ws.level(level_idx) else { return true };
    if col.len() != crate::diagram::LEAF_WIDTH {
        return false; // some pass compacted, erased or appended to the column
    }
    weight_key(&col[label_idx]) == weight_key(expect)
}
