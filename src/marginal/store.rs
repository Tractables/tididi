//! Reading and writing the per-node marginal count / weight stores.

use rustc_hash::FxHashMap;

use crate::value_fold::{CountRead, CountVec, COUNT_OVERFLOW};
use crate::engine::ReservePolicy;
use crate::value_fold::Count;
use crate::reduce::slots::{count_key_at};
use crate::diagram::WeightVal;
use crate::diagram::{BigSide, ChildSide, LeafLabel, MarginalSide, NodeIdx, SideView, TddLevel, ValueRef, Tdd};
use crate::diagram::remap_side_refs;
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use super::column::LevelColumns;
use super::leaf_count;

/// Free the dead per-node store of `parent`'s already-marginal children at the
/// moment `parent` itself becomes marginal.
///
/// Once `parent` is marginal it holds the aggregate that summed out its whole
/// subtree and (being marginal) carries no pair lists referencing anything
/// below it. A vtree node has exactly one parent, so each marginal child is now
/// unreachable from the root: its per-node data is dead weight that nothing will
/// ever read again. We free it here — exactly once, at the marginalization of
/// `parent`, touching only the two children — so the invariant "a marginal level
/// under a marginal parent carries no data" holds with O(1) work and no sweep.
///
/// Count/weight-preserving by construction: a marginal level's data IS its
/// subtree's value and it has no pairs to descend through, so `model_count` /
/// `weighted_output_value` stop at `parent` and never touch the freed children.
/// Handles both representations — integer (`marginal_counts`) and weighted (the
/// external `WeightStore` slot, cleared via `ws` when present; a level's slot
/// carrier `weight_width` is zeroed either way so `width()` reports 0.
pub(super) fn free_subsumed_marginal_children(
    tdd: &mut Tdd,
    vtree: &Vtree,
    parent: VtreeIdx,
    mut ws: Option<&mut WeightStore>,
) {
    if vtree.node(parent).is_leaf() {
        return;
    }
    let (l, r) = vtree.children(parent);
    for c in [l.idx(), r.idx()] {
        let lvl = &mut tdd.levels[c];
        if !lvl.is_marginal() {
            // A structural child under a marginal parent never arises here:
            // marginalization makes a level marginal only after both children
            // are marginal-or-leaf (`assert_can_make_marginal`). Skip leaves.
            continue;
        }
        // Integer-marginal child: empty the count store but keep `Some` so the
        // level stays marginal/terminal; drop any big-overflow side-vec.
        if let Some(v) = lvl.marginal_counts_mut() {
            if !v.is_empty() {
                *v = Vec::new();
            }
            lvl.clear_marginal_big();
        }
        // Weight-marginal child: zero the slot carrier (→ width 0) and drop the
        // external store. `is_weight_marginal()` stays true (flag untouched).
        //
        // EXCEPT a vtree LEAF — the PIN INVARIANT (see `marginalize_leaf_weighted`).
        // A weight-marginal leaf's column is not this `Tdd`'s data to free: it is
        // the label-ordered 3-slot cache of `WeightStore::leaf_val`, keyed by vtree
        // index and shared with every `Tdd` this one's store reaches (fresh
        // clause diagrams whose leaf level is still STRUCTURAL hold bare leaf-LABEL
        // refs that alias its slots by position). Erasing it here leaves those
        // holders reading an empty column — a panic on `&values[slot]`, or a silent
        // width-0 mass drop through the `map_or(0, len)` readers. The dead-data
        // reclaim this function exists for simply does not apply: the column is a
        // cache of three constants, O(1) and re-derivable, not a per-node store
        // that grows with the diagram.
        if lvl.is_weight_marginal() && lvl.weight_width() != 0 && !vtree.node(VtreeIdx(c as u32)).is_leaf() {
            lvl.set_weight_width(0);
            if let Some(ws) = ws.as_deref_mut() {
                ws.set_level(c, Vec::new());
            }
        }
    }
}


/// Resolve one child ref to a count read, against a level slice and the
/// per-batch `computed` scratch. One reader for both contexts — the finished
/// `Tdd` of the marginal cascade and the in-flight `levels` of a streaming
/// apply — which differ only in the reservation policy of the scratch column.
/// A `Big` read hands back the borrowed `BigUint` directly.
///
/// Marginal level: self-describing decode under the bit-30-clear==slot
/// polarity. Bit 30 alone disambiguates:
///   bit-30 SET   → inline count value (strip the tag; ≤ 2^30−1, so never an
///                  overflow sentinel).
///   bit-30 CLEAR → bare slot index into `ic` (a pre-tag mid-batch ref is a
///                  bare node index, which IS its slot index).
/// The flag-gated decode this polarity replaced was a miscount waiting to
/// happen: a parent level rebuilt from fresh scratch (e.g. by the
/// clause-specialized apply) can lose its `marginal_inlined_*` marker while its
/// pairs still carry bit-30 inline refs.
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
            // A marginal LEAF keeps an empty store under the inline path (all
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
                // Belt-and-braces fallback: the overflow value may only be
                // recorded in the in-flight `computed` column.
                if let Some(bv) = computed[level_idx].as_ref().and_then(|cv| cv.big_val(node_idx)) {
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


/// Weighted analogue of [`read_count`]: resolve a child node's exact
/// semiring value for the weighted marginalization cascade. Reads, in order:
///   0. **LEAF levels resolve by LABEL**, never through the `WeightStore` column
///      — the weighted mirror of [`read_count`]'s fixed-count leaf arm.
///      A leaf-side ref is a bare `LeafLabel` index in both representations: a
///      structural leaf's implicit {One, Pos, Neg} nodes, and a weight-marginal
///      leaf's pinned 3-slot column (installed in exactly that order by
///      [`marginalize_leaf_weighted`]). Routing through the column instead would
///      key on the SHARED store rather than on THIS `Tdd`'s marginality:
///      a structural leaf level of a fresh clause diagram would then decode its
///      genuine label refs against whatever column the store happens to hold
///      for that vtree index. Label resolution is correct for both, and is
///      the reading the pin invariant exists to keep exact.
///   1. the external [`WeightStore`] for a level already weight-marginalized
///      (this batch or a prior one) — marginal-side refs are bare slots in weighted
///      mode;
///   2. the per-batch `computed_weights` buffer for a level computed earlier in
///      this batch but not yet stored to the `WeightStore`.
///
/// Returns `Cow`: store-slot and per-batch reads borrow (no clone); only the
/// ZERO sentinel and leaf bases materialize an owned value.
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
            // ZERO sentinel — mirrors read_count. Leaf levels only ever
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
    // INTERNAL level: per-Tdd flag FIRST, mirroring the leaf arm above and the
    // integer twin `read_count` (whose store lives inside the level, so
    // it is per-Tdd by construction). The store can hold a column at this index
    // installed by ANOTHER live Tdd it was merged with (a sibling accumulator) while
    // THIS Tdd's level is still structural — its node indices are not slots of
    // that foreign column. At a leaf the label/slot aliasing makes such a read
    // value-correct anyway (the pin); an internal level has no such backstop, so
    // the store read is gated on this Tdd's own marginality and a structural
    // level falls through to the per-batch computed buffer (the WEIGHTED STORE
    // MIRRORS THE READ Tdd invariant — asserted in `ensure_weights`' walk guard).
    if let Some(values) = cols.get(level_idx) {
            let raw = node_idx as u32;
            if MarginalSide(raw).is_zero_sentinel() {
                // ZERO sentinel — mirrors read_count
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
    // No trailing leaf arm: the leaf branch is the FIRST test above, so this point
    // is only reached on an internal level (single source of truth for leaf reads).
    unreachable!("weighted value not available for level {}", level_idx);
}

/// Debug-only companion to the leaf branch of [`read_weight`]: when the
/// pinned leaf column IS installed, its slot must equal the label's `leaf_val`.
/// A mismatch means some pass compacted, reordered, or appended to the column —
/// exactly what the pin invariant forbids. Absent / short columns are not an
/// error here (a leaf level may simply not be weight-marginal yet).
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
        // Any other length means a pass compacted / erased / appended to the
        // column (the zero-slot subsumed state included — it is no longer
        // written). Report it so the debug build catches the regression.
        return false;
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



// ── Invariant-10 marginalize helpers (dedup_fresh_store + parent-ref remap) ───────
//
// invariant 10 — no two slots at a marginal level share a model count — is established
// **at birth** for the stores the marginalize pass builds (`marginal::fold`) by
// these two helpers: `dedup_fresh_store` merges
// duplicate-count slots before the store is installed, and
// `remap_parent_refs_pretag` redirects the parent level's marginal-side refs onto
// the surviving canonical slots. (The apply streaming-emit path establishes invariant 10
// later, at post-tagger slot-prune in `reduce/slot_prune.rs`.)

/// Compact a freshly-built marginal store so that **each count value occupies
/// at most one slot** (invariant 10), returning the deduped store and a
/// slot-index remap table: `remap[old] = new` (identity where no dedup occurred,
/// canonical-slot index otherwise).
///
/// The caller must remap every parent-side ref that indexes into the old store
/// using `remap[old_slot]`.  Refs are not remapped here — this function only
/// touches the store itself.
///
/// # Store is born satisfying invariant 10 (for the marginalize pass's callers): no duplicate
/// count values; enforced here. Apply-emit-born stores do not call this at
/// emit time — their invariant 10 is established later by `prune_value_slots`.
///
/// Duplicate slots are merged to the FIRST occurrence of each value. The returned vecs may
/// be shorter than the inputs when duplicates were found; if no duplicates
/// exist they are returned unchanged.
///
/// The fast count column is compacted **in place**: callers hand it over by
/// move (the marginalize pass `take`s the level's `CountVec` and passes
/// `into_parts()`), so no second full-length store
/// is ever resident beside this one at the peak. The sparse overflow table is
/// rekeyed into a fresh [`BigSide`] instead — its keys are slot indices, and a
/// survivor's index changes — which costs at most the surviving overflow
/// entries, never a width-sized buffer. Same mechanism and same soundness
/// argument as `IntFold::compact_store` (`reduce/slot_prune.rs`), which
/// compacts an already-installed store; both take their value-dedup key from
/// the shared `count_key_at`, so the Small/Big split is decided in one place.
pub(crate) fn dedup_fresh_store(
    mut counts: Vec<u128>,
    big: Option<BigSide>,
) -> (Vec<u128>, Option<BigSide>, Vec<u32>) {
    let n = counts.len();
    // Written on every path below (mint or merge), for every `i`.
    let mut remap: Vec<u32> = vec![0; n];
    let mut count_to_canonical: FxHashMap<Count, u32> = FxHashMap::default();
    let mut new_len = 0usize;

    // SOUNDNESS (why a move can't clobber a slot still to be read): dedup never
    // grows the store — distinct values ≤ slots — so the write cursor `new_len`
    // is at or behind the read cursor `i` at every step (`new_len` advances at
    // most once per `i`). The key at `i` is read before the move, and every
    // later read is at a strictly larger index than any write done so far.
    //
    // `count_to_canonical` maps a value to the COMPACTED index of its first
    // slot, so the remap is final as it is written — no second composition pass
    // over a `compact_idx` table, and no reading of the destroyed layout.
    //
    // The overflow table is not touched in here: it is keyed by slot, so it is
    // rekeyed in one drain after `remap` is complete (below).
    for i in 0..n {
        let key = count_key_at(&counts, big.as_ref(), i);
        // A hit means `i` holds a value an earlier surviving slot already
        // carries (invariant 10 merge); a miss mints the next compacted slot.
        if let Some(&hit) = count_to_canonical.get(&key) {
            remap[i] = hit;
            continue;
        }
        count_to_canonical.insert(key, new_len as u32);
        counts[new_len] = counts[i];
        remap[i] = new_len as u32;
        new_len += 1;
    }

    if new_len == n {
        // No duplicates: every slot minted its own, so the store already satisfies invariant 10
        // and `remap` is the identity — hand both back untouched, overflow
        // table included (rekeying it would be the identity too).
        return (counts, big, remap);
    }

    // Rekey the overflow table: consume it in one ascending drain and re-file
    // each value under its slot's compacted index. A slot that merged away maps
    // onto its canonical's index and writes an EQUAL value over it (equality is
    // what made them merge), so the result is the same either way — and a
    // merged-away `BigUint` is dropped as the drain passes it. Values move;
    // nothing here clones.
    let new_big = big.map(|b| {
        b.into_iter().map(|(slot, v)| (remap[slot as usize], v)).collect::<BigSide>()
    });

    counts.truncate(new_len);
    // Slack ceiling: the fast column is compacted IN PLACE, so the capacity
    // observed here is the PRE-compaction one. Shrinking at 2× therefore
    // reclaims exactly when the store more than halved — the effective ceiling
    // on the slack this level's store keeps for its lifetime, matching
    // `IntFold::compact_store`. The rebuilt overflow table needs no such policy:
    // its slack is bounded by the surviving overflow set, not by the width.
    if counts.capacity() > 64 && counts.capacity() > 2 * counts.len() {
        counts.shrink_to_fit();
    }

    (counts, new_big, remap)
}

/// Remap parent-level marginal-side refs into a child level using a slot remap
/// table built by [`dedup_fresh_store`].
///
/// At the **pre-tagger** construction sites (the marginalize pass in `marginal::fold`)
/// every parent ref into the child is a bare slot index (bit-30 clear, never an
/// inline count). `remap[old_slot] = new_slot` was returned by `dedup_fresh_store`.
///
/// The walk itself is [`remap_side_refs`]; what this adds is the identity
/// early-return and the redirection of `tdd.output.local` when the diagram root
/// lives at the marginal level (rare but defensive).
///
/// # Store is born satisfying invariant 10: no duplicate count values; enforced here, not by a
/// later canon pass.
pub(super) fn remap_parent_refs_pretag(
    tdd: &mut Tdd,
    child_v: VtreeIdx,
    parent_v: VtreeIdx,
    side: ChildSide,
    remap: &[u32],
) {
    if remap.iter().enumerate().all(|(i, &r)| r == i as u32) {
        // Identity remap — nothing to do.
        return;
    }

    remap_side_refs(&mut tdd.levels[parent_v.idx()], side, SideView::marginal(), remap);

    // Output update when diagram root is at this marginal level (rare but defensive).
    if tdd.output.vtree == child_v {
        let old = tdd.output.local.idx() as u32;
        tdd.output.local = NodeIdx(remap[old as usize]);
    }
}
