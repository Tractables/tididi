//! Reading and writing the per-node marginal count / weight stores.

use crate::engine::Engine;
use rustc_hash::FxHashMap;

use crate::value_fold::{
    ensure_fold_walk, unwrap_infallible, ColumnRetention, Count, CountRead, CountVec, IntFold,
    WeightFold, STREAM_OVERFLOW,
};
use crate::engine::RecoveryPanic;
use crate::reduce::slots::{CountKey, count_key_at};
use crate::diagram::WeightVal;
use crate::diagram::{BigSide, LeafLabel, MargSide, ValueRef, Tdd};
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use super::column::LevelColumns;

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
        // clause TDDs whose leaf level is still STRUCTURAL hold bare leaf-LABEL
        // refs that alias its slots by position). Erasing it here leaves those
        // holders reading an empty column — a panic on `&vals[slot]`, or a silent
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

/// Ensure counts are available for a given level (compute from pairs if still
/// explicit). The shared [`ensure_fold_walk`] with this context's readers
/// wired into [`IntFold::fold`] via [`compute_marginal_node_int`]. Early-outs
/// (walk guard): memoization cache hit, counts already inlined on the TDD
/// level itself (`is_marginal`), or a leaf (counts come from the formula on
/// demand inside the reader).
///
/// [`ColumnRetention::All`] is mandatory here and takes no caller knob: the
/// sole caller is the integer freeze pass, which needs EVERY walked level's
/// column — the freeze cascade `take`s each one to install it as that
/// level's marginal store, and the buffer is shared across all batch targets.
pub(super) fn ensure_counts(
    eng: &Engine,
    tdd: &Tdd,
    level_idx: VtreeIdx,
    vtree: &Vtree,
    computed: &mut [Option<CountVec<RecoveryPanic>>],
) {
    unwrap_infallible(ensure_fold_walk::<IntFold, RecoveryPanic, _, _>(
        eng,
        level_idx.idx(),
        vtree,
        &tdd.levels,
        computed,
        &Count::Fast(0),
        &|i| tdd.levels[i].is_marginal(),
        &|lvl, i, l_i, r_i, computed| {
            compute_marginal_node_int(tdd, &tdd.levels[lvl], i, l_i, r_i, computed)
        },
        ColumnRetention::All,
    ));
}

/// Resolve one child ref to a count read on a finished `Tdd`.
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
/// clause-specialized apply) can lose its `marg_inlined_*` marker while its
/// pairs still carry bit-30 inline refs.
#[inline]
fn read_marginal_count<'a>(
    tdd: &'a Tdd,
    level_idx: usize,
    node_idx: usize,
    computed: &'a [Option<CountVec<RecoveryPanic>>],
) -> CountRead<'a> {
    if let Some(ic) = tdd.levels[level_idx].marginal_counts() {
        let raw = node_idx as u32;
        if MargSide(raw).is_zero_sentinel() {
            return CountRead::Fast(0); // ZERO sentinel — never decode (mirrors emit_or_tag)
        }
        return match ValueRef::from_raw(MargSide(raw)) {
            ValueRef::Inline(v) => CountRead::Fast(v as u128),
            // A marginal LEAF keeps an empty store under the inline path (all
            // counts live inline at the parent), so a bare slot ref here is a
            // leaf-label index with a fixed count — decode it directly rather
            // than indexing the (empty) store. Reached by paths that leave a
            // leaf-side ref bare (e.g. projection) instead of inlining it.
            ValueRef::Slot(s) if tdd.vtree.node(VtreeIdx(level_idx as u32)).is_leaf() => {
                CountRead::Fast(match LeafLabel::from_idx(s as usize) {
                    LeafLabel::Zero => 0,
                    LeafLabel::One => 2,
                    LeafLabel::Pos | LeafLabel::Neg => 1,
                })
            }
            ValueRef::Slot(s) => {
                let v = ic[s as usize];
                if v != STREAM_OVERFLOW {
                    return CountRead::Fast(v);
                }
                if let Some(bv) = tdd.levels[level_idx]
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
    if tdd.vtree.node(VtreeIdx(level_idx as u32)).is_leaf() {
        return CountRead::Fast(match LeafLabel::from_idx(node_idx) {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        });
    }
    unreachable!("counts not available for level {}", level_idx);
}

/// Fold one marginal node: `Σ over pairs (left_count × right_count)`.
/// The finished-`Tdd` reader adapter over [`IntFold::fold`] — the one shared
/// two-pass integer discipline (u128 fast pass; exact `BigUint` re-pass with
/// mixed-magnitude branching on overflow; exact-max promotion owned by
/// `Count::from_u128`).
pub(super) fn compute_marginal_node_int(
    tdd: &Tdd,
    level: &crate::diagram::TddLevel,
    i: usize,
    li: usize,
    ri: usize,
    computed: &[Option<CountVec<RecoveryPanic>>],
) -> Count {
    IntFold::fold(
        level.pairs_iter_of_idx(i),
        |k| read_marginal_count(tdd, li, k, computed),
        |k| read_marginal_count(tdd, ri, k, computed),
    )
}

/// Weighted analogue of [`read_marginal_count`]: resolve a child node's exact
/// semiring value for the weighted marginalization cascade. Reads, in order:
///   0. **LEAF levels resolve by LABEL**, never through the `WeightStore` column
///      — the weighted mirror of [`read_marginal_count`]'s fixed-count leaf arm.
///      A leaf-side ref is a bare `LeafLabel` index in BOTH representations: a
///      structural leaf's implicit {One, Pos, Neg} nodes, and a weight-marginal
///      leaf's pinned 3-slot column (installed in exactly that order by
///      [`marginalize_leaf_weighted`]). Routing through the column instead would
///      key on the SHARED store rather than on THIS `Tdd`'s marginality:
///      a structural leaf level of a fresh clause TDD would then decode its
///      genuine label refs against whatever column the store happens to hold
///      for that vtree index. Label resolution is correct for both, and is
///      the reading the pin invariant exists to keep exact.
///   1. the external [`WeightStore`] for a level already weight-marginalized
///      (this batch or a prior one) — marg-side refs are bare slots in weighted
///      mode;
///   2. the per-batch `computed_weights` buffer for a level computed earlier in
///      this batch but not yet stored to the `WeightStore`.
///
/// Returns `Cow`: store-slot and per-batch reads borrow (no clone); only the
/// ZERO sentinel and leaf bases materialize an owned value.
fn read_marginal_weight<'a>(
    tdd: &Tdd,
    level_idx: usize,
    node_idx: usize,
    cols: &LevelColumns<'a>,
    computed_weights: &'a [Option<Vec<WeightVal>>],
) -> std::borrow::Cow<'a, WeightVal> {
    let ws = cols.store();
    if let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(VtreeIdx(level_idx as u32)) {
        let raw = node_idx as u32;
        if MargSide(raw).is_zero_sentinel() {
            // ZERO sentinel — mirrors read_marginal_count. Leaf levels only ever
            // carry Pos/Neg/One, but the bit is tested before every decode.
            return std::borrow::Cow::Owned(ws.wzero());
        }
        let label_idx = match ValueRef::from_raw(MargSide(raw)) {
            ValueRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
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
    // integer twin `read_marginal_count` (whose store lives inside the level, so
    // it is per-Tdd by construction). The store can hold a column at this index
    // installed by ANOTHER live Tdd it was merged with (a sibling accumulator) while
    // THIS Tdd's level is still structural — its node indices are NOT slots of
    // that foreign column. At a leaf the label/slot aliasing makes such a read
    // value-correct anyway (the pin); an internal level has no such backstop, so
    // the store read is gated on this Tdd's own marginality and a structural
    // level falls through to the per-batch computed buffer (the WEIGHTED STORE
    // MIRRORS THE READ Tdd invariant — asserted in `ensure_weights`' walk guard).
    if let Some(vals) = cols.get(level_idx) {
            let raw = node_idx as u32;
            if MargSide(raw).is_zero_sentinel() {
                // ZERO sentinel — mirrors read_marginal_count
                return std::borrow::Cow::Owned(ws.wzero());
            }
            let slot = match ValueRef::from_raw(MargSide(raw)) {
                ValueRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
                ValueRef::Slot(s) => s as usize,
            };
            return std::borrow::Cow::Borrowed(&vals[slot]);
        }
    if let Some(w) = &computed_weights[level_idx] {
        return std::borrow::Cow::Borrowed(&w[node_idx]);
    }
    // No trailing leaf arm: the leaf branch is the FIRST test above, so this point
    // is only reached on an internal level (single source of truth for leaf reads).
    unreachable!("weighted value not available for level {}", level_idx);
}

/// Debug-only companion to the leaf branch of [`read_marginal_weight`]: when the
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

/// Weighted analogue of [`compute_marginal_node_int`]: the finished-`Tdd`
/// reader adapter over [`WeightFold::fold`] — one clean pass over the exact
/// semiring (rationals don't overflow).
pub(super) fn compute_marginal_node_weight(
    tdd: &Tdd,
    level: &crate::diagram::TddLevel,
    i: usize,
    li: usize,
    ri: usize,
    ws: &WeightStore,
    computed_weights: &[Option<Vec<WeightVal>>],
) -> WeightVal {
    let cols = LevelColumns::new(ws, &tdd.levels);
    WeightFold::fold(
        level.pairs_iter_of_idx(i),
        |k| read_marginal_weight(tdd, li, k, &cols, computed_weights),
        |k| read_marginal_weight(tdd, ri, k, &cols, computed_weights),
        ws.wzero(),
    )
}

/// Weighted analogue of [`ensure_counts`] — the same shared
/// [`ensure_fold_walk`] with [`WeightFold`]. Early-outs (walk guard): cached
/// in buffer, already weight-marginal (`ws.is_set` — this quadrant's
/// marginality predicate), or a leaf (base values come from the semiring on
/// demand in `read_marginal_weight`). The scratch column reserves through
/// `RecoveryPanic` (fallible-allocation parity, A1) — a pathological width
/// raises the controlled recovery-split panic instead of an allocator abort.
///
/// `retain` is the caller's column-lifetime policy, and this is the one ensure
/// wrapper whose callers genuinely differ: the weighted freeze pass
/// needs [`ColumnRetention::All`] (its cascade `take`s every level's column),
/// while [`weighted_output_value`] reads ONLY the walk root and passes
/// [`ColumnRetention::Frontier`].
pub(super) fn ensure_weights(
    eng: &Engine,
    tdd: &Tdd,
    level_idx: VtreeIdx,
    vtree: &Vtree,
    ws: &WeightStore,
    computed_weights: &mut [Option<Vec<WeightVal>>],
    retain: ColumnRetention,
) {
    unwrap_infallible(ensure_fold_walk::<WeightFold, RecoveryPanic, _, _>(
        eng,
        level_idx.idx(),
        vtree,
        &tdd.levels,
        computed_weights,
        &ws.wzero(),
        // WEIGHTED STORE MIRRORS THE READ Tdd: a global column at an INTERNAL
        // index belongs to THIS Tdd, so `is_set` and this Tdd's marginality
        // agree within the walk's reach. Leaves are exempt — the pinned
        // label-aliased column is legitimately global while another Tdd still
        // reads the leaf structurally, and the guard's value is inconsequential
        // there, since leaf bases resolve by label in `read_marginal_weight`
        // either way.
        &|i| ws.is_set(i),
        &|lvl, i, l_i, r_i, cw| {
            compute_marginal_node_weight(tdd, &tdd.levels[lvl], i, l_i, r_i, ws, cw)
        },
        retain,
    ));
}

// ── Born-C3 marginalize helpers (dedup_fresh_store + parent-ref remap) ───────
//
// C3 — no two slots at a marginal level share a model count — is established
// **at birth** for the stores the freeze pass builds (`marginal::fold`) by
// these two helpers: `dedup_fresh_store` merges
// duplicate-count slots before the store is installed, and
// `remap_parent_refs_pretag` redirects the parent level's marg-side refs onto
// the surviving canonical slots. (The apply streaming-emit path establishes C3
// later, at post-tagger slot-prune in `minimize/slot_prune.rs`.)

/// Compact a freshly-built marginal store so that **each count value occupies
/// at most one slot** (invariant C3), returning the deduped store and a
/// slot-index remap table: `remap[old] = new` (identity where no dedup occurred,
/// canonical-slot index otherwise).
///
/// The caller must remap every parent-side ref that indexes into the old store
/// using `remap[old_slot]`.  Refs are NOT remapped here — this function only
/// touches the store itself.
///
/// # Store is born C3 (for the marginalize pass's callers): no duplicate
/// count values; enforced here. Apply-emit-born stores do NOT call this at
/// emit time — their C3 is established later by `prune_marg_slots`.
///
/// Duplicate slots are merged to the FIRST occurrence of each value. The returned vecs may
/// be shorter than the inputs when duplicates were found; if no duplicates
/// exist they are returned unchanged.
///
/// The fast count column is compacted **in place**: callers hand it over by
/// move (the freeze pass `take`s the level's `CountVec` and passes
/// `into_parts()`), so no second full-length store
/// is ever resident beside this one at the peak. The sparse overflow table is
/// rekeyed into a fresh [`BigSide`] instead — its keys are slot indices, and a
/// survivor's index changes — which costs at most the surviving overflow
/// entries, never a width-sized buffer. Same mechanism and same soundness
/// argument as `IntFold::compact_store` (`minimize/slot_prune.rs`), which
/// compacts an already-installed store; both take their value-dedup key from
/// the shared `count_key_at`, so the Small/Big split is decided in one place.
pub(crate) fn dedup_fresh_store(
    mut counts: Vec<u128>,
    big: Option<BigSide>,
) -> (Vec<u128>, Option<BigSide>, Vec<u32>) {
    let n = counts.len();
    // Written on every path below (mint or merge), for every `i`.
    let mut remap: Vec<u32> = vec![0; n];
    let mut count_to_canonical: FxHashMap<CountKey, u32> = FxHashMap::default();
    let mut new_len = 0usize;

    // SOUNDNESS (why a move can't clobber a slot still to be read): dedup never
    // grows the store — distinct values ≤ slots — so the write cursor `new_len`
    // is at or behind the read cursor `i` at every step (`new_len` advances at
    // most once per `i`). The key at `i` is read BEFORE the move, and every
    // later read is at a strictly larger index than any write done so far.
    //
    // `count_to_canonical` maps a value to the COMPACTED index of its first
    // slot, so the remap is final as it is written — no second composition pass
    // over a `compact_idx` table, and no reading of the destroyed layout.
    //
    // The overflow table is NOT touched in here: it is keyed by slot, so it is
    // rekeyed in one drain after `remap` is complete (below).
    for i in 0..n {
        let key = count_key_at(&counts, big.as_ref(), i);
        // A hit means `i` holds a value an earlier surviving slot already
        // carries (C3 merge); a miss mints the next compacted slot.
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
        // No duplicates: every slot minted its own, so the store is already C3
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

/// Remap parent-level marg-side refs into a child level using a slot remap
/// table built by [`dedup_fresh_store`].
///
/// At the **pre-tagger** construction sites (the freeze pass in `marginal::fold`)
/// every parent ref into the child is a bare slot index (bit-30 clear, never an
/// inline count). `remap[old_slot] = new_slot` was returned by `dedup_fresh_store`.
///
/// Also redirects `tdd.output.local` when the TDD root lives at the marginal
/// level (rare but defensive).
///
/// # Store is born C3: no duplicate count values; enforced here, not by a
/// later canon pass.
pub(super) fn remap_parent_refs_pretag(
    tdd: &mut Tdd,
    child_v: VtreeIdx,
    parent_v: VtreeIdx,
    t1_is_left: bool,
    remap: &[u32],
) {
    use crate::diagram::{NodeIdx, SideView};

    if remap.iter().enumerate().all(|(i, &r)| r == i as u32) {
        // Identity remap — nothing to do.
        return;
    }

    // Bare slot remap: bit-30 clear = bare slot; mask strips high bits.
    let remap_ref = |raw: u32| -> u32 {
        if MargSide(raw).is_zero_sentinel() {
            return raw; // ZERO sentinel
        }
        // Pre-tagger: no inline refs exist yet; all marg-side refs are bare slots.
        ValueRef::slot_raw(remap[SideView::valued().coord(NodeIdx(raw)).idx()])
    };

    let plevel = &mut tdd.levels[parent_v.idx()];
    for node_idx in 0..plevel.nodes.len() {
        if plevel.nodes[node_idx].is_inline() {
            let node = &mut plevel.nodes[node_idx];
            if t1_is_left {
                node.a = remap_ref(node.a);
            } else {
                node.b = remap_ref(node.b);
            }
        } else if plevel.nodes[node_idx].is_multi() {
            let pairs = plevel.pairs_mut(node_idx);
            for p in pairs.iter_mut() {
                let f = if t1_is_left { &mut p.left } else { &mut p.right };
                *f = NodeIdx(remap_ref(f.idx() as u32));
            }
        }
    }

    // Output update when TDD root is at this marginal level (rare but defensive).
    if tdd.output.vtree == child_v {
        let old = tdd.output.local.idx() as u32;
        tdd.output.local = NodeIdx(remap[old as usize]);
    }
}
