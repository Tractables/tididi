//! Shared marginal-slot primitives.
//!
//! These are the count-keyed slot building blocks used across the
//! marginal-canonical machinery: the marginalize path (`weight.rs`), the
//! post-tagger compaction (`minimize/slot_prune.rs`), the (P) same-left fusion
//! (`minimize/contract/p_fusion.rs`), and the marginal invariant checkers
//! (`query/validate_marg.rs`). One shared home, no copies.

use num_bigint::BigUint;
use rustc_hash::FxHashMap;

use crate::tdd::counts::ApplyBudget;
use crate::tdd::transform::pairwise::conjoin::{try_push, ApplyError};
use crate::tdd::types::{BigSide, Tdd};
use crate::vtree::{VtreeIdx, VtreeNode};

/// Side of a parent's vtree node at which a marginal child sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildSide {
    Left,
    Right,
}

/// Counts and big counts for a marginal-level slot, hashable / orderable
/// so we can group equal-count slots.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CountKey {
    Small(u128),
    Big(BigUint),
}

/// Append `key` as a NEW slot on a marginal store, fallibly. Returns the new
/// slot index.
///
/// One home for the store's OVERFLOW convention: `counts[i]` holds the small
/// count, or the `u128::MAX` sentinel meaning "the real value is `big`'s entry
/// for slot `i`". `big` is a sparse slot-keyed table allocated on the first
/// overflow, so a `Small` push writes nothing there (an absent entry IS the
/// "fits the fast lane" encoding) and a `Big` push records exactly one entry.
///
/// Every growth is budget-tracked (`try_push` / [`BigSide::try_insert`] under
/// [`ApplyBudget`], the same accounting the fast column uses), so an
/// over-budget store push surfaces as `ApplyError::OverBudget` rather than
/// aborting.
///
/// This is the minting half of every production slot path: a caller that wants
/// C3-by-construction reuse checks [`SlotInterner`]'s map first and only pushes
/// on a miss (`apply_p_fusion`).
pub(crate) fn push_count_key(
    counts: &mut Vec<u128>,
    big: &mut Option<BigSide>,
    key: &CountKey,
) -> Result<u32, ApplyError> {
    let new_idx = counts.len() as u32;
    match key {
        CountKey::Small(c) => {
            // A small count landing exactly on the sentinel would be re-read as
            // OVERFLOW with no `big` entry behind it; producers must route that
            // value to `Big` (see `sum_marginal_counts` and its pinned test).
            debug_assert!(*c != u128::MAX, "small count must not alias the overflow sentinel");
            try_push(counts, *c)?;
        }
        CountKey::Big(v) => {
            try_push(counts, u128::MAX)?;
            big.get_or_insert_with(BigSide::default)
                .try_insert::<ApplyBudget>(counts.len() - 1, v.clone())?;
        }
    }
    Ok(new_idx)
}

/// Apply `f` to every reference the nodes of `level` hold on `side`.
///
/// This is the ONE home for the side→field mapping: `ChildSide::Left` means
/// `pair.left.0` for a multi-pair node and `node.a` for an inline one;
/// `ChildSide::Right` means `pair.right.0` / `node.b`. (An inline node stores
/// its single pair in its own `(a, b)` words — and inlining requires `b` to
/// carry no `LEAF_BIT`, so `b` is a plain index there, exactly like
/// `pair.right.0`.) Leaves and tombstones hold no refs and are skipped.
///
/// Used by every pass that rewrites one side's refs through a remap table:
/// slot-prune's parent-ref rewrite (integer and weighted) and the C2
/// content-twin grandparent rewrite.
///
/// NOT usable by `minimize::prune`'s node-index remap: that loop filters each
/// node on a `reachable` bitmap and rewrites BOTH sides in a single visit —
/// a different traversal, not a `side` instantiation of this one. Don't try to
/// fold it in here.
#[inline]
pub(crate) fn for_each_side_ref_mut(
    level: &mut crate::tdd::types::TddLevel,
    side: ChildSide,
    mut f: impl FnMut(&mut u32),
) {
    for ni in 0..level.nodes.len() {
        if level.nodes[ni].is_leaf() {
            continue;
        }
        if level.nodes[ni].is_multi() {
            for p in level.pairs_mut(ni) {
                f(match side {
                    ChildSide::Left => &mut p.left.0,
                    ChildSide::Right => &mut p.right.0,
                });
            }
        } else {
            let node = &mut level.nodes[ni];
            f(match side {
                ChildSide::Left => &mut node.a,
                ChildSide::Right => &mut node.b,
            });
        }
    }
}

/// Locate the side at which `child` sits in `parent`.
fn side_of(tdd: &Tdd, parent: VtreeIdx, child: VtreeIdx) -> ChildSide {
    match tdd.vtree.node(parent) {
        VtreeNode::Internal { left, right, .. } => {
            if *left == child {
                ChildSide::Left
            } else {
                debug_assert_eq!(*right, child, "child must be left or right of parent");
                ChildSide::Right
            }
        }
        _ => panic!("parent must be internal vtree node"),
    }
}

/// The boundary-marginal test for ONE vtree node: `Some((v, parent, side))`
/// iff `v`'s level is marginal and its vtree parent's level is not. This is the
/// single definition of "boundary marginal level" — both collectors below are
/// just different traversals feeding it.
#[inline]
fn boundary_entry(tdd: &Tdd, v: VtreeIdx) -> Option<(VtreeIdx, VtreeIdx, ChildSide)> {
    if !tdd.levels[v.idx()].is_marginal() {
        return None;
    }
    let parent = tdd.vtree.node(v).parent()?; // root: no parent
    if tdd.levels[parent.idx()].is_marginal() {
        return None; // deep marginal: parent also marginal
    }
    Some((v, parent, side_of(tdd, parent, v)))
}

/// Fill `out` with every boundary marginal level and its non-marginal parent.
pub(crate) fn boundary_marginal_levels_into(
    tdd: &Tdd,
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    out.extend((0..tdd.levels.len()).filter_map(|i| boundary_entry(tdd, VtreeIdx(i as u32))));
}

/// Iterate boundary marginal levels with their non-marginal parent.
pub(crate) fn boundary_marginal_levels(tdd: &Tdd) -> Vec<(VtreeIdx, VtreeIdx, ChildSide)> {
    let mut out = Vec::new();
    boundary_marginal_levels_into(tdd, &mut out);
    out
}

/// Fill `out` with the boundary marginal levels **whose parent is in
/// `parents`** — the same triples `boundary_marginal_levels` would yield, in
/// the same (ascending marginal-child index) order, restricted to that parent
/// set.
///
/// Why this exists: a boundary's parent is by definition the vtree parent of
/// its marginal level, so a caller that already knows the parents it cares
/// about can reach their (at most two) boundaries directly. The all-levels scan
/// costs O(levels) *per call*, and `apply_p_fusion_inner` is called once per
/// marginal-boundary parent inside the contract fixpoint — turning a per-parent
/// constant into an O(parents x levels) sweep over the whole diagram, plus a
/// throwaway `Vec` each time, to keep at most two entries.
pub(crate) fn boundary_marginal_levels_of(
    tdd: &Tdd,
    parents: &[VtreeIdx],
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    for &p in parents {
        if let VtreeNode::Internal { left, right, .. } = tdd.vtree.node(p) {
            out.extend(boundary_entry(tdd, *left));
            out.extend(boundary_entry(tdd, *right));
        }
    }
    // Restore the all-levels traversal's ordering and its once-per-level
    // property (a repeated parent in `parents` would otherwise yield its
    // boundaries twice). A marginal level has exactly one boundary entry, so
    // keying both on the child index is exact.
    out.sort_unstable_by_key(|&(v, _, _)| v.0);
    out.dedup_by_key(|&mut (v, _, _)| v.0);
}

// ── SlotInterner ─────────────────────────────────────────────────────────────

/// Seeded dedup map from [`CountKey`] to slot index, used by the p-fusion and
/// slot-prune compaction paths to keep marginal stores C3 (one slot per value).
pub(crate) struct SlotInterner {
    pub(super) map: FxHashMap<CountKey, u32>,
}

impl SlotInterner {
    /// Create an empty interner.
    pub(crate) fn new() -> Self {
        Self { map: FxHashMap::default() }
    }

    /// Seed from an existing `(counts, big)` store so that subsequent lookups
    /// reuse existing slots for equal values.
    /// Duplicate counts in the seed are collapsed to the first occurrence
    /// (same dedup semantics as `dedup_fresh_store`).
    pub(crate) fn seed(
        &mut self,
        counts: &[u128],
        big: Option<&BigSide>,
    ) {
        for i in 0..counts.len() {
            let key = count_key_at(counts, big, i);
            self.map.entry(key).or_insert(i as u32);
        }
    }
}

/// Read the marginal count at `slot` as a `CountKey`.
///
/// Mirrors the OVERFLOW-sentinel convention: `counts[slot] == u128::MAX`
/// means the real value is `big`'s entry for `slot`.
pub(crate) fn count_key_at(
    counts: &[u128],
    big: Option<&BigSide>,
    slot: usize,
) -> CountKey {
    const OVERFLOW: u128 = u128::MAX;
    let c = counts[slot];
    if c == OVERFLOW {
        let b = big
            .and_then(|b| b.get(slot))
            .expect("OVERFLOW sentinel requires a marginal_counts_big entry")
            .clone();
        CountKey::Big(b)
    } else {
        CountKey::Small(c)
    }
}

/// Sum marginal counts at `indices`. Returns a CountKey (Small or Big).
pub(crate) fn sum_marginal_counts(
    counts: &[u128],
    big: Option<&BigSide>,
    indices: &[u32],
) -> CountKey {
    use crate::tdd::types::MargRef;
    const OVERFLOW: u128 = u128::MAX;
    // Decode one marg-side ref to its contributing count: a ref is EITHER an
    // inline count (bit-30 clear: the value IS the count, no array load) or a
    // tagged slot (bit-30 set: index `counts`). `MargRef::from_raw` does this
    // split; its bit-31 assert will fire (debug) if a ZERO sentinel ever
    // reaches here — by design, so the source is localized rather than papered
    // over with a guessed 0.
    let load = |i: u32| -> u128 {
        match MargRef::from_raw(i) {
            MargRef::Inline(v) => v as u128,
            MargRef::Slot(s) => counts[s as usize],
        }
    };
    // First pass: try all-small sum without overflow. If any contribution is the
    // OVERFLOW sentinel or the sum would overflow u128, fall through to BigUint.
    let mut acc: u128 = 0;
    let mut any_big = false;
    for &i in indices {
        let c = load(i);
        if c == OVERFLOW {
            any_big = true;
            break;
        }
        match acc.checked_add(c) {
            Some(s) => acc = s,
            None => {
                any_big = true;
                break;
            }
        }
    }
    // `acc == OVERFLOW` falls through to the BigUint path even though the sum
    // fits u128: a `Small(u128::MAX)` would later be *stored* as the OVERFLOW
    // sentinel without a `marginal_counts_big` entry, and the next
    // `count_key_at` read of that slot would panic.
    if !any_big && acc != OVERFLOW {
        return CountKey::Small(acc);
    }
    // BigUint sum. An inline ref's value (≤ MARG_INLINE_MAX) is never OVERFLOW,
    // so only tagged slots can route into `big`.
    let mut big_acc = BigUint::default();
    for &i in indices {
        match MargRef::from_raw(i) {
            MargRef::Slot(s) => {
                let idx = s as usize;
                if counts[idx] == OVERFLOW {
                    let b = big
                        .and_then(|b| b.get(idx))
                        .expect("OVERFLOW sentinel requires a marginal_counts_big entry");
                    big_acc += b;
                } else {
                    big_acc += BigUint::from(counts[idx]);
                }
            }
            MargRef::Inline(v) => {
                big_acc += BigUint::from(v);
            }
        }
    }
    CountKey::Big(big_acc)
}

#[cfg(test)]
#[path = "marg_slots_sum_tests.rs"]
mod sum_tests;
