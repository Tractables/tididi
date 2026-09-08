//! Re-resolving a swapped-in parent level's marginal refs into the output
//! child's store space.

use num_bigint::BigUint;
use rustc_hash::FxHashMap;

use crate::tdd::limits::ApplyError;
use super::super::level::TddLevel;
use super::{BigSide, MargRef, marg_inline_max, MARG_OVERFLOW_TAG, MARG_VALUE_MASK};

/// Slot value in [`resolve_swapped_marg_side`]'s interners meaning "this count
/// has no dst slot yet" — the pre-scan collected the key, and the dst seed pass
/// found no existing slot carrying it. Real slot indices are `< MARG_OVERFLOW_TAG`
/// (2^30, asserted by `MargRef::to_raw`), so `u32::MAX` cannot collide with one.
const SLOT_UNSEEDED: u32 = u32::MAX;

/// What [`resolve_swapped_marg_side`] must do with one marg-side ref of the
/// swapped-in parent.
///
/// The SINGLE classification point: the pre-scan and the rewrite pass both
/// branch on this, so "needs a dst slot" in the pre-scan is *definitionally*
/// the condition the rewrite hits — the two cannot drift apart.
enum SwapRef {
    /// Store-independent: a ZERO sentinel (bit 31) or an already-inline count
    /// (bit 30). Passes through untouched.
    Keep,
    /// Bare slot whose source count fits inline: rewritten to an inline ref,
    /// which is store-independent. Touches no store.
    Inline(u32),
    /// Bare slot whose source count is above the inline threshold (or is the
    /// `u128::MAX` BigUint sentinel): must be interned into the dst store.
    /// Carries the SOURCE slot index and its source count.
    Mint(usize, u128),
}

/// Classify one marg-side ref of a swapped-in parent. `inline_max` is read once
/// by the caller (not per ref) so both passes use the same threshold — the
/// test-only override is a thread-local cell that a re-read could observe
/// differently.
///
/// Panics (index OOB) if a bare ref points outside the source store, exactly as
/// the rewrite would; the pre-scan runs first, so that panic now precedes any
/// mutation instead of landing half-way through one.
#[inline]
fn classify_swap_ref(raw: u32, src_counts: &[u128], inline_max: u128) -> SwapRef {
    if raw & (1 << 31) != 0 {
        return SwapRef::Keep; // ZERO sentinel
    }
    if raw & MARG_OVERFLOW_TAG != 0 {
        return SwapRef::Keep; // already an inline count (bit-30 set)
    }
    // Bare slot (bit-30 clear): store-relative index into the source store.
    let s = (raw & MARG_VALUE_MASK) as usize;
    let c = src_counts[s];
    if c != u128::MAX && c <= inline_max {
        SwapRef::Inline(c as u32) // store-independent once written
    } else {
        SwapRef::Mint(s, c)
    }
}

/// Every marg-side ref of `level` on the given side, in the order the rewrite
/// loops at the end of [`resolve_swapped_marg_side`] visit them: the inline-node
/// home (`node.a`/`node.b`) first, then the pairs arena (which includes pairs no
/// live node references — contraction leaves those behind, and the rewrite
/// remaps them too). Read-only twin of those loops: the pre-scan sees exactly
/// the refs the rewrite will.
fn marg_side_refs(level: &TddLevel, is_left: bool) -> impl Iterator<Item = u32> + '_ {
    level
        .nodes
        .iter()
        .filter(|n| n.is_inline())
        .map(move |n| if is_left { n.a } else { n.b })
        .chain(level.pairs.iter().map(move |p| if is_left { p.left.0 } else { p.right.0 }))
}

/// Marg-canonical no-re-expand rule: re-resolve a swapped-in parent level's
/// marginal refs from a SOURCE child store-space into the OUTPUT child
/// store-space.
///
/// The apply engine's identity fast paths — where one operand is constant-true
/// at this subtree — `mem::swap` a parent level out of an operand's store into
/// the apply output. A bare ref
/// (bit-30 clear) on a marginal-child side is a *store-relative* slot index into
/// the operand's child `marginal_counts`; after the swap it must point into the
/// OUTPUT child store (`levels[ci]`) instead. For each bare slot ref: read the
/// source count, then either inline it (≤ `MARG_INLINE_MAX` ⇒ store-independent,
/// bit-30 set) or re-mint a fresh slot in the output child store (recording the
/// exact value under the new slot key in [`BigSide`] when it overflowed). Inline
/// refs (bit-30 set) and ZERO sentinels (bit-31) are store-independent and pass
/// through untouched.
///
/// `ti` (swapped-in parent, in `levels`) and `ci` (output child, in `levels`) are
/// distinct; `src_child` is the operand's corresponding child level (a *different*
/// `Tdd`'s store), borrowed immutably.
///
/// # Errors
///
/// `Err(ApplyError::OverBudget)` when an interner entry or the destination
/// store's growth cannot be reserved. Every allocation is front-loaded by the
/// pre-scan BEFORE the rewrite touches a single ref, and the rewrite pass is
/// infallible by construction — so an over-budget swap leaves the TDD exactly as
/// it found it. A half-remapped level would not merely be large: its unrewritten
/// refs still index the SOURCE store, which miscounts silently.
pub(crate) fn resolve_swapped_marg_side(
    levels: &mut [TddLevel],
    ti: usize,
    ci: usize,
    src_child: &TddLevel,
    is_left: bool,
) -> Result<(), ApplyError> {
    debug_assert_ne!(ti, ci);
    // WEIGHTED: nothing to re-resolve, by construction. The whole
    // remap exists because the integer marginal store is PER-`Tdd`, so a swapped-in
    // parent's bare slot refs are relative to the operand's store and must be
    // re-minted into the output's. The weighted store is not per-`Tdd`: the two
    // operands' stores are merged into the output's, so the source child level and
    // the output child level at `ci` share the very same column and a
    // bare slot ref is already in the destination store-space. Returning here is
    // the correct no-op — and the only correct action, since a weight-marginal
    // level's `marginal_counts` is `None` (the two representations are mutually
    // exclusive within a compile) and both `expect`s below would fire.
    if src_child.is_weight_marginal() || levels[ci].is_weight_marginal() {
        debug_assert!(
            src_child.is_weight_marginal() && levels[ci].is_weight_marginal(),
            "mixed marginal representations at level {ci}: integer on one side, \
             weighted on the other"
        );
        return Ok(());
    }
    let src_counts = src_child
        .marginal_counts
        .as_deref()
        .expect("resolve_swapped_marg_side: src child missing marginal_counts");
    let src_big = src_child.marginal_counts_big.as_ref();
    // Read the inline threshold ONCE, not per ref: the pre-scan and the rewrite
    // must classify every ref identically, and the test-only override backing
    // `marg_inline_max` is a thread-local cell a re-read could observe changed.
    let inline_max = marg_inline_max() as u128;
    // Disjoint &mut borrows of the parent (ti) and output child (ci) levels.
    let (parent, dst_child) = if ti < ci {
        let (a, b) = levels.split_at_mut(ci);
        (&mut a[ti], &mut b[0])
    } else {
        let (a, b) = levels.split_at_mut(ti);
        (&mut b[0], &mut a[ci])
    };
    // The destination side table is SPARSE (`BigSide`), so it needs no
    // pre-alignment to the destination store's width — a re-minted overflow
    // slot simply records its own key. Disjoint field borrows of `dst_child`.
    let dst_big = &mut dst_child.marginal_counts_big;
    let dst_counts = dst_child
        .marginal_counts
        .as_mut()
        .expect("resolve_swapped_marg_side: dst child missing marginal_counts");

    let src = SwapSource { counts: src_counts, big: src_big, inline_max };
    let mut interners = collect_swap_mints(parent, is_left, &src)?;
    reserve_and_seed_dst(&mut interners, dst_counts, dst_big)?;
    rewrite_swapped_refs(parent, is_left, &src, &mut interners, dst_counts, dst_big);
    Ok(())
}

/// The source child's count store and the inline threshold, read once and
/// carried through both passes so they classify every ref identically.
struct SwapSource<'a> {
    counts: &'a [u128],
    big: Option<&'a BigSide>,
    inline_max: u128,
}

/// The destination slot each mintable source count will use: `SLOT_UNSEEDED`
/// until a dst slot is found or minted for it.
struct SwapInterners {
    small: FxHashMap<u128, u32>,
    big: FxHashMap<BigUint, u32>,
    /// OVERFLOW-sentinel source slots carrying no exact value to key on. The
    /// rewrite's debug_assert rejects them; in release each one pushes its own
    /// dst slot and cannot dedup, so each needs its own reservation.
    orphan_overflow: usize,
}

/// Pre-scan: intern the counts that actually need a destination slot.
///
/// Only a `Mint` ref reaches the dst store at all — `Keep` and `Inline` refs
/// are store-independent — so a swap carrying none of them needs no interner,
/// no store growth and no side table, and must allocate NOTHING. The
/// interners are therefore keyed by what this scan finds (bounded by the
/// parent's ref count), not seeded from the whole dst store: the store-sized
/// seed cost ~1.5-2× the store in hash entries plus one `BigUint` clone per
/// dst overflow slot, built before knowing whether one ref needed re-minting.
///
/// The store is born free of duplicate count values; enforced here, not by a later
/// canon pass. Key: `u128` for above-threshold counts, `BigUint` for
/// OVERFLOW-sentinel counts (so two numerically equal BigUints share one dst
/// slot). Counts ≤ `inline_max` ride inline at the ref and never become
/// slots, so they need no entry.
///
/// # Errors
///
/// `Err(ApplyError::OverBudget)` when an interner entry cannot be reserved.
fn collect_swap_mints(
    parent: &TddLevel,
    is_left: bool,
    src: &SwapSource<'_>,
) -> Result<SwapInterners, ApplyError> {
    let mut small_to_slot: FxHashMap<u128, u32> = FxHashMap::default();
    let mut big_to_slot: FxHashMap<BigUint, u32> = FxHashMap::default();
    let mut orphan_overflow = 0usize;
    for raw in marg_side_refs(parent, is_left) {
        let SwapRef::Mint(s, c) = classify_swap_ref(raw, src.counts, src.inline_max) else {
            continue;
        };
        if c == u128::MAX {
            match src.big.and_then(|sb| sb.get(s)) {
                Some(b) if !big_to_slot.contains_key(b) => {
                    big_to_slot.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
                    big_to_slot.insert(b.clone(), SLOT_UNSEEDED);
                }
                Some(_) => {} // key already interned by an earlier ref
                None => orphan_overflow += 1,
            }
        } else if !small_to_slot.contains_key(&c) {
            small_to_slot.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
            small_to_slot.insert(c, SLOT_UNSEEDED);
        }
    }
    Ok(SwapInterners { small: small_to_slot, big: big_to_slot, orphan_overflow })
}

/// Front-load every allocation the rewrite can need, then point each interned
/// count at a dst slot that already carries it.
///
/// # Errors
///
/// `Err(ApplyError::OverBudget)` when the destination store's growth or the
/// side table cannot be reserved.
fn reserve_and_seed_dst(
    interners: &mut SwapInterners,
    dst_counts: &mut Vec<u128>,
    dst_big: &mut Option<BigSide>,
) -> Result<(), ApplyError> {
    use crate::tdd::counts::{ApplyBudget, ReservePolicy};

    // Upper bound on the slots the rewrite can mint: one per distinct interned
    // count (a second ref carrying it dedups onto the first) plus one per
    // un-keyable overflow.
    let new_slots = interners.small.len() + interners.big.len() + interners.orphan_overflow;
    if new_slots == 0 {
        return Ok(());
    }
    // The rewrite must not fail part-way (see `resolve_swapped_marg_side`'s
    // error contract). `reserve` (doubling) matches the growth the `push`es
    // would have taken on their own, and routes through the ONE apply budget
    // accounting path.
    ApplyBudget::reserve(dst_counts, new_slots)?;
    if !interners.big.is_empty() {
        // A keyed overflow mint always lands here, so the table exists by
        // the time the rewrite inserts into it. Creating it is not a new
        // side effect: with no table there is nothing for an overflow ref to
        // dedup against, so the first such ref minted — and created it —
        // before this change too.
        dst_big
            .get_or_insert_with(BigSide::default)
            .try_reserve::<ApplyBudget>(interners.big.len())?;
    }
    // Seed the interners from the dst slots already carrying a wanted count,
    // so an equal count reuses its slot instead of pushing a duplicate.
    // Lowest index wins. Slots holding a count nothing asked for are skipped —
    // that is what keeps the interners sized by the scan rather than by the store.
    for i in 0..dst_counts.len() {
        let c = dst_counts[i];
        if c == u128::MAX {
            if let Some(b) = dst_big.as_ref().and_then(|b| b.get(i)) {
                if let Some(slot) = interners.big.get_mut(b) {
                    if *slot == SLOT_UNSEEDED {
                        *slot = i as u32;
                    }
                }
            }
        } else if let Some(slot) = interners.small.get_mut(&c) {
            if *slot == SLOT_UNSEEDED {
                *slot = i as u32;
            }
        }
    }
    Ok(())
}

/// Re-resolve one marg-side ref into the destination store space, minting a
/// slot for it when no interned one carries its count yet.
///
/// INFALLIBLE: every push has reserved capacity from [`reserve_and_seed_dst`].
fn remap_swap_ref(
    raw: u32,
    src: &SwapSource<'_>,
    interners: &mut SwapInterners,
    dst_counts: &mut Vec<u128>,
    dst_big: &mut Option<BigSide>,
) -> u32 {
    let (s, c) = match classify_swap_ref(raw, src.counts, src.inline_max) {
        // ZERO sentinel or already-inline count: store-independent.
        SwapRef::Keep => return raw,
        // Small enough to carry in the ref (bit-30 set): store-independent.
        SwapRef::Inline(c) => return MargRef::Inline(c).to_raw(),
        SwapRef::Mint(s, c) => (s, c),
    };
    // Big (`u128::MAX` sentinel) or large-but-u128 count: re-mint into dst store,
    // deduplicating via the inline interner so equal large counts share one slot.
    if c == u128::MAX {
        // BigUint path: look up in the big interner first.
        let big_val = src.big.and_then(|sb| sb.get(s));
        debug_assert!(
            big_val.is_some(),
            "resolve_swapped_marg_side: src slot {s} is the u128::MAX sentinel \
             but has no BigUint entry"
        );
        if let Some(b) = big_val {
            match interners.big.get(b) {
                Some(&existing) if existing != SLOT_UNSEEDED => {
                    return MargRef::slot_raw(existing);
                }
                _ => {}
            }
        }
        let new_idx = dst_counts.len() as u32;
        dst_counts.push(u128::MAX);
        if let Some(b) = big_val {
            dst_big
                .as_mut()
                .expect("keyed overflow ⇒ pre-scan created the side table")
                .insert(new_idx as usize, b.clone());
            // The key was interned by the pre-scan; record the slot it just got.
            if let Some(slot) = interners.big.get_mut(b) {
                *slot = new_idx;
            }
        }
        MargRef::slot_raw(new_idx)
    } else {
        // Small (but above inline threshold) count: look up in the small interner.
        // Nothing to mirror into the sparse side table — a non-overflow slot
        // simply has no entry there.
        match interners.small.get(&c) {
            Some(&existing) if existing != SLOT_UNSEEDED => {
                return MargRef::slot_raw(existing);
            }
            _ => {}
        }
        let new_idx = dst_counts.len() as u32;
        dst_counts.push(c);
        if let Some(slot) = interners.small.get_mut(&c) {
            *slot = new_idx;
        }
        MargRef::slot_raw(new_idx)
    }
}

/// Rewrite every marg-side ref of the swapped-in parent, in the order
/// [`marg_side_refs`] reports them.
fn rewrite_swapped_refs(
    parent: &mut TddLevel,
    is_left: bool,
    src: &SwapSource<'_>,
    interners: &mut SwapInterners,
    dst_counts: &mut Vec<u128>,
    dst_big: &mut Option<BigSide>,
) {
    for node in &mut parent.nodes {
        if node.is_inline() {
            if is_left {
                node.a = remap_swap_ref(node.a, src, interners, dst_counts, dst_big);
            } else {
                node.b = remap_swap_ref(node.b, src, interners, dst_counts, dst_big);
            }
        }
    }
    for p in &mut parent.pairs {
        if is_left {
            p.left.0 = remap_swap_ref(p.left.0, src, interners, dst_counts, dst_big);
        } else {
            p.right.0 = remap_swap_ref(p.right.0, src, interners, dst_counts, dst_big);
        }
    }
}


/// Direct contract tests for [`resolve_swapped_marg_side`]. Integration-level
/// coverage cannot discriminate this fixup: across the benchmark instances
/// that both solve and traverse it, and across the full
/// test suite's in-process traversals, no-op'ing the function changes no
/// count — the identity-fast-path chains that trigger it produce output child
/// stores content-identical to the source store, so the remap is semantically
/// idempotent there. The fixup is load-bearing exactly when the stores
/// DIVERGE (different slot order / absent counts / different lengths), which
/// these tests construct directly.
#[cfg(test)]
#[path = "marg_resolve_swap_tests.rs"]
mod resolve_swap_tests;
