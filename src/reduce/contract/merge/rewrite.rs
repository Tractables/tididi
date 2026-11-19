//! Rewriting the parent level's refs onto the surviving twins.

use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildPair, NodeIdx, StoreRoom, Tdd, TddLevel};

use super::super::scratch::MergeRemap;
#[cfg(all(target_arch = "x86_64", not(miri)))]
use crate::isa::Avx2;
use crate::isa::avx2_fn;

/// Set in a merged-away node's `final_remap` entry, above its survivor's
/// index before the merge, where the parent rewrite drops the pairs naming
/// it: every member but a content-equal one redirected onto its survivor.
/// The parent rewrite reads one entry per pair and needs only the bit;
/// readers that want the survivor's new index read it off the survivor's
/// entry ([`survivor`]). A stored side never has bit 31 set, so no node
/// index reaches it.
pub(in crate::reduce::contract) const DROPPED: u32 = 1 << 31;

/// The `final_remap` entry of a node the merge drops with its pairs, no
/// pair naming it: it has no survivor.
pub(in crate::reduce::contract) const DISCARDED: u32 = u32::MAX;

/// The new index of the survivor of node `old` after the merge: its own
/// entry, or, for a member whose pairs were dropped, its survivor's.
#[inline]
pub(in crate::reduce::contract) fn survivor(final_remap: &[NodeIdx], old: usize) -> NodeIdx {
    let to = final_remap[old].0;
    debug_assert_ne!(to, DISCARDED, "a node the merge drops as unnamed has no survivor");
    if to & DROPPED == 0 { NodeIdx(to) } else { final_remap[(to & !DROPPED) as usize] }
}

/// Filter out parent pairs referencing non-canonical twins and apply the
/// gap-closing remap.
///
/// Twins have identical parent contexts, so every pair referencing a
/// non-canonical twin has a duplicate referencing the canonical one; dropping
/// the non-canonical pairs is an O(n) filter keyed on which twin a pair
/// references, whatever order the pairs are in. t1 is never marginal here (see
/// `contract_twins`), so the t1-side refs are plain node indices that
/// `final_remap` indexes directly.
///
/// Infallible: every allocation it could need was reserved before the pass
/// mutated anything (`room`, the stored arena of an implicit parent).
///
/// `ordered` says the parent is known to hold its ranges in node order,
/// which is then not read again. Returns whether the parent is left so,
/// and its live pairs, counted as they are written: the merge leaves them
/// for the close.
pub(super) fn rewrite_parent(
    tdd: &mut Tdd,
    parent: VtreeIdx,
    t1_side: ChildSide,
    remap: &MergeRemap,
    room: Option<StoreRoom>,
    ordered: bool,
) -> (bool, usize) {
    // Arena garbage from the whole rewrite, accumulated and noted in one charge
    // below: the only reader (`compact_pairs_if_stale`) runs after the loop, so a
    // per-iteration saturating add buys nothing.
    let mut dead_acc = 0usize;
    let parent_level = &mut tdd.levels[parent.idx()];
    if parent_level.implicit().is_some() {
        parent_level.store_moved(room.expect("an implicit parent's room is reserved"), |_| true, |l| l, |r| r);
    }
    // Where the arena holds the ranges in node order, each node's kept pairs
    // move down to a write cursor as they are rewritten, which leaves the
    // arena with no dead pairs and the parent no sweep to run; otherwise
    // each node shrinks in place and its tail is left dead. The side is a
    // constant of each slide's loop.
    debug_assert!(!ordered || parent_level.ranges_in_node_order(), "a parent known in node order is so");
    if ordered || parent_level.ranges_in_node_order() {
        let final_remap = &remap.final_remap;
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        if let Some(avx2) = Avx2::detect() {
            // Safety: an `Avx2` exists only where the processor has the
            // features `remap_avx2` enables.
            let live = match t1_side {
                ChildSide::Left => parent_level.slide_pairs(|seg, from| match seg.len() - from < 4 {
                    true => remap_pairs::<true>(seg, from, 0, final_remap),
                    false => unsafe { remap_avx2::<true>(avx2, seg, from, final_remap) },
                }),
                ChildSide::Right => parent_level.slide_pairs(|seg, from| match seg.len() - from < 4 {
                    true => remap_pairs::<false>(seg, from, 0, final_remap),
                    false => unsafe { remap_avx2::<false>(avx2, seg, from, final_remap) },
                }),
            };
            return (true, live);
        }
        let live = match t1_side {
            ChildSide::Left => parent_level.slide_pairs(|seg, from| remap_pairs::<true>(seg, from, 0, final_remap)),
            ChildSide::Right => parent_level.slide_pairs(|seg, from| remap_pairs::<false>(seg, from, 0, final_remap)),
        };
        return (true, live);
    }
    let mut live = 0usize;
    for node_idx in 0..parent_level.nodes().len() {
        let kind = parent_level.node(node_idx).kind();
        let Some(range) = parent_level.arena_range(kind) else {
            remap_inline_node(parent_level, node_idx, t1_side, &remap.final_remap);
            live += 1;
            continue;
        };
        let new_len = keep_canonical_pairs(parent_level, node_idx, t1_side, remap);
        live += new_len;
        if new_len < range.len() {
            dead_acc += parent_level.reencode_shrunk(node_idx, range.start, range.len(), new_len);
        }
    }
    parent_level.note_dead_pairs(dead_acc);
    (false, live)
}

/// Point an inline node's single pair at the survivor of its T1-side twin
/// class. By the twin invariant that pair always references a canonical node,
/// so it is never dropped — only remapped, and the node stays inline.
fn remap_inline_node(
    level: &mut TddLevel,
    node_idx: usize,
    t1_side: ChildSide,
    final_remap: &[NodeIdx],
) {
    let node = &mut level.nodes.stored_mut()[node_idx];
    let tv_old = if t1_side == ChildSide::Left { node.a } else { node.b };
    let tv_new = survivor(final_remap, tv_old as usize).0;
    if t1_side == ChildSide::Left {
        node.a = tv_new;
    } else {
        node.b = tv_new;
    }
}

/// Compact a multi-pair node in place, keeping the pairs whose T1-side ref
/// survives the merge and remapping each onto its survivor. Returns how many
/// pairs are left; the node's recorded length is not touched.
///
/// A duplicate redirected content-equal twin is kept, not dropped: remapping it onto
/// the survivor mints a duplicate `(survivor, marginal)` pair whose count pair
/// fusion sums. That is also how a fusion redex is minted here — two pairs at
/// one node sharing an explicit-side ref with distinct marginal-side refs.
fn keep_canonical_pairs(
    level: &mut TddLevel,
    node_idx: usize,
    t1_side: ChildSide,
    remap: &MergeRemap,
) -> usize {
    let pairs = level.pairs_mut(node_idx);
    let mut write = 0;
    for read in 0..pairs.len() {
        let (pair, kept) = canonical_pair_on(pairs[read], t1_side, &remap.final_remap);
        pairs[write] = pair;
        write += usize::from(kept);
    }
    write
}

/// A parent pair after the merge, remapped onto its T1-side ref's survivor,
/// and whether it is kept: where that ref survives the merge or was
/// redirected as a content-equal duplicate. A pair not kept duplicates one
/// naming the survivor. The callers write every pair at their write cursor
/// and advance it by the flag, with no branch on it: which pairs are kept
/// follows no pattern a branch predictor learns. `LEFT` says whether T1 is
/// the left child.
#[inline(always)]
fn canonical_pair<const LEFT: bool>(mut pair: ChildPair, final_remap: &[NodeIdx]) -> (ChildPair, bool) {
    let f = if LEFT { &mut pair.left } else { &mut pair.right };
    let to = final_remap[f.0 as usize].0;
    debug_assert_ne!(to, DISCARDED, "a pair names a node the merge drops as unnamed");
    *f = NodeIdx(to & !DROPPED).into();
    (pair, to & DROPPED == 0)
}

/// Rewrite `seg[read..]` through [`canonical_pair`], writing the kept pairs
/// in order from `seg[kept]`, and return the kept count; `kept` is at most
/// `read`.
#[inline(always)]
pub(super) fn remap_pairs<const LEFT: bool>(seg: &mut [ChildPair], read: usize, mut kept: usize, final_remap: &[NodeIdx]) -> usize {
    debug_assert!(kept <= read);
    for r in read..seg.len() {
        let (pair, k) = canonical_pair::<LEFT>(seg[r], final_remap);
        // Safety: `kept` grows by at most one a pair read, so it stays at
        // most `r`.
        unsafe { *seg.get_unchecked_mut(kept) = pair };
        kept += usize::from(k);
    }
    kept
}

/// For each of the 16 keep masks of four pairs, the 32-bit lanes that move
/// the kept pairs, in order, to the front of a vector.
#[cfg(all(target_arch = "x86_64", not(miri)))]
const KEEP_LANES: [[i32; 8]; 16] = {
    let mut lanes = [[0i32; 8]; 16];
    let mut mask = 0;
    while mask < 16 {
        let (mut pair, mut to) = (0, 0);
        while pair < 4 {
            if (mask >> pair) & 1 == 1 {
                lanes[mask][2 * to] = 2 * pair;
                lanes[mask][2 * to + 1] = 2 * pair + 1;
                to += 1;
            }
            pair += 1;
        }
        mask += 1;
    }
    lanes
};

avx2_fn! {
    /// [`remap_pairs`] from `seg[from]` with the kept count at zero, four
    /// pairs at a time: their T1-side refs gathered from `final_remap`,
    /// the kept ones moved to the front of the vector and the vector
    /// stored at the write cursor, which is at most the read cursor, so
    /// the store overwrites no pair not yet read. Four pairs with a ref
    /// past `final_remap`, and the last pairs, are left to `remap_pairs`.
    #[inline(never)]
    pub(super) fn remap_avx2<const LEFT: bool>(
        _: Avx2,
        seg: &mut [ChildPair],
        from: usize,
        final_remap: &[NodeIdx],
    ) -> usize {
        use core::arch::x86_64::*;
        let n = seg.len();
        let base = seg.as_mut_ptr();
        let remap = final_remap.as_ptr().cast::<i32>();
        let low = _mm256_set1_epi64x(0xFFFF_FFFF);
        let bound = _mm256_set1_epi64x(final_remap.len() as i64);
        let unflag = _mm_set1_epi32(!DROPPED as i32);
        let (mut read, mut kept) = (from, 0usize);
        while read + 4 <= n {
            // Safety: `read + 4 <= n`, and `ChildPair` is two `u32`s
            // (`repr(C)`), left first.
            let v = unsafe { _mm256_loadu_si256(base.add(read).cast()) };
            let refs = if LEFT { _mm256_and_si256(v, low) } else { _mm256_srli_epi64::<32>(v) };
            if _mm256_movemask_epi8(_mm256_cmpgt_epi64(bound, refs)) != -1 {
                break;
            }
            // Safety: every ref is below `final_remap.len()`, checked above.
            let to = unsafe { _mm256_i64gather_epi32::<4>(remap, refs) };
            let keep = !_mm_movemask_ps(_mm_castsi128_ps(to)) as usize & 0xF;
            let to = _mm256_cvtepu32_epi64(_mm_and_si128(to, unflag));
            let pairs = if LEFT {
                _mm256_or_si256(_mm256_andnot_si256(low, v), to)
            } else {
                _mm256_or_si256(_mm256_and_si256(v, low), _mm256_slli_epi64::<32>(to))
            };
            // Safety: `KEEP_LANES[keep]` is eight `i32`s.
            let lanes = unsafe { _mm256_loadu_si256(KEEP_LANES[keep].as_ptr().cast()) };
            // Safety: `kept <= read`, so the four pairs stored lie within
            // `seg[..read + 4]`.
            unsafe {
                _mm256_storeu_si256(
                    base.add(kept).cast(),
                    _mm256_permutevar8x32_epi32(pairs, lanes),
                );
            }
            kept += keep.count_ones() as usize;
            read += 4;
        }
        remap_pairs::<LEFT>(seg, read, kept, final_remap)
    }
}

/// [`canonical_pair`] with T1's side read at run time.
#[inline]
fn canonical_pair_on(pair: ChildPair, t1_side: ChildSide, final_remap: &[NodeIdx]) -> (ChildPair, bool) {
    match t1_side {
        ChildSide::Left => canonical_pair::<true>(pair, final_remap),
        ChildSide::Right => canonical_pair::<false>(pair, final_remap),
    }
}
