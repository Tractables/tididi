//! Rebuild the two diagram levels affected by a vtree rotation.
//!
//! The outer level retains its node count and ordering, so references from
//! higher levels remain valid. Other levels keep their original pairs.
//! Rebuilding expands each outer pair into triples and regroups them:
//!
//! - Left: `(A, (B, C))` becomes `((A, B), C)`. Group triples by `C` and
//!   intern the resulting sets of `(A, B)` pairs at the new inner level.
//! - Right: `((A, B), C)` becomes `(A, (B, C))`. Group by `A` and intern
//!   the sets of `(B, C)` pairs instead.
//!
//! ## Marginal context
//!
//! Without marginal levels, equal cells share an inner node and duplicate
//! outer pairs can be removed. With marginal levels, identical products may
//! represent different assignment families whose values must add. The rewrite
//! then retains one inner node per distinct inner pair and preserves both
//! cell and outer-pair multiplicities. Regrouping preserves their value sum.

use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, NodeIdx, Tdd, TddLevel};


use crate::vtree::rotate::RotationInfo;
use crate::vtree::RotationKind;

// The `marginal_ctx` branches below expand fully instead of sharing and
// deduping; the argument is the module doc's "Marginal context" section.

use super::scratch::{BucketScratch, RestructureScratch, release_or_clear};
use crate::limits::{Limits, OperationError, Transient};
use crate::sort::{Radix, RADIX_MIN_ROWS};

/// Where the four fields of a packed search triple sit: `(inner.left,
/// inner.right, axis, src)` from the highest bits down, field `k` taking
/// `width[k]` bits from `shift[k]` up. The numeric order of the packed words
/// is the lexicographic order of the fields, so sorting them groups the cells
/// by their inner pair and orders a group's cells by `(axis, src)`, which is
/// what the group scan below relies on, as one integer compare instead of a
/// four-field branchy tuple compare. The high fields are the `inner` pair (its
/// own sort key); the low ones are the `(axis, src)` "cell". `src` is the
/// lowest field because the triples are collected in ascending order of it,
/// which the sort then need not place ([`Word::sort`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct Layout {
    shift: [u32; 4],
    width: [u32; 4],
}

impl Layout {
    /// Every field at 32 bits: the layout of a [`u128`] word.
    pub(super) const WIDE: Layout = Layout { shift: [96, 64, 32, 0], width: [32; 4] };

    /// Each field at the bits that `bound`, which its values do not exceed
    /// bit for bit, needs.
    pub(super) fn fitted(bound: [u32; 4]) -> Self {
        let width = bound.map(|x| 32 - x.leading_zeros());
        Layout { shift: [width[1] + width[2] + width[3], width[2] + width[3], width[3], 0], width }
    }

    fn bits(&self) -> u32 {
        self.width.iter().sum()
    }
}

/// A packed search triple: a [`u64`] where the fitted fields fit it, which is
/// the usual case, and a [`u128`] at [`Layout::WIDE`] where they do not.
pub(super) trait Word: Copy + Ord + Default {
    fn pack(layout: &Layout, inner: ChildPair, src: u32, axis: EncodedChildRef) -> Self;
    /// Field `k` (0 to 3) under `layout`.
    fn field(self, layout: &Layout, k: usize) -> u32;
    /// The bits of the inner pair, fields 0 and 1.
    fn inner_key(self, layout: &Layout) -> u64;
    /// The bits of the cell, fields 2 and 3: the fingerprint and dedup key.
    fn cell(self, layout: &Layout) -> u64;
    /// Phase 2's sort: the triples, in ascending order of `src` as
    /// [`collect_triples`] leaves them, into ascending order.
    ///
    /// The sort is not charged to the work clock: the probe charges the pairs
    /// it rebuilds, and the pool search's work bound is set in that measure.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] if the sort's buffers are refused.
    fn sort(lim: &Limits, layout: &Layout, triples: &mut Vec<Self>) -> Result<(), OperationError>;
}

impl Word for u64 {
    #[inline]
    fn pack(layout: &Layout, inner: ChildPair, src: u32, axis: EncodedChildRef) -> Self {
        // A field of no bits may sit at bit 64, past a plain shift.
        let at = |x: u32, k: usize| u64::from(x).checked_shl(layout.shift[k]).unwrap_or(0);
        at(inner.left.0, 0) | at(inner.right.0, 1) | at(axis.0, 2) | u64::from(src)
    }
    #[inline]
    fn field(self, layout: &Layout, k: usize) -> u32 {
        (self.checked_shr(layout.shift[k]).unwrap_or(0) & !u64::MAX.checked_shl(layout.width[k]).unwrap_or(0)) as u32
    }
    #[inline]
    fn inner_key(self, layout: &Layout) -> u64 {
        self.checked_shr(layout.shift[1]).unwrap_or(0)
    }
    #[inline]
    fn cell(self, layout: &Layout) -> u64 {
        self & !u64::MAX.checked_shl(layout.shift[1]).unwrap_or(0)
    }
    /// A radix sort ([`Radix::sort`]) of the bits above `src`, or a
    /// comparison sort below [`RADIX_MIN_ROWS`].
    fn sort(lim: &Limits, layout: &Layout, triples: &mut Vec<Self>) -> Result<(), OperationError> {
        if triples.len() < RADIX_MIN_ROWS {
            triples.sort_unstable();
            return Ok(());
        }
        debug_assert!(triples.windows(2).all(|w| w[0].field(layout, 3) <= w[1].field(layout, 3)), "the triples come in ascending order of their src");
        let src = layout.width[3];
        let mut radix = Radix::default();
        let sorted = radix.sort_polling(lim, triples, src as usize, (layout.bits() - src) as usize, &mut |_| Ok(()));
        radix.discard(lim);
        sorted
    }
}

impl Word for u128 {
    #[inline]
    fn pack(layout: &Layout, inner: ChildPair, src: u32, axis: EncodedChildRef) -> Self {
        debug_assert_eq!(layout.width, Layout::WIDE.width, "a u128 triple takes the wide layout");
        (u128::from(inner.left.0) << 96) | (u128::from(inner.right.0) << 64) | (u128::from(axis.0) << 32) | u128::from(src)
    }
    #[inline]
    fn field(self, _: &Layout, k: usize) -> u32 {
        (self >> (96 - 32 * k)) as u32
    }
    #[inline]
    fn inner_key(self, _: &Layout) -> u64 {
        (self >> 64) as u64
    }
    #[inline]
    fn cell(self, _: &Layout) -> u64 {
        self as u64
    }
    /// A comparison sort: the wide layout is taken only where the fields do
    /// not fit one word between them.
    fn sort(_: &Limits, _: &Layout, triples: &mut Vec<Self>) -> Result<(), OperationError> {
        triples.sort_unstable();
        Ok(())
    }
}

#[inline]
fn tri_inner<W: Word>(p: W, layout: &Layout) -> ChildPair {
    ChildPair::new(EncodedChildRef::from_raw(p.field(layout, 0)), EncodedChildRef::from_raw(p.field(layout, 1)))
}
#[inline]
fn tri_src<W: Word>(p: W, layout: &Layout) -> u32 {
    p.field(layout, 3)
}
#[inline]
fn tri_axis<W: Word>(p: W, layout: &Layout) -> EncodedChildRef {
    EncodedChildRef::from_raw(p.field(layout, 2))
}

/// Rebuild the two levels of a rotation in `dir` and return the levels the
/// rotation replaced, or `None` if the probe was abandoned.
///
/// Cells are grouped by sorting the inner pairs and scanning the runs, which
/// keeps the probe free of per-pair allocations.
///
/// The two old levels are read in place and only replaced once both new ones
/// are built, so `Ok(None)` and every error leave the diagram byte-for-byte
/// what it was on entry and the caller may probe the next candidate without
/// any undo of its own. The rebuild returns `Ok(None)` when the rotation would
/// exceed `max_pairs`, or when a bail check shows it cannot produce a
/// well-formed pair of levels.
///
/// `dir` says whether the rotation promoted `w` from v's right child (a left
/// rotation) or its left child (a right rotation), which fixes the geometry
/// of the triple expansion.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a buffer the rebuild needs is refused,
/// either by the allocator or by the armed byte budget. Every buffer the
/// expansion grows, the pair table included, is charged through the engine's
/// limits, so a rotation too wide for the host comes back as an answer rather
/// than an abort.
pub(crate) fn rebuild_rotated_levels(
    lim: &Limits,
    tdd: &mut Tdd,
    info: &RotationInfo,
    dir: RotationKind,
    scratch: &mut RestructureScratch,
    max_pairs: usize,
) -> Result<Option<(TddLevel, TddLevel)>, OperationError> {
    let marginal = tdd.has_marginal_level();
    rebuild_levels(lim, tdd, info, dir, false, marginal, scratch, max_pairs)
}

/// [`rebuild_rotated_levels`], given whether `tdd` has a marginal level
/// (`marginal`). A caller that rebuilds many diagrams knows that once:
/// each rebuild forgets what the diagram knows of its levels, and asking
/// again reads every level.
///
/// Where `crossed`, the rotation's promoted node had its two children
/// swapped first ([`swap_children`](crate::vtree::rotate::swap_children)):
/// the old w-level is read with its sides exchanged, so the grouping the
/// rotation makes pairs the other grandchild with `v`'s other child.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rebuild_levels(
    lim: &Limits,
    tdd: &mut Tdd,
    info: &RotationInfo,
    dir: RotationKind,
    crossed: bool,
    marginal: bool,
    scratch: &mut RestructureScratch,
    max_pairs: usize,
) -> Result<Option<(TddLevel, TddLevel)>, OperationError> {
    debug_assert_eq!(marginal, tdd.has_marginal_level());
    let (old_v, old_w) = (&tdd.levels[info.v_idx.idx()], &tdd.levels[info.w_idx.idx()]);
    // The triples are packed one per word where their fields fit one: a
    // fitted `u64` moves half the bytes of a `u128` through every phase.
    let layout = Layout::fitted(field_bounds(old_v, old_w, dir, crossed));
    if layout.bits() <= 64 {
        let mut triples = std::mem::take(&mut scratch.narrow);
        let rebuilt = rebuild_with(lim, tdd, info, dir, crossed, marginal, &layout, &mut triples, scratch, max_pairs);
        scratch.narrow = triples;
        rebuilt
    } else {
        let mut triples = std::mem::take(&mut scratch.wide);
        let rebuilt = rebuild_with(lim, tdd, info, dir, crossed, marginal, &Layout::WIDE, &mut triples, scratch, max_pairs);
        scratch.wide = triples;
        rebuilt
    }
}

/// [`rebuild_levels`] with the triples packed as `W` under `layout`, in
/// `triples`, which is left empty.
#[allow(clippy::too_many_arguments)]
fn rebuild_with<W: Word>(
    lim: &Limits,
    tdd: &mut Tdd,
    info: &RotationInfo,
    dir: RotationKind,
    crossed: bool,
    marginal_ctx: bool,
    layout: &Layout,
    triples: &mut Vec<W>,
    scratch: &mut RestructureScratch,
    max_pairs: usize,
) -> Result<Option<(TddLevel, TddLevel)>, OperationError> {
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    // The group table indexes `triples` with `u32` offsets; `collect_triples`
    // gives up once the count reaches `max_pairs`.
    let max_pairs = max_pairs.min(u32::MAX as usize);
    // Read in place: nothing leaves the diagram until both new levels exist, so
    // an early exit has nothing to undo.
    let (old_v, old_w) = (&tdd.levels[v_idx], &tdd.levels[w_idx]);

    triples.clear();
    if !collect_triples(lim, old_v, old_w, dir, crossed, layout, triples, max_pairs)? {
        triples.clear();
        return Ok(None);
    }

    // Phase 2: sort the packed triples. After sorting, cells for each inner
    // pair are contiguous and sorted — no per-group sort needed.
    W::sort(lim, layout, triples)?;

    // `group_info` addresses `triples` with u32 offsets. The u32 width of a
    // `NodeIdx` bounds node indices, not this arena-scale offset: past 2^32
    // triples the `as u32` casts below would wrap, `same_cells` would compare
    // wrong-but-in-range cell slices, and the resulting inner-node sharing would
    // silently change the count. `write <= read <= n`, so this single check
    // covers every cast in the scan.
    let n = triples.len();
    debug_assert!(
        u32::try_from(n).is_ok(),
        "rotation restructure: {n} triples exceeds the u32 group offsets into `triples`",
    );

    // In marginal context (full expansion) the cell multiset is kept: a duplicate
    // (axis, src) cell is a legitimate separate count-contribution (two
    // marginalization-collapsed twin primes), so cell-deduping it would drop
    // count-mass. Boolean mode dedups.
    scratch.group_info.clear();
    group_by_inner_pair(lim, triples, layout, &mut scratch.group_info, marginal_ctx)?;

    let Some(inner_level) = build_inner_level(
        lim,
        triples,
        layout,
        &mut scratch.group_info,
        &mut scratch.bucket,
        marginal_ctx,
        max_pairs,
    )?
    else {
        triples.clear();
        return Ok(None);
    };

    // Neither level is installed until both are built; a refusal in between
    // drops the inner one and hands its charge back.
    let inner_level = Transient::new(lim, inner_level);
    let outer_level = build_outer_level(lim, old_v, triples, layout, scratch, dir, marginal_ctx)?;

    Ok(Some(tdd.replace_level_pair(
        (info.v_idx, outer_level),
        (info.w_idx, inner_level.keep()),
    )))
}

/// Bounds on the four fields of a rotation's triples, bit for bit: every
/// value a field takes has no bit its bound lacks. The fields are read from
/// the two old levels' pairs, which are fewer than the triples they expand
/// into.
fn field_bounds(old_v_level: &TddLevel, old_w_level: &TddLevel, dir: RotationKind, crossed: bool) -> [u32; 4] {
    let mut v_axis = 0u32;
    for i in 0..old_v_level.nodes().len() {
        for vp in old_v_level.pairs_iter_of_idx(i) {
            v_axis |= match dir {
                RotationKind::Left => vp.left.0,
                RotationKind::Right => vp.right.0,
            };
        }
    }
    let (mut stored_left, mut stored_right) = (0u32, 0u32);
    for i in 0..old_w_level.nodes().len() {
        for wp in old_w_level.pairs_iter_of_idx(i) {
            stored_left |= wp.left.0;
            stored_right |= wp.right.0;
        }
    }
    // A crossed rotation's w had its children swapped: its left side is the
    // stored right one.
    let (w_left, w_right) = if crossed { (stored_right, stored_left) } else { (stored_left, stored_right) };
    let src = old_v_level.nodes().len().saturating_sub(1) as u32;
    match dir {
        RotationKind::Left => [v_axis, w_left, w_right, src],
        RotationKind::Right => [w_right, v_axis, w_left, src],
    }
}

/// Phase 1: expand every old v-pair against the w-level into packed triples.
/// Returns `false` if the triple count reaches `max_pairs` (bail check 1): the
/// distinct inner pairs are never more numerous than the triples, so a count
/// of them could not bail earlier.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the triple buffer's growth is refused.
#[allow(clippy::too_many_arguments)]
fn collect_triples<W: Word>(
    lim: &Limits,
    old_v_level: &TddLevel,
    old_w_level: &TddLevel,
    dir: RotationKind,
    crossed: bool,
    layout: &Layout,
    triples: &mut Vec<W>,
    max_pairs: usize,
) -> Result<bool, OperationError> {
    for i in 0..old_v_level.nodes().len() {
        let src = i as u32;
        for vp in old_v_level.pairs_iter_of_idx(i) {
            let (w_local, v_axis) = match dir {
                RotationKind::Left => (ChildDecoder::structural().node(vp.right).idx(), vp.left),
                RotationKind::Right => (ChildDecoder::structural().node(vp.left).idx(), vp.right),
            };
            for wp in old_w_level.pairs_iter_of_idx(w_local) {
                // A crossed rotation's w had its children swapped: its left
                // side is the stored right one.
                let (w_left, w_right) = if crossed { (wp.right, wp.left) } else { (wp.left, wp.right) };
                let (inner, axis) = match dir {
                    RotationKind::Left => (
                        ChildPair::new(v_axis, w_left),
                        w_right,
                    ),
                    RotationKind::Right => (
                        ChildPair::new(w_right, v_axis),
                        w_left,
                    ),
                };
                lim.try_push(triples, W::pack(layout, inner, src, axis))?;
                // Checked per triple: one `vp` whose `w_local` fans out widely
                // can push `triples` far past `max_pairs` within one `vp`.
                if triples.len() >= max_pairs {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// One distinct inner pair and where its cells sit in the packed `triples`.
#[derive(Clone, Copy)]
pub(super) struct PairGroup {
    /// A fingerprint of the cell list; equal lists have equal hashes.
    hash: u64,
    inner: ChildPair,
    /// The `[start, end)` bounds of the cells in `triples`.
    start: u32,
    end: u32,
    /// The inner node the pair went under, once the inner level is built.
    node: NodeIdx,
}

/// Phase 2: dedup cells in-place within each inner-pair group of the sorted
/// `triples` and record the group boundaries with a rolling fingerprint hash.
/// `keep_cells` retains the cell multiset instead of deduping it.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the group table's growth is refused.
fn group_by_inner_pair<W: Word>(
    lim: &Limits,
    triples: &mut Vec<W>,
    layout: &Layout,
    group_info: &mut Vec<PairGroup>,
    keep_cells: bool,
) -> Result<(), OperationError> {
    let mut read = 0;
    let mut write = 0;
    let n = triples.len();
    while read < n {
        let first = triples[read];
        let inner_key = first.inner_key(layout);
        let group_start = write as u32;
        let mut fp_hash: u64 = 0;
        let mut prev_cell = u64::MAX;
        while read < n && triples[read].inner_key(layout) == inner_key {
            let cell = triples[read].cell(layout);
            if keep_cells || cell != prev_cell {
                triples[write] = triples[read];
                write += 1;
                fp_hash = fp_hash.wrapping_mul(0x517cc1b727220a95)
                    .wrapping_add(cell);
                prev_cell = cell;
            }
            read += 1;
        }
        let group = PairGroup { hash: fp_hash, inner: tri_inner(first, layout), start: group_start, end: write as u32, node: NodeIdx(u32::MAX) };
        lim.try_push(group_info, group)?;
    }
    triples.truncate(write);
    Ok(())
}

/// Phases 3 and 4: turn the inner-pair groups into one inner level, recording
/// in each group the node its pair went under. Returns `Ok(None)` when bail
/// check 2 says the result would exceed `max_pairs`.
///
/// Two shapes, chosen by `marginal_ctx`: full expansion, one node per distinct
/// inner pair, or cell-list clustering, one node per distinct cell list.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the pair table's growth or the level's
/// arena is refused. The partially built level is dropped, so its charge goes
/// back first.
fn build_inner_level<W: Word>(
    lim: &Limits,
    triples: &[W],
    layout: &Layout,
    group_info: &mut [PairGroup],
    bucket: &mut BucketScratch,
    marginal_ctx: bool,
    max_pairs: usize,
) -> Result<Option<TddLevel>, OperationError> {
    // One group per distinct inner pair, so this is the new inner level's
    // pair count, which bail check 2 adds to its node count.
    let n_w_pairs = group_info.len();
    let mut level = Transient::new(lim, TddLevel::new());
    if marginal_ctx {
        // Bail check 2 (full-expand): one inner node per distinct inner pair.
        if group_info.len() + n_w_pairs >= max_pairs {
            return Ok(None);
        }
        // Every group's pair gets an entry, in either shape.
        expand_every_pair(lim, &mut level, group_info)?;
    } else {
        // Phase 3: number the distinct cell lists in the order of their first
        // groups, which is the number of the inner node each group goes under.
        let nodes = number_cell_lists(lim, triples, layout, group_info, &mut bucket.table)?;
        if nodes + n_w_pairs >= max_pairs {
            return Ok(None);
        }
        let built = cluster_by_cell_list(lim, &mut level, group_info, nodes, bucket);
        release_or_clear(lim, &mut bucket.table);
        release_or_clear(lim, &mut bucket.ends);
        release_or_clear(lim, &mut bucket.pairs);
        built?;
    }
    Ok(Some(level.keep()))
}

/// Marginal full-expand: sharing is suppressed, so each distinct inner pair
/// becomes its own node and no two pairs merge under one.
///
/// The Boolean `(a∧b)∨(a'∧b')` share that miscounts a marginalized grandchild
/// never forms. With the kept cell multiset and the kept outer multiset, Σ over
/// triples = the pre-rotation count exactly.
fn expand_every_pair(
    lim: &Limits,
    inner_level: &mut TddLevel,
    group_info: &mut [PairGroup],
) -> Result<(), OperationError> {
    for g in group_info {
        g.node = inner_level.push_node(lim, &[g.inner])?;
    }
    Ok(())
}

/// Set each group's `node` to the number of its cell list among the distinct
/// ones, numbered in the order of the first group that carries each, and
/// return how many there are: the inner level's node count.
///
/// An open-addressing table of the first groups, keyed by the cell lists'
/// fingerprints, finds a group's list among the earlier ones. The
/// fingerprint only proposes a match; membership is decided by comparing the
/// cell lists themselves, so a collision costs a comparison and never a
/// wrong share.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the table is refused.
fn number_cell_lists<W: Word>(
    lim: &Limits,
    triples: &[W],
    layout: &Layout,
    group_info: &mut [PairGroup],
    table: &mut Vec<u32>,
) -> Result<usize, OperationError> {
    let slots = (2 * group_info.len()).next_power_of_two().max(16);
    let mask = slots - 1;
    let shift = 64 - slots.trailing_zeros();
    table.clear();
    lim.try_resize(table, slots, u32::MAX)?;
    let mut nodes = 0u32;
    for j in 0..group_info.len() {
        let hash = group_info[j].hash;
        // A multiplicative hash's high bits, which every bit of the
        // fingerprint reaches.
        let mut at = (hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize;
        loop {
            let first = table[at];
            if first == u32::MAX {
                table[at] = j as u32;
                group_info[j].node = NodeIdx(nodes);
                nodes += 1;
                break;
            }
            let other = &group_info[first as usize];
            if other.hash == hash && same_cells(triples, layout, other, &group_info[j]) {
                group_info[j].node = other.node;
                break;
            }
            at = (at + 1) & mask;
        }
    }
    Ok(nodes as usize)
}

/// Phase 4: emit the inner nodes, node `k` holding the inner pairs of every
/// group numbered `k`, in the groups' order.
fn cluster_by_cell_list(
    lim: &Limits,
    inner_level: &mut TddLevel,
    group_info: &[PairGroup],
    nodes: usize,
    scratch: &mut BucketScratch,
) -> Result<(), OperationError> {
    let BucketScratch { ends, pairs, .. } = scratch;
    // Each node's pair count, at the slot after the node's.
    ends.clear();
    lim.try_resize(ends, nodes + 1, 0)?;
    for g in group_info {
        ends[g.node.idx() + 1] += 1;
    }
    for k in 0..nodes {
        ends[k + 1] += ends[k];
    }
    pairs.clear();
    lim.try_resize(pairs, group_info.len(), ChildPair::new(NodeIdx(0), NodeIdx(0)))?;
    for g in group_info {
        let at = &mut ends[g.node.idx()];
        pairs[*at as usize] = g.inner;
        *at += 1;
    }
    // Each node's start moved to its end: node `k`'s pairs end at `ends[k]`.
    // No canonicalizing sort: this rotated level is queued for twin
    // contraction, but twin detection is order-independent
    // (`find_twin_groups` sorts each signature slice before comparing), so the
    // node's pair order is free (see `ChildPair`).
    let mut begin = 0;
    for (k, &end) in ends[..nodes].iter().enumerate() {
        let idx = inner_level.push_node(lim, &pairs[begin as usize..end as usize])?;
        debug_assert_eq!(idx.idx(), k, "a pushed node is the level's last");
        begin = end;
    }
    Ok(())
}

/// Phase 5: build the outer level from the deduped (packed) triples, one node
/// per old v-node. `triples` is released once its pairs have been filed, on
/// the error path as well as the normal one.
///
/// Each node's pairs come out in ascending order, the order a sort of each
/// node's list leaves, and in the Boolean build free of duplicates.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if a filing buffer or the level's arena is
/// refused. The partially built level is dropped, so its charge goes back
/// first.
fn build_outer_level<W: Word>(
    lim: &Limits,
    old_v_level: &TddLevel,
    triples: &mut Vec<W>,
    layout: &Layout,
    scratch: &mut RestructureScratch,
    dir: RotationKind,
    marginal_ctx: bool,
) -> Result<TddLevel, OperationError> {
    let n_v = old_v_level.nodes().len();
    let filed = file_outer_pairs(lim, triples, layout, scratch, n_v, dir, marginal_ctx);
    // Last read of `triples` and the groups: the filed lists now hold every
    // outer pair. Release the triples before the arena that copies those
    // pairs is built.
    release_or_clear(lim, triples);
    release_or_clear(lim, &mut scratch.group_info);
    release_or_clear(lim, &mut scratch.by_axis);
    release_or_clear(lim, &mut scratch.axis_ends);
    let built = filed.and_then(|()| {
        let mut outer_level = Transient::new(lim, TddLevel::new());
        let ends = &scratch.outer_ends;
        let pairs = &scratch.outer_pairs;
        let mut begin = 0;
        for &end in &ends[..n_v] {
            outer_level.push_node(lim, &pairs[begin as usize..end as usize])?;
            begin = end;
        }
        Ok(outer_level.keep())
    });
    release_or_clear(lim, &mut scratch.outer_ends);
    release_or_clear(lim, &mut scratch.outer_pairs);
    built
}

/// The groups whose cells become outer pairs, in `group_info`'s order.
///
/// In the Boolean build the groups that went under one inner node carry the
/// same cell list, so only the first of them, the one that opened the node,
/// is filed: the others would file the same outer pairs again. The nodes are
/// numbered in the order of their first groups, so these come in ascending
/// order of their node. In marginal context every group has its own node and every
/// one is filed.
fn filed_groups(group_info: &[PairGroup], marginal_ctx: bool) -> impl Iterator<Item = &PairGroup> {
    let mut opened = 0u32;
    group_info.iter().filter(move |g| {
        if marginal_ctx {
            return true;
        }
        if g.node.0 != opened {
            return false;
        }
        opened += 1;
        true
    })
}

/// File each triple's outer pair, through the inner node its group went
/// under, under the old v-node it came from: node `i`'s pairs end up at
/// `outer_pairs[outer_ends[i - 1]..outer_ends[i]]`, from 0 for the first.
///
/// The filed groups ([`filed_groups`]) come in ascending order of their node
/// and a group's cells in ascending order of their axis. So a left rotation,
/// whose outer pair is `(node, axis)`, files each node's pairs in ascending
/// order. A right rotation's pair is `(axis, node)`: its cells are first
/// placed in ascending order of their axis, in a counting pass that keeps
/// the node order within an axis, and then filed, which leaves each node's
/// pairs in ascending order as well. Where the axis references are too sparse
/// for a counting pass, each node's pairs are sorted instead. Marginal context
/// keeps the outer multiset in the order filed: a duplicate outer pair is a
/// legitimate separate count-mass (two marginalization-collapsed twin
/// primes), so the sum over the kept multiset is the pre-rotation count
/// exactly.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if a buffer is refused.
fn file_outer_pairs<W: Word>(
    lim: &Limits,
    triples: &[W],
    layout: &Layout,
    scratch: &mut RestructureScratch,
    n_v: usize,
    dir: RotationKind,
    marginal_ctx: bool,
) -> Result<(), OperationError> {
    let RestructureScratch { group_info, by_axis, axis_ends, outer_ends: ends, outer_pairs: pairs, .. } = scratch;
    let filed = || filed_groups(group_info, marginal_ctx).flat_map(|g| triples[g.start as usize..g.end as usize].iter().map(move |&p| (g.node, p)));
    // Each node's pair count, at the slot after the node's.
    ends.clear();
    lim.try_resize(ends, n_v + 1, 0)?;
    let mut cells = 0usize;
    let mut max_axis = 0u32;
    for (_, p) in filed() {
        ends[tri_src(p, layout) as usize + 1] += 1;
        max_axis = max_axis.max(tri_axis(p, layout).0);
        cells += 1;
    }
    // Each node's start; the filing below moves each to the node's end.
    for i in 0..n_v {
        ends[i + 1] += ends[i];
    }
    pairs.clear();
    lim.try_resize(pairs, cells, ChildPair::new(NodeIdx(0), NodeIdx(0)))?;
    let mut file = |src: u32, pair: ChildPair| {
        let at = &mut ends[src as usize];
        pairs[*at as usize] = pair;
        *at += 1;
    };
    let axis_range = max_axis as usize + 1;
    let by_axis_order = !marginal_ctx && dir == RotationKind::Right && axis_range <= 2 * cells + 64;
    if by_axis_order {
        axis_ends.clear();
        lim.try_resize(axis_ends, axis_range + 1, 0)?;
        for (_, p) in filed() {
            axis_ends[tri_axis(p, layout).0 as usize + 1] += 1;
        }
        for a in 0..axis_range {
            axis_ends[a + 1] += axis_ends[a];
        }
        by_axis.clear();
        lim.try_resize(by_axis, cells, 0)?;
        for (node, p) in filed() {
            let at = &mut axis_ends[tri_axis(p, layout).0 as usize];
            by_axis[*at as usize] = (u64::from(tri_src(p, layout)) << 32) | u64::from(node.0);
            *at += 1;
        }
        let mut begin = 0;
        for (a, &end) in axis_ends[..axis_range].iter().enumerate() {
            for &e in &by_axis[begin as usize..end as usize] {
                file((e >> 32) as u32, ChildPair::new(EncodedChildRef::from_raw(a as u32), NodeIdx(e as u32)));
            }
            begin = end;
        }
    } else {
        for (node, p) in filed() {
            let outer_pair = match dir {
                RotationKind::Left => ChildPair::new(node, tri_axis(p, layout)),
                RotationKind::Right => ChildPair::new(tri_axis(p, layout), node),
            };
            file(tri_src(p, layout), outer_pair);
        }
        if !marginal_ctx && dir == RotationKind::Right {
            let mut begin = 0;
            for &end in &ends[..n_v] {
                pairs[begin as usize..end as usize].sort_unstable();
                begin = end;
            }
        }
    }
    debug_assert!(
        marginal_ctx || (0..n_v).all(|i| {
            let begin = if i == 0 { 0 } else { ends[i - 1] };
            pairs[begin as usize..ends[i] as usize].windows(2).all(|w| w[0] < w[1])
        }),
        "each outer node's pairs are ascending and distinct",
    );
    Ok(())
}


/// Whether two groups carry the same cell list in the deduped (packed)
/// triples array. Two cells are equal iff their `(axis, src)` parts match,
/// the low bits of the packed word ([`Word::cell`]).
#[inline]
fn same_cells<W: Word>(triples: &[W], layout: &Layout, a: &PairGroup, b: &PairGroup) -> bool {
    let a = &triples[a.start as usize..a.end as usize];
    let b = &triples[b.start as usize..b.end as usize];
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| x.cell(layout) == y.cell(layout))
}

#[cfg(test)]
#[path = "tests/relevel/mod.rs"]
mod tests;
