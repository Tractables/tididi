//! The four-way scatter join and the chunked emit that drains it.

use crate::diagram::EncodedChildRef;

use super::*;
use crate::apply::conjoin::marginal_plan::Carrier;
use crate::apply::conjoin::setup::LevelShape;
use crate::diagram::Sides;
use crate::limits::Limits;

use crate::Engine;

/// A candidate's `(left_prod, right_prod)` from the products the join met it by:
/// the inner side's product is the left child's normally and the right
/// sibling's when swapped, the outer side's the other way round.
#[inline(always)]
fn orient<const SWAPPED: bool>(inner: u32, outer: u32) -> (u32, u32) {
    if !SWAPPED { (inner, outer) } else { (outer, inner) }
}

/// Where the scatter puts the candidates it finds.
pub(super) enum Collect<'a> {
    /// A bucket per f parent, `par_buckets`.
    Buckets,
    /// One list of every candidate with its f parent, `par_flat`, sorted by
    /// parent once the scatter is done.
    Flat,
    /// The output level's pair arena, on a level where f and g have one node
    /// each: every candidate is a pair of the one product the level can
    /// have, so each is written where it stays. See [`finish_direct`].
    Direct(&'a mut TddLevel),
}

/// The leaf arm of the scatter: one side of the join is a vtree leaf, so the
/// leaf-side product comes straight from the conjunction table and the walk
/// stays selective by iterating the non-leaf product list.
///
/// A marginal pass-through side takes the leaf side's place: with `carrier`,
/// the inner side's product is the carrier's field there, and every
/// candidate survives on it.
// The level's steps are kept out of line from one another. Each runs once per
// level (or, for the chunk steps, once per chunk), so the call costs nothing
// against what it then does, and holding them apart means a change inside one
// cannot re-balance the inlining, register allocation or layout of the others:
// a measurement of one step then says what it means, instead of moving a step
// the change never touched.
#[inline(never)]
fn scatter_leaf_arm<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pl: Sides<&[ProductEntry]>,
    collect: Collect<'_>,
    carrier: Option<Carrier>,
) -> Result<(), OperationError> {
    match carrier {
        None => leaf_arm_into::<SWAPPED>(eng, ws, pl, collect, |f_label, g_label| {
            let grid_prod = CONJOIN_GRID[f_label as usize][g_label as usize];
            (grid_prod != NO_PRODUCT).then_some(grid_prod)
        }),
        Some(Carrier::F) => leaf_arm_into::<SWAPPED>(eng, ws, pl, collect, |f_field, _| Some(f_field)),
        Some(Carrier::G) => leaf_arm_into::<SWAPPED>(eng, ws, pl, collect, |_, g_field| Some(g_field)),
    }
}

/// [`scatter_leaf_arm`] with the inner side's product given by
/// `inner_product` of the f and g fields there, into `collect`.
#[inline(always)]
fn leaf_arm_into<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pl: Sides<&[ProductEntry]>,
    collect: Collect<'_>,
    inner_product: impl Fn(u32, u32) -> Option<u32>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let SparseWorkspace { f_by_outer, g_by_outer, par_buckets, par_flat, .. } = ws;
    let (f_by_outer, g_by_outer) = (f_by_outer.view(), g_by_outer.view());
    match collect {
        Collect::Buckets => leaf_join::<SWAPPED>(lim, f_by_outer, g_by_outer, pl, inner_product,
            |p1, entry| lim.try_push(&mut par_buckets[p1 as usize], entry)),
        Collect::Flat => leaf_join::<SWAPPED>(lim, f_by_outer, g_by_outer, pl, inner_product,
            |p1, entry| lim.try_push(par_flat, Candidate { parent: p1, entry })),
        Collect::Direct(level) => leaf_join::<SWAPPED>(lim, f_by_outer, g_by_outer, pl, inner_product,
            |_, entry| try_push_pair_into(eng, level, candidate_pair(&entry))),
    }
}

/// The leaf arm's walk: iterate the outer child's product list; the inner
/// side's product comes from `inner_product`, and each candidate is handed
/// to `push` with its f parent.
#[inline(always)]
fn leaf_join<const SWAPPED: bool>(
    lim: &Limits,
    f_by_outer: GroupedView<'_, RevEntry>,
    g_by_outer: GroupedView<'_, RevEntry>,
    pl: Sides<&[ProductEntry]>,
    inner_product: impl Fn(u32, u32) -> Option<u32>,
    mut push: impl FnMut(u32, ParEntry) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    // ── Leaf arm ──
    // Iterate the non-leaf product list; the leaf-side product comes from
    // `CONJOIN_GRID`, or is the carried field. g_by_outer's inner child is
    // the leaf label or the carried field here (normal: a2 with left the
    // inner side; swapped: s2 with right the inner side).
    //
    // Amortized cancellation/deadline poll — same rationale/soundness
    // as the general arm below; bail lands where `try_push` recovers.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    let pl_outer = if !SWAPPED { pl.right } else { pl.left };
    for &ProductEntry { f_idx: FNodeIdx(outer1), g_idx: GNodeIdx(outer2), prod_idx: ProductNodeIdx(outer_prod) } in pl_outer {
        let f_under_outer = f_by_outer.bucket(outer1 as usize);
        if f_under_outer.is_empty() { continue; }
        for &RevEntry { parent: g_parent, other: inner2 } in g_by_outer.bucket(outer2 as usize) {
            for &RevEntry { parent: p1, other: inner1 } in f_under_outer {
                // The inner side's product comes from its fields, the
                // product list gives the other side's.
                if let Some(inner_prod) = inner_product(inner1, inner2) {
                    let (left_prod, right_prod) = orient::<SWAPPED>(inner_prod, outer_prod);
                    push(p1, ParEntry { g_parent, left_prod, right_prod })?;
                }
            }
            ticker.poll(f_under_outer.len() as u64)?;
        }
    }
    Ok(())
}

/// The scatter: the four-way join of f/g parent and child/sibling product
/// lists. `SWAPPED = false` outer-loops
/// by right sibling s1; `SWAPPED = true` by left child a1 (every difference is a
/// pure left↔right role rename; the `if SWAPPED` branches fold at compile time).
/// A filtered per-outer g index makes the emit walk only alive `(p2, product)`
/// entries. The emitted ParEntry *set* into `par_buckets` is order-free — sound
/// because pair lists are order-independent.
///
/// Two arms behind a shared front-end (the two reverse-index builds):
///
/// **Leaf arm** (the inner child is a leaf, see [`runs_leaf_arm`]): iterate
/// the non-leaf product list; `CONJOIN_GRID` supplies the leaf-side product.
/// The `g_by_outer` keying is the same as the general arm's (normal → by
/// right, swapped → by left), so the front-end is shared. With `carrier`,
/// the inner side is a marginal pass-through instead, and its product is the
/// carrier's field.
///
/// **General arm** (both children non-leaf), per outer key:
///   1. Build `filtered`: bucket the g parents under the outer's live g keys
///      by the join's inner-g child, attaching the live product — from
///      whichever g index is the cheaper to walk, and only for the children
///      the emit will read when finding those costs less than it saves.
///   2. Emit: for each f-parent sharing the outer, for each alive inner product,
///      push the precomputed alive `(p2, product)` entries — zero dead probes.
///   3. Clear only the `filtered` buckets touched this outer.
///
/// The direction puts a leaf child, and a pass-through side, on the inner
/// side (`leaf_direction`), so the general arm sees only levels with two
/// joined non-leaf children, which are the levels the direction estimate ran
/// for: its counts start the general arm's index builds. `collect` says
/// where the candidates go.
#[expect(clippy::too_many_arguments)]
pub(super) fn scatter_join<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    leaves: Sides<bool>,
    collect: Collect<'_>,
    carrier: Option<Carrier>,
) -> Result<(), OperationError> {
    let leaf_arm = carrier.is_some() || runs_leaf_arm(SWAPPED, leaves);
    debug_assert!(leaf_arm || !(leaves.left || leaves.right), "a leaf child must be the inner side");
    build_scatter_indexes::<SWAPPED>(eng, ws, f_level, g_level, shape, !leaf_arm)?;
    if leaf_arm {
        return scatter_leaf_arm::<SWAPPED>(eng, ws, pl, collect, carrier);
    }
    build_inner_index::<SWAPPED>(eng, ws, g_level, shape)?;
    scatter_general_arm::<SWAPPED>(eng, ws, shape, pl, collect)
}

/// Whether the join runs its leaf arm: its inner child, the left one
/// unswapped and the right one swapped, is a leaf.
pub(super) fn runs_leaf_arm(swapped: bool, leaves: Sides<bool>) -> bool {
    if !swapped { leaves.left } else { leaves.right }
}

/// Build the two reverse indexes both arms read.
///
/// f is keyed by the outer-loop dimension (normal → right sibling `s1`,
/// swapped → left child `a1`). g is keyed by the general arm's filtering axis
/// — which is also exactly the keying the leaf arm wants, since that groups g
/// by the non-leaf outer child, so one build serves both arms: normal → by
/// right `s2`, entries `(p2, a2)`; swapped → by left `a2`, entries `(p2, s2)`.
///
/// With `counted`, the direction estimate has run for this level and its
/// counts start each build.
#[inline(never)]
fn build_scatter_indexes<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    counted: bool,
) -> Result<(), OperationError> {
    let SparseWorkspace { est_counts, f_by_outer, g_by_outer, .. } = ws;
    let counts = counted.then(|| EstCounts::of(est_counts, shape));
    // Keyed by the outer dimension: the right sibling normally, the left child
    // when swapped.
    if !SWAPPED {
        build_reverse_index::<true>(eng, f_level, shape.f.right, counts.as_ref().map(|c| c.f_right), f_by_outer)?;
        build_reverse_index::<true>(eng, g_level, shape.g.right, counts.as_ref().map(|c| c.g_right), g_by_outer)?;
    } else {
        build_reverse_index::<false>(eng, f_level, shape.f.left, counts.as_ref().map(|c| c.f_left), f_by_outer)?;
        build_reverse_index::<false>(eng, g_level, shape.g.left, counts.as_ref().map(|c| c.g_left), g_by_outer)?;
    }
    Ok(())
}

/// Build g's reverse index keyed by the join's inner-g child — the key
/// `g_by_outer` is not keyed by — for the general arm's second way of
/// building an outer's `filtered` index. The direction estimate has run for
/// the level, and its counts start the build.
#[inline(never)]
fn build_inner_index<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    g_level: &TddLevel,
    shape: LevelShape,
) -> Result<(), OperationError> {
    let SparseWorkspace { est_counts, g_by_inner, .. } = ws;
    let counts = EstCounts::of(est_counts, shape);
    if !SWAPPED {
        build_reverse_index::<false>(eng, g_level, shape.g.left, Some(counts.g_left), g_by_inner)
    } else {
        build_reverse_index::<true>(eng, g_level, shape.g.right, Some(counts.g_right), g_by_inner)
    }
}

/// Fill `offsets` with the bucket bounds of `list` over `width` f indices:
/// `list[offsets[i]..offsets[i + 1]]` is the run of entries with `f_idx`
/// `i`. One pass, since the list is already in that order — every producer
/// emits a product list in ascending `f_idx` order: the dense grid scan
/// row by row, the identity fill, the sparse emit by f parent, the
/// sparse-marginal rows in order. A list that is not leaves entries
/// unconsumed, which the check at the end catches.
fn bucket_offsets(
    lim: &crate::limits::Limits,
    list: &[ProductEntry],
    width: usize,
    offsets: &mut Vec<u32>,
) -> Result<(), OperationError> {
    lim.try_resize(offsets, width + 1, 0u32)?;
    let mut pos = 0usize;
    for (i, slot) in offsets.iter_mut().enumerate().take(width) {
        *slot = pos as u32;
        while pos < list.len() && list[pos].f_idx.idx() == i {
            pos += 1;
        }
    }
    offsets[width] = pos as u32;
    cheap_assert!(
        pos == list.len(),
        "a product list is not in ascending f order ({} of {} entries bucketed over width {width})",
        pos, list.len()
    );
    Ok(())
}

/// One direction's view of the scatter workspace.
///
/// `SWAPPED` renames left↔right throughout the join. Selecting the buffers
/// once, here, leaves the join itself written a single time: `outer` is the
/// dimension the emit loop
/// iterates, `inner` the one it joins against, and `filtered` the per-outer
/// index rebuilt from the g reverse index.
struct ScatterSides<'w> {
    /// f's reverse index, keyed by the outer dimension.
    f_by_outer: GroupedView<'w, RevEntry>,
    /// g's reverse index, keyed by the filtering axis.
    g_by_outer: GroupedView<'w, RevEntry>,
    /// Live products of the outer child, by their f index (see
    /// [`bucket_offsets`]).
    outer: GroupedView<'w, ProductEntry>,
    /// Live products of the inner child, likewise.
    inner: GroupedView<'w, ProductEntry>,
    /// g's reverse index keyed by the join's inner-g child.
    g_by_inner: GroupedView<'w, RevEntry>,
    /// Per-outer g index, keyed by the join's inner-g child.
    filtered: TouchedBuckets<'w>,
    /// The inner-g children this outer's emit reads, and the inner f children
    /// already walked to find them.
    wanted: EpochFlags<'w>,
    wanted_keys: &'w mut Vec<u32>,
    inner_seen: EpochFlags<'w>,
    /// The outer's live g keys and their products, for the walk by inner-g
    /// child.
    outer_keys: EpochFlags<'w>,
    outer_attached: &'w mut Vec<u32>,
    /// Surviving candidates, bucketed by f parent — or, on a level that
    /// collects them flat, appended with their parent for the sort.
    par_buckets: &'w mut Vec<Vec<ParEntry>>,
    par_flat: &'w mut Vec<Candidate>,
    /// How many outer keys the emit loop walks.
    outer_k: usize,
}

/// A bucket array cleared per outer key by replaying the indices written into
/// it, so the clear costs the touched buckets rather than the whole array.
struct TouchedBuckets<'a> {
    buckets: &'a mut Vec<Vec<(u32, u32)>>,
    touched: &'a mut Vec<u32>,
}

impl TouchedBuckets<'_> {
    /// Every bucket, indexed by key, for a walk that reads many of them.
    fn as_slice(&self) -> &[Vec<(u32, u32)>] {
        self.buckets
    }

    fn push(&mut self, lim: &crate::limits::Limits, key: u32, v: (u32, u32)) -> Result<(), OperationError> {
        let bucket = &mut self.buckets[key as usize];
        if bucket.is_empty() {
            self.touched.push(key);
        }
        lim.try_push(bucket, v)
    }

    fn clear_touched(&mut self) {
        for &key in self.touched.iter() {
            self.buckets[key as usize].clear();
        }
        self.touched.clear();
    }
}

/// A marking array emptied by advancing a stamp rather than by clearing it,
/// so starting a round is free however many keys the last one marked. The
/// stamp lives in the workspace and only ever moves forward, so a slot left
/// by an earlier round, level or apply reads as unmarked.
struct EpochFlags<'a> {
    stamps: &'a mut Vec<u32>,
    epoch: &'a mut u32,
    cur: u32,
}

impl EpochFlags<'_> {
    /// Empty the array: every key is unmarked again.
    #[inline]
    fn begin(&mut self) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            // The stamp wrapped, so a slot left by an older round could read
            // as marked. This costs one pass per 2^32 rounds.
            self.stamps.fill(0);
            self.cur = 1;
        }
        *self.epoch = self.cur;
    }

    /// Mark `key`, and report whether this call is the one that marked it.
    #[inline]
    fn mark(&mut self, key: u32) -> bool {
        let slot = &mut self.stamps[key as usize];
        if *slot == self.cur {
            return false;
        }
        *slot = self.cur;
        true
    }

    #[inline]
    fn is_set(&self, key: u32) -> bool {
        self.stamps[key as usize] == self.cur
    }
}

/// Take this direction's view of the workspace, with every bucket array this
/// arm writes cleared to `shape`'s dimensions and both product lists
/// bucketed by f index.
#[inline(never)]
fn sides<'w, const SWAPPED: bool>(
    eng: &Engine,
    ws: &'w mut SparseWorkspace,
    shape: LevelShape,
    pl_inner: &'w [ProductEntry],
    pl_outer: &'w [ProductEntry],
) -> Result<ScatterSides<'w>, OperationError> {
    let LevelShape { f, g, .. } = shape;
    let (inner_k, outer_k, filtered_dim, outer_g_dim) = if !SWAPPED {
        (f.left, f.right, g.left, g.right)
    } else {
        (f.right, f.left, g.right, g.left)
    };
    let SparseWorkspace {
        f_by_outer, g_by_outer, g_by_inner,
        inner_offsets, outer_offsets,
        filtered, filtered_touched, par_buckets, par_flat,
        wanted, wanted_epoch, wanted_keys, inner_seen, inner_seen_epoch,
        outer_keys, outer_keys_epoch, outer_attached,
        ..
    } = ws;
    bucket_offsets(eng.limits(), pl_inner, inner_k, inner_offsets)?;
    bucket_offsets(eng.limits(), pl_outer, outer_k, outer_offsets)?;
    ensure_buckets_cleared(eng, filtered, filtered_dim)?;
    filtered_touched.clear();
    eng.limits().try_resize(wanted, filtered_dim, 0u32)?;
    wanted_keys.clear();
    eng.limits().try_resize(inner_seen, inner_k, 0u32)?;
    eng.limits().try_resize(outer_keys, outer_g_dim, 0u32)?;
    eng.limits().try_resize(outer_attached, outer_g_dim, 0u32)?;
    Ok(ScatterSides {
        f_by_outer: f_by_outer.view(),
        g_by_outer: g_by_outer.view(),
        g_by_inner: g_by_inner.view(),
        outer: GroupedView { offsets: outer_offsets, entries: pl_outer },
        inner: GroupedView { offsets: inner_offsets, entries: pl_inner },
        filtered: TouchedBuckets { buckets: filtered, touched: filtered_touched },
        wanted: EpochFlags { cur: *wanted_epoch, stamps: wanted, epoch: wanted_epoch },
        wanted_keys,
        inner_seen: EpochFlags { cur: *inner_seen_epoch, stamps: inner_seen, epoch: inner_seen_epoch },
        outer_keys: EpochFlags { cur: *outer_keys_epoch, stamps: outer_keys, epoch: outer_keys_epoch },
        outer_attached,
        par_buckets,
        par_flat,
        outer_k,
    })
}

impl ScatterSides<'_> {
    /// Fill `filtered` for one outer key: for each live `(inner_live, attached)`
    /// in the outer's liveness bucket, walk the opposite-keyed g index and
    /// bucket each g parent by its inner child, carrying `attached` along.
    ///
    /// With `FILTER`, only the keys `mark_wanted_for_outer` found are
    /// bucketed: the emit reads no other, so a g parent under any other one
    /// would be bucketed, cleared and never looked at. Without it the walk
    /// is bounded already and the marking has not run.
    fn build_filtered_for_outer<const FILTER: bool>(
        &mut self,
        lim: &crate::limits::Limits,
        outer: usize,
    ) -> Result<(), OperationError> {
        for e in self.outer.bucket(outer) {
            let (right_key, attached) = (e.g_idx.0, e.prod_idx.0);
            for &RevEntry { parent: p2, other: inner_c2 } in self.g_by_outer.bucket(right_key as usize) {
                if FILTER && !self.wanted.is_set(inner_c2) {
                    continue;
                }
                self.filtered.push(lim, inner_c2, (p2, attached))?;
            }
        }
        Ok(())
    }

    /// The f pairs under one outer key: what the emit walks for it, and so
    /// the least the outer costs whatever else is done for it.
    fn f_pairs_under(&self, outer: usize) -> usize {
        self.f_by_outer.len(outer)
    }

    /// Mark the inner-g children this outer's emit will read: for each
    /// distinct inner f child under the outer key, the g children its live
    /// left products name.
    ///
    /// This is the join's own semi-join, one level up. What
    /// `build_filtered_for_outer` buckets is every g parent under the outer's
    /// g keys, which is unrelated to how many of them the emit then reads:
    /// where the two sides meet in few places, most of that index is written,
    /// cleared and never looked at. Marking first bounds the build by what
    /// the emit reads, and the marking walk is itself bounded by the emit's
    /// own outer loop — it visits the same products, once per distinct inner
    /// child rather than once per f parent.
    fn mark_wanted_for_outer(&mut self, outer: usize) {
        self.wanted.begin();
        self.wanted_keys.clear();
        self.inner_seen.begin();
        for &RevEntry { other: inner1, .. } in self.f_by_outer.bucket(outer) {
            if !self.inner_seen.mark(inner1) {
                continue;
            }
            for e in self.inner.bucket(inner1 as usize) {
                let inner_c2 = e.g_idx.0;
                if self.wanted.mark(inner_c2) {
                    self.wanted_keys.push(inner_c2);
                }
            }
        }
    }

    /// Fill `filtered` for one outer key the other way round: for each wanted
    /// inner-g child, walk its g parents and keep those under one of the
    /// outer's live g keys, carrying that key's product along.
    ///
    /// Same entries as [`ScatterSides::build_filtered_for_outer`], from the
    /// other index. That walk costs the g pairs under the outer's keys; this
    /// one costs the g pairs under the wanted children. A g operand free over
    /// the outer child keeps every one of its pairs under a single key, so
    /// the first walk reads the whole level once per outer while this one
    /// reads a few parents; on other levels the first is the cheaper. Each
    /// outer takes whichever its two index slices say is smaller.
    fn build_filtered_by_inner(
        &mut self,
        lim: &crate::limits::Limits,
        outer: usize,
    ) -> Result<(), OperationError> {
        let ScatterSides {
            outer: outer_products, outer_keys, outer_attached, wanted_keys,
            g_by_inner, filtered, ..
        } = self;
        outer_keys.begin();
        for e in outer_products.bucket(outer) {
            outer_keys.mark(e.g_idx.0);
            outer_attached[e.g_idx.idx()] = e.prod_idx.0;
        }
        for &inner_c2 in wanted_keys.iter() {
            for &RevEntry { parent: p2, other: right_key } in g_by_inner.bucket(inner_c2 as usize) {
                if !outer_keys.is_set(right_key) {
                    continue;
                }
                filtered.push(lim, inner_c2, (p2, outer_attached[right_key as usize]))?;
            }
        }
        Ok(())
    }

    /// The g index entries building this outer's `filtered` by the outer's
    /// g keys walks.
    fn build_cost_by_key(&self, outer: usize) -> usize {
        self.outer
            .bucket(outer)
            .iter()
            .map(|e| self.g_by_outer.len(e.g_idx.idx()))
            .sum()
    }

    /// The g index entries building this outer's `filtered` by the wanted
    /// inner-g children walks; the marking has run.
    fn build_cost_by_inner(&self) -> usize {
        self.wanted_keys
            .iter()
            .map(|&a| self.g_by_inner.len(a as usize))
            .sum()
    }

    /// Emit for one outer key: walk the f parents sharing it and, for each
    /// alive inner product, replay the precomputed alive `filtered` entries —
    /// so the inner loop probes no dead cell. Each candidate goes to its f
    /// parent's bucket, or with `FLAT` to the flat list with its parent for
    /// the sort that groups them afterwards.
    fn emit_for_outer<const SWAPPED: bool, const FLAT: bool>(
        &mut self,
        lim: &crate::limits::Limits,
        outer: usize,
        ticker: &mut crate::limits::PollGate,
    ) -> Result<(), OperationError> {
        let filtered = self.filtered.as_slice();
        for &RevEntry { parent: p1, other: inner1 } in self.f_by_outer.bucket(outer) {
            // The bucket is resolved once per f parent, outside the walk of
            // its products.
            if FLAT {
                let flat = &mut *self.par_flat;
                emit_candidates::<SWAPPED>(self.inner, filtered, inner1, ticker,
                    |entry| lim.try_push(flat, Candidate { parent: p1, entry }))?;
            } else {
                let bucket = &mut self.par_buckets[p1 as usize];
                emit_candidates::<SWAPPED>(self.inner, filtered, inner1, ticker,
                    |entry| lim.try_push(bucket, entry))?;
            }
        }
        Ok(())
    }

    /// [`ScatterSides::emit_for_outer`] on a level whose candidates go
    /// straight into the output level's pair arena ([`Collect::Direct`]).
    fn emit_direct_for_outer<const SWAPPED: bool>(
        &self,
        eng: &Engine,
        level: &mut TddLevel,
        outer: usize,
        ticker: &mut crate::limits::PollGate,
    ) -> Result<(), OperationError> {
        let filtered = self.filtered.as_slice();
        for &RevEntry { other: inner1, .. } in self.f_by_outer.bucket(outer) {
            emit_candidates::<SWAPPED>(self.inner, filtered, inner1, ticker,
                |entry| try_push_pair_into(eng, level, candidate_pair(&entry)))?;
        }
        Ok(())
    }
}

/// The candidates of one f parent under the current outer key: for each of
/// its alive inner products, every alive `(p2, attached)` the `filtered`
/// index holds for the product's inner-g child, handed to `push`.
///
/// A product whose inner-g child has no live g parent under this outer finds
/// its bucket empty, and where the two sides meet in few places most do:
/// such a miss reads the bucket's length and moves on, and only a hit
/// reaches the push loop and the poll.
#[inline(always)]
fn emit_candidates<const SWAPPED: bool>(
    inner: GroupedView<'_, ProductEntry>,
    filtered: &[Vec<(u32, u32)>],
    inner1: u32,
    ticker: &mut crate::limits::PollGate,
    mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    for e in inner.bucket(inner1 as usize) {
        let fb = &filtered[e.g_idx.idx()];
        if fb.is_empty() {
            continue;
        }
        let inner_prod = e.prod_idx.0;
        for &(g_parent, attached) in fb {
            let (left_prod, right_prod) = orient::<SWAPPED>(inner_prod, attached);
            push(ParEntry { g_parent, left_prod, right_prod })?;
        }
        ticker.poll(fb.len() as u64)?;
    }
    Ok(())
}

/// The general arm: both children are non-leaf. Per outer key, build the
/// filtered g index, emit against it, then clear only the buckets this outer
/// touched.
#[inline(never)]
fn scatter_general_arm<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    mut collect: Collect<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (pl_inner, pl_outer) = if !SWAPPED { (pl.left, pl.right) } else { (pl.right, pl.left) };
    let mut s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer)?;

    // Amortized cancellation/deadline poll. The sparse join has no other
    // mid-level break, so a wide level could otherwise wait out an expired
    // deadline; the bail lands at a loop level `try_push`'s recovery already
    // covers, so the workspace stays reusable.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for outer in 0..s.outer_k {
        if s.outer.bucket(outer).is_empty() { continue; }
        let by_key = s.build_cost_by_key(outer);
        if by_key <= s.f_pairs_under(outer) {
            // The walk by the outer's g keys costs no more than the emit's
            // own walk of the f pairs under it, and the marking costs at
            // least that walk: neither it nor the filter it feeds can pay
            // for itself here.
            s.build_filtered_for_outer::<false>(lim, outer)?;
        } else {
            s.mark_wanted_for_outer(outer);
            if s.build_cost_by_inner() < by_key {
                s.build_filtered_by_inner(lim, outer)?;
            } else {
                s.build_filtered_for_outer::<true>(lim, outer)?;
            }
        }
        match &mut collect {
            Collect::Flat => s.emit_for_outer::<SWAPPED, true>(lim, outer, &mut ticker)?,
            Collect::Buckets => s.emit_for_outer::<SWAPPED, false>(lim, outer, &mut ticker)?,
            Collect::Direct(level) => s.emit_direct_for_outer::<SWAPPED>(eng, level, outer, &mut ticker)?,
        }
        s.filtered.clear_touched();
    }
    Ok(())
}

/// Group a flat level's candidates by f parent: counting-sort `par_flat`
/// into `par_sorted`.
#[inline(never)]
pub(super) fn sort_candidates(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    parents: usize,
) -> Result<(), OperationError> {
    let SparseWorkspace { par_flat, par_sorted, .. } = ws;
    counting_sort(
        eng.limits(), parents, par_flat.iter().copied(),
        |c| (c.parent as usize, c.entry),
        None, ParEntry { g_parent: 0, left_prod: 0, right_prod: 0 }, par_sorted,
    )
}

/// Greedy bin-pack of f-parent indices into chunks whose projected dedup and
/// node-build transient byte cost stays under `bytes_budget`. `candidates` gives each
/// of the `left_width` parents' candidate count in parent order. Returns
/// boundary indices `[0, p1_a, p1_b, ..., left_width]`; each chunk processes
/// the parents `boundaries[i] .. boundaries[i+1]`.
///
/// A single p1's bucket is never split. Returns the single-chunk degenerate
/// list `[0, left_width]` when `bytes_budget` is `usize::MAX`, or when the
/// whole level fits in one chunk — in which case the call site's loop runs
/// exactly once and the path is byte-for-byte equivalent to the unchunked code.
#[inline]
pub(super) fn plan_chunks(
    candidates: impl Iterator<Item = usize>,
    left_width: usize,
    bytes_budget: usize,
) -> SmallVec<[u32; 8]> {
    let mut out: SmallVec<[u32; 8]> = SmallVec::new();
    out.push(0);
    if bytes_budget == usize::MAX {
        out.push(left_width as u32);
        return out;
    }
    let entries_budget = bytes_budget / BYTES_PER_PAR_ENTRY;
    let mut acc = 0usize;
    for (p1, n) in candidates.enumerate().take(left_width) {
        if acc != 0 && acc.saturating_add(n) > entries_budget {
            out.push(p1 as u32);
            acc = 0;
        }
        acc += n;
    }
    out.push(left_width as u32);
    out
}

/// Emit the output nodes of the f parents in `parents`, one parent at a time:
/// [`emit_parent`] numbers each parent's products and writes their nodes
/// straight into `level`. With `drop_consumed`, each parent's bucket is
/// released once it is emitted, so the level's output grows in the space its
/// candidates leave. `flat` says the candidates sit in the sorted flat list
/// rather than the buckets.
///
/// Pre: `pl_output.len()` is the number of products the level's earlier
/// parents made, so `prod_idx` stays sequential across chunks.
#[inline(never)]
#[expect(clippy::too_many_arguments)]
pub(super) fn emit_chunk(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    parents: std::ops::Range<usize>,
    flat: bool,
    drop_consumed: bool,
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    if flat {
        // The sorted list is moved out for the walk for the reason a bucket
        // is below, and handed back whatever the walk found: a level's
        // worth of candidates is worth keeping warm.
        let sorted = std::mem::take(&mut ws.par_sorted);
        let mut walked = Ok(());
        for p1 in parents {
            let candidates = sorted.view().bucket(p1);
            if candidates.is_empty() { continue; }
            walked = emit_parent(eng, ws, level, pl_output, p1, candidates, duplicates_legal);
            if walked.is_err() { break; }
        }
        ws.par_sorted = sorted;
        return walked;
    }

    // Each bucket is moved out for its walk rather than borrowed in place: the
    // emit writes `ws.p2_map` and `ws.pair_counts`, which an outstanding
    // borrow of `ws.par_buckets` conflicts with, and indexing the bucket per
    // entry to work around that re-reads its pointer and length for every
    // candidate.
    //
    // A chunked level wants the consumed bucket's memory freed anyway, before
    // the output grows further, so there it simply is not handed back. A
    // level in one chunk hands it back, because the next apply's
    // `ensure_buckets_cleared` only `.clear()`s (length=0, capacity retained)
    // and that capacity saves the next apply's scatter pushes from growing
    // the bucket again.
    for p1 in parents {
        let bucket = std::mem::take(&mut ws.par_buckets[p1]);
        if !bucket.is_empty() {
            emit_parent(eng, ws, level, pl_output, p1, &bucket, duplicates_legal)?;
        }
        if !drop_consumed {
            ws.par_buckets[p1] = bucket;
        }
    }
    Ok(())
}

/// The output pair a candidate contributes to its product's node.
#[inline(always)]
fn candidate_pair(entry: &ParEntry) -> ChildPair {
    // A joined child is never marginal (guarded at `apply_sparse_level`
    // entry), so its ref is a plain structural index. A pass-through side
    // holds the carrier's marginal ref verbatim, which the level's value-ref
    // markers (`mark_passthrough_inlined`) keep the end-of-apply tagger off.
    // Either way there is no bit-30 slot tagging here.
    ChildPair::new(EncodedChildRef::from_raw(entry.left_prod), EncodedChildRef::from_raw(entry.right_prod))
}

/// Emit one f parent: dedup its candidates' g parents into output products
/// and push each product's node, holding its candidates' pairs, onto `level`.
///
/// Products are numbered in the order their g parent first appears among
/// the candidates, and each node lists its pairs in candidate order: the
/// order a stable sort of the candidates by product would give. `p2_map`
/// maps a g parent to its product while the parent is emitted and is
/// restored to `NO_PRODUCT` before returning; a bail leaves entries behind,
/// which `WsGuard` repairs.
///
/// A parent whose candidates all name one g parent, or all distinct ones,
/// has its pairs in node order already, and they are copied into `level` as
/// they come. Otherwise a counting sort by product writes each pair into
/// its node's range of `level.pairs` directly; a product with one pair is
/// stored inline in its node and takes no range.
///
/// `duplicates_legal` says a node's pair list may repeat a pair: some level
/// of an operand, or of the output so far, is marginal, so pair lists are
/// multisets feeding a sum. Read only by the debug duplicate check below.
#[inline]
fn emit_parent(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    candidates: &[ParEntry],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let SparseWorkspace { p2_map, pair_counts, single_pairs, .. } = ws;
    let first = pl_output.len();
    pair_counts.clear();
    for entry in candidates {
        let slot = p2_map[entry.g_parent as usize];
        if slot == NO_PRODUCT {
            let idx = pl_output.len() as u32;
            p2_map[entry.g_parent as usize] = idx;
            lim.try_push(pl_output, ProductEntry {
                f_idx: FNodeIdx(p1 as u32),
                g_idx: GNodeIdx(entry.g_parent),
                prod_idx: ProductNodeIdx(idx),
            })?;
            lim.try_push(pair_counts, 1)?;
        } else {
            pair_counts[slot as usize - first] += 1;
        }
    }
    let products = pl_output.len() - first;
    let first_node = level.nodes.len();
    // The parent's pairs are appended to the level's arena under one reserve
    // and cut into nodes; the output-pair meter is charged once for the
    // growth, as the dense walk's choke point charges per growth event.
    let pre_pairs_cap = level.pairs.capacity();
    lim.reserve(&mut level.nodes, products)?;
    let built = if products == candidates.len() {
        candidates.iter().try_for_each(|entry| emit_single_pair(eng, level, candidate_pair(entry)))
    } else if products == 1 {
        push_node_from(eng, level, candidates.iter().map(candidate_pair))
    } else {
        push_nodes_sorted(eng, level, pair_counts, single_pairs, candidates, &p2_map[..], first)
    };
    lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    for e in &pl_output[first..] {
        p2_map[e.g_idx.idx()] = NO_PRODUCT;
    }
    built?;
    // No sort and no dedup: pair lists are order-free, and canonical child
    // levels make the grid lookups injective, so a duplicate in a purely
    // Boolean diagram is an upstream canonicity violation. Once any level
    // is marginal, duplicates are legal (`duplicates_legal`).
    debug_assert!(
        duplicates_legal || (first_node..level.nodes.len()).all(|i| {
            let mut seen = std::collections::HashSet::new();
            level.pairs_of_idx(i).iter().all(|p| seen.insert(*p))
        }),
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}

/// Push one node holding every pair of `pairs`, at least two, written
/// straight into the level's pair arena.
#[inline]
fn push_node_from(
    eng: &Engine,
    level: &mut TddLevel,
    pairs: impl ExactSizeIterator<Item = ChildPair>,
) -> Result<(), OperationError> {
    debug_assert!(pairs.len() >= 2, "a one-pair node is pushed inline");
    let start = level.pairs.len();
    reserve_pairs_for_emit(eng, level, pairs.len())?;
    level.pairs.extend(pairs);
    finish_node(eng, level, start)?;
    Ok(())
}

/// Push a parent's nodes when its candidates interleave several products:
/// a counting sort by product writes each multi-pair node's pairs into its
/// range of the level's pair arena, in candidate order, and a product with
/// one pair keeps it in `singles` for its inline node.
///
/// `cursors[k]` enters as product `first + k`'s candidate count and is
/// spent as its write cursor.
fn push_nodes_sorted(
    eng: &Engine,
    level: &mut TddLevel,
    cursors: &mut [u32],
    singles: &mut Vec<ChildPair>,
    candidates: &[ParEntry],
    p2_map: &[u32],
    first: usize,
) -> Result<(), OperationError> {
    // A product with one pair has no range; any cursor is below the total.
    const SINGLE: u32 = u32::MAX;
    let zero = ChildPair::new(EncodedChildRef::from_raw(0), EncodedChildRef::from_raw(0));
    let base = level.pairs.len();
    let mut total = 0u32;
    for cursor in cursors.iter_mut() {
        if *cursor >= 2 {
            let start = total;
            total += *cursor;
            *cursor = start;
        } else {
            *cursor = SINGLE;
        }
    }
    reserve_pairs_for_emit(eng, level, total as usize)?;
    level.pairs.resize(base + total as usize, zero);
    singles.clear();
    eng.limits().try_resize(singles, cursors.len(), zero)?;
    for entry in candidates {
        let k = p2_map[entry.g_parent as usize] as usize - first;
        let cursor = cursors[k];
        if cursor == SINGLE {
            singles[k] = candidate_pair(entry);
        } else {
            level.pairs[base + cursor as usize] = candidate_pair(entry);
            cursors[k] = cursor + 1;
        }
    }
    // Each range now ends where its cursor stopped, and the ranges were laid
    // out in product order, so each one starts where the one before ended.
    let mut start = 0u32;
    for (k, &end) in cursors.iter().enumerate() {
        if end == SINGLE {
            emit_single_pair(eng, level, singles[k])?;
        } else {
            level.try_push_multi_by_range(base + start as usize, (end - start) as usize)
                .map_err(|_| OperationError::OverBudget)?;
            start = end;
        }
    }
    Ok(())
}

/// Close a level the scatter wrote straight into `level`'s pair arena from
/// `base` on ([`Collect::Direct`]): the pairs there are the one product's,
/// `(0, 0)`, and become its node. No pair means the product is false and the
/// level stays empty.
pub(super) fn finish_direct(
    eng: &Engine,
    level: &mut TddLevel,
    base: usize,
    pl_output: &mut Vec<ProductEntry>,
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    if level.pair_tail_len(base) == 0 {
        return Ok(());
    }
    lim.reserve(&mut level.nodes, 1)?;
    let node = finish_node(eng, level, base)?.expect("the scatter wrote a pair");
    let idx = pl_output.len() as u32;
    debug_assert_eq!(node.0, idx, "the level's one product is its first node");
    lim.try_push(pl_output, ProductEntry {
        f_idx: FNodeIdx(0),
        g_idx: GNodeIdx(0),
        prod_idx: ProductNodeIdx(idx),
    })?;
    debug_assert!(
        duplicates_legal || {
            let mut seen = std::collections::HashSet::new();
            level.pairs_of_idx(node.idx()).iter().all(|p| seen.insert(*p))
        },
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}
