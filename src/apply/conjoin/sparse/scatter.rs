//! The scatter joins that find a level's candidates, and the walks that
//! hand them to the emit.

use crate::diagram::{EncodedChildRef, EncodedNode};

use super::*;
use crate::apply::conjoin::marginal_plan::Carrier;
use crate::apply::conjoin::setup::LevelShape;
use crate::diagram::Sides;

use crate::Engine;

/// A candidate's `(left_prod, right_prod)` from the products the join met it by:
/// the inner side's product is the left child's normally and the right
/// sibling's when swapped, the outer side's the other way round.
#[inline(always)]
fn orient<const SWAPPED: bool>(inner: u32, outer: u32) -> (u32, u32) {
    if !SWAPPED { (inner, outer) } else { (outer, inner) }
}

/// Where the scatter puts the candidates it finds.
pub(super) enum Collect<'a, 'c> {
    /// The level's nodes, each f parent's candidates turned into its
    /// products' nodes by the emit as the walk of f's nodes leaves the
    /// parent ([`walk_nodes`]).
    Nodes(Emit<'a>),
    /// The output level's pair arena, on a level where f and g have one node
    /// each: every candidate is a pair of the one product the level can
    /// have, so each is written where it stays. See [`finish_direct`].
    Direct(&'a mut TddLevel),
    /// A count, on a level with one product as for [`Collect::Direct`]:
    /// each candidate is folded into it and not stored.
    Fold(&'a mut CandidateFold<'c>),
    /// Sums, on a root with one product as for [`Collect::Direct`] and one
    /// child summed out: each candidate adds its summed-side node's count to
    /// its kept product's sum and is not stored. See [`ChildSum`].
    Sum(&'a mut ChildSum),
}

/// The level the emit writes: its nodes, its product list, whether its
/// pair lists may repeat a pair (see [`emit_walked`]), the thresholds the
/// routed walk reads ([`SparseThresholds::pipelined_bytes`]), and the live
/// pairs of what the walk made, which it sets where it ends.
pub(super) struct Emit<'a> {
    pub(super) level: &'a mut TddLevel,
    pub(super) out: &'a mut Vec<ProductEntry>,
    pub(super) duplicates_legal: bool,
    pub(super) thresholds: SparseThresholds,
    pub(super) live: &'a mut Option<usize>,
}

/// What a walk into the level's nodes ([`Collect::Nodes`]) made of the level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Made {
    /// The level's nodes and pairs, written.
    Written,
    /// The products listed, and no node or pair written: each product is its
    /// f node's, made of the node's pairs as they are, and f's level holds
    /// its nodes and its arena as the level would, node for node and pair
    /// for pair. The caller moves f's nodes and arena into the level.
    F,
    /// The level's nodes and pairs, written as f's level holds them, node for
    /// node, each node's pairs in order: each product is its f node's, and the
    /// level held nothing before.
    CopiedF,
}

/// Where the leaf arm reads the inner side's product from when the inner
/// child is no vtree leaf: the child is joined on its fields alone, and kills
/// no candidate.
#[derive(Clone, Copy)]
pub(super) enum Inner {
    /// A marginal pass-through side: the carrier's field.
    Carried(Carrier),
    /// A complete child (`Products::is_complete`):
    /// cell `(i, j)` is node `i * g_width + j`, `g_width` being `g`'s width
    /// there.
    Complete { g_width: usize },
}

/// The leaf arm: the level's inner child is joined on its fields alone, so
/// a candidate is an f pair, a live product of its outer child and a g pair
/// under that product's g node, the inner product read from the two pairs'
/// inner fields. The inner child is a vtree leaf, whose products are the
/// conjunction table's, a marginal pass-through side, whose product is the
/// carrier's field ([`Inner::Carried`]), or a complete child read by
/// arithmetic ([`Inner::Complete`]); either way it kills no candidate the
/// table does not.
///
/// f's nodes are walked in order: each pair through the outer child's live
/// products of its outer child, from the product list by f index
/// ([`bucket_offsets`]), and each product through the g pairs under its g
/// node, from g's reverse index by the outer child. A node's candidates are
/// all found when its walk ends, so they go to `collect` a node at a time,
/// and only g is indexed: f's pairs are read once, in place. The candidates
/// are those a join by outer product finds, and the clock is charged the
/// same, the g pairs under each product once per f pair that meets it.
///
/// `collect` is the level's nodes ([`Collect::Nodes`]), each f parent
/// emitted as its walk ends, or, on a level with one product, its pair
/// arena, a count or sums.
///
/// Under a vtree leaf, with g's pairs read as a table of owners, sets
/// [`SparseWorkspace::outer_held`] where every live product of the outer
/// child is held for every label ([`holds_every_label`]): each f pair of
/// its outer child then meets a g pair under each of them.
// The level's steps are kept out of line from one another. Each runs once per
// level, so the call costs nothing
// against what it then does, and holding them apart means a change inside one
// cannot re-balance the inlining, register allocation or layout of the others:
// a measurement of one step then says what it means, instead of moving a step
// the change never touched.
#[inline(never)]
#[expect(clippy::too_many_arguments)]
pub(super) fn scatter_leaf_arm<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    collect: Collect<'_, '_>,
    inner: Option<Inner>,
) -> Result<Made, OperationError> {
    let (f_outer, g_outer, pl_outer) = if !SWAPPED {
        (shape.f.right, shape.g.right, pl.right)
    } else {
        (shape.f.left, shape.g.left, pl.left)
    };
    let lim = eng.limits();
    // Under a vtree leaf, g's pairs by outer child and label make a table of
    // their owners, one slot per label, where no pair repeats
    // ([`build_owner_table`]): an f pair reads the slots of the labels its
    // own meets, under each product.
    if inner.is_none() {
        let built = if !SWAPPED {
            build_owner_table::<true>(lim, g_level, g_outer, &mut ws.owners, &mut ws.owners_clean)?
        } else {
            build_owner_table::<false>(lim, g_level, g_outer, &mut ws.owners, &mut ws.owners_clean)?
        };
        if let Some(ones) = built {
            // Into the level's nodes from f's nodes, stored or implied at one
            // pair a node ([`walkable`]), where each f index
            // has one live product, in order: the walk reads each product and
            // its slots from the list and the table themselves, nothing
            // gathered, and the table is cleared after it.
            let collect = match collect {
                Collect::Nodes(to) if walkable(f_level) && one_each(pl_outer, f_outer) => {
                    let gathered = gather_listed_products(lim, pl_outer, &ws.owners[..OWNER_SLOTS * g_outer], &mut ws.owned);
                    if !SWAPPED {
                        clear_owner_table::<true>(g_level, g_outer, &mut ws.owners, &mut ws.owners_clean);
                    } else {
                        clear_owner_table::<false>(g_level, g_outer, &mut ws.owners, &mut ws.owners_clean);
                    }
                    let (plain, held) = gathered?;
                    ws.outer_held = held;
                    let plain = plain && !ones;
                    let SparseWorkspace { owned, node_owners, emit, .. } = &mut *ws;
                    let listed = Listed { owned: &owned[..f_outer], plain };
                    let (f, none) = (shape.f.here, GroupedView { offsets: &[], entries: &[] });
                    return if ones {
                        walk_owned::<SWAPPED, true, true, false>(eng, node_owners, emit, f_level, f, none, listed, to)
                    } else {
                        walk_owned::<SWAPPED, false, true, false>(eng, node_owners, emit, f_level, f, none, listed, to)
                    };
                }
                collect => collect,
            };
            // Each live product of the outer child with its slots, read once;
            // the table is then done with.
            let gathered = gather_owned_products(lim, pl_outer, f_outer, &ws.owners[..OWNER_SLOTS * g_outer],
                &mut ws.outer_offsets, &mut ws.owned);
            if !SWAPPED {
                clear_owner_table::<true>(g_level, g_outer, &mut ws.owners, &mut ws.owners_clean);
            } else {
                clear_owner_table::<false>(g_level, g_outer, &mut ws.owners, &mut ws.owners_clean);
            }
            let held;
            (ws.owned_len, held) = gathered?;
            ws.outer_held = held && ws.owned_len == pl_outer.len();
            // Into the level's nodes from f's nodes, the walk runs as
            // one loop over them ([`owner_walk_nodes`]), each pair's products
            // read from its outer child's bucket.
            let collect = match collect {
                Collect::Nodes(to) if walkable(f_level) => {
                    let SparseWorkspace { owned, owned_len, outer_offsets, node_owners, emit, .. } = ws;
                    let outer = GroupedView { offsets: &outer_offsets[..=f_outer], entries: &owned[..*owned_len] };
                    let (f, none) = (shape.f.here, Listed { owned: &[], plain: false });
                    // Without One, a product held for every label is held
                    // with both literals.
                    return match (ones, held) {
                        (true, _) => walk_owned::<SWAPPED, true, false, false>(eng, node_owners, emit, f_level, f, outer, none, to),
                        (false, false) => walk_owned::<SWAPPED, false, false, false>(eng, node_owners, emit, f_level, f, outer, none, to),
                        (false, true) => walk_owned::<SWAPPED, false, false, true>(eng, node_owners, emit, f_level, f, outer, none, to),
                    };
                }
                collect => collect,
            };
            if ones {
                leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
                    |parts| OwnerWalk::<true>::of(parts, f_outer))?;
            } else {
                leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
                    |parts| OwnerWalk::<false>::of(parts, f_outer))?;
            }
            return Ok(Made::Written);
        }
    }
    // g's reverse index by the outer child: normally the right child,
    // entries `(p2, a2)`; swapped the left one, entries `(p2, s2)`. Under a
    // vtree leaf the entries of one outer child are grouped by their leaf
    // label as well, so that an f pair reads only the labels its own meets.
    let labelled = inner.is_none();
    match (labelled, SWAPPED) {
        (false, false) => build_reverse_index::<true>(eng, g_level, g_outer, None, &mut ws.g_by_outer)?,
        (false, true) => build_reverse_index::<false>(eng, g_level, g_outer, None, &mut ws.g_by_outer)?,
        (true, false) => build_labelled_index::<true>(eng, g_level, g_outer, &mut ws.g_by_outer)?,
        (true, true) => build_labelled_index::<false>(eng, g_level, g_outer, &mut ws.g_by_outer)?,
    }
    bucket_offsets(lim, pl_outer, f_outer, &mut ws.outer_offsets)?;
    let walked = match inner {
        None => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            |parts| LabelWalk { g_by_outer: parts.g_by_outer, outer: parts.outer }),
        Some(Inner::Carried(Carrier::F)) => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            plain_walk(|f_field: u32, _: u32| f_field)),
        Some(Inner::Carried(Carrier::G)) => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            plain_walk(|_: u32, g_field: u32| g_field)),
        Some(Inner::Complete { g_width }) => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            plain_walk(move |f_node: u32, g_node: u32| (f_node as usize * g_width + g_node as usize) as u32)),
    };
    walked.map(|()| Made::Written)
}

/// A [`LeafWalk`] with `inner_product`, for [`leaf_walk`] to make.
fn plain_walk<'a, P: Fn(u32, u32) -> u32>(inner_product: P) -> impl FnOnce(WalkParts<'a>) -> LeafWalk<'a, P> {
    move |parts| LeafWalk { g_by_outer: parts.g_by_outer, outer: parts.outer, inner_product }
}

/// What a leaf walk may read: g's reverse index, the outer child's live
/// products by f index, and those that meet an owner with their slots
/// ([`gather_owned_products`]), by f index through the same offsets.
struct WalkParts<'a> {
    g_by_outer: GroupedView<'a, RevEntry>,
    owned: &'a [OwnedProduct],
    outer: GroupedView<'a, ProductEntry>,
}

/// [`scatter_leaf_arm`]'s walk of f's nodes, each node's pairs read by the
/// walk `walk` makes of g's reverse index and the outer child's products,
/// into `collect`.
#[inline(always)]
fn leaf_walk<'w, const SWAPPED: bool, W: NodeWalk>(
    eng: &Engine,
    ws: &'w mut SparseWorkspace,
    f_level: &TddLevel,
    shape: LevelShape,
    pl_outer: &'w [ProductEntry],
    collect: Collect<'_, '_>,
    walk: impl FnOnce(WalkParts<'w>) -> W,
) -> Result<(), OperationError> {
    let SparseWorkspace { g_by_outer, owned, owned_len, outer_offsets, node_owners, emit, .. } = ws;
    let walk = walk(WalkParts {
        g_by_outer: g_by_outer.view(),
        owned: &owned[..*owned_len],
        outer: GroupedView { offsets: outer_offsets, entries: pl_outer },
    });
    walk_nodes::<SWAPPED, W>(eng, &walk, f_level, Walked::All(shape.f.here), collect, node_owners, emit)
}

/// The f nodes a walk of f's nodes visits: all of the level's, or those
/// whose bit is set in `bits`, one bit per node.
#[derive(Clone, Copy)]
pub(super) enum Walked<'a> {
    All(usize),
    Marked { bits: &'a [u64], nodes: usize },
}

impl Walked<'_> {
    /// The nodes the walk runs over: one past the last it may visit.
    fn end(self) -> usize {
        match self {
            Walked::All(nodes) | Walked::Marked { nodes, .. } => nodes,
        }
    }
}

/// Hand `f` each node `walked` names, in order, with its pairs.
#[inline(always)]
fn try_for_each_walked<E>(
    f_level: &TddLevel,
    walked: Walked<'_>,
    mut f: impl FnMut(usize, &[ChildPair]) -> Result<(), E>,
) -> Result<(), E> {
    let (bits, nodes) = match walked {
        Walked::All(nodes) => return f_level.try_for_each_node(0..nodes, f),
        Walked::Marked { bits, nodes } => (bits, nodes),
    };
    // Each run of set bits is one range of nodes.
    let mut at = 0;
    while at < nodes {
        let word = bits[at >> 6] & (!0u64 << (at & 63));
        if word == 0 {
            at = ((at >> 6) + 1) << 6;
            continue;
        }
        let start = (at & !63) + word.trailing_zeros() as usize;
        let mut end = start;
        while end < nodes {
            let clear = !bits[end >> 6] & (!0u64 << (end & 63));
            if clear != 0 {
                end = (end & !63) + clear.trailing_zeros() as usize;
                break;
            }
            end = ((end >> 6) + 1) << 6;
        }
        let end = end.min(nodes);
        f_level.try_for_each_node(start..end, &mut f)?;
        at = end;
    }
    Ok(())
}

/// The growth of a level's pair arena as f is walked: once the walk has
/// passed one in [`PROJECT_AFTER`] of what it walks, an arena too full for
/// what follows is grown to the level's projected candidates, the
/// candidates so far scaled by the rest of the walk, and an eighth more,
/// in one step where doubling would copy it once per doubling. The
/// projection is at most [`PROJECT_AFTER`] times the candidates so far.
///
/// A walk of a level's nodes projects where a candidate finds the arena
/// full, by the nodes walked ([`grow_by_nodes`]), so that a node costs
/// nothing more. A level of one product has one f node, and its walk
/// projects before each stretch of its pairs or each outer key's, by the
/// pairs walked.
struct Projection {
    /// The arena's length when the walk began.
    start: usize,
    /// The f pairs walked so far.
    done: usize,
    /// The f pairs the walk visits, counted at the first projection.
    total: Option<usize>,
}

/// The share of its f pairs or nodes a walk passes before it projects its
/// level's candidates: one in this many.
const PROJECT_AFTER: usize = 64;

impl Projection {
    fn new(level: &mut TddLevel) -> Self {
        Projection { start: level.pairs.stored_len(), done: 0, total: None }
    }

    /// Before the walk of the next `pairs` f pairs: grow the arena to the
    /// projection when their expected candidates, twice the level's rate so
    /// far, would not fit. `total` counts the f pairs the walk visits.
    #[inline(always)]
    fn before(&mut self, eng: &Engine, level: &mut TddLevel, pairs: usize, total: impl FnOnce() -> usize) {
        let done = self.done;
        self.done += pairs;
        let v = level.pairs.stored_mut();
        let (len, spare) = (v.len(), v.capacity() - v.len());
        let made = len.saturating_sub(self.start);
        if done == 0 || (spare as u128) * (done as u128) >= 2 * (made as u128) * (pairs as u128) {
            return;
        }
        let total = *self.total.get_or_insert_with(total);
        let expected = (2 * made as u128 * pairs as u128 / done as u128) as usize;
        grow_to_projection(eng, v, made, done, total, expected);
    }
}

/// Grow `v`, holding `made` candidates of a walk `done` of whose `total`
/// units are walked, to the projection of [`Projection`], or to `least`
/// more where that is more. Nothing before the walk has passed one in
/// [`PROJECT_AFTER`] of its units, on a level of bounded growth, where the
/// room is there already, or where the memory left could not take the
/// growth three times over: the ordinary doubling then grows the arena.
#[cold]
#[inline(never)]
fn grow_to_projection(eng: &Engine, v: &mut Vec<ChildPair>, made: usize, done: usize, total: usize, least: usize) {
    let lim = eng.limits();
    let Some(want) = projected_growth(lim, made, done, total, least, v.capacity() - v.len()) else {
        return;
    };
    let before = v.capacity();
    if lim.reserve_exact(v, want).is_ok() {
        crate::apply::conjoin::note_pairs_grown();
        lim.charge_output_pairs(v.capacity().saturating_sub(before));
    }
}

/// The pairs [`grow_to_projection`] grows an arena of `spare` room by, or
/// `None` where it leaves it to grow by doubling.
#[inline(always)]
fn projected_growth(lim: &crate::limits::Limits, made: usize, done: usize, total: usize, least: usize, spare: usize) -> Option<usize> {
    if done == 0 || done.saturating_mul(PROJECT_AFTER) < total || lim.bounded_growth() {
        return None;
    }
    let projected = (made as u128 * total as u128 / done as u128) as usize;
    let want = projected.saturating_add(projected / 8).saturating_sub(made).max(least);
    let bytes = (want as u64).saturating_mul(3).saturating_mul(std::mem::size_of::<ChildPair>() as u64);
    if want <= spare || bytes >= lim.headroom() {
        return None;
    }
    Some(want)
}

/// [`grow_to_projection`] for a walk of a level's nodes whose arena held
/// `start` pairs when the walk began, at a candidate of node `p1` of the
/// walk's `nodes`, which finds the arena full.
#[inline(always)]
fn grow_by_nodes(eng: &Engine, v: &mut Vec<ChildPair>, start: usize, p1: usize, nodes: usize) {
    grow_to_projection(eng, v, v.len().saturating_sub(start), p1, nodes, 1);
}

/// The pairs of the f nodes `walked` names.
fn walked_pairs(f_level: &TddLevel, walked: Walked<'_>) -> usize {
    let mut total = 0usize;
    let _ = try_for_each_walked::<()>(f_level, walked, |_, pairs| {
        total += pairs.len();
        Ok(())
    });
    total
}

/// Walk f's nodes in order, each node's candidates found by `walk`, into
/// `collect`: into the level's nodes, each node emitted as its walk ends
/// ([`emit_walked`]), or, on a level with one product, its pair arena, a
/// count or sums.
#[inline(always)]
pub(super) fn walk_nodes<const SWAPPED: bool, W: NodeWalk>(
    eng: &Engine,
    walk: &W,
    f_level: &TddLevel,
    nodes: Walked<'_>,
    collect: Collect<'_, '_>,
    node_owners: &mut Vec<u32>,
    emit: &mut EmitScratch,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    // Amortized cancellation/deadline poll; the sparse join has no other
    // mid-level break, and the bail lands where `try_push`'s recovery
    // already covers it.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    match collect {
        // Each candidate's pair goes straight to the level's pair arena and
        // its g parent beside it; `emit_walked` makes the node's products of
        // them, each candidate one pair of one of them.
        Collect::Nodes(Emit { level, out, duplicates_legal, live, .. }) => {
            let start = level.pairs.stored_len();
            let walked_nodes = nodes.end();
            let mut candidates = 0;
            try_for_each_walked(f_level, nodes, |p1, pairs| {
                let base = level.pairs.stored_len();
                node_owners.clear();
                walk.node::<SWAPPED>(pairs, &mut ticker, |entry| {
                    try_push_pair_growing(eng, level.pairs.stored_mut(), candidate_pair(&entry), move |v| grow_by_nodes(eng, v, start, p1, walked_nodes))?;
                    lim.try_push(node_owners, entry.g_parent)
                })?;
                candidates += node_owners.len();
                emit_walked(eng, emit, level, out, p1, base, node_owners, duplicates_legal)
            })?;
            *live = Some(candidates);
            Ok(())
        }
        // The level's one f node is walked in stretches of a
        // `PROJECT_AFTER`th of its pairs, each projected before it.
        Collect::Direct(level) => {
            let mut projection = Projection::new(level);
            try_for_each_walked(f_level, nodes, |_, pairs| {
                for stretch in pairs.chunks(pairs.len().div_ceil(PROJECT_AFTER)) {
                    projection.before(eng, level, stretch.len(), || walked_pairs(f_level, nodes));
                    walk.node::<SWAPPED>(stretch, &mut ticker, |entry| try_push_pair_into(eng, level, candidate_pair(&entry)))?;
                }
                Ok(())
            })
        }
        Collect::Fold(fold) => try_for_each_walked(f_level, nodes, |_, pairs| {
            walk.node::<SWAPPED>(pairs, &mut ticker, |entry| {
                fold.push(candidate_pair(&entry));
                Ok(())
            })
        }),
        Collect::Sum(sum) => try_for_each_walked(f_level, nodes, |_, pairs| {
            walk.node::<SWAPPED>(pairs, &mut ticker, |entry| sum.push(lim, candidate_pair(&entry)))
        }),
    }
}

/// How [`leaf_walk`] finds the candidates of one f node.
pub(super) trait NodeWalk {
    /// Hand every candidate of the f node holding `pairs` to `push`, in the
    /// order of its pairs, the products they meet and the g pairs under each.
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError>;
}

/// What [`leaf_walk`] reads for each f pair: the outer child's live
/// products by f index, g's pairs by outer child, and the inner product.
struct LeafWalk<'a, P> {
    g_by_outer: GroupedView<'a, RevEntry>,
    outer: GroupedView<'a, ProductEntry>,
    inner_product: P,
}

/// [`LeafWalk`] where the inner child is a vtree leaf: g's pairs are grouped
/// by outer child and, within it, by leaf label ([`build_labelled_index`]),
/// so an f pair reads only the g pairs whose label its own label meets. Its
/// inner product is its own label where that is a literal and the g pair's
/// otherwise.
struct LabelWalk<'a> {
    g_by_outer: GroupedView<'a, RevEntry>,
    outer: GroupedView<'a, ProductEntry>,
}

impl NodeWalk for LabelWalk<'_> {
    #[inline(always)]
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        const ONE: u32 = crate::diagram::ONE_LEAF_IDX.0;
        const POS: u32 = crate::diagram::POS_LEAF_IDX.0;
        const NEG: u32 = crate::diagram::NEG_LEAF_IDX.0;
        let offsets = self.g_by_outer.offsets;
        for pair in pairs {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            // The runs of one outer child are Pos, One, Neg: a positive
            // literal meets the first two, a negative one the last two, One
            // all three.
            let first = usize::from(inner1 == NEG);
            let last = 3 - usize::from(inner1 == POS);
            for e in self.outer.bucket(outer1 as usize) {
                let at = LABEL_RUNS * e.g_idx.idx();
                let under = &self.g_by_outer.entries[offsets[at + first] as usize..offsets[at + last] as usize];
                for &RevEntry { parent: g_parent, other: inner2 } in under {
                    let inner_prod = if inner1 == ONE { inner2 } else { inner1 };
                    let (left_prod, right_prod) = orient::<SWAPPED>(inner_prod, e.prod_idx.0);
                    push(ParEntry { g_parent, left_prod, right_prod })?;
                }
                // Charged as the walk of every label under the product was.
                ticker.poll(u64::from(offsets[at + LABEL_RUNS] - offsets[at]))?;
            }
        }
        Ok(())
    }
}

/// [`LeafWalk`] where the inner child is a vtree leaf and no g pair repeats:
/// the outer child's live products that meet an owner, each with the slots
/// of g's owner table ([`build_owner_table`]) for its g node, the g nodes
/// holding it with the labels Pos, One, Neg ([`gather_owned_products`]). An
/// f pair reads, under each product, the slots of the labels its own label
/// meets. `ONES` says some g pair has the label One; without it that slot
/// is never read.
struct OwnerWalk<'a, const ONES: bool> {
    outer: GroupedView<'a, OwnedProduct>,
}

impl<'a, const ONES: bool> OwnerWalk<'a, ONES> {
    /// The walk of a level whose f has `f_outer` outer children: its bounds
    /// of the workspace's grow-only offsets.
    fn of(parts: WalkParts<'a>, f_outer: usize) -> Self {
        OwnerWalk { outer: GroupedView { offsets: &parts.outer.offsets[..=f_outer], entries: parts.owned } }
    }
}

impl<const ONES: bool> NodeWalk for OwnerWalk<'_, ONES> {
    #[inline(always)]
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        const POS: u32 = crate::diagram::POS_LEAF_IDX.0;
        const NEG: u32 = crate::diagram::NEG_LEAF_IDX.0;
        for pair in pairs {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            for e in self.outer.bucket(outer1 as usize) {
                let [pos, one, neg] = e.slots;
                let mut meet = |g_parent: u32, label: u32| {
                    if g_parent == NO_PRODUCT {
                        return Ok(());
                    }
                    let (left_prod, right_prod) = orient::<SWAPPED>(label, e.prod);
                    push(ParEntry { g_parent, left_prod, right_prod })
                };
                // The order of the runs a reverse index by label would
                // hold, Pos, One, Neg; a literal meets One and its own.
                if inner1 != NEG {
                    meet(pos, POS)?;
                }
                if ONES {
                    meet(one, inner1)?;
                }
                if inner1 != POS {
                    meet(neg, NEG)?;
                }
                // Charged as the walk of every g pair under the product was.
                let held = u64::from(pos != NO_PRODUCT) + u64::from(neg != NO_PRODUCT);
                ticker.poll(if ONES { held + u64::from(one != NO_PRODUCT) } else { held })?;
            }
        }
        Ok(())
    }
}

/// [`walk_nodes`] of an [`OwnerWalk`] into the level's nodes, over f's first
/// `f_nodes` nodes as `f_level` reads them ([`FNodes`]): the candidates,
/// products, nodes and clock charges of that walk, in the same order. f's
/// nodes are walked in this loop, not handed one by one to a closure, the
/// clock is charged once a node, and a node's g parent is held as its walk
/// goes while every candidate so far has the same one: the owners are
/// written out only once a second g parent appears, and a node of one g
/// parent is made where its walk ends. The products are `outer`'s by f
/// index; `DENSE` says each outer child has one, read from `dense` by its
/// index instead ([`one_each`]). `BOTH` says no g pair has the label One and
/// each product's g node holds it with both literals: each slot a pair's
/// label meets holds a candidate, and a product's g pairs are two. Returns
/// what the walk made of the level and the live pairs of the nodes it made.
#[expect(clippy::too_many_arguments)]
#[inline(never)]
fn owner_walk_nodes<'f, const SWAPPED: bool, const ONES: bool, const DENSE: bool, const BOTH: bool, N: FNodes<'f>>(
    eng: &Engine,
    node_owners: &mut Vec<u32>,
    emit: &mut EmitScratch,
    f_level: N,
    f_nodes: usize,
    outer: GroupedView<'_, OwnedProduct>,
    dense: Listed<'_>,
    to: Emit<'_>,
) -> Result<(Made, usize), OperationError> {
    const POS: u32 = crate::diagram::POS_LEAF_IDX.0;
    const NEG: u32 = crate::diagram::NEG_LEAF_IDX.0;
    let lim = eng.limits();
    let Emit { level, out, duplicates_legal, .. } = to;
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    let start = level.pairs.stored_len();
    let empty = start == 0 && level.nodes.stored().is_empty();
    // The live pairs of the nodes made: the copied nodes' and then one a
    // candidate.
    let (copied, mut made_pairs) = if DENSE {
        match copy_nodes::<SWAPPED, ONES, N>(eng, &f_level, f_nodes, dense, level, start, out, &mut ticker)? {
            Copied::Level(pairs) => return Ok((Made::F, pairs)),
            Copied::Nodes(made, pairs) if empty && made == f_level.len() => return Ok((Made::CopiedF, pairs)),
            Copied::Nodes(made, pairs) => (made, pairs),
        }
    } else {
        (0, 0)
    };
    // The node at which the walk sizes the level's arenas and products from
    // the share of its nodes before it ([`reserve_walk_ahead`]), none
    // under bounded growth; their lengths and capacities as it begins; and
    // the pair capacity the output-pair meter is charged for where that
    // moved the arena past it, `usize::MAX` elsewhere.
    let walked = f_nodes - copied;
    let after = crate::apply::conjoin::ahead_after_forced().unwrap_or(PROJECT_AFTER);
    let ahead = if walked >= after && !lim.bounded_growth() && !crate::apply::conjoin::ahead_forced_off() {
        copied + walked / after
    } else {
        usize::MAX
    };
    let from = [level.pairs.stored_len(), level.nodes.stored().len(), out.len()];
    let seeds = [level.pairs.capacity(), level.nodes.stored_mut().capacity(), out.capacity()];
    let mut charged = usize::MAX;
    // The level's pair arena, nodes and products, held here while the walk
    // writes past their lengths, and set back into them before a call that
    // takes them and where the walk ends or fails: a candidate, node or
    // product is a store and a step. The arena is held as its start `at`,
    // the next candidate's place `next` and the end of its room
    // `room_end`, within the capacity the meter is charged for; a candidate
    // that finds no room is pushed as the arena's own push would push it,
    // growth and charges included ([`push_past_room`]), and so is a node or
    // product, by the calls the walk made before.
    let arena = level.pairs.stored_mut();
    let mut at = arena.as_mut_ptr();
    // Safety: the length and capacity are within the allocation.
    let (mut next, mut room_end) = unsafe { (at.add(arena.len()), at.add(arena.capacity())) };
    let mut nodes = Tail::of(level.nodes.stored_mut());
    let mut prods = Tail::of(out);
    macro_rules! settle {
        () => {
            // Safety: the slots below each length are written, those of
            // the arena below `next`.
            unsafe {
                level.pairs.stored_mut().set_len(next.offset_from_unsigned(at));
                nodes.settle(level.nodes.stored_mut());
                prods.settle(out);
            }
        };
    }
    macro_rules! retake {
        () => {
            let arena = level.pairs.stored_mut();
            at = arena.as_mut_ptr();
            // Safety: as at the start.
            unsafe { (next, room_end) = (at.add(arena.len()), at.add(arena.capacity().min(charged))) };
            nodes = Tail::of(level.nodes.stored_mut());
            prods = Tail::of(out);
        };
    }
    macro_rules! tri {
        ($result:expr) => {
            match $result {
                Ok(value) => value,
                Err(e) => {
                    settle!();
                    return Err(e);
                }
            }
        };
    }
    for (p1, node) in (copied..f_nodes).zip(f_level.words(copied, f_nodes)) {
        if p1 == ahead {
            settle!();
            charged = reserve_walk_ahead(eng, level, out, from, seeds, p1 - copied, walked)?;
            retake!();
        }
        // Where the node's candidates start.
        let mut first = next;
        // The g parent the node's next candidate has while its candidates
        // so far have one, or two taking turns from the first: then `turn`
        // is the two's exclusive or, which turns `owner` from one to the
        // other at each candidate, and otherwise 0. Once they are any other
        // way, `node_owners` holds every candidate's and `owner` is
        // [`MIXED`], which no candidate's equals: a candidate of the g
        // parent expected costs one comparison.
        let (mut owner, mut turn) = (NO_PRODUCT, 0u32);
        // The clock units of the node's visits; on a `BOTH` level, how far
        // twice its candidates overstate them.
        let mut held = 0u64;
        // Macros, not closures: a closure holding the node's state is
        // called out of line, its state kept in memory.
        macro_rules! push {
            ($g_parent:expr, $label:expr, $prod:expr) => {{
                let g_parent: u32 = $g_parent;
                debug_assert!(g_parent < MIXED, "a g parent is a node index");
                let (left_prod, right_prod) = orient::<SWAPPED>($label, $prod);
                let pair = ChildPair::new(EncodedChildRef::from_raw(left_prod), EncodedChildRef::from_raw(right_prod));
                if next < room_end {
                    // Safety: `next` is below the room's end.
                    unsafe {
                        next.write(pair);
                        next = next.add(1);
                    }
                } else {
                    // Safety: the slots below `next` hold the arena's
                    // pairs and the node's candidates so far, and it is
                    // the room's end.
                    (at, first, next, room_end) =
                        tri!(unsafe { push_past_room(eng, level.pairs.stored_mut(), at, first, next, pair, start, p1, f_nodes, &mut charged) });
                }
                if g_parent == owner {
                    owner ^= turn;
                } else if owner == NO_PRODUCT {
                    owner = g_parent;
                } else if owner == MIXED {
                    tri!(lim.try_push(node_owners, g_parent));
                } else if turn == 0 && next == first.wrapping_add(2) {
                    // The second g parent second: the next candidate's
                    // is the first's.
                    turn = owner ^ g_parent;
                } else {
                    // Safety: both lie in the arena, `next` past `first`.
                    let before = unsafe { next.offset_from_unsigned(first) } - 1;
                    tri!(spill_owners(lim, node_owners, owner, turn, before, g_parent));
                    (owner, turn) = (MIXED, 0);
                }
            }};
        }
        macro_rules! meet {
            ($g_parent:expr, $label:expr, $prod:expr) => {{
                let g_parent: u32 = $g_parent;
                if g_parent != NO_PRODUCT {
                    push!(g_parent, $label, $prod);
                }
            }};
        }
        // A product's candidates with an f pair of label `inner1`, which
        // meets Pos where `pos` and Neg where `neg`: [`OwnerWalk`]'s order,
        // Pos, One, Neg.
        macro_rules! visit {
            ($e:expr, $inner1:expr, $pos:expr, $neg:expr) => {{
                let e: OwnedProduct = $e;
                let [pos, one, neg] = e.slots;
                if $pos {
                    meet!(pos, POS, e.prod);
                }
                if ONES {
                    meet!(one, $inner1, e.prod);
                }
                if $neg {
                    meet!(neg, NEG, e.prod);
                }
                // Charged as the walk of every g pair under the product
                // was; a held slot is below `NO_PRODUCT`, a test the count
                // takes as a carry.
                held += u64::from(pos < NO_PRODUCT) + u64::from(neg < NO_PRODUCT);
                if ONES {
                    held += u64::from(one < NO_PRODUCT);
                }
            }};
        }
        macro_rules! each {
            ($outer1:expr, $inner1:expr, $pos:expr, $neg:expr) => {{
                if DENSE {
                    visit!(dense.entry($outer1), $inner1, $pos, $neg);
                } else {
                    // Most buckets hold one product: it is visited without
                    // the loop's tests.
                    let bucket = owned_bucket(outer, $outer1);
                    if let [e] = bucket {
                        visit!(*e, $inner1, $pos, $neg);
                    } else {
                        for e in bucket {
                            visit!(*e, $inner1, $pos, $neg);
                        }
                    }
                }
            }};
        }
        // The pair's label is tested once, not at each product: a loop for
        // each literal, and one for any other label, which meets all three.
        for pair in f_level.of(&node) {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            if BOTH {
                // Every slot read holds a candidate, and a product's
                // visit is charged two units: a literal's pair meets one
                // candidate a product, and any other label's two.
                let bucket = owned_bucket(outer, outer1);
                if let &[OwnedProduct { prod, slots: [pos, _, neg] }] = bucket {
                    // A bucket of one product, as most are, without the
                    // loops' tests.
                    if inner1 == POS {
                        push!(pos, POS, prod);
                    } else if inner1 == NEG {
                        push!(neg, NEG, prod);
                    } else {
                        held += 2;
                        push!(pos, POS, prod);
                        push!(neg, NEG, prod);
                    }
                } else if inner1 == POS {
                    for &OwnedProduct { prod, slots: [pos, _, _] } in bucket {
                        push!(pos, POS, prod);
                    }
                } else if inner1 == NEG {
                    for &OwnedProduct { prod, slots: [_, _, neg] } in bucket {
                        push!(neg, NEG, prod);
                    }
                } else {
                    held += 2 * bucket.len() as u64;
                    for &OwnedProduct { prod, slots: [pos, _, neg] } in bucket {
                        push!(pos, POS, prod);
                        push!(neg, NEG, prod);
                    }
                }
                continue;
            }
            if inner1 == POS {
                each!(outer1, POS, true, false);
            } else if inner1 == NEG {
                each!(outer1, NEG, false, true);
            } else {
                each!(outer1, inner1, true, true);
            }
        }
        // Safety: both lie in the arena, `next` not before `first`, and
        // `first` not before `at`.
        let (count, base) = unsafe { (next.offset_from_unsigned(first), first.offset_from_unsigned(at)) };
        tri!(ticker.poll(if BOTH { 2 * count as u64 - held } else { held }));
        made_pairs += count;
        if count == 0 {
            continue;
        }
        if owner == MIXED {
            settle!();
            emit_mixed_apart(eng, emit, level, out, p1, base, node_owners, duplicates_legal)?;
            retake!();
            continue;
        }
        if turn != 0 {
            if count == 2 && prods.room() >= 2 && nodes.room() >= 2 {
                // [`emit_alternating`] of two candidates, where neither
                // vector grows: each candidate its own g parent's node.
                // Safety: the candidates are at `first`, which the arena
                // then drops.
                let pairs = unsafe { [first.read(), first.add(1).read()] };
                next = first;
                for (g_parent, pair) in [owner, owner ^ turn].into_iter().zip(pairs) {
                    let product = ProductEntry {
                        f_idx: FNodeIdx(p1 as u32),
                        g_idx: GNodeIdx(g_parent),
                        prod_idx: ProductNodeIdx(prods.len as u32),
                    };
                    // Safety: there is room.
                    unsafe {
                        prods.put(product);
                        nodes.put(EncodedNode::inline(pair));
                    }
                }
            } else if matches!(count, 4 | 6 | 8) && prods.room() >= 2 && nodes.room() >= 2 && base + count < 1 << 31 {
                // [`emit_alternating`] of four, six or eight candidates,
                // where neither vector grows: the first g parent's half
                // moved before the second's, in place, and each half its
                // g parent's node.
                // Safety: the candidates are the `count` pairs at `first`.
                unsafe {
                    match count {
                        4 => split_turns::<4>(first),
                        6 => split_turns::<6>(first),
                        _ => split_turns::<8>(first),
                    }
                }
                let half = count / 2;
                for (k, g_parent) in [owner, owner ^ turn].into_iter().enumerate() {
                    let product = ProductEntry {
                        f_idx: FNodeIdx(p1 as u32),
                        g_idx: GNodeIdx(g_parent),
                        prod_idx: ProductNodeIdx(prods.len as u32),
                    };
                    // Safety: there is room.
                    unsafe {
                        prods.put(product);
                        nodes.put(EncodedNode::multi_pair((base + k * half) as u32, half as u32));
                    }
                }
                // As in `emit_alternating`: canonical children make each
                // node's pairs distinct unless pair lists are multisets.
                #[cfg(debug_assertions)]
                {
                    settle!();
                    debug_assert!(
                        duplicates_legal || {
                            let made = level.nodes().len();
                            (made - 2..made).all(|i| {
                                let mut seen = std::collections::HashSet::new();
                                level.pairs_iter_of_idx(i).all(|p| seen.insert(p))
                            })
                        },
                        "sparse apply: duplicate pair emitted — canonicity violated"
                    );
                }
            } else {
                settle!();
                if matches!(count, 2 | 4 | 6 | 8) {
                    emit_alternating(eng, level, out, p1, base, owner, owner ^ turn, count, duplicates_legal)?;
                } else {
                    emit_turns(eng, emit, level, out, p1, base, owner, turn, node_owners, duplicates_legal)?;
                }
                retake!();
            }
            continue;
        }
        let product = ProductEntry {
            f_idx: FNodeIdx(p1 as u32),
            g_idx: GNodeIdx(owner),
            prod_idx: ProductNodeIdx(prods.len as u32),
        };
        if prods.room() > 0 {
            // Safety: there is room.
            unsafe { prods.put(product) };
        } else {
            settle!();
            lim.try_push(out, product)?;
            retake!();
        }
        if count == 1 {
            // Safety: the node's candidate is at `first`, which the arena
            // then drops.
            let pair = unsafe { first.read() };
            next = first;
            if nodes.room() > 0 {
                // Safety: there is room.
                unsafe { nodes.put(EncodedNode::inline(pair)) };
            } else {
                settle!();
                emit_single_pair(eng, level, pair)?;
                retake!();
            }
        } else {
            if nodes.room() > 0 && (base | count) < 1 << 31 {
                // Safety: there is room.
                unsafe { nodes.put(EncodedNode::multi_pair(base as u32, count as u32)) };
            } else {
                settle!();
                reserve_room(lim, level.nodes.stored_mut(), 1)?;
                level.try_push_multi_by_range(base, count).map_err(|_| OperationError::OverBudget)?;
                retake!();
            }
            // As in `emit_walked`: canonical children make a node's pairs
            // distinct unless pair lists are multisets.
            #[cfg(debug_assertions)]
            {
                settle!();
                debug_assert!(
                    duplicates_legal || {
                        let mut seen = std::collections::HashSet::new();
                        level.pairs_iter_of_idx(level.nodes().len() - 1).all(|p| seen.insert(p))
                    },
                    "sparse apply: duplicate pair emitted — canonicity violated"
                );
            }
        }
    }
    settle!();
    debug_assert!(charged == usize::MAX || level.pairs.stored_len() <= charged, "a walk's pairs stay within the capacity the meter is charged for");
    if ahead != usize::MAX {
        trim_walk(level, out, seeds, charged);
    }
    Ok((Made::Written, made_pairs))
}

/// The `N` pairs at `first`, which take turns between two g parents from
/// the first, put the first parent's before the second's, each half in its
/// order: [`emit_alternating`]'s split, on the walk's arena in place.
///
/// # Safety
///
/// `first` points at `N` initialized pairs.
#[inline(always)]
unsafe fn split_turns<const N: usize>(first: *mut ChildPair) {
    // Safety: as the caller's.
    let read: [ChildPair; N] = unsafe { first.cast::<[ChildPair; N]>().read_unaligned() };
    for i in 0..N / 2 {
        // Safety: both places are among the `N`.
        unsafe {
            first.add(i).write(read[2 * i]);
            first.add(N / 2 + i).write(read[2 * i + 1]);
        }
    }
}

/// Whether [`owner_walk_nodes`] reads f's level: one that stores its pairs,
/// or an implicit one of one pair a node ([`ImpliedNodes`]).
fn walkable(f_level: &TddLevel) -> bool {
    f_level.pairs.implicit().is_none_or(|d| d.pairs_per_node() == 1)
}

/// [`owner_walk_nodes`] over f's nodes, stored or implied ([`walkable`]).
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn walk_owned<const SWAPPED: bool, const ONES: bool, const DENSE: bool, const BOTH: bool>(
    eng: &Engine,
    node_owners: &mut Vec<u32>,
    emit: &mut EmitScratch,
    f_level: &TddLevel,
    f_nodes: usize,
    outer: GroupedView<'_, OwnedProduct>,
    dense: Listed<'_>,
    to: Emit<'_>,
) -> Result<Made, OperationError> {
    // The walk returns the live pairs of what it made, and they are set
    // here: a place to set them, held across the walk's loop, would take
    // one of its registers.
    let Emit { level, out, duplicates_legal, thresholds, live } = to;
    let to = Emit { level, out, duplicates_legal, thresholds, live: &mut *live };
    let (made, pairs) = match f_level.stored() {
        Some(stored) => owner_walk_nodes::<SWAPPED, ONES, DENSE, BOTH, _>(eng, node_owners, emit, stored, f_nodes, outer, dense, to),
        None => {
            let nodes = ImpliedNodes::of(f_level).expect("an implicit level the walk reads has one pair a node");
            crate::apply::conjoin::note_implied_walk(DENSE);
            owner_walk_nodes::<SWAPPED, ONES, DENSE, BOTH, _>(eng, node_owners, emit, nodes, f_nodes, outer, dense, to)
        }
    }?;
    *live = Some(pairs);
    Ok(made)
}

/// f's nodes as [`owner_walk_nodes`] and [`copy_nodes`] read them, from the
/// first in increasing order: a stored level's words and pairs, or those
/// of an implicit level of one pair a node ([`ImpliedNodes`]).
trait FNodes<'a> {
    /// The words of nodes `from..to`, in order.
    type Words: Iterator<Item = EncodedNode>;
    /// The level's nodes.
    fn len(&self) -> usize;
    /// The words of nodes `from..to`, below [`len`](Self::len), in order.
    fn words(&self, from: usize, to: usize) -> Self::Words;
    /// The pairs of `node`, a word [`word`](Self::word) gave.
    fn of<'n>(&self, node: &'n EncodedNode) -> &'n [ChildPair]
    where
        'a: 'n;
    /// The level's stored pairs, which [`copy_nodes`] can lend; `None` on
    /// an implicit level.
    fn stored(&self) -> Option<crate::diagram::StoredPairs<'a>>;
}

impl<'a> FNodes<'a> for crate::diagram::StoredPairs<'a> {
    type Words = std::iter::Copied<std::slice::Iter<'a, EncodedNode>>;

    #[inline(always)]
    fn len(&self) -> usize {
        self.nodes().len()
    }

    #[inline(always)]
    fn words(&self, from: usize, to: usize) -> Self::Words {
        self.nodes()[from..to].iter().copied()
    }

    #[inline(always)]
    fn of<'n>(&self, node: &'n EncodedNode) -> &'n [ChildPair]
    where
        'a: 'n,
    {
        crate::diagram::StoredPairs::of(*self, node)
    }

    #[inline(always)]
    fn stored(&self) -> Option<crate::diagram::StoredPairs<'a>> {
        Some(*self)
    }
}

/// The nodes of an implicit level of one pair a node, each its pair
/// inline, read off its description in increasing order by a cursor that
/// steps each node's pair on from the last one's.
struct ImpliedNodes<'a> {
    nodes: usize,
    cursor: crate::diagram::NodeCursor<'a>,
}

/// The words of nodes `at..end` of [`ImpliedNodes`], in order.
struct ImpliedWords<'a> {
    cursor: crate::diagram::NodeCursor<'a>,
    at: usize,
    end: usize,
}

impl Iterator for ImpliedWords<'_> {
    type Item = EncodedNode;

    #[inline(always)]
    fn next(&mut self) -> Option<EncodedNode> {
        if self.at == self.end {
            return None;
        }
        let (l, r) = self.cursor.first_of(self.at);
        self.at += 1;
        Some(EncodedNode::inline(ChildPair::new(EncodedChildRef::from_raw(l as u32), EncodedChildRef::from_raw(r as u32))))
    }
}

impl<'a> ImpliedNodes<'a> {
    /// The nodes `level` implies, where it is implicit with one pair a node.
    fn of(level: &'a TddLevel) -> Option<Self> {
        let d = level.pairs.implicit().filter(|d| d.pairs_per_node() == 1)?;
        Some(ImpliedNodes { nodes: d.nodes(), cursor: d.cursor() })
    }
}

impl<'a> FNodes<'a> for ImpliedNodes<'a> {
    type Words = ImpliedWords<'a>;

    #[inline(always)]
    fn len(&self) -> usize {
        self.nodes
    }

    #[inline(always)]
    fn words(&self, from: usize, to: usize) -> ImpliedWords<'a> {
        assert!(from <= to && to <= self.nodes, "nodes of the level");
        ImpliedWords { cursor: self.cursor.clone(), at: from, end: to }
    }

    #[inline(always)]
    fn of<'n>(&self, node: &'n EncodedNode) -> &'n [ChildPair]
    where
        'a: 'n,
    {
        crate::diagram::inline_pair(node)
    }

    #[inline(always)]
    fn stored(&self) -> Option<crate::diagram::StoredPairs<'a>> {
        None
    }
}

/// The owner [`owner_walk_nodes`] holds for a node whose candidates' g
/// parents are other than one, or two taking turns: no g parent's, a g
/// parent being a node index, which a child reference holds below 2^31.
const MIXED: u32 = NO_PRODUCT - 1;

/// Outer child `outer`'s products in `view`, the walk's view of
/// [`gather_owned_products`]'s buckets, read without bounds tests.
#[inline(always)]
fn owned_bucket(view: GroupedView<'_, OwnedProduct>, outer: u32) -> &[OwnedProduct] {
    let o = outer as usize;
    debug_assert!(o + 1 < view.offsets.len(), "an f pair's outer child is one of f's");
    // Safety: a structural reference indexes its child level, over whose
    // width, one less than the offsets' length, the products were
    // gathered; the gathered offsets do not decrease, and the last is the
    // entries' length.
    unsafe {
        let (start, end) = (*view.offsets.get_unchecked(o) as usize, *view.offsets.get_unchecked(o + 1) as usize);
        debug_assert!(start <= end && end <= view.entries.len(), "a bucket lies within the entries");
        view.entries.get_unchecked(start..end)
    }
}

/// Push `pair` onto `arena` for [`owner_walk_nodes`], whose writes past the
/// arena's length, from its start `at`, have reached the end of its room at
/// `next`, the node's candidates starting at `first`: the length set to
/// `next`'s index and the pair pushed as [`try_push_pair_growing`] pushes
/// it, the arena grown first as [`grow_by_nodes`] projects at node `p1` of
/// `nodes` from `start`. Where a reservation ahead moved the arena past
/// `charged`, the capacity the output-pair meter is charged for, the room
/// ends there too, the meter is charged there as the arena would have
/// grown ([`charge_walk_growth`]), and the arena doubles where it fills.
/// Returns the arena's start, the node's
/// first candidate's place, the next one's and the room's end, for the
/// writes that follow.
///
/// # Safety
///
/// `at` is `arena`'s start, `first` and then `next` lie in its allocation,
/// in that order, and the slots below `next` are initialized.
#[expect(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
unsafe fn push_past_room(
    eng: &Engine,
    arena: &mut Vec<ChildPair>,
    at: *mut ChildPair,
    first: *mut ChildPair,
    next: *mut ChildPair,
    pair: ChildPair,
    start: usize,
    p1: usize,
    nodes: usize,
    charged: &mut usize,
) -> Result<PlacesPastRoom, OperationError> {
    // Safety: as the caller's.
    let (base, len) = unsafe { (first.offset_from_unsigned(at), next.offset_from_unsigned(at)) };
    // Safety: as the caller's.
    unsafe { arena.set_len(len) };
    if *charged == usize::MAX {
        try_push_pair_growing(eng, arena, pair, |v| grow_by_nodes(eng, v, start, p1, nodes))?;
    } else {
        // The arena was reserved ahead: the meter is charged where its
        // room ends, and the arena doubles where it fills.
        if len == *charged {
            *charged = charge_walk_growth(eng.limits(), *charged, start, p1, nodes);
        }
        if len == arena.capacity() {
            crate::apply::conjoin::note_pairs_grown();
            eng.limits().try_push(arena, pair)?;
        } else {
            arena.push(pair);
        }
    }
    let at = arena.as_mut_ptr();
    let end = arena.capacity().min(*charged);
    // Safety: `base` and the length are within the capacity.
    unsafe { Ok((at, at.add(base), at.add(arena.len()), at.add(end))) }
}

/// The capacity a walk's pair arena full at `charged` pairs, the
/// capacity the output-pair meter is charged for, would grow to as
/// [`push_past_room`] grows it at a candidate of node `p1` of the walk's
/// `nodes`, which began at `start` pairs ([`grow_by_nodes`], or doubling),
/// the meter charged for it: an arena reserved ahead
/// ([`reserve_walk_ahead`]) grows on its own.
#[cold]
#[inline(never)]
fn charge_walk_growth(lim: &crate::limits::Limits, charged: usize, start: usize, p1: usize, nodes: usize) -> usize {
    let grown = match projected_growth(lim, charged.saturating_sub(start), p1, nodes, 1, 0) {
        Some(want) => charged + want,
        None => crate::apply::conjoin::budget::doubled_pairs_capacity(charged),
    };
    lim.charge_output_pairs(grown - charged);
    grown
}

/// Move a level's pair and node arenas and its products, at lengths
/// `from` where an owner walk of `walked` f nodes began and `done` of them
/// walked, to fresh ones of the lengths those nodes project, an eighth
/// more, copying only what they hold. Grown by doubling or projection from
/// their operand-sized seeds where they fill instead, each was copied
/// whole at its growth, and those copies took a twentieth of the cycles of
/// the programs whose sparse levels are large. Each is at most
/// [`WALK_AHEAD_FACTOR`] times its capacity `seeds` as the walk began, and
/// the three together at most half the memory left; one that fills after
/// doubles. Returns the capacity the output-pair meter is charged for
/// where the pair arena moved past it, which the walk then charges as the
/// arena would have grown ([`charge_walk_growth`]), and `usize::MAX` where
/// it did not.
#[cold]
#[inline(never)]
fn reserve_walk_ahead(
    eng: &Engine,
    level: &mut TddLevel,
    out: &mut Vec<ProductEntry>,
    from: [usize; 3],
    seeds: [usize; 3],
    done: usize,
    walked: usize,
) -> Result<usize, OperationError> {
    const SIZES: [usize; 3] = [size_of::<ChildPair>(), size_of::<EncodedNode>(), size_of::<ProductEntry>()];
    let lim = eng.limits();
    let lens = [level.pairs.stored_len(), level.nodes.stored().len(), out.len()];
    let caps = [level.pairs.capacity(), level.nodes.stored_mut().capacity(), out.capacity()];
    let mut to = [0; 3];
    let mut bytes = 0u64;
    for i in 0..3 {
        let projected = lens[i].saturating_sub(from[i]) as u128 * walked as u128 / done as u128;
        let most = WALK_AHEAD_FACTOR.saturating_mul(seeds[i]) as u128;
        to[i] = from[i] + (projected + projected / 8).min(most) as usize;
        bytes += (to[i].saturating_sub(caps[i]) * SIZES[i]) as u64;
    }
    if bytes == 0 || bytes > lim.headroom() / 2 {
        return Ok(usize::MAX);
    }
    if to[0] > caps[0] {
        crate::apply::conjoin::note_pairs_grown();
    }
    crate::apply::conjoin::note_ahead(true);
    lim.reserve_exact_moving(level.pairs.stored_mut(), to[0].saturating_sub(lens[0]))?;
    lim.reserve_exact_moving(level.nodes.stored_mut(), to[1].saturating_sub(lens[1]))?;
    lim.reserve_exact_moving(out, to[2].saturating_sub(lens[2]))?;
    Ok(if to[0] > caps[0] { caps[0] } else { usize::MAX })
}

/// The most [`reserve_walk_ahead`] reserves a vector for, as a multiple of
/// its capacity as the walk began.
const WALK_AHEAD_FACTOR: usize = 8;

/// A level's pair and node arenas and its products, once an owner walk
/// that reserved them ahead ([`reserve_walk_ahead`]) is done, given back
/// the room the reservation left past what growth would have given them:
/// the pair arena past `charged`, the capacity the output-pair meter is
/// charged for, where the reservation moved it, and the nodes and products
/// past their doubling from their capacities `seeds` as the walk began. A
/// projection the walk's first nodes overstate would otherwise stay with
/// the level for as long as it is held.
#[inline]
fn trim_walk(level: &mut TddLevel, out: &mut Vec<ProductEntry>, seeds: [usize; 3], charged: usize) {
    use crate::apply::conjoin::budget::{doubled_to, trim_to};
    if charged != usize::MAX {
        trim_to(level.pairs.stored_mut(), charged);
    }
    let nodes = level.nodes.stored_mut();
    trim_to(nodes, doubled_to(seeds[1], nodes.len()));
    trim_to(out, doubled_to(seeds[2], out.len()));
}

/// What [`push_past_room`] returns: the arena's start, the node's first
/// candidate's place, the next one's and the room's end.
type PlacesPastRoom = (*mut ChildPair, *mut ChildPair, *mut ChildPair, *mut ChildPair);

/// A vector's buffer, length and capacity, held apart from it while a loop
/// writes past its length ([`owner_walk_nodes`]): a write is a store and a
/// step, and the vector's length is set once the loop is done with it.
struct Tail<T> {
    at: *mut T,
    len: usize,
    cap: usize,
}

impl<T> Tail<T> {
    /// `v`'s buffer, length and capacity.
    #[inline(always)]
    fn of(v: &mut Vec<T>) -> Self {
        Tail { at: v.as_mut_ptr(), len: v.len(), cap: v.capacity() }
    }

    /// The writes left before the capacity.
    #[inline(always)]
    fn room(&self) -> usize {
        self.cap - self.len
    }

    /// Write `x` at the length and step past it.
    ///
    /// # Safety
    ///
    /// There is room.
    #[inline(always)]
    unsafe fn put(&mut self, x: T) {
        debug_assert!(self.len < self.cap, "a tail is written within its capacity");
        // Safety: as the caller's.
        unsafe { self.at.add(self.len).write(x) };
        self.len += 1;
    }

    /// Set `v`'s length to this one's.
    ///
    /// # Safety
    ///
    /// `v` is the vector this was taken from, its buffer unmoved, and its
    /// slots below this length are written.
    #[inline(always)]
    unsafe fn settle(&self, v: &mut Vec<T>) {
        debug_assert!(v.as_mut_ptr() == self.at && v.capacity() == self.cap, "a tail settles into its own vector");
        // Safety: as the caller's.
        unsafe { v.set_len(self.len) };
    }
}

/// Whether `list` holds one product for each of `width` f indices, in
/// order.
fn one_each(list: &[ProductEntry], width: usize) -> bool {
    list.len() == width && list.iter().enumerate().all(|(i, e)| e.f_idx.idx() == i)
}

/// The products of a list with one product per f index, in order
/// ([`one_each`]), each with its slots, read once from the owner table
/// ([`gather_listed_products`]): the walk reads a pair's product and slots
/// in one read, at its outer child.
#[derive(Clone, Copy)]
struct Listed<'a> {
    owned: &'a [OwnedProduct],
    /// No g pair has the label One, and each product is numbered as its f
    /// index and its g node holds it with both literals: an f pair of
    /// either literal meets one g pair, of its own label, under the
    /// product numbered as its outer child.
    plain: bool,
}

impl Listed<'_> {
    /// Outer child `outer`'s one product with its slots.
    #[inline(always)]
    fn entry(self, outer: u32) -> OwnedProduct {
        self.owned[outer as usize]
    }
}

/// The g parent of the one candidate of two g pairs' parents `a` and `b`,
/// or `NO_PRODUCT` where both or neither is there.
#[inline(always)]
fn sole(a: u32, b: u32) -> u32 {
    if a != NO_PRODUCT && b != NO_PRODUCT { NO_PRODUCT } else { a.min(b) }
}

/// What f pair `pair` meets on a level of [`copy_nodes`]: the g parent of
/// the one g pair it meets with its own leaf label, under the product
/// numbered as its outer child (`NO_PRODUCT` where it meets another or
/// none), and above it the clock units the walk charges for its visit.
#[inline(always)]
fn copy_code<const SWAPPED: bool, const ONES: bool>(pair: ChildPair, entries: Listed<'_>) -> u64 {
    const POS: u32 = crate::diagram::POS_LEAF_IDX.0;
    const NEG: u32 = crate::diagram::NEG_LEAF_IDX.0;
    let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
    let e = entries.entry(outer1);
    let [pos, one, neg] = e.slots;
    let mut held = u64::from(pos < NO_PRODUCT) + u64::from(neg < NO_PRODUCT);
    if ONES {
        held += u64::from(one < NO_PRODUCT);
    }
    // The walk's meets of the pair's label: Pos meets Pos and One, Neg
    // meets One and Neg, both under their own label; One meets all three,
    // under its own only in One's.
    let one = if ONES { one } else { NO_PRODUCT };
    let g = match inner1 {
        POS => sole(pos, one),
        NEG => sole(one, neg),
        _ if pos == NO_PRODUCT && neg == NO_PRODUCT => one,
        _ => NO_PRODUCT,
    };
    let g = if e.prod == outer1 { g } else { NO_PRODUCT };
    held << 32 | u64::from(g)
}

/// What [`copy_nodes`] made of the level, with the live pairs of the nodes
/// it made.
enum Copied {
    /// The level's first nodes, this many, written as the walk writes them.
    Nodes(usize, usize),
    /// Every node's product, and no node or pair written: the level is f's,
    /// node for node and pair for pair ([`Made::F`]).
    Level(usize),
}

/// The nodes [`copy_nodes`] takes while nothing is written, from the first,
/// the level's arena holding `n` of f's pairs and the capacities of its
/// arena and nodes `room`: each node one pair or a range at the next of f's
/// arena pairs, within both capacities, copied and of a size the node words
/// hold. Each node's product is listed and the clock charged as the walk
/// lists and charges them. It stops, having done nothing for it, at the
/// first node that is not so, which [`copy_nodes`] takes from the start.
/// Returns how many it took, the arena pairs they hold with `n`'s, and
/// the pairs they hold.
///
/// With `PLAIN` the entries are [`Listed::plain`]: a node's units are two a
/// pair, and a pair of either literal is copied by the slot of its label.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn take_lent<const SWAPPED: bool, const ONES: bool, const PLAIN: bool>(
    lim: &crate::limits::Limits,
    stored: crate::diagram::StoredPairs<'_>,
    f_nodes: usize,
    entries: Listed<'_>,
    mut n: usize,
    room: (usize, usize),
    out: &mut Vec<ProductEntry>,
    ticker: &mut crate::limits::PollGate<'_>,
) -> Result<(usize, usize, usize), OperationError> {
    use crate::diagram::NodeKind;
    const POS: u32 = crate::diagram::POS_LEAF_IDX.0;
    debug_assert!(!PLAIN || (entries.plain && !ONES), "plain entries");
    // A pair's g parent, and above it its units.
    let code = |pair: ChildPair| -> u64 {
        if PLAIN {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            // The labels One, Pos, Neg (0, 1, 2) to their slots 1, 0, 2,
            // with no branch: a One pair meets all three of g's labels, and
            // reads the One slot, empty on a plain level. The slot is read
            // in place, not from a copy of the entry.
            let slot = (inner1 ^ u32::from(inner1 <= POS)) as usize;
            match entries.owned.get(outer1 as usize) {
                Some(e) if slot < OWNER_SLOTS => u64::from(e.slots[slot]),
                _ => u64::from(NO_PRODUCT),
            }
        } else {
            copy_code::<SWAPPED, ONES>(pair, entries)
        }
    };
    let (arena_room, node_room) = room;
    let f_arena = stored.arena();
    // A node's pairs end within the arena's capacity and below 2^31, where
    // the node words hold their place, and, where in the arena, within f's
    // arena: one test bounds a node's end for all three.
    let pair_room = arena_room.min((1 << 31) - 1);
    let range_room = pair_room.min(f_arena.len());
    let mut prods = Tail::of(out);
    let (mut taken, mut pairs) = (0, 0);
    for node in &stored.nodes()[..f_nodes.min(node_room)] {
        // The node's code, its pairs, and its pairs in the arena.
        let (code, count, end) = match node.kind() {
            NodeKind::Inline(pair) if n < pair_room => (code(pair), 1, n),
            NodeKind::Multi { start, len } if start as usize == n && len != 0 && n + len as usize <= range_room => {
                let end = n + len as usize;
                // Safety: `end` is within f's arena (`range_room`) and the
                // node holds a pair.
                let (first, rest) = unsafe { f_arena.get_unchecked(n..end).split_first().unwrap_unchecked() };
                let mut sum = code(*first);
                let mut copies = true;
                for &pair in rest {
                    let next = code(pair);
                    copies &= next as u32 == sum as u32;
                    if !PLAIN {
                        sum += next & !u64::from(u32::MAX);
                    }
                }
                (if copies { sum } else { u64::from(NO_PRODUCT) }, len as usize, end)
            }
            _ => break,
        };
        let owner = code as u32;
        if owner == NO_PRODUCT {
            break;
        }
        let poll = ticker.poll(if PLAIN { 2 * count as u64 } else { code >> 32 });
        if poll.is_err() {
            // Safety: the slots below its length are written.
            unsafe { prods.settle(out) };
            poll?;
        }
        let product = ProductEntry {
            f_idx: FNodeIdx(taken as u32),
            g_idx: GNodeIdx(owner),
            prod_idx: ProductNodeIdx(prods.len as u32),
        };
        if prods.room() > 0 {
            // Safety: there is room.
            unsafe { prods.put(product) };
        } else {
            // Safety: as above.
            unsafe { prods.settle(out) };
            lim.try_push(out, product)?;
            prods = Tail::of(out);
        }
        (n, taken, pairs) = (end, taken + 1, pairs + count);
    }
    // Safety: as above.
    unsafe { prods.settle(out) };
    Ok((taken, n, pairs))
}

/// The first of [`owner_walk_nodes`] on a level where each outer child of f
/// has one product (`DENSE`), its arena holding `start` pairs before the
/// walk: f's nodes from the first for as long as each is its own
/// product's node. That holds of a node each of whose pairs meets one g
/// pair, of the pair's own leaf label and under the product numbered as
/// the pair's outer child, all of one g parent ([`copy_code`]): its
/// candidates are its pairs, in order. Each such node's product, node and
/// pairs are made, and the clock charged and the arena grown, as the walk
/// makes, charges and grows them. Return how many nodes it made, or
/// [`Copied::Level`] where they are all of f's nodes and the level would
/// hold f's nodes and arena as f holds them, which only a level f stores
/// can hold, with the pairs those nodes hold.
#[expect(clippy::too_many_arguments)]
#[inline(never)]
fn copy_nodes<'f, const SWAPPED: bool, const ONES: bool, N: FNodes<'f>>(
    eng: &Engine,
    f_level: &N,
    f_nodes: usize,
    entries: Listed<'_>,
    level: &mut TddLevel,
    start: usize,
    out: &mut Vec<ProductEntry>,
    ticker: &mut crate::limits::PollGate<'_>,
) -> Result<Copied, OperationError> {
    use crate::diagram::NodeKind;
    let lim = eng.limits();
    let (arena, nodes) = (level.pairs.stored_mut(), level.nodes.stored_mut());
    // While each node made so far is f's node as f stores it, its pairs at
    // f's arena's start and in order, and the walk's pushes of them would
    // grow neither the level's nodes nor its arena, nothing is written:
    // `unwritten` counts the pairs those nodes hold. They are written from
    // f's at the first node that is not so; where none is, the level is f's.
    // f's nodes are lent only where it stores them.
    let lendable = f_level.stored();
    let mut unwritten = (lendable.is_some() && start == 0 && nodes.is_empty() && !crate::apply::conjoin::lend_forced_off()).then_some(0usize);
    let write = |arena: &mut Vec<ChildPair>, nodes: &mut Vec<crate::diagram::EncodedNode>, made: usize, pairs: usize| {
        // Within both capacities, as the nodes were taken: nothing grows.
        let stored = lendable.expect("nodes are unwritten only where f stores them");
        arena.extend_from_slice(&stored.arena()[..pairs]);
        nodes.extend_from_slice(&stored.nodes()[..made]);
    };
    let (mut taken, mut made_pairs) = (0, 0);
    if let (Some(n), Some(stored)) = (unwritten, lendable) {
        let lent = if entries.plain {
            take_lent::<SWAPPED, ONES, true>(lim, stored, f_nodes, entries, n, (arena.capacity(), nodes.capacity()), out, ticker)
        } else {
            take_lent::<SWAPPED, ONES, false>(lim, stored, f_nodes, entries, n, (arena.capacity(), nodes.capacity()), out, ticker)
        };
        let (lent, n, pairs) = lent?;
        (taken, unwritten, made_pairs) = (lent, Some(n), pairs);
    }
    for (p1, word) in (taken..f_nodes).zip(f_level.words(taken, f_nodes)) {
        let node = &word;
        let pairs = f_level.of(node);
        let Some((first, rest)) = pairs.split_first() else {
            if let Some(n) = unwritten {
                write(arena, nodes, p1, n);
            }
            return Ok(Copied::Nodes(p1, made_pairs));
        };
        let code = copy_code::<SWAPPED, ONES>(*first, entries);
        let (owner, mut held) = (code as u32, code >> 32);
        let mut copies = owner != NO_PRODUCT;
        for &pair in rest {
            let code = copy_code::<SWAPPED, ONES>(pair, entries);
            copies &= code as u32 == owner;
            held += code >> 32;
        }
        let (base, count) = (unwritten.unwrap_or(arena.len()), pairs.len());
        // A node too far into the arena for its words is the walk's to make;
        // their sum bounds both.
        if !copies || base + count >= 1 << 31 {
            if let Some(n) = unwritten {
                write(arena, nodes, p1, n);
            }
            return Ok(Copied::Nodes(p1, made_pairs));
        }
        ticker.poll(held)?;
        let product = ProductEntry {
            f_idx: FNodeIdx(p1 as u32),
            g_idx: GNodeIdx(owner),
            prod_idx: ProductNodeIdx(out.len() as u32),
        };
        if out.len() < out.capacity() {
            out.push(product);
        } else {
            lim.try_push(out, product)?;
        }
        made_pairs += count;
        if let Some(n) = unwritten {
            // A node of one pair is made only where the arena has room for
            // the push the walk takes back, and one of more where it has
            // room for all of them.
            let own = match node.kind() {
                NodeKind::Inline(_) => n < arena.capacity(),
                NodeKind::Multi { start, .. } => start as usize == n && n + count <= arena.capacity(),
                NodeKind::MultiRanged(_) => false,
            };
            if own && p1 < nodes.capacity() {
                unwritten = Some(if count == 1 { n } else { n + count });
                continue;
            }
            write(arena, nodes, p1, n);
            unwritten = None;
        }
        // The walk's pushes of the candidates: a node of one pair takes its
        // pair back out of the arena, its push having grown it if full, and
        // left it as it was if not.
        let node = if count == 1 {
            if arena.len() == arena.capacity() {
                try_push_pair_growing(eng, arena, *first, |v| grow_by_nodes(eng, v, start, p1, f_nodes))?;
                arena.pop();
            }
            *node
        } else {
            for &pair in pairs {
                try_push_pair_growing(eng, arena, pair, |v| grow_by_nodes(eng, v, start, p1, f_nodes))?;
            }
            crate::diagram::EncodedNode::multi_pair(base as u32, count as u32)
        };
        if nodes.len() < nodes.capacity() {
            nodes.push(node);
        } else {
            reserve_room(lim, nodes, 1)?;
            nodes.push(node);
        }
    }
    match (unwritten, lendable) {
        (Some(n), Some(stored)) if n == stored.arena().len() && f_nodes == stored.nodes().len() => {
            crate::apply::conjoin::note_lent();
            Ok(Copied::Level(made_pairs))
        }
        (Some(n), _) => {
            write(arena, nodes, f_nodes, n);
            Ok(Copied::Nodes(f_nodes, made_pairs))
        }
        (None, _) => Ok(Copied::Nodes(f_nodes, made_pairs)),
    }
}

/// [`emit_mixed`] out of line, for [`owner_walk_nodes`], whose loop keeps
/// the node of one g parent's path to itself.
#[expect(clippy::too_many_arguments)]
#[inline(never)]
fn emit_mixed_apart(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    owners: &mut [u32],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    emit_mixed(eng, emit, level, pl_output, p1, base, owners, duplicates_legal)
}

/// [`emit_mixed`] out of line on the candidates at `base` of a node of
/// [`owner_walk_nodes`] whose g parents take turns, `owner` the next one's
/// and `turn` the two's exclusive or, where [`emit_alternating`] cannot
/// take them: the owners written out.
#[expect(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn emit_turns(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    owner: u32,
    turn: u32,
    owners: &mut Vec<u32>,
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let n = level.pairs.stored_len() - base;
    // As if a candidate out of turn followed, taken back.
    spill_owners(eng.limits(), owners, owner, turn, n, owner)?;
    owners.pop();
    emit_mixed(eng, emit, level, pl_output, p1, base, owners, duplicates_legal)
}

/// Write out the owners of a node's candidates once its walk meets a g
/// parent out of turn: the `before` candidates so far, of `owner` and
/// `owner ^ turn` taking turns with `owner` the next one's, and then
/// `g_parent`'s.
#[cold]
#[inline(never)]
fn spill_owners(
    lim: &crate::limits::Limits,
    owners: &mut Vec<u32>,
    owner: u32,
    turn: u32,
    before: usize,
    g_parent: u32,
) -> Result<(), OperationError> {
    owners.clear();
    reserve_room(lim, owners, before + 1)?;
    // Candidate `before` would have had `owner`, so one an even number of
    // places before it too.
    owners.extend((0..before).map(|i| if (before - i).is_multiple_of(2) { owner } else { owner ^ turn }));
    owners.push(g_parent);
    Ok(())
}

impl<P: Fn(u32, u32) -> u32> NodeWalk for LeafWalk<'_, P> {
    #[inline(always)]
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        for pair in pairs {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            for e in self.outer.bucket(outer1 as usize) {
                let under = self.g_by_outer.bucket(e.g_idx.idx());
                for &RevEntry { parent: g_parent, other: inner2 } in under {
                    let inner_prod = (self.inner_product)(inner1, inner2);
                    if inner_prod != NO_PRODUCT {
                        let (left_prod, right_prod) = orient::<SWAPPED>(inner_prod, e.prod_idx.0);
                        push(ParEntry { g_parent, left_prod, right_prod })?;
                    }
                }
                ticker.poll(under.len() as u64)?;
            }
        }
        Ok(())
    }
}

/// The general arm as a walk of f's nodes: an f pair reads, for each live
/// product of its inner child, the routes its outer child's `filtered`
/// index holds for the product's inner-g child ([`Routes`]). The candidates
/// and the clock's charges are the general arm's; only their order differs.
///
/// With `pipelined`, on route tables too large for the cache, a node's
/// lookups go in groups of [`LOOK_GROUP`]: the walk asks for each lookup's
/// first slot as it finds the lookup, probes the group's slots once it is
/// full, asking for each hit's run, and only then reads the runs. Each
/// lookup would otherwise wait on a miss of its own before the next one
/// starts. The candidates come in the same order either way.
struct RoutedWalk<'a> {
    inner: GroupedView<'a, ProductEntry>,
    routes: RoutesView<'a>,
    pipelined: bool,
}

/// One lookup of a pipelined [`RoutedWalk`]: the outer's table, the key
/// and the inner product it was found for.
#[derive(Clone, Copy, Default)]
struct Lookup {
    segment: Segment,
    key: u32,
    prod: u32,
}

/// The lookups a pipelined [`RoutedWalk`] asks the cache for at once:
/// about as many misses as a core keeps in flight.
const LOOK_GROUP: usize = 16;

impl RoutedWalk<'_> {
    /// The candidates of one f pair, of inner child `inner1`, under the
    /// outer of `segment`.
    #[inline(always)]
    fn pair<const SWAPPED: bool>(
        &self,
        segment: Segment,
        inner1: u32,
        ticker: &mut crate::limits::PollGate,
        push: &mut impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        for e in self.inner.bucket(inner1 as usize) {
            let under = self.routes.find(segment, e.g_idx.0);
            if under.is_empty() {
                continue;
            }
            let inner_prod = e.prod_idx.0;
            for &(g_parent, attached) in under {
                let (left_prod, right_prod) = orient::<SWAPPED>(inner_prod, attached);
                push(ParEntry { g_parent, left_prod, right_prod })?;
            }
            ticker.poll(under.len() as u64)?;
        }
        Ok(())
    }

    /// The pipelined walk of one node's pairs ([`RoutedWalk`]).
    #[inline(always)]
    fn node_pipelined<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        push: &mut impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let mut group = [Lookup::default(); LOOK_GROUP];
        let mut n = 0;
        for pair in pairs {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            let segment = self.routes.segments[outer1 as usize];
            if segment.bits == 0 {
                continue;
            }
            for e in self.inner.bucket(inner1 as usize) {
                let key = e.g_idx.0;
                prefetch_at(self.routes.slots, segment.first_slot(key));
                group[n] = Lookup { segment, key, prod: e.prod_idx.0 };
                n += 1;
                if n == LOOK_GROUP {
                    self.read_group::<SWAPPED>(&group, ticker, push)?;
                    n = 0;
                }
            }
        }
        self.read_group::<SWAPPED>(&group[..n], ticker, push)
    }

    /// Probe each of `group`'s slots, asking for each hit's run, then hand
    /// every hit's candidates to `push`, in lookup order.
    #[inline(always)]
    fn read_group<const SWAPPED: bool>(
        &self,
        group: &[Lookup],
        ticker: &mut crate::limits::PollGate,
        push: &mut impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let mut hits = [(RouteSlot::EMPTY, 0u32); LOOK_GROUP];
        let mut m = 0;
        for l in group {
            if let Some(slot) = self.routes.slot(l.segment, l.key) {
                prefetch_at(self.routes.entries, slot.start as usize);
                hits[m] = (slot, l.prod);
                m += 1;
            }
        }
        for &(slot, inner_prod) in &hits[..m] {
            let under = &self.routes.entries[slot.start as usize..(slot.start + slot.len) as usize];
            for &(g_parent, attached) in under {
                let (left_prod, right_prod) = orient::<SWAPPED>(inner_prod, attached);
                push(ParEntry { g_parent, left_prod, right_prod })?;
            }
            ticker.poll(under.len() as u64)?;
        }
        Ok(())
    }
}

impl NodeWalk for RoutedWalk<'_> {
    #[inline(always)]
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        if self.pipelined {
            return self.node_pipelined::<SWAPPED>(pairs, ticker, &mut push);
        }
        for pair in pairs {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            let segment = self.routes.segments[outer1 as usize];
            if segment.bits == 0 {
                continue;
            }
            self.pair::<SWAPPED>(segment, inner1, ticker, &mut push)?;
        }
        Ok(())
    }
}

/// The general arm into the level's nodes, by f node: every outer key's
/// `filtered` index, with the entries [`scatter_general_arm`] would bucket,
/// is built into [`Routes`], and f's nodes are then walked in order
/// ([`RoutedWalk`]), each emitted into `emit` as its walk ends. No
/// candidate is held past its node. f's index by outer key holds only the
/// pairs the route builds read through it ([`build_kept_index`]).
#[inline(never)]
pub(super) fn scatter_routed<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    emit: Emit<'_>,
) -> Result<(), OperationError> {
    build_g_index::<SWAPPED>(eng, ws, g_level, shape)?;
    build_inner_index::<SWAPPED>(eng, ws, g_level, shape)?;
    let lim = eng.limits();
    let (pl_inner, pl_outer) = if !SWAPPED { (pl.left, pl.right) } else { (pl.right, pl.left) };
    let f_pairs = f_under::<SWAPPED>(&ws.est_counts, shape).iter().map(|&n| n as usize).sum();
    build_kept_index::<SWAPPED>(eng, ws, f_level, shape, pl_outer)?;
    let mut s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer, true)?;
    let marked = route_level::<SWAPPED>(lim, &mut s, f_level, shape.f.here, f_pairs)?;
    let SparseWorkspace { routes, inner_offsets, node_owners, emit: scratch, .. } = ws;
    let walk = RoutedWalk {
        inner: GroupedView { offsets: inner_offsets, entries: pl_inner },
        routes: routes.view(),
        pipelined: std::mem::size_of_val(&routes.slots[..]) + std::mem::size_of_val(&routes.entries[..])
            >= emit.thresholds.pipelined_bytes,
    };
    let nodes = if marked {
        Walked::Marked { bits: &routes.walked, nodes: shape.f.here }
    } else {
        Walked::All(shape.f.here)
    };
    walk_nodes::<SWAPPED, _>(eng, &walk, f_level, nodes, Collect::Nodes(emit), node_owners, scratch)
}

/// Build every outer key's route. Whether the walk of f's nodes visits
/// only the nodes holding a pair under an outer with a route
/// ([`MARKED_SHARE`]), and if so their bits.
fn route_level<const SWAPPED: bool>(
    lim: &crate::limits::Limits,
    s: &mut ScatterSides<'_>,
    f_level: &TddLevel,
    nodes: usize,
    f_pairs: usize,
) -> Result<bool, OperationError> {
    s.routes.reset(lim, s.outer_k, s.inner_g)?;
    // The f pairs under an outer with a route: the pairs the general arm
    // would walk.
    let mut routed_pairs = 0usize;
    for outer in 0..s.outer_k {
        if s.outer.bucket(outer).is_empty() || s.f_pairs_under(outer) == 0 {
            continue;
        }
        s.route_outer(lim, outer)?;
        if s.routes.segments[outer].bits != 0 {
            routed_pairs += s.f_pairs_under(outer);
        }
    }
    // Where those are few of f's pairs, only the nodes holding one are
    // walked, as the general arm walks only them.
    let marked = routed_pairs < f_pairs / MARKED_SHARE;
    if marked {
        s.routes.mark_parents::<SWAPPED>(lim, f_level, nodes)?;
    }
    Ok(marked)
}

/// [`scatter_routed`] walks only the f nodes holding a pair under an outer
/// with a route when those pairs are fewer than one in this many of f's.
const MARKED_SHARE: usize = 2;

/// The general arm of the scatter on a level with one product, whose
/// children are both internal and joined through their product lists: the
/// four-way join of f/g parent and child/sibling product lists.
/// `SWAPPED = false` outer-loops by right sibling s1; `SWAPPED = true` by
/// left child a1 (every difference is a pure left↔right role rename; the
/// `if SWAPPED` branches fold at compile time). A filtered per-outer g index
/// makes the walk read only alive `(p2, product)` entries. The candidates
/// come in outer-key order, which is sound because pair lists are
/// order-independent.
///
/// Per outer key:
///   1. Build `filtered`: bucket the g parents under the outer's live g keys
///      by the join's inner-g child, attaching the live product — from
///      whichever g index is the cheaper to walk, and only for the children
///      the emit will read when finding those costs less than it saves.
///   2. Emit: for each f-parent sharing the outer, for each alive inner product,
///      push the precomputed alive `(p2, product)` entries — zero dead probes.
///   3. Clear only the `filtered` buckets touched this outer.
///
/// A leaf child, a pass-through side and a complete child read by
/// arithmetic go to the leaf arm ([`scatter_leaf_arm`]), so this arm sees
/// only the levels the direction estimate ran for: its counts start the
/// index builds. `collect` says where the candidates go; the level's nodes
/// are not one of the places: this arm finds a parent's candidates under
/// many outer keys, and [`scatter_routed`] builds the nodes by f node
/// instead.
pub(super) fn scatter_join<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    collect: Collect<'_, '_>,
) -> Result<(), OperationError> {
    build_scatter_indexes::<SWAPPED>(eng, ws, f_level, g_level, shape)?;
    build_inner_index::<SWAPPED>(eng, ws, g_level, shape)?;
    // Decided once per level, so that no walk tests it per pair.
    let pl_inner = if !SWAPPED { pl.left } else { pl.right };
    if prefetching(std::mem::size_of_val(pl_inner)) {
        scatter_general_arm::<SWAPPED, true>(eng, ws, shape, pl, collect)
    } else {
        scatter_general_arm::<SWAPPED, false>(eng, ws, shape, pl, collect)
    }
}

/// Build the two reverse indexes the general arm reads, from the direction
/// estimate's counts.
///
/// f is keyed by the outer-loop dimension (normal → right sibling `s1`,
/// swapped → left child `a1`), and g by the general arm's filtering axis:
/// normal → by right `s2`, entries `(p2, a2)`; swapped → by left `a2`,
/// entries `(p2, s2)`.
#[inline(never)]
fn build_scatter_indexes<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
) -> Result<(), OperationError> {
    let SparseWorkspace { est_counts, f_by_outer, .. } = ws;
    let counts = EstCounts::of(est_counts, shape);
    // Keyed by the outer dimension: the right sibling normally, the left child
    // when swapped.
    if !SWAPPED {
        build_reverse_index::<true>(eng, f_level, shape.f.right, Some(counts.f_right), f_by_outer)?;
    } else {
        build_reverse_index::<false>(eng, f_level, shape.f.left, Some(counts.f_left), f_by_outer)?;
    }
    build_g_index::<SWAPPED>(eng, ws, g_level, shape)
}

/// Build g's reverse index keyed by the general arm's filtering axis, the
/// second index of [`build_scatter_indexes`].
#[inline(never)]
fn build_g_index<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    g_level: &TddLevel,
    shape: LevelShape,
) -> Result<(), OperationError> {
    let SparseWorkspace { est_counts, g_by_outer, .. } = ws;
    let counts = EstCounts::of(est_counts, shape);
    if !SWAPPED {
        build_reverse_index::<true>(eng, g_level, shape.g.right, Some(counts.g_right), g_by_outer)
    } else {
        build_reverse_index::<false>(eng, g_level, shape.g.left, Some(counts.g_left), g_by_outer)
    }
}

/// f's pairs under each outer key, from the direction estimate's counts:
/// the bucket sizes of f's full index by outer key.
fn f_under<const SWAPPED: bool>(est_counts: &[u32], shape: LevelShape) -> &[u32] {
    let counts = EstCounts::of(est_counts, shape);
    if !SWAPPED { counts.f_right } else { counts.f_left }
}

/// Build f's index by outer key for [`scatter_routed`], over only the pairs
/// [`route_level`] reads through it: those under an outer key whose route
/// build marks the keys its emit reads ([`ScatterSides::build_plan`]),
/// which walks them. The walk of f's nodes reads f in node order, and the
/// estimate's counts give every outer's pairs
/// ([`ScatterSides::f_pairs_under`]).
#[inline(never)]
fn build_kept_index<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    shape: LevelShape,
    pl_outer: &[ProductEntry],
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let SparseWorkspace { est_counts, g_by_outer, outer_offsets, f_marked, f_kept, f_by_outer, .. } = ws;
    let under = f_under::<SWAPPED>(est_counts, shape);
    let width = under.len();
    bucket_offsets(lim, pl_outer, width, outer_offsets)?;
    let outer = GroupedView { offsets: outer_offsets, entries: pl_outer };
    let g_by_outer = g_by_outer.view();
    f_marked.clear();
    lim.try_resize(f_marked, width.div_ceil(64), 0u64)?;
    f_kept.clear();
    lim.try_resize(f_kept, width, 0u32)?;
    let mut any = false;
    for (o, &n) in under.iter().enumerate() {
        let live = outer.bucket(o);
        if n > 0 && !live.is_empty() && cost_by_key(live, g_by_outer) > n as usize {
            f_marked[o >> 6] |= 1 << (o & 63);
            f_kept[o] = n;
            any = true;
        }
    }
    counting_sort(
        lim, width, KeptPairs::<SWAPPED> { level: f_level, marked: any.then_some(&f_marked[..]) },
        |(parent, pair): (u32, ChildPair)| {
            let (inner1, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
            (outer1 as usize, RevEntry { parent, other: inner1 })
        },
        Some(&f_kept[..]), RevEntry { parent: 0, other: 0 }, f_by_outer,
    )
}

/// The pairs [`build_kept_index`] keeps, in node order: every pair under
/// an outer key with its bit set in `marked`, none where no outer key is.
struct KeptPairs<'a, const SWAPPED: bool> {
    level: &'a TddLevel,
    marked: Option<&'a [u64]>,
}

impl<const SWAPPED: bool> SortItems for KeptPairs<'_, SWAPPED> {
    type Item = (u32, ChildPair);
    #[inline(always)]
    fn each(&self, mut f: impl FnMut((u32, ChildPair))) {
        let Some(marked) = self.marked else {
            return;
        };
        self.level.try_for_each_node::<std::convert::Infallible>(0..self.level.node_count(), |p1, pairs| {
            for &pair in pairs {
                let (_, outer1) = orient::<SWAPPED>(pair.left.0, pair.right.0);
                if marked[outer1 as usize >> 6] >> (outer1 & 63) & 1 != 0 {
                    f((p1 as u32, pair));
                }
            }
            Ok(())
        }).unwrap_or_else(|never| match never {});
    }
}

/// The g index entries an outer key's route build by its live products'
/// g keys walks: the products' `g_by_outer` buckets.
#[inline]
fn cost_by_key(live: &[ProductEntry], g_by_outer: GroupedView<'_, RevEntry>) -> usize {
    live.iter().map(|e| g_by_outer.len(e.g_idx.idx())).sum()
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
pub(super) fn bucket_offsets(
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
    /// f's reverse index, keyed by the outer dimension; on a routed level
    /// only the pairs read through it ([`build_kept_index`]).
    f_by_outer: GroupedView<'w, RevEntry>,
    /// f's pairs under each outer key, all of them.
    f_under: &'w [u32],
    /// Where f's index is kept, a bit for each outer key whose build marks
    /// the keys it reads ([`ScatterSides::build_plan`]).
    f_marked: Option<&'w [u64]>,
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
    /// Every outer's `filtered`, kept for [`scatter_routed`]'s walk.
    routes: &'w mut Routes,
    /// How many outer keys the emit loop walks.
    outer_k: usize,
    /// How many inner-g children `filtered` is keyed by.
    inner_g: usize,
}

/// Which g index [`ScatterSides::build_plan`] builds one outer key's index
/// from.
#[derive(Clone, Copy)]
enum BuildFrom {
    /// The outer's g keys, for every inner-g child.
    Keys,
    /// The outer's g keys, for the wanted inner-g children only.
    WantedKeys,
    /// The wanted inner-g children.
    Inner,
}

/// How many bytes a level's inner products (or its parents' buckets) span
/// before its walks ask the cache for what they read ahead: below it they
/// mostly find it in the last-level cache, and the prefetches only add
/// instructions.
const PREFETCH_BYTES: usize = 32 << 20;

/// Whether a level whose walks read `bytes` bytes at random asks the cache
/// for them ahead: past [`PREFETCH_BYTES`], or as a test pins it.
#[inline(always)]
fn prefetching(bytes: usize) -> bool {
    forced_prefetch().unwrap_or(bytes >= PREFETCH_BYTES)
}

// Tests pin the gate and note which walks asked ahead; production always
// sizes it.
#[cfg(test)]
use super::tests::{forced_prefetch, note_prefetch};

#[cfg(not(test))]
#[inline(always)]
fn forced_prefetch() -> Option<bool> {
    None
}

#[cfg(not(test))]
#[inline(always)]
fn note_prefetch(_site: usize) {}

/// A bucket array cleared per outer key by replaying the indices written into
/// it, so the clear costs the touched buckets rather than the whole array.
/// `held` has a bit per bucket, set while it holds an entry: the emit tests
/// it before it reads the bucket, so a miss reads a bit set that stays in
/// cache rather than a bucket header that, on a wide level, does not.
struct TouchedBuckets<'a> {
    buckets: &'a mut [Vec<(u32, u32)>],
    touched: &'a mut Vec<u32>,
    held: &'a mut Vec<u64>,
}

impl TouchedBuckets<'_> {
    /// Every bucket, indexed by key, for a walk that reads many of them.
    fn index(&self) -> Pushed<'_> {
        Pushed { buckets: self.buckets, held: self.held }
    }

    fn push(&mut self, lim: &crate::limits::Limits, key: u32, v: (u32, u32)) -> Result<(), OperationError> {
        let bucket = &mut self.buckets[key as usize];
        if bucket.is_empty() {
            self.touched.push(key);
            self.held[key as usize >> 6] |= 1 << (key & 63);
        }
        lim.try_push(bucket, v)
    }

    fn clear_touched(&mut self) {
        for &key in self.touched.iter() {
            self.buckets[key as usize].clear();
            // Every bit set in this word is a touched bucket's.
            self.held[key as usize >> 6] = 0;
        }
        self.touched.clear();
    }
}

/// The buckets as the emit reads them ([`TouchedBuckets::index`]), with the
/// bit per bucket that says whether it holds an entry.
#[derive(Clone, Copy)]
struct Pushed<'a> {
    buckets: &'a [Vec<(u32, u32)>],
    held: &'a [u64],
}

impl Pushed<'_> {
    /// False only where the bucket is empty: the emit's first test, before it
    /// reads the bucket.
    #[inline(always)]
    fn holds(&self, key: usize) -> bool {
        self.held[key >> 6] >> (key & 63) & 1 != 0
    }

    #[inline(always)]
    fn entries_at(&self, key: usize) -> &[(u32, u32)] {
        &self.buckets[key]
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
/// bucketed by f index. Where f's index is `kept` ([`build_kept_index`]),
/// the outer products are bucketed already and the outers that mark their
/// keys are known.
#[inline(never)]
fn sides<'w, const SWAPPED: bool>(
    eng: &Engine,
    ws: &'w mut SparseWorkspace,
    shape: LevelShape,
    pl_inner: &'w [ProductEntry],
    pl_outer: &'w [ProductEntry],
    kept: bool,
) -> Result<ScatterSides<'w>, OperationError> {
    let LevelShape { f, g, .. } = shape;
    let (inner_k, outer_k, filtered_dim, outer_g_dim) = if !SWAPPED {
        (f.left, f.right, g.left, g.right)
    } else {
        (f.right, f.left, g.right, g.left)
    };
    let SparseWorkspace {
        f_by_outer, g_by_outer, g_by_inner, est_counts,
        inner_offsets, outer_offsets,
        filtered, filtered_touched, filtered_held,
        wanted, wanted_epoch, wanted_keys, inner_seen, inner_seen_epoch,
        outer_keys, outer_keys_epoch, outer_attached, routes, f_marked,
        ..
    } = ws;
    bucket_offsets(eng.limits(), pl_inner, inner_k, inner_offsets)?;
    if !kept {
        bucket_offsets(eng.limits(), pl_outer, outer_k, outer_offsets)?;
    }
    filtered.reset(eng.limits(), filtered_dim)?;
    filtered_touched.clear();
    // A round that bailed can leave bits set; the level starts with none.
    let held_words = filtered_dim.div_ceil(64);
    eng.limits().try_resize(filtered_held, held_words, 0u64)?;
    filtered_held[..held_words].fill(0);
    eng.limits().try_resize(wanted, filtered_dim, 0u32)?;
    wanted_keys.clear();
    eng.limits().try_resize(inner_seen, inner_k, 0u32)?;
    eng.limits().try_resize(outer_keys, outer_g_dim, 0u32)?;
    eng.limits().try_resize(outer_attached, outer_g_dim, 0u32)?;
    Ok(ScatterSides {
        f_by_outer: f_by_outer.view(),
        f_under: f_under::<SWAPPED>(est_counts, shape),
        f_marked: kept.then_some(&f_marked[..]),
        g_by_outer: g_by_outer.view(),
        g_by_inner: g_by_inner.view(),
        outer: GroupedView { offsets: outer_offsets, entries: pl_outer },
        inner: GroupedView { offsets: inner_offsets, entries: pl_inner },
        filtered: TouchedBuckets { buckets: filtered, touched: filtered_touched, held: filtered_held },
        wanted: EpochFlags { cur: *wanted_epoch, stamps: wanted, epoch: wanted_epoch },
        wanted_keys,
        inner_seen: EpochFlags { cur: *inner_seen_epoch, stamps: inner_seen, epoch: inner_seen_epoch },
        outer_keys: EpochFlags { cur: *outer_keys_epoch, stamps: outer_keys, epoch: outer_keys_epoch },
        outer_attached,
        routes,
        outer_k,
        inner_g: filtered_dim,
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
        self.f_under[outer] as usize
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
        self.mark_outer_keys(outer);
        let ScatterSides { outer_keys, outer_attached, wanted_keys, g_by_inner, filtered, .. } = self;
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

    /// Fill `filtered` for one outer key, by [`ScatterSides::build_plan`].
    fn build_for_outer(&mut self, lim: &crate::limits::Limits, outer: usize) -> Result<(), OperationError> {
        match self.build_plan(outer) {
            BuildFrom::Keys => self.build_filtered_for_outer::<false>(lim, outer),
            BuildFrom::WantedKeys => self.build_filtered_for_outer::<true>(lim, outer),
            BuildFrom::Inner => self.build_filtered_by_inner(lim, outer),
        }
    }

    /// How one outer key's index is built: from whichever g index is the
    /// cheaper to walk, and only for the children the emit will read when
    /// finding those costs less than it saves. Marks the wanted children
    /// where it plans to read them.
    fn build_plan(&mut self, outer: usize) -> BuildFrom {
        // A kept index decided the test below for every outer as it was
        // built.
        if let Some(marked) = self.f_marked
            && marked[outer >> 6] >> (outer & 63) & 1 == 0
        {
            return BuildFrom::Keys;
        }
        let by_key = self.build_cost_by_key(outer);
        if by_key <= self.f_pairs_under(outer) {
            // The walk by the outer's g keys costs no more than the emit's
            // own walk of the f pairs under it, and the marking costs at
            // least that walk: neither it nor the filter it feeds can pay
            // for itself here.
            return BuildFrom::Keys;
        }
        self.mark_wanted_for_outer(outer);
        if self.build_cost_by_inner() < by_key { BuildFrom::Inner } else { BuildFrom::WantedKeys }
    }

    /// Give one outer key its table in `routes`, by
    /// [`ScatterSides::build_plan`]: the entries [`ScatterSides::build_for_outer`]
    /// would bucket, written into `routes` as runs, without the buckets.
    fn route_outer(&mut self, lim: &crate::limits::Limits, outer: usize) -> Result<(), OperationError> {
        match self.build_plan(outer) {
            BuildFrom::Keys => self.route_by_key::<false>(lim, outer)?,
            BuildFrom::WantedKeys => self.route_by_key::<true>(lim, outer)?,
            BuildFrom::Inner => self.route_by_inner(lim, outer)?,
        }
        self.routes.place(lim, outer, self.inner_g)
    }

    /// [`ScatterSides::build_filtered_for_outer`] into `routes`: the entries
    /// come by the outer's g keys, so a first walk counts each inner-g
    /// child's and a second writes each into its run.
    fn route_by_key<const FILTER: bool>(&mut self, lim: &crate::limits::Limits, outer: usize) -> Result<(), OperationError> {
        let ScatterSides { outer: outer_products, g_by_outer, wanted, routes, .. } = self;
        let Routes { entries, pending, cursor, .. } = &mut **routes;
        pending.clear();
        let mut total = 0usize;
        for e in outer_products.bucket(outer) {
            for &RevEntry { other: key, .. } in g_by_outer.bucket(e.g_idx.idx()) {
                if FILTER && !wanted.is_set(key) {
                    continue;
                }
                let count = &mut cursor[key as usize];
                if *count == 0 {
                    pending.push(RouteSlot { key, start: 0, len: 0 });
                }
                *count += 1;
                total += 1;
            }
        }
        let mut at = entries.len();
        if at + total > u32::MAX as usize {
            return Err(OperationError::IndexOverflow);
        }
        for slot in pending.iter_mut() {
            slot.start = at as u32;
            slot.len = cursor[slot.key as usize];
            cursor[slot.key as usize] = at as u32;
            at += slot.len as usize;
        }
        reserve_room(lim, entries, total)?;
        entries.resize(at, (0, 0));
        for e in outer_products.bucket(outer) {
            let attached = e.prod_idx.0;
            for &RevEntry { parent: p2, other: key } in g_by_outer.bucket(e.g_idx.idx()) {
                if FILTER && !wanted.is_set(key) {
                    continue;
                }
                let next = &mut cursor[key as usize];
                entries[*next as usize] = (p2, attached);
                *next += 1;
            }
        }
        for slot in pending.iter() {
            cursor[slot.key as usize] = 0;
        }
        Ok(())
    }

    /// [`ScatterSides::build_filtered_by_inner`] into `routes`: the entries
    /// come by wanted inner-g child, each child's as one run.
    fn route_by_inner(&mut self, lim: &crate::limits::Limits, outer: usize) -> Result<(), OperationError> {
        let most = self.build_cost_by_inner();
        self.mark_outer_keys(outer);
        let ScatterSides { outer_keys, outer_attached, wanted_keys, g_by_inner, routes, .. } = self;
        let Routes { entries, pending, .. } = &mut **routes;
        pending.clear();
        if entries.len() + most > u32::MAX as usize {
            return Err(OperationError::IndexOverflow);
        }
        reserve_room(lim, entries, most)?;
        for &key in wanted_keys.iter() {
            let start = entries.len();
            for &RevEntry { parent: p2, other: right_key } in g_by_inner.bucket(key as usize) {
                if outer_keys.is_set(right_key) {
                    entries.push((p2, outer_attached[right_key as usize]));
                }
            }
            if entries.len() > start {
                pending.push(RouteSlot { key, start: start as u32, len: (entries.len() - start) as u32 });
            }
        }
        Ok(())
    }

    /// Mark the outer's live g keys, each with its product, for the walks by
    /// inner-g child.
    fn mark_outer_keys(&mut self, outer: usize) {
        self.outer_keys.begin();
        for e in self.outer.bucket(outer) {
            self.outer_keys.mark(e.g_idx.0);
            self.outer_attached[e.g_idx.idx()] = e.prod_idx.0;
        }
    }

    /// The g index entries building this outer's `filtered` by the outer's
    /// g keys walks.
    fn build_cost_by_key(&self, outer: usize) -> usize {
        cost_by_key(self.outer.bucket(outer), self.g_by_outer)
    }

    /// The g index entries building this outer's `filtered` by the wanted
    /// inner-g children walks; the marking has run.
    fn build_cost_by_inner(&self) -> usize {
        self.wanted_keys
            .iter()
            .map(|&a| self.g_by_inner.len(a as usize))
            .sum()
    }

    /// The candidates of the f pairs under one outer key, folded into a
    /// count ([`Collect::Fold`]): for each f pair, each alive inner product
    /// replays the alive `filtered` entries of its inner-g child
    /// ([`emit_candidates`]). With `FAR`, on a level whose inner products
    /// span more than [`PREFETCH_BYTES`], each pair's inner run is asked for
    /// some pairs ahead ([`prefetch_walk`]).
    fn emit_fold_for_outer<const SWAPPED: bool, const FAR: bool>(
        &self,
        fold: &mut CandidateFold<'_>,
        outer: usize,
        ticker: &mut crate::limits::PollGate,
    ) -> Result<(), OperationError> {
        let filtered = self.filtered.index();
        if !FAR {
            for &RevEntry { other: inner1, .. } in self.f_by_outer.bucket(outer) {
                emit_candidates::<SWAPPED>(self.inner, filtered, inner1, ticker, |entry| {
                    fold.push(candidate_pair(&entry));
                    Ok(())
                })?;
            }
            return Ok(());
        }
        let under = self.f_by_outer.entries_of(self.outer_k);
        for j in self.f_by_outer.offsets[outer] as usize..self.f_by_outer.offsets[outer + 1] as usize {
            prefetch_walk(under, j, self.inner);
            note_prefetch(0);
            emit_candidates::<SWAPPED>(self.inner, filtered, under[j].other, ticker, |entry| {
                fold.push(candidate_pair(&entry));
                Ok(())
            })?;
        }
        Ok(())
    }

    /// [`ScatterSides::emit_fold_for_outer`] on a level whose candidates go
    /// straight into the output level's pair arena ([`Collect::Direct`]).
    fn emit_direct_for_outer<const SWAPPED: bool, const FAR: bool>(
        &self,
        eng: &Engine,
        level: &mut TddLevel,
        outer: usize,
        ticker: &mut crate::limits::PollGate,
    ) -> Result<(), OperationError> {
        let filtered = self.filtered.index();
        if !FAR {
            for &RevEntry { other: inner1, .. } in self.f_by_outer.bucket(outer) {
                emit_candidates::<SWAPPED>(self.inner, filtered, inner1, ticker,
                    |entry| try_push_pair_into(eng, level, candidate_pair(&entry)))?;
            }
            return Ok(());
        }
        let under = self.f_by_outer.entries_of(self.outer_k);
        for j in self.f_by_outer.offsets[outer] as usize..self.f_by_outer.offsets[outer + 1] as usize {
            prefetch_walk(under, j, self.inner);
            note_prefetch(1);
            emit_candidates::<SWAPPED>(self.inner, filtered, under[j].other, ticker,
                |entry| try_push_pair_into(eng, level, candidate_pair(&entry)))?;
        }
        Ok(())
    }
}

impl ScatterSides<'_> {
    /// [`ScatterSides::emit_fold_for_outer`] on a root whose candidates are
    /// added into sums ([`Collect::Sum`]).
    fn emit_sum_for_outer<const SWAPPED: bool, const FAR: bool>(
        &self,
        lim: &crate::limits::Limits,
        sum: &mut ChildSum,
        outer: usize,
        ticker: &mut crate::limits::PollGate,
    ) -> Result<(), OperationError> {
        let filtered = self.filtered.index();
        if !FAR {
            for &RevEntry { other: inner1, .. } in self.f_by_outer.bucket(outer) {
                emit_candidates::<SWAPPED>(self.inner, filtered, inner1, ticker,
                    |entry| sum.push(lim, candidate_pair(&entry)))?;
            }
            return Ok(());
        }
        let under = self.f_by_outer.entries_of(self.outer_k);
        for j in self.f_by_outer.offsets[outer] as usize..self.f_by_outer.offsets[outer + 1] as usize {
            prefetch_walk(under, j, self.inner);
            // Noted as a walk building a one-product level, which it is.
            note_prefetch(1);
            emit_candidates::<SWAPPED>(self.inner, filtered, under[j].other, ticker,
                |entry| sum.push(lim, candidate_pair(&entry)))?;
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
/// such a miss reads the bucket's bit and moves on, and only a hit reaches
/// the push loop and the poll.
#[inline(always)]
fn emit_candidates<const SWAPPED: bool>(
    inner: GroupedView<'_, ProductEntry>,
    filtered: Pushed<'_>,
    inner1: u32,
    ticker: &mut crate::limits::PollGate,
    mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    for e in inner.bucket(inner1 as usize) {
        let key = e.g_idx.idx();
        if !filtered.holds(key) {
            continue;
        }
        let fb = filtered.entries_at(key);
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

/// The general arm on a level with one product: both children are
/// non-leaf. Per outer key, build the filtered g index, walk the f pairs
/// under the key against it, then clear only the buckets this outer
/// touched. `FAR` runs the walks that ask the cache ahead
/// ([`ScatterSides::emit_fold_for_outer`]).
#[inline(never)]
fn scatter_general_arm<const SWAPPED: bool, const FAR: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    mut collect: Collect<'_, '_>,
) -> Result<(), OperationError> {
    if let Collect::Fold(fold) = &mut collect
        && fold.grouped()
    {
        return count_arm::<SWAPPED>(eng, ws, shape, pl, fold);
    }
    let lim = eng.limits();
    let (pl_inner, pl_outer) = if !SWAPPED { (pl.left, pl.right) } else { (pl.right, pl.left) };
    let mut s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer, false)?;

    // Amortized cancellation/deadline poll. The sparse join has no other
    // mid-level break, so a wide level could otherwise wait out an expired
    // deadline; the bail lands at a loop level `try_push`'s recovery already
    // covers, so the workspace stays reusable.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    // Into the level's pair arena, projected before each outer key's f pairs.
    let mut projection = match &mut collect {
        Collect::Direct(level) => Some(Projection::new(level)),
        _ => None,
    };
    for outer in 0..s.outer_k {
        if s.outer.bucket(outer).is_empty() { continue; }
        s.build_for_outer(lim, outer)?;
        match &mut collect {
            Collect::Direct(level) => {
                if let Some(projection) = &mut projection {
                    let walked = || (0..s.outer_k).filter(|&o| !s.outer.bucket(o).is_empty()).map(|o| s.f_pairs_under(o)).sum();
                    projection.before(eng, level, s.f_pairs_under(outer), walked);
                }
                s.emit_direct_for_outer::<SWAPPED, FAR>(eng, level, outer, &mut ticker)?
            }
            Collect::Fold(fold) => s.emit_fold_for_outer::<SWAPPED, FAR>(fold, outer, &mut ticker)?,
            Collect::Sum(sum) => s.emit_sum_for_outer::<SWAPPED, FAR>(lim, sum, outer, &mut ticker)?,
            Collect::Nodes(_) => unreachable!("a walk of f's nodes builds a level's nodes"),
        }
        s.filtered.clear_touched();
    }
    Ok(())
}

/// How many inner f children ahead of the one it reads a counted root's walk
/// asks the cache for (see [`GroupedView::prefetch_bucket`]).
pub(super) const WALK_AHEAD: usize = 4;

/// The general arm on a counted level whose counts all fit `u64`
/// ([`CandidateFold::grouped`]): [`count_general_arm`], with `FILTER` where
/// the level's walks test their keys against a filter first
/// ([`CandidateFold::filters_walks`]), decided once for the level.
#[inline(never)]
fn count_arm<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    fold: &mut CandidateFold<'_>,
) -> Result<(), OperationError> {
    if fold.filters_walks::<SWAPPED>() {
        count_general_arm::<SWAPPED, true>(eng, ws, shape, pl, fold)
    } else {
        count_general_arm::<SWAPPED, false>(eng, ws, shape, pl, fold)
    }
}

/// The general arm on a counted level whose counts all fit `u64`
/// ([`CandidateFold::grouped`]): the candidates are summed, never listed.
///
/// Under one outer key, the candidates of an inner product `e` are `e`
/// against every live outer product that a g pair of `e`'s inner-g child
/// `k` reaches, so they contribute `count(e) × S[k]`, with `S[k]` the summed
/// counts of those outer products. The emit walk visits each `e` once and
/// reads `S`, which is built one of two ways:
///
/// - by key: every g pair under the outer's live g keys adds its outer
///   product's count to its inner-g child, as the unfiltered `filtered`
///   build would bucket it;
/// - by inner child: each key the walk reads sums its own g pairs, each
///   reading its outer-g child's live count (0 when it has none).
///
/// An outer whose build by key walks no more than the emit walk builds by
/// key outright, and the emit walk reads it. Otherwise one walk opens the
/// keys the emit would read, weighs each by `Σ count(e)` over the inner
/// products that read it, and prices the build by inner child exactly, as
/// `mark_wanted_for_outer` does for the listing arm; the cheaper build runs
/// and the outer adds each opened key's weight times its sum, so the emit
/// walk is never repeated. By key, the build adds only to the opened keys,
/// a read where it misses. The fold keeps one round per outer key
/// ([`CandidateFold::begin_round`]): which keys are opened and which outer-g
/// children are live are bit sets, tested for every g pair a build walks.
#[inline(never)]
fn count_general_arm<const SWAPPED: bool, const FILTER: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    fold: &mut CandidateFold<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (pl_inner, pl_outer) = if !SWAPPED { (pl.left, pl.right) } else { (pl.right, pl.left) };
    let s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer, false)?;
    let ScatterSides {
        f_by_outer, g_by_outer, g_by_inner, outer: outer_products, inner,
        wanted, outer_keys, outer_k, ..
    } = s;
    // The fold's round stands in for `wanted` and `outer_keys`, which are
    // only read for their widths.
    fold.prepare_sums(lim, wanted.stamps.len(), outer_keys.stamps.len())?;
    if FILTER {
        fold.prepare_filter(lim, wanted.stamps.len())?;
    }
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for outer in 0..outer_k {
        let live = outer_products.bucket(outer);
        if live.is_empty() {
            continue;
        }
        let under = f_by_outer.bucket(outer);
        let walk: usize = under.iter().map(|r| inner.len(r.other as usize)).sum();
        let by_key: usize = live.iter().map(|e| g_by_outer.len(e.g_idx.idx())).sum();
        fold.begin_round();
        if by_key <= walk {
            // The build by key costs no more than the walk that reads it.
            for e in live {
                let count = fold.outer_count::<SWAPPED>(e.prod_idx.0);
                for &RevEntry { other: key, .. } in g_by_outer.bucket(e.g_idx.idx()) {
                    fold.add_to_sum(lim, key, count)?;
                }
            }
            ticker.poll(by_key as u64)?;
            if FILTER {
                fold.walk_grouped::<SWAPPED>(under, inner, &mut ticker)?;
            } else {
                for (j, &RevEntry { other: inner1, .. }) in under.iter().enumerate() {
                    if let Some(ahead) = under.get(j + WALK_AHEAD) {
                        inner.prefetch_bucket(ahead.other as usize);
                    }
                    let products = inner.bucket(inner1 as usize);
                    for e in products {
                        fold.add_grouped::<SWAPPED>(e.prod_idx.0, e.g_idx.0);
                    }
                    ticker.poll(products.len() as u64)?;
                }
            }
        } else {
            // Open the keys the walk reads, weigh each by the counts of the
            // inner products that read it, and price summing each by its own
            // g pairs against the build by key. The walk is then over: the
            // outer's candidates are each opened key's weight times its sum.
            let mut by_inner = 0usize;
            for (j, &RevEntry { other: inner1, .. }) in under.iter().enumerate() {
                if let Some(ahead) = under.get(j + WALK_AHEAD) {
                    inner.prefetch_bucket(ahead.other as usize);
                }
                for e in inner.bucket(inner1 as usize) {
                    let key = e.g_idx.0;
                    if fold.open_weighted(lim, key, fold.inner_count::<SWAPPED>(e.prod_idx.0))? {
                        by_inner += g_by_inner.len(key as usize);
                    }
                }
            }
            ticker.poll(walk as u64)?;
            if by_inner < by_key {
                for e in live {
                    fold.set_outer(lim, e.g_idx.0, fold.outer_count::<SWAPPED>(e.prod_idx.0))?;
                }
                for i in 0..fold.opened_len() {
                    let sum = g_by_inner.bucket(fold.opened_key(i) as usize).iter()
                        .map(|r| u128::from(fold.outer_or_zero(r.other)))
                        .sum();
                    fold.set_sum_at(i, sum);
                }
                ticker.poll(by_inner as u64)?;
            } else {
                for e in live {
                    let count = fold.outer_count::<SWAPPED>(e.prod_idx.0);
                    for &RevEntry { other: key, .. } in g_by_outer.bucket(e.g_idx.idx()) {
                        fold.add_to_open(key, count);
                    }
                }
                ticker.poll(by_key as u64)?;
            }
            fold.add_weighted();
            ticker.poll(fold.opened_len() as u64)?;
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

/// Make f parent `p1`'s products of the candidates a walk wrote into
/// `level`'s pair arena from `base` on, the g parent of each in `owners`,
/// and push each product's node, holding its candidates' pairs.
///
/// Products are numbered in the order their g parent first appears among
/// the candidates, and each node lists its pairs in candidate order: the
/// order a stable sort of the candidates by product would give. One
/// candidate is its product's inline node, and candidates of one g parent
/// are one node whose pairs stay where they were written; a few are grouped
/// by comparison ([`emit_few`]), more through the map ([`emit_owned`]).
///
/// No sort and no dedup: pair lists are order-free, and canonical child
/// levels make the lookups injective, so a duplicate in a purely Boolean
/// diagram is an upstream canonicity violation. `duplicates_legal` says a
/// node's pair list may repeat a pair: some level of an operand, or of the
/// output so far, is marginal, so pair lists are multisets feeding a sum.
/// Read only by the debug duplicate checks.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn emit_walked(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    owners: &mut [u32],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let Some(&g_parent) = owners.first() else {
        return Ok(());
    };
    let product = ProductEntry {
        f_idx: FNodeIdx(p1 as u32),
        g_idx: GNodeIdx(g_parent),
        prod_idx: ProductNodeIdx(pl_output.len() as u32),
    };
    if owners.len() == 1 {
        lim.try_push(pl_output, product)?;
        let pair = level.pop_pair().expect("the walk wrote the candidate's pair");
        return emit_single_pair(eng, level, pair);
    }
    if owners.iter().all(|&g| g == g_parent) {
        lim.try_push(pl_output, product)?;
        reserve_room(lim, level.nodes.stored_mut(), 1)?;
        let node = finish_node(eng, level, base)?.expect("the walk wrote the candidates' pairs");
        // As in `emit_walked`: canonical children make a node's pairs
        // distinct unless pair lists are multisets.
        debug_assert!(
            duplicates_legal || {
                let mut seen = std::collections::HashSet::new();
                level.pairs_iter_of_idx(node.idx()).all(|p| seen.insert(p))
            },
            "sparse apply: duplicate pair emitted — canonicity violated"
        );
        return Ok(());
    }
    emit_mixed(eng, emit, level, pl_output, p1, base, owners, duplicates_legal)
}

/// [`emit_walked`] on candidates of more than one g parent: two g parents
/// taking turns in place ([`emit_alternating`]), a few others grouped by
/// comparison ([`emit_few`]), more through the map ([`emit_owned`]).
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn emit_mixed(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    owners: &mut [u32],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let n = owners.len();
    if let [a, b, ..] = *owners
        && matches!(n, 2 | 4 | 6 | 8)
        && owners.chunks_exact(2).all(|c| c[0] == a && c[1] == b)
    {
        return emit_alternating(eng, level, pl_output, p1, base, a, b, n, duplicates_legal);
    }
    if owners.len() <= FEW_CANDIDATES {
        // The pairs leave the arena, so the products' ranges take no more of
        // it than the walk gave them.
        let mut pairs = [ChildPair::new(EncodedChildRef::from_raw(0), EncodedChildRef::from_raw(0)); FEW_CANDIDATES];
        let arena = level.pairs.stored_mut();
        let n = owners.len();
        pairs[..n].copy_from_slice(&arena[base..base + n]);
        arena.truncate(base);
        return emit_few(eng, level, pl_output, p1, owners, &pairs[..n], duplicates_legal);
    }
    if owners.len() >= u32::MAX as usize {
        return Err(OperationError::IndexOverflow);
    }
    emit_owned(eng, emit, level, pl_output, p1, base, owners, duplicates_legal)
}

/// [`emit_mixed`] where the `n` candidates at `base`, an even number up to
/// [`FEW_CANDIDATES`], alternate between g parents `a` and `b`, the usual
/// node of f pairs whose leaf label meets both of g's: the products of `a`
/// and `b`, and their nodes as [`emit_few`] makes them, the pairs of each
/// moved into its half of the candidates' place.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn emit_alternating(
    eng: &Engine,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    a: u32,
    b: u32,
    n: usize,
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    use crate::diagram::EncodedNode;
    debug_assert!(a != b && matches!(n, 2 | 4 | 6 | 8), "two g parents taking turns");
    let lim = eng.limits();
    let first = pl_output.len();
    reserve_room(lim, pl_output, 2)?;
    for (k, g_parent) in [a, b].into_iter().enumerate() {
        pl_output.push(ProductEntry {
            f_idx: FNodeIdx(p1 as u32),
            g_idx: GNodeIdx(g_parent),
            prod_idx: ProductNodeIdx((first + k) as u32),
        });
    }
    reserve_room(lim, level.nodes.stored_mut(), 2)?;
    let arena = level.pairs.stored_mut();
    let half = n / 2;
    if half == 1 {
        let (pa, pb) = (arena[base], arena[base + 1]);
        arena.truncate(base);
        let nodes = level.nodes.stored_mut();
        nodes.push(EncodedNode::inline(pa));
        nodes.push(EncodedNode::inline(pb));
        return Ok(());
    }
    /// The even places' pairs, then the odd places', of `N` pairs.
    #[inline(always)]
    fn split<const N: usize>(pairs: &mut [ChildPair]) {
        let read: [ChildPair; N] = pairs[..N].try_into().expect("the candidates' place");
        for i in 0..N / 2 {
            pairs[i] = read[2 * i];
            pairs[N / 2 + i] = read[2 * i + 1];
        }
    }
    let place = &mut arena[base..base + n];
    match n {
        4 => split::<4>(place),
        6 => split::<6>(place),
        _ => split::<8>(place),
    }
    for start in [base, base + half] {
        level.try_push_multi_by_range(start, half).map_err(|_| OperationError::OverBudget)?;
    }
    debug_assert!(
        duplicates_legal || {
            let nodes = level.nodes().len();
            (nodes - 2..nodes).all(|i| {
                let mut seen = std::collections::HashSet::new();
                level.pairs_iter_of_idx(i).all(|p| seen.insert(p))
            })
        },
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}

/// [`emit_walked`] on more than [`FEW_CANDIDATES`] candidates of more than
/// one g parent, through `p2_map`, the map from a g parent to its product.
/// Each owner is replaced in place by its product's place among the
/// parent's products, and each pair is written into its product's range;
/// the ranges take no more of the arena than the walk gave the pairs.
///
/// Up to [`FEW_PRODUCTS`] products, each product's pairs are counted and
/// then gathered by a pass of their own over the candidates, whose cursor
/// stays in a register ([`gather_by_product`]). With more, one pass counts
/// them all and one writes each pair at its product's cursor, read from and
/// written back to memory.
#[expect(clippy::too_many_arguments)]
fn emit_owned(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    owners: &mut [u32],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let EmitScratch { p2_map, pair_counts, single_pairs, moved_pairs } = emit;
    let first = pl_output.len();
    let keyed = owners.iter_mut().try_for_each(|owner| {
        let g_parent = *owner;
        let slot = p2_map[g_parent as usize];
        *owner = if slot == NO_PRODUCT {
            let idx = pl_output.len() as u32;
            p2_map[g_parent as usize] = idx;
            lim.try_push(pl_output, ProductEntry {
                f_idx: FNodeIdx(p1 as u32),
                g_idx: GNodeIdx(g_parent),
                prod_idx: ProductNodeIdx(idx),
            })?;
            idx - first as u32
        } else {
            slot - first as u32
        };
        Ok(())
    });
    let first_node = level.nodes().len();
    let built = keyed.and_then(|()| {
        let products = pl_output.len() - first;
        reserve_room(lim, level.nodes.stored_mut(), products)?;
        let n = owners.len();
        let arena = level.pairs.stored_mut();
        if products == n {
            // Each pair is its own product's inline node, read where the
            // walk wrote it; the nodes have their room.
            let pairs = &arena[base..base + n];
            level.nodes.stored_mut().extend(pairs.iter().map(|&pair| crate::diagram::EncodedNode::inline(pair)));
            arena.truncate(base);
            return Ok(());
        }
        pair_counts.clear();
        lim.try_resize(pair_counts, products, 0)?;
        let few = products <= FEW_PRODUCTS;
        if few {
            for (k, count) in pair_counts.iter_mut().enumerate() {
                *count = owners.iter().filter(|&&o| o == k as u32).count() as u32;
            }
        } else {
            for &k in owners.iter() {
                pair_counts[k as usize] += 1;
            }
        }
        let total = prefix_offsets(pair_counts, true)? as usize;
        single_pairs.clear();
        lim.try_resize(single_pairs, products, arena[base])?;
        if few {
            if moved_pairs.len() <= total {
                lim.try_resize(moved_pairs, total + 1, arena[base])?;
            }
            gather_by_product(&arena[base..base + n], owners, pair_counts, single_pairs, moved_pairs);
            arena[base..base + total].copy_from_slice(&moved_pairs[..total]);
            arena.truncate(base + total);
        } else {
            moved_pairs.clear();
            lim.reserve(moved_pairs, n)?;
            moved_pairs.extend_from_slice(&arena[base..base + n]);
            arena.truncate(base + total);
            for (&k, &pair) in owners.iter().zip(moved_pairs.iter()) {
                let cursor = pair_counts[k as usize];
                if cursor == INLINE {
                    single_pairs[k as usize] = pair;
                } else {
                    arena[base + cursor as usize] = pair;
                    pair_counts[k as usize] = cursor + 1;
                }
            }
        }
        push_ranged_nodes(eng, level, base, pair_counts, single_pairs)
    });
    for e in &pl_output[first..] {
        p2_map[e.g_idx.idx()] = NO_PRODUCT;
    }
    built?;
    // As in `emit_walked`: canonical children make a node's pairs distinct
    // unless pair lists are multisets.
    debug_assert!(
        duplicates_legal || (first_node..level.nodes().len()).all(|i| {
            let mut seen = std::collections::HashSet::new();
            level.pairs_iter_of_idx(i).all(|p| seen.insert(p))
        }),
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}

/// The most products [`emit_owned`] gathers a pass each.
const FEW_PRODUCTS: usize = 4;

/// Gather `pairs` by product, `owners[i]` the product of `pairs[i]`, into
/// `into` for [`emit_owned`]: each product's pairs, in candidate order, at
/// the start `cursors` gives it, `cursors[k]` left at the range's end; a
/// product of one pair ([`INLINE`]) is its pair in `singles`.
///
/// Each product takes a pass over every candidate: the pair is written at
/// the product's cursor whatever its product, and the cursor steps past it
/// only for the product's own, so the next pair overwrites any other. The
/// last write of a pass lands at the range's end, one place past the last
/// range's, which `into` holds.
#[inline(always)]
fn gather_by_product(pairs: &[ChildPair], owners: &[u32], cursors: &mut [u32], singles: &mut [ChildPair], into: &mut [ChildPair]) {
    for (k, cursor) in cursors.iter_mut().enumerate() {
        let k = k as u32;
        if *cursor == INLINE {
            let at = owners.iter().position(|&o| o == k).expect("a product holds a candidate");
            singles[k as usize] = pairs[at];
            continue;
        }
        let mut at = *cursor as usize;
        for (&o, &pair) in owners.iter().zip(pairs) {
            into[at] = pair;
            at += usize::from(o == k);
        }
        *cursor = at as u32;
    }
}

/// The most candidates of more than one g parent that [`emit_few`] groups
/// by comparing their g parents, rather than through [`emit_owned`]'s map.
const FEW_CANDIDATES: usize = 8;

/// Make f parent `p1`'s products of two to [`FEW_CANDIDATES`] candidates of
/// more than one g parent, candidate `i` the pair `pairs[i]` of g parent
/// `owners[i]`: the products, nodes and pair ranges [`emit_owned`] makes
/// through its map, found by comparing the g parents with each other.
/// Products are numbered in the order their g parent first appears, a
/// product of one pair is its node inline, and those of more take the next
/// ranges of the arena in product order, each in candidate order.
#[inline(always)]
fn emit_few(
    eng: &Engine,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    owners: &[u32],
    pairs: &[ChildPair],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let n = owners.len();
    debug_assert!((2..=FEW_CANDIDATES).contains(&n) && pairs.len() == n, "a few candidates");
    // Each product's g parent, its candidates as a bit each, and its pair
    // count; and the pairs of the products of more than one, which take a
    // range of the arena.
    let mut parents = [0u32; FEW_CANDIDATES];
    let mut members = [0u8; FEW_CANDIDATES];
    let mut counts = [0u32; FEW_CANDIDATES];
    let (mut products, mut ranged) = (0, 0);
    for (i, &g_parent) in owners.iter().enumerate() {
        let k = parents[..products].iter().position(|&p| p == g_parent).unwrap_or(products);
        parents[k] = g_parent;
        products += usize::from(k == products);
        members[k] |= 1 << i;
        let count = counts[k] + 1;
        counts[k] = count;
        ranged += match count {
            1 => 0,
            2 => 2,
            _ => 1,
        };
    }
    let first = pl_output.len();
    reserve_room(lim, pl_output, products)?;
    pl_output.extend(parents[..products].iter().enumerate().map(|(k, &g_parent)| ProductEntry {
        f_idx: FNodeIdx(p1 as u32),
        g_idx: GNodeIdx(g_parent),
        prod_idx: ProductNodeIdx((first + k) as u32),
    }));
    // One reserve for the level's pairs, and the output-pair meter
    // charged once for whatever the arena grew by.
    let pre_pairs_cap = level.pairs.stored_mut().capacity();
    reserve_room(lim, level.nodes.stored_mut(), products)?;
    reserve_pairs_for_emit(eng, level, ranged)?;
    let first_node = level.nodes().len();
    // Each product's node in product order: one pair inline, more as the
    // next range of the arena, its candidates read off their bits in order.
    let mut built = Ok(());
    for k in 0..products {
        let mut bits = members[k];
        if counts[k] == 1 {
            built = emit_single_pair(eng, level, pairs[bits.trailing_zeros() as usize]);
        } else {
            let start = level.pairs.stored_len();
            let arena = level.pairs.stored_mut();
            while bits != 0 {
                arena.push(pairs[bits.trailing_zeros() as usize]);
                bits &= bits - 1;
            }
            built = level.try_push_multi_by_range(start, counts[k] as usize).map_err(|_| OperationError::OverBudget);
        }
        if built.is_err() {
            break;
        }
    }
    lim.charge_output_pairs(level.pairs.stored_mut().capacity().saturating_sub(pre_pairs_cap));
    built?;
    debug_assert!(
        duplicates_legal || (first_node..level.nodes().len()).all(|i| {
            let mut seen = std::collections::HashSet::new();
            level.pairs_iter_of_idx(i).all(|p| seen.insert(p))
        }),
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}

/// The cursor of a product of one pair ([`prefix_offsets`]'s mark): it has
/// no range, and any other cursor is below the total.
const INLINE: u32 = u32::MAX;

/// Push the nodes of a parent's products in product order, once each
/// product's pairs are in its range of the arena from `base` on: product
/// `k`'s range ends at `cursors[k]` and starts where the one before ended,
/// and a product marked [`INLINE`] is its one pair `singles[k]`.
fn push_ranged_nodes(
    eng: &Engine,
    level: &mut TddLevel,
    base: usize,
    cursors: &[u32],
    singles: &[ChildPair],
) -> Result<(), OperationError> {
    let mut start = 0u32;
    for (k, &end) in cursors.iter().enumerate() {
        if end == INLINE {
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
    lim.reserve(level.nodes.stored_mut(), 1)?;
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
            level.pairs_iter_of_idx(node.idx()).all(|p| seen.insert(p))
        },
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}
