//! The scatter joins that find a level's candidates, and the walks that
//! hand them to the emit.

use crate::diagram::EncodedChildRef;

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
/// pair lists may repeat a pair (see [`emit_parent`]), and the thresholds
/// the walk of f's nodes reads (from [`SparseThresholds::collected_pairs`]
/// on).
pub(super) struct Emit<'a> {
    pub(super) level: &'a mut TddLevel,
    pub(super) out: &'a mut Vec<ProductEntry>,
    pub(super) duplicates_legal: bool,
    pub(super) thresholds: SparseThresholds,
}

/// The f nodes [`scatter_routed`] walked by outer key: each node's row of
/// `rows`, [`NO_ROW`] for a node walked in order.
#[derive(Clone, Copy)]
struct ByOuter<'a> {
    row_of: &'a [u32],
    rows: &'a [Vec<ParEntry>],
}

/// No row of [`ByOuter`]: a node index never reaches `u32::MAX`.
const NO_ROW: u32 = u32::MAX;

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
) -> Result<(), OperationError> {
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
            // Each live product of the outer child with its slots, read once;
            // the table is then done with.
            let gathered = gather_owned_products(lim, pl_outer, f_outer, &ws.owners[..OWNER_SLOTS * g_outer],
                &mut ws.outer_offsets, &mut ws.owned);
            if !SWAPPED {
                clear_owner_table::<true>(g_level, g_outer, &mut ws.owners, &mut ws.owners_clean);
            } else {
                clear_owner_table::<false>(g_level, g_outer, &mut ws.owners, &mut ws.owners_clean);
            }
            ws.owned_len = gathered?;
            if ones {
                leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
                    |parts| OwnerWalk::<true>::of(parts, f_outer))?;
            } else {
                leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
                    |parts| OwnerWalk::<false>::of(parts, f_outer))?;
            }
            return Ok(());
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
    match inner {
        None => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            |parts| LabelWalk { g_by_outer: parts.g_by_outer, outer: parts.outer }),
        Some(Inner::Carried(Carrier::F)) => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            plain_walk(|f_field: u32, _: u32| f_field)),
        Some(Inner::Carried(Carrier::G)) => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            plain_walk(|_: u32, g_field: u32| g_field)),
        Some(Inner::Complete { g_width }) => leaf_walk::<SWAPPED, _>(eng, ws, f_level, shape, pl_outer, collect,
            plain_walk(move |f_node: u32, g_node: u32| (f_node as usize * g_width + g_node as usize) as u32)),
    }
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
    let SparseWorkspace { g_by_outer, owned, owned_len, outer_offsets, node_candidates, node_owners, emit, .. } = ws;
    let walk = walk(WalkParts {
        g_by_outer: g_by_outer.view(),
        owned: &owned[..*owned_len],
        outer: GroupedView { offsets: outer_offsets, entries: pl_outer },
    });
    walk_nodes::<SWAPPED, W>(eng, &walk, f_level, Walked::All(shape.f.here), collect, None, node_candidates, node_owners, emit)
}

/// The f nodes a walk of f's nodes visits: all of the level's, or those
/// whose bit is set in `bits`, one bit per node.
#[derive(Clone, Copy)]
enum Walked<'a> {
    All(usize),
    Marked { bits: &'a [u64], nodes: usize },
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

/// Walk f's nodes in order, each node's candidates found by `walk`, into
/// `collect`: into the level's nodes, each node emitted as its walk ends
/// ([`emit_walked`]), or, on a level with one product, its pair arena, a
/// count or sums. Into the level's nodes, a node `by_outer` holds a row for
/// is emitted from its row instead.
#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn walk_nodes<const SWAPPED: bool, W: NodeWalk>(
    eng: &Engine,
    walk: &W,
    f_level: &TddLevel,
    nodes: Walked<'_>,
    collect: Collect<'_, '_>,
    by_outer: Option<ByOuter<'_>>,
    node_candidates: &mut Vec<ParEntry>,
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
        // them. A node of many pairs collects its candidates for
        // `emit_parent` instead, which would otherwise read them back off
        // the arena.
        Collect::Nodes(Emit { level, out, duplicates_legal, thresholds }) => try_for_each_walked(f_level, nodes, |p1, pairs| {
            if W::ROWS
                && let Some(ByOuter { row_of, rows }) = by_outer
                && row_of[p1] != NO_ROW
            {
                let candidates = &rows[row_of[p1] as usize];
                if candidates.is_empty() {
                    return Ok(());
                }
                return emit_parent(eng, emit, level, out, p1, candidates, duplicates_legal);
            }
            if pairs.len() >= thresholds.collected_pairs {
                node_candidates.clear();
                walk.node::<SWAPPED>(pairs, &mut ticker, |entry| lim.try_push(node_candidates, entry))?;
                if node_candidates.is_empty() {
                    return Ok(());
                }
                return emit_parent(eng, emit, level, out, p1, node_candidates, duplicates_legal);
            }
            let base = level.pairs.stored_len();
            node_owners.clear();
            walk.node::<SWAPPED>(pairs, &mut ticker, |entry| {
                try_push_pair_into(eng, level, candidate_pair(&entry))?;
                lim.try_push(node_owners, entry.g_parent)
            })?;
            emit_walked(eng, emit, level, out, p1, base, node_owners, node_candidates, duplicates_legal)
        }),
        Collect::Direct(level) => try_for_each_walked(f_level, nodes, |_, pairs| {
            walk.node::<SWAPPED>(pairs, &mut ticker, |entry| try_push_pair_into(eng, level, candidate_pair(&entry)))
        }),
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
trait NodeWalk {
    /// Whether the walk of f's nodes may find a node's candidates in a row
    /// ([`ByOuter`]): only the routed walk's big nodes have one, and the
    /// leaf walks then carry no check for it.
    const ROWS: bool = false;

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
                prefetch_at(self.routes.slots, segment.base as usize + route_slot(key, segment.bits));
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
    const ROWS: bool = true;

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
        }
        Ok(())
    }
}

/// The general arm into the level's nodes, by f node: every outer key's
/// `filtered` index is built as [`scatter_general_arm`] builds it and kept
/// in [`Routes`], and f's nodes are then walked in order ([`RoutedWalk`]),
/// each emitted into `emit` as its walk ends. No candidate is held past its
/// node, except a big node's.
///
/// A node of [`SparseThresholds::by_outer_pairs`] pairs or more is walked
/// by outer key instead, as each outer's `filtered` is built and before it
/// is cleared: its pairs under the outer read the index while it is in
/// cache, where the walk in pair order would read a different outer's
/// routes at almost every pair. Its candidates are held in a row of its
/// own until the walk of f's nodes reaches it and emits them. An outer
/// holding no pair of a smaller node gets no route.
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
    build_scatter_indexes::<SWAPPED>(eng, ws, f_level, g_level, shape)?;
    build_inner_index::<SWAPPED>(eng, ws, g_level, shape)?;
    let lim = eng.limits();
    let (pl_inner, pl_outer) = if !SWAPPED { (pl.left, pl.right) } else { (pl.right, pl.left) };
    let f_pairs = ws.f_by_outer.offsets[if !SWAPPED { shape.f.right } else { shape.f.left }] as usize;
    // The big nodes' rows are taken out of the workspace while `sides`
    // borrows it.
    let mut row_of = std::mem::take(&mut ws.by_outer_row);
    let mut rows = std::mem::take(&mut ws.by_outer);
    let built = rows_by_outer(lim, f_level, shape.f.here, f_pairs, emit.thresholds.by_outer_pairs, &mut row_of)
        .and_then(|big| {
            rows.reset(lim, big)?;
            let mut s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer)?;
            let marked = route_level::<SWAPPED>(lim, &mut s, (big > 0).then_some(&row_of[..]), &mut rows, shape.f.here)?;
            Ok((big, marked))
        });
    ws.by_outer_row = row_of;
    ws.by_outer = rows;
    let (big, marked) = built?;
    let SparseWorkspace { routes, inner_offsets, node_candidates, node_owners, emit: scratch, by_outer_row, by_outer, .. } = ws;
    let walk = RoutedWalk {
        inner: GroupedView { offsets: inner_offsets, entries: pl_inner },
        routes: RoutesView { segments: &routes.segments, slots: &routes.slots, entries: &routes.entries },
        pipelined: std::mem::size_of_val(&routes.slots[..]) + std::mem::size_of_val(&routes.entries[..])
            >= emit.thresholds.pipelined_bytes,
    };
    let nodes = if marked {
        Walked::Marked { bits: &routes.walked, nodes: shape.f.here }
    } else {
        Walked::All(shape.f.here)
    };
    let by_outer = (big > 0).then_some(ByOuter { row_of: &by_outer_row[..], rows: &by_outer[..] });
    walk_nodes::<SWAPPED, _>(eng, &walk, f_level, nodes, Collect::Nodes(emit), by_outer, node_candidates, node_owners, scratch)
}

/// Give each of `f_level`'s `nodes` of at least `min_pairs` pairs a row in
/// `row_of`, in node order, and every other node [`NO_ROW`]; the rows
/// given. None where the level's `pairs` are fewer than `min_pairs`.
fn rows_by_outer(
    lim: &crate::limits::Limits,
    f_level: &TddLevel,
    nodes: usize,
    pairs: usize,
    min_pairs: usize,
    row_of: &mut Vec<u32>,
) -> Result<usize, OperationError> {
    if pairs < min_pairs {
        return Ok(0);
    }
    row_of.clear();
    lim.try_resize(row_of, nodes, NO_ROW)?;
    let mut rows = 0;
    for (p1, row) in row_of.iter_mut().enumerate() {
        if f_level.pair_count_at(p1) >= min_pairs {
            *row = rows as u32;
            rows += 1;
        }
    }
    Ok(rows)
}

/// Build every outer key's `filtered` index and keep it in the level's
/// routes, walking the f pairs under the outer of each node `row_of` gives
/// a row into that row of `rows` while the index is in cache. Whether the
/// walk of f's nodes visits only the nodes holding a pair under an outer
/// with a route ([`MARKED_SHARE`]), and if so their bits, a big node with
/// candidates among them.
fn route_level<const SWAPPED: bool>(
    lim: &crate::limits::Limits,
    s: &mut ScatterSides<'_>,
    row_of: Option<&[u32]>,
    rows: &mut Rows<ParEntry>,
    nodes: usize,
) -> Result<bool, OperationError> {
    s.routes.reset(lim, s.outer_k)?;
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    // The f pairs of nodes walked in order under an outer with a route:
    // the pairs the general arm would walk for them.
    let mut routed_pairs = 0usize;
    for outer in 0..s.outer_k {
        if s.outer.bucket(outer).is_empty() || s.f_pairs_under(outer) == 0 {
            continue;
        }
        s.build_for_outer(lim, outer)?;
        let mut in_order = s.f_pairs_under(outer);
        if let Some(row_of) = row_of.filter(|_| !s.filtered.touched.is_empty()) {
            let (inner, filtered) = (s.inner, s.filtered.index());
            in_order = 0;
            for &RevEntry { parent, other: inner1 } in s.f_by_outer.bucket(outer) {
                let row = row_of[parent as usize];
                if row == NO_ROW {
                    in_order += 1;
                    continue;
                }
                let row = &mut rows[row as usize];
                emit_candidates::<SWAPPED>(inner, filtered, inner1, &mut ticker, |entry| lim.try_push(row, entry))?;
            }
        }
        if in_order > 0 {
            if !s.filtered.touched.is_empty() {
                routed_pairs += in_order;
            }
            s.routes.seal(lim, outer, s.filtered.touched, s.filtered.buckets)?;
        }
        s.filtered.clear_touched();
    }
    // Where those are few of f's pairs, only the nodes holding one are
    // walked, as the general arm walks only them, and the big nodes with a
    // candidate.
    let marked = routed_pairs < s.f_by_outer.offsets[s.outer_k] as usize / MARKED_SHARE;
    if marked {
        s.routes.mark_parents(lim, nodes, s.f_by_outer)?;
        if let Some(row_of) = row_of {
            for (p1, &row) in row_of.iter().enumerate() {
                if row != NO_ROW && !rows[row as usize].is_empty() {
                    s.routes.walked[p1 >> 6] |= 1 << (p1 & 63);
                }
            }
        }
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
    let SparseWorkspace { est_counts, f_by_outer, g_by_outer, .. } = ws;
    let counts = EstCounts::of(est_counts, shape);
    // Keyed by the outer dimension: the right sibling normally, the left child
    // when swapped.
    if !SWAPPED {
        build_reverse_index::<true>(eng, f_level, shape.f.right, Some(counts.f_right), f_by_outer)?;
        build_reverse_index::<true>(eng, g_level, shape.g.right, Some(counts.g_right), g_by_outer)?;
    } else {
        build_reverse_index::<false>(eng, f_level, shape.f.left, Some(counts.f_left), f_by_outer)?;
        build_reverse_index::<false>(eng, g_level, shape.g.left, Some(counts.g_left), g_by_outer)?;
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
    /// Every outer's `filtered`, kept for [`scatter_routed`]'s walk.
    routes: &'w mut Routes,
    /// How many outer keys the emit loop walks.
    outer_k: usize,
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
        filtered, filtered_touched, filtered_held,
        wanted, wanted_epoch, wanted_keys, inner_seen, inner_seen_epoch,
        outer_keys, outer_keys_epoch, outer_attached, routes,
        ..
    } = ws;
    bucket_offsets(eng.limits(), pl_inner, inner_k, inner_offsets)?;
    bucket_offsets(eng.limits(), pl_outer, outer_k, outer_offsets)?;
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

    /// Fill `filtered` for one outer key, from whichever g index is the
    /// cheaper to walk, and only for the children the emit will read when
    /// finding those costs less than it saves.
    fn build_for_outer(&mut self, lim: &crate::limits::Limits, outer: usize) -> Result<(), OperationError> {
        let by_key = self.build_cost_by_key(outer);
        if by_key <= self.f_pairs_under(outer) {
            // The walk by the outer's g keys costs no more than the emit's
            // own walk of the f pairs under it, and the marking costs at
            // least that walk: neither it nor the filter it feeds can pay
            // for itself here.
            self.build_filtered_for_outer::<false>(lim, outer)
        } else {
            self.mark_wanted_for_outer(outer);
            if self.build_cost_by_inner() < by_key {
                self.build_filtered_by_inner(lim, outer)
            } else {
                self.build_filtered_for_outer::<true>(lim, outer)
            }
        }
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
    let mut s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer)?;

    // Amortized cancellation/deadline poll. The sparse join has no other
    // mid-level break, so a wide level could otherwise wait out an expired
    // deadline; the bail lands at a loop level `try_push`'s recovery already
    // covers, so the workspace stays reusable.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for outer in 0..s.outer_k {
        if s.outer.bucket(outer).is_empty() { continue; }
        s.build_for_outer(lim, outer)?;
        match &mut collect {
            Collect::Direct(level) => s.emit_direct_for_outer::<SWAPPED, FAR>(eng, level, outer, &mut ticker)?,
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
    let s = sides::<SWAPPED>(eng, ws, shape, pl_inner, pl_outer)?;
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

/// Emit one f parent: dedup its candidates' g parents into output products
/// and push each product's node, holding its candidates' pairs, onto `level`.
///
/// Products are numbered in the order their g parent first appears among
/// the candidates, and each node lists its pairs in candidate order: the
/// order a stable sort of the candidates by product would give. `p2_map`
/// maps a g parent to its product while the parent is emitted and is
/// restored to `NO_PRODUCT` before returning; a bail leaves entries behind,
/// which `WsGuard` repairs. A parent whose candidates all name one g
/// parent has one product and skips the map, as a parent with one
/// candidate does: each node of an operand that holds one node per value
/// of a shared block, at that block's levels. A parent with a few
/// candidates of several g parents skips it as well ([`emit_few`]).
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
pub(super) fn emit_parent(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    candidates: &[ParEntry],
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    if candidates.len() >= u32::MAX as usize {
        return Err(OperationError::IndexOverflow);
    }
    let EmitScratch { p2_map, pair_counts, single_pairs } = emit;
    let first = pl_output.len();
    if let [entry] = candidates {
        // One candidate is one product of one pair: no `g` parent to group
        // by, so the map is neither read nor written.
        lim.try_push(pl_output, ProductEntry {
            f_idx: FNodeIdx(p1 as u32),
            g_idx: GNodeIdx(entry.g_parent),
            prod_idx: ProductNodeIdx(first as u32),
        })?;
        return emit_single_pair(eng, level, candidate_pair(entry));
    }
    // Candidates of one g parent are the pairs of one product.
    let g_parent = candidates[0].g_parent;
    let one_product = candidates.iter().all(|entry| entry.g_parent == g_parent);
    if !one_product && candidates.len() <= FEW_CANDIDATES {
        let n = candidates.len();
        let mut owners = [0u32; FEW_CANDIDATES];
        let mut pairs = [ChildPair::new(EncodedChildRef::from_raw(0), EncodedChildRef::from_raw(0)); FEW_CANDIDATES];
        for (i, entry) in candidates.iter().enumerate() {
            owners[i] = entry.g_parent;
            pairs[i] = candidate_pair(entry);
        }
        return emit_few(eng, level, pl_output, p1, &owners[..n], &pairs[..n], duplicates_legal);
    }
    if one_product {
        lim.try_push(pl_output, ProductEntry {
            f_idx: FNodeIdx(p1 as u32),
            g_idx: GNodeIdx(g_parent),
            prod_idx: ProductNodeIdx(first as u32),
        })?;
    } else {
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
    }
    let products = pl_output.len() - first;
    let first_node = level.nodes().len();
    // The parent's pairs are appended to the level's arena under one reserve
    // and cut into nodes; the output-pair meter is charged once for the
    // growth, as the dense walk's choke point charges per growth event.
    // The level is built here, so its arena is stored.
    let pre_pairs_cap = level.pairs.stored_mut().capacity();
    reserve_room(lim, level.nodes.stored_mut(), products)?;
    let built = if products == candidates.len() {
        candidates.iter().try_for_each(|entry| emit_single_pair(eng, level, candidate_pair(entry)))
    } else if products == 1 {
        push_node_from(eng, level, candidates.iter().map(candidate_pair))
    } else {
        push_nodes_sorted(eng, level, pair_counts, single_pairs, candidates, &p2_map[..], first)
    };
    lim.charge_output_pairs(level.pairs.stored_mut().capacity().saturating_sub(pre_pairs_cap));
    if !one_product {
        for e in &pl_output[first..] {
            p2_map[e.g_idx.idx()] = NO_PRODUCT;
        }
    }
    built?;
    // No sort and no dedup: pair lists are order-free, and canonical child
    // levels make the grid lookups injective, so a duplicate in a purely
    // Boolean diagram is an upstream canonicity violation. Once any level
    // is marginal, duplicates are legal (`duplicates_legal`).
    debug_assert!(
        duplicates_legal || (first_node..level.nodes().len()).all(|i| {
            let mut seen = std::collections::HashSet::new();
            level.pairs_iter_of_idx(i).all(|p| seen.insert(p))
        }),
        "sparse apply: duplicate pair emitted — canonicity violated"
    );
    Ok(())
}

/// Make f parent `p1`'s products of the candidates a walk wrote into
/// `level`'s pair arena from `base` on, the g parent of each in `owners`:
/// the nodes and product list [`emit_parent`] makes of the same candidates.
/// One candidate is its product's inline node, and candidates of one g
/// parent are one node whose pairs stay where they were written; otherwise
/// the candidates are read back off the arena for [`emit_parent`], into
/// `candidates`.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn emit_walked(
    eng: &Engine,
    emit: &mut EmitScratch,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1: usize,
    base: usize,
    owners: &[u32],
    candidates: &mut Vec<ParEntry>,
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
        // As in `emit_parent`: canonical children make a node's pairs
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
    candidates.clear();
    lim.reserve(candidates, owners.len())?;
    let arena = level.pairs.stored_mut();
    candidates.extend(owners.iter().zip(&arena[base..]).map(|(&g_parent, pair)| ParEntry {
        g_parent,
        left_prod: pair.left.raw(),
        right_prod: pair.right.raw(),
    }));
    arena.truncate(base);
    emit_parent(eng, emit, level, pl_output, p1, candidates, duplicates_legal)
}

/// The most candidates of more than one g parent that [`emit_few`] groups
/// by comparing their g parents, rather than through [`emit_parent`]'s map.
const FEW_CANDIDATES: usize = 8;

/// Make f parent `p1`'s products of two to [`FEW_CANDIDATES`] candidates of
/// more than one g parent, candidate `i` the pair `pairs[i]` of g parent
/// `owners[i]`: the products, nodes and pair ranges [`emit_parent`] makes
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
    // Each candidate's product and each product's g parent and pair count,
    // and the pairs of the products of more than one, which take a range of
    // the arena. That sum is kept as the counts are: summed from them after
    // the loop, it read several counts in one load while their last writes
    // were still in flight, and such a load waits for the writes to retire.
    let mut product_of = [0u32; FEW_CANDIDATES];
    let mut parents = [0u32; FEW_CANDIDATES];
    let mut counts = [0u32; FEW_CANDIDATES];
    let (mut products, mut ranged) = (0, 0);
    for (i, &g_parent) in owners.iter().enumerate() {
        let k = parents[..products].iter().position(|&p| p == g_parent).unwrap_or(products);
        if k == products {
            parents[k] = g_parent;
            products += 1;
        }
        product_of[i] = k as u32;
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
    // As in `emit_parent`: one reserve for the level's pairs, and the
    // output-pair meter charged once for whatever the arena grew by.
    let pre_pairs_cap = level.pairs.stored_mut().capacity();
    reserve_room(lim, level.nodes.stored_mut(), products)?;
    reserve_pairs_for_emit(eng, level, ranged)?;
    let first_node = level.nodes().len();
    let built = counts[..products].iter().enumerate().try_for_each(|(k, &count)| {
        let mut mine = (0..n).filter(|&i| product_of[i] as usize == k).map(|i| pairs[i]);
        if count == 1 {
            emit_single_pair(eng, level, mine.next().expect("the product holds a candidate"))
        } else {
            let start = level.pairs.stored_len();
            level.pairs.stored_mut().extend(mine);
            level.try_push_multi_by_range(start, count as usize).map_err(|_| OperationError::OverBudget)
        }
    });
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

/// Push one node holding every pair of `pairs`, at least two, written
/// straight into the level's pair arena.
#[inline]
fn push_node_from(
    eng: &Engine,
    level: &mut TddLevel,
    pairs: impl ExactSizeIterator<Item = ChildPair>,
) -> Result<(), OperationError> {
    debug_assert!(pairs.len() >= 2, "a one-pair node is pushed inline");
    let start = level.pairs.stored_len();
    reserve_pairs_for_emit(eng, level, pairs.len())?;
    level.pairs.stored_mut().extend(pairs);
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
    let base = level.pairs.stored_len();
    let total = prefix_offsets(cursors, true)?;
    reserve_pairs_for_emit(eng, level, total as usize)?;
    let arena = level.pairs.stored_mut();
    arena.resize(base + total as usize, zero);
    singles.clear();
    eng.limits().try_resize(singles, cursors.len(), zero)?;
    for entry in candidates {
        let k = p2_map[entry.g_parent as usize] as usize - first;
        let cursor = cursors[k];
        if cursor == SINGLE {
            singles[k] = candidate_pair(entry);
        } else {
            arena[base + cursor as usize] = candidate_pair(entry);
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
