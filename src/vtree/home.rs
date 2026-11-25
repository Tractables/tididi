//! Projections of a root vtree on a subset of its leaves, keeping variable ids.
//!
//! The root is the projection's home. A projection's leaves are the home's
//! leaves it keeps, and its internal nodes are the home nodes with a kept leaf
//! under each child, so its shape depends only on the home and the leaf set.
//! Everything here costs time in the number of kept leaves and a logarithm of
//! the home's size; the one table sized by the home is its [`HomeIndex`],
//! built by [`Home::new`] and shared by all its projections.

use std::ops::Range;
use std::sync::Arc;

use super::build::InOrderBuilder;
use super::layout::SameLayout;
use super::topo::TopoOrder;
use super::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};
use crate::OperationError;
use crate::diagram::{TddBuildError, Weights};

/// In-order positions in a root vtree, and lowest common ancestors of its
/// leaves by range minimum over depths.
///
/// The leaves have ranks `0..n` left to right. Gap `g` lies between the
/// leaves of ranks `g` and `g + 1`; exactly one internal node splits its
/// leaves there, the lowest common ancestor of those two leaves.
#[derive(Debug)]
pub(crate) struct HomeIndex {
    /// By node: a leaf's rank, an internal node's gap.
    pub(super) slot: Vec<u32>,
    pub(super) leaf_at: Vec<VtreeIdx>,
    pub(super) var_at: Vec<VarId>,
    /// By variable id: its leaf's rank, `u32::MAX` for an id with no leaf.
    rank_of: Vec<u32>,
    pub(super) gap_node: Vec<VtreeIdx>,
    /// By node: its depth, the root's 0.
    depth: Vec<u32>,
    /// By gap: `depth << 32 | gap`, so that the least word of a range of gaps
    /// is its shallowest gap's.
    gap_word: Vec<u64>,
    /// By gap `g`, bit `t`: whether gap `s + t` of `g`'s block, which starts
    /// at `s`, is at or before `g` and has a smaller word than every gap after
    /// it up to `g`. The least word of gaps `lo..=g` within a block is so at
    /// the lowest bit of `in_block[g]` that is not before `lo`.
    in_block: Vec<u32>,
    /// By gap: the least word from the start of its block to it, and from it
    /// to the end of its block, so that a range over several blocks reads its
    /// two partial blocks in one load each.
    block_ends: Vec<[u64; 2]>,
    /// Row `j`, from `j * blocks`: the least word of blocks `i..i + 2^j` of
    /// [`GAP_BLOCK`] gaps, for each block `i` where they fit.
    block_least: Vec<u64>,
    blocks: usize,
}

/// The gaps per block of [`HomeIndex`]'s range minimum, the bits of an
/// [`HomeIndex::in_block`] mask.
const GAP_BLOCK: usize = 32;

impl HomeIndex {
    fn new(home: &Vtree) -> Self {
        let n = home.num_leaves() as usize;
        let gaps = n.saturating_sub(1);
        let mut slot = vec![0; home.num_nodes()];
        let mut node_depth = vec![0u32; home.num_nodes()];
        let (mut leaf_at, mut var_at, mut gap_node, mut gap_depth) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(gaps), Vec::<u32>::with_capacity(gaps));
        let mut stack = vec![(home.root(), 0u32, false)];
        while let Some((t, depth, left_done)) = stack.pop() {
            node_depth[t.idx()] = depth;
            match *home.node(t) {
                VtreeNode::Leaf { var, .. } => {
                    slot[t.idx()] = leaf_at.len() as u32;
                    leaf_at.push(t);
                    var_at.push(var);
                }
                VtreeNode::Internal { .. } if left_done => {
                    slot[t.idx()] = gap_node.len() as u32;
                    gap_node.push(t);
                    gap_depth.push(depth);
                }
                VtreeNode::Internal { left, right, .. } => {
                    stack.push((right, depth + 1, false));
                    stack.push((t, depth, true));
                    stack.push((left, depth + 1, false));
                }
            }
        }
        let gap_word: Vec<u64> = gap_depth.iter().enumerate().map(|(g, &depth)| u64::from(depth) << 32 | g as u64).collect();
        let blocks = gaps.div_ceil(GAP_BLOCK);
        let mut block_least: Vec<u64> =
            gap_word.chunks(GAP_BLOCK).map(|chunk| chunk.iter().copied().fold(u64::MAX, u64::min)).collect();
        let mut width = 1;
        while 2 * width <= blocks {
            let row = block_least.len() - blocks;
            for i in 0..blocks {
                let least = match i + 2 * width <= blocks {
                    true => block_least[row + i].min(block_least[row + i + width]),
                    false => u64::MAX,
                };
                block_least.push(least);
            }
            width *= 2;
        }
        let mut in_block = vec![0u32; gaps];
        let mut block_ends = vec![[0u64; 2]; gaps];
        for (b, words) in gap_word.chunks(GAP_BLOCK).enumerate() {
            let ends = &mut block_ends[b * GAP_BLOCK..b * GAP_BLOCK + words.len()];
            let (mut from_start, mut to_end) = (u64::MAX, u64::MAX);
            for t in 0..words.len() {
                from_start = from_start.min(words[t]);
                ends[t][0] = from_start;
                let r = words.len() - 1 - t;
                to_end = to_end.min(words[r]);
                ends[r][1] = to_end;
            }
            let mut mask = 0u32;
            for (t, &word) in words.iter().enumerate() {
                while mask != 0 && words[mask.ilog2() as usize] > word {
                    mask &= !(1 << mask.ilog2());
                }
                mask |= 1 << t;
                in_block[b * GAP_BLOCK + t] = mask;
            }
        }
        let mut rank_of = vec![u32::MAX; home.num_vars() as usize + 1];
        for (rank, var) in var_at.iter().enumerate() {
            rank_of[var.0 as usize] = rank as u32;
        }
        HomeIndex { slot, leaf_at, var_at, rank_of, gap_node, depth: node_depth, gap_word, in_block, block_ends, block_least, blocks }
    }

    /// The least word of gaps `lo..=last`, which lie in one block.
    #[inline]
    fn least_in_block(&self, lo: usize, last: usize) -> u64 {
        let start = last - last % GAP_BLOCK;
        let mask = self.in_block[last] & (u32::MAX << (lo - start));
        self.gap_word[start + mask.trailing_zeros() as usize]
    }

    /// The gap of the lowest common ancestor of the leaves of ranks `a < b`.
    pub(super) fn ancestor_gap(&self, a: u32, b: u32) -> u32 {
        self.ancestor_word(a, b) as u32
    }

    /// The word of gap `g`, `depth << 32 | g`.
    #[inline]
    pub(super) fn gap_word(&self, g: u32) -> u64 {
        self.gap_word[g as usize]
    }

    /// [`ancestor_gap`](Self::ancestor_gap) as `depth << 32 | gap`.
    #[inline]
    pub(super) fn ancestor_word(&self, a: u32, b: u32) -> u64 {
        debug_assert!(a < b);
        let (lo, end) = (a as usize, b as usize - 1);
        let (first, last) = (lo / GAP_BLOCK, end / GAP_BLOCK);
        if first == last {
            self.least_in_block(lo, end)
        } else {
            let ends = self.block_ends[lo][1].min(self.block_ends[end][0]);
            match last - first {
                1 => ends,
                span => {
                    let (i, j) = (first + 1, last);
                    let row = (span - 1).ilog2() as usize;
                    let at = row * self.blocks;
                    ends.min(self.block_least[at + i]).min(self.block_least[at + j - (1 << row)])
                }
            }
        }
    }
}

/// What a root vtree carries in one allocation: its index once it is a
/// home, and the weights of its variables once it is a weighted home. A clone
/// of the vtree is not a home, so `clone` gives no index: a copy that a search
/// rotates never carries an index of the shape it had. The weights are a fact
/// about the variables, which a rotation keeps, so a clone keeps them.
#[derive(Default)]
pub(crate) struct RootIndex(Option<Box<RootData>>);

struct RootData {
    index: Option<HomeIndex>,
    weights: Option<Weights>,
}

impl RootIndex {
    #[inline]
    fn home(&self) -> Option<&HomeIndex> {
        self.0.as_deref().and_then(|data| data.index.as_ref())
    }

    #[inline]
    fn weights(&self) -> Option<&Weights> {
        self.0.as_deref().and_then(|data| data.weights.as_ref())
    }

    fn new(index: Option<HomeIndex>, weights: Option<Weights>) -> Self {
        match index.is_none() && weights.is_none() {
            true => RootIndex(None),
            false => RootIndex(Some(Box::new(RootData { index, weights }))),
        }
    }

    fn take_home(&mut self) -> Option<HomeIndex> {
        let data = self.0.take()?;
        let RootData { index, weights } = *data;
        *self = RootIndex::new(None, weights);
        index
    }

    fn set_home(&mut self, index: Option<HomeIndex>) {
        let weights = self.0.take().and_then(|data| data.weights);
        *self = RootIndex::new(index, weights);
    }
}

impl Clone for RootIndex {
    fn clone(&self) -> Self {
        RootIndex::new(None, self.weights().cloned())
    }
}

impl std::fmt::Debug for RootIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootIndex")
            .field("indexed", &self.home().is_some())
            .field("weights", &self.weights())
            .finish()
    }
}

/// A root vtree that carries an index over its leaves: the home of its
/// projections ([`Vtree::projection`]), and itself the projection on all
/// of its leaves.
///
/// The leaves have ranks `0..n` left to right, and the leaves under a node
/// are a range of ranks. A `Home` does not change; a clone shares the vtree
/// and the index.
///
/// ```
/// use tididi::vtree::{Home, VarId, Vtree};
///
/// let home = Home::new(Vtree::balanced(8));
/// let own = Vtree::projection(&home, [VarId(7), VarId(2)])?;
/// let (left, right) = own.children(own.root());
/// assert_eq!((own.leaf_var(left), own.leaf_var(right)), (VarId(2), VarId(7)));
/// assert_eq!(own.home_node(own.root()), home.vtree().root());
/// assert_eq!(home.node_over(home.rank(VarId(2)).unwrap(), home.rank(VarId(7)).unwrap()), home.vtree().root());
/// assert_eq!(home.rank_range(home.vtree().root()), 0..8);
/// # Ok::<(), tididi::vtree::VtreeError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Home(Arc<Vtree>);

impl Home {
    /// The home with root `vtree`, its index built in time and memory linear
    /// in the vtree's size and kept in the vtree. On a home's root or on a
    /// projection, that home, in constant time.
    ///
    /// # Panics
    ///
    /// Panics if `vtree` is a root that is not a home and whose `Arc` is
    /// shared: the index cannot be set on it, and a home on a copy would not
    /// be the home of the diagrams already on it. Make the home first and
    /// share [`Home::vtree`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::vtree::{Home, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(4));
    /// let again = Home::new(Arc::clone(home.vtree()));
    /// assert!(Arc::ptr_eq(again.vtree(), home.vtree()));
    /// ```
    pub fn new(vtree: impl Into<Arc<Vtree>>) -> Self {
        let mut vtree = vtree.into();
        if let Some(home) = vtree.home() {
            return home;
        }
        let root = Arc::get_mut(&mut vtree).expect("Home::new: the vtree is shared and not a home; make the home first and share Home::vtree");
        root.index.set_home(Some(HomeIndex::new(root)));
        Home(vtree)
    }

    /// [`Home::new`] with `weights` for its variables: every diagram on the
    /// home is weighted, and a sum-out on it stores weight values computed
    /// under `weights` ([`Tdd::weighted_value`](crate::Tdd::weighted_value)).
    ///
    /// ```
    /// use tididi::{Tdd, Vtree};
    /// use tididi::diagram::{Arithmetic, RationalWeights, Weights};
    /// use tididi::vtree::Home;
    ///
    /// let weights = Weights::new(RationalWeights::unit(2), Arithmetic::ExactRational);
    /// let home = Home::weighted(Vtree::balanced(2), weights)?;
    /// let f = Tdd::clause(home.vtree(), [1, 2])?;
    /// assert_eq!(f.weighted_value()?.expect("the home is weighted").into_rational(), num_rational::BigRational::from_integer(3.into()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`TddBuildError::MissingVariableWeight`] when the table has no weight
    /// for a leaf's variable. [`TddBuildError::IncompatibleWeights`] when
    /// `vtree` is already a home's root or a projection, and that home does
    /// not have these weights: its diagrams were made under its own, so the
    /// weights of a home are fixed when it is made. On such a vtree with
    /// these weights, that home.
    ///
    /// # Panics
    ///
    /// As [`Home::new`]: if `vtree` is a root that is not a home and whose
    /// `Arc` is shared.
    ///
    /// [`TddBuildError::MissingVariableWeight`]: crate::diagram::TddBuildError::MissingVariableWeight
    /// [`TddBuildError::IncompatibleWeights`]: crate::diagram::TddBuildError::IncompatibleWeights
    pub fn weighted(vtree: impl Into<Arc<Vtree>>, weights: Weights) -> Result<Self, TddBuildError> {
        let mut vtree = vtree.into();
        if let Some(home) = vtree.home() {
            return match home.weights() == Some(&weights) {
                true => Ok(home),
                false => Err(TddBuildError::IncompatibleWeights),
            };
        }
        weights.check_variables(vtree.leaf_bottomup().map(|(_, var)| var))?;
        let root = Arc::get_mut(&mut vtree).expect("Home::weighted: the vtree is shared and not a home; make the home first and share Home::vtree");
        root.index = RootIndex::new(Some(HomeIndex::new(root)), Some(weights));
        Ok(Home(vtree))
    }

    /// The weights of the home's variables, `None` on a home made by
    /// [`Home::new`].
    pub fn weights(&self) -> Option<&Weights> {
        self.0.index.weights()
    }

    /// The root vtree.
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.0
    }

    pub(super) fn index(&self) -> &HomeIndex {
        self.0.index.home().expect("a home's root carries its index")
    }

    /// The depth of node `t` of the home's tree, the root's 0.
    #[inline]
    pub(crate) fn depth(&self, t: VtreeIdx) -> u32 {
        self.index().depth[t.idx()]
    }

    /// The rank of `var`'s leaf; `None` when it has none.
    #[inline]
    pub fn rank(&self, var: VarId) -> Option<u32> {
        self.index().rank_of.get(var.0 as usize).copied().filter(|&rank| rank != u32::MAX)
    }

    /// The lowest node with the leaves of ranks `a` and `b` under it: their
    /// lowest common ancestor, or the leaf when `a == b`. Logarithmic in the
    /// number of leaves.
    ///
    /// # Panics
    ///
    /// Panics if a rank is not below [`num_leaves`](Vtree::num_leaves).
    pub fn node_over(&self, a: u32, b: u32) -> VtreeIdx {
        let index = self.index();
        let (a, b) = (a.min(b), a.max(b));
        match a == b {
            true => index.leaf_at[a as usize],
            false => {
                assert!((b as usize) < index.leaf_at.len(), "node_over: rank {b} is not a leaf");
                index.gap_node[index.ancestor_gap(a, b) as usize]
            }
        }
    }

    /// The ranks of the leaves under node `h`. Logarithmic in the number of
    /// leaves, squared.
    ///
    /// # Panics
    ///
    /// Panics if `h` is not below [`num_nodes`](Vtree::num_nodes).
    pub fn rank_range(&self, h: VtreeIdx) -> Range<u32> {
        let index = self.index();
        let slot = index.slot[h.idx()];
        if self.vtree().node(h).is_leaf() {
            return slot..slot + 1;
        }
        let n = index.leaf_at.len() as u32;
        let first = first_in(0, slot, |r| index.ancestor_gap(r, slot + 1) == slot);
        let end = first_in(slot + 1, n, |r| index.ancestor_gap(slot, r) != slot);
        first..end
    }
}

/// The first `r` in `lo..hi` where `holds` does, `hi` if none; `holds` is
/// false and then true over the range.
fn first_in(mut lo: u32, mut hi: u32, holds: impl Fn(u32) -> bool) -> u32 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if holds(mid) { hi = mid } else { lo = mid + 1 }
    }
    lo
}

/// The number of ranks of `short` in `long`, both ascending, `short` not the
/// longer: a search of each in `long` when that is cheaper than a merge, so
/// that `short` inside a much longer `long` costs `short`'s length.
fn common_ranks(short: &[u32], long: &[u32]) -> usize {
    let searches = short.len() * (usize::BITS - long.len().leading_zeros()) as usize;
    if searches > short.len() + long.len() {
        let (mut i, mut j, mut common) = (0, 0, 0);
        while i < short.len() && j < long.len() {
            common += usize::from(short[i] == long[j]);
            let (x, y) = (short[i], long[j]);
            i += usize::from(x <= y);
            j += usize::from(y <= x);
        }
        return common;
    }
    let (mut from, mut common) = (0, 0);
    for &r in short {
        let j = from + long[from..].partition_point(|&l| l < r);
        common += usize::from(long.get(j) == Some(&r));
        from = j;
    }
    common
}

/// What [`InOrderBuilder::finish`] returns: the nodes, the root and the slot
/// table.
pub(super) type Built = (Vec<VtreeNode>, VtreeIdx, Vec<u32>);

/// The word of gap `i` of the `k` positions of `slot`: the gap staged there,
/// or the lowest common ancestor of the ranks on either side.
#[inline]
fn staged_word(index: &HomeIndex, slot: &[u32], k: usize, i: usize) -> u64 {
    match slot[k + i] {
        u32::MAX => index.ancestor_word(slot[i], slot[i + 1]),
        gap => index.gap_word(gap),
    }
}

/// By position of a vtree on a home: the childless node, its rank, the home
/// gap between it and the position before, and the internal node between it
/// and the position after.
struct Positions<C, R, G, I> {
    childless: C,
    rank: R,
    gap_before: G,
    internal: I,
}

/// What a projection knows of its home.
///
/// A projection on `k` childless nodes numbers them `0..k` (variable leaves
/// left to right, then regions left to right) and its internal nodes
/// `k..2k - 1`, as [`InOrderBuilder`] closes them. Rotations and child swaps
/// keep node indices, so what is stored by node stays true of a tree moved
/// in place; what is stored by position does not.
#[derive(Clone, Debug)]
pub(crate) struct HomeLink {
    pub(super) home: Home,
    /// `3k - 2` entries: the home ranks of the childless nodes by position,
    /// ascending (a variable leaf's rank, a region's anchor); the own
    /// internal node at each gap between positions; the home gap of each own
    /// internal node, by index minus `k`.
    pub(super) slot: Box<[u32]>,
    /// The regions, by position ascending.
    pub(super) regions: Box<[Region]>,
}

/// Home ranks as half-open intervals, ascending, neither touching nor
/// overlapping.
pub(crate) type Extent = Box<[(u32, u32)]>;

/// The home variables a region ([`VtreeNode::is_region`]) stands for, and
/// where it sits.
///
/// The anchor, `ranks[at]`, is a rank of the extent: the leftmost at
/// creation, and afterwards it moves only as a variable leaf's rank does.
/// The own shape is the projection on the anchors, so a change of the home
/// that makes another extent rank the leftmost does not change it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    /// The index of its anchor in [`HomeLink::ranks`].
    pub(super) at: u32,
    pub(super) extent: Extent,
    /// The number of ranks in `extent`.
    pub(super) count: u32,
}

impl Region {
    /// Whether the extent holds every rank of `lo..hi`, which is not empty.
    fn covers(&self, lo: u32, hi: u32) -> bool {
        let i = self.extent.partition_point(|&(_, end)| end <= lo);
        self.extent.get(i).is_some_and(|&(start, end)| start <= lo && hi <= end)
    }
}

impl HomeLink {
    /// The number of childless nodes.
    #[inline]
    fn k(&self) -> usize {
        self.slot.len().div_ceil(3)
    }

    /// The home ranks of the childless nodes, ascending.
    #[inline]
    pub(super) fn ranks(&self) -> &[u32] {
        &self.slot[..self.k()]
    }

    /// The ranks, the internal node by gap and the home gap by internal node
    /// minus `k`.
    #[inline]
    fn parts(&self) -> (&[u32], &[u32], &[u32]) {
        let k = self.k();
        let (ranks, rest) = self.slot.split_at(k);
        let (node_at, gap_of) = rest.split_at(k.saturating_sub(1));
        (ranks, node_at, gap_of)
    }

    /// The own internal node between positions `i` and `i + 1`.
    #[inline]
    fn node_at(&self, i: usize) -> VtreeIdx {
        VtreeIdx(self.slot[self.k() + i])
    }

    /// The home gap of own internal node `t`.
    #[inline]
    fn gap_of(&self, t: VtreeIdx) -> u32 {
        self.slot[self.k() - 1 + t.idx()]
    }

    /// The childless node at position `i`.
    #[inline]
    fn childless(&self, i: usize) -> VtreeIdx {
        match self.regions.is_empty() {
            true => VtreeIdx(i as u32),
            false => self.childless_among_regions(i),
        }
    }

    #[cold]
    fn childless_among_regions(&self, i: usize) -> VtreeIdx {
        let before = self.regions.partition_point(|region| (region.at as usize) < i);
        match self.regions.get(before).is_some_and(|region| region.at as usize == i) {
            true => VtreeIdx((self.k() - self.regions.len() + before) as u32),
            false => VtreeIdx((i - before) as u32),
        }
    }

    /// The position of childless node `t`.
    #[inline]
    fn position(&self, t: VtreeIdx) -> usize {
        match self.regions.is_empty() {
            true => t.idx(),
            false => self.position_among_regions(t),
        }
    }

    #[cold]
    fn position_among_regions(&self, t: VtreeIdx) -> usize {
        let leaves = self.k() - self.regions.len();
        if t.idx() >= leaves {
            return self.regions[t.idx() - leaves].at as usize;
        }
        let mut i = t.idx();
        for region in &self.regions {
            if region.at as usize > i {
                break;
            }
            i += 1;
        }
        i
    }

    /// The home node own node `t` stands for; for a region, its anchor's
    /// leaf.
    #[inline]
    fn home_node(&self, t: VtreeIdx) -> VtreeIdx {
        let index = self.home.index();
        match t.idx() < self.k() {
            true => index.leaf_at[self.ranks()[self.position(t)] as usize],
            false => index.gap_node[self.gap_of(t) as usize],
        }
    }

    pub(super) fn leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        let i = self.ranks().binary_search(&self.home.rank(var)?).ok()?;
        if !self.regions.is_empty() && self.regions.binary_search_by_key(&(i as u32), |r| r.at).is_ok() {
            return None;
        }
        Some(self.childless(i))
    }

    /// The region whose extent holds every rank of `lo..hi`.
    pub(super) fn region_over(&self, lo: u32, hi: u32) -> Option<&Region> {
        self.regions.iter().find(|region| region.covers(lo, hi))
    }

    fn region_node(&self, region: &Region) -> VtreeIdx {
        self.childless(region.at as usize)
    }

    /// The number of home variables under regions.
    fn in_extents(&self) -> u32 {
        self.regions.iter().map(|region| region.count).sum()
    }
}

/// The extent covering `intervals`, which do not overlap.
pub(crate) fn coalesce(mut intervals: Vec<(u32, u32)>) -> Extent {
    intervals.sort_unstable();
    let mut extent: Vec<(u32, u32)> = Vec::with_capacity(intervals.len());
    for (lo, hi) in intervals {
        match extent.last_mut() {
            Some(last) if last.1 == lo => last.1 = hi,
            _ => extent.push((lo, hi)),
        }
    }
    extent.into_boxed_slice()
}

/// Whether every rank of `short` is in `long`, both ascending; `found` is
/// given the position in `long` of each rank of `short` in turn, up to the
/// first missing one. Each rank is searched in what follows the last one
/// found, or both lists are merged once where `short.len()·log₂ long.len()`
/// exceeds `short.len() + long.len()`.
pub(crate) fn ranks_within(short: &[u32], long: &[u32], mut found: impl FnMut(usize)) -> bool {
    let log = (usize::BITS - long.len().leading_zeros()) as usize;
    let mut from = 0;
    if short.len().saturating_mul(log) > short.len() + long.len() {
        for &r in short {
            while long.get(from).is_some_and(|&x| x < r) {
                from += 1;
            }
            if long.get(from) != Some(&r) {
                return false;
            }
            found(from);
            from += 1;
        }
        return true;
    }
    for &r in short {
        match long[from..].binary_search(&r) {
            Ok(i) => {
                from += i;
                found(from);
                from += 1;
            }
            Err(_) => return false,
        }
    }
    true
}

/// How the leaf sets of two vtrees with one home compare.
#[derive(Debug)]
pub(crate) enum LeafSets {
    /// The same leaf set and the same layout.
    Equal(SameLayout),
    /// The first vtree's leaves are a proper subset of the second's.
    AInB,
    /// The second vtree's leaves are a proper subset of the first's.
    BInA,
    /// The projection on the union of both leaf sets.
    Union(Arc<Vtree>),
}

/// What [`Vtree::leaf_sets_reading`] gives the nodes of a union to.
pub(crate) trait UnionReader {
    /// The union has `k` childless nodes; called before any node is read.
    ///
    /// # Errors
    ///
    /// The reader's, which [`Vtree::leaf_sets_reading`] returns.
    fn begin(&mut self, k: usize) -> Result<(), OperationError>;

    /// How `a` and `b` read node `u` of the union: every leaf, then the
    /// internal nodes, children before parents.
    fn read(&mut self, u: VtreeIdx, readings: [Reading; 2]);
}

/// The reader [`Vtree::leaf_sets`] gives none to.
enum NoReader {}

impl UnionReader for NoReader {
    fn begin(&mut self, _: usize) -> Result<(), OperationError> {
        match *self {}
    }

    fn read(&mut self, _: VtreeIdx, _: [Reading; 2]) {
        match *self {}
    }
}

/// How an operand of a union reads a node of the union projection
/// ([`Vtree::leaf_sets_reading`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reading {
    /// The operand's own node.
    Own(VtreeIdx),
    /// The operand's leaves under the node are all under this child of it.
    Pass(VtreeIdx),
    /// No leaf of the operand is under the node.
    Free,
}

impl Vtree {
    /// The projection of `home` on the leaves of `vars`: the same variable
    /// ids, the same left and right order, and an internal node for each node
    /// of `home` with one of those leaves under each child.
    ///
    /// Two projections of one home on one leaf set have the same node indices.
    /// Repeated variables count once. Costs time in the number of variables
    /// times the logarithm of the home's size.
    ///
    /// A diagram on a projection does not depend on the home's other
    /// variables. [`Engine::and`](crate::Engine::and) of diagrams on two
    /// projections of one home returns a diagram on a projection holding both
    /// leaf sets; [`Engine::and_clause`](crate::Engine::and_clause) first
    /// moves the diagram onto the projection that adds the clause's
    /// variables; quantifying or conditioning a variable of the home with no
    /// leaf here returns the diagram, unless a condition names it in both
    /// polarities, which gives false. Model counts and evaluations are over
    /// the home's variables.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if `vars` is empty or names a variable with no
    /// leaf in `home`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(8));
    /// let own = Vtree::projection(&home, [VarId(7), VarId(2)])?;
    /// assert_eq!(own.num_leaves(), 2);
    /// assert!(Arc::ptr_eq(own.home().unwrap().vtree(), home.vtree()));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn projection(home: &Home, vars: impl IntoIterator<Item = VarId>) -> Result<Arc<Vtree>, VtreeError> {
        let vars = vars.into_iter();
        let mut words = Vec::with_capacity(vars.size_hint().0);
        for var in vars {
            let rank =
                home.rank(var).ok_or_else(|| VtreeError::Invalid(format!("variable {} has no leaf to project on", var.0)))?;
            words.push(u64::from(rank) << 32 | u64::from(var.0));
        }
        if words.is_empty() {
            return Err(VtreeError::Invalid("a projection needs at least one variable".to_string()));
        }
        words.sort_unstable();
        words.dedup();
        Ok(Self::seated(home, words.len(), false, || {
            let index = home.index();
            let mut build = InOrderBuilder::<true>::new(words.len());
            for &word in &words {
                build.leaf((word >> 32) as u32, VarId(word as u32));
            }
            let built = build.finish(|_, i| index.ancestor_word((words[i] >> 32) as u32, (words[i + 1] >> 32) as u32));
            Self::linked(home, built, Vec::new())
        }))
    }

    /// [`Vtree::projection`] on the variables of `ranks`, home ranks
    /// ascending without repeats, at least one.
    pub(crate) fn projection_on_ranks(home: &Home, ranks: Vec<u32>) -> Arc<Vtree> {
        Self::seated(home, ranks.len(), false, || Self::project_ranks(home, &ranks, Vec::new()))
    }

    /// The projection of `home` on no leaf, the vtree of a constant: no node,
    /// so [`root`](Self::root) names none, and the home's variables.
    ///
    /// Not cached in the home: the home holds no reference to its
    /// projections, and this one would hold the home.
    ///
    /// ```
    /// use tididi::Tdd;
    /// use tididi::vtree::{Home, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(5));
    /// let empty = Vtree::empty_projection(&home);
    /// assert_eq!(empty.num_nodes(), 0);
    /// assert!(Tdd::one(&empty).is_one());
    /// assert_eq!(Tdd::one(&empty).model_count().unwrap(), 32u32.into());
    /// ```
    pub fn empty_projection(home: &Home) -> Arc<Vtree> {
        let link = HomeLink { home: home.clone(), slot: Box::default(), regions: Box::default() };
        Arc::new(Vtree {
            context: Arc::clone(&home.vtree().context),
            nodes: Vec::new(),
            root: VtreeIdx(0),
            var_to_leaf: Vec::new(),
            leaf_count: 0,
            topo: TopoOrder::default(),
            home: Some(link),
            index: RootIndex::default(),
        })
    }

    /// The topmost own node at or under home node `h`, `None` when no own leaf
    /// or region anchor is under `h`; on a root vtree, `h` itself when it is
    /// a node. Logarithmic in [`num_leaves`](Self::num_leaves) times the
    /// logarithm of the home's size.
    pub fn own_at(&self, h: VtreeIdx) -> Option<VtreeIdx> {
        let Some(link) = self.home.as_ref() else {
            return (h.idx() < self.nodes.len()).then_some(h);
        };
        let (index, ranks) = (link.home.index(), link.ranks());
        let gap = *index.slot.get(h.idx())?;
        if link.home.vtree().node(h).is_leaf() {
            return self.node_at_home(h);
        }
        // The home leaves under `h` are a range of ranks around its gap, so
        // the own leaves under it are a range of own leaves.
        let under = |r: u32| {
            if r <= gap { index.ancestor_gap(r, gap + 1) == gap } else { index.ancestor_gap(gap, r) == gap }
        };
        let split = ranks.partition_point(|&r| r <= gap);
        let first = ranks[..split].partition_point(|&r| !under(r));
        let end = split + ranks[split..].partition_point(|&r| under(r));
        match end - first {
            0 => None,
            1 => Some(link.childless(first)),
            _ => self.node_at_home(index.gap_node[index.ancestor_gap(ranks[first], ranks[end - 1]) as usize]),
        }
    }

    /// The number of home leaves this vtree has neither as a leaf nor in an
    /// extent; 0 on a root.
    pub(crate) fn free_leaves(&self) -> u32 {
        self.home.as_ref().map_or(0, |link| link.home.vtree().num_leaves() - self.leaf_count - link.in_extents())
    }

    /// The home this vtree is on: a projection's home, or this vtree when it
    /// is a home's root, the projection of its home on every leaf; `None` on
    /// a root vtree that is not a home.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(4));
    /// let own = Vtree::projection(&home, [VarId(1), VarId(3)])?;
    /// assert!(Arc::ptr_eq(own.home().unwrap().vtree(), home.vtree()));
    /// assert!(Arc::ptr_eq(home.vtree().home().unwrap().vtree(), home.vtree()));
    /// assert!(Arc::new(Vtree::balanced(4)).home().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn home(self: &Arc<Self>) -> Option<Home> {
        match self.link_home() {
            Some(home) => Some(home.clone()),
            None => self.is_home().then(|| Home(Arc::clone(self))),
        }
    }

    /// Whether this vtree is on a home: [`home`](Self::home) without the
    /// clone.
    #[inline]
    pub(crate) fn on_home(&self) -> bool {
        self.home.is_some() || self.is_home()
    }

    /// The home of a projection that holds a link to it; `None` on a home's
    /// root, whose leaves and nodes are the home's own.
    #[inline]
    pub(crate) fn link_home(&self) -> Option<&Home> {
        self.home.as_ref().map(|link| &link.home)
    }

    /// The node of [`Vtree::home`] that own node `t` stands for; `t` itself on
    /// a root vtree.
    ///
    /// # Panics
    ///
    /// Panics if `t` is not below [`num_nodes`](Self::num_nodes).
    pub fn home_node(&self, t: VtreeIdx) -> VtreeIdx {
        match self.home.as_ref() {
            None => {
                assert!(t.idx() < self.nodes.len(), "home_node: {} is not a node", t.0);
                t
            }
            Some(link) => link.home_node(t),
        }
    }

    /// A projection without regions: its home and the home rank of each
    /// leaf, by node index, ascending. `None` on a root vtree or a projection
    /// with regions.
    #[inline]
    pub(crate) fn leaf_ranks(&self) -> Option<(&Home, &[u32])> {
        self.home.as_ref().filter(|link| link.regions.is_empty()).map(|link| (&link.home, link.ranks()))
    }

    /// The own node standing for home node `h`, if this vtree has one; on a
    /// root vtree, `h` itself when it is a node. Logarithmic in
    /// [`num_leaves`](Self::num_leaves).
    pub fn node_at_home(&self, h: VtreeIdx) -> Option<VtreeIdx> {
        let Some(link) = self.home.as_ref() else {
            return (h.idx() < self.nodes.len()).then_some(h);
        };
        debug_assert_eq!(self.validate_home(), Ok(()));
        let slot = *link.home.index().slot.get(h.idx())?;
        let ranks = link.ranks();
        if link.home.vtree().node(h).is_leaf() {
            let i = ranks.binary_search(&slot).ok()?;
            return Some(link.childless(i));
        }
        // Home leaves `slot` and `slot + 1` are the two sides of `h`'s gap.
        let right = ranks.partition_point(|&r| r <= slot);
        if right == 0 || right == ranks.len() {
            return None;
        }
        let own = link.node_at(right - 1);
        (link.gap_of(own) == slot).then_some(own)
    }

    /// Whether this vtree is a projection of `home`, or is `home`'s vtree
    /// itself: whether its diagrams combine with those on `home`'s other
    /// projections.
    pub fn is_on(&self, home: &Home) -> bool {
        match self.link_home() {
            Some(own) => Arc::ptr_eq(own.vtree(), home.vtree()),
            None => std::ptr::eq(self, &**home.vtree()),
        }
    }

    /// Whether `inner`, a projection with no region, has its leaves among
    /// fewer than all of this vtree's, which has no region and is on the same
    /// home or is it; if so, each leaf of `inner` and the leaf of this vtree
    /// with its variable are pushed to `leaves`, in rank order. The ranks are
    /// found as [`Vtree::leaf_sets`] finds them.
    pub(crate) fn nested_leaves(&self, inner: &Vtree, leaves: &mut Vec<(VtreeIdx, VtreeIdx)>) -> bool {
        let Some(li) = inner.home.as_ref().filter(|li| li.regions.is_empty()) else { return false };
        let own = |j: usize| VtreeIdx(j as u32);
        match self.home.as_ref() {
            None if std::ptr::eq(self, &**li.home.vtree()) => {
                let index = li.home.index();
                leaves.extend(li.ranks().iter().enumerate().map(|(j, &r)| (own(j), index.leaf_at[r as usize])));
                true
            }
            Some(ls) if ls.regions.is_empty() && Arc::ptr_eq(ls.home.vtree(), li.home.vtree()) && li.k() < ls.k() => {
                let mut j = 0;
                ranks_within(li.ranks(), ls.ranks(), |i| {
                    leaves.push((own(j), own(i)));
                    j += 1;
                })
            }
            _ => false,
        }
    }

    /// How the leaf sets of `a` and `b` compare, regions counted as leaves.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when their homes differ (a root vtree
    /// is its own home); [`OperationError::MarginalLevel`] naming a region of
    /// one whose extent holds a leaf of the other or meets one of its
    /// extents.
    pub(crate) fn leaf_sets(a: &Arc<Vtree>, b: &Arc<Vtree>) -> Result<LeafSets, OperationError> {
        Self::leaf_sets_reading(a, b, None::<&mut NoReader>)
    }

    /// [`OperationError::VtreeMismatch`] when `a` and `b` have different
    /// homes (a root vtree is its own home).
    pub(crate) fn refuse_other_home(a: &Arc<Vtree>, b: &Arc<Vtree>) -> Result<(), OperationError> {
        let root_of = |v: &Arc<Vtree>| Arc::as_ptr(v.link_home().map_or(v, Home::vtree));
        match root_of(a) == root_of(b) {
            true => Ok(()),
            false => Err(OperationError::VtreeMismatch),
        }
    }

    /// [`Vtree::leaf_sets`], giving `read`, in the union case of operands
    /// with no region, each node of the union with how `a` and `b` read it.
    ///
    /// # Errors
    ///
    /// Those of [`Vtree::leaf_sets`], and those of
    /// [`UnionReader::begin`].
    pub(crate) fn leaf_sets_reading(
        a: &Arc<Vtree>,
        b: &Arc<Vtree>,
        mut read: Option<&mut impl UnionReader>,
    ) -> Result<LeafSets, OperationError> {
        Self::refuse_other_home(a, b)?;
        debug_assert_eq!(a.validate_home(), Ok(()));
        debug_assert_eq!(b.validate_home(), Ok(()));
        let first_region = |link: &HomeLink| {
            debug_assert!(
                link.k() < link.home.index().leaf_at.len() || !link.regions.is_empty(),
                "a projection on every leaf is the home"
            );
            link.regions.first().map(|region| link.region_node(region))
        };
        let (la, lb) = match (a.home.as_ref(), b.home.as_ref()) {
            (None, None) => return Ok(LeafSets::Equal(SameLayout::new(a, b))),
            (Some(la), None) => return first_region(la).map_or(Ok(LeafSets::AInB), |t| Err(OperationError::MarginalLevel(t))),
            (None, Some(lb)) => return first_region(lb).map_or(Ok(LeafSets::BInA), |t| Err(OperationError::MarginalLevel(t))),
            (Some(la), Some(lb)) => (la, lb),
        };
        if !la.regions.is_empty() || !lb.regions.is_empty() {
            return Self::region_leaf_sets((a, la), (b, lb), false);
        }
        if Arc::ptr_eq(a, b) {
            return Ok(LeafSets::Equal(SameLayout::new(a, b)));
        }
        let (ra, rb) = (la.ranks(), lb.ranks());
        if ra == rb {
            return Ok(LeafSets::Equal(SameLayout::new(a, b)));
        }
        let (short, long) = if ra.len() <= rb.len() { (ra, rb) } else { (rb, ra) };
        let common = common_ranks(short, long);
        Ok(match (common == short.len(), ra.len() <= rb.len()) {
            (true, true) => LeafSets::AInB,
            (true, false) => LeafSets::BInA,
            (false, _) => {
                let k = long.len() + short.len() - common;
                if let Some(read) = read.as_deref_mut() {
                    read.begin(k)?;
                }
                match read {
                    None => LeafSets::Union(Self::seated(&la.home, k, false, || Self::union_of((a, la), (b, lb), k, None::<&mut fn(VtreeIdx, [Reading; 2], u32)>))),
                    Some(read) if k == la.home.index().leaf_at.len() => {
                        let index = la.home.index();
                        // The union's nodes are the home's: each built node is
                        // read under the id of the home node it stands for.
                        let mut home_of = vec![VtreeIdx(0); 2 * k - 1];
                        let mut read_home = |t: VtreeIdx, readings: [Reading; 2], seat: u32| {
                            let at = match t.idx() < k {
                                true => index.leaf_at[seat as usize],
                                false => index.gap_node[seat as usize],
                            };
                            home_of[t.idx()] = at;
                            read.read(at, readings.map(|r| match r {
                                Reading::Pass(c) => Reading::Pass(home_of[c.idx()]),
                                r => r,
                            }));
                        };
                        Self::union_of((a, la), (b, lb), k, Some(&mut read_home));
                        LeafSets::Union(Arc::clone(la.home.vtree()))
                    }
                    Some(read) => {
                        let mut read_own = |t: VtreeIdx, readings: [Reading; 2], _: u32| read.read(t, readings);
                        LeafSets::Union(Arc::new(Self::union_of((a, la), (b, lb), k, Some(&mut read_own))))
                    }
                }
            }
        })
    }

    /// The projection on the `k` merged ranks of `a` and `b`, which have no
    /// region. Two neighbours in the union that are neighbours in an operand
    /// meet where they do there; only neighbours from different operands are
    /// looked up. `read`, when given, is given each node as
    /// [`Vtree::leaf_sets_reading`] gives it, with the home rank of a leaf or
    /// the home gap of an internal node.
    fn union_of(
        (a, la): (&Vtree, &HomeLink),
        (b, lb): (&Vtree, &HomeLink),
        k: usize,
        mut read: Option<&mut impl FnMut(VtreeIdx, [Reading; 2], u32)>,
    ) -> Vtree {
        let index = la.home.index();
        let ((ra, na, ga), (rb, nb, gb)) = (la.parts(), lb.parts());
        let mut build = InOrderBuilder::<true>::new(k);
        // By position, how many ranks of each operand come before it.
        let mut before: Vec<[u32; 2]> = Vec::new();
        if read.is_some() {
            before.reserve_exact(k + 1);
        }
        let (mut i, mut j, mut u) = (0, 0, 0);
        let (mut last_a, mut last_b) = (false, false);
        while i < ra.len() || j < rb.len() {
            let (x, y) = (ra.get(i).copied().unwrap_or(u32::MAX), rb.get(j).copied().unwrap_or(u32::MAX));
            let (in_a, in_b) = (x <= y, y <= x);
            if u > 0 {
                build.stage(u - 1, match (last_a && in_a, last_b && in_b) {
                    (true, _) => ga[na[i - 1] as usize - ra.len()],
                    (_, true) => gb[nb[j - 1] as usize - rb.len()],
                    _ => u32::MAX,
                });
            }
            build.leaf(x.min(y), if in_a { a.leaf_var(VtreeIdx(i as u32)) } else { b.leaf_var(VtreeIdx(j as u32)) });
            if let Some(read) = read.as_deref_mut() {
                before.push([i as u32, j as u32]);
                let own = |inside: bool, at: usize| if inside { Reading::Own(VtreeIdx(at as u32)) } else { Reading::Free };
                read(VtreeIdx(u as u32), [own(in_a, i), own(in_b, j)], x.min(y));
            }
            i += usize::from(in_a);
            j += usize::from(in_b);
            u += 1;
            (last_a, last_b) = (in_a, in_b);
        }
        let gap = |slot: &[u32], i: usize| staged_word(index, slot, k, i);
        let built = match read {
            None => build.finish(gap),
            Some(read) => {
                before.push([i as u32, j as u32]);
                let node_at = [na, nb];
                build.finish_closing(gap, |c| {
                    let (lo, mid, hi) = (c.lo as usize, c.at as usize + 1, c.hi as usize + 1);
                    let reading = |x: usize| {
                        let (left, right) = (before[mid][x] - before[lo][x], before[hi][x] - before[mid][x]);
                        match (left > 0, right > 0) {
                            (true, true) => Reading::Own(VtreeIdx(node_at[x][before[mid][x] as usize - 1])),
                            (true, false) => Reading::Pass(c.left),
                            (false, true) => Reading::Pass(c.right),
                            (false, false) => Reading::Free,
                        }
                    };
                    read(c.t, [reading(0), reading(1)], c.home_gap);
                })
            }
        };
        Self::linked(&la.home, built, Vec::new())
    }

    /// The projection on this vtree's leaves and the leaves `added`, home
    /// ranks ascending with their variables, none of them a rank here; ids
    /// in closing order, so not the home on every leaf. Fills `map`, of
    /// this vtree's size, with each node's image, and gives `fresh` each
    /// internal node that is no image, children before parents: a
    /// [`Reading::Pass`] over the child with this vtree's leaves under it,
    /// or [`Reading::Free`]. Only gaps next to an added rank are looked up.
    /// This vtree is a projection with no region.
    pub(crate) fn extended(&self, added: &[(u32, VarId)], map: &mut [VtreeIdx], mut fresh: impl FnMut(VtreeIdx, Reading)) -> Vtree {
        let link = self.home.as_ref().expect("an extension is of a projection");
        debug_assert!(link.regions.is_empty(), "a projection with a region is extended by a union");
        debug_assert!(added.windows(2).all(|w| w[0].0 < w[1].0) && !added.is_empty());
        let index = link.home.index();
        let (ranks, node_at, gap_of) = link.parts();
        let (k, n) = (ranks.len(), ranks.len() + added.len());
        let mut build = InOrderBuilder::<true>::new(n);
        // By position, how many of this vtree's leaves come before it.
        let mut before: Vec<u32> = Vec::with_capacity(n + 1);
        let (mut i, mut j, mut last_own) = (0, 0, false);
        for u in 0..n {
            let own = j == added.len() || (i < k && ranks[i] < added[j].0);
            if u > 0 {
                build.stage(u - 1, if own && last_own { gap_of[node_at[i - 1] as usize - k] } else { u32::MAX });
            }
            before.push(i as u32);
            if own {
                build.leaf(ranks[i], self.leaf_var(VtreeIdx(i as u32)));
                map[i] = VtreeIdx(u as u32);
                i += 1;
            } else {
                build.leaf(added[j].0, added[j].1);
                j += 1;
            }
            last_own = own;
        }
        before.push(k as u32);
        let gap = |slot: &[u32], i: usize| staged_word(index, slot, n, i);
        let built = build.finish_closing(gap, |c| {
            let (lo, mid, hi) = (c.lo as usize, c.at as usize + 1, c.hi as usize + 1);
            match (before[mid] > before[lo], before[hi] > before[mid]) {
                (true, true) => map[node_at[before[mid] as usize - 1] as usize] = c.t,
                (true, false) => fresh(c.t, Reading::Pass(c.left)),
                (false, true) => fresh(c.t, Reading::Pass(c.right)),
                (false, false) => fresh(c.t, Reading::Free),
            }
        });
        Self::linked(&link.home, built, Vec::new())
    }

    /// The projection of `home` on the variable leaves of this vtree for which
    /// `keep` holds and on its regions; this vtree itself when that is all of
    /// them. Linear in this vtree's size, with no lookup in the home but its
    /// depths.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if this vtree is not on `home`, or if `keep`
    /// holds for no leaf and there is no region.
    ///
    /// ```
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(8));
    /// let own = Vtree::projection(&home, [VarId(1), VarId(4), VarId(6)])?;
    /// let sub = own.sub_projection(&home, |leaf| own.leaf_var(leaf) != VarId(4))?;
    /// assert!(sub.same_tree(&*Vtree::projection(&home, [VarId(1), VarId(6)])?));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn sub_projection(self: &Arc<Self>, home: &Home, keep: impl FnMut(VtreeIdx) -> bool) -> Result<Arc<Vtree>, VtreeError> {
        self.sub_projection_mapped(home, keep, None)
    }

    /// [`Vtree::sub_projection`], filling `map`, when given, by node of this
    /// vtree with the node of the result standing for the same home node, if
    /// any. On a result that is this vtree the map is the identity.
    pub(crate) fn sub_projection_mapped(
        self: &Arc<Self>,
        home: &Home,
        keep: impl FnMut(VtreeIdx) -> bool,
        map: Option<&mut Vec<Option<VtreeIdx>>>,
    ) -> Result<Arc<Vtree>, VtreeError> {
        if !self.is_on(home) {
            return Err(VtreeError::Invalid("a sub-projection is taken on the vtree's own home".to_string()));
        }
        debug_assert_eq!(self.validate_home(), Ok(()));
        let index = home.index();
        match self.home.as_ref() {
            Some(link) => {
                let (ranks, node_at, gap_of) = link.parts();
                let k = ranks.len();
                let positions = Positions {
                    childless: |i| link.childless(i),
                    rank: |i| ranks[i],
                    gap_before: |i| gap_of[node_at[i - 1] as usize - k],
                    internal: |i| VtreeIdx(node_at[i]),
                };
                self.sub_of(home, k, &link.regions, positions, keep, map)
            }
            None => {
                let positions = Positions {
                    childless: |i| index.leaf_at[i],
                    rank: |i| i as u32,
                    gap_before: |i| i as u32 - 1,
                    internal: |i| index.gap_node[i],
                };
                self.sub_of(home, index.leaf_at.len(), &[], positions, keep, map)
            }
        }
    }

    /// [`Vtree::sub_projection_mapped`] on the `k` positions of this vtree,
    /// regions `old` among them.
    #[inline]
    fn sub_of(
        self: &Arc<Self>,
        home: &Home,
        k: usize,
        old: &[Region],
        at: Positions<impl Fn(usize) -> VtreeIdx, impl Fn(usize) -> u32, impl Fn(usize) -> u32, impl Fn(usize) -> VtreeIdx>,
        mut keep: impl FnMut(VtreeIdx) -> bool,
        map: Option<&mut Vec<Option<VtreeIdx>>>,
    ) -> Result<Arc<Vtree>, VtreeError> {
        let index = home.index();
        // By kept position: its position here, the least home gap since the
        // kept position before, and the gap position where that gap is.
        let mut kept: Vec<(u32, u32, u32)> = Vec::with_capacity(k);
        let mut regions = Vec::new();
        let mut least = (u64::MAX, 0);
        for i in 0..k {
            if i > 0 {
                let word = index.gap_word((at.gap_before)(i));
                if word < least.0 {
                    least = (word, i as u32 - 1);
                }
            }
            if let Some(region) = old.get(regions.len()).filter(|region| region.at as usize == i) {
                regions.push(Region { at: kept.len() as u32, ..region.clone() });
            } else if !keep((at.childless)(i)) {
                continue;
            }
            kept.push((i as u32, least.0 as u32, least.1));
            least = (u64::MAX, 0);
        }
        if kept.len() == k {
            if let Some(map) = map {
                map.clear();
                map.extend((0..self.num_nodes() as u32).map(|t| Some(VtreeIdx(t))));
            }
            return Ok(Arc::clone(self));
        }
        if kept.is_empty() {
            return Err(VtreeError::Invalid("a projection needs at least one variable".to_string()));
        }
        let own = Self::seated(home, kept.len(), !regions.is_empty(), || {
            let mut build = InOrderBuilder::<true>::new(kept.len());
            let mut next_region = regions.iter().map(|region| region.at as usize).peekable();
            for (j, &(i, gap, _)) in kept.iter().enumerate() {
                let i = i as usize;
                if j > 0 {
                    build.stage(j - 1, gap);
                }
                match next_region.next_if_eq(&j) {
                    Some(_) => build.region((at.rank)(i)),
                    None => build.leaf((at.rank)(i), self.leaf_var((at.childless)(i))),
                }
            }
            let k = kept.len();
            Self::linked(home, build.finish(|slot, i| index.gap_word(slot[k + i])), regions)
        });
        if let Some(map) = map {
            map.clear();
            map.resize(self.num_nodes(), None);
            let to = own.home.as_ref().expect("a sub-projection has fewer leaves than its home");
            for (j, &(i, _, p)) in kept.iter().enumerate() {
                map[(at.childless)(i as usize).idx()] = Some(to.childless(j));
                if j > 0 {
                    map[(at.internal)(p as usize).idx()] = Some(to.node_at(j - 1));
                }
            }
        }
        Ok(own)
    }

    /// The projection of the common home of `a` and `b` on both leaf sets:
    /// `a` or `b` itself when it holds the other's leaves (the home when
    /// either is the home), else a new projection, the one
    /// [`Vtree::projection`] gives on the union of their variables.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when their homes differ (a root vtree
    /// is its own home).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(8));
    /// let a = Vtree::projection(&home, [VarId(1), VarId(5)])?;
    /// let b = Vtree::projection(&home, [VarId(2), VarId(5)])?;
    /// let both = Vtree::union(&a, &b).unwrap();
    /// assert!(both.same_tree(&*Vtree::projection(&home, [VarId(1), VarId(2), VarId(5)])?));
    /// assert!(Arc::ptr_eq(&Vtree::union(&a, &a).unwrap(), &a));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn union(a: &Arc<Vtree>, b: &Arc<Vtree>) -> Result<Arc<Vtree>, OperationError> {
        Ok(match Self::leaf_sets(a, b)? {
            LeafSets::Equal(_) | LeafSets::BInA => Arc::clone(a),
            LeafSets::AInB => Arc::clone(b),
            LeafSets::Union(union) => union,
        })
    }

    /// [`Vtree::leaf_sets`], except that a region of `a` with the anchor and
    /// extent of a region of `b` is one region of both: operands that summed
    /// out the same variables there.
    ///
    /// # Errors
    ///
    /// Those of [`Vtree::leaf_sets`], but for the regions both have.
    pub(crate) fn leaf_sets_sharing(a: &Arc<Vtree>, b: &Arc<Vtree>) -> Result<LeafSets, OperationError> {
        match (a.home.as_ref(), b.home.as_ref()) {
            (Some(la), Some(lb)) if !la.regions.is_empty() && !lb.regions.is_empty() => {
                Self::refuse_other_home(a, b)?;
                Self::region_leaf_sets((a, la), (b, lb), true)
            }
            _ => Self::leaf_sets(a, b),
        }
    }

    /// [`Vtree::leaf_sets`] of two projections of one home, one with a
    /// region; with `shared`, [`Vtree::leaf_sets_sharing`]. With no overlap
    /// but the shared regions, the operands share only variable leaves.
    #[cold]
    fn region_leaf_sets((a, la): (&Arc<Vtree>, &HomeLink), (b, lb): (&Arc<Vtree>, &HomeLink), shared: bool) -> Result<LeafSets, OperationError> {
        let anchor = |link: &HomeLink, region: &Region| link.ranks()[region.at as usize];
        let in_other = |link: &HomeLink, other: &HomeLink, region: &Region| {
            shared && other.regions.iter().any(|r| r.extent == region.extent && anchor(other, r) == anchor(link, region))
        };
        let mut intervals = Vec::new();
        for (link, other, first) in [(la, lb, true), (lb, la, false)] {
            for region in &link.regions {
                let node = link.region_node(region);
                if in_other(link, other, region) {
                    if first {
                        intervals.extend(region.extent.iter().map(|&(lo, hi)| (lo, hi, node)));
                    }
                    continue;
                }
                for &(lo, hi) in &region.extent {
                    let j = other.ranks().partition_point(|&r| r < lo);
                    if other.ranks().get(j).is_some_and(|&r| r < hi) {
                        return Err(OperationError::MarginalLevel(node));
                    }
                    intervals.push((lo, hi, node));
                }
            }
        }
        // Within one operand the extents are disjoint, so an interval that
        // starts before the furthest end so far meets the other operand's.
        intervals.sort_unstable_by_key(|&(lo, _, _)| lo);
        let mut furthest = 0;
        for (lo, hi, node) in intervals {
            if lo < furthest {
                return Err(OperationError::MarginalLevel(node));
            }
            furthest = furthest.max(hi);
        }
        let mut union = [la.ranks(), lb.ranks()].concat();
        union.sort_unstable();
        union.dedup();
        match (union.len() > lb.k(), union.len() > la.k()) {
            (false, false) => return Ok(LeafSets::Equal(SameLayout::new(a, b))),
            (false, true) => return Ok(LeafSets::AInB),
            (true, false) => return Ok(LeafSets::BInA),
            (true, true) => {}
        }
        let mut regions: Vec<Region> = [(la, lb), (lb, la)]
            .iter()
            .enumerate()
            .flat_map(|(i, &(link, other))| {
                link.regions.iter().filter(move |region| i == 0 || !in_other(link, other, region)).map(|region| {
                    let at = union.binary_search(&anchor(link, region)).expect("a position of an operand") as u32;
                    Region { at, ..region.clone() }
                })
            })
            .collect();
        regions.sort_unstable_by_key(|region| region.at);
        Ok(LeafSets::Union(Arc::new(Self::project_ranks(&la.home, &union, regions))))
    }

    /// Whether `var` has a leaf in the home but neither a leaf here nor a
    /// place in an extent: a variable a diagram on this vtree does not depend
    /// on.
    pub(crate) fn home_has(&self, var: VarId) -> bool {
        let Some(link) = self.home.as_ref() else { return false };
        let Some(rank) = link.home.rank(var) else { return false };
        self.leaf_of(var).is_none() && link.region_over(rank, rank + 1).is_none()
    }

    /// The region whose extent holds `var`: a home variable a diagram on
    /// this vtree has summed out, and which no operand it meets may have as
    /// a leaf. A region's [`home_node`](Self::home_node) is the home leaf it
    /// is anchored at, which need not be the only one in its extent.
    ///
    /// ```
    /// use tididi::{Engine, Tdd};
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let engine = Engine::new();
    /// let home = Home::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&Vtree::projection(&home, [VarId(1), VarId(2)])?, [1, 2])?;
    /// let g = Tdd::clause(&Vtree::projection(&home, [VarId(2), VarId(3)])?, [-2, 3])?;
    /// let summed = engine.and_marginalizing_where(&f, &g, |v| v != VarId(3))?;
    /// let region = summed.vtree().region_of(VarId(1)).expect("1 is summed out");
    /// assert_eq!(summed.vtree().region_of(VarId(2)), Some(region));
    /// assert_eq!(summed.vtree().region_of(VarId(3)), None);
    /// assert_eq!(summed.vtree().region_of(VarId(4)), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn region_of(&self, var: VarId) -> Option<VtreeIdx> {
        let link = self.home.as_ref()?;
        let rank = link.home.rank(var)?;
        link.region_over(rank, rank + 1).map(|region| link.region_node(region))
    }

    /// `var` has no leaf here: [`OperationError::MarginalLevel`] naming its
    /// region when an extent holds it, else
    /// [`OperationError::VariableNotInVtree`].
    #[cold]
    pub(crate) fn no_leaf_for(&self, var: VarId) -> OperationError {
        match self.region_of(var) {
            Some(region) => OperationError::MarginalLevel(region),
            None => OperationError::VariableNotInVtree(var),
        }
    }

    /// For `u`, the union of the leaf sets of projections `a` and `b`, whose
    /// root has, as its two children, the projections of `a`'s and `b`'s
    /// leaf sets: by own node of `a` and of `b`, the node of `u` standing for
    /// it, and whether `a` is on the left.
    ///
    /// A projection `u` is in the builder's numbering, so the positions and
    /// gaps of `a` and `b` are at offsets in its own. That holds only when the
    /// gap between the two sets is shallower than every gap inside them,
    /// which the test of the root's children gives. A home `u` keeps the
    /// numbering its builder and rotations left: there each node is the
    /// home node it stands for.
    pub(crate) fn split_at_root(u: &Vtree, a: &Vtree, b: &Vtree) -> Option<(Vec<VtreeIdx>, Vec<VtreeIdx>, bool)> {
        let (la, lb) = (a.home.as_ref()?, b.home.as_ref()?);
        if u.num_nodes() == 0 || u.node(u.root()).is_leaf() {
            return None;
        }
        let (l, r) = u.children(u.root());
        let (ha, hb) = (a.home_node(a.root()), b.home_node(b.root()));
        let Some(lu) = u.home.as_ref() else {
            let a_left = match (l, r) {
                (hl, hr) if hl == ha && hr == hb => true,
                (hl, hr) if hl == hb && hr == ha => false,
                _ => return None,
            };
            let home_nodes = |v: &Vtree, link: &HomeLink| (0..v.num_nodes() as u32).map(|t| link.home_node(VtreeIdx(t))).collect();
            return Some((home_nodes(a, la), home_nodes(b, lb), a_left));
        };
        let a_left = match (lu.home_node(l), lu.home_node(r)) {
            (hl, hr) if hl == ha && hr == hb => true,
            (hl, hr) if hl == hb && hr == ha => false,
            _ => return None,
        };
        let map = |link: &HomeLink, offset: usize| -> Vec<VtreeIdx> {
            let k = link.k();
            let mut map = vec![VtreeIdx(0); 2 * k - 1];
            for i in 0..k {
                map[link.childless(i).idx()] = lu.childless(offset + i);
            }
            for i in 0..k - 1 {
                map[link.node_at(i).idx()] = lu.node_at(offset + i);
            }
            map
        };
        let (a_offset, b_offset) = if a_left { (0, la.k()) } else { (lb.k(), 0) };
        Some((map(la, a_offset), map(lb, b_offset), a_left))
    }

    /// The projection of `home` on `k` childless nodes, `build` making it:
    /// the home's own vtree when they are all its leaves and none is a
    /// region. Every projection the library hands out comes from here, so
    /// one leaf set has one allocation when it is the whole home.
    fn seated(home: &Home, k: usize, regions: bool, build: impl FnOnce() -> Vtree) -> Arc<Vtree> {
        match !regions && k == home.index().leaf_at.len() {
            true => Arc::clone(home.vtree()),
            false => Arc::new(build()),
        }
    }

    /// Whether this vtree is a home's root.
    pub(crate) fn is_home(&self) -> bool {
        self.index.home().is_some()
    }

    /// The weights of this vtree's home, or of this root: those it was given
    /// as a weighted home, kept by a copy and a rotation.
    #[inline]
    pub(crate) fn weights(&self) -> Option<&Weights> {
        match &self.home {
            Some(link) => link.home.weights(),
            None => self.index.weights(),
        }
    }

    /// This root vtree, which is not a home, with `weights` for its
    /// variables: every diagram on it is weighted as on a
    /// [`Home::weighted`] and carries no weights of its own, and the level
    /// sum-outs of a root vtree
    /// ([`Engine::and_marginalizing`](crate::Engine::and_marginalizing),
    /// [`Engine::marginalize_levels`](crate::Engine::marginalize_levels))
    /// store weight values computed under `weights`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::diagram::{Arithmetic, RationalWeights, Weights};
    ///
    /// let weights = Weights::new(RationalWeights::unit(2), Arithmetic::ExactRational);
    /// let vtree = Arc::new(Vtree::balanced(2).with_weights(weights)?);
    /// let engine = Engine::new();
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let summed = engine.and_marginalizing(&f, engine.one(&vtree), &[vtree.root()])?;
    /// assert_eq!(summed.weighted_value()?.expect("the root is weighted").into_rational(), num_rational::BigRational::from_integer(3.into()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`TddBuildError::MissingVariableWeight`] when the table has no weight
    /// for a leaf's variable; [`TddBuildError::IncompatibleWeights`] on a
    /// home's root or a projection, which are weighted by
    /// [`Home::weighted`] alone, and on a root that already has other
    /// weights.
    ///
    /// [`TddBuildError::MissingVariableWeight`]: crate::diagram::TddBuildError::MissingVariableWeight
    /// [`TddBuildError::IncompatibleWeights`]: crate::diagram::TddBuildError::IncompatibleWeights
    pub fn with_weights(mut self, weights: Weights) -> Result<Vtree, TddBuildError> {
        if self.home.is_some() || self.is_home() || self.weights().is_some_and(|own| *own != weights) {
            return Err(TddBuildError::IncompatibleWeights);
        }
        weights.check_variables(self.leaf_bottomup().map(|(_, var)| var))?;
        self.index = RootIndex::new(None, Some(weights));
        Ok(self)
    }

    /// Give a root that is not a home the weights of the root it was made
    /// from.
    pub(crate) fn keep_weights_of(&mut self, from: &Vtree) {
        debug_assert!(self.home.is_none() && !self.is_home());
        self.index = RootIndex::new(None, from.weights().cloned());
    }

    /// `vtree` to change in place, copied first when shared. A home's root
    /// changed in place stops being a home here, before any change, so no
    /// diagram that can be cloned out of a search sees a stale index.
    #[inline]
    pub(crate) fn make_mut(vtree: &mut Arc<Vtree>) -> &mut Vtree {
        let own = Arc::make_mut(vtree);
        own.stop_being_home();
        own
    }

    /// Make the root `vtree`, which changed shape, a home again: the end of
    /// a search that kept moves on a home's root.
    #[cold]
    #[inline(never)]
    pub(crate) fn become_home(vtree: &mut Arc<Vtree>) {
        let own = Arc::make_mut(vtree);
        let index = HomeIndex::new(own);
        own.index.set_home(Some(index));
    }

    /// The index of a home's root, which stops being a home.
    pub(crate) fn take_index(&mut self) -> RootIndex {
        RootIndex::new(self.index.take_home(), None)
    }

    /// Put back the index [`take_index`](Self::take_index) took, the tree
    /// having its shape again.
    pub(crate) fn put_index(&mut self, mut index: RootIndex) {
        self.index.set_home(index.take_home());
    }

    /// Drop the index of a home's root about to change shape.
    #[inline]
    pub(crate) fn stop_being_home(&mut self) {
        if self.index.home().is_some() {
            self.index.set_home(None);
        }
    }

    /// The anchor rank and extent of the region that folding own node `t`
    /// of this projection makes: the ranks of the variable leaves under `t`
    /// and the extents of the regions under it, the anchor at the leftmost
    /// childless node under `t`. On a home's root, `t`'s leaves.
    pub(crate) fn folded_region(&self, t: VtreeIdx) -> (u32, Extent) {
        let mut leftmost = t;
        while let VtreeNode::Internal { left, .. } = *self.node(leftmost) {
            leftmost = left;
        }
        let leaves = self.subtree(t).filter(|&u| self.node(u).is_leaf());
        let Some(link) = self.home.as_ref() else {
            let slot = &self.index.home().expect("a projection or a home's root").slot;
            return (slot[leftmost.idx()], coalesce(leaves.map(|u| (slot[u.idx()], slot[u.idx()] + 1)).collect()));
        };
        let ranks = link.ranks();
        let mut intervals = Vec::new();
        for u in leaves {
            let i = link.position(u);
            match link.regions.binary_search_by_key(&(i as u32), |region| region.at) {
                Ok(j) => intervals.extend_from_slice(&link.regions[j].extent),
                Err(_) => intervals.push((ranks[i], ranks[i] + 1)),
            }
        }
        (ranks[link.position(leftmost)], coalesce(intervals))
    }

    /// The anchor variable and the variables of region `t` of this
    /// projection.
    pub(crate) fn region_vars(&self, t: VtreeIdx) -> (VarId, Vec<VarId>) {
        let link = self.home.as_ref().expect("a projection");
        let (anchor, extent) = self.region_at(t);
        let var = |rank: u32| link.home.vtree().leaf_var(link.home.index().leaf_at[rank as usize]);
        (var(anchor), extent.iter().flat_map(|&(lo, hi)| lo..hi).map(var).collect())
    }

    /// Whether this vtree has a region.
    #[inline]
    pub(crate) fn has_regions(&self) -> bool {
        self.num_leaf_nodes() != self.num_leaves() as usize
    }

    /// The number of childless nodes, variable leaves and regions; the
    /// nodes are stored leaves first, so these are its first nodes.
    #[inline]
    pub(crate) fn num_leaf_nodes(&self) -> usize {
        self.num_nodes().div_ceil(2)
    }

    /// The regions of this vtree, by position.
    pub(crate) fn regions(&self) -> impl Iterator<Item = VtreeIdx> + '_ {
        self.home.as_ref().into_iter().flat_map(|link| link.regions.iter().map(|region| link.region_node(region)))
    }

    /// The anchor rank and extent of region `t` of this projection.
    pub(crate) fn region_at(&self, t: VtreeIdx) -> (u32, Extent) {
        let link = self.home.as_ref().expect("a projection");
        let leaves = link.k() - link.regions.len();
        let region = t.idx().checked_sub(leaves).and_then(|j| link.regions.get(j)).expect("a region");
        (link.ranks()[region.at as usize], region.extent.clone())
    }

    /// The projection of `home` on the leaves of `ranks` and on `regions`,
    /// each an anchor rank and an extent holding it; together not empty, the
    /// extents holding no other rank.
    pub(crate) fn project_regions(home: &Home, mut ranks: Vec<u32>, regions: Vec<(u32, Extent)>) -> Arc<Vtree> {
        ranks.extend(regions.iter().map(|&(anchor, _)| anchor));
        ranks.sort_unstable();
        let mut regions: Vec<Region> = regions
            .into_iter()
            .map(|(anchor, extent)| Region {
                at: ranks.binary_search(&anchor).expect("an anchor") as u32,
                count: extent.iter().map(|&(lo, hi)| hi - lo).sum(),
                extent,
            })
            .collect();
        regions.sort_unstable_by_key(|region| region.at);
        Arc::new(Self::project_ranks(home, &ranks, regions))
    }

    /// The projection of `home` on the leaves of `ranks`, ascending and not
    /// empty, the positions of `regions` being regions.
    fn project_ranks(home: &Home, ranks: &[u32], regions: Vec<Region>) -> Vtree {
        let own = Self::linked(home, Self::build_ranks(home, ranks, &regions), regions);
        debug_assert_eq!(own.validate(), Ok(()));
        own
    }

    /// The nodes, root and slot table of [`Vtree::project_ranks`].
    pub(super) fn build_ranks(home: &Home, ranks: &[u32], regions: &[Region]) -> Built {
        let index = home.index();
        let mut build = InOrderBuilder::<true>::new(ranks.len());
        let mut next_region = regions.iter().map(|region| region.at as usize).peekable();
        for (i, &rank) in ranks.iter().enumerate() {
            match next_region.next_if_eq(&i) {
                Some(_) => build.region(rank),
                None => build.leaf(rank, index.var_at[rank as usize]),
            }
        }
        build.finish(|_, i| index.ancestor_word(ranks[i], ranks[i + 1]))
    }

    /// The projection `build` makes on `home`, `regions` its regions.
    pub(super) fn linked(home: &Home, (nodes, root, slot): Built, regions: Vec<Region>) -> Vtree {
        let leaf_count = (nodes.len().div_ceil(2) - regions.len()) as u32;
        let topo = TopoOrder::identity(&nodes, leaf_count);
        Vtree {
            context: Arc::clone(&home.vtree().context),
            nodes,
            root,
            var_to_leaf: Vec::new(),
            leaf_count,
            topo,
            home: Some(HomeLink { home: home.clone(), slot: slot.into_boxed_slice(), regions: regions.into_boxed_slice() }),
            index: RootIndex::default(),
        }
    }

    /// Kept out of line so that [`Vtree::leaf_of`] on a root vtree stays as
    /// small as it was before projections.
    #[cold]
    #[inline(never)]
    pub(super) fn projected_leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        self.home.as_ref()?.leaf_of(var)
    }

    /// [`Vtree::num_vars`] of a vtree with an empty leaf table, which only a
    /// projection has.
    #[cold]
    #[inline(never)]
    pub(super) fn projected_num_vars(&self) -> u32 {
        self.home.as_ref().map_or(0, |link| link.home.vtree().num_vars())
    }

    /// A projection's childless nodes in its home's order, a home's root's
    /// leaves in its own; nothing on another root.
    pub(crate) fn projected_leaves(&self) -> impl Iterator<Item = VtreeIdx> + '_ {
        let projected = self.home.as_ref().into_iter().flat_map(|link| (0..link.k()).map(|i| link.childless(i)));
        projected.chain(self.index.home().into_iter().flat_map(|index| index.leaf_at.iter().copied()))
    }

    /// This tree as a root over the variables `1..=k`, with the same node
    /// indices and leaf `t` named `name(t)`: a private tree whose leaf table
    /// is sized by `k`, not by a home's variables.
    pub(crate) fn local_twin(&self, name: impl Fn(VtreeIdx) -> VarId, k: u32) -> Vtree {
        let mut nodes = self.nodes.clone();
        let mut var_to_leaf = vec![VtreeIdx(0); k as usize];
        for (i, node) in nodes.iter_mut().enumerate() {
            if let VtreeNode::Leaf { var, .. } = node {
                debug_assert!(*var != super::REGION_VAR, "a private tree has no region");
                *var = name(VtreeIdx(i as u32));
                var_to_leaf[var.idx()] = VtreeIdx(i as u32);
            }
        }
        let twin = Vtree {
            context: Arc::clone(&self.context),
            nodes,
            root: self.root,
            var_to_leaf,
            leaf_count: self.leaf_count,
            topo: self.topo.clone(),
            home: None,
            index: RootIndex::default(),
        };
        debug_assert_eq!(twin.validate(), Ok(()));
        twin
    }

    /// The regions of `link` are this tree's regions, by position, each
    /// extent well formed, holding its anchor and no other position's rank,
    /// and no two extents meet.
    fn validate_regions(&self, link: &HomeLink) -> Result<(), VtreeError> {
        let fail = |msg: String| Err(VtreeError::Invalid(format!("projection: {msg}")));
        let n = link.home.vtree().num_leaves();
        let mut intervals = Vec::new();
        for (j, region) in link.regions.iter().enumerate() {
            let at = region.at as usize;
            if j > 0 && link.regions[j - 1].at >= region.at {
                return fail("the regions are not in ascending positions".to_string());
            }
            if at >= link.k() || !self.nodes.get(link.childless(at).idx()).is_some_and(VtreeNode::is_region) {
                return fail(format!("position {at} is not a region"));
            }
            let mut count = 0;
            for (i, &(lo, hi)) in region.extent.iter().enumerate() {
                if lo >= hi || hi > n || (i > 0 && region.extent[i - 1].1 >= lo) {
                    return fail(format!("the extent of position {at} is not ascending disjoint rank intervals"));
                }
                count += hi - lo;
                let inside = link.ranks().partition_point(|&r| r < hi) - link.ranks().partition_point(|&r| r < lo);
                let anchor_here = (lo..hi).contains(&link.ranks()[at]);
                if inside != usize::from(anchor_here) {
                    return fail(format!("the extent of position {at} holds another position's rank"));
                }
                intervals.push((lo, hi));
            }
            if count != region.count || !region.covers(link.ranks()[at], link.ranks()[at] + 1) {
                return fail(format!("the extent of position {at} does not hold its anchor or its count"));
            }
        }
        intervals.sort_unstable();
        if intervals.windows(2).any(|w| w[1].0 < w[0].1) {
            return fail("two extents meet".to_string());
        }
        Ok(())
    }

    /// A projection's link agrees with its home: its ranks are ascending home
    /// ranks, its regions are well formed (each extent holds its anchor and
    /// meets neither another extent nor a leaf's rank), and its nodes and slot
    /// table are the ones the builder gives its rank list: each leaf carries
    /// its home leaf's variable, each internal node stands for the lowest
    /// common ancestor in the home of the childless nodes on either side of
    /// it.
    pub(crate) fn validate_home(&self) -> Result<(), VtreeError> {
        let Some(link) = self.home.as_ref() else {
            return Ok(());
        };
        let fail = |msg: &str| Err(VtreeError::Invalid(format!("projection: {msg}")));
        let k = self.num_leaves() as usize + link.regions.len();
        if link.k() != k || link.slot.len() != (3 * k).saturating_sub(2) || self.nodes.len() != (2 * k).saturating_sub(1) {
            return fail("the link does not fit the tree");
        }
        if k == 0 {
            return Ok(());
        }
        let ranks = link.ranks();
        if ranks.windows(2).any(|w| w[0] >= w[1]) || ranks[k - 1] >= link.home.vtree().num_leaves() {
            return fail("the ranks are not ascending home ranks");
        }
        self.validate_regions(link)?;
        let (nodes, root, slot) = Self::build_ranks(&link.home, ranks, &link.regions);
        if root != self.root || nodes != self.nodes {
            return fail("the nodes are not the ones the builder gives their rank list");
        }
        if *slot != *link.slot {
            return fail("the slot table is not the one the builder gives the rank list");
        }
        Ok(())
    }
}
