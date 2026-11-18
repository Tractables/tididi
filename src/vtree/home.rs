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

use super::build::in_order_tree;
use super::layout::SameLayout;
use super::topo::TopoOrder;
use super::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};
use crate::OperationError;

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
    pub(super) gap_node: Vec<VtreeIdx>,
    /// By gap: `depth << 32 | gap`, so that the least word of a range of gaps
    /// is its shallowest gap's.
    gap_word: Vec<u64>,
    /// By gap `g`, bit `t`: whether gap `s + t` of `g`'s block, which starts
    /// at `s`, is at or before `g` and has a smaller word than every gap after
    /// it up to `g`. The least word of gaps `lo..=g` within a block is so at
    /// the lowest bit of `in_block[g]` that is not before `lo`.
    in_block: Vec<u32>,
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
        let (mut leaf_at, mut gap_node, mut gap_depth) =
            (Vec::with_capacity(n), Vec::with_capacity(gaps), Vec::<u32>::with_capacity(gaps));
        let mut stack = vec![(home.root(), 0u32, false)];
        while let Some((t, depth, left_done)) = stack.pop() {
            match *home.node(t) {
                VtreeNode::Leaf { .. } => {
                    slot[t.idx()] = leaf_at.len() as u32;
                    leaf_at.push(t);
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
        for (b, words) in gap_word.chunks(GAP_BLOCK).enumerate() {
            let mut mask = 0u32;
            for (t, &word) in words.iter().enumerate() {
                while mask != 0 && words[mask.ilog2() as usize] > word {
                    mask &= !(1 << mask.ilog2());
                }
                mask |= 1 << t;
                in_block[b * GAP_BLOCK + t] = mask;
            }
        }
        HomeIndex { slot, leaf_at, gap_node, gap_word, in_block, block_least, blocks }
    }

    /// The least word of gaps `lo..=last`, which lie in one block.
    fn least_in_block(&self, lo: usize, last: usize) -> u64 {
        let start = last - last % GAP_BLOCK;
        let mask = self.in_block[last] & (u32::MAX << (lo - start));
        self.gap_word[start + mask.trailing_zeros() as usize]
    }

    pub(super) fn gap_depth(&self, g: u32) -> u32 {
        (self.gap_word[g as usize] >> 32) as u32
    }

    /// The gap of the lowest common ancestor of the leaves of ranks `a < b`.
    pub(super) fn ancestor_gap(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a < b);
        let (lo, end) = (a as usize, b as usize - 1);
        let (first, last) = (lo / GAP_BLOCK, end / GAP_BLOCK);
        let least = if first == last {
            self.least_in_block(lo, end)
        } else {
            let ends = self.least_in_block(lo, (first + 1) * GAP_BLOCK - 1).min(self.least_in_block(last * GAP_BLOCK, end));
            match last - first {
                1 => ends,
                span => {
                    let (i, j) = (first + 1, last);
                    let row = (span - 1).ilog2() as usize;
                    let at = row * self.blocks;
                    ends.min(self.block_least[at + i]).min(self.block_least[at + j - (1 << row)])
                }
            }
        };
        least as u32
    }

    fn depth_of(&self, home: &Vtree, h: VtreeIdx) -> u32 {
        match home.node(h) {
            VtreeNode::Leaf { .. } => u32::MAX,
            VtreeNode::Internal { .. } => self.gap_depth(self.slot[h.idx()]),
        }
    }
}

/// The index a root vtree carries once it is a home. A clone of the vtree is
/// not a home, so `clone` gives none: a copy that a search rotates never
/// carries an index of the shape it had.
#[derive(Default)]
pub(crate) struct RootIndex(Option<Box<HomeIndex>>);

impl Clone for RootIndex {
    fn clone(&self) -> Self {
        RootIndex(None)
    }
}

impl std::fmt::Debug for RootIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "RootIndex(indexed)" } else { "RootIndex(None)" })
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
            return home.clone();
        }
        if vtree.index.0.is_none() {
            let root = Arc::get_mut(&mut vtree).expect("Home::new: the vtree is shared and not a home; make the home first and share Home::vtree");
            root.index = RootIndex(Some(Box::new(HomeIndex::new(root))));
        }
        Home(vtree)
    }

    /// The root vtree.
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.0
    }

    pub(super) fn index(&self) -> &HomeIndex {
        self.0.index.0.as_deref().expect("a home's root carries its index")
    }

    /// The rank of `var`'s leaf; `None` when it has none.
    pub fn rank(&self, var: VarId) -> Option<u32> {
        self.vtree().leaf_of(var).map(|leaf| self.index().slot[leaf.idx()])
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

/// What a projection knows of its home.
#[derive(Clone, Debug)]
pub(crate) struct HomeLink {
    pub(super) home: Home,
    /// The home ranks of the own childless nodes, ascending: a variable
    /// leaf's rank, a region's anchor.
    pub(super) ranks: Vec<u32>,
    /// By own node: the home node; for a region, its anchor's leaf.
    pub(super) node_of: Vec<VtreeIdx>,
    /// The own nodes in order: childless node `i` at `2i`, the node between
    /// childless nodes `i` and `i + 1` at `2i + 1`.
    pub(super) in_order: Vec<VtreeIdx>,
    /// The regions, by position ascending.
    pub(super) regions: Vec<Region>,
}

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
    /// Home ranks as half-open intervals, ascending, neither touching nor
    /// overlapping.
    pub(super) extent: Box<[(u32, u32)]>,
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
    pub(super) fn leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        let i = self.ranks.binary_search(&self.home.rank(var)?).ok()?;
        if !self.regions.is_empty() && self.regions.binary_search_by_key(&(i as u32), |r| r.at).is_ok() {
            return None;
        }
        Some(self.in_order[2 * i])
    }

    /// The home gap at which childless nodes `i - 1` and `i` meet.
    fn gap_before(&self, i: usize) -> u32 {
        self.home.index().slot[self.node_of[self.in_order[2 * i - 1].idx()].idx()]
    }

    /// The region whose extent holds every rank of `lo..hi`.
    fn region_over(&self, lo: u32, hi: u32) -> Option<&Region> {
        self.regions.iter().find(|region| region.covers(lo, hi))
    }

    fn region_node(&self, region: &Region) -> VtreeIdx {
        self.in_order[2 * region.at as usize]
    }

    /// The number of home variables under regions.
    fn in_extents(&self) -> u32 {
        self.regions.iter().map(|region| region.count).sum()
    }
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
        let mut ranks = Vec::new();
        for var in vars {
            let rank =
                home.rank(var).ok_or_else(|| VtreeError::Invalid(format!("variable {} has no leaf to project on", var.0)))?;
            ranks.push(rank);
        }
        if ranks.is_empty() {
            return Err(VtreeError::Invalid("a projection needs at least one variable".to_string()));
        }
        ranks.sort_unstable();
        ranks.dedup();
        Ok(Self::seated(home, ranks.len(), false, || Self::project_ranks(home, ranks, Vec::new())))
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
        let link = HomeLink {
            home: home.clone(),
            ranks: Vec::new(),
            node_of: Vec::new(),
            in_order: Vec::new(),
            regions: Vec::new(),
        };
        Arc::new(Vtree {
            context: Arc::clone(&home.vtree().context),
            nodes: Vec::new(),
            root: VtreeIdx(0),
            var_to_leaf: Vec::new(),
            leaf_count: 0,
            topo: TopoOrder::default(),
            home: Some(Arc::new(link)),
            index: RootIndex::default(),
        })
    }

    /// The topmost own node at or under home node `h`, `None` when no own leaf
    /// is under `h`; the region whose extent holds every home leaf under
    /// `h`, if one does; on a root vtree, `h` itself when it is a node.
    /// Logarithmic in [`num_leaves`](Self::num_leaves) times the logarithm
    /// of the home's size, and linear in the number of regions.
    pub fn own_at(&self, h: VtreeIdx) -> Option<VtreeIdx> {
        let Some(link) = self.home.as_deref() else {
            return (h.idx() < self.nodes.len()).then_some(h);
        };
        let (index, ranks) = (link.home.index(), &link.ranks);
        let gap = *index.slot.get(h.idx())?;
        if !link.regions.is_empty() {
            let under = link.home.rank_range(h);
            if let Some(region) = link.region_over(under.start, under.end) {
                return Some(link.region_node(region));
            }
        }
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
            1 => Some(link.in_order[2 * first]),
            _ => self.node_at_home(index.gap_node[index.ancestor_gap(ranks[first], ranks[end - 1]) as usize]),
        }
    }

    /// The number of home leaves this vtree has neither as a leaf nor in an
    /// extent; 0 on a root.
    pub(crate) fn free_leaves(&self) -> u32 {
        self.home.as_deref().map_or(0, |link| link.home.vtree().num_leaves() - self.leaf_count - link.in_extents())
    }

    /// The variables of the home leaves this vtree has neither as a leaf nor
    /// in an extent, left to right; none on a root.
    pub(crate) fn free_vars(&self) -> Vec<VarId> {
        let Some(link) = self.home.as_deref() else { return Vec::new() };
        let mut own = link.ranks.iter().copied().peekable();
        let mut extents: Vec<(u32, u32)> = link.regions.iter().flat_map(|region| region.extent.iter().copied()).collect();
        extents.sort_unstable();
        let mut extents = extents.into_iter().peekable();
        let mut free = Vec::with_capacity(self.free_leaves() as usize);
        for (rank, &leaf) in link.home.index().leaf_at.iter().enumerate() {
            let rank = rank as u32;
            while extents.next_if(|&(_, end)| end <= rank).is_some() {}
            let in_extent = extents.peek().is_some_and(|&(start, _)| start <= rank);
            if own.next_if_eq(&rank).is_none() && !in_extent {
                free.push(link.home.vtree().leaf_var(leaf));
            }
        }
        free
    }

    /// The variables of the home leaves under own node `t`'s home node that
    /// this vtree does not have; none on a root.
    pub(crate) fn free_vars_under(&self, t: VtreeIdx) -> Vec<VarId> {
        let Some(home) = self.home().map(Home::vtree).filter(|_| self.free_leaves() > 0) else { return Vec::new() };
        home.subtree(self.home_node(t))
            .filter(|&h| home.node(h).is_leaf())
            .map(|h| home.leaf_var(h))
            .filter(|&var| self.leaf_of(var).is_none())
            .collect()
    }

    /// The home this vtree is a projection of; `None` on a root vtree, a
    /// home's root included ([`Tdd::home`](crate::Tdd::home) gives that one).
    pub fn home(&self) -> Option<&Home> {
        self.home.as_deref().map(|link| &link.home)
    }

    /// The node of [`Vtree::home`] that own node `t` stands for; `t` itself on
    /// a root vtree.
    ///
    /// # Panics
    ///
    /// Panics if `t` is not below [`num_nodes`](Self::num_nodes).
    pub fn home_node(&self, t: VtreeIdx) -> VtreeIdx {
        match self.home.as_deref() {
            None => {
                assert!(t.idx() < self.nodes.len(), "home_node: {} is not a node", t.0);
                t
            }
            Some(link) => link.node_of[t.idx()],
        }
    }

    /// The own node standing for home node `h`, if this vtree has one; on a
    /// root vtree, `h` itself when it is a node. Logarithmic in
    /// [`num_leaves`](Self::num_leaves).
    pub fn node_at_home(&self, h: VtreeIdx) -> Option<VtreeIdx> {
        let Some(link) = self.home.as_deref() else {
            return (h.idx() < self.nodes.len()).then_some(h);
        };
        let slot = *link.home.index().slot.get(h.idx())?;
        if link.home.vtree().node(h).is_leaf() {
            let i = link.ranks.binary_search(&slot).ok()?;
            return Some(link.in_order[2 * i]);
        }
        // Home leaves `slot` and `slot + 1` are the two sides of `h`'s gap.
        let right = link.ranks.partition_point(|&r| r <= slot);
        if right == 0 || right == link.ranks.len() {
            return None;
        }
        let own = link.in_order[2 * right - 1];
        (link.node_of[own.idx()] == h).then_some(own)
    }

    /// Whether `home` is this vtree's home, or this vtree when it is a root.
    pub(crate) fn is_on(&self, home: &Home) -> bool {
        match self.home() {
            Some(own) => Arc::ptr_eq(own.vtree(), home.vtree()),
            None => std::ptr::eq(self, &**home.vtree()),
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
        let root_of = |v: &Arc<Vtree>| Arc::as_ptr(v.home().map_or(v, Home::vtree));
        if root_of(a) != root_of(b) {
            return Err(OperationError::VtreeMismatch);
        }
        let first_region = |link: &HomeLink| {
            debug_assert!(
                link.ranks.len() < link.home.index().leaf_at.len() || !link.regions.is_empty(),
                "a projection on every leaf is the home"
            );
            link.regions.first().map(|region| link.region_node(region))
        };
        let (la, lb) = match (a.home.as_deref(), b.home.as_deref()) {
            (None, None) => return Ok(LeafSets::Equal(SameLayout::new(a, b))),
            (Some(la), None) => return first_region(la).map_or(Ok(LeafSets::AInB), |t| Err(OperationError::MarginalLevel(t))),
            (None, Some(lb)) => return first_region(lb).map_or(Ok(LeafSets::BInA), |t| Err(OperationError::MarginalLevel(t))),
            (Some(la), Some(lb)) => (la, lb),
        };
        if !la.regions.is_empty() || !lb.regions.is_empty() {
            return Self::region_leaf_sets(la, lb);
        }
        let (ra, rb) = (&la.ranks, &lb.ranks);
        let mut union = Vec::with_capacity(ra.len() + rb.len());
        let (mut i, mut j, mut a_only, mut b_only) = (0, 0, false, false);
        while i < ra.len() || j < rb.len() {
            let (x, y) = (ra.get(i).copied().unwrap_or(u32::MAX), rb.get(j).copied().unwrap_or(u32::MAX));
            union.push(x.min(y));
            a_only |= x < y;
            b_only |= y < x;
            i += usize::from(x <= y);
            j += usize::from(y <= x);
        }
        Ok(match (a_only, b_only) {
            (false, false) => LeafSets::Equal(SameLayout::new(a, b)),
            (false, true) => LeafSets::AInB,
            (true, false) => LeafSets::BInA,
            (true, true) => LeafSets::Union(Self::seated(&la.home, union.len(), false, || Self::union_of(la, lb, union))),
        })
    }

    /// The projection on `union`, the merged ranks of `la` and `lb`. Two
    /// neighbours in the union that are neighbours in an operand meet where
    /// they do there; only neighbours from different operands are looked up.
    fn union_of(la: &HomeLink, lb: &HomeLink, union: Vec<u32>) -> Vtree {
        let index = la.home.index();
        let (ra, rb) = (&la.ranks, &lb.ranks);
        let mut gaps = Vec::with_capacity(union.len() - 1);
        let (mut i, mut j) = (0, 0);
        let (mut last_a, mut last_b) = (false, false);
        for (u, &r) in union.iter().enumerate() {
            let (in_a, in_b) = (ra.get(i) == Some(&r), rb.get(j) == Some(&r));
            if u > 0 {
                gaps.push(match (last_a && in_a, last_b && in_b) {
                    (true, _) => la.gap_before(i),
                    (_, true) => lb.gap_before(j),
                    _ => index.ancestor_gap(union[u - 1], r),
                });
            }
            i += usize::from(in_a);
            j += usize::from(in_b);
            (last_a, last_b) = (in_a, in_b);
        }
        let own = Self::link_projection(&la.home, union, gaps, Vec::new());
        debug_assert_eq!(own.validate(), Ok(()));
        own
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
    pub fn sub_projection(self: &Arc<Self>, home: &Home, mut keep: impl FnMut(VtreeIdx) -> bool) -> Result<Arc<Vtree>, VtreeError> {
        if !self.is_on(home) {
            return Err(VtreeError::Invalid("a sub-projection is taken on the vtree's own home".to_string()));
        }
        let index = home.index();
        let (k, link) = match self.home.as_deref() {
            Some(link) => (link.ranks.len(), Some(link)),
            None => (index.leaf_at.len(), None),
        };
        let (mut ranks, mut gaps, mut regions) = (Vec::new(), Vec::new(), Vec::new());
        let mut least = u64::MAX;
        for i in 0..k {
            let (rank, leaf, gap) = match link {
                Some(link) => (link.ranks[i], link.in_order[2 * i], (i > 0).then(|| link.gap_before(i))),
                None => (i as u32, index.leaf_at[i], (i > 0).then(|| i as u32 - 1)),
            };
            if let Some(g) = gap {
                least = least.min(index.gap_word[g as usize]);
            }
            let region = link.and_then(|link| link.regions.get(regions.len()).filter(|region| region.at as usize == i));
            if let Some(region) = region {
                regions.push(Region { at: ranks.len() as u32, ..region.clone() });
            } else if !keep(leaf) {
                continue;
            }
            if !ranks.is_empty() {
                gaps.push(least as u32);
            }
            ranks.push(rank);
            least = u64::MAX;
        }
        if ranks.len() == k {
            return Ok(Arc::clone(self));
        }
        if ranks.is_empty() {
            return Err(VtreeError::Invalid("a projection needs at least one variable".to_string()));
        }
        Ok(Self::seated(home, ranks.len(), !regions.is_empty(), || {
            let own = Self::link_projection(home, ranks, gaps, regions);
            debug_assert_eq!(own.validate(), Ok(()));
            own
        }))
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

    /// [`Vtree::leaf_sets`] of two projections of one home, one with a
    /// region. With no overlap, the operands share only variable leaves.
    #[cold]
    fn region_leaf_sets(la: &HomeLink, lb: &HomeLink) -> Result<LeafSets, OperationError> {
        let mut intervals = Vec::new();
        for (link, other) in [(la, lb), (lb, la)] {
            for region in &link.regions {
                let node = link.region_node(region);
                for &(lo, hi) in &region.extent {
                    let j = other.ranks.partition_point(|&r| r < lo);
                    if other.ranks.get(j).is_some_and(|&r| r < hi) {
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
        let mut union = [la.ranks.as_slice(), lb.ranks.as_slice()].concat();
        union.sort_unstable();
        union.dedup();
        match (union.len() > lb.ranks.len(), union.len() > la.ranks.len()) {
            (false, false) => unreachable!("an operand with a region has a position the other lacks"),
            (false, true) => return Ok(LeafSets::AInB),
            (true, false) => return Ok(LeafSets::BInA),
            (true, true) => {}
        }
        let mut regions: Vec<Region> = [la, lb]
            .iter()
            .flat_map(|link| {
                link.regions.iter().map(|region| {
                    let at = union.binary_search(&link.ranks[region.at as usize]).expect("a position of an operand") as u32;
                    Region { at, ..region.clone() }
                })
            })
            .collect();
        regions.sort_unstable_by_key(|region| region.at);
        Ok(LeafSets::Union(Arc::new(Self::project_ranks(&la.home, union, regions))))
    }

    /// Whether `var` has a leaf in the home but neither a leaf here nor a
    /// place in an extent: a variable a diagram on this vtree does not depend
    /// on.
    pub(crate) fn home_has(&self, var: VarId) -> bool {
        let Some(link) = self.home.as_deref() else { return false };
        let Some(rank) = link.home.rank(var) else { return false };
        self.leaf_of(var).is_none() && link.region_over(rank, rank + 1).is_none()
    }

    /// For `u`, the union of the leaf sets of projections `a` and `b`, whose
    /// root has, as its two children, the projections of `a`'s and `b`'s
    /// leaf sets: by own node of `a` and of `b`, the node of `u` standing for
    /// it, and whether `a` is on the left.
    ///
    /// A projection `u` is in the builder's numbering, so the nodes of `a`
    /// and `b` are at offsets in its in-order list. That holds only when the
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
            return Some((la.node_of.clone(), lb.node_of.clone(), a_left));
        };
        let a_left = match (lu.node_of[l.idx()], lu.node_of[r.idx()]) {
            (hl, hr) if hl == ha && hr == hb => true,
            (hl, hr) if hl == hb && hr == ha => false,
            _ => return None,
        };
        let map = |link: &HomeLink, offset: usize| -> Vec<VtreeIdx> {
            let mut map = vec![VtreeIdx(0); link.in_order.len()];
            for (j, &t) in link.in_order.iter().enumerate() {
                map[t.idx()] = lu.in_order[offset + j];
            }
            map
        };
        let (a_offset, b_offset) = if a_left { (0, la.in_order.len() + 1) } else { (lb.in_order.len() + 1, 0) };
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
        self.index.0.is_some()
    }

    /// The home `vtree` is on: a projection's home, or `vtree` itself when
    /// it is a home's root; `None` on a root that is not a home.
    pub(crate) fn home_of(vtree: &Arc<Vtree>) -> Option<Home> {
        match vtree.home() {
            Some(home) => Some(home.clone()),
            None => vtree.is_home().then(|| Home(Arc::clone(vtree))),
        }
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
        own.index = RootIndex(Some(Box::new(HomeIndex::new(own))));
    }

    /// The index of a home's root, which stops being a home.
    pub(crate) fn take_index(&mut self) -> RootIndex {
        std::mem::take(&mut self.index)
    }

    /// Put back the index [`take_index`](Self::take_index) took, the tree
    /// having its shape again.
    pub(crate) fn put_index(&mut self, index: RootIndex) {
        self.index = index;
    }

    /// Drop the index of a home's root about to change shape.
    #[inline]
    pub(crate) fn stop_being_home(&mut self) {
        if self.index.0.is_some() {
            self.index = RootIndex::default();
        }
    }

    /// The projection of `home` on the leaves of `ranks`, ascending and not
    /// empty, the positions of `regions` being regions.
    fn project_ranks(home: &Home, ranks: Vec<u32>, regions: Vec<Region>) -> Vtree {
        let own = Self::build_projection(home, ranks, regions);
        debug_assert_eq!(own.validate(), Ok(()));
        own
    }

    /// [`Vtree::project_ranks`] unchecked.
    pub(super) fn build_projection(home: &Home, ranks: Vec<u32>, regions: Vec<Region>) -> Vtree {
        let index = home.index();
        let gaps = ranks.windows(2).map(|w| index.ancestor_gap(w[0], w[1])).collect();
        Self::link_projection(home, ranks, gaps, regions)
    }

    /// The projection of `home` on `ranks`, unchecked, where `gaps[i]` is the
    /// home gap at which `ranks[i]` and `ranks[i + 1]` meet.
    fn link_projection(home: &Home, ranks: Vec<u32>, gaps: Vec<u32>, regions: Vec<Region>) -> Vtree {
        let (root, index) = (home.vtree(), home.index());
        let k = ranks.len();
        let at: Vec<u32> = regions.iter().map(|region| region.at).collect();
        let (nodes, own_root, in_order) = in_order_tree(
            k,
            |i| index.gap_depth(gaps[i]),
            &at,
            |i| root.leaf_var(index.leaf_at[ranks[i] as usize]),
        );
        let mut node_of = vec![VtreeIdx(0); in_order.len()];
        for (i, &rank) in ranks.iter().enumerate() {
            node_of[in_order[2 * i].idx()] = index.leaf_at[rank as usize];
        }
        for (i, &g) in gaps.iter().enumerate() {
            node_of[in_order[2 * i + 1].idx()] = index.gap_node[g as usize];
        }
        let leaf_count = (k - regions.len()) as u32;
        let topo = TopoOrder::by_level(&nodes, own_root);
        Vtree {
            context: Arc::clone(&root.context),
            nodes,
            root: own_root,
            var_to_leaf: Vec::new(),
            leaf_count,
            topo,
            home: Some(Arc::new(HomeLink { home: home.clone(), ranks, node_of, in_order, regions })),
            index: RootIndex::default(),
        }
    }

    /// Kept out of line so that [`Vtree::leaf_of`] on a root vtree stays as
    /// small as it was before projections.
    #[cold]
    #[inline(never)]
    pub(super) fn projected_leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        self.home.as_deref()?.leaf_of(var)
    }

    /// [`Vtree::num_vars`] of a vtree with an empty leaf table, which only a
    /// projection has.
    #[cold]
    #[inline(never)]
    pub(super) fn projected_num_vars(&self) -> u32 {
        self.home.as_deref().map_or(0, |link| link.home.vtree().num_vars())
    }

    /// Called by every kept change of shape that does not end on a lifted
    /// home, as a rotation search's does: a projection becomes a root with a
    /// dense leaf table. Inline, because a restructure commits rotations by
    /// the hundred thousand on vtrees that are not projections.
    #[inline]
    pub(crate) fn reshaped(&mut self) {
        if self.home.is_some() {
            self.leave_home();
        }
    }

    /// Undo [`reshaped`](Self::reshaped) after moves that kept node indices:
    /// take back the link of `before`, this tree before the moves, so that
    /// [`lifted_home`](Self::lifted_home) can read where each node stood.
    pub(crate) fn rejoin_home(&mut self, before: &Vtree) {
        self.home = before.home.clone();
        self.var_to_leaf = Vec::new();
    }

    #[cold]
    #[inline(never)]
    fn leave_home(&mut self) {
        let Some(link) = self.home.take() else { return };
        assert!(link.regions.is_empty(), "a vtree with a region changed shape");
        self.var_to_leaf = vec![VtreeIdx(0); link.home.vtree().num_vars() as usize];
        for (i, node) in self.nodes.iter().enumerate() {
            if let VtreeNode::Leaf { var, .. } = node {
                self.var_to_leaf[var.idx()] = VtreeIdx(i as u32);
            }
        }
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
            let is_region = link.in_order.get(2 * at).is_some_and(|t| self.nodes.get(t.idx()).is_some_and(VtreeNode::is_region));
            if at >= link.ranks.len() || !is_region {
                return fail(format!("position {at} is not a region"));
            }
            let mut count = 0;
            for (i, &(lo, hi)) in region.extent.iter().enumerate() {
                if lo >= hi || hi > n || (i > 0 && region.extent[i - 1].1 >= lo) {
                    return fail(format!("the extent of position {at} is not ascending disjoint rank intervals"));
                }
                count += hi - lo;
                let inside = link.ranks.partition_point(|&r| r < hi) - link.ranks.partition_point(|&r| r < lo);
                let anchor_here = (lo..hi).contains(&link.ranks[at]);
                if inside != usize::from(anchor_here) {
                    return fail(format!("the extent of position {at} holds another position's rank"));
                }
                intervals.push((lo, hi));
            }
            if count != region.count || !region.covers(link.ranks[at], link.ranks[at] + 1) {
                return fail(format!("the extent of position {at} does not hold its anchor or its count"));
            }
        }
        intervals.sort_unstable();
        if intervals.windows(2).any(|w| w[1].0 < w[0].1) {
            return fail("two extents meet".to_string());
        }
        Ok(())
    }

    /// A projection's link agrees with its home: the in-order list is this
    /// tree's in-order walk, each leaf stands for the home leaf of its rank
    /// with the same variable, each region for the home leaf of its anchor,
    /// each internal node for the lowest common ancestor in the home of the
    /// childless nodes on either side of it, and a parent for a shallower
    /// home node than its child; the node indices are the ones the builder
    /// gives the rank list. Each extent holds its anchor and meets neither
    /// another extent nor a leaf's rank.
    pub(super) fn validate_home(&self) -> Result<(), VtreeError> {
        let Some(link) = self.home.as_deref() else {
            return Ok(());
        };
        let fail = |msg: String| Err(VtreeError::Invalid(format!("projection: {msg}")));
        let (home, index) = (&**link.home.vtree(), link.home.index());
        let k = self.num_leaves() as usize + link.regions.len();
        if link.ranks.len() != k || link.node_of.len() != self.nodes.len() {
            return fail("the link does not fit the tree".to_string());
        }
        if link.ranks.windows(2).any(|w| w[0] >= w[1]) || link.ranks.last().is_some_and(|&r| r >= home.num_leaves()) {
            return fail("the ranks are not ascending home ranks".to_string());
        }
        self.validate_regions(link)?;
        let mut walk = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![(self.root, false)];
        while let Some((t, left_done)) = stack.pop() {
            match *self.node(t) {
                VtreeNode::Internal { left, right, .. } if !left_done => {
                    stack.push((right, false));
                    stack.push((t, true));
                    stack.push((left, false));
                }
                _ => walk.push(t),
            }
        }
        if walk != link.in_order {
            return fail("the in-order list is not the in-order walk".to_string());
        }
        let at: Vec<u32> = link.regions.iter().map(|region| region.at).collect();
        let gaps: Vec<u32> = link.ranks.windows(2).map(|w| index.ancestor_gap(w[0], w[1])).collect();
        let (_, root, in_order) =
            in_order_tree(k, |i| index.gap_depth(gaps[i]), &at, |i| home.leaf_var(index.leaf_at[link.ranks[i] as usize]));
        if root != self.root || in_order != link.in_order {
            return fail("the nodes are not in the order the builder gives their rank list".to_string());
        }
        for (i, &t) in link.in_order.iter().enumerate() {
            let expected = if i % 2 == 0 {
                index.leaf_at[link.ranks[i / 2] as usize]
            } else {
                index.gap_node[index.ancestor_gap(link.ranks[i / 2], link.ranks[i / 2 + 1]) as usize]
            };
            if link.node_of[t.idx()] != expected {
                return fail(format!("node {} stands for {} instead of {}", t.0, link.node_of[t.idx()].0, expected.0));
            }
            if i % 2 == 0 && !self.node(t).is_region() && home.leaf_var(expected) != self.leaf_var(t) {
                return fail(format!("leaf {} carries another variable than its home leaf", t.0));
            }
            if let Some(p) = self.node(t).parent()
                && index.depth_of(home, link.node_of[p.idx()]) >= index.depth_of(home, link.node_of[t.idx()])
            {
                return fail(format!("node {} is not below its parent in the home", t.0));
            }
        }
        Ok(())
    }
}
