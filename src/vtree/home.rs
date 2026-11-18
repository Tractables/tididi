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
    pub(super) gap_depth: Vec<u32>,
    /// A segment tree over the gaps, leaves at `gaps..2 * gaps`; each entry
    /// is the shallowest gap of its range.
    shallowest: Vec<u32>,
}

impl HomeIndex {
    fn new(home: &Vtree) -> Self {
        let n = home.num_leaves() as usize;
        let gaps = n.saturating_sub(1);
        let mut slot = vec![0; home.num_nodes()];
        let (mut leaf_at, mut gap_node, mut gap_depth) =
            (Vec::with_capacity(n), Vec::with_capacity(gaps), Vec::with_capacity(gaps));
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
        let mut shallowest = vec![0; 2 * gaps];
        for g in 0..gaps {
            shallowest[gaps + g] = g as u32;
        }
        for i in (1..gaps).rev() {
            shallowest[i] = Self::shallower(&gap_depth, shallowest[2 * i], shallowest[2 * i + 1]);
        }
        HomeIndex { slot, leaf_at, gap_node, gap_depth, shallowest }
    }

    fn shallower(depth: &[u32], a: u32, b: u32) -> u32 {
        if depth[b as usize] < depth[a as usize] { b } else { a }
    }

    /// The gap of the lowest common ancestor of the leaves of ranks `a < b`.
    pub(super) fn ancestor_gap(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a < b);
        let gaps = self.gap_node.len();
        let (mut lo, mut hi) = (a as usize + gaps, b as usize + gaps);
        let mut best = a;
        while lo < hi {
            if lo & 1 == 1 {
                best = Self::shallower(&self.gap_depth, best, self.shallowest[lo]);
                lo += 1;
            }
            if hi & 1 == 1 {
                hi -= 1;
                best = Self::shallower(&self.gap_depth, best, self.shallowest[hi]);
            }
            lo /= 2;
            hi /= 2;
        }
        best
    }

    fn depth_of(&self, home: &Vtree, h: VtreeIdx) -> u32 {
        match home.node(h) {
            VtreeNode::Leaf { .. } => u32::MAX,
            VtreeNode::Internal { .. } => self.gap_depth[self.slot[h.idx()] as usize],
        }
    }
}

/// A root vtree and an index over its leaves: the home of its projections
/// ([`Vtree::projection`]).
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
pub struct Home(Arc<HomeParts>);

#[derive(Debug)]
struct HomeParts {
    vtree: Arc<Vtree>,
    index: HomeIndex,
}

impl Home {
    /// The home with root `vtree`, its index built in time and memory linear
    /// in the vtree's size. On a projection, the projection's home.
    pub fn new(vtree: impl Into<Arc<Vtree>>) -> Self {
        let vtree = vtree.into();
        if let Some(home) = vtree.home() {
            return home.clone();
        }
        let index = HomeIndex::new(&vtree);
        Home(Arc::new(HomeParts { vtree, index }))
    }

    /// The root vtree.
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.0.vtree
    }

    pub(super) fn index(&self) -> &HomeIndex {
        &self.0.index
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
    pub(super) fn leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        let i = self.ranks.binary_search(&self.home.rank(var)?).ok()?;
        if !self.regions.is_empty() && self.regions.binary_search_by_key(&(i as u32), |r| r.at).is_ok() {
            return None;
        }
        Some(self.in_order[2 * i])
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

/// How the leaf sets of two vtrees with one home compare.
#[derive(Debug)]
pub(crate) enum LeafSets {
    /// The same leaf set and the same layout.
    Equal,
    /// The first vtree's leaves are a proper subset of the second's, or the
    /// second is the home.
    AInB,
    /// The second vtree's leaves are a proper subset of the first's, or the
    /// first is the home.
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
        Ok(Arc::new(Self::project_ranks(home, ranks, Vec::new())))
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
        })
    }

    /// The topmost own node at or under home node `h`, `None` when no own leaf
    /// or region anchor is under `h`; on a root vtree, `h` itself when it is
    /// a node. Logarithmic in [`num_leaves`](Self::num_leaves) times the
    /// logarithm of the home's size.
    pub fn own_at(&self, h: VtreeIdx) -> Option<VtreeIdx> {
        let Some(link) = self.home.as_deref() else {
            return (h.idx() < self.nodes.len()).then_some(h);
        };
        let (index, ranks) = (link.home.index(), &link.ranks);
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

    /// The home this vtree is a projection of; `None` on a root vtree.
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
        let first_region = |link: &HomeLink| link.regions.first().map(|region| link.region_node(region));
        let (la, lb) = match (a.home.as_deref(), b.home.as_deref()) {
            (None, None) => return Ok(LeafSets::Equal),
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
            (false, false) => LeafSets::Equal,
            (false, true) => LeafSets::AInB,
            (true, false) => LeafSets::BInA,
            (true, true) => LeafSets::Union(Arc::new(Self::project_ranks(&la.home, union, Vec::new()))),
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

    /// The region whose extent holds `var`.
    pub(crate) fn region_of(&self, var: VarId) -> Option<VtreeIdx> {
        let link = self.home.as_deref()?;
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

    /// For a projection `u` whose root has, as its two children, the
    /// projections of `a`'s and `b`'s leaf sets: by own node of `a` and of
    /// `b`, the node of `u` standing for it, and whether `a` is on the left.
    pub(crate) fn split_at_root(u: &Vtree, a: &Vtree, b: &Vtree) -> Option<(Vec<VtreeIdx>, Vec<VtreeIdx>, bool)> {
        let (lu, la, lb) = (u.home.as_ref()?, a.home.as_ref()?, b.home.as_ref()?);
        if u.node(u.root()).is_leaf() {
            return None;
        }
        let (l, r) = u.children(u.root());
        let (ha, hb) = (a.home_node(a.root()), b.home_node(b.root()));
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

    /// The anchor rank and extent of the region that folding own node `t`
    /// of this projection makes: the ranks of the variable leaves under `t`
    /// and the extents of the regions under it, the anchor at the leftmost
    /// childless node under `t`.
    pub(crate) fn folded_region(&self, t: VtreeIdx) -> (u32, Extent) {
        let link = self.home.as_deref().expect("a projection");
        let position = |u: VtreeIdx| link.ranks.binary_search(&link.home.index().slot[link.node_of[u.idx()].idx()]).expect("a position");
        let mut intervals = Vec::new();
        for u in self.subtree(t).filter(|&u| self.node(u).is_leaf()) {
            match link.regions.binary_search_by_key(&(position(u) as u32), |region| region.at) {
                Ok(j) => intervals.extend_from_slice(&link.regions[j].extent),
                Err(_) => intervals.push((link.ranks[position(u)], link.ranks[position(u)] + 1)),
            }
        }
        let mut leftmost = t;
        while let VtreeNode::Internal { left, .. } = *self.node(leftmost) {
            leftmost = left;
        }
        (link.ranks[position(leftmost)], coalesce(intervals))
    }

    /// The anchor variable and the variables of region `t` of this
    /// projection.
    pub(crate) fn region_vars(&self, t: VtreeIdx) -> (VarId, Vec<VarId>) {
        let link = self.home.as_deref().expect("a projection");
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
        self.home.as_deref().into_iter().flat_map(|link| link.regions.iter().map(|region| link.region_node(region)))
    }

    /// The anchor rank and extent of region `t` of this projection.
    pub(crate) fn region_at(&self, t: VtreeIdx) -> (u32, Extent) {
        let link = self.home.as_deref().expect("a projection");
        let region = link.regions.iter().find(|region| link.region_node(region) == t).expect("a region");
        (link.ranks[region.at as usize], region.extent.clone())
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
        Arc::new(Self::project_ranks(home, ranks, regions))
    }

    /// The projection of `home` on the leaves of `ranks`, ascending and not
    /// empty, the positions of `regions` being regions.
    fn project_ranks(home: &Home, ranks: Vec<u32>, regions: Vec<Region>) -> Vtree {
        let own = Self::build_projection(home, ranks, regions);
        debug_assert_eq!(own.validate(), Ok(()));
        own
    }

    /// [`Vtree::project_ranks`] unchecked. The tree is built on local ids
    /// `1..=k`, so that nothing is sized by the home's id space, and then
    /// takes the home's ids.
    pub(super) fn build_projection(home: &Home, ranks: Vec<u32>, regions: Vec<Region>) -> Vtree {
        let (root, index) = (home.vtree(), home.index());
        let k = ranks.len();
        let (gaps, depths): (Vec<u32>, Vec<u32>) = ranks
            .windows(2)
            .map(|w| {
                let g = index.ancestor_gap(w[0], w[1]);
                (g, index.gap_depth[g as usize])
            })
            .unzip();
        let local: Vec<VarId> = (1..=k as u32).map(VarId).collect();
        let at: Vec<u32> = regions.iter().map(|region| region.at).collect();
        let (mut own, in_order) =
            Self::from_in_order_with_map(&local, &depths, k as u32, &at).expect("k distinct local ids and k - 1 depths");
        let mut node_of = vec![VtreeIdx(0); in_order.len()];
        for (i, &rank) in ranks.iter().enumerate() {
            let (t, h) = (in_order[2 * i], index.leaf_at[rank as usize]);
            node_of[t.idx()] = h;
            if let VtreeNode::Leaf { var, .. } = &mut own.nodes[t.idx()] {
                *var = root.leaf_var(h);
            }
        }
        // The loop above gave a region its anchor's variable, so that a tree
        // without regions runs no test there.
        for region in &regions {
            if let VtreeNode::Leaf { var, .. } = &mut own.nodes[in_order[2 * region.at as usize].idx()] {
                *var = super::REGION_VAR;
            }
        }
        for (i, &g) in gaps.iter().enumerate() {
            node_of[in_order[2 * i + 1].idx()] = index.gap_node[g as usize];
        }
        own.var_to_leaf = Vec::new();
        own.context = Arc::clone(&root.context);
        own.home = Some(Arc::new(HomeLink { home: home.clone(), ranks, node_of, in_order, regions }));
        own
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

    /// Called by every kept change of shape (a reverted rotation probe
    /// restores the shape the link describes): a projection becomes a root
    /// with a dense leaf table. Inline, because a rotation search commits
    /// rotations by the hundred thousand on vtrees that are not projections.
    #[inline]
    pub(crate) fn reshaped(&mut self) {
        if self.home.is_some() {
            self.leave_home();
        }
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
    /// home node than its child. Each extent holds its anchor and meets
    /// neither another extent nor a leaf's rank.
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
