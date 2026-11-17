//! Projections of a root vtree on a subset of its leaves, keeping variable ids.
//!
//! The root is the projection's home. A projection's leaves are the home's
//! leaves it keeps, and its internal nodes are the home nodes with a kept leaf
//! under each child, so its shape depends only on the home and the leaf set.
//! Everything here costs time in the number of kept leaves and a logarithm of
//! the home's size; the one table sized by the home is the home's
//! [`HomeIndex`], built once and shared by all its projections.

use std::ptr;
use std::sync::{Arc, Mutex, Weak};

use super::topo::TopoOrder;
use super::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};

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
        let gaps = n - 1;
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

/// What a projection knows of its home.
#[derive(Clone, Debug)]
pub(crate) struct HomeLink {
    pub(super) home: Arc<Vtree>,
    pub(super) index: Arc<HomeIndex>,
    /// The home ranks of the own leaves, ascending.
    pub(super) ranks: Vec<u32>,
    /// By own node: the home node.
    pub(super) node_of: Vec<VtreeIdx>,
    /// The own nodes in order: leaf `i` at `2i`, the node between leaves `i`
    /// and `i + 1` at `2i + 1`.
    pub(super) in_order: Vec<VtreeIdx>,
}

impl HomeLink {
    pub(super) fn leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        let h = self.home.leaf_of(var)?;
        let i = self.ranks.binary_search(&self.index.slot[h.idx()]).ok()?;
        Some(self.in_order[2 * i])
    }
}

/// The index of each home, by the home's allocation. A rotation reaches a
/// vtree in an `Arc` only through `Arc::make_mut`, which moves a vtree with
/// weak references to a new allocation, so an entry never describes a changed
/// shape. A dropped home's entry is freed at the next miss. Not a field of
/// [`Context`](crate::Context): one that needs dropping cost a consumer that
/// makes a context per vtree 0.03% of its instructions.
static HOMES: Mutex<Vec<(Weak<Vtree>, Arc<HomeIndex>)>> = Mutex::new(Vec::new());

fn cached_index(entries: &[(Weak<Vtree>, Arc<HomeIndex>)], home: &Arc<Vtree>) -> Option<Arc<HomeIndex>> {
    entries.iter().find(|(h, _)| ptr::eq(h.as_ptr(), Arc::as_ptr(home))).map(|(_, index)| Arc::clone(index))
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
    /// A projection of a projection is a projection of its home. Repeated
    /// variables count once. Costs time in the number of variables times the
    /// logarithm of the home's size; the first projection of a home also builds
    /// an index over the home, in time and memory linear in the home.
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
    /// use tididi::vtree::{VarId, Vtree};
    ///
    /// let home = Arc::new(Vtree::balanced(8));
    /// let own = Vtree::projection(&home, [VarId(7), VarId(2)])?;
    /// assert_eq!(own.num_leaves(), 2);
    /// let (left, right) = own.children(own.root());
    /// assert_eq!((own.leaf_var(left), own.leaf_var(right)), (VarId(2), VarId(7)));
    /// assert_eq!(own.home_node(own.root()), home.root());
    /// assert!(Arc::ptr_eq(own.home().unwrap(), &home));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn projection(home: &Arc<Vtree>, vars: impl IntoIterator<Item = VarId>) -> Result<Arc<Vtree>, VtreeError> {
        let root = home.home().unwrap_or(home);
        let index = root.home_index();
        let mut ranks = Vec::new();
        for var in vars {
            let leaf = home
                .leaf_of(var)
                .ok_or_else(|| VtreeError::Invalid(format!("variable {} has no leaf to project on", var.0)))?;
            ranks.push(index.slot[home.home_node(leaf).idx()]);
        }
        if ranks.is_empty() {
            return Err(VtreeError::Invalid("a projection needs at least one variable".to_string()));
        }
        ranks.sort_unstable();
        ranks.dedup();
        Ok(Arc::new(Self::project_ranks(root, index, ranks)))
    }

    /// The projection of `home` on no leaf, the vtree of a constant: no node,
    /// so [`root`](Self::root) names none, and the home's variables.
    ///
    /// Not cached in the home: the home holds no reference to its
    /// projections, and this one would hold the home.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let home = Arc::new(Vtree::balanced(5));
    /// let empty = Vtree::empty_projection(&home);
    /// assert_eq!(empty.num_nodes(), 0);
    /// assert!(Tdd::one(&empty).is_one());
    /// assert_eq!(Tdd::one(&empty).model_count().unwrap(), 32u32.into());
    /// ```
    pub fn empty_projection(home: &Arc<Vtree>) -> Arc<Vtree> {
        let root = home.home().unwrap_or(home);
        let link = HomeLink {
            home: Arc::clone(root),
            index: root.home_index(),
            ranks: Vec::new(),
            node_of: Vec::new(),
            in_order: Vec::new(),
        };
        Arc::new(Vtree {
            context: Arc::clone(&root.context),
            nodes: Vec::new(),
            root: VtreeIdx(0),
            var_to_leaf: Vec::new(),
            leaf_count: 0,
            topo: TopoOrder::default(),
            home: Some(Arc::new(link)),
        })
    }

    /// The topmost own node at or under home node `h`, `None` when no own leaf
    /// is under `h`; on a root vtree, `h` itself when it is a node.
    /// Logarithmic in [`num_leaves`](Self::num_leaves) times the logarithm
    /// of the home's size.
    pub fn own_at(&self, h: VtreeIdx) -> Option<VtreeIdx> {
        let Some(link) = self.home.as_deref() else {
            return (h.idx() < self.nodes.len()).then_some(h);
        };
        let (index, ranks) = (&*link.index, &link.ranks);
        let gap = *index.slot.get(h.idx())?;
        if link.home.node(h).is_leaf() {
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

    /// The number of home leaves this vtree does not have; 0 on a root.
    pub(crate) fn free_leaves(&self) -> u32 {
        self.home.as_deref().map_or(0, |link| link.home.num_leaves() - self.leaf_count)
    }

    /// The variables of the home leaves this vtree does not have, left to
    /// right; none on a root.
    pub(crate) fn free_vars(&self) -> Vec<VarId> {
        let Some(link) = self.home.as_deref() else { return Vec::new() };
        let mut own = link.ranks.iter().copied().peekable();
        let mut free = Vec::with_capacity(self.free_leaves() as usize);
        for (rank, &leaf) in link.index.leaf_at.iter().enumerate() {
            if own.next_if_eq(&(rank as u32)).is_none() {
                free.push(link.home.leaf_var(leaf));
            }
        }
        free
    }

    /// The variables of the home leaves under own node `t`'s home node that
    /// this vtree does not have; none on a root.
    pub(crate) fn free_vars_under(&self, t: VtreeIdx) -> Vec<VarId> {
        let Some(home) = self.home().filter(|_| self.free_leaves() > 0) else { return Vec::new() };
        home.subtree(self.home_node(t))
            .filter(|&h| home.node(h).is_leaf())
            .map(|h| home.leaf_var(h))
            .filter(|&var| self.leaf_of(var).is_none())
            .collect()
    }

    /// The home this vtree is a projection of; `None` on a root vtree.
    pub fn home(&self) -> Option<&Arc<Vtree>> {
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
        let slot = *link.index.slot.get(h.idx())?;
        if link.home.node(h).is_leaf() {
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

    /// How the leaf sets of `a` and `b` compare; `None` when their homes
    /// differ (a root vtree is its own home).
    pub(crate) fn leaf_sets(a: &Arc<Vtree>, b: &Arc<Vtree>) -> Option<LeafSets> {
        let root_of = |v: &Arc<Vtree>| Arc::as_ptr(v.home().unwrap_or(v));
        if root_of(a) != root_of(b) {
            return None;
        }
        let (la, lb) = match (a.home.as_deref(), b.home.as_deref()) {
            (None, None) => return Some(LeafSets::Equal),
            (Some(_), None) => return Some(LeafSets::AInB),
            (None, Some(_)) => return Some(LeafSets::BInA),
            (Some(la), Some(lb)) => (la, lb),
        };
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
        Some(match (a_only, b_only) {
            (false, false) => LeafSets::Equal,
            (false, true) => LeafSets::AInB,
            (true, false) => LeafSets::BInA,
            (true, true) => LeafSets::Union(Arc::new(Self::project_ranks(&la.home, Arc::clone(&la.index), union))),
        })
    }

    /// Whether `var` has no leaf here but one in the home: a variable a
    /// diagram on this vtree does not depend on.
    pub(crate) fn home_has(&self, var: VarId) -> bool {
        self.home().is_some_and(|home| home.leaf_of(var).is_some()) && self.leaf_of(var).is_none()
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

    pub(super) fn home_index(self: &Arc<Self>) -> Arc<HomeIndex> {
        let homes = || HOMES.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(index) = cached_index(&homes(), self) {
            return index;
        }
        let index = Arc::new(HomeIndex::new(self));
        let mut entries = homes();
        if let Some(theirs) = cached_index(&entries, self) {
            return theirs;
        }
        entries.retain(|(h, _)| h.strong_count() > 0);
        entries.push((Arc::downgrade(self), Arc::clone(&index)));
        index
    }

    /// The projection of root `home` on the leaves of `ranks`, ascending and
    /// not empty. The tree is built on local ids `1..=k`, so that nothing is
    /// sized by the home's id space, and then takes the home's ids.
    fn project_ranks(home: &Arc<Vtree>, index: Arc<HomeIndex>, ranks: Vec<u32>) -> Vtree {
        let k = ranks.len();
        let (gaps, depths): (Vec<u32>, Vec<u32>) = ranks
            .windows(2)
            .map(|w| {
                let g = index.ancestor_gap(w[0], w[1]);
                (g, index.gap_depth[g as usize])
            })
            .unzip();
        let local: Vec<VarId> = (1..=k as u32).map(VarId).collect();
        let (mut own, in_order) =
            Self::from_in_order_with_map(&local, &depths, k as u32).expect("k distinct local ids and k - 1 depths");
        let mut node_of = vec![VtreeIdx(0); in_order.len()];
        for (i, &rank) in ranks.iter().enumerate() {
            let (t, h) = (in_order[2 * i], index.leaf_at[rank as usize]);
            node_of[t.idx()] = h;
            if let VtreeNode::Leaf { var, .. } = &mut own.nodes[t.idx()] {
                *var = home.leaf_var(h);
            }
        }
        for (i, &g) in gaps.iter().enumerate() {
            node_of[in_order[2 * i + 1].idx()] = index.gap_node[g as usize];
        }
        own.var_to_leaf = Vec::new();
        own.context = Arc::clone(&home.context);
        own.home = Some(Arc::new(HomeLink { home: Arc::clone(home), index, ranks, node_of, in_order }));
        debug_assert_eq!(own.validate(), Ok(()));
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
        self.home.as_deref().map_or(0, |link| link.home.num_vars())
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
        self.var_to_leaf = vec![VtreeIdx(0); link.home.num_vars() as usize];
        for (i, node) in self.nodes.iter().enumerate() {
            if let VtreeNode::Leaf { var, .. } = node {
                self.var_to_leaf[var.idx()] = VtreeIdx(i as u32);
            }
        }
    }

    /// A projection's link agrees with its home: the in-order list is this
    /// tree's in-order walk, each leaf stands for the home leaf of its rank
    /// with the same variable, each internal node for the lowest common
    /// ancestor in the home of the leaves on either side of it, and a parent
    /// for a shallower home node than its child.
    pub(super) fn validate_home(&self) -> Result<(), VtreeError> {
        let Some(link) = self.home.as_deref() else {
            return Ok(());
        };
        let fail = |msg: String| Err(VtreeError::Invalid(format!("projection: {msg}")));
        let (home, index) = (&*link.home, &*link.index);
        let k = self.num_leaves() as usize;
        if home.home().is_some() || link.ranks.len() != k || link.node_of.len() != self.nodes.len() {
            return fail("the link does not fit the tree".to_string());
        }
        if link.ranks.windows(2).any(|w| w[0] >= w[1]) || link.ranks.last().is_some_and(|&r| r >= home.num_leaves()) {
            return fail("the ranks are not ascending home ranks".to_string());
        }
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
            if i % 2 == 0 && home.leaf_var(expected) != self.leaf_var(t) {
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
