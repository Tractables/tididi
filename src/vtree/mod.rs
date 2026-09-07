//! Variable tree (vtree): the structural backbone of a TDD.
//!
//! A vtree is a rooted binary tree whose leaves correspond to Boolean variables.
//! It governs how a TDD decomposes its Boolean function: each internal vtree node
//! `t` with children `t_L, t_R` defines a partition of variables, and the TDD
//! nodes at level `t` represent sub-functions `f(vars(t_L), vars(t_R))` as
//! disjunctions of input pairs `(left_child, right_child)`.
//!
//! This module is deliberately a copy of vitri's `src/vtree/` module, kept in
//! sync by hand: same public names, same accessor style, same semantics, so a
//! vtree built by vitri's construction heuristics carries over as `.vtree` text
//! (or as a node relabel) without translation. The heuristics that decide which
//! vtree to build live in vitri; this module owns the *structure* — topology,
//! traversal order, LCA, rotation, text I/O — and the programmatic constructors
//! ([`Vtree::leaf`], [`Vtree::join`], [`Vtree::balanced_over`],
//! [`Vtree::linear_from_order`], [`Vtree::graft`], [`Vtree::project_to_vars`]).
//!
//! ## Variable ids
//!
//! A vtree may cover a sparse subset of variable ids. [`Vtree::num_vars`] is
//! the id space (`max VarId + 1`); [`Vtree::num_leaves`] is the number of ids
//! the tree actually carries; [`Vtree::leaf_of`] is defined only for covered
//! ids. The two counts agree exactly when the leaves are `0..num_vars`.
//!
//! ## Node layout
//!
//! Nodes are stored in bottom-up level order: all leaves first (`0..num_leaves`),
//! then internal nodes (`num_leaves..n`). This guarantees `child.idx() < parent.idx()`
//! for every edge at construction. A rotation relinks nodes without moving them,
//! so on a rotated tree the array order is no longer topological and
//! [`Vtree::bottomup`] is: every traversal reads that order rather than `0..n`.

use std::fmt;

/// A 0-indexed variable identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VarId(pub u32);

/// Index into the vtree node array.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct VtreeIdx(pub u32);

impl VtreeIdx {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl VarId {
    /// The variable number as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

/// A literal: a variable with a polarity.
///
/// Lives lib-side (alongside `VarId`) so the pure-TDD layer (`clause_to_tdd`,
/// `apply_and_clause`) can accept `&[Literal]` slices without depending on the
/// CNF module. The CNF `Clause`/`CnfFormula` types build on it and re-export it.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Literal {
    /// The variable this literal refers to.
    pub var: VarId,
    /// `true` for a positive literal, `false` for a negated one.
    pub positive: bool,
}

impl Literal {
    /// Construct a literal over `var` with the given polarity.
    pub fn new(var: VarId, positive: bool) -> Self {
        Literal { var, positive }
    }

    /// The positive literal over `var`.
    pub fn pos(var: VarId) -> Self {
        Literal::new(var, true)
    }

    /// The negated literal over `var`.
    pub fn neg(var: VarId) -> Self {
        Literal::new(var, false)
    }

    /// This literal with its polarity flipped.
    #[must_use]
    pub fn negated(self) -> Self {
        Literal {
            var: self.var,
            positive: !self.positive,
        }
    }
}

/// Build a `Literal` from a signed **DIMACS** integer.
///
/// DIMACS variables are 1-based: `1` is the first variable (`VarId(0)`), `2` the
/// second, and so on; a negative value denotes a negated literal. The magnitude
/// is decremented to the 0-based [`VarId`] used internally — the same convention
/// as the CNF parser (`VarId(val.unsigned_abs() - 1)`).
///
/// # Panics
/// Panics on `0`, which is not a valid DIMACS literal (in the DIMACS format `0`
/// terminates a clause rather than naming a variable).
///
/// ```
/// use tididi::vtree::{Literal, VarId};
/// assert_eq!(Literal::from(1), Literal::pos(VarId(0)));
/// assert_eq!(Literal::from(-2), Literal::neg(VarId(1)));
/// ```
impl From<i32> for Literal {
    fn from(n: i32) -> Self {
        assert!(
            n != 0,
            "0 is not a DIMACS literal (it terminates a clause, not a variable)"
        );
        let var = VarId(n.unsigned_abs() - 1);
        if n > 0 {
            Literal::pos(var)
        } else {
            Literal::neg(var)
        }
    }
}

/// Why a vtree could not be built, parsed, or checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VtreeError {
    /// `.vtree` text that does not describe a single tree.
    Text(String),
    /// Two of the trees being combined both carry this variable.
    OverlappingVariable(VarId),
    /// A structural invariant that does not hold (see [`Vtree::validate`]),
    /// or a construction handed nothing to build from.
    Invalid(String),
}

impl fmt::Display for VtreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VtreeError::Text(msg) => write!(f, "malformed vtree text: {msg}"),
            VtreeError::OverlappingVariable(var) => write!(
                f,
                "variable {} is carried by more than one of the trees being combined",
                var.0 + 1
            ),
            VtreeError::Invalid(msg) => write!(f, "invalid vtree: {msg}"),
        }
    }
}

impl std::error::Error for VtreeError {}

/// Which rotation direction a `fixup_topo_after_rotate` call corresponds to —
/// selects which grandchild subtree may violate children-before-parents after
/// the rotation.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum RotationKind {
    /// A left rotation.
    Left,
    /// A right rotation.
    Right,
}

/// A node in the vtree (variable tree).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VtreeNode {
    /// A leaf holding a single variable.
    Leaf {
        /// The variable at this leaf.
        var: VarId,
        /// Parent node index, or `None` at the root.
        parent: Option<VtreeIdx>,
    },
    /// An internal node with two children.
    Internal {
        /// Left child index.
        left: VtreeIdx,
        /// Right child index.
        right: VtreeIdx,
        /// Parent node index, or `None` at the root.
        parent: Option<VtreeIdx>,
    },
}

impl VtreeNode {
    /// This node's parent index, or `None` if it is the root.
    pub fn parent(&self) -> Option<VtreeIdx> {
        match self {
            VtreeNode::Leaf { parent, .. } => *parent,
            VtreeNode::Internal { parent, .. } => *parent,
        }
    }

    /// Whether this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, VtreeNode::Leaf { .. })
    }
}

/// Where each piece of a graft landed in the finished vtree. Returned beside
/// the tree by the crate-internal graft so the TDD-side graft can relocate
/// per-part levels and wire the spine's pair links.
///
/// Piece order matches the graft's arguments: `[subtree 0, …, subtree k-1,
/// spine_var 0 leaf, …, spine_var m-1 leaf]`. `chain_internals[j]` is the
/// right-linear join that incorporates piece j+1 (left child: the running
/// chain root; right child: piece j+1). For a single piece it is empty.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct GraftLayout {
    /// `comp_to_full[k][c]` = final `VtreeIdx` of subtree k's own node `c`
    /// (indexed as in that subtree's node array, `0..num_nodes()`).
    pub comp_to_full: Vec<Vec<VtreeIdx>>,
    /// Final `VtreeIdx` of each spine join, in build order
    /// (`chain_internals.len() == max(pieces - 1, 0)`).
    pub chain_internals: Vec<VtreeIdx>,
}

/// The precondition every constructor shares: a vtree has a root, so it has at
/// least one leaf. `#[track_caller]` puts the panic at the constructor the
/// caller named.
#[track_caller]
fn require_nonempty(num_vars: u32) {
    assert!(num_vars > 0, "a vtree needs at least one variable");
}

/// Append a leaf carrying `var` to a node list under construction.
fn push_leaf(nodes: &mut Vec<VtreeNode>, var: VarId) -> VtreeIdx {
    let idx = VtreeIdx(nodes.len() as u32);
    nodes.push(VtreeNode::Leaf { var, parent: None });
    idx
}

/// Append a node joining two subtrees. Parents are left unset: every
/// construction ends in [`Vtree::from_nodes`], which derives them from the
/// child links.
fn push_internal(nodes: &mut Vec<VtreeNode>, left: VtreeIdx, right: VtreeIdx) -> VtreeIdx {
    let idx = VtreeIdx(nodes.len() as u32);
    nodes.push(VtreeNode::Internal {
        left,
        right,
        parent: None,
    });
    idx
}

/// Append a copy of `sub`'s nodes (child links shifted, every leaf renamed
/// through `var`) and return the index its root now has. The one "merge an
/// arena" step [`Vtree::join`] and the graft share.
fn append_subtree(nodes: &mut Vec<VtreeNode>, sub: &Vtree, var: impl Fn(VarId) -> VarId) -> VtreeIdx {
    let offset = nodes.len() as u32;
    for node in &sub.nodes {
        match *node {
            VtreeNode::Leaf { var: local, .. } => {
                push_leaf(nodes, var(local));
            }
            VtreeNode::Internal { left, right, .. } => {
                push_internal(nodes, VtreeIdx(left.0 + offset), VtreeIdx(right.0 + offset));
            }
        }
    }
    VtreeIdx(sub.root.0 + offset)
}

/// A variable tree (vtree): a rooted binary tree whose leaves correspond to variables.
///
/// Nodes are stored with all leaves first (`0..num_leaves`), then all internal
/// nodes (`num_leaves..n`) in bottom-up level order — see the module docs for
/// the `child.idx() < parent.idx()` ordering that gives, and for why a rotated
/// tree is read through [`Vtree::bottomup`] instead.
///
/// # Representation
///
/// The node list, the root and the variable-to-leaf inversion are the tree
/// itself; the traversal orders beside them are derived from it, and every
/// mutating operation re-derives them, which is why none of them is reachable
/// from outside. Read the tree through [`Vtree::node`], [`Vtree::root`],
/// [`Vtree::children`], [`Vtree::leaf_of`], [`Vtree::leaf_var`],
/// [`Vtree::bottomup`] and [`Vtree::lca`]. Three things hold for as long as
/// the `Vtree` lives:
///
/// - **A node's index is its identity.** The node list is never reordered or
///   resized after construction, so a [`VtreeIdx`] a caller holds keeps
///   pointing at the same node — across rotations included. (What a rotation
///   *does* change is the shape: which nodes those indices are linked to.)
/// - **Links run both ways and stay consistent.** Every node names its parent
///   (`None` at the root alone), every internal node names its two children,
///   and a parent's children contain the child that named it.
/// - **Leaves invert.** [`leaf_var(leaf_of(v)) == v`](Vtree::leaf_of) for every
///   variable the vtree covers.
///
/// [`Vtree::validate`] checks all three.
#[derive(Clone, Debug)]
pub struct Vtree {
    /// All vtree nodes: at construction, leaves first (`0..num_leaves`) then
    /// internal nodes in bottom-up level order. A rotation relinks nodes
    /// without reordering this list, which is why `topo` and not the list
    /// order is the topological one.
    nodes: Vec<VtreeNode>,
    /// Index of the root node — the one node whose `parent` is `None`.
    root: VtreeIdx,
    /// Maps a [`VarId`] to the index of the leaf carrying it. Entries for ids
    /// that carry no leaf are meaningless, which is why [`Vtree::leaf_of`] is
    /// documented for covered variables only.
    var_to_leaf: Vec<VtreeIdx>,
    /// Actual number of leaf nodes. When None, equals `var_to_leaf.len()`.
    /// Set explicitly when `VarIds` are sparse (not all entries in `var_to_leaf`
    /// correspond to actual leaves).
    leaf_count: Option<u32>,
    /// Bottom-up topological order over `nodes`. Decoupled from node identity
    /// (a node's index in `nodes` never changes after construction; its
    /// position in `topo` may change after a rotation). Maintained so that
    /// every parent appears after both its children.
    ///
    /// **Root-last property**: for every node `t`, `topo_pos[t]` is the
    /// **maximum** of `topo_pos[d]` over `d ∈ {t} ∪ descendants(t)`. Each
    /// subtree's root sits at the latest topo position among its members.
    /// This is not currently asserted after every operation, but it is
    /// preserved inductively by `rebuild_topo` (strict postorder).
    ///
    /// **Subtree contiguity is NOT guaranteed**: after a sequence of rotations
    /// + fixups, a subtree's members may occupy a non-contiguous set of
    /// positions in `topo`. Consumers must walk parent pointers / child links
    /// to enumerate a subtree, not slice `topo` by position range.
    topo: Vec<VtreeIdx>,
    /// Inverse of `topo`: `topo_pos[idx.idx()]` is the position of node `idx`
    /// in `topo`. Used by `lca()` and as a topological-rank comparator.
    /// Inherits the root-last property from `topo`.
    topo_pos: Vec<u32>,
    /// `topo` filtered to internal nodes only. Recomputed alongside `topo`.
    internal_topo: Vec<VtreeIdx>,
    /// `topo` filtered to leaf nodes only. Recomputed alongside `topo`.
    leaf_topo: Vec<VtreeIdx>,
}

impl Vtree {
    /// Number of leaf nodes. Since nodes are stored leaves-first, indices
    /// `0..num_leaves()` are leaves and `num_leaves()..n` are internal nodes.
    #[inline]
    pub fn num_leaves(&self) -> u32 {
        self.leaf_count.unwrap_or(self.var_to_leaf.len() as u32)
    }

    /// Every node once, children before parents — the order a bottom-up pass
    /// over the tree must visit them in. Reverse it (the iterator is
    /// double-ended) for a top-down pass. This is the maintained topological
    /// order, not `0..num_nodes`: after a rotation the two disagree, and only
    /// this one is still topological.
    pub fn bottomup(&self) -> impl DoubleEndedIterator<Item = VtreeIdx> + ExactSizeIterator + '_ {
        self.topo.iter().copied()
    }

    /// Bottom-up traversal of leaf nodes only, yielding (`node_idx`, `var_id`).
    /// Walks the cached `leaf_topo` slice; preserves topological order of leaves.
    pub fn leaf_bottomup(
        &self,
    ) -> impl DoubleEndedIterator<Item = (VtreeIdx, VarId)> + ExactSizeIterator + '_ {
        self.leaf_topo.iter().map(move |&t| {
            let var = match self.node(t) {
                VtreeNode::Leaf { var, .. } => *var,
                _ => unreachable!("leaf_topo entry is not a leaf"),
            };
            (t, var)
        })
    }

    /// Bottom-up traversal of internal nodes only, yielding (`node_idx`, `left_child`, `right_child`).
    /// Walks the cached `internal_topo` slice; remains valid after rotations.
    pub fn internal_bottomup(
        &self,
    ) -> impl DoubleEndedIterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)> + ExactSizeIterator + '_
    {
        self.internal_topo.iter().map(move |&t| match self.node(t) {
            VtreeNode::Internal { left, right, .. } => (t, *left, *right),
            _ => unreachable!("internal_topo entry is not internal"),
        })
    }

    /// The internal nodes in bottom-up order as a slice — the same sequence
    /// [`Vtree::internal_bottomup`] yields, for a caller that indexes into it.
    #[doc(hidden)]
    pub fn internal_topo_slice(&self) -> &[VtreeIdx] {
        &self.internal_topo
    }

    /// A vtree of one leaf carrying `var`. Its id space is `var.0 + 1`, so a
    /// leaf over `VarId(4)` has `num_vars() == 5` and `num_leaves() == 1`.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree};
    /// let v = Vtree::leaf(VarId(4));
    /// assert_eq!((v.num_leaves(), v.num_vars(), v.num_nodes()), (1, 5, 1));
    /// ```
    pub fn leaf(var: VarId) -> Self {
        let mut nodes = Vec::with_capacity(1);
        let root = push_leaf(&mut nodes, var);
        Self::from_nodes(nodes, root, var.0 + 1)
    }

    /// A new root with `left` and `right` as its subtrees — the composition
    /// primitive every other constructor can be expressed with. The id space
    /// is the larger of the two operands'. `O(|left| + |right|)`.
    ///
    /// # Errors
    ///
    /// [`VtreeError::OverlappingVariable`] if both operands carry the same
    /// variable.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree};
    /// let v = Vtree::join(&Vtree::leaf(VarId(0)), &Vtree::balanced_over(&[VarId(2), VarId(1)])).unwrap();
    /// assert_eq!((v.num_leaves(), v.num_vars()), (3, 3));
    /// assert!(Vtree::join(&v, &Vtree::leaf(VarId(1))).is_err());
    /// ```
    pub fn join(left: &Vtree, right: &Vtree) -> Result<Self, VtreeError> {
        let num_vars = left.num_vars().max(right.num_vars());
        let mut nodes = Vec::with_capacity(left.num_nodes() + right.num_nodes() + 1);
        let l = append_subtree(&mut nodes, left, |v| v);
        let r = append_subtree(&mut nodes, right, |v| v);
        let root = push_internal(&mut nodes, l, r);
        Self::from_nodes_checked(nodes, root, num_vars)
    }

    /// Build a balanced binary vtree over `num_vars` variables (`0..num_vars`)
    /// in natural order — [`Vtree::balanced_over`] on `0, 1, …, n-1`.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn balanced(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (0..num_vars).map(VarId).collect();
        Self::balanced_over(&vars)
    }

    /// A balanced binary vtree whose leaves read `order` left to right: the
    /// order is split in half recursively, so the shape is a function of
    /// `order.len()` alone and only the leaf labels follow `order`. For a
    /// power-of-two length the tree is perfectly symmetric; otherwise one half
    /// carries one more variable.
    ///
    /// The id space is `max(order) + 1`; ids skipped by `order` are uncovered.
    ///
    /// # Panics
    ///
    /// Panics if `order` is empty. `order` must not repeat a variable (checked
    /// in debug builds).
    pub fn balanced_over(order: &[VarId]) -> Self {
        require_nonempty(order.len() as u32);
        let num_vars = order.iter().map(|v| v.0).max().unwrap() + 1;
        let mut nodes = Vec::with_capacity(2 * order.len() - 1);
        let root = Self::build_balanced_recursive(order, &mut nodes);
        let vtree = Self::from_nodes(nodes, root, num_vars);
        debug_assert_eq!(vtree.validate(), Ok(()));
        vtree
    }

    /// Recursively build a balanced vtree over `vars`, appending nodes into
    /// `nodes` and returning the index of the constructed subtree's root.
    /// Parents are left unset; [`Vtree::from_nodes`] derives them.
    #[doc(hidden)]
    pub fn build_balanced_recursive(vars: &[VarId], nodes: &mut Vec<VtreeNode>) -> VtreeIdx {
        if vars.len() == 1 {
            return push_leaf(nodes, vars[0]);
        }
        let mid = vars.len() / 2;
        let left = Self::build_balanced_recursive(&vars[..mid], nodes);
        let right = Self::build_balanced_recursive(&vars[mid..], nodes);
        push_internal(nodes, left, right)
    }

    /// Build a linear vtree over `num_vars` variables (`0..num_vars`).
    /// Structure: each internal node has a single leaf and a subtree containing the
    /// remaining variables. This corresponds to a linear variable order (like OBDDs).
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn linear(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (0..num_vars).rev().map(VarId).collect();
        Self::linear_from_order(&vars)
    }

    /// A right-linear vtree whose leaves read `vars` left to right: each
    /// internal node has `vars[i]` as its left child and everything after it
    /// as its right subtree, so `vars` is the OBDD variable order.
    ///
    /// ```text
    /// vars = [a, b, c, d]:      ∘
    ///                          / \
    ///                         a   ∘
    ///                            / \
    ///                           b   ∘
    ///                              / \
    ///                             c   d
    /// ```
    ///
    /// The id space is `max(vars) + 1`; ids skipped by `vars` are uncovered.
    ///
    /// # Panics
    ///
    /// Panics if `vars` is empty. `vars` must not repeat a variable (checked
    /// in debug builds).
    pub fn linear_from_order(vars: &[VarId]) -> Self {
        require_nonempty(vars.len() as u32);
        let num_vars = vars.iter().map(|v| v.0).max().unwrap() + 1;
        let mut nodes = Vec::with_capacity(2 * vars.len() - 1);
        let mut right = push_leaf(&mut nodes, *vars.last().unwrap());
        for &var in vars[..vars.len() - 1].iter().rev() {
            let left = push_leaf(&mut nodes, var);
            right = push_internal(&mut nodes, left, right);
        }
        let vtree = Self::from_nodes(nodes, right, num_vars);
        debug_assert_eq!(vtree.validate(), Ok(()));
        vtree
    }

    /// Build a random vtree over `num_vars` variables (`0..num_vars`).
    /// Repeatedly picks two random trees from a forest and joins them, until one tree remains.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn random(num_vars: u32, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(seed);
        Self::random_with_rng(num_vars, &mut rng)
    }

    /// Build a random vtree using an externally provided RNG.
    pub(crate) fn random_with_rng(num_vars: u32, rng: &mut impl rand::Rng) -> Self {
        use rand::RngExt;
        use rand::seq::SliceRandom;
        require_nonempty(num_vars);

        let mut nodes = Vec::with_capacity(2 * num_vars as usize - 1);
        let mut var_ids: Vec<u32> = (0..num_vars).collect();
        var_ids.shuffle(rng);
        let mut forest: Vec<VtreeIdx> = var_ids
            .iter()
            .map(|&v| push_leaf(&mut nodes, VarId(v)))
            .collect();

        while forest.len() > 1 {
            let i = rng.random_range(0..forest.len());
            let left = forest.swap_remove(i);
            let j = rng.random_range(0..forest.len());
            let right = forest.swap_remove(j);
            forest.push(push_internal(&mut nodes, left, right));
        }

        Self::from_nodes(nodes, forest[0], num_vars)
    }

    /// Join independent subtrees and single-variable leaves under one
    /// right-linear spine: `subtrees[0]` is the leftmost piece, each later
    /// subtree and then each `spine_vars` leaf is hung one join further down
    /// the right spine, in the order given. A TDD over the result is what
    /// [`crate::tdd::Tdd::graft`] builds from TDDs over the pieces.
    ///
    /// ```text
    /// graft([S0, S1], [x]):        ∘
    ///                             / \
    ///                            ∘   x
    ///                           / \
    ///                          S0  S1
    /// ```
    ///
    /// The id space is the largest any piece needs. `O(total nodes)`.
    ///
    /// # Errors
    ///
    /// [`VtreeError::OverlappingVariable`] if two pieces carry the same
    /// variable; [`VtreeError::Invalid`] if there is no piece at all.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree};
    /// let parts = [Vtree::balanced_over(&[VarId(0), VarId(1)]), Vtree::leaf(VarId(3))];
    /// let v = Vtree::graft(&parts, &[VarId(2)]).unwrap();
    /// assert_eq!((v.num_leaves(), v.num_vars()), (4, 4));
    /// ```
    pub fn graft(subtrees: &[Vtree], spine_vars: &[VarId]) -> Result<Self, VtreeError> {
        let num_vars = subtrees
            .iter()
            .map(Vtree::num_vars)
            .chain(spine_vars.iter().map(|v| v.0 + 1))
            .max()
            .unwrap_or(0);
        let refs: Vec<&Vtree> = subtrees.iter().collect();
        Self::graft_with_layout(&refs, |_, v| v, spine_vars, num_vars).map(|(vtree, _)| vtree)
    }

    /// [`Vtree::graft`] with each subtree's leaves renamed through
    /// `rename(k, local)` on the way in, an explicit id space (which must hold
    /// every renamed id), and the [`GraftLayout`] the TDD-side graft places
    /// levels by. The one graft implementation; the solver's component
    /// compile, whose parts live in per-component id spaces, is what keeps
    /// it reachable from outside the crate.
    #[doc(hidden)]
    pub fn graft_with_layout(
        subtrees: &[&Vtree],
        rename: impl Fn(usize, VarId) -> VarId,
        spine_vars: &[VarId],
        num_vars: u32,
    ) -> Result<(Self, GraftLayout), VtreeError> {
        if subtrees.is_empty() && spine_vars.is_empty() {
            return Err(VtreeError::Invalid(
                "graft needs at least one subtree or spine variable".to_string(),
            ));
        }
        let total: usize = subtrees.iter().map(|s| s.num_nodes()).sum::<usize>()
            + spine_vars.len()
            + subtrees.len();
        let mut nodes: Vec<VtreeNode> = Vec::with_capacity(total);
        let mut pieces: Vec<VtreeIdx> = Vec::with_capacity(subtrees.len() + spine_vars.len());
        let mut comp_offsets: Vec<u32> = Vec::with_capacity(subtrees.len());

        for (k, sub) in subtrees.iter().enumerate() {
            comp_offsets.push(nodes.len() as u32);
            pieces.push(append_subtree(&mut nodes, sub, |v| rename(k, v)));
        }
        for &var in spine_vars {
            pieces.push(push_leaf(&mut nodes, var));
        }

        let mut chain_pre: Vec<VtreeIdx> = Vec::with_capacity(pieces.len() - 1);
        let mut root = pieces[0];
        for &next in &pieces[1..] {
            root = push_internal(&mut nodes, root, next);
            chain_pre.push(root);
        }

        Self::check_each_var_once(&nodes, num_vars)?;
        let (vtree, old_to_new) = Self::reindex_bottomup_with_map(root, nodes, vec![VtreeIdx(0); num_vars as usize]);
        debug_assert_eq!(vtree.validate(), Ok(()));

        let comp_to_full: Vec<Vec<VtreeIdx>> = comp_offsets
            .iter()
            .zip(subtrees)
            .map(|(&offset, sub)| {
                (0..sub.num_nodes())
                    .map(|c| old_to_new[offset as usize + c])
                    .collect()
            })
            .collect();
        let chain_internals: Vec<VtreeIdx> =
            chain_pre.iter().map(|p| old_to_new[p.idx()]).collect();
        Ok((
            vtree,
            GraftLayout {
                comp_to_full,
                chain_internals,
            },
        ))
    }

    /// The overlap check [`Vtree::join`] and the graft share: every leaf in
    /// `nodes` carries a distinct variable below `num_vars`.
    fn check_each_var_once(nodes: &[VtreeNode], num_vars: u32) -> Result<(), VtreeError> {
        let mut seen = vec![false; num_vars as usize];
        for node in nodes {
            if let VtreeNode::Leaf { var, .. } = node {
                if std::mem::replace(&mut seen[var.idx()], true) {
                    return Err(VtreeError::OverlappingVariable(*var));
                }
            }
        }
        Ok(())
    }

    /// [`Vtree::from_nodes`] behind the overlap check, for the constructors
    /// that combine caller-supplied trees.
    fn from_nodes_checked(nodes: Vec<VtreeNode>, root: VtreeIdx, num_vars: u32) -> Result<Self, VtreeError> {
        Self::check_each_var_once(&nodes, num_vars)?;
        let vtree = Self::from_nodes(nodes, root, num_vars);
        debug_assert_eq!(vtree.validate(), Ok(()));
        Ok(vtree)
    }

    /// Restrict this vtree to a subset of its variables, renumbering them.
    ///
    /// `local_of(v)` returns `Some(local_var)` for a variable to KEEP (with its
    /// id in the projected vtree's `0..num_local` space) and `None` for one to
    /// drop. Kept leaves survive verbatim; an internal node whose subtree keeps
    /// variables on only ONE side is spliced out (replaced by that side), and
    /// one that keeps nothing disappears. The surviving skeleton therefore
    /// preserves the original vtree's variable *grouping* — the property vtree
    /// quality actually depends on — without re-running any construction
    /// heuristic.
    ///
    /// O(nodes of `self`), which is what makes it usable in an inner loop that
    /// projects one root vtree onto hundreds of thousands of small residual
    /// formulas: building a fresh vtree per residual is orders of magnitude
    /// more expensive and, in that regime, measurably no better.
    ///
    /// `num_local` must equal the number of variables `local_of` keeps, and the
    /// local ids it yields must be exactly `0..num_local` (each once) — so the
    /// result satisfies `num_leaves() == num_local`, which every compile entry
    /// asserts against its formula's `num_vars`.
    ///
    /// Returns `None` when `local_of` keeps no variable at all (there is no
    /// such thing as an empty vtree).
    pub fn project_to_vars<F>(&self, local_of: F, num_local: u32) -> Option<Vtree>
    where
        F: Fn(VarId) -> Option<VarId>,
    {
        if num_local == 0 {
            return None;
        }
        // `new_of[old.idx()]` = the surviving node that old node `old` maps to,
        // in the fresh (pre-reindex) node array. A spliced-out internal node maps
        // to its single surviving child, so parents see one contiguous skeleton.
        let mut new_of: Vec<Option<VtreeIdx>> = vec![None; self.nodes.len()];
        let mut nodes: Vec<VtreeNode> = Vec::with_capacity(2 * num_local as usize);

        // `bottomup()` walks the side `topo` array, so children are always
        // visited before their parent even after rotations.
        for t in self.bottomup() {
            match &self.nodes[t.idx()] {
                VtreeNode::Leaf { var, .. } => {
                    if let Some(local) = local_of(*var) {
                        new_of[t.idx()] = Some(push_leaf(&mut nodes, local));
                    }
                }
                VtreeNode::Internal { left, right, .. } => {
                    new_of[t.idx()] = match (new_of[left.idx()], new_of[right.idx()]) {
                        (Some(l), Some(r)) => Some(push_internal(&mut nodes, l, r)),
                        // Exactly one side survives: splice this node out.
                        (Some(x), None) | (None, Some(x)) => Some(x),
                        (None, None) => None,
                    };
                }
            }
        }

        let root = new_of[self.root.idx()]?;
        let vtree = Self::from_nodes(nodes, root, num_local);
        debug_assert_eq!(vtree.validate(), Ok(()));
        Some(vtree)
    }

    /// Re-index all nodes in bottom-up level order (leaves first, root last).
    ///
    /// This ordering guarantees `parent.idx()` > `child.idx()`, which enables:
    /// - O(1) bottom-up traversal via `0..n`
    /// - O(depth) LCA via "advance the lower index" (see `lca()`)
    /// Within each tree level, nodes appear left-to-right.
    pub(crate) fn reindex_bottomup(
        root: VtreeIdx,
        old_nodes: Vec<VtreeNode>,
        var_to_leaf: Vec<VtreeIdx>,
    ) -> Self {
        Self::reindex_bottomup_with_map(root, old_nodes, var_to_leaf).0
    }

    /// Same as `reindex_bottomup` but also returns the `old_to_new` permutation
    /// so callers can translate pre-reindex `VtreeIdx` into the final layout.
    fn reindex_bottomup_with_map(
        root: VtreeIdx,
        old_nodes: Vec<VtreeNode>,
        mut var_to_leaf: Vec<VtreeIdx>,
    ) -> (Self, Vec<VtreeIdx>) {
        use std::collections::VecDeque;

        let n = old_nodes.len();
        let mut old_to_new = vec![VtreeIdx(0); n];

        // BFS from root to collect nodes by level (top-down)
        let mut levels: Vec<Vec<VtreeIdx>> = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while !queue.is_empty() {
            let level_size = queue.len();
            let mut level = Vec::with_capacity(level_size);
            for _ in 0..level_size {
                let idx = queue.pop_front().unwrap();
                level.push(idx);
                if let VtreeNode::Internal { left, right, .. } = &old_nodes[idx.idx()] {
                    queue.push_back(*left);
                    queue.push_back(*right);
                }
            }
            levels.push(level);
        }

        // Assign new indices in two passes: all leaves first (bottom-up),
        // then all internals (bottom-up). This guarantees leaves occupy
        // indices 0..num_leaves and internals num_leaves..n, while preserving
        // child.idx() < parent.idx() for every edge.
        let mut new_nodes = Vec::with_capacity(n);

        // Pass 1: all leaves, bottom-up
        for level in levels.iter().rev() {
            for &old_idx in level {
                if let VtreeNode::Leaf { var, .. } = &old_nodes[old_idx.idx()] {
                    let new_idx = VtreeIdx(new_nodes.len() as u32);
                    old_to_new[old_idx.idx()] = new_idx;
                    new_nodes.push(VtreeNode::Leaf {
                        var: *var,
                        parent: None,
                    });
                    var_to_leaf[var.idx()] = new_idx;
                }
            }
        }
        let actual_leaf_count = new_nodes.len() as u32;

        // Pass 2: all internals, bottom-up (children already have lower indices)
        for level in levels.iter().rev() {
            for &old_idx in level {
                if let VtreeNode::Internal { left, right, .. } = &old_nodes[old_idx.idx()] {
                    let new_left = old_to_new[left.idx()];
                    let new_right = old_to_new[right.idx()];
                    let new_idx = VtreeIdx(new_nodes.len() as u32);
                    old_to_new[old_idx.idx()] = new_idx;
                    new_nodes.push(VtreeNode::Internal {
                        left: new_left,
                        right: new_right,
                        parent: None,
                    });
                    Self::set_parent(&mut new_nodes, new_left, new_idx);
                    Self::set_parent(&mut new_nodes, new_right, new_idx);
                }
            }
        }

        let new_root = old_to_new[root.idx()];
        // Set leaf_count explicitly when var_to_leaf is larger than the actual
        // number of leaves (sparse VarIds, e.g. after expand_equivalences with DVE gaps).
        let leaf_count = if actual_leaf_count != var_to_leaf.len() as u32 {
            Some(actual_leaf_count)
        } else {
            None
        };
        let n = new_nodes.len();
        // After reindex_bottomup, the node array is laid out so that idx ==
        // bottom-up topological position. Initialize topo / topo_pos to identity,
        // and split into leaf/internal partitions in bottom-up order.
        let topo: Vec<VtreeIdx> = (0..n as u32).map(VtreeIdx).collect();
        let topo_pos: Vec<u32> = (0..n as u32).collect();
        let mut leaf_topo = Vec::with_capacity(actual_leaf_count as usize);
        let mut internal_topo = Vec::with_capacity(n - actual_leaf_count as usize);
        for &t in &topo {
            if new_nodes[t.idx()].is_leaf() {
                leaf_topo.push(t);
            } else {
                internal_topo.push(t);
            }
        }
        let vtree = Vtree {
            nodes: new_nodes,
            root: new_root,
            var_to_leaf,
            leaf_count,
            topo,
            topo_pos,
            internal_topo,
            leaf_topo,
        };
        (vtree, old_to_new)
    }

    /// Recompute `topo`, `topo_pos`, `internal_topo`, `leaf_topo` from the
    /// current parent/child structure by a full `O(n_nodes)` iterative
    /// postorder — the oracle the rotation tests check the localized
    /// `fixup_topo_after_rotate` against.
    #[cfg(test)]
    pub(crate) fn rebuild_topo(&mut self) {
        let n = self.nodes.len();
        self.topo.clear();
        self.topo.reserve(n);
        self.internal_topo.clear();
        self.leaf_topo.clear();
        if self.topo_pos.len() != n {
            self.topo_pos.resize(n, 0);
        }

        // Iterative postorder via explicit stack: push (idx, visited_yet).
        // First pop pushes children; second pop emits the node.
        let mut stack: Vec<(VtreeIdx, bool)> = Vec::with_capacity(n);
        stack.push((self.root, false));
        while let Some((idx, done)) = stack.pop() {
            if done {
                self.topo_pos[idx.idx()] = self.topo.len() as u32;
                self.topo.push(idx);
                if self.nodes[idx.idx()].is_leaf() {
                    self.leaf_topo.push(idx);
                } else {
                    self.internal_topo.push(idx);
                }
            } else {
                stack.push((idx, true));
                if let VtreeNode::Internal { left, right, .. } = self.nodes[idx.idx()] {
                    // Push right first so left is popped first → left visited
                    // before right (matches the bottom-up order used elsewhere).
                    stack.push((right, false));
                    stack.push((left, false));
                }
            }
        }
        debug_assert_eq!(
            self.topo.len(),
            n,
            "topo missed nodes (disconnected vtree?)"
        );
    }

    /// Localized topo update after a single rotation. O(subtree) instead of
    /// `rebuild_topo`'s `O(n_nodes)`; used by the rotation search hot loop.
    /// See `vtree::rotate` module documentation for the proof.
    pub(crate) fn fixup_topo_after_rotate(&mut self, info: &rotate::RotationInfo, kind: RotationKind) {
        self.fixup_topo_pointers_only_after_rotate(info, kind);
        self.refresh_filtered_topo();
    }

    /// Pointer-only variant of `fixup_topo_after_rotate`: updates `topo` /
    /// `topo_pos` but skips the `O(n_nodes)` `refresh_filtered_topo` walk. Use
    /// when the caller won't read `internal_topo` / `leaf_topo` until a later
    /// explicit `refresh_filtered_topo()`. Rotation search uses this in its hot
    /// loop and refilters once at the end.
    pub(crate) fn fixup_topo_pointers_only_after_rotate(
        &mut self,
        info: &rotate::RotationInfo,
        kind: RotationKind,
    ) {
        let w_pos = self.topo_pos[info.w_idx.idx()] as usize;
        // The single new children-before-parents constraint introduced by a
        // rotation:
        //   Left rotation  v=(A,w),w=(B,C) → v=(w,C),w=(A,B): need A < w.
        //   Right rotation v=(w,C),w=(A,B) → v=(A,w),w=(B,C): need C < w.
        let misplaced_root = match kind {
            RotationKind::Left => info.a_idx,
            RotationKind::Right => info.c_idx,
        };
        // By root-last, topo_pos[misplaced_root] is the maximum topo position
        // over the entire misplaced subtree.
        let m_end = self.topo_pos[misplaced_root.idx()] as usize;

        if m_end < w_pos {
            // Existing topo already satisfies the new constraint. Nothing to do.
            return;
        }

        debug_assert!(
            m_end > w_pos,
            "misplaced_root and w cannot share a topo position"
        );

        // Slice [w_pos ..= m_end] currently starts with w (at w_pos) and ends
        // with the misplaced subtree's root (at m_end). After rotate_left(1),
        // w sits at m_end (one past every element of the misplaced subtree
        // that lay in the slice), and elements in (w_pos..=m_end] shift one
        // position to the left. This is a single contiguous memmove.
        //
        // Subtree contiguity is NOT required: even if non-misplaced elements
        // lie in (w_pos..m_end), the slice rotation preserves children-before-
        // parents for every edge in the post-rotation tree. The full proof is
        // in the `vtree::rotate` module doc.
        self.topo[w_pos..=m_end].rotate_left(1);

        for (offset, &node) in self.topo[w_pos..=m_end].iter().enumerate() {
            self.topo_pos[node.idx()] = (w_pos + offset) as u32;
        }
    }

    /// Refilter `internal_topo` and `leaf_topo` from the current `topo`.
    /// `O(n_nodes)`; called by `fixup_topo_after_rotate` to keep the filtered
    /// views consistent without rebuilding the full topo array.
    pub(crate) fn refresh_filtered_topo(&mut self) {
        self.internal_topo.clear();
        self.leaf_topo.clear();
        for &t in &self.topo {
            if self.nodes[t.idx()].is_leaf() {
                self.leaf_topo.push(t);
            } else {
                self.internal_topo.push(t);
            }
        }
    }

    /// Bottom-up topological order over all nodes (children before parents) as
    /// a slice — the same sequence [`Vtree::bottomup`] yields.
    #[inline]
    pub fn bottomup_topo(&self) -> &[VtreeIdx] {
        &self.topo
    }

    /// Topological position of `idx` (0 = first in bottom-up order, root = last).
    /// A rank comparator: `topo_pos(a) < topo_pos(b)` whenever `a` is a proper
    /// descendant of `b`.
    #[inline]
    #[doc(hidden)]
    pub fn topo_pos(&self, idx: VtreeIdx) -> u32 {
        self.topo_pos[idx.idx()]
    }

    /// Takes `nodes` by mutable slice rather than `&mut self` so it can be
    /// called during vtree construction before the owning `Vtree` is assembled.
    pub(crate) fn set_parent(nodes: &mut [VtreeNode], child: VtreeIdx, parent: VtreeIdx) {
        match &mut nodes[child.idx()] {
            VtreeNode::Leaf { parent: p, .. } => *p = Some(parent),
            VtreeNode::Internal { parent: p, .. } => *p = Some(parent),
        }
    }

    /// Construct a Vtree from a raw node list and root index, reindexing bottom-up.
    ///
    /// The derived tables come from the child links alone: parent links are
    /// wired here (whatever `nodes` says about them is ignored), and the
    /// variable-to-leaf table is filled for every leaf the root reaches, so a
    /// construction hands over child links and nothing else. `num_vars` sizes
    /// the id space — wider than the leaf set is what makes
    /// [`num_leaves`](Vtree::num_leaves) differ from [`num_vars`](Vtree::num_vars).
    ///
    /// Unchecked: `nodes` must describe a single tree rooted at `root` with
    /// each variable on at most one leaf (see [`Vtree::validate`]).
    #[doc(hidden)]
    pub fn from_nodes(nodes: Vec<VtreeNode>, root: VtreeIdx, num_vars: u32) -> Self {
        let var_to_leaf = vec![VtreeIdx(0); num_vars as usize];
        Self::reindex_bottomup(root, nodes, var_to_leaf)
    }

    /// The node at `idx`.
    #[inline]
    pub fn node(&self, idx: VtreeIdx) -> &VtreeNode {
        &self.nodes[idx.idx()]
    }

    /// The root: the one node with no parent, and where a top-down walk starts.
    #[inline]
    pub fn root(&self) -> VtreeIdx {
        self.root
    }

    /// The leaf carrying `var` — the inverse of [`Vtree::leaf_var`].
    ///
    /// For a variable the vtree covers. A vtree whose leaves skip variable ids
    /// answers for the ids in between too, and that answer is meaningless: it
    /// is a leaf, but not one carrying `var`.
    #[inline]
    pub fn leaf_of(&self, var: VarId) -> VtreeIdx {
        self.var_to_leaf[var.idx()]
    }

    /// The variable space this vtree spans: `max(VarId) + 1`, which a formula
    /// compiled against it has to fit inside.
    ///
    /// Equal to [`Vtree::num_leaves`] unless the leaves skip variable ids.
    #[inline]
    pub fn num_vars(&self) -> u32 {
        self.var_to_leaf.len() as u32
    }

    /// Total number of vtree nodes (leaves + internals).
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the children of an internal node. Panics on leaf.
    ///
    /// # Panics
    ///
    /// Panics if `idx` refers to a leaf node.
    pub fn children(&self, idx: VtreeIdx) -> (VtreeIdx, VtreeIdx) {
        match &self.nodes[idx.idx()] {
            VtreeNode::Internal { left, right, .. } => (*left, *right),
            VtreeNode::Leaf { .. } => panic!("children() called on leaf node"),
        }
    }

    /// Whether `other` is the same tree: the same shape, carrying the same
    /// variable at every corresponding leaf — corresponding meaning reached by
    /// the same sequence of left/right steps from the root.
    ///
    /// This is what "the same vtree" means. Node indices, positions in
    /// [`Vtree::bottomup`] and the ids in [`Vtree::to_vtree_text`] are
    /// numbering, not identity, and two constructions that arrive at one tree
    /// are free to number it differently — a rotated tree in particular keeps
    /// its old numbering, so it serializes differently from the same shape
    /// built from scratch while being equal here.
    pub fn same_tree(&self, other: &Vtree) -> bool {
        let mut pairs = vec![(self.root(), other.root())];
        while let Some((a, b)) = pairs.pop() {
            match (self.node(a), other.node(b)) {
                (VtreeNode::Leaf { var: va, .. }, VtreeNode::Leaf { var: vb, .. }) => {
                    if va != vb {
                        return false;
                    }
                }
                (VtreeNode::Internal { .. }, VtreeNode::Internal { .. }) => {
                    let (a_left, a_right) = self.children(a);
                    let (b_left, b_right) = other.children(b);
                    pairs.push((a_left, b_left));
                    pairs.push((a_right, b_right));
                }
                _ => return false,
            }
        }
        true
    }

    /// Get the variable at a leaf node. Panics on internal.
    ///
    /// # Panics
    ///
    /// Panics if `idx` refers to an internal node.
    pub fn leaf_var(&self, idx: VtreeIdx) -> VarId {
        match &self.nodes[idx.idx()] {
            VtreeNode::Leaf { var, .. } => *var,
            VtreeNode::Internal { .. } => panic!("leaf_var() called on internal node"),
        }
    }

    /// Get the sibling of a node (the other child of its parent). Panics if root.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is the root node (it has no parent, hence no sibling).
    pub fn sibling(&self, idx: VtreeIdx) -> VtreeIdx {
        let parent = self.node(idx).parent().expect("sibling() called on root");
        let (left, right) = self.children(parent);
        if left == idx { right } else { left }
    }

    /// Lowest common ancestor of two vtree nodes.
    ///
    /// Compares topological position via `topo_pos`: at each step, advance the
    /// node with the lower topological rank to its parent until both paths
    /// converge. Independent of raw `VtreeIdx` ordering, so this remains
    /// correct after rotations leave the node array in non-topological idx
    /// order. O(depth), no allocation.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` do not belong to the same tree (their paths never converge).
    pub fn lca(&self, mut a: VtreeIdx, mut b: VtreeIdx) -> VtreeIdx {
        while a != b {
            if self.topo_pos[a.idx()] < self.topo_pos[b.idx()] {
                a = self.node(a).parent().expect("nodes should share a root");
            } else {
                b = self.node(b).parent().expect("nodes should share a root");
            }
        }
        a
    }

    /// Check the representation invariants: exactly one parentless node and it
    /// is the root; every child's parent link names its parent and no node is
    /// the child of two nodes; the bottom-up order covers every node once with
    /// children before parents (so there is no cycle) and its filtered views
    /// agree with it; every variable sits on at most one leaf, within the id
    /// space, with [`Vtree::leaf_of`] pointing at it; and
    /// [`Vtree::num_leaves`] counts the leaves.
    ///
    /// `O(nodes)`. Every constructor that combines caller-supplied trees runs
    /// this in debug builds; call it yourself after hand-built input.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] naming the first invariant found broken.
    pub fn validate(&self) -> Result<(), VtreeError> {
        let n = self.nodes.len();
        let invalid = |msg: String| Err(VtreeError::Invalid(msg));
        if n == 0 {
            return invalid("no nodes".to_string());
        }
        if self.root.idx() >= n {
            return invalid(format!("root {} is not a node index", self.root.0));
        }

        // Links: one parentless node (the root), every child slot names a
        // node whose parent link points back, no node claimed twice.
        let mut claimed = vec![false; n];
        for (i, node) in self.nodes.iter().enumerate() {
            if let VtreeNode::Internal { left, right, .. } = node {
                for child in [*left, *right] {
                    if child.idx() >= n {
                        return invalid(format!("node {i} names child {} outside the node list", child.0));
                    }
                    if self.nodes[child.idx()].parent() != Some(VtreeIdx(i as u32)) {
                        return invalid(format!("node {} is a child of {i} but its parent link disagrees", child.0));
                    }
                    if std::mem::replace(&mut claimed[child.idx()], true) {
                        return invalid(format!("node {} is a child of two nodes", child.0));
                    }
                }
            }
        }
        for (i, node) in self.nodes.iter().enumerate() {
            match (node.parent(), i == self.root.idx()) {
                (None, false) => return invalid(format!("node {i} has no parent but is not the root")),
                (Some(_), true) => return invalid(format!("root {i} has a parent")),
                (Some(p), false) if !claimed[i] || p.idx() >= n => {
                    return invalid(format!("node {i} names a parent that does not list it as a child"));
                }
                _ => {}
            }
        }

        // Bottom-up order: every node once, children before parents, inverse
        // positions and filtered views consistent.
        if self.topo.len() != n || self.topo_pos.len() != n {
            return invalid("bottom-up order does not cover the node list".to_string());
        }
        let mut seen = vec![false; n];
        for (pos, &t) in self.topo.iter().enumerate() {
            if t.idx() >= n || std::mem::replace(&mut seen[t.idx()], true) {
                return invalid(format!("bottom-up order lists node {} twice or out of range", t.0));
            }
            if self.topo_pos[t.idx()] as usize != pos {
                return invalid(format!("bottom-up position of node {} is inconsistent", t.0));
            }
            if let VtreeNode::Internal { left, right, .. } = &self.nodes[t.idx()] {
                if !seen[left.idx()] || !seen[right.idx()] {
                    return invalid(format!("node {} precedes one of its children in the bottom-up order", t.0));
                }
            }
        }
        let leaves_in_order = self.topo.iter().filter(|t| self.nodes[t.idx()].is_leaf()).count();
        if self.leaf_topo.len() != leaves_in_order
            || self.internal_topo.len() != n - leaves_in_order
            || !self.leaf_topo.iter().all(|t| self.nodes[t.idx()].is_leaf())
            || self.internal_topo.iter().any(|t| self.nodes[t.idx()].is_leaf())
        {
            return invalid("leaf/internal views disagree with the bottom-up order".to_string());
        }

        // Leaves: distinct variables inside the id space, inverted by leaf_of.
        let mut leaf_count = 0u32;
        let mut carried = vec![false; self.var_to_leaf.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            if let VtreeNode::Leaf { var, .. } = node {
                leaf_count += 1;
                if var.idx() >= carried.len() {
                    return invalid(format!("leaf {i} carries variable {} outside the id space {}", var.0, carried.len()));
                }
                if std::mem::replace(&mut carried[var.idx()], true) {
                    return invalid(format!("variable {} sits on two leaves", var.0));
                }
                if self.var_to_leaf[var.idx()].idx() != i {
                    return invalid(format!("leaf_of({}) does not point at leaf {i}", var.0));
                }
            }
        }
        if leaf_count != self.num_leaves() {
            return invalid(format!("num_leaves() is {} but {leaf_count} leaves exist", self.num_leaves()));
        }
        Ok(())
    }
}

/// The `.vtree` text codec, in both directions.
mod text;

pub mod rotate; // In-place vtree left/right rotations + topo fixup

#[cfg(test)]
#[path = "vtree_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "topo_microbench.rs"]
mod topo_microbench;
