//! The vtree node enum and the `Vtree` structure itself, with its readers.

use super::topo::TopoOrder;
use super::{VarId, VtreeIdx};

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

/// A variable tree (vtree): a rooted binary tree whose leaves correspond to variables.
///
/// Use [`Vtree::balanced`] to group contiguous variables into subtrees, or
/// [`Vtree::linear`] for a right-linear tree representing a variable order.
/// The tree determines the full variable universe, including free variables
/// that a particular function does not mention.
///
/// Wrap the tree in one `Arc` and share that allocation among operands:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and, literal, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = literal(&vtree, 1)?;
/// let g = literal(f.vtree(), -2)?;
/// assert!(Arc::ptr_eq(f.vtree(), g.vtree()));
/// let both = and(f, g)?;
/// assert_eq!(both.model_count()?, 2u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// Two independently constructed trees are different allocations, even when
/// their shapes and variable labels match; binary diagram operations reject
/// that combination. Cloning the `Arc` preserves compatibility.
///
/// For sparse ids, [`Vtree::balanced_over`] and [`Vtree::linear_from_order`]
/// take the variables explicitly. [`Vtree::num_leaves`] counts present
/// variables; [`Vtree::num_vars`] is the variable-id space, large enough to
/// index every named variable.
/// Weight tables use the latter size, while model counts range over the former.
///
/// # Choosing a tree
///
/// Start with [`Vtree::balanced`] for a small experiment. Its grouping follows
/// the variable ids; it does not inspect your constraints. When a variable order
/// is already known, [`Vtree::linear_from_order`] represents that order.
/// [`Vtree::balanced_over`] lets you keep related variables together while
/// retaining a balanced shape; [`Vtree::join`] makes the groups explicit.
///
/// The tree can greatly affect diagram size and the cost of building it.
/// A balanced shape alone does not guarantee a compact diagram. Compare
/// [`Tdd::pair_count`](crate::Tdd::pair_count) for your functions under different
/// groupings, and use [`Context::with_limits`](crate::Context::with_limits) when
/// exploring larger inputs.
/// Once a diagram is built, [`Tdd::rotation_search`](crate::Tdd::rotation_search)
/// can search nearby tree shapes. [`minimize`](crate::Tdd::minimize) instead
/// removes redundancy under the current tree and keeps its variable grouping.
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
    /// Reusable execution scratch shared by clones of this tree.
    pub(crate) context: std::sync::Arc<crate::Context>,
    /// All vtree nodes: at construction, leaves first (`0..num_leaves`) then
    /// internal nodes in bottom-up level order. A rotation relinks nodes
    /// without reordering this list, which is why `topo` and not the list
    /// order is the topological one.
    pub(super) nodes: Vec<VtreeNode>,
    /// Index of the root node — the one node whose `parent` is `None`.
    pub(super) root: VtreeIdx,
    /// Maps a [`VarId`] to the index of the leaf carrying it. Entries for ids
    /// that carry no leaf are meaningless, which is why [`Vtree::leaf_of`] is
    /// documented for covered variables only.
    pub(super) var_to_leaf: Vec<VtreeIdx>,
    /// Actual number of leaf nodes. When None, equals `var_to_leaf.len()`.
    /// Set explicitly when `VarIds` are sparse (not all entries in `var_to_leaf`
    /// correspond to actual leaves).
    pub(super) leaf_count: Option<u32>,
    /// Bottom-up topological order over `nodes`, with its inverse and the two
    /// filtered views. Decoupled from node identity: a node's index in `nodes`
    /// never changes after construction, but its position in the order may
    /// change after a rotation. See [`TopoOrder`] for the properties it
    /// maintains.
    pub(super) topo: TopoOrder,
}

impl Vtree {
    /// The reusable execution context associated with this tree.
    ///
    /// Cloning a vtree preserves its context. Context sharing does not change
    /// the requirement that binary diagram operands share one vtree allocation.
    #[must_use]
    pub fn context(&self) -> &std::sync::Arc<crate::Context> {
        &self.context
    }

    /// Associate this tree with `context`, preserving its shape and variable ids.
    ///
    /// Set the context before sharing the tree with diagrams; [`Context::bind`](crate::Context::bind)
    /// also wraps it in an `Arc`.
    #[must_use]
    pub fn with_context(mut self, context: std::sync::Arc<crate::Context>) -> Self {
        self.context = context;
        self
    }

    /// Number of leaf nodes. Since nodes are stored leaves-first, indices
    /// `0..num_leaves()` are leaves and `num_leaves()..n` are internal nodes.
    #[inline]
    pub fn num_leaves(&self) -> u32 {
        self.leaf_count.unwrap_or(self.var_to_leaf.len() as u32)
    }

    /// Takes `nodes` by mutable slice rather than `&mut self` so it can be
    /// called during vtree construction before the owning `Vtree` is assembled.
    pub(crate) fn set_parent(nodes: &mut [VtreeNode], child: VtreeIdx, parent: VtreeIdx) {
        match &mut nodes[child.idx()] {
            VtreeNode::Leaf { parent: p, .. } => *p = Some(parent),
            VtreeNode::Internal { parent: p, .. } => *p = Some(parent),
        }
    }

    /// The node at `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not below [`num_nodes`](Self::num_nodes).
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
    /// `None` when the vtree has no leaf for `var`: the id may be past the
    /// variable space, or it may be one of the ids a vtree whose leaves skip
    /// variable ids leaves uncarried.
    #[inline]
    pub fn leaf_of(&self, var: VarId) -> Option<VtreeIdx> {
        let leaf = *self.var_to_leaf.get(var.idx())?;
        match self.nodes.get(leaf.idx()) {
            Some(VtreeNode::Leaf { var: on_leaf, .. }) if *on_leaf == var => Some(leaf),
            _ => None,
        }
    }

    /// Size of the variable-id space, at least the largest leaf's [`VarId`] plus one.
    ///
    /// This is a sufficient size for a table indexed by variable id. It equals
    /// [`Vtree::num_leaves`] when the leaves are exactly `0..num_vars`; sparse ids
    /// or a larger space passed to [`Vtree::from_nodes`] make it larger.
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
    #[inline]
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
    /// [`Vtree::bottomup`] and the ids in [`Vtree::to_text`] are
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
    #[inline]
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

    /// Lowest common ancestor of two vtree nodes. O(depth), no allocation;
    /// reads [`Vtree::topo_pos`], so it stays correct on a rotated tree.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` do not belong to the same tree (their paths never converge).
    pub fn lca(&self, mut a: VtreeIdx, mut b: VtreeIdx) -> VtreeIdx {
        while a != b {
            if self.topo.pos(a) < self.topo.pos(b) {
                a = self.node(a).parent().expect("nodes should share a root");
            } else {
                b = self.node(b).parent().expect("nodes should share a root");
            }
        }
        a
    }
}
