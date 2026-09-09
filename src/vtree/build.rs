//! Constructors: the node-list primitives, the named vtree shapes, and the
//! bottom-up reindex every construction finishes with.

use super::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};

/// The precondition every constructor shares: a vtree has a root, so it has at
/// least one leaf. `#[track_caller]` puts the panic at the constructor the
/// caller named.
#[track_caller]
pub(super) fn require_nonempty(num_vars: u32) {
    assert!(num_vars > 0, "a vtree needs at least one variable");
}

/// Append a leaf carrying `var` to a node list under construction.
pub(super) fn push_leaf(nodes: &mut Vec<VtreeNode>, var: VarId) -> VtreeIdx {
    let idx = VtreeIdx(nodes.len() as u32);
    nodes.push(VtreeNode::Leaf { var, parent: None });
    idx
}

/// Append a node joining two subtrees. Parents are left unset: every
/// construction ends in [`Vtree::from_nodes`], which derives them from the
/// child links.
pub(super) fn push_internal(nodes: &mut Vec<VtreeNode>, left: VtreeIdx, right: VtreeIdx) -> VtreeIdx {
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
pub(super) fn append_subtree(nodes: &mut Vec<VtreeNode>, sub: &Vtree, var: impl Fn(VarId) -> VarId) -> VtreeIdx {
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

impl Vtree {
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
        Self::linear_over(&vars)
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
    pub fn linear_over(vars: &[VarId]) -> Self {
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

    /// The overlap check [`Vtree::join`] and the graft share: every leaf in
    /// `nodes` carries a distinct variable below `num_vars`.
    pub(super) fn check_each_var_once(nodes: &[VtreeNode], num_vars: u32) -> Result<(), VtreeError> {
        let mut seen = vec![false; num_vars as usize];
        for node in nodes {
            if let VtreeNode::Leaf { var, .. } = node
                && std::mem::replace(&mut seen[var.idx()], true) {
                    return Err(VtreeError::OverlappingVariable(*var));
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

    /// Re-index all nodes in bottom-up level order (leaves first, root last).
    ///
    /// This ordering guarantees `parent.idx()` > `child.idx()`, which enables:
    /// - O(1) bottom-up traversal via `0..n`
    /// - O(depth) LCA via "advance the lower index" (see `lca()`)
    ///
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
    pub(super) fn reindex_bottomup_with_map(
        root: VtreeIdx,
        old_nodes: Vec<VtreeNode>,
        mut var_to_leaf: Vec<VtreeIdx>,
    ) -> (Self, Vec<VtreeIdx>) {
        let levels = levels_from_root(root, &old_nodes);
        let (new_nodes, old_to_new, actual_leaf_count) =
            relabel_leaves_then_internals(&levels, &old_nodes, &mut var_to_leaf);

        let new_root = old_to_new[root.idx()];
        // Set leaf_count explicitly when var_to_leaf is larger than the actual
        // number of leaves (sparse VarIds, e.g. After expand_equivalences with DVE gaps).
        let leaf_count = if actual_leaf_count != var_to_leaf.len() as u32 {
            Some(actual_leaf_count)
        } else {
            None
        };
        // After reindex_bottomup, the node array is laid out so that idx ==
        // bottom-up topological position, so the identity order is correct.
        let topo = crate::vtree::topo::TopoOrder::identity(&new_nodes);
        let vtree = Vtree {
            nodes: new_nodes,
            root: new_root,
            var_to_leaf,
            leaf_count,
            topo,
        };
        (vtree, old_to_new)
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
    pub fn from_nodes(nodes: Vec<VtreeNode>, root: VtreeIdx, num_vars: u32) -> Self {
        let var_to_leaf = vec![VtreeIdx(0); num_vars as usize];
        Self::reindex_bottomup(root, nodes, var_to_leaf)
    }
}

/// The nodes reachable from `root`, grouped by depth (top-down BFS).
fn levels_from_root(root: VtreeIdx, old_nodes: &[VtreeNode]) -> Vec<Vec<VtreeIdx>> {
    use std::collections::VecDeque;

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
    levels
}

/// Rebuild the node list in two bottom-up passes: all leaves first, then all
/// internals. This puts leaves at `0..num_leaves` and internals above them
/// while preserving `child.idx() < parent.idx()` for every edge. Returns the
/// new nodes, the `old_to_new` permutation, and the leaf count.
fn relabel_leaves_then_internals(
    levels: &[Vec<VtreeIdx>],
    old_nodes: &[VtreeNode],
    var_to_leaf: &mut [VtreeIdx],
) -> (Vec<VtreeNode>, Vec<VtreeIdx>, u32) {
    let n = old_nodes.len();
    let mut old_to_new = vec![VtreeIdx(0); n];
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
                Vtree::set_parent(&mut new_nodes, new_left, new_idx);
                Vtree::set_parent(&mut new_nodes, new_right, new_idx);
            }
        }
    }
    (new_nodes, old_to_new, actual_leaf_count)
}
