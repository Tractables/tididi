//! Constructors: the node-list primitives, the named vtree shapes, and the
//! bottom-up reindex every construction finishes with.

use super::rng::Lcg;
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
        Self::from_nodes(nodes, root, var.0 + 1).expect("one leaf is a tree")
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
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    /// let v = Vtree::join(&Vtree::leaf(VarId(0)), &Vtree::balanced_over(&[VarId(2), VarId(1)])).unwrap();
    /// assert_eq!((v.num_leaves(), v.num_vars()), (3, 3));
    /// // Var 1 is already in `v`, so the join is refused and names the clash.
    /// let clash = Vtree::join(&v, &Vtree::leaf(VarId(1)));
    /// assert!(matches!(clash, Err(VtreeError::OverlappingVariable(VarId(1)))));
    /// ```
    pub fn join(left: &Vtree, right: &Vtree) -> Result<Self, VtreeError> {
        let num_vars = left.num_vars().max(right.num_vars());
        let mut nodes = Vec::with_capacity(left.num_nodes() + right.num_nodes() + 1);
        let l = append_subtree(&mut nodes, left, |v| v);
        let r = append_subtree(&mut nodes, right, |v| v);
        let root = push_internal(&mut nodes, l, r);
        Self::from_nodes(nodes, root, num_vars)
    }

    /// Build a balanced binary vtree over `num_vars` variables (`0..num_vars`)
    /// in natural order — [`Vtree::balanced_over`] on `0, 1, …, n-1`.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree};
    ///
    /// let vtree = Vtree::balanced(4);
    /// assert_eq!(vtree.num_vars(), 4);
    /// assert_eq!(vtree.num_nodes(), 7);          // four leaves, three internal nodes
    /// assert_eq!(vtree.node(vtree.root()).is_leaf(), false);
    /// // Natural order: the leaves carry 0, 1, 2, 3 left to right.
    /// assert!(vtree.leaf_of(VarId(3)).is_some());
    /// ```
    pub fn balanced(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (0..num_vars).map(VarId).collect();
        Self::balanced_over(&vars)
    }

    /// A balanced binary vtree whose leaves read `order` left to right: the
    /// order is split in half recursively, so the shape is a function of
    /// `order.len()` alone and only the leaf labels follow `order`. For a
    /// power-of-two length the tree is perfectly symmetric; otherwise the
    /// right half of an odd split carries one more variable.
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
        Self::from_nodes(nodes, root, num_vars).expect("the recursion appends one tree")
    }

    /// Recursively build a balanced vtree over `vars`, appending nodes into
    /// `nodes` and returning the index of the constructed subtree's root.
    /// Parents are left unset; [`Vtree::from_nodes`] derives them. `vars`
    /// must not be empty.
    pub fn build_balanced_recursive(vars: &[VarId], nodes: &mut Vec<VtreeNode>) -> VtreeIdx {
        if vars.len() == 1 {
            return push_leaf(nodes, vars[0]);
        }
        let mid = vars.len() / 2;
        let left = Self::build_balanced_recursive(&vars[..mid], nodes);
        let right = Self::build_balanced_recursive(&vars[mid..], nodes);
        push_internal(nodes, left, right)
    }

    /// A right-linear vtree over `0..num_vars` in ascending order:
    /// [`Vtree::linear_from_order`] on `0, …, n-1`, so variable `0` is the
    /// root's left leaf and variable `n-1` sits deepest.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn linear(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (0..num_vars).map(VarId).collect();
        Self::linear_from_order(&vars)
    }

    /// A right-linear vtree over `n-1, …, 0`, reversing [`Vtree::linear`]'s order:
    /// variable `n-1` is the root's left leaf and variable `0` sits deepest.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn reverse_linear(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (0..num_vars).rev().map(VarId).collect();
        Self::linear_from_order(&vars)
    }

    /// A right-linear vtree whose leaves read `vars` left to right: each
    /// internal node has `vars[i]` as its left child and everything after it
    /// as its right subtree, so `vars` is the variable order an ordered binary
    /// decision diagram would read.
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
        Self::from_nodes(nodes, right, num_vars).expect("the chain is one tree")
    }

    /// Build a random vtree over `num_vars` variables (`0..num_vars`).
    /// Repeatedly picks two random trees from a forest and joins them, until
    /// one tree remains. The same `seed` gives the same tree, on every
    /// platform and in every release.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn random(num_vars: u32, seed: u64) -> Self {
        require_nonempty(num_vars);
        let rng = &mut Lcg::new(seed);

        let mut nodes = Vec::with_capacity(2 * num_vars as usize - 1);
        let mut var_ids: Vec<u32> = (0..num_vars).collect();
        for i in (1..var_ids.len()).rev() {
            var_ids.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut forest: Vec<VtreeIdx> = var_ids
            .iter()
            .map(|&v| push_leaf(&mut nodes, VarId(v)))
            .collect();

        while forest.len() > 1 {
            let i = rng.below(forest.len() as u64) as usize;
            let left = forest.swap_remove(i);
            let j = rng.below(forest.len() as u64) as usize;
            let right = forest.swap_remove(j);
            forest.push(push_internal(&mut nodes, left, right));
        }

        Self::from_nodes(nodes, forest[0], num_vars).expect("the forest collapses to one tree")
    }

    /// The overlap check [`Vtree::from_nodes`] and the graft share: every leaf
    /// in `nodes` carries a distinct variable below `num_vars`.
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

    /// Re-index all nodes in bottom-up level order (leaves first, root last),
    /// returning the tree and the `old_to_new` permutation a caller translates
    /// pre-reindex `VtreeIdx` values through.
    ///
    /// This ordering guarantees `parent.idx()` > `child.idx()`, which enables:
    /// - O(1) bottom-up traversal via `0..n`
    /// - O(depth) LCA via "advance the lower index" (see `lca()`)
    ///
    /// Within each tree level, nodes appear left-to-right.
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
        // number of leaves (sparse `VarId`s, left by a caller whose variable
        // numbering has gaps).
        let leaf_count = if actual_leaf_count != var_to_leaf.len() as u32 {
            Some(actual_leaf_count)
        } else {
            None
        };
        // After the reindex, the node array is laid out so that idx ==
        // bottom-up topological position, so the identity order is correct.
        let topo = crate::vtree::topo::TopoOrder::identity(&new_nodes);
        let vtree = Vtree {
            context: std::sync::Arc::new(crate::engine::Context::new()),
            nodes: new_nodes,
            root: new_root,
            var_to_leaf,
            leaf_count,
            topo,
        };
        (vtree, old_to_new)
    }

    /// Construct a vtree from a raw node list and root index, reindexing
    /// bottom-up.
    ///
    /// The derived tables come from the child links alone: parent links are
    /// wired here (whatever `nodes` says about them is ignored), and the
    /// variable-to-leaf table is filled for every leaf the root reaches, so a
    /// construction hands over child links and nothing else. `num_vars` sizes
    /// the id space — wider than the leaf set is what makes
    /// [`num_leaves`](Vtree::num_leaves) differ from [`num_vars`](Vtree::num_vars).
    ///
    /// The list is checked before it is read, so no caller can build a vtree
    /// that [`Vtree::validate`] would reject.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if `nodes` is empty, if an index names no node,
    /// if a leaf carries a variable at or past `num_vars`, or if the links do
    /// not reach every node exactly once from `root`;
    /// [`VtreeError::OverlappingVariable`] if two leaves carry one variable.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};
    ///
    /// let nodes = vec![
    ///     VtreeNode::Leaf { var: VarId(0), parent: None },
    ///     VtreeNode::Leaf { var: VarId(1), parent: None },
    ///     VtreeNode::Internal { left: VtreeIdx(0), right: VtreeIdx(1), parent: None },
    /// ];
    /// let v = Vtree::from_nodes(nodes, VtreeIdx(2), 2).unwrap();
    /// assert_eq!((v.num_leaves(), v.num_vars()), (2, 2));
    ///
    /// // A lone leaf the root cannot reach makes the list something other
    /// // than one tree.
    /// let stray = vec![
    ///     VtreeNode::Leaf { var: VarId(0), parent: None },
    ///     VtreeNode::Leaf { var: VarId(1), parent: None },
    /// ];
    /// assert!(matches!(
    ///     Vtree::from_nodes(stray, VtreeIdx(0), 2),
    ///     Err(VtreeError::Invalid(_)),
    /// ));
    /// ```
    pub fn from_nodes(
        nodes: Vec<VtreeNode>,
        root: VtreeIdx,
        num_vars: u32,
    ) -> Result<Self, VtreeError> {
        check_node_list(&nodes, root, num_vars)?;
        let var_to_leaf = vec![VtreeIdx(0); num_vars as usize];
        let (vtree, _) = Self::reindex_bottomup_with_map(root, nodes, var_to_leaf);
        debug_assert_eq!(vtree.validate(), Ok(()));
        Ok(vtree)
    }
}

/// The check [`Vtree::from_nodes`] runs before it reads the list: every index
/// names a node, the links reach every node exactly once from `root`, and every
/// variable sits on one leaf, inside the id space.
///
/// It cannot be [`Vtree::validate`] on the result. The reindex walks the child
/// links to build the vtree at all, so a list with a cycle or a doubly-parented
/// node loops or indexes out of bounds there — before any vtree exists to
/// validate.
fn check_node_list(nodes: &[VtreeNode], root: VtreeIdx, num_vars: u32) -> Result<(), VtreeError> {
    let n = nodes.len();
    if n == 0 {
        return Err(VtreeError::Invalid("a vtree needs at least one node".to_string()));
    }
    if root.idx() >= n {
        return Err(VtreeError::Invalid(format!("root {} is not a node index", root.0)));
    }
    for node in nodes {
        match *node {
            VtreeNode::Leaf { var, .. } => {
                if var.0 >= num_vars {
                    return Err(VtreeError::Invalid(format!(
                        "leaf variable {} is outside an id space of {num_vars}",
                        var.0
                    )));
                }
            }
            VtreeNode::Internal { left, right, .. } => {
                for child in [left, right] {
                    if child.idx() >= n {
                        return Err(VtreeError::Invalid(format!(
                            "child {} is not a node index",
                            child.0
                        )));
                    }
                }
            }
        }
    }
    let mut seen = vec![false; n];
    seen[root.idx()] = true;
    let mut reached = 1usize;
    let mut stack = vec![root];
    while let Some(idx) = stack.pop() {
        if let VtreeNode::Internal { left, right, .. } = nodes[idx.idx()] {
            for child in [left, right] {
                if std::mem::replace(&mut seen[child.idx()], true) {
                    return Err(VtreeError::Invalid(format!(
                        "node {} is reached twice, so the links are not a tree",
                        child.idx()
                    )));
                }
                reached += 1;
                stack.push(child);
            }
        }
    }
    if reached != n {
        return Err(VtreeError::Invalid(format!(
            "{} of the {n} nodes are unreachable from the root",
            n - reached
        )));
    }
    Vtree::check_each_var_once(nodes, num_vars)
}

/// The nodes reachable from `root`, grouped by depth, walked breadth-first from the root.
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
