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

/// The id space a vtree may cover when the node list gives no better bound: a
/// single leaf carrying `VarId(4_000_000_000)` is a legal one-line file, and
/// the tables below would be 16 GB for it. Sparse ids stay legal — `to_text`
/// writes them back verbatim, so they have to round-trip — it is the span they
/// cover that is bounded, and this is the floor under that bound.
const MIN_VAR_SPACE_BYTES: usize = 64 * 1024 * 1024;

/// The floor on the id space, the widest `num_vars` whose `var_to_leaf` entry
/// still fits [`MIN_VAR_SPACE_BYTES`].
const MIN_NUM_VARS: u32 = {
    let entries = MIN_VAR_SPACE_BYTES / size_of::<VtreeIdx>();
    if entries > u32::MAX as usize { u32::MAX } else { entries as u32 }
};

/// Bound variable-indexed tables by the node list's footprint, with a floor
/// that permits sparse identifiers in small vtrees.
fn max_num_vars(num_nodes: usize) -> u32 {
    let bytes = num_nodes.saturating_mul(size_of::<VtreeNode>());
    let entries = (bytes / size_of::<VtreeIdx>()).min(u32::MAX as usize) as u32;
    entries.max(MIN_NUM_VARS)
}

/// Refuse an id space too wide for the node list to index. Every construction
/// checks this before allocating a table indexed by variable id.
pub(crate) fn check_var_space(num_vars: u32, num_nodes: usize) -> Result<(), VtreeError> {
    let max_num_vars = max_num_vars(num_nodes);
    if num_vars > max_num_vars {
        return Err(VtreeError::VariableSpaceTooLarge { num_vars, max_num_vars });
    }
    Ok(())
}

/// The id space of a leaf order, its largest number, refusing an empty order.
fn id_space(vars: &[VarId]) -> Result<u32, VtreeError> {
    vars.iter()
        .map(|v| v.0)
        .max()
        .ok_or_else(|| VtreeError::Invalid("a vtree needs at least one variable".to_string()))
}

/// Point `child`'s parent link at `parent`, in a node list that may still be
/// under construction.
pub(super) fn set_parent(nodes: &mut [VtreeNode], child: VtreeIdx, parent: VtreeIdx) {
    match &mut nodes[child.idx()] {
        VtreeNode::Leaf { parent: p, .. } => *p = Some(parent),
        VtreeNode::Internal { parent: p, .. } => *p = Some(parent),
    }
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
    /// A vtree of one leaf carrying `var`. Its id space is `var.0`, so a
    /// leaf over `VarId(5)` has `num_vars() == 5` and `num_leaves() == 1`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is zero, which is not a variable, or if it is wider
    /// than the id space one node may declare (the bound behind
    /// [`VtreeError::VariableSpaceTooLarge`]).
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree};
    /// let vtree = Vtree::leaf(VarId(5));
    /// assert_eq!((vtree.num_leaves(), vtree.num_vars(), vtree.num_nodes()), (1, 5, 1));
    /// ```
    pub fn leaf(var: VarId) -> Self {
        let mut nodes = Vec::with_capacity(1);
        let root = push_leaf(&mut nodes, var);
        Self::from_nodes(nodes, root, var.0)
            .unwrap_or_else(|error| panic!("Vtree::leaf({}): {error}", var.0))
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
    /// let vtree = Vtree::join(&Vtree::leaf(VarId(1)), &Vtree::balanced_over(&[VarId(3), VarId(2)])?)?;
    /// assert_eq!((vtree.num_leaves(), vtree.num_vars()), (3, 3));
    /// // Var 1 is already in `vtree`, so the join is refused and names the clash.
    /// let clash = Vtree::join(&vtree, &Vtree::leaf(VarId(2)));
    /// assert!(matches!(clash, Err(VtreeError::OverlappingVariable(VarId(2)))));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn join(left: &Vtree, right: &Vtree) -> Result<Self, VtreeError> {
        let num_vars = left.num_vars().max(right.num_vars());
        let mut nodes = Vec::with_capacity(left.num_nodes() + right.num_nodes() + 1);
        let l = append_subtree(&mut nodes, left, |v| v);
        let r = append_subtree(&mut nodes, right, |v| v);
        let root = push_internal(&mut nodes, l, r);
        Self::from_nodes(nodes, root, num_vars)
    }

    /// Build a balanced binary vtree over `num_vars` variables
    /// (`1..=num_vars`) in natural order — [`Vtree::balanced_over`] on
    /// `1, 2, …, n`.
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
    /// // Natural order: the leaves carry 1, 2, 3, 4 left to right.
    /// assert!(vtree.leaf_of(VarId(4)).is_some());
    /// ```
    pub fn balanced(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (1..=num_vars).map(VarId).collect();
        Self::balanced_over(&vars).expect("the ids are distinct")
    }

    /// A balanced binary vtree whose leaves read `order` left to right: the
    /// order is split in half recursively, so the shape is a function of
    /// `order.len()` alone and only the leaf labels follow `order`. For a
    /// power-of-two length the tree is perfectly symmetric; otherwise the
    /// right half of an odd split carries one more variable.
    ///
    /// The id space is `max(order)`; ids skipped by `order` are uncovered.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if `order` is empty;
    /// [`VtreeError::OverlappingVariable`] if it repeats a variable.
    pub fn balanced_over(order: &[VarId]) -> Result<Self, VtreeError> {
        let num_vars = id_space(order)?;
        let mut nodes = Vec::with_capacity(2 * order.len() - 1);
        let root = Self::build_balanced_recursive(order, &mut nodes);
        Self::from_nodes(nodes, root, num_vars)
    }

    /// Recursively build a balanced vtree over `vars`, appending nodes into
    /// `nodes` and returning the index of the constructed subtree's root.
    /// Parents are left unset; [`Vtree::from_nodes`] derives them.
    ///
    /// # Panics
    ///
    /// Panics if `vars` is empty.
    pub fn build_balanced_recursive(vars: &[VarId], nodes: &mut Vec<VtreeNode>) -> VtreeIdx {
        assert!(!vars.is_empty(), "a balanced subtree needs at least one variable");
        if vars.len() == 1 {
            return push_leaf(nodes, vars[0]);
        }
        let mid = vars.len() / 2;
        let left = Self::build_balanced_recursive(&vars[..mid], nodes);
        let right = Self::build_balanced_recursive(&vars[mid..], nodes);
        push_internal(nodes, left, right)
    }

    /// A right-linear vtree over `1..=num_vars` in ascending order:
    /// [`Vtree::linear_from_order`] on `1, …, n`, so variable `1` is the
    /// root's left leaf and variable `n` sits deepest.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn linear(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (1..=num_vars).map(VarId).collect();
        Self::linear_from_order(&vars).expect("the ids are distinct")
    }

    /// A right-linear vtree over `num_vars, …, 1`, reversing [`Vtree::linear`]'s
    /// order: variable `num_vars` is the root's left leaf and `1` sits deepest.
    ///
    /// # Panics
    ///
    /// Panics if `num_vars` is zero.
    pub fn reverse_linear(num_vars: u32) -> Self {
        require_nonempty(num_vars);
        let vars: Vec<VarId> = (1..=num_vars).rev().map(VarId).collect();
        Self::linear_from_order(&vars).expect("the ids are distinct")
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
    /// The id space is `max(vars)`; ids skipped by `vars` are uncovered.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if `vars` is empty;
    /// [`VtreeError::OverlappingVariable`] if it repeats a variable.
    pub fn linear_from_order(vars: &[VarId]) -> Result<Self, VtreeError> {
        let num_vars = id_space(vars)?;
        let mut nodes = Vec::with_capacity(2 * vars.len() - 1);
        let (&last, rest) = vars.split_last().expect("checked nonempty");
        let mut right = push_leaf(&mut nodes, last);
        for &var in rest.iter().rev() {
            let left = push_leaf(&mut nodes, var);
            right = push_internal(&mut nodes, left, right);
        }
        Self::from_nodes(nodes, right, num_vars)
    }

    /// Build a random vtree over `num_vars` variables (`1..=num_vars`).
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
        let mut var_ids: Vec<u32> = (1..=num_vars).collect();
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

    /// Validate the child links and variables, then reindex bottom-up,
    /// returning the tree and the `old_to_new` permutation a caller translates
    /// pre-reindex `VtreeIdx` values through.
    ///
    /// Leaves take the indices `0..num_leaves`, then the internal nodes in
    /// bottom-up level order, left to right within a level, so the identity
    /// order is a valid initial [`TopoOrder`](crate::vtree::topo::TopoOrder).
    pub(super) fn from_nodes_with_map(
        old_nodes: Vec<VtreeNode>,
        root: VtreeIdx,
        num_vars: u32,
    ) -> Result<(Self, Vec<VtreeIdx>), VtreeError> {
        // The walk from the root checks the list as it goes; where it
        // refuses one, the full check, which reads the whole list, names what
        // is wrong. Both refuse the same lists.
        let n = old_nodes.len();
        let shape = |t: VtreeIdx| match old_nodes[t.idx()] {
            VtreeNode::Leaf { var, .. } => Shape::Leaf(var),
            VtreeNode::Internal { left, right, .. } => Shape::Internal(left, right),
        };
        let reindexed = if n > 0 && root.idx() < n && check_var_space(num_vars, n).is_ok() {
            reindex(root, n, num_vars, shape)
        } else {
            None
        };
        let Some(reindexed) = reindexed else {
            check_node_list(&old_nodes, root, num_vars)?;
            return Err(VtreeError::Invalid("the links are not one tree".to_string()));
        };
        Ok(reindexed)
    }

    /// Build a vtree with leaves `vars` in left-to-right order, using the
    /// depths of adjacent leaves' lowest common ancestors to choose its shape.
    /// There must be one depth per adjacent pair: `depths.len() == vars.len() - 1`.
    ///
    /// The smallest depth chooses the root split; ties choose the leftmost
    /// split. Each side is built by the same rule. Only depth comparisons
    /// matter, so any depth values are accepted; equal depths produce
    /// [`Vtree::linear_from_order`]'s right-linear tree.
    ///
    /// To project an existing tree onto selected leaves, pass those leaves
    /// in left-to-right order and their adjacent lowest-common-ancestor
    /// depths in the original tree. `num_vars` sizes the variable id space,
    /// as for [`Vtree::from_nodes`].
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if `vars` is empty, if `depths` does not hold
    /// one depth fewer than `vars` holds leaves, or if a variable is outside
    /// `1..=num_vars`; [`VtreeError::OverlappingVariable`] if two leaves
    /// carry one variable; [`VtreeError::VariableSpaceTooLarge`] as for
    /// [`Vtree::from_nodes`].
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    ///
    /// // Leaves 1 and 2 meet below the root, which joins them to leaf 3.
    /// let vars = [VarId(1), VarId(2), VarId(3)];
    /// let vtree = Vtree::from_in_order(&vars, &[4, 2], 3)?;
    /// let (left, right) = vtree.children(vtree.root());
    /// assert_eq!(vtree.leaf_var(right), VarId(3));
    /// let (a, b) = vtree.children(left);
    /// assert_eq!((vtree.leaf_var(a), vtree.leaf_var(b)), (VarId(1), VarId(2)));
    ///
    /// // A depth for each pair of neighbours, no more and no fewer.
    /// assert!(matches!(Vtree::from_in_order(&vars, &[1], 3), Err(VtreeError::Invalid(_))));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn from_in_order(vars: &[VarId], depths: &[u32], num_vars: u32) -> Result<Self, VtreeError> {
        let k = vars.len();
        if k == 0 {
            return Err(VtreeError::Invalid("a vtree needs at least one variable".to_string()));
        }
        if depths.len() + 1 != k {
            return Err(VtreeError::Invalid(format!(
                "{k} leaves take {} depths, not {}",
                k - 1,
                depths.len()
            )));
        }
        let n = 2 * k - 1;
        check_var_space(num_vars, n)?;
        // The nodes in left-to-right order: leaf i is node 2i, the node
        // between leaves i and i + 1 is node 2i + 1, and that node's children
        // are kids[2i] and kids[2i + 1]. The nodes whose right side is still
        // open are kept on `open`, depths ascending from the root; a node
        // takes as its left side those of them deeper than it.
        let mut kids = vec![VtreeIdx(0); n - 1];
        let mut open: Vec<usize> = Vec::with_capacity(k);
        for (i, &depth) in depths.iter().enumerate() {
            let mut side = VtreeIdx(2 * i as u32);
            while let Some(&top) = open.last() {
                if depths[top] <= depth {
                    break;
                }
                open.pop();
                kids[2 * top + 1] = side;
                side = VtreeIdx(2 * top as u32 + 1);
            }
            kids[2 * i] = side;
            open.push(i);
        }
        let mut root = VtreeIdx(2 * (k - 1) as u32);
        while let Some(top) = open.pop() {
            kids[2 * top + 1] = root;
            root = VtreeIdx(2 * top as u32 + 1);
        }
        let shape = |t: VtreeIdx| {
            let t = t.idx();
            if t.is_multiple_of(2) { Shape::Leaf(vars[t / 2]) } else { Shape::Internal(kids[t - 1], kids[t]) }
        };
        match reindex(root, n, num_vars, shape) {
            Some((vtree, _)) => Ok(vtree),
            None => {
                let mut variables = vec![false; num_vars as usize];
                for &var in vars {
                    check_leaf_var(var, &mut variables)?;
                }
                unreachable!("a tree built from depths is one tree, so only its variables can be refused")
            }
        }
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
    /// Construction validates the child links and variables; every returned
    /// tree passes [`Vtree::validate`].
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] if `nodes` is empty, if an index names no node,
    /// if a leaf carries a variable outside `1..=num_vars`, or if the links do
    /// not reach every node exactly once from `root`;
    /// [`VtreeError::OverlappingVariable`] if two leaves carry one variable;
    /// [`VtreeError::VariableSpaceTooLarge`] if `num_vars` is wider than the
    /// node list can justify a table indexed by variable id being.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};
    ///
    /// let nodes = vec![
    ///     VtreeNode::Leaf { var: VarId(1), parent: None },
    ///     VtreeNode::Leaf { var: VarId(2), parent: None },
    ///     VtreeNode::Internal { left: VtreeIdx(0), right: VtreeIdx(1), parent: None },
    /// ];
    /// let vtree = Vtree::from_nodes(nodes, VtreeIdx(2), 2)?;
    /// assert_eq!((vtree.num_leaves(), vtree.num_vars()), (2, 2));
    ///
    /// // A lone leaf the root cannot reach makes the list something other
    /// // than one tree.
    /// let stray = vec![
    ///     VtreeNode::Leaf { var: VarId(1), parent: None },
    ///     VtreeNode::Leaf { var: VarId(2), parent: None },
    /// ];
    /// assert!(matches!(
    ///     Vtree::from_nodes(stray, VtreeIdx(0), 2),
    ///     Err(VtreeError::Invalid(_)),
    /// ));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn from_nodes(
        nodes: Vec<VtreeNode>,
        root: VtreeIdx,
        num_vars: u32,
    ) -> Result<Self, VtreeError> {
        Self::from_nodes_with_map(nodes, root, num_vars).map(|(vtree, _)| vtree)
    }
}

/// Diagnose a node list rejected by the combined validation and reindexing
/// walk. The full scan preserves error precedence: variable and child-index
/// errors in list order, then repeated or unreachable nodes.
fn check_node_list(nodes: &[VtreeNode], root: VtreeIdx, num_vars: u32) -> Result<(), VtreeError> {
    let n = nodes.len();
    if n == 0 {
        return Err(VtreeError::Invalid("a vtree needs at least one node".to_string()));
    }
    if root.idx() >= n {
        return Err(VtreeError::Invalid(format!("root {} is not a node index", root.0)));
    }
    check_var_space(num_vars, n)?;
    let mut variables = vec![false; num_vars as usize];
    for node in nodes {
        match *node {
            VtreeNode::Leaf { var, .. } => check_leaf_var(var, &mut variables)?,
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
    // The stack holds fewer nodes than the list: sized once, it never grows.
    let mut stack = Vec::with_capacity(n);
    stack.push(root);
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
    Ok(())
}

/// Refuse a leaf's variable outside the id space `1..=variables.len()`, or
/// one an earlier leaf carries; `variables` marks the variables seen.
fn check_leaf_var(var: VarId, variables: &mut [bool]) -> Result<(), VtreeError> {
    let num_vars = variables.len();
    if var.0 == 0 || var.0 as usize > num_vars {
        return Err(VtreeError::Invalid(format!(
            "leaf variable {} is outside the variables 1 to {num_vars}",
            var.0
        )));
    }
    if std::mem::replace(&mut variables[var.idx()], true) {
        return Err(VtreeError::OverlappingVariable(var));
    }
    Ok(())
}

/// An entry of `var_to_leaf` or `old_to_new` the walk from the root has
/// reached, until the reindex writes it.
const REACHED: VtreeIdx = VtreeIdx(u32::MAX);

/// A node as the reindex reads it, from whatever list or rule describes the
/// tree: a leaf's variable, or an internal node's two children.
#[derive(Clone, Copy)]
enum Shape {
    Leaf(VarId),
    Internal(VtreeIdx, VtreeIdx),
}

/// The tree of `n` nodes that `shape` describes from `root`, reindexed
/// bottom-up (see [`Vtree::from_nodes_with_map`]), with the `old_to_new`
/// permutation; `None` where it is not one tree whose leaves carry distinct
/// variables in `1..=num_vars`. `root` names one of the `n` nodes, and
/// `num_vars` has passed [`check_var_space`].
fn reindex(root: VtreeIdx, n: usize, num_vars: u32, shape: impl Fn(VtreeIdx) -> Shape) -> Option<(Vtree, Vec<VtreeIdx>)> {
    let mut var_to_leaf = vec![VtreeIdx(0); num_vars as usize];
    let mut old_to_new = vec![VtreeIdx(0); n];
    let (order, starts) = levels_from_root(root, n, &shape, (&mut var_to_leaf, &mut old_to_new))?;
    let (nodes, leaf_count) = relabel_leaves_then_internals((&order, &starts), n, &shape, (&mut var_to_leaf, &mut old_to_new));
    // After the reindex, the node array is laid out so that idx ==
    // bottom-up topological position, so the identity order is correct.
    let topo = crate::vtree::topo::TopoOrder::identity(&nodes, leaf_count);
    let vtree = Vtree {
        context: std::sync::Arc::new(crate::Context::new()),
        nodes,
        root: old_to_new[root.idx()],
        var_to_leaf,
        leaf_count,
        topo,
    };
    debug_assert_eq!(vtree.validate(), Ok(()));
    Some((vtree, old_to_new))
}

/// The nodes reachable from `root` in breadth-first order from the root, and
/// where each depth starts in that order (one entry per depth, then the end).
/// The order is its own queue: a level is the run of nodes the level before
/// it appended, so the walk allocates two lists however deep the tree is.
///
/// The walk is also the check of the `n` nodes `shape` describes, `root` one
/// of them: `None` unless every child link names one, the links reach every
/// node exactly once, and every leaf carries its own variable in
/// `1..=num_vars`, where `num_vars` is `var_to_leaf`'s length and `n` is
/// `old_to_new`'s. Both come in zeroed; the walk marks what it reaches with
/// [`REACHED`], and the reindex writes every entry so marked.
fn levels_from_root(
    root: VtreeIdx,
    n: usize,
    shape: impl Fn(VtreeIdx) -> Shape,
    (var_to_leaf, old_to_new): (&mut [VtreeIdx], &mut [VtreeIdx]),
) -> Option<(Vec<VtreeIdx>, Vec<usize>)> {
    let mut order = Vec::with_capacity(n);
    // A start per depth and the end: a binary tree of `n` nodes is at most
    // `(n + 1) / 2` levels deep, one internal node and one leaf per level
    // but the last.
    let mut starts = Vec::with_capacity(n.div_ceil(2) + 1);
    starts.push(0);
    order.push(root);
    old_to_new[root.idx()] = REACHED;
    let mut at = 0;
    while at < order.len() {
        let end = order.len();
        while at < end {
            match shape(order[at]) {
                Shape::Internal(left, right) => {
                    for child in [left, right] {
                        let entry = old_to_new.get_mut(child.idx())?;
                        if std::mem::replace(entry, REACHED) == REACHED {
                            return None;
                        }
                        order.push(child);
                    }
                }
                Shape::Leaf(var) => {
                    let entry = var_to_leaf.get_mut((var.0 as usize).checked_sub(1)?)?;
                    if std::mem::replace(entry, REACHED) == REACHED {
                        return None;
                    }
                }
            }
            at += 1;
        }
        starts.push(end);
    }
    (order.len() == n).then_some((order, starts))
}

/// Rebuild the node list with the leaves first, then the internal nodes,
/// each group in bottom-up level order and left to right within a level.
/// This puts leaves at `0..num_leaves` and internals above them while
/// preserving `child.idx() < parent.idx()` for every edge. Returns the new
/// nodes and the leaf count, and writes the `old_to_new` permutation and the
/// leaves' entries of `var_to_leaf`.
///
/// The tree is binary over every node, so it has one leaf more than it has
/// internal nodes, and one walk up the levels places both groups: a node's
/// children sit a level below it and are placed before it.
fn relabel_leaves_then_internals(
    (order, starts): (&[VtreeIdx], &[usize]),
    n: usize,
    shape: impl Fn(VtreeIdx) -> Shape,
    (var_to_leaf, old_to_new): (&mut [VtreeIdx], &mut [VtreeIdx]),
) -> (Vec<VtreeNode>, u32) {
    let num_leaves = n.div_ceil(2) as u32;
    let mut new_nodes = vec![VtreeNode::Leaf { var: VarId(0), parent: None }; n];
    let (mut next_leaf, mut next_internal) = (0, num_leaves);
    for level in starts.windows(2).rev().map(|w| &order[w[0]..w[1]]) {
        for &old_idx in level {
            match shape(old_idx) {
                Shape::Leaf(var) => {
                    let new_idx = VtreeIdx(next_leaf);
                    next_leaf += 1;
                    old_to_new[old_idx.idx()] = new_idx;
                    new_nodes[new_idx.idx()] = VtreeNode::Leaf { var, parent: None };
                    var_to_leaf[var.idx()] = new_idx;
                }
                Shape::Internal(left, right) => {
                    let (new_left, new_right) = (old_to_new[left.idx()], old_to_new[right.idx()]);
                    let new_idx = VtreeIdx(next_internal);
                    next_internal += 1;
                    old_to_new[old_idx.idx()] = new_idx;
                    new_nodes[new_idx.idx()] = VtreeNode::Internal {
                        left: new_left,
                        right: new_right,
                        parent: None,
                    };
                    set_parent(&mut new_nodes, new_left, new_idx);
                    set_parent(&mut new_nodes, new_right, new_idx);
                }
            }
        }
    }
    debug_assert_eq!((next_leaf, next_internal as usize), (num_leaves, n));
    (new_nodes, num_leaves)
}
