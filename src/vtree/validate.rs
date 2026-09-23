//! The representation-invariant check.

use super::{Vtree, VtreeError, VtreeIdx, VtreeNode};

/// The single shape every invariant failure is reported in.
fn invalid<T>(msg: String) -> Result<T, VtreeError> {
    Err(VtreeError::Invalid(msg))
}

impl Vtree {
    /// Check the representation invariants: exactly one parentless node and it
    /// is the root; every child's parent link names its parent and no node is
    /// the child of two nodes; the bottom-up order covers every node once with
    /// children before parents (so there is no cycle) and its filtered views
    /// agree with it; every variable sits on at most one leaf, within the id
    /// space, with [`Vtree::leaf_of`] pointing at it; and
    /// [`Vtree::num_leaves`] counts the leaves.
    ///
    /// `O(nodes)`. Every constructor runs it in debug builds, and
    /// [`Vtree::from_nodes`] refuses a node list it would reject, so a vtree in
    /// hand already holds. It stays public as the statement of what "holds"
    /// means.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] naming the first invariant found broken.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};
    ///
    /// assert_eq!(Vtree::balanced(4).validate(), Ok(()));
    ///
    /// // Hand-built child links, with the same variable on both leaves: the
    /// // constructor refuses them, so no vtree carries them.
    /// let nodes = vec![
    ///     VtreeNode::Leaf { var: VarId(1), parent: None },
    ///     VtreeNode::Leaf { var: VarId(1), parent: None },
    ///     VtreeNode::Internal { left: VtreeIdx(0), right: VtreeIdx(1), parent: None },
    /// ];
    /// match Vtree::from_nodes(nodes, VtreeIdx(2), 1) {
    ///     Ok(_) => unreachable!("the duplicate variable should be caught"),
    ///     Err(VtreeError::OverlappingVariable(VarId(1))) => {}
    ///     Err(other) => unreachable!("{other}"),
    /// }
    /// ```
    pub fn validate(&self) -> Result<(), VtreeError> {
        let n = self.nodes.len();
        if n == 0 {
            return invalid("no nodes".to_string());
        }
        if self.root.idx() >= n {
            return invalid(format!("root {} is not a node index", self.root.0));
        }
        self.validate_links(n)?;
        self.topo.validate(&self.nodes).map_err(VtreeError::Invalid)?;
        self.validate_leaves()
    }

    /// One parentless node (the root), every child slot naming a node whose
    /// parent link points back, and no node claimed as a child twice.
    fn validate_links(&self, n: usize) -> Result<(), VtreeError> {
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
        Ok(())
    }

    /// Distinct variables inside the id space, each inverted by `leaf_of`, and
    /// a leaf count that matches [`Vtree::num_leaves`].
    fn validate_leaves(&self) -> Result<(), VtreeError> {
        let mut leaf_count = 0u32;
        let mut carried = vec![false; self.var_to_leaf.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            if let VtreeNode::Leaf { var, .. } = node {
                leaf_count += 1;
                if var.0 == 0 || var.idx() >= carried.len() {
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
