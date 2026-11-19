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
    /// children before parents (so there is no cycle), variable leaves first,
    /// then regions, then internal nodes; every variable sits on at most one leaf, within the id
    /// space, with [`Vtree::leaf_of`] pointing at it;
    /// [`Vtree::num_leaves`] counts the leaves; and on a projection, each
    /// node stands for the node of [`Vtree::home`] the projection defines,
    /// with the indices the projection's builder gives. A vtree with no node
    /// holds only as a projection on no leaf.
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
            return match self.home.as_deref() {
                Some(link) if link.ranks.is_empty() && link.node_of.is_empty() && link.in_order.is_empty() => Ok(()),
                _ => invalid("no nodes".to_string()),
            };
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
    /// a leaf count that matches [`Vtree::num_leaves`]; regions are not
    /// leaves here.
    fn validate_leaves(&self) -> Result<(), VtreeError> {
        let mut leaf_count = 0u32;
        let num_vars = self.num_vars();
        let table = if self.link_home().is_some() { 0 } else { num_vars as usize };
        if self.var_to_leaf.len() != table {
            return invalid(format!("the leaf table has {} entries for an id space of {num_vars}", self.var_to_leaf.len()));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if let VtreeNode::Leaf { var, .. } = node {
                if *var == super::REGION_VAR {
                    continue;
                }
                leaf_count += 1;
                if var.0 == 0 || var.0 > num_vars {
                    return invalid(format!("leaf {i} carries variable {} outside the id space {num_vars}", var.0));
                }
                // A variable on two leaves fails here on one of them.
                if self.leaf_of(*var) != Some(VtreeIdx(i as u32)) {
                    return invalid(format!("leaf_of({}) does not point at leaf {i}", var.0));
                }
            }
        }
        if leaf_count != self.num_leaves() {
            return invalid(format!("num_leaves() is {} but {leaf_count} leaves exist", self.num_leaves()));
        }
        self.validate_home()
    }
}
