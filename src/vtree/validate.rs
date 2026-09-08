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
    /// `O(nodes)`. Every constructor that combines caller-supplied trees runs
    /// this in debug builds; call it yourself after hand-built input.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] naming the first invariant found broken.
    pub fn validate(&self) -> Result<(), VtreeError> {
        let n = self.nodes.len();
        if n == 0 {
            return invalid("no nodes".to_string());
        }
        if self.root.idx() >= n {
            return invalid(format!("root {} is not a node index", self.root.0));
        }
        self.validate_links(n)?;
        self.validate_bottomup_order(n)?;
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

    /// The bottom-up order lists every node once, children before parents, with
    /// consistent inverse positions and filtered views.
    fn validate_bottomup_order(&self, n: usize) -> Result<(), VtreeError> {
        if !self.topo.covers(n) {
            return invalid("bottom-up order does not cover the node list".to_string());
        }
        let mut seen = vec![false; n];
        for (pos, &t) in self.topo.all().iter().enumerate() {
            if t.idx() >= n || std::mem::replace(&mut seen[t.idx()], true) {
                return invalid(format!("bottom-up order lists node {} twice or out of range", t.0));
            }
            if self.topo.pos(t) as usize != pos {
                return invalid(format!("bottom-up position of node {} is inconsistent", t.0));
            }
            if let VtreeNode::Internal { left, right, .. } = &self.nodes[t.idx()] {
                if !seen[left.idx()] || !seen[right.idx()] {
                    return invalid(format!("node {} precedes one of its children in the bottom-up order", t.0));
                }
            }
        }
        let leaves_in_order = self.topo.all().iter().filter(|t| self.nodes[t.idx()].is_leaf()).count();
        if self.topo.leaves().len() != leaves_in_order
            || self.topo.internal().len() != n - leaves_in_order
            || !self.topo.leaves().iter().all(|t| self.nodes[t.idx()].is_leaf())
            || self.topo.internal().iter().any(|t| self.nodes[t.idx()].is_leaf())
        {
            return invalid("leaf/internal views disagree with the bottom-up order".to_string());
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
