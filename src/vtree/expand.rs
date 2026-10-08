//! Replacing each leaf of a vtree with a subtree over a class of variables.

use super::build::push_internal;
use super::{VarId, Vtree, VtreeError, VtreeIdx};

impl Vtree {
    /// This vtree with each leaf replaced by a balanced subtree over the
    /// variables `class_of` names for the leaf's variable.
    ///
    /// Each class is arranged as [`Vtree::balanced_over`] arranges its order,
    /// under the node that held the leaf, and the nodes above the leaves keep
    /// their shape. A class of one variable renames the leaf. `class_of` is
    /// called once per leaf, bottom-up. The result has the id space
    /// `1..=num_vars` and this vtree's execution context; building it is
    /// linear in its size.
    ///
    /// [`Tdd::expand_variables`](crate::Tdd::expand_variables) places a
    /// diagram's expansion on this vtree.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    ///
    /// // Variable 1 stands for 1 and 2, variable 2 for 3.
    /// let classes = [vec![VarId(1), VarId(2)], vec![VarId(3)]];
    /// let vtree = Vtree::balanced(2).expand_leaves(|v| classes[v.idx()].clone(), 3)?;
    /// assert_eq!((vtree.num_leaves(), vtree.num_vars()), (3, 3));
    /// let (one, two) = (vtree.leaf_of(VarId(1)).unwrap(), vtree.leaf_of(VarId(2)).unwrap());
    /// assert_eq!(vtree.node(one).parent(), vtree.node(two).parent());
    ///
    /// // A class may not repeat a variable another class holds.
    /// let clash = Vtree::balanced(2).expand_leaves(|v| [v, VarId(2)], 2);
    /// assert!(matches!(clash, Err(VtreeError::OverlappingVariable(VarId(2)))));
    /// # Ok::<(), VtreeError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`VtreeError::Invalid`] for an empty class or a variable outside
    /// `1..=num_vars`; [`VtreeError::OverlappingVariable`] for a variable named
    /// twice; [`VtreeError::VariableSpaceTooLarge`] if `num_vars` is wider
    /// than the result's node list allows, as for [`Vtree::from_nodes`].
    pub fn expand_leaves<I>(&self, mut class_of: impl FnMut(VarId) -> I, num_vars: u32) -> Result<Vtree, VtreeError>
    where
        I: IntoIterator<Item = VarId>,
    {
        let mut nodes = Vec::new();
        let mut new_of: Vec<Option<VtreeIdx>> = vec![None; self.num_nodes()];
        let mut class = Vec::new();
        for (leaf, var) in self.leaf_bottomup() {
            class.clear();
            class.extend(class_of(var));
            if class.is_empty() {
                return Err(VtreeError::Invalid(format!("variable {} has an empty class", var.0)));
            }
            new_of[leaf.idx()] = Some(Vtree::build_balanced_recursive(&class, &mut nodes));
        }
        for (t, left, right) in self.internal_bottomup() {
            let left = new_of[left.idx()].expect("a child is rebuilt before its parent");
            let right = new_of[right.idx()].expect("a child is rebuilt before its parent");
            new_of[t.idx()] = Some(push_internal(&mut nodes, left, right));
        }
        let root = new_of[self.root().idx()].expect("the root is rebuilt");
        Ok(Vtree::from_nodes(nodes, root, num_vars)?.with_context(std::sync::Arc::clone(self.context())))
    }
}

#[cfg(test)]
#[path = "tests/expand.rs"]
mod tests;
