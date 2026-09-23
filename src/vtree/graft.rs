//! Joining independent subtrees and spine variables under one left-linear spine.

use super::build::{append_subtree, push_internal, push_leaf};
use super::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};

/// Where each subtree of a [`Vtree::graft_over`] landed in the finished vtree,
/// so a caller can translate a node index of a piece into one of the result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GraftLayout {
    /// `comp_to_full[k][c]` = final `VtreeIdx` of subtree `k`'s own node `c`
    /// (indexed as in that subtree's node array, `0..num_nodes()`).
    pub comp_to_full: Vec<Vec<VtreeIdx>>,
    /// Final `VtreeIdx` of each spine join, in build order. Pieces are
    /// `[subtree 0, …, subtree k-1, spine_var 0 leaf, …, spine_var m-1 leaf]`
    /// and `chain_internals[j]` is the join whose left child is the running
    /// chain root and whose right child is piece `j+1`; empty for a single
    /// piece.
    pub(crate) chain_internals: Vec<VtreeIdx>,
}

impl Vtree {
    /// Join independent subtrees and single-variable leaves under one
    /// left-linear spine: `subtrees[0]` is the leftmost piece, each later
    /// subtree and then each `spine_vars` leaf is attached to a new root on
    /// the right, in the order given. A diagram over the result is what
    /// [`crate::Tdd::graft`] builds from diagrams over the pieces.
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
    /// A context shared by all subtrees is retained; otherwise the result gets
    /// a fresh context.
    ///
    /// # Errors
    ///
    /// [`VtreeError::OverlappingVariable`] if two pieces carry the same
    /// variable; [`VtreeError::Invalid`] if there is no piece or a spine variable
    /// is zero; [`VtreeError::VariableSpaceTooLarge`] if the id space is too large.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    /// let parts = [Vtree::balanced_over(&[VarId(1), VarId(2)])?, Vtree::leaf(VarId(4))];
    /// let vtree = Vtree::graft(&parts, &[VarId(3)])?;
    /// assert_eq!((vtree.num_leaves(), vtree.num_vars()), (4, 4));
    ///
    /// // A spine variable one of the pieces already carries is refused.
    /// let clash = Vtree::graft(&parts, &[VarId(2)]);
    /// assert!(matches!(clash, Err(VtreeError::OverlappingVariable(VarId(2)))));
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn graft(subtrees: &[Vtree], spine_vars: &[VarId]) -> Result<Self, VtreeError> {
        let num_vars = subtrees
            .iter()
            .map(Vtree::num_vars)
            .chain(spine_vars.iter().map(|v| v.0))
            .max()
            .unwrap_or(0);
        let refs: Vec<&Vtree> = subtrees.iter().collect();
        Self::graft_over(&refs, |_, v| v, spine_vars, num_vars).map(|(vtree, _)| vtree)
    }

    /// [`Vtree::graft`] with each subtree's leaves renamed through
    /// `rename(k, local)` on the way in, an explicit id space (which must hold
    /// every renamed id), and the [`GraftLayout`] the diagram-side graft places
    /// levels by. Use it when the pieces have separate variable id spaces.
    ///
    /// # Errors
    ///
    /// As [`Vtree::graft`], over the renamed ids: [`VtreeError::OverlappingVariable`]
    /// if two pieces land on the same variable, [`VtreeError::Invalid`] if
    /// there is nothing to graft or a variable is outside `1..=num_vars`;
    /// [`VtreeError::VariableSpaceTooLarge`] if the id space is too large.
    ///
    /// ```
    /// use tididi::vtree::{VarId, Vtree, VtreeError};
    ///
    /// let a = Vtree::balanced_over(&[VarId(1), VarId(2)])?;
    /// let b = Vtree::balanced_over(&[VarId(1), VarId(2)])?;
    /// // Each piece is compiled in its own id space, so piece 1 is shifted up.
    /// let shift = |k: usize, v: VarId| VarId(v.0 + 2 * k as u32);
    /// let (vtree, layout) = Vtree::graft_over(&[&a, &b], shift, &[VarId(5)], 5)?;
    /// assert_eq!(vtree.num_leaves(), 5);
    /// assert_eq!(layout.comp_to_full.len(), 2);
    ///
    /// // The same call without the rename lands both pieces on the same ids.
    /// match Vtree::graft_over(&[&a, &b], |_, v| v, &[VarId(5)], 5) {
    ///     Ok(_) => unreachable!(),
    ///     Err(VtreeError::OverlappingVariable(_)) => {}
    ///     Err(other) => unreachable!("{other}"),
    /// }
    /// # Ok::<(), tididi::vtree::VtreeError>(())
    /// ```
    pub fn graft_over(
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

        let (mut vtree, old_to_new) = Self::from_nodes_with_map(nodes, root, num_vars)?;
        if let Some(first) = subtrees.first()
            && subtrees.iter().all(|sub| std::sync::Arc::ptr_eq(sub.context(), first.context()))
        {
            vtree.context = std::sync::Arc::clone(first.context());
        }

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
}
