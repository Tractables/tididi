//! Read-only structural query: the literals forced true in every model.


use crate::diagram::{EncodedChildRef, Literal, Tdd};
use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};
use crate::OperationError;



/// Implied literals (the backbone) of `f`: every literal that holds in every
/// model of `f`, sorted by variable. One O(size) pass over the pairs, no
/// conditioning and no counting.
///
/// Requires `f` minimized: the pass reads which leaf labels each variable's
/// leaf is referenced with, and an unreachable reference on an unminimized
/// diagram would count as a label. The zero diagram and summed-out variables
/// contribute nothing, and a variable no pair references is not implied. No
/// engine and no limit are involved. [`Engine::implied_literals`](crate::Engine::implied_literals)
/// accepts an unminimized structural diagram and honors engine limits.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::query::implied_literals;
/// use tididi::reduce::minimize;
///
/// let tree = Arc::new(Vtree::balanced(3));
/// let mut f = Tdd::clause(&tree, [1, 2]) & Tdd::clause(&tree, [1, -2]);
/// minimize(&mut f);
/// assert_eq!(implied_literals(&f), vec![1.into()]); // x1 is forced; x2 and x3 are free
/// # tididi::test_helpers::assert_canonical(&f);
/// ```
#[must_use]
pub fn implied_literals(f: &Tdd) -> Vec<Literal> {
    let mut out = Vec::new();
    visit_leaf_labels(f, |_| Ok(()), |var, labels| {
        if let Some(literal) = labels.implied(var) {
            out.push(literal);
        }
        Ok(())
    }).expect("unlimited leaf-label scan");
    out.sort_unstable_by_key(|lit| lit.var.0);
    out
}


/// Referenced positive, negative and free labels of one non-marginal leaf.
#[derive(Clone, Copy, Default)]
pub(super) struct LeafLabels(u8);

impl LeafLabels {
    /// Include a referenced label; zero sentinels contribute nothing.
    fn insert(&mut self, child: EncodedChildRef) {
        self.0 |= if child == POS_LEAF_IDX.into() { 1 }
            else if child == NEG_LEAF_IDX.into() { 2 }
            else if child == ONE_LEAF_IDX.into() { 4 }
            else { 0 };
    }

    /// Whether a positive or negative label makes this variable part of the support.
    pub(super) fn depends(self) -> bool { self.0 & 3 != 0 }

    /// The forced literal, if every reference uses the same non-free label.
    pub(super) fn implied(self, var: VarId) -> Option<Literal> {
        match self.0 {
            1 => Some(Literal::pos(var)),
            2 => Some(Literal::neg(var)),
            _ => None,
        }
    }
}

/// Visit each referenced structural leaf's label summary on a minimized diagram.
///
/// Each leaf has one parent, so scanning that parent's pairs finishes its summary.
/// `poll` receives one work unit per leaf reference, within the pair loop.
pub(super) fn visit_leaf_labels(
    f: &Tdd,
    mut poll: impl FnMut(u64) -> Result<(), OperationError>,
    mut visit: impl FnMut(VarId, LeafLabels) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    if f.is_zero() { return Ok(()); }
    if let VtreeNode::Leaf { var, .. } = *f.vtree.node(f.output.vtree) {
        if !f.levels[f.output.vtree.idx()].is_marginal() {
            poll(1)?;
            let mut labels = LeafLabels::default();
            labels.insert(f.output.local.into());
            visit(var, labels)?;
        }
        return Ok(());
    }
    let leaf_var = |child: VtreeIdx| match *f.vtree.node(child) {
        VtreeNode::Leaf { var, .. } if !f.levels[child.idx()].is_marginal() => Some(var),
        _ => None,
    };
    for (t, left, right) in f.vtree.internal_bottomup() {
        let vars = [leaf_var(left), leaf_var(right)];
        let work = vars.iter().filter(|var| var.is_some()).count() as u64;
        if work == 0 { continue; }
        let mut labels = [LeafLabels::default(); 2];
        let level = &f.levels[t.idx()];
        for node in level.nodes.iter().filter(|node| !node.is_leaf()) {
            for pair in level.pairs_of(node) {
                poll(work)?;
                if vars[0].is_some() { labels[0].insert(pair.left); }
                if vars[1].is_some() { labels[1].insert(pair.right); }
            }
        }
        for (var, labels) in vars.into_iter().zip(labels) {
            if let Some(var) = var && labels.0 != 0 {
                visit(var, labels)?;
            }
        }
    }
    Ok(())
}
