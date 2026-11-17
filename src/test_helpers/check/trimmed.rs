//! The trimmed form ([`Tdd::trim`]): no pair names a leaf as `One`.

use crate::diagram::{Tdd, ONE_LEAF_IDX};
use crate::vtree::VtreeNode;

/// No pair of `tdd` names a leaf as `One`, nor does the output; a constant
/// has no leaf. With `minimized`, also that the leaves are the
/// [`support`](Tdd::support).
///
/// Cost: O(diagram size).
pub fn check_trimmed(tdd: &Tdd, minimized: bool) -> Result<(), String> {
    let vtree = tdd.vtree();
    if (tdd.is_zero() || tdd.is_one()) != (vtree.num_nodes() == 0) {
        return Err(format!("a constant on {} vtree nodes, or a diagram on none", vtree.num_nodes()));
    }
    if vtree.num_nodes() == 0 {
        return Ok(());
    }
    if vtree.node(vtree.root()).is_leaf() && tdd.output().local == ONE_LEAF_IDX {
        return Err("the output names its leaf as One".to_string());
    }
    for t in vtree.bottomup() {
        let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
        let level = tdd.level(t);
        for j in 0..level.slot_count() {
            for pair in level.pairs_iter_of_idx(j) {
                for (child, side) in [(left, pair.left), (right, pair.right)] {
                    if vtree.node(child).is_leaf() && side.raw() == ONE_LEAF_IDX.0 {
                        return Err(format!("node {j} of level {} names leaf {} as One", t.0, child.0));
                    }
                }
            }
        }
    }
    if minimized {
        let mut leaves: Vec<_> = vtree.bottomup().filter(|&t| vtree.node(t).is_leaf()).map(|t| vtree.leaf_var(t)).collect();
        leaves.sort_unstable();
        let mut support = tdd.support().map_err(|e| e.to_string())?;
        support.sort_unstable();
        if leaves != support {
            return Err(format!("leaves {leaves:?} are not the support {support:?}"));
        }
    }
    Ok(())
}
