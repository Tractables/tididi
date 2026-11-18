//! The trimmed form ([`Tdd::trim`]): no pair names a leaf as `One`.

use crate::diagram::{Tdd, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::vtree::VtreeNode;

/// The leaf sides of the reachable pairs of `tdd`, summed leaves and summed
/// levels skipped: `Err` when some level names a leaf child `One` in one pair
/// and not in another, else the first leaf named `One` (or a one-leaf output
/// that is `One`), if any. Dead slots are not read: they may hold stale pairs.
pub fn reachable_true_labels(tdd: &Tdd) -> Result<Option<String>, String> {
    let vtree = tdd.vtree();
    if vtree.num_nodes() == 0 {
        return Ok(None);
    }
    if vtree.node(vtree.root()).is_leaf() {
        return Ok((tdd.output().local == ONE_LEAF_IDX && !tdd.level(vtree.root()).is_marginal()).then(|| "the output names its leaf as One".to_string()));
    }
    let reach = tdd.reachable_nodes();
    let mut first = None;
    for t in vtree.bottomup() {
        let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
        let level = tdd.level(t);
        if level.is_marginal() {
            continue;
        }
        for (child, is_left) in [(left, true), (right, false)] {
            if !vtree.node(child).is_leaf() || tdd.level(child).is_marginal() {
                continue;
            }
            let (mut one, mut other) = (false, false);
            for j in (0..level.slot_count()).filter(|&j| reach[t.idx()].get(j).copied().unwrap_or(false)) {
                for pair in level.pairs_iter_of_idx(j) {
                    let side = if is_left { pair.left } else { pair.right };
                    if side.raw() == ONE_LEAF_IDX.0 {
                        one = true;
                        first.get_or_insert_with(|| format!("node {j} of level {} names leaf {} as One", t.0, child.0));
                    } else {
                        other = true;
                    }
                }
            }
            if one && other {
                return Err(format!("level {} names leaf {} as One in some pairs and not in others", t.0, child.0));
            }
        }
    }
    Ok(first)
}

/// No pair of `tdd` names a leaf as `One`, nor does the output, dead nodes
/// included; a constant has no leaf; the loose levels, when known, are
/// complete ([`check_loose_levels`](super::check_loose_levels)); a certified
/// `tdd` reaches every node it stores and, with no summed-out level, reads no
/// leaf child through twin literals in every node. An uncertified `tdd` may
/// hold nodes its output does not reach. With `minimized`, also that
/// the leaves are the
/// [`support`](Tdd::support).
///
/// Cost: O(diagram size).
pub fn check_trimmed(tdd: &Tdd, minimized: bool) -> Result<(), String> {
    let vtree = tdd.vtree();
    if (tdd.is_zero() || tdd.is_one()) != (vtree.num_nodes() == 0) {
        return Err(format!(
            "a constant on {} vtree nodes, or a diagram on none (false {}, true {})",
            vtree.num_nodes(),
            tdd.is_zero(),
            tdd.is_one()
        ));
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
    super::check_loose_levels(tdd)?;
    if tdd.levels.is_canonical(tdd.output()) {
        every_node_reached(tdd)?;
        if !tdd.has_marginal_level() {
            full_twin_side(tdd)?;
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

/// The output of a certified `tdd` reaches every node of its structural
/// internal levels.
fn every_node_reached(tdd: &Tdd) -> Result<(), String> {
    let vtree = tdd.vtree();
    let reach = tdd.reachable_nodes();
    for t in vtree.bottomup() {
        let level = tdd.level(t);
        if vtree.node(t).is_leaf() || level.is_marginal() {
            continue;
        }
        if let Some(j) = (0..level.slot_count()).find(|&j| !reach[t.idx()].get(j).copied().unwrap_or(false)) {
            return Err(format!("certified, yet the output does not reach node {j} of level {}", t.0));
        }
    }
    Ok(())
}

/// No level of a canonical `tdd` reads a leaf child the same through both
/// literals in every reachable node: such a leaf is one the diagram does not
/// depend on, which a closed result leaves. An unminimized result may hold
/// such a level as contraction debt, so only a certified one is held to it.
fn full_twin_side(tdd: &Tdd) -> Result<(), String> {
    let vtree = tdd.vtree();
    let reach = tdd.reachable_nodes();
    for t in vtree.bottomup() {
        let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
        let level = tdd.level(t);
        for (child, is_left) in [(left, true), (right, false)] {
            if !vtree.node(child).is_leaf() {
                continue;
            }
            let mut reached = (0..level.slot_count()).filter(|&j| reach[t.idx()].get(j).copied().unwrap_or(false)).peekable();
            if reached.peek().is_none() {
                continue;
            }
            let twins = reached.all(|j| {
                let (mut pos, mut neg) = (Vec::new(), Vec::new());
                for pair in level.pairs_iter_of_idx(j) {
                    let (leaf, other) = if is_left { (pair.left, pair.right) } else { (pair.right, pair.left) };
                    match leaf.raw() {
                        r if r == POS_LEAF_IDX.0 => pos.push(other.raw()),
                        r if r == NEG_LEAF_IDX.0 => neg.push(other.raw()),
                        _ => return false,
                    }
                }
                pos.sort_unstable();
                neg.sort_unstable();
                pos == neg
            });
            if twins {
                return Err(format!("level {} reads leaf {} the same through both literals in every node", t.0, child.0));
            }
        }
    }
    Ok(())
}
