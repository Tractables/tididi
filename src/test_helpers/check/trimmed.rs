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

/// The projection form of `docs/invariants.md`: invariant 15 over every
/// stored pair; on a home 22 and 23 (`regions`); 17
/// ([`check_loose_levels`](super::check_loose_levels)); on a certified `tdd`
/// 4 and, with no summed-out level, 20 over the reachable nodes; with
/// `minimized` and no summed-out level 21, the leaves read against
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
    let root = vtree.root();
    if vtree.node(root).is_leaf() && !vtree.node(root).is_region() && tdd.output().local == ONE_LEAF_IDX {
        return Err("the output names its leaf as One".to_string());
    }
    if vtree.home().is_some() {
        regions(tdd)?;
    }
    for t in vtree.bottomup() {
        let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
        let level = tdd.level(t);
        if level.is_marginal() {
            continue;
        }
        for j in 0..level.slot_count() {
            for pair in level.pairs_iter_of_idx(j) {
                for (child, side) in [(left, pair.left), (right, pair.right)] {
                    if vtree.node(child).is_var_leaf() && !tdd.level(child).is_marginal() && side.raw() == ONE_LEAF_IDX.0 {
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
    if minimized && !tdd.has_marginal_level() {
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

/// On a home: a summed-out level stands only at a region or a variable leaf
/// (which a builder can seat) and every region holds one (R1); the extents are pairwise disjoint and hold the rank of no
/// variable leaf (R2).
fn regions(tdd: &Tdd) -> Result<(), String> {
    let vtree = tdd.vtree();
    let home = vtree.home().expect("a vtree on a home");
    for t in vtree.bottomup() {
        let summed_leaf = vtree.node(t).is_var_leaf() && tdd.level(t).is_marginal();
        if tdd.level(t).is_marginal() != vtree.node(t).is_region() && !summed_leaf {
            return Err(format!("level {} is summed out {}, a region {}", t.0, tdd.level(t).is_marginal(), vtree.node(t).is_region()));
        }
    }
    let mut spans: Vec<(u32, u32)> = vtree.regions().flat_map(|t| vtree.region_at(t).1.into_vec()).collect();
    spans.extend(vtree.bottomup().filter(|&t| vtree.node(t).is_var_leaf()).map(|t| {
        let rank = home.rank(vtree.leaf_var(t)).expect("a variable of the home");
        (rank, rank + 1)
    }));
    spans.sort_unstable();
    match spans.windows(2).find(|w| w[1].0 < w[0].1) {
        Some(w) => Err(format!("extents or variable leaves overlap: ranks {:?} and {:?}", w[0], w[1])),
        None => Ok(()),
    }
}

/// A certified `tdd` holds to its certificate: invariants 4, 19, 20 and 3
/// ([`check_canonicity`](super::check_canonicity)), and a copy with the
/// certificate dropped and every level owing the contraction passes
/// minimizes to as many nodes and pairs (18). An uncertified `tdd` passes;
/// on one with a summed-out level only 4 is checked.
///
/// Cost: O(diagram size), and a minimization of a copy.
pub fn check_certified(tdd: &Tdd) -> Result<(), String> {
    if !tdd.levels.is_canonical(tdd.output()) || tdd.vtree().num_nodes() == 0 {
        return Ok(());
    }
    every_node_reached(tdd)?;
    if tdd.has_marginal_level() {
        return Ok(());
    }
    full_twin_side(tdd)?;
    super::check_canonicity(tdd, 3)?;
    let eng = crate::Engine::new();
    let mut copy = tdd.clone();
    copy.levels.forget();
    let internal = tdd.vtree().internal_bottomup_slice().iter().copied();
    copy.dirty = crate::diagram::Dirty::default().seeded(tdd.vtree(), internal, Some(&eng)).map_err(|e| e.to_string())?;
    // Counted on a copy: a count kept on the levels would show in `tdd`.
    let before = tdd.clone();
    let before = (before.node_count(), before.pair_count());
    eng.minimize_plain(&mut copy).map_err(|e| e.to_string())?;
    let after = (copy.node_count(), copy.pair_count());
    if after != before {
        return Err(format!("certified with (nodes, pairs) {before:?}, which minimize to {after:?}"));
    }
    Ok(())
}

/// No stored node of an internal level denotes every assignment of the
/// variables under its vtree node, on a diagram with no summed-out level.
/// Such a node is the only node of its level (item 1), so every parent pair
/// names it and the parent level only passes its other child through. Item
/// 15 excludes the form that names `One`; this one is built from literals,
/// as `{(Pos, Pos), (Pos, Neg), (Neg, Pos), (Neg, Neg)}` is, and item 21
/// excludes it from a certified diagram. Levels over more than 126 variables
/// are skipped.
///
/// Cost: O(diagram size) `u128` operations.
pub fn check_no_true_node(tdd: &Tdd) -> Result<(), String> {
    let vtree = tdd.vtree();
    if vtree.num_nodes() == 0 || tdd.has_marginal_level() || tdd.is_zero() {
        return Ok(());
    }
    let counts = tdd.node_counts_u128().map_err(|e| e.to_string())?;
    let mut width = vec![0u32; vtree.num_nodes()];
    for t in vtree.bottomup() {
        width[t.idx()] = match *vtree.node(t) {
            VtreeNode::Internal { left, right, .. } => width[left.idx()] + width[right.idx()],
            _ => 1,
        };
        if vtree.node(t).is_leaf() || width[t.idx()] > 126 {
            continue;
        }
        if let Some(j) = counts[t.idx()].iter().position(|&c| c == 1u128 << width[t.idx()]) {
            return Err(format!("node {j} of level {} denotes every assignment of its {} variables", t.0, width[t.idx()]));
        }
    }
    Ok(())
}

/// On a plain root, with no summed-out level: a leaf is named `One` (by its
/// parent level, or as the output of a one-leaf diagram) exactly when its
/// variable is outside the [`support`](Tdd::support), so the only levels that
/// pass a child through are those of variables the function ignores. Holds of
/// a certified diagram (Corollary A1 of the projection form); an
/// unminimized one may read a variable it ignores through literals.
///
/// Cost: O(diagram size).
pub fn check_padded_support(tdd: &Tdd) -> Result<(), String> {
    let vtree = tdd.vtree();
    if vtree.num_nodes() == 0 || tdd.has_marginal_level() || tdd.is_zero() || tdd.is_one() {
        return Ok(());
    }
    let support = tdd.support().map_err(|e| e.to_string())?;
    for t in vtree.bottomup() {
        if !vtree.node(t).is_var_leaf() {
            continue;
        }
        let var = vtree.leaf_var(t);
        let named_one = match vtree.node(t).parent() {
            None => tdd.output().local == ONE_LEAF_IDX,
            Some(p) => {
                let VtreeNode::Internal { left, .. } = *vtree.node(p) else { unreachable!("a parent is internal") };
                let level = tdd.level(p);
                let mut sides = (0..level.slot_count()).flat_map(|j| level.pairs_iter_of_idx(j)).map(|pair| if t == left { pair.left } else { pair.right });
                sides.next().is_some_and(|side| side.raw() == ONE_LEAF_IDX.0)
            }
        };
        if named_one == support.contains(&var) {
            return Err(format!("leaf {} of variable {} is named One {named_one}, in the support {}", t.0, var.0, !named_one));
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
