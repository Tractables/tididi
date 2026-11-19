//! Structural checkers: vtree agreement, the absence of false nodes and the
//! known loose levels.

use crate::diagram::{Tdd, ZERO};

// ── Public checker functions ─────────────────────────────────────────────────

/// Validate that every diagram node matches its vtree position.
///
/// Checks:
/// - Leaf vtree levels have no stored nodes (implicit representation)
/// - Internal vtree levels contain only internal diagram nodes
/// - All `ChildPair` child references are in bounds
/// - Output node is at the vtree root with a valid local index
///
/// Cost: O(diagram size).
pub fn validate_vtree_structure(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;

    if tdd.output.vtree != vtree.root() {
        return Err(format!(
            "output vtree {:?} != root {:?}",
            tdd.output.vtree, vtree.root()
        ));
    }

    if tdd.output.local == ZERO {
        return Ok(());
    }

    let root_width = super::checked_width(tdd, vtree.root());
    if tdd.output.local.idx() >= root_width {
        return Err(format!(
            "output local index {} >= root width {}",
            tdd.output.local.0, root_width
        ));
    }

    for (t, _var) in vtree.leaf_bottomup() {
        if !tdd.level(t).nodes().is_empty() {
            return Err(format!(
                "vtree leaf {:?} has non-empty nodes vec (len {}) — \
                 leaf levels should be implicit",
                t, tdd.level(t).nodes().len()
            ));
        }
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let left_width = super::checked_width(tdd, left);
        let right_width = super::checked_width(tdd, right);
        let level = tdd.level(t);
        let left_view = tdd.level(left).child_decoder();
        let right_view = tdd.level(right).child_decoder();

        for (i, node) in level.nodes().iter().enumerate() {
            for (j, pair) in level.pairs_iter_of(&node).enumerate() {
                // An inline marginal ref carries its value in the reference
                // itself and indexes nothing, so only the two indexing forms
                // have a width to be in bounds of.
                if let Some(l) = left_view.child(pair.left).index()
                    && l >= left_width
                {
                    return Err(format!(
                        "vtree {:?} node {} input {} left index {} >= left child width {}",
                        t, i, j, l, left_width
                    ));
                }
                if let Some(r) = right_view.child(pair.right).index()
                    && r >= right_width
                {
                    return Err(format!(
                        "vtree {:?} node {} input {} right index {} >= right child width {}",
                        t, i, j, r, right_width
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Check that no node in any level computes the constant-false function.
///
/// This invariant holds for every public diagram (even before minimize):
/// - A leaf level stores no nodes, so the constant-false atom never appears
/// - No internal nodes with empty pairs exist in any level
/// - If UNSAT (after minimize), all internal levels are empty
///
/// Cost: O(total nodes).
pub fn check_no_false_nodes(tdd: &Tdd) -> Result<(), String> {
    check_no_false_nodes_in_levels(tdd)?;

    if tdd.output.local == ZERO {
        for (t_idx, level) in tdd.levels.iter().enumerate() {
            if !level.nodes().is_empty() {
                return Err(format!(
                    "UNSAT TDD (ZERO output) has non-empty level at vtree index {} \
                     (width {}) — expected empty after minimize",
                    t_idx,
                    level.slot_count()
                ));
            }
        }
    }
    Ok(())
}

/// Check that no node in any level computes constant-false.
///
/// This is the per-level subset of [`check_no_false_nodes`] — it does *not*
/// require all levels to be empty on UNSAT. Useful for checking raw
/// conjunction output before `minimize`.
///
/// Cost: O(total nodes).
pub fn check_no_false_nodes_in_levels(tdd: &Tdd) -> Result<(), String> {
    for t in tdd.vtree.bottomup() {
        // A leaf level stores nothing: its three nodes are implicit and none of
        // them is the constant-false atom.
        if tdd.vtree.node(t).is_leaf() {
            continue;
        }
        let level = tdd.level(t);
        for (i, node) in level.nodes().iter().enumerate() {
            if level.pairs_iter_of(&node).next().is_none() {
                return Err(format!(
                    "vtree {:?} node {}: Internal with empty inputs — no real node \
                     should compute constant-false (use the `ZERO` sentinel instead)",
                    t, i
                ));
            }
        }
    }
    Ok(())
}

/// Check the loose levels a diagram knows
/// (`Dirty::loose`): each structural
/// internal level below the root that they leave out has every node named
/// by a pair of its parent level. A diagram that does not know them passes.
///
/// Cost: O(total pairs).
pub fn check_loose_levels(tdd: &Tdd) -> Result<(), String> {
    let Some(loose) = tdd.dirty.loose() else { return Ok(()) };
    if tdd.is_zero() || tdd.is_one() {
        return Ok(());
    }
    let vtree = &tdd.vtree;
    for (p, left, right) in vtree.internal_bottomup() {
        let level = &tdd.levels[p.idx()];
        for (c, on_left) in [(left, true), (right, false)] {
            if vtree.node(c).is_leaf() || tdd.levels[c.idx()].is_marginal() || loose.contains(&c.0) {
                continue;
            }
            let decoder = tdd.levels[c.idx()].child_decoder();
            let mut named = vec![false; tdd.levels[c.idx()].slot_count()];
            for node in level.nodes().iter() {
                for pair in level.pairs_iter_of(&node) {
                    let side = if on_left { pair.left } else { pair.right };
                    if let Some(i) = decoder.child(side).index() {
                        named[i] = true;
                    }
                }
            }
            if let Some(i) = named.iter().position(|&n| !n) {
                return Err(format!(
                    "level {} is not listed loose, yet no pair of its parent {} names its node {i}",
                    c.0, p.0
                ));
            }
        }
    }
    Ok(())
}
