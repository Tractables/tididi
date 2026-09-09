//! Structural checkers: vtree agreement and the absence of false nodes.

use crate::diagram::*;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};

/// The index a child reference reads at its level, or `None` for an inline
/// marginal value, which stands alone.
fn child_index(child: ChildRef) -> Option<usize> {
    match child {
        ChildRef::Node(NodeIdx(s)) | ChildRef::Value(ValueRef::Slot(s)) => Some(s as usize),
        ChildRef::Value(ValueRef::Inline(_)) => None,
    }
}

// ── Public checker functions ─────────────────────────────────────────────────

/// Validate that every diagram node matches its vtree position.
///
/// Checks:
/// - Leaf vtree levels have no stored nodes (implicit representation)
/// - Internal vtree levels contain only internal diagram nodes
/// - All `InputPair` child references are in bounds
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

    let root_width = tdd.effective_width(vtree.root());
    if tdd.output.local.idx() >= root_width {
        return Err(format!(
            "output local index {} >= root width {}",
            tdd.output.local.0, root_width
        ));
    }

    for (t, _var) in vtree.leaf_bottomup() {
        if !tdd.level(t).nodes.is_empty() {
            return Err(format!(
                "vtree leaf {:?} has non-empty nodes vec (len {}) — \
                 leaf levels should be implicit",
                t, tdd.level(t).nodes.len()
            ));
        }
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let left_width = tdd.effective_width(left);
        let right_width = tdd.effective_width(right);
        let level = tdd.level(t);
        let left_view = tdd.level(left).side_view();
        let right_view = tdd.level(right).side_view();

        for (i, node) in level.nodes.iter().enumerate() {
            if !node.is_internal() {
                return Err(format!(
                    "vtree internal {:?} has non-Internal node at index {}",
                    t, i
                ));
            }
            for (j, pair) in level.pairs_iter_of(node).enumerate() {
                // An inline marginal ref carries its value in the reference
                // itself and indexes nothing, so only the two indexing forms
                // have a width to be in bounds of.
                if let Some(l) = child_index(left_view.child(pair.left))
                    && l >= left_width
                {
                    return Err(format!(
                        "vtree {:?} node {} input {} left index {} >= left child width {}",
                        t, i, j, l, left_width
                    ));
                }
                if let Some(r) = child_index(right_view.child(pair.right))
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
/// - Leaf levels are marginal (no stored nodes) — Zero never appears
/// - No internal nodes with empty pairs exist in any level
/// - If UNSAT (after minimize), all internal levels are empty
///
/// Cost: O(total nodes).
pub fn check_no_false_nodes(tdd: &Tdd) -> Result<(), String> {
    check_no_false_nodes_in_levels(tdd)?;

    if tdd.output.local == ZERO {
        for (t_idx, level) in tdd.levels.iter().enumerate() {
            if !level.nodes.is_empty() {
                return Err(format!(
                    "UNSAT TDD (ZERO output) has non-empty level at vtree index {} \
                     (width {}) — expected empty after minimize",
                    t_idx,
                    level.width()
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
/// `apply_and` output before `minimize`.
///
/// Cost: O(total nodes).
pub fn check_no_false_nodes_in_levels(tdd: &Tdd) -> Result<(), String> {
    for t in tdd.vtree.bottomup() {
        // Skip leaf levels — they are marginal and always contain Pos, Neg, One (no Zero).
        if tdd.vtree.node(t).is_leaf() {
            continue;
        }
        let level = tdd.level(t);
        for (i, node) in level.nodes.iter().enumerate() {
            if node.is_internal() && level.pairs_iter_of(node).next().is_none() {
                return Err(format!(
                    "vtree {:?} node {}: Internal with empty inputs — no real node \
                     should compute constant-false (use ZERO sentinel instead)",
                    t, i
                ));
            }
        }
    }
    Ok(())
}
