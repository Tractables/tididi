//! Reduction metrics for compiled diagrams.
//!
//! [`reduced_size`] estimates how much smaller the diagram would be under one of
//! the SDD-style reduction rules (see `docs/tdd.md`).

use num_bigint::BigUint;

use crate::diagram::ChildSide;
use super::node_counts;
use crate::diagram::*;
use crate::vtree::VtreeIdx;

/// The pairs a rule would remove.
///
/// Both rules ask the same question of every node: do all its pairs share one
/// side's child, while the other side covers everything its child level can
/// offer? A node like that computes a function of one child alone and can be
/// short-circuited down to it, taking its whole pair list with it. The rules
/// differ only in what "covers everything" means, which is `covers`.
///
/// `covers(t, node, child, pairs, side)` is asked about the varying side:
/// `side` says which component of a pair indexes into `child`.
fn reducible_pairs(
    tdd: &Tdd,
    covers: impl Fn(VtreeIdx, usize, VtreeIdx, &[InputPair], ChildSide) -> bool,
) -> usize {
    let mut reducible = 0usize;
    for (t, left, right) in tdd.vtree.internal_bottomup() {
        let level = &tdd.levels[t.idx()];
        for (node_i, node) in level.nodes.iter().enumerate() {
            if !node.is_internal() {
                continue;
            }
            // Materialize the pair list — cheap next to the sums below, and it
            // survives both packed and unpacked levels.
            let pairs: Vec<InputPair> = level.pairs_iter_of(node).collect();
            if pairs.is_empty() {
                continue;
            }
            // Reducible to the right child: every pair names the same right,
            // and the left side covers its whole child.
            let first_right = pairs[0].right;
            if pairs.iter().all(|p| p.right == first_right)
                && covers(t, node_i, left, &pairs, ChildSide::Left)
            {
                reducible += pairs.len();
                continue;
            }
            // Reducible to the left child: the mirror image.
            let first_left = pairs[0].left;
            if pairs.iter().all(|p| p.left == first_left)
                && covers(t, node_i, right, &pairs, ChildSide::Right)
            {
                reducible += pairs.len();
            }
        }
    }
    reducible
}

/// One side of a node's pairs, as indices into `child`'s column. A marginal
/// child's references carry a tag, which the side view strips.
fn side_slots(tdd: &Tdd, child: VtreeIdx, pairs: &[InputPair], side: ChildSide) -> Vec<usize> {
    let view = tdd.levels[child.idx()].side_view();
    pairs
        .iter()
        .map(|p| {
            let r = if side == ChildSide::Left { p.left } else { p.right };
            view.coord(r).idx()
        })
        .collect()
}

/// The r1SDD metric of [`reduced_size`].
///
/// A node is **reducible** if its sub-function depends on only one child's
/// variables — it can be short-circuited down to the relevant child. The side
/// that varies covers everything when its model counts sum to `2^|vars(child)|`:
/// the side admits every assignment of its variables, so the function does not
/// depend on them.
///
/// Returns `tdd.size() - reducible_pairs`. See `docs/tdd.md` for details.
fn r1_sdd_size(tdd: &Tdd) -> usize {
    // `ZERO` sentinel: the diagram is UNSAT, size is 0.
    if tdd.is_zero() {
        return 0;
    }
    let counts = node_counts(tdd);
    let mut subtree_vars = vec![0u32; tdd.levels.len()];
    for (t, _var) in tdd.vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in tdd.vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }

    let reducible = reducible_pairs(tdd, |t, node_i, child, pairs, side| {
        let all_models = BigUint::from(1u32) << subtree_vars[child.idx()] as usize;
        let slots = side_slots(tdd, child, pairs, side);
        let sum: BigUint = slots.iter().map(|&s| &counts[child.idx()][s]).sum();
        if sum != all_models {
            return false;
        }
        // The node's own count must then be the product form: the covering
        // side contributes all of its models, the constant side contributes
        // its own count. (A debug build checks it; release takes the sum's
        // word for it.)
        debug_assert_eq!(
            {
                let (l, r) = tdd.vtree.children(t);
                let other = if side == ChildSide::Left { r } else { l };
                let p = pairs[0];
                let kept = if side == ChildSide::Left { p.right } else { p.left };
                let kept_slot = tdd.levels[other.idx()].side_view().coord(kept).idx();
                &counts[other.idx()][kept_slot] * &all_models
            },
            counts[t.idx()][node_i],
            "reducible node {node_i} at vtree {t:?} is not in product form",
        );
        true
    });

    tdd.size().saturating_sub(reducible)
}

/// Reduced diagram size under the r2TDD rule (structural variant, always ≤ r1SDD).
///
/// Like [`r1_sdd_size`], but it decides coverage structurally — one side
/// enumerates every node of its child level — instead of by model counts. That
/// is cheaper (no `BigUint` arithmetic) and strictly more aggressive: every
/// r1SDD-reducible node is r2TDD-reducible, not the other way round.
///
/// In a purely Boolean diagram pair lists are duplicate-free, so "all lefts
/// identical and `pairs.len() == child_level.width()`" implies the right
/// indices are exactly `0..width`. In a marginalized diagram pair lists are
/// multisets and a repeated pair can inflate `pairs.len()` to the level width
/// without covering it — this diagnostic size metric may then over-count
/// reducible pairs. It feeds reporting only, never a model count.
fn r2_tdd_size(tdd: &Tdd) -> usize {
    if tdd.is_zero() {
        return 0;
    }
    let reducible = reducible_pairs(tdd, |_t, _node_i, child, pairs, side| {
        covers_child_width(tdd, child, pairs, side)
    });
    tdd.size().saturating_sub(reducible)
}

/// Check if a set of pairs covers the full width of a child level.
///
/// For internal children: `pairs.len() == width` (all indices 0..width).
/// For leaf children: the referenced labels must cover all 2^1 = 2 models
/// of the single leaf variable. Pos and Neg each contribute 1 model;
/// one alone contributes 2 (covers both assignments).
fn covers_child_width(tdd: &Tdd, child: crate::vtree::VtreeIdx, pairs: &[InputPair], side: ChildSide) -> bool {
    if tdd.vtree.node(child).is_leaf() {
        let mut seen = [false; LEAF_WIDTH];
        for pair in pairs {
            let idx = if side == ChildSide::Left { pair.left.idx() } else { pair.right.idx() };
            seen[idx] = true;
        }
        // Model weights: One(0)=2, Pos(1)=1, Neg(2)=1.
        const LEAF_MODEL_WEIGHT: [usize; LEAF_WIDTH] = [2, 1, 1];
        let coverage: usize = (0..LEAF_WIDTH)
            .filter(|&i| seen[i])
            .map(|i| LEAF_MODEL_WEIGHT[i])
            .sum();
        coverage >= 2 // 2^1 = total models for one leaf variable
    } else {
        pairs.len() == tdd.levels[child.idx()].width()
    }
}

/// Which reduction rule [`reduced_size`] measures against.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ReductionRule {
    /// A node reduces when its sub-function depends on one child's variables
    /// only, decided by model counts. The rule SDDs are canonical under.
    R1Sdd,
    /// A node reduces when one side enumerates every node of its child level.
    /// Purely structural, cheaper, and strictly more aggressive than
    /// [`ReductionRule::R1Sdd`]; a diagnostic, never a model count.
    R2Tdd,
}

/// Estimate the diagram's size after applying `rule`.
///
/// Returns `tdd.size()` minus the pairs the rule would remove. See
/// `docs/tdd.md` for what each rule reduces.
pub fn reduced_size(f: &Tdd, rule: ReductionRule) -> usize {
    match rule {
        ReductionRule::R1Sdd => r1_sdd_size(f),
        ReductionRule::R2Tdd => r2_tdd_size(f),
    }
}
