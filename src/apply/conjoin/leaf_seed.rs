//! Seeding the conjunction output's marginal vtree leaves from its operands.

use crate::diagram::{Tdd, TddLevel, WeightStore};
use crate::vtree::Vtree;
use super::marginal_plan::Sides;

/// Mark the output's marginal vtree leaves, which the bottom-up loop never
/// visits as a level of its own, and collect the weight-marginal leaves whose
/// refs still have to be canonicalized once the output's pairs are final.
///
/// A marginalized leaf variable is private to one operand, so the other is the
/// identity there and the parent's marginal-child dispatch carries the refs
/// through. A restricted apply skips the sweep: its output levels are merged
/// back into the accumulator's, which already carries its own marginal leaves.
// The debug assertion enumerates the three legal marginal-leaf shapes; a
// factored form hides which case is which.
#[allow(clippy::nonminimal_bool)]
pub(crate) fn seed_output_leaves(
    f: &Tdd,
    g: &Tdd,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    identity: Sides<&[bool]>,
    ws: Option<&WeightStore>,
) -> Vec<usize> {
    let (left_identity, right_identity) = (identity.left, identity.right);
    // Leaf marginalization: a marginal vtree leaf is never visited as a `t` by
    // the bottom-up loop, so — unlike a marginal internal child — its output level
    // is never marked marginal. Seed it here from the operands. A marginalized
    // leaf var is private (summed only once its every clause is compiled), so the
    // other operand is identity at that leaf; the parent's marginal-child dispatch
    // then routes Route A and the passthrough path carries the carrier's inline
    // `ValueRef` refs through verbatim. The output store stays empty (all leaf
    // counts are inline at the parent).
    // In weighted mode the leaf's counts are not inline at the parent: the
    // weighted leaf-marginal installs a real per-slot column in the (vtree-indexed)
    // `WeightStore` and leaves the parent's bare leaf-label refs to
    // decode as `ValueRef::Slot`. That column is pinned (invariant 11), so the
    // output level reports `LEAF_WIDTH` and this only re-flags it.
    //
    // `canon_leaves` collects the leaves flagged weight-marginal on one operand's
    // authority: the other operand was structural there, so its genuine leaf-label
    // refs flow through `CONJOIN_GRID` into the output and may not be canonical
    // (`marginalize::leaf_canon_map`). They are canonicalized once the output's
    // pair lists are final — the bottom-up loop further down emits them, so there is
    // nothing to rewrite here yet. When both operands are weight-marginal both
    // sides are already canonical and the grid is closed over each canon class
    // (`{One,Pos}`, `{One,Neg}`, `{One}` are each closed under ∧), so nothing is
    // recorded.
    let mut canon_leaves: Vec<usize> = Vec::new();
    for (leaf, _) in vtree.leaf_bottomup() {
        let left_idx = leaf.idx();
        let left_m = f.levels[left_idx].is_marginal();
        let right_m = g.levels[left_idx].is_marginal();
        if left_m || right_m {
            debug_assert!(
                (left_m && right_m) || (left_m && right_identity[left_idx]) || (right_m && left_identity[left_idx]),
                "marginal leaf {left_idx} conjoined with a non-identity operand \
                 (var not private?): left_m={left_m} right_m={right_m} \
                 left_id={} right_id={}",
                left_identity[left_idx], right_identity[left_idx],
            );
            let w1 = f.levels[left_idx].is_weight_marginal();
            let w2 = g.levels[left_idx].is_weight_marginal();
            if w1 || w2 {
                // The column is pinned (invariant 11), so the output level's
                // slot count is `LEAF_WIDTH`.
                //
                // Flagging it directly (rather than reading the column's length)
                // is what makes this robust: a `map_or(0, len)` read reports width
                // 0 whenever the global column happens not to be installed for
                // this vtree index, and a width-0 weight-marginal leaf is silently
                // skipped by `marginalize_batch_weighted` and read as an empty
                // column by the streaming child view — dropping the leaf's entire
                // mass with no error anywhere.
                let leaf_slots = crate::diagram::LEAF_WIDTH;
                debug_assert!(
                    ws.is_none_or(|w| w.level(left_idx).is_none_or(|v| v.len() == leaf_slots)),
                    "weight-marginal leaf {left_idx}: WeightStore column is not the \
                     pinned {leaf_slots}-slot leaf_val cache",
                );
                debug_assert!(
                    (!w1 || f.levels[left_idx].width() == leaf_slots)
                        && (!w2 || g.levels[left_idx].width() == leaf_slots),
                    "weight-marginal leaf {left_idx}: operand slot carriers \
                     (f={}, g={}) disagree with `LEAF_WIDTH` ({leaf_slots})",
                    f.levels[left_idx].width(), g.levels[left_idx].width(),
                );
                levels[left_idx].become_marginal_weighted(leaf_slots as u32);
                if w1 != w2 {
                    canon_leaves.push(left_idx);
                }
            } else {
                levels[left_idx].become_marginal(Vec::new(), None);
            }
        }
    }
    canon_leaves
}
