//! Seeding the conjunction output's marginal vtree leaves from its operands.

use crate::diagram::{Tdd, TddLevel, WeightStore};
use crate::vtree::Vtree;
use crate::diagram::Sides;

/// Mark the output's marginal vtree leaves, which the bottom-up loop never
/// visits as a level of its own, and collect the weight-marginal leaves whose
/// refs still have to be canonicalized once the output's pairs are final.
///
/// A marginalized leaf variable is private to one operand, so the other is the
/// identity there and the parent's marginal-child dispatch carries the refs
/// through.
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
    // The bottom-up loop never visits a leaf as `t`, so a marginal leaf's output
    // level is flagged here from the operands. An integer-marginal leaf keeps
    // its counts inline at the parent, so the output store stays empty. A
    // weight-marginal leaf holds a pinned per-slot column in the `WeightStore`
    // (`check_leaf_columns_pinned`), so the output level reports `LEAF_WIDTH`
    // and this only re-flags it.
    //
    // `canon_leaves` collects the leaves weight-marginal on one operand only:
    // the other operand's leaf-label refs flow through the grid into the output
    // and may not be canonical (`diagram::leaf_canon_map`), so they are
    // rewritten once the output's pair lists are final. With both operands
    // weight-marginal each canon class is closed under conjunction, so nothing
    // is recorded.
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
                // skipped by the weighted marginalization pass and read as an empty
                // column by the streaming child view — dropping the leaf's entire
                // mass with no error anywhere.
                let leaf_slots = crate::diagram::LEAF_WIDTH;
                debug_assert!(
                    ws.is_none_or(|w| w.level(left_idx).is_none_or(|v| v.len() == leaf_slots)),
                    "weight-marginal leaf {left_idx}: WeightStore column is not the \
                     pinned {leaf_slots}-slot leaf_val cache",
                );
                debug_assert!(
                    (!w1 || f.levels[left_idx].slot_count() == leaf_slots)
                        && (!w2 || g.levels[left_idx].slot_count() == leaf_slots),
                    "weight-marginal leaf {left_idx}: operand slot carriers \
                     (f={}, g={}) disagree with `LEAF_WIDTH` ({leaf_slots})",
                    f.levels[left_idx].slot_count(), g.levels[left_idx].slot_count(),
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
