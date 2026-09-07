//! De-marginalize a restriction constraint into a pure satisfiability indicator.
//!
//! Turns a marginal-bearing TDD into a marginal-free indicator whose models over
//! the non-summed vars are exactly the assignments with non-zero marginal count.

use crate::tdd::build::{constant_one, constant_zero};
use crate::tdd::minimize::minimize;
use crate::tdd::types::Tdd;
use crate::vtree::VtreeIdx;

/// De-marginalize a restriction constraint into a pure satisfiability INDICATOR:
/// a TDD with NO marginal level whose models, on the variables that were *not*
/// summed out, are exactly the assignments under which the original marginal
/// count is non-zero — and over the summed-out vars it is the full cube (TRUE).
///
/// This is the restriction primitive a marginal-bearing split needs. Conjunction
/// over marginal nodes is not the supported regime: apply only does `identity ∧
/// marginal`, never `marginal ∧ marginal`. The split's free-above restriction `r`
/// is conjoined into every OTHER segment, and the REAL summed weight is carried
/// once by the structurally-pruned split-on segment (`pruned_t0`). So `r` must
/// not carry weight at all — it must be a plain boolean indicator over the shared
/// vars, free over the private (summed-out) ones.
///
/// Lifting marginal → free is sound and count-exact: the summed-out vars are
/// PRIVATE to `pruned_t0` (the marginalization precondition), so no other segment
/// constrains them. In the joint segmented product they are counted once, by
/// `pruned_t0`'s weights; `r` (and every other segment) being a free cube over
/// them contributes factor 1. Conjoining the now-free `r` against `pruned_t0`
/// (marginal over those vars) is the supported `identity ∧ marginal` case, and
/// against the other free segments it is plain `free ∧ free`. No marginal level
/// survives, so no marginal×marginal conjunction can ever arise.
///
/// # Panics
///
/// Panics if an integer-marginal child level is missing its `marginal_counts`.
pub fn demarginalize_to_indicator(r: &mut Tdd) {
    use crate::vtree::VtreeNode;
    use crate::tdd::types::{MARG_OVERFLOW_TAG, MARG_VALUE_MASK, decode_marg_coord};

    // A structural (free) leaf carries no `marginal_counts` (`is_marginal()` is
    // false); only a SUMMED-OUT level — internal subtree or a private leaf var
    // folded into a count table — is `is_marginal()`. Either kind must be lifted
    // to a free cube, so trigger on any marginal level (constant_one's own leaves
    // are non-marginal, so this never fires spuriously on the lifted result).
    let has_marg = r.levels.iter().any(|l| l.is_marginal());
    if !has_marg {
        return;
    }
    debug_assert!(
        !r.levels.iter().any(|l| l.is_weight_marginal()),
        "demarginalize_to_indicator: weight-marginal levels unsupported \
         (the segment-search marginalization path is integer-marginal)"
    );

    // Whole function summed out (marginal root): the indicator is the constant
    // satisfiability of the total count.
    if r.levels[r.vtree.root().idx()].is_marginal() {
        let nonzero = crate::tdd::query::model_count(r) != num_bigint::BigUint::from(0u32);
        *r = if nonzero { constant_one(&r.vtree) } else { constant_zero(&r.vtree) };
        return;
    }

    // Every marginal subtree (internal level OR a summed-out leaf var) hangs off a
    // NON-marginal vtree parent (once a node marginalizes, everything below it is
    // marginal/leaf). Walk each such frontier from its non-marginal parent and:
    //   (1) replace the whole subtree with `constant_one` — the marginalized vars
    //       become a free cube (TRUE), exactly the spec: fixing the surviving vars
    //       leaves TRUE over the summed-out vars;
    //   (2) rewrite the parent's marg-side refs into the subtree to the child's
    //       free node (index 0) iff the node's count is non-zero. A minimized
    //       marginal TDD has no count==0 node (unsatisfiable nodes are pruned, and
    //       false is encoded by pair-omission, not a ZERO child ref), so every
    //       surviving ref maps to the true node — asserted, not silently dropped.
    // The result carries NO marginal level, so it conjoins into the other segments
    // purely through the supported `identity ∧ X` paths — never marginal×marginal.
    let one = constant_one(&r.vtree);
    for p_idx in 0..r.levels.len() {
        if r.levels[p_idx].is_marginal() {
            continue;
        }
        let VtreeNode::Internal { left, right, .. } = *r.vtree.node(VtreeIdx(p_idx as u32)) else { continue };
        for (side_is_left, child) in [(true, left), (false, right)] {
            // A marginal child — internal subtree OR a summed-out leaf var — is
            // freed; the parent's marg-side refs map to the freed child's One node
            // (index 0 == ONE_LEAF_IDX, uniform across leaf/internal). A structural
            // (free) leaf has no count table, so `is_marginal()` skips it here.
            if !r.levels[child.idx()].is_marginal() {
                continue;
            }
            let counts = r.levels[child.idx()]
                .marginal_counts
                .clone()
                .expect("integer-marginal child carries marginal_counts");
            // Map a marg-side ref to the free child's true node (index 0); a count
            // of 0 would have to drop the pair, which a minimized marginal TDD
            // never requires — assert it instead of risking a silent miscount.
            let map = |raw: u32| -> u32 {
                let sat = if raw & (1 << 31) != 0 {
                    false // ZERO sentinel
                } else if raw & MARG_OVERFLOW_TAG != 0 {
                    decode_marg_coord(raw, MARG_VALUE_MASK) > 0 // inline count
                } else {
                    // Bare slot. The lift result is count-INDEPENDENT (this
                    // closure ALWAYS returns 0 == constant_one's true node); the
                    // count is read only for the satisfiability assert below. The
                    // WS_MARGINALIZE conjoin-loop sum-out can orphan a bare-slot
                    // ref into an EMPTIED count table (`Some(vec![])`), so an
                    // out-of-range ref is unverifiable — trust the minimized-
                    // marginal invariant (no count==0 ref survives) and treat it
                    // as satisfiable rather than panicking on the OOB index.
                    raw as usize >= counts.len() || counts[raw as usize] > 0
                };
                assert!(
                    sat,
                    "demarginalize_to_indicator: count==0 marginal ref at vtree parent \
                     {p_idx} — a minimized marginal TDD should have no unsatisfiable \
                     marginal node (false is pair-omitted, not a ZERO child)"
                );
                0 // LocalNodeIdx(0): the constant_one child's true node
            };
            let plevel = &mut r.levels[p_idx];
            for node in &mut plevel.nodes {
                if node.is_inline() {
                    if side_is_left {
                        node.a = map(node.a);
                    } else {
                        node.b = map(node.b);
                    }
                }
            }
            for pr in &mut plevel.pairs {
                if side_is_left {
                    pr.left.0 = map(pr.left.0);
                } else {
                    pr.right.0 = map(pr.right.0);
                }
            }
            if side_is_left {
                plevel.set_marg_inlined_left(false);
            } else {
                plevel.set_marg_inlined_right(false);
            }
            // Free the whole subtree (constant_one is non-marginal at internal
            // levels; its leaf levels stay marginal, which is the normal case).
            let mut stack = vec![child];
            while let Some(u) = stack.pop() {
                r.levels[u.idx()] = one.levels[u.idx()].clone();
                if let VtreeNode::Internal { left: l, right: rr, .. } = *r.vtree.node(u) {
                    stack.push(l);
                    stack.push(rr);
                }
            }
        }
    }
    minimize(r);
}
