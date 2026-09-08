//! Structural satisfiability queries on compiled TDDs.

use crate::vtree::VtreeIdx;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use crate::diagram::*;

// ---------------------------------------------------------------------------

/// Check whether a TDD is satisfiable (has at least one model).
///
/// Requires a minimized TDD. After minimization, dead input pairs (pairs where
/// a child computes zero) have been removed, so an internal node with a
/// non-empty input set is guaranteed to have at least one satisfying assignment.
/// Checking the output node structurally is therefore O(1) and avoids the
/// O(size × `BigUint`) cost of `model_count`.
pub fn is_sat_minimized(f: &Tdd) -> bool {
    // ZERO sentinel means the TDD computes the constant-false function.
    if f.is_zero() {
        return false;
    }
    let out_vtree = f.output.vtree;
    if f.vtree.node(out_vtree).is_leaf() {
        // Implicit leaf: any index in {One=0, Pos=1, Neg=2} is satisfiable.
        true
    } else {
        let out_level = &f.levels[out_vtree.idx()];
        let out_node = &out_level.nodes[f.output.local.idx()];
        out_level.pairs_iter_of(out_node).next().is_some()
    }
}

/// True iff the TDD's output node is satisfiable (has ≥1 model), computed by a full
/// boolean bottom-up pass — the satisfiability complement of [`model_count_hybrid`].
///
/// Unlike [`is_sat_minimized`], which is O(1) but *assumes a reduced/minimized diagram* (output
/// node has a pair ⟹ satisfiable), this performs the same O(|D|) traversal as the
/// model counter with every count collapsed to a single bit (`> 0`). It is therefore
/// correct even on a NON-canonical diagram whose output node has pairs that all bottom
/// out in zero-count children — the structurally-false-but-not-`ZERO` state that an
/// apply can emit when a pass-through level copies a child that is satisfiable in
/// isolation but dead in the conjunction.
///
/// By construction it agrees with `model_count(tdd) > 0` on every input: same leaf
/// seeds (only `Zero` is unsatisfiable), same `resolve_marg_ref`/marginal handling,
/// boolean OR/AND in place of the counter's `+`/`×`. (See the `debug_assert` in
/// `apply_and_fallible` and the differential test in `query_tests.rs`.) So a caller may
/// collapse an unsatisfiable result to the `ZERO` sentinel without ever changing a
/// model count — restoring the [`Tdd::is_zero`]/[`is_sat_minimized`] invariant that downstream
/// applies rely on.
pub fn is_sat_structural(f: &Tdd) -> bool {
    if f.is_zero() {
        return false;
    }
    let vtree = &f.vtree;
    // sat[ti][i] = node i at vtree level ti has ≥1 model (count > 0).
    let mut sat: Vec<Vec<bool>> = (0..vtree.num_nodes())
        .map(|i| vec![false; f.effective_width(VtreeIdx(i as u32))])
        .collect();
    for (t, _var) in vtree.leaf_bottomup() {
        let ti = t.idx();
        for i in 0..LEAF_WIDTH {
            // Mirror model_count_hybrid's leaf seeds: only `Zero` is unsatisfiable
            // (and `Zero` is never stored at an implicit leaf level).
            sat[ti][i] = !matches!(LeafLabel::from_idx(i), LeafLabel::Zero);
        }
    }
    let (out_t, out_i) = (f.output.vtree.idx(), f.output.local.idx());
    for (t, left, right) in vtree.internal_bottomup() {
        let ti = t.idx();
        let (li, ri) = (left.idx(), right.idx());
        let level = &f.levels[ti];
        if level.is_marginal() {
            // A marginal slot is satisfiable iff its summed count is nonzero. OVERFLOW
            // (u128::MAX) is ≠ 0, so an overflowed (hence huge, > 0) count is satisfiable.
            let ic = level.marginal_counts.as_ref().unwrap();
            for (i, &c) in ic.iter().enumerate() {
                sat[ti][i] = c != 0;
            }
        } else {
            let li_view = f.levels[li].side_view();
            let ri_view = f.levels[ri].side_view();
            for (i, pairs) in level.internal_inputs_iter() {
                let mut ok = false;
                for pair in pairs {
                    let lc = match li_view.child(pair.left) {
                        ChildRef::Value(ValueRef::Inline(c)) => c != 0,
                        ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => { let idx = idx as usize; sat[li][idx] },
                    };
                    if !lc {
                        continue;
                    }
                    let rc = match ri_view.child(pair.right) {
                        ChildRef::Value(ValueRef::Inline(c)) => c != 0,
                        ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => { let idx = idx as usize; sat[ri][idx] },
                    };
                    if rc {
                        ok = true;
                        break;
                    }
                }
                sat[ti][i] = ok;
            }
        }
        // The vtree is a tree: a node has exactly ONE parent, so its column has
        // exactly one consumer and is dead once that parent's column is complete
        // (a marginal parent reads its children not at all — it re-derives its
        // column from `marginal_counts`). Free it here so the live set is the
        // frontier, not every level at once. `out_t` is the one column read after
        // the walk (it is the root under the output-at-root invariant, hence never
        // a child here, but the walk does not rely on that).
        for c in [li, ri] {
            if c != out_t {
                sat[c] = Vec::new();
            }
        }
    }
    sat[out_t][out_i]
}
