//! Test-only helpers shared across `tdd::*` test modules and the downstream
//! crates' integration tests. These are `#[doc(hidden)]` pub rather than
//! `#[cfg(test)]`-gated because dependency crates never compile with
//! `cfg(test)`, so a downstream integration test could not otherwise reach
//! them across the crate boundary. Consequently they DO ship in the rlib —
//! `doc(hidden)` keeps them off the published API surface, not out of the binary.

use num_bigint::BigUint;

use crate::tdd::query::compute_node_counts;
use crate::tdd::types::{Tdd, assert_can_make_marginal};
use crate::vtree::{VtreeIdx, VtreeNode};

/// `BigUint` → u128, panicking if the value exceeds 128 bits. Used by tests
/// that feed `compute_node_counts` output into `make_marginal`, which
/// requires u128 counts.
pub fn big_to_u128(b: &BigUint) -> u128 {
    let digits = b.to_u64_digits();
    match digits.len() {
        0 => 0,
        1 => digits[0] as u128,
        2 => ((digits[1] as u128) << 64) | (digits[0] as u128),
        _ => panic!("test fixture count overflows u128: {b}"),
    }
}

/// Bottom-up marginalize every internal, non-marginal, width≥1 level in
/// the subtree rooted at `root` (inclusive). Counts are derived from the
/// current TDD shape via `compute_node_counts`. Mirrors production's
/// `marginalize_batch` + `cascade_marginalize` semantics for a single
/// subtree, without the streaming-marginal hooks.
pub fn marginalize_subtree(tdd: &mut Tdd, root: VtreeIdx) {
    let vtree = tdd.vtree.clone();
    let counts = compute_node_counts(tdd);
    for &t in vtree.bottomup_topo() {
        let ti = t.idx();
        let mut under = t == root;
        if !under {
            let mut cur = t;
            while let Some(p) = vtree.node(cur).parent() {
                if p == root {
                    under = true;
                    break;
                }
                cur = p;
            }
        }
        if !under {
            continue;
        }
        if matches!(*vtree.node(VtreeIdx(ti as u32)), VtreeNode::Leaf { .. }) || tdd.levels[ti].is_marginal() {
            continue;
        }
        let w = tdd.levels[ti].width();
        if w == 0 {
            continue;
        }
        let u128_counts: Vec<u128> = (0..w).map(|i| big_to_u128(&counts[ti][i])).collect();
        assert_can_make_marginal(&tdd.levels, &vtree, t);
        tdd.levels[ti].make_marginal(u128_counts, None);
    }
    // Emulate production marginalization, which tags every persisted marg-side
    // slot ref (bit 30) so the 0=inline decode invariant holds. Without this the
    // strict decode assert fires when a later reader hits a raw slot ref.
    crate::tdd::types::tag_all_marg_side_slots(tdd, None);
}
