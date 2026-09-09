use super::*;
use crate::engine::Engine;

fn pair(l: u32, r: u32) -> InputPair {
    InputPair { left: NodeIdx(l), right: NodeIdx(r) }
}

/// A square level shape: every one of the six widths is `k`.
fn square_shape(k: usize) -> crate::apply::conjoin::setup::LevelShape {
    use crate::vtree::VtreeIdx;
    crate::apply::conjoin::setup::LevelShape {
        t: VtreeIdx(0), left: VtreeIdx(1), right: VtreeIdx(2),
        t_idx: 0, left_idx: 1, right_idx: 2,
        left_width: k, k1_left: k, k1_right: k,
        right_width: k, left_child_stride: k, right_child_stride: k,
    }
}

fn entry(f: u32, g: u32) -> ProductEntry {
    ProductEntry {
        left_idx: LeftNodeIdx(f),
        right_idx: RightNodeIdx(g),
        prod_idx: ProductNodeIdx(0),
    }
}

/// The estimator's four counter arrays live in the pooled workspace instead
/// of four fresh `vec![0u32; k]`s, so the per-call re-zero is now
/// load-bearing: a wider level's counts sit under a narrower level's shorter
/// prefix and would be counted twice. Verdict on a dirty buffer must equal
/// the verdict on a fresh one.
#[test]
fn pooled_counters_are_rezeroed_between_levels() {
    let eng = Engine::new();
    // Shape A, 2 slots per child side: f's parent node puts both of its
    // refs on left-child 0, so the normal direction probes twice what the
    // swapped one does ⇒ swap.
    let mut left_a = TddLevel::new();
    left_a.push_internal_node(&[pair(0, 0), pair(0, 1)]);
    let mut right_a = TddLevel::new();
    right_a.push_internal_node(&[pair(0, 0)]);
    let pl_a = [entry(0, 0)];

    // Shape B, 1 slot per child side: both directions probe once ⇒ no swap.
    let mut left_b = TddLevel::new();
    left_b.push_internal_node(&[pair(0, 0)]);
    let mut right_b = TddLevel::new();
    right_b.push_internal_node(&[pair(0, 0)]);
    let pl_b = [entry(0, 0)];

    let mut fresh: Vec<u32> = Vec::new();
    let b_alone = estimate_scatter_direction(
        &eng,
        &mut fresh, &left_b, &right_b, &pl_b, &pl_b, square_shape(1),
    ).expect("estimate on shape B");
    assert!(!b_alone, "symmetric level: the estimator must not swap");

    let mut pooled: Vec<u32> = Vec::new();
    let a_first = estimate_scatter_direction(
        &eng,
        &mut pooled, &left_a, &right_a, &pl_a, &pl_a, square_shape(2),
    ).expect("estimate on shape A");
    assert!(a_first, "left-heavy level: the estimator must swap");

    // Same buffer, now holding A's counts beyond B's shorter prefix.
    let b_after_a = estimate_scatter_direction(
        &eng,
        &mut pooled, &left_b, &right_b, &pl_b, &pl_b, square_shape(1),
    ).expect("estimate on shape B after A");
    assert_eq!(
        b_after_a, b_alone,
        "pooled counters leaked a wider level's residue into a narrower one",
    );
}
