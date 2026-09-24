use super::*;
use crate::Engine;
use crate::test_helpers::pair;

/// A square level shape: every one of the six widths is `k`.
fn square_shape(k: usize) -> crate::apply::conjoin::setup::LevelShape {
    use crate::vtree::VtreeIdx;
    use crate::apply::conjoin::setup::OperandWidths;
    crate::apply::conjoin::setup::LevelShape {
        t: VtreeIdx(0), left: VtreeIdx(1), right: VtreeIdx(2),
        f: OperandWidths { here: k, left: k, right: k },
        g: OperandWidths { here: k, left: k, right: k },
    }
}

fn entry(f: u32, g: u32) -> ProductEntry {
    ProductEntry {
        f_idx: FNodeIdx(f),
        g_idx: GNodeIdx(g),
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
    // refs on left-child 0, so the walk through the left products is twice
    // the walk through the right ones ⇒ swap.
    let mut left_a = TddLevel::new();
    left_a.push_internal_node(&[pair(0, 0), pair(0, 1)]);
    let mut right_a = TddLevel::new();
    right_a.push_internal_node(&[pair(0, 0)]);
    let pl_a = [entry(0, 0)];

    // Shape B, 1 slot per child side: both directions cost the same and the
    // children are equally wide ⇒ no swap.
    let mut left_b = TddLevel::new();
    left_b.push_internal_node(&[pair(0, 0)]);
    let mut right_b = TddLevel::new();
    right_b.push_internal_node(&[pair(0, 0)]);
    let pl_b = [entry(0, 0)];

    let mut fresh: Vec<u32> = Vec::new();
    let b_alone = estimate_scatter_direction(
        &eng,
        &mut fresh, &left_b, &right_b, &pl_b, &pl_b, square_shape(1),
    ).expect("estimate on shape B").swapped;
    assert!(!b_alone, "symmetric level: the estimator must not swap");

    let mut pooled: Vec<u32> = Vec::new();
    let a_first = estimate_scatter_direction(
        &eng,
        &mut pooled, &left_a, &right_a, &pl_a, &pl_a, square_shape(2),
    ).expect("estimate on shape A").swapped;
    assert!(a_first, "left-heavy level: the estimator must swap");

    // Same buffer, now holding A's counts beyond B's shorter prefix.
    let b_after_a = estimate_scatter_direction(
        &eng,
        &mut pooled, &left_b, &right_b, &pl_b, &pl_b, square_shape(1),
    ).expect("estimate on shape B after A").swapped;
    assert_eq!(
        b_after_a, b_alone,
        "pooled counters leaked a wider level's residue into a narrower one",
    );
}
