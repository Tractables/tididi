use super::*;

fn pair(l: u32, r: u32) -> InputPair {
    InputPair { left: LocalNodeIdx(l), right: LocalNodeIdx(r) }
}

fn entry(c1: u32, c2: u32) -> ProductEntry {
    ProductEntry {
        c1_idx: C1NodeIdx(c1),
        c2_idx: C2NodeIdx(c2),
        prod_idx: ProdNodeIdx(0),
    }
}

/// The estimator's four counter arrays live in the pooled workspace instead
/// of four fresh `vec![0u32; k]`s, so the per-call re-zero is now
/// load-bearing: a wider level's counts sit under a narrower level's shorter
/// prefix and would be counted twice. Verdict on a dirty buffer must equal
/// the verdict on a fresh one.
#[test]
fn pooled_counters_are_rezeroed_between_levels() {
    // Shape A, 2 slots per child side: c1's parent node puts BOTH of its
    // refs on left-child 0, so the normal direction probes twice what the
    // swapped one does ⇒ swap.
    let mut c1_a = TddLevel::new();
    c1_a.push_internal_node(&[pair(0, 0), pair(0, 1)]);
    let mut c2_a = TddLevel::new();
    c2_a.push_internal_node(&[pair(0, 0)]);
    let pl_a = [entry(0, 0)];

    // Shape B, 1 slot per child side: both directions probe once ⇒ no swap.
    let mut c1_b = TddLevel::new();
    c1_b.push_internal_node(&[pair(0, 0)]);
    let mut c2_b = TddLevel::new();
    c2_b.push_internal_node(&[pair(0, 0)]);
    let pl_b = [entry(0, 0)];

    let mut fresh: Vec<u32> = Vec::new();
    let b_alone = estimate_scatter_direction(
        &mut fresh, &c1_b, &c2_b, &pl_b, &pl_b, 1, 1, 1, 1,
    ).expect("estimate on shape B");
    assert!(!b_alone, "symmetric level: the estimator must not swap");

    let mut pooled: Vec<u32> = Vec::new();
    let a_first = estimate_scatter_direction(
        &mut pooled, &c1_a, &c2_a, &pl_a, &pl_a, 2, 2, 2, 2,
    ).expect("estimate on shape A");
    assert!(a_first, "left-heavy level: the estimator must swap");

    // Same buffer, now holding A's counts beyond B's shorter prefix.
    let b_after_a = estimate_scatter_direction(
        &mut pooled, &c1_b, &c2_b, &pl_b, &pl_b, 1, 1, 1, 1,
    ).expect("estimate on shape B after A");
    assert_eq!(
        b_after_a, b_alone,
        "pooled counters leaked a wider level's residue into a narrower one",
    );
}
