use super::*;

#[test]
fn return_bounds_fanout_and_checkout_clears_retained_lists() {
    let eng = crate::Engine::new();
    let allocation;
    {
        let mut scratch = eng.scratch.restructure.checkout(eng.limits());
        scratch.per_v_pairs = (0..PER_V_PAIRS_RETAIN + 1)
            .map(|_| vec![ChildPair::new(NodeIdx(0), NodeIdx(0))])
            .collect();
        allocation = scratch.per_v_pairs[0].as_ptr();
    }
    let scratch = eng.scratch.restructure.checkout(eng.limits());
    assert_eq!(scratch.per_v_pairs.len(), PER_V_PAIRS_RETAIN);
    assert!(scratch.per_v_pairs.iter().all(Vec::is_empty));
    assert_eq!(scratch.per_v_pairs[0].as_ptr(), allocation);
}
