use super::*;
use crate::diagram::NodeIdx;

#[test]
fn checkout_clears_retained_buffers_and_keeps_their_allocation() {
    let eng = crate::Engine::new();
    let allocation;
    {
        let mut scratch = eng.scratch.restructure.checkout(&eng);
        scratch.outer_pairs = vec![ChildPair::new(NodeIdx(0), NodeIdx(0)); 16];
        scratch.outer_ends = vec![16];
        allocation = scratch.outer_pairs.as_ptr();
    }
    let scratch = eng.scratch.restructure.checkout(&eng);
    assert!(scratch.outer_pairs.is_empty() && scratch.outer_ends.is_empty());
    assert_eq!(scratch.outer_pairs.as_ptr(), allocation);
}
