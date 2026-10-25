//! The sweep's set of pending parents against an ordered set.

use super::*;
use crate::vtree::rng::Lcg;
use std::collections::BTreeSet;

/// Inserted positions come back highest first, each once, as an ordered
/// set gives them, where each insert after the first pops is below the
/// position popped last, as a sweep's are; the set is empty after, and
/// starts again over a diagram of another size.
#[test]
fn pending_parents_pop_highest_first_once() {
    let eng = Engine::new();
    let lim = eng.limits();
    let mut rng = Lcg::new(41);
    let mut pending = Pending::default();
    for &num_nodes in &[1usize, 63, 64, 65, 4096, 4097, 20_000, 300] {
        pending.start(lim, num_nodes).unwrap();
        let mut model = BTreeSet::new();
        for _ in 0..rng.below(40) {
            let pos = rng.below(num_nodes as u64) as u32;
            pending.insert(pos);
            model.insert(pos);
        }
        while let Some(pos) = pending.pop() {
            assert_eq!(Some(pos), model.pop_last(), "{num_nodes} nodes");
            // A popped parent's children, some already pending.
            for _ in 0..rng.below(3) {
                if pos == 0 {
                    break;
                }
                let child = if rng.coin() && !model.is_empty() {
                    *model.iter().next_back().unwrap()
                } else {
                    rng.below(u64::from(pos)) as u32
                };
                pending.insert(child);
                model.insert(child);
            }
        }
        assert!(model.is_empty(), "{num_nodes} nodes: positions left out");
    }
}
