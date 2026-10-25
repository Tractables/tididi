use super::*;
use crate::diagram::{NodeIdx, TddNodeId};

/// Each node of the root's left child made the output of a copy of its
/// diagram. Keying a level frees its children's keys, the output's level
/// among them once the root is keyed; the comparison reads the outputs'
/// keys after the last level.
#[test]
fn outputs_below_the_root_are_compared_by_their_keys() {
    let clauses = [[1, 2, 3], [-1, 4, 5], [2, -3, 6], [-4, -5, 1], [3, -6, -2], [-1, -2, 4]];
    let vtree = Arc::new(Vtree::balanced(6));
    let mut f = clauses.iter().fold(constant_one(&Engine::new(), &vtree), |f, clause| {
        apply_and(f, Tdd::clause(&vtree, *clause).unwrap())
    });
    f.minimize().unwrap();
    let below = vtree.children(vtree.root()).0;
    let n = f.levels[below.idx()].node_count() as u32;
    assert!(n >= 2, "the root's left child holds several nodes");
    let at = |i: u32| {
        Tdd::from_levels_unchecked(Arc::clone(&vtree), f.levels.clone().into_vec(), TddNodeId { vtree: below, local: NodeIdx(i) })
    };
    for i in 0..n {
        for j in 0..n {
            assert_eq!(at(i).equivalent(&at(j)), Ok(i == j), "nodes {i} and {j} of the minimized level");
        }
    }
}
