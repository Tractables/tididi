use super::*;
use crate::Engine;
use crate::apply::conjoin::setup::{LevelShape, OperandWidths};
use crate::test_helpers::pair;
use crate::vtree::VtreeIdx;

fn shape(f: [usize; 3], g: [usize; 3]) -> LevelShape {
    LevelShape {
        t: VtreeIdx(0), left: VtreeIdx(1), right: VtreeIdx(2),
        f: OperandWidths { here: f[0], left: f[1], right: f[2] },
        g: OperandWidths { here: g[0], left: g[1], right: g[2] },
    }
}

fn entry(f: u32, g: u32) -> ProductEntry {
    ProductEntry {
        f_idx: FNodeIdx(f),
        g_idx: GNodeIdx(g),
        prod_idx: ProductNodeIdx(0),
    }
}

/// A level of `parents` nodes, node `k` holding the pairs `pairs(k)`.
fn level(parents: u32, pairs: impl Fn(u32) -> Vec<crate::diagram::ChildPair>) -> TddLevel {
    let mut level = TddLevel::new();
    for k in 0..parents {
        level.push_internal_node(&pairs(k));
    }
    level
}

/// g free over the right child keeps every one of its pairs under the one
/// right key, so keying the outer loop by the right child would rebuild the
/// whole g level for every outer key, while keying it by the left child
/// walks one g pair per outer key. The candidates the emit produces are the
/// same either way, so only the work around it can decide, and it says
/// swap.
#[test]
fn g_free_over_the_right_child_keys_the_outer_loop_by_the_left() {
    let eng = Engine::new();
    // f: four parents, parent k pairing its left child k with every right
    // child. g: four parents, parent k pairing left child k with the one
    // right child.
    let f = level(4, |k| (0..4).map(|j| pair(k, j)).collect());
    let g = level(4, |k| vec![pair(k, 0)]);
    let pl_left: Vec<_> = (0..4).map(|k| entry(k, k)).collect();
    let pl_right: Vec<_> = (0..4).map(|j| entry(j, 0)).collect();
    let mut counts = Vec::new();
    let swap = estimate_scatter_direction(
        &eng, &mut counts, &f, &g, &pl_left, &pl_right, shape([4, 4, 4], [4, 4, 1]),
    ).expect("estimate").swapped;
    assert!(swap, "a g free over the right child must key the outer loop by the left child");
}

/// The mirror image on the f side: f free over the right child puts every f
/// pair under the one right key, so keying by the left child would walk
/// every right product for every f pair. The emit's candidates are again
/// the same either way; the walk decides, and it says do not swap.
#[test]
fn f_free_over_the_right_child_keys_the_outer_loop_by_the_right() {
    let eng = Engine::new();
    let f = level(4, |k| vec![pair(k, 0)]);
    let g = level(4, |k| (0..4).map(|j| pair(k, j)).collect());
    let pl_left: Vec<_> = (0..4).map(|k| entry(k, k)).collect();
    let pl_right: Vec<_> = (0..4).map(|j| entry(0, j)).collect();
    let mut counts = Vec::new();
    let swap = estimate_scatter_direction(
        &eng, &mut counts, &f, &g, &pl_left, &pl_right, shape([4, 4, 1], [4, 4, 4]),
    ).expect("estimate").swapped;
    assert!(!swap, "an f free over the right child must key the outer loop by the right child");
}
