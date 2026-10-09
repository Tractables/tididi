use super::{build_live_cols_bitmask, build_reach_masks, bucket_shift, PrefilterSideMasks};
use crate::apply::conjoin::NO_PRODUCT;
use crate::test_helpers::{with_stored_copy, x_decision_diagram};
use crate::vtree::rng::Lcg;
use crate::Engine;

/// Each row's mask, cut out of the grid's bits, is the mask the definition
/// gives: bit `b >> shift` set where some column `b` of the row's bucket
/// holds a product. Over every width the masks take, the words' and the
/// halves' edges among them, row counts that end a grid inside a word and
/// on its edge, grids from all dead to all alive, and a grid that does not
/// start the slab.
#[test]
fn the_live_column_masks_are_the_rows_alive_columns() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x1d1e_0015);
    let mut out = PrefilterSideMasks::default();
    for width in (1..=70).chain([126, 127, 128, 129, 200, 256, 300]) {
        for rows in [0, 1, 2, 3, 31, 64, 65] {
            let alive = rng.below(5);
            let base = rng.below(7) as usize;
            let slab: Vec<u32> = (0..base + rows * width)
                .map(|c| if c >= base && rng.below(4) < alive { c as u32 } else { NO_PRODUCT })
                .collect();
            let shift = bucket_shift(width);
            build_live_cols_bitmask(&eng, rows, width, base, &slab, &mut out, shift).unwrap();
            let want: Vec<u128> = (0..rows)
                .map(|a| (0..width).filter(|&b| slab[base + a * width + b] != NO_PRODUCT).fold(0u128, |m, b| m | 1u128 << (b >> shift)))
                .collect();
            assert_eq!(out.live_cols, want, "width {width}, rows {rows}");
        }
    }
}

/// Each g node's reach mask has the bits of the children its pairs name on
/// the side, an implicit level's read off its description as its stored
/// copy's are off its arena, under every shift.
#[test]
fn the_reach_masks_are_the_nodes_children() {
    let eng = Engine::new();
    let (implicit, v, _) = x_decision_diagram(32);
    let stored = with_stored_copy(&implicit, v);
    assert!(implicit.levels[v.idx()].implicit().is_some() && stored.levels[v.idx()].implicit().is_none());
    let nodes = stored.levels[v.idx()].nodes().len();
    let mut reach = Vec::new();
    for right in [false, true] {
        let side = |p: &crate::diagram::ChildPair| (if right { p.right } else { p.left }).raw() as usize;
        for shift in 0..3 {
            let want: Vec<u128> = (0..nodes)
                .map(|i| stored.levels[v.idx()].pairs_iter_of_idx(i).fold(0u128, |m, p| m | 1u128 << (side(&p) >> shift)))
                .collect();
            for tdd in [&implicit, &stored] {
                build_reach_masks(&eng, &tdd.levels[v.idx()], nodes, &mut reach, side, shift).unwrap();
                assert_eq!(reach, want, "right {right}, shift {shift}");
            }
        }
    }
}
