use super::{build_live_cols_bitmask, build_pair_row_columns, build_reach_columns, build_reach_masks, build_row_columns, bucket_shift, PrefilterSideMasks, REACH_COLUMNS_MIN_WIDTH, ROW_COLUMNS_MAX_WIDTH, ROW_LIVE};
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

/// The reach columns hold bit `j` of mask bit `b` exactly where g node
/// `j`'s reach mask has bit `b`, over widths on either side of a word and
/// of the fewest g nodes that build them.
#[test]
fn the_reach_columns_are_the_reach_masks_by_bit() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x3c51_9e07);
    let mut columns = Vec::new();
    for width in [1, 31, 32, 63, 64, 65, 200] {
        let reach: Vec<u128> = (0..width)
            .map(|_| (0..3).fold(0u128, |m, _| m | 1u128 << rng.below(100)))
            .collect();
        build_reach_columns(&eng, &reach, &mut columns).unwrap();
        if width < REACH_COLUMNS_MIN_WIDTH {
            assert!(columns.is_empty(), "width {width}");
            continue;
        }
        let words = width.div_ceil(64);
        let bits = columns.len() / words;
        assert_eq!(bits, 128 - reach.iter().fold(0u128, |a, &r| a | r).leading_zeros() as usize, "width {width}");
        for b in 0..128 {
            for (j, &r) in reach.iter().enumerate() {
                let set = b < bits && columns[b * words + j / 64] >> (j % 64) & 1 == 1;
                assert_eq!(set, r >> b & 1 == 1, "width {width}, bit {b}, node {j}");
            }
        }
    }
}

/// Each child row's word holds the g nodes whose reach meets its live
/// columns, and the flag where it has any: over widths up to the most the
/// words take and one past, and rows from no live bit to more live bits
/// than g nodes, so both ways of building a word are taken.
#[test]
fn the_row_columns_are_the_reach_masks_read_through_each_row() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5a0d_c3e1);
    let mut row_cols = Vec::new();
    for width in [1, 2, 7, 31, 62, 63, 64] {
        let reach: Vec<u128> = (0..width)
            .map(|_| (0..1 + rng.below(3)).fold(0u128, |m, _| m | 1u128 << rng.below(100)))
            .collect();
        let live_cols: Vec<u128> = (0..200)
            .map(|_| (0..rng.below(120)).fold(0u128, |m, _| m | 1u128 << rng.below(128)))
            .collect();
        build_row_columns(&eng, &live_cols, &reach, &mut row_cols).unwrap();
        if width > ROW_COLUMNS_MAX_WIDTH {
            assert!(row_cols.is_empty(), "width {width}");
            continue;
        }
        assert_eq!(row_cols.len(), live_cols.len(), "width {width}");
        for (a, (&word, &live)) in row_cols.iter().zip(&live_cols).enumerate() {
            assert_eq!(word & ROW_LIVE != 0, live != 0, "width {width}, row {a}");
            for (j, &r) in reach.iter().enumerate() {
                assert_eq!(word >> j & 1 == 1, live & r != 0, "width {width}, row {a}, node {j}");
            }
            assert_eq!(word & !ROW_LIVE, word & ((1u64 << width) - 1) & !ROW_LIVE, "width {width}, row {a}: no bit past the g nodes");
        }
    }
}

/// The row columns of a level read by pairs, read off its child grid's
/// bits, are the row columns of the rows' live column masks: over child
/// widths up to a word and none, levels of no pair to past the pairs read
/// one at a time, pairs sharing a child, rows from all dead to all alive,
/// and a grid that does not start the slab.
#[test]
fn the_pair_row_columns_are_the_row_columns_of_the_live_masks() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x6b1f_2d93);
    let (mut out, mut want_cols) = (PrefilterSideMasks::default(), Vec::new());
    for width in 0..=64 {
        for pairs in [0usize, 1, 2, 3, 4, 5, 9, 40] {
            if width == 0 && pairs > 0 {
                continue;
            }
            for rows in [0, 1, 3, 64, 65] {
                let alive = rng.below(5);
                let base = rng.below(7) as usize;
                let slab: Vec<u32> = (0..base + rows * width)
                    .map(|c| if c >= base && rng.below(4) < alive { c as u32 } else { NO_PRODUCT })
                    .collect();
                let reach: Vec<u128> = (0..pairs).map(|_| 1u128 << rng.below(width as u64)).collect();
                build_live_cols_bitmask(&eng, rows, width, base, &slab, &mut out, 0).unwrap();
                let live_cols = out.live_cols.clone();
                build_row_columns(&eng, &live_cols, &reach, &mut want_cols).unwrap();
                out.reach.clone_from(&reach);
                build_pair_row_columns(&eng, rows, width, base, &slab, &mut out).unwrap();
                assert_eq!(out.row_cols, want_cols, "width {width}, pairs {pairs}, rows {rows}");
                assert!(out.live_cols.is_empty(), "width {width}, pairs {pairs}, rows {rows}");
            }
        }
    }
}
