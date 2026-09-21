use super::*;

mod child_grid_mul;

/// Read both ends of each row, including a slab offset and the final cell.
#[test]
fn dense_and_marginal_structural_lookups_read_the_same_slab() {
    let slab: Vec<u32> = (0..17).map(|i| 100 + i).collect();
    let dense = DenseLookup { base: 2, stride: 5 };
    let marginal = MarginalLookup { base: 2, stride: 5, passthrough: false, pt_c1: false };
    for row in 0..3 {
        for col in 0..5 {
            let expected = slab[2 + row as usize * 5 + col as usize];
            assert_eq!(dense.get_in_row(&slab, dense.row(row), col), expected);
            assert_eq!(marginal.get_in_row(&slab, marginal.row(row), col), expected);
        }
    }
}

/// Carried marginal values are payloads, even when they resemble huge indices.
#[test]
fn marginal_passthrough_never_reads_the_grid() {
    for pt_c1 in [false, true] {
        let lookup = MarginalLookup { base: 99, stride: 99, passthrough: true, pt_c1 };
        assert!(lookup.passthrough());
        assert_eq!(lookup.get(&[], u32::MAX, 1 << 30), if pt_c1 { u32::MAX } else { 1 << 30 });
    }
}
