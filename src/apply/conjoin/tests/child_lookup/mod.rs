use super::*;

// Its products pass 2^32, which a 32-bit `usize` does not hold.
#[cfg(target_pointer_width = "64")]
mod child_grid_mul;

/// Read both ends of each row, including a slab offset and the final cell.
#[test]
fn dense_and_marginal_structural_lookups_read_the_same_slab() {
    let slab: Vec<u32> = (0..17).map(|i| 100 + i).collect();
    let dense = DenseLookup { base: 2, stride: 5 };
    let marginal = MarginalLookup { base: 2, stride: 5, passthrough: false, carry_f: false };
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
    for carry_f in [false, true] {
        let lookup = MarginalLookup { base: 99, stride: 99, passthrough: true, carry_f };
        assert!(lookup.passthrough());
        assert_eq!(lookup.get(&[], u32::MAX, 1 << 30), if carry_f { u32::MAX } else { 1 << 30 });
    }
}

/// A structural read one cell past the slab fails at the read in a debug
/// build, for either lookup, rather than reading past it.
#[cfg(debug_assertions)]
#[test]
fn a_structural_read_past_the_slab_fails_in_debug() {
    let slab: Vec<u32> = (0..17).collect();
    let dense = DenseLookup { base: 2, stride: 5 };
    let marginal = MarginalLookup { base: 2, stride: 5, passthrough: false, carry_f: false };
    let past = |read: &dyn Fn() -> u32| {
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(read)).expect_err("a read past the slab fails");
        let message = failed.downcast_ref::<&str>().copied().unwrap_or_default();
        assert_eq!(message, "a grid read is past the child's slab");
    };
    past(&|| dense.get_in_row(&slab, dense.row(3), 0));
    past(&|| marginal.get_in_row(&slab, marginal.row(2), 5));
}
