use super::*;

/// A child level can exceed 2^32 cells, so the row-offset multiply must widen
/// both factors to `usize` first: 70_000 * 70_000 exceeds `u32::MAX`, and a
/// `u32` product would wrap to 605_032_704.
#[test]
fn child_grid_mul_does_not_wrap_past_u32() {
    let a: u32 = 70_000;
    let right_width: u32 = 70_000;
    assert_eq!(child_grid_mul(a, right_width), 4_900_000_000usize);
    let wrapped = a.wrapping_mul(right_width) as usize;
    assert_eq!(wrapped, 605_032_704usize);
    assert_ne!(child_grid_mul(a, right_width), wrapped);
}

/// Exactness at the 2^32 boundary in both directions.
#[test]
fn child_grid_mul_exact_around_boundary() {
    assert_eq!(child_grid_mul(u32::MAX, 1), 4_294_967_295usize);
    assert_eq!(child_grid_mul(1, u32::MAX), 4_294_967_295usize);
    assert_eq!(child_grid_mul(65_536, 65_536), 4_294_967_296usize); // exactly 2^32
    assert_eq!(child_grid_mul(0, u32::MAX), 0usize);
}
