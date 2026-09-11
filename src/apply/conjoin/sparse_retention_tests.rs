//! Bucket-array retention triggers on retained bytes, not outer-row count: a
//! few-but-fat bucket array — a handful of outer rows each parking a
//! product-list-sized inner `Vec` — has a footprint far above the retention
//! cap while its length stays small. These lock in the byte trigger.
use super::drop_if_large;
use crate::limits::pool::SCRATCH_RETAIN_BYTES;

#[test]
fn releases_few_but_fat_rows() {
    // 2 outer rows, each with capacity for enough (u32,u32) entries that the
    // pair exceeds the byte limit. The outer length is 2, so a length-based
    // trigger would keep this array; the byte trigger must drop it.
    // `with_capacity` reserves without faulting pages in (len stays 0), so the
    // test's real RSS is tiny.
    let per_row = SCRATCH_RETAIN_BYTES
        / (2 * std::mem::size_of::<(u32, u32)>())
        + 1;
    let mut v: Vec<Vec<(u32, u32)>> =
        vec![Vec::with_capacity(per_row), Vec::with_capacity(per_row)];
    assert!(v.len() < 16_384, "precondition: a small outer length");
    drop_if_large(&mut v);
    assert_eq!(v.capacity(), 0, "few-but-fat array must be released on bytes");
}

#[test]
fn retains_many_small_rows() {
    // Many small rows whose total footprint stays well under the limit must
    // be RETAINED (capacity unchanged) so the amortized reuse is preserved.
    let mut v: Vec<Vec<(u32, u32)>> =
        (0..1000).map(|_| Vec::with_capacity(4)).collect();
    let before = v.capacity();
    drop_if_large(&mut v);
    assert_eq!(v.capacity(), before, "small array must be retained");
    assert_eq!(v.len(), 1000);
}
