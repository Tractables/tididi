//! P5: bucket-array retention must trigger on retained BYTES, not outer-row
//! count. The prior length-based `drop_if_large` (len > 16 384) kept a
//! few-but-fat bucket array — a handful of outer rows each parking a
//! product-list-sized inner Vec — even though its footprint dwarfed the
//! 32 MiB arena policy. These lock in the byte trigger.
use super::{drop_if_large, SPARSE_BUCKET_BYTE_LIMIT};

#[test]
fn releases_few_but_fat_rows() {
    // 2 outer rows, each with capacity for enough (u32,u32) entries that the
    // pair exceeds the byte limit (2·cap·8 B > 32 MiB). Outer length is 2 —
    // far under the old 16 384 length cap — so the length trigger would KEEP
    // this; the byte trigger must DROP it. `with_capacity` reserves without
    // faulting pages in (len stays 0), so the test's real RSS is tiny.
    let per_row = SPARSE_BUCKET_BYTE_LIMIT
        / (2 * std::mem::size_of::<(u32, u32)>())
        + 1;
    let mut v: Vec<Vec<(u32, u32)>> =
        vec![Vec::with_capacity(per_row), Vec::with_capacity(per_row)];
    assert!(v.len() < 16_384, "precondition: outer length below the old cap");
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
