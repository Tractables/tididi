use super::bucket_shift;

#[test]
fn bucket_shift_bounds() {
    assert_eq!(bucket_shift(1), 0);
    assert_eq!(bucket_shift(128), 0);
    assert_eq!(bucket_shift(129), 1);
    assert_eq!(bucket_shift(256), 1);
    assert_eq!(bucket_shift(257), 2);
    for right_width in [1usize, 2, 127, 128, 129, 255, 256, 257, 4096, 4097, 1 << 20] {
        let s = bucket_shift(right_width);
        // Bucket count fits in a u128 mask...
        assert!((right_width - 1) >> s < 128, "right_width={right_width} s={s}");
        // ...and the shift is minimal.
        if s > 0 {
            assert!((right_width - 1) >> (s - 1) >= 128, "right_width={right_width} s={s} not minimal");
        }
    }
}
