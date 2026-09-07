use super::bucket_shift;

#[test]
fn bucket_shift_bounds() {
    assert_eq!(bucket_shift(1), 0);
    assert_eq!(bucket_shift(128), 0);
    assert_eq!(bucket_shift(129), 1);
    assert_eq!(bucket_shift(256), 1);
    assert_eq!(bucket_shift(257), 2);
    for k2 in [1usize, 2, 127, 128, 129, 255, 256, 257, 4096, 4097, 1 << 20] {
        let s = bucket_shift(k2);
        // Bucket count fits in a u128 mask...
        assert!((k2 - 1) >> s < 128, "k2={k2} s={s}");
        // ...and the shift is minimal.
        if s > 0 {
            assert!((k2 - 1) >> (s - 1) >= 128, "k2={k2} s={s} not minimal");
        }
    }
}
