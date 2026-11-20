use super::*;

/// Word records and wide records give the order a comparison sort of
/// the keys gives, small and past the radix threshold.
#[test]
fn stored_order_sorts_keys() {
    let lim = Limits::new();
    let mut s = OrderScratch::default();
    for (n, right, left) in [(5u64, 7u64, 11u64), (5000, 997, 4099), (5000, 1 << 31, 1 << 31), (40, 1 << 31, 1 << 31)] {
        let keys: Vec<u64> = (0..n).map(|i| ((i * 2654435761 % right) << 32) | ((i * 104729 + n - i) % left)).collect();
        let mut want: Vec<u32> = (0..n as u32).collect();
        want.sort_by_key(|&atom| keys[atom as usize]);
        assert!(keys.windows(2).any(|w| w[0] >= w[1]));
        let mut distinct = keys.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), keys.len());
        assert_eq!(stored_order(&lim, &mut s, keys.iter().copied()).unwrap(), want, "{n} {right} {left}");
    }
}
