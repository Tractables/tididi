use super::twin_table_size;

/// The probe loops have no empty-slot fallback: a table that can be filled
/// completely wedges the next probe for an absent fingerprint in an endless
/// wrap. Pin the strict inequality against the occupancy bound — and the 3/4
/// load cap that keeps linear probing's cluster length bounded — across the
/// whole shape of the formula (exact powers of two, just above, just below).
#[test]
fn table_is_strictly_larger_than_its_occupancy_bound() {
    for w in 0..4096usize {
        let t = twin_table_size(w);
        assert!(t.is_power_of_two(), "width {w}: mask probing needs a power-of-two table, got {t}");
        assert!(t > w, "width {w}: table {t} can fill completely — a probe would wrap forever");
        assert!(4 * w <= 3 * t, "width {w}: table {t} is past the 3/4 load cap");
    }
    // The point of the sizing: a width just above a power of two gets half
    // the table the former `2 · width` rule allocated.
    assert_eq!(twin_table_size(9), 16);
    assert_eq!(twin_table_size((1 << 20) + 1), 1 << 21);
}
