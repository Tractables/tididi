use super::mix64;

/// Pins the finalizer's output on a fixed vector. Every twin fingerprint,
/// context or content, keys its hash table on this function, so a drift in
/// constants or shift schedule would silently re-shuffle every fingerprint
/// distribution.
#[test]
fn mix64_matches_reference_splitmix64_finalizer() {
    assert_eq!(mix64(0x0000000000000000), 0x0000000000000000);
    assert_eq!(mix64(0x0000000000000001), 0x5692161d100b05e5);
    assert_eq!(mix64(0xFFFFFFFFFFFFFFFF), 0xb4d055fcf2cbbd7b);
    assert_eq!(mix64(0x0123456789ABCDEF), 0xb2c058e4ebb5112c);
    // A packed context entry: (parent_i=3, sibling_j=7).
    assert_eq!(mix64(0x0000000300000007), 0x08070fade3326d87);
}
