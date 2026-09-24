//! The seeded generator behind [`Vtree::random`](super::Vtree::random) and
//! every randomized sweep in the crate.

/// A seeded generator: one stream per seed, the same on every platform and in
/// every release. A linear congruential step, with the low bits — which have
/// short periods — dropped.
#[derive(Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// The stream for `seed`.
    pub fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// The next draw. The top 31 bits of the state, so a caller may take the
    /// remainder by any small modulus without inheriting a short period.
    pub fn next_u64(&mut self) -> u64 {
        self.state =
            self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state >> 33
    }

    /// A draw in `0..n`, for `n` up to 2^31.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A 62-bit draw, two steps of the stream, for a range [`below`](Self::below)
    /// is too narrow for.
    ///
    /// Only the invariant checkers' signatures and the generators draw this
    /// wide, so it follows `test_helpers` out of a release build.
    #[cfg(any(test, debug_assertions, feature = "testing"))]
    pub fn wide(&mut self) -> u64 {
        (self.next_u64() << 31) | self.next_u64()
    }

    /// A fair coin. Drawn only by the generators and sweeps, which compile
    /// under the `testing` feature.
    #[cfg(any(test, feature = "testing"))]
    pub fn coin(&mut self) -> bool {
        self.next_u64().is_multiple_of(2)
    }
}
