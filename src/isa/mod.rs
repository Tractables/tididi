//! A second copy of hot loops compiled with the x86-64 vector and
//! bit-manipulation features (`avx2`, `fma`, `bmi1`, `bmi2`, `lzcnt` and
//! `popcnt`), run where the processor has them. The [`avx2_fn`] macro
//! compiles a function with those features; code built without them calls
//! it through an [`Avx2`], which only [`Avx2::detect`] makes, so the
//! binary runs on any x86-64 processor. Rust never fuses a multiply and an
//! add on its own, so both copies compute the same floats.

/// Proof that the processor has the features [`avx2_fn`] enables.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx2(());

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl Avx2 {
    /// An `Avx2` where the processor has the features, none otherwise or
    /// in a test that asks for the target's own instructions alone.
    #[inline(always)]
    pub(crate) fn detect() -> Option<Self> {
        #[cfg(test)]
        if tests::BASE_ISA.with(std::cell::Cell::get) {
            return None;
        }
        let has = std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("popcnt");
        has.then_some(Avx2(()))
    }
}

/// Compile the function `$f` with the features [`Avx2::detect`] checks,
/// on `x86_64` outside Miri (elsewhere `$f` does not exist). Code compiled
/// without them calls `$f` in an `unsafe` block, sound once an [`Avx2`]
/// exists.
macro_rules! avx2_fn {
    ($f:item) => {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        #[target_feature(enable = "avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
        $f
    };
}
pub(crate) use avx2_fn;

#[cfg(test)]
pub(crate) mod tests;
