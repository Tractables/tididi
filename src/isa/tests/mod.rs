//! The switch a test turns to run the target's own instructions alone.

thread_local! {
    /// Whether [`Avx2::detect`](super::Avx2::detect) finds no features on
    /// this thread, as on a processor without `avx2`.
    pub(crate) static BASE_ISA: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
