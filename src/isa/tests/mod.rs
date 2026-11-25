//! The switch a test turns to run the target's own instructions alone, and
//! the identities the register hints must be.

thread_local! {
    /// Whether [`Avx2::detect`](super::Avx2::detect) finds no features on
    /// this thread, as on a processor without `avx2`.
    pub(crate) static BASE_ISA: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[test]
fn defined_here_is_the_slice_itself() {
    let v = [3_u32, 1, 4, 1, 5];
    for k in 0..=v.len() {
        let s = super::defined_here(&v[k..]);
        assert!(core::ptr::eq(s, &v[k..]));
        assert_eq!(s, &v[k..]);
    }
}
