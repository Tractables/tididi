//! The failure injections a test arms on a `Limits`.

use super::*;

impl Limits {
    /// Arm the allocation-failure injection to refuse the `(n+1)`-th reserve
    /// this `Limits` is asked for: the next `n` are granted, the one after is
    /// refused, and the injection disarms itself.
    pub(crate) fn refuse_nth_reserve(&self, n: u32) {
        self.refuse_after.set(Some(n));
    }

    /// Disarm the allocation-failure injection.
    pub(crate) fn grant_every_reserve(&self) {
        self.refuse_after.set(None);
    }

    /// Pin the post-conjunction walks' poll stride, returning the prior pin.
    pub(crate) fn pin_reduce_poll_stride(&self, stride: Option<u64>) -> Option<u64> {
        self.poll_stride_pin.replace(stride)
    }
}

impl Limits {
    /// Whether this reserve triggers the armed failure injection.
    #[inline(always)]
    pub(crate) fn refuses_reserve(&self) -> bool {
        match self.refuse_after.get() {
            None => false,
            Some(n) => self.count_down_refusal(n),
        }
    }

    /// Countdown arm of [`Limits::refuses_reserve`], reached only while the
    /// injection is armed.
    #[cold]
    #[inline(never)]
    fn count_down_refusal(&self, n: u32) -> bool {
        self.refuse_after.set(n.checked_sub(1));
        n == 0
    }

}
