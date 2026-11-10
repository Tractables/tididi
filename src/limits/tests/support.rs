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

    /// Whether the armed injection is still waiting: fewer reserves were
    /// asked for since [`refuse_nth_reserve`](Self::refuse_nth_reserve) than
    /// it grants before its refusal.
    pub(crate) fn refusal_pending(&self) -> bool {
        self.refuse_after.get().is_some()
    }

    /// Pin the post-conjunction walks' poll stride, returning the prior pin.
    pub(crate) fn pin_reduce_poll_stride(&self, stride: Option<u64>) -> Option<u64> {
        self.poll_stride_pin.replace(stride)
    }

    /// Pin a level width cap standing in for the one a `NodeIdx` imposes, so a
    /// test reaches the refusal without filling 31 bits of index.
    pub(crate) fn pin_level_width_cap(&self, cap: Option<usize>) {
        self.width_cap_pin.set(cap);
    }

    /// Pin a slot cap standing in for the one the marginal reference
    /// encoding imposes, so a test reaches the refusal without a column of
    /// 2^30 values.
    pub(crate) fn pin_slot_cap(&self, cap: Option<usize>) {
        self.slot_cap_pin.set(cap);
    }

    /// Pin a list cap standing in for the 32 bits of [`Limits::list_index`],
    /// so a test reaches the refusal without a list of 2^32 items.
    pub(crate) fn pin_list_cap(&self, cap: Option<usize>) {
        self.list_cap_pin.set(cap);
    }

    /// Run `op` under each cap `pin` sets, 0, 1, 2, 4 and on, until it goes
    /// through, and return what it built and how many caps refused it. A
    /// refusal must be [`OperationError::IndexOverflow`].
    pub(crate) fn through_caps<T>(
        &self,
        pin: fn(&Limits, Option<usize>),
        mut op: impl FnMut() -> Result<T, OperationError>,
    ) -> (T, u32) {
        let (mut cap, mut refused) = (0, 0);
        loop {
            pin(self, Some(cap));
            let got = op();
            pin(self, None);
            match got {
                Ok(built) => return (built, refused),
                Err(OperationError::IndexOverflow) => refused += 1,
                Err(e) => panic!("refused with {e:?} under a cap of {cap}"),
            }
            assert!(cap < 1 << 24, "refused under every cap");
            cap = (cap * 2).max(1);
        }
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
