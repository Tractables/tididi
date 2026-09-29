use super::*;
use super::identity::level_marginal_is_constant_true;

mod conjunction;
mod marginal_leaf_target;
mod marginal_level;
mod marginal_orphan;
mod marginal_subsumed;
mod owed;
mod restoring;
mod self_conjunction;

mod target_completion;
mod workspace;

thread_local! {
    /// Whether `and_marginalizing` takes the two-step path on this thread
    /// ([`two_step`]).
    static TWO_STEP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The `and_marginalizing` calls on this thread whose root summed its
    /// target out ([`summed_roots`]).
    static SUMMED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn two_step_forced() -> bool {
    TWO_STEP.with(std::cell::Cell::get)
}

pub(super) fn note_summed() {
    SUMMED.with(|n| n.set(n.get() + 1));
}

/// Run `f` with `and_marginalizing` on the two-step path: the conjunction
/// built in full, then the target marginalized and the root's pairs fused.
pub(super) fn two_step<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            TWO_STEP.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(TWO_STEP.with(|c| c.replace(true)));
    f()
}

/// How many `and_marginalizing` calls on this thread summed their target
/// out at the root so far.
pub(super) fn summed_roots() -> u64 {
    SUMMED.with(std::cell::Cell::get)
}
