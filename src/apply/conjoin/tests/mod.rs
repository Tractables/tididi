use super::*;
use super::identity::level_marginal_is_constant_true;

mod complete;
mod conjunction;
mod implicit;
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

thread_local! {
    /// Whether every conjunction level on this thread reads its child sides
    /// from the grid ([`grid_lookups`]).
    static GRID_LOOKUPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether every conjunction level on this thread writes its pairs
    /// ([`written_levels`]).
    static WRITTEN_LEVELS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread that read a complete child side by
    /// arithmetic: both sides, the left alone, the right alone; and the times
    /// a reserved arena charged the meter for a growth it did not make
    /// ([`complete_census`]).
    static COMPLETE: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
}

pub(super) fn grid_lookups_forced() -> bool {
    GRID_LOOKUPS.with(std::cell::Cell::get)
}

pub(super) fn written_levels_forced() -> bool {
    WRITTEN_LEVELS.with(std::cell::Cell::get)
}

pub(super) fn note_lookups(lookups: PlainLookups) {
    let kind = match lookups {
        PlainLookups::Complete { .. } => 0,
        PlainLookups::CompleteLeft => 1,
        PlainLookups::CompleteRight => 2,
        PlainLookups::Grid => return,
    };
    count_complete(kind);
}

pub(super) fn note_scheduled_charge() {
    count_complete(3);
}

fn count_complete(kind: usize) {
    COMPLETE.with(|c| {
        let mut census = c.get();
        census[kind] += 1;
        c.set(census);
    });
}

/// Run `f` with every conjunction level reading its child sides from the
/// grid, as if no child were complete: the oracle the arithmetic lookups are
/// checked against.
pub(super) fn grid_lookups<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            GRID_LOOKUPS.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(GRID_LOOKUPS.with(|c| c.replace(true)));
    f()
}

/// Run `f` with every conjunction level writing its pairs, none taking the
/// implicit route: the oracle the implicit route is checked against.
pub(super) fn written_levels<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            WRITTEN_LEVELS.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(WRITTEN_LEVELS.with(|c| c.replace(true)));
    f()
}

/// The levels on this thread so far that read a complete child side by
/// arithmetic: with both sides complete, the left alone, the right alone;
/// and the times a reserved arena charged the meter for a growth it did not
/// make.
pub(super) fn complete_census() -> [u64; 4] {
    COMPLETE.with(std::cell::Cell::get)
}
