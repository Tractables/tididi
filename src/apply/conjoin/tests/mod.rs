use super::*;
use super::identity::level_marginal_is_constant_true;

mod complete;
mod conjunction;
mod implicit;
mod marginal_leaf_target;
mod marginal_level;
mod marginal_orphan;
mod marginal_subsumed;
mod one_sided;
mod owed;
mod relabel;
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
    /// The levels on this thread that read a complete child side by
    /// arithmetic: both sides, the left alone, the right alone; the times a
    /// reserved arena charged the meter for a growth it did not make; and
    /// the levels of two complete sides that read a lone g pair in every
    /// cell ([`complete_census`]).
    static COMPLETE: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
}

pub(super) fn grid_lookups_forced() -> bool {
    GRID_LOOKUPS.with(std::cell::Cell::get)
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

pub(super) fn note_lone_pair() {
    count_complete(4);
}

fn count_complete(kind: usize) {
    COMPLETE.with(|c| {
        let mut census = c.get();
        census[kind] += 1;
        c.set(census);
    });
}

thread_local! {
    /// The times a level's pairs arena outgrew its capacity on this thread
    /// ([`pairs_grown`]).
    static GROWN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn note_pairs_grown() {
    GROWN.with(|n| n.set(n.get() + 1));
}

/// The times a level's pairs arena outgrew its capacity on this thread so
/// far.
pub(super) fn pairs_grown() -> u64 {
    GROWN.with(std::cell::Cell::get)
}

thread_local! {
    /// Whether no level on this thread builds the dead-pair masks with one
    /// multi-pair operand ([`one_sided_masks_off`]).
    static ONE_SIDED_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread that built the masks with one multi-pair
    /// operand ([`one_sided_masked_levels`]).
    static ONE_SIDED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn one_sided_masks_forced_off() -> bool {
    ONE_SIDED_OFF.with(std::cell::Cell::get)
}

pub(super) fn note_one_sided_masks() {
    ONE_SIDED.with(|n| n.set(n.get() + 1));
}

/// Run `f` with no level building the dead-pair masks with one multi-pair
/// operand, each such level on the row loop without masks: the oracle the
/// one-sided masks are checked against.
pub(super) fn one_sided_masks_off<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            ONE_SIDED_OFF.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(ONE_SIDED_OFF.with(|c| c.replace(true)));
    f()
}

/// The levels on this thread so far that built the dead-pair masks with one
/// multi-pair operand.
pub(super) fn one_sided_masked_levels() -> u64 {
    ONE_SIDED.with(std::cell::Cell::get)
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

/// The levels on this thread so far that read a complete child side by
/// arithmetic: with both sides complete, the left alone, the right alone;
/// the times a reserved arena charged the meter for a growth it did not
/// make; and the levels of two complete sides that read a lone g pair in
/// every cell.
pub(super) fn complete_census() -> [u64; 5] {
    COMPLETE.with(std::cell::Cell::get)
}

/// Borrowed build-mode entry for tests of the sweep's consumption contract.
pub(crate) fn apply_and_fallible(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    targets: VtreeMask<'_>,
    quantified: VtreeMask<'_>,
    filter: Option<&mut dyn FnMut(VtreeIdx, NodeIdx, NodeIdx) -> bool>,
) -> Result<Tdd, OperationError> {
    apply_and_core(eng, f, g, targets, quantified, filter, ConjoinMode::Build, Operands::default()).map(Conjoined::diagram)
}

thread_local! {
    /// Whether the relabelling route is closed on this thread ([`no_relabel`]).
    static NO_RELABEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread the relabelling route took: moved whole,
    /// and rebuilt ([`relabel_census`]).
    static RELABELLED: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
}

pub(super) fn relabel_forced_off() -> bool {
    NO_RELABEL.with(std::cell::Cell::get)
}

pub(super) fn note_relabelled(moved: bool) {
    RELABELLED.with(|c| {
        let mut census = c.get();
        census[usize::from(!moved)] += 1;
        c.set(census);
    });
}

/// Run `f` with the relabelling route closed, every level it would take
/// going to the general routes: the oracle it is checked against.
pub(super) fn no_relabel<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            NO_RELABEL.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(NO_RELABEL.with(|c| c.replace(true)));
    f()
}

/// The levels on this thread so far the relabelling route moved whole and
/// rebuilt.
pub(super) fn relabel_census() -> [u64; 2] {
    RELABELLED.with(std::cell::Cell::get)
}
