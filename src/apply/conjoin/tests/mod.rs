use super::*;
use super::identity::level_marginal_is_constant_true;

mod complete;
mod conjunction;
mod implicit;
mod kept_counts;
mod marginal_leaf_target;
mod marginal_level;
mod marginal_orphan;
mod marginal_subsumed;
mod one_sided;
mod owed;
mod pair_rows;
mod relabel;
mod probe;
mod reach_columns;
mod restoring;
mod self_conjunction;
mod single_rows;
mod word_cells;

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
    /// reserved arena charged the meter for a growth it did not make; the
    /// levels of two complete sides that read a lone g pair in every cell;
    /// the levels of one complete side whose every column holds one g pair;
    /// and the rows of those written whole ([`complete_census`]).
    static COMPLETE: std::cell::Cell<[u64; 7]> = const { std::cell::Cell::new([0; 7]) };
    /// Whether every cell on this thread reads its column's pairs
    /// ([`column_reads`]).
    static COLUMN_READS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn lone_columns_forced_off() -> bool {
    COLUMN_READS.with(std::cell::Cell::get)
}

/// Run `f` with every cell reading its column's pairs, as if no level held
/// one pair a column: the oracle the levels that do are checked against.
pub(super) fn column_reads<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            COLUMN_READS.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(COLUMN_READS.with(|c| c.replace(true)));
    f()
}

pub(super) fn grid_lookups_forced() -> bool {
    GRID_LOOKUPS.with(std::cell::Cell::get)
}

pub(super) fn note_lookups(lookups: PlainLookups) {
    let kind = match lookups {
        PlainLookups::Complete { .. } => 0,
        PlainLookups::CompleteLeft { .. } => 1,
        PlainLookups::CompleteRight { .. } => 2,
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

pub(super) fn note_lone_columns() {
    count_complete(5);
}

pub(super) fn note_whole_row() {
    count_complete(6);
}

fn count_complete(kind: usize) {
    COMPLETE.with(|c| {
        let mut census = c.get();
        census[kind] += 1;
        c.set(census);
    });
}

thread_local! {
    /// The times a level's pairs arena grew on this thread, filled or ahead
    /// of a projection ([`pairs_grown`]).
    static GROWN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn note_pairs_grown() {
    GROWN.with(|n| n.set(n.get() + 1));
}

/// The times a level's pairs arena grew on this thread so far.
pub(super) fn pairs_grown() -> u64 {
    GROWN.with(std::cell::Cell::get)
}

thread_local! {
    /// The route each level built on this thread took, in build order, and
    /// the arena growths counted as its build began, while [`levels_built`]
    /// lists them.
    static ROUTES: std::cell::RefCell<Option<Vec<(VtreeIdx, Route, u64)>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn note_route(t: VtreeIdx, route: Route) {
    ROUTES.with(|r| {
        if let Some(routes) = r.borrow_mut().as_mut() {
            routes.push((t, route, pairs_grown()));
        }
    });
}

/// Run `f` and list each level it built, in build order, with its route and
/// the times its pairs arena grew.
pub(super) fn levels_built<R>(f: impl FnOnce() -> R) -> (R, Vec<(VtreeIdx, Route, u64)>) {
    struct Reset(Option<Vec<(VtreeIdx, Route, u64)>>);
    impl Drop for Reset {
        fn drop(&mut self) {
            ROUTES.with(|r| *r.borrow_mut() = self.0.take());
        }
    }
    let reset = Reset(ROUTES.with(|r| r.replace(Some(Vec::new()))));
    let out = f();
    let mut levels = ROUTES.with(|r| r.borrow_mut().take()).unwrap_or_default();
    drop(reset);
    // A level is noted as its build starts: its growths are those counted
    // before the next level's note, or before `f` returned.
    let mut next = pairs_grown();
    for (_, _, at) in levels.iter_mut().rev() {
        (*at, next) = (next - *at, *at);
    }
    (out, levels)
}

/// Run `f` and list the route each level it built took, in build order.
pub(super) fn routes_taken<R>(f: impl FnOnce() -> R) -> (R, Vec<(VtreeIdx, Route)>) {
    let (out, levels) = levels_built(f);
    (out, levels.into_iter().map(|(t, route, _)| (t, route)).collect())
}

thread_local! {
    /// Whether no level on this thread builds the dead-pair masks with one
    /// multi-pair operand ([`one_sided_masks_off`]).
    static ONE_SIDED_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread that built the masks with one multi-pair
    /// operand ([`one_sided_masked_levels`]).
    static ONE_SIDED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

thread_local! {
    /// Whether no level on this thread builds reach or row columns
    /// ([`reach_columns_off`]).
    static REACH_COLUMNS_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The rows on this thread whose live cells were read off the reach
    /// columns ([`reach_columns_rows`]).
    static REACH_COLUMNS_ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Whether no level on this thread builds row columns
    /// ([`row_columns_off`]).
    static ROW_COLUMNS_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The rows on this thread whose live cells were read off the row
    /// columns ([`row_columns_rows`]).
    static ROW_COLUMNS_ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Those of them that read a side whole ([`row_columns_rows`]).
    static ROW_COLUMNS_WHOLE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

thread_local! {
    /// Whether no level on this thread is read by pairs
    /// ([`pair_rows_off`]).
    static PAIR_ROWS_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread read by pairs, those of them with a
    /// complete side, and those of one pair a column
    /// ([`pair_rows_levels`]).
    static PAIR_ROWS: std::cell::Cell<[u64; 3]> = const { std::cell::Cell::new([0; 3]) };
}

pub(super) fn pair_rows_forced_off() -> bool {
    PAIR_ROWS_OFF.with(std::cell::Cell::get)
}

pub(super) fn note_pair_rows(complete: bool, one_pair: bool) {
    PAIR_ROWS.with(|n| {
        let [all, whole, one] = n.get();
        n.set([all + 1, whole + u64::from(complete), one + u64::from(one_pair)]);
    });
}

/// Run `f` with every masked level on the row loop of any level: the
/// oracle the levels read by pairs are checked against.
pub(super) fn pair_rows_off<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            PAIR_ROWS_OFF.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(PAIR_ROWS_OFF.with(|c| c.replace(true)));
    f()
}

/// The levels on this thread so far read by pairs, those of them with a
/// complete side, and those of one pair a column.
pub(super) fn pair_rows_levels() -> [u64; 3] {
    PAIR_ROWS.with(std::cell::Cell::get)
}

thread_local! {
    /// Whether every level on this thread that can read its cells by g-pair
    /// vectors does, or none ([`word_cells_on`], [`word_cells_off`]).
    static WORD_CELLS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// The levels on this thread read by g-pair vectors, those of two grids
    /// and those with a complete side, their cells walked in the grouped
    /// order, their rows read by vectors, those of these that walk their
    /// live cells alone, and those that write them off the sorted list of
    /// their live candidates; and the streaming targets' levels among the
    /// levels, and those read through marginal lookups; and the
    /// sparse-marginal levels ([`word_cells_levels`]).
    static WORD_CELLS_LEVELS: std::cell::Cell<[u64; 9]> = const { std::cell::Cell::new([0; 9]) };
}

/// [`WORD_CELLS`], or, where it is unset, `TIDIDI_TEST_WORD_CELLS` (`1`
/// every level that can, `0` none), which runs the whole suite either way.
pub(super) fn word_cells_forced() -> Option<bool> {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    WORD_CELLS.with(std::cell::Cell::get).or_else(|| {
        *ENV.get_or_init(|| std::env::var("TIDIDI_TEST_WORD_CELLS").ok().map(|v| v == "1"))
    })
}

pub(super) fn note_word_cells(what: super::WordNote) {
    WORD_CELLS_LEVELS.with(|n| {
        let mut counts = n.get();
        counts[what as usize] += 1;
        n.set(counts);
    });
}

fn word_cells_set<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            WORD_CELLS.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(WORD_CELLS.with(|c| c.replace(Some(on))));
    f()
}

/// Run `f` with every level that can read its cells by g-pair vectors
/// reading them, whether or not they pay.
pub(super) fn word_cells_on<R>(f: impl FnOnce() -> R) -> R {
    word_cells_set(true, f)
}

/// Run `f` with no level reading its cells by g-pair vectors: their oracle.
pub(super) fn word_cells_off<R>(f: impl FnOnce() -> R) -> R {
    word_cells_set(false, f)
}

thread_local! {
    /// Whether every row on this thread read by g-pair vectors that does
    /// not walk its live cells writes them off the sorted list of its live
    /// candidates ([`word_rows_sorted`]).
    static WORD_ROWS_SORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn word_rows_sorted_forced() -> bool {
    WORD_ROWS_SORTED.with(std::cell::Cell::get)
}

/// Run `f` with every row read by g-pair vectors that does not walk its
/// live cells writing them off the sorted list of its live candidates.
pub(super) fn word_rows_sorted<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            WORD_ROWS_SORTED.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(WORD_ROWS_SORTED.with(|c| c.replace(true)));
    f()
}

/// The levels on this thread so far read by g-pair vectors, those of two
/// grids and those with a complete side, their cells walked in the grouped
/// order, their rows read by vectors, those of these that walk their live
/// cells alone, and those that write them off the sorted list of their
/// live candidates; and the streaming targets' levels among the levels,
/// and those read through marginal lookups; and the sparse-marginal
/// levels.
pub(super) fn word_cells_levels() -> [u64; 9] {
    WORD_CELLS_LEVELS.with(std::cell::Cell::get)
}

thread_local! {
    /// Whether no level on this thread takes the loop of one-pair cells
    /// ([`single_rows_off`]).
    static SINGLE_ROWS_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread that loop built ([`single_rows_levels`]).
    static SINGLE_ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn single_rows_forced_off() -> bool {
    SINGLE_ROWS_OFF.with(std::cell::Cell::get)
}

pub(super) fn note_single_rows() {
    SINGLE_ROWS.with(|n| n.set(n.get() + 1));
}

/// Run `f` with every level of one-pair nodes on the row loop of any
/// level: the oracle the loop of one-pair cells is checked against.
pub(super) fn single_rows_off<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            SINGLE_ROWS_OFF.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(SINGLE_ROWS_OFF.with(|c| c.replace(true)));
    f()
}

/// The levels on this thread so far that the loop of one-pair cells built.
pub(super) fn single_rows_levels() -> u64 {
    SINGLE_ROWS.with(std::cell::Cell::get)
}

pub(super) fn row_columns_forced_off() -> bool {
    ROW_COLUMNS_OFF.with(std::cell::Cell::get)
}

pub(super) fn note_row_columns_row(whole: bool) {
    ROW_COLUMNS_ROWS.with(|n| n.set(n.get() + 1));
    ROW_COLUMNS_WHOLE.with(|n| n.set(n.get() + u64::from(whole)));
}

/// Run `f` with no level building row columns: a masked level of few g
/// nodes reads its rows' live columns off the reach masks.
pub(super) fn row_columns_off<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            ROW_COLUMNS_OFF.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(ROW_COLUMNS_OFF.with(|c| c.replace(true)));
    f()
}

/// The rows on this thread so far whose live cells were read off the row
/// columns, and those of them that read a side whole.
pub(super) fn row_columns_rows() -> (u64, u64) {
    (ROW_COLUMNS_ROWS.with(std::cell::Cell::get), ROW_COLUMNS_WHOLE.with(std::cell::Cell::get))
}

pub(super) fn reach_columns_forced_off() -> bool {
    REACH_COLUMNS_OFF.with(std::cell::Cell::get)
}

pub(super) fn note_reach_columns_row() {
    REACH_COLUMNS_ROWS.with(|n| n.set(n.get() + 1));
}

/// Run `f` with no level building reach or row columns, every masked row
/// testing its columns' reach masks one by one: the oracle the reach and
/// row columns are checked against.
pub(super) fn reach_columns_off<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            REACH_COLUMNS_OFF.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(REACH_COLUMNS_OFF.with(|c| c.replace(true)));
    f()
}

/// The rows on this thread so far whose live cells were read off the
/// reach columns.
pub(super) fn reach_columns_rows() -> u64 {
    REACH_COLUMNS_ROWS.with(std::cell::Cell::get)
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
/// make; the levels of two complete sides that read a lone g pair in every
/// cell; and the levels of one complete side whose every column holds one g
/// pair.
pub(super) fn complete_census() -> [u64; 7] {
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
    /// Those of them that read through a marginal child ([`read_through_census`]).
    static READ_THROUGH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The nodes the route wrote on this thread holding a pair twice
    /// ([`repeated_pair_census`]).
    static REPEATED_PAIRS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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

pub(super) fn note_read_through() {
    READ_THROUGH.with(|c| c.set(c.get() + 1));
}

/// The levels on this thread so far the relabelling route took that read
/// through a marginal child.
pub(super) fn read_through_census() -> u64 {
    READ_THROUGH.with(std::cell::Cell::get)
}

/// Note the pairs the route just wrote for one node, `pairs`, when two of
/// them are one pair.
pub(super) fn note_written_pairs(pairs: &[ChildPair]) {
    let mut sorted = pairs.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        REPEATED_PAIRS.with(|c| c.set(c.get() + 1));
    }
}

/// The nodes on this thread so far the relabelling route wrote with a pair
/// twice.
pub(super) fn repeated_pair_census() -> u64 {
    REPEATED_PAIRS.with(std::cell::Cell::get)
}

thread_local! {
    /// The levels on this thread a count read from an operand's kept counts
    /// ([`kept_counts_census`]).
    static KEPT_COUNTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn note_kept_counts(levels: usize) {
    KEPT_COUNTS.with(|c| c.set(c.get() + levels as u64));
}

/// The levels on this thread so far a count read from an operand's kept
/// counts instead of folding them.
pub(super) fn kept_counts_census() -> u64 {
    KEPT_COUNTS.with(std::cell::Cell::get)
}

thread_local! {
    /// Whether every sparse level on this thread lists each child it joins
    /// ([`list_complete_sides`]).
    static LIST_COMPLETE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The sparse levels on this thread that joined their left child, and
    /// their right one, by arithmetic ([`complete_side_census`]).
    static COMPLETE_SIDES: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// The one-product roots on this thread that a sum took with a child
    /// read by arithmetic ([`summed_complete_roots`]).
    static SUMMED_COMPLETE_ROOTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(super) fn complete_sides_listed() -> bool {
    LIST_COMPLETE.with(std::cell::Cell::get)
}

pub(super) fn note_complete_side(side: crate::diagram::ChildSide) {
    COMPLETE_SIDES.with(|c| {
        let mut census = c.get();
        census[usize::from(side == crate::diagram::ChildSide::Right)] += 1;
        c.set(census);
    });
}

pub(super) fn note_summed_complete_root() {
    SUMMED_COMPLETE_ROOTS.with(|c| c.set(c.get() + 1));
}

/// Run `f` with every sparse level listing each child it joins, a complete
/// one too: the oracle the arithmetic reads are checked against.
pub(super) fn list_complete_sides<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&LIST_COMPLETE, f)
}

/// The sparse levels on this thread so far that joined their left child,
/// and their right one, by arithmetic.
pub(super) fn complete_side_census() -> [u64; 2] {
    COMPLETE_SIDES.with(std::cell::Cell::get)
}

/// The one-product roots on this thread so far that a sum took with a
/// child read by arithmetic.
pub(super) fn summed_complete_roots() -> u64 {
    SUMMED_COMPLETE_ROOTS.with(std::cell::Cell::get)
}

thread_local! {
    /// Whether the probe join is closed on this thread ([`no_probe`]).
    static NO_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether the probe join takes every level it is admissible on, whatever
    /// it prices at ([`always_probe`]).
    static ALWAYS_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The levels on this thread each probe built: by left, by right, by
    /// pair ([`probe_census`]).
    static PROBED: std::cell::Cell<[u64; 3]> = const { std::cell::Cell::new([0; 3]) };
    /// Whether the probe by pair hashes `g`'s pairs however few cells a
    /// dense map of them would take ([`hashed_pairs`]).
    static HASHED_PAIRS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn pairs_hashed_forced() -> bool {
    HASHED_PAIRS.with(std::cell::Cell::get)
}

/// Run `f` with the probe by pair hashing `g`'s pairs on every level.
pub(super) fn hashed_pairs<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&HASHED_PAIRS, f)
}

pub(super) fn probe_forced_off() -> bool {
    NO_PROBE.with(std::cell::Cell::get)
}

pub(super) fn probe_forced() -> bool {
    ALWAYS_PROBE.with(std::cell::Cell::get)
}

pub(super) fn note_probe(probe: super::sparse::probe::Probe) {
    use super::sparse::probe::Probe;
    PROBED.with(|c| {
        let mut census = c.get();
        census[match probe { Probe::Left => 0, Probe::Right => 1, Probe::Pairs => 2 }] += 1;
        c.set(census);
    });
}

/// Run `f` with `flag` set on this thread, restored on the way out.
fn with_flag<R>(flag: &'static std::thread::LocalKey<std::cell::Cell<bool>>, f: impl FnOnce() -> R) -> R {
    struct Reset(&'static std::thread::LocalKey<std::cell::Cell<bool>>, bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            self.0.with(|c| c.set(self.1));
        }
    }
    let _reset = Reset(flag, flag.with(|c| c.replace(true)));
    f()
}

/// Run `f` with the probe join closed, every level it would take going to
/// the scatter: the oracle it is checked against.
pub(super) fn no_probe<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&NO_PROBE, f)
}

/// Run `f` with the probe join taking every level it is admissible on.
pub(super) fn always_probe<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&ALWAYS_PROBE, f)
}

/// The levels on this thread so far each probe built.
pub(super) fn probe_census() -> [u64; 3] {
    PROBED.with(std::cell::Cell::get)
}
