//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::engine::Engine;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed, SeedConvention};
use super::super::fold::{fold_bottom_up, fold_level, LevelFold, Side};
use crate::limits::PollGate;
use crate::limits::ApplyError;
use crate::diagram::PairsIter;
use crate::value::{ColumnRetention, Count, CountRead, CountVec, IntFold};
use crate::limits::RecoveryPanic;
use crate::diagram::*;
use crate::vtree::{VarId, VtreeIdx};
use std::marker::PhantomData;

/// The u128-primary counting fold: native arithmetic for the vast majority of
/// nodes, spilling a node to the exact `BigUint` side table only where it
/// overflows.
pub(super) struct OverflowingCounts<'a> {
    pub(super) pins: &'a [Option<bool>],
    pub(super) convention: SeedConvention,
}

impl LevelFold for OverflowingCounts<'_> {
    type Value = Count;
    type Col = CountVec<RecoveryPanic>;

    fn alloc(&self, eng: &Engine, width: usize) -> CountVec<RecoveryPanic> {
        CountVec::with_width(eng, width)
    }

    fn set(&self, eng: &Engine, col: &mut CountVec<RecoveryPanic>, i: usize, v: Count) {
        col.set_i(eng, i, v);
    }

    fn leaf(&self, var: VarId, label: LeafLabel) -> Count {
        let pin = self.pins.get(var.idx()).copied().flatten();
        Count::from_u128(leaf_seed(label, pin, self.convention))
    }

    /// A marginal level's counts are pin-independent — summed out before any pin
    /// existed — so they are read across verbatim.
    fn marginal_column(
        &self,
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        col: &mut CountVec<RecoveryPanic>,
    ) {
        let level = &tdd.levels[t.idx()];
        let counts = level.marginal_counts().expect("a marginal level carries counts");
        let big = level.marginal_counts_big();
        for i in 0..counts.len() {
            col.set_i(eng, i, CountRead::from_slot(counts, big, i).to_count());
        }
    }

    /// The shared two-pass integer fold, with this query's child readers.
    ///
    /// Reading a child is the only thing that differs from any other integer
    /// fold: a pinned counter resolves through a [`SideView`], which knows
    /// whether the ref is a node index or a marginal-side value.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, CountVec<RecoveryPanic>>,
        right: Side<'_, CountVec<RecoveryPanic>>,
    ) -> Count {
        IntFold::fold(pairs, |k| read_side(left, k), |k| read_side(right, k))
    }
}

/// Resolve one child ref of a pinned counter to a count read.
///
/// The sentinel ⟺ big-slot invariant, the exact-max promotion, and the
/// stale-overflow clear on recompute (a node may stop overflowing when pins
/// change) are all owned by [`CountVec::set`] / [`Count::from_u128`].
#[inline]
fn read_side<'a>(side: Side<'a, CountVec<RecoveryPanic>>, k: usize) -> CountRead<'a> {
    let idx = match side.view.child(NodeIdx(k as u32)) {
        ChildRef::Value(ValueRef::Inline(c)) => return CountRead::Fast(c as u128),
        ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => idx as usize,
    };
    side.col.get(idx)
}

/// Column-lifetime policy as a type: [`KeepAllColumns`] or [`KeepFrontier`].
///
/// The policy decides which reads a counter can serve at all — the per-node
/// array and the incremental [`recompute`](IncrementalCounter::recompute) exist
/// only under [`KeepAllColumns`] — so it is a type parameter of
/// [`IncrementalCounter`] rather than a field, and the reads it does not
/// support are absent instead of asserting.
pub trait Retention: sealed::Sealed {
    /// The runtime policy the shared walk takes.
    const RETAIN: ColumnRetention;
}

/// Keep every level's column for the counter's lifetime. Costs a column per
/// level; buys the per-node array and the incremental
/// [`recompute`](IncrementalCounter::recompute).
#[derive(Debug, Clone, Copy)]
pub struct KeepAllColumns;

/// Keep only the walk frontier: each child column is freed as its parent's
/// completes. Peak is the frontier rather than the whole diagram, and the root
/// count is the only read.
#[derive(Debug, Clone, Copy)]
pub struct KeepFrontier;

impl Retention for KeepAllColumns {
    const RETAIN: ColumnRetention = ColumnRetention::All;
}

impl Retention for KeepFrontier {
    const RETAIN: ColumnRetention = ColumnRetention::Frontier;
}

/// Whether a counter holds counts yet: [`Unevaluated`] or [`Evaluated`].
///
/// Every column starts at zero, so a read before the first pass returns a
/// count indistinguishable from UNSAT. [`IncrementalCounter::compute`] is the only
/// way to reach [`Evaluated`], and the reads live only there.
pub trait CounterState: sealed::Sealed {}

/// Allocated, pinned, never passed over. No count to read.
#[derive(Debug, Clone, Copy)]
pub struct Unevaluated;

/// Carries the counts of one completed pass under the pins in force at it.
#[derive(Debug, Clone, Copy)]
pub struct Evaluated;

impl CounterState for Unevaluated {}
impl CounterState for Evaluated {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::KeepAllColumns {}
    impl Sealed for super::KeepFrontier {}
    impl Sealed for super::Unevaluated {}
    impl Sealed for super::Evaluated {}
}

/// Incremental pinned model counter for a Gray-code cofactor sum.
///
/// One hybrid `CountVec` column per vtree level (u128-primary, `BigUint` side
/// table on overflow, which keeps most of the arithmetic off the heap; the
/// discipline is shared with the apply and marginalize contexts, see
/// `value`).
/// Under [`KeepAllColumns`] it holds the full per-node count array; after one
/// [`compute`](Self::compute), changing a few variables' pins and calling
/// [`recompute`](Self::recompute) re-folds only the levels between those
/// leaves and the root, so the root count updates in the size of that cone
/// instead of the size of the diagram — every other node's cached count is
/// reused verbatim. The result equals `pinned_counts`.
///
/// The two type parameters are the counter's capabilities rather than
/// documentation: `R` ([`Retention`]) decides whether the per-node array and
/// the incremental recompute exist, and `S` ([`CounterState`]) whether there
/// is anything to read.
///
/// The counter owns its columns and pins (sized from `tdd` at construction) and
/// does not borrow the diagram — every method takes `tdd` as an argument. One
/// counter therefore serves many evaluations of the same diagram: re-pin, then
/// either [`recompute`](Self::recompute) under [`KeepAllColumns`] or a fresh
/// [`compute`](Self::compute). Callers must pass the same `tdd` the counter was
/// sized from; a structurally different diagram is a logic error, since the
/// arrays would be mis-sized. The diagram need not be canonical. A
/// count-marginal level is read from its stored counts, which no pin reaches;
/// a weight-marginal level cannot be counted and panics in the pass. An
/// unpinned variable counts as free; a pinned one counts per the
/// [`SeedConvention`].
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::query::{IncrementalCounter, KeepAllColumns, SeedConvention, Unevaluated};
/// use tididi::vtree::{VarId, Vtree};
///
/// let engine = Engine::new();
/// let vtree = Arc::new(Vtree::balanced(4));
/// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// tididi::reduce::minimize(&mut f);
///
/// // No pin: the count is the diagram's own.
/// let counter = IncrementalCounter::<KeepAllColumns, Unevaluated>::new(
///     &engine, &f, 4, SeedConvention::Fixed);
/// let counter = counter.compute(&engine, &f);
/// assert_eq!(counter.output_count(&f), f.model_count());
///
/// // Pin x1 to true and re-fold only the levels between that leaf and the root.
/// let mut counter = counter;
/// counter.set_pin(VarId(0), Some(true));
/// counter.recompute(&engine, &f);
/// let pinned = counter.output_count(&f);
/// assert!(pinned <= f.model_count());
/// ```
pub struct IncrementalCounter<R: Retention, S: CounterState> {
    cols: Vec<CountVec<RecoveryPanic>>,
    pins: Vec<Option<bool>>,
    /// Variables whose pin changed since the last pass, each once: what
    /// [`recompute`](Self::recompute) re-folds from.
    changed: Vec<VarId>,
    /// The leaf-seed convention this counter pins with.
    convention: SeedConvention,
    _marker: PhantomData<(R, S)>,
}

impl<R: Retention, S: CounterState> std::fmt::Debug for IncrementalCounter<R, S> {
    /// The pass state and how many level columns are currently held — the two
    /// things that distinguish one counter from another at a breakpoint.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = std::any::type_name::<S>().rsplit("::").next().unwrap_or("?");
        f.debug_struct("IncrementalCounter")
            .field("state", &state)
            .field("retention", &R::RETAIN)
            .field("convention", &self.convention)
            .field("columns_held", &self.cols.iter().filter(|c| c.width() > 0).count())
            .field("pins", &self.pins.len())
            .field("changed_since_pass", &self.changed.len())
            .finish()
    }
}

impl<R: Retention> IncrementalCounter<R, Unevaluated> {
    /// Allocate the count array with pin slots `0..n_pins`. No pass run yet.
    ///
    /// `convention` is the leaf seed a pinned variable gets
    /// ([`SeedConvention`]); with zero pins the two coincide.
    ///
    /// Under [`KeepAllColumns`] every level's column is sized here and kept; under
    /// [`KeepFrontier`] the columns are allocated on write and freed as parents
    /// complete, so pre-sizing them would commit exactly the whole-diagram
    /// array that policy exists to avoid.
    pub fn new(eng: &Engine, tdd: &Tdd, n_pins: usize, convention: SeedConvention) -> Self {
        let cols = (0..tdd.vtree.num_nodes())
            .map(|i| match R::RETAIN {
                ColumnRetention::All => {
                    CountVec::with_width(eng, tdd.effective_width(VtreeIdx(i as u32)))
                }
                ColumnRetention::Frontier => CountVec::with_width(eng, 0),
            })
            .collect();
        Self { cols, pins: vec![None; n_pins], changed: Vec::new(), convention, _marker: PhantomData }
    }
}

impl<R: Retention, S: CounterState> IncrementalCounter<R, S> {
    /// Set one variable's pin (does not recompute).
    ///
    /// A pin set to the value it already holds changes nothing, and the next
    /// [`recompute`](Self::recompute) does no work for it.
    ///
    /// # Panics
    ///
    /// Panics if `var.idx()` is not below the `n_pins` the counter was built
    /// with.
    #[inline]
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) {
        if self.pins[var.idx()] == val {
            return;
        }
        self.pins[var.idx()] = val;
        if !self.changed.contains(&var) {
            self.changed.push(var);
        }
    }

    /// Full bottom-up pass under the current pins (every leaf + every internal
    /// level), yielding the counter its counts can be read from.
    ///
    /// This is also how a counter that already holds counts takes a fresh pass
    /// after re-pinning — the only route under [`KeepFrontier`], which has no
    /// incremental path.
    #[must_use = "the pass produces a new counter; the receiver is consumed"]
    pub fn compute(self, eng: &Engine, tdd: &Tdd) -> IncrementalCounter<R, Evaluated> {
        self.try_compute(eng, tdd, None)
            .expect("an unpolled pass observes no stop axis")
    }

    /// [`compute`](Self::compute) under a stop axis: the pass is cut between
    /// levels, where every level below the cut holds a complete column and
    /// nothing has been read yet.
    ///
    /// # Errors
    ///
    /// Propagates the armed stop, polled at every internal level boundary. The
    /// cut counter is dropped with its partial columns — there is no reading a
    /// pass that did not finish.
    pub(crate) fn try_compute(
        mut self,
        eng: &Engine,
        tdd: &Tdd,
        poll: Option<&mut PollGate>,
    ) -> Result<IncrementalCounter<R, Evaluated>, ApplyError> {
        self.changed.clear();
        let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
        let cols = &mut self.cols;
        if R::RETAIN == ColumnRetention::Frontier {
            // Free-before-rebuild: drop the previous pass's surviving column
            // (the root's, plus any level this pass will not revisit) before
            // allocating anything new, so two passes' peaks never overlap.
            for c in cols.iter_mut() {
                *c = CountVec::with_width(eng, 0);
            }
        }
        fold_bottom_up(&fold, eng, tdd, cols, R::RETAIN, poll, |cols, ti| {
            // Under `KeepAllColumns` the constructor pre-sized every column and
            // nothing shrinks them, so this is one length compare per level;
            // under `KeepFrontier` it is the allocate-on-write step for a
            // column that starts — or was freed — empty. A freshly allocated
            // column is all-zero, which is what a fresh counter's column holds,
            // so slots no pass writes (tombstones, which the fold skips) read
            // the same under both policies.
            let w = tdd.effective_width(VtreeIdx(ti as u32));
            if cols[ti].len() != w {
                cols[ti] = CountVec::with_width(eng, w);
            }
        })?;
        Ok(IncrementalCounter {
            cols: self.cols,
            pins: self.pins,
            changed: self.changed,
            convention: self.convention,
            _marker: PhantomData,
        })
    }
}

impl<R: Retention> IncrementalCounter<R, Evaluated> {
    /// The current root (output) model count.
    ///
    /// # Panics
    ///
    /// Panics on a diagram that [`is_zero`](Tdd::is_zero): the `ZERO`
    /// sentinel names no count slot. Test for ⊥ first; its count is zero.
    #[inline]
    pub fn output_count(&self, tdd: &Tdd) -> BigUint {
        let (t, i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
        match self.cols[t].get(i) {
            CountRead::Fast(v) => BigUint::from(v),
            CountRead::Big(b) => b.clone(),
        }
    }
}

impl IncrementalCounter<KeepAllColumns, Evaluated> {
    /// Bring the counts up to date with the pins changed since the last pass.
    ///
    /// Re-folds exactly the levels between each changed variable's leaf and
    /// the root, children before parents: a leaf is re-seeded from its pin,
    /// an internal level re-summed from its children, every other level's
    /// cached column read as it stands. With no pin changed it does nothing.
    ///
    /// Only [`KeepAllColumns`] offers this: the re-fold reads cached columns
    /// outside the cone, which [`KeepFrontier`] frees as parents complete.
    pub fn recompute(&mut self, eng: &Engine, tdd: &Tdd) {
        let vtree = &tdd.vtree;
        let mut in_cone = vec![false; vtree.num_nodes()];
        let mut cone = Vec::new();
        for &var in &self.changed {
            // A pin on a variable the vtree does not carry seeds no leaf.
            let Some(mut t) = vtree.leaf_of(var) else { continue };
            // Once a level is in the cone so are all of its ancestors.
            while !in_cone[t.idx()] {
                in_cone[t.idx()] = true;
                cone.push(t);
                match vtree.node(t).parent() {
                    Some(p) => t = p,
                    None => break,
                }
            }
        }
        self.changed.clear();
        let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
        for &t in vtree.bottom_up_subset(cone).levels() {
            fold_level(&fold, eng, tdd, &mut self.cols, t);
        }
    }

    /// Consume the counter, returning the per-node u128 count columns
    /// (`fast[t][i]`) and discarding the `BigUint` side table. A slot that
    /// counted past `u128` saturates to `u128::MAX` and its exact
    /// magnitude is dropped. A count of zero stays exact: the u128 array is authoritative for
    /// zero — only a *non-zero* overflow ever spills to the Big side table — so
    /// `fast[t][i] == 0` iff node `(t,i)` has no models. For callers that need
    /// only monotone ordering, a small-threshold compare, and exact-zero
    /// detection, never an overflowed node's exact value.
    ///
    /// Only [`KeepAllColumns`] offers this: [`KeepFrontier`] keeps the root column
    /// alone, so there is no per-node array to hand out.
    pub(crate) fn into_fast_counts(self) -> Vec<Vec<u128>> {
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
    }
}
