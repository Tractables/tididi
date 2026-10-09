//! Progress and resource measurements for running operations.

use super::Limits;
use crate::vtree::VtreeIdx;
use std::time::Instant;

/// Progress recorded when [`LimitConfig::with_conjunction_progress`](crate::limits::LimitConfig::with_conjunction_progress)
/// is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConjunctionProgress {
    /// When the conjunction began.
    pub started_at: std::time::Instant,
    /// The internal vtree level being built, 1-based in bottom-up order; 0
    /// before the first.
    pub level: u32,
    /// How many levels it walks in all.
    pub levels: u32,
}

/// A conjunction level refused before any of its storage was claimed: the
/// fewest bytes building it would take are more than the bytes left.
///
/// A conjunction builds its levels bottom-up, and each level's output holds
/// one pair for every pair of `f` and every pair of `g` there whose left
/// children and right children both have a satisfiable conjunction. Once a
/// level's children are built, that count, or a lower bound on it, is read off
/// the operands' pairs and the children's live products in one linear pass,
/// and a level whose dense route would claim its whole product grid is priced
/// by the grid. A level found to need more than the memory left is refused
/// there, with [`OperationError::OverBudget`](crate::OperationError::OverBudget),
/// rather than after it has grown into the budget. A level of operands that
/// depend on disjoint variables under it, where every product of a node of
/// `f` and a node of `g` is satisfiable, is the common case: its output is
/// the product of the two widths.
///
/// Only an engine with a byte budget or memory hooks
/// ([`LimitConfig`](crate::limits::LimitConfig)) prices levels; the bytes left
/// are the budget less what the operation holds, or, with no budget, the
/// address space the hooks report as free.
///
/// ```
/// use std::sync::Arc;
/// use tididi::limits::LimitConfig;
/// use tididi::vtree::{VarId, Vtree};
/// use tididi::{Engine, OperationError, Tdd};
///
/// // `z` selects a code `k`; `f` makes `x` equal it and `g` makes `y` equal
/// // it, each over seven bits. Under the node joining `x` and `y`, `f`'s 128
/// // nodes vary only in `x` and `g`'s only in `y`, so all 128 × 128 of their
/// // products are satisfiable: the level holds 16 384 pairs, 128 KiB.
/// let bits = |base: u32| (base..base + 7).map(VarId).collect::<Vec<_>>();
/// let (x, y, z) = (bits(1), bits(8), bits(15));
/// let xy = Vtree::join(&Vtree::balanced_over(&x)?, &Vtree::balanced_over(&y)?)?;
/// let vtree = Arc::new(Vtree::join(&xy, &Vtree::balanced_over(&z)?)?);
/// let equal = |other: &[VarId]| {
///     let vars: Vec<VarId> = z.iter().chain(other).copied().collect();
///     let rows: Vec<u64> = (0..128u64).map(|k| k | k << 7).collect();
///     Tdd::from_models(&vtree, &vars, &rows)
/// };
/// let (f, g) = (equal(&x)?, equal(&y)?);
///
/// let engine = Engine::new();
/// let budget = LimitConfig::none().with_memory_budget_bytes(Some(64 * 1024));
/// let refused = engine.limits().scope(budget);
/// assert_eq!(engine.and(f.clone(), g.clone()).err(), Some(OperationError::OverBudget));
/// let level = engine.limits().meters().refused_level.expect("the level was priced");
/// assert_eq!(level.level, xy_root(&vtree));
/// assert!(level.needed_bytes > level.headroom_bytes);
/// drop(refused);
///
/// // Without the budget the conjunction is built: `x = y = z`.
/// let h = engine.and(f, g)?;
/// assert_eq!(h.model_count()?, 128u32.into());
/// # fn xy_root(vtree: &Vtree) -> tididi::vtree::VtreeIdx { vtree.children(vtree.root()).0 }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LevelRefusal {
    /// The vtree node whose level was refused.
    pub level: VtreeIdx,
    /// The fewest bytes building the level would claim.
    pub needed_bytes: u64,
    /// The bytes left when the level was priced.
    pub headroom_bytes: u64,
}

/// Work and resource measurements returned by [`Limits::meters`](crate::limits::Limits::meters).
/// [`Limits::armed`](crate::limits::Limits::armed) reports the active configuration.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct OperationMetrics {
    /// Bytes the tracked reserves have charged since [`Limits::reset_meters`](crate::limits::Limits::reset_meters)
    /// or the start of the last operation, whichever is later.
    pub in_flight_bytes: u64,
    /// Output pairs built by the pairwise conjunction in flight, or by the last
    /// one finished (capacity for the level being built, exact for finished
    /// levels). Zeroed when any operation starts.
    pub pairs_in_flight: u64,
    /// The work clock: units the operations have polled through. Monotone and
    /// never reset, so an interval is a subtraction of two reads.
    pub work_units: u64,
    /// Bytes asked for by the most recent reserve the allocator refused, or
    /// `None` if none was refused since [`Limits::reset_meters`](crate::limits::Limits::reset_meters).
    /// Reset before a call when using this to distinguish that call's allocator
    /// refusal from a soft-budget refusal: the value persists across operations.
    /// Both refusals surface as [`OperationError::OverBudget`](crate::OperationError::OverBudget).
    pub refused_reserve_bytes: Option<u64>,
    /// Where the conjunction in flight stands, or where the last one ended,
    /// while conjunction progress is enabled; `None` before the first. Nothing clears it, so
    /// `started_at` is what tells one conjunction from the next.
    pub conjunction: Option<ConjunctionProgress>,
    /// The conjunction level the operation in flight, or the last one,
    /// refused before building it ([`LevelRefusal`]); `None` when it refused
    /// none. Zeroed when an operation starts, so after an
    /// [`OperationError::OverBudget`](crate::OperationError::OverBudget) it
    /// tells a level priced out of the budget from a growth the budget
    /// refused on the way.
    pub refused_level: Option<LevelRefusal>,
}

impl Limits {
    /// Charge newly reserved pair capacity without adding work to each pair push.
    /// [`Self::level_settled`] replaces this estimate with the completed level's
    /// actual pair count.
    #[inline]
    pub(crate) fn charge_output_pairs(&self, delta: usize) {
        if delta == 0 {
            return;
        }
        let delta = delta as u64;
        self.pairs_in_flight
            .set(self.pairs_in_flight.get().saturating_add(delta));
        self.pairs_level_charge
            .set(self.pairs_level_charge.get().saturating_add(delta));
    }

    /// Swap the level's charged capacity for the pairs it actually holds, so
    /// only the level in flight is ever an estimate and the arena's slack
    /// cannot accumulate over the thousands of levels one conjunction walks.
    ///
    /// The early-exit routes that skip the per-level tail never settle, so what
    /// they charged comes off at the next boundary instead: the meter reads low
    /// there, which is the direction a size floor tolerates.
    #[inline]
    pub(crate) fn level_settled(&self, exact_pairs: u64) {
        let charged = self.pairs_level_charge.replace(0);
        let total = self.pairs_in_flight.get().saturating_sub(charged);
        self.pairs_in_flight.set(total.saturating_add(exact_pairs));
    }


    /// Whether conjunction progress is being recorded.
    #[inline]
    pub(crate) fn conjunction_progress_enabled(&self) -> bool {
        self.conjunction_progress.get()
    }

    /// A conjunction beginning, over `levels` vtree levels. Clears whatever the
    /// last one left, so a caller can tell two apart by the instant alone.
    pub(crate) fn conjunction_began(&self, levels: u32) {
        self.conjunction.set(Some(ConjunctionProgress {
            started_at: Instant::now(),
            level: 0,
            levels,
        }));
    }

    /// Refuse the conjunction level at `level`, priced before it was built
    /// at `needed_bytes` against `headroom_bytes`, and say so in the meters.
    #[cold]
    pub(crate) fn refuse_level(&self, level: VtreeIdx, needed_bytes: u64, headroom_bytes: u64) -> crate::OperationError {
        self.refused_level.set(Some(LevelRefusal { level, needed_bytes, headroom_bytes }));
        crate::OperationError::OverBudget
    }

    /// Record the current level while keeping the conjunction's start time.
    pub(crate) fn conjunction_reached(&self, level: u32) {
        if let Some(m) = self.conjunction.get() {
            self.conjunction.set(Some(ConjunctionProgress { level, ..m }));
        }
    }
}
