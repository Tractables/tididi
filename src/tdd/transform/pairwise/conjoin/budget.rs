//! The apply engine's own accounting on top of [`crate::tdd::limits`]: the
//! output-pair meter, the bounded emit growth for near-cap levels, the poll
//! strides of the cell loops, and the per-level boundary check.

use crate::tdd::limits::{
    apply_deadline_expired, apply_headroom_bytes_or_vas, apply_output_node_cap,
    budget_reserve_exact, mem_preflight_alloc, try_push, ApplyError, MergePosition, APPLY_LIMITS,
};

/// Sentinel for dead product cells: c1[i] ∧ c2[j] = ⊥ (no output node created).
///
/// Same bit pattern as `ZERO` in types.rs but semantically distinct:
/// - `ZERO` marks a TDD whose output is UNSAT (top-level concept)
/// - `DEAD` marks a single product grid cell that produced no live pairs (local to apply)
pub(crate) const DEAD: u32 = u32::MAX;

/// Charge the output-pair meter ([`ApplyLimits::pairs_in_flight`]): `delta`
/// more slots of capacity in an OUTPUT level's pair arena. The single writer —
/// every site that grows such an arena routes its capacity delta here, and
/// nothing else touches the cell.
///
/// Separate from [`account_capacity_delta`] on purpose, and not derivable from
/// it: that one is typed `T` and sees a `Vec<InputPair>` it cannot tell apart
/// from the hoisted C2 column table, which is a transient scratch and not
/// output. The distinction is the whole point of the meter.
#[inline]
pub(super) fn account_output_pairs(delta: usize) {
    if delta == 0 {
        return;
    }
    APPLY_LIMITS.with(|l| {
        l.pairs_in_flight.set(l.pairs_in_flight.get().saturating_add(delta as u64));
        l.pairs_level_charge.set(l.pairs_level_charge.get().saturating_add(delta as u64));
    });
}

/// Settle the level that just finished: replace the capacity
/// [`account_output_pairs`] charged for it with the `pairs` it actually holds.
/// Called once per level from the per-level tail, which is the one place that
/// knows both. Keeps [`ApplyLimits::pairs_in_flight`] exact over finished
/// levels, so only the level in flight is ever an estimate.
#[inline]
pub(super) fn settle_output_pairs(exact_pairs: usize) {
    APPLY_LIMITS.with(|l| {
        let charged = l.pairs_level_charge.replace(0);
        let total = l.pairs_in_flight.get().saturating_sub(charged);
        l.pairs_in_flight.set(total.saturating_add(exact_pairs as u64));
    });
}

/// Amortization stride for the sparse-scatter/collapse-collector [`PollTicker`]s
/// — one poll per ~1M units of inner work, matching `nxm_deadline_check`'s
/// `1 << 20` cadence.
pub(super) const APPLY_POLL_STRIDE: u64 = 1 << 20;

/// Amortization stride for the dense between-cell [`PollTicker`]s — one poll
/// per ~65536 cell iterations.
pub(super) const DENSE_CELL_POLL_STRIDE: u64 = 1 << 16;

/// Grow `v` up to `new_len`, filling with `DEAD`. Returns `OverBudget`
/// if the allocator refuses (e.g. under `ulimit -v`) or the soft budget
/// would be exceeded.
#[inline]
pub(super) fn try_resize_dead(v: &mut Vec<u32>, new_len: usize) -> Result<(), ApplyError> {
    if v.len() >= new_len { return Ok(()); }
    let additional = new_len - v.len();
    budget_reserve_exact(v, additional)?;
    v.resize(new_len, DEAD);
    Ok(())
}

/// Interleaved-map variant of `try_resize_dead`: grows a `[ct, dt]`-entry
/// vec, filling with `[DEAD, DEAD]`.
#[inline]
pub(crate) fn try_resize_dead2(v: &mut Vec<[u32; 2]>, new_len: usize) -> Result<(), ApplyError> {
    if v.len() >= new_len { return Ok(()); }
    let additional = new_len - v.len();
    budget_reserve_exact(v, additional)?;
    v.resize(new_len, [DEAD, DEAD]);
    Ok(())
}

/// Fallible pair push: stores into `level.pairs`, routing growth through
/// `try_push`. Packing paths (packed-pairs, Phase F, Lever 14) have been deleted.
///
/// This is the single choke point for `level.pairs` growth on the dense emit
/// walk. When `decide_emit_growth_mode` has flagged the level as
/// near-cap ([`ApplyLimits::pairs_bounded_growth`]), a growth event (len ==
/// capacity) routes through [`grow_pairs_bounded`] — bounded, headroom-aware
/// increments instead of `Vec`'s 2× doubling — so the reallocation transient
/// stays `current + increment` rather than doubling's 3×current. In the
/// default mode (flag off) the extra branch is one predicted-false TLS check
/// on growth events only; the capacity-available fast path is unchanged.
///
/// Split like [`try_push`]: a bare `len < capacity` store here, everything else
/// in [`push_pair_grow`]. See `try_push`'s doc for why the spare-capacity path
/// was already inert (no preflight, no TLS read, zero budget charged) and why
/// exiling the rest is what lets LLVM keep `len`/`capacity`/the arena base in
/// registers across the emit walk's pushes.
#[inline(always)]
pub(super) fn try_push_pair_into(level: &mut crate::tdd::types::TddLevel, pair: crate::tdd::types::InputPair) -> Result<(), ApplyError> {
    let v = &mut level.pairs;
    if v.len() < v.capacity() {
        v.push(pair);
        return Ok(());
    }
    push_pair_grow(v, pair)
}

/// Growth arm of [`try_push_pair_into`] — the original body, verbatim (the
/// `pairs_bounded_growth` TLS check gating [`grow_pairs_bounded`], then
/// [`try_push`], which supplies the `budget_reserve` → `account_capacity_delta`
/// → push accounting). Reached only when `len == capacity`, so the
/// `len == v.capacity()` re-test is trivially true and kept only so this body
/// stays a literal copy of the pre-split one.
#[cold]
#[inline(never)]
fn push_pair_grow(
    v: &mut Vec<crate::tdd::types::InputPair>,
    pair: crate::tdd::types::InputPair,
) -> Result<(), ApplyError> {
    let pre_cap = v.capacity();
    if v.len() == v.capacity() && APPLY_LIMITS.with(|l| l.pairs_bounded_growth.get()) {
        grow_pairs_bounded(v)?;
    }
    let out = try_push(v, pair);
    // Output-pair meter: charged here rather than in `try_push_pair_into`
    // because this is the arm a growth event reaches, and growth is the only
    // thing that moves capacity. See [`ApplyLimits::pairs_in_flight`].
    account_output_pairs(v.capacity().saturating_sub(pre_cap));
    out
}

/// Minimum bounded-growth increment for `level.pairs`, in bytes (1 M pairs at
/// 8 B). Floors the increment so growth never degenerates to per-push
/// reallocation. It cannot cause O(n²) memcpy in practice: the floor only
/// binds when half the remaining headroom is below it (headroom
/// < 2 × this), i.e. within one or two growth events of `OverBudget` —
/// everywhere else the increment is half-headroom (geometric) or full
/// doubling.
const PAIRS_GROW_MIN_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

/// Byte size of one `level.pairs` element — the single element-size constant
/// for every pair-arena growth computation.
const PAIR_ELEM_BYTES: u64 = std::mem::size_of::<crate::tdd::types::InputPair>() as u64;

/// Pure increment policy for [`grow_pairs_bounded`] (factored out for unit
/// tests): grow a full `cap`-capacity Vec by
/// `min(cap, max(min_chunk, headroom/2))` elements, i.e.
/// `next_cap = min(2×cap, cap + max(min_chunk, half-headroom))`.
///
/// - Plentiful headroom (`headroom/2 ≥ cap × elem_bytes`): increment = `cap`
///   — plain doubling, amortized O(n) pushes.
/// - Shrinking headroom: increment = half the remaining headroom, so the
///   reallocation transient (`old cap + increment`) always fits, and
///   successive increments halve geometrically — O(log) growth events.
/// - Near exhaustion: the `min_chunk` floor; at most a couple of events
///   before the budget/allocator trips `OverBudget` (see the constant doc).
fn bounded_grow_increment(cap: usize, headroom_bytes: u64, elem_bytes: u64) -> usize {
    let eb = elem_bytes.max(1);
    let half_room = usize::try_from(headroom_bytes / 2 / eb).unwrap_or(usize::MAX);
    let min_chunk = (PAIRS_GROW_MIN_CHUNK_BYTES / eb) as usize;
    half_room.max(min_chunk).min(cap)
}

/// Bounded-growth increment for a `cap`-capacity `level.pairs`: the ONE place
/// that feeds the pair element size and the current headroom into
/// [`bounded_grow_increment`]. Shared by [`grow_pairs_bounded`] (per-push choke
/// point) and [`reserve_pairs_for_emit`] (bulk twin), so both grow by the same
/// policy.
#[inline]
fn bounded_pairs_increment(cap: usize) -> usize {
    bounded_grow_increment(cap, apply_headroom_bytes_or_vas(), PAIR_ELEM_BYTES)
}

/// Bounded, headroom-aware growth for a full `level.pairs` Vec (the near-cap
/// emit mode — see [`try_push_pair_into`]). Reserves the
/// [`bounded_grow_increment`] via `budget_reserve_exact`, so the chunk goes
/// through the ONE existing accounting path (`account_capacity_delta`,
/// preflight decay check, OS-failure → `OverBudget`) — no second budget
/// mechanism. Cold: called once per growth event, never per push.
#[cold]
#[inline(never)]
fn grow_pairs_bounded(v: &mut Vec<crate::tdd::types::InputPair>) -> Result<(), ApplyError> {
    let cap = v.capacity();
    if cap == 0 {
        // Fresh vec: `try_push`'s doubling from empty is trivially transient-safe.
        return Ok(());
    }
    budget_reserve_exact(v, bounded_pairs_increment(cap))
}

/// Bulk twin of [`try_push_pair_into`]: guarantee room for `additional` more
/// pairs in `level.pairs`, so the caller can then emit them with plain
/// `Vec::push` / `extend_from_slice` instead of a per-pair reserve.
///
/// Growth obeys the SAME per-level mode as the per-push choke point — bounded,
/// headroom-aware [`bounded_grow_increment`] chunks when
/// `decide_emit_growth_mode` armed this level, plain `Vec` doubling otherwise —
/// so `level.pairs` has one growth policy, not two. A refusal maps to
/// `OverBudget`, the same failure the caller's other reserves return, so the
/// recovery cascade sees no new error class.
///
/// The ONE intentional divergence from the accounted reserves is the budget
/// CHARGE: this serves the clause apply, which runs outside [`reset_meters`]
/// (only `apply_and_fallible` entry clears that counter), so charging its pair
/// arena would accumulate across clauses and trip the soft budget spuriously.
/// The allocator preflight and the `OverBudget` refusal channel are the same.
#[inline]
pub(crate) fn reserve_pairs_for_emit(
    level: &mut crate::tdd::types::TddLevel,
    additional: usize,
) -> Result<(), ApplyError> {
    let v = &mut level.pairs;
    if additional <= v.capacity() - v.len() {
        return Ok(());
    }
    if APPLY_LIMITS.with(|l| l.pairs_bounded_growth.get()) {
        let inc = bounded_pairs_increment(v.capacity()).max(additional);
        mem_preflight_alloc(
            (inc as u64).saturating_mul(PAIR_ELEM_BYTES),
        );
        return v.try_reserve_exact(inc).map_err(|_| ApplyError::OverBudget);
    }
    mem_preflight_alloc(
        (v.capacity().max(additional) as u64).saturating_mul(PAIR_ELEM_BYTES),
    );
    v.try_reserve(additional).map_err(|_| ApplyError::OverBudget)
}

/// Whether to publish where this apply has got to — one `Cell` load, asked once
/// per apply and once before each level, so an unwatched compile pays that and
/// nothing else.
pub(super) fn merges_watched() -> bool {
    APPLY_LIMITS.with(|l| l.merges_watched.get())
}

/// An apply BEGINNING, over `levels` vtree levels. Clears whatever the last one
/// left, so a watcher can tell two applies apart by the instant alone.
pub(super) fn merge_began(levels: u32) {
    APPLY_LIMITS.with(|l| {
        l.merge.set(Some(MergePosition { began: std::time::Instant::now(), level: 0, levels }))
    });
}

/// An apply REACHING level `level`. One store, no clock — the watcher reads the
/// clock it was already reading.
pub(super) fn merge_reached(level: u32) {
    APPLY_LIMITS.with(|l| {
        if let Some(m) = l.merge.get() {
            l.merge.set(Some(MergePosition { level, ..m }));
        }
    });
}

/// Reset the per-apply meters — the in-flight byte counters that accumulate
/// across the apply currently in flight — in one TLS touch. Called once, at the
/// top of `apply_and_fallible_inner`.
pub(super) fn reset_meters() {
    APPLY_LIMITS.with(|l| {
        l.budget_in_flight.set(0);
        l.pairs_in_flight.set(0);
        l.pairs_level_charge.set(0);
        // Hygiene: the per-level emit-growth mode must never survive into a
        // new apply (it is also disarmed once per level).
        l.pairs_bounded_growth.set(false);
    });
}

/// Set the per-level bounded-emit-growth mode consulted by
/// `try_push_pair_into` (see [`ApplyLimits::pairs_bounded_growth`]). Disarmed
/// (false) once per level — by `decide_emit_growth_mode` on entry, or by the
/// cell-build loop on the routes that skip it; armed (true) only by
/// `decide_emit_growth_mode`'s near-cap decision.
#[inline]
pub(super) fn set_pairs_bounded_growth(on: bool) {
    APPLY_LIMITS.with(|l| l.pairs_bounded_growth.set(on));
}

/// The per-level-boundary cut check, in fixed order: (gated) wall deadline,
/// then output-node cap. Called once per vtree level by
/// `apply_and_fallible_inner`. Order and short-circuit behavior are
/// load-bearing — do not reorder.
///
/// `out_nodes_so_far` / `live_counts` are the running output-node-cap inputs
/// (the full re-sum is only formed, for the `debug_assert`, when the cap is
/// armed).
pub(super) fn check_level_boundary(
    out_nodes_so_far: u64,
    live_counts: &[usize],
) -> Result<(), ApplyError> {
    // Per-iteration `ApplyLimits::deadline` poll, gated by the
    // `enable_apply_deadline_check()` override (default-OFF). This is the ONLY
    // wall-deadline cut for the per-level apply orchestration
    // (plan/finalize). Without it a
    // stuck branch grinds wide levels for minutes/hours and conditioning can
    // never deepen it. When the env var is unset `apply_deadline_expired()`
    // folds to `false` (OnceLock) → byte-identical hot path, so the
    // submission's no-env default is unchanged.
    if apply_deadline_expired() {
        return Err(ApplyError::Deadline);
    }

    // Output-size bail: when the segment-conjoin driver arms an output-node
    // cap (`ApplyLimits::output_node_cap`), abort the moment the output nodes built
    // so far exceed the cap. This bounds the conjunction's RESULT — a product
    // whose output balloons far past the segment's intended size is doomed
    // and not worth grinding to the wall deadline or the looser byte budget.
    // `out_nodes_so_far` is the running `sum(live_counts)`, maintained in O(1)
    // by `bump_live_count` at every build path (was an O(levels²) re-sum here);
    // the `debug_assert_eq!` cross-checks it against the full sum. Gated behind
    // the cap being `Some`, so the no-env hot path pays one TLS read only.
    if let Some(cap) = apply_output_node_cap() {
        debug_assert_eq!(
            out_nodes_so_far,
            live_counts.iter().map(|&c| c as u64).sum::<u64>(),
            "out_nodes_so_far desynced from live_counts sum — a live_counts \
             write bypassed bump_live_count",
        );
        if out_nodes_so_far > cap {
            return Err(ApplyError::OutputCap);
        }
    }

    Ok(())
}

/// Threshold (in scatter iterations = `c1.pairs.len() * c2.pairs.len()`)
/// above which the dense path runs the emit-growth mode decision
/// (`decide_emit_growth_mode`): compare the worst-case Vec-doubling transient
/// of the level's pair-product bound against the byte headroom and arm the
/// bounded-increment growth mode when it doesn't provably fit.
///
/// Fixed at 128 M iterations. Below this, doubling pays ~hundreds of MiB of
/// transient peak — acceptable — so small levels skip the decision entirely
/// and never pay the `apply_headroom_bytes_or_vas()` read (whose VAS fallback
/// does a ~µs `mapped_bytes()` epoch-advance read). Above, doubling from cap
/// N→2N transients 3N (e.g. canary 131's 1.25 G→2.5 G entries = 30 GiB peak,
/// trips the 31 GiB MCC cap) — exactly what the bounded mode protects.
pub(super) const DENSE_GROWTH_DECISION_THRESHOLD: usize = 128 * 1024 * 1024;


#[cfg(test)]
#[path = "budget_bounded_growth_tests.rs"]
mod bounded_growth_tests;

#[cfg(test)]
mod stall_rope_tests {
    use super::{account_output_pairs, check_level_boundary, settle_output_pairs};
    use crate::tdd::limits::{
        apply_deadline_check_enabled, apply_deadline_expired, apply_limits, apply_meters,
        charge_apply_in_flight_for_test, charge_compile_work, enable_apply_deadline_check,
        reset_apply_deadline_check_for_test, reset_apply_meters,
        ApplyError, RopeLimit, APPLY_LIMITS,
    };
    use std::time::{Duration, Instant};

    fn reset_pairs() {
        APPLY_LIMITS.with(|l| {
            l.pairs_in_flight.set(0);
            l.pairs_level_charge.set(0);
        });
    }

    /// **The stall rope needs BOTH halves, and it is asked at the INTRA-LEVEL
    /// poll — not only at the level boundary.**
    ///
    /// It is the give-up rule's answer to a step whose inputs were small and
    /// whose output is not: the caller arms the rule's own pair floor — in
    /// PAIRS, the unit that floor is stated in — as the trigger, and the step's
    /// usual share of the wall as the rope. The meter is `pairs_in_flight`,
    /// which is what lets the check ride `limits_reached` — the poll the wall
    /// deadline already uses from inside the dense loops. An apply that never
    /// finishes a level never reaches a level boundary, and that is precisely
    /// the apply this is for.
    #[test]
    fn a_stall_rope_cuts_at_the_intra_level_poll_once_built_and_out_of_time() {
        let was_armed = apply_deadline_check_enabled();
        enable_apply_deadline_check();
        reset_pairs();
        let spent = RopeLimit::Wall(Instant::now() - Duration::from_secs(1));
        let unspent = RopeLimit::Wall(Instant::now() + Duration::from_secs(60));

        // Nothing armed: the poll is inert, whatever the apply has built.
        account_output_pairs(1 << 20);
        assert!(apply_meters().stall_rope.is_none());
        assert!(!apply_deadline_expired());
        assert!(check_level_boundary(0, &[]).is_ok());

        reset_pairs();
        {
            let _g = apply_limits().stall_rope(Some((1_000, spent))).apply();
            assert_eq!(apply_meters().stall_rope, Some((1_000, spent)));
            // Share spent, output still under the floor: this is a step whose
            // long run is search, which is the whole reason the rule has a size
            // factor at all. It is not cut.
            account_output_pairs(999);
            assert!(!apply_deadline_expired(), "under the floor, not at it");
            // At the floor with the share spent: a big diagram, and out of time.
            account_output_pairs(1);
            assert_eq!(apply_meters().pairs_in_flight, 1_000);
            assert!(apply_deadline_expired());
            // …and the level boundary reports it as what it is, because it asks
            // that same poll rather than keeping a second meter of its own.
            assert!(matches!(check_level_boundary(0, &[]), Err(ApplyError::Deadline)));
        }

        {
            // Built past the floor, but the share it was given is still on the
            // clock — the floor moves who is ELIGIBLE, never when the rope falls.
            let _g = apply_limits().stall_rope(Some((1_000, unspent))).apply();
            assert!(!apply_deadline_expired());
        }

        // …and the guard put the axis back, so nothing outside the step it
        // belonged to can be cut by it.
        assert!(apply_meters().stall_rope.is_none());
        assert!(!apply_deadline_expired());
        reset_pairs();
        if !was_armed {
            reset_apply_deadline_check_for_test();
        }
    }

    /// **A step that is heavy in BYTES but light in built PAIRS is not
    /// eligible.**
    ///
    /// The floor asks how big a diagram this step holds. `budget_in_flight`
    /// answers a different question — how much memory the apply has claimed,
    /// including grids, live scratch and hoisted transients — and metering the
    /// rule with it made a tiny node that touched 8 MB of scratch look like a
    /// million-pair diagram. That is the miss this pins: the byte meter is over
    /// any plausible floor, the pair meter is one pair under, and the rope holds.
    /// **A level's unused arena slack does not survive the level.**
    ///
    /// Within a level the meter can only estimate — it reads the arena's
    /// CAPACITY, which is seeded up to `LEVEL_RESERVE_PAIRS_CAP` before a
    /// single pair is emitted. Left standing, that seed would accumulate once
    /// per level over the thousands one apply walks, and the meter would grow
    /// with the LEVEL COUNT instead of the diagram. So each boundary swaps the
    /// estimate for the truth.
    #[test]
    fn a_finished_level_contributes_its_pairs_not_its_capacity() {
        reset_pairs();
        // Level 1: seeded for 8192, emitted 3.
        account_output_pairs(8192);
        assert_eq!(apply_meters().pairs_in_flight, 8192, "in flight, capacity is all there is");
        settle_output_pairs(3);
        assert_eq!(apply_meters().pairs_in_flight, 3);
        // Level 2 does the same: the slack does not compound.
        account_output_pairs(8192);
        settle_output_pairs(3);
        assert_eq!(apply_meters().pairs_in_flight, 6, "two low-survival levels, six pairs");
        // A level that genuinely builds keeps what it built.
        account_output_pairs(2_000_000);
        settle_output_pairs(1_500_000);
        assert_eq!(apply_meters().pairs_in_flight, 1_500_006);
        reset_pairs();
    }

    #[test]
    fn bytes_are_not_pairs_a_big_apply_that_built_little_is_not_cut() {
        let was_armed = apply_deadline_check_enabled();
        enable_apply_deadline_check();
        reset_apply_meters();
        reset_pairs();
        let spent = RopeLimit::Wall(Instant::now() - Duration::from_secs(1));

        let _g = apply_limits().stall_rope(Some((1_000_000, spent))).apply();
        // A gigabyte of charged apply memory, and a diagram of 999_999 pairs.
        charge_apply_in_flight_for_test(1 << 30);
        account_output_pairs(999_999);
        assert!(
            !apply_deadline_expired(),
            "eligibility is the pair count, not the byte count",
        );
        // One more pair — the same rope, now genuinely met.
        account_output_pairs(1);
        assert!(apply_deadline_expired());

        drop(_g);
        reset_apply_meters();
        reset_pairs();
        if !was_armed {
            reset_apply_deadline_check_for_test();
        }
    }

    /// **A work rope falls on the work clock and on nothing else.**
    ///
    /// The whole point of the currency: no deadline is installed, no time passes,
    /// and the cut happens anyway — because the compile did the work. Its
    /// counterpart is the wall rope above, which the clock decides and charging
    /// cannot move.
    #[test]
    fn a_work_rope_falls_on_the_work_clock_and_not_on_the_wall() {
        let was_armed = apply_deadline_check_enabled();
        enable_apply_deadline_check();
        reset_pairs();

        let stride = 1u64 << 20;
        let at = apply_meters().work_units.saturating_add(4 * stride);
        let _g = apply_limits().stall_rope(Some((0, RopeLimit::Work(at)))).apply();
        assert_eq!(apply_meters().stall_rope, Some((0, RopeLimit::Work(at))));
        // Nothing has a wall here: an unbudgeted apply, and a rope that is not
        // due. The floor is zero, so what holds the rope back is the clock alone.
        assert!(!apply_deadline_expired(), "a work rope fell before its work was done");
        charge_compile_work(3 * stride);
        assert!(!apply_deadline_expired(), "one stride short is short");
        charge_compile_work(stride);
        assert!(apply_deadline_expired());
        assert!(matches!(check_level_boundary(0, &[]), Err(ApplyError::Deadline)));

        drop(_g);
        assert!(!apply_deadline_expired(), "the guard left a work rope armed");
        reset_pairs();
        if !was_armed {
            reset_apply_deadline_check_for_test();
        }
    }
}
