//! Choosing how a level is walked and how its output arrays grow.

use super::*;

/// The bottom-up sweep's level walk: the tuned lazy `internal_bottomup` iterator
/// (the byte-identical main-compile order) or the spine-bounded apply's
/// restricted level list. A two-variant enum, not `Box<dyn Iterator>` — that box
/// cost one heap allocation per apply and an indirect `next()` per level, while
/// the loop body below is ONE loop either way.
pub(super) enum LevelWalk<'a, I> {
    Depth(I),
    /// Spine-bounded apply: `R` in `topo_pos` order (the `Depth` order with the
    /// levels that would take an identity fast path removed).
    Restricted(std::slice::Iter<'a, VtreeIdx>, &'a crate::vtree::Vtree),
}

impl<'a, I: Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)>> Iterator for LevelWalk<'a, I> {
    type Item = (VtreeIdx, VtreeIdx, VtreeIdx);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LevelWalk::Depth(it) => it.next(),
            LevelWalk::Restricted(it, vtree) => it.next().map(|&t| {
                let (l, r) = vtree.children(t);
                (t, l, r)
            }),
        }
    }
}

/// Conservative per-cell byte factor for the apply's product grid: pairs (8B)
/// + nodes (8B) + scratch (4–8B) ≈ 24B. Shared by the in-apply predictive budget
/// check (above) and the pre-apply size gate so both agree on the byte
/// conversion.
pub(crate) const APPLY_BYTES_PER_CELL: u64 = 24;

/// Byte ceiling on each of the two per-level exact reserves (`level.nodes` and
/// `level.pairs`, both sized from that level's exact upper bound).
///
/// The bounds — `k1 × k2` live cells, `|c1.pairs| × |c2.pairs|` emitted pairs —
/// are exact but loose: most levels have low survival, so an uncapped reserve
/// would routinely grab orders of magnitude more than the level ends up using
/// (and charge every byte of it to the soft budget). Capping bounds the
/// over-allocation per level; a level that outgrows the cap keeps growing
/// through the ordinary fallible push path, and `shrink_arrays` at
/// `finalize_level` hands the unused tail back (it shrinks at cap > 4 × len).
/// ONE definition — the two element-count caps below derive from it.
pub(super) const LEVEL_RESERVE_CAP_BYTES: usize = 64 * 1024;

/// [`LEVEL_RESERVE_CAP_BYTES`] in `TddNodeData`s — the `level.nodes` arm.
pub(super) const LEVEL_RESERVE_NODES_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<TddNodeData>();

/// [`LEVEL_RESERVE_CAP_BYTES`] in `InputPair`s — the `level.pairs` arm.
pub(super) const LEVEL_RESERVE_PAIRS_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<InputPair>();

/// Per-level emit-growth MODE decision before emit — walk-free: the mode is
/// chosen from the caller's `emit_pair_bound` alone, with no pre-sizing pass
/// over the level. `emit_pair_bound` must be a SOUND upper bound on the pairs
/// this level can emit: the dense walk passes `c1_pairs × c2_pairs` (every
/// product pair emits at most once), the clause apply passes its per-level
/// worst case (up to 3 c_t pairs + 1 d_t pair per input pair).
///
/// When the bound of a huge level exceeds `DENSE_GROWTH_DECISION_THRESHOLD` and
/// the worst-case Vec-doubling transient is not provably affordable, arms the
/// bounded-increment growth mode (`set_pairs_bounded_growth` → the increment
/// policy at the `try_push_pair_into` / `reserve_pairs_for_emit` choke points).
/// No counting walk runs in either mode.
///
/// Disarms first, so the armed mode is a pure function of THIS level's bound for
/// every caller. The dense cell-build loop disarms on the routes that skip this
/// call, so every level gets exactly one disarm.
#[inline(always)]
pub(crate) fn decide_emit_growth_mode(
    stream_marginal: bool,
    emit_pair_bound: u128,
) {
    set_pairs_bounded_growth(false);
    // When the bound exceeds the threshold, the resulting `level.pairs` may be
    // large enough that Vec's 2× doubling growth would peak at 3× the final size
    // (alloc new + copy + free old) — the mode decision below bounds that
    // transient. Small levels keep plain doubling (~hundreds of MiB transient
    // at the 128 M-iteration threshold — acceptable, same regime the old
    // precount also skipped) and never pay the headroom read.
    //
    // `!stream_marginal` gate: streaming targets truncate pairs per cell, so
    // peak transient is bounded by single-cell pair count and Vec-doubling on
    // `level.pairs` is irrelevant.
    if !stream_marginal && emit_pair_bound > DENSE_GROWTH_DECISION_THRESHOLD as u128 {
        // ── Mode decision: plain doubling vs bounded-increment growth ────
        //
        //   PLAIN DOUBLING — `emit_pair_bound` is a SOUND upper bound on emitted
        //   pairs, so the worst-case doubling transient is ≤ 3 × it × pair_bytes.
        //   When even that fits the headroom, nothing to protect: leave the
        //   default doubling growth. Per-push `try_push` accounting still
        //   enforces any budget during emit.
        //
        //   BOUNDED GROWTH — the doubling transient is NOT provably
        //   affordable: on a level whose pair count is already a sizeable
        //   fraction of the headroom, 3× it is not. Flag the level so the
        //   emit's `level.pairs` growth
        //   (`try_push_pair_into` per push, `reserve_pairs_for_emit` per node)
        //   takes headroom-aware increments: transient = current + increment
        //   instead of doubling's 3×current.
        //
        // Headroom is `apply_headroom_bytes_or_vas()`: the soft-budget
        // remaining when one is armed, else `RLIMIT_AS − current VAS usage`,
        // else a large finite value when RLIMIT_AS is unlimited. Count-safe throughout (growth policy never changes
        // output).
        let pair_bytes = std::mem::size_of::<InputPair>() as u128;
        let headroom = apply_headroom_bytes_or_vas() as u128;
        if emit_pair_bound.saturating_mul(3).saturating_mul(pair_bytes) >= headroom {
            // Near-cap: bounded-increment emit growth for this level.
            set_pairs_bounded_growth(true);
        }
    }
}
