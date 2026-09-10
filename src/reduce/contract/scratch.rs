use crate::engine::Engine;

use smallvec::SmallVec;


use super::merge::GroupPlan;

// Contract-path arena growth goes through `conjoin::budget`'s `try_push` /
// `try_resize` — the single budget-charged implementation, shared with apply.
// Do not add local fallible push/resize helpers here: a raw
// `try_reserve_exact(1)` bypasses the soft-budget charge and grows by one per
// push at capacity (quadratic-prone).

// ── Thread-local scratch buffer pool ────────────────────────────────────────
//
// All scratch buffers are bundled into a single struct, taken once at the start
// of contract_all_twins and returned at the end via Cell::take()/Cell::set()
// (see types.rs for details on this pooling pattern). One Cell per call avoids
// per-helper Cell::with overhead.

/// Open-addressing slot for the twin-grouping tables: fingerprint co-located
/// with the occupant index so a probe costs one random load instead of two
/// (ht[slot] then fingerprints[occupant]). idx == u64::MAX marks an empty slot.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct TwinSlot {
    pub(super) fp: u64,
    pub(super) idx: u64,
}

pub(super) const EMPTY_SLOT: TwinSlot = TwinSlot { fp: 0, idx: u64::MAX };

/// Reusable generation-stamped scatter for `pair_fusion::collect_fusion_plans`'
/// per-node same-explicit-side grouping. Replaces the former per-boundary
/// `FxHashMap<u32, SmallVec<[u32; 4]>>`: on the common path `x_idx` is a DENSE
/// node/slot index into the explicit child level, so a dense scatter groups
/// pairs by x with no hashing.
///
/// PERSISTENCE is the whole point: this lives in `ContractScratch` (taken once
/// per contract run) so the width-sized `stamp`/`slot_of_x` arrays survive
/// across the hundreds of `collect_fusion_plans` calls one contraction makes.
/// A fresh width-sized alloc+init per call (child levels reach ~1M nodes) would
/// dwarf the grouping it replaces. Per node we bump `gen` instead of clearing
/// `stamp` (O(1) reset), only zeroing on the rare u32 wrap.
///
/// `groups` SmallVecs are REUSED across nodes via `clear()`, retaining grown
/// capacity; `touched` records the first-occurrence x order (parallel to the
/// live `groups[0..touched.len()]` prefix — slot i ↔ touched[i]).
#[derive(Default)]
pub(super) struct PFusionScratch {
    /// Per-x generation mark. `stamp[x] == gen` ⟺ x already has a group for the
    /// node currently being processed. Grown on demand to `max(x)+1`.
    pub(super) stamp: Vec<u32>,
    /// x → its group slot (index into `touched`/`groups`). Valid only when
    /// `stamp[x] == gen`; otherwise stale and ignored.
    pub(super) slot_of_x: Vec<u32>,
    /// This node's x's in first-occurrence order. Cleared per node.
    pub(super) touched: Vec<u32>,
    /// Per-group occurrence MULTISET of marginal-side refs (no dedup). Index i holds
    /// the group for `touched[i]`. Reused across nodes via `clear()`; a fresh
    /// `SmallVec` is pushed only when a node needs more distinct x-groups than
    /// any prior node.
    pub(super) groups: Vec<SmallVec<[u32; 4]>>,
    /// Current node's generation stamp. Bumped once per node; on u32 wrap the
    /// stamps are zeroed and it restarts at 1 (0 is the "never stamped"
    /// sentinel, so it must never equal a live generation).
    pub(super) generation: u32,
}

/// Per-call working buffers of `merge::contract_twins`, bundled so the whole
/// set is taken and returned in one move.
///
/// Bundling avoids six fresh allocations (five `Vec`s and one `FxHashSet`) per
/// `contract_twins` call, which on a workload of many tiny diagrams dominates
/// the function's allocator traffic. They
/// cannot live as plain `ContractScratch` fields because the merge path passes
/// `&resolve_keeps` and `&mut scratch` to `compact_and_fork_down` in the same
/// call (two borrows of one struct); moving the bundle OUT of the scratch for
/// the duration of the call keeps the existing body untouched and the borrows
/// disjoint.
#[derive(Default)]
pub(super) struct MergeBuffers {
    /// Kept-node indices whose t1 refs need fork-down resolution. u32-wide,
    /// matching `flat_groups` / `merge_target` — these ARE node indices, and
    /// `compact_and_fork_down` consumes them as `&[u32]`.
    pub(super) resolve_keeps: Vec<u32>,
    /// Overlap-filtered group members eligible for concat merge. u32-wide for
    /// the same reason as `resolve_keeps` — they are node indices.
    pub(super) filtered: Vec<u32>,
    /// Content-equal (duplicate pair list) group members. u32-wide, as above.
    pub(super) duplicate_members: Vec<u32>,
    /// The kept node's sorted pair list, for the content-equality test.
    pub(super) keep_pairs_sorted: Vec<(u32, u32)>,
    /// The candidate member's pair list, sorted for the same test.
    pub(super) member_pairs: Vec<(u32, u32)>,
    /// Pairs already claimed by an accepted member of the current group
    /// (support-overlap detection).
    pub(super) seen_pairs: rustc_hash::FxHashSet<(u32, u32)>,
    /// Pass A's flat selection buffer: every acting group's members
    /// contiguously, SURVIVOR FIRST. u32 node indices, as above.
    pub(super) sel: Vec<u32>,
    /// Pass A's decided per-group actions, each naming its `start..end` range in
    /// `sel`. A `GroupPlan` is a plain `(action, start, end)` POD — it owns no
    /// heap data — so retaining this `Vec` retains one allocation, not a fan-out
    /// of inner ones; no outer-length cap is needed here (contrast
    /// `restructure::relevel`'s `PER_V_PAIRS_RETAIN`, whose elements are `Vec`s).
    pub(super) group_plans: Vec<GroupPlan>,
}

impl MergeBuffers {
    /// Empty every buffer, retaining capacity.
    fn clear(&mut self) {
        self.resolve_keeps.clear();
        self.filtered.clear();
        self.duplicate_members.clear();
        self.keep_pairs_sorted.clear();
        self.member_pairs.clear();
        self.seen_pairs.clear();
        self.sel.clear();
        self.group_plans.clear();
    }

    /// Drop the allocation of any buffer whose retained capacity exceeds
    /// `max_bytes`, INDEPENDENTLY per buffer — same policy (and the same
    /// reasoning) as `return_scratch`'s per-buffer release below. Every buffer
    /// here is cleared on check-out, so a dropped one costs the next
    /// `contract_twins` call one reallocation and nothing else.
    fn release_oversized(&mut self, max_bytes: usize) {
        crate::engine::pool::release_if_oversized(&mut self.resolve_keeps, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.filtered, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.duplicate_members, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.keep_pairs_sorted, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.member_pairs, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.sel, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.group_plans, max_bytes);
        // `FxHashSet` has no `Vec` shape for `release_if_oversized`; its table
        // is `capacity` (u32, u32) entries plus control bytes, so the same
        // element-count bound applies.
        if self.seen_pairs.capacity().saturating_mul(std::mem::size_of::<(u32, u32)>())
            > max_bytes
        {
            self.seen_pairs = rustc_hash::FxHashSet::default();
        }
    }
}

/// Per-node working buffers of `duplicate_pair_resolve::resolve_duplicate_pairs_in_node`.
///
/// Granularity is why these are not taken from the pool directly: fork-down
/// resolves one SURVIVOR NODE per call (`compact_and_fork_down`'s
/// `resolve_keeps` loop), so a per-call `Cell` round-trip would cost more than
/// the three allocations it saves. The bundle lives in `ContractScratch`
/// instead and is handed down by `&mut` from the loop, cleared per node inside
/// the callee.
#[derive(Default)]
pub(super) struct DuplicateScratch {
    /// The node's pair list, decoded to `(left_raw, right_raw)`.
    pub(super) pairs: Vec<(u32, u32)>,
    /// Multiplicity of each distinct pair. Only ever probed by key
    /// (`entry`/`len`) and iterated to build `out`.
    pub(super) counts: rustc_hash::FxHashMap<(u32, u32), u32>,
    /// The rewritten pair list, copied back over the node's slice.
    pub(super) out: Vec<crate::diagram::InputPair>,
}

impl DuplicateScratch {
    /// Empty every buffer, retaining capacity. Called at the top of each
    /// `resolve_duplicate_pairs_in_node` so a handed-down scratch is
    /// indistinguishable from a fresh one; the mid-function `?` bails
    /// therefore need no cleanup.
    pub(super) fn clear(&mut self) {
        self.pairs.clear();
        self.counts.clear();
        self.out.clear();
    }

    /// Per-buffer capacity release, same policy as [`MergeBuffers`].
    fn release_oversized(&mut self, max_bytes: usize) {
        crate::engine::pool::release_if_oversized(&mut self.pairs, max_bytes);
        crate::engine::pool::release_if_oversized(&mut self.out, max_bytes);
        if self.counts.capacity().saturating_mul(std::mem::size_of::<((u32, u32), u32)>())
            > max_bytes
        {
            self.counts = rustc_hash::FxHashMap::default();
        }
    }
}

/// Scratch buffers reused across contract_all_twins calls.
///
/// All buffers are grow-only (never shrunk). Each call to take_scratch() gets
/// the previous call's buffers with their retained capacity, then clears/resizes
/// as needed. This avoids repeated heap allocation on every contraction pass.
#[derive(Default)]
pub(crate) struct ContractScratch {
    // ── find_twin_groups buffers ──
    /// Per-node count of parent pairs referencing it (signature length), then
    /// repurposed in place as the prefix-sum offset table into `entries`.
    ///
    /// u32 because both roles are bounded by the level's CANDIDATE MASS (the
    /// summed fan-out of the twin candidates alone — see
    /// `build_twin_groups_after_collision`). Nothing structural caps a level's
    /// fan-out at 2^32 (`MultiPairRange::start`/`len` are u64), so the bound is not
    /// assumed: the scatter that fills this array counts the mass it wrote and
    /// bails with `OverBudget` before the prefix sum if it reaches u32::MAX.
    /// The same width is already load-bearing for the identical quantity on the
    /// apply side (`SparseWorkspace::rev_offsets_c1`, `pair_counts`).
    pub(super) counts: Vec<u32>,
    /// Flat signature buffer: packed (parent_idx, sibling_idx) entries per node.
    pub(super) entries: Vec<u64>,
    /// Write cursor into `entries` for each node during signature fill, then
    /// reused by the grouping pass as node i's twin representative. u32: the
    /// first role shares `counts`' candidate-mass bound (checked, see above),
    /// the second holds a node index (`NodeIdx` is u32).
    pub(super) cursors: Vec<u32>,
    /// Open-addressing hash table for twin grouping: each slot stores a
    /// fingerprint + occupant index together so a probe is one random load
    /// (instead of loading the index then chasing it to fingerprints[]). An
    /// empty slot has idx == u64::MAX (see EMPTY_SLOT).
    pub(super) twin_hash_table: Vec<TwinSlot>,
    /// Per-node XOR fingerprint of all context hashes (cheap twin pre-screen).
    pub(super) fingerprints: Vec<u64>,
    /// No-reexpand marginal twin contraction only: per-node count of parent
    /// pairs that reference it *via an explicit slot* (inline refs are skipped
    /// in `for_each_target_sibling`, so they don't count). A marginal node with
    /// `sig_len == 0` is referenced by no slot — its count either lives inline
    /// in the parent pairs verbatim (must not be merged) or it is fully dead.
    /// Such empty-signature nodes XOR-collide at fingerprint 0 and would be
    /// falsely grouped as twins; `mark_candidates` excludes them. Only filled
    /// when the marginal+no-reexpand gate is active (kept allocation-free and
    /// byte-identical otherwise).
    pub(super) sig_len: Vec<u32>,
    /// Node indices of twin group members, stored contiguously. u32 because
    /// these ARE node indices: every ref into a level is a `NodeIdx(u32)`
    /// and the contract path already stores them u32-wide (`merge_target`,
    /// `final_remap`), so a level's width is u32-bounded by construction.
    pub(super) flat_groups: Vec<u32>,
    /// Start offsets into `flat_groups` for each twin group. u32: a node belongs
    /// to at most one group, so `flat_groups.len() <= width` (same bound).
    pub(super) group_starts: Vec<u32>,
    /// Per-node "could have a twin" flag: true iff this node shares its context
    /// fingerprint with ≥1 other node. Only candidates get their full signature
    /// materialized in `build_twin_groups_after_collision`; the unique-fingerprint
    /// majority is provably twin-free and skipped.
    pub(super) is_candidate: Vec<bool>,
    /// Per-node "signature slice needs sorting" flag, set during the Pass-2
    /// entry scatter when a written entry compares below its slice predecessor.
    /// Slices arrive from the scatter in parent-index-major order and are
    /// almost always already sorted, so the canonicalization pass sorts only
    /// the flagged minority.
    pub(super) slice_unsorted: Vec<bool>,

    // ── contract_twins buffers ──
    /// Maps old node index → canonical (kept) node index within a twin group.
    pub(super) merge_target: Vec<u32>,
    /// Maps old node index → new compacted index after twin removal.
    pub(super) final_remap: Vec<crate::diagram::NodeIdx>,
    /// Per-node flag: this merged-away node is a content-equal (identical pair
    /// list) twin redirected onto its survivor. The parent rewrite KEEPS its
    /// referencing pairs (remapped onto the survivor) instead of dropping them
    /// — the resulting duplicate (survivor, marginal) parent pairs carry the twin's
    /// multiplicity and are folded by pair fusion into a summed count. Only set at
    /// plain t1 levels under a marginal-flagged parent.
    pub(super) duplicate_redirect: Vec<bool>,
    /// `has_marginal_below[v]` for every vtree node — computed at most once per
    /// scratch checkout (see `duplicate_pair_resolve::compute_has_marginal_below_into`). Drives
    /// the concat-then-fork-down path for overlapping twins at plain levels.
    pub(super) has_marginal_below: Vec<bool>,
    /// Is [`has_marginal_below`](Self::has_marginal_below) filled for the diagram this
    /// checkout is working on? Cleared by `take_scratch`, set by the fill in
    /// `strategies::contract_child`.
    ///
    /// The map is read by one thing — the merge's `t1_scalable` test — so it is
    /// filled on the first merge of a sweep rather than up front: a sweep that
    /// finds no twins (the overwhelmingly common case, and every sweep at all
    /// on a marginal-free diagram) then skips an O(vtree nodes) resize + level scan
    /// it was never going to read. Once filled it stays valid for the rest of
    /// the checkout: contraction merges nodes, and marginalization converts
    /// levels only between compile phases, so no level's `is_marginal()` can
    /// flip underneath it mid-sweep.
    pub(super) has_marginal_below_valid: bool,

    // ── contract_all_twins top-down heap ──
    /// Per-parent dedup flag: true if this parent is currently queued in the
    /// top-down heap (`contract_all_twins_topdown`). Reset when the parent is
    /// popped so the all-false invariant holds on entry/exit.
    pub(super) needs_check: Vec<bool>,

    // ── Same-left pair fusion buffers ──
    /// Generation-stamped scatter reused by `pair_fusion::collect_fusion_plans`
    /// (replaces its former per-boundary grouping `FxHashMap`).
    pub(super) pair_fusion: PFusionScratch,
    /// Boundary-marginal levels for the current `fuse_pairs_inner` sweep
    /// (`diagram::boundary_marginal_levels{,_of}`). A separate field from
    /// `pair_fusion` so the per-boundary plan collection can borrow that one while
    /// this list is being iterated by index.
    pub(super) boundaries: Vec<(crate::vtree::VtreeIdx, crate::vtree::VtreeIdx, crate::diagram::ChildSide)>,

    // ── contract_twins per-call buffers ──
    /// Parked home of the merge path's working buffers; see [`MergeBuffers`].
    /// Empty while a `contract_twins` call has them checked out.
    pub(super) merge: MergeBuffers,
    /// Fork-down duplicate resolution's per-node buffers; see [`DuplicateScratch`].
    /// Borrowed in place (never moved out) — its only user takes it by `&mut`.
    pub(super) duplicate: DuplicateScratch,
}

impl ContractScratch {
    /// Check out the merge buffers, cleared and ready to use. A nested or
    /// early-returning call simply gets a fresh (empty) set.
    pub(super) fn take_merge_buffers(&mut self) -> MergeBuffers {
        let mut b = std::mem::take(&mut self.merge);
        b.clear();
        b
    }

    /// Park the merge buffers back for the next call. Skipping this (an `?`
    /// bail) costs only the buffers' capacity.
    pub(super) fn put_merge_buffers(&mut self, b: MergeBuffers) {
        self.merge = b;
    }
}


pub(super) fn take_scratch(eng: &Engine) -> ContractScratch {
    let mut s: ContractScratch = eng.reduce().contract.take().unwrap_or_default();
    // The parked `has_marginal_below` describes whatever diagram last checked the
    // scratch out. Invalidate on checkout, not on return, so no path can read a
    // stale marginal map even if it bails before parking.
    s.has_marginal_below_valid = false;
    s
}

/// Byte cap on the retained capacity of a SINGLE `ContractScratch` buffer.
/// Mirrors the flat-arena policy `pool_put_bounded` uses on `SCRATCH_*` (same
/// 32 MiB `MAX_LEVEL_ARENA_BYTES`) — these buffers are pooled for the engine's
/// lifetime, so a rare peak level would otherwise park its high-water mark in
/// RSS for the rest of the process.
const CONTRACT_SCRATCH_BYTE_LIMIT: usize = crate::diagram::MAX_LEVEL_ARENA_BYTES;

// ── Allocation-failure injection (test-only) ────────────────────────────────
//
// A one-shot countdown consulted at the fallible reserve sites on the
// `contract_twins` merge path — every one of which sits ahead of the pass's
// first mutation. Armed by the OverBudget-safety regression tests to fire a
// synthetic `ApplyError::OverBudget` at a chosen consult, exercising the
// transactional reserve: a clean pre-mutation bail that must leave the count
// unchanged and the diagram exactly as it was. Compiled out of release
// entirely (no arming path, no consult), so zero production cost.
/// Arm the injection to fire on the `(n+1)`-th consult: the first `n` consults
/// return `false` (counting down), the next returns `true` exactly once and
/// disarms. `arm_fail_after(0)` fires on the very next consult.
#[cfg(test)]
pub(crate) fn arm_fail_after(eng: &Engine, n: u32) {
    eng.reduce().fail_countdown.set(Some(n));
}

/// Disarm the injection so no consult fires.
#[cfg(test)]
pub(crate) fn disarm_fail(eng: &Engine) {
    eng.reduce().fail_countdown.set(None);
}

/// Consult the injection point. Returns `true` exactly once — on the armed
/// consult — and `false` at every other time (including when disarmed).
#[cfg(test)]
pub(super) fn fail_point(eng: &Engine) -> bool {
    let c = &eng.reduce().fail_countdown;
    match c.get() {
        None => false,
        Some(0) => {
            c.set(None);
            true
        }
        Some(k) => {
            c.set(Some(k - 1));
            false
        }
    }
}

pub(super) fn return_scratch(eng: &Engine, mut s: ContractScratch) {
    // Bound every buffer INDEPENDENTLY. The predecessor gated the whole set on
    // `entries.capacity()`, which is sized by the level's candidate mass and is
    // zero on a twin-free level — so on a run of wide twin-free levels the
    // trigger was permanently false while the width-sized buffers beside it
    // (`fingerprints`, `merge_target`, `final_remap`, `pair_fusion.stamp`, …) grew
    // on every call and pinned their high-water mark for the process lifetime.
    //
    // Releasing is free of behavioural consequence: every buffer here is
    // grow-only (`try_resize` never shrinks) and is `fill`ed/`resize`d over the
    // range it is about to be read on, so a dropped buffer costs the next call
    // one reallocation and nothing else. `pair_fusion.stamp` regrows zeroed, which
    // its generation stamp (always ≥ 1) already reads as "never stamped".
    let cap = CONTRACT_SCRATCH_BYTE_LIMIT;
    crate::engine::pool::release_if_oversized(&mut s.counts, cap);
    crate::engine::pool::release_if_oversized(&mut s.entries, cap);
    crate::engine::pool::release_if_oversized(&mut s.cursors, cap);
    crate::engine::pool::release_if_oversized(&mut s.twin_hash_table, cap);
    crate::engine::pool::release_if_oversized(&mut s.fingerprints, cap);
    crate::engine::pool::release_if_oversized(&mut s.sig_len, cap);
    crate::engine::pool::release_if_oversized(&mut s.flat_groups, cap);
    crate::engine::pool::release_if_oversized(&mut s.group_starts, cap);
    crate::engine::pool::release_if_oversized(&mut s.is_candidate, cap);
    crate::engine::pool::release_if_oversized(&mut s.slice_unsorted, cap);
    crate::engine::pool::release_if_oversized(&mut s.merge_target, cap);
    crate::engine::pool::release_if_oversized(&mut s.final_remap, cap);
    crate::engine::pool::release_if_oversized(&mut s.duplicate_redirect, cap);
    crate::engine::pool::release_if_oversized(&mut s.has_marginal_below, cap);
    crate::engine::pool::release_if_oversized(&mut s.needs_check, cap);
    crate::engine::pool::release_if_oversized(&mut s.pair_fusion.stamp, cap);
    crate::engine::pool::release_if_oversized(&mut s.pair_fusion.slot_of_x, cap);
    // `touched`/`groups` are sized by one node's distinct-x count, not by the
    // level width, so the spine bound is the operative one — the `groups`
    // SmallVec inners only spill past 4 refs for a single (node, x) group.
    crate::engine::pool::release_if_oversized(&mut s.pair_fusion.touched, cap);
    crate::engine::pool::release_if_oversized(&mut s.pair_fusion.groups, cap);
    // Same treatment for the parked `contract_twins` merge buffers.
    s.merge.release_oversized(cap);
    s.duplicate.release_oversized(cap);
    eng.reduce().contract.put(Some(s));
}
