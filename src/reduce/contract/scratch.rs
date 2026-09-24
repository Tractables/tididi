use crate::limits::pool::PooledScratch;

use smallvec::SmallVec;


use super::merge::GroupPlan;

// Arena growth here goes through `limits`' `try_push` / `try_resize`, the one
// budget-charged implementation; a raw `try_reserve_exact(1)` would bypass the
// budget charge and grow by one per push at capacity.

// ── Engine-owned scratch buffer pool ────────────────────────────────────────
//
// All scratch buffers are bundled into one struct, taken from the engine at
// the start of a contract run and returned at the end.

/// Open-addressing slot for the twin-grouping tables: fingerprint co-located
/// with the occupant index so a probe costs one random load instead of two
/// (`ht[slot]` then `fingerprints[occupant]`). An `idx` of `u64::MAX` marks an empty slot.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct TwinSlot {
    pub(super) fp: u64,
    pub(super) idx: u64,
}

pub(super) const EMPTY_SLOT: TwinSlot = TwinSlot { fp: 0, idx: u64::MAX };

/// One cell of the grouping table in [`PFusionScratch`].
#[derive(Clone, Copy, Default)]
pub(super) struct GroupCell {
    /// The generation that last wrote this cell. `stamp == generation` ⟺ the
    /// cell holds a group of the node currently being processed.
    pub(super) stamp: u32,
    /// The explicit-side ref the group is keyed on. Valid only when stamped
    /// with the current generation; otherwise stale and ignored.
    pub(super) key: u32,
    /// The group slot (index into `touched`/`groups`). Same validity as `key`.
    pub(super) slot: u32,
}

/// Reusable generation-stamped grouping table for
/// `pair_fusion::collect_fusion_plans`' per-node same-explicit-side grouping.
/// The key is the raw explicit-side ref (a node or slot index, or an inline
/// count on a both-marginal parent), hashed into an open-addressing table
/// sized by the node's pair count. Per node the generation is bumped instead
/// of clearing the cells, zeroing only on u32 wrap. `groups` entries are
/// reused across nodes via `clear()`; `touched` records the first-occurrence x
/// order, slot `i` ↔ `touched[i]`.
#[derive(Default)]
pub(super) struct PFusionScratch {
    /// The grouping table. Its length is a power of two, at least twice the
    /// pair count of the widest node grouped so far, so a probe always finds
    /// an unstamped cell.
    pub(super) cells: Vec<GroupCell>,
    /// This node's x's in first-occurrence order. Cleared per node.
    pub(super) touched: Vec<u32>,
    /// Per-group occurrence multiset of marginal-side refs (no dedup). Index i holds
    /// the group for `touched[i]`. Reused across nodes via `clear()`; a fresh
    /// `SmallVec` is pushed only when a node needs more distinct x-groups than
    /// any prior node.
    pub(super) groups: Vec<SmallVec<[u32; 4]>>,
    /// Current node's generation stamp. Bumped once per node; on u32 wrap the
    /// stamps are zeroed and it restarts at 1 (0 is the "never stamped"
    /// sentinel, so it must never equal a live generation).
    pub(super) generation: u32,
}

impl PFusionScratch {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        self.groups.iter().filter(|group| group.spilled()).fold(
            capacity_bytes(&self.cells).saturating_add(capacity_bytes(&self.touched)).saturating_add(capacity_bytes(&self.groups)),
            |bytes, group| bytes.saturating_add(group.capacity().saturating_mul(std::mem::size_of::<u32>())),
        )
    }
}

/// Per-level working buffers for planning and committing twin merges.
#[derive(Default)]
pub(super) struct MergeBuffers {
    /// Kept-node indices whose t1 refs need fork-down resolution. u32-wide,
    /// matching `flat_groups` / `merge_target` — node indices in both cases, and
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
    /// (support-overlap detection). A map to `()` rather than a set so its
    /// growth goes through `Limits::reserve_map`.
    pub(super) seen_pairs: rustc_hash::FxHashMap<(u32, u32), ()>,
    /// Pass A's flat selection buffer: every acting group's members
    /// contiguously, survivor first. u32 node indices, as above.
    pub(super) sel: Vec<u32>,
    /// Pass A's decided per-group actions, each naming its `start..end` range in
    /// `sel`.
    pub(super) group_plans: Vec<GroupPlan>,
}

impl MergeBuffers {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        [
            capacity_bytes(&self.resolve_keeps),
            capacity_bytes(&self.filtered),
            capacity_bytes(&self.duplicate_members),
            capacity_bytes(&self.keep_pairs_sorted),
            capacity_bytes(&self.member_pairs),
            capacity_bytes(&self.seen_pairs),
            capacity_bytes(&self.sel),
            capacity_bytes(&self.group_plans),
        ].into_iter().sum()
    }

    /// Empty every buffer, retaining capacity.
    pub(super) fn clear(&mut self) {
        self.resolve_keeps.clear();
        self.filtered.clear();
        self.duplicate_members.clear();
        self.keep_pairs_sorted.clear();
        self.member_pairs.clear();
        self.seen_pairs.clear();
        self.sel.clear();
        self.group_plans.clear();
    }

    /// Drop the allocation of any buffer whose retained capacity exceeds the
    /// scratch-retention cap, the same policy as `PooledScratch::retain`.
    fn release_oversized(&mut self, lim: &crate::limits::Limits) {
        crate::limits::pool::release_if_oversized(lim, &mut self.resolve_keeps);
        crate::limits::pool::release_if_oversized(lim, &mut self.filtered);
        crate::limits::pool::release_if_oversized(lim, &mut self.duplicate_members);
        crate::limits::pool::release_if_oversized(lim, &mut self.keep_pairs_sorted);
        crate::limits::pool::release_if_oversized(lim, &mut self.member_pairs);
        crate::limits::pool::release_if_oversized(lim, &mut self.sel);
        crate::limits::pool::release_if_oversized(lim, &mut self.group_plans);
        crate::limits::pool::release_if_oversized(lim, &mut self.seen_pairs);
    }
}

/// Per-node working buffers of `duplicate_pair_resolve::resolve_duplicate_pairs_in_node`,
/// held in `ContractScratch` and handed down by `&mut` from the fork-down loop,
/// cleared per node inside the callee.
#[derive(Default)]
pub(super) struct DuplicateScratch {
    /// The node's pair list, decoded to `(left_raw, right_raw)`.
    pub(super) pairs: Vec<(u32, u32)>,
    /// Multiplicity of each distinct pair. Only ever probed by key
    /// (`entry`/`len`) and iterated to build `out`.
    pub(super) counts: rustc_hash::FxHashMap<(u32, u32), u32>,
    /// The rewritten pair list, copied back over the node's slice.
    pub(super) out: Vec<crate::diagram::ChildPair>,
}

impl DuplicateScratch {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        [
            capacity_bytes(&self.pairs),
            capacity_bytes(&self.counts),
            capacity_bytes(&self.out),
        ].into_iter().sum()
    }

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
    fn release_oversized(&mut self, lim: &crate::limits::Limits) {
        crate::limits::pool::release_if_oversized(lim, &mut self.pairs);
        crate::limits::pool::release_if_oversized(lim, &mut self.out);
        crate::limits::pool::release_if_oversized(lim, &mut self.counts);
    }
}

/// Node redirects and compacted indices produced by twin contraction.
#[derive(Default)]
pub(super) struct MergeRemap {
    /// Maps old node index → canonical (kept) node index within a twin group.
    pub(super) merge_target: Vec<u32>,
    /// Maps old node index → new compacted index after twin removal.
    pub(super) final_remap: Vec<crate::diagram::NodeIdx>,
    /// Per-node flag: this merged-away node is a content-equal (identical pair
    /// list) twin redirected onto its survivor. The parent rewrite keeps its
    /// referencing pairs (remapped onto the survivor) rather than dropping them
    /// — the resulting duplicate (survivor, marginal) parent pairs carry the twin's
    /// multiplicity and are folded by pair fusion into a summed count. Only set at
    /// plain t1 levels under a marginal-flagged parent.
    pub(super) duplicate_redirect: Vec<bool>,
}

/// Scratch buffers reused across `contract_all_twins` calls.
///
/// Each checkout reuses bounded capacity from the previous call and
/// invalidates the cached marginal map before scanning another diagram.
#[derive(Default)]
pub(crate) struct ContractScratch {
    // ── `find_twin_groups` buffers ──
    // Node indices and per-node counts are u32-wide throughout: every ref into
    // a level is a `NodeIdx(u32)`, and the candidate mass that bounds `counts`
    // and `cursors` is checked against `u32::MAX` by `count_candidate_entries`
    // before any offset is stored.
    /// Per-node count of parent pairs referencing it (signature length), then
    /// repurposed in place as the prefix-sum offset table into `entries`.
    pub(super) counts: Vec<u32>,
    /// Flat signature buffer: packed (parent_idx, sibling_idx) entries per node.
    pub(super) entries: Vec<u64>,
    /// Write cursor into `entries` for each node during signature fill, then
    /// reused by the grouping pass as node i's twin representative.
    pub(super) cursors: Vec<u32>,
    /// Open-addressing hash table for twin grouping; see [`TwinSlot`].
    pub(super) twin_hash_table: Vec<TwinSlot>,
    /// Per-node fingerprint, combining all context hashes (a cheap twin pre-screen).
    pub(super) fingerprints: Vec<u64>,
    /// Node indices of twin group members, stored contiguously.
    pub(super) flat_groups: Vec<u32>,
    /// Start offsets into `flat_groups` for each twin group; a node belongs to
    /// at most one group, so `flat_groups.len() <= width`.
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

    // ── `contract_twins` buffers ──
    /// Twin survivor selection and compacted node indices.
    pub(super) remap: MergeRemap,
    /// `has_marginal_below[v]` for every vtree node — computed at most once per
    /// scratch checkout (see `duplicate_pair_resolve::compute_has_marginal_below_into`). Drives
    /// the concat-then-fork-down path for overlapping twins at plain levels.
    pub(super) has_marginal_below: Vec<bool>,
    /// Is [`has_marginal_below`](Self::has_marginal_below) filled for the diagram
    /// this checkout is working on? Cleared by checkout, set by the fill in
    /// `sweep::contract_child`, which runs on the first merge of a sweep
    /// rather than up front, since a sweep that finds no twins never reads it.
    pub(super) has_marginal_below_valid: bool,

    // ── `contract_all_twins` top-down heap ──
    /// Per-parent dedup flag: true if this parent is currently queued in the
    /// top-down heap (`contract_all_twins`). Reset when the parent is
    /// popped so the all-false invariant holds on entry/exit.
    pub(super) needs_check: Vec<bool>,

    // ── Same-left pair fusion buffers ──
    /// Generation-stamped grouping table reused by
    /// `pair_fusion::collect_fusion_plans`.
    pub(super) pair_fusion: PFusionScratch,
    /// Boundary-marginal levels for the current `fuse_pairs_inner` sweep
    /// (`diagram::boundary_marginal_levels_into`). A separate field from
    /// `pair_fusion` so the per-boundary plan collection can borrow that one while
    /// this list is being iterated by index.
    pub(super) boundaries: Vec<(crate::vtree::VtreeIdx, crate::vtree::VtreeIdx, crate::diagram::ChildSide)>,

    // ── `contract_twins` per-call buffers ──
    /// Working buffers cleared before planning each level's merges.
    pub(super) merge: MergeBuffers,
    /// Fork-down duplicate resolution's per-node buffers; see [`DuplicateScratch`].
    /// Borrowed in place (never moved out) — its only user takes it by `&mut`.
    pub(super) duplicate: DuplicateScratch,
}

impl PooledScratch for ContractScratch {
    fn retained_bytes(&self) -> usize {
        use crate::limits::pool::capacity_bytes;
        [
            capacity_bytes(&self.counts),
            capacity_bytes(&self.entries),
            capacity_bytes(&self.cursors),
            capacity_bytes(&self.twin_hash_table),
            capacity_bytes(&self.fingerprints),
            capacity_bytes(&self.flat_groups),
            capacity_bytes(&self.group_starts),
            capacity_bytes(&self.is_candidate),
            capacity_bytes(&self.slice_unsorted),
            capacity_bytes(&self.remap.merge_target).saturating_add(capacity_bytes(&self.remap.final_remap)).saturating_add(capacity_bytes(&self.remap.duplicate_redirect)),
            capacity_bytes(&self.has_marginal_below),
            capacity_bytes(&self.needs_check),
            self.pair_fusion.retained_bytes(),
            capacity_bytes(&self.boundaries),
            self.merge.retained_bytes(),
            self.duplicate.retained_bytes(),
        ].into_iter().sum()
    }

    fn prepare(&mut self) {
        // The parked `has_marginal_below` describes whatever diagram last checked the
        // scratch out. Invalidate on checkout, not on return, so no path can read a
        // stale marginal map even if it bails before parking.
        self.has_marginal_below_valid = false;
    }

    fn retain(&mut self, lim: &crate::limits::Limits) {
        // Bound each buffer against its own capacity, not against one buffer
        // standing in for the set: `entries` is empty on a twin-free level, so
        // gating on it would let the width-sized buffers grow unchecked over a run
        // of wide twin-free levels. Releasing has no behavioural consequence: every
        // buffer is filled or resized over the range it is read on, so a dropped
        // one costs the next call a reallocation; `pair_fusion.cells` regrows
        // zeroed, which its generation stamp (always ≥ 1) reads as never stamped.
        crate::limits::pool::release_if_oversized(lim, &mut self.counts);
        crate::limits::pool::release_if_oversized(lim, &mut self.entries);
        crate::limits::pool::release_if_oversized(lim, &mut self.cursors);
        crate::limits::pool::release_if_oversized(lim, &mut self.twin_hash_table);
        crate::limits::pool::release_if_oversized(lim, &mut self.fingerprints);
        crate::limits::pool::release_if_oversized(lim, &mut self.flat_groups);
        crate::limits::pool::release_if_oversized(lim, &mut self.group_starts);
        crate::limits::pool::release_if_oversized(lim, &mut self.is_candidate);
        crate::limits::pool::release_if_oversized(lim, &mut self.slice_unsorted);
        crate::limits::pool::release_if_oversized(lim, &mut self.remap.merge_target);
        crate::limits::pool::release_if_oversized(lim, &mut self.remap.final_remap);
        crate::limits::pool::release_if_oversized(lim, &mut self.remap.duplicate_redirect);
        crate::limits::pool::release_if_oversized(lim, &mut self.has_marginal_below);
        crate::limits::pool::release_if_oversized(lim, &mut self.needs_check);
        // The grouping table, `touched` and `groups` are sized by one node's pair
        // count, not by the level width, so the spine bound is the operative one —
        // the `groups` SmallVec inners only spill past 4 refs for a single
        // (node, x) group.
        crate::limits::pool::release_if_oversized(lim, &mut self.pair_fusion.cells);
        crate::limits::pool::release_if_oversized(lim, &mut self.pair_fusion.touched);
        crate::limits::pool::release_if_oversized(lim, &mut self.pair_fusion.groups);
        // Same treatment for the parked `contract_twins` merge buffers.
        self.merge.release_oversized(lim);
        self.duplicate.release_oversized(lim);
    }
}
