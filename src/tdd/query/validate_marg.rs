//! Structural invariant checks for marginal (post-marginalization) TDDs:
//! slot canonicalization, orphan/twin/fusion-redex detection, P-saturation,
//! and model-count-preservation assertions used by the marginalizing compile.

use std::sync::atomic::{AtomicUsize, Ordering};

use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::tdd::types::{InputPair, MargRef, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::tdd::marg_slots::{boundary_marginal_levels, count_key_at, ChildSide, CountKey};

/// TDD-wide marginal invariant check (**I2**, inline discipline): among slots
/// actually REFERENCED from a non-marginal parent's marg-side pair refs, no
/// slot holds an inline-eligible count (`≤ MARG_INLINE_MAX` with no
/// big-overflow entry) — the writer sites inline such counts directly into the
/// parent refs (`MargRef::Inline`).
///
/// Scope (mirrors C3):
/// - **Unreferenced/stale slots are exempt** — count vectors are never shrunk
///   (the no-shrink invariant is described in `types.rs`), so a slot whose
///   refs were all retagged inline legitimately retains its small count.
/// - **Deep-marginal and root levels are exempt** — a marginal level whose
///   parent is itself marginal (or that has no parent) carries no parent pair
///   refs to inline into; its counts (e.g. the final model count) must live in
///   explicit slots regardless of magnitude.
///
/// Slot count uniqueness lives in C3 (`check_slot_count_uniqueness`): with
/// `prune_marg_slots` collecting orphaned slots, ALL slots at a marginal level
/// must carry pairwise-distinct counts at the canonical-form fixpoint.
#[doc(hidden)] // test-support: reached only by integration tests
pub fn check_tdd_marg_invariants(tdd: &Tdd) -> Result<(), String> {
    let mut slots = RefSlotScratch::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        let vlevel = &tdd.levels[v.idx()];
        let Some(counts) = vlevel.marginal_counts.as_ref() else { continue };
        let big = vlevel.marginal_counts_big.as_ref();
        for &s in referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut slots) {
            let i = s as usize;
            if i >= counts.len() {
                continue; // OOB refs are #63's checker's job.
            }
            let has_big = big.and_then(|b| b.get(i)).is_some();
            if counts[i] <= crate::tdd::types::marg_inline_max() as u128 && !has_big {
                return Err(format!(
                    "I2 violation at marg level {} (parent {}) slot {}: referenced count {} \u{2264} marg_inline_max ({}); must be inline at parent refs",
                    v.idx(), parent.idx(), i, counts[i], crate::tdd::types::marg_inline_max(),
                ));
            }
        }
    }
    Ok(())
}

/// Caller-owned scratch for [`referenced_marg_slots`].
///
/// The pass runs once per boundary-marginal level on every slot-prune sweep,
/// and every sweep runs inside the per-merge minimize — so a freshly allocated
/// result `Vec` plus dedup `FxHashSet` per level is pure allocator churn on a
/// workload made of many tiny diagrams (the `--canopy` leaf loop compiles
/// hundreds of thousands of them). One scratch, cleared per level, reused for
/// the whole sweep.
#[derive(Default)]
pub(crate) struct RefSlotScratch {
    /// The deduped, sorted slot list — the pass's result, borrowed by the caller.
    pub(crate) referenced: Vec<u32>,
    seen: FxHashSet<u32>,
}

impl RefSlotScratch {
    /// Empty both buffers, retaining their allocations. The single clear used
    /// both by [`referenced_marg_slots`] (per level) and by the sweep-lifetime
    /// pool in `minimize::slot_prune` (on take), so a pooled scratch differs
    /// from a fresh one only in capacity.
    pub(crate) fn clear(&mut self) {
        self.referenced.clear();
        self.seen.clear();
    }

    /// Drop the allocation of either buffer whose retained capacity exceeds
    /// `max_bytes`, INDEPENDENTLY per buffer — the retention policy
    /// `minimize::contract::scratch` applies field by field. Both are refilled
    /// from scratch on every use, so a released one costs the next sweep one
    /// reallocation and nothing else.
    pub(crate) fn release_oversized(&mut self, max_bytes: usize) {
        crate::tdd::utils::release_if_oversized(&mut self.referenced, max_bytes);
        // `FxHashSet` has no `Vec` shape for `release_if_oversized`; its table is
        // `capacity` u32 entries plus control bytes, so the same element-count
        // bound applies.
        if self.seen.capacity().saturating_mul(std::mem::size_of::<u32>()) > max_bytes {
            self.seen = FxHashSet::default();
        }
    }
}

/// Fill `scratch.referenced` with the slots of a boundary-marginal level that
/// are referenced from `plevel`'s marg-side pair refs (deduped, sorted). Skips
/// ZERO sentinels and inline refs; OOB filtering is the caller's choice.
///
/// Dedup stays hash-based rather than push-then-sort-dedup on purpose: the
/// number of *refs* walked is unbounded (a wide parent level can hold millions
/// of pairs) while the number of *distinct slots* is bounded by the store, so
/// hashing keeps the sort at store size instead of ref-occurrence size.
pub(crate) fn referenced_marg_slots<'a>(
    plevel: &TddLevel,
    side: ChildSide,
    scratch: &'a mut RefSlotScratch,
) -> &'a [u32] {
    scratch.clear();
    let RefSlotScratch { referenced, seen } = scratch;
    for n in 0..plevel.nodes.len() {
        if plevel.nodes[n].is_leaf() {
            continue;
        }
        // Borrowed directly — the old copy-into-a-buffer step was a memcpy of
        // every pair on the level for a read-only walk.
        for p in plevel.pairs_of_idx(n) {
            let raw = match side {
                ChildSide::Right => p.right.0,
                ChildSide::Left => p.left.0,
            };
            if raw & (1u32 << 31) != 0 {
                continue; // ZERO sentinel
            }
            if let MargRef::Slot(s) = MargRef::from_raw(raw) {
                if seen.insert(s) {
                    referenced.push(s);
                }
            }
        }
    }
    referenced.sort_unstable();
    referenced
}

// ── #72 marginal canonical form ──────────────────────────────────────────────
//
// At the fixpoint of contract-twins → canonicalize-by-count → p-fusion, every
// boundary-marginal level `v` (marginal child of a non-marginal parent `P`)
// satisfies a canonical form. A parent node's pairs case-split over pairwise
// mutex non-marginal children (partition invariant), so each node is a
// *function* from distinct x-children to marginal counts:
//
//   (C1) P-saturation — within a parent node, no two pairs share the same
//        non-marginal-side child ref. Established by `apply_p_fusion` (which
//        fuses every same-x group); a later twin-merge can recreate same-x
//        groups, so C1 holds only immediately after a fusion sweep, NOT at
//        arbitrary pipeline points. Parents whose OTHER child is marginal too
//        are exempt — there is no non-marginal side there, and one sweep is not
//        a fixpoint across the parent's two boundaries (see
//        `check_p_saturation`).
//   (C2) Twin canonicality — no two nodes at the parent level have equal pair
//        multisets (such nodes are twins and must have been merged).
//   (C3) Slot count uniqueness — at every marginal level, ALL slot counts are
//        pairwise distinct (the count is the anonymous identity of a marginal
//        node). For compile_marginalize-path stores C3 is a birth invariant:
//        `dedup_fresh_store` (via `SlotInterner`) merges duplicates at mint time.
//        For apply-emit-born stores C3 is established at post-tagger slot-prune
//        (`prune_marg_slots`); emit-site dedup is forbidden on that path (see
//        EMIT-SITE DEDUP IS FORBIDDEN comment in `tididi/src/tdd/transform/pairwise/conjoin/mod.rs`).
//        `prune_marg_slots` also collects orphan slots.
//   (C4) Inline discipline (= I2 above) — no REFERENCED slot holds an
//        inline-eligible count; small counts live inline in the parent refs.
//        Stale slots and deep-marginal/root levels (no parent refs) exempt.
//
// All checks decode marg-side refs with `MargRef::from_raw` (post-tagger
// encoding). Callers at pre-tagger pipeline stages should not use these.

/// Collect node `n`'s pairs.
fn node_pairs_into(level: &TddLevel, n: usize, out: &mut Vec<InputPair>) {
    out.clear();
    out.extend_from_slice(level.pairs_of_idx(n));
}

/// C1: within each boundary-marginal parent node, every non-marginal-side
/// child ref appears in at most one pair. `filter`, when given, restricts the
/// walk to those parent vtree nodes (mirroring `apply_p_fusion_at_parents`).
///
/// BOTH-MARGINAL PARENTS ARE OUT OF SCOPE. When *both* children of `parent`
/// are marginal, `boundary_marginal_levels` yields the parent twice (once per
/// child) and the "x" ref C1 would key on is itself a marg ref, not a
/// non-marginal child — the premise of the invariant (pairwise-mutex explicit
/// children) does not apply. Such a parent is also not a fusion fixpoint after
/// a single sweep: the second boundary's fusion can hand two groups the SAME
/// marg ref by design (count-keyed slot sharing / equal inline counts, see
/// `apply_p_fusion_inner` Phase 2), recreating a same-x group at the first
/// boundary. Those duplicate pairs are sound under multiset pair lists and the
/// next sweep closes them (the explicit side's inline marker is raised by then,
/// so `collect_fusion_plans` takes its opaque-key path). Skipping keeps the
/// check faithful to what C1 actually claims.
pub(crate) fn check_p_saturation(tdd: &Tdd, filter: Option<&[VtreeIdx]>) -> Result<(), String> {
    let mut pairs_buf: Vec<InputPair> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if let Some(f) = filter {
            if !f.contains(&parent) {
                continue;
            }
        }
        let (pleft, pright) = tdd.vtree.children(parent);
        let sibling = match side {
            ChildSide::Left => pright,
            ChildSide::Right => pleft,
        };
        if tdd.levels[sibling.idx()].is_marginal() {
            continue; // both-marginal parent — see doc above
        }
        let plevel = &tdd.levels[parent.idx()];
        for n in 0..plevel.nodes.len() {
            if plevel.nodes[n].is_leaf() {
                continue;
            }
            node_pairs_into(plevel, n, &mut pairs_buf);
            if pairs_buf.len() < 2 {
                continue;
            }
            seen.clear();
            for p in &pairs_buf {
                let x = match side {
                    ChildSide::Right => p.left.0,
                    ChildSide::Left => p.right.0,
                };
                if !seen.insert(x) {
                    return Err(format!(
                        "C1 (P-saturation) violation at parent level {} (marg child {}, side {:?}): \
                         node {} holds \u{2265}2 pairs sharing non-marginal child ref {:#x}",
                        parent.idx(), v.idx(), side, n, x,
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Compute per-level reachability from `tdd.output` by top-down DFS.
///
/// Returns a `Vec<Vec<bool>>` indexed `[vtree_idx][local_node_idx]`.
/// Only walks non-marginal levels (marginal levels have no pairs to propagate
/// through; their "reachability" is from count-slot refs, not node refs).
/// Mirroring the `classic_mark` pass in `prune.rs`.
fn compute_reachable(tdd: &Tdd) -> Vec<Vec<bool>> {
    let num_nodes = tdd.vtree.num_nodes();
    let mut reachable: Vec<Vec<bool>> = (0..num_nodes)
        .map(|i| vec![false; tdd.levels[i].width().max(1)])
        .collect();

    let out_vi = tdd.output.vtree.idx();
    let out_li = tdd.output.local.idx();
    if out_li < reachable[out_vi].len() {
        reachable[out_vi][out_li] = true;
    }

    // Walk top-down (reverse of bottom-up topo order).
    let topo = tdd.vtree.bottomup_topo();
    for v in topo.iter().rev() {
        let t_idx = v.idx();
        if tdd.vtree.node(*v).is_leaf() {
            continue;
        }
        if tdd.levels[t_idx].is_marginal() {
            // Marginal levels have no pair-children to propagate to.
            continue;
        }
        let (left, right) = tdd.vtree.children(*v);
        let left_marg = tdd.levels[left.idx()].is_marginal();
        let right_marg = tdd.levels[right.idx()].is_marginal();

        let level = &tdd.levels[t_idx];
        let width = level.width();
        for i in 0..width {
            if !level.nodes[i].is_internal() {
                continue;
            }
            if i >= reachable[t_idx].len() || !reachable[t_idx][i] {
                continue;
            }
            let pairs: Vec<InputPair> = level.pairs_of_idx(i).to_vec();
            for p in pairs {
                // Left child ref.
                if left_marg {
                    if let MargRef::Slot(s) = MargRef::from_raw(p.left.0) {
                        let lv = &mut reachable[left.idx()];
                        if (s as usize) < lv.len() {
                            lv[s as usize] = true;
                        }
                    }
                    // Inline refs don't reference a node index.
                } else {
                    let li = p.left.idx();
                    let lv = &mut reachable[left.idx()];
                    if li < lv.len() {
                        lv[li] = true;
                    }
                }
                // Right child ref.
                if right_marg {
                    if let MargRef::Slot(s) = MargRef::from_raw(p.right.0) {
                        let rv = &mut reachable[right.idx()];
                        if (s as usize) < rv.len() {
                            rv[s as usize] = true;
                        }
                    }
                } else {
                    let ri = p.right.idx();
                    let rv = &mut reachable[right.idx()];
                    if ri < rv.len() {
                        rv[ri] = true;
                    }
                }
            }
        }
    }
    reachable
}

/// Env-gated detailed C2 violation reporter. Scans the same level set as
/// `check_twin_canonicality` for twin pairs and prints diagnostic detail
/// (reachability, raw pair words, level kinds) for the first `DETAIL_CAP`
/// violations.  Called only from `assert_joint_fixpoint` which is already gated
/// by `canon_check_enabled()`.
fn report_c2_violations_detailed(tdd: &Tdd) {
    static DETAIL_COUNT: AtomicUsize = AtomicUsize::new(0);
    const DETAIL_CAP: usize = 20;

    let reachable = compute_reachable(tdd);
    let mut pairs_buf: Vec<InputPair> = Vec::new();

    for parent in crate::tdd::minimize::contract::content_twin::c2_scan_levels(tdd) {
        let plevel = &tdd.levels[parent.idx()];
        let parent_marg = plevel.is_marginal();
        // parent of parent: to describe parent's "kind"
        let parent_parent_is_marg = tdd.vtree.node(parent).parent()
            .map(|pp| tdd.levels[pp.idx()].is_marginal())
            .unwrap_or(false);
        // Whether this level is a boundary parent (some child marginal) or plain.
        let (cl, cr) = tdd.vtree.children(parent);
        let child_marg =
            tdd.levels[cl.idx()].is_marginal() || tdd.levels[cr.idx()].is_marginal();

        let mut key_to_node: FxHashMap<Vec<(u32, u32)>, usize> = FxHashMap::default();
        for n in 0..plevel.nodes.len() {
            if plevel.nodes[n].is_leaf() {
                continue;
            }
            node_pairs_into(plevel, n, &mut pairs_buf);
            let mut key: Vec<(u32, u32)> =
                pairs_buf.iter().map(|p| (p.left.0, p.right.0)).collect();
            key.sort_unstable();

            if let Some(&m) = key_to_node.get(&key) {
                let prior = DETAIL_COUNT.fetch_add(1, Ordering::Relaxed);
                if prior < DETAIL_CAP {
                    let reach_m = reachable.get(parent.idx())
                        .and_then(|v| v.get(m)).copied().unwrap_or(false);
                    let reach_n = reachable.get(parent.idx())
                        .and_then(|v| v.get(n)).copied().unwrap_or(false);
                    let reach_str = match (reach_m, reach_n) {
                        (true, true)   => "BOTH reachable",
                        (true, false)  => "node_m reachable, node_n UNREACHABLE",
                        (false, true)  => "node_m UNREACHABLE, node_n reachable",
                        (false, false) => "BOTH unreachable (garbage)",
                    };
                    // Raw pair words for the first pair (the twins share the same key).
                    let raw_pairs_hex: Vec<String> = key.iter()
                        .map(|(l, r)| format!("({:#010x},{:#010x})", l, r))
                        .collect();
                    eprintln!(
                        "C2-detail #{}: parent_level={} nodes=({},{}) {} \
                         | pairs_len={} raw={} \
                         | parent_is_marg={} parent_parent_is_marg={} child_marg={}",
                        prior + 1,
                        parent.idx(),
                        m, n,
                        reach_str,
                        key.len(),
                        raw_pairs_hex.join(","),
                        parent_marg,
                        parent_parent_is_marg,
                        child_marg,
                    );
                }
            } else {
                key_to_node.insert(key, n);
            }
        }
    }
}

/// C2: no two non-leaf nodes at any C2-canonicalized level carry equal pair
/// multisets — equal-pair-list nodes are twins and must have merged.
///
/// The level set is `contract::content_twin::c2_scan_levels`, the same one the
/// merge itself walks (every explicit level of a marginalized diagram, empty
/// otherwise), so the checker is exactly the merge's postcondition and cannot
/// drift from it.
pub(crate) fn check_twin_canonicality(tdd: &Tdd) -> Result<(), String> {
    let mut pairs_buf: Vec<InputPair> = Vec::new();
    for parent in crate::tdd::minimize::contract::content_twin::c2_scan_levels(tdd) {
        let plevel = &tdd.levels[parent.idx()];
        let mut key_to_node: FxHashMap<Vec<(u32, u32)>, usize> = FxHashMap::default();
        for n in 0..plevel.nodes.len() {
            if plevel.nodes[n].is_leaf() {
                continue;
            }
            node_pairs_into(plevel, n, &mut pairs_buf);
            let mut key: Vec<(u32, u32)> =
                pairs_buf.iter().map(|p| (p.left.0, p.right.0)).collect();
            key.sort_unstable();
            if let Some(&m) = key_to_node.get(&key) {
                return Err(format!(
                    "C2 (twin canonicality) violation at parent level {}: nodes {} and {} \
                     have identical pair multisets ({} pairs) — unmerged twins",
                    parent.idx(), m, n, key.len(),
                ));
            }
            key_to_node.insert(key, n);
        }
    }
    Ok(())
}

/// C4 garbage-freedom: post-fixpoint-including-prune, boundary stores contain
/// exactly the referenced slots; no orphaned or dead-store entries remain.
///
/// Two garbage classes (mirroring `prune_marg_slots`):
///
/// 1. **Boundary orphans** — every slot index in `0..store_len` at a boundary
///    marginal level (marginal child of a non-marginal parent) must be
///    referenced by at least one parent marg-side ref.  An unreferenced slot is
///    a boundary orphan that `prune_marg_slots` should have collected.
///
/// 2. **Dead deep stores** — a marginal level whose vtree-parent is also
///    marginal holds a dead store (consumed at cascade-marginalize time).
///    After prune the store must be empty (`len == 0`).
///
/// **Exemptions** (mirrors `slot_prune.rs`):
/// - The output level's store (`tdd.output.vtree`) is never touched — it holds
///   the final count or a component sub-TDD's count.
/// - The root marginal level (no parent) is also exempt (its store is the
///   final model-count store, not consumed by any parent).
#[doc(hidden)] // test-support: reached only by integration tests
pub fn check_no_orphan_slots(tdd: &Tdd) -> Result<(), String> {
    let out_v = tdd.output.vtree;
    let mut slots = RefSlotScratch::default();

    // --- (1) Dead deep stores must be empty ---
    for i in 0..tdd.levels.len() {
        if !tdd.levels[i].is_marginal() {
            continue;
        }
        let v = VtreeIdx(i as u32);
        if v == out_v {
            continue;
        }
        let Some(parent) = tdd.vtree.node(v).parent() else { continue };
        if !tdd.levels[parent.idx()].is_marginal() {
            continue; // boundary level: checked below
        }
        // Deep store: parent is also marginal.
        let counts = tdd.levels[i].marginal_counts.as_ref().unwrap();
        if !counts.is_empty() {
            return Err(format!(
                "C4 (garbage-freedom) violation at marg level {} (deep store, \
                 marginal parent {}): expected empty store after prune, \
                 found {} non-empty slots",
                i,
                parent.idx(),
                counts.len(),
            ));
        }
    }

    // --- (2) Boundary stores must have no orphan slots ---
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if v == out_v {
            continue;
        }
        let store_len = tdd.levels[v.idx()]
            .marginal_counts
            .as_ref()
            .map_or(0, |c| c.len());
        if store_len == 0 {
            continue; // nothing to check
        }
        let referenced =
            referenced_marg_slots(&tdd.levels[parent.idx()], side, &mut slots);
        let ref_count = referenced.len();
        if ref_count == store_len {
            // Fast path: every slot is referenced (dense store).
            continue;
        }
        // Build a bitset of referenced slots.
        let mut ref_set = vec![false; store_len];
        for &s in referenced {
            if (s as usize) < store_len {
                ref_set[s as usize] = true;
            }
        }
        for slot in 0..store_len {
            if !ref_set[slot] {
                return Err(format!(
                    "C4 (garbage-freedom) violation at boundary marg level {} \
                     (non-marginal parent {}, side {:?}): slot {} is not \
                     referenced by any parent ref (store_len={}, referenced={})",
                    v.idx(),
                    parent.idx(),
                    side,
                    slot,
                    store_len,
                    ref_count,
                ));
            }
        }
    }

    Ok(())
}

/// C3: at every marginal level, ALL slot count keys are pairwise distinct.
/// The count is the anonymous identity of a marginal node, so two slots with
/// equal counts are the same node stored twice. C3 is enforced at birth by
/// `dedup_fresh_store` for compile_marginalize-path stores, and at post-tagger
/// slot-prune (`prune_marg_slots`) for apply-emit-born stores. This check is
/// a postcondition verifier, not a trigger for a rewrite pass.
#[doc(hidden)] // test-support: reached only by integration tests
pub fn check_slot_count_uniqueness(tdd: &Tdd) -> Result<(), String> {
    let mut key_to_slot: FxHashMap<CountKey, usize> = FxHashMap::default();
    for (li, level) in tdd.levels.iter().enumerate() {
        let Some(counts) = level.marginal_counts.as_ref() else { continue };
        let big = level.marginal_counts_big.as_ref();
        key_to_slot.clear();
        let mut sentinels = 0usize;
        for i in 0..counts.len() {
            let has_big = big.and_then(|b| b.get(i)).is_some();
            if counts[i] == u128::MAX && !has_big {
                return Err(format!(
                    "C3 walk at marg level {li}: slot {i} holds the OVERFLOW \
                     sentinel with no marginal_counts_big entry",
                ));
            }
            sentinels += usize::from(counts[i] == u128::MAX);
            let key = count_key_at(counts, big, i);
            if let Some(&prev) = key_to_slot.get(&key) {
                return Err(format!(
                    "C3 (slot count uniqueness) violation at marg level {li}: \
                     slots {prev} and {i} carry equal counts",
                ));
            }
            key_to_slot.insert(key, i);
        }
        // Other direction of the same invariant: the sparse overflow table is
        // keyed by slot, so an entry whose fast cell is no longer the sentinel
        // is dead weight AND a stale value a later rekey would carry forward.
        // The loop above proved every sentinel has an entry; equal counts then
        // prove there are no extras.
        let entries = big.map_or(0, |b| b.len());
        if entries != sentinels {
            return Err(format!(
                "C3 walk at marg level {li}: marginal_counts_big holds {entries} \
                 entries for {sentinels} OVERFLOW slots — stale entry at a slot \
                 that no longer overflows",
            ));
        }
    }
    Ok(())
}

/// Full #72 canonical-form check: C4/I2 (`check_tdd_marg_invariants`) + C1 +
/// C2 + C3. Valid at the contract → canon → p-fusion → slot-prune fixpoint — in
/// practice: on a freshly minimized TDD immediately after a full
/// `apply_p_fusion` sweep followed by `prune_marg_slots`.
#[doc(hidden)] // test-support: reached only by integration tests
pub fn check_marg_canonical_form(tdd: &Tdd) -> Result<(), String> {
    check_tdd_marg_invariants(tdd)?;
    check_p_saturation(tdd, None)?;
    check_twin_canonicality(tdd)?;
    check_slot_count_uniqueness(tdd)
}

/// No p-fusion redexes: at every boundary-marginal parent level, no node holds
/// two pairs sharing the same EXPLICIT-side child ref. This is the change-C name
/// for C1 (P-saturation): a p-fusion redex is exactly a same-explicit-different-
/// count pair group, so a TDD at the joint fixpoint of twin-contract + p-fusion
/// satisfies this. Delegates to `check_p_saturation` (the full-TDD C1 scan).
///
/// Returns `Err` on the first redex found (same format as C1).
#[doc(hidden)] // test-support: reached only by integration tests
pub fn check_no_fusion_redexes(tdd: &Tdd) -> Result<(), String> {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return Ok(());
    }
    check_p_saturation(tdd, None)
}

/// No unmerged twins: no two distinct non-leaf nodes at a boundary-marginal
/// parent level carry identical pair multisets. This is the change-C name for
/// C2 (twin canonicality): any surviving pair of twins is a contraction redex.
/// Delegates to `check_twin_canonicality` (the full-TDD C2 scan).
///
/// Returns `Err` on the first twin pair found (same format as C2).
#[doc(hidden)] // test-support: reached only by integration tests
pub fn check_no_twins(tdd: &Tdd) -> Result<(), String> {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return Ok(());
    }
    check_twin_canonicality(tdd)
}

/// Joint-fixpoint checker: asserts that `tdd` is at the fixpoint of both
/// twin-contraction and p-fusion (no redexes remain), that marginal slot
/// counts are pairwise distinct (C3), and that no orphaned or garbage slots
/// remain in boundary or deep stores (C4). Valid immediately after a full
/// minimize (`try_minimize` / minimize) including slot-prune. Gated by
/// `TIDIDI_MARG_CANON_CHECK` so it never runs in a normal bench.
///
/// Panics with the first violation found.
pub(crate) fn assert_joint_fixpoint(tdd: &Tdd, label: &str) {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return;
    }
    if !canon_check_enabled() {
        return;
    }
    if let Err(e) = check_no_fusion_redexes(tdd) {
        panic!("TIDIDI_MARG_CANON_CHECK: joint fixpoint violated at [{label}] — {e}");
    }
    if let Err(e) = check_no_twins(tdd) {
        report_c2_violations_detailed(tdd);
        panic!("TIDIDI_MARG_CANON_CHECK: joint fixpoint violated at [{label}] — {e}");
    }
    if let Err(e) = check_slot_count_uniqueness(tdd) {
        panic!("TIDIDI_MARG_CANON_CHECK: joint fixpoint violated at [{label}] — {e}");
    }
    if let Err(e) = check_no_orphan_slots(tdd) {
        panic!("TIDIDI_MARG_CANON_CHECK: joint fixpoint violated at [{label}] — {e}");
    }
}

/// True iff `TIDIDI_MARG_CANON_CHECK` is set (memoized). Controls the joint-
/// fixpoint sweep wired into the minimize exit. Never true in a normal bench
/// (one memoized atomic load, then `None`), so cost is negligible in release.
pub(crate) fn canon_check_enabled() -> bool {
    crate::tdd::config::get().marg_canon_check
}

/// Debug-only enforcement of C1 at the moments it is guaranteed: immediately
/// after an `apply_p_fusion` / `apply_p_fusion_at_parents` sweep (pass the same
/// parent filter the sweep used). Panics with the violation. Compiled out of
/// release builds; always on in debug (cargo test).
///
/// # Panics
///
/// Panics if (P)-saturation is violated (a same-left pair pair that the sweep
/// should have fused survives).
#[cfg(debug_assertions)]
pub fn debug_assert_p_saturated(tdd: &Tdd, filter: Option<&[VtreeIdx]>, label: &str) {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return;
    }
    if let Err(e) = check_p_saturation(tdd, filter) {
        panic!("(P)-saturation violated immediately after p_fusion [{label}]: {e}");
    }
}

// ── #70 count-preservation localizer ─────────────────────────────────────────
//
// A *count-neutral* marginal rewrite — p_fusion, contract's marginal pass,
// reexpand — must leave the TDD's model count
// unchanged: it re-encodes / merges marginal nodes but represents the same set
// of models. The m139 no-reexpand ×2 over-count is exactly such a rewrite (or an
// apply step) silently doubling the count. Computing a full `model_count` is
// EXPENSIVE (BigUint bottom-up over the whole TDD), so this guard is gated behind
// BOTH `debug_assertions` AND the `TIDIDI_MARG_MC_CHECK` env var — it never runs
// in a normal bench. When enabled it pins the doubling op precisely: the first
// wrapped rewrite whose before-count != after-count panics with the op label.
//
// Affordable per-rewrite only on small inputs (e.g. the shrink-search toy CNF);
// on m139-scale every snapshot costs seconds, so prefer the toy repro or a
// coarse per-component placement there. NOT `#[cfg(debug_assertions)]`-gated:
// the env check alone keeps it inert in a normal bench (one memoized atomic load,
// then `None`), and gating on debug only would force a debug build — far too slow
// for the m139-scale repro. Works in `--release`.

// ── C3: slot count uniqueness ─────────────────────────────────────────────────

/// Check that a marginal store satisfies **C3** (each count value appears in at
/// most one slot). Returns `Ok(())` when all slot values are distinct, or
/// `Err(description)` naming the first duplicate pair found.
///
/// This is the constructor invariant for stores built by `dedup_fresh_store`
/// or through a seeded `SlotInterner` map, and also the postcondition for
/// apply-emit-born stores after `prune_marg_slots`. It is weaker than a full
/// `check_marg_invariants` sweep; use it in unit tests immediately after store
/// birth (or after slot-prune) to confirm C3 holds. Production code relies on
/// C3 being guaranteed by construction or slot-prune and does NOT call this on
/// every store.
#[cfg(test)]
pub(crate) fn check_store_counts_c3(
    counts: &[u128],
    big: Option<&crate::tdd::types::BigSide>,
) -> Result<(), String> {
    let mut seen: FxHashMap<CountKey, usize> = FxHashMap::default();
    for i in 0..counts.len() {
        let key = count_key_at(counts, big, i);
        if let Some(&first) = seen.get(&key) {
            return Err(format!(
                "C3 violation: slot {} and slot {} share the same count value ({:?})",
                first, i, key
            ));
        }
        seen.insert(key, i);
    }
    Ok(())
}

/// True iff `TIDIDI_MARG_MC_CHECK` is set (memoized). Off ⇒ snapshots are skipped.
pub(crate) fn mc_check_enabled() -> bool {
    crate::tdd::config::get().marg_mc_check
}

/// Snapshot the TDD's model count *iff* the check is enabled (else `None`,
/// avoiding the expensive count). Pair with [`mc_assert_preserved`] around a
/// count-neutral marginal rewrite.
pub fn mc_snapshot(tdd: &Tdd) -> Option<BigUint> {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return None;
    }
    mc_check_enabled().then(|| crate::tdd::query::model_count(tdd))
}

/// Assert the model count is unchanged vs a prior [`mc_snapshot`]. Panics with
/// the op label on mismatch — this is the #70 doubler localizer. No-op when the
/// snapshot was `None` (check disabled).
///
/// # Panics
///
/// Panics if the current model count differs from `before` (a count-neutral op
/// changed the count). No-op when `before` is `None`.
pub fn mc_assert_preserved(tdd: &Tdd, before: Option<BigUint>, op: &str) {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return;
    }
    let Some(before) = before else { return };
    let after = crate::tdd::query::model_count(tdd);
    if after != before {
        // Surface the multiplicative factor (×2 for the m139 doubler) when it
        // divides cleanly, to make the signature unmistakable in the panic.
        let factor = if before != BigUint::ZERO && &after % &before == BigUint::ZERO {
            format!(" (after = {}× before)", &after / &before)
        } else {
            String::new()
        };
        panic!(
            "TIDIDI_MARG_MC_CHECK: count-neutral op `{op}` CHANGED the model count{factor}\n  \
             before = {before}\n  after  = {after}"
        );
    }
}

/// Forward slot-bounds checker (#63). Every marg-side parent ref into a
/// *marginal* child must decode to: ZERO (bit-31), an inline count (bit-30
/// clear), or a slot index `< child.marginal_counts.len()`. A tagged slot
/// `>= len` means a parent ref out-ran a shrunk child store — the no-reexpand
/// over-mint that OOBs the downstream `read_marg_count`. Returns the first
/// `(parent_t, child_idx, side, slot, child_len)` violation in bottom-up order,
/// or `None`. Debug-only diagnostic; visits every marg-slot ref a reader can
/// follow (inline node `a`/`b` + every pair `left`/`right`).
///
/// Debug-only like its callers (the env-gated release diagnostics that once
/// needed it present in release builds were retired with their env vars).
#[cfg(debug_assertions)]
pub(crate) fn first_marg_slot_oob_levels(
    levels: &[TddLevel],
    vtree: &crate::vtree::Vtree,
) -> Option<(usize, usize, char, u32, usize)> {
    use crate::tdd::types::{MARG_OVERFLOW_TAG, MARG_VALUE_MASK, decode_marg_coord};
    let oob = |raw: u32, len: usize| -> Option<u32> {
        if raw & (1 << 31) != 0 {
            return None; // ZERO sentinel
        }
        if raw & MARG_OVERFLOW_TAG != 0 {
            return None; // inline count (bit-30 set) — store-independent
        }
        let slot = decode_marg_coord(raw, MARG_VALUE_MASK); // bit-30 clear ⟹ bare slot index
        if (slot as usize) >= len { Some(slot) } else { None }
    };
    for (t, lc, rc) in vtree.internal_bottomup() {
        let (t, lc, rc) = (t.idx(), lc.idx(), rc.idx());
        let l_marg = levels[lc].is_marginal();
        let r_marg = levels[rc].is_marginal();
        if !l_marg && !r_marg {
            continue;
        }
        // A marginal parent has no pairs and propagates nothing (its children are
        // orphaned once its own pairs go). Only a non-marginal parent's *live*
        // node pairs are consumed against the child stores — mirror that exactly,
        // walking per-live-node pair ranges (the `pairs` arena is NOT compacted
        // by prune, so iterating it raw would hit dead refs → false positives).
        if levels[t].is_marginal() {
            continue;
        }
        let l_len = levels[lc].marginal_counts.as_ref().map_or(0, |c| c.len());
        let r_len = levels[rc].marginal_counts.as_ref().map_or(0, |c| c.len());
        let lvl = &levels[t];
        let width = lvl.nodes.len();
        for i in 0..width {
            if !lvl.nodes[i].is_internal() {
                continue;
            }
            macro_rules! check_pair {
                ($pair:expr) => {{
                    let p = $pair;
                    if l_marg {
                        if let Some(s) = oob(p.left.0, l_len) {
                            return Some((t, lc, 'L', s, l_len));
                        }
                    }
                    if r_marg {
                        if let Some(s) = oob(p.right.0, r_len) {
                            return Some((t, rc, 'R', s, r_len));
                        }
                    }
                }};
            }
            for p in lvl.pairs_of_idx(i) {
                check_pair!(*p);
            }
        }
    }
    None
}

/// Shared toy-TDD builder for marginal-invariant unit tests (validate_marg + slot_prune).
#[cfg(test)]
#[path = "validate_marg_test_fixtures.rs"]
pub(crate) mod test_fixtures;

#[cfg(test)]
#[path = "validate_marg_canonical_form_tests.rs"]
mod canonical_form_tests;

// ── Change-C: joint-fixpoint property tests ──────────────────────────────────
//
// The compile-driven property tests (property tests over `compile_cnf_mc` +
// the end-to-end cascade zero-footprint test), including their `make_cnf` /
// `brute_force_mc_raw` helpers, moved to `tests/tdd_validate_marg_compile.rs`
// (crate-split: `tididi` cannot depend on CNF parsing, which lives in the
// CNF front end, or on compilation, which lives in the downstream driver
// crate).
//
// The directed hand-built fixture for fusion-redex → twin is in contract.rs's
// test module so it can access the private `contract_all_twins_topdown` directly.

#[cfg(debug_assertions)]
thread_local! {
    static SLOT_CHECK_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SLOT_CHECK_FIRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Latched reporter wrapping [`first_marg_slot_oob_levels`] (#63). Env gated
/// (`TIDIDI_MARG_SLOT_CHECK`). Each call bumps a sequence counter; the *first*
/// call (across all pass exits) that finds a violation prints and latches, so
/// the earliest birth site/pass wins and the rest stay silent. `label` names
/// the pass exit (e.g. "APPLY", "PRUNE", "CONTRACT").
#[cfg(debug_assertions)]
pub(crate) fn marg_slot_check(tdd: &Tdd, label: &str) {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return;
    }
    marg_slot_check_levels(&tdd.levels, &tdd.vtree, label);
}

/// `(levels, vtree)` form of [`marg_slot_check`] for apply-internal call sites.
#[cfg(debug_assertions)]
pub(crate) fn marg_slot_check_levels(levels: &[TddLevel], vtree: &crate::vtree::Vtree, label: &str) {
    // weighted mode: marginal levels carry no integer counts; skip count-reading dedup/checks.
    if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        return;
    }
    if !crate::tdd::config::get().marg_slot_check {
        return;
    }
    let seq = SLOT_CHECK_SEQ.with(|c| {
        let v = c.get() + 1;
        c.set(v);
        v
    });
    if SLOT_CHECK_FIRED.with(|c| c.get()) {
        return;
    }
    if let Some((t, child, side, slot, len)) = first_marg_slot_oob_levels(levels, vtree) {
        SLOT_CHECK_FIRED.with(|c| c.set(true));
        eprintln!(
            "MARG_SLOT_OOB FIRST label={label} seq={seq} parent_t={t} child={child} side={side} slot={slot} child_len={len}"
        );
    }
}


// ── Unit tests for C3 construction invariant ─────────────────────────────────
//
// Each test constructs a duplicate-prone store and asserts that after
// `dedup_fresh_store` (or SlotInterner) the result satisfies C3 immediately —
// no post-hoc canon pass required. Tests are authored for compilation; run
// with `cargo test` (no --include-ignored needed).
#[cfg(test)]
#[path = "validate_marg_c3_tests.rs"]
mod c3_tests;
