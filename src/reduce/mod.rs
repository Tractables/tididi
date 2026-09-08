//! TDD minimization: reduce a TDD to its canonical (smallest) form.
//!
//! This module is a thin ORCHESTRATOR over two self-contained phase modules —
//! prune and twin-contraction. `mod.rs` owns the public entry points
//! (`minimize`, `try_minimize`, `minimize_after_rotation`,
//! the `instrumented_prune`/`contract_only{,_at}` phase wrappers), the shared
//! config cells, and the phase wrappers.
//! The phase mechanisms themselves live in the submodules:
//!
//! 1. **Prune** (`prune.rs`): remove nodes not reachable from the output.
//!    Top-down reachability mark, then bottom-up compaction with monotone remap.
//! 2. **Twin contraction** (`contract/`): merge nodes with identical parent
//!    context (same set of (parent node, sibling) pairs). Twins compute
//!    functions whose disjunction can replace them both without affecting the output.
//!    Submodules: `contract/strategies.rs` (inner-node twins), `contract/contract_leaf.rs`
//!    (leaf-side specialization), `contract/p_fusion.rs` (same-left pair fusion),
//!    and `contract/content_twin.rs` (the content-equal merge
//!    mechanism, driven by the `canonicalize_content_twins` loop in `content_twins.rs`).
//!
//! **Prune ↔ contract interface.** The two phases are decoupled except through
//! the `Tdd` dirty-contract worklists: prune (and the content-twin merge) call
//! `Tdd::mark_contract_dirty`, seeding the `dirty_contract`/`dirty_leaf_contract`
//! worklists that the contract pass then drains. This shared
//! state is the ONLY coupling — neither phase reaches into the other's internals.
//!
//! Compress (node deduplication) is provably unnecessary and has been removed:
//! - After `apply_and`: canonical leaf ordering guarantees no duplicate nodes
//!   by induction (verified empirically: 0 dedup triggers across 167 benchmarks).
//! - After prune: the monotone remap preserves node distinctness.

mod prune;
// `pub` for the path to `contract::p_fusion` (binary caller: compile/step.rs).
pub(crate) mod contract;
pub(crate) mod slot_prune; // post-tagger marginal-slot compaction (binary caller: compile/step.rs)
mod content_twins;

#[cfg(test)]
pub(crate) use content_twins::canonicalize_content_twins;


// ── Minimize options ─────────────────────────────────────────────────────────

/// Which reduction passes [`try_minimize`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MinimizePasses {
    /// Prune, twin + leaf-twin contraction, marginal-slot prune and the
    /// content-twin canonicalization — the full canonical form.
    #[default]
    Full,
    /// Prune unreachable nodes and compact the marginal count slots they
    /// orphaned; no contraction. Keeps a diagram clean without paying for the
    /// scatter-write contraction pass.
    PruneOnly,
    /// Inner-node twin contraction only — no prune, no leaf-twin pass, no
    /// content-twin scan.
    ContractOnly,
}

/// Scheduling state for the content-twin canonicalization scan above its
/// size cap: below the cap every minimize scans, above it the first call scans
/// (`next_at` starts at 0) and the next probe is scheduled at 4x the pre-scan
/// size — unless the scan landed back under the cap, which resets `next_at` to
/// 0 so the next above-cap call scans again.
///
/// A caller that minimizes a *fresh* diagram each step (a bottom-up compile
/// accumulator, say) must keep one of these across the steps and hand it to
/// [`MinimizeOptions::content_twin_probe`]; state carried on the diagram itself
/// would reset to "always scan" every step. Passing none is equivalent to
/// passing a fresh probe: the scan runs and the updated schedule is discarded.
#[derive(Debug, Clone, Default)]
pub struct ContentTwinProbe {
    /// Node count at which a skipped (above-cap) scan is re-attempted.
    /// 0 = scan on the next above-cap call.
    pub next_at: u64,
}

/// What [`try_minimize`] should do.
///
/// `MinimizeOptions::default()` is the full canonical reduction with no probe
/// state carried across calls.
#[derive(Debug, Default)]
pub struct MinimizeOptions<'a> {
    /// Which passes to run.
    pub passes: MinimizePasses,
    /// Skip the content-twin canonicalization pass. It is size- and
    /// canonicity-only — never count-affecting — so skipping it is sound, and
    /// worth it on a diagram that is about to be discarded or split. Ignored
    /// unless `passes` is [`MinimizePasses::Full`].
    pub skip_content_twins: bool,
    /// Probe schedule for the content-twin scan, carried across calls by the
    /// caller. See [`ContentTwinProbe`].
    pub content_twin_probe: Option<&'a mut ContentTwinProbe>,
}

use self::contract::contract_leaf::contract_leaf_twins;
use self::contract::contract_all_twins;
#[cfg(debug_assertions)]
use self::contract::contract_all_twins_with_locality;
use self::prune::prune_unreachable;
use crate::limits::ApplyError;
use crate::vtree::VtreeIdx;
use crate::diagram::Tdd;

/// Snapshot per-level `is_marginal` flags so a later
/// `assert_no_demarginalization` can detect a violation of I1 (marginality is
/// permanent — see `validate::marg`) and name the offending pass.
#[cfg(debug_assertions)]
fn snapshot_marginal_flags(tdd: &Tdd) -> Vec<bool> {
    tdd.levels.iter().map(|l| l.is_marginal()).collect()
}

/// Assert no level present in `before` (as marginal) has become structural.
/// Panics naming the level and the `pass` that violated I1.
#[cfg(debug_assertions)]
fn assert_no_demarginalization(tdd: &Tdd, before: &[bool], pass: &str) {
    for (i, &was_marg) in before.iter().enumerate() {
        if was_marg && !tdd.levels[i].is_marginal() {
            panic!(
                "I1 invariant violated: vtree level {i} was marginal before `{pass}` \
                 but is structural after — minimize must never un-marginalize a node \
                 A marginal node's mass may only roll UP into a \
                 marginalized parent, never be discarded."
            );
        }
    }
}

// ── Phase wrappers ───────────────────────────────────────────────────────
//
// Each wrapper calls a minimize phase. Every minimize variant goes through
// these, so the phase boilerplate lives in exactly one place.

/// Run prune. Fallible: prune's `total`-proportional scratch buffers are
/// `try_reserve`-guarded (multi-GiB on a blown-up diagram); on `Err` the
/// diagram is untouched (well-formed, not poisoned).
fn instrumented_prune(tdd: &mut Tdd) -> Result<(), ApplyError> {
    prune_unreachable(tdd)
}

// ── Public minimize variants ─────────────────────────────────────────────

/// Minimize a TDD to its canonical form.
///
/// Two phases:
/// 1. **Prune**: remove nodes not reachable from the output
/// 2. **Twin contraction**: merge nodes with identical parent context
///
/// After minimization, each vtree level has exactly `S_t` nodes (one per
/// non-trivial X_t-subfunction).
///
/// **Allocation failure**: this entry point is infallible, for callers that do
/// not want to thread a `Result` through their plumbing. It PANICS when an
/// allocation is refused. A caller that must survive a refusal — by splitting
/// the diagram, or by giving the reduction more room — calls [`try_minimize`]
/// and handles [`ApplyError::OverBudget`].
///
/// **Deadline shield**, the same one the infallible apply entries install: this
/// entry has nowhere to report a cut to, so a deadline poll firing here would
/// turn a budget expiry into a panic. Clearing the installed deadline for the
/// call's lifetime makes the poll unreachable from inside it and restores the
/// caller's deadline on drop, so the fallible entries keep cutting as before.
pub fn minimize(tdd: &mut Tdd) {
    let _shield = crate::limits::apply_limits().deadline(None).apply();
    try_minimize(tdd, MinimizeOptions::default())
        .expect("minimize: an allocation was refused; use try_minimize to handle it");
}

/// Fallible version of `minimize` — returns `Err(OverBudget)` if any internal
/// allocation is refused (OS allocator under `RLIMIT_AS`, or — when wired into
/// the budget tracker — the soft apply-budget envelope). Callers in the
/// hot compile loop use this so v-split recovery can engage on contract OOMs
/// instead of the process dying.
///
/// ## Error contract
///
/// - `Err(Deadline)` ⇒ the diagram is left **well-formed** (a clean early exit
///   at a pass boundary); the caller may keep and count it.
/// - `Err(OverBudget)` ⇒ well-formed **unless** `tdd.scratch.poisoned` is set. Twin
///   contraction reserves its whole arena growth up front, so a cross-group
///   `OverBudget` bails before any mutation; the one irreducible allocation in
///   the middle of a parent rewrite sets `tdd.scratch.poisoned` on failure.
/// - `tdd.scratch.poisoned == true` ⇒ the structure is inconsistent and its count is
///   unreliable. The caller MUST drop the diagram (recovery / abort the segment
///   attempt) — never count it or feed it to another apply. `model_count`
///   asserts `!poisoned` as a backstop.
///
/// So on `Err`: keep-and-continue is sound iff `!tdd.scratch.poisoned`; a poisoned TDD
/// must be discarded.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a budget-gated reduction step is
/// refused. On `Err` the diagram is sound unless `tdd.scratch.poisoned` is set (see above).
pub fn try_minimize(tdd: &mut Tdd, opts: MinimizeOptions<'_>) -> Result<(), ApplyError> {
    match opts.passes {
        MinimizePasses::ContractOnly => return contract_only(tdd),
        MinimizePasses::PruneOnly => {
            // Prune is packed-aware (see `pairs_remap_indexed`), so the unpack
            // is skipped entirely here: levels stay packed across the call.
            // Prune removes nodes, which can create twins in a shrunk level's
            // children; `prune_unreachable` seeds both contract worklists with
            // those shrunk levels so a later contraction pass covers them in
            // O(|dirty|).
            instrumented_prune(tdd)?;
            // Pairs killed by the prune may have orphaned marginal count slots;
            // see the slot-prune note below.
            crate::reduce::slot_prune::prune_marg_slots(tdd);
            return Ok(());
        }
        MinimizePasses::Full => {}
    }

    // Prune now operates packed-aware (see `pairs_remap_indexed`), so we
    // skip the pre-prune unpack and run prune directly against the Phase F
    // packed buffers. The unpack moves down to just before contract, which
    // still requires slice access for its push/extend/pop sites.
    // I1 guard: snapshot marginal flags before the structural passes so we can
    // pinpoint a pass that un-marginalizes a node (see `assert_no_demarginalization`).
    #[cfg(debug_assertions)]
    let i1_snap = snapshot_marginal_flags(tdd);

    instrumented_prune(tdd)?;
    #[cfg(debug_assertions)]
    assert_no_demarginalization(tdd, &i1_snap, "prune");

    // `prune_unreachable` has already seeded both contract worklists with the
    // levels it shrank (prune-created twins live in a shrunk level's children).
    // Together with the incoming dirty levels (e.g. a clause spine), that is the
    // complete set needing contraction — no all-levels reseed required.
    // Contract is now packed-safe end-to-end: `find_twin_groups` reads via
    // `pairs_iter_of`, and `contract_twins` lazy-unpacks parent and t1 only
    // when a productive merge fires. The level-pool-dropped wide levels stay
    // packed through contract, eliminating their 8 N-byte unpack write.
    //
    // For narrow levels whose pool-retained `pairs` Vec is still resident,
    // the in-place fast path is virtually free (a reinterpret-cast, no
    // fresh alloc). Eager-unpacking those wins back the per-access slice-
    // read speed in `find_twin_groups` without paying any unpack cost.
    // Always-run canonicalization tier (twin + leaf-twin contraction). Shared
    // with the segment-search gate via `contract_twins_and_leaves`. The two
    // debug demarginalization asserts collapse to one at the tier boundary —
    // the invariant is still checked after the full tier.
    contract_twins_and_leaves(tdd)?;
    #[cfg(debug_assertions)]
    assert_no_demarginalization(tdd, &i1_snap, "contract+leaf");

    // Slot-prune: the marginal-store counterpart of node-prune. Node-prune
    // above deliberately keeps marginal stores at full length (its remap is
    // identity there — refs could be minted against the old length before the
    // passes finish), so pairs it killed may have orphaned count slots. Now
    // that prune AND contract are done, refs are stable: compact each boundary
    // store to its parent-referenced set and clear dead deep stores.
    //
    // Value-merge loop: prune's value-dedup of equal-valued referenced slots
    // can MINT new content-equal twins at boundary-parent
    // levels after contract already ran. Example: parent nodes p = (X, c1) and
    // q = (X, c2) with c1 ≠ c2 as slot indices but equal stored values become
    // raw-identical after prune merges c1→c onto c2→c. Contract uses parent-
    // context signatures (set of (parent_node, sibling) pairs) to detect
    // twins; since p and q may have different parent contexts (different
    // siblings), context-based contract CANNOT detect them.
    //
    // Fix: after slot-prune, scan ALL explicit levels for content twins
    // (`merge_content_equal_nodes`). The scan is
    // unconditional: twins are minted not only by slot value-merges but also
    // directly by inline refs — p-fusion and the tagger emit small counts
    // inline without ever touching a slot, so two boundary parents can become
    // raw-identical (e.g. both {(X, Inline(1))}) with `values_merged == 0` —
    // and by the merge's OWN ref rewrites, which can make two nodes at a PLAIN
    // level identical.
    // Do NOT gate the scan on slot-prune's `values_merged`, and do not restrict
    // it to the value-merged levels: such a gate is blind to the inline-born
    // twins and lets content-twin violations reach minimize exit.
    // The expensive prune+contract round still runs only when the scan finds
    // a twin — actual twin minting is rare; otherwise the loop exits with all
    // four invariants intact:
    //   Twin canonicality: the scan is exactly `check_twin_canonicality`'s
    //       predicate (content equality at explicit levels) — zero dups found
    //       means no twins.
    //   Fusion saturation: a value merge cannot create a fusion redex — same-X
    //       pairs are fused before slot-prune ever runs, so no node holds two
    //       pairs whose refs could collapse onto the same slot.
    //   Slot-count uniqueness and inline discipline: slot-prune just ran.
    // The cheap scan-only pass is what makes the unconditional check
    // affordable: value merges are common, twin minting is not, so gating the
    // ROUND on `values_merged` alone is a large measured regression on
    // fusion-heavy CNFs.
    // Loop terminates: each productive iteration strictly reduces the
    // referenced node count, which is finite.

    // Content-twin-scan eligibility, the weighted/inline-weighted handling and the
    // galloping-probe policy are all documented on `c2_gated`.
    if !opts.skip_content_twins {
        content_twins::c2_gated(tdd, opts.content_twin_probe)?;
    }

    // Release Vec-doubling overshoot left behind when contract rebuilt the
    // pair arena. `shrink_arrays` is gated by capacity > 4*len, so this is a
    // no-op on levels without slack — only the rebuilt ones pay any cost.
    for level in &mut tdd.levels {
        level.shrink_arrays();
    }

    Ok(())
}

/// The always-run canonicalization tier: twin contraction + leaf-twin
/// contraction (with a re-contract if the leaf pass fired). Both passes are
/// dirty-scoped with an O(1) empty early-return (`contract_all_twins_topdown`,
/// `contract_leaf_twins`), so on a clean diagram this is a provable no-op —
/// safe to run after *every* op. Single source of truth for the twin+leaf
/// sequence: `try_minimize` (full) and the segment-search gate's `ContractOnly`
/// tier both call it. ([`MinimizePasses::ContractOnly`] stays twin-ONLY because
/// the bottom-up contract-only branch is byte-identity-pinned to that variant.)
fn contract_twins_and_leaves(tdd: &mut Tdd) -> Result<(), ApplyError> {
    contract_only(tdd)?;
    // Inner-node twin contraction can't reach leaf labels (Pos/Neg/One are
    // implicit, not stored nodes), so a single leaf-twin pass is needed to
    // reach canonical form. The rewrite may create new inner-node twins, so
    // contract again afterwards.
    let fired = contract_leaf_twins(tdd);
    if fired {
        contract_only(tdd)?;
    }
    Ok(())
}


/// Minimize after a vtree rotation. Skips prune AND leaf-twin contraction —
/// both are provable no-ops post-rotation under rotation locality. The
/// entire minimize collapses to a single locality-asserting inner-node
/// contract pass at `w_idx`.
///
/// Correct precondition: the input was canonical before the rotation, and the
/// rotation only restructured `v_idx` and `w_idx` levels (preserving the set
/// of child references). Three claims, all following from Rotation Locality:
///
/// 1. **Prune is a no-op.** Every node referenced before is still referenced
///    after, just regrouped — the rotation only re-axes the (v, w)
///    neighborhood, it doesn't drop any subfunction.
///
/// 2. **Inner-node twin contraction is single-level.** Under canonicity, the
///    only level that can have fresh twins after `restructure_after_*_rotation`
///    is the newly-introduced inner-node level at `w_idx`. The outer level at
///    `v_idx` inherits canonicity from the pre-rotation `v_idx` level by
///    parent-context bijection (same node count, same parent contexts at the
///    unchanged grandparent). Every level outside `{v_idx, w_idx}` is
///    bit-identical pre/post. We use `contract_only_at(w_idx)` which asserts
///    (debug builds) that no productive merge fires at any other level — a
///    runtime check on the rotation-locality tightening.
///
/// 3. **Leaf-twin contraction is rotation-invariant.** Each leaf's eligibility
///    for the `(Pos_x, S) + (Neg_x, S) → (One_x, S)` rewrite (∀upper context,
///    f independent of leaf-var) is a function-level property; rotation
///    preserves the function. The pre-rotation TDD was canonical, so each
///    leaf is already in the correct mode; the structural restructure
///    inherits the leaf labels verbatim. So `contract_leaf_twins` either
///    finds no literals (One-mode leaf) or finds literals but bails on the
///    level-wide check (literal-mode leaf). Either way it's a guaranteed
///    no-op, so we skip it entirely.
///
/// Does NOT reseed the contract worklist on all levels: `with_levels` already
/// seeds every internal level and contraction re-checks conservatively. Rotation sites push their changed
/// levels onto `dirty_contract` directly.
// Live in every build: called after each accepted rotation by the generic joint
// probe bodies (`joint_try_rotate_generic` / `joint_try_rotate_memo_generic`) in
// `search.rs` and by `rotate.rs`'s try/apply protocols. (search.rs imports it
// unconditionally.)
pub(crate) fn minimize_after_rotation(tdd: &mut Tdd, #[cfg_attr(not(debug_assertions), allow(unused_variables))] w_idx: VtreeIdx) {
    // Rotation locality, extended (inner-node contract is a no-op post-rotation):
    // After restructure_after_*_rotation, each new inner-level node at w_idx
    // has a unique parent context by construction: its fingerprint (the set of
    // (src_v_node, axis) cells it occurs in, computed in rotate.rs pass 2) is
    // its parent context in the outer level, and nodes are assigned one-per-
    // distinct-fingerprint (pass 3). Distinct fingerprints → distinct parent
    // contexts → no twins at w_idx → find_twin_groups always returns false →
    // contract_only_at is a guaranteed no-op.
    //
    // In debug: run it and assert the w_idx width is unchanged (no merges).
    // In release: just clear the dirty lists and skip.
    //
    // EXCEPTION — marginal context: all three rotation-locality claims above assume a CANONICAL
    // pre-rotation diagram. When any level is marginal, the marginal-context full
    // expansion (rotate.rs `diagram_has_marginal`) deliberately keeps the child
    // multiset *without* dedup, so the post-rotation w_idx level genuinely has
    // twins. Contracting them would (a) trip the no-op asserts, (b) merge
    // nodes and so shrink levels *outside* `{v_idx, w_idx}`, breaking both the
    // rotation-locality size predictor (search.rs `size_after_rotation`) and the
    // reject-path partial restore (which only snapshots v/w). Count-safety here
    // comes from the preserved multiset, not from canonicalization — exactly what
    // release relies on. So in marginal context debug must mirror release: drain
    // the dirty lists and skip the contract entirely.
    #[cfg(debug_assertions)]
    if !tdd.levels.iter().any(|l| l.is_marginal()) {
        let width_before = tdd.levels[w_idx.idx()].width();
        contract_only_at(tdd, w_idx)
            .expect("rotation-locality check: an allocation was refused");
        debug_assert_eq!(
            tdd.levels[w_idx.idx()].width(),
            width_before,
            "rotation locality: contract post-rotation fired at w_idx but fingerprint \
             uniqueness guarantees no twins",
        );
        // Leaf-twin-contraction-invariance runtime check: contract_leaf_twins
        // is provably a no-op post-rotation on a canonical diagram. Run it once
        // and assert nothing fired — guards against future code that violates the
        // invariant.
        let fired = contract_leaf_twins(tdd);
        debug_assert!(
            !fired,
            "contract_leaf_twins fired post-rotation but is provably a no-op",
        );
    }
    tdd.scratch.dirty_contract.clear();
    tdd.scratch.dirty_leaf_contract.clear();
}

// ── Internal helpers ─────────────────────────────────────────────────────


/// Run twin contraction unconditionally. `contract_all_twins` already
/// early-exits in O(1) when the dirty-list is empty, so an earlier guard that
/// scanned every level for `max_width ≤ 1` was pure redundancy — an
/// `O(num_vtree_nodes)` pass on every one of the rotation-search loop's many
/// thousand minimize calls. We dropped the guard and always call through.
fn contract_only(tdd: &mut Tdd) -> Result<(), ApplyError> {
    contract_all_twins(tdd)?;
    Ok(())
}

/// Locality-asserting contract pass: equivalent to `contract_only` plus a
/// debug-only assertion that no productive twin merge fires at any level
/// except `expected_only`. Used immediately after `restructure_after_*_rotation`
/// to verify the rotation-locality tightening — only the newly-introduced `w_idx` level can
/// have fresh twins.
#[cfg(debug_assertions)]
fn contract_only_at(tdd: &mut Tdd, expected_only: VtreeIdx) -> Result<(), ApplyError> {
    contract_all_twins_with_locality(tdd, expected_only)?;
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
