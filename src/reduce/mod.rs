//! Canonical form: pruning, twin contraction, pair fusion, slot pruning.
//!
//! A conjunction leaves a diagram that denotes the right function but is not
//! the smallest representation of it; the passes here bring it back to the
//! canonical one. Producing the diagram is [`crate::apply`]; summing levels out
//! is [`crate::marginal`], whose epilogue calls the last two passes here.
//!
//! Entry points: [`minimize`] is the infallible form, [`try_minimize`] the one
//! that hands a refused reservation back, and [`MinimizeOptions`] selects which
//! passes run.
//!
//! The passes, in the order a full reduction runs them:
//!
//! 1. **Prune** (`prune.rs`): remove nodes not reachable from the output, by a
//!    top-down reachability mark and a bottom-up compaction with a monotone
//!    remap.
//! 2. **Twin contraction** (`contract/`): merge nodes with identical parent
//!    context — the same set of (parent node, sibling) pairs. Twins compute
//!    functions whose disjunction replaces them both without changing the
//!    output.
//! 3. **Pair fusion and slot pruning** (`contract/pair_fusion/`,
//!    `slot_prune.rs`): what a freshly marginalized level needs — fusing pairs
//!    that share a structural-side child, and dropping value slots nothing
//!    references.
//!
//! **Prune to contract.** The two phases are decoupled except through the
//! `Tdd` dirty-contract worklists: prune (and the content-twin merge) call
//! `Tdd::mark_contract_dirty`, seeding the `dirty_contract`/`dirty_leaf_contract`
//! worklists that the contract pass then drains. This shared
//! state is the only coupling — neither phase reaches into the other's internals.
//!
//! There is no separate node-deduplication phase: after a conjunction,
//! canonical leaf ordering leaves no duplicate nodes by induction, and after
//! prune the monotone remap preserves node distinctness.

mod prune;
pub(crate) mod scratch;
pub(crate) mod contract;
pub(crate) mod slot_prune; // post-tagger marginal-slot compaction
mod content_twins;

#[cfg(test)]
pub(crate) use content_twins::canonicalize_content_twins;


// ── Minimize options ─────────────────────────────────────────────────────────

/// Which reduction passes [`try_minimize`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MinimizeScope {
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
/// (`next_scan_at_nodes` starts at 0) and the next probe is scheduled at 4x the pre-scan
/// size — unless the scan landed back under the cap, which resets `next_scan_at_nodes` to
/// 0 so the next above-cap call scans again.
///
/// A caller that minimizes a *fresh* diagram each step (a bottom-up compile
/// accumulator, say) must keep one of these across the steps and hand it to
/// [`MinimizeOptions::content_twin_probe`]; state carried on the diagram itself
/// would reset to "always scan" every step. Passing none is equivalent to
/// passing a fresh probe: the scan runs and the updated schedule is discarded.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ContentTwinProbe {
    /// Node count at which a skipped (above-cap) scan is re-attempted.
    /// 0 = scan on the next above-cap call.
    pub next_scan_at_nodes: u64,
}

/// What [`try_minimize`] should do.
///
/// `MinimizeOptions::default()` is the full canonical reduction with no probe
/// state carried across calls.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct MinimizeOptions<'a> {
    /// Which passes to run.
    pub passes: MinimizeScope,
    /// Skip the content-twin canonicalization pass. It is size- and
    /// canonicity-only — never count-affecting — so skipping it is sound, and
    /// worth it on a diagram that is about to be discarded or split. Ignored
    /// unless `passes` is [`MinimizeScope::Full`].
    pub skip_content_twins: bool,
    /// Probe schedule for the content-twin scan, carried across calls by the
    /// caller. See [`ContentTwinProbe`].
    pub content_twin_probe: Option<&'a mut ContentTwinProbe>,
}

use crate::engine::Engine;
use self::contract::contract_leaf::contract_leaf_twins;
use self::contract::contract_all_twins;
#[cfg(debug_assertions)]
use self::contract::contract_all_twins_with_locality;
use self::prune::prune_unreachable;
use crate::limits::ApplyError;
use crate::vtree::VtreeIdx;
use crate::diagram::Tdd;

/// Snapshot per-level `is_marginal` flags so a later
/// `assert_no_demarginalization` can detect a violation of invariant 5 (marginality is
/// permanent — see `check::marginal`) and name the offending pass.
#[cfg(debug_assertions)]
fn snapshot_marginal_flags(tdd: &Tdd) -> Vec<bool> {
    tdd.levels.iter().map(|l| l.is_marginal()).collect()
}

/// Assert no level present in `before` (as marginal) has become structural.
/// Panics naming the level and the `pass` that violated invariant 5.
#[cfg(debug_assertions)]
fn assert_no_demarginalization(tdd: &Tdd, before: &[bool], pass: &str) {
    for (i, &was_marginal) in before.iter().enumerate() {
        if was_marginal && !tdd.levels[i].is_marginal() {
            panic!(
                "invariant 5 violated: vtree level {i} was marginal before `{pass}` \
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
/// diagram is untouched and well-formed.
fn instrumented_prune(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    prune_unreachable(eng, tdd)
}

// ── Public minimize variants ─────────────────────────────────────────────

/// Minimize a diagram to its canonical form.
///
/// Two phases:
/// 1. **Prune**: remove nodes not reachable from the output
/// 2. **Twin contraction**: merge nodes with identical parent context
///
/// After minimization no two nodes at one level compute the same function,
/// and every node is reachable from the output.
///
/// **Allocation failure**: this entry point is infallible, for callers that do
/// not want to thread a `Result` through their plumbing. It panics when an
/// allocation is refused. A caller that must survive a refusal — by splitting
/// the diagram, or by giving the reduction more room — calls [`try_minimize`]
/// and handles [`ApplyError::OverBudget`].
///
/// Runs on limits of its own, with nothing armed, the same way the infallible
/// conjunction entries do: this entry has nowhere to report a cut to, so a stop
/// poll firing inside it would turn an expiry into a panic.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::reduce::minimize;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(4));
/// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let before = (f.size(), f.model_count());
///
/// minimize(&mut f);
/// assert_eq!(f.model_count(), before.1);   // the function is unchanged
/// assert!(f.size() <= before.0);           // the representation is canonical
/// ```
pub fn minimize(f: &mut Tdd) {
    let eng = Engine::new();
    try_minimize(&eng, f, MinimizeOptions::default())
        .expect("minimize: an allocation was refused; use try_minimize to handle it");
}

/// Fallible version of `minimize`: returns `Err(OverBudget)` if any internal
/// allocation is refused, by the OS allocator or by the engine's memory
/// budget. A caller in a compile loop uses this so its own recovery can
/// engage on a refusal instead of the process dying.
///
/// ## Error contract
///
/// - `Err(Deadline)` ⇒ the diagram is left **well-formed** (a clean early exit
///   at a pass boundary); the caller may keep and count it.
/// - `Err(OverBudget)` ⇒ well-formed. Every pass reserves its arena growth
///   before it mutates anything, so a refusal unwinds from a pass boundary
///   with the diagram exactly as it was.
///
/// So on `Err`, in either case: the diagram is sound and the caller may keep
/// and count it.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a budget-gated reduction step is
/// refused. On `Err` the diagram is untouched at a pass boundary (see above).
///
/// ```
/// use std::sync::Arc;
/// use tididi::{ApplyError, Engine, Tdd};
/// use tididi::limits::LimitSet;
/// use tididi::reduce::{try_minimize, MinimizeOptions};
/// use tididi::vtree::Vtree;
///
/// let engine = Engine::new();
/// let vtree = Arc::new(Vtree::balanced(20_000));
/// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let before = f.model_count();
///
/// // A byte budget of zero refuses the first budget-gated pass.
/// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
/// match try_minimize(&engine, &mut f, MinimizeOptions::default()) {
///     Ok(()) => {}
///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
/// }
/// // Either way the diagram is well-formed and still counts the same.
/// assert_eq!(f.model_count(), before);
/// ```
pub fn try_minimize(eng: &Engine, f: &mut Tdd, opts: MinimizeOptions<'_>) -> Result<(), ApplyError> {
    match opts.passes {
        MinimizeScope::ContractOnly => return contract_only(eng, f),
        MinimizeScope::PruneOnly => {
            // Prune is packed-aware (see `pairs_remap_indexed`), so the unpack
            // is skipped entirely here: levels stay packed across the call.
            // Prune removes nodes, which can create twins in a shrunk level's
            // children; `prune_unreachable` seeds both contract worklists with
            // those shrunk levels so a later contraction pass covers them in
            // O(|dirty|).
            instrumented_prune(eng, f)?;
            // Pairs killed by the prune may have orphaned marginal count slots;
            // see the slot-prune note below.
            crate::reduce::slot_prune::prune_value_slots(eng, f);
            return Ok(());
        }
        MinimizeScope::Full => {}
    }

    // Prune now operates packed-aware (see `pairs_remap_indexed`), so we
    // skip the pre-prune unpack and run prune directly against the Phase F
    // packed buffers. The unpack moves down to just before contract, which
    // still requires slice access for its push/extend/pop sites.
    // invariant 5 guard: snapshot marginal flags before the structural passes so we can
    // pinpoint a pass that un-marginalizes a node (see `assert_no_demarginalization`).
    #[cfg(debug_assertions)]
    let i1_snap = snapshot_marginal_flags(f);

    instrumented_prune(eng, f)?;
    #[cfg(debug_assertions)]
    assert_no_demarginalization(f, &i1_snap, "prune");

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
    contract_twins_and_leaves(eng, f)?;
    #[cfg(debug_assertions)]
    assert_no_demarginalization(f, &i1_snap, "contract+leaf");

    // Content-twin-scan eligibility, the weighted/inline-weighted handling and
    // the galloping-probe policy are all documented on `right_gated`, which
    // also drives the slot-prune sweep the structural passes above leave due.
    if !opts.skip_content_twins {
        content_twins::right_gated(eng, f, opts.content_twin_probe)?;
    }

    // Release Vec-doubling overshoot left behind when contract rebuilt the
    // pair arena. `shrink_arrays` is gated by capacity > 4*len, so this is a
    // no-op on levels without slack — only the rebuilt ones pay any cost.
    for level in &mut f.levels {
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
/// tier both call it. ([`MinimizeScope::ContractOnly`] stays twin-only because
/// the bottom-up contract-only branch is byte-identity-pinned to that variant.)
fn contract_twins_and_leaves(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    contract_only(eng, tdd)?;
    // Inner-node twin contraction can't reach leaf labels (Pos/Neg/One are
    // implicit, not stored nodes), so a single leaf-twin pass is needed to
    // reach canonical form. The rewrite may create new inner-node twins, so
    // contract again afterwards.
    if contract_leaf_twins(eng, tdd)? {
        contract_only(eng, tdd)?;
    }
    Ok(())
}


/// Minimize after a vtree rotation. Skips prune, and skips leaf-twin
/// contraction too: both are provable no-ops post-rotation under rotation
/// locality. The
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
///    only level that can have fresh twins after `relevel_after_{left,right}_rotation`
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
///    preserves the function. The pre-rotation diagram was canonical, so each
///    leaf is already in the correct mode; the structural restructure
///    inherits the leaf labels verbatim. So `contract_leaf_twins` either
///    finds no literals (One-mode leaf) or finds literals but bails on the
///    level-wide check (literal-mode leaf). Either way it's a guaranteed
///    no-op, so we skip it entirely.
///
/// Does not reseed the contract worklist on all levels: `from_levels_unchecked` already
/// seeds every internal level and contraction re-checks conservatively. Rotation sites push their changed
/// levels onto `dirty_contract` directly.
// Live in every build: called after each accepted rotation by the generic joint
// probe bodies (`joint_try_rotate_generic` / `joint_try_rotate_memo_generic`) in
// `search.rs` and by `rotate.rs`'s try/apply protocols. (search.rs imports it
// unconditionally.)
pub(crate) fn minimize_after_rotation(
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] eng: &Engine,
    tdd: &mut Tdd,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] w_idx: VtreeIdx,
) {
    // Rotation locality, extended (inner-node contract is a no-op post-rotation):
    // After relevel_after_{left,right}_rotation, each new inner-level node at w_idx
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
    // Marginal context is the exception: all three rotation-locality claims above
    // assume a canonical pre-rotation diagram. When any level is marginal, the marginal-context full
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
        contract_only_at(eng, tdd, w_idx)
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
        let fired = contract_leaf_twins(eng, tdd)
            .expect("rotation-locality check: an allocation was refused");
        debug_assert!(
            !fired,
            "contract_leaf_twins fired post-rotation but is provably a no-op",
        );
    }
    tdd.clear_worklists();
}

// ── Internal helpers ─────────────────────────────────────────────────────


/// Run twin contraction unconditionally. `contract_all_twins` early-exits in
/// O(1) when the dirty-list is empty, so guarding the call on a scan for
/// `max_width ≤ 1` would only add an `O(num_vtree_nodes)` pass to every
/// minimize call the rotation search makes.
fn contract_only(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    contract_all_twins(eng, tdd)?;
    Ok(())
}

/// Locality-asserting contract pass: equivalent to `contract_only` plus a
/// debug-only assertion that no productive twin merge fires at any level
/// except `expected_only`. Used immediately after `relevel_after_{left,right}_rotation`
/// to verify the rotation-locality tightening — only the newly-introduced `w_idx` level can
/// have fresh twins.
#[cfg(debug_assertions)]
fn contract_only_at(eng: &Engine, tdd: &mut Tdd, expected_only: VtreeIdx) -> Result<(), ApplyError> {
    contract_all_twins_with_locality(eng, tdd, expected_only)?;
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
