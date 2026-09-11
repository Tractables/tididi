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
use self::prune::prune_unreachable;
use crate::limits::ApplyError;
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
        MinimizeScope::ContractOnly => return contract_all_twins(eng, f),
        MinimizeScope::PruneOnly => {
            // Prune is packed-aware (see `pairs_remap_indexed`), so the unpack
            // is skipped entirely here: levels stay packed across the call.
            // Prune removes nodes, which can create twins in a shrunk level's
            // children; `prune_unreachable` seeds both contract worklists with
            // those shrunk levels so a later contraction pass covers them in
            // O(|dirty|).
            prune_unreachable(eng, f)?;
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

    prune_unreachable(eng, f)?;
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
/// dirty-scoped with an O(1) empty early-return (`contract_all_twins`,
/// `contract_leaf_twins`), so on a clean diagram this is a provable no-op —
/// safe to run after *every* op. Single source of truth for the twin+leaf
/// sequence: `try_minimize` (full), the content-twin loop and the
/// segment-search gate's `ContractOnly` tier all call it.
/// ([`MinimizeScope::ContractOnly`] stays twin-only because the bottom-up
/// contract-only branch is byte-identity-pinned to that variant.)
pub(super) fn contract_twins_and_leaves(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    contract_all_twins(eng, tdd)?;
    // Inner-node twin contraction can't reach leaf labels (Pos/Neg/One are
    // implicit, not stored nodes), so a single leaf-twin pass is needed to
    // reach canonical form. The rewrite may create new inner-node twins, so
    // contract again afterwards.
    if contract_leaf_twins(eng, tdd)? {
        contract_all_twins(eng, tdd)?;
    }
    Ok(())
}


/// Finish a vtree rotation on a diagram that was canonical before it: no
/// reduction pass has anything to do, so only the worklists the rotation
/// seeded are drained.
///
/// Precondition: the input was canonical before the rotation, and the
/// rotation only restructured the `v_idx` and `w_idx` levels (preserving the
/// set of child references). Three claims, all following from rotation
/// locality:
///
/// 1. **Prune is a no-op.** Every node referenced before is still referenced
///    after, just regrouped — the rotation only re-axes the (v, w)
///    neighborhood, it doesn't drop any subfunction.
///
/// 2. **Inner-node twin contraction is a no-op.** Every level outside
///    `{v_idx, w_idx}` is bit-identical pre/post. The outer level at `v_idx`
///    inherits canonicity from the pre-rotation `v_idx` level by
///    parent-context bijection (same node count, same parent contexts at the
///    unchanged grandparent). Each node of the new inner level at `w_idx` is
///    minted one per distinct fingerprint, and its fingerprint is its parent
///    context in the outer level, so `w_idx` has no twins either.
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
/// `check::debug_assert_rotation_locality` runs both contractions and asserts
/// that neither fires; the rotation probe calls it before this in debug
/// builds.
///
/// Does not reseed the contract worklist on all levels: `from_levels_unchecked` already
/// seeds every internal level and contraction re-checks conservatively. Rotation sites push their changed
/// levels onto `dirty_contract` directly.
pub(crate) fn minimize_after_rotation(tdd: &mut Tdd) {
    tdd.clear_worklists();
}

#[cfg(test)]
mod tests;
