//! Spine-bounded ("restricted") conjunction — the O(spine) batch merge used by
//! the indicator-compile def loop.
//!
//! # Why
//!
//! `merge_batch_into_acc` folds a small batch diagram (a handful of definition
//! clauses) into a huge accumulator. The generic apply
//! (`apply_and_fallible_inner`) walks EVERY internal vtree level on every such
//! merge — snapshotting widths, seeding leaf identity, laying out grids, and
//! visiting each level only to take an identity fast path. On a vtree with
//! hundreds of thousands of levels that fixed cost dominates the merge even
//! though the batch touches a few thousand levels. This module runs the SAME
//! apply over a restricted level set instead.
//!
//! # The restricted set `R`
//!
//! `R` is not just the batch's spine. It is exactly the set of internal levels
//! the generic apply does NOT dispatch to `try_level_fast_paths`:
//!
//! * **`S`** — the batch's spine (the ancestor-closed union of the root-paths
//!   of its clauses' variable leaves), as reported by `walk_mark_spine` in
//!   `build_one_batch`. Off `S` the batch is constant-true with width 1, so the
//!   generic apply takes FP1 there and carries the accumulator's level through
//!   by reference. Every `S` internal node has an on-spine child (it is an
//!   ancestor of a clause leaf), so FP1's `c2_identity[left] &&
//!   c2_identity[right]` guard never holds on `S` — the generic apply rebuilds
//!   all of it.
//!
//! * **`AncClosure(P)`** — `P` is the set of *structural* levels with a
//!   *marginal* child (the parents of the maximal marginal subtrees the
//!   marginalize-behind-the-frontier schedule has already summed out). FP1's
//!   last guard (`!(!c1.is_marginal(t) && (out_left_marg || out_right_marg))`)
//!   and FP2's mirror of it both decline there, so `P` is rebuilt — and a
//!   rebuilt level sets NEITHER identity flag, so every ancestor of a `P` level
//!   fails the same guard and is rebuilt too, all the way to the root.
//!
//! Both pieces are ancestor-closed, so `R` is ancestor-closed and its
//! complement is descendant-closed — which is what makes the contraction seed
//! at the end exact rather than merely sound (same argument as
//! `conjoin_clause::try_apply_and_clause`; see the note at its `dirty_contract`
//! seed).
//!
//! # What the restricted apply skips
//!
//! Relative to the generic path, restricted mode:
//! * iterates `R` instead of `vtree.internal_bottomup()`;
//! * computes widths / grid layout / live counts only over `R ∪ children(R)`;
//! * skips `init_leaf_identity` on both operands (see `c2_identity` below);
//! * skips the leaf-marginalization seeding sweep (the batch has no marginal
//!   levels and the accumulator's marginal leaves are already in place — the
//!   output array is merged back into the accumulator's, so they survive);
//! * skips `try_level_fast_paths` (no level in `R` takes one — see above) and
//!   the `drop_dead_operand_level` calls (they would free accumulator levels
//!   that ride through untouched);
//! * restricts the end-of-apply `tag_all_marg_side_slots` sweep to `R` (only a
//!   structural level with a marginal child does any work there, and off `R`
//!   neither the level nor its children changed);
//! * merges the rebuilt levels back into the accumulator's own level array
//!   instead of returning a fresh full-length one.
//!
//! `c1_identity` is passed all-false and `c2_identity` as `!on_spine`. Both are
//! exactly what the generic path holds at every index restricted mode reads,
//! with ONE deliberate exception: if the batch happens to be constant-true at
//! an `S` level (a tautological fold), or the accumulator constant-true at an
//! `R` level, the generic path takes FP1/FP2 there and restricted mode rebuilds
//! instead. Both rebuilds reproduce the carried level (a `k×1` / `1×k` product
//! against a width-1 identity operand emits the same nodes in the same order).
//! `spine_bounded_merge_matches_generic_apply` (`apply_tests.rs`) keeps that
//! claim honest: it merges the same batches both ways and asserts bit-identity.
//!
//! # Declines
//!
//! The restricted path is an optimization, never a semantic change: anything it
//! is not proven exact for falls back to the generic merge, which is still THE
//! apply for every other caller.

use std::cell::Cell;
use std::sync::Arc;

use crate::tdd::types::{self, Tdd};
use crate::tdd::utils::{pool_put, pool_take};
use crate::vtree::VtreeIdx;

use super::ApplyError;

thread_local! {
    /// `on_spine[t]` — the batch's spine union. All-false between calls.
    static SCRATCH_SPINE_FLAGS: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// `in_rebuild[t]` — membership in `R`. All-false between calls.
    static SCRATCH_REBUILD_FLAGS: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// `R`, bottom-up (children before parents).
    static SCRATCH_REBUILD: Cell<Vec<VtreeIdx>> = const { Cell::new(Vec::new()) };
    /// `R ∪ children(R)` — every index whose width / grid / live count the
    /// restricted apply reads or writes.
    static SCRATCH_TOUCHED: Cell<Vec<VtreeIdx>> = const { Cell::new(Vec::new()) };
    /// The leaves in `touched` (the restricted `apply_leaf_levels` domain).
    static SCRATCH_LEAF_CHILDREN: Cell<Vec<VtreeIdx>> = const { Cell::new(Vec::new()) };
}

/// The restriction the apply core runs under. Borrowed from a [`RestrictPlan`].
pub(super) struct Restrict<'a> {
    /// Internal levels to rebuild, children before parents (`internal_topo`
    /// order, so the per-level results are the generic path's verbatim).
    pub(super) rebuild: &'a [VtreeIdx],
    /// `in_rebuild[t.idx()]`.
    pub(super) in_rebuild: &'a [bool],
    /// `on_spine[t.idx()]` — `c2` (the batch) is constant-true exactly off this
    /// set, which is what makes `c2_identity == !on_spine` correct.
    pub(super) on_spine: &'a [bool],
    /// `rebuild ∪ children(rebuild)`.
    pub(super) touched: &'a [VtreeIdx],
    /// The leaves of `touched`.
    pub(super) leaf_children: &'a [VtreeIdx],
    /// The value the generic apply's global pre-scan would compute for this
    /// operand pair, derived in `O(|spine|)` by `build_plan` from the caller's
    /// cached widest-internal-level width. Matched exactly (rather than forced
    /// either way) so the sparse / sparse-marg routes fire at the same levels
    /// they would generically.
    pub(super) might_use_sparse: bool,
}

/// Pooled backing storage for a [`Restrict`].
struct RestrictPlan {
    rebuild: Vec<VtreeIdx>,
    in_rebuild: Vec<bool>,
    on_spine: Vec<bool>,
    touched: Vec<VtreeIdx>,
    leaf_children: Vec<VtreeIdx>,
    might_use_sparse: bool,
}

impl RestrictPlan {
    fn as_restrict(&self) -> Restrict<'_> {
        Restrict {
            rebuild: &self.rebuild,
            in_rebuild: &self.in_rebuild,
            on_spine: &self.on_spine,
            touched: &self.touched,
            leaf_children: &self.leaf_children,
            might_use_sparse: self.might_use_sparse,
        }
    }

    /// Reset the two all-false-invariant flag arrays over exactly the entries
    /// this plan set, then hand every buffer back to its pool.
    fn recycle(mut self) {
        for &t in &self.touched {
            self.on_spine[t.idx()] = false;
            self.in_rebuild[t.idx()] = false;
        }
        // `on_spine` also covers spine LEAVES that are not children of a
        // rebuild level (a clause leaf whose parent chain is all in `R`, but
        // the leaf itself is only reachable as a child of a rebuilt level —
        // covered — or via a marginal sibling; be exhaustive rather than
        // clever and clear the whole spine too).
        for &t in &self.rebuild {
            self.on_spine[t.idx()] = false;
            self.in_rebuild[t.idx()] = false;
        }
        debug_assert!(
            self.on_spine.iter().all(|&b| !b) && self.in_rebuild.iter().all(|&b| !b),
            "RestrictPlan::recycle left a flag set — the pooled all-false invariant is broken"
        );
        pool_put(&SCRATCH_REBUILD, self.rebuild);
        pool_put(&SCRATCH_REBUILD_FLAGS, self.in_rebuild);
        pool_put(&SCRATCH_SPINE_FLAGS, self.on_spine);
        pool_put(&SCRATCH_TOUCHED, self.touched);
        pool_put(&SCRATCH_LEAF_CHILDREN, self.leaf_children);
    }
}

/// The two accumulator width maxima [`try_apply_and_batch_owned`] is handed and
/// gives back, taken over the levels it rebuilt.
///
/// Those are the only levels a restricted merge can have widened — every other
/// level rides through byte for byte — so a caller that keeps both quantities
/// as running maxima folds these two numbers in after each merge instead of
/// re-sweeping the whole level array. That is the point of the type: sweeping
/// a multi-million-node accumulator once per merge would cost more than the
/// merge.
///
/// The two maxima are different quantities and both are needed:
///
/// * `live` counts live nodes only, and is exactly what [`Tdd::max_width`]
///   reports for the merged diagram.
/// * `raw_internal` counts tombstoned slots too, and is taken over internal
///   levels only. It is the quantity that decides the apply's grid layout, so
///   it cannot be recovered from `live`.
///
/// `live_at` and `raw_at` name the level each maximum was read at. A caller
/// keeping a running maximum needs them: without the level, a later merge that
/// rebuilds that same level cannot tell whether the recorded maximum still
/// stands or has just been invalidated.
#[derive(Clone, Copy, Debug)]
pub struct RebuiltMax {
    /// Widest rebuilt level counting live nodes only.
    pub live: usize,
    /// The level `live` was read at.
    pub live_at: VtreeIdx,
    /// Widest rebuilt internal level counting tombstoned slots too.
    pub raw_internal: usize,
    /// The level `raw_internal` was read at.
    pub raw_at: VtreeIdx,
}

impl RebuiltMax {
    /// Read both maxima off `rebuild` in the merged diagram. `rebuild` is
    /// internal-only and non-empty for every call that reaches here (an empty
    /// spine declines), so the `at` fields always name a level that was walked.
    fn over(merged: &Tdd, rebuild: &[VtreeIdx]) -> Self {
        let mut m = RebuiltMax {
            live: 0,
            live_at: merged.output.vtree,
            raw_internal: 0,
            raw_at: merged.output.vtree,
        };
        for &t in rebuild {
            let l = &merged.levels[t.idx()];
            let lw = l.live_width();
            if lw > m.live {
                m.live = lw;
                m.live_at = t;
            }
            let w = l.width();
            if w > m.raw_internal {
                m.raw_internal = w;
                m.raw_at = t;
            }
        }
        m
    }
}

/// Outcome of [`try_apply_and_batch_owned`].
pub enum BatchMerge {
    /// The restricted merge ran. The diagram is `acc ∧ batch` — bit for bit what
    /// the generic conjunction would have produced — and the [`RebuiltMax`] is
    /// both width maxima re-read over the levels it rebuilt.
    Merged(Tdd, RebuiltMax),
    /// The restricted merge declined. Both operands come back untouched, in the
    /// order they were passed, for the caller to hand to
    /// [`try_apply_and_both_owned`](super::try_apply_and_both_owned). This is
    /// not an answer and never a failure.
    Declined(Tdd, Tdd),
}

/// Conjoin a small `batch` diagram into a large `acc`, visiting only the vtree
/// levels the batch can have changed.
///
/// [`try_apply_and_both_owned`](super::try_apply_and_both_owned) walks every
/// internal vtree level on every call. When one operand is small — a handful of
/// clauses folded together — and the other is a large accumulator, that fixed
/// per-level cost dominates a merge whose real work touches a small fraction of
/// the vtree. This entry point runs the same apply over a restricted level set
/// instead, and produces the bit-identical diagram, provided the caller's
/// `spine` argument meets the contract below.
///
/// # The `spine` contract
///
/// `spine` must name every vtree node at which `batch` is not constant true:
/// every internal node whose level in `batch` is wider than one, and every leaf
/// `batch` reaches other than through the constant-one child. It must be
/// **ancestor closed** — if a node is listed, so is every node on its path to
/// the root. Order does not matter and duplicates are harmless. Listing extra
/// nodes only costs time; listing too few is **unsound**, so a caller that
/// cannot certify the set must use the generic conjunction instead.
///
/// The set is a by-product of building the batch, which is why it is a
/// parameter rather than something derived here. A caller that folds `k`
/// clauses into `batch` obtains it by marking each clause variable's leaf and
/// walking to the root, stopping at the first node already marked: `O(|spine|)`
/// over the whole fold rather than `O(k × height)`, and already paid for. The
/// alternative — recovering it here — is a sweep of the entire vtree per merge,
/// which is the cost this entry point exists to avoid.
///
/// # The accumulator's arguments
///
/// `marg_parents` names the levels of `acc` that have a marginalized child, the
/// levels a preceding marginalization left behind. Only the caller knows when
/// the accumulator was last marginalized, so this too is passed in. It is a
/// seed set: naming a level that has itself since gone marginal is fine and is
/// re-filtered here, but as with `spine`, it must not be short.
///
/// `acc_max_width` and `acc_widest_internal` are `acc`'s [`Tdd::max_width`] and
/// the widest `width()` over its internal levels (tombstoned slots included).
/// They are the two whole-diagram quantities a restricted merge cannot re-read
/// cheaply. [`RebuiltMax`], returned alongside a successful merge, is how a
/// caller keeps both current across a run of merges without ever re-sweeping
/// the level array.
///
/// # Declining
///
/// Returns [`BatchMerge::Declined`], with both operands intact and in the order
/// they were passed, whenever the restricted path is not provably exact for
/// this call: an empty spine, a batch wider than the accumulator, operands that
/// do not share a vtree, a constant-false operand, a marginalized batch root,
/// or an armed cap or weight context, neither of which the restricted level
/// set models. A caller must treat a decline as "run the
/// generic conjunction on these operands"; it is not an answer, and it is never
/// a failure.
///
/// # Errors
///
/// Propagates [`ApplyError`] from the apply core. As with every other owned
/// apply entry point, an `Err` means both operands are spent — they must not be
/// reused, only rebuilt.
pub fn try_apply_and_batch_owned(
    acc: Tdd,
    batch: Tdd,
    spine: &[VtreeIdx],
    marg_parents: &[VtreeIdx],
    acc_max_width: usize,
    acc_widest_internal: usize,
) -> Result<BatchMerge, ApplyError> {
    if decline_reason(&acc, &batch, spine, acc_max_width).is_some() {
        return Ok(BatchMerge::Declined(acc, batch));
    }
    let plan = build_plan(&acc, &batch, spine, marg_parents, acc_widest_internal);

    let mut acc = acc;
    let mut batch = batch;
    let result = {
        let r = plan.as_restrict();
        let out = super::apply_and_fallible_restricted(&mut acc, &mut batch, &r);
        types::return_levels(std::mem::take(&mut acc.levels));
        types::return_levels2(std::mem::take(&mut batch.levels));
        out
    };
    // `RebuiltMax` has to be read before the plan's buffers go back to their
    // pools, and the plan has to be recycled on the error path too.
    let merged = match result {
        Ok(t) => {
            let m = RebuiltMax::over(&t, &plan.rebuild);
            (t, m)
        }
        Err(e) => {
            plan.recycle();
            return Err(e);
        }
    };
    plan.recycle();
    Ok(BatchMerge::Merged(merged.0, merged.1))
}

/// Cheap pre-checks, all `O(1)` or `O(|spine|)`, run before any buffer is taken.
/// `Some(reason)` means DECLINE; the reason names the cause for the reader (it
/// is not surfaced at runtime).
fn decline_reason(
    acc: &Tdd,
    batch: &Tdd,
    spine: &[VtreeIdx],
    acc_max_width: usize,
) -> Option<&'static str> {
    if spine.is_empty() {
        return Some("empty spine");
    }
    // The owned generic merge puts the NARROWER operand on `c2`
    // (`try_apply_and_both_owned_with_schedule`), and which operand is `c1`
    // decides the emitted node order. The restricted merge cannot swap — `c1`
    // must be the accumulator whose levels ride through — so decline rather
    // than emit a different (still correct, but not bit-identical) diagram.
    // O(|spine|): off the spine the batch is width-1 by certificate.
    let batch_width = spine.iter().map(|&t| batch.level(t).live_width()).max().unwrap_or(0);
    if batch_width > acc_max_width {
        return Some("batch wider than accumulator (generic would swap operands)");
    }
    if !Arc::ptr_eq(&acc.vtree, &batch.vtree) || acc.output.vtree != batch.output.vtree {
        return Some("operands do not share a vtree / output root");
    }
    // The apply core's own early-outs (ZERO operand, self-conjunction) are
    // cheaper than anything here; let the generic path take them.
    if acc.is_zero() || batch.is_zero() {
        return Some("ZERO operand");
    }
    // A restricted apply visits only `R`, so `out_nodes_so_far` covers only
    // `R` — an output-node cap would trip at a different point than generically.
    // Weighted marginals bring the leaf-canonicalization sweep, which is a
    // whole-diagram pass the restriction does not model.
    if super::budget::apply_output_node_cap().is_some()
        || crate::tdd::transform::unary::marginalize::weight_ctx_active()
    {
        return Some("a cap / weight context is armed");
    }
    // The certificate the caller is asserting: the batch constrains nothing off
    // its spine. Verified per-level under debug; spot-checked at the root here.
    if batch.levels[batch.output.vtree.idx()].is_marginal() {
        return Some("batch root is marginal");
    }
    None
}

/// Build `R` and the derived index sets.
///
/// `marg_parents` and `acc_widest` are the caller's cached stand-ins for the two
/// whole-level-array quantities this used to stream: the levels with a marginal
/// child, and the widest internal level. See [`try_apply_and_batch_owned`].
fn build_plan(
    acc: &Tdd,
    batch: &Tdd,
    spine: &[VtreeIdx],
    marg_parents: &[VtreeIdx],
    acc_widest: usize,
) -> RestrictPlan {
    let vtree = &acc.vtree;
    let n = vtree.num_nodes();

    let mut on_spine: Vec<bool> = pool_take(&SCRATCH_SPINE_FLAGS);
    let mut in_rebuild: Vec<bool> = pool_take(&SCRATCH_REBUILD_FLAGS);
    if on_spine.len() < n {
        on_spine.resize(n, false);
    }
    if in_rebuild.len() < n {
        in_rebuild.resize(n, false);
    }
    let mut rebuild: Vec<VtreeIdx> = pool_take(&SCRATCH_REBUILD);
    let mut touched: Vec<VtreeIdx> = pool_take(&SCRATCH_TOUCHED);
    let mut leaf_children: Vec<VtreeIdx> = pool_take(&SCRATCH_LEAF_CHILDREN);
    rebuild.clear();
    touched.clear();
    leaf_children.clear();

    for &t in spine {
        on_spine[t.idx()] = true;
        if !vtree.node(t).is_leaf() {
            in_rebuild[t.idx()] = true;
            rebuild.push(t);
        }
    }

    // `AncClosure(P)`: every structural level with a marginal child, plus all of
    // its ancestors. `P` is `marg_parents` filtered to the levels that are still
    // structural — a level inside a marginal subtree is covered by that
    // subtree's own boundary parent. This used to be a sweep over every level of
    // the accumulator; the caller now maintains the seed set at the one place
    // the accumulator's marginal levels change (see `try_apply_and_batch_owned`),
    // and the closure below is `O(|R|)` because it stops at the first level
    // already in `R`.
    for &p in marg_parents {
        if acc.levels[p.idx()].is_marginal() {
            continue; // interior of a marginal subtree — its parent handles it
        }
        let mut cur = p;
        loop {
            if in_rebuild[cur.idx()] {
                break;
            }
            in_rebuild[cur.idx()] = true;
            rebuild.push(cur);
            match vtree.node(cur).parent() {
                Some(q) => cur = q,
                None => break,
            }
        }
    }
    debug_assert!(
        {
            // The cached seed set must cover every structural level with a
            // marginal child; a MISSING one silently carries a level the generic
            // apply rebuilds, which is a wrong diagram, not a slower one.
            (0..n).all(|i| {
                !acc.levels[i].is_marginal()
                    || vtree.node(VtreeIdx(i as u32)).parent().is_none_or(|p| {
                        acc.levels[p.idx()].is_marginal() || in_rebuild[p.idx()]
                    })
            })
        },
        "spine-bounded merge: `marg_parents` missed a structural level with a \
         marginal child — the caller's cache is stale"
    );
    debug_assert_eq!(
        acc_widest,
        (0..n)
            .filter(|&i| !vtree.node(VtreeIdx(i as u32)).is_leaf())
            .map(|i| acc.levels[i].width())
            .max()
            .unwrap_or(0),
        "spine-bounded merge: cached widest internal level disagrees with the diagram"
    );

    // `internal_topo` is `topo` filtered to internal nodes, so sorting by
    // `topo_pos` reproduces `internal_bottomup()`'s relative order exactly —
    // the generic level order, restricted.
    rebuild.sort_unstable_by_key(|&t| vtree.topo_pos(t));

    for &t in &rebuild {
        touched.push(t);
        let (l, r) = vtree.children(t);
        for c in [l, r] {
            if !in_rebuild[c.idx()] {
                touched.push(c);
                if vtree.node(c).is_leaf() {
                    leaf_children.push(c);
                }
            }
        }
    }
    debug_assert!(
        rebuild.iter().all(|&t| !acc.levels[t.idx()].is_marginal()),
        "spine-bounded merge: a rebuilt level is marginal in the accumulator — \
         either the batch constrains a summed-out variable (marginalize-schedule \
         bug) or `AncClosure(P)` reached inside a marginal subtree"
    );
    // Internal levels only: a LEAF level is `LEAF_WIDTH` wide in every diagram
    // (the implicit Pos/Neg/One nodes), on the spine or not. What makes an
    // off-spine leaf identity is that nothing REFERENCES anything but `One`
    // there, which is the `c2_identity` claim, not a width claim.
    debug_assert!(
        touched.iter().all(|&t| {
            on_spine[t.idx()] || vtree.node(t).is_leaf() || batch.effective_width(t) == 1
        }),
        "spine-bounded merge: batch is not width-1 off its reported spine"
    );

    // `might_use_sparse`, EXACTLY as the generic pre-scan
    // (`∃ internal t: w1(t)·w2(t) > min_grid`) would compute it, in
    // `O(|spine|)` rather than a full width sweep:
    //
    // * If the widest internal accumulator level exceeds `min_grid`, that level
    //   alone answers `true` — its `w2` is at least 1 whether it is on the
    //   spine or not.
    // * Otherwise every OFF-spine internal level has `w1·w2 = w1·1 ≤ min_grid`
    //   and contributes nothing, so only the spine can tip the scan — and the
    //   spine is exactly what we are already allowed to walk.
    //
    // Matched rather than forced (either way) so the sparse / sparse-marg
    // routes fire at the same levels the unrestricted apply would fire them at.
    let min_grid = super::sparse::sparse_config().min_grid;
    let might_use_sparse = acc_widest > min_grid
        || spine.iter().any(|&t| {
            !vtree.node(t).is_leaf()
                && acc.levels[t.idx()].width().saturating_mul(batch.levels[t.idx()].width())
                    > min_grid
        });

    RestrictPlan { rebuild, in_rebuild, on_spine, touched, leaf_children, might_use_sparse }
}
