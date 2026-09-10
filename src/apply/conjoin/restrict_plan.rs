//! The decline pre-checks and the restricted level set they guard.
//!
//! Split out of `restrict` so the module that runs the merge holds only the
//! merge. `must_decline` is every cheap test that can refuse a batch before a
//! buffer is taken; `collect_touched` and `build_plan` compute the restricted
//! level set `R` the merge then walks.

use crate::engine::Engine;
use crate::apply::scoped_flags::ScopedFlags;
use super::RestrictPlan;
use std::sync::Arc;
use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

/// Cheap pre-checks, all `O(1)` or `O(|spine|)`, run before any buffer is taken.
/// True means the batch must take the generic merge instead.
pub(super) fn must_decline(
    eng: &Engine,
    acc: &Tdd,
    batch: &Tdd,
    spine: &[VtreeIdx],
    acc_max_width: usize,
) -> bool {
    let lim = eng.limits();
    if spine.is_empty() {
        return true;
    }
    // The owned generic merge puts the NARROWER operand on `g`
    // (`conjoin_owned`), and which operand is `f`
    // decides the emitted node order. The restricted merge cannot swap — `f`
    // must be the accumulator whose levels ride through — so decline rather
    // than emit a different (still correct, but not bit-identical) diagram.
    // O(|spine|): off the spine the batch is width-1 by certificate.
    let batch_width = spine.iter().map(|&t| batch.level(t).live_width()).max().unwrap_or(0);
    if batch_width > acc_max_width {
        return true;
    }
    // Operands must share a vtree and an output root.
    if !Arc::ptr_eq(&acc.vtree, &batch.vtree) || acc.output.vtree != batch.output.vtree {
        return true;
    }
    // The apply core's own early-outs (ZERO operand, self-conjunction) are
    // cheaper than anything here; let the generic path take them.
    if acc.is_zero() || batch.is_zero() {
        return true;
    }
    // A restricted apply visits only `R`, so `out_nodes_so_far` covers only
    // `R` — an output-node cap would trip at a different point than generically.
    // Weighted marginals bring the leaf-canonicalization sweep, which is a
    // whole-diagram pass the restriction does not model.
    if lim.output_node_cap().is_some() || acc.weights().is_some() {
        return true;
    }
    // The certificate the caller is asserting: the batch constrains nothing off
    // its spine. Verified per-level under debug; spot-checked at the root here.
    if batch.levels[batch.output.vtree.idx()].is_marginal() {
        return true;
    }
    false
}

/// Build `R` and the derived index sets.
///
/// `marginal_parents` and `acc_widest` are the caller's cached stand-ins for two
/// whole-level-array quantities — the levels with a marginal child, and the
/// widest internal level — so this need not sweep every level. See
/// [`conjoin_batch`].
/// Collect the levels the merge reads — every rebuilt level plus the children
/// it reaches into — and check that the plan matches what the merge assumes:
/// No rebuilt level is marginal in the accumulator, and off the spine the batch
/// is width-1 at every internal level.
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_touched(
    acc: &Tdd,
    batch: &Tdd,
    vtree: &crate::vtree::Vtree,
    rebuild: &[VtreeIdx],
    in_rebuild: &[bool],
    on_spine: &[bool],
    touched: &mut Vec<VtreeIdx>,
    leaf_children: &mut Vec<VtreeIdx>,
) {
    for &t in rebuild {
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
    // there, which is the `right_identity` claim, not a width claim.
    debug_assert!(
        touched.iter().all(|&t| {
            on_spine[t.idx()] || vtree.node(t).is_leaf() || batch.effective_width(t) == 1
        }),
        "spine-bounded merge: batch is not width-1 off its reported spine"
    );
}

pub(super) fn build_plan<'a>(
    eng: &'a Engine,
    acc: &Tdd,
    batch: &Tdd,
    spine: &[VtreeIdx],
    marginal_parents: &[VtreeIdx],
    acc_widest: usize,
) -> RestrictPlan<'a> {
    let vtree = &acc.vtree;
    let n = vtree.num_nodes();

    let pool = eng.restrict_pool();
    let mut on_spine = ScopedFlags::take(&pool.spine_flags, n);
    let mut in_rebuild = ScopedFlags::take(&pool.rebuild_flags, n);
    let mut rebuild: Vec<VtreeIdx> = pool.rebuild.take();
    let mut touched: Vec<VtreeIdx> = pool.touched.take();
    let mut leaf_children: Vec<VtreeIdx> = pool.leaf_children.take();
    rebuild.clear();
    touched.clear();
    leaf_children.clear();

    for &t in spine {
        on_spine.set(t);
        if !vtree.node(t).is_leaf() {
            in_rebuild.set(t);
            rebuild.push(t);
        }
    }

    // `AncClosure(P)`: every structural level with a marginal child, plus all of
    // its ancestors. `P` is `marginal_parents` filtered to the levels that are still
    // structural — a level inside a marginal subtree is covered by that
    // subtree's own boundary parent. The caller maintains the seed set at the
    // one place the accumulator's marginal levels change (see `conjoin_batch`),
    // so no sweep over every level is needed, and the closure below is
    // `O(|R|)`: it stops at the first level already in `R`.
    for &p in marginal_parents {
        if acc.levels[p.idx()].is_marginal() {
            continue; // interior of a marginal subtree — its parent handles it
        }
        let mut cur = p;
        loop {
            if in_rebuild[cur.idx()] {
                break;
            }
            in_rebuild.set(cur);
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
        "spine-bounded merge: `marginal_parents` missed a structural level with a \
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

    collect_touched(
        acc, batch, vtree, &rebuild, &in_rebuild, &on_spine,
        &mut touched, &mut leaf_children,
    );

    // `might_use_sparse`, exactly as the generic pre-scan
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
    // Matched rather than forced (either way) so the sparse / sparse-marginal
    // routes fire at the same levels the unrestricted apply would fire them at.
    let min_grid = crate::apply::conjoin::sparse::sparse_config().min_grid;
    let might_use_sparse = acc_widest > min_grid
        || spine.iter().any(|&t| {
            !vtree.node(t).is_leaf()
                && acc.levels[t.idx()].width().saturating_mul(batch.levels[t.idx()].width())
                    > min_grid
        });

    RestrictPlan { rebuild, in_rebuild, on_spine, touched, leaf_children, might_use_sparse, eng }
}
