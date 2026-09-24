//! Duplicate-pair resolution by multiplicity fork-down ("scale" rewrite).
//!
//! Twin contraction above a marginal boundary can leave duplicate `(L, R)`
//! entries in a plain (no inlined side) node's pair list, and a plain level has
//! no count field for the multiplicity k. The duplicates are not count-wrong:
//! the count recurrences sum over a node's pairs, so k copies already
//! contribute `k·c(L)·c(R)`, and every plain-level pass is stated for
//! multisets. The rewrite here is a size optimization: the k copies become one
//! pair whose marginal-carrying side is scaled by k (an inline re-encode or one
//! fresh slot; at a weight-marginal leaf, whose column is pinned, only a value
//! the column already holds).
//!
//! ## Cost policy: only an O(1) absorber is taken
//!
//! A marginal child absorbs the factor by multiplying one count. A structural
//! child would have to be cloned with a marginal-side descendant scaled,
//! minting a copy of every node on the way down, so it is never taken: when
//! neither child of the plain level is marginal, the run stays as k multiset
//! terms and `resolve_duplicate_pairs_in_node` early-outs. Scaled counts are
//! fresh slots; slot-prune value-merge later shares them with equal-valued
//! slots.

use crate::diagram::{ChildPair, ChildSide, EncodedChildRef, NodeKind, Sides, Tdd};

use crate::Engine;
use super::scratch::DuplicateScratch;
use crate::limits::{Limits, OperationError};

mod scale;
use scale::{has_o1_absorber, scale_pair_one_side};
use crate::vtree::VtreeIdx;

/// `has_marginal_below[v]` — v's vtree subtree (including v itself) contains a
/// marginal level. Stable across one minimize pass: marginalization converts
/// levels between compile phases, never during contraction. Fills `below` in
/// place (buffer reused across contract runs via ContractScratch), charging
/// its growth to `lim`.
///
/// O(vtree nodes), so the caller fills it lazily — on the first merge a sweep
/// attempts, never on a sweep that finds no twins. See
/// `ContractScratch::has_marginal_below_valid`.
pub(super) fn compute_has_marginal_below_into(
    lim: &Limits,
    tdd: &Tdd,
    below: &mut Vec<bool>,
) -> Result<(), OperationError> {
    let n = tdd.vtree.num_nodes();
    below.clear();
    lim.try_resize(below, n, false)?;
    for i in 0..n.min(tdd.levels.len()) {
        if tdd.levels[i].is_marginal() {
            let mut v = Some(VtreeIdx(i as u32));
            while let Some(x) = v {
                if below[x.idx()] {
                    break; // ancestors above already marked by an earlier walk
                }
                below[x.idx()] = true;
                v = tdd.vtree.node(x).parent();
            }
        }
    }
    Ok(())
}

/// Resolve duplicate pairs in node `idx` at plain structural level `pv`:
/// each run of k > 1 equal `(L, R)` pairs whose level has an O(1) absorber is
/// replaced by a single pair with the marginal side scaled by k. Returns whether
/// the node changed. No-op (`Ok(false)`) when the pair list is duplicate-free,
/// and — by the cost policy — when neither child of `pv` is marginal, in which
/// case the duplicates stay as legal multiset terms.
///
/// `scratch` is caller-owned and reused across the survivors of one fork-down
/// pass (`merge::compact_and_fork_down`); every buffer is cleared here on
/// entry, so it carries capacity across nodes and nothing else, and a `?` bail
/// need not clean it up.
pub(super) fn resolve_duplicate_pairs_in_node(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    idx: usize,
    scratch: &mut DuplicateScratch,
) -> Result<bool, OperationError> {
    // `pv` is a plain (non-marginal) level — marginal levels are pair fusion's
    // domain. Its inline markers are not asserted clear: scaling a marginal child
    // ref can mint an inline marginal ref into `pv`'s pairs, which raises `pv`'s
    // `has_value_refs` marker (below). The caller resolves several survivors per
    // pass, so the second and later calls legitimately see the marker already up.
    debug_assert!(
        !tdd.levels[pv.idx()].is_marginal(),
        "resolve_duplicate_pairs_in_node: marginal levels are pair fusion's domain"
    );
    // With no marginal child there is no O(1) absorber and every run would be
    // kept, so skip the collect and hash-count.
    if !has_o1_absorber(tdd, pv) {
        return Ok(false);
    }
    let lim = eng.limits();
    scratch.clear();
    let DuplicateScratch { pairs, counts, out } = scratch;
    let node_pairs = tdd.levels[pv.idx()].pairs_of_idx(idx);
    if node_pairs.len() < 2 {
        return Ok(false);
    }
    lim.reserve_exact(pairs, node_pairs.len())?;
    pairs.extend(node_pairs.iter().map(|p| (p.left.0, p.right.0)));
    // Pair lists are unordered, so duplicates are grouped by a hash count in
    // O(p) rather than a sort; `out` comes back in hash order, which is allowed
    // because nothing reads a pair list positionally, and scaling distinct
    // pairs scales distinct children, so order does not matter.
    lim.reserve_map(counts, pairs.len())?;
    for &p in pairs.iter() {
        *counts.entry(p).or_insert(0) += 1;
    }
    if counts.len() == pairs.len() {
        // No duplicate pairs.
        return Ok(false);
    }

    let inlined = scale_duplicate_runs(eng, tdd, pv, counts, out, pairs.len())?;
    debug_assert!(out.len() <= pairs.len());
    if out.len() == pairs.len() {
        // Nothing absorbed — the pair list is unchanged as a multiset.
        return Ok(false);
    }
    write_back_resolved_pairs(tdd, pv, idx, out, pairs.len(), inlined);
    Ok(true)
}

/// Replace each run of k > 1 equal pairs with one pair whose marginal side is
/// scaled by k, keeping the run verbatim wherever the scale is declined.
/// Returns which sides received an inline marginal ref.
fn scale_duplicate_runs(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    counts: &rustc_hash::FxHashMap<(u32, u32), u32>,
    out: &mut Vec<ChildPair>,
    expected: usize,
) -> Result<Sides<bool>, OperationError> {
    // Worst case (nothing absorbs) `out` is the input multiset verbatim, so
    // one reservation covers every push below.
    eng.limits().reserve_exact(out, expected)?;
    let mut inlined = Sides { left: false, right: false };
    for (&(l, r), &k) in counts.iter() {
        let pair = ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r));
        if k == 1 {
            out.push(pair);
            continue;
        }
        let Some(res) = scale_pair_one_side(eng, tdd, pv, l, r, k) else {
            // The marginal side exists (checked at entry) but declined this
            // ref (see `try_scale_child`); keep the run as k multiset terms.
            for _ in 0..k {
                out.push(pair);
            }
            continue;
        };
        let scaled = res?;
        match scaled.inlined {
            Some(ChildSide::Left) => inlined.left = true,
            Some(ChildSide::Right) => inlined.right = true,
            None => {}
        }
        out.push(scaled.pair);
    }
    Ok(inlined)
}

/// Overwrite the node's pair-list prefix with the resolved pairs and shrink it,
/// raising the level's marginal-inline markers for any side that got an inline ref.
fn write_back_resolved_pairs(
    tdd: &mut Tdd,
    pv: VtreeIdx,
    idx: usize,
    out: &[ChildPair],
    old_len: usize,
    inlined: Sides<bool>,
) {
    // Write back: overwrite the prefix in place and shrink.
    let level = &mut tdd.levels[pv.idx()];
    // A scaled marginal ref may have come back inline (bit-30 tagged). Raise the
    // side's marker or the apply reader decodes the tagged count as a grid
    // coordinate.
    if inlined.left {
        level.set_has_value_refs(ChildSide::Left, true);
    }
    if inlined.right {
        level.set_has_value_refs(ChildSide::Right, true);
    }
    debug_assert!(
        !matches!(level.nodes[idx].kind(), NodeKind::Inline(_)),
        "inline single-pair node cannot hold duplicates"
    );
    let start = level.pair_range_at(idx).start;
    level.pairs_mut(idx)[..out.len()].copy_from_slice(out);
    let dead = level.reencode_shrunk(idx, start, old_len, out.len());
    level.note_dead_pairs(dead);
}
