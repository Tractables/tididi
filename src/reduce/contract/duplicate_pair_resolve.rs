//! Duplicate-pair resolution by multiplicity fork-down ("scale" rewrite).
//!
//! Twin contraction above a marginal boundary can produce duplicate `(L, R)`
//! entries in a plain (no inlined side) node's pair list: merging content-
//! equal or partially-overlapping context-twins folds k disjoint upstream
//! plan families onto the same structural pair, and at plain levels there is
//! no count field to carry the multiplicity k.
//!
//! Leaving the duplicates in place is not *count*-wrong — the count recurrences
//! sum over a node's pairs, so k copies of `(L, R)` already contribute
//! `k·c(L)·c(R)`, and since the
//! content-twin merge was widened to every explicit level the whole plain-level
//! machinery is stated for multisets (`contract/merge.rs`'s duplicate check and
//! `conjoin/sparse.rs`'s Phase F check are both diagram-scoped now). So the
//! rewrite below is a SIZE optimization — k pair slots become one — never a
//! correctness obligation.
//!
//! Resolution: replace the k copies of
//! `(L, R)` with a single pair whose marg-carrying side is *scaled by k* — a
//! fresh value denoting k times the original's. Scaling a marg-side ref
//! multiplies one count (inline re-encode, or one fresh slot) — except at a
//! weight-marginal LEAF, whose 3-slot column is pinned and admits no mint: there
//! the fold succeeds only when the scaled value is one the column already carries
//! (`scale_weight_leaf_by_lookup`), which after equal-value ref canonicalization
//! is the common `2·Pos = One` case.
//!
//! ## Cost policy: only an O(1) absorber is taken
//!
//! A pair has an absorbing side only where marginalization lies below it, and
//! the two kinds of absorber are not comparable:
//!
//! * a **marginal child** absorbs in O(1) — the factor multiplies one count;
//! * a **structural child** would have to be CLONED with one of *its* marg-side
//!   children scaled, recursing down the vtree until some count absorbs the
//!   factor — minting a scaled copy of every node on the way down.
//!
//! The structural descent used to run whenever no marginal side was available.
//! Measured on `mc2024_track1_062` it entered ~43 M times and minted
//! ~6.3 M cloned nodes in 11 s — trading ~17 M pair slots
//! for ~6.3 M new nodes plus the work every later pass then does on the bigger
//! diagram — and it accounted for ~37–41 % of the compile's self time. It never
//! paid: it *grows* the diagram to shrink pair lists.
//!
//! So the resolution now takes the O(1) absorber and nothing else. When neither
//! child of the plain level is marginal, the duplicate run is left in place as k
//! legal multiset terms — the twin merge that produced it still stands (it is the
//! merge that removed k−1 NODES), only its pair-list representation is left
//! un-collapsed. `resolve_duplicate_pairs_in_node` early-outs on that level
//! shape, so a level with no O(1) absorber pays nothing at all.
//!
//! Scaled counts are fresh slots; slot-prune value-merge later shares them with
//! existing equal-valued slots (slot-count uniqueness is restored by that pass, not by construction
//! here).


use crate::engine::Engine;
use super::scratch::DupScratch;
use crate::diagram::ChildSide;
use crate::error::ApplyError;
use crate::diagram::*;

#[path = "dup_scale.rs"]
mod scale;
use scale::{has_o1_absorber, scale_pair_one_side};
use crate::vtree::VtreeIdx;

/// `has_marg_below[v]` — v's vtree subtree (including v itself) contains a
/// marginal level. Stable across one minimize pass: marginalization converts
/// levels between compile phases, never during contraction. Fills `below` in
/// place (buffer reused across contract runs via ContractScratch).
///
/// O(vtree nodes), so the caller fills it lazily — on the first merge a sweep
/// attempts, never on a sweep that finds no twins. See
/// `ContractScratch::has_marg_below_valid`.
pub(crate) fn compute_has_marg_below_into(tdd: &Tdd, below: &mut Vec<bool>) {
    let n = tdd.vtree.num_nodes();
    below.clear();
    below.resize(n, false);
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
/// entry, so it carries capacity across nodes and nothing else — including out
/// of the `?` bails below, which leave it dirty by design.
pub(super) fn resolve_duplicate_pairs_in_node(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    idx: usize,
    scratch: &mut DupScratch,
) -> Result<bool, ApplyError> {
    // `pv` is a plain (non-marginal) level — marginal levels are p-fusion's
    // domain. Its inline markers are NOT asserted clear: scaling a marginal child
    // ref can mint an INLINE marg ref into `pv`'s pairs, which raises `pv`'s
    // `MARG_INLINED_*` marker (below). The caller resolves several survivors per
    // pass, so the second and later calls legitimately see the marker already up.
    debug_assert!(
        !tdd.levels[pv.idx()].is_marginal(),
        "resolve_duplicate_pairs_in_node: marginal levels are p-fusion's domain"
    );
    // Cost policy early-out: with no marginal child there is no O(1) absorber,
    // so every run would be kept anyway — don't pay the collect + hash-count.
    // This is the whole cost on levels the widened content merge made duplicate-
    // rich (millions of calls over tens of millions of pairs).
    if !has_o1_absorber(tdd, pv) {
        return Ok(false);
    }
    scratch.clear();
    let DupScratch { pairs, counts, out } = scratch;
    pairs.extend(
        tdd.levels[pv.idx()]
            .pairs_of_idx(idx)
            .iter()
            .map(|p| (p.left.0, p.right.0)),
    );
    if pairs.len() < 2 {
        return Ok(false);
    }
    // Pair lists are unordered sets — group duplicates with a
    // hash count in O(p) instead of an O(p·log p) sort. The output multiset
    // {(distinct pair, multiplicity k)} is identical to the former sort+run-length
    // form; `out` is written back in arbitrary (hash) order, which is allowed
    // because a pair list is a multiset, and `try_scale_child(child, k)` is
    // order-independent (distinct pairs scale distinct children). This is the
    // fork-down dup-resolution hot path on contraction-bound diagrams, where the
    // survivor list can grow large.
    //
    // The map is REUSED across the pass's nodes (cleared above), so its table
    // can be wider than a fresh `reserve` would make it and the hash order —
    // hence `out`'s order — need not match a cold call's. Nothing downstream
    // reads a pair list positionally.
    counts.reserve(pairs.len());
    for &p in pairs.iter() {
        *counts.entry(p).or_insert(0) += 1;
    }
    if counts.len() == pairs.len() {
        // No duplicate pairs.
        return Ok(false);
    }

    let (inl_left, inl_right) = scale_duplicate_runs(eng, tdd, pv, counts, out, pairs.len())?;
    debug_assert!(out.len() <= pairs.len());
    if out.len() == pairs.len() {
        // Nothing absorbed — the pair list is unchanged as a multiset.
        return Ok(false);
    }
    write_back_resolved_pairs(eng, tdd, pv, idx, out, pairs.len(), inl_left, inl_right)?;
    Ok(true)
}

/// Replace each run of k > 1 equal pairs with one pair whose marginal side is
/// scaled by k, keeping the run verbatim wherever the scale is declined.
/// Returns which sides received an inline marg ref.
fn scale_duplicate_runs(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    counts: &rustc_hash::FxHashMap<(u32, u32), u32>,
    out: &mut Vec<InputPair>,
    expected: usize,
) -> Result<(bool, bool), ApplyError> {
    // Worst case (nothing absorbs) `out` is the input multiset verbatim.
    out.reserve(expected);
    let mut inl_left = false;
    let mut inl_right = false;
    for (&(l, r), &k) in counts.iter() {
        let pair = InputPair { left: NodeIdx(l), right: NodeIdx(r) };
        if k == 1 {
            out.push(pair);
            continue;
        }
        let Some(res) = scale_pair_one_side(eng, tdd, pv, l, r, k) else {
            // The marginal side exists (checked at entry) but declined this
            // particular ref — an integer-marginal leaf label whose scaled value
            // will not inline (minting a slot into that leaf store would be
            // decoded back as a label), or a weight-marginal leaf whose pinned
            // global `leaf_val` column does not already carry the scaled value
            // (and admits no minted slot to hold it). Keep the run as k legal
            // multiset terms.
            for _ in 0..k {
                out.push(pair);
            }
            continue;
        };
        let scaled = res?;
        match scaled.inlined {
            Some(ChildSide::Left) => inl_left = true,
            Some(ChildSide::Right) => inl_right = true,
            None => {}
        }
        out.push(scaled.pair);
    }
    Ok((inl_left, inl_right))
}

/// Overwrite the node's pair-list prefix with the resolved pairs and shrink it,
/// raising the level's marg-inline markers for any side that got an inline ref.
// The contraction scratch buffers are passed separately so they can be
// borrowed independently of the diagram they index into.
#[allow(clippy::too_many_arguments)]
fn write_back_resolved_pairs(
    eng: &Engine,
    tdd: &mut Tdd,
    pv: VtreeIdx,
    idx: usize,
    out: &[InputPair],
    old_len: usize,
    inl_left: bool,
    inl_right: bool,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    // Write back: overwrite the prefix in place and shrink.
    let level = &mut tdd.levels[pv.idx()];
    // A scaled marg ref may have come back INLINE (bit-30 tagged). Raise the
    // side's marker or the apply reader decodes the tagged count as a grid
    // coordinate.
    if inl_left {
        level.set_marg_inlined_left(true);
    }
    if inl_right {
        level.set_marg_inlined_right(true);
    }
    if level.nodes[idx].is_inline() {
        unreachable!("inline single-pair node cannot hold duplicates");
    }
    let new_len = out.len();
    // Slots the shrink leaves behind are unreferenced arena; the two `new_len == 1`
    // re-encodes abandon the whole old range (inline word, or a fresh tail slot).
    let abandoned = if new_len == 1 { old_len } else { old_len - new_len };
    if new_len == 1 {
        let surviving = out[0];
        if surviving.can_inline() {
            level.nodes[idx] = TddNodeData::inline(surviving);
        } else {
            // Single pair that can't inline: extended multi with len=1, whose
            // pair is PUSHED as a fresh tail slot rather than aliased in place —
            // `out` is a rewritten pair, not necessarily one already sitting in
            // this node's range. That is why `abandoned` above is the WHOLE old
            // range: the node stops referencing every one of its former slots.
            let pair_start = level.pairs.len();
            lim.try_push(&mut level.pairs, surviving)?;
            let ext_idx = level.ext.len();
            lim.try_push(&mut level.ext, ExtMulti { start: pair_start as u64, len: 1 })?;
            level.nodes[idx] = TddNodeData::multi_extended(ext_idx as u32);
        }
    } else {
        let dst = level.pairs_mut(idx);
        dst[..new_len].copy_from_slice(out);
        level.set_pair_len(idx, new_len as u32);
    }
    level.note_dead_pairs(abandoned);
    Ok(())
}
