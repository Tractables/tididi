//! Content-twin merge (a twin-contraction mechanism).
//!
//! Two nodes at the same level can be raw-identical (same pair multisets) yet
//! sit in different parent contexts, so the context-based
//! `sweep::contract_all_twins` — which groups by the multiset of
//! `(parent_node, sibling)` contexts — cannot see them.
//! `merge_content_equal_nodes` detects them by pair-multiset content, through
//! the same grouping over a node's own pairs instead of its contexts, and
//! rewrites parent/output refs onto the canonical node.
//!
//! The prune, merge, contract fixpoint that drives the merge is
//! `reduce::driver::Reduction::content_twins`.

use crate::Engine;

use crate::diagram::Tdd;
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;

use super::fingerprint::{group_twins_by_entries, ContentEntries};
use super::scratch::ContractScratch;

/// The levels this merge canonicalizes, children before parents; the checker
/// `test_helpers::check::marginal::check_twin_canonicality` walks the same set.
///
/// Empty on a diagram with no marginal level (the `# Soundness` block on
/// `merge_content_equal_nodes`). Otherwise every internal vtree node whose own
/// level and whose parent's level are explicit: a marginal level has counts,
/// not pair structure, and a level under a marginal ancestor is unreferenced.
pub(crate) fn content_twin_scan_levels(tdd: &Tdd) -> Vec<VtreeIdx> {
    if !tdd.has_marginal_level() {
        return Vec::new();
    }
    tdd.vtree
        .internal_bottomup_slice()
        .iter()
        .copied()
        .filter(|&v| {
            !tdd.levels[v.idx()].is_marginal()
                && tdd
                    .vtree
                    .node(v)
                    .parent()
                    .is_none_or(|p| !tdd.levels[p.idx()].is_marginal())
        })
        .collect()
}

/// Content-based twin merge over every explicit level of a marginalized
/// diagram, returning how many duplicate nodes were redirected onto their
/// canonical twin. A return of 0 means invariant 9 holds everywhere the filter
/// reached.
///
/// Two nodes at one level with equal pair multisets are found by the twin
/// grouping over each node's own pairs, and the parent's references and the
/// output reference are rewritten onto the lowest index. The duplicates are
/// left in place as unreferenced nodes, not tombstoned (a streaming apply
/// asserts tombstone-free levels); the caller must follow with a prune. The
/// parent level is marked dirty for the next contract pass. Levels are
/// scanned children before parents, so a merge at `L` that makes two nodes of
/// `parent(L)` content-equal is caught later in the same pass.
///
/// # Soundness
///
/// The pass stands down on a diagram with no marginal level. Content equality
/// is function equality there, which invariant 1 forbids between two nodes of
/// one level, and the duplicate pair a redirect leaves at a parent is legal
/// only in a marginalized diagram: a pair list is a multiset feeding a sum,
/// so `c(x)·c(B₁) + c(x)·c(B₂)` with `B₁`, `B₂` content-identical is
/// `2·c(x)·c(B₁)`, two assignment families sharing a value. A content twin in
/// Boolean mode is an upstream determinism violation, not work for this pass.
///
/// `filter`: with `Some(set)`, a level `L` is scanned only when `L` or its
/// marginal child is in `set` (the slot prune reports value merges under the
/// marginal level's index while the twins they mint appear at the parent).
/// With `None`, every explicit level is scanned. A filtered-out level can only
/// be left with unmerged twins, which costs size, not correctness. The set is
/// taken by value because the pass adds to it as it goes.
pub(crate) fn merge_content_equal_nodes(
    eng: &Engine,
    tdd: &mut Tdd,
    filter: Option<rustc_hash::FxHashSet<u32>>,
) -> Result<usize, OperationError> {
    // Marginalized diagrams only (`# Soundness` above).
    if !tdd.has_marginal_level() {
        return Ok(0);
    }

    let lim = eng.limits();
    let mut dups_merged = 0usize;

    // Children-before-parents order, collected upfront to avoid borrow issues
    // during the mut walk. `internal_bottomup_slice` is the bottom-up topological
    // order, so a level's parent is always visited strictly later in this pass —
    // which is what lets a single pass chase the merge cascade upward.
    let order = content_twin_scan_levels(tdd);

    // The worklist filter, grown in-pass. A merge at level L rewrites
    // parent(L)'s refs, so parent(L) must be scanned even if last round's
    // worklist did not name it; it is later in `order`, so inserting it here
    // takes effect within this same pass.
    let mut live = filter;

    // The contraction scratch: grouping by content is the contraction's
    // grouping with a node's own pairs as its entries, and no contraction
    // sweep runs while this pass does.
    let mut scratch = eng.reduce_scratch().contract.checkout(lim);

    for level_v in order {
        let level_idx = level_v.idx();

        // Worklist filter: skip this level if neither it nor a marginal child
        // was touched in the previous round (or earlier in this pass).
        if let Some(set) = live.as_ref() {
            let (cl, cr) = tdd.vtree.children(level_v);
            let touched = set.contains(&level_v.0)
                || (tdd.levels[cl.idx()].is_marginal() && set.contains(&cl.0))
                || (tdd.levels[cr.idx()].is_marginal() && set.contains(&cr.0));
            if !touched {
                continue;
            }
        }

        let level = &tdd.levels[level_idx];
        let width = level.slot_count();
        if width <= 1 {
            continue;
        }
        if !group_twins_by_entries(eng, &ContentEntries(level), width, &mut scratch)? {
            continue;
        }
        dups_merged += remap_groups_onto_first(lim, &mut scratch, width)?;
        redirect_parent_refs(tdd, level_v, &scratch.remap.merge_target[..width], &mut live);
    }

    Ok(dups_merged)
}

/// Fill `scratch.remap.merge_target[..width]` with each node's canonical
/// index: itself, or the lowest index of its group. Returns how many nodes
/// map onto another.
fn remap_groups_onto_first(
    lim: &Limits,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<usize, OperationError> {
    let ContractScratch { remap, group_starts, flat_groups, .. } = scratch;
    let target = &mut remap.merge_target;
    lim.try_resize(target, width, 0u32)?;
    for (i, slot) in target.iter_mut().enumerate().take(width) {
        *slot = i as u32;
    }
    let mut dups = 0usize;
    for g in 0..group_starts.len() {
        let start = group_starts[g] as usize;
        let end = group_starts.get(g + 1).map_or(flat_groups.len(), |&e| e as usize);
        let first = flat_groups[start];
        for &member in &flat_groups[start + 1..end] {
            target[member as usize] = first;
        }
        dups += end - start - 1;
    }
    Ok(dups)
}

/// Point the output ref and the parent's refs at each duplicate's canonical
/// node, then mark the parent for the follow-up contract and content scans.
fn redirect_parent_refs(
    tdd: &mut Tdd,
    level_v: VtreeIdx,
    remap: &[u32],
    live: &mut Option<rustc_hash::FxHashSet<u32>>,
) {
    let Some(parent) = tdd.merge_level_nodes(level_v, remap) else { return };
    // In-pass cascade: the rewrite may have made two of the parent's nodes
    // content-equal. The parent is later in `order`, so admitting it to the
    // live worklist means the current pass catches the new twins.
    if let Some(set) = live.as_mut() {
        set.insert(parent.0);
    }
}
