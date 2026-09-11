//! Negation: make-full transformation and complement operation.
//!
//! A diagram is **t-full** at vtree level `t` if the disjunction of all t-nodes
//! equals the constant-true function. A diagram is **full** if t-full at every level.
//! `expand_full` materializes the fill nodes explicitly (paper Prop 5.3), used by
//! `negate_tdd`, which in turn powers the disjunction in the sibling
//! `disjoin` module.

use std::collections::HashSet;
use std::sync::Arc;

use crate::diagram::*;

/// Negate a diagram: make it full, then complement at the root, then minimize.
///
/// Exact, but it can grow the diagram sharply — a diagram stores only the pair
/// structure of its satisfying assignments, so the fill that has to precede the
/// complement typically dominates. When only the count of `¬f` is wanted,
/// `2^n - count(f)` avoids building it at all.
#[must_use]
pub fn negate(f: Tdd) -> Tdd {
    let mut result = negate_tdd_owned(f);
    crate::reduce::minimize(&mut result);
    result
}


/// Make `tdd` full and complement it at the root, consuming the operand.
/// Negation without the reduction that finishes it: [`negate()`] is this plus
/// `minimize`, and disjunction reaches it directly for its two operands and for
/// the conjunction De Morgan leaves it with.
pub(crate) fn negate_tdd_owned(mut tdd: Tdd) -> Tdd {
    if tdd.is_zero() {
        return Tdd::one(&tdd.vtree);
    }

    let vtree = Arc::clone(&tdd.vtree);
    expand_full(&mut tdd);
    complement_full_at_root(tdd, &vtree)
}

/// Complement an already-made-full diagram at its root (paper Prop 5.4): collect all
/// root-level pairs not in the output node, filter dead pairs. `orig_vtree` is
/// the operand's vtree (used for the constant-zero/one fallbacks). Shared by
/// [`negate_tdd_owned`], its one caller.
fn complement_full_at_root(full_tdd: Tdd, orig_vtree: &Arc<crate::vtree::Vtree>) -> Tdd {
    let vtree = &full_tdd.vtree;
    let root = vtree.root();
    let root_idx = root.idx();
    let out_local = full_tdd.output.local;

    let mut levels = full_tdd.levels;

    if vtree.node(root).is_leaf() {
        // Implicit leaf: the output index and the label are the same number. It stays in
        // {Pos,Neg,One} unless the output is One (complement = Zero, returned as the zero
        // constant diagram).
        let Some(neg_local) = complement_leaf_root(out_local) else {
            return Tdd::zero(orig_vtree);
        };
        Tdd::from_levels_unchecked(
            Arc::clone(orig_vtree),
            levels,
            TddNodeId { vtree: root, local: neg_local },
        )
    } else {
        // After expand_full (which expands One → Pos+Neg), child widths:
        // - Leaf children: 2 (the disjoint set {Pos, Neg})
        // - Internal children: stored width (includes any fill nodes)
        let (left, right) = vtree.children(root);
        let lefts = ChildBasis::of(vtree, &levels, left);
        let rights = ChildBasis::of(vtree, &levels, right);

        let mut neg_pairs = collect_complement_pairs(&levels[root_idx], out_local, lefts, rights);

        // Filter out dead pairs: pairs where a child computes the Zero function
        // (internal node with empty pairs). This happens when expand_full adds fill
        // nodes that compute Zero (complement of all existing nodes). Leaf children
        // are always non-Zero (implicit Pos/Neg/One).
        // Precondition: root's children must not be marginal (project_var already
        // asserts no marginal ancestor; negate_tdd must not be called when root
        // children are marginal, because the structural complement is undefined
        // on a partially-aggregated diagram).
        let left_is_leaf = vtree.node(left).is_leaf();
        let right_is_leaf = vtree.node(right).is_leaf();
        debug_assert!(
            left_is_leaf || !levels[left.idx()].is_marginal(),
            "negate_tdd: root's left child VtreeIdx({}) is marginal — \
             complement is undefined on a partially-aggregated TDD",
            left.idx()
        );
        debug_assert!(
            right_is_leaf || !levels[right.idx()].is_marginal(),
            "negate_tdd: root's right child VtreeIdx({}) is marginal",
            right.idx()
        );
        neg_pairs.retain(|pair| {
            let left_alive = left_is_leaf || {
                let node = &levels[left.idx()].nodes[pair.left.idx()];
                node.is_internal() && !levels[left.idx()].pairs_of(node).is_empty()
            };
            let right_alive = right_is_leaf || {
                let node = &levels[right.idx()].nodes[pair.right.idx()];
                node.is_internal() && !levels[right.idx()].pairs_of(node).is_empty()
            };
            left_alive && right_alive
        });

        if neg_pairs.is_empty() {
            return Tdd::zero(orig_vtree);
        }

        let neg_idx = levels[root_idx].push_internal_node(&neg_pairs);

        Tdd::from_levels_unchecked(
            Arc::clone(orig_vtree),
            levels,
            TddNodeId { vtree: root, local: neg_idx },
        )
    }
}

// ── expand_full: explicit fill-node materialization ────────────────────────────

/// Make a diagram t-full by materializing fill nodes explicitly (paper Prop 5.3).
///
/// Expands every level to ensure each node pair has symmetric children.
/// Called by `negate_tdd` (and transitively by `apply_or`) before complementing.
pub(crate) fn expand_full(tdd: &mut Tdd) {
    let vtree = tdd.vtree.clone();

    // Expand One → {Pos, Neg} at levels with leaf children so all leaf
    // references are disjoint. After this, the leaf basis is {Pos, Neg}.
    expand_ones_at_leaf_parents(tdd);

    for (t, left, right) in vtree.internal_bottomup() {
        // Marginal levels have been streamed to marginal_counts; their node lists
        // are gone and cannot be made full. Skip them.
        if tdd.levels[t.idx()].is_marginal() {
            continue;
        }
        let lefts = ChildBasis::of(&vtree, &tdd.levels, left);
        let rights = ChildBasis::of(&vtree, &tdd.levels, right);
        expand_internal_explicit(&mut tdd.levels[t.idx()], lefts, rights);
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Complement a leaf label: Pos↔Neg, One↔Zero.
fn complement_label(label: LeafLabel) -> LeafLabel {
    match label {
        LeafLabel::Pos => LeafLabel::Neg,
        LeafLabel::Neg => LeafLabel::Pos,
        LeafLabel::One => LeafLabel::Zero,
        LeafLabel::Zero => LeafLabel::One,
    }
}

/// Complement of an implicit-leaf root, where the output index and the label
/// are the same number.
///
/// Returns `Some(complement_index)` when the complement is expressible in the
/// implicit leaf set {Pos, Neg, One}, or `None` if the complement would be Zero
/// (callers should substitute `Tdd::zero` in that case).
#[inline]
fn complement_leaf_root(out_local: NodeIdx) -> Option<NodeIdx> {
    let neg_label = complement_label(LeafLabel::from_idx(out_local.idx()));
    if neg_label == LeafLabel::Zero {
        None
    } else {
        Some(NodeIdx(neg_label as u32))
    }
}

// Leaf levels are always full by construction: implicit One=0, Pos=1, Neg=2
// cover all single-variable functions. No make_leaf_full needed.

/// Expand One-references at levels with leaf children to Pos+Neg pairs.
///
/// With implicit leaves, One (index 0) overlaps semantically with Pos (1) and
/// Neg (2). For `expand_full`'s cross-product to work correctly, we must expand
/// one into {Pos, Neg} pairs so all leaf references are disjoint.
///
/// After expansion, the leaf universe is {Pos=1, Neg=2} (width 2 per leaf child).
fn expand_ones_at_leaf_parents(tdd: &mut Tdd) {
    let vtree = tdd.vtree.clone();
    for (t, left, right) in vtree.internal_bottomup() {
        let left_leaf = vtree.node(left).is_leaf();
        let right_leaf = vtree.node(right).is_leaf();
        if left_leaf || right_leaf {
            expand_ones_in_level(&mut tdd.levels[t.idx()], left_leaf, right_leaf);
        }
    }
}

/// Expand One-references in a single level's pairs to Pos+Neg.
///
/// For each input pair, if a leaf child index is 0 (One), replace it with two
/// pairs: one for Pos (1) and one for Neg (2). Rebuilds the pairs arena and
/// nodes in-place with sorted, deduplicated pairs per node.
fn expand_ones_in_level(level: &mut TddLevel, left_leaf: bool, right_leaf: bool) {
    let mut new_pairs: Vec<InputPair> = Vec::with_capacity(level.pairs.len() * 2);
    let mut new_nodes: Vec<TddNodeData> = Vec::with_capacity(level.nodes.len());
    // Discard the old multi-pair range table — it references the old pairs arena which is
    // about to be replaced. encode_multi below will rebuild it as needed.
    level.multi_pairs.clear();

    for i in 0..level.nodes.len() {
        let node = level.nodes[i];
        if !node.is_internal() {
            new_nodes.push(node);
            continue;
        }
        let pair_start = new_pairs.len();
        // Iterate old pairs by index to avoid holding a borrow on `level`.
        let old_len = level.pairs_of_idx(i).len();
        for k in 0..old_len {
            let pair = level.pairs_of_idx(i)[k];
            // One → expand to both Pos and Neg (the two assignments under x).
            let lefts: &[u32] = if left_leaf && pair.left == ONE_LEAF_IDX {
                &[POS_LEAF_IDX.0, NEG_LEAF_IDX.0]
            } else {
                &[pair.left.0]
            };
            let rights: &[u32] = if right_leaf && pair.right == ONE_LEAF_IDX {
                &[POS_LEAF_IDX.0, NEG_LEAF_IDX.0]
            } else {
                &[pair.right.0]
            };
            for &l in lefts {
                for &r in rights {
                    // Push unconditionally; the per-node sort+dedup below produces
                    // the same pair set in O(m log m), where a membership scan
                    // per candidate over the growing slice would cost O(m²) on
                    // wide levels.
                    new_pairs.push(InputPair {
                        left: NodeIdx(l),
                        right: NodeIdx(r),
                    });
                }
            }
        }

        // Dedup this node's pair slice once via sort. No canonicalizing-order
        // requirement: make-full output feeds negation → minimize → twin
        // contraction, but twin detection is order-independent (`find_twin_groups`
        // sorts each signature slice before comparing), so the node's pair order is
        // free. Pair lists are unordered sets (see `InputPair`).
        {
            let tail = &mut new_pairs[pair_start..];
            tail.sort_unstable();
            let mut w = 0usize;
            for r in 0..tail.len() {
                if r == 0 || tail[r] != tail[w - 1] {
                    tail[w] = tail[r];
                    w += 1;
                }
            }
            new_pairs.truncate(pair_start + w);
        }
        let pair_len = new_pairs.len() - pair_start;
        new_nodes.push(if pair_len == 1 {
            let data = level.encode_single(pair_start, new_pairs[pair_start]);
            if data.is_inline() {
                // The pair rides in the node itself; its arena slot goes.
                new_pairs.pop();
            }
            data
        } else {
            level.encode_multi(pair_start, pair_len)
        });
    }

    level.pairs = new_pairs;
    level.nodes = new_nodes;
    // The rebuilt arena is garbage-free by construction — carrying the old
    // count over would trigger a pointless full-arena sweep later.
    level.dead_pairs = 0;
}

// The leaf basis is a contiguous range only because Pos and Neg are adjacent.
const _: () = assert!(NEG_LEAF_IDX.0 == POS_LEAF_IDX.0 + 1);

/// Expanded child basis: the actual local indices the cross-product spans.
///
/// For leaf children after One-expansion the disjoint basis is `{Pos, Neg}` —
/// indices `{1, 2}` under the new leaf encoding (One=0, Pos=1, Neg=2). For
/// internal children every stored node is its own basis element, so the range
/// is `0..width`. Both cases are contiguous, so membership is a bounds test and
/// enumeration is a range walk — no index vector is materialized (the internal
/// case would be one `u32` per node of a child level that can be very wide).
#[derive(Copy, Clone)]
struct ChildBasis {
    /// First index in the basis.
    start: u32,
    /// One past the last index in the basis.
    end: u32,
}

impl ChildBasis {
    /// The basis `child` contributes, read off the vtree shape and level width.
    fn of(
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        child: crate::vtree::VtreeIdx,
    ) -> ChildBasis {
        if vtree.node(child).is_leaf() {
            ChildBasis { start: POS_LEAF_IDX.0, end: NEG_LEAF_IDX.0 + 1 }
        } else {
            ChildBasis { start: 0, end: levels[child.idx()].width() as u32 }
        }
    }

    /// Number of basis elements — one factor of the cross-product cell count.
    #[inline]
    fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    /// The one membership check: in-basis iff inside the contiguous range.
    #[inline]
    fn contains(self, idx: u32) -> bool {
        idx >= self.start && idx < self.end
    }

    /// Enumerate the basis, in the same order the materialized vector had.
    #[inline]
    fn iter(self) -> std::ops::Range<u32> {
        self.start..self.end
    }
}

/// Make an internal level t-full by materializing fill pairs explicitly.
fn expand_internal_explicit(
    level: &mut TddLevel,
    lefts: ChildBasis,
    rights: ChildBasis,
) {
    // A level is full iff its nodes cover every cell of the `lefts × rights`
    // basis. We deduplicate the covered cells into a set: this is correct even
    // when the level is non-canonical: an un-minimized diagram can list the same
    // (l,r) cell under two un-merged nodes. Counting pair-list lengths with
    // multiplicity instead would let those duplicates reach the basis size and
    // judge the level full, dropping genuine fill pairs. `used` is restricted to
    // in-basis cells so out-of-range pairs can't mask a gap.
    let basis = lefts.len() * rights.len();
    // `used` gains at most one entry per pair actually iterated, so reserve
    // against the level's pair mass (arena pairs plus at most one inline pair
    // per node), not against `basis` — the cell count is quadratic in the child
    // widths and reserving it outright allocates gigabytes on a wide level.
    let cap = basis.min(level.pair_count() + level.nodes.len());
    let mut used: HashSet<(u32, u32)> = HashSet::with_capacity(cap);
    for node in &level.nodes {
        if node.is_internal() {
            for pair in level.pairs_of(node) {
                if lefts.contains(pair.left.0) && rights.contains(pair.right.0) {
                    used.insert((pair.left.0, pair.right.0));
                }
            }
        }
    }
    // Covered every basis cell ⇒ already full; skip the O(|L|·|R|) enumeration.
    if used.len() == basis {
        return;
    }

    let fill_pairs = missing_cells(&used, lefts, rights);
    if fill_pairs.is_empty() {
        return;
    }

    level.push_internal_node(&fill_pairs);
}

/// Collect all pairs that are not in the excluded node's pair set.
///
/// Iterates `lefts × rights` and returns pairs not in the excluded node. Used
/// for negation at the root level.
fn collect_complement_pairs(
    level: &TddLevel,
    exclude_node: NodeIdx,
    lefts: ChildBasis,
    rights: ChildBasis,
) -> Vec<InputPair> {
    let exclude_pairs = level.pairs_of_idx(exclude_node.idx());
    let mut excluded = HashSet::with_capacity(exclude_pairs.len());
    for pair in exclude_pairs.iter() {
        excluded.insert((pair.left.0, pair.right.0));
    }
    missing_cells(&excluded, lefts, rights)
}

/// The cells of `lefts × rights` that `used` does not cover.
fn missing_cells(
    used: &HashSet<(u32, u32)>,
    lefts: ChildBasis,
    rights: ChildBasis,
) -> Vec<InputPair> {
    let mut out = Vec::new();
    for l in lefts.iter() {
        for r in rights.iter() {
            if !used.contains(&(l, r)) {
                out.push(InputPair { left: NodeIdx(l), right: NodeIdx(r) });
            }
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
