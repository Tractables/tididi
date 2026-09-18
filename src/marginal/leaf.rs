//! Summing out a single-variable vtree leaf, in both representations.

use crate::diagram::WeightValue;
use crate::diagram::{for_each_side_ref_mut, ChildSide, LeafLabel, MarginalSide, ValueRef, Tdd, TddLevel};
use crate::diagram::{leaf_canon_map, leaf_column_vals, leaf_count};
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Sum out a single-variable vtree leaf by inlining its fixed model count
/// directly into the parent's leaf-side refs.
///
/// A leaf's marginal count is fixed by its label (One→2, Pos/Neg→1, Zero→0), so
/// it always fits `ValueRef::Inline` and the leaf's store stays empty;
/// `become_marginal(vec![], None)` only flips the `is_marginal()` signal.
/// Rewriting Pos and Neg to the same `Inline(1)` makes the parent's `(·,x)` and
/// `(·,¬x)` branches structurally equal, so the contraction and pair-fusion
/// passes merge the now-twin parent nodes.
///
/// No-op when the parent is already marginal: the leaf's count was then folded
/// into the parent's store, so there are no pairs left to rewrite.
pub(crate) fn marginalize_leaf_inline(
    tdd: &mut Tdd,
    leaf: VtreeIdx,
    vtree: &Vtree,
) {
    debug_assert!(vtree.node(leaf).is_leaf());
    if tdd.levels[leaf.idx()].is_marginal() {
        return;
    }
    // The inline range must hold the largest leaf count (One→2).
    const _: () = assert!(crate::diagram::MARGINAL_INLINE_MAX >= 2);
    if let Some(parent_vi) = vtree.node(leaf).parent() {
        let pi = parent_vi.idx();
        if !tdd.levels[pi].is_marginal() {
            let (pl, _) = vtree.children(parent_vi);
            let leaf_is_left = pl == leaf;
            let side = if leaf_is_left { ChildSide::Left } else { ChildSide::Right };
            inline_leaf_refs_at_parent(tdd, parent_vi, side);
            tdd.invalidate(parent_vi);
            if leaf_is_left {
                tdd.levels[pi].set_marginal_inlined_left(true);
            } else {
                tdd.levels[pi].set_marginal_inlined_right(true);
            }
        }
    }
    // Flip the reader/apply signal; the store stays empty (all counts are inline
    // at the parent).
    tdd.levels[leaf.idx()].become_marginal(Vec::new(), None);
}

/// Rewrite every leaf-side ref of `parent_v`'s nodes from a `LeafLabel` index
/// (One/Pos/Neg) into a `ValueRef::Inline(count)` (2/1/1). Mirrors
/// `remap_refs_into`, but maps leaf labels to inline counts instead of
/// remapping slot indices. Bit 30 (the inline tag) is disjoint from
/// `LEAF_BIT/MULTI_BIT` (bit 31), so the rewritten refs keep their inline/multi
/// node encoding.
fn inline_leaf_refs_at_parent(tdd: &mut Tdd, parent_v: VtreeIdx, side: ChildSide) {
    let to_inline = |raw: u32| -> u32 {
        if MarginalSide(raw).is_zero_sentinel() {
            return raw; // ZERO sentinel (count 0) — already self-describing
        }
        // Idempotent: a ref that already carries the inline tag (bit 30) is an
        // Inline(count) we wrote on a prior pass — leave it untouched. Without
        // this guard a re-entry (parent revisited while its leaf-side refs are
        // already inline) would feed a bit-30 value into `LeafLabel::from_idx`,
        // whose `_ => unreachable!` panics (the leaf labels are only 0/1/2).
        if ValueRef::is_inline_raw(raw) {
            return raw;
        }
        let count: u128 = match raw {
            0..=2 => leaf_count(LeafLabel::from_idx(raw as usize)),
            // Zero is the sentinel index — defensive; not normally stored.
            3 => leaf_count(LeafLabel::Zero),
            other => panic!("inline_leaf_refs_at_parent: unexpected leaf-side ref {other}"),
        };
        ValueRef::inline_raw(count).expect("leaf count 0/1/2 always fits inline")
    };
    for_each_side_ref_mut(&mut tdd.levels[parent_v.idx()], side, |r| *r = to_inline(*r));
}

/// Rewrite every leaf-side ref of `plevel`'s nodes onto the canonical slot of an
/// equal-value class in a weight-marginal leaf's pinned column (`canon` from
/// [`leaf_canon_map`]); `canon` must not be the identity map.
///
/// A ref only ever moves onto a slot holding the same value, so every reader
/// resolves it to the number it resolved to before. What changes is structure:
/// `(·, Pos)` and `(·, Neg)` become byte-identical when w⁺ = w⁻, so the
/// parent's nodes become twins and contraction collapses them. That is sound
/// because a marginalized leaf's variable is private (no further conjunction
/// can case-split on it). The column itself is never touched, which is what
/// the pin (invariant 11) permits.
pub(crate) fn canonicalize_leaf_refs_at_parent(
    plevel: &mut TddLevel,
    side: ChildSide,
    canon: &[u32; 3],
) {
    debug_assert!(
        *canon != [0, 1, 2],
        "canonicalize_leaf_refs_at_parent: identity map — the caller must skip \
         the walk rather than pay a level scan that rewrites nothing"
    );
    let to_canon = |raw: u32| -> u32 {
        if MarginalSide(raw).is_zero_sentinel() {
            return raw; // ZERO sentinel — carries no slot
        }
        // A weighted leaf side is a bare slot in the pinned label range: the
        // weighted scale never mints an inline ref, and an inline ref has bit
        // 30 set, so it is out of range here too.
        debug_assert!(
            (raw as usize) < crate::diagram::LEAF_WIDTH,
            "canonicalize_leaf_refs_at_parent: leaf-side ref {raw} outside the \
             pinned label range"
        );
        // Out of range means the pin is already broken; leave the ref alone so
        // `test_helpers::check::marginal::check_leaf_columns_pinned` check #2 reports it at its own site
        // rather than this one panicking on an index.
        canon.get(raw as usize).copied().unwrap_or(raw)
    };
    for_each_side_ref_mut(plevel, side, |r| *r = to_canon(*r));
}

/// Weighted analogue of [`marginalize_leaf_inline`]: sum out a single-variable
/// vtree leaf carrying semiring values.
///
/// A weighted value has no inline encoding, so the leaf level gets a three-slot
/// weighted store in [`LeafLabel::from_idx`] order (0 = One, 1 = Pos, 2 = Neg),
/// filled from [`WeightStore::leaf_val`]. A parent's leaf-side refs are bare
/// leaf-label indices, which are valid slot indices into that column, so no
/// parent-ref rewrite is needed. A Zero leaf-side ref carries bit 31 and every
/// weighted reader tests that bit before decoding a slot.
///
/// In the exact domain, [`leaf_canon_map`] partitions the column by value and
/// [`canonicalize_leaf_refs_at_parent`] moves each leaf-side ref of a structural
/// parent onto its class's smallest slot (Neg→Pos when w⁺ = w⁻, Pos→One when
/// w⁻ = 0, Neg→One when w⁺ = 0), the weighted form of the integer arm's
/// Pos/Neg→`Inline(1)` twin merge. The log domain skips it: its key equality is
/// `f64` bit equality, not value equality.
///
/// # Soundness
///
/// The column installed here is pinned (architecture invariant 11): it is
/// shared by every diagram whose store this one's was merged into, while a
/// parent-ref rewrite reaches one `Tdd` only, so compacting, reordering or
/// appending to it would desynchronise every other holder.
/// [`check_leaf_columns_pinned`](crate::test_helpers::check::marginal::check_leaf_columns_pinned)
/// decides the invariant.
pub(crate) fn marginalize_leaf_weighted(
    tdd: &mut Tdd,
    leaf: VtreeIdx,
    vtree: &Vtree,
    ws: &mut WeightStore,
) {
    debug_assert!(vtree.node(leaf).is_leaf());
    let left_idx = leaf.idx();
    if tdd.levels[left_idx].is_marginal() {
        return;
    }
    let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(left_idx as u32)) else { return };
    let parent = vtree.node(leaf).parent();
    // Slot i ≡ `LeafLabel::from_idx(i)`, which is what makes the parent's existing
    // bare leaf-label refs valid slot refs without a rewrite. If that order ever
    // changes, every parent ref into a weight-marginal leaf silently reads the
    // wrong base.
    debug_assert!(matches!(
        (LeafLabel::from_idx(0), LeafLabel::from_idx(1), LeafLabel::from_idx(2)),
        (LeafLabel::One, LeafLabel::Pos, LeafLabel::Neg)
    ));
    let values: Vec<WeightValue> = leaf_column_vals(ws, var);
    // A subsumed leaf (parent already marginal) gets the same full column: the
    // column is shared with every diagram this one's store reaches, and another
    // holder whose leaf level is still structural decodes its bare leaf-label
    // refs against it, so a shorter column would misread there. Subsumption
    // only skips the ref rewrite and the parent invalidation: a marginal parent
    // has folded this leaf's bases into its own aggregate and has no leaf-side
    // pairs left.
    if let Some(parent_vi) = parent
        && !tdd.levels[parent_vi.idx()].is_marginal() {
            // Exact domain only; see the doc above.
            if !ws.is_log() {
                let canon = leaf_canon_map(&values);
                if canon != [0, 1, 2] {
                    let (pl, _) = vtree.children(parent_vi);
                    let side =
                        if pl == leaf { ChildSide::Left } else { ChildSide::Right };
                    canonicalize_leaf_refs_at_parent(
                        &mut tdd.levels[parent_vi.idx()],
                        side,
                        &canon,
                    );
                }
            }
            tdd.invalidate(parent_vi);
        }
    tdd.levels[left_idx].become_marginal_weighted(values.len() as u32);
    ws.set_level(left_idx, values);
}

/// Rewrite the leaf-side refs of every leaf that one operand made
/// weight-marginal onto that leaf's canonical slots.
///
/// It runs after the bottom-up loop because the parent's pairs are only final
/// then, and it shares the walk the marginalization pass uses, so a leaf whose
/// column holds equal values ends up with one representative rather than two
/// slots the contraction would have to recognize as twins.
pub(crate) fn canonicalize_apply_leaf_refs(
    canon_leaves: &[usize],
    vtree: &Vtree,
    levels: &mut [TddLevel],
    ws: Option<&WeightStore>,
) {
    // The structural operand contributes leaf-side refs that never passed
    // through the canon map, and the grid carries them into the output
    // unchanged wherever the marginal side reads `One`; moving them onto the
    // canonical slot of their value class is what lets the contraction that
    // follows see the parent's `(·, Pos)` / `(·, Neg)` branches as twins.
    for &left_idx in canon_leaves {
        let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(left_idx as u32)) else { continue };
        let Some(parent) = vtree.node(VtreeIdx(left_idx as u32)).parent() else { continue };
        // A marginal parent has no leaf-side pairs left to rewrite.
        if levels[parent.idx()].is_marginal() {
            continue;
        }
        // Exact domain only: the log domain's key equality is `f64` bit
        // equality, not value equality.
        let Some(w) = ws.as_ref() else { continue };
        let Some(canon) =
            (!w.is_log()).then(|| leaf_canon_map(&leaf_column_vals(w, var)))
        else {
            continue;
        };
        if canon == [0, 1, 2] {
            continue; // no equal-valued slots — the walk would rewrite nothing
        }
        let (pl, _) = vtree.children(parent);
        let side =
            if pl.idx() == left_idx { ChildSide::Left } else { ChildSide::Right };
        canonicalize_leaf_refs_at_parent(&mut levels[parent.idx()], side, &canon);
    }
}
