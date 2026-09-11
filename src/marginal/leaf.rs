//! Summing out a single-variable vtree leaf, in both representations.

use crate::diagram::Changed;
use crate::diagram::WeightVal;
use crate::diagram::{for_each_side_ref_mut, ChildSide, LeafLabel, MarginalSide, ValueRef, Tdd, TddLevel};
use crate::diagram::{leaf_canon_map, leaf_column_vals, leaf_count};
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Sum out a single-variable vtree leaf by inlining its fixed model count
/// directly into the parent's leaf-side refs.
///
/// A leaf's marginal count is fixed by its label (One→2, Pos/Neg→1, Zero→0), so
/// it always fits `ValueRef::Inline` — no slot store is needed. The leaf's store
/// stays empty; `become_marginal(vec![], None)` only flips the `is_marginal()`
/// reader/apply signal (every reader then routes through the marginal branch and
/// decodes the inline refs). Rewriting Pos and Neg to the byte-identical
/// `Inline(1)` is the size win: the parent's `(·,x)` and `(·,¬x)` branches become
/// structurally equal, so the standard contraction / pair fusion passes merge the
/// now-twin parent nodes — we only seed `mark_contract_dirty`, no new machinery.
///
/// No-op when the parent is already marginal: the leaf was then folded into the
/// parent's store via the leaf-fixed-count fold (`read_marginal_count`'s leaf
/// branch), so there are no pairs left to rewrite.
pub(crate) fn marginalize_leaf_inline(
    eng: &crate::engine::Engine,
    tdd: &mut Tdd,
    leaf: VtreeIdx,
    vtree: &Vtree,
) {
    debug_assert!(vtree.node(leaf).is_leaf());
    if tdd.levels[leaf.idx()].is_marginal() {
        return;
    }
    // Inlining drops the leaf's Boolean structure, so a caller that still reads
    // its Pos/Neg labels — ∃-forget's cofactor walk does — must turn it off. The
    // parent's ordinary internal marginalize then sums the leaf via its fixed
    // label, exactly as before leaf-marginal; only the size win is forgone.
    if !eng.leaf_marginalize_inlines() {
        return;
    }
    // Inlining a leaf's count (bit-30 ref) is leaf-marginal's entire mechanism: Pos/Neg
    // both → Inline(1) makes the parent's branches twins for contraction. It needs
    // the inline budget to hold the max leaf count (One→2), which the full 30-bit
    // range always does.
    const _: () = assert!(crate::diagram::MARGINAL_INLINE_MAX >= 2);
    if let Some(parent_vi) = vtree.node(leaf).parent() {
        let pi = parent_vi.idx();
        if !tdd.levels[pi].is_marginal() {
            let (pl, _) = vtree.children(parent_vi);
            let leaf_is_left = pl == leaf;
            let side = if leaf_is_left { ChildSide::Left } else { ChildSide::Right };
            inline_leaf_refs_at_parent(tdd, parent_vi, side);
            tdd.invalidate(parent_vi, Changed::PAIRS);
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
/// [`leaf_canon_map`]). The weighted analogue of `inline_leaf_refs_at_parent`'s
/// twin bonus, and the one implementation of that walk — the leaf-marginal pass and
/// conjoin's leaf-marginal propagation both call it.
///
/// Value-preserving by construction: a ref is only ever moved onto a slot holding
/// the same value, so every reader (`read_marginal_weight`, the streaming child
/// view, `check::marginal`) resolves it to the number it resolved to before.
/// What changes is structure — `(·, Pos)` and `(·, Neg)` become byte-identical
/// when w⁺ = w⁻, so the parent's nodes become twins and contraction collapses
/// them. That is sound only because a marginalized leaf's variable is private (no
/// further conjunction can case-split on it), the same premise the integer arm's
/// Pos/Neg → `Inline(1)` rewrite rests on.
///
/// The column itself is never touched — this walk moves refs of one `Tdd` only,
/// which is exactly what the pin permits (see the pin invariant on
/// [`marginalize_leaf_weighted`]).
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
        // Weighted leaf sides never carry an inline (bit-30) ref: a weighted
        // `ValueRef::Inline(gidx)` indexes the `WeightStore`'s global intern table,
        // which is rebuilt at every component graft, so nothing mints one into a
        // pair list (the weighted scale refuses, and the leaf column
        // exists precisely so leaf refs stay bare slots).
        debug_assert!(
            !ValueRef::is_inline_raw(raw),
            "canonicalize_leaf_refs_at_parent: inline ref {raw} on a weighted leaf side"
        );
        if ValueRef::is_inline_raw(raw) {
            return raw;
        }
        debug_assert!(
            (raw as usize) < crate::diagram::LEAF_WIDTH,
            "canonicalize_leaf_refs_at_parent: leaf-side ref {raw} outside the \
             pinned label range"
        );
        // Out of range means the pin is already broken; leave the ref alone so
        // `check::marginal::check_leaf_columns_pinned` check #2 reports it at its own site
        // rather than this one panicking on an index.
        canon.get(raw as usize).copied().unwrap_or(raw)
    };
    for_each_side_ref_mut(plevel, side, |r| *r = to_canon(*r));
}

/// Weighted analogue of [`marginalize_leaf_inline`]: sum out a single-variable
/// vtree leaf carrying exact semiring values.
///
/// The representation deliberately differs from the integer arm. The integer path
/// rewrites the parent's leaf-side refs into self-describing `ValueRef::Inline`
/// counts (One→2, Pos/Neg→1) and leaves the leaf store empty; a weighted value
/// has no such self-describing encoding.
///
/// Instead we install a real 3-slot weighted store on the leaf level, in
/// [`LeafLabel::from_idx`] order (0 = One, 1 = Pos, 2 = Neg). A parent's leaf-side
/// refs are already bare leaf-label indices, and a bare marginal-side ref is itself
/// a slot index, so they decode as the correct `ValueRef::Slot` with no parent-ref rewrite.
/// The values come from [`WeightStore::leaf_val`] — the one place every weighted
/// leaf read resolves its bases (One = w⁺+w⁻, Pos = w⁺, Neg = w⁻) — so a parent
/// marginalized later reads exactly what it would have read with the leaf still
/// structural. A Zero leaf-side ref never reaches the slot decode: Zero is a
/// sentinel with bit 31 set (`Tdd::is_zero`; leaf levels only ever carry
/// Pos/Neg/One), and every weighted reader tests that bit before decoding.
///
/// This arm does attempt the integer arm's Pos/Neg→`Inline(1)` twin-merge bonus
/// — but only where it is a *value-preserving* rewrite. The integer arm may merge Pos
/// and Neg unconditionally because both leaf counts are 1; under weights the two
/// slots may hold different numbers, so the merge is licensed exactly when they
/// hold the same number. [`leaf_canon_map`] computes that equal-value partition of
/// the pinned column and [`canonicalize_leaf_refs_at_parent`] moves each leaf-side
/// ref onto its class's canonical (smallest) slot — Neg→Pos when w⁺ = w⁻ (the
/// common case, and the one that restores the twin cascade), Pos→One when w⁻ = 0,
/// Neg→One when w⁺ = 0, nothing at all when the three values are distinct. Only
/// refs move; the column is untouched, so the pin still holds. The walk is
/// restricted to the exact-rational domain, where `weight_key`
/// equality means value equality.
/// `mark_contract_dirty` is seeded for a structural parent, so contraction gets to
/// act on the new marginal boundary — and after canonicalization it has real work:
/// the parent's `(·, Pos)` / `(·, Neg)` branches are now byte-identical twins.
/// (Weighted pair fusion does run at leaf boundaries, but folds by sum lookup only:
/// a redex group whose summed value already sits in the pinned column collapses
/// to one pair naming that slot — `(·,Pos) + (·,Neg) = w⁺+w⁻ = the One slot`, by
/// definition and for every weight table — and a group whose sum is not in the
/// column is left exactly as it was. Minting a 4th slot at a leaf stays
/// forbidden; see `SlotValues::leaf_ref`.)
///
/// # Soundness
///
/// The column installed here is pinned (architecture invariant 11). It is
/// shared: every diagram whose store this one was merged into reads the same
/// slots, while a parent-ref rewrite reaches one `Tdd` only, so compacting,
/// reordering or appending would desynchronise every other holder. The slot
/// prune, the twin fold, weighted pair fusion and the subsumption reclaim all
/// decline at leaves for that reason, and [`check_leaf_columns_pinned`](crate::check::marginal::check_leaf_columns_pinned)
/// decides the invariant at slot-prune entry. A leaf mint is reachable only
/// from an exact-domain weighted compile, which is a production configuration.
pub(crate) fn marginalize_leaf_weighted(
    eng: &crate::engine::Engine,
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
    // Same opt-out as `marginalize_leaf_inline`: ∃-forget cofactors leaves via
    // `condition_leaf`, whose `assert_conditionable` fail-fasts on a marginal
    // leaf level. The parent's ordinary internal marginalize still sums the leaf
    // via its semiring bases.
    if !eng.leaf_marginalize_inlines() {
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
    let values: Vec<WeightVal> = leaf_column_vals(ws, var);
    // A subsumed leaf (parent already marginal) gets the same full column — the
    // pin invariant admits no second leaf state. The column is not per-`Tdd`
    // data: it is shared with every diagram this one's store reaches, and every
    // other holder of this leaf — a fresh clause diagram whose leaf level is
    // still structural, a sibling partial product — decodes its bare leaf-label
    // refs against it. A zero-slot column would make those reads panic on
    // `&values[slot]` or, through the `map_or(0, len)` width readers, silently
    // drop the leaf's whole mass. Three cached constants cost nothing to keep.
    // The only thing subsumption still changes is contract seeding: a marginal
    // parent is not a fusion boundary, so it is not marked dirty.
    if let Some(parent_vi) = parent
        && !tdd.levels[parent_vi.idx()].is_marginal() {
            // Equal-value ref canonicalization: the weighted form of the integer
            // arm's Pos/Neg → `Inline(1)` twin bonus (`marginalize_leaf_inline`).
            // Same guard as there: a marginal parent has already folded this leaf's
            // bases into its own aggregate, so there are no leaf-side pairs left to
            // rewrite. Exact domain only — `leaf_canon_map`'s `weight_key` equality
            // is value equality there, whereas a `WeightKey::Log` compares `f64`
            // bit patterns and would merge refs on a rounding coincidence.
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
            tdd.invalidate(parent_vi, Changed::PAIRS);
        }
    tdd.levels[left_idx].become_marginal_weighted(values.len() as u32);
    ws.set_level(left_idx, values);
}

/// Rewrite the leaf-side refs of every leaf that one operand made
/// weight-marginal onto that leaf's canonical slots.
///
/// It runs after the bottom-up loop because the parent's pairs are only final
/// then, and it shares the walk the marginalize pass uses, so a leaf whose
/// column holds equal values ends up with one representative rather than two
/// slots the contraction would have to recognize as twins.
pub(crate) fn canonicalize_apply_leaf_refs(
    canon_leaves: &[usize],
    vtree: &Vtree,
    levels: &mut [TddLevel],
    ws: Option<&WeightStore>,
) {
    // Equal-value leaf-ref canonicalization, the apply-side mirror of
    // `marginalize::marginalize_leaf_weighted`'s pass and sharing its one walk.
    // Runs after the apply's bottom-up loop, not at the flag site inside it:
    // the parent level's pairs are emitted by that loop, so this is the first
    // point at which they are final.
    //
    // Scope is the leaves the apply recorded — flagged weight-marginal on one
    // operand's authority. The structural operand contributes leaf-side refs that
    // never passed through the canon map, and `CONJOIN_GRID` carries them into the
    // output unchanged wherever the marginal side reads `One`. Rewriting them onto
    // the canonical slot of their value class is value-preserving (same column
    // entry) and is what lets the contraction that follows this apply see the
    // parent's `(·, Pos)` / `(·, Neg)` branches as twins.
    for &left_idx in canon_leaves {
        let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(left_idx as u32)) else { continue };
        let Some(parent) = vtree.node(VtreeIdx(left_idx as u32)).parent() else { continue };
        // A marginal parent folded the leaf's bases into its own aggregate — no
        // leaf-side pairs remain to rewrite (same guard as the marginalize pass).
        if levels[parent.idx()].is_marginal() {
            continue;
        }
        // Exact domain only: `WeightKey::Log` compares `f64` bit patterns, so
        // "equal" there is representation identity, not value identity.
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
