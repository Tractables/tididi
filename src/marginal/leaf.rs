//! Summing out a single-variable vtree leaf, in both representations.

use crate::diagram::Changed;
use crate::diagram::WeightVal;
use crate::diagram::{LeafLabel, MarginalSide, ValueRef, Tdd, TddLevel};
use crate::diagram::WeightStore;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};
use crate::apply::conjoin::marginal_plan::Sides;

/// Sum out a single-variable vtree LEAF by inlining its fixed model count
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
    // the inline budget to hold the max leaf count (One→2). In production
    // `marginal_inline_max` is the full 30-bit range so this always holds; only a
    // test-lowered budget (<2) fails it, and there we leave the leaf structural
    // (exact, no size win) rather than synthesize a slot store the bare-ref decode
    // path doesn't integrate correctly.
    if crate::diagram::marginal_inline_max() < 2 {
        return;
    }
    if let Some(parent_vi) = vtree.node(leaf).parent() {
        let pi = parent_vi.idx();
        if !tdd.levels[pi].is_marginal() {
            let (pl, _) = vtree.children(parent_vi);
            let leaf_is_left = pl == leaf;
            inline_leaf_refs_at_parent(tdd, parent_vi, leaf_is_left);
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
/// `remap_parent_refs_pretag`, but maps leaf labels to inline counts instead of
/// remapping slot indices. Bit 30 (the inline tag) is disjoint from
/// `LEAF_BIT/MULTI_BIT` (bit 31), so the rewritten refs keep their inline/multi
/// node encoding.
fn inline_leaf_refs_at_parent(tdd: &mut Tdd, parent_v: VtreeIdx, leaf_is_left: bool) {
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
            0 => 2,        // One
            1 | 2 => 1,    // Pos / Neg
            3 => 0,        // Zero (sentinel index — defensive; not normally stored)
            other => panic!("inline_leaf_refs_at_parent: unexpected leaf-side ref {other}"),
        };
        ValueRef::inline_raw(count).expect("leaf count 0/1/2 always fits inline")
    };
    let plevel = &mut tdd.levels[parent_v.idx()];
    for node_idx in 0..plevel.nodes.len() {
        if plevel.nodes[node_idx].is_inline() {
            let node = &mut plevel.nodes[node_idx];
            if leaf_is_left {
                node.a = to_inline(node.a);
            } else {
                node.b = to_inline(node.b);
            }
        } else if plevel.nodes[node_idx].is_multi() {
            let pairs = plevel.pairs_mut(node_idx);
            for p in pairs.iter_mut() {
                if leaf_is_left {
                    p.left = crate::diagram::NodeIdx(to_inline(p.left.idx() as u32));
                } else {
                    p.right = crate::diagram::NodeIdx(to_inline(p.right.idx() as u32));
                }
            }
        }
    }
}

/// The pinned column of a weight-marginal vtree LEAF: [`WeightStore::leaf_val`]
/// for One/Pos/Neg in [`LeafLabel::from_idx`] slot order (0 = One = w⁺+w⁻,
/// 1 = Pos = w⁺, 2 = Neg = w⁻).
///
/// THE definition of that column. `marginalize_leaf_weighted` installs what this
/// returns; the apply-side canon pass, the streaming child view and
/// [`debug_check_leaf_columns_pinned`] all re-derive it here rather than each
/// spelling the triple out, so "what the column holds" cannot drift between them.
pub(crate) fn leaf_column_vals(ws: &WeightStore, var: VarId) -> Vec<WeightVal> {
    (0..crate::diagram::LEAF_WIDTH)
        .map(|i| ws.leaf_val(var, LeafLabel::from_idx(i)))
        .collect()
}

/// Canonical-slot map for a weight-marginal leaf's pinned column: `canon[r]` is
/// the SMALLEST slot whose value equals `values[r]`, so every ref that names a
/// value names it by one agreed slot.
///
/// Equality is `weight_key` — the crate's one value-identity choke point, and in
/// the exact-rational domain it is exact equality (`ExactSmall`/`Exact` partition
/// the value space, so equal numbers always produce equal keys). Callers must
/// restrict this to the exact domain: a `WeightKey::Log` compares `f64` bit patterns,
/// which is a *representation* identity, not the value identity this map claims.
///
/// The shapes it can take, given `values = [w⁺+w⁻, w⁺, w⁻]`:
///   * `w⁺ = w⁻` → `[0, 1, 1]` (Neg → Pos) — the common case, and the one that
///     recovers the integer arm's twin bonus;
///   * `w⁻ = 0`  → `[0, 0, 2]` (Pos → One);
///   * `w⁺ = 0`  → `[0, 1, 0]` (Neg → One);
///   * otherwise → the identity `[0, 1, 2]`, and the caller skips the walk.
///
/// (`w⁺ = w⁻ = 0` collapses all three onto slot 0, which the same rule produces.)
pub(crate) fn leaf_canon_map(values: &[WeightVal]) -> [u32; 3] {
    use crate::diagram::semiring::weight_key;
    debug_assert_eq!(
        values.len(),
        crate::diagram::LEAF_WIDTH,
        "leaf_canon_map: not a pinned leaf column"
    );
    let keys = [weight_key(&values[0]), weight_key(&values[1]), weight_key(&values[2])];
    let mut canon = [0u32, 1, 2];
    // `s < r` and equality is transitive, so the first earlier slot carrying
    // `values[r]` IS the minimum one (an even earlier match would have matched `s`
    // too, and `s` was taken as the first).
    for r in 1..crate::diagram::LEAF_WIDTH {
        for s in 0..r {
            if keys[s] == keys[r] {
                canon[r] = s as u32;
                break;
            }
        }
    }
    canon
}

/// The pinned column's slot for a VALUE: the SMALLEST slot `s < LEAF_WIDTH` of
/// level `level_idx`'s column whose value equals `want`, or `None` when the level
/// carries no column or no slot holds that value.
///
/// The one value-search over a pinned leaf column. A leaf column can never grow
/// (THE PIN INVARIANT on [`marginalize_leaf_weighted`]), so "is this value
/// representable at this leaf?" IS this lookup — which is what both mint-free
/// leaf folds ask: `minimize::contract::duplicate_pair_resolve::scale_weight_leaf_by_lookup`
/// (is `k·slot` in the column?) and
/// `minimize::contract::pair_fusion::resolve_leaf_fusion_refs_by_lookup` (is a
/// pair fusion group's SUM in the column?), plus the census that sizes the second.
///
/// ASCENDING order is a SOUNDNESS requirement, not a style choice. The slot
/// returned here becomes a leaf-side ref, and every leaf-side ref must name the
/// CANONICAL (smallest) slot of its value class ([`leaf_canon_map`]) or pin check
/// #4 in [`debug_check_leaf_columns_pinned`] fires — scanning from 0 and taking
/// the first hit is exactly that minimum. The scan is also bounded at
/// `LEAF_WIDTH` rather than the slice length, so a column that somehow grew past
/// the pin can never hand back a ref no remap window is sized for.
///
/// Equality is `weight_key`, so callers must restrict this to the exact domain for the
/// same reason [`leaf_canon_map`] does: a `WeightKey::Log` compares `f64` bit
/// patterns, and a "hit" there would be a rounding coincidence rather than a
/// value identity.
pub(crate) fn find_leaf_slot_by_value(
    ws: &WeightStore,
    level_idx: usize,
    want: &WeightVal,
) -> Option<u32> {
    use crate::diagram::semiring::weight_key;
    let col = ws.level(level_idx)?;
    let want = weight_key(want);
    col.iter()
        .take(crate::diagram::LEAF_WIDTH)
        .position(|v| weight_key(v) == want)
        .map(|s| s as u32)
}

/// Rewrite every leaf-side ref of `plevel`'s nodes onto the canonical slot of an
/// equal-value class in a weight-marginal LEAF's pinned column (`canon` from
/// [`leaf_canon_map`]). The weighted analogue of `inline_leaf_refs_at_parent`'s
/// twin bonus, and the one implementation of that walk — the leaf-marginal pass and
/// conjoin's leaf-marginal propagation both call it.
///
/// Value-preserving by construction: a ref is only ever moved onto a slot holding
/// the same value, so every reader (`read_marginal_weight`, the streaming child
/// view, `check::marginal`) resolves it to the number it resolved to before.
/// What changes is structure — `(·, Pos)` and `(·, Neg)` become byte-identical
/// when w⁺ = w⁻, so the parent's nodes become twins and contraction collapses
/// them. That is sound only because a marginalized leaf's variable is PRIVATE (no
/// further conjunction can case-split on it), the same premise the integer arm's
/// Pos/Neg → `Inline(1)` rewrite rests on.
///
/// The column itself is never touched — this walk moves refs of one `Tdd` only,
/// which is exactly what the pin permits (see THE PIN INVARIANT on
/// [`marginalize_leaf_weighted`]).
pub(crate) fn canonicalize_leaf_refs_at_parent(
    plevel: &mut TddLevel,
    leaf_is_left: bool,
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
        // pair list (`duplicate_pair_resolve::scale_weight_ref` refuses, and the leaf column
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
        // `debug_check_leaf_columns_pinned` check #2 reports it at its own site
        // rather than this one panicking on an index.
        canon.get(raw as usize).copied().unwrap_or(raw)
    };
    for node_idx in 0..plevel.nodes.len() {
        if plevel.nodes[node_idx].is_inline() {
            let node = &mut plevel.nodes[node_idx];
            if leaf_is_left {
                node.a = to_canon(node.a);
            } else {
                node.b = to_canon(node.b);
            }
        } else if plevel.nodes[node_idx].is_multi() {
            let pairs = plevel.pairs_mut(node_idx);
            for p in pairs.iter_mut() {
                if leaf_is_left {
                    p.left = crate::diagram::NodeIdx(to_canon(p.left.idx() as u32));
                } else {
                    p.right = crate::diagram::NodeIdx(to_canon(p.right.idx() as u32));
                }
            }
        }
    }
}

/// Weighted analogue of [`marginalize_leaf_inline`]: sum out a single-variable
/// vtree LEAF carrying exact semiring values.
///
/// The representation deliberately differs from the integer arm. The integer path
/// rewrites the parent's leaf-side refs into self-describing `ValueRef::Inline`
/// counts (One→2, Pos/Neg→1) and leaves the leaf store empty; a weighted value
/// has no such self-describing encoding.
///
/// Instead we install a real 3-slot weighted store on the leaf level, in
/// [`LeafLabel::from_idx`] order (0 = One, 1 = Pos, 2 = Neg). A parent's leaf-side
/// refs are already bare leaf-LABEL indices, and a bare marginal-side ref IS its slot
/// index, so they decode as the correct `ValueRef::Slot` with no parent-ref rewrite.
/// The values come from [`WeightStore::leaf_val`] — the one place every weighted
/// leaf read resolves its bases (One = w⁺+w⁻, Pos = w⁺, Neg = w⁻) — so a parent
/// marginalized later reads exactly what it would have read with the leaf still
/// structural. A Zero leaf-side ref never reaches the slot decode: Zero is a
/// sentinel with bit 31 set (`Tdd::is_zero`; leaf levels only ever carry
/// Pos/Neg/One), and every weighted reader tests that bit before decoding.
///
/// The integer arm's Pos/Neg→`Inline(1)` twin-merge bonus IS attempted here — but
/// only where it is a *value-preserving* rewrite. The integer arm may merge Pos
/// and Neg unconditionally because both leaf counts are 1; under weights the two
/// slots may hold different numbers, so the merge is licensed exactly when they
/// hold the same number. [`leaf_canon_map`] computes that equal-value partition of
/// the pinned column and [`canonicalize_leaf_refs_at_parent`] moves each leaf-side
/// ref onto its class's canonical (smallest) slot — Neg→Pos when w⁺ = w⁻ (the
/// common case, and the one that restores the twin cascade), Pos→One when w⁻ = 0,
/// Neg→One when w⁺ = 0, nothing at all when the three values are distinct. Only
/// refs move; the column is untouched, so the pin below still holds. The walk is
/// restricted to the exact-rational domain, where `weight_key`
/// equality IS value equality.
/// `mark_contract_dirty` is seeded for a STRUCTURAL parent, so contraction gets to
/// act on the new marginal boundary — and after canonicalization it has real work:
/// the parent's `(·, Pos)` / `(·, Neg)` branches are now byte-identical twins.
/// (Weighted pair fusion DOES run at leaf boundaries, but folds by SUM-LOOKUP only:
/// a redex group whose summed value already sits in the pinned column collapses
/// to one pair naming that slot — `(·,Pos) + (·,Neg) = w⁺+w⁻ = the One slot`, by
/// definition and for every weight table — and a group whose sum is not in the
/// column is left exactly as it was. Minting a 4th slot at a leaf stays
/// forbidden; see `pair_fusion::resolve_leaf_fusion_refs_by_lookup`.)
///
/// # THE PIN INVARIANT
///
/// **A weight-marginal LEAF level's column is an immutable, label-ordered,
/// exactly-`LEAF_WIDTH` cache of [`WeightStore::leaf_val`]. No pass may compact,
/// erase, reorder, or append to it, ever.** The column is SHARED — every diagram
/// whose store this one was merged into reads the same slots — while a parent-ref
/// rewrite can only reach one `Tdd`, so any mutation desynchronises every other
/// holder — including fresh
/// clause diagrams whose leaf level is still structural and hold genuine leaf-LABEL
/// refs. Enforced at:
///   * `minimize::slot_prune::prune_marginal_slots_generic` — both walks skip
///     weight-marginal leaves (no compaction, no dead-store clear);
///   * `minimize::contract::duplicate_pair_resolve::try_scale_child` — a G twin-fold into a
///     weight-marginal leaf LOOKS the scaled value up among the column's own
///     three slots and takes that slot if it is there, declining otherwise. It
///     never mints, and never writes the column;
///   * `minimize::contract::pair_fusion::resolve_leaf_fusion_refs_by_lookup` — the
///     weighted arm folds a LEAF boundary by SUM-LOOKUP only: the fusion group's
///     summed value is folded onto the column slot that already holds it (found
///     via [`find_leaf_slot_by_value`], so the ref is canonical), and the plan is
///     DROPPED when no slot holds it. It never mints, never writes the column,
///     and never bumps the level's width;
///   * [`free_subsumed_marginal_children`] — leaves are exempt from the
///     subsumed-data reclaim;
///   * [`read_marginal_weight`] — leaf refs resolve by LABEL, never through the
///     column;
///   * `conjoin`'s leaf-marginal propagation — flags the output level `LEAF_WIDTH`
///     directly rather than reading the column's length;
///   * [`canonicalize_leaf_refs_at_parent`] — the equal-value ref rewrite (this
///     function, and conjoin's leaf-marginal propagation) moves REFS of one `Tdd`
///     between slots that already hold the same value; it reads the column and
///     writes nothing to it.
///
/// Checked centrally by [`debug_check_leaf_columns_pinned`] at slot-prune entry.
///
/// WHERE THE EXACT REGIME LIVES. Weighted pair fusion — the growth-direction
/// breaker — is inactive whenever the store is in the bounded LOG domain, so a
/// leaf mint is reachable only from an exact-domain weighted compile. Do not
/// read "the log domain is fine" as "the bug is unreachable" — exact-domain
/// compiles are production.
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
    // SUBSUMED LEAF (parent already marginal) gets the same full column — the pin
    // invariant admits no second leaf state. An earlier revision installed ZERO
    // slots here, on the theory that the parent's aggregate already folded the
    // bases in so a column would be dead per-`Tdd` data. It is not per-`Tdd` data:
    // the column is SHARED with every diagram this one's store reaches, and every
    // OTHER holder of this leaf — a fresh clause diagram
    // whose leaf level is still structural, a sibling partial product — decodes its
    // bare leaf-LABEL refs against it. A zero-slot column made those reads panic on
    // `&values[slot]` or, through the `map_or(0, len)` width readers, silently drop
    // the leaf's whole mass. Three cached constants cost nothing to keep.
    // The only thing subsumption still changes is contract seeding: a marginal
    // parent is not a fusion boundary, so it is not marked dirty.
    if let Some(parent_vi) = parent
        && !tdd.levels[parent_vi.idx()].is_marginal() {
            // EQUAL-VALUE REF CANONICALIZATION — the weighted form of the integer
            // arm's Pos/Neg → `Inline(1)` twin bonus (`marginalize_leaf_inline`).
            // Same guard as there: a MARGINAL parent has already folded this leaf's
            // bases into its own aggregate, so there are no leaf-side pairs left to
            // rewrite. Exact domain only — `leaf_canon_map`'s `weight_key` equality
            // is value equality there, whereas a `WeightKey::Log` compares `f64`
            // bit patterns and would merge refs on a rounding coincidence.
            if !ws.is_log() {
                let canon = leaf_canon_map(&values);
                if canon != [0, 1, 2] {
                    let (pl, _) = vtree.children(parent_vi);
                    canonicalize_leaf_refs_at_parent(
                        &mut tdd.levels[parent_vi.idx()],
                        pl == leaf,
                        &canon,
                    );
                }
            }
            tdd.invalidate(parent_vi, Changed::PAIRS);
        }
    tdd.levels[left_idx].become_marginal_weighted(values.len() as u32);
    ws.set_level(left_idx, values);
}

/// PIN-INVARIANT CHECK (debug builds only): every weight-marginal vtree LEAF must
/// advertise exactly `LEAF_WIDTH` slots, and — when its column is
/// installed — that column must equal the `leaf_val` triple in `LeafLabel` order.
///
/// This is the one invariant that makes bare leaf-LABEL refs and `ValueRef::Slot`
/// refs interchangeable at a leaf, which is what lets `marginalize_leaf_weighted`
/// flip a leaf marginal without rewriting a single parent ref. Every pass that
/// could break it (slot-prune compaction, duplicate resolution twin-fold minting, weighted
/// pair fusion allocation, subsumption reclaim) declines to touch leaves; this check
/// is placed on a hot, frequently-run path (slot-prune entry) so a regression in
/// any of them surfaces immediately instead of as a silently low weighted count.
///
/// Four things are checked, in the order a breakage shows up:
///   1. the level advertises `LEAF_WIDTH` slots (catches a `weight_width`
///      bump — how weighted pair fusion's `allocate_fusion_slots_weighted` records a
///      minted slot);
///   2. No parent ref into the leaf names a slot ≥ `LEAF_WIDTH` (catches a minted
///      REF that outlived the width, and is the check that fails closest to the
///      real damage: a `Slot(3)` ref is decoded by every label-first reader —
///      `query::count`, `query::sat`, `validate`, `duplicate_pair_resolve` — as
///      `LeafLabel::from_idx(3)`, the never-satisfied ZERO sentinel, so the
///      models under it vanish with no error anywhere, and `prune_unreachable`
///      indexes the NEIGHBOURING level's remap window with it);
///   3. the installed column equals the `leaf_val` triple in label order
///      (catches compaction / erasure / reordering);
///   4. Every bare leaf-side ref is the CANONICAL slot of its value class
///      ([`leaf_canon_map`]) — catches a site that CREATES a leaf-side ref and
///      skips the canon pass. That is a size regression rather than a wrong
///      count (a non-canonical ref still resolves to the right value), so it has
///      no other symptom: without this check the twin cascade would just quietly
///      stop firing on the affected leaves. Exact domain only, and only once the
///      column is installed — the canon partition is undefined
///      otherwise.
///
/// No-op in release and whenever the diagram carries no weight store.
#[cfg(debug_assertions)]
pub(crate) fn debug_check_leaf_columns_pinned(tdd: &Tdd) {
    use crate::diagram::semiring::weight_key;
    use crate::diagram::ChildSide;
    use crate::reduce::slots::{RefSlotScratch, referenced_marginal_slots};
    let Some(ws) = tdd.weights.as_ref() else {
        return;
    };
    let mut scratch = RefSlotScratch::default();
    {
        for i in 0..tdd.levels.len() {
            let VtreeNode::Leaf { var, .. } = *tdd.vtree.node(VtreeIdx(i as u32)) else { continue };
            if !tdd.levels[i].is_weight_marginal() {
                continue;
            }
            debug_assert_eq!(
                tdd.levels[i].width(),
                crate::diagram::LEAF_WIDTH,
                "pin invariant: weight-marginal leaf level {i} must advertise \
                 LEAF_WIDTH slots"
            );
            // No parent ref may name a slot outside the pinned label range.
            if let Some(parent) = tdd.vtree.node(VtreeIdx(i as u32)).parent() {
                let side = match tdd.vtree.node(parent) {
                    VtreeNode::Internal { left, .. } if left.idx() == i => ChildSide::Left,
                    _ => ChildSide::Right,
                };
                let refs = referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut scratch);
                debug_assert!(
                    refs.last().is_none_or(|&s| (s as usize) < crate::diagram::LEAF_WIDTH),
                    "pin invariant: weight-marginal leaf level {i} is referenced at slot \
                     {:?} — outside the label range, so every label-first reader decodes \
                     it as the ZERO sentinel and drops that branch's mass",
                    refs.last()
                );
                // #4 — canonicality. Every surviving ref must already name the
                // smallest slot of its value class; anything else means some site
                // minted a leaf-side ref without running
                // `canonicalize_leaf_refs_at_parent`.
                if !ws.is_log()
                    && let Some(col) = ws.level(i) {
                        let canon = leaf_canon_map(col);
                        for &s in refs {
                            // Out-of-range refs are check #2's report, not ours.
                            debug_assert!(
                                (s as usize) >= crate::diagram::LEAF_WIDTH
                                    || canon[s as usize] == s,
                                "pin invariant: weight-marginal leaf level {i} is referenced \
                                 at NON-CANONICAL slot {s} (canonical slot for that value is \
                                 {}) — a leaf-side ref was created without the equal-value \
                                 canon pass, so the twin cascade cannot fire there",
                                canon[s as usize]
                            );
                        }
                    }
            }
            let Some(col) = ws.level(i) else { continue };
            debug_assert_eq!(
                col.len(),
                crate::diagram::LEAF_WIDTH,
                "pin invariant: weight-marginal leaf level {i} column was \
                 compacted/erased/appended to"
            );
            for (k, slot_val) in col.iter().enumerate() {
                debug_assert!(
                    weight_key(slot_val) == weight_key(&ws.leaf_val(var, LeafLabel::from_idx(k))),
                    "pin invariant: weight-marginal leaf level {i} column slot {k} \
                     is not the label-ordered leaf_val cache"
                );
            }
        }
    }
}

#[cfg(not(debug_assertions))]
#[inline(always)]
pub(crate) fn debug_check_leaf_columns_pinned(_tdd: &Tdd) {}

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
    // EQUAL-VALUE LEAF-REF CANONICALIZATION, apply-side mirror of
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
        canonicalize_leaf_refs_at_parent(
            &mut levels[parent.idx()],
            pl.idx() == left_idx,
            &canon,
        );
    }
}

/// Mark the output's marginal vtree LEAVES, which the bottom-up loop never
/// visits as a level of its own, and collect the weight-marginal leaves whose
/// refs still have to be canonicalized once the output's pairs are final.
///
/// A marginalized leaf variable is private to one operand, so the other is the
/// identity there and the parent's marginal-child dispatch carries the refs
/// through. A restricted apply skips the sweep: its output levels are merged
/// back into the accumulator's, which already carries its own marginal leaves.
// The debug assertion enumerates the three legal marginal-leaf shapes; a
// factored form hides which case is which.
#[allow(clippy::nonminimal_bool)]
pub(crate) fn seed_output_leaves(
    f: &Tdd,
    g: &Tdd,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    identity: Sides<&[bool]>,
    ws: Option<&WeightStore>,
) -> Vec<usize> {
    let (left_identity, right_identity) = (identity.left, identity.right);
    // Leaf marginalization: a marginal vtree LEAF is never visited as a `t` by
    // the bottom-up loop, so — unlike a marginal internal child — its OUTPUT level
    // is never marked marginal. Seed it here from the operands. A marginalized
    // leaf var is PRIVATE (summed only once its every clause is compiled), so the
    // other operand is identity at that leaf; the parent's marginal-child dispatch
    // then routes Route A and the passthrough path carries the carrier's inline
    // `ValueRef` refs through verbatim. The output store stays empty (all leaf
    // counts are inline at the parent).
    // In weighted mode the leaf's counts are not inline at the parent: the
    // weighted leaf-marginal installs a real per-slot column in the (vtree-indexed)
    // `WeightStore` and leaves the parent's bare leaf-label refs to
    // decode as `ValueRef::Slot`. That column is PINNED — immutable, label-ordered,
    // exactly `LEAF_WIDTH` slots, never compacted / erased / appended to by any
    // pass — so the output level reports `LEAF_WIDTH` and this only re-flags it.
    //
    // `canon_leaves` collects the leaves flagged weight-marginal on one operand's
    // authority: the other operand was structural there, so its genuine leaf-LABEL
    // refs flow through `CONJOIN_GRID` into the output and may not be canonical
    // (`marginalize::leaf_canon_map`). They are canonicalized once the output's
    // pair lists are final — the bottom-up loop BELOW emits them, so there is
    // nothing to rewrite here yet. When both operands are weight-marginal both
    // sides are already canonical and the grid is closed over each canon class
    // (`{One,Pos}`, `{One,Neg}`, `{One}` are each closed under ∧), so nothing is
    // recorded.
    let mut canon_leaves: Vec<usize> = Vec::new();
    for (leaf, _) in vtree.leaf_bottomup() {
        let left_idx = leaf.idx();
        let left_m = f.levels[left_idx].is_marginal();
        let right_m = g.levels[left_idx].is_marginal();
        if left_m || right_m {
            debug_assert!(
                (left_m && right_m) || (left_m && right_identity[left_idx]) || (right_m && left_identity[left_idx]),
                "marginal leaf {left_idx} conjoined with a non-identity operand \
                 (var not private?): left_m={left_m} right_m={right_m} \
                 left_id={} right_id={}",
                left_identity[left_idx], right_identity[left_idx],
            );
            let w1 = f.levels[left_idx].is_weight_marginal();
            let w2 = g.levels[left_idx].is_weight_marginal();
            if w1 || w2 {
                // PIN INVARIANT (`marginalize::marginalize_leaf_weighted`): a
                // weight-marginal LEAF's column is an IMMUTABLE, label-ordered,
                // exactly-`LEAF_WIDTH` cache of `WeightStore::leaf_val`. No pass
                // compacts, erases, reorders or appends to it — slot-prune,
                // duplicate resolution's twin fold, weighted pair fusion and the subsumption
                // reclaim all decline at leaves — so the output level's slot count
                // is `LEAF_WIDTH`, full stop.
                //
                // Flagging it directly (rather than reading the column's length)
                // is what makes this robust: a `map_or(0, len)` read reports width
                // 0 whenever the global column happens not to be installed for
                // this vtree index, and a width-0 weight-marginal leaf is silently
                // skipped by `marginalize_batch_weighted` and read as an empty
                // column by the streaming child view — dropping the leaf's entire
                // mass with no error anywhere.
                let leaf_slots = crate::diagram::LEAF_WIDTH;
                debug_assert!(
                    ws.is_none_or(|w| w.level(left_idx).is_none_or(|v| v.len() == leaf_slots)),
                    "weight-marginal leaf {left_idx}: WeightStore column is not the \
                     pinned {leaf_slots}-slot leaf_val cache",
                );
                debug_assert!(
                    (!w1 || f.levels[left_idx].width() == leaf_slots)
                        && (!w2 || g.levels[left_idx].width() == leaf_slots),
                    "weight-marginal leaf {left_idx}: operand slot carriers \
                     (f={}, g={}) disagree with LEAF_WIDTH ({leaf_slots})",
                    f.levels[left_idx].width(), g.levels[left_idx].width(),
                );
                levels[left_idx].become_marginal_weighted(leaf_slots as u32);
                if w1 != w2 {
                    canon_leaves.push(left_idx);
                }
            } else {
                levels[left_idx].become_marginal(Vec::new(), None);
            }
        }
    }
    canon_leaves
}
