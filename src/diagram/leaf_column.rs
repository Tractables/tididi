//! The values of a vtree leaf's marginal level, in both domains.
//!
//! A leaf has three label slots — One, Pos, Neg — and a marginalized leaf level
//! holds one value per slot. The integer column is a constant (`[2, 1, 1]`); the
//! weighted one is read from the [`WeightStore`]. Both are pinned: no pass
//! compacts, reorders or appends to them, so every reader re-derives the column
//! here instead of spelling the triple out, and what the column holds cannot
//! drift between them.

use crate::diagram::{LeafLabel, WeightStore, WeightVal};
use crate::vtree::VarId;

/// The pinned column of an integer-marginal vtree leaf: the model count of
/// One/Pos/Neg in [`LeafLabel::from_idx`] slot order (0 = One = 2, 1 = Pos = 1,
/// 2 = Neg = 1).
///
/// This is the sole definition of that column, and the integer twin of
/// [`leaf_column_vals`].
/// A `static` so a streaming leaf view can borrow it rather than mint a fresh
/// `Vec` per level.
pub(crate) static LEAF_COUNTS: [u128; crate::diagram::LEAF_WIDTH] = [2, 1, 1];

/// The model count of a leaf label. [`LeafLabel::Zero`] is a sentinel, not a
/// column slot, and counts 0.
#[inline]
pub(crate) fn leaf_count(label: LeafLabel) -> u128 {
    match label {
        LeafLabel::Zero => 0,
        _ => LEAF_COUNTS[label as usize],
    }
}

/// The pinned column of a weight-marginal vtree leaf: [`WeightStore::leaf_val`]
/// for One/Pos/Neg in [`LeafLabel::from_idx`] slot order (0 = One = w⁺+w⁻,
/// 1 = Pos = w⁺, 2 = Neg = w⁻).
///
/// This is the sole definition of that column. `marginalize_leaf_weighted` installs what this
/// returns; the apply-side canon pass, the streaming child view and
/// [`crate::check::marginal::check_leaf_columns_pinned`] all re-derive it here rather than each
/// spelling the triple out, so "what the column holds" cannot drift between them.
pub(crate) fn leaf_column_vals(ws: &WeightStore, var: VarId) -> Vec<WeightVal> {
    (0..crate::diagram::LEAF_WIDTH)
        .map(|i| ws.leaf_val(var, LeafLabel::from_idx(i)))
        .collect()
}

/// Canonical-slot map for a weight-marginal leaf's pinned column: `canon[r]` is
/// the smallest slot whose value equals `values[r]`, so every ref that names a
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
    // `values[r]` can only be the minimum one (an even earlier match would have
    // matched `s` too, and `s` was taken as the first).
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

/// The pinned column's slot for a value: the smallest slot `s < LEAF_WIDTH` of
/// level `level_idx`'s column whose value equals `want`, or `None` when the level
/// carries no column or no slot holds that value.
///
/// The one value-search over a pinned leaf column. A leaf column can never grow
/// (the pin invariant on [`crate::marginal::marginalize_leaf_weighted`]), so "is this value
/// representable at this leaf?" reduces to nothing but this lookup — which is
/// what both mint-free leaf folds ask:
/// `reduce::contract::duplicate_pair_resolve::scale_weight_leaf_by_lookup`
/// (is `k·slot` in the column?) and
/// `value::slots::SlotValues::leaf_ref` (is a
/// pair fusion group's sum in the column?), plus the census that sizes the second.
///
/// Scanning in ascending order is required for soundness, not a matter of style.
/// The slot returned here becomes a leaf-side ref, and every leaf-side ref must
/// name the canonical (smallest) slot of its value class ([`leaf_canon_map`]) or pin check
/// #4 in [`crate::check::marginal::check_leaf_columns_pinned`] fires — scanning from 0 and taking
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
