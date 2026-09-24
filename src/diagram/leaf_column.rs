//! The values of a vtree leaf's marginal level, in both domains.
//!
//! A leaf has three label slots — One, Pos, Neg — and a marginalized leaf level
//! holds one value per slot. The integer column is a constant (`[2, 1, 1]`); the
//! weighted one is read from the [`WeightStore`]. Both are pinned: no pass
//! compacts, reorders or appends to them
//! ([`check_leaf_columns_pinned`](crate::test_helpers::check::marginal::check_leaf_columns_pinned)).

use crate::diagram::{LeafLabel, WeightStore, WeightValue};
use crate::vtree::VarId;

/// The pinned column of an integer-marginal vtree leaf: the model count of
/// One/Pos/Neg in [`LeafLabel::from_idx`] slot order (0 = One = 2, 1 = Pos = 1,
/// 2 = Neg = 1).
///
/// The integer twin of [`leaf_column_vals`]; a `static` so a leaf view can
/// borrow it.
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
pub(crate) fn leaf_column_vals(ws: &WeightStore, var: VarId) -> Vec<WeightValue> {
    (0..crate::diagram::LEAF_WIDTH)
        .map(|i| ws.leaf_val(var, LeafLabel::from_idx(i)))
        .collect()
}

/// Canonical-slot map for a weight-marginal leaf's pinned column: `canon[r]` is
/// the smallest slot whose value equals `values[r]`, so every ref that names a
/// value names it by one agreed slot.
///
/// Equality is `weight_key`, exact in the rational domain. Callers must
/// restrict this to the exact domain: a `WeightKey::Log` compares `f64` bit
/// patterns, a representation identity rather than a value identity.
///
/// The shapes it can take, given `values = [w⁺+w⁻, w⁺, w⁻]`:
///   * `w⁺ = w⁻` → `[0, 1, 1]` (Neg → Pos) — the common case, and the one that
///     recovers the integer arm's twin bonus;
///   * `w⁻ = 0`  → `[0, 0, 2]` (Pos → One);
///   * `w⁺ = 0`  → `[0, 1, 0]` (Neg → One);
///   * otherwise → the identity `[0, 1, 2]`, and the caller skips the walk.
///
/// (`w⁺ = w⁻ = 0` collapses all three onto slot 0, which the same rule produces.)
pub(crate) fn leaf_canon_map(values: &[WeightValue]) -> [u32; 3] {
    use crate::diagram::semiring::same_value;
    debug_assert_eq!(
        values.len(),
        crate::diagram::LEAF_WIDTH,
        "leaf_canon_map: not a pinned leaf column"
    );
    let mut canon = [0u32, 1, 2];
    // `s < r` and equality is transitive, so the first earlier slot carrying
    // `values[r]` can only be the minimum one (an even earlier match would have
    // matched `s` too, and `s` was taken as the first).
    for r in 1..crate::diagram::LEAF_WIDTH {
        for s in 0..r {
            if same_value(&values[s], &values[r]) {
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
/// # Soundness
///
/// The result becomes a leaf-side ref, which must name the smallest slot of
/// its value class ([`leaf_canon_map`], checked by
/// [`check_leaf_columns_pinned`](crate::test_helpers::check::marginal::check_leaf_columns_pinned));
/// scanning from 0 and taking the first hit is that minimum. Equality is
/// that of `weight_key`, so callers must restrict this to the exact domain,
/// as for [`leaf_canon_map`].
pub(crate) fn find_leaf_slot_by_value(
    ws: &WeightStore,
    level_idx: usize,
    want: &WeightValue,
) -> Option<u32> {
    use crate::diagram::semiring::same_value;
    let col = ws.level(level_idx)?;
    col.iter()
        .take(crate::diagram::LEAF_WIDTH)
        .position(|v| same_value(v, want))
        .map(|s| s as u32)
}
