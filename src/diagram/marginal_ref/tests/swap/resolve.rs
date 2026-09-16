use super::*;
use crate::engine::Engine;
use crate::test_helpers::pair;

fn marginal_level(counts: Vec<u128>) -> TddLevel {
    let mut l = TddLevel::new();
    l.set_counts_state(counts, None);
    l
}

/// A count too wide to sit in a ref, so every fixture that wants a slot rather
/// than an inline count states one of these rather than lowering a threshold.
const WIDE: u128 = 1u128 << 40;
const WIDER: u128 = (1u128 << 40) + 7;

/// Left-side remap over a diverged store: a small count inlines, a wide count
/// already present in the dst store dedups onto the existing slot, an absent
/// wide count re-mints a fresh dst slot; inline refs and ZERO sentinels pass
/// through untouched. Covers both parent ref homes: the pairs arena
/// (multi-pair node) and the inline node encoding (`node.a`).
#[test]
fn resolve_left_inline_dedup_mint_passthrough() {
    let eng = Engine::new();
    let src = marginal_level(vec![3, WIDE, WIDER]);
    // dst store: src slot 1's count already present (at a different
    // index), src slot 2's count absent.
    let mut levels = vec![TddLevel::new(), marginal_level(vec![WIDE])];
    levels[0].push_internal_node(&[
        pair(ValueRef::slot_raw(0), 0), // → inline (3 fits a ref)
        pair(ValueRef::slot_raw(1), 0), // → dedup onto dst slot 0
        pair(ValueRef::slot_raw(2), 0), // → re-mint dst slot 1
        pair(ValueRef::inline_raw(2).unwrap(), 0), // inline: untouched
        pair((1 << 31) | 5, 0),        // ZERO sentinel: untouched
    ]);
    levels[0].push_internal_node(&[pair(ValueRef::slot_raw(1), 0)]);

    resolve_swapped_marginal_side(&eng, &mut levels, 0, 1, &src, true).expect("within budget");

    let p = &levels[0].pairs;
    assert_eq!(p[0].left.0, ValueRef::inline_raw(3).unwrap(), "small count must inline");
    assert_eq!(p[1].left.0, ValueRef::slot_raw(0), "existing dst count must dedup");
    assert_eq!(p[2].left.0, ValueRef::slot_raw(1), "absent count must re-mint");
    assert_eq!(p[3].left.0, ValueRef::inline_raw(2).unwrap(), "inline ref must pass through");
    assert_eq!(p[4].left.0, (1 << 31) | 5, "ZERO sentinel must pass through");
    assert_eq!(levels[0].nodes[1].a, ValueRef::slot_raw(0), "inline-node ref must remap too");
    assert_eq!(
        levels[1].marginal_counts(),
        Some(&[WIDE, WIDER][..]),
        "dst store must gain exactly the one absent count",
    );
}

/// Pre-scan early-return: when every bare ref's source count is inlinable,
/// nothing has to be interned into the dst store — so the fixup takes the
/// zero-allocation path (no interner, no store growth, no side table) and
/// still rewrites every ref exactly as the interning path would. Guards the
/// pre-scan against classifying a ref differently from the rewrite.
#[test]
fn resolve_all_inlinable_leaves_dst_store_untouched() {
    let eng = Engine::new();
    let src = marginal_level(vec![3, 1, 4]);
    let mut levels = vec![TddLevel::new(), marginal_level(vec![WIDE])];
    levels[0].push_internal_node(&[
        pair(ValueRef::slot_raw(0), 0),            // → inline 3
        pair(ValueRef::slot_raw(2), 0),            // → inline 4
        pair(ValueRef::inline_raw(2).unwrap(), 0), // inline: untouched
        pair((1 << 31) | 5, 0),                   // ZERO sentinel: untouched
    ]);
    levels[0].push_internal_node(&[pair(ValueRef::slot_raw(1), 0)]);

    resolve_swapped_marginal_side(&eng, &mut levels, 0, 1, &src, true).expect("allocates nothing");

    let p = &levels[0].pairs;
    assert_eq!(p[0].left.0, ValueRef::inline_raw(3).unwrap());
    assert_eq!(p[1].left.0, ValueRef::inline_raw(4).unwrap());
    assert_eq!(p[2].left.0, ValueRef::inline_raw(2).unwrap());
    assert_eq!(p[3].left.0, (1 << 31) | 5);
    assert_eq!(levels[0].nodes[1].a, ValueRef::inline_raw(1).unwrap());
    assert_eq!(
        levels[1].marginal_counts(),
        Some(&[WIDE][..]),
        "no ref needs a dst slot: the store must not grow",
    );
    assert!(
        levels[1].marginal_counts_big().is_none(),
        "no overflow re-mint: no side table may be built",
    );
}

/// Right-side remap of a BigUint-overflow slot: re-mints into the dst
/// store, creating the sparse big side-table on demand, and dedups equal
/// BigUints onto one slot.
#[test]
fn resolve_right_biguint_mint_and_dedup() {
    let eng = Engine::new();
    let big: BigUint = BigUint::from(u128::MAX) * 7u32;
    let mut src = marginal_level(vec![u128::MAX]);
    src.set_counts_state(
        vec![u128::MAX],
        Some([(0u32, big.clone())].into_iter().collect()),
    );
    let mut levels = vec![TddLevel::new(), marginal_level(vec![500_000])];
    levels[0].push_internal_node(&[
        pair(0, ValueRef::slot_raw(0)),
        pair(1, ValueRef::slot_raw(0)),
    ]);

    resolve_swapped_marginal_side(&eng, &mut levels, 0, 1, &src, false).expect("within budget");

    let p = &levels[0].pairs;
    assert_eq!(p[0].right.0, ValueRef::slot_raw(1), "big count must re-mint a dst slot");
    assert_eq!(p[1].right.0, ValueRef::slot_raw(1), "equal BigUint must dedup onto one slot");
    assert_eq!(
        levels[1].marginal_counts(),
        Some(&[500_000, u128::MAX][..]),
    );
    assert_eq!(
        levels[1].marginal_counts_big(),
        Some(&[(1u32, big)].into_iter().collect::<CountOverflow>()),
        "big side-table must be created and key the minted value by its new slot",
    );
}
