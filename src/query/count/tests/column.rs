use super::*;
use num_bigint::BigUint;

#[test]
fn widening_and_big_overflow_preserve_other_slots() {
    let eng = Engine::new();
    let mut col = QueryCounts::try_with_width(&eng, 3).unwrap();
    col.set(&eng, 0, Count::Fast(u64::MAX as u128)).unwrap();
    col.set(&eng, 1, Count::Fast(7)).unwrap();
    assert!(matches!(col, QueryCounts::Narrow(_)));
    col.set(&eng, 2, Count::Fast(u64::MAX as u128 + 1)).unwrap();
    assert!(matches!(col, QueryCounts::Wide(_)));
    assert!(matches!(col.get(0), CountRead::Fast(x) if x == u64::MAX as u128));
    assert!(matches!(col.get(1), CountRead::Fast(7)));
    let big: BigUint = BigUint::from(1u32) << 160;
    col.set(&eng, 2, Count::Big(big.clone())).unwrap();
    assert!(matches!(col.get(2), CountRead::Big(x) if *x == big));
    col.set(&eng, 2, Count::Fast(11)).unwrap();
    assert!(matches!(col.get(2), CountRead::Fast(11)));
}

#[test]
fn refused_promotion_preserves_the_narrow_column_and_retries() {
    let eng = Engine::new();
    let big: BigUint = BigUint::from(1u32) << 160;
    let mut refusals = 0;
    for nth in 0..8 {
        let mut col = QueryCounts::try_with_width(&eng, 2).unwrap();
        col.set(&eng, 0, Count::Fast(17)).unwrap();
        col.set(&eng, 1, Count::Fast(23)).unwrap();
        eng.limits().refuse_nth_reserve(nth);
        let result = col.set(&eng, 0, Count::Big(big.clone()));
        eng.limits().grant_every_reserve();
        if result.is_err() {
            refusals += 1;
            assert!(matches!(col, QueryCounts::Narrow(_)));
            assert!(matches!(col.get(0), CountRead::Fast(17)));
            assert!(matches!(col.get(1), CountRead::Fast(23)));
            col.set(&eng, 0, Count::Big(big.clone())).unwrap();
        }
        assert!(matches!(col.get(0), CountRead::Big(x) if *x == big));
        assert!(matches!(col.get(1), CountRead::Fast(23)));
    }
    assert!(refusals >= 2);
}

#[test]
fn mixed_column_widths_use_exact_widening_products() {
    let eng = Engine::new();
    let mut narrow = QueryCounts::try_with_width(&eng, 1).unwrap();
    narrow.set(&eng, 0, Count::Fast(u64::MAX as u128)).unwrap();
    let mut wide = CountVec::try_with_width(&eng, 1).unwrap();
    wide.set(&eng, 0, Count::Fast(u64::MAX as u128)).unwrap();
    let wide = QueryCounts::Wide(wide);
    let pair = crate::test_helpers::pair(0, 0);
    let expected = (u64::MAX as u128) * (u64::MAX as u128);
    for (left, right) in [(&narrow, &narrow), (&narrow, &wide), (&wide, &narrow), (&wide, &wide)] {
        assert_eq!(QueryCounts::fold_structural([pair].into_iter(), left, right), Some(expected));
        assert_eq!(QueryCounts::fold_structural([pair, pair].into_iter(), left, right), None);
    }
}

#[test]
fn leaf_sized_columns_need_no_heap_until_widening() {
    let eng = Engine::new();
    let mut col = QueryCounts::try_with_width(&eng, crate::diagram::LEAF_WIDTH).unwrap();
    assert_eq!(col.charged_bytes(), 0);
    col.set(&eng, 0, Count::Fast(u64::MAX as u128)).unwrap();
    assert_eq!(col.charged_bytes(), 0);
    col.set(&eng, 0, Count::Fast(u64::MAX as u128 + 1)).unwrap();
    assert!(col.charged_bytes() > 0);
    assert!(matches!(col.get(0), CountRead::Fast(x) if x == u64::MAX as u128 + 1));
}
