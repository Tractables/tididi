use super::*;

mod sum;
mod compaction;

#[test]
fn refused_weight_slot_preserves_values_and_width() {
    use std::sync::Arc;
    use crate::diagram::{Arithmetic, RationalWeights};
    use crate::test_helpers::{assert_canonical, compile_clauses, rat};
    use crate::Vtree;

    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut tdd = compile_clauses(&vtree, &[
        vec![1, 2], vec![-2, 3], vec![3, 4], vec![-1, -4],
    ]);
    assert_canonical(&tdd);
    tdd.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
    let (level, _) = vtree.children(vtree.root());
    eng.marginalize_levels(&mut tdd, &[level]).unwrap();
    assert_canonical(&tdd);
    let values: Vec<_> = tdd.weight_store().level(level.idx()).unwrap().iter().map(|v| v.clone().into_rational()).collect();
    let width = tdd.levels[level.idx()].slot_count();
    let expected = tdd.weighted_value().unwrap().unwrap().into_rational();
    // Force the next append to grow instead of using spare capacity.
    let column = tdd.weight_store_mut().level_vals_mut(level.idx()).unwrap();
    *column = std::mem::take(column).into_boxed_slice().into_vec();
    eng.limits().refuse_nth_reserve(0);
    let result = WeightFold::push_slot(&eng, &mut tdd, level, WeightValue::ExactSmall(0));
    eng.limits().grant_every_reserve();
    assert_eq!(result, Err(OperationError::OverBudget));
    assert_eq!(tdd.weight_store().level(level.idx()).unwrap().iter().map(|v| v.clone().into_rational()).collect::<Vec<_>>(), values);
    assert_eq!(tdd.levels[level.idx()].slot_count(), width);
    assert_eq!(tdd.weighted_value().unwrap().unwrap().into_rational(), expected);
    assert_canonical(&tdd);

    let slot = WeightFold::push_slot(&eng, &mut tdd, level, WeightValue::ExactSmall(0)).unwrap();
    assert_eq!(slot as usize, width);
    assert_eq!(tdd.levels[level.idx()].slot_count(), width + 1);
    assert_eq!(tdd.weight_store().level(level.idx()).unwrap()[slot as usize].clone().into_rational(), rat(0, 1));
    // Slot creation precedes reference rewriting; discard this unused test slot.
    tdd.minimize().unwrap();
    assert_eq!(tdd.weighted_value().unwrap().unwrap().into_rational(), expected);
    assert_canonical(&tdd);
}

#[test]
fn slot_growth_checks_the_reference_payload_limit() {
    let limit = 1usize << 30;
    assert_eq!(next_slot_index(0), Ok(0));
    assert_eq!(next_slot_index(limit - 1), Ok((limit - 1) as u32));
    for len in [limit, u32::MAX as usize, usize::MAX] {
        assert_eq!(next_slot_index(len), Err(OperationError::IndexOverflow));
    }
}
