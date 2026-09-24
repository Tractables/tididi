use std::sync::Arc;

use crate::{and, Engine, Tdd, Vtree};
use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::test_helpers::assert_canonical;
use crate::vtree::VarId;

#[test]
fn copied_and_moved_parts_agree_on_renamed_sparse_layouts() {
    let eng = Engine::new();
    let maps = [vec![VarId(4), VarId(1), VarId(6)], vec![VarId(3), VarId(7)]];
    for seed in 0..8 {
        let a = Arc::new(Vtree::random(3, seed));
        let b = Arc::new(Vtree::random(2, seed + 10));
        let rows: Vec<u64> = (0..8).filter(|&row| (row * 3 + seed) % 5 < 3).collect();
        let mut first = Tdd::from_models(&a, &[VarId(1), VarId(2), VarId(3)], &rows).unwrap();
        first.set_weights(WeightStore::new(RationalWeights::unit(3), Arithmetic::ExactRational)).unwrap();
        let second = Tdd::clause(&b, [1, -2]).unwrap();
        assert_canonical(&first);
        assert_canonical(&second);
        let want_count = first.model_count().unwrap() * second.model_count().unwrap() * 4u32;
        let (grafted, layout) = Tdd::graft_over(&eng,
            vec![(first.clone(), maps[0].clone()), (second.clone(), maps[1].clone())],
            &[VarId(2), VarId(5)], 7, None).unwrap();
        assert_canonical(&grafted);
        assert!(grafted.weights().is_none());
        assert_eq!(grafted.model_count().unwrap(), want_count);

        let copies: Vec<_> = [&first, &second].iter().enumerate().map(|(part, source)| {
            let (copy, placed) = source.embed(grafted.vtree(), |var| maps[part][var.idx()]).unwrap();
            assert_canonical(&copy);
            assert!(copy.weights().is_none());
            assert_eq!(placed.as_slice(), layout.part(part).as_slice());
            copy
        }).collect();
        let mut copies = copies.into_iter();
        let embedded = and(copies.next().unwrap(), copies.next().unwrap()).unwrap();
        assert_canonical(&embedded);
        assert!(embedded.equivalent(&grafted).unwrap());
        assert!(first.weights().is_some());
        assert_canonical(&first);
    }
}

#[test]
fn leaf_labels_survive_copying_and_moving_through_connectors() {
    let source = Arc::new(Vtree::leaf(VarId(5)));
    for circuit in [Tdd::zero(&source), Tdd::one(&source),
        Tdd::cube(&source, [5]).unwrap(), Tdd::cube(&source, [-5]).unwrap()] {
        assert_canonical(&circuit);
        let moved = Tdd::graft(vec![circuit.clone()], &[VarId(1), VarId(9)]).unwrap();
        let (copied, _) = circuit.embed(moved.vtree(), |v| v).unwrap();
        assert_canonical(&moved);
        assert_canonical(&copied);
        assert_eq!(moved.model_count().unwrap(), circuit.model_count().unwrap() * 4u32);
        assert!(copied.equivalent(&moved).unwrap());
    }
}

#[test]
fn moving_preserves_level_allocations_and_relocates_weight_columns() {
    let eng = Engine::new();
    let source = Arc::new(Vtree::balanced(3));
    let mut circuit = Tdd::clause(&source, [1, -2]).unwrap();
    circuit.set_weights(WeightStore::new(RationalWeights::unit(3), Arithmetic::ExactRational)).unwrap();
    let (_, inner) = source.children(source.root());
    eng.marginalize_levels(&mut circuit, &[inner]).unwrap();
    assert_canonical(&circuit);
    let root = source.root();
    let nodes = circuit.level(root).nodes().as_ptr();
    let column = circuit.weights().unwrap().level(inner.idx()).unwrap().as_ptr();
    let value = circuit.weighted_value().unwrap().unwrap();
    let (placed, map) = Tdd::graft_over(&eng,
        vec![(circuit, vec![VarId(4), VarId(2), VarId(5)])], &[VarId(1)], 5,
        Some(WeightStore::new(RationalWeights::unit(5), Arithmetic::ExactRational))).unwrap();
    assert_canonical(&placed);
    assert_eq!(placed.level(map.part(0).level_of(root)).nodes().as_ptr(), nodes);
    assert_eq!(placed.weights().unwrap().level(map.part(0).level_of(inner).idx()).unwrap().as_ptr(), column);
    assert_eq!(placed.weighted_value().unwrap().unwrap().as_rational().into_owned(), value.as_rational().into_owned() * num_bigint::BigInt::from(2));
}
