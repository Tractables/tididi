//! `nor_many`: the conjunction of the operands' complements.

use crate::{Tdd, Vtree, nor_many, or_many};
use std::sync::Arc;

#[test]
fn nor_many_is_the_conjunction_of_the_complements() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x27bb_2ee6);
    for num_vars in [3u32, 5, 7] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for n in [1usize, 2, 3, 5] {
                let operands: Vec<Tdd> = (0..n)
                    .map(|_| {
                        let c = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 2, width: 2 });
                        compile_clauses_on(eng, &vtree, &c)
                    })
                    .collect();
                // The reference: fold `and` over the negations, one by one.
                let mut want = Tdd::one(&vtree);
                for f in &operands {
                    want = crate::and(want, eng.negate(f.clone()).unwrap()).unwrap();
                }
                eng.minimize(&mut want).unwrap();
                let got = nor_many(operands).unwrap();
                assert!(got.equivalent(&want).unwrap(), "nor_many disagrees with the fold");
            }
        }
    }
}

#[test]
fn or_many_is_the_complement_of_nor_many() {
    let vtree = Arc::new(Vtree::balanced(4));
    let cubes = || {
        [
            Tdd::cube(&vtree, [1, 2]).unwrap(),
            Tdd::cube(&vtree, [3]).unwrap(),
            Tdd::cube(&vtree, [-4]).unwrap(),
        ]
    };
    let disjunction = or_many(cubes()).unwrap();
    let complement = nor_many(cubes()).unwrap().negate().unwrap();
    assert!(disjunction.equivalent(&complement).unwrap());
    assert_eq!(disjunction.model_count().unwrap(), 13u32.into());
}

#[test]
fn a_false_operand_drops_out_and_all_false_is_true() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::cube(&vtree, [1]).unwrap();
    let zero = Tdd::zero(&vtree);
    let with_zero = nor_many([f.clone(), zero.clone()]).unwrap();
    let without = nor_many([f]).unwrap();
    assert!(with_zero.equivalent(&without).unwrap());
    let all_false = nor_many([zero.clone(), zero]).unwrap();
    assert_eq!(all_false.model_count().unwrap(), 8u32.into());
    assert!(matches!(
        nor_many(Vec::new()),
        Err(crate::OperationError::EmptyOperands)
    ));
}
