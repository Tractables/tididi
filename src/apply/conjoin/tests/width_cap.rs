use crate::Engine;
use crate::limits::Limits;
use crate::test_helpers::{CnfShape, Lcg, clause_to_tdd, compile_clauses_on, literals, rand_cnf, vtree_shapes};

/// A conjunction whose level would hold more nodes than a node index
/// addresses is refused with [`IndexOverflow`](crate::OperationError::IndexOverflow),
/// here under a pinned level width, whichever way the level's nodes are
/// written: two compiled diagrams, and a clause conjoined as a diagram and
/// as a clause. The conjunction that goes through is the conjunction.
#[test]
fn levels_past_the_width_cap_are_refused() {
    let mut rng = Lcg::new(0x71d7_c0de_2026_1110);
    let (mut refused, mut clause_refused) = (0, 0);
    for n in [8u32, 12] {
        let shape = |clauses| CnfShape { clauses, width: 3 };
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape(3 * n as usize)));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape(n as usize)));
            let want = eng.and(&f, &g).unwrap();
            let (got, k) = eng.limits().through_caps(Limits::pin_level_width_cap, || eng.and(&f, &g));
            assert!(eng.equivalent(&got, &want).unwrap(), "{name} over {n}: the conjunction differs");
            refused += k;
            let clause = loop {
                if let Some(clause) = rand_cnf(&mut rng, n, shape(1)).pop() {
                    break clause;
                }
            };
            let want = eng.and(&f, clause_to_tdd(&vtree, &literals(&clause))).unwrap();
            let (got, k) = eng
                .limits()
                .through_caps(Limits::pin_level_width_cap, || eng.and_clause(f.clone(), &clause[..]));
            assert!(eng.equivalent(&got, &want).unwrap(), "{name} over {n}: the clause conjunction differs");
            clause_refused += k;
        }
    }
    assert!(refused > 0, "no level width refused a conjunction");
    assert!(clause_refused > 0, "no level width refused a clause conjunction");
}
