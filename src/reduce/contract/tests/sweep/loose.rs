//! The loose levels a contraction leaves known.

use crate::Engine;
use crate::reduce::ReductionPlan;
use crate::test_helpers::{CnfShape, Lcg, assert_same_shape, rand_cnf, vtree_shapes};

/// Minimized clauses conjoined one at a time and contracted after each, as
/// a caller does between conjunctions: the contraction keeps the loose
/// levels known, so the next conjunction knows its own, and the prune of
/// the result, which walks from them, leaves what a prune of the whole
/// diagram leaves.
/// Debug builds check after each sweep that every level holding a node its
/// parent level does not name is listed.
#[test]
fn a_contraction_keeps_the_loose_levels_known() {
    let eng = Engine::new();
    let (mut known, mut merged, mut listed) = (0, 0, 0);
    for seed in 0..40u64 {
        let num_vars = 6 + (seed % 5) as u32;
        let mut rng = Lcg::new(seed ^ 0x1005_e1e7);
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 16, width: 3 });
        for (label, vtree) in vtree_shapes(num_vars) {
            // A minimized operand knows its loose levels: it has none.
            let settled = |mut g: crate::Tdd| {
                eng.minimize(&mut g).unwrap();
                g
            };
            let mut f = settled(eng.cube(&vtree, std::iter::empty::<i32>()).unwrap());
            for c in &clauses {
                f = eng.and(f, settled(eng.clause(&vtree, c.iter().copied()).unwrap())).unwrap();
                let was_known = f.dirty.loose().is_some();
                let before = f.node_count();
                eng.reduce(&mut f, ReductionPlan::Contract).unwrap();
                assert_eq!(f.dirty.loose().is_some(), was_known, "seed {seed}, {label}");
                if was_known {
                    known += 1;
                    merged += usize::from(f.node_count() < before);
                    listed += usize::from(f.dirty.loose().is_some_and(|l| !l.is_empty()));
                }
            }
            let mut whole = f.clone();
            whole.dirty.set_loose(None);
            eng.reduce(&mut f, ReductionPlan::Prune).unwrap();
            eng.reduce(&mut whole, ReductionPlan::Prune).unwrap();
            assert_same_shape(&f, &whole, &format!("seed {seed}, {label}"));
        }
    }
    eprintln!("contractions with the loose levels known: {known}, merging: {merged}, leaving a level listed: {listed}");
    assert!(merged > 0 && listed > 0, "{known} contractions knew their loose levels, {merged} merged, {listed} left one listed");
}
