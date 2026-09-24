//! The leaf form a negation emits in.
//!
//! Determinism allows a level to name a leaf child either as `One` or as
//! `Pos`/`Neg`, never both, so a fill node and the complement at the root are
//! emitted in whichever form the level already uses. These tests pin that:
//! mixing the two would be an invalid diagram, and it is also what blocks
//! leaf-twin contraction, which is all-or-nothing per level.

use super::*;
use crate::diagram::{ChildSide, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd};
use std::sync::Arc;

/// `(uses One, uses Pos or Neg)` at `side` of level `t`.
fn labels_at(tdd: &Tdd, t: usize, side: ChildSide) -> (bool, bool) {
    let level = &tdd.levels[t];
    let (mut one, mut literal) = (false, false);
    if level.is_marginal() {
        return (one, literal);
    }
    for j in 0..level.slot_count() {
        for p in level.pairs_of_idx(j) {
            let label = if side == ChildSide::Left { p.left.0 } else { p.right.0 };
            one |= label == ONE_LEAF_IDX.0;
            literal |= label == POS_LEAF_IDX.0 || label == NEG_LEAF_IDX.0;
        }
    }
    (one, literal)
}

/// Panics naming the level if any leaf side of `tdd` names both forms.
fn assert_no_mixed_leaf_side(tdd: &Tdd, what: &str) {
    let vtree = Arc::clone(tdd.vtree());
    for (t, left, right) in vtree.internal_bottomup() {
        for (child, side) in [(left, ChildSide::Left), (right, ChildSide::Right)] {
            if !vtree.node(child).is_leaf() {
                continue;
            }
            let (one, literal) = labels_at(tdd, t.idx(), side);
            assert!(
                !(one && literal),
                "{what}: level {} names leaf {} as One and as Pos/Neg",
                t.idx(),
                child.idx()
            );
        }
    }
}

#[test]
fn the_fill_joins_a_level_in_the_form_that_level_already_uses() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(crate::Vtree::linear(4));
    // A cube over variable 1 leaves every level below it naming its leaf as
    // `One`; the levels above name theirs as `Pos`.
    let f = Tdd::cube(&vtree, [1]).unwrap();
    let mut full = f.clone();
    expand_full(eng, &mut full).unwrap();
    assert_no_mixed_leaf_side(&full, "expand_full");
    assert_no_mixed_leaf_side(&eng.negate(f).unwrap(), "negate");
}

#[test]
fn no_negation_mixes_the_two_leaf_forms_on_random_formulas() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x2545_f491);
    for num_vars in [3u32, 5, 8] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let f = compile_clauses_on(eng, &vtree, &clauses);
                if f.is_zero() {
                    continue;
                }
                let mut full = f.clone();
                expand_full(eng, &mut full).unwrap();
                assert_no_mixed_leaf_side(&full, "expand_full");
                // Before `reduce`, so the complement at the root is checked in
                // the form it was emitted in rather than after contraction.
                let raw = crate::apply::negate::negate_on(eng, f.clone()).unwrap();
                assert_no_mixed_leaf_side(&raw, "negate_on");
                assert_no_mixed_leaf_side(&eng.negate(f).unwrap(), "negate");
            }
        }
    }
}

#[test]
fn a_fill_on_a_one_form_level_is_emitted_as_one() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x1405_7b7e);
    let mut one_form_fills = 0usize;
    for num_vars in [3u32, 5, 8] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let tdd = compile_clauses_on(eng, &vtree, &clauses);
                if tdd.is_zero() {
                    continue;
                }
                let before: Vec<usize> = tdd.levels.iter().map(|l| l.slot_count()).collect();
                // Only a leaf child reads index 0 as the label `One`; on an
                // internal child it is an ordinary node.
                let mut forms = vec![(false, false); vtree.num_nodes()];
                for (t, l, r) in vtree.internal_bottomup() {
                    forms[t.idx()] = (
                        vtree.node(l).is_leaf() && labels_at(&tdd, t.idx(), ChildSide::Left).0,
                        vtree.node(r).is_leaf() && labels_at(&tdd, t.idx(), ChildSide::Right).0,
                    );
                }
                let mut full = tdd.clone();
                expand_full(eng, &mut full).unwrap();
                for (t, _l, _r) in Arc::clone(full.vtree()).internal_bottomup() {
                    let ti = t.idx();
                    if full.levels[ti].slot_count() == before[ti] {
                        continue; // no fill here
                    }
                    let fill = full.levels[ti].slot_count() - 1;
                    for p in full.levels[ti].pairs_of_idx(fill) {
                        if forms[ti].0 {
                            assert_eq!(p.left.0, ONE_LEAF_IDX.0, "fill split a One on level {ti}");
                        }
                        if forms[ti].1 {
                            assert_eq!(p.right.0, ONE_LEAF_IDX.0, "fill split a One on level {ti}");
                        }
                    }
                    one_form_fills += usize::from(forms[ti].0 || forms[ti].1);
                }
            }
        }
    }
    assert!(one_form_fills > 0, "no One-form level ever took a fill");
}
