//! Variable support, conditioning and implied literals.

use super::*;
use crate::Engine;
use crate::diagram::Literal;
use crate::test_helpers::clause;

fn cls(vtree: &Arc<Vtree>, lits: &[(u32, bool)]) -> Tdd {
    clause_to_tdd(vtree, &clause(lits))
}

fn unit_forced_tdd() -> Tdd {
    let vtree = Arc::new(Vtree::balanced(3));
    let c0 = cls(&vtree, &[(1, true)]);
    let f = cls(&vtree, &[(1, true), (2, true)]);
    let g = cls(&vtree, &[(2, true), (3, true)]);
    apply_and(apply_and(c0, f), g)
}

fn xor2_tdd() -> (Arc<Vtree>, Tdd) {
    let vtree = Arc::new(Vtree::balanced(2));
    let t = and2(
        &cls(&vtree, &[(1, true), (2, true)]),
        &cls(&vtree, &[(1, false), (2, false)]),
    );
    (vtree, t)
}

#[test]
fn support_mask_tracks_dependence() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = and2(&cls(&vtree, &[(1, true)]), &cls(&vtree, &[(3, true)]));
    let sup = support_mask(&f);
    assert_eq!(sup, vec![true, false, true], "support should be {{x0,x2}}");
    for x in 0..3u32 {
        let projected = f.clone().exists_var(VarId(x + 1)).unwrap();
        assert_eq!(
            !equiv(&f, &projected),
            sup[x as usize],
            "support[{x}] mismatch vs oracle"
        );
    }
}

#[test]
fn support_bits_covers_support_mask_and_detects_disjoint() {
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = cls(&vtree, &[(1, true)]);
    let f = and2(&x0, &cls(&vtree, &[(3, true)]));
    let (mask, bits) = (support_mask(&f), support_bits(&f));
    for (x, &m) in mask.iter().enumerate() {
        let b = (bits[x / 64] >> (x % 64)) & 1 == 1;
        assert_eq!(b, m, "support_bits exact on minimized f at var {x}");
    }
    let bg = support_bits(&cls(&vtree, &[(2, true)]));
    assert!(bits.iter().zip(&bg).all(|(a, b)| a & b == 0));
    let bx0 = support_bits(&x0);
    assert!(!bits.iter().zip(&bx0).all(|(a, b)| a & b == 0));
}

#[test]
fn condition_var_detects_unit_forced_apply() {
    let t = unit_forced_tdd();
    assert!(!count_is_zero(&t));
    assert!(count_is_zero(
        &t.clone().condition_var(VarId(1), false).unwrap()
    ));
    assert!(!count_is_zero(
        &t.clone().condition_var(VarId(1), true).unwrap()
    ));
}

#[test]
fn condition_var_canonicalizes_a_dead_result() {
    let t = unit_forced_tdd();
    let dead = t.clone().condition_var(VarId(1), false).unwrap();
    assert!(count_is_zero(&dead) && dead.is_zero());
    assert!(count_is_zero(&and2(&dead, &t)));
}

#[test]
fn condition_var_on_marginalized_leaf_fails_fast() {
    let eng = &Engine::new();
    let (vtree, mut t) = xor2_tdd();
    let leaf = vtree.leaf_of(VarId(2)).unwrap();
    crate::marginal::marginalize_leaf_inline(&mut t, leaf, &vtree);
    assert!(t.levels[leaf.idx()].is_marginal());
    t.minimize().unwrap();
    assert_canonical(&t);
    assert_eq!(
        eng.condition_var(t, VarId(2), true).unwrap_err(),
        crate::OperationError::MarginalLevel(leaf)
    );
}

#[test]
fn condition_var_through_marginal_parent_fails_fast() {
    let eng = Engine::new();
    let (vtree, mut t) = xor2_tdd();
    let leaf = vtree.leaf_of(VarId(1)).unwrap();
    let parent = vtree.node(leaf).parent().unwrap();
    crate::marginal::marginalize_batch(&eng, &mut t, &[parent], &vtree).unwrap();
    assert!(!t.levels[leaf.idx()].is_marginal());
    assert_canonical(&t);
    assert_eq!(
        eng.condition_var(t, VarId(1), true).unwrap_err(),
        crate::OperationError::MarginalLevel(parent)
    );
}

#[test]
fn implied_literals_matches_condition_oracle() {
    let oracle = |f: &Tdd, nvars: u32| -> Vec<Literal> {
        let mut out = Vec::new();
        for v in 0..nvars {
            for val in [false, true] {
                if count_is_zero(&f.clone().condition_var(VarId(v + 1), !val).unwrap()) {
                    out.push(Literal::new(VarId(v + 1), val));
                }
            }
        }
        out
    };
    let vtree = Arc::new(Vtree::balanced(3));
    let x0 = cls(&vtree, &[(1, true)]);
    let nx0 = cls(&vtree, &[(1, false)]);
    let x1 = cls(&vtree, &[(2, true)]);
    let or12 = cls(&vtree, &[(2, true), (3, true)]);

    let mut f = and2(&x0, &or12);
    f.minimize().unwrap();
    let bb = f.implied_literals().unwrap();
    assert_eq!(bb, oracle(&f, 3));
    assert_eq!(bb, [Literal::pos(VarId(1))]);

    let mut g = and2(&nx0, &x1);
    g.minimize().unwrap();
    let bbg = g.implied_literals().unwrap();
    assert_eq!(bbg, oracle(&g, 3));
    assert!(bbg.iter().all(|lit| lit.var != VarId(3)));

    let mut z = and2(&x0, &nx0);
    z.minimize().unwrap();
    assert_canonical(&z);
    assert_eq!(z.implied_literals().unwrap(), oracle(&z, 3));
    assert_eq!(z.implied_literals().unwrap().len(), 6);
}
