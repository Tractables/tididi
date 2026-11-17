//! Conjunction of diagrams on projections of one home, checked against the
//! conjunction of the operands embedded on the home.

use super::*;
use crate::Engine;
use crate::test_helpers::{assert_canonical, rotate_left, rotate_right};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree};

fn rotated_home(n: u32, seed: u64) -> Arc<Vtree> {
    let mut vtree = Vtree::random(n, seed);
    let mut rng = Lcg::new(seed ^ 0x9e37_79b9);
    for _ in 0..n {
        let v = VtreeIdx(rng.below(vtree.num_nodes() as u64) as u32);
        if rng.coin() {
            rotate_left(&mut vtree, v);
        } else {
            rotate_right(&mut vtree, v);
        }
    }
    Arc::new(vtree)
}

fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    vtree.subtree(t).filter(|&s| vtree.node(s).is_leaf()).map(|s| vtree.leaf_var(s)).collect()
}

/// A nonempty random part of `vars`.
fn part(vars: &[VarId], rng: &mut Lcg) -> Vec<VarId> {
    let keep: Vec<VarId> = vars.iter().copied().filter(|_| rng.coin()).collect();
    if keep.is_empty() { vec![vars[rng.below(vars.len() as u64) as usize]] } else { keep }
}

/// A random clause of one to three literals over `vars`.
fn clause(vars: &[VarId], rng: &mut Lcg) -> Vec<i32> {
    let width = 1 + rng.below(3);
    (0..width)
        .map(|_| {
            let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
            if rng.coin() { var } else { -var }
        })
        .collect()
}

/// A random CNF over the projection of `home` on `vars`, built on it.
fn cnf_on(eng: &Engine, home: &Arc<Vtree>, vars: &[VarId], rng: &mut Lcg) -> Tdd {
    let vtree = Vtree::projection(home, vars.iter().copied()).unwrap();
    let mut f = Tdd::one(&vtree);
    for _ in 0..1 + rng.below(4) {
        f = eng.and_clause(f, &clause(vars, rng)).unwrap();
    }
    f
}

/// Pairs of leaf sets of every kind: equal, nested, overlapping, disjoint,
/// and under the two children of one node.
fn leaf_set_pairs(home: &Vtree, rng: &mut Lcg) -> Vec<(Vec<VarId>, Vec<VarId>)> {
    let all = vars_under(home, home.root());
    let a = part(&all, rng);
    let mut pairs = vec![(a.clone(), a.clone()), (a.clone(), part(&a, rng)), (part(&a, rng), a.clone())];
    pairs.push((part(&all, rng), part(&all, rng)));
    let (b, c): (Vec<VarId>, Vec<VarId>) = all.iter().partition(|_| rng.coin());
    if !b.is_empty() && !c.is_empty() {
        pairs.push((part(&b, rng), part(&c, rng)));
    }
    let internal: Vec<VtreeIdx> = home.bottomup().filter(|&t| !home.node(t).is_leaf()).collect();
    let t = internal[rng.below(internal.len() as u64) as usize];
    let (l, r) = home.children(t);
    pairs.push((part(&vars_under(home, l), rng), part(&vars_under(home, r), rng)));
    pairs.push((part(&vars_under(home, r), rng), part(&vars_under(home, l), rng)));
    pairs
}

fn on_home(eng: &Engine, f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    eng.embed(f, home, |v| v).unwrap().0
}

#[test]
fn and_on_projections_is_and_on_the_home() {
    let eng = Engine::new();
    for seed in 0..30u64 {
        let home = rotated_home(2 + (seed % 12) as u32 * 2, seed);
        let mut rng = Lcg::new(seed ^ 0x5bd1_e995);
        for (fv, gv) in leaf_set_pairs(&home, &mut rng) {
            let (f, g) = (cnf_on(&eng, &home, &fv, &mut rng), cnf_on(&eng, &home, &gv, &mut rng));
            let expected = eng.and(on_home(&eng, &f, &home), on_home(&eng, &g, &home)).unwrap();
            let union = Vtree::projection(&home, fv.iter().chain(&gv).copied()).unwrap();
            let results = [
                eng.and(&f, &g).unwrap(),
                eng.and(f.clone(), &g).unwrap(),
                eng.and(&f, g.clone()).unwrap(),
                eng.and(f.clone(), g.clone()).unwrap(),
            ];
            for mut h in results {
                assert!(Arc::ptr_eq(h.vtree().home().unwrap(), &home));
                assert_eq!(h.vtree().num_leaves(), union.num_leaves(), "seed {seed}");
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&eng, &h, &home), &expected).unwrap(), "seed {seed}");
                eng.minimize(&mut h).unwrap();
                assert_canonical(&h);
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
            }
            // The operands are intact.
            assert_eq!(f.vtree().num_leaves() as usize, {
                let mut vars = fv.clone();
                vars.sort_unstable();
                vars.dedup();
                vars.len()
            });
        }
    }
}

#[test]
fn and_across_homes_is_refused() {
    let eng = Engine::new();
    let (a, b) = (Arc::new(Vtree::balanced(4)), Arc::new(Vtree::balanced(4)));
    let f = Tdd::clause(&Vtree::projection(&a, [VarId(1)]).unwrap(), [1]).unwrap();
    let g = Tdd::clause(&Vtree::projection(&b, [VarId(2)]).unwrap(), [2]).unwrap();
    assert_eq!(eng.and(&f, &g).unwrap_err(), OperationError::VtreeMismatch);
    let h = Tdd::clause(&a, [3]).unwrap();
    assert_eq!(eng.and(&h, &g).unwrap_err(), OperationError::VtreeMismatch);
}

#[test]
fn and_clause_extends_to_the_clause_variables() {
    let eng = Engine::new();
    for seed in 0..30u64 {
        let home = rotated_home(3 + (seed % 10) as u32 * 2, seed);
        let all = vars_under(&home, home.root());
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        for _ in 0..6 {
            let fv = part(&all, &mut rng);
            let f = cnf_on(&eng, &home, &fv, &mut rng);
            let lits = clause(&all, &mut rng);
            let expected = eng.and_clause(on_home(&eng, &f, &home), &lits).unwrap();
            let typed: Vec<crate::Literal> = lits.iter().map(|&l| crate::Literal::try_from(l).unwrap()).collect();
            for h in [eng.and_clause(f.clone(), &lits).unwrap(), eng.and_clause(f.clone(), &typed).unwrap()] {
                assert!(Arc::ptr_eq(h.vtree().home().unwrap(), &home));
                for &l in &lits {
                    assert!(h.vtree().leaf_of(VarId(l.unsigned_abs())).is_some());
                }
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&eng, &h, &home), &expected).unwrap(), "seed {seed}");
            }
        }
    }
}

#[test]
fn absent_variables_leave_the_operand() {
    let eng = Engine::new();
    let home = Arc::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [VarId(2), VarId(5)]).unwrap();
    let f = eng.and_clause(Tdd::one(&own), &[2, -5]).unwrap();
    let same = |g: Tdd| {
        assert!(Arc::ptr_eq(g.vtree(), &own));
        assert!(eng.equivalent(&g, &f).unwrap());
    };
    same(eng.exists_vars(f.clone(), &[VarId(1), VarId(6)]).unwrap());
    same(eng.forall_vars(f.clone(), &[VarId(3)]).unwrap());
    same(eng.condition(f.clone(), [crate::Literal::pos(VarId(4))]).unwrap());
    same(eng.condition_vars(f.clone(), &[VarId(1)], false).unwrap());
    assert_eq!(
        eng.exists_vars(f.clone(), &[VarId(7)]).unwrap_err(),
        OperationError::VariableNotInVtree(VarId(7))
    );
    let g = eng.exists_vars(f.clone(), &[VarId(1), VarId(2)]).unwrap();
    assert_eq!(eng.model_count(&g).unwrap(), 64u32.into());
}

#[test]
fn reductions_and_counts_run_on_projections() {
    use crate::reduce::{ContentTwinPolicy, ReductionPlan};
    let eng = Engine::new();
    for seed in 0..10u64 {
        let home = rotated_home(8 + seed as u32, seed);
        let mut rng = Lcg::new(seed ^ 0x1234_5678);
        for (fv, gv) in leaf_set_pairs(&home, &mut rng) {
            let (f, g) = (cnf_on(&eng, &home, &fv, &mut rng), cnf_on(&eng, &home, &gv, &mut rng));
            let h = eng.and(&f, &g).unwrap();
            let expected = on_home(&eng, &h, &home);
            for plan in [
                ReductionPlan::Prune,
                ReductionPlan::Contract,
                ReductionPlan::Full(ContentTwinPolicy::Skip),
                ReductionPlan::default(),
            ] {
                let mut r = h.clone();
                eng.reduce(&mut r, plan).unwrap();
                assert!(Arc::ptr_eq(r.vtree(), h.vtree()));
                assert_eq!(eng.model_count(&r).unwrap(), eng.model_count(&h).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&eng, &r, &home), &expected).unwrap(), "seed {seed}");
            }
        }
    }
}
