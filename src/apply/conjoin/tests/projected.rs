//! Conjunction of diagrams on projections of one home, checked against the
//! conjunction of the operands embedded on the home.

use super::*;
use crate::Engine;
use crate::reduce::ReductionPlan;
use crate::test_helpers::{assert_canonical, rotate_left, rotate_right};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};

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
fn cnf_on(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
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

fn leaf_vars(f: &Tdd) -> Vec<VarId> {
    f.vtree().leaf_bottomup().map(|(_, var)| var).collect()
}

fn on_home(f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    crate::test_helpers::embedded(f, home, |v| v).unwrap().0
}

#[test]
fn and_on_projections_is_and_on_the_home() {
    let eng = Engine::new();
    for seed in 0..30u64 {
        let hm = Home::new(rotated_home(2 + (seed % 12) as u32 * 2, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x5bd1_e995);
        for (fv, gv) in leaf_set_pairs(&home, &mut rng) {
            let (f, g) = (cnf_on(&eng, &hm, &fv, &mut rng), cnf_on(&eng, &hm, &gv, &mut rng));
            let expected = eng.and(on_home(&f, &home), on_home(&g, &home)).unwrap();
            let (f_vars, g_vars) = (leaf_vars(&f), leaf_vars(&g));
            let union = f_vars.len() + g_vars.iter().filter(|v| !f_vars.contains(v)).count();
            let results = [
                eng.and(&f, &g).unwrap(),
                eng.and(f.clone(), &g).unwrap(),
                eng.and(&f, g.clone()).unwrap(),
                eng.and(f.clone(), g.clone()).unwrap(),
            ];
            for mut h in results {
                assert!(Arc::ptr_eq(h.home().unwrap().vtree(), &home));
                if !h.is_zero() {
                    assert_eq!(h.vtree().num_leaves() as usize, union, "seed {seed}");
                }
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&h, &home), &expected).unwrap(), "seed {seed}");
                eng.minimize(&mut h).unwrap();
                assert_canonical(&h);
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
            }
            // The operands are intact.
            assert_eq!(leaf_vars(&f), f_vars);
        }
    }
}

#[test]
fn join_at_the_root_carries_what_the_operands_owe() {
    let eng = Engine::new();
    let mut owed = 0;
    for seed in 0..40u64 {
        let hm = Home::new(rotated_home(4 + (seed % 10) as u32 * 2, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        let (l, r) = home.children(home.root());
        let (fv, gv) = (part(&vars_under(&home, l), &mut rng), part(&vars_under(&home, r), &mut rng));
        let (f, g) = (cnf_on(&eng, &hm, &fv, &mut rng), cnf_on(&eng, &hm, &gv, &mut rng));
        let expected = eng.and(on_home(&f, &home), on_home(&g, &home)).unwrap();
        // Contraction visits only the levels a diagram owes it.
        let contracted = |mut d: Tdd| {
            eng.reduce_plain(&mut d, ReductionPlan::Contract).unwrap();
            d
        };
        let joined_reduced = eng.and(contracted(f.clone()), contracted(g.clone())).unwrap();
        owed += usize::from(joined_reduced.pair_count() < eng.and(&f, &g).unwrap().pair_count());
        for h in [eng.and(f.clone(), g.clone()).unwrap(), eng.and(&f, &g).unwrap()] {
            let h = contracted(h);
            assert_eq!(h.pair_count(), joined_reduced.pair_count(), "seed {seed}");
            assert!(eng.equivalent(&on_home(&h, &home), &expected).unwrap(), "seed {seed}");
            let mut h = h;
            eng.minimize(&mut h).unwrap();
            assert_canonical(&h);
        }
    }
    assert!(owed > 0, "no pair of operands owed a contraction");
}

#[test]
fn and_across_homes_is_refused() {
    let eng = Engine::new();
    let (a, b) = (Home::new(Vtree::balanced(4)), Home::new(Vtree::balanced(4)));
    let f = Tdd::clause(&Vtree::projection(&a, [VarId(1)]).unwrap(), [1]).unwrap();
    let g = Tdd::clause(&Vtree::projection(&b, [VarId(2)]).unwrap(), [2]).unwrap();
    assert_eq!(eng.and(&f, &g).unwrap_err(), OperationError::VtreeMismatch);
    let h = Tdd::clause(a.vtree(), [3]).unwrap();
    assert_eq!(eng.and(&h, &g).unwrap_err(), OperationError::VtreeMismatch);
}

#[test]
fn and_clause_extends_to_the_clause_variables() {
    let eng = Engine::new();
    for seed in 0..30u64 {
        let hm = Home::new(rotated_home(3 + (seed % 10) as u32 * 2, seed));
        let home = Arc::clone(hm.vtree());
        let all = vars_under(&home, home.root());
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        for _ in 0..6 {
            let fv = part(&all, &mut rng);
            let f = cnf_on(&eng, &hm, &fv, &mut rng);
            let lits = clause(&all, &mut rng);
            let expected = eng.and_clause(on_home(&f, &home), &lits).unwrap();
            let typed: Vec<crate::Literal> = lits.iter().map(|&l| crate::Literal::try_from(l).unwrap()).collect();
            for h in [eng.and_clause(f.clone(), &lits).unwrap(), eng.and_clause(f.clone(), &typed).unwrap()] {
                assert!(Arc::ptr_eq(h.home().unwrap().vtree(), &home));
                if lits.iter().any(|&l| lits.contains(&-l)) {
                    assert!(Arc::ptr_eq(h.vtree(), f.vtree()), "a true clause leaves the vtree");
                } else if !h.is_zero() {
                    for &l in &lits {
                        assert!(h.vtree().leaf_of(VarId(l.unsigned_abs())).is_some());
                    }
                }
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&h, &home), &expected).unwrap(), "seed {seed}");
            }
        }
    }
}

#[test]
fn absent_variables_leave_the_operand() {
    let eng = Engine::new();
    let hm = Home::new(Vtree::balanced(6));
    let own = Vtree::projection(&hm, [VarId(2), VarId(5)]).unwrap();
    let f = eng.and_clause(crate::test_helpers::padded(Tdd::one(&own), &own), &[2, -5]).unwrap();
    let same = |g: Tdd| {
        assert!(Arc::ptr_eq(g.vtree(), &own));
        assert!(eng.equivalent(&g, &f).unwrap());
    };
    same(eng.exists_vars(f.clone(), &[VarId(1), VarId(6)]).unwrap());
    same(eng.forall_vars(f.clone(), &[VarId(3)]).unwrap());
    same(eng.condition(f.clone(), [crate::Literal::pos(VarId(4))]).unwrap());
    same(eng.condition_vars(f.clone(), &[VarId(1)], false).unwrap());
    let x = [crate::Literal::pos(VarId(4)), crate::Literal::neg(VarId(4))];
    assert!(!eng.is_sat(&eng.condition(f.clone(), x).unwrap()).unwrap());
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
        let hm = Home::new(rotated_home(8 + seed as u32, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x1234_5678);
        for (fv, gv) in leaf_set_pairs(&home, &mut rng) {
            let (f, g) = (cnf_on(&eng, &hm, &fv, &mut rng), cnf_on(&eng, &hm, &gv, &mut rng));
            let h = eng.and(&f, &g).unwrap();
            let expected = on_home(&h, &home);
            for plan in [
                ReductionPlan::Prune,
                ReductionPlan::Contract,
                ReductionPlan::Full(ContentTwinPolicy::Skip),
                ReductionPlan::default(),
            ] {
                let mut r = h.clone();
                eng.reduce_plain(&mut r, plan).unwrap();
                assert!(Arc::ptr_eq(r.vtree(), h.vtree()));
                assert_eq!(eng.model_count(&r).unwrap(), eng.model_count(&h).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&r, &home), &expected).unwrap(), "seed {seed}");
            }
        }
    }
}

/// `and` of two proper projections whose union is every leaf of the home is
/// `and` of the operands embedded on a bare copy of the home's root, for
/// sets that split at the root (the join) and sets that overlap.
#[test]
fn a_conjunction_placed_on_the_root_is_and_on_a_bare_root() {
    let eng = Engine::new();
    for seed in 0..60u64 {
        let n = 2 + (seed % 10) as u32;
        let hm = Home::new(rotated_home(n, seed));
        let root = hm.vtree();
        let bare = Arc::new(Vtree::clone(root));
        let mut rng = Lcg::new(seed ^ 0x7f4a_7c15);
        let (l, r) = root.children(root.root());
        let all = vars_under(root, root.root());
        let first = all[0];
        let last = all[all.len() - 1];
        let pairs = [
            (vars_under(root, l), vars_under(root, r)),
            (
                all.iter().copied().filter(|&v| v != last && (v == first || rng.coin())).collect(),
                all.iter().copied().filter(|&v| v != first && (v == last || rng.coin())).collect(),
            ),
        ];
        for (fv, gv) in pairs {
            let mut missing: Vec<VarId> = all.iter().copied().filter(|v| !fv.contains(v) && !gv.contains(v)).collect();
            let gv: Vec<VarId> = gv.into_iter().chain(missing.drain(..)).collect();
            let (f, g) = (cnf_on(&eng, &hm, &fv, &mut rng), cnf_on(&eng, &hm, &gv, &mut rng));
            if f.vtree().home().is_none() || g.vtree().home().is_none() {
                continue;
            }
            let expected = eng.and(on_home(&f, &bare), on_home(&g, &bare)).unwrap();
            for mut h in [eng.and(&f, &g).unwrap(), eng.and(f.clone(), g.clone()).unwrap()] {
                assert!(Arc::ptr_eq(h.home().unwrap().vtree(), root), "seed {seed}");
                assert_eq!(eng.model_count(&h).unwrap(), eng.model_count(&expected).unwrap(), "seed {seed}");
                assert!(eng.equivalent(&on_home(&h, &bare), &expected).unwrap(), "seed {seed}");
                eng.minimize(&mut h).unwrap();
                assert_canonical(&h);
            }
        }
    }
}

/// The nodes of each structural internal level the output does not reach.
fn unreached(t: &Tdd) -> Vec<(u32, usize)> {
    if t.is_zero() || t.is_one() {
        return Vec::new();
    }
    let reach = t.reachable_nodes();
    (0..t.vtree().num_nodes() as u32)
        .map(VtreeIdx)
        .filter(|&v| !t.vtree().node(v).is_leaf() && !t.levels[v.idx()].is_marginal())
        .map(|v| (v.0, reach[v.idx()].iter().filter(|&&r| !r).count()))
        .filter(|&(_, dead)| dead > 0)
        .collect()
}

/// A literal placed under a pass-through chain to the root holds the
/// output's literal alone there.
#[test]
fn and_with_a_literal_keeps_only_reached_nodes() {
    let eng = Engine::new();
    let hm = Home::new(Vtree::random(6, 101));
    let x1 = Vtree::projection(&hm, [VarId(1)]).unwrap();
    let x3_x5 = Vtree::projection(&hm, [VarId(3), VarId(5)]).unwrap();
    let literal = Tdd::clause(&x1, [-1]).unwrap();
    let mut reduced = literal.clone();
    eng.minimize(&mut reduced).unwrap();
    for f in [literal, reduced] {
        let h = crate::and(f, Tdd::clause(&x3_x5, [3, -3, -5]).unwrap()).unwrap();
        assert_eq!(unreached(&h), [], "{:?}", h.dirty.loose());
    }
}

/// A diagram over `vars`: now and then a literal or a padded true on one
/// of them, which a pass-through chain reads, else a random CNF; minimized
/// one time in three.
fn operand(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
    let one = Vtree::projection(home, [vars[0]]).unwrap();
    let mut f = match rng.below(6) {
        0 => Tdd::clause(&one, [if rng.coin() { vars[0].0 as i32 } else { -(vars[0].0 as i32) }]).unwrap(),
        1 => crate::test_helpers::padded(Tdd::one(&one), &one),
        _ => cnf_on(eng, home, vars, rng),
    };
    if rng.below(3) == 0 {
        eng.minimize(&mut f).unwrap();
    }
    f
}

/// `and` across projections, each operand owned or borrowed, lists every
/// level that may hold a node no pair of its parent names, where it knows
/// its loose levels, and reaches every node where it lists none.
#[test]
fn and_across_projections_lists_its_loose_levels() {
    use crate::test_helpers::check::check_loose_levels;
    let eng = Engine::new();
    let (mut known, mut unreached_known) = (0, 0);
    for seed in 0..60u64 {
        let hm = Home::new(rotated_home(2 + (seed % 10) as u32, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x3c6e_f372);
        for (fv, gv) in leaf_set_pairs(&home, &mut rng) {
            let (f, g) = (operand(&eng, &hm, &fv, &mut rng), operand(&eng, &hm, &gv, &mut rng));
            let results = [
                eng.and(&f, &g).unwrap(),
                eng.and(f.clone(), &g).unwrap(),
                eng.and(&f, g.clone()).unwrap(),
                eng.and(f.clone(), g.clone()).unwrap(),
            ];
            for h in results {
                let at = format!("seed {seed}: {fv:?} and {gv:?}");
                check_loose_levels(&h).unwrap_or_else(|e| panic!("{at}: {e}"));
                let Some(loose) = h.dirty.loose() else { continue };
                known += 1;
                unreached_known += usize::from(!unreached(&h).is_empty());
                assert!(!loose.is_empty() || unreached(&h).is_empty(), "{at}: {:?} unreached, none listed", unreached(&h));
            }
        }
    }
    assert!(known > 100 && unreached_known > 5, "{known} results know their loose levels, {unreached_known} of them hold a node they do not reach");
}
