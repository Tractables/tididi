//! Operations of two or more diagrams on projections of one home, checked
//! against the same operations on the operands placed on the home.

use std::sync::Arc;

use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::{rotate_left, rotate_right};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

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

fn leaf_vars(vtree: &Vtree) -> impl Iterator<Item = VarId> + '_ {
    vtree.leaf_bottomup().map(|(_, var)| var)
}

fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    vtree.subtree(t).filter(|&s| vtree.node(s).is_leaf()).map(|s| vtree.leaf_var(s)).collect()
}

/// A random part of `vars`, empty with probability one in eight.
fn part(vars: &[VarId], rng: &mut Lcg) -> Vec<VarId> {
    if vars.is_empty() || rng.below(8) == 0 {
        return Vec::new();
    }
    let keep: Vec<VarId> = vars.iter().copied().filter(|_| rng.coin()).collect();
    if keep.is_empty() { vec![vars[rng.below(vars.len() as u64) as usize]] } else { keep }
}

/// A random trimmed diagram on a projection of `home` inside `vars`: a CNF
/// of one to four clauses, a constant on the empty projection when `vars`
/// is empty.
fn operand(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
    let empty = Vtree::empty_projection(home);
    if vars.is_empty() {
        return if rng.coin() { Tdd::one(&empty) } else { Tdd::zero(&empty) };
    }
    let vtree = Vtree::projection(home, vars.iter().copied()).unwrap();
    let mut f = Tdd::one(&vtree);
    for _ in 0..1 + rng.below(4) {
        let width = 1 + rng.below(3);
        let clause: Vec<i32> = (0..width)
            .map(|_| {
                let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
                if rng.coin() { var } else { -var }
            })
            .collect();
        f = eng.and_clause(f, &clause).unwrap();
    }
    eng.trim(&mut f, home).unwrap();
    f
}

/// Leaf sets of every kind: equal, nested both ways, overlapping, disjoint,
/// under the two children of one node, and empty.
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
    pairs.push((Vec::new(), a.clone()));
    pairs.push((a, Vec::new()));
    pairs.push((Vec::new(), Vec::new()));
    pairs
}

/// `f` on `home`, placed independently of the code under test.
fn on_home(eng: &Engine, f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    if f.vtree().num_nodes() == 0 {
        return if f.is_one() { Tdd::one(home) } else { Tdd::zero(home) };
    }
    eng.embed(f, home, |v| v).unwrap().0
}

struct Case<'a> {
    eng: &'a Engine,
    home: &'a Arc<Vtree>,
    what: String,
}

impl Case<'_> {
    /// `got`, a result on projections, is `expected`, the result on the
    /// home, and in trimmed form.
    fn check(&self, op: &str, got: Tdd, expected: &Tdd) {
        let (eng, what) = (self.eng, &self.what);
        if got.vtree().on_home() {
            check_trimmed(&got, false).unwrap_or_else(|e| panic!("{op} {what}: {e}"));
            let mut minimized = got.clone();
            eng.minimize(&mut minimized).unwrap();
            let mut base = expected.clone();
            eng.minimize(&mut base).unwrap();
            eng.trim(&mut base, &Home::new(Arc::clone(self.home))).unwrap();
            if base.vtree().num_nodes() == minimized.vtree().num_nodes() && base.vtree().num_nodes() != 0 {
                let counts = |f: &Tdd| (f.node_count(), f.pair_count());
                assert_eq!(counts(&minimized), counts(&base), "{op} {what}: node for node");
            }
        }
        assert!(eng.equivalent(&on_home(eng, &got, self.home), expected).unwrap(), "{op} {what}");
    }

    /// `f` as it was given, after an operation that borrowed it.
    fn intact(&self, op: &str, f: &Tdd, before: &Tdd) {
        assert!(Arc::ptr_eq(f.vtree(), before.vtree()), "{op} {}: vtree changed", self.what);
        assert_eq!(f.pair_count(), before.pair_count(), "{op} {}", self.what);
        let (eng, home) = (self.eng, self.home);
        assert!(eng.equivalent(&on_home(eng, f, home), &on_home(eng, before, home)).unwrap(), "{op} {}", self.what);
    }
}

/// The operations covered, each run on `f`, `g` (and `h` where it takes a
/// third operand) and compared with the same operation on the home.
#[allow(clippy::too_many_lines)]
fn check_operations(case: &Case<'_>, f: &Tdd, g: &Tdd, h: &Tdd, rng: &mut Lcg) {
    let (eng, home) = (case.eng, case.home);
    let (hf, hg, hh) = (on_home(eng, f, home), on_home(eng, g, home), on_home(eng, h, home));
    let (f0, g0) = (f.clone(), g.clone());

    case.check("and", eng.and(f, g).unwrap(), &eng.and(&hf, &hg).unwrap());
    case.intact("and", f, &f0);
    case.intact("and", g, &g0);
    case.check("and owned", eng.and(f.clone(), g.clone()).unwrap(), &eng.and(&hf, &hg).unwrap());
    case.check("and_restoring", eng.and_restoring(f, g).map_err(|r| r.error).unwrap(), &eng.and(&hf, &hg).unwrap());
    case.intact("and_restoring", f, &f0);
    case.check("or", eng.or(f.clone(), g.clone()).unwrap(), &eng.or(hf.clone(), hg.clone()).unwrap());
    case.check("and_not", eng.and_not(f.clone(), g.clone()).unwrap(), &eng.and_not(hf.clone(), hg.clone()).unwrap());
    case.check("xor", eng.xor(f.clone(), g.clone()).unwrap(), &eng.xor(hf.clone(), hg.clone()).unwrap());
    let only_f = eng.and_not(f.clone(), g.clone()).unwrap();
    let only_f_home = eng.and_not(hf.clone(), hg.clone()).unwrap();
    case.check("add", eng.add(only_f.clone(), g.clone()).unwrap(), &eng.add(only_f_home.clone(), hg.clone()).unwrap());
    case.check(
        "ite",
        eng.ite(f.clone(), g.clone(), h.clone()).unwrap(),
        &eng.ite(hf.clone(), hg.clone(), hh.clone()).unwrap(),
    );
    let three = || vec![f.clone(), g.clone(), h.clone()];
    let three_home = || vec![hf.clone(), hg.clone(), hh.clone()];
    case.check("or_many", eng.or_many(three()).unwrap(), &eng.or_many(three_home()).unwrap());
    case.check("nor_many", eng.nor_many(three()).unwrap(), &eng.nor_many(three_home()).unwrap());

    assert_eq!(eng.equivalent(f, g).unwrap(), eng.equivalent(&hf, &hg).unwrap(), "equivalent {}", case.what);
    assert_eq!(eng.implies(f, g).unwrap(), eng.implies(&hf, &hg).unwrap(), "implies {}", case.what);
    case.intact("implies", f, &f0);
    case.intact("implies", g, &g0);
    let fg = eng.and(f, g).unwrap();
    assert!(eng.equivalent(&fg, &eng.and(g, f).unwrap()).unwrap(), "equivalent {}", case.what);

    let all = vars_under(home, home.root());
    let vars: Vec<VarId> = all.iter().copied().filter(|_| rng.coin()).collect();
    case.check(
        "and_exists",
        eng.and_exists(f.clone(), g.clone(), &vars).unwrap(),
        &eng.and_exists(hf.clone(), hg.clone(), &vars).unwrap(),
    );

    let product = eng.and(&hf, &hg).unwrap();
    for loose in [false, true] {
        let got = match loose {
            false => eng.and_in_place(f.clone(), g).unwrap(),
            true => eng.and_in_place_loose(f.clone(), g).unwrap(),
        };
        case.check("and_in_place", got, &product);
        case.intact("and_in_place", g, &g0);
    }
    if f.vtree().num_nodes() <= g.vtree().num_nodes() && g.vtree().num_nodes() != 0 {
        let index = eng.pair_index(g).unwrap();
        if let Ok(got) = eng.and_indexed(f, g, &index) {
            case.check("and_indexed", got, &product);
            case.intact("and_indexed", f, &f0);
            case.intact("and_indexed", g, &g0);
        } else {
            assert!(!leaf_vars(f.vtree()).all(|v| g.vtree().leaf_of(v).is_some()), "and_indexed {}", case.what);
        }
    }

    let count = eng.model_count(&product).unwrap();
    assert_eq!(eng.and_model_count(f.clone(), g.clone(), &[]).unwrap(), count, "and_model_count {}", case.what);
    assert_eq!(eng.marginal_product_count(f, g).unwrap(), count, "marginal_product_count {}", case.what);
    case.check("and_marginalizing_where", eng.and_marginalizing_where(f, g, |_| false).unwrap(), &product);
    if f.vtree().home().is_some() {
        assert!(matches!(eng.and_marginalizing(f, g, &[]), Err(OperationError::InertOption { .. })), "{}", case.what);
    }
    if f.vtree().num_nodes() != 0 {
        let t = VtreeIdx(rng.below(f.vtree().num_nodes() as u64) as u32);
        let at_home = f.vtree().home_node(t);
        assert_eq!(
            eng.and_model_count(f.clone(), g.clone(), &[t]).unwrap(),
            eng.and_model_count(hf.clone(), hg.clone(), &[at_home]).unwrap(),
            "and_model_count with a target {}",
            case.what
        );
        let home_vars: Vec<VarId> = home.leaf_bottomup().map(|(_, v)| v).collect();
        let pick = home_vars[rng.below(home_vars.len() as u64) as usize];
        let summed = eng.and_marginalizing_where(f, g, |v| v != pick).unwrap();
        assert_eq!(eng.model_count(&summed).unwrap(), count, "and_marginalizing_where {}", case.what);
    }

    for (care, care_home) in [(g, &hg), (h, &hh)] {
        let expected = eng.and(&hf, care_home).unwrap();
        let got = eng.restrict_to_care(f.clone(), care).unwrap().into_tdd();
        if got.vtree().home().is_some() {
            check_trimmed(&got, false).unwrap_or_else(|e| panic!("restrict_to_care {}: {e}", case.what));
        }
        let restricted = eng.and(on_home(eng, &got, home), care_home).unwrap();
        assert!(eng.equivalent(&restricted, &expected).unwrap(), "restrict_to_care {}", case.what);
    }

    let f_vars: Vec<VarId> = leaf_vars(f.vtree()).collect();
    if !f_vars.is_empty() {
        let var = f_vars[rng.below(f_vars.len() as u64) as usize];
        case.check(
            "substitute",
            eng.substitute(f.clone(), &[(var, g)]).unwrap(),
            &eng.substitute(hf.clone(), &[(var, &hg)]).unwrap(),
        );
        case.intact("substitute", g, &g0);
    }
}

#[test]
fn operations_on_projections_are_operations_on_the_home() {
    let eng = Engine::new();
    for seed in 0..150u64 {
        let hm = Home::new(rotated_home(2 + (seed % 9) as u32, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x5851_f42d);
        let all = vars_under(&home, home.root());
        for (k, (fv, gv)) in leaf_set_pairs(&home, &mut rng).into_iter().enumerate() {
            let f = operand(&eng, &hm, &fv, &mut rng);
            let g = operand(&eng, &hm, &gv, &mut rng);
            let hv = part(&all, &mut rng);
            let h = operand(&eng, &hm, &hv, &mut rng);
            let case = Case { eng: &eng, home: &home, what: format!("seed {seed} pair {k} f {fv:?} g {gv:?} h {hv:?}") };
            check_operations(&case, &f, &g, &h, &mut rng);
        }
    }
}

#[test]
fn constants_on_the_empty_projection() {
    let eng = Engine::new();
    let home = Home::new(rotated_home(5, 3));
    let empty = Vtree::empty_projection(&home);
    let (one, zero) = (Tdd::one(&empty), Tdd::zero(&empty));
    for (c, value) in [(&one, true), (&zero, false)] {
        let negated = eng.negate(c.clone()).unwrap();
        assert!(negated.vtree().num_nodes() == 0 && negated.is_one() != value);
        let conditioned = eng.condition_var(c.clone(), VarId(1), true).unwrap();
        assert!(conditioned.vtree().num_nodes() == 0 && conditioned.is_one() == value);
        let conditioned = eng.condition(c.clone(), [1, -2]).unwrap();
        assert!(conditioned.vtree().num_nodes() == 0 && conditioned.is_one() == value);
        let quantified = eng.exists_vars(c.clone(), &[VarId(1), VarId(2)]).unwrap();
        assert!(quantified.vtree().num_nodes() == 0 && quantified.is_one() == value);
        assert!(eng.support(c).unwrap().is_empty());
        for d in [&one, &zero] {
            let both = eng.and(c, d).unwrap();
            assert_eq!(both.is_one(), value && d.is_one());
            let either = eng.or(c.clone(), d.clone()).unwrap();
            check_trimmed(&either, false).unwrap();
            assert_eq!(either.is_one(), value || d.is_one());
        }
    }
    let x = Vtree::projection(&home, [VarId(2)]).unwrap();
    let lit = Tdd::cube(&x, [2]).unwrap();
    let excluded = eng.or(lit.clone(), eng.negate(lit.clone()).unwrap()).unwrap();
    assert!(excluded.is_one() && excluded.vtree().num_nodes() == 0, "x or not x is the constant");
    let contradiction = eng.and_not(lit.clone(), lit.clone()).unwrap();
    assert!(contradiction.is_zero() && contradiction.vtree().num_nodes() == 0, "x and not x is the constant");
    assert!(Arc::ptr_eq(eng.and(&one, &lit).unwrap().vtree(), lit.vtree()));
}

#[test]
fn false_of_nested_or_overlapping_leaf_sets_is_on_no_node() {
    let eng = Engine::new();
    let home = Home::new(rotated_home(5, 3));
    let on = |vars: &[u32]| Vtree::projection(&home, vars.iter().map(|&v| VarId(v))).unwrap();
    let x = Tdd::cube(&on(&[2]), [2]).unwrap();
    let wide = eng.and_clause(Tdd::cube(&on(&[2, 4, 5]), [-2]).unwrap(), &[4, 5]).unwrap();
    let across = eng.and_clause(Tdd::cube(&on(&[2, 3]), [2]).unwrap(), &[3]).unwrap();
    let all = on(&[1, 2, 3, 4, 5]);
    let (every, but_x) = (Tdd::cube(&all, [1, 2, 3, 4, 5]).unwrap(), Tdd::cube(&all, [1, -2, 3, 4, 5]).unwrap());
    assert!(Arc::ptr_eq(every.vtree(), home.vtree()));
    for (f, g) in [(&x, &wide), (&wide, &x), (&wide, &across), (&every, &but_x)] {
        let both = eng.and(f, g).unwrap();
        assert!(both.is_zero(), "{:?}", both.vtree().num_nodes());
        check_trimmed(&both, false).unwrap();
        check_trimmed(&eng.and_restoring(f, g).map_err(|r| r.error).unwrap(), false).unwrap();
    }
}

#[test]
fn summed_out_internal_levels_do_not_move() {
    let eng = Engine::new();
    let home = Home::new(rotated_home(6, 11));
    let f = Tdd::clause(&Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap(), [1, 2, 3]).unwrap();
    let g = Tdd::clause(&Vtree::projection(&home, [VarId(4), VarId(5)]).unwrap(), [4, -5]).unwrap();
    let mut summed = f.clone();
    let root = summed.vtree().root();
    crate::marginal::marginalize_levels(&eng, &mut summed, &[root]).unwrap();
    assert!(summed.has_marginal_level());
    for result in [eng.or(summed.clone(), g.clone()), eng.and_not(g.clone(), summed.clone()), eng.add(summed.clone(), g.clone())] {
        assert!(matches!(result.unwrap_err(), OperationError::MarginalLevel(_)));
    }
    let other = Arc::new(Vtree::balanced(6));
    let elsewhere = Tdd::clause(&other, [1, 2]).unwrap();
    assert_eq!(eng.or(f, elsewhere).unwrap_err(), OperationError::VtreeMismatch);
}



/// A random diagram on the projection of `home` on `vars`, not trimmed, so
/// that it may name a leaf of a variable it does not depend on as true.
fn untrimmed(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
    let mut f = Tdd::one(&Vtree::projection(home, vars.iter().copied()).unwrap());
    for _ in 0..rng.below(4) {
        let clause: Vec<i32> = (0..1 + rng.below(3))
            .map(|_| {
                let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
                if rng.coin() { var } else { -var }
            })
            .collect();
        f = eng.and_clause(f, &clause).unwrap();
    }
    f
}

#[test]
fn a_conjunction_with_one_leaf_set_inside_the_other_is_the_conjunction_on_the_home() {
    let eng = Engine::new();
    let mut nested = 0;
    for seed in 0..300u64 {
        let hm = Home::new(rotated_home(2 + (seed % 11) as u32, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        let all = leaf_vars(&home).collect::<Vec<_>>();
        let outer = match part(&all, &mut rng) {
            vars if vars.len() >= 2 => vars,
            _ => all.clone(),
        };
        let inner: Vec<VarId> = match outer.iter().copied().filter(|_| rng.coin()).collect::<Vec<_>>() {
            vars if !vars.is_empty() && vars.len() < outer.len() => vars,
            _ => outer[..1].to_vec(),
        };
        let f = untrimmed(&eng, &hm, &outer, &mut rng);
        let g = untrimmed(&eng, &hm, &inner, &mut rng);
        let case = Case { eng: &eng, home: &home, what: format!("seed {seed} f {outer:?} g {inner:?}") };
        let (f_vars, g_vars) = (leaf_vars(f.vtree()).collect::<Vec<_>>(), leaf_vars(g.vtree()).collect::<Vec<_>>());
        if g_vars.len() >= f_vars.len() || g_vars.iter().any(|v| !f_vars.contains(v)) {
            continue;
        }
        nested += 1;
        let expected = eng.and(on_home(&eng, &f, &home), on_home(&eng, &g, &home)).unwrap();
        for out in [eng.and(f.clone(), &g), eng.and(&g, f.clone()), eng.and(f.clone(), g.clone())] {
            let out = out.unwrap();
            assert!(Arc::ptr_eq(out.vtree(), f.vtree()) || out.vtree().num_nodes() == 0, "{}", case.what);
            assert!(eng.equivalent(&on_home(&eng, &out, &home), &expected).unwrap(), "{}", case.what);
        }
    }
    assert!(nested >= 100, "{nested} of 300 seeds nested");
}

#[test]
fn a_conjunction_with_a_literal_inside_an_equivalence_keeps_its_cone() {
    let eng = Engine::new();
    for seed in 0..12u64 {
        let hm = Home::new(rotated_home(2 + seed as u32 % 6, seed));
        let home = Arc::clone(hm.vtree());
        let vars = leaf_vars(&home).collect::<Vec<_>>();
        let (x, y) = (vars[0].0 as i32, vars[vars.len() - 1].0 as i32);
        let own = Vtree::projection(&hm, vars.iter().copied()).unwrap();
        let mut f = Tdd::one(&own);
        for clause in [[x, -y], [-x, y]] {
            f = eng.and_clause(f, &clause).unwrap();
        }
        for lit in [x, -x, y, -y] {
            let g = Tdd::clause(&Vtree::projection(&hm, [VarId(lit.unsigned_abs())]).unwrap(), [lit]).unwrap();
            let expected = eng.and(on_home(&eng, &f, &home), on_home(&eng, &g, &home)).unwrap();
            let out = eng.and(f.clone(), &g).unwrap();
            assert!(Arc::ptr_eq(out.vtree(), f.vtree()), "seed {seed} literal {lit}");
            assert!(eng.equivalent(&on_home(&eng, &out, &home), &expected).unwrap(), "seed {seed} literal {lit}");
            assert_eq!(eng.model_count(&out).unwrap(), num_bigint::BigUint::from(1u32) << (vars.len() - 2), "seed {seed} literal {lit}");
        }
    }
}

/// An unreduced conjunction of two random clauses on the projection of
/// `home` on `vars`, which may hold a node it does not reach.
fn unreduced(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
    let vtree = Vtree::projection(home, vars.iter().copied()).unwrap();
    let mut clause = || -> Vec<i32> {
        let lits: Vec<i32> = vars
            .iter()
            .filter_map(|v| match rng.below(3) {
                0 => None,
                1 => Some(v.0 as i32),
                _ => Some(-(v.0 as i32)),
            })
            .collect();
        if lits.is_empty() { vec![vars[0].0 as i32] } else { lits }
    };
    let (c, d) = (clause(), clause());
    eng.and(Tdd::clause(&vtree, c).unwrap(), Tdd::clause(&vtree, d).unwrap()).unwrap()
}

/// `and(f, g)` with `g` on a part of `f`'s variables equals the conjunction
/// on the home, over random homes; counts the cases `case` picks.
fn nested_cases(case: impl Fn(&Tdd, &Tdd) -> bool) -> usize {
    let eng = Engine::new();
    let mut picked = 0;
    for seed in 0..2000u64 {
        let hm = Home::new(rotated_home(3 + (seed % 8) as u32, seed));
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x51ed_27a3);
        let all = leaf_vars(&home).collect::<Vec<_>>();
        let f = eng.and(unreduced(&eng, &hm, &all, &mut rng), unreduced(&eng, &hm, &all, &mut rng)).unwrap();
        let f_vars = leaf_vars(f.vtree()).collect::<Vec<_>>();
        let inner: Vec<VarId> = f_vars.iter().copied().filter(|_| rng.below(3) == 0).collect();
        if f.is_zero() || f.vtree().num_nodes() <= 1 || inner.len() < 2 || inner.len() >= f_vars.len() {
            continue;
        }
        let g = eng.and(unreduced(&eng, &hm, &inner, &mut rng), unreduced(&eng, &hm, &inner, &mut rng)).unwrap();
        if g.is_zero() || g.vtree().num_nodes() <= 1 || leaf_vars(g.vtree()).count() != inner.len() || !case(&f, &g) {
            continue;
        }
        picked += 1;
        let what = format!("seed {seed} f {f_vars:?} g {inner:?}");
        let expected = eng.and(on_home(&eng, &f, &home), on_home(&eng, &g, &home)).unwrap();
        let out = eng.and(f.clone(), &g).unwrap();
        assert!(Arc::ptr_eq(out.vtree(), f.vtree()) || out.vtree().num_nodes() == 0, "{what}");
        assert!(eng.equivalent(&on_home(&eng, &out, &home), &expected).unwrap(), "{what}");
        assert_eq!(eng.model_count(&out).unwrap(), eng.model_count(&expected).unwrap(), "{what}");
    }
    picked
}

/// The lowest node of `f`'s vtree with every variable of `g` under it.
fn top_of(f: &Tdd, g: &Tdd) -> VtreeIdx {
    let vars = leaf_vars(g.vtree()).collect::<Vec<_>>();
    let vtree = f.vtree();
    (0..vtree.num_nodes() as u32)
        .map(VtreeIdx)
        .filter(|&t| {
            let under = vars_under(vtree, t);
            vars.iter().all(|v| under.contains(v))
        })
        .min_by_key(|&t| vars_under(vtree, t).len())
        .expect("the root holds every variable")
}

#[test]
fn a_conjunction_keeps_an_unreachable_node_of_f_above_the_top_of_g() {
    let picked = nested_cases(|f, g| {
        let reach = f.reachable_nodes();
        let mut up = f.vtree().node(top_of(f, g)).parent();
        let mut unreached = false;
        while let Some(t) = up {
            unreached |= reach[t.idx()].contains(&false);
            up = f.vtree().node(t).parent();
        }
        unreached
    });
    assert!(picked >= 5, "{picked} cases with an unreachable node above the top of g");
}

#[test]
fn a_conjunction_reads_two_nodes_at_the_top_level_of_g() {
    use crate::diagram::{ChildPair, TddNodeId, NEG_LEAF_IDX, POS_LEAF_IDX};
    let eng = Engine::new();
    let mut cases = 0;
    for seed in 0..60u64 {
        let hm = Home::new(match seed % 3 {
            0 => Arc::new(Vtree::linear(4 + (seed % 9) as u32)),
            _ => rotated_home(4 + (seed % 9) as u32, seed),
        });
        let home = Arc::clone(hm.vtree());
        let mut rng = Lcg::new(seed ^ 0x7f4a_7c15);
        let f = operand(&eng, &hm, &leaf_vars(&home).collect::<Vec<_>>(), &mut rng);
        let f_vars = leaf_vars(f.vtree()).collect::<Vec<_>>();
        if f.is_zero() || f_vars.len() < 3 {
            continue;
        }
        let k = rng.below(f_vars.len() as u64 - 1) as usize;
        let (a, b) = (f_vars[k], f_vars[k + 1]);
        // `a ∧ b` at the output, and `¬a ∧ b` beside it, which it does not
        // reach, both at the top level of `g`.
        let vtree = Vtree::projection(&hm, [a, b]).unwrap();
        let root = vtree.root();
        let mut builder = Tdd::builder(&eng, &vtree).unwrap();
        let out = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
        builder.push(&eng, root, &[ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
        let g = builder.finish(TddNodeId { vtree: root, local: out }).unwrap();
        assert_eq!(g.level(root).slot_count(), 2);
        cases += 1;
        let what = format!("seed {seed} f {f_vars:?} g {a:?} {b:?}");
        let expected = eng.and(on_home(&eng, &f, &home), on_home(&eng, &g, &home)).unwrap();
        let out = eng.and(f.clone(), &g).unwrap();
        assert!(Arc::ptr_eq(out.vtree(), f.vtree()) || out.vtree().num_nodes() == 0, "{what}");
        assert!(eng.equivalent(&on_home(&eng, &out, &home), &expected).unwrap(), "{what}");
        assert_eq!(eng.model_count(&out).unwrap(), eng.model_count(&expected).unwrap(), "{what}");
    }
    assert!(cases >= 20, "{cases} cases");
}

/// `and(f, g)` of `f` and `g`, whose leaf sets are neither equal nor nested,
/// against the conjunction on the home, and against the conjunction that
/// places both operands on the union, node for node where both are trimmed;
/// the result is on the projection on the merged variables, node for node.
fn check_union(eng: &Engine, hm: &Home, f: &Tdd, g: &Tdd, trimmed: bool, what: &str) {
    let home = hm.vtree();
    let mut merged: Vec<VarId> = leaf_vars(f.vtree()).chain(leaf_vars(g.vtree())).collect();
    merged.sort_unstable();
    merged.dedup();
    let union = Vtree::projection(hm, merged.iter().copied()).unwrap();
    let expected = eng.and(on_home(eng, f, home), on_home(eng, g, home)).unwrap();
    let mut placed = crate::apply::conjoin::conjoin_onto_as(eng, (f.into(), |v| v), (g.into(), |v| v), &union, false)
        .map_err(|r| r.error)
        .unwrap();
    crate::reduce::close_constant(eng, &mut placed);
    let outs = [eng.and(f, g), eng.and(g, f), eng.and(f.clone(), g), eng.and(f.clone(), g.clone())];
    for (i, out) in outs.into_iter().enumerate() {
        let out = out.unwrap();
        let what = format!("{what} call {i}");
        assert!(eng.equivalent(&on_home(eng, &out, home), &expected).unwrap(), "{what}");
        assert_eq!(eng.model_count(&out).unwrap(), eng.model_count(&expected).unwrap(), "{what}");
        if out.vtree().num_nodes() == 0 {
            assert!(out.is_zero(), "{what}");
            continue;
        }
        let vtree = out.vtree();
        let node_for_node = vtree.root() == union.root()
            && vtree.num_nodes() == union.num_nodes()
            && (0..vtree.num_nodes() as u32).all(|t| vtree.node(VtreeIdx(t)) == union.node(VtreeIdx(t)));
        assert!(Arc::ptr_eq(vtree, home) || node_for_node, "{what}: not the projection on the merged variables");
        if trimmed {
            assert_eq!(out.node_count(), placed.node_count(), "{what}: another form than the placed conjunction's");
        }
    }
}

#[test]
fn a_conjunction_across_leaf_sets_neither_equal_nor_nested_is_the_conjunction_on_the_home() {
    let eng = Engine::new();
    let mut kinds = [0usize; 4];
    for seed in 0..600u64 {
        let hm = Home::new(rotated_home(2 + (seed % 11) as u32, seed));
        let all = leaf_vars(hm.vtree()).collect::<Vec<_>>();
        let mut rng = Lcg::new(seed ^ 0x6a09_e667);
        let (f_vars, g_vars, kind) = match seed % 4 {
            0 => {
                let (a, b): (Vec<VarId>, Vec<VarId>) = all.iter().partition(|_| rng.coin());
                (a, b, "disjoint")
            }
            1 => (part(&all, &mut rng), part(&all, &mut rng), "interleaved"),
            2 => {
                let lit = all[rng.below(all.len() as u64) as usize];
                (all.iter().copied().filter(|&v| v != lit && rng.below(4) != 0).collect(), vec![lit], "one literal")
            }
            _ => (part(&all, &mut rng), part(&all, &mut rng), "untrimmed"),
        };
        if f_vars.is_empty() || g_vars.is_empty() || f_vars.iter().all(|v| g_vars.contains(v)) || g_vars.iter().all(|v| f_vars.contains(v)) {
            continue;
        }
        let trimmed = seed % 4 != 3;
        let (f, g) = match trimmed {
            true => (operand(&eng, &hm, &f_vars, &mut rng), operand(&eng, &hm, &g_vars, &mut rng)),
            false => (untrimmed(&eng, &hm, &f_vars, &mut rng), untrimmed(&eng, &hm, &g_vars, &mut rng)),
        };
        let (fv, gv) = (leaf_vars(f.vtree()).collect::<Vec<_>>(), leaf_vars(g.vtree()).collect::<Vec<_>>());
        if fv.is_empty() || gv.is_empty() || fv.iter().all(|v| gv.contains(v)) || gv.iter().all(|v| fv.contains(v)) {
            continue;
        }
        kinds[(seed % 4) as usize] += 1;
        check_union(&eng, &hm, &f, &g, trimmed, &format!("seed {seed} {kind} f {fv:?} g {gv:?}"));
    }
    assert!(kinds.iter().all(|&k| k >= 25), "cases by kind {kinds:?}");
}

/// Above `f`'s top the parent of the top holds one node for each node of
/// `g` there, and is not the root: home ((a, b), d), `g` on a, `f` on b and d.
#[test]
fn a_conjunction_across_leaf_sets_builds_the_parent_of_a_top_that_is_not_the_root() {
    let eng = Engine::new();
    for (home, f_vars, g_vars) in [
        // ((a, b), d): a = 1, b = 2, d = 3.
        (Vtree::from_in_order(&[VarId(1), VarId(2), VarId(3)], &[2, 1], 3).unwrap(), vec![2, 3], vec![1]),
        // ((a, (b, c)), d): g on a, b and d, f = c.
        (Vtree::from_in_order(&[VarId(1), VarId(2), VarId(3), VarId(4)], &[2, 3, 1], 4).unwrap(), vec![3], vec![1, 2, 4]),
    ] {
        let hm = Home::new(home);
        let (fv, gv): (Vec<VarId>, Vec<VarId>) = (f_vars.into_iter().map(VarId).collect(), g_vars.into_iter().map(VarId).collect());
        let (fp, gp) = (Vtree::projection(&hm, fv.iter().copied()).unwrap(), Vtree::projection(&hm, gv.iter().copied()).unwrap());
        let lits = |vars: &[VarId], signs: u32| vars.iter().enumerate().map(|(i, v)| if signs >> i & 1 == 1 { v.0 as i32 } else { -(v.0 as i32) }).collect::<Vec<_>>();
        for fs in 0..1u32 << fv.len() {
            for gs in 0..1u32 << gv.len() {
                let f = eng.clause(&fp, lits(&fv, fs)).unwrap();
                let g = eng.clause(&gp, lits(&gv, gs)).unwrap();
                check_union(&eng, &hm, &f, &g, true, &format!("f {:?} g {:?}", lits(&fv, fs), lits(&gv, gs)));
                let (f, g) = (eng.cube(&fp, lits(&fv, fs)).unwrap(), eng.cube(&gp, lits(&gv, gs)).unwrap());
                check_union(&eng, &hm, &f, &g, true, &format!("cubes f {:?} g {:?}", lits(&fv, fs), lits(&gv, gs)));
            }
        }
    }
}

/// Each overlay of operands on different leaf sets, read where they are, is
/// the overlay of the operands placed on their union, node for node, and the
/// operation on the home.
#[test]
fn an_overlay_across_leaf_sets_is_the_overlay_of_the_placed_operands() {
    use crate::apply::overlay::{overlay_on, Overlay};
    use std::borrow::Cow;
    let eng = Engine::new();
    let mut kinds = [0usize; 5];
    for seed in 0..1000u64 {
        let hm = Home::new(rotated_home(2 + (seed % 9) as u32, seed));
        let home = Arc::clone(hm.vtree());
        let all = leaf_vars(&home).collect::<Vec<_>>();
        let mut rng = Lcg::new(seed ^ 0xbb67_ae85);
        let kind = (seed % 5) as usize;
        let (f_vars, g_vars) = match kind {
            0 => all.iter().partition(|_| rng.coin()),
            1 => {
                let a = part(&all, &mut rng);
                let b = part(&a, &mut rng);
                (a, b)
            }
            2 => {
                let a = part(&all, &mut rng);
                (part(&a, &mut rng), a)
            }
            3 => (part(&all, &mut rng), part(&all, &mut rng)),
            _ => {
                let a = part(&all, &mut rng);
                (a.clone(), a)
            }
        };
        if f_vars.is_empty() || g_vars.is_empty() {
            continue;
        }
        let (f, g) = match seed % 2 == 0 {
            true => (operand(&eng, &hm, &f_vars, &mut rng), operand(&eng, &hm, &g_vars, &mut rng)),
            false => (untrimmed(&eng, &hm, &f_vars, &mut rng), untrimmed(&eng, &hm, &g_vars, &mut rng)),
        };
        kinds[kind] += 1;
        for op in [Overlay::Or, Overlay::AndNot, Overlay::Add] {
            let f = match op {
                Overlay::Add => eng.and_not(f.clone(), g.clone()).unwrap(),
                _ => f.clone(),
            };
            let what = format!("seed {seed} kind {kind} {op:?} f {:?} g {:?}", leaf_vars(f.vtree()).collect::<Vec<_>>(), leaf_vars(g.vtree()).collect::<Vec<_>>());
            let got = overlay_on(&eng, op, f.clone(), g.clone()).unwrap();
            let [pf, pg] = crate::apply::extend::on_one(&eng, [Cow::Owned(f.clone()), Cow::Owned(g.clone())]).unwrap().map(Cow::into_owned);
            let base = overlay_on(&eng, op, pf, pg).unwrap();
            assert!(got.node_count() <= base.node_count(), "{what}: more nodes than the placed overlay");
            let minimized = |t: &Tdd| {
                let mut t = t.clone();
                eng.minimize(&mut t).unwrap();
                (t.node_count(), t.pair_count())
            };
            assert_eq!(minimized(&got), minimized(&base), "{what}: node for node");
            if got.vtree().on_home() {
                check_trimmed(&got, false).unwrap_or_else(|e| panic!("{what}: {e}"));
            }
            let (hf, hg) = (on_home(&eng, &f, &home), on_home(&eng, &g, &home));
            let expected = match op {
                Overlay::Or => eng.or(hf, hg),
                Overlay::AndNot => eng.and_not(hf, hg),
                Overlay::Add => eng.add(hf, hg),
            }
            .unwrap();
            assert!(eng.equivalent(&on_home(&eng, &got, &home), &expected).unwrap(), "{what}");
        }
    }
    assert!(kinds.iter().all(|&k| k >= 100), "cases by kind {kinds:?}");
}
