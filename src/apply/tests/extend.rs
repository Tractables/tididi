//! Operations of two or more diagrams on projections of one home, checked
//! against the same operations on the operands placed on the home.

use std::sync::Arc;

use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::{rotate_left, rotate_right};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree, VtreeIdx};
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
fn operand(eng: &Engine, home: &Arc<Vtree>, vars: &[VarId], rng: &mut Lcg) -> Tdd {
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
    eng.trim(&mut f).unwrap();
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
        self.check_as(op, got, expected, true);
    }

    /// [`Case::check`] of a conjunction, whose false result stays on the
    /// union of its operands' leaf sets.
    fn check_and(&self, op: &str, got: Tdd, expected: &Tdd) {
        let trimmed = !got.is_zero();
        self.check_as(op, got, expected, trimmed);
    }

    fn check_as(&self, op: &str, got: Tdd, expected: &Tdd, trimmed: bool) {
        let (eng, what) = (self.eng, &self.what);
        if trimmed && got.vtree().home().is_some() {
            check_trimmed(&got, false).unwrap_or_else(|e| panic!("{op} {what}: {e}"));
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

    case.check_and("and", eng.and(f, g).unwrap(), &eng.and(&hf, &hg).unwrap());
    case.intact("and", f, &f0);
    case.intact("and", g, &g0);
    case.check_and("and owned", eng.and(f.clone(), g.clone()).unwrap(), &eng.and(&hf, &hg).unwrap());
    case.check_and("and_restoring", eng.and_restoring(f, g).map_err(|r| r.error).unwrap(), &eng.and(&hf, &hg).unwrap());
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
        case.check_and("and_in_place", got, &product);
        case.intact("and_in_place", g, &g0);
    }
    if f.vtree().num_nodes() <= g.vtree().num_nodes() && g.vtree().num_nodes() != 0 {
        let index = eng.pair_index(g).unwrap();
        if let Ok(got) = eng.and_indexed(f, g, &index) {
            case.check_and("and_indexed", got, &product);
            case.intact("and_indexed", f, &f0);
            case.intact("and_indexed", g, &g0);
        } else {
            assert!(!leaf_vars(f.vtree()).all(|v| g.vtree().leaf_of(v).is_some()), "and_indexed {}", case.what);
        }
    }

    let count = eng.model_count(&product).unwrap();
    assert_eq!(eng.and_model_count(f.clone(), g.clone(), &[]).unwrap(), count, "and_model_count {}", case.what);
    assert_eq!(eng.marginal_product_count(f, g).unwrap(), count, "marginal_product_count {}", case.what);
    case.check_and("and_marginalizing", eng.and_marginalizing(f, g, &[]).unwrap(), &product);
    if f.vtree().num_nodes() != 0 {
        let t = VtreeIdx(rng.below(f.vtree().num_nodes() as u64) as u32);
        let at_home = f.vtree().home_node(t);
        assert_eq!(
            eng.and_model_count(f.clone(), g.clone(), &[t]).unwrap(),
            eng.and_model_count(hf.clone(), hg.clone(), &[at_home]).unwrap(),
            "and_model_count with a target {}",
            case.what
        );
        let summed = eng.and_marginalizing(f, g, &[t]).unwrap();
        assert_eq!(eng.model_count(&summed).unwrap(), count, "and_marginalizing with a target {}", case.what);
    }

    for (care, care_home) in [(g, &hg), (h, &hh)] {
        let expected = eng.and(&hf, care_home).unwrap();
        let got = eng.restrict_to_care(f.clone(), care.clone()).unwrap().into_tdd();
        if got.vtree().home().is_some() {
            check_trimmed(&got, false).unwrap_or_else(|e| panic!("restrict_to_care {}: {e}", case.what));
        }
        let restricted = eng.and(on_home(eng, &got, home), care_home).unwrap();
        assert!(eng.equivalent(&restricted, &expected).unwrap(), "restrict_to_care {}", case.what);
        let got = eng.restrict_to_care_bounded(f.clone(), care, 1 + rng.below(64)).unwrap().into_tdd();
        let restricted = eng.and(on_home(eng, &got, home), care_home).unwrap();
        assert!(eng.equivalent(&restricted, &expected).unwrap(), "restrict_to_care_bounded {}", case.what);
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
        let home = rotated_home(2 + (seed % 9) as u32, seed);
        let mut rng = Lcg::new(seed ^ 0x5851_f42d);
        let all = vars_under(&home, home.root());
        for (k, (fv, gv)) in leaf_set_pairs(&home, &mut rng).into_iter().enumerate() {
            let f = operand(&eng, &home, &fv, &mut rng);
            let g = operand(&eng, &home, &gv, &mut rng);
            let hv = part(&all, &mut rng);
            let h = operand(&eng, &home, &hv, &mut rng);
            let case = Case { eng: &eng, home: &home, what: format!("seed {seed} pair {k} f {fv:?} g {gv:?} h {hv:?}") };
            check_operations(&case, &f, &g, &h, &mut rng);
        }
    }
}

#[test]
fn constants_on_the_empty_projection() {
    let eng = Engine::new();
    let home = rotated_home(5, 3);
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
    assert!(Arc::ptr_eq(eng.and(&one, &lit).unwrap().vtree(), &x));
}

#[test]
fn weights_and_summed_out_levels_do_not_move() {
    let eng = Engine::new();
    let home = rotated_home(6, 11);
    let f = Tdd::clause(&Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap(), [1, 2, 3]).unwrap();
    let g = Tdd::clause(&Vtree::projection(&home, [VarId(4), VarId(5)]).unwrap(), [4, -5]).unwrap();
    let mut summed = f.clone();
    let root = summed.vtree().root();
    eng.marginalize_levels(&mut summed, &[root]).unwrap();
    assert!(summed.has_marginal_level());
    for result in [eng.or(summed.clone(), g.clone()), eng.and_not(g.clone(), summed.clone()), eng.add(summed.clone(), g.clone())] {
        assert_eq!(result.unwrap_err(), OperationError::WeightsAcrossProjections);
    }
    let other = Arc::new(Vtree::balanced(6));
    let elsewhere = Tdd::clause(&other, [1, 2]).unwrap();
    assert_eq!(eng.or(f, elsewhere).unwrap_err(), OperationError::VtreeMismatch);
}
