//! Conjunction of diagrams with weights and regions on projections of one
//! home, checked against the conjunction of the diagrams they were summed
//! from, placed on the home: the sum over every variable of the product of
//! the inputs.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::marginal::tests::region::{check_form, check_values, max_intervals, on_home, operand, targets, under, weights};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, LeafSets, VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// A diagram on a projection, possibly with regions, and the structural
/// diagram it was summed from, on the home.
struct Operand {
    f: Tdd,
    smooth: Tdd,
}

thread_local! {
    static SWEPT: Cell<usize> = const { Cell::new(0) };
}

/// Count the finished nodes of `and_marginalizing_where` that the sweep
/// summed out.
pub(in crate::apply::conjoin) fn note_swept(n: usize) {
    SWEPT.with(|c| c.set(c.get() + n));
}

/// The home variables `f` has summed out.
fn summed(f: &Tdd) -> BTreeSet<VarId> {
    let mut vars = BTreeSet::new();
    let vtree = f.vtree();
    let Some(home) = vtree.home() else { return vars };
    for t in vtree.bottomup() {
        if vtree.node(t).is_region() {
            for &(lo, hi) in vtree.region_at(t).1.iter() {
                vars.extend(home.vtree().leaf_bottomup().map(|(_, v)| v).filter(|&v| (lo..hi).contains(&home.rank(v).unwrap())));
            }
        } else if vtree.node(t).is_leaf() && f.level(t).is_marginal() {
            vars.insert(vtree.leaf_var(t));
        }
    }
    vars
}

/// The variables `f` has a structural leaf for.
fn leaves(f: &Tdd) -> BTreeSet<VarId> {
    f.vtree().leaf_bottomup().filter(|&(t, _)| !f.level(t).is_marginal()).map(|(_, v)| v).collect()
}

/// Whether one of `ops` has summed out a variable another has a leaf for or
/// has summed out too.
fn conflict(ops: &[&Tdd]) -> bool {
    ops.iter().enumerate().any(|(i, a)| {
        let s = summed(a);
        ops.iter().enumerate().any(|(j, b)| i != j && (leaves(b).iter().any(|v| s.contains(v)) || summed(b).iter().any(|v| s.contains(v))))
    })
}

fn has_summed_leaf(f: &Tdd) -> bool {
    f.vtree().leaf_bottomup().any(|(t, _)| f.level(t).is_marginal())
}

/// Whether a region of `f` has another parent in `result`.
fn new_parent(f: &Tdd, result: &Tdd) -> bool {
    let (own, into) = (f.vtree(), result.vtree());
    own.regions().any(|r| {
        let Some(image) = into.node_at_home(own.home_node(r)) else { return false };
        match (own.node(r).parent(), into.node(image).parent()) {
            (Some(p), Some(q)) => own.home_node(p) != into.home_node(q),
            _ => false,
        }
    })
}

/// A random diagram over a random part of the home, summed out at random
/// home nodes zero to two times.
fn make(eng: &Engine, home: &Home, weighted: Option<(&RationalWeights, Arithmetic)>, rng: &mut Lcg) -> Operand {
    let vtree = home.vtree();
    let all: Vec<VarId> = vtree.leaf_bottomup().map(|(_, v)| v).collect();
    let used: Vec<VarId> = match rng.below(6) {
        0 => Vec::new(),
        1 | 2 => {
            let s = VtreeIdx(rng.below(vtree.num_nodes() as u64) as u32);
            vtree.subtree(s).filter(|&t| vtree.node(t).is_leaf()).map(|t| vtree.leaf_var(t)).filter(|_| rng.below(4) != 0).collect()
        }
        _ => all.iter().copied().filter(|_| rng.below(3) == 0).collect(),
    };
    let mut f = operand(eng, home, &used, rng);
    let mut smooth = on_home(eng, &f, vtree);
    if let Some((w, arithmetic)) = weighted {
        f.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
        smooth.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
    }
    let mut before = Vec::new();
    for _ in 0..rng.below(3) {
        let hs = targets(vtree, &before, rng);
        eng.marginalize(&mut f, under(home, &hs)).unwrap();
        before = hs;
    }
    Operand { f, smooth }
}

/// The projection a conjunction of `a` and `b` lives on.
fn union(a: &Tdd, b: &Tdd) -> Option<Arc<Vtree>> {
    Some(match Vtree::leaf_sets(a.vtree(), b.vtree()).ok()? {
        LeafSets::Equal(_) | LeafSets::BInA => Arc::clone(a.vtree()),
        LeafSets::AInB => Arc::clone(b.vtree()),
        LeafSets::Union(u) => u,
    })
    .filter(|into| into.num_nodes() != 0)
}

#[derive(Default)]
struct Seen {
    regions: usize,
    weighted: usize,
    new_parent: usize,
    refused: usize,
    leaf_refused: usize,
    intervals: usize,
}

struct Case {
    n: u32,
    arithmetic: Option<Arithmetic>,
    seen: Seen,
}

impl Case {
    /// Check `result` of an entry on `ops` against `smooth`, the conjunction
    /// on the home; `extra` says the entry may sum more than the operands
    /// did.
    fn check(&mut self, what: &str, ops: &[&Tdd], result: Result<Tdd, OperationError>, smooth: &Tdd, extra: bool) {
        let clash = conflict(ops);
        let result = match result {
            Err(OperationError::MarginalLevel(_)) if clash => {
                self.seen.refused += 1;
                return;
            }
            Err(OperationError::MarginalLevel(_)) if ops.iter().any(|f| has_summed_leaf(f)) => {
                self.seen.leaf_refused += 1;
                return;
            }
            Err(e) => panic!("{what}: {e:?}"),
            // A zero product of two operands drops their regions before the third.
            Ok(result) if clash => return assert!(result.is_zero() && smooth.is_zero(), "{what}: a conflict was not refused"),
            Ok(result) => result,
        };
        check_form(&result, what);
        let vars = summed(&result);
        let operands: BTreeSet<VarId> = ops.iter().flat_map(|f| summed(f)).collect();
        match extra {
            _ if result.is_zero() => {}
            true => assert!(vars.is_superset(&operands), "{what}: summed {vars:?}, operands {operands:?}"),
            false => assert_eq!(vars, operands, "{what}: {result:?}"),
        }
        assert_eq!(result.weights().is_some(), self.arithmetic.is_some(), "{what}: weights");
        for t in result.vtree().regions().filter(|_| !result.is_zero()) {
            assert!(result.level(t).is_marginal(), "{what}: region {} is not summed out: {result:?}", t.0);
        }
        for t in result.vtree().bottomup() {
            assert!(!result.level(t).is_weight_marginal() || result.level(t).weight_values().is_some(), "{what}: no column at {}: {result:?}", t.0);
        }
        check_values(&result, smooth, self.n, self.arithmetic, what);
        self.seen.intervals = self.seen.intervals.max(max_intervals(&result));
        self.seen.regions += usize::from(ops.iter().any(|f| f.vtree().regions().next().is_some()));
        self.seen.weighted += usize::from(self.arithmetic.is_some());
        self.seen.new_parent += usize::from(ops.iter().any(|f| new_parent(f, &result)));
    }
}

#[test]
fn and_on_projections_with_regions_matches_the_smooth_form() {
    let eng = Engine::new();
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let mut rng = Lcg::new(0x0a1d_5eed);
    let mut case = Case { n: 0, arithmetic: None, seen: Seen::default() };
    for index in 0..400u64 {
        let n = 1 + rng.below(64) as u32;
        let home = Home::new(Arc::new(Vtree::random(n, index)));
        let arithmetic = modes[index as usize % modes.len()];
        let w = weights(n, &mut rng);
        let weighted = arithmetic.map(|a| (&w, a));
        (case.n, case.arithmetic) = (n, arithmetic);
        let a = make(&eng, &home, weighted, &mut rng);
        let b = make(&eng, &home, weighted, &mut rng);
        let smooth = eng.and(&a.smooth, &b.smooth).unwrap();
        let ops = [&a.f, &b.f];
        let what = |entry: &str| format!("case {index} {entry} arithmetic {arithmetic:?}");
        case.check(&what("and owned"), &ops, eng.and(a.f.clone(), b.f.clone()), &smooth, false);
        case.check(&what("and borrowed"), &ops, eng.and(&a.f, &b.f), &smooth, false);
        case.check(&what("and mixed"), &ops, eng.and(&a.f, b.f.clone()), &smooth, false);
        case.check(&what("and_restoring"), &ops, eng.and_restoring(&a.f, &b.f).map_err(|r| r.error), &smooth, false);
        if let Some(into) = union(&a.f, &b.f) {
            let onto = eng.and_onto(&a.f, |v| v, &b.f, |v| v, &into).map_err(|r| match r.error {
                crate::restructure::EmbedError::Operation(e) => e,
                e => panic!("{}: {e:?}", what("and_onto")),
            });
            case.check(&what("and_onto"), &ops, onto, &smooth, false);
        }
        let hs = targets(home.vtree(), &[], &mut rng);
        case.check(&what("and_marginalizing_where"), &ops, eng.and_marginalizing_where(&a.f, &b.f, under(&home, &hs)), &smooth, true);
        // A clause over random home variables.
        let all: Vec<VarId> = home.vtree().leaf_bottomup().map(|(_, v)| v).collect();
        let mut picked: Vec<i32> = (0..1 + rng.below(3)).map(|_| all[rng.below(all.len() as u64) as usize].0 as i32).collect();
        picked.sort_unstable();
        picked.dedup();
        let clause: Vec<i32> = picked.into_iter().map(|var| if rng.coin() { var } else { -var }).collect();
        let clause_vars: Vec<VarId> = clause.iter().map(|l| VarId(l.unsigned_abs())).collect();
        let lits = Vtree::projection(&home, clause_vars).unwrap();
        let lits = Tdd::clause(&lits, clause.iter().copied()).unwrap();
        let smooth_clause = eng.and_clause(a.smooth.clone(), &clause).unwrap();
        case.check(&what("and_clause"), &[&a.f, &lits], eng.and_clause(a.f.clone(), &clause), &smooth_clause, false);
        // Three operands, left and right.
        let c = make(&eng, &home, weighted, &mut rng);
        let smooth3 = eng.and(&smooth, &c.smooth).unwrap();
        let ops3 = [&a.f, &b.f, &c.f];
        let left = eng.and(&a.f, &b.f).and_then(|ab| eng.and(ab, &c.f));
        case.check(&what("and of three, left"), &ops3, left, &smooth3, false);
        let right = eng.and(&b.f, c.f.clone()).and_then(|bc| eng.and(&a.f, bc));
        case.check(&what("and of three, right"), &ops3, right, &smooth3, false);
    }
    let Seen { regions, weighted, new_parent, refused, leaf_refused, intervals } = case.seen;
    println!("largest number of intervals in an extent: {intervals}");
    assert!(regions > 250 && weighted > 1000 && new_parent > 15 && refused > 100, "{regions} {weighted} {new_parent} {refused} {leaf_refused}");
}

/// The clause `lits` on the projection of `home` on its variables, with its
/// smooth form on the home, summed out at the home nodes `at`.
fn summed_clause(eng: &Engine, home: &Home, lits: &[i32], at: &[VtreeIdx], weighted: Option<(&RationalWeights, Arithmetic)>) -> Operand {
    let own = Vtree::projection(home, lits.iter().map(|l| VarId(l.unsigned_abs()))).unwrap();
    let mut f = Tdd::clause(&own, lits.iter().copied()).unwrap();
    let mut smooth = on_home(eng, &f, home.vtree());
    if let Some((w, arithmetic)) = weighted {
        f.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
        smooth.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
    }
    if !at.is_empty() {
        eng.marginalize(&mut f, under(home, at)).unwrap();
    }
    Operand { f, smooth }
}

/// The home node over the leaves of `vars`.
fn over(home: &Home, vars: &[u32]) -> VtreeIdx {
    let ranks: Vec<u32> = vars.iter().map(|&v| home.rank(VarId(v)).unwrap()).collect();
    home.node_over(*ranks.iter().min().unwrap(), *ranks.iter().max().unwrap())
}

#[test]
fn and_refuses_a_leaf_inside_a_region() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let a = summed_clause(&eng, &home, &[1, 2, 3], &[over(&home, &[1, 2])], None);
    let b = summed_clause(&eng, &home, &[-1, 4], &[], None);
    assert!(a.f.vtree().regions().next().is_some());
    assert!(matches!(eng.and(&a.f, &b.f), Err(OperationError::MarginalLevel(_))));
    assert!(matches!(eng.and(&b.f, &a.f), Err(OperationError::MarginalLevel(_))));
}

#[test]
fn and_refuses_intersecting_regions() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let a = summed_clause(&eng, &home, &[1, 2, 3], &[over(&home, &[1, 2])], None);
    let b = summed_clause(&eng, &home, &[1, 3, 4], &[home.vtree().root()], None);
    assert!(b.f.vtree().regions().next().is_some());
    assert!(matches!(eng.and(&a.f, &b.f), Err(OperationError::MarginalLevel(_))));
    let c = summed_clause(&eng, &home, &[-2, 3], &[over(&home, &[1, 2])], None);
    assert!(c.f.vtree().regions().next().is_none(), "a lone leaf is not summed");
    assert!(matches!(eng.and(&a.f, &c.f), Err(OperationError::MarginalLevel(_))), "a leaf in the other's region");
}

/// A region over 1 and 2 under the node of 1, 2 and 5 meets a leaf 3: in the
/// union its parent is the node over 1 to 4, which it passes through.
#[test]
fn and_gives_a_region_a_new_parent() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let w = weights(8, &mut Lcg::new(11));
    let mut case = Case { n: 8, arithmetic: None, seen: Seen::default() };
    for arithmetic in modes {
        case.arithmetic = arithmetic;
        let weighted = arithmetic.map(|a| (&w, a));
        let a = summed_clause(&eng, &home, &[1, -2, 5], &[over(&home, &[1, 2])], weighted);
        let b = summed_clause(&eng, &home, &[3, -5], &[], weighted);
        let smooth = eng.and(&a.smooth, &b.smooth).unwrap();
        let what = format!("new parent {arithmetic:?}");
        case.check(&what, &[&a.f, &b.f], eng.and(&a.f, &b.f), &smooth, false);
        case.check(&what, &[&a.f, &b.f], eng.and(b.f.clone(), a.f.clone()), &smooth, false);
    }
    assert_eq!(case.seen.new_parent, 6);
}

/// Two diagrams of `1 ∨ 3` and `2 ∨ 4` on a balanced home of four
/// variables, each summed whole: two extents of two intervals each, every
/// rank of one between two of the other's.
fn interleaved(eng: &Engine, home: &Home) -> (Operand, Operand) {
    let root = [home.vtree().root()];
    (summed_clause(eng, home, &[1, 3], &root, None), summed_clause(eng, home, &[2, 4], &root, None))
}

#[test]
fn a_leaf_between_two_ranks_of_an_extent_is_accepted() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let a = summed_clause(&eng, &home, &[1, 3], &[home.vtree().root()], None);
    assert_eq!(max_intervals(&a.f), 2);
    let b = summed_clause(&eng, &home, &[-2], &[], None);
    let mut case = Case { n: 4, arithmetic: None, seen: Seen::default() };
    let smooth = eng.and(&a.smooth, &b.smooth).unwrap();
    case.check("leaf between", &[&a.f, &b.f], eng.and(&a.f, &b.f), &smooth, false);
    case.check("leaf between, swapped", &[&a.f, &b.f], eng.and(&b.f, &a.f), &smooth, false);
    assert_eq!(case.seen.refused + case.seen.leaf_refused, 0);
}

#[test]
fn two_regions_under_one_home_node_from_two_diagrams() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let (a, b) = interleaved(&eng, &home);
    let ab = eng.and(&a.f, &b.f).unwrap();
    check_form(&ab, "two regions");
    assert_eq!(ab.vtree().regions().count(), 2);
    assert_eq!(ab.model_count().unwrap(), eng.and(&a.smooth, &b.smooth).unwrap().model_count().unwrap());
}

#[test]
fn two_sibling_regions_are_closed_into_one() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let (a, b) = interleaved(&eng, &home);
    let ab = eng.and_marginalizing_where(&a.f, &b.f, |_| false).unwrap();
    check_form(&ab, "closed");
    let regions: Vec<VtreeIdx> = ab.vtree().regions().collect();
    assert_eq!(regions.len(), 1);
    assert_eq!(&*ab.vtree().region_at(regions[0]).1, &[(0, 4)]);
    assert_eq!(ab.model_count().unwrap(), BigUint::from(9u32));
}

/// Home `((1, 2), (3, 4))`, `f` on 1 and 2 and `g` on 1: the node over 1 and
/// 2 has the free variables 3 and 4 beside it under the root, and is summed
/// while the product is built.
#[test]
fn a_summed_node_with_unused_home_variables_beside_it_is_summed_in_the_sweep() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let f = summed_clause(&eng, &home, &[1, 2], &[], None);
    let g = summed_clause(&eng, &home, &[-1, 3], &[], None);
    SWEPT.with(|c| c.set(0));
    let fg = eng.and_marginalizing_where(&f.f, &g.f, |v| v.0 <= 2).unwrap();
    assert_eq!(SWEPT.with(Cell::get), 1);
    check_form(&fg, "swept");
    assert_eq!(fg.vtree().regions().count(), 1);
    assert_eq!(fg.model_count().unwrap(), eng.and(&f.smooth, &g.smooth).unwrap().model_count().unwrap());
}

/// The variable leaves of `f`'s vtree, summed out or not.
fn var_leaves(f: &Tdd) -> BTreeSet<VarId> {
    f.vtree().leaf_bottomup().filter(|&(t, _)| !f.vtree().node(t).is_region()).map(|(_, v)| v).collect()
}

#[derive(Default)]
struct Pool {
    cases: usize,
    regions: usize,
    intervals: usize,
    wrong_summed: usize,
    wrong_refused: usize,
}

/// A pool of three to six random diagrams on random projections, joined in
/// random order by `and_marginalizing_where` with the predicate "no other
/// live diagram has this leaf and it is not held", with sum-outs of single
/// diagrams between; with `wrong`, one answer of the predicate at one join
/// is true for a variable another live diagram has. Every result is checked
/// for form, and the last against the conjunction of the inputs on the home.
fn pool_case(eng: &Engine, index: u64, wrong: bool, seen: &mut Pool, rng: &mut Lcg) {
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let n = 1 + rng.below(64) as u32;
    let home = Home::new(Arc::new(Vtree::random(n, index)));
    let arithmetic = modes[index as usize % modes.len()];
    let w = weights(n, rng);
    let all: Vec<VarId> = home.vtree().leaf_bottomup().map(|(_, v)| v).collect();
    let held: BTreeSet<VarId> = all.iter().copied().filter(|_| rng.below(8) == 0).collect();
    let mut pool = Vec::new();
    let mut smooth: Option<Tdd> = None;
    for _ in 0..3 + rng.below(4) {
        let s = VtreeIdx(rng.below(home.vtree().num_nodes() as u64) as u32);
        let used: Vec<VarId> =
            home.vtree().subtree(s).filter(|&t| home.vtree().node(t).is_leaf()).map(|t| home.vtree().leaf_var(t)).filter(|_| rng.below(3) != 0).collect();
        let mut f = operand(eng, &home, &used, rng);
        let mut on = on_home(eng, &f, home.vtree());
        if let Some(arithmetic) = arithmetic {
            f.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
            on.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
        }
        smooth = Some(match smooth {
            None => on,
            Some(s) => eng.and(s, on).unwrap(),
        });
        pool.push(f);
    }
    let smooth = smooth.unwrap();
    let what = |step: usize| format!("pool {index} step {step} wrong {wrong} arithmetic {arithmetic:?}");
    let mut wrong_var: Option<VarId> = None;
    let mut wrong_summed = false;
    let mut refused = false;
    let wrong_step = rng.below(pool.len() as u64 - 1) as usize;
    let mut step = 0;
    while pool.len() > 1 {
        if rng.below(3) == 0 {
            let i = rng.below(pool.len() as u64) as usize;
            let others: BTreeSet<VarId> = pool.iter().enumerate().filter(|&(j, _)| j != i).flat_map(|(_, f)| var_leaves(f)).collect();
            eng.marginalize(&mut pool[i], |v| !held.contains(&v) && !others.contains(&v)).unwrap_or_else(|e| panic!("{}: {e:?}", what(step)));
            check_form(&pool[i], &what(step));
        }
        let j = 1 + rng.below(pool.len() as u64 - 1) as usize;
        let i = rng.below(j as u64) as usize;
        let b = pool.swap_remove(j);
        let a = pool.swap_remove(i);
        let others: BTreeSet<VarId> = pool.iter().flat_map(var_leaves).collect();
        let mine: BTreeSet<VarId> = var_leaves(&a).union(&var_leaves(&b)).copied().collect();
        if wrong && step == wrong_step {
            wrong_var = mine.intersection(&others).next().copied();
        }
        let wrong_now = wrong_var.filter(|_| step == wrong_step);
        let summable = |v: VarId| Some(v) == wrong_now || (!held.contains(&v) && !others.contains(&v));
        let result = if rng.coin() { eng.and_marginalizing_where(&a, &b, summable) } else { eng.and_marginalizing_where(a, b, summable) };
        match result {
            Ok(r) => {
                check_form(&r, &what(step));
                seen.regions += r.vtree().regions().count();
                seen.intervals = seen.intervals.max(max_intervals(&r));
                if step == wrong_step && wrong_var.is_some_and(|v| summed(&r).contains(&v)) {
                    seen.wrong_summed += 1;
                    wrong_summed = true;
                }
                pool.push(r);
            }
            Err(OperationError::MarginalLevel(_)) if wrong_summed => {
                seen.wrong_refused += 1;
                refused = true;
                break;
            }
            Err(e) => panic!("{}: {e:?}", what(step)),
        }
        step += 1;
    }
    seen.cases += 1;
    if wrong_summed {
        assert!(refused || pool[0].is_zero(), "{}: a wrong sum was not refused", what(step));
        return;
    }
    check_values(&pool[0], &smooth, n, arithmetic, &what(step));
}

#[test]
fn a_pool_joined_with_a_predicate_matches_the_sum_of_the_inputs() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x9001_5eed);
    let mut seen = Pool::default();
    for index in 0..300u64 {
        pool_case(&eng, index, false, &mut seen, &mut rng);
    }
    let Pool { cases, regions, intervals, .. } = seen;
    println!("largest number of intervals in an extent: {intervals}");
    assert!(cases == 300 && regions > 300 && intervals > 2, "{regions} regions, {intervals} intervals");
}

#[test]
fn a_wrong_answer_of_the_predicate_is_refused_where_it_meets_a_leaf() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x0bad_5eed);
    let mut seen = Pool::default();
    for index in 0..300u64 {
        pool_case(&eng, index, true, &mut seen, &mut rng);
    }
    let Pool { wrong_summed, wrong_refused, .. } = seen;
    assert!(wrong_summed > 10 && wrong_refused > 10, "{wrong_summed} wrong sums, {wrong_refused} refusals");
}

/// cl-bugfix1: the root of a projection summed out, the root's nodes merged.
#[test]
fn summing_out_the_root_follows_the_output_when_equal_counts_merge() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(2));
    let not_x1 = Tdd::clause(&Vtree::projection(&home, [VarId(1)]).unwrap(), [-1]).unwrap();
    let summed = eng.and_marginalizing_where(Tdd::one(home.vtree()), &not_x1, |_| true).unwrap();
    assert_eq!(summed.model_count().unwrap(), BigUint::from(2u32));
}

/// Example E9c: home `(x, (((e1, e2), e3), z))`, `Φ₁(x, e)` with `e = 000`
/// only beside `x = 0`, `001` and `010` only beside `x = 1`, `011`, `100`
/// and `101` beside both, and `Φ = Φ₁ ∧ z`; the sum over `e` before and
/// after the conjunction with `z`.
#[test]
fn e9c_summing_before_and_after_a_conjunction_with_a_literal() {
    let eng = Engine::new();
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let join = |a: &Vtree, b: &Vtree| Vtree::join(a, b).unwrap();
    let e = join(&join(&leaf(2), &leaf(3)), &leaf(4));
    let home = Home::new(join(&leaf(1), &join(&e, &leaf(5))));
    let own = Vtree::projection(&home, [1, 2, 3, 4].map(VarId)).unwrap();
    let mut phi1 = Tdd::one(&own);
    for clause in [&[-1, 2, 3, 4][..], &[1, 2, 3, -4], &[1, 2, -3, 4], &[-2, -3]] {
        phi1 = phi1.and_clause(clause).unwrap();
    }
    let z = Tdd::clause(&Vtree::projection(&home, [VarId(5)]).unwrap(), [5]).unwrap();
    let is_e = |v: VarId| (2..=4).contains(&v.0);
    let mut before = phi1.clone();
    before.marginalize(is_e).unwrap();
    let before = eng.and(&before, &z).unwrap();
    let mut after = eng.and(&phi1, &z).unwrap();
    after.marginalize(is_e).unwrap();
    let fused = eng.and_marginalizing_where(&phi1, &z, is_e).unwrap();
    for f in [&before, &after, &fused] {
        check_form(f, "E9c");
        assert_eq!(f.model_count().unwrap(), BigUint::from(9u32));
    }
    println!("E9c pairs: sum then conjoin {}, conjoin then sum {}, summing conjunction {}", before.pair_count(), after.pair_count(), fused.pair_count());
}

/// `((1, 2), (3, 4))` on a root vtree, `g = (1 ∨ 2) ∧ (3 ∨ 4)`.
fn t1_operands(weighted: bool) -> (Arc<Vtree>, Tdd, [Tdd; 2]) {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut g = Tdd::clause(&vtree, [1, 2]).unwrap().and_clause([3, 4]).unwrap();
    let mut reads = Tdd::clause(&vtree, [-1, 3]).unwrap();
    let mut ignores = Tdd::clause(&vtree, [3]).unwrap();
    if weighted {
        let w = weights(4, &mut Lcg::new(3));
        for f in [&mut g, &mut reads, &mut ignores] {
            f.set_weights(WeightStore::new(w.clone(), Arithmetic::ExactRational)).unwrap();
        }
    }
    (vtree, g, [reads, ignores])
}

/// The answer of `f ∧ g` once `g` is summed at `levels`: refused where `f`
/// reads a summed variable, the sum of the product where it does not.
fn t1_check(entry: &dyn Fn(&Tdd, &Tdd) -> Result<Tdd, OperationError>, levels: &[VtreeIdx], weighted: bool, what: &str) {
    let eng = Engine::new();
    let (_, g, [reads, ignores]) = t1_operands(weighted);
    let mut summed = g.clone();
    crate::marginal::marginalize_levels(&eng, &mut summed, levels).unwrap();
    for (f, g) in [(&reads, &summed), (&summed, &reads)] {
        assert!(matches!(entry(f, g), Err(OperationError::MarginalLevel(_))), "{what}: a read of a summed variable");
    }
    let expected = eng.and(&ignores, &g).unwrap();
    for out in [entry(&ignores, &summed).unwrap(), entry(&summed, &ignores).unwrap()] {
        match weighted {
            false => assert_eq!(out.model_count().unwrap(), expected.model_count().unwrap(), "{what}"),
            true => assert_eq!(
                out.weighted_value().unwrap().unwrap().into_rational_opt(),
                expected.weighted_value().unwrap().unwrap().into_rational_opt(),
                "{what}"
            ),
        }
    }
}

#[test]
fn t1a_an_internal_summed_level_read_by_the_other_operand_is_refused() {
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    for weighted in [false, true] {
        t1_check(&|f, g| Engine::new().and(f, g), &[left], weighted, "T1a");
    }
}

#[test]
fn t1b_a_summed_leaf_read_by_the_other_operand_is_refused() {
    let vtree = Arc::new(Vtree::balanced(4));
    let x1 = vtree.leaf_of(VarId(1)).unwrap();
    for weighted in [false, true] {
        t1_check(&|f, g| Engine::new().and(f, g), &[x1], weighted, "T1b");
        t1_check(&|f, g| Engine::new().and_restoring(f, g).map_err(|r| r.error), &[x1], weighted, "T1b restoring");
    }
}

#[test]
fn t1c_a_summed_level_read_while_its_parent_is_summed_is_refused() {
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let root = vtree.root();
    for weighted in [false, true] {
        t1_check(&|f, g| Engine::new().and_marginalizing(f, g, &[root]), &[left], weighted, "T1c");
    }
}
