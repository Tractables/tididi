//! Conjunction of diagrams with weights and regions on projections of one
//! home, checked against the conjunction of the diagrams they were summed
//! from, placed on the home and summed there.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::marginal::tests::region::{check_form, check_values, on_home, operand, targets, weights};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, LeafSets, VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// A diagram on a projection, possibly with regions, and the structural
/// diagram it was summed from, on the home.
struct Operand {
    f: Tdd,
    smooth: Tdd,
}

/// The home variables `f` has summed out, and the home nodes over them.
fn summed(f: &Tdd) -> (BTreeSet<VarId>, Vec<VtreeIdx>) {
    let (mut vars, mut nodes) = (BTreeSet::new(), Vec::new());
    let vtree = f.vtree();
    let Some(home) = vtree.home() else { return (vars, nodes) };
    for t in vtree.bottomup() {
        if vtree.node(t).is_region() {
            for &(lo, hi) in vtree.region_at(t).1.iter() {
                nodes.push(home.node_over(lo, hi - 1));
                vars.extend(home.vtree().leaf_bottomup().map(|(_, v)| v).filter(|&v| (lo..hi).contains(&home.rank(v).unwrap())));
            }
        } else if vtree.node(t).is_leaf() && f.level(t).is_marginal() {
            let var = vtree.leaf_var(t);
            vars.insert(var);
            nodes.push(home.vtree().leaf_of(var).unwrap());
        }
    }
    (vars, nodes)
}

/// The variables `f` has a structural leaf for.
fn leaves(f: &Tdd) -> BTreeSet<VarId> {
    f.vtree().leaf_bottomup().filter(|&(t, _)| !f.level(t).is_marginal()).map(|(_, v)| v).collect()
}

/// Whether one of `ops` has summed out a variable another has a leaf for or
/// has summed out too.
fn conflict(ops: &[&Tdd]) -> bool {
    ops.iter().enumerate().any(|(i, a)| {
        let (s, _) = summed(a);
        ops.iter().enumerate().any(|(j, b)| i != j && (leaves(b).iter().any(|v| s.contains(v)) || summed(b).0.iter().any(|v| s.contains(v))))
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
        eng.marginalize(&mut f, &hs).unwrap();
        before = hs;
    }
    Operand { f, smooth }
}

/// The projection a conjunction of `a` and `b` lives on.
fn union(a: &Tdd, b: &Tdd) -> Option<Arc<Vtree>> {
    Some(match Vtree::leaf_sets(a.vtree(), b.vtree()).ok()? {
        LeafSets::Equal(_) | LeafSets::BInA =>Arc::clone(a.vtree()),
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
}

struct Case<'a> {
    eng: &'a Engine,
    n: u32,
    arithmetic: Option<Arithmetic>,
    seen: Seen,
}

impl Case<'_> {
    /// Check `result` of an entry on `ops` against `smooth`, the conjunction
    /// on the home, summed where the result says it summed; `extra` says the
    /// entry may sum more than the operands did.
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
        let (vars, nodes) = summed(&result);
        let operands: BTreeSet<VarId> = ops.iter().flat_map(|f| summed(f).0).collect();
        match extra {
            _ if result.is_zero() => {}
            true => assert!(vars.is_superset(&operands), "{what}: summed {vars:?}, operands {operands:?}"),
            false => assert_eq!(vars, operands, "{what}: {result:?}"),
        }
        let mut expected = smooth.clone();
        if !nodes.is_empty() && !expected.is_zero() {
            self.eng.marginalize_levels(&mut expected, &nodes).unwrap();
        }
        assert_eq!(result.weights().is_some(), self.arithmetic.is_some(), "{what}: weights");
        for t in result.vtree().regions().filter(|_| !result.is_zero()) {
            assert!(result.level(t).is_marginal(), "{what}: region {} is not summed out: {result:?}", t.0);
        }
        if let Some(store) = result.weights() {
            for t in result.vtree().bottomup() {
                assert!(!result.level(t).is_weight_marginal() || store.is_set(t.idx()), "{what}: no column at {}: {result:?}", t.0);
            }
        }
        check_values(&result, &expected, self.n, self.arithmetic, what);
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
    let mut case = Case { eng: &eng, n: 0, arithmetic: None, seen: Seen::default() };
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
        case.check(&what("and_marginalizing"), &ops, eng.and_marginalizing(&a.f, &b.f, &hs), &smooth, true);
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
    let Seen { regions, weighted, new_parent, refused, leaf_refused } = case.seen;
    assert!(regions > 250 && weighted > 1000 && new_parent > 15 && refused > 100 && leaf_refused > 0, "{regions} {weighted} {new_parent} {refused} {leaf_refused}");
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
        eng.marginalize(&mut f, at).unwrap();
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
    assert!(matches!(eng.and(&a.f, &c.f), Err(OperationError::MarginalLevel(_))), "a region summed out by both");
}

/// A region over 1 and 2 under the node of 1, 2 and 5 meets a leaf 3: in the
/// union its parent is the node over 1 to 4, which it passes through.
#[test]
fn and_gives_a_region_a_new_parent() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let w = weights(8, &mut Lcg::new(11));
    let mut case = Case { eng: &eng, n: 8, arithmetic: None, seen: Seen::default() };
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
