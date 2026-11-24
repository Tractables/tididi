//! Summing out home nodes of diagrams on projections: defects, one failing
//! test each, and a randomized comparison with the diagram on the home.

// An entry is called for whether it panics; what it returns is dropped.
#![allow(dropping_copy_types)]

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_bigint::BigUint;
use num_rational::BigRational;
use tididi::diagram::{Arithmetic, ChildPair, LiteralWeights, RationalWeights, TddNodeId, Weights, POS_LEAF_IDX};
use tididi::vtree::{Home, VarId, Vtree, VtreeIdx};
use tididi::{Engine, OperationError, Tdd};

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }

    fn coin(&mut self) -> bool {
        self.below(2) == 1
    }
}

static PANIC_AT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Record where each panic starts instead of printing it.
fn quiet() {
    std::panic::set_hook(Box::new(|info| {
        *PANIC_AT.lock().unwrap() = info.location().map(ToString::to_string).unwrap_or_default();
    }));
}

fn text(e: &(dyn std::any::Any + Send)) -> String {
    let what = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default();
    format!("{what} at {}", PANIC_AT.lock().unwrap())
}

/// Summing out a node under which the diagram has no variable left, then
/// conjoining a clause.
#[test]
fn and_clause_after_a_sum_out_answers() {
    quiet();
    let home = Home::new(Vtree::balanced(4));
    let (left, _) = home.vtree().children(home.vtree().root());
    let mut outcomes = Vec::new();
    for (name, vars, clause, target) in [
        ("true, a node with no own leaf", &[1u32, 2, 3][..], &[1, -1][..], left),
        ("x3, the node over x1 and x2", &[1, 2, 3], &[3], left),
        ("x1 or x2, the node over them", &[1, 2, 3], &[1, 2], left),
        ("x1 or x3, the root", &[1, 2, 3], &[1, 3], home.vtree().root()),
    ] {
        let own = Vtree::projection(&home, vars.iter().map(|&v| VarId(v))).unwrap();
        let mut f = Tdd::clause(&own, clause.to_vec()).unwrap();
        f.marginalize(under(&home, &[target])).unwrap();
        let count = f.model_count().unwrap();
        for c in [[3, 3], [4, 4], [1, 1], [-3, 4]] {
            let r = catch_unwind(AssertUnwindSafe(|| f.clone().and_clause(c).map(|g| g.model_count())));
            if let Err(e) = r {
                outcomes.push(format!("{name}, count {count}, and_clause {c:?}: {}", text(&*e)));
            }
        }
    }
    eprintln!("{}", outcomes.join("\n"));
    assert!(outcomes.is_empty(), "{}", outcomes.join("\n"));
}

/// The predicate "under one of the home nodes `hs`".
fn under<'a>(home: &'a Home, hs: &'a [VtreeIdx]) -> impl Fn(VarId) -> bool + 'a {
    move |v| home.rank(v).is_some_and(|r| hs.iter().any(|&h| home.rank_range(h).contains(&r)))
}

fn leaf(v: u32) -> Vtree {
    Vtree::leaf(VarId(v))
}

fn join(l: &Vtree, r: &Vtree) -> Vtree {
    Vtree::join(l, r).unwrap()
}

/// A refused sum-out leaves the count as it was.
///
/// Home `(m, (((x, u), z), y))` with `u` unused: the leaf of `m` is summed
/// out, `y` is conditioned, and the node over `x`, `u` and `z` is summed out.
#[test]
fn a_refused_marginalize_keeps_the_count() {
    let (m, x, u, z, y) = (1, 2, 3, 4, 5);
    let home = Home::new(join(&leaf(m), &join(&join(&join(&leaf(x), &leaf(u)), &leaf(z)), &leaf(y))));
    let own = Vtree::projection(&home, [m, x, z, y].map(VarId)).unwrap();
    let mut f = Tdd::one(&own);
    for clause in [[m as i32, x as i32], [x as i32, z as i32], [y as i32, y as i32]] {
        f = f.and_clause(clause).unwrap();
    }
    f.marginalize(|v| v == VarId(m)).unwrap();
    let mut f = f.condition_var(VarId(y), true).unwrap();
    let before = f.model_count().unwrap();
    let over = home.node_over(home.rank(VarId(x)).unwrap(), home.rank(VarId(z)).unwrap());
    let outcome = f.marginalize(under(&home, &[over]));
    assert_eq!(f.model_count().unwrap(), before, "after marginalize returned {outcome:?}");
}

/// `x1 or x2 or x3 or x4` on a balanced home, the node over x1 and x2 summed
/// out: a region beside the leaves of x3 and x4.
fn with_a_region() -> (Home, Tdd) {
    let home = Home::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&Vtree::projection(&home, [1, 2, 3, 4].map(VarId)).unwrap(), [1, 2, 3, 4]).unwrap();
    let (left, _) = home.vtree().children(home.vtree().root());
    f.marginalize(under(&home, &[left])).unwrap();
    assert!(f.vtree().bottomup().any(|t| f.vtree().node(t).is_region()), "the diagram has a region");
    (home, f)
}

#[test]
fn finish_refuses_pairs_that_name_a_region_as_a_literal() {
    let eng = Engine::new();
    let (_, f) = with_a_region();
    let vtree = Arc::clone(f.vtree());
    let root = vtree.root();
    let (l, r) = vtree.children(root);
    let (region, other) = if vtree.node(l).is_region() { (l, r) } else { (r, l) };
    assert!(vtree.node(region).is_region() && !vtree.node(other).is_leaf());
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    let below = b.push(&eng, other, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let pair = if region == l { ChildPair::new(POS_LEAF_IDX, below) } else { ChildPair::new(below, POS_LEAF_IDX) };
    let top = b.push(&eng, root, &[pair]).unwrap();
    quiet();
    let built = catch_unwind(AssertUnwindSafe(|| b.finish(TddNodeId { vtree: root, local: top }).map(|g| (g.model_count(), g.is_sat()))));
    eprintln!("finish at a region: {:?}", built.as_ref().map(|r| r.as_ref().map_err(|e| format!("{e:?}"))).map_err(|e| text(&**e)));
    assert!(matches!(built, Ok(Err(_))), "finish accepted a structural pair at a region: {:?}", built.map(|r| r.is_ok()).map_err(|e| text(&*e)));
}

/// Every public entry here answers or returns an error on a diagram with a
/// region.
#[test]
fn entries_on_a_diagram_with_a_region_do_not_panic() {
    quiet();
    let eng = Engine::new();
    let (home, f) = with_a_region();
    let other = Vtree::projection(&Home::new(Vtree::linear(4)), [3, 4].map(VarId)).unwrap();
    let x3 = Tdd::clause(&Vtree::projection(&home, [VarId(3)]).unwrap(), [3]).unwrap();
    let mut panics = Vec::new();
    let mut run = |name: &str, call: &mut dyn FnMut()| {
        if let Err(e) = catch_unwind(AssertUnwindSafe(call)) {
            panics.push(format!("{name}: {}", text(&*e)));
        }
    };
    run("model_count", &mut || drop(f.model_count()));
    run("is_sat", &mut || drop(f.is_sat()));
    run("support", &mut || drop(f.support()));
    run("implied_literals", &mut || drop(f.implied_literals()));
    run("satisfying_assignment", &mut || drop(f.satisfying_assignment()));
    run("counter", &mut || drop(f.counter().and_then(|mut c| c.model_count())));
    run("negate", &mut || drop(f.clone().negate().map(|g| g.model_count())));
    run("minimize", &mut || drop(f.clone().minimize()));
    run("trim", &mut || drop(f.clone().trim(&home)));
    run("condition_var", &mut || drop(f.clone().condition_var(VarId(3), true).map(|g| g.model_count())));
    run("condition_var summed", &mut || drop(f.clone().condition_var(VarId(1), true).map(|g| g.model_count())));
    run("exists_vars", &mut || drop(f.clone().exists_vars(&[VarId(4)]).map(|g| g.model_count())));
    run("exists_vars summed", &mut || drop(f.clone().exists_vars(&[VarId(2)]).map(|g| g.model_count())));
    run("and_clause", &mut || drop(f.clone().and_clause([3, -4]).map(|g| g.model_count())));
    run("and_clause summed", &mut || drop(f.clone().and_clause([1, -4]).map(|g| g.model_count())));
    run("and", &mut || drop(eng.and(&f, &f).map(|g| g.model_count())));
    run("and with x3", &mut || drop(eng.and(&f, &x3).map(|g| g.model_count())));
    run("or", &mut || drop(eng.or(f.clone(), f.clone()).map(|g| g.model_count())));
    run("equivalent", &mut || drop(eng.equivalent(&f, &f.clone())));
    run("substitute", &mut || drop(f.clone().substitute(&[(VarId(3), &x3)])));
    run("restructure_to", &mut || drop(f.clone().restructure_to(&other, usize::MAX)));
    run("write_tdd_binary", &mut || drop(tididi::io::write_tdd_binary(&mut Vec::new(), &f)));
    run("write_tdd", &mut || drop(tididi::io::write_tdd(&mut Vec::new(), &f)));
    run("Vtree::to_text", &mut || drop(f.vtree().to_text()));
    run("Vtree::join", &mut || drop(Vtree::join(f.vtree(), &Vtree::leaf(VarId(9)))));
    run("Vtree::project_to_vars", &mut || drop(f.vtree().project_to_vars(|v| (v.0 > 2).then(|| VarId(v.0 - 2)), 2)));
    run("Vtree::validate", &mut || drop(f.vtree().validate()));
    run("model_columns", &mut || drop(f.model_columns(&[&[VarId(3)], &[VarId(4)]])));
    run("ordered_models", &mut || drop(f.ordered_models(&[&[VarId(3)], &[VarId(4)]], &[false, false], 1, 10)));
    run("marginalize again", &mut || drop(f.clone().marginalize(|_| true)));
    run("marginalize_levels", &mut || {
        let mut g = f.clone();
        let root = g.vtree().root();
        drop(g.marginalize_levels(&[root]).map(|()| g.model_count()));
    });
    eprintln!("{} entries panic:\n{}", panics.len(), panics.join("\n"));
    assert!(panics.is_empty(), "{} entries panic:\n{}", panics.len(), panics.join("\n"));
}

/// A diagram placed on a vtree holding a region it has not summed out is
/// refused by every entry that builds the copy whole, naming the region;
/// conjoined with the diagram that summed the region, it answers.
#[test]
fn a_conjunction_with_a_variable_a_region_summed_out_commutes() {
    quiet();
    let eng = Engine::new();
    let (home, f) = with_a_region();
    let x3 = Tdd::clause(&Vtree::projection(&home, [VarId(3)]).unwrap(), [3]).unwrap();
    let both = eng.and(&x3, &f).unwrap();
    assert_eq!(both.model_count().unwrap(), eng.and(&f, &x3).unwrap().model_count().unwrap());
}

fn weights(n: u32, rng: &mut Rng) -> Weights {
    let mut w = || BigRational::new((1 + rng.below(6) as i64).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..n).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
    Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

/// The count, and the weighted value where weights are attached.
fn value(t: &Tdd) -> Result<String, String> {
    let read = || match t.weights() {
        Some(_) => format!("{:?}", t.weighted_value()),
        None => format!("{:?}", t.model_count()),
    };
    catch_unwind(AssertUnwindSafe(read)).map_err(|e| format!("PANIC reading: {}", text(&*e)))
}

/// Random sequences of sum-outs and other operations on a diagram on a
/// projection and on the same diagram on the home, where the sum-outs are
/// `marginalize_levels`: the counts and weighted values agree after each
/// step, and a sum-out changes neither.
#[test]
fn sum_outs_between_other_operations_agree_with_the_home() {
    quiet();
    let mut seen: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut steps_run = 0usize;
    for seed in 0..1500u64 {
        let mut rng = Rng(seed ^ 0x9e37_79b9);
        let n = 2 + rng.below(6) as u32;
        let home = match rng.below(3) {
            0 => Home::weighted(Vtree::random(n, seed), weights(n, &mut rng)).unwrap(),
            _ => Home::new(Vtree::random(n, seed)),
        };
        let vars: Vec<VarId> = (1..=n).filter(|_| rng.below(4) != 0).map(VarId).collect();
        if vars.is_empty() {
            continue;
        }
        let own = Vtree::projection(&home, vars.iter().copied()).unwrap();
        let (mut f, mut twin) = (Tdd::one(&own), Tdd::one(home.vtree()));
        let clause = |rng: &mut Rng| -> Vec<i32> {
            (0..1 + rng.below(3)).map(|_| vars[rng.below(vars.len() as u64) as usize].0 as i32 * if rng.coin() { 1 } else { -1 }).collect()
        };
        for _ in 0..1 + rng.below(4) {
            let c = clause(&mut rng);
            f = f.and_clause(&c).unwrap();
            twin = twin.and_clause(&c).unwrap();
        }
        let mut trace = format!("seed {seed} n {n} vars {:?}:", vars.iter().map(|v| v.0).collect::<Vec<_>>());
        let mut note = |what: String, trace: &str| seen.entry(what).or_insert((0, trace.to_string())).0 += 1;
        for _ in 0..2 + rng.below(4) {
            let var = VarId(1 + rng.below(u64::from(n)) as u32);
            let targets: Vec<VtreeIdx> = (0..1 + rng.below(2)).map(|_| VtreeIdx(rng.below(home.vtree().num_nodes() as u64) as u32)).collect();
            let (sign, c) = (rng.coin(), clause(&mut rng));
            let kind = rng.below(8);
            let name = ["marginalize", "marginalize", "marginalize", "condition", "exists", "and_clause", "minimize", "trim"][kind as usize];
            trace += &match kind {
                0..=2 => format!(" {name} {:?}", targets.iter().map(|h| home.rank_range(*h)).collect::<Vec<_>>()),
                3 | 4 => format!(" {name} x{}", var.0),
                5 => format!(" {name} {c:?}"),
                _ => format!(" {name}"),
            };
            let step = |t: &Tdd, on_home: bool| -> Result<Tdd, OperationError> {
                let mut t = t.clone();
                match kind {
                    0..=2 if on_home => t.marginalize_levels(&targets).map(|()| t),
                    0..=2 => t.marginalize(under(&home, &targets)).map(|()| t),
                    3 => t.condition_var(var, sign),
                    4 => t.exists_vars(&[var]),
                    5 => t.and_clause(&c),
                    6 => t.minimize().map(|()| t),
                    _ if on_home => Ok(t),
                    _ => t.trim(&home).map(|_| t),
                }
            };
            steps_run += 1;
            let before = value(&f);
            // A sum-out is also run in place, to see what a refusal leaves.
            if kind <= 2 {
                let mut g = f.clone();
                match catch_unwind(AssertUnwindSafe(|| g.marginalize(under(&home, &targets)))) {
                    Err(e) => {
                        note(format!("{name}: PANIC {}", text(&*e)), &trace);
                        break;
                    }
                    Ok(outcome) => {
                        if value(&g) != before {
                            note(format!("{name}: VALUE CHANGED, returned {}", if outcome.is_ok() { "Ok" } else { "Err" }), &trace);
                            break;
                        }
                    }
                }
            }
            let next = match catch_unwind(AssertUnwindSafe(|| step(&f, false))) {
                Err(e) => {
                    note(format!("{name}: PANIC {}", text(&*e)), &trace);
                    break;
                }
                Ok(Err(e)) => {
                    let e = format!("{e:?}");
                    note(format!("{name}: refused {}", e.split('(').next().unwrap_or("")), &trace);
                    continue;
                }
                Ok(Ok(next)) => next,
            };
            let Ok(Ok(next_twin)) = catch_unwind(AssertUnwindSafe(|| step(&twin, true))) else {
                note(format!("{name}: the home refuses or panics where the projection answers"), &trace);
                break;
            };
            (f, twin) = (next, next_twin);
            if f.weights().is_some() != twin.weights().is_some() {
                note(format!("{name}: WEIGHTS on one side only"), &trace);
                break;
            }
            match (value(&f), value(&twin)) {
                (Err(e), _) => {
                    note(format!("{name}: {e}"), &trace);
                    break;
                }
                (Ok(a), Ok(b)) if a != b => {
                    note(format!("{name}: DIFFERS from the home"), &format!("{trace} [{a} against {b}]"));
                    break;
                }
                _ => {}
            }
        }
    }
    let report: String = seen.iter().map(|(what, (count, first))| format!("{count:>6}  {what}\n        first: {first}\n")).collect();
    eprintln!("{steps_run} steps\n{report}");
    assert!(!report.contains("PANIC") && !report.contains("CHANGED") && !report.contains("DIFFERS"), "{report}");
}

#[allow(dead_code)]
fn unused(_: BigUint) {}
