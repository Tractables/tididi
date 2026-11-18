//! Operations on operands in every form a projection of a home vtree allows
//! (on the home, trimmed, on a wider projection, constants with and without
//! a leaf, two `Home` values of one vtree), checked against truth tables,
//! counts over the home and the canonical diagram of each result.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_bigint::BigUint;
use num_rational::BigRational;
use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use tididi::query::PinSemantics;
use tididi::restructure::search::{MinimizePairs, RotationSearchConfig};
use tididi::test_helpers::{eval, Lcg};
use tididi::query::Retention;
use tididi::vtree::{Home, VarId, Vtree, VtreeIdx};
use tididi::{Engine, Literal, OperationError, Tdd};

type Table = Vec<bool>;

fn table(t: &Tdd, n: u32) -> Table {
    if t.is_zero() {
        return vec![false; 1 << n];
    }
    if t.is_one() {
        return vec![true; 1 << n];
    }
    (0..1u32 << n)
        .map(|a| {
            let asn: Vec<bool> = (0..n).map(|i| a >> i & 1 == 1).collect();
            eval(t, &asn)
        })
        .collect()
}

fn zip(a: &Table, b: &Table, op: impl Fn(bool, bool) -> bool) -> Table {
    a.iter().zip(b).map(|(&x, &y)| op(x, y)).collect()
}

fn popcount(t: &Table) -> BigUint {
    BigUint::from(t.iter().filter(|&&b| b).count())
}

#[derive(Default)]
struct Log {
    seen: BTreeMap<String, (usize, String)>,
    case: String,
    runs: usize,
}

impl Log {
    fn note(&mut self, what: String) {
        // Refusals by design, then the defects `projection_defects.rs` holds
        // one test each for.
        let known = [
            "and_cube: refused",
            "rename_vars: refused VariableNotInVtree",
            "restructure_to: refused Variables",
            "and_indexed: refused VtreeMismatch",
            "and_indexed: PANIC index out of bounds: the len is 0",
            "and_marginalizing some levels",
        ];
        if known.iter().any(|k| what.starts_with(k)) {
            return;
        }
        let case = self.case.clone();
        self.seen.entry(what).or_insert((0, case)).0 += 1;
    }

    fn tdd(&mut self, n: u32, op: &str, call: impl FnOnce() -> Result<Tdd, OperationError>, expected: &Table) {
        self.runs += 1;
        match catch_unwind(AssertUnwindSafe(call)) {
            Err(e) => self.note(format!("{op}: PANIC {}", text(&*e))),
            Ok(Err(e)) => self.note(format!("{op}: refused {e:?}")),
            Ok(Ok(got)) => {
                match catch_unwind(AssertUnwindSafe(|| (table(&got, n), got.model_count(), checks(&got)))) {
                    Err(e) => self.note(format!("{op}: PANIC reading the result: {}", text(&*e))),
                    Ok((t, count, checks)) => {
                        if &t != expected {
                            self.note(format!("{op}: WRONG function"));
                        } else if count != Ok(popcount(expected)) {
                            self.note(format!("{op}: WRONG count {count:?}"));
                        } else if let Err(e) = checks {
                            self.note(format!("{op}: INVALID {}", e.split(':').next().unwrap_or("")));
                        }
                    }
                }
            }
        }
    }

    fn value<T: PartialEq + std::fmt::Debug>(&mut self, op: &str, call: impl FnOnce() -> Result<T, OperationError>, expected: T) {
        self.runs += 1;
        match catch_unwind(AssertUnwindSafe(call)) {
            Err(e) => self.note(format!("{op}: PANIC {}", text(&*e))),
            Ok(Err(e)) => self.note(format!("{op}: refused {e:?}")),
            Ok(Ok(got)) => {
                if got != expected {
                    self.note(format!("{op}: WRONG value"));
                }
            }
        }
    }

    fn report(&self) -> String {
        let mut out = format!("{} runs\n", self.runs);
        for (what, (k, case)) in &self.seen {
            out += &format!("{k:6}  {what}\n        first: {case}\n");
        }
        out
    }
}

fn text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default()
}

fn subset(n: u32, rng: &mut Lcg) -> Vec<u32> {
    let s: Vec<u32> = (1..=n).filter(|_| rng.coin()).collect();
    if s.is_empty() { vec![1 + rng.below(u64::from(n)) as u32] } else { s }
}

fn projection(home: &Home, vars: &[u32]) -> Arc<Vtree> {
    Vtree::projection(home, vars.iter().map(|&v| VarId(v))).unwrap()
}

/// A diagram of a random function in a random form, and what the form is.
fn operand(eng: &Engine, homes: &[Home], n: u32, rng: &mut Lcg) -> (Tdd, String) {
    let (f, name, _) = operand_and_twin(eng, homes, n, rng);
    (f, name)
}

/// [`operand`], and the same function built on the home's root vtree.
fn operand_and_twin(eng: &Engine, homes: &[Home], n: u32, rng: &mut Lcg) -> (Tdd, String, Tdd) {
    let home = &homes[rng.below(homes.len() as u64) as usize];
    let root = home.vtree();
    let kind = rng.below(14);
    let s = subset(n, rng);
    match kind {
        0 => return (Tdd::one(root), "one on the home".into(), Tdd::one(root)),
        1 => return (Tdd::zero(root), "zero on the home".into(), Tdd::zero(root)),
        2 => return (Tdd::one(&Vtree::empty_projection(home)), "one on no leaf".into(), Tdd::one(root)),
        3 => return (Tdd::zero(&Vtree::empty_projection(home)), "zero on no leaf".into(), Tdd::zero(root)),
        4 => return (Tdd::one(&projection(home, &s)), format!("one on {s:?}"), Tdd::one(root)),
        5 => return (Tdd::zero(&projection(home, &s)), format!("zero on {s:?}"), Tdd::zero(root)),
        _ => {}
    }
    let form = rng.below(5);
    let (vtree, name) = match form {
        0 => (Arc::clone(root), "home".to_string()),
        1 => (Arc::clone(root), "home trimmed".to_string()),
        2 => (projection(home, &s), format!("on {s:?}")),
        3 => {
            let mut wide = s.clone();
            wide.extend(subset(n, rng));
            wide.sort_unstable();
            wide.dedup();
            (projection(home, &wide), format!("on {wide:?}"))
        }
        _ => (projection(home, &(1..=n).collect::<Vec<_>>()), "on every leaf".to_string()),
    };
    let mut clauses = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let width = 1 + rng.below(3);
        let clause: Vec<i32> = (0..width)
            .map(|_| {
                let var = s[rng.below(s.len() as u64) as usize] as i32;
                if rng.coin() { var } else { -var }
            })
            .collect();
        clauses.push(clause);
    }
    let mut twin = Tdd::one(root);
    for clause in &clauses {
        twin = eng.and_clause(twin, clause).unwrap();
    }
    let mut f = Tdd::one(&vtree);
    for clause in &clauses {
        f = if rng.coin() { eng.and_clause(f, clause).unwrap() } else { eng.and(f, Tdd::clause(&vtree, clause).unwrap()).unwrap() };
    }
    let mut name = format!("{clauses:?} {name}");
    if rng.coin() {
        f.minimize().unwrap();
        name += " minimized";
    }
    if form == 1 || rng.below(4) == 0 {
        eng.trim(&mut f, home).unwrap();
        name += " trimmed";
    }
    (f, name, twin)
}

fn binary(log: &mut Log, eng: &Engine, n: u32, f: &Tdd, g: &Tdd, h: &Tdd, rng: &mut Lcg) {
    let (tf, tg, th) = (table(f, n), table(g, n), table(h, n));
    let and = zip(&tf, &tg, |x, y| x && y);
    log.tdd(n, "and", || eng.and(f, g), &and);
    log.tdd(n, "and owned", || eng.and(f.clone(), g.clone()), &and);
    log.tdd(n, "and owned borrowed", || eng.and(f.clone(), g), &and);
    log.tdd(n, "and_restoring", || eng.and_restoring(f.clone(), g).map_err(|r| r.error), &and);
    log.tdd(n, "or", || eng.or(f.clone(), g.clone()), &zip(&tf, &tg, |x, y| x || y));
    log.tdd(n, "and_not", || eng.and_not(f.clone(), g.clone()), &zip(&tf, &tg, |x, y| x && !y));
    log.tdd(n, "xor", || eng.xor(f.clone(), g.clone()), &zip(&tf, &tg, |x, y| x != y));
    log.tdd(n, "add", || eng.add(eng.and_not(f.clone(), g.clone())?, g.clone()), &zip(&tf, &tg, |x, y| x || y));
    let ite: Table = (0..tf.len()).map(|i| if tf[i] { tg[i] } else { th[i] }).collect();
    log.tdd(n, "ite", || eng.ite(f.clone(), g.clone(), h.clone()), &ite);
    let or3: Table = (0..tf.len()).map(|i| tf[i] || tg[i] || th[i]).collect();
    log.tdd(n, "or_many", || eng.or_many(vec![f.clone(), g.clone(), h.clone()]), &or3);
    log.tdd(n, "nor_many", || eng.nor_many(vec![f.clone(), g.clone(), h.clone()]), &or3.iter().map(|&b| !b).collect());
    log.value("equivalent", || eng.equivalent(f, g), tf == tg);
    log.value("implies", || eng.implies(f, g), tf.iter().zip(&tg).all(|(&x, &y)| !x || y));
    let vars: Vec<u32> = (1..=n).filter(|_| rng.coin()).collect();
    let exists = |t: &Table| -> Table {
        let mut t = t.clone();
        for &v in &vars {
            let bit = 1usize << (v - 1);
            t = (0..t.len()).map(|a| t[a | bit] || t[a & !bit]).collect();
        }
        t
    };
    let var_ids: Vec<VarId> = vars.iter().map(|&v| VarId(v)).collect();
    log.tdd(n, "and_exists", || eng.and_exists(f.clone(), g.clone(), &var_ids), &exists(&and));
    log.tdd(n, "and_in_place", || eng.and_in_place(f.clone(), g), &and);
    log.tdd(n, "and_in_place_loose", || eng.and_in_place_loose(f.clone(), g), &and);
    log.tdd(n, "and_indexed", || eng.and_indexed(f, g, &eng.pair_index(g)?), &and);
    log.value("and_model_count", || eng.and_model_count(f.clone(), g.clone(), &[]), popcount(&and));
    log.value("marginal_product_count", || eng.marginal_product_count(f, g), popcount(&and));
    log.tdd(n, "and_marginalizing", || eng.and_marginalizing(f, g, &[]), &and);
    let home = HOME.with_borrow(Clone::clone).expect("the case names its home");
    let targets: Vec<VtreeIdx> = f.vtree().bottomup().filter(|_| rng.coin()).collect();
    let summed = || eng.and_marginalizing(f, g, &targets);
    log.value("and_marginalizing some levels: count", || summed()?.model_count(), popcount(&and));
    log.value("and_marginalizing some levels: is_sat", || summed()?.is_sat(), and.iter().any(|&v| v));
    log.value("and_marginalizing some levels, trim: count", || summed().and_then(|mut t| { let _ = t.trim(&home); t.model_count() }), popcount(&and));
    log.value("and_marginalizing some levels, minimize, trim: count", || summed().and_then(|mut t| { t.minimize()?; let _ = t.trim(&home); t.model_count() }), popcount(&and));
    log.value("and_marginalizing some levels, counter", || summed().and_then(|t| t.counter()?.model_count()), popcount(&and));
    log.value("and_model_count some levels", || eng.and_model_count(f.clone(), g.clone(), &targets), popcount(&and));
    let (ws, literal_weights) = weights(n, rng);
    let (weighted, _) = tididi::test_helpers::weighted_sum(&and, &literal_weights);
    let with_weights = |t: &Tdd| { let mut t = t.clone(); t.set_weights(ws.clone()).map(|()| t) };
    let exact = |t: &Tdd| t.weighted_value().map(|v| v.map(|v| tididi::test_helpers::exact_weight(&v)));
    for (name, targets) in [("weighted and", &[][..]), ("weighted and_marginalizing some levels", &targets[..])] {
        log.same(
            name,
            || match eng.and_marginalizing(with_weights(f).unwrap(), with_weights(g).unwrap(), targets) {
                Err(OperationError::WeightsAcrossProjections) => Ok(Some(weighted.clone())),
                other => other.and_then(|t| exact(&t)),
            },
            || Ok::<_, OperationError>(Some(weighted.clone())),
        );
        log.same(
            &format!("{name}, trim"),
            || match eng.and_marginalizing(with_weights(f).unwrap(), with_weights(g).unwrap(), targets) {
                Err(OperationError::WeightsAcrossProjections) => Ok(Some(weighted.clone())),
                other => other.and_then(|mut t| { let _ = t.trim(&home); exact(&t) }),
            },
            || Ok::<_, OperationError>(Some(weighted.clone())),
        );
    }
    log.tdd(n, "restrict_to_care", || eng.and(eng.restrict_to_care(f.clone(), g.clone())?.into_tdd(), g), &and);
    log.tdd(n, "restrict_to_care_bounded", || eng.and(eng.restrict_to_care_bounded(f.clone(), g, 1 + rng.below(64))?.into_tdd(), g), &and);
    let var = 1 + rng.below(u64::from(n)) as u32;
    let bit = 1usize << (var - 1);
    let substituted: Table = (0..tf.len()).map(|a| if tg[a] { tf[a | bit] } else { tf[a & !bit] }).collect();
    log.tdd(n, "substitute", || eng.substitute(f.clone(), &[(VarId(var), g)]), &substituted);
}

fn quiet() {
    std::panic::set_hook(Box::new(|_| {}));
}

#[test]
fn operations_of_two_diagrams_in_every_form() {
    quiet();
    let eng = Engine::new();
    let mut log = Log::default();
    for seed in 0..400u64 {
        let mut rng = Lcg::new(seed ^ 0x51ed_270b);
        let n = 1 + rng.below(6) as u32;
        let root = Arc::new(Vtree::random(n, seed));
        // Two `Home` values of one root vtree.
        let homes = [Home::new(Arc::clone(&root)), Home::new(Arc::clone(&root))];
        HOME.set(Some(homes[0].clone()));
        let (f, nf) = operand(&eng, &homes, n, &mut rng);
        let (g, ng) = operand(&eng, &homes, n, &mut rng);
        let (h, nh) = operand(&eng, &homes, n, &mut rng);
        log.case = format!("seed {seed} n {n}: f {nf}; g {ng}; h {nh}");
        binary(&mut log, &eng, n, &f, &g, &h, &mut rng);
    }
    let report = log.report();
    eprintln!("{report}");
    assert!(log.seen.is_empty(), "{report}");
}

impl Log {
    /// A result compared with the same call on the twin.
    fn same<T: PartialEq + std::fmt::Debug, E: std::fmt::Debug>(&mut self, op: &str, on_form: impl FnOnce() -> Result<T, E>, on_home: impl FnOnce() -> Result<T, E>) {
        self.runs += 1;
        let got = match catch_unwind(AssertUnwindSafe(on_form)) {
            Err(e) => return self.note(format!("{op}: PANIC {}", text(&*e))),
            Ok(got) => got,
        };
        let expected = match catch_unwind(AssertUnwindSafe(on_home)) {
            Err(e) => return self.note(format!("{op}: PANIC on the home {}", text(&*e))),
            Ok(expected) => expected,
        };
        match (got, expected) {
            (Ok(g), Ok(e)) if g == e => {}
            (Ok(g), Ok(e)) => self.note(format!("{op}: WRONG {} for {}", short(&g), short(&e))),
            (Err(g), Ok(_)) => self.note(format!("{op}: refused {g:?}")),
            (Ok(_), Err(e)) => self.note(format!("{op}: answers where the home refuses {e:?}")),
            (Err(_), Err(_)) => {}
        }
    }

    fn any<E: std::fmt::Debug>(&mut self, n: u32, op: &str, call: impl FnOnce() -> Result<Tdd, E>, expected: &Table) {
        self.runs += 1;
        match catch_unwind(AssertUnwindSafe(call)) {
            Err(e) => self.note(format!("{op}: PANIC {}", text(&*e))),
            Ok(Err(e)) => self.note(format!("{op}: refused {}", short(&e))),
            Ok(Ok(got)) => match catch_unwind(AssertUnwindSafe(|| (table(&got, n), got.model_count(), checks(&got)))) {
                Err(e) => self.note(format!("{op}: PANIC reading the result: {}", text(&*e))),
                Ok((t, count, checks)) => {
                    if &t != expected {
                        self.note(format!("{op}: WRONG function"));
                    } else if count != Ok(popcount(expected)) {
                        self.note(format!("{op}: WRONG count {count:?} for {}", popcount(expected)));
                    } else if let Err(e) = checks {
                        self.note(format!("{op}: INVALID {}", e.split(':').next().unwrap_or("")));
                    }
                }
            },
        }
    }
}

fn short(x: &impl std::fmt::Debug) -> String {
    let s = format!("{x:?}");
    s.chars().take(80).collect()
}

thread_local! {
    /// The home the results of the running case are judged on.
    static HOME: std::cell::RefCell<Option<Home>> = const { std::cell::RefCell::new(None) };
}

fn size(t: &Tdd) -> (usize, usize, u32) {
    (t.node_count(), t.pair_count(), t.vtree().num_leaves())
}

/// Minimized and trimmed, `t` is the canonical diagram of its function on
/// the case's home.
fn canonical(t: &Tdd) -> Result<(), String> {
    use tididi::test_helpers::check::check_canonicity;
    let Some(home) = HOME.with_borrow(Clone::clone) else { return Ok(()) };
    let on = t.vtree().home().map_or(t.vtree(), Home::vtree);
    if !Arc::ptr_eq(on, home.vtree()) {
        return Ok(());
    }
    let n = home.vtree().num_vars();
    let truth = table(t, n);
    let vars: Vec<VarId> = (1..=n).map(VarId).collect();
    let mut expected = tididi::test_helpers::or_of_cubes(home.vtree(), &vars, |row| truth[row]);
    expected.minimize().unwrap();
    expected.trim(&home).unwrap();
    let mut got = t.clone();
    got.minimize().map_err(|e| format!("minimize refused: {e:?}"))?;
    got.trim(&home).map_err(|e| format!("trim refused: {e:?}"))?;
    if size(&got) != size(&expected) {
        return Err(format!("not canonical after minimize and trim: {:?} for {:?}", size(&got), size(&expected)));
    }
    if got.vtree().num_nodes() != 0 {
        check_canonicity(&got, 3).map_err(|e| format!("canonicity: {e}"))?;
    }
    let mut again = t.clone();
    again.trim(&home).map_err(|e| format!("trim refused: {e:?}"))?;
    again.minimize().map_err(|e| format!("minimize refused: {e:?}"))?;
    again.trim(&home).map_err(|e| format!("trim refused: {e:?}"))?;
    if size(&again) != size(&expected) {
        return Err(format!("not canonical after trim, minimize and trim: {:?} for {:?}", size(&again), size(&expected)));
    }
    Ok(())
}

fn checks(t: &Tdd) -> Result<(), String> {
    use tididi::test_helpers::check::{check_determinism, check_no_false_nodes, validate_vtree_structure};
    canonical(t)?;
    if t.vtree().num_nodes() == 0 {
        return Ok(());
    }
    validate_vtree_structure(t).map_err(|e| format!("structure: {e}"))?;
    if !t.is_zero() {
        check_no_false_nodes(t).map_err(|e| format!("false nodes: {e}"))?;
    }
    check_determinism(t).map_err(|e| format!("determinism: {e}"))
}

fn weights(n: u32, rng: &mut Lcg) -> (WeightStore, Vec<LiteralWeights<BigRational>>) {
    let mut w = || BigRational::new((rng.below(7) as i64 - 2).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..n).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
    (WeightStore::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational), literals)
}

#[allow(clippy::too_many_lines)]
fn unary(log: &mut Log, eng: &Engine, home: &Home, n: u32, f: &Tdd, hf: &Tdd, rng: &mut Lcg) {
    let tf = table(f, n);
    assert_eq!(tf, table(hf, n), "the twin is the same function: {}", log.case);
    let size = tf.len();
    let var = |rng: &mut Lcg| 1 + rng.below(u64::from(n)) as u32;
    let lit = |v: u32, sign: bool| Literal { var: VarId(v), sign };
    let (x, b) = (var(rng), rng.coin());
    let bit = 1usize << (x - 1);
    let cond: Table = (0..size).map(|a| tf[if b { a | bit } else { a & !bit }]).collect();
    let some: Vec<u32> = (1..=n).filter(|_| rng.coin()).collect();
    let some_ids: Vec<VarId> = some.iter().map(|&v| VarId(v)).collect();
    let quantify = |all: bool| -> Table {
        let mut t = tf.clone();
        for &v in &some {
            let bit = 1usize << (v - 1);
            t = (0..size).map(|a| if all { t[a | bit] && t[a & !bit] } else { t[a | bit] || t[a & !bit] }).collect();
        }
        t
    };

    log.any(n, "negate", || eng.negate(f), &tf.iter().map(|&v| !v).collect());
    log.any(n, "condition_var", || eng.condition_var(f.clone(), VarId(x), b), &cond);
    log.any(n, "condition", || eng.condition(f.clone(), [lit(x, b)]), &cond);
    log.any(n, "condition both ways", || eng.condition(f.clone(), [lit(x, b), lit(x, !b)]), &vec![false; size]);
    log.any(n, "exists_var", || eng.exists_var(f.clone(), VarId(x)), &(0..size).map(|a| tf[a | bit] || tf[a & !bit]).collect());
    log.any(n, "exists_vars", || eng.exists_vars(f.clone(), &some_ids), &quantify(false));
    log.any(n, "forall_vars", || eng.forall_vars(f.clone(), &some_ids), &quantify(true));
    let clause: Vec<Literal> = (0..1 + rng.below(3)).map(|_| lit(var(rng), rng.coin())).collect();
    let holds = |a: usize, l: &Literal| (a >> (l.var.0 - 1) & 1 == 1) == l.sign;
    log.any(n, "and_clause", || eng.and_clause(f.clone(), &clause), &(0..size).map(|a| tf[a] && clause.iter().any(|l| holds(a, l))).collect());
    log.any(n, "or_cube", || eng.or_cube(f.clone(), &clause), &(0..size).map(|a| tf[a] || clause.iter().all(|l| holds(a, l))).collect());
    log.any(n, "and_cube", || eng.and_cube(f.clone(), clause.clone()), &(0..size).map(|a| tf[a] && clause.iter().all(|l| holds(a, l))).collect());
    let (p, q) = (var(rng), var(rng));
    if p != q {
        let swap = |a: usize| {
            let (bp, bq) = (a >> (p - 1) & 1, a >> (q - 1) & 1);
            a & !(1 << (p - 1)) & !(1 << (q - 1)) | bq << (p - 1) | bp << (q - 1)
        };
        log.any(n, "rename_vars", || eng.rename_vars(f.clone(), &[(VarId(p), VarId(q)), (VarId(q), VarId(p))]), &(0..size).map(|a| tf[swap(a)]).collect());
    }
    let mutate = |op: &dyn Fn(&mut Tdd) -> Result<(), OperationError>| {
        let mut t = f.clone();
        op(&mut t).map(|()| t)
    };
    log.any(n, "minimize", || mutate(&|t| t.minimize()), &tf);
    log.any(n, "trim", || mutate(&|t| eng.trim(t, home).map(drop)), &tf);
    log.any(n, "minimize, trim", || mutate(&|t| t.minimize().and_then(|()| eng.trim(t, home).map(drop))), &tf);
    log.any(n, "trim, minimize", || mutate(&|t| eng.trim(t, home).and_then(|_| t.minimize())), &tf);
    log.any(n, "reduce", || mutate(&|t| t.reduce(tididi::reduce::ReductionPlan::default())), &tf);
    log.any(n, "renumber_top_down", || f.renumber_top_down(), &tf);
    log.any(n, "counts, trim", || mutate(&|t| t.attach_level_counts().and_then(|()| eng.trim(t, home).map(drop))), &tf);
    log.any(n, "counts, minimize, trim", || mutate(&|t| t.attach_level_counts().and_then(|()| t.minimize()).and_then(|()| eng.trim(t, home).map(drop))), &tf);
    log.any(n, "minimize, counts, trim", || mutate(&|t| t.minimize().and_then(|()| t.attach_level_counts()).and_then(|()| eng.trim(t, home).map(drop))), &tf);
    // A canonical trimmed diagram is the same whichever form it came from.
    log.same(
        "canonical size",
        || mutate(&|t| t.minimize().and_then(|()| eng.trim(t, home).map(drop))).map(|t| (t.node_count(), t.pair_count(), t.vtree().num_leaves())),
        || {
            let mut t = hf.clone();
            t.minimize()?;
            eng.trim(&mut t, home)?;
            Ok((t.node_count(), t.pair_count(), t.vtree().num_leaves()))
        },
    );

    // Queries.
    log.value("model_count", || f.model_count(), popcount(&tf));
    log.value("is_sat", || f.is_sat(), tf.iter().any(|&v| v));
    let depends: Vec<VarId> = (1..=n).filter(|&v| (0..size).any(|a| tf[a] != tf[a ^ 1 << (v - 1)])).map(VarId).collect();
    log.value("support", || f.support().map(|mut s| { s.sort_unstable(); s }), depends);
    let sorted = |mut l: Vec<(VarId, bool)>| { l.sort_unstable(); l };
    log.same("implied_literals", || f.implied_literals().map(|l| sorted(l.iter().map(|l| (l.var, l.sign)).collect())), || hf.implied_literals().map(|l| sorted(l.iter().map(|l| (l.var, l.sign)).collect())));
    log.value(
        "satisfying_assignment",
        || f.satisfying_assignment().map(|m| m.map(|m| {
            // Unnamed variables are free: both completions hold.
            let fixed: usize = m.iter().filter(|l| l.sign).map(|l| 1usize << (l.var.0 - 1)).sum();
            let named: usize = m.iter().map(|l| 1usize << (l.var.0 - 1)).sum();
            (0..size).filter(|a| a & named == fixed).all(|a| tf[a])
        })),
        tf.iter().any(|&v| v).then_some(true),
    );
    log.same("projected_model_count", || f.projected_model_count(&some_ids), || hf.projected_model_count(&some_ids));
    let (ws, literal_weights) = weights(n, rng);
    let algebra = ws.algebra().clone();
    let (weighted, _) = tididi::test_helpers::weighted_sum(&tf, &literal_weights);
    log.value("evaluate", || f.evaluate(&algebra), weighted.clone());
    let weighted_value = |t: &Tdd| {
        let mut w = t.clone();
        w.set_weights(ws.clone()).map_err(|e| format!("{e:?}"))?;
        w.weighted_value().map(|v| v.map(|v| tididi::test_helpers::exact_weight(&v))).map_err(|e| format!("{e:?}"))
    };
    log.same("weighted_value", || weighted_value(f), || weighted_value(hf));
    let levels: Vec<VtreeIdx> = f.vtree().bottomup().filter(|_| rng.coin()).collect();
    let summed = |weights: bool| -> Result<Tdd, OperationError> {
        let mut t = f.clone();
        if weights {
            t.set_weights(ws.clone()).expect("weights for the id space");
        }
        t.marginalize_levels(&levels)?;
        Ok(t)
    };
    let exact = |t: &Tdd| t.weighted_value().map(|v| v.map(|v| tididi::test_helpers::exact_weight(&v)));
    log.value("marginalize_levels: count", || summed(false)?.model_count(), popcount(&tf));
    log.value("marginalize_levels: counter", || summed(false)?.counter()?.model_count(), popcount(&tf));
    // A refused trim leaves the diagram and its count as they were.
    log.value("marginalize_levels, trim: count", || summed(false).and_then(|mut t| { let _ = t.trim(home); t.model_count() }), popcount(&tf));
    log.value("marginalize_levels, minimize, trim: count", || summed(false).and_then(|mut t| { t.minimize()?; let _ = t.trim(home); t.model_count() }), popcount(&tf));
    log.value("marginalize_levels weighted", || exact(&summed(true)?), Some(weighted.clone()));
    log.value("marginalize_levels weighted, trim", || summed(true).and_then(|mut t| { let _ = t.trim(home); exact(&t) }), Some(weighted.clone()));
    log.value(
        "marginalize_levels, stored and read: count",
        || {
            let t = summed(false)?;
            let mut bytes = Vec::new();
            if tididi::io::write_tdd_binary(&mut bytes, &t).is_err() {
                return Ok(popcount(&tf));
            }
            tididi::io::read_tdd_binary(&mut bytes.as_slice(), t.vtree()).map_err(|_| OperationError::VtreeMismatch)?.model_count()
        },
        popcount(&tf),
    );
    for (name, minimized) in [("weighted, trim", false), ("weighted, minimize, trim", true)] {
        log.same(
            name,
            || {
                let mut w = f.clone();
                w.set_weights(ws.clone()).map_err(|e| format!("{e:?}"))?;
                if minimized {
                    w.minimize().map_err(|e| format!("{e:?}"))?;
                }
                eng.trim(&mut w, home).map_err(|e| format!("{e:?}"))?;
                w.weighted_value().map(|v| v.map(|v| tididi::test_helpers::exact_weight(&v))).map_err(|e| format!("{e:?}"))
            },
            || Ok(Some(weighted.clone())),
        );
    }
    for convention in [PinSemantics::Cofactor, PinSemantics::Evidence] {
        let pins: Vec<(VarId, Option<bool>)> = (0..rng.below(3)).map(|_| (VarId(var(rng)), Some(rng.coin()))).collect();
        let cols = vec![VarId(var(rng)), VarId(var(rng))];
        let count = |t: &Tdd, counts: bool| -> Result<_, OperationError> {
            let mut t = t.clone();
            if counts {
                t.attach_level_counts()?;
            }
            let mut c = t.counter_with(Retention::All, convention)?;
            let before = c.model_count()?;
            c.set_pins(&pins)?;
            let pinned = c.model_count()?;
            let cells = if cols[0] == cols[1] { Vec::new() } else { c.count_table(&cols)? };
            c.clear_pins();
            Ok((before, pinned, cells, c.model_count()?))
        };
        log.same("counter", || count(f, false), || count(hf, false));
        log.same("counter with level counts", || count(f, true), || count(hf, false));
    }
    let pins: Vec<(VarId, Option<bool>)> = (0..1 + rng.below(3)).map(|_| (VarId(var(rng)), Some(rng.coin()))).collect();
    let value = |t: &Tdd| -> Result<_, OperationError> {
        let mut e = t.evaluator(algebra.clone())?;
        let before = e.value()?;
        e.set_pins(&pins)?;
        Ok((before, e.value()?))
    };
    log.same("evaluator", || value(f), || value(hf));
    log.same("counts then count", || mutate(&|t| t.attach_level_counts()).and_then(|t| t.model_count()), || Ok(popcount(&tf)));
    log.same(
        "counts, trim then count",
        || mutate(&|t| t.attach_level_counts().and_then(|()| eng.trim(t, home).map(drop))).and_then(|t| Ok((t.model_count()?, t.counter()?.model_count()?))),
        || Ok((popcount(&tf), popcount(&tf))),
    );

    // Serialization against the diagram's own vtree.
    log.any(n, "text", || {
        let mut text = Vec::new();
        tididi::io::write_tdd(&mut text, f)?;
        tididi::io::read_tdd(&mut text.as_slice(), f.vtree())
    }, &tf);
    log.any(n, "binary", || {
        let mut bytes = Vec::new();
        tididi::io::write_tdd_binary(&mut bytes, f)?;
        tididi::io::read_tdd_binary(&mut bytes.as_slice(), f.vtree())
    }, &tf);

    // Moves.
    log.any(n, "transport onto its home", || eng.transport(f, home, |v| v, 0).map(|r| r.0).map_err(|r| r.error), &tf);
    let other = Home::new(Vtree::random(n, rng.below(1000)));
    let rotating = |g: Cow<'_, Tdd>, to: &Home| eng.transport(g, to, |v| v, usize::MAX).map(|r| r.0).map_err(|r| r.error);
    log.any(n, "transport with rotations", || rotating(Cow::Borrowed(f), &other), &tf);
    log.any(n, "transport with rotations and back", || rotating(Cow::Owned(rotating(Cow::Borrowed(f), &other)?), home), &tf);
    log.same(
        "transport refused hands the diagram back",
        || match eng.transport(f, &other, |v| v, 0) {
            Ok(_) => Ok::<_, ()>(true),
            Err(r) => Ok(Arc::ptr_eq(r.tdd.vtree(), f.vtree()) && table(&r.tdd, n) == tf && r.tdd.pair_count() == f.pair_count()),
        },
        || Ok(true),
    );
    for bound in [1, 4, 8, 12, 16, 24, 32, 48] {
        log.same(
            "transport refused under a bound hands back the diagram as it was",
            || match eng.transport(f.clone(), &other, |v| v, bound) {
                Ok(_) => Ok::<_, ()>(true),
                Err(r) => Ok(Arc::ptr_eq(r.tdd.vtree(), f.vtree()) && table(&r.tdd, n) == tf && r.tdd.pair_count() == f.pair_count()),
            },
            || Ok(true),
        );
    }
    let before = (**home.vtree()).clone();
    log.any(n, "rotation_search", || mutate(&|t| t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(drop)), &tf);
    assert!(home.vtree().same_tree(&before), "a search changed the home: {}", log.case);
    log.any(n, "rotation_search, transport back", || {
        let mut t = f.clone();
        t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map_err(|e| format!("{e:?}"))?;
        eng.transport(t, home, |v| v, usize::MAX).map(|r| r.0).map_err(|r| format!("{:?}", r.error))
    }, &tf);
    let target = Arc::new(Vtree::random(n, rng.below(1000)));
    log.any(n, "restructure_to", || {
        let mut t = f.clone();
        t.restructure_to(&target, usize::MAX).map(|_| t)
    }, &tf);
    log.any(n, "embed", || f.embed(home.vtree(), |v| v).map(|r| r.0), &tf);
}

#[test]
fn operations_of_one_diagram_in_every_form() {
    quiet();
    let eng = Engine::new();
    let mut log = Log::default();
    for seed in 0..600u64 {
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        let n = 1 + rng.below(6) as u32;
        let root = Arc::new(Vtree::random(n, seed));
        let homes = [Home::new(Arc::clone(&root))];
        HOME.set(Some(homes[0].clone()));
        let (f, nf, hf) = operand_and_twin(&eng, &homes, n, &mut rng);
        log.case = format!("seed {seed} n {n}: f {nf}");
        unary(&mut log, &eng, &homes[0], n, &f, &hf, &mut rng);
    }
    let report = log.report();
    eprintln!("{report}");
    assert!(log.seen.is_empty(), "{report}");
}

#[test]
fn transports_in_every_form() {
    quiet();
    let eng = Engine::new();
    let mut log = Log::default();
    for seed in 0..400u64 {
        let mut rng = Lcg::new(seed ^ 0x7f4a_7c15);
        let n = [2u32, 4, 8][rng.below(3) as usize];
        let home = Home::new(Vtree::balanced(n));
        HOME.set(Some(home.clone()));
        let (f, nf, _) = operand_and_twin(&eng, std::slice::from_ref(&home), n, &mut rng);
        // On a balanced vtree of 2^k leaves, exchanging the leaves whose
        // zero-based positions differ in the bits of `mask` mirrors the
        // levels of those bits.
        let mask = rng.below(u64::from(n)) as u32;
        let positions: Vec<VarId> = home.vtree().leaf_bottomup().map(|(_, var)| var).collect();
        let mut in_order = positions.clone();
        in_order.sort_by_key(|&v| home.rank(v).unwrap());
        let map = |v: VarId| in_order[(home.rank(v).unwrap() ^ mask) as usize];
        log.case = format!("seed {seed} n {n} mask {mask}: f {nf}");
        let tf = table(&f, n);
        let moved: Table = (0..tf.len())
            .map(|a| {
                // Variable v of f reads the value of map(v).
                let source: usize = (1..=n).filter(|&v| a >> (map(VarId(v)).0 - 1) & 1 == 1).map(|v| 1usize << (v - 1)).sum();
                tf[source]
            })
            .collect();
        let moving = |g, to: &Home| eng.transport(g, to, map, 0).map(|r| r.0).map_err(|r| format!("{:?}", r.error));
        log.any(n, "transport", || moving(Cow::Borrowed(&f), &home), &moved);
        log.any(n, "transport owned", || moving(Cow::Owned(f.clone()), &home), &moved);
        log.any(n, "minimize, transport", || {
            let mut g = f.clone();
            g.minimize().map_err(|e| format!("{e:?}"))?;
            let g = moving(Cow::Owned(g), &home)?;
            if g.vtree().num_nodes() != 0 {
                tididi::test_helpers::check::check_canonicity(&g, 3)?;
            }
            Ok::<_, String>(g)
        }, &moved);
        log.any(n, "counts, transport", || {
            let mut g = f.clone();
            g.attach_level_counts().map_err(|e| format!("{e:?}"))?;
            let g = moving(Cow::Owned(g), &home)?;
            let mut counter = g.counter().map_err(|e| format!("{e:?}"))?;
            if counter.model_count() != g.model_count() {
                return Err("the counter and the kept counts differ".to_string());
            }
            drop(counter);
            Ok(g)
        }, &moved);
        // A renaming carries the table only where it weighs each image as
        // the variable it renames.
        let (ws, literal_weights) = weights(n, &mut rng);
        let (weighted, _) = tididi::test_helpers::weighted_sum(&moved, &literal_weights);
        let kept = f.vtree().leaf_bottomup().all(|(_, v)| literal_weights[v.idx()] == literal_weights[map(v).idx()]);
        log.same(
            "weighted, transport",
            || {
                let mut w = f.clone();
                w.set_weights(ws.clone()).map_err(|e| format!("{e:?}"))?;
                let w = moving(Cow::Owned(w), &home)?;
                w.weighted_value().map(|v| v.map(|v| tididi::test_helpers::exact_weight(&v))).map_err(|e| format!("{e:?}"))
            },
            || match kept {
                true => Ok(Some(weighted.clone())),
                false => Err("Weights(IncompatibleWeights)".to_string()),
            },
        );
        // Onto a wider home, shifted.
        let wide = Home::new(Vtree::balanced(2 * n));
        let shift = if rng.coin() { n } else { 0 };
        HOME.set(None);
        log.same(
            "transport onto a wider home",
            || eng.transport(&f, &wide, |v| VarId(v.0 + shift), 0).map(|(g, _)| (g.model_count().unwrap(), g.vtree().num_leaves())).map_err(|r| r.error),
            || Ok((popcount(&tf) << n as usize, f.vtree().num_leaves())),
        );
        // A map that is not one to one, and one that leaves the home.
        if f.vtree().num_leaves() >= 2 {
            log.same(
                "transport refuses a shared image",
                || Ok::<_, ()>(eng.transport(&f, &home, |_| VarId(1), 0).is_err()),
                || Ok(true),
            );
        }
        if f.vtree().num_leaves() >= 1 {
            for out in [0, n + 1, u32::MAX] {
                log.same(
                    "transport refuses an image outside the home",
                    || Ok::<_, ()>(eng.transport(&f, &home, |_| VarId(out), 0).is_err()),
                    || Ok(true),
                );
            }
        }
    }
    let report = log.report();
    eprintln!("{report}");
    assert!(log.seen.is_empty(), "{report}");
}

#[test]
fn pools_in_every_form() {
    use tididi::restructure::search::PoolSearchConfig;
    quiet();
    let eng = Engine::new();
    let mut log = Log::default();
    let mut kept = 0;
    for seed in 0..300u64 {
        let mut rng = Lcg::new(seed ^ 0x1b87_3593);
        let n = 2 + rng.below(6) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let before = (**home.vtree()).clone();
        let homes = std::slice::from_ref(&home);
        let mut members: Vec<(Tdd, String)> = (0..2 + rng.below(3)).map(|_| operand(&eng, homes, n, &mut rng)).collect();
        let (outsider, no) = operand(&eng, homes, n, &mut rng);
        log.case = format!("seed {seed} n {n}: {:?}; outsider {no}", members.iter().map(|m| &m.1).collect::<Vec<_>>());
        let tables: Vec<Table> = members.iter().map(|m| table(&m.0, n)).collect();
        let counts: Vec<_> = members.iter().map(|m| m.0.model_count().unwrap()).collect();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut refs: Vec<&mut Tdd> = members.iter_mut().map(|m| &mut m.0).collect();
            eng.pool_search(&mut refs, &PoolSearchConfig::default()).map(|stats| stats.accepts)
        }));
        log.runs += 1;
        let accepts = match outcome {
            Err(e) => {
                log.note(format!("pool_search: PANIC {}", text(&*e)));
                continue;
            }
            Ok(Err(e)) => {
                log.note(format!("pool_search: refused {}", short(&e)));
                continue;
            }
            Ok(Ok(accepts)) => accepts,
        };
        assert!(home.vtree().same_tree(&before), "the search changed the home: {}", log.case);
        for (i, (m, _)) in members.iter().enumerate() {
            log.value("pool_search keeps each function", || Ok(table(m, n)), tables[i].clone());
            log.value("pool_search keeps each count", || m.model_count(), counts[i].clone());
            log.same("pool_search leaves valid members", || checks(m), || Ok(()));
        }
        for i in 1..members.len() {
            let both = zip(&tables[0], &tables[i], |x, y| x && y);
            log.tdd(n, "and of two members after a search", || eng.and(&members[0].0, &members[i].0), &both);
        }
        let t_out = table(&outsider, n);
        if accepts == 0 {
            log.tdd(n, "and with a diagram outside a search that kept nothing", || eng.and(&members[0].0, &outsider), &zip(&tables[0], &t_out, |x, y| x && y));
        } else {
            kept += 1;
            let new_home = members[0].0.vtree().home().cloned().unwrap_or_else(|| Home::new(Arc::clone(members[0].0.vtree())));
            log.same(
                "and with a diagram left on the old home answers or refuses",
                || Ok::<_, ()>(eng.and(&members[0].0, &outsider).map_or(true, |t| table(&t, n) == zip(&tables[0], &t_out, |x, y| x && y))),
                || Ok(true),
            );
            log.tdd(
                n,
                "and with a diagram outside a search, transported",
                || eng.and(&members[0].0, eng.transport(&outsider, &new_home, |v| v, usize::MAX).map_err(|r| match r.error {
                    tididi::restructure::TransportError::Operation(e) => e,
                    _ => OperationError::VtreeMismatch,
                })?.0),
                &zip(&tables[0], &t_out, |x, y| x && y),
            );
        }
    }
    let report = log.report();
    eprintln!("{kept} searches kept a move\n{report}");
    assert!(log.seen.is_empty(), "{report}");
}

#[allow(dead_code)]
fn unused(_: VtreeIdx) {}
