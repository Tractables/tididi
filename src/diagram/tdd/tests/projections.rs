//! Every public query and single-operand operation on a diagram trimmed to
//! a projection answers as the same call on the diagram on its home, or
//! refuses with a named error; none panics.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};

use num_bigint::BigUint;
use num_rational::BigRational;

use crate::cnf::{ClauseSink, CnfScheme};
use crate::diagram::{Arithmetic, ColumnAlgebra, LeafLabel, LiteralWeights, PairIndex, RationalWeights, SlotPairs, Weights};
use crate::query::PinSemantics;
use crate::restructure::search::{MinimizePairs, RotationSearchConfig};
use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::{and2, exact_weight, rand_conj};
use crate::value::Retention;
use crate::vtree::rng::Lcg;
use crate::vtree::{FreeVars, Home, VarId, VtreeIdx};
use crate::{Engine, Literal, OperationError};

use super::*;

/// Model counts, a column of big integers per level, valuing free
/// variables when `free` is set.
struct Counts {
    free: bool,
}

impl ColumnAlgebra for Counts {
    type Column = Vec<BigUint>;
    type Value = BigUint;
    fn zero(&self) -> BigUint {
        BigUint::ZERO
    }
    fn column(&self, _: VtreeIdx, width: usize) -> Vec<BigUint> {
        vec![BigUint::ZERO; width]
    }
    fn leaf(&self, _: VtreeIdx, _: VarId, label: LeafLabel, col: &mut Vec<BigUint>) {
        col[label as usize] = if label == LeafLabel::One { 2u32 } else { 1 }.into();
    }
    fn fold(&self, _: VtreeIdx, slot: usize, pairs: SlotPairs<'_>, left: &Vec<BigUint>, right: &Vec<BigUint>, out: &mut Vec<BigUint>) {
        out[slot] = pairs.map(|(l, r)| &left[l] * &right[r]).sum();
    }
    fn read(&self, _: VtreeIdx, col: Vec<BigUint>, slot: usize) -> BigUint {
        col[slot].clone()
    }
    fn free(&self, _: VtreeIdx, vars: &FreeVars<'_>, col: &mut Vec<BigUint>) -> bool {
        col.iter_mut().for_each(|v| *v <<= vars.len());
        self.free
    }
}

struct Sink {
    last: i32,
    clauses: usize,
}

impl ClauseSink for Sink {
    fn fresh_var(&mut self) -> i32 {
        self.last += 1;
        self.last
    }
    fn leaf_literal(&mut self, var: VarId) -> i32 {
        var.0 as i32
    }
    fn clause(&mut self, _: &[i32]) {
        self.clauses += 1;
    }
}

#[derive(Default)]
struct Tally {
    same: usize,
    refused: BTreeMap<String, usize>,
    differs: Vec<String>,
    panics: Vec<String>,
    unreached: usize,
}

#[derive(Default)]
struct Audit {
    tallies: BTreeMap<&'static str, Tally>,
    case: String,
    /// When set, every diagram a call returns is checked to reach all its
    /// nodes, and each that does not is listed here.
    #[cfg(debug_assertions)]
    unreached_by: Option<Vec<String>>,
}

static PANIC_AT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Whether `t` holds a node its output does not reach, where its known
/// loose levels allow one ([`check_loose_levels`](crate::test_helpers::check::check_loose_levels)).
/// Its stored nodes, reached or not, are deterministic.
fn unreached(t: &Tdd) -> Result<bool, String> {
    crate::test_helpers::check::check_loose_levels(t)?;
    crate::test_helpers::check::check_certified(t)?;
    if !t.has_marginal_level() {
        crate::test_helpers::check::check_determinism(t)?;
    }
    if t.is_zero() || t.is_one() {
        return Ok(false);
    }
    let reach = t.reachable_nodes();
    let vtree = t.vtree();
    let any = (0..vtree.num_nodes())
        .filter(|&v| !vtree.node(VtreeIdx(v as u32)).is_leaf() && !t.levels[v].is_marginal())
        .any(|v| reach[v].iter().any(|&r| !r));
    match t.dirty.loose() {
        Some(loose) if loose.is_empty() && any => Err("a node the output does not reach, and no level listed loose".into()),
        _ => Ok(any),
    }
}

/// The panic of [`debug_assert_all_reached`](crate::reduce::prune::debug_assert_all_reached) on `t`, if any.
#[cfg(debug_assertions)]
fn all_reached(t: &Tdd) -> Option<String> {
    catch_unwind(AssertUnwindSafe(|| crate::reduce::prune::debug_assert_all_reached(t))).err().map(|e| panic_text(&*e))
}

fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    let text = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default();
    format!("{text} at {}", PANIC_AT.lock().unwrap())
}

impl Audit {
    /// Run `on_projection` and `on_home`; equal results under `same`, or
    /// both refused, count as agreement.
    fn check<T: 'static, E: Debug>(
        &mut self,
        name: &'static str,
        on_projection: impl FnOnce() -> Result<T, E>,
        on_home: impl FnOnce() -> Result<T, E>,
        same: impl FnOnce(&T, &T) -> bool,
    ) {
        let case = self.case.clone();
        let tally = self.tallies.entry(name).or_default();
        let got = match catch_unwind(AssertUnwindSafe(on_projection)) {
            Ok(got) => got,
            Err(e) => return tally.panics.push(format!("{case}: {}", panic_text(&*e))),
        };
        let expected = match catch_unwind(AssertUnwindSafe(on_home)) {
            Ok(expected) => expected,
            Err(e) => return tally.panics.push(format!("{case}: on the home: {}", panic_text(&*e))),
        };
        #[cfg(debug_assertions)]
        if let Some(list) = &mut self.unreached_by {
            for (side, r) in [("projection", got.as_ref().ok().map(|g| g as &dyn std::any::Any)), ("home", expected.as_ref().ok().map(|f| f as &dyn std::any::Any))] {
                if let Some(e) = r.and_then(|r| r.downcast_ref::<Tdd>()).and_then(all_reached) {
                    list.push(format!("{name} on the {side}: {e}"));
                }
            }
        }
        if let Ok(g) = &got
            && let Some(t) = (g as &dyn std::any::Any).downcast_ref::<Tdd>()
        {
            match unreached(t) {
                Ok(any) => tally.unreached += usize::from(any),
                Err(e) => return tally.differs.push(format!("{case}: {e}")),
            }
        }
        match (got, expected) {
            (Ok(g), Ok(f)) => {
                let agree = match catch_unwind(AssertUnwindSafe(|| same(&g, &f))) {
                    Ok(agree) => agree,
                    Err(e) => return tally.panics.push(format!("{case}: comparing: {}", panic_text(&*e))),
                };
                if agree {
                    tally.same += 1;
                } else {
                    tally.differs.push(case);
                }
            }
            (Err(e), Ok(_)) => *tally.refused.entry(format!("{e:?}")).or_default() += 1,
            (Ok(_), Err(e)) => tally.differs.push(format!("{case}: only the home refused: {e:?}")),
            (Err(_), Err(_)) => tally.same += 1,
        }
    }

    /// A call with nothing to compare: it answers or refuses, and does not panic.
    fn runs<T: 'static, E: Debug>(&mut self, name: &'static str, call: impl FnOnce() -> Result<T, E>) {
        self.check(name, || call().map(drop), || Ok::<(), E>(()), |_, _| true);
    }

    fn report(&self) -> String {
        let mut out = String::new();
        for (name, t) in &self.tallies {
            out += &format!("{name}: {} same", t.same);
            for (e, n) in &t.refused {
                out += &format!(", {n} refused {e}");
            }
            if t.unreached > 0 {
                out += &format!(", {} hold a node they do not reach", t.unreached);
            }
            if !t.differs.is_empty() {
                out += &format!(", {} DIFFER (first {})", t.differs.len(), t.differs[0]);
            }
            let mut panics: BTreeMap<&str, (usize, &str)> = BTreeMap::new();
            for p in &t.panics {
                let (case, what) = p.split_once(": ").unwrap_or(("", p));
                panics.entry(what).or_insert((0, case)).0 += 1;
            }
            for (what, (k, case)) in panics {
                out += &format!(", {k} PANIC {what} (first {case})");
            }
            out += "\n";
        }
        out
    }
}

fn weights(n: u32, rng: &mut Lcg) -> Weights {
    let mut w = || BigRational::new((rng.below(7) as i64 - 2).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..=n).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
    Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

/// A diagram on `home` after random conditioning, forgetting, conjunction
/// and minimization; one in six is a constant.
fn diagram(home: &Arc<Vtree>, n: u32, rng: &mut Lcg) -> Tdd {
    match rng.below(12) {
        0 => return crate::test_helpers::padded(Tdd::one(home), home),
        1 => return crate::test_helpers::padded(Tdd::zero(home), home),
        _ => {}
    }
    let mut f = rand_conj(home, n, 4, 3, false, rng);
    for _ in 0..1 + rng.below(4) {
        let var = VarId(1 + rng.below(u64::from(n)) as u32);
        f = match rng.below(4) {
            0 => f.condition_var(var, rng.coin()).unwrap(),
            1 => f.exists_vars(&[var]).unwrap(),
            2 => and2(&f, &rand_conj(home, n, 2, 3, false, rng)),
            _ => {
                Engine::new().minimize_plain(&mut f).unwrap();
                f
            }
        };
    }
    crate::test_helpers::padded(f, home)
}

/// `g`, on a projection of `home`, placed on `home`.
/// `t`, on a home of `onto`'s shape and variables or a projection of one,
/// moved to the same place on `onto`.
fn reseated(t: &Tdd, onto: &Home) -> Tdd {
    let own = if t.vtree().is_home() {
        Arc::clone(onto.vtree())
    } else if t.vtree().num_nodes() == 0 {
        Vtree::empty_projection(onto)
    } else {
        Vtree::projection(onto, t.vtree().leaf_bottomup().map(|(_, v)| v)).unwrap()
    };
    let mut t = t.clone();
    // Safety: the two homes have one shape, so the two projections do.
    unsafe { t.reseat_vtree_unchecked(&own) };
    t
}

fn on_home(g: &Tdd, home: &Arc<Vtree>) -> Tdd {
    if g.vtree().num_nodes() == 0 {
        return if g.is_one() { Tdd::one(home) } else { Tdd::zero(home) };
    }
    if Arc::ptr_eq(g.vtree(), home) {
        return g.clone();
    }
    crate::test_helpers::embedded(g, home, |v| v).unwrap().0
}

/// `g` on `home` when it is on a projection of it or on it.
fn placed(g: &Tdd, home: &Arc<Vtree>) -> Option<Tdd> {
    let on = |v: &Arc<Vtree>| Arc::ptr_eq(v.link_home().map(Home::vtree).unwrap_or(v), home);
    on(g.vtree()).then(|| on_home(g, home))
}

/// Whether `x` and `y` hold the same assignments to variables `1..=n`.
fn same_models(x: &Tdd, y: &Tdd, n: u32) -> bool {
    let holds = |t: &Tdd, a: u32| {
        let cube = (1..=n).filter(|&v| t.vtree().leaf_of(VarId(v)).is_some()).map(|v| Literal { var: VarId(v), sign: a >> (v - 1) & 1 == 1 });
        t.clone().condition(cube).unwrap().is_sat().unwrap()
    };
    (0..1u32 << n).all(|a| holds(x, a) == holds(y, a))
}

// Embedding drops the unreachable nodes a diagram may keep, so per-node
// answers are compared on the reachable nodes of the level.
fn reached<T: Ord + Clone>(t: &Tdd, at: VtreeIdx, v: &[T]) -> Vec<T> {
    let mask = &t.reachable_nodes()[at.idx()];
    if t.vtree().node(at).is_leaf() || mask.len() != v.len() {
        return sorted(v);
    }
    sorted(&v.iter().zip(mask).filter(|&(_, &r)| r).map(|(x, _)| x.clone()).collect::<Vec<_>>())
}

fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort();
    v
}

#[allow(clippy::too_many_lines)]
fn audit_case(a: &mut Audit, eng: &Engine, home: &Arc<Vtree>, f: &Tdd, g: &Tdd, n: u32, rng: &mut Lcg) {
    let hm = Home::new(Arc::clone(home));
    let var = |rng: &mut Lcg| VarId(1 + rng.below(u64::from(n)) as u32);
    let vars = |rng: &mut Lcg| -> Vec<VarId> { (1..=n).filter(|_| rng.coin()).map(VarId).collect() };
    let equiv = |x: &Tdd, y: &Tdd| match (placed(x, home), placed(y, home)) {
        (Some(x), Some(y)) => eng.equivalent(&x, &y).unwrap(),
        _ => same_models(x, y, n),
    };
    let same_tdd = |x: &Tdd, y: &Tdd| equiv(x, y);
    // Every result on a home with no summed-out level is closed; the readers
    // place on the caller's vtree and may pad it.
    let closed = |x: &Tdd, y: &Tdd| {
        if x.vtree().on_home() && !x.has_marginal_level() {
            check_trimmed(x, false).unwrap_or_else(|e| panic!("not closed: {e}"));
        }
        equiv(x, y)
    };

    // Queries.
    a.check("model_count", || g.model_count(), || f.model_count(), PartialEq::eq);
    a.check("is_sat", || g.is_sat(), || f.is_sat(), PartialEq::eq);
    a.check("support", || g.support(), || f.support(), |x, y| sorted(x) == sorted(y));
    let key = |l: &Vec<Literal>| sorted(&l.iter().map(|l| (l.var, l.sign)).collect::<Vec<_>>());
    a.check("implied_literals", || g.implied_literals(), || f.implied_literals(), |x, y| key(x) == key(y));
    a.check(
        "satisfying_assignment",
        || g.satisfying_assignment(),
        || f.satisfying_assignment(),
        |x, y| match (x, y) {
            (None, None) => true,
            (Some(m), Some(r)) => m.len() == r.len() && f.clone().condition(m.iter().copied()).unwrap().is_sat().unwrap(),
            _ => false,
        },
    );
    let pv = vars(rng);
    a.check("projected_model_count", || g.projected_model_count(&pv), || f.projected_model_count(&pv), PartialEq::eq);
    let ws = weights(n, rng);
    let algebra = ws.algebra().clone();
    let weighted = Home::weighted(Vtree::clone(home), ws).unwrap();
    a.check("evaluate", || g.evaluate(&algebra), || f.evaluate(&algebra), PartialEq::eq);
    a.check("evaluate_columns", || g.evaluate_columns(&Counts { free: true }), || f.evaluate_columns(&Counts { free: true }), PartialEq::eq);
    if g.vtree().free_leaves() > 0 && !g.is_zero() {
        let refused = g.evaluate_columns(&Counts { free: false });
        // The constant true values its first free variable as a leaf.
        let named = g.vtree().free_set().unwrap().vars().nth(usize::from(g.is_one())).unwrap();
        a.check("evaluate_columns without free", || Ok::<_, OperationError>(refused), || Ok(Err(OperationError::VariableNotInVtree(named))), PartialEq::eq);
    }
    a.check(
        "weighted_value",
        || reseated(g, &weighted).weighted_value().map(|v| v.map(|v| exact_weight(&v))),
        || reseated(f, &weighted).weighted_value().map(|v| v.map(|v| exact_weight(&v))),
        PartialEq::eq,
    );
    if g.vtree().num_nodes() > 0 {
        let e = on_home(g, home);
        let at = VtreeIdx(rng.below(g.vtree().num_nodes() as u64) as u32);
        let h = g.vtree().home_node(at);
        a.check("evaluate_at", || g.evaluate_at(at, &algebra).map(|v| reached(g, at, &v)), || e.evaluate_at(h, &algebra).map(|v| reached(&e, h, &v)), PartialEq::eq);
        let counts = Counts { free: true };
        a.check(
            "evaluate_columns_at",
            || g.evaluate_columns_at(at, &counts).map(|v| reached(g, at, &v)),
            || e.evaluate_columns_at(h, &counts).map(|v| reached(&e, h, &v)),
            PartialEq::eq,
        );
        a.runs("level_view", || Ok::<_, ()>(g.level_view(at).level().nodes().len()));
        a.runs("reference_slot_count", || Ok::<_, ()>(g.reference_slot_count(at)));
        if !g.vtree().node(at).is_leaf() && !g.is_zero() {
            let index = PairIndex::new(g).unwrap();
            let e_index = PairIndex::new(&e).unwrap();
            a.check(
                "sum_indexed",
                || eng.sum_indexed(g, g, &index, at, &counts).map(|v| reached(g, at, &v)),
                || eng.sum_indexed(&e, &e, &e_index, h, &counts).map(|v| reached(&e, h, &v)),
                PartialEq::eq,
            );
        }
        a.runs("project_to_subtree", || eng.project_to_subtree(g, at).map(|t| t.node_count()));
        a.runs("at_least_distinct", || eng.at_least_distinct(g, at, 2).map(|t| t.node_count()));
        a.runs("rotate_marginal_cluster", || eng.rotate_marginal_cluster(&mut g.clone(), at, 8, &mut Vec::new()));
    }
    a.runs("node_counts_u128", || g.node_counts_u128());
    a.runs("reachable_nodes", || Ok::<_, ()>(g.reachable_nodes().len()));
    a.runs("level_sizes_to_dot", || Ok::<_, ()>(g.level_sizes_to_dot().len()));
    a.runs("tdd_to_dot", || crate::io::tdd_to_dot(g).map(|s| s.len()));
    a.runs("sizes", || Ok::<_, ()>((g.max_width(), g.node_count(), g.pair_count(), g.pair_count_at_most(3), g.retired_marginal_slots())));
    a.runs("encode_cnf", || g.encode_cnf(CnfScheme::Equivalence, n as i32 + 1, &mut Sink { last: n as i32 + 1, clauses: 0 }).map(|_| ()));
    let columns: Vec<Vec<VarId>> = (1..=n).map(|v| vec![VarId(v)]).collect();
    let cols: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
    a.check(
        "model_columns",
        || g.model_columns(&cols).map(|mut m| sorted(&m.to_columns())),
        || f.model_columns(&cols).map(|mut m| sorted(&m.to_columns())),
        PartialEq::eq,
    );
    let descending = vec![false; cols.len()];
    a.check("ordered_models", || g.ordered_models(&cols, &descending, 1, 5), || f.ordered_models(&cols, &descending, 1, 5), PartialEq::eq);
    a.check("ordered_keys", || g.ordered_keys(&cols, &descending, 5), || f.ordered_keys(&cols, &descending, 5), PartialEq::eq);

    // Counters and evaluators, pins on any home variable.
    for convention in [PinSemantics::Cofactor, PinSemantics::Evidence] {
        let pins: Vec<(VarId, Option<bool>)> = (0..rng.below(3)).map(|_| (var(rng), Some(rng.coin()))).collect();
        let table = vec![var(rng), var(rng)];
        let count = |t: &Tdd| -> Result<_, OperationError> {
            let mut c = t.counter_with(Retention::All, convention)?;
            let before = c.model_count()?;
            c.set_pins(&pins)?;
            let pinned = c.model_count()?;
            let table = if table[0] == table[1] { Vec::new() } else { c.count_table(&table)? };
            c.clear_pins();
            Ok((before, pinned, table, c.model_count()?))
        };
        a.check("counter", || count(g), || count(f), PartialEq::eq);
    }
    let pins: Vec<(VarId, Option<bool>)> = (0..1 + rng.below(3)).map(|_| (var(rng), Some(rng.coin()))).collect();
    let value = |t: &Tdd| -> Result<_, OperationError> {
        let mut e = t.evaluator(algebra.clone())?;
        let before = e.value()?;
        e.set_pins(&pins)?;
        Ok((before, e.value()?))
    };
    a.check("evaluator", || value(g), || value(f), PartialEq::eq);

    // Single-operand operations.
    let x = var(rng);
    let b = rng.coin();
    let some = vars(rng);
    let lit = |v: VarId, b: bool| Literal { var: v, sign: b };
    a.check("negate", || g.clone().negate(), || f.clone().negate(), closed);
    a.check("condition_var", || g.clone().condition_var(x, b), || f.clone().condition_var(x, b), closed);
    a.check("condition", || g.clone().condition([lit(x, b)]), || f.clone().condition([lit(x, b)]), closed);
    a.check("condition_vars", || g.clone().condition_vars(&some, b), || f.clone().condition_vars(&some, b), closed);
    a.check("exists_var", || g.clone().exists_var(x), || f.clone().exists_var(x), closed);
    a.check("exists_vars", || g.clone().exists_vars(&some), || f.clone().exists_vars(&some), closed);
    a.check("forall_vars", || g.clone().forall_vars(&some), || f.clone().forall_vars(&some), closed);
    let clause = [lit(x, b), lit(var(rng), rng.coin())];
    a.check("and_clause", || g.clone().and_clause(clause), || f.clone().and_clause(clause), closed);
    a.check("or_cube", || g.clone().or_cube(clause), || f.clone().or_cube(clause), closed);
    a.check("and_cube", || eng.and_cube(g.clone(), clause), || eng.and_cube(f.clone(), clause), closed);
    a.check("renumber_top_down", || g.renumber_top_down(), || f.renumber_top_down(), closed);
    let mutate = |t: &Tdd, op: &dyn Fn(&mut Tdd) -> Result<(), OperationError>| {
        let mut t = t.clone();
        op(&mut t).map(|()| t)
    };
    a.check("minimize", || mutate(g, &|t| t.minimize()), || mutate(f, &|t| t.minimize()), closed);
    a.check("trim", || mutate(g, &|t| t.trim(&hm).map(drop)), || mutate(f, &|t| t.trim(&hm).map(drop)), closed);
    a.check("reduce", || mutate(g, &|t| t.reduce(crate::reduce::ReductionPlan::default())), || mutate(f, &|t| t.reduce(crate::reduce::ReductionPlan::default())), closed);
    a.check("attach_level_counts", || mutate(g, &|t| t.attach_level_counts()).and_then(|t| t.model_count()), || mutate(f, &|t| t.attach_level_counts()).and_then(|t| t.model_count()), PartialEq::eq);
    let root = g.vtree().root();
    if g.vtree().num_nodes() > 0 {
        a.check(
            "marginalize_levels",
            || mutate(g, &|t| t.marginalize_levels(&[root])).and_then(|t| t.model_count()),
            || Ok(f.model_count().unwrap()),
            PartialEq::eq,
        );
    }
    let model: Vec<Literal> = (1..=n).map(|v| lit(VarId(v), rng.coin())).collect();
    a.check("insert_model", || mutate(g, &|t| t.insert_model(&model)), || mutate(f, &|t| t.insert_model(&model)), closed);
    a.check("remove_model", || mutate(g, &|t| t.remove_model(&model)), || mutate(f, &|t| t.remove_model(&model)), closed);
    a.check(
        "maintain",
        || mutate(g, &|t| t.maintain().and_then(|mut m| m.insert_model(&model))),
        || mutate(f, &|t| t.maintain().and_then(|mut m| m.insert_model(&model))),
        closed,
    );
    a.runs("filter_nodes", || g.filter_nodes(|_| true).map(drop));
    a.check(
        "rotation_search",
        || mutate(g, &|t| t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(drop)),
        || mutate(f, &|t| t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(drop)),
        closed,
    );
    a.runs("rotate_if", || mutate(g, &|t| t.rotate_if(&[], usize::MAX, |_| true).map(drop)).map(drop));
    let other = vars(rng);
    let rename: Vec<(VarId, VarId)> = match (some.first(), other.first()) {
        (Some(&p), Some(&q)) if p != q => vec![(p, q), (q, p)],
        _ => Vec::new(),
    };
    a.check("rename_vars", || g.clone().rename_vars(&rename), || f.clone().rename_vars(&rename), closed);
    let literal_x = crate::literal(home, i32::try_from(x.0).unwrap()).unwrap();
    let y = var(rng);
    a.check("substitute", || g.clone().substitute(&[(y, &literal_x)]), || f.clone().substitute(&[(y, &literal_x)]), closed);

    // Moves between vtrees.
    a.check("transport", || eng.transport(g, &hm, |v| v, 0).map(|r| r.0).map_err(|r| r.error), || Ok(f.clone()), closed);
    a.check(
        "transport_projection",
        || eng.transport_projection(g, &hm, |v| v).map(|v| v.num_leaves()),
        || Ok(g.vtree().num_leaves()),
        PartialEq::eq,
    );
    let target = Arc::new(Vtree::random(n, rng.below(1000)));
    a.runs("restructure_to", || g.clone().restructure_to(&target, usize::MAX));

    // Serialization against the diagram's own vtree.
    let mut text = Vec::new();
    a.check(
        "write_tdd and read_tdd",
        || {
            crate::io::write_tdd(&mut text, g)?;
            crate::io::read_tdd(&mut text.as_slice(), g.vtree())
        },
        || Ok::<_, crate::io::IoError>(f.clone()),
        same_tdd,
    );
    let mut bytes = Vec::new();
    a.check(
        "binary",
        || {
            crate::io::write_tdd_binary(&mut bytes, g)?;
            crate::io::read_tdd_binary(&mut bytes.as_slice(), g.vtree())
        },
        || Ok::<_, crate::io::IoError>(f.clone()),
        same_tdd,
    );
}

/// The audit's operands and calls for `seed`; returns the two operands.
fn audit_seed(audit: &mut Audit, eng: &Engine, seed: u64) -> (Tdd, Tdd) {
    let mut rng = Lcg::new(seed ^ 0xa0d1);
    let n = 2 + rng.below(8) as u32;
    let home = Arc::clone(Home::new(Vtree::random(n, seed)).vtree());
    let f = diagram(&home, n, &mut rng);
    let mut g = f.clone();
    g.trim(&Home::new(Arc::clone(&home))).unwrap();
    audit.case = format!("seed {seed}, {n} variables, {} of them kept", g.vtree().num_leaves());
    audit_case(audit, eng, &home, &f, &g, n, &mut rng);
    (f, g)
}

#[test]
fn every_entry_point_answers_on_a_projection_as_on_the_home() {
    let eng = Engine::new();
    let mut audit = Audit::default();
    std::panic::set_hook(Box::new(|info| {
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        let caller = trace
            .lines()
            .find(|l| l.contains("at ./src/") && !l.contains("vtree/node.rs") && !l.contains("tests/projections.rs"))
            .map(|l| l.trim().to_string())
            .unwrap_or_default();
        *PANIC_AT.lock().unwrap() = format!("{} from {caller}", info.location().map(ToString::to_string).unwrap_or_default());
    }));
    for seed in 0..200u64 {
        audit_seed(&mut audit, &eng, seed);
    }
    let report = audit.report();
    eprintln!("{report}");
    assert!(!report.contains("DIFFER") && !report.contains("PANIC"), "{report}");
}

/// Every operand of the audit's case at seed 108 and every diagram a call
/// returns on it reaches all its nodes.
#[cfg(debug_assertions)]
#[test]
#[ignore = "operations return diagrams that keep unreached nodes"]
fn every_result_of_the_audit_at_seed_108_reaches_all_its_nodes() {
    let eng = Engine::new();
    let mut audit = Audit { unreached_by: Some(Vec::new()), ..Audit::default() };
    let (f, g) = audit_seed(&mut audit, &eng, 108);
    let mut failures: Vec<String> = [("home operand", &f), ("projection operand", &g)]
        .into_iter()
        .filter_map(|(what, t)| all_reached(t).map(|e| format!("{what}: {e}")))
        .collect();
    failures.extend(audit.unreached_by.take().unwrap());
    let shape = |t: &Tdd| format!("{} leaves, {} levels, {} nodes, {} pairs", t.vtree().num_leaves(), t.levels.len(), t.node_count(), t.pair_count());
    assert!(failures.is_empty(), "operands: home {}, projection {}\n{}", shape(&f), shape(&g), failures.join("\n"));
}

/// `x1 ∧ ¬x3` over `Vtree::balanced(4)`, on the home and trimmed to the
/// projection on x1 and x3.
fn x1_not_x3(eng: &Engine) -> (Arc<Vtree>, Tdd, Tdd) {
    let home = Arc::clone(Home::new(Vtree::balanced(4)).vtree());
    let lit = |v: u32, sign: bool| Literal { var: VarId(v), sign };
    let f = crate::test_helpers::padded(eng.cube(&home, [lit(1, true), lit(3, false)]).unwrap(), &home);
    let mut g = f.clone();
    assert!(g.trim(&Home::new(Arc::clone(&home))).unwrap());
    assert_eq!(g.vtree().free_set().unwrap().vars().collect::<Vec<_>>(), [VarId(2), VarId(4)]);
    (home, f, g)
}

#[test]
fn column_evaluation_values_the_free_variables_or_refuses() {
    let eng = Engine::new();
    let (home, f, g) = x1_not_x3(&eng);
    let counts = Counts { free: true };
    let four = BigUint::from(4u32);
    assert_eq!(f.evaluate_columns(&counts).unwrap(), four);
    assert_eq!(g.evaluate_columns(&counts).unwrap(), four);
    assert_eq!(g.evaluate_columns(&Counts { free: false }), Err(OperationError::VariableNotInVtree(VarId(2))));
    let (root, home_root) = (g.vtree().root(), home.root());
    assert_eq!(g.vtree().home_node(root), home_root);
    assert_eq!(g.evaluate_columns_at(root, &counts).unwrap(), f.evaluate_columns_at(home_root, &counts).unwrap());
    let sum = |t: &Tdd, at| eng.sum_indexed(t, t, &PairIndex::new(t).unwrap(), at, &counts).unwrap();
    assert_eq!(sum(&g, root), sum(&f, home_root));
    assert_eq!(eng.sum_indexed(&g, &g, &PairIndex::new(&g).unwrap(), root, &Counts { free: false }), Err(OperationError::VariableNotInVtree(VarId(2))));

    let hm = &g.vtree().home().unwrap();
    let mut one = Tdd::one(&home);
    one.trim(hm).unwrap();
    assert_eq!(one.vtree().num_nodes(), 0);
    assert_eq!(one.evaluate_columns(&counts).unwrap(), BigUint::from(16u32));
    assert_eq!(one.evaluate_columns(&Counts { free: false }), Err(OperationError::VariableNotInVtree(VarId(2))));
    let mut zero = Tdd::zero(&home);
    zero.trim(hm).unwrap();
    assert_eq!(zero.evaluate_columns(&Counts { free: false }).unwrap(), BigUint::ZERO);
}

#[test]
fn pins_on_free_variables_count_as_on_the_home() {
    let eng = Engine::new();
    let (_, f, g) = x1_not_x3(&eng);
    for (convention, factor) in [(PinSemantics::Evidence, 1u32), (PinSemantics::Cofactor, 2)] {
        let count = |t: &Tdd| {
            let mut c = t.counter_with(Retention::All, convention).unwrap();
            c.set_pins(&[(VarId(2), Some(true))]).unwrap();
            let pinned = c.model_count().unwrap();
            let table = c.count_table(&[VarId(4), VarId(1)]).unwrap();
            (pinned, table)
        };
        let (pinned, table) = count(&g);
        assert_eq!(pinned, BigUint::from(2 * factor), "{convention:?}");
        assert_eq!(table, [0, 0, factor.pow(3), factor.pow(3)].map(BigUint::from), "{convention:?}");
        assert_eq!((pinned, table), count(&f), "{convention:?}");
    }
    assert_eq!(g.counter().unwrap().set_pin(VarId(5), Some(true)), Err(OperationError::VariableNotInVtree(VarId(5))));
}

#[test]
fn diagrams_on_projections_are_written_and_read() {
    let eng = Engine::new();
    let (home, f, g) = x1_not_x3(&eng);
    let mut one = Tdd::one(&home);
    let mut zero = Tdd::zero(&home);
    let hm = &g.vtree().home().unwrap();
    one.trim(hm).unwrap();
    zero.trim(hm).unwrap();
    let same_leaves = Vtree::projection(hm, [VarId(3), VarId(1)]).unwrap();
    let cases = [(&f, home.clone()), (&g, g.vtree().clone()), (&g, same_leaves), (&one, one.vtree().clone()), (&zero, zero.vtree().clone())];
    for (t, vtree) in cases {
        let mut text = Vec::new();
        crate::io::write_tdd(&mut text, t).unwrap();
        let read = crate::io::read_tdd(&mut text.as_slice(), &vtree).unwrap();
        let mut bytes = Vec::new();
        crate::io::write_tdd_binary(&mut bytes, t).unwrap();
        let binary = crate::io::read_tdd_binary(&mut bytes.as_slice(), &vtree).unwrap();
        for back in [read, binary] {
            assert!(back.vtree().is_on(hm));
            // f pads x2 and x4 with `One`; read onto the home it is on g's leaves.
            assert_eq!(back.vtree().num_leaves(), if vtree.num_nodes() > 0 { 2 } else { 0 });
            assert_eq!((back.is_one(), back.is_zero()), (t.is_one(), t.is_zero()));
            assert_eq!(back.model_count().unwrap(), t.model_count().unwrap());
            if vtree.num_nodes() > 0 {
                assert!(eng.equivalent(&on_home(&back, &home), &f).unwrap());
            }
        }
    }
    let mut text = Vec::new();
    crate::io::write_tdd(&mut text, &one).unwrap();
    let back = crate::io::read_tdd(&mut text.as_slice(), g.vtree()).unwrap();
    assert!(back.is_one() && back.vtree().num_nodes() == 0 && back.vtree().is_on(hm));
}

#[test]
fn the_vtree_check_agrees_with_transport() {
    let eng = Engine::new();
    let mut accepted = 0;
    for seed in 0..60u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(5) as u32;
        let home = Arc::clone(Home::new(Vtree::random(n, seed)).vtree());
        let source = if rng.coin() {
            let mut g = diagram(&home, n, &mut rng);
            g.trim(&Home::new(Arc::clone(&home))).unwrap();
            g
        } else {
            rand_conj(&Arc::new(Vtree::random(n, seed + 1000)), n, 3, 2, false, &mut rng)
        };
        let shift = rng.below(2) as u32;
        let wide = Arc::clone(Home::new(Vtree::random(n + shift, seed + 2000)).vtree());
        let target = if rng.coin() { &home } else { &wide };
        let map = |v: VarId| VarId(v.0 + shift * u32::from(Arc::ptr_eq(target, &wide)));
        let target_home = Home::new(Arc::clone(target));
        let checked = eng.transport_projection(&source, &target_home, map);
        match eng.transport(&source, &target_home, map, 0) {
            Ok((moved, _)) => {
                accepted += 1;
                let leaves = |v: &Vtree| v.leaf_bottomup().map(|(_, var)| var).collect::<Vec<_>>();
                let checked = checked.unwrap();
                assert_eq!(moved.vtree().num_nodes(), checked.num_nodes(), "seed {seed}");
                assert_eq!(leaves(moved.vtree()), leaves(&checked), "seed {seed}");
            }
            Err(refused) => assert_eq!(checked.unwrap_err(), refused.error, "seed {seed}"),
        }
    }
    assert!(accepted > 5 && accepted < 55, "{accepted} of 60 accepted");
}
