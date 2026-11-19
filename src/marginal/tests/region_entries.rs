//! Every public entry point of one diagram on a diagram with regions, against
//! the same diagram on the home with the same nodes summed out by
//! `marginalize_levels`: equal answers, or a refusal on the projection.
//!
//! The reference exists where each region stands for one home node whose
//! leaves the diagram all had, so the cases are built that way.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_bigint::BigUint;

use super::region::{on_root_copy, operand, under, weights};
use crate::cnf::{ClauseSink, CnfScheme};
use crate::diagram::{Arithmetic, ColumnAlgebra, LeafLabel, SlotPairs, WeightStore};
use crate::query::PinSemantics;
use crate::restructure::search::{MinimizePairs, MultistartConfig, PoolSearchConfig, RotationMove, RotationSearchConfig};
use crate::test_helpers::exact_weight;
use crate::value::Retention;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, RotationKind, VarId, Vtree, VtreeIdx};
use crate::{Engine, Literal, OperationError, Tdd};

/// Model counts, a column of big integers per level.
struct Counts;

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
}

struct Sink {
    last: i32,
}

impl ClauseSink for Sink {
    fn fresh_var(&mut self) -> i32 {
        self.last += 1;
        self.last
    }
    fn leaf_literal(&mut self, var: VarId) -> i32 {
        var.0 as i32
    }
    fn clause(&mut self, _: &[i32]) {}
}

#[derive(Default)]
struct Tally {
    same: usize,
    refused: BTreeMap<String, usize>,
    differs: Vec<String>,
    panics: Vec<String>,
}

#[derive(Default)]
struct Audit {
    tallies: BTreeMap<&'static str, Tally>,
    case: String,
}

static PANIC_AT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    let text = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default();
    format!("{text} at {}", PANIC_AT.lock().unwrap())
}

impl Audit {
    /// Equal results under `same`, or both refused, agree; a refusal on the
    /// projection alone is tallied by its error.
    fn check<T, E: Debug>(
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
        match (got, expected) {
            (Ok(g), Ok(f)) => match catch_unwind(AssertUnwindSafe(|| same(&g, &f))) {
                Ok(true) => tally.same += 1,
                Ok(false) => tally.differs.push(case),
                Err(e) => tally.panics.push(format!("{case}: comparing: {}", panic_text(&*e))),
            },
            (Err(e), Ok(_)) => *tally.refused.entry(format!("{e:?}").split('(').next().unwrap_or("").to_string()).or_default() += 1,
            (Ok(_), Err(e)) => tally.differs.push(format!("{case}: only the home refused: {e:?}")),
            (Err(_), Err(_)) => tally.same += 1,
        }
    }

    /// A call with nothing to compare: it answers or refuses, and does not panic.
    fn runs<T, E: Debug>(&mut self, name: &'static str, call: impl FnOnce() -> Result<T, E>) {
        self.check(name, || call().map(drop), || Ok::<(), E>(()), |_, _| true);
    }

    fn report(&self) -> String {
        let mut out = String::new();
        for (name, t) in &self.tallies {
            out += &format!("{name}: {} same", t.same);
            for (e, n) in &t.refused {
                out += &format!(", {n} refused {e}");
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

/// The model count and, with weights, the exact weighted value.
pub(super) fn value(t: &Tdd) -> Result<(BigUint, Option<String>), OperationError> {
    let weighted = match t.weights() {
        Some(_) => t.weighted_value()?.map(|v| format!("{:?}", exact_weight(&v))),
        None => None,
    };
    Ok((t.model_count()?, weighted))
}

fn same_value(x: &Tdd, y: &Tdd) -> bool {
    format!("{:?}", value(x)) == format!("{:?}", value(y))
}

fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort();
    v
}

/// One or two home nodes, none under another, each an internal node all of
/// whose leaves `g` has.
fn targets(home: &Home, g: &Tdd, rng: &mut Lcg) -> Vec<VtreeIdx> {
    let h = home.vtree();
    let full: Vec<VtreeIdx> = h
        .bottomup()
        .filter(|&t| !h.node(t).is_leaf() && h.subtree(t).filter(|&u| h.node(u).is_leaf()).all(|u| g.vtree().leaf_of(h.leaf_var(u)).is_some()))
        .collect();
    let mut chosen: Vec<VtreeIdx> = Vec::new();
    for _ in 0..1 + rng.below(2) {
        if full.is_empty() {
            break;
        }
        let t = full[rng.below(full.len() as u64) as usize];
        if chosen.iter().all(|&c| !h.in_subtree(c, t) && !h.in_subtree(t, c)) {
            chosen.push(t);
        }
    }
    chosen
}

/// A diagram with regions and its smooth form on the home, for `seed`.
pub(crate) fn region_case(eng: &Engine, seed: u64, max_leaves: u32) -> Option<(Home, u32, Tdd, Tdd)> {
    region_case_and_source(eng, seed, max_leaves).map(|(home, n, f, g, _, _)| (home, n, f, g))
}

/// [`region_case`], with the diagram it was summed from and the home nodes
/// summed.
pub(crate) fn region_case_and_source(eng: &Engine, seed: u64, max_leaves: u32) -> Option<(Home, u32, Tdd, Tdd, Tdd, Vec<VtreeIdx>)> {
    let mut rng = Lcg::new(seed ^ 0x7e61_0a4d);
    let n = 2 + rng.below(u64::from(max_leaves) - 1) as u32;
    let home = Home::new(Vtree::random(n, seed));
    let used: Vec<VarId> = (1..=n).filter(|_| rng.below(4) != 0).map(VarId).collect();
    let mut g = operand(eng, &home, &used, &mut rng);
    if g.vtree().num_nodes() == 0 {
        return None;
    }
    let mut f = on_root_copy(eng, &g, home.vtree());
    if rng.below(3) == 0 {
        let w = weights(n, &mut rng);
        g.set_weights(WeightStore::new(w.clone(), Arithmetic::ExactRational)).unwrap();
        f.set_weights(WeightStore::new(w, Arithmetic::ExactRational)).unwrap();
    }
    let hs = targets(&home, &g, &mut rng);
    if hs.is_empty() {
        return None;
    }
    let source = g.clone();
    eng.marginalize(&mut g, under(&home, &hs)).unwrap();
    eng.marginalize_levels(&mut f, &hs).unwrap();
    g.vtree().has_regions().then_some((home, n, f, g, source, hs))
}

#[allow(clippy::too_many_lines)]
fn audit_case(a: &mut Audit, eng: &Engine, home: &Home, f: &Tdd, g: &Tdd, n: u32, rng: &mut Lcg) {
    let var = |rng: &mut Lcg| VarId(1 + rng.below(u64::from(n)) as u32);
    let vars = |rng: &mut Lcg| -> Vec<VarId> { (1..=n).filter(|_| rng.coin()).map(VarId).collect() };
    let lit = |v: VarId, b: bool| Literal { var: v, sign: b };

    a.check("model_count", || g.model_count(), || f.model_count(), PartialEq::eq);
    a.check("weighted_value", || value(g), || value(f), PartialEq::eq);
    a.check("is_sat", || g.is_sat(), || f.is_sat(), PartialEq::eq);
    a.check("is_zero", || Ok::<_, ()>(g.is_zero()), || Ok(f.is_zero()), PartialEq::eq);
    a.check("support", || g.support(), || f.support(), |x, y| sorted(x) == sorted(y));
    let key = |l: &Vec<Literal>| sorted(&l.iter().map(|l| (l.var, l.sign)).collect::<Vec<_>>());
    a.check("implied_literals", || g.implied_literals(), || f.implied_literals(), |x, y| key(x) == key(y));
    a.runs("satisfying_assignment", || g.satisfying_assignment());
    let pv = vars(rng);
    a.check("projected_model_count", || g.projected_model_count(&pv), || f.projected_model_count(&pv), PartialEq::eq);
    a.check("evaluate_columns", || g.evaluate_columns(&Counts), || f.evaluate_columns(&Counts), PartialEq::eq);
    let at = VtreeIdx(rng.below(g.vtree().num_nodes() as u64) as u32);
    a.runs("evaluate_columns_at", || g.evaluate_columns_at(at, &Counts));
    a.runs("level_view", || Ok::<_, ()>(g.level_view(at).level().nodes().len()));
    a.runs("reference_slot_count", || Ok::<_, ()>(g.reference_slot_count(at)));
    a.runs("project_to_subtree", || eng.project_to_subtree(g, at).map(|t| t.node_count()));
    a.runs("at_least_distinct", || eng.at_least_distinct(g, at, 2).map(|t| t.node_count()));
    a.runs("node_counts_u128", || g.node_counts_u128());
    a.runs("reachable_nodes", || Ok::<_, ()>(g.reachable_nodes().len()));
    a.runs("level_sizes_to_dot", || Ok::<_, ()>(g.level_sizes_to_dot().len()));
    a.runs("tdd_to_dot", || crate::io::tdd_to_dot(g).map(|s| s.len()));
    a.runs("sizes", || Ok::<_, ()>((g.max_width(), g.node_count(), g.pair_count(), g.pair_count_at_most(3), g.retired_marginal_slots())));
    a.runs("encode_cnf", || g.encode_cnf(CnfScheme::Equivalence, n as i32 + 1, &mut Sink { last: n as i32 + 1 }).map(|_| ()));
    let columns: Vec<Vec<VarId>> = (1..=n).map(|v| vec![VarId(v)]).collect();
    let cols: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
    a.runs("model_columns", || g.model_columns(&cols).map(|mut m| m.to_columns().len()));
    let descending = vec![false; cols.len()];
    a.runs("ordered_models", || g.ordered_models(&cols, &descending, 1, 5));
    a.runs("ordered_keys", || g.ordered_keys(&cols, &descending, 5));
    for convention in [PinSemantics::Cofactor, PinSemantics::Evidence] {
        let pins: Vec<(VarId, Option<bool>)> = (0..rng.below(3)).map(|_| (var(rng), Some(rng.coin()))).collect();
        let count = |t: &Tdd| -> Result<_, OperationError> {
            let mut c = t.counter_with(Retention::All, convention)?;
            let before = c.model_count()?;
            c.set_pins(&pins)?;
            let pinned = c.model_count()?;
            c.clear_pins();
            Ok((before, pinned, c.model_count()?))
        };
        a.check("counter", || count(g), || count(f), PartialEq::eq);
    }

    let held: Vec<VarId> = (1..=n).map(VarId).filter(|&v| g.vtree().leaf_of(v).is_some() || g.vtree().region_of(v).is_some()).collect();
    let pick = |rng: &mut Lcg| held[rng.below(held.len() as u64) as usize];
    let (x, b) = (pick(rng), rng.coin());
    let some: Vec<VarId> = held.iter().copied().filter(|_| rng.coin()).collect();
    let mutate = |t: &Tdd, op: &dyn Fn(&mut Tdd) -> Result<(), OperationError>| {
        let mut t = t.clone();
        op(&mut t).map(|()| t)
    };
    a.check("negate", || g.clone().negate(), || f.clone().negate(), same_value);
    a.check("condition_var", || g.clone().condition_var(x, b), || f.clone().condition_var(x, b), same_value);
    a.check("condition_vars", || g.clone().condition_vars(&some, b), || f.clone().condition_vars(&some, b), same_value);
    a.check("exists_var", || g.clone().exists_var(x), || f.clone().exists_var(x), same_value);
    a.check("exists_vars", || g.clone().exists_vars(&some), || f.clone().exists_vars(&some), same_value);
    a.check("forall_vars", || g.clone().forall_vars(&some), || f.clone().forall_vars(&some), same_value);
    let clause = [lit(x, b), lit(var(rng), rng.coin())];
    a.check("and_clause", || g.clone().and_clause(clause), || f.clone().and_clause(clause), same_value);
    a.check("or_cube", || g.clone().or_cube(clause), || f.clone().or_cube(clause), same_value);
    a.check("and_cube", || eng.and_cube(g.clone(), clause), || eng.and_cube(f.clone(), clause), same_value);
    a.check("renumber_top_down", || g.renumber_top_down(), || f.renumber_top_down(), same_value);
    a.check("minimize", || mutate(g, &|t| t.minimize()), || mutate(f, &|t| t.minimize()), same_value);
    a.check("trim", || mutate(g, &|t| t.trim(home).map(drop)), || Ok(f.clone()), same_value);
    a.check("reduce", || mutate(g, &|t| t.reduce(crate::reduce::ReductionPlan::default())), || mutate(f, &|t| t.reduce(crate::reduce::ReductionPlan::default())), same_value);
    a.check("attach_level_counts", || mutate(g, &|t| t.attach_level_counts()), || mutate(f, &|t| t.attach_level_counts()), same_value);
    a.check("marginalize all", || mutate(g, &|t| t.marginalize(|_| true)), || mutate(f, &|t| t.marginalize_levels(&[t.vtree().root()])), same_value);
    let model: Vec<Literal> = (1..=n).map(|v| lit(VarId(v), rng.coin())).collect();
    a.runs("insert_model", || mutate(g, &|t| t.insert_model(&model)));
    a.runs("remove_model", || mutate(g, &|t| t.remove_model(&model)));
    a.runs("maintain", || mutate(g, &|t| t.maintain().and_then(|mut m| m.insert_model(&model))));
    a.runs("filter_nodes", || g.filter_nodes(|_| true).map(drop));
    a.check(
        "rotation_search",
        || mutate(g, &|t| t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(drop)),
        || mutate(f, &|t| t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(drop)),
        same_value,
    );
    a.check(
        "rotation_multistart",
        || mutate(g, &|t| eng.rotation_multistart(t, &mut MinimizePairs, &MultistartConfig::default()).map(drop)),
        || mutate(f, &|t| eng.rotation_multistart(t, &mut MinimizePairs, &MultistartConfig::default()).map(drop)),
        same_value,
    );
    a.check(
        "pool_search",
        || mutate(g, &|t| eng.pool_search(&mut [t], &PoolSearchConfig::default()).map(drop)),
        || mutate(f, &|t| eng.pool_search(&mut [t], &PoolSearchConfig::default()).map(drop)),
        same_value,
    );
    let internal: Vec<VtreeIdx> = g.vtree().internal_bottomup().map(|(t, _, _)| t).collect();
    if let Some(&pivot) = internal.get(rng.below(internal.len().max(1) as u64) as usize) {
        let kind = if rng.coin() { RotationKind::Left } else { RotationKind::Right };
        a.runs("rotate_if", || mutate(g, &|t| t.rotate_if(&[RotationMove { pivot, kind }], usize::MAX, |_| true).map(drop)));
    }
    for t in g.vtree().bottomup() {
        a.check("rotate_marginal_cluster", || mutate(g, &|u| eng.rotate_marginal_cluster(u, t, 8, &mut Vec::new()).map(drop)), || Ok(f.clone()), same_value);
    }
    let rename: Vec<(VarId, VarId)> = match (some.first(), vars(rng).first()) {
        (Some(&p), Some(&q)) if p != q => vec![(p, q), (q, p)],
        _ => Vec::new(),
    };
    a.runs("rename_vars", || g.clone().rename_vars(&rename));
    let literal_x = crate::literal(g.vtree(), i32::try_from(x.0).unwrap());
    a.runs("substitute", || literal_x.clone().and_then(|l| g.clone().substitute(&[(var(&mut Lcg::new(x.0.into())), &l)])));

    let hv = home.vtree();
    a.runs("embed", || g.embed(hv, |v| v).map(|r| r.0));
    a.runs("embed_moving", || eng.embed_moving(g.clone(), hv, |v| v).map(|r| r.0).map_err(|r| r.error));
    a.runs("embed_over", || eng.embed_over(g, hv, |v| v, None).map(|r| r.0));
    a.runs("embedding_plan", || eng.embedding_plan(g.vtree(), hv, |v| v).and_then(|plan| eng.embed_with(g, &plan)));
    a.check("transport", || eng.transport(g, home, |v| v, 0).map(|(t, _)| t).map_err(|r| r.error), || Ok(f.clone()), same_value);
    a.runs("project_over", || eng.project_over(g, hv, Some, None));
    a.runs("project_to_vars", || g.project_to_vars(Some, n));
    a.runs("restructure_to", || g.clone().restructure_to(&Arc::new(Vtree::random(n, rng.below(1000))), usize::MAX));
    a.runs("Vtree::to_text", || Vtree::from_text(&g.vtree().to_text()));
    a.runs("Vtree::join", || Vtree::join(g.vtree(), &Vtree::leaf(VarId(n + 1))));
    a.runs("Vtree::graft", || Vtree::graft(&[Vtree::clone(g.vtree())], &[VarId(n + 1)]));
    a.runs("Vtree::project_to_vars", || g.vtree().project_to_vars(Some, n).ok_or(()));
    let mut text = Vec::new();
    a.runs("write_tdd", || crate::io::write_tdd(&mut text, g));
    a.runs("write_tdd_binary", || crate::io::write_tdd_binary(&mut Vec::new(), g));
}

#[test]
fn every_entry_point_on_a_diagram_with_regions_answers_as_the_smooth_form_or_refuses() {
    let eng = Engine::new();
    let mut audit = Audit::default();
    std::panic::set_hook(Box::new(|info| {
        let at = info.location().map(ToString::to_string).unwrap_or_default();
        if std::env::var_os("REGION_ENTRIES_TRACE").is_some() {
            eprintln!("panic at {at}: {info}");
        }
        *PANIC_AT.lock().unwrap() = at;
    }));
    let mut cases = 0;
    for seed in 0..1000u64 {
        let Some((home, n, f, g)) = region_case(&eng, seed, 10) else { continue };
        cases += 1;
        let mut rng = Lcg::new(seed);
        audit.case = format!("seed {seed}, {n} variables, {} regions", g.vtree().regions().count());
        audit_case(&mut audit, &eng, &home, &f, &g, n, &mut rng);
    }
    let _ = std::panic::take_hook();
    let report = audit.report();
    eprintln!("{cases} cases\n{report}");
    assert!(cases > 300, "{cases} cases");
    assert!(!report.contains("DIFFER") && !report.contains("PANIC"), "{report}");
}
