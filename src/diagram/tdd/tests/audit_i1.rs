//! No reachable pair of a result names a variable leaf `One`, and each level
//! names a leaf child `One` in all its reachable pairs or in none, for every
//! public producer on operands that satisfy both, on projections of one home.
//! [`EXPECTED_OPEN`] lists the producers known to name a leaf `One` today:
//! each must do so at least once, every other producer never. Every other
//! result on a home also passes `check_all_fast`, I1 over dead nodes
//! included, and minimizes to a diagram whose leaves are its support.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};

use num_rational::BigRational;

use crate::apply::{FilterOutcome, FoldCodes, FoldMonoid, FoldRange};
use crate::diagram::{Arithmetic, ChildPair, LeafLabel, LiteralWeights, NodeIdx, PairIndex, RationalWeights, TddNodeId, Weights};
use crate::reduce::{ContentTwinPolicy, ReductionPlan};
use crate::restructure::search::{MinimizePairs, MultistartConfig, PoolSearchConfig, RotationMove, RotationSearchConfig};
use crate::test_helpers::check::{check_all_fast, check_no_true_node, check_trimmed, reachable_true_labels};
use crate::test_helpers::rand_conj_over;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, RotationKind, VarId, VtreeIdx};
use crate::{Engine, Literal, OperationError};

use super::*;

/// The producers whose results name a leaf `One` today, with the reason. A
/// line marked `unpredicted` was found by this test, not by an audit.
const EXPECTED_OPEN: &[(&str, &str)] = &[
    ("expand_variables", "a plain root, on which a class of one literal the operand does not read is a leaf named One"),
];

// Predicted open, but no result named One on these inputs, so the audit
// holds them to I1: `and, weighted`, `marginalize_levels`,
// `and_marginalizing`, `rotate_marginal_cluster`, `prefix_fold [fixed]`.

#[derive(Default)]
struct Seen {
    runs: usize,
    refused: usize,
    named_one: usize,
    first_one: Option<String>,
    leaf_form: Vec<String>,
    panics: Vec<String>,
    false_with_nodes: usize,
    untrimmed: Vec<String>,
    off_support: Vec<String>,
    /// Unminimized results holding a node that denotes every assignment of
    /// its variables: contraction debt, which `minimize` must clear.
    true_node: usize,
}

#[derive(Default)]
struct Audit {
    seen: BTreeMap<&'static str, Seen>,
    case: String,
    /// The operands of the case, written out when a result breaches.
    inputs: String,
}

impl Audit {
    fn run<E: Debug>(&mut self, name: &'static str, call: impl FnOnce() -> Result<Tdd, E>) {
        let case = self.case.clone();
        let seen = self.seen.entry(name).or_default();
        match catch_unwind(AssertUnwindSafe(call)) {
            Err(e) => {
                seen.panics.push(format!("{case}: {}", panic_text(&*e)));
            }
            Ok(Err(_)) => seen.refused += 1,
            Ok(Ok(t)) => {
                seen.runs += 1;
                seen.false_with_nodes += usize::from(t.is_zero() && t.vtree().num_nodes() > 0);
                // The smooth form `marginalize_levels` leaves on a home's
                // root keeps its internal summed levels.
                let smooth = t.vtree().bottomup().any(|u| !t.vtree().node(u).is_leaf() && t.level(u).is_marginal());
                if t.vtree().on_home() && !smooth {
                    let fast = catch_unwind(AssertUnwindSafe(|| check_all_fast(&t, "result")));
                    if let Err(e) = check_trimmed(&t, t.is_canonical()).and(fast.map_err(|e| panic_text(&*e))) {
                        seen.untrimmed.push(format!("{case}: {e}"));
                    }
                    if !t.has_marginal_level() {
                        seen.true_node += usize::from(check_no_true_node(&t).is_err());
                        let mut m = t.clone();
                        let minimized = m.minimize().map_err(|e| e.to_string()).and_then(|()| check_trimmed(&m, true)).and_then(|()| check_no_true_node(&m));
                        if let Err(e) = minimized {
                            seen.off_support.push(format!("{case}: {e}; {}", self.inputs));
                        }
                    }
                }
                match catch_unwind(AssertUnwindSafe(|| reachable_true_labels(&t))) {
                    Err(_) => seen.panics.push(format!("{case}: the scan")),
                    Ok(Err(e)) => seen.leaf_form.push(format!("{case}: {e}")),
                    Ok(Ok(Some(w))) => {
                        seen.named_one += 1;
                        let inputs = &self.inputs;
                        seen.first_one.get_or_insert(format!("{case}: {w}; {inputs}"));
                    }
                    Ok(Ok(None)) => {}
                }
            }
        }
    }

    /// The breaches off the table, the table lines that never breached, and
    /// every leaf-form failure and panic; with a line per producer.
    fn verdict(&self) -> (Vec<String>, String) {
        let mut findings = Vec::new();
        let mut report = String::new();
        for (&name, s) in &self.seen {
            let open = EXPECTED_OPEN.iter().any(|&(n, _)| n == name);
            report += &format!(
                "{name}: {} results, {} named One, {} refused, {} false on a vtree with nodes, {} not trimmed, {} off the support once minimized, {} with a true node{}\n",
                s.runs,
                s.named_one,
                s.refused,
                s.false_with_nodes,
                s.untrimmed.len(),
                s.off_support.len(),
                s.true_node,
                s.first_one.as_ref().map(|w| format!(" (first {w})")).unwrap_or_default()
            );
            if let Some(e) = s.leaf_form.first() {
                findings.push(format!("{name}: leaf form fails {} times, first {e}", s.leaf_form.len()));
            }
            let held = !open;
            if let (true, Some(e)) = (held, s.untrimmed.first()) {
                findings.push(format!("{name}: not trimmed {} times, first {e}", s.untrimmed.len()));
            }
            if let (true, Some(e)) = (held, s.off_support.first()) {
                findings.push(format!("{name}: off the support once minimized {} times, first {e}", s.off_support.len()));
            }
            if let Some(p) = s.panics.first() {
                findings.push(format!("{name}: {} panics, first {p}", s.panics.len()));
            }
            match (open, &s.first_one) {
                (false, Some(w)) => findings.push(format!("{name}: names One off the table, first {w}")),
                (true, None) => findings.push(format!("{name}: on the table, names no One in {} results", s.runs)),
                _ => {}
            }
        }
        for &(name, _) in EXPECTED_OPEN {
            if !self.seen.contains_key(name) {
                findings.push(format!("{name}: on the table, never run"));
            }
        }
        (findings, report)
    }
}

fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default()
}

fn weights(n: u32, rng: &mut Lcg) -> Weights {
    let mut w = || BigRational::new((1 + rng.below(5) as i64).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..=n).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
    Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)
}

/// `t`, on a projection of a home of `onto`'s shape and variables, moved to
/// the projection of `onto` on the same variables.
fn reseated(t: &Tdd, onto: &Home) -> Tdd {
    let vars: Vec<VarId> = t.vtree().leaf_bottomup().map(|(_, v)| v).collect();
    let own = if vars.is_empty() { Vtree::empty_projection(onto) } else { Vtree::projection(onto, vars).unwrap() };
    let mut t = t.clone();
    // Safety: the two homes have one shape, so the two projections do.
    unsafe { t.reseat_vtree_unchecked(&own) };
    t
}

/// A random function of `vars`, minimized and trimmed: it names no leaf
/// `One` and sits on the projection of the home on its support.
fn operand(home: &Arc<Vtree>, vars: &[u32], rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj_over(home, vars, 4, 3, false, rng);
    for _ in 0..rng.below(3) {
        let var = VarId(vars[rng.below(vars.len() as u64) as usize]);
        f = match rng.below(3) {
            0 => f.condition_var(var, rng.coin()).unwrap(),
            1 => f.exists_vars(&[var]).unwrap(),
            _ => crate::test_helpers::and2(&f, &rand_conj_over(home, vars, 2, 3, false, rng)),
        };
    }
    f.minimize().unwrap();
    f.trim(&Home::new(Arc::clone(home))).unwrap();
    f
}

/// `t`'s variables and its models over them, character `i` of a row the
/// value of the `i`-th variable.
fn models(t: &Tdd) -> String {
    if t.vtree().num_nodes() == 0 {
        return if t.is_one() { "true".into() } else { "false".into() };
    }
    let vars = vars_of(t);
    let rows: Vec<String> = (0..1u32 << vars.len())
        .filter(|&r| {
            let cube = vars.iter().enumerate().map(|(i, &v)| Literal { var: v, sign: r >> i & 1 == 1 });
            t.clone().condition(cube).unwrap().is_sat().unwrap()
        })
        .map(|r| (0..vars.len()).map(|i| if r >> i & 1 == 1 { '1' } else { '0' }).collect::<String>())
        .collect();
    format!("on {:?} rows {}", vars.iter().map(|v| v.0).collect::<Vec<_>>(), rows.join(" "))
}

fn vars_of(t: &Tdd) -> Vec<VarId> {
    let v = t.vtree();
    v.bottomup().filter(|&i| v.node(i).is_leaf()).map(|i| v.leaf_var(i)).collect()
}

fn internal_nodes(t: &Tdd) -> Vec<VtreeIdx> {
    let v = t.vtree();
    v.bottomup().filter(|&i| !v.node(i).is_leaf()).collect()
}

fn pick<T: Copy>(v: &[T], rng: &mut Lcg) -> Option<T> {
    (!v.is_empty()).then(|| v[rng.below(v.len() as u64) as usize])
}

#[allow(clippy::too_many_lines)]
fn random_case(a: &mut Audit, eng: &Engine, home: &Arc<Vtree>, used: &[u32], rng: &mut Lcg) {
    let hm = Home::new(Arc::clone(home));
    let n = home.num_vars();
    let (f, g, h) = (operand(home, used, rng), operand(home, used, rng), operand(home, used, rng));
    a.inputs = format!("f {}, g {}, h {}", models(&f), models(&g), models(&h));
    for t in [&f, &g, &h] {
        a.run("operand", || Ok::<_, ()>(t.clone()));
    }
    let some = |rng: &mut Lcg| -> Vec<VarId> { used.iter().copied().filter(|_| rng.coin()).map(VarId).collect() };
    let lit = |v: VarId, sign: bool| Literal { var: v, sign };
    let i = rng.below(used.len() as u64) as usize;
    let (x, y) = (VarId(used[i]), VarId(used[(i + 1) % used.len()]));
    let (b1, b2) = (rng.coin(), rng.coin());
    let pair = [lit(x, b1), lit(y, b2)];
    let mutate = |t: &Tdd, op: &dyn Fn(&mut Tdd) -> Result<(), OperationError>| {
        let mut t = t.clone();
        op(&mut t).map(|()| t)
    };

    // Predicted to name no One.
    a.run("and", || eng.and(&f, &g));
    a.run("and_in_place", || eng.and_in_place(f.clone(), &g));
    a.run("and_in_place_loose", || eng.and_in_place_loose(f.clone(), &g));
    a.run("and_indexed", || PairIndex::new(&g).and_then(|index| eng.and_indexed(&f, &g, &index)));
    let mut own = Vec::new();
    for v in vars_of(&f) {
        if rng.coin() {
            own.push(lit(v, rng.coin()));
        }
    }
    // These panic on a constant on the empty vtree:
    // `constant_operands_do_not_panic`.
    let constant = f.vtree().num_nodes() == 0;
    if !constant {
        a.run("and_cube", || eng.and_cube(f.clone(), own.iter().copied()));
    }
    a.run("and_clause", || f.clone().and_clause(pair));
    a.run("or_cube", || f.clone().or_cube(pair));
    a.run("or", || eng.or(f.clone(), g.clone()));
    a.run("and_not", || eng.and_not(f.clone(), g.clone()));
    a.run("add", || eng.and_not(f.clone(), g.clone()).and_then(|d| eng.add(d, g.clone())));
    a.run("ite", || eng.ite(f.clone(), g.clone(), h.clone()));
    a.run("xor", || eng.xor(f.clone(), g.clone()));
    a.run("substitute", || {
        let mut l = crate::literal(&crate::vtree::Home::unchecked(home), i32::try_from(x.0).unwrap())?;
        l.trim(&hm)?;
        f.clone().substitute(&[(y, &l)])
    });
    a.run("restrict_to_care", || eng.restrict_to_care(f.clone(), &g).map(crate::apply::RestrictionOutcome::into_tdd));
    a.run("negate", || f.clone().negate());
    a.run("negate_with(Prune)", || crate::apply::with_plan::negate_with(eng, &f, ReductionPlan::Prune));
    a.run("transport", || eng.transport(&f, &hm, |v| v, 0).map(|(t, _)| t).map_err(|r| r.error));
    if let Some(pivot) = pick(&internal_nodes(&f), rng) {
        let kind = if rng.coin() { RotationKind::Left } else { RotationKind::Right };
        a.run("rotate_if", || mutate(&f, &|t| t.rotate_if(&[RotationMove { pivot, kind }], usize::MAX, |_| true).map(drop)));
    }
    a.run("rotation_search", || mutate(&f, &|t| t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(drop)));
    let mut multistart = MultistartConfig::default();
    multistart.search.max_inner_pairs = 64;
    if !constant {
        a.run("rotation_multistart", || mutate(&f, &|t| eng.rotation_multistart(t, &mut MinimizePairs, &multistart).map(drop)));
    }
    a.run("pool_search", || {
        let (mut p, mut q) = (f.clone(), g.clone());
        eng.pool_search(&mut [&mut p, &mut q], &PoolSearchConfig::default()).map(|_| p)
    });
    let target = Home::new(Vtree::random(n, rng.below(1000)));
    a.run("restructure_to", || {
        let mut t = f.clone();
        t.restructure_to(&target, usize::MAX).map(|_| t)
    });
    a.run("transport with a bound", || eng.transport(&f, &hm, |v| v, usize::MAX).map(|(t, _)| t).map_err(|r| format!("{r:?}")));
    a.run("renumber_top_down", || f.renumber_top_down());
    let model: Vec<Literal> = vars_of(&f).into_iter().map(|v| lit(v, rng.coin())).collect();
    if !constant {
        // The batch's diagram after the `minimize` that ends a batch.
        a.run("Maintenance::insert_model", || mutate(&f, &|t| t.maintain().and_then(|mut m| m.insert_model(&model)).and_then(|()| t.minimize())));
        a.run("Maintenance::remove_model", || mutate(&f, &|t| t.maintain().and_then(|mut m| m.remove_model(&model)).and_then(|()| t.minimize())));
        a.run("Tdd::insert_model", || mutate(&f, &|t| t.insert_model(&model).and_then(|()| t.minimize())));
    }
    a.run("reseat_vtree_unchecked", || {
        let mut t = f.clone();
        let same = Arc::clone(t.vtree());
        // Safety: the diagram's own vtree.
        unsafe { t.reseat_vtree_unchecked(&same) };
        Ok::<_, ()>(t)
    });
    let rename: Vec<(VarId, VarId)> = if x == y { Vec::new() } else { vec![(x, y), (y, x)] };
    a.run("rename_vars", || f.clone().rename_vars(&rename));
    a.run("filter_nodes_with(Prune)", || match crate::apply::with_plan::filter_nodes_with(eng, &f, |_| rng_keep(), ReductionPlan::Prune)? {
        FilterOutcome::Unchanged => Ok(f.clone()),
        FilterOutcome::Filtered { tdd, .. } | FilterOutcome::Unsatisfiable { tdd, .. } => Ok::<_, OperationError>(tdd),
    });
    a.run("or_many", || eng.or_many(vec![f.clone(), g.clone(), h.clone()]));
    a.run("nor_many", || eng.nor_many(vec![f.clone(), g.clone(), h.clone()]));
    a.run("operator &", || Ok::<_, ()>(f.clone() & g.clone()));
    a.run("operator |", || Ok::<_, ()>(f.clone() | g.clone()));
    a.run("operator !", || Ok::<_, ()>(!f.clone()));
    a.run("and_restoring", || eng.and_restoring(&f, &g).map_err(|r| r.error));
    a.run("try_clone", || eng.try_clone(&f));
    a.run("clone", || Ok::<_, ()>(f.clone()));
    a.run("trim", || mutate(&f, &|t| eng.trim(t, &hm).map(drop)));
    a.run("reduce(Prune)", || mutate(&f, &|t| t.reduce(ReductionPlan::Prune)));
    a.run("reduce(Contract)", || mutate(&f, &|t| t.reduce(ReductionPlan::Contract)));
    a.run("attach_level_counts", || mutate(&f, &|t| t.attach_level_counts()));
    a.run("unnamed_nodes", || mutate(&f, &|t| eng.unnamed_nodes(t).map(drop)));
    a.run("literal", || eng.literal(&crate::vtree::Home::unchecked(home), lit(x, b1)));
    a.run("one", || Ok::<_, ()>(eng.one(&crate::vtree::Home::unchecked(home))));
    a.run("zero", || Ok::<_, ()>(eng.zero(&crate::vtree::Home::unchecked(home))));
    a.run("cube", || eng.cube(&crate::vtree::Home::unchecked(home), pair));
    a.run("clause", || eng.clause(&crate::vtree::Home::unchecked(home), pair));
    a.run("from_models", || Tdd::from_models(&crate::vtree::Home::unchecked(home), &[x, y], &[0b01, 0b10]));
    a.run("from_columns", || eng.from_columns(&crate::vtree::Home::unchecked(home), &[&[x, y][..]], &[&[1u32, 2][..]], crate::diagram::RowSelection::All));
    // Cube x = 1, y free: value bits, then fixed bits.
    a.run("from_cubes", || eng.from_cubes(&crate::vtree::Home::unchecked(home), &[x, y], &[0b01, 0b01]));
    a.run("expand_variables", || {
        let classes = (1..=n).map(|v| vec![lit(VarId(v), true)]).collect();
        let expansion = crate::restructure::VariableExpansion { classes, constants: Vec::new(), free: Vec::new(), num_vars: n };
        eng.expand_variables(&f, &expansion).map_err(|e| format!("{e:?}"))
    });
    a.run("rotation_search_with", || {
        let config = RotationSearchConfig { max_inner_pairs: 64, ..Default::default() };
        mutate(&f, &|t| eng.rotation_search_with(t, &mut MinimizePairs, &mut crate::restructure::search::Greedy, &config).map(drop))
    });
    let home_internal: Vec<VtreeIdx> = home.bottomup().filter(|&i| !home.node(i).is_leaf()).collect();
    if let Some(pivot) = pick(&home_internal, rng) {
        let rotation = RotationMove { pivot, kind: if rng.coin() { RotationKind::Left } else { RotationKind::Right } };
        let crossed = rng.coin();
        a.run("rotate_pool_if", || {
            let (mut p, mut q) = (f.clone(), g.clone());
            eng.rotate_pool_if(&mut [&mut p, &mut q], crate::restructure::search::PoolMove { rotation, crossed }, usize::MAX, |_| true).map(|_| p)
        });
    }
    a.run("write_tdd, read_tdd", || {
        let mut bytes = Vec::new();
        crate::io::write_tdd(&mut bytes, &f).map_err(|e| e.to_string())?;
        crate::io::read_tdd(&mut bytes.as_slice(), &crate::vtree::Home::unchecked(home)).map_err(|e| e.to_string())
    });
    a.run("write_tdd_binary, read_tdd_binary", || {
        let mut bytes = Vec::new();
        crate::io::write_tdd_binary(&mut bytes, &f).map_err(|e| e.to_string())?;
        crate::io::read_tdd_binary(&mut bytes.as_slice(), &crate::vtree::Home::unchecked(home)).map_err(|e| e.to_string())
    });
    a.run("TddBuilder::finish", || rebuilt(eng, &f));

    // Predicted to name One on some inputs.
    let b = rng.coin();
    let vs = some(rng);
    a.run("condition", || f.clone().condition([lit(x, b)]));
    a.run("condition_var", || f.clone().condition_var(x, b));
    a.run("condition_vars", || f.clone().condition_vars(&vs, b));
    a.run("exists_var", || f.clone().exists_var(x));
    a.run("exists_vars", || f.clone().exists_vars(&vs));
    a.run("forall_vars", || f.clone().forall_vars(&vs));
    a.run("and_exists", || eng.and_exists(f.clone(), g.clone(), &vs));
    a.run("and_exists_with(Product)", || eng.and_exists_with(f.clone(), g.clone(), &vs, crate::Quantification::Product));
    a.run("exists_vars_with(Prune)", || eng.exists_vars_with(f.clone(), &vs, ReductionPlan::Prune));
    if let Some(at) = pick(&internal_nodes(&f), rng) {
        a.run("at_least_distinct", || eng.at_least_distinct(&f, at, 2));
        a.run("project_to_subtree", || eng.project_to_subtree(&f, at));
        a.run("marginalize_levels", || mutate(&f, &|t| t.marginalize(crate::test_helpers::summing(t.vtree(), &[at]))));
        a.run("and_marginalizing", || crate::test_helpers::and_marginalizing_under(eng, &f, &g, &[at]));
        let under_at: Vec<VarId> = f.vtree().leaf_bottomup().filter(|&(l, _)| f.vtree().in_subtree(l, at)).map(|(_, v)| v).collect();
        a.run("marginalize", || mutate(&f, &|t| t.marginalize(|v| under_at.contains(&v))));
        a.run("and_marginalizing_where", || eng.and_marginalizing_where(&f, &g, |v| under_at.contains(&v)));
        a.run("rotate_marginal_cluster", || {
            let mut t = f.clone();
            t.marginalize(crate::test_helpers::summing(t.vtree(), &[at]))?;
            let root = t.vtree().root();
            eng.rotate_marginal_cluster(&mut t, root, 8, &mut Vec::new()).map(|_| t)
        });
    }
    // An unminimized operand: some nodes dropped, nothing contracted.
    if let Ok(FilterOutcome::Filtered { tdd: u, .. }) = crate::apply::with_plan::filter_nodes_with(eng, &f, |_| rng_keep(), ReductionPlan::Prune)
        && matches!(reachable_true_labels(&u), Ok(None))
    {
        a.run("minimize", || mutate(&u, &|t| t.minimize()));
        a.run("reduce(Full)", || mutate(&u, &|t| t.reduce(ReductionPlan::Full(ContentTwinPolicy::Fresh))));
        a.run("negate_with(Full)", || crate::apply::with_plan::negate_with(eng, &u, ReductionPlan::Full(ContentTwinPolicy::Fresh)));
    }
    a.run("filter_nodes_with(Full)", || match crate::apply::with_plan::filter_nodes_with(eng, &f, |_| rng_keep(), ReductionPlan::default())? {
        FilterOutcome::Unchanged => Ok(f.clone()),
        FilterOutcome::Filtered { tdd, .. } | FilterOutcome::Unsatisfiable { tdd, .. } => Ok::<_, OperationError>(tdd),
    });
    let weighted = Home::weighted(Vtree::clone(home), weights(n, rng)).unwrap();
    let (fw, gw) = (reseated(&f, &weighted), reseated(&g, &weighted));
    a.run("and, weighted", || eng.and(&fw, &gw));
    a.run("or, weighted", || eng.or(fw.clone(), gw.clone()));
}

thread_local! {
    static KEEP: std::cell::Cell<u64> = const { std::cell::Cell::new(0x5eed) };
}

/// A deterministic coin for the node filters, whose closures cannot borrow
/// the case's generator.
fn rng_keep() -> bool {
    KEEP.with(|k| {
        let mut r = Lcg::new(k.get());
        let keep = r.below(4) != 0;
        k.set(k.get().wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1));
        keep
    })
}

/// `f` built again pair by pair through the public builder.
fn rebuilt(eng: &Engine, f: &Tdd) -> Result<Tdd, String> {
    let v = f.vtree();
    if v.num_nodes() == 0 || v.node(v.root()).is_leaf() {
        return Ok(f.clone());
    }
    let mut b = Tdd::builder(eng, v).map_err(|e| format!("{e:?}"))?;
    let mut ids: Vec<Vec<NodeIdx>> = vec![Vec::new(); v.num_nodes()];
    let map = |ids: &[Vec<NodeIdx>], child: VtreeIdx, raw: u32| if v.node(child).is_leaf() { NodeIdx(raw) } else { ids[child.idx()][raw as usize] };
    for t in v.bottomup().filter(|&t| !v.node(t).is_leaf()) {
        let (l, r) = v.children(t);
        let level = f.level(t);
        for j in 0..level.slot_count() {
            let pairs: Vec<ChildPair> = level.pairs_iter_of_idx(j).map(|p| ChildPair::new(map(&ids, l, p.left.raw()), map(&ids, r, p.right.raw()))).collect();
            let id = b.push(eng, t, &pairs).map_err(|e| format!("{e:?}"))?;
            ids[t.idx()].push(id);
        }
    }
    let out = f.output();
    b.finish(TddNodeId { vtree: out.vtree, local: ids[out.vtree.idx()][out.local.0 as usize] }).map_err(|e| format!("{e:?}"))
}

/// A diagram built pair by pair on `vtree`, its output the last node pushed
/// at the root.
fn built(eng: &Engine, vtree: &Arc<Vtree>, nodes: &[(VtreeIdx, Vec<(u32, u32)>)]) -> Tdd {
    let mut b = Tdd::builder(eng, vtree).unwrap();
    let mut last = NodeIdx(0);
    for (t, pairs) in nodes {
        let pairs: Vec<ChildPair> = pairs.iter().map(|&(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r))).collect();
        last = b.push(eng, *t, &pairs).unwrap();
    }
    // Seated unchecked: `finish` on a home would trim a padded operand.
    unsafe { b.finish_unchecked(TddNodeId { vtree: nodes.last().unwrap().0, local: last }) }
}

const POS: u32 = LeafLabel::Pos as u32;
const NEG: u32 = LeafLabel::Neg as u32;

fn leaf(v: u32) -> Vtree {
    Vtree::leaf(VarId(v))
}

/// The inputs of the audit, each a case: ((x y) z) with x, y, z = 1, 2, 3.
#[allow(clippy::too_many_lines)]
fn fixed_cases(a: &mut Audit, eng: &Engine) {
    a.inputs.clear();
    let lit = |v: u32, sign: bool| Literal { var: VarId(v), sign };
    let xyz = Arc::clone(&Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap()));
    let hm = Home::new(Arc::clone(&xyz));
    let (root, xy) = (xyz.root(), xyz.children(xyz.root()).0);
    let trimmed = |mut t: Tdd| {
        t.trim(&hm).unwrap();
        t
    };
    a.case = "or_cube(x or z on {x, z}, [x, y]) on ((x y) z)".into();
    let x_or_z = trimmed(eng.clause(&crate::vtree::Home::unchecked(&xyz), [1, 3]).unwrap());
    a.run("or_cube by complement [fixed]", || x_or_z.clone().or_cube([lit(1, true), lit(2, true)]));
    a.case = "or_cube(x or z on {x, z}, [])".into();
    a.run("or_cube, empty cube [fixed]", || x_or_z.clone().or_cube::<Literal>([]));

    // x on ((x y) z), unminimized: y is read and both its values agree.
    let unminimized = |second: (u32, u32)| built(eng, &xyz, &[(xy, vec![(POS, POS), second]), (root, vec![(0, POS)])]);
    let x_y_any = unminimized((POS, NEG));
    let y_x_any = unminimized((NEG, POS));
    a.case = "restructure_to (x (y z)) from ((x y) z), level (x y) {(Pos, Pos), (Neg, Pos)}, root {(n0, Pos)}".into();
    let x_yz = Arc::clone(&Home::new(Vtree::join(&leaf(1), &Vtree::join(&leaf(2), &leaf(3)).unwrap()).unwrap()));
    a.run("restructure_to [fixed]", || {
        let mut t = y_x_any.clone();
        t.restructure_to(&x_yz, usize::MAX).map(|_| t)
    });
    a.run("transport with a bound [fixed]", || eng.transport(&y_x_any, &Home::new(Arc::clone(&x_yz)), |v| v, usize::MAX).map(|(t, _)| t).map_err(|r| format!("{r:?}")));
    a.run("rotation_search [fixed]", || {
        let mut t = y_x_any.clone();
        t.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default()).map(|_| t)
    });
    a.run("pool_search [fixed]", || {
        let (mut p, mut q) = (y_x_any.clone(), x_y_any.clone());
        eng.pool_search(&mut [&mut p, &mut q], &PoolSearchConfig::default()).map(|_| p)
    });
    a.case = "level (x y) {(Pos, Pos), (Pos, Neg)}, root {(n0, Pos)}".into();
    a.run("minimize [fixed]", || {
        let mut t = x_y_any.clone();
        t.minimize().map(|()| t)
    });

    a.case = "x or not x on the projection on x, weighted".into();
    let weighted = Home::weighted(Vtree::clone(&xyz), weights(3, &mut Lcg::new(7))).unwrap();
    let x = trimmed(crate::literal(&crate::vtree::Home::unchecked(&xyz), 1).unwrap());
    let xw = reseated(&x, &weighted);
    a.run("or of x and not x, weighted [fixed]", || eng.or(xw.clone(), xw.clone().negate().unwrap()));
    a.run("add of x and not x, weighted [fixed]", || eng.add(xw.clone(), xw.clone().negate().unwrap()));
    a.case = "x or not x on the projection on x".into();
    a.run("or of x and not x [fixed]", || eng.or(x.clone(), x.clone().negate().unwrap()));

    a.case = "(x, y) root {(Pos, Pos), (Neg, Neg)}, key the leaf x".into();
    let two = Arc::clone(&Home::new(Vtree::join(&leaf(1), &leaf(2)).unwrap()));
    let same = built(eng, &two, &[(two.root(), vec![(POS, POS), (NEG, NEG)])]);
    let key = two.children(two.root()).0;
    a.run("project_to_subtree on a leaf [fixed]", || eng.project_to_subtree(&same, key));
    a.run("at_least_distinct on a leaf [fixed]", || eng.at_least_distinct(&same, key, 1));

    a.case = "and_clause(y on {y}, [x, -x]) on (x y)".into();
    let y = {
        let mut t = crate::literal(&crate::vtree::Home::unchecked(&two), 2).unwrap();
        t.trim(&Home::new(Arc::clone(&two))).unwrap();
        t
    };
    a.run("and_clause, tautological [fixed]", || y.clone().and_clause([1, -1]));

    a.case = "rename_vars(x and y and z on (x (y z)), [(y, z)])".into();
    let all = eng.cube(&crate::vtree::Home::unchecked(&x_yz), [1, 2, 3]).unwrap();
    a.run("rename_vars [fixed]", || all.clone().rename_vars(&[(VarId(2), VarId(3))]));

    a.case = "constants and constructors on ((x y) z)".into();
    a.run("negate of false on a vtree with nodes [fixed]", || Tdd::zero(&crate::vtree::Home::unchecked(&xyz)).negate());
    a.run("one on a vtree with nodes [fixed]", || Ok::<_, ()>(Tdd::one(&crate::vtree::Home::unchecked(&xyz))));
    a.run("literal [fixed]", || crate::literal(&crate::vtree::Home::unchecked(&xyz), 1));
    a.run("cube [fixed]", || eng.cube(&crate::vtree::Home::unchecked(&xyz), [1, -2]));
    a.run("clause [fixed]", || eng.clause(&crate::vtree::Home::unchecked(&xyz), [1, -2]));
    a.run("from_models [fixed]", || Tdd::from_models(&crate::vtree::Home::unchecked(&xyz), &[VarId(1)], &[1]));

    a.case = "tag_level of ((x y) z) = (x and y and z) or (not (x and y) and not z) at (x y), both nodes tagged 0".into();
    a.run("tag_level [fixed]", || {
        let s = |e: OperationError| e.to_string();
        let f = eng.or(eng.cube(&crate::vtree::Home::unchecked(&xyz), [1, 2, 3]).map_err(s)?, eng.and(&eng.clause(&crate::vtree::Home::unchecked(&xyz), [-1, -2]).map_err(s)?, &eng.literal(&crate::vtree::Home::unchecked(&xyz), -3).map_err(s)?).map_err(s)?).map_err(s)?;
        let v = f.vtree();
        let at = v.bottomup().find(|&t| !v.node(t).is_leaf() && v.node(v.children(t).0).is_leaf() && v.node(v.children(t).1).is_leaf()).ok_or("no level over x y")?;
        let slots = f.reference_slot_count(at);
        let into = Arc::clone(&Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &Vtree::balanced_over(&[VarId(4), VarId(5)]).unwrap()).unwrap()));
        eng.tag_level(&f, at, &into, |v| v, &vec![Some(0); slots], &[VarId(4), VarId(5)], &[0b10]).map_err(|e| format!("{e:?}"))
    });

    a.case = "prefix_fold, sums every store makes on day 0 only".into();
    a.run("prefix_fold [fixed]", prefix_fold_case);
}

struct Sum;
impl FoldMonoid for Sum {
    type Value = u64;
    fn value(&self, code: u64) -> u64 {
        code
    }
    fn combine(&self, a: &u64, b: &u64) -> u64 {
        a + b
    }
}

/// The running totals of `prefix_fold`'s first example with every sale on
/// day 0, so that the totals do not read the day of an open store.
fn prefix_fold_case() -> Result<Tdd, String> {
    let chain = |low: Vtree| Vtree::join(&leaf(1), &Vtree::join(&leaf(2), &Vtree::join(&leaf(3), &low).unwrap()).unwrap()).unwrap();
    let vtree = Arc::clone(&Home::new(chain(Vtree::join(&leaf(4), &leaf(5)).unwrap())));
    let into = Arc::new(chain(Vtree::join(&leaf(4), &Vtree::join(&leaf(5), &leaf(6)).unwrap()).unwrap()));
    let row = |store: u64, day: u64, amount: u64| store | (day >> 1) << 1 | (day & 1) << 2 | amount << 3;
    let vars: Vec<VarId> = (1..=5).map(VarId).collect();
    let hm = Home::new(Arc::clone(&vtree));
    let mut sales = Tdd::from_models(&crate::vtree::Home::unchecked(&vtree), &vars, &[row(0, 0, 2), row(1, 0, 1)]).map_err(|e| e.to_string())?;
    sales.trim(&hm).map_err(|e| e.to_string())?;
    let open: Vec<u64> = (0..4).map(|d| row(0, d, 0)).chain((0..3).map(|d| row(1, d, 0))).collect();
    let mut open = Tdd::from_models(&crate::vtree::Home::unchecked(&vtree), &vars[..3], &open).map_err(|e| e.to_string())?;
    open.trim(&hm).map_err(|e| e.to_string())?;
    let coding = |totals: &[u64]| FoldCodes { into: into.clone(), code_vars: vec![VarId(4), VarId(5), VarId(6)], codes: totals.to_vec() };
    Engine::new()
        .prefix_fold(&open, &sales, &[VarId(2), VarId(3)], &[VarId(4), VarId(5)], FoldRange::AtMost, &Sum, coding)
        .map_err(|e| format!("{e:?}"))
}

/// True and false on the empty vtree, through the entries that panic on them
/// today: an index past an empty vtree's nodes, and a draw below zero in
/// the multistart.
#[test]
fn audit_i1_constant_operands_do_not_panic() {
    let eng = Engine::new();
    let home = Arc::clone(&Home::new(Vtree::balanced(3)));
    let hm = Home::new(Arc::clone(&home));
    for mut c in [Tdd::one(&crate::vtree::Home::unchecked(&home)), Tdd::zero(&crate::vtree::Home::unchecked(&home))] {
        c.trim(&hm).unwrap();
        assert_eq!(c.vtree().num_nodes(), 0);
        let none: [Literal; 0] = [];
        let _ = catch_unwind(AssertUnwindSafe(|| eng.and_cube(c.clone(), none))).expect("and_cube");
        let _ = catch_unwind(AssertUnwindSafe(|| c.clone().insert_model(none))).expect("insert_model");
        let _ = catch_unwind(AssertUnwindSafe(|| c.clone().maintain().and_then(|mut m| m.remove_model(none)))).expect("remove_model");
        let mut config = MultistartConfig::default();
        config.search.max_inner_pairs = 64;
        let _ = catch_unwind(AssertUnwindSafe(|| eng.rotation_multistart(&mut c.clone(), &mut MinimizePairs, &config))).expect("rotation_multistart");
    }
}

/// Half the cases use every variable of a small home, half two to five
/// variables scattered over a home of up to 24.
#[test]
fn audit_i1_every_producer() {
    let eng = Engine::new();
    let mut audit = Audit::default();
    for seed in 0..160u64 {
        let mut rng = Lcg::new(seed ^ 0x11a7);
        let scattered = seed % 2 == 1;
        let n = if scattered { 8 + rng.below(17) as u32 } else { 3 + rng.below(6) as u32 };
        let home = Arc::clone(&Home::new(Vtree::random(n, seed)));
        let mut used: Vec<u32> = (1..=n).collect();
        if scattered {
            for i in (1..used.len()).rev() {
                used.swap(i, rng.below(i as u64 + 1) as usize);
            }
            used.truncate(2 + rng.below(4) as usize);
        }
        audit.case = format!("seed {seed}, {n} variables, operands over {used:?}");
        random_case(&mut audit, &eng, &home, &used, &mut rng);
    }
    fixed_cases(&mut audit, &eng);
    let (findings, report) = audit.verdict();
    eprintln!("{report}");
    assert!(findings.is_empty(), "{}\n{report}", findings.join("\n"));
}

/// E3: `minimize` of x ∧ z on ((x y) z), level (x y) {(Pos, Pos), (Pos, Neg)},
/// on a home, stopped at each poll in turn. Each stop returns `Err` and
/// leaves no `One`, and the finished minimization names none.
#[test]
fn audit_i1_minimize_stopped_after_a_contraction() {
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let eng = Engine::new();
    let xyz = Arc::clone(&Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap()));
    let (root, xy) = (xyz.root(), xyz.children(xyz.root()).0);
    let f = built(&eng, &xyz, &[(xy, vec![(POS, POS), (POS, NEG)]), (root, vec![(0, POS)])]);
    let polls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&polls);
    let counting = StopCallback::new(move |_, _| {
        seen.fetch_add(1, Ordering::Relaxed);
        StopDecision::Continue
    });
    {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(counting)));
        eng.minimize(&mut f.clone()).unwrap();
    }
    let mut outcomes = Vec::new();
    for allowed in 0..polls.load(Ordering::Relaxed) {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let stop = StopCallback::new(move |_, _| if seen.fetch_add(1, Ordering::Relaxed) >= allowed { StopDecision::Stop } else { StopDecision::Continue });
        let mut m = f.clone();
        let done = {
            let _scope = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(stop)));
            eng.minimize(&mut m)
        };
        outcomes.push((allowed, done.is_err(), reachable_true_labels(&m)));
    }
    assert!(outcomes.iter().all(|(_, refused, _)| *refused), "{outcomes:?}");
    assert!(outcomes.iter().all(|(_, _, labels)| matches!(labels, Ok(None))), "a stop leaves y named One: {outcomes:?}");
    let mut done = f.clone();
    eng.minimize(&mut done).unwrap();
    assert_eq!(reachable_true_labels(&done), Ok(None));
    assert_eq!(done.vtree().num_leaves(), 2, "x and z");
}

/// ((a b) c) with a, b, c = 1, 2, 3; `f` is c, its level (a b) one node
/// naming both leaves `One`.
fn c_over_true(eng: &Engine) -> (Arc<Vtree>, Tdd) {
    let abc = Arc::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap());
    let (root, ab) = (abc.root(), abc.children(abc.root()).0);
    let one = LeafLabel::One as u32;
    let f = built(eng, &abc, &[(ab, vec![(one, one)]), (root, vec![(0, POS)])]);
    (abc, f)
}

/// T1a: `g` = (a or b) and c summed over the subtree (a b), `f` naming a and
/// b `One` in every pair. Not refused: `f`'s identity at (a b) admits the
/// summed level, and the count is the unsummed conjunction's.
#[test]
fn audit_i1_and_of_a_summed_subtree_beside_true_leaves() {
    let eng = Engine::new();
    let (abc, f) = c_over_true(&eng);
    let ab = abc.children(abc.root()).0;
    let mut g = eng.and(eng.clause(&crate::vtree::Home::unchecked(&abc), [1, 2]).unwrap(), eng.literal(&crate::vtree::Home::unchecked(&abc), 3).unwrap()).unwrap();
    g.marginalize(crate::test_helpers::summing(g.vtree(), &[ab])).unwrap();
    let got = eng.and(&f, &g).map(|t| t.model_count());
    assert_eq!(got, Ok(Ok(3u32.into())));
}

/// T1c: T1a through `and_marginalizing` with the root, the summed subtree's
/// parent, as the target: the same count.
#[test]
fn audit_i1_and_marginalizing_of_a_summed_subtree_beside_true_leaves() {
    let eng = Engine::new();
    let (abc, f) = c_over_true(&eng);
    let ab = abc.children(abc.root()).0;
    let mut g = eng.and(eng.clause(&crate::vtree::Home::unchecked(&abc), [1, 2]).unwrap(), eng.literal(&crate::vtree::Home::unchecked(&abc), 3).unwrap()).unwrap();
    g.marginalize(crate::test_helpers::summing(g.vtree(), &[ab])).unwrap();
    let got = crate::test_helpers::and_marginalizing_under(&eng, &f, &g, &[abc.root()]).map(|t| t.model_count());
    assert_eq!(got, Ok(Ok(3u32.into())));
}

/// T1b: `g` = (x or y) and z summed at the leaf x, `f` = y and z naming x
/// `One` in every pair; integer and weighted. Nothing refuses it; the
/// answer equals the conjunction's before the sum.
#[test]
#[ignore = "not refused: answers as the unsummed conjunction"]
fn audit_i1_and_of_a_summed_leaf_beside_a_true_leaf() {
    let eng = Engine::new();
    let one = LeafLabel::One as u32;
    for weighted in [false, true] {
        let mut xyz = Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap();
        if weighted {
            xyz = xyz.with_weights(weights(3, &mut Lcg::new(3))).unwrap();
        }
        let xyz = Arc::new(xyz);
        let (root, xy) = (xyz.root(), xyz.children(xyz.root()).0);
        let x = xyz.children(xy).0;
        let f = built(&eng, &xyz, &[(xy, vec![(one, POS)]), (root, vec![(0, POS)])]);
        let mut g = eng.and(eng.clause(&crate::vtree::Home::unchecked(&xyz), [1, 2]).unwrap(), eng.literal(&crate::vtree::Home::unchecked(&xyz), 3).unwrap()).unwrap();
        let want = eng.and(eng.cube(&crate::vtree::Home::unchecked(&xyz), [2, 3]).unwrap(), &g).unwrap();
        g.marginalize(crate::test_helpers::summing(g.vtree(), &[x])).unwrap();
        let value = |t: &Tdd| if weighted { format!("{:?}", t.weighted_value().map(|v| v.map(|v| crate::test_helpers::exact_weight(&v)))) } else { format!("{:?}", t.model_count()) };
        let got = catch_unwind(AssertUnwindSafe(|| eng.and(&f, &g).map(|t| value(&t)))).map_err(|_| "a panic");
        eprintln!("T1b weighted {weighted}: {got:?}, the unsummed conjunction {}", value(&want));
        assert_eq!(got, Ok(Ok(value(&want))), "weighted {weighted}");
    }
}

/// The variables whose leaves every reachable pair of `t` above them names
/// `One`; `Err` for a leaf some of those pairs name `One` and others not.
fn true_leaves(t: &Tdd) -> Result<Vec<VarId>, String> {
    let v = t.vtree();
    let reach = t.reachable_nodes();
    let mut named = Vec::new();
    for p in v.bottomup() {
        if v.node(p).is_leaf() {
            continue;
        }
        let (l, r) = v.children(p);
        for (child, is_left) in [(l, true), (r, false)] {
            if !v.node(child).is_leaf() {
                continue;
            }
            let sides: Vec<bool> = (0..t.level(p).slot_count())
                .filter(|&j| reach[p.idx()].get(j).copied().unwrap_or(false))
                .flat_map(|j| t.level(p).pairs_iter_of_idx(j).map(move |q| if is_left { q.left } else { q.right }).collect::<Vec<_>>())
                .map(|side| side.raw() == LeafLabel::One as u32)
                .collect();
            match (sides.iter().all(|&o| o), sides.iter().any(|&o| o)) {
                (true, true) => named.push(v.leaf_var(child)),
                (false, true) => return Err(format!("leaf {} named One in some pairs", v.leaf_var(child).0)),
                _ => {}
            }
        }
    }
    named.sort_unstable();
    Ok(named)
}

/// S1 to S4: `exists` through the structural entry with a pruning
/// reduction names each target leaf `One` in every pair that reaches it, and
/// no other leaf.
#[test]
fn audit_i1_exists_with_pruning_names_exactly_the_targets() {
    let eng = Engine::new();
    let axb = Arc::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap());
    let xyzc = Arc::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &Vtree::join(&leaf(3), &leaf(4)).unwrap()).unwrap());
    let or_cubes = |vtree: &Arc<Vtree>, cubes: &[&[i32]]| {
        cubes.iter().map(|c| eng.cube(&crate::vtree::Home::unchecked(vtree), c.iter().copied()).unwrap()).reduce(|f, g| eng.or(f, g).unwrap()).unwrap()
    };
    // a, x, b = 1, 2, 3; x, y, z, c = 1, 2, 3, 4.
    let cases: [(&str, Tdd, Vec<VarId>); 4] = [
        ("S1 (a <-> x) and b", or_cubes(&axb, &[&[1, 2, 3], &[-1, -2, 3]]), vec![VarId(2)]),
        ("S2", or_cubes(&axb, &[&[1, 2, 3], &[-1, -2, -3]]), vec![VarId(2)]),
        ("S3", or_cubes(&axb, &[&[1, 2, 3], &[1, -2, -3]]), vec![VarId(2)]),
        ("S4", or_cubes(&xyzc, &[&[1, 2, 3, 4], &[-1, 2, -3, 4]]), vec![VarId(1), VarId(2), VarId(3)]),
    ];
    for (what, f, targets) in cases {
        assert_eq!(reachable_true_labels(&f), Ok(None), "{what}: the operand");
        let got = eng.exists_vars_with(f, &targets, ReductionPlan::Prune).unwrap();
        assert_eq!(true_leaves(&got), Ok(targets.clone()), "{what}");
        if what == "S3" {
            let b = got.vtree().root();
            let (one, literal) = crate::test_helpers::check::labels_at(&got, b.idx(), false);
            assert!(!one && literal, "{what}: b stays a literal");
        }
    }
}

/// ((x y) z) with x, y, z = 1, 2, 3 on a home weighted by `ws`, x ∧ z with
/// the level (x y) {(Pos, Pos), (Pos, Neg)}: y a full twin side.
fn x_and_z_over_a_twin(eng: &Engine, ws: Weights) -> (Arc<Vtree>, Tdd) {
    let xyz = Arc::clone(&Home::weighted(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap(), ws).unwrap());
    let (root, xy) = (xyz.root(), xyz.children(xyz.root()).0);
    let f = built(eng, &xyz, &[(xy, vec![(POS, POS), (POS, NEG)]), (root, vec![(0, POS)])]);
    (xyz, f)
}

#[test]
fn audit_i1_weighted_minimize_keeps_the_value() {
    let eng = Engine::new();
    let (_, mut f) = x_and_z_over_a_twin(&eng, weights(3, &mut Lcg::new(11)));
    let before = f.weighted_value().unwrap().unwrap().as_rational().into_owned();
    f.minimize().unwrap();
    assert_eq!(vars_of(&f), [VarId(1), VarId(3)]);
    assert_eq!(reachable_true_labels(&f), Ok(None));
    assert_eq!(f.weighted_value().unwrap().unwrap().as_rational().into_owned(), before);
}

#[test]
fn audit_i1_weighted_and_across_projections_is_on_the_variables_used() {
    let eng = Engine::new();
    let home = Home::weighted(Vtree::balanced(4), weights(4, &mut Lcg::new(5))).unwrap();
    let on = |vars: &[u32], clause: &[i32]| {
        let own = Vtree::projection(&home, vars.iter().map(|&v| VarId(v))).unwrap();
        eng.clause(&crate::vtree::Home::unchecked(&own), clause.iter().copied()).unwrap()
    };
    let (f, g) = (on(&[1, 2], &[1, 2]), on(&[1, 2, 3], &[1, -2]));
    let mut h = eng.and(&f, &g).unwrap();
    assert_eq!(vars_of(&h), [VarId(1), VarId(2)], "the product keeps the twins at x2");
    assert_eq!(reachable_true_labels(&h), Ok(None));
    assert!(!crate::test_helpers::trim_by_rebuild(&mut h.clone(), &home).unwrap());
    h.minimize().unwrap();
    assert_eq!(vars_of(&h), [VarId(1)], "x1 or x2, and x1 or not x2, is x1");
    assert!(h.weights().is_some());
}
