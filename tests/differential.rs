//! A randomized differential suite: random formulas, random vtrees, and every
//! answer checked against enumeration or against a second route to the same
//! answer.
//!
//! The suite is one `#[ignore]` test so an ordinary `cargo test` skips it;
//! `cargo test -- --ignored differential` runs it. It draws a case, runs the
//! whole battery on it, and draws the next until its time is up.
//! `TIDIDI_FUZZ_SECONDS` (default 60) sets the budget and `TIDIDI_FUZZ_SEED`
//! the stream it starts from; with the seed unset the run draws one from the
//! clock, so no two runs repeat each other, and prints it before the first
//! case. A test may read the environment, production code may not.
//!
//! The battery, one function per claim:
//!
//! - [`count_matches_enumeration`] — the compiled diagram's model count is the
//!   count over all `2^n` assignments.
//! - [`orders_agree`] — three operation orders over one clause set minimize to
//!   structurally identical diagrams.
//! - [`operations_match_enumeration`] — conjunction, disjunction, negation,
//!   conditioning, projection and restriction each answer the transformed
//!   function's truth table.
//! - [`marginalizing_preserves_the_count`] — summing a random downward-closed
//!   set of levels out leaves the count alone.
//! - [`text_round_trip`] — a diagram written to `.tdd` and read back is the
//!   same diagram.
//! - [`weighted_counts_match_enumeration`] — exact rational weights reproduce
//!   the weighted sum, and the log domain reproduces it to `1e-9` of the sum of
//!   the term magnitudes, which is the scale a signed fold's accuracy is against.
//! - [`weighted_composition_matches_enumeration`] — restriction retains the input
//!   weights and grafting preserves weighted values through renaming and marginalization.
//! - [`a_tight_budget_refuses_rather_than_panics`] — under a byte budget too
//!   small for the work, every entry point returns or refuses, and the engine
//!   still answers correctly once the budget is lifted.
//!
//! Every claim runs through [`check_case`], so a failure found by the loop is
//! reproduced by writing its printed case down as a `Case::literal` and calling
//! [`check_case`] on it from an ordinary `#[test]` — the regression tests at the
//! bottom of this file. A failure prints the seed, the claim that broke, the
//! formula in DIMACS and the vtree before it panics.
//!
//! The regression tests at the bottom are cases the sweep found, each written
//! down as a `Case::literal`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use num_rational::BigRational;
use num_traits::Zero;

use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, SignedLog, WeightStore};
use tididi::limits::LimitConfig;
use tididi::io::{load_tdd, save_tdd};



use tididi::test_helpers::{
    assert_canonical, assert_marginal_canonical, assert_restrict_ok, assert_same_shape,
    brute_force_count, eval, rand_cnf, CnfShape, Lcg,
};
use tididi::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};
use tididi::{Engine, Literal, Tdd};

// ── The case ────────────────────────────────────────────────────────────────

/// One drawn problem: a formula, the vtree it is compiled against, and the
/// seed that produced both.
struct Case {
    seed: u64,
    num_vars: u32,
    clauses: Vec<Vec<i32>>,
    vtree: Arc<Vtree>,
    /// How the vtree was drawn, for the failure report.
    vtree_kind: String,
}

impl Case {
    /// A case written down by hand, for a regression test. The seed is carried
    /// because the claims that draw of their own — the weights, the levels to
    /// sum out — draw from it, so a recorded case replays only with the seed it
    /// was found under.
    fn literal(seed: u64, num_vars: u32, clauses: Vec<Vec<i32>>, vtree_text: &str) -> Case {
        let vtree = Arc::new(Vtree::from_text(vtree_text).expect("the recorded vtree parses"));
        Case { seed, num_vars, clauses, vtree, vtree_kind: "recorded".to_string() }
    }

    /// The seed, the formula in DIMACS and the vtree in its text format: what a
    /// reader needs to write the case down as a regression test.
    fn report(&self) -> String {
        let mut s = format!(
            "seed {}, {} vars, {} clauses, vtree {}\n",
            self.seed,
            self.num_vars,
            self.clauses.len(),
            self.vtree_kind
        );
        s.push_str(&format!("p cnf {} {}\n", self.num_vars, self.clauses.len()));
        for clause in &self.clauses {
            for lit in clause {
                s.push_str(&format!("{lit} "));
            }
            s.push_str("0\n");
        }
        s.push_str("--- vtree ---\n");
        s.push_str(&self.vtree.to_text());
        s.push_str("--- as a regression test ---\n");
        let written: Vec<String> = self
            .clauses
            .iter()
            .map(|c| {
                let lits: Vec<String> = c.iter().map(|l| l.to_string()).collect();
                format!("vec![{}]", lits.join(", "))
            })
            .collect();
        s.push_str(&format!(
            "check_case(&Case::literal({}, {}, vec![{}], {:?}));\n",
            self.seed,
            self.num_vars,
            written.join(", "),
            self.vtree.to_text()
        ));
        s
    }
}

/// Draw a formula and a vtree from one seed.
///
/// Small on purpose: enumeration over `2^n` is the oracle, so `n` stays within
/// ten, and the interesting cases are the degenerate clause shapes rather than
/// the large ones. Duplicate, unit and tautological clauses are added on some
/// draws because each is a shape the clause builder and the reduction rules
/// treat specially: a repeated literal and a variable in both polarities both
/// make the clause's literal list something other than a set.
fn draw(seed: u64) -> Case {
    let mut rng = Lcg::new(seed);
    let num_vars = 1 + rng.below(10) as u32;
    let shape = CnfShape { clauses: 25, width: 4 };
    let mut clauses = rand_cnf(&mut rng, num_vars, shape);
    if rng.below(3) == 0 && !clauses.is_empty() {
        let i = rng.below(clauses.len() as u64) as usize;
        clauses.push(clauses[i].clone());
    }
    if rng.below(3) == 0 {
        let v = 1 + rng.below(u64::from(num_vars)) as i32;
        clauses.push(vec![if rng.coin() { v } else { -v }]);
    }
    if rng.below(4) == 0 {
        let v = 1 + rng.below(u64::from(num_vars)) as i32;
        clauses.push(vec![v, -v]);
    }
    if rng.below(5) == 0 && !clauses.is_empty() {
        let i = rng.below(clauses.len() as u64) as usize;
        let lit = clauses[i][0];
        clauses[i].push(lit);
    }
    if clauses.is_empty() {
        clauses.push(vec![1]);
    }
    clauses.truncate(25);
    let (vtree, kind) = draw_vtree(&mut rng, num_vars);
    Case { seed, num_vars, clauses, vtree, vtree_kind: kind }
}

/// A vtree over `1..=num_vars`, drawn from the shapes whose differences the
/// diagram can see: the two regular shapes, the two regular shapes over a
/// shuffled variable order, and an unbalanced random tree.
fn draw_vtree(rng: &mut Lcg, num_vars: u32) -> (Arc<Vtree>, String) {
    let mut order: Vec<VarId> = (1..=num_vars).map(VarId).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below((i + 1) as u64) as usize);
    }
    let seed = rng.next_u64();
    match rng.below(5) {
        0 => (Arc::new(Vtree::balanced(num_vars)), "balanced".to_string()),
        1 => (Arc::new(Vtree::linear(num_vars)), "linear".to_string()),
        2 => (Arc::new(Vtree::balanced_over(&order).unwrap()), format!("balanced_over({order:?})")),
        3 => (Arc::new(Vtree::linear_from_order(&order).unwrap()), format!("linear_from_order({order:?})")),
        _ => (Arc::new(Vtree::random(num_vars, seed)), format!("random({seed})")),
    }
}

// ── Oracles this file adds to the shared ones ───────────────────────────────

/// The truth table of a formula, indexed by the assignment read as a bit mask
/// with variable `i` in bit `i`.
fn clause_truth(num_vars: u32, clauses: &[Vec<i32>]) -> Vec<bool> {
    (0..(1u32 << num_vars))
        .map(|mask| {
            clauses.iter().all(|clause| {
                clause.iter().any(|&lit| {
                    let val = (mask >> (lit.unsigned_abs() - 1)) & 1 == 1;
                    (lit > 0) == val
                })
            })
        })
        .collect()
}

/// The truth table of a diagram, read off the stored encoding by the
/// apply-free evaluator, so it shares no machinery with the operation that
/// produced the diagram.
fn diagram_truth(f: &Tdd, num_vars: u32) -> Vec<bool> {
    (0..(1u32 << num_vars))
        .map(|mask| {
            let asn: Vec<bool> = (0..num_vars).map(|i| (mask >> i) & 1 == 1).collect();
            eval(f, &asn)
        })
        .collect()
}

/// The assignment behind a truth-table index, for a failure message.
fn assignment(mask: u32, num_vars: u32) -> Vec<bool> {
    (0..num_vars).map(|i| (mask >> i) & 1 == 1).collect()
}

/// Two truth tables agree, or a panic naming the first assignment where they
/// do not.
fn assert_truth(got: &[bool], want: &[bool], num_vars: u32, what: &str) {
    for (mask, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{what}: wrong at assignment {:?}",
            assignment(mask as u32, num_vars)
        );
    }
}

/// The invariants a finished diagram carries are post-pass properties: an
/// apply establishes determinism and emits no false node, and a reduction pass
/// is what establishes the rest. So a result is checked for them after a pass,
/// while its truth table is read off the result as it came out.
fn assert_canonical_after_minimize(t: &Tdd) {
    if !t.has_marginal_level() {
        let satisfiable = t.model_count().unwrap() != BigUint::from(0u32);
        assert_eq!(!t.is_zero(), satisfiable);
        assert_eq!(Engine::new().is_sat(t).unwrap(), satisfiable);
    }
    let mut m = t.clone();
    m.minimize().unwrap();
    assert_finished_canonical(&m);
}

/// Use value-slot invariants after marginalization, and Boolean signatures otherwise.
fn assert_finished_canonical(t: &Tdd) {
    if t.has_marginal_level() {
        assert_marginal_canonical(t);
    } else {
        assert_canonical(t);
    }
}

/// DIMACS literals as the library's own.
fn lits(clause: &[i32]) -> Vec<Literal> {
    clause.iter().map(|&l| Literal::try_from(l).unwrap()).collect()
}

/// The clause as a canonical diagram.
fn clause_tdd(vtree: &Arc<Vtree>, clause: &[i32]) -> Tdd {
    Tdd::clause(vtree, lits(clause)).unwrap()
}

// ── The battery ─────────────────────────────────────────────────────────────

thread_local! {
    /// The claim currently running, so a failure report names it as well as
    /// the case. A panic unwinds out of the claim, so the cell still holds it.
    static STEP: std::cell::Cell<&'static str> = const { std::cell::Cell::new("none") };
}

/// Name what runs next, so a failure report says which claim broke.
fn step(name: &'static str) {
    STEP.with(|s| s.set(name));
}

/// Every claim the suite makes about one case. The loop calls this, and so does
/// each regression test.
fn check_case(case: &Case) {
    /// One claim of the battery, by the name a failure report gives it.
    type Claim = (&'static str, fn(&Case));
    let claims: [Claim; 9] = [
        ("count against enumeration", count_matches_enumeration),
        ("operation orders agree", orders_agree),
        ("operations against enumeration", operations_match_enumeration),
        ("marginalizing preserves the count", marginalizing_preserves_the_count),
        ("text round trip", text_round_trip),
        ("weighted counts against enumeration", weighted_counts_match_enumeration),
        ("streaming marginalization against enumeration", streaming_marginalization_matches_enumeration),
        ("weighted composition against enumeration", weighted_composition_matches_enumeration),
        ("a tight budget refuses", a_tight_budget_refuses_rather_than_panics),
    ];
    for (name, claim) in claims {
        step(name);
        claim(case);
    }
    step("none");
}

/// Conjoin the clauses one at a time, minimizing after each: the canonical
/// result counts what enumeration counts.
fn compile(case: &Case) -> Tdd {
    let eng = Engine::new();
    let mut acc = Tdd::one(&case.vtree);
    for clause in &case.clauses {
        acc = eng.and(acc, clause_tdd(&case.vtree, clause)).expect("an unarmed engine refuses nothing");
        acc.minimize().unwrap();
    }
    acc
}

fn count_matches_enumeration(case: &Case) {
    let f = compile(case);
    assert_canonical(&f);
    assert_eq!(
        f.model_count().unwrap(),
        BigUint::from(brute_force_count(case.num_vars, &case.clauses)),
        "model count disagrees with enumeration"
    );
    assert_truth(
        &diagram_truth(&f, case.num_vars),
        &clause_truth(case.num_vars, &case.clauses),
        case.num_vars,
        "compiled diagram",
    );
}

/// One clause set reached three ways: a left fold of conjunctions, a pairwise
/// tree, and a fold of the clause-at-a-time entry point. The canonical form is
/// a property of the function and the vtree, so all three are the same diagram.
fn orders_agree(case: &Case) {
    let eng = Engine::new();

    let mut left = compile(case);
    left.minimize().unwrap();
    assert_canonical(&left);

    let mut queue: Vec<Tdd> =
        case.clauses.iter().map(|c| clause_tdd(&case.vtree, c)).collect();
    while queue.len() > 1 {
        let mut next = Vec::with_capacity(queue.len().div_ceil(2));
        let mut it = queue.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(eng.and(a, b).expect("an unarmed engine refuses nothing")),
                None => next.push(a),
            }
        }
        queue = next;
    }
    let mut tree = queue.pop().unwrap_or_else(|| Tdd::one(&case.vtree));
    tree.minimize().unwrap();
    assert_canonical(&tree);
    assert_same_shape(&left, &tree, "clause fold against pairwise tree");

    let mut by_clause = Tdd::one(&case.vtree);
    for clause in &case.clauses {
        by_clause = by_clause.and_clause(lits(clause)).unwrap();
    }
    by_clause.minimize().unwrap();
    assert_canonical(&by_clause);
    assert_same_shape(&left, &by_clause, "clause fold against clause-at-a-time");
}

/// Conjunction, disjunction, negation, conditioning, projection and
/// restriction, each against the truth table of the function it claims to
/// compute.
fn operations_match_enumeration(case: &Case) {
    let n = case.num_vars;
    let eng = Engine::new();
    let split = case.clauses.len().div_ceil(2);
    let head: Vec<Vec<i32>> = case.clauses[..split].to_vec();
    let tail: Vec<Vec<i32>> = case.clauses[split..].to_vec();

    step("compiling the halves");
    let f = compile(&Case { clauses: head.clone(), vtree_kind: String::new(), ..borrow(case) });
    let g = compile(&Case { clauses: tail.clone(), vtree_kind: String::new(), ..borrow(case) });
    let tf = diagram_truth(&f, n);
    let tg = diagram_truth(&g, n);

    step("conjunction");
    let conj = eng.and(f.clone(), g.clone()).expect("an unarmed engine refuses nothing");
    assert_canonical_after_minimize(&conj);
    let want: Vec<bool> = tf.iter().zip(&tg).map(|(a, b)| *a && *b).collect();
    assert_truth(&diagram_truth(&conj, n), &want, n, "conjunction");

    step("disjunction");
    let disj = eng.or(f.clone(), g.clone()).expect("an unarmed engine refuses nothing");
    assert_canonical_after_minimize(&disj);
    let want: Vec<bool> = tf.iter().zip(&tg).map(|(a, b)| *a || *b).collect();
    assert_truth(&diagram_truth(&disj, n), &want, n, "disjunction");

    step("negation");
    let neg = (f.clone()).negate().unwrap();
    assert_canonical_after_minimize(&neg);
    let want: Vec<bool> = tf.iter().map(|a| !*a).collect();
    assert_truth(&diagram_truth(&neg, n), &want, n, "negation");

    // Conditioning and projection keep the vtree, so the touched variable is
    // free in the result and its two branches carry the same value.
    for x in 0..n {
        for value in [false, true] {
            step("conditioning");
            let c = (f).clone().condition_var(VarId(x + 1), value).unwrap();
            assert_canonical_after_minimize(&c);
            let want: Vec<bool> = (0..(1u32 << n))
                .map(|mask| {
                    let pinned = if value { mask | (1 << x) } else { mask & !(1 << x) };
                    tf[pinned as usize]
                })
                .collect();
            assert_truth(&diagram_truth(&c, n), &want, n, "condition");
        }
        step("projection");
        let p = (f).clone().exists_var(VarId(x + 1)).unwrap();
        assert_canonical_after_minimize(&p);
        let want: Vec<bool> = (0..(1u32 << n))
            .map(|mask| tf[(mask | (1 << x)) as usize] || tf[(mask & !(1 << x)) as usize])
            .collect();
        assert_truth(&diagram_truth(&p, n), &want, n, "projection");
    }

    step("restriction");
    assert_restrict_ok(&f, &g, n);
    assert_restrict_ok(&g, &f, n);
}

/// Borrow a case's vtree and variable count into a fresh case over other
/// clauses, so the halves of `operations_match_enumeration` compile the same
/// way the whole does.
fn borrow(case: &Case) -> Case {
    Case {
        seed: case.seed,
        num_vars: case.num_vars,
        clauses: Vec::new(),
        vtree: Arc::clone(&case.vtree),
        vtree_kind: case.vtree_kind.clone(),
    }
}

/// Sum a random set of levels out. A level may go only once its children are
/// marginal or are leaves, so the drawn set is closed downward and ordered
/// bottom-up; the count the diagram answers is the same before and after.
fn marginalizing_preserves_the_count(case: &Case) {
    let mut f = compile(case);
    let before = f.model_count().unwrap();
    let targets = draw_marginal_targets(case);
    if targets.is_empty() {
        return;
    }
    let eng = Engine::new();
    eng.marginalize_levels(&mut f, &targets).expect("an unarmed engine refuses nothing");
    assert_canonical_after_minimize(&f);
    assert_eq!(f.model_count().unwrap(), before, "marginalizing changed the count");
    // An unsatisfiable formula compiles to the sentinel, which has no levels
    // to sum out, so only a diagram with storage is expected to gain one.
    assert!(
        f.is_zero() || f.has_marginal_level(),
        "marginalizing left no marginal level"
    );
}

/// A downward-closed set of internal levels, in bottom-up order, drawn from
/// the case's own seed so a failure replays.
fn draw_marginal_targets(case: &Case) -> Vec<VtreeIdx> {
    let mut rng = Lcg::new(case.seed ^ 0x5eed_4a49);
    let vtree = &case.vtree;
    let mut chosen = vec![false; vtree.num_nodes()];
    for (t, _left, _right) in vtree.internal_bottomup() {
        // A node goes in when drawn, and must go in when a chosen node is
        // below it would otherwise leave the set open downward.
        if rng.below(3) == 0 {
            mark_subtree(vtree, t, &mut chosen);
        }
    }
    vtree
        .internal_bottomup()
        .filter(|(t, _, _)| chosen[t.idx()])
        .map(|(t, _, _)| t)
        .collect()
}

fn mark_subtree(vtree: &Vtree, t: VtreeIdx, chosen: &mut [bool]) {
    if let VtreeNode::Internal { left, right, .. } = *vtree.node(t) {
        chosen[t.idx()] = true;
        mark_subtree(vtree, left, chosen);
        mark_subtree(vtree, right, chosen);
    }
}

/// The `.tdd` text format carries the diagram: what comes back is the same
/// diagram and answers the same count.
fn text_round_trip(case: &Case) {
    let f = compile(case);
    let dir = std::env::temp_dir().join(format!("tididi-fuzz-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a writable temporary directory");
    let path = dir.join(format!("case-{:?}.tdd", std::thread::current().id()));
    save_tdd(&f, &path).expect("the diagram is structural, so it is writable");
    let back = load_tdd(&path, &case.vtree).expect("what was just written reads back");
    assert_canonical(&back);
    assert_same_shape(&f, &back, "text round trip");
    assert_eq!(f.model_count().unwrap(), back.model_count().unwrap(), "text round trip changed the count");
    let _ = std::fs::remove_file(&path);
}

/// Weighted counting against the weighted sum over the truth table, exactly in
/// the rational domain and, where the loop still asks it, in the log domain.
fn weighted_counts_match_enumeration(case: &Case) {
    let w = weighted_case(case);

    let semiring = RationalWeights::from_literals(&w.weights);
    assert_eq!(
        w.f.evaluate(&semiring).unwrap(),
        w.want,
        "exact weighted evaluation disagrees with enumeration"
    );

    let eng = Engine::new();
    let mut exact = w.f.clone();
    exact.set_weights(WeightStore::new(
        RationalWeights::from_literals(&w.weights),
        Arithmetic::ExactRational,
    )).unwrap();
    eng.marginalize_levels(&mut exact, &w.targets).expect("an unarmed engine refuses nothing");
    let got = exact.weighted_value().unwrap().expect("a store is attached");
    assert_eq!(
        got.as_rational().into_owned(),
        w.want,
        "the exact weighted marginal fold disagrees with enumeration"
    );

    log_weighted_count_matches_enumeration(case);
}

/// The log domain reproduces the weighted sum to `1e-9` of the sum of the term
/// magnitudes — the scale a signed fold's accuracy is against, since signed
/// weights cancel and a sum near zero is the difference of large terms.
fn log_weighted_count_matches_enumeration(case: &Case) {
    let w = weighted_case(case);
    let eng = Engine::new();
    let mut logged = w.f.clone();
    logged.set_weights(WeightStore::new(
        RationalWeights::from_literals(&w.weights),
        Arithmetic::SignedLog,
    )).unwrap();
    eng.marginalize_levels(&mut logged, &w.targets).expect("an unarmed engine refuses nothing");
    let got = logged.weighted_value().unwrap().expect("a store is attached");
    let got = *got.as_log().expect("a log store answers in the log domain");
    let want_f = ratio_to_f64(&w.want);
    let got_f = f64::from(got.sign) * got.ln_abs.exp();
    let scale = ratio_to_f64(&w.magnitude).max(f64::MIN_POSITIVE);
    assert!(
        (got_f - want_f).abs() <= 1e-9 * scale,
        "the log-domain weighted fold is off: {got_f} against {want_f}"
    );
}

/// The weighted problem drawn from a case: the literal weights, the compiled
/// diagram, the levels to sum out, and what enumeration says the answer is.
struct Weighted {
    weights: Vec<LiteralWeights<BigRational>>,
    f: Tdd,
    targets: Vec<VtreeIdx>,
    want: BigRational,
    /// The sum of the term magnitudes, the scale the log domain is held to.
    magnitude: BigRational,
}

fn weighted_case(case: &Case) -> Weighted {
    let n = case.num_vars;
    let mut rng = Lcg::new(case.seed ^ 0x_0057_4d43);
    // Eighths, so a product over ten variables stays a short rational, and a
    // negative draw now and then so the log domain's sign tracking is exercised.
    let weights: Vec<LiteralWeights<BigRational>> = (0..n)
        .map(|_| {
            let mut draw = || {
                let num = 1 + rng.below(7) as i64;
                let num = if rng.below(4) == 0 { -num } else { num };
                BigRational::new(num.into(), 8.into())
            };
            LiteralWeights { negative: draw(), positive: draw() }
        })
        .collect();

    let truth = clause_truth(n, &case.clauses);
    let (want, magnitude) = weighted_sum(&truth, &weights);

    Weighted {
        weights,
        f: compile(case),
        targets: draw_marginal_targets(case),
        want,
        magnitude,
    }
}

/// Enumerate the weighted sum and term magnitudes of a truth table.
fn weighted_sum(truth: &[bool], weights: &[LiteralWeights<BigRational>]) -> (BigRational, BigRational) {
    let mut want = BigRational::zero();
    let mut magnitude = BigRational::zero();
    for (mask, &sat) in truth.iter().enumerate() {
        if !sat {
            continue;
        }
        let mut term = BigRational::new(1.into(), 1.into());
        for (i, w) in weights.iter().enumerate() {
            let positive = (mask >> i) & 1 == 1;
            term *= if positive { w.positive.clone() } else { w.negative.clone() };
        }
        magnitude += if term < BigRational::zero() { -term.clone() } else { term.clone() };
        want += term;
    }
    (want, magnitude)
}

/// Compare both stored arithmetics against an enumerated sum on its term-magnitude scale.
fn assert_weighted_sum(tdd: &Tdd, want: &BigRational, magnitude: &BigRational) {
    let got = tdd.weighted_value().unwrap().expect("composition retains its weight configuration");
    if let Some(log) = got.as_log() {
        let got_f = f64::from(log.sign) * log.ln_abs.exp();
        let scale = ratio_to_f64(magnitude).max(f64::MIN_POSITIVE);
        assert!((got_f - ratio_to_f64(want)).abs() <= 1e-9 * scale, "weighted composition changed the log value");
    } else {
        assert_eq!(got.as_rational().as_ref(), want, "weighted composition changed the exact value");
    }
}

/// Restrict under the operand's weights, then graft a renamed marginal part and a free variable.
fn weighted_composition_matches_enumeration(case: &Case) {
    let w = weighted_case(case);
    let eng = Engine::new();
    let care = Tdd::clause(&case.vtree, [if case.seed & 1 == 0 { 1 } else { -1 }]).unwrap();
    assert_canonical(&care);
    let original_truth = diagram_truth(&w.f, case.num_vars);
    let care_truth = diagram_truth(&care, case.num_vars);
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        step("weighted restriction against enumeration");
        let mut source = w.f.clone();
        source.set_weights(WeightStore::new(RationalWeights::from_literals(&w.weights), arithmetic)).unwrap();
        assert_canonical(&source);
        let mut restricted = eng.restrict_to_care(source.clone(), care.clone()).unwrap().into_tdd();
        restricted.minimize().unwrap();
        assert_canonical(&restricted);
        let truth = diagram_truth(&restricted, case.num_vars);
        for ((&got, &original), &cared) in truth.iter().zip(&original_truth).zip(&care_truth) {
            assert_eq!(got && cared, original && cared, "weighted restriction changed the care region");
        }
        let (want, magnitude) = weighted_sum(&truth, &w.weights);
        assert_weighted_sum(&restricted, &want, &magnitude);

        step("weighted graft against enumeration");
        let targets = if case.seed & 1 == 0 { vec![case.vtree.root()] } else { w.targets.clone() };
        eng.marginalize_levels(&mut source, &targets).unwrap();
        source.minimize().unwrap();
        assert_finished_canonical(&source);
        let map: Vec<VarId> = (1..=case.num_vars).rev().map(VarId).collect();
        let mut global: Vec<_> = w.weights.iter().rev().cloned().collect();
        global.push(LiteralWeights { negative: BigRational::from_integer(2.into()), positive: BigRational::from_integer(3.into()) });
        let (grafted, _) = Tdd::graft_over(&eng, vec![(source, map)], &[VarId(case.num_vars + 1)], case.num_vars + 1,
            Some(WeightStore::new(RationalWeights::from_literals(&global), arithmetic))).unwrap();
        assert_finished_canonical(&grafted);
        assert_weighted_sum(&grafted, &(&w.want * BigRational::from_integer(5.into())), &(&w.magnitude * BigRational::from_integer(5.into())));
    }
}

/// A rational as the nearest `f64`, for the log-domain comparison.
fn ratio_to_f64(r: &BigRational) -> f64 {
    use num_traits::ToPrimitive;
    r.to_f64().expect("an eighths-weighted sum over ten variables fits an f64")
}

/// Under a byte budget too small for the work, every entry point either
/// answers correctly or refuses, never panics, and the engine still answers
/// correctly once the budget is lifted.
///
/// The budget is drawn across several magnitudes: the smallest refuses the
/// first allocation, the largest lets most of the fold through, and the
/// interesting refusals are in between.
fn a_tight_budget_refuses_rather_than_panics(case: &Case) {
    let n = case.num_vars;
    let mut rng = Lcg::new(case.seed ^ 0x_b0d6_e700);
    let eng = Engine::new();

    for budget in [1u64, 1 << 6, 1 << 10, 1 << 14, rng.below(1 << 16)] {
        let _armed = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(budget)));
        let mut acc = Tdd::one(&case.vtree);
        for (i, clause) in case.clauses.iter().enumerate() {
            let Ok(cl) = eng.clause(&case.vtree, lits(clause)) else { break };
            let Ok(next) = eng.and(acc, cl) else { break };
            acc = next;
            let opts = tididi::reduce::ReductionPlan::default();
            if eng.reduce(&mut acc, opts).is_err() {
                break;
            }
            let folded = i + 1;
            // Whatever the budget let through has to be the conjunction of the
            // clauses it got through, or a refusal was answered with a wrong
            // diagram rather than an error.
            let want = clause_truth(n, &case.clauses[..folded]);
            assert_truth(&diagram_truth(&acc, n), &want, n, "under a byte budget");
            if let Ok(count) = eng.model_count(&acc) {
                assert_eq!(
                    count,
                    BigUint::from(brute_force_count(n, &case.clauses[..folded])),
                    "a count answered under a byte budget is wrong"
                );
            }
        }
    }

    eng.clear_scratch();
    let want = BigUint::from(brute_force_count(n, &case.clauses));
    let mut acc = Tdd::one(&case.vtree);
    for clause in &case.clauses {
        acc = eng.and(acc, clause_tdd(&case.vtree, clause)).expect("the budget is lifted");
        acc.minimize().unwrap();
    }
    assert_canonical(&acc);
    assert_eq!(
        eng.model_count(&acc).expect("the budget is lifted"),
        want,
        "the engine answered wrongly after a refusal"
    );
}

// ── The loop ────────────────────────────────────────────────────────────────

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The seed the run starts from: `TIDIDI_FUZZ_SEED` when it is set, otherwise
/// the clock. A constant default would give every unattended run the same
/// stream, so two long sweeps would decide the same cases twice and nothing
/// else ever; the clock makes each run a fresh draw. The seed is printed
/// before the first case, so a run that fails replays under the variable.
fn first_seed() -> u64 {
    if let Ok(v) = std::env::var("TIDIDI_FUZZ_SEED")
        && let Ok(n) = v.parse()
    {
        return n;
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0x_d1ff_0001, |d| d.as_nanos() as u64)
}

#[test]
#[ignore = "a timed randomized sweep; run it with --ignored"]
fn differential() {
    let budget = Duration::from_secs(env_u64("TIDIDI_FUZZ_SECONDS", 60));
    let first = first_seed();
    println!("differential: seed {first}");
    let start = Instant::now();
    let mut iterations = 0u64;
    let mut seed = first;
    while start.elapsed() < budget {
        let case = draw(seed);
        let report = case.report();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check_case(&case)));
        if outcome.is_err() {
            let step = STEP.with(|s| s.get());
            println!("\n=== differential failure in: {step} ===\n{report}");
            panic!("the differential suite failed on seed {seed}");
        }
        iterations += 1;
        seed = seed.wrapping_add(1);
    }
    println!(
        "differential: {iterations} cases in {:.1}s from seed {first}",
        start.elapsed().as_secs_f64()
    );
    assert!(iterations > 0, "the budget was too short to draw a single case");
}

// ── Regression cases ────────────────────────────────────────────────────────
//
// A case the suite failed on, written down so it keeps being checked after the
// bug it found is fixed.

/// A log-domain accumulation answers a number.
///
/// Two opposite-signed values whose log-magnitudes differ by less than the
/// `f64` epsilon cancel to `ln(1 - 1) = -∞` while the sign of the larger is
/// kept, which is the one state the domain says cannot occur: a magnitude of
/// zero is spelled by a sign of zero. Adding two of those subtracts `-∞` from
/// `-∞`, and every later multiply and add carries the resulting `NaN`.
#[test]
fn a_log_domain_sum_is_a_number() {
    let mut cancelled = SignedLog { ln_abs: 0.0, sign: 1 };
    cancelled.add_assign(&SignedLog { ln_abs: 1e-17, sign: -1 });
    assert!(
        cancelled.ln_abs.is_finite() || cancelled.sign == 0,
        "a cancelled value carries a sign with no magnitude"
    );
    let mut doubled = cancelled;
    doubled.add_assign(&cancelled);
    assert!(doubled.ln_abs.is_finite() || doubled.sign == 0, "the sum is not a number");
}

/// The same, reached through weighted model counting rather than by hand: this
/// formula and vtree, with the weights the suite drew, fold to `NaN` in the log
/// domain while the exact domain answers a small negative rational.
#[test]
fn a_log_domain_weighted_count_is_a_number() {
    log_weighted_count_matches_enumeration(&Case::literal(
        3_524_041_525,
        8,
        vec![vec![-7, -1, 5], vec![-2], vec![-3, 1, 8], vec![-1, 4, 6], vec![-1, 4, 6]],
        "vtree 15\nL 0 7\nL 1 5\nL 2 2\nL 3 4\nL 4 3\nL 5 1\nL 6 6\nL 7 8\n\
         I 8 0 1\nI 9 2 3\nI 10 8 4\nI 11 5 9\nI 12 6 10\nI 13 11 12\nI 14 13 7\n",
    ));
}

/// Conditioning answers the cofactor without leaving a node that computes ⊥
/// reachable from the output.
///
/// Invariant 2 admits no such node, and a reduction pass is what would remove
/// one; here the node survives `minimize`, so the conditioned diagram is a
/// structurally invalid representation of a function whose count it still
/// reports correctly. Every conditioning of a clause over three or more
/// variables shows it.
#[test]
fn conditioning_leaves_no_node_computing_false() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = clause_tdd(&vtree, &[1, 2]);
    let mut c = (f).clone().condition_var(VarId(2), false).unwrap();
    c.minimize().unwrap();
    assert_eq!(c.model_count().unwrap(), BigUint::from(4u32), "the cofactor's count is unaffected");
    assert_canonical(&c);
}

/// A clause is the disjunction of the literals it names, whatever variables
/// they repeat: `x1 ∨ ¬x1` is satisfied by every assignment, and `x1 ∨ x1` by
/// the half that sets `x1`.
///
/// The clause builder seeds each variable's leaf from the first literal naming
/// it and skips the rest, so today it answers `x1 ∨ ¬x1` with `¬x1` — a
/// different function, returned without a refusal or a panic, where the cube
/// builder panics on the same input shape. Conjoining such a clause into an
/// accumulator therefore loses models silently.
#[test]
fn a_clause_naming_one_variable_twice_is_the_clause_it_spells() {
    let vtree = Arc::new(Vtree::balanced(3));
    let count = |clause: &[i32]| clause_tdd(&vtree, clause).model_count().unwrap();
    assert_eq!(count(&[1, -1]), BigUint::from(8u32), "x1 ∨ ¬x1 holds everywhere");
    assert_eq!(count(&[-1, 1]), BigUint::from(8u32), "¬x1 ∨ x1 holds everywhere");
    assert_eq!(count(&[1, -1, 2]), BigUint::from(8u32), "a tautology stays one");
    assert_eq!(count(&[1, 1]), BigUint::from(4u32), "x1 ∨ x1 is x1");

    check_case(&Case::literal(
        3_523_149_841,
        4,
        vec![vec![3, -4], vec![4, -3], vec![-4], vec![3, 2, 1], vec![-4, -3], vec![4, -4]],
        "vtree 7\nL 0 2\nL 1 1\nL 2 3\nL 3 4\nI 4 0 1\nI 5 2 4\nI 6 3 5\n",
    ));
}

/// Streaming and standalone installation preserve the independently enumerated value.
fn streaming_marginalization_matches_enumeration(case: &Case) {
    let split = case.clauses.len() / 2;
    let mut left_case = borrow(case);
    let mut right_case = borrow(case);
    left_case.clauses = case.clauses[..split].to_vec();
    right_case.clauses = case.clauses[split..].to_vec();
    let (left, right) = (compile(&left_case), compile(&right_case));
    let eng = Engine::new();
    let mut targets = draw_marginal_targets(case);
    if case.seed & 1 != 0 {
        targets.extend(case.vtree.leaf_bottomup().map(|(t, _)| t));
    }
    let mut integer = eng.and_marginalizing(left.clone(), right.clone(), &targets).unwrap();
    integer.minimize().unwrap();
    assert_finished_canonical(&integer);
    assert_eq!(integer.model_count().unwrap(), BigUint::from(brute_force_count(case.num_vars, &case.clauses)));
    let w = weighted_case(case);
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let store = WeightStore::new(RationalWeights::from_literals(&w.weights), arithmetic);
        let (mut f, mut g) = (left.clone(), right.clone());
        f.set_weights(store.clone()).unwrap();
        g.set_weights(store).unwrap();
        let mut result = eng.and_marginalizing(f, g, &targets).unwrap();
        result.minimize().unwrap();
        assert_finished_canonical(&result);
        let got = eng.weighted_value(&result).unwrap().unwrap();
        match arithmetic {
            Arithmetic::ExactRational => assert_eq!(got.as_rational().into_owned(), w.want),
            Arithmetic::SignedLog => {
                let got = got.as_log().unwrap();
                let value = f64::from(got.sign) * got.ln_abs.exp();
                let scale = ratio_to_f64(&w.magnitude).max(f64::MIN_POSITIVE);
                assert!((value - ratio_to_f64(&w.want)).abs() <= 1e-9 * scale);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn streaming_and_standalone_marginalization_preserve_overflow_values() {
    let tree = Arc::new(Vtree::balanced(260));
    let (f, g) = (Tdd::clause(&tree, [1]).unwrap(), Tdd::clause(&tree, [260]).unwrap());
    assert_canonical(&f);
    assert_canonical(&g);
    let eng = Engine::new();
    let targets: Vec<_> = tree.bottomup().collect();
    let want = BigUint::from(1u32) << 258usize;
    let mut streamed = eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap();
    let mut standalone = eng.and(f.clone(), g.clone()).unwrap();
    eng.marginalize_levels(&mut standalone, &targets).unwrap();
    for result in [&mut streamed, &mut standalone] {
        result.minimize().unwrap();
        assert_finished_canonical(result);
        assert_eq!(result.model_count().unwrap(), want);
    }
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let (mut f, mut g) = (f.clone(), g.clone());
        let store = WeightStore::new(RationalWeights::unit(260), arithmetic);
        f.set_weights(store.clone()).unwrap();
        g.set_weights(store).unwrap();
        let mut result = eng.and_marginalizing(f, g, &targets).unwrap();
        result.minimize().unwrap();
        assert_finished_canonical(&result);
        let value = eng.weighted_value(&result).unwrap().unwrap();
        match arithmetic {
            Arithmetic::ExactRational => assert_eq!(value.as_rational().into_owned(), BigRational::from_integer(want.clone().into())),
            Arithmetic::SignedLog => {
                let value = value.as_log().unwrap();
                assert_eq!(value.sign, 1);
                assert!((value.ln_abs - 258.0 * std::f64::consts::LN_2).abs() < 1e-9);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn streaming_weighted_leaf_fusion_respects_arithmetic_and_pinned_columns() {
    check_case(&Case::literal(
        20260917,
        9,
        vec![vec![-2, -5], vec![-9, -8]],
        "vtree 17\nL 0 2\nL 1 4\nL 2 9\nL 3 3\nL 4 1\nL 5 7\nL 6 8\nL 7 6\nL 8 5\nI 9 0 1\nI 10 2 9\nI 11 3 10\nI 12 4 11\nI 13 5 12\nI 14 6 13\nI 15 7 14\nI 16 8 15\n",
    ));
}

#[test]
fn streaming_marginal_values_need_the_marginal_canonicality_oracle() {
    check_case(&Case::literal(
        20260955,
        10,
        vec![vec![-2, -8], vec![2], vec![5, -6], vec![5, -2, 3], vec![5, 2, -7, 10], vec![-1, 4, 3], vec![2, -3, 7], vec![2, 3, -9], vec![-9, -1, 7], vec![-7, 9, 5, -2], vec![7, -6, -8], vec![2, -3, 7], vec![7], vec![10, -10]],
        "vtree 19\nL 0 3\nL 1 8\nL 2 7\nL 3 10\nL 4 9\nL 5 5\nL 6 4\nL 7 6\nL 8 2\nL 9 1\nI 10 0 1\nI 11 2 3\nI 12 10 4\nI 13 11 5\nI 14 6 12\nI 15 7 13\nI 16 8 9\nI 17 14 15\nI 18 16 17\n",
    ));
}
