//! The probe against exhaustive search, on seeded diagrams implied by the
//! clauses, with and without summed-out levels, cut encodings and Unknown
//! answers.

use num_bigint::BigUint;

use super::*;

/// A diagram, and clauses over its variables whose every model satisfies
/// it once its summed-out variables are dropped.
struct Case {
    f: Tdd,
    phi: Vec<Vec<i32>>,
}

/// Seeded cases over a number of variables drawn from `vars`: clauses
/// drawn, and some of them compiled on two vtree shapes drawn, as they are,
/// conjoined without reduction from two halves, and with one inner subtree
/// and one leaf outside it summed out. The clauses left out constrain the
/// diagram as a compiler's later clauses would.
fn cases(seed: u64, count: usize, vars: std::ops::Range<u32>) -> Vec<Case> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let n = vars.start + rng.below(u64::from(vars.end - vars.start)) as u32;
        let phi = rand_cnf(&mut rng, n, CnfShape { clauses: 2 * n as usize, width: 3 });
        let set = &phi[..1 + rng.below(phi.len() as u64) as usize];
        let shapes = vtree_shapes(n);
        for _ in 0..2 {
            let vtree = &shapes[rng.below(shapes.len() as u64) as usize].1;
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, vtree, set);
            let half = set.len() / 2;
            let split = eng.and(compile_clauses_on(&eng, vtree, &set[..half]), compile_clauses_on(&eng, vtree, &set[half..])).unwrap();
            out.push(Case { f: split, phi: phi.clone() });
            let inner: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
            if !f.is_zero() && !inner.is_empty() {
                let summed = inner[rng.below(inner.len() as u64) as usize];
                let leaves: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx)
                    .filter(|&t| vtree.node(t).is_leaf() && !under(vtree, t, summed)).collect();
                let leaf = leaves[rng.below(leaves.len() as u64) as usize];
                let mut g = f.clone();
                eng.marginalize_levels(&mut g, &[summed, leaf]).unwrap();
                eng.minimize(&mut g).unwrap();
                out.push(Case { f: g, phi: phi.clone() });
            }
            out.push(Case { f, phi: phi.clone() });
        }
    }
    out
}

/// The assignments of `n` variables, indexed by `VarId::idx`, that satisfy
/// `phi`.
fn models(n: u32, phi: &[Vec<i32>]) -> Vec<Vec<bool>> {
    (0u32..1 << n)
        .map(|bits| (0..n).map(|v| bits >> v & 1 == 1).collect::<Vec<bool>>())
        .filter(|x| phi.iter().all(|clause| clause.iter().any(|&l| x[l.unsigned_abs() as usize - 1] == (l > 0))))
        .collect()
}

/// Whether some model of `phi` makes node `id` of `f` true.
fn alive(f: &Tdd, id: TddNodeId, models: &[Vec<bool>]) -> bool {
    models.iter().any(|x| node_value(f, id, &|_| false, x) > BigUint::ZERO)
}

/// Check an outcome against the models: every dead node has none, and with
/// `exact` every node with a literal is dead exactly when it has none and
/// every other node exactly when its count is zero. The counts agree with
/// the calls.
fn check(case: &Case, encoding: &CnfEncoding, outcome: &ProbeOutcome, calls: &[Call], exact: bool, what: &str) {
    let f = &case.f;
    let models = models(f.vtree().num_vars(), &case.phi);
    let counts = f.node_counts_u128().unwrap();
    let mut dead = 0;
    for id in internal_nodes(f) {
        let is_dead = outcome.is_dead(id);
        dead += u64::from(is_dead);
        let alive = alive(f, id, &models);
        assert!(!(is_dead && alive), "{what}: {id:?} is dead but has a model");
        if exact {
            let expected = match encoding.literal(id) {
                Some(_) => !alive,
                None => counts[id.vtree.idx()][id.local.idx()] == 0,
            };
            assert_eq!(is_dead, expected, "{what}: {id:?}");
        }
    }
    let stats = outcome.stats();
    assert_eq!(stats.dead, dead, "{what}");
    assert_eq!(outcome.dead().iter().flatten().filter(|&&d| d).count() as u64, dead, "{what}");
    assert!(stats.zero_count_dead + stats.enumeration_dead + stats.probes_unsat <= dead, "{what}: {stats:?}");
    assert_eq!(stats.rounds, stats.rounds_sat + stats.rounds_unsat + stats.rounds_unknown, "{what}");
    assert!(stats.rounds_unsat + stats.rounds_unknown <= 1, "{what}");
    let rounds = calls.iter().filter(|call| matches!(call, Call::Solve(_, SolveCall::Round { .. }, _))).count() as u64;
    let probes = calls.iter().filter(|call| matches!(call, Call::Solve(_, SolveCall::Probe { .. }, _))).count() as u64;
    assert_eq!((rounds, probes), (stats.rounds, stats.probes), "{what}");
    let answers = |status: SolveStatus| calls.iter().filter(|call| matches!(call, Call::Solve(_, SolveCall::Probe { .. }, s) if *s == status)).count() as u64;
    assert_eq!((answers(SolveStatus::Unsat), answers(SolveStatus::Unknown)), (stats.probes_unsat, stats.probes_unknown), "{what}");
    assert_eq!(calls.iter().filter(|call| matches!(call, Call::Value(..))).count() as u64, stats.value_reads, "{what}");
    assert_eq!(calls.first(), Some(&Call::Poll(ProbePoint::Begin)), "{what}");
}

/// The rounds and witness settings every case is probed with.
const SETTINGS: [(u32, Witness); 6] = [
    (0, Witness::None), (0, Witness::NodeLiterals), (1, Witness::NodeLiterals),
    (4, Witness::None), (4, Witness::NodeLiterals), (64, Witness::NodeLiterals),
];

/// How many probes, over all cases, killed or skipped nodes each way. A
/// count of zero is left to the hand-built diagrams: compiled ones have no
/// node without a model.
#[derive(Debug, Default)]
struct Reach {
    enumeration: u64,
    unsat: u64,
    sides: u64,
    witness_skips: u64,
}

impl Reach {
    fn add(&mut self, stats: ProbeStats) {
        self.enumeration += u64::from(stats.enumeration_dead > 0);
        self.unsat += u64::from(stats.probes_unsat > 0);
        self.sides += u64::from(stats.dead > stats.zero_count_dead + stats.enumeration_dead + stats.probes_unsat);
        self.witness_skips += u64::from(stats.witness_skips > 0);
    }

    /// Fail unless every way was taken by some probe.
    fn assert_all(&self) {
        let Reach { enumeration, unsat, sides, witness_skips } = *self;
        assert!([enumeration, unsat, sides, witness_skips].iter().all(|&n| n > 0), "{self:?}");
    }
}

#[test]
fn dead_nodes_are_exactly_the_nodes_without_a_model() {
    let mut reach = Reach::default();
    for (k, case) in cases(0x9e0b, 120, 4..8).iter().enumerate() {
        for (rounds, witness) in SETTINGS {
            let (outcome, encoding, host) = probe(&case.f, &case.phi, rounds, witness);
            check(case, &encoding, &outcome, &host.calls, true, &format!("case {k}, {rounds} rounds, {witness:?}"));
            reach.add(outcome.stats());
        }
    }
    reach.assert_all();
}

#[test]
fn a_cut_encoding_is_probed_on_its_prefix() {
    let mut reach = Reach::default();
    for (k, case) in cases(0x9e0c, 60, 4..8).iter().enumerate() {
        let levels = case.f.vtree().internal_bottomup().count();
        for cut in 0..levels {
            for (rounds, witness) in SETTINGS {
                let (encoding, mut host) = Host::over(&case.f, &case.phi, Some(cut));
                let outcome = host.probe(&case.f, &encoding, rounds, witness);
                check(case, &encoding, &outcome, &host.calls, true, &format!("case {k}, cut {cut}, {rounds} rounds, {witness:?}"));
                reach.add(outcome.stats());
            }
        }
    }
    reach.assert_all();
}

#[test]
fn unknown_answers_leave_nodes_alive() {
    let mut rng = Lcg::new(0x0dd);
    for (k, case) in cases(0x9e0d, 60, 4..8).iter().enumerate() {
        for (rounds, witness) in SETTINGS {
            // Every solve Unknown: nothing dies but by its count or its
            // sides, and no model is read.
            let (encoding, mut host) = Host::over(&case.f, &case.phi, None);
            for solve in 0..4096 { host.store.answer_unknown(solve); }
            let outcome = host.probe(&case.f, &encoding, rounds, witness);
            let what = format!("case {k}, {rounds} rounds, {witness:?}, all unknown");
            check(case, &encoding, &outcome, &host.calls, false, &what);
            let stats = outcome.stats();
            assert_eq!((stats.probes_unsat, stats.enumeration_dead, stats.value_reads, stats.rounds_sat), (0, 0, 0, 0), "{what}");

            // Some solves Unknown: a node whose probe was Unknown stays
            // alive.
            let (encoding, mut host) = Host::over(&case.f, &case.phi, None);
            for solve in 0..64 { if rng.below(3) == 0 { host.store.answer_unknown(solve); } }
            let outcome = host.probe(&case.f, &encoding, rounds, witness);
            let what = format!("case {k}, {rounds} rounds, {witness:?}, some unknown");
            check(case, &encoding, &outcome, &host.calls, false, &what);
            for call in &host.calls {
                if let Call::Poll(ProbePoint::Probed { node, status: SolveStatus::Unknown }) = call {
                    assert!(!outcome.is_dead(*node), "{what}: {node:?} died after an Unknown probe");
                }
            }
        }
    }
}
