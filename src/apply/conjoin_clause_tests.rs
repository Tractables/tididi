use super::*;
use crate::engine::Engine;
use crate::vtree::Vtree;
use num_bigint::BigUint;

/// Independent oracle: brute-force model count over `n` vars. Shares no
/// machinery with the spine conjoin, so it cannot agree with a mis-sized
/// emit on a wrong answer.
fn brute(n: u32, cnf: &[&[i32]]) -> BigUint {
    let mut count: u64 = 0;
    for mask in 0u32..(1u32 << n) {
        let sat = cnf.iter().all(|c| {
            c.iter().any(|&l| {
                let bit = (mask >> (l.unsigned_abs() - 1)) & 1 == 1;
                (l > 0) == bit
            })
        });
        if sat {
            count += 1;
        }
    }
    BigUint::from(count)
}

/// Fold a CNF into a TDD over `vtree`, one clause at a time through
/// `apply_and_clause` — the rebuild path under test.
fn fold_cnf(_eng: &Engine, vtree: &Arc<Vtree>, cnf: &[&[i32]]) -> Tdd {
    let mut acc = Tdd::one(vtree);
    for clause in cnf {
        let lits: Vec<Literal> = clause.iter().map(|&l| l.into()).collect();
        acc = apply_and_clause(&mut acc, &lits);
    }
    acc
}

/// Per-level emit sizing is demand-driven, so a rebuild whose real output
/// lands far below the 4x worst-case bound must still emit every pair.
/// The three trailing unit clauses are the collapse: each rebuilt level
/// keeps at most one c_t pair per surviving node where the bound allows
/// four, so the level grows past the initial slab on the early clauses and
/// stays far under it on the late ones — both regimes of the top-up.
#[test]
fn clause_rebuild_exact_when_output_far_below_worst_case() {
    let eng = Engine::new();
    let n: u32 = 7;
    let cnf: &[&[i32]] = &[
        &[1, -2, 3],
        &[-3, 4],
        &[2, 5, -6],
        &[4, -5, 7],
        &[-1, 6, -7],
        &[3, -4, 5, -6],
        &[1],
        &[-2],
        &[7],
    ];
    let acc = fold_cnf(&eng, &Arc::new(Vtree::random(n, 7)), cnf);
    let expected = brute(n, cnf);
    assert!(expected > BigUint::from(0u32), "fixture must stay satisfiable");
    assert_eq!(acc.model_count(), expected);
}

/// The same conjunction, one clause at a time against a wide accumulator
/// whose levels are rebuilt with BOTH children on the spine (`both_rel`,
/// the 3-pairs-per-input-pair case) and with the complement lane live
/// (`compute_dt`) — the widest per-node top-up. Counts must match the
/// oracle exactly.
#[test]
fn both_relevant_rebuild_exact_under_demand_reserve() {
    let eng = Engine::new();
    let n: u32 = 6;
    // Every clause spans variables from both halves of the vtree, so the
    // meet levels take the both_rel path.
    let cnf: &[&[i32]] = &[
        &[1, 4],
        &[-1, 5],
        &[2, -4, 6],
        &[-2, -5, 3],
        &[3, -6],
        &[-3, 4, -5],
    ];
    let acc = fold_cnf(&eng, &Arc::new(Vtree::random(n, 3)), cnf);
    assert_eq!(acc.model_count(), brute(n, cnf));
}
