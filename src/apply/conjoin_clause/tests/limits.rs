//! The clause rebuild under the engine's limits: a stop decision and the
//! output cap both cut it between two spine levels, and the work clock is
//! charged for the pairs it reads and writes.

use std::sync::Arc;

use crate::limits::LimitConfig;
use crate::test_helpers::{compile_clauses_on, rand_cnf, stopping_engine, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree};
use crate::{Literal, OperationError, Engine, Tdd};

#[test]
fn a_stop_decision_cuts_the_rebuild() {
    let vtree = Arc::new(Vtree::balanced(4));
    let eng = stopping_engine();
    let f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [2, 3]).unwrap();
    assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(3).unwrap(), crate::Literal::try_from(4).unwrap()]).err(), Some(OperationError::Stopped));
}

#[test]
fn the_output_cap_counts_the_rebuilt_levels() {
    let vtree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [2, 3]).unwrap();
    let _armed = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(0)));
    assert_eq!(eng.and_clause(f, &[crate::Literal::try_from(3).unwrap(), crate::Literal::try_from(4).unwrap()]).err(), Some(OperationError::OutputCap));
}

/// The pairs of `d`'s levels at `spine`.
fn spine_pairs(d: &Tdd, spine: &[usize]) -> u64 {
    spine.iter().map(|&t| d.levels[t].pair_counts().sum::<usize>() as u64).sum()
}

/// Random formulas on every vtree shape, each conjoined with clauses of two
/// and three literals: the conjunction charges a unit for each pair of a
/// spine level it reads and each it writes, and beyond them no more than
/// two units for each literal and three for each vtree node, which the
/// clause's checks and its spine's walk charge.
#[test]
fn the_rebuild_charges_the_pairs_it_reads_and_writes() {
    let mut rng = Lcg::new(0xc1a0_5e11);
    let n = 14u32;
    let mut charged = 0u64;
    for (name, vtree) in vtree_shapes(n) {
        for case in 0..6 {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: 24, width: 3 }));
            let mut vars: Vec<u32> = (1..=n).collect();
            let width = 2 + case % 2;
            for k in 0..width {
                let j = k + rng.below(u64::from(n) - k as u64) as usize;
                vars.swap(k, j);
            }
            let clause: Vec<Literal> = vars[..width].iter()
                .map(|&v| if rng.coin() { Literal::pos(VarId(v)) } else { Literal::neg(VarId(v)) })
                .collect();
            // The levels whose subtree holds a clause variable.
            let mut spine = Vec::new();
            for lit in &clause {
                let mut cur = vtree.node(vtree.leaf_of(lit.var).unwrap()).parent();
                while let Some(t) = cur {
                    if !spine.contains(&t.idx()) {
                        spine.push(t.idx());
                    }
                    cur = vtree.node(t).parent();
                }
            }
            let read = spine_pairs(&f, &spine);
            let mark = eng.limits().mark();
            let out = eng.and_clause(f, &clause).unwrap();
            let work = eng.limits().work_since(mark);
            let pairs = read + spine_pairs(&out, &spine);
            let slack = (2 * width + 3 * vtree.num_nodes()) as u64;
            assert!(
                (pairs..=pairs + slack).contains(&work),
                "{name}, case {case}, clause {clause:?}: charged {work} for {pairs} pairs read and written"
            );
            charged += pairs;
        }
    }
    assert!(charged > 0, "expected spine levels with pairs");
}
