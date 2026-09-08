//! Test-only helpers shared across the crate test modules.

use std::sync::Arc;

use num_bigint::BigUint;

use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::query::compute_node_counts;
use crate::apply::conjoin::apply_and;
use crate::diagram::{Tdd, assert_can_make_marginal};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// DIMACS-style literals (`±(var+1)`) to `Literal`s.
pub fn lits(clause: &[i32]) -> Vec<Literal> {
    clause.iter().map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0)).collect()
}

/// Conjoin DIMACS-style clauses one at a time, minimizing after each.
pub fn compile_clauses(vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut acc = constant_one(vtree);
    for clause in clauses {
        let cl = clause_to_tdd(vtree, &lits(clause));
        acc = apply_and(acc, cl);
        minimize(&mut acc);
    }
    acc
}

/// Model count of DIMACS-style clauses by enumeration.
pub fn brute_force_count(num_vars: u32, clauses: &[Vec<i32>]) -> u64 {
    (0..(1u64 << num_vars))
        .filter(|assignment| {
            clauses.iter().all(|clause| {
                clause.iter().any(|&lit| {
                    let val = (assignment >> (lit.unsigned_abs() - 1)) & 1 == 1;
                    (lit > 0) == val
                })
            })
        })
        .count() as u64
}

/// Formulas covering SAT, UNSAT, unit, wide, and don't-care shapes.
pub fn test_cases() -> Vec<(u32, Vec<Vec<i32>>)> {
    vec![
        (3, vec![vec![1, 2], vec![-2, 3], vec![-1, -3]]),
        (4, vec![vec![1, 2], vec![3, 4], vec![-1, -3], vec![-2, -4]]),
        (4, vec![vec![1], vec![2, -3], vec![-1, 3, 4], vec![-2, -4]]),
        (3, vec![vec![1, 2, 3]]),
        (2, vec![vec![1, 2], vec![-1, -2], vec![1, -2], vec![-1, 2]]),
        (1, vec![vec![1], vec![-1]]),
        (2, vec![vec![1], vec![2], vec![-1, -2]]),
        (3, vec![vec![1], vec![-1], vec![2, 3]]),
        (4, vec![vec![1, 2], vec![-1, -2], vec![1, -2], vec![-1, 2],
                 vec![3, 4], vec![-3, -4], vec![3, -4], vec![-3, 4]]),
        (3, vec![vec![1]]),
        (3, vec![vec![-2]]),
        (4, vec![vec![1, 2, 3, 4]]),
        (4, vec![vec![1, 2], vec![3], vec![-4]]),
        (4, vec![vec![1, 2], vec![-1, -2]]),
        (5, vec![vec![1]]),
        (3, vec![vec![1, 2], vec![-1, -2], vec![3]]),
        (6, vec![vec![1, 2], vec![-3, -4]]),
    ]
}

/// Per-level pair lists with node indices renamed to a canonical order, so two
/// diagrams over one vtree compare equal iff they are the same up to node
/// numbering. Leaf levels and marginal levels compare by width only; refs into
/// a marginal child are slot or inline refs and compare by raw value.
pub fn normalized_levels(tdd: &Tdd) -> Vec<Vec<Vec<(u32, u32)>>> {
    let vtree = &tdd.vtree;
    let n = vtree.num_nodes();
    let mut remap: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut out: Vec<Vec<Vec<(u32, u32)>>> = vec![Vec::new(); n];
    for (t, _) in vtree.leaf_bottomup() {
        remap[t.idx()] = (0..crate::diagram::LEAF_WIDTH as u32).collect();
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let level = &tdd.levels[t.idx()];
        if level.is_marginal() {
            remap[t.idx()] = (0..level.width() as u32).collect();
            out[t.idx()] = vec![Vec::new(); level.width()];
            continue;
        }
        let left_marg = tdd.levels[left.idx()].is_marginal();
        let right_marg = tdd.levels[right.idx()].is_marginal();
        let mut indexed: Vec<(usize, Vec<(u32, u32)>)> = (0..level.nodes.len())
            .map(|i| {
                let mut pairs: Vec<(u32, u32)> = level
                    .pairs_of_idx(i)
                    .iter()
                    .map(|p| {
                        (
                            if left_marg { p.left.0 } else { remap[left.idx()][p.left.idx()] },
                            if right_marg { p.right.0 } else { remap[right.idx()][p.right.idx()] },
                        )
                    })
                    .collect();
                pairs.sort();
                (i, pairs)
            })
            .collect();
        indexed.sort_by(|a, b| a.1.cmp(&b.1));
        remap[t.idx()] = vec![0; level.nodes.len()];
        for (new_i, (old_i, pairs)) in indexed.into_iter().enumerate() {
            remap[t.idx()][old_i] = new_i as u32;
            out[t.idx()].push(pairs);
        }
    }
    out
}

/// `BigUint` → u128, panicking if the value exceeds 128 bits. Used by tests
/// that feed `compute_node_counts` output into `make_marginal`, which
/// requires u128 counts.
pub fn big_to_u128(b: &BigUint) -> u128 {
    let digits = b.to_u64_digits();
    match digits.len() {
        0 => 0,
        1 => digits[0] as u128,
        2 => ((digits[1] as u128) << 64) | (digits[0] as u128),
        _ => panic!("test fixture count overflows u128: {b}"),
    }
}

/// Bottom-up marginalize every internal, non-marginal, width≥1 level in
/// the subtree rooted at `root` (inclusive). Counts are derived from the
/// current TDD shape via `compute_node_counts`. Mirrors production's
/// `marginalize_batch` + `cascade_marginalize` semantics for a single
/// subtree, without the streaming-marginal hooks.
pub fn marginalize_subtree(tdd: &mut Tdd, root: VtreeIdx) {
    let vtree = tdd.vtree.clone();
    let counts = compute_node_counts(tdd);
    for &t in vtree.bottomup_topo() {
        let ti = t.idx();
        let mut under = t == root;
        if !under {
            let mut cur = t;
            while let Some(p) = vtree.node(cur).parent() {
                if p == root {
                    under = true;
                    break;
                }
                cur = p;
            }
        }
        if !under {
            continue;
        }
        if matches!(*vtree.node(VtreeIdx(ti as u32)), VtreeNode::Leaf { .. }) || tdd.levels[ti].is_marginal() {
            continue;
        }
        let w = tdd.levels[ti].width();
        if w == 0 {
            continue;
        }
        let u128_counts: Vec<u128> = (0..w).map(|i| big_to_u128(&counts[ti][i])).collect();
        assert_can_make_marginal(&tdd.levels, &vtree, t);
        tdd.levels[ti].make_marginal(u128_counts, None);
    }
    // Emulate production marginalization, which tags every persisted marg-side
    // slot ref (bit 30) so the 0=inline decode invariant holds. Without this the
    // strict decode assert fires when a later reader hits a raw slot ref.
    crate::diagram::tag_all_marg_side_slots(tdd, None);
}
