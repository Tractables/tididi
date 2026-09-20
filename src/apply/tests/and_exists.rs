//! `and_exists` against its own reference route.
//!
//! [`Quantification::Fused`](crate::Quantification::Fused) removes what it can
//! before the product and builds only the part of it the answer depends on;
//! [`Quantification::Product`](crate::Quantification::Product) builds the whole
//! conjunction and quantifies that. They must agree on the diagram, not merely
//! on the function, so every comparison here is `assert_same_shape` over two
//! minimized results.

use super::*;

use crate::Quantification;
use crate::limits::LimitConfig;
use crate::OperationError;
use crate::vtree::VtreeIdx;

/// The variable subsets a formula over `n` variables is quantified by: none,
/// all, each one alone, the odd ones, the even ones, and the first half.
fn quantified_subsets(n: u32) -> Vec<Vec<VarId>> {
    let all: Vec<VarId> = (1..=n).map(VarId).collect();
    let mut out = vec![Vec::new(), all.clone()];
    out.extend(all.iter().map(|&v| vec![v]));
    out.push(all.iter().copied().step_by(2).collect());
    out.push(all.iter().copied().skip(1).step_by(2).collect());
    out.push(all[..(n as usize).div_ceil(2)].to_vec());
    out
}

/// Split a formula's clauses in two, compile each half, and demand the two
/// routes agree on every subset of the variables.
fn both_routes_agree(eng: &Engine, vtree: &Arc<Vtree>, num_vars: u32, clauses: &[Vec<i32>], what: &str) {
    let mid = clauses.len().div_ceil(2);
    let f = compile_clauses(vtree, &clauses[..mid]);
    let g = compile_clauses(vtree, &clauses[mid..]);
    for vars in quantified_subsets(num_vars) {
        let fused = eng
            .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Fused)
            .unwrap();
        let product = eng
            .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Product)
            .unwrap();
        assert_canonical(&fused);
        assert_canonical(&product);
        assert_same_shape(&fused, &product, &format!("{what}, quantifying {vars:?}"));
    }
}

#[test]
fn the_two_routes_agree_on_the_fixed_corpus() {
    let eng = Engine::new();
    for (num_vars, clauses) in test_cases() {
        for (shape, vtree) in vtree_shapes(num_vars) {
            both_routes_agree(&eng, &vtree, num_vars, &clauses, shape);
        }
    }
}

#[test]
fn the_two_routes_agree_on_seeded_random_formulas() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5eed_f05ed);
    for num_vars in [3u32, 5, 8] {
        for round in 0..24 {
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
            for (shape, vtree) in vtree_shapes(num_vars) {
                both_routes_agree(
                    &eng, &vtree, num_vars, &clauses,
                    &format!("{shape}, {num_vars} vars, round {round}"),
                );
            }
        }
    }
}

/// The operand-locality rule's own shape: every quantified variable is
/// constrained by at most one operand, so the fused route quantifies both
/// operands before the product and the product never carries a target.
#[test]
fn the_two_routes_agree_when_each_target_is_local_to_one_operand() {
    let eng = Engine::new();
    let num_vars = 8;
    let odd: Vec<Vec<i32>> = vec![vec![1, 3], vec![-3, 5], vec![-5, 7], vec![-1, -7]];
    let even: Vec<Vec<i32>> = vec![vec![2, 4], vec![-4, 6], vec![-6, 8], vec![-2, -8]];
    for (shape, vtree) in vtree_shapes(num_vars) {
        let f = compile_clauses(&vtree, &odd);
        let g = compile_clauses(&vtree, &even);
        for vars in [
            vec![VarId(1), VarId(3), VarId(5), VarId(7)],
            vec![VarId(2), VarId(4), VarId(6), VarId(8)],
            (1..=8).map(VarId).collect::<Vec<_>>(),
            vec![VarId(1), VarId(2)],
        ] {
            let fused = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Fused)
                .unwrap();
            let product = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Product)
                .unwrap();
            assert_canonical(&fused);
            assert_same_shape(&fused, &product, &format!("{shape}, quantifying {vars:?}"));
        }
    }
}

/// The two ends of the range: a false operand, and a quantification that takes
/// every variable of the vtree.
#[test]
fn the_two_routes_agree_on_empty_and_total_quantifications() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let all: Vec<VarId> = (1..=6).map(VarId).collect();
    let satisfiable = compile_clauses(&vtree, &[vec![1, 2], vec![-3, 4]]);
    let contradictory = compile_clauses(&vtree, &[vec![5], vec![-5]]);
    let one = Tdd::one(&vtree);
    for (name, f, g) in [
        ("false on the left", contradictory.clone(), satisfiable.clone()),
        ("false on the right", satisfiable.clone(), contradictory.clone()),
        ("both false", contradictory.clone(), contradictory.clone()),
        ("true on the right", satisfiable.clone(), one.clone()),
        ("true on the left", one.clone(), satisfiable.clone()),
        ("both true", one.clone(), one.clone()),
    ] {
        for vars in [Vec::new(), all.clone(), vec![VarId(5)]] {
            let fused = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Fused)
                .unwrap();
            let product = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Product)
                .unwrap();
            assert_canonical(&fused);
            assert_same_shape(&fused, &product, &format!("{name}, quantifying {vars:?}"));
        }
    }
}

/// A target only one operand constrains never reaches the product. The pin is
/// the emitted-node cap: the reference route has to build the whole
/// conjunction of two eight-variable halves and trips it, while the fused
/// route quantifies each half on its own and answers inside the same cap.
#[test]
fn a_target_local_to_one_operand_never_reaches_the_product() {
    // Interleaved variable order, so the conjunction of the two halves is a
    // genuine product rather than two stacked blocks.
    let vtree = Arc::new(Vtree::linear(16));
    let odd: Vec<Vec<i32>> = (1..=7)
        .step_by(2)
        .map(|v| vec![-v, v + 2])
        .chain([vec![9, 11], vec![-11, 13], vec![-13, 15], vec![1, 15]])
        .collect();
    let even: Vec<Vec<i32>> = (2..=8)
        .step_by(2)
        .map(|v| vec![-v, v + 2])
        .chain([vec![10, 12], vec![-12, 14], vec![-14, 16], vec![2, 16]])
        .collect();
    let all: Vec<VarId> = (1..=16).map(VarId).collect();
    let cap = LimitConfig::none().with_output_node_cap(Some(64));
    std::sync::Arc::clone(vtree.context()).with_limits(cap, |eng| {
        let f = compile_clauses_on(eng, &vtree, &odd);
        let g = compile_clauses_on(eng, &vtree, &even);
        let fused = eng.and_exists_with(f.clone(), g.clone(), &all, Quantification::Fused);
        assert!(fused.is_ok(), "the fused route built the product: {fused:?}");
        let product = eng.and_exists_with(f, g, &all, Quantification::Product);
        assert_eq!(product.unwrap_err(), OperationError::OutputCap);
    });
}

/// The whole-subtree collapse's own shape: the quantified variables are one
/// contiguous block of the vtree that both operands constrain, so nothing can
/// be pushed into an operand and the fusion has to happen inside the sweep.
#[test]
fn the_two_routes_agree_when_a_quantified_block_straddles_both_operands() {
    let eng = Engine::new();
    let num_vars = 9;
    for (shape, vtree) in vtree_shapes(num_vars) {
        // Both formulas constrain the middle block (4, 5, 6) and one outer one.
        let f = compile_clauses(&vtree, &[vec![1, 4], vec![-2, 5], vec![-4, 6], vec![3, -5]]);
        let g = compile_clauses(&vtree, &[vec![7, 4], vec![-8, -5], vec![-6, 9], vec![7, 5]]);
        for vars in [
            vec![VarId(4), VarId(5), VarId(6)],
            vec![VarId(4), VarId(5)],
            vec![VarId(1), VarId(4), VarId(5), VarId(6), VarId(9)],
        ] {
            let fused = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Fused)
                .unwrap();
            let product = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Product)
                .unwrap();
            assert_canonical(&fused);
            assert_same_shape(&fused, &product, &format!("{shape}, quantifying {vars:?}"));
        }
    }
}

/// Quantifying variables of a vtree that holds more of them than the operands
/// mention: the levels above the block are untouched and the sweep has to stop
/// climbing where its partition stops changing.
#[test]
fn the_two_routes_agree_over_a_vtree_wider_than_the_formulas() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(12));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3]]);
    let g = compile_clauses(&vtree, &[vec![3, 4], vec![-4, 5]]);
    for vars in [
        vec![VarId(3)],
        vec![VarId(1), VarId(2), VarId(3), VarId(4), VarId(5)],
        vec![VarId(9), VarId(10), VarId(11), VarId(12)],
        (1..=12).map(VarId).collect::<Vec<_>>(),
    ] {
        let fused = eng
            .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Fused)
            .unwrap();
        let product = eng
            .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Product)
            .unwrap();
        assert_canonical(&fused);
        assert_same_shape(&fused, &product, &format!("quantifying {vars:?}"));
    }
}

/// Every internal vtree node of a small tree taken as the quantified block, so
/// the sweep meets a whole subtree, a straddling node and an untouched one at
/// every position they can occupy.
#[test]
fn the_two_routes_agree_with_every_subtree_of_the_vtree_quantified() {
    let eng = Engine::new();
    let num_vars = 7;
    let clauses = [vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5], vec![5, 6], vec![-6, 7]];
    for (shape, vtree) in vtree_shapes(num_vars) {
        let f = compile_clauses(&vtree, &clauses[..3]);
        let g = compile_clauses(&vtree, &clauses[3..]);
        for node in 0..vtree.num_nodes() {
            let vars: Vec<VarId> = (1..=num_vars)
                .map(VarId)
                .filter(|&v| {
                    let leaf = vtree.leaf_of(v).expect("every variable has a leaf");
                    is_below(&vtree, VtreeIdx(node as u32), leaf)
                })
                .collect();
            if vars.is_empty() {
                continue;
            }
            let fused = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Fused)
                .unwrap();
            let product = eng
                .and_exists_with(f.clone(), g.clone(), &vars, Quantification::Product)
                .unwrap();
            assert_canonical(&fused);
            assert_same_shape(&fused, &product, &format!("{shape}, subtree {node}"));
        }
    }
}

/// Whether `leaf` sits in the subtree rooted at `root`.
fn is_below(vtree: &Vtree, root: VtreeIdx, leaf: VtreeIdx) -> bool {
    let mut at = Some(leaf);
    while let Some(node) = at {
        if node == root {
            return true;
        }
        at = vtree.node(node).parent();
    }
    false
}

/// A subtree every leaf of which is quantified is never built. The pin is the
/// emitted-node cap again: both operands constrain the whole block, so nothing
/// can be pushed into an operand and the collapse inside the sweep is the only
/// thing that can keep the count down. The reference route emits the block's
/// product — about a thousand nodes — where the fused route emits one per level
/// of it.
#[test]
fn a_quantified_subtree_is_never_built() {
    // A right-linear stick, so everything but the first variable is one
    // subtree, and that subtree is what gets quantified.
    let vtree = Arc::new(Vtree::linear(18));
    // `f` pairs 2..9 with 10..17 bit by bit; `g` pairs them rotated by three,
    // so each has 2^8 cofactor classes over the block and neither operand is
    // constant anywhere in it.
    let mut f_clauses: Vec<Vec<i32>> = Vec::new();
    let mut g_clauses: Vec<Vec<i32>> = Vec::new();
    for i in 0..8i32 {
        let (bit, same, rotated) = (2 + i, 10 + i, 10 + (i + 3) % 8);
        f_clauses.push(vec![-bit, same]);
        f_clauses.push(vec![bit, -same]);
        g_clauses.push(vec![-bit, rotated]);
        g_clauses.push(vec![bit, -rotated]);
    }
    let f = compile_clauses(&vtree, &f_clauses);
    let g = compile_clauses(&vtree, &g_clauses);
    let vars: Vec<VarId> = (2..=18).map(VarId).collect();
    let capped = |how, cap| {
        let cfg = LimitConfig::none().with_output_node_cap(Some(cap));
        std::sync::Arc::clone(vtree.context())
            .with_limits(cfg, |eng| eng.and_exists_with(f.clone(), g.clone(), &vars, how))
    };
    assert!(capped(Quantification::Fused, 64).is_ok(), "the fused route built the block");
    assert_eq!(capped(Quantification::Product, 64).unwrap_err(), OperationError::OutputCap);
    assert_same_shape(
        &capped(Quantification::Fused, u64::MAX).unwrap(),
        &capped(Quantification::Product, u64::MAX).unwrap(),
        "a quantified block both operands constrain",
    );
}
