//! Levels whose cells are read by g-pair vectors, checked against the row
//! loop that reads every candidate off the child grids: the same diagram,
//! each node's pairs in the same order, and the same work.

use super::pair_rows::{conjoin, split_support_cases};
use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, SparseRoute};
use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf, rat, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx};

fn word_cell_arms(run: impl Fn() -> (Tdd, u64)) -> [(&'static str, (Tdd, u64)); 3] {
    [
        ("every level", word_cells_on(&run)),
        (
            "every level, rows sorted,",
            word_cells_on(|| word_rows_sorted(&run)),
        ),
        ("the levels that pay", run()),
    ]
}

/// Conjoined with every level that can read its cells by vectors reading
/// them (once with every row that does not walk its live cells writing
/// them off its sorted live candidates), with the levels that pay reading
/// them, and with none: the same levels, each node's pairs in the same
/// order, and the same work.
fn check(f: &Tdd, g: &Tdd, what: &str) {
    let (oracle, work) = word_cells_off(|| conjoin(f, g));
    let eng = Engine::new();
    for (name, (out, out_work)) in word_cell_arms(|| conjoin(f, g)) {
        assert!(
            eng.equivalent(&out, &oracle).unwrap(),
            "{what}: {name} read by vectors built another function"
        );
        assert!(
            is_self_conjunction(&out, &oracle),
            "{what}: {name} read by vectors built other levels"
        );
        assert_eq!(
            out_work, work,
            "{what}: {name} read by vectors took other work"
        );
    }
}

/// Random formulas over 18 variables on every vtree shape, dense random
/// functions over 10 and 14 variables, whose levels hold many pairs, sparse
/// ones over 16 and 18 variables, and random functions over 12 variables
/// that leave alternate variables under a vtree node to each operand, whose
/// parent level has a complete side: the levels read by vectors build what
/// the row loop builds.
#[test]
fn the_levels_read_by_vectors_build_what_the_row_loop_builds() {
    let mut rng = Lcg::new(0x77d0_c311);
    let before = word_cells_levels();
    let mut cases = 0usize;
    for shape in [
        CnfShape {
            clauses: 30,
            width: 3,
        },
        CnfShape {
            clauses: 60,
            width: 4,
        },
    ] {
        for (name, vtree) in vtree_shapes(18) {
            for case in 0..3 {
                let eng = Engine::new();
                let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 18, shape));
                check(&f, &g, &format!("{name}, {shape:?}, case {case}"));
                cases += 1;
            }
        }
    }
    // Sparse functions over 16 and 18 variables: nodes of more than two
    // words of pairs whose live cells hold a live candidate or two.
    for (n, one_ins) in [
        (10u32, &[2, 3, 8][..]),
        (14, &[2, 3, 8]),
        (16, &[64, 256, 1024]),
        (18, &[256, 1024]),
    ] {
        let vars: Vec<VarId> = (1..=n).map(VarId).collect();
        for (name, vtree) in vtree_shapes(n) {
            for &one_in in one_ins {
                let mut operand = || {
                    let models: Vec<u64> =
                        (0..1u64 << n).filter(|_| rng.below(one_in) == 0).collect();
                    Tdd::from_models(&vtree, &vars, &models).unwrap()
                };
                let (f, g) = (operand(), operand());
                check(
                    &f,
                    &g,
                    &format!("{name}, random functions of {n} variables, 1 in {one_in}"),
                );
                cases += 1;
            }
        }
    }
    for (what, f, g) in split_support_cases(12, &[(2, 3), (2, 16)], &mut rng) {
        check(&f, &g, &what);
        check(&g, &f, &what);
        cases += 2;
    }
    let after = word_cells_levels();
    let [grids, complete, grouped, rows, cells, sorted, _, _, _]: [u64; 9] =
        std::array::from_fn(|k| after[k] - before[k]);
    let bits = rows - cells - sorted;
    eprintln!(
        "levels read by vectors: {grids} of two grids, {complete} with a complete side, {grouped} grouped cells, {rows} rows ({cells} walking their live cells, {bits} off cell bits, {sorted} off sorted candidates), over {cases} cases"
    );
    assert!(
        grids > 0,
        "expected levels of two grids read by vectors, got none over {cases} cases"
    );
    assert!(
        complete > 0,
        "expected levels with a complete side read by vectors, got none over {cases} cases"
    );
    assert!(
        grouped > 0,
        "expected grouped cells read by vectors, got none over {cases} cases"
    );
    assert!(
        rows > 0,
        "expected rows read by vectors, got none over {cases} cases"
    );
    assert!(
        cells > 0,
        "expected rows walking their live cells alone, got none over {cases} cases"
    );
    assert!(
        bits > 0,
        "expected rows written off cell bits, got none over {cases} cases"
    );
    assert!(
        sorted > 0,
        "expected rows written off sorted candidates, got none over {cases} cases"
    );
}

/// Conjoin `f` and `g` on a fresh engine, summing out `targets`, on
/// `route`, and the work it took.
fn marginalize(f: &Tdd, g: &Tdd, targets: &[VtreeIdx], route: SparseRoute) -> (Tdd, u64) {
    let eng = Engine::new();
    let _route = eng
        .limits()
        .scope(LimitConfig::none().with_sparse_route(route));
    let mark = eng.limits().mark();
    let out = eng
        .and_marginalizing(f.clone(), g.clone(), targets)
        .unwrap();
    (out, eng.limits().work_since(mark))
}

/// [`check`] for a conjunction that sums out `targets`: the same levels,
/// counts and value, and the same work. Once with every level on a dense
/// route, and once with every level whose one marginal child is not
/// carried through on the sparse-marginal route.
fn check_summed(f: &Tdd, g: &Tdd, targets: &[VtreeIdx], what: &str) {
    for route in [
        SparseRoute {
            sparsity: 64,
            min_grid: usize::MAX,
        },
        SparseRoute {
            sparsity: u32::MAX,
            min_grid: 0,
        },
    ] {
        check_summed_on(f, g, targets, route, &format!("{what}, {route:?}"));
    }
}

fn check_summed_on(f: &Tdd, g: &Tdd, targets: &[VtreeIdx], route: SparseRoute, what: &str) {
    let (oracle, work) = word_cells_off(|| marginalize(f, g, targets, route));
    for (name, (out, out_work)) in word_cell_arms(|| marginalize(f, g, targets, route)) {
        assert_eq!(
            format!("{:?}", out.levels),
            format!("{:?}", oracle.levels),
            "{what}: {name} read by vectors built other levels"
        );
        assert_eq!(
            out.model_count(),
            oracle.model_count(),
            "{what}: {name} read by vectors counted otherwise"
        );
        assert_eq!(
            format!("{:?}", out.weighted_value().unwrap()),
            format!("{:?}", oracle.weighted_value().unwrap()),
            "{what}: {name} read by vectors summed another value"
        );
        assert_eq!(
            out_work, work,
            "{what}: {name} read by vectors took other work"
        );
    }
}

/// Random formulas over 16 variables and dense random functions over 10
/// and 14 variables on every vtree shape, unweighted and weighted, summing
/// out every internal level, the root, one level below it, or a random
/// set of levels: the streamed levels read by vectors fold what the row
/// loop folds, and the sparse-marginal levels build what it builds.
#[test]
fn the_streamed_levels_read_by_vectors_fold_what_the_row_loop_folds() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    let mut rng = Lcg::new(0x3c5e_a1f7);
    let before = word_cells_levels();
    let mut cases = 0usize;
    let mut operands: Vec<(String, Arc<Vtree>, Tdd, Tdd)> = Vec::new();
    for (name, vtree) in vtree_shapes(16) {
        for case in 0..2 {
            let eng = Engine::new();
            let shape = CnfShape {
                clauses: 40,
                width: 3,
            };
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 16, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 16, shape));
            operands.push((
                format!("{name}, formulas, case {case}"),
                Arc::clone(&vtree),
                f,
                g,
            ));
        }
    }
    for (n, one_ins) in [(10u32, &[2, 3][..]), (14, &[3, 8])] {
        let vars: Vec<VarId> = (1..=n).map(VarId).collect();
        for (name, vtree) in vtree_shapes(n) {
            for &one_in in one_ins {
                let mut operand = || {
                    let models: Vec<u64> =
                        (0..1u64 << n).filter(|_| rng.below(one_in) == 0).collect();
                    Tdd::from_models(&vtree, &vars, &models).unwrap()
                };
                let (f, g) = (operand(), operand());
                operands.push((
                    format!("{name}, random functions of {n} variables, 1 in {one_in}"),
                    Arc::clone(&vtree),
                    f,
                    g,
                ));
            }
        }
    }
    for (what, vtree, f, g) in operands {
        let internal: Vec<VtreeIdx> = vtree
            .bottomup()
            .filter(|&v| !vtree.node(v).is_leaf())
            .collect();
        let root = vtree.root();
        let below = internal
            .iter()
            .copied()
            .find(|&v| vtree.node(v).parent() == Some(root));
        let some: Vec<VtreeIdx> = internal.iter().copied().filter(|_| rng.coin()).collect();
        let mut target_sets = vec![
            ("every internal level", internal.clone()),
            ("the root", vec![root]),
            ("some levels", some),
        ];
        target_sets.extend(below.map(|v| ("a level below the root", vec![v])));
        let values: Vec<_> = (0..vtree.num_vars())
            .map(|_| LiteralWeights {
                negative: rat(1 + rng.below(7) as i64, 8),
                positive: rat(1 + rng.below(5) as i64, 6),
            })
            .collect();
        let weighted = |t: &Tdd, arithmetic| {
            let mut t = t.clone();
            t.set_weights(WeightStore::new(
                RationalWeights::from_literals(&values),
                arithmetic,
            ))
            .unwrap();
            t
        };
        for (targets_name, targets) in &target_sets {
            check_summed(
                &f,
                &g,
                targets,
                &format!("{what}, summing out {targets_name}"),
            );
            for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
                let (wf, wg) = (weighted(&f, arithmetic), weighted(&g, arithmetic));
                check_summed(
                    &wf,
                    &wg,
                    targets,
                    &format!("{what}, weighted {arithmetic:?}, summing out {targets_name}"),
                );
            }
            cases += 3;
        }
    }
    let after = word_cells_levels();
    let [
        _,
        _,
        grouped,
        rows,
        cells,
        sorted,
        streamed,
        marginal,
        sparse,
    ] = std::array::from_fn(|k| after[k] - before[k]);
    let bits = rows - cells - sorted;
    eprintln!(
        "streamed levels read by vectors: {streamed} ({marginal} through marginal lookups), sparse-marginal levels {sparse}, {grouped} grouped cells, {rows} rows ({cells} walking their live cells, {bits} off cell bits, {sorted} off sorted candidates), over {cases} cases"
    );
    assert!(
        sparse > 0,
        "expected sparse-marginal levels read by vectors, got none over {cases} cases"
    );
    assert!(
        streamed > marginal,
        "expected streamed levels read by vectors off two grids, got {} over {cases} cases",
        streamed - marginal
    );
    assert!(
        marginal > 0,
        "expected streamed levels read by vectors through marginal lookups, got none over {cases} cases"
    );
    assert!(
        rows > 0,
        "expected rows of streamed levels read by vectors, got none over {cases} cases"
    );
}
