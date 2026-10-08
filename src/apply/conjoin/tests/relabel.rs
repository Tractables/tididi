//! The relabelling route against the general routes it replaces: the same
//! diagram, the same count and the same quantified diagram, on every vtree
//! shape, with one operand's support under each subtree in turn, on both
//! product-store layouts; and operands handed back on a refusal.

use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, SparseRoute, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, assert_same_shape, rand_conj_over, same_storage, vtree_shapes, Lcg};
use crate::test_helpers::check::check_no_false_nodes_in_levels;
use crate::vtree::{VarId, Vtree, VtreeNode};

/// The variables at the leaves under `t`.
pub(super) fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<u32> {
    let mut stack = vec![t];
    let mut vars = Vec::new();
    while let Some(n) = stack.pop() {
        match vtree.node(n) {
            VtreeNode::Leaf { var, .. } => vars.push(var.0),
            VtreeNode::Internal { left, right, .. } => stack.extend([*left, *right]),
        }
    }
    vars.sort_unstable();
    vars
}

/// A random function of `vars`, minimized.
pub(super) fn function_of(vtree: &Arc<Vtree>, vars: &[u32], rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj_over(vtree, vars, 8, 3, false, rng);
    f.minimize().unwrap();
    f
}

/// Pairs of a function `g` over the variables under each internal level
/// below the root, which is then one node with one pair on every level
/// above that one, the levels the route takes; and a function `f` over
/// every variable, where `g` kills some of `f`'s products, or over the
/// variables outside that level, where it kills none and the levels above
/// it are moved whole.
pub(super) fn cases(nvars: u32, seed: u64) -> Vec<(String, Tdd, Tdd)> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for (shape, vtree) in vtree_shapes(nvars) {
        let all: Vec<u32> = (1..=nvars).collect();
        for (t, _, _) in vtree.internal_bottomup() {
            if t == vtree.root() {
                continue;
            }
            let under = vars_under(&vtree, t);
            let outside: Vec<u32> = all.iter().copied().filter(|v| !under.contains(v)).collect();
            let g = function_of(&vtree, &under, &mut rng);
            let f = function_of(&vtree, &all, &mut rng);
            out.push((format!("{shape}, f over all, g under {t:?}"), f, g.clone()));
            let f = function_of(&vtree, &outside, &mut rng);
            out.push((format!("{shape}, f outside, g under {t:?}"), f, g));
        }
    }
    out
}

/// Conjoin both ways round, with the route open and closed, and require the
/// same diagram, with no false node, canonical once minimized; under `route`
/// when given.
fn same_as_general_routes_with(what: &str, f: &Tdd, g: &Tdd, route: Option<SparseRoute>) {
    for (a, b) in [(f, g), (g, f)] {
        let eng = Engine::new();
        let _scope = route.map(|r| eng.limits().scope(LimitConfig::none().with_sparse_route(r)));
        let mut out = eng.and(a.clone(), b.clone()).unwrap();
        let oracle = no_relabel(|| eng.and(a.clone(), b.clone()).unwrap());
        check_no_false_nodes_in_levels(&out).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_same_shape(&out, &oracle, what);
        eng.minimize(&mut out).unwrap();
        assert_canonical(&out);
    }
}

fn same_as_general_routes(what: &str, f: &Tdd, g: &Tdd) {
    same_as_general_routes_with(what, f, g, None);
}

#[test]
fn relabelled_levels_are_the_general_routes_levels() {
    let before = relabel_census();
    for (what, f, g) in cases(9, 0x7e1a_be11) {
        same_as_general_routes(&what, &f, &g);
    }
    let census = relabel_census();
    assert!(census[0] > before[0], "no level was moved whole");
    assert!(census[1] > before[1], "no level was rebuilt");
}

/// With the sparse gate at its floor the product store claims grids level by
/// level, and a level the sparse route builds keeps a product list and no
/// grid: the route reads its children from lists, and its parents read the
/// complete levels it moved, which hold neither, by their cell order.
#[test]
fn relabelled_levels_read_product_lists() {
    let before = relabel_census();
    for (what, f, g) in cases(9, 0x11_57ed) {
        same_as_general_routes_with(&what, &f, &g, Some(SparseRoute { sparsity: 1, min_grid: 0 }));
    }
    let census = relabel_census();
    assert!(census[0] > before[0] && census[1] > before[1], "the route never ran on product lists");
}

/// The count of a conjunction whose root a relabelled level feeds, and the
/// conjunction that quantifies a variable on the way, are the general
/// routes' own.
#[test]
fn counts_and_quantified_conjunctions_agree() {
    let mut relabelled = 0;
    for (what, f, g) in cases(8, 0xc0_4e7) {
        let eng = Engine::new();
        let before = relabel_census();
        let count = eng.and_model_count(f.clone(), g.clone(), &[]).unwrap();
        let after = relabel_census();
        relabelled += after[0] + after[1] - before[0] - before[1];
        let oracle = no_relabel(|| eng.and_model_count(f.clone(), g.clone(), &[]).unwrap());
        assert_eq!(count, oracle, "{what}: count");
        assert_eq!(count, eng.model_count(&eng.and(f.clone(), g.clone()).unwrap()).unwrap(), "{what}: built count");
        let quantified = [VarId(1), VarId(5)];
        let out = eng.and_exists(f.clone(), g.clone(), &quantified).unwrap();
        let oracle = no_relabel(|| eng.and_exists(f.clone(), g.clone(), &quantified).unwrap());
        assert!(eng.equivalent(&out, &oracle).unwrap(), "{what}: and_exists");
    }
    assert!(relabelled > 0, "no count relabelled a level");
}

/// A conjunction that sums a subtree out, alone or beside its sibling, is
/// the general routes' own level for level, and so is its count: the
/// relabelling route takes the levels over the targets' parents and beside
/// them.
#[test]
fn marginalizing_conjunctions_agree() {
    let mut relabelled = 0;
    for (what, f, g) in cases(8, 0x3a_291) {
        let vtree = Arc::clone(f.vtree());
        for (t, _, _) in vtree.internal_bottomup() {
            if t == vtree.root() {
                continue;
            }
            for targets in [vec![t], vec![t, vtree.sibling(t)]] {
                let eng = Engine::new();
                let before = relabel_census();
                let out = eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap();
                let after = relabel_census();
                relabelled += after[0] + after[1] - before[0] - before[1];
                let oracle = no_relabel(|| eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap());
                let what = format!("{what}, targets {targets:?}");
                assert_same_shape(&out, &oracle, &what);
                let count = eng.model_count(&out).unwrap();
                assert_eq!(count, eng.model_count(&oracle).unwrap(), "{what}: count");
                let plain = eng.and(f.clone(), g.clone()).unwrap();
                assert_eq!(count, eng.model_count(&plain).unwrap(), "{what}: plain count");
                let counted = eng.and_model_count(f.clone(), g.clone(), &targets).unwrap();
                assert_eq!(counted, count, "{what}: and_model_count");
                let oracle = no_relabel(|| eng.and_model_count(f.clone(), g.clone(), &targets).unwrap());
                assert_eq!(counted, oracle, "{what}: and_model_count oracle");
            }
        }
    }
    assert!(relabelled > 0, "no marginalizing conjunction relabelled a level");
}

/// An operand an earlier sum left a marginal level in is read through that
/// level where the other operand is constant-true over it: the general
/// routes' diagram and count, and the count of the conjunction unsummed.
#[test]
fn a_summed_operand_is_read_through_its_marginal_level() {
    let mut read_through = 0;
    for (what, f, g) in cases(8, 0x5e_77) {
        let vtree = Arc::clone(f.vtree());
        let support = crate::test_helpers::support_mask(&g);
        let eng = Engine::new();
        let whole = eng.model_count(&eng.and(f.clone(), g.clone()).unwrap()).unwrap();
        for (t, _, _) in vtree.internal_bottomup() {
            if t == vtree.root() || vars_under(&vtree, t).iter().any(|&v| support[v as usize - 1]) {
                continue;
            }
            let mut summed = f.clone();
            eng.marginalize_levels(&mut summed, &[t]).unwrap();
            if !summed.level(t).is_marginal() {
                continue;
            }
            let what = format!("{what}, {t:?} summed");
            let before = read_through_census();
            let out = eng.and_marginalizing(summed.clone(), g.clone(), &[]).unwrap();
            read_through += read_through_census() - before;
            let oracle = no_relabel(|| eng.and_marginalizing(summed.clone(), g.clone(), &[]).unwrap());
            assert_same_shape(&out, &oracle, &what);
            assert_eq!(eng.model_count(&out).unwrap(), whole, "{what}: count");
            let counted = eng.and_model_count(summed.clone(), g.clone(), &[]).unwrap();
            assert_eq!(counted, whole, "{what}: and_model_count");
        }
    }
    assert!(read_through > 0, "no conjunction read through a marginal level");
}

/// A level of one node the route wrote in its carrier's order, each run of
/// one left reference with its right sides out of order where the map of
/// the right side is not monotone, then quantified on a variable under its
/// left side, the right side untouched: the same function as quantifying
/// the minimized conjunction, canonical once minimized.
///
/// The quantification never leaves such a cell's order to the sort of its
/// runs alone, which
/// `a_level_of_one_node_with_unsorted_runs_is_written_in_order` reaches by
/// a direct call: the sweep starts from a pruned diagram, where the one
/// node at the level names every node of its left child, so a map of the
/// left side that changed anything merges two of them, shares a cell
/// between two or permutes the cells, and each of those sorts every cell.
#[test]
fn a_relabelled_level_quantified_on_its_left_side_agrees() {
    let (mut levels, mut quantified) = (0usize, 0usize);
    let relabelled = relabel_census();
    for (nvars, seed) in [(8, 0x0f2_e2e), (10, 0x0f2_e2f), (12, 0x0f2_e30), (12, 0x0f2_e31)] {
        for (what, f, g) in cases(nvars, seed) {
            for (a, b) in [(&f, &g), (&g, &f)] {
                let eng = Engine::new();
                let out = eng.and(a.clone(), b.clone()).unwrap();
                let mut minimized = out.clone();
                eng.minimize(&mut minimized).unwrap();
                let vtree = Arc::clone(out.vtree());
                for (t, left, _) in vtree.internal_bottomup() {
                    let level = &out.levels[t.idx()];
                    if level.nodes().len() != 1 {
                        continue;
                    }
                    let pairs = level.pairs_vec(0);
                    if !pairs.is_sorted_by_key(|pair| pair.left) || pairs.windows(2).all(|w| w[0] < w[1]) {
                        continue;
                    }
                    levels += 1;
                    for v in vars_under(&vtree, left) {
                        let got = eng.exists_vars(out.clone(), &[VarId(v)]).unwrap();
                        let oracle = eng.exists_vars(minimized.clone(), &[VarId(v)]).unwrap();
                        let what = format!("{what}, {t:?}, ∃{v}");
                        assert!(eng.equivalent(&got, &oracle).unwrap(), "{what}");
                        let mut got = got;
                        eng.minimize(&mut got).unwrap();
                        assert_canonical(&got);
                        assert_same_shape(&got, &oracle, &what);
                        quantified += 1;
                    }
                }
            }
        }
    }
    let rebuilt = relabel_census()[1] - relabelled[1];
    assert!(rebuilt > 0 && levels > 10 && quantified > 20,
        "rebuilt {rebuilt}, levels with unsorted runs {levels}, quantified {quantified}");
}

/// A conjunction that quantifies every variable under a subtree with
/// [`Quantification::FusedSubtrees`] collapses that subtree, whose products
/// are then one `⊤` for every satisfiable cell: the route writes the level
/// above with two carrier pairs meeting in one, which the quantification's
/// regroup of that level merges. The result is the general routes' own and
/// the quantified product's, and some level so written held a pair twice.
#[test]
fn a_quantifying_conjunction_relabels_above_a_collapsed_subtree() {
    use crate::Quantification;
    let (relabelled, repeated) = (relabel_census(), repeated_pair_census());
    for (what, f, g) in cases(8, 0x5_7b7e) {
        let vtree = Arc::clone(f.vtree());
        for (s, _, _) in vtree.internal_bottomup() {
            if s == vtree.root() {
                continue;
            }
            let vars: Vec<VarId> = vars_under(&vtree, s).into_iter().map(VarId).collect();
            for (a, b) in [(&f, &g), (&g, &f)] {
                let eng = Engine::new();
                let what = format!("{what}, quantifying under {s:?}");
                let out = eng.and_exists_with(a.clone(), b.clone(), &vars, Quantification::FusedSubtrees).unwrap();
                let oracle = no_relabel(|| {
                    eng.and_exists_with(a.clone(), b.clone(), &vars, Quantification::FusedSubtrees).unwrap()
                });
                assert_canonical(&out);
                assert_same_shape(&out, &oracle, &what);
                let product = eng.and_exists_with(a.clone(), b.clone(), &vars, Quantification::Product).unwrap();
                assert_same_shape(&out, &product, &format!("{what}, against the product"));
                assert_eq!(eng.model_count(&out).unwrap(), eng.model_count(&product).unwrap(), "{what}: count");
            }
        }
    }
    let rebuilt = relabel_census()[1] - relabelled[1];
    let repeated = repeated_pair_census() - repeated;
    assert!(rebuilt > 0, "the route rebuilt no level");
    assert!(repeated > 0, "no relabelled node held a pair twice");
}

/// A conjunction refused at any work point after the route moved a level
/// gives both operands back as they were.
#[test]
fn a_refused_conjunction_gives_moved_levels_back() {
    let vtree = Arc::new(Vtree::balanced(12));
    let eng = Engine::new();
    let u = vtree.internal_bottomup().map(|(t, _, _)| t).find(|&t| vars_under(&vtree, t).len() == 3).expect("a level over three variables");
    let under = vars_under(&vtree, u);
    let outside: Vec<i32> = (1..=12).filter(|v| !under.contains(v)).map(|v| v as i32).collect();
    let conj = |clauses: &[Vec<i32>]| {
        let mut f = eng.cube(&vtree, std::iter::empty::<i32>()).unwrap();
        for c in clauses {
            f = eng.and(f, eng.clause(&vtree, c.iter().copied()).unwrap()).unwrap();
        }
        eng.minimize(&mut f).unwrap();
        f
    };
    let f = conj(&outside.chunks(3).map(|c| vec![c[0], -c[1], c[2]]).collect::<Vec<_>>());
    let (a, b, c) = (under[0] as i32, under[1] as i32, under[2] as i32);
    let g = conj(&[vec![a, b], vec![-b, c]]);
    let expected = eng.and(f.clone(), g.clone()).unwrap();
    let before = relabel_census();
    let mut refusals = 0;
    for n in 0.. {
        let start = eng.limits().work_units();
        let rules = StopRules { unconditional: Some(StopAt::WorkUnits(start + n)), after_pairs: None };
        let scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        let outcome = eng.and_restoring(f.clone(), g.clone());
        drop(scope);
        match outcome {
            Ok(out) => {
                assert!(eng.equivalent(&out, &expected).unwrap(), "granted at {n}: a different function");
                break;
            }
            Err(refused) => {
                assert!(same_storage(&refused.f, &f) && same_storage(&refused.g, &g), "refused at {n}: an operand changed");
                refusals += 1;
            }
        }
        assert!(n < 10_000, "never granted");
    }
    assert!(refusals > 0, "never refused");
    assert!(relabel_census()[0] > before[0], "no level was moved whole");
}
