use std::sync::Arc;

use super::*;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::*;
use crate::vtree::Vtree;

/// Each level's nodes, each as its pairs in their order: what a reader of the
/// levels sees, stored or implied alike.
fn pairs_by_node(f: &Tdd) -> Vec<Vec<Vec<ChildPair>>> {
    (0..f.levels.len()).map(|t| {
        let t = VtreeIdx(t as u32);
        if !renumbered(f, t) { return Vec::new(); }
        let level = f.level(t);
        (0..level.slot_count()).map(|i| level.pairs_iter_of_idx(i).collect()).collect()
    }).collect()
}

/// Level `t`'s kept counts, every one of which fits the fast lane here.
fn column(f: &Tdd, t: VtreeIdx) -> Option<Vec<u128>> {
    let column = f.levels.counts()?.column(t)?;
    Some((0..column.len()).map(|i| match column.get(i) {
        CountRead::Fast(v) => v,
        CountRead::Big(_) => panic!("a test count past 128 bits"),
    }).collect())
}

/// `g` is `f` renumbered: the same function, the same level widths and pair
/// counts, and every node of `g` holds the pairs of a node of `f` in their
/// order, its sides renumbered as their nodes are.
fn assert_renumbering_of(g: &Tdd, f: &Tdd) {
    if f.is_zero() {
        assert!(g.is_zero());
        return;
    }
    assert!(g.equivalent(f).unwrap());
    assert_eq!(g.output().vtree, f.output().vtree);
    for t in 0..f.levels.len() {
        let t = VtreeIdx(t as u32);
        assert_eq!(g.level(t).slot_count(), f.level(t).slot_count(), "level {t:?}");
        if !f.level(t).is_marginal() {
            assert_eq!(g.level(t).live_pairs(), f.level(t).live_pairs(), "level {t:?}");
        }
    }
    // Recover the renumbering top-down from the outputs, and check each
    // node's pairs through it.
    let (fp, gp) = (pairs_by_node(f), pairs_by_node(g));
    let mut to_f: Vec<Vec<Option<usize>>> = gp.iter().map(|level| vec![None; level.len()]).collect();
    if renumbered(f, f.output().vtree) {
        to_f[g.output().vtree.idx()][g.output().local.idx()] = Some(f.output().local.idx());
    }
    let structural = ChildDecoder::structural();
    let top_down: Vec<_> = f.vtree().internal_bottomup().collect::<Vec<_>>().into_iter().rev().collect();
    for (t, left, right) in top_down {
        if !renumbered(f, t) { continue; }
        for j in 0..gp[t.idx()].len() {
            let i = to_f[t.idx()][j].unwrap_or_else(|| {
                // A node no parent names: its pairs identify it among the
                // unnamed nodes, which keep their order.
                let named: Vec<usize> = to_f[t.idx()].iter().flatten().copied().collect();
                (0..fp[t.idx()].len()).find(|i| !named.contains(i)).expect("as many nodes")
            });
            to_f[t.idx()][j] = Some(i);
            let (got, want) = (&gp[t.idx()][j], &fp[t.idx()][i]);
            assert_eq!(got.len(), want.len(), "node {j} of level {t:?}");
            for (g_pair, f_pair) in got.iter().zip(want) {
                for (child, g_side, f_side) in [(left, g_pair.left, f_pair.left), (right, g_pair.right, f_pair.right)] {
                    if !renumbered(f, child) || f_side == ZERO.into() {
                        assert_eq!(g_side, f_side, "a side that keeps its slot");
                        continue;
                    }
                    let slot = &mut to_f[child.idx()][structural.node(g_side).idx()];
                    let old = structural.node(f_side).idx();
                    assert_eq!(*slot.get_or_insert(old), old, "one node of {child:?} renumbered two ways");
                }
            }
        }
    }
}

/// The walk the numbering follows: from the output, a level's nodes in order
/// and their pairs in order, names each child level's nodes in ascending
/// order the first time, and names none of the nodes it leaves to the end.
fn assert_walk_order(g: &Tdd) {
    if g.is_zero() { return; }
    let structural = ChildDecoder::structural();
    if renumbered(g, g.output().vtree) {
        assert_eq!(g.output().local, NodeIdx(0));
    }
    for (t, left, right) in g.vtree().internal_bottomup() {
        if !renumbered(g, t) { continue; }
        let level = g.level(t);
        for child in [left, right] {
            if !renumbered(g, child) { continue; }
            let mut next = 0usize;
            for i in 0..level.slot_count() {
                for pair in level.pairs_iter_of_idx(i) {
                    let side = if child == left { pair.left } else { pair.right };
                    if side == ZERO.into() { continue; }
                    let c = structural.node(side).idx();
                    assert!(c <= next, "level {child:?}: node {c} named before node {next}");
                    next = next.max(c + 1);
                }
            }
        }
    }
}

#[test]
fn random_diagrams_keep_their_function_nodes_and_pairs() {
    for f in random_diagrams(0x7e170b, 48, 3..10) {
        let g = f.renumber_top_down().unwrap();
        if f.levels.is_canonical(f.output()) {
            assert_canonical(&g);
            assert!(g.levels.is_canonical(g.output()));
        }
        assert_eq!((g.node_count(), g.pair_count()), (f.node_count(), f.pair_count()));
        assert_renumbering_of(&g, &f);
        assert_walk_order(&g);
    }
}

#[test]
fn renumbering_twice_changes_nothing() {
    for f in random_diagrams(0x27e17, 24, 4..10) {
        let g = f.renumber_top_down().unwrap();
        let h = g.renumber_top_down().unwrap();
        assert_eq!(pairs_by_node(&h), pairs_by_node(&g));
        assert_eq!(h.output(), g.output());
    }
}

#[test]
fn the_output_comes_first_and_unreached_nodes_last() {
    // A conjunction left unreduced keeps nodes the output does not reach.
    let vtree = Arc::new(Vtree::balanced(6));
    let eng = Engine::new();
    let a = compile_clauses_on(&eng, &vtree, &[vec![1, -4], vec![2, 5], vec![-3, 6]]);
    let b = compile_clauses_on(&eng, &vtree, &[vec![-1, 4], vec![-2, -6]]);
    let f = eng.and(a, b).unwrap();
    let g = eng.renumber_top_down(&f).unwrap();
    assert_renumbering_of(&g, &f);
    assert_walk_order(&g);
    let reached = g.reachable_nodes();
    for (t, _, _) in g.vtree().internal_bottomup() {
        let marks = &reached[t.idx()];
        // The reached nodes are the first ones.
        assert!(marks.windows(2).all(|w| w[0] || !w[1]), "level {t:?}: {marks:?}");
    }
}

#[test]
fn level_counts_follow_their_nodes() {
    for mut f in random_diagrams(0xc07e17, 16, 4..9) {
        if !f.levels.is_canonical(f.output()) { continue; }
        f.attach_level_counts().unwrap();
        let g = f.renumber_top_down().unwrap();
        assert!(g.has_level_counts());
        let mut fresh = f.renumber_top_down().unwrap();
        fresh.levels.forget();
        fresh.attach_level_counts().unwrap();
        for t in 0..g.levels.len() {
            let t = VtreeIdx(t as u32);
            assert_eq!(column(&g, t), column(&fresh, t), "level {t:?}");
        }
        assert_eq!(g.model_count().unwrap(), f.model_count().unwrap());
    }
}

#[test]
fn marginal_and_leaf_levels_keep_their_slots() {
    for (f, summed) in marginal_diagrams(0x3a7e17, 12, 4..9) {
        let g = f.renumber_top_down().unwrap();
        assert_marginal_canonical(&g);
        assert!(g.level(summed).is_marginal());
        assert_eq!(g.model_count().unwrap(), f.model_count().unwrap());
        for t in 0..f.levels.len() {
            let t = VtreeIdx(t as u32);
            if f.level(t).is_marginal() {
                assert_eq!(g.level(t).marginal_counts(), f.level(t).marginal_counts(), "level {t:?}");
            }
        }
        assert_walk_order(&g);
    }
}

#[test]
fn weighted_marginal_levels_keep_their_values() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let mut f = compile_clauses(&vtree, &[vec![1, 3, -6], vec![-2, 5, 7], vec![4, -8]]);
    let weights = vec![LiteralWeights { negative: rat(2, 1), positive: rat(3, 1) }; 8];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    let forgotten = vtree.children(vtree.root()).0;
    eng.marginalize_levels(&mut f, &[forgotten]).unwrap();
    assert_canonical(&f);
    let g = eng.renumber_top_down(&f).unwrap();
    assert_canonical(&g);
    assert!(g.weights().is_some_and(|ws| ws.compatible(f.weights().unwrap())));
    assert_eq!(
        exact_weight(&eng.weighted_value(&g).unwrap().unwrap()),
        exact_weight(&eng.weighted_value(&f).unwrap().unwrap()),
    );
}

#[test]
fn an_implicit_level_is_read_off_its_description() {
    let (f, v, _) = x_decision_diagram(64);
    assert!(f.level(v).implicit().is_some());
    let g = f.renumber_top_down().unwrap();
    // The walk meets the nodes in their stored order already.
    assert_eq!(pairs_by_node(&g), pairs_by_node(&f));
    assert!(g.level(v).implicit().is_some());
    assert_renumbering_of(&g, &f);
}

#[test]
fn false_is_copied() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1]).unwrap() & Tdd::clause(&vtree, [-1]).unwrap();
    f.minimize().unwrap();
    assert!(f.is_zero());
    assert_canonical(&f);
    let g = f.renumber_top_down().unwrap();
    assert!(g.is_zero());
    assert_canonical(&g);
}

#[test]
fn renumbering_honors_cancellation_memory_and_the_output_cap() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, 2, 3, 4]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    {
        let _guard = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(0)), ..StopRules::default()
        }));
        assert_eq!(eng.renumber_top_down(&f).err(), Some(OperationError::Stopped));
    }
    {
        let _guard = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(eng.renumber_top_down(&f).err(), Some(OperationError::OverBudget));
    }
    let _guard = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(0)));
    assert_eq!(eng.renumber_top_down(&f).err(), Some(OperationError::OutputCap));
}

#[test]
fn a_refused_allocation_leaves_the_operand_intact() {
    let vtree = Arc::new(Vtree::balanced(8));
    let mut f = compile_clauses(&vtree, &[vec![1, 2, -5], vec![-2, 3, 6], vec![4, -7, 8], vec![-1, 5, -8], vec![3, -4, 7]]);
    f.attach_level_counts().unwrap();
    let before = pairs_by_node(&f);
    let expected = Engine::new().renumber_top_down(&f).unwrap();
    let mut completed = false;
    for nth in 0..4096 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        let result = eng.renumber_top_down(&f);
        eng.limits().grant_every_reserve();
        assert_eq!(pairs_by_node(&f), before, "reservation {nth} changed the operand");
        match result {
            Ok(g) => {
                assert_eq!(pairs_by_node(&g), pairs_by_node(&expected));
                assert_canonical(&g);
                completed = true;
                break;
            }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed, "every reservation must be covered");
}
