use std::sync::Arc;

use super::*;
use crate::diagram::{EncodedNode, NEG_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::*;
use crate::vtree::{Vtree, VtreeIdx};

mod builder_sweep;
mod marginal;

fn node(vtree: VtreeIdx, local: u32) -> TddNodeId {
    TddNodeId { vtree, local: NodeIdx(local) }
}

/// The diagram of a filtered outcome, minimized and checked.
fn filtered(outcome: FilterOutcome) -> (Tdd, FilterStats) {
    let FilterOutcome::Filtered { mut tdd, stats } = outcome else { panic!("expected a filtered diagram, got {outcome:?}") };
    tdd.minimize().unwrap();
    assert_canonical(&tdd);
    (tdd, stats)
}

/// What a reader can observe of each level: its nodes, each node's pairs, its
/// counts and its inlined-side markers.
type Storage = Vec<(Vec<EncodedNode>, Vec<Vec<ChildPair>>, Option<Vec<u128>>, u8)>;

fn storage(f: &Tdd) -> Storage {
    f.levels.iter().map(|level| (
        level.nodes().to_vec(),
        (0..level.nodes().len()).map(|i| level.pairs_of_idx(i).to_vec()).collect(),
        level.marginal_counts().map(<[u128]>::to_vec),
        level.value_ref_sides,
    )).collect()
}

#[test]
fn keep_is_asked_about_every_stored_node_bottom_up() {
    let (f, [b, a, root]) = chain();
    let order: Vec<VtreeIdx> = f.vtree().internal_bottomup().map(|(t, _, _)| t).collect();
    assert_eq!(order, [b, a, root]);
    let mut asked = Vec::new();
    let outcome = f.filter_nodes(|id| { asked.push(id); true }).unwrap();
    assert!(matches!(outcome, FilterOutcome::Unchanged));
    assert_eq!(outcome.stats(), FilterStats::default());
    let expected: Vec<TddNodeId> = (0..4).map(|i| node(b, i)).chain((0..3).map(|i| node(a, i))).chain([node(root, 0)]).collect();
    assert_eq!(asked, expected);
}

#[test]
fn a_false_operand_is_unchanged_without_a_callback() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::zero(&tree);
    assert_canonical(&f);
    assert!(matches!(f.filter_nodes(|_| panic!("a false operand has no node to ask about")).unwrap(), FilterOutcome::Unchanged));
}

#[test]
fn the_counts_cover_every_kept_node() {
    let (f, [b, _, _]) = chain();
    // Rejecting x3 x4, x3 ¬x4 and the unreachable ¬x3 ¬x4 empties both nodes
    // at the middle level that use them, and the output loses its x1 pair.
    let (g, stats) = filtered(f.filter_nodes(|id| id.vtree != b || id.local.0 == 2).unwrap());
    assert_eq!(stats, FilterStats { pairs_dropped: 4, emptied_nodes: 2 });
    assert!(g.equivalent(&Tdd::cube(f.vtree(), [-1, -2, -3, 4]).unwrap()).unwrap());
    // Rejecting only an unreachable node still rebuilds, and empties the
    // unreachable node above it.
    let (g, stats) = filtered(f.filter_nodes(|id| id != node(b, 3)).unwrap());
    assert_eq!(stats, FilterStats { pairs_dropped: 1, emptied_nodes: 1 });
    let mut reduced = f.clone();
    reduced.minimize().unwrap();
    assert_canonical(&reduced);
    assert_same_shape(&g, &reduced, "rejecting an unreachable node");
}

#[test]
fn an_emptied_or_rejected_output_is_unsatisfiable() {
    let (f, [b, _, root]) = chain();
    let outcome = f.filter_nodes(|id| id != node(root, 0)).unwrap();
    assert_eq!(outcome.stats(), FilterStats::default());
    let FilterOutcome::Unsatisfiable { tdd, .. } = outcome else { panic!("the output was rejected") };
    assert!(tdd.is_zero());
    assert_canonical(&tdd);
    let outcome = f.filter_nodes(|id| id.vtree != b).unwrap();
    assert_eq!(outcome.stats(), FilterStats { pairs_dropped: 6, emptied_nodes: 4 });
    let FilterOutcome::Unsatisfiable { tdd, .. } = outcome else { panic!("every path was removed") };
    assert!(tdd.is_zero());
}

#[test]
fn survivors_keep_their_order_and_their_pairs_order() {
    let (f, [b, a, root]) = chain();
    let eng = Engine::new();
    let mut remap = ask(&eng, &f, |id| id != node(b, 1)).unwrap().expect("a node was rejected");
    let (assembly, stats) = sweep(&eng, &f, &mut remap).unwrap();
    assert_eq!(stats, FilterStats { pairs_dropped: 1, emptied_nodes: 0 });
    assert_eq!(remap[b.idx()], [0, DEAD, 1, 2]);
    assert_eq!(remap[a.idx()], [0, 1, 2]);
    assert_eq!(remap[root.idx()], [0]);
    let pairs = |t: VtreeIdx, i: usize| assembly.level(t).pairs_of_idx(i).to_vec();
    assert_eq!(pairs(b, 1), [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)]);
    assert_eq!(pairs(a, 0), [ChildPair::new(POS_LEAF_IDX, NodeIdx(0))]);
    assert_eq!(pairs(a, 1), [ChildPair::new(NEG_LEAF_IDX, NodeIdx(1))]);
    assert_eq!(pairs(a, 2), [ChildPair::new(POS_LEAF_IDX, NodeIdx(2))]);
    assert_eq!(pairs(root, 0), [ChildPair::new(POS_LEAF_IDX, NodeIdx(0)), ChildPair::new(NEG_LEAF_IDX, NodeIdx(1))]);
}

/// A seeded predicate rejecting about one node in `every`.
fn sparse_rejection(seed: u64, every: u64) -> impl Fn(TddNodeId) -> bool {
    move |id| !(u64::from(id.vtree.0).wrapping_mul(0x9e37_79b9) ^ u64::from(id.local.0).wrapping_mul(0x85eb_ca6b) ^ seed).is_multiple_of(every)
}

/// [`sparse_rejection`] keeping the output node of `f`, so that only a
/// cascade can empty it.
fn sparing_output(f: &Tdd, seed: u64, every: u64) -> impl Fn(TddNodeId) -> bool + use<> {
    let (output, reject) = (f.output(), sparse_rejection(seed, every));
    move |id| id == output || reject(id)
}

#[test]
fn renumbering_is_monotone_and_pair_lists_keep_their_order() {
    let marginal = marginal_diagrams(0x51f8, 20, 8..12).into_iter().map(|(f, _)| f);
    for (k, f) in random_diagrams(0x51f7, 40, 8..13).into_iter().chain(marginal).enumerate() {
        let f = &f;
        let eng = Engine::new();
        let Some(mut remap) = ask(&eng, f, sparse_rejection(k as u64, 3)).unwrap() else { continue };
        let (assembly, _) = sweep(&eng, f, &mut remap).unwrap();
        for (t, left, right) in f.vtree.internal_bottomup() {
            if f.levels[t.idx()].is_marginal() { continue; }
            let kept: Vec<u32> = remap[t.idx()].iter().copied().filter(|&r| r != DEAD).collect();
            assert_eq!(kept, (0..kept.len() as u32).collect::<Vec<_>>(), "level {t:?} renumbered out of order");
            let back = |child: VtreeIdx, side: EncodedChildRef| -> EncodedChildRef {
                if f.vtree.node(child).is_leaf() || f.levels[child.idx()].is_marginal() { return side; }
                let i = remap[child.idx()].iter().position(|&n| n == side.0).expect("a remapped child survives");
                NodeIdx(i as u32).into()
            };
            for (i, &new) in remap[t.idx()].iter().enumerate() {
                if new == DEAD { continue; }
                let mut source = f.levels[t.idx()].pairs_of_idx(i).iter();
                for pair in assembly.level(t).pairs_of_idx(new as usize) {
                    let original = ChildPair::new(back(left, pair.left), back(right, pair.right));
                    assert!(source.any(|&p| p == original), "node {i} at {t:?} reordered or invented a pair");
                }
            }
        }
    }
}

#[test]
fn filtering_nodes_only_removes_models_and_pairs() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(6));
    let f = eng.clause(&tree, [1, 2, 3, 4, 5, 6]).unwrap();
    assert_canonical(&f);
    for divisor in 2..7 {
        let (FilterOutcome::Filtered { tdd: mut g, .. } | FilterOutcome::Unsatisfiable { tdd: mut g, .. }) =
            eng.filter_nodes(&f, |id| (id.vtree.idx() + id.local.idx()) % divisor != 0).unwrap()
        else { panic!("a node was rejected") };
        eng.minimize(&mut g).unwrap();
        assert_canonical(&g);
        assert!(g.pair_count() <= f.pair_count());
        let conjunction = eng.and(g.clone(), f.clone()).unwrap();
        assert!(eng.equivalent(&g, &conjunction).unwrap());
    }
}

#[test]
fn node_filter_honors_cancellation_before_the_callback() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    let _guard = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(0)), ..StopRules::default()
    }));
    assert_eq!(eng.filter_nodes(&f, |_| panic!("canceled before callback")).err(), Some(OperationError::Stopped));
}

#[test]
fn filter_respects_memory_refusal_and_the_output_cap() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, 2, 3, 4]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    {
        let _guard = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(eng.filter_nodes(&f, |_| panic!("refused before callback")).err(), Some(OperationError::OverBudget));
    }
    let _guard = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(0)));
    let right = tree.children(tree.root()).1;
    assert_eq!(eng.filter_nodes(&f, |id| id.vtree != right || id.local.0 != 0).err(), Some(OperationError::OutputCap));
}

#[test]
fn a_refused_allocation_leaves_the_operand_intact() {
    let tree = Arc::new(Vtree::balanced(8));
    let f = compile_clauses(&tree, &[vec![1, 2, -5], vec![-2, 3, 6], vec![4, -7, 8], vec![-1, 5, -8], vec![3, -4, 7]]);
    assert_canonical(&f);
    let (before, output) = (storage(&f), f.output());
    let keep = sparse_rejection(7, 4);
    let (expected, expected_stats) = filtered(Engine::new().filter_nodes_with(&f, &keep, ReductionPlan::default()).unwrap());
    let mut completed = false;
    for nth in 0..4096 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        let result = eng.filter_nodes_with(&f, &keep, ReductionPlan::default());
        eng.limits().grant_every_reserve();
        assert_eq!(storage(&f), before, "reservation {nth} changed the operand");
        assert_eq!(f.output(), output);
        match result {
            Ok(outcome) => {
                let (g, stats) = filtered(outcome);
                assert_eq!(stats, expected_stats);
                assert_same_shape(&g, &expected, "after refusals");
                completed = true;
                break;
            }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed, "every reservation must be covered");
}

#[test]
fn weighted_operands_keep_their_weights_on_every_outcome() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(8));
    let mut f = Tdd::clause(&tree, [1, 3, 5, 7]).unwrap();
    let weights = vec![LiteralWeights { negative: rat(2, 1), positive: rat(3, 1) }; 8];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    let forgotten = tree.children(tree.root()).0;
    eng.marginalize_levels(&mut f, &[forgotten]).unwrap();
    assert_canonical(&f);
    let root = f.output();
    let mut rejected = None;
    let outcome = eng.filter_nodes(&f, |id| {
        assert!(!f.level(id.vtree).is_marginal());
        if id != root && rejected.is_none() { rejected = Some(id); false } else { true }
    }).unwrap();
    assert!(rejected.is_some());
    let FilterOutcome::Filtered { tdd: mut g, .. } = outcome else { panic!("the output was kept") };
    // The copied weighted marginal columns evaluate the same before and after
    // minimization.
    let before = eng.weighted_value(&g).unwrap().unwrap();
    eng.minimize(&mut g).unwrap();
    assert_canonical(&g);
    assert!(g.pair_count() < f.pair_count());
    assert_eq!(exact_weight(&before), exact_weight(&eng.weighted_value(&g).unwrap().unwrap()));
    let FilterOutcome::Unsatisfiable { tdd: z, .. } = eng.filter_nodes(&f, |id| id != root).unwrap() else { panic!("the output was rejected") };
    assert!(z.is_zero());
    assert!(z.weights().is_some_and(|ws| ws.compatible(f.weights().unwrap())));
    assert!(eng.weighted_value(&f).unwrap().unwrap().as_log().is_none());
}
