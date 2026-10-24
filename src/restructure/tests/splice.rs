use std::sync::Arc;

use crate::diagram::{ChildPair, LeafLabel, NodeIdx, NodeKind, TddLevel, TddNodeId};
use crate::test_helpers::check::marginal::check_pair_fusion_saturation;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses_on, rand_cnf, CnfShape};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd, literal};

/// Over eight variables, the right operand's level over x5 x6 holds two
/// twins, x5 x6 and x5' x6', which its parent pairs with the same x7 x8; the
/// parent is on the operand's worklist, and no node is unreachable, so only
/// that worklist entry leads a reduction to them. The splice keeps it.
#[test]
fn a_splice_keeps_what_the_moved_levels_owe() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let root = vtree.root();
    let (l, r) = vtree.children(root);
    let (ll, lr) = vtree.children(l);
    let (rl, rr) = vtree.children(r);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);
    let pair = ChildPair::new;

    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let free_ll = levels[ll.idx()].push_internal_node(&[pair(one, one)]);
    let free_lr = levels[lr.idx()].push_internal_node(&[pair(one, one)]);
    let free = levels[l.idx()].push_internal_node(&[pair(free_ll, free_lr)]);
    let equal_true = levels[rl.idx()].push_internal_node(&[pair(pos, pos)]);
    let equal_false = levels[rl.idx()].push_internal_node(&[pair(neg, neg)]);
    let both = levels[rr.idx()].push_internal_node(&[pair(pos, pos)]);
    let twins = levels[r.idx()].push_internal_node(&[pair(equal_true, both), pair(equal_false, both)]);
    let top = levels[root.idx()].push_internal_node(&[pair(free, twins)]);
    let mut right = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: top });
    right.seed_contract_worklist([r.0]);

    let mut f = literal(&vtree, 1).unwrap();
    f.minimize().unwrap();
    // Safety: both operands are on `vtree` and unweighted, and each holds
    // one pair at the root whose other side is true.
    unsafe { f.splice_subtree_unchecked(&eng, right, root) };
    f.minimize().unwrap();
    assert_eq!(f.levels[rl.idx()].slot_count(), 1, "the twins merge");
    assert_canonical(&f);
    // x1, x5 = x6, x7 and x8, with x2, x3 and x4 free.
    assert_eq!(f.model_count().unwrap(), 16u32.into());
}

/// Random formulas over the root's two sides of an eight-variable balanced
/// vtree, each with a level on its own side marginalized and minimized, then
/// spliced at the root: the result keeps the left operand's pair fusion
/// records on the left side and the right operand's on the right, each of
/// those levels has no fusion left to find, and minimizing gives the diagram
/// minimizing with every record forgotten gives.
#[test]
fn a_splice_keeps_each_sides_fusion_records() {
    let mut rng = Lcg::new(43);
    let vtree = Arc::new(Vtree::balanced(8));
    let root = vtree.root();
    let (l, r) = vtree.children(root);
    let mut carried = 0;
    for round in 0..40 {
        let eng = Engine::new();
        let mut side = |half: VtreeIdx| {
            let vars: Vec<i32> = (1..=8u32)
                .filter(|&x| vtree.subtree(half).any(|t| Some(t) == vtree.leaf_of(VarId(x))))
                .map(|x| x as i32)
                .collect();
            let clauses: Vec<Vec<i32>> = rand_cnf(&mut rng, 4, CnfShape { clauses: 6, width: 3 })
                .into_iter()
                .map(|c| c.into_iter().map(|x| x.signum() * vars[x.unsigned_abs() as usize - 1]).collect())
                .collect();
            let mut f = compile_clauses_on(&eng, &vtree, &clauses);
            let (a, b) = vtree.children(half);
            let target = if rng.below(2) == 0 { a } else { b };
            eng.marginalize_levels(&mut f, &[target]).unwrap();
            eng.minimize(&mut f).unwrap();
            let sole = f.output().vtree == root
                && f.levels[root.idx()].slot_count() == 1
                && matches!(f.levels[root.idx()].node(0).kind(), NodeKind::Inline(_));
            sole.then_some(f)
        };
        let (Some(mut f), Some(g)) = (side(l), side(r)) else { continue };
        let records = |tdd: &Tdd, half: VtreeIdx| -> Vec<VtreeIdx> {
            vtree.subtree(half).filter(|t| tdd.dirty.is_fused(t.0)).collect()
        };
        let (left, right) = (records(&f, l), records(&g, r));
        // Safety: both operands are on `vtree` and unweighted, and each holds
        // one pair at the root whose other side is true.
        unsafe { f.splice_subtree_unchecked(&eng, g, root) };
        let label = format!("round {round}");
        for t in left.iter().chain(right.iter()) {
            assert!(f.dirty.is_fused(t.0), "{label}: level {} lost its record", t.idx());
        }
        carried += left.len() + right.len();
        let now: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|t| f.dirty.is_fused(t.0)).collect();
        if let Err(e) = check_pair_fusion_saturation(&f, Some(&now)) {
            panic!("{label}: {e}");
        }
        let mut forced = f.clone();
        forced.dirty.forget_all_fused();
        eng.minimize(&mut f).unwrap();
        eng.minimize(&mut forced).unwrap();
        assert_canonical(&f);
        assert_same_shape(&f, &forced, &label);
    }
    assert!(carried > 0, "no splice carried a record");
}
