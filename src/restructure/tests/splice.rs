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

/// The home ((x1 x2) (x3 x4)), the root `t` of its tree.
fn home_of_two_pairs() -> crate::vtree::Home {
    let leaf = |v| Vtree::leaf(VarId(v));
    let pair = |a, b| Vtree::join(&leaf(a), &leaf(b)).unwrap();
    crate::vtree::Home::new(Vtree::join(&pair(1, 2), &pair(3, 4)).unwrap())
}

/// On a home, operands padded onto the home's tree from the projections of
/// the two sides of the root, spliced at the root, give the diagram on the
/// leaves it uses that `and` gives.
#[test]
fn a_splice_at_the_root_of_a_home_reads_every_leaf_and_names_none_true() {
    use crate::test_helpers::check::check_trimmed;
    let eng = Engine::new();
    let home = home_of_two_pairs();
    let all = Arc::clone(home.vtree());
    let f = eng.xor(eng.literal(&all, 1).unwrap(), eng.literal(&all, 2).unwrap()).unwrap();
    let g = eng.or(eng.literal(&all, 3).unwrap(), eng.literal(&all, -4).unwrap()).unwrap();
    assert_eq!(f.vtree().num_leaves(), 2);
    let want = eng.and(f.clone(), g.clone()).unwrap();
    let mut left = eng.embed(&f, &all, |v| v).unwrap().0;
    let right = eng.embed(&g, &all, |v| v).unwrap().0;
    // Safety: both on `all` and unweighted, each one pair at the root whose
    // other side is true.
    unsafe { left.splice_subtree_unchecked(&eng, right, all.root()) };
    check_trimmed(&left, false).unwrap();
    assert!(Arc::ptr_eq(left.vtree(), want.vtree()));
    assert!(eng.equivalent(&left, &want).unwrap());
}

/// Below the root the result keeps the left operand's pass-through level
/// above `t`, which names x3 true: the splice of x1 and x2 at (x1 x2) on the
/// home ((x1 x2) x3) is x1 and x2 on all three leaves, which `minimize`
/// keeps and `trim` drops.
#[test]
fn a_splice_below_the_root_of_a_home_keeps_the_padding_above() {
    use crate::test_helpers::check::{check_trimmed, reachable_true_labels};
    let eng = Engine::new();
    let leaf = |v| Vtree::leaf(VarId(v));
    let home = crate::vtree::Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap());
    let all = Arc::clone(home.vtree());
    let t = all.children(all.root()).0;
    let mut left = eng.embed(&eng.literal(&all, 1).unwrap(), &all, |v| v).unwrap().0;
    let right = eng.embed(&eng.literal(&all, 2).unwrap(), &all, |v| v).unwrap().0;
    // Safety: both on `all` and unweighted, each one pair at `t` whose other
    // side is true, and the root level above names `t`'s one node.
    unsafe { left.splice_subtree_unchecked(&eng, right, t) };
    assert!(check_trimmed(&left, false).is_err());
    assert!(matches!(reachable_true_labels(&left), Ok(Some(_))));
    assert!(eng.equivalent(&left, &eng.cube(&all, [1, 2]).unwrap()).unwrap());
    let mut closed = left;
    closed.minimize().unwrap();
    assert!(check_trimmed(&closed, false).is_err(), "no reduce drops a leaf named true in every pair");
    closed.trim(&home).unwrap();
    check_trimmed(&closed, true).unwrap();
    assert_eq!(closed.vtree().num_leaves(), 2);
}
