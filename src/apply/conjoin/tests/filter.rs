use super::*;
use crate::test_helpers::assert_canonical;

#[test]
fn product_filter_retains_conjunction_or_removes_root() {
    let engine = Engine::new();
    let tree = Arc::new(crate::Vtree::balanced(6));
    let f = engine.clause(&tree, [1, 3, 5]).unwrap();
    let g = engine.clause(&tree, [-2, 4, -6]).unwrap();
    assert_canonical(&f); assert_canonical(&g);
    let expected = engine.and(f.clone(), g.clone()).unwrap();
    let mut actual = engine.and_filter_products(f.clone(), g.clone(), |_,_,_| true).unwrap();
    engine.minimize(&mut actual).unwrap(); assert_canonical(&actual);
    assert!(engine.equivalent(&actual, &expected).unwrap());
    let mut zero = engine.and_filter_products(f, g, |t,_,_| t != tree.root()).unwrap();
    engine.minimize(&mut zero).unwrap(); assert_canonical(&zero);
    assert!(zero.is_zero());
}

#[test]
fn product_filter_is_monotone_through_identity_and_self_conjunctions() {
    let engine = Engine::new();
    let tree = Arc::new(crate::Vtree::balanced(6));
    let f = engine.clause(&tree, [1, 2, 3, 4, 5, 6]).unwrap();
    let one = engine.cube(&tree, [] as [i32; 0]).unwrap();
    assert_canonical(&f); assert_canonical(&one);
    for g in [one, f.clone()] {
        for divisor in 2..5 {
            let mut filtered = engine.and_filter_products(f.clone(), g.clone(), |t,a,b| {
                (t.idx() + a.idx() + b.idx()) % divisor != 0
            }).unwrap();
            engine.minimize(&mut filtered).unwrap(); assert_canonical(&filtered);
            let intersection = engine.and(filtered.clone(), f.clone()).unwrap();
            assert!(engine.equivalent(&filtered, &intersection).unwrap());
        }
    }
}

fn filtered_truth(f: &Tdd, g: &Tdd, t: VtreeIdx, a: NodeIdx, b: NodeIdx, assignment: u32, divisor: usize) -> bool {
    if a == ZERO || b == ZERO { return false; }
    match f.vtree().node(t) {
        crate::vtree::VtreeNode::Leaf { var, .. } => {
            let positive = assignment & (1 << (var.0 - 1)) != 0;
            let value = |i| i == ONE_LEAF_IDX || (positive == (i == POS_LEAF_IDX));
            value(a) && value(b)
        }
        crate::vtree::VtreeNode::Internal { left, right, .. } => {
            if (t.idx() + a.idx() + b.idx()).is_multiple_of(divisor) { return false; }
            let af = f.level(t); let bg = g.level(t);
            af.pairs_of_idx(a.idx()).iter().any(|ap| bg.pairs_of_idx(b.idx()).iter().any(|bp| {
                let (ChildRef::Node(al), ChildRef::Node(ar)) = (af.child_decoder().child(ap.left), af.child_decoder().child(ap.right)) else { panic!("structural fixture"); };
                let (ChildRef::Node(bl), ChildRef::Node(br)) = (bg.child_decoder().child(bp.left), bg.child_decoder().child(bp.right)) else { panic!("structural fixture"); };
                filtered_truth(f,g,*left,al,bl,assignment,divisor) && filtered_truth(f,g,*right,ar,br,assignment,divisor)
            }))
        }
    }
}

#[test]
fn product_filter_matches_recursive_product_semantics() {
    let engine = Engine::new();
    let tree = Arc::new(crate::Vtree::balanced(4));
    for a in [[1,2],[-1,3],[2,4],[-3,-4]] {
        for b in [[1,-2],[-1,3],[2,4],[-2,-4]] {
            let f = engine.clause(&tree,a).unwrap();
            let g = engine.clause(&tree,b).unwrap();
            assert_canonical(&f); assert_canonical(&g);
            for divisor in 2..6 {
                let mut actual = engine.and_filter_products(f.clone(),g.clone(),|t,a,b| (t.idx()+a.idx()+b.idx())%divisor!=0).unwrap();
                engine.minimize(&mut actual).unwrap(); assert_canonical(&actual);
                for assignment in 0u32..16 {
                    let expected = filtered_truth(&f,&g,tree.root(),f.output().local,g.output().local,assignment,divisor);
                    let lits = (1..=4).map(|v| if assignment & (1 << (v-1)) != 0 {v} else {-v});
                    let cube = engine.cube(&tree,lits).unwrap(); assert_canonical(&cube);
                    let restricted = engine.and(actual.clone(),cube).unwrap();
                    assert_eq!(!restricted.is_zero(),expected,"{a:?} {b:?} {divisor} {assignment}");
                }
            }
        }
    }
}

#[test]
fn product_filter_honors_cancellation_and_memory_refusal() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let tree = Arc::new(crate::Vtree::balanced(4));
    let f = Tdd::clause(&tree,[1,2]).unwrap();
    let g = Tdd::clause(&tree,[3,4]).unwrap();
    assert_canonical(&f); assert_canonical(&g);
    let engine = Engine::new();
    {
        let _guard = engine.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(0)), ..StopRules::default()
        }));
        assert_eq!(engine.and_filter_products(f.clone(),g.clone(),|_,_,_| panic!("already canceled")).err(),Some(OperationError::Stopped));
    }
    let _guard = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(engine.and_filter_products(f,g,|_,_,_|true).err(),Some(OperationError::OverBudget));
}
