use super::*;
use crate::vtree::Vtree;

/// Restriction compacts each node's pair list inside the arena range that
/// node already owns — the level is never rebuilt into a second arena.
/// Fixture, conditioning the LEFT leaf to ⊤ (Pos pairs survive and lose
/// their label, Neg pairs drop, One passes through):
///
/// - node0, multi `[(Pos,One), (Neg,Pos), (One,Neg)]` → `[(One,One), (One,Neg)]`
///   — the third pair moves down over an already-read slot, one slot is abandoned
/// - node1, inline `(Pos,One)` → inline `(One,One)`
/// - node2, inline `(Neg,One)` → emptied to the zero-pair placeholder
///
/// The arena length is what pins "in place": the former rebuild replaced the
/// arena with a freshly pushed one holding only the two live pairs, whereas
/// the shrink leaves the abandoned slot behind and reports it dead.
#[test]
fn rewrite_for_restrict_shrinks_pair_lists_in_place() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut tdd = crate::build::constant_one(eng, &vtree);
    let level = &mut tdd.levels[root.idx()];
    level.clear();
    level.push_internal_node(&[
        ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX),
        ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX),
        ChildPair::new(ONE_LEAF_IDX, NEG_LEAF_IDX),
    ]);
    level.push_internal_node(&[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]);
    level.push_internal_node(&[ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)]);
    let arena_len_before = level.pairs.len();

    rewrite_for_restrict(&mut tdd, root, ChildSide::Left, Polarity::Positive);

    let level = &tdd.levels[root.idx()];
    assert_eq!(level.nodes.len(), 3, "node indices are preserved");
    assert_eq!(
        level.pairs_of_idx(0),
        &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(ONE_LEAF_IDX, NEG_LEAF_IDX)],
        "survivors compacted into the node's own range, sorted",
    );
    assert_eq!(
        level.pairs_of_idx(1),
        &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)],
        "the inline node's restricted pair stays inline",
    );
    assert!(level.pairs_of_idx(2).is_empty(), "an all-dropped node keeps no pairs");
    assert_eq!(level.pairs.len(), arena_len_before, "no second arena, and no growth");
    assert_eq!(level.dead_pairs, 1, "the one abandoned slot is reported as dead");
}

/// Conditioning empties every node whose pairs all belonged to the opposite
/// cofactor, and nothing above it may go on naming those nodes (invariant 2).
///
/// `x1 ∨ x2` over a balanced vtree of three variables puts the clause's two
/// disjuncts in a level under the root. Conditioning x2 to ⊥ drops every pair
/// of that level's node that named x2 = ⊤, emptying it — and the root's pairs
/// still pointed at it until the falsity sweep ran.
#[test]
fn conditioning_leaves_no_node_computing_false() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    let mut c = (f).clone().condition_var(crate::vtree::VarId(2), false).unwrap();
    c.minimize().unwrap();
    assert_eq!(c.model_count().unwrap(), num_bigint::BigUint::from(4u32), "the cofactor's count is unaffected");
    crate::test_helpers::check::check_no_false_nodes(&c).expect("no node computes false");
    crate::test_helpers::check_minimize_soundness(&mut c, 1).expect("every stored node is reachable and reduced");
}

/// A variable the vtree does not carry is the caller's input, so the engine
/// forms name it in an error rather than aborting the process.
#[test]
fn conditioning_a_variable_outside_the_vtree_is_an_error() {
    use crate::vtree::VarId;
    use crate::OperationError;
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert!(matches!(
        eng.condition_var(f.clone(), VarId(10), true),
        Err(OperationError::VariableNotInVtree(VarId(10))),
    ));
    assert!(matches!(
        eng.condition_vars(f, &[VarId(1), VarId(10)], true),
        Err(OperationError::VariableNotInVtree(VarId(10))),
    ));
    assert!(matches!(
        eng.condition_var(Tdd::zero(&vtree), VarId(10), true),
        Err(OperationError::VariableNotInVtree(VarId(10))),
    ));
}

#[test]
fn a_false_cofactor_leaves_no_empty_internal_node() {
    let tree = std::sync::Arc::new(crate::vtree::Vtree::balanced(2));
    let f = crate::Tdd::clause(&tree, [1]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    let result = crate::Engine::new().condition_var(f, crate::vtree::VarId(1), false).unwrap();
    assert!(result.is_zero());
    crate::test_helpers::assert_canonical(&result);
}
