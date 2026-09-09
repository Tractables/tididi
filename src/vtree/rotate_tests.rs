use super::*;

/// Verify parent pointers are consistent and topo order is a valid
/// bottom-up linearization (children-before-parents) AND satisfies the
/// root-last property used by `Vtree::fixup_topo_after_rotate`.
fn assert_invariants(vtree: &Vtree) {
    // topo / topo_pos / internal_topo / leaf_topo completeness:
    // every node appears exactly once across topo, topo_pos is its inverse,
    // and internal_topo ∪ leaf_topo partitions the node set.
    let n = vtree.num_nodes();
    assert_eq!(vtree.bottomup_slice().len(), n, "topo length mismatch");
    let mut seen = vec![false; n];
    for &t in vtree.bottomup_slice() {
        assert!(!seen[t.idx()], "duplicate {t:?} in topo");
        seen[t.idx()] = true;
    }
    assert!(seen.iter().all(|&b| b), "topo missing a node");
    let mut filt_seen = vec![false; n];
    for (t, _, _) in vtree.internal_bottomup() {
        assert!(!filt_seen[t.idx()], "duplicate {t:?} in internal_topo");
        filt_seen[t.idx()] = true;
    }
    for (t, _) in vtree.leaf_bottomup() {
        assert!(!filt_seen[t.idx()], "duplicate {t:?} in leaf_topo");
        filt_seen[t.idx()] = true;
    }
    assert!(filt_seen.iter().all(|&b| b), "internal_topo + leaf_topo missing a node");

    for (t, left, right) in vtree.internal_bottomup() {
        assert_eq!(vtree.node(left).parent(), Some(t));
        assert_eq!(vtree.node(right).parent(), Some(t));
        assert!(
            vtree.topo_pos(left) < vtree.topo_pos(t),
            "left child {left:?} not before parent {t:?} in topo"
        );
        assert!(
            vtree.topo_pos(right) < vtree.topo_pos(t),
            "right child {right:?} not before parent {t:?} in topo"
        );
    }
    // Root-last: every node's topo position is the max over its subtree.
    // Iterating bottom-up, each internal node's subtree max equals the
    // larger of its two children's subtree maxes plus itself; we walk
    // explicitly here rather than caching to keep the assertion
    // self-contained.
    let n = vtree.num_nodes();
    let mut subtree_max: Vec<u32> = (0..n as u32).map(|_| 0).collect();
    for &t in vtree.bottomup_slice() {
        let pos = vtree.topo_pos(t);
        let m = match vtree.node(t) {
            VtreeNode::Leaf { .. } => pos,
            VtreeNode::Internal { left, right, .. } => {
                pos.max(subtree_max[left.idx()]).max(subtree_max[right.idx()])
            }
        };
        subtree_max[t.idx()] = m;
        assert_eq!(
            m, pos,
            "root-last violated at {t:?}: topo_pos={pos} but subtree max={m}",
        );
    }
}

fn assert_equal(a: &Vtree, b: &Vtree) {
    assert_eq!(a.num_nodes(), b.num_nodes());
    for i in 0..a.num_nodes() {
        let idx = VtreeIdx(i as u32);
        match (a.node(idx), b.node(idx)) {
            (VtreeNode::Leaf { var: v1, parent: p1 }, VtreeNode::Leaf { var: v2, parent: p2 }) => {
                assert_eq!(v1, v2); assert_eq!(p1, p2);
            }
            (VtreeNode::Internal { left: l1, right: r1, parent: p1 },
             VtreeNode::Internal { left: l2, right: r2, parent: p2 }) => {
                assert_eq!(l1, l2); assert_eq!(r1, r2); assert_eq!(p1, p2);
            }
            _ => panic!("node type mismatch at {idx:?}"),
        }
    }
}

#[test]
fn left_rotate_then_unrotate_is_identity() {
    let mut vtree = Vtree::balanced(4);
    let original = vtree.clone();
    let root = vtree.root;
    let info = rotate_left(&mut vtree, root).unwrap();
    assert_invariants(&vtree);
    unrotate_left(&mut vtree, &info);
    assert_equal(&vtree, &original);
}

#[test]
fn left_rotate_at_leaf_or_leaf_child_returns_none() {
    let mut vtree = Vtree::balanced(4);
    // Leaf 0 -> None
    assert!(rotate_left(&mut vtree, VtreeIdx(0)).is_none());
    // Internal node whose right child is a leaf:
    // balanced(4): 6=(4,5), 4=(0,1), 5=(2,3). At node 4, right=1 is a leaf.
    assert!(rotate_left(&mut vtree, VtreeIdx(4)).is_none());
}

#[test]
fn right_rotate_after_left_recovers_original() {
    let mut vtree = Vtree::linear(4);
    let original = vtree.clone();
    let root = vtree.root;
    let _info = rotate_left(&mut vtree, root).unwrap();
    assert_invariants(&vtree);
    let _info_right = rotate_right(&mut vtree, root).unwrap();
    assert_invariants(&vtree);
    assert_equal(&vtree, &original);
}

#[test]
fn right_rotate_then_unrotate_is_identity() {
    let mut vtree = Vtree::linear(4);
    let root = vtree.root;
    let _ = rotate_left(&mut vtree, root).unwrap();
    let snap = vtree.clone();
    let info = rotate_right(&mut vtree, root).unwrap();
    assert_invariants(&vtree);
    unrotate_right(&mut vtree, &info);
    assert_equal(&vtree, &snap);
}

#[test]
fn right_rotate_works_immediately_on_balanced() {
    // Pre-refactor this returned None (would have violated child<parent
    // invariant); now it must succeed.
    let mut vtree = Vtree::balanced(4);
    let root = vtree.root;
    let info = rotate_right(&mut vtree, root);
    assert!(info.is_some(), "right rotation should be unconditionally applicable");
    assert_invariants(&vtree);
}

/// Apply a rotation two ways — pointer-only + the localized commit vs
/// pointer-only + a full order rebuild — and check that both produce
/// vtrees satisfying the topo invariants. The two `topo` arrays may
/// differ (fixup doesn't promise strict postorder), but each must be a
/// valid bottom-up order satisfying the root-last property.
fn check_fixup_equivalence_left(mut vtree: Vtree, v: VtreeIdx) {
    let mut via_rebuild = vtree.clone();
    let info_a = rotate_left_pointers(&mut vtree, v).expect("applicable").commit(&mut vtree);
    assert_invariants(&vtree);

    let info_b = rotate_left_pointers(&mut via_rebuild, v).expect("applicable").abandon();
    via_rebuild.rebuild_topo();
    assert_invariants(&via_rebuild);
    assert_eq!(info_a.v_idx, info_b.v_idx);
    assert_eq!(info_a.w_idx, info_b.w_idx);
}

fn check_fixup_equivalence_right(mut vtree: Vtree, v: VtreeIdx) {
    let mut via_rebuild = vtree.clone();
    let info_a = rotate_right_pointers(&mut vtree, v).expect("applicable").commit(&mut vtree);
    assert_invariants(&vtree);

    let info_b = rotate_right_pointers(&mut via_rebuild, v).expect("applicable").abandon();
    via_rebuild.rebuild_topo();
    assert_invariants(&via_rebuild);
    assert_eq!(info_a.v_idx, info_b.v_idx);
    assert_eq!(info_a.w_idx, info_b.w_idx);
}

#[test]
fn fixup_equivalence_left_at_root() {
    check_fixup_equivalence_left(Vtree::linear(5), Vtree::linear(5).root);
}

#[test]
fn fixup_equivalence_right_at_root_balanced() {
    // balanced(4) is the case the early-exit cannot use (right rotation
    // pre-fixup has C after w in topo) — exercises the slice-rotate path.
    check_fixup_equivalence_right(Vtree::balanced(4), Vtree::balanced(4).root);
}

#[test]
fn fixup_equivalence_right_at_root_balanced_8() {
    check_fixup_equivalence_right(Vtree::balanced(8), Vtree::balanced(8).root);
}

#[test]
fn fixup_equivalence_at_internal_non_root() {
    // balanced(8): root is the topmost internal node. Pick an internal
    // child to exercise rotation at a non-root node.
    let vtree = Vtree::balanced(8);
    // Find an internal node that itself has two internal children, so
    // both directions are applicable.
    let target = (0..vtree.num_nodes() as u32)
        .map(VtreeIdx)
        .find(|&t| {
            if let VtreeNode::Internal { left, right, parent } = *vtree.node(t) {
                parent.is_some()
                    && !vtree.node(left).is_leaf()
                    && !vtree.node(right).is_leaf()
            }
            else { false }
        })
        .expect("balanced(8) has at least one such node");
    check_fixup_equivalence_left(vtree.clone(), target);
    check_fixup_equivalence_right(vtree, target);
}

/// Round-trip: a sequence of rotations followed by their inverses (via
/// `unrotate_*`) restores the structure bit-for-bit and produces a
/// vtree that still satisfies the topo invariants at every step.
#[test]
fn fixup_round_trip_random_sequence() {
    let mut vtree = Vtree::balanced(8);
    let original = vtree.clone();
    let mut history: Vec<(RotationInfo, RotKindLocal)> = Vec::new();

    // Deterministic pseudo-random walk: try a left rotation at every
    // internal node bottom-up, then a right rotation, recording each
    // applicable success.
    let internal: Vec<VtreeIdx> = vtree.bottomup_slice()
        .iter()
        .copied()
        .filter(|&t| !vtree.node(t).is_leaf())
        .collect();
    for &t in &internal {
        if let Some(info) = rotate_left(&mut vtree, t) {
            assert_invariants(&vtree);
            history.push((info, RotKindLocal::Left));
        }
        if let Some(info) = rotate_right(&mut vtree, t) {
            assert_invariants(&vtree);
            history.push((info, RotKindLocal::Right));
        }
    }
    // Undo in reverse.
    while let Some((info, kind)) = history.pop() {
        match kind {
            RotKindLocal::Left => unrotate_left(&mut vtree, &info),
            RotKindLocal::Right => unrotate_right(&mut vtree, &info),
        }
        assert_invariants(&vtree);
    }
    assert_equal(&vtree, &original);
}

/// Edge case: rotation where the misplaced subtree's root is a leaf.
/// This is the smallest possible misplaced subtree. The slice has at
/// most two elements and `rotate_left(1)` is a single swap.
#[test]
fn fixup_handles_single_node_misplaced_subtree() {
    // linear(4) is right-linear: root = (leaf_a, internal_subtree). For
    // a left rotation at root, `a = leaf_a` (single-node subtree). After
    // a previous internal rotation we may also have a leaf as the
    // misplaced root for a right rotation — we exercise both directions.
    let mut vtree = Vtree::linear(4);
    let root = vtree.root;
    let info = rotate_left(&mut vtree, root).expect("applicable");
    assert_invariants(&vtree);
    unrotate_left(&mut vtree, &info);
    assert_invariants(&vtree);
}

/// Edge case: two consecutive rotations at the same node.
#[test]
fn fixup_handles_consecutive_rotations_at_same_node() {
    let mut vtree = Vtree::balanced(8);
    let root = vtree.root;
    let _ = rotate_left(&mut vtree, root).expect("applicable");
    assert_invariants(&vtree);
    // After a left rotation at root, the new root's right child is what
    // was C (an internal subtree in balanced(8)) — try a left rotation
    // again. Applicability depends on whether new C's right subtree is
    // internal; if not, this is a no-op None.
    let _ = rotate_left(&mut vtree, root);
    assert_invariants(&vtree);
}

// Local enum mirroring `RotationKind` so the round-trip test
// can record kinds without leaking that crate-private type.
#[derive(Copy, Clone, Debug)]
enum RotKindLocal { Left, Right }
