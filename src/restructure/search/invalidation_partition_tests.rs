//! Empirically validates the rotation-search invalidation set.
//!
//! For a hand-picked left rotation at `v_idx`, computes the score-determining
//! grandchild partition `(P, Q, R)` for *every* `(u, kind)` candidate before
//! and after the rotation. Asserts that the partition differs iff
//! `(u, kind)` belongs to the analytically-derived invalidation set:
//!
//!   { (v_idx, Left), (v_idx, Right),
//!     (w_idx, Left), (w_idx, Right),
//!     (parent(v_idx), kind_through_v_idx) }

use std::collections::BTreeSet;

use crate::vtree::rotate::rotate_left;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Left,
    Right,
}

fn child_pair(vtree: &Vtree, u: VtreeIdx) -> (VtreeIdx, VtreeIdx) {
    match *vtree.node(u) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        _ => panic!("expected internal at {:?}", u),
    }
}

fn vars_under(vtree: &Vtree, node: VtreeIdx) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    fn rec(vtree: &Vtree, node: VtreeIdx, out: &mut BTreeSet<u32>) {
        match *vtree.node(node) {
            VtreeNode::Leaf { var, .. } => {
                out.insert(var.0);
            }
            VtreeNode::Internal { left, right, .. } => {
                rec(vtree, left, out);
                rec(vtree, right, out);
            }
        }
    }
    rec(vtree, node, &mut out);
    out
}

type Partition = (BTreeSet<u32>, BTreeSet<u32>, BTreeSet<u32>);

/// Grandchild partition for probe `(u, kind)` — the triple of var-sets the
/// canonicity formula reads. `None` if the probe is structurally inapplicable.
fn partition(vtree: &Vtree, u: VtreeIdx, kind: Kind) -> Option<Partition> {
    let (left, right) = match *vtree.node(u) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        _ => return None,
    };
    match kind {
        Kind::Left => {
            let (m, r) = match *vtree.node(right) {
                VtreeNode::Internal { left, right, .. } => (left, right),
                _ => return None,
            };
            Some((vars_under(vtree, left), vars_under(vtree, m), vars_under(vtree, r)))
        }
        Kind::Right => {
            let (l, m) = match *vtree.node(left) {
                VtreeNode::Internal { left, right, .. } => (left, right),
                _ => return None,
            };
            Some((vars_under(vtree, l), vars_under(vtree, m), vars_under(vtree, right)))
        }
    }
}

#[test]
fn invalidation_set_matches_partition_diffs() {
    // balanced(16) gives a 4-deep balanced tree:
    //
    //                root (= p)
    //               /          \
    //             X            v_idx
    //            / \           /     \
    //          ..  ..         A      w_idx
    //                        / \      / \
    //                       .. .. (B inner) (C inner)
    //                              / \      / \
    //                            .. ..    .. ..
    //
    // Every named subtree (p, X, v_idx, A, w_idx, B, C) is internal, so every
    // probe direction is applicable both pre- and post-rotation.
    let mut vtree = Vtree::balanced(16);

    let p = vtree.root();
    let (x_idx, v_idx) = child_pair(&vtree, p);
    let (a_idx, w_idx) = child_pair(&vtree, v_idx);
    let (b_idx, c_idx) = child_pair(&vtree, w_idx);

    // v_idx = p.right, so the through-v_idx parent direction is Left.
    let parent_kind_through_v = Kind::Left;

    // Sanity: all named subtrees are internal.
    for (name, idx) in [
        ("p", p),
        ("X", x_idx),
        ("v_idx", v_idx),
        ("A", a_idx),
        ("w_idx", w_idx),
        ("B", b_idx),
        ("C", c_idx),
    ] {
        assert!(
            matches!(*vtree.node(idx), VtreeNode::Internal { .. }),
            "{} ({:?}) must be internal",
            name,
            idx
        );
    }

    let snapshot = |vt: &Vtree| -> Vec<((VtreeIdx, Kind), Option<Partition>)> {
        let mut out = Vec::new();
        for i in 0..vt.num_nodes() {
            if !matches!(vt.node(VtreeIdx(i as u32)), VtreeNode::Internal { .. }) {
                continue;
            }
            let u = VtreeIdx(i as u32);
            for k in [Kind::Left, Kind::Right] {
                out.push(((u, k), partition(vt, u, k)));
            }
        }
        out
    };

    let snapshot_pre = snapshot(&vtree);
    rotate_left(&mut vtree, v_idx).expect("left rotation at v_idx must be applicable");
    let snapshot_post = snapshot(&vtree);

    assert_eq!(snapshot_pre.len(), snapshot_post.len());

    let in_invalidation_set = |u: VtreeIdx, kind: Kind| -> bool {
        if u == v_idx || u == w_idx {
            return true;
        }
        if u == p && kind == parent_kind_through_v {
            return true;
        }
        false
    };

    println!();
    println!(
        "{:>6} {:>6} {:<6} {:<10} {:<10}  partition pre  ->  partition post",
        "u", "kind", "expect", "applic", "verdict"
    );
    println!("{}", "-".repeat(100));

    let mut violations = 0usize;
    for ((key, pre_part), (key2, post_part)) in snapshot_pre.iter().zip(snapshot_post.iter()) {
        assert_eq!(key, key2);
        let (u, kind) = *key;
        let expected_diff = in_invalidation_set(u, kind);
        let actual_diff = pre_part != post_part;
        let label = if u == p {
            "p"
        } else if u == v_idx {
            "v_idx"
        } else if u == w_idx {
            "w_idx"
        } else if u == x_idx {
            "X"
        } else if u == a_idx {
            "A"
        } else if u == b_idx {
            "B"
        } else if u == c_idx {
            "C"
        } else {
            "."
        };
        let applic = match (pre_part, post_part) {
            (Some(_), Some(_)) => "applic2",
            (None, None) => "N/A both",
            (Some(_), None) => "lost",
            (None, Some(_)) => "gained",
        };
        let verdict = if expected_diff == actual_diff {
            if expected_diff {
                "INVALIDATE OK"
            } else {
                "stable OK"
            }
        } else {
            violations += 1;
            "**MISMATCH**"
        };
        println!(
            "{:>6}({}) {:>6?} {:<6} {:<10} {:<14}  {:?}  ->  {:?}",
            u.idx(),
            label,
            kind,
            if expected_diff { "diff" } else { "same" },
            applic,
            verdict,
            pre_part,
            post_part
        );
    }
    assert_eq!(violations, 0, "partition-vs-invalidation-set mismatch");
}
