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

use crate::test_helpers::rotate_left;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Left,
    Right,
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
    let VtreeNode::Internal { left, right, .. } = *vtree.node(u) else {
        return None;
    };
    match kind {
        Kind::Left => {
            let VtreeNode::Internal {
                left: m, right: r, ..
            } = *vtree.node(right)
            else {
                return None;
            };
            Some((
                vars_under(vtree, left),
                vars_under(vtree, m),
                vars_under(vtree, r),
            ))
        }
        Kind::Right => {
            let VtreeNode::Internal {
                left: l, right: m, ..
            } = *vtree.node(left)
            else {
                return None;
            };
            Some((
                vars_under(vtree, l),
                vars_under(vtree, m),
                vars_under(vtree, right),
            ))
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
    let (x_idx, v_idx) = vtree.children(p);
    let (a_idx, w_idx) = vtree.children(v_idx);
    let (b_idx, c_idx) = vtree.children(w_idx);

    // v_idx = p.right, so the through-v_idx parent direction is Left.
    let parent_kind_through_v = Kind::Left;

    let named = [
        ("p", p),
        ("X", x_idx),
        ("v_idx", v_idx),
        ("A", a_idx),
        ("w_idx", w_idx),
        ("B", b_idx),
        ("C", c_idx),
    ];
    for (name, idx) in named {
        assert!(
            matches!(*vtree.node(idx), VtreeNode::Internal { .. }),
            "{name} ({idx:?}) must be internal",
        );
    }

    let snapshot = |vt: &Vtree| -> Vec<((VtreeIdx, Kind), Option<Partition>)> {
        (0..vt.num_nodes())
            .map(|i| VtreeIdx(i as u32))
            .filter(|&u| matches!(vt.node(u), VtreeNode::Internal { .. }))
            .flat_map(|u| [Kind::Left, Kind::Right].map(|k| ((u, k), partition(vt, u, k))))
            .collect()
    };

    let snapshot_pre = snapshot(&vtree);
    rotate_left(&mut vtree, v_idx).expect("left rotation at v_idx must be applicable");
    let snapshot_post = snapshot(&vtree);
    assert_eq!(snapshot_pre.len(), snapshot_post.len());

    let in_invalidation_set = |u: VtreeIdx, kind: Kind| -> bool {
        u == v_idx || u == w_idx || (u == p && kind == parent_kind_through_v)
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
        let label = named
            .iter()
            .find_map(|&(name, idx)| (idx == u).then_some(name))
            .unwrap_or(".");
        let applic = match (pre_part, post_part) {
            (Some(_), Some(_)) => "applic2",
            (None, None) => "N/A both",
            (Some(_), None) => "lost",
            (None, Some(_)) => "gained",
        };
        let verdict = match (expected_diff == actual_diff, expected_diff) {
            (true, true) => "INVALIDATE OK",
            (true, false) => "stable OK",
            (false, _) => {
                violations += 1;
                "**MISMATCH**"
            }
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
