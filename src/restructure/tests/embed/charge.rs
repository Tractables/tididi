//! The work `Engine::embed_with` charges for a copied diagram: one unit per
//! internal level and one per slot of it, one per level for queueing every
//! level for the contraction passes, and the queue itself, which later
//! reductions pay for. The vtree search races its candidates by these units.
use std::sync::Arc;

use crate::Engine;
use crate::diagram::Dirty;
use crate::test_helpers::{compile_clauses, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// `image` with a fresh leaf joined beside every internal node whose
/// bottom-up position is set in `wrap`, on the left where it is set in
/// `left`, and the root joined with one more: every pass-through carries an
/// internal node.
fn spread(image: &Vtree, wrap: u64, left: u64) -> Vtree {
    let mut nodes = Vec::new();
    let mut copy_of = vec![VtreeIdx(0); image.num_nodes()];
    let mut next = image.num_vars() + 1;
    let mut fresh = |nodes: &mut Vec<VtreeNode>| {
        nodes.push(VtreeNode::Leaf { var: VarId(next), parent: None });
        next += 1;
        VtreeIdx(nodes.len() as u32 - 1)
    };
    let join = |nodes: &mut Vec<VtreeNode>, l: VtreeIdx, r: VtreeIdx| {
        nodes.push(VtreeNode::Internal { left: l, right: r, parent: None });
        VtreeIdx(nodes.len() as u32 - 1)
    };
    for (i, t) in image.bottomup().enumerate() {
        copy_of[t.idx()] = VtreeIdx(nodes.len() as u32);
        nodes.push(match *image.node(t) {
            VtreeNode::Leaf { var, .. } => VtreeNode::Leaf { var, parent: None },
            VtreeNode::Internal { left, right, .. } => VtreeNode::Internal {
                left: copy_of[left.idx()],
                right: copy_of[right.idx()],
                parent: None,
            },
        });
        if !image.node(t).is_leaf() && wrap >> (i % 64) & 1 == 1 {
            let leaf = fresh(&mut nodes);
            copy_of[t.idx()] = match left >> (i % 64) & 1 == 1 {
                true => join(&mut nodes, leaf, copy_of[t.idx()]),
                false => join(&mut nodes, copy_of[t.idx()], leaf),
            };
        }
    }
    let leaf = fresh(&mut nodes);
    let root = join(&mut nodes, copy_of[image.root().idx()], leaf);
    Vtree::from_nodes(nodes, root, next - 1).expect("a tree")
}

#[test]
fn a_copied_embedding_charges_its_levels_slots_and_contraction_queue() {
    let eng = Engine::new();
    for (num_vars, clauses) in test_cases() {
        if !(2..=6).contains(&num_vars) {
            continue;
        }
        for (label, small) in vtree_shapes(num_vars) {
            let mut f = compile_clauses(&small, &clauses);
            f.minimize().unwrap();
            if f.is_zero() {
                continue;
            }
            for (wrap, left) in [(0, 0), (!0, 0), (!0, !0), (0x5555, 0x3333)] {
                let into = Arc::new(spread(&small, wrap, left));
                let plan = eng.embedding_plan(&small, &into, |v| v).unwrap();
                let mark = eng.limits().mark();
                let g = eng.embed_with(&f, &plan).unwrap();
                let work = eng.limits().work_since(mark);

                let internal = into.internal_bottomup_slice();
                let slots: u64 = internal.iter().map(|&t| g.level(t).slot_count() as u64).sum();
                let what = format!("{label} {clauses:?} wrap {wrap:x} left {left:x}");
                assert_eq!(work, 2 * internal.len() as u64 + slots, "{what}: work");
                let mut queued = Dirty::default().seeded(&into, internal.iter().copied(), None).unwrap();
                queued.set_loose(Some(Vec::new()));
                assert_eq!(format!("{:?}", g.dirty), format!("{queued:?}"), "{what}: contraction queue");
            }
        }
    }
}
