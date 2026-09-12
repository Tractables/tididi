//! A bottom-up model count written directly against the stored encoding.
//!
//! The walk reads the diagram the way every counting query in the crate does:
//! children before parents, per-node counts summed over pairs, with the three
//! level states (leaf, structural, marginal) handled explicitly. It is a test
//! because it is also the traversal contract's executable statement: a change
//! to the encoding that this walk cannot follow is a breaking change.

use crate::engine::Engine;
use crate::diagram::ChildRef;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::Tdd;
use crate::marginal::marginalize_levels;
use crate::diagram::{
    ChildDecoder, CountOverflow, ChildPair, ValueRef, NEG_LEAF_IDX, ONE_LEAF_IDX,
    POS_LEAF_IDX, TddLevel, TddNodeId,
};
use crate::vtree::{Vtree, VtreeIdx};

fn count(t: &Tdd) -> BigUint {
    // The constant-false function is the ZERO sentinel in `output`; no stored
    // node computes it, so the walk below never has to special-case it.
    if t.is_zero() {
        return BigUint::ZERO;
    }
    // One count per node slot, sized by `reference_slot_count` so that leaf levels
    // (which store nothing) get their three implicit slots.
    let mut c: Vec<Vec<BigUint>> = (0..t.vtree.num_nodes())
        .map(|i| vec![BigUint::ZERO; t.reference_slot_count(VtreeIdx(i as u32))])
        .collect();

    // Leaf levels: One is satisfied by both values of the variable, Pos and
    // Neg by one each. The local index IS the label (`LeafLabel::from_idx`).
    for (leaf, _var) in t.vtree.leaf_bottomup() {
        c[leaf.idx()][ONE_LEAF_IDX.idx()] = 2u32.into();
        c[leaf.idx()][POS_LEAF_IDX.idx()] = 1u32.into();
        c[leaf.idx()][NEG_LEAF_IDX.idx()] = 1u32.into();
    }

    // Internal levels, children first.
    for (v, l, r) in t.vtree.internal_bottomup() {
        let lvl = t.level(v);

        // A marginal level has no structure: its counts are stored.
        // `u128::MAX` marks an overflow whose exact value is in the side table.
        if lvl.is_marginal() {
            let counts = lvl.marginal_counts().expect("marginal level stores counts");
            for (i, &n) in counts.iter().enumerate() {
                c[v.idx()][i] = if n != u128::MAX {
                    n.into()
                } else {
                    lvl.marginal_counts_big()
                        .as_ref()
                        .and_then(|big| big.get(i))
                        .expect("overflow sentinel has a side-table entry")
                        .clone()
                };
            }
            continue;
        }

        // A structural level: each node is the disjoint union of its pairs, so
        // its count is the sum over pairs of the product of the two sides.
        // A side whose child level is marginal is a tagged reference — either
        // the count itself or an index into the child's counts — so it goes
        // through that child's `child_decoder`.
        let (lm, rm) = (t.level(l).child_decoder(), t.level(r).child_decoder());
        let side = |s, view: ChildDecoder, child: &[BigUint]| match view.child(s) {
            ChildRef::Value(ValueRef::Inline(k)) => BigUint::from(k),
            r => child[r.index().unwrap()].clone(),
        };
        // `internal_inputs_iter` skips tombstones; `i` is the slot index.
        for (i, pairs) in lvl.internal_inputs_iter() {
            let mut total = BigUint::ZERO;
            for p in pairs {
                total += side(p.left, lm, &c[l.idx()]) * side(p.right, rm, &c[r.idx()]);
            }
            c[v.idx()][i] = total;
        }
    }
    c[t.output.vtree.idx()][t.output.local.idx()].clone()
}

#[test]
fn a_hand_written_traversal_agrees_with_the_model_counter() {
    let eng = Engine::new();
    // 1. A structural diagram: (x1 ∨ x2) ∧ (x3 ∨ ¬x4) has 3 · 3 = 9 models.
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [3, -4]);
    assert_eq!(count(&f), BigUint::from(9u32));
    assert_eq!(count(&f), f.model_count());

    // 2. The same function after the left subtree {x1, x2} is marginalized:
    //    the root's pairs now carry inline counts on their left side.
    let (left, _right) = vtree.children(vtree.root());
    marginalize_levels(&eng, &mut f, &[left]).expect("no limits installed");
    assert!(f.level(left).is_marginal());
    assert_eq!(count(&f), BigUint::from(9u32));
    assert_eq!(count(&f), f.model_count());

    // 3. A hand-built diagram whose marginal child has an overflowed count.
    //    Left subtree: two marginal slots, counts 2^130 (overflow) and 5.
    //    Right subtree: one node (x3, ⊤), count 2. Root: both left slots
    //    paired with that node, so the total is (2^130 + 5) · 2.
    let huge: BigUint = BigUint::from(1u32) << 130;
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    levels[left.idx()].become_marginal(
        vec![u128::MAX, 5],
        Some(CountOverflow::from_iter([(0u32, huge.clone())])),
    );
    let (_, right) = vtree.children(vtree.root());
    let r0 = levels[right.idx()]
        .push_internal_node(&[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]);
    let root = levels[vtree.root().idx()].push_internal_node(&[
        ChildPair::new(ValueRef::Slot(0).side(), r0),
        ChildPair::new(ValueRef::Slot(1).side(), r0),
    ]);
    let g = Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: vtree.root(), local: root });
    let expected = (huge + BigUint::from(5u32)) * BigUint::from(2u32);
    assert_eq!(count(&g), expected);
    assert_eq!(g.model_count(), expected);
}
