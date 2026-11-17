//! [`Tdd::trim`] by a rebuild through [`TddBuilder`](crate::diagram::TddBuilder),
//! the oracle the in-place trim is tested and measured against: the same
//! diagram up to node numbering, seated fresh (every level owes the
//! reduction passes, no level counts, not certified canonical).
//!
//! Each round classes the nodes of each kept level top-down by the node of
//! the top of its chain of dropped nodes they belong to, then builds each
//! level of the projection bottom-up from the merged classes. A chain ending
//! at a leaf maps a class to Pos, Neg, or One when it holds both, which the
//! next round drops.

use std::sync::Arc;

use crate::Engine;
use crate::build::constant_without_leaf;
use crate::diagram::{ChildPair, NodeIdx, Tdd, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::OperationError;
use crate::vtree::{Home, VarId, Vtree, VtreeIdx, VtreeNode};

/// [`Tdd::trim`] by a rebuild: the same function on the same projection.
///
/// # Errors
///
/// As [`Tdd::trim`].
pub fn trim_by_rebuild(f: &mut Tdd, home: &Home) -> Result<bool, OperationError> {
    let context = Arc::clone(f.context());
    context.run(|eng| {
        let _op = eng.limits().enter()?;
        f.require_structure()?;
        if !f.vtree().is_on(home) {
            return Err(OperationError::VtreeMismatch);
        }
        let mut changed = false;
        while trim_once(eng, f, home)? {
            changed = true;
        }
        Ok(changed)
    })
}

/// A node of a level that no node of the level above names.
const UNNAMED: u32 = u32::MAX;

/// One round: drop the leaves named `One` now. Merging may name a kept leaf
/// `One`, for the next round.
fn trim_once(eng: &Engine, f: &mut Tdd, home: &Home) -> Result<bool, OperationError> {
    let vtree = Arc::clone(&f.vtree);
    if vtree.num_nodes() == 0 {
        return Ok(false);
    }
    if f.is_zero() {
        replace(eng, f, constant_without_leaf(&Vtree::empty_projection(home), false));
        return Ok(true);
    }
    eng.limits().check_stop()?;
    let n = vtree.num_nodes();
    // Whether a kept leaf is under the node.
    let mut kept = vec![false; n];
    let mut dropped_any = false;
    for t in vtree.bottomup() {
        kept[t.idx()] = match *vtree.node(t) {
            VtreeNode::Internal { left, right, .. } => kept[left.idx()] || kept[right.idx()],
            VtreeNode::Leaf { var: crate::vtree::REGION_VAR, .. } => crate::vtree::region_unsupported("the trim oracle"),
            VtreeNode::Leaf { parent, .. } => {
                let named = match parent {
                    None => f.output.local,
                    Some(p) => {
                        let pair = f.levels[p.idx()].pairs_iter_of_idx(0).next().expect("a node holds a pair");
                        let side = if vtree.children(p).0 == t { pair.left } else { pair.right };
                        NodeIdx(side.raw())
                    }
                };
                dropped_any |= named == ONE_LEAF_IDX;
                named != ONE_LEAF_IDX
            }
        };
    }
    if !dropped_any {
        return Ok(false);
    }
    if !kept[vtree.root().idx()] {
        replace(eng, f, constant_without_leaf(&Vtree::empty_projection(home), true));
        return Ok(true);
    }
    let own = {
        let vars = vtree.bottomup().filter(|&t| vtree.node(t).is_leaf() && kept[t.idx()]).map(|t| vtree.leaf_var(t));
        Vtree::projection(home, vars.collect::<Vec<VarId>>()).expect("kept leaves of the diagram's vtree")
    };
    let both = |t: VtreeIdx| match *vtree.node(t) {
        VtreeNode::Internal { left, right, .. } => kept[left.idx()] && kept[right.idx()],
        VtreeNode::Leaf { .. } => true,
    };

    // Top-down: the class of each node of a kept level is the node of the
    // top of its chain it belongs to; the top's class is itself.
    let mut class: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut top = vec![VtreeIdx(0); n];
    for t in vtree.bottomup().rev() {
        if !kept[t.idx()] {
            continue;
        }
        let width = f.reference_slot_count(t);
        match vtree.node(t).parent() {
            Some(p) if !both(p) => {
                let side_left = vtree.children(p).0 == t;
                let mut own_class = vec![UNNAMED; width];
                let level = &f.levels[p.idx()];
                for (j, &c) in class[p.idx()].iter().enumerate() {
                    if c == UNNAMED {
                        continue;
                    }
                    for pair in level.pairs_iter_of_idx(j) {
                        own_class[(if side_left { pair.left } else { pair.right }).raw() as usize] = c;
                    }
                }
                class[t.idx()] = own_class;
                top[t.idx()] = top[p.idx()];
            }
            _ => {
                class[t.idx()] = (0..width as u32).collect();
                top[t.idx()] = t;
            }
        }
    }

    // Bottom-up over the ends of the chains: `merged[b][c]` is the reference
    // that replaces a reference to node `c` of the top of `b`'s chain.
    let mut merged: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut end_of = vec![VtreeIdx(0); n];
    let mut builder = Tdd::builder(eng, &own)?;
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut pairs: Vec<ChildPair> = Vec::new();
    for t in vtree.bottomup() {
        if !kept[t.idx()] || !both(t) {
            continue;
        }
        let classes = f.reference_slot_count(top[t.idx()]);
        end_of[top[t.idx()].idx()] = t;
        match *vtree.node(t) {
            VtreeNode::Leaf { var: crate::vtree::REGION_VAR, .. } => crate::vtree::region_unsupported("the trim oracle"),
            VtreeNode::Leaf { .. } => {
                let mut held = vec![0u8; classes];
                for (j, &c) in class[t.idx()].iter().enumerate() {
                    if c != UNNAMED {
                        held[c as usize] |= 1 << j;
                    }
                }
                merged[t.idx()] = held
                    .iter()
                    .map(|&bits| match bits {
                        b if b == 1 << POS_LEAF_IDX.0 => POS_LEAF_IDX.0,
                        b if b == 1 << NEG_LEAF_IDX.0 => NEG_LEAF_IDX.0,
                        _ => ONE_LEAF_IDX.0,
                    })
                    .collect();
            }
            VtreeNode::Internal { left, right, .. } => {
                eng.limits().check_stop()?;
                let at = own.node_at_home(vtree.home_node(t)).expect("a node with kept leaves on both sides");
                let (ml, mr) = (&merged[end_of[left.idx()].idx()], &merged[end_of[right.idx()].idx()]);
                members.iter_mut().for_each(Vec::clear);
                members.resize_with(classes, Vec::new);
                for (j, &c) in class[t.idx()].iter().enumerate() {
                    if c != UNNAMED {
                        members[c as usize].push(j);
                    }
                }
                let level = &f.levels[t.idx()];
                for (c, nodes) in members.iter().enumerate().take(classes) {
                    pairs.clear();
                    for &j in nodes {
                        pairs.extend(level.pairs_iter_of_idx(j).map(|p| {
                            ChildPair::new(NodeIdx(ml[p.left.raw() as usize]), NodeIdx(mr[p.right.raw() as usize]))
                        }));
                    }
                    debug_assert!(!pairs.is_empty(), "every node of a chain's top names a node of its end");
                    let index = builder.push(eng, at, &pairs)?;
                    debug_assert_eq!(index.idx(), c);
                }
                merged[t.idx()] = (0..classes as u32).collect();
            }
        }
    }
    let root = vtree.root();
    let local = NodeIdx(merged[end_of[root.idx()].idx()][f.output.local.idx()]);
    // Safety: every level is built bottom-up from nodes of `f`, whose merged
    // nodes are disjoint because the nodes of the top of their chain are.
    let mut trimmed = unsafe { builder.finish_unchecked(TddNodeId { vtree: own.root(), local }) };
    trimmed.weights = f.weights.take();
    replace(eng, f, trimmed);
    Ok(true)
}

fn replace(eng: &Engine, f: &mut Tdd, mut by: Tdd) {
    if by.weights.is_none() {
        by.weights = f.weights.take();
    }
    let old = std::mem::replace(f, by);
    crate::diagram::return_levels(eng, old.levels.into_vec());
}
