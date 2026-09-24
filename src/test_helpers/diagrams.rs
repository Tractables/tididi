//! A hand-built diagram with unreachable nodes, seeded diagrams with the
//! storage that only unreduced or summed-out diagrams have, and a direct
//! evaluation of their nodes.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, Tdd, TddNodeId, ValueRef, NEG_LEAF_IDX, POS_LEAF_IDX, ZERO};
use crate::vtree::{Vtree, VtreeIdx};
use crate::Engine;

use super::compile::compile_clauses_on;
use super::oracle::{assert_canonical, assert_marginal_canonical};
use super::r#gen::{rand_cnf, vtree_shapes, CnfShape, Lcg};

/// A variable count drawn from `vars`.
fn draw_vars(rng: &mut Lcg, vars: &Range<u32>) -> u32 {
    vars.start + rng.below(u64::from(vars.end - vars.start)) as u32
}

/// A hand-built diagram on the right-linear vtree over four variables, with
/// one unreachable node on each of the two lower levels, and those levels
/// from the bottom up.
///
/// Level `b` holds the four cubes over x3 and x4, `x3 x4`, `x3 ¬x4`,
/// `¬x3 x4` and `¬x3 ¬x4`, the last unreachable. Level `a` holds
/// `x2 (x3 x4) ∨ ¬x2 (x3 ¬x4)`, `¬x2 (¬x3 x4)` and the unreachable
/// `x2 (¬x3 ¬x4)`. The output joins `x1` with the first and `¬x1` with the
/// second.
pub fn chain() -> (Tdd, [VtreeIdx; 3]) {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::linear(4));
    let root = vtree.root();
    let a = vtree.children(root).1;
    let b = vtree.children(a).1;
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let cubes = [(POS_LEAF_IDX, POS_LEAF_IDX), (POS_LEAF_IDX, NEG_LEAF_IDX), (NEG_LEAF_IDX, POS_LEAF_IDX), (NEG_LEAF_IDX, NEG_LEAF_IDX)];
    let [p, w, q, z] = cubes.map(|(l, r)| builder.push(&eng, b, &[ChildPair::new(l, r)]).unwrap());
    let s = builder.push(&eng, a, &[ChildPair::new(POS_LEAF_IDX, p), ChildPair::new(NEG_LEAF_IDX, w)]).unwrap();
    let t = builder.push(&eng, a, &[ChildPair::new(NEG_LEAF_IDX, q)]).unwrap();
    builder.push(&eng, a, &[ChildPair::new(POS_LEAF_IDX, z)]).unwrap();
    let out = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, s), ChildPair::new(NEG_LEAF_IDX, t)]).unwrap();
    (builder.finish(TddNodeId { vtree: root, local: out }).unwrap(), [b, a, root])
}

/// Seeded diagrams over a number of variables drawn from `vars`: each drawn
/// clause set compiled on every vtree shape, and every other one also conjoined
/// without reduction with a second draw, which leaves unreachable nodes and
/// tombstones in place.
pub fn random_diagrams(seed: u64, count: usize, vars: Range<u32>) -> Vec<Tdd> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let n = draw_vars(&mut rng, &vars);
        let shape = CnfShape { clauses: n as usize, width: 4 };
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            assert_canonical(&f);
            if out.len() % 2 == 1 {
                let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
                assert_canonical(&g);
                out.push(eng.and(f.clone(), g).unwrap());
            }
            out.push(f);
        }
    }
    out
}

/// Seeded diagrams holding inline marginal references, over a number of
/// variables drawn from `vars`: each drawn clause set compiled on every vtree
/// shape, then one internal subtree below the output and one leaf outside it
/// summed out, and the result minimized. Each comes with the summed subtree.
pub fn marginal_diagrams(seed: u64, count: usize, vars: Range<u32>) -> Vec<(Tdd, VtreeIdx)> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let n = draw_vars(&mut rng, &vars);
        let shape = CnfShape { clauses: n as usize, width: 4 };
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let inner: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
            if f.is_zero() || inner.is_empty() { continue; }
            let summed = inner[rng.below(inner.len() as u64) as usize];
            let leaves: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx)
                .filter(|&t| vtree.node(t).is_leaf() && !under(&vtree, t, summed)).collect();
            let leaf = leaves[rng.below(leaves.len() as u64) as usize];
            eng.marginalize_levels(&mut f, &[summed, leaf]).unwrap();
            eng.minimize(&mut f).unwrap();
            assert_marginal_canonical(&f);
            out.push((f, summed));
        }
    }
    out
}

/// Whether `t` lies in the subtree rooted at `root`.
pub fn under(vtree: &Vtree, mut t: VtreeIdx, root: VtreeIdx) -> bool {
    loop {
        if t == root { return true; }
        match vtree.node(t).parent() {
            Some(parent) => t = parent,
            None => return false,
        }
    }
}

/// The value of node `id` of `f` at `assignment`, indexed by `VarId::idx`,
/// with every structural node `rejected` names read as false: a pair is the
/// product of its sides, a marginal side is its count and a leaf side is its
/// label's truth under the assignment.
pub fn node_value(f: &Tdd, id: TddNodeId, rejected: &dyn Fn(TddNodeId) -> bool, assignment: &[bool]) -> BigUint {
    fn side(
        f: &Tdd, rejected: &dyn Fn(TddNodeId) -> bool, assignment: &[bool],
        memo: &mut HashMap<TddNodeId, BigUint>, child: VtreeIdx, raw: EncodedChildRef,
    ) -> BigUint {
        if raw == ZERO.into() { return BigUint::ZERO; }
        let level = &f.levels[child.idx()];
        if level.is_marginal() {
            return match ChildDecoder::marginal().value(raw) {
                ValueRef::Inline(count) => BigUint::from(count),
                ValueRef::Slot(slot) => BigUint::from(level.marginal_counts().expect("a count level")[slot as usize]),
            };
        }
        if f.vtree.node(child).is_leaf() {
            let x = assignment[f.vtree.leaf_var(child).idx()];
            return BigUint::from(match raw.raw() { 0 => 1u8, 1 => u8::from(x), 2 => u8::from(!x), other => panic!("leaf label {other}") });
        }
        let id = TddNodeId { vtree: child, local: ChildDecoder::structural().node(raw) };
        if let Some(v) = memo.get(&id) { return v.clone(); }
        let v = if rejected(id) {
            BigUint::ZERO
        } else {
            let (left, right) = f.vtree.children(child);
            level.pairs_of_idx(id.local.idx()).iter()
                .map(|p| side(f, rejected, assignment, memo, left, p.left) * side(f, rejected, assignment, memo, right, p.right))
                .sum()
        };
        memo.insert(id, v.clone());
        v
    }
    side(f, rejected, assignment, &mut HashMap::new(), id.vtree, id.local.into())
}

/// Which variables of `f` still sit below structural levels only.
pub fn free_vars(f: &Tdd) -> Vec<bool> {
    let vtree = f.vtree();
    let mut free = vec![false; vtree.num_vars() as usize];
    for i in 0..vtree.num_nodes() {
        let mut t = VtreeIdx(i as u32);
        if !vtree.node(t).is_leaf() { continue; }
        let var = vtree.leaf_var(t);
        free[var.idx()] = loop {
            if f.levels[t.idx()].is_marginal() { break false; }
            match vtree.node(t).parent() { Some(p) => t = p, None => break true }
        };
    }
    free
}
