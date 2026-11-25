//! Hand-built diagrams with unreachable nodes or summed-out children, seeded
//! diagrams with the storage that only unreduced or summed-out diagrams have,
//! a direct evaluation of their nodes, and the levels holding a node no pair
//! of their parent level names.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, NodeIdx, Tdd, TddBuilder, TddNodeId, ValueRef, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, ZERO};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

use super::compile::compile_clauses_on;
use super::oracle::{assert_canonical, assert_marginal_canonical};
use super::r#gen::{bare_roots, rand_cnf, vtree_shapes, CnfShape, Lcg};

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

/// Bonsai's inline marginal fixture on the balanced vtree over four
/// variables. Level `v4 = (x1, x2)` with x2 summed out holds `x1 · 5` and
/// `¬x1 · 0`, which is dead; level `v5 = (x3, x4)` holds `x3` and `¬x3`;
/// the output joins the first of each and the second of each.
pub fn inline_marginal() -> (Tdd, [VtreeIdx; 3]) {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let root = vtree.root();
    let (v4, v5) = vtree.children(root);
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    sum_out_leaf(&eng, &vtree, &mut builder, vtree.children(v4).1);
    let p = builder.push(&eng, v4, &[ChildPair::new(POS_LEAF_IDX, inline_side(5))]).unwrap();
    let dead = builder.push(&eng, v4, &[ChildPair::new(NEG_LEAF_IDX, inline_side(0))]).unwrap();
    let s0 = builder.push(&eng, v5, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let s1 = builder.push(&eng, v5, &[ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let out = builder.push(&eng, root, &[ChildPair::new(p, s0), ChildPair::new(dead, s1)]).unwrap();
    (builder.finish(TddNodeId { vtree: root, local: out }).unwrap(), [v4, v5, root])
}

/// Bonsai's donor of a level with two count slots, 2^31 and 2^32, too large
/// to sit inline: the right child of the balanced vtree over 64 variables,
/// summed out under an output that reads it through its first two nodes.
fn marginal_slot_donor(eng: &Engine) -> (Tdd, VtreeIdx) {
    let vtree = Arc::new(Vtree::balanced(64));
    let mut builder = Tdd::builder(eng, &vtree).unwrap();
    let root = vtree.root();
    let mut labels = vec![[ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX]; vtree.num_nodes()];
    for level in vtree.bottomup() {
        if vtree.node(level).is_leaf() || level == root { continue; }
        let (left, right) = vtree.children(level);
        let right_true = labels[right.idx()][0];
        labels[level.idx()] = std::array::from_fn(|label| {
            // A level names its leaf child in one form: true over it by its
            // two literals.
            if label == 0 && vtree.node(left).is_leaf() {
                let pairs = [ChildPair::new(POS_LEAF_IDX, right_true), ChildPair::new(NEG_LEAF_IDX, right_true)];
                return builder.push(eng, level, &pairs).unwrap();
            }
            builder.push(eng, level, &[ChildPair::new(labels[left.idx()][label], right_true)]).unwrap()
        });
    }
    let (left, source) = vtree.children(root);
    let output = builder.push(eng, root, &[
        ChildPair::new(labels[left.idx()][1], labels[source.idx()][0]),
        ChildPair::new(labels[left.idx()][2], labels[source.idx()][1]),
    ]).unwrap();
    // Each level holds true beside its literals, nodes that overlap, which
    // `finish` refuses as a repeated pair; only the summed level is kept.
    let mut donor = unsafe { builder.finish_unchecked(TddNodeId { vtree: root, local: output }) };
    crate::test_helpers::marginalize_under(eng, &mut donor, &[source]).unwrap();
    let mut counts = donor.level(source).marginal_counts().unwrap().to_vec();
    counts.sort_unstable();
    assert_eq!(counts, [1u128 << 31, 1u128 << 32]);
    (donor, source)
}

/// Bonsai's marginal boundary fixture. On the vtree
/// `((x1, (x2, x5)), (x3, x4))` the level `m = (x2, x5)` is summed out with
/// two count slots. Level `v4 = (x1, m)` holds `x1 · slot 0` and
/// `second · slot 1`, level `v5 = (x3, x4)` holds `x3`, and the output joins
/// each node of `v4` with it.
pub fn marginal_boundary(second: LeafLabel) -> (Tdd, [VtreeIdx; 4]) {
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let vtree = Arc::new(Vtree::join(
        &Vtree::join(&leaf(1), &Vtree::join(&leaf(2), &leaf(5)).unwrap()).unwrap(),
        &Vtree::join(&leaf(3), &leaf(4)).unwrap(),
    ).unwrap());
    let root = vtree.root();
    let (v4, v5) = vtree.children(root);
    let m = vtree.children(v4).1;
    let eng = Engine::new();
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let (donor, source) = marginal_slot_donor(&eng);
    builder.replace_level(&eng, m, donor.level_view(source)).unwrap();
    let n1 = builder.push(&eng, v4, &[ChildPair::new(POS_LEAF_IDX, slot_side(0))]).unwrap();
    let n2 = builder.push(&eng, v4, &[ChildPair::new(NodeIdx(second as u32), slot_side(1))]).unwrap();
    let s0 = builder.push(&eng, v5, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let out = builder.push(&eng, root, &[ChildPair::new(n1, s0), ChildPair::new(n2, s0)]).unwrap();
    (builder.finish(TddNodeId { vtree: root, local: out }).unwrap(), [m, v4, v5, root])
}

/// An inline marginal side holding `count`.
pub fn inline_side(count: u32) -> EncodedChildRef {
    ValueRef::Inline(count).side().expect("the fixture's counts fit inline")
}

/// A marginal side naming count slot `slot`.
pub fn slot_side(slot: u32) -> EncodedChildRef {
    ValueRef::Slot(slot).side().expect("the fixture's slots fit")
}

/// Give `builder` a summed-out leaf at `leaf`, whose sides are all inline.
pub fn sum_out_leaf(eng: &Engine, vtree: &Arc<Vtree>, builder: &mut TddBuilder, leaf: VtreeIdx) {
    let mut donor = eng.cube(&crate::vtree::Home::unchecked(vtree), std::iter::empty::<i32>()).unwrap();
    crate::test_helpers::marginalize_under(eng, &mut donor, &[leaf]).unwrap();
    assert_eq!(donor.level(leaf).marginal_counts(), Some([].as_slice()));
    builder.replace_level(eng, leaf, donor.level_view(leaf)).unwrap();
}

/// Seeded diagrams over a number of variables drawn from `vars`: each drawn
/// clause set compiled on every vtree shape, and every other one also conjoined
/// without reduction with a second draw, which leaves unreachable nodes in
/// place.
pub fn random_diagrams(seed: u64, count: usize, vars: Range<u32>) -> Vec<Tdd> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let n = draw_vars(&mut rng, &vars);
        let shape = CnfShape { clauses: n as usize, width: 4 };
        for (_, vtree) in bare_roots(vtree_shapes(n)) {
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
        for (_, vtree) in bare_roots(vtree_shapes(n)) {
            let eng = Engine::new();
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let inner: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
            if f.is_zero() || inner.is_empty() { continue; }
            let summed = inner[rng.below(inner.len() as u64) as usize];
            let leaves: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx)
                .filter(|&t| vtree.node(t).is_leaf() && !under(&vtree, t, summed)).collect();
            let leaf = leaves[rng.below(leaves.len() as u64) as usize];
            crate::test_helpers::marginalize_under(&eng, &mut f, &[summed, leaf]).unwrap();
            eng.minimize(&mut f).unwrap();
            assert_marginal_canonical(&f);
            out.push((f, summed));
        }
    }
    out
}

/// The internal levels below the root of `tdd` that hold a node no pair of
/// their parent level names, in order, by a read of every level that shares
/// nothing with the conjunction.
pub fn unnamed_levels(tdd: &Tdd) -> Vec<u32> {
    let mut out: Vec<u32> = unnamed_by_level(tdd).filter(|&(_, n)| n > 0).map(|(c, _)| c).collect();
    out.sort_unstable();
    out
}

/// How many nodes of the internal levels below the root of `tdd` no pair of
/// their parent level names, read as [`unnamed_levels`] reads them.
pub fn unnamed_node_count(tdd: &Tdd) -> usize {
    unnamed_by_level(tdd).map(|(_, n)| n).sum()
}

/// Each internal level below the root of `tdd` with how many of its nodes
/// no pair of its parent level names.
fn unnamed_by_level(tdd: &Tdd) -> impl Iterator<Item = (u32, usize)> + '_ {
    tdd.vtree.internal_bottomup().filter(|&(p, _, _)| tdd.is_structural_internal(p)).flat_map(move |(p, left, right)| {
        [(left, true), (right, false)].into_iter().filter(|&(c, _)| tdd.is_structural_internal(c)).map(move |(c, on_left)| {
            let view = tdd.levels[c.idx()].child_decoder();
            let parent = &tdd.levels[p.idx()];
            let named: std::collections::BTreeSet<usize> = (0..parent.slot_count())
                .flat_map(|i| parent.pairs_vec(i))
                .filter_map(|pair| view.child(if on_left { pair.left } else { pair.right }).index())
                .collect();
            (c.0, tdd.levels[c.idx()].slot_count() - named.len())
        })
    })
}

/// Whether `t` lies in the subtree rooted at `root`.
pub fn under(vtree: &Vtree, t: VtreeIdx, root: VtreeIdx) -> bool {
    vtree.in_subtree(t, root)
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
            level.pairs_vec(id.local.idx()).iter()
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

/// Whether a variable has its leaf under one of `levels` of `vtree`: the
/// sum-out by variables of the subtrees at `levels`, which also sums a node
/// whose children are all summed.
pub fn summing(vtree: &Vtree, levels: &[VtreeIdx]) -> impl Fn(VarId) -> bool + use<> {
    let mut marked = vec![false; vtree.num_nodes()];
    for &t in levels {
        if let Some(m) = marked.get_mut(t.idx()) {
            *m = true;
        }
    }
    let order: Vec<VtreeIdx> = vtree.bottomup().collect();
    for &t in order.iter().rev() {
        if vtree.node(t).parent().is_some_and(|p| marked[p.idx()]) {
            marked[t.idx()] = true;
        }
    }
    let mut vars = vec![false; vtree.num_vars() as usize + 1];
    for (leaf, var) in vtree.leaf_bottomup() {
        vars[var.0 as usize] = marked[leaf.idx()];
    }
    move |v| vars.get(v.0 as usize).copied().unwrap_or(false)
}

/// [`Engine::marginalize`] of the variables under `levels` of `f`'s vtree.
pub fn marginalize_under(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), crate::OperationError> {
    let summable = summing(f.vtree(), levels);
    eng.marginalize(f, summable)
}

/// [`Engine::and_marginalizing_where`] of the variables under `targets` of
/// `f`'s vtree.
pub fn and_marginalizing_under<'f, 'g>(
    eng: &Engine,
    f: impl Into<std::borrow::Cow<'f, Tdd>>,
    g: impl Into<std::borrow::Cow<'g, Tdd>>,
    targets: &[VtreeIdx],
) -> Result<Tdd, crate::OperationError> {
    let f = f.into();
    let summable = summing(f.vtree(), targets);
    eng.and_marginalizing_where(f, g, summable)
}

/// [`Engine::and_model_count`] summing the variables under `targets` of
/// `f`'s vtree.
pub fn and_model_count_under(eng: &Engine, f: Tdd, g: Tdd, targets: &[VtreeIdx]) -> Result<BigUint, crate::OperationError> {
    let summable = summing(f.vtree(), targets);
    eng.and_model_count(f, g, summable)
}

/// Sum out `levels` themselves, through the internal pass: the level
/// shapes (a root leaf, a lone leaf output, one of two siblings) that a
/// sum-out by variables does not make.
pub fn sum_levels(f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), crate::OperationError> {
    Arc::clone(f.context()).run(|eng| crate::marginal::tests::marginalize_levels(eng, f, levels))
}
