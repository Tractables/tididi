//! Filtering diagrams with marginal levels.
//!
//! A structural level whose side toward a marginal child already holds
//! inline counts records that in its inlined-side marker, and the filter
//! copies the marker. These tests follow a filtered diagram through
//! reduction, further marginalization and conjunction, with the marker
//! carried and with it cleared as a public builder leaves it, and compare
//! every step with a direct evaluation of the operand in which the rejected
//! nodes read as false.

use std::collections::HashMap;

use num_bigint::BigUint;

use super::builder_sweep::{image, shape};
use super::*;
use crate::diagram::TddLevel;
use crate::query::PinSemantics;

/// The value of `f` at `assignment`, indexed by `VarId::idx`, with every
/// structural node `rejected` names read as false: a pair is the product of
/// its sides, a marginal side is its count and a leaf side is its label's
/// truth under the assignment.
fn value(f: &Tdd, rejected: &dyn Fn(TddNodeId) -> bool, assignment: &[bool]) -> BigUint {
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
    let out = f.output();
    side(f, rejected, assignment, &mut HashMap::new(), out.vtree, out.local.into())
}

/// Which variables of `f` still sit below structural levels only.
fn free_vars(f: &Tdd) -> Vec<bool> {
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

/// Check `g` against `f` with the rejected nodes false, the variables `g`
/// no longer has free summed out, and `literal`, when given, conjoined.
fn check(g: &Tdd, f: &Tdd, rejected: &dyn Fn(TddNodeId) -> bool, literal: Option<i32>, what: &str) {
    let (in_f, in_g) = (free_vars(f), free_vars(g));
    let vars: Vec<usize> = (0..in_f.len()).filter(|&v| in_f[v]).collect();
    let mut expected: HashMap<Vec<bool>, BigUint> = HashMap::new();
    for bits in 0u32..1 << vars.len() {
        let mut assignment = vec![false; in_f.len()];
        for (k, &v) in vars.iter().enumerate() { assignment[v] = bits >> k & 1 == 1; }
        let holds = literal.is_none_or(|l| assignment[l.unsigned_abs() as usize - 1] == (l > 0));
        let key: Vec<bool> = (0..in_f.len()).map(|v| in_g[v] && assignment[v]).collect();
        let entry = expected.entry(key).or_default();
        if holds { *entry += value(f, rejected, &assignment); }
    }
    for (key, want) in expected {
        let pins: Vec<Option<bool>> = (0..in_g.len()).map(|v| in_g[v].then_some(key[v])).collect();
        let got = if g.is_zero() { BigUint::ZERO } else { pinned_counts(g, &pins, PinSemantics::Evidence) };
        assert_eq!(got, want, "{what}: value at {pins:?}");
    }
}

/// The subtree to sum out after `summed`: its parent below the output, or
/// failing that its internal sibling.
fn next_target(vtree: &Vtree, summed: VtreeIdx) -> Option<VtreeIdx> {
    let parent = vtree.node(summed).parent()?;
    if parent != vtree.root() { return Some(parent); }
    let (left, right) = vtree.children(parent);
    let sibling = if left == summed { right } else { left };
    (!vtree.node(sibling).is_leaf()).then_some(sibling)
}

/// Conjoin `g` with a literal over one of its free variables, chosen by
/// `k`, and minimize; `None` when no variable is free.
fn conjoin_free_literal(eng: &Engine, g: &Tdd, k: usize) -> Option<(Tdd, i32)> {
    let free = free_vars(g);
    let vars: Vec<usize> = (0..free.len()).filter(|&v| free[v]).collect();
    let v = *vars.get(k % vars.len().max(1))? as i32 + 1;
    let literal = if k.is_multiple_of(2) { v } else { -v };
    let mut h = eng.and(g.clone(), eng.literal(g.vtree(), literal).unwrap()).unwrap();
    eng.minimize(&mut h).unwrap();
    Some((h, literal))
}

#[test]
fn marginal_levels_are_copied_and_the_marker_carried() {
    // Operands on which the two differ after each step: the reduction, a
    // conjunction, summing out more, and a conjunction after that.
    let mut shapes_differ = [0usize; 4];
    let mut markers_differ = [0usize; 4];
    let mut compared = 0;
    for (k, (f, summed)) in marginal_diagrams(0x3a7c, 200).into_iter().enumerate() {
        let keep = sparing_output(&f, k as u64, 10);
        let rejected = |id: TddNodeId| !keep(id);
        let eng = Engine::new();
        let FilterOutcome::Filtered { tdd: carried, .. } = eng.filter_nodes(&f, &keep).unwrap() else { continue };
        assert!(f.levels.iter().any(TddLevel::any_inlined_side));
        compared += 1;
        // Every marginal level is copied whole, and every structural level
        // keeps the operand's markers.
        for (level, source) in carried.levels.iter().zip(f.levels.iter()) {
            if source.is_marginal() {
                assert_eq!(format!("{level:?}"), format!("{source:?}"));
            } else {
                assert_eq!(level.inlined_sides, source.inlined_sides);
            }
        }
        check(&carried, &f, &rejected, None, "filtered");
        // The same diagram as a public builder would leave it.
        let mut cleared = carried.clone();
        for level in cleared.levels.iter_mut() { level.inlined_sides = 0; }
        let mut steps: Vec<Vec<Option<Tdd>>> = Vec::new();
        for (name, mut g) in [("carried", carried), ("cleared", cleared)] {
            eng.minimize(&mut g).unwrap();
            check(&g, &f, &rejected, None, name);
            let conjoined = conjoin_free_literal(&eng, &g, k).map(|(h, literal)| {
                check(&h, &f, &rejected, Some(literal), name);
                h
            });
            let summed_more = next_target(f.vtree(), summed).map(|next| {
                let mut h = g.clone();
                eng.marginalize_levels(&mut h, &[next]).unwrap();
                eng.minimize(&mut h).unwrap();
                check(&h, &f, &rejected, None, name);
                h
            });
            let conjoined_after = summed_more.as_ref().and_then(|h| conjoin_free_literal(&eng, h, k)).map(|(h, literal)| {
                check(&h, &f, &rejected, Some(literal), name);
                h
            });
            steps.push(vec![Some(g), conjoined, summed_more, conjoined_after]);
        }
        for step in 0..4 {
            let (Some(a), Some(b)) = (&steps[0][step], &steps[1][step]) else { continue };
            if shape(a) != shape(b) { shapes_differ[step] += 1; }
            if image(a) != image(b) { markers_differ[step] += 1; }
        }
    }
    assert!(compared >= 40, "too few operands compared: {compared}");
    // Both are right at every step, and they are the same diagram at every
    // step. The reduction keeps the carried markers and leaves the cleared
    // ones clear; the next conjunction or marginalization sets them again.
    assert_eq!(shapes_differ, [0; 4]);
    assert_eq!(markers_differ, [compared, 0, 0, 0]);
}
