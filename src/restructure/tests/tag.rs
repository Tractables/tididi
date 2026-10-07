use std::sync::Arc;

use crate::diagram::{TddNodeId, NodeIdx, LEAF_WIDTH};
use crate::restructure::{EmbedError, TagError};
use crate::test_helpers::{assert_canonical, compile_clauses, node_value, random_diagrams, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// Two-bit codes, the first and the last equal so that two nodes share one.
const CODES: [u64; 4] = [0b01, 0b10, 0b11, 0b01];

/// `at`'s subtree of `f`'s vtree, its variables renumbered `1..=m` in leaf
/// order, joined with a two-leaf code subtree over `m + 1` and `m + 2`;
/// with the renaming.
fn destination(f: &Tdd, at: VtreeIdx) -> (Arc<Vtree>, Vec<Option<VarId>>, [VarId; 2]) {
    let vtree = f.vtree();
    let mut local = vec![None; vtree.num_vars() as usize];
    let mut m = 0u32;
    for (t, v) in vtree.leaf_bottomup() {
        if under(vtree, t, at) {
            m += 1;
            local[v.idx()] = Some(VarId(m));
        }
    }
    let sub = vtree.project_to_vars(|v| local[v.idx()], m).unwrap();
    let code = [VarId(m + 1), VarId(m + 2)];
    let into = Vtree::join(&sub, &Vtree::balanced_over(&code).unwrap()).unwrap();
    (Arc::new(into), local, code)
}

fn under(vtree: &Vtree, mut t: VtreeIdx, root: VtreeIdx) -> bool {
    loop {
        if t == root {
            return true;
        }
        match vtree.node(t).parent() {
            Some(p) => t = p,
            None => return false,
        }
    }
}

/// Every assignment to `g`'s variables against the tags read node by node.
fn check(f: &Tdd, at: VtreeIdx, tags: &[Option<u32>]) {
    let (into, local, code_vars) = destination(f, at);
    let g = Engine::new().tag_level(f, at, &into, |v| local[v.idx()].unwrap(), tags, &code_vars, &CODES).unwrap();
    assert_canonical(&g);
    let n = into.num_vars();
    assert!(n <= 16, "brute force");
    let mut count = 0u64;
    for bits in 0..1u64 << n {
        let x = |v: VarId| bits >> v.idx() & 1 == 1;
        // The source assignment, read back through the renaming.
        let source: Vec<bool> = (1..=f.vtree().num_vars()).map(|v| local[(v - 1) as usize].is_some_and(&x)).collect();
        let y = u64::from(x(code_vars[0])) | u64::from(x(code_vars[1])) << 1;
        let expected = tags.iter().enumerate().any(|(i, tag)| {
            tag.is_some_and(|k| CODES[k as usize] == y)
                && node_value(f, TddNodeId { vtree: at, local: NodeIdx(i as u32) }, &|_| false, &source) > 0u32.into()
        });
        let literals: Vec<i32> = (1..=n).map(|v| if x(VarId(v)) { v as i32 } else { -(v as i32) }).collect();
        let got = g.clone().condition(literals.iter().copied()).unwrap();
        assert_eq!(!got.is_zero(), expected, "assignment {bits:b}");
        count += u64::from(expected);
    }
    assert_eq!(g.model_count().unwrap(), count.into());
}

/// Tags of every slot of `at`, cycling through the codes and skipping one.
fn cycling(f: &Tdd, at: VtreeIdx) -> Vec<Option<u32>> {
    let width = if f.vtree().node(at).is_leaf() { LEAF_WIDTH } else { f.level(at).nodes().len() };
    (0..width).map(|i| (i != 1).then_some(i as u32 % CODES.len() as u32)).collect()
}

#[test]
fn tagged_nodes_carry_their_codes() {
    for (num_vars, clauses) in test_cases().into_iter().filter(|(n, _)| (2..=8).contains(n)) {
        for (_, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            let (left, right) = vtree.children(vtree.root());
            for at in [vtree.root(), left, right] {
                if vtree.node(at).is_leaf() {
                    continue;
                }
                check(&f, at, &cycling(&f, at));
            }
        }
    }
}

#[test]
fn unreduced_diagrams_tag_to_canonical_ones() {
    for f in random_diagrams(11, 16, 3..9) {
        let (left, _) = f.vtree().children(f.vtree().root());
        if f.is_zero() || f.vtree().node(left).is_leaf() {
            continue;
        }
        check(&f, left, &cycling(&f, left));
    }
}

#[test]
fn a_leaf_level_tags_its_literals() {
    let vtree = Arc::new(Vtree::balanced(2));
    let f = compile_clauses(&vtree, &[vec![1, 2]]);
    let (left, _) = vtree.children(vtree.root());
    check(&f, left, &[None, Some(0), Some(1)]);
    check(&f, left, &[Some(2), None, None]);
    let (into, local, code) = destination(&f, left);
    assert!(matches!(
        Engine::new().tag_level(&f, left, &into, |v| local[v.idx()].unwrap(), &[Some(0), Some(1), None], &code, &CODES),
        Err(TagError::Tags(_)),
    ));
}

#[test]
fn nothing_tagged_is_false() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![3, 4]]);
    let (left, _) = vtree.children(vtree.root());
    let (into, local, code) = destination(&f, left);
    let none = vec![None; f.level(left).nodes().len()];
    let g = Engine::new().tag_level(&f, left, &into, |v| local[v.idx()].unwrap(), &none, &code, &CODES).unwrap();
    assert!(g.is_zero());
    assert_canonical(&g);
}

#[test]
fn shapes_codes_and_tags_are_checked() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![3, 4]]);
    let (left, _) = vtree.children(vtree.root());
    let (into, local, code) = destination(&f, left);
    let eng = Engine::new();
    let tags = cycling(&f, left);
    let map = |v: VarId| local[v.idx()].unwrap();
    // The subtree mapped onto the wrong leaves.
    assert!(matches!(
        eng.tag_level(&f, left, &into, |v| VarId(3 - map(v).0), &tags, &code, &CODES),
        Err(TagError::Placement(EmbedError::NotIsomorphic { .. })),
    ));
    // A code variable left out, and one that is no leaf of the code subtree.
    assert!(matches!(eng.tag_level(&f, left, &into, map, &tags, &code[..1], &CODES), Err(TagError::CodeVariables(_))));
    assert!(matches!(eng.tag_level(&f, left, &into, map, &tags, &[code[0], VarId(1)], &CODES), Err(TagError::CodeVariables(_))));
    // Too few tags, a tag past the codes, and a ragged code list.
    assert!(matches!(eng.tag_level(&f, left, &into, map, &tags[1..], &code, &CODES), Err(TagError::Tags(_))));
    let past: Vec<Option<u32>> = tags.iter().map(|t| t.map(|_| 9)).collect();
    assert!(matches!(eng.tag_level(&f, left, &into, map, &past, &code, &CODES), Err(TagError::Tags(_))));
    assert!(matches!(eng.tag_level(&f, left, &into, map, &tags, &code, &[]), Err(TagError::Tags(_))));
    // A level the vtree does not have.
    assert!(matches!(
        eng.tag_level(&f, VtreeIdx(99), &into, map, &tags, &code, &CODES),
        Err(TagError::Operation(OperationError::LevelNotInVtree(_))),
    ));
}
