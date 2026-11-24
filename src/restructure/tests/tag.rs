use std::sync::Arc;

use crate::diagram::{TddNodeId, NodeIdx, LEAF_WIDTH};
use crate::restructure::{EmbedError, TagError};
use crate::test_helpers::{assert_canonical, compile_clauses, node_value, random_diagrams, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// Two-bit codes, the first and the last equal so that two nodes share one.
const CODES: [u64; 4] = [0b01, 0b10, 0b11, 0b01];

/// Project `at`'s subtree of `vtree` onto variables `1..=m` in leaf order,
/// returning `(subtree, local_map, m)`.
fn subtree_projection(vtree: &Vtree, at: VtreeIdx) -> (Vtree, Vec<Option<VarId>>, u32) {
    let mut local = vec![None; vtree.num_vars() as usize];
    let mut m = 0u32;
    for (t, v) in vtree.leaf_bottomup() {
        if vtree.in_subtree(t, at) {
            m += 1;
            local[v.idx()] = Some(VarId(m));
        }
    }
    let sub = vtree.project_to_vars(|v| local[v.idx()], m).unwrap();
    (sub, local, m)
}

/// `at`'s subtree of `f`'s vtree, its variables renumbered `1..=m` in leaf
/// order, joined with a two-leaf code subtree over `m + 1` and `m + 2`;
/// with the renaming.
fn destination(f: &Tdd, at: VtreeIdx) -> (Arc<Vtree>, Vec<Option<VarId>>, [VarId; 2]) {
    let (sub, local, m) = subtree_projection(f.vtree(), at);
    let code = [VarId(m + 1), VarId(m + 2)];
    let into = Vtree::join(&sub, &Vtree::balanced_over(&code).unwrap()).unwrap();
    (Arc::new(into), local, code)
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
fn a_level_summed_beside_the_tagged_one_is_not_read() {
    let eng = Engine::new();
    for (num_vars, clauses) in test_cases().into_iter().filter(|(n, _)| (4..=8).contains(n)) {
        for (_, vtree) in vtree_shapes(num_vars) {
            let (left, right) = vtree.children(vtree.root());
            if vtree.node(left).is_leaf() || vtree.node(right).is_leaf() {
                continue;
            }
            let mut f = compile_clauses(&vtree, &clauses);
            if f.is_zero() {
                continue;
            }
            eng.marginalize_levels(&mut f, &[right]).unwrap();
            assert!(f.level(right).is_marginal());
            check(&f, left, &cycling(&f, left));
            // A summed level under the tagged one is refused.
            let (into, local, code) = destination(&f, vtree.root());
            let tags = vec![Some(0); f.level(vtree.root()).nodes().len()];
            assert!(matches!(
                eng.tag_level(&f, vtree.root(), &into, |v| local[v.idx()].unwrap(), &tags, &code, &CODES),
                Err(TagError::Operation(OperationError::MarginalLevel(level))) if vtree.in_subtree(level, right),
            ));
        }
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

/// Code subtrees of `width` leaves over `m + 1 ..= m + width`: balanced in
/// code order, and balanced, right-linear and random in a shuffled order, so
/// that a node's leaves read bits that are not one run of the code.
fn code_subtrees(m: u32, width: u32, rng: &mut impl FnMut() -> u64) -> Vec<Vtree> {
    let order: Vec<VarId> = (m + 1..=m + width).map(VarId).collect();
    let mut shuffled = order.clone();
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, (rng() % (i as u64 + 1)) as usize);
    }
    let random = Vtree::random(width, rng()).project_to_vars(|v| Some(VarId(m + v.0)), m + width).unwrap();
    vec![
        Vtree::balanced_over(&order).unwrap(),
        Vtree::balanced_over(&shuffled).unwrap(),
        Vtree::linear_from_order(&shuffled).unwrap(),
        random,
    ]
}

/// One more code than `slots`, the last a copy of the first, each a choice
/// per byte between two values, so that codes agree on many parts.
fn agreeing_codes(slots: usize, width: u32, rng: &mut impl FnMut() -> u64) -> Vec<u64> {
    let words = (width as usize).div_ceil(64).max(1);
    let bytes = (width as usize).div_ceil(8);
    let pool: Vec<[u64; 2]> = (0..bytes).map(|_| [rng() & 0xff, rng() & 0xff]).collect();
    let mut codes = vec![0u64; (slots + 1) * words];
    for k in 0..slots {
        for (j, choice) in pool.iter().enumerate() {
            let byte = choice[(rng() & 1) as usize];
            codes[k * words + j / 8] |= byte << (8 * (j % 8));
        }
        // No bit past the code's width.
        if !width.is_multiple_of(64) {
            codes[k * words + words - 1] &= (1u64 << (width % 64)) - 1;
        }
    }
    let first = codes[..words].to_vec();
    codes[slots * words..].copy_from_slice(&first);
    codes
}

/// Every assignment to the subtree's variables beside every code named,
/// against the tags read node by node, and nothing beside a code not named.
fn check_codes(f: &Tdd, at: VtreeIdx, into: &Arc<Vtree>, local: &[Option<VarId>], tags: &[Option<u32>], code_vars: &[VarId], codes: &[u64]) {
    let g = Engine::new().tag_level(f, at, into, |v| local[v.idx()].unwrap(), tags, code_vars, codes).unwrap();
    assert_canonical(&g);
    let words = code_vars.len().div_ceil(64).max(1);
    let code = |k: u32| &codes[k as usize * words..(k as usize + 1) * words];
    let mut named: Vec<&[u64]> = tags.iter().flatten().map(|&k| code(k)).collect();
    named.sort_unstable();
    named.dedup();
    let m = local.iter().flatten().count() as u32;
    let mut count = 0u64;
    for bits in 0..1u64 << m {
        let x = |v: VarId| bits >> v.idx() & 1 == 1;
        let source: Vec<bool> = local.iter().map(|l| l.is_some_and(&x)).collect();
        for &c in &named {
            let expected = tags.iter().enumerate().any(|(i, tag)| {
                tag.is_some_and(|k| code(k) == c)
                    && node_value(f, TddNodeId { vtree: at, local: NodeIdx(i as u32) }, &|_| false, &source) > 0u32.into()
            });
            let subtree = (1..=m).map(|v| if x(VarId(v)) { v as i32 } else { -(v as i32) });
            let bit = |i: usize| c[i / 64] >> (i % 64) & 1 == 1;
            let coded = code_vars.iter().enumerate().map(|(i, v)| if bit(i) { v.0 as i32 } else { -(v.0 as i32) });
            let got = g.clone().condition(subtree.chain(coded)).unwrap();
            assert_eq!(!got.is_zero(), expected, "assignment {bits:b}, code {c:x?}");
            count += u64::from(expected);
        }
    }
    assert_eq!(g.model_count().unwrap(), count.into());
}

#[test]
fn wide_codes_share_the_cube_nodes_they_agree_on() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for f in random_diagrams(23, 8, 3..7) {
        let vtree = f.vtree().clone();
        let (left, _) = vtree.children(vtree.root());
        let at = match vtree.node(left).is_leaf() {
            true => vtree.root(),
            false => left,
        };
        if f.is_zero() {
            continue;
        }
        let slots = f.level(at).nodes().len();
        let (sub, local, m) = subtree_projection(&vtree, at);
        // Widths under one word, over one, and past the widest key.
        for width in [5, 23, 70, 140] {
            let codes = agreeing_codes(slots, width, &mut rng);
            let code_vars: Vec<VarId> = (m + 1..=m + width).map(VarId).collect();
            // Two slots on the first code's copy, one left out.
            let tags: Vec<Option<u32>> = (0..slots).map(|i| match i {
                0 if slots > 2 => Some(slots as u32),
                1 if slots > 2 => None,
                _ => Some((i * 7 % (slots + 1)) as u32),
            }).collect();
            for code_tree in code_subtrees(m, width, &mut rng) {
                let into = Arc::new(Vtree::join(&sub, &code_tree).unwrap());
                check_codes(&f, at, &into, &local, &tags, &code_vars, &codes);
            }
        }
    }
}

/// A tagging refused at each reserve in turn: each refusal answers
/// `OverBudget` with the operand's model count as given, the engine tags
/// after it, and every granted attempt is the tagging.
#[test]
fn every_refused_reserve_of_a_tagging_leaves_the_operand() {
    let mut refused = 0;
    for (num_vars, clauses) in test_cases().into_iter().filter(|(n, _)| (4..=8).contains(n)).take(4) {
        let vtree = vtree_shapes(num_vars).into_iter().next().unwrap().1;
        let f = compile_clauses(&vtree, &clauses);
        let count = f.model_count().unwrap();
        let (left, right) = vtree.children(vtree.root());
        for at in [vtree.root(), left, right] {
            if vtree.node(at).is_leaf() {
                continue;
            }
            let (into, local, code_vars) = destination(&f, at);
            let tags = cycling(&f, at);
            let tag = |eng: &Engine| eng.tag_level(&f, at, &into, |v| local[v.idx()].unwrap(), &tags, &code_vars, &CODES);
            let want = tag(&Engine::new()).unwrap();
            // The checks fill the levels' kept pair counts, which the comparison reads.
            assert_canonical(&want);
            let eng = Engine::new();
            refused += eng
                .sweep_reserves(
                    &TagError::Operation(OperationError::OverBudget),
                    || {
                        let out = tag(&eng);
                        assert_eq!(f.model_count().unwrap(), count, "at {at:?}");
                        out
                    },
                    |g, n| {
                        assert_canonical(&g);
                        assert_eq!(format!("{:?}", g.levels), format!("{:?}", want.levels), "at {at:?}, {n}");
                    },
                )
                .refused;
        }
    }
    assert!(refused > 0, "no tagging refused");
}

/// The doc example into a vtree on a home, with the code subtree on either
/// side of the home: the result is on the projection on the tagged and code
/// variables and is the conjunction the example builds there.
#[test]
fn tagging_into_a_vtree_on_a_home_lands_on_its_projection() {
    use crate::vtree::Home;
    let eng = Engine::new();
    let pair = Arc::new(Vtree::balanced(2));
    let f = eng.clause(&pair, [1, 2]).unwrap();
    let (tagged, code) = (Vtree::balanced(2), Vtree::balanced_over(&[VarId(3), VarId(4)]).unwrap());
    for code_left in [false, true] {
        let (l, r) = if code_left { (&code, &tagged) } else { (&tagged, &code) };
        let home = Home::new(Vtree::join(&Vtree::join(l, r).unwrap(), &Vtree::leaf(VarId(5))).unwrap());
        // `into` lacks x2 and x4: the result's projection holds them.
        let into = Vtree::projection(&home, [VarId(1), VarId(3), VarId(5)]).unwrap();
        let g = eng.tag_level(&f, pair.root(), &into, |v| v, &[Some(0)], &[VarId(3), VarId(4)], &[0b10]).unwrap();
        let on = Vtree::projection(&home, (1..=4).map(VarId)).unwrap();
        assert!(g.vtree().is_on(&home) && g.vtree().same_tree(&on), "code on the left: {code_left}");
        assert_canonical(&g);
        let expected = eng.and(eng.clause(&on, [1, 2]).unwrap(), eng.cube(&on, [-3, 4]).unwrap()).unwrap();
        for bits in 0..1u64 << 5 {
            let asn: Vec<bool> = (0..5).map(|i| bits >> i & 1 == 1).collect();
            assert_eq!(crate::test_helpers::eval(&g, &asn), crate::test_helpers::eval(&expected, &asn), "{bits:05b}");
        }
        // A code variable inside the tagged side's subtree is refused.
        let mixed = eng.tag_level(&f, pair.root(), &into, |v| v, &[Some(0)], &[VarId(2), VarId(4)], &[0b10]);
        assert!(mixed.is_err(), "code on the left: {code_left}");
        let mirrored = Vtree::balanced_over(&[VarId(2), VarId(1)]).unwrap();
        let (l, r) = if code_left { (&code, &mirrored) } else { (&mirrored, &code) };
        let home = Home::new(Vtree::join(&Vtree::join(l, r).unwrap(), &Vtree::leaf(VarId(5))).unwrap());
        let into = Vtree::projection(&home, (1..=4).map(VarId)).unwrap();
        let refused = eng.tag_level(&f, pair.root(), &into, |v| v, &[Some(0)], &[VarId(3), VarId(4)], &[0b10]);
        assert!(
            matches!(refused, Err(TagError::Placement(EmbedError::NotIsomorphic { .. }))),
            "a mirrored tagged side, code on the left: {code_left}"
        );
    }
}
