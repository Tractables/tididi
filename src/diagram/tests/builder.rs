use crate::vtree::Home;
use crate::diagram::{ChildPair, TddNodeId, POS_LEAF_IDX, NEG_LEAF_IDX, ONE_LEAF_IDX};
use crate::{Engine, Tdd};
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use std::sync::Arc;

// Padded form: the true-label form the conjunction detects.
#[test]
fn interning_indexes_prior_pushes_and_copy_replaces_the_entire_level() {
    // The clause's level names x2 `One`.
    let tree = Arc::new(Vtree::balanced(2));
    let home = crate::vtree::Home::unchecked(&tree);
    let eng = Engine::new();
    let root = tree.root();
    let pair = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let other = [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)];
    let mut builder = Tdd::builder(&eng, &tree).unwrap();
    let first = builder.push(&eng, root, &pair).unwrap();
    assert_eq!(builder.intern(&eng, root, &pair).unwrap(), first);
    let second = builder.push(&eng, root, &other).unwrap();
    assert_eq!(builder.intern(&eng, root, &other).unwrap(), second);
    assert_ne!(builder.push(&eng, root, &pair).unwrap(), first);
    assert_eq!(builder.intern(&eng, root, &pair).unwrap(), first);
    let source = Tdd::clause(&home, [1]).unwrap();
    assert_canonical(&source);
    builder.replace_level(&eng, root, source.level_view(root)).unwrap();
    builder.replace_level(&eng, root, source.level_view(root)).unwrap();
    assert_eq!(builder.level(root).slot_count(), source.level(root).slot_count());
    let index = builder.intern(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    assert_eq!(index, source.output().local);
    let result = builder.finish(TddNodeId { vtree: root, local: index }).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), 2u32.into());
}

/// `[(One, Pos), (Pos, Neg)]` over `(x, y)` is `x ∨ y` but names `x` both as
/// `One` and as `Pos`; `trim` reads the first pair and made it true.
#[test]
fn finish_and_the_reader_refuse_a_level_that_mixes_leaf_forms() {
    use crate::diagram::TddBuildError;
    let eng = Engine::new();
    let tree = Home::new(Vtree::balanced(2));
    let root = tree.root();
    let (x, _) = tree.children(root);
    let mut b = Tdd::builder(&eng, &tree).unwrap();
    let mixed = [ChildPair::new(ONE_LEAF_IDX, POS_LEAF_IDX), ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let node = b.push(&eng, root, &mixed).unwrap();
    let refused = b.finish(TddNodeId { vtree: root, local: node }).unwrap_err();
    assert_eq!(refused, TddBuildError::MixedLeafForm { level: root, leaf: x });
    let read = |text: &str| crate::io::read_tdd(&mut text.as_bytes(), &tree);
    assert!(read("p tdd 1 2 3 2 0\nL 0 1\nL 1 2\nI 2 0 1 0 1 1 2\n").is_err());
    let literals = read("p tdd 1 2 3 2 0\nL 0 1\nL 1 2\nI 2 0 1 1 1 1 2 2 1\n").unwrap();
    assert_eq!(literals.model_count().unwrap(), 3u32.into());
}

/// A leaf named both ways is refused as the right child of a narrow level,
/// and as the left child beside a child 16 nodes wide.
#[test]
fn finish_refuses_mixed_leaf_forms_on_either_side_and_beside_a_wide_child() {
    use crate::diagram::{NodeIdx, TddBuildError};
    use crate::vtree::VarId;
    let eng = Engine::new();
    let tree = Home::new(Vtree::balanced(2));
    let root = tree.root();
    let mut b = Tdd::builder(&eng, &tree).unwrap();
    let node = b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let refused = b.finish(TddNodeId { vtree: root, local: node }).unwrap_err();
    assert_eq!(refused, TddBuildError::MixedLeafForm { level: root, leaf: tree.children(root).1 });

    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let pair = |l: u32, r: u32| Vtree::join(&leaf(l), &leaf(r)).unwrap();
    let wide = Vtree::join(&pair(2, 3), &pair(4, 5)).unwrap();
    let tree = Home::new(Vtree::join(&leaf(1), &wide).unwrap());
    let root = tree.root();
    let build = |first| {
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        push_every_pair_below_root(&eng, &tree, &mut b);
        let pairs = [ChildPair::new(first, NodeIdx(0)), ChildPair::new(NEG_LEAF_IDX, NodeIdx(15))];
        let node = b.push(&eng, root, &pairs).unwrap();
        b.finish(TddNodeId { vtree: root, local: node })
    };
    assert!(build(POS_LEAF_IDX).is_ok());
    let refused = build(ONE_LEAF_IDX).map(|_| ()).unwrap_err();
    assert_eq!(refused, TddBuildError::MixedLeafForm { level: root, leaf: tree.children(root).0 });
}

#[test]
fn finish_seats_the_constants_on_a_vtree_with_no_node() {
    use crate::diagram::{NodeIdx, TddBuildError, TRUE, ZERO};
    use crate::vtree::Home;
    let eng = Engine::new();
    let empty = Vtree::empty_projection(&Home::new(Vtree::balanced(5)));
    let at = |local| TddNodeId { vtree: empty.root(), local };
    let one = Tdd::builder(&eng, &empty).unwrap().finish(at(TRUE)).unwrap();
    assert!(one.is_one());
    assert_eq!(one.model_count().unwrap(), 32u32.into());
    assert!(Tdd::builder(&eng, &empty).unwrap().finish(at(ZERO)).unwrap().is_zero());
    let refused = Tdd::builder(&eng, &empty).unwrap().finish(at(NodeIdx(0))).unwrap_err();
    assert_eq!(refused, TddBuildError::BadOutput(at(NodeIdx(0))));
}

/// Assemble `x1 ∧ ¬x2` over `tree`, returning the output node for
/// [`TddBuilder::finish`].
fn fill(
    eng: &Engine,
    b: &mut crate::diagram::TddBuilder,
    tree: &Arc<Vtree>,
) -> Result<TddNodeId, crate::OperationError> {
    let root = tree.root();
    let (left, right) = tree.children(root);
    b.reserve(eng, left, 4, 4)?;
    let pair = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let l = b.push(eng, left, &pair)?;
    let again = b.intern(eng, left, &pair)?;
    let r = b.push(eng, right, &[ChildPair::new(ONE_LEAF_IDX, ONE_LEAF_IDX)])?;
    let out = b.intern(eng, root, &[ChildPair::new(l, r)])?;
    assert_eq!(again, l, "interning an existing pair list appends nothing");
    Ok(TddNodeId { vtree: root, local: out })
}

#[test]
fn every_refusal_point_answers_over_budget_and_returns_the_buffers() {
    let tree = Home::new(Vtree::balanced(4));
    let mut refused = 0;
    for cut in 0..10u32 {
        let eng = Engine::new();
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        eng.refuse_nth_reserve(cut);
        match fill(&eng, &mut b, &tree) {
            Ok(output) => {
                eng.limits().grant_every_reserve();
                let f = b.finish(output).unwrap();
                assert_canonical(&f);
                assert_eq!(f.model_count().unwrap(), 4u32.into());
            }
            Err(e) => {
                assert_eq!(e, crate::OperationError::OverBudget, "cut {cut}");
                eng.limits().grant_every_reserve();
                b.abandon(&eng);
                assert_eq!(eng.scratch.levels.occupancy(), 1, "cut {cut} left a buffer outside the pool");
                refused += 1;
            }
        }
    }
    assert!(refused > 0, "no reservation was refused across the sweep");
}

#[test]
fn a_refused_level_take_leaves_the_builder_over_budget() {
    let tree = Home::new(Vtree::balanced(4));
    let eng = Engine::new();
    eng.refuse_nth_reserve(0);
    assert_eq!(Tdd::builder(&eng, &tree).unwrap_err(), crate::OperationError::OverBudget);
    eng.limits().grant_every_reserve();
    assert!(Tdd::builder(&eng, &tree).is_ok());
}

#[test]
fn a_full_level_answers_index_overflow() {
    let tree = Home::new(Vtree::balanced(2));
    let eng = Engine::new();
    let root = tree.root();
    let mut b = Tdd::builder(&eng, &tree).unwrap();
    eng.limits().pin_level_width_cap(Some(1));
    let first = [ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)];
    let second = [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)];
    let one = b.push(&eng, root, &first).unwrap();
    // Interning a pair list the level already holds reads the index instead of
    // appending, so the cap does not apply to it.
    assert_eq!(b.intern(&eng, root, &first).unwrap(), one);
    assert_eq!(b.push(&eng, root, &second).unwrap_err(), crate::OperationError::IndexOverflow);
    assert_eq!(b.intern(&eng, root, &second).unwrap_err(), crate::OperationError::IndexOverflow);
    let f = b.finish(TddNodeId { vtree: root, local: one }).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), 1u32.into());
}

/// A node survives a refused index update; retry must find it rather than append a duplicate.
#[test]
fn interning_recovers_after_an_index_update_is_refused() {
    use crate::diagram::NodeIdx;
    use crate::OperationError;

    for multiple_pairs in [false, true] {
        let vtree = Home::new(Vtree::balanced(2));
        let root = vtree.root();
        let eng = Engine::new();
        let mut builder = Tdd::builder(&eng, &vtree).unwrap();
        let mut first = vec![ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)];
        let mut second = vec![ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)];
        if multiple_pairs {
            first.push(ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX));
            second.push(ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX));
        }
        let original = builder.intern(&eng, root, &first).unwrap();
        builder.reserve(&eng, root, 1, second.len()).unwrap();
        // The append fits the reservation, so the first reserve the push
        // takes is its hash-table update: refuse that one.
        eng.refuse_nth_reserve(0);
        assert_eq!(builder.push(&eng, root, &second), Err(OperationError::OverBudget));
        eng.limits().grant_every_reserve();
        assert_eq!(builder.level(root).slot_count(), 2);
        // Rebuilding the discarded index may also be refused and retried.
        eng.refuse_nth_reserve(0);
        assert_eq!(builder.intern(&eng, root, &second), Err(OperationError::OverBudget));
        eng.limits().grant_every_reserve();
        let recovered = builder.intern(&eng, root, &second).unwrap();
        assert_eq!(recovered, NodeIdx(1));
        assert_eq!(builder.intern(&eng, root, &first).unwrap(), original);
        assert_eq!(builder.level(root).slot_count(), 2);

        // The unfinished level has an unused node and may have uncontracted twins.
        let mut result = builder.finish(TddNodeId { vtree: root, local: recovered }).unwrap();
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), if multiple_pairs { 2u32.into() } else { 1u32.into() });
    }
}

/// Over the balanced vtree of 8 variables, each non-root level holds one node
/// per pair of its children's nodes, the literals at a leaf, so the root's
/// children are 16 nodes wide. The root's 256 distinct pairs, scattered over
/// its nodes, pass; one of them again in another node is refused and named.
#[test]
fn a_repeated_pair_is_found_among_many() {
    use crate::diagram::NodeIdx;
    let eng = Engine::new();
    let tree = Home::new(Vtree::balanced(8));
    let root = tree.root();
    let build = |extra: Option<(u32, u32)>| {
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        push_products(&eng, &tree, &mut b);
        let mut pairs: Vec<_> = (0..256u32).map(|k| (k * 7 % 256 / 16, k * 7 % 16)).collect();
        pairs.extend(extra);
        let mut out = NodeIdx(0);
        for chunk in pairs.chunks(4) {
            let chunk: Vec<_> = chunk.iter().map(|&(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r))).collect();
            out = b.push(&eng, root, &chunk).unwrap();
        }
        b.finish(TddNodeId { vtree: root, local: out })
    };
    assert!(build(None).is_ok());
    match build(Some((5, 2))) {
        Err(crate::diagram::TddBuildError::RepeatedPair { level, pair }) => {
            assert_eq!((level, pair), (root, ChildPair::new(NodeIdx(5), NodeIdx(2))));
        }
        r => panic!("{:?}", r.map(|_| ())),
    }
}

/// Over a balanced vtree of 32 variables whose root has two children 65536
/// nodes wide, more possible pairs than the bit grid covers, a pair repeated
/// in the root level is still refused and named.
#[test]
fn a_repeated_pair_is_found_between_wide_children() {
    use crate::diagram::NodeIdx;
    let eng = Engine::new();
    let tree = Home::new(Vtree::balanced(32));
    let root = tree.root();
    let build = |extra: Option<(u32, u32)>| {
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        push_products(&eng, &tree, &mut b);
        let mut pairs: Vec<_> = (0..1000u32).map(|k| ChildPair::new(NodeIdx(k * 7), NodeIdx(k * 13))).collect();
        pairs.extend(extra.map(|(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r))));
        let out = b.push(&eng, root, &pairs).unwrap();
        b.finish(TddNodeId { vtree: root, local: out })
    };
    assert!(build(None).is_ok());
    match build(Some((700, 1300))) {
        Err(crate::diagram::TddBuildError::RepeatedPair { level, pair }) => {
            assert_eq!((level, pair), (root, ChildPair::new(NodeIdx(700), NodeIdx(1300))));
        }
        r => panic!("{:?}", r.map(|_| ())),
    }
}

/// Push at every level below the root one node per pair of its children's
/// nodes, `Pos` and `Neg` standing for a leaf's: node `i` pairs left
/// `i / wr` with right `i % wr`.
fn push_products(eng: &Engine, tree: &Vtree, b: &mut crate::diagram::TddBuilder) {
    use crate::diagram::NodeIdx;
    let root = tree.root();
    let mut width = vec![2u32; tree.num_nodes()];
    for (t, l, r) in tree.internal_bottomup().filter(|&(t, _, _)| t != root) {
        let (wl, wr) = (width[l.idx()], width[r.idx()]);
        let side = |c: crate::vtree::VtreeIdx, i: u32| match tree.node(c).is_leaf() {
            true => [POS_LEAF_IDX, NEG_LEAF_IDX][i as usize],
            false => NodeIdx(i),
        };
        for i in 0..wl * wr {
            b.push(eng, t, &[ChildPair::new(side(l, i / wr), side(r, i % wr))]).unwrap();
        }
        width[t.idx()] = wl * wr;
    }
}

/// Below the root of a balanced vtree, a one-pair node for every pair of each
/// level's children: the four pairs of literals over two leaves, then every
/// pair of the nodes below.
fn push_every_pair_below_root(eng: &Engine, tree: &Vtree, b: &mut crate::diagram::TddBuilder) {
    use crate::diagram::NodeIdx;
    let literals = [POS_LEAF_IDX, NEG_LEAF_IDX];
    let mut nodes: Vec<Vec<NodeIdx>> = vec![literals.to_vec(); tree.num_nodes()];
    for (t, l, r) in tree.internal_bottomup().filter(|&(t, _, _)| t != tree.root()) {
        let mut level = Vec::new();
        for &left in &nodes[l.idx()] {
            for &right in &nodes[r.idx()] {
                level.push(b.push(eng, t, &[ChildPair::new(left, right)]).unwrap());
            }
        }
        nodes[t.idx()] = level;
    }
}


/// Over ((x1,x2),x3), the inner level holds the nodes `inner` and the root the
/// one node `root`; every pair on the right of the root names x3.
fn assemble(eng: &Engine, inner: &[&[ChildPair]], root: &[u32]) -> (Arc<Vtree>, crate::diagram::TddBuilder, TddNodeId) {
    let tree = Arc::new(Vtree::from_text("vtree 5\nL 0 1\nL 1 2\nI 2 0 1\nL 3 3\nI 4 2 3\n").unwrap());
    crate::vtree::Home::unchecked(&tree);
    let (left, _) = tree.children(tree.root());
    let mut b = Tdd::builder(eng, &tree).unwrap();
    for pairs in inner {
        b.push(eng, left, pairs).unwrap();
    }
    let pairs: Vec<_> = root.iter().map(|&i| ChildPair::new(crate::diagram::NodeIdx(i), POS_LEAF_IDX)).collect();
    let out = b.push(eng, tree.root(), &pairs).unwrap();
    let out = TddNodeId { vtree: tree.root(), local: out };
    (tree, b, out)
}

/// A pair twice in one level, within a node or across two, denotes one event
/// twice: negation and the count read it as two disjoint ones. The builder
/// and the text reader refuse it; distinct pairs over the same children pass.
#[test]
fn a_pair_repeated_in_a_level_is_refused() {
    let eng = Engine::new();
    let not1_and_2 = [ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)];
    let one_and_2 = [ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)];
    let twice = [not1_and_2[0], not1_and_2[0]];
    let (_, b, out) = assemble(&eng, &[&not1_and_2, &one_and_2], &[0, 1]);
    assert_eq!(b.finish(out).unwrap().model_count().unwrap(), 2u32.into());
    let cases: [(&[&[ChildPair]], &[u32]); 3] = [
        (&[&not1_and_2, &not1_and_2], &[1]),
        (&[&not1_and_2, &not1_and_2], &[0, 1]),
        (&[&twice], &[0]),
    ];
    let mut wrong = Vec::new();
    for (inner, root) in cases {
        let (tree, b, out) = assemble(&eng, inner, root);
        match b.finish(out) {
            Err(e) if e.to_string().contains("twice") => {}
            Err(e) => wrong.push(format!("{inner:?} {root:?}: {e}")),
            Ok(f) => wrong.push(format!("{inner:?} {root:?}: built, counts {}", f.model_count().unwrap())),
        }
        // The writers keep only reachable nodes, so the first case reaches
        // the readers by hand below.
        let (_, b, out) = assemble(&eng, inner, root);
        let f = unsafe { b.finish_unchecked(out) };
        let (mut text, mut binary) = (Vec::new(), Vec::new());
        crate::io::write_tdd(&mut text, &f).unwrap();
        crate::io::write_tdd_binary(&mut binary, &f).unwrap();
        let pruned = root.len() == 1 && inner.len() == 2;
        let r = crate::io::read_tdd(&mut text.as_slice(), &crate::vtree::Home::unchecked(&tree));
        if pruned != r.as_ref().is_ok() && !r.as_ref().is_err_and(|e| e.to_string().contains("twice")) {
            wrong.push(format!("{inner:?} {root:?}: text reader {:?}", r.map(|g| g.model_count().unwrap())));
        }
        // The binary reader trusts its levels: it gives back what was written.
        match crate::io::read_tdd_binary(&mut binary.as_slice(), &crate::vtree::Home::unchecked(&tree)) {
            Ok(g) if g.model_count().unwrap() == f.model_count().unwrap() => {}
            r => wrong.push(format!("{inner:?} {root:?}: binary reader {:?}", r.map(|g| g.model_count().unwrap()))),
        }
    }
    let text = "p tdd 1 3 5 4 0\nL 0 1\nL 1 2\nL 3 3\nI 2 0 1 2 1\nI 2 0 1 2 1\nI 4 2 3 1 1\n";
    let tree = Arc::new(Vtree::from_text("vtree 5\nL 0 1\nL 1 2\nI 2 0 1\nL 3 3\nI 4 2 3\n").unwrap());
    match crate::io::read_tdd(&mut text.as_bytes(), &crate::vtree::Home::unchecked(&tree)) {
        Err(e) if e.to_string().contains("twice") => {}
        r => wrong.push(format!("first case by hand: text reader {:?}", r.map(|g| g.model_count().unwrap()))),
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A plain root, neither a home's root nor a projection of one, is refused
/// by name where a public entry takes a vtree.
#[test]
fn a_vtree_off_every_home_is_refused_by_name() {
    use crate::restructure::RestructureError;
    let eng = Engine::new();
    let plain = Arc::new(Vtree::balanced(2));
    assert!(matches!(Tdd::builder(&eng, &plain), Err(crate::OperationError::NotOnHome)));
    let mut f = Tdd::clause(&Home::new(Vtree::linear(2)), [1, 2]).unwrap();
    assert_eq!(f.restructure_to(&plain, usize::MAX), Err(RestructureError::Operation(crate::OperationError::NotOnHome)));
}
