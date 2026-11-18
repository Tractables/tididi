use crate::diagram::{ChildPair, TddNodeId, POS_LEAF_IDX, NEG_LEAF_IDX, ONE_LEAF_IDX};
use crate::{Engine, Tdd};
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use std::sync::Arc;

#[test]
fn interning_indexes_prior_pushes_and_copy_replaces_the_entire_level() {
    let tree = Arc::new(Vtree::balanced(2));
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
    let source = Tdd::clause(&tree, [1]).unwrap();
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
    let tree = Arc::new(Vtree::balanced(4));
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
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    eng.refuse_nth_reserve(0);
    assert_eq!(Tdd::builder(&eng, &tree).unwrap_err(), crate::OperationError::OverBudget);
    eng.limits().grant_every_reserve();
    assert!(Tdd::builder(&eng, &tree).is_ok());
}

#[test]
fn a_full_level_answers_index_overflow() {
    let tree = Arc::new(Vtree::balanced(2));
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
        let vtree = Arc::new(Vtree::balanced(2));
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

/// Over ((x1,x2),(x3,x4)) with nine one-pair nodes in each child level, the
/// root's 81 distinct pairs, scattered over its nodes, pass; one of them
/// again in another node is refused and named.
#[test]
fn a_repeated_pair_is_found_among_many() {
    use crate::diagram::NodeIdx;
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let root = tree.root();
    let build = |extra: Option<(u32, u32)>| {
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        for (t, _, _) in tree.internal_bottomup().filter(|&(t, _, _)| t != root) {
            for i in 0..9 {
                b.push(&eng, t, &[ChildPair::new(NodeIdx(i / 3), NodeIdx(i % 3))]).unwrap();
            }
        }
        let mut pairs: Vec<_> = (0..81u32).map(|k| (k * 7 % 81 / 9, k * 7 % 9)).collect();
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

/// Over a balanced vtree of 16 variables whose root has two children 6561
/// nodes wide, more possible pairs than the bit grid covers, a pair repeated
/// in the root level is still refused and named.
#[test]
fn a_repeated_pair_is_found_between_wide_children() {
    use crate::diagram::NodeIdx;
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(16));
    let root = tree.root();
    let build = |extra: Option<(u32, u32)>| {
        let mut b = Tdd::builder(&eng, &tree).unwrap();
        let mut width = vec![3u32; tree.num_nodes()];
        for (t, l, r) in tree.internal_bottomup().filter(|&(t, _, _)| t != root) {
            let (wl, wr) = (width[l.idx()], width[r.idx()]);
            for i in 0..wl * wr {
                b.push(&eng, t, &[ChildPair::new(NodeIdx(i / wr), NodeIdx(i % wr))]).unwrap();
            }
            width[t.idx()] = wl * wr;
        }
        let mut pairs: Vec<_> = (0..1000u32).map(|k| ChildPair::new(NodeIdx(k * 7 % 6561), NodeIdx(k * 13 % 6561))).collect();
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

/// Over ((x1,x2),x3), the inner level holds the nodes `inner` and the root the
/// one node `root`; every pair on the right of the root names x3.
fn assemble(eng: &Engine, inner: &[&[ChildPair]], root: &[u32]) -> (Arc<Vtree>, crate::diagram::TddBuilder, TddNodeId) {
    let tree = Arc::new(Vtree::from_text("vtree 5\nL 0 1\nL 1 2\nI 2 0 1\nL 3 3\nI 4 2 3\n").unwrap());
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
        let r = crate::io::read_tdd(&mut text.as_slice(), &tree);
        if pruned != r.as_ref().is_ok() && !r.as_ref().is_err_and(|e| e.to_string().contains("twice")) {
            wrong.push(format!("{inner:?} {root:?}: text reader {:?}", r.map(|g| g.model_count().unwrap())));
        }
        // The binary reader trusts its levels: it gives back what was written.
        match crate::io::read_tdd_binary(&mut binary.as_slice(), &tree) {
            Ok(g) if g.model_count().unwrap() == f.model_count().unwrap() => {}
            r => wrong.push(format!("{inner:?} {root:?}: binary reader {:?}", r.map(|g| g.model_count().unwrap()))),
        }
    }
    let text = "p tdd 1 3 5 4 0\nL 0 1\nL 1 2\nL 3 3\nI 2 0 1 2 1\nI 2 0 1 2 1\nI 4 2 3 1 1\n";
    let tree = Arc::new(Vtree::from_text("vtree 5\nL 0 1\nL 1 2\nI 2 0 1\nL 3 3\nI 4 2 3\n").unwrap());
    match crate::io::read_tdd(&mut text.as_bytes(), &tree) {
        Err(e) if e.to_string().contains("twice") => {}
        r => wrong.push(format!("first case by hand: text reader {:?}", r.map(|g| g.model_count().unwrap()))),
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
