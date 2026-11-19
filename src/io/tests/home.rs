//! Reading onto a home: the projection on the leaves a file declares, then
//! the trim of `TddBuilder::finish`.

use std::sync::Arc;

use crate::diagram::{ChildPair, NodeIdx, TddNodeId, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::io::{read_tdd, read_tdd_binary, write_tdd, write_tdd_binary, IoError};
use crate::test_helpers::check::check_trimmed;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

fn text(f: &Tdd) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_tdd(&mut bytes, f).unwrap();
    bytes
}

fn binary(f: &Tdd) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_tdd_binary(&mut bytes, f).unwrap();
    bytes
}

/// `f` read back from its text and from its binary file over `vtree`.
fn both_readers(f: &Tdd, vtree: &Arc<Vtree>) -> [Result<Tdd, IoError>; 2] {
    [read_tdd(&mut text(f).as_slice(), vtree), read_tdd_binary(&mut binary(f).as_slice(), vtree)]
}

/// The same leaves, nodes, pairs and output.
fn assert_same_nodes(a: &Tdd, b: &Tdd, what: &str) {
    let leaves = |t: &Tdd| t.vtree().leaf_bottomup().map(|(_, v)| v).collect::<Vec<_>>();
    assert_eq!(leaves(a), leaves(b), "{what}: leaves");
    assert_eq!(a.vtree().num_nodes(), b.vtree().num_nodes(), "{what}: vtree");
    assert_eq!(a.output(), b.output(), "{what}: output");
    for t in a.vtree().bottomup() {
        let (x, y) = (a.level(t), b.level(t));
        assert_eq!(x.slot_count(), y.slot_count(), "{what}: level {t:?} width");
        for i in 0..x.slot_count() {
            assert_eq!(x.pairs_vec(i), y.pairs_vec(i), "{what}: level {t:?} node {i}");
        }
    }
}

/// x1 ∧ x3 over ((x1 x2) (x3 x4)), x2 and x4 named `One`: on `vtree` as
/// `finish` seats it.
fn x1_x3_padded(eng: &Engine, vtree: &Arc<Vtree>) -> Result<Tdd, crate::diagram::TddBuildError> {
    let (l, r) = vtree.children(vtree.root());
    let mut b = Tdd::builder(eng, vtree).unwrap();
    let left = b.push(eng, l, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let right = b.push(eng, r, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let out = b.push(eng, vtree.root(), &[ChildPair::new(left, right)]).unwrap();
    b.finish(TddNodeId { vtree: vtree.root(), local: out })
}

#[test]
fn a_diagram_on_a_projection_reads_back_node_for_node() {
    let home = Home::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [2, 3, 5].map(VarId)).unwrap();
    let f = Tdd::clause(&own, [2, -5]).unwrap().and_clause([3, 5]).unwrap();
    assert!(f.vtree().is_on(&home) && f.vtree().num_leaves() == 3);
    let other = Vtree::projection(&home, [VarId(1)]).unwrap();
    let first = read_tdd(&mut text(&f).as_slice(), f.vtree()).unwrap();
    check_trimmed(&first, false).unwrap();
    assert_eq!(first.model_count().unwrap(), f.model_count().unwrap());
    for vtree in [f.vtree(), &own, home.vtree(), &other] {
        for back in both_readers(&f, vtree) {
            let back = back.unwrap();
            assert!(back.vtree().is_on(&home));
            assert_same_nodes(&back, &first, "read back");
        }
    }
}

#[test]
fn a_smooth_file_reads_onto_the_projection_on_its_support() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let root = Arc::new(Vtree::clone(home.vtree()));
    let padded = x1_x3_padded(&eng, &root).unwrap();
    assert!(Arc::ptr_eq(padded.vtree(), &root));
    let own = Vtree::projection(&home, [1, 3].map(VarId)).unwrap();
    let expected = Tdd::cube(&own, [1, 3]).unwrap();
    for back in both_readers(&padded, home.vtree()) {
        let back = back.unwrap();
        check_trimmed(&back, false).unwrap();
        assert_same_nodes(&back, &expected, "the support");
        assert_eq!(back.model_count().unwrap(), padded.model_count().unwrap());
    }
}

#[test]
fn finish_drops_a_leaf_every_pair_names_one() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let f = x1_x3_padded(&eng, home.vtree()).unwrap();
    check_trimmed(&f, false).unwrap();
    let own = Vtree::projection(&home, [1, 3].map(VarId)).unwrap();
    let mut b = Tdd::builder(&eng, &own).unwrap();
    let out = b.push(&eng, own.root(), &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let built = b.finish(TddNodeId { vtree: own.root(), local: out }).unwrap();
    assert_same_nodes(&f, &built, "without the leaves");
    assert_eq!(f.model_count().unwrap(), 4u32.into());
}

#[test]
fn finish_seats_a_diagram_with_a_summed_out_level_untrimmed() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(2));
    let vtree = home.vtree();
    let (x1, x2) = vtree.children(vtree.root());
    let mut donor = Tdd::cube(&Arc::new(Vtree::clone(vtree)), std::iter::empty::<i32>()).unwrap();
    donor.marginalize_levels(&[x2]).unwrap();
    let mut b = Tdd::builder(&eng, vtree).unwrap();
    b.replace_level(&eng, x2, donor.level_view(x2)).unwrap();
    let side = crate::test_helpers::inline_side(1);
    let out = b.push(&eng, vtree.root(), &[ChildPair::new(POS_LEAF_IDX, side), ChildPair::new(ONE_LEAF_IDX, side)]);
    // A level names its leaf child in one form.
    assert!(out.is_ok());
    let f = b.finish(TddNodeId { vtree: vtree.root(), local: NodeIdx(0) });
    assert!(matches!(f, Err(crate::diagram::TddBuildError::MixedLeafForm { leaf, .. }) if leaf == x1));
    let mut b = Tdd::builder(&eng, vtree).unwrap();
    b.replace_level(&eng, x2, donor.level_view(x2)).unwrap();
    let out = b.push(&eng, vtree.root(), &[ChildPair::new(ONE_LEAF_IDX, side)]).unwrap();
    let f = b.finish(TddNodeId { vtree: vtree.root(), local: out }).unwrap();
    assert!(Arc::ptr_eq(f.vtree(), vtree));
    assert!(f.has_marginal_level());
    assert_eq!(f.level(vtree.root()).pairs_vec(0), [ChildPair::new(ONE_LEAF_IDX, side)]);
}

#[test]
fn a_false_file_reads_to_the_constant() {
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [1, 3].map(VarId)).unwrap();
    let root = Arc::new(Vtree::clone(home.vtree()));
    let ones = (own.num_leaves(), own.num_nodes(), own.topo_pos(own.root()));
    let files = [
        "p tdd 1 0 0 0 ZERO\n".to_string(),
        "p tdd 1 4 7 6 ZERO\n".to_string(),
        format!("p tdd 1 {} {} {} ZERO\n", ones.0, ones.1, ones.2),
    ];
    for (written, vtree) in [(&root, home.vtree()), (&own, &own)] {
        for back in both_readers(&Tdd::zero(written), vtree) {
            let back = back.unwrap();
            assert!(back.is_zero() && back.vtree().num_nodes() == 0 && back.vtree().is_on(&home));
        }
        let one = read_tdd(&mut "p tdd 1 0 0 0 ONE\n".as_bytes(), vtree).unwrap();
        assert!(one.is_one() && one.vtree().num_nodes() == 0 && one.vtree().is_on(&home));
    }
    for (file, vtree) in files.iter().zip([home.vtree(), home.vtree(), &own]) {
        let back = read_tdd(&mut file.as_bytes(), vtree).unwrap();
        assert!(back.is_zero() && back.vtree().num_nodes() == 0 && back.vtree().is_on(&home), "{file}");
    }
    // Counts that are neither 0 0 nor the supplied vtree's.
    assert!(matches!(read_tdd(&mut files[2].as_bytes(), home.vtree()), Err(IoError::Format(_))));
    assert!(matches!(read_tdd(&mut "p tdd 1 4 7 6 ONE\n".as_bytes(), home.vtree()), Err(IoError::Format(_))));
}

#[test]
fn a_file_of_another_tree_is_a_format_error() {
    let home = Home::new(Vtree::balanced(4));
    let linear = Arc::new(Vtree::linear(4));
    let wider = Arc::new(Vtree::balanced(5));
    let cases = [Tdd::cube(&linear, [1, 4]).unwrap(), Tdd::cube(&wider, [1, 5]).unwrap()];
    for f in &cases {
        for back in both_readers(f, home.vtree()) {
            assert!(matches!(back, Err(IoError::Format(_))), "{:?}", back.map(|_| ()));
        }
    }
    // A header larger than the home, read before any allocation from it.
    let big = "p tdd 1 4000000000 7999999999 0 0\n";
    assert!(matches!(read_tdd(&mut big.as_bytes(), home.vtree()), Err(IoError::Format(_))));
}

#[test]
fn a_version_2_file_whose_trim_drops_a_leaf_keeps_its_count() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let root = Arc::new(Vtree::clone(home.vtree()));
    let mut padded = x1_x3_padded(&eng, &root).unwrap();
    padded.attach_level_counts().unwrap();
    let bytes = binary(&padded);
    assert_eq!(bytes[8], 2, "a version 2 file");
    let back = read_tdd_binary(&mut bytes.as_slice(), home.vtree()).unwrap();
    assert_eq!(back.vtree().num_leaves(), 2);
    assert!(back.has_level_counts());
    assert_eq!(back.model_count().unwrap(), padded.model_count().unwrap());
    assert_eq!(back.model_count().unwrap(), 4u32.into());
}

/// A version 1 binary file of `body`.
fn sealed(body: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x89TDD\r\n\x1a\n".to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(0x0A0B_0C0Du32.to_le_bytes());
    bytes.extend((body.len() as u64).to_le_bytes());
    bytes.extend(body);
    let sum = crate::io::binary::xxh64(&bytes);
    bytes.extend(sum.to_le_bytes());
    bytes
}

#[test]
fn a_pair_repeated_in_a_level_is_refused() {
    // ((x1 x2) x3): the level (x1 x2) holds ¬x1 ∧ x2 twice; the root reads
    // the second copy, or both.
    let vtree = Arc::new(Vtree::from_text("vtree 5\nL 0 1\nL 1 2\nI 2 0 1\nL 3 3\nI 4 2 3\n").unwrap());
    let home = Home::new(Vtree::clone(&vtree));
    let texts = ["I 4 2 3 1 1\n", "I 4 2 3 0 1 1 1\n"].map(|root| format!("p tdd 1 3 5 4 0\nL 0 1\nL 1 2\nI 2 0 1 2 1\nI 2 0 1 2 1\nL 3 3\n{root}"));
    // The postorder 1 2 0 3 0, the output, the level (x1 x2): two one-pair
    // nodes, each (2, 1) in two bits a side; the root: one node.
    let lower = [2, 0, 2, 1, 2 | 1 << 2 | 2 << 4 | 1 << 6];
    let roots: [&[u8]; 2] = [&[1, 0, 1, 1, 1 | 1 << 1], &[1, 1, 2, 1, 1, 0, 1 << 3 | 1 << 5 | 1 << 6]];
    for (file, root) in texts.iter().zip(roots) {
        for vtree in [&vtree, home.vtree()] {
            match read_tdd(&mut file.as_bytes(), vtree) {
                Err(IoError::Format(m)) => assert!(m.contains("twice"), "{m}"),
                other => panic!("{:?}", other.map(|_| ())),
            }
        }
        let body: Vec<u8> = [3, 5, 1, 2, 0, 3, 0, 1].iter().chain(&lower).chain(root).copied().collect();
        let bytes = sealed(&body);
        assert!(crate::io::read_tdd_binary(&mut bytes.as_slice(), &vtree).is_ok(), "trusted over a root");
        match crate::io::read_tdd_binary(&mut bytes.as_slice(), home.vtree()) {
            Err(IoError::Format(m)) => assert!(m.contains("twice"), "{m}"),
            other => panic!("{:?}", other.map(|_| ())),
        }
    }
}

#[test]
fn a_vtree_with_a_region_is_refused() {
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::project_regions(&home, vec![0, 1], vec![(2, vec![(2, 4)].into_boxed_slice())]);
    assert!(own.has_regions());
    let f = Tdd::cube(&Arc::new(Vtree::balanced(4)), [1, 2]).unwrap();
    for back in both_readers(&f, &own) {
        assert!(matches!(back, Err(IoError::Format(_))), "{:?}", back.map(|_| ()));
    }
}

/// A trim refused on a home is reported as the refusal it is, not as a
/// format error; every other build error stays a format error. The trim runs
/// on a context engine whose limits are reset, so only the allocator can
/// refuse it, and no test can arm that end to end.
#[test]
fn a_refused_trim_is_an_allocation_refusal() {
    use crate::diagram::TddBuildError;
    let refused = crate::io::build_error(TddBuildError::OverBudget, |_| unreachable!("a refusal is no format error"));
    assert!(matches!(refused, IoError::OverBudget), "{refused:?}");
    assert!(refused.to_string().contains("allocator refused"), "{refused}");
    let format = crate::io::build_error(TddBuildError::IncompatibleWeights, |e| IoError::Format(e.to_string()));
    assert!(matches!(format, IoError::Format(_)), "{format:?}");
}
