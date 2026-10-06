//! The binary format: its checksum, round trips that match the text reader
//! node for node, and refusals of every truncation, flipped bit and
//! resealed corruption without a panic.

use std::sync::Arc;

use crate::diagram::Tdd;
use crate::io::binary::{encode_levels, postorder, xxh64};
use crate::io::{read_tdd, read_tdd_binary, write_tdd, write_tdd_binary, IoError};
use crate::test_helpers::{assert_canonical, compile_clauses, rand_cnf, test_cases, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree};

/// How many multi-pair nodes the file of `tdd` writes in each node code,
/// indexed by code.
fn node_codes(tdd: &Tdd) -> [u64; 3] {
    let mut codes = [0; 3];
    if !tdd.is_zero() {
        encode_levels(tdd, &postorder(&tdd.vtree), &mut Vec::new(), &mut codes);
    }
    codes
}

fn binary(f: &Tdd) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_tdd_binary(&mut bytes, f).unwrap();
    bytes
}

fn text(f: &Tdd) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_tdd(&mut bytes, f).unwrap();
    bytes
}

/// A file the reader refuses with a format error, not a panic.
fn refused(bytes: &[u8], vtree: &Arc<Vtree>, what: &str) {
    match read_tdd_binary(&mut &bytes[..], vtree) {
        Err(IoError::Format(message)) => assert!(!message.is_empty()),
        other => panic!("{what}: expected a format error, got {other:?}"),
    }
}

/// Rewrite the length and the checksum after a change to the body, so the
/// reader gets past them to the records.
fn reseal(bytes: &mut Vec<u8>) {
    bytes.truncate(bytes.len() - 8);
    let body = (bytes.len() - 24) as u64;
    bytes[16..24].copy_from_slice(&body.to_le_bytes());
    let sum = xxh64(bytes);
    bytes.extend_from_slice(&sum.to_le_bytes());
}

/// The two readers built the same levels: the same nodes in the same order,
/// each with the same pairs in the same order, and the same output.
fn assert_same_levels(a: &Tdd, b: &Tdd, what: &str) {
    assert_eq!(a.output(), b.output(), "{what}: output");
    for t in a.vtree().bottomup() {
        let (x, y) = (a.level(t), b.level(t));
        assert_eq!(x.nodes().len(), y.nodes().len(), "{what}: level {t:?} width");
        for i in 0..x.nodes().len() {
            assert_eq!(x.pairs_of_idx(i), y.pairs_of_idx(i), "{what}: level {t:?} node {i}");
        }
    }
}

/// Write `f` both ways, read both back, and check they agree with each other
/// and with `f`.
fn round_trip(f: &Tdd, what: &str) {
    let vtree = f.vtree();
    let bytes = binary(f);
    let back = read_tdd_binary(&mut bytes.as_slice(), vtree).unwrap();
    let from_text = read_tdd(&mut text(f).as_slice(), vtree).unwrap();
    assert!(Arc::ptr_eq(back.vtree(), vtree));
    assert_same_levels(&back, &from_text, what);
    assert_canonical(&back);
    assert_eq!(back.model_count().unwrap(), f.model_count().unwrap(), "{what}: count");
    assert!(back.equivalent(f).unwrap(), "{what}: function");
    assert_eq!(binary(&back), bytes, "{what}: rewriting what was read changes the bytes");
}

#[test]
fn the_checksum_is_xxh64() {
    // Reference values from the `xxHash` project's `XXH64` with seed 0.
    let long: Vec<u8> = (0..768).map(|i| (i % 256) as u8).collect();
    for (data, want) in [
        (&b""[..], 0xef46_db37_51d8_e999),
        (b"a", 0xd24e_c4f1_a98c_6e5b),
        (b"abc", 0x44bc_2cf5_ad77_0999),
        (&long[..], 0x8e03_c838_c596_036f),
    ] {
        assert_eq!(xxh64(data), want, "{} bytes", data.len());
    }
}

#[test]
fn compiled_formulas_round_trip_as_the_text_reader_reads_them() {
    for (num_vars, clauses) in test_cases() {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            assert_canonical(&f);
            round_trip(&f, &format!("{clauses:?} on {shape}"));
        }
    }
}

#[test]
fn constants_and_a_single_leaf_round_trip() {
    for num_vars in [1, 2, 5] {
        let vtree = Arc::new(Vtree::balanced(num_vars));
        for f in [Tdd::zero(&vtree), Tdd::one(&vtree), Tdd::clause(&vtree, [1]).unwrap(), Tdd::clause(&vtree, [-1]).unwrap()] {
            assert_canonical(&f);
            round_trip(&f, &format!("{num_vars} vars"));
        }
    }
}

#[test]
fn wide_levels_round_trip() {
    // Random rows over twenty variables: levels of hundreds of nodes, so the
    // counts take several varint bytes and the indices many bits.
    let mut rng = Lcg::new(7);
    let vars: Vec<VarId> = (1..=20).map(VarId).collect();
    let rows: Vec<u64> = (0..3000).map(|_| rng.next_u64() & ((1 << 20) - 1)).collect();
    for (shape, vtree) in vtree_shapes(20) {
        let f = Tdd::from_models(&vtree, &vars, &rows).unwrap();
        assert_canonical(&f);
        assert!(!f.level(vtree.root()).nodes().is_empty());
        let bytes = binary(&f);
        assert!(bytes.len() * 2 < text(&f).len(), "{shape}: binary {} bytes, text {}", bytes.len(), text(&f).len());
        round_trip(&f, shape);
    }
}

#[test]
fn every_node_code_is_written_and_read_back() {
    // Rows of a relation as a template compiles them: a node of many pairs
    // whose left indices ascend, or whose right indices are close together,
    // is cheaper in a code of its own.
    let mut rng = Lcg::new(3);
    let vars: Vec<VarId> = (1..=16).map(VarId).collect();
    let mut codes = [0; 3];
    for (n, skew) in [(2000, false), (2000, true), (300, true)] {
        let rows: Vec<u64> = (0..n)
            .map(|_| {
                let (x, y) = (rng.next_u64() & 0xff, rng.next_u64() & 0xff);
                if skew { (x >> rng.below(8)) | (y >> rng.below(8)) << 8 } else { x | y << 8 }
            })
            .collect();
        for (shape, vtree) in vtree_shapes(16) {
            let f = Tdd::from_models(&vtree, &vars, &rows).unwrap();
            assert_canonical(&f);
            for (code, count) in node_codes(&f).into_iter().enumerate() {
                codes[code] += count;
            }
            round_trip(&f, &format!("{n} rows, skew {skew}, {shape}"));
        }
    }
    assert!(codes.iter().all(|&c| c > 0), "node codes written {codes:?}");
}

/// A file over `Vtree::balanced(2)` whose root level is one node of `pairs`
/// pairs, its stream the `(value, bits)` fields in order.
fn two_leaf_file(pairs: u8, fields: &[(u32, u32)]) -> Vec<u8> {
    let mut stream = Vec::new();
    let (mut acc, mut held) = (0u64, 0u32);
    for &(v, bits) in fields {
        acc |= u64::from(v) << held;
        held += bits;
        while held >= 8 {
            stream.push(acc as u8);
            acc >>= 8;
            held -= 8;
        }
    }
    if held > 0 {
        stream.push(acc as u8);
    }
    // The vtree's leaf and node counts, its postorder, the output plus one;
    // the root level: one node, one of them multi-pair, its pairs, the
    // stream's bytes, the bitmap and the pair count less two.
    let mut body = vec![2, 3, 1, 2, 0, 1, 1, 1, pairs, stream.len() as u8, 1, pairs - 2];
    body.extend(stream);
    let mut bytes = b"\x89TDD\r\n\x1a\n".to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(0x0A0B_0C0Du32.to_le_bytes());
    bytes.extend([0; 8]);
    bytes.extend(body);
    bytes.extend([0; 8]);
    reseal(&mut bytes);
    bytes
}

#[test]
fn hand_written_node_codes_are_read_or_refused() {
    let vtree = Arc::new(Vtree::balanced(2));
    let xor = crate::and(Tdd::clause(&vtree, [1, 2]).unwrap(), Tdd::clause(&vtree, [-1, -2]).unwrap()).unwrap();
    assert_canonical(&xor);
    // Leaf indices: 1 is the variable, 2 its negation. The pairs are
    // (x1, not x2) and (not x1, x2), a two-bit index each side.
    let offset = [(1, 2), (1, 2), (1, 5), (1, 2), (1, 1), (2, 2), (0, 1)];
    let ascending = [(2, 2), (1, 2), (0, 5), (1, 2), (1, 5), (1, 1), (0, 1)];
    for (fields, what) in [(&offset[..], "offset"), (&ascending[..], "ascending")] {
        let back = read_tdd_binary(&mut two_leaf_file(2, fields).as_slice(), &vtree).unwrap();
        assert_canonical(&back);
        assert!(back.equivalent(&xor).unwrap(), "{what}");
    }
    let mut padded = offset.to_vec();
    padded.push((1, 1));
    let mut longer = offset.to_vec();
    longer.push((0, 8));
    for (fields, what) in [
        (&[(3, 2), (0, 14)][..], "node code 3"),
        (&[(2, 2), (1, 2), (0, 5), (1, 2), (0, 5)][..], "pairs in no bits"),
        (&padded[..], "a padding bit set"),
        (&longer[..], "a stream longer than its pairs"),
        (&offset[..2], "a stream shorter than its pairs"),
    ] {
        refused(&two_leaf_file(2, fields), &vtree, what);
    }
    // More pairs than the stream has bits.
    refused(&two_leaf_file(100, &offset), &vtree, "a hundred pairs in two bytes");
}

#[test]
fn a_level_over_one_node_levels_holds_one_pair() {
    // x1 and x2 and x3 and x4 over ((x1 x2) (x3 x4)): each child of the root
    // has one node, so the root's one pair, (0, 0), takes no bits.
    let vtree = Arc::new(Vtree::balanced(4));
    let literal = |v: i32| Tdd::clause(&vtree, [v]).unwrap();
    let low = crate::and(literal(1), literal(2)).unwrap();
    let high = crate::and(literal(3), literal(4)).unwrap();
    let f = crate::and(low, high).unwrap();
    let bytes = binary(&f);
    // The root's section ends the body: one node, none of them multi-pair,
    // one pair, a stream of no bytes.
    let at = bytes.len() - 8 - 4;
    assert_eq!(bytes[at..bytes.len() - 8], [1, 0, 1, 0]);
    let back = read_tdd_binary(&mut bytes.as_slice(), &vtree).unwrap();
    assert!(back.equivalent(&f).unwrap());
    let max_live = [0xFF, 0xFF, 0xFF, 0xFF, 0x07];
    for (section, what) in [
        (vec![2, 0, 2, 0], "two nodes of the one pair"),
        ([&max_live[..], &[0], &max_live[..], &[0]].concat(), "2^31 - 1 nodes in no bytes"),
        (vec![1, 1, 3, 1, 1, 1, 0], "a node of the one pair three times"),
    ] {
        let mut changed = bytes[..at].to_vec();
        changed.extend(section);
        changed.extend([0; 8]);
        reseal(&mut changed);
        refused(&changed, &vtree, what);
    }
}

#[test]
fn unreachable_nodes_are_dropped_as_the_text_writer_drops_them() {
    use crate::diagram::{ChildPair, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
    let eng = crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, right) = vtree.children(vtree.root());
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    // Two nodes on each child level and two at the root; the output reaches
    // the second of each.
    let _ = b.push(&eng, left, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let l = b.push(&eng, left, &[ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let _ = b.push(&eng, right, &[ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let r = b.push(&eng, right, &[ChildPair::new(ONE_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    let _ = b.push(&eng, vtree.root(), &[ChildPair::new(l, ONE_LEAF_IDX)]).unwrap();
    let out = b.push(&eng, vtree.root(), &[ChildPair::new(l, r)]).unwrap();
    let f = b.finish(TddNodeId { vtree: vtree.root(), local: out }).unwrap();
    let back = read_tdd_binary(&mut binary(&f).as_slice(), &vtree).unwrap();
    assert_same_levels(&back, &read_tdd(&mut text(&f).as_slice(), &vtree).unwrap(), "pruned");
    assert_eq!(back.node_count(), 3);
    assert!(back.equivalent(&f).unwrap());
}

#[test]
fn several_diagrams_follow_each_other_in_one_stream() {
    let vtree = Arc::new(Vtree::balanced(4));
    let a = compile_clauses(&vtree, &[vec![1, -2], vec![3]]);
    let b = compile_clauses(&vtree, &[vec![-1, 4]]);
    let mut stream = binary(&a);
    stream.extend(binary(&b));
    stream.extend(binary(&Tdd::zero(&vtree)));
    let mut r = stream.as_slice();
    assert!(read_tdd_binary(&mut r, &vtree).unwrap().equivalent(&a).unwrap());
    assert!(read_tdd_binary(&mut r, &vtree).unwrap().equivalent(&b).unwrap());
    assert!(read_tdd_binary(&mut r, &vtree).unwrap().is_zero());
    assert!(r.is_empty());
    refused(r, &vtree, "an empty stream");
}

#[test]
fn a_file_holds_one_diagram() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, -3]).unwrap();
    let path = std::env::temp_dir().join(format!("tididi-binary-trailing-{}.tddb", std::process::id()));
    let mut bytes = binary(&f);
    crate::io::save_tdd_binary(&f, &path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert!(crate::io::load_tdd_binary(&path, &vtree).unwrap().equivalent(&f).unwrap());
    bytes.push(0);
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(crate::io::load_tdd_binary(&path, &vtree), Err(IoError::Format(_))));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn every_truncation_is_refused() {
    let vtree = Arc::new(Vtree::balanced(6));
    let f = compile_clauses(&vtree, &[vec![1, 2, -3], vec![-2, 4], vec![5, -6, 1]]);
    let bytes = binary(&f);
    for end in 0..bytes.len() {
        refused(&bytes[..end], &vtree, &format!("the first {end} bytes"));
    }
}

#[test]
fn every_flipped_bit_is_refused() {
    let vtree = Arc::new(Vtree::random(5, 3));
    let f = compile_clauses(&vtree, &[vec![1, -2], vec![2, 3, -4], vec![-5, 1]]);
    let bytes = binary(&f);
    for bit in 0..bytes.len() * 8 {
        let mut bad = bytes.clone();
        bad[bit / 8] ^= 1 << (bit % 8);
        refused(&bad, &vtree, &format!("bit {bit} flipped"));
    }
}

#[test]
fn the_header_is_checked_field_by_field() {
    let vtree = Arc::new(Vtree::balanced(3));
    let bytes = binary(&Tdd::clause(&vtree, [2, 3]).unwrap());
    let mut text_file = Vec::new();
    write_tdd(&mut text_file, &Tdd::clause(&vtree, [2, 3]).unwrap()).unwrap();
    refused(&text_file, &vtree, "a text file");
    for (at, what) in [(0, "magic"), (8, "version"), (12, "byte-order mark"), (16, "body length")] {
        let mut bad = bytes.clone();
        bad[at] ^= 0x40;
        refused(&bad, &vtree, what);
    }
    let mut big_endian = bytes.clone();
    big_endian[12..16].reverse();
    refused(&big_endian, &vtree, "a big-endian mark");
    let mut huge = bytes.clone();
    huge[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    refused(&huge, &vtree, "a length past memory");
}

#[test]
fn another_vtree_is_refused() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![-3, 4]]);
    let bytes = binary(&f);
    let order = [VarId(2), VarId(1), VarId(3), VarId(4)];
    for other in [
        Vtree::linear(4),
        Vtree::balanced(5),
        Vtree::balanced_over(&order).unwrap(),
        Vtree::linear_from_order(&order).unwrap(),
    ] {
        refused(&bytes, &Arc::new(other), "another vtree");
    }
    // The same tree under another numbering is the same vtree.
    let same = Arc::new(Vtree::from_text(&vtree.to_text()).unwrap());
    assert!(read_tdd_binary(&mut bytes.as_slice(), &same).unwrap().equivalent(&compile_clauses(&same, &[vec![1, 2], vec![-3, 4]])).unwrap());
    // The false diagram stores its vtree too.
    refused(&binary(&Tdd::zero(&vtree)), &Arc::new(Vtree::linear(4)), "false over another vtree");
}

/// Corrupt the body, rewrite the length and checksum, and read: whatever
/// comes back is refused or is a diagram whose storage the builder checked.
#[test]
fn resealed_corruption_is_refused_or_read_without_a_panic() {
    let mut rng = Lcg::new(11);
    let mut read = 0;
    for case in 0..40u64 {
        let num_vars = 2 + rng.below(9) as u32;
        let vtree = Arc::new(Vtree::random(num_vars, case));
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
        let bytes = binary(&compile_clauses(&vtree, &clauses));
        for _ in 0..200 {
            let mut bad = bytes.clone();
            let body = 24..bad.len() - 8;
            match rng.below(4) {
                0 => {
                    let at = body.start + rng.below(body.len() as u64) as usize;
                    bad[at] = rng.next_u64() as u8;
                }
                1 => {
                    let at = body.start + rng.below(body.len() as u64) as usize;
                    bad[at] ^= 1 << rng.below(8);
                }
                2 => {
                    let at = body.start + rng.below(body.len() as u64) as usize;
                    bad.remove(at);
                }
                _ => {
                    let at = body.start + rng.below(body.len() as u64 + 1) as usize;
                    bad.insert(at, rng.next_u64() as u8);
                }
            }
            reseal(&mut bad);
            match read_tdd_binary(&mut bad.as_slice(), &vtree) {
                Ok(f) => {
                    read += 1;
                    f.model_count().unwrap();
                }
                Err(IoError::Format(_)) => {}
                Err(other) => panic!("an in-memory read failed with {other:?}"),
            }
        }
    }
    assert!(read < 40 * 200, "every corruption was read");
}

#[test]
fn a_marginal_diagram_is_not_written() {
    let eng = crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = crate::test_helpers::compile_clauses_on(&eng, &vtree, &[vec![1, 2], vec![-2, 3]]);
    let (left, _) = vtree.children(vtree.root());
    let (target, _) = vtree.children(left);
    crate::marginal::marginalize_batch(&eng, &mut f, &[target], &vtree).unwrap();
    let mut bytes = Vec::new();
    assert!(matches!(write_tdd_binary(&mut bytes, &f), Err(IoError::Format(_))));
    assert!(bytes.is_empty());
    let path = std::env::temp_dir().join(format!("tididi-binary-marginal-{}.tddb", std::process::id()));
    assert!(matches!(crate::io::save_tdd_binary(&f, &path), Err(IoError::Format(_))));
    assert!(!path.exists());
}
