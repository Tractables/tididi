//! Reader and writer for the [binary format](crate::io#binary-format).
//!
//! The writer encodes the whole file in memory, since the header carries the
//! body's length and the trailer its checksum. The reader takes the whole file
//! before interpreting any of it: the checksum is checked first, then every
//! count is bounded by the bytes left to hold it, so a damaged or hostile file
//! is refused before it can size an allocation.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, EncodedNode, LevelCounts, NodeIdx, Tdd, TddLevel, TddNodeId, LEAF_WIDTH, ZERO};
use crate::value::{CountRead, CountVec};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

use super::numbering::{number_reachable, OMITTED};

use super::IoError;

/// The first eight bytes of every binary diagram file.
const MAGIC: [u8; 8] = *b"\x89TDD\r\n\x1a\n";

/// The binary format version the writer emits for a diagram without level
/// counts; the reader accepts it and [`COUNTS_FORMAT_VERSION`].
const BINARY_FORMAT_VERSION: u32 = 1;

/// The version of a file that also holds the level counts the diagram keeps
/// ([`Tdd::attach_level_counts`](crate::Tdd::attach_level_counts)).
const COUNTS_FORMAT_VERSION: u32 = 2;

/// Written little-endian after the version; a file whose four bytes read
/// otherwise was not written by a little-endian encoder of this format.
const BYTE_ORDER_MARK: u32 = 0x0A0B_0C0D;

/// Magic, version, byte-order mark and body length.
const HEADER_BYTES: usize = 24;

/// The checksum after the body.
const TRAILER_BYTES: usize = 8;

/// A node's pairs as stored: each left index, then each right index.
const PLAIN: u32 = 0;
/// A node's pairs with the right indices as offsets from their smallest.
const OFFSET: u32 = 1;
/// A node's pairs in ascending order of left index, each left after the first
/// as its gap from the previous one less one, the right indices as in `OFFSET`.
const ASCENDING: u32 = 2;
/// The bits of a field that holds another field's width.
const FIELD_WIDTH_BITS: u32 = 5;

/// Write a diagram to a file in the binary format, creating or truncating the
/// file.
///
/// The file holds the same structure as [`save_tdd`](super::save_tdd) in a
/// fraction of the bytes, and [`load_tdd_binary`] reads it back without
/// parsing text.
///
/// # Errors
///
/// [`IoError::Format`] if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]); nothing is written and no file is created in
/// that case. [`IoError::Io`] if the file cannot be created or written; a
/// partial file may then be left behind.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{load_tdd_binary, save_tdd_binary};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let path = std::env::temp_dir().join("tididi-doc-save.tddb");
/// save_tdd_binary(&f, &path)?;
/// let restored = load_tdd_binary(&path, &vtree)?;
/// assert!(restored.equivalent(&f)?);
/// # tididi::test_helpers::assert_canonical(&f);
/// # tididi::test_helpers::assert_canonical(&restored);
/// # std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn save_tdd_binary(f: &Tdd, path: impl AsRef<Path>) -> Result<(), IoError> {
    super::reject_marginal_levels(f, "save_tdd_binary")?;
    std::fs::write(path.as_ref(), encode(f))?;
    Ok(())
}

/// Write a diagram in the binary format to any writer. `w` is not flushed.
///
/// The bytes are those [`save_tdd_binary`] writes to a file. Like the text
/// format, the binary format stores the reachable nodes only, renumbered in
/// their level order with each node's pairs in their stored order, and no
/// weights: a diagram carrying a weight store reads back in integer mode. It
/// stores the level counts the diagram keeps
/// ([`Tdd::attach_level_counts`](crate::Tdd::attach_level_counts)), each
/// internal level's where every one of them fits 128 bits, in format
/// version 2; a diagram with none to store, such as the false diagram or one
/// over a single variable, is written in version 1 and reads back without
/// them.
///
/// # Errors
///
/// [`IoError::Format`] if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]); nothing is written to `w` in that case.
/// [`IoError::Io`] if a write to `w` fails.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{read_tdd_binary, write_tdd, write_tdd_binary};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let (mut binary, mut text) = (Vec::new(), Vec::new());
/// write_tdd_binary(&mut binary, &f)?;
/// write_tdd(&mut text, &f)?;
/// assert!(binary.len() < text.len());
/// let restored = read_tdd_binary(&mut binary.as_slice(), &vtree)?;
/// assert_eq!(restored.model_count()?, f.model_count()?);
/// # tididi::test_helpers::assert_canonical(&f);
/// # tididi::test_helpers::assert_canonical(&restored);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_tdd_binary<W: Write>(w: &mut W, tdd: &Tdd) -> Result<(), IoError> {
    super::reject_marginal_levels(tdd, "write_tdd_binary")?;
    w.write_all(&encode(tdd))?;
    Ok(())
}

/// Read a diagram from a binary file written by [`save_tdd_binary`].
///
/// The diagram shares the supplied vtree; see [`read_tdd_binary`] for what is
/// checked. A file holds one diagram: bytes after its checksum are refused.
///
/// # Errors
///
/// Returns [`IoError::Io`] if the file cannot be read, or the format errors
/// from [`read_tdd_binary`].
pub fn load_tdd_binary(path: impl AsRef<Path>, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    decode(&std::fs::read(path.as_ref())?, vtree)
}

/// Read one diagram in the binary format from any reader, over `vtree`.
///
/// Reads exactly the diagram's bytes, so several diagrams can follow each
/// other in one stream. Supply the vtree the diagram was written over: the
/// file stores that vtree's structure, and the reader compares it with
/// `vtree`'s, leaf variables included, regardless of either's in-memory
/// numbering. The result shares the supplied `Arc<Vtree>`, has no attached
/// weights, keeps the level counts a version 2 file holds, and has the
/// nodes, node order and pair order that
/// [`read_tdd`](super::read_tdd) gives for the same diagram's text; reading
/// does not minimize.
///
/// Nothing in the file is trusted before it is checked: the magic, version and
/// byte-order mark, then the checksum over all the bytes, then every count
/// against the bytes left to hold it and every pair side against its child
/// level. As for the text format, the reader validates the encoding, not
/// semantic determinism; it relies on determinism only to refuse more than
/// one pair in a level over two one-node levels, whose pairs take no bits.
/// The level counts of a version 2 file are bounded, not recomputed, and are
/// trusted from there on ([format](crate::io#binary-format)).
///
/// # Errors
///
/// Returns [`IoError::Format`] for a file that is not in this format, an
/// unsupported version, a truncated file, a checksum mismatch, a vtree that is
/// not `vtree`, or a malformed or out-of-range record. Returns [`IoError::Io`]
/// if reading fails.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{read_tdd_binary, write_tdd_binary, IoError};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let mut bytes = Vec::new();
/// write_tdd_binary(&mut bytes, &f)?;
///
/// // A damaged byte fails the checksum.
/// let last = bytes.len() - 1;
/// bytes[last] ^= 1;
/// let result = read_tdd_binary(&mut bytes.as_slice(), &vtree);
/// assert!(matches!(result, Err(IoError::Format(_))));
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_tdd_binary<R: Read>(r: &mut R, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let mut bytes = vec![0u8; HEADER_BYTES];
    read_full(r, &mut bytes)?;
    let (total, _) = check_header(&bytes)?;
    // Grows only as bytes arrive, so a damaged length cannot size an allocation.
    r.take((total - HEADER_BYTES) as u64).read_to_end(&mut bytes)?;
    decode(&bytes, vtree)
}

/// Fill `buf` from `r`, refusing a stream that ends first.
fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), IoError> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => return Err(truncated(filled, buf.len())),
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn malformed(what: impl std::fmt::Display) -> IoError {
    IoError::Format(format!("tdd binary: {what}"))
}

fn truncated(have: usize, want: usize) -> IoError {
    malformed(format!("the file is truncated: {have} of its {want} bytes are present"))
}

// ── Header and checksum ─────────────────────────────────────────────────────

/// Check the magic, version and byte-order mark, and return the file's whole
/// length as the header declares it, and its version.
fn check_header(bytes: &[u8]) -> Result<(usize, u32), IoError> {
    if bytes.len() < MAGIC.len() || bytes[..MAGIC.len()] != MAGIC {
        return Err(malformed("the file does not start with the binary diagram magic bytes"));
    }
    if bytes.len() < HEADER_BYTES {
        return Err(truncated(bytes.len(), HEADER_BYTES + TRAILER_BYTES));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("four bytes"));
    if version != BINARY_FORMAT_VERSION && version != COUNTS_FORMAT_VERSION {
        return Err(malformed(format!(
            "the file is binary format version {version}; this reader understands versions \
             {BINARY_FORMAT_VERSION} and {COUNTS_FORMAT_VERSION}"
        )));
    }
    let mark = u32::from_le_bytes(bytes[12..16].try_into().expect("four bytes"));
    if mark != BYTE_ORDER_MARK {
        return Err(malformed(format!(
            "byte-order mark {mark:#010x} is not {BYTE_ORDER_MARK:#010x}; the file is not \
             little-endian"
        )));
    }
    let body = u64::from_le_bytes(bytes[16..24].try_into().expect("eight bytes"));
    let total = usize::try_from(body)
        .ok()
        .and_then(|b| b.checked_add(HEADER_BYTES + TRAILER_BYTES))
        .ok_or_else(|| malformed(format!("body length {body} cannot be addressed")))?;
    Ok((total, version))
}

/// Check the header, the length and the checksum of one whole file, then
/// parse its body.
fn decode(bytes: &[u8], vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let (total, version) = check_header(bytes)?;
    if bytes.len() < total {
        return Err(truncated(bytes.len(), total));
    }
    if bytes.len() > total {
        return Err(malformed(format!(
            "{} bytes follow the checksum, which ends the diagram",
            bytes.len() - total
        )));
    }
    let (covered, trailer) = bytes.split_at(total - TRAILER_BYTES);
    let stored = u64::from_le_bytes(trailer.try_into().expect("eight bytes"));
    let computed = xxh64(covered);
    if stored != computed {
        return Err(malformed(format!(
            "checksum mismatch: the file records {stored:#018x}, its bytes hash to \
             {computed:#018x}"
        )));
    }
    parse_body(&covered[HEADER_BYTES..], vtree, version == COUNTS_FORMAT_VERSION)
}

/// `XXH64` with seed 0, the checksum the trailer records.
pub(super) fn xxh64(data: &[u8]) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    const P5: u64 = 0x27D4_EB2F_1656_67C5;
    fn round(acc: u64, input: u64) -> u64 {
        acc.wrapping_add(input.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
    }
    fn merge(acc: u64, val: u64) -> u64 {
        (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
    }
    let word = |i: usize| u64::from_le_bytes(data[i..i + 8].try_into().expect("eight bytes"));
    let len = data.len();
    let mut i = 0;
    let mut h = if len >= 32 {
        let mut v = [P1.wrapping_add(P2), P2, 0, 0u64.wrapping_sub(P1)];
        while i + 32 <= len {
            for (lane, acc) in v.iter_mut().enumerate() {
                *acc = round(*acc, word(i + 8 * lane));
            }
            i += 32;
        }
        let mut h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        for lane in v {
            h = merge(h, lane);
        }
        h
    } else {
        P5
    };
    h = h.wrapping_add(len as u64);
    while i + 8 <= len {
        h ^= round(0, word(i));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }
    if i + 4 <= len {
        let half = u32::from_le_bytes(data[i..i + 4].try_into().expect("four bytes"));
        h ^= u64::from(half).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }
    for &byte in &data[i..] {
        h ^= u64::from(byte).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

// ── Shared layout ───────────────────────────────────────────────────────────

/// The vtree's nodes in structural postorder from the root: left subtree,
/// right subtree, then the node. Defined by the tree alone, so a writer and a
/// reader holding the same tree under different numberings agree on it.
pub(super) fn postorder(vtree: &Vtree) -> Vec<VtreeIdx> {
    let mut out = Vec::with_capacity(vtree.num_nodes());
    let mut stack = vec![(vtree.root(), false)];
    while let Some((t, expanded)) = stack.pop() {
        match *vtree.node(t) {
            VtreeNode::Internal { left, right, .. } if !expanded => {
                stack.push((t, true));
                stack.push((right, false));
                stack.push((left, false));
            }
            _ => out.push(t),
        }
    }
    out
}

/// Bits that hold every index below `width`: none when there is at most one.
fn index_bits(width: usize) -> u32 {
    if width <= 1 { 0 } else { usize::BITS - (width - 1).leading_zeros() }
}

/// Bits that hold every value up to `v`: none for zero.
fn value_bits(v: u32) -> u32 {
    u32::BITS - v.leading_zeros()
}

// ── Writer ──────────────────────────────────────────────────────────────────

/// The whole file: header, body and checksum.
fn encode(tdd: &Tdd) -> Vec<u8> {
    let vtree = &tdd.vtree;
    // The counts the diagram keeps, a level's where it has no overflow slot.
    let counts = tdd.levels.counts().filter(|_| !tdd.is_zero());
    let column = |t: VtreeIdx| counts.and_then(|c| c.column(t)).filter(|c| (0..c.len()).all(|i| matches!(c.get(i), CountRead::Fast(_))));
    let with_counts = vtree.internal_bottomup().any(|(t, _, _)| column(t).is_some());
    let version = if with_counts { COUNTS_FORMAT_VERSION } else { BINARY_FORMAT_VERSION };
    let mut out = Vec::with_capacity(HEADER_BYTES + TRAILER_BYTES + 64 + tdd.pair_count() * 5);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&BYTE_ORDER_MARK.to_le_bytes());
    out.extend_from_slice(&[0; 8]);

    let order = postorder(vtree);
    push_varint(&mut out, u64::from(vtree.num_leaves()));
    push_varint(&mut out, vtree.num_nodes() as u64);
    for &t in &order {
        match *vtree.node(t) {
            VtreeNode::Leaf { var, .. } => push_varint(&mut out, u64::from(var.0)),
            VtreeNode::Internal { .. } => push_varint(&mut out, 0),
        }
    }
    if tdd.is_zero() {
        push_varint(&mut out, 0);
    } else {
        let remap = encode_levels(tdd, &order, &mut out, &mut [0; 3]);
        if with_counts {
            // Each internal level's counts, of the nodes written, in their order.
            for &t in &order {
                if vtree.node(t).is_leaf() {
                    continue;
                }
                let Some(c) = column(t) else {
                    push_varint(&mut out, 0);
                    continue;
                };
                push_varint(&mut out, 1);
                for (i, &to) in remap[t.idx()].iter().enumerate() {
                    if to != OMITTED {
                        let CountRead::Fast(v) = c.get(i) else { unreachable!("a column with an overflow slot is not written") };
                        push_varint128(&mut out, v);
                    }
                }
            }
        }
    }

    let body = (out.len() - HEADER_BYTES) as u64;
    out[16..24].copy_from_slice(&body.to_le_bytes());
    let sum = xxh64(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

/// The output and one section per internal level, children first. Unreachable
/// nodes are dropped and the rest renumbered in level order, as the text
/// writer does; a leaf level keeps its three implicit indices. `codes`
/// counts the multi-pair nodes written in each node code. Returns each
/// level's renumbering, [`OMITTED`] for a node not written.
pub(super) fn encode_levels(tdd: &Tdd, order: &[VtreeIdx], out: &mut Vec<u8>, codes: &mut [u64; 3]) -> Vec<Vec<u32>> {
    let vtree = &tdd.vtree;
    let numbering = number_reachable(tdd);
    let out_local = numbering[tdd.output.vtree.idx()].local[tdd.output.local.idx()];
    debug_assert_ne!(out_local, OMITTED, "the output is reachable");
    push_varint(out, u64::from(out_local) + 1);

    let structural = ChildDecoder::structural();
    for &t in order {
        let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
        let level = tdd.level(t);
        let local = &numbering[t.idx()].local;
        let live = || level.internal_inputs_iter().filter(|(i, _)| local[*i] != OMITTED);
        let (mut nodes, mut multi, mut pairs) = (0u64, 0u64, 0u64);
        for (_, node_pairs) in live() {
            nodes += 1;
            multi += u64::from(node_pairs.len() > 1);
            pairs += node_pairs.len() as u64;
        }
        let (left_map, right_map) = (&numbering[left.idx()].local, &numbering[right.idx()].local);
        let (left_bits, right_bits) = (index_bits(numbering[left.idx()].width), index_bits(numbering[right.idx()].width));
        let mut stream = Vec::new();
        let mut bits = BitWriter::new(&mut stream);
        let mut mapped: Vec<(u32, u32)> = Vec::new();
        for (_, node_pairs) in live() {
            mapped.clear();
            mapped.extend(node_pairs.map(|pair| {
                (left_map[structural.node(pair.left).idx()], right_map[structural.node(pair.right).idx()])
            }));
            match *mapped.as_slice() {
                [(l, r)] => {
                    bits.put(l, left_bits);
                    bits.put(r, right_bits);
                }
                _ => codes[write_node(&mut bits, &mapped, left_bits, right_bits) as usize] += 1,
            }
        }
        bits.finish();

        push_varint(out, nodes);
        push_varint(out, multi);
        push_varint(out, pairs);
        push_varint(out, stream.len() as u64);
        if multi > 0 {
            let mut bits = BitWriter::new(out);
            for (_, node_pairs) in live() {
                bits.put(u32::from(node_pairs.len() > 1), 1);
            }
            bits.finish();
            for (_, node_pairs) in live() {
                if node_pairs.len() > 1 {
                    push_varint(out, node_pairs.len() as u64 - 2);
                }
            }
        }
        out.extend_from_slice(&stream);
    }
    numbering.into_iter().map(|n| n.local).collect()
}

/// A node with two or more pairs, in whichever of the three node codes takes
/// the fewest bits, the lowest code on a tie. Each code is chosen only where
/// every pair takes at least one bit, which bounds a level's pairs by its
/// stream's bits.
fn write_node(bits: &mut BitWriter<'_>, pairs: &[(u32, u32)], left_bits: u32, right_bits: u32) -> u32 {
    let n = pairs.len() as u64;
    let (low, high) = pairs.iter().fold((u32::MAX, 0), |(low, high), &(_, r)| (low.min(r), high.max(r)));
    let offset_bits = value_bits(high - low);
    let ascending = pairs.windows(2).all(|w| w[0].0 < w[1].0);
    let gap_bits = pairs.windows(2).map(|w| value_bits(w[1].0.wrapping_sub(w[0].0).wrapping_sub(1))).max().unwrap_or(0);
    let plain = n * u64::from(left_bits + right_bits);
    let offset = (left_bits + offset_bits >= 1)
        .then(|| u64::from(right_bits + FIELD_WIDTH_BITS) + n * u64::from(left_bits + offset_bits));
    let gapped = (ascending && gap_bits + offset_bits >= 1).then(|| {
        u64::from(left_bits + right_bits + 2 * FIELD_WIDTH_BITS) + n * u64::from(offset_bits) + (n - 1) * u64::from(gap_bits)
    });
    let code = match (offset, gapped) {
        (_, Some(g)) if g < plain && offset.is_none_or(|o| g < o) => ASCENDING,
        (Some(o), _) if o < plain => OFFSET,
        _ => PLAIN,
    };
    bits.put(code, 2);
    match code {
        PLAIN => {
            for &(l, r) in pairs {
                bits.put(l, left_bits);
                bits.put(r, right_bits);
            }
        }
        OFFSET => {
            bits.put(low, right_bits);
            bits.put(offset_bits, FIELD_WIDTH_BITS);
            for &(l, r) in pairs {
                bits.put(l, left_bits);
                bits.put(r - low, offset_bits);
            }
        }
        _ => {
            bits.put(pairs[0].0, left_bits);
            bits.put(gap_bits, FIELD_WIDTH_BITS);
            bits.put(low, right_bits);
            bits.put(offset_bits, FIELD_WIDTH_BITS);
            bits.put(pairs[0].1 - low, offset_bits);
            for w in pairs.windows(2) {
                bits.put(w[1].0 - w[0].0 - 1, gap_bits);
                bits.put(w[1].1 - low, offset_bits);
            }
        }
    }
    code
}

/// Append `v` as an unsigned `LEB128` varint: seven bits a byte, low first.
fn push_varint(out: &mut Vec<u8>, v: u64) {
    push_varint128(out, u128::from(v));
}

/// [`push_varint`] for a value up to 128 bits, a count.
fn push_varint128(out: &mut Vec<u8>, mut v: u128) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Bit-packs values into a byte buffer, least significant bit first.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    held: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        BitWriter { out, acc: 0, held: 0 }
    }

    /// Append the low `bits` bits of `v`, which holds no higher bit; `bits`
    /// is at most 32.
    #[inline]
    fn put(&mut self, v: u32, bits: u32) {
        debug_assert!(bits <= 32 && u64::from(v) >> bits == 0, "{v} does not fit {bits} bits");
        self.acc |= u64::from(v) << self.held;
        self.held += bits;
        if self.held >= 32 {
            self.out.extend_from_slice(&(self.acc as u32).to_le_bytes());
            self.acc >>= 32;
            self.held -= 32;
        }
    }

    /// Write the bits still held, padding the last byte with zeros.
    fn finish(self) {
        let bytes = self.held.div_ceil(8) as usize;
        self.out.extend_from_slice(&self.acc.to_le_bytes()[..bytes]);
    }
}

// ── Reader ──────────────────────────────────────────────────────────────────

/// The body after the header, read front to back.
struct Body<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Body<'a> {
    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }

    /// One `LEB128` varint, refusing one longer than a `u64` holds.
    fn varint(&mut self, what: &str) -> Result<u64, IoError> {
        let start = self.at;
        let v = self.varint_bits(what, 64)?;
        u64::try_from(v).map_err(|_| malformed(format!("body byte {start}: {what} does not fit 64 bits")))
    }

    /// One `LEB128` varint of at most `bits` bits, 64 or 128.
    fn varint_bits(&mut self, what: &str, bits: u32) -> Result<u128, IoError> {
        let start = self.at;
        let mut v = 0u128;
        for shift in (0..bits).step_by(7) {
            let Some(&byte) = self.bytes.get(self.at) else {
                return Err(malformed(format!("body byte {start}: {what} runs past the end of the body")));
            };
            self.at += 1;
            let low = u128::from(byte & 0x7f);
            if shift + 7 > bits && low >> (bits - shift) != 0 {
                break;
            }
            v |= low << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(malformed(format!("body byte {start}: {what} does not fit {bits} bits")))
    }

    /// A varint that counts something, refused above `max`.
    fn count(&mut self, what: &str, max: u64) -> Result<usize, IoError> {
        let start = self.at;
        let v = self.varint(what)?;
        if v > max {
            return Err(malformed(format!("body byte {start}: {what} {v} is more than the {max} the file can hold")));
        }
        Ok(v as usize)
    }

    /// The next `n` bytes.
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], IoError> {
        if n > self.left() {
            return Err(malformed(format!(
                "body byte {}: {what} needs {n} bytes, {} are left",
                self.at,
                self.left()
            )));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }
}

/// Reads one level's stream of pairs, bit-packed least significant bit
/// first. It keeps the largest side of each kind, which the caller compares
/// with the child widths once, after the last pair.
struct PairReader<'a> {
    bytes: &'a [u8],
    /// The next bit to read.
    at: usize,
    left_bits: u32,
    right_bits: u32,
    left_max: u64,
    right_max: u64,
}

impl PairReader<'_> {
    /// At least 57 bits from bit `at` on, reading past the end of the bytes
    /// as zeros; the caller compares the bits read with the bytes at the end.
    #[inline(always)]
    fn window(&self) -> u64 {
        let byte = self.at / 8;
        let word = match self.bytes.get(byte..byte + 8) {
            Some(eight) => u64::from_le_bytes(eight.try_into().expect("eight bytes")),
            None => {
                let tail = self.bytes.get(byte..).unwrap_or_default();
                let mut eight = [0u8; 8];
                eight[..tail.len()].copy_from_slice(tail);
                u64::from_le_bytes(eight)
            }
        };
        word >> (self.at % 8)
    }

    /// The next `bits` bits, at most 32.
    #[inline(always)]
    fn take(&mut self, bits: u32) -> u32 {
        let v = self.window() & ((1u64 << bits) - 1);
        self.at += bits as usize;
        v as u32
    }

    /// The next two fields, `a` bits then `b` bits: one window when both
    /// fit it.
    #[inline(always)]
    fn take2(&mut self, a: u32, b: u32) -> (u32, u32) {
        if a + b <= 57 {
            let w = self.window();
            self.at += (a + b) as usize;
            ((w & ((1u64 << a) - 1)) as u32, ((w >> a) & ((1u64 << b) - 1)) as u32)
        } else {
            (self.take(a), self.take(b))
        }
    }

    /// A pair of sides, noted for the range check.
    #[inline(always)]
    fn record(&mut self, left: u64, right: u64) -> ChildPair {
        self.left_max = self.left_max.max(left);
        self.right_max = self.right_max.max(right);
        ChildPair::new(EncodedChildRef::from_raw(left as u32), EncodedChildRef::from_raw(right as u32))
    }

    /// The next pair, as stored.
    #[inline(always)]
    fn pair(&mut self) -> ChildPair {
        let (l, r) = self.take2(self.left_bits, self.right_bits);
        self.record(u64::from(l), u64::from(r))
    }

    /// A node of `len` pairs, two or more, in the code its first two bits
    /// name, appended to `out`.
    fn node(&mut self, len: usize, out: &mut Vec<ChildPair>) -> Result<(), IoError> {
        match self.take(2) {
            PLAIN => {
                for _ in 0..len {
                    let pair = self.pair();
                    out.push(pair);
                }
            }
            OFFSET => {
                let low = u64::from(self.take(self.right_bits));
                let offset_bits = self.take(FIELD_WIDTH_BITS);
                if self.left_bits + offset_bits == 0 {
                    return Err(malformed("a node's pairs are coded in no bits"));
                }
                for _ in 0..len {
                    let (l, offset) = self.take2(self.left_bits, offset_bits);
                    let pair = self.record(u64::from(l), low + u64::from(offset));
                    out.push(pair);
                }
            }
            ASCENDING => {
                let mut l = u64::from(self.take(self.left_bits));
                let gap_bits = self.take(FIELD_WIDTH_BITS);
                let low = u64::from(self.take(self.right_bits));
                let offset_bits = self.take(FIELD_WIDTH_BITS);
                if gap_bits + offset_bits == 0 {
                    return Err(malformed("a node's pairs are coded in no bits"));
                }
                let offset = self.take(offset_bits);
                let pair = self.record(l, low + u64::from(offset));
                out.push(pair);
                for _ in 1..len {
                    let (gap, offset) = self.take2(gap_bits, offset_bits);
                    l += u64::from(gap) + 1;
                    let pair = self.record(l, low + u64::from(offset));
                    out.push(pair);
                }
            }
            code => return Err(malformed(format!("node code {code} is not defined"))),
        }
        Ok(())
    }
}

/// Check the stored vtree against `vtree`, then read the levels and seat the
/// diagram.
fn parse_body(bytes: &[u8], vtree: &Arc<Vtree>, with_counts: bool) -> Result<Tdd, IoError> {
    let mut body = Body { bytes, at: 0 };
    let leaves = body.varint("leaf count")?;
    let nodes = body.varint("vtree node count")?;
    if nodes != vtree.num_nodes() as u64 || leaves != u64::from(vtree.num_leaves()) {
        return Err(malformed(format!(
            "the file's vtree has {leaves} leaves and {nodes} nodes, the vtree read against \
             has {} and {}",
            vtree.num_leaves(),
            vtree.num_nodes()
        )));
    }
    let order = postorder(vtree);
    for (k, &t) in order.iter().enumerate() {
        let token = body.varint("vtree node")?;
        let expected = match *vtree.node(t) {
            VtreeNode::Leaf { var, .. } => u64::from(var.0),
            VtreeNode::Internal { .. } => 0,
        };
        if token != expected {
            let describe = |v: u64| if v == 0 { "an internal node".to_string() } else { format!("the leaf of variable {v}") };
            return Err(malformed(format!(
                "vtree node {k} in postorder is {} in the file and {} in the vtree read against",
                describe(token),
                describe(expected)
            )));
        }
    }

    let root = vtree.root();
    let output = body.varint("output")?;
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    let mut width = vec![LEAF_WIDTH; vtree.num_nodes()];
    if output != 0 {
        for (k, &t) in order.iter().enumerate() {
            let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
            let level = read_level(&mut body, width[left.idx()], width[right.idx()]).map_err(|e| match e {
                IoError::Format(m) => IoError::Format(format!("{m} (the level of vtree node {k} in postorder)")),
                other => other,
            })?;
            width[t.idx()] = level.nodes().len();
            levels[t.idx()] = level;
        }
    }
    let counts = if with_counts && output != 0 { Some(read_counts(&mut body, vtree, &order, &width)?) } else { None };
    if body.left() != 0 {
        return Err(malformed(format!("{} bytes follow the last record of the body", body.left())));
    }
    let local = match output {
        0 => ZERO,
        _ if output - 1 < width[root.idx()] as u64 => NodeIdx((output - 1) as u32),
        _ => {
            return Err(malformed(format!(
                "output node {} is past the root level's {} nodes",
                output - 1,
                width[root.idx()]
            )));
        }
    };
    let mut tdd = crate::diagram::TddBuilder::from_levels(Arc::clone(vtree), levels, None)
        .finish(TddNodeId { vtree: root, local })
        .map_err(|e| malformed(format!("the records do not form a diagram: {e}")))?;
    if let Some(counts) = counts {
        tdd.levels.keep_counts(counts);
    }
    Ok(tdd)
}

/// The counts section of a version 2 file: per internal level in `order`,
/// none, or one count per node of its `width`. Each count is checked to be
/// at most the assignments of the variables under its level; the counts are
/// otherwise kept as written.
fn read_counts(body: &mut Body<'_>, vtree: &Vtree, order: &[VtreeIdx], width: &[usize]) -> Result<LevelCounts, IoError> {
    let mut leaves = vec![0u32; vtree.num_nodes()];
    let mut counts = LevelCounts::from_columns(vec![None; vtree.num_nodes()]);
    for (k, &t) in order.iter().enumerate() {
        let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else {
            leaves[t.idx()] = 1;
            continue;
        };
        leaves[t.idx()] = leaves[left.idx()] + leaves[right.idx()];
        match body.varint("a level's counts marker")? {
            0 => continue,
            1 => {}
            m => return Err(malformed(format!("counts marker {m} of the level of vtree node {k} in postorder is not 0 or 1"))),
        }
        let n = width[t.idx()];
        if n > body.left() {
            return Err(malformed(format!("{n} counts of the level of vtree node {k} in postorder cannot fit {} bytes", body.left())));
        }
        let most = 1u128.checked_shl(leaves[t.idx()]).unwrap_or(u128::MAX);
        let mut fast = Vec::new();
        fast.try_reserve_exact(n).map_err(|_| malformed("no memory for a level's counts"))?;
        for _ in 0..n {
            let c = body.varint_bits("a node's count", 128)?;
            if c > most || c == u128::MAX {
                return Err(malformed(format!(
                    "a count of the level of vtree node {k} in postorder is past the {most} assignments of its variables"
                )));
            }
            fast.push(c);
        }
        counts.set(t, Arc::new(CountVec::from_fast(fast)));
    }
    Ok(counts)
}

/// One internal level's section, its pair sides checked against the widths
/// of its child levels.
fn read_level(body: &mut Body<'_>, left_width: usize, right_width: usize) -> Result<TddLevel, IoError> {
    let (left_bits, right_bits) = (index_bits(left_width), index_bits(right_width));
    let nodes = body.count("node count", NodeIdx::MAX_LIVE as u64)?;
    let multi = body.count("multi-pair node count", nodes as u64)?;
    let pairs = body.count("pair count", u64::MAX)?;
    let stream_bytes = body.count("stream length", body.left() as u64)?;
    // Every pair takes at least one bit of the stream unless both children
    // have one node, so its bits bound the pairs, and the nodes, which hold
    // one pair or more: no count can size an allocation the file cannot
    // fill. Over two one-node children the only pair is (0, 0), and
    // structural determinism puts it in one node, once.
    let most = if left_bits + right_bits == 0 {
        (left_width * right_width) as u64
    } else {
        (stream_bytes as u64).saturating_mul(8)
    };
    if pairs as u64 > most {
        return Err(malformed(format!(
            "{pairs} pairs cannot fit the child levels' {left_width} and {right_width} nodes \
             in a stream of {stream_bytes} bytes"
        )));
    }
    if pairs < nodes {
        return Err(malformed(format!("{nodes} nodes cannot hold only {pairs} pairs")));
    }

    let (bitmap, lengths) = if multi > 0 {
        let bitmap = body.take(nodes.div_ceil(8), "the multi-pair bitmap")?;
        let marked: usize = bitmap.iter().map(|b| b.count_ones() as usize).sum();
        if marked != multi || (!nodes.is_multiple_of(8) && bitmap[nodes / 8] >> (nodes % 8) != 0) {
            return Err(malformed(format!(
                "the multi-pair bitmap marks {marked} nodes, the section declares {multi}"
            )));
        }
        let mut lengths = Vec::new();
        lengths.try_reserve_exact(multi).map_err(|_| malformed("no memory for the pair counts"))?;
        let mut in_arena = 0u64;
        for _ in 0..multi {
            let extra = body.varint("pair count of a multi-pair node")?;
            let len = extra.checked_add(2).ok_or_else(|| malformed("a node's pair count overflows"))?;
            in_arena = in_arena.saturating_add(len);
            lengths.push(len);
        }
        if in_arena.checked_add((nodes - multi) as u64) != Some(pairs as u64) {
            return Err(malformed(format!(
                "the nodes' pair counts sum to {}, the section declares {pairs}",
                in_arena.saturating_add((nodes - multi) as u64)
            )));
        }
        (bitmap, lengths)
    } else {
        if pairs != nodes {
            return Err(malformed(format!(
                "{nodes} single-pair nodes cannot hold {pairs} pairs"
            )));
        }
        (&[][..], Vec::new())
    };

    let stream = body.take(stream_bytes, "the stream of pairs")?;
    let mut reader = PairReader { bytes: stream, at: 0, left_bits, right_bits, left_max: 0, right_max: 0 };
    let mut level = TddLevel::new();
    let arena = pairs - (nodes - multi);
    level.nodes.stored_mut().try_reserve_exact(nodes).map_err(|_| malformed("no memory for the level's nodes"))?;
    level.pairs.stored_mut().try_reserve_exact(arena).map_err(|_| malformed("no memory for the level's pairs"))?;
    if multi == 0 {
        level.nodes.stored_mut().extend((0..nodes).map(|_| EncodedNode::inline(reader.pair())));
    } else {
        let mut lengths = lengths.into_iter();
        for i in 0..nodes {
            if bitmap[i / 8] >> (i % 8) & 1 == 1 {
                let len = lengths.next().expect("one length per marked node") as usize;
                let start = level.pairs.len();
                reader.node(len, level.pairs.stored_mut())?;
                let node = level.encode_multi(start, len);
                level.nodes.stored_mut().push(node);
            } else {
                level.nodes.stored_mut().push(EncodedNode::inline(reader.pair()));
            }
        }
    }
    if reader.at.div_ceil(8) != stream.len() {
        return Err(malformed(format!(
            "the pairs take {} bits, the stream holds {} bytes",
            reader.at,
            stream.len()
        )));
    }
    if !reader.at.is_multiple_of(8) && stream[reader.at / 8] >> (reader.at % 8) != 0 {
        return Err(malformed("the stream's padding bits are not zero"));
    }
    if pairs > 0 && (reader.left_max >= left_width as u64 || reader.right_max >= right_width as u64) {
        return Err(malformed(format!(
            "a pair side is past its child level (left {left_width} nodes, right {right_width})"
        )));
    }
    Ok(level)
}
