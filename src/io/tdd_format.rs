//! The `.tdd` text format: writing a diagram out, and reading one back.
//!
//! A whitespace-separated line format, one record per line, reachable nodes
//! only, in vtree bottom-up order:
//!
//! ```text
//! c <comment>
//! p tdd <version> <num_leaves> <num_vtree_nodes> <out_vtree> <out_local>
//! L <vtree_idx> <var>
//! I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> [<l1> <r1> ...]
//! ```
//!
//! - **`p`** — the problem line. `<version>` is the format version, which the
//!   writers here emit as 1. The circuit's output node is
//!   `(<out_vtree>, <out_local>)`; `<out_local>` is the literal token `ZERO`
//!   when the function is unsatisfiable, and then no `L` or `I` lines follow.
//! - **`L`** — a vtree leaf: vtree node `<vtree_idx>` tests DIMACS variable
//!   `<var>` (1-indexed). Each leaf has three implicit diagram nodes, never
//!   written, at local indices 0 = one (constant true), 1 = the positive
//!   literal, 2 = the negative literal.
//! - **`I`** — an internal diagram node at vtree node `<vtree_idx>`, a
//!   deterministic disjunction of conjunction pairs: the node equals
//!   `OR_k (l_k AND r_k)`. Each pair names its children by local index, `<lk>`
//!   into the node list of `<left_vtree>` and `<rk>` into that of `<right_vtree>`.
//!
//! Local indices are per vtree node and 0-based, in the order nodes are
//! emitted: the implicit 0/1/2 at a leaf, and for an internal vtree node the
//! count of `I` lines at that index so far in file order. A tautology is
//! `out_local = 0` at a leaf vtree node — the `one` node.
//!
//! The format carries neither the vtree's shape nor marginal levels. A
//! `.tdd` file names a vtree node only where the diagram occupies it, so the
//! ancestors of the output and every subtree the output does not reach leave no
//! trace — which is why [`read_tdd`] takes the vtree as an argument rather than
//! reconstructing one. Marginal levels hold per-node model counts instead of
//! nodes, so a pair into one carries a count where the format wants an index;
//! the writers refuse such a diagram outright. A weighted diagram is written,
//! not refused: what the file drops is the semiring, so it reads back in
//! integer mode and the caller attaches its weights again with
//! [`Tdd::set_weights`](crate::Tdd::set_weights).
//!
//! The version in the problem line is what makes the format interchange: a
//! file written by version n loads in every reader whose own version is n or
//! greater. A reader accepts any version up to its own, refuses a higher one
//! naming both versions, and refuses a problem line with no version at all as a
//! file written before the format was versioned. Comment lines it does not
//! recognize are ignored, so a writer may annotate a file freely; a record
//! letter it does not recognize is refused, so a format that needs a new record
//! raises the version.
//!
//! Every file opens with a comment block spelling the above out, so a file is
//! readable without this module. Keep the two in step.

use std::io::{BufRead, BufReader, BufWriter, Seek, Write};
use std::path::Path;
use std::sync::Arc;

use crate::diagram::{InputPair, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

use super::IoError;

/// The `.tdd` format version the writers here emit and the highest one the
/// reader here accepts.
///
/// A file written by version n loads in every reader whose version is n or
/// greater, so this number rises only when the records change in a way an older
/// reader would misread. Adding a comment line is not such a change; adding a
/// record letter is.
const TDD_FORMAT_VERSION: u32 = 1;

/// Estimate output size in bytes: ~12 bytes per pair entry + overhead.
fn estimate_size(tdd: &Tdd) -> usize {
    tdd.size() * 12 + 4096
}

/// Write a diagram to a file in .tdd text format.
///
/// The file is pre-sized from an estimate so the writes do not each extend it,
/// then truncated to what was actually written.
///
/// # Errors
///
/// [`IoError::Format`] if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the format is structural and cannot express
/// a level that stores per-node model counts instead of nodes. Nothing is
/// written and no file is created in that case. [`IoError::Io`] if the file
/// cannot be created or a write to it fails.
///
/// ```
/// # use std::sync::Arc;
/// # use tididi::{Engine, Tdd};
/// # use tididi::io::IoError;
/// # use tididi::marginal::marginalize;
/// # use tididi::vtree::Vtree;
/// # let vtree = Arc::new(Vtree::balanced(4));
/// # let engine = Engine::new();
/// # let (left, _right) = vtree.children(vtree.root());
/// # let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// use tididi::io::{load_tdd, save_tdd};
///
/// let path = std::env::temp_dir().join("tididi-doc-save.tdd");
/// save_tdd(&f, &path).unwrap();
/// let g = load_tdd(&path, &vtree).unwrap();
/// assert_eq!(g.model_count(), f.model_count());
/// std::fs::remove_file(&path).unwrap();
///
/// // A diagram with a level summed out has no structural form to write.
/// let mut m = f.clone();
/// marginalize(&engine, &mut m, &[left]).unwrap();
/// match save_tdd(&m, &path) {
///     Ok(()) => unreachable!("a marginal level cannot be written"),
///     Err(IoError::Format(msg)) => assert!(!msg.is_empty()),
///     Err(IoError::Io(e)) => unreachable!("{e}"),
///     Err(other) => unreachable!("{other}"),
/// }
/// assert!(!path.exists());   // nothing was created
/// ```
pub fn save_tdd(f: &Tdd, path: impl AsRef<Path>) -> Result<(), IoError> {
    // Checked before `File::create` so a rejected diagram leaves no stray file.
    super::reject_marginal_levels(f, "save_tdd")?;

    let file = std::fs::File::create(path.as_ref())?;

    // Pre-size the file so the writes below do not each extend it. This is
    // best-effort: an error leaves an ordinary growing write, and the
    // truncation after the flush trims whatever was over-allocated.
    let _ = file.set_len(estimate_size(f) as u64);

    let mut w = BufWriter::with_capacity(8 << 20, file); // 8MB buffer
    write_tdd(&mut w, f)?;
    w.flush()?;

    // Trim what the pre-sizing over-allocated. The write cursor is the number
    // of bytes actually written; the file's length is still the pre-sized one,
    // so reading the length here would keep the padding and hand the reader a
    // tail of zero bytes.
    let mut actual = w.into_inner().map_err(|e| e.into_error())?;
    let written = actual.stream_position()?;
    actual.set_len(written)?;

    Ok(())
}

/// Append an integer to a byte buffer using `itoa` (avoids `fmt::Formatter` overhead).
#[inline(always)]
fn push_int(buf: &mut Vec<u8>, n: u32) {
    let mut b = itoa::Buffer::new();
    buf.extend_from_slice(b.format(n).as_bytes());
}

/// Append a usize to a byte buffer.
#[inline(always)]
fn push_usize(buf: &mut Vec<u8>, n: usize) {
    let mut b = itoa::Buffer::new();
    buf.extend_from_slice(b.format(n).as_bytes());
}

/// Write a diagram in .tdd text format to any writer.
///
/// # Errors
///
/// [`IoError::Format`] if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the format is structural and cannot express
/// a level that stores per-node model counts instead of nodes. Nothing is
/// written to `w` in that case. [`IoError::Io`] if a write to `w` fails.
///
/// A weighted diagram is written rather than refused: the file carries the
/// Boolean structure, so it reads back in integer mode and the caller attaches
/// the weights again with [`Tdd::set_weights`](crate::Tdd::set_weights). The
/// weight store needs no rejection of its own, because every level whose values
/// live in that store is a marginal level and the check above already refuses
/// it.
///
/// ```
/// # use std::sync::Arc;
/// # use tididi::{Engine, Tdd};
/// # use tididi::io::IoError;
/// # use tididi::marginal::marginalize;
/// # use tididi::vtree::Vtree;
/// # let vtree = Arc::new(Vtree::balanced(4));
/// # let engine = Engine::new();
/// # let (left, _right) = vtree.children(vtree.root());
/// # let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// use tididi::io::{read_tdd, write_tdd};
///
/// let mut bytes: Vec<u8> = Vec::new();
/// write_tdd(&mut bytes, &f).unwrap();
/// let g = read_tdd(&mut bytes.as_slice(), &vtree).unwrap();
/// assert_eq!(g.model_count(), f.model_count());
///
/// let mut m = f.clone();
/// marginalize(&engine, &mut m, &[left]).unwrap();
/// let mut refused: Vec<u8> = Vec::new();
/// match write_tdd(&mut refused, &m) {
///     Ok(()) => unreachable!("a marginal level cannot be written"),
///     Err(IoError::Format(msg)) => assert!(!msg.is_empty()),
///     Err(IoError::Io(e)) => unreachable!("{e}"),
///     Err(other) => unreachable!("{other}"),
/// }
/// assert!(refused.is_empty());
/// ```
pub fn write_tdd<W: Write>(w: &mut W, tdd: &Tdd) -> Result<(), IoError> {
    super::reject_marginal_levels(tdd, "write_tdd")?;

    let vtree = &tdd.vtree;
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    push_format_header(&mut buf);

    if tdd.is_zero() {
        push_problem_line(&mut buf, tdd, None);
        w.write_all(&buf)?;
        return Ok(());
    }

    let reachable = tdd.reachable_nodes();
    let remap = local_index_remap(tdd, &reachable);
    let out_local = remap[tdd.output.vtree.idx()][tdd.output.local.idx()];
    debug_assert_ne!(out_local, u32::MAX, "output node must be reachable");

    push_problem_line(&mut buf, tdd, Some(out_local));
    w.write_all(&buf)?;
    buf.clear();

    // "L <vtree_idx> <var>": the vtree leaf → variable mapping. Each leaf has 3
    // implicit diagram nodes — one(0), pos(1), neg(2) — which are not written.
    for (t, var) in vtree.leaf_bottomup() {
        buf.extend_from_slice(b"L ");
        push_int(&mut buf, t.0);
        buf.push(b' ');
        push_int(&mut buf, var.0 + 1); // 1-indexed DIMACS variable
        buf.push(b'\n');
    }
    w.write_all(&buf)?;
    buf.clear();

    write_internal_lines(w, tdd, &reachable, &remap, &mut buf)
}

/// The comment block that documents the format inside every file it writes.
/// Keep it in sync with what the writers below emit.
fn push_format_header(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"c TiDiDi TDD circuit\n");
    buf.extend_from_slice(
        b"c\n\
          c Format: a Tree Decision Diagram (TDD) over a vtree. Whitespace-separated\n\
          c tokens, one record per line. Reachable nodes only, in vtree bottom-up order.\n\
          c\n\
          c   p tdd <version> <num_leaves> <num_vtree_nodes> <out_vtree> <out_local>\n\
          c       Problem line. The circuit's output node is (<out_vtree>, <out_local>).\n\
          c       <out_local> is the literal token ZERO when the function is UNSAT\n\
          c       (no further L/I lines follow in that case).\n\
          c       <version> is the format version, and this file is version ",
    );
    push_int(buf, TDD_FORMAT_VERSION);
    buf.extend_from_slice(
        b".\n\
          c       A file written by version n loads in every reader whose own version\n\
          c       is n or greater: a reader accepts any version up to its own and\n\
          c       refuses a higher one. A comment line a reader does not recognize is\n\
          c       ignored; a record letter it does not recognize is refused, so a\n\
          c       format needing a new record raises the version.\n\
          c\n\
          c   L <vtree_idx> <var>\n\
          c       A vtree leaf: vtree node <vtree_idx> tests DIMACS variable <var>\n\
          c       (1-indexed). Each leaf has 3 implicit TDD nodes, NOT written, with\n\
          c       local indices: 0 = one (constant true), 1 = positive literal\n\
          c       (var=true), 2 = negative literal (var=false).\n\
          c\n\
          c   I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> [<l1> <r1> ...]\n\
          c       An internal TDD node at vtree node <vtree_idx>, decomposing into a\n\
          c       deterministic OR of AND-pairs: the node equals OR_k (left_k AND right_k).\n\
          c       Each pair (<lk> <rk>) references a child by LOCAL index: <lk> into the\n\
          c       node list of <left_vtree>, <rk> into that of <right_vtree>.\n\
          c\n\
          c   Local indices are per vtree node, 0-based, in the order nodes are emitted\n\
          c   (leaf locals are the implicit 0/1/2 above; internal locals count I lines at\n\
          c   that vtree_idx, in file order). The output (out_vtree, out_local) uses the\n\
          c   same scheme; a tautology is out_local = 0 at a leaf vtree (the 'one' node).\n\
          c\n",
    );
}

/// The `p tdd` line. `out_local` is `None` for the unsatisfiable diagram, which
/// writes the `ZERO` token in its place and ends the file.
fn push_problem_line(buf: &mut Vec<u8>, tdd: &Tdd, out_local: Option<u32>) {
    buf.extend_from_slice(b"p tdd ");
    push_int(buf, TDD_FORMAT_VERSION);
    buf.push(b' ');
    push_int(buf, tdd.vtree.num_leaves());
    buf.push(b' ');
    push_usize(buf, tdd.vtree.num_nodes());
    buf.push(b' ');
    push_int(buf, tdd.output.vtree.0);
    match out_local {
        Some(local) => {
            buf.push(b' ');
            push_int(buf, local);
        }
        None => buf.extend_from_slice(b" ZERO"),
    }
    buf.push(b'\n');
}

/// Per vtree node, the map from a node's index in the level to the local index
/// the file gives it. Unreachable nodes are dropped (`u32::MAX`), so internal
/// levels compact; leaf levels keep the identity map, since a reader
/// reconstructs all three implicit nodes regardless.
fn local_index_remap(tdd: &Tdd, reachable: &[Vec<bool>]) -> Vec<Vec<u32>> {
    let vtree = &tdd.vtree;
    let mut remap: Vec<Vec<u32>> = Vec::with_capacity(vtree.num_nodes());
    for (vi, reach) in reachable.iter().enumerate() {
        let mut map = vec![u32::MAX; reach.len()];
        if vtree.node(VtreeIdx(vi as u32)).is_leaf() {
            for (j, slot) in map.iter_mut().enumerate() {
                *slot = j as u32;
            }
        } else {
            let mut next = 0u32;
            for (i, _) in tdd.level(VtreeIdx(vi as u32)).internal_inputs_iter() {
                if reach[i] {
                    map[i] = next;
                    next += 1;
                }
            }
        }
        remap.push(map);
    }
    remap
}

/// The `I` lines: one per reachable internal node, its pairs written through
/// `remap` so the reader's sequential local indices line up. Flushed to `w` in
/// buffer-sized chunks rather than held in one allocation.
fn write_internal_lines<W: Write>(
    w: &mut W,
    tdd: &Tdd,
    reachable: &[Vec<bool>],
    remap: &[Vec<u32>],
    buf: &mut Vec<u8>,
) -> Result<(), IoError> {
    for (t, left_vtree, right_vtree) in tdd.vtree.internal_bottomup() {
        let reach = &reachable[t.idx()];
        let left_remap = &remap[left_vtree.idx()];
        let right_remap = &remap[right_vtree.idx()];
        for (i, pairs) in tdd.level(t).internal_inputs_iter() {
            if !reach[i] {
                continue;
            }
            buf.extend_from_slice(b"I ");
            push_int(buf, t.0);
            buf.push(b' ');
            push_int(buf, left_vtree.0);
            buf.push(b' ');
            push_int(buf, right_vtree.0);
            // Marginal levels are refused at entry, so both sides are plain
            // node indices — no value ref can appear here.
            for pair in pairs {
                buf.push(b' ');
                push_int(buf, left_remap[pair.left.idx()]);
                buf.push(b' ');
                push_int(buf, right_remap[pair.right.idx()]);
            }
            buf.push(b'\n');
            if buf.len() > 64 * 1024 {
                w.write_all(buf)?;
                buf.clear();
            }
        }
        if !buf.is_empty() {
            w.write_all(buf)?;
            buf.clear();
        }
    }
    Ok(())
}


// ── Reading ──────────────────────────────────────────────────────────────────

/// Read a diagram from a `.tdd` file written by [`save_tdd`].
///
/// # Errors
///
/// [`IoError::Io`] if the file cannot be opened or read; [`IoError::Format`]
/// for anything the bytes get wrong — see [`read_tdd`], which this wraps.
///
/// ```
/// # use std::sync::Arc;
/// # use tididi::Tdd;
/// # use tididi::io::IoError;
/// # use tididi::vtree::Vtree;
/// # let vtree = Arc::new(Vtree::balanced(4));
/// use tididi::io::{load_tdd, save_tdd};
///
/// let path = std::env::temp_dir().join("tididi-doc-load.tdd");
/// let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// save_tdd(&f, &path).unwrap();
/// assert_eq!(load_tdd(&path, &vtree).unwrap().model_count(), f.model_count());
/// std::fs::remove_file(&path).unwrap();
///
/// // The file is gone now, so opening it fails on the underlying error.
/// match load_tdd(&path, &vtree) {
///     Ok(_) => unreachable!("the file was removed"),
///     Err(IoError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
///     Err(IoError::Format(msg)) => unreachable!("{msg}"),
///     Err(other) => unreachable!("{other}"),
/// }
/// ```
pub fn load_tdd(path: impl AsRef<Path>, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let file = std::fs::File::open(path.as_ref())?;
    read_tdd(&mut BufReader::new(file), vtree)
}

/// Read a diagram in `.tdd` format from any reader, over `vtree`.
///
/// The vtree is an argument, not something the file carries. A `.tdd` file
/// describes the vtree only where the diagram touches it: leaves get an `L`
/// line each, but an internal vtree node is named only by the `I` lines of the
/// diagram nodes living there, so a vtree node the diagram never uses — every
/// ancestor of the output, and every node in a subtree the output does not
/// reach — leaves no trace. Rather than guess at a shape and hand back a
/// diagram over a vtree that merely resembles the original, the reader takes
/// the vtree it is reading against and checks the file against it.
///
/// The check the file supports is a partial one — the leaf count, the vtree
/// node count and the kind of each named node — so a caller holding two vtrees
/// of the same shape settles which one the file belongs to itself, with
/// [`Vtree::same_tree`](crate::vtree::Vtree::same_tree).
///
/// Round trip: `read_tdd(write_tdd(f), f.vtree)` is `f` up to the unreachable
/// nodes the writer drops and the local renumbering that compacts what is
/// left — the function and the level-by-level structure are unchanged.
///
/// # Errors
///
/// [`IoError::Format`] on a malformed or inconsistent file: a missing or
/// unparsable problem line, a problem line whose format version is higher than
/// this reader's or absent altogether (a file written before the format was
/// versioned), a record whose vtree index is out of range or has
/// the wrong kind, an `L` line disagreeing with the vtree's variable, an `I`
/// line whose declared children are not the vtree's, an odd number of pair
/// tokens, a pair side naming a node that does not exist, or an output node
/// that was never defined. [`IoError::Io`] if the reader fails.
///
/// ```
/// # use std::sync::Arc;
/// # use tididi::Tdd;
/// # use tididi::io::IoError;
/// # use tididi::vtree::Vtree;
/// # let vtree = Arc::new(Vtree::balanced(4));
/// use tididi::io::{read_tdd, write_tdd};
///
/// let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let mut bytes: Vec<u8> = Vec::new();
/// write_tdd(&mut bytes, &f).unwrap();
/// assert_eq!(read_tdd(&mut bytes.as_slice(), &vtree).unwrap().model_count(), f.model_count());
///
/// // Bytes that are not a `.tdd` file are refused with what they got wrong.
/// match read_tdd(&mut &b"not a diagram\n"[..], &vtree) {
///     Ok(_) => unreachable!("these bytes are not a diagram"),
///     Err(IoError::Format(msg)) => assert!(!msg.is_empty()),
///     Err(IoError::Io(e)) => unreachable!("{e}"),
///     Err(other) => unreachable!("{other}"),
/// }
/// ```
pub fn read_tdd<R: BufRead>(r: &mut R, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    let mut header: Option<ProblemLine> = None;
    // Pairs name their children by the local index the writer assigned, which
    // for an internal level counts `I` lines at that vtree node in file order —
    // exactly the order `push_internal_node` assigns, so nothing needs mapping.
    for (n, line) in r.lines().enumerate() {
        let line = line?;
        let mut tok = line.split_ascii_whitespace();
        match tok.next() {
            None | Some("c") => {}
            Some("p") => {
                let h = parse_problem_line(&mut tok, n)?;
                check_problem_line(&h, vtree, n)?;
                header = Some(h);
            }
            Some("L") => read_leaf_line(&mut tok, vtree, n)?,
            Some("I") => read_internal_line(&mut tok, vtree, &mut levels, n)?,
            Some(other) => {
                return Err(malformed(n, format!("unknown record type {other:?}")));
            }
        }
    }
    let header = header.ok_or_else(|| IoError::Format("tdd: no `p tdd` problem line".into()))?;
    build_diagram(header, levels, vtree)
}

/// The `p tdd` line's fields. `out_local` is `None` for the `ZERO` token.
struct ProblemLine {
    num_leaves: u32,
    num_vtree_nodes: usize,
    out_vtree: VtreeIdx,
    out_local: Option<u32>,
    line: usize,
}

fn malformed(line: usize, what: impl std::fmt::Display) -> IoError {
    IoError::Format(format!("tdd: line {}: {what}", line + 1))
}

/// One whitespace-separated `u32`, named for the error message.
fn next_u32<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    what: &str,
    line: usize,
) -> Result<u32, IoError> {
    let t = tok.next().ok_or_else(|| malformed(line, format!("missing {what}")))?;
    t.parse().map_err(|_| malformed(line, format!("{what} is not a number: {t:?}")))
}

/// A vtree index, checked against the tree's node count.
fn next_vtree_idx<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    what: &str,
    vtree: &Vtree,
    line: usize,
) -> Result<VtreeIdx, IoError> {
    let raw = next_u32(tok, what, line)?;
    if raw as usize >= vtree.num_nodes() {
        return Err(malformed(
            line,
            format!("{what} {raw} is past the vtree's {} nodes", vtree.num_nodes()),
        ));
    }
    Ok(VtreeIdx(raw))
}

/// The version field of the problem line, against the version this reader
/// understands.
///
/// `fields` is everything after `p tdd`. A version-1 line has five of them, so
/// four or fewer is a file from before the format was versioned: its first
/// field is the leaf count, and reading it as a version would turn a
/// pre-release file into a nonsense diagram. Beyond that the arity is the
/// version's business, so a longer line is left to the version check to refuse.
fn check_version(fields: &[&str], line: usize) -> Result<(), IoError> {
    let Some(&raw) = fields.first().filter(|_| fields.len() >= 5) else {
        return Err(malformed(
            line,
            format!(
                "the problem line carries no format version, so this file was written \
                 before the format was versioned; this reader understands version \
                 {TDD_FORMAT_VERSION} and can only load a file that names its own"
            ),
        ));
    };
    let version: u32 = raw
        .parse()
        .map_err(|_| malformed(line, format!("format version is not a number: {raw:?}")))?;
    if version > TDD_FORMAT_VERSION {
        return Err(malformed(
            line,
            format!(
                "the file is format version {version}; this reader understands version \
                 {TDD_FORMAT_VERSION}, and a reader loads a file only up to its own version"
            ),
        ));
    }
    Ok(())
}

fn parse_problem_line<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    line: usize,
) -> Result<ProblemLine, IoError> {
    match tok.next() {
        Some("tdd") => {}
        other => return Err(malformed(line, format!("expected `p tdd`, found `p {other:?}`"))),
    }
    // The version is the first field, and the rest of the line is read behind
    // it — an unreadable version means the fields after it are not this
    // format's and must not be parsed as if they were.
    let fields: Vec<&str> = tok.collect();
    check_version(&fields, line)?;
    let mut rest = fields.into_iter().skip(1);
    let tok = &mut rest;
    let num_leaves = next_u32(tok, "leaf count", line)?;
    let num_vtree_nodes = next_u32(tok, "vtree node count", line)? as usize;
    let out_vtree = VtreeIdx(next_u32(tok, "output vtree node", line)?);
    let out_local = match tok.next() {
        Some("ZERO") => None,
        Some(t) => Some(
            t.parse().map_err(|_| malformed(line, format!("output local index: {t:?}")))?,
        ),
        None => return Err(malformed(line, "missing output local index")),
    };
    Ok(ProblemLine { num_leaves, num_vtree_nodes, out_vtree, out_local, line })
}

/// The header describes the same tree the caller passed, or the file is not
/// this diagram's.
fn check_problem_line(h: &ProblemLine, vtree: &Vtree, line: usize) -> Result<(), IoError> {
    if h.num_vtree_nodes != vtree.num_nodes() {
        return Err(malformed(
            line,
            format!(
                "file has {} vtree nodes, the vtree read against has {}",
                h.num_vtree_nodes,
                vtree.num_nodes()
            ),
        ));
    }
    if h.num_leaves != vtree.num_leaves() {
        return Err(malformed(
            line,
            format!(
                "file has {} leaves, the vtree read against carries {}",
                h.num_leaves,
                vtree.num_leaves()
            ),
        ));
    }
    if h.out_vtree.idx() >= vtree.num_nodes() {
        return Err(malformed(line, format!("output vtree node {:?} is past the tree", h.out_vtree)));
    }
    Ok(())
}

/// `L <vtree_idx> <var>`: nothing to store — the vtree already says which
/// variable a leaf tests. Read to check the file and the vtree agree.
fn read_leaf_line<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    vtree: &Vtree,
    line: usize,
) -> Result<(), IoError> {
    let t = next_vtree_idx(tok, "leaf vtree node", vtree, line)?;
    let var = next_u32(tok, "variable", line)?;
    match vtree.node(t) {
        VtreeNode::Leaf { var: v, .. } if v.0 + 1 == var => Ok(()),
        VtreeNode::Leaf { var: v, .. } => Err(malformed(
            line,
            format!("leaf {t:?} tests variable {} in the vtree, {var} in the file", v.0 + 1),
        )),
        _ => Err(malformed(line, format!("{t:?} is an internal vtree node, not a leaf"))),
    }
}

/// `I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> ...`: one internal node,
/// appended to its level in file order.
fn read_internal_line<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    line: usize,
) -> Result<(), IoError> {
    let t = next_vtree_idx(tok, "node vtree index", vtree, line)?;
    let left = next_vtree_idx(tok, "left child vtree index", vtree, line)?;
    let right = next_vtree_idx(tok, "right child vtree index", vtree, line)?;
    let VtreeNode::Internal { left: vl, right: vr, .. } = *vtree.node(t) else {
        return Err(malformed(line, format!("{t:?} is a vtree leaf; an `I` record needs an internal node")));
    };
    if (vl, vr) != (left, right) {
        return Err(malformed(
            line,
            format!("node at {t:?} declares children ({left:?}, {right:?}); the vtree has ({vl:?}, {vr:?})"),
        ));
    }
    let mut pairs: Vec<InputPair> = Vec::new();
    while let Some(l) = tok.next() {
        let l: u32 = l.parse().map_err(|_| malformed(line, format!("left pair index: {l:?}")))?;
        let r = next_u32(tok, "right pair index (pair tokens come two at a time)", line)?;
        pairs.push(InputPair { left: NodeIdx(l), right: NodeIdx(r) });
    }
    if pairs.is_empty() {
        return Err(malformed(line, format!("node at {t:?} has no pairs")));
    }
    levels[t.idx()].push_internal_node(&pairs);
    Ok(())
}

/// Assemble what the records built, with the full structural validation
/// [`check_levels`](crate::diagram::builder::check_levels) runs — every pair
/// side in range, marginality, and an output that exists.
fn build_diagram(
    h: ProblemLine,
    levels: Vec<TddLevel>,
    vtree: &Arc<Vtree>,
) -> Result<Tdd, IoError> {
    let output = match h.out_local {
        None => TddNodeId { vtree: h.out_vtree, local: crate::diagram::ZERO },
        Some(local) => TddNodeId { vtree: h.out_vtree, local: NodeIdx(local) },
    };
    crate::diagram::builder::check_levels(vtree, &levels, output, false)
        .map_err(|e| malformed(h.line, format!("the records do not form a diagram: {e}")))?;
    Ok(Tdd::from_levels_unchecked(Arc::clone(vtree), levels, output))
}
