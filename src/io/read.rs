//! Reading a diagram back from the `.tdd` text format.
//!
//! The records and the version rule are documented in `super::write`.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use crate::diagram::{ChildPair, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

use super::{IoError, TDD_FORMAT_VERSION};

/// Read a diagram from a `.tdd` file written by [`save_tdd`](super::save_tdd).
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
/// versioned), a problem line whose leaf or vtree node count is not `vtree`'s,
/// a record letter other than `c`, `p`, `L` and `I`, a record whose vtree
/// index is out of range or has the wrong kind, an `L` line disagreeing with
/// the vtree's variable, an `I` line whose declared children are not the
/// vtree's, an `I` line with no pairs or an odd number of pair tokens, a pair
/// side naming a node that does not exist, or an output node that was never
/// defined. The message names the line. [`IoError::Io`] if the reader fails.
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
    let mut pairs: Vec<ChildPair> = Vec::new();
    while let Some(l) = tok.next() {
        let l: u32 = l.parse().map_err(|_| malformed(line, format!("left pair index: {l:?}")))?;
        let r = next_u32(tok, "right pair index (pair tokens come two at a time)", line)?;
        pairs.push(ChildPair { left: NodeIdx(l), right: NodeIdx(r) });
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
    crate::diagram::builder::check_levels(vtree, &levels, output, None)
        .map_err(|e| malformed(h.line, format!("the records do not form a diagram: {e}")))?;
    Ok(Tdd::from_levels_unchecked(Arc::clone(vtree), levels, output))
}
