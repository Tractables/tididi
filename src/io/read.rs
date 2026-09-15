//! Readers for the [public text format](crate::io#text-format).

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use crate::diagram::{EncodedChildRef, ChildPair, NodeIdx, Tdd, TddLevel, TddNodeId};
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
/// let f = Tdd::clause(&vtree, [1, -2])? & Tdd::clause(&vtree, [2, 3])?;
/// save_tdd(&f, &path).unwrap();
/// assert_eq!(load_tdd(&path, &vtree).unwrap().model_count()?, f.model_count()?);
/// std::fs::remove_file(&path).unwrap();
///
/// // The file is gone now, so opening it fails on the underlying error.
/// match load_tdd(&path, &vtree) {
///     Ok(_) => unreachable!("the file was removed"),
///     Err(IoError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
///     Err(IoError::Format(msg)) => unreachable!("{msg}"),
///     Err(other) => unreachable!("{other}"),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_tdd(path: impl AsRef<Path>, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let file = std::fs::File::open(path.as_ref())?;
    read_tdd(&mut BufReader::new(file), vtree)
}

/// Read a diagram in `.tdd` format from any reader, over `vtree`.
///
/// Supply the tree saved alongside the diagram: the reader validates counts,
/// leaf labels, and declared child indices against it, but the false diagram
/// has no node records from which to check its shape.
/// The result shares the supplied `Arc<Vtree>`, has no attached weights, and
/// preserves the file's node order; reading does not minimize.
///
/// The reader validates the encoding, not semantic determinism. Files must
/// describe TDDs satisfying the [data-model rules](crate::guide::model), as the
/// library's writer produces; arbitrary overlapping pair lists are not valid
/// inputs for model counting.
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
/// defined. Exactly one problem line precedes the node records; fixed-length
/// records have no trailing fields. Nonzero diagrams declare every leaf once;
/// a `ZERO` output has no node records and cannot be spelled as a numeric index. The message names the line. [`IoError::Io`] if the reader fails.
///
/// Read from an in-memory buffer:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{read_tdd, write_tdd};
///
/// let tree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&tree, [1, -2])?;
/// let mut bytes = Vec::new();
/// write_tdd(&mut bytes, &f)?;
/// let restored = read_tdd(&mut bytes.as_slice(), &tree)?;
/// assert_eq!(restored.model_count()?, f.model_count()?);
/// # tididi::test_helpers::assert_canonical(&f);
/// # tididi::test_helpers::assert_canonical(&restored);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Malformed input reports a format error:
///
/// ```
/// use std::sync::Arc;
/// use tididi::Vtree;
/// use tididi::io::{read_tdd, IoError};
///
/// let tree = Arc::new(Vtree::balanced(3));
/// let result = read_tdd(&mut b"not a diagram".as_slice(), &tree);
/// assert!(matches!(result, Err(IoError::Format(_))));
/// ```
pub fn read_tdd<R: BufRead>(r: &mut R, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    let mut header: Option<ProblemLine> = None;
    let mut leaves = vec![false; vtree.num_leaves() as usize];
    // Pairs name their children by the local index the writer assigned, which
    // for an internal level counts `I` lines at that vtree node in file order —
    // exactly the order `push_internal_node` assigns, so nothing needs mapping.
    for (n, line) in r.lines().enumerate() {
        let line = line?;
        let mut tok = line.split_ascii_whitespace();
        match tok.next() {
            None | Some("c") => {}
            Some("p") => {
                if header.is_some() { return Err(malformed(n, "duplicate problem line")); }
                let h = parse_problem_line(&mut tok, n)?;
                check_problem_line(&h, vtree, n)?;
                header = Some(h);
            }
            Some(kind @ ("L" | "I")) => {
                let h = header.as_ref().ok_or_else(|| malformed(n, "node record before the problem line"))?;
                if h.out_local.is_none() { return Err(malformed(n, "node record after a ZERO output")); }
                if kind == "L" {
                    let leaf = read_leaf_line(&mut tok, vtree, n)?;
                    if leaves[leaf.idx()] { return Err(malformed(n, format!("duplicate leaf {}", leaf.idx()))); }
                    leaves[leaf.idx()] = true;
                } else {
                    read_internal_line(&mut tok, vtree, &mut levels, n)?;
                }
            }
            Some(other) => {
                return Err(malformed(n, format!("unknown record type {other:?}")));
            }
        }
    }
    let header = header.ok_or_else(|| IoError::Format("tdd: no `p tdd` problem line".into()))?;
    if header.out_local.is_some() && let Some(missing) = leaves.iter().position(|&seen| !seen) {
        return Err(malformed(header.line, format!("missing leaf record for vtree node {missing}")));
    }
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
        Some(t) => {
            let local = t.parse().map_err(|_| malformed(line, format!("output local index: {t:?}")))?;
            if local == u32::MAX { return Err(malformed(line, "a zero output must use the ZERO token")); }
            Some(local)
        }
        None => return Err(malformed(line, "missing output local index")),
    };
    end_of_record(tok, line)?;
    Ok(ProblemLine { num_leaves, num_vtree_nodes, out_vtree, out_local, line })
}

/// Reject fields after a fixed-length record.
fn end_of_record<'a>(tok: &mut impl Iterator<Item = &'a str>, line: usize) -> Result<(), IoError> {
    if let Some(extra) = tok.next() { return Err(malformed(line, format!("unexpected trailing field {extra:?}"))); }
    Ok(())
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
) -> Result<VtreeIdx, IoError> {
    let t = next_vtree_idx(tok, "leaf vtree node", vtree, line)?;
    let var = next_u32(tok, "variable", line)?;
    end_of_record(tok, line)?;
    match vtree.node(t) {
        VtreeNode::Leaf { var: v, .. } if v.0 + 1 == var => Ok(t),
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
        pairs.push(ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r)));
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
