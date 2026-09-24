//! Readers for the [public text format](crate::io#text-format).

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use crate::diagram::{EncodedChildRef, ChildPair, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

use super::{IoError, TDD_FORMAT_VERSION};

/// Read a diagram from a `.tdd` file written by [`save_tdd`](super::save_tdd).
///
/// The diagram shares the supplied vtree; see [`read_tdd`] for format requirements.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{load_tdd, save_tdd};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let path = std::env::temp_dir().join("tididi-doc-load.tdd");
/// save_tdd(&f, &path)?;
/// let restored = load_tdd(&path, &vtree)?;
/// assert!(restored.equivalent(&f)?);
/// # tididi::test_helpers::assert_canonical(&f);
/// # tididi::test_helpers::assert_canonical(&restored);
/// # std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`IoError::Io`] if the file cannot be opened or read, or the format
/// errors from [`read_tdd`].
pub fn load_tdd(path: impl AsRef<Path>, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let file = std::fs::File::open(path.as_ref())?;
    read_tdd(&mut BufReader::new(file), vtree)
}

/// Read a diagram in `.tdd` format from any reader, over `vtree`.
///
/// Supply the vtree saved alongside the diagram: the reader validates counts,
/// leaf labels, and child relationships against it. File-local vtree IDs need not
/// equal the supplied vtree's in-memory indices. The false diagram
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
/// Returns [`IoError::Format`] for unsupported versions, malformed records,
/// missing or duplicate declarations, references outside the stored diagram,
/// or declarations inconsistent with `vtree`. See the [text format](crate::io#text-format)
/// for the record requirements. Error messages identify the affected line.
/// Returns [`IoError::Io`] if reading fails.
///
/// Read from an in-memory buffer:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{read_tdd, write_tdd};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let mut bytes = Vec::new();
/// write_tdd(&mut bytes, &f)?;
/// let restored = read_tdd(&mut bytes.as_slice(), &vtree)?;
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
/// let vtree = Arc::new(Vtree::balanced(3));
/// let result = read_tdd(&mut b"not a diagram".as_slice(), &vtree);
/// assert!(matches!(result, Err(IoError::Format(_))));
/// ```
pub fn read_tdd<R: BufRead>(r: &mut R, vtree: &Arc<Vtree>) -> Result<Tdd, IoError> {
    let mut levels = vec![FileLevel::default(); vtree.num_nodes()];
    let mut header: Option<ProblemLine> = None;
    // Pairs name their children by the local index the writer assigned, which
    // for an internal level counts `I` lines at that vtree node in file order —
    // exactly the order `push_internal_node` assigns. Only vtree IDs need mapping.
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
                    read_leaf_line(&mut tok, vtree, &mut levels, n)?;
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
    build_diagram(header, levels, vtree)
}

/// One file-local vtree declaration and the diagram nodes stored at it.
#[derive(Clone, Default)]
struct FileLevel {
    node: Option<VtreeNode>,
    line: usize,
    diagram: TddLevel,
}

impl FileLevel {
    fn declare(&mut self, node: VtreeNode, line: usize) -> Result<(), IoError> {
        if let Some(previous) = &self.node {
            if node.is_leaf() || *previous != node {
                return Err(malformed(line, "duplicate or conflicting vtree declaration"));
            }
        } else {
            self.node = Some(node);
            self.line = line;
        }
        Ok(())
    }
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

/// Check the format version before interpreting the remaining header fields.
fn parse_problem_line<'a>(
    tok: &mut (impl Iterator<Item = &'a str> + Clone),
    line: usize,
) -> Result<ProblemLine, IoError> {
    match tok.next() {
        Some("tdd") => {}
        other => return Err(malformed(line, format!("expected `p tdd`, found `p {other:?}`"))),
    }
    // Versioned headers have at least five fields. Peek without consuming them
    // so a shorter, unversioned header is not mistaken for a version number.
    if tok.clone().nth(4).is_none() {
        return Err(malformed(
            line,
            format!(
                "the problem line carries no format version, so this file was written \
                 before the format was versioned; this reader understands version \
                 {TDD_FORMAT_VERSION} and can only load a file that names its own"
            ),
        ));
    }
    let version = next_u32(tok, "format version", line)?;
    if version != TDD_FORMAT_VERSION {
        return Err(malformed(
            line,
            format!(
                "the file is format version {version}; this reader understands version \
                 {TDD_FORMAT_VERSION}"
            ),
        ));
    }
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

/// Record the variable at a file-local leaf, independent of in-memory indices.
fn read_leaf_line<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    vtree: &Vtree,
    levels: &mut [FileLevel],
    line: usize,
) -> Result<(), IoError> {
    let t = next_vtree_idx(tok, "leaf vtree node", vtree, line)?;
    let var = next_u32(tok, "variable", line)?;
    end_of_record(tok, line)?;
    if var == 0 { return Err(malformed(line, "variables are numbered from 1")); }
    levels[t.idx()].declare(VtreeNode::Leaf { var: VarId(var), parent: None }, line)
}

/// `I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> ...`: one internal node,
/// appended to its level in file order.
fn read_internal_line<'a>(
    tok: &mut impl Iterator<Item = &'a str>,
    vtree: &Vtree,
    levels: &mut [FileLevel],
    line: usize,
) -> Result<(), IoError> {
    let t = next_vtree_idx(tok, "node vtree index", vtree, line)?;
    let left = next_vtree_idx(tok, "left child vtree index", vtree, line)?;
    let right = next_vtree_idx(tok, "right child vtree index", vtree, line)?;
    levels[t.idx()].declare(VtreeNode::Internal { left, right, parent: None }, line)?;
    let mut pairs: Vec<ChildPair> = Vec::new();
    while let Some(l) = tok.next() {
        let l: u32 = l.parse().map_err(|_| malformed(line, format!("left pair index: {l:?}")))?;
        let r = next_u32(tok, "right pair index (pair tokens come two at a time)", line)?;
        pairs.push(ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r)));
    }
    if pairs.is_empty() {
        return Err(malformed(line, format!("node at {t:?} has no pairs")));
    }
    levels[t.idx()].diagram.push_internal_node(&pairs);
    Ok(())
}

/// Assemble what the records built, with the full structural validation
/// [`check_levels`](crate::diagram::check_levels) runs — every pair
/// side in range, marginality, and an output that exists.
fn build_diagram(
    h: ProblemLine,
    mut stored: Vec<FileLevel>,
    vtree: &Arc<Vtree>,
) -> Result<Tdd, IoError> {
    let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    if h.out_vtree.idx() >= stored.len() {
        return Err(malformed(h.line, "output vtree node is outside the declared range"));
    }
    if h.out_local.is_some() {
        let mut seen = vec![false; stored.len()];
        let mut pending = vec![(h.out_vtree, vtree.root())];
        while let Some((file, target)) = pending.pop() {
            if std::mem::replace(&mut seen[file.idx()], true) {
                return Err(malformed(stored[file.idx()].line, "vtree node reached more than once"));
            }
            let entry = &mut stored[file.idx()];
            let node = entry.node.as_ref().ok_or_else(|| {
                malformed(h.line, format!("missing vtree declaration for node {}", file.0))
            })?;
            match (node, vtree.node(target)) {
                (VtreeNode::Leaf { var: a, .. }, VtreeNode::Leaf { var: b, .. }) if a == b => {}
                (VtreeNode::Internal { left, right, .. }, VtreeNode::Internal { left: l, right: r, .. }) => {
                    pending.push((*right, *r));
                    pending.push((*left, *l));
                }
                _ => return Err(malformed(entry.line, format!("vtree node {} disagrees with the supplied vtree", file.0))),
            }
            levels[target.idx()] = std::mem::take(&mut entry.diagram);
        }
        if seen.iter().any(|seen| !seen) {
            return Err(malformed(h.line, "vtree declarations do not form one complete vtree"));
        }
    } else if h.out_vtree != vtree.root()
        && h.out_vtree.0 != vtree.topo_pos(vtree.root()) {
        return Err(malformed(h.line, "ZERO output must name the vtree root"));
    }
    let output = TddNodeId {
        vtree: vtree.root(),
        local: h.out_local.map_or(crate::diagram::ZERO, NodeIdx),
    };
    crate::diagram::check_levels(vtree, &levels, output, None)
        .map_err(|e| malformed(h.line, format!("the records do not form a diagram: {e}")))?;
    Ok(Tdd::from_levels_unchecked(Arc::clone(vtree), levels, output))
}
