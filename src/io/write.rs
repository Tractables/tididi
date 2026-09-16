//! Writers for the [public text format](crate::io#text-format).

use crate::diagram::ChildDecoder;

use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

use super::{IoError, TDD_FORMAT_VERSION};

/// Write a diagram to a file in `.tdd` text format, creating or truncating
/// the file.
///
/// # Errors
///
/// [`IoError::Format`] if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the format is structural and cannot express
/// a level that stores per-node values instead of nodes. Nothing is written
/// and no file is created in that case. [`IoError::Io`] if the file cannot be
/// created or a write to it fails; a partial file may then be left behind.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{load_tdd, save_tdd};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let path = std::env::temp_dir().join("tididi-doc-save.tdd");
/// save_tdd(&f, &path)?;
/// let restored = load_tdd(&path, &vtree)?;
/// assert!(restored.equivalent(&f)?);
/// # tididi::test_helpers::assert_canonical(&f);
/// # tididi::test_helpers::assert_canonical(&restored);
/// # std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn save_tdd(f: &Tdd, path: impl AsRef<Path>) -> Result<(), IoError> {
    // Checked before `File::create` so a rejected diagram leaves no stray file.
    super::reject_marginal_levels(f, "save_tdd")?;

    const BUFFER_BYTES: usize = 8 << 20;
    let file = std::fs::File::create(path.as_ref())?;
    let estimated_bytes = f.pair_count() * 12 + 4096;
    let presize = estimated_bytes > BUFFER_BYTES;
    // Pre-size multi-buffer outputs so writes need not each extend the file.
    if presize { let _ = file.set_len(estimated_bytes as u64); }
    let mut writer = BufWriter::with_capacity(BUFFER_BYTES, file);
    write_tdd(&mut writer, f)?;
    writer.flush()?;
    if presize {
        let file = writer.get_mut();
        let written = file.stream_position()?;
        file.set_len(written)?;
    }
    Ok(())
}

/// Append an integer to a byte buffer through `itoa`, without `fmt::Formatter`.
#[inline(always)]
fn push_num<N: itoa::Integer>(buf: &mut Vec<u8>, n: N) {
    let mut b = itoa::Buffer::new();
    buf.extend_from_slice(b.format(n).as_bytes());
}

/// Write a diagram in `.tdd` text format to any writer. `w` is not flushed.
///
/// # Errors
///
/// [`IoError::Format`] if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the format is structural and cannot express
/// a level that stores per-node values instead of nodes. Nothing is written to
/// `w` in that case. [`IoError::Io`] if a write to `w` fails.
///
/// A diagram carrying a weight store and no marginal level is written; the
/// file drops the store, so it reads back in integer mode and the caller
/// attaches weights again with [`Tdd::set_weights`](crate::Tdd::set_weights).
///
/// Save the vtree and diagram together, then restore them:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::io::{read_tdd, write_tdd};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let vtree_text = vtree.to_text();
/// let mut bytes = Vec::new();
/// write_tdd(&mut bytes, &f)?;
///
/// let loaded_vtree = Arc::new(Vtree::from_text(&vtree_text)?);
/// let loaded = read_tdd(&mut bytes.as_slice(), &loaded_vtree)?;
/// let expected = Tdd::clause(&loaded_vtree, [1, -2])?;
/// assert!(loaded.equivalent(&expected)?);
/// # tididi::test_helpers::assert_canonical(&f);
/// # tididi::test_helpers::assert_canonical(&loaded);
/// # tididi::test_helpers::assert_canonical(&expected);
/// # Ok::<(), Box<dyn std::error::Error>>(())
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
        push_num(&mut buf, t.0);
        buf.push(b' ');
        push_num(&mut buf, var.0 + 1); // one-based variable number
        buf.push(b'\n');
    }
    w.write_all(&buf)?;
    buf.clear();

    write_internal_lines(w, tdd, &reachable, &remap, &mut buf)
}

/// A compact record reference travels with the data; the full specification is in rustdoc.
fn push_format_header(buf: &mut Vec<u8>) {
    buf.extend_from_slice(
        b"c tididi: Tree Decision Diagram over a separately stored vtree.\n\
          c Records (whitespace-separated; blank lines and c comments are ignored):\n\
          c   p tdd <version> <num_leaves> <num_vtree_nodes> <out_vtree> <out_local>\n\
          c   L <vtree_idx> <var>\n\
          c   I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> [<l1> <r1> ...]\n\
          c The p record comes first. This writer uses format version ",
    );
    push_num(buf, TDD_FORMAT_VERSION);
    buf.extend_from_slice(
        b".\n\
          c Readers accept only explicitly supported format versions.\n\
          c Output is node (out_vtree, out_local); out_vtree is the vtree root.\n\
          c For false, out_local is ZERO and no L or I records follow.\n\
          c Otherwise every leaf has one L record; var is one-based.\n\
          c Each I record is OR_k(left_k AND right_k), with deterministic pairs.\n\
          c Pair indices are zero-based within their respective child levels.\n\
          c Leaf indices: 0 = true, 1 = positive literal, 2 = negative literal.\n\
          c Internal indices follow I-record order within each vtree level.\n\
          c Writers emit reachable nodes only, children before parents.\n\
          c Weights and marginal values are not stored.\n",
    );
}

/// The `p tdd` line. `out_local` is `None` for the unsatisfiable diagram, which
/// writes the `ZERO` token in its place and ends the file.
fn push_problem_line(buf: &mut Vec<u8>, tdd: &Tdd, out_local: Option<u32>) {
    buf.extend_from_slice(b"p tdd ");
    push_num(buf, TDD_FORMAT_VERSION);
    buf.push(b' ');
    push_num(buf, tdd.vtree.num_leaves());
    buf.push(b' ');
    push_num(buf, tdd.vtree.num_nodes());
    buf.push(b' ');
    push_num(buf, tdd.output.vtree.0);
    match out_local {
        Some(local) => {
            buf.push(b' ');
            push_num(buf, local);
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
            push_num(buf, t.0);
            buf.push(b' ');
            push_num(buf, left_vtree.0);
            buf.push(b' ');
            push_num(buf, right_vtree.0);
            // Marginal levels are refused at entry, so both sides are plain
            // node indices — no value ref can appear here.
            for pair in pairs {
                buf.push(b' ');
                push_num(buf, left_remap[ChildDecoder::structural().node(pair.left).idx()]);
                buf.push(b' ');
                push_num(buf, right_remap[ChildDecoder::structural().node(pair.right).idx()]);
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
