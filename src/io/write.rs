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
/// # use std::sync::Arc;
/// # use tididi::{Engine, Tdd};
/// # use tididi::io::IoError;
/// #
/// # use tididi::vtree::Vtree;
/// # let vtree = Arc::new(Vtree::balanced(4));
/// # let engine = Engine::new();
/// # let (left, _right) = vtree.children(vtree.root());
/// # let f = Tdd::clause(&vtree, [1, -2])? & Tdd::clause(&vtree, [2, 3])?;
/// use tididi::io::{load_tdd, save_tdd};
///
/// let path = std::env::temp_dir().join("tididi-doc-save.tdd");
/// save_tdd(&f, &path).unwrap();
/// let g = load_tdd(&path, &vtree).unwrap();
/// assert_eq!(g.model_count()?, f.model_count()?);
/// std::fs::remove_file(&path).unwrap();
///
/// // A diagram with a level summed out has no structural form to write.
/// let mut m = f.clone();
/// engine.marginalize_levels(&mut m, &[left]).unwrap();
/// match save_tdd(&m, &path) {
///     Ok(()) => unreachable!("a marginal level cannot be written"),
///     Err(IoError::Format(msg)) => assert!(!msg.is_empty()),
///     Err(IoError::Io(e)) => unreachable!("{e}"),
///     Err(other) => unreachable!("{other}"),
/// }
/// assert!(!path.exists());   // nothing was created
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
/// Save the tree and diagram together, then restore them into a shared domain:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Vtree};
/// use tididi::io::{read_tdd, write_tdd};
///
/// let engine = Engine::new();
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = engine.clause(&vtree, [1, -2])?;
/// let tree_text = vtree.to_text();
/// let mut bytes = Vec::new();
/// write_tdd(&mut bytes, &f)?;
///
/// let loaded_tree = Arc::new(Vtree::from_text(&tree_text)?);
/// let loaded = read_tdd(&mut bytes.as_slice(), &loaded_tree)?;
/// let expected = engine.clause(&loaded_tree, [1, -2])?;
/// assert!(engine.equivalent(&loaded, &expected)?);
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
        push_num(&mut buf, var.0 + 1); // 1-indexed DIMACS variable
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
    push_num(buf, TDD_FORMAT_VERSION);
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
          c       (1-indexed). Each leaf has 3 implicit TDD nodes, not written, with\n\
          c       local indices: 0 = one (constant true), 1 = positive literal\n\
          c       (var=true), 2 = negative literal (var=false).\n\
          c\n\
          c   I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> [<l1> <r1> ...]\n\
          c       An internal TDD node at vtree node <vtree_idx>, decomposing into a\n\
          c       deterministic OR of AND-pairs: the node equals OR_k (left_k AND right_k).\n\
          c       Each pair (<lk> <rk>) references a child by local index: <lk> into the\n\
          c       node list of <left_vtree>, <rk> into that of <right_vtree>.\n\
          c\n\
          c   Local indices are per vtree node, 0-based, in the order nodes are emitted\n\
          c   (leaf locals are the implicit 0/1/2 above; internal locals count I lines at\n\
          c   that vtree_idx, in file order). The output (out_vtree, out_local) uses the\n\
          c   same scheme, and <out_vtree> is always the root of the vtree.\n\
          c\n",
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
