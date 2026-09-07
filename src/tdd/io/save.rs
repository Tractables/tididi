//! TDD circuit serialization to text format (.tdd files).
//!
//! Format overview:
//! ```text
//! c TiDiDi TDD circuit
//! p tdd <num_vars> <num_vtree_nodes> <output_vtree> <output_local>
//! L <vtree_idx> <var>
//! I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> ...
//! ```
//!
//! - `L` lines: vtree leaf mapping — `<var>` is 1-indexed DIMACS variable.
//!   Each leaf has 3 implicit TDD nodes: one(0), pos(1), neg(2).
//! - `I` lines: TDD internal nodes — `<left_vtree>`/`<right_vtree>` are child vtree indices,
//!   followed by alternating left/right local index pairs
//! - Nodes are written in vtree bottom-up order; only reachable nodes are serialized.
//!
//! Leaf index convention: `one(0), pos(1), neg(2)`.

use std::io::{BufWriter, Write};

use crate::tdd::types::{MargResolved, Tdd, resolve_marg_ref};
use crate::vtree::VtreeIdx;

/// Estimate output size in bytes: ~12 bytes per pair entry + overhead.
fn estimate_size(tdd: &Tdd) -> usize {
    tdd.size() * 12 + 4096
}

/// Write a TDD to a file in .tdd text format.
///
/// Uses `fallocate` to pre-allocate disk space (avoids ext4 metadata updates
/// during writes), an 8MB `BufWriter`, and `itoa` for fast integer formatting.
///
/// # Errors
///
/// Returns `Err(ErrorKind::InvalidInput)` if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the format is structural and cannot express
/// a level that stores per-node model counts instead of nodes. Nothing is
/// written and no file is created in that case.
///
/// Also returns `Err` if the file cannot be created or a write to it fails.
pub fn save_tdd(tdd: &Tdd, path: &str) -> std::io::Result<()> {
    // Checked before `File::create` so a rejected diagram leaves no stray file.
    super::reject_marginal_levels(tdd, "save_tdd")?;

    let file = std::fs::File::create(path)?;

    // Pre-allocate file space to avoid incremental block allocation on ext4.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let est = estimate_size(tdd) as i64;
        // Ignore errors — fallocate is an optimization, not required.
        // SAFETY: `file` is a freshly-opened `std::fs::File`; `as_raw_fd()`
        // returns a valid fd for the file's lifetime, which spans this call.
        // Mode=0 and offset=0 are the documented defaults for "allocate from
        // the start of the file". The call is best-effort and any error is
        // ignored.
        unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, est); }
    }

    let mut w = BufWriter::with_capacity(8 << 20, file); // 8MB buffer
    write_tdd(&mut w, tdd)?;
    w.flush()?;

    // Truncate to actual size (fallocate may have over-allocated).
    let actual = w.into_inner().map_err(|e| e.into_error())?;
    let pos = actual.metadata()?.len();
    actual.set_len(pos)?;

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

/// Write a TDD in .tdd text format to any writer.
///
/// # Errors
///
/// Returns `Err(ErrorKind::InvalidInput)` if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the format is structural and cannot express
/// a level that stores per-node model counts instead of nodes. Nothing is
/// written to `w` in that case.
///
/// Also returns `Err` if a write to `w` fails.
#[doc(hidden)] // test-support: reached only by integration tests
pub fn write_tdd<W: Write>(w: &mut W, tdd: &Tdd) -> std::io::Result<()> {
    super::reject_marginal_levels(tdd, "write_tdd")?;

    let vtree = &tdd.vtree;
    let num_vars = vtree.num_leaves();
    let num_vtree_nodes = vtree.num_nodes();

    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);

    buf.extend_from_slice(b"c TiDiDi TDD circuit\n");
    // Self-documenting header. Lines starting with 'c' are comments and are
    // ignored by readers; keep this block in sync with the writes below.
    buf.extend_from_slice(
        b"c\n\
          c Format: a Tree Decision Diagram (TDD) over a vtree. Whitespace-separated\n\
          c tokens, one record per line. Reachable nodes only, in vtree bottom-up order.\n\
          c\n\
          c   p tdd <num_vars> <num_vtree_nodes> <out_vtree> <out_local>\n\
          c       Problem line. The circuit's output node is (<out_vtree>, <out_local>).\n\
          c       <out_local> is the literal token ZERO when the function is UNSAT\n\
          c       (no further L/I lines follow in that case).\n\
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

    if tdd.is_zero() {
        buf.extend_from_slice(b"p tdd ");
        push_int(&mut buf, num_vars);
        buf.push(b' ');
        push_usize(&mut buf, num_vtree_nodes);
        buf.push(b' ');
        push_int(&mut buf, tdd.output.vtree.0);
        buf.extend_from_slice(b" ZERO\n");
        w.write_all(&buf)?;
        return Ok(());
    }

    let reachable = tdd.reachable_nodes();

    // Build local-index remap: old_idx → new sequential idx (skipping unreachable).
    // Leaf levels keep identity mapping (validator always creates all 3 marginal nodes).
    // Internal levels get compacted: only reachable nodes are written.
    let mut remap: Vec<Vec<u32>> = Vec::with_capacity(vtree.num_nodes());
    for (vi, reach) in reachable.iter().enumerate() {
        let mut map = vec![u32::MAX; reach.len()];
        if vtree.node(VtreeIdx(vi as u32)).is_leaf() {
            for j in 0..reach.len() { map[j] = j as u32; }
        } else {
            let level = tdd.level(crate::vtree::VtreeIdx(vi as u32));
            let mut next = 0u32;
            for (i, _) in level.internal_inputs_iter() {
                if reach[i] {
                    map[i] = next;
                    next += 1;
                }
            }
        }
        remap.push(map);
    }

    let out_local = remap[tdd.output.vtree.idx()][tdd.output.local.idx()];
    debug_assert_ne!(out_local, u32::MAX, "output node must be reachable");

    buf.extend_from_slice(b"p tdd ");
    push_int(&mut buf, num_vars);
    buf.push(b' ');
    push_usize(&mut buf, num_vtree_nodes);
    buf.push(b' ');
    push_int(&mut buf, tdd.output.vtree.0);
    buf.push(b' ');
    push_int(&mut buf, out_local);
    buf.push(b'\n');
    w.write_all(&buf)?;
    buf.clear();

    // Write L lines (vtree leaf → variable mapping): "L <vtree_idx> <var>"
    // Each vtree leaf has 3 implicit TDD nodes: one(0), pos(1), neg(2).
    for (t, var) in vtree.leaf_bottomup() {
        buf.extend_from_slice(b"L ");
        push_int(&mut buf, t.0);
        buf.push(b' ');
        push_int(&mut buf, var.0 + 1); // 1-indexed DIMACS variable
        buf.push(b'\n');
    }
    w.write_all(&buf)?;
    buf.clear();

    // Write I lines (internal TDD nodes): "I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> ..."
    // Pair indices are remapped so the validator's sequential assignment matches.
    for (t, left_vtree, right_vtree) in vtree.internal_bottomup() {
        let level = tdd.level(t);
        let t_val = t.0;
        let reach = &reachable[t.idx()];
        let left_remap = &remap[left_vtree.idx()];
        let right_remap = &remap[right_vtree.idx()];
        let left_marg = tdd.level(left_vtree).is_marginal();
        let right_marg = tdd.level(right_vtree).is_marginal();
        for (i, pairs) in level.internal_inputs_iter() {
            if !reach[i] {
                continue;
            }
            buf.extend_from_slice(b"I ");
            push_int(&mut buf, t_val);
            buf.push(b' ');
            push_int(&mut buf, left_vtree.0);
            buf.push(b' ');
            push_int(&mut buf, right_vtree.0);
            for pair in pairs {
                let l = match resolve_marg_ref(pair.left.0, left_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("marginal levels are refused at entry, so no pair can carry an inline marg ref here"),
                };
                let r = match resolve_marg_ref(pair.right.0, right_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("marginal levels are refused at entry, so no pair can carry an inline marg ref here"),
                };
                buf.push(b' ');
                push_int(&mut buf, left_remap[l]);
                buf.push(b' ');
                push_int(&mut buf, right_remap[r]);
            }
            buf.push(b'\n');
            if buf.len() > 64 * 1024 {
                w.write_all(&buf)?;
                buf.clear();
            }
        }
        if !buf.is_empty() {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    Ok(())
}
