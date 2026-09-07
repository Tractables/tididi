//! Microbenchmark: per-call cost of `Vtree::rebuild_topo` vs
//! `Vtree::fixup_topo_after_rotate`. Trajectory divergence in real searches
//! makes the end-to-end A/B uninterpretable, so this isolates the topo-update
//! cost from search dynamics.
//!
//! Run with:
//!   cargo test --release -p tididi vtree::topo_microbench -- --ignored --nocapture
//!
//! Each measurement clones a freshly-rebuilt vtree, applies a single
//! pointer-only rotation, and times the topo update. Reports ns per call.

use std::time::Instant;
use super::rotate::{rotate_left_pointers, rotate_right_pointers};
use super::{RotationKind, Vtree, VtreeIdx, VtreeNode};

fn collect_internals(v: &Vtree) -> Vec<VtreeIdx> {
    v.internal_bottomup().map(|(t, _, _)| t).collect()
}

/// Pick the first internal node `v` for which a `kind` rotation is applicable
/// (i.e. its `w = right(v)` for left, or `left(v)` for right, is itself internal).
fn pickable(v: &Vtree, kind: RotationKind) -> Option<VtreeIdx> {
    for (t, left, right) in v.internal_bottomup() {
        let w = match kind {
            RotationKind::Left => right,
            RotationKind::Right => left,
        };
        if let VtreeNode::Internal { .. } = *v.node(w) {
            return Some(t);
        }
    }
    None
}

fn time_path<F: FnMut()>(iters: u64, mut f: F) -> u64 {
    // Warm up.
    for _ in 0..(iters / 8).max(1) {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_nanos() as u64 / iters
}

#[test]
#[ignore]
fn topo_update_microbench() {
    println!();
    println!(
        "{:>6}  {:>5}  {:>10}  {:>10}  {:>9}  {:>9}",
        "n_vars", "kind", "rebuild ns", "fixup ns", "ratio", "slice"
    );
    println!("{:->60}", "");

    for &n in &[16u32, 32, 64, 128, 256, 512, 1024, 2048] {
        for &kind in &[RotationKind::Left, RotationKind::Right] {
        let base = Vtree::balanced(n);

        // Pick a rotation that is applicable on the *original* vtree.
        let v = pickable(&base, kind).expect("balanced has pickable internal");

        // Take a snapshot AFTER pointer surgery (topo is now stale relative
        // to the new pointer state — this is exactly what the call sites
        // pass in). We need the same starting state for every iteration.
        let snapshot = {
            let mut vt = base.clone();
            let info = match kind {
                RotationKind::Left => rotate_left_pointers(&mut vt, v),
                RotationKind::Right => rotate_right_pointers(&mut vt, v),
            };
            (vt, info.unwrap())
        };

        let iters = (10_000_000u64 / n as u64).max(200);

        // rebuild_topo path: clone, rebuild, drop.
        let rebuild_ns = time_path(iters, || {
            let mut vt = snapshot.0.clone();
            vt.rebuild_topo();
            std::hint::black_box(vt);
        });

        // fixup path: clone, fixup, drop.
        let fixup_ns = time_path(iters, || {
            let mut vt = snapshot.0.clone();
            vt.fixup_topo_after_rotate(&snapshot.1, kind);
            std::hint::black_box(vt);
        });

        // Subtract the clone cost — the call sites don't clone, they call on
        // the live vtree.
        let clone_ns = time_path(iters, || {
            let vt = snapshot.0.clone();
            std::hint::black_box(vt);
        });

        let r = rebuild_ns.saturating_sub(clone_ns);
        let f = fixup_ns.saturating_sub(clone_ns);
        let ratio = if r == 0 { 0.0 } else { f as f64 / r as f64 };
        let mp = match kind {
            RotationKind::Left => snapshot.1.a_idx,
            RotationKind::Right => snapshot.1.c_idx,
        };
        let w_pos = snapshot.0.topo_pos(snapshot.1.w_idx) as usize;
        let m_end = snapshot.0.topo_pos(mp) as usize;
        let slice = if m_end >= w_pos { m_end - w_pos + 1 } else { 0 };
        let kstr = match kind { RotationKind::Left => "L", RotationKind::Right => "R" };
        println!("{n:>6}  {kstr:>5}  {r:>10}  {f:>10}  {ratio:>9.2}  {slice:>9}");

        let _ = collect_internals(&snapshot.0); // keep symbol used
        }
    }
}

/// As above, but also reports the slice size that `fixup_topo_after_rotate`
/// rotates through — gives a sense of how expensive the slice work is
/// relative to total `n`.
#[test]
#[ignore]
fn topo_update_slice_size() {
    use std::sync::atomic::{AtomicU64, Ordering};
    println!();
    println!(
        "{:>6}  {:>10}  {:>12}  {:>10}",
        "n_vars", "n_nodes", "slice_size", "slice/n%"
    );
    println!("{:->50}", "");

    for &n in &[16u32, 32, 64, 128, 256, 512, 1024, 2048] {
        let base = Vtree::balanced(n);
        let kind = RotationKind::Left;
        let v = pickable(&base, kind).unwrap();
        let mut vt = base.clone();
        let info = match kind {
            RotationKind::Left => rotate_left_pointers(&mut vt, v).unwrap(),
            RotationKind::Right => rotate_right_pointers(&mut vt, v).unwrap(),
        };
        let n_nodes = vt.num_nodes();
        let w_pos = vt.topo_pos(info.w_idx) as usize;
        let mp = match kind {
            RotationKind::Left => info.a_idx,
            RotationKind::Right => info.c_idx,
        };
        let m_end = vt.topo_pos(mp) as usize;
        let slice = if m_end >= w_pos { m_end - w_pos + 1 } else { 0 };
        println!(
            "{n:>6}  {n_nodes:>10}  {slice:>12}  {:>9.1}%",
            100.0 * slice as f64 / n_nodes as f64
        );
        let _ = AtomicU64::new(0).load(Ordering::Relaxed); // suppress unused
    }
}
