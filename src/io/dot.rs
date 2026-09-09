//! DOT/Graphviz visualization for vtrees and TDD circuits.

use std::fmt::Write;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};

use crate::vtree::{Vtree, VtreeIdx};

use crate::diagram::{LeafLabel, Tdd, LEAF_WIDTH};

/// Convert a number to Unicode subscript digits.
fn subscript(n: u32) -> String {
    const SUBSCRIPT_DIGITS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    if n == 0 {
        return SUBSCRIPT_DIGITS[0].to_string();
    }
    let mut digits = Vec::new();
    let mut rem = n;
    while rem > 0 {
        digits.push(SUBSCRIPT_DIGITS[(rem % 10) as usize]);
        rem /= 10;
    }
    digits.reverse();
    digits.into_iter().collect()
}

/// Compute the total number of input pairs across all internal t-nodes at a given vtree level.
fn level_pairs(tdd: &Tdd, idx: VtreeIdx) -> usize {
    let level = tdd.level(idx);
    level.slots().iter().map(|n| level.pairs_iter_of(n).len()).sum()
}

/// Map a normalized intensity t ∈ [0,1] to a fill color and contrasting font color.
///
/// Color scale: light yellow (0.0) → orange (0.5) → dark red (1.0). The fill is
/// a hex triplet, the font color a Graphviz color name.
fn heatmap_color(t: f64) -> (String, &'static str) {
    // Anchor points: light yellow → orange → dark red
    let (r, g, b) = if t < 0.5 {
        let u = t * 2.0;
        lerp_rgb((0xff, 0xff, 0xb2), (0xfd, 0x8d, 0x3c), u)
    } else {
        let u = (t - 0.5) * 2.0;
        lerp_rgb((0xfd, 0x8d, 0x3c), (0x80, 0x00, 0x26), u)
    };
    let fontcolor = if t > 0.65 { "white" } else { "black" };
    (format!("#{:02x}{:02x}{:02x}", r, g, b), fontcolor)
}

fn lerp_rgb(lo: (u8, u8, u8), hi: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let lerp = |a: u8, b: u8| (a as f64 + t * (b as f64 - a as f64)).round() as u8;
    (lerp(lo.0, hi.0), lerp(lo.1, hi.1), lerp(lo.2, hi.2))
}

/// Generate DOT representation of a vtree.
///
/// If `tdd` is provided, internal vtree nodes are filled with a heatmap color
/// (light yellow → dark red) proportional to their input-pair count `s`, and
/// annotated with `w` (t-node count) and `s` via an external xlabel.
/// Leaf annotations are omitted (width is always 2).
pub fn vtree_to_dot(vtree: &Vtree, tdd: Option<&Tdd>) -> String {
    // Pre-compute per-node pairs and max for normalization
    let mut pairs_per_node = vec![0usize; vtree.num_nodes()];
    if let Some(tdd) = tdd {
        for (t, _left, _right) in vtree.internal_bottomup() {
            pairs_per_node[t.idx()] = level_pairs(tdd, t);
        }
    }
    let max_pairs = pairs_per_node.iter().copied().max().unwrap_or(0);

    let mut dot = String::new();
    writeln!(dot, "graph vtree {{").unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();

    // Emit leaf nodes
    for (t, var) in vtree.leaf_bottomup() {
        let i = t.idx();
        // +1: emit 1-indexed DIMACS variable, matching the .tdd / .vtree formats.
        writeln!(dot, "    v{} [shape=box, label=\"X{}\"];", i, subscript(var.0 + 1)).unwrap();
    }
    // Emit internal nodes
    for (t, _left, _right) in vtree.internal_bottomup() {
        let i = t.idx();
        if let Some(tdd) = tdd {
            let s = pairs_per_node[i];
            let norm = if max_pairs > 0 { s as f64 / max_pairs as f64 } else { 0.0 };
            let (fill, font) = heatmap_color(norm);
            writeln!(
                dot,
                "    v{} [shape=circle, style=filled, fillcolor=\"{}\", fontcolor=\"{}\", label=\"{}\", xlabel=<<FONT COLOR=\"#888888\" POINT-SIZE=\"8\">w={} s={}</FONT>>];",
                i, fill, font, i, tdd.effective_width(t), s
            ).unwrap();
        } else {
            writeln!(dot, "    v{} [shape=circle, label=\"{}\"];", i, i).unwrap();
        }
    }

    // Emit tree edges
    for (t, left, right) in vtree.internal_bottomup() {
        let i = t.idx();
        writeln!(dot, "    v{} -- v{};", i, left.0).unwrap();
        writeln!(dot, "    v{} -- v{};", i, right.0).unwrap();
    }

    writeln!(dot, "}}").unwrap();
    dot
}

/// Generate DOT representation of a TDD circuit.
///
/// # Errors
///
/// [`IoError::Format`](super::IoError::Format) if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the rendering is structural (every pair is
/// drawn as edges to its two children) and a level that stores per-node model
/// counts instead of nodes has no such edges to draw.
pub fn tdd_to_dot(f: &Tdd) -> Result<String, super::IoError> {
    super::reject_marginal_levels(f, "tdd_to_dot")?;

    // ZERO sentinel: empty TDD (UNSAT) — return a minimal DOT graph.
    if f.is_zero() {
        return Ok("graph tdd {\n    rankdir=TB;\n    label=\"UNSAT\";\n}\n".to_string());
    }

    let reachable = f.reachable_nodes();
    let mut dot = String::new();
    writeln!(dot, "graph tdd {{").unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(dot, "    compound=true;").unwrap();
    writeln!(dot, "    newrank=true;").unwrap();

    // Nodes, grouped into one cluster per vtree level: internal levels first,
    // so the layout runs top-down, then the leaves.
    for (t, _left, _right) in f.vtree.internal_bottomup().rev() {
        emit_level_cluster(&mut dot, f, &reachable, t);
    }
    for (t, _var) in f.vtree.leaf_bottomup().rev() {
        emit_level_cluster(&mut dot, f, &reachable, t);
    }
    emit_pair_edges(&mut dot, f, &reachable);

    writeln!(dot, "}}").unwrap();
    Ok(dot)
}

/// One vtree level as a dashed subgraph cluster holding its reachable nodes.
/// A leaf level draws its three implicit nodes as labelled boxes; an internal
/// level draws one plain node per stored node. The output node is drawn thick.
fn emit_level_cluster(dot: &mut String, f: &Tdd, reachable: &[Vec<bool>], t: VtreeIdx) {
    if !reachable[t.idx()].iter().any(|&r| r) {
        return;
    }
    writeln!(dot, "    subgraph cluster_v{} {{", t.0).unwrap();
    writeln!(dot, "        label=\"\";").unwrap();
    writeln!(dot, "        style=dashed;").unwrap();
    if f.vtree.node(t).is_leaf() {
        emit_leaf_nodes(dot, f, reachable, t);
    } else {
        emit_internal_nodes(dot, f, reachable, t);
    }
    writeln!(dot, "    }}").unwrap();
}

/// The three implicit nodes of a leaf level, each labelled with its variable in
/// subscript and coloured by its label.
fn emit_leaf_nodes(dot: &mut String, f: &Tdd, reachable: &[Vec<bool>], t: VtreeIdx) {
    // +1: emit 1-indexed DIMACS variable, matching the .tdd / .vtree formats.
    let sub = subscript(f.vtree.leaf_var(t).0 + 1);
    // The index is a leaf-label ordinal, not a position in one array.
    #[allow(clippy::needless_range_loop)]
    for i in 0..LEAF_WIDTH {
        if !reachable[t.idx()][i] {
            continue;
        }
        let (label_str, color) = match LeafLabel::from_idx(i) {
            LeafLabel::One => (format!("1{sub}"), "#90ee90"),
            LeafLabel::Zero => (format!("0{sub}"), "#ffb6c1"),
            LeafLabel::Pos => (format!("X{sub}"), "#6cb4ee"),
            LeafLabel::Neg => (format!("\u{00ac}X{sub}"), "#ffb347"),
        };
        writeln!(
            dot,
            "        v{}_n{} [shape=box, label=\"{}\", style=filled, fillcolor=\"{}\"{}];",
            t.0, i, label_str, color, output_emphasis(f, t, i)
        )
        .unwrap();
    }
}

/// The stored nodes of an internal level, labelled `<vtree>:<local>`.
fn emit_internal_nodes(dot: &mut String, f: &Tdd, reachable: &[Vec<bool>], t: VtreeIdx) {
    for (node, _slot) in f.level(t).slots_iter() {
        let i = node.idx();
        if !reachable[t.idx()][i] {
            continue;
        }
        writeln!(dot, "        v{}_n{} [label=\"{}:{}\"{}];", t.0, i, t.0, i, output_emphasis(f, t, i))
            .unwrap();
    }
}

/// The attribute that thickens the output node's outline, empty for any other.
fn output_emphasis(f: &Tdd, t: VtreeIdx, i: usize) -> &'static str {
    if f.output.vtree == t && f.output.local.idx() == i { ", penwidth=3" } else { "" }
}

/// Every pair as a small junction node with three edges: up to the node that
/// owns the pair, and down to each of its two children. A pair is a conjunction,
/// so drawing it as a point keeps its two sides visually one thing.
fn emit_pair_edges(dot: &mut String, f: &Tdd, reachable: &[Vec<bool>]) {
    for (t, left_vtree, right_vtree) in f.vtree.internal_bottomup() {
        let level = f.level(t);
        let left_view = f.level(left_vtree).side_view();
        let right_view = f.level(right_vtree).side_view();
        for (node, slot) in level.slots_iter() {
            let i = node.idx();
            if !reachable[t.idx()][i] {
                continue;
            }
            for (p, pair) in level.pairs_iter_of(slot).enumerate() {
                let l = child_index(left_view.child(pair.left));
                let r = child_index(right_view.child(pair.right));
                writeln!(dot, "    v{}_n{}_p{} [shape=point, width=0.08];", t.0, i, p).unwrap();
                writeln!(dot, "    v{}_n{} -- v{}_n{}_p{};", t.0, i, t.0, i, p).unwrap();
                writeln!(dot, "    v{}_n{}_p{} -- v{}_n{};", t.0, i, p, left_vtree.0, l).unwrap();
                writeln!(dot, "    v{}_n{}_p{} -- v{}_n{};", t.0, i, p, right_vtree.0, r).unwrap();
            }
        }
    }
}

/// The local index a pair side names. Marginal levels are refused at entry, so
/// no side here carries an inline count.
fn child_index(child: ChildRef) -> usize {
    match child {
        ChildRef::Node(NodeIdx(s)) | ChildRef::Value(ValueRef::Slot(s)) => s as usize,
        ChildRef::Value(ValueRef::Inline(_)) => {
            unreachable!("marginal levels are refused at entry, so no pair can carry an inline marg ref here")
        }
    }
}
