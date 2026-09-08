//! DOT/Graphviz visualization for vtrees and TDD circuits.

use std::fmt::Write;

use crate::vtree::{Vtree, VtreeIdx};

use crate::diagram::{LeafLabel, MargResolved, Tdd, LEAF_WIDTH, resolve_marg_ref};

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
    level.nodes.iter().map(|n| level.pairs_iter_of(n).len()).sum()
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
/// Returns `Err(ErrorKind::InvalidInput)` if the diagram has a marginal level
/// ([`Tdd::has_marginal_level`]) — the rendering is structural (every pair is
/// drawn as edges to its two children) and a level that stores per-node model
/// counts instead of nodes has no such edges to draw.
pub fn tdd_to_dot(tdd: &Tdd) -> std::io::Result<String> {
    super::reject_marginal_levels(tdd, "tdd_to_dot")?;

    let vtree = &tdd.vtree;

    // ZERO sentinel: empty TDD (UNSAT) — return a minimal DOT graph.
    if tdd.is_zero() {
        return Ok("graph tdd {\n    rankdir=TB;\n    label=\"UNSAT\";\n}\n".to_string());
    }

    let reachable = tdd.reachable_nodes();

    let mut dot = String::new();
    writeln!(dot, "graph tdd {{").unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(dot, "    compound=true;").unwrap();
    writeln!(dot, "    newrank=true;").unwrap();

    // Emit nodes grouped by vtree level (top-down layout = reversed bottom-up).
    // Each vtree level becomes a DOT subgraph cluster containing its TDD nodes.
    let mut emit_cluster = |t: VtreeIdx| {
        let level = tdd.level(t);
        let is_leaf_level = vtree.node(t).is_leaf();
        let has_reachable = reachable[t.idx()].iter().any(|&r| r);
        if !has_reachable {
            return;
        }

        writeln!(dot, "    subgraph cluster_v{} {{", t.0).unwrap();
        writeln!(dot, "        label=\"\";").unwrap();
        writeln!(dot, "        style=dashed;").unwrap();

        if is_leaf_level {
            // Implicit leaf nodes: iterate 0..LEAF_WIDTH
            let var = vtree.leaf_var(t);
            // +1: emit 1-indexed DIMACS variable, matching the .tdd / .vtree formats.
            let sub = subscript(var.0 + 1);
            for i in 0..LEAF_WIDTH {
                if !reachable[t.idx()][i] {
                    continue;
                }
                let label = LeafLabel::from_idx(i);
                let node_id = format!("v{}_n{}", t.0, i);
                let is_output = tdd.output.vtree == t && tdd.output.local.idx() == i;
                let (label_str, color) = match label {
                    LeafLabel::One => (format!("1{}", sub), "#90ee90"),
                    LeafLabel::Zero => (format!("0{}", sub), "#ffb6c1"),
                    LeafLabel::Pos => (format!("X{}", sub), "#6cb4ee"),
                    LeafLabel::Neg => (format!("\u{00ac}X{}", sub), "#ffb347"),
                };
                let extra = if is_output { ", penwidth=3" } else { "" };
                writeln!(
                    dot,
                    "        {} [shape=box, label=\"{}\", style=filled, fillcolor=\"{}\"{}];",
                    node_id, label_str, color, extra
                ).unwrap();
            }
        } else {
            // Internal level: iterate stored nodes
            for (i, _node) in level.nodes.iter().enumerate() {
                if !reachable[t.idx()][i] {
                    continue;
                }
                let node_id = format!("v{}_n{}", t.0, i);
                let is_output = tdd.output.vtree == t && tdd.output.local.idx() == i;
                let extra = if is_output { ", penwidth=3" } else { "" };
                writeln!(
                    dot,
                    "        {} [label=\"{}:{}\"{}];",
                    node_id, t.0, i, extra
                ).unwrap();
            }
        }

        writeln!(dot, "    }}").unwrap();
    };
    // Internal nodes first (top-down), then leaves
    for (t, _left, _right) in vtree.internal_bottomup().rev() {
        emit_cluster(t);
    }
    for (t, _var) in vtree.leaf_bottomup().rev() {
        emit_cluster(t);
    }

    // Emit edges (input pairs via junction nodes)
    for (t, left_vtree, right_vtree) in vtree.internal_bottomup() {
        let level = tdd.level(t);
        let left_marg = tdd.level(left_vtree).is_marginal();
        let right_marg = tdd.level(right_vtree).is_marginal();
        for (i, node) in level.nodes.iter().enumerate() {
            if !reachable[t.idx()][i] {
                continue;
            }
            for (p, pair) in level.pairs_iter_of(node).enumerate() {
                let l = match resolve_marg_ref(pair.left.0, left_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("marginal levels are refused at entry, so no pair can carry an inline marg ref here"),
                };
                let r = match resolve_marg_ref(pair.right.0, right_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("marginal levels are refused at entry, so no pair can carry an inline marg ref here"),
                };
                // Small junction node to visually group each pair
                let jid = format!("v{}_n{}_p{}", t.0, i, p);
                writeln!(
                    dot,
                    "    {} [shape=point, width=0.08];",
                    jid
                ).unwrap();
                writeln!(
                    dot,
                    "    v{}_n{} -- {};",
                    t.0, i, jid
                ).unwrap();
                writeln!(
                    dot,
                    "    {} -- v{}_n{};",
                    jid, left_vtree.0, l
                ).unwrap();
                writeln!(
                    dot,
                    "    {} -- v{}_n{};",
                    jid, right_vtree.0, r
                ).unwrap();
            }
        }
    }

    writeln!(dot, "}}").unwrap();
    Ok(dot)
}
