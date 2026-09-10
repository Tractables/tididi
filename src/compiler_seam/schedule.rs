//! Deciding which vtree levels may be marginal after each compilation step.

use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// Refine one step's [`marginalize_schedule`] group down to the individual
/// clauses of that step's batch.
///
/// `completes_at[i]` lists the vtree nodes whose last mention within the batch
/// is clause `clause_lits[i]`: once that clause is conjoined, nothing later in
/// the batch reads those subtrees, so they can be summed out mid-batch rather
/// than at the batch's end. Only nodes the cross-step schedule already freed at
/// this step (`cross_step_targets`) are considered — the rest wait for their own
/// step.
///
/// Costs O(total clause length + number of vtree nodes) per batch.
pub fn intra_batch_completions(
    clause_lits: &[&[Literal]],
    vtree: &Vtree,
    cross_step_targets: &[VtreeIdx],
) -> Vec<Vec<VtreeIdx>> {
    let n = vtree.num_nodes();
    let num_vars = vtree.num_vars() as usize;
    let mut freed = vec![false; n];
    for &t in cross_step_targets {
        freed[t.idx()] = true;
    }

    // last_clause_pos[v] = largest i where clause_lits[i] mentions v, else None.
    let mut last_clause_pos: Vec<Option<u32>> = vec![None; num_vars];
    for (i, literals) in clause_lits.iter().enumerate() {
        for lit in *literals {
            let v = lit.var.idx();
            if v < num_vars {
                last_clause_pos[v] = Some(i as u32);
            }
        }
    }

    // Bottom-up max over the vtree. Sub-vtrees outside subtree(current_node)
    // have no in-batch mentions (clauses scoped to current_node have all vars
    // in V_{current_node}), so their completion stays None.
    let mut completion: Vec<Option<u32>> = vec![None; n];
    for &t in vtree.bottomup_slice() {
        match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => {
                let v = var.idx();
                if v < num_vars {
                    completion[t.idx()] = last_clause_pos[v];
                }
            }
            VtreeNode::Internal { left, right, .. } => {
                let cl = completion[left.idx()];
                let cr = completion[right.idx()];
                completion[t.idx()] = match (cl, cr) {
                    (None, None) => None,
                    (Some(a), None) | (None, Some(a)) => Some(a),
                    (Some(a), Some(b)) => Some(a.max(b)),
                };
            }
        }
    }

    let mut completes_at: Vec<Vec<VtreeIdx>> = vec![Vec::new(); clause_lits.len()];
    for node_idx in 0..n {
        if !freed[node_idx] { continue; }
        if let Some(c) = completion[node_idx] {
            completes_at[c as usize].push(VtreeIdx(node_idx as u32));
        }
    }

    completes_at
}

/// Decide which vtree levels may be marginal after each compilation step.
///
/// Returns `schedule[t]`: the vtree nodes whose levels [`marginalize`](crate::marginal::marginalize) may sum
/// out once step `t` completes, each group sorted bottom-up so a node's
/// children are marginal before it. A node is scheduled at the step of the
/// highest-scoped clause mentioning any variable of its subtree — after that
/// step nothing reads the subtree explicitly again.
///
/// `keep_explicit` holds variables that must stay explicit for the whole
/// compile: every vtree node whose subtree contains one is left off the
/// schedule, so its pair structure survives for a later conjunction.
/// `defer_nodes` names nodes at which the caller conjoins a further diagram
/// after the step; every leaf under such a node has its marginalize point lifted to
/// that node's own step, since conjoining an explicit operand against a level
/// already marginal is not defined.
pub fn marginalize_schedule(
    clause_lits: &[&[Literal]],
    vtree: &Vtree,
    clauses_at: &[Vec<usize>],
    keep_explicit: &std::collections::HashSet<VarId>,
    defer_nodes: &[VtreeIdx],
) -> Vec<Vec<VtreeIdx>> {
    let n = vtree.num_nodes();
    let num_vars = vtree.num_vars() as usize;

    let topo_pos_of = |idx: VtreeIdx| -> u32 { vtree.topo_pos(idx) };
    let topo_at = |pos: u32| -> VtreeIdx { vtree.bottomup_slice()[pos as usize] };

    // Step 1: for each variable, find the highest-scoped clause mentioning it.
    let mut last_scope_pos = last_scope_positions(clause_lits, vtree, clauses_at);
    lift_deferred_leaves(vtree, defer_nodes, &mut last_scope_pos);

    // Step 2: compute completion_pos bottom-up.
    let mut completion_pos: Vec<u32> = vec![0; n];
    for &t in vtree.bottomup_slice() {
        match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => {
                let v = var.idx();
                if v < num_vars && last_scope_pos[v] > 0 {
                    completion_pos[t.idx()] = last_scope_pos[v];
                } else {
                    completion_pos[t.idx()] = topo_pos_of(t);
                }
            }
            VtreeNode::Internal { left, right, .. } => {
                completion_pos[t.idx()] = completion_pos[left.idx()]
                    .max(completion_pos[right.idx()]);
            }
        }
    }

    // Step 3: group the nodes by the step that frees them.
    let mut marginalize_at: Vec<Vec<VtreeIdx>> = vec![Vec::new(); n];

    let contains_excluded = subtrees_with_kept_vars(vtree, keep_explicit);
    let exclusion_active = !contains_excluded.is_empty();

    for node_idx in 0..vtree.num_nodes() {
        if vtree.node(VtreeIdx(node_idx as u32)).parent().is_some() {
            if exclusion_active && contains_excluded[node_idx] {
                continue;
            }
            let save_pos = completion_pos[node_idx];
            let save_vtree_idx = topo_at(save_pos);
            marginalize_at[save_vtree_idx.idx()].push(VtreeIdx(node_idx as u32));
        }
    }

    // Sort each group in bottom-up topo order so children are processed first.
    for group in &mut marginalize_at {
        if group.len() > 1 {
            group.sort_by_key(|&idx| topo_pos_of(idx));
        }
    }

    marginalize_at
}

/// For each variable, the bottom-up topo position of the highest-scoped clause
/// that mentions it, or 0 when no clause does.
fn last_scope_positions(
    clause_lits: &[&[Literal]],
    vtree: &Vtree,
    clauses_at: &[Vec<usize>],
) -> Vec<u32> {
    let num_vars = vtree.num_vars() as usize;
    let mut last_scope_pos: Vec<u32> = vec![0; num_vars];
    for (scope_idx, clause_indices) in clauses_at.iter().enumerate() {
        if clause_indices.is_empty() {
            continue;
        }
        let scope_pos = vtree.topo_pos(VtreeIdx(scope_idx as u32));
        for &ci in clause_indices {
            for lit in clause_lits[ci] {
                let v = lit.var.idx();
                if v < num_vars {
                    last_scope_pos[v] = last_scope_pos[v].max(scope_pos);
                }
            }
        }
    }
    last_scope_pos
}

/// Lift every leaf under a defer node so its marginalize point is no earlier
/// than that node's own step.
///
/// The top-down pass (reversed bottom-up topo) carries each defer node's
/// position to its subtree leaves; the caller's bottom-up completion pass then
/// carries the lifted positions back up to the internals. A defer node at the
/// root keeps the whole vtree explicit until the final sum-out.
fn lift_deferred_leaves(vtree: &Vtree, defer_nodes: &[VtreeIdx], last_scope_pos: &mut [u32]) {
    if defer_nodes.is_empty() {
        return;
    }
    let n = vtree.num_nodes();
    let num_vars = vtree.num_vars() as usize;
    let mut defer_to = vec![0u32; n];
    for &t in defer_nodes {
        defer_to[t.idx()] = vtree.topo_pos(t);
    }
    let mut deferral = vec![0u32; n];
    for &t in vtree.bottomup_slice().iter().rev() {
        let base = match vtree.node(t).parent() {
            Some(p) => deferral[p.idx()],
            None => 0,
        };
        deferral[t.idx()] = base.max(defer_to[t.idx()]);
    }
    for &t in vtree.bottomup_slice() {
        if let VtreeNode::Leaf { var, .. } = vtree.node(t) {
            let v = var.idx();
            if v < num_vars {
                last_scope_pos[v] = last_scope_pos[v].max(deferral[t.idx()]);
            }
        }
    }
}

/// Bottom-up "subtree contains a kept variable" bit; a flagged node is omitted
/// from the schedule so its pair structure stays available for a later
/// conjunction. Empty when nothing is kept (the common case).
fn subtrees_with_kept_vars(
    vtree: &Vtree,
    keep_explicit: &std::collections::HashSet<VarId>,
) -> Vec<bool> {
    if keep_explicit.is_empty() {
        return Vec::new();
    }
    let mut v = vec![false; vtree.num_nodes()];
    for &t in vtree.bottomup_slice() {
        match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => {
                v[t.idx()] = keep_explicit.contains(var);
            }
            VtreeNode::Internal { left, right, .. } => {
                v[t.idx()] = v[left.idx()] || v[right.idx()];
            }
        }
    }
    v
}
