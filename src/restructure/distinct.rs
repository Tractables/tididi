//! Count, per assignment to one subtree, the distinct assignments of its
//! sibling: the projection onto a subtree and the distinct-count threshold,
//! each read off the levels on one root-to-node path.
//!
//! Every model of a diagram `f` passes through exactly one node of each
//! level, and every node a pair names is satisfiable, so `∃` of the
//! variables outside a vtree node `p`'s subtree is the disjoint union of
//! the nodes of `p`'s level that the output reaches. Let `key` be a child of
//! `p` and `s` its sibling. Each reached node is the disjoint union of its
//! pairs' products `κ × σ`, `κ` a node of `key`'s level and `σ` one of
//! `s`'s, and an assignment `k` to `key`'s variables lies in at most one
//! `κ`, since the nodes of a level are disjoint. So, with `d` the
//! variables outside `p`'s subtree, the assignments `s` with
//! `∃d. f(k, s, d)` are the disjoint union of the `σ` paired with `k`'s `κ`
//! in the reached nodes, and
//!
//! ```text
//! #{s : ∃d. f(k, s, d)} = Σ |σ|   over the pairs (κ, σ) of the reached nodes.
//! ```
//!
//! One walk down the levels from the root to `p` finds the reached nodes,
//! one bottom-up pass over `s`'s subtree gives every `|σ|`, capped at `m`,
//! and one pass over the reached pairs sums them per `κ`. The keys with at
//! least `m` are the union of the `κ` whose sum reaches `m`: one node
//! holding all their pairs, over the levels below `key` copied. With
//! `m = 1` this is `∃` of every variable outside `key`'s subtree, a node
//! map: no level is regrouped, and no level off the path is read.

use std::sync::Arc;

use super::DistinctError;

use crate::Engine;
use crate::diagram::{
    sort_pairs, Assembly, ChildDecoder, ChildPair, LevelView, NodeIdx, Tdd, TddNodeId, LEAF_WIDTH, NEG_LEAF_IDX,
    ONE_LEAF_IDX, POS_LEAF_IDX,
};
use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx};

impl Engine {
    /// The assignments `k` to the variables under `key` with at least `m`
    /// distinct assignments `s` to the variables under `key`'s sibling such
    /// that `∃d. f(k, s, d)`, `d` every variable outside the subtree of
    /// `key`'s parent: `[#{s : ∃d. f(k, s, d)} ≥ m]`, as a diagram on a
    /// vtree of `key`'s subtree, with the same variable ids.
    ///
    /// `key` may be any node but the root. Lay a vtree out for it with the
    /// key and the counted variables as two siblings, the variables to drop
    /// anywhere outside them: `((key, counted), dropped)` reads two levels
    /// above the key. `m = 0` is true and `m = 1` is the projection onto
    /// `key`'s variables, [`project_to_subtree`](Self::project_to_subtree).
    ///
    /// The cost is the pairs of the nodes the output reaches on the levels
    /// from the root down to `key`'s parent, the size of the sibling's
    /// subtree, a copy of the levels under `key`, and the minimization of
    /// the result; no level off that path is read.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // On ((x1 (x2 x3)) x4): x1 is the key, x2 x3 the counted pair, x4
    /// // dropped. f = (x1 ∧ x4) ∨ (¬x1 ∧ x2 ∧ x3): x1 = 1 has four values
    /// // of (x2, x3), x1 = 0 has one.
    /// let engine = Engine::new();
    /// let counted = Vtree::balanced_over(&[VarId(2), VarId(3)])?;
    /// let left = Vtree::join(&Vtree::leaf(VarId(1)), &counted)?;
    /// let vtree = Arc::new(Vtree::join(&left, &Vtree::leaf(VarId(4)))?);
    /// let f = engine.or(engine.cube(&vtree, [1, 4])?, engine.cube(&vtree, [-1, 2, 3])?)?;
    /// let key = vtree.leaf_of(VarId(1)).unwrap();
    /// let two = engine.at_least_distinct(&f, key, 2)?;
    /// assert_eq!(two.vtree().num_leaves(), 1);
    /// assert_eq!(engine.model_count(&two)?, 1u32.into());      // x1 = 1
    /// let one = engine.at_least_distinct(&f, key, 1)?;
    /// assert_eq!(engine.model_count(&one)?, 2u32.into());      // both values
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`DistinctError::Root`] when `key` is the root, which has no sibling;
    /// [`DistinctError::Vtree`] with
    /// [`VtreeError::VariableSpaceTooLarge`](crate::vtree::VtreeError::VariableSpaceTooLarge)
    /// when the diagram's variable ids are too wide for a vtree of `key`'s
    /// subtree to declare;
    /// [`DistinctError::Operation`] with [`OperationError::LevelNotInVtree`]
    /// for a `key` that is not a node of the diagram's vtree,
    /// [`OperationError::MarginalLevel`] for a diagram that has discarded the
    /// structure at a level, and for a refused allocation or an armed stop.
    pub fn at_least_distinct(&self, tdd: &Tdd, key: VtreeIdx, m: u64) -> Result<Tdd, DistinctError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let vtree = tdd.vtree();
        if key.idx() >= vtree.num_nodes() {
            return Err(OperationError::LevelNotInVtree(key).into());
        }
        let Some(parent) = vtree.node(key).parent() else {
            return Err(DistinctError::Root);
        };
        let counted = vtree.sibling(key);
        let mut gate = lim.gate();

        // The result's vtree: `key`'s subtree, its ids kept, and the image
        // there of each node of that subtree.
        let below = subtree(vtree, key);
        // The result keeps the diagram's ids on a vtree of the subtree's
        // nodes, which may be too few to declare them.
        crate::vtree::check_var_space(vtree.num_vars(), below.len()).map_err(DistinctError::Vtree)?;
        let mut keep = Vec::new();
        lim.try_resize(&mut keep, vtree.num_vars() as usize + 1, false)?;
        for &t in &below {
            if vtree.node(t).is_leaf() {
                keep[vtree.leaf_var(t).0 as usize] = true;
            }
        }
        let into = vtree
            .project_to_vars(|v| keep[v.0 as usize].then_some(v), vtree.num_vars())
            .expect("a subtree keeps a variable");
        let into = Arc::new(into);
        let mut image = Vec::new();
        lim.try_resize(&mut image, vtree.num_nodes(), VtreeIdx(0))?;
        let mut stack = vec![(key, into.root())];
        while let Some((s, d)) = stack.pop() {
            image[s.idx()] = d;
            if !vtree.node(s).is_leaf() {
                let ((sl, sr), (dl, dr)) = (vtree.children(s), into.children(d));
                stack.push((sl, dl));
                stack.push((sr, dr));
            }
        }
        gate.poll(below.len() as u64)?;

        if m == 0 {
            return Ok(crate::build::constant_on(self, &into, true)?);
        }
        if tdd.is_zero() {
            return Ok(crate::build::constant_zero(self, &into));
        }

        // `|σ|` for every node of the sibling's level, capped at `m`.
        let size = capped_counts(self, tdd, counted, m)?;

        // The nodes of the parent's level the output reaches, level by
        // level down the path from the root.
        let structural = ChildDecoder::structural();
        let mut path = vec![parent];
        while let Some(up) = vtree.node(*path.last().expect("not empty")).parent() {
            path.push(up);
        }
        let mut reached = vec![tdd.output().local];
        for step in path.windows(2).rev() {
            let (below, above) = (step[0], step[1]);
            let left = vtree.children(above).0 == below;
            let mut seen = Vec::new();
            lim.try_resize(&mut seen, tdd.level(below).nodes().len(), false)?;
            let mut next = Vec::new();
            for &n in &reached {
                let pairs = tdd.level(above).pairs_iter_of_idx(n.idx());
                gate.poll(pairs.len() as u64)?;
                for pair in pairs {
                    let c = structural.node(if left { pair.left } else { pair.right });
                    if !seen[c.idx()] {
                        seen[c.idx()] = true;
                        lim.try_push(&mut next, c)?;
                    }
                }
            }
            reached = next;
        }

        // Each node of `key`'s level, its sum over the reached pairs.
        let key_left = vtree.children(parent).0 == key;
        let key_leaf = vtree.node(key).is_leaf();
        let width = if key_leaf { LEAF_WIDTH } else { tdd.level(key).nodes().len() };
        let mut sum = Vec::new();
        lim.try_resize(&mut sum, width, 0u64)?;
        for &n in &reached {
            let pairs = tdd.level(parent).pairs_iter_of_idx(n.idx());
            gate.poll(pairs.len() as u64)?;
            for pair in pairs {
                let (k, s) = match key_left {
                    true => (structural.node(pair.left), structural.node(pair.right)),
                    false => (structural.node(pair.right), structural.node(pair.left)),
                };
                let at = &mut sum[k.idx()];
                *at = at.saturating_add(size[s.idx()]).min(m);
            }
        }

        if key_leaf {
            // A leaf's nodes are its literals and true; the union of those
            // chosen is one of them, or nothing.
            let chosen = |n: NodeIdx| sum[n.idx()] >= m;
            let local = match (chosen(ONE_LEAF_IDX), chosen(POS_LEAF_IDX), chosen(NEG_LEAF_IDX)) {
                (true, _, _) | (_, true, true) => ONE_LEAF_IDX,
                (false, true, false) => POS_LEAF_IDX,
                (false, false, true) => NEG_LEAF_IDX,
                (false, false, false) => return Ok(crate::build::constant_zero(self, &into)),
            };
            let assembly = Assembly::new(self, &into)?;
            gate.flush()?;
            return Ok(assembly.finish(TddNodeId { vtree: into.root(), local })?);
        }

        // One node holding the pairs of every chosen node, over the levels
        // below `key`.
        let level = tdd.level(key);
        let mut pairs: Vec<ChildPair> = Vec::new();
        for (k, &total) in sum.iter().enumerate() {
            if total >= m {
                let of = level.pairs_iter_of_idx(k);
                gate.poll(of.len() as u64)?;
                lim.reserve_exact(&mut pairs, of.len())?;
                pairs.extend(of);
            }
        }
        if pairs.is_empty() {
            return Ok(crate::build::constant_zero(self, &into));
        }
        sort_pairs(&mut pairs);
        let mut assembly = Assembly::new(self, &into)?;
        for &t in below.iter().rev() {
            if t != key && !vtree.node(t).is_leaf() {
                gate.poll(1)?;
                let view = LevelView::unweighted(tdd.level(t)).expect("a structural diagram");
                assembly.replace_level(self, image[t.idx()], view)?;
            }
        }
        let top = assembly.push(self, into.root(), &pairs)?;
        gate.flush()?;
        let mut result = assembly.finish(TddNodeId { vtree: into.root(), local: top })?;
        self.minimize(&mut result)?;
        Ok(result)
    }

    /// `∃` of every variable outside `key`'s subtree, as a diagram on a
    /// vtree of that subtree with the same variable ids:
    /// [`at_least_distinct`](Self::at_least_distinct) with `m = 1`.
    ///
    /// The union of the nodes of `key`'s level that the output reaches is
    /// the projection, so no level is regrouped: the cost is the reached
    /// pairs on the levels above `key` and a copy of the levels below it.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// // (x1 ∨ x2) ∧ (x3 ∨ x4) on ((x1 x2) (x3 x4)), onto x1 and x2.
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.and(engine.clause(&vtree, [1, 2])?, engine.clause(&vtree, [3, 4])?)?;
    /// let left = vtree.children(vtree.root()).0;
    /// let g = engine.project_to_subtree(&f, left)?;
    /// assert_eq!(engine.model_count(&g)?, 3u32.into());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`at_least_distinct`](Self::at_least_distinct).
    pub fn project_to_subtree(&self, tdd: &Tdd, key: VtreeIdx) -> Result<Tdd, DistinctError> {
        self.at_least_distinct(tdd, key, 1)
    }
}

/// The nodes of `t`'s subtree, each before its children.
fn subtree(vtree: &Vtree, t: VtreeIdx) -> Vec<VtreeIdx> {
    let mut out = Vec::new();
    let mut stack = vec![t];
    while let Some(s) = stack.pop() {
        out.push(s);
        if !vtree.node(s).is_leaf() {
            let (l, r) = vtree.children(s);
            stack.push(r);
            stack.push(l);
        }
    }
    out
}

/// The models of each node of `t`'s level over the variables under `t`,
/// capped at `m`: one slot per node, [`LEAF_WIDTH`] at a leaf.
fn capped_counts(eng: &Engine, tdd: &Tdd, t: VtreeIdx, m: u64) -> Result<Vec<u64>, OperationError> {
    let lim = eng.limits();
    let mut gate = lim.gate();
    let vtree = tdd.vtree();
    let structural = ChildDecoder::structural();
    let leaf = |eng: &Engine| -> Result<Vec<u64>, OperationError> {
        let mut out = Vec::new();
        eng.limits().try_resize(&mut out, LEAF_WIDTH, 0)?;
        out[ONE_LEAF_IDX.idx()] = 2.min(m);
        out[POS_LEAF_IDX.idx()] = 1.min(m);
        out[NEG_LEAF_IDX.idx()] = 1.min(m);
        Ok(out)
    };
    let mut counts: Vec<Option<Vec<u64>>> = Vec::new();
    lim.try_resize(&mut counts, vtree.num_nodes(), None)?;
    for s in subtree(vtree, t).into_iter().rev() {
        let here = match vtree.node(s).is_leaf() {
            true => leaf(eng)?,
            false => {
                let (l, r) = vtree.children(s);
                let (lc, rc) = (counts[l.idx()].take().expect("a child first"), counts[r.idx()].take().expect("a child first"));
                let level = tdd.level(s);
                let mut out = Vec::new();
                lim.try_resize(&mut out, level.nodes().len(), 0)?;
                for (n, pairs) in level.internal_inputs_iter() {
                    let mut total = 0u64;
                    for pair in pairs {
                        let product = lc[structural.node(pair.left).idx()].saturating_mul(rc[structural.node(pair.right).idx()]);
                        total = total.saturating_add(product).min(m);
                    }
                    out[n] = total;
                }
                gate.poll(level.nodes().len() as u64)?;
                out
            }
        };
        counts[s.idx()] = Some(here);
    }
    gate.flush()?;
    Ok(counts[t.idx()].take().expect("computed above"))
}

#[cfg(test)]
#[path = "tests/distinct.rs"]
mod tests;
