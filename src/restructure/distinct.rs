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

use super::{DistinctError, DistinctKey};

use crate::Engine;
use crate::diagram::{
    Assembly, ChildDecoder, ChildPair, LEAF_WIDTH, LevelView, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX,
    POS_LEAF_IDX, Tdd, TddNodeId, sort_pairs,
};
use crate::limits::{Limits, OperationError, PollGate};
use crate::vtree::{Home, Vtree, VtreeIdx};

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
    /// `key` is a [`DistinctKey`]: a `VtreeIdx`, or the `&[VarId]` under the
    /// node. On a diagram with a [home](Vtree::home), the node is one of the
    /// home, not of the diagram's own vtree: an own index is read as a home
    /// node without an error, while variables name the same node whichever
    /// vtree the diagram is on. `s` ranges over every home variable under
    /// the sibling, those the diagram has no leaf for included, and the
    /// result is on the projection of the same home on the diagram's leaves
    /// under `key` ([`Vtree::sub_projection`]): a constant when it has none
    /// there. On a root vtree, `key` is a node of that vtree.
    ///
    /// The result is certified
    /// ([`Tdd::is_canonical`](crate::Tdd::is_canonical)).
    ///
    /// ```
    /// use tididi::vtree::Home;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // On ((x1 (x2 x3)) x4): x1 is the key, x2 x3 the counted pair, x4
    /// // dropped. f = (x1 ∧ x4) ∨ (¬x1 ∧ x2 ∧ x3): x1 = 1 has four values
    /// // of (x2, x3), x1 = 0 has one.
    /// let engine = Engine::new();
    /// let counted = Vtree::balanced_over(&[VarId(2), VarId(3)])?;
    /// let left = Vtree::join(&Vtree::leaf(VarId(1)), &counted)?;
    /// let vtree = Home::new(Vtree::join(&left, &Vtree::leaf(VarId(4)))?);
    /// let f = engine.or(engine.cube(&vtree, [1, 4])?, engine.cube(&vtree, [-1, 2, 3])?)?;
    /// let key = vtree.leaf_of(VarId(1)).unwrap();
    /// let two = engine.at_least_distinct(&f, key, 2)?;
    /// assert_eq!(two.vtree().num_leaves(), 1);
    /// // Counts are over the home's four variables.
    /// assert_eq!(engine.model_count(&two)?, 8u32.into());      // x1 = 1
    /// let one = engine.at_least_distinct(&f, key, 1)?;
    /// assert_eq!(engine.model_count(&one)?, 16u32.into());     // both values
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`DistinctError::Root`] when `key` is the root (of the home, on a
    /// home), which has no sibling;
    /// [`DistinctError::Key`] for variables that are not exactly the leaves
    /// of one node;
    /// [`DistinctError::Vtree`] with
    /// [`VtreeError::VariableSpaceTooLarge`](crate::vtree::VtreeError::VariableSpaceTooLarge)
    /// when the diagram, on a root, has variable ids too wide for a vtree of
    /// `key`'s subtree to declare;
    /// [`DistinctError::Operation`] with [`OperationError::LevelNotInVtree`]
    /// for a `key` that is not a node of the home, or of the diagram's vtree
    /// on a root,
    /// [`OperationError::MarginalLevel`] for a diagram that has discarded the
    /// structure at a level, and for a refused allocation or an armed stop.
    pub fn at_least_distinct<'k>(
        &self,
        tdd: &Tdd,
        key: impl Into<DistinctKey<'k>>,
        m: u64,
    ) -> Result<Tdd, DistinctError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let vtree = tdd.vtree();
        let key = key_node(lim, vtree, key.into())?;
        if let Some(home) = vtree.home() {
            return distinct_on_home(self, tdd, &home, key, m);
        }
        if key.idx() >= vtree.num_nodes() {
            return Err(OperationError::LevelNotInVtree(key).into());
        }
        if vtree.node(key).parent().is_none() {
            return Err(DistinctError::Root);
        }
        let mut gate = lim.gate();
        let into = project_onto_key_subtree(lim, vtree, key)?;
        if m == 0 {
            return Ok(crate::build::constant_on(self, &into, true)?);
        }
        if tdd.is_zero() {
            return Ok(crate::build::constant_zero(self, &into));
        }
        let size = capped_counts(self, tdd, vtree.sibling(key), m)?;
        threshold(self, &mut gate, tdd, key, &size, m, &into)
    }

    /// `∃` of every variable outside `key`'s subtree, as a diagram on a
    /// vtree of that subtree with the same variable ids:
    /// [`at_least_distinct`](Self::at_least_distinct) with `m = 1`.
    ///
    /// The union of the nodes of `key`'s level that the output reaches is
    /// the projection, so no level is regrouped: the cost is the reached
    /// pairs on the levels above `key` and a copy of the levels below it.
    ///
    /// `key` is read as [`at_least_distinct`](Self::at_least_distinct) reads
    /// it: on a diagram with a [home](Vtree::home), an index is a home node
    /// and an own index is read as one without an error; the result is on
    /// the projection of the home on the diagram's leaves under `key`.
    ///
    /// The result is certified
    /// ([`Tdd::is_canonical`](crate::Tdd::is_canonical)).
    ///
    /// ```
    /// use tididi::vtree::Home;
    /// use tididi::{Engine, Vtree};
    ///
    /// // (x1 ∨ x2) ∧ (x3 ∨ x4) on ((x1 x2) (x3 x4)), onto x1 and x2.
    /// let engine = Engine::new();
    /// let vtree = Home::new(Vtree::balanced(4));
    /// let f = engine.and(engine.clause(&vtree, [1, 2])?, engine.clause(&vtree, [3, 4])?)?;
    /// let left = vtree.children(vtree.root()).0;
    /// let g = engine.project_to_subtree(&f, left)?;
    /// // Three of the four values of (x1, x2), counted over the home's four variables.
    /// assert_eq!(engine.model_count(&g)?, 12u32.into());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`at_least_distinct`](Self::at_least_distinct).
    pub fn project_to_subtree<'k>(&self, tdd: &Tdd, key: impl Into<DistinctKey<'k>>) -> Result<Tdd, DistinctError> {
        self.at_least_distinct(tdd, key, 1)
    }
}

/// The node `key` names: on `vtree`'s home if it has one, else on `vtree`.
fn key_node(lim: &Limits, vtree: &Vtree, key: DistinctKey<'_>) -> Result<VtreeIdx, DistinctError> {
    let vars = match key {
        DistinctKey::Node(node) => return Ok(node),
        DistinctKey::Vars(vars) => vars,
    };
    let refused = || match lim.try_to_vec(vars) {
        Ok(vars) => DistinctError::Key(vars),
        Err(error) => error.into(),
    };
    let mut sorted = lim.try_to_vec(vars)?;
    sorted.sort_unstable();
    if sorted.is_empty() || sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(refused());
    }
    let (node, leaves) = match vtree.link_home() {
        Some(home) => {
            let (mut lo, mut hi) = (u32::MAX, 0);
            for &v in &sorted {
                let rank = home.rank(v).ok_or_else(refused)?;
                (lo, hi) = (lo.min(rank), hi.max(rank));
            }
            let node = home.node_over(lo, hi);
            (node, home.rank_range(node).len())
        }
        None => {
            let mut node = vtree.leaf_of(sorted[0]).ok_or_else(refused)?;
            for &v in &sorted[1..] {
                node = vtree.lca(node, vtree.leaf_of(v).ok_or_else(refused)?);
            }
            (node, vtree.subtree(node).filter(|&t| vtree.node(t).is_leaf()).count())
        }
    };
    match leaves == sorted.len() {
        true => Ok(node),
        false => Err(refused()),
    }
}

/// [`Engine::at_least_distinct`] of a diagram on a projection of `home`,
/// `key` a node of `home`.
///
/// With `K` and `S` the diagram's leaves under `key` and under its sibling,
/// each value of `S` stands for `2^(n - |S|)` values of the sibling's `n`
/// home variables. With `S` empty every satisfying value of `K` has all
/// `2^n` of them; with `K` empty the result is a constant.
fn distinct_on_home(eng: &Engine, tdd: &Tdd, home: &Home, key: VtreeIdx, m: u64) -> Result<Tdd, DistinctError> {
    let (lim, vtree, at_home) = (eng.limits(), tdd.vtree(), home);
    if key.idx() >= at_home.num_nodes() {
        return Err(OperationError::LevelNotInVtree(key).into());
    }
    if at_home.node(key).parent().is_none() {
        return Err(DistinctError::Root);
    }
    let sibling = at_home.sibling(key);
    let (own_key, own_sibling) = (vtree.own_at(key), vtree.own_at(sibling));
    let into = match own_key {
        None => Vtree::empty_projection(home),
        Some(_) => {
            let under = home.rank_range(key);
            vtree.sub_projection(home, |leaf| home.rank(vtree.leaf_var(leaf)).is_some_and(|r| under.contains(&r))).map_err(DistinctError::Vtree)?
        }
    };
    if m == 0 {
        return Ok(crate::build::constant_on(eng, &into, true)?);
    }
    if tdd.is_zero() {
        return Ok(crate::build::constant_zero(eng, &into));
    }
    let used = own_sibling.map_or(0, |s| vtree.subtree(s).filter(|&t| vtree.node(t).is_leaf()).count());
    let unused = home.rank_range(sibling).len() - used;
    // `n` may be past any shift: a nonzero count times `2^unused`, capped.
    let scale = |count: u64| match (count, unused) {
        (0, _) => 0,
        (_, 64..) => m,
        (_, e) => count.saturating_mul(1 << e).min(m),
    };
    let mut gate = lim.gate();
    match (own_key, own_sibling) {
        (Some(k), Some(s)) => {
            debug_assert_eq!(vtree.sibling(k), s, "a projection keeps the parent of its leaf sets");
            let mut size = capped_counts(eng, tdd, s, m)?;
            size.iter_mut().for_each(|count| *count = scale(*count));
            threshold(eng, &mut gate, tdd, k, &size, m, &into)
        }
        (Some(_), None) if scale(1) < m => Ok(crate::build::constant_zero(eng, &into)),
        (Some(k), None) if k == vtree.root() => {
            let mut result = tdd.try_clone_on(eng)?;
            eng.minimize(&mut result)?;
            Ok(result)
        }
        (Some(k), None) => {
            let size = capped_counts(eng, tdd, vtree.sibling(k), 1)?;
            threshold(eng, &mut gate, tdd, k, &size, 1, &into)
        }
        (None, None) => Ok(crate::build::constant_on(eng, &into, scale(1) >= m)?),
        (None, Some(s)) => {
            let size = capped_counts(eng, tdd, s, m)?;
            let mut total = 0u64;
            for n in reached_at_parent(lim, &mut gate, tdd, s)? {
                total = total.saturating_add(size[n.idx()]).min(m);
            }
            gate.flush()?;
            Ok(crate::build::constant_on(eng, &into, scale(total) >= m)?)
        }
    }
}

/// The keys at `key`, a node of `tdd`'s vtree other than its root, whose
/// sums of `size` over the reached pairs of `key`'s parent reach `m`, on
/// `into`, a vtree of `key`'s subtree.
fn threshold(
    eng: &Engine,
    gate: &mut PollGate<'_>,
    tdd: &Tdd,
    key: VtreeIdx,
    size: &[u64],
    m: u64,
    into: &Arc<Vtree>,
) -> Result<Tdd, DistinctError> {
    let lim = eng.limits();
    let vtree = tdd.vtree();
    let parent = vtree.node(key).parent().expect("the key is not the root");
    let reached = reached_at_parent(lim, gate, tdd, parent)?;

    // Each node of `key`'s level, its sum over the reached pairs.
    let structural = ChildDecoder::structural();
    let key_left = vtree.children(parent).0 == key;
    let key_leaf = vtree.node(key).is_leaf();
    let width = if key_leaf {
        LEAF_WIDTH
    } else {
        tdd.level(key).nodes().len()
    };
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
        let local = match (
            chosen(ONE_LEAF_IDX),
            chosen(POS_LEAF_IDX),
            chosen(NEG_LEAF_IDX),
        ) {
            (true, _, _) | (_, true, true) => return Ok(crate::build::constant_on(eng, into, true)?),
            (false, true, false) => POS_LEAF_IDX,
            (false, false, true) => NEG_LEAF_IDX,
            (false, false, false) => return Ok(crate::build::constant_zero(eng, into)),
        };
        let assembly = Assembly::new(eng, into)?;
        gate.flush()?;
        let mut literal = assembly.finish(TddNodeId {
            vtree: into.root(),
            local,
        })?;
        literal.certify_built();
        return Ok(literal);
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
        return Ok(crate::build::constant_zero(eng, into));
    }
    sort_pairs(&mut pairs);
    let mut assembly = Assembly::new(eng, into)?;
    // `into` has `key`'s subtree's shape: walk both in step.
    let mut stack = vec![(key, into.root())];
    while let Some((s, d)) = stack.pop() {
        if vtree.node(s).is_leaf() {
            continue;
        }
        if s != key {
            gate.poll(1)?;
            let view = LevelView::unweighted(tdd.level(s)).expect("a structural diagram");
            assembly.replace_level(eng, d, view)?;
        }
        let ((sl, sr), (dl, dr)) = (vtree.children(s), into.children(d));
        lim.try_push(&mut stack, (sl, dl))?;
        lim.try_push(&mut stack, (sr, dr))?;
    }
    let top = assembly.push(eng, into.root(), &pairs)?;
    gate.flush()?;
    let mut result = assembly.finish(TddNodeId {
        vtree: into.root(),
        local: top,
    })?;
    eng.minimize(&mut result)?;
    Ok(result)
}

/// The result's vtree on a root: `key`'s subtree, its variable ids kept.
fn project_onto_key_subtree(lim: &Limits, vtree: &Vtree, key: VtreeIdx) -> Result<Arc<Vtree>, DistinctError> {
    let below: Vec<VtreeIdx> = vtree.subtree(key).collect();
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
    Ok(Arc::new(into))
}

/// The nodes of `parent`'s level the output reaches, level by level down the
/// path from the root.
fn reached_at_parent(
    lim: &Limits,
    gate: &mut PollGate<'_>,
    tdd: &Tdd,
    parent: VtreeIdx,
) -> Result<Vec<NodeIdx>, OperationError> {
    let vtree = tdd.vtree();
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
        let width = match vtree.node(below).is_leaf() {
            true => LEAF_WIDTH,
            false => tdd.level(below).nodes().len(),
        };
        lim.try_resize(&mut seen, width, false)?;
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
    Ok(reached)
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
    let below: Vec<VtreeIdx> = vtree.subtree(t).collect();
    for s in below.into_iter().rev() {
        let here = match vtree.node(s).is_leaf() {
            true => leaf(eng)?,
            false => {
                let (l, r) = vtree.children(s);
                let (lc, rc) = (
                    counts[l.idx()].take().expect("a child first"),
                    counts[r.idx()].take().expect("a child first"),
                );
                let level = tdd.level(s);
                let mut out = Vec::new();
                lim.try_resize(&mut out, level.nodes().len(), 0)?;
                for (n, pairs) in level.internal_inputs_iter() {
                    let mut total = 0u64;
                    for pair in pairs {
                        let product = lc[structural.node(pair.left).idx()]
                            .saturating_mul(rc[structural.node(pair.right).idx()]);
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
