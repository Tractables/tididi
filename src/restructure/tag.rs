//! Pair the nodes of one level with codes on variables of their own.
//!
//! A level's nodes are disjoint classes of assignments to its variables.
//! Tagging gives each class a code on fresh variables: the result's models
//! are the assignments of a tagged node extended by its code. It is the
//! diagram of a function from the classes to the codes, built from the
//! level's subtree and the codes alone.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{EmbedError, TagError};

use crate::Engine;
use crate::diagram::{ChildPair, LevelView, NodeIdx, Tdd, TddNodeId, Assembly, LEAF_WIDTH, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree, VtreeError, VtreeIdx};

impl Engine {
    /// The diagram on `into` that pairs each node of `tdd`'s level `at` with
    /// a code: its models are the assignments in a node `n` with
    /// `tags[n] = Some(k)`, renamed through `map`, each extended by code `k`.
    ///
    /// The left child of `into`'s root holds the subtree of `tdd`'s vtree
    /// under `at`, each variable renamed through `map`; the right child's
    /// leaves are exactly `code_vars`. Codes are bit-packed rows over
    /// `code_vars`, as [`Tdd::from_models`] reads its rows: with
    /// `w = code_vars.len().div_ceil(64).max(1)` words each, bit `i` of
    /// `codes[k * w ..]` is the value of `code_vars[i]` in code `k`. `tags`
    /// has one entry per slot of the level — [`LEAF_WIDTH`] for a leaf, one
    /// per node otherwise — and `None` leaves a node out.
    ///
    /// The nodes of a level are disjoint, so the result is a function from
    /// the tagged assignments to the codes, whatever the codes; two nodes
    /// may share one. The levels under `at` are copied, a cube is built for
    /// each code a tag names, and the result is minimized, which merges the
    /// nodes that share a code. The cost is the size of the subtree's levels
    /// plus the codes named times the code variables, and the minimization.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // x1 ∨ x2 on (x1 x2); at the root of (x1 x2) the one node.
    /// let engine = Engine::new();
    /// let pair = Arc::new(Vtree::balanced(2));
    /// let f = engine.clause(&pair, [1, 2])?;
    /// // Tag it with code 0b10 on x3, x4: x3 false, x4 true.
    /// let into = Arc::new(Vtree::join(&Vtree::balanced(2), &Vtree::balanced_over(&[VarId(3), VarId(4)])?)?);
    /// let g = engine.tag_level(&f, pair.root(), &into, |v| v, &[Some(0)], &[VarId(3), VarId(4)], &[0b10])?;
    /// let expected = engine.clause(&into, [1, 2])? & engine.cube(&into, [-3, 4])?;
    /// assert!(engine.equivalent(&g, &expected)?);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`TagError::Placement`] when `into`'s left subtree is not `at`'s
    /// subtree under `map` (as [`Tdd::embed`] reports a shape that does not
    /// match) or `into`'s root is a leaf; [`TagError::CodeVariables`] when
    /// `code_vars` are not exactly the right subtree's leaves;
    /// [`TagError::Tags`] when `tags` does not have one entry per slot or
    /// names a code past the end of `codes`, whose length is not a multiple
    /// of the words per code, or tags both a leaf's true node and one of its
    /// literals, which overlap; [`TagError::Operation`] with
    /// [`OperationError::LevelNotInVtree`] for an `at` that is not a node of
    /// `tdd`'s vtree, [`OperationError::MarginalLevel`] for a diagram that has
    /// discarded the structure at a level, and for a refused allocation or an
    /// armed stop.
    #[allow(clippy::too_many_arguments)]
    pub fn tag_level(
        &self,
        tdd: &Tdd,
        at: VtreeIdx,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
        tags: &[Option<u32>],
        code_vars: &[VarId],
        codes: &[u64],
    ) -> Result<Tdd, TagError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let source = tdd.vtree();
        if at.idx() >= source.num_nodes() {
            return Err(OperationError::LevelNotInVtree(at).into());
        }
        if into.node(into.root()).is_leaf() {
            return Err(TagError::Placement(EmbedError::NotIsomorphic { source: at }));
        }
        let (left, right) = into.children(into.root());
        let mut gate = lim.gate();

        // The image of each node of `at`'s subtree under `into`'s left child.
        let mut image = Vec::new();
        lim.try_resize(&mut image, source.num_nodes(), None)?;
        let mut stack = Vec::new();
        lim.try_push(&mut stack, (at, left))?;
        while let Some((s, d)) = stack.pop() {
            gate.poll(1)?;
            match (source.node(s).is_leaf(), into.node(d).is_leaf()) {
                (true, true) => {
                    let var = map(source.leaf_var(s));
                    if into.leaf_of(var).is_none() {
                        return Err(TagError::Placement(EmbedError::VariableOutOfRange { variable: var, num_vars: into.num_vars() }));
                    }
                    if into.leaf_var(d) != var {
                        return Err(TagError::Placement(EmbedError::NotIsomorphic { source: s }));
                    }
                }
                (false, false) => {
                    let ((sl, sr), (dl, dr)) = (source.children(s), into.children(d));
                    lim.try_push(&mut stack, (sl, dl))?;
                    lim.try_push(&mut stack, (sr, dr))?;
                }
                _ => return Err(TagError::Placement(EmbedError::NotIsomorphic { source: s })),
            }
            image[s.idx()] = Some(d);
        }

        // The code variables: each a leaf under the right child, once.
        let mut position: FxHashMap<VarId, usize> = FxHashMap::default();
        for (i, &v) in code_vars.iter().enumerate() {
            if position.insert(v, i).is_some() {
                return Err(TagError::CodeVariables(VtreeError::OverlappingVariable(v)));
            }
        }
        let mut leaves = 0usize;
        let mut leaf_bit = Vec::new();
        lim.try_resize(&mut leaf_bit, into.num_nodes(), usize::MAX)?;
        for (t, v) in into.leaf_bottomup() {
            if under(into, t, right) {
                let Some(&i) = position.get(&v) else {
                    return Err(TagError::CodeVariables(VtreeError::Invalid(format!("leaf {} under the code subtree is not a code variable", v.0))));
                };
                leaf_bit[t.idx()] = i;
                leaves += 1;
            }
        }
        if leaves != code_vars.len() {
            return Err(TagError::CodeVariables(VtreeError::Invalid("a code variable is not a leaf under the code subtree".to_string())));
        }

        // One tag per slot, naming a code there is.
        let words = code_vars.len().div_ceil(64).max(1);
        if !codes.len().is_multiple_of(words) {
            return Err(TagError::Tags(format!("{} code words are not a whole number of {words}-word codes", codes.len())));
        }
        let num_codes = codes.len() / words;
        let width = match source.node(at).is_leaf() {
            true => LEAF_WIDTH,
            false => tdd.level(at).nodes().len(),
        };
        if tags.len() != width {
            return Err(TagError::Tags(format!("{} tags for a level of {width} slots", tags.len())));
        }
        // A leaf's `One` holds both literals' assignments.
        if source.node(at).is_leaf() && tags[ONE_LEAF_IDX.idx()].is_some() && (tags[POS_LEAF_IDX.idx()].is_some() || tags[NEG_LEAF_IDX.idx()].is_some()) {
            return Err(TagError::Tags("a leaf's true node and a literal overlap".to_string()));
        }
        if let Some(k) = tags.iter().flatten().find(|&&k| k as usize >= num_codes) {
            return Err(TagError::Tags(format!("tag {k} names no code of {num_codes}")));
        }
        if tdd.is_zero() || tags.iter().all(Option::is_none) {
            return Ok(crate::build::constant_zero(self, into));
        }

        let mut assembly = Assembly::new(self, into)?;
        // The subtree, copied level for level.
        for s in source.bottomup() {
            if let Some(d) = image[s.idx()]
                && !source.node(s).is_leaf()
            {
                gate.poll(1)?;
                let view = LevelView::unweighted(tdd.level(s)).expect("a structural diagram");
                assembly.replace_level(self, d, view)?;
            }
        }
        // The cube of each code named, its levels shared where codes agree.
        let internal: Vec<VtreeIdx> = into.internal_bottomup().filter(|&(t, _, _)| under(into, t, right)).map(|(t, _, _)| t).collect();
        let mut cube = Vec::new();
        lim.try_resize(&mut cube, num_codes, None)?;
        let mut node_at = Vec::new();
        lim.try_resize(&mut node_at, into.num_nodes(), NodeIdx(0))?;
        for &k in tags.iter().flatten() {
            if cube[k as usize].is_some() {
                continue;
            }
            gate.poll(code_vars.len() as u64 + 1)?;
            let code = &codes[k as usize * words..(k as usize + 1) * words];
            let literal = |t: VtreeIdx| {
                let i = leaf_bit[t.idx()];
                match code[i / 64] >> (i % 64) & 1 == 1 {
                    true => POS_LEAF_IDX,
                    false => NEG_LEAF_IDX,
                }
            };
            let top = match into.node(right).is_leaf() {
                true => literal(right),
                false => {
                    for &t in &internal {
                        let (l, r) = into.children(t);
                        let side = |c: VtreeIdx| if into.node(c).is_leaf() { literal(c) } else { node_at[c.idx()] };
                        node_at[t.idx()] = assembly.intern(self, t, &[ChildPair::new(side(l), side(r))])?;
                    }
                    node_at[right.idx()]
                }
            };
            cube[k as usize] = Some(top);
        }
        // The root: each tagged node beside its code's cube.
        let mut pairs = Vec::new();
        for (n, tag) in tags.iter().enumerate() {
            if let Some(k) = tag {
                lim.try_push(&mut pairs, ChildPair::new(NodeIdx(n as u32), cube[*k as usize].expect("built above")))?;
            }
        }
        let root = assembly.push(self, into.root(), &pairs)?;
        gate.flush()?;
        let mut result = assembly.finish(TddNodeId { vtree: into.root(), local: root })?;
        self.minimize(&mut result)?;
        Ok(result)
    }
}

/// Whether `t` lies in the subtree of `root`.
fn under(vtree: &Vtree, mut t: VtreeIdx, root: VtreeIdx) -> bool {
    loop {
        if t == root {
            return true;
        }
        match vtree.node(t).parent() {
            Some(p) => t = p,
            None => return false,
        }
    }
}

#[cfg(test)]
#[path = "tests/tag.rs"]
mod tests;
