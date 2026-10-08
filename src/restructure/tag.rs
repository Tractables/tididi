//! Pair the nodes of one level with codes on variables of their own.
//!
//! A level's nodes are disjoint classes of assignments to its variables.
//! Tagging gives each class a code on fresh variables: the result's models
//! are the assignments of a tagged node extended by its code. It is the
//! diagram of a function from the classes to the codes, built from the
//! level's subtree and the codes alone.

use std::collections::hash_map::Entry;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{EmbedError, TagError};

use crate::Engine;
use crate::diagram::{ChildPair, LevelView, NodeIdx, Tdd, TddNodeId, Assembly, LEAF_WIDTH, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Limits, OperationError};
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
    /// nodes that share a code. A cube is built from the top of the code
    /// subtree down, and stops at each node whose leaves read a part of the
    /// code that an earlier cube read there: it shares that cube's node. The
    /// cost is the size of the subtree's levels, plus a step per node of the
    /// cubes and per node they share, and the minimization.
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
    /// discarded the structure at a level under `at` (one beside or above it
    /// is not read), and for a refused allocation or an armed stop.
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
        let source = tdd.vtree();
        if at.idx() >= source.num_nodes() {
            return Err(OperationError::LevelNotInVtree(at).into());
        }
        // Only the levels under `at` are read: a level summed out beside
        // or above it keeps no structure this copies.
        for s in source.bottomup().filter(|&s| under(source, s, at)) {
            tdd.require_structure_at(s)?;
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
        // The cube of each code named, built from the top of the code subtree
        // down to the first node whose leaves read a part of the code that an
        // earlier cube read there too: that node is the earlier cube's.
        let mut cube = Vec::new();
        lim.try_resize(&mut cube, num_codes, None)?;
        let named = {
            let mut seen = Vec::new();
            lim.try_resize(&mut seen, num_codes, false)?;
            tags.iter().flatten().filter(|&&k| !std::mem::replace(&mut seen[k as usize], true)).count()
        };
        let mut memo = Memos::new(lim, into, right, &leaf_bit, named)?;
        let mut stack = Vec::new();
        let mut built = Vec::new();
        for &k in tags.iter().flatten() {
            if cube[k as usize].is_some() {
                continue;
            }
            gate.poll(1)?;
            let code = &codes[k as usize * words..(k as usize + 1) * words];
            let literal = |t: VtreeIdx| {
                let i = leaf_bit[t.idx()];
                match code[i / 64] >> (i % 64) & 1 == 1 {
                    true => POS_LEAF_IDX,
                    false => NEG_LEAF_IDX,
                }
            };
            lim.try_push(&mut stack, Step::Enter(right))?;
            while let Some(step) = stack.pop() {
                match step {
                    Step::Enter(t) if into.node(t).is_leaf() => lim.try_push(&mut built, literal(t))?,
                    Step::Enter(t) => match memo.find(lim, t, code)? {
                        Ok(n) => lim.try_push(&mut built, n)?,
                        Err(slot) => {
                            let (l, r) = into.children(t);
                            lim.try_push(&mut stack, Step::Exit(t, slot))?;
                            lim.try_push(&mut stack, Step::Enter(r))?;
                            lim.try_push(&mut stack, Step::Enter(l))?;
                        }
                    },
                    Step::Exit(t, slot) => {
                        gate.poll(1)?;
                        let r = built.pop().expect("the right child's node");
                        let l = built.pop().expect("the left child's node");
                        let pair = [ChildPair::new(l, r)];
                        // A sub-code not seen at `t` has children no node of
                        // `t` has yet, so it is pushed; a level too wide to key
                        // has no memo and interns instead.
                        let n = match memo.keyed(t) {
                            true => assembly.push(self, t, &pair)?,
                            false => assembly.intern(self, t, &pair)?,
                        };
                        memo.fill(t, slot, n);
                        lim.try_push(&mut built, n)?;
                    }
                }
            }
            cube[k as usize] = Some(built.pop().expect("the cube's top node"));
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

/// A step of the walk that builds one cube: enter a node, or come back to
/// one whose children are built, with the slot its node goes in.
#[derive(Clone, Copy)]
enum Step {
    Enter(VtreeIdx),
    Exit(VtreeIdx, usize),
}

/// The widest part of a code a node is looked up by.
const KEY_BITS: usize = 128;

/// The widest part of a code looked up in an array, one slot per value.
const DIRECT_BITS: usize = 16;

/// No node built yet.
const UNBUILT: NodeIdx = NodeIdx(u32::MAX);

/// The cube nodes built so far at each internal node of the code subtree,
/// by the part of the code its leaves read.
struct Memos {
    /// Per vtree node, the code bits its leaves read, as runs of
    /// `(first bit, length)` in increasing order.
    runs: Vec<Vec<(usize, usize)>>,
    memo: Vec<Memo>,
    /// The nodes the hashed memos name, by the slot their key holds.
    built: Vec<NodeIdx>,
}

#[derive(Clone)]
enum Memo {
    /// Wider than [`KEY_BITS`]: every node is interned.
    Unkeyed,
    /// One slot per value of the part, [`UNBUILT`] where none is yet.
    Direct(Vec<NodeIdx>),
    /// A slot of [`Memos::built`] per part seen.
    Hashed(FxHashMap<u128, u32>),
}

impl Memos {
    /// The memos of the internal nodes under `right`, whose leaves `t` read
    /// bit `leaf_bit[t]` of a code, for a build of `named` cubes.
    fn new(lim: &Limits, into: &Vtree, right: VtreeIdx, leaf_bit: &[usize], named: usize) -> Result<Self, OperationError> {
        let mut runs: Vec<Vec<(usize, usize)>> = Vec::new();
        lim.try_resize(&mut runs, into.num_nodes(), Vec::new())?;
        let mut memo = Vec::new();
        lim.try_resize(&mut memo, into.num_nodes(), Memo::Unkeyed)?;
        // The bits under each node, merged from its children's bottom-up.
        let mut bits: Vec<Vec<usize>> = Vec::new();
        lim.try_resize(&mut bits, into.num_nodes(), Vec::new())?;
        for (t, l, r) in into.internal_bottomup() {
            if !under(into, t, right) {
                continue;
            }
            let side = |c: VtreeIdx, bits: &mut Vec<Vec<usize>>| match into.node(c).is_leaf() {
                true => vec![leaf_bit[c.idx()]],
                false => std::mem::take(&mut bits[c.idx()]),
            };
            let mut mine = side(l, &mut bits);
            mine.extend(side(r, &mut bits));
            mine.sort_unstable();
            let width = mine.len();
            for &b in &mine {
                match runs[t.idx()].last_mut() {
                    Some((first, len)) if *first + *len == b => *len += 1,
                    _ => lim.try_push(&mut runs[t.idx()], (b, 1))?,
                }
            }
            // An array is at most a few slots per cube.
            memo[t.idx()] = match width {
                w if w <= DIRECT_BITS && 1usize << w <= 8 * named + 64 => {
                    let mut slots = Vec::new();
                    lim.try_resize(&mut slots, 1usize << w, UNBUILT)?;
                    Memo::Direct(slots)
                }
                w if w <= KEY_BITS => Memo::Hashed(FxHashMap::default()),
                _ => Memo::Unkeyed,
            };
            bits[t.idx()] = mine;
        }
        Ok(Self { runs, memo, built: Vec::new() })
    }

    /// Whether the nodes of `t` are looked up by the part of the code.
    fn keyed(&self, t: VtreeIdx) -> bool {
        !matches!(self.memo[t.idx()], Memo::Unkeyed)
    }

    /// The part of `code` that `t`'s leaves read, packed low bit first.
    fn key(&self, t: VtreeIdx, code: &[u64]) -> u128 {
        let mut key = 0u128;
        let mut at = 0;
        for &(first, len) in &self.runs[t.idx()] {
            let (mut b, mut left) = (first, len);
            while left > 0 {
                let take = (64 - b % 64).min(left);
                let piece = code[b / 64] >> (b % 64) & u64::MAX >> (64 - take);
                key |= u128::from(piece) << at;
                (at, b, left) = (at + take, b + take, left - take);
            }
        }
        key
    }

    /// The node of `t` built for the part of `code` it reads, or the slot
    /// to [`fill`](Self::fill) with the node about to be built for it.
    fn find(&mut self, lim: &Limits, t: VtreeIdx, code: &[u64]) -> Result<Result<NodeIdx, usize>, OperationError> {
        if !self.keyed(t) {
            return Ok(Err(0));
        }
        let key = self.key(t, code);
        match &mut self.memo[t.idx()] {
            Memo::Unkeyed => unreachable!("keyed above"),
            Memo::Direct(slots) => Ok(match slots[key as usize] {
                UNBUILT => Err(key as usize),
                n => Ok(n),
            }),
            Memo::Hashed(map) => {
                lim.reserve_map(map, 1)?;
                match map.entry(key) {
                    Entry::Occupied(e) => Ok(Ok(self.built[*e.get() as usize])),
                    Entry::Vacant(e) => {
                        let slot = self.built.len();
                        e.insert(slot as u32);
                        lim.try_push(&mut self.built, UNBUILT)?;
                        Ok(Err(slot))
                    }
                }
            }
        }
    }

    /// Record `n` as the node of `t` in the slot [`find`](Self::find) gave.
    fn fill(&mut self, t: VtreeIdx, slot: usize, n: NodeIdx) {
        match &mut self.memo[t.idx()] {
            Memo::Unkeyed => {}
            Memo::Direct(slots) => slots[slot] = n,
            Memo::Hashed(_) => self.built[slot] = n,
        }
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
