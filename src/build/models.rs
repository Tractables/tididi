//! Canonical construction from a set of assignments to some of the vtree's
//! variables.
//!
//! The diagram has one node per *atom*. At a vtree node `v`, two values of
//! the variables in `v`'s subtree belong to the same atom when the rows
//! extend them by the same set of assignments to the variables outside that
//! subtree. Atoms are therefore disjoint sets of values, which is what makes
//! the nodes at a level pairwise mutex; grouping the values by their own
//! subfunction instead would produce overlapping nodes that no reduction
//! pass can repair.
//!
//! The atoms are computed from the root down (see `split`), each node from
//! its parent's distinct values, and the nodes are then stored bottom-up.

use std::sync::Arc;

use crate::diagram::{Assembly, ChildPair, NodeIdx, Tdd, TddNodeId, ONE_LEAF_IDX};
use crate::limits::{Charged, OperationError};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

mod rows;
mod split;
pub(crate) mod layout;
use layout::Layout;
use rows::{words_per_row, distinct_rows};
use split::{Decomposition, Plan};

impl Tdd {
    /// The canonical diagram whose models are exactly `rows`, read as
    /// assignments to `vars`, with every other variable of `vtree` free.
    ///
    /// Rows are bit-packed, `vars.len().div_ceil(64).max(1)` words each: with
    /// `w` words per row, row `k` occupies `rows[k * w .. (k + 1) * w]`, and
    /// bit `i` of that row — bit `i % 64` of word `i / 64` — is the value
    /// assigned to `vars[i]`. Bits at or past `vars.len()` are ignored, and a
    /// repeated row denotes one model.
    ///
    /// Empty `rows` gives the constant-false diagram, and an empty `vars` with
    /// at least one row gives the constant-true diagram: the same rule read at
    /// the degenerate size, where a row is one word of ignored bits.
    ///
    /// The result is canonical for `vtree`, so no
    /// [`minimize`](Self::minimize) step follows it, unlike a diagram
    /// assembled through [`TddBuilder`](crate::diagram::TddBuilder).
    ///
    /// Construction sorts the packed rows, then walks the vtree from the root
    /// down, splitting each node's distinct values into its children's, and
    /// finally stores the nodes bottom-up. A node's work is proportional to
    /// its parent's distinct values rather than to the rows, plus the
    /// numbering of its own values when it is a right child. Where a value
    /// is narrow enough to index a table, one pass in the parent's order
    /// hashes them, with no sort; at the root it is a radix sort on the
    /// value alone. Otherwise it is a sort by value: a radix sort of at most
    /// three passes while that value and the value's index fit one word, and
    /// beyond that one counting pass on the top digit followed by a
    /// comparison sort of each run the digit leaves.
    ///
    /// The row sort compares the vtree's leaves from left to right, a false
    /// before a true, so rows that already come in that order are only
    /// checked: with `vars` in leaf order, a table sorted on columns laid out
    /// left to right, each code's bits most significant first. Temporary
    /// storage includes the packed rows, the distinct values of the nodes
    /// still to be split, sorting scratch and every level's pairs.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let vars = [VarId(1), VarId(2), VarId(3)];
    /// // Bit 0 is read access, bit 1 write access, bit 2 sharing.
    /// let rows = [0b001, 0b011, 0b101, 0b011];
    /// let f = Tdd::from_models(&vtree, &vars, &rows)?;
    /// println!("Distinct permission sets: {}", f.model_count()?);
    /// # assert_eq!(f.model_count()?, 3u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a variable of `vars` that is
    /// not a leaf of `vtree`, [`OperationError::DuplicateVariable`] for a
    /// repeated one, [`OperationError::RaggedRows`] when `rows.len()` is not a
    /// multiple of the words per row, [`OperationError::OverBudget`] for a
    /// refused allocation, [`OperationError::IndexOverflow`] when a level would
    /// outgrow the index that addresses it, and [`OperationError::Stopped`]
    /// when an armed stop fires.
    pub fn from_models(
        vtree: &Arc<Vtree>,
        vars: &[VarId],
        rows: &[u64],
    ) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.from_models(vtree, vars, rows))
    }
}

impl Engine {
    /// Run [`Tdd::from_models`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::from_models`].
    pub fn from_models(
        &self,
        vtree: &Arc<Vtree>,
        vars: &[VarId],
        rows: &[u64],
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let w = words_per_row(vars.len());
        if !rows.len().is_multiple_of(w) {
            return Err(OperationError::RaggedRows { words: rows.len(), per_row: w });
        }

        let mut layout = self.scratch.model_layout.checkout(self);
        layout.prepare_for(lim, vtree, vars)?;
        if rows.is_empty() || vars.is_empty() {
            return super::constant_on(self, vtree, !rows.is_empty());
        }
        if rows.len() / w > NodeIdx::MAX_LIVE {
            // A level holds at most one node per row, and the pass indexes the
            // rows with the same width the level's nodes are indexed with.
            return Err(OperationError::IndexOverflow);
        }

        // The split sorts through the same buffers the rows did.
        let mut radix = rows::Radix::default();
        let sorted = distinct_rows(lim, &mut radix, vars.len(), &layout, rows, w)?;
        let mut plans = split::plan(lim, vtree, &layout, sorted, w, radix)?;

        let mut assembly = Assembly::new(self, vtree)?;
        let output = fill(self, &mut assembly, vtree, &layout, &mut plans)?;
        // The levels are canonical as built: seat them with nothing to reduce.
        let (levels, _) = assembly.parts_mut();
        Ok(super::seat_canonical(self, vtree, std::mem::take(levels), output))
    }
}

/// A vtree node the pass has finished with, waiting for its parent.
struct Finished {
    /// The node each atom was stored at, in atom order.
    locals: Vec<NodeIdx>,
}

impl Charged for Finished {
    fn charged_bytes(&self) -> u64 {
        self.locals.charged_bytes()
    }
}

/// Store every level bottom-up from the plans and return the diagram's
/// output node.
fn fill(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    layout: &Layout,
    plans: &mut [Option<Plan>],
) -> Result<TddNodeId, OperationError> {
    let lim = eng.limits();
    let mut state: Vec<Option<Finished>> = Vec::new();
    lim.reserve_exact(&mut state, vtree.num_nodes())?;
    state.resize_with(vtree.num_nodes(), || None);
    let mut pair_list = Vec::new();
    let mut emitted = 0u64;

    for t in vtree.bottomup() {
        lim.check_stop()?;
        let leaf = vtree.node(t).is_leaf();
        let finished = match (layout.count[t.idx()], one_sided_child(vtree, layout, t)) {
            (0, _) => free_subtree(eng, assembly, vtree, &state, t, leaf)?,
            (_, Some(constrained)) => {
                carry_child(eng, assembly, vtree, &state, t, constrained)?
            }
            _ => match plans[t.idx()].take().expect("every split node is planned") {
                Plan::Leaf(locals) => Finished { locals },
                Plan::Branch(split) => {
                    store_level(eng, assembly, vtree, &state, &mut pair_list, t, split)?
                }
            },
        };
        if !leaf {
            // A leaf level stores no node, so it emits none either.
            emitted += finished.locals.len() as u64;
            lim.check_output_cap(emitted)?;
            let (left, right) = vtree.children(t);
            for child in [left, right] {
                if let Some(done) = state[child.idx()].take() {
                    lim.discard(done);
                }
            }
        }
        state[t.idx()] = Some(finished);
    }

    let root = vtree.root();
    let done = state[root.idx()].as_ref().expect("the root was just finished");
    debug_assert_eq!(done.locals.len(), 1, "the root level holds one atom");
    Ok(TddNodeId { vtree: root, local: done.locals[0] })
}

/// The constrained child of an internal node whose other child constrains
/// nothing. `None` for a leaf, and for an internal node with constrained
/// variables on both sides or on neither.
fn one_sided_child(vtree: &Vtree, layout: &Layout, t: VtreeIdx) -> Option<VtreeIdx> {
    if vtree.node(t).is_leaf() {
        return None;
    }
    let (left, right) = vtree.children(t);
    match (layout.count[left.idx()], layout.count[right.idx()]) {
        (0, 0) => None,
        (0, _) => Some(right),
        (_, 0) => Some(left),
        _ => None,
    }
}

/// A node whose subtree holds no constrained variable: one atom, true over
/// every assignment to the subtree's leaves.
fn free_subtree(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    t: VtreeIdx,
    leaf: bool,
) -> Result<Finished, OperationError> {
    let local = if leaf {
        ONE_LEAF_IDX
    } else {
        let (left, right) = vtree.children(t);
        let pair = ChildPair::new(true_node(state, left), true_node(state, right));
        assembly.push(eng, t, &[pair])?
    };
    let mut locals = Vec::new();
    eng.limits().reserve_exact(&mut locals, 1)?;
    locals.push(local);
    Ok(Finished { locals })
}

/// The node a finished free subtree stored its one atom at.
fn true_node(state: &[Option<Finished>], t: VtreeIdx) -> NodeIdx {
    below(state, t).locals[0]
}

/// The finished record of a child.
fn below(state: &[Option<Finished>], t: VtreeIdx) -> &Finished {
    state[t.idx()].as_ref().expect("a child is finished before its parent")
}

/// An internal node whose constrained variables all sit under one child: its
/// atoms are that child's, each paired with the free side's true node.
fn carry_child(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    t: VtreeIdx,
    constrained: VtreeIdx,
) -> Result<Finished, OperationError> {
    let (left, _) = vtree.children(t);
    let free_local = true_node(state, vtree.sibling(constrained));
    let from = below(state, constrained);
    let mut locals = Vec::new();
    eng.limits().reserve_exact(&mut locals, from.locals.len())?;
    assembly.reserve(eng, t, from.locals.len(), 0)?;
    for &child in &from.locals {
        let pair = if constrained == left {
            ChildPair::new(child, free_local)
        } else {
            ChildPair::new(free_local, child)
        };
        locals.push(assembly.push(eng, t, &[pair])?);
    }
    Ok(Finished { locals })
}

/// Store one node per atom of a node constrained on both sides, and record
/// where each landed.
#[allow(clippy::too_many_arguments)]
fn store_level(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    pair_list: &mut Vec<ChildPair>,
    t: VtreeIdx,
    split: Decomposition,
) -> Result<Finished, OperationError> {
    let lim = eng.limits();
    let (left, right) = vtree.children(t);
    let (l, r) = (&below(state, left).locals, &below(state, right).locals);
    // A child whose nodes are not in atom order, as a leaf's literals are
    // not, reorders the pairs.
    let ascending = |nodes: &[NodeIdx]| nodes.windows(2).all(|n| n[0] < n[1]);
    let (atoms, triples) = match split {
        Decomposition::Triples { atoms, triples } => (atoms, triples),
        Decomposition::Grouped { ends, lows } => {
            // The one atom's pairs, which run in child-atom order.
            assembly.reserve(eng, t, 1, lows.len())?;
            lim.gate().poll(lows.len() as u64)?;
            pair_list.clear();
            lim.reserve_exact(pair_list, lows.len())?;
            let mut start = 0;
            for (&high, &end) in l.iter().zip(&ends) {
                pair_list.extend(lows[start..end as usize].iter().map(|&low| ChildPair::new(high, r[low as usize])));
                start = end as usize;
            }
            if !(ascending(l) && ascending(r)) {
                pair_list.sort_unstable();
            }
            let mut locals = Vec::new();
            lim.reserve_exact(&mut locals, 1)?;
            locals.push(assembly.push(eng, t, pair_list)?);
            lim.discard(ends);
            lim.discard(lows);
            return Ok(Finished { locals });
        }
    };
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, atoms)?;
    assembly.reserve(eng, t, atoms, triples.len())?;
    let mut gate = lim.gate();
    if atoms == triples.len() {
        // Each atom has one pair, which needs neither sorting nor merging.
        for &[atom, a, b] in &triples {
            gate.poll(1)?;
            debug_assert_eq!(atom as usize, locals.len());
            let pair = ChildPair::new(l[a as usize], r[b as usize]);
            locals.push(assembly.push(eng, t, &[pair])?);
        }
    } else {
        // The triples run in child-atom order.
        let ordered = ascending(l) && ascending(r);
        pair_list.clear();
        lim.reserve_exact(pair_list, triples.len())?;
        pair_list.extend(triples.iter().map(|&[_, a, b]| ChildPair::new(l[a as usize], r[b as usize])));
        let mut at = 0usize;
        for atom in 0..atoms {
            let start = at;
            while at < triples.len() && triples[at][0] as usize == atom {
                at += 1;
            }
            debug_assert!(at > start, "every atom is realized by a row");
            gate.poll((at - start) as u64)?;
            let pairs = &mut pair_list[start..at];
            if !ordered {
                pairs.sort_unstable();
            }
            locals.push(assembly.push(eng, t, pairs)?);
        }
    }
    gate.flush()?;
    lim.discard(triples);
    Ok(Finished { locals })
}

#[cfg(test)]
#[path = "tests/models.rs"]
mod tests;
