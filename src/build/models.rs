//! Canonical construction from a set of assignments to some of the vtree's
//! variables.
//!
//! The diagram is assembled bottom-up, one node per *atom*. At a vtree node
//! `v`, two values of the variables in `v`'s subtree belong to the same atom
//! when the rows extend them by the same set of assignments to the variables
//! outside that subtree. Atoms are therefore disjoint sets of values, which is
//! what makes the nodes at a level pairwise mutex; grouping the values by
//! their own subfunction instead would produce overlapping nodes that no
//! reduction pass can repair.

use std::hash::Hasher;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHasher};

use crate::diagram::{ChildPair, NodeIdx, Tdd, TddBuilder, TddNodeId};
use crate::diagram::{NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

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
    /// Cost is one sort of the rows and then one pass over the vtree. A vtree
    /// node with constrained variables on both sides costs a pass over the
    /// rows, and fewer than `vars.len()` nodes are like that; a node with them
    /// on one side costs a pass over its own width, and a node with none costs
    /// nothing. Working memory is the packed rows, plus one index per row for
    /// each subtree that is finished and not yet used by its parent — at most
    /// the vtree's depth of those at once.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // Pairs of two-bit numbers: bits 0 and 1 of a row hold the first
    /// // number, bits 2 and 3 the second, high bit first within each.
    /// let vtree = Arc::new(Vtree::linear(5));
    /// let vars: Vec<VarId> = (1..=4).map(VarId).collect();
    /// let row = |a: u64, b: u64| (a >> 1) | ((a & 1) << 1) | ((b >> 1) << 2) | ((b & 1) << 3);
    /// let rows = [row(0, 1), row(1, 2), row(2, 0)];
    ///
    /// let f = Tdd::from_models(&vtree, &vars, &rows)?;
    /// // Three pairs, and the fifth variable is free.
    /// assert_eq!(f.model_count()?, 6u32.into());
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
        from_models(self, vtree, vars, rows)
    }
}

/// Words one packed row occupies. The floor of one keeps a row of no
/// variables a readable slice rather than an empty one.
fn words_per_row(num_vars: usize) -> usize {
    num_vars.div_ceil(64).max(1)
}

/// Where each constrained variable sits once the rows are re-encoded, and how
/// many of them each vtree node's subtree holds.
///
/// Re-encoding puts the variables in the vtree's left-to-right leaf order, so
/// every node's constrained variables become the contiguous bit range
/// `[lo, lo + count)` of a row. Reading a node's value is then a shift and a
/// mask instead of a gather over scattered bits.
struct Layout {
    /// Constrained variables in each vtree node's subtree.
    count: Vec<u32>,
    /// First re-encoded bit position of each vtree node's subtree.
    lo: Vec<u32>,
    /// Re-encoded bit position of each variable of `vars`.
    position: Vec<u32>,
}

impl Layout {
    /// Place `vars` in leaf order, refusing a variable the vtree does not
    /// carry and one that appears twice.
    fn new(lim: &Limits, vtree: &Vtree, vars: &[VarId]) -> Result<Layout, OperationError> {
        let n = vtree.num_nodes();
        let mut count = Vec::new();
        let mut lo = Vec::new();
        let mut position = Vec::new();
        lim.try_resize(&mut count, n, 0u32)?;
        lim.try_resize(&mut lo, n, 0u32)?;
        lim.try_resize(&mut position, vars.len(), 0u32)?;

        for &var in vars {
            let leaf = vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
            if count[leaf.idx()] != 0 {
                return Err(OperationError::DuplicateVariable(var));
            }
            count[leaf.idx()] = 1;
        }
        for (t, left, right) in vtree.internal_bottomup() {
            count[t.idx()] = count[left.idx()] + count[right.idx()];
        }
        for t in vtree.bottomup().rev() {
            if !vtree.node(t).is_leaf() {
                let (left, right) = vtree.children(t);
                lo[left.idx()] = lo[t.idx()];
                lo[right.idx()] = lo[t.idx()] + count[left.idx()];
            }
        }
        for (i, &var) in vars.iter().enumerate() {
            let leaf = vtree.leaf_of(var).expect("checked above");
            position[i] = lo[leaf.idx()];
        }
        Ok(Layout { count, lo, position })
    }

    /// Whether `vars` was already given in leaf order, so that re-encoding a
    /// row is a copy.
    fn is_identity(&self) -> bool {
        self.position.iter().enumerate().all(|(i, &p)| p as usize == i)
    }
}

/// Which atom of a finished vtree node each row belongs to.
enum AtomOfRow {
    /// The subtree constrains nothing, so every row is in its one atom.
    Uniform,
    /// One atom index per row, in the sorted row order.
    PerRow(Vec<u32>),
}

/// A vtree node the pass has finished with, waiting for its parent.
struct Finished {
    atoms: AtomOfRow,
    /// The node each atom was stored at, in atom order.
    locals: Vec<NodeIdx>,
}

impl Finished {
    /// The atom row `row` falls in.
    #[inline]
    fn atom_of(&self, row: usize) -> usize {
        match &self.atoms {
            AtomOfRow::Uniform => 0,
            AtomOfRow::PerRow(of_row) => of_row[row] as usize,
        }
    }
}

impl Charged for Finished {
    fn charged_bytes(&self) -> u64 {
        let index = match &self.atoms {
            AtomOfRow::Uniform => 0,
            AtomOfRow::PerRow(of_row) => of_row.charged_bytes(),
        };
        index + self.locals.charged_bytes()
    }
}

/// The buffers the level pass reuses from one vtree node to the next.
#[derive(Default)]
struct Scratch {
    /// Row indices ordered by their value at the current node.
    order: Vec<u32>,
    /// Sort keys for a node whose value fits one word: the value above the row
    /// index, so the sort is a plain integer sort that leaves rows of equal
    /// value in row order.
    keys: Vec<u128>,
    /// Bucket offsets for the counting sort of a narrow node.
    buckets: Vec<u32>,
    /// A row's words with the current node's own bits cleared.
    outside: Vec<u64>,
    /// The current node's own bits, one mask per word its value reaches into,
    /// starting at word `lo / 64`.
    value: Vec<u64>,
    /// Where in `order` each atom's first run sits, as `(start, length)`.
    runs: Vec<(u32, u32)>,
    /// The next atom sharing an entry's hash, or `u32::MAX`.
    next: Vec<u32>,
    /// The first atom under each hash of a run's completions.
    head: FxHashMap<u64, u32>,
    /// One run's first row and its atom, one entry per run.
    run_atoms: Vec<(u32, u32)>,
    /// Which of the two values each atom of a constrained leaf holds.
    leaf_values: Vec<u8>,
    /// The level's pairs, each above its atom, ready to sort.
    pairs: Vec<u128>,
    /// One atom's pairs, as the builder takes them.
    pair_list: Vec<ChildPair>,
}

/// Build the diagram, handing the level buffers back on any refusal.
fn from_models(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    vars: &[VarId],
    rows: &[u64],
) -> Result<Tdd, OperationError> {
    let w = words_per_row(vars.len());
    if !rows.len().is_multiple_of(w) {
        return Err(OperationError::RaggedRows { words: rows.len(), per_row: w });
    }
    let lim = eng.limits();
    let _op = lim.begin_operation();
    lim.check_stop()?;

    let layout = Layout::new(lim, vtree, vars)?;
    if rows.is_empty() {
        return Ok(super::constant_zero(eng, vtree));
    }
    if vars.is_empty() {
        return Ok(super::constant_one(eng, vtree));
    }
    if rows.len() / w > NodeIdx::MAX_LIVE {
        // A level holds at most one node per row, and the pass indexes the
        // rows with the same width the level's nodes are indexed with.
        return Err(OperationError::IndexOverflow);
    }

    let sorted = distinct_rows(lim, vars.len(), &layout, rows, w)?;
    let m = sorted.len() / w;

    let mut builder = Tdd::builder(eng, vtree)?;
    match fill(eng, &mut builder, vtree, &layout, &sorted, w, m) {
        Ok(output) => Ok(builder.finish(output)?),
        Err(e) => {
            builder.abandon(eng);
            Err(e)
        }
    }
}

/// Re-encode the rows into leaf order, sort them and drop the repeats.
///
/// The order is by bit position, the highest first, which is the order
/// [`compare_value`] reads a node's value in. A node whose bits reach the top
/// of the row therefore finds the rows already grouped — see
/// [`order_by_value`].
fn distinct_rows(
    lim: &Limits,
    num_vars: usize,
    layout: &Layout,
    rows: &[u64],
    w: usize,
) -> Result<Vec<u64>, OperationError> {
    let n = rows.len() / w;
    let mut packed: Vec<u64> = Vec::new();
    lim.try_resize(&mut packed, rows.len(), 0u64)?;
    let mut gate = lim.gate();
    if layout.is_identity() {
        // `vars` already runs in the vtree's leaf order, so re-encoding a row
        // is a copy and only the bits past the last variable have to go.
        let tail = num_vars - (w - 1) * 64;
        let tail_mask = if tail == 64 { !0u64 } else { (1u64 << tail) - 1 };
        for k in 0..n {
            gate.poll(w as u64)?;
            let to = &mut packed[k * w..(k + 1) * w];
            to.copy_from_slice(&rows[k * w..(k + 1) * w]);
            to[w - 1] &= tail_mask;
        }
    } else {
        for k in 0..n {
            gate.poll(w as u64)?;
            for (j, &word) in rows[k * w..(k + 1) * w].iter().enumerate() {
                // A row's bits past the last variable carry no assignment.
                let used = num_vars - j * 64;
                let mut live = if used >= 64 { word } else { word & ((1u64 << used) - 1) };
                while live != 0 {
                    let bit = live.trailing_zeros() as usize;
                    live &= live - 1;
                    let to = layout.position[j * 64 + bit] as usize;
                    packed[k * w + to / 64] |= 1u64 << (to % 64);
                }
            }
        }
    }

    if w == 1 {
        // One word is the whole row, so the words sort and deduplicate where
        // they are and the detour through a permutation buys nothing.
        gate.flush()?;
        packed.sort_unstable();
        packed.dedup();
        return Ok(packed);
    }

    let mut order: Vec<u32> = Vec::new();
    lim.reserve_exact(&mut order, n)?;
    order.extend(0..n as u32);
    order.sort_unstable_by(|&a, &b| compare_row(&packed, w, a, b));

    let mut out: Vec<u64> = Vec::new();
    lim.reserve_exact(&mut out, packed.len())?;
    for &k in &order {
        gate.poll(w as u64)?;
        let row = row_at(&packed, w, k);
        if out.len() < w || &out[out.len() - w..] != row {
            out.extend_from_slice(row);
        }
    }
    gate.flush()?;
    lim.discard(packed);
    lim.discard(order);
    Ok(out)
}

/// Compare two whole rows, high word first, so that the order agrees with
/// [`compare_value`] on any value that reaches the top of the row.
fn compare_row(packed: &[u64], w: usize, a: u32, b: u32) -> std::cmp::Ordering {
    let (x, y) = (row_at(packed, w, a), row_at(packed, w, b));
    for i in (0..w).rev() {
        let ord = x[i].cmp(&y[i]);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// Row `k` of a buffer of `w`-word rows.
#[inline]
fn row_at(packed: &[u64], w: usize, k: u32) -> &[u64] {
    let start = k as usize * w;
    &packed[start..start + w]
}

/// The value of the `width` bits starting at `lo` in a row, for a `width` of
/// at most 64.
#[inline]
fn value_at(row: &[u64], lo: usize, width: usize) -> u64 {
    let (word, bit) = (lo / 64, lo % 64);
    let mut v = row[word] >> bit;
    if bit + width > 64 {
        v |= row[word + 1] << (64 - bit);
    }
    if width < 64 {
        v &= (1u64 << width) - 1;
    }
    v
}

/// The bits of `word` that a value of `width` bits starting at `lo` covers.
#[inline]
fn value_mask(lo: usize, width: usize, word: usize) -> u64 {
    let (start, end) = (lo.max(word * 64), (lo + width).min((word + 1) * 64));
    if start >= end {
        return 0;
    }
    let span = end - start;
    let base = if span == 64 { !0u64 } else { (1u64 << span) - 1 };
    base << (start - word * 64)
}

/// The words a value of `width` bits starting at `lo` reaches into.
#[inline]
fn value_words(lo: usize, width: usize) -> std::ops::RangeInclusive<usize> {
    lo / 64..=(lo + width - 1) / 64
}

/// Whether two rows agree on one node's bits, `mask` holding that node's bits
/// of the words from `first` on — the level computes it once, in
/// `Scratch::value`, rather than deriving it from `(lo, width)` per row.
#[inline]
fn same_value(x: &[u64], y: &[u64], first: usize, mask: &[u64]) -> bool {
    mask.iter().enumerate().all(|(i, &m)| (x[first + i] ^ y[first + i]) & m == 0)
}

/// Compare two rows by the `width` bits starting at `lo`, high word first,
/// which is the order the bit positions run in.
fn compare_value(
    sorted: &[u64],
    w: usize,
    lo: usize,
    width: usize,
    a: u32,
    b: u32,
) -> std::cmp::Ordering {
    let (x, y) = (row_at(sorted, w, a), row_at(sorted, w, b));
    for i in value_words(lo, width).rev() {
        let mask = value_mask(lo, width, i);
        let ord = (x[i] & mask).cmp(&(y[i] & mask));
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// Fill every level bottom-up and return the diagram's output node.
fn fill(
    eng: &Engine,
    builder: &mut TddBuilder,
    vtree: &Arc<Vtree>,
    layout: &Layout,
    sorted: &[u64],
    w: usize,
    m: usize,
) -> Result<TddNodeId, OperationError> {
    let lim = eng.limits();
    let mut scratch = Scratch::default();
    let mut state: Vec<Option<Finished>> = Vec::new();
    lim.reserve_exact(&mut state, vtree.num_nodes())?;
    state.resize_with(vtree.num_nodes(), || None);
    let mut emitted = 0u64;
    let num_vars = layout.count[vtree.root().idx()] as usize;

    for t in vtree.bottomup() {
        lim.check_stop()?;
        let leaf = vtree.node(t).is_leaf();
        let width = layout.count[t.idx()] as usize;
        let finished = match (width, one_sided_child(vtree, layout, t)) {
            (0, _) => free_subtree(eng, builder, vtree, &state, t, leaf)?,
            (_, Some(constrained)) => {
                carry_child(eng, builder, vtree, &mut state, t, constrained)?
            }
            _ => {
                let lo = layout.lo[t.idx()] as usize;
                let span = ValueSpan { lo, width, at_top: lo + width == num_vars };
                let atoms = group_rows(lim, &mut scratch, sorted, w, span, m)?;
                store_level(eng, builder, vtree, &state, &mut scratch, t, atoms)?
            }
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
    builder: &mut TddBuilder,
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
        builder.push(eng, t, &[pair])?
    };
    let mut locals = Vec::new();
    eng.limits().reserve_exact(&mut locals, 1)?;
    locals.push(local);
    Ok(Finished { atoms: AtomOfRow::Uniform, locals })
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
    builder: &mut TddBuilder,
    vtree: &Arc<Vtree>,
    state: &mut [Option<Finished>],
    t: VtreeIdx,
    constrained: VtreeIdx,
) -> Result<Finished, OperationError> {
    let (left, _) = vtree.children(t);
    let free_local = true_node(state, vtree.sibling(constrained));
    let from = state[constrained.idx()].take().expect("a child is finished before its parent");
    let mut locals = Vec::new();
    eng.limits().reserve_exact(&mut locals, from.locals.len())?;
    builder.reserve(eng, t, from.locals.len(), 0)?;
    for &child in &from.locals {
        let pair = if constrained == left {
            ChildPair::new(child, free_local)
        } else {
            ChildPair::new(free_local, child)
        };
        locals.push(builder.push(eng, t, &[pair])?);
    }
    Ok(Finished { atoms: from.atoms, locals })
}

/// Where one vtree node's constrained variables sit in a row: the bit range
/// `[lo, lo + width)`, and whether it reaches the row's top, which is what
/// lets the node read the rows in the order they already lie in.
#[derive(Clone, Copy)]
struct ValueSpan {
    lo: usize,
    width: usize,
    at_top: bool,
}

/// The atoms of one vtree node.
struct Atoms {
    count: usize,
    of_row: Vec<u32>,
}

/// Group the rows at one vtree node into atoms.
///
/// Leaves `scratch.order` holding the row indices ordered by their value at
/// this node, `scratch.runs` the first run of each atom, `scratch.run_atoms`
/// one entry per run, and `scratch.leaf_values` the values each atom of a
/// constrained leaf holds.
fn group_rows(
    lim: &Limits,
    scratch: &mut Scratch,
    sorted: &[u64],
    w: usize,
    span: ValueSpan,
    m: usize,
) -> Result<Atoms, OperationError> {
    let ValueSpan { lo, width, .. } = span;
    order_by_value(lim, scratch, sorted, w, span, m)?;

    let words = value_words(lo, width);
    let first = *words.start();
    scratch.value.clear();
    lim.reserve_exact(&mut scratch.value, words.clone().count())?;
    scratch.outside.clear();
    lim.reserve_exact(&mut scratch.outside, w)?;
    scratch.outside.resize(w, !0u64);
    for i in words {
        let mask = value_mask(lo, width, i);
        scratch.value.push(mask);
        scratch.outside[i] &= !mask;
    }

    scratch.runs.clear();
    scratch.next.clear();
    scratch.head.clear();
    scratch.run_atoms.clear();
    scratch.leaf_values.clear();
    let mut of_row: Vec<u32> = Vec::new();
    lim.try_resize(&mut of_row, m, 0u32)?;

    let mut gate = lim.gate();
    let mut i = 0usize;
    while i < m {
        gate.poll(w as u64)?;
        let head = row_at(sorted, w, scratch.order[i]);
        // One pass finds where the run ends and hashes its completions.
        let mut hasher = FxHasher::default();
        write_completion(&mut hasher, head, &scratch.outside);
        let mut j = i + 1;
        while j < m {
            let row = row_at(sorted, w, scratch.order[j]);
            if !same_value(head, row, first, &scratch.value) {
                break;
            }
            write_completion(&mut hasher, row, &scratch.outside);
            j += 1;
        }
        let atom = atom_of_run(scratch, sorted, w, hasher.finish(), i, j);
        for &k in &scratch.order[i..j] {
            of_row[k as usize] = atom as u32;
        }
        let entry = (scratch.order[i], atom as u32);
        lim.try_push(&mut scratch.run_atoms, entry)?;
        if width == 1 {
            // Only a constrained leaf is this narrow: an internal node with one
            // constrained variable has a free side and never reaches here.
            scratch.leaf_values[atom] |= 1 << value_at(row_at(sorted, w, scratch.order[i]), lo, 1);
        }
        i = j;
    }
    gate.flush()?;
    Ok(Atoms { count: scratch.runs.len(), of_row })
}

/// Order the rows by their value at this node, leaving rows of equal value in
/// row order so that each run lists its completions canonically.
fn order_by_value(
    lim: &Limits,
    scratch: &mut Scratch,
    sorted: &[u64],
    w: usize,
    span: ValueSpan,
    m: usize,
) -> Result<(), OperationError> {
    let ValueSpan { lo, width, at_top } = span;
    scratch.order.clear();
    lim.reserve_exact(&mut scratch.order, m)?;
    if at_top {
        // The rows are sorted by the whole row, highest bit first, so a value
        // that reaches the top of the row already runs in order. Every node on
        // the vtree's right spine is like that, which for a right-linear vtree
        // is every internal node, and they are the widest ones.
        scratch.order.extend(0..m as u32);
    } else if width == 1 {
        // Only a constrained leaf is this narrow, and over two buckets the
        // counting sort is a stable partition: place the zeros from the front
        // and the ones from the back in one pass and turn the ones round,
        // rather than pass over the rows once to count and again to place.
        scratch.order.resize(m, 0);
        let (mut zeros, mut ones) = (0usize, m);
        for k in 0..m {
            let bit = value_at(row_at(sorted, w, k as u32), lo, 1) as usize;
            ones -= bit;
            scratch.order[if bit == 0 { zeros } else { ones }] = k as u32;
            zeros += 1 - bit;
        }
        scratch.order[zeros..].reverse();
    } else if width <= 16 {
        // A counting sort beats a comparison sort while the value range is
        // small, and most nodes of a vtree wider than the query are narrow.
        let range = 1usize << width;
        lim.try_resize(&mut scratch.buckets, range + 1, 0u32)?;
        scratch.buckets[..range + 1].fill(0);
        for k in 0..m {
            scratch.buckets[value_at(row_at(sorted, w, k as u32), lo, width) as usize + 1] += 1;
        }
        for b in 1..=range {
            scratch.buckets[b] += scratch.buckets[b - 1];
        }
        scratch.order.resize(m, 0);
        for k in 0..m {
            let bucket = value_at(row_at(sorted, w, k as u32), lo, width) as usize;
            scratch.order[scratch.buckets[bucket] as usize] = k as u32;
            scratch.buckets[bucket] += 1;
        }
    } else if width <= 64 {
        scratch.keys.clear();
        lim.reserve_exact(&mut scratch.keys, m)?;
        for k in 0..m {
            let v = value_at(row_at(sorted, w, k as u32), lo, width);
            scratch.keys.push(((v as u128) << 32) | k as u128);
        }
        scratch.keys.sort_unstable();
        scratch.order.extend(scratch.keys.iter().map(|&key| key as u32));
    } else {
        scratch.order.extend(0..m as u32);
        scratch
            .order
            .sort_unstable_by(|&a, &b| compare_value(sorted, w, lo, width, a, b).then(a.cmp(&b)));
    }
    Ok(())
}

/// Write one row's completion — its words outside the node's own bits — into
/// the hash of the run it belongs to.
#[inline]
fn write_completion(hasher: &mut FxHasher, row: &[u64], outside: &[u64]) {
    for (word, &mask) in row.iter().zip(outside) {
        hasher.write_u64(word & mask);
    }
}

/// The atom of the run `order[i..j]`, whose completions hash to `hash`, adding
/// one when that set of completions has not been seen at this node.
///
/// Rows of one run share the node's value and so differ only outside it, and
/// the run lists those completions in the order the whole-row sort put them
/// in. Two runs are the same atom exactly when those lists match.
fn atom_of_run(
    scratch: &mut Scratch,
    sorted: &[u64],
    w: usize,
    hash: u64,
    i: usize,
    j: usize,
) -> usize {
    let first = scratch.head.get(&hash).copied().unwrap_or(u32::MAX);
    let mut candidate = first;
    while candidate != u32::MAX {
        let (start, len) = scratch.runs[candidate as usize];
        if len as usize == j - i
            && same_completions(scratch, sorted, w, start as usize, len as usize, i)
        {
            return candidate as usize;
        }
        candidate = scratch.next[candidate as usize];
    }
    let atom = scratch.runs.len();
    scratch.runs.push((i as u32, (j - i) as u32));
    scratch.next.push(first);
    scratch.head.insert(hash, atom as u32);
    scratch.leaf_values.push(0);
    atom
}

/// Whether the run at `start` and the one at `other`, both `len` rows long,
/// list the same completions.
fn same_completions(
    scratch: &Scratch,
    sorted: &[u64],
    w: usize,
    start: usize,
    len: usize,
    other: usize,
) -> bool {
    (0..len).all(|n| {
        let x = row_at(sorted, w, scratch.order[start + n]);
        let y = row_at(sorted, w, scratch.order[other + n]);
        x.iter().zip(y).zip(&scratch.outside).all(|((a, b), mask)| (a ^ b) & mask == 0)
    })
}

/// Store one node per atom, and record where each landed.
fn store_level(
    eng: &Engine,
    builder: &mut TddBuilder,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    scratch: &mut Scratch,
    t: VtreeIdx,
    atoms: Atoms,
) -> Result<Finished, OperationError> {
    let lim = eng.limits();
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, atoms.count)?;
    if vtree.node(t).is_leaf() {
        // A leaf level stores nothing: an atom holding one of the two values is
        // that literal, and an atom holding both leaves the variable free.
        for a in 0..atoms.count {
            locals.push(match scratch.leaf_values[a] {
                0b01 => NEG_LEAF_IDX,
                0b10 => POS_LEAF_IDX,
                _ => ONE_LEAF_IDX,
            });
        }
        return Ok(Finished { atoms: AtomOfRow::PerRow(atoms.of_row), locals });
    }

    let (left, right) = vtree.children(t);
    let (under_left, under_right) = (below(state, left), below(state, right));
    scratch.pairs.clear();
    lim.reserve_exact(&mut scratch.pairs, scratch.run_atoms.len())?;
    let mut gate = lim.gate();
    for &(row, atom) in &scratch.run_atoms {
        gate.poll(1)?;
        // Every row of a run shares this node's value, hence both children's
        // values, hence both children's atoms: one pair per run.
        let l = under_left.locals[under_left.atom_of(row as usize)];
        let r = under_right.locals[under_right.atom_of(row as usize)];
        scratch.pairs.push(((atom as u128) << 64) | ((l.0 as u128) << 32) | r.0 as u128);
    }
    gate.flush()?;
    // Two values of one atom can decompose into the same pair of child atoms.
    scratch.pairs.sort_unstable();
    scratch.pairs.dedup();

    builder.reserve(eng, t, atoms.count, scratch.pairs.len())?;
    let mut at = 0usize;
    for a in 0..atoms.count {
        scratch.pair_list.clear();
        while at < scratch.pairs.len() && (scratch.pairs[at] >> 64) as usize == a {
            let packed = scratch.pairs[at] as u64;
            let pair = ChildPair::new(NodeIdx((packed >> 32) as u32), NodeIdx(packed as u32));
            scratch.pair_list.push(pair);
            at += 1;
        }
        debug_assert!(!scratch.pair_list.is_empty(), "every atom is realized by a row");
        locals.push(builder.push(eng, t, &scratch.pair_list)?);
    }
    Ok(Finished { atoms: AtomOfRow::PerRow(atoms.of_row), locals })
}

#[cfg(test)]
#[path = "tests/models.rs"]
mod tests;
