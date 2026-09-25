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

use crate::diagram::{Assembly, ChildPair, NodeIdx, Tdd, TddNodeId};
use crate::diagram::{NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

mod rows;
pub(crate) mod layout;
use layout::Layout;
use rows::{words_per_row, distinct_rows, row_at};

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
    /// Construction sorts the packed rows and groups their projections while
    /// walking the vtree bottom-up. Small projections use counting or radix
    /// sorting; wider ones require comparisons. Work depends on row count,
    /// row width and how the vtree groups constrained variables. Temporary
    /// storage includes packed rows, sorting scratch and unfinished child maps.
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

        let mut layout = self.model_layout().checkout(lim);
        layout.prepare_for(lim, vtree, vars)?;
        if rows.is_empty() || vars.is_empty() {
            return super::constant_on(self, vtree, !rows.is_empty());
        }
        if rows.len() / w > NodeIdx::MAX_LIVE {
            // A level holds at most one node per row, and the pass indexes the
            // rows with the same width the level's nodes are indexed with.
            return Err(OperationError::IndexOverflow);
        }

        let sorted = distinct_rows(lim, vars.len(), &layout, rows, w)?;
        let m = sorted.len() / w;

        let mut assembly = Assembly::new(self, vtree)?;
        let output = fill(self, &mut assembly, vtree, &layout, &sorted, w, m)?;
        // The levels are canonical as built: seat them with nothing to reduce.
        let (levels, _) = assembly.parts_mut();
        Ok(super::seat_canonical(self, vtree, std::mem::take(levels), output))
    }
}

/// Which atom of a finished vtree node each row belongs to.
enum AtomOfRow {
    /// Every row is in the subtree's one atom, which may constrain it.
    Uniform,
    /// A constrained leaf whose two values have different completions.
    /// Read its atom directly from the packed row, including after carrying
    /// the leaf through a subtree whose other variables are free.
    Bit { word: usize, mask: u64 },
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
    fn atom_of(&self, row: usize, sorted: &[u64], w: usize) -> usize {
        match &self.atoms {
            AtomOfRow::Uniform => 0,
            AtomOfRow::Bit { word, mask } => usize::from(sorted[row * w + word] & mask != 0),
            AtomOfRow::PerRow(of_row) => of_row[row] as usize,
        }
    }
}

impl Charged for Finished {
    fn charged_bytes(&self) -> u64 {
        let index = match &self.atoms {
            AtomOfRow::Uniform | AtomOfRow::Bit { .. } => 0,
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
    /// The level's pairs, each above its atom, ready to sort.
    pairs: Vec<u128>,
    /// One atom's pairs, as the assembly takes them.
    pair_list: Vec<ChildPair>,
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
    assembly: &mut Assembly<'_>,
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
    let varying = varying_bits(lim, sorted, w)?;

    for t in vtree.bottomup() {
        lim.check_stop()?;
        let leaf = vtree.node(t).is_leaf();
        let width = layout.count[t.idx()] as usize;
        let finished = match (width, one_sided_child(vtree, layout, t)) {
            (0, _) => free_subtree(eng, assembly, vtree, &state, t, leaf)?,
            (_, _) if leaf => leaf_atoms(lim, sorted, w, layout.lo[t.idx()] as usize, &varying)?,
            (_, Some(constrained)) => {
                carry_child(eng, assembly, vtree, &mut state, t, constrained)?
            }
            _ if width == num_vars => {
                close_subtree(eng, assembly, vtree, &state, &mut scratch, t, sorted, w)?
            }
            _ => {
                let lo = layout.lo[t.idx()] as usize;
                let span = ValueSpan { lo, width, at_top: lo + width == num_vars };
                let atoms = group_rows(lim, &mut scratch, sorted, w, span, m)?;
                store_level(eng, assembly, vtree, &state, &mut scratch, t, atoms, sorted, w)?
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

/// All constrained columns lie below this node, so every row has the same
/// empty external completion. Its single atom is the union of the realized
/// child-atom pairs; an atom index per row and completion hashing add nothing.
#[allow(clippy::too_many_arguments)]
fn close_subtree(
    eng: &Engine, assembly: &mut Assembly<'_>, vtree: &Vtree,
    state: &[Option<Finished>], scratch: &mut Scratch, t: VtreeIdx,
    sorted: &[u64], w: usize,
) -> Result<Finished, OperationError> {
    let lim = eng.limits();
    let (left, right) = vtree.children(t);
    let (l, r) = (below(state, left), below(state, right));
    let pairs = &mut scratch.pair_list;
    pairs.clear();
    let rows = sorted.len() / w;
    lim.reserve_exact(pairs, rows)?;
    let mut gate = lim.gate();
    for row in 0..rows {
        gate.poll(1)?;
        pairs.push(ChildPair::new(l.locals[l.atom_of(row, sorted, w)], r.locals[r.atom_of(row, sorted, w)]));
    }
    gate.flush()?;
    pairs.sort_unstable();
    pairs.dedup();
    lim.check_stop()?;
    assembly.reserve(eng, t, 1, pairs.len())?;
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, 1)?;
    locals.push(assembly.push(eng, t, pairs)?);
    Ok(Finished { atoms: AtomOfRow::Uniform, locals })
}

/// Bits that differ from the first row somewhere in the nonempty relation.
fn varying_bits(lim: &Limits, sorted: &[u64], w: usize) -> Result<Vec<u64>, OperationError> {
    let mut varying = Vec::new();
    lim.try_resize(&mut varying, w, 0u64)?;
    let first = &sorted[..w];
    let mut gate = lim.gate();
    for row in sorted.chunks_exact(w).skip(1) {
        gate.poll(w as u64)?;
        for j in 0..w { varying[j] |= row[j] ^ first[j]; }
    }
    gate.flush()?;
    Ok(varying)
}

/// A leaf has one literal atom, two literal atoms, or one free atom. The
/// two values merge exactly when toggling the bit permutes the row set.
fn leaf_atoms(
    lim: &Limits, sorted: &[u64], w: usize, bit: usize, varying: &[u64],
) -> Result<Finished, OperationError> {
    let (word, mask) = (bit / 64, 1u64 << (bit % 64));
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, 2)?;
    let atoms = if varying[word] & mask == 0 {
        locals.push(if sorted[word] & mask == 0 { NEG_LEAF_IDX } else { POS_LEAF_IDX });
        AtomOfRow::Uniform
    } else if independent_bit(lim, sorted, w, word, mask)? {
        locals.push(ONE_LEAF_IDX);
        AtomOfRow::Uniform
    } else {
        locals.extend([NEG_LEAF_IDX, POS_LEAF_IDX]);
        AtomOfRow::Bit { word, mask }
    };
    Ok(Finished { atoms, locals })
}

/// Sorted distinct rows closed under one bit's toggle. Within each higher-
/// bit prefix, the zero and one runs must have equal lower-bit sequences.
/// Compare those sequences exactly; a hash cannot establish independence.
fn independent_bit(
    lim: &Limits, sorted: &[u64], w: usize, word: usize, mask: u64,
) -> Result<bool, OperationError> {
    let m = sorted.len() / w;
    if !m.is_multiple_of(2) { return Ok(false); }
    let mut gate = lim.gate();
    let answer = (|| {
        let mut start = 0;
        while start < m {
            gate.poll(1)?;
            if sorted[start * w + word] & mask != 0 { return Ok(false); }
            let mut split = start + 1;
            while split < m && sorted[split * w + word] & mask == 0 {
                gate.poll(1)?;
                split += 1;
            }
            let len = split - start;
            if len > m - split { return Ok(false); }
            for offset in 0..len {
                gate.poll(w as u64)?;
                let a = &sorted[(start + offset) * w..][..w];
                let b = &sorted[(split + offset) * w..][..w];
                for j in 0..w {
                    if a[j] ^ b[j] != if j == word { mask } else { 0 } { return Ok(false); }
                }
            }
            start = split + len;
        }
        Ok(true)
    })();
    gate.flush()?;
    answer
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
    assembly: &mut Assembly<'_>,
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
    assembly.reserve(eng, t, from.locals.len(), 0)?;
    for &child in &from.locals {
        let pair = if constrained == left {
            ChildPair::new(child, free_local)
        } else {
            ChildPair::new(free_local, child)
        };
        locals.push(assembly.push(eng, t, &[pair])?);
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
/// one entry per run. Leaves are handled separately by `leaf_atoms`.
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
        let atom = atom_of_run(lim, scratch, sorted, w, hasher.finish(), i, j)?;
        for &k in &scratch.order[i..j] {
            of_row[k as usize] = atom as u32;
        }
        let entry = (scratch.order[i], atom as u32);
        lim.try_push(&mut scratch.run_atoms, entry)?;
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
    lim: &Limits,
    scratch: &mut Scratch,
    sorted: &[u64],
    w: usize,
    hash: u64,
    i: usize,
    j: usize,
) -> Result<usize, OperationError> {
    let first = scratch.head.get(&hash).copied().unwrap_or(u32::MAX);
    let mut candidate = first;
    while candidate != u32::MAX {
        let (start, len) = scratch.runs[candidate as usize];
        if len as usize == j - i
            && same_completions(scratch, sorted, w, start as usize, len as usize, i)
        {
            return Ok(candidate as usize);
        }
        candidate = scratch.next[candidate as usize];
    }
    let atom = scratch.runs.len();
    lim.try_push(&mut scratch.runs, (i as u32, (j - i) as u32))?;
    lim.try_push(&mut scratch.next, first)?;
    lim.reserve_map(&mut scratch.head, 1)?;
    scratch.head.insert(hash, atom as u32);
    Ok(atom)
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
#[allow(clippy::too_many_arguments)]
fn store_level(
    eng: &Engine,
    assembly: &mut Assembly<'_>,
    vtree: &Arc<Vtree>,
    state: &[Option<Finished>],
    scratch: &mut Scratch,
    t: VtreeIdx,
    atoms: Atoms,
    sorted: &[u64],
    w: usize,
) -> Result<Finished, OperationError> {
    let lim = eng.limits();
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, atoms.count)?;
    let (left, right) = vtree.children(t);
    let (under_left, under_right) = (below(state, left), below(state, right));
    if atoms.count == scratch.run_atoms.len() {
        // Each value run introduced a new atom, in encounter order. Its
        // single child pair needs neither sorting nor deduplication.
        assembly.reserve(eng, t, atoms.count, atoms.count)?;
        let mut gate = lim.gate();
        for &(row, atom) in &scratch.run_atoms {
            gate.poll(1)?;
            debug_assert_eq!(atom as usize, locals.len());
            let l = under_left.locals[under_left.atom_of(row as usize, sorted, w)];
            let r = under_right.locals[under_right.atom_of(row as usize, sorted, w)];
            locals.push(assembly.push(eng, t, &[ChildPair::new(l, r)])?);
        }
        gate.flush()?;
        return Ok(Finished { atoms: AtomOfRow::PerRow(atoms.of_row), locals });
    }
    scratch.pairs.clear();
    lim.reserve_exact(&mut scratch.pairs, scratch.run_atoms.len())?;
    let mut gate = lim.gate();
    for &(row, atom) in &scratch.run_atoms {
        gate.poll(1)?;
        // Every row of a run shares this node's value, hence both children's
        // values, hence both children's atoms: one pair per run.
        let l = under_left.locals[under_left.atom_of(row as usize, sorted, w)];
        let r = under_right.locals[under_right.atom_of(row as usize, sorted, w)];
        scratch.pairs.push(((atom as u128) << 64) | ((l.0 as u128) << 32) | r.0 as u128);
    }
    gate.flush()?;
    // Two values of one atom can decompose into the same pair of child atoms.
    scratch.pairs.sort_unstable();
    scratch.pairs.dedup();

    assembly.reserve(eng, t, atoms.count, scratch.pairs.len())?;
    let mut at = 0usize;
    for a in 0..atoms.count {
        scratch.pair_list.clear();
        while at < scratch.pairs.len() && (scratch.pairs[at] >> 64) as usize == a {
            let packed = scratch.pairs[at] as u64;
            let pair = ChildPair::new(NodeIdx((packed >> 32) as u32), NodeIdx(packed as u32));
            lim.try_push(&mut scratch.pair_list, pair)?;
            at += 1;
        }
        debug_assert!(!scratch.pair_list.is_empty(), "every atom is realized by a row");
        locals.push(assembly.push(eng, t, &scratch.pair_list)?);
    }
    Ok(Finished { atoms: AtomOfRow::PerRow(atoms.of_row), locals })
}

#[cfg(test)]
#[path = "tests/models.rs"]
mod tests;
