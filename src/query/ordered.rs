//! The first models of a diagram in the order of some of its columns, read
//! without listing the rest.
//!
//! [`Tdd::ordered_models`] reads models as rows of codes, as
//! [`Tdd::model_columns`] does, but only the first `limit` rows in the order
//! of the leading `keys` columns, each ascending or descending;
//! [`Tdd::ordered_keys`] the first `limit` distinct values of the key
//! columns alone, projecting away everything else.
//!
//! The order is read off the vtree: its leaves, left to right, must read
//! the key columns' bits before any other listed bit, column 0's from the
//! most significant down, then column 1's, and so on. Then every vtree node
//! `v` over key bits orders its own: where `v = (L, R)` and both hold key
//! bits, every key bit of `L` is more significant than every one of `R`.
//! Under that layout each node's models over its key bits are a sorted
//! stream:
//!
//! - at a key leaf, the values its label allows, in the column's direction;
//! - at a node whose pairs `(l, r)` both hold key bits, the merge, over its
//!   pairs, of the product of `l`'s stream and `r`'s (left outermost);
//! - at a node with key bits on one side only, the merge of that side's
//!   streams, the other side's node pending.
//!
//! A node's pairs are not disjoint on the keys — two pairs may share a left
//! child, and two left children may share a key value once an unlisted
//! variable is projected — so the merge is a heap over the pairs, each a
//! cursor into its children's streams, and a stream is kept as it is
//! produced, since every pair that references a node reads it. A pair
//! enters its node's heap at its first key, its children's first entries —
//! or, where both children hold key bits, at the bound its left child's
//! first entry gives, its right child read only once that bound reaches the
//! top of the heap. So a stream is produced only as far as a reader asks,
//! and nothing below a node no read reaches is touched. Below the keys, a
//! model's other bits are listed depth first from the nodes its key left
//! pending.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::diagram::{LeafLabel, Tdd};
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};
use crate::Engine;

use super::columns::MAX_COLUMN_BITS;

impl Tdd {
    /// The first `limit` models of this diagram in the order of its first
    /// `keys` columns, as columns of codes (one vector per column, at most
    /// `limit` entries each) — or `None` where the vtree is not laid out for
    /// that order.
    ///
    /// Columns are read as [`model_columns`](Self::model_columns) reads
    /// them: column `j` is the variables `columns[j]` as one unsigned code,
    /// most significant first, and the diagram must not depend on a variable
    /// no column lists. The models are distinct rows, ordered by column 0's
    /// code, ties by column 1's, up to column `keys - 1`'s — each column
    /// largest code first where `descending[j]` — and rows that tie on those
    /// come in a fixed order the call does not choose. A free listed
    /// variable yields `0` before `1`, or `1` before `0` on a descending
    /// column.
    ///
    /// The layout it needs: the vtree's leaves, left to right, read the key
    /// columns' variables before any other listed one, column 0's from the
    /// most significant down, then column 1's, and so on; unlisted
    /// variables may sit anywhere. Otherwise the result is `Ok(None)`.
    ///
    /// The work is the models written, each through the nodes on its path
    /// and the pairs of the nodes it opens, each pair read to its first key;
    /// it does not depend on how many models the diagram has beyond them,
    /// nor on the nodes no such read reaches.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // A stick over 1, 2, 3 reads every leading run of its leaves
    /// // first: order by (1, 2) descending, then 3 ascending.
    /// let vtree = Arc::new(Vtree::linear(3));
    /// let f = Tdd::clause(&vtree, [1, 3])?;
    /// let key: &[VarId] = &[VarId(1), VarId(2)];
    /// let then: &[VarId] = &[VarId(3)];
    /// let first = f.ordered_models(&[key, then], &[true, false], 2, 3)?.expect("a spine");
    /// assert_eq!(first, vec![vec![3, 3, 2], vec![0, 1, 0]]);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`model_columns`](Self::model_columns):
    /// [`OperationError::VariableNotInVtree`],
    /// [`OperationError::DuplicateVariable`],
    /// [`OperationError::ColumnTooWide`],
    /// [`OperationError::UnlistedLiteral`] when a model's path references a
    /// literal of a variable no column lists,
    /// [`OperationError::MarginalLevel`], and [`OperationError::Stopped`]
    /// when an armed stop fires.
    ///
    /// # Panics
    ///
    /// Panics if `descending` does not hold one flag per column or `keys`
    /// exceeds the columns.
    pub fn ordered_models(
        &self,
        columns: &[&[VarId]],
        descending: &[bool],
        keys: usize,
        limit: u64,
    ) -> Result<Option<Vec<Vec<u32>>>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.ordered_models(self, columns, descending, keys, limit))
    }

    /// The first `limit` distinct values the diagram's models take on
    /// `columns`, in their order — column 0's code first, ties by column
    /// 1's, and so on, each column largest first where `descending[j]` —
    /// as columns of codes; or `None` where the vtree is not laid out for
    /// that order.
    ///
    /// Every column is a key, so the layout is
    /// [`ordered_models`](Self::ordered_models)'s with every listed
    /// variable a key; every other variable is projected away, so the
    /// diagram may depend on variables no column lists. Each node's stream
    /// is kept distinct as it is merged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // x1 ∨ x3 on a stick over 1, 2, 3: the key (1, 2) takes the values
    /// // 3, 2 (x1 set) and 1, 0 (x1 clear, x3 set).
    /// let vtree = Arc::new(Vtree::linear(3));
    /// let f = Tdd::clause(&vtree, [1, 3])?;
    /// let key: &[VarId] = &[VarId(1), VarId(2)];
    /// assert_eq!(f.ordered_keys(&[key], &[true], 3)?, Some(vec![vec![3, 2, 1]]));
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`],
    /// [`OperationError::DuplicateVariable`],
    /// [`OperationError::ColumnTooWide`],
    /// [`OperationError::MarginalLevel`], and [`OperationError::Stopped`]
    /// when an armed stop fires.
    ///
    /// # Panics
    ///
    /// Panics if `descending` does not hold one flag per column.
    pub fn ordered_keys(
        &self,
        columns: &[&[VarId]],
        descending: &[bool],
        limit: u64,
    ) -> Result<Option<Vec<Vec<u32>>>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.ordered_keys(self, columns, descending, limit))
    }
}

impl Engine {
    /// Run [`Tdd::ordered_models`] using this batch's resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::ordered_models`]. Stops are checked once per model and per
    /// level of the first pass.
    ///
    /// # Panics
    ///
    /// As [`Tdd::ordered_models`].
    pub fn ordered_models(
        &self,
        tdd: &Tdd,
        columns: &[&[VarId]],
        descending: &[bool],
        keys: usize,
        limit: u64,
    ) -> Result<Option<Vec<Vec<u32>>>, OperationError> {
        assert_eq!(descending.len(), columns.len(), "one direction per column");
        assert!(keys <= columns.len(), "{keys} key columns of {}", columns.len());
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let vtree = Arc::clone(tdd.vtree());
        let role = roles(&vtree, columns)?;
        let key_bits: Vec<VarId> = columns[..keys].iter().flat_map(|c| c.iter().copied()).collect();
        let Some(layout) = KeyLayout::new(&vtree, &role, descending, &key_bits) else { return Ok(None) };
        let mut walk = Walk {
            tdd,
            role: &role,
            descending,
            cur: vec![0u32; columns.len()],
            out: vec![Vec::new(); columns.len()],
            limit,
            cells: Vec::new(),
            gate: lim.gate(),
        };
        if limit == 0 || tdd.is_zero() {
            return Ok(Some(walk.out));
        }
        let root = tdd.output().local.idx() as u32;
        if key_bits.is_empty() {
            walk.cells.push(Cell { t: vtree.root(), n: root, next: NIL });
            walk.go(0)?;
            walk.gate.finish()?;
            return Ok(Some(walk.out));
        }
        let mut streams = Streams::new(tdd, &layout, false);
        let mut at = 0usize;
        loop {
            streams.ensure(vtree.root(), root, at + 1)?;
            walk.gate.poll(std::mem::take(&mut streams.work))?;
            let Some(&(key, rest)) = streams.of(vtree.root(), root).out.get(at) else { break };
            at += 1;
            layout.decode(key, &role, |j, bit, b| walk.set(j, bit, b));
            // The nodes the key left pending, copied into the walk's cells.
            walk.cells.clear();
            let mut list = NIL;
            let mut c = rest;
            while c != NIL {
                let Cell { t, n, next } = streams.cells[c as usize];
                walk.cells.push(Cell { t, n, next: list });
                list = walk.cells.len() as u32 - 1;
                c = next;
            }
            if walk.go(list)? {
                break;
            }
        }
        walk.gate.finish()?;
        Ok(Some(walk.out))
    }

    /// Run [`Tdd::ordered_keys`] using this batch's resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::ordered_keys`]. Stops are checked once per value and per
    /// level of the first pass.
    ///
    /// # Panics
    ///
    /// As [`Tdd::ordered_keys`].
    pub fn ordered_keys(
        &self,
        tdd: &Tdd,
        columns: &[&[VarId]],
        descending: &[bool],
        limit: u64,
    ) -> Result<Option<Vec<Vec<u32>>>, OperationError> {
        assert_eq!(descending.len(), columns.len(), "one direction per column");
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let vtree = Arc::clone(tdd.vtree());
        let role = roles(&vtree, columns)?;
        let key_bits: Vec<VarId> = columns.iter().flat_map(|c| c.iter().copied()).collect();
        let Some(layout) = KeyLayout::new(&vtree, &role, descending, &key_bits) else { return Ok(None) };
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); columns.len()];
        if limit == 0 || tdd.is_zero() {
            return Ok(Some(out));
        }
        if key_bits.is_empty() {
            // No key bit: the one empty value, the codes all 0.
            for column in &mut out {
                column.push(0);
            }
            return Ok(Some(out));
        }
        let mut gate = lim.gate();
        let root = tdd.output().local.idx() as u32;
        let mut streams = Streams::new(tdd, &layout, true);
        let mut cur = vec![0u32; columns.len()];
        for at in 0..usize::try_from(limit).unwrap_or(usize::MAX) {
            streams.ensure(vtree.root(), root, at + 1)?;
            gate.poll(std::mem::take(&mut streams.work))?;
            let Some(&(key, _)) = streams.of(vtree.root(), root).out.get(at) else { break };
            layout.decode(key, &role, |j, bit, b| {
                let c = &mut cur[j as usize];
                *c = (*c & !(1u32 << bit)) | (b << bit);
            });
            for (column, &code) in out.iter_mut().zip(&cur) {
                column.push(code);
            }
            gate.poll(1)?;
        }
        gate.finish()?;
        Ok(Some(out))
    }
}

/// Each variable's column and bit (`0` the least significant) in
/// `columns`.
fn roles(vtree: &Vtree, columns: &[&[VarId]]) -> Result<Vec<Option<(u32, u32)>>, OperationError> {
    let mut role: Vec<Option<(u32, u32)>> = vec![None; vtree.num_vars() as usize + 1];
    for (j, vars) in columns.iter().enumerate() {
        if vars.len() > MAX_COLUMN_BITS {
            return Err(OperationError::ColumnTooWide { column: j, bits: vars.len() });
        }
        for (i, &v) in vars.iter().enumerate() {
            if vtree.leaf_of(v).is_none() {
                return Err(OperationError::VariableNotInVtree(v));
            }
            let slot = &mut role[v.0 as usize];
            if slot.is_some() {
                return Err(OperationError::DuplicateVariable(v));
            }
            *slot = Some((j as u32, (vars.len() - 1 - i) as u32));
        }
    }
    Ok(role)
}

/// The most key bits the walk orders on: a stream's keys are one `u128`.
const MAX_KEY_BITS: usize = 128;

/// The end of a pending list, and a stream entry with nothing pending.
const NIL: u32 = u32::MAX;

/// Where the key bits sit: per vtree node, how many it holds, and per key
/// leaf, whether its column is descending.
struct KeyLayout<'a> {
    vtree: &'a Vtree,
    key_bits: &'a [VarId],
    /// Per vtree node (by index), the key bits below it.
    held: Vec<u32>,
    /// Per variable, whether it is a key bit of a descending column.
    down: Vec<bool>,
}

impl<'a> KeyLayout<'a> {
    /// The layout of `key_bits` in `vtree`, or `None` where its leaves, left
    /// to right, do not read them first and in order among the listed
    /// variables (`role`).
    fn new(vtree: &'a Vtree, role: &[Option<(u32, u32)>], descending: &[bool], key_bits: &'a [VarId]) -> Option<KeyLayout<'a>> {
        if key_bits.len() > MAX_KEY_BITS {
            return None;
        }
        let mut is_key = vec![false; vtree.num_vars() as usize + 1];
        let mut down = vec![false; vtree.num_vars() as usize + 1];
        for &v in key_bits {
            is_key[v.0 as usize] = true;
            let (j, _) = role[v.0 as usize].expect("a listed key bit");
            down[v.0 as usize] = descending[j as usize];
        }
        // Left to right, the listed leaves begin with the key bits in order.
        let mut stack = vec![vtree.root()];
        let mut seen = 0usize;
        while let Some(t) = stack.pop() {
            match *vtree.node(t) {
                VtreeNode::Internal { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
                VtreeNode::Leaf { var, .. } => {
                    if role[var.0 as usize].is_none() {
                        continue;
                    }
                    if seen == key_bits.len() {
                        break;
                    }
                    if key_bits[seen] != var {
                        return None;
                    }
                    seen += 1;
                }
            }
        }
        if seen != key_bits.len() {
            return None;
        }
        let mut held = vec![0u32; vtree.num_nodes()];
        for t in vtree.bottomup() {
            held[t.idx()] = match *vtree.node(t) {
                VtreeNode::Leaf { var, .. } => u32::from(is_key[var.0 as usize]),
                VtreeNode::Internal { left, right, .. } => held[left.idx()] + held[right.idx()],
            };
        }
        Some(KeyLayout { vtree, key_bits, held, down })
    }

    fn held(&self, t: VtreeIdx) -> u32 {
        self.held[t.idx()]
    }

    /// Calls `set(column, bit, value)` for every key bit of a root key.
    fn decode(&self, key: u128, role: &[Option<(u32, u32)>], mut set: impl FnMut(u32, u32, u32)) {
        let b = self.key_bits.len();
        for (q, v) in self.key_bits.iter().enumerate() {
            let g = ((key >> (b - 1 - q)) & 1) as u32;
            let (j, bit) = role[v.0 as usize].expect("a listed key bit");
            set(j, bit, g ^ u32::from(self.down[v.0 as usize]));
        }
    }

    /// The flipped values a key leaf's label allows, in order.
    fn leaf_values(&self, var: VarId, label: LeafLabel) -> &'static [u128] {
        let down = self.down[var.0 as usize];
        match label {
            LeafLabel::One => &[0, 1],
            // The bit is 1; flipped where descending.
            LeafLabel::Pos => if down { &[0] } else { &[1] },
            _ => if down { &[1] } else { &[0] },
        }
    }
}

/// The models of one node over the key bits below its vtree node, in
/// order, as far as they were asked for.
#[derive(Clone, Debug, Default)]
struct Stream {
    /// Each model's key bits below the node, each flipped where its column
    /// is descending (so the order is ascending), the leftmost the most
    /// significant; and the nodes it left pending, a list in
    /// [`Streams::cells`], or [`NIL`].
    out: Vec<(u128, u32)>,
    /// Whether the node's pairs entered the heap.
    open: bool,
    /// Per pair, its next model's key, the pair, the places in its
    /// children's streams that model is read from, and whether the key is
    /// only a bound (the pair's first, its right child's least key not yet
    /// read): the least first.
    heap: BinaryHeap<Reverse<Next>>,
}

/// A pair's next model in a [`Stream`]'s heap: its key, the pair, its
/// places in the children's streams, and whether the key is only a bound.
type Next = (u128, u32, u32, u32, bool);

/// Every node's stream, produced as far as a reader asked.
struct Streams<'a> {
    tdd: &'a Tdd,
    layout: &'a KeyLayout<'a>,
    /// Whether a stream keeps one entry per key and leaves nothing pending.
    distinct: bool,
    /// Per vtree node, the streams of the nodes read so far.
    streams: Vec<FxHashMap<u32, Stream>>,
    /// The pending lists, sharing tails.
    cells: Vec<Cell>,
    /// Pairs entered into heaps since the caller last polled.
    work: u64,
}

impl<'a> Streams<'a> {
    /// The streams of `tdd`, none produced yet.
    fn new(tdd: &'a Tdd, layout: &'a KeyLayout<'a>, distinct: bool) -> Streams<'a> {
        let vtree = layout.vtree;
        Streams { tdd, layout, distinct, streams: vec![FxHashMap::default(); vtree.num_nodes()], cells: Vec::new(), work: 0 }
    }

    /// Node `n`'s least key at vtree node `t`: its stream's first entry.
    fn least(&mut self, t: VtreeIdx, n: u32) -> Result<u128, OperationError> {
        Ok(self.entry(t, n, 0)?.map_or(u128::MAX, |(key, _)| key))
    }

    fn of(&self, t: VtreeIdx, n: u32) -> &Stream {
        &self.streams[t.idx()][&n]
    }

    /// Entry `pos` of node `n`'s stream at vtree node `t`, if it has one.
    fn entry(&mut self, t: VtreeIdx, n: u32, pos: u32) -> Result<Option<(u128, u32)>, OperationError> {
        self.ensure(t, n, pos as usize + 1)?;
        Ok(self.streams[t.idx()][&n].out.get(pos as usize).copied())
    }

    /// Produces node `n`'s stream at vtree node `t` up to `count` entries,
    /// or to its end.
    fn ensure(&mut self, t: VtreeIdx, n: u32, count: usize) -> Result<(), OperationError> {
        if self.streams[t.idx()].get(&n).is_some_and(|s| s.out.len() >= count) {
            return Ok(());
        }
        let mut stream = self.streams[t.idx()].remove(&n).unwrap_or_default();
        let result = self.produce(t, n, &mut stream, count);
        self.streams[t.idx()].insert(n, stream);
        result
    }

    /// A pending cell `(t, n)` before the list `next`.
    fn pend(&mut self, t: VtreeIdx, n: u32, next: u32) -> u32 {
        if self.distinct {
            return NIL;
        }
        self.cells.push(Cell { t, n, next });
        self.cells.len() as u32 - 1
    }

    /// The list `a` then the list `b`, `a`'s cells copied.
    fn join(&mut self, a: u32, b: u32) -> u32 {
        let mut list = b;
        let mut c = a;
        while c != NIL {
            let Cell { t, n, next } = self.cells[c as usize];
            list = self.pend(t, n, list);
            c = next;
        }
        list
    }

    fn produce(&mut self, t: VtreeIdx, n: u32, stream: &mut Stream, count: usize) -> Result<(), OperationError> {
        let vtree = self.layout.vtree;
        let (left, right) = match *vtree.node(t) {
            VtreeNode::Leaf { var, .. } => {
                if !stream.open {
                    stream.open = true;
                    let label = LeafLabel::from_idx(n as usize);
                    stream.out.extend(self.layout.leaf_values(var, label).iter().map(|&g| (g, NIL)));
                }
                return Ok(());
            }
            VtreeNode::Internal { left, right, .. } => (left, right),
        };
        let (hl, hr) = (self.layout.held(left), self.layout.held(right));
        let pairs = self.tdd.level(t).pairs_of_idx(n as usize);
        if !stream.open {
            stream.open = true;
            self.work += pairs.len() as u64;
            // Each pair enters at its least key — where both sides hold key
            // bits, at the bound its left child's least key gives, the right
            // child's read only once the pair reaches the top.
            for (p, pair) in pairs.iter().enumerate() {
                let (l, r) = (pair.left.raw(), pair.right.raw());
                let (key, bound) = match (hl, hr) {
                    (0, _) => (self.least(right, r)?, false),
                    (_, 0) => (self.least(left, l)?, false),
                    _ => (self.least(left, l)? << hr, true),
                };
                stream.heap.push(Reverse((key, p as u32, 0, 0, bound)));
            }
        }
        while stream.out.len() < count {
            let Some(Reverse((key, p, i, j, bound))) = stream.heap.pop() else { break };
            let pair = pairs[p as usize];
            let (l, r) = (pair.left.raw(), pair.right.raw());
            if bound {
                // The pair's first key, which every key it bounds sorts at
                // or after: back in, unless it is the bound itself.
                let exact = key | self.least(right, r)?;
                if exact != key {
                    stream.heap.push(Reverse((exact, p, i, j, false)));
                    continue;
                }
            }
            // The entry, and the pair's next one.
            let (rest, next) = match (hl, hr) {
                (0, _) => {
                    let (_, below) = self.entry(right, r, i)?.expect("an entry the heap holds");
                    let rest = self.pend(left, l, below);
                    (rest, self.entry(right, r, i + 1)?.map(|(k, _)| (k, i + 1, 0)))
                }
                (_, 0) => {
                    let (_, below) = self.entry(left, l, i)?.expect("an entry the heap holds");
                    let rest = self.pend(right, r, below);
                    (rest, self.entry(left, l, i + 1)?.map(|(k, _)| (k, i + 1, 0)))
                }
                _ => {
                    let (lk, lrest) = self.entry(left, l, i)?.expect("an entry the heap holds");
                    let (_, rrest) = self.entry(right, r, j)?.expect("an entry the heap holds");
                    let rest = self.join(lrest, rrest);
                    let next = match self.entry(right, r, j + 1)? {
                        Some((rk, _)) => Some(((lk << hr) | rk, i, j + 1)),
                        None => match self.entry(left, l, i + 1)? {
                            Some((lk, _)) => Some(((lk << hr) | self.least(right, r)?, i + 1, 0)),
                            None => None,
                        },
                    };
                    (rest, next)
                }
            };
            if !(self.distinct && stream.out.last().is_some_and(|&(last, _)| last == key)) {
                stream.out.push((key, rest));
            }
            if let Some((k, i, j)) = next {
                stream.heap.push(Reverse((k, p, i, j, false)));
            }
        }
        Ok(())
    }
}

/// One pending node of a model under construction: node `n` of vtree node
/// `t`, then the list from `next`.
#[derive(Clone, Copy)]
struct Cell {
    t: VtreeIdx,
    n: u32,
    next: u32,
}

/// The models below the spine, in storage order, each completing the key
/// bits already set.
struct Walk<'a> {
    tdd: &'a Tdd,
    role: &'a [Option<(u32, u32)>],
    descending: &'a [bool],
    /// The model's codes so far.
    cur: Vec<u32>,
    out: Vec<Vec<u32>>,
    limit: u64,
    /// The pending lists, as cells sharing tails; a branch's cells are
    /// dropped when it returns.
    cells: Vec<Cell>,
    gate: crate::limits::PollGate<'a>,
}

impl Walk<'_> {
    /// Every model of the nodes pending from cell `list`, with the bits set
    /// so far; `true` once `limit` models are written.
    fn go(&mut self, mut list: u32) -> Result<bool, OperationError> {
        let vtree = Arc::clone(self.tdd.vtree());
        loop {
            if list == NIL {
                for (column, &code) in self.out.iter_mut().zip(&self.cur) {
                    column.push(code);
                }
                self.gate.poll(1)?;
                return Ok(self.out[0].len() as u64 >= self.limit);
            }
            let Cell { t, n, next } = self.cells[list as usize];
            match *vtree.node(t) {
                VtreeNode::Leaf { var, .. } => {
                    let label = LeafLabel::from_idx(n as usize);
                    match (self.role[var.0 as usize], label) {
                        (Some((j, bit)), LeafLabel::One) => {
                            let order: [u32; 2] = if self.descending[j as usize] { [1, 0] } else { [0, 1] };
                            for b in order {
                                self.set(j, bit, b);
                                let mark = self.cells.len();
                                let done = self.go(next)?;
                                self.cells.truncate(mark);
                                if done {
                                    return Ok(true);
                                }
                            }
                            return Ok(false);
                        }
                        (Some((j, bit)), LeafLabel::Pos) => self.set(j, bit, 1),
                        (Some((j, bit)), _) => self.set(j, bit, 0),
                        (None, LeafLabel::One) => {}
                        (None, _) => return Err(OperationError::UnlistedLiteral(var)),
                    }
                    list = next;
                }
                VtreeNode::Internal { left, right, .. } => {
                    let pairs: Vec<(u32, u32)> = self
                        .tdd
                        .level(t)
                        .pairs_of_idx(n as usize)
                        .iter()
                        .map(|p| (p.left.raw(), p.right.raw()))
                        .collect();
                    if let [(l, r)] = pairs[..] {
                        list = self.push(left, l, right, r, next);
                        continue;
                    }
                    for (l, r) in pairs {
                        let mark = self.cells.len();
                        let head = self.push(left, l, right, r, next);
                        let done = self.go(head)?;
                        self.cells.truncate(mark);
                        if done {
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
            }
        }
    }

    /// Sets bit `bit` of column `j` to `b`.
    fn set(&mut self, j: u32, bit: u32, b: u32) {
        let c = &mut self.cur[j as usize];
        *c = (*c & !(1u32 << bit)) | (b << bit);
    }

    /// A pending list of the left node, the right node, then `next`.
    fn push(&mut self, left: VtreeIdx, l: u32, right: VtreeIdx, r: u32, next: u32) -> u32 {
        self.cells.push(Cell { t: right, n: r, next });
        let r_at = self.cells.len() as u32 - 1;
        self.cells.push(Cell { t: left, n: l, next: r_at });
        self.cells.len() as u32 - 1
    }
}

#[cfg(test)]
#[path = "tests/ordered.rs"]
mod tests;
