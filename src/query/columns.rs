//! A diagram's models written out as columns of codes.
//!
//! [`Tdd::model_columns`] reads a diagram's models as rows: each column is a
//! list of variables read as one unsigned code, its first variable the most
//! significant bit, and each model is one row. [`ModelColumns`] holds what
//! the writing needs, and [`ModelColumns::write`] fills caller-owned buffers
//! with any range of the rows, so a table too large to hold is written in
//! batches.
//!
//! The rows are written column by column, never one at a time. Every node's
//! models have a fixed order: its pairs in storage order, and within a pair
//! the left child's models outermost. A node's count, over the listed
//! variables, places each of its pairs in a range of that order. With `S`
//! rows below a pair's right child, each row of the left child is repeated
//! `S` times, and the right child's rows recur as `P` equal tiles, `P` the
//! rows below the left child. The left child is written with a repeat that
//! much wider; the right child is written once, into the first tile, and
//! copied into the others by doubling copies, masked to the bits that
//! child's variables own. A node with one row is a constant over its range.
//! A vtree node whose variables are exactly one column's bits, plus
//! unlisted variables, writes its nodes' codes as whole words, each stored
//! once: the one-row side of a pair joins a prefix or-ed into the words
//! below it, so a chain of constants costs nothing per row, and a node that
//! two pairs reference keeps its code list, within a budget, to be copied
//! wherever it recurs.
//!
//! The fixed order is the rows' ascending order — column 0's code, ties by
//! column 1's, and so on — exactly when two things hold, and
//! [`ModelColumns::ascending`] checks both. The layout: the vtree's leaves,
//! left to right, read the listed variables as column 0's bits from the most
//! significant down, then column 1's, and so on (unlisted variables may sit
//! anywhere between them). And at every node, each pair's last row precedes
//! the next pair's first row. Pairs are stored in no particular order, so
//! the second can fail even under the layout; [`ModelColumns::sort_pairs`]
//! puts each node's pairs in the order of their first rows, which satisfies
//! it wherever no two pairs of one node interleave. Where both hold, rows
//! `0..k` are the `k` least rows, written without sorting anything.
//!
//! Distinct rows follow from determinism. The pairs of a node denote
//! disjoint functions, so no two pairs share a model, and a node whose
//! function ignores the unlisted variables has its distinct models on the
//! listed ones. [`Tdd::model_columns`] checks that no reachable pair
//! references a literal of an unlisted variable, which keeps every node
//! independent of them.

use std::ops::Range;
use std::sync::Arc;

use crate::diagram::{LeafLabel, Tdd};
use crate::limits::OperationError;
use crate::vtree::{VarId, VtreeIdx, VtreeNode};
use crate::Engine;

/// The widest column a code holds.
pub const MAX_COLUMN_BITS: usize = 32;

/// A node's code lists stay in memory while they hold at most this many codes
/// in all; past it, a vtree node's nodes are written bit by bit.
const MEMO_BUDGET: usize = 1 << 28;

/// Pairs scanned in order before a binary search finds the first pair a
/// range reaches (one under test, so the tests' small diagrams search too).
const LINEAR_PAIRS: usize = if cfg!(test) { 1 } else { 16 };

impl Tdd {
    /// The models of this diagram as rows of `columns.len()` codes, ready to
    /// be written with [`ModelColumns::write`].
    ///
    /// Column `j` reads the variables `columns[j]` as one unsigned code, its
    /// first variable the most significant bit: a model sets bit
    /// `columns[j].len() - 1 - i` of the code exactly when it sets
    /// `columns[j][i]`. Each model of the function is one row, and every row
    /// is distinct. The diagram must not depend on any variable no column
    /// lists; every such variable is free, and its values do not multiply the
    /// rows. A listed variable the function leaves free on some path yields
    /// both of its values there, so a column whose bits are all free there
    /// holds every code of its width.
    ///
    /// The diagram is borrowed and unchanged; the result owns copies of the
    /// reachable nodes' pairs and counts, so it outlives the diagram. Building
    /// it is linear in the reachable diagram. The rows are numbered in a fixed
    /// order that does not depend on how they are later written.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // Variables 1 and 2 are a two-bit code, variable 3 is a one-bit code.
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 3])?;
    /// let mut table = f.model_columns(&[&[VarId(1), VarId(2)], &[VarId(3)]])?;
    /// let columns = table.to_columns();
    /// let mut rows: Vec<(u32, u32)> = columns[0].iter().copied().zip(columns[1].iter().copied()).collect();
    /// rows.sort();
    /// assert_eq!(rows, [(0, 1), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a listed variable the vtree
    /// lacks, [`OperationError::DuplicateVariable`] for one listed twice,
    /// [`OperationError::ColumnTooWide`] for a column of more than
    /// [`MAX_COLUMN_BITS`] variables, [`OperationError::UnlistedLiteral`] when a
    /// reachable pair references a literal of a variable no column lists
    /// (quantify it out with [`exists_vars`](Engine::exists_vars), or
    /// [`minimize`](Self::minimize) a diagram that ignores it but still
    /// references it), [`OperationError::IndexOverflow`] when the rows do not
    /// fit a `u64`, [`OperationError::MarginalLevel`] for a diagram with a
    /// marginal level, [`OperationError::OverBudget`] for a refused allocation
    /// and [`OperationError::Stopped`] when an armed stop fires.
    pub fn model_columns(&self, columns: &[&[VarId]]) -> Result<ModelColumns, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.model_columns(self, columns))
    }
}

impl Engine {
    /// Run [`Tdd::model_columns`] using this batch's resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::model_columns`]. The copies of the reachable pairs are
    /// charged to the byte budget; stops are checked once per level.
    pub fn model_columns(&self, tdd: &Tdd, columns: &[&[VarId]]) -> Result<ModelColumns, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let mut gate = lim.gate();
        let vtree = Arc::clone(tdd.vtree());
        let num_nodes = vtree.num_nodes();

        // Each listed variable's column and bit.
        let mut role: Vec<Option<(u32, u32)>> = vec![None; vtree.num_vars() as usize + 1];
        let mut widths = Vec::with_capacity(columns.len());
        for (j, vars) in columns.iter().enumerate() {
            if vars.len() > MAX_COLUMN_BITS {
                return Err(OperationError::ColumnTooWide { column: j, bits: vars.len() });
            }
            widths.push(vars.len() as u32);
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

        // Per vtree node, the bits of each column its variables own.
        let mut level_touch: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_nodes];
        for t in vtree.bottomup() {
            level_touch[t.idx()] = match vtree.node(t) {
                VtreeNode::Leaf { var, .. } => match role[var.0 as usize] {
                    Some((j, bit)) => vec![(j, 1u32 << bit)],
                    None => Vec::new(),
                },
                VtreeNode::Internal { left, right, .. } => {
                    merge_masks(&level_touch[left.idx()], &level_touch[right.idx()])
                }
            };
        }
        let full = |j: u32| match widths[j as usize] {
            32 => u32::MAX,
            w => (1u32 << w) - 1,
        };
        let mut touched = Vec::new();
        let mut levels = Vec::with_capacity(num_nodes);
        for list in &level_touch {
            let start = touched.len() as u32;
            touched.extend(list.iter().map(|&(j, mask)| Touch { column: j, mask, whole: mask == full(j) }));
            let closed = match list.as_slice() {
                [(j, mask)] if *mask == full(*j) => *j,
                _ => NONE,
            };
            levels.push(Level { touch: start, touches: list.len() as u32, closed });
        }
        drop(level_touch);

        // Whether the leaves, left to right, read the listed variables as
        // column 0's bits from the most significant down, then column 1's,
        // and so on: the layout under which the fixed order can be the rows'
        // ascending order.
        let mut in_order = Vec::new();
        let mut stack = vec![vtree.root()];
        while let Some(t) = stack.pop() {
            match *vtree.node(t) {
                VtreeNode::Leaf { var, .. } => in_order.extend(role[var.0 as usize]),
                VtreeNode::Internal { left, right, .. } => stack.extend([right, left]),
            }
        }
        let lexicographic = in_order
            .iter()
            .copied()
            .eq(widths.iter().enumerate().flat_map(|(j, &w)| (0..w).rev().map(move |bit| (j as u32, bit))));
        drop(in_order);

        let mut table = ModelColumns {
            widths,
            rows: 0,
            root: NONE,
            nodes: Vec::new(),
            pairs: Vec::new(),
            bases: std::collections::HashMap::new(),
            cursor: std::collections::HashMap::new(),
            konst: Vec::new(),
            levels,
            touched,
            memo_of: Vec::new(),
            memo: Vec::new(),
            memo_codes: 0,
            scratch: Vec::new(),
            lexicographic,
        };
        if tdd.is_zero() {
            return Ok(table);
        }

        // Reachable nodes, marked from the output down.
        let mut gid: Vec<Vec<u32>> = (0..num_nodes)
            .map(|t| {
                let t = VtreeIdx(t as u32);
                let width = match vtree.node(t).is_leaf() {
                    true => crate::diagram::LEAF_WIDTH,
                    false => tdd.level(t).slot_count(),
                };
                let mut map = Vec::new();
                lim.reserve_exact(&mut map, width)?;
                map.resize(width, NONE);
                Ok(map)
            })
            .collect::<Result<_, OperationError>>()?;
        let out = tdd.output();
        gid[out.vtree.idx()][out.local.idx()] = 0;
        for t in vtree.bottomup().rev() {
            gate.poll(1)?;
            let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else { continue };
            let level = tdd.level(t);
            let (here, below) = split_three(&mut gid, t.idx(), left.idx(), right.idx());
            for (i, &mark) in here.iter().enumerate() {
                if mark == NONE {
                    continue;
                }
                for p in level.pairs_iter_of_idx(i) {
                    below.0[p.left.raw() as usize] = 0;
                    below.1[p.right.raw() as usize] = 0;
                }
            }
        }

        // Number them bottom-up, with each node's count and pairs.
        let mut total_pairs = 0usize;
        for t in vtree.bottomup() {
            if !vtree.node(t).is_leaf() {
                let level = tdd.level(t);
                for (i, &mark) in gid[t.idx()].iter().enumerate() {
                    if mark != NONE {
                        total_pairs += level.pair_count_at(i);
                    }
                }
            }
        }
        lim.reserve_exact(&mut table.pairs, total_pairs)?;
        for t in vtree.bottomup() {
            gate.poll(1)?;
            match *vtree.node(t) {
                VtreeNode::Leaf { var, .. } => {
                    for label in [LeafLabel::One, LeafLabel::Pos, LeafLabel::Neg] {
                        if gid[t.idx()][label as usize] == NONE {
                            continue;
                        }
                        let id = table.nodes.len() as u32;
                        gid[t.idx()][label as usize] = id;
                        let node = match (role[var.0 as usize], label) {
                            (Some((j, bit)), LeafLabel::One) => {
                                Node { count: 2, level: t.0, kind: Kind::Free, a: j, b: bit, refs: 0 }
                            }
                            (Some((j, bit)), LeafLabel::Pos) => {
                                let at = table.konst.len() as u32;
                                table.konst.push((j, 1u32 << bit));
                                Node { count: 1, level: t.0, kind: Kind::Const, a: at, b: 1, refs: 0 }
                            }
                            (Some(_), _) | (None, LeafLabel::One) => {
                                Node { count: 1, level: t.0, kind: Kind::Const, a: 0, b: 0, refs: 0 }
                            }
                            (None, _) => return Err(OperationError::UnlistedLiteral(var)),
                        };
                        table.nodes.push(node);
                    }
                }
                VtreeNode::Internal { left, right, .. } => {
                    let level = tdd.level(t);
                    let (here, below) = split_three(&mut gid, t.idx(), left.idx(), right.idx());
                    for (i, mark) in here.iter_mut().enumerate() {
                        if *mark == NONE {
                            continue;
                        }
                        let id = table.nodes.len() as u32;
                        *mark = id;
                        let first = table.pairs.len() as u32;
                        let mut count: u64 = 0;
                        for p in level.pairs_iter_of_idx(i) {
                            let (l, r) = (below.0[p.left.raw() as usize], below.1[p.right.raw() as usize]);
                            let product = table.nodes[l as usize]
                                .count
                                .checked_mul(table.nodes[r as usize].count)
                                .ok_or(OperationError::IndexOverflow)?;
                            table.pairs.push([l, r]);
                            for c in [l, r] {
                                let refs = &mut table.nodes[c as usize].refs;
                                *refs = (*refs + 1).min(2);
                            }
                            count = count.checked_add(product).ok_or(OperationError::IndexOverflow)?;
                        }
                        let n = table.pairs.len() as u32 - first;
                        let node = match count {
                            1 => {
                                // One pair, each side one row: the constants of both.
                                let [l, r] = table.pairs[first as usize];
                                table.pairs.truncate(first as usize);
                                let (a, b) = table.merge_konst(l, r);
                                Node { count: 1, level: t.0, kind: Kind::Const, a, b, refs: 0 }
                            }
                            _ => Node { count, level: t.0, kind: Kind::Pairs, a: first, b: n, refs: 0 },
                        };
                        table.nodes.push(node);
                    }
                }
            }
        }
        table.root = gid[out.vtree.idx()][out.local.idx()];
        table.rows = table.nodes[table.root as usize].count;
        table.memo_of = vec![NONE; table.nodes.len()];
        gate.finish()?;
        Ok(table)
    }
}

/// The marker of an unreached node, an unclosed level and an absent code list.
const NONE: u32 = u32::MAX;

/// What a node writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// One row: the column bits `konst[a .. a + b]`.
    Const,
    /// A listed variable left free: rows `0` and `1` of bit `b` of column `a`.
    Free,
    /// Pairs `pairs[a .. a + b]`.
    Pairs,
}

/// One reachable node.
#[derive(Clone, Copy, Debug)]
struct Node {
    /// Its models, read on the listed variables.
    count: u64,
    /// Its vtree node.
    level: u32,
    kind: Kind,
    a: u32,
    b: u32,
    /// The reachable pairs that reference it, up to two: a node referenced
    /// once is written where its one parent puts it, and keeps no code list.
    refs: u32,
}

/// The bits of one column that one vtree node's variables own.
#[derive(Clone, Copy, Debug)]
struct Touch {
    column: u32,
    mask: u32,
    /// Every bit of the column: no variable outside the vtree node writes it.
    whole: bool,
}

/// One vtree node's columns.
#[derive(Clone, Copy, Debug)]
struct Level {
    /// Its columns, `touched[touch .. touch + touches]`.
    touch: u32,
    touches: u32,
    /// The one column whose bits are exactly its listed variables, or [`NONE`].
    closed: u32,
}

/// A diagram's models, as rows of codes, prepared by [`Tdd::model_columns`].
///
/// It owns what writing the rows needs and borrows nothing. Writing keeps a
/// cache of code lists between calls, so it takes `&mut self`.
#[derive(Clone, Debug)]
pub struct ModelColumns {
    widths: Vec<u32>,
    rows: u64,
    root: u32,
    nodes: Vec<Node>,
    pairs: Vec<[u32; 2]>,
    /// Per node with more than [`LINEAR_PAIRS`] pairs that a range began
    /// inside, each pair's first row in the node's order: what finds the
    /// first pair a range reaches. Written on first need.
    bases: std::collections::HashMap<u32, Box<[u64]>>,
    /// Per node with more than [`LINEAR_PAIRS`] pairs, the last pair a
    /// write reached and its first row: where the next write resumes.
    cursor: std::collections::HashMap<u32, (usize, u64)>,
    /// Constant nodes' column bits, `(column, bits)`.
    konst: Vec<(u32, u32)>,
    levels: Vec<Level>,
    touched: Vec<Touch>,
    /// Per node, its code list in `memo`, or [`NONE`].
    memo_of: Vec<u32>,
    memo: Vec<Box<[u32]>>,
    memo_codes: usize,
    /// Buffers for code lists written once and dropped, reused.
    scratch: Vec<Vec<u32>>,
    /// The leaves read the listed bits column by column, most significant
    /// first.
    lexicographic: bool,
}

impl ModelColumns {
    /// The number of rows: the diagram's models, counted on the listed
    /// variables.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The number of columns.
    pub fn width(&self) -> usize {
        self.widths.len()
    }

    /// The bits of each column, as listed.
    pub fn column_bits(&self) -> impl Iterator<Item = usize> + '_ {
        self.widths.iter().map(|&w| w as usize)
    }

    /// Write rows `rows` into `out`: row `k` of the table is entry
    /// `k - rows.start` of each column's buffer.
    ///
    /// The rows of one table are numbered once and for all, so the ranges of
    /// several calls tile the table exactly when the ranges tile
    /// `0..rows()`. The buffers' previous contents are overwritten. Each
    /// call is linear in the cells it writes, plus the nodes on the two
    /// boundary paths of its range, plus a node's code list the first time it
    /// is needed.
    ///
    /// # Panics
    ///
    /// Panics if `rows` reaches past [`rows`](Self::rows), if `out` does not
    /// hold one buffer per column, or if a buffer's length is not
    /// `rows.end - rows.start`.
    pub fn write(&mut self, rows: Range<u64>, out: &mut [&mut [u32]]) {
        for column in out.iter_mut() {
            column.fill(0);
        }
        self.write_zeroed(rows, out);
    }

    /// [`write`](Self::write) into buffers that hold only zeros, such as
    /// freshly allocated ones: the same rows, without clearing them first.
    ///
    /// # Panics
    ///
    /// As [`write`](Self::write). A buffer that holds anything but zeros is
    /// not detected; its rows come out or-ed with what it held.
    pub fn write_zeroed(&mut self, rows: Range<u64>, out: &mut [&mut [u32]]) {
        assert!(rows.start <= rows.end && rows.end <= self.rows, "rows {rows:?} reach past {}", self.rows);
        assert_eq!(out.len(), self.widths.len(), "one buffer per column");
        let len = (rows.end - rows.start) as usize;
        for column in out.iter() {
            assert_eq!(column.len(), len, "a buffer holds the rows written");
        }
        if len > 0 {
            self.fill(self.root, 0, 1, rows.start, rows.end, rows.start, out);
        }
    }

    /// Every row, one vector per column.
    pub fn to_columns(&mut self) -> Vec<Vec<u32>> {
        let n = usize::try_from(self.rows).expect("the rows fit in memory");
        let mut columns: Vec<Vec<u32>> = self.widths.iter().map(|_| vec![0u32; n]).collect();
        let mut out: Vec<&mut [u32]> = columns.iter_mut().map(|c| c.as_mut_slice()).collect();
        self.write_zeroed(0..self.rows, &mut out);
        columns
    }

    /// Whether the rows, numbered as [`write`](Self::write) numbers them,
    /// ascend: by column 0's code, ties by column 1's, and so on. Then rows
    /// `0..k` are the `k` least rows, so a range from row 0 is the first `k`
    /// rows of the table sorted on its codes, column by column.
    ///
    /// The check is exact when the vtree's leaves, left to right, read the
    /// listed variables as column 0's bits from the most significant down,
    /// then column 1's, and so on, with unlisted variables anywhere between
    /// them; under any other layout it is `false` for more than one row.
    /// Under that layout the rows ascend exactly when, at every node, each
    /// pair's last row precedes the next pair's first row: a node's rows are
    /// its pairs' in storage order, and a pair's are its left child's rows,
    /// each followed by every row of its right child. A free listed variable
    /// writes `0` before `1`. [`sort_pairs`](Self::sort_pairs) puts each
    /// node's pairs in the order of their first rows, which makes the rows
    /// ascend wherever no two pairs of a node interleave.
    ///
    /// Linear in the nodes and pairs, times the columns.
    pub fn ascending(&self) -> bool {
        if self.rows <= 1 {
            return true;
        }
        if !self.lexicographic {
            return false;
        }
        let w = self.widths.len();
        let (first, last) = self.extremes();
        self.nodes.iter().filter(|node| node.kind == Kind::Pairs).all(|node| {
            let pairs = &self.pairs[node.a as usize..(node.a + node.b) as usize];
            pairs.windows(2).all(|p| cmp_pair_rows(w, &last, p[0], &first, p[1]).is_lt())
        })
    }

    /// Puts each node's pairs in the order of their first rows (column 0's
    /// code first), so that the rows ascend wherever no two pairs of one node
    /// interleave; [`ascending`](Self::ascending) says whether they do. The
    /// rows are the same set, numbered anew; code lists kept from earlier
    /// writes are dropped.
    pub fn sort_pairs(&mut self) {
        let w = self.widths.len();
        let mut first = vec![0u32; self.nodes.len() * w];
        let mut last = vec![0u32; self.nodes.len() * w];
        for n in 0..self.nodes.len() {
            let node = self.nodes[n];
            if node.kind == Kind::Pairs {
                self.pairs[node.a as usize..(node.a + node.b) as usize]
                    .sort_by(|&p, &q| cmp_pair_rows(w, &first, p, &first, q));
            }
            self.extreme(n, &mut first, &mut last);
        }
        self.bases.clear();
        self.cursor.clear();
        self.memo_of.fill(NONE);
        self.memo.clear();
        self.memo_codes = 0;
    }

    /// Each node's first and last row, `w` codes per node, the bits outside
    /// its vtree node zero.
    fn extremes(&self) -> (Vec<u32>, Vec<u32>) {
        let w = self.widths.len();
        let mut first = vec![0u32; self.nodes.len() * w];
        let mut last = vec![0u32; self.nodes.len() * w];
        for n in 0..self.nodes.len() {
            self.extreme(n, &mut first, &mut last);
        }
        (first, last)
    }

    /// Node `n`'s first and last row, from its children's (numbered before it).
    fn extreme(&self, n: usize, first: &mut [u32], last: &mut [u32]) {
        let w = self.widths.len();
        let node = self.nodes[n];
        let at = n * w;
        match node.kind {
            Kind::Const => {
                for &(j, bits) in &self.konst[node.a as usize..(node.a + node.b) as usize] {
                    first[at + j as usize] |= bits;
                    last[at + j as usize] |= bits;
                }
            }
            Kind::Free => last[at + node.a as usize] |= 1u32 << node.b,
            Kind::Pairs => {
                let [l0, r0] = self.pairs[node.a as usize];
                let [l1, r1] = self.pairs[(node.a + node.b - 1) as usize];
                for j in 0..w {
                    first[at + j] = first[l0 as usize * w + j] | first[r0 as usize * w + j];
                    last[at + j] = last[l1 as usize * w + j] | last[r1 as usize * w + j];
                }
            }
        }
    }

    /// The constants of two one-row nodes, merged by column, as a range of
    /// `konst`.
    fn merge_konst(&mut self, l: u32, r: u32) -> (u32, u32) {
        let range = |n: &Node| (n.a as usize, (n.a + n.b) as usize);
        let (ls, le) = range(&self.nodes[l as usize]);
        let (rs, re) = range(&self.nodes[r as usize]);
        if le - ls == 0 {
            return (rs as u32, (re - rs) as u32);
        }
        if re - rs == 0 {
            return (ls as u32, (le - ls) as u32);
        }
        let merged = merge_masks(&self.konst[ls..le], &self.konst[rs..re]);
        let at = self.konst.len() as u32;
        self.konst.extend_from_slice(&merged);
        (at, merged.len() as u32)
    }

    /// Writes node `n`'s rows, each repeated `rep` times from row `start`,
    /// where they meet rows `lo..hi`; entry `i` of each buffer of `out` is
    /// row `base + i`.
    #[allow(clippy::too_many_arguments)]
    fn fill(&mut self, n: u32, start: u64, rep: u64, lo: u64, hi: u64, base: u64, out: &mut [&mut [u32]]) {
        let node = self.nodes[n as usize];
        let end = start + node.count * rep;
        let (a, b) = (start.max(lo), end.min(hi));
        if a >= b {
            return;
        }
        match node.kind {
            Kind::Const => {
                for &(j, bits) in &self.konst[node.a as usize..(node.a + node.b) as usize] {
                    or_range(&mut out[j as usize][(a - base) as usize..(b - base) as usize], bits);
                }
            }
            Kind::Free => {
                let (s, e) = ((start + rep).max(a), b);
                if s < e {
                    or_range(&mut out[node.a as usize][(s - base) as usize..(e - base) as usize], 1u32 << node.b);
                }
            }
            Kind::Pairs => {
                let closed = self.levels[node.level as usize].closed;
                if closed != NONE {
                    let whole = a == start && b == end;
                    let column = &mut out[closed as usize];
                    if whole && rep == 1 {
                        // Its codes are this range of the column, word for word.
                        self.emit(n, closed, 0, &mut column[(a - base) as usize..(b - base) as usize]);
                        return;
                    }
                    // Its code list: kept when shared, else written for this
                    // call when every row of the range is.
                    let list = match self.memo_for(n, closed) {
                        Some(at) => Some(std::mem::take(&mut self.memo[at as usize]).into_vec()),
                        None if whole => {
                            let mut list = self.take_scratch(node.count as usize);
                            self.emit_pairs(node, closed, 0, &mut list);
                            Some(list)
                        }
                        None => None,
                    };
                    if let Some(list) = list {
                        if rep == 1 {
                            column[(a - base) as usize..(b - base) as usize]
                                .copy_from_slice(&list[(a - start) as usize..(b - start) as usize]);
                        } else {
                            let (k0, k1) = ((a - start) / rep, (b - start).div_ceil(rep));
                            for k in k0..k1 {
                                let s = (start + k * rep).max(a);
                                let e = (start + (k + 1) * rep).min(b);
                                column[(s - base) as usize..(e - base) as usize].fill(list[k as usize]);
                            }
                        }
                        match self.memo_of[n as usize] {
                            NONE => self.scratch.push(list),
                            at => self.memo[at as usize] = list.into_boxed_slice(),
                        }
                        return;
                    }
                }
                self.fill_pairs(node, start, rep, a, b, base, out);
            }
        }
    }

    /// Writes node `n`'s codes, each or-ed with `prefix`, into `dst`, which
    /// holds one entry per row of the node: `n`'s vtree node owns column `j`'s
    /// bits and no other column's, so each row is one word of column `j`,
    /// stored. A one-row side of a pair joins the prefix rather than being
    /// or-ed into every row, so each word is written once below a chain of
    /// constants.
    fn emit(&mut self, n: u32, j: u32, prefix: u32, dst: &mut [u32]) {
        let node = self.nodes[n as usize];
        match node.kind {
            Kind::Const => dst[0] = prefix | self.const_bits(node, j),
            Kind::Free => {
                dst[0] = prefix;
                dst[1] = prefix | 1u32 << node.b;
            }
            Kind::Pairs => match self.memo_for(n, j) {
                Some(at) => {
                    let codes = &self.memo[at as usize];
                    match prefix {
                        0 => dst.copy_from_slice(codes),
                        _ => {
                            for (d, &c) in dst.iter_mut().zip(codes.iter()) {
                                *d = c | prefix;
                            }
                        }
                    }
                }
                None => self.emit_pairs(node, j, prefix, dst),
            },
        }
    }

    /// [`emit`](Self::emit) of a node with pairs, pair by pair.
    fn emit_pairs(&mut self, node: Node, j: u32, prefix: u32, dst: &mut [u32]) {
        let mut at = 0usize;
        for p in node.a..node.a + node.b {
            let [l, r] = self.pairs[p as usize];
            let (cl, cr) = (self.nodes[l as usize].count as usize, self.nodes[r as usize].count as usize);
            let seg = &mut dst[at..at + cl * cr];
            at += cl * cr;
            if cr == 1 {
                let bits = self.const_bits(self.nodes[r as usize], j);
                self.emit(l, j, prefix | bits, seg);
            } else if cl == 1 {
                let bits = self.const_bits(self.nodes[l as usize], j);
                self.emit(r, j, prefix | bits, seg);
            } else {
                // The right child's codes in every tile, the left child's
                // code or-ed into its tile.
                self.emit(r, j, prefix, &mut seg[..cr]);
                copy_tiles(seg, cr, cl, u32::MAX, true);
                let mut left = self.take_scratch(cl);
                self.emit(l, j, 0, &mut left);
                for (tile, &c) in seg.chunks_exact_mut(cr).zip(left.iter()) {
                    if c != 0 {
                        for x in tile {
                            *x |= c;
                        }
                    }
                }
                self.scratch.push(left);
            }
        }
    }

    /// Column `j`'s bits of a one-row node.
    fn const_bits(&self, node: Node, j: u32) -> u32 {
        debug_assert_eq!(node.kind, Kind::Const, "a one-row node is a constant");
        self.konst[node.a as usize..(node.a + node.b) as usize]
            .iter()
            .find(|&&(c, _)| c == j)
            .map_or(0, |&(_, bits)| bits)
    }

    /// A buffer of `len` entries from the scratch pool.
    fn take_scratch(&mut self, len: usize) -> Vec<u32> {
        let mut v = self.scratch.pop().unwrap_or_default();
        v.clear();
        v.resize(len, 0);
        v
    }

    /// [`fill`](Self::fill) of a node with pairs, over rows `a..b` of its range.
    #[allow(clippy::too_many_arguments)]
    fn fill_pairs(&mut self, node: Node, start: u64, rep: u64, a: u64, b: u64, base: u64, out: &mut [&mut [u32]]) {
        let (first, n) = (node.a as usize, node.b as usize);
        let (e0, e1) = ((a - start) / rep, (b - start).div_ceil(rep));
        // The first pair whose rows reach `e0`, and its first row.
        let (mut p, mut at) = match n > LINEAR_PAIRS && e0 > 0 {
            true => self.first_pair(node, e0),
            false => (0, 0),
        };
        let mut reached = (p, at);
        while p < n && at < e1 {
            reached = (p, at);
            let [l, r] = self.pairs[first + p];
            let (nl, nr) = (self.nodes[l as usize], self.nodes[r as usize]);
            let (cl, cr) = (nl.count, nr.count);
            let ps = start + at * rep;
            p += 1;
            at += cl * cr;
            if cl * cr == 1 && rep == 1 {
                // One row, at `ps`, below `b` by the loop's bound: both
                // sides' constants, or-ed in.
                if ps >= a {
                    let row = (ps - base) as usize;
                    for side in [nl, nr] {
                        for &(j, bits) in &self.konst[side.a as usize..(side.a + side.b) as usize] {
                            out[j as usize][row] |= bits;
                        }
                    }
                }
                continue;
            }
            let pe = ps + cl * cr * rep;
            if pe <= a {
                continue;
            }
            let (pa, pb) = (ps.max(a), pe.min(b));
            self.fill(l, ps, rep * cr, pa, pb, base, out);
            if cr == 1 {
                self.fill(r, ps, cl * rep, pa, pb, base, out);
                continue;
            }
            // The right child's rows recur as `cl` tiles of `tile` rows.
            let tile = cr * rep;
            let t0 = (pa - ps) / tile;
            let t1 = (pb - ps).div_ceil(tile);
            let f0 = (pa - ps).div_ceil(tile);
            let f1 = (pb - ps) / tile;
            if f0 + 1 >= f1 {
                for t in t0..t1 {
                    self.fill(r, ps + t * tile, rep, pa, pb, base, out);
                }
                continue;
            }
            for t in t0..f0 {
                self.fill(r, ps + t * tile, rep, pa, pb, base, out);
            }
            let first_tile = ps + f0 * tile;
            self.fill(r, first_tile, rep, pa, pb, base, out);
            let level = self.levels[nr.level as usize];
            for touch in &self.touched[level.touch as usize..(level.touch + level.touches) as usize] {
                copy_tiles(
                    &mut out[touch.column as usize][(first_tile - base) as usize..],
                    tile as usize,
                    (f1 - f0) as usize,
                    touch.mask,
                    touch.whole,
                );
            }
            for t in f1..t1 {
                self.fill(r, ps + t * tile, rep, pa, pb, base, out);
            }
        }
        if n > LINEAR_PAIRS {
            self.cursor.insert(node.a, reached);
        }
    }

    /// The first pair of `node` whose rows reach row `e0` of its order, with
    /// that pair's first row: a short scan on from where the last write into
    /// the node stopped, so batches written in order never need the node's
    /// bases, else a binary search over them.
    fn first_pair(&mut self, node: Node, e0: u64) -> (usize, u64) {
        if let Some(&(mut p, mut at)) = self.cursor.get(&node.a) {
            let n = node.b as usize;
            let mut steps = 0;
            while at <= e0 && p < n && steps <= LINEAR_PAIRS {
                let [l, r] = self.pairs[node.a as usize + p];
                let next = at + self.nodes[l as usize].count * self.nodes[r as usize].count;
                if next > e0 {
                    return (p, at);
                }
                (p, at, steps) = (p + 1, next, steps + 1);
            }
        }
        let bases = self.bases_of(node);
        let p = bases.partition_point(|&at| at <= e0) - 1;
        (p, bases[p])
    }

    /// Each pair's first row in `node`'s order, written on first need.
    fn bases_of(&mut self, node: Node) -> &[u64] {
        let (pairs, nodes) = (&self.pairs, &self.nodes);
        self.bases.entry(node.a).or_insert_with(|| {
            let mut at = 0u64;
            pairs[node.a as usize..(node.a + node.b) as usize]
                .iter()
                .map(|&[l, r]| {
                    let first = at;
                    at += nodes[l as usize].count * nodes[r as usize].count;
                    first
                })
                .collect()
        })
    }

    /// The kept code list of node `n`, whose vtree node owns column `j`'s
    /// bits and no other column's, as an index into `memo`: written on first
    /// use when two pairs reference the node, while the lists stay within
    /// the budget; `None` otherwise.
    fn memo_for(&mut self, n: u32, j: u32) -> Option<u32> {
        let at = self.memo_of[n as usize];
        if at != NONE {
            return Some(at);
        }
        let node = self.nodes[n as usize];
        let count = node.count as usize;
        if node.refs < 2 || self.memo_codes + count > MEMO_BUDGET {
            return None;
        }
        let mut codes = vec![0u32; count];
        self.emit_pairs(node, j, 0, &mut codes);
        self.memo_codes += count;
        self.memo.push(codes.into_boxed_slice());
        let at = self.memo.len() as u32 - 1;
        self.memo_of[n as usize] = at;
        Some(at)
    }
}

/// Pair `p`'s row in `a` against pair `q`'s in `b` (rows of `w` codes per
/// node, as [`ModelColumns::extremes`] lays them out), column by column.
fn cmp_pair_rows(w: usize, a: &[u32], p: [u32; 2], b: &[u32], q: [u32; 2]) -> std::cmp::Ordering {
    let (pl, pr, ql, qr) = (p[0] as usize * w, p[1] as usize * w, q[0] as usize * w, q[1] as usize * w);
    (0..w)
        .map(|j| (a[pl + j] | a[pr + j]).cmp(&(b[ql + j] | b[qr + j])))
        .find(|o| o.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// `dst[i] |= bits` over the slice.
#[inline]
fn or_range(dst: &mut [u32], bits: u32) {
    if bits != 0 {
        for x in dst {
            *x |= bits;
        }
    }
}

/// Copies the first `tile` entries of `column` into the next `tiles - 1`
/// tiles, by doubling: the bits `mask` selects, or-ed in, or the whole word
/// when `whole` (no other variable writes the column there).
fn copy_tiles(column: &mut [u32], tile: usize, tiles: usize, mask: u32, whole: bool) {
    let mut done = 1;
    while done < tiles {
        let k = done.min(tiles - done);
        let (src, dst) = column.split_at_mut(done * tile);
        let (src, dst) = (&src[..k * tile], &mut dst[..k * tile]);
        if whole {
            dst.copy_from_slice(src);
        } else {
            for (d, s) in dst.iter_mut().zip(src) {
                *d |= s & mask;
            }
        }
        done += k;
    }
}

/// Two lists of `(column, bits)` sorted by column, merged by or-ing the bits
/// of a column both hold.
fn merge_masks(a: &[(u32, u32)], b: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut k) = (0, 0);
    while i < a.len() || k < b.len() {
        match (a.get(i), b.get(k)) {
            (Some(&(ja, ma)), Some(&(jb, mb))) if ja == jb => {
                out.push((ja, ma | mb));
                i += 1;
                k += 1;
            }
            (Some(&x), Some(&(jb, _))) if x.0 < jb => {
                out.push(x);
                i += 1;
            }
            (_, Some(&y)) => {
                out.push(y);
                k += 1;
            }
            (Some(&x), None) => {
                out.push(x);
                i += 1;
            }
            (None, None) => unreachable!("the loop runs while a list has an entry"),
        }
    }
    out
}

/// Three distinct entries of `maps`, the first shared and the other two
/// mutable.
fn split_three(maps: &mut [Vec<u32>], here: usize, left: usize, right: usize) -> (&mut [u32], (&mut [u32], &mut [u32])) {
    assert!(here != left && here != right && left != right, "three distinct vtree nodes");
    assert!(here < maps.len() && left < maps.len() && right < maps.len(), "indices in range");
    let ptr = maps.as_mut_ptr();
    // Safety: the three indices are distinct and in range, so the borrows
    // do not alias.
    unsafe {
        let h = &mut *ptr.add(here);
        let l = &mut *ptr.add(left);
        let r = &mut *ptr.add(right);
        (h.as_mut_slice(), (l.as_mut_slice(), r.as_mut_slice()))
    }
}

#[cfg(test)]
#[path = "tests/columns.rs"]
mod tests;
