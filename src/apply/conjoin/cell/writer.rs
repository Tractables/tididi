//! Cells and rows written straight into the arenas of the level being
//! built.

use crate::diagram::{EncodedChildRef, reserve_node_room};

use super::*;

/// Emit a product node from the pairs accumulated in
/// `level.pairs[pair_start..]` and record it at `grid_pos`; a cell that
/// pushed none leaves `NO_PRODUCT` there. A node of one pair, which is
/// stored inline, is counted in `inline`.
#[inline(always)]
pub(crate) fn emit_product_node(
    eng: &Engine,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    grid_pos: usize,
    pair_start: usize,
    inline: &mut usize,
) -> Result<(), OperationError> {
    let one = level.pair_tail_len(pair_start) == 1;
    if let Some(node) = finish_node(eng, level, pair_start)? {
        node_idx[grid_pos] = node.0;
        *inline += usize::from(one);
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn raw_pair(lc: u32, rc: u32) -> ChildPair {
    ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc))
}

/// Emit a node holding exactly `pair`, stored inline. The push is
/// budget-charged, and refused past the level's width cap
/// ([`reserve_node_room`]).
#[inline(always)]
pub(in crate::apply::conjoin) fn emit_single_pair(
    eng: &Engine,
    level: &mut TddLevel,
    pair: ChildPair,
) -> Result<(), OperationError> {
    let nodes = level.nodes.stored_mut();
    reserve_node_room(eng.limits(), nodes, 1)?;
    nodes.push(EncodedNode::inline(pair));
    Ok(())
}

/// A cell written straight into the pair arena of the level being built,
/// opened ([`PairSink::open_cell`]) where the arena has room for every pair
/// the cell can write, so each survivor is stored without a capacity test,
/// and closed into the level's next node as [`finish_node`] closes one.
/// Pushing the survivors one by one would not have grown the arena either,
/// so its capacity and the meters charged from it are as they would be.
/// A node of one pair is counted in the sink's count of them.
pub(crate) struct ArenaCell<'a> {
    level: &'a mut TddLevel,
    inline: &'a mut usize,
    out: *mut ChildPair,
    start: usize,
    len: usize,
    /// One past the last slot the cell may write.
    #[cfg(debug_assertions)]
    end: usize,
}

impl<'a> ArenaCell<'a> {
    /// A cell of at most `bound` pairs at the end of `level`'s arena, which
    /// `room`, at most the arena's spare capacity, says it has room for;
    /// `None` where it has not.
    #[inline(always)]
    pub(super) fn open(
        level: &'a mut TddLevel,
        inline: &'a mut usize,
        room: usize,
        bound: usize,
    ) -> Option<Self> {
        if room < bound {
            return None;
        }
        let pairs = level.pairs.stored_mut();
        debug_assert!(
            room <= pairs.capacity() - pairs.len(),
            "a sink's room is within its arena's capacity"
        );
        let (start, out) = (pairs.len(), pairs.as_mut_ptr());
        Some(ArenaCell {
            level,
            inline,
            out,
            start,
            len: start,
            #[cfg(debug_assertions)]
            end: start + bound,
        })
    }

    /// Store a survivor of the cell.
    ///
    /// # Safety
    ///
    /// A cell takes at most the `bound` pairs it was opened for.
    #[inline(always)]
    pub(crate) unsafe fn push(&mut self, lc: u32, rc: u32) {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.len < self.end,
            "a cell wrote past the pairs it was opened for"
        );
        // Safety: `open` found the arena's spare capacity to hold `end -
        // start` pairs past `start`, and the caller writes no more.
        unsafe {
            self.out.add(self.len).write(raw_pair(lc, rc))
        };
        self.len += 1;
    }

    /// Turn the cell's survivors into the level's next node and record it
    /// at `grid_pos`: no node for none, one pair stored inline, more as the
    /// node's range of the arena.
    #[inline(always)]
    pub(crate) fn close(
        self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), OperationError> {
        let ArenaCell {
            level,
            inline,
            start,
            len,
            ..
        } = self;
        // Safety: the slots below `len` hold the old pairs and one written
        // pair a push.
        unsafe { level.pairs.stored_mut().set_len(len) };
        let node = level.nodes.stored().len() as u32;
        match len - start {
            0 => return Ok(()),
            1 => {
                let pair = level.pop_pair().expect("the cell wrote one pair");
                emit_single_pair(eng, level, pair)?;
                *inline += 1;
            }
            n => level
                .try_push_multi_by_range(eng.limits().level_width_cap(), start, n)?,
        }
        node_idx[grid_pos] = node;
        Ok(())
    }
}

/// A run of cells written straight into the level being built, opened
/// ([`PairSink::open_row`]) where the node arena has room for a node a cell:
/// each survivor is stored, and each cell closed into the level's next node,
/// with no capacity test, the lengths held here and stored in the arenas
/// when the row is dropped. A cell is written only where it
/// [fits](Self::fits) the pair arena's room. A row may stay open across
/// several of the row loop's rows, each written where it
/// [has room](Self::has_room). Each cell closes as [`ArenaCell::close`]
/// closes one, so the nodes, and the capacities and meters charged from
/// them, are as one [`ArenaCell`] a cell would leave them. Its nodes of one
/// pair are counted here too, and added to the sink's count of them when
/// the row is dropped.
pub(crate) struct ArenaRow<'a> {
    pairs: &'a mut Vec<ChildPair>,
    nodes: &'a mut Vec<EncodedNode>,
    inline: &'a mut usize,
    /// The row's nodes of one pair.
    ones: usize,
    out: *mut ChildPair,
    node_out: *mut EncodedNode,
    /// The arena's length past the closed cells, the open cell's survivors,
    /// and the node arena's length.
    len: usize,
    cell: usize,
    node_len: usize,
    /// The length the pair arena may reach: the sink's room, within the 31
    /// bits a node holds its range's start in.
    pair_end: usize,
    /// The node arena's capacity, within the level's width cap.
    node_cap: usize,
}

impl<'a> ArenaRow<'a> {
    /// A row of at most `nodes` cells at the end of `level`'s arenas whose
    /// first `pairs` pairs fit, `room` pairs past the pair arena's length
    /// being at most its spare capacity; `None` where either arena is short.
    /// The node arena's room ends at `width_cap` nodes, so a cell past it
    /// goes to a push that refuses it ([`reserve_node_room`]).
    #[inline(always)]
    pub(super) fn open(
        level: &'a mut TddLevel,
        inline: &'a mut usize,
        (room, pairs, nodes): (usize, usize, usize),
        width_cap: usize,
    ) -> Option<Self> {
        let (pair_arena, node_arena) = (level.pairs.stored_mut(), level.nodes.stored_mut());
        let (len, node_len) = (pair_arena.len(), node_arena.len());
        let pair_end = (len + room).min((1 << 31) - 1);
        let node_cap = node_arena.capacity().min(width_cap);
        if len + pairs > pair_end || node_cap.saturating_sub(node_len) < nodes {
            return None;
        }
        debug_assert!(
            room <= pair_arena.capacity() - len,
            "a sink's room is within its arena's capacity"
        );
        Some(ArenaRow {
            out: pair_arena.as_mut_ptr(),
            node_out: node_arena.as_mut_ptr(),
            pairs: pair_arena,
            nodes: node_arena,
            inline,
            ones: 0,
            len,
            cell: 0,
            node_len,
            pair_end,
            node_cap,
        })
    }

    /// Whether `nodes` more cells, the first `pairs` pairs of them, fit the
    /// room past the closed cells.
    #[inline(always)]
    pub(crate) fn has_room(&self, pairs: usize, nodes: usize) -> bool {
        self.len + pairs <= self.pair_end && self.node_len + nodes <= self.node_cap
    }

    /// How many more cells the node arena has room for.
    #[inline(always)]
    pub(crate) fn node_room(&self) -> usize {
        self.node_cap - self.node_len
    }

    /// Whether a cell of at most `pairs` pairs fits the room past the closed
    /// cells.
    #[inline(always)]
    pub(crate) fn fits(&self, pairs: usize) -> bool {
        self.len + pairs <= self.pair_end
    }

    /// Store a survivor of the open cell.
    ///
    /// # Safety
    ///
    /// The open cell [fits](Self::fits) the pairs it is given.
    #[inline(always)]
    pub(crate) unsafe fn push(&mut self, lc: u32, rc: u32) {
        debug_assert!(
            self.len + self.cell < self.pair_end,
            "a row wrote past its room"
        );
        // Safety: the arena's spare capacity holds `pair_end` past its
        // length, and the caller writes no further.
        let pair = raw_pair(lc, rc);
        unsafe { self.out.add(self.len + self.cell).write(pair) };
        self.cell += 1;
    }

    /// Turn the open cell's survivors into the level's next node and record
    /// it at `grid_pos`, as [`ArenaCell::close`] does, and open the next.
    ///
    /// # Safety
    ///
    /// The row closes at most the cells its open or a
    /// [`has_room`](Self::has_room) admitted.
    #[inline(always)]
    pub(crate) unsafe fn close(&mut self, node_idx: &mut [u32], grid_pos: usize) {
        // Safety: as the caller's.
        if let Some(node) = unsafe { self.close_node() } {
            node_idx[grid_pos] = node;
        }
    }

    /// [`close`](Self::close), the node's index returned for the caller to
    /// record, `None` for a cell of no survivor.
    ///
    /// # Safety
    ///
    /// As [`close`](Self::close)'s.
    #[inline(always)]
    pub(crate) unsafe fn close_node(&mut self) -> Option<u32> {
        let node = match self.cell {
            0 => return None,
            1 => {
                self.ones += 1;
                // Safety: the slot at `len` holds the cell's one survivor.
                EncodedNode::inline(unsafe { self.out.add(self.len).read() })
            }
            n => {
                let node = EncodedNode::multi_pair(self.len as u32, n as u32);
                self.len += n;
                node
            }
        };
        self.cell = 0;
        debug_assert!(
            self.node_len < self.node_cap,
            "a row closed more cells than its node arena holds"
        );
        // Safety: the node arena's capacity holds a node for every cell the
        // row was opened or found room for, and the caller closes no more.
        unsafe { self.node_out.add(self.node_len).write(node) };
        self.node_len += 1;
        Some((self.node_len - 1) as u32)
    }

    /// A cell of the one survivor `(lc, rc)`, closed into the level's next
    /// node and recorded at `grid_pos` as [`push`](Self::push) and
    /// [`close`](Self::close) would, without the pair arena.
    ///
    /// # Safety
    ///
    /// No cell is open, and the row closes at most the cells its open or a
    /// [`has_room`](Self::has_room) admitted.
    #[inline(always)]
    pub(crate) unsafe fn single(
        &mut self,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) {
        // Safety: as the caller's.
        node_idx[grid_pos] = unsafe { self.single_node(lc, rc) };
    }

    /// [`single`](Self::single), the node's index returned for the caller
    /// to record.
    ///
    /// # Safety
    ///
    /// As [`single`](Self::single)'s.
    #[inline(always)]
    pub(crate) unsafe fn single_node(&mut self, lc: u32, rc: u32) -> u32 {
        debug_assert_eq!(self.cell, 0, "a cell is open");
        debug_assert!(
            self.node_len < self.node_cap,
            "a row closed more cells than its node arena holds"
        );
        let node = EncodedNode::inline(raw_pair(lc, rc));
        self.ones += 1;
        // Safety: as in `close`.
        unsafe { self.node_out.add(self.node_len).write(node) };
        self.node_len += 1;
        (self.node_len - 1) as u32
    }
}

impl ArenaRow<'_> {
    /// The row's write position, for a loop that holds it in registers and
    /// stores it back with [`set_cursor`](Self::set_cursor); no cell is
    /// open.
    #[inline(always)]
    pub(crate) fn cursor(&self) -> RowCursor {
        debug_assert_eq!(self.cell, 0, "a cell is open");
        RowCursor {
            out: self.out,
            node_out: self.node_out,
            len: self.len,
            node_len: self.node_len,
            pair_end: self.pair_end,
            node_cap: self.node_cap,
        }
    }

    /// Take the write position a loop moved from [`cursor`](Self::cursor),
    /// and count the nodes of one pair it closed: a cursor closes cells of
    /// one pair inline and cells of two into the pair arena, so they are
    /// its nodes less half its pairs, counted here rather than in the loop.
    #[inline(always)]
    pub(crate) fn set_cursor(&mut self, cursor: RowCursor) {
        self.ones += (cursor.node_len - self.node_len) - (cursor.len - self.len) / 2;
        (self.len, self.node_len) = (cursor.len, cursor.node_len);
    }
}

/// An [`ArenaRow`]'s write position, apart from the arenas, for a loop that
/// holds it in registers: it closes cells into nodes as the row closes
/// them, each a whole cell, with no capacity test.
#[derive(Clone, Copy)]
pub(crate) struct RowCursor {
    out: *mut ChildPair,
    node_out: *mut EncodedNode,
    len: usize,
    node_len: usize,
    pair_end: usize,
    node_cap: usize,
}

impl RowCursor {
    /// How many more pairs the room past the closed cells holds.
    #[inline(always)]
    pub(crate) fn pair_room(&self) -> usize {
        self.pair_end - self.len
    }

    /// How many more cells the node arena has room for.
    #[inline(always)]
    pub(crate) fn node_room(&self) -> usize {
        self.node_cap - self.node_len
    }

    /// A cell of the one survivor `(lc, rc)` closed into the level's next
    /// node, as [`ArenaRow::single_node`] closes one; its index returned.
    ///
    /// # Safety
    ///
    /// The node arena has room for the node ([`node_room`](Self::node_room)).
    #[inline(always)]
    pub(crate) unsafe fn single_node(&mut self, lc: u32, rc: u32) -> u32 {
        // Safety: as the caller's.
        unsafe {
            self.emit(EncodedNode::inline(raw_pair(lc, rc)))
        }
    }

    /// A cell of the two survivors `(l0, r0)` and `(l1, r1)`, in that order,
    /// closed into the level's next node as two [`ArenaRow::push`]es and an
    /// [`ArenaRow::close_node`] close one; its index returned.
    ///
    /// # Safety
    ///
    /// The room past the closed cells holds two pairs
    /// ([`pair_room`](Self::pair_room)) and the node arena the node.
    #[inline(always)]
    pub(crate) unsafe fn pair_node(&mut self, l0: u32, r0: u32, l1: u32, r1: u32) -> u32 {
        debug_assert!(self.len + 2 <= self.pair_end, "a row wrote past its room");
        // Safety: the arena's spare capacity holds `pair_end` past its
        // length, and the caller writes no further.
        unsafe {
            self.out.add(self.len).write(raw_pair(l0, r0));
            self.out.add(self.len + 1).write(raw_pair(l1, r1));
        }
        let node = EncodedNode::multi_pair(self.len as u32, 2);
        self.len += 2;
        // Safety: as the caller's.
        unsafe { self.emit(node) }
    }

    /// Store `node` as the level's next node and return its index.
    ///
    /// # Safety
    ///
    /// The node arena has room for it.
    #[inline(always)]
    unsafe fn emit(&mut self, node: EncodedNode) -> u32 {
        debug_assert!(
            self.node_len < self.node_cap,
            "a row closed more cells than its node arena holds"
        );
        // Safety: as the caller's.
        unsafe { self.node_out.add(self.node_len).write(node) };
        self.node_len += 1;
        (self.node_len - 1) as u32
    }
}

impl Drop for ArenaRow<'_> {
    /// Store the lengths of the closed cells' pairs and nodes in the arenas,
    /// an open cell's survivors left past them, and add the row's nodes of
    /// one pair to the sink's count.
    #[inline(always)]
    fn drop(&mut self) {
        *self.inline += self.ones;
        // Safety: the slots below each length hold the arena's old entries
        // and those the closed cells wrote.
        unsafe {
            self.pairs.set_len(self.len);
            self.nodes.set_len(self.node_len);
        }
    }
}
