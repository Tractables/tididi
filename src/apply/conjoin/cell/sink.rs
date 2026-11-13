//! The per-cell product walk (`process_cell`) is one kernel generic over the
//! per-pair action, expressed as a `PairSink`: every action runs the same loop
//! and only the sink differs.

use super::*;

/// Per-pair action of the cell product walk. All hooks are `#[inline(always)]`
/// in impls, so each kernel instantiation monomorphizes to a specialized walk.
pub(crate) trait PairSink {
    /// 1×1 cell fast path: the cell's single surviving pair. The emit impl
    /// writes `node_idx[grid_pos]` and builds the node directly, without a
    /// push-then-pop round trip through the pair arena.
    fn single(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError>;

    /// Start a multi-pair cell; returns the start token `end` consumes
    /// (the emit impl snapshots the length of its [`buf`](Self::buf)).
    fn begin(&mut self) -> usize;

    /// One surviving (lc, rc) pair of a multi-pair cell, pushed through the
    /// buffer's growth policy.
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError>;

    /// The buffer [`pair`](Self::pair) pushes onto, which [`push_kept`] writes
    /// into directly while it has [room](Self::room).
    fn buf(&mut self) -> &mut Vec<ChildPair>;

    /// How many pairs [`push_kept`] may write into [`buf`](Self::buf)
    /// before the next one has to go through [`pair`](Self::pair): the
    /// buffer's spare capacity, unless the sink grows its buffer on a
    /// schedule of its own.
    #[inline(always)]
    fn room(&mut self) -> usize {
        let v = self.buf();
        v.capacity() - v.len()
    }

    /// Finish a multi-pair cell.
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<(), OperationError>;

    /// Whether f row `i`'s cells sit in grid row `i`; `false` for a sink
    /// that builds every row in grid row 0 and records where each cell
    /// sits ([`row`](Self::row)).
    const SLAB: bool = true;

    /// The f row the cells that follow are of, for a sink that records
    /// where each cell it finishes sits.
    #[inline(always)]
    fn row(&mut self, _i: usize) {}

    /// A cell of at most `bound` pairs written straight into the level the
    /// sink builds, where the sink builds one and has [room](Self::room)
    /// for them all; `None` sends the cell through [`begin`](Self::begin),
    /// [`pair`](Self::pair) and [`end`](Self::end).
    #[inline(always)]
    fn open_cell(&mut self, _bound: usize) -> Option<ArenaCell<'_>> {
        None
    }

    /// A row of at most `nodes` cells written straight into the level the
    /// sink builds, where the sink builds one, its node arena has room for
    /// them within `width_cap` nodes ([`Limits::level_width_cap`]) and its
    /// pair arena for their first `pairs` pairs; `None` sends each cell
    /// through [`open_cell`](Self::open_cell) or [`begin`](Self::begin).
    ///
    /// [`Limits::level_width_cap`]: crate::limits::Limits::level_width_cap
    #[inline(always)]
    fn open_row(&mut self, _pairs: usize, _nodes: usize, _width_cap: usize) -> Option<ArenaRow<'_>> {
        None
    }

    /// Row `i` of a row loop, entered through the sink where the arenas a
    /// row opens ([`open_row`](Self::open_row)) have no room for it, before
    /// its cells: a sink that sizes the arenas it builds sizes them here.
    #[inline(always)]
    fn enter_row(&mut self, _eng: &Engine, _i: usize) -> Result<(), OperationError> {
        Ok(())
    }
}

/// Build the output level: push pairs, emit product nodes, write `node_idx`.
///
/// The output-pair meter is charged from the pair arena's capacity as it
/// grows. Where the arena holds room ahead of the pairs written, reserved
/// before the row loop or sized ahead by [`ProjectedEmitSink`], the sink
/// keeps the meter, and every stop rule that reads it, where an arena grown
/// by pushes would have it: `charged` is the capacity that arena would have
/// now, and when the arena's length reaches it the meter is charged for the
/// next doubling ([`doubled_pairs_capacity`]) before the pair is pushed.
/// With `charged` at `usize::MAX` the room is the arena's spare capacity and
/// a full arena grows through [`try_push_pair_into`].
pub(crate) struct EmitSink<'a> {
    pub(crate) level: &'a mut TddLevel,
    /// The pair capacity the meters have been charged for, or `usize::MAX`.
    pub(crate) charged: usize,
    /// The nodes of one pair written, which the level stores inline: its
    /// live pairs are its arena's and these
    /// ([`TddLevel::leave_written_pairs`]).
    pub(crate) inline: &'a mut usize,
}

impl PairSink for EmitSink<'_> {
    #[inline(always)]
    fn single(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError> {
        let pair = raw_pair(lc, rc);
        let nid = self.level.nodes.stored().len() as u32;
        node_idx[grid_pos] = nid;
        *self.inline += 1;
        emit_single_pair(eng, self.level, pair)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.buf().len()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        if self.buf().len() == self.charged {
            let grown = doubled_pairs_capacity(self.charged);
            eng.limits().charge_output_pairs(grown - self.charged);
            self.charged = grown;
            super::super::note_scheduled_charge();
            debug_assert!(
                self.buf().len() < self.buf().capacity(),
                "a reserved arena holds every pair its level writes"
            );
        }
        try_push_pair_into(
            eng,
            self.level,
            raw_pair(lc, rc),
        )
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.level.pairs.stored_mut()
    }

    /// Zero where the length is past the charged capacity.
    #[inline(always)]
    fn room(&mut self) -> usize {
        let charged = self.charged;
        let v = self.buf();
        charged.min(v.capacity()).saturating_sub(v.len())
    }

    #[inline(always)]
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<(), OperationError> {
        emit_product_node(eng, self.level, node_idx, grid_pos, start, self.inline)
    }

    /// A cell the meter's schedule leaves room for: one that would reach
    /// `charged` goes through [`pair`](PairSink::pair), which charges it.
    #[inline(always)]
    fn open_cell(&mut self, bound: usize) -> Option<ArenaCell<'_>> {
        let room = self.room();
        ArenaCell::open(self.level, self.inline, room, bound)
    }

    /// A row the meter's schedule leaves room for, as
    /// [`open_cell`](PairSink::open_cell).
    #[inline(always)]
    fn open_row(&mut self, pairs: usize, nodes: usize, width_cap: usize) -> Option<ArenaRow<'_>> {
        let room = self.room();
        ArenaRow::open(self.level, self.inline, (room, pairs, nodes), width_cap)
    }
}

/// The share of a row loop's rows a [`ProjectedEmitSink`] writes before it
/// may reserve its level's arenas, as one in this many.
const AHEAD_AFTER: usize = 64;

/// The most a [`ProjectedEmitSink`] reserves its pair arena for, as a
/// multiple of the larger operand level's pairs, and its node arena for,
/// of the larger operand level's nodes.
const AHEAD_OPERAND_FACTOR: usize = 8;

/// An [`EmitSink`] that sizes its level's arenas by projection: at
/// the first row past one in [`AHEAD_AFTER`] of the row loop's rows that
/// they have no room for, it moves them to fresh ones of what the rows
/// before it project the loop writes, an eighth more, copying only what
/// they hold, and a pair arena that fills after that grows to the
/// projection of the rows written by then. Grown
/// by doubling from their seeds instead, the arenas of a large level are
/// copied whole at each growth, and those copies and the page faults that
/// follow them took a twentieth of the cycles of a program whose levels
/// are wide.
///
/// The output-pair meter stays on the doubling schedule an arena grown by
/// pushes grows on, so no stop rule reading it moves; the
/// arenas are charged to the budget as they grow. A projection is bounded
/// by what the level can write, by [`AHEAD_OPERAND_FACTOR`] times the
/// larger operand level, and by half the memory left; an arena it leaves
/// short doubles. Under bounded growth the sink is an [`EmitSink`] with
/// `charged` at `usize::MAX`.
pub(crate) struct ProjectedEmitSink<'a> {
    sink: EmitSink<'a>,
    /// The most pairs and nodes a projection sizes the arenas for, past
    /// the lengths `start` they had when the row loop began.
    most: (usize, usize),
    start: (usize, usize),
    /// The node arena's capacity when the row loop began.
    node_seed: usize,
    /// Where the pair arena's room ends: its capacity, or the capacity the
    /// output-pair meter is charged for where that is less. At most the
    /// capacity; set again wherever either moves.
    limit: usize,
    /// The row loop's rows, and the first row that sizes the arenas
    /// where it goes through the sink, `usize::MAX` for none or once one
    /// has.
    total: usize,
    at: usize,
    /// The row loop's rows once a row has sized the arenas, and 0 before;
    /// the row the cells that follow are of.
    rows: usize,
    row: usize,
}

impl<'a> ProjectedEmitSink<'a> {
    /// The sink of `level` for a row loop of `total` rows, whose
    /// projections reach at most `most` pairs and nodes past the arenas'
    /// lengths now ([`ahead_most`]).
    #[inline]
    pub(crate) fn new(
        lim: &crate::limits::Limits,
        most: (usize, usize),
        total: usize,
        level: &'a mut TddLevel,
        inline: &'a mut usize,
    ) -> Self {
        let bounded = lim.bounded_growth();
        let start = (level.pairs.len(), level.nodes.stored().len());
        let node_seed = level.nodes.stored_mut().capacity();
        let limit = level.pairs.capacity();
        let charged = if bounded { usize::MAX } else { limit };
        let after = super::super::ahead_after_forced().unwrap_or(AHEAD_AFTER);
        let at = if bounded || total < after || super::super::ahead_forced_off() {
            usize::MAX
        } else {
            total / after
        };
        ProjectedEmitSink {
            sink: EmitSink {
                level,
                charged,
                inline,
            },
            most,
            start,
            node_seed,
            limit,
            total,
            at,
            rows: 0,
            row: 0,
        }
    }

    /// The arenas sized for the row loop from the `done` rows written: to
    /// what those rows project it writes, where that fits half the memory
    /// left.
    #[inline(never)]
    pub(crate) fn reserve_ahead(
        &mut self,
        eng: &Engine,
        done: usize,
    ) -> Result<(), OperationError> {
        let rows = self.total;
        let lim = eng.limits();
        if done == 0 || self.sink.charged == usize::MAX || lim.bounded_growth() {
            return Ok(());
        }
        self.rows = rows;
        let level = &mut *self.sink.level;
        let (pairs, nodes) = (level.pairs.len(), level.nodes.stored().len());
        let pairs_to = projected_len(pairs, self.start.0, done, rows, self.most.0);
        let nodes_to = projected_len(nodes, self.start.1, done, rows, self.most.1);
        let more_pairs = pairs_to.saturating_sub(level.pairs.capacity());
        let more_nodes = nodes_to.saturating_sub(level.nodes.stored_mut().capacity());
        let bytes = (more_pairs * std::mem::size_of::<ChildPair>()
            + more_nodes * std::mem::size_of::<EncodedNode>()) as u64;
        if bytes == 0 || bytes > lim.headroom() / 2 {
            return Ok(());
        }
        super::super::note_pairs_grown();
        super::super::note_ahead(false);
        lim.reserve_exact_moving(level.pairs.stored_mut(), pairs_to.saturating_sub(pairs))?;
        lim.reserve_exact_moving(level.nodes.stored_mut(), nodes_to.saturating_sub(nodes))?;
        self.set_limit();
        Ok(())
    }

    /// The pair arena's limit set again from its capacity and the meter's.
    #[inline(always)]
    fn set_limit(&mut self) {
        self.limit = self.sink.charged.min(self.sink.level.pairs.capacity());
    }

    /// A pair at the limit, pushed as [`EmitSink`] pushes it, a full
    /// arena grown first ([`grow_projected`]).
    #[cold]
    #[inline(never)]
    fn pair_past_limit(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        let v = self.sink.level.pairs.stored_mut();
        if v.len() == v.capacity() && self.sink.charged != usize::MAX {
            grow_projected(eng, v, self.start.0, self.most.0, self.row, self.rows)?;
        }
        self.sink.pair(eng, lc, rc)?;
        self.set_limit();
        Ok(())
    }

    /// The arenas, once the row loop is done, given back the room a
    /// projection left past what doubling would have grown them to: the
    /// pair arena past the capacity the output-pair meter is charged for,
    /// the node arena past its doubling from its capacity as the loop
    /// began. A projection the first rows overstate would otherwise stay
    /// with the level for as long as it is held.
    #[inline]
    pub(crate) fn trim(&mut self) {
        if self.rows == 0 {
            return;
        }
        let level = &mut *self.sink.level;
        super::super::budget::trim_to(level.pairs.stored_mut(), self.sink.charged);
        let nodes = level.nodes.stored_mut();
        super::super::budget::trim_to(
            nodes,
            super::super::budget::doubled_to(self.node_seed, nodes.len()),
        );
    }
}

/// The most pairs and nodes a [`ProjectedEmitSink`] sizes a level of
/// `cells` cells of the levels `f` and `g` for: what it can write, within
/// [`AHEAD_OPERAND_FACTOR`] times the larger operand level.
#[inline]
pub(crate) fn ahead_most(f: &TddLevel, g: &TddLevel, cells: usize) -> (usize, usize) {
    let (af, nf, ag, ng) = (
        f.pairs.len() as u128,
        f.node_count(),
        g.pairs.len() as u128,
        g.node_count(),
    );
    // As `open_level_arenas` bounds the level's arena.
    let bound = af
        .saturating_mul(ag.saturating_add(ng as u128))
        .saturating_add((nf as u128).saturating_mul(ag));
    let factor = (AHEAD_OPERAND_FACTOR as u128).saturating_mul(af.max(ag));
    (
        bound.min(factor) as usize,
        cells.min(AHEAD_OPERAND_FACTOR.saturating_mul(nf.max(ng))),
    )
}

/// The length an arena that holds `len`, `start` of it before a walk
/// `done` of whose `total` units are walked, is projected to reach, an
/// eighth more, within `most` past `start`.
#[inline(always)]
fn projected_len(len: usize, start: usize, done: usize, total: usize, most: usize) -> usize {
    let projected = len.saturating_sub(start) as u128 * total as u128 / done as u128;
    start + (projected + projected / 8).min(most as u128) as usize
}

/// A full pair arena of a [`ProjectedEmitSink`] grown, charged to the
/// budget and not to the output-pair meter, which [`EmitSink`]
/// charges on its own schedule: to the projection of the `row` rows of
/// `rows` written, past the arena's start `start` and within `most` past
/// it, where the row loop has passed the row it projects at and the
/// projection fits half the memory left, and by doubling elsewhere.
#[cold]
#[inline(never)]
fn grow_projected(
    eng: &Engine,
    v: &mut Vec<ChildPair>,
    start: usize,
    most: usize,
    row: usize,
    rows: usize,
) -> Result<(), OperationError> {
    super::super::note_pairs_grown();
    let lim = eng.limits();
    let len = v.len();
    if rows != 0 && row != 0 {
        let to = projected_len(len, start, row, rows, most);
        if to > len && ((to - len) * std::mem::size_of::<ChildPair>()) as u64 <= lim.headroom() / 2
        {
            return lim.reserve_exact(v, to - len);
        }
    }
    lim.reserve(v, 1)
}

impl PairSink for ProjectedEmitSink<'_> {
    #[inline(always)]
    fn single(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError> {
        self.sink.single(eng, node_idx, grid_pos, lc, rc)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.sink.begin()
    }

    /// A pair stored below the limit with one test; at the limit, pushed as
    /// [`EmitSink`] pushes it ([`pair_past_limit`](Self::pair_past_limit)).
    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        let limit = self.limit;
        let v = self.sink.level.pairs.stored_mut();
        let len = v.len();
        if len < limit {
            // Safety: the limit is within the capacity.
            unsafe {
                v.as_mut_ptr().add(len).write(raw_pair(lc, rc));
                v.set_len(len + 1);
            }
            return Ok(());
        }
        self.pair_past_limit(eng, lc, rc)
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.sink.buf()
    }

    /// None where the length is past the limit.
    #[inline(always)]
    fn room(&mut self) -> usize {
        self.limit.saturating_sub(self.sink.level.pairs.stored_mut().len())
    }

    #[inline(always)]
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<(), OperationError> {
        self.sink.end(eng, node_idx, grid_pos, start)
    }

    /// The row recorded; at the first row past [`AHEAD_AFTER`]'s share
    /// of them, the arenas sized ([`reserve_ahead`](Self::reserve_ahead)).
    #[inline(always)]
    fn enter_row(&mut self, eng: &Engine, i: usize) -> Result<(), OperationError> {
        self.row = i;
        if i < self.at {
            return Ok(());
        }
        self.at = usize::MAX;
        self.reserve_ahead(eng, i)
    }

    #[inline(always)]
    fn open_cell(&mut self, bound: usize) -> Option<ArenaCell<'_>> {
        let room = self.room();
        ArenaCell::open(self.sink.level, self.sink.inline, room, bound)
    }

    #[inline(always)]
    fn open_row(&mut self, pairs: usize, nodes: usize, width_cap: usize) -> Option<ArenaRow<'_>> {
        let room = self.room();
        ArenaRow::open(self.sink.level, self.sink.inline, (room, pairs, nodes), width_cap)
    }
}

/// [`EmitSink`] on a level built a row at a time in one reused grid row
/// starting at `base`: each cell that gets a node is recorded as a
/// `ProductEntry` of the current [row](PairSink::row) instead of being left
/// in a dense slab.
pub(super) struct ListSink<'a> {
    pub(super) emit: EmitSink<'a>,
    pub(super) product_list: &'a mut Vec<ProductEntry>,
    pub(super) base: usize,
    pub(super) row: usize,
}

impl ListSink<'_> {
    /// Record the cell at `grid_pos` if it got a node.
    #[inline(always)]
    fn record(
        &mut self,
        eng: &Engine,
        node_idx: &[u32],
        grid_pos: usize,
    ) -> Result<(), OperationError> {
        let nid = node_idx[grid_pos];
        if nid == NO_PRODUCT {
            return Ok(());
        }
        let entry = ProductEntry {
            f_idx: FNodeIdx(self.row as u32),
            g_idx: GNodeIdx((grid_pos - self.base) as u32),
            prod_idx: ProductNodeIdx(nid),
        };
        eng.limits().try_push(self.product_list, entry)
    }
}

impl PairSink for ListSink<'_> {
    const SLAB: bool = false;

    #[inline(always)]
    fn single(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError> {
        self.emit.single(eng, node_idx, grid_pos, lc, rc)?;
        self.record(eng, node_idx, grid_pos)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.emit.begin()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        self.emit.pair(eng, lc, rc)
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.emit.buf()
    }

    #[inline(always)]
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<(), OperationError> {
        self.emit.end(eng, node_idx, grid_pos, start)?;
        self.record(eng, node_idx, grid_pos)
    }

    #[inline(always)]
    fn row(&mut self, i: usize) {
        self.row = i;
    }
}
