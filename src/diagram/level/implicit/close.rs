//! Closing, rewriting and materializing implicit levels.

use crate::diagram::EncodedNode;
use crate::limits::{Limits, OperationError};
use super::super::arena::ArenaGrowth;
use super::super::LevelState;
use super::{ChildPair, ChildSide, ImplicitLevel, PairArena, TddLevel, STORED, floor, stored_levels_forced, pair};

impl crate::diagram::Tdd {
    /// The end of an operation that may have built levels stored: close every
    /// level ([`TddLevel::close`]), so that the canonical form of implicit
    /// levels holds when the operation returns. A debug build checks it.
    pub(crate) fn close_levels(&mut self) {
        self.levels.close();
        self.debug_check_implicit_levels();
    }

    /// [`close_levels`](Self::close_levels) at the end of an operation that
    /// names the levels it built or changed, `changed`: every other level is
    /// as the end of an earlier operation left it, closed already.
    pub(crate) fn close_changed_levels(&mut self, changed: &[crate::vtree::VtreeIdx]) {
        self.levels.close_changed(changed);
        self.debug_check_implicit_levels();
    }

    /// [`close_levels`](Self::close_levels) after edits that mark the levels
    /// they change, as a reduction's passes do: those levels when every
    /// level was closed before the edits (`closed_before`), every level
    /// otherwise.
    pub(crate) fn close_marked_levels(&mut self, closed_before: bool) {
        if closed_before {
            self.levels.close_marked();
        } else {
            self.levels.close();
        }
        self.debug_check_implicit_levels();
    }

    /// Record that every level is closed again, for levels put back as they
    /// were when every level was closed. A debug build checks it.
    pub(crate) fn reinstate_closed_levels(&mut self) {
        self.levels.reinstate_closed();
        self.debug_check_implicit_levels();
    }

    /// In a debug build, panic unless every level is in the canonical form of
    /// implicit levels; nothing in a release build.
    #[inline]
    pub(crate) fn debug_check_implicit_levels(&self) {
        #[cfg(debug_assertions)]
        if let Err(e) = crate::test_helpers::check::check_implicit_levels(self) {
            panic!("a level is out of canonical form at an operation's end: {e}");
        }
    }
}
impl TddLevel {
    /// The description of the level's pairs, when the level is implicit:
    /// node `i` holds pairs `i · k .. (i + 1) · k` of the arena, `k` the
    /// pairs of every node. `None` on a stored level.
    #[inline]
    pub fn implicit(&self) -> Option<&ImplicitLevel> {
        let d = self.pairs.implicit()?;
        debug_assert!({
            let k = d.per_node;
            self.nodes().len() == d.nodes
                && self.nodes().iter().enumerate().all(|(i, n)| match k {
                    1 => n == d.node_word(i),
                    _ => self.arena_range(n.kind()) == Some(i * k..(i + 1) * k),
                })
        });
        Some(d)
    }

    /// Move the child slots of an implicit level's pairs on `side` through
    /// `f`, as a renumbering of the child level there moves them: the level
    /// stays implicit when `f` is one to one on the slots its pairs name and
    /// the moved pairs are affine as numbered, and is stored moved otherwise
    /// ([`store_moved`](Self::store_moved)). A move that is not one to one
    /// makes two child nodes, or a leaf's two labels, one: twins, a
    /// duplicate pair or a fusion group the reduction that follows merges
    /// on the stored level, and closes. What the move reads and the stored
    /// level are reserved through `lim`.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when a reservation is refused; the
    /// level is then as it was.
    pub(crate) fn move_described(&mut self, lim: &Limits, side: ChildSide, f: impl Fn(i64) -> i64) -> Result<(), OperationError> {
        let d = self.pairs.implicit().expect("move_described on a stored level");
        let id = |x: i64| x;
        let moved = if d.one_to_one_on(lim, side, &f)? {
            match side {
                ChildSide::Left => d.pruned(lim, d.nodes, Some, || 0..d.nodes, &f, id)?,
                ChildSide::Right => d.pruned(lim, d.nodes, Some, || 0..d.nodes, id, &f)?,
            }
        } else {
            None
        };
        if let Some(moved) = moved {
            self.pairs.redescribe(moved);
            return Ok(());
        }
        let room = self.store_room(lim)?;
        match side {
            ChildSide::Left => self.store_moved(room, |_| true, f, id),
            ChildSide::Right => self.store_moved(room, |_| true, id, f),
        }
        Ok(())
    }

    /// Close a stored level: hold it as the description of its pairs when it
    /// can be one (see the canonical form of [`ImplicitLevel`]), its nodes
    /// implied, the arena keeping its length, capacity and dead slots and
    /// the node arena its capacity, so that the meters and the sweeps read
    /// it as they read the stored one. Nothing on an implicit level or one
    /// that cannot be, nor on an arena past 2^31 pairs, whose nodes' ranges
    /// may take the side table. Reads the pairs up to the first that is not
    /// affine, and charges nothing.
    #[inline]
    pub(crate) fn close(&mut self) {
        if self.closes_by_fit() {
            self.close_stored();
        }
    }

    /// Whether [`close`](Self::close) reads a fit of this level's pairs: a
    /// stored structural level of an arena it closes, whose node 0 holds
    /// one pair or more and whose nodes, at that count each, hold the
    /// floor's pairs or more. Only such a level can fit a description of
    /// the canonical form, and one that closing left stored, as at an
    /// operation's boundary, fits none.
    #[inline]
    pub(crate) fn closes_by_fit(&self) -> bool {
        // An implicit level's vectors are empty, below any floor: the one
        // test that most levels, small or implicit, take. A level of one
        // pair a node holds its pairs in its nodes.
        let (pairs, nodes) = (self.pairs.stored_vec_len(), self.nodes.stored().len());
        (pairs >= floor() || nodes >= floor())
            && pairs < 1 << 31
            && nodes != 0
            && self.pairs.implicit().is_none()
            && !stored_levels_forced()
            && matches!(self.state, LevelState::Structural(_))
            && {
                let k = self.pair_count_at(0);
                k >= 1 && nodes.saturating_mul(k) >= floor()
            }
    }

    /// [`close`](Self::close) on a level that [`closes_by_fit`](Self::closes_by_fit).
    /// Kept out of line: an operation closes every level of its result, and
    /// most are implicit or small.
    #[inline(never)]
    fn close_stored(&mut self) {
        // Most levels a close reads were read before, by the close at the end
        // of the operation they came from, and fail at the node that close
        // found.
        let d = match ImplicitLevel::fit_or_uneven(self) {
            Ok(d) => d,
            Err(uneven) => {
                if let Some(i) = uneven {
                    self.uneven = u16::try_from(i).unwrap_or(0);
                }
                return;
            }
        };
        // A fit's nodes hold node 0's pairs each, in an arena below 2^31
        // pairs, so the description implies them.
        debug_assert!(d.per_node >= 1 && d.pairs() >= floor() && d.implies_nodes());
        let node_capacity = self.nodes.imply();
        self.pairs.describe_stored(d, node_capacity);
    }

    /// Rewrite an implicit level's pairs through `rewrite_pair` and drop
    /// those it answers `None` for, as the in-place rewrites of a stored
    /// level's pairs do: `rewrite_pair` sees each pair with its node, its
    /// place in the node and the node's pair count, in the description's
    /// order. The pairs are read off the description; once one is dropped or
    /// changed, or from the first when `sorted` says the in-place route sorts
    /// every node's pairs and the description's are not in order, the level
    /// is built stored as the in-place route leaves a stored level: node
    /// `i`'s pairs at the start of its range `i · k .. (i + 1) · k`, sorted
    /// when `sorted` says so, a node of one pair inline and a node of none
    /// the empty placeholder, the slots it gave up dead, in an arena of the
    /// length and capacity the description stands for, reserved through
    /// `growth` ([`store_room`](Self::store_room)) when it is built. A level
    /// the rewrite leaves as it was stays as it was. Nothing closes the
    /// level here: the form holds at the operation's end. Answers whether a
    /// node was left with no pairs.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the room is refused; the level
    /// is then as it was.
    pub(crate) fn rewrite_described<G: ArenaGrowth>(
        &mut self,
        growth: &G,
        sorted: bool,
        rewrite_pair: impl FnMut(usize, usize, usize, ChildPair) -> Option<ChildPair>,
    ) -> Result<bool, OperationError> {
        self.rewrite_described_in(growth, None, sorted, rewrite_pair)
    }

    /// [`rewrite_described`](Self::rewrite_described) in `room`, reserved
    /// before through `growth` ([`store_room`](Self::store_room)), for a
    /// pass that reserves everything before it changes anything: with a
    /// room it is not refused. A room the rewrite does not use is dropped.
    ///
    /// # Errors
    ///
    /// As [`rewrite_described`](Self::rewrite_described), only without a
    /// room.
    pub(crate) fn rewrite_described_in<G: ArenaGrowth>(
        &mut self,
        growth: &G,
        mut room: Option<StoreRoom>,
        sorted: bool,
        mut rewrite_pair: impl FnMut(usize, usize, usize, ChildPair) -> Option<ChildPair>,
    ) -> Result<bool, OperationError> {
        let d = self.pairs.implicit().expect("rewrite_described on a stored level").clone();
        let (k, len) = (d.per_node, self.pairs.len());
        // A node's pairs are its first shifted by offsets every node shares,
        // so they are in order at every node or at none.
        let reorder = sorted && !d.places(0).is_sorted();
        let filler = d.places(0).next().expect("a node has pairs");
        let empty = self.encode_multi(0, 0);
        // Once built at one pair a node: the nodes' words, which hold their
        // pairs, and the room of the arena the level keeps; at more, the
        // level's arena is stored and its pairs written into it.
        let mut words: Option<(Vec<EncodedNode>, Vec<ChildPair>)> = None;
        let mut built = false;
        // The pair node 0 is left with, and what the nodes keep.
        let mut first = filler;
        let (mut stored, mut dead, mut emptied) = (0usize, 0usize, false);
        for i in 0..d.nodes {
            let mut kept = 0;
            for (r, p) in d.places(i).enumerate() {
                let np = rewrite_pair(i, r, k, p);
                if !built && (np != Some(p) || reorder) {
                    // Every pair before this one stood as it was.
                    let StoreRoom { nodes, pairs: mut arena } = match room.take() {
                        Some(room) => room,
                        None => self.store_room(growth)?,
                    };
                    stored = i * k;
                    if k == 1 {
                        let mut nodes = nodes;
                        let mut cursor = d.cursor();
                        nodes.extend((0..i).map(|j| {
                            let (l, r) = cursor.first_of(j);
                            EncodedNode::inline(pair(l, r))
                        }));
                        words = Some((nodes, arena));
                    } else {
                        self.store_implied_nodes_in(nodes);
                        let mut cursor = d.cursor();
                        for j in 0..i {
                            d.places_from(cursor.first_of(j)).write_into(&mut arena);
                        }
                        arena.extend(d.places(i).take(r));
                        self.pairs = PairArena::from(arena);
                    }
                    kept = r;
                    built = true;
                }
                if built && let Some(np) = np {
                    match words.as_mut() {
                        Some((words, _)) => {
                            if i == 0 {
                                first = np;
                            }
                            words.push(EncodedNode::inline(np));
                        }
                        None => self.pairs.stored_mut().push(np),
                    }
                    kept += 1;
                }
            }
            if !built {
                continue;
            }
            stored += kept;
            emptied |= kept == 0;
            if let Some((words, _)) = words.as_mut() {
                if kept == 0 {
                    words.push(empty);
                }
                continue;
            }
            let arena = self.pairs.stored_mut();
            arena.resize((i + 1) * k, filler);
            if sorted {
                super::super::sort_pairs(&mut arena[i * k..i * k + kept]);
            }
            if kept < k {
                dead += self.reencode_shrunk(i, i * k, k, kept);
            }
        }
        if !built {
            return Ok(false);
        }
        STORED.fetch_add(stored as u64, std::sync::atomic::Ordering::Relaxed);
        match words {
            // Every node held its pair inline, and holds what is left of it
            // inline, or the empty placeholder; the arena keeps its length,
            // its slots copies of node 0's pair.
            Some((words, arena)) => self.store_inline_nodes(words, arena, first),
            None => {
                let arena = self.pairs.stored_mut();
                let first = arena[0];
                arena.resize(len, first);
                self.note_dead_pairs(dead);
            }
        }
        Ok(emptied)
    }

    /// Build an implicit level stored where its pairs lie, before a pass
    /// changes it in place ([`store_moved`](Self::store_moved), keeping
    /// every node and moving nothing), its room reserved through `growth`;
    /// nothing on a stored level. The store is out of line: most levels a
    /// pass changes are stored.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the room is refused; the level
    /// is then as it was.
    #[inline]
    pub(crate) fn store_if_implicit<G: ArenaGrowth>(&mut self, growth: &G) -> Result<(), OperationError> {
        if self.pairs.implicit().is_some() {
            self.store_described(growth)?;
        }
        Ok(())
    }

    /// [`store_if_implicit`](Self::store_if_implicit) on an implicit level.
    #[cold]
    #[inline(never)]
    fn store_described<G: ArenaGrowth>(&mut self, growth: &G) -> Result<(), OperationError> {
        let room = self.store_room(growth)?;
        self.store_moved(room, |_| true, |l| l, |r| r);
        Ok(())
    }

    /// Reserve through `growth` the room an implicit level is stored in
    /// ([`StoreRoom`]): a node arena of the capacity the level's would have,
    /// where the description implies its nodes, and a pair arena of the
    /// capacity and length the description stands for. Nothing is written,
    /// and nothing charged: the meters count those capacities already
    /// ([`ArenaGrowth::counted_room`]).
    ///
    /// A level of billions of pairs takes gigabytes stored: the room is
    /// asked of the allocator before any of it is written, so that a
    /// refusal leaves the level as it was.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when either arena is refused;
    /// nothing is held then.
    pub(crate) fn store_room<G: ArenaGrowth>(&self, growth: &G) -> Result<StoreRoom, OperationError> {
        let d = self.pairs.implicit().expect("store_room on a stored level");
        let nodes = match self.implied_by() {
            Some(_) => self.pairs.node_capacity().max(d.nodes),
            None => 0,
        };
        let mut room = StoreRoom::default();
        growth.counted_room(&mut room.nodes, nodes)?;
        growth.counted_room(&mut room.pairs, self.pairs.capacity().max(self.pairs.len()))?;
        Ok(room)
    }

    /// Store the pairs of an implicit level's nodes `keep` names, moved
    /// through `left` and `right`, in `room`, reserved for it
    /// ([`store_room`](Self::store_room)), when what a prune or a renumbering
    /// of its child levels leaves of it is not affine as numbered: node `i`
    /// at pairs `i · k .. (i + 1) · k` of an arena of the length and capacity
    /// the implicit one stood for, or, at one pair a node, holding its pair
    /// inline. The slots of the other nodes and those past the described
    /// pairs hold copies of the first pair as described, not moved, since
    /// its children may be gone; nothing reads them.
    pub(crate) fn store_moved(&mut self, room: StoreRoom, keep: impl Fn(usize) -> bool, left: impl Fn(i64) -> i64, right: impl Fn(i64) -> i64) {
        let StoreRoom { nodes: mut words, pairs: mut vec } = room;
        let d = self.pairs.implicit().expect("store_moved on a stored level");
        let fill = d.places(0).next();
        if d.per_node == 1 {
            // Each node holds its moved pair inline, those `keep` does not
            // name the first; the arena holds no pair.
            let fill = fill.expect("a node has its pair");
            let (mut cursor, mut stored) = (d.cursor(), 0usize);
            words.extend((0..d.nodes).map(|i| {
                if !keep(i) {
                    return EncodedNode::inline(fill);
                }
                stored += 1;
                let (l, r) = cursor.first_of(i);
                EncodedNode::inline(pair(left(l), right(r)))
            }));
            STORED.fetch_add(stored as u64, std::sync::atomic::Ordering::Relaxed);
            self.store_inline_nodes(words, vec, fill);
            return;
        }
        self.store_implied_nodes_in(words);
        let d = self.pairs.implicit().expect("store_moved on a stored level");
        let len = self.pairs.len();
        let mut stored = 0usize;
        for i in 0..d.nodes {
            if keep(i) {
                vec.extend(d.places(i).map(|p| pair(left(i64::from(p.left.raw())), right(i64::from(p.right.raw())))));
                stored += d.per_node;
            } else if let Some(fill) = fill {
                vec.resize(vec.len() + d.per_node, fill);
            }
        }
        if let Some(fill) = fill {
            vec.resize(len, fill);
        }
        STORED.fetch_add(stored as u64, std::sync::atomic::Ordering::Relaxed);
        self.pairs = PairArena::from(vec);
    }

    /// Store `words`, in the room reserved for them, as the nodes of an
    /// implicit level of one pair a node, and `pairs`, the room reserved for
    /// its pair arena, as the stored one it stands for, of its length, its
    /// slots copies of `fill`: nothing reads them.
    fn store_inline_nodes(&mut self, words: Vec<EncodedNode>, mut pairs: Vec<ChildPair>, fill: ChildPair) {
        debug_assert!(self.implied_by().is_some_and(|d| d.per_node == 1));
        let len = self.pairs.len();
        self.nodes = super::super::NodeArena::from(words);
        pairs.resize(len, fill);
        self.pairs = PairArena::from(pairs);
    }

    /// Store the words of the nodes the level's description implies in
    /// `room`, reserved for them ([`store_room`](Self::store_room)), before
    /// the level is stored: nothing on a level that stores its nodes, whose
    /// room is empty.
    fn store_implied_nodes_in(&mut self, mut room: Vec<EncodedNode>) {
        if self.implied_by().is_none() {
            return;
        }
        room.extend(self.nodes().iter());
        self.nodes = super::super::NodeArena::from(room);
    }
}

/// The arenas an implicit level is stored in, reserved before any of it is
/// written ([`TddLevel::store_room`]): a node arena, empty, of the capacity
/// the level's would have where its description implies its nodes, and a
/// pair arena, empty, of the capacity the description stands for. The
/// meters count them as the level's already; one that goes unused is
/// dropped.
#[derive(Debug, Default)]
pub(crate) struct StoreRoom {
    nodes: Vec<EncodedNode>,
    pairs: Vec<ChildPair>,
}

