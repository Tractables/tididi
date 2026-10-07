//! Reading a level: pair views, decoding, remapping, per-node pair counts, and
//! the canonical sort a rewritten pair list is put back in.

use crate::diagram::{ChildSide, EncodedChildRef};

use crate::diagram::marginal_ref::ChildDecoder;
use crate::diagram::PairsIter;
use crate::diagram::primitives::{ChildPair, EncodedNode, NodeKind};
use super::implicit::NodeCursor;
use super::{ImplicitLevel, LevelState, TddLevel};

/// A level's pairs as the level holds them: stored in its arena, or, on an
/// implicit level, as their description. Code that reads pairs one node at a
/// time takes [`TddLevel::pairs_iter_of_idx`], which generates an implicit
/// level's; code that wants slices or arithmetic says which it has here.
#[derive(Clone, Copy, Debug)]
pub enum Pairs<'a> {
    /// The pairs are stored.
    Stored(StoredPairs<'a>),
    /// The level is implicit: its pairs are those of this description.
    Implicit(&'a ImplicitLevel),
}

/// Where a node of an implicit level begins: the description, and the
/// slots of the node's first pair. The readers kept out of line return
/// this, and the node's [`PairsIter`] is made from it where the pairs are
/// read ([`described_iter`]): no call writes the iterator, so it can stay
/// in registers there, on stored levels too.
type NodeStart<'a> = (&'a ImplicitLevel, (u32, u32));

/// The pairs of the node that begins at `start`.
#[inline(always)]
fn described_iter((d, (l, r)): NodeStart<'_>) -> PairsIter<'_> {
    PairsIter::described(d.places_from((i64::from(l), i64::from(r))))
}

/// The one pair an inline node holds, as a slice of the node itself.
#[inline(always)]
fn inline_pair(node: &EncodedNode) -> &[ChildPair] {
    // Safety: EncodedNode is #[repr(C)] {a: u32, b: u32}.
    //         ChildPair is #[repr(C)] {left: EncodedChildRef(u32), right: EncodedChildRef(u32)}.
    //         For inline nodes, a == left.0 and b == right.0 by construction.
    //         Both types have identical {u32, u32} layout, so the cast is valid.
    unsafe { std::slice::from_ref(&*(node as *const EncodedNode as *const ChildPair)) }
}

/// The pairs of a level that stores them: a node's pairs are a slice of the
/// arena, or the node itself for a single pair.
#[derive(Clone, Copy, Debug)]
pub struct StoredPairs<'a> {
    level: &'a TddLevel,
    arena: &'a [ChildPair],
}

impl<'a> StoredPairs<'a> {
    /// The pairs of `node`, which must describe a node of this level.
    ///
    /// The slice borrows both the level and `node`, because an inline pair
    /// lives in the node.
    ///
    /// ```compile_fail,E0597
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let diagram = Tdd::one(&vtree);
    /// let level = diagram.level(vtree.root());
    /// let stored = level.stored().unwrap();
    /// let pairs;
    /// {
    ///     let node = level.nodes()[0];
    ///     pairs = stored.of(&node);
    /// }
    /// assert!(!pairs.is_empty()); // the copied node no longer exists
    /// ```
    #[inline]
    pub fn of<'n>(self, node: &'n EncodedNode) -> &'n [ChildPair]
    where
        'a: 'n,
    {
        match node.kind() {
            NodeKind::Inline(_) => inline_pair(node),
            NodeKind::Multi { .. } | NodeKind::MultiRanged(_) => &self.arena[self.level.multi_range(node)],
        }
    }

    /// The pairs of node `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not below `nodes().len()`.
    #[inline]
    pub fn of_idx(self, idx: usize) -> &'a [ChildPair] {
        self.of(&self.level.nodes[idx])
    }
}

impl TddLevel {
    /// The level's pairs, stored or described. Not valid on a marginal
    /// level, whose arena is empty and reads as stored.
    #[inline]
    pub fn pair_view(&self) -> Pairs<'_> {
        match self.pairs.stored() {
            Some(arena) => Pairs::Stored(StoredPairs { level: self, arena }),
            None => Pairs::Implicit(self.pairs.implicit().expect("an arena is stored or described")),
        }
    }

    /// The level's pairs when it stores them; `None` on an implicit level.
    #[inline]
    pub fn stored(&self) -> Option<StoredPairs<'_>> {
        match self.pair_view() {
            Pairs::Stored(s) => Some(s),
            Pairs::Implicit(_) => None,
        }
    }

    /// Every node as `(local index, pairs)`.
    /// The index is the node's slot in `nodes`, so it is valid for arrays
    /// sized by `slot_count()`. Empty on a marginal level.
    pub fn internal_inputs_iter(&self) -> impl Iterator<Item = (usize, PairsIter<'_>)> + '_ {
        self.internal_inputs_range(0..self.nodes.len())
    }

    /// Calls `f(i, pair)` with every pair of every node `i`, node by node:
    /// a stored level's pairs read as slices of its arena, an implicit
    /// level's generated into a buffer a node at a time. One loop calls
    /// `f`, so that it inlines there.
    #[inline]
    pub(crate) fn for_each_node_pair(&self, mut f: impl FnMut(usize, ChildPair)) {
        let stored = self.stored();
        let mut buf = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let pairs = match stored {
                Some(stored) => stored.of(node),
                None => self.pairs_read(i, &mut buf),
            };
            for &pair in pairs {
                f(i, pair);
            }
        }
    }

    /// Iterate structural nodes in a valid slot range, retaining their level indices.
    /// On an implicit level each node's first pair is stepped on from the
    /// one before it ([`NodeCursor`]), which the first read makes, boxed
    /// ([`described_cursor`](Self::described_cursor)).
    #[inline]
    pub(crate) fn internal_inputs_range(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = (usize, PairsIter<'_>)> + '_ {
        let start = range.start;
        let mut cursor = None;
        self.nodes[range].iter().enumerate().map(move |(i, n)| {
            let pairs = self.read_node(n, |at| {
                let cursor = cursor.get_or_insert_with(|| self.described_cursor());
                described_iter(self.described_next(cursor, at))
            });
            (start + i, pairs)
        })
    }

    /// The pairs of `node`, a node of this level: its inline pair, a slice
    /// of the stored arena, or, on an implicit level, what `described`
    /// generates from the arena position the node's pairs start at. The
    /// stored cases are the reads of a plain arena; an implicit level's
    /// nodes fail the arena's bounds check ([`PairArena::slots`]).
    ///
    /// [`PairArena::slots`]: super::PairArena::slots
    #[inline(always)]
    fn read_node<'a>(&'a self, node: &'a EncodedNode, described: impl FnOnce(usize) -> PairsIter<'a>) -> PairsIter<'a> {
        match node.kind() {
            NodeKind::Inline(_) => PairsIter::slice(inline_pair(node)),
            NodeKind::Multi { .. } | NodeKind::MultiRanged(_) => {
                let range = self.multi_range(node);
                match self.pairs.slots(range.clone()) {
                    Some(pairs) => PairsIter::slice(pairs),
                    None => described(range.start),
                }
            }
        }
    }

    /// The pairs of `node` as a slice: its inline pair or a slice of the
    /// stored arena; `None` on an implicit level, whose nodes fail the
    /// arena's bounds check.
    #[inline(always)]
    pub(crate) fn stored_of<'a>(&'a self, node: &'a EncodedNode) -> Option<&'a [ChildPair]> {
        match node.kind() {
            NodeKind::Inline(_) => Some(inline_pair(node)),
            NodeKind::Multi { .. } | NodeKind::MultiRanged(_) => self.pairs.slots(self.multi_range(node)),
        }
    }

    /// The pair-arena range a node of this `kind` owns, decoded from either
    /// the packed or the ranged (side-table) encoding; `None` for an inline
    /// node, whose pair is in the node itself.
    #[inline]
    pub(crate) fn arena_range(&self, kind: NodeKind) -> Option<std::ops::Range<usize>> {
        let (start, len) = match kind {
            NodeKind::Inline(_) => return None,
            NodeKind::Multi { start, len } => (start as usize, len as usize),
            NodeKind::MultiRanged(idx) => {
                let e = &self.ranges[idx as usize];
                (e.start as usize, e.len as usize)
            }
        };
        Some(start..start + len)
    }

    /// A multi-pair node's pair-arena range.
    ///
    /// # Panics
    ///
    /// Panics on an inline node, whose pair is not in the arena.
    #[inline]
    pub(crate) fn multi_range(&self, node: &EncodedNode) -> std::ops::Range<usize> {
        self.arena_range(node.kind()).expect("multi_range on an inline node")
    }

    /// The pairs of node `idx`, stored or generated from the level's
    /// description; not valid on a marginal level.
    #[inline]
    pub fn pairs_iter_of_idx(&self, idx: usize) -> PairsIter<'_> {
        debug_assert!(!self.is_marginal(), "pairs_iter_of_idx({idx}) called on marginal level");
        self.pairs_iter_of(&self.nodes[idx])
    }

    /// The pairs of node `idx`, collected: for tests and checkers, which
    /// compare and index them.
    #[cfg(any(test, debug_assertions, feature = "testing"))]
    #[doc(hidden)]
    pub fn pairs_vec(&self, idx: usize) -> Vec<ChildPair> {
        self.pairs_iter_of_idx(idx).collect()
    }

    /// Node `idx`'s pairs decoded to the bare coordinates structural use
    /// wants ([`ChildDecoder::coord`]): a stored level's own slice when
    /// neither child is marginal, else the pairs decoded or generated into
    /// `scratch`. The stored read inlines where it is called; the decode is
    /// out of line.
    #[inline]
    pub(crate) fn pairs_view_decoded<'a>(
        &'a self,
        idx: usize,
        scratch: &'a mut Vec<ChildPair>,
        left: ChildDecoder,
        right: ChildDecoder,
    ) -> &'a [ChildPair] {
        if !left.is_marginal() && !right.is_marginal() {
            return self.pairs_read(idx, scratch);
        }
        self.pairs_decoded(idx, scratch, left, right)
    }

    /// [`pairs_view_decoded`](Self::pairs_view_decoded) with a marginal
    /// child: the pairs decoded into `scratch`.
    #[inline(never)]
    fn pairs_decoded<'a>(
        &'a self,
        idx: usize,
        scratch: &'a mut Vec<ChildPair>,
        left: ChildDecoder,
        right: ChildDecoder,
    ) -> &'a [ChildPair] {
        scratch.clear();
        self.decode_pairs_into(idx, scratch, left, right);
        scratch.as_slice()
    }

    /// Append `idx`'s pairs, marginal-decoded, onto `out` (no clear). The
    /// caller pre-reserves `out` when the total is known.
    #[inline]
    pub(crate) fn decode_pairs_into(
        &self,
        idx: usize,
        out: &mut Vec<ChildPair>,
        left: ChildDecoder,
        right: ChildDecoder,
    ) {
        self.pairs_iter_of_idx(idx).for_each(|p| {
            out.push(ChildPair::new(EncodedChildRef::from_raw(left.coord(p.left)), EncodedChildRef::from_raw(right.coord(p.right))));
        });
    }

    /// The pairs of `node`, a node of this level, as an iterator: stored, or
    /// generated from the level's description.
    #[inline]
    pub fn pairs_iter_of<'a>(&'a self, node: &'a EncodedNode) -> PairsIter<'a> {
        self.read_node(node, |start| described_iter(self.described_pairs(start)))
    }

    /// This implicit level's description.
    fn described(&self) -> &ImplicitLevel {
        self.pairs.implicit().expect("an arena is stored or described")
    }

    /// Where the pairs of the node of this implicit level whose pairs the
    /// arena it stands for holds from `start` begin ([`described_iter`]).
    /// Kept out of line, so that the stored levels' read inlines where it
    /// is called.
    #[inline(never)]
    fn described_pairs(&self, start: usize) -> NodeStart<'_> {
        let d = self.described();
        let (l, r) = d.node_first(start / d.pairs_per_node());
        (d, (l as u32, r as u32))
    }

    /// [`described_pairs`](Self::described_pairs) for nodes read in
    /// increasing order, their first pairs stepped on by `cursor`.
    #[inline(never)]
    fn described_next<'a>(&'a self, cursor: &mut NodeCursor<'a>, start: usize) -> NodeStart<'a> {
        let d = self.described();
        let (l, r) = cursor.first_of(start / d.pairs_per_node());
        (d, (l as u32, r as u32))
    }

    /// A [`NodeCursor`] on this implicit level, at node 0. Boxed: a reader
    /// is passed its address, and an iterator that held it in place would
    /// stay in memory where it is read, on stored levels too.
    #[inline(never)]
    fn described_cursor(&self) -> Box<NodeCursor<'_>> {
        Box::new(self.described().cursor())
    }

    /// A stored multi-pair node's pairs, to change in place.
    ///
    /// # Panics
    ///
    /// Panics on an inline node or an implicit level.
    #[inline]
    #[track_caller]
    pub(crate) fn pairs_mut(&mut self, idx: usize) -> &mut [ChildPair] {
        debug_assert!(self.nodes[idx].kind().pairs_in_arena(), "pairs_mut called on inline node");
        let range = self.multi_range(&self.nodes[idx]);
        &mut self.pairs.stored_mut()[range]
    }

    /// Index-remap a stored multi-pair node's pairs in place: each side is
    /// rewritten through its lookup slice and [`ChildDecoder::remap`], which
    /// leaves a marginal side's inline values alone.
    ///
    /// Precondition (debug-asserted): `self.nodes[idx].kind().pairs_in_arena()`; every
    /// structural coordinate looked up is within its remap slice.
    #[inline]
    pub(crate) fn pairs_remap_indexed(
        &mut self,
        idx: usize,
        left_remap: &[u32],
        right_remap: &[u32],
        left: ChildDecoder,
        right: ChildDecoder,
    ) {
        for pair in self.pairs_mut(idx) {
            pair.left = left.remap(pair.left, left_remap);
            pair.right = right.remap(pair.right, right_remap);
        }
    }

    /// [`multi_range`](Self::multi_range) of the node at `idx`.
    #[inline]
    pub(crate) fn pair_range_at(&self, idx: usize) -> std::ops::Range<usize> {
        self.multi_range(&self.nodes[idx])
    }

    /// Number of pairs of the node at `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not below `nodes().len()`.
    ///
    /// Read off the node's words without decoding it: a multi-pair node
    /// never holds one pair, so a count of one is an inline node's, or the
    /// sentinel of a ranged node's, whose count is in the side table.
    #[inline]
    pub fn pair_count_at(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        match node.held_count() {
            1 => match node.kind() {
                NodeKind::MultiRanged(e) => self.ranges[e as usize].len as usize,
                _ => 1,
            },
            k => k as usize,
        }
    }

    /// A node holding other than `k` pairs, when one does: the node the
    /// last close found ([`uneven`](Self::uneven)) when it still does, else
    /// the first.
    pub(crate) fn uneven_node(&self, k: usize) -> Option<usize> {
        let hint = self.uneven as usize;
        if hint < self.nodes.len() && self.pair_count_at(hint) != k {
            return Some(hint);
        }
        (1..self.nodes.len()).find(|&i| self.pair_count_at(i) != k)
    }

    /// The pair count of every node in index order. Empty on a marginal
    /// level, which holds no nodes.
    #[inline]
    pub(crate) fn pair_counts(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.nodes.len()).map(|i| self.pair_count_at(i))
    }

    /// The pairs of this level's nodes, the nodes with pairs, and the nodes
    /// with one pair; nothing on a marginal level, which holds no nodes.
    ///
    /// Each node's count is read off its words without a branch
    /// ([`EncodedNode::held_count`]), and a ranged node's, read there as
    /// one pair, is put right from the side table after the pass, on a
    /// level that has one: the pass itself does not look for them. The counts fit `u64`: fewer than `2^32` nodes of
    /// fewer than `2^32` pairs each. Inlined, so that a caller that reads
    /// only the pairs counts nothing else.
    #[inline(always)]
    pub(crate) fn pair_census(&self) -> (u64, u64, u64) {
        let (mut pairs, mut live, mut single) = (0u64, 0u64, 0u64);
        for node in &self.nodes {
            let k = node.held_count();
            pairs += u64::from(k);
            live += u64::from(k != 0);
            single += u64::from(k == 1);
        }
        if !self.ranges.is_empty() {
            for node in &self.nodes {
                if let NodeKind::MultiRanged(e) = node.kind() {
                    let k = self.ranges[e as usize].len;
                    pairs = pairs - 1 + k;
                    live = live - 1 + u64::from(k != 0);
                    single = single - 1 + u64::from(k == 1);
                }
            }
        }
        (pairs, live, single)
    }

    /// The pairs held by this level's live nodes: an implicit level's off
    /// its description, every node of which holds its `k` pairs, a stored
    /// one's off its nodes ([`pair_census`](Self::pair_census)).
    #[inline]
    pub(crate) fn live_pairs(&self) -> usize {
        match self.implicit() {
            Some(d) => d.pairs(),
            None => self.pair_census().0 as usize,
        }
    }

    /// [`live_pairs`](Self::live_pairs) on a level of a closed diagram: a
    /// stored structural level's off the count kept once read
    /// ([`HeldPairs`](super::HeldPairs)), read and kept here otherwise.
    #[inline]
    pub(crate) fn live_pairs_closed(&self) -> usize {
        if let Some(d) = self.implicit() {
            return d.pairs();
        }
        let LevelState::Structural(held) = &self.state else {
            return self.pair_census().0 as usize;
        };
        match held.get() {
            Some(n) => n as usize,
            None => self.read_held_pairs(held),
        }
    }

    /// The census [`live_pairs_closed`](Self::live_pairs_closed) keeps, out
    /// of line.
    #[inline(never)]
    fn read_held_pairs(&self, held: &super::HeldPairs) -> usize {
        let n = self.pair_census().0;
        held.set(n);
        n as usize
    }

    /// Forget the live pairs kept on this level: an operation changed it.
    #[inline]
    pub(crate) fn forget_held_pairs(&mut self) {
        if let LevelState::Structural(held) = &mut self.state {
            *held = super::HeldPairs::unknown();
        }
    }

    /// Exchange the two sides of every pair: the level of the same functions
    /// over this vtree node with its two children swapped. An implicit
    /// level's description has its sides exchanged.
    ///
    /// A level's nodes are classes of assignments to the node's variables,
    /// and a node's pairs are the (left class, right class) products it
    /// holds, so the classes do not depend on which child is called left and
    /// the swapped level is canonical when this one is. Node indices do not
    /// move. Dead arena slots are swapped too; nothing reads them.
    pub(crate) fn swap_sides(&mut self) {
        for node in &mut self.nodes {
            if matches!(node.kind(), NodeKind::Inline(_)) {
                std::mem::swap(&mut node.a, &mut node.b);
            }
        }
        if self.pairs.implicit().is_some() {
            self.pairs.swap_described_sides();
        } else {
            for pair in self.pairs.stored_mut().iter_mut() {
                std::mem::swap(&mut pair.left, &mut pair.right);
            }
        }
        let left = self.has_value_refs(ChildSide::Left);
        self.set_has_value_refs(ChildSide::Left, self.has_value_refs(ChildSide::Right));
        self.set_has_value_refs(ChildSide::Right, left);
    }
}

/// Sorting network for 3..=8 elements: a fixed sequence of conditional swaps
/// (`cswap!(a, b)` = "if `s[a] > s[b]`, swap them") on any indexable,
/// swappable container.
macro_rules! sorting_network {
    ($s:expr, $n:expr) => {{
        macro_rules! cswap {
            ($a:expr, $b:expr) => { if $s[$a] > $s[$b] { $s.swap($a, $b); } };
        }
        match $n {
            3 => { cswap!(0,1); cswap!(1,2); cswap!(0,1); }
            4 => { cswap!(0,1); cswap!(2,3); cswap!(0,2); cswap!(1,3); cswap!(1,2); }
            5 => {
                cswap!(0,1); cswap!(3,4); cswap!(2,4); cswap!(2,3);
                cswap!(0,3); cswap!(0,2); cswap!(1,4); cswap!(1,3); cswap!(1,2);
            }
            6 => {
                cswap!(0,1); cswap!(2,3); cswap!(4,5);
                cswap!(0,2); cswap!(1,3); cswap!(0,4); cswap!(1,5);
                cswap!(1,2); cswap!(3,5); cswap!(2,4); cswap!(3,4); cswap!(1,2);
            }
            7 => {
                // Green's construction (Knuth, The Art of Computer Programming,
                // volume 3; 16 comparators).
                cswap!(0,4); cswap!(1,5); cswap!(2,6);
                cswap!(0,2); cswap!(1,3); cswap!(4,6);
                cswap!(2,4); cswap!(3,5);
                cswap!(0,1); cswap!(2,3); cswap!(4,5);
                cswap!(1,4); cswap!(3,6);
                cswap!(1,2); cswap!(3,4); cswap!(5,6);
            }
            8 => {
                cswap!(0,1); cswap!(2,3); cswap!(4,5); cswap!(6,7);
                cswap!(0,2); cswap!(1,3); cswap!(4,6); cswap!(5,7);
                cswap!(1,2); cswap!(5,6);
                cswap!(0,4); cswap!(3,7); cswap!(1,5); cswap!(2,6);
                cswap!(1,4); cswap!(3,6); cswap!(2,4); cswap!(3,5); cswap!(3,4);
            }
            _ => unreachable!("sorting_network called with n={}, expected 3..=8", $n),
        }
    }};
}

/// Sort a slice of input pairs in-place into ascending `(left, right)` order.
///
/// Sorting networks for ≤8 pairs, insertion sort for ≤24, and `sort_unstable`
/// for larger, after a check for an already sorted slice. Pair lists carry no
/// sorted invariant; this is for a rewrite that canonicalizes a list before
/// pushing the node.
#[inline]
pub(crate) fn sort_pairs(pairs: &mut [ChildPair]) {
    let n = pairs.len();
    if n < 2 { return; }
    if n == 2 {
        if pairs[0] > pairs[1] { pairs.swap(0, 1); }
        return;
    }
    // Already sorted is the common case.
    let mut sorted = true;
    for i in 1..n {
        if pairs[i - 1] > pairs[i] { sorted = false; break; }
    }
    if sorted { return; }
    match n {
        3..=8 => { sorting_network!(pairs, n); }
        9..=24 => {
            // Insertion sort: O(n²) with a low constant at this length.
            for i in 1..n {
                let key = pairs[i];
                let mut j = i;
                while j > 0 && pairs[j - 1] > key {
                    pairs[j] = pairs[j - 1];
                    j -= 1;
                }
                pairs[j] = key;
            }
        }
        _ => {
            // Sort via explicit u64 key `(left << 32) | right` instead of the
            // derived field-by-field Ord. The derive expands to a branchy
            // `left.cmp(&right) else right.cmp(...)`; the u64 form is a single
            // unsigned compare and lets the branchless partition in
            // `sort_unstable` kick in.
            pairs.sort_unstable_by_key(|p| p.key());
        }
    }
}

#[cfg(test)]
#[path = "tests/pairs/mod.rs"]
mod tests;
