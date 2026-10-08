//! Closing, rewriting and materializing implicit levels.

use crate::diagram::EncodedNode;
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
    /// on the stored level, and closes.
    pub(crate) fn move_described(&mut self, side: ChildSide, f: impl Fn(i64) -> i64) {
        let d = self.pairs.implicit().expect("move_described on a stored level");
        let id = |x: i64| x;
        let moved = if d.one_to_one_on(side, &f) {
            match side {
                ChildSide::Left => d.pruned(d.nodes, Some, || 0..d.nodes, &f, id),
                ChildSide::Right => d.pruned(d.nodes, Some, || 0..d.nodes, id, &f),
            }
        } else {
            None
        };
        match (moved, side) {
            (Some(moved), _) => self.pairs.redescribe(moved),
            (None, ChildSide::Left) => self.store_moved(|_| true, f, id),
            (None, ChildSide::Right) => self.store_moved(|_| true, id, f),
        }
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
    /// length and capacity the description stands for. A level the rewrite
    /// leaves as it was stays as it was. Nothing closes the level here: the
    /// form holds at the operation's end. Answers whether a node was left
    /// with no pairs.
    pub(crate) fn rewrite_described(
        &mut self,
        sorted: bool,
        mut rewrite_pair: impl FnMut(usize, usize, usize, ChildPair) -> Option<ChildPair>,
    ) -> bool {
        let d = self.pairs.implicit().expect("rewrite_described on a stored level").clone();
        let k = d.per_node;
        // A node's pairs are its first shifted by offsets every node shares,
        // so they are in order at every node or at none.
        let reorder = sorted && !d.offsets().is_sorted();
        // Once built: the arena, and how many pairs each node keeps.
        let mut built: Option<(Vec<ChildPair>, Vec<usize>)> = None;
        for i in 0..d.nodes {
            let mut kept = 0;
            for (r, p) in d.places(i).enumerate() {
                let np = rewrite_pair(i, r, k, p);
                if built.is_none() && (np != Some(p) || reorder) {
                    // Every pair before this one stood as it was.
                    let mut vec = Vec::with_capacity(self.pairs.capacity());
                    vec.extend((0..i).flat_map(|j| d.places(j)));
                    vec.extend(d.places(i).take(r));
                    kept = r;
                    built = Some((vec, vec![k; i]));
                }
                if let (Some((vec, _)), Some(np)) = (built.as_mut(), np) {
                    vec.push(np);
                    kept += 1;
                }
            }
            if let Some((vec, lens)) = built.as_mut() {
                vec.resize((i + 1) * k, d.places(0).next().expect("a node has pairs"));
                lens.push(kept);
            }
        }
        let Some((mut vec, lens)) = built else { return false };
        STORED.fetch_add(lens.iter().sum::<usize>() as u64, std::sync::atomic::Ordering::Relaxed);
        if k == 1 {
            // Every node held its pair inline, and holds what is left of it
            // inline, or the empty placeholder; the arena stays empty.
            let empty = self.encode_multi(0, 0);
            let words = vec.iter().zip(&lens).map(|(&p, &w)| if w == 1 { EncodedNode::inline(p) } else { empty });
            self.store_inline_nodes(words, vec[0]);
            return lens.contains(&0);
        }
        self.store_implied_nodes();
        let filler = vec[0];
        vec.resize(self.pairs.len(), filler);
        if sorted {
            for (i, &w) in lens.iter().enumerate() {
                super::super::sort_pairs(&mut vec[i * k..i * k + w]);
            }
        }
        self.pairs = PairArena::from(vec);
        let (mut dead, mut emptied) = (0, false);
        for (i, &w) in lens.iter().enumerate() {
            if w < k {
                dead += self.reencode_shrunk(i, i * k, k, w);
                emptied |= w == 0;
            }
        }
        self.note_dead_pairs(dead);
        emptied
    }

    /// Build an implicit level stored where its pairs lie, before a pass
    /// changes it in place ([`store_moved`](Self::store_moved), keeping
    /// every node and moving nothing); nothing on a stored level. The store
    /// is out of line: most levels a pass changes are stored.
    #[inline]
    pub(crate) fn store_if_implicit(&mut self) {
        if self.pairs.implicit().is_some() {
            self.store_described();
        }
    }

    /// [`store_if_implicit`](Self::store_if_implicit) on an implicit level.
    #[cold]
    #[inline(never)]
    fn store_described(&mut self) {
        self.store_moved(|_| true, |l| l, |r| r);
    }

    /// Store the pairs of an implicit level's nodes `keep` names, moved
    /// through `left` and `right`, when what a prune or a renumbering of its
    /// child levels leaves of it is not affine as numbered: node `i` at pairs
    /// `i · k .. (i + 1) · k` of an arena of the length and capacity the
    /// implicit one stood for, or, at one pair a node, holding its pair
    /// inline. The slots of the other nodes and those past the described
    /// pairs hold copies of the first pair as described, not moved, since
    /// its children may be gone; nothing reads them.
    pub(crate) fn store_moved(&mut self, keep: impl Fn(usize) -> bool, left: impl Fn(i64) -> i64, right: impl Fn(i64) -> i64) {
        let d = self.pairs.implicit().expect("store_moved on a stored level");
        let fill = d.places(0).next();
        if d.per_node == 1 {
            // Each node holds its moved pair inline, those `keep` does not
            // name the first; the arena holds no pair.
            let fill = fill.expect("a node has its pair");
            let (mut cursor, mut stored) = (d.cursor(), 0usize);
            let words: Vec<EncodedNode> = (0..d.nodes)
                .map(|i| {
                    if !keep(i) {
                        return EncodedNode::inline(fill);
                    }
                    stored += 1;
                    let (l, r) = cursor.first_of(i);
                    EncodedNode::inline(pair(left(l), right(r)))
                })
                .collect();
            STORED.fetch_add(stored as u64, std::sync::atomic::Ordering::Relaxed);
            self.store_inline_nodes(words.into_iter(), fill);
            return;
        }
        self.store_implied_nodes();
        let d = self.pairs.implicit().expect("store_moved on a stored level");
        let (len, capacity) = (self.pairs.len(), self.pairs.capacity());
        let mut vec = Vec::with_capacity(capacity);
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

    /// Store `words` as the nodes of an implicit level of one pair a node,
    /// at the capacity its node arena would have, and its pair arena as the
    /// stored one it stands for, of its length and capacity, its slots
    /// copies of `fill`: nothing reads them.
    fn store_inline_nodes(&mut self, words: impl ExactSizeIterator<Item = EncodedNode>, fill: ChildPair) {
        debug_assert!(self.implied_by().is_some_and(|d| d.per_node == 1));
        let (len, capacity) = (self.pairs.len(), self.pairs.capacity());
        let mut nodes = Vec::with_capacity(self.node_capacity().max(words.len()));
        nodes.extend(words);
        self.nodes = super::super::NodeArena::from(nodes);
        let mut pairs = Vec::with_capacity(capacity);
        pairs.resize(len, fill);
        self.pairs = PairArena::from(pairs);
    }
}
