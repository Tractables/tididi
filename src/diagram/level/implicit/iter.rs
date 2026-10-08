//! Pair iteration and node numbering for mixed-radix descriptions.

use crate::limits::OperationError;
use super::{ChildPair, EncodedChildRef, EncodedNode, Digit, ImplicitLevel, TddLevel, NO_PAIRS, RUN_PAIRS, each_place, inline_at, run_of};

impl ImplicitLevel {
    /// The child slots of the first pair of node `node`.
    #[inline]
    pub fn node_first(&self, node: usize) -> (i64, i64) {
        debug_assert!(node < self.nodes && self.counts_nodes());
        let mut at = Odometer::<0>::new(self.first);
        at.seat(&self.digits[self.within..], 0, node);
        at.slots()
    }

    /// The pairs of node `node`, in their order.
    #[inline]
    pub(crate) fn places(&self, node: usize) -> Places<'_> {
        self.places_from(self.node_first(node))
    }

    /// The pairs of the node whose first pair has the slots `first`, in
    /// their order.
    #[inline]
    pub(crate) fn places_from(&self, first: (i64, i64)) -> Places<'_> {
        debug_assert!(self.per_node < 1 << 31);
        Places { run: self.run.iter(), at: Odometer::new(first), run_no: 0, rest: (self.per_node - self.run.len()) as u32, level: self }
    }

    /// The place digits past a run's, which step a run's first pair on.
    #[inline]
    fn run_steps(&self) -> &[Digit] {
        &self.digits[self.run_digits..self.within]
    }

    /// The slots of the first pair of a node's run `run_no`, from `at`,
    /// the previous run's. Out of line, and by value, so that a reader of
    /// the places holds none of them in memory.
    #[inline(never)]
    fn run_after(&self, mut at: Odometer<0>, run_no: u32) -> Odometer<0> {
        at.step(self.run_steps(), run_no as usize);
        at
    }

    /// A reader of the nodes' first pairs in increasing node order
    /// ([`NodeCursor`]), at node 0.
    pub(crate) fn cursor(&self) -> NodeCursor<'_> {
        debug_assert!(self.counts_nodes());
        NodeCursor::new(&self.digits[self.within..], self.first)
    }

    /// Append the pairs of node `node` to `out`, in their order.
    pub fn pairs_of(&self, node: usize, out: &mut Vec<ChildPair>) {
        self.places(node).write_into(out);
    }

    /// Append the pairs of nodes `0..nodes` to `buf`, node by node in
    /// their order: the first `nodes · k` pairs the description generates.
    pub(crate) fn pairs_of_first(&self, nodes: usize, buf: &mut Vec<ChildPair>) {
        debug_assert!(nodes <= self.nodes);
        buf.reserve(nodes * self.per_node);
        let mut cursor = self.cursor();
        for i in 0..nodes {
            self.places_from(cursor.first_of(i)).write_into(buf);
        }
    }

    /// The word of node `i` of a level whose nodes this description
    /// implies: its pair inline at one pair a node, else the range of pairs
    /// `i · k .. (i + 1) · k`.
    #[inline]
    pub(crate) fn node_word(&self, i: usize) -> EncodedNode {
        let k = self.per_node;
        if k == 1 {
            let (l, r) = self.node_first(i);
            inline_at((l as u32, r as u32), (0, 0))
        } else {
            EncodedNode::multi_pair((i * k) as u32, k as u32)
        }
    }

    /// [`node_word`](Self::node_word) for nodes read in increasing order:
    /// at one pair a node their pairs stepped on by `cursor`, which the
    /// first read makes, boxed, so that a reader that holds it stays small.
    #[inline]
    pub(crate) fn node_word_next<'a>(&'a self, cursor: &mut Option<Box<NodeCursor<'a>>>, i: usize) -> EncodedNode {
        let k = self.per_node;
        if k == 1 {
            let (l, r) = cursor.get_or_insert_with(|| Box::new(self.cursor())).first_of(i);
            inline_at((l as u32, r as u32), (0, 0))
        } else {
            EncodedNode::multi_pair((i * k) as u32, k as u32)
        }
    }

    /// Write the nodes of a level this describes whose nodes it does not
    /// imply, past 2^31 pairs, into `level`, whose nodes and pairs are
    /// empty, as the conjunction's row loop writes them: node `i` the arena
    /// range of pairs `i · k .. (i + 1) · k`, in the side table where the
    /// plain word does not hold it. The arena itself is not written.
    pub(crate) fn write_nodes(&self, level: &mut TddLevel) -> Result<(), OperationError> {
        debug_assert!(level.nodes().is_empty() && level.pairs.is_empty() && !self.implies_nodes());
        let k = self.per_node;
        for i in 0..self.nodes {
            level.try_push_multi_by_range(i * k, k).map_err(|()| OperationError::OverBudget)?;
        }
        Ok(())
    }

    /// The nodes of a description of one pair a node, in runs: `run(from,
    /// offsets, at)` for consecutive pieces of them, in order, nodes `from ..
    /// from + offsets.len()` holding the pairs `at` plus `offsets`, wrapping;
    /// up to the first `Err`, which it returns.
    ///
    /// A run counts the places of the fastest node digits, the most whose
    /// places are at most `RUN_PAIRS` and at least the fastest; the offsets
    /// of those places are read once, and each run's first pair is stepped
    /// on from the last's by the node digits past them. A fastest digit of
    /// more places than `RUN_PAIRS` is read in pieces of `RUN_PAIRS` places,
    /// each piece's first pair as many units of it on from the last's.
    pub(super) fn node_runs<E>(&self, mut run: impl FnMut(usize, &[(u32, u32)], (u32, u32)) -> Result<(), E>) -> Result<(), E> {
        debug_assert!(self.per_node == 1 && self.counts_nodes());
        let one = [Digit { radix: 1, left: 0, right: 0, node: 1 }];
        let digits = match &self.digits[self.within..] {
            [] => &one[..],
            digits => digits,
        };
        let (run_digits, places) = match run_of(digits) {
            (0, _) => (1, digits[0].radix),
            run => run,
        };
        let piece = places.min(RUN_PAIRS);
        let lead = [Digit { radix: piece, ..digits[0] }];
        let mut offsets = Vec::with_capacity(piece);
        each_place(if places > piece { &lead } else { &digits[..run_digits] }, (0, 0), |l, r| {
            offsets.push((l as u32, r as u32));
        });
        let leap = ((piece as i64 * digits[0].left) as u32, (piece as i64 * digits[0].right) as u32);
        let mut runs = Odometer::<NODE_COUNTERS>::new(self.first);
        for (r, start) in (0..self.nodes).step_by(places).enumerate() {
            if r > 0 {
                runs.step(&digits[run_digits..], r);
            }
            let end = self.nodes.min(start + places);
            let mut at = runs.at;
            for from in (start..end).step_by(piece) {
                run(from, &offsets[..piece.min(end - from)], at)?;
                at = (at.0.wrapping_add(leap.0), at.1.wrapping_add(leap.1));
            }
        }
        Ok(())
    }

    /// Folds the pairs of nodes `from..` in their order, each with its node:
    /// at one pair a node a run of nodes at a time
    /// ([`node_runs`](Self::node_runs)), a pair a step; else node by node,
    /// each node's pairs in runs ([`Places`]) and its first pair stepped on
    /// from the last node's ([`NodeCursor`]). What a pass over every pair of
    /// the level reads, with no call a node.
    #[inline]
    pub(crate) fn fold_pairs<B>(&self, from: usize, init: B, mut f: impl FnMut(B, usize, ChildPair) -> B) -> B {
        if from >= self.nodes {
            return init;
        }
        if self.per_node == 1 {
            let mut acc = Some(init);
            let _ = self.node_runs::<std::convert::Infallible>(|start, offsets, at| {
                if start + offsets.len() > from {
                    let skip = from.saturating_sub(start);
                    let mut a = acc.take().expect("the fold's value between runs");
                    for (j, &o) in offsets[skip..].iter().enumerate() {
                        a = f(a, start + skip + j, pair_at(at, o));
                    }
                    acc = Some(a);
                }
                Ok(())
            });
            return acc.expect("the fold's value after the runs");
        }
        let mut cursor = self.cursor();
        let mut acc = init;
        for i in from..self.nodes {
            acc = self.places_from(cursor.first_of(i)).fold(acc, |a, pair| f(a, i, pair));
        }
        acc
    }

    /// Whether `nodes` are the nodes of this description of one pair a node,
    /// each holding its pair inline: compared word for word in runs
    /// ([`node_runs`](Self::node_runs)), a run at a time, where every slot
    /// the digits reach fits a stored side, so that the wrapping sums are
    /// the slots and no multi-pair word matches one.
    pub(super) fn holds_inline(&self, nodes: &[EncodedNode]) -> bool {
        nodes.len() == self.nodes
            && self.slots_fit_sides()
            && self.node_runs(|from, offsets, at| {
                let run = &nodes[from..from + offsets.len()];
                let same = run.iter().zip(offsets).fold(true, |same, (n, &o)| same & (*n == inline_at(at, o)));
                if same { Ok(()) } else { Err(()) }
            }).is_ok()
    }

}

/// The pair at `offset` from the slots `at`, summed wrapping, as a run's
/// pairs are.
#[inline(always)]
fn pair_at(at: (u32, u32), offset: (u32, u32)) -> ChildPair {
    ChildPair::new(EncodedChildRef::from_raw(at.0.wrapping_add(offset.0)), EncodedChildRef::from_raw(at.1.wrapping_add(offset.1)))
}

/// The pairs of an implicit level in their order, each with the index of
/// its node: what [`TddLevel::pairs_with_parent`] reads off such a level.
/// Read a pair at a time, a node's pairs are its [`Places`], its first pair
/// stepped on from the last node's by a [`NodeCursor`] the first read
/// makes; folded, as a pass over every pair reads them, the rest are read a
/// run of nodes at a time ([`ImplicitLevel::fold_pairs`]).
#[derive(Clone, Debug)]
pub(crate) struct LevelPairs<'a> {
    level: &'a ImplicitLevel,
    /// The node whose pairs `places` holds the rest of.
    node: u32,
    places: Places<'a>,
    /// The node after it.
    next: usize,
    cursor: Option<Box<NodeCursor<'a>>>,
}

impl<'a> LevelPairs<'a> {
    /// The pairs `level` describes, from node 0.
    #[inline]
    pub(crate) fn new(level: &'a ImplicitLevel) -> Self {
        LevelPairs { level, node: 0, places: Places::empty(), next: 0, cursor: None }
    }

    /// No pairs: what a stored level reads beside its arena.
    #[inline]
    pub(crate) fn empty() -> Self {
        LevelPairs::new(&NO_PAIRS)
    }
}

impl Iterator for LevelPairs<'_> {
    type Item = (u32, ChildPair);

    #[inline]
    fn next(&mut self) -> Option<(u32, ChildPair)> {
        loop {
            if let Some(pair) = self.places.next() {
                return Some((self.node, pair));
            }
            let level = self.level;
            if self.next >= level.nodes {
                return None;
            }
            let first = self.cursor.get_or_insert_with(|| Box::new(level.cursor())).first_of(self.next);
            self.places = level.places_from(first);
            self.node = self.next as u32;
            self.next += 1;
        }
    }

    /// The current node's pairs, then the rest a run of nodes at a time.
    #[inline]
    fn fold<B, F: FnMut(B, (u32, ChildPair)) -> B>(self, init: B, mut f: F) -> B {
        let node = self.node;
        let acc = self.places.fold(init, |acc, pair| f(acc, (node, pair)));
        self.level.fold_pairs(self.next, acc, |acc, i, pair| f(acc, (i as u32, pair)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.places.len() + (self.level.nodes - self.next) * self.level.per_node;
        (n, Some(n))
    }
}

impl ExactSizeIterator for LevelPairs<'_> {}

/// The digits a [`NodeCursor`]'s [`Odometer`] counts in place: every node
/// digit of a level, whose nodes are numbered in `u32` and whose radices
/// are two or more.
const NODE_COUNTERS: usize = 32;

/// The slots at a place of a run of digits, fastest first, counted like an
/// odometer: one place on, the fastest digit goes up one and carries into
/// the next where it wraps, so a step costs an add or two, not a division
/// a digit. The settings of the first `N` digits are counted in place; a
/// carry past them reads the next digits' settings off the place.
///
/// The slots are summed in `u32`, wrapping: a place's slots are child slots,
/// which fit in one, so the wrapping sums are the slots.
#[derive(Clone, Copy, Debug)]
struct Odometer<const N: usize> {
    /// The slots at the place counted.
    at: (u32, u32),
    /// The settings of the first `N` digits there.
    counts: [u32; N],
}

impl<const N: usize> Odometer<N> {
    /// At place 0, the slots `first`.
    #[inline]
    fn new(first: (i64, i64)) -> Self {
        Odometer { at: (first.0 as u32, first.1 as u32), counts: [0; N] }
    }

    /// The slots at the place counted.
    #[inline]
    fn slots(&self) -> (i64, i64) {
        (i64::from(self.at.0), i64::from(self.at.1))
    }

    /// Add `times` units of digit `d` to the slots.
    #[inline]
    fn add(&mut self, d: &Digit, times: i64) {
        self.at = (self.at.0.wrapping_add((times * d.left) as u32), self.at.1.wrapping_add((times * d.right) as u32));
    }

    /// One place on, to `place`, of `digits`, where the fastest digit is
    /// counted in place: the step that carries nowhere inlined, any other
    /// out of line.
    #[inline(always)]
    fn step_counted(&mut self, digits: &[Digit], place: usize) {
        if let (Some(c), Some(d)) = (self.counts.first_mut(), digits.first())
            && (*c as usize) + 1 < d.radix
        {
            *c += 1;
            self.add(d, 1);
            return;
        }
        self.step_carrying(digits, place);
    }

    /// [`step`](Self::step), out of line.
    #[inline(never)]
    fn step_carrying(&mut self, digits: &[Digit], place: usize) {
        self.step(digits, place);
    }

    /// One place on, to `place`, of `digits`.
    #[inline]
    fn step(&mut self, digits: &[Digit], place: usize) {
        let mut period = 1;
        for (j, d) in digits.iter().enumerate() {
            let wrapped = match self.counts.get_mut(j) {
                Some(c) => {
                    *c += 1;
                    let wrapped = *c as usize == d.radix;
                    if wrapped {
                        *c = 0;
                    }
                    wrapped
                }
                None => (place / period).is_multiple_of(d.radix),
            };
            if !wrapped {
                self.add(d, 1);
                return;
            }
            self.add(d, 1 - d.radix as i64);
            period *= d.radix;
        }
    }

    /// From place `from` to place `to` of `digits`, read off its digits:
    /// a digit counted in place has its setting at `from` there.
    fn seat(&mut self, digits: &[Digit], from: usize, to: usize) {
        let (mut b, mut period) = (to, 1usize);
        for (j, d) in digits.iter().enumerate() {
            let is = b % d.radix;
            b /= d.radix;
            let was = match self.counts.get_mut(j) {
                Some(c) => std::mem::replace(c, is as u32) as usize,
                None => from / period % d.radix,
            };
            self.add(d, is as i64 - was as i64);
            period = period.wrapping_mul(d.radix);
        }
    }

    /// `by` places on, of `digits`, every one of them counted in place:
    /// `by` added to the settings with its carries, the digits past the
    /// last one a carry reaches left as they are.
    fn advance(&mut self, digits: &[Digit], by: usize) {
        debug_assert!(digits.len() <= N, "a digit not counted in place");
        let mut carry = by;
        for (c, d) in self.counts.iter_mut().zip(digits) {
            if carry == 0 {
                return;
            }
            let sum = *c as usize + carry;
            let is = sum % d.radix;
            carry = sum / d.radix;
            let times = is as i64 - i64::from(*c);
            *c = is as u32;
            self.at = (self.at.0.wrapping_add((times * d.left) as u32), self.at.1.wrapping_add((times * d.right) as u32));
        }
    }
}

/// The pairs of one node of an implicit level, in their order, generated
/// from its description: what [`PairsIter`](crate::diagram::PairsIter)
/// yields on such a level. They are read in runs, each a run's first pair
/// plus the offsets the description keeps of its fastest place digits, and
/// each run's first pair stepped on from the last's by the place digits
/// past those ([`Odometer`]).
#[derive(Clone, Debug)]
pub(crate) struct Places<'a> {
    /// The offsets of the current run still to come.
    run: std::slice::Iter<'a, (u32, u32)>,
    /// At the first pair of the current run, the `run_no`th of the node's.
    at: Odometer<0>,
    run_no: u32,
    /// The pairs of the runs after the current one.
    rest: u32,
    /// The description.
    level: &'a ImplicitLevel,
}

impl Places<'_> {
    /// No pairs: what a stored node's [`PairsIter`](crate::diagram::PairsIter)
    /// holds beside its slice.
    #[inline]
    pub(crate) fn empty() -> Self {
        Places { run: [].iter(), at: Odometer { at: (0, 0), counts: [] }, run_no: 0, rest: 0, level: &NO_PAIRS }
    }

    /// The pairs still to come.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.run.len() + self.rest as usize
    }

    /// The pair at `offset` from the current run's first.
    #[inline(always)]
    fn shifted(&self, offset: (u32, u32)) -> ChildPair {
        let (l, r) = self.at.at;
        ChildPair::new(EncodedChildRef::from_raw(l.wrapping_add(offset.0)), EncodedChildRef::from_raw(r.wrapping_add(offset.1)))
    }

    /// Appends the pairs still to come to `buf`, room made for them at
    /// once and each run written as a slice's map.
    #[inline]
    pub(crate) fn write_into(mut self, buf: &mut Vec<ChildPair>) {
        buf.reserve(self.len());
        loop {
            let run = std::mem::replace(&mut self.run, [].iter());
            let (l, r) = self.at.at;
            buf.extend(run.map(|&(a, b)| {
                ChildPair::new(EncodedChildRef::from_raw(l.wrapping_add(a)), EncodedChildRef::from_raw(r.wrapping_add(b)))
            }));
            if self.rest == 0 {
                return;
            }
            self.next_run();
        }
    }

    /// On to the next run, of which there must be one.
    #[inline(always)]
    fn next_run(&mut self) {
        let d = self.level;
        self.run_no += 1;
        self.at = d.run_after(self.at, self.run_no);
        self.rest -= d.run.len() as u32;
        self.run = d.run.iter();
    }
}

/// The first pairs of an implicit level's nodes, read in increasing node
/// order, in runs: a run is the places of the fastest node digits, the
/// most whose places are at most `RUN_PAIRS` and at least the fastest, and
/// a node's first pair is its run's plus what its place in the run adds,
/// read off a table of the run's places. Where the fastest digit alone has
/// more places, a run is that digit's places, read in stretches of
/// `RUN_PAIRS` off a table of its units. A run's first pair is stepped on
/// from the last's by the node digits past the run's ([`Odometer`]), one
/// further ahead by adding the distance to their settings, one behind read
/// off its digits.
///
/// The table is held in place, so that the cursor owns no memory of its
/// own: a reader that holds one drops it with nothing to free.
#[derive(Clone, Debug)]
pub(crate) struct NodeCursor<'a> {
    /// The node digits past a run's, fastest first.
    slow: &'a [Digit],
    /// At the first pair of run `run`, a run being `cycle` nodes.
    runs: Odometer<NODE_COUNTERS>,
    run: usize,
    pub(super) cycle: usize,
    /// The nodes of a stretch: a run's, or `RUN_PAIRS` of a lead digit's.
    pub(super) span: usize,
    /// What a unit of a lead digit adds; nothing in a tabled run.
    unit: (u32, u32),
    /// The current stretch: its first node, its nodes and its first pair.
    start: usize,
    places: usize,
    base: (u32, u32),
    /// What each place of a stretch adds to its first pair.
    offsets: [(u32, u32); RUN_PAIRS],
}

const _: () = assert!(RUN_PAIRS.is_power_of_two(), "a stretch's place is masked into the table");

impl<'a> NodeCursor<'a> {
    /// At node 0 of the nodes `digits` count, whose first pair is `first`.
    fn new(digits: &'a [Digit], first: (i64, i64)) -> Self {
        let mut offsets = [(0, 0); RUN_PAIRS];
        let (run_digits, cycle, span, unit) = match (run_of(digits), digits.first()) {
            ((0, _), Some(d)) => {
                let unit = (d.left as u32, d.right as u32);
                for (k, o) in offsets.iter_mut().enumerate() {
                    *o = ((k as u32).wrapping_mul(unit.0), (k as u32).wrapping_mul(unit.1));
                }
                (1, d.radix, RUN_PAIRS, unit)
            }
            ((run_digits, places), _) => {
                let mut k = 0;
                each_place(&digits[..run_digits], (0, 0), |l, r| {
                    offsets[k] = (l as u32, r as u32);
                    k += 1;
                });
                (run_digits, places, places, (0, 0))
            }
        };
        let runs = Odometer::new(first);
        let base = runs.at;
        NodeCursor { slow: &digits[run_digits..], runs, run: 0, cycle, span, unit, start: 0, places: span.min(cycle), base, offsets }
    }

    /// The slots of the first pair of node `i`, as
    /// [`ImplicitLevel::node_first`] gives them: a node of the current
    /// stretch in the caller, any other's stretch out of line
    /// ([`seek`](Self::seek)).
    #[inline(always)]
    pub(crate) fn first_of(&mut self, i: usize) -> (i64, i64) {
        let k = i.wrapping_sub(self.start);
        if k < self.places { self.place(k) } else { self.seek(i) }
    }

    /// The first pair of place `k` of the current stretch, `k` under its
    /// places, which are at most `RUN_PAIRS`: masked, the table is read
    /// without a bound check.
    #[inline(always)]
    fn place(&self, k: usize) -> (i64, i64) {
        let (a, b) = self.offsets[k & (RUN_PAIRS - 1)];
        (i64::from(self.base.0.wrapping_add(a)), i64::from(self.base.1.wrapping_add(b)))
    }

    /// On to the stretch of node `i`, which is not the current one, and
    /// `i`'s first pair: the next run stepped on to, one further ahead by
    /// adding the distance to the digits' settings, which costs a division
    /// a digit its carry reaches, and one behind read off its digits.
    #[inline(never)]
    fn seek(&mut self, i: usize) -> (i64, i64) {
        let run = i / self.cycle;
        if run != self.run {
            if run == self.run + 1 {
                self.runs.step_counted(self.slow, run);
            } else if run > self.run && self.slow.len() <= NODE_COUNTERS {
                self.runs.advance(self.slow, run - self.run);
            } else {
                self.runs.seat(self.slow, self.run, run);
            }
            self.run = run;
        }
        let within = i - run * self.cycle;
        let skip = within - within % self.span;
        self.start = run * self.cycle + skip;
        self.places = self.span.min(self.cycle - skip);
        let (l, r) = self.runs.at;
        let skip = skip as u32;
        self.base = (l.wrapping_add(skip.wrapping_mul(self.unit.0)), r.wrapping_add(skip.wrapping_mul(self.unit.1)));
        self.place(i - self.start)
    }
}

impl Iterator for Places<'_> {
    type Item = ChildPair;

    #[inline]
    fn next(&mut self) -> Option<ChildPair> {
        match self.run.next() {
            Some(&o) => Some(self.shifted(o)),
            None if self.rest == 0 => None,
            None => {
                self.next_run();
                self.run.next().map(|&o| self.shifted(o))
            }
        }
    }

    /// Each run folded as a slice's.
    #[inline]
    fn fold<B, F: FnMut(B, ChildPair) -> B>(mut self, init: B, mut f: F) -> B {
        let mut acc = init;
        loop {
            let run = std::mem::replace(&mut self.run, [].iter());
            acc = run.fold(acc, |acc, &o| f(acc, self.shifted(o)));
            if self.rest == 0 {
                return acc;
            }
            self.next_run();
        }
    }

    /// Skips to the place `n` on in one step, its run's first pair read off
    /// the digits past a run's.
    #[inline]
    fn nth(&mut self, n: usize) -> Option<ChildPair> {
        let here = self.run.len();
        if n < here {
            return self.run.nth(n).map(|&o| self.shifted(o));
        }
        let skip = n - here;
        if skip >= self.rest as usize {
            self.run = [].iter();
            self.rest = 0;
            return None;
        }
        let d = self.level;
        let size = d.run.len();
        let to = self.run_no as usize + skip / size + 1;
        self.at.seat(d.run_steps(), self.run_no as usize, to);
        self.rest -= ((to - self.run_no as usize) * size) as u32;
        self.run_no = to as u32;
        self.run = d.run[skip % size..].iter();
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
