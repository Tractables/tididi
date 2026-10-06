//! Implicit levels: a level whose pairs are a complete mixed-radix product,
//! kept as the description of its pairs instead of its pair arena.
//!
//! A conjunction whose operands' levels at a vtree node are such products,
//! and whose children's products are complete, writes at that node the
//! product of the two: every node of one operand with every node of the
//! other, each with every pair of the one with every pair of the other. Its
//! pairs are then an affine function of the digits of a mixed radix, and
//! [`ImplicitLevel`] holds that function: the level's nodes stay stored, its
//! pair arena does not. [`TddLevel::materialize`] writes the arena the
//! conjunction would have written, pair for pair and in the same order.

use std::ops::{Deref, DerefMut};
use std::panic::Location;
use std::sync::{Mutex, OnceLock};

use crate::diagram::primitives::{ChildPair, EncodedChildRef, EncodedNode};
use crate::diagram::ChildSide;
use crate::limits::{Charged, Limits, OperationError};

use super::{LevelState, TddLevel};

/// One digit of an implicit level's numbering of its pairs: its radix, and
/// what one unit of it adds to a pair's left and right child slots and to
/// the index of the pair's node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digit {
    /// The places of the digit, two or more.
    pub radix: usize,
    /// What one unit adds to a pair's slot in the left child level.
    pub left: i64,
    /// What one unit adds to a pair's slot in the right child level.
    pub right: i64,
    /// What one unit adds to the index of the pair's node: zero for a digit
    /// of a pair's place within its node.
    pub node: i64,
}

/// A level held as the description of its pairs.
///
/// Every node has [`pairs_per_node`](Self::pairs_per_node) pairs. Pair `m`
/// of node `i` has the child slots `first() + Σ c_d · (left_d, right_d)`
/// over the [`digits`](Self::digits), where `c_d` are the digits of the
/// pair's position `i · pairs_per_node + m` in the level's pairs, fastest
/// first: the first [`within`](Self::within) digits number the place `m`,
/// the others the node `i`, whose index is `Σ c_d · node_d`.
///
/// The slots are the raw words of the pairs' sides, which on an implicit
/// level always name nodes of internal child levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplicitLevel {
    nodes: usize,
    per_node: usize,
    first: (i64, i64),
    digits: Vec<Digit>,
    within: usize,
}

impl ImplicitLevel {
    /// The level's nodes.
    #[inline]
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// The pairs of every node.
    #[inline]
    pub fn pairs_per_node(&self) -> usize {
        self.per_node
    }

    /// The level's pairs.
    #[inline]
    pub fn pairs(&self) -> usize {
        self.nodes * self.per_node
    }

    /// The child slots, left and right, of the first pair of node 0.
    #[inline]
    pub fn first(&self) -> (i64, i64) {
        self.first
    }

    /// The digits, fastest first: the [`within`](Self::within) digits of a
    /// pair's place in its node, then the digits of its node.
    #[inline]
    pub fn digits(&self) -> &[Digit] {
        &self.digits
    }

    /// How many of [`digits`](Self::digits) number a pair's place within its
    /// node.
    #[inline]
    pub fn within(&self) -> usize {
        self.within
    }

    /// The child slots of the first pair of node `node`.
    #[inline]
    pub fn node_first(&self, node: usize) -> (i64, i64) {
        debug_assert!(node < self.nodes);
        let mut at = self.first;
        let mut rest = node;
        for d in self.digits[self.within..].iter().rev() {
            let period = d.node as usize;
            let c = (rest / period) as i64;
            rest %= period;
            at = (at.0 + c * d.left, at.1 + c * d.right);
        }
        at
    }

    /// Append the pairs of node `node` to `out`, in their order.
    pub fn pairs_of(&self, node: usize, out: &mut Vec<ChildPair>) {
        let at = self.node_first(node);
        out.reserve(self.per_node);
        each_place(&self.digits[..self.within], at, |l, r| out.push(pair(l, r)));
    }

    /// The length of the pair arena the level stands for: its pairs, or none
    /// when every node has one pair and holds it inline.
    #[inline]
    pub(crate) fn arena_len(&self) -> usize {
        if self.per_node >= 2 { self.pairs() } else { 0 }
    }

    /// Calls `f` with the child slots of every pair, in the order of the
    /// level's pairs.
    #[inline]
    pub(crate) fn for_each_pair(&self, f: impl FnMut(i64, i64)) {
        each_place(&self.digits, self.first, f);
    }

    /// Write the level's nodes into `level`, whose nodes and pairs are empty,
    /// as the conjunction's row loop writes them: node `i` holds its pair
    /// inline when it has one, else the arena range of pairs `i · k ..
    /// (i + 1) · k`. The arena itself is not written.
    pub(crate) fn write_nodes(&self, lim: &Limits, level: &mut TddLevel) -> Result<(), OperationError> {
        debug_assert!(level.nodes.is_empty() && level.pairs.is_empty());
        if self.per_node == 1 {
            let mut out = Ok(());
            self.for_each_pair(|l, r| {
                if out.is_ok() {
                    out = lim.try_push(&mut level.nodes, EncodedNode::inline(pair(l, r)));
                }
            });
            return out;
        }
        for i in 0..self.nodes {
            level
                .try_push_multi_by_range(i * self.per_node, self.per_node)
                .map_err(|()| OperationError::OverBudget)?;
        }
        Ok(())
    }

    /// Write the pairs, in their order, into `out`.
    fn write_pairs(&self, out: &mut Vec<ChildPair>) {
        out.reserve(self.pairs());
        self.for_each_pair(|l, r| out.push(pair(l, r)));
    }

    /// Append the pairs at positions `range` of the level's pairs to `out`.
    fn write_range(&self, range: std::ops::Range<usize>, out: &mut Vec<ChildPair>) {
        let k = self.per_node;
        out.reserve(range.len());
        if range.start.is_multiple_of(k) && range.len() == k {
            let at = self.node_first(range.start / k);
            each_place(&self.digits[..self.within], at, |l, r| out.push(pair(l, r)));
            return;
        }
        for p in range {
            let (l, r) = self.at(p);
            out.push(pair(l, r));
        }
    }

    /// The child slots of the pair at position `p` of the level's pairs.
    fn at(&self, p: usize) -> (i64, i64) {
        let (mut l, mut r) = self.first;
        let mut rest = p;
        for d in &self.digits {
            let c = (rest % d.radix) as i64;
            rest /= d.radix;
            l += c * d.left;
            r += c * d.right;
        }
        (l, r)
    }

    /// The description of `level`, when it is one: every node has the same
    /// number of pairs, a node's pairs are its first shifted by offsets all
    /// nodes share, the offsets are affine in the digits of a pair's place,
    /// and the first pairs affine in the digits of the node's index. Read
    /// and checked in one pass over the level's pairs.
    pub(crate) fn fit(level: &TddLevel) -> Option<ImplicitLevel> {
        if !matches!(level.state, LevelState::Structural) {
            return None;
        }
        let nodes = level.nodes.len();
        if nodes == 0 {
            return None;
        }
        let first_pairs = level.pairs_of_idx(0);
        let per_node = first_pairs.len();
        if per_node == 0 || (1..nodes).any(|i| level.pair_count_at(i) != per_node) {
            return None;
        }
        let first = slots(&first_pairs[0]);
        let offset = |m: usize| {
            let (l, r) = slots(&first_pairs[m]);
            Some((l - first.0, r - first.1))
        };
        let within = read_digits(per_node, offset)?;
        let across = read_digits(nodes, |i| level.pairs_of_idx(i).first().map(slots))?;
        let fitted = ImplicitLevel::assemble(nodes, per_node, first, &within, &across);
        fitted.holds(|i| Some(level.pairs_of_idx(i).iter().map(slots))).then_some(fitted)
    }

    /// The child slots every pair of a node adds to its first, in their
    /// order.
    fn places(&self) -> Vec<(i64, i64)> {
        let mut places = Vec::with_capacity(self.per_node);
        each_place(&self.digits[..self.within], (0, 0), |l, r| places.push((l, r)));
        places
    }

    /// Whether `node(i)` gives the pairs of node `i` of this description,
    /// in their order, for every node: read node by node, up to the first
    /// pair that differs.
    fn holds<I: Iterator<Item = (i64, i64)>>(&self, mut node: impl FnMut(usize) -> Option<I>) -> bool {
        let places = self.places();
        (0..self.nodes).all(|i| {
            let at = self.node_first(i);
            node(i).is_some_and(|pairs| pairs.eq(places.iter().map(|p| (at.0 + p.0, at.1 + p.1))))
        })
    }

    /// The description of the level of `nodes` nodes of `per_node` pairs
    /// whose first pair is `first`, with the place digits `within` of a
    /// node's pairs and the digits `across` of its index, each a radix and
    /// what one unit adds to the slots, fastest first.
    fn assemble(
        nodes: usize,
        per_node: usize,
        first: (i64, i64),
        within: &[(usize, (i64, i64))],
        across: &[(usize, (i64, i64))],
    ) -> ImplicitLevel {
        let mut digits: Vec<Digit> = within.iter().map(|&(radix, (l, r))| Digit { radix, left: l, right: r, node: 0 }).collect();
        let mut period = 1usize;
        for &(radix, (l, r)) in across {
            digits.push(Digit { radix, left: l, right: r, node: period as i64 });
            period *= radix;
        }
        ImplicitLevel { nodes, per_node, first, digits, within: within.len() }
    }

    /// The description of what a prune leaves of this level, when it is
    /// one: the `nodes` nodes it keeps, the `j`th of them node `kept(j)` of
    /// this description, each with its pairs, whose child slots `left` and
    /// `right` move to where the prune put the children's nodes. Read off
    /// the moved pairs and checked at every one of them, without writing
    /// any. `None` when they are not affine in a mixed radix, or `kept`
    /// names no node of this description.
    pub(crate) fn pruned(
        &self,
        nodes: usize,
        mut kept: impl FnMut(usize) -> Option<usize>,
        left: impl Fn(i64) -> i64,
        right: impl Fn(i64) -> i64,
    ) -> Option<ImplicitLevel> {
        let mut node = |j: usize| kept(j).filter(|&i| i < self.nodes).map(|i| self.node_first(i));
        let places = self.places();
        let moved = |at: (i64, i64), p: &(i64, i64)| (left(at.0 + p.0), right(at.1 + p.1));
        let at = node(0)?;
        let first = moved(at, &places[0]);
        let within = read_digits(self.per_node, |m| {
            let (l, r) = moved(at, &places[m]);
            Some((l - first.0, r - first.1))
        })?;
        let across = read_digits(nodes, |j| Some(moved(node(j)?, &places[0])))?;
        let fitted = ImplicitLevel::assemble(nodes, self.per_node, first, &within, &across);
        fitted.holds(|j| node(j).map(|at| places.iter().map(move |p| moved(at, p)))).then_some(fitted)
    }

    /// The description of the conjunction's level whose operands' levels
    /// are `f` and `g`: node `(i, j)` at index `i · g.nodes() + j`, and its
    /// pairs every pair of `f`'s node `i` with every pair of `g`'s node `j`,
    /// the product's child slots `a · stride + c` on each side, for slots
    /// `a` of `f` and `c` of `g` and `stride` the widths of `g`'s child
    /// levels.
    ///
    /// The order is the row loop's: `f`'s pairs outer, `g`'s inner, or, on a
    /// `grouped` level, a run of `f`'s pairs that share a left slot, then a
    /// run of `g`'s, then the pairs of the one run, then of the other. `None`
    /// when a grouped level's runs are not digits of the operands.
    pub(crate) fn product(f: &ImplicitLevel, g: &ImplicitLevel, stride: (usize, usize), grouped: bool) -> Option<ImplicitLevel> {
        let (sl, sr) = (stride.0 as i64, stride.1 as i64);
        let scaled = |d: &Digit, node: i64| Digit { radix: d.radix, left: d.left * sl, right: d.right * sr, node: d.node * node };
        let (fw, fn_) = f.digits.split_at(f.within);
        let (gw, gn) = g.digits.split_at(g.within);
        let mut digits = Vec::with_capacity(f.digits.len() + g.digits.len());
        if grouped {
            let (rf, rg) = (f.runs()?, g.runs()?);
            digits.extend_from_slice(&gw[..rg]);
            digits.extend(fw[..rf].iter().map(|d| scaled(d, 0)));
            digits.extend_from_slice(&gw[rg..]);
            digits.extend(fw[rf..].iter().map(|d| scaled(d, 0)));
        } else {
            digits.extend_from_slice(gw);
            digits.extend(fw.iter().map(|d| scaled(d, 0)));
        }
        digits.extend_from_slice(gn);
        digits.extend(fn_.iter().map(|d| scaled(d, g.nodes as i64)));
        Some(ImplicitLevel {
            nodes: f.nodes * g.nodes,
            per_node: f.per_node * g.per_node,
            first: (f.first.0 * sl + g.first.0, f.first.1 * sr + g.first.1),
            digits,
            within: f.within + g.within,
        })
    }

    /// How many of the place digits, from the fastest, number a run of a
    /// node's pairs that share a left slot, where every run, as the row
    /// loop finds them, is such a block: the leading digits that leave the
    /// left slot alone, with the slot moving at every step past them.
    /// `None` when a step past them leaves the slot where it was.
    fn runs(&self) -> Option<usize> {
        let place = &self.digits[..self.within];
        let r = place.iter().take_while(|d| d.left == 0).count();
        // Stepping digit `j` resets the digits before it: the left slot moves
        // by its step less what those had added.
        let mut back = 0i64;
        for d in &place[r..] {
            if d.left - back == 0 {
                return None;
            }
            back += (d.radix as i64 - 1) * d.left;
        }
        Some(r)
    }

    /// Whether the child level on `side`, of `width` nodes, holds no twins
    /// under this level, read off the digits.
    ///
    /// Let `A` be the digits that move the slot on `side` and `B` the others.
    /// When the slot is a one-to-one function of the digits in `A`, and those
    /// digits take `width` values in all, every node of the child is named,
    /// and by the pairs of exactly one setting `a` of the digits in `A`, with
    /// the digits in `B` free. The contexts of the node, the pairs' nodes and
    /// slots on the other side, are then one set `S`, over the settings of
    /// `B`, shifted by what `a` adds: two nodes are twins only when their
    /// settings add the same, since a finite set shifted by a nonzero amount
    /// is another set. So when what the digits in `A` add to the node and the
    /// other side's slot is also one-to-one, the child has no twins. `false`
    /// says only that the digits do not show it.
    pub(crate) fn twin_free(&self, side: ChildSide, width: usize) -> bool {
        type Slot = fn(&Digit) -> i64;
        let (this, other): (Slot, Slot) = match side {
            ChildSide::Left => (|d| d.left, |d| d.right),
            ChildSide::Right => (|d| d.right, |d| d.left),
        };
        let moved: Vec<&Digit> = self.digits.iter().filter(|d| this(d) != 0).collect();
        if moved.iter().map(|d| d.radix).product::<usize>() != width {
            return false;
        }
        // The node and the other side's slot packed into one number: the
        // node's index times one more than the other side's whole range.
        let span = self.digits.iter().map(|d| (d.radix as i128 - 1) * i128::from(other(d)).abs()).sum::<i128>() + 1;
        let place = |step: i128, d: &Digit| (step.unsigned_abs(), (d.radix - 1) as u128 * step.unsigned_abs());
        one_to_one(moved.iter().map(|d| place(i128::from(this(d)), d)))
            && one_to_one(moved.iter().map(|d| place(i128::from(d.node) * span + i128::from(other(d)), d)))
    }
}

/// The child slots of a pair, as raw words.
#[inline]
fn slots(p: &ChildPair) -> (i64, i64) {
    (i64::from(p.left.raw()), i64::from(p.right.raw()))
}

/// The pair of two child slots.
#[inline]
fn pair(l: i64, r: i64) -> ChildPair {
    debug_assert!((0..=i64::from(u32::MAX)).contains(&l) && (0..=i64::from(u32::MAX)).contains(&r));
    ChildPair::new(EncodedChildRef::from_raw(l as u32), EncodedChildRef::from_raw(r as u32))
}

/// Calls `f` at every place of `digits`, counted like an odometer from the
/// fastest, with the slots `at` plus what the place adds.
#[inline]
fn each_place(digits: &[Digit], at: (i64, i64), mut f: impl FnMut(i64, i64)) {
    let Some((inner, outer)) = digits.split_first() else {
        f(at.0, at.1);
        return;
    };
    let mut count = vec![0usize; outer.len()];
    let mut at = at;
    loop {
        let (mut l, mut r) = at;
        for _ in 0..inner.radix {
            f(l, r);
            l += inner.left;
            r += inner.right;
        }
        let mut j = 0;
        loop {
            let Some(d) = outer.get(j) else { return };
            count[j] += 1;
            if count[j] < d.radix {
                at = (at.0 + d.left, at.1 + d.right);
                break;
            }
            count[j] = 0;
            let back = (d.radix - 1) as i64;
            at = (at.0 - back * d.left, at.1 - back * d.right);
            j += 1;
        }
    }
}

/// The digits of `value` over `0..n`, if it is affine in some mixed radix:
/// each digit's step read off the first place past the digits before it,
/// its radix the longest run of that step, cut to a divisor of what the
/// digits before it leave. `None` where a place has no value or no radix
/// fits; the caller checks every place against the digits found.
fn read_digits(n: usize, mut value: impl FnMut(usize) -> Option<(i64, i64)>) -> Option<Vec<(usize, (i64, i64))>> {
    let v0 = value(0)?;
    let mut digits = Vec::new();
    let mut period = 1usize;
    while period < n {
        if !n.is_multiple_of(period) {
            return None;
        }
        let rest = n / period;
        let v1 = value(period)?;
        let step = (v1.0 - v0.0, v1.1 - v0.1);
        let mut run = 2usize;
        while run < rest && value(period * run)? == (v0.0 + run as i64 * step.0, v0.1 + run as i64 * step.1) {
            run += 1;
        }
        let mut radix = run.min(rest);
        while radix > 1 && !rest.is_multiple_of(radix) {
            radix -= 1;
        }
        if radix < 2 {
            return None;
        }
        digits.push((radix, step));
        period *= radix;
    }
    Some(digits)
}

/// Whether a value that sums digits, each given by the least gap between
/// two of its places' contributions and their span, is one-to-one: the
/// gaps, from the least, each beyond the spans of the digits before it.
fn one_to_one(digits: impl Iterator<Item = (u128, u128)>) -> bool {
    let mut digits: Vec<(u128, u128)> = digits.collect();
    digits.sort_unstable();
    let mut reach = 0u128;
    for (gap, span) in digits {
        if gap <= reach {
            return false;
        }
        reach += span;
    }
    true
}

/// A level's pair arena: the pairs, or, on an implicit level, the
/// description of the pairs the arena would hold.
///
/// An implicit arena is the one place the two kinds of level meet. Reading
/// it as a vector of pairs ([`Deref`]) writes the pairs once into a copy it
/// keeps beside the description; changing it ([`DerefMut`]) writes them into
/// the arena itself, at the capacity the arena would have had, and drops the
/// description. Either way the reader sees the arena the conjunction would
/// have written, so code that does not know implicit levels reads them
/// correctly; the code that does asks for [`implicit`](Self::implicit)
/// first. [`len`](Self::len) and [`capacity`](Self::capacity) are answered
/// without writing anything. Every write is counted, by the code that asked
/// for it (see [`materialized`]).
///
/// A prune that keeps an implicit level's survivors as a description
/// ([`redescribe`](Self::redescribe)) renumbers them from the start of the
/// arena and leaves its length where it was: the slots past the described
/// pairs stand for those of the nodes it dropped, which a written arena keeps
/// until a sweep reclaims them, so that the length, the capacity and the
/// sweeps are those of the written arena. Nothing reads those slots; written
/// out, they hold copies of the first described pair.
#[derive(Clone, Debug, Default)]
pub(crate) struct PairArena {
    vec: Vec<ChildPair>,
    lazy: Option<Box<Lazy>>,
}

/// What an implicit arena holds in place of its pairs.
#[derive(Clone, Debug)]
struct Lazy {
    described: ImplicitLevel,
    /// The arena's length: the described pairs, then the slots of pairs a
    /// prune dropped.
    len: usize,
    /// The capacity the arena would have.
    capacity: usize,
    /// The pairs, once a reader has asked for them.
    copy: OnceLock<Vec<ChildPair>>,
}

impl PairArena {
    /// The arena's length, written or described.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.lazy {
            None => self.vec.len(),
            Some(l) => l.len,
        }
    }

    /// Whether the arena holds no pairs, written or described.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The arena's capacity: on an implicit arena, the capacity it would
    /// have, which the level pool and the meters read as they would read the
    /// written one's.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        match &self.lazy {
            None => self.vec.capacity(),
            Some(l) => l.capacity,
        }
    }

    /// The description of the pairs, on an implicit arena.
    #[inline]
    pub(crate) fn implicit(&self) -> Option<&ImplicitLevel> {
        self.lazy.as_deref().map(|l| &l.described)
    }

    /// Hold `described`'s pairs as their description, at the capacity the
    /// written arena would have. The arena is empty; its allocation is
    /// dropped.
    pub(crate) fn describe(&mut self, described: ImplicitLevel, capacity: usize) {
        debug_assert!(self.is_empty() && described.per_node >= 2);
        let len = described.arena_len();
        DESCRIBED.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        self.vec = Vec::new();
        self.lazy = Some(Box::new(Lazy { described, len, capacity, copy: OnceLock::new() }));
    }

    /// Hold `described` in place of an implicit arena's description: what a
    /// prune leaves of the level, its nodes renumbered from the start of the
    /// arena. The arena keeps its length and capacity, the slots past the
    /// described pairs standing for those the prune dropped.
    pub(crate) fn redescribe(&mut self, described: ImplicitLevel) {
        let lazy = self.lazy.as_mut().expect("redescribe on a written arena");
        debug_assert!(described.per_node >= 2 && described.arena_len() <= lazy.len);
        REDESCRIBED.fetch_add(described.arena_len() as u64, std::sync::atomic::Ordering::Relaxed);
        lazy.described = described;
        lazy.copy = OnceLock::new();
    }

    /// Shorten the arena to `len`, as [`Vec::truncate`] does. An implicit
    /// arena cut no shorter than its described pairs drops the slots past
    /// them without writing anything; any other is written first.
    #[track_caller]
    pub(crate) fn truncate(&mut self, len: usize) {
        match &mut self.lazy {
            Some(l) if len >= l.described.arena_len() => l.len = l.len.min(len),
            _ => self.deref_mut().truncate(len),
        }
    }

    /// Write an implicit arena's pairs into the arena and drop the
    /// description; any other arena is left as it is.
    #[cold]
    #[track_caller]
    pub(crate) fn materialize(&mut self) {
        let Some(lazy) = self.lazy.take() else { return };
        let Lazy { described, len, capacity, copy } = *lazy;
        let mut vec = Vec::with_capacity(capacity);
        match copy.into_inner() {
            Some(pairs) => vec.extend_from_slice(&pairs),
            None => {
                described.write_pairs(&mut vec);
                count_materialized(Location::caller(), Written::InPlace, vec.len());
                pad(&mut vec, len);
            }
        }
        debug_assert_eq!(vec.capacity(), capacity.max(vec.len()));
        self.vec = vec;
    }

    /// Empty the arena, keeping its capacity, as [`Vec::clear`] does.
    #[inline]
    pub(crate) fn clear(&mut self) {
        if let Some(lazy) = self.lazy.take() {
            self.vec = Vec::with_capacity(lazy.capacity);
        }
        self.vec.clear();
    }

    /// Drop the capacity past the arena's length, as [`Vec::shrink_to_fit`]
    /// does.
    #[inline]
    pub(crate) fn shrink_to_fit(&mut self) {
        match &mut self.lazy {
            None => self.vec.shrink_to_fit(),
            Some(l) => l.capacity = l.len,
        }
    }

    /// A copy of the arena, reserved through `lim` as
    /// [`TddLevel::try_clone_on`] reserves the others: a written arena is
    /// copied at its length, an implicit one keeps its description, its
    /// length as its capacity, and has that length charged.
    pub(crate) fn try_clone_on(&self, lim: &Limits) -> Result<PairArena, OperationError> {
        match &self.lazy {
            None => {
                let mut vec = Vec::new();
                lim.reserve_exact(&mut vec, self.vec.len())?;
                vec.extend_from_slice(&self.vec);
                Ok(PairArena { vec, lazy: None })
            }
            Some(l) => {
                let len = l.len;
                lim.charge_bytes((len as u64).saturating_mul(std::mem::size_of::<ChildPair>() as u64))?;
                Ok(PairArena {
                    vec: Vec::new(),
                    lazy: Some(Box::new(Lazy { described: l.described.clone(), len, capacity: len, copy: OnceLock::new() })),
                })
            }
        }
    }
}

impl From<Vec<ChildPair>> for PairArena {
    #[inline]
    fn from(vec: Vec<ChildPair>) -> Self {
        PairArena { vec, lazy: None }
    }
}

impl Deref for PairArena {
    type Target = Vec<ChildPair>;

    /// The pairs; on an implicit arena, a copy written on first use.
    #[inline]
    #[track_caller]
    fn deref(&self) -> &Vec<ChildPair> {
        match &self.lazy {
            None => &self.vec,
            Some(l) => l.copied(Location::caller()),
        }
    }
}

impl Lazy {
    /// The described pairs written out, padded to the arena's length; kept
    /// out of line so that the readers of written arenas, nearly all of
    /// them, do not carry it.
    #[cold]
    #[inline(never)]
    fn copied(&self, at: &'static Location<'static>) -> &Vec<ChildPair> {
        self.copy.get_or_init(|| {
            let mut v = Vec::with_capacity(self.len);
            self.described.write_pairs(&mut v);
            count_materialized(at, Written::Copy, v.len());
            pad(&mut v, self.len);
            v
        })
    }
}

impl DerefMut for PairArena {
    /// The pairs to change; an implicit arena is written in place first.
    #[inline]
    #[track_caller]
    fn deref_mut(&mut self) -> &mut Vec<ChildPair> {
        if self.lazy.is_some() {
            self.materialize();
        }
        &mut self.vec
    }
}

impl PartialEq for PairArena {
    /// Whether the two arenas hold the same pairs; two implicit arenas with
    /// the same description are compared without writing either.
    fn eq(&self, other: &PairArena) -> bool {
        self.len() == other.len()
            && match (self.implicit(), other.implicit()) {
                (Some(a), Some(b)) if a == b => true,
                _ => **self == **other,
            }
    }
}

impl Charged for PairArena {
    #[inline]
    fn charged_bytes(&self) -> u64 {
        (self.capacity() as u64).saturating_mul(std::mem::size_of::<ChildPair>() as u64)
    }
}

impl crate::execution::pool::Scratch for PairArena {
    #[inline]
    fn release(&mut self) {
        *self = PairArena::default();
    }
}

/// Fill `pairs`, an implicit arena's described pairs written out, to the
/// arena's length `len` with copies of the first: the slots of the pairs a
/// prune dropped, which nothing reads.
fn pad(pairs: &mut Vec<ChildPair>, len: usize) {
    if let Some(&first) = pairs.first() {
        pairs.resize(len, first);
    }
}

/// Whether an implicit arena's pairs were written as a copy for a reader or
/// in place for a writer.
#[derive(Clone, Copy)]
enum Written {
    Copy,
    InPlace,
}

/// The pairs arenas have held as their description, over the process.
static DESCRIBED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pairs a prune has kept as the description of what it left of an
/// implicit level, over the process.
static REDESCRIBED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pairs implicit arenas have had written, by the code that asked.
static MATERIALIZED: Mutex<Vec<Materialized>> = Mutex::new(Vec::new());

/// The pairs of implicit levels written out for code that reads or changes a
/// level's pair arena without knowing its description, summed over the
/// process by the place in the source that asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Materialized {
    /// The code that asked for the pairs.
    pub reader: &'static Location<'static>,
    /// Pairs written as a copy beside the description, for a reader.
    pub copied: u64,
    /// Pairs written into the arena, for a writer.
    pub in_place: u64,
}

fn count_materialized(at: &'static Location<'static>, how: Written, pairs: usize) {
    let mut all = MATERIALIZED.lock().unwrap_or_else(|e| e.into_inner());
    let i = match all.iter().position(|m| m.reader == at) {
        Some(i) => i,
        None => {
            all.push(Materialized { reader: at, copied: 0, in_place: 0 });
            all.len() - 1
        }
    };
    match how {
        Written::Copy => all[i].copied += pairs as u64,
        Written::InPlace => all[i].in_place += pairs as u64,
    }
}

/// Every place in the source that has had the pairs of an implicit level
/// written out, with how many, in the order they first asked. An implicit
/// level holds the description of its pairs instead of the pairs; code that
/// reads its arena without asking for the description gets them written. The
/// count is for the whole process.
pub fn materialized() -> Vec<Materialized> {
    MATERIALIZED.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The pairs the conjunction has held as the description of implicit
/// levels instead of writing them, over the whole process.
pub fn described() -> u64 {
    DESCRIBED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The pairs prunes have kept as the description of what they left of
/// implicit levels instead of writing them, over the whole process. A level
/// a prune shrinks, or whose children it renumbers, stays implicit when what
/// is left is affine in a mixed radix.
pub fn redescribed() -> u64 {
    REDESCRIBED.load(std::sync::atomic::Ordering::Relaxed)
}

impl TddLevel {
    /// The description of the level's pairs, when the level is implicit and
    /// its nodes are still the ones the description numbers: node `i` holds
    /// pairs `i · k .. (i + 1) · k` of the arena, `k` the pairs of every
    /// node. `None` on any other level.
    ///
    /// O(nodes): the nodes are checked against the description.
    pub fn implicit(&self) -> Option<&ImplicitLevel> {
        let d = self.pairs.implicit()?;
        let k = d.per_node;
        (self.nodes.len() == d.nodes
            && self.nodes.iter().enumerate().all(|(i, n)| self.arena_range(n.kind()) == Some(i * k..(i + 1) * k)))
        .then_some(d)
    }

    /// The pairs of node `i`, without writing an implicit level's arena: a
    /// slice of the arena, or of `buf`, which they are written into. The
    /// walks that read every pair of a level once go through here, so that a
    /// level's description does not have to be written out for them. Not
    /// valid on a marginal level.
    #[inline]
    pub fn pairs_read<'a>(&'a self, i: usize, buf: &'a mut Vec<ChildPair>) -> &'a [ChildPair] {
        let node = &self.nodes[i];
        if let Some(d) = self.pairs.implicit()
            && let Some(range) = self.arena_range(node.kind())
        {
            buf.clear();
            d.write_range(range, buf);
            return buf;
        }
        self.pairs_of(node)
    }

    /// Write an implicit level's pairs into its arena, leaving it an ordinary
    /// level with the arena the conjunction that made it would have written.
    #[track_caller]
    pub(crate) fn materialize(&mut self) {
        self.pairs.materialize();
    }
}

#[cfg(test)]
#[path = "tests/implicit.rs"]
mod tests;
