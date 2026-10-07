//! Implicit levels: a level whose pairs are a complete mixed-radix product,
//! kept as the description of its pairs instead of its pair arena.
//!
//! A conjunction whose operands' levels at a vtree node are such products,
//! and whose children's products are complete, writes at that node the
//! product of the two: every node of one operand with every node of the
//! other, each with every pair of the one with every pair of the other. Its
//! pairs are then an affine function of the digits of a mixed radix, and
//! [`ImplicitLevel`] holds that function: the level's nodes stay stored, its
//! pair arena does not, and nothing writes it out.

use crate::diagram::primitives::{ChildPair, EncodedChildRef, EncodedNode};
use crate::diagram::ChildSide;
use crate::limits::{Charged, Limits, OperationError};

use super::{LevelState, TddLevel};

// A test builds every level stored, none described, as the oracle the
// implicit levels are checked against (`test_helpers::stored_levels`), or
// lowers the floor so that small diagrams hold implicit levels
// (`test_helpers::with_floor`).
#[cfg(test)]
use crate::test_helpers::{floor as floor_here, stored_levels_forced as stored_here};

#[cfg(not(test))]
#[inline(always)]
const fn stored_here() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
const fn floor_here() -> usize {
    FLOOR
}

/// Whether every level is built stored: never, outside a test.
#[inline(always)]
pub(crate) fn stored_levels_forced() -> bool {
    stored_here()
}

/// The fewest pairs a level holds as their description: [`FLOOR`], outside
/// a test.
#[inline(always)]
pub(crate) fn floor() -> usize {
    floor_here()
}

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
/// The slots are the raw words of the pairs' sides.
///
/// # Canonical form
///
/// A level is implicit exactly when it can be: it is structural, its `n`
/// nodes have the same `k ≥ 2` pairs each,
/// `n · k` is at least [`FLOOR`], and its pairs as numbered, pair `m` of
/// node `i` at place `i · k + m`, are affine in a mixed radix. Its
/// description is then the digits a greedy read takes off those pairs,
/// a function of the level as numbered, so two equal levels have equal
/// descriptions and no level that can be implicit is stored. Diagrams are
/// canonical up to the numbering of nodes and the order of pairs, and no
/// numbering is preferred: whether a level is implicit depends on the one
/// it has. Below the floor a description, a box and its digits, costs more
/// than the pairs it stands for.
///
/// The form holds at operation boundaries, not inside an operation. A pass
/// that changes a level builds it stored where it lies; the seating of a
/// diagram and the end of a prune or a full reduction close every level
/// (`TddLevel::close`), describing again what is affine.
///
/// The reductions read implicit levels as they read stored ones: a level
/// the close describes need not be reduced, since a prune-only reduction or
/// the seating of a diagram closes levels no contraction has visited. Twin
/// merging, leaf contraction, pair fusion and duplicate resolution find
/// their work in the description, and the one that changes a level stores
/// it first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplicitLevel {
    nodes: usize,
    per_node: usize,
    first: (i64, i64),
    digits: Vec<Digit>,
    within: usize,
}

/// The fewest pairs a level holds as their description; a level with fewer
/// is stored.
pub const FLOOR: usize = 64;

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

    /// The pairs of node `node`, in their order.
    #[inline]
    pub(crate) fn places(&self, node: usize) -> Places<'_> {
        Places { digits: &self.digits[..self.within], at: self.node_first(node), next: 0, end: self.per_node }
    }

    /// The description of the level with its two sides exchanged, as
    /// [`TddLevel::swap_sides`] exchanges a stored level's: the same nodes,
    /// each pair's slots swapped.
    pub(crate) fn swapped(&self) -> ImplicitLevel {
        let digits = self.digits.iter().map(|d| Digit { left: d.right, right: d.left, ..*d }).collect();
        ImplicitLevel { first: (self.first.1, self.first.0), digits, ..*self }
    }

    /// The description in normal form: the digits the greedy read takes off
    /// the pairs this one describes, which are those [`fit`](Self::fit)
    /// reads off the level. Probes about the sum of the radices; writes no
    /// pairs.
    pub(crate) fn normal(&self) -> ImplicitLevel {
        let (k, first) = (self.per_node, self.first);
        let within = read_digits(k, |m| {
            let (l, r) = self.at(m);
            Some((l - first.0, r - first.1))
        });
        let across = read_digits(self.nodes, |i| Some(self.at(i * k)));
        let (Some(within), Some(across)) = (within, across) else {
            unreachable!("the pairs of a description are affine")
        };
        ImplicitLevel::assemble(self.nodes, k, first, &within, &across)
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
        let stored = level.stored()?;
        let first_pairs = stored.of_idx(0);
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
        let across = read_digits(nodes, |i| stored.of_idx(i).first().map(slots))?;
        let fitted = ImplicitLevel::assemble(nodes, per_node, first, &within, &across);
        fitted.holds(|i| Some(stored.of_idx(i).iter().map(slots))).then_some(fitted)
    }

    /// Whether `f` gives distinct slots for the distinct child slots the
    /// level's pairs name on `side`: read at every setting of the digits
    /// that move that side's slot.
    fn one_to_one_on(&self, side: ChildSide, f: impl Fn(i64) -> i64) -> bool {
        let first = match side {
            ChildSide::Left => self.first.0,
            ChildSide::Right => self.first.1,
        };
        // Each moved slot with the slot it came from.
        let mut from = rustc_hash::FxHashMap::default();
        let mut one = true;
        Self::each_on_side(&self.digits, side, first, |s, _| {
            if one {
                one = *from.entry(f(s)).or_insert(s) == s;
            }
        });
        one
    }

    /// Calls `f` with every slot on `side` that one of `digits`, read from
    /// `start`, reaches, and the node index the setting of the node digits
    /// among them gives: every setting of the digits that move the side's
    /// slot, those that leave it alone read at zero only.
    fn each_on_side(digits: &[Digit], side: ChildSide, start: i64, mut f: impl FnMut(i64, usize)) {
        let step = |d: &Digit| match side {
            ChildSide::Left => d.left,
            ChildSide::Right => d.right,
        };
        let moving: Vec<Digit> =
            digits.iter().filter(|d| step(d) != 0).map(|d| Digit { left: step(d), right: d.node, ..*d }).collect();
        each_place(&moving, (start, 0), |s, node| f(s, node as usize));
    }

    /// The child slots on `side` the pairs of a node add to its first: one
    /// for every setting of the place digits that move the side's slot.
    pub(crate) fn side_offsets(&self, side: ChildSide) -> Vec<i64> {
        let mut out = Vec::new();
        Self::each_on_side(&self.digits[..self.within], side, 0, |s, _| out.push(s));
        out
    }

    /// Calls `f` with the child slot on `side` of the first pair of every
    /// node, once for every setting of the node digits that move the side's
    /// slot, and a node that starts there: every slot on that side the
    /// level's nodes start at.
    pub(crate) fn each_side_first(&self, side: ChildSide, f: impl FnMut(i64, usize)) {
        let start = match side {
            ChildSide::Left => self.first.0,
            ChildSide::Right => self.first.1,
        };
        Self::each_on_side(&self.digits[self.within..], side, start, f);
    }

    /// The child slots every pair of a node adds to its first, in their
    /// order.
    fn offsets(&self) -> Vec<(i64, i64)> {
        let mut offsets = Vec::with_capacity(self.per_node);
        each_place(&self.digits[..self.within], (0, 0), |l, r| offsets.push((l, r)));
        offsets
    }

    /// Whether a node's pairs repeat one: whether two of the offsets every
    /// node adds to its first pair are equal.
    pub(crate) fn repeats_a_pair(&self) -> bool {
        let mut offsets = self.offsets();
        offsets.sort_unstable();
        offsets.windows(2).any(|w| w[0] == w[1])
    }

    /// Whether `node(i)` gives the pairs of node `i` of this description,
    /// in their order, for every node: read node by node, up to the first
    /// pair that differs.
    fn holds<I: Iterator<Item = (i64, i64)>>(&self, mut node: impl FnMut(usize) -> Option<I>) -> bool {
        let places = self.offsets();
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
    /// the moved pairs and checked at every one of them, side by side,
    /// without writing any ([`ImplicitLevel::holds_moved`]). `None` when
    /// they are not affine in a mixed radix, or `kept` names no node of this
    /// description.
    pub(crate) fn pruned(
        &self,
        nodes: usize,
        mut kept: impl FnMut(usize) -> Option<usize>,
        left: impl Fn(i64) -> i64,
        right: impl Fn(i64) -> i64,
    ) -> Option<ImplicitLevel> {
        let mut node = |j: usize| kept(j).filter(|&i| i < self.nodes).map(|i| self.node_first(i));
        let places = self.offsets();
        let moved = |at: (i64, i64), p: &(i64, i64)| (left(at.0 + p.0), right(at.1 + p.1));
        let at = node(0)?;
        let first = moved(at, &places[0]);
        let within = read_digits(self.per_node, |m| {
            let (l, r) = moved(at, &places[m]);
            Some((l - first.0, r - first.1))
        })?;
        let across = read_digits(nodes, |j| Some(moved(node(j)?, &places[0])))?;
        let fitted = ImplicitLevel::assemble(nodes, self.per_node, first, &within, &across);
        // Side by side reads the places that move each side; when those are
        // as many as a node's pairs, pair by pair reads fewer.
        let moving = |side| self.side_offsets(side).len();
        let holds = if moving(ChildSide::Left) + moving(ChildSide::Right) < self.per_node {
            self.holds_moved(&fitted, ChildSide::Left, &mut node, &left)
                && self.holds_moved(&fitted, ChildSide::Right, &mut node, &right)
        } else {
            fitted.holds(|j| node(j).map(|at| places.iter().map(move |p| moved(at, p))))
        };
        holds.then_some(fitted)
    }

    /// Whether `fitted` gives, on `side`, the child slots of the pairs of
    /// this description's nodes `node(j)` names by their first pairs, moved
    /// through `f`: for every node `j` of `fitted` and every place `m`,
    /// `f(first(j) + offset(m))`, the side's slot of `node(j)`'s first pair
    /// and of the place's offset from it.
    ///
    /// A place's offset on `side` depends only on the place digits that move
    /// the side's slot, so it is checked at the places where the others are
    /// zero, at every node, and `fitted`'s offsets are checked, once, to
    /// depend on no other: about `nodes · Π radices` reads of the moving
    /// digits and `per_node` of the offsets, where a pair-by-pair check
    /// makes `nodes · per_node`.
    fn holds_moved(
        &self,
        fitted: &ImplicitLevel,
        side: ChildSide,
        node: &mut impl FnMut(usize) -> Option<(i64, i64)>,
        f: &impl Fn(i64) -> i64,
    ) -> bool {
        let of = |v: (i64, i64)| match side {
            ChildSide::Left => v.0,
            ChildSide::Right => v.1,
        };
        let step = |d: &Digit| of((d.left, d.right));
        // Each place with the digits that leave the side alone zeroed, and
        // the place that has them zeroed.
        let mut period = 1usize;
        let mut zeroed: Vec<(usize, usize)> = Vec::new();
        let mut moving: Vec<(usize, i64)> = vec![(0, 0)];
        for d in &self.digits[..self.within] {
            if step(d) == 0 {
                zeroed.push((period, d.radix));
            } else {
                let at_zero = moving.len();
                for c in 1..d.radix {
                    for t in 0..at_zero {
                        let (m, off) = moving[t];
                        moving.push((m + c * period, off + c as i64 * step(d)));
                    }
                }
            }
            period *= d.radix;
        }
        let offsets = fitted.offsets();
        let projected = |m: usize| zeroed.iter().fold(m, |m, &(p, r)| m - (m / p % r) * p);
        if !zeroed.is_empty() && (0..self.per_node).any(|m| of(offsets[m]) != of(offsets[projected(m)])) {
            return false;
        }
        (0..fitted.nodes).all(|j| {
            let Some(at) = node(j) else { return false };
            let (base, to) = (of(at), of(fitted.node_first(j)));
            moving.iter().all(|&(m, off)| f(base + off) == to + of(offsets[m]))
        })
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
        let product = ImplicitLevel {
            nodes: f.nodes * g.nodes,
            per_node: f.per_node * g.per_node,
            first: (f.first.0 * sl + g.first.0, f.first.1 * sr + g.first.1),
            digits,
            within: f.within + g.within,
        };
        Some(product.normal())
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

/// A level's pair arena: the pairs, or, on an implicit level, the
/// description of the pairs in their place.
///
/// The two are told apart wherever pairs are read or changed:
/// [`stored`](Self::stored) gives a stored arena's pairs and
/// [`implicit`](Self::implicit) an implicit one's description, and nothing
/// writes an implicit arena's pairs out. [`len`](Self::len) and
/// [`capacity`](Self::capacity) are those of the arena the level would have
/// stored, which the meters, the sweeps and the level pool read as they read
/// a stored one's.
///
/// A prune that keeps an implicit level's survivors as a description
/// ([`redescribe`](Self::redescribe)) renumbers them from the start of the
/// arena and leaves its length where it was: the slots past the described
/// pairs stand for those of the nodes it dropped, which a stored arena keeps
/// until a sweep reclaims them, so that the length, the capacity and the
/// sweeps are those of the stored arena. Nothing reads those slots.
///
/// The description is boxed: an arena takes the room of a vector of pairs.
#[derive(Debug)]
pub(crate) enum PairArena {
    Stored(Vec<ChildPair>),
    Described(Box<Described>),
}

impl Default for PairArena {
    #[inline]
    fn default() -> Self {
        PairArena::Stored(Vec::new())
    }
}

/// What an implicit arena holds in place of its pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Described {
    level: ImplicitLevel,
    /// The arena's length: the described pairs, then the slots of pairs a
    /// prune dropped.
    len: usize,
    /// The capacity the arena would have.
    capacity: usize,
}

impl PairArena {
    /// The arena's length, stored or described.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            PairArena::Stored(vec) => vec.len(),
            PairArena::Described(d) => d.len,
        }
    }

    /// Whether the arena holds no pairs, stored or described.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The arena's capacity: on an implicit arena, the capacity it would
    /// have, which the level pool and the meters read as they would read the
    /// stored one's.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        match self {
            PairArena::Stored(vec) => vec.capacity(),
            PairArena::Described(d) => d.capacity,
        }
    }

    /// The description of the pairs, on an implicit arena.
    #[inline]
    pub(crate) fn implicit(&self) -> Option<&ImplicitLevel> {
        match self {
            PairArena::Stored(_) => None,
            PairArena::Described(d) => Some(&d.level),
        }
    }

    /// The pairs of a stored arena; `None` on an implicit one.
    #[inline]
    pub(crate) fn stored(&self) -> Option<&[ChildPair]> {
        match self {
            PairArena::Stored(vec) => Some(vec.as_slice()),
            PairArena::Described(_) => None,
        }
    }

    /// The pairs of a stored arena, to change or extend.
    ///
    /// # Panics
    ///
    /// Panics on an implicit arena, whose pairs are not stored: code that
    /// changes a level's pairs in place takes an implicit level through its
    /// description, and code that builds a level starts from a cleared one.
    #[inline]
    #[track_caller]
    pub(crate) fn stored_mut(&mut self) -> &mut Vec<ChildPair> {
        match self {
            PairArena::Stored(vec) => vec,
            PairArena::Described(_) => panic!("an implicit level's pairs are not stored"),
        }
    }

    /// Hold `described`'s pairs as their description, at the capacity the
    /// stored arena would have. The arena is empty; its allocation is
    /// dropped.
    pub(crate) fn describe(&mut self, described: ImplicitLevel, capacity: usize) {
        debug_assert!(self.is_empty() && described.per_node >= 2);
        let len = described.arena_len();
        DESCRIBED.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        crate::test_helpers::note_described();
        *self = PairArena::Described(Box::new(Described { level: described, len, capacity }));
    }

    /// Hold a stored arena's pairs as their description `described`, keeping
    /// the arena's length and capacity: the slots past the described pairs
    /// stand for the dead slots a sweep would drop.
    pub(crate) fn describe_stored(&mut self, described: ImplicitLevel) {
        debug_assert!(self.stored().is_some() && described.per_node >= 2 && described.arena_len() <= self.len());
        let (len, capacity) = (self.len(), self.capacity());
        DESCRIBED.fetch_add(described.arena_len() as u64, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        crate::test_helpers::note_described();
        *self = PairArena::Described(Box::new(Described { level: described, len, capacity }));
    }

    /// Hold `described` in place of an implicit arena's description: what a
    /// prune leaves of the level, its nodes renumbered from the start of the
    /// arena. The arena keeps its length and capacity, the slots past the
    /// described pairs standing for those the prune dropped.
    pub(crate) fn redescribe(&mut self, described: ImplicitLevel) {
        let PairArena::Described(d) = self else { panic!("redescribe on a stored arena") };
        debug_assert!(described.per_node >= 2 && described.arena_len() <= d.len);
        REDESCRIBED.fetch_add(described.arena_len() as u64, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        crate::test_helpers::note_described();
        d.level = described;
    }

    /// Exchange the two sides of an implicit arena's description.
    pub(crate) fn swap_described_sides(&mut self) {
        let PairArena::Described(d) = self else { panic!("a stored arena swaps its pairs") };
        d.level = d.level.swapped();
    }

    /// Shorten the arena to `len`, as [`Vec::truncate`] does. An implicit
    /// arena drops the slots past its described pairs.
    ///
    /// # Panics
    ///
    /// Panics on an implicit arena cut shorter than its described pairs.
    #[track_caller]
    pub(crate) fn truncate(&mut self, len: usize) {
        match self {
            PairArena::Stored(vec) => vec.truncate(len),
            PairArena::Described(d) => {
                assert!(len >= d.level.arena_len(), "a truncation into an implicit level's pairs");
                d.len = d.len.min(len);
            }
        }
    }

    /// Empty the arena, keeping its capacity, as [`Vec::clear`] does: an
    /// implicit arena becomes an empty stored one of the capacity it would
    /// have had.
    #[inline]
    pub(crate) fn clear(&mut self) {
        match self {
            PairArena::Stored(vec) => vec.clear(),
            PairArena::Described(d) => *self = PairArena::Stored(Vec::with_capacity(d.capacity)),
        }
    }

    /// Drop the capacity past the arena's length, as [`Vec::shrink_to_fit`]
    /// does.
    #[inline]
    pub(crate) fn shrink_to_fit(&mut self) {
        match self {
            PairArena::Stored(vec) => vec.shrink_to_fit(),
            PairArena::Described(d) => d.capacity = d.len,
        }
    }

    /// A copy of the arena, reserved through `lim` as
    /// [`TddLevel::try_clone_on`] reserves the others: a stored arena is
    /// copied at its length, an implicit one keeps its description, its
    /// length as its capacity, and has that length charged.
    pub(crate) fn try_clone_on(&self, lim: &Limits) -> Result<PairArena, OperationError> {
        match self {
            PairArena::Stored(pairs) => {
                let mut vec = Vec::new();
                lim.reserve_exact(&mut vec, pairs.len())?;
                vec.extend_from_slice(pairs);
                Ok(PairArena::Stored(vec))
            }
            PairArena::Described(d) => {
                let len = d.len;
                lim.charge_bytes((len as u64).saturating_mul(std::mem::size_of::<ChildPair>() as u64))?;
                Ok(PairArena::Described(Box::new(Described { level: d.level.clone(), len, capacity: len })))
            }
        }
    }
}

impl Clone for PairArena {
    /// A copy as [`Vec::clone`] makes one, at the arena's length: a stored
    /// arena's pairs, or the description with its length as its capacity.
    fn clone(&self) -> Self {
        match self {
            PairArena::Stored(vec) => PairArena::Stored(vec.clone()),
            PairArena::Described(d) => PairArena::Described(Box::new(Described { level: d.level.clone(), len: d.len, capacity: d.len })),
        }
    }
}

impl From<Vec<ChildPair>> for PairArena {
    #[inline]
    fn from(vec: Vec<ChildPair>) -> Self {
        PairArena::Stored(vec)
    }
}

impl PartialEq for PairArena {
    /// Whether the two arenas are the same: the same stored pairs, or the
    /// same description at the same length. A level is implicit exactly when
    /// its pairs as numbered can be described (see [`ImplicitLevel`]), so a
    /// stored arena and an implicit one never hold the same level's pairs.
    fn eq(&self, other: &PairArena) -> bool {
        match (self, other) {
            (PairArena::Stored(a), PairArena::Stored(b)) => a == b,
            (PairArena::Described(a), PairArena::Described(b)) => a.len == b.len && a.level == b.level,
            _ => false,
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

/// The pairs of one node of an implicit level, in their order, generated
/// from its description: what [`PairsIter`](crate::diagram::PairsIter)
/// yields on such a level.
#[derive(Clone, Debug)]
pub(crate) struct Places<'a> {
    digits: &'a [Digit],
    at: (i64, i64),
    next: usize,
    end: usize,
}

impl Places<'_> {
    /// The pairs still to come.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.end - self.next
    }
}

impl Iterator for Places<'_> {
    type Item = ChildPair;

    #[inline]
    fn next(&mut self) -> Option<ChildPair> {
        if self.next == self.end {
            return None;
        }
        let (mut l, mut r) = self.at;
        let mut rest = self.next;
        for d in self.digits {
            let c = (rest % d.radix) as i64;
            rest /= d.radix;
            l += c * d.left;
            r += c * d.right;
        }
        self.next += 1;
        Some(pair(l, r))
    }

    /// Skips to the place `n` on in one step: a place's pair is its own
    /// sum of digits, independent of the ones before it.
    #[inline]
    fn nth(&mut self, n: usize) -> Option<ChildPair> {
        self.next = self.next.saturating_add(n).min(self.end);
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}


/// The pairs arenas have held as their description, over the process.
static DESCRIBED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pairs a prune has kept as the description of what it left of an
/// implicit level, over the process.
static REDESCRIBED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pairs of implicit levels a renumbering of their children left not
/// affine, stored moved, over the process.
static STORED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pairs of implicit levels that a renumbering of their child levels
/// left not affine as numbered, or that a change rewrote in place, which
/// were stored, over the whole process.
pub fn stored_moved() -> u64 {
    STORED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The pairs the conjunction and the closes at the ends of operations have
/// held as the description of implicit levels instead of storing them, over
/// the whole process.
pub fn described() -> u64 {
    DESCRIBED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The pairs prunes have kept as the description of what they left of
/// implicit levels instead of storing them, over the whole process. A level
/// a prune shrinks, or whose children it renumbers, stays implicit when what
/// is left is affine in a mixed radix.
pub fn redescribed() -> u64 {
    REDESCRIBED.load(std::sync::atomic::Ordering::Relaxed)
}


impl crate::diagram::Tdd {
    /// The end of an operation that may have built levels stored: close every
    /// level ([`TddLevel::close`]), so that the canonical form of implicit
    /// levels holds when the operation returns. A debug build checks it.
    pub(crate) fn close_levels(&mut self) {
        self.levels.close();
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
            self.nodes.len() == d.nodes
                && self.nodes.iter().enumerate().all(|(i, n)| self.arena_range(n.kind()) == Some(i * k..(i + 1) * k))
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
                ChildSide::Left => d.pruned(d.nodes, Some, &f, id),
                ChildSide::Right => d.pruned(d.nodes, Some, id, &f),
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
    /// can be one (see the canonical form of [`ImplicitLevel`]), its nodes at
    /// pairs `i · k .. (i + 1) · k`, the arena keeping its length, capacity
    /// and dead slots, so that the meters and the sweeps read it as they
    /// read the stored one. Nothing on an implicit level or one that cannot
    /// be, nor on an arena past 2^31 pairs, whose nodes' ranges may take the
    /// side table. Reads the pairs up to the first that is not affine, and
    /// charges nothing.
    pub(crate) fn close(&mut self) {
        if self.pairs.implicit().is_some()
            || self.pairs.len() < floor()
            || self.pairs.len() >= 1 << 31
            || stored_levels_forced()
        {
            return;
        }
        let Some(d) = ImplicitLevel::fit(self) else { return };
        if d.per_node < 2 || d.pairs() < floor() {
            return;
        }
        let k = d.per_node;
        for (i, node) in self.nodes.iter_mut().enumerate() {
            *node = EncodedNode::multi_pair((i * k) as u32, k as u32);
        }
        self.pairs.describe_stored(d);
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
        let filler = vec[0];
        vec.resize(self.pairs.len(), filler);
        if sorted {
            for (i, &w) in lens.iter().enumerate() {
                super::sort_pairs(&mut vec[i * k..i * k + w]);
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

    /// Store the pairs of an implicit level's nodes `keep` names, moved
    /// through `left` and `right`, when what a prune or a renumbering of its
    /// child levels leaves of it is not affine as numbered: node `i` at pairs
    /// `i · k .. (i + 1) · k` of an arena of the length and capacity the
    /// implicit one stood for. The slots of the other nodes and those past
    /// the described pairs hold copies of the first pair; nothing reads them.
    pub(crate) fn store_moved(&mut self, keep: impl Fn(usize) -> bool, left: impl Fn(i64) -> i64, right: impl Fn(i64) -> i64) {
        let d = self.pairs.implicit().expect("store_moved on a stored level");
        let (len, capacity) = (self.pairs.len(), self.pairs.capacity());
        let mut vec = Vec::with_capacity(capacity);
        let fill = d.places(0).next().map(|p| pair(left(i64::from(p.left.raw())), right(i64::from(p.right.raw()))));
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

    /// The pairs of node `i`: a slice of a stored level's arena, or of `buf`,
    /// which an implicit level's are generated into. Not valid on a marginal
    /// level.
    #[inline]
    pub fn pairs_read<'a>(&'a self, i: usize, buf: &'a mut Vec<ChildPair>) -> &'a [ChildPair] {
        match self.pair_view() {
            super::Pairs::Stored(s) => s.of_idx(i),
            super::Pairs::Implicit(d) => {
                buf.clear();
                buf.extend(d.places(i));
                buf
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/implicit.rs"]
mod tests;
