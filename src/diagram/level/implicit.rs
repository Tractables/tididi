//! Implicit levels: a level whose pairs are a complete mixed-radix product,
//! kept as the description of its pairs instead of its pair arena.
//!
//! A conjunction whose operands' levels at a vtree node are such products,
//! and whose children's products are complete, writes at that node the
//! product of the two: every node of one operand with every node of the
//! other, each with every pair of the one with every pair of the other. Its
//! pairs are then an affine function of the digits of a mixed radix, and
//! [`ImplicitLevel`] holds that function: the level stores neither its pair
//! arena nor its nodes, which the description implies, and nothing writes
//! them out.

use crate::diagram::primitives::{ChildPair, EncodedChildRef, EncodedNode};
use crate::diagram::ChildSide;
use crate::limits::{Limits, OperationError, Transient};

use super::{LevelState, TddLevel};

mod storage;
mod iter;
mod close;
mod kept;
pub(crate) use close::StoreRoom;
pub(crate) use kept::ChildKept;
pub(crate) use storage::{kept_capacity, PairArena};
pub(crate) use iter::{LevelPairs, NodeCursor, Places};

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
/// nodes have the same `k ≥ 1` pairs each,
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
/// An implicit level stores neither its pairs nor its nodes: node `i` is
/// implied by the description ([`TddLevel::node`]), holding its one pair
/// inline at one pair a node, else the range of pairs `i · k .. (i + 1) ·
/// k`. Past 2^31 pairs, where a range takes the side table, the nodes'
/// words are stored beside the description.
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
///
/// Beside its digits a description keeps what the places of its fastest
/// place digits add to the slots, at most `RUN_PAIRS` of them: a node's
/// pairs are read in runs of these, each run's first pair stepped on from
/// the last run's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplicitLevel {
    nodes: usize,
    per_node: usize,
    first: (i64, i64),
    digits: Vec<Digit>,
    within: usize,
    /// What each place of the first `run_digits` digits adds to the slots,
    /// in their order, as wrapping `u32` sums: a run of a node's pairs.
    run: Vec<(u32, u32)>,
    /// How many of the place digits a run counts: the most, from the
    /// fastest, whose places are at most [`RUN_PAIRS`].
    run_digits: usize,
}

/// The fewest pairs a level holds as their description; a level with fewer
/// is stored.
pub const FLOOR: usize = 64;

/// The most places of a description's fastest place digits it keeps the
/// offsets of: a node's pairs are read in runs of them.
const RUN_PAIRS: usize = 256;

/// The description of no pairs, which an empty [`Places`] refers to.
static NO_PAIRS: ImplicitLevel =
    ImplicitLevel { nodes: 0, per_node: 0, first: (0, 0), digits: Vec::new(), within: 0, run: Vec::new(), run_digits: 0 };

impl ImplicitLevel {
    /// The description of `nodes` nodes of `per_node` pairs whose first pair
    /// is `first`, numbered by `digits`, the first `within` of them the
    /// place digits, with the offsets of a run of its pairs.
    fn new(nodes: usize, per_node: usize, first: (i64, i64), digits: Vec<Digit>, within: usize) -> ImplicitLevel {
        let (run_digits, places) = run_of(&digits[..within]);
        let mut run = Vec::with_capacity(places);
        each_place(&digits[..run_digits], (0, 0), |l, r| run.push((l as u32, r as u32)));
        ImplicitLevel { nodes, per_node, first, digits, within, run, run_digits }
    }

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

    /// Whether the node digits count the node index in a mixed radix, each
    /// a unit of the product of the radices before it, as the numbering of
    /// the pairs makes them: what reading a node off them assumes.
    fn counts_nodes(&self) -> bool {
        self.digits[self.within..].iter().scan(1, |period, d| {
            let counts = *period == d.node;
            *period *= d.radix as i64;
            Some(counts)
        }).all(|counts| counts)
    }

    /// The description of the level with its two sides exchanged, as
    /// [`TddLevel::swap_sides`] exchanges a stored level's: the same nodes,
    /// each pair's slots swapped.
    pub(crate) fn swapped(&self) -> ImplicitLevel {
        let digits = self.digits.iter().map(|d| Digit { left: d.right, right: d.left, ..*d }).collect();
        ImplicitLevel::new(self.nodes, self.per_node, (self.first.1, self.first.0), digits, self.within)
    }

    /// The description in normal form: the digits the greedy read takes off
    /// the pairs this one describes, which are those [`fit`](Self::fit)
    /// reads off the level. Reads the digits only, no pair.
    ///
    /// The greedy read ([`read_digits`]) takes a digit's step off the first
    /// place past the digits before it and runs it as far as the places stay
    /// on its line. Past a digit of these the next place is the next digit's
    /// unit, on the line exactly when that digit's step is the run's places
    /// times the step, and then every setting of it is; so the run ends at a
    /// boundary of these digits, the first whose step is off the line, and
    /// the greedy digits are these with each digit on the line merged into
    /// the one before it ([`merged`]).
    pub(crate) fn normal(&self) -> ImplicitLevel {
        let (within, across) = self.digits.split_at(self.within);
        ImplicitLevel::assemble(self.nodes, self.per_node, self.first, &merged(within), &merged(across))
    }

    /// [`normal`](Self::normal) as the greedy read takes it off the pairs
    /// this describes, read at about the sum of the radices: what the
    /// checkers compare a description with.
    #[cfg(any(test, debug_assertions, feature = "testing"))]
    pub(crate) fn read_normal(&self) -> ImplicitLevel {
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
    #[cfg(any(test, debug_assertions, feature = "testing"))]
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

    /// The length of the pair arena the level stands for: its pairs, or none
    /// when every node has one pair and holds it inline.
    #[inline]
    pub(crate) fn arena_len(&self) -> usize {
        if self.per_node >= 2 { self.pairs() } else { 0 }
    }

    /// Whether a level this describes leaves its nodes implied: every node's
    /// word fits the plain encoding, its one pair inline or its range of
    /// pairs below 2^31. Past that a node's range takes the side table,
    /// and the words are stored.
    #[inline]
    pub(crate) fn implies_nodes(&self) -> bool {
        self.per_node == 1 || self.pairs() < 1 << 31
    }

    /// Whether every slot the digits reach from the first pair, at any
    /// setting of them, is one a stored side can hold: from 0 to below
    /// bit 31.
    fn slots_fit_sides(&self) -> bool {
        let fits = |first: i64, step: fn(&Digit) -> i64| {
            let (mut low, mut high) = (i128::from(first), i128::from(first));
            for d in &self.digits {
                let span = (d.radix as i128 - 1) * i128::from(step(d));
                if span < 0 { low += span } else { high += span }
            }
            low >= 0 && high < 1 << 31
        };
        fits(self.first.0, |d| d.left) && fits(self.first.1, |d| d.right)
    }

    /// The description of `level`, when it is one: every node has the same
    /// number of pairs, a node's pairs are its first shifted by offsets all
    /// nodes share, the offsets are affine in the digits of a pair's place,
    /// and the first pairs affine in the digits of the node's index. Read
    /// and checked in one pass over the level's pairs.
    pub(crate) fn fit(level: &TddLevel) -> Option<ImplicitLevel> {
        Self::fit_or_uneven(level).ok()
    }

    /// [`fit`](Self::fit), or why there is none: `Err(Some(i))` when node
    /// `i` holds a different number of pairs from node 0, `Err(None)` for
    /// any other reason.
    fn fit_or_uneven(level: &TddLevel) -> Result<ImplicitLevel, Option<usize>> {
        if !matches!(level.state, LevelState::Structural(_)) {
            return Err(None);
        }
        let nodes = level.nodes().len();
        if nodes == 0 {
            return Err(None);
        }
        let stored = level.stored().ok_or(None)?;
        let first_pairs = stored.of_idx(0);
        let per_node = first_pairs.len();
        // The counts first: a level whose nodes hold different numbers of
        // pairs fails here, before any digit is read or stepped. Where node 0
        // holds one pair and the arena none, no node holds more, and one of
        // none fails the compare of the words below.
        if per_node == 0 {
            return Err(None);
        }
        if (per_node >= 2 || level.pairs.stored_vec_len() != 0)
            && let Some(i) = level.uneven_node(per_node)
        {
            return Err(Some(i));
        }
        let first = slots(&first_pairs[0]);
        let offset = |m: usize| {
            let (l, r) = slots(&first_pairs[m]);
            Some((l - first.0, r - first.1))
        };
        let within = read_digits(per_node, offset).ok_or(None)?;
        // Node 0's pairs against the digits read off a few of them, before
        // the node digits are read or a description is built: a level that
        // fits none fails here most often.
        if !digits_hold(&within, per_node, offset) {
            return Err(None);
        }
        let across = read_digits(nodes, |i| stored.of_idx(i).first().map(slots)).ok_or(None)?;
        let fitted = ImplicitLevel::assemble(nodes, per_node, first, &within, &across);
        // Node 0's pairs are what the place digits give at every place, so
        // the pairs of the node whose first pair is `at` are node 0's
        // shifted by `at` less node 0's first: read off them, not off a list
        // of the offsets, which takes twice the room of the node's pairs.
        let shifted = |at: (i64, i64)| {
            let by = (at.0 - first.0, at.1 - first.1);
            first_pairs.iter().map(move |p| {
                let (l, r) = slots(p);
                (by.0 + l, by.1 + r)
            })
        };
        // A level of one pair a node, its pairs inline, compares its nodes'
        // words in runs; any other, or one where they differ, reads them
        // node by node.
        if (per_node == 1 && fitted.holds_inline(level.nodes.stored())) || fitted.holds(shifted, |i| Some(stored.of_idx(i).iter().map(slots))) {
            Ok(fitted)
        } else {
            Err(None)
        }
    }

    /// Whether `f` gives distinct slots for the distinct child slots the
    /// level's pairs name on `side`: read at every setting of the digits
    /// that move that side's slot, each moved slot held with the slot it
    /// came from, up to the first two that `f` moves to one, in a table
    /// grown through `lim`.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the table's growth is refused:
    /// it holds up to as many slots as the child level has on that side.
    fn one_to_one_on(&self, lim: &Limits, side: ChildSide, f: impl Fn(i64) -> i64) -> Result<bool, OperationError> {
        let first = match side {
            ChildSide::Left => self.first.0,
            ChildSide::Right => self.first.1,
        };
        let mut from = Transient::new(lim, rustc_hash::FxHashMap::default());
        let (mut one, mut refused) = (true, Ok(()));
        Self::each_on_side(&self.digits, side, first, |s, _| {
            if !one || refused.is_err() {
                return;
            }
            if from.len() == from.capacity() {
                let room = from.len().max(16);
                refused = lim.reserve_map(&mut from, room);
                if refused.is_err() {
                    return;
                }
            }
            one = *from.entry(f(s)).or_insert(s) == s;
        });
        refused.map(|()| one)
    }

    /// Calls `f` with every slot on `side` that one of `digits`, read from
    /// `start`, reaches, and the node index the setting of the node digits
    /// among them gives: every setting of the digits that move the side's
    /// slot, those that leave it alone read at zero only.
    fn each_on_side(digits: &[Digit], side: ChildSide, start: i64, mut f: impl FnMut(i64, usize)) {
        let step = |d: &Digit| Self::step(d, side);
        let moving: Vec<Digit> =
            digits.iter().filter(|d| step(d) != 0).map(|d| Digit { left: step(d), right: d.node, ..*d }).collect();
        each_place(&moving, (start, 0), |s, node| f(s, node as usize));
    }

    /// What a step of digit `d` adds to the child slot on `side`.
    #[inline(always)]
    fn step(d: &Digit, side: ChildSide) -> i64 {
        match side {
            ChildSide::Left => d.left,
            ChildSide::Right => d.right,
        }
    }

    /// The child slots on `side` the pairs of a node add to its first: one
    /// for every setting of the place digits that move the side's slot,
    /// written to `out`, empty, its room reserved through `lim`.
    ///
    /// A node of an implicit level may have billions of pairs, and as many
    /// offsets on a side: the room is reserved before the first is written.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the room is refused.
    pub(crate) fn side_offsets(&self, side: ChildSide, lim: &Limits, out: &mut Vec<i64>) -> Result<(), OperationError> {
        lim.reserve_exact(out, self.side_offset_count(side))?;
        Self::each_on_side(&self.digits[..self.within], side, 0, |s, _| out.push(s));
        Ok(())
    }

    /// How many offsets [`side_offsets`](Self::side_offsets) gives on
    /// `side`: the product of the radices of the place digits that move the
    /// side's slot, at most a node's pairs.
    pub(crate) fn side_offset_count(&self, side: ChildSide) -> usize {
        self.digits[..self.within].iter().filter(|d| Self::step(d, side) != 0).map(|d| d.radix).product()
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
    /// order, in a list reserved through `lim` and charged while it is
    /// held.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the list is refused: it takes
    /// sixteen bytes a pair of a node, which may have billions.
    fn offsets<'l>(&self, lim: &'l Limits) -> Result<Transient<'l, Vec<(i64, i64)>>, OperationError> {
        let mut offsets = Transient::new(lim, Vec::new());
        lim.reserve_exact(&mut offsets, self.per_node)?;
        each_place(&self.digits[..self.within], (0, 0), |l, r| offsets.push((l, r)));
        Ok(offsets)
    }

    /// Whether a node's pairs repeat one: whether two of the offsets every
    /// node adds to its first pair are equal. Where the place digits show
    /// the offsets distinct ([`places_distinct`](Self::places_distinct)),
    /// none are listed; otherwise they are listed, through `lim`, and
    /// sorted.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the list is refused.
    pub(crate) fn repeats_a_pair(&self, lim: &Limits) -> Result<bool, OperationError> {
        if self.places_distinct() {
            return Ok(false);
        }
        let mut offsets = self.offsets(lim)?;
        offsets.sort_unstable();
        Ok(offsets.windows(2).any(|w| w[0] == w[1]))
    }

    /// Whether the place digits show that a node's pairs are distinct: the
    /// two slots of a step packed into one number, the right one's times
    /// one more than twice the left steps' whole range, which is one to one
    /// on the offsets, and the packed steps each beyond what the smaller
    /// ones reach ([`one_to_one`]). `false` says only that the digits do
    /// not show it, and where the packed numbers do not fit.
    fn places_distinct(&self) -> bool {
        let place = &self.digits[..self.within];
        let packed = || -> Option<Vec<(u128, u128)>> {
            let reach = place.iter().try_fold(0i128, |reach, d| {
                reach.checked_add((d.radix as i128 - 1).checked_mul(i128::from(d.left).abs())?)
            })?;
            let scale = reach.checked_mul(2)?.checked_add(1)?;
            place.iter().map(|d| {
                let gap = scale.checked_mul(i128::from(d.right))?.checked_add(i128::from(d.left))?.unsigned_abs();
                Some((gap, gap.checked_mul((d.radix - 1) as u128)?))
            }).collect()
        };
        packed().is_some_and(|steps| one_to_one(steps.into_iter()))
    }

    /// Whether `node(i)` gives the pairs of node `i` of this description,
    /// in their order, for every node: read node by node, up to the first
    /// pair that differs. `slots(at)` gives the description's pairs of the
    /// node whose first pair is `at`, a node's pairs in all.
    fn holds<I, P>(&self, mut slots: impl FnMut((i64, i64)) -> P, mut node: impl FnMut(usize) -> Option<I>) -> bool
    where
        I: ExactSizeIterator<Item = (i64, i64)>,
        P: Iterator<Item = (i64, i64)>,
    {
        let mut cursor = self.cursor();
        (0..self.nodes).all(|i| {
            let at = cursor.first_of(i);
            node(i).is_some_and(|pairs| pairs.len() == self.per_node && pairs.zip(slots(at)).all(|(s, p)| s == p))
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
        ImplicitLevel::new(nodes, per_node, first, digits, within.len())
    }

    /// The description of what a prune leaves of this level, when it is
    /// one: the `nodes` nodes it keeps, the `j`th of them node `kept(j)` of
    /// this description, each with its pairs, whose child slots `left` and
    /// `right` move to where the prune put the children's nodes. `kept` is
    /// read at the few ranks that read the digits, and `in_order` gives
    /// every node kept, in their order, for the check, which steps from
    /// each to the next. Read off the moved pairs and checked at every one
    /// of them, side by side, without writing any
    /// ([`ImplicitLevel::holds_moved`]). `None` when they are not affine in
    /// a mixed radix, or `kept` names no node of this description.
    ///
    /// The offsets of a node's pairs from its first, this description's and
    /// the moved one's, are listed through `lim`, sixteen bytes a pair of a
    /// node each.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when a list is refused.
    pub(crate) fn pruned<K: Iterator<Item = usize>>(
        &self,
        lim: &Limits,
        nodes: usize,
        mut kept: impl FnMut(usize) -> Option<usize>,
        in_order: impl Fn() -> K,
        left: impl Fn(i64) -> i64,
        right: impl Fn(i64) -> i64,
    ) -> Result<Option<ImplicitLevel>, OperationError> {
        let mut cursor = self.cursor();
        let mut node = |j: usize| kept(j).filter(|&i| i < self.nodes).map(|i| cursor.first_of(i));
        let places = self.offsets(lim)?;
        let moved = |at: (i64, i64), p: &(i64, i64)| (left(at.0 + p.0), right(at.1 + p.1));
        let Some(at) = node(0) else { return Ok(None) };
        let first = moved(at, &places[0]);
        let within = read_digits(self.per_node, |m| {
            let (l, r) = moved(at, &places[m]);
            Some((l - first.0, r - first.1))
        });
        let Some(within) = within else { return Ok(None) };
        let Some(across) = read_digits(nodes, |j| Some(moved(node(j)?, &places[0]))) else { return Ok(None) };
        let fitted = ImplicitLevel::assemble(nodes, self.per_node, first, &within, &across);
        // The check reads the nodes kept in turn, the `j`th when asked for
        // rank `j`, which it asks for in order.
        let in_turn = || {
            let (mut cursor, mut kept) = (self.cursor(), in_order());
            move |_: usize| kept.next().filter(|&i| i < self.nodes).map(|i| cursor.first_of(i))
        };
        // Side by side reads the places that move each side; when those are
        // as many as a node's pairs, pair by pair reads fewer.
        let moving = |side| self.side_offset_count(side);
        let holds = if moving(ChildSide::Left) + moving(ChildSide::Right) < self.per_node {
            self.holds_moved(lim, &fitted, ChildSide::Left, &mut in_turn(), &left)?
                && self.holds_moved(lim, &fitted, ChildSide::Right, &mut in_turn(), &right)?
        } else {
            let offsets = fitted.offsets(lim)?;
            let shifted = |at: (i64, i64)| offsets.iter().map(move |p| (at.0 + p.0, at.1 + p.1));
            let mut node = in_turn();
            fitted.holds(shifted, |j| node(j).map(|at| places.iter().map(move |p| moved(at, p))))
        };
        Ok(holds.then_some(fitted))
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
    /// makes `nodes · per_node`. The places that move the side, with their
    /// offsets, and `fitted`'s offsets are listed through `lim`.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when a list is refused: each may
    /// take sixteen bytes a pair of a node.
    fn holds_moved(
        &self,
        lim: &Limits,
        fitted: &ImplicitLevel,
        side: ChildSide,
        node: &mut impl FnMut(usize) -> Option<(i64, i64)>,
        f: &impl Fn(i64) -> i64,
    ) -> Result<bool, OperationError> {
        let of = |v: (i64, i64)| match side {
            ChildSide::Left => v.0,
            ChildSide::Right => v.1,
        };
        let step = |d: &Digit| of((d.left, d.right));
        // Each place with the digits that leave the side alone zeroed, and
        // the place that has them zeroed.
        let mut period = 1usize;
        let mut zeroed: Vec<(usize, usize)> = Vec::new();
        let mut moving: Transient<'_, Vec<(usize, i64)>> = Transient::new(lim, Vec::new());
        lim.reserve_exact(&mut moving, self.side_offset_count(side))?;
        moving.push((0, 0));
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
        let offsets = fitted.offsets(lim)?;
        let projected = |m: usize| zeroed.iter().fold(m, |m, &(p, r)| m - (m / p % r) * p);
        if !zeroed.is_empty() && (0..self.per_node).any(|m| of(offsets[m]) != of(offsets[projected(m)])) {
            return Ok(false);
        }
        let mut cursor = fitted.cursor();
        Ok((0..fitted.nodes).all(|j| {
            let Some(at) = node(j) else { return false };
            let (base, to) = (of(at), of(cursor.first_of(j)));
            moving.iter().all(|&(m, off)| f(base + off) == to + of(offsets[m]))
        }))
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
        let product = ImplicitLevel::new(
            f.nodes * g.nodes,
            f.per_node * g.per_node,
            (f.first.0 * sl + g.first.0, f.first.1 * sr + g.first.1),
            digits,
            f.within + g.within,
        );
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
        let flip = side == ChildSide::Right;
        let this = |d: &Digit| if flip { d.right } else { d.left };
        let other = |d: &Digit| if flip { d.left } else { d.right };
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

/// The node holding inline the pair at `offset` from the slots `at`,
/// wrapping.
#[inline(always)]
fn inline_at(at: (u32, u32), offset: (u32, u32)) -> EncodedNode {
    let (l, r) = (EncodedChildRef::from_raw(at.0.wrapping_add(offset.0)), EncodedChildRef::from_raw(at.1.wrapping_add(offset.1)));
    EncodedNode::inline(ChildPair::new(l, r))
}

/// The most of the fastest of `digits` whose places are at most
/// `RUN_PAIRS`, and their places: a run of them.
fn run_of(digits: &[Digit]) -> (usize, usize) {
    let (mut run_digits, mut places) = (0, 1usize);
    while let Some(d) = digits.get(run_digits)
        && places.saturating_mul(d.radix) <= RUN_PAIRS
    {
        places *= d.radix;
        run_digits += 1;
    }
    (run_digits, places)
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

/// `digits` as radices and steps, fastest first, each one whose step is the
/// places of the digits merged before it times their step merged into them:
/// the digits a greedy read ([`read_digits`]) takes off the places they
/// number ([`ImplicitLevel::normal`]).
fn merged(digits: &[Digit]) -> Vec<(usize, (i64, i64))> {
    let mut out: Vec<(usize, (i64, i64))> = Vec::with_capacity(digits.len());
    for d in digits {
        if let Some((radix, (l, r))) = out.last_mut() {
            let places = i64::try_from(*radix).ok();
            let on_line = |s: i64, t: i64| places.and_then(|p| p.checked_mul(s)) == Some(t);
            if on_line(*l, d.left) && on_line(*r, d.right) {
                *radix *= d.radix;
                continue;
            }
        }
        out.push((d.radix, (d.left, d.right)));
    }
    out
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
        let radix = divisor_at_most(rest, run);
        if radix < 2 {
            return None;
        }
        digits.push((radix, step));
        period *= radix;
    }
    Some(digits)
}

/// The largest divisor of `n` that is at most `m`, both one or more:
/// counted down from `m` where that takes at most `√n` steps, and
/// otherwise read off the divisors up to `√n`, each `d` of which names a
/// second one, `n / d`, at or above it.
fn divisor_at_most(n: usize, m: usize) -> usize {
    if m >= n {
        return n;
    }
    let root = n.isqrt();
    let down = |from: usize| (1..=from).rev().find(|&d| n.is_multiple_of(d)).unwrap_or(1);
    if m <= root {
        return down(m);
    }
    // A divisor `n / d` in `root..=m` has `d` in `n.div_ceil(m)..=root`,
    // and the least such `d` names the greatest; without one, the
    // greatest divisor at most `m` is at most `root`.
    match (n.div_ceil(m)..=root).find(|&d| n.is_multiple_of(d)) {
        Some(d) => n / d,
        None => down(root),
    }
}

/// Whether `value(m)` is, at every place `m` of `0..n`, what the digits
/// `digits` (radices of product `n` and the slots one unit of each adds,
/// fastest first) add at that place: counted like an odometer, up to the
/// first place that differs.
fn digits_hold(digits: &[(usize, (i64, i64))], n: usize, mut value: impl FnMut(usize) -> Option<(i64, i64)>) -> bool {
    let mut count = vec![0usize; digits.len()];
    let mut at = (0i64, 0i64);
    for m in 0..n {
        if value(m) != Some(at) {
            return false;
        }
        for (c, &(radix, (l, r))) in count.iter_mut().zip(digits) {
            *c += 1;
            if *c < radix {
                at = (at.0 + l, at.1 + r);
                break;
            }
            *c = 0;
            let back = (radix - 1) as i64;
            at = (at.0 - back * l, at.1 - back * r);
        }
    }
    true
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

#[cfg(test)]
#[path = "tests/implicit.rs"]
mod tests;
