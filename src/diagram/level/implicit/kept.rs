//! What a prune keeps of an implicit level, read off its description.

use super::{Digit, ImplicitLevel, one_to_one};

/// What a prune does to the child level on one side of a level it keeps
/// described: keeps it as it is, its slots where they were, or drops some of
/// its nodes and numbers the `n` it keeps from 0 in their order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildKept {
    /// The child level kept every node; the slots stay.
    Whole,
    /// The child level kept `n` nodes, renumbered in their order. Only for
    /// a child whose slots are node indices: a structural child, not a
    /// marginal one.
    Renumbered(usize),
}

impl ImplicitLevel {
    /// The description of what a prune leaves of this level, derived from
    /// this one without reading a pair, when the prune keeps a box of it:
    /// the `kept` nodes `own` marks (a word a node block, bit `i` node `i`)
    /// are every setting of the node digits within a range of each, and on
    /// each side the child level is kept whole or renumbered onto exactly
    /// the slots the box's pairs name, in an order the digits that move
    /// that side's slot count lexicographically. Such a prune drops the
    /// settings outside the ranges and renumbers each side's moving digits
    /// as a mixed radix of their ranges, from the least step; a step of
    /// either sign is fine, a pair's slot being its rank among the box's.
    ///
    /// `None` when the marks are not a box, or a side's moving digits name
    /// a slot twice, or the renumbered child kept more nodes than the box's
    /// pairs name (a node another reference keeps): the caller then reads
    /// what is left pair by pair. The result is in normal form, the greedy
    /// read of the pairs it describes, as one fitted to the pairs is. Reads
    /// the marks of the box's runs of consecutive nodes, a word at a time
    /// within a run, and about the sum of the radices to normalize.
    pub(crate) fn kept_box(&self, own: &[u64], kept: usize, left: ChildKept, right: ChildKept) -> Option<ImplicitLevel> {
        debug_assert!(self.counts_nodes());
        let digits = &self.digits[self.within..];
        let (lo, hi) = (first_marked(own)?, last_marked(own)?);
        if hi >= self.nodes {
            return None;
        }
        // Each node digit's least setting in the box and its settings there,
        // read off the box's first and last nodes; the box is the marks
        // when it holds `kept` nodes and every one of them is marked.
        let mut ranges = Vec::with_capacity(digits.len());
        let mut size = 1usize;
        for d in digits {
            let period = d.node as usize;
            let (a, b) = (lo / period % d.radix, hi / period % d.radix);
            if a > b {
                return None;
            }
            ranges.push((a, b - a + 1));
            size = size.checked_mul(b - a + 1)?;
        }
        if size != kept || !box_marked(own, digits, &ranges) {
            return None;
        }
        // The place digits and the box's node digits, with the radix each
        // keeps; a node digit kept at one setting is no digit.
        let mut kept_digits: Vec<(usize, Digit)> = self.digits[..self.within].iter().map(|d| (d.radix, *d)).collect();
        let mut first = self.first;
        for (d, &(a, c)) in digits.iter().zip(&ranges) {
            first = (first.0 + a as i64 * d.left, first.1 + a as i64 * d.right);
            if c >= 2 {
                kept_digits.push((c, *d));
            }
        }
        let left = moved_side(&kept_digits, first.0, left, |d| d.left)?;
        let right = moved_side(&kept_digits, first.1, right, |d| d.right)?;
        let step = |j: usize| (left.1[j], right.1[j]);
        let within: Vec<(usize, (i64, i64))> = (0..self.within).map(|j| (kept_digits[j].0, step(j))).collect();
        let across: Vec<(usize, (i64, i64))> = (self.within..kept_digits.len()).map(|j| (kept_digits[j].0, step(j))).collect();
        Some(ImplicitLevel::assemble(kept, self.per_node, (left.0, right.0), &within, &across).normal())
    }
}

impl ImplicitLevel {
    /// The description of what a prune leaves of this level, derived from
    /// this one without reading a pair or a mark, whatever `kept` nodes it
    /// keeps, when each side is one a drop of nodes cannot bend: a side
    /// whose slot no node digit moves, kept whole or renumbered onto
    /// exactly the slots a node's pairs name, which every kept node names
    /// alike; or a renumbered side whose slot rises, or falls, with the
    /// pair's position in the level, every digit of the position moving it
    /// past the reach of the faster ones, and whose child kept as many
    /// nodes as the kept nodes hold pairs, which are then the slots they
    /// name, each pair's new slot its position's rank among the kept pairs.
    ///
    /// `None` on any other side: the caller derives what is left of a box
    /// ([`kept_box`](Self::kept_box)) or reads it pair by pair. The result is
    /// in normal form. Reads the digits only.
    pub(crate) fn kept_any(&self, kept: usize, left: ChildKept, right: ChildKept) -> Option<ImplicitLevel> {
        debug_assert!(kept >= 1 && kept <= self.nodes);
        let k = self.per_node;
        let left = self.side_kept(kept, self.first.0, left, |d| d.left)?;
        let right = self.side_kept(kept, self.first.1, right, |d| d.right)?;
        let within: Vec<(usize, (i64, i64))> = (0..self.within).map(|j| (self.digits[j].radix, (left.1[j], right.1[j]))).collect();
        let across: Vec<(usize, (i64, i64))> = if kept >= 2 { vec![(kept, (left.2, right.2))] } else { Vec::new() };
        Some(ImplicitLevel::assemble(kept, k, (left.0, right.0), &within, &across).normal())
    }

    /// What [`kept_any`](Self::kept_any) makes of one side, `step` reading
    /// a digit's step there: its first slot, the step of each place digit,
    /// and the step of the one node digit of the `kept` nodes.
    fn side_kept(&self, kept: usize, first: i64, child: ChildKept, step: impl Fn(&Digit) -> i64) -> Option<(i64, Vec<i64>, i64)> {
        let (place, node) = self.digits.split_at(self.within);
        if node.iter().all(|d| step(d) == 0) {
            let places: Vec<(usize, Digit)> = place.iter().map(|d| (d.radix, *d)).collect();
            let (first, steps) = moved_side(&places, first, child, &step)?;
            return Some((first, steps, 0));
        }
        let ChildKept::Renumbered(n) = child else { return None };
        let k = self.per_node;
        if Some(n) != kept.checked_mul(k) || !monotone(&self.digits, &step) {
            return None;
        }
        // The kept pairs' positions, numbered in their order, counted down
        // where the slot falls.
        let sign = step(&self.digits[0]).signum();
        let mut unit = 1i64;
        let steps = place.iter().map(|d| {
            let s = sign * unit;
            unit *= d.radix as i64;
            s
        }).collect();
        let first = if sign < 0 { (kept * k) as i64 - 1 } else { 0 };
        Some((first, steps, sign * k as i64))
    }
}

/// Whether the slot `step` reads is strictly monotone in the position the
/// digits number, fastest first: every step of one sign, each past the
/// most the faster digits add.
fn monotone(digits: &[Digit], step: impl Fn(&Digit) -> i64) -> bool {
    let Some(sign) = digits.first().map(|d| step(d).signum()).filter(|&s| s != 0) else {
        return false;
    };
    let mut reach = 0u128;
    digits.iter().all(|d| {
        let s = step(d);
        let past = s.signum() == sign && u128::from(s.unsigned_abs()) > reach;
        reach += (d.radix as u128 - 1) * u128::from(s.unsigned_abs());
        past
    })
}

/// One side's first slot and the step each of `digits` takes it by, once
/// the prune keeps the child level there as `child`: as they were when it
/// is kept whole; when it is renumbered, each slot's rank among the slots
/// the digits name, where those are one to one in the digits and as many as
/// the child kept. `digits` are each with the radix the box keeps.
fn moved_side(digits: &[(usize, Digit)], first: i64, child: ChildKept, step: impl Fn(&Digit) -> i64) -> Option<(i64, Vec<i64>)> {
    let ChildKept::Renumbered(n) = child else {
        return Some((first, digits.iter().map(|(_, d)| step(d)).collect()));
    };
    let moving = || digits.iter().enumerate().filter(|(_, (_, d))| step(d) != 0);
    let named = moving().try_fold(1usize, |p, (_, &(c, _))| p.checked_mul(c))?;
    let place = |&(c, d): &(usize, Digit)| (step(&d).unsigned_abs() as u128, (c - 1) as u128 * step(&d).unsigned_abs() as u128);
    if named != n || !one_to_one(moving().map(|(_, cd)| place(cd))) {
        return None;
    }
    // From the least step up, each digit counts the slots named by the
    // digits before it; a falling one counts down from its top setting.
    let mut order: Vec<usize> = moving().map(|(j, _)| j).collect();
    order.sort_unstable_by_key(|&j| step(&digits[j].1).unsigned_abs());
    let mut steps = vec![0i64; digits.len()];
    let (mut unit, mut first) = (1i64, 0i64);
    for j in order {
        let (c, d) = digits[j];
        let s = if step(&d) > 0 { unit } else { -unit };
        if s < 0 {
            first += (c as i64 - 1) * unit;
        }
        steps[j] = s;
        unit *= c as i64;
    }
    Some((first, steps))
}

/// The first marked slot of `own`.
fn first_marked(own: &[u64]) -> Option<usize> {
    own.iter().position(|&w| w != 0).map(|w| (w << 6) + own[w].trailing_zeros() as usize)
}

/// The last marked slot of `own`.
fn last_marked(own: &[u64]) -> Option<usize> {
    own.iter().rposition(|&w| w != 0).map(|w| (w << 6) + 63 - own[w].leading_zeros() as usize)
}

/// Whether every node of the box `ranges` (each node digit's least setting
/// and its settings) is marked in `own`: read in runs of consecutive nodes,
/// those of the fastest digits the box takes whole and of the first it
/// does not, a run a word of marks at a time.
fn box_marked(own: &[u64], digits: &[Digit], ranges: &[(usize, usize)]) -> bool {
    let Some(u) = ranges.iter().zip(digits).position(|(&(_, c), d)| c < d.radix) else {
        // The box is every node.
        return true;
    };
    let len = digits[u].node as usize * ranges[u].1;
    let start: usize = ranges.iter().zip(digits).map(|(&(a, _), d)| a * d.node as usize).sum();
    let outer = &digits[u + 1..];
    let mut count = vec![0usize; outer.len()];
    let mut at = start;
    loop {
        if !run_marked(own, at, len) {
            return false;
        }
        let mut j = 0;
        loop {
            let Some(d) = outer.get(j) else { return true };
            count[j] += 1;
            if count[j] < ranges[u + 1 + j].1 {
                at += d.node as usize;
                break;
            }
            at -= (count[j] - 1) * d.node as usize;
            count[j] = 0;
            j += 1;
        }
    }
}

/// Whether slots `from .. from + len` of `own` are all marked.
fn run_marked(own: &[u64], from: usize, len: usize) -> bool {
    let end = from + len;
    let mut at = from;
    while at < end {
        let w = at >> 6;
        let (lo, hi) = (at & 63, (end - (w << 6)).min(64));
        let mask = (!0u64 << lo) & if hi == 64 { !0 } else { (1u64 << hi) - 1 };
        if own[w] & mask != mask {
            return false;
        }
        at = (w + 1) << 6;
    }
    true
}
