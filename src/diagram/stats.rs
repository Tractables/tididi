//! Whole-level-array quantities a diagram carries with it, kept current
//! incrementally instead of re-derived by a sweep per read.
//!
//! Three quantities, all of them properties of the diagram and all of them
//! `O(levels)` to derive from scratch:
//!
//! * [`Tdd::max_width`], the widest level counting live nodes only;
//! * the widest internal level counting tombstoned slots too, which is what
//!   decides an apply's grid layout and cannot be recovered from the first;
//! * the levels that have a marginal child, which is the seed set the
//!   spine-bounded merge grows its rebuild closure from.
//!
//! A caller that merges a small diagram into a large one needs all three
//! before every merge. Sweeping a multi-million-node level array once per
//! merge costs more than the merge, so the diagram keeps them.
//!
//! # The upper-bound discipline
//!
//! The two maxima are **upper bounds**, never claimed exact values, and each
//! one carries the level it was last read at. Every operation that can widen a
//! level folds that level's current widths in ([`Tdd::observe_level`], reached
//! from [`Tdd::invalidate`] and from the marginal fold); every other pass over
//! a diagram only ever narrows one. So a recorded maximum is exact the moment
//! the level it names still holds the recorded value — nothing else can have
//! reached above it — and a read that finds otherwise falls back to one sweep,
//! which happens only when the widest level itself shrank.
//!
//! A stale-high maximum is never returned. Both readers change behaviour on
//! one, so approximating here would change the diagram a merge produces.
//!
//! The marginal-parent set is not an approximation in the same sense: it must
//! never be short, since a level missing from it is a level the spine-bounded
//! merge would carry through unrebuilt, which is a wrong diagram rather than a
//! slower one. It is a pure accumulation — a level never goes back from
//! marginal to structural — so the fold that mints marginal levels is the one
//! place it moves.

use super::level::TddLevel;
use super::tdd::Tdd;
use crate::vtree::VtreeIdx;

/// Slot sentinel for "this level is not in `parents`".
const NOT_A_PARENT: u32 = u32::MAX;

/// The cache itself. Cloned with the diagram, since it describes the diagram.
#[derive(Clone, Debug)]
pub(crate) struct LevelStats {
    /// Every level that is the parent of a marginal level. May name a level
    /// that has since gone marginal itself only between the two halves of
    /// [`Tdd::note_marginalized`]; every reader re-filters regardless.
    parents: Vec<VtreeIdx>,
    /// `slot[t.idx()]` is `t`'s index in `parents`, or [`NOT_A_PARENT`]. Grown
    /// on first insertion, so a diagram that never marginalizes carries no
    /// allocation for it.
    slot: Vec<u32>,
    /// Widest level counting live nodes only.
    max_live: usize,
    /// The level `max_live` was read at.
    live_at: VtreeIdx,
    /// Widest internal level counting tombstoned slots too.
    max_raw: usize,
    /// The level `max_raw` was read at.
    raw_at: VtreeIdx,
    /// False until a sweep has run. A diagram fresh out of an apply starts
    /// here: the operation that built it knows nothing about which levels of
    /// which operand ended up widest.
    valid: bool,
}

impl LevelStats {
    /// A cache that knows nothing, so the first read sweeps.
    pub(crate) fn unknown() -> Self {
        LevelStats {
            parents: Vec::new(),
            slot: Vec::new(),
            max_live: 0,
            live_at: VtreeIdx(0),
            max_raw: 0,
            raw_at: VtreeIdx(0),
            valid: false,
        }
    }

    /// Drop everything the cache claims: the next read sweeps.
    pub(crate) fn forget(&mut self) {
        self.valid = false;
    }

    fn insert_parent(&mut self, p: VtreeIdx, num_nodes: usize) {
        if self.slot.len() < num_nodes {
            self.slot.resize(num_nodes, NOT_A_PARENT);
        }
        if self.slot[p.idx()] != NOT_A_PARENT {
            return;
        }
        self.slot[p.idx()] = self.parents.len() as u32;
        self.parents.push(p);
    }

    fn remove_parent(&mut self, p: VtreeIdx) {
        let Some(&at) = self.slot.get(p.idx()) else { return };
        if at == NOT_A_PARENT {
            return;
        }
        self.slot[p.idx()] = NOT_A_PARENT;
        let moved = *self.parents.last().expect("a slotted parent is in `parents`");
        self.parents.swap_remove(at as usize);
        if moved != p {
            self.slot[moved.idx()] = at;
        }
    }

    fn clear_parents(&mut self, num_nodes: usize) {
        self.parents.clear();
        self.slot.clear();
        self.slot.resize(num_nodes, NOT_A_PARENT);
    }
}

/// The two width maxima a spine-bounded merge rebuilt, taken over the levels
/// it rebuilt.
///
/// Those are the only levels such a merge can have widened — every other level
/// rides through byte for byte — so folding these two numbers into the merged
/// diagram's cache stands in for a sweep of the whole level array.
///
/// The two maxima are different quantities and both are needed: `live` counts
/// live nodes only and is what [`Tdd::max_width`] reports; `raw_internal`
/// counts tombstoned slots too and is taken over internal levels only, which
/// is the quantity that decides the apply's grid layout.
///
/// `live_at` and `raw_at` name the level each maximum was read at, without
/// which a later pass cannot tell whether a recorded maximum still stands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RebuiltWidths {
    pub(crate) live: usize,
    pub(crate) live_at: VtreeIdx,
    pub(crate) raw_internal: usize,
    pub(crate) raw_at: VtreeIdx,
}

impl RebuiltWidths {
    /// Read both maxima off `rebuild` in the merged diagram. `rebuild` is
    /// internal-only and non-empty for every call that reaches here (an empty
    /// spine declines), so the `at` fields always name a level that was walked.
    pub(crate) fn over(merged: &Tdd, rebuild: &[VtreeIdx]) -> Self {
        let mut m = RebuiltWidths {
            live: 0,
            live_at: merged.output().vtree,
            raw_internal: 0,
            raw_at: merged.output().vtree,
        };
        for &t in rebuild {
            let l = &merged.levels()[t.idx()];
            let lw = l.live_width();
            if lw > m.live {
                m.live = lw;
                m.live_at = t;
            }
            let w = l.width();
            if w > m.raw_internal {
                m.raw_internal = w;
                m.raw_at = t;
            }
        }
        m
    }
}

impl Tdd {
    /// Both width maxima: `(max_width(), widest internal level's width())`,
    /// exact, sweeping first if the cache cannot vouch for them.
    ///
    /// A caller that then reads [`marginal_parents`](Self::marginal_parents)
    /// must come through here first — that is what makes the parent set
    /// complete.
    pub(crate) fn widths(&mut self) -> (usize, usize) {
        let stale = !self.stats.valid
            || self.levels[self.stats.live_at.idx()].live_width() != self.stats.max_live
            || self.levels[self.stats.raw_at.idx()].width() != self.stats.max_raw;
        if stale {
            self.recompute_stats();
        }
        debug_assert_eq!(
            (self.stats.max_live, self.stats.max_raw),
            self.sweep_widths(),
            "the diagram's cached width maxima disagree with its levels"
        );
        (self.stats.max_live, self.stats.max_raw)
    }

    /// Every level of this diagram that has a marginal child.
    ///
    /// Complete only once [`widths`](Self::widths) has run, which is what
    /// sweeps a cache that never had one.
    pub(crate) fn marginal_parents(&self) -> &[VtreeIdx] {
        debug_assert!(
            self.stats.valid,
            "the marginal-parent set was read before the diagram had swept for one"
        );
        &self.stats.parents
    }

    /// Fold one level's current widths into the running maxima.
    ///
    /// Raising a bound is all this can do, so a call that catches a level
    /// mid-rewrite is harmless: the value it reads is either the level's own
    /// or one the level held a moment ago, and a maximum recorded at a level
    /// that later narrows is caught by the read's own check.
    #[inline]
    pub(crate) fn observe_level(&mut self, t: VtreeIdx) {
        let l = &self.levels[t.idx()];
        let lw = l.live_width();
        if lw > self.stats.max_live {
            self.stats.max_live = lw;
            self.stats.live_at = t;
        }
        if !self.vtree.node(t).is_leaf() {
            let w = l.width();
            if w > self.stats.max_raw {
                self.stats.max_raw = w;
                self.stats.raw_at = t;
            }
        }
    }

    /// Fold in the maxima a spine-bounded merge re-read over the levels it
    /// rebuilt — the only levels that merge can have widened.
    pub(crate) fn note_grown(&mut self, m: RebuiltWidths) {
        if m.live > self.stats.max_live {
            self.stats.max_live = m.live;
            self.stats.live_at = m.live_at;
        }
        if m.raw_internal > self.stats.max_raw {
            self.stats.max_raw = m.raw_internal;
            self.stats.raw_at = m.raw_at;
        }
    }

    /// Record that the marginal fold has run over `levels`.
    ///
    /// Making a level marginal makes its parent a boundary parent and stops
    /// the level itself from being one. It rewrites the level and its parent —
    /// the passes that follow only narrow, wherever they reach — so those are
    /// also the levels whose width may have moved up.
    ///
    /// Marginality is read off the diagram rather than assumed of `levels`: a
    /// fold is free to skip a target it finds nothing to do at, and a level
    /// wrongly dropped from the parent set is a wrong diagram later.
    pub(crate) fn note_marginalized(&mut self, levels: &[VtreeIdx]) {
        let n = self.vtree.num_nodes();
        for &d in levels {
            if self.levels[d.idx()].is_marginal() {
                self.stats.remove_parent(d);
            }
        }
        for &d in levels {
            self.observe_level(d);
            let Some(p) = self.vtree.node(d).parent() else { continue };
            self.observe_level(p);
            if self.levels[d.idx()].is_marginal() && !self.levels[p.idx()].is_marginal() {
                self.stats.insert_parent(p, n);
            }
        }
    }

    /// The one full sweep: re-derive all three quantities from the levels.
    fn recompute_stats(&mut self) {
        let n = self.vtree.num_nodes();
        self.stats.clear_parents(n);
        self.stats.max_live = 0;
        self.stats.live_at = self.vtree.root();
        self.stats.max_raw = 0;
        self.stats.raw_at = self.vtree.root();
        for i in 0..n {
            let t = VtreeIdx(i as u32);
            self.observe_level(t);
            if !self.levels[i].is_marginal() {
                continue;
            }
            let Some(p) = self.vtree.node(t).parent() else { continue };
            if !self.levels[p.idx()].is_marginal() {
                self.stats.insert_parent(p, n);
            }
        }
        self.stats.valid = true;
    }

    /// Both maxima, swept: what [`max_width`](Self::max_width) falls back to
    /// when the cache cannot vouch for its value, and the debug cross-check of
    /// the incremental values.
    pub(super) fn sweep_widths(&self) -> (usize, usize) {
        let live = self.levels.iter().map(TddLevel::live_width).max().unwrap_or(0);
        let raw = self
            .levels
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.vtree.node(VtreeIdx(*i as u32)).is_leaf())
            .map(|(_, l)| l.width())
            .max()
            .unwrap_or(0);
        (live, raw)
    }

    /// The cached live-node maximum if the cache can vouch for it.
    ///
    /// The read side of [`max_width`](Self::max_width), which sweeps when this
    /// comes back `None`. Nothing is written back: the caller holds a shared
    /// reference, and a sweep it pays for once is cheaper than the mutable
    /// access that would let it be recorded.
    #[inline]
    pub(crate) fn cached_max_width(&self) -> Option<usize> {
        let s = &self.stats;
        (s.valid && self.levels[s.live_at.idx()].live_width() == s.max_live).then_some(s.max_live)
    }

    /// Move the cache out of this diagram, leaving it knowing nothing.
    ///
    /// For an operation that consumes a diagram and produces one whose levels
    /// are largely the same array: the maxima and the parent set it carried
    /// still bound the result, so re-deriving them would be a sweep for
    /// nothing.
    pub(crate) fn take_stats(&mut self) -> LevelStats {
        std::mem::replace(&mut self.stats, LevelStats::unknown())
    }

    /// Install a cache taken from the diagram this one was built out of.
    pub(crate) fn install_stats(&mut self, stats: LevelStats) {
        self.stats = stats;
    }

    /// Drop what the cache claims: the next read sweeps. For a rewrite whose
    /// widened level set is not known.
    pub(crate) fn forget_stats(&mut self) {
        self.stats.forget();
    }
}
