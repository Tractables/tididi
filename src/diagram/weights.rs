//! External side-table of per-node weighted marginal values for algebraic
//! model counting.
//!
//! Why a side-table and not a `TddLevel` field: `TddLevel` is at its size budget
//! (a static assert guards it), so a `Vec<WeightVal>` field would overflow it and
//! perturb the hot sequential-scan stride of the integer path. Instead a
//! level is in its weighted state and its values
//! live here, indexed by vtree level. The integer marginalization store
//! (`marginal_counts` / `marginal_counts_big`) is untouched and stays `None` in
//! weighted mode — the two are mutually exclusive within one compile.
//!
//! The weighted cascade reuses the same structural marginalization machinery as
//! the integer path (scheduling, cascade order, parent-ref remap, dedup); only
//! the per-node payload differs — a [`WeightVal`] (exact `BigRational`, or in
//! the log domain a bounded-precision `SignedLog`) instead of a `u128`/`BigUint`
//! model count. Leaf base values and the fold arithmetic come from
//! [`RationalWeights`] (always parsed exactly), converted once per leaf read
//! to the active mode by `WeightStore::leaf_val`.


use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::diagram::{EvalAlgebra, RationalWeights, SignedLog, WeightVal};
use crate::diagram::LeafLabel;
use crate::vtree::VarId;

/// The numeric representation a weighted marginalization's values use: exact
/// `BigRational`, or
/// the bounded-precision `SignedLog` domain, whose every operation is O(1)
/// `f64` work. The two never mix within one store; a caller decides once per
/// weighted run and passes it once to [`WeightStore::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Arithmetic {
    /// Exact rationals.
    ExactRational,
    /// Bounded-precision signed log domain.
    SignedLog,
}

/// Per-level weighted marginal values: an entry for a vtree level exists once
/// that level is weight-marginalized, and `values[slot]` is the value of
/// the node occupying that marginal slot (post-dedup slot index, the same index
/// the level's marginal-side pair refs point at).
///
/// Attach one to a diagram with [`Tdd::set_weights`] to put it in weighted
/// mode. A store holds only the levels that are marginal, and shares its weight
/// table with every store derived from it by [`empty_like`], so a diagram that
/// has marginal nothing carries almost nothing.
///
/// One value domain. The weight table is a [`RationalWeights`] and the
/// arithmetic is one of [`Arithmetic`]'s two modes; nothing here is generic
/// over an algebra. Exact rationals and the bounded log domain are two
/// arithmetics over the same weights, which is why they are an enum rather
/// than two stores.
///
/// [`Tdd::set_weights`]: crate::Tdd::set_weights
/// [`empty_like`]: Self::empty_like
#[derive(Clone)]
pub struct WeightStore {
    per_level: FxHashMap<usize, Vec<WeightVal>>,
    algebra: Arc<RationalWeights>,
    arithmetic: Arithmetic,
}

impl std::fmt::Debug for WeightStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightStore")
            .field("levels", &self.per_level.len())
            .field("arithmetic", &self.arithmetic)
            .finish()
    }
}

impl WeightStore {
    /// A store over `algebra` with no level marginal yet.
    pub fn new(algebra: RationalWeights, arithmetic: Arithmetic) -> Self {
        Self { per_level: FxHashMap::default(), algebra: Arc::new(algebra), arithmetic }
    }

    /// A second empty store over the same weight table, for another diagram of
    /// the same weighted build. The table is shared, not copied.
    pub fn empty_like(&self) -> Self {
        Self {
            per_level: FxHashMap::default(),
            algebra: Arc::clone(&self.algebra),
            arithmetic: self.arithmetic,
        }
    }

    /// The algebra the values are folded in: the weight table and its
    /// operations.
    #[inline]
    pub fn algebra(&self) -> &RationalWeights {
        &self.algebra
    }

    /// The store's arithmetic, fixed at construction.
    #[inline]
    pub fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
    }

    /// Take over every level `other` holds that this store does not.
    ///
    /// Used where two diagrams meet: their marginal levels are the two disjoint
    /// vtree subtrees they were built over, except for leaf columns, which are
    /// a pure function of the weight table and therefore already equal.
    pub fn absorb(&mut self, other: Self) {
        for (level, values) in other.per_level {
            self.per_level.entry(level).or_insert(values);
        }
    }

    /// True in the bounded-precision log domain.
    #[inline]
    pub(crate) fn is_log(&self) -> bool {
        self.arithmetic == Arithmetic::SignedLog
    }

    /// The additive identity in the active mode.
    #[inline]
    pub(crate) fn wzero(&self) -> WeightVal {
        if self.is_log() {
            WeightVal::Log(SignedLog::zero())
        } else {
            // The canonical exact zero: 0 always fits the small representation.
            WeightVal::ExactSmall(0)
        }
    }

    /// Leaf base value for `(var, label)` in the active mode. The exact parsed
    /// weight from [`RationalWeights`] is authoritative; in log mode it is
    /// converted to `SignedLog` exactly once here (per leaf read).
    #[inline]
    pub(crate) fn leaf_val(&self, var: VarId, label: LeafLabel) -> WeightVal {
        let r = self.algebra.leaf(var, label);
        if self.is_log() {
            WeightVal::Log(SignedLog::from_rational(&r))
        } else {
            WeightVal::exact(r)
        }
    }

    /// Set the per-node values of weight-marginal level `level` (one entry per
    /// node, indexed like a marginal level's count table).
    pub fn set_level(&mut self, level: usize, values: Vec<WeightVal>) {
        self.per_level.insert(level, values);
    }

    /// The per-node values of level `level`, or `None` if that level is not
    /// weight-marginal. A weight-marginal level ([`TddLevel::is_weight_marginal`])
    /// keeps its values here rather than in the diagram, so this is how a
    /// traversal reads them; a parent pair's side into such a level decodes
    /// through the level's [`SideView`] to an index into this slice.
    ///
    /// [`TddLevel::is_weight_marginal`]: crate::diagram::TddLevel::is_weight_marginal
    /// [`SideView`]: crate::diagram::SideView
    #[inline]
    pub fn level(&self, level: usize) -> Option<&[WeightVal]> {
        self.per_level.get(&level).map(Vec::as_slice)
    }

    /// Scoped `&mut` into one level's value vec, for the slot-prune boundary
    /// compaction (`WeightFold::compact_store`) and nothing else.
    ///
    /// That pass is the only writer that rewrites a level's values in place
    /// (survivors swapped down into the prefix, then truncated). It cannot use
    /// [`WeightStore::level`] (read-only) and using [`WeightStore::set_level`]
    /// costs exactly what the in-place form exists to avoid: a second
    /// full-length vec of `WeightVal`s live beside the old one at peak, each
    /// value no smaller than a `u128` and usually a multi-limb `BigRational`.
    ///
    /// Deliberately not a general mutation hook — every other writer goes
    /// through `set_level` (replace a level wholesale) or `push_value` (append
    /// one slot, get its index back). Those two disciplines are what the
    /// marginal-side ref walkers assume; an arbitrary in-place edit that moved or
    /// dropped slots without rewriting the parent refs in the same pass would
    /// silently invalidate them.
    #[inline]
    pub(crate) fn level_vals_mut(&mut self, level: usize) -> Option<&mut Vec<WeightVal>> {
        self.per_level.get_mut(&level)
    }

    /// Append `val` as a fresh slot to a weight-marginalized level, returning the
    /// new slot index. Mirrors the integer count mint used by the
    /// G twin-fold (`duplicate_pair_resolve`): no value interning here — slot-prune merges
    /// equal-valued slots on the next prune pass. Panics if the level was not yet
    /// `set_level`'d (a scaled ref into a non-marginalized level is a bug).
    ///
    /// # Panics
    ///
    /// Panics if `level` has no weighted store allocated (it was never
    /// `set_level`'d).
    pub(crate) fn push_value(&mut self, level: usize, val: WeightVal) -> usize {
        let vec = self
            .per_level
            .get_mut(&level)
            .expect("push_value: level has no weighted store");
        let idx = vec.len();
        vec.push(val);
        idx
    }

    /// Remove `level`'s values from this store and hand them over, or `None` if
    /// that level is not weight-marginal. The move a graft needs: a part's
    /// column goes into the merged store under the level's new index, without
    /// duplicating the rationals.
    #[inline]
    pub(crate) fn take_level(&mut self, level: usize) -> Option<Vec<WeightVal>> {
        self.per_level.remove(&level)
    }

    /// True once `set_level` has been called for `level`.
    #[inline]
    pub(crate) fn is_set(&self, level: usize) -> bool {
        self.per_level.contains_key(&level)
    }
}
