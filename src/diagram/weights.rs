//! External side-table of per-node weighted marginal values for algebraic
//! model counting.
//!
//! A weight-marginal level's values live here, indexed by vtree level, rather
//! than in `TddLevel`, whose size a static assert holds down. The integer
//! count store (`marginal_counts` / `marginal_counts_big`) stays `None` in
//! weighted mode. The per-node payload is a [`WeightValue`] (exact `BigRational`
//! or bounded-precision `SignedLog`); leaf base values come from the store's
//! [`RationalWeights`], converted to the active mode by `WeightStore::leaf_val`.


use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::diagram::{EvalAlgebra, RationalWeights, SignedLog, WeightValue};
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
/// table with every store derived from it by [`empty_like`]. The table is a
/// [`RationalWeights`] and the arithmetic one of [`Arithmetic`]'s two modes.
///
/// [`Tdd::set_weights`]: crate::Tdd::set_weights
/// [`empty_like`]: Self::empty_like
#[derive(Clone)]
pub struct WeightStore {
    per_level: FxHashMap<usize, Vec<WeightValue>>,
    config: WeightConfig,
}

/// The immutable interpretation shared by computed columns.
#[derive(Clone, PartialEq, Eq)]
struct WeightConfig {
    algebra: Arc<RationalWeights>,
    arithmetic: Arithmetic,
}

impl std::fmt::Debug for WeightStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightStore")
            .field("levels", &self.per_level.len())
            .field("arithmetic", &self.config.arithmetic)
            .finish()
    }
}

impl WeightStore {
    /// Whether columns from these stores have the same interpretation.
    pub(crate) fn compatible(&self, other: &Self) -> bool {
        self.config == other.config
    }

    /// Check that the table covers the vtree and every marginal level has its values.
    pub(crate) fn check_levels(&self, vtree: &crate::vtree::Vtree, levels: &[crate::diagram::TddLevel]) -> Result<(), crate::diagram::TddBuildError> {
        use crate::diagram::{TddBuildError, LEAF_WIDTH};
        for (leaf, var) in vtree.leaf_bottomup() {
            if var.idx() >= self.algebra().num_vars() {
                return Err(TddBuildError::MissingVariableWeight(var));
            }
            if levels[leaf.idx()].is_weight_marginal() {
                let values = self.level(leaf.idx()).unwrap_or(&[]);
                if levels[leaf.idx()].slot_count() != LEAF_WIDTH || values.len() != LEAF_WIDTH {
                    return Err(TddBuildError::InvalidWeightColumn { level: leaf, reason: "must hold three leaf-label slots" });
                }
                for (slot, value) in values.iter().enumerate() {
                    let expected = self.leaf_val(var, LeafLabel::from_idx(slot));
                    if crate::diagram::semiring::weight_key(value) != crate::diagram::semiring::weight_key(&expected) {
                        return Err(TddBuildError::InvalidWeightColumn { level: leaf, reason: "does not match its pinned leaf values" });
                    }
                }
            }
        }
        for t in vtree.bottomup() {
            let level = &levels[t.idx()];
            if !level.is_marginal() { continue; }
            if !level.is_weight_marginal() {
                return Err(TddBuildError::CountLevelWithWeights { level: t });
            }
            let values = self.level(t.idx()).unwrap_or(&[]);
            if values.len() != level.slot_count() {
                return Err(TddBuildError::InvalidWeightColumn { level: t, reason: "does not match the level's slot count" });
            }
            if values.iter().any(|v| matches!(v, WeightValue::Log(_)) != self.is_log()) {
                return Err(TddBuildError::InvalidWeightColumn { level: t, reason: "uses a different arithmetic" });
            }
        }
        Ok(())
    }

    /// A store over `algebra` with no level marginal yet.
    pub fn new(algebra: RationalWeights, arithmetic: Arithmetic) -> Self {
        Self { per_level: FxHashMap::default(), config: WeightConfig { algebra: Arc::new(algebra), arithmetic } }
    }

    /// A second empty store over the same weight table, for another diagram of
    /// the same weighted build. The table is shared, not copied.
    pub fn empty_like(&self) -> Self {
        Self {
            per_level: FxHashMap::default(),
            config: self.config.clone(),
        }
    }

    /// The algebra the values are folded in: the weight table and its
    /// operations.
    #[inline]
    pub fn algebra(&self) -> &RationalWeights {
        &self.config.algebra
    }

    /// The store's arithmetic, fixed at construction.
    #[inline]
    pub fn arithmetic(&self) -> Arithmetic {
        self.config.arithmetic
    }

    /// Take over every level `other` holds that this store does not.
    ///
    /// Requires compatible weight configurations. Used where two diagrams meet:
    /// their marginal levels are the two disjoint
    /// vtree subtrees they were built over, except for leaf columns, which are
    /// a pure function of the weight table and therefore already equal.
    pub(crate) fn absorb(&mut self, other: Self) {
        debug_assert!(self.compatible(&other), "merged columns must use compatible weights");
        for (level, values) in other.per_level {
            self.per_level.entry(level).or_insert(values);
        }
    }

    /// True in the bounded-precision log domain.
    #[inline]
    pub(crate) fn is_log(&self) -> bool {
        self.config.arithmetic == Arithmetic::SignedLog
    }

    /// The additive identity in the active mode.
    #[inline]
    pub(crate) fn wzero(&self) -> WeightValue {
        if self.is_log() {
            WeightValue::Log(SignedLog::zero())
        } else {
            // The canonical exact zero: 0 always fits the small representation.
            WeightValue::ExactSmall(0)
        }
    }

    /// Leaf base value for `(var, label)` in the active mode. The exact parsed
    /// weight from [`RationalWeights`] is authoritative; in log mode it is
    /// converted to `SignedLog` exactly once here (per leaf read).
    #[inline]
    pub(crate) fn leaf_val(&self, var: VarId, label: LeafLabel) -> WeightValue {
        let r = self.config.algebra.leaf(var, label);
        if self.is_log() {
            WeightValue::Log(SignedLog::from_rational(&r))
        } else {
            WeightValue::exact(r)
        }
    }

    /// Set the per-node values of weight-marginal level `level` (one entry per
    /// node, indexed like a marginal level's count table).
    pub(crate) fn set_level(&mut self, level: usize, values: Vec<WeightValue>) {
        self.per_level.insert(level, values);
    }

    /// The per-node values of the level of vtree node `level` (its
    /// `VtreeIdx::idx()`), or `None` if that level is not weight-marginal. A
    /// weight-marginal level ([`TddLevel::is_weight_marginal`]) keeps its
    /// values here rather than in the diagram, so this is how a traversal
    /// reads them; a parent pair's side into such a level decodes through the
    /// level's [`ChildDecoder`] to an index into this slice.
    ///
    /// [`TddLevel::is_weight_marginal`]: crate::diagram::TddLevel::is_weight_marginal
    /// [`ChildDecoder`]: crate::diagram::ChildDecoder
    #[inline]
    pub fn level(&self, level: usize) -> Option<&[WeightValue]> {
        self.per_level.get(&level).map(Vec::as_slice)
    }

    /// `&mut` into one level's value vec, for in-place compaction. A caller
    /// that moves or drops slots must rewrite the parent refs into this level
    /// in the same pass.
    #[inline]
    pub(crate) fn level_vals_mut(&mut self, level: usize) -> Option<&mut Vec<WeightValue>> {
        self.per_level.get_mut(&level)
    }

    /// Append `val` as a fresh slot to a weight-marginal level, returning the
    /// new slot index. No value interning: slot prune merges equal-valued slots
    /// on its next pass.
    ///
    /// # Panics
    ///
    /// Panics if `level` has no values yet (`set_level` was never called for it).
    pub(crate) fn push_value(&mut self, level: usize, val: WeightValue) -> usize {
        let vec = self
            .per_level
            .get_mut(&level)
            .expect("push_value: level has no weighted store");
        let idx = vec.len();
        vec.push(val);
        idx
    }

    /// Remove `level`'s values from this store and hand them over, or `None` if
    /// that level is not weight-marginal.
    #[inline]
    pub(crate) fn take_level(&mut self, level: usize) -> Option<Vec<WeightValue>> {
        self.per_level.remove(&level)
    }

    /// True once `set_level` has been called for `level`.
    #[inline]
    pub(crate) fn is_set(&self, level: usize) -> bool {
        self.per_level.contains_key(&level)
    }
}
