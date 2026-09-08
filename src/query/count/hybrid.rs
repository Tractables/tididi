//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::engine::Engine;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed_u128, leaf_seed_u128_fix, SeedConvention};
use crate::counts::{
    Count, CountRead, CountVec, RecoveryPanic, STREAM_OVERFLOW as OVERFLOW,
};
use crate::counts::ColumnRetention;
use crate::diagram::*;
use crate::vtree::{VarId, VtreeIdx};

/// Seed one leaf vtree level `ti` (hybrid counts) under variable pin `pin`. `fix`
/// selects the FIX convention (pinned var counted ×1, see [`leaf_seed_u128_fix`])
/// over the freed convention (×2, [`leaf_seed_u128`]).
#[inline]
fn hybrid_seed_leaf(eng: &Engine, cols: &mut [CountVec<RecoveryPanic>], ti: usize, pin: Option<bool>, fix: bool) {
    for i in 0..LEAF_WIDTH {
        let seed = if fix {
            leaf_seed_u128_fix(LeafLabel::from_idx(i), pin)
        } else {
            leaf_seed_u128(LeafLabel::from_idx(i), pin)
        };
        cols[ti].set_i(eng, i, Count::from_u128(seed));
    }
}

/// Recompute one internal vtree level `t` with u128-primary arithmetic, spilling a node
/// to the `BigUint` side-table only on overflow — the pinned mirror of
/// [`model_count_hybrid`]'s internal pass. Reads both children's columns (already
/// computed), writes `cols[t]`. The sentinel ⟺ big-slot invariant, the exact-max
/// promotion, and the stale-overflow clear on recompute (a node may stop overflowing
/// when pins change) are all owned by [`CountVec::set`]/[`Count::from_u128`].
fn hybrid_recompute_internal(eng: &Engine, tdd: &Tdd, cols: &mut [CountVec<RecoveryPanic>], t: VtreeIdx) {
    let ti = t.idx();
    let level = &tdd.levels[ti];
    if level.is_marginal() {
        let ic = level.marginal_counts.as_ref().unwrap();
        for (i, &c) in ic.iter().enumerate() {
            if c == OVERFLOW {
                let bv = level
                    .marginal_counts_big
                    .as_ref()
                    .and_then(|m| m.get(i).cloned())
                    .expect("marginal OVERFLOW slot without a big entry — level invariant violated");
                cols[ti].set_i(eng, i, Count::Big(bv));
            } else {
                cols[ti].set_i(eng, i, Count::from_u128(c));
            }
        }
        return;
    }
    let (left_child, right_child) = tdd.vtree.children(t);
    let li = left_child.idx();
    let ri = right_child.idx();
    let li_view = tdd.levels[li].side_view();
    let ri_view = tdd.levels[ri].side_view();
    for (i, pairs) in level.internal_inputs_iter() {
        let mut total: u128 = 0;
        let mut overflowed = false;
        for pair in pairs {
            let lc = match li_view.child(pair.left) {
                ChildRef::Value(ValueRef::Inline(c)) => c as u128,
                ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => { let idx = idx as usize; cols[li].fast_val(idx) },
            };
            // Zero-operand pairs (left subfunction UNSAT under the pins) contribute
            // 0·rc = 0: skip without even resolving rc. On the pinned cofactor eval these
            // dominate — pinning the relaxed vars leaves 50–90% of pairs with a zero
            // operand — so this short-circuit avoids the bulk of the multiplies.
            if lc == 0 {
                continue;
            }
            let rc = match ri_view.child(pair.right) {
                ChildRef::Value(ValueRef::Inline(c)) => c as u128,
                ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => { let idx = idx as usize; cols[ri].fast_val(idx) },
            };
            if rc == 0 {
                continue;
            }
            // OVERFLOW × 1 wouldn't trip checked_mul, so test the sentinel explicitly.
            if lc == OVERFLOW || rc == OVERFLOW {
                overflowed = true;
                break;
            }
            match lc.checked_mul(rc).and_then(|p| total.checked_add(p)) {
                Some(v) => total = v,
                None => {
                    overflowed = true;
                    break;
                }
            }
        }
        if !overflowed {
            // `from_u128` owns the exact-max promotion (a natural total of exactly
            // u128::MAX routes to Big so parents reading the sentinel find a big
            // entry); `set` owns the stale-overflow clear.
            cols[ti].set_i(eng, i, Count::from_u128(total));
        } else {
            let mut bt = BigUint::ZERO;
            for pair in level.pairs_iter_of_idx(i) {
                // Resolve each operand to (u128 view, Some(&big) iff it overflowed).
                // Skip zero operands before any allocation — zero is always a clean
                // u128 (only OVERFLOW forces a big read). Then dispatch by width:
                //   both small  → u128 mul (no BigUint operand allocs at all);
                //   mixed       → scalar mul `&big * u128` (no small-operand alloc,
                //                 faster than promoting to BigUint + general mul);
                //   both big    → `&big * &big`, operands borrowed not cloned.
                let (lu, lbig) = match li_view.child(pair.left) {
                    ChildRef::Value(ValueRef::Inline(0)) => continue,
                    ChildRef::Value(ValueRef::Inline(c)) => (c as u128, None),
                    ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => {
                        let idx = idx as usize;
                        match cols[li].fast_val(idx) {
                            0 => continue,
                            OVERFLOW => (OVERFLOW, Some(sentinel_big(&cols[li], idx))),
                            v => (v, None),
                        }
                    }
                };
                let (ru, rbig) = match ri_view.child(pair.right) {
                    ChildRef::Value(ValueRef::Inline(0)) => continue,
                    ChildRef::Value(ValueRef::Inline(c)) => (c as u128, None),
                    ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => {
                        let idx = idx as usize;
                        match cols[ri].fast_val(idx) {
                            0 => continue,
                            OVERFLOW => (OVERFLOW, Some(sentinel_big(&cols[ri], idx))),
                            v => (v, None),
                        }
                    }
                };
                match (lbig, rbig) {
                    (None, None) => match lu.checked_mul(ru) {
                        Some(p) => bt += p,
                        None => bt += BigUint::from(lu) * BigUint::from(ru),
                    },
                    (Some(lb), None) => bt += lb * ru,
                    (None, Some(rb)) => bt += rb * lu,
                    (Some(lb), Some(rb)) => bt += lb * rb,
                }
            }
            cols[ti].set_i(eng, i, Count::Big(bt));
        }
    }
}

/// Borrow the big value of a slot known to hold the overflow sentinel.
/// Precondition: `cols.fast_val(node) == OVERFLOW` (then `big_val` is `Some`
/// by the `CountVec` invariant).
#[inline]
fn sentinel_big(col: &CountVec<RecoveryPanic>, node: usize) -> &BigUint {
    col.big_val(node)
        .expect("CountVec: sentinel fast slot without a big value — invariant violated")
}

/// Incremental pinned model counter for a Gray-code cofactor sum.
///
/// One hybrid `CountVec` column per vtree level (u128-primary, `BigUint` side
/// table on overflow — keeps 99%+ of arithmetic off the heap; the discipline
/// shared with the apply/marginalize contexts, see `tididi/src/tdd/counts.rs`). Holds
/// the full per-node count array; after one [`recompute_all`](Self::recompute_all), flipping a
/// few variables' pins and calling [`recompute_dirty`](Self::recompute_dirty) on just
/// the affected vtree levels (the "dirty cone" from those leaves to the root) updates the
/// root count in `O(cone)` instead of the `O(|D|)` of a fresh full pass — every unaffected
/// node's cached count is reused verbatim. The result equals [`pinned_counts`].
pub struct IncrementalPinnedCounter {
    cols: Vec<CountVec<RecoveryPanic>>,
    pins: Vec<Option<bool>>,
    /// The leaf-seed convention this counter pins with.
    convention: SeedConvention,
    /// Column-lifetime policy for [`recompute_all`](Self::recompute_all); see
    /// [`ColumnRetention`]. `Frontier` makes the counter root-read-only.
    retain: ColumnRetention,
    /// Whether a pass has run. Every column starts at zero, so a read before the
    /// first pass returns a count that is indistinguishable from UNSAT; the
    /// reads assert on this rather than letting that pass silently.
    computed: bool,
}

impl IncrementalPinnedCounter {
    /// Allocate the count array with pin slots `0..n_pins`. No pass run yet.
    ///
    /// `convention` is the leaf seed a pinned variable gets
    /// ([`SeedConvention`]); with zero pins the two coincide.
    ///
    /// `retain` is the column-lifetime policy ([`ColumnRetention`]):
    /// - `All` allocates every level's column up front and keeps it. Required by
    ///   [`recompute_dirty`](Self::recompute_dirty) (the Gray-code dirty-cone
    ///   update re-reads cached columns) and by
    ///   `into_fast_counts` (which hands the whole array
    ///   out). Both fail fast under `Frontier`.
    /// - `Frontier` allocates each column only when the pass writes it and frees
    ///   each child as its parent completes, so peak is the pass frontier rather
    ///   than the whole diagram. ROOT-ONLY: the only legal read afterwards is
    ///   [`root_count`](Self::root_count).
    ///
    /// The counter OWNS its `cols`/`pins` arrays (sized from `tdd` here) and does
    /// not borrow `tdd` — every method takes `tdd` as an argument. This lets a caller keep
    /// one counter alive across many evaluations of the SAME diagram (re-pinning + a dirty-
    /// cone [`recompute_dirty`](Self::recompute_dirty) under `All`, or re-pinning + a
    /// fresh [`recompute_all`](Self::recompute_all) under `Frontier`, instead of a new
    /// allocation each time). Callers MUST pass the same `tdd` the counter was sized from;
    /// passing a structurally different diagram is a logic error (the arrays would be
    /// mis-sized).
    pub fn new(eng: &Engine, tdd: &Tdd, n_pins: usize, convention: SeedConvention, retain: ColumnRetention) -> Self {
        let cols = (0..tdd.vtree.num_nodes())
            .map(|i| match retain {
                ColumnRetention::All => {
                    CountVec::with_width(eng, tdd.effective_width(VtreeIdx(i as u32)))
                }
                // Frontier: allocate on write (`ensure_col`), free on parent
                // completion — pre-sizing here would commit the whole-diagram
                // array this policy exists to avoid.
                ColumnRetention::Frontier => CountVec::with_width(eng, 0),
            })
            .collect();
        Self {
            cols,
            pins: vec![None; n_pins],
            convention,
            retain,
            computed: false,
        }
    }

    /// Set one variable's pin (does not recompute). `var.idx()` must be `< n_pins`.
    #[inline]
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) {
        self.pins[var.idx()] = val;
    }

    /// (Re)allocate `cols[ti]` to level `ti`'s effective width when it is not
    /// already that size. A no-op under [`ColumnRetention::All`] (the
    /// constructor pre-sized every column and nothing shrinks them), so this is
    /// one length compare per level on that path; under `Frontier` it is the
    /// allocate-on-write step for a column that starts — or was freed — empty.
    /// A freshly allocated column is all-zero, which is exactly what a fresh
    /// counter's column holds, so slots no pass writes (tombstones, which
    /// `internal_inputs_iter` skips) read the same under both policies.
    #[inline]
    fn ensure_col(&mut self, eng: &Engine, tdd: &Tdd, ti: usize) {
        let w = tdd.effective_width(VtreeIdx(ti as u32));
        if self.cols[ti].len() != w {
            self.cols[ti] = CountVec::with_width(eng, w);
        }
    }

    /// Full bottom-up pass under the current pins (every leaf + every internal level).
    /// Call once for the starting Gray-code state — or once per pin assignment when
    /// the counter is `Frontier` (which has no incremental path).
    pub fn recompute_all(&mut self, eng: &Engine, tdd: &Tdd) {
        self.computed = true;
        let out_t = tdd.output.vtree.idx();
        if self.retain == ColumnRetention::Frontier {
            // Free-before-rebuild: drop the previous pass's surviving column
            // (the root's, plus any level this pass will not revisit) BEFORE
            // allocating anything new, so two passes' peaks never overlap.
            for c in &mut self.cols {
                *c = CountVec::with_width(eng, 0);
            }
        }
        for (t, var) in tdd.vtree.leaf_bottomup() {
            let pin = self.pins.get(var.idx()).copied().flatten();
            self.ensure_col(eng, tdd, t.idx());
            hybrid_seed_leaf(eng, &mut self.cols, t.idx(), pin, matches!(self.convention, SeedConvention::Fix));
        }
        for (t, l, r) in tdd.vtree.internal_bottomup() {
            self.ensure_col(eng, tdd, t.idx());
            hybrid_recompute_internal(eng, tdd, &mut self.cols, t);
            if self.retain == ColumnRetention::Frontier {
                // The vtree is a tree: `t` is the ONE parent of `l`/`r`, so
                // their columns are dead now that `t`'s is complete. `out_t` is
                // the single column read after the pass — it is the root under
                // the output-at-root invariant (hence never a child here), but
                // an all-backbone compile can collapse the output onto a LEAF
                // level that IS a child, so the guard is load-bearing.
                for c in [l.idx(), r.idx()] {
                    if c != out_t {
                        self.cols[c] = CountVec::with_width(eng, 0);
                    }
                }
            }
        }
    }

    /// Recompute exactly `levels`, in the given order — which MUST be children-before-
    /// parents (a `bottomup_topo`-ordered subset). Leaf levels are re-seeded from the
    /// current pins; internal levels are re-summed from their (already-updated) children.
    ///
    /// # Panics
    ///
    /// Panics unless the counter was built with [`ColumnRetention::All`] — the
    /// dirty-cone update reads cached columns outside `levels`, which
    /// `Frontier` frees as parents complete.
    pub fn recompute_dirty(&mut self, eng: &Engine, tdd: &Tdd, levels: &[VtreeIdx]) {
        assert_eq!(
            self.retain,
            ColumnRetention::All,
            "recompute_dirty requires ColumnRetention::All: the dirty-cone update re-reads \
             cached child columns, which ColumnRetention::Frontier frees as parents complete"
        );
        self.computed = true;
        for &t in levels {
            if tdd.vtree.node(t).is_leaf() {
                let var = tdd.vtree.leaf_var(t);
                let pin = self.pins.get(var.idx()).copied().flatten();
                hybrid_seed_leaf(eng, &mut self.cols, t.idx(), pin, matches!(self.convention, SeedConvention::Fix));
            } else {
                hybrid_recompute_internal(eng, tdd, &mut self.cols, t);
            }
        }
    }

    /// The current root (output) model count.
    ///
    /// Requires a completed pass ([`recompute_all`](Self::recompute_all) or
    /// [`recompute_dirty`](Self::recompute_dirty)). A fresh counter's columns
    /// are all zero, so reading one would report an UNSAT count for a diagram
    /// that was never counted; that is a `debug_assert` here, not a `None`.
    #[inline]
    pub fn root_count(&self, tdd: &Tdd) -> BigUint {
        debug_assert!(self.computed, "root_count before any pass reads zero, not the count");
        let (t, i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
        match self.cols[t].get(i) {
            CountRead::Fast(v) => BigUint::from(v),
            CountRead::Big(b) => b.clone(),
        }
    }

    /// Consume the counter, returning the per-node u128 count columns
    /// (`fast[t][i]`) and discarding the `BigUint` side table. A slot that
    /// counted past `u128` saturates to `OVERFLOW` (`u128::MAX`) and its exact
    /// magnitude is dropped. ZERO is exact: the u128 array is authoritative for
    /// zero — only a *non-zero* overflow ever spills to the Big side table — so
    /// `fast[t][i] == 0` iff node `(t,i)` has no models. For callers that need
    /// only monotone ordering, a small-threshold compare, and exact-zero
    /// detection (sat-prune MC-priority), never an overflowed node's exact value.
    ///
    /// # Panics
    ///
    /// Panics unless the counter was built with [`ColumnRetention::All`] —
    /// `Frontier` keeps only the root column, so there is no per-node array to
    /// hand out.
    pub(crate) fn into_fast_counts(self) -> Vec<Vec<u128>> {
        assert_eq!(
            self.retain,
            ColumnRetention::All,
            "into_fast_counts requires ColumnRetention::All: ColumnRetention::Frontier keeps \
             only the root column, so the per-node array does not exist"
        );
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
    }
}
