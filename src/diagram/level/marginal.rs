//! Converting a level to its marginal form, and the marginal-side slot writer.

use crate::diagram::marginal_ref::refs::{for_each_side_ref_mut, ChildSide};
use crate::diagram::marginal_ref::{INLINE_VALUE_BIT, ValueRef};
use crate::diagram::NodeIdx;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use super::{CountOverflow, LevelState, TddLevel};

impl TddLevel {
    /// One level's share of `inline_small_marginal_refs`: for each bare slot
    /// reference on a side with counts, look up the child's count and
    /// replace the reference with the count itself (bit 30 set) when it
    /// fits; a larger count, the overflow marker included, keeps its slot.
    pub(crate) fn inline_small_refs(
        &mut self,
        left_counts: Option<&[u128]>,
        right_counts: Option<&[u128]>,
    ) {
        // Inline rule: a bare marginal-side slot ref is inlined when its count
        // fits an inline value (`ValueRef::inline_raw`); otherwise it stays a
        // bare slot, which is already a correct reference.
        // Counts need not be unique on the child level: a marginal node has no
        // identity beyond its count, and duplicate pairs are summed, not
        // deduped (see `conjoin_clause/rebuild.rs`), so two same-count refs
        // collapsing to one inline value preserves the total.
        //   bit-31 set → `ZERO` sentinel, pass through.
        //   bit-30 set → already inline, pass through.
        //   bit-30 clear → bare slot: inline its count when small.
        fn emit_or_tag(raw: u32, counts: &[u128]) -> u32 {
            if NodeIdx(raw).is_reserved() {
                return raw;
            }
            if raw & INLINE_VALUE_BIT != 0 {
                return raw; // already inline (bit-30 set) — idempotent
            }
            let slot = (raw & crate::diagram::marginal_ref::MARGINAL_VALUE_MASK) as usize;
            if slot >= counts.len() {
                return raw; // OOB ⟹ keep as a bare slot
            }
            // Counts at marginalization are ≥ 1 on any compile path (apply
            // is zero-suppressed), so `Inline(0)` arises only from a
            // hand-built diagram. A count too large to inline, the
            // `u128::MAX` overflow marker included, keeps its bare slot.
            ValueRef::inline_raw(counts[slot]).unwrap_or(raw)
        }
        if let Some(lc) = left_counts {
            for_each_side_ref_mut(self, ChildSide::Left, |r| *r = emit_or_tag(*r, lc));
        }
        if let Some(rc) = right_counts {
            for_each_side_ref_mut(self, ChildSide::Right, |r| *r = emit_or_tag(*r, rc));
        }
    }

    /// Replace this level's structure with per-node counts: `counts[i]` for
    /// node `i`, with `u128::MAX` marking an overflow whose exact value is
    /// `big.get(i)`.
    ///
    /// Marginality must stay downward-closed, so both child levels must
    /// already be marginal or leaves. This does not check; parents that refer
    /// to this level keep their indices, which remain valid as bare slot
    /// references (see [`ChildDecoder::child`](crate::diagram::ChildDecoder::child)).
    pub(crate) fn become_marginal(&mut self, counts: Vec<u128>, big: Option<CountOverflow>) {
        self.drop_structure();
        self.state = LevelState::Counts { counts, big, retired: 0 };
    }


    /// Marginalize this level into `slots` weighted slots, whose values live in the
    /// diagram's external `WeightStore`.
    ///
    /// `slots` is explicit because it is not always the node count: a
    /// streaming marginalization remaps parent refs to compacted cell indices.
    /// `slot_count()` reads it back, so parent marginal-side refs stay in bounds.
    pub(crate) fn become_marginal_weighted(&mut self, slots: u32) {
        debug_assert!(
            !matches!(self.state, LevelState::Counts { .. }),
            "a level already holding counts cannot become weight-marginal"
        );
        self.drop_structure();
        self.state = LevelState::Weights { width: slots, retired: 0 };
    }
}

/// The child of `t` that keeps `t` from being marginal, if any: an internal
/// child that is not marginal itself. Marginality is downward-closed; a leaf
/// child counts as marginal, its counts being fixed by label, so a leaf `t`
/// has no such child.
pub(crate) fn non_marginal_child(levels: &[TddLevel], vtree: &Vtree, t: VtreeIdx) -> Option<VtreeIdx> {
    let VtreeNode::Internal { left, right, .. } = *vtree.node(t) else {
        return None;
    };
    [left, right].into_iter().find(|child| {
        !matches!(*vtree.node(*child), VtreeNode::Leaf { .. }) && !levels[child.idx()].is_marginal()
    })
}

/// Soundness precondition for [`TddLevel::become_marginal`]: both children
/// of the target vtree node `t` must already be marginal
/// ([`non_marginal_child`] finds none).
///
/// # Panics
///
/// Panics if an internal child of `t` is not yet marginal; process targets
/// bottom-up so the precondition holds.
#[inline]
pub(crate) fn assert_can_make_marginal(levels: &[TddLevel], vtree: &Vtree, t: VtreeIdx) {
    if let Some(child) = non_marginal_child(levels, vtree, t) {
        panic!(
            "become_marginal({}) precondition violated: child {} is internal \
             but not yet marginal. Process marginalization targets bottom-up so \
             children are marginalized before parents.",
            t.idx(),
            child.idx(),
        );
    }
}
