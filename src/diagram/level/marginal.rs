//! Converting a level to its marginal form, and the marginal-side slot writer.

use crate::diagram::marginal_ref::refs::{for_each_side_ref_mut, ChildSide};
use crate::diagram::marginal_ref::{CountOverflow, MARGINAL_OVERFLOW_TAG, ValueRef};
use crate::diagram::NodeIdx;
use super::{LevelState, TddLevel};

impl TddLevel {
    /// Inline-emit writer (end-of-apply tagger inner): for each bare marginal-side
    /// slot ref, look up its child count and either inline it (bit-30 set) when
    /// small, or keep it a bare self-describing slot (bit-30 clear) when
    /// large/big-table.
    pub(crate) fn emit_marginal_side_slots(
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
            if raw & MARGINAL_OVERFLOW_TAG != 0 {
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
