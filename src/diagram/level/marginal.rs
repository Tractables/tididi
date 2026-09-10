//! Converting a level to its marginal form, and the marginal-side slot writer.

use crate::diagram::marginal_ref::{BigSide, MARGINAL_OVERFLOW_TAG, ValueRef, marginal_inline_max};
use crate::diagram::NodeIdx;
use super::{LevelState, TddLevel};

impl TddLevel {
    /// Inline-emit writer (end-of-apply tagger inner): for each bare marginal-side
    /// slot ref, look up its child count and either INLINE it (bit-30 set) when
    /// small, or keep it a bare self-describing slot (bit-30 clear) when
    /// large/big-table.
    pub(crate) fn emit_marginal_side_slots(
        &mut self,
        left_counts: Option<&[u128]>,
        right_counts: Option<&[u128]>,
    ) {
        // Inline rule: a marginal-side ref is inlined whenever its count is
        // INLINABLE (≤ MARGINAL_INLINE_MAX, not a u128::MAX overflow); a large or
        // overflow count stays a TAGGED SLOT. Counts need not be unique on the
        // child level — the count IS the anonymous identity of a marginal node,
        // so two slots sharing a count are interchangeable: count consumers read
        // the same value either way; the only structural use of a marginal slot as a grid coordinate is the
        // invariant-forbidden marginal×marginal conjoin (marginal×identity is
        // pass-through, grid result discarded); and duplicate pairs are summed,
        // not deduped (see the `// No \`pairs.dedup()\`` notes in
        // conjoin_clause.rs / conjoin/sparse.rs), so collapsing two same-count
        // refs to one inline value preserves the total.
        // Rewrite one marginal-side ref toward the inline OPTIMISATION. Under the
        // bit-30-clear==slot polarity bit 30 alone disambiguates — no marker or
        // self-describing flag needed:
        //   bit-31 set → ZERO sentinel, pass through.
        //   bit-30 set → already an inline count → idempotent pass-through.
        //   bit-30 clear → bare slot: resolve its count and INLINE it (set bit-30)
        //                  when small (and, under the gate, unique); else leave it
        //                  a bare slot — which is already a correct, self-describing
        //                  reference, so doing nothing is sound.
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
            let c = counts[slot];
            // Inline whenever the count fits the inline width. Duplicates are
            // allowed: the count is the anonymous identity of a marginal node, and
            // duplicate pairs are summed (never deduped), so collapsing two
            // same-count refs to one inline value preserves the total.
            let inlinable = c != u128::MAX && c <= marginal_inline_max() as u128;
            if inlinable {
                // Invariant: counts at marginalization are ≥ 1. Dead/UNSAT nodes
                // are zero-suppressed during apply and eliminated by prune_unreachable
                // before any count is taken; every surviving node therefore has at
                // least one model. Inline(0) is unreachable on any natural compile
                // path — only artificially-constructed diagrams (e.g. unit tests) can
                // produce it here.
                ValueRef::Inline(c as u32).to_raw().0 // INLINE: bit-30 set
            } else {
                raw // keep as a bare slot (bit-30 clear)
            }
        }
        for node in &mut self.nodes {
            if node.is_inline() {
                if let Some(lc) = left_counts {
                    node.a = emit_or_tag(node.a, lc);
                }
                if let Some(rc) = right_counts {
                    node.b = emit_or_tag(node.b, rc);
                }
            }
        }
        for p in &mut self.pairs {
            if let Some(lc) = left_counts {
                p.left.0 = emit_or_tag(p.left.0, lc);
            }
            if let Some(rc) = right_counts {
                p.right.0 = emit_or_tag(p.right.0, rc);
            }
        }
    }

    /// Replace this level's structure with per-node counts: `counts[i]` for
    /// node `i`, with `u128::MAX` marking an overflow whose exact value is
    /// `big.get(i)`.
    ///
    /// Marginality must stay downward-closed, so both child levels must
    /// already be marginal or leaves. This does not check; parents that refer
    /// to this level keep their indices, which remain valid as bare slot
    /// references (see [`SideView::child`](crate::diagram::SideView::child)).
    pub fn become_marginal(&mut self, counts: Vec<u128>, big: Option<BigSide>) {
        self.drop_structure();
        self.state = LevelState::Counts { counts, big, retired: 0 };
    }


    /// Marginalize this level into `slots` weighted slots, whose values live in the
    /// diagram's external `WeightStore`.
    ///
    /// The slot count is explicit because it is not always the node count: the
    /// streaming path remaps parent refs to compacted CELL indices, so its
    /// count is the number of alive cells. Clearing `nodes` (as the integer
    /// `become_marginal` does) is what makes every structural traversal a no-op
    /// on a weight-marginal level instead of indexing the freed `pairs`; the
    /// width readers fall back to this count, so parent marginal-side refs — bare
    /// slot indices — stay in bounds.
    pub(crate) fn become_marginal_weighted(&mut self, slots: u32) {
        debug_assert!(
            !matches!(self.state, LevelState::Counts { .. }),
            "a level already holding counts cannot become weight-marginal"
        );
        self.drop_structure();
        self.state = LevelState::Weights { width: slots, retired: 0 };
    }
}
