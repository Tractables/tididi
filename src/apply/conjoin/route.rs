//! Choosing how a level is walked and how its output arrays grow.

use super::*;

/// The bottom-up sweep's level walk: the tuned lazy `internal_bottomup` iterator
/// (the byte-identical main-compile order) or the spine-bounded apply's
/// restricted level list. A two-variant enum, not `Box<dyn Iterator>` — that box
/// cost one heap allocation per apply and an indirect `next()` per level, while
/// the loop body below is ONE loop either way.
pub(super) enum LevelWalk<'a, I> {
    Depth(I),
    /// Spine-bounded apply: `R` in `topo_pos` order (the `Depth` order with the
    /// levels that would take an identity fast path removed).
    Restricted(std::slice::Iter<'a, VtreeIdx>, &'a crate::vtree::Vtree),
}

impl<'a, I: Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)>> Iterator for LevelWalk<'a, I> {
    type Item = (VtreeIdx, VtreeIdx, VtreeIdx);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LevelWalk::Depth(it) => it.next(),
            LevelWalk::Restricted(it, vtree) => it.next().map(|&t| {
                let (l, r) = vtree.children(t);
                (t, l, r)
            }),
        }
    }
}

/// Conservative per-cell byte factor for the apply's product grid: pairs (8B)
/// + nodes (8B) + scratch (4–8B) ≈ 24B. Shared by the in-apply predictive budget
/// check (above) and the pre-apply size gate so both agree on the byte
/// conversion.
pub(crate) const APPLY_BYTES_PER_CELL: u64 = 24;

/// Byte ceiling on each of the two per-level exact reserves (`level.nodes` and
/// `level.pairs`, both sized from that level's exact upper bound).
///
/// The bounds — `k1 × k2` live cells, `|c1.pairs| × |c2.pairs|` emitted pairs —
/// are exact but loose: most levels have low survival, so an uncapped reserve
/// would routinely grab orders of magnitude more than the level ends up using
/// (and charge every byte of it to the soft budget). Capping bounds the
/// over-allocation per level; a level that outgrows the cap keeps growing
/// through the ordinary fallible push path, and `shrink_arrays` at
/// `finalize_level` hands the unused tail back (it shrinks at cap > 4 × len).
/// ONE definition — the two element-count caps below derive from it.
pub(super) const LEVEL_RESERVE_CAP_BYTES: usize = 64 * 1024;

/// [`LEVEL_RESERVE_CAP_BYTES`] in `TddNodeData`s — the `level.nodes` arm.
pub(super) const LEVEL_RESERVE_NODES_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<TddNodeData>();

/// [`LEVEL_RESERVE_CAP_BYTES`] in `InputPair`s — the `level.pairs` arm.
pub(super) const LEVEL_RESERVE_PAIRS_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<InputPair>();

