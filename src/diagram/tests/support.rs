//! Node constructors the reduction tests stage arenas with by hand.

use super::primitives::{LEAF_BIT, TOMBSTONE_B};
use super::*;

impl EncodedNode {
    /// Create a leaf node for `label`.
    pub(crate) fn leaf(label: LeafLabel) -> Self {
        EncodedNode { a: label as u32, b: LEAF_BIT }
    }

    /// A dead node slot kept in place (not compacted) by the index-stable
    /// conjoin. `a = u32::MAX` is a tripwire: it is not a valid leaf label, so
    /// `leaf_label()` panics in debug if a tombstone is ever mistaken for a real
    /// leaf. See `TOMBSTONE_B`.
    pub(crate) fn tombstone() -> Self { EncodedNode { a: u32::MAX, b: TOMBSTONE_B } }
}
