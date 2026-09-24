//! Node constructors the reduction tests stage arenas with by hand.

use super::primitives::LEAF_BIT;
use super::*;

impl EncodedNode {
    /// Create a leaf node for `label`.
    pub(crate) fn leaf(label: LeafLabel) -> Self {
        EncodedNode { a: label as u32, b: LEAF_BIT }
    }
}
