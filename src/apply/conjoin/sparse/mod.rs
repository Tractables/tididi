//! Sparse product construction for the apply algorithm.
//!
//! For levels where left_width * right_width exceeds `min_grid`, the dense grid iteration is
//! replaced by a scatter-filter-dedup pipeline. This module also contains the
//! leaf-level processing, identity product lists, and output index computation.

use smallvec::SmallVec;

use crate::vtree::VtreeIdx;
use crate::Engine;
use super::{OperationError, NO_PRODUCT, Tdd, TddLevel, ChildPair, ZERO,
    NodeIdx, CONJOIN_GRID,
};

mod config;
use config::*;
pub(crate) use config::{sparse_thresholds, SparseThresholds};
mod index;
use index::*;
pub(crate) use index::{LeftNodeIdx, ProductEntry, ProductNodeIdx, RightNodeIdx, SparseWorkspace};
mod scatter;
use scatter::*;
mod level;
pub(crate) use level::{
    apply_leaf_levels, apply_sparse_level, compute_apply_output, fill_identity_product_list,
    is_self_conjunction, ProductLists,
};

#[cfg(test)]
mod tests;
