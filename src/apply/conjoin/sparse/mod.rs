//! Sparse product construction for the apply algorithm.
//!
//! For levels where left_width * right_width exceeds `min_grid`, the dense grid iteration is
//! replaced by a scatter-filter-dedup pipeline. This module also contains the
//! leaf-level processing, identity product lists, and output index computation.

use smallvec::SmallVec;

use crate::vtree::VtreeIdx;
use crate::engine::Engine;
use super::{ApplyError, NO_PRODUCT, Tdd, TddLevel, InputPair, ZERO,
    NodeIdx, CONJOIN_GRID,
};

mod config;
pub(crate) use config::*;
mod index;
pub(crate) use index::*;
mod scatter;
pub(crate) use scatter::*;
mod level;
pub(crate) use level::*;

#[cfg(test)]
mod tests;
