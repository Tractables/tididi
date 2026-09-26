//! The sparse route: a level whose product grid would be far larger than
//! its live products is built by a scatter-filter-dedup pipeline over the
//! children's product lists instead of by walking the grid. `route.rs`
//! decides which levels take it.

use smallvec::SmallVec;

use crate::vtree::VtreeIdx;
use crate::Engine;
use super::{OperationError, NO_PRODUCT, Tdd, TddLevel, ChildPair, CONJOIN_GRID, finish_node, reserve_pairs_for_emit, try_push_pair_into};
use super::cell::emit_single_pair;
use super::products::{FNodeIdx, ProductEntry, ProductLists, ProductNodeIdx, GNodeIdx};

mod config;
use config::*;
pub(crate) use config::{sparse_thresholds, SparseThresholds};
mod index;
use index::*;
pub(crate) use index::SparseWorkspace;
mod scatter;
use scatter::*;
mod level;
pub(crate) use level::{apply_sparse_level, count_sparse_level, Passthrough};
mod fold;
pub(crate) use fold::CandidateFold;
pub(crate) mod stream;

#[cfg(test)]
mod tests;
