//! Sparse product construction for the apply algorithm.
//!
//! For levels where k1 * k2 > SPARSE_THRESHOLD, the dense grid iteration is
//! replaced by a scatter-filter-dedup pipeline. This module also contains the
//! leaf-level processing, identity product lists, and output index computation.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;

use smallvec::SmallVec;

use crate::vtree::VtreeIdx;
use super::{ApplyError, DEAD, Tdd, TddLevel, InputPair, ZERO,
    LocalNodeIdx, LevelGrid,
    try_push, try_resize, try_resize_dead, budget_reserve_exact, budget_reserve,
    CONJOIN_GRID, bump_live_count,
};

mod config;
pub(crate) use config::*;
mod index;
pub(crate) use index::*;
mod scatter;
pub(crate) use scatter::*;
mod level;
pub(crate) use level::*;
