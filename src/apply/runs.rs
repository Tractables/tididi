//! Items laid end to end in runs, each run found by two 32-bit offsets.
//!
//! An operation that lists, per node of a level, its products (or pieces,
//! states, sets) keeps them in one flat vector and the offsets where each
//! node's run begins in another. [`Runs`] is that pair: a run is closed by
//! recording where it ends, which is refused with
//! [`OperationError::IndexOverflow`] once the items no longer fit the
//! offsets, and read back as a slice. The items keep the caller's type and
//! are pushed into [`Runs::items`] directly or through [`Runs::push`].

use crate::limits::{Limits, OperationError};

/// Items in runs: run `k` is `items[starts[k]..starts[k + 1]]`.
///
/// Begun ([`begin`](Self::begin), [`begin_exact`](Self::begin_exact)), it
/// holds no run and the items pushed form the run open at the end, which
/// [`close`](Self::close) closes. Not begun, it holds no run and none can be
/// closed.
pub(crate) struct Runs<T> {
    /// The items, run after run, then those of the run open at the end.
    pub(crate) items: Vec<T>,
    /// The first run's start, then each closed run's end.
    starts: Vec<u32>,
}

impl<T> Default for Runs<T> {
    fn default() -> Self {
        Runs {
            items: Vec::new(),
            starts: Vec::new(),
        }
    }
}

impl<T> Runs<T> {
    /// Runs read from `starts` into `items`, as built elsewhere: `starts`
    /// begins at 0, never decreases, and ends within `items`.
    pub(crate) fn from_parts(items: Vec<T>, starts: Vec<u32>) -> Self {
        debug_assert!(starts.first().is_none_or(|&s| s == 0), "the first run starts at 0");
        debug_assert!(starts.is_sorted(), "runs do not overlap");
        debug_assert!(
            starts.last().is_none_or(|&s| s as usize <= items.len()),
            "runs end within the items"
        );
        Runs { items, starts }
    }

    /// Empty both lists, keeping their allocations.
    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.starts.clear();
    }

    /// Empty both lists and begin, room for the offsets grown as runs close.
    pub(crate) fn begin(&mut self, lim: &Limits) -> Result<(), OperationError> {
        self.clear();
        lim.try_push(&mut self.starts, 0)
    }

    /// Empty both lists and begin, with room for the offsets of `runs` runs
    /// reserved exactly.
    pub(crate) fn begin_exact(&mut self, lim: &Limits, runs: usize) -> Result<(), OperationError> {
        self.clear();
        lim.reserve_exact(&mut self.starts, runs + 1)?;
        self.starts.push(0);
        Ok(())
    }

    /// Push `x` onto the open run.
    #[inline(always)]
    pub(crate) fn push(&mut self, lim: &Limits, x: T) -> Result<(), OperationError> {
        lim.try_push(&mut self.items, x)
    }

    /// Close the open run at the items pushed so far.
    ///
    /// # Errors
    ///
    /// [`OperationError::IndexOverflow`] when the items no longer fit a
    /// 32-bit offset ([`Limits::list_index`]), and refused growth.
    #[inline]
    pub(crate) fn close(&mut self, lim: &Limits) -> Result<(), OperationError> {
        let end = lim.list_index(self.items.len())?;
        lim.try_push(&mut self.starts, end)
    }

    /// Close the open run, then empty ones, until `runs` are closed; none
    /// when as many are already.
    ///
    /// # Errors
    ///
    /// As [`close`](Self::close).
    #[inline]
    pub(crate) fn close_until(&mut self, lim: &Limits, runs: usize) -> Result<(), OperationError> {
        if self.starts.len() <= runs {
            let end = lim.list_index(self.items.len())?;
            while self.starts.len() <= runs {
                lim.try_push(&mut self.starts, end)?;
            }
        }
        Ok(())
    }

    /// The closed runs.
    #[inline]
    pub(crate) fn runs(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Run `k`'s items.
    #[inline(always)]
    pub(crate) fn run(&self, k: usize) -> &[T] {
        &self.items[self.starts[k] as usize..self.starts[k + 1] as usize]
    }

    /// The open run's items, after the last closed run.
    #[inline]
    pub(crate) fn open_run_mut(&mut self) -> &mut [T] {
        let at = self.starts.last().map_or(0, |&s| s as usize);
        &mut self.items[at..]
    }
}

#[cfg(test)]
#[path = "tests/runs.rs"]
mod tests;
