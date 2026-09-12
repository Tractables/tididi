//! A pooled per-level flag array whose all-false invariant survives any exit.

use crate::limits::pool::Pool;
use crate::vtree::VtreeIdx;

/// A pooled per-vtree-level flag array, handed out all-false and restored
/// all-false however the caller leaves.
///
/// The array is cleared in `Drop` by replaying the levels that were marked, so
/// a call touching a few levels of a large vtree pays for those levels rather
/// than the whole array, and the restore runs on the error paths too.
pub(crate) struct ScopedFlags<'a> {
    flags: Vec<bool>,
    set: Vec<VtreeIdx>,
    pool: &'a Pool<Vec<bool>>,
}

impl<'a> ScopedFlags<'a> {
    /// Take the pooled array, grown to cover `num_nodes` levels.
    pub(crate) fn take(pool: &'a Pool<Vec<bool>>, num_nodes: usize) -> Self {
        let mut flags = pool.take();
        if flags.len() < num_nodes {
            flags.resize(num_nodes, false);
        }
        ScopedFlags { flags, set: Vec::new(), pool }
    }

    /// Mark level `t`. There is no unmark: the restore is the only one.
    #[inline]
    pub(crate) fn set(&mut self, t: VtreeIdx) {
        if !self.flags[t.idx()] {
            self.flags[t.idx()] = true;
            self.set.push(t);
        }
    }

    /// Mark levels through a walk that reports what it marked, for the callers
    /// whose marking is a tree walk rather than a sequence of `set`s.
    ///
    /// `mark` must raise exactly the flags it reports and lower none.
    pub(crate) fn mark(&mut self, mark: impl FnOnce(&mut [bool], &mut Vec<VtreeIdx>)) {
        mark(&mut self.flags, &mut self.set);
    }
}

impl std::ops::Deref for ScopedFlags<'_> {
    type Target = [bool];
    #[inline]
    fn deref(&self) -> &[bool] {
        &self.flags
    }
}

impl Drop for ScopedFlags<'_> {
    fn drop(&mut self) {
        for &t in &self.set {
            self.flags[t.idx()] = false;
        }
        debug_assert!(
            self.flags.iter().all(|&b| !b),
            "a scoped flag array was left marked",
        );
        self.pool.put(std::mem::take(&mut self.flags));
    }
}
