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
    pool: &'a Pool<FlagBuffer>,
}

/// The flags and their rollback log reuse capacity together.
#[derive(Default)]
pub(crate) struct FlagBuffer {
    flags: Vec<bool>,
    set: Vec<VtreeIdx>,
}

impl<'a> ScopedFlags<'a> {
    /// Take the pooled array, grown to cover `num_nodes` levels.
    pub(crate) fn take(lim: &crate::limits::Limits, pool: &'a Pool<FlagBuffer>, num_nodes: usize) -> Result<Self, crate::limits::OperationError> {
        let FlagBuffer { mut flags, mut set } = pool.take();
        lim.try_resize(&mut flags, num_nodes, false)?;
        lim.reserve_exact(&mut set, num_nodes)?;
        Ok(ScopedFlags { flags, set, pool })
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
    pub(crate) fn mark<R>(&mut self, mark: impl FnOnce(&mut [bool], &mut Vec<VtreeIdx>) -> R) -> R {
        mark(&mut self.flags, &mut self.set)
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
        self.set.clear();
        crate::limits::pool::release_if_oversized(&mut self.set);
        crate::limits::pool::release_if_oversized(&mut self.flags);
        self.pool.put(FlagBuffer { flags: std::mem::take(&mut self.flags), set: std::mem::take(&mut self.set) });
    }
}

#[cfg(test)]
mod tests;
