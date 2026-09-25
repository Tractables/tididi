//! Validated placement of relation columns in a vtree.

use std::sync::Arc;
use crate::limits::{Limits, OperationError};
use crate::vtree::{VarId, Vtree};

/// Where each constrained variable sits once the rows are re-encoded, and how
/// many of them each vtree node's subtree holds.
///
/// Re-encoding puts the variables in the vtree's left-to-right leaf order, so
/// every node's constrained variables become the contiguous bit range
/// `[lo, lo + count)` of a row. Reading a node's value is then a shift and a
/// mask instead of a gather over scattered bits.
#[derive(Default)]
pub(crate) struct Layout {
    vtree: std::sync::Weak<Vtree>,
    vars: Vec<VarId>,
    identity: bool,
    /// Constrained variables in each vtree node's subtree.
    pub(super) count: Vec<u32>,
    /// First re-encoded bit position of each vtree node's subtree.
    pub(super) lo: Vec<u32>,
    /// Re-encoded bit position of each variable of `vars`.
    pub(super) position: Vec<u32>,
}

impl Layout {
    /// Reuse only the exact vtree allocation and input column order.
    pub(super) fn prepare_for(&mut self, lim: &Limits, vtree: &Arc<Vtree>, vars: &[VarId]) -> Result<(), OperationError> {
        if self.vtree.as_ptr() != Arc::as_ptr(vtree) || self.vars != vars {
            *self = Self::new(lim, vtree, vars)?;
        }
        Ok(())
    }

    /// Place `vars` in leaf order, refusing a variable the vtree does not
    /// carry and one that appears twice.
    fn new(lim: &Limits, vtree: &Arc<Vtree>, vars: &[VarId]) -> Result<Layout, OperationError> {
        let n = vtree.num_nodes();
        let mut count = Vec::new();
        let mut lo = Vec::new();
        let mut position = Vec::new();
        lim.try_resize(&mut count, n, 0u32)?;
        lim.try_resize(&mut lo, n, 0u32)?;
        lim.try_resize(&mut position, vars.len(), 0u32)?;

        for &var in vars {
            let leaf = vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
            if count[leaf.idx()] != 0 {
                return Err(OperationError::DuplicateVariable(var));
            }
            count[leaf.idx()] = 1;
        }
        for (t, left, right) in vtree.internal_bottomup() {
            count[t.idx()] = count[left.idx()] + count[right.idx()];
        }
        for t in vtree.bottomup().rev() {
            if !vtree.node(t).is_leaf() {
                let (left, right) = vtree.children(t);
                lo[left.idx()] = lo[t.idx()];
                lo[right.idx()] = lo[t.idx()] + count[left.idx()];
            }
        }
        for (i, &var) in vars.iter().enumerate() {
            let leaf = vtree.leaf_of(var).expect("checked above");
            position[i] = lo[leaf.idx()];
        }
        let mut owned_vars = Vec::new();
        lim.reserve_exact(&mut owned_vars, vars.len())?;
        owned_vars.extend_from_slice(vars);
        let identity = position.iter().enumerate().all(|(i, &p)| p as usize == i);
        Ok(Layout { vtree: Arc::downgrade(vtree), vars: owned_vars, identity, count, lo, position })
    }

    /// Whether `vars` was already given in leaf order, so that re-encoding a
    /// row is a copy.
    pub(super) fn is_identity(&self) -> bool {
        self.identity
    }
}

impl crate::limits::pool::Buffers for Layout {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::limits::pool::Scratch)) {
        visit(&mut self.vars);
        visit(&mut self.count);
        visit(&mut self.lo);
        visit(&mut self.position);
    }
}

impl crate::limits::pool::PooledScratch for Layout {
    fn prepare(&mut self) {}
    /// The buffers describe one placement together, so they are kept or
    /// released together: released past the byte cap or once the vtree is gone.
    fn retain(&mut self, lim: &Limits) {
        use crate::limits::pool::Buffers;
        if self.retained_bytes() > crate::limits::pool::SCRATCH_RETAIN_BYTES || self.vtree.strong_count() == 0 {
            self.release_all(lim);
            *self = Self::default();
        }
    }
}
