//! Representation-specialized child lookups for the conjunction inner loop.
//!
//! The dense emit path (`cell::process_cell`) resolves each child node by
//! a single point lookup into a flat per-level grid slab:
//! `node_idx[base + row * stride + col]`.
//!
//! This module abstracts that single lookup behind the [`ChildLookup`] trait so
//! the emit can be monomorphized per child representation:
//! - [`DenseLookup`] — the grid index. Holds no borrow (it takes the
//!   `node_idx` slab as a call argument) so it compiles back to the original
//!   `get_unchecked`, and so it never aliases the `&mut node_idx` the emit
//!   writes its output into.
//! - [`MarginalLookup`] — the marginal-aware Route A lookup (below).

/// Flat row offset `a * stride` of child node row `a` in a child grid whose rows
/// are `stride` columns wide.
///
/// Must be computed in `usize`: the child grid `node_idx` is allocated with
/// `usize` arithmetic (rows times columns), so a single child level can
/// legitimately exceed 2^32 cells. The operands `a` (`NodeIdx`, u32) and
/// `stride` (child column count, u32) would overflow a u32 multiply and silently
/// wrap — reading the wrong child node (→ wrong model count) or running off the
/// slab end (an out-of-bounds read). Widening each operand to `usize` before the multiply makes
/// the product exact on 64-bit targets at zero cost (`#[inline(always)]`).
#[inline(always)]
fn child_grid_mul(a: u32, stride: u32) -> usize {
    a as usize * stride as usize
}

/// Resolve a child node index at grid position `(row, col)`. `node_idx` is
/// the shared per-apply grid slab. Passing the slab per call rather than
/// holding a borrow keeps the dense impl free of an aliasing borrow against
/// the output writes into the same slab.
pub(super) trait ChildLookup {
    fn get(&self, node_idx: &[u32], row: u32, col: u32) -> u32;

    /// True when this side is a marginal pass-through carrier: `get` returns
    /// the carried operand field verbatim (an inline model count or tagged
    /// big-count slot), never a grid read, and the side is unconditionally
    /// alive (a marginal cofactor never kills a pair). The cell kernel gates
    /// its reach/liveness-mask culls on `!passthrough()`; the plain lookups
    /// return a constant `false`, so those guards fold away and the plain
    /// instantiations keep branch-free inner loops.
    #[inline(always)]
    fn passthrough(&self) -> bool {
        false
    }
}

/// Existing grid representation: flat `node_idx[base + row*stride + col]`.
pub(super) struct DenseLookup {
    pub(super) base: usize,
    pub(super) stride: u32,
}

impl ChildLookup for DenseLookup {
    #[inline(always)]
    fn get(&self, node_idx: &[u32], row: u32, col: u32) -> u32 {
        // Safety: callers only query positions within the child's
        // rows-by-columns slab. `child_grid_mul` widens before
        // the multiply so the index is exact on 64-bit targets.
        unsafe {
            *node_idx.get_unchecked(self.base + child_grid_mul(row, self.stride) + col as usize)
        }
    }
}

/// Marginal-aware child lookup for the Route A (≥1 marginal child) cell walk.
///
/// Off pass-through this is exactly a [`DenseLookup`] grid read. On a
/// pass-through side the raw operand field (`row` = the f pair field, `col` =
/// the g pair field) is carried verbatim — it is an inline model count or a
/// tagged big-count slot, not a grid coordinate, so the grid is never
/// consulted (indexing with it would read far out of bounds; see
/// `plan_marginal_level`). `passthrough` is a per-level runtime flag, so the marginal
/// inner loops pay one branch per access.
pub(super) struct MarginalLookup {
    base: usize,
    stride: u32,
    passthrough: bool,
    /// Carrier selector when `passthrough`: true ⇒ carry the f field (`row`).
    pt_c1: bool,
}

impl MarginalLookup {
    /// The lookup for one child side of a level.
    pub(super) fn new(p: &super::cell::ChildPlan<'_>) -> Self {
        Self {
            base: p.base,
            stride: p.stride,
            passthrough: p.plan.carrier.is_some(),
            pt_c1: matches!(p.plan.carrier, Some(super::marginal_plan::Carrier::F)),
        }
    }
}

impl ChildLookup for MarginalLookup {
    #[inline(always)]
    fn get(&self, node_idx: &[u32], row: u32, col: u32) -> u32 {
        if self.passthrough {
            if self.pt_c1 { row } else { col }
        } else {
            // Safety: identical access to `DenseLookup::get` — off pass-through
            // the fields are structural coordinates within the child's
            // rows-by-columns slab. `child_grid_mul` widens before the
            // multiply so the index is exact on 64-bit targets.
            unsafe {
                *node_idx.get_unchecked(self.base + child_grid_mul(row, self.stride) + col as usize)
            }
        }
    }

    #[inline(always)]
    fn passthrough(&self) -> bool {
        self.passthrough
    }
}

#[cfg(test)]
#[path = "child_lookup_tests.rs"]
mod tests;
