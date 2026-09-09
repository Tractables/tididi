//! Representation-specialized child lookups for the conjunction inner loop.
//!
//! The dense emit path (`cell::process_cell`) resolves each child node by
//! a single point lookup into a flat per-level grid slab:
//! `node_idx[base + row * right_width + col]`.
//!
//! This module abstracts that single lookup behind the [`ChildLookup`] trait so
//! the emit can be monomorphized per child representation:
//! - [`DenseLookup`] — the grid index. Holds no borrow (it takes the
//!   `node_idx` slab as a call argument) so it compiles back to the original
//!   `get_unchecked`, and so it never aliases the `&mut node_idx` the emit
//!   writes its output into.
//! - [`MargLookup`] — the marginal-aware Route A lookup (below).

/// Flat row offset `a * right_width` of child node row `a` in a `right_width`-column child grid.
///
/// MUST be computed in `usize`: the child grid `node_idx` is allocated with
/// `usize` arithmetic (`grid_end += k1 * right_width`), so a single child level can
/// legitimately exceed 2^32 cells. The operands `a` (`NodeIdx`, u32) and
/// `right_width` (child column count, u32) would overflow a u32 multiply and silently
/// wrap — reading the wrong child node (→ wrong model count) or running off the
/// slab end (→ OOB). Widening each operand to `usize` before the multiply makes
/// the product exact on 64-bit targets at zero cost (`#[inline(always)]`).
#[inline(always)]
fn child_grid_mul(a: u32, right_width: u32) -> usize {
    a as usize * right_width as usize
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

/// Existing grid representation: flat `node_idx[base + row*right_width + col]`.
pub(super) struct DenseLookup {
    pub(super) base: usize,
    pub(super) right_width: u32,
}

impl ChildLookup for DenseLookup {
    #[inline(always)]
    fn get(&self, node_idx: &[u32], row: u32, col: u32) -> u32 {
        // SAFETY: callers only query positions within the child's
        // [base, base + k1*right_width) slab — identical access to the historical
        // hand-written `get_unchecked`. `child_grid_mul` widens before the
        // multiply so the index is exact on 64-bit targets.
        unsafe {
            *node_idx.get_unchecked(self.base + child_grid_mul(row, self.right_width) + col as usize)
        }
    }
}

/// Marginal-aware child lookup for the Route A (≥1 marginal child) cell walk.
///
/// Off pass-through this is exactly a [`DenseLookup`] grid read. On a
/// pass-through side the raw operand field (`row` = the f pair field, `col` =
/// the g pair field) is carried verbatim — it is an inline model count or a
/// tagged big-count slot, NOT a grid coordinate, so the grid is never
/// consulted (indexing with it would read far out of bounds; see
/// `plan_marg_level`). `passthrough` is a per-level runtime flag, so the marg
/// inner loops pay one branch per access.
pub(super) struct MargLookup {
    base: usize,
    right_width: u32,
    passthrough: bool,
    /// Carrier selector when `passthrough`: true ⇒ carry the f field (`row`).
    pt_c1: bool,
}

impl MargLookup {
    /// The lookup for one child side of a level.
    pub(super) fn new(p: &super::cell::ChildPlan<'_>) -> Self {
        Self {
            base: p.base,
            right_width: p.right_width,
            passthrough: p.plan.carrier.is_some(),
            pt_c1: matches!(p.plan.carrier, Some(super::marg_plan::Carrier::F)),
        }
    }
}

impl ChildLookup for MargLookup {
    #[inline(always)]
    fn get(&self, node_idx: &[u32], row: u32, col: u32) -> u32 {
        if self.passthrough {
            if self.pt_c1 { row } else { col }
        } else {
            // SAFETY: identical access to `DenseLookup::get` — off pass-through
            // the fields are structural coordinates within the child's
            // [base, base + k1*right_width) slab. `child_grid_mul` widens before the
            // multiply so the index is exact on 64-bit targets.
            unsafe {
                *node_idx.get_unchecked(self.base + child_grid_mul(row, self.right_width) + col as usize)
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
