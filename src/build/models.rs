//! Canonical construction from a set of assignments to some of the vtree's
//! variables.
//!
//! The diagram has one node per *atom*. At a vtree node `v`, two values of
//! the variables in `v`'s subtree belong to the same atom when the rows
//! extend them by the same set of assignments to the variables outside that
//! subtree. Atoms are therefore disjoint sets of values, which is what makes
//! the nodes at a level pairwise mutex; grouping the values by their own
//! subfunction instead would produce overlapping nodes that no reduction
//! pass can repair.
//!
//! The atoms are computed from the root down (see `split`), each node from
//! its parent's distinct values, and the nodes are then stored bottom-up,
//! each level's in the order of their least pairs (see `emit`).

use std::sync::Arc;

use crate::diagram::{Assembly, NodeIdx, Tdd};
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

mod columns;
mod cubes;
mod rows;
mod split;
mod emit;
pub(crate) mod layout;
pub use columns::RowSelection;
use layout::Layout;
use rows::{words_per_row, distinct_rows};
use split::Direct;

impl Tdd {
    /// The canonical diagram whose models are exactly `rows`, read as
    /// assignments to `vars`, with every other variable of `vtree` free.
    ///
    /// Rows are bit-packed, `vars.len().div_ceil(64).max(1)` words each: with
    /// `w` words per row, row `k` occupies `rows[k * w .. (k + 1) * w]`, and
    /// bit `i` of that row — bit `i % 64` of word `i / 64` — is the value
    /// assigned to `vars[i]`. Bits at or past `vars.len()` are ignored, and a
    /// repeated row denotes one model.
    ///
    /// Empty `rows` gives the constant-false diagram, and an empty `vars` with
    /// at least one row gives the constant-true diagram: the same rule read at
    /// the degenerate size, where a row is one word of ignored bits.
    ///
    /// The result is canonical for `vtree`, so no
    /// [`minimize`](Self::minimize) step follows it, unlike a diagram
    /// assembled through [`TddBuilder`](crate::diagram::TddBuilder). Each
    /// level numbers its nodes by their least `(right child, left child)`
    /// pair. Reordering or repeating the input rows preserves this numbering.
    ///
    /// The row sort compares the vtree's leaves from left to right, a false
    /// before a true, so rows that already come in that order are only
    /// checked: with `vars` in leaf order, a table sorted on columns laid out
    /// left to right, each code's bits most significant first. Temporary
    /// storage includes the packed rows, the distinct values of the nodes
    /// still to be split, sorting scratch and every level's pairs.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let vars = [VarId(1), VarId(2), VarId(3)];
    /// // Bit 0 is read access, bit 1 write access, bit 2 sharing.
    /// let rows = [0b001, 0b011, 0b101, 0b011];
    /// let f = Tdd::from_models(&vtree, &vars, &rows)?;
    /// println!("Distinct permission sets: {}", f.model_count()?);
    /// # assert_eq!(f.model_count()?, 3u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a variable of `vars` that is
    /// not a leaf of `vtree`, [`OperationError::DuplicateVariable`] for a
    /// repeated one, [`OperationError::RaggedRows`] when `rows.len()` is not a
    /// multiple of the words per row, [`OperationError::OverBudget`] for a
    /// refused allocation, [`OperationError::IndexOverflow`] when a level would
    /// outgrow the index that addresses it, and [`OperationError::Stopped`]
    /// when an armed stop fires.
    pub fn from_models(
        vtree: &Arc<Vtree>,
        vars: &[VarId],
        rows: &[u64],
    ) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.from_models(vtree, vars, rows))
    }
}

impl Engine {
    /// Run [`Tdd::from_models`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::from_models`].
    pub fn from_models(
        &self,
        vtree: &Arc<Vtree>,
        vars: &[VarId],
        rows: &[u64],
    ) -> Result<Tdd, OperationError> {
        self.models_built(vtree, vars, rows, Direct::Off)
    }

    /// [`from_models`](Self::from_models) with buffer reuse and hashing for
    /// wide values that repeat often. Produces the same diagram, including
    /// node numbering and pair order.
    ///
    /// This route can avoid sorting a wide field with few distinct values
    /// and copying pairs at a level with one node and two internal children.
    /// The benefit depends on the rows and vtree; compare both constructors
    /// on representative inputs before selecting it for repeated builds.
    ///
    /// # Errors
    ///
    /// As [`Tdd::from_models`].
    pub fn from_models_direct(
        &self,
        vtree: &Arc<Vtree>,
        vars: &[VarId],
        rows: &[u64],
    ) -> Result<Tdd, OperationError> {
        self.models_built(vtree, vars, rows, Direct::On)
    }

    /// The build behind both entry points.
    fn models_built(
        &self,
        vtree: &Arc<Vtree>,
        vars: &[VarId],
        rows: &[u64],
        direct: Direct,
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let w = words_per_row(vars.len());
        if !rows.len().is_multiple_of(w) {
            return Err(OperationError::RaggedRows { words: rows.len(), per_row: w });
        }

        let mut layout = self.scratch.model_layout.checkout(self);
        layout.prepare_for(lim, vtree, vars)?;
        if rows.is_empty() || vars.is_empty() {
            return super::constant_on(self, vtree, !rows.is_empty());
        }
        if rows.len() / w > NodeIdx::MAX_LIVE {
            // A level holds at most one node per row, and the pass indexes the
            // rows with the same width the level's nodes are indexed with.
            return Err(OperationError::IndexOverflow);
        }

        // The split sorts through the same buffers the rows did.
        let mut radix = crate::sort::Radix::default();
        let sorted = distinct_rows(lim, &mut radix, vars.len(), &layout, rows, w)?;
        build_sorted(self, vtree, &layout, sorted, w, radix, direct)
    }
}

/// The diagram of `sorted`, distinct rows in `layout`'s order, `w` words
/// each, built with the radix sort's buffers `radix`, the split writing
/// pairs as `direct` says.
///
/// After the row sort, construction walks the vtree from the root down,
/// splitting each node's distinct values into its children's, and finally
/// stores the nodes bottom-up. A node's work is proportional to
/// its parent's distinct values rather than to the rows, plus the
/// numbering of its own values when it is a right child. Where a value
/// is narrow enough to index a table, one pass in the parent's order
/// hashes them, with no sort; at the root it is a radix sort on the
/// value alone. Otherwise it is a sort by value: a radix sort of at most
/// three passes while that value and the value's index fit one word, and
/// beyond that one counting pass on the top digit followed by a
/// comparison sort of each run the digit leaves.
fn build_sorted(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    layout: &Layout,
    sorted: std::borrow::Cow<'_, [u64]>,
    w: usize,
    radix: crate::sort::Radix,
    direct: Direct,
) -> Result<Tdd, OperationError> {
    let mut plans = split::plan(eng.limits(), vtree, layout, sorted, w, radix, direct)?;
    let mut assembly = Assembly::new(eng, vtree)?;
    let output = emit::fill(eng, &mut assembly, vtree, layout, &mut plans)?;
    // The levels are canonical as built: seat them with nothing to reduce.
    let (levels, _) = assembly.parts_mut();
    Ok(super::seat_canonical(eng, vtree, std::mem::take(levels), output))
}

#[cfg(test)]
#[path = "tests/models.rs"]
mod tests;
