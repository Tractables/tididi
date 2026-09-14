//! Save and restore structural diagrams, or render them as Graphviz text.
//!
//! Use [`write_tdd`] and [`read_tdd`] with streams, or [`save_tdd`] and [`load_tdd`]
//! with paths. Save the vtree separately using [`Vtree::to_text`](crate::Vtree::to_text)
//! and restore it with [`Vtree::from_text`](crate::Vtree::from_text).
//! A loaded diagram shares the vtree allocation passed to its reader.
//!
//! The format preserves Boolean structure, dropping unreachable nodes and
//! renumbering the rest. It does not store literal weights or marginal values;
//! attach weights again after reading, and serialize before marginalizing.
//! [`tdd_to_dot`] has the same structural requirement; [`vtree_to_dot`] renders
//! just the variable tree.
//!
//! # Text format
//!
//! The `.tdd` format is whitespace-separated, with one record per line:
//!
//! ```text
//! c <comment>
//! p tdd <version> <num_leaves> <num_vtree_nodes> <out_vtree> <out_local>
//! L <vtree_idx> <var>
//! I <vtree_idx> <left_vtree> <right_vtree> <l0> <r0> [<l1> <r1> ...]
//! ```
//!
//! The `p` line comes before node records. Writers emit version `1`;
//! `out_vtree` is the vtree root and `out_local` is its output node's local index.
//! For the false diagram, `out_local` is the token `ZERO` and no node records
//! follow. A nonfalse diagram declares every vtree leaf once with an `L` line,
//! where `var` is a one-based variable number.
//!
//! Each `I` record defines a node as a disjunction of pairs. The pair sides are
//! zero-based local indices into the declared left and right child levels.
//! Writers emit children before parents; readers validate references after all
//! records are read. Leaf indices are implicit: `0` means true,
//! `1` the positive literal, and `2` the negative literal. Internal indices are
//! assigned in the order the `I` records for that level appear.
//!
//! For example, this is `x1 ∧ ¬x2` over `Vtree::balanced(2)`:
//!
//! ```text
//! p tdd 1 2 3 2 0
//! L 0 1
//! L 1 2
//! I 2 0 1 1 2
//! ```
//!
//! Blank lines and `c` comments are ignored. Fixed-length records have no trailing
//! fields; an internal record has at least one complete pair. [`read_tdd`]
//! documents malformed-input errors and the supplied vtree's role.
//!
//! # Format versions
//!
//! The version belongs to the file format, independently of the crate version.
//! A reader rejects an absent or unsupported version rather than guessing a
//! layout. The current reader supports version `1`; later format revisions must
//! retain readers for earlier supported versions. Adding comments does not
//! change the version.

pub(crate) mod dot;
pub(crate) mod error;
pub(crate) mod read;
pub(crate) mod write;

pub use dot::{tdd_to_dot, vtree_to_dot};
pub use error::IoError;
pub use read::{load_tdd, read_tdd};
pub use write::{save_tdd, write_tdd};

#[cfg(test)]
mod tests;

use crate::diagram::Tdd;

/// The `.tdd` format version the writers here emit and the highest one the
/// reader here accepts.
///
/// A file written by version n loads in every reader whose version is n or
/// greater, so this number rises only when the records change in a way an older
/// reader would misread. Adding a comment line is not such a change; adding a
/// record letter is.
const TDD_FORMAT_VERSION: u32 = 1;

/// Reject a diagram carrying any marginal level, for the writers that cannot
/// represent one.
///
/// Both output formats are *structural*: a pair names its children by local
/// node index. A marginal level stores per-node model counts instead of nodes,
/// so a pair pointing into one carries an inline count rather than an index and
/// there is nothing faithful to emit for it. The writers refuse such a diagram
/// here instead of failing on the inline ref deep inside the emit loop.
///
/// Single detection point for `save_tdd`/`write_tdd` and `tdd_to_dot`; `what`
/// names the calling operation in the message.
pub(crate) fn reject_marginal_levels(tdd: &Tdd, what: &str) -> Result<(), IoError> {
    if tdd.has_marginal_level() {
        return Err(IoError::Format(format!(
            "{what}: the diagram has one or more marginal levels, which store per-node \
             model counts rather than nodes and have no structural representation in this \
             format. Serialize or render the diagram before marginalizing it."
        )));
    }
    Ok(())
}
