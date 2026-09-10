//! The `.tdd` text format, both directions, and Graphviz rendering.
//!
//! Both formats are structural: a pair names its two children by index, so a
//! diagram with a summed-out level ([`crate::marginal`]) has no encoding here
//! and is refused. The `.vtree` text format belongs to [`crate::vtree`].
//!
//! Entry points: [`save_tdd`] and [`write_tdd`] out, [`load_tdd`] and
//! [`read_tdd`] back in, [`tdd_to_dot`] and [`vtree_to_dot`] for looking at one.
//! [`IoError`] is what any of them reports.

pub(crate) mod dot;
pub(crate) mod error;
pub(crate) mod tdd_format;

pub use dot::{tdd_to_dot, vtree_to_dot};
pub use error::IoError;
pub use tdd_format::{load_tdd, read_tdd, save_tdd, write_tdd};

#[cfg(test)]
mod tests;

use crate::diagram::Tdd;

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
