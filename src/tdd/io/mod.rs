//! Serialization of TDDs to text formats (output only).
//!
//! - **dot** — DOT/Graphviz rendering of vtrees and TDDs.
//! - **save** — TDD circuit serialization to the `.tdd` text format.

pub mod dot;
pub mod save;

#[cfg(test)]
mod tests;

use crate::tdd::types::Tdd;

/// Reject a diagram carrying any marginal level, for the writers that cannot
/// represent one.
///
/// Both output formats are *structural*: a pair names its children by local
/// node index. A marginal level stores per-node model COUNTS instead of nodes,
/// so a pair pointing into one carries an inline count rather than an index and
/// there is nothing faithful to emit for it. The writers refuse such a diagram
/// here instead of failing on the inline ref deep inside the emit loop.
///
/// Single detection point for `save_tdd`/`write_tdd` and `tdd_to_dot`; `what`
/// names the calling operation in the message.
pub(crate) fn reject_marginal_levels(tdd: &Tdd, what: &str) -> std::io::Result<()> {
    if tdd.has_marginal_level() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "{what}: the diagram has one or more marginal levels, which store per-node \
                 model counts rather than nodes and have no structural representation in this \
                 format. Serialize or render the diagram before marginalizing it."
            ),
        ));
    }
    Ok(())
}
