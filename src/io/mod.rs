//! Save and restore structural diagrams, or render them as Graphviz text.
//!
//! Use [`write_tdd`] and [`read_tdd`] with streams, or [`save_tdd`] and [`load_tdd`]
//! with paths. Save the vtree separately using [`Vtree::to_text`](crate::Vtree::to_text)
//! and restore it with [`Vtree::from_text`](crate::Vtree::from_text).
//! A loaded diagram shares the vtree allocation passed to its reader.
//! [`write_tdd_binary`], [`read_tdd_binary`], [`save_tdd_binary`] and
//! [`load_tdd_binary`] do the same in the [binary format](#binary-format), which
//! is smaller and is read without parsing text.
//!
//! Both formats preserve Boolean structure, dropping unreachable nodes and
//! renumbering the rest. They do not store literal weights or marginal values;
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
//! where `var` is a one-based variable number. Vtree IDs are file-local labels
//! in `0..num_vtree_nodes`. The writer uses the same labels as [`Vtree::to_text`](crate::Vtree::to_text);
//! the reader matches leaf variables and child relationships to the supplied
//! vtree, regardless of its in-memory numbering.
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
//! # Binary format
//!
//! A binary file stores what a text file stores, in this layout. Fixed-width
//! integers are little-endian.
//!
//! ```text
//! magic        8 bytes   89 54 44 44 0D 0A 1A 0A
//! version      u32       1, or 2 for a file that holds level counts
//! byte order   u32       0x0A0B0C0D
//! body length  u64       bytes between the header and the checksum
//! body
//! checksum     u64       XXH64, seed 0, of every byte before it
//! ```
//!
//! The body is a sequence of unsigned `LEB128` varints and bit-packed runs:
//!
//! 1. The leaf count and the node count of the vtree, then one varint per vtree
//!    node in postorder from the root (left subtree, right subtree, node): `0`
//!    for an internal node and `v` for the leaf of variable `v`. This sequence
//!    determines the tree.
//! 2. The output: `0` for the false diagram, which ends the body; otherwise one
//!    more than the output's local index in the root level.
//! 3. One section per internal vtree node, in the same postorder, so a level
//!    comes after both of its children: its node count, the count of its nodes
//!    with more than one pair, its pair count and the byte length of its
//!    stream of pairs. When some node has more than one pair, a bitmap follows
//!    with one bit per node, set for those nodes, and then one varint per such
//!    node, its pair count minus two. Then comes the stream, node by node in
//!    the order of their nodes: a node of one pair as its left-child index
//!    followed by its right-child index, and a node of more pairs as a two-bit
//!    code followed by its pairs, in their order, in that code:
//!    - `0`: each pair as a one-pair node's;
//!    - `1`: the smallest right index, the width `h` of the offsets in five
//!      bits, then each pair as its left index and its right index less the
//!      smallest, in `h` bits;
//!    - `2`, for left indices that ascend: the first left index, the width `g`
//!      of the gaps in five bits, the smallest right index and `h` as in `1`,
//!      the first pair's offset, then each later pair as its left index less
//!      the previous one's less one, in `g` bits, and its offset.
//!
//!    An index into a level of `w` nodes takes `ceil(log2(w))` bits, none when
//!    `w` is 1; a leaf level has three, numbered as in the text format. A
//!    writer uses a code only where each pair takes at least one bit, and a
//!    reader refuses one that does not.
//! 4. In version 2 only, the level counts the diagram keeps
//!    ([`Engine::attach_level_counts`](crate::Engine::attach_level_counts)):
//!    one record per internal vtree node in the same postorder, `0` for a
//!    level written without counts, or `1` followed by one varint per node of
//!    the level, in its order, the node's model count over the variables
//!    under the level. A writer emits version 2 for a diagram that keeps
//!    them, and leaves out a level with a count past `u128`. A reader checks
//!    each count against the assignments of its level's variables and keeps
//!    the counts with the diagram as written: the checksum guards them, but
//!    confirming them would be the fold that computes them.
//!
//! Bitmaps and streams are written least significant bit first and padded
//! with zero bits to a whole byte. Local indices are those the text format
//! assigns: a level's nodes are numbered in the order their section lists them.
//! A reader checks the magic, the version, the byte order, the length and the
//! checksum before reading the body; [`read_tdd_binary`] lists the checks that
//! follow.
//!
//! # Format versions
//!
//! The version belongs to the file format, independently of the crate version,
//! and each format numbers its own versions. A reader rejects an absent or
//! unsupported version rather than guessing a layout. The current readers
//! support version `1` of each format, and the binary reader version `2` as
//! well; later format revisions must retain readers for earlier supported
//! versions. Adding comments does not change the
//! text format's version.

mod binary;
mod numbering;
mod dot;
mod read;
mod write;

pub use binary::{load_tdd_binary, read_tdd_binary, save_tdd_binary, write_tdd_binary};
pub use dot::{tdd_to_dot, vtree_to_dot};
pub use read::{load_tdd, read_tdd};
pub use write::{save_tdd, write_tdd};

/// An I/O failure or invalid diagram data.
///
/// Match [`IoError::Io`] for stream failures and [`IoError::Format`] for
/// unsupported or malformed data. Format messages provide context for the
/// user; use the variant, rather than the message text, to handle an error.
#[derive(Debug)]
#[non_exhaustive]
pub enum IoError {
    /// The underlying file or stream failed.
    Io(std::io::Error),
    /// The data cannot be represented or read in the requested format.
    ///
    /// Writing: the diagram has a marginal level, which stores per-node
    /// values rather than nodes and has no structural form to emit. Reading: a
    /// record is malformed, references a node that does not exist, or
    /// contradicts the vtree the caller supplied. The message names the line
    /// (text) or the field and level (binary) and what was expected; a binary
    /// file whose header or checksum is wrong is refused here before any of
    /// it is parsed.
    Format(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "{e}"),
            IoError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IoError::Io(e) => Some(e),
            IoError::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Io(e)
    }
}

#[cfg(test)]
mod tests;

use crate::diagram::Tdd;

/// The `.tdd` format version emitted by the writer and accepted by the reader.
const TDD_FORMAT_VERSION: u32 = 1;

/// Reject marginal values that structural serialization and rendering cannot represent.
fn reject_marginal_levels(tdd: &Tdd, what: &str) -> Result<(), IoError> {
    if tdd.has_marginal_level() {
        return Err(IoError::Format(format!(
            "{what}: the diagram has one or more marginal levels, which store per-node \
             values rather than nodes and have no structural representation in this \
             format. Serialize or render the diagram before marginalizing it."
        )));
    }
    Ok(())
}
