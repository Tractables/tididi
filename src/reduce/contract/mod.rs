//! Twin contraction phase: merge diagram nodes with identical parent contexts.
//!
//! Two nodes at the same vtree level are "twins" if they appear in exactly
//! the same positions in parent input pairs (same parent node, same sibling partner).
//! Since twins always co-occur, their functions can be disjoined into a single
//! node without changing the diagram's overall function.

pub(crate) mod scratch;
mod fingerprint;
mod sweep;
mod merge;
mod duplicate_pair; // duplicate-pair scaling and resolution
pub(crate) mod contract_leaf; // twin contraction at leaf-adjacent levels
pub(crate) mod content_twin; // content-twin merge over every explicit level
pub(crate) mod pair_fusion; // same-structural-side pair fusion

pub(crate) use sweep::contract_all_twins;

#[cfg(test)]
mod tests;
