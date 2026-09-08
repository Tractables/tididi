//! The error type every vtree construction, parse and check reports through.

use super::VarId;
use std::fmt;

/// Why a vtree could not be built, parsed, or checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VtreeError {
    /// `.vtree` text that does not describe a single tree.
    Text(String),
    /// Two of the trees being combined both carry this variable.
    OverlappingVariable(VarId),
    /// A structural invariant that does not hold (see [`Vtree::validate`]),
    /// or a construction handed nothing to build from.
    Invalid(String),
}

impl fmt::Display for VtreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VtreeError::Text(msg) => write!(f, "malformed vtree text: {msg}"),
            VtreeError::OverlappingVariable(var) => write!(
                f,
                "variable {} is carried by more than one of the trees being combined",
                var.0 + 1
            ),
            VtreeError::Invalid(msg) => write!(f, "invalid vtree: {msg}"),
        }
    }
}

impl std::error::Error for VtreeError {}
