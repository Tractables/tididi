//! Change a diagram's vtree while preserving its function.
//!
//! [`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for rotations
//! that improve a [`search::RotationObjective`], and
//! [`Tdd::rotate_if`](crate::Tdd::rotate_if) applies rotations one
//! sequence at a time under the caller's own decision.
//! [`Tdd::graft`](crate::Tdd::graft) joins diagrams over disjoint variables on
//! a grafted vtree. [`Tdd::embed`](crate::Tdd::embed) copies one diagram onto a
//! larger vtree under a renaming of its variables,
//! [`Engine::transport`](crate::Engine::transport) moves one onto the
//! projection of a home vtree on its renamed variables, and
//! [`Tdd::expand_variables`](crate::Tdd::expand_variables) expands each of its
//! variables into a class of new ones.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod embed;
pub(crate) mod expand;
pub(crate) mod graft;
mod distinct;
pub(crate) mod placement;
mod project;
mod tag;
mod splice;
mod transport;
pub(crate) mod target;
#[cfg(test)]
#[path = "tests/leaf_form.rs"]
mod leaf_form_tests;

pub use crate::vtree::graft::Embedding;
pub use embed::EmbeddingPlan;
pub use expand::VariableExpansion;
pub use target::TransportStats;

use crate::diagram::TddBuildError;
use crate::limits::OperationError;
use crate::vtree::{VarId, VtreeError, VtreeIdx};

/// Why a diagram could not be moved onto the requested vtree.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RestructureError {
    /// The two vtrees do not have the same variables: this one is a leaf of
    /// one of them only.
    Variables {
        /// The variable only one vtree has.
        variable: VarId,
    },
    /// The diagram has a weight table, which the move does not carry.
    Weighted,
    /// A rotation at this vtree node would have expanded more pairs than the
    /// bound allows.
    Bound {
        /// The node the refused rotation turns.
        pivot: VtreeIdx,
    },
    /// An operation the move runs was refused: a rotation that would turn a
    /// summed-out level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for RestructureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Variables { variable } => write!(f, "variable {} is a leaf of only one of the two vtrees", variable.0),
            Self::Weighted => write!(f, "a weighted diagram cannot be moved to another vtree"),
            Self::Bound { pivot } => write!(f, "the rotation at vtree node {} exceeds the pair bound", pivot.idx()),
            Self::Operation(error) => write!(f, "moving the diagram: {error}"),
        }
    }
}

impl std::error::Error for RestructureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::Variables { .. } | Self::Weighted | Self::Bound { .. } => None,
        }
    }
}

impl From<OperationError> for RestructureError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why a diagram could not be copied onto the requested vtree.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmbedError {
    /// The renamed variables cannot form a vtree: two of them share an image.
    Vtree(VtreeError),
    /// A renamed variable is not a leaf of the destination.
    VariableOutOfRange {
        /// The destination variable.
        variable: VarId,
        /// The largest variable id the destination's id space holds.
        num_vars: u32,
    },
    /// The destination vtree does not contain the source vtree's shape under
    /// the renaming, so no level-by-level copy exists.
    NotIsomorphic {
        /// The source vtree node the destination has no image of: the
        /// parent of the source node met on the left where the destination
        /// joins it to a node that is not its sibling, or to its sibling in
        /// the other order where the match is exact.
        source: VtreeIdx,
    },
    /// The diagram's stored values cannot be interpreted in the destination:
    /// a weight-marginal level without a destination store, or a destination
    /// table that disagrees with the diagram's under the renaming.
    SourceWeights(TddBuildError),
    /// The destination weight table does not cover the result's variables or
    /// columns.
    DestinationWeights(TddBuildError),
    /// A projection of a diagram with weighted marginal values drops a
    /// variable that no marginal level it keeps sums over, which would take
    /// the variable's weights out of the diagram's value.
    Unsummed {
        /// The dropped variable.
        variable: VarId,
    },
    /// An operation the copy runs was refused: a source that has discarded
    /// the structure at a level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

/// An embedding [`Engine::embed_moving`](crate::Engine::embed_moving)
/// refused, with the diagram as it was given.
#[derive(Debug)]
pub struct EmbedRefused {
    /// Why the embedding was refused.
    pub error: EmbedError,
    /// The diagram, unchanged.
    pub tdd: crate::Tdd,
}

/// A transport [`Engine::transport`](crate::Engine::transport) refused, with
/// the diagram as it was given: an owned one unchanged, a borrowed one as the
/// borrow.
#[derive(Debug)]
pub struct TransportRefused<'a> {
    /// Why the transport was refused.
    pub error: TransportError,
    /// The diagram, unchanged.
    pub tdd: std::borrow::Cow<'a, crate::Tdd>,
}

/// Why [`Engine::transport`](crate::Engine::transport) refused a move.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransportError {
    /// An image that is not a leaf of the destination home.
    Variable {
        /// The image.
        variable: VarId,
    },
    /// Two variables of the diagram have this image.
    Overlap {
        /// The shared image.
        variable: VarId,
    },
    /// A rotation is needed at this node of the diagram's vtree and the bound
    /// does not allow it.
    Bound {
        /// The node of the diagram's vtree.
        pivot: VtreeIdx,
    },
    /// The diagram's weight table does not cover the images, or does not
    /// weigh each image as the variable it renames.
    Weights(TddBuildError),
    /// An operation the move runs was refused: a summed-out level, a refused
    /// allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Variable { variable } => write!(f, "variable {} is not a leaf of the destination home", variable.0),
            Self::Overlap { variable } => write!(f, "two variables are renamed to variable {}", variable.0),
            Self::Bound { pivot } => write!(f, "a rotation at vtree node {} is needed and not allowed by the bound", pivot.idx()),
            Self::Weights(error) => write!(f, "transported weights: {error}"),
            Self::Operation(error) => write!(f, "transporting the diagram: {error}"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Weights(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Variable { .. } | Self::Overlap { .. } | Self::Bound { .. } => None,
        }
    }
}

impl From<OperationError> for TransportError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "embedding: {error}"),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "renamed variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::NotIsomorphic { source } => write!(f, "the destination vtree does not contain the source vtree's shape at node {}", source.idx()),
            Self::SourceWeights(error) => write!(f, "embedded diagram: {error}"),
            Self::DestinationWeights(error) => write!(f, "embedding destination: {error}"),
            Self::Unsummed { variable } => write!(f, "variable {} is dropped outside every marginal level that sums over it", variable.0),
            Self::Operation(error) => write!(f, "copying the diagram: {error}"),
        }
    }
}

impl std::error::Error for EmbedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::SourceWeights(error) | Self::DestinationWeights(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::VariableOutOfRange { .. } | Self::NotIsomorphic { .. } | Self::Unsummed { .. } => None,
        }
    }
}

impl From<VtreeError> for EmbedError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for EmbedError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why diagrams could not be joined on a grafted vtree with their structure
/// and weight interpretation preserved.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraftError {
    /// The renamed variable sets cannot form a vtree.
    Vtree(VtreeError),
    /// A part's map has no entry for one of its variables.
    MissingVariableMapping {
        /// The part's position in the input vector.
        part: usize,
        /// The unmapped local variable.
        variable: VarId,
    },
    /// A renamed or free variable is outside the destination's id space.
    VariableOutOfRange {
        /// The destination variable.
        variable: VarId,
        /// The largest variable id the destination's id space holds.
        num_vars: u32,
    },
    /// A part's stored values cannot be interpreted in the requested destination.
    PartWeights {
        /// The part's position in the input vector.
        part: usize,
        /// The inconsistent weight or marginal-level state.
        source: TddBuildError,
    },
    /// The destination weight table does not cover the result's variables or columns.
    DestinationWeights(TddBuildError),
    /// An operation the assembly runs was refused: an operand that has
    /// discarded the structure at a level, a refused allocation, or an armed
    /// stop.
    Operation(OperationError),
}

impl std::fmt::Display for GraftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "grafted vtree: {error}"),
            Self::MissingVariableMapping { part, variable } => write!(f, "part {part} has no mapping for variable {}", variable.0),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "grafted variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::PartWeights { part, source } => write!(f, "part {part}: {source}"),
            Self::DestinationWeights(error) => write!(f, "graft destination: {error}"),
            Self::Operation(error) => write!(f, "assembling the result: {error}"),
        }
    }
}

impl std::error::Error for GraftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::PartWeights { source, .. } | Self::DestinationWeights(source) => Some(source),
            Self::Operation(error) => Some(error),
            Self::MissingVariableMapping { .. } | Self::VariableOutOfRange { .. } => None,
        }
    }
}

impl From<VtreeError> for GraftError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for GraftError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why a diagram could not be expanded as [`Tdd::expand_variables`](crate::Tdd::expand_variables) asks.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExpandError {
    /// The expanded variables cannot form a vtree: one is named twice, or
    /// their id space is too wide for the result.
    Vtree(VtreeError),
    /// A variable of the diagram's vtree has no class, or an empty one.
    MissingClass {
        /// The variable without a class.
        variable: VarId,
    },
    /// A variable the diagram's vtree does not carry has a nonempty class.
    ClassWithoutLeaf {
        /// The variable the class was given for.
        variable: VarId,
    },
    /// An expanded variable is outside the expansion's id space.
    VariableOutOfRange {
        /// The expanded variable.
        variable: VarId,
        /// The largest variable id the id space holds.
        num_vars: u32,
    },
    /// An operation the expansion runs was refused: a diagram that has summed
    /// out a level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "expanded vtree: {error}"),
            Self::MissingClass { variable } => write!(f, "variable {} of the diagram has no class", variable.0),
            Self::ClassWithoutLeaf { variable } => write!(f, "variable {} has a class but is not a leaf of the diagram's vtree", variable.0),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "expanded variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::Operation(error) => write!(f, "expanding the diagram: {error}"),
        }
    }
}

impl std::error::Error for ExpandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::MissingClass { .. } | Self::ClassWithoutLeaf { .. } | Self::VariableOutOfRange { .. } => None,
        }
    }
}

impl From<VtreeError> for ExpandError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for ExpandError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why the nodes of a level could not be tagged with codes
/// ([`Engine::tag_level`](crate::Engine::tag_level)).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TagError {
    /// The destination's root has no left subtree holding the level's
    /// subtree under the renaming, as an embedding reports it.
    Placement(EmbedError),
    /// The code variables are not exactly the leaves of the destination
    /// root's right subtree.
    CodeVariables(VtreeError),
    /// The tags or the codes do not fit the level: a tag count other than
    /// the level's width, a tag naming no code, a code list that is not a
    /// whole number of codes, or overlapping leaf slots tagged.
    Tags(String),
    /// An operation the construction runs was refused: a diagram that has
    /// discarded the structure at a level, a level that is not in the
    /// vtree, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for TagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement(error) => write!(f, "tagged level: {error}"),
            Self::CodeVariables(error) => write!(f, "code variables: {error}"),
            Self::Tags(message) => write!(f, "tags: {message}"),
            Self::Operation(error) => write!(f, "tagging the level: {error}"),
        }
    }
}

impl std::error::Error for TagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Placement(error) => Some(error),
            Self::CodeVariables(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Tags(_) => None,
        }
    }
}

impl From<OperationError> for TagError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why the distinct values under a subtree could not be counted
/// ([`Engine::at_least_distinct`](crate::Engine::at_least_distinct)).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DistinctError {
    /// The key is the vtree's root, which has no sibling to count.
    Root,
    /// The result's vtree cannot keep the diagram's variable ids: with
    /// [`VtreeError::VariableSpaceTooLarge`], the id space is wider than a
    /// vtree of the key's subtree may declare.
    Vtree(VtreeError),
    /// An operation the pass runs was refused: a key that is not in the
    /// vtree, a diagram that has discarded the structure at a level, a
    /// refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for DistinctError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Root => write!(f, "the key is the vtree's root, which has no sibling to count"),
            Self::Vtree(error) => write!(f, "the result's vtree: {error}"),
            Self::Operation(error) => write!(f, "counting distinct values: {error}"),
        }
    }
}

impl std::error::Error for DistinctError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::Vtree(error) => Some(error),
            Self::Root => None,
        }
    }
}

impl From<OperationError> for DistinctError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}
