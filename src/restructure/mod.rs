//! Change a diagram's vtree while preserving its function.
//!
//! [`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for rotations
//! that improve a [`search::RotationObjective`], and
//! [`Tdd::rotate_if`](crate::Tdd::rotate_if) applies rotations one
//! sequence at a time under the caller's own decision.
//! [`Engine::transport`](crate::Engine::transport) moves a diagram onto the
//! projection of a home vtree on its renamed variables, and
//! [`Tdd::expand_variables`](crate::Tdd::expand_variables) expands each of its
//! variables into a class of new ones.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod embed;
pub(crate) mod expand;
mod distinct;
pub(crate) mod placement;
mod tag;
mod transport;
pub(crate) mod target;
#[cfg(test)]
#[path = "tests/leaf_form.rs"]
mod leaf_form_tests;

pub use crate::vtree::graft::Embedding;
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
    /// The destination holds a region the diagram has not summed out, so the
    /// copy has no value to give it. A conjunction with the diagram that
    /// summed the region ([`Engine::and`](crate::Engine::and)) is defined.
    UnsummedRegion {
        /// The region, a node of the destination.
        region: VtreeIdx,
    },
    /// An operation the copy runs was refused: a source that has discarded
    /// the structure at a level, a refused allocation, or an armed stop.
    Operation(OperationError),
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
            Self::UnsummedRegion { region } => write!(f, "destination node {} is a region the diagram has not summed out", region.idx()),
            Self::Operation(error) => write!(f, "copying the diagram: {error}"),
        }
    }
}

impl std::error::Error for EmbedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::VariableOutOfRange { .. } | Self::NotIsomorphic { .. } | Self::UnsummedRegion { .. } => None,
        }
    }
}

impl From<VtreeError> for EmbedError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for EmbedError {
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

/// The node [`Engine::at_least_distinct`](crate::Engine::at_least_distinct)
/// and [`Engine::project_to_subtree`](crate::Engine::project_to_subtree)
/// key on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistinctKey<'a> {
    /// A node of the diagram's [home](crate::Vtree::home). Any index below
    /// the home's node count is a home node, so
    /// an index of the diagram's own vtree is read as the home node with
    /// that number, without an error.
    Node(VtreeIdx),
    /// The node of the home whose leaves are exactly these variables, in any
    /// order. The same variables name the same node whichever projection the
    /// diagram is on.
    Vars(&'a [VarId]),
}

impl From<VtreeIdx> for DistinctKey<'_> {
    fn from(node: VtreeIdx) -> Self { Self::Node(node) }
}

impl<'a> From<&'a [VarId]> for DistinctKey<'a> {
    fn from(vars: &'a [VarId]) -> Self { Self::Vars(vars) }
}

/// Why the distinct values under a subtree could not be counted
/// ([`Engine::at_least_distinct`](crate::Engine::at_least_distinct)).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DistinctError {
    /// The key is the vtree's root, which has no sibling to count.
    Root,
    /// A [`DistinctKey::Vars`] whose variables, listed, are not exactly the
    /// leaves of one node: empty, repeated, a variable with no leaf, or not
    /// all of a subtree.
    Key(Vec<VarId>),
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
            Self::Key(vars) => {
                let vars: Vec<u32> = vars.iter().map(|v| v.0).collect();
                write!(f, "variables {vars:?} are not the leaves of one vtree node")
            }
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
            Self::Root | Self::Key(_) => None,
        }
    }
}

impl From<OperationError> for DistinctError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}
