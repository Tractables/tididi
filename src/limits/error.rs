//! Errors from checked diagram operations.

/// An invalid operation input or a resource refusal from a checked operation.
///
/// Resource failures are [`OverBudget`](Self::OverBudget),
/// [`OutputCap`](Self::OutputCap), and [`Stopped`](Self::Stopped);
/// [`IndexOverflow`](Self::IndexOverflow) looks like one and is not, since no
/// budget makes it succeed. The other variants identify incompatible operands
/// or invalid literals, variables and levels.
/// Even an engine with no limits installed can report `OverBudget` when a
/// buffer reservation fails.
///
/// Recovery depends on how the operation takes its diagram. A consuming
/// operation such as [`and`](crate::and) drops its operands on
/// error; keep copies before the call if a retry needs them. A borrowed query
/// leaves its input unchanged. An in-place operation such as
/// [`Tdd::minimize`](crate::Tdd::minimize) or
/// [`Tdd::rotation_search`](crate::Tdd::rotation_search) documents which
/// completed edits remain after a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationError {
    /// A signed integer does not denote a literal; zero is invalid.
    InvalidLiteral(i32),
    /// A reservation failed or exceeded the byte budget.
    /// [`OperationMetrics::refused_reserve_bytes`](crate::limits::OperationMetrics::refused_reserve_bytes)
    /// distinguishes recorded allocator refusals from budget refusals. Raising
    /// the budget or splitting the work is what answers either; a structure too
    /// large to index is [`IndexOverflow`](Self::IndexOverflow) instead.
    OverBudget,
    /// An intermediate structure grew past what a 32-bit index can address.
    ///
    /// Not a resource condition: no budget makes it succeed, and retrying is
    /// pointless. Splitting the operation, or ordering the vtree so the level
    /// it happened at stays narrower, is what answers it.
    IndexOverflow,
    /// The operands do not share the same vtree allocation.
    VtreeMismatch,
    /// The operands use different literal weights or arithmetic, or mix weights with stored integer counts.
    IncompatibleWeights,
    /// A target level index is outside the operand's vtree.
    LevelNotInVtree(crate::vtree::VtreeIdx),
    /// The operation needs structure at a level that has already been summed out.
    MarginalLevel(crate::vtree::VtreeIdx),
    /// The installed deadline passed, or an installed schedule concluded that
    /// the operation should stop, at one of the operation's poll points.
    Stopped,
    /// The installed cap on produced output nodes tripped: a deliberate size
    /// cut, not an allocation failure. Conjunction, checked construction,
    /// structural projection, and care rebuilding check the cap against
    /// emitted nodes; compound operations propagate it.
    OutputCap,
    /// The operation names a variable the operand's vtree does not carry.
    /// Validation timing is stated on the operation; a consumed operand is
    /// not returned on error.
    VariableNotInVtree(crate::vtree::VarId),
    /// An input cube or substitution map names the same source variable more than once.
    DuplicateVariable(crate::vtree::VarId),
    /// An option was left without the bound the operation needs, so the work
    /// it would do has no size limit.
    ///
    /// Not a resource condition: nothing about the machine makes it succeed.
    /// Setting the named option is what answers it.
    UnboundedSearch {
        /// The option that has no bound.
        option: &'static str,
        /// What needs it bounded.
        needed_by: &'static str,
    },
    /// A diagram assembled or reweighted for the operation failed the storage
    /// checks of [`TddBuilder::finish`](crate::diagram::TddBuilder::finish).
    /// `?` on a [`TddBuildError`](crate::diagram::TddBuildError) produces it,
    /// except that its `IncompatibleWeights` becomes
    /// [`IncompatibleWeights`](Self::IncompatibleWeights).
    InvalidDiagram(crate::diagram::TddBuildError),
}

impl std::fmt::Display for OperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationError::InvalidLiteral(literal) => write!(f, "{literal} is not a literal; use a nonzero signed integer"),
            OperationError::VtreeMismatch => f.write_str("operands must share the same vtree allocation"),
            OperationError::IncompatibleWeights => f.write_str("operands require compatible literal weights and arithmetic; stored integer counts cannot be reweighted"),
            OperationError::LevelNotInVtree(level) => write!(f, "level {} is outside the vtree", level.idx()),
            OperationError::MarginalLevel(level) => write!(f, "operation requires structural data at marginal level {}", level.idx()),
            OperationError::OverBudget => f.write_str("memory allocation refused by the byte budget or the allocator"),
            OperationError::IndexOverflow => f.write_str("an intermediate structure outgrew the 32-bit index that addresses it"),
            OperationError::Stopped => f.write_str("operation stopped"),
            OperationError::OutputCap => f.write_str("output node cap exceeded"),
            OperationError::DuplicateVariable(var) => write!(f, "input names variable x{} twice", var.0),
            OperationError::UnboundedSearch { option, needed_by } => {
                write!(f, "{option} has no bound, which {needed_by} requires")
            }
            OperationError::InvalidDiagram(source) => write!(f, "invalid diagram: {source}"),
            OperationError::VariableNotInVtree(var) => {
                write!(f, "variable x{} is not in the vtree", var.0)
            }
        }
    }
}

impl std::error::Error for OperationError {
    /// The storage check that [`InvalidDiagram`](OperationError::InvalidDiagram)
    /// wraps. The other variants carry no inner error.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OperationError::InvalidDiagram(source) => Some(source),
            _ => None,
        }
    }
}

/// Typed literals convert infallibly at the same boundary as checked integer inputs.
impl From<std::convert::Infallible> for OperationError {
    fn from(never: std::convert::Infallible) -> Self { match never {} }
}

/// Storage checks fail at the same boundary as invalid operation inputs.
impl From<crate::diagram::TddBuildError> for OperationError {
    fn from(source: crate::diagram::TddBuildError) -> Self {
        match source {
            crate::diagram::TddBuildError::IncompatibleWeights => OperationError::IncompatibleWeights,
            other => OperationError::InvalidDiagram(other),
        }
    }
}
