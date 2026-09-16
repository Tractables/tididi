//! Errors from checked diagram operations.

/// An invalid operation input or a resource refusal from a checked operation.
///
/// Resource failures are [`OverBudget`](Self::OverBudget),
/// [`OutputCap`](Self::OutputCap), and [`Stopped`](Self::Stopped). The other
/// variants identify incompatible operands or invalid literals, variables and levels.
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
    /// A reservation failed, exceeded the byte budget, or needed more than a
    /// 32-bit index can address. [`OperationMetrics::refused_reserve_bytes`](crate::limits::OperationMetrics::refused_reserve_bytes)
    /// distinguishes recorded allocator refusals from budget refusals.
    OverBudget,
    /// The operands do not share the same vtree allocation.
    VtreeMismatch,
    /// Conjunction operands have outputs at different vtree nodes.
    RootMismatch,
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
    /// not returned on error. Display uses the one-based variable number.
    VariableNotInVtree(crate::vtree::VarId),
    /// An input cube or substitution map names the same source variable more than once.
    DuplicateVariable(crate::vtree::VarId),
}

impl std::fmt::Display for OperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationError::InvalidLiteral(literal) => write!(f, "{literal} is not a literal; use a nonzero signed integer"),
            OperationError::VtreeMismatch => f.write_str("operands must share the same vtree allocation"),
            OperationError::RootMismatch => f.write_str("conjunction operands must have the same output vtree node"),
            OperationError::IncompatibleWeights => f.write_str("operands require compatible literal weights and arithmetic; stored integer counts cannot be reweighted"),
            OperationError::LevelNotInVtree(level) => write!(f, "level {} is outside the vtree", level.idx()),
            OperationError::MarginalLevel(level) => write!(f, "operation requires structural data at marginal level {}", level.idx()),
            OperationError::OverBudget => f.write_str("memory allocation refused: budget, allocator, or capacity limit"),
            OperationError::Stopped => f.write_str("operation stopped"),
            OperationError::OutputCap => f.write_str("output node cap exceeded"),
            OperationError::DuplicateVariable(var) => write!(f, "input names variable x{} twice", u64::from(var.0) + 1),
            OperationError::VariableNotInVtree(var) => {
                write!(f, "variable x{} is not in the vtree", u64::from(var.0) + 1)
            }
        }
    }
}

impl std::error::Error for OperationError {}

/// Typed literals convert infallibly at the same boundary as checked integer inputs.
impl From<std::convert::Infallible> for OperationError {
    fn from(never: std::convert::Infallible) -> Self { match never {} }
}
