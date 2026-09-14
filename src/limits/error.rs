//! The error types.
//!
//! One flat enum, [`OperationError`], for every operation that runs under a limit.
//! The domain errors belong to the modules that raise them:
//! [`VtreeError`](crate::vtree::VtreeError) to [`crate::vtree`],
//! [`IoError`](crate::io::IoError) to [`crate::io`], and
//! [`TddBuildError`](crate::diagram::TddBuildError) to [`crate::diagram`].
//!
//! Entry point: [`OperationError`], returned by every fallible
//! [`Engine`](crate::Engine) method and by [`crate::marginal::marginalize_levels`] and
//! [`crate::reduce::try_reduce`].

/// An invalid operation input or a resource refusal from a checked operation.
///
/// Resource failures are [`OverBudget`](Self::OverBudget),
/// [`OutputCap`](Self::OutputCap), and [`Stopped`](Self::Stopped). The other
/// variants identify incompatible operands or invalid variables and levels.
/// Even an engine with no limits installed can report `OverBudget` when a
/// buffer reservation fails.
///
/// Recovery depends on how the operation takes its diagram. A consuming
/// operation such as [`Engine::and`](crate::Engine::and) drops its operands on
/// error; keep copies before the call if a retry needs them. A borrowed query
/// leaves its input unchanged. An in-place operation such as
/// [`try_minimize`](crate::reduce::try_minimize) or
/// [`Engine::rotation_search`](crate::Engine::rotation_search) documents which
/// completed edits remain after a refusal.
///
/// The enum is exhaustive and callers may construct its variants; adding a
/// variant requires a breaking release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationError {
    /// The allocator refused, or the installed byte budget would be exceeded
    /// by a growth this operation needs, or a level outgrew an internal 32-bit
    /// index. A single product-grid resize can ask for many GiB, so this is the
    /// variant a caller that wants to survive a too-large conjunction — by
    /// splitting it, or by choosing another vtree — must handle. The infallible
    /// wrappers panic on it. [`OperationMetrics::refused_reserve_bytes`](crate::limits::OperationMetrics::refused_reserve_bytes)
    /// records allocator refusals since the last meter reset.
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
    /// not returned on error. Display uses the 1-based DIMACS variable number.
    VariableNotInVtree(crate::vtree::VarId),
    /// An input cube or substitution map names the same source variable more than once.
    DuplicateVariable(crate::vtree::VarId),
}

impl std::fmt::Display for OperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationError::VtreeMismatch => f.write_str("operands must share the same vtree allocation"),
            OperationError::RootMismatch => f.write_str("conjunction operands must have the same output vtree node"),
            OperationError::IncompatibleWeights => f.write_str("operands require compatible literal weights and arithmetic; stored integer counts cannot be reweighted"),
            OperationError::LevelNotInVtree(level) => write!(f, "level {} is outside the vtree", level.idx()),
            OperationError::MarginalLevel(level) => write!(f, "operation requires structural data at marginal level {}", level.idx()),
            OperationError::OverBudget => f.write_str("memory budget exceeded"),
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
