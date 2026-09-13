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

/// Why a fallible operation stopped before returning a result.
///
/// What an error leaves behind depends on how the operation took its diagram,
/// not on the variant. An operand taken by value ([`Engine::and`](crate::Engine::and)
/// and the other `Engine` methods) is consumed on `Err` as on `Ok`, and the
/// partial output is discarded. A diagram taken by `&mut`
/// ([`marginalize_levels`](crate::marginal::marginalize_levels), [`try_reduce`](crate::reduce::try_reduce),
/// [`Engine::rotation_search`](crate::Engine::rotation_search)) is left
/// well-formed and count-correct at the point each documents. A borrowed one
/// ([`Engine::model_count`](crate::Engine::model_count)) is untouched. A caller
/// that installed no limits ([`LimitConfig`](crate::limits::LimitConfig)) can still
/// see `OverBudget`, because the allocator can refuse a reservation on its
/// own.
///
/// A caller may also mint one for its own resource failure: the enum is a flat
/// `Copy` type whose payloads are the caller's own input, so a caller that
/// refuses a reservation of its own before calling in returns `OverBudget`
/// rather than growing a parallel error of the same shape.
///
/// Because callers mint it, the enum is and stays exhaustive: it carries no
/// `#[non_exhaustive]`, a `match` over its variants needs no wildcard arm, and
/// a further variant would be a breaking change rather than an additive one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationError {
    /// The allocator refused, or the installed byte budget would be exceeded
    /// by a growth this operation needs, or a level outgrew an internal 32-bit
    /// index. A single product-grid resize can ask for many GiB, so this is the
    /// variant a caller that wants to survive a too-large conjunction — by
    /// splitting it, or by choosing another vtree — must handle. The infallible
    /// wrappers panic on it. [`OperationMetrics::refused_reserve_bytes`](crate::limits::OperationMetrics::refused_reserve_bytes)
    /// tells an allocator refusal from a budget one.
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
    /// The operation names a variable the operand's vtree does not carry. This
    /// is validated before changing the operation's state; an operand taken by
    /// value is consumed on error.
    /// Displays the variable as its 1-based DIMACS number.
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
