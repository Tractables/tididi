//! The error types.
//!
//! One flat enum, [`ApplyError`], for every operation that runs under a limit.
//! The domain errors belong to the modules that raise them:
//! [`VtreeError`](crate::vtree::VtreeError) to [`crate::vtree`],
//! [`IoError`](crate::io::IoError) to [`crate::io`], and
//! [`TddBuildError`](crate::diagram::TddBuildError) to [`crate::diagram`].
//!
//! Entry point: [`ApplyError`], returned by every fallible
//! [`Engine`](crate::Engine) method and by [`crate::marginal::marginalize`] and
//! [`crate::reduce::try_minimize`].

/// Why a fallible operation stopped before producing a diagram.
///
/// Every variant means the same thing to the operand diagrams: they are spent,
/// and the partial output is discarded. A caller that installed no limits
/// ([`LimitSet`](crate::engine::LimitSet)) can still see `OverBudget`, because the OS allocator can
/// refuse a product grid on its own.
///
/// A caller may also mint one for its own resource failure: the enum is a flat
/// `Copy` type whose payloads are the caller's own input, so a driver that
/// refuses a reservation of its own before calling in returns `OverBudget`
/// rather than growing a parallel error of the same shape.
///
/// Because callers mint it, the enum is and stays exhaustive: it carries no
/// `#[non_exhaustive]`, a `match` over its variants needs no wildcard arm, and
/// a further variant would be a breaking change rather than an additive one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyError {
    /// The OS allocator refused, or the installed byte budget would be exceeded
    /// by a growth this operation needs. A single product-grid resize can ask
    /// for many GiB, so this is the variant a caller that wants to survive a
    /// too-large conjunction — by splitting it, or by choosing another vtree —
    /// must handle. The infallible wrappers panic on it.
    OverBudget,
    /// The installed deadline passed, or an installed schedule concluded that
    /// the operation should stop, at one of the operation's poll points.
    Deadline,
    /// The installed cap on produced output nodes tripped: a deliberate size
    /// cut, not an allocation failure.
    OutputCap,
    /// The operation names a variable the operand's vtree does not carry. This
    /// is the caller's input rather than a resource failure, and the operation
    /// does no work before reporting it; the operand is spent all the same.
    VariableNotInVtree(crate::vtree::VarId),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplyError::OverBudget => f.write_str("memory budget exceeded"),
            ApplyError::Deadline => f.write_str("deadline reached"),
            ApplyError::OutputCap => f.write_str("output node cap exceeded"),
            ApplyError::VariableNotInVtree(var) => {
                write!(f, "variable x{} is not in the vtree", var.0 + 1)
            }
        }
    }
}

impl std::error::Error for ApplyError {}
