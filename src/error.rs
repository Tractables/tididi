//! The one error every fallible operation in this crate returns.

/// Why a fallible operation stopped before producing a diagram.
///
/// Every variant means the same thing to the operand diagrams: they are spent,
/// and the partial output is discarded. A caller that installed no limits
/// ([`LimitSet`](crate::engine::LimitSet)) can still see `OverBudget`, because the OS allocator can
/// refuse a product grid on its own.
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
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            ApplyError::OverBudget => "memory budget exceeded",
            ApplyError::Deadline => "deadline reached",
            ApplyError::OutputCap => "output node cap exceeded",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ApplyError {}
