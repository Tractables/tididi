//! Which levels this apply sums out.

/// The vtree nodes whose levels the bottom-up sweep marginalizes.
///
/// Most applies marginalize nothing, and every consumer used to ask that
/// question by unwrapping an `Option<&[bool]>` — so the two states were spelled
/// out at each of them. Here the absence is a variant, and the three questions
/// actually asked of the set are its methods.
#[derive(Clone, Copy)]
pub(crate) enum MargTargets<'a> {
    /// This apply marginalizes nothing.
    None,
    /// `true` at every vtree node whose level is summed out.
    At(&'a [bool]),
}

impl<'a> MargTargets<'a> {
    /// The set given, or `None` when none was.
    pub(crate) fn new(targets: Option<&'a [bool]>) -> Self {
        match targets {
            Some(t) => MargTargets::At(t),
            None => MargTargets::None,
        }
    }

    /// Whether anything is summed out at all — what the streaming scratch and
    /// its pooling are conditioned on.
    #[inline]
    pub(crate) fn any(self) -> bool {
        matches!(self, MargTargets::At(_))
    }

    /// Whether level `t_idx` is summed out.
    #[inline]
    pub(crate) fn is_target(self, t_idx: usize) -> bool {
        match self {
            MargTargets::None => false,
            MargTargets::At(t) => t[t_idx],
        }
    }

    /// Single source of truth for the streaming-eligibility gate: a level
    /// streams its marginal iff it is a target AND the streaming gate is on
    /// (off ⇒ don't stream, materialize + post-apply `marginalize_batch`).
    /// Consulted per level by the emit-growth mode decision and by
    /// `build_stream_state`'s setup; the commit then keys off `stream_state`
    /// being `Some` rather than re-reading the predicate. Do NOT re-inline it
    /// at a call site — it is cheap, and the cold per-level path can afford the
    /// call.
    #[inline]
    pub(crate) fn stream_eligible(self, t_idx: usize) -> bool {
        self.is_target(t_idx) && super::cell::both_marginal_collapse_enabled()
    }
}
