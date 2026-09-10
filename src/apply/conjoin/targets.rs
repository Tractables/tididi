//! Which levels this apply sums out.

/// The vtree nodes whose levels the bottom-up sweep marginalizes.
///
/// Most applies marginalize nothing, so the absence is a variant rather than
/// an `Option` each consumer unwraps; the three questions asked of the set are
/// its methods.
#[derive(Clone, Copy)]
pub(crate) enum MarginalTargets<'a> {
    /// This apply marginalizes nothing.
    None,
    /// `true` at every vtree node whose level is summed out.
    At(&'a [bool]),
}

impl<'a> MarginalTargets<'a> {
    /// The set given, or `None` when none was.
    pub(crate) fn new(targets: Option<&'a [bool]>) -> Self {
        match targets {
            Some(t) => MarginalTargets::At(t),
            None => MarginalTargets::None,
        }
    }

    /// Whether anything is summed out at all — what the streaming scratch and
    /// its pooling are conditioned on.
    #[inline]
    pub(crate) fn any(self) -> bool {
        matches!(self, MarginalTargets::At(_))
    }

    /// Whether level `t_idx` is summed out.
    #[inline]
    pub(crate) fn is_target(self, t_idx: usize) -> bool {
        match self {
            MarginalTargets::None => false,
            MarginalTargets::At(t) => t[t_idx],
        }
    }

    /// Single source of truth for the streaming-eligibility gate: a level
    /// streams its marginal iff it is a target and
    /// `cell::BOTH_MARGINAL_COLLAPSE_ENABLED` holds. Consulted per level by the
    /// emit-growth mode decision and by `build_stream_state`'s setup; the
    /// commit then keys off `stream_state` being `Some` rather than re-reading
    /// the predicate.
    #[inline]
    pub(crate) fn stream_eligible(self, t_idx: usize) -> bool {
        self.is_target(t_idx) && super::cell::BOTH_MARGINAL_COLLAPSE_ENABLED
    }
}
