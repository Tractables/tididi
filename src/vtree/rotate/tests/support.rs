//! Releasing a rotation's order obligation by hand.

use super::*;

impl PendingTopo {
    /// Discard the obligation because the caller is about to overwrite the
    /// order by other means. Only the rebuild-equivalence test needs this: it
    /// rotates and then recomputes the whole order from scratch.
    pub(crate) fn abandon(mut self) -> RotationInfo {
        self.settled = true;
        self.info
    }
}
