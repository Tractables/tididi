//! Setters that stage a level in a shape no operation produces.

use super::*;

impl TddLevel {
    /// Put this level into its counts state without touching the arenas, so a
    /// test can build a level whose shape `become_marginal` would have thrown
    /// away — including one an invariant check is supposed to reject.
    pub(crate) fn set_counts_state(&mut self, counts: Vec<u128>, big: Option<CountOverflow>) {
        self.state = LevelState::Counts(CountBox::new(CountState { counts, big, retired: 0 }));
    }
}
