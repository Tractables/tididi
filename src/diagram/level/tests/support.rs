//! Setters that stage a level in a shape no operation produces.

use super::*;

impl TddLevel {
    /// Give the node at `at` a new pair list, keeping its index.
    ///
    /// Every other node keeps its index too, so no parent reference has to be
    /// rewritten. The node's old pair range is abandoned in the arena and
    /// reclaimed by the next compaction.
    ///
    /// No operation edits a node's pairs in place — the reduction passes rebuild
    /// a level instead — so this exists for the tests that stage an arena with an
    /// abandoned pair range.
    pub(crate) fn replace_node_pairs(&mut self, at: NodeIdx, input_pairs: &[ChildPair]) {
        let fresh = self.push_internal_node(input_pairs);
        self.nodes[at.idx()] = self.nodes[fresh.idx()];
        self.nodes.pop();
    }

    /// Put this level into its counts state without touching the arenas, so a
    /// test can build a level whose shape `become_marginal` would have thrown
    /// away — including one an invariant check is supposed to reject.
    pub(crate) fn set_counts_state(&mut self, counts: Vec<u128>, big: Option<CountOverflow>) {
        self.state = LevelState::Counts { counts, big, retired: 0 };
    }
}
