//! The home a projection's moves imply.
//!
//! Rotations and swaps on a projection keep its node indices, so its link
//! still names, for each own node, the home node it stood for. Between an own
//! node and its own parent the home has a chain of nodes, each with own
//! leaves under one child only; the chain moves with the own node. The lifted
//! home gives each own internal node's home node, as children, the tops of
//! the chains above its own children now; every other home node keeps its
//! children. The projection of the lifted home on the own leaves is then the
//! moved tree, whatever moves led to it.

use std::sync::Arc;

use super::{Home, Vtree, VtreeIdx, VtreeNode};

impl Vtree {
    /// The home of which this tree, a projection moved by rotations and
    /// swaps since its link was made, is the projection on its leaves;
    /// `None` on a root vtree. Linear in the home's size.
    pub(crate) fn lifted_home(&self) -> Option<Home> {
        let link = self.home.as_deref()?;
        let old = link.home.vtree();
        let internals = self.internal_bottomup_slice();
        let mut stands = vec![false; old.num_nodes()];
        for &t in internals {
            stands[self.home_node(t).idx()] = true;
        }
        let top = |c: VtreeIdx| {
            let mut h = self.home_node(c);
            while let Some(p) = old.node(h).parent().filter(|p| !stands[p.idx()]) {
                h = p;
            }
            h
        };
        let mut nodes = old.nodes.clone();
        for &t in internals {
            let (left, right) = self.children(t);
            nodes[self.home_node(t).idx()] = VtreeNode::Internal { left: top(left), right: top(right), parent: None };
        }
        let (mut lifted, _) =
            Self::from_nodes_with_map(nodes, old.root(), old.num_vars()).expect("chains moved whole form one tree");
        lifted.context = Arc::clone(&old.context);
        Some(Home::new(lifted))
    }
}
