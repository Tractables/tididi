//! Pruning and tombstones.
//!
//! Sibling of `tests.rs`, which holds the fixtures these read.

use std::sync::Arc;
use crate::reduce::minimize;
use crate::reduce::prune::prune_unreachable;
use crate::query::model_count;
use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::diagram::TddNodeData;
use crate::vtree::Vtree;


    /// An unreferenced tombstone changes no reported metric, and prune drops
    /// it.
    #[test]
    fn readers_skip_tombstones_and_prune_reclaims() {
        let eng = &crate::engine::Engine::new();
        let vtree = Arc::new(Vtree::balanced(4));
        let mut tdd = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![3, -4]]);
        minimize(&mut tdd);
        assert_canonical(&tdd);

        let size0 = tdd.size();
        let total0 = tdd.node_count();
        let maxw0 = tdd.max_width();
        let mc0 = model_count(&tdd);
        assert!(total0 > 0);

        let target = tdd
            .levels
            .iter()
            .position(|l| !l.is_marginal() && l.nodes.iter().any(|n| n.is_internal()))
            .expect("a level with an internal node");
        let len_before = tdd.levels[target].nodes.len();
        tdd.levels[target].nodes.push(TddNodeData::tombstone());
        tdd.levels[target].n_tombstones += 1;

        assert_eq!(tdd.size(), size0);
        assert_eq!(tdd.node_count(), total0);
        assert_eq!(tdd.max_width(), maxw0);
        assert_eq!(model_count(&tdd), mc0);
        assert_eq!(tdd.levels[target].width(), len_before + 1);
        assert_eq!(tdd.levels[target].live_width(), len_before);
        assert!(tdd.levels[target].nodes.last().unwrap().is_tombstone());

        prune_unreachable(eng, &mut tdd).expect("tiny scratch reservation cannot fail");
        assert_canonical(&tdd);
        assert_eq!(tdd.levels[target].n_tombstones, 0);
        assert!(!tdd.levels.iter().any(|l| l.nodes.iter().any(|n| n.is_tombstone())));
        assert_eq!(tdd.size(), size0);
        assert_eq!(tdd.node_count(), total0);
        assert_eq!(tdd.max_width(), maxw0);
        assert_eq!(model_count(&tdd), mc0);
    }
