use super::*;

#[test]
fn test_single_variable_vtree() {
    let vtree = Vtree::balanced(1);
    assert_eq!(vtree.num_nodes(), 1);
    assert!(vtree.node(vtree.root).is_leaf());
    assert_eq!(vtree.leaf_var(vtree.root), VarId(0));
    assert_eq!(vtree.bottomup().count(), 1);
}

#[test]
fn test_two_variable_vtree() {
    let vtree = Vtree::balanced(2);
    assert_eq!(vtree.num_nodes(), 3); // root + 2 leaves
    assert!(!vtree.node(vtree.root).is_leaf());
    let (l, r) = vtree.children(vtree.root);
    assert_eq!(vtree.leaf_var(l), VarId(0));
    assert_eq!(vtree.leaf_var(r), VarId(1));
    // Level-order bottom-up: leaves first, then root
    let bo: Vec<VtreeIdx> = vtree.bottomup().collect();
    assert_eq!(bo, vec![l, r, vtree.root]);
}

#[test]
fn test_four_variable_vtree() {
    let vtree = Vtree::balanced(4);
    // 4 leaves + 3 internal = 7 nodes
    assert_eq!(vtree.num_nodes(), 7);
    assert_eq!(vtree.bottomup().count(), 7);

    // Root should be last (highest index)
    assert_eq!(vtree.root, VtreeIdx(6));

    // All leaves should be indices 0..4 (first layer)
    for var in 0..4u32 {
        let leaf_idx = vtree.var_to_leaf[var as usize];
        assert!(vtree.node(leaf_idx).is_leaf());
        assert_eq!(vtree.leaf_var(leaf_idx), VarId(var));
        assert!(leaf_idx.0 < 4, "leaves should come first");
    }
}

#[test]
fn test_bottomup_is_sequential() {
    let vtree = Vtree::balanced(5);
    let bo: Vec<VtreeIdx> = vtree.bottomup().collect();
    let expected: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx).collect();
    assert_eq!(bo, expected);
}

#[test]
fn test_children_before_parents() {
    // In bottom-up level order, children always have lower indices than parents
    for n in 2..=8u32 {
        let vtree = Vtree::balanced(n);
        for (i, node) in vtree.nodes.iter().enumerate() {
            if let VtreeNode::Internal { left, right, .. } = node {
                assert!(left.0 < i as u32, "left child {} >= parent {}", left.0, i);
                assert!(right.0 < i as u32, "right child {} >= parent {}", right.0, i);
            }
        }
    }
}

#[test]
fn test_level_order_balanced() {
    // For balanced vtrees, all leaves at the same depth come first
    for n in [2, 4, 8u32] {
        let vtree = Vtree::balanced(n);
        let num_leaves = n as usize;
        for (i, node) in vtree.nodes.iter().enumerate() {
            if node.is_leaf() {
                assert!(i < num_leaves, "leaf at index {} but {} leaves total", i, num_leaves);
            } else {
                assert!(i >= num_leaves, "internal at index {} but {} leaves total", i, num_leaves);
            }
        }
    }
}

#[test]
fn test_var_to_leaf_mapping() {
    let vtree = Vtree::balanced(5);
    for var in 0..5u32 {
        let leaf = vtree.var_to_leaf[var as usize];
        assert_eq!(vtree.leaf_var(leaf), VarId(var));
    }
}

#[test]
fn test_sibling() {
    let vtree = Vtree::balanced(2);
    let (l, r) = vtree.children(vtree.root);
    assert_eq!(vtree.sibling(l), r);
    assert_eq!(vtree.sibling(r), l);
}

#[test]
fn test_linear_structure() {
    let vtree = Vtree::linear(4);
    // 4 leaves + 3 internal = 7 nodes
    assert_eq!(vtree.num_nodes(), 7);

    // Root's left child should be a leaf (x3, highest-indexed var with reversed order)
    let (l, r) = vtree.children(vtree.root);
    assert!(vtree.node(l).is_leaf());
    assert_eq!(vtree.leaf_var(l), VarId(3));
    assert!(!vtree.node(r).is_leaf());

    // All vars mapped correctly
    for var in 0..4u32 {
        let leaf = vtree.var_to_leaf[var as usize];
        assert_eq!(vtree.leaf_var(leaf), VarId(var));
    }
}

#[test]
fn test_random_structure() {
    let vtree = Vtree::random(5, 42);
    // 5 leaves + 4 internal = 9 nodes
    assert_eq!(vtree.num_nodes(), 9);
    assert_eq!(vtree.bottomup().count(), 9);

    // Root should be last index
    assert_eq!(vtree.root, VtreeIdx(8));

    // All vars mapped correctly
    for var in 0..5u32 {
        let leaf = vtree.var_to_leaf[var as usize];
        assert_eq!(vtree.leaf_var(leaf), VarId(var));
    }
}

#[test]
fn test_random_deterministic_with_seed() {
    let v1 = Vtree::random(6, 123);
    let v2 = Vtree::random(6, 123);
    // Same seed → same structure
    assert_eq!(v1.nodes.len(), v2.nodes.len());
    assert_eq!(v1.root, v2.root);
}

#[test]
fn test_single_var_all_shapes() {
    let b = Vtree::balanced(1);
    let l = Vtree::linear(1);
    let r = Vtree::random(1, 0);
    assert_eq!(b.num_nodes(), 1);
    assert_eq!(l.num_nodes(), 1);
    assert_eq!(r.num_nodes(), 1);
}

// --- to_text / from_text tests ---

/// Parse the vtree format string into (node_count_from_header, Vec<line_tokens>).
fn parse_vtree_format(s: &str) -> (usize, Vec<Vec<String>>) {
    let mut lines = s.lines();
    let header = lines.next().unwrap();
    let parts: Vec<&str> = header.split_whitespace().collect();
    assert_eq!(parts[0], "vtree");
    let n: usize = parts[1].parse().unwrap();
    let node_lines: Vec<Vec<String>> = lines
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect();
    (n, node_lines)
}

#[test]
fn test_sdd_format_header_node_count() {
    // Header says "vtree N" where N = 2*num_vars - 1.
    for num_vars in [1u32, 2, 3, 4, 5, 8, 16] {
        let vtree = Vtree::balanced(num_vars);
        let fmt = vtree.to_text();
        let (n, node_lines) = parse_vtree_format(&fmt);
        let expected = 2 * num_vars as usize - 1;
        assert_eq!(n, expected, "num_vars={}", num_vars);
        assert_eq!(node_lines.len(), n, "num_vars={}", num_vars);
    }
}

#[test]
fn test_sdd_format_vars_one_indexed() {
    // Leaf variable IDs in the SDD format are 1-indexed (internal 0-indexed VarId + 1).
    let num_vars = 5u32;
    let vtree = Vtree::balanced(num_vars);
    let fmt = vtree.to_text();
    let (_, node_lines) = parse_vtree_format(&fmt);
    let mut var_ids: Vec<u32> = node_lines
        .iter()
        .filter(|toks| toks[0] == "L")
        .map(|toks| toks[2].parse().unwrap())
        .collect();
    var_ids.sort();
    let expected: Vec<u32> = (1..=num_vars).collect();
    assert_eq!(var_ids, expected);
}

#[test]
fn test_sdd_format_children_before_parents() {
    // Internal node IDs must be greater than both their children's IDs.
    for vtree in [Vtree::balanced(6), Vtree::linear(6), Vtree::random(6, 42)] {
        let fmt = vtree.to_text();
        let (_, node_lines) = parse_vtree_format(&fmt);
        for toks in &node_lines {
            if toks[0] == "I" {
                let id: usize = toks[1].parse().unwrap();
                let left: usize = toks[2].parse().unwrap();
                let right: usize = toks[3].parse().unwrap();
                assert!(left < id, "left {} >= parent {}", left, id);
                assert!(right < id, "right {} >= parent {}", right, id);
            }
        }
    }
}

#[test]
fn test_sdd_format_node_types() {
    // Every line is either "L id var" (3 tokens) or "I id left right" (4 tokens).
    let vtree = Vtree::balanced(4);
    let fmt = vtree.to_text();
    let (n, node_lines) = parse_vtree_format(&fmt);
    assert_eq!(node_lines.len(), n);
    let mut leaf_count = 0usize;
    let mut internal_count = 0usize;
    for toks in &node_lines {
        match toks[0].as_str() {
            "L" => {
                assert_eq!(toks.len(), 3, "leaf line should have 3 tokens");
                leaf_count += 1;
            }
            "I" => {
                assert_eq!(toks.len(), 4, "internal line should have 4 tokens");
                internal_count += 1;
            }
            other => panic!("unexpected node type: {}", other),
        }
    }
    assert_eq!(leaf_count, 4);       // 4 vars → 4 leaves
    assert_eq!(internal_count, 3);   // 4 vars → 3 internal
}

#[test]
fn test_sdd_format_all_vtree_types() {
    // All three vtree construction methods produce a valid, structurally consistent format.
    let num_vars = 7u32;
    for vtree in [Vtree::balanced(num_vars), Vtree::linear(num_vars), Vtree::random(num_vars, 42)] {
        let fmt = vtree.to_text();
        let (n, node_lines) = parse_vtree_format(&fmt);
        assert_eq!(n, 2 * num_vars as usize - 1);
        assert_eq!(node_lines.len(), n);

        // All node IDs in the lines are sequential 0..n.
        let ids: Vec<usize> = node_lines.iter()
            .map(|toks| toks[1].parse::<usize>().unwrap())
            .collect();
        assert_eq!(ids, (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn test_sdd_format_single_var() {
    // Single-variable vtree: one leaf node, no internals.
    let vtree = Vtree::balanced(1);
    let fmt = vtree.to_text();
    let (n, node_lines) = parse_vtree_format(&fmt);
    assert_eq!(n, 1);
    assert_eq!(node_lines.len(), 1);
    assert_eq!(node_lines[0][0], "L");
    assert_eq!(node_lines[0][2], "1"); // 1-indexed var
}

#[test]
fn test_vtree_format_deterministic() {
    // Same vtree type + same seed → identical format output.
    let fmt1 = Vtree::random(8, 17).to_text();
    let fmt2 = Vtree::random(8, 17).to_text();
    assert_eq!(fmt1, fmt2);
}

#[test]
fn test_vtree_format_roundtrip_balanced() {
    for n in 1..=10 {
        let vtree = Vtree::balanced(n);
        let fmt = vtree.to_text();
        let loaded = Vtree::from_text(&fmt).expect("roundtrip parse failed");
        let fmt2 = loaded.to_text();
        assert_eq!(fmt, fmt2, "roundtrip failed for balanced vtree with {} vars", n);
    }
}

#[test]
fn test_vtree_format_roundtrip_linear() {
    for n in 1..=10 {
        let vtree = Vtree::linear(n);
        let fmt = vtree.to_text();
        let loaded = Vtree::from_text(&fmt).expect("roundtrip parse failed");
        let fmt2 = loaded.to_text();
        assert_eq!(fmt, fmt2, "roundtrip failed for linear vtree with {} vars", n);
    }
}

#[test]
fn test_vtree_format_roundtrip_random() {
    for seed in 0..5 {
        let vtree = Vtree::random(8, seed);
        let fmt = vtree.to_text();
        let loaded = Vtree::from_text(&fmt).expect("roundtrip parse failed");
        let fmt2 = loaded.to_text();
        assert_eq!(fmt, fmt2, "roundtrip failed for random vtree seed {}", seed);
    }
}


// ── project_to_vars ───────────────────────────────────────────────────────────

/// Structural invariants of a projected vtree: exactly the kept variables
/// appear as leaves (under their local ids), `num_leaves()` matches the local
/// var count, every internal node has two distinct children, and the
/// leaves-first / children-before-parent layout is intact.
fn assert_projection_wellformed(proj: &Vtree, num_local: u32) {
    assert_eq!(proj.num_leaves(), num_local);
    let leaves: Vec<VarId> = {
        let mut v: Vec<VarId> = proj.leaf_bottomup().map(|(_, var)| var).collect();
        v.sort();
        v
    };
    assert_eq!(leaves, (0..num_local).map(VarId).collect::<Vec<_>>());

    // A binary tree over K leaves has exactly K-1 internal nodes: proof that
    // no unary (spliced) node survived.
    assert_eq!(proj.num_nodes(), (2 * num_local - 1) as usize);

    let n_leaves = proj.num_leaves() as usize;
    for (i, node) in proj.nodes.iter().enumerate() {
        match node {
            VtreeNode::Leaf { .. } => assert!(i < n_leaves, "leaf at {i}, num_leaves={n_leaves}"),
            VtreeNode::Internal { left, right, .. } => {
                assert!(i >= n_leaves, "internal at {i}, num_leaves={n_leaves}");
                assert_ne!(left, right, "internal node {i} has a duplicated child");
                assert!(left.idx() < i && right.idx() < i, "children not below parent at {i}");
            }
        }
    }
    for v in 0..num_local {
        assert_eq!(proj.leaf_var(proj.var_to_leaf[v as usize]), VarId(v));
    }
}

#[test]
fn test_project_to_vars_balanced_subset() {
    // balanced(8): ((0 1)(2 3)) ((4 5)(6 7)). Keep {1, 2, 4, 7} → local 0..3.
    let vtree = Vtree::balanced(8);
    let keep = [VarId(1), VarId(2), VarId(4), VarId(7)];
    let proj = vtree
        .project_to_vars(
            |v| keep.iter().position(|&k| k == v).map(|i| VarId(i as u32)),
            keep.len() as u32,
        )
        .expect("non-empty projection");
    assert_projection_wellformed(&proj, keep.len() as u32);

    // Grouping is inherited: 1 and 2 stayed under the old left half, 4 and 7
    // under the old right half, so the projected root separates {0,1} from {2,3}.
    let (l, r) = proj.children(proj.root);
    let mut left_vars: Vec<u32> = collect_subtree_vars(&proj, l);
    let mut right_vars: Vec<u32> = collect_subtree_vars(&proj, r);
    left_vars.sort();
    right_vars.sort();
    assert!(
        (left_vars == vec![0, 1] && right_vars == vec![2, 3])
            || (left_vars == vec![2, 3] && right_vars == vec![0, 1]),
        "projected root split {left_vars:?} | {right_vars:?}"
    );
}

fn collect_subtree_vars(vtree: &Vtree, node: VtreeIdx) -> Vec<u32> {
    let mut out = Vec::new();
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match vtree.node(n) {
            VtreeNode::Leaf { var, .. } => out.push(var.0),
            VtreeNode::Internal { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    out
}

#[test]
fn test_project_to_vars_edge_cases() {
    let vtree = Vtree::random(9, 13);

    // Keeping everything is the identity on the variable set.
    let all = vtree
        .project_to_vars(Some, 9)
        .expect("full projection");
    assert_projection_wellformed(&all, 9);

    // Keeping a single variable yields a bare leaf (no internal nodes).
    let one = vtree
        .project_to_vars(|v| (v == VarId(4)).then_some(VarId(0)), 1)
        .expect("single-var projection");
    assert_eq!(one.num_leaves(), 1);
    assert_eq!(one.num_nodes(), 1);
    assert!(one.node(one.root).is_leaf());

    // Keeping nothing has no valid answer.
    assert!(vtree.project_to_vars(|_| None, 0).is_none());

    // Every 2-subset projects to a well-formed 3-node vtree, whichever way the
    // splice-outs fall.
    for a in 0..9u32 {
        for b in (a + 1)..9u32 {
            let proj = vtree
                .project_to_vars(
                    |v| match v.0 {
                        x if x == a => Some(VarId(0)),
                        x if x == b => Some(VarId(1)),
                        _ => None,
                    },
                    2,
                )
                .expect("2-var projection");
            assert_projection_wellformed(&proj, 2);
        }
    }
}

// ── leaf / join / balanced_over / graft / validate ───────────────────────────

fn sorted_vars(v: &Vtree) -> Vec<u32> {
    let mut vars: Vec<u32> = v.leaf_bottomup().map(|(_, var)| var.0).collect();
    vars.sort();
    vars
}

#[test]
fn leaf_is_a_one_node_tree_over_a_sparse_id_space() {
    let v = Vtree::leaf(VarId(4));
    assert_eq!(v.num_nodes(), 1);
    assert_eq!(v.num_leaves(), 1);
    assert_eq!(v.num_vars(), 5);
    assert_eq!(v.leaf_var(v.root()), VarId(4));
    assert_eq!(v.leaf_of(VarId(4)).expect("the vtree carries this variable"), v.root());
    assert_eq!(v.validate(), Ok(()));
}

#[test]
fn join_composes_and_takes_the_wider_id_space() {
    let left = Vtree::leaf(VarId(0));
    let right = Vtree::balanced_over(&[VarId(5), VarId(2)]);
    let v = Vtree::join(&left, &right).unwrap();
    assert_eq!(v.validate(), Ok(()));
    assert_eq!(v.num_vars(), 6);
    assert_eq!(v.num_leaves(), 3);
    assert_eq!(sorted_vars(&v), vec![0, 2, 5]);
    let (l, r) = v.children(v.root());
    assert_eq!(v.leaf_var(l), VarId(0));
    let (rl, rr) = v.children(r);
    assert_eq!((v.leaf_var(rl), v.leaf_var(rr)), (VarId(5), VarId(2)));
}

#[test]
fn join_rejects_a_shared_variable() {
    let a = Vtree::balanced_over(&[VarId(0), VarId(1)]);
    let b = Vtree::leaf(VarId(1));
    assert_eq!(Vtree::join(&a, &b).err(), Some(VtreeError::OverlappingVariable(VarId(1))));
}

#[test]
fn join_of_leaves_is_linear_from_order() {
    let joined = Vtree::join(
        &Vtree::leaf(VarId(2)),
        &Vtree::join(&Vtree::leaf(VarId(0)), &Vtree::leaf(VarId(1))).unwrap(),
    )
    .unwrap();
    assert!(joined.same_tree(&Vtree::linear_over(&[VarId(2), VarId(0), VarId(1)])));
}

#[test]
fn balanced_over_natural_order_is_balanced() {
    for n in 1..9 {
        let order: Vec<VarId> = (0..n).map(VarId).collect();
        assert!(Vtree::balanced_over(&order).same_tree(&Vtree::balanced(n)));
    }
}

#[test]
fn balanced_over_follows_the_order_and_allows_gaps() {
    let v = Vtree::balanced_over(&[VarId(7), VarId(1), VarId(3)]);
    assert_eq!(v.validate(), Ok(()));
    assert_eq!(v.num_vars(), 8);
    assert_eq!(v.num_leaves(), 3);
    let (l, r) = v.children(v.root());
    assert_eq!(v.leaf_var(l), VarId(7));
    let (rl, rr) = v.children(r);
    assert_eq!((v.leaf_var(rl), v.leaf_var(rr)), (VarId(1), VarId(3)));
}

#[test]
fn graft_hangs_pieces_down_a_right_spine() {
    let parts = [
        Vtree::balanced_over(&[VarId(0), VarId(1)]),
        Vtree::balanced_over(&[VarId(4), VarId(5)]),
    ];
    let v = Vtree::graft(&parts, &[VarId(2), VarId(3)]).unwrap();
    assert_eq!(v.validate(), Ok(()));
    assert_eq!(sorted_vars(&v), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(v.num_vars(), 6);
    // root = (((S0, S1), x2), x3)
    let (t, x3) = v.children(v.root());
    assert_eq!(v.leaf_var(x3), VarId(3));
    let (t, x2) = v.children(t);
    assert_eq!(v.leaf_var(x2), VarId(2));
    let (s0, s1) = v.children(t);
    assert_eq!((v.leaf_var(v.children(s0).0), v.leaf_var(v.children(s1).1)), (VarId(0), VarId(5)));
}

#[test]
fn graft_rejects_overlap_and_emptiness() {
    let a = Vtree::balanced_over(&[VarId(0), VarId(1)]);
    assert_eq!(
        Vtree::graft(&[a.clone(), Vtree::leaf(VarId(1))], &[]).err(),
        Some(VtreeError::OverlappingVariable(VarId(1)))
    );
    assert_eq!(
        Vtree::graft(std::slice::from_ref(&a), &[VarId(0)]).err(),
        Some(VtreeError::OverlappingVariable(VarId(0)))
    );
    assert!(matches!(Vtree::graft(&[], &[]), Err(VtreeError::Invalid(_))));
    assert!(Vtree::graft(&[], &[VarId(3)]).unwrap().same_tree(&Vtree::leaf(VarId(3))));
}

#[test]
fn constructions_round_trip_through_vtree_text() {
    let trees = [
        Vtree::leaf(VarId(3)),
        Vtree::balanced_over(&[VarId(6), VarId(0), VarId(2)]),
        Vtree::join(&Vtree::leaf(VarId(9)), &Vtree::linear_over(&[VarId(1), VarId(4)])).unwrap(),
        Vtree::graft(&[Vtree::balanced(3), Vtree::leaf(VarId(7))], &[VarId(5)]).unwrap(),
    ];
    for v in &trees {
        let back = Vtree::from_text(&v.to_text()).unwrap();
        assert!(back.same_tree(v));
        assert_eq!(back.num_vars(), v.num_vars());
        assert_eq!(back.num_leaves(), v.num_leaves());
        assert_eq!(back.validate(), Ok(()));
    }
}

#[test]
fn validate_reports_a_bad_text_tree_before_it_is_built() {
    // Two leaves naming one variable, and a child reachable twice.
    assert!(matches!(
        Vtree::from_text("vtree 3\nL 0 1\nL 1 1\nI 2 0 1\n"),
        Err(VtreeError::Text(_))
    ));
    assert!(matches!(
        Vtree::from_text("vtree 2\nL 0 1\nI 1 0 0\n"),
        Err(VtreeError::Text(_))
    ));
}

#[test]
fn validate_passes_every_builder_and_survives_rotation() {
    let mut trees = vec![Vtree::balanced(7), Vtree::linear(5), Vtree::random(9, 3)];
    let proj = trees[0]
        .project_to_vars(|v| (v.0 % 2 == 0).then_some(VarId(v.0 / 2)), 4)
        .unwrap();
    trees.push(proj);
    for v in &mut trees {
        assert_eq!(v.validate(), Ok(()));
        let internal: Vec<VtreeIdx> = v.internal_bottomup().map(|(t, _, _)| t).collect();
        for t in internal {
            if rotate::rotate_left(v, t).is_some() {
                assert_eq!(v.validate(), Ok(()));
            }
        }
    }
}

#[test]
fn same_tree_ignores_numbering() {
    let a = Vtree::linear_over(&[VarId(0), VarId(1), VarId(2)]);
    let b = Vtree::from_text("vtree 5\nL 0 3\nL 1 2\nI 2 1 0\nL 3 1\nI 4 3 2\n").unwrap();
    assert!(a.same_tree(&b));
    assert!(!a.same_tree(&Vtree::linear_over(&[VarId(1), VarId(0), VarId(2)])));
    let left_deep = Vtree::join(
        &Vtree::join(&Vtree::leaf(VarId(0)), &Vtree::leaf(VarId(1))).unwrap(),
        &Vtree::leaf(VarId(2)),
    )
    .unwrap();
    assert!(!a.same_tree(&left_deep));
}
