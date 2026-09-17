use super::*;

/// The text reader reports malformed input rather than panicking.
fn refused(text: &str) {
    assert!(matches!(Vtree::from_text(text), Err(VtreeError::Text(message)) if !message.is_empty()), "accepted {text:?}");
}

#[test]
fn repeated_node_ids_and_inconsistent_record_counts_are_refused() {
    for text in [
        "vtree 3\nL 0 1\nL 1 2\nI 2 0 1\nI 2 0 1\n",
        "vtree 3\nL 0 1\nL 0 2\nI 2 0 1\n",
        "vtree 3\nL 0 1\nL 1 2\nvtree 3\nI 2 0 1\n",
        "vtree 1\nL 0 1\nL 0 2\n",
    ] { refused(text); }
    refused(&format!("vtree {}\n", usize::MAX));
}

#[test]
fn malformed_tree_graphs_and_records_are_refused() {
    for text in [
        "", "\n c comments only\n", "comment\nvtree 1\nL 0 1\n", "vtree 0\n", "vtree 3\n", "vtree 1 extra\nL 0 1\n",
        "vtree 1\nL 0 0\n", "vtree 1\nL 0 1 extra\n",
        "vtree 3\nL 0 1\nL 1 1\nI 2 0 1\n",
        "vtree 3\nL 0 1\nL 1 2\nI 2 0 0\n",
        "vtree 3\nL 0 1\nL 1 2\nI 2 2 1\n",
        "vtree 3\nL 0 1\nI 1 2 0\nI 2 1 0\n",
        "vtree 3\nL 0 1\nL 1 2\nL 2 3\n",
        "vtree 3\nL 0 1\nL 1 2\nI 2 0 3\n",
    ] { refused(text); }
}

#[test]
fn deterministic_token_mutations_are_refused_and_valid_text_round_trips() {
    for tree in [Vtree::leaf(VarId(8)), Vtree::balanced(4), Vtree::linear(4), Vtree::random(5, 97)] {
        let text = tree.to_text();
        let lines: Vec<Vec<&str>> = text.lines().map(|line| line.split_whitespace().collect()).collect();
        for (line_index, tokens) in lines.iter().enumerate() {
            for token_index in 1..tokens.len() {
                for replacement in ["-1", "4294967296", "nope"] {
                    let mut changed = lines.clone();
                    changed[line_index][token_index] = replacement;
                    refused(&changed.iter().map(|fields| format!("{}\n", fields.join(" "))).collect::<String>());
                }
            }
        }
        for end in 0..text.trim_end().len() {
            if let Ok(parsed) = Vtree::from_text(&text[..end]) { assert_eq!(parsed.validate(), Ok(())); }
        }
        for variant in [text.clone(), text.replace('\n', "\r\n"), text.trim_end().to_owned(), text.replace('\n', "\n\n"),
            format!("\n  c vtree with comments\n\n{}c end\n", text.replace('\n', "\n c between records\n"))] {
            let restored = Vtree::from_text(&variant).unwrap();
            assert_eq!(restored.validate(), Ok(()));
            assert!(restored.same_tree(&tree));
        }
    }
}

#[test]
fn a_tiny_file_naming_a_huge_variable_is_refused_rather_than_sized_from_it() {
    // The id-indexed tables are sized by the largest id, not by the number of
    // leaves, so these two lines used to ask for gigabytes.
    refused("vtree 1\nL 0 4000000000\n");
    // Through the node list the refusal keeps its variant; the text reader
    // flattens every build error into `Text`, as its `# Errors` section says.
    let nodes = vec![VtreeNode::Leaf { var: VarId(1), parent: None }];
    assert!(matches!(
        Vtree::from_nodes(nodes, VtreeIdx(0), u32::MAX),
        Err(VtreeError::VariableSpaceTooLarge { .. }),
    ));
}

#[test]
fn a_tree_that_really_carries_the_variables_is_not_refused() {
    // The bound tracks the node list, so the named shapes over a wide id space
    // build: the table they size is proportional to the tree, not to one id.
    let tree = Vtree::balanced(20_000_000);
    assert_eq!(tree.num_vars(), 20_000_000);
}

#[test]
fn a_sparse_id_space_under_the_cap_still_round_trips() {
    let tree = Vtree::from_text("vtree 1\nL 0 1000000\n").expect("sparse ids stay legal");
    assert_eq!(tree.num_vars(), 1_000_000);
    assert_eq!(tree.num_leaves(), 1);
    assert!(Vtree::from_text(&tree.to_text()).unwrap().same_tree(&tree));
}
