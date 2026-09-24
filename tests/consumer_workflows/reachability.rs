use tididi::Engine;
use tididi::vtree::VarId;

use super::support::*;

#[test]
fn every_symbolic_image_matches_explicit_graph_search() {
    let engine = Engine::new();
    // Bit (4 * source + target) denotes an edge. Include empty/dense graphs,
    // a chain, cycles, branches, self-loops and disconnected components.
    let graphs = [
        0u16, 0xffff, 0x0042, 0x0242, 0x8421, 0x1002, 0x8006, 0x2184, 0x5a3c,
    ];
    for (shape, tree) in trees(5).iter().enumerate() {
        for edges in graphs {
            let transition = or_of_cubes(tree, &[VarId(1), VarId(2), VarId(3), VarId(4)], |row| {
                edges & (1 << (4 * (row & 3) + ((row >> 2) & 3))) != 0
            });
            for initial in [0u8, 1, 2, 5, 15] {
                let mut explicit = initial;
                let mut reached = or_of_cubes(tree, &[VarId(1), VarId(2)], |s| {
                    initial & (1 << s) != 0
                });
                let mut retained = Vec::new();
                for step in 0..=4 {
                    let context = format!(
                        "shape {shape}, edges {edges:#06x}, initial {initial:#x}, step {step}"
                    );
                    let expected: Vec<_> = (0..32)
                        .map(|row| explicit & (1 << (row & 3)) != 0)
                        .collect();
                    assert_truth(&engine, &reached, &expected, &context);
                    retained.push((reached.clone(), expected));

                    let mut image = 0u8;
                    for source in 0..4 {
                        if explicit & (1 << source) != 0 {
                            for target in 0..4 {
                                if edges & (1 << (4 * source + target)) != 0 {
                                    image |= 1 << target;
                                }
                            }
                        }
                    }
                    let successors = engine
                        .and_exists(
                            reached.clone(),
                            transition.clone(),
                            &[VarId(1), VarId(2)],
                        )
                        .unwrap();
                    let successors = engine
                        .rename_vars(successors, &[(VarId(3), VarId(1)), (VarId(4), VarId(2))])
                        .unwrap();
                    let image_truth: Vec<_> =
                        (0..32).map(|row| image & (1 << (row & 3)) != 0).collect();
                    assert_truth(
                        &engine,
                        &successors,
                        &image_truth,
                        &format!("{context}, image"),
                    );
                    let next_explicit = explicit | image;
                    let next = engine.or(reached.clone(), successors).unwrap();
                    let stable = engine.equivalent(&reached, &next).unwrap();
                    assert_eq!(stable, explicit == next_explicit, "{context}, convergence");
                    if stable {
                        break;
                    }
                    assert!(step < 4, "{context}, failed to converge");
                    reached = next;
                    explicit = next_explicit;
                }
                // Later steps must leave earlier retained functions unchanged.
                for (step, (diagram, expected)) in retained.iter().enumerate() {
                    assert_truth(
                        &engine,
                        diagram,
                        expected,
                        &format!(
                            "retained shape {shape}, edges {edges:#06x}, initial {initial:#x}, step {step}"
                        ),
                    );
                }
            }
        }
    }
}
