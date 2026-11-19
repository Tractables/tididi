//! Move a diagram onto projections of a balanced home vtree, `reps` times,
//! by `Engine::transport` or by `Engine::embed_moving` followed by
//! `Engine::trim`; `none` builds the diagram only.
//!
//! The home has `h` variables (rounded up to a power of two). The diagram is
//! the conjunction of `blocks` blocks spread evenly over the home, each
//! `x_i <-> y_i` for `i < p`, with the `x` on the left half of an aligned
//! subtree and the `y` on its right half, so each block has about `2^(p+1)`
//! nodes. With `mirror`, the moves alternate between the home and its mirror
//! image, so that every level is mirrored in each move. Prints the diagram's
//! size and its model count over its own variables, which every mode
//! agrees on.
//!
//! `rotate` ignores `p`, `blocks` and `mirror`: it moves
//! `(x1 ∨ x3) ∧ (x2 ∨ x4)` back and forth between a balanced home, where its
//! projection is `((x1 x2) (x3 x4))`, and one where it is
//! `(x1 (x2 (x3 x4)))`, one rotation each move. Its cost per move should not
//! depend on `h`: the difference of two `reps` counts measures it.
//!
//! ```text
//! cargo bench --bench transport -- <h> <p> <blocks> <none|transport|embed|rotate> [reps] [mirror]
//! ```
//!
//! Without arguments, as `cargo test` runs it: `1024 4 4 transport 4 mirror`,
//! then `1024 0 0 rotate 4`.


use tididi::vtree::{Home, VarId, Vtree, VtreeIdx, VtreeNode};
use tididi::{Engine, Tdd};

fn mirror_image(home: &Vtree) -> Vtree {
    let mut nodes = Vec::with_capacity(home.num_nodes());
    let mut copy_of = vec![VtreeIdx(0); home.num_nodes()];
    for t in home.bottomup() {
        copy_of[t.idx()] = VtreeIdx(nodes.len() as u32);
        nodes.push(match *home.node(t) {
            VtreeNode::Leaf { var, .. } => VtreeNode::Leaf { var, parent: None },
            VtreeNode::Internal { left, right, .. } => {
                VtreeNode::Internal { left: copy_of[right.idx()], right: copy_of[left.idx()], parent: None }
            }
        });
    }
    Vtree::from_nodes(nodes, copy_of[home.root().idx()], home.num_vars()).expect("a mirror image of a tree")
}

/// `rotate`: `reps` moves of a 4-leaf diagram, each one rotation, between
/// two homes of `h` variables.
fn rotate(h: u32, reps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let balanced = Home::new(Vtree::balanced(h));
    let mut order: Vec<VarId> = (5..=h).map(VarId).collect();
    for (var, at) in [(1, 0), (2, h / 2), (3, 3 * h / 4), (4, 3 * h / 4 + 1)] {
        order.insert(at as usize, VarId(var));
    }
    let homes = [balanced.clone(), Home::new(Vtree::balanced_over(&order)?)];
    let eng = Engine::new();
    let own = Vtree::projection(&balanced, (1..=4).map(VarId))?;
    let mut f = eng.and_clause(eng.clause(&own, [1, 3])?, &[2, 4])?;
    let mut rotations = 0;
    for r in 0..reps {
        let (g, stats) = eng.transport(f, &homes[(r + 1) % 2], |v| v, usize::MAX).map_err(|r| r.error)?;
        rotations += stats.rotations;
        f = g;
    }
    println!(
        "h {h} mode rotate reps {reps} rotations {rotations} nodes {} pairs {} count {}",
        f.node_count(),
        f.pair_count(),
        eng.model_count(&f)? >> (h - 4),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    if args.is_empty() {
        rotate(1024, 4)?;
        args = ["1024", "4", "4", "transport", "4", "mirror"].map(String::from).to_vec();
    }
    let usage = "usage: transport <h> <p> <blocks> <none|transport|embed|rotate> [reps] [mirror]";
    let h: u32 = args.first().ok_or(usage)?.parse::<u32>()?.next_power_of_two();
    let p: u32 = args.get(1).ok_or(usage)?.parse()?;
    let blocks: u32 = args.get(2).ok_or(usage)?.parse()?;
    let mode = args.get(3).map(String::as_str).ok_or(usage)?;
    if !matches!(mode, "none" | "transport" | "embed" | "rotate") {
        return Err(usage.into());
    }
    let reps: usize = args.get(4).map_or(Ok(1), |r| r.parse())?;
    if mode == "rotate" {
        if h < 8 || args.get(5).is_some() {
            return Err("rotate needs h of at least 8 and takes no mirror".into());
        }
        return rotate(h, reps);
    }
    let mirror = match args.get(5).map(String::as_str) {
        None => false,
        Some("mirror") => true,
        Some(_) => return Err(usage.into()),
    };
    let width = (2 * p).next_power_of_two();
    let stride = (h / blocks.max(1)) / width * width;
    if p == 0 || blocks == 0 || stride == 0 {
        return Err("p and blocks must be positive and the blocks must fit in the home".into());
    }

    let home = Home::new(Vtree::balanced(h));
    let homes = [home.clone(), Home::new(mirror_image(home.vtree()))];
    let eng = Engine::new();
    let mut f: Option<Tdd> = None;
    for b in 0..blocks {
        let first = 1 + b * stride;
        let x = |i: u32| first + i;
        let y = |i: u32| first + width / 2 + i;
        let vars = (0..p).flat_map(|i| [VarId(x(i)), VarId(y(i))]);
        let mut block = Tdd::one(&Vtree::projection(&home, vars)?);
        for i in 0..p {
            let (x, y) = (x(i) as i32, y(i) as i32);
            block = eng.and_clause(block, &[x, -y])?;
            block = eng.and_clause(block, &[-x, y])?;
        }
        f = Some(match f {
            Some(f) => eng.and(f, block)?,
            None => block,
        });
    }
    let mut f = f.expect("a block");

    for r in 0..reps {
        let to = &homes[usize::from(mirror) * ((r + 1) % 2)];
        f = match mode {
            "transport" => eng.transport(f, to, |v| v, 0).map_err(|r| r.error)?.0,
            "embed" => {
                let (mut g, _) = eng.embed_moving(f, to.vtree(), |v| v).map_err(|r| r.error)?;
                eng.trim(&mut g, to)?;
                g
            }
            _ => f,
        };
    }
    println!(
        "h {h} p {p} blocks {blocks} mode {mode} reps {reps} mirror {mirror} nodes {} pairs {} leaves {} count {}",
        f.node_count(),
        f.pair_count(),
        f.vtree().num_leaves(),
        eng.model_count(&f)? >> (h - f.vtree().num_leaves()),
    );
    Ok(())
}
