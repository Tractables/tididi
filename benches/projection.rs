//! Build projections of a random home vtree of `h` variables, `reps` times,
//! for a count of instructions per node built.
//!
//! - `project`: [`Vtree::projection`] on `k` random variables.
//! - `union`: [`Vtree::union`] of two projections of `k` random variables.
//! - `sub`: [`Vtree::sub_projection`] keeping `k` of the leaves of a
//!   projection of `2k` random variables.
//! - `inorder`: [`Vtree::from_in_order`] on `k` local variables and the
//!   depths of a projection on `k` random variables, which is what a
//!   consumer's own projector would call.
//! - `none`: the setup only, which every mode shares.
//!
//! Prints the number of nodes built; a mode's count of instructions minus
//! `none`'s, over that number, is its price per node.
//!
//! ```text
//! cargo bench --bench projection -- <h> <k> <none|project|union|sub|inorder> [reps]
//! ```
//!
//! Without arguments, as `cargo test` runs it: `1000 8 union 100`.

use std::sync::Arc;

use tididi::vtree::{Home, VarId, Vtree, VtreeIdx};

const POOL: usize = 64;

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) % n
    }

    fn vars(&mut self, h: u32, k: usize) -> Vec<VarId> {
        let mut vars: Vec<VarId> = Vec::with_capacity(k);
        while vars.len() < k {
            let var = VarId(1 + self.below(u64::from(h)) as u32);
            if !vars.contains(&var) {
                vars.push(var);
            }
        }
        vars
    }
}

fn depth(vtree: &Vtree, mut t: VtreeIdx) -> u32 {
    let mut depth = 0;
    while let Some(parent) = vtree.node(t).parent() {
        t = parent;
        depth += 1;
    }
    depth
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    if args.is_empty() {
        args = ["1000", "8", "union", "100"].map(String::from).to_vec();
    }
    let usage = "usage: projection <h> <k> <none|project|union|sub|inorder> [reps]";
    let h: u32 = args.first().ok_or(usage)?.parse()?;
    let k: usize = args.get(1).ok_or(usage)?.parse()?;
    let mode = args.get(2).map(String::as_str).ok_or(usage)?;
    if !matches!(mode, "none" | "project" | "union" | "sub" | "inorder") {
        return Err(usage.into());
    }
    let reps: usize = args.get(3).map_or(Ok(1), |r| r.parse())?;
    if k == 0 || 2 * k as u64 > u64::from(h) {
        return Err("k must be positive and 2k at most h".into());
    }

    let home = Home::new(Vtree::random(h, 7));
    let rng = &mut Rng(1);
    let sets: Vec<Vec<VarId>> = (0..POOL).map(|_| rng.vars(h, k)).collect();
    let pairs: Vec<(Arc<Vtree>, Arc<Vtree>)> = (0..POOL)
        .map(|_| Ok((Vtree::projection(&home, rng.vars(h, k))?, Vtree::projection(&home, rng.vars(h, k))?)))
        .collect::<Result<_, tididi::vtree::VtreeError>>()?;
    let wides: Vec<(Arc<Vtree>, Vec<bool>)> = (0..POOL)
        .map(|_| {
            let wide = Vtree::projection(&home, rng.vars(h, 2 * k))?;
            let mut keep = vec![false; wide.num_leaves() as usize];
            let mut kept = 0;
            while kept < k {
                let i = rng.below(keep.len() as u64) as usize;
                kept += usize::from(!std::mem::replace(&mut keep[i], true));
            }
            Ok((wide, keep))
        })
        .collect::<Result<_, tididi::vtree::VtreeError>>()?;
    let in_orders: Vec<(Vec<VarId>, Vec<u32>)> = sets
        .iter()
        .map(|vars| {
            let mut ranks: Vec<u32> = vars.iter().map(|&v| home.rank(v).expect("a home variable")).collect();
            ranks.sort_unstable();
            let depths = ranks.windows(2).map(|w| depth(home.vtree(), home.node_over(w[0], w[1]))).collect();
            ((1..=k as u32).map(VarId).collect(), depths)
        })
        .collect();

    let mut nodes = 0usize;
    for r in 0..reps {
        let i = r % POOL;
        nodes += match mode {
            "project" => Vtree::projection(&home, sets[i].iter().copied())?.num_nodes(),
            "union" => Vtree::union(&pairs[i].0, &pairs[i].1)?.num_nodes(),
            "sub" => {
                let (wide, keep) = &wides[i];
                wide.sub_projection(&home, |leaf| keep[leaf.idx()])?.num_nodes()
            }
            "inorder" => {
                let (locals, depths) = &in_orders[i];
                Vtree::from_in_order(locals, depths, k as u32)?.num_nodes()
            }
            _ => 0,
        };
    }
    println!("h {h} k {k} mode {mode} reps {reps} nodes {nodes}");
    Ok(())
}
