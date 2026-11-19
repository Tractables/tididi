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
//! - `extend`: [`Engine::and_clause`](tididi::Engine::and_clause) of a
//!   copy of the clause on `k` random variables, on their projection, and
//!   a clause on `m` others, which moves the copy onto the projection on
//!   all `k + m` before the clause walk.
//! - `copy`: the copy alone, to take from `extend`.
//! - `none`: the setup only, which every mode shares.
//!
//! Prints the number of nodes built and the allocations made; a mode's
//! count of instructions minus `none`'s, over that number, is its price per
//! node.
//!
//! ```text
//! cargo bench --bench projection -- <h> <k> <none|project|union|sub|inorder|extend|copy> [reps] [m]
//! ```
//!
//! Without arguments, as `cargo test` runs it: `1000 8 union 100`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tididi::vtree::{Home, VarId, Vtree, VtreeIdx};
use tididi::{Engine, Tdd};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

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
    let usage = "usage: projection <h> <k> <none|project|union|sub|inorder|extend|copy> [reps] [m]";
    let h: u32 = args.first().ok_or(usage)?.parse()?;
    let k: usize = args.get(1).ok_or(usage)?.parse()?;
    let mode = args.get(2).map(String::as_str).ok_or(usage)?;
    if !matches!(mode, "none" | "project" | "union" | "sub" | "inorder" | "extend" | "copy") {
        return Err(usage.into());
    }
    let reps: usize = args.get(3).map_or(Ok(1), |r| r.parse())?;
    let m: usize = args.get(4).map_or(Ok(1), |m| m.parse())?;
    if k == 0 || 2 * k as u64 > u64::from(h) || k + m > h as usize {
        return Err("k must be positive, 2k at most h and k + m at most h".into());
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

    let eng = Engine::new();
    let clauses: Vec<(Tdd, Vec<i32>)> = (0..POOL)
        .map(|_| {
            let both = rng.vars(h, k + m);
            let own = Vtree::projection(&home, both[..k].iter().copied())?;
            let f = Tdd::clause(&own, both[..k].iter().map(|v| v.0 as i32))?;
            Ok((f, both[k..].iter().map(|v| v.0 as i32).collect()))
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    let mut nodes = 0usize;
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
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
            "extend" => {
                let (f, clause) = &clauses[i];
                eng.and_clause(f.clone(), clause)?.vtree().num_nodes()
            }
            "copy" => {
                let copy = clauses[i].0.clone();
                std::hint::black_box(&copy);
                2 * (k + m) - 1
            }
            _ => 0,
        };
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations;
    println!("h {h} k {k} m {m} mode {mode} reps {reps} allocations {allocations} nodes {nodes}");
    Ok(())
}
