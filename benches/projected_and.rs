//! Conjoin random 3-literal clauses on a balanced vtree of `n` variables,
//! each diagram either on the projection of the vtree on its variables or
//! on the whole vtree.
//!
//! The first half of the clauses are built as diagrams and conjoined in
//! pairs; the second half are folded into the pairs by `and_clause`, two to
//! a pair. Clauses on random variables of a balanced vtree cross its root,
//! so a pair's width doubles with each clause folded in. Prints the sum
//! over the results of the fraction of their models, in units of 2^-20,
//! which both modes agree on.
//!
//! ```text
//! cargo bench --bench projected_and -- <n> <projected|smooth> [clauses]
//! ```
//!
//! Without arguments, as `cargo test` runs it: `64 projected 40`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tididi::vtree::{Home, VarId, Vtree};
use tididi::{Engine, Tdd};

/// The system allocator, counting allocations.
struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    if args.is_empty() {
        args = ["64", "projected", "40"].map(String::from).to_vec();
    }
    let usage = "usage: projected_and <n> <projected|smooth> [clauses] | <n> nested <k> <calls> [balanced|linear|alternate] | <n> union <shared> <calls>";
    let n: u32 = args.first().ok_or(usage)?.parse()?;
    let projected = match args.get(1).map(String::as_str) {
        Some("projected") => true,
        Some("smooth") => false,
        Some("nested" | "union") => true,
        _ => return Err(usage.into()),
    };
    match args[1].as_str() {
        "nested" => return nested(n, &args[2..]),
        "union" => return union(n, &args[2..]),
        _ => {}
    }
    let count: usize = args.get(2).map_or(Ok(1000), |c| c.parse())?;
    if n < 3 {
        return Err("n must be at least 3".into());
    }

    let home = Home::new(Vtree::balanced(n));
    let vtree_of = |vars: &[u32]| -> Result<Arc<Vtree>, Box<dyn std::error::Error>> {
        Ok(match projected {
            true => Vtree::projection(&home, vars.iter().map(|&v| VarId(v)))?,
            false => Arc::clone(&home),
        })
    };

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |below: u32| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % u64::from(below)) as u32
    };
    let clauses: Vec<[i32; 3]> = (0..count)
        .map(|_| {
            let mut vars = [0u32; 3];
            for i in 0..3 {
                vars[i] = loop {
                    let v = 1 + next(n);
                    if !vars[..i].contains(&v) {
                        break v;
                    }
                };
            }
            vars.map(|v| if next(2) == 0 { v as i32 } else { -(v as i32) })
        })
        .collect();

    let eng = Engine::new();
    let (built, folded) = clauses.split_at(count / 2);
    let mut results = Vec::new();
    for pair in built.chunks(2) {
        let mut acc: Option<Tdd> = None;
        for lits in pair {
            let vars = lits.map(|l| l.unsigned_abs());
            let c = eng.clause(&vtree_of(&vars)?, lits)?;
            acc = Some(match acc {
                Some(acc) => eng.and(acc, c)?,
                None => c,
            });
        }
        results.extend(acc);
    }
    let per_pair = folded.len().div_ceil(results.len().max(1)).max(1);
    let mut groups = folded.chunks(per_pair);
    let results = results
        .into_iter()
        .map(|mut f| {
            for lits in groups.next().unwrap_or_default() {
                f = eng.and_clause(f, lits)?;
            }
            Ok(f)
        })
        .collect::<Result<Vec<Tdd>, tididi::OperationError>>()?;

    let mut sum = 0u64;
    for f in &results {
        let scaled = (eng.model_count(f)? << 20u32) >> n;
        sum += u64::try_from(scaled)?;
    }
    println!("n {n} mode {} clauses {count} diagrams {} sum {sum}", args[1], results.len());
    Ok(())
}

/// `nested <k> <calls> [balanced|linear|alternate]`: `calls` conjunctions
/// `F = and(F, G)`, `F` owned on the projection on `k` variables spread
/// evenly over the home of `n`, a chain of 2-literal clauses over
/// consecutive ones, and `G` a 2-literal clause on two consecutive
/// variables of `F`, borrowed, on their projection, drawn from 1000 built
/// before the calls. Every clause holds where a variable is true unless
/// divisible by 3, so no conjunction is false. `balanced` homes are
/// balanced, with `G` on a random pair; `linear` homes right-linear, with
/// `G` on the two deepest variables; `alternate` alternates two `F`, on
/// `n / 2` and on `k` variables of a balanced home. Prints the node count
/// of the last `F`s.
fn nested(n: u32, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: projected_and <n> nested <k> <calls> [balanced|linear|alternate]";
    let k: u32 = args.first().ok_or(usage)?.parse()?;
    let calls: usize = args.get(1).ok_or(usage)?.parse()?;
    let shape = args.get(2).map_or("balanced", String::as_str);
    if k < 2 || k > n {
        return Err("k must be at least 2 and at most n".into());
    }
    let home = Home::new(match shape {
        "balanced" | "alternate" => Vtree::balanced(n),
        "linear" => Vtree::linear(n),
        _ => return Err(usage.into()),
    });
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |below: u32| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % u64::from(below)) as u32
    };
    let held = |v: u32| if v.is_multiple_of(3) { -(v as i32) } else { v as i32 };
    let eng = Engine::new();
    let spread = |k: u32| (0..k).map(|i| 1 + i * (n / k)).collect::<Vec<u32>>();
    let chain = |vars: &[u32]| -> Result<Tdd, Box<dyn std::error::Error>> {
        let own = Vtree::projection(&home, vars.iter().map(|&v| VarId(v)))?;
        let mut f = eng.cube(&own, std::iter::empty::<i32>())?;
        for w in vars.windows(2) {
            f = eng.and_clause(f, &[held(w[0]), -held(w[1])])?;
        }
        Ok(f)
    };
    let small = spread(k);
    let mut fs = match shape {
        "alternate" => vec![Some(chain(&spread(n / 2))?), Some(chain(&small)?)],
        _ => vec![Some(chain(&small)?)],
    };
    let mut gs = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let j = match shape {
            "linear" => small.len() - 2,
            _ => next(k - 1) as usize,
        };
        let (a, b) = (small[j], small[j + 1]);
        let second = if next(2) == 0 { held(b) } else { -held(b) };
        gs.push(eng.clause(&Vtree::projection(&home, [VarId(a), VarId(b)])?, [held(a), second])?);
    }
    for i in 0..calls {
        let at = i % fs.len();
        let f = fs[at].take().expect("put back after each call");
        fs[at] = Some(eng.and(f, &gs[i % gs.len()])?);
    }
    let nodes: usize = fs.iter().flatten().map(Tdd::node_count).sum();
    println!("n {n} mode nested k {k} calls {calls} shape {shape} nodes {nodes}");
    Ok(())
}

/// `union <shared> <calls>`: `calls` conjunctions `and(&F, &G)` of borrowed
/// operands, each on the projection on 128 variables of a balanced home of
/// `n`, sharing `shared` of them, a chain of 2-literal clauses over
/// consecutive ones as `nested` builds it; drawn from 16 pairs built before
/// the calls. Prints the node count of the last result and the allocations
/// the calls made, with the instructions they take read as the count at
/// `calls` less the count at 0.
fn union(n: u32, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: projected_and <n> union <shared> <calls>";
    let shared: u32 = args.first().ok_or(usage)?.parse()?;
    let calls: usize = args.get(1).ok_or(usage)?.parse()?;
    const K: u32 = 128;
    if shared >= K || n < 2 * K {
        return Err("shared must be below 128 and n at least 256".into());
    }
    let home = Home::new(Vtree::balanced(n));
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |below: u32| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % u64::from(below)) as u32
    };
    let held = |v: u32| if v.is_multiple_of(3) { -(v as i32) } else { v as i32 };
    let eng = Engine::new();
    let chain = |vars: &[u32]| -> Result<Tdd, Box<dyn std::error::Error>> {
        let own = Vtree::projection(&home, vars.iter().map(|&v| VarId(v)))?;
        let mut f = eng.cube(&own, std::iter::empty::<i32>())?;
        for w in vars.windows(2) {
            f = eng.and_clause(f, &[held(w[0]), -held(w[1])])?;
        }
        Ok(f)
    };
    let mut pairs = Vec::with_capacity(16);
    for _ in 0..16 {
        let mut taken = vec![false; n as usize + 1];
        let mut draw = |taken: &mut Vec<bool>| loop {
            let v = 1 + next(n);
            if !taken[v as usize] {
                taken[v as usize] = true;
                break v;
            }
        };
        let mut f: Vec<u32> = (0..K).map(|_| draw(&mut taken)).collect();
        let mut g: Vec<u32> = (0..K - shared).map(|_| draw(&mut taken)).collect();
        for _ in 0..shared {
            let v = loop {
                let v = f[next(K) as usize];
                if !g.contains(&v) {
                    break v;
                }
            };
            g.push(v);
        }
        f.sort_unstable();
        g.sort_unstable();
        pairs.push((chain(&f)?, chain(&g)?));
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let mut nodes = 0;
    for i in 0..calls {
        let (f, g) = &pairs[i % pairs.len()];
        nodes = eng.and(f, g)?.node_count();
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    println!("n {n} mode union shared {shared} calls {calls} nodes {nodes} allocations {allocations}");
    Ok(())
}
