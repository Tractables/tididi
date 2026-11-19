//! Operations on diagrams seated on projections of a home vtree of `n`
//! variables, one site per operation whose cost may grow with `n` or with
//! the larger operand rather than with the diagram it returns.
//!
//! When `n` is a power of two of at least 2^14, the random variables are
//! drawn from the 2^14 variables `1 + i n / 2^14`, so the projections, and
//! the diagrams on them, have the same shape at every such `n`.
//!
//! A run with `calls` 0 does the setup alone; a site's count of
//! instructions minus that run's, over `calls`, is its price per call.
//!
//! - `eval <n> <k> <calls>`: [`Engine::evaluate`] of a clause of `k`
//!   literals, on the projection of a balanced home on its variables, with
//!   a weight algebra over `f64` that keeps the default `free` (its free
//!   factor is one, so the value does not depend on `n`).
//! - `unit <n> <k> <calls>`: the same with `free` answering `None`, as an
//!   algebra whose weights sum to one may.
//! - `count <n> <k> <calls>`: the same clause, model counts modulo the prime
//!   2^61 - 1, with `free` answering `2^len` by repeated squaring.
//! - `and <n> <k> <calls>`: [`Engine::and`] of an owned clone of `F` and a
//!   borrowed `G`. `F` is a chain of 2-literal clauses over consecutive
//!   variables of `k` random variables of a balanced home, on their
//!   projection; `G` is a clause on two adjacent variables of `F`, on the
//!   projection on those two.
//! - `restoring <n> <k> <calls>`: [`Engine::and_restoring`] of the same `F`
//!   and `G`, both borrowed.
//! - `clone <n> <k> <calls>`: the clone of `F` that `and` makes, dropped.
//! - `rehome <n> <k> <calls>`: [`Engine::transport`] of a borrowed chain over
//!   `k` random variables of a balanced home onto a random home of the same
//!   variables, where the projection differs in shape, so it restructures.
//!
//! Prints a checksum of the results.
//!
//! ```text
//! cargo bench --bench zero_cost -- <site> <n> <k> <calls>
//! ```
//!
//! Without arguments, as `cargo test` runs it: `and 1000 64 10`.

use std::hint::black_box;
use std::sync::Arc;

use tididi::diagram::{EvalAlgebra, LeafLabel};
use tididi::vtree::{FreeVars, Home, VarId, Vtree};
use tididi::{Engine, Tdd};

type Error = Box<dyn std::error::Error>;

const GRID: u32 = 1 << 14;

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) % n
    }

    fn sorted_vars(&mut self, n: u32, k: usize) -> Vec<u32> {
        let grid = if n.is_power_of_two() && n >= GRID { GRID } else { n };
        let mut vars = std::collections::BTreeSet::new();
        while vars.len() < k {
            vars.insert(1 + self.below(u64::from(grid)) as u32 * (n / grid));
        }
        vars.into_iter().collect()
    }
}

/// Weights read from a table, as a consumer's are. `One` is computed as the
/// sum of the two literals so the default `free` cannot fold away; the
/// weights are dyadic, so the sum is exactly one.
struct Weights(Vec<f64>);

impl EvalAlgebra for Weights {
    type Value = f64;
    fn zero(&self) -> f64 {
        0.0
    }
    fn one(&self) -> f64 {
        1.0
    }
    fn leaf(&self, var: VarId, label: LeafLabel) -> f64 {
        let pos = self.0[var.idx()];
        match label {
            LeafLabel::Pos => pos,
            LeafLabel::Neg => 1.0 - pos,
            LeafLabel::One => pos + (1.0 - pos),
            LeafLabel::Zero => 0.0,
        }
    }
    fn add_assign(&self, acc: &mut f64, other: &f64) {
        *acc += other;
    }
    fn mul(&self, a: &f64, b: &f64) -> f64 {
        a * b
    }
}

/// [`Weights`] declaring that every variable's weights sum to one.
struct Unit(Weights);

impl EvalAlgebra for Unit {
    type Value = f64;
    fn zero(&self) -> f64 {
        0.0
    }
    fn one(&self) -> f64 {
        1.0
    }
    fn leaf(&self, var: VarId, label: LeafLabel) -> f64 {
        self.0.leaf(var, label)
    }
    fn add_assign(&self, acc: &mut f64, other: &f64) {
        *acc += other;
    }
    fn mul(&self, a: &f64, b: &f64) -> f64 {
        a * b
    }
    fn free(&self, _: &FreeVars<'_>) -> Option<f64> {
        None
    }
}

const P: u64 = (1 << 61) - 1;

/// Model counts modulo [`P`].
struct Count;

impl EvalAlgebra for Count {
    type Value = u64;
    fn zero(&self) -> u64 {
        0
    }
    fn one(&self) -> u64 {
        1
    }
    fn leaf(&self, _: VarId, label: LeafLabel) -> u64 {
        if label == LeafLabel::One { 2 } else { 1 }
    }
    fn add_assign(&self, acc: &mut u64, other: &u64) {
        *acc = (*acc + other) % P;
    }
    fn mul(&self, a: &u64, b: &u64) -> u64 {
        (u128::from(*a) * u128::from(*b) % u128::from(P)) as u64
    }
    fn free(&self, free: &FreeVars<'_>) -> Option<u64> {
        let (mut power, mut base, mut e) = (1, 2, free.len());
        while e > 0 {
            if e & 1 == 1 {
                power = self.mul(&power, &base);
            }
            base = self.mul(&base, &base);
            e >>= 1;
        }
        Some(power)
    }
}

fn projection(home: &Home, vars: &[u32]) -> Result<Arc<Vtree>, Error> {
    Ok(Vtree::projection(home, vars.iter().map(|&v| VarId(v)))?)
}

fn chain(eng: &Engine, home: &Home, vars: &[u32]) -> Result<Tdd, Error> {
    let own = projection(home, vars)?;
    let lit = |i: usize| vars[i] as i32 * if i.is_multiple_of(3) { -1 } else { 1 };
    let mut f = eng.clause(&own, [lit(0), lit(1)])?;
    for i in 1..vars.len() - 1 {
        f = eng.and_clause(f, &[lit(i), -lit(i + 1)])?;
    }
    Ok(f)
}

fn main() -> Result<(), Error> {
    let mut args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    if args.is_empty() {
        args = ["and", "1000", "64", "10"].map(String::from).to_vec();
    }
    let usage = "usage: zero_cost <eval|unit|count|and|restoring|clone|rehome> <n> <k> <calls>";
    let site = args.first().ok_or(usage)?.as_str();
    let n: u32 = args.get(1).ok_or(usage)?.parse()?;
    let k: usize = args.get(2).ok_or(usage)?.parse()?;
    let calls: usize = args.get(3).ok_or(usage)?.parse()?;
    if k < 2 || k as u64 > u64::from(n) {
        return Err("k must be at least 2 and at most n".into());
    }

    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(n));
    let rng = &mut Rng(1);
    let vars = rng.sorted_vars(n, k);
    let mut sum = 0.0f64;
    match site {
        "eval" | "unit" | "count" => {
            let lits: Vec<i32> = vars.iter().map(|&v| v as i32).collect();
            let f = eng.clause(&projection(&home, &vars)?, lits)?;
            let weights = Weights((0..n).map(|v| 0.25 + f64::from(v % 7) / 16.0).collect());
            let unit = Unit(weights);
            for _ in 0..calls {
                sum += match site {
                    "eval" => eng.evaluate(black_box(&f), &unit.0)?,
                    "unit" => eng.evaluate(black_box(&f), &unit)?,
                    _ => eng.evaluate(black_box(&f), &Count)? as f64,
                };
            }
        }
        "and" | "restoring" | "clone" => {
            let f = chain(&eng, &home, &vars)?;
            let (a, b) = (vars[k / 3], vars[k / 3 + 1]);
            let g = eng.clause(&projection(&home, &[a, b])?, [-(a as i32), b as i32])?;
            for _ in 0..calls {
                let r = match site {
                    "and" => eng.and(black_box(&f).clone(), &g)?,
                    "restoring" => eng.and_restoring(black_box(&f), &g).map_err(|r| r.error)?,
                    _ => black_box(&f).clone(),
                };
                sum += r.node_count() as f64;
            }
        }
        "rehome" => {
            let f = chain(&eng, &home, &vars)?;
            let other = Home::new(Vtree::random(n, 7));
            if eng.transport(&f, &other, |v| v, 0).is_ok() {
                return Err("the projections have one shape: transport would not restructure".into());
            }
            for _ in 0..calls {
                let (r, _) = eng.transport(black_box(&f), &other, |v| v, usize::MAX).map_err(|r| r.error)?;
                sum += r.node_count() as f64;
            }
        }
        _ => return Err(usage.into()),
    }
    println!("site {site} n {n} k {k} calls {calls} sum {sum}");
    Ok(())
}
