//! The formulas and vtrees a test runs on, fixed and seeded.

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::vtree::{VarId, Vtree};

/// The seeded generator every randomized sweep in the crate draws from. One
/// stream per seed, reproducible across runs and platforms: a linear
/// congruential step, with the low bits — which have short periods — dropped.
#[derive(Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// The next draw. The top 31 bits of the state, so a caller may take the
    /// remainder by any small modulus without inheriting a short period.
    pub fn next_u64(&mut self) -> u64 {
        self.state =
            self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state >> 33
    }

    /// A draw in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A fair coin.
    pub fn coin(&mut self) -> bool {
        self.next_u64().is_multiple_of(2)
    }
}

/// How wide and how many, for [`rand_cnf`].
#[derive(Clone, Copy, Debug)]
pub struct CnfShape {
    /// Clause count, drawn from `1..=clauses`.
    pub clauses: usize,
    /// Literals per clause, drawn from `1..=width`. A variable drawn twice in
    /// one clause is dropped, so a clause may come out narrower.
    pub width: usize,
}

/// A random DIMACS-style formula over `num_vars` variables. Repeated variables
/// within a clause are dropped rather than redrawn, so the clause widths are a
/// distribution rather than a constant — which is what the sweeps want.
pub fn rand_cnf(rng: &mut Lcg, num_vars: u32, shape: CnfShape) -> Vec<Vec<i32>> {
    let nclauses = 1 + rng.below(shape.clauses as u64) as usize;
    let mut out = Vec::with_capacity(nclauses);
    for _ in 0..nclauses {
        let width = 1 + rng.below(shape.width.min(num_vars as usize) as u64) as usize;
        let mut seen = vec![false; num_vars as usize];
        let mut clause = Vec::with_capacity(width);
        for _ in 0..width {
            let v = rng.below(u64::from(num_vars)) as usize;
            if seen[v] {
                continue;
            }
            seen[v] = true;
            let lit = v as i32 + 1;
            clause.push(if rng.coin() { lit } else { -lit });
        }
        if !clause.is_empty() {
            out.push(clause);
        }
    }
    out
}

/// Formulas covering SAT, UNSAT, unit, wide, and don't-care shapes.
#[cfg(test)]
pub fn test_cases() -> Vec<(u32, Vec<Vec<i32>>)> {
    vec![
        (3, vec![vec![1, 2], vec![-2, 3], vec![-1, -3]]),
        (4, vec![vec![1, 2], vec![3, 4], vec![-1, -3], vec![-2, -4]]),
        (4, vec![vec![1], vec![2, -3], vec![-1, 3, 4], vec![-2, -4]]),
        (3, vec![vec![1, 2, 3]]),
        (2, vec![vec![1, 2], vec![-1, -2], vec![1, -2], vec![-1, 2]]),
        (1, vec![vec![1], vec![-1]]),
        (2, vec![vec![1], vec![2], vec![-1, -2]]),
        (3, vec![vec![1], vec![-1], vec![2, 3]]),
        (4, vec![vec![1, 2], vec![-1, -2], vec![1, -2], vec![-1, 2],
                 vec![3, 4], vec![-3, -4], vec![3, -4], vec![-3, 4]]),
        (3, vec![vec![1]]),
        (3, vec![vec![-2]]),
        (4, vec![vec![1, 2, 3, 4]]),
        (4, vec![vec![1, 2], vec![3], vec![-4]]),
        (4, vec![vec![1, 2], vec![-1, -2]]),
        (5, vec![vec![1]]),
        (3, vec![vec![1, 2], vec![-1, -2], vec![3]]),
        (6, vec![vec![1, 2], vec![-3, -4]]),
        (6, vec![vec![1, 2], vec![3, 4], vec![5, 6],
                 vec![-1, -3], vec![-1, -5], vec![-3, -5],
                 vec![-2, -4], vec![-2, -6], vec![-4, -6]]),
        (5, vec![vec![-1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5]]),
        (7, vec![vec![-1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5],
                 vec![-5, 6], vec![-6, 7]]),
        (8, vec![vec![-1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5],
                 vec![-5, 6], vec![-6, 7], vec![-7, 8]]),
        (12, vec![vec![1, 3], vec![-3, 5], vec![5, -7], vec![-1, 7],
                  vec![3, -5, 9], vec![-7, 9], vec![1, -9], vec![-3, -9, 11]]),
        (4, vec![vec![-1, 2], vec![1, -2], vec![-2, 3], vec![2, -3],
                 vec![-3, 4], vec![3, -4]]),
        (4, vec![vec![-1, -2], vec![-1, -3], vec![-1, -4],
                 vec![-2, -3], vec![-2, -4], vec![-3, -4]]),
        (4, vec![vec![1, 2, 3, 4],
                 vec![-1, -2], vec![-1, -3], vec![-1, -4],
                 vec![-2, -3], vec![-2, -4], vec![-3, -4]]),
        (2, vec![vec![1, 2], vec![-1, -2]]),
        (3, vec![vec![1, 2, 3], vec![-1, -2, 3], vec![-1, 2, -3], vec![1, -2, -3]]),
        (4, vec![vec![1, 2], vec![-1, 3], vec![-2, 4], vec![-3, -4]]),
        (5, vec![vec![1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5], vec![-5, -1]]),
        (4, vec![vec![-1, -2, 3], vec![-3, 4], vec![-4, -1], vec![1]]),
        (5, vec![vec![1], vec![-1, 2], vec![-2, 3], vec![-1, -3, 4], vec![-4, 5]]),
        (6, vec![vec![1], vec![2, 3], vec![-1, -2, -3], vec![4, 5, 6],
                 vec![-4, -5], vec![-5, -6]]),
        (3, vec![vec![1, 2], vec![1, 3], vec![2, 3],
                 vec![-1, -2], vec![-1, -3], vec![-2, -3],
                 vec![1, 2, 3]]),
        (8, vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]),
        (8, vec![vec![1, 2, 3, 4, 5, 6, 7, 8],
                 vec![-1, -2, -3, -4, -5, -6, -7, -8]]),
        (6, vec![vec![1, 2], vec![-1, -2], vec![3, 4], vec![-3, -4],
                 vec![5, 6], vec![-5, -6]]),
        (1, vec![vec![1]]),
        (1, vec![vec![-1]]),
    ]
}

/// One vtree per shape a test wants to see a formula compiled against:
/// balanced, linear, three random seeds, the reversed linear order, and an
/// interleaved linear order.
#[cfg(test)]
pub fn vtree_shapes(num_vars: u32) -> Vec<(&'static str, Arc<Vtree>)> {
    let mut shapes = vec![
        ("balanced", Arc::new(Vtree::balanced(num_vars))),
        ("linear", Arc::new(Vtree::linear(num_vars))),
        ("random(0)", Arc::new(Vtree::random(num_vars, 0))),
        ("random(1)", Arc::new(Vtree::random(num_vars, 1))),
        ("random(42)", Arc::new(Vtree::random(num_vars, 42))),
    ];
    if num_vars >= 2 {
        let reversed: Vec<VarId> = (0..num_vars).rev().map(VarId).collect();
        shapes.push(("linear reversed", Arc::new(Vtree::linear_over(&reversed))));
    }
    if num_vars >= 4 {
        let mut interleaved: Vec<VarId> = (1..num_vars).step_by(2).map(VarId).collect();
        interleaved.extend((0..num_vars).step_by(2).map(VarId));
        shapes.push(("linear interleaved", Arc::new(Vtree::linear_over(&interleaved))));
    }
    shapes
}

/// N-queens as DIMACS-style clauses over `n * n` variables: one row clause per
/// row, and a pairwise exclusion for every row, column and diagonal conflict.
#[cfg(test)]
pub fn queens_clauses(n: i32) -> (u32, Vec<Vec<i32>>) {
    let var = |row: i32, col: i32| row * n + col + 1;
    let mut clauses: Vec<Vec<i32>> = (0..n).map(|r| (0..n).map(|c| var(r, c)).collect()).collect();
    for r in 0..n {
        for c1 in 0..n {
            for c2 in (c1 + 1)..n {
                clauses.push(vec![-var(r, c1), -var(r, c2)]);
            }
        }
    }
    for c in 0..n {
        for r1 in 0..n {
            for r2 in (r1 + 1)..n {
                clauses.push(vec![-var(r1, c), -var(r2, c)]);
            }
        }
    }
    for r1 in 0..n {
        for c1 in 0..n {
            for r2 in (r1 + 1)..n {
                for c2 in 0..n {
                    if (r1 - r2).abs() == (c1 - c2).abs() {
                        clauses.push(vec![-var(r1, c1), -var(r2, c2)]);
                    }
                }
            }
        }
    }
    ((n * n) as u32, clauses)
}
