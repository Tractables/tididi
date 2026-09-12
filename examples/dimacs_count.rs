//! Count the models of a DIMACS CNF file, save the diagram, and sum variables out.
//!
//! ```sh
//! cargo run --example dimacs_count -- examples/tiny.cnf 5 6 --check
//! ```
//!
//! The first argument is the CNF file; the remaining integers are variables to
//! sum out. `--check` enumerates every assignment and compares, which is only
//! affordable for a small formula.
//!
//! The library takes no part in reading CNF: it operates on the vtree it is
//! given, and clauses reach it as literals. The parser below is therefore part
//! of the example, not of the crate.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use num_bigint::BigUint;
use num_rational::BigRational;

use tididi::apply::{QuantificationStrategy, apply_and_clause, exists_vars};
use tididi::diagram::RationalWeights;
use tididi::io::{load_tdd, save_tdd};
use tididi::query::evaluate;
use tididi::reduce::minimize;
use tididi::vtree::{VarId, Vtree};
use tididi::{Literal, Tdd};

/// A CNF as the variable count and the clauses, each a list of DIMACS literals.
struct Cnf {
    num_vars: u32,
    clauses: Vec<Vec<i32>>,
}

/// Reads the DIMACS subset this example needs: `c` comments, one `p cnf n m`
/// header, and clauses of signed integers terminated by `0`, line breaks
/// anywhere.
fn parse_dimacs(text: &str) -> Result<Cnf, String> {
    let mut num_vars = None;
    let mut clauses = Vec::new();
    let mut current: Vec<i32> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
            continue;
        }
        if let Some(header) = line.strip_prefix("p cnf ") {
            let n = header
                .split_whitespace()
                .next()
                .ok_or("header has no variable count")?;
            num_vars = Some(n.parse::<u32>().map_err(|e| format!("bad header: {e}"))?);
            continue;
        }
        for token in line.split_whitespace() {
            let lit: i32 = token.parse().map_err(|e| format!("bad literal: {e}"))?;
            if lit == 0 {
                clauses.push(std::mem::take(&mut current));
            } else {
                current.push(lit);
            }
        }
    }
    if !current.is_empty() {
        clauses.push(current);
    }
    let num_vars = num_vars.ok_or("no `p cnf` header")?;
    for lit in clauses.iter().flatten() {
        if lit.unsigned_abs() > num_vars {
            return Err(format!("literal {lit} is outside the header's {num_vars}"));
        }
    }
    Ok(Cnf { num_vars, clauses })
}

/// Conjoins the clauses into one diagram over `vtree` and reduces it.
///
/// Each clause goes in through `apply_and_clause`, which never builds the
/// clause as a diagram of its own; the accumulator is count-correct after
/// every clause and canonical after the final `minimize`.
fn compile(cnf: &Cnf, vtree: &Arc<Vtree>) -> Tdd {
    let mut f = Tdd::one(vtree);
    let mut lits: Vec<Literal> = Vec::new();
    for clause in &cnf.clauses {
        lits.clear();
        lits.extend(clause.iter().map(Literal::from));
        f = apply_and_clause(f, &lits);
    }
    minimize(&mut f);
    f
}

/// Counts the models by enumerating every assignment, for `--check`.
fn brute_force_count(cnf: &Cnf) -> u64 {
    assert!(cnf.num_vars < 32, "--check enumerates 2^n assignments");
    let mut models = 0;
    for bits in 0..(1u64 << cnf.num_vars) {
        let holds = |lit: &i32| {
            let value = bits >> (lit.unsigned_abs() - 1) & 1 == 1;
            value == (*lit > 0)
        };
        if cnf.clauses.iter().all(|c| c.iter().any(holds)) {
            models += 1;
        }
    }
    models
}

/// Counts the assignments to the variables left after `summed_out` that extend
/// to a model, for `--check`.
fn brute_force_projected_count(cnf: &Cnf, summed_out: &[u32]) -> u64 {
    assert!(cnf.num_vars < 32, "--check enumerates 2^n assignments");
    let mut seen = std::collections::HashSet::new();
    for bits in 0..(1u64 << cnf.num_vars) {
        let holds = |lit: &i32| {
            let value = bits >> (lit.unsigned_abs() - 1) & 1 == 1;
            value == (*lit > 0)
        };
        if cnf.clauses.iter().all(|c| c.iter().any(holds)) {
            let mut kept = bits;
            for &v in summed_out {
                kept &= !(1u64 << (v - 1));
            }
            seen.insert(kept);
        }
    }
    seen.len() as u64
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: dimacs_count <file.cnf> [vars...] [--check]")?;
    let mut check = false;
    let mut summed_out: Vec<u32> = Vec::new();
    for arg in args {
        if arg == "--check" {
            check = true;
        } else {
            summed_out.push(
                arg.parse()
                    .map_err(|e| format!("`{arg}` is not a variable: {e}"))?,
            );
        }
    }

    let text = fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let cnf = parse_dimacs(&text)?;
    // The free `exists_vars` panics on a variable the vtree does not carry,
    // so a name off the command line is checked here. `Engine::exists_vars`
    // returns that case as an error instead.
    if let Some(&v) = summed_out.iter().find(|&&v| v == 0 || v > cnf.num_vars) {
        return Err(format!(
            "{v} is not one of the formula's {} variables",
            cnf.num_vars
        ));
    }
    println!(
        "{path}: {} variables, {} clauses",
        cnf.num_vars,
        cnf.clauses.len()
    );

    // The library builds no vtree from CNF structure; `balanced` is one of the
    // shapes it offers over a variable count. (`vitri`, at
    // https://github.com/Tractables/vitri, derives a vtree from the formula
    // and writes the `.vtree` format `Vtree::from_text` reads.)
    let vtree = Arc::new(Vtree::balanced(cnf.num_vars));

    let f = compile(&cnf, &vtree);
    let count = f.model_count();
    println!("model count: {count}");
    println!("diagram size: {} pairs, {} nodes", f.pair_count(), f.node_count());

    // Round trip. The `.tdd` format records the diagram and not the vtree, so
    // the reader is handed the vtree the file belongs to.
    let saved = env::temp_dir().join("dimacs_count.tdd");
    save_tdd(&f, &saved).map_err(|e| format!("saving {}: {e}", saved.display()))?;
    let reloaded =
        load_tdd(&saved, &vtree).map_err(|e| format!("loading {}: {e}", saved.display()))?;
    let reloaded_count = reloaded.model_count();
    if reloaded_count != count {
        return Err(format!(
            "round trip changed the count: {count} -> {reloaded_count}"
        ));
    }
    println!("saved to {} and reloaded: count matches", saved.display());

    // Sum variables out. `exists_vars` is existential quantification: the
    // result keeps the whole vtree, so each forgotten variable still ranges
    // over both values in `model_count` and the count of the remainder is the
    // quotient by 2^k.
    let remainder = if summed_out.is_empty() {
        None
    } else {
        let vars: Vec<VarId> = summed_out.iter().map(|&v| VarId(v - 1)).collect();
        let g = exists_vars(&f, &vars, QuantificationStrategy::Automatic);
        let free = BigUint::from(2u32).pow(vars.len() as u32);
        let remainder = g.model_count() / &free;
        let names: Vec<String> = summed_out.iter().map(|v| format!("x{v}")).collect();
        println!("after summing out {}: {remainder}", names.join(", "));
        Some(remainder)
    };

    // One weighted count. Every variable is given weight 1/3 when false and
    // 2/3 when true, so the value is the probability of the formula under
    // independent variables.
    let third = BigRational::new(1.into(), 3.into());
    let two_thirds = BigRational::new(2.into(), 3.into());
    let weights: Vec<(BigRational, BigRational)> = (0..cnf.num_vars)
        .map(|_| (third.clone(), two_thirds.clone()))
        .collect();
    let weighted = evaluate(&f, &RationalWeights::from_weights(&weights));
    println!("weighted count (w(x)=2/3, w(!x)=1/3): {weighted}");

    if check {
        let expected = brute_force_count(&cnf);
        if count != BigUint::from(expected) {
            return Err(format!(
                "brute force says {expected}, the diagram says {count}"
            ));
        }
        println!("--check: brute force agrees on {expected} models");
        if let Some(remainder) = remainder {
            let expected = brute_force_projected_count(&cnf, &summed_out);
            if remainder != BigUint::from(expected) {
                return Err(format!(
                    "brute force says {expected} after summing out, the diagram says {remainder}"
                ));
            }
            println!("--check: brute force agrees on {expected} after summing out");
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
