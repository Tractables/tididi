//! Operator sugar for `Tdd` — thin delegations to [`crate::apply`].
//!
//! `&` and `|` are the public spelling of conjunction and disjunction over
//! owned diagrams, and [`crate::negate`] is the named form of `!`. To run
//! either under a limit, use `Engine::and` or `Engine::or`, which return an
//! error instead of aborting.
//!
//! All impls are by-value for symmetry: `&`/`|` consume both operands, `!`
//! consumes its operand. Each is a one-line forward to the existing operation.

use std::ops::{BitAnd, BitOr, Not};

use crate::apply::apply_and;
use crate::apply::apply_or;
use crate::apply::negate;
use crate::diagram::Tdd;

/// `f & g` — conjunction. Consumes both operands.
impl BitAnd for Tdd {
    type Output = Tdd;
    fn bitand(self, rhs: Tdd) -> Tdd {
        apply_and(self, rhs)
    }
}

/// `f | g` — disjunction. Consumes both operands.
impl BitOr for Tdd {
    type Output = Tdd;
    fn bitor(self, rhs: Tdd) -> Tdd {
        apply_or(self, rhs)
    }
}

/// `!f` — negation. Delegates to [`negate`]; consumes its operand.
impl Not for Tdd {
    type Output = Tdd;
    fn not(self) -> Tdd {
        negate(self)
    }
}
