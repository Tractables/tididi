//! Operator sugar for `Tdd` — thin delegations to `transform::{pairwise, unary}`;
//! the named functions (`apply_and`, `apply_or`, `negate`) remain the primary API.
//!
//! All impls are by-value for symmetry: `&`/`|` consume both operands, `!`
//! consumes its operand. Each is a one-line forward to the existing operation.

use std::ops::{BitAnd, BitOr, Not};

use crate::tdd::transform::pairwise::conjoin::apply_and;
use crate::tdd::transform::pairwise::disjoin::apply_or;
use crate::tdd::transform::unary::negate::negate;
use crate::tdd::types::Tdd;

/// `f & g` — conjunction. Delegates to [`apply_and`]; consumes both operands.
impl BitAnd for Tdd {
    type Output = Tdd;
    fn bitand(self, rhs: Tdd) -> Tdd {
        apply_and(self, rhs)
    }
}

/// `f | g` — disjunction. Delegates to [`apply_or`]; consumes both operands.
impl BitOr for Tdd {
    type Output = Tdd;
    fn bitor(self, rhs: Tdd) -> Tdd {
        apply_or(&self, &rhs)
    }
}

/// `!f` — negation. Delegates to [`negate`]; consumes its operand.
impl Not for Tdd {
    type Output = Tdd;
    fn not(self) -> Tdd {
        negate(&self)
    }
}
