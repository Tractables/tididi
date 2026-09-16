//! Consuming Boolean operators for [`Tdd`].
//!
//! Operators use the vtree's execution context and panic on operation errors.
//! Use [`crate::and`], [`crate::or`] or [`Tdd::negate`] to handle those errors.

use std::ops::{BitAnd, BitOr, Not};

use crate::diagram::Tdd;
use crate::apply::{apply_and, apply_or};

/// `f & g` — conjunction, as [`and`](crate::and): consumes
/// both operands, which must share a vtree, and panics where that function
/// would return an error.
impl BitAnd for Tdd {
    type Output = Tdd;
    fn bitand(self, rhs: Tdd) -> Tdd {
        apply_and(self, rhs)
    }
}

/// `f | g` — disjunction, as [`or`](crate::or): consumes both
/// operands, which must share a vtree and have no marginal level, and panics
/// where that function would return an error.
impl BitOr for Tdd {
    type Output = Tdd;
    fn bitor(self, rhs: Tdd) -> Tdd {
        apply_or(self, rhs)
    }
}

/// `!f` — negation. Delegates to [`Tdd::negate`]; consumes its operand.
impl Not for Tdd {
    type Output = Tdd;
    fn not(self) -> Tdd {
        self.negate().expect("negation failed; use Tdd::negate to handle errors")
    }
}
