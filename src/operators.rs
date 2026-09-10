//! The `&`, `|` and `!` impls for [`Tdd`].
//!
//! Each is a one-line forward to [`crate::apply`] and holds no logic of its own.
//! To run one under a limit, call [`Engine::and`](crate::Engine::and),
//! [`Engine::or`](crate::Engine::or) or [`crate::negate`], which report a cut
//! instead of aborting.
//!
//! Entry points: the [`std::ops::BitAnd`], [`std::ops::BitOr`] and
//! [`std::ops::Not`] impls on [`Tdd`]. All are by-value for symmetry: `&` and
//! `|` consume both operands, `!` consumes its operand.

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
