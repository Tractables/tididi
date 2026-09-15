//! The `&`, `|` and `!` impls for [`Tdd`].
//!
//! Each is a one-line forward to the operation beside it and holds no logic of
//! its own. All three reuse the vtree's execution context and panic
//! where the checked operation would return an error. To run `&` or `|` under a
//! limit, call [`Engine::and`](crate::Engine::and) or
//! [`Engine::or`](crate::Engine::or), which report a cut instead of aborting;
//! `!` uses [`Tdd::negate`].
//!
//! Entry points: the [`std::ops::BitAnd`], [`std::ops::BitOr`] and
//! [`std::ops::Not`] impls on [`Tdd`]. All are by-value for symmetry: `&` and
//! `|` consume both operands, `!` consumes its operand.

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
