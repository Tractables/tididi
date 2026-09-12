//! The `&`, `|` and `!` impls for [`Tdd`].
//!
//! Each is a one-line forward to the operation beside it and holds no logic of
//! its own. All three run on a transient engine with nothing armed and panic
//! where the engine form would return an error. To run `&` or `|` under a
//! limit, call [`Engine::and`](crate::Engine::and) or
//! [`Engine::or`](crate::Engine::or), which report a cut instead of aborting;
//! `!` forwards to [`negate()`], which has no engine form.
//!
//! Entry points: the [`std::ops::BitAnd`], [`std::ops::BitOr`] and
//! [`std::ops::Not`] impls on [`Tdd`]. All are by-value for symmetry: `&` and
//! `|` consume both operands, `!` consumes its operand.

use std::ops::{BitAnd, BitOr, Not};

use crate::apply::apply_and;
use crate::apply::apply_or;
use crate::apply::negate;
use crate::diagram::Tdd;

/// `f & g` — conjunction, as [`Engine::and`](crate::Engine::and): consumes
/// both operands, which must share a vtree, and panics where that method
/// would return an error.
impl BitAnd for Tdd {
    type Output = Tdd;
    fn bitand(self, rhs: Tdd) -> Tdd {
        apply_and(self, rhs)
    }
}

/// `f | g` — disjunction, as [`Engine::or`](crate::Engine::or): consumes both
/// operands, which must share a vtree and have no marginal level, and panics
/// where that method would return an error.
impl BitOr for Tdd {
    type Output = Tdd;
    fn bitor(self, rhs: Tdd) -> Tdd {
        apply_or(self, rhs)
    }
}

/// `!f` — negation. Delegates to [`negate()`]; consumes its operand.
impl Not for Tdd {
    type Output = Tdd;
    fn not(self) -> Tdd {
        negate(self)
    }
}
