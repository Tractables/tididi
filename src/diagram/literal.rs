//! Literals: a variable with a polarity.

use crate::vtree::VarId;

/// A literal: a variable with a polarity.
///
/// Lives lib-side (alongside `VarId`) so the pure-TDD layer (`Tdd::clause`,
/// `apply_and_clause`) can accept `&[Literal]` slices without depending on the
/// CNF module. The CNF `Clause`/`CnfFormula` types build on it and re-export it.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Literal {
    /// The variable this literal refers to.
    pub var: VarId,
    /// `true` for a positive literal, `false` for a negated one.
    pub positive: bool,
}

impl Literal {
    /// Construct a literal over `var` with the given polarity.
    pub fn new(var: VarId, positive: bool) -> Self {
        Literal { var, positive }
    }

    /// The positive literal over `var`.
    pub fn pos(var: VarId) -> Self {
        Literal::new(var, true)
    }

    /// The negated literal over `var`.
    pub fn neg(var: VarId) -> Self {
        Literal::new(var, false)
    }

    /// This literal with its polarity flipped.
    #[must_use]
    pub fn negated(self) -> Self {
        Literal {
            var: self.var,
            positive: !self.positive,
        }
    }
}

/// Build a `Literal` from a signed **DIMACS** integer.
///
/// DIMACS variables are 1-based: `1` is the first variable (`VarId(0)`), `2` the
/// second, and so on; a negative value denotes a negated literal. The magnitude
/// is decremented to the 0-based [`VarId`] used internally — the same convention
/// as the CNF parser (`VarId(val.unsigned_abs() - 1)`).
///
/// # Panics
/// Panics on `0`, which is not a valid DIMACS literal (in the DIMACS format `0`
/// terminates a clause rather than naming a variable).
///
/// ```
/// use tididi::diagram::Literal;
/// use tididi::vtree::VarId;
/// assert_eq!(Literal::from(1), Literal::pos(VarId(0)));
/// assert_eq!(Literal::from(-2), Literal::neg(VarId(1)));
/// ```
impl From<i32> for Literal {
    fn from(n: i32) -> Self {
        assert!(
            n != 0,
            "0 is not a DIMACS literal (it terminates a clause, not a variable)"
        );
        let var = VarId(n.unsigned_abs() - 1);
        if n > 0 {
            Literal::pos(var)
        } else {
            Literal::neg(var)
        }
    }
}
