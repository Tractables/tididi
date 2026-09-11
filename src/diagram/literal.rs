//! Literals: a variable with a polarity.

use crate::vtree::VarId;

/// A literal: a variable with a polarity.
///
/// `Tdd::clause` and `apply_and_clause` accept `&[Literal]` slices; a signed
/// DIMACS integer converts into one through [`From`].
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
/// is decremented to the 0-based [`VarId`] used internally.
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

/// Whether `clause` names one variable in both polarities, which makes the
/// disjunction true under every assignment.
///
/// A clause is a set of literals: a variable repeated in one polarity says
/// nothing the first occurrence did not, and a variable in both polarities
/// satisfies the clause whatever that variable is. Both clause builders index
/// one column per variable of the clause and so read a repeat as a single
/// literal, which is the right answer for the first case and the wrong one for
/// the second; each calls this first and answers ⊤ instead.
///
/// A short clause is scanned pairwise and a long one through a set, so the
/// check stays below the vtree walk that follows it either way.
pub(crate) fn is_tautological(clause: &[Literal]) -> bool {
    /// Above this many literals the pairwise scan is no longer the cheaper one.
    const PAIRWISE_MAX: usize = 32;
    if clause.len() <= PAIRWISE_MAX {
        return clause.iter().enumerate().any(|(i, l)| {
            clause[..i].iter().any(|e| e.var == l.var && e.positive != l.positive)
        });
    }
    let mut seen: rustc_hash::FxHashMap<VarId, bool> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(clause.len(), Default::default());
    clause
        .iter()
        .any(|l| matches!(seen.insert(l.var, l.positive), Some(p) if p != l.positive))
}
