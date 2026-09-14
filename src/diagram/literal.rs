//! Literals: a variable with a polarity.

use crate::vtree::VarId;

/// A Boolean variable and its polarity.
///
/// Integer inputs use signed, one-based literals: `1` is positive `VarId(0)`,
/// and `-2` is negative `VarId(1)`. Zero is invalid and its conversion panics.
/// Use the typed constructors when your application already has zero-based ids:
///
/// ```
/// use tididi::Literal;
/// use tididi::vtree::VarId;
///
/// assert_eq!(Literal::from(1), Literal::pos(VarId(0)));
/// assert_eq!(Literal::from(-2), Literal::neg(VarId(1)));
/// assert_eq!(Literal::pos(VarId(0)).negated(), Literal::neg(VarId(0)));
/// ```
///
/// Constructors such as [`Engine::clause`](crate::Engine::clause) accept iterators
/// of integers or typed literals, by value or reference. Methods taking a
/// `&[Literal]`, such as [`Engine::and_clause`](crate::Engine::and_clause), require
/// a typed slice; convert integer data before that call.
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

/// Build a `Literal` from a borrowed DIMACS integer, as `From<i32>` does from
/// an owned one.
///
/// A slice iterates as references, so this is what lets a `&[i32]` or a
/// `&Vec<i32>` of DIMACS literals go straight into a builder that takes
/// `impl IntoIterator<Item = impl Into<Literal>>`.
///
/// # Panics
/// Panics on `0`, as `From<i32>` does.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let dimacs: Vec<i32> = vec![1, -2];
/// let from_slice = Tdd::clause(&vtree, &dimacs);
/// assert_eq!(from_slice.model_count(), Tdd::clause(&vtree, [1, -2]).model_count());
/// ```
impl From<&i32> for Literal {
    fn from(n: &i32) -> Self {
        Literal::from(*n)
    }
}

/// Copy a borrowed literal, so a `&[Literal]` feeds a builder that takes
/// `impl IntoIterator<Item = impl Into<Literal>>` without a collect.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Literal, Tdd};
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let lits: Vec<Literal> = vec![Literal::from(1), Literal::from(-2)];
/// let from_slice = Tdd::clause(&vtree, &lits);
/// assert_eq!(from_slice.model_count(), Tdd::clause(&vtree, [1, -2]).model_count());
/// ```
impl From<&Literal> for Literal {
    fn from(l: &Literal) -> Self {
        *l
    }
}

/// Whether `clause` names one variable in both polarities, which makes the
/// disjunction true under every assignment.
///
/// The clause builders index one column per variable and would read such a
/// clause as a single literal, so each calls this first and answers ⊤. A
/// short clause is scanned pairwise, a long one through a set.
pub(crate) fn is_tautological(lim: &crate::limits::Limits, clause: &[Literal]) -> Result<bool, crate::limits::OperationError> {
    let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
    /// Above this many literals the pairwise scan is no longer the cheaper one.
    const PAIRWISE_MAX: usize = 32;
    let mut seen = rustc_hash::FxHashMap::default();
    for (i, lit) in clause.iter().enumerate() {
        lim.poll(&mut gate, 1)?;
        let conflict = if clause.len() <= PAIRWISE_MAX {
            clause[..i].iter().any(|e| e.var == lit.var && e.positive != lit.positive)
        } else {
            if !seen.contains_key(&lit.var) { lim.reserve_map(&mut seen, 1)?; }
            matches!(seen.insert(lit.var, lit.positive), Some(p) if p != lit.positive)
        };
        if conflict { lim.flush_poll(&mut gate)?; return Ok(true); }
    }
    lim.flush_poll(&mut gate)?;
    Ok(false)
}
