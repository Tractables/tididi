//! Literals: a variable with a polarity.

use crate::vtree::VarId;

/// A Boolean variable and its polarity.
///
/// Integer inputs use signed literals: `1` is positive `VarId(1)`,
/// and `-2` is negative `VarId(2)`. Zero is invalid and conversion returns an error.
/// Use the typed constructors when your application already holds variable ids:
///
/// ```
/// use tididi::{Literal, OperationError};
/// use tididi::vtree::VarId;
///
/// assert_eq!(Literal::try_from(1)?, Literal::pos(VarId(1)));
/// assert_eq!(Literal::try_from(-2)?, Literal::neg(VarId(2)));
/// assert!(Literal::pos(VarId(1)).sign);
/// assert!(!Literal::neg(VarId(1)).sign);
/// assert_eq!(Literal::pos(VarId(1)).negated(), Literal::neg(VarId(1)));
/// assert_eq!(Literal::try_from(0), Err(OperationError::InvalidLiteral(0)));
/// # Ok::<(), OperationError>(())
/// ```
///
/// Constructors such as [`Tdd::clause`](crate::Tdd::clause) accept iterators
/// of integers or typed literals, by value or reference.
/// [`Tdd::and_clause`](crate::Tdd::and_clause) and
/// [`ModelCounter::observe`](crate::query::ModelCounter::observe) accept arrays,
/// slices and vectors of either type.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Literal {
    /// The variable this literal refers to.
    pub var: VarId,
    /// `true` for a positive literal, `false` for a negated one.
    pub sign: bool,
}

impl Literal {
    /// Construct a literal over `var` with the given sign.
    pub fn new(var: VarId, sign: bool) -> Self {
        Literal { var, sign }
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
            sign: !self.sign,
        }
    }
}

/// Build a `Literal` from a signed **DIMACS** integer.
///
/// The magnitude is the variable number (`1` is `VarId(1)`) and a negative
/// value denotes a negated literal.
///
/// # Errors
/// Returns [`OperationError::InvalidLiteral`](crate::OperationError::InvalidLiteral)
/// for zero, which does not name a variable.
/// Every nonzero `i32` is accepted; the operation using the literal checks
/// whether its variable belongs to the vtree.
impl TryFrom<i32> for Literal {
    type Error = crate::OperationError;

    fn try_from(n: i32) -> Result<Self, Self::Error> {
        if n == 0 { return Err(crate::OperationError::InvalidLiteral(n)); }
        let var = VarId(n.unsigned_abs());
        Ok(Literal::new(var, n > 0))
    }
}

/// Convert a borrowed signed integer with the same checks as [`Literal::try_from`].
///
/// A slice iterates as references, so this is what lets a `&[i32]` or a
/// `&Vec<i32>` of DIMACS literals go straight into [`crate::Tdd::clause`].
///
/// # Errors
/// Returns [`OperationError::InvalidLiteral`](crate::OperationError::InvalidLiteral) for zero.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let dimacs: Vec<i32> = vec![1, -2];
/// let from_slice = Tdd::clause(&vtree, &dimacs)?;
/// assert_eq!(from_slice.model_count()?, Tdd::clause(&vtree, [1, -2])?.model_count()?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl TryFrom<&i32> for Literal {
    type Error = crate::OperationError;

    fn try_from(n: &i32) -> Result<Self, Self::Error> {
        Literal::try_from(*n)
    }
}

/// Copy a borrowed literal, so [`crate::Tdd::clause`] accepts a `&[Literal]`.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Literal, Tdd};
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let lits: Vec<Literal> = vec![Literal::pos(VarId(1)), Literal::neg(VarId(2))];
/// let from_slice = Tdd::clause(&vtree, &lits)?;
/// assert_eq!(from_slice.model_count()?, Tdd::clause(&vtree, [1, -2])?.model_count()?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl From<&Literal> for Literal {
    fn from(l: &Literal) -> Self {
        *l
    }
}

/// A signed integer or typed literal accepted by clause conjunction and observations.
///
/// Implemented for [`Literal`] and signed, one-based `i32` literals. Pass an
/// array, slice or vector to [`Tdd::and_clause`](crate::Tdd::and_clause) or
/// [`ModelCounter::observe`](crate::query::ModelCounter::observe).
/// This trait is sealed.
pub trait LiteralInput: input::Sealed {}

impl LiteralInput for Literal {}
impl LiteralInput for i32 {}

mod input {
    use super::Literal;
    use crate::vtree::Vtree;
    use crate::{Engine, OperationError, Tdd};
    use crate::apply::conjoin_clause::{conjoin_clause_owned, disjoin_cube_owned};

    pub trait Sealed: Copy {
        fn literal(self) -> Result<Literal, OperationError>;
        /// Convert the clause when necessary, then invoke the typed conjunction.
        fn conjoin(eng: &Engine, f: Tdd, clause: &[Self]) -> Result<Tdd, OperationError>;
        /// Convert the cube when necessary, then invoke the typed disjunction.
        fn disjoin(eng: &Engine, f: Tdd, cube: &[Self]) -> Result<Tdd, OperationError>;
        /// Append these inputs to `out` as typed literals, checking each
        /// variable against `vtree`.
        fn collect(eng: &Engine, vtree: &Vtree, input: &[Self], out: &mut Vec<Literal>) -> Result<(), OperationError>;
    }

    impl Sealed for Literal {
        fn literal(self) -> Result<Literal, OperationError> { Ok(self) }

        #[inline]
        fn conjoin(eng: &Engine, f: Tdd, clause: &[Self]) -> Result<Tdd, OperationError> {
            conjoin_clause_owned(eng, f, clause)
        }

        #[inline]
        fn disjoin(eng: &Engine, f: Tdd, cube: &[Self]) -> Result<Tdd, OperationError> {
            disjoin_cube_owned(eng, f, cube)
        }

        fn collect(eng: &Engine, vtree: &Vtree, input: &[Self], out: &mut Vec<Literal>) -> Result<(), OperationError> {
            let lim = eng.limits();
            let mut gate = lim.gate();
            for &literal in input {
                gate.poll(1)?;
                if vtree.leaf_of(literal.var).is_none() {
                    return Err(OperationError::VariableNotInVtree(literal.var));
                }
                lim.try_push(out, literal)?;
            }
            gate.flush()
        }
    }

    impl Sealed for i32 {
        fn literal(self) -> Result<Literal, OperationError> { Literal::try_from(self) }

        fn conjoin(eng: &Engine, f: Tdd, clause: &[Self]) -> Result<Tdd, OperationError> {
            let lim = eng.limits();
            let _op = lim.begin_operation();
            lim.check_stop()?;
            let literals = typed(eng, &f, clause)?;
            conjoin_clause_owned(eng, f, &literals)
        }

        fn disjoin(eng: &Engine, f: Tdd, cube: &[Self]) -> Result<Tdd, OperationError> {
            let lim = eng.limits();
            let _op = lim.begin_operation();
            lim.check_stop()?;
            let literals = typed(eng, &f, cube)?;
            disjoin_cube_owned(eng, f, &literals)
        }

        fn collect(eng: &Engine, vtree: &Vtree, input: &[Self], out: &mut Vec<Literal>) -> Result<(), OperationError> {
            let lim = eng.limits();
            let mut gate = lim.gate();
            for &value in input {
                gate.poll(1)?;
                let literal = Literal::try_from(value)?;
                if vtree.leaf_of(literal.var).is_none() {
                    return Err(OperationError::VariableNotInVtree(literal.var));
                }
                lim.try_push(out, literal)?;
            }
            gate.flush()
        }
    }

    /// Convert a slice of signed integers, checking each variable against the
    /// operand's vtree. Runs inside the caller's operation scope, so the
    /// conversion is charged to the operation it prepares.
    fn typed(eng: &Engine, f: &Tdd, input: &[i32]) -> Result<Vec<Literal>, OperationError> {
        let mut literals = Vec::new();
        <i32 as Sealed>::collect(eng, f.vtree(), input, &mut literals)?;
        Ok(literals)
    }
}

/// Whether `clause` names one variable in both polarities, which makes the
/// disjunction true under every assignment.
///
/// The clause builders index one column per variable and would read such a
/// clause as a single literal, so each calls this first and answers ⊤. A
/// short clause is scanned pairwise, a long one through a set.
pub(crate) fn is_tautological(lim: &crate::limits::Limits, clause: &[Literal]) -> Result<bool, crate::limits::OperationError> {
    let mut gate = lim.gate();
    /// Above this many literals the pairwise scan is no longer the cheaper one.
    const PAIRWISE_MAX: usize = 32;
    let mut seen = rustc_hash::FxHashMap::default();
    for (i, lit) in clause.iter().enumerate() {
        gate.poll(1)?;
        let conflict = if clause.len() <= PAIRWISE_MAX {
            clause[..i].iter().any(|e| e.var == lit.var && e.sign != lit.sign)
        } else {
            if !seen.contains_key(&lit.var) { lim.reserve_map(&mut seen, 1)?; }
            matches!(seen.insert(lit.var, lit.sign), Some(p) if p != lit.sign)
        };
        if conflict { gate.flush()?; return Ok(true); }
    }
    gate.flush()?;
    Ok(false)
}
