//! Inspectors and the budget-tracked clone of a count vector.

use super::*;

impl CountVec {
    /// Test-only inspector (production reads the certificate through the
    /// borrowed view, [`CountRef::all_u64`]).
    pub(crate) fn all_u64(&self) -> bool {
        self.all_u64
    }

    /// Test-only inspector (production readers go through `get`/`big_val`).
    pub(crate) fn has_big(&self) -> bool {
        self.big.is_some()
    }

    /// Fallible clone: reserves both backing arrays exactly before copying,
    /// so an over-budget duplicate raises the policy's error instead of an
    /// infallible allocator abort. Test-only: it is the round-trip coverage
    /// of the fast/big split.
    pub(crate) fn try_clone(&self, eng: &Engine) -> Result<Self, crate::limits::OperationError> {
        let mut fast: Vec<u128> = Vec::new();
        eng.limits().reserve_exact(&mut fast, self.fast.len())?;
        fast.extend_from_slice(&self.fast);
        let big = match &self.big {
            Some(b) => Some(b.try_clone(eng)?),
            None => None,
        };
        Ok(CountVec {
            fast,
            big,
            all_u64: self.all_u64,
        })
    }
}

impl CountVec {
    /// Infallible convenience wrapper (test allocations are expected to succeed).
    pub(crate) fn with_width(eng: &Engine, width: usize) -> Self {
        Self::try_with_width(eng, width).expect("test allocation succeeds")
    }

    pub(crate) fn set_i(&mut self, eng: &Engine, i: usize, c: Count) {
        self.set(eng, i, c).expect("test allocation succeeds")
    }
}
