//! Inspectors and the budget-tracked clone of a count vector.

use super::*;

impl<R: ReservePolicy> CountVec<R> {
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
    pub(crate) fn try_clone(&self, eng: &Engine) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(eng, &mut fast, self.fast.len())?;
        fast.extend_from_slice(&self.fast);
        let big = match &self.big {
            Some(b) => Some(b.try_clone::<R>(eng)?),
            None => None,
        };
        Ok(CountVec {
            fast,
            big,
            all_u64: self.all_u64,
            _res: PhantomData,
        })
    }
}
