use super::*;

mod home;
mod home_root;
mod projection;
mod region;
mod structure;
mod text;

impl crate::vtree::Home {
    /// `vtree` as a `Home` whatever it is, and on a plain root admitted by
    /// [`Tdd::builder`](crate::Tdd::builder) on this thread: the door for
    /// tests of the padded form, each tagged with the change that deletes
    /// the form and the test.
    pub(crate) fn unchecked(vtree: &std::sync::Arc<crate::vtree::Vtree>) -> crate::vtree::Home {
        if !vtree.on_home() {
            ADMITTED.with(|admitted| admitted.borrow_mut().push(std::sync::Arc::as_ptr(vtree)));
        }
        crate::vtree::Home(std::sync::Arc::clone(vtree))
    }
}

std::thread_local! {
    // Addresses, not `Weak`s: a weak count makes `Arc::get_mut` fail on the
    // root. A root dropped and another allocated at its address on the same
    // thread is admitted too, so a test of the refusal admits nothing.
    static ADMITTED: std::cell::RefCell<Vec<*const crate::vtree::Vtree>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether [`Home::unchecked`](crate::vtree::Home::unchecked) admitted
/// `vtree` on this thread.
pub(crate) fn admitted(vtree: &crate::vtree::Vtree) -> bool {
    ADMITTED.with(|admitted| admitted.borrow().contains(&std::ptr::from_ref(vtree)))
}

impl crate::vtree::Vtree {
    /// This root vtree, which is not a home, with `weights` for its
    /// variables: every diagram on it is weighted as on a
    /// [`Home::weighted`] and carries no weights of its own.
    ///
    /// # Errors
    ///
    /// [`TddBuildError::MissingVariableWeight`] when the table has no weight
    /// for a leaf's variable; [`TddBuildError::IncompatibleWeights`] on a
    /// home's root or a projection, and on a root that already has other
    /// weights.
    pub(crate) fn with_weights(mut self, weights: crate::diagram::Weights) -> Result<Vtree, crate::diagram::TddBuildError> {
        if self.home.is_some() || self.is_home() || self.weights().is_some_and(|own| *own != weights) {
            return Err(crate::diagram::TddBuildError::IncompatibleWeights);
        }
        weights.check_variables(self.leaf_bottomup().map(|(_, var)| var))?;
        self.index = super::home::RootIndex::new(None, Some(weights));
        Ok(self)
    }
}
