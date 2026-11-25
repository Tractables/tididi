use super::*;

mod home;
mod home_root;
mod projection;
mod region;
mod structure;
mod text;

impl crate::vtree::Home {
    /// `vtree` as a `Home` whatever it is, so a test still on a plain root
    /// reaches the constructors.
    pub(crate) fn unchecked(vtree: &std::sync::Arc<crate::vtree::Vtree>) -> crate::vtree::Home {
        crate::vtree::Home(std::sync::Arc::clone(vtree))
    }
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
