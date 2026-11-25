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
