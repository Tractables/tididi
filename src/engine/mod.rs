//! The session object every operation runs on: the limits it is held to and the
//! scratch it reuses.
//!
//! An [`Engine`] is what a caller keeps between operations. Holding the scratch
//! makes the reuse explicit — two engines never share a buffer, and dropping one
//! frees everything it warmed up — and holding the limits makes the arming
//! explicit: what a conjunction is allowed to spend is a field a caller sets,
//! not ambient state it inherits.
//!
//! The free functions and operator sugar elsewhere in this crate are thin
//! wrappers that build a transient engine, run the operation on it, and panic on
//! failure. There is one implementation underneath.

mod limits;
mod memory;
mod meters;
mod poll;
mod stop;

pub use limits::{LimitSet, Limits};
pub use memory::MemPressure;
pub use meters::{ApplyMeters, MergePosition};
pub use stop::{Scheduled, Stop, StopAt};

pub(crate) use limits::{PollGate, PAIR_ELEM_BYTES};

#[cfg(test)]
pub(crate) use limits::DENSE_GROWTH_DECISION_THRESHOLD;

#[cfg(test)]
pub(crate) use memory::{vas_headroom_with_margin, SOFT_HEADROOM_MARGIN_BYTES};

#[cfg(test)]
#[path = "headroom_tests.rs"]
mod headroom_tests;

#[cfg(test)]
#[path = "limits_tests.rs"]
mod limits_tests;

/// The limits and scratch one caller's operations run on.
///
/// Build one per compile and thread it through: every conjunction, reduction,
/// marginalization and restructuring takes `&mut Engine`, reuses the buffers it
/// holds, and is cut by the limits armed on it.
#[derive(Default)]
pub struct Engine {
    limits: Limits,
}

impl Engine {
    /// A fresh engine: nothing armed, no scratch warmed up.
    #[must_use]
    pub fn new() -> Engine {
        Engine { limits: Limits::new() }
    }

    /// A fresh engine with `set` armed.
    #[must_use]
    pub fn with_limits(set: LimitSet) -> Engine {
        let engine = Engine::new();
        engine.limits.install(set);
        engine
    }

    /// The limits armed on this engine.
    #[must_use]
    pub fn limit_set(&self) -> LimitSet {
        self.limits.armed()
    }

    /// Arm `set`, returning what was armed before — which is what a caller
    /// restores when its scope ends.
    pub fn set_limits(&mut self, set: LimitSet) -> LimitSet {
        self.limits.install(set)
    }

    /// The limits themselves, for reading the meters and for the operations
    /// that charge against them.
    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Release everything this engine retains — every scratch allocation and
    /// every meter — leaving the armed limits alone.
    ///
    /// Called between a failed operation and whatever a caller does to recover
    /// from it, so the recovery starts on a clean allocator slate rather than
    /// inheriting the peak the failure left behind. Also the way back to a known
    /// state after an operation was cut by an unwind, which can leave a scoped
    /// arming installed.
    pub fn reset(&mut self) {
        let armed = self.limits.armed();
        *self = Engine::new();
        self.limits.install(armed);
    }
}
