//! Concrete numeric foundations for executable-IR noise simulation.
//!
//! Graph planning, relation discovery, and evaluation are deliberately kept
//! out of this initial crate surface.  The pure state and bound types here are
//! designed to be consumed by those later layers.

pub mod bound;
pub mod error;
pub mod eval;
pub mod family;
pub mod identity;
pub mod plan;
pub mod relation;
pub mod report;
pub mod request;
pub mod state;

pub use bound::{
    BoundError, ProductGeometry, centered_residue_bound, convolution_factor, left_action_gain,
    product_bound, right_action_gain,
};
pub use error::{DiagnosticSite, SimulationError};
pub use identity::{FamilyViewId, GadgetDescriptor, SelectorId, SourceId, ValueId};
pub use report::{
    DroppedCarrierDiagnostic, RootNoiseReport, SimulationDiagnostics, SimulationReport,
};
pub use request::{
    ExternalInputFact, ExternalInputValue, SimulationLimits, SimulationProgram, SimulationRequest,
    SimulationRoot, SimulationStage, StageId,
};
pub use state::{
    AbstractValue, BooleanState, FamilyState, IntegerState, MatrixState, RightCarrier, StateError,
    TrapdoorState, exact_matrix, gadget_decomposition, gadget_matrix, gaussian_sample,
    plain_hash_sample, preimage_sample, trapdoor_public_matrix, uniform_interval_sample,
    uniform_residue_sample, zero_matrix,
};

/// Evaluate every reached graph output under one concrete parameter environment.
pub fn simulate(request: &SimulationRequest) -> Result<SimulationReport, SimulationError> {
    eval::run(request)
}
