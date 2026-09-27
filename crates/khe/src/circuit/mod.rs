//! Circuit models and their lowering to encoding schemes.
//!
//! - `poly_circuit`: [`PolyCircuit`], polynomial circuits with gates ([`PolyGate`],
//!   [`PolyGateKind`]), sub-circuit calls, and serialization.
//! - `boolean`: Boolean circuit shapes and data with constant, copy, not, and, and xor gates.
//! - `boolean_dsl`: DSL-level Boolean circuit families and their validity and satisfaction
//!   predicates.
//! - `public_lut`: public lookup programs.
//! - `lowering`: the traits a concrete encoding scheme implements (arithmetic, slot operations,
//!   public lookups, and structured lowering) and `lower_circuit`, which drives them gate by gate
//!   with a [`GateInstance`] (call path, local gate, and operation occurrence).

pub mod boolean;
pub mod boolean_dsl;
pub mod gate;
pub mod lowering;
pub mod poly_circuit;
pub mod public_lut;
pub mod serde;

pub use boolean::{
    BooleanCircuitAnalysis, BooleanCircuitData, BooleanCircuitError, BooleanCircuitShape,
    BooleanGateData, BooleanGateKind, to_poly_circuit,
};
pub use boolean_dsl::{
    BOOLEAN_INSTANCE_INPUT, BOOLEAN_WITNESS_INPUT, BooleanCircuitFamilyInputs,
    BooleanCircuitFamilyParams, BooleanLayerGate, GateSlot, boolean_circuit_satisfaction_predicate,
    boolean_circuit_validity_predicate, evaluate_boolean_family, evaluate_boolean_matrix_family,
    select_boolean_matrix_output, select_boolean_output,
};
pub use gate::{
    GateParamSource, PolyGate, PolyGateKind, PolyGateType, SlotTransferSpec, SubCircuitParamKind,
    SubCircuitParamSpec, SubCircuitParamValue,
};
pub use lowering::{
    ArithmeticCircuitLowering, CircuitLowerError, CircuitLoweringTypes, GateInstance,
    GraphCircuitLowering, PublicLookupLowering, SlotOperationLowering, StructuredCircuitLowering,
    lower_circuit, lower_circuit_structured,
};
pub use poly_circuit::*;
pub use public_lut::{LutExpr, LutInterval, PublicLutError, PublicLutProgram};
