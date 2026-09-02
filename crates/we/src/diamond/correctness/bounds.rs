//! Diamond-specific coefficient bounds.
//!
//! This module intentionally is not a generic IR noise analyser.  It contains
//! only the recurrence used by the Diamond correctness claim.  In particular,
//! a negacyclic product has one contribution per output coefficient, hence its
//! factor is `n`; a matrix product has the additional inner-dimension factor.

use super::super::parameter_search::DiamondSelectedParameters;
use crate::diamond::graph::NOISY_PLAINTEXT_OUTPUT;
use mxx_ir_core::{ConcreteLinkedProgram, ValidatedLinkedProgram};
use num_bigint::BigUint;
use num_traits::{CheckedSub, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Version of the Diamond bound syntax.  It is part of cache identities.
pub const DIAMOND_BOUND_SCHEMA_VERSION: u32 = 1;

/// Concrete values consumed by [`BoundExpr::evaluate`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoundEnvironment {
    pub modulus: BigUint,
    pub ring_dimension: BigUint,
    pub state_rows: BigUint,
    pub state_columns: BigUint,
    pub gadget_columns: BigUint,
    pub error_coefficient_bound: BigUint,
    pub preimage_coefficient_bound: BigUint,
    pub gadget_decomposition_bound: BigUint,
    pub input_steps: BigUint,
    pub circuit_layers: BigUint,
}

/// Leaves available to the Diamond recurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum BoundParameter {
    Modulus,
    RingDimension,
    StateRows,
    StateColumns,
    GadgetColumns,
    ErrorCoefficientBound,
    PreimageCoefficientBound,
    GadgetDecompositionBound,
    InputSteps,
    CircuitLayers,
}

impl BoundParameter {
    fn value(self, environment: &BoundEnvironment) -> BigUint {
        match self {
            Self::Modulus => environment.modulus.clone(),
            Self::RingDimension => environment.ring_dimension.clone(),
            Self::StateRows => environment.state_rows.clone(),
            Self::StateColumns => environment.state_columns.clone(),
            Self::GadgetColumns => environment.gadget_columns.clone(),
            Self::ErrorCoefficientBound => environment.error_coefficient_bound.clone(),
            Self::PreimageCoefficientBound => environment.preimage_coefficient_bound.clone(),
            Self::GadgetDecompositionBound => environment.gadget_decomposition_bound.clone(),
            Self::InputSteps => environment.input_steps.clone(),
            Self::CircuitLayers => environment.circuit_layers.clone(),
        }
    }
}

/// The small, explicit AST used by the Diamond noise proof.
///
/// `NegacyclicProduct` evaluates to `n * left * right`, and
/// `MatrixProduct` evaluates to `d * n * left * right`.  There is no
/// operation which expands both canonical coordinates and pays `n²`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BoundExpr {
    Literal(BigUint),
    Parameter(BoundParameter),
    Add(Vec<Self>),
    Multiply(Vec<Self>),
    Maximum(Vec<Self>),
    NegacyclicProduct { left: Box<Self>, right: Box<Self> },
    MatrixProduct { inner_dimension: Box<Self>, left: Box<Self>, right: Box<Self> },
}

impl BoundExpr {
    pub fn literal(value: impl Into<BigUint>) -> Self {
        Self::Literal(value.into())
    }

    pub fn parameter(parameter: BoundParameter) -> Self {
        Self::Parameter(parameter)
    }

    pub fn add(terms: impl IntoIterator<Item = Self>) -> Self {
        Self::Add(terms.into_iter().collect())
    }

    pub fn multiply(terms: impl IntoIterator<Item = Self>) -> Self {
        Self::Multiply(terms.into_iter().collect())
    }

    /// The tight coefficient infinity-norm product factor for `Z[X]/(Xⁿ+1)`.
    pub fn negacyclic_product(left: Self, right: Self) -> Self {
        Self::NegacyclicProduct { left: Box::new(left), right: Box::new(right) }
    }

    /// Matrix product bound with explicit inner dimension `d`.
    pub fn matrix_product(inner_dimension: Self, left: Self, right: Self) -> Self {
        Self::MatrixProduct {
            inner_dimension: Box::new(inner_dimension),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn evaluate(&self, environment: &BoundEnvironment) -> Result<BigUint, BoundEvalError> {
        match self {
            Self::Literal(value) => Ok(value.clone()),
            Self::Parameter(parameter) => Ok(parameter.value(environment)),
            Self::Add(terms) => terms
                .iter()
                .try_fold(BigUint::zero(), |sum, term| Ok(sum + term.evaluate(environment)?)),
            Self::Multiply(terms) => terms.iter().try_fold(BigUint::from(1u8), |product, term| {
                Ok(product * term.evaluate(environment)?)
            }),
            Self::Maximum(terms) => terms
                .iter()
                .map(|term| term.evaluate(environment))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .ok_or(BoundEvalError::EmptyMaximum),
            Self::NegacyclicProduct { left, right } => {
                let n = &environment.ring_dimension;
                Ok(n * left.evaluate(environment)? * right.evaluate(environment)?)
            }
            Self::MatrixProduct { inner_dimension, left, right } => {
                let d = inner_dimension.evaluate(environment)?;
                let n = &environment.ring_dimension;
                Ok(d * n * left.evaluate(environment)? * right.evaluate(environment)?)
            }
        }
    }

    pub fn evaluate_checked(
        &self,
        environment: &BoundEnvironment,
    ) -> Result<BigUint, BoundEvalError> {
        let value = self.evaluate(environment)?;
        if value > environment.modulus {
            return Err(BoundEvalError::ExceedsModulus {
                value,
                modulus: environment.modulus.clone(),
            });
        }
        Ok(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum BoundEvalError {
    #[error("maximum requires at least one term")]
    EmptyMaximum,
    #[error("bound {value} exceeds modulus {modulus}")]
    ExceedsModulus { value: BigUint, modulus: BigUint },
}

/// The emitted expression, its closed environment, and its exact value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoundData {
    pub schema_version: u32,
    pub expression: BoundExpr,
    pub environment: BoundEnvironment,
    pub value: BigUint,
}

/// Parameters needed by the Diamond recurrence independent of the search
/// bookkeeping and simulator output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiamondBoundParameters {
    pub modulus: BigUint,
    pub ring_dimension: usize,
    pub state_rows: usize,
    pub state_columns: usize,
    pub gadget_columns: usize,
    pub error_coefficient_bound: BigUint,
    pub preimage_coefficient_bound: BigUint,
    pub gadget_decomposition_bound: BigUint,
    pub input_steps: usize,
    pub circuit_layers: usize,
}

impl DiamondBoundParameters {
    pub fn environment(&self) -> BoundEnvironment {
        BoundEnvironment {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.into(),
            state_rows: self.state_rows.into(),
            state_columns: self.state_columns.into(),
            gadget_columns: self.gadget_columns.into(),
            error_coefficient_bound: self.error_coefficient_bound.clone(),
            preimage_coefficient_bound: self.preimage_coefficient_bound.clone(),
            gadget_decomposition_bound: self.gadget_decomposition_bound.clone(),
            input_steps: self.input_steps.into(),
            circuit_layers: self.circuit_layers.into(),
        }
    }

    pub fn from_selected(selected: &DiamondSelectedParameters) -> Result<Self, BoundError> {
        let config = &selected.compiler.config;
        let state_rows = config.state_rows();
        let state_columns = config.state_columns().map_err(|error| {
            BoundError::InvalidParameter { name: "state_columns", message: error.to_string() }
        })?;
        let gadget_decomposition_bound = config
            .gadget_base
            .to_biguint()
            .and_then(|base| base.checked_sub(&BigUint::from(1u8)))
            .ok_or_else(|| BoundError::InvalidParameter {
                name: "gadget_decomposition_bound",
                message: "gadget base must be greater than one".to_owned(),
            })?;
        let error_coefficient_bound =
            config.error_max_coefficient_bound.to_biguint().ok_or_else(|| {
                BoundError::InvalidParameter {
                    name: "error_coefficient_bound",
                    message: "must be non-negative".to_owned(),
                }
            })?;
        let preimage_coefficient_bound = config
            .preimage_max_coefficient_bound
            .to_biguint()
            .ok_or_else(|| BoundError::InvalidParameter {
                name: "preimage_coefficient_bound",
                message: "must be non-negative".to_owned(),
            })?;
        Ok(Self {
            modulus: selected.modulus.clone(),
            ring_dimension: selected.ring_dimension as usize,
            state_rows,
            state_columns,
            gadget_columns: config.digit_count,
            error_coefficient_bound,
            preimage_coefficient_bound,
            gadget_decomposition_bound,
            input_steps: config.input_count,
            circuit_layers: selected.compiler.shape.depth,
        })
    }
}

/// A program view accepted by the Diamond bound derivation.
pub trait DiamondProgramView {
    fn validate_diamond_outputs(&self) -> Result<(), BoundError>;
}

impl DiamondProgramView for ValidatedLinkedProgram {
    fn validate_diamond_outputs(&self) -> Result<(), BoundError> {
        let projection = self
            .semantic_projection()
            .map_err(|error| BoundError::Program { message: error.to_string() })?;
        projection.validate_diamond_outputs()
    }
}

impl DiamondProgramView for ConcreteLinkedProgram {
    fn validate_diamond_outputs(&self) -> Result<(), BoundError> {
        let has_noisy_output = self.stages.iter().any(|stage| {
            stage.named_outputs.iter().any(|output| output.name == NOISY_PLAINTEXT_OUTPUT)
        });
        let has_decoded_output = self
            .stages
            .iter()
            .any(|stage| stage.named_outputs.iter().any(|output| output.name == "diamond-decoded"));
        let has_decryption_stage = self
            .stages
            .iter()
            .any(|stage| stage.key.contains("decryption") || stage.key == "decrypt");
        if self.stages.len() < 2 ||
            !has_decryption_stage ||
            !has_noisy_output ||
            !has_decoded_output
        {
            Err(BoundError::UnsupportedShape {
                message: "expected linked encryption/decryption stages and noisy/decoded outputs"
                    .to_owned(),
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum BoundError {
    #[error("Diamond program is not a valid linked semantic program: {message}")]
    Program { message: String },
    #[error("unsupported Diamond graph shape: {message}")]
    UnsupportedShape { message: String },
    #[error("unsupported Diamond node at stage {stage}, scope {scope}, node {node}: {kind}")]
    UnsupportedNode { stage: String, scope: usize, node: usize, kind: String },
    #[error("Diamond bound parameter {name} is invalid: {message}")]
    InvalidParameter { name: &'static str, message: String },
    #[error(transparent)]
    Evaluation(#[from] BoundEvalError),
}

/// Derive the Diamond recurrence from a validated linked program.
///
/// The program is consulted for the typed output identity and topology gate;
/// no caller-provided simulator root or raw node number participates in the
/// result.  All arithmetic is exact `BigUint` arithmetic.
pub fn derive_output_noise_bound(
    program: &impl DiamondProgramView,
    parameters: &DiamondSelectedParameters,
) -> Result<BoundData, BoundError> {
    let parameters = DiamondBoundParameters::from_selected(parameters)?;
    derive_output_noise_bound_with_parameters(program, &parameters)
}

pub fn derive_output_noise_bound_with_parameters(
    program: &impl DiamondProgramView,
    parameters: &DiamondBoundParameters,
) -> Result<BoundData, BoundError> {
    program.validate_diamond_outputs()?;
    derive_output_noise_bound_from_parameters(parameters)
}

/// Derive the canonical recurrence from already validated Diamond parameters.
/// This is used by cache validation, where the linked program is not retained
/// in the cache key but the parameter-derived recurrence must still be exact.
pub fn derive_output_noise_bound_from_parameters(
    parameters: &DiamondBoundParameters,
) -> Result<BoundData, BoundError> {
    validate_parameters(parameters)?;
    let expression = derive_recurrence_expression(parameters);
    let environment = parameters.environment();
    let value = expression.evaluate(&environment)?;
    Ok(BoundData { schema_version: DIAMOND_BOUND_SCHEMA_VERSION, expression, environment, value })
}

/// Build the finite Diamond recurrence as a plain expression tree.
///
/// `C` and `I` are the carrier and error bounds after each input-injector step.
/// `P` is the first BGG payload error, `A` is the gadget decomposition error,
/// `B` is the circuit-layer error, and `F` is the final fuse error.  The
/// recurrence is deliberately expanded for the two finite loop counts so the
/// emitted Lean expression has exactly the same operations as this function.
fn derive_recurrence_expression(parameters: &DiamondBoundParameters) -> BoundExpr {
    let n = BoundExpr::parameter(BoundParameter::RingDimension);
    let r = BoundExpr::parameter(BoundParameter::StateRows);
    let c = BoundExpr::parameter(BoundParameter::StateColumns);
    let g = BoundExpr::parameter(BoundParameter::GadgetColumns);
    let epsilon = BoundExpr::parameter(BoundParameter::ErrorCoefficientBound);
    let kappa = BoundExpr::parameter(BoundParameter::PreimageCoefficientBound);
    let delta = BoundExpr::parameter(BoundParameter::GadgetDecompositionBound);

    let mut carrier = BoundExpr::literal(1u8);
    let mut input_error = epsilon.clone();
    for _ in 0..parameters.input_steps {
        let current_carrier = carrier.clone();
        // C_{t+1} = r * n * C_t: one output coefficient of each state product.
        let next_carrier = BoundExpr::multiply([r.clone(), n.clone(), carrier]);
        // I_{t+1} = r*n*C_t*epsilon + c*n*I_t*kappa.
        let next_error = BoundExpr::add([
            BoundExpr::multiply([r.clone(), n.clone(), current_carrier, epsilon.clone()]),
            BoundExpr::multiply([c.clone(), n.clone(), input_error, kappa.clone()]),
        ]);
        carrier = next_carrier;
        input_error = next_error;
    }

    // P = c*n*I_t*kappa: the first payload/preimage product.
    let payload = BoundExpr::multiply([c.clone(), n.clone(), input_error, kappa.clone()]);
    // A = g*n*delta: the gadget decomposition error.
    let decomposition = BoundExpr::multiply([g.clone(), n.clone(), delta.clone()]);
    let mut circuit_error = payload.clone();
    for _ in 0..parameters.circuit_layers {
        // B_{l+1} = (2*A + 4) * B_l: one boolean-circuit layer.
        circuit_error = BoundExpr::multiply([
            BoundExpr::add([
                BoundExpr::multiply([BoundExpr::literal(2u8), decomposition.clone()]),
                BoundExpr::literal(4u8),
            ]),
            circuit_error,
        ]);
    }
    // F = 2*P + g*n*(P + B_l)*delta: the final fuse error.
    BoundExpr::add([
        BoundExpr::multiply([BoundExpr::literal(2u8), payload.clone()]),
        BoundExpr::multiply([g, n, BoundExpr::add([payload, circuit_error]), delta]),
    ])
}

fn validate_parameters(parameters: &DiamondBoundParameters) -> Result<(), BoundError> {
    if parameters.modulus < BigUint::from(4u8) {
        return Err(BoundError::InvalidParameter {
            name: "modulus",
            message: "must be at least four".to_owned(),
        });
    }
    for (name, value) in [
        ("ring_dimension", parameters.ring_dimension),
        ("state_rows", parameters.state_rows),
        ("state_columns", parameters.state_columns),
        ("gadget_columns", parameters.gadget_columns),
        ("input_steps", parameters.input_steps),
        ("circuit_layers", parameters.circuit_layers),
    ] {
        if value == 0 {
            return Err(BoundError::InvalidParameter {
                name,
                message: "must be non-zero".to_owned(),
            });
        }
    }
    if parameters.state_rows != 2 {
        return Err(BoundError::InvalidParameter {
            name: "state_rows",
            message: "Diamond input injector requires two state rows".to_owned(),
        });
    }
    if parameters.gadget_decomposition_bound == BigUint::zero() {
        return Err(BoundError::InvalidParameter {
            name: "gadget_decomposition_bound",
            message: "must be non-zero for a valid gadget base".to_owned(),
        });
    }
    Ok(())
}

impl fmt::Display for BoundParameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment() -> BoundEnvironment {
        BoundEnvironment {
            modulus: 1_000u32.into(),
            ring_dimension: 8u32.into(),
            state_rows: 2u32.into(),
            state_columns: 3u32.into(),
            gadget_columns: 3u32.into(),
            error_coefficient_bound: 2u32.into(),
            preimage_coefficient_bound: 5u32.into(),
            gadget_decomposition_bound: 1u32.into(),
            input_steps: 1u32.into(),
            circuit_layers: 0u32.into(),
        }
    }

    #[test]
    fn negacyclic_product_has_one_ring_factor() {
        let value = BoundExpr::negacyclic_product(BoundExpr::literal(2u8), BoundExpr::literal(3u8))
            .evaluate(&environment())
            .unwrap();
        assert_eq!(value, 48u32.into());
    }

    #[test]
    fn matrix_product_has_inner_dimension_and_one_ring_factor() {
        let value = BoundExpr::matrix_product(
            BoundExpr::parameter(BoundParameter::StateColumns),
            BoundExpr::literal(2u8),
            BoundExpr::literal(3u8),
        )
        .evaluate(&environment())
        .unwrap();
        assert_eq!(value, 144u32.into());
    }

    #[test]
    fn recurrence_uses_one_ring_factor_per_product() {
        let parameters = DiamondBoundParameters {
            modulus: 1_000_000_000u32.into(),
            ring_dimension: 8,
            state_rows: 2,
            state_columns: 10,
            gadget_columns: 3,
            error_coefficient_bound: 2u32.into(),
            preimage_coefficient_bound: 5u32.into(),
            gadget_decomposition_bound: 1u32.into(),
            input_steps: 1,
            circuit_layers: 2,
        };
        let expression = derive_recurrence_expression(&parameters);
        let value = expression.evaluate(&parameters.environment()).unwrap();
        let carrier: u64 = 2 * 8;
        let input_error: u64 = 2 * 8 * 1 * 2 + 10 * 8 * 2 * 5;
        let payload: u64 = 10 * 8 * input_error * 5;
        let decomposition: u64 = 3 * 8;
        let layer: u64 = (2 * decomposition + 4) * (2 * decomposition + 4);
        let expected: u64 = 2 * payload + 3 * 8 * (payload + layer * payload);
        assert_eq!(carrier, 16);
        assert_eq!(value, expected.into());
    }
}
