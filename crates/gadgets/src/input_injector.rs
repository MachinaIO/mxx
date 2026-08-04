//! BGG-independent input-injection preprocessing shared by Diamond applications.

use mxx_dsl::{
    BodyTraceRemapper, DslError, Family, GatherConstructionTrace, Int, LoopConstructionTrace, Mat,
    Parallel, RemapConstructionTrace, Ring, SelectConstructionTrace, Sequential, TrapdoorFamily,
    parallel_zip_bundle_result_traced,
};
use mxx_ir_core::{IntExpr, RealExpr, ValueHandle, node::ConcatAxis};
use num_bigint::BigInt;
use thiserror::Error;

pub const DIAMOND_SECRET_DIMENSION: usize = 1;
pub const DIAMOND_PREFIX_DIMENSION: usize = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputConfig {
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub input_count: usize,
    pub digit_base: usize,
    pub batch_bits: usize,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub error_max_coefficient_bound: BigInt,
    pub preimage_max_coefficient_bound: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputParams {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub input_count: IntExpr,
    pub digit_base: IntExpr,
    pub batch_bits: IntExpr,
    pub gadget_base: IntExpr,
    pub digit_count: IntExpr,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub error_max_coefficient_bound: IntExpr,
    pub preimage_max_coefficient_bound: IntExpr,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DiamondInputConfigError {
    #[error("the Diamond input-injection ring modulus must be positive")]
    InvalidModulus,
    #[error(
        "the Diamond input-injection ring dimension, input count, and digit count must be positive"
    )]
    ZeroDimension,
    #[error("batch_bits must be positive and fit the host digit representation")]
    InvalidBatchBits,
    #[error("digit_base must be at least 2^batch_bits")]
    InvalidDigitBase,
    #[error("the gadget base must be at least two")]
    InvalidGadgetBase,
    #[error("sampler coefficient bounds must be nonnegative")]
    InvalidSamplerBound,
    #[error("a Diamond input-injection layout calculation overflowed")]
    LayoutOverflow,
    #[error("Diamond input-injection transition artifacts do not match the configured layout")]
    InvalidTransitionLayout,
}

#[derive(Debug, Error)]
pub enum DiamondInputPreprocessError {
    #[error(transparent)]
    Config(#[from] DiamondInputConfigError),
    #[error(transparent)]
    Dsl(#[from] DslError),
}

pub struct DiamondInputPreprocessing {
    /// The initial input-injection vector p.
    pub p: Mat,
    /// Rectangular transition family indexed by `(level, digit, state)`.
    pub transitions: Family<Mat>,
    /// Trapdoors for the final state bases, returned for application-specific projections.
    pub final_trapdoors: TrapdoorFamily,
    #[doc(hidden)]
    pub construction_trace: DiamondInputPreprocessingConstructionTrace,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct OperationConstructionTrace {
    pub inputs: Vec<ValueHandle>,
    pub outputs: Vec<ValueHandle>,
}

impl RemapConstructionTrace for OperationConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            inputs: self.inputs.remap_current_body(map)?,
            outputs: self.outputs.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct PreimageConstructionTrace {
    pub sample: OperationConstructionTrace,
    pub materialize: OperationConstructionTrace,
}

impl RemapConstructionTrace for PreimageConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            sample: self.sample.remap_current_body(map)?,
            materialize: self.materialize.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct TransitionTargetConstructionTrace {
    pub digit_secret: ValueHandle,
    pub target_public: ValueHandle,
    pub selector: ValueHandle,
    pub selector_construction: SelectorConstructionTrace,
    pub error_sample: OperationConstructionTrace,
    pub selector_product: OperationConstructionTrace,
    pub target_sum: OperationConstructionTrace,
}

impl RemapConstructionTrace for TransitionTargetConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            digit_secret: self.digit_secret.remap_current_body(map)?,
            target_public: self.target_public.remap_current_body(map)?,
            selector: self.selector.remap_current_body(map)?,
            selector_construction: self.selector_construction.remap_current_body(map)?,
            error_sample: self.error_sample.remap_current_body(map)?,
            selector_product: self.selector_product.remap_current_body(map)?,
            target_sum: self.target_sum.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SelectorConstructionTrace {
    pub regular: OperationConstructionTrace,
    pub k_identity: OperationConstructionTrace,
    pub k: OperationConstructionTrace,
    pub initial_select: SelectConstructionTrace,
    pub bit_scan: LoopConstructionTrace<SelectorBitConstructionTrace>,
}

impl RemapConstructionTrace for SelectorConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            regular: self.regular.remap_current_body(map)?,
            k_identity: self.k_identity.remap_current_body(map)?,
            k: self.k.remap_current_body(map)?,
            initial_select: self.initial_select.remap_current_body(map)?,
            bit_scan: self.bit_scan.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SelectorBitConstructionTrace {
    pub bit_extract: OperationConstructionTrace,
    pub bit_to_int: OperationConstructionTrace,
    pub bit_zero: OperationConstructionTrace,
    pub bit_one: OperationConstructionTrace,
    pub bit_select: SelectConstructionTrace,
    pub special_product: OperationConstructionTrace,
    pub special_top: OperationConstructionTrace,
    pub special_bottom: OperationConstructionTrace,
    pub special: OperationConstructionTrace,
    pub state_match: OperationConstructionTrace,
    pub state_match_to_int: OperationConstructionTrace,
    pub selector: SelectConstructionTrace,
}

impl RemapConstructionTrace for SelectorBitConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            bit_extract: self.bit_extract.remap_current_body(map)?,
            bit_to_int: self.bit_to_int.remap_current_body(map)?,
            bit_zero: self.bit_zero.remap_current_body(map)?,
            bit_one: self.bit_one.remap_current_body(map)?,
            bit_select: self.bit_select.remap_current_body(map)?,
            special_product: self.special_product.remap_current_body(map)?,
            special_top: self.special_top.remap_current_body(map)?,
            special_bottom: self.special_bottom.remap_current_body(map)?,
            special: self.special.remap_current_body(map)?,
            state_match: self.state_match.remap_current_body(map)?,
            state_match_to_int: self.state_match_to_int.remap_current_body(map)?,
            selector: self.selector.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct DiamondInputPreprocessingConstructionTrace {
    pub trapdoor_samples: LoopConstructionTrace<OperationConstructionTrace>,
    pub secret_sample: OperationConstructionTrace,
    pub message_selector: OperationConstructionTrace,
    pub initial_error_sample: OperationConstructionTrace,
    pub initial_public_product: OperationConstructionTrace,
    pub initial_state: OperationConstructionTrace,
    pub transition_source_indices: LoopConstructionTrace<ValueHandle>,
    pub transition_target_indices: LoopConstructionTrace<ValueHandle>,
    pub digit_secret_indices: LoopConstructionTrace<ValueHandle>,
    pub digit_secret_samples: LoopConstructionTrace<OperationConstructionTrace>,
    pub digit_secrets: LoopConstructionTrace<GatherConstructionTrace>,
    pub transition_sources: LoopConstructionTrace<GatherConstructionTrace>,
    pub target_public_matrices: LoopConstructionTrace<GatherConstructionTrace>,
    pub transition_targets: LoopConstructionTrace<TransitionTargetConstructionTrace>,
    pub transition_preimages: LoopConstructionTrace<PreimageConstructionTrace>,
    pub final_indices: LoopConstructionTrace<ValueHandle>,
    pub final_trapdoors: LoopConstructionTrace<GatherConstructionTrace>,
}

/// Online result of applying the input-selected transition matrices.
///
/// `states[0]` is the default `(s, k)` state.  The remaining entries are the
/// bit-specific states in the same order returned by
/// [`DiamondInputConfig::bit_state_index`].
pub struct DiamondInputEvaluation {
    pub states: Family<Mat>,
    #[doc(hidden)]
    pub construction_trace: DiamondInputEvaluationConstructionTrace,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct DiamondInputEvaluationConstructionTrace {
    pub initial_states_expansion: LoopConstructionTrace<SelectConstructionTrace>,
    pub initial_states: ValueHandle,
    pub packed_digits: ValueHandle,
    pub transitions: ValueHandle,
    pub state_scan: LoopConstructionTrace<DiamondInputEvaluationBodyConstructionTrace>,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct DiamondInputEvaluationBodyConstructionTrace {
    pub body_states: ValueHandle,
    pub body_packed_digits: ValueHandle,
    pub body_transitions: ValueHandle,
    pub selected_digit: GatherConstructionTrace,
    pub source_indices: LoopConstructionTrace<ValueHandle>,
    pub source_states: LoopConstructionTrace<GatherConstructionTrace>,
    pub transition_indices: LoopConstructionTrace<ValueHandle>,
    pub selected_transitions: LoopConstructionTrace<GatherConstructionTrace>,
    pub state_products: LoopConstructionTrace<MatrixProductConstructionTrace>,
    pub body_output: ValueHandle,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MatrixProductConstructionTrace {
    pub left: ValueHandle,
    pub right: ValueHandle,
    pub output: ValueHandle,
}

impl RemapConstructionTrace for MatrixProductConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            left: self.left.remap_current_body(map)?,
            right: self.right.remap_current_body(map)?,
            output: self.output.remap_current_body(map)?,
        })
    }
}

impl RemapConstructionTrace for DiamondInputEvaluationBodyConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            body_states: self.body_states.remap_current_body(map)?,
            body_packed_digits: self.body_packed_digits.remap_current_body(map)?,
            body_transitions: self.body_transitions.remap_current_body(map)?,
            selected_digit: self.selected_digit.remap_current_body(map)?,
            source_indices: self.source_indices.remap_current_body(map)?,
            source_states: self.source_states.remap_current_body(map)?,
            transition_indices: self.transition_indices.remap_current_body(map)?,
            selected_transitions: self.selected_transitions.remap_current_body(map)?,
            state_products: self.state_products.remap_current_body(map)?,
            body_output: self.body_output.remap_current_body(map)?,
        })
    }
}

#[derive(Clone)]
pub struct DiamondInputInjector {
    pub params: DiamondInputParams,
}

impl DiamondInputConfig {
    pub fn validate(&self) -> Result<(), DiamondInputConfigError> {
        if self.modulus <= BigInt::from(0) {
            return Err(DiamondInputConfigError::InvalidModulus);
        }
        if self.ring_dimension == 0 || self.input_count == 0 || self.digit_count == 0 {
            return Err(DiamondInputConfigError::ZeroDimension);
        }
        if self.batch_bits == 0 || self.batch_bits >= usize::BITS as usize {
            return Err(DiamondInputConfigError::InvalidBatchBits);
        }
        let required_base = 1usize
            .checked_shl(self.batch_bits as u32)
            .ok_or(DiamondInputConfigError::LayoutOverflow)?;
        if self.digit_base < required_base {
            return Err(DiamondInputConfigError::InvalidDigitBase);
        }
        if self.gadget_base < BigInt::from(2) {
            return Err(DiamondInputConfigError::InvalidGadgetBase);
        }
        if self.error_max_coefficient_bound < BigInt::from(0) ||
            self.preimage_max_coefficient_bound < BigInt::from(0)
        {
            return Err(DiamondInputConfigError::InvalidSamplerBound);
        }
        self.witness_size()?;
        self.state_columns()?;
        Ok(())
    }

    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension)
    }

    pub fn witness_size(&self) -> Result<usize, DiamondInputConfigError> {
        self.input_count.checked_mul(self.batch_bits).ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn state_rows(&self) -> usize {
        DIAMOND_PREFIX_DIMENSION * DIAMOND_SECRET_DIMENSION
    }

    pub fn state_columns(&self) -> Result<usize, DiamondInputConfigError> {
        self.state_rows()
            .checked_mul(
                self.digit_count.checked_add(2).ok_or(DiamondInputConfigError::LayoutOverflow)?,
            )
            .ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn state_count_at_level(&self, level: usize) -> Result<usize, DiamondInputConfigError> {
        level
            .checked_mul(self.batch_bits)
            .and_then(|count| count.checked_add(1))
            .ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn bit_state_index(
        &self,
        digit_index: usize,
        bit_index: usize,
    ) -> Result<usize, DiamondInputConfigError> {
        digit_index
            .checked_mul(self.batch_bits)
            .and_then(|index| index.checked_add(bit_index))
            .and_then(|index| index.checked_add(1))
            .ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn gadget_base_expr(&self) -> IntExpr {
        self.gadget_base.clone().into()
    }

    pub fn digit_count_expr(&self) -> IntExpr {
        self.digit_count.into()
    }

    pub fn params(&self) -> DiamondInputParams {
        DiamondInputParams {
            modulus: self.modulus.clone().into(),
            ring_dimension: self.ring_dimension.into(),
            input_count: self.input_count.into(),
            digit_base: self.digit_base.into(),
            batch_bits: self.batch_bits.into(),
            gadget_base: self.gadget_base_expr(),
            digit_count: self.digit_count_expr(),
            trapdoor_sigma: self.trapdoor_sigma.clone(),
            error_sigma: self.error_sigma.clone(),
            error_max_coefficient_bound: self.error_max_coefficient_bound.clone().into(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone().into(),
        }
    }
}

impl DiamondInputParams {
    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }

    pub fn witness_size(&self) -> IntExpr {
        IntExpr::Mul(Box::new(self.input_count.clone()), Box::new(self.batch_bits.clone()))
            .canonicalize()
    }

    pub fn state_rows(&self) -> IntExpr {
        IntExpr::constant(DIAMOND_PREFIX_DIMENSION * DIAMOND_SECRET_DIMENSION)
    }

    pub fn state_columns(&self) -> IntExpr {
        IntExpr::Mul(
            Box::new(self.state_rows()),
            Box::new(IntExpr::Add(
                Box::new(self.digit_count.clone()),
                Box::new(IntExpr::constant(2)),
            )),
        )
        .canonicalize()
    }

    pub fn max_state_count(&self) -> IntExpr {
        IntExpr::Add(Box::new(IntExpr::constant(1)), Box::new(self.witness_size())).canonicalize()
    }
}

impl DiamondInputInjector {
    pub fn new(config: DiamondInputConfig) -> Result<Self, DiamondInputConfigError> {
        config.validate()?;
        Ok(Self { params: config.params() })
    }

    pub fn parameterized(params: DiamondInputParams) -> Self {
        Self { params }
    }

    pub fn preprocess(
        &self,
        message: Mat,
    ) -> Result<DiamondInputPreprocessing, DiamondInputPreprocessError> {
        let ring = self.params.ring();
        let state_rows = self.params.state_rows();
        let state_columns = self.params.state_columns();
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        let digit_state_count =
            IntExpr::Mul(Box::new(digit_base.clone()), Box::new(max_state_count.clone()))
                .canonicalize();
        let base_count = IntExpr::Mul(
            Box::new(IntExpr::Add(Box::new(level_count.clone()), Box::new(IntExpr::constant(1)))),
            Box::new(max_state_count.clone()),
        )
        .canonicalize();
        let (bases, trapdoor_samples) = Parallel::range(base_count).map_values_traced(|_| {
            let trapdoor = ring.sample_trapdoor(
                state_rows.clone(),
                self.params.trapdoor_sigma.clone(),
                self.params.gadget_base.clone(),
                self.params.digit_count.clone(),
                self.params.preimage_max_coefficient_bound.clone(),
            );
            let trace = OperationConstructionTrace {
                inputs: Vec::new(),
                outputs: vec![
                    trapdoor.public_matrix().value_handle().clone(),
                    trapdoor.value_handle().clone(),
                ],
            };
            (trapdoor, trace)
        })?;

        let secret_epsilon = ternary_secret(&ring);
        let secret_sample = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![secret_epsilon.value_handle().clone()],
        };
        let selector_inputs =
            vec![secret_epsilon.value_handle().clone(), message.value_handle().clone()];
        let selector = Mat::concat(ConcatAxis::Columns, vec![secret_epsilon, message]);
        let message_selector = OperationConstructionTrace {
            inputs: selector_inputs,
            outputs: vec![selector.value_handle().clone()],
        };
        let base_public = bases.get_static(0).public_matrix();
        let initial_product_inputs =
            vec![selector.value_handle().clone(), base_public.value_handle().clone()];
        let initial_public_product_value = selector * base_public;
        let initial_public_product = OperationConstructionTrace {
            inputs: initial_product_inputs,
            outputs: vec![initial_public_product_value.value_handle().clone()],
        };
        let initial_error = ring.gaussian(
            (1, state_columns.clone()),
            self.params.error_sigma.clone(),
            self.params.error_max_coefficient_bound.clone(),
        );
        let initial_error_sample = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![initial_error.value_handle().clone()],
        };
        let initial_sum_inputs = vec![
            initial_public_product_value.value_handle().clone(),
            initial_error.value_handle().clone(),
        ];
        let p = initial_public_product_value + initial_error;
        let initial_state = OperationConstructionTrace {
            inputs: initial_sum_inputs,
            outputs: vec![p.value_handle().clone()],
        };

        let transition_count =
            IntExpr::Mul(Box::new(level_count.clone()), Box::new(digit_state_count.clone()))
                .canonicalize();
        let (transition_indices, transition_source_indices) =
            Parallel::range(transition_count.clone()).map_values_traced(|slot| {
                let flat = slot.as_int();
                let state = flat.clone().rem(Int::evaluate(max_state_count.clone()));
                let level = flat.clone().div(Int::evaluate(digit_state_count.clone()));
                let first_new =
                    level.clone().mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let source_state = first_new
                    .less_equal(state.clone())
                    .to_int()
                    .select_int(vec![state, Int::constant(0)])
                    .expect("two integer branches");
                let output = level.mul(Int::evaluate(max_state_count.clone())).add(source_state);
                (output.clone(), output.value_handle().clone())
            })?;
        let (target_indices, transition_target_indices) = Parallel::range(transition_count.clone())
            .map_values_traced(|slot| {
                let flat = slot.as_int();
                let state = flat.clone().rem(Int::evaluate(max_state_count.clone()));
                let level = flat.div(Int::evaluate(digit_state_count.clone()));
                let output = level
                    .add(Int::constant(1))
                    .mul(Int::evaluate(max_state_count.clone()))
                    .add(state);
                (output.clone(), output.value_handle().clone())
            })?;
        let (digit_secret_indices, digit_secret_indices_trace) = Parallel::range(transition_count)
            .map_values_traced(|slot| {
                let output = slot.as_int().div(Int::evaluate(max_state_count.clone()));
                (output.clone(), output.value_handle().clone())
            })?;
        let (digit_secret_samples_family, digit_secret_samples) = Parallel::range(IntExpr::Mul(
            Box::new(level_count.clone()),
            Box::new(digit_base.clone()),
        ))
        .map_values_traced(|_| {
            let secret = ternary_secret(&ring);
            let trace = OperationConstructionTrace {
                inputs: Vec::new(),
                outputs: vec![secret.value_handle().clone()],
            };
            (secret, trace)
        })?;
        let (digit_secrets, digit_secrets_trace) =
            digit_secret_samples_family.parallel_gather_traced(digit_secret_indices)?;
        let (sources, transition_sources) =
            bases.clone().parallel_gather_traced(transition_indices)?;
        let (target_public, target_public_matrices) =
            bases.public_matrices().parallel_gather_traced(target_indices)?;
        let sigma = self.params.error_sigma.clone();
        let error_bound = self.params.error_max_coefficient_bound.clone();
        let (targets, transition_targets) = parallel_zip_bundle_result_traced(
            (digit_secrets, target_public),
            |slot, (secret, public)| {
                let digit_secret = secret.value_handle().clone();
                let target_public = public.value_handle().clone();
                let flat = slot.as_int();
                let state = flat.clone().rem(Int::evaluate(max_state_count.clone()));
                let digit = flat
                    .clone()
                    .div(Int::evaluate(max_state_count.clone()))
                    .rem(Int::evaluate(digit_base.clone()));
                let level = flat.div(Int::evaluate(digit_state_count.clone()));
                let first_new = level.mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let regular_inputs =
                    vec![secret.value_handle().clone(), secret.value_handle().clone()];
                let regular = regular_selector(secret.clone());
                let regular_trace = OperationConstructionTrace {
                    inputs: regular_inputs,
                    outputs: vec![regular.value_handle().clone()],
                };
                let k_identity = ring.identity(1);
                let k_identity_trace = OperationConstructionTrace {
                    inputs: Vec::new(),
                    outputs: vec![k_identity.value_handle().clone()],
                };
                let k_inputs =
                    vec![secret.value_handle().clone(), k_identity.value_handle().clone()];
                let k = Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), k_identity]);
                let k_trace = OperationConstructionTrace {
                    inputs: k_inputs,
                    outputs: vec![k.value_handle().clone()],
                };
                let initial_match = state.clone().equal(Int::constant(0)).to_int();
                let initial_select_selector = initial_match.value_handle().clone();
                let initial_select_branches =
                    vec![regular.value_handle().clone(), k.value_handle().clone()];
                let selector = initial_match.select(vec![regular, k])?;
                let initial_select = SelectConstructionTrace {
                    selector: initial_select_selector,
                    branches: initial_select_branches,
                    output: selector.value_handle().clone(),
                };
                let (selector, bit_scan) = Sequential::range(batch_bits.clone()).scan_traced(
                    selector,
                    (digit, (state, (first_new, secret))),
                    |bit, selector, (digit, (state, (first_new, secret)))| {
                        let bit_extract_input = digit.value_handle().clone();
                        let extracted = digit.clone().bit(bit.expression());
                        let bit_extract = OperationConstructionTrace {
                            inputs: vec![bit_extract_input],
                            outputs: vec![extracted.value_handle().clone()],
                        };
                        let bit_to_int_input = extracted.value_handle().clone();
                        let extracted_int = extracted.to_int();
                        let bit_to_int = OperationConstructionTrace {
                            inputs: vec![bit_to_int_input],
                            outputs: vec![extracted_int.value_handle().clone()],
                        };
                        let bit_zero_value = ring.zero((1, 1));
                        let bit_zero = OperationConstructionTrace {
                            inputs: Vec::new(),
                            outputs: vec![bit_zero_value.value_handle().clone()],
                        };
                        let bit_one_value = ring.identity(1);
                        let bit_one = OperationConstructionTrace {
                            inputs: Vec::new(),
                            outputs: vec![bit_one_value.value_handle().clone()],
                        };
                        let bit_select_selector = extracted_int.value_handle().clone();
                        let bit_select_branches = vec![
                            bit_zero_value.value_handle().clone(),
                            bit_one_value.value_handle().clone(),
                        ];
                        let bit_value =
                            extracted_int.select(vec![bit_zero_value, bit_one_value])?;
                        let bit_select = SelectConstructionTrace {
                            selector: bit_select_selector,
                            branches: bit_select_branches,
                            output: bit_value.value_handle().clone(),
                        };
                        let special_product_inputs =
                            vec![secret.value_handle().clone(), bit_value.value_handle().clone()];
                        let special_product = secret.clone() * bit_value;
                        let special_product_trace = OperationConstructionTrace {
                            inputs: special_product_inputs,
                            outputs: vec![special_product.value_handle().clone()],
                        };
                        let special_top_inputs = vec![
                            secret.value_handle().clone(),
                            special_product.value_handle().clone(),
                        ];
                        let special_top =
                            Mat::concat(ConcatAxis::Columns, vec![secret, special_product]);
                        let special_top_trace = OperationConstructionTrace {
                            inputs: special_top_inputs,
                            outputs: vec![special_top.value_handle().clone()],
                        };
                        let special_bottom_value = ring.zero((1, 2));
                        let special_bottom = OperationConstructionTrace {
                            inputs: Vec::new(),
                            outputs: vec![special_bottom_value.value_handle().clone()],
                        };
                        let special_inputs = vec![
                            special_top.value_handle().clone(),
                            special_bottom_value.value_handle().clone(),
                        ];
                        let special =
                            Mat::concat(ConcatAxis::Rows, vec![special_top, special_bottom_value]);
                        let special_trace = OperationConstructionTrace {
                            inputs: special_inputs,
                            outputs: vec![special.value_handle().clone()],
                        };
                        let expected_state = first_new.add(bit.as_int());
                        let state_match_inputs = vec![
                            state.value_handle().clone(),
                            expected_state.value_handle().clone(),
                        ];
                        let state_match_value = state.equal(expected_state);
                        let state_match = OperationConstructionTrace {
                            inputs: state_match_inputs,
                            outputs: vec![state_match_value.value_handle().clone()],
                        };
                        let state_match_to_int_input = state_match_value.value_handle().clone();
                        let state_match_int = state_match_value.to_int();
                        let state_match_to_int = OperationConstructionTrace {
                            inputs: vec![state_match_to_int_input],
                            outputs: vec![state_match_int.value_handle().clone()],
                        };
                        let selector_input = state_match_int.value_handle().clone();
                        let selector_branches =
                            vec![selector.value_handle().clone(), special.value_handle().clone()];
                        let output = state_match_int.select(vec![selector, special])?;
                        Ok((
                            output.clone(),
                            SelectorBitConstructionTrace {
                                bit_extract,
                                bit_to_int,
                                bit_zero,
                                bit_one,
                                bit_select,
                                special_product: special_product_trace,
                                special_top: special_top_trace,
                                special_bottom,
                                special: special_trace,
                                state_match,
                                state_match_to_int,
                                selector: SelectConstructionTrace {
                                    selector: selector_input,
                                    branches: selector_branches,
                                    output: output.value_handle().clone(),
                                },
                            },
                        ))
                    },
                )?;
                let selector_handle = selector.value_handle().clone();
                let product_inputs = vec![selector_handle.clone(), public.value_handle().clone()];
                let selector_product_value = selector * public;
                let selector_product = OperationConstructionTrace {
                    inputs: product_inputs,
                    outputs: vec![selector_product_value.value_handle().clone()],
                };
                let error = ring.gaussian(
                    (state_rows.clone(), state_columns.clone()),
                    sigma.clone(),
                    error_bound.clone(),
                );
                let error_sample = OperationConstructionTrace {
                    inputs: Vec::new(),
                    outputs: vec![error.value_handle().clone()],
                };
                let sum_inputs = vec![
                    selector_product_value.value_handle().clone(),
                    error.value_handle().clone(),
                ];
                let target = selector_product_value + error;
                Ok::<_, DslError>((
                    target.clone(),
                    TransitionTargetConstructionTrace {
                        digit_secret,
                        target_public,
                        selector: selector_handle,
                        selector_construction: SelectorConstructionTrace {
                            regular: regular_trace,
                            k_identity: k_identity_trace,
                            k: k_trace,
                            initial_select,
                            bit_scan,
                        },
                        error_sample,
                        selector_product,
                        target_sum: OperationConstructionTrace {
                            inputs: sum_inputs,
                            outputs: vec![target.value_handle().clone()],
                        },
                    },
                ))
            },
        )?;
        let (transitions, transition_preimages) =
            sources.parallel_zip_mat_values_traced(targets, |_, source, target| {
                let sample_inputs = vec![
                    source.public_matrix().value_handle().clone(),
                    source.value_handle().clone(),
                    target.value_handle().clone(),
                ];
                let sample =
                    source.sample_preimage(target, (state_columns.clone(), state_columns.clone()));
                let sample_handle = sample.value_handle().clone();
                let materialized = sample.as_mat();
                (
                    materialized.clone(),
                    PreimageConstructionTrace {
                        sample: OperationConstructionTrace {
                            inputs: sample_inputs,
                            outputs: vec![sample_handle.clone()],
                        },
                        materialize: OperationConstructionTrace {
                            inputs: vec![sample_handle],
                            outputs: vec![materialized.value_handle().clone()],
                        },
                    },
                )
            })?;
        let (final_indices, final_indices_trace) = Parallel::range(max_state_count.clone())
            .map_values_traced(|state| {
                let output = Int::evaluate(IntExpr::Mul(
                    Box::new(level_count.clone()),
                    Box::new(max_state_count.clone()),
                ))
                .add(state.as_int());
                (output.clone(), output.value_handle().clone())
            })?;
        let (final_trapdoors, final_trapdoors_trace) =
            bases.parallel_gather_traced(final_indices)?;
        Ok(DiamondInputPreprocessing {
            p,
            transitions,
            final_trapdoors,
            construction_trace: DiamondInputPreprocessingConstructionTrace {
                trapdoor_samples,
                secret_sample,
                message_selector,
                initial_error_sample,
                initial_public_product,
                initial_state,
                transition_source_indices,
                transition_target_indices,
                digit_secret_indices: digit_secret_indices_trace,
                digit_secret_samples,
                digit_secrets: digit_secrets_trace,
                transition_sources,
                target_public_matrices,
                transition_targets,
                transition_preimages,
                final_indices: final_indices_trace,
                final_trapdoors: final_trapdoors_trace,
            },
        })
    }

    /// Applies the preprocessed transition matrices to one packed input.
    ///
    /// The transition layout is exactly the one returned by [`Self::preprocess`]:
    /// `[level][digit][state]`.  Selection is represented by the DSL `Select`
    /// node and all independent state transitions at a level are represented by
    /// one IR parallel loop.
    pub fn evaluate(
        &self,
        initial_state: Mat,
        input_digits: Family<Int>,
        transitions: Family<Mat>,
    ) -> Result<DiamondInputEvaluation, DiamondInputPreprocessError> {
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        let digit_state_count =
            IntExpr::Mul(Box::new(digit_base.clone()), Box::new(max_state_count.clone()))
                .canonicalize();
        let expected_transitions =
            IntExpr::Mul(Box::new(level_count.clone()), Box::new(digit_state_count.clone()))
                .canonicalize();
        if input_digits.count().canonicalize() != level_count.canonicalize() ||
            transitions.count().canonicalize() != expected_transitions
        {
            return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
        }
        let (initial, initial_states_expansion) = Parallel::range(max_state_count.clone())
            .map_values_traced(|state| {
                let selector = state.as_int().equal(Int::constant(0)).to_int();
                let zero = self.params.ring().zero((1, self.params.state_columns()));
                let selector_handle = selector.value_handle().clone();
                let branches =
                    vec![zero.value_handle().clone(), initial_state.value_handle().clone()];
                let output = selector
                    .select(vec![zero, initial_state.clone()])
                    .expect("matching state matrices");
                (
                    output.clone(),
                    SelectConstructionTrace {
                        selector: selector_handle,
                        branches,
                        output: output.value_handle().clone(),
                    },
                )
            })?;
        let initial_states = initial.value_handle().clone();
        let packed_digits = input_digits.value_handle().clone();
        let transition_family = transitions.value_handle().clone();
        let (states, state_scan) = Sequential::range(level_count).scan_traced(
            initial,
            (input_digits, transitions),
            |level, states, (input_digits, transitions)| {
                let level = level.as_int();
                let digit_index = level.clone();
                let digit_index_handle = digit_index.value_handle().clone();
                let digit_family_handle = input_digits.value_handle().clone();
                let digit = input_digits.get(digit_index);
                let body_states = states.value_handle().clone();
                let body_packed_digits = input_digits.value_handle().clone();
                let body_transitions = transitions.value_handle().clone();
                let first_new =
                    level.clone().mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let (source_indices, source_indices_trace) =
                    Parallel::range(max_state_count.clone()).map_values_traced(|state| {
                        let state = state.as_int();
                        let output = first_new
                            .clone()
                            .less_equal(state.clone())
                            .to_int()
                            .select_int(vec![state, Int::constant(0)])
                            .expect("two integer branches");
                        (output.clone(), output.value_handle().clone())
                    })?;
                let (source_states, source_states_trace) =
                    states.parallel_gather_traced(source_indices)?;
                let (transition_indices, transition_indices_trace) =
                    Parallel::range(max_state_count.clone()).map_values_traced(|state| {
                        let output = level
                            .clone()
                            .mul(Int::evaluate(digit_state_count.clone()))
                            .add(digit.clone().mul(Int::evaluate(max_state_count.clone())))
                            .add(state.as_int());
                        (output.clone(), output.value_handle().clone())
                    })?;
                let (selected, selected_transitions_trace) =
                    transitions.parallel_gather_traced(transition_indices)?;
                let selected_digit = GatherConstructionTrace {
                    index: digit_index_handle,
                    sources: vec![digit_family_handle],
                    outputs: vec![digit.value_handle().clone()],
                };
                let (products, state_products) = parallel_zip_bundle_result_traced(
                    (source_states, selected),
                    |_, (state, transition)| {
                        let left = state.value_handle().clone();
                        let right = transition.value_handle().clone();
                        let output = state * transition;
                        Ok((
                            output.clone(),
                            MatrixProductConstructionTrace {
                                left,
                                right,
                                output: output.value_handle().clone(),
                            },
                        ))
                    },
                )?;
                let body_output = products.value_handle().clone();
                Ok((
                    products,
                    DiamondInputEvaluationBodyConstructionTrace {
                        body_states,
                        body_packed_digits,
                        body_transitions,
                        selected_digit,
                        source_indices: source_indices_trace,
                        source_states: source_states_trace,
                        transition_indices: transition_indices_trace,
                        selected_transitions: selected_transitions_trace,
                        state_products,
                        body_output,
                    },
                ))
            },
        )?;
        Ok(DiamondInputEvaluation {
            construction_trace: DiamondInputEvaluationConstructionTrace {
                initial_states_expansion,
                initial_states,
                packed_digits,
                transitions: transition_family,
                state_scan,
            },
            states,
        })
    }
}

fn ternary_secret(ring: &Ring) -> Mat {
    ring.uniform_in((1, 1), -1, 1)
}

fn regular_selector(secret: Mat) -> Mat {
    Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), secret])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, node::NodeKind};

    fn config() -> DiamondInputConfig {
        DiamondInputConfig {
            modulus: BigInt::from(257),
            ring_dimension: 8,
            input_count: 2,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: BigInt::from(4),
            digit_count: 2,
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(3),
            error_max_coefficient_bound: BigInt::from(19),
            preimage_max_coefficient_bound: BigInt::from(64),
        }
    }

    #[test]
    fn preprocessing_builds_p_transitions_and_final_trapdoors() {
        let config = config();
        let ring = config.ring();
        let injector = DiamondInputInjector::new(config).unwrap();
        let preprocessing =
            injector.preprocess(ring.input("message", (1, 1))).expect("preprocessing");
        let trace = preprocessing.construction_trace.clone();
        assert_eq!(preprocessing.transitions.count(), &IntExpr::constant(12));
        assert_eq!(preprocessing.final_trapdoors.count(), &IntExpr::constant(3));

        let (built, freeze_map) = DslContext::new("diamond-input-preprocessing")
            .output("p", preprocessing.p)
            .unwrap()
            .output("transition", preprocessing.transitions.get_static(11))
            .unwrap()
            .build_with_freeze_map()
            .unwrap();
        freeze_map
            .resolve_unique(&trace.initial_state.outputs[0])
            .expect("initial state trace resolves exactly");
        freeze_map
            .resolve_unique(&trace.transition_preimages.outputs[0])
            .expect("transition preimage loop resolves exactly");
        freeze_map
            .resolve_unique(&trace.transition_preimages.scope.body.sample.outputs[0])
            .expect("nested transition preimage sample resolves exactly");
        freeze_map
            .resolve_unique(
                &trace.transition_targets.scope.body.selector_construction.bit_scan.outputs[0],
            )
            .expect("nested selector scan resolves exactly");
        freeze_map
            .resolve_unique(
                &trace
                    .transition_targets
                    .scope
                    .body
                    .selector_construction
                    .bit_scan
                    .scope
                    .body
                    .special
                    .outputs[0],
            )
            .expect("selector special-matrix construction resolves exactly");
        let validated = built.validate(&ParamEnv::default()).unwrap();
        assert!(
            validated
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::PreimageSample { .. }))
        );
    }

    #[test]
    fn online_evaluation_selects_transitions_and_uses_parallel_state_updates() {
        let config = config();
        let input_count = config.input_count;
        let ring = config.ring();
        let injector = DiamondInputInjector::new(config).unwrap();
        let preprocessing =
            injector.preprocess(ring.input("message", (1, 1))).expect("preprocessing");
        let digits = Family::pack(
            (0..input_count)
                .map(|digit| ring.input(format!("digit-{digit}"), (1, 1)).extract_coefficient(0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let evaluation = injector
            .evaluate(preprocessing.p, digits, preprocessing.transitions)
            .expect("online evaluation");
        let trace = evaluation.construction_trace.clone();
        let (graph, freeze_map) = DslContext::new("diamond-input-online")
            .output("default-state", evaluation.states.get_static(0))
            .unwrap()
            .output("last-state", evaluation.states.get_static(2))
            .unwrap()
            .build_with_freeze_map()
            .unwrap();
        freeze_map
            .resolve_unique(&trace.state_scan.outputs[0])
            .expect("state scan trace resolves exactly");
        freeze_map
            .resolve_unique(&trace.state_scan.scope.body.state_products.outputs[0])
            .expect("nested state-product loop resolves exactly");
        freeze_map
            .resolve_unique(&trace.state_scan.scope.body.state_products.scope.body.output)
            .expect("nested state product resolves exactly");
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        assert!(
            validated
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::ParallelLoop { .. }))
        );
        assert!(
            validated
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::Select { .. }))
        );
    }
}
