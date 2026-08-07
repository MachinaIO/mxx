//! BGG-independent input-injection preprocessing shared by Diamond applications.

use mxx_dsl::{
    DslError, Family, Int, Mat, Parallel, Ring, Sequential, TrapdoorFamily,
    parallel_zip_bundle_result,
};
use mxx_ir_core::{IntExpr, RealExpr, node::ConcatAxis};
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
}

/// Online result of applying the input-selected transition matrices.
///
/// `states[0]` is the default `(s, k)` state.  The remaining entries are the
/// bit-specific states in the same order returned by
/// [`DiamondInputConfig::bit_state_index`].
pub struct DiamondInputEvaluation {
    pub states: Family<Mat>,
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
        let bases = Parallel::range(base_count).map_values(|_| {
            ring.sample_trapdoor(
                state_rows.clone(),
                self.params.trapdoor_sigma.clone(),
                self.params.gadget_base.clone(),
                self.params.digit_count.clone(),
                self.params.preimage_max_coefficient_bound.clone(),
            )
        })?;

        let secret_epsilon = ternary_secret(&ring);
        let selector = Mat::concat(ConcatAxis::Columns, vec![secret_epsilon, message]);
        let base_public = bases.get_static(0).public_matrix();
        let initial_public_product_value = selector * base_public;
        let initial_error = ring.gaussian(
            (1, state_columns.clone()),
            self.params.error_sigma.clone(),
            self.params.error_max_coefficient_bound.clone(),
        );
        let p = initial_public_product_value + initial_error;

        let transition_count =
            IntExpr::Mul(Box::new(level_count.clone()), Box::new(digit_state_count.clone()))
                .canonicalize();
        let transition_indices = Parallel::range(transition_count.clone()).map_values(|slot| {
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
            level.mul(Int::evaluate(max_state_count.clone())).add(source_state)
        })?;
        let target_indices = Parallel::range(transition_count.clone()).map_values(|slot| {
            let flat = slot.as_int();
            let state = flat.clone().rem(Int::evaluate(max_state_count.clone()));
            let level = flat.div(Int::evaluate(digit_state_count.clone()));
            level.add(Int::constant(1)).mul(Int::evaluate(max_state_count.clone())).add(state)
        })?;
        let digit_secret_indices = Parallel::range(transition_count)
            .map_values(|slot| slot.as_int().div(Int::evaluate(max_state_count.clone())))?;
        let digit_secret_samples_family = Parallel::range(IntExpr::Mul(
            Box::new(level_count.clone()),
            Box::new(digit_base.clone()),
        ))
        .map_values(|_| ternary_secret(&ring))?;
        let digit_secrets = digit_secret_samples_family.parallel_gather(digit_secret_indices)?;
        let sources = bases.clone().parallel_gather(transition_indices)?;
        let target_public = bases.public_matrices().parallel_gather(target_indices)?;
        let sigma = self.params.error_sigma.clone();
        let error_bound = self.params.error_max_coefficient_bound.clone();
        let targets = parallel_zip_bundle_result(
            (digit_secrets, target_public),
            |slot, (secret, public)| {
                let flat = slot.as_int();
                let state = flat.clone().rem(Int::evaluate(max_state_count.clone()));
                let digit = flat
                    .clone()
                    .div(Int::evaluate(max_state_count.clone()))
                    .rem(Int::evaluate(digit_base.clone()));
                let level = flat.div(Int::evaluate(digit_state_count.clone()));
                let first_new = level.mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let regular = regular_selector(secret.clone());
                let k_identity = ring.identity(1);
                let k = Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), k_identity]);
                let initial_match = state.clone().equal(Int::constant(0)).to_int();
                let selector = initial_match.select(vec![regular, k])?;
                let selector = Sequential::range(batch_bits.clone()).scan(
                    selector,
                    (digit, (state, (first_new, secret))),
                    |bit, selector, (digit, (state, (first_new, secret)))| {
                        let extracted = digit.clone().bit(bit.expression());
                        let extracted_int = extracted.to_int();
                        let bit_zero_value = ring.zero((1, 1));
                        let bit_one_value = ring.identity(1);
                        let bit_value =
                            extracted_int.select(vec![bit_zero_value, bit_one_value])?;
                        let special_product = secret.clone() * bit_value;
                        let special_top =
                            Mat::concat(ConcatAxis::Columns, vec![secret, special_product]);
                        let special_bottom_value = ring.zero((1, 2));
                        let special =
                            Mat::concat(ConcatAxis::Rows, vec![special_top, special_bottom_value]);
                        let expected_state = first_new.add(bit.as_int());
                        let state_match_value = state.equal(expected_state);
                        let state_match_int = state_match_value.to_int();
                        state_match_int.select(vec![selector, special])
                    },
                )?;
                let selector_product_value = selector * public;
                let error = ring.gaussian(
                    (state_rows.clone(), state_columns.clone()),
                    sigma.clone(),
                    error_bound.clone(),
                );
                Ok::<_, DslError>(selector_product_value + error)
            },
        )?;
        let transitions = sources.parallel_zip_mat_values(targets, |_, source, target| {
            source.sample_preimage(target, (state_columns.clone(), state_columns.clone())).as_mat()
        })?;
        let final_indices = Parallel::range(max_state_count.clone()).map_values(|state| {
            Int::evaluate(IntExpr::Mul(
                Box::new(level_count.clone()),
                Box::new(max_state_count.clone()),
            ))
            .add(state.as_int())
        })?;
        let final_trapdoors = bases.parallel_gather(final_indices)?;
        Ok(DiamondInputPreprocessing { p, transitions, final_trapdoors })
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
        let initial = Parallel::range(max_state_count.clone()).map_values(|state| {
            let selector = state.as_int().equal(Int::constant(0)).to_int();
            let zero = self.params.ring().zero((1, self.params.state_columns()));
            selector.select(vec![zero, initial_state.clone()]).expect("matching state matrices")
        })?;
        let states = Sequential::range(level_count).scan(
            initial,
            (input_digits, transitions),
            |level, states, (input_digits, transitions)| {
                let level = level.as_int();
                let digit = input_digits.get(level.clone());
                let first_new =
                    level.clone().mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let source_indices =
                    Parallel::range(max_state_count.clone()).map_values(|state| {
                        let state = state.as_int();
                        first_new
                            .clone()
                            .less_equal(state.clone())
                            .to_int()
                            .select_int(vec![state, Int::constant(0)])
                            .expect("two integer branches")
                    })?;
                let source_states = states.parallel_gather(source_indices)?;
                let transition_indices =
                    Parallel::range(max_state_count.clone()).map_values(|state| {
                        level
                            .clone()
                            .mul(Int::evaluate(digit_state_count.clone()))
                            .add(digit.clone().mul(Int::evaluate(max_state_count.clone())))
                            .add(state.as_int())
                    })?;
                let selected = transitions.parallel_gather(transition_indices)?;
                parallel_zip_bundle_result((source_states, selected), |_, (state, transition)| {
                    Ok(state * transition)
                })
            },
        )?;
        Ok(DiamondInputEvaluation { states })
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
        assert_eq!(preprocessing.transitions.count(), &IntExpr::constant(12));
        assert_eq!(preprocessing.final_trapdoors.count(), &IntExpr::constant(3));

        let built = DslContext::new("diamond-input-preprocessing")
            .output("p", preprocessing.p)
            .unwrap()
            .output("transition", preprocessing.transitions.get_static(11))
            .unwrap()
            .build()
            .unwrap();
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
        let graph = DslContext::new("diamond-input-online")
            .output("default-state", evaluation.states.get_static(0))
            .unwrap()
            .output("last-state", evaluation.states.get_static(2))
            .unwrap()
            .build()
            .unwrap();
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
