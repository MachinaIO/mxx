//! BGG-independent input-injection preprocessing shared by Diamond applications.

use mxx_dsl::{DslError, Family, Int, Mat, Preimage, Ring, Trapdoor, iterate, parallel, select};
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
    pub transitions: Family<Preimage>,
    /// Trapdoors for the final state bases, returned for application-specific projections.
    pub final_trapdoors: Family<Trapdoor>,
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
        (&self.input_count * &self.batch_bits).canonicalize()
    }

    pub fn state_rows(&self) -> IntExpr {
        IntExpr::constant(DIAMOND_PREFIX_DIMENSION * DIAMOND_SECRET_DIMENSION)
    }

    pub fn state_columns(&self) -> IntExpr {
        (self.state_rows() * (&self.digit_count + IntExpr::constant(2))).canonicalize()
    }

    pub fn max_state_count(&self) -> IntExpr {
        (IntExpr::constant(1) + self.witness_size()).canonicalize()
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
        let digit_state_count = (&digit_base * &max_state_count).canonicalize();
        let base_count = ((&level_count + IntExpr::constant(1)) * &max_state_count).canonicalize();
        let bases = parallel(base_count, |_| {
            Ok(ring.sample_trapdoor(
                state_rows.clone(),
                self.params.trapdoor_sigma.clone(),
                self.params.gadget_base.clone(),
                self.params.digit_count.clone(),
                self.params.preimage_max_coefficient_bound.clone(),
            ))
        })?;

        let secret_epsilon = ternary_secret(&ring);
        let selector = Mat::concat(ConcatAxis::Columns, vec![secret_epsilon, message]);
        let base_public = bases.at(0).public_matrix();
        let initial_public_product_value = selector * base_public;
        let initial_error = ring.gaussian(
            (1, state_columns.clone()),
            self.params.error_sigma.clone(),
            self.params.error_max_coefficient_bound.clone(),
        );
        let p = initial_public_product_value + initial_error;

        let transition_count = (&level_count * &digit_state_count).canonicalize();
        // One secret belongs to each (level, digit), shared by all its state transitions.
        let digit_secrets = parallel(&level_count * &digit_base, |_| Ok(ternary_secret(&ring)))?;
        let sigma = self.params.error_sigma.clone();
        let error_bound = self.params.error_max_coefficient_bound.clone();
        let targets = parallel(transition_count.clone(), |flat| {
            let state = &flat % &max_state_count;
            let digit_index = &flat / &max_state_count;
            let digit = &digit_index % &digit_base;
            let level = flat / &digit_state_count;
            let secret = digit_secrets.at(digit_index);
            let target_index = (&level + 1) * &max_state_count + &state;
            let public = bases.at(target_index).public_matrix();
            let first_new = level * &batch_bits + 1;
            let regular = regular_selector(secret.clone());
            let k_identity = ring.identity(1);
            let k = Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), k_identity]);
            let initial_match = state.clone().equal(0);
            let selector = select(initial_match, vec![regular, k])?;
            let selector = iterate(batch_bits.clone(), selector, |bit, selector| {
                let extracted = digit.clone().bit(bit.clone())?;
                let bit_zero_value = ring.zero((1, 1));
                let bit_one_value = ring.identity(1);
                let bit_value = select(extracted, vec![bit_zero_value, bit_one_value])?;
                let special_product = &secret * bit_value;
                let special_top =
                    Mat::concat(ConcatAxis::Columns, vec![secret.clone(), special_product]);
                let special_bottom_value = ring.zero((1, 2));
                let special =
                    Mat::concat(ConcatAxis::Rows, vec![special_top, special_bottom_value]);
                let expected_state = &first_new + bit;
                let state_match = state.clone().equal(expected_state);
                select(state_match, vec![selector, special])
            })?;
            let selector_product_value = selector * public;
            let error = ring.gaussian(
                (state_rows.clone(), state_columns.clone()),
                sigma.clone(),
                error_bound.clone(),
            );
            Ok(selector_product_value + error)
        })?;
        let transitions = parallel(transition_count, |flat| {
            let state = &flat % &max_state_count;
            let level = &flat / &digit_state_count;
            let first_new = &level * &batch_bits + 1;
            let source_state = select(first_new.less_equal(&state), vec![state, Int::constant(0)])?;
            let source_index = level * &max_state_count + source_state;
            let source = bases.at(source_index);
            Ok(source
                .sample_preimage(targets.at(flat), (state_columns.clone(), state_columns.clone())))
        })?;
        let final_trapdoors = parallel(max_state_count.clone(), |state| {
            let index = Int::evaluate(&level_count * &max_state_count) + state;
            Ok(bases.at(index))
        })?;
        Ok(DiamondInputPreprocessing { p, transitions, final_trapdoors })
    }

    /// Applies the preprocessed transition matrices to one packed input.
    ///
    /// The transition layout is exactly the one returned by [`Self::preprocess`]:
    /// `[level][digit][state]`. Each level reads the selected transitions by index and
    /// updates all independent states in one parallel loop.
    pub fn evaluate(
        &self,
        initial_state: Mat,
        input_digits: Family<Int>,
        transitions: Family<Preimage>,
    ) -> Result<DiamondInputEvaluation, DiamondInputPreprocessError> {
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        let digit_state_count = (&digit_base * &max_state_count).canonicalize();
        let expected_transitions = (&level_count * &digit_state_count).canonicalize();
        if input_digits.count().canonicalize() != level_count.canonicalize() ||
            transitions.count().canonicalize() != expected_transitions
        {
            return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
        }
        let initial = parallel(max_state_count.clone(), |state| {
            let selector = state.equal(0);
            let zero = self.params.ring().zero((1, self.params.state_columns()));
            select(selector, vec![zero, initial_state.clone()])
        })?;
        let states = iterate(level_count, initial, |level, states| {
            let digit = input_digits.at(&level);
            let first_new = &level * &batch_bits + 1;
            parallel(max_state_count.clone(), |state| {
                let source_index = select(
                    first_new.clone().less_equal(&state),
                    vec![state.clone(), Int::constant(0)],
                )?;
                let transition_index =
                    &level * &digit_state_count + &digit * &max_state_count + state;
                let transition = transitions.at(transition_index);
                Ok(transition.mul_small_rhs(states.at(source_index)))
            })
        })?;
        Ok(DiamondInputEvaluation { states })
    }
}

fn ternary_secret(ring: &Ring) -> Mat {
    ring.uniform_interval((1, 1), -1, 1)
}

fn regular_selector(secret: Mat) -> Mat {
    Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), secret])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, node::NodeKind, types::WireType};

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
        let transition = preprocessing.transitions.at(11);
        assert!(matches!(transition.value_handle().wire_type(), WireType::Preimage { .. }));
        let transition_product = transition.clone().mul_small_rhs(preprocessing.p.clone());

        let built = DslContext::new("diamond-input-preprocessing")
            .output("p", preprocessing.p)
            .unwrap()
            .output("transition", transition)
            .unwrap()
            .output("transition-product", transition_product)
            .unwrap()
            .build()
            .unwrap();
        let validated = built.validate(&ParamEnv::default()).unwrap();
        let nodes =
            validated.source.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::PreimageSample { .. })));
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)));
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::MatrixScale { .. })));
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
            .output("default-state", evaluation.states.at(0))
            .unwrap()
            .output("last-state", evaluation.states.at(2))
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
