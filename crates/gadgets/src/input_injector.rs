//! BGG-independent input-injection preprocessing shared by Diamond applications.

pub mod correctness;

use mxx_dsl::{DslError, Family, Int, Mat, Parallel, Ring, Trapdoor};
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
    /// Transition preimages indexed by input level, digit, and state.
    pub transitions: Vec<Vec<Vec<Mat>>>,
    /// Trapdoors for the final state bases, returned for application-specific projections.
    pub final_trapdoors: Vec<Trapdoor>,
}

/// Online result of applying the input-selected transition matrices.
///
/// `states[0]` is the default `(s, k)` state.  The remaining entries are the
/// bit-specific states in the same order returned by
/// [`DiamondInputConfig::bit_state_index`].
pub struct DiamondInputEvaluation {
    pub states: Vec<Mat>,
}

#[derive(Clone)]
pub struct DiamondInputInjector {
    pub config: DiamondInputConfig,
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
}

impl DiamondInputInjector {
    pub fn new(config: DiamondInputConfig) -> Result<Self, DiamondInputConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn preprocess(
        &self,
        message: Mat,
    ) -> Result<DiamondInputPreprocessing, DiamondInputPreprocessError> {
        let ring = self.config.ring();
        let state_rows = self.config.state_rows();
        let state_columns = self.config.state_columns()?;
        let mut bases = Vec::with_capacity(self.config.input_count + 1);
        for level in 0..=self.config.input_count {
            let state_count = self.config.state_count_at_level(level)?;
            bases.push(
                (0..state_count)
                    .map(|_| {
                        ring.sample_trapdoor(
                            state_rows,
                            self.config.trapdoor_sigma.clone(),
                            self.config.gadget_base_expr(),
                            self.config.digit_count_expr(),
                            self.config.preimage_max_coefficient_bound.clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let secret_epsilon = ternary_secret(&ring);
        let selector = Mat::concat(ConcatAxis::Columns, vec![secret_epsilon, message]);
        let p = selector * bases[0][0].public_matrix() +
            ring.gaussian(
                (1, state_columns),
                self.config.error_sigma.clone(),
                self.config.error_max_coefficient_bound.clone(),
            );

        let mut transitions = Vec::with_capacity(self.config.input_count);
        for level in 1..=self.config.input_count {
            let state_count = self.config.state_count_at_level(level)?;
            let first_new_state = 1 + (level - 1) * self.config.batch_bits;
            let digit_secrets =
                Parallel::range(self.config.digit_base).map(|_| ternary_secret(&ring))?;
            let mut state_transitions = Vec::with_capacity(state_count);
            for state in 0..state_count {
                let source_state = if state >= first_new_state { 0 } else { state };
                let source = bases[level - 1][source_state].clone();
                let public = bases[level][state].public_matrix();
                let ring = ring.clone();
                let sigma = self.config.error_sigma.clone();
                let error_bound = self.config.error_max_coefficient_bound.clone();
                let new_bit = (state >= first_new_state).then(|| state - first_new_state);
                let build_transition = move |secret_mask: Mat, bit: Option<Mat>| {
                    let selector = if let Some(bit) = bit {
                        special_selector(&ring, secret_mask, bit)
                    } else if state == 0 {
                        k_selector(&ring, secret_mask)
                    } else {
                        regular_selector(secret_mask)
                    };
                    let target = selector * public.clone() +
                        ring.gaussian(
                            (state_rows, state_columns),
                            sigma.clone(),
                            error_bound.clone(),
                        );
                    source.sample_preimage(target, (state_columns, state_columns)).as_mat()
                };
                let family = if let Some(bit_index) = new_bit {
                    let bits = mxx_dsl::Family::pack(
                        (0..self.config.digit_base)
                            .map(|digit| {
                                if ((digit >> bit_index) & 1) == 0 {
                                    self.config.ring().zero((1, 1))
                                } else {
                                    self.config.ring().identity(1)
                                }
                            })
                            .collect(),
                    )?;
                    digit_secrets
                        .clone()
                        .parallel_zip(bits, |_, secret, bit| build_transition(secret, Some(bit)))?
                } else {
                    digit_secrets
                        .clone()
                        .parallel_map(|_, secret| build_transition(secret, None))?
                };
                state_transitions.push(family);
            }
            let level_transitions = (0..self.config.digit_base)
                .map(|digit| {
                    state_transitions
                        .iter()
                        .map(|family| family.get_static(digit))
                        .collect::<Vec<_>>()
                })
                .collect();
            transitions.push(level_transitions);
        }

        let final_trapdoors =
            bases.pop().expect("Diamond input preprocessing always has a final level");
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
        input_digits: &[Int],
        transitions: &[Vec<Vec<Mat>>],
    ) -> Result<DiamondInputEvaluation, DiamondInputPreprocessError> {
        self.config.validate()?;
        if input_digits.len() != self.config.input_count ||
            transitions.len() != self.config.input_count
        {
            return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
        }

        let mut states = vec![initial_state];
        for level in 1..=self.config.input_count {
            let state_count = self.config.state_count_at_level(level)?;
            let level_transitions = &transitions[level - 1];
            if level_transitions.len() != self.config.digit_base ||
                level_transitions.iter().any(|branch| branch.len() != state_count)
            {
                return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
            }
            let first_new_state = 1 + (level - 1) * self.config.batch_bits;
            let source_states = Family::pack(
                (0..state_count)
                    .map(|state| {
                        let source = if state >= first_new_state { 0 } else { state };
                        states[source].clone()
                    })
                    .collect(),
            )?;
            let selected_transitions = Family::pack(
                (0..state_count)
                    .map(|state| {
                        input_digits[level - 1].clone().select(
                            (0..self.config.digit_base)
                                .map(|digit| level_transitions[digit][state].clone())
                                .collect(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            let next = source_states
                .parallel_zip(selected_transitions, move |_, state, transition| {
                    state * transition
                })?;
            states = (0..state_count).map(|state| next.get_static(state)).collect();
        }
        Ok(DiamondInputEvaluation { states })
    }
}

fn ternary_secret(ring: &Ring) -> Mat {
    ring.uniform_in((1, 1), -1, 1)
}

fn regular_selector(secret: Mat) -> Mat {
    Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), secret])
}

fn k_selector(ring: &Ring, secret: Mat) -> Mat {
    Mat::concat(ConcatAxis::Diagonal, vec![secret, ring.identity(1)])
}

fn special_selector(ring: &Ring, secret: Mat, bit: Mat) -> Mat {
    let top = Mat::concat(ConcatAxis::Columns, vec![secret.clone(), secret * bit]);
    let bottom = ring.zero((1, 2));
    Mat::concat(ConcatAxis::Rows, vec![top, bottom])
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
        let injector = DiamondInputInjector::new(config()).unwrap();
        let ring = injector.config.ring();
        let preprocessing =
            injector.preprocess(ring.input("message", (1, 1))).expect("preprocessing");
        assert_eq!(preprocessing.transitions.len(), 2);
        assert_eq!(preprocessing.transitions[0].len(), 2);
        assert_eq!(preprocessing.transitions[0][0].len(), 2);
        assert_eq!(preprocessing.transitions[1][0].len(), 3);
        assert_eq!(preprocessing.final_trapdoors.len(), 3);

        let built = DslContext::new("diamond-input-preprocessing")
            .output("p", preprocessing.p)
            .unwrap()
            .output("transition", preprocessing.transitions[1][1][2].clone())
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
        let injector = DiamondInputInjector::new(config()).unwrap();
        let ring = injector.config.ring();
        let preprocessing =
            injector.preprocess(ring.input("message", (1, 1))).expect("preprocessing");
        let digits = (0..injector.config.input_count)
            .map(|digit| ring.input(format!("digit-{digit}"), (1, 1)).extract_coefficient(0))
            .collect::<Vec<_>>();
        let evaluation = injector
            .evaluate(preprocessing.p, &digits, &preprocessing.transitions)
            .expect("online evaluation");
        let graph = DslContext::new("diamond-input-online")
            .output("default-state", evaluation.states[0].clone())
            .unwrap()
            .output("last-state", evaluation.states.last().unwrap().clone())
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
