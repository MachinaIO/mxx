//! BGG-independent input-injection preprocessing shared by Diamond applications.

use mxx_dsl::{DslError, Mat, Parallel, Ring, Trapdoor};
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
    #[error("a Diamond input-injection layout calculation overflowed")]
    LayoutOverflow,
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
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let secret_epsilon = ternary_secret(&ring);
        let selector = Mat::concat(ConcatAxis::Columns, vec![secret_epsilon, message]);
        let p = selector * bases[0][0].public_matrix() +
            ring.gaussian((1, state_columns), self.config.error_sigma.clone());

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
                        ring.gaussian((state_rows, state_columns), sigma.clone());
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
        let elaborated = built.elaborate(&ParamEnv::default()).unwrap();
        assert!(!elaborated.preimage_relations.is_empty());
    }
}
