//! BGG-independent input-injection preprocessing shared by Diamond applications.

use mxx_dsl::{
    DslError, Family, FamilyAxisSelection, Int, Mat, Parallel, Ring, Sequential, TrapdoorFamily,
    parallel_zip_bundle,
};
use mxx_ir_core::{IndexMap, IntExpr, RealExpr, expr::IndexExpr, node::ConcatAxis};
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
    /// Initial input-injection state family indexed by `state`.
    pub initial: Family<Mat>,
    /// Rectangular transition preimages indexed by `(level, state, digit)`.
    pub transitions: Family<mxx_dsl::Preimage>,
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

        let bases = bases.reindex(
            vec![
                IntExpr::Add(Box::new(level_count.clone()), Box::new(IntExpr::constant(1))),
                max_state_count.clone(),
            ],
            IndexMap::new([IndexExpr::Add(
                Box::new(IndexExpr::Multiply(
                    Box::new(IndexExpr::Axis(0)),
                    Box::new(IndexExpr::try_from(max_state_count.clone()).expect("state count")),
                )),
                Box::new(IndexExpr::Axis(1)),
            )]),
        )?;
        let source_state = IndexExpr::Select {
            selector: Box::new(IndexExpr::LessEqual(
                Box::new(IndexExpr::Add(
                    Box::new(IndexExpr::Multiply(
                        Box::new(IndexExpr::Axis(0)),
                        Box::new(IndexExpr::try_from(batch_bits.clone()).expect("batch bits")),
                    )),
                    Box::new(IndexExpr::Constant(1.into())),
                )),
                Box::new(IndexExpr::Axis(1)),
            )),
            branches: vec![IndexExpr::Axis(1), IndexExpr::Constant(0.into())],
        };
        let source_map = IndexMap::new([IndexExpr::Axis(0), source_state]);
        let group_trapdoors = bases
            .clone()
            .reindex(vec![level_count.clone(), max_state_count.clone()], source_map)?;
        let digit_secrets = Parallel::range(IntExpr::Mul(
            Box::new(level_count.clone()),
            Box::new(digit_base.clone()),
        ))
        .map_values(|_| ternary_secret(&ring))?;
        let sigma = self.params.error_sigma.clone();
        let error_bound = self.params.error_max_coefficient_bound.clone();
        let target_count = IntExpr::Mul(
            Box::new(IntExpr::Mul(
                Box::new(level_count.clone()),
                Box::new(max_state_count.clone()),
            )),
            Box::new(digit_base.clone()),
        );
        let state_digit_count = IndexExpr::Multiply(
            Box::new(IndexExpr::try_from(max_state_count.clone()).expect("state count")),
            Box::new(IndexExpr::try_from(digit_base.clone()).expect("digit base")),
        );
        let flat = IndexExpr::Axis(0);
        let target_public = bases.public_matrices().reindex(
            vec![target_count],
            IndexMap::new([
                IndexExpr::Add(
                    Box::new(IndexExpr::Divide(
                        Box::new(flat.clone()),
                        Box::new(state_digit_count.clone()),
                    )),
                    Box::new(IndexExpr::constant(1)),
                ),
                IndexExpr::Divide(
                    Box::new(IndexExpr::Remainder(Box::new(flat), Box::new(state_digit_count))),
                    Box::new(IndexExpr::try_from(digit_base.clone()).expect("digit base")),
                ),
            ]),
        )?;
        let target_state_count = max_state_count.clone();
        let target_digit_base = digit_base.clone();
        let targets = target_public
            .parallel_map_values(|flat, public| {
                let flat = flat.as_int();
                let state_digit_count = Int::evaluate(IntExpr::Mul(
                    Box::new(target_state_count.clone()),
                    Box::new(target_digit_base.clone()),
                ));
                let level = flat.clone().div(state_digit_count.clone());
                let within_level = flat.rem(state_digit_count);
                let digit_base = Int::evaluate(target_digit_base.clone());
                let state = within_level.clone().div(digit_base.clone());
                let digit = within_level.rem(digit_base.clone());
                let secret_index = level.clone().mul(digit_base).add(digit.clone());
                let secret = digit_secrets.get(secret_index);
                let first_new = level.mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let regular = regular_selector(secret.clone());
                let k_identity = ring.identity(1);
                // The selected carrier s is either the regular diagonal
                // secret or the special transition carrier.  The appended
                // identity slot is the k coordinate used by the input state.
                let k = Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), k_identity]);
                let initial_match = state.clone().equal(Int::constant(0)).to_int();
                let selector =
                    initial_match.select(vec![regular, k]).expect("matching matrix branches");
                let selector = Sequential::range(batch_bits.clone())
                    .scan(
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
                            let special = Mat::concat(
                                ConcatAxis::Rows,
                                vec![special_top, special_bottom_value],
                            );
                            let expected_state = first_new.add(bit.as_int());
                            let state_match_value = state.equal(expected_state);
                            let state_match_int = state_match_value.to_int();
                            state_match_int.select(vec![selector, special])
                        },
                    )
                    .expect("selector scan");
                // For a sampled public base B and selector s, this is
                // b = s * B + e_b.  Its trapdoor preimage K satisfies
                // B * K = P + E, hence b * K = s * P + s * E + e_b * K;
                // the same equation covers regular, first-new, and special
                // bit transitions selected above.
                let selector_product_value = selector * public;
                let error = ring.gaussian(
                    (state_rows.clone(), state_columns.clone()),
                    sigma.clone(),
                    error_bound.clone(),
                );
                selector_product_value + error
            })?
            .reindex(
                vec![level_count.clone(), max_state_count.clone(), digit_base.clone()],
                IndexMap::new([IndexExpr::Add(
                    Box::new(IndexExpr::Multiply(
                        Box::new(IndexExpr::Add(
                            Box::new(IndexExpr::Multiply(
                                Box::new(IndexExpr::Axis(0)),
                                Box::new(
                                    IndexExpr::try_from(max_state_count.clone())
                                        .expect("state count"),
                                ),
                            )),
                            Box::new(IndexExpr::Axis(1)),
                        )),
                        Box::new(IndexExpr::try_from(digit_base.clone()).expect("digit base")),
                    )),
                    Box::new(IndexExpr::Axis(2)),
                )]),
            )?;
        let transitions = group_trapdoors
            .sample_preimage_branches(targets, (state_columns.clone(), state_columns.clone()))?;
        let initial_public = bases.public_matrices().reindex(
            vec![max_state_count.clone()],
            IndexMap::new([IndexExpr::constant(0), IndexExpr::Axis(0)]),
        )?;
        let initial = initial_public.parallel_map_values(|state, public| {
            let state = state.as_int();
            let secret = ternary_secret(&ring);
            // The carrier formula is b_state = c_state * B_0 + e_state with
            // c_state = [s | m].  Since is_initial is 1 exactly when state=0,
            // the [zero, c_state] and [zero, e_sample] branches select
            // b_0 = [s | m] * B_0 + e_sample at state zero; every other state
            // selects both zero branches and therefore remains exactly zero.
            let carrier = Mat::concat(ConcatAxis::Columns, vec![secret, message.clone()]);
            let is_initial = state.clone().equal(Int::constant(0)).to_int();
            let zero_carrier = Mat::concat(
                ConcatAxis::Columns,
                vec![ring.zero((1, DIAMOND_SECRET_DIMENSION)), ring.zero((1, 1))],
            );
            let carrier = is_initial
                .clone()
                .select(vec![zero_carrier, carrier])
                .expect("initial carrier branches");
            let sampled_error =
                ring.gaussian((1, state_columns.clone()), sigma.clone(), error_bound.clone());
            let error = is_initial
                .select(vec![ring.zero((1, state_columns.clone())), sampled_error])
                .expect("initial error branches");
            carrier * public + error
        })?;
        // The final trapdoor family is retained for application projections;
        // its public relation is still B*K = P+E, so later multiplication by
        // a selected state realizes the same s*P+s*E+e_b*K equation.
        let final_trapdoors = bases.reindex(
            vec![max_state_count.clone()],
            IndexMap::new([
                IndexExpr::try_from(level_count.clone()).expect("level count index"),
                IndexExpr::Axis(0),
            ]),
        )?;
        Ok(DiamondInputPreprocessing { initial, transitions, final_trapdoors })
    }

    /// Applies the preprocessed transition matrices to one packed input.
    ///
    /// The transition layout is exactly the one returned by [`Self::preprocess`]:
    /// `[level][state][digit]`. Selection is represented by the DSL `Select`
    /// node and all independent state transitions at a level are represented by
    /// one rank-one IR parallel grid.
    pub fn evaluate(
        &self,
        initial_state: Family<Mat>,
        input_digits: Family<Int>,
        transitions: Family<mxx_dsl::Preimage>,
    ) -> Result<DiamondInputEvaluation, DiamondInputPreprocessError> {
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        if input_digits.count().canonicalize() != level_count.canonicalize() ||
            transitions.shape() !=
                &[level_count.clone(), max_state_count.clone(), digit_base.clone()]
        {
            return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
        }
        let states = Sequential::range(level_count).scan(
            initial_state,
            (input_digits, transitions),
            |level, states, (input_digits, transitions)| {
                let level = level.as_int();
                let digit = input_digits.get(level.clone());
                let source_state = IndexExpr::Select {
                    selector: Box::new(IndexExpr::LessEqual(
                        Box::new(IndexExpr::Add(
                            Box::new(IndexExpr::Multiply(
                                Box::new(IndexExpr::LoopIndex(0)),
                                Box::new(
                                    IndexExpr::try_from(batch_bits.clone()).expect("batch bits"),
                                ),
                            )),
                            Box::new(IndexExpr::Constant(1.into())),
                        )),
                        Box::new(IndexExpr::Axis(0)),
                    )),
                    branches: vec![IndexExpr::Axis(0), IndexExpr::Constant(0.into())],
                };
                let source_states =
                    states.reindex(vec![max_state_count.clone()], IndexMap::new([source_state]))?;
                let level_transitions = transitions.clone().reindex(
                    vec![max_state_count.clone(), digit_base.clone()],
                    IndexMap::new([
                        IndexExpr::LoopIndex(0),
                        IndexExpr::Axis(0),
                        IndexExpr::Axis(1),
                    ]),
                )?;
                let selected_transitions = match level_transitions.select_axis(1, digit)? {
                    FamilyAxisSelection::Family(family) => family,
                    FamilyAxisSelection::Scalar(_) => return Err(DslError::Schema),
                };
                // Selecting one digit chooses the corresponding preimage K;
                // applying it explicitly computes state * K and consumes the
                // transition relation rather than treating K as metadata.
                parallel_zip_bundle(
                    (source_states, selected_transitions),
                    |_, (source, transition)| source.mul_small_rhs(transition),
                )
            },
        )?;
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
        assert_eq!(
            preprocessing.transitions.shape(),
            &[IntExpr::constant(2), IntExpr::constant(3), IntExpr::constant(2)]
        );
        assert_eq!(preprocessing.final_trapdoors.count(), &IntExpr::constant(3));

        let built = DslContext::new("diamond-input-preprocessing")
            .output("p", preprocessing.initial.get_static(vec![IndexExpr::Constant(0.into())]))
            .unwrap()
            .preimage_output(
                "transition",
                preprocessing.transitions.get_static(vec![
                    IndexExpr::Constant(1.into()),
                    IndexExpr::Constant(1.into()),
                    IndexExpr::Constant(1.into()),
                ]),
            )
            .unwrap()
            .output(
                "final-public",
                preprocessing
                    .final_trapdoors
                    .public_matrices()
                    .get_static(vec![IndexExpr::Constant(0.into())]),
            )
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
                .any(|node| matches!(node.kind(), NodeKind::FamilyPreimageSample { .. }))
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
            .evaluate(preprocessing.initial, digits, preprocessing.transitions)
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
                .any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
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
