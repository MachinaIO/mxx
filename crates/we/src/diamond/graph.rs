use super::{DiamondArtifactNames, DiamondConfigError, DiamondWeConfig};
use mxx_bgg::{
    BggEncodingCompiler, BggEncodingFamily, BggEncodingWire, BggPublicKeyCompiler,
    BggPublicKeyFamily, BggPublicKeySampler, BggPublicKeyWire, BggSamplerLayout,
    DynamicBooleanBggError, evaluate_boolean_encoding_layers, evaluate_boolean_public_key_layers,
};
use mxx_dsl::{
    Bool, BuiltGraph, DslContext, DslError, Int, Mat, Parallel, parallel_zip_bundle_result,
};
use mxx_gadgets::{
    circuit::{
        BOOLEAN_INSTANCE_INPUT, BOOLEAN_WITNESS_INPUT, BooleanCircuitError,
        BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanCircuitShape,
    },
    input_injector::{DiamondInputInjector, DiamondInputParams, DiamondInputPreprocessError},
};
use mxx_ir_core::{
    IndexExpr, IndexMap, IntExpr, ParamEnv,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, IndexRange},
};
use thiserror::Error;

pub const HASH_KEY_INPUT: &str = "diamond-hash-key";
pub const MESSAGE_INPUT: &str = "diamond-message";
pub const DECODED_OUTPUT: &str = "diamond-decoded";
pub const NOISY_PLAINTEXT_OUTPUT: &str = "diamond-noisy-plaintext";

#[derive(Clone)]
struct DiamondGraphParams {
    input: DiamondInputParams,
}

impl DiamondGraphParams {
    const MODULUS: &'static str = "diamond_modulus";
    const RING_DIMENSION: &'static str = "diamond_ring_dimension";
    const INPUT_COUNT: &'static str = "diamond_input_count";
    const DIGIT_BASE: &'static str = "diamond_digit_base";
    const BATCH_BITS: &'static str = "diamond_batch_bits";
    const GADGET_BASE: &'static str = "diamond_gadget_base";
    const DIGIT_COUNT: &'static str = "diamond_digit_count";
    const TRAPDOOR_SIGMA: &'static str = "diamond_trapdoor_sigma";
    const ERROR_SIGMA: &'static str = "diamond_error_sigma";
    const ERROR_BOUND: &'static str = "diamond_error_max_coefficient_bound";
    const PREIMAGE_BOUND: &'static str = "diamond_preimage_max_coefficient_bound";

    fn declare(mut context: DslContext) -> (DslContext, Self) {
        for name in [
            Self::MODULUS,
            Self::RING_DIMENSION,
            Self::INPUT_COUNT,
            Self::DIGIT_BASE,
            Self::BATCH_BITS,
            Self::GADGET_BASE,
            Self::DIGIT_COUNT,
            Self::ERROR_BOUND,
            Self::PREIMAGE_BOUND,
        ] {
            context = context.int_parameter(name);
        }
        context = context.real_parameter(Self::TRAPDOOR_SIGMA);
        context = context.real_parameter(Self::ERROR_SIGMA);
        let var = |name: &str| mxx_ir_core::IntExpr::Var(name.to_owned());
        (
            context,
            Self {
                input: DiamondInputParams {
                    modulus: var(Self::MODULUS),
                    ring_dimension: var(Self::RING_DIMENSION),
                    input_count: var(Self::INPUT_COUNT),
                    digit_base: var(Self::DIGIT_BASE),
                    batch_bits: var(Self::BATCH_BITS),
                    gadget_base: var(Self::GADGET_BASE),
                    digit_count: var(Self::DIGIT_COUNT),
                    trapdoor_sigma: mxx_ir_core::RealExpr::Var(Self::TRAPDOOR_SIGMA.to_owned()),
                    error_sigma: mxx_ir_core::RealExpr::Var(Self::ERROR_SIGMA.to_owned()),
                    error_max_coefficient_bound: var(Self::ERROR_BOUND),
                    preimage_max_coefficient_bound: var(Self::PREIMAGE_BOUND),
                },
            },
        )
    }
}

fn padded_witness_public_key_indices(
    instance_width: Int,
    witness_size: IntExpr,
    max_layer_width: IntExpr,
) -> Result<mxx_dsl::Family<Int>, DslError> {
    let witness_end = instance_width.clone().add(Int::evaluate(witness_size).sub(Int::constant(1)));
    Parallel::range(max_layer_width).map_values(move |slot| {
        let slot = slot.as_int();
        let after_instance = instance_width.clone().less_equal(slot.clone()).to_int();
        let before_end = slot.clone().less_equal(witness_end.clone()).to_int();
        let output = after_instance
            .mul(before_end)
            .select_int(vec![
                Int::constant(0),
                slot.sub(instance_width.clone()).add(Int::constant(1)),
            ])
            .expect("two public-key indices");
        output
    })
}

pub struct DiamondEncryptionGraph {
    pub graph: BuiltGraph,
}

pub struct DiamondDecryptionGraph {
    pub graph: BuiltGraph,
}

struct DiamondEncryptionBuild {
    graph: DiamondEncryptionGraph,
}

struct DiamondDecryptionBuild {
    graph: DiamondDecryptionGraph,
}
#[derive(Clone, Debug)]
pub struct DiamondWeCompiler {
    pub config: DiamondWeConfig,
    pub shape: BooleanCircuitShape,
}

/// The parameter-independent Diamond WE protocol family.
///
/// All circuit dimensions and cryptographic values are declared as symbolic Graph IR parameters.
/// The tag is the sole fixed domain-separation value committed to by this declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondWeProtocolFamily {
    bgg_tag: Vec<u8>,
}

impl DiamondWeProtocolFamily {
    pub fn new(bgg_tag: impl Into<Vec<u8>>) -> Self {
        Self { bgg_tag: bgg_tag.into() }
    }
}

#[derive(Debug, Error)]
pub enum DiamondCompileError {
    #[error(transparent)]
    Config(#[from] DiamondConfigError),
    #[error(transparent)]
    Shape(#[from] BooleanCircuitError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    Bgg(#[from] DynamicBooleanBggError),
    #[error(transparent)]
    Input(#[from] DiamondInputPreprocessError),
    #[error("the fixed Boolean witness width does not match the Diamond input layout")]
    WitnessWidth,
    #[error("the Boolean circuit maximum layer width is smaller than its input width")]
    FamilyWidth,
    #[error("Diamond parameter expression evaluation failed: {0}")]
    ParameterExpression(String),
}

impl DiamondWeCompiler {
    pub fn new(
        config: DiamondWeConfig,
        shape: BooleanCircuitShape,
    ) -> Result<Self, DiamondCompileError> {
        config.validate()?;
        shape.validate()?;
        if config.witness_size()? != shape.witness_width {
            return Err(DiamondCompileError::WitnessWidth);
        }
        if shape.analyze()?.maximum_layer_width < shape.input_width()? {
            return Err(DiamondCompileError::FamilyWidth);
        }
        Ok(Self { config, shape })
    }

    pub fn circuit_bindings(&self) -> Result<ParamEnv, DiamondCompileError> {
        let analysis = self.shape.analyze()?;
        Ok(ParamEnv {
            integers: [
                (
                    BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(),
                    self.shape.instance_width.into(),
                ),
                (
                    BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(),
                    self.shape.witness_width.into(),
                ),
                (BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(), analysis.depth.into()),
                (
                    BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(),
                    analysis.maximum_layer_width.into(),
                ),
                (DiamondGraphParams::MODULUS.to_owned(), self.config.modulus.clone()),
                (DiamondGraphParams::RING_DIMENSION.to_owned(), self.config.ring_dimension.into()),
                (DiamondGraphParams::INPUT_COUNT.to_owned(), self.config.input_count.into()),
                (DiamondGraphParams::DIGIT_BASE.to_owned(), self.config.digit_base.into()),
                (DiamondGraphParams::BATCH_BITS.to_owned(), self.config.batch_bits.into()),
                (DiamondGraphParams::GADGET_BASE.to_owned(), self.config.gadget_base.clone()),
                (DiamondGraphParams::DIGIT_COUNT.to_owned(), self.config.digit_count.into()),
                (
                    DiamondGraphParams::ERROR_BOUND.to_owned(),
                    self.config.error_max_coefficient_bound.clone(),
                ),
                (
                    DiamondGraphParams::PREIMAGE_BOUND.to_owned(),
                    self.config.preimage_max_coefficient_bound.clone(),
                ),
            ]
            .into_iter()
            .collect(),
            reals: [
                (
                    DiamondGraphParams::TRAPDOOR_SIGMA.to_owned(),
                    self.config.trapdoor_sigma.evaluate_rational(&ParamEnv::default()).map_err(
                        |error| DiamondCompileError::ParameterExpression(error.to_string()),
                    )?,
                ),
                (
                    DiamondGraphParams::ERROR_SIGMA.to_owned(),
                    self.config.error_sigma.evaluate_rational(&ParamEnv::default()).map_err(
                        |error| DiamondCompileError::ParameterExpression(error.to_string()),
                    )?,
                ),
            ]
            .into_iter()
            .collect(),
            ..ParamEnv::default()
        })
    }

    pub fn build_encryption(&self) -> Result<DiamondEncryptionGraph, DiamondCompileError> {
        self.config.validate()?;
        self.shape.validate()?;
        Ok(DiamondWeProtocolFamily::new(self.config.bgg_tag.clone()).build_encryption()?.graph)
    }

    pub fn build_decryption(
        &self,
        encryption: ProductionId,
    ) -> Result<DiamondDecryptionGraph, DiamondCompileError> {
        self.config.validate()?;
        self.shape.validate()?;
        Ok(DiamondWeProtocolFamily::new(self.config.bgg_tag.clone())
            .build_decryption(encryption)?
            .graph)
    }
}

impl DiamondWeProtocolFamily {
    fn build_encryption(&self) -> Result<DiamondEncryptionBuild, DiamondCompileError> {
        let (context, circuit_params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("diamond-we-encryption"));
        let (context, graph_params) = DiamondGraphParams::declare(context);
        let ring = graph_params.input.ring();
        let circuit_data = BooleanCircuitFamilyInputs::protocol_inputs(&context, &circuit_params);
        let instance = context
            .int_family_input(BOOLEAN_INSTANCE_INPUT, circuit_params.max_layer_width.clone());
        let message_input = ring.bool_input(MESSAGE_INPUT);
        let message_int = message_input.to_int();
        let message_zero = ring.zero((1, 1));
        let message_one = ring.identity(1);
        let message = message_int.select(vec![message_zero, message_one])?;
        let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
        let input_preprocessing =
            DiamondInputInjector::parameterized(graph_params.input.clone()).preprocess(message)?;
        let public_key_compiler = Self::public_key_compiler(&graph_params);
        let witness_size = graph_params.input.witness_size();
        let public_keys = BggPublicKeySampler { layout: Self::sampler_layout(&graph_params) }
            .sample_family(
                hash_key.clone(),
                self.tag(b":witness_public_keys"),
                mxx_ir_core::IntExpr::Add(
                    Box::new(witness_size.clone()),
                    Box::new(mxx_ir_core::IntExpr::constant(1)),
                ),
                graph_params.input.digit_count.clone(),
            )?;
        let one_public_key =
            BggPublicKeyWire { matrix: public_keys.matrices.get_static(0), reveal_plaintext: true };
        let zero_public_key = public_key_compiler.sub(&one_public_key, &one_public_key);
        let instance_width = context.evaluate_int(circuit_params.instance_width.clone());
        let witness_end =
            instance_width.clone().add(Int::evaluate(witness_size.clone()).sub(Int::constant(1)));
        let public_indices = padded_witness_public_key_indices(
            instance_width.clone(),
            witness_size.clone(),
            circuit_params.max_layer_width.clone(),
        )?;
        let public_candidates = public_keys.matrices.clone().parallel_gather(public_indices)?;
        let packed_inputs = public_candidates.parallel_map_values({
            let zero = zero_public_key.matrix.clone();
            let instance_width = instance_width.clone();
            move |slot, candidate| {
                let slot = slot.as_int();
                let after_instance = instance_width.clone().less_equal(slot.clone()).to_int();
                let before_end = slot.less_equal(witness_end.clone()).to_int();
                let in_range = before_end
                    .select(vec![zero.clone(), candidate])
                    .expect("matching public-key types");
                after_instance
                    .select(vec![zero.clone(), in_range])
                    .expect("matching public-key types")
            }
        })?;
        let circuit_input_matrices =
            parallel_zip_bundle_result((instance, packed_inputs), |slot, (bit, packed)| {
                let index = slot.as_int();
                let selected_instance = bit
                    .select(vec![zero_public_key.matrix.clone(), one_public_key.matrix.clone()])
                    .expect("matching public-key matrix types");
                let active =
                    index.clone().less_equal(instance_width.clone().sub(Int::constant(1))).to_int();
                let selected_source = active.select(vec![packed, selected_instance])?;
                Ok::<_, DslError>(selected_source)
            })?;
        let circuit_inputs =
            BggPublicKeyFamily { matrices: circuit_input_matrices, reveal_plaintext: true };
        let circuit_output_family = evaluate_boolean_public_key_layers(
            &context,
            &circuit_params,
            circuit_data.clone(),
            circuit_inputs,
            one_public_key.clone(),
            public_key_compiler.clone(),
        )?;
        let circuit_output_index = circuit_data.output_source();
        let circuit_output_matrix =
            circuit_output_family.matrices.get(circuit_output_index.clone());
        let circuit_output =
            BggPublicKeyWire { matrix: circuit_output_matrix, reveal_plaintext: true };

        let gadget = ring.gadget(
            1,
            graph_params.input.gadget_base.clone(),
            graph_params.input.digit_count.clone(),
        );
        let public_columns = graph_params.input.digit_count.clone();
        let state_columns = graph_params.input.state_columns();
        let scalar_zero = ring.zero((1, 1));
        let scalar_one = ring.identity(1);
        let top_row = Mat::concat(ConcatAxis::Rows, vec![scalar_one.clone(), scalar_zero.clone()]);
        let bottom_row = Mat::concat(ConcatAxis::Rows, vec![scalar_zero, scalar_one]);
        // These two rows retain the full two-coordinate source basis.  A
        // projection target must therefore keep both source matrices visible:
        // the top coordinate carries the public key and the bottom coordinate
        // carries the gadget correction.
        let one_target =
            top_row.clone() * one_public_key.matrix.clone() - top_row.clone() * gadget.clone();
        // The one preimage is sampled for T_one = top*(A_one - G), so applying
        // it later to the initial state consumes exactly this target relation.
        let projection_trapdoor = input_preprocessing.final_trapdoors.get_static(0);
        let one_trapdoor = projection_trapdoor.clone();
        let one_sample = one_trapdoor
            .sample_preimage(one_target, (state_columns.clone(), public_columns.clone()));
        let one_preimage = one_sample;
        let witness_source_map = IndexMap::new([IndexExpr::Add(
            Box::new(IndexExpr::Axis(0)),
            Box::new(IndexExpr::constant(1)),
        )]);
        let witness_trapdoors = input_preprocessing
            .final_trapdoors
            .clone()
            .reindex(vec![witness_size.clone()], witness_source_map.clone())?;
        let witness_public_keys =
            public_keys.matrices.clone().reindex(vec![witness_size.clone()], witness_source_map)?;
        let witness_targets = witness_public_keys.parallel_map_values({
            let top_row = top_row.clone();
            let bottom_row = bottom_row.clone();
            let gadget = gadget.clone();
            // Each witness target is T_w = top*A_w - bottom*G.  Both source
            // rows remain in the target because the final injector state has
            // two coordinates, even though only one public column is selected.
            move |_, public_key| top_row.clone() * public_key - bottom_row.clone() * gadget.clone()
        })?;
        let witness_preimages =
            witness_trapdoors.parallel_zip_mat_values(witness_targets, |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (state_columns.clone(), public_columns.clone()))
            })?;

        let k_public_key_matrix = ring.hash_matrix(
            hash_key.clone(),
            self.tag(b":k_public_key"),
            (1, public_columns.clone()),
        );
        let k_public_key =
            BggPublicKeyWire { matrix: k_public_key_matrix, reveal_plaintext: false };
        let first_column = Some(IndexRange { start: 0.into(), end: 1.into() });
        let k_public_key_first = k_public_key.matrix.clone().slice(None, first_column.clone());
        // The decoder subtracts the K encoding.  Sampling K against ceil(q / 2) therefore
        // leaves the canonical Boolean-one center floor(q / 2) modulo q for both even and odd q.
        let half_modulus = IntExpr::RoundDiv(
            Box::new(graph_params.input.modulus.clone()),
            Box::new(mxx_ir_core::IntExpr::constant(2)),
        );
        let half_modulus_polynomial = ring.polynomial([half_modulus.into()]);
        // K is centered at ceil(q/2) in the bottom source coordinate:
        // T_K = top*K_first + bottom*floor(q/2).  The large scalar is thus a
        // scalar times the gadget-shaped coordinate, not an arbitrary target.
        let k_target = top_row.clone() * k_public_key_first.clone() +
            bottom_row.clone() * half_modulus_polynomial;
        let k_trapdoor = projection_trapdoor.clone();
        let k_sample = k_trapdoor.sample_preimage(k_target, (state_columns.clone(), 1));
        let k_preimage = k_sample;
        let r = ring.hash_matrix(hash_key, self.tag(b":r"), (1, public_columns.clone()));
        let r_column = r.slice(None, first_column);
        let r_decomposition = r_column.decompose(
            graph_params.input.gadget_base.clone(),
            graph_params.input.digit_count.clone(),
        );
        let difference = public_key_compiler.sub(&one_public_key, &circuit_output);
        // R is consumed through its explicit decomposition.  The residual
        // public target is T_dec = K_first + (A_one-A_circuit)R, retaining the
        // source matrix product before the decoder preimage is sampled.
        let projected_difference = difference.matrix *
            r_decomposition.clone().into_preimage_relation().materialize_exact();
        let decoder_public_key = k_public_key_first + projected_difference;
        let decoder_target = top_row * decoder_public_key;
        let decoder_trapdoor = projection_trapdoor;
        let decoder_sample = decoder_trapdoor.sample_preimage(decoder_target, (state_columns, 1));
        let decoder_preimage = decoder_sample;

        let graph = context
            .public_family_output(DiamondArtifactNames::INITIAL_STATE, input_preprocessing.initial)?
            .public_preimage_output(DiamondArtifactNames::ONE_PREIMAGE, one_preimage)?
            .public_preimage_output(DiamondArtifactNames::K_PREIMAGE, k_preimage)?
            .public_preimage_output(DiamondArtifactNames::DECODER_PREIMAGE, decoder_preimage)?
            .public_preimage_output(
                DiamondArtifactNames::R_DECOMPOSED,
                r_decomposition.into_preimage_relation(),
            )?
            .public_family_output(DiamondArtifactNames::PUBLIC_KEYS, public_keys.matrices)?
            .public_preimage_family_output(
                DiamondArtifactNames::TRANSITIONS,
                input_preprocessing.transitions,
            )?
            .public_preimage_family_output(
                DiamondArtifactNames::WITNESS_PREIMAGES,
                witness_preimages,
            )?
            .build()?;
        Ok(DiamondEncryptionBuild { graph: DiamondEncryptionGraph { graph } })
    }

    fn build_decryption(
        &self,
        encryption: ProductionId,
    ) -> Result<DiamondDecryptionBuild, DiamondCompileError> {
        let (context, circuit_params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("diamond-we-decryption"));
        let (context, graph_params) = DiamondGraphParams::declare(context);
        let ring = graph_params.input.ring();
        let max_state_count = graph_params.input.max_state_count();
        let circuit_data = BooleanCircuitFamilyInputs::protocol_inputs(&context, &circuit_params);
        let instance = context
            .int_family_input(BOOLEAN_INSTANCE_INPUT, circuit_params.max_layer_width.clone());
        let state_columns = graph_params.input.state_columns();
        let public_columns = graph_params.input.digit_count.clone();
        let initial_state = ring.family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::INITIAL_STATE,
            max_state_count.clone(),
            (1, state_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let witness =
            context.int_family_input(BOOLEAN_WITNESS_INPUT, circuit_params.max_layer_width.clone());
        let witness_size = graph_params.input.witness_size();
        let witness_indices = Parallel::range(witness_size.clone())
            .map_values(|bit| bit.as_int().add(Int::constant(0)))?;
        let witness_bits = witness.clone().parallel_gather(witness_indices)?;
        let witness_digits = witness_bits.clone().parallel_pack_little_endian_bits(
            graph_params.input.input_count.clone(),
            graph_params.input.batch_bits.clone(),
        )?;
        let max_state_count = graph_params.input.max_state_count();
        let transitions = ring.preimage_family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::TRANSITIONS,
            vec![
                graph_params.input.input_count.clone(),
                max_state_count.clone(),
                graph_params.input.digit_base.clone(),
            ],
            (state_columns.clone(), state_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let input_evaluation = DiamondInputInjector::parameterized(graph_params.input.clone())
            .evaluate(initial_state, witness_digits, transitions)?;
        let states = input_evaluation.states;
        let public_key_compiler = Self::public_key_compiler(&graph_params);
        let encoding_compiler = BggEncodingCompiler { public_key: public_key_compiler.clone() };
        let public_key_matrices = ring.family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::PUBLIC_KEYS,
            IntExpr::Add(Box::new(witness_size.clone()), Box::new(IntExpr::constant(1))),
            (1, public_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let public_keys =
            BggPublicKeyFamily { matrices: public_key_matrices, reveal_plaintext: true };
        let one_preimage = ring.preimage_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::ONE_PREIMAGE,
            (state_columns.clone(), public_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let k_preimage = ring.preimage_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::K_PREIMAGE,
            (state_columns.clone(), 1),
            ArtifactConfidentiality::Public,
        );
        let decoder_preimage = ring.preimage_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::DECODER_PREIMAGE,
            (state_columns.clone(), 1),
            ArtifactConfidentiality::Public,
        );
        // Applying each preimage is the online projection step: state * K
        // consumes the corresponding encryption-time target relation.
        let initial_projection_state = states.get_static(0);
        let one_vector = initial_projection_state.clone().apply_preimage(one_preimage);
        let k_vector = initial_projection_state.clone().apply_preimage(k_preimage);
        let decoder = initial_projection_state.apply_preimage(decoder_preimage);
        let one_public_key_matrix = public_keys.matrices.get_static(0);
        let one_plaintext_matrix = ring.identity(1);
        let one_encoding = BggEncodingWire {
            vector: one_vector,
            pubkey: BggPublicKeyWire { matrix: one_public_key_matrix, reveal_plaintext: true },
            plaintext: Some(one_plaintext_matrix),
        };
        let zero_encoding = encoding_compiler.sub(&one_encoding, &one_encoding).expect("revealed");
        let witness_preimages = ring.preimage_family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::WITNESS_PREIMAGES,
            vec![witness_size.clone()],
            (state_columns.clone(), public_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        // This is a static coordinate map, not a runtime selector. Reusing the
        // exact map preserves the source relation across the artifact boundary
        // without asking the generic checker to prove arithmetic expressions
        // from two stages equivalent.
        let witness_source_map = IndexMap::new([IndexExpr::Add(
            Box::new(IndexExpr::Axis(0)),
            Box::new(IndexExpr::constant(1)),
        )]);
        let witness_states =
            states.reindex(vec![witness_size.clone()], witness_source_map.clone())?;
        // Witness states are projected by explicit ApplyPreimage operations;
        // the preimage artifact is consumed on the right of each state row.
        let witness_vectors = parallel_zip_bundle_result(
            (witness_states, witness_preimages),
            |_, (state, preimage)| Ok::<_, DslError>(state.apply_preimage(preimage)),
        )?;
        let witness_public_keys =
            public_keys.matrices.clone().reindex(vec![witness_size.clone()], witness_source_map)?;
        let witness_zero_plaintexts =
            Parallel::range(witness_size.clone()).map_values(|_| ring.zero((1, 1)))?;
        let witness_one_plaintexts =
            Parallel::range(witness_size.clone()).map_values(|_| ring.identity(1))?;
        let witness_plaintexts = witness_bits
            .parallel_select_mats(vec![witness_zero_plaintexts, witness_one_plaintexts])?;
        let instance_width = context.evaluate_int(circuit_params.instance_width.clone());
        let witness_end =
            instance_width.clone().add(Int::evaluate(witness_size.clone()).sub(Int::constant(1)));
        let packed_indices =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values({
                let instance_width = instance_width.clone();
                let witness_end = witness_end.clone();
                move |slot| {
                    let slot = slot.as_int();
                    let after_instance = instance_width.clone().less_equal(slot.clone()).to_int();
                    let before_end = slot.clone().less_equal(witness_end.clone()).to_int();
                    // Select the nonnegative base before subtracting it.  This is
                    // equivalent to selecting `0` versus `slot - instance_width`,
                    // but keeps the abstract selector family in
                    // `[0, witness_size)` without requiring Boolean correlation.
                    let output = after_instance
                        .mul(before_end)
                        .select_int(vec![instance_width.clone(), slot])
                        .expect("two witness indices")
                        .sub(instance_width.clone());
                    output
                }
            })?;
        let packed_vectors = witness_vectors.parallel_gather(packed_indices.clone())?;
        let packed_public_keys = witness_public_keys.parallel_gather(packed_indices.clone())?;
        let packed_plaintexts = witness_plaintexts.parallel_gather(packed_indices)?;
        let active_witness =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values({
                let instance_width = instance_width.clone();
                move |slot| {
                    let slot = slot.as_int();
                    let output = instance_width
                        .clone()
                        .less_equal(slot.clone())
                        .to_int()
                        .mul(slot.less_equal(witness_end.clone()).to_int());
                    output
                }
            })?;
        let active_zero_vectors = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| zero_encoding.vector.clone())?;
        let packed_vectors = active_witness
            .clone()
            .parallel_select_mats(vec![active_zero_vectors, packed_vectors])?;
        let active_zero_public_keys = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| zero_encoding.pubkey.matrix.clone())?;
        let packed_public_keys = active_witness
            .clone()
            .parallel_select_mats(vec![active_zero_public_keys, packed_public_keys])?;
        let active_zero_plaintexts = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| zero_encoding.plaintext.clone().expect("revealed"))?;
        let packed_plaintexts =
            active_witness.parallel_select_mats(vec![active_zero_plaintexts, packed_plaintexts])?;
        let selectors = instance;
        let instance_zero_vectors = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| zero_encoding.vector.clone())?;
        let instance_one_vectors = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| one_encoding.vector.clone())?;
        let selected_instance_vectors = selectors
            .clone()
            .parallel_select_mats(vec![instance_zero_vectors, instance_one_vectors])?;
        let instance_zero_public_keys = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| zero_encoding.pubkey.matrix.clone())?;
        let instance_one_public_keys = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| one_encoding.pubkey.matrix.clone())?;
        let selected_instance_keys = selectors
            .clone()
            .parallel_select_mats(vec![instance_zero_public_keys, instance_one_public_keys])?;
        let instance_zero_plaintexts = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| zero_encoding.plaintext.clone().expect("revealed"))?;
        let instance_one_plaintexts = Parallel::range(circuit_params.max_layer_width.clone())
            .map_values(|_| one_encoding.plaintext.clone().expect("revealed"))?;
        let selected_instance_plaintexts = selectors
            .parallel_select_mats(vec![instance_zero_plaintexts, instance_one_plaintexts])?;
        let active_instance =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values(|slot| {
                let output =
                    slot.as_int().less_equal(instance_width.clone().sub(Int::constant(1))).to_int();
                output
            })?;
        let circuit_vectors = active_instance
            .clone()
            .parallel_select_mats(vec![packed_vectors, selected_instance_vectors])?;
        let circuit_public_keys = active_instance
            .clone()
            .parallel_select_mats(vec![packed_public_keys, selected_instance_keys])?;
        let circuit_plaintexts = active_instance
            .parallel_select_mats(vec![packed_plaintexts, selected_instance_plaintexts])?;
        let circuit_inputs = BggEncodingFamily {
            vectors: circuit_vectors,
            public_keys: BggPublicKeyFamily {
                matrices: circuit_public_keys,
                reveal_plaintext: true,
            },
            plaintexts: circuit_plaintexts,
        };
        let circuit_output_family = evaluate_boolean_encoding_layers(
            &context,
            &circuit_params,
            circuit_data.clone(),
            circuit_inputs,
            one_encoding.clone(),
            encoding_compiler,
        )?;
        let circuit_output_index = circuit_data.output_source();
        let circuit_vector = circuit_output_family.vectors.get(circuit_output_index.clone());
        let r_decomposed = ring.preimage_artifact_input(
            encryption,
            DiamondArtifactNames::R_DECOMPOSED,
            (public_columns, 1),
            ArtifactConfidentiality::Public,
        );
        // The residual decoder target is R_dec = (C_one-C_out) * R.  Apply the
        // stored decomposition to consume R_dec, then subtract the K and
        // residual projections from the decoder state.
        let one_minus_circuit = one_encoding.vector - circuit_vector;
        let projected_difference = one_minus_circuit.apply_preimage(r_decomposed);
        let k_plus_projection = k_vector + projected_difference;
        let noisy_plaintext = decoder - k_plus_projection;
        let decoded = decode_boolean_interval(noisy_plaintext.clone(), graph_params.input.modulus);
        let graph = context
            .output(NOISY_PLAINTEXT_OUTPUT, noisy_plaintext)?
            .bool_output(DECODED_OUTPUT, decoded)?
            .build()?;
        Ok(DiamondDecryptionBuild { graph: DiamondDecryptionGraph { graph } })
    }

    fn public_key_compiler(params: &DiamondGraphParams) -> BggPublicKeyCompiler {
        BggPublicKeyCompiler {
            ring: params.input.ring(),
            base: params.input.gadget_base.clone(),
            digit_count: params.input.digit_count.clone(),
        }
    }

    fn sampler_layout(params: &DiamondGraphParams) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: params.input.modulus.clone(),
            ring_dimension: params.input.ring_dimension.clone(),
            secret_dimension: 1,
            digit_count: 1,
            gadget_base: params.input.gadget_base.clone(),
        }
    }

    fn tag(&self, suffix: &[u8]) -> Vec<u8> {
        let mut tag = self.bgg_tag.clone();
        tag.extend_from_slice(suffix);
        tag
    }
}

fn decode_boolean_interval(noisy_plaintext: Mat, modulus: IntExpr) -> Bool {
    let coefficient = noisy_plaintext.extract_coefficient(0);
    let quarter = Int::evaluate(IntExpr::RoundDiv(
        Box::new(IntExpr::Sub(Box::new(modulus), Box::new(IntExpr::constant(2)))),
        Box::new(IntExpr::constant(4)),
    ));
    let upper = quarter.clone().mul(Int::constant(3));
    let lower_ok = quarter.less_equal(coefficient.clone());
    let upper_ok = coefficient.less_equal(upper);
    lower_ok.to_int().add(upper_ok.to_int()).equal(Int::constant(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        RealExpr,
        artifact::{ProductionId, SpecHash, export_validated_manifest},
        node::NodeKind,
        types::WireType,
    };
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    fn compiler() -> DiamondWeCompiler {
        DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: 257.into(),
                ring_dimension: 8,
                input_count: 1,
                digit_base: 2,
                batch_bits: 1,
                gadget_base: 4.into(),
                digit_count: 2,
                trapdoor_sigma: RealExpr::from_integer(4),
                error_sigma: RealExpr::from_integer(1),
                error_max_coefficient_bound: 6.into(),
                preimage_max_coefficient_bound: 26.into(),
                bgg_tag: b"dynamic-diamond-test".to_vec(),
            },
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 1,
                depth: 2,
                max_layer_width: 3,
            },
        )
        .unwrap()
    }

    #[test]
    fn one_graph_accepts_runtime_circuit_families_in_both_stages() {
        let compiler = compiler();
        let encryption = compiler.build_encryption().unwrap().graph;
        let bindings = compiler.circuit_bindings().unwrap();
        let validated = encryption.validate(&bindings).unwrap();
        for family in [
            DiamondArtifactNames::PUBLIC_KEYS,
            DiamondArtifactNames::TRANSITIONS,
            DiamondArtifactNames::WITNESS_PREIMAGES,
        ] {
            assert!(validated.source.outputs().contains_key(family));
        }
        assert!(!validated.source.outputs().keys().any(|name| {
            name.starts_with("diamond_public_key_") ||
                name.starts_with("diamond_transition_") ||
                name.starts_with("diamond_witness_preimage_")
        }));
        let production = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [9; 32] };
        let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
        let decryption = compiler.build_decryption(production.clone()).unwrap().graph;
        decryption
            .validate_with_manifests(&bindings, &BTreeMap::from([(production, manifest)]))
            .unwrap();

        for graph in [&encryption.graph, &decryption.graph] {
            assert_eq!(
                graph
                    .parameters()
                    .iter()
                    .map(|parameter| parameter.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["instance_width", "witness_width", "depth", "max_layer_width"]
                    .into_iter()
                    .chain([
                        DiamondGraphParams::MODULUS,
                        DiamondGraphParams::RING_DIMENSION,
                        DiamondGraphParams::INPUT_COUNT,
                        DiamondGraphParams::DIGIT_BASE,
                        DiamondGraphParams::BATCH_BITS,
                        DiamondGraphParams::GADGET_BASE,
                        DiamondGraphParams::DIGIT_COUNT,
                        DiamondGraphParams::ERROR_BOUND,
                        DiamondGraphParams::PREIMAGE_BOUND,
                        DiamondGraphParams::TRAPDOOR_SIGMA,
                        DiamondGraphParams::ERROR_SIGMA,
                    ])
                    .collect::<Vec<_>>()
            );
            assert!(graph.root_scope().nodes().iter().any(|node| {
                matches!(
                    node.kind(),
                    NodeKind::Input { name, .. } if name == "circuit-gate-kind"
                )
            }));
            assert!(
                graph
                    .root_scope()
                    .nodes()
                    .iter()
                    .any(|node| matches!(node.kind(), NodeKind::SequentialLoop(_)))
            );
            assert!(!graph.root_scope().nodes().iter().any(|node| {
                matches!(node.kind(), NodeKind::Input { name, .. } if name.starts_with("circuit-kind-"))
            }));
        }
    }

    #[test]
    fn diamond_projection_artifacts_retain_preimage_types_and_use_strict_consumers() {
        let compiler = compiler();
        let encryption = compiler.build_encryption().unwrap().graph;
        let bindings = compiler.circuit_bindings().unwrap();
        let validated = encryption.validate(&bindings).unwrap();

        for name in [
            DiamondArtifactNames::ONE_PREIMAGE,
            DiamondArtifactNames::K_PREIMAGE,
            DiamondArtifactNames::DECODER_PREIMAGE,
            DiamondArtifactNames::R_DECOMPOSED,
        ] {
            let output = encryption.graph.outputs()[name].value;
            let output_type = &encryption
                .graph
                .root_scope()
                .node(output.node)
                .expect("projection artifact output node")
                .output_types()[output.port.0 as usize];
            assert!(matches!(output_type, WireType::Preimage(_)), "{name} lost its relation");
        }

        let witness_output =
            encryption.graph.outputs()[DiamondArtifactNames::WITNESS_PREIMAGES].value;
        let witness_type = &encryption
            .graph
            .root_scope()
            .node(witness_output.node)
            .expect("witness preimage family output node")
            .output_types()[witness_output.port.0 as usize];
        assert!(matches!(
            witness_type,
            WireType::Family { element, .. }
                if matches!(element.as_ref(), WireType::Preimage(_))
        ));
        assert!(
            encryption
                .graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::MaterializePreimageExact)),
            "the exact public-key projection must use the guarded materialization path"
        );

        let production = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [9; 32] };
        let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
        let decryption = compiler.build_decryption(production.clone()).unwrap().graph;
        decryption
            .validate_with_manifests(&bindings, &BTreeMap::from([(production, manifest)]))
            .unwrap();
        assert!(
            decryption
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ApplyPreimage))
                .count() >=
                4
        );
    }

    #[test]
    fn projection_target_layout_rewrite_matches_direct_row_concatenation() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let ring = mxx_dsl::Ring::new(
            num_bigint::BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let gadget = ring.gadget(1, 16, 5);
        let public = gadget.clone() + ring.identity(1).tensor(gadget.clone());
        let scalar = ring.polynomial([7.into()]);
        let zero_row = ring.zero((1, 5));
        let zero_scalar = ring.zero((1, 1));
        let one_scalar = ring.identity(1);
        let top = Mat::concat(ConcatAxis::Rows, vec![one_scalar.clone(), zero_scalar.clone()]);
        let bottom = Mat::concat(ConcatAxis::Rows, vec![zero_scalar.clone(), one_scalar]);

        let one_direct =
            Mat::concat(ConcatAxis::Rows, vec![public.clone() - gadget.clone(), zero_row.clone()]);
        let one_rewritten = top.clone() * public.clone() - top.clone() * gadget.clone();
        let witness_direct = Mat::concat(ConcatAxis::Rows, vec![public.clone(), -gadget.clone()]);
        let witness_rewritten = top.clone() * public.clone() - bottom.clone() * gadget.clone();
        let scalar_direct =
            Mat::concat(ConcatAxis::Rows, vec![ring.polynomial([5.into()]), scalar.clone()]);
        let scalar_rewritten = top.clone() * ring.polynomial([5.into()]) + bottom.clone() * scalar;
        let decoder_direct =
            Mat::concat(ConcatAxis::Rows, vec![ring.polynomial([9.into()]), zero_scalar]);
        let decoder_rewritten = top * ring.polynomial([9.into()]);

        let graph = DslContext::new("diamond-projection-target-layout")
            .output("one-direct", one_direct)
            .unwrap()
            .output("one-rewritten", one_rewritten)
            .unwrap()
            .output("witness-direct", witness_direct)
            .unwrap()
            .output("witness-rewritten", witness_rewritten)
            .unwrap()
            .output("scalar-direct", scalar_direct)
            .unwrap()
            .output("scalar-rewritten", scalar_rewritten)
            .unwrap()
            .output("decoder-direct", decoder_direct)
            .unwrap()
            .output("decoder-rewritten", decoder_rewritten)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (direct, rewritten) in [
            ("one-direct", "one-rewritten"),
            ("witness-direct", "witness-rewritten"),
            ("scalar-direct", "scalar-rewritten"),
            ("decoder-direct", "decoder-rewritten"),
        ] {
            let RuntimeValue::Matrix(direct) = &result.outputs[direct] else {
                panic!("direct target must be a matrix")
            };
            let RuntimeValue::Matrix(rewritten) = &result.outputs[rewritten] else {
                panic!("rewritten target must be a matrix")
            };
            assert_eq!(direct.as_ref(), rewritten.as_ref());
        }
    }

    #[test]
    fn padded_witness_public_key_indices_never_leave_the_exported_family() {
        let indices = padded_witness_public_key_indices(
            Int::constant(1),
            IntExpr::constant(1),
            IntExpr::constant(4),
        )
        .unwrap();
        let graph = DslContext::new("padded-witness-public-key-indices")
            .int_family_output("indices", indices)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Family(indices) = &result.outputs["indices"] else {
            panic!("indices output must be an integer family")
        };
        let actual = indices
            .iter()
            .map(|value| match value {
                RuntimeValue::Int(value) => value.clone(),
                _ => panic!("index family contains a non-integer"),
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![0.into(), 1.into(), 0.into(), 0.into()]);
    }
}
