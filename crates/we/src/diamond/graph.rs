use super::{DiamondArtifactNames, DiamondConfigError, DiamondWeConfig};
use crate::{WitnessEncryptionInterface, WitnessEncryptionProtocolDecl};
use mxx_bgg::{
    BggEncodingCompiler, BggEncodingFamily, BggEncodingWire, BggPublicKeyCompiler,
    BggPublicKeyFamily, BggPublicKeySampler, BggPublicKeyWire, BggSamplerLayout,
    DynamicBooleanBggError, evaluate_boolean_encoding_layers, evaluate_boolean_public_key_layers,
};
use mxx_correctness::{
    ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec,
    EndpointAnchor, EndpointAnchors, EndpointSemanticBinding, EndpointSpecId, InputContract,
    InputContractEntry, InputValueContract, OperationalDecoderKind, OperationalDecoderTarget,
    OutputRef, ParameterDecl, ParameterKind, ProtocolDecl, ProtocolInputBinding,
    ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec, ProtocolStage, StageId,
    StageInputName, Workflow,
};
use mxx_dsl::{
    Bool, BuiltGraph, DslContext, DslError, Int, Mat, Parallel, PurePredicateSpec, SemanticAnchor,
    Sequential, parallel_zip_bundle_result,
};
use mxx_gadgets::{
    circuit::{
        BOOLEAN_INSTANCE_INPUT, BOOLEAN_WITNESS_INPUT, BooleanCircuitError,
        BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanCircuitShape,
        boolean_circuit_satisfaction_predicate, boolean_circuit_validity_predicate,
    },
    input_injector::{DiamondInputInjector, DiamondInputParams, DiamondInputPreprocessError},
};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
    node::{ConcatAxis, IndexRange},
};
use thiserror::Error;

pub const HASH_KEY_INPUT: &str = "diamond-hash-key";
pub const MESSAGE_INPUT: &str = "diamond-message";
pub const DECODED_OUTPUT: &str = "diamond-decoded";
pub const NOISY_PLAINTEXT_OUTPUT: &str = "diamond-noisy-plaintext";
pub const DIAMOND_PROTOCOL_SOURCE_PATHS: &[&str] = &[
    "crates/bgg/Cargo.toml",
    "crates/bgg/src",
    "crates/correctness/Cargo.toml",
    "crates/correctness/src",
    "crates/dsl/Cargo.toml",
    "crates/dsl/src",
    "crates/gadgets/Cargo.toml",
    "crates/gadgets/src",
    "crates/ir-core/Cargo.toml",
    "crates/ir-core/src",
    "crates/we/Cargo.toml",
    "crates/we/examples/emit_correctness.rs",
    "crates/we/src",
];
const IDEAL_MESSAGE_OUTPUT: &str = "message";

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

fn diamond_parameter_validity_predicate(
    context: DslContext,
    circuit: &BooleanCircuitFamilyParams,
    params: &DiamondGraphParams,
) -> Result<PurePredicateSpec, DslError> {
    let evaluate = |expression| context.evaluate_int(expression);
    let modulus = evaluate(params.input.modulus.clone());
    let ring_dimension = evaluate(params.input.ring_dimension.clone());
    let input_count = evaluate(params.input.input_count.clone());
    let digit_base = evaluate(params.input.digit_base.clone());
    let batch_bits = evaluate(params.input.batch_bits.clone());
    let gadget_base = evaluate(params.input.gadget_base.clone());
    let digit_count = evaluate(params.input.digit_count.clone());
    let error_bound = evaluate(params.input.error_max_coefficient_bound.clone());
    let preimage_bound = evaluate(params.input.preimage_max_coefficient_bound.clone());
    let witness_width = evaluate(circuit.witness_width.clone());
    let two_to_batch_bits = Sequential::range(params.input.batch_bits.clone()).scan(
        Int::constant(1),
        Int::constant(0),
        |_, power, _| Ok(power.mul(Int::constant(2))),
    )?;
    let conditions = [
        Int::constant(1).less_equal(modulus),
        Int::constant(1).less_equal(ring_dimension),
        Int::constant(1).less_equal(input_count.clone()),
        Int::constant(1).less_equal(batch_bits.clone()),
        Int::constant(1).less_equal(digit_count),
        Int::constant(2).less_equal(gadget_base),
        Int::constant(0).less_equal(error_bound),
        Int::constant(0).less_equal(preimage_bound),
        two_to_batch_bits.less_equal(digit_base),
        witness_width.equal(input_count.mul(batch_bits)),
    ];
    let valid = conditions
        .into_iter()
        .map(Bool::to_int)
        .fold(Int::constant(1), Int::mul)
        .equal(Int::constant(1));
    PurePredicateSpec::new(context.bool_output("valid-parameters", valid)?.build()?)
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
    #[error("Diamond protocol declaration failed: {0}")]
    Protocol(String),
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
                    self.config
                        .trapdoor_sigma
                        .evaluate_rational(&ParamEnv::default())
                        .map_err(|error| DiamondCompileError::Protocol(error.to_string()))?,
                ),
                (
                    DiamondGraphParams::ERROR_SIGMA.to_owned(),
                    self.config
                        .error_sigma
                        .evaluate_rational(&ParamEnv::default())
                        .map_err(|error| DiamondCompileError::Protocol(error.to_string()))?,
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

    pub fn protocol_decl(&self) -> Result<WitnessEncryptionProtocolDecl, DiamondCompileError> {
        DiamondWeProtocolFamily::new(self.config.bgg_tag.clone()).protocol_decl()
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
        let circuit_output_matrix = circuit_output_family
            .matrices
            .get(circuit_output_index.clone())
            .semantic_anchor("diamond.encrypt.selected-circuit-public-key")?;
        let circuit_output =
            BggPublicKeyWire { matrix: circuit_output_matrix, reveal_plaintext: true };

        let gadget = ring.gadget(
            1,
            graph_params.input.gadget_base.clone(),
            graph_params.input.digit_count.clone(),
        );
        let public_columns = graph_params.input.digit_count.clone();
        let state_columns = graph_params.input.state_columns();
        let zero_row = ring.zero((1, public_columns.clone()));
        let one_difference = one_public_key.matrix.clone() - gadget.clone();
        let one_target = Mat::concat(ConcatAxis::Rows, vec![one_difference, zero_row]);
        let projection_trapdoor = input_preprocessing.final_trapdoors.get_static(0);
        let one_trapdoor = projection_trapdoor.clone();
        let one_sample = one_trapdoor
            .sample_preimage(one_target, (state_columns.clone(), public_columns.clone()));
        let one_preimage = one_sample.as_mat();
        let witness_indices =
            Parallel::range(witness_size).map_values(|bit| bit.as_int().add(Int::constant(1)))?;
        let witness_trapdoors =
            input_preprocessing.final_trapdoors.clone().parallel_gather(witness_indices.clone())?;
        let witness_public_keys = public_keys.matrices.clone().parallel_gather(witness_indices)?;
        let witness_targets = witness_public_keys.parallel_map_values({
            let gadget = gadget.clone();
            move |_, public_key| {
                let negated_gadget = -gadget.clone();
                Mat::concat(ConcatAxis::Rows, vec![public_key, negated_gadget])
            }
        })?;
        let witness_preimages =
            witness_trapdoors.parallel_zip_mat_values(witness_targets, |_, trapdoor, target| {
                let sample = trapdoor
                    .sample_preimage(target, (state_columns.clone(), public_columns.clone()));
                sample.as_mat()
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
        let half_modulus_polynomial = ring
            .polynomial([half_modulus.into()])
            .semantic_anchor("diamond.encrypt.message-carrier")?;
        let k_target = Mat::concat(
            ConcatAxis::Rows,
            vec![k_public_key_first.clone(), half_modulus_polynomial],
        );
        let k_trapdoor = projection_trapdoor.clone();
        let k_sample = k_trapdoor.sample_preimage(k_target, (state_columns.clone(), 1));
        let k_preimage = k_sample.as_mat();
        let r = ring.hash_matrix(hash_key, self.tag(b":r"), (1, public_columns.clone()));
        let r_column = r.slice(None, first_column);
        let r_decomposition = r_column.decompose(
            graph_params.input.gadget_base.clone(),
            graph_params.input.digit_count.clone(),
        );
        let r_materialized = r_decomposition.as_mat();
        let r_decomposed = r_materialized;
        let difference = public_key_compiler.sub(&one_public_key, &circuit_output);
        let projected_difference = difference.matrix * r_decomposed.clone();
        let decoder_public_key = k_public_key_first + projected_difference;
        let decoder_zero = ring.zero((1, 1));
        let decoder_target = Mat::concat(ConcatAxis::Rows, vec![decoder_public_key, decoder_zero]);
        let decoder_trapdoor = projection_trapdoor;
        let decoder_sample = decoder_trapdoor.sample_preimage(decoder_target, (state_columns, 1));
        let decoder_preimage = decoder_sample.as_mat();

        let graph = context
            .public_output(DiamondArtifactNames::INITIAL_STATE, input_preprocessing.p)?
            .public_output(DiamondArtifactNames::ONE_PREIMAGE, one_preimage)?
            .public_output(DiamondArtifactNames::K_PREIMAGE, k_preimage)?
            .public_output(DiamondArtifactNames::DECODER_PREIMAGE, decoder_preimage)?
            .public_output(DiamondArtifactNames::R_DECOMPOSED, r_decomposed)?
            .public_family_output(DiamondArtifactNames::PUBLIC_KEYS, public_keys.matrices)?
            .public_family_output(
                DiamondArtifactNames::TRANSITIONS,
                input_preprocessing.transitions,
            )?
            .public_family_output(DiamondArtifactNames::WITNESS_PREIMAGES, witness_preimages)?
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
        let circuit_data = BooleanCircuitFamilyInputs::protocol_inputs(&context, &circuit_params);
        let instance = context
            .int_family_input(BOOLEAN_INSTANCE_INPUT, circuit_params.max_layer_width.clone());
        let state_columns = graph_params.input.state_columns();
        let public_columns = graph_params.input.digit_count.clone();
        let initial_state = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::INITIAL_STATE,
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
        let transition_count = IntExpr::Mul(
            Box::new(graph_params.input.input_count.clone()),
            Box::new(IntExpr::Mul(
                Box::new(graph_params.input.digit_base.clone()),
                Box::new(max_state_count.clone()),
            )),
        )
        .canonicalize();
        let transitions = ring.family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::TRANSITIONS,
            transition_count,
            (state_columns.clone(), state_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let input_evaluation = DiamondInputInjector::parameterized(graph_params.input.clone())
            .evaluate(initial_state, witness_digits, transitions)?;
        let states =
            input_evaluation.states.semantic_anchor("diamond.decrypt.input-injector-states")?;
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
        let one_preimage = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::ONE_PREIMAGE,
            (state_columns.clone(), public_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let k_preimage = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::K_PREIMAGE,
            (state_columns.clone(), 1),
            ArtifactConfidentiality::Public,
        );
        let decoder_preimage = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::DECODER_PREIMAGE,
            (state_columns.clone(), 1),
            ArtifactConfidentiality::Public,
        );
        let initial_projection_state = states.get_static(0);
        let one_vector = initial_projection_state.clone() * one_preimage;
        let k_vector = initial_projection_state.clone() * k_preimage;
        let decoder = initial_projection_state * decoder_preimage;
        let one_public_key_matrix = public_keys.matrices.get_static(0);
        let one_plaintext_matrix = ring.identity(1);
        let one_encoding = BggEncodingWire {
            vector: one_vector,
            pubkey: BggPublicKeyWire { matrix: one_public_key_matrix, reveal_plaintext: true },
            plaintext: Some(one_plaintext_matrix),
        };
        let zero_encoding = encoding_compiler.sub(&one_encoding, &one_encoding).expect("revealed");
        let witness_preimages = ring.family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::WITNESS_PREIMAGES,
            witness_size.clone(),
            (state_columns.clone(), public_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let witness_state_indices = Parallel::range(witness_size.clone())
            .map_values(|bit| bit.as_int().add(Int::constant(1)))?;
        let witness_states = states.parallel_gather(witness_state_indices)?;
        let witness_vectors = parallel_zip_bundle_result(
            (witness_states, witness_preimages),
            |_, (state, preimage)| Ok::<_, DslError>(state * preimage),
        )?;
        let witness_public_indices = Parallel::range(witness_size.clone())
            .map_values(|bit| bit.as_int().add(Int::constant(1)))?;
        let witness_public_keys =
            public_keys.matrices.clone().parallel_gather(witness_public_indices)?;
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
                    let output = after_instance
                        .mul(before_end)
                        .select_int(vec![Int::constant(0), slot.sub(instance_width.clone())])
                        .expect("two witness indices");
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
        let circuit_vector = circuit_output_family
            .vectors
            .get(circuit_output_index.clone())
            .semantic_anchor("diamond.decrypt.selected-circuit-vector")?;
        let r_decomposed = ring.artifact_input(
            encryption,
            DiamondArtifactNames::R_DECOMPOSED,
            (public_columns, 1),
            ArtifactConfidentiality::Public,
        );
        let one_minus_circuit = one_encoding.vector - circuit_vector;
        let projected_difference = one_minus_circuit * r_decomposed;
        let k_plus_projection = k_vector + projected_difference;
        let noisy_plaintext =
            (decoder - k_plus_projection).semantic_anchor("diamond.decoder.residual")?;
        let decoded = decode_boolean_interval(noisy_plaintext.clone(), graph_params.input.modulus);
        let decoded = decoded.semantic_anchor("diamond.decoder.result")?;
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

    pub fn protocol_decl(&self) -> Result<WitnessEncryptionProtocolDecl, DiamondCompileError> {
        const ENCRYPT: &str = "encrypt";
        const DECRYPT: &str = "decrypt";
        let encrypt_id = StageId(ENCRYPT.to_owned());
        let decrypt_id = StageId(DECRYPT.to_owned());
        let encryption_build = self.build_encryption()?;
        // Runtime production identity is deliberately absent from protocol identity. Stage-relative
        // artifact bindings below carry the semantic producer relation; a concrete compiler
        // replaces this placeholder with the actual parameter-bound production id before
        // execution.
        let encryption_production =
            ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
        let decryption_build = self.build_decryption(encryption_production)?;
        let bindings = [
            (DiamondArtifactNames::INITIAL_STATE, false),
            (DiamondArtifactNames::ONE_PREIMAGE, false),
            (DiamondArtifactNames::K_PREIMAGE, false),
            (DiamondArtifactNames::DECODER_PREIMAGE, false),
            (DiamondArtifactNames::R_DECOMPOSED, false),
            (DiamondArtifactNames::TRANSITIONS, true),
            (DiamondArtifactNames::WITNESS_PREIMAGES, true),
            (DiamondArtifactNames::PUBLIC_KEYS, true),
        ]
        .into_iter()
        .map(|(artifact, family)| ArtifactBinding {
            consumer_input: StageInputName(if family {
                format!("artifact:{artifact}")
            } else {
                artifact.to_owned()
            }),
            producer_stage: encrypt_id.clone(),
            producer_output: ArtifactName(artifact.to_owned()),
        })
        .collect::<Vec<_>>();
        let encryption = encryption_build.graph.graph;
        let decryption = decryption_build.graph.graph;

        let (valid_context, _) =
            DiamondGraphParams::declare(DslContext::new("diamond-we-valid-circuit-data"));
        let valid = boolean_circuit_validity_predicate(valid_context)?;
        let (satisfied_context, _) =
            DiamondGraphParams::declare(DslContext::new("diamond-we-satisfied-circuit"));
        let satisfied = boolean_circuit_satisfaction_predicate(satisfied_context)?;
        let (parameter_context, parameter_circuit) =
            BooleanCircuitFamilyParams::declare(DslContext::new("diamond-we-valid-parameters"));
        let (parameter_context, parameter_values) = DiamondGraphParams::declare(parameter_context);
        let valid_parameters = diamond_parameter_validity_predicate(
            parameter_context,
            &parameter_circuit,
            &parameter_values,
        )?;
        let (ideal_context, ideal_params) =
            DiamondGraphParams::declare(DslContext::new("diamond-we-ideal"));
        let ideal_ring = ideal_params.input.ring();
        let (ideal_context, _) = BooleanCircuitFamilyParams::declare(ideal_context);
        let ideal = mxx_dsl::IdealSpec::new(
            ideal_context
                .bool_output(IDEAL_MESSAGE_OUTPUT, ideal_ring.bool_input(MESSAGE_INPUT))?
                .build()?,
        )?;
        let circuit_names = [
            "circuit-active-gate-count",
            "circuit-gate-kind",
            "circuit-left-source",
            "circuit-right-source",
            "circuit-output-source",
        ];
        let parameter = |name: &str| IntExpr::Var(name.to_owned());
        let max_layer_width = parameter(BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER);
        let flattened_gate_count = IntExpr::Mul(
            Box::new(parameter(BooleanCircuitFamilyParams::DEPTH_PARAMETER)),
            Box::new(max_layer_width.clone()),
        );
        let max_source =
            IntExpr::Sub(Box::new(max_layer_width.clone()), Box::new(IntExpr::constant(1)));
        let integer_family =
            |count: IntExpr, lower: IntExpr, upper: IntExpr| InputValueContract::Family {
                count,
                element: Box::new(InputValueContract::IntegerRange { lower, upper }),
            };
        let circuit_contracts = [
            (
                circuit_names[0],
                integer_family(
                    parameter(BooleanCircuitFamilyParams::DEPTH_PARAMETER),
                    IntExpr::constant(0),
                    max_layer_width.clone(),
                ),
            ),
            (
                circuit_names[1],
                integer_family(
                    flattened_gate_count.clone(),
                    IntExpr::constant(0),
                    IntExpr::constant(5),
                ),
            ),
            (
                circuit_names[2],
                integer_family(
                    flattened_gate_count.clone(),
                    IntExpr::constant(0),
                    max_source.clone(),
                ),
            ),
            (
                circuit_names[3],
                integer_family(flattened_gate_count, IntExpr::constant(0), max_source.clone()),
            ),
            (
                circuit_names[4],
                integer_family(IntExpr::constant(1), IntExpr::constant(0), max_source),
            ),
        ];
        let boolean_family =
            || integer_family(max_layer_width.clone(), IntExpr::constant(0), IntExpr::constant(1));
        let mut input_contracts = circuit_contracts
            .iter()
            .map(|(name, value)| InputContractEntry {
                id: ProtocolInputId::from(*name),
                name: (*name).to_owned(),
                value: value.clone(),
            })
            .collect::<Vec<_>>();
        input_contracts.extend([
            InputContractEntry {
                id: ProtocolInputId::from(BOOLEAN_INSTANCE_INPUT),
                name: BOOLEAN_INSTANCE_INPUT.to_owned(),
                value: boolean_family(),
            },
            InputContractEntry {
                id: ProtocolInputId::from(BOOLEAN_WITNESS_INPUT),
                name: BOOLEAN_WITNESS_INPUT.to_owned(),
                value: boolean_family(),
            },
            InputContractEntry {
                id: ProtocolInputId::from(MESSAGE_INPUT),
                name: MESSAGE_INPUT.to_owned(),
                value: InputValueContract::Boolean,
            },
            InputContractEntry {
                id: ProtocolInputId::from(HASH_KEY_INPUT),
                name: HASH_KEY_INPUT.to_owned(),
                value: InputValueContract::Bytes { length: IntExpr::constant(32) },
            },
        ]);
        let workflow_destination =
            |stage: &StageId, name: &str| ProtocolInputDestination::WorkflowStage {
                stage: stage.clone(),
                input: StageInputName(name.to_owned()),
            };
        let mut input_bindings = circuit_names
            .iter()
            .map(|name| ProtocolInputBinding {
                input: ProtocolInputId::from(*name),
                destinations: vec![
                    workflow_destination(&encrypt_id, name),
                    workflow_destination(&decrypt_id, name),
                    ProtocolInputDestination::Requirement {
                        requirement: 1,
                        input: (*name).to_owned(),
                    },
                    ProtocolInputDestination::Requirement {
                        requirement: 2,
                        input: (*name).to_owned(),
                    },
                ],
            })
            .collect::<Vec<_>>();
        input_bindings.extend([
            ProtocolInputBinding {
                input: ProtocolInputId::from(BOOLEAN_INSTANCE_INPUT),
                destinations: vec![
                    workflow_destination(&encrypt_id, BOOLEAN_INSTANCE_INPUT),
                    workflow_destination(&decrypt_id, BOOLEAN_INSTANCE_INPUT),
                    ProtocolInputDestination::Requirement {
                        requirement: 2,
                        input: BOOLEAN_INSTANCE_INPUT.to_owned(),
                    },
                ],
            },
            ProtocolInputBinding {
                input: ProtocolInputId::from(BOOLEAN_WITNESS_INPUT),
                destinations: vec![
                    workflow_destination(&decrypt_id, BOOLEAN_WITNESS_INPUT),
                    ProtocolInputDestination::Requirement {
                        requirement: 2,
                        input: BOOLEAN_WITNESS_INPUT.to_owned(),
                    },
                ],
            },
            ProtocolInputBinding {
                input: ProtocolInputId::from(MESSAGE_INPUT),
                destinations: vec![
                    workflow_destination(&encrypt_id, MESSAGE_INPUT),
                    ProtocolInputDestination::Ideal { input: MESSAGE_INPUT.to_owned() },
                ],
            },
            ProtocolInputBinding {
                input: ProtocolInputId::from(HASH_KEY_INPUT),
                destinations: vec![workflow_destination(&encrypt_id, HASH_KEY_INPUT)],
            },
        ]);
        let endpoint = EndpointSpecId::DiamondBooleanInterval;
        let decoder_node = decryption.graph.outputs()[DECODED_OUTPUT].value.node;
        let declaration = ProtocolDecl {
            params: [
                (
                    ParameterKind::Dimension,
                    &[
                        BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER,
                        BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER,
                        BooleanCircuitFamilyParams::DEPTH_PARAMETER,
                        BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER,
                        DiamondGraphParams::RING_DIMENSION,
                        DiamondGraphParams::INPUT_COUNT,
                        DiamondGraphParams::DIGIT_BASE,
                        DiamondGraphParams::BATCH_BITS,
                        DiamondGraphParams::DIGIT_COUNT,
                    ][..],
                ),
                (
                    ParameterKind::Integer,
                    &[
                        DiamondGraphParams::MODULUS,
                        DiamondGraphParams::GADGET_BASE,
                        DiamondGraphParams::ERROR_BOUND,
                        DiamondGraphParams::PREIMAGE_BOUND,
                    ][..],
                ),
                (
                    ParameterKind::Rational,
                    &[DiamondGraphParams::TRAPDOOR_SIGMA, DiamondGraphParams::ERROR_SIGMA][..],
                ),
            ]
            .into_iter()
            .flat_map(|(kind, names)| {
                names
                    .iter()
                    .map(move |name| ParameterDecl { name: (*name).to_owned(), kind: kind.clone() })
            })
            .collect(),
            bundle: ClosedProtocolBundle {
                workflow: Workflow {
                    stages: vec![
                        ProtocolStage {
                            id: encrypt_id.clone(),
                            graph: encryption.graph,
                            semantic_anchors: encryption.anchors,
                            derivation_attachments: encryption.derivation_attachments,
                            bindings: Vec::new(),
                        },
                        ProtocolStage {
                            id: decrypt_id.clone(),
                            graph: decryption.graph,
                            semantic_anchors: decryption.anchors,
                            derivation_attachments: decryption.derivation_attachments,
                            bindings,
                        },
                    ],
                    entrypoint: decrypt_id.clone(),
                },
                ideal,
                requirements: vec![valid_parameters, valid, satisfied],
                comparator: ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: DECODED_OUTPUT.to_owned(),
                        ideal_input: IDEAL_MESSAGE_OUTPUT.to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                },
                endpoints: EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: decrypt_id.clone(),
                        semantic_anchor: "diamond.decoder.result".to_owned(),
                        semantics: EndpointSemanticBinding::DiamondBoolean {
                            residual_stage: decrypt_id.clone(),
                            residual_anchor: "diamond.decoder.residual".to_owned(),
                            carrier_stage: encrypt_id.clone(),
                            carrier_anchor: "diamond.encrypt.message-carrier".to_owned(),
                            message: ProtocolInputId::from(MESSAGE_INPUT),
                        },
                        workflow_output: OutputRef {
                            stage: decrypt_id.clone(),
                            output: DECODED_OUTPUT.to_owned(),
                        },
                        ideal_output: IDEAL_MESSAGE_OUTPUT.to_owned(),
                    }],
                },
                operational_decoder_targets: vec![OperationalDecoderTarget {
                    target_id: "diamond-boolean-interval".to_owned(),
                    residual_stage: decrypt_id.clone(),
                    residual_output: NOISY_PLAINTEXT_OUTPUT.to_owned(),
                    decoder_stage: decrypt_id.clone(),
                    decoder_node,
                    kind: OperationalDecoderKind::BooleanInterval,
                }],
                endpoint_specs: vec![endpoint],
                input_contract: InputContract { inputs: input_contracts },
                input_bindings,
                precondition_spec: ProtocolPreconditionSpec {
                    requirement_outputs: vec![
                        "valid-parameters".to_owned(),
                        "valid".to_owned(),
                        "satisfied".to_owned(),
                    ],
                },
            },
        };
        let declaration = ProtocolDecl::new(declaration)
            .map_err(|error| DiamondCompileError::Protocol(error.to_string()))?;
        WitnessEncryptionProtocolDecl::new(
            declaration,
            WitnessEncryptionInterface {
                encryption_stage: encrypt_id,
                decryption_stage: decrypt_id,
                message: ProtocolInputId::from(MESSAGE_INPUT),
                instance: ProtocolInputId::from(BOOLEAN_INSTANCE_INPUT),
                witness: ProtocolInputId::from(BOOLEAN_WITNESS_INPUT),
            },
        )
        .map_err(|error| DiamondCompileError::Protocol(error.to_string()))
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
    };
    use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
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
        let decoded_output = decryption.graph.outputs()[DECODED_OUTPUT].value;
        let decoded_anchor =
            decryption.anchors.get("diamond.decoder.result").expect("decoded endpoint anchor");
        assert_eq!(decoded_anchor.len(), 1);
        assert_eq!(decoded_anchor[0].wire, decoded_output);
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
        let RuntimeValue::IndexedFamily(indices) = &result.outputs["indices"] else {
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

    #[test]
    fn parameter_predicate_enforces_dynamic_layout_invariants() {
        let (context, circuit) =
            BooleanCircuitFamilyParams::declare(DslContext::new("diamond-parameter-validity"));
        let (context, params) = DiamondGraphParams::declare(context);
        let predicate = diamond_parameter_validity_predicate(context, &circuit, &params).unwrap();
        let execute_with = |bindings: &ParamEnv| {
            let validated = mxx_ir_core::validate(&predicate.graph, bindings).unwrap();
            let result = execute(
                &validated,
                &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            matches!(result.outputs.get("valid-parameters"), Some(RuntimeValue::Bool(true)))
        };
        let valid = compiler().circuit_bindings().unwrap();
        assert!(execute_with(&valid));

        let mut wrong_witness_width = valid.clone();
        wrong_witness_width
            .integers
            .insert(BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(), 2.into());
        assert!(!execute_with(&wrong_witness_width));

        let mut undersized_digit_base = valid;
        undersized_digit_base.integers.insert(DiamondGraphParams::BATCH_BITS.to_owned(), 2.into());
        assert!(!execute_with(&undersized_digit_base));
    }

    #[test]
    fn protocol_declaration_covers_dynamic_circuit_and_witness_inputs() {
        let compiler = compiler();
        let declaration = compiler.protocol_decl().unwrap();
        assert_eq!(declaration.protocol().bundle.requirements.len(), 3);
        assert_eq!(declaration.protocol().params.len(), 15);
        assert_eq!(
            declaration
                .protocol()
                .bundle
                .input_contract
                .inputs
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<std::collections::BTreeSet<_>>(),
            [
                "circuit-active-gate-count",
                "circuit-gate-kind",
                "circuit-left-source",
                "circuit-right-source",
                "circuit-output-source",
                BOOLEAN_INSTANCE_INPUT,
                BOOLEAN_WITNESS_INPUT,
                MESSAGE_INPUT,
                HASH_KEY_INPUT,
            ]
            .into_iter()
            .collect()
        );
        let witness = declaration
            .protocol()
            .bundle
            .input_bindings
            .iter()
            .find(|binding| binding.input.0 == BOOLEAN_WITNESS_INPUT)
            .unwrap();
        assert_eq!(
            witness.destinations,
            vec![
                ProtocolInputDestination::WorkflowStage {
                    stage: StageId("decrypt".to_owned()),
                    input: StageInputName(BOOLEAN_WITNESS_INPUT.to_owned()),
                },
                ProtocolInputDestination::Requirement {
                    requirement: 2,
                    input: BOOLEAN_WITNESS_INPUT.to_owned(),
                },
            ]
        );
        assert!(declaration.protocol().bundle.input_bindings.iter().any(
            |binding| binding.input.0 == "circuit-gate-kind" && binding.destinations.len() == 4
        ));
        assert_eq!(
            declaration.protocol().bundle.precondition_spec.requirement_outputs,
            ["valid-parameters", "valid", "satisfied"]
        );
        assert_eq!(
            declaration.protocol().bundle.endpoints.entries[0].semantic_anchor,
            "diamond.decoder.result"
        );
        assert!(matches!(
            &declaration.protocol().bundle.comparator,
            ComparatorSpec::Equality { endpoints }
                if endpoints.len() == 1 && endpoints[0].failure_value
        ));
        let emitted = mxx_correctness::emit_protocol_for(
            "diamond-we-family",
            declaration.protocol(),
            "MxxWe",
            DIAMOND_PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        assert!(
            emitted
                .proof_ir
                .contains("def DiamondWeFamily_protocol : Mxx.Certificate.ClosedProtocolDecl")
        );
        assert!(!emitted.proof_ir.contains("SparseCertificate"));
        assert!(
            emitted
                .proof_ir
                .contains(".roundDivide (.parameter \"diamond_modulus\") (.constant (2 : Int))")
        );
        assert!(!emitted.proof_ir.contains(
            ".roundDivide (.subtract (.parameter \"diamond_modulus\") (.constant (1 : Int))) \
             (.constant (2 : Int))"
        ));
    }

    #[test]
    fn protocol_hash_is_independent_of_runtime_parameter_bindings() {
        let first = compiler();
        let mut second_config = first.config.clone();
        second_config.modulus = 769.into();
        second_config.ring_dimension = 16;
        second_config.digit_count = 3;
        let second = DiamondWeCompiler::new(
            second_config,
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 1,
                depth: 3,
                max_layer_width: 4,
            },
        )
        .unwrap();
        let direct =
            DiamondWeProtocolFamily::new(first.config.bgg_tag.clone()).protocol_decl().unwrap();
        let first = mxx_correctness::emit_protocol_for(
            "diamond-we-family",
            first.protocol_decl().unwrap().protocol(),
            "MxxWe",
            DIAMOND_PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        let second = mxx_correctness::emit_protocol_for(
            "diamond-we-family",
            second.protocol_decl().unwrap().protocol(),
            "MxxWe",
            DIAMOND_PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        let direct = mxx_correctness::emit_protocol_for(
            "diamond-we-family",
            direct.protocol(),
            "MxxWe",
            DIAMOND_PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        assert_eq!(first.freshness.workflow_hash, second.freshness.workflow_hash);
        assert_eq!(first.freshness.workflow_hash, direct.freshness.workflow_hash);
    }

    #[test]
    #[ignore = "measures Lean compilation of the midsize binary transport"]
    fn binary_transport_midsize_timing_gate() {
        use std::{path::Path, time::Instant};

        let declaration = DiamondWeProtocolFamily::new(b"mxx:diamond-we").protocol_decl().unwrap();
        let emit_started = Instant::now();
        let emitted = mxx_correctness::emit_protocol_for(
            "diamond-we-family-midsize",
            declaration.protocol(),
            "MxxWe",
            DIAMOND_PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        let emit_elapsed = emit_started.elapsed();
        let lean_workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lean");
        let cold_started = Instant::now();
        let prepared =
            mxx_correctness::prepare_emitted_operational_checker(&lean_workspace, &emitted)
                .unwrap();
        let cold_elapsed = cold_started.elapsed();
        let warm_started = Instant::now();
        let warm = mxx_correctness::prepare_emitted_operational_checker(&lean_workspace, &emitted)
            .unwrap();
        let warm_elapsed = warm_started.elapsed();
        assert_eq!(prepared.olean_path(), warm.olean_path());
        eprintln!(
            "midsize binary transport: generated_bytes={} emit_seconds={:.3} cold_prepare_seconds={:.3} warm_prepare_seconds={:.3}",
            emitted.ir.len(),
            emit_elapsed.as_secs_f64(),
            cold_elapsed.as_secs_f64(),
            warm_elapsed.as_secs_f64(),
        );
    }
}
