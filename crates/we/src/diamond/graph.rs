use super::{DiamondArtifactNames, DiamondConfigError, DiamondWeConfig};
use crate::{WitnessEncryptionInterface, WitnessEncryptionProtocol, WitnessEncryptionProtocolDecl};
use mxx_bgg::{
    BggEncodingCompiler, BggEncodingFamily, BggEncodingWire, BggPublicKeyCompiler,
    BggPublicKeyFamily, BggPublicKeyFamilySamplingTrace, BggPublicKeySampler, BggPublicKeyWire,
    BggSamplerLayout, DynamicBooleanBggError,
    boolean::{
        DecompositionConstructionTrace, EncodingBooleanConstructionTrace,
        EncodingSelectionConstructionTrace, MatrixBinaryConstructionTrace,
        PublicKeyBooleanConstructionTrace,
    },
    evaluate_boolean_encoding_layers, evaluate_boolean_public_key_layers,
};
use mxx_correctness::{
    ArtifactBinding, ArtifactName, Comparator, CorrectnessDecl, OutputRef, ParameterDecl,
    ParameterKind, ProtoInputName, ProtocolDecl, ProtocolStage, StageId, StageInputName,
    certificate::{
        ArtifactProvenance, BggPublicKeySamplingLayout, BinaryNodeRef, BooleanLayerMetadataLayout,
        BooleanLayersLayout, CertifiedLoopInputMode, CoreNodeRef, CoreOperandRef, CoreWireRef,
        DecoderLayout, DecoderTargetLayout, DecryptEncodingRhsDecomposition,
        DecryptionInitialEncodingsLayout, DiamondArtifactPreprocessingLayout, DiamondCertificate,
        DiamondInputPreprocessingLayout, DiamondWorkflowLayout, DynamicFamilyGetRef,
        EncodingBooleanLoopLayout, EncodingComponentOperationsLayout,
        EncryptPublicKeyRhsDecomposition, EncryptionInitialPublicKeysLayout, EvaluateIntRef,
        FamilyBooleanGateLayout, FamilyProductRef, InitialStateExpansionRef, InputInjectionLayout,
        KTargetLayout, LayerFamilyMetadataRef, LayerScalarMetadataRef, LocalBooleanGateLayout,
        LocalGadgetDecompositionRef, MatrixBinaryRef, MessageConstructionLayout, OneTargetLayout,
        OperationRef, ParallelCircuitInputPublicKeyLayout, ParallelDecompositionConsumer,
        ParallelFamilyGetRef, ParallelGatherRef, ParallelIndexFormulaRef, ParallelLoopRef,
        ParallelMatrixBinaryRef, ParallelOperationRef, ParallelPackedPublicKeyLayout,
        ParallelPreimageRef, ParallelSixWaySelectRef, ParallelTransitionTargetRef,
        ParallelTwoWaySelectRef, ParallelWitnessTargetLayout, PreimageRef,
        PublicKeyBooleanLoopLayout, SemanticCertificate, SequentialLoopRef, SixWaySelectRef,
        StageInputLayout, StageInterfaceLayout, StageOutputLayout, StaticTrapdoorLayout,
        TransitionSelectorBitLayout, TransitionSelectorLayout, TransitionTargetRef,
        TwoWaySelectRef, UnaryNodeRef, WitnessDigitPackingRef,
    },
};
use mxx_dsl::{
    BodyTraceRemapper, Bool, BuiltGraph, DslContext, DslError, EvaluateIntConstructionTrace,
    GatherConstructionTrace, Int, LoopConstructionTrace, Mat, PackedBitsConstructionTrace,
    Parallel, PurePredicateSpec, RemapConstructionTrace, SelectConstructionTrace, Sequential,
    parallel_zip_bundle_result_traced,
};
use mxx_gadgets::{
    circuit::{
        BOOLEAN_INSTANCE_INPUT, BOOLEAN_WITNESS_INPUT, BooleanCircuitError,
        BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanCircuitShape,
        LayerMetadataConstructionTrace, boolean_circuit_satisfaction_predicate,
        boolean_circuit_validity_predicate,
    },
    input_injector::{
        DiamondInputEvaluationConstructionTrace, DiamondInputInjector, DiamondInputParams,
        DiamondInputPreprocessError, DiamondInputPreprocessingConstructionTrace,
        MatrixProductConstructionTrace, OperationConstructionTrace, PreimageConstructionTrace,
        TransitionTargetConstructionTrace,
    },
};
use mxx_ir_core::{
    FreezeMap, FrozenGraphScopeId, Graph, IntExpr, ParamEnv, ValueHandle, WireRef,
    artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
    node::{ConcatAxis, IndexRange, NodeKind},
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

#[cfg(test)]
fn padded_witness_public_key_indices(
    instance_width: Int,
    witness_size: IntExpr,
    max_layer_width: IntExpr,
) -> Result<mxx_dsl::Family<Int>, DslError> {
    padded_witness_public_key_indices_traced(instance_width, witness_size, max_layer_width)
        .map(|(indices, _)| indices)
}

fn padded_witness_public_key_indices_traced(
    instance_width: Int,
    witness_size: IntExpr,
    max_layer_width: IntExpr,
) -> Result<(mxx_dsl::Family<Int>, LoopConstructionTrace<ValueHandle>), DslError> {
    let witness_end = instance_width.clone().add(Int::evaluate(witness_size).sub(Int::constant(1)));
    Parallel::range(max_layer_width).map_values_traced(move |slot| {
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
        (output.clone(), output.value_handle().clone())
    })
}

pub struct DiamondEncryptionGraph {
    pub graph: BuiltGraph,
}

pub struct DiamondDecryptionGraph {
    pub graph: BuiltGraph,
}

#[derive(Clone)]
struct NamedValueConstructionTrace {
    name: String,
    value: ValueHandle,
}

#[derive(Clone)]
struct DynamicFamilyGetConstructionTrace {
    family: ValueHandle,
    index: ValueHandle,
    output: ValueHandle,
}

#[derive(Clone)]
struct OrderedOperationConstructionTrace {
    inputs: Vec<ValueHandle>,
    output: ValueHandle,
}

#[derive(Clone)]
struct DecoderConstructionTrace {
    one_vector: OrderedOperationConstructionTrace,
    k_vector: OrderedOperationConstructionTrace,
    decoder_vector: OrderedOperationConstructionTrace,
    one_minus_circuit: OrderedOperationConstructionTrace,
    projected_difference: OrderedOperationConstructionTrace,
    k_plus_projection: OrderedOperationConstructionTrace,
    residual: OrderedOperationConstructionTrace,
    extract_coefficient: OrderedOperationConstructionTrace,
    threshold: ValueHandle,
    lower_compare: OrderedOperationConstructionTrace,
    upper_scale: OrderedOperationConstructionTrace,
    upper_compare: OrderedOperationConstructionTrace,
    lower_to_int: OrderedOperationConstructionTrace,
    upper_to_int: OrderedOperationConstructionTrace,
    comparison_sum: OrderedOperationConstructionTrace,
    equals_two: OrderedOperationConstructionTrace,
    decoded: ValueHandle,
}

#[derive(Clone)]
struct DecoderTailConstructionTrace {
    extract_coefficient: OrderedOperationConstructionTrace,
    threshold: ValueHandle,
    lower_compare: OrderedOperationConstructionTrace,
    upper_scale: OrderedOperationConstructionTrace,
    upper_compare: OrderedOperationConstructionTrace,
    lower_to_int: OrderedOperationConstructionTrace,
    upper_to_int: OrderedOperationConstructionTrace,
    comparison_sum: OrderedOperationConstructionTrace,
    equals_two: OrderedOperationConstructionTrace,
}

#[derive(Clone)]
struct DiamondEncryptionConstructionTrace {
    inputs: Vec<NamedValueConstructionTrace>,
    outputs: Vec<NamedValueConstructionTrace>,
    message: MessageConstructionTrace,
    preprocessing: DiamondInputPreprocessingConstructionTrace,
    public_key_sampling: BggPublicKeyFamilySamplingTrace,
    initial_public_keys: EncryptionInitialPublicKeysConstructionTrace,
    artifact_preprocessing: DiamondArtifactPreprocessingConstructionTrace,
    boolean_layers: PublicKeyBooleanConstructionTrace,
    selected_circuit_output: DynamicFamilyGetConstructionTrace,
}

#[derive(Clone)]
struct MessageConstructionTrace {
    to_int: OperationConstructionTrace,
    zero: OperationConstructionTrace,
    one: OperationConstructionTrace,
    select: SelectConstructionTrace,
}

#[derive(Clone, Debug)]
struct PackedPublicKeyConstructionTrace {
    in_range: SelectConstructionTrace,
    padded: SelectConstructionTrace,
}

impl RemapConstructionTrace for PackedPublicKeyConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            in_range: self.in_range.remap_current_body(map)?,
            padded: self.padded.remap_current_body(map)?,
        })
    }
}

#[derive(Clone, Debug)]
struct CircuitInputPublicKeyConstructionTrace {
    selected_instance: SelectConstructionTrace,
    selected_source: SelectConstructionTrace,
}

impl RemapConstructionTrace for CircuitInputPublicKeyConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            selected_instance: self.selected_instance.remap_current_body(map)?,
            selected_source: self.selected_source.remap_current_body(map)?,
        })
    }
}

#[derive(Clone, Debug)]
struct EncryptionInitialPublicKeysConstructionTrace {
    one_public_key: OperationConstructionTrace,
    zero_public_key: OperationConstructionTrace,
    instance_width: EvaluateIntConstructionTrace,
    public_indices: LoopConstructionTrace<ValueHandle>,
    public_candidates: LoopConstructionTrace<GatherConstructionTrace>,
    packed_inputs: LoopConstructionTrace<PackedPublicKeyConstructionTrace>,
    circuit_inputs: LoopConstructionTrace<CircuitInputPublicKeyConstructionTrace>,
}

#[derive(Clone)]
struct DiamondArtifactPreprocessingConstructionTrace {
    projection_trapdoor: StaticTrapdoorConstructionTrace,
    one_target: OneTargetConstructionTrace,
    one_preimage: PreimageConstructionTrace,
    witness_indices: LoopConstructionTrace<ValueHandle>,
    witness_trapdoors: LoopConstructionTrace<GatherConstructionTrace>,
    witness_public_keys: LoopConstructionTrace<GatherConstructionTrace>,
    witness_targets: LoopConstructionTrace<WitnessTargetConstructionTrace>,
    witness_preimages: LoopConstructionTrace<PreimageConstructionTrace>,
    k_target: KTargetConstructionTrace,
    k_preimage: PreimageConstructionTrace,
    r_hash: OperationConstructionTrace,
    r_slice: OperationConstructionTrace,
    r_decomposition: OperationConstructionTrace,
    r_materialization: OperationConstructionTrace,
    r_reshape: OperationConstructionTrace,
    decoder_target: DecoderTargetConstructionTrace,
    decoder_preimage: PreimageConstructionTrace,
}

#[derive(Clone)]
struct StaticTrapdoorConstructionTrace {
    public: OperationConstructionTrace,
    secret: OperationConstructionTrace,
}

#[derive(Clone)]
struct OneTargetConstructionTrace {
    gadget: OperationConstructionTrace,
    difference: OperationConstructionTrace,
    zero_row: OperationConstructionTrace,
    target: OperationConstructionTrace,
}

#[derive(Clone, Debug)]
struct WitnessTargetConstructionTrace {
    negated_gadget: OperationConstructionTrace,
    target: OperationConstructionTrace,
}

impl RemapConstructionTrace for WitnessTargetConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            negated_gadget: self.negated_gadget.remap_current_body(map)?,
            target: self.target.remap_current_body(map)?,
        })
    }
}

#[derive(Clone)]
struct KTargetConstructionTrace {
    public_key_hash: OperationConstructionTrace,
    first_column: OperationConstructionTrace,
    half_modulus: OperationConstructionTrace,
    target: OperationConstructionTrace,
}

#[derive(Clone)]
struct DecoderTargetConstructionTrace {
    public_key_difference: OperationConstructionTrace,
    projected_difference: OperationConstructionTrace,
    public_key_sum: OperationConstructionTrace,
    zero: OperationConstructionTrace,
    target: OperationConstructionTrace,
}

#[derive(Clone)]
struct DiamondDecryptionConstructionTrace {
    inputs: Vec<NamedValueConstructionTrace>,
    outputs: Vec<NamedValueConstructionTrace>,
    artifact_inputs: Vec<NamedValueConstructionTrace>,
    input_injection: DiamondInputEvaluationConstructionTrace,
    initial_encodings: DecryptionInitialEncodingsConstructionTrace,
    boolean_layers: EncodingBooleanConstructionTrace,
    selected_circuit_output: DynamicFamilyGetConstructionTrace,
    decoder: DecoderConstructionTrace,
}

#[derive(Clone, Debug)]
struct DecryptionInitialEncodingsConstructionTrace {
    witness_indices: LoopConstructionTrace<ValueHandle>,
    witness_bits: LoopConstructionTrace<GatherConstructionTrace>,
    witness_digits: LoopConstructionTrace<PackedBitsConstructionTrace>,
    initial_projection_state: OperationConstructionTrace,
    one_public_key: OperationConstructionTrace,
    one_plaintext: OperationConstructionTrace,
    zero_encoding: [OperationConstructionTrace; 3],
    witness_state_indices: LoopConstructionTrace<ValueHandle>,
    witness_states: LoopConstructionTrace<GatherConstructionTrace>,
    witness_vectors: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    witness_public_indices: LoopConstructionTrace<ValueHandle>,
    witness_public_keys: LoopConstructionTrace<GatherConstructionTrace>,
    witness_plaintext_constants: [LoopConstructionTrace<ValueHandle>; 2],
    witness_plaintexts: LoopConstructionTrace<SelectConstructionTrace>,
    instance_width: EvaluateIntConstructionTrace,
    packed_indices: LoopConstructionTrace<ValueHandle>,
    packed_vectors: LoopConstructionTrace<GatherConstructionTrace>,
    packed_public_keys: LoopConstructionTrace<GatherConstructionTrace>,
    packed_plaintexts: LoopConstructionTrace<GatherConstructionTrace>,
    active_witness: LoopConstructionTrace<ValueHandle>,
    active_witness_zeroes: [LoopConstructionTrace<ValueHandle>; 3],
    active_witness_selection: EncodingSelectionConstructionTrace,
    instance_constants: [[LoopConstructionTrace<ValueHandle>; 2]; 3],
    selected_instance: EncodingSelectionConstructionTrace,
    active_instance: LoopConstructionTrace<ValueHandle>,
    circuit_inputs: EncodingSelectionConstructionTrace,
}

struct DiamondEncryptionBuild {
    graph: DiamondEncryptionGraph,
    freeze_map: FreezeMap,
    trace: DiamondEncryptionConstructionTrace,
}

struct DiamondDecryptionBuild {
    graph: DiamondDecryptionGraph,
    freeze_map: FreezeMap,
    trace: DiamondDecryptionConstructionTrace,
}

fn named_value(name: impl Into<String>, value: &Mat) -> NamedValueConstructionTrace {
    NamedValueConstructionTrace { name: name.into(), value: value.value_handle().clone() }
}

fn named_family<T: mxx_dsl::FamilyElement>(
    name: impl Into<String>,
    value: &mxx_dsl::Family<T>,
) -> NamedValueConstructionTrace {
    NamedValueConstructionTrace { name: name.into(), value: value.value_handle().clone() }
}

fn named_artifact_family_input<T: mxx_dsl::FamilyElement>(
    artifact_name: &str,
    value: &mxx_dsl::Family<T>,
) -> NamedValueConstructionTrace {
    named_family(format!("artifact:{artifact_name}"), value)
}

fn artifact_name_from_consumer_input(name: &str) -> &str {
    name.strip_prefix("artifact:").unwrap_or(name)
}

fn named_circuit_inputs(circuit: &BooleanCircuitFamilyInputs) -> Vec<NamedValueConstructionTrace> {
    [
        ("circuit-active-gate-count", &circuit.active_gate_counts),
        ("circuit-gate-kind", &circuit.gate_kinds),
        ("circuit-left-source", &circuit.left_sources),
        ("circuit-right-source", &circuit.right_sources),
        ("circuit-output-source", &circuit.output_sources),
    ]
    .into_iter()
    .map(|(name, family)| NamedValueConstructionTrace {
        name: name.to_owned(),
        value: family.value_handle().clone(),
    })
    .collect()
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
        let instance_input_handle = instance.value_handle().clone();
        let message_input = ring.bool_input(MESSAGE_INPUT);
        let message_input_handle = message_input.value_handle().clone();
        let message_int = message_input.to_int();
        let message_to_int = OperationConstructionTrace {
            inputs: vec![message_input_handle.clone()],
            outputs: vec![message_int.value_handle().clone()],
        };
        let message_zero = ring.zero((1, 1));
        let message_zero_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![message_zero.value_handle().clone()],
        };
        let message_one = ring.identity(1);
        let message_one_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![message_one.value_handle().clone()],
        };
        let message_select_inputs =
            vec![message_zero.value_handle().clone(), message_one.value_handle().clone()];
        let message_select_selector = message_int.value_handle().clone();
        let message = message_int.select(vec![message_zero, message_one])?;
        let message_trace = MessageConstructionTrace {
            to_int: message_to_int,
            zero: message_zero_trace,
            one: message_one_trace,
            select: SelectConstructionTrace {
                selector: message_select_selector,
                branches: message_select_inputs,
                output: message.value_handle().clone(),
            },
        };
        let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
        let inputs = named_circuit_inputs(&circuit_data)
            .into_iter()
            .chain([
                NamedValueConstructionTrace {
                    name: BOOLEAN_INSTANCE_INPUT.to_owned(),
                    value: instance_input_handle,
                },
                NamedValueConstructionTrace {
                    name: MESSAGE_INPUT.to_owned(),
                    value: message_input_handle,
                },
                NamedValueConstructionTrace {
                    name: HASH_KEY_INPUT.to_owned(),
                    value: hash_key.value_handle().clone(),
                },
            ])
            .collect::<Vec<_>>();
        let input_preprocessing =
            DiamondInputInjector::parameterized(graph_params.input.clone()).preprocess(message)?;
        let preprocessing_trace = input_preprocessing.construction_trace.clone();
        let public_key_compiler = Self::public_key_compiler(&graph_params);
        let witness_size = graph_params.input.witness_size();
        let (public_keys, public_key_sampling) =
            BggPublicKeySampler { layout: Self::sampler_layout(&graph_params) }
                .sample_family_traced(
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
        let one_public_key_trace = OperationConstructionTrace {
            inputs: vec![public_keys.matrices.value_handle().clone()],
            outputs: vec![one_public_key.matrix.value_handle().clone()],
        };
        let zero_public_key_inputs = vec![
            one_public_key.matrix.value_handle().clone(),
            one_public_key.matrix.value_handle().clone(),
        ];
        let zero_public_key = public_key_compiler.sub(&one_public_key, &one_public_key);
        let zero_public_key_trace = OperationConstructionTrace {
            inputs: zero_public_key_inputs,
            outputs: vec![zero_public_key.matrix.value_handle().clone()],
        };
        let (instance_width, instance_width_trace) =
            context.evaluate_int_traced(circuit_params.instance_width.clone());
        let witness_end =
            instance_width.clone().add(Int::evaluate(witness_size.clone()).sub(Int::constant(1)));
        let (public_indices, public_indices_trace) = padded_witness_public_key_indices_traced(
            instance_width.clone(),
            witness_size.clone(),
            circuit_params.max_layer_width.clone(),
        )?;
        let (public_candidates, public_candidates_trace) =
            public_keys.matrices.clone().parallel_gather_traced(public_indices)?;
        let (packed_inputs, packed_inputs_trace) =
            public_candidates.parallel_map_values_traced({
                let zero = zero_public_key.matrix.clone();
                let instance_width = instance_width.clone();
                move |slot, candidate| {
                    let slot = slot.as_int();
                    let after_instance = instance_width.clone().less_equal(slot.clone()).to_int();
                    let before_end = slot.less_equal(witness_end.clone()).to_int();
                    let in_range_selector = before_end.value_handle().clone();
                    let in_range_branches =
                        vec![zero.value_handle().clone(), candidate.value_handle().clone()];
                    let in_range = before_end
                        .select(vec![zero.clone(), candidate])
                        .expect("matching public-key types");
                    let in_range_trace = SelectConstructionTrace {
                        selector: in_range_selector,
                        branches: in_range_branches,
                        output: in_range.value_handle().clone(),
                    };
                    let padded_selector = after_instance.value_handle().clone();
                    let padded_branches =
                        vec![zero.value_handle().clone(), in_range.value_handle().clone()];
                    let padded = after_instance
                        .select(vec![zero.clone(), in_range])
                        .expect("matching public-key types");
                    (
                        padded.clone(),
                        PackedPublicKeyConstructionTrace {
                            in_range: in_range_trace,
                            padded: SelectConstructionTrace {
                                selector: padded_selector,
                                branches: padded_branches,
                                output: padded.value_handle().clone(),
                            },
                        },
                    )
                }
            })?;
        let (circuit_input_matrices, circuit_inputs_trace) =
            parallel_zip_bundle_result_traced((instance, packed_inputs), |slot, (bit, packed)| {
                let index = slot.as_int();
                let selected_instance_selector = bit.value_handle().clone();
                let selected_instance_branches = vec![
                    zero_public_key.matrix.value_handle().clone(),
                    one_public_key.matrix.value_handle().clone(),
                ];
                let selected_instance = bit
                    .select(vec![zero_public_key.matrix.clone(), one_public_key.matrix.clone()])
                    .expect("matching public-key matrix types");
                let selected_instance_trace = SelectConstructionTrace {
                    selector: selected_instance_selector,
                    branches: selected_instance_branches,
                    output: selected_instance.value_handle().clone(),
                };
                let active =
                    index.clone().less_equal(instance_width.clone().sub(Int::constant(1))).to_int();
                let selected_source_selector = active.value_handle().clone();
                let selected_source_branches =
                    vec![packed.value_handle().clone(), selected_instance.value_handle().clone()];
                let selected_source = active.select(vec![packed, selected_instance])?;
                Ok::<_, DslError>((
                    selected_source.clone(),
                    CircuitInputPublicKeyConstructionTrace {
                        selected_instance: selected_instance_trace,
                        selected_source: SelectConstructionTrace {
                            selector: selected_source_selector,
                            branches: selected_source_branches,
                            output: selected_source.value_handle().clone(),
                        },
                    },
                ))
            })?;
        let circuit_inputs =
            BggPublicKeyFamily { matrices: circuit_input_matrices, reveal_plaintext: true };
        let (circuit_output_family, circuit_trace) = evaluate_boolean_public_key_layers(
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
        let selected_circuit_output = DynamicFamilyGetConstructionTrace {
            family: circuit_output_family.matrices.value_handle().clone(),
            index: circuit_output_index.value_handle().clone(),
            output: circuit_output_matrix.value_handle().clone(),
        };
        let circuit_output =
            BggPublicKeyWire { matrix: circuit_output_matrix, reveal_plaintext: true };

        let gadget = ring.gadget(
            1,
            graph_params.input.gadget_base.clone(),
            graph_params.input.digit_count.clone(),
        );
        let gadget_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![gadget.value_handle().clone()],
        };
        let public_columns = graph_params.input.digit_count.clone();
        let state_columns = graph_params.input.state_columns();
        let zero_row = ring.zero((1, public_columns.clone()));
        let zero_row_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![zero_row.value_handle().clone()],
        };
        let one_difference_inputs =
            vec![one_public_key.matrix.value_handle().clone(), gadget.value_handle().clone()];
        let one_difference = one_public_key.matrix.clone() - gadget.clone();
        let one_difference_trace = OperationConstructionTrace {
            inputs: one_difference_inputs,
            outputs: vec![one_difference.value_handle().clone()],
        };
        let one_target_inputs =
            vec![one_difference.value_handle().clone(), zero_row.value_handle().clone()];
        let one_target = Mat::concat(ConcatAxis::Rows, vec![one_difference, zero_row]);
        let one_target_trace = OperationConstructionTrace {
            inputs: one_target_inputs,
            outputs: vec![one_target.value_handle().clone()],
        };
        let projection_public_family =
            input_preprocessing.final_trapdoors.public_matrices().value_handle().clone();
        let projection_secret_family =
            input_preprocessing.final_trapdoors.secret_value_handle().clone();
        let projection_trapdoor = input_preprocessing.final_trapdoors.get_static(0);
        let projection_trapdoor_trace = StaticTrapdoorConstructionTrace {
            public: OperationConstructionTrace {
                inputs: vec![projection_public_family],
                outputs: vec![projection_trapdoor.public_matrix().value_handle().clone()],
            },
            secret: OperationConstructionTrace {
                inputs: vec![projection_secret_family],
                outputs: vec![projection_trapdoor.value_handle().clone()],
            },
        };
        let one_trapdoor = projection_trapdoor.clone();
        let one_sample_inputs = vec![
            one_trapdoor.public_matrix().value_handle().clone(),
            one_trapdoor.value_handle().clone(),
            one_target.value_handle().clone(),
        ];
        let one_sample = one_trapdoor
            .sample_preimage(one_target, (state_columns.clone(), public_columns.clone()));
        let one_sample_handle = one_sample.value_handle().clone();
        let one_preimage = one_sample.as_mat();
        let one_preimage_trace = PreimageConstructionTrace {
            sample: OperationConstructionTrace {
                inputs: one_sample_inputs,
                outputs: vec![one_sample_handle.clone()],
            },
            materialize: OperationConstructionTrace {
                inputs: vec![one_sample_handle],
                outputs: vec![one_preimage.value_handle().clone()],
            },
        };
        let (witness_indices, witness_indices_trace) = Parallel::range(witness_size)
            .map_values_traced(|bit| {
                let output = bit.as_int().add(Int::constant(1));
                (output.clone(), output.value_handle().clone())
            })?;
        let (witness_trapdoors, witness_trapdoors_trace) = input_preprocessing
            .final_trapdoors
            .clone()
            .parallel_gather_traced(witness_indices.clone())?;
        let (witness_public_keys, witness_public_keys_trace) =
            public_keys.matrices.clone().parallel_gather_traced(witness_indices)?;
        let (witness_targets, witness_targets_trace) = witness_public_keys
            .parallel_map_values_traced({
                let gadget = gadget.clone();
                move |_, public_key| {
                    let negated_gadget_input = gadget.value_handle().clone();
                    let negated_gadget = -gadget.clone();
                    let target_inputs = vec![
                        public_key.value_handle().clone(),
                        negated_gadget.value_handle().clone(),
                    ];
                    let target =
                        Mat::concat(ConcatAxis::Rows, vec![public_key, negated_gadget.clone()]);
                    (
                        target.clone(),
                        WitnessTargetConstructionTrace {
                            negated_gadget: OperationConstructionTrace {
                                inputs: vec![negated_gadget_input],
                                outputs: vec![negated_gadget.value_handle().clone()],
                            },
                            target: OperationConstructionTrace {
                                inputs: target_inputs,
                                outputs: vec![target.value_handle().clone()],
                            },
                        },
                    )
                }
            })?;
        let (witness_preimages, witness_preimage_trace) = witness_trapdoors
            .parallel_zip_mat_values_traced(witness_targets, |_, trapdoor, target| {
                let sample_inputs = vec![
                    trapdoor.public_matrix().value_handle().clone(),
                    trapdoor.value_handle().clone(),
                    target.value_handle().clone(),
                ];
                let sample = trapdoor
                    .sample_preimage(target, (state_columns.clone(), public_columns.clone()));
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

        let k_hash_key = hash_key.value_handle().clone();
        let k_public_key_matrix = ring.hash_matrix(
            hash_key.clone(),
            self.tag(b":k_public_key"),
            (1, public_columns.clone()),
        );
        let k_public_key_hash = OperationConstructionTrace {
            inputs: vec![k_hash_key],
            outputs: vec![k_public_key_matrix.value_handle().clone()],
        };
        let k_public_key =
            BggPublicKeyWire { matrix: k_public_key_matrix, reveal_plaintext: false };
        let first_column = Some(IndexRange { start: 0.into(), end: 1.into() });
        let k_first_column_input = k_public_key.matrix.value_handle().clone();
        let k_public_key_first = k_public_key.matrix.clone().slice(None, first_column.clone());
        let k_first_column_trace = OperationConstructionTrace {
            inputs: vec![k_first_column_input],
            outputs: vec![k_public_key_first.value_handle().clone()],
        };
        // The decoder subtracts the K encoding.  Sampling K against ceil(q / 2) therefore
        // leaves the canonical Boolean-one center floor(q / 2) modulo q for both even and odd q.
        let half_modulus = IntExpr::RoundDiv(
            Box::new(graph_params.input.modulus.clone()),
            Box::new(mxx_ir_core::IntExpr::constant(2)),
        );
        let half_modulus_polynomial = ring.polynomial([half_modulus.into()]);
        let half_modulus_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![half_modulus_polynomial.value_handle().clone()],
        };
        let k_target_inputs = vec![
            k_public_key_first.value_handle().clone(),
            half_modulus_polynomial.value_handle().clone(),
        ];
        let k_target = Mat::concat(
            ConcatAxis::Rows,
            vec![k_public_key_first.clone(), half_modulus_polynomial],
        );
        let k_target_trace = OperationConstructionTrace {
            inputs: k_target_inputs,
            outputs: vec![k_target.value_handle().clone()],
        };
        let k_trapdoor = projection_trapdoor.clone();
        let k_sample_inputs = vec![
            k_trapdoor.public_matrix().value_handle().clone(),
            k_trapdoor.value_handle().clone(),
            k_target.value_handle().clone(),
        ];
        let k_sample = k_trapdoor.sample_preimage(k_target, (state_columns.clone(), 1));
        let k_sample_handle = k_sample.value_handle().clone();
        let k_preimage = k_sample.as_mat();
        let k_preimage_trace = PreimageConstructionTrace {
            sample: OperationConstructionTrace {
                inputs: k_sample_inputs,
                outputs: vec![k_sample_handle.clone()],
            },
            materialize: OperationConstructionTrace {
                inputs: vec![k_sample_handle],
                outputs: vec![k_preimage.value_handle().clone()],
            },
        };
        let r_hash_key = hash_key.value_handle().clone();
        let r = ring.hash_matrix(hash_key, self.tag(b":r"), (1, public_columns.clone()));
        let r_hash_trace = OperationConstructionTrace {
            inputs: vec![r_hash_key],
            outputs: vec![r.value_handle().clone()],
        };
        let r_slice_input = r.value_handle().clone();
        let r_column = r.slice(None, first_column);
        let r_slice_trace = OperationConstructionTrace {
            inputs: vec![r_slice_input],
            outputs: vec![r_column.value_handle().clone()],
        };
        let r_decomposition_input = r_column.value_handle().clone();
        let r_decomposition = r_column.decompose(
            graph_params.input.gadget_base.clone(),
            graph_params.input.digit_count.clone(),
        );
        let r_decomposition_handle = r_decomposition.value_handle().clone();
        let r_materialized = r_decomposition.as_mat();
        let r_materialized_handle = r_materialized.value_handle().clone();
        let r_decomposed = r_materialized.reshape(public_columns, 1);
        let r_decomposition_trace = OperationConstructionTrace {
            inputs: vec![r_decomposition_input],
            outputs: vec![r_decomposition_handle.clone()],
        };
        let r_materialization_trace = OperationConstructionTrace {
            inputs: vec![r_decomposition_handle],
            outputs: vec![r_materialized_handle.clone()],
        };
        let r_reshape_trace = OperationConstructionTrace {
            inputs: vec![r_materialized_handle],
            outputs: vec![r_decomposed.value_handle().clone()],
        };
        let difference_inputs = vec![
            one_public_key.matrix.value_handle().clone(),
            circuit_output.matrix.value_handle().clone(),
        ];
        let difference = public_key_compiler.sub(&one_public_key, &circuit_output);
        let difference_trace = OperationConstructionTrace {
            inputs: difference_inputs,
            outputs: vec![difference.matrix.value_handle().clone()],
        };
        let projected_difference_inputs =
            vec![difference.matrix.value_handle().clone(), r_decomposed.value_handle().clone()];
        let projected_difference = difference.matrix * r_decomposed.clone();
        let projected_difference_trace = OperationConstructionTrace {
            inputs: projected_difference_inputs,
            outputs: vec![projected_difference.value_handle().clone()],
        };
        let decoder_public_key_inputs = vec![
            k_public_key_first.value_handle().clone(),
            projected_difference.value_handle().clone(),
        ];
        let decoder_public_key = k_public_key_first + projected_difference;
        let decoder_public_key_trace = OperationConstructionTrace {
            inputs: decoder_public_key_inputs,
            outputs: vec![decoder_public_key.value_handle().clone()],
        };
        let decoder_zero = ring.zero((1, 1));
        let decoder_zero_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![decoder_zero.value_handle().clone()],
        };
        let decoder_target_inputs =
            vec![decoder_public_key.value_handle().clone(), decoder_zero.value_handle().clone()];
        let decoder_target = Mat::concat(ConcatAxis::Rows, vec![decoder_public_key, decoder_zero]);
        let decoder_target_trace = OperationConstructionTrace {
            inputs: decoder_target_inputs,
            outputs: vec![decoder_target.value_handle().clone()],
        };
        let decoder_trapdoor = projection_trapdoor;
        let decoder_sample_inputs = vec![
            decoder_trapdoor.public_matrix().value_handle().clone(),
            decoder_trapdoor.value_handle().clone(),
            decoder_target.value_handle().clone(),
        ];
        let decoder_sample = decoder_trapdoor.sample_preimage(decoder_target, (state_columns, 1));
        let decoder_sample_handle = decoder_sample.value_handle().clone();
        let decoder_preimage = decoder_sample.as_mat();
        let decoder_preimage_trace = PreimageConstructionTrace {
            sample: OperationConstructionTrace {
                inputs: decoder_sample_inputs,
                outputs: vec![decoder_sample_handle.clone()],
            },
            materialize: OperationConstructionTrace {
                inputs: vec![decoder_sample_handle],
                outputs: vec![decoder_preimage.value_handle().clone()],
            },
        };

        let outputs = vec![
            named_value(DiamondArtifactNames::INITIAL_STATE, &input_preprocessing.p),
            named_value(DiamondArtifactNames::ONE_PREIMAGE, &one_preimage),
            named_value(DiamondArtifactNames::K_PREIMAGE, &k_preimage),
            named_value(DiamondArtifactNames::DECODER_PREIMAGE, &decoder_preimage),
            named_value(DiamondArtifactNames::R_DECOMPOSED, &r_decomposed),
            named_family(DiamondArtifactNames::PUBLIC_KEYS, &public_keys.matrices),
            named_family(DiamondArtifactNames::TRANSITIONS, &input_preprocessing.transitions),
            named_family(DiamondArtifactNames::WITNESS_PREIMAGES, &witness_preimages),
        ];
        let context = context
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
            .public_family_output(DiamondArtifactNames::WITNESS_PREIMAGES, witness_preimages)?;
        let (graph, freeze_map) = context.build_with_freeze_map()?;
        Ok(DiamondEncryptionBuild {
            graph: DiamondEncryptionGraph { graph },
            freeze_map,
            trace: DiamondEncryptionConstructionTrace {
                inputs,
                outputs,
                message: message_trace,
                preprocessing: preprocessing_trace,
                public_key_sampling,
                initial_public_keys: EncryptionInitialPublicKeysConstructionTrace {
                    one_public_key: one_public_key_trace,
                    zero_public_key: zero_public_key_trace,
                    instance_width: instance_width_trace,
                    public_indices: public_indices_trace,
                    public_candidates: public_candidates_trace,
                    packed_inputs: packed_inputs_trace,
                    circuit_inputs: circuit_inputs_trace,
                },
                artifact_preprocessing: DiamondArtifactPreprocessingConstructionTrace {
                    projection_trapdoor: projection_trapdoor_trace,
                    one_target: OneTargetConstructionTrace {
                        gadget: gadget_trace,
                        difference: one_difference_trace,
                        zero_row: zero_row_trace,
                        target: one_target_trace,
                    },
                    one_preimage: one_preimage_trace,
                    witness_indices: witness_indices_trace,
                    witness_trapdoors: witness_trapdoors_trace,
                    witness_public_keys: witness_public_keys_trace,
                    witness_targets: witness_targets_trace,
                    witness_preimages: witness_preimage_trace,
                    k_target: KTargetConstructionTrace {
                        public_key_hash: k_public_key_hash,
                        first_column: k_first_column_trace,
                        half_modulus: half_modulus_trace,
                        target: k_target_trace,
                    },
                    k_preimage: k_preimage_trace,
                    r_hash: r_hash_trace,
                    r_slice: r_slice_trace,
                    r_decomposition: r_decomposition_trace,
                    r_materialization: r_materialization_trace,
                    r_reshape: r_reshape_trace,
                    decoder_target: DecoderTargetConstructionTrace {
                        public_key_difference: difference_trace,
                        projected_difference: projected_difference_trace,
                        public_key_sum: decoder_public_key_trace,
                        zero: decoder_zero_trace,
                        target: decoder_target_trace,
                    },
                    decoder_preimage: decoder_preimage_trace,
                },
                boolean_layers: circuit_trace,
                selected_circuit_output,
            },
        })
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
        let instance_input_handle = instance.value_handle().clone();
        let state_columns = graph_params.input.state_columns();
        let public_columns = graph_params.input.digit_count.clone();
        let initial_state = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::INITIAL_STATE,
            (1, state_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let initial_state_input = named_value(DiamondArtifactNames::INITIAL_STATE, &initial_state);
        let witness =
            context.int_family_input(BOOLEAN_WITNESS_INPUT, circuit_params.max_layer_width.clone());
        let witness_input_handle = witness.value_handle().clone();
        let witness_size = graph_params.input.witness_size();
        let (witness_indices, witness_indices_trace) = Parallel::range(witness_size.clone())
            .map_values_traced(|bit| {
                let output = bit.as_int().add(Int::constant(0));
                (output.clone(), output.value_handle().clone())
            })?;
        let (witness_bits, witness_bits_trace) =
            witness.clone().parallel_gather_traced(witness_indices)?;
        let (witness_digits, witness_digits_trace) =
            witness_bits.clone().parallel_pack_little_endian_bits_traced(
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
        let transitions_input =
            named_artifact_family_input(DiamondArtifactNames::TRANSITIONS, &transitions);
        let input_evaluation = DiamondInputInjector::parameterized(graph_params.input.clone())
            .evaluate(initial_state, witness_digits, transitions)?;
        let input_injection_trace = input_evaluation.construction_trace.clone();
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
        let public_keys_input =
            named_artifact_family_input(DiamondArtifactNames::PUBLIC_KEYS, &public_key_matrices);
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
        let one_preimage_input = named_value(DiamondArtifactNames::ONE_PREIMAGE, &one_preimage);
        let k_preimage_input = named_value(DiamondArtifactNames::K_PREIMAGE, &k_preimage);
        let decoder_preimage_input =
            named_value(DiamondArtifactNames::DECODER_PREIMAGE, &decoder_preimage);
        let initial_projection_family = states.value_handle().clone();
        let initial_projection_state = states.get_static(0);
        let initial_projection_state_trace = OperationConstructionTrace {
            inputs: vec![initial_projection_family],
            outputs: vec![initial_projection_state.value_handle().clone()],
        };
        let one_vector_trace_inputs = vec![
            initial_projection_state.value_handle().clone(),
            one_preimage.value_handle().clone(),
        ];
        let one_vector = initial_projection_state.clone() * one_preimage;
        let one_vector_trace = OrderedOperationConstructionTrace {
            inputs: one_vector_trace_inputs,
            output: one_vector.value_handle().clone(),
        };
        let k_vector_trace_inputs = vec![
            initial_projection_state.value_handle().clone(),
            k_preimage.value_handle().clone(),
        ];
        let k_vector = initial_projection_state.clone() * k_preimage;
        let k_vector_trace = OrderedOperationConstructionTrace {
            inputs: k_vector_trace_inputs,
            output: k_vector.value_handle().clone(),
        };
        let decoder_vector_trace_inputs = vec![
            initial_projection_state.value_handle().clone(),
            decoder_preimage.value_handle().clone(),
        ];
        let decoder = initial_projection_state * decoder_preimage;
        let decoder_vector_trace = OrderedOperationConstructionTrace {
            inputs: decoder_vector_trace_inputs,
            output: decoder.value_handle().clone(),
        };
        let one_public_key_family = public_keys.matrices.value_handle().clone();
        let one_public_key_matrix = public_keys.matrices.get_static(0);
        let one_public_key_trace = OperationConstructionTrace {
            inputs: vec![one_public_key_family],
            outputs: vec![one_public_key_matrix.value_handle().clone()],
        };
        let one_plaintext_matrix = ring.identity(1);
        let one_plaintext_trace = OperationConstructionTrace {
            inputs: Vec::new(),
            outputs: vec![one_plaintext_matrix.value_handle().clone()],
        };
        let one_encoding = BggEncodingWire {
            vector: one_vector,
            pubkey: BggPublicKeyWire { matrix: one_public_key_matrix, reveal_plaintext: true },
            plaintext: Some(one_plaintext_matrix),
        };
        let zero_vector_inputs = vec![
            one_encoding.vector.value_handle().clone(),
            one_encoding.vector.value_handle().clone(),
        ];
        let zero_public_key_inputs = vec![
            one_encoding.pubkey.matrix.value_handle().clone(),
            one_encoding.pubkey.matrix.value_handle().clone(),
        ];
        let one_plaintext = one_encoding.plaintext.as_ref().expect("revealed");
        let zero_plaintext_inputs =
            vec![one_plaintext.value_handle().clone(), one_plaintext.value_handle().clone()];
        let zero_encoding = encoding_compiler.sub(&one_encoding, &one_encoding).expect("revealed");
        let zero_encoding_trace = [
            OperationConstructionTrace {
                inputs: zero_vector_inputs,
                outputs: vec![zero_encoding.vector.value_handle().clone()],
            },
            OperationConstructionTrace {
                inputs: zero_public_key_inputs,
                outputs: vec![zero_encoding.pubkey.matrix.value_handle().clone()],
            },
            OperationConstructionTrace {
                inputs: zero_plaintext_inputs,
                outputs: vec![
                    zero_encoding.plaintext.as_ref().expect("revealed").value_handle().clone(),
                ],
            },
        ];
        let witness_preimages = ring.family_artifact_input(
            encryption.clone(),
            DiamondArtifactNames::WITNESS_PREIMAGES,
            witness_size.clone(),
            (state_columns.clone(), public_columns.clone()),
            ArtifactConfidentiality::Public,
        );
        let witness_preimages_input = named_artifact_family_input(
            DiamondArtifactNames::WITNESS_PREIMAGES,
            &witness_preimages,
        );
        let (witness_state_indices, witness_state_indices_trace) =
            Parallel::range(witness_size.clone()).map_values_traced(|bit| {
                let output = bit.as_int().add(Int::constant(1));
                (output.clone(), output.value_handle().clone())
            })?;
        let (witness_states, witness_states_trace) =
            states.parallel_gather_traced(witness_state_indices)?;
        let (witness_vectors, witness_vectors_trace) = parallel_zip_bundle_result_traced(
            (witness_states, witness_preimages),
            |_, (state, preimage)| {
                let left = state.value_handle().clone();
                let right = preimage.value_handle().clone();
                let output = state * preimage;
                Ok::<_, DslError>((
                    output.clone(),
                    MatrixBinaryConstructionTrace {
                        left,
                        right,
                        output: output.value_handle().clone(),
                    },
                ))
            },
        )?;
        let (witness_public_indices, witness_public_indices_trace) =
            Parallel::range(witness_size.clone()).map_values_traced(|bit| {
                let output = bit.as_int().add(Int::constant(1));
                (output.clone(), output.value_handle().clone())
            })?;
        let (witness_public_keys, witness_public_keys_trace) =
            public_keys.matrices.clone().parallel_gather_traced(witness_public_indices)?;
        let (witness_zero_plaintexts, witness_zero_plaintexts_trace) =
            Parallel::range(witness_size.clone()).map_values_traced(|_| {
                let output = ring.zero((1, 1));
                (output.clone(), output.value_handle().clone())
            })?;
        let (witness_one_plaintexts, witness_one_plaintexts_trace) =
            Parallel::range(witness_size.clone()).map_values_traced(|_| {
                let output = ring.identity(1);
                (output.clone(), output.value_handle().clone())
            })?;
        let (witness_plaintexts, witness_plaintexts_trace) = witness_bits
            .parallel_select_mats_traced(vec![witness_zero_plaintexts, witness_one_plaintexts])?;
        let (instance_width, instance_width_trace) =
            context.evaluate_int_traced(circuit_params.instance_width.clone());
        let witness_end =
            instance_width.clone().add(Int::evaluate(witness_size.clone()).sub(Int::constant(1)));
        let (packed_indices, packed_indices_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced({
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
                    (output.clone(), output.value_handle().clone())
                }
            })?;
        let (packed_vectors, packed_vectors_trace) =
            witness_vectors.parallel_gather_traced(packed_indices.clone())?;
        let (packed_public_keys, packed_public_keys_trace) =
            witness_public_keys.parallel_gather_traced(packed_indices.clone())?;
        let (packed_plaintexts, packed_plaintexts_trace) =
            witness_plaintexts.parallel_gather_traced(packed_indices)?;
        let (active_witness, active_witness_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced({
                let instance_width = instance_width.clone();
                move |slot| {
                    let slot = slot.as_int();
                    let output = instance_width
                        .clone()
                        .less_equal(slot.clone())
                        .to_int()
                        .mul(slot.less_equal(witness_end.clone()).to_int());
                    (output.clone(), output.value_handle().clone())
                }
            })?;
        let (active_zero_vectors, active_zero_vectors_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                (zero_encoding.vector.clone(), zero_encoding.vector.value_handle().clone())
            })?;
        let (packed_vectors, active_witness_vectors_trace) = active_witness
            .clone()
            .parallel_select_mats_traced(vec![active_zero_vectors, packed_vectors])?;
        let (active_zero_public_keys, active_zero_public_keys_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                (
                    zero_encoding.pubkey.matrix.clone(),
                    zero_encoding.pubkey.matrix.value_handle().clone(),
                )
            })?;
        let (packed_public_keys, active_witness_public_keys_trace) = active_witness
            .clone()
            .parallel_select_mats_traced(vec![active_zero_public_keys, packed_public_keys])?;
        let (active_zero_plaintexts, active_zero_plaintexts_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                let zero = zero_encoding.plaintext.clone().expect("revealed");
                (zero.clone(), zero.value_handle().clone())
            })?;
        let (packed_plaintexts, active_witness_plaintexts_trace) = active_witness
            .parallel_select_mats_traced(vec![active_zero_plaintexts, packed_plaintexts])?;
        let active_witness_selection = EncodingSelectionConstructionTrace {
            vectors: active_witness_vectors_trace,
            public_keys: active_witness_public_keys_trace,
            plaintexts: active_witness_plaintexts_trace,
        };
        let selectors = instance;
        let (instance_zero_vectors, instance_zero_vectors_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                (zero_encoding.vector.clone(), zero_encoding.vector.value_handle().clone())
            })?;
        let (instance_one_vectors, instance_one_vectors_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                (one_encoding.vector.clone(), one_encoding.vector.value_handle().clone())
            })?;
        let (selected_instance_vectors, selected_instance_vectors_trace) = selectors
            .clone()
            .parallel_select_mats_traced(vec![instance_zero_vectors, instance_one_vectors])?;
        let (instance_zero_public_keys, instance_zero_public_keys_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                (
                    zero_encoding.pubkey.matrix.clone(),
                    zero_encoding.pubkey.matrix.value_handle().clone(),
                )
            })?;
        let (instance_one_public_keys, instance_one_public_keys_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                (
                    one_encoding.pubkey.matrix.clone(),
                    one_encoding.pubkey.matrix.value_handle().clone(),
                )
            })?;
        let (selected_instance_keys, selected_instance_keys_trace) =
            selectors.clone().parallel_select_mats_traced(vec![
                instance_zero_public_keys,
                instance_one_public_keys,
            ])?;
        let (instance_zero_plaintexts, instance_zero_plaintexts_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                let zero = zero_encoding.plaintext.clone().expect("revealed");
                (zero.clone(), zero.value_handle().clone())
            })?;
        let (instance_one_plaintexts, instance_one_plaintexts_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|_| {
                let one = one_encoding.plaintext.clone().expect("revealed");
                (one.clone(), one.value_handle().clone())
            })?;
        let (selected_instance_plaintexts, selected_instance_plaintexts_trace) = selectors
            .parallel_select_mats_traced(vec![instance_zero_plaintexts, instance_one_plaintexts])?;
        let selected_instance = EncodingSelectionConstructionTrace {
            vectors: selected_instance_vectors_trace,
            public_keys: selected_instance_keys_trace,
            plaintexts: selected_instance_plaintexts_trace,
        };
        let (active_instance, active_instance_trace) =
            Parallel::range(circuit_params.max_layer_width.clone()).map_values_traced(|slot| {
                let output =
                    slot.as_int().less_equal(instance_width.clone().sub(Int::constant(1))).to_int();
                (output.clone(), output.value_handle().clone())
            })?;
        let (circuit_vectors, circuit_vectors_trace) = active_instance
            .clone()
            .parallel_select_mats_traced(vec![packed_vectors, selected_instance_vectors])?;
        let (circuit_public_keys, circuit_public_keys_trace) = active_instance
            .clone()
            .parallel_select_mats_traced(vec![packed_public_keys, selected_instance_keys])?;
        let (circuit_plaintexts, circuit_plaintexts_trace) = active_instance
            .parallel_select_mats_traced(vec![packed_plaintexts, selected_instance_plaintexts])?;
        let circuit_inputs = BggEncodingFamily {
            vectors: circuit_vectors,
            public_keys: BggPublicKeyFamily {
                matrices: circuit_public_keys,
                reveal_plaintext: true,
            },
            plaintexts: circuit_plaintexts,
        };
        let circuit_inputs_trace = EncodingSelectionConstructionTrace {
            vectors: circuit_vectors_trace,
            public_keys: circuit_public_keys_trace,
            plaintexts: circuit_plaintexts_trace,
        };
        let (circuit_output_family, circuit_trace) = evaluate_boolean_encoding_layers(
            &context,
            &circuit_params,
            circuit_data.clone(),
            circuit_inputs,
            one_encoding.clone(),
            encoding_compiler,
        )?;
        let circuit_output_index = circuit_data.output_source();
        let circuit_vector = circuit_output_family.vectors.get(circuit_output_index.clone());
        let selected_circuit_output = DynamicFamilyGetConstructionTrace {
            family: circuit_output_family.vectors.value_handle().clone(),
            index: circuit_output_index.value_handle().clone(),
            output: circuit_vector.value_handle().clone(),
        };
        let r_decomposed = ring.artifact_input(
            encryption,
            DiamondArtifactNames::R_DECOMPOSED,
            (public_columns, 1),
            ArtifactConfidentiality::Public,
        );
        let r_decomposed_input = named_value(DiamondArtifactNames::R_DECOMPOSED, &r_decomposed);
        let one_minus_inputs =
            vec![one_encoding.vector.value_handle().clone(), circuit_vector.value_handle().clone()];
        let one_minus_circuit = one_encoding.vector - circuit_vector;
        let one_minus_circuit_trace = OrderedOperationConstructionTrace {
            inputs: one_minus_inputs,
            output: one_minus_circuit.value_handle().clone(),
        };
        let projected_inputs =
            vec![one_minus_circuit.value_handle().clone(), r_decomposed.value_handle().clone()];
        let projected_difference = one_minus_circuit * r_decomposed;
        let projected_difference_trace = OrderedOperationConstructionTrace {
            inputs: projected_inputs,
            output: projected_difference.value_handle().clone(),
        };
        let k_plus_inputs =
            vec![k_vector.value_handle().clone(), projected_difference.value_handle().clone()];
        let k_plus_projection = k_vector + projected_difference;
        let k_plus_projection_trace = OrderedOperationConstructionTrace {
            inputs: k_plus_inputs,
            output: k_plus_projection.value_handle().clone(),
        };
        let residual_inputs =
            vec![decoder.value_handle().clone(), k_plus_projection.value_handle().clone()];
        let noisy_plaintext = decoder - k_plus_projection;
        let residual_trace = OrderedOperationConstructionTrace {
            inputs: residual_inputs,
            output: noisy_plaintext.value_handle().clone(),
        };
        let (decoded, decoder_tail) =
            decode_boolean_interval(noisy_plaintext.clone(), graph_params.input.modulus);
        let decoder_trace = DecoderConstructionTrace {
            one_vector: one_vector_trace,
            k_vector: k_vector_trace,
            decoder_vector: decoder_vector_trace,
            one_minus_circuit: one_minus_circuit_trace,
            projected_difference: projected_difference_trace,
            k_plus_projection: k_plus_projection_trace,
            residual: residual_trace,
            extract_coefficient: decoder_tail.extract_coefficient,
            threshold: decoder_tail.threshold,
            lower_compare: decoder_tail.lower_compare,
            upper_scale: decoder_tail.upper_scale,
            upper_compare: decoder_tail.upper_compare,
            lower_to_int: decoder_tail.lower_to_int,
            upper_to_int: decoder_tail.upper_to_int,
            comparison_sum: decoder_tail.comparison_sum,
            equals_two: decoder_tail.equals_two,
            decoded: decoded.value_handle().clone(),
        };
        let outputs = vec![
            named_value(NOISY_PLAINTEXT_OUTPUT, &noisy_plaintext),
            NamedValueConstructionTrace {
                name: DECODED_OUTPUT.to_owned(),
                value: decoded.value_handle().clone(),
            },
        ];
        let artifact_inputs = vec![
            initial_state_input,
            transitions_input,
            one_preimage_input,
            k_preimage_input,
            decoder_preimage_input,
            public_keys_input,
            witness_preimages_input,
            r_decomposed_input,
        ];
        let inputs = named_circuit_inputs(&circuit_data)
            .into_iter()
            .chain([
                NamedValueConstructionTrace {
                    name: BOOLEAN_INSTANCE_INPUT.to_owned(),
                    value: instance_input_handle,
                },
                NamedValueConstructionTrace {
                    name: BOOLEAN_WITNESS_INPUT.to_owned(),
                    value: witness_input_handle,
                },
            ])
            .chain(artifact_inputs.iter().cloned())
            .collect::<Vec<_>>();
        let context = context
            .output(NOISY_PLAINTEXT_OUTPUT, noisy_plaintext)?
            .bool_output(DECODED_OUTPUT, decoded)?;
        let (graph, freeze_map) = context.build_with_freeze_map()?;
        Ok(DiamondDecryptionBuild {
            graph: DiamondDecryptionGraph { graph },
            freeze_map,
            trace: DiamondDecryptionConstructionTrace {
                inputs,
                outputs,
                artifact_inputs,
                input_injection: input_injection_trace,
                initial_encodings: DecryptionInitialEncodingsConstructionTrace {
                    witness_indices: witness_indices_trace,
                    witness_bits: witness_bits_trace,
                    witness_digits: witness_digits_trace,
                    initial_projection_state: initial_projection_state_trace,
                    one_public_key: one_public_key_trace,
                    one_plaintext: one_plaintext_trace,
                    zero_encoding: zero_encoding_trace,
                    witness_state_indices: witness_state_indices_trace,
                    witness_states: witness_states_trace,
                    witness_vectors: witness_vectors_trace,
                    witness_public_indices: witness_public_indices_trace,
                    witness_public_keys: witness_public_keys_trace,
                    witness_plaintext_constants: [
                        witness_zero_plaintexts_trace,
                        witness_one_plaintexts_trace,
                    ],
                    witness_plaintexts: witness_plaintexts_trace,
                    instance_width: instance_width_trace,
                    packed_indices: packed_indices_trace,
                    packed_vectors: packed_vectors_trace,
                    packed_public_keys: packed_public_keys_trace,
                    packed_plaintexts: packed_plaintexts_trace,
                    active_witness: active_witness_trace,
                    active_witness_zeroes: [
                        active_zero_vectors_trace,
                        active_zero_public_keys_trace,
                        active_zero_plaintexts_trace,
                    ],
                    active_witness_selection,
                    instance_constants: [
                        [instance_zero_vectors_trace, instance_one_vectors_trace],
                        [instance_zero_public_keys_trace, instance_one_public_keys_trace],
                        [instance_zero_plaintexts_trace, instance_one_plaintexts_trace],
                    ],
                    selected_instance,
                    active_instance: active_instance_trace,
                    circuit_inputs: circuit_inputs_trace,
                },
                boolean_layers: circuit_trace,
                selected_circuit_output,
                decoder: decoder_trace,
            },
        })
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
            gaussian_max_coefficient_bound: params.input.error_max_coefficient_bound.clone(),
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
        let bindings = decryption_build
            .trace
            .artifact_inputs
            .iter()
            .map(|artifact| ArtifactBinding {
                consumer_input: StageInputName(artifact.name.clone()),
                producer_stage: encrypt_id.clone(),
                producer_output: ArtifactName(
                    artifact_name_from_consumer_input(&artifact.name).to_owned(),
                ),
            })
            .collect::<Vec<_>>();
        let certificate = build_diamond_certificate_from_traces(
            &encrypt_id,
            &decrypt_id,
            &encryption_build,
            &decryption_build,
        )?;
        let encryption = encryption_build.graph.graph.graph;
        let decryption = decryption_build.graph.graph.graph;

        let both_stages = |name: String| {
            (
                ProtoInputName(name.clone()),
                vec![
                    (encrypt_id.clone(), StageInputName(name.clone())),
                    (decrypt_id.clone(), StageInputName(name)),
                ],
            )
        };
        let mut protocol_inputs = Vec::new();
        for name in [
            "circuit-active-gate-count",
            "circuit-gate-kind",
            "circuit-left-source",
            "circuit-right-source",
            "circuit-output-source",
            BOOLEAN_INSTANCE_INPUT,
        ] {
            protocol_inputs.push(both_stages(name.to_owned()));
        }
        protocol_inputs.push((
            ProtoInputName(BOOLEAN_WITNESS_INPUT.to_owned()),
            vec![(decrypt_id.clone(), StageInputName(BOOLEAN_WITNESS_INPUT.to_owned()))],
        ));
        protocol_inputs.push((
            ProtoInputName(MESSAGE_INPUT.to_owned()),
            vec![(encrypt_id.clone(), StageInputName(MESSAGE_INPUT.to_owned()))],
        ));
        protocol_inputs.push((
            ProtoInputName(HASH_KEY_INPUT.to_owned()),
            vec![(encrypt_id.clone(), StageInputName(HASH_KEY_INPUT.to_owned()))],
        ));

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
            ideal_context.bool_output("message", ideal_ring.bool_input(MESSAGE_INPUT))?.build()?,
        )?;
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
            stages: vec![
                ProtocolStage { id: encrypt_id.clone(), graph: encryption, bindings: Vec::new() },
                ProtocolStage { id: decrypt_id.clone(), graph: decryption, bindings },
            ],
            entrypoint: decrypt_id.clone(),
            semantic_certificate: SemanticCertificate::Diamond(Box::new(certificate)),
            correctness: CorrectnessDecl {
                protocol_inputs,
                requires: vec![valid_parameters, valid, satisfied],
                ideal,
                compared_outputs: vec![OutputRef {
                    stage: decrypt_id.clone(),
                    output: DECODED_OUTPUT.to_owned(),
                }],
                comparator: Comparator::Equal,
            },
        };
        let declaration = ProtocolDecl::new(declaration)
            .map_err(|error| DiamondCompileError::Protocol(error.to_string()))?;
        WitnessEncryptionProtocolDecl::new(
            declaration,
            WitnessEncryptionInterface {
                encryption_stage: encrypt_id,
                decryption_stage: decrypt_id,
                message: ProtoInputName(MESSAGE_INPUT.to_owned()),
                instance: ProtoInputName(BOOLEAN_INSTANCE_INPUT.to_owned()),
                witness: ProtoInputName(BOOLEAN_WITNESS_INPUT.to_owned()),
            },
        )
        .map_err(|error| DiamondCompileError::Protocol(error.to_string()))
    }
}

fn certificate_error(message: impl Into<String>) -> DiamondCompileError {
    DiamondCompileError::Protocol(message.into())
}

struct CertificateRefBuilder<'a> {
    stage: StageId,
    graph: &'a Graph,
    freeze_map: &'a FreezeMap,
}

struct ResolvedLoopBoundary {
    operation: CoreNodeRef,
    body_scope: FrozenGraphScopeId,
    arguments: Vec<CoreOperandRef>,
    body_inputs: Vec<CoreWireRef>,
    body_outputs: Vec<CoreWireRef>,
    outputs: Vec<CoreWireRef>,
}

impl CertificateRefBuilder<'_> {
    fn wire(&self, value: &ValueHandle) -> Result<CoreWireRef, DiamondCompileError> {
        let resolved = self.freeze_map.resolve_unique(value).map_err(|error| {
            certificate_error(format!("construction trace resolution failed: {error}"))
        })?;
        Ok(self.scoped_wire(&resolved.scope, resolved.wire))
    }

    fn scoped_wire(&self, scope: &FrozenGraphScopeId, wire: WireRef) -> CoreWireRef {
        CoreWireRef {
            node: CoreNodeRef::new(self.stage.clone(), scope.clone(), wire.node),
            port: wire.port,
        }
    }

    fn operand(
        &self,
        operation: &CoreNodeRef,
        index: usize,
        input: &ValueHandle,
    ) -> Result<CoreOperandRef, DiamondCompileError> {
        Ok(operation.operand(index as u32, self.wire(input)?))
    }
}

fn trace_operation(
    refs: &CertificateRefBuilder<'_>,
    trace: &OperationConstructionTrace,
) -> Result<OperationRef, DiamondCompileError> {
    trace_operation_handles(refs, &trace.inputs, &trace.outputs)
}

fn trace_operation_handles(
    refs: &CertificateRefBuilder<'_>,
    inputs: &[ValueHandle],
    output_handles: &[ValueHandle],
) -> Result<OperationRef, DiamondCompileError> {
    let outputs =
        output_handles.iter().map(|output| refs.wire(output)).collect::<Result<Vec<_>, _>>()?;
    let operation = outputs
        .first()
        .ok_or_else(|| certificate_error("construction operation has no output"))?
        .node
        .clone();
    if outputs.iter().any(|output| output.node != operation) {
        return Err(certificate_error("construction operation outputs have different producers"));
    }
    let inputs = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| refs.operand(&operation, index, input))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OperationRef { operation, inputs, outputs })
}

fn trace_select(
    refs: &CertificateRefBuilder<'_>,
    trace: &SelectConstructionTrace,
) -> Result<OperationRef, DiamondCompileError> {
    let inputs =
        std::iter::once(&trace.selector).chain(&trace.branches).cloned().collect::<Vec<_>>();
    trace_operation_handles(refs, &inputs, std::slice::from_ref(&trace.output))
}

fn trace_loop_boundary<T>(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<T>,
) -> Result<ResolvedLoopBoundary, DiamondCompileError> {
    let outputs =
        trace.outputs.iter().map(|output| refs.wire(output)).collect::<Result<Vec<_>, _>>()?;
    let operation = outputs
        .first()
        .ok_or_else(|| certificate_error("construction loop has no output"))?
        .node
        .clone();
    if outputs.iter().any(|output| output.node != operation) {
        return Err(certificate_error("construction loop outputs have different producers"));
    }
    let parent = refs
        .graph
        .scope(&operation.scope)
        .ok_or_else(|| certificate_error("construction loop parent scope is missing"))?;
    let node = parent
        .node(operation.node)
        .ok_or_else(|| certificate_error("construction loop node is missing"))?;
    let body_scope = refs
        .graph
        .child_scope_id(&operation.scope, operation.node)
        .ok_or_else(|| certificate_error("construction loop body scope is missing"))?;
    let body = refs
        .graph
        .scope(&body_scope)
        .ok_or_else(|| certificate_error("construction loop body is missing"))?;
    let arguments = parent
        .arguments(node)
        .ok_or_else(|| certificate_error("construction loop arguments are missing"))?
        .iter()
        .enumerate()
        .map(|(index, wire)| {
            operation.operand(index as u32, refs.scoped_wire(&operation.scope, *wire))
        })
        .collect();
    let body_inputs =
        body.inputs().iter().map(|wire| refs.scoped_wire(&body_scope, *wire)).collect();
    let body_outputs =
        body.outputs().iter().map(|wire| refs.scoped_wire(&body_scope, *wire)).collect();
    Ok(ResolvedLoopBoundary {
        operation,
        body_scope,
        arguments,
        body_inputs,
        body_outputs,
        outputs,
    })
}

fn trace_parallel_loop<T>(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<T>,
) -> Result<ParallelLoopRef, DiamondCompileError> {
    let boundary = trace_loop_boundary(refs, trace)?;
    let scope = refs
        .graph
        .scope(&boundary.operation.scope)
        .ok_or_else(|| certificate_error("parallel loop parent scope is missing"))?;
    let node = scope
        .node(boundary.operation.node)
        .ok_or_else(|| certificate_error("parallel loop node is missing"))?;
    let NodeKind::ParallelLoop(specification) = node.kind() else {
        return Err(certificate_error("construction trace does not resolve to a parallel loop"));
    };
    Ok(ParallelLoopRef {
        operation: boundary.operation,
        body_scope: boundary.body_scope,
        count: specification.count.clone(),
        index_slot: specification.index_slot,
        bindings: specification.bindings.clone(),
        input_modes: specification.input_modes.iter().map(CertifiedLoopInputMode::from).collect(),
        arguments: boundary.arguments,
        body_inputs: boundary.body_inputs,
        body_outputs: boundary.body_outputs,
        outputs: boundary.outputs,
    })
}

fn trace_sequential_loop<T>(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<T>,
) -> Result<SequentialLoopRef, DiamondCompileError> {
    let boundary = trace_loop_boundary(refs, trace)?;
    let scope = refs
        .graph
        .scope(&boundary.operation.scope)
        .ok_or_else(|| certificate_error("sequential loop parent scope is missing"))?;
    let node = scope
        .node(boundary.operation.node)
        .ok_or_else(|| certificate_error("sequential loop node is missing"))?;
    let NodeKind::SequentialLoop(specification) = node.kind() else {
        return Err(certificate_error("construction trace does not resolve to a sequential loop"));
    };
    Ok(SequentialLoopRef {
        operation: boundary.operation,
        body_scope: boundary.body_scope,
        count: specification.count.clone(),
        index_slot: specification.index_slot,
        bindings: specification.bindings.clone(),
        carried_count: specification.carried_count,
        arguments: boundary.arguments,
        body_inputs: boundary.body_inputs,
        body_outputs: boundary.body_outputs,
        outputs: boundary.outputs,
    })
}

fn trace_dynamic_get(
    refs: &CertificateRefBuilder<'_>,
    trace: &GatherConstructionTrace,
) -> Result<DynamicFamilyGetRef, DiamondCompileError> {
    if trace.sources.len() != 1 || trace.outputs.len() != 1 {
        return Err(certificate_error("scalar gather trace has an invalid arity"));
    }
    let output = refs.wire(&trace.outputs[0])?;
    let operation = output.node.clone();
    Ok(DynamicFamilyGetRef {
        family: refs.operand(&operation, 0, &trace.sources[0])?,
        index: refs.operand(&operation, 1, &trace.index)?,
        operation,
        output,
    })
}

fn trace_dynamic_get_trace(
    refs: &CertificateRefBuilder<'_>,
    trace: &DynamicFamilyGetConstructionTrace,
) -> Result<DynamicFamilyGetRef, DiamondCompileError> {
    let output = refs.wire(&trace.output)?;
    let operation = output.node.clone();
    Ok(DynamicFamilyGetRef {
        family: refs.operand(&operation, 0, &trace.family)?,
        index: refs.operand(&operation, 1, &trace.index)?,
        operation,
        output,
    })
}

fn trace_parallel_gather(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<GatherConstructionTrace>,
) -> Result<ParallelGatherRef, DiamondCompileError> {
    let parallel_loop = trace_parallel_loop(refs, trace)?;
    let body = &trace.scope.body;
    if body.sources.len() != body.outputs.len() ||
        parallel_loop.arguments.len() != body.sources.len() + 1
    {
        return Err(certificate_error("parallel gather trace has an invalid arity"));
    }
    let gets = body
        .sources
        .iter()
        .zip(&body.outputs)
        .map(|(source, output)| {
            trace_dynamic_get(
                refs,
                &GatherConstructionTrace {
                    index: body.index.clone(),
                    sources: vec![source.clone()],
                    outputs: vec![output.clone()],
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ParallelGatherRef {
        index_family: parallel_loop.arguments[0].clone(),
        source_families: parallel_loop.arguments[1..].to_vec(),
        body_index: refs.wire(&body.index)?,
        body_sources: body
            .sources
            .iter()
            .map(|source| refs.wire(source))
            .collect::<Result<Vec<_>, _>>()?,
        gets,
        output_families: parallel_loop.outputs.clone(),
        parallel_loop,
    })
}

fn trace_parallel_family_get(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<GatherConstructionTrace>,
) -> Result<ParallelFamilyGetRef, DiamondCompileError> {
    let gather = trace_parallel_gather(refs, trace)?;
    if gather.source_families.len() != 1 || gather.gets.len() != 1 {
        return Err(certificate_error("matrix family gather does not have one source"));
    }
    Ok(ParallelFamilyGetRef {
        index_family: gather.index_family,
        source_family: gather.source_families[0].clone(),
        body_index: gather.body_index,
        body_source: gather.body_sources[0].clone(),
        get: gather.gets[0].clone(),
        output_family: gather.output_families[0].clone(),
        parallel_loop: gather.parallel_loop,
    })
}

fn trace_preimage(
    refs: &CertificateRefBuilder<'_>,
    trace: &PreimageConstructionTrace,
) -> Result<PreimageRef, DiamondCompileError> {
    Ok(PreimageRef {
        sample: trace_operation(refs, &trace.sample)?,
        materialize: trace_operation(refs, &trace.materialize)?,
    })
}

fn trace_parallel_preimage(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<PreimageConstructionTrace>,
) -> Result<ParallelPreimageRef, DiamondCompileError> {
    Ok(ParallelPreimageRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body: trace_preimage(refs, &trace.scope.body)?,
    })
}

fn trace_matrix_binary(
    refs: &CertificateRefBuilder<'_>,
    left: &ValueHandle,
    right: &ValueHandle,
    output_handle: &ValueHandle,
) -> Result<MatrixBinaryRef, DiamondCompileError> {
    let output = refs.wire(output_handle)?;
    let operation = output.node.clone();
    Ok(MatrixBinaryRef {
        left: refs.operand(&operation, 0, left)?,
        right: refs.operand(&operation, 1, right)?,
        operation,
        output,
    })
}

fn trace_parallel_matrix_binary(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<MatrixBinaryConstructionTrace>,
) -> Result<ParallelMatrixBinaryRef, DiamondCompileError> {
    trace_parallel_matrix_binary_parts(
        refs,
        trace,
        &trace.scope.body.left,
        &trace.scope.body.right,
        &trace.scope.body.output,
    )
}

fn trace_parallel_state_product(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<MatrixProductConstructionTrace>,
) -> Result<ParallelMatrixBinaryRef, DiamondCompileError> {
    trace_parallel_matrix_binary_parts(
        refs,
        trace,
        &trace.scope.body.left,
        &trace.scope.body.right,
        &trace.scope.body.output,
    )
}

fn trace_parallel_matrix_binary_parts<T>(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<T>,
    left: &ValueHandle,
    right: &ValueHandle,
    output: &ValueHandle,
) -> Result<ParallelMatrixBinaryRef, DiamondCompileError> {
    let parallel_loop = trace_parallel_loop(refs, trace)?;
    if parallel_loop.arguments.len() != 2 || parallel_loop.body_inputs.len() != 2 {
        return Err(certificate_error("parallel matrix binary trace has an invalid arity"));
    }
    Ok(ParallelMatrixBinaryRef {
        parallel_loop: parallel_loop.operation.clone(),
        body_scope: parallel_loop.body_scope.clone(),
        left_family: parallel_loop.arguments[0].clone(),
        right_family: parallel_loop.arguments[1].clone(),
        body_left: parallel_loop.body_inputs[0].clone(),
        body_right: parallel_loop.body_inputs[1].clone(),
        operation: trace_matrix_binary(refs, left, right, output)?,
        body_output: refs.wire(output)?,
        output_family: parallel_loop.outputs[0].clone(),
    })
}

fn trace_operation_from_ordered(
    refs: &CertificateRefBuilder<'_>,
    trace: &OrderedOperationConstructionTrace,
) -> Result<OperationRef, DiamondCompileError> {
    trace_operation(
        refs,
        &OperationConstructionTrace {
            inputs: trace.inputs.clone(),
            outputs: vec![trace.output.clone()],
        },
    )
}

fn trace_binary_node(
    refs: &CertificateRefBuilder<'_>,
    trace: &OrderedOperationConstructionTrace,
) -> Result<BinaryNodeRef, DiamondCompileError> {
    let operation = trace_operation_from_ordered(refs, trace)?;
    if operation.inputs.len() != 2 || operation.outputs.len() != 1 {
        return Err(certificate_error("binary construction trace has an invalid arity"));
    }
    Ok(BinaryNodeRef {
        operation: operation.operation,
        left: operation.inputs[0].clone(),
        right: operation.inputs[1].clone(),
        output: operation.outputs[0].clone(),
    })
}

fn trace_unary_node(
    refs: &CertificateRefBuilder<'_>,
    trace: &OrderedOperationConstructionTrace,
) -> Result<UnaryNodeRef, DiamondCompileError> {
    let operation = trace_operation_from_ordered(refs, trace)?;
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(certificate_error("unary construction trace has an invalid arity"));
    }
    Ok(UnaryNodeRef {
        operation: operation.operation,
        input: operation.inputs[0].clone(),
        output: operation.outputs[0].clone(),
    })
}

fn trace_ordered_matrix_binary(
    refs: &CertificateRefBuilder<'_>,
    trace: &OrderedOperationConstructionTrace,
) -> Result<MatrixBinaryRef, DiamondCompileError> {
    if trace.inputs.len() != 2 {
        return Err(certificate_error("matrix binary construction trace has an invalid arity"));
    }
    trace_matrix_binary(refs, &trace.inputs[0], &trace.inputs[1], &trace.output)
}

fn trace_stage_interface(
    refs: &CertificateRefBuilder<'_>,
    inputs: &[NamedValueConstructionTrace],
    outputs: &[NamedValueConstructionTrace],
) -> Result<StageInterfaceLayout, DiamondCompileError> {
    let mut inputs = inputs
        .iter()
        .map(|input| {
            Ok(StageInputLayout { name: input.name.clone(), node: refs.wire(&input.value)?.node })
        })
        .collect::<Result<Vec<_>, DiamondCompileError>>()?;
    inputs.sort_by_key(|input| input.node.node);
    let mut outputs = outputs
        .iter()
        .map(|output| {
            Ok(StageOutputLayout { name: output.name.clone(), wire: refs.wire(&output.value)? })
        })
        .collect::<Result<Vec<_>, DiamondCompileError>>()?;
    outputs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(StageInterfaceLayout { stage: refs.stage.clone(), inputs, outputs })
}

fn trace_artifact_provenance(
    producer: &StageInterfaceLayout,
    consumer: &StageInterfaceLayout,
    name: &str,
) -> Result<ArtifactProvenance, DiamondCompileError> {
    let producer_output =
        producer.outputs.iter().find(|output| output.name == name).cloned().ok_or_else(|| {
            certificate_error(format!("artifact producer output `{name}` is missing"))
        })?;
    let consumer_input = consumer
        .inputs
        .iter()
        .find(|input| artifact_name_from_consumer_input(&input.name) == name)
        .cloned()
        .ok_or_else(|| certificate_error(format!("artifact consumer input `{name}` is missing")))?;
    Ok(ArtifactProvenance {
        producer_stage: producer.stage.clone(),
        producer_output,
        consumer_stage: consumer.stage.clone(),
        consumer_input,
    })
}

fn trace_workflow(
    encrypt: &CertificateRefBuilder<'_>,
    decrypt: &CertificateRefBuilder<'_>,
    encryption: &DiamondEncryptionConstructionTrace,
    decryption: &DiamondDecryptionConstructionTrace,
) -> Result<DiamondWorkflowLayout, DiamondCompileError> {
    let encryption = trace_stage_interface(encrypt, &encryption.inputs, &encryption.outputs)?;
    let decryption = trace_stage_interface(decrypt, &decryption.inputs, &decryption.outputs)?;
    let artifacts = [
        DiamondArtifactNames::INITIAL_STATE,
        DiamondArtifactNames::ONE_PREIMAGE,
        DiamondArtifactNames::K_PREIMAGE,
        DiamondArtifactNames::DECODER_PREIMAGE,
        DiamondArtifactNames::R_DECOMPOSED,
        DiamondArtifactNames::TRANSITIONS,
        DiamondArtifactNames::WITNESS_PREIMAGES,
        DiamondArtifactNames::PUBLIC_KEYS,
    ]
    .into_iter()
    .map(|name| trace_artifact_provenance(&encryption, &decryption, name))
    .collect::<Result<Vec<_>, _>>()?;
    Ok(DiamondWorkflowLayout { encryption, decryption, artifacts })
}

fn trace_artifact<'a>(
    workflow: &'a DiamondWorkflowLayout,
    name: &str,
) -> Result<&'a ArtifactProvenance, DiamondCompileError> {
    workflow
        .artifacts
        .iter()
        .find(|artifact| artifact.producer_output.name == name)
        .ok_or_else(|| certificate_error(format!("artifact provenance `{name}` is missing")))
}

fn trace_parallel_operation(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<OperationConstructionTrace>,
) -> Result<ParallelOperationRef, DiamondCompileError> {
    Ok(ParallelOperationRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body: trace_operation(refs, &trace.scope.body)?,
    })
}

fn trace_parallel_index_formula(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<ValueHandle>,
) -> Result<ParallelIndexFormulaRef, DiamondCompileError> {
    Ok(ParallelIndexFormulaRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body_output: refs.wire(&trace.scope.body)?,
    })
}

fn trace_initial_state_expansion(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<SelectConstructionTrace>,
) -> Result<InitialStateExpansionRef, DiamondCompileError> {
    Ok(InitialStateExpansionRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body_output: refs.wire(&trace.scope.body.output)?,
    })
}

fn trace_witness_digit_packing(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<PackedBitsConstructionTrace>,
) -> Result<WitnessDigitPackingRef, DiamondCompileError> {
    Ok(WitnessDigitPackingRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body_output: refs.wire(&trace.scope.body.output)?,
        bit_scan: trace_sequential_loop(refs, &trace.scope.body.scan)?,
    })
}

fn trace_parallel_transition_target(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<TransitionTargetConstructionTrace>,
) -> Result<ParallelTransitionTargetRef, DiamondCompileError> {
    let body = &trace.scope.body;
    Ok(ParallelTransitionTargetRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body: TransitionTargetRef {
            digit_secret: refs.wire(&body.digit_secret)?,
            target_public: refs.wire(&body.target_public)?,
            selector: refs.wire(&body.selector)?,
            selector_construction: trace_transition_selector(refs, &body.selector_construction)?,
            error_sample: trace_operation(refs, &body.error_sample)?,
            selector_product: trace_operation(refs, &body.selector_product)?,
            target_sum: trace_operation(refs, &body.target_sum)?,
        },
    })
}

fn trace_transition_selector(
    refs: &CertificateRefBuilder<'_>,
    trace: &mxx_gadgets::input_injector::SelectorConstructionTrace,
) -> Result<TransitionSelectorLayout, DiamondCompileError> {
    let bit = &trace.bit_scan.scope.body;
    Ok(TransitionSelectorLayout {
        regular: trace_operation(refs, &trace.regular)?,
        k_identity: trace_operation(refs, &trace.k_identity)?,
        k: trace_operation(refs, &trace.k)?,
        initial_select: trace_select(refs, &trace.initial_select)?,
        bit_scan: trace_sequential_loop(refs, &trace.bit_scan)?,
        bit_body: TransitionSelectorBitLayout {
            bit_extract: trace_operation(refs, &bit.bit_extract)?,
            bit_to_int: trace_operation(refs, &bit.bit_to_int)?,
            bit_zero: trace_operation(refs, &bit.bit_zero)?,
            bit_one: trace_operation(refs, &bit.bit_one)?,
            bit_select: trace_select(refs, &bit.bit_select)?,
            special_product: trace_operation(refs, &bit.special_product)?,
            special_top: trace_operation(refs, &bit.special_top)?,
            special_bottom: trace_operation(refs, &bit.special_bottom)?,
            special: trace_operation(refs, &bit.special)?,
            state_match: trace_operation(refs, &bit.state_match)?,
            state_match_to_int: trace_operation(refs, &bit.state_match_to_int)?,
            selector: trace_select(refs, &bit.selector)?,
        },
    })
}

fn trace_input_preprocessing(
    refs: &CertificateRefBuilder<'_>,
    workflow: &DiamondWorkflowLayout,
    trace: &DiamondInputPreprocessingConstructionTrace,
) -> Result<DiamondInputPreprocessingLayout, DiamondCompileError> {
    Ok(DiamondInputPreprocessingLayout {
        initial_state_artifact: trace_artifact(workflow, DiamondArtifactNames::INITIAL_STATE)?
            .clone(),
        transitions_artifact: trace_artifact(workflow, DiamondArtifactNames::TRANSITIONS)?.clone(),
        trapdoor_samples: trace_parallel_operation(refs, &trace.trapdoor_samples)?,
        secret_sample: trace_operation(refs, &trace.secret_sample)?,
        message_selector: trace_operation(refs, &trace.message_selector)?,
        initial_error_sample: trace_operation(refs, &trace.initial_error_sample)?,
        initial_public_product: trace_operation(refs, &trace.initial_public_product)?,
        initial_state: trace_operation(refs, &trace.initial_state)?,
        transition_source_indices: trace_parallel_index_formula(
            refs,
            &trace.transition_source_indices,
        )?,
        transition_target_indices: trace_parallel_index_formula(
            refs,
            &trace.transition_target_indices,
        )?,
        digit_secret_indices: trace_parallel_index_formula(refs, &trace.digit_secret_indices)?,
        digit_secret_samples: trace_parallel_operation(refs, &trace.digit_secret_samples)?,
        digit_secrets: trace_parallel_gather(refs, &trace.digit_secrets)?,
        transition_sources: trace_parallel_gather(refs, &trace.transition_sources)?,
        target_public_matrices: trace_parallel_gather(refs, &trace.target_public_matrices)?,
        transition_targets: trace_parallel_transition_target(refs, &trace.transition_targets)?,
        transition_preimages: trace_parallel_preimage(refs, &trace.transition_preimages)?,
        final_indices: trace_parallel_loop(refs, &trace.final_indices)?,
        final_trapdoors: trace_parallel_gather(refs, &trace.final_trapdoors)?,
    })
}

fn trace_message_construction(
    refs: &CertificateRefBuilder<'_>,
    trace: &MessageConstructionTrace,
) -> Result<MessageConstructionLayout, DiamondCompileError> {
    Ok(MessageConstructionLayout {
        to_int: trace_operation(refs, &trace.to_int)?,
        zero: trace_operation(refs, &trace.zero)?,
        one: trace_operation(refs, &trace.one)?,
        select: trace_select(refs, &trace.select)?,
    })
}

fn trace_public_key_sampling(
    refs: &CertificateRefBuilder<'_>,
    workflow: &DiamondWorkflowLayout,
    trace: &BggPublicKeyFamilySamplingTrace,
) -> Result<BggPublicKeySamplingLayout, DiamondCompileError> {
    Ok(BggPublicKeySamplingLayout {
        public_keys_artifact: trace_artifact(workflow, DiamondArtifactNames::PUBLIC_KEYS)?.clone(),
        packed_hash: trace_operation_handles(
            refs,
            std::slice::from_ref(&trace.hash_key),
            std::slice::from_ref(&trace.packed),
        )?,
        slices: ParallelOperationRef {
            parallel_loop: trace_parallel_loop(refs, &trace.slices)?,
            body: trace_operation_handles(
                refs,
                std::slice::from_ref(&trace.slices.scope.body.packed),
                std::slice::from_ref(&trace.slices.scope.body.slice),
            )?,
        },
    })
}

fn trace_encryption_initial_public_keys(
    refs: &CertificateRefBuilder<'_>,
    trace: &EncryptionInitialPublicKeysConstructionTrace,
) -> Result<EncryptionInitialPublicKeysLayout, DiamondCompileError> {
    Ok(EncryptionInitialPublicKeysLayout {
        one_public_key: trace_operation(refs, &trace.one_public_key)?,
        zero_public_key: trace_operation(refs, &trace.zero_public_key)?,
        instance_width: trace_evaluate_int(
            refs,
            &trace.instance_width.expression,
            Some(&trace.instance_width.zero),
            &trace.instance_width.output,
        )?,
        public_indices: trace_parallel_loop(refs, &trace.public_indices)?,
        public_candidates: trace_parallel_gather(refs, &trace.public_candidates)?,
        packed_inputs: ParallelPackedPublicKeyLayout {
            parallel_loop: trace_parallel_loop(refs, &trace.packed_inputs)?,
            in_range: trace_select(refs, &trace.packed_inputs.scope.body.in_range)?,
            padded: trace_select(refs, &trace.packed_inputs.scope.body.padded)?,
        },
        circuit_inputs: ParallelCircuitInputPublicKeyLayout {
            parallel_loop: trace_parallel_loop(refs, &trace.circuit_inputs)?,
            selected_instance: trace_select(
                refs,
                &trace.circuit_inputs.scope.body.selected_instance,
            )?,
            selected_source: trace_select(refs, &trace.circuit_inputs.scope.body.selected_source)?,
        },
    })
}

fn trace_witness_target(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<WitnessTargetConstructionTrace>,
) -> Result<ParallelWitnessTargetLayout, DiamondCompileError> {
    Ok(ParallelWitnessTargetLayout {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        negated_gadget: trace_operation(refs, &trace.scope.body.negated_gadget)?,
        target: trace_operation(refs, &trace.scope.body.target)?,
    })
}

fn trace_artifact_preprocessing(
    refs: &CertificateRefBuilder<'_>,
    workflow: &DiamondWorkflowLayout,
    trace: &DiamondArtifactPreprocessingConstructionTrace,
) -> Result<DiamondArtifactPreprocessingLayout, DiamondCompileError> {
    Ok(DiamondArtifactPreprocessingLayout {
        one_preimage_artifact: trace_artifact(workflow, DiamondArtifactNames::ONE_PREIMAGE)?
            .clone(),
        witness_preimages_artifact: trace_artifact(
            workflow,
            DiamondArtifactNames::WITNESS_PREIMAGES,
        )?
        .clone(),
        k_preimage_artifact: trace_artifact(workflow, DiamondArtifactNames::K_PREIMAGE)?.clone(),
        r_decomposed_artifact: trace_artifact(workflow, DiamondArtifactNames::R_DECOMPOSED)?
            .clone(),
        decoder_preimage_artifact: trace_artifact(
            workflow,
            DiamondArtifactNames::DECODER_PREIMAGE,
        )?
        .clone(),
        projection_trapdoor: StaticTrapdoorLayout {
            public: trace_operation(refs, &trace.projection_trapdoor.public)?,
            secret: trace_operation(refs, &trace.projection_trapdoor.secret)?,
        },
        one_target: OneTargetLayout {
            gadget: trace_operation(refs, &trace.one_target.gadget)?,
            difference: trace_operation(refs, &trace.one_target.difference)?,
            zero_row: trace_operation(refs, &trace.one_target.zero_row)?,
            target: trace_operation(refs, &trace.one_target.target)?,
        },
        one_preimage: trace_preimage(refs, &trace.one_preimage)?,
        witness_indices: trace_parallel_loop(refs, &trace.witness_indices)?,
        witness_trapdoors: trace_parallel_gather(refs, &trace.witness_trapdoors)?,
        witness_public_keys: trace_parallel_gather(refs, &trace.witness_public_keys)?,
        witness_targets: trace_witness_target(refs, &trace.witness_targets)?,
        witness_preimages: trace_parallel_preimage(refs, &trace.witness_preimages)?,
        k_target: KTargetLayout {
            public_key_hash: trace_operation(refs, &trace.k_target.public_key_hash)?,
            first_column: trace_operation(refs, &trace.k_target.first_column)?,
            half_modulus: trace_operation(refs, &trace.k_target.half_modulus)?,
            target: trace_operation(refs, &trace.k_target.target)?,
        },
        k_preimage: trace_preimage(refs, &trace.k_preimage)?,
        r_hash: trace_operation(refs, &trace.r_hash)?,
        r_slice: trace_operation(refs, &trace.r_slice)?,
        r_decomposition: trace_operation(refs, &trace.r_decomposition)?,
        r_materialization: trace_operation(refs, &trace.r_materialization)?,
        r_reshape: trace_operation(refs, &trace.r_reshape)?,
        decoder_target: DecoderTargetLayout {
            public_key_difference: trace_operation(
                refs,
                &trace.decoder_target.public_key_difference,
            )?,
            projected_difference: trace_operation(
                refs,
                &trace.decoder_target.projected_difference,
            )?,
            public_key_sum: trace_operation(refs, &trace.decoder_target.public_key_sum)?,
            zero: trace_operation(refs, &trace.decoder_target.zero)?,
            target: trace_operation(refs, &trace.decoder_target.target)?,
        },
        decoder_preimage: trace_preimage(refs, &trace.decoder_preimage)?,
    })
}

fn trace_parallel_select_operation(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<SelectConstructionTrace>,
) -> Result<ParallelOperationRef, DiamondCompileError> {
    Ok(ParallelOperationRef {
        parallel_loop: trace_parallel_loop(refs, trace)?,
        body: trace_select(refs, &trace.scope.body)?,
    })
}

fn trace_encoding_component_operations(
    refs: &CertificateRefBuilder<'_>,
    trace: &EncodingSelectionConstructionTrace,
) -> Result<EncodingComponentOperationsLayout, DiamondCompileError> {
    Ok(EncodingComponentOperationsLayout {
        vectors: trace_parallel_select_operation(refs, &trace.vectors)?,
        public_keys: trace_parallel_select_operation(refs, &trace.public_keys)?,
        plaintexts: trace_parallel_select_operation(refs, &trace.plaintexts)?,
    })
}

fn trace_decryption_initial_encodings(
    refs: &CertificateRefBuilder<'_>,
    workflow: &DiamondWorkflowLayout,
    trace: &DecryptionInitialEncodingsConstructionTrace,
) -> Result<DecryptionInitialEncodingsLayout, DiamondCompileError> {
    let loops = |traces: &[LoopConstructionTrace<ValueHandle>]| {
        traces.iter().map(|trace| trace_parallel_loop(refs, trace)).collect::<Result<Vec<_>, _>>()
    };
    Ok(DecryptionInitialEncodingsLayout {
        initial_state_artifact: trace_artifact(workflow, DiamondArtifactNames::INITIAL_STATE)?
            .clone(),
        one_preimage_artifact: trace_artifact(workflow, DiamondArtifactNames::ONE_PREIMAGE)?
            .clone(),
        witness_preimages_artifact: trace_artifact(
            workflow,
            DiamondArtifactNames::WITNESS_PREIMAGES,
        )?
        .clone(),
        public_keys_artifact: trace_artifact(workflow, DiamondArtifactNames::PUBLIC_KEYS)?.clone(),
        witness_indices: trace_parallel_loop(refs, &trace.witness_indices)?,
        witness_bits: trace_parallel_gather(refs, &trace.witness_bits)?,
        witness_digits: trace_witness_digit_packing(refs, &trace.witness_digits)?,
        initial_projection_state: trace_operation(refs, &trace.initial_projection_state)?,
        one_public_key: trace_operation(refs, &trace.one_public_key)?,
        one_plaintext: trace_operation(refs, &trace.one_plaintext)?,
        zero_encoding: trace
            .zero_encoding
            .iter()
            .map(|operation| trace_operation(refs, operation))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| certificate_error("zero encoding trace has invalid arity"))?,
        witness_state_indices: trace_parallel_loop(refs, &trace.witness_state_indices)?,
        witness_states: trace_parallel_gather(refs, &trace.witness_states)?,
        witness_vectors: trace_parallel_matrix_binary(refs, &trace.witness_vectors)?,
        witness_public_indices: trace_parallel_loop(refs, &trace.witness_public_indices)?,
        witness_public_keys: trace_parallel_gather(refs, &trace.witness_public_keys)?,
        witness_plaintext_constants: loops(&trace.witness_plaintext_constants)?
            .try_into()
            .map_err(|_| certificate_error("witness plaintext trace has invalid arity"))?,
        witness_plaintexts: trace_parallel_select_operation(refs, &trace.witness_plaintexts)?,
        instance_width: trace_evaluate_int(
            refs,
            &trace.instance_width.expression,
            Some(&trace.instance_width.zero),
            &trace.instance_width.output,
        )?,
        packed_indices: trace_parallel_loop(refs, &trace.packed_indices)?,
        packed_vectors: trace_parallel_gather(refs, &trace.packed_vectors)?,
        packed_public_keys: trace_parallel_gather(refs, &trace.packed_public_keys)?,
        packed_plaintexts: trace_parallel_gather(refs, &trace.packed_plaintexts)?,
        active_witness: trace_parallel_loop(refs, &trace.active_witness)?,
        active_witness_zeroes: loops(&trace.active_witness_zeroes)?
            .try_into()
            .map_err(|_| certificate_error("active witness zero trace has invalid arity"))?,
        active_witness_selection: trace_encoding_component_operations(
            refs,
            &trace.active_witness_selection,
        )?,
        instance_constants: trace
            .instance_constants
            .iter()
            .map(|pair| {
                loops(pair)?.try_into().map_err(|_| {
                    certificate_error("instance constant trace has invalid inner arity")
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| certificate_error("instance constant trace has invalid outer arity"))?,
        selected_instance: trace_encoding_component_operations(refs, &trace.selected_instance)?,
        active_instance: trace_parallel_loop(refs, &trace.active_instance)?,
        circuit_inputs: trace_encoding_component_operations(refs, &trace.circuit_inputs)?,
    })
}

fn trace_input_injection(
    refs: &CertificateRefBuilder<'_>,
    trace: &DiamondInputEvaluationConstructionTrace,
) -> Result<InputInjectionLayout, DiamondCompileError> {
    let boundary = trace_loop_boundary(refs, &trace.state_scan)?;
    let body = &trace.state_scan.scope.body;
    if boundary.arguments.len() < 3 || boundary.outputs.len() != 1 {
        return Err(certificate_error("input-injection scan has an invalid boundary"));
    }
    Ok(InputInjectionLayout {
        state_scan: boundary.operation.clone(),
        body_scope: boundary.body_scope,
        initial_states_expansion: trace_initial_state_expansion(
            refs,
            &trace.initial_states_expansion,
        )?,
        initial_states: refs.operand(&boundary.operation, 0, &trace.initial_states)?,
        packed_digits: refs.operand(&boundary.operation, 1, &trace.packed_digits)?,
        transition_family: refs.operand(&boundary.operation, 2, &trace.transitions)?,
        final_states: boundary.outputs[0].clone(),
        body_initial_states: refs.wire(&body.body_states)?,
        body_packed_digits: refs.wire(&body.body_packed_digits)?,
        body_transition_family: refs.wire(&body.body_transitions)?,
        selected_digit: trace_dynamic_get(refs, &body.selected_digit)?,
        source_indices: trace_parallel_index_formula(refs, &body.source_indices)?,
        source_states: trace_parallel_family_get(refs, &body.source_states)?,
        transition_indices: trace_parallel_index_formula(refs, &body.transition_indices)?,
        selected_transitions: trace_parallel_family_get(refs, &body.selected_transitions)?,
        body_final_states: refs.wire(&body.body_output)?,
        state_product: trace_parallel_state_product(refs, &body.state_products)?,
    })
}

fn trace_evaluate_int(
    refs: &CertificateRefBuilder<'_>,
    expression: &ValueHandle,
    zero: Option<&ValueHandle>,
    output: &ValueHandle,
) -> Result<EvaluateIntRef, DiamondCompileError> {
    let evaluated = refs.wire(expression)?;
    let scope = refs
        .graph
        .scope(&evaluated.node.scope)
        .ok_or_else(|| certificate_error("EvaluateInt scope is missing"))?;
    let node = scope
        .node(evaluated.node.node)
        .ok_or_else(|| certificate_error("EvaluateInt node is missing"))?;
    let NodeKind::EvaluateInt(int_expression) = node.kind() else {
        return Err(certificate_error("construction trace does not resolve to EvaluateInt"));
    };
    let materialization = match zero {
        Some(zero) => Some(trace_binary_node(
            refs,
            &OrderedOperationConstructionTrace {
                inputs: vec![expression.clone(), zero.clone()],
                output: output.clone(),
            },
        )?),
        None => None,
    };
    Ok(EvaluateIntRef {
        operation: evaluated.node.clone(),
        expression: int_expression.clone(),
        evaluated,
        materialization,
        output: refs.wire(output)?,
    })
}

fn trace_capture_operand<T>(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<T>,
    explicit_count: usize,
    child: &ValueHandle,
) -> Result<(CoreOperandRef, CoreWireRef, ValueHandle), DiamondCompileError> {
    let (index, capture) = trace
        .scope
        .captures
        .iter()
        .enumerate()
        .find(|(_, capture)| &capture.child_placeholder == child)
        .ok_or_else(|| certificate_error("construction loop capture is missing"))?;
    let boundary = trace_loop_boundary(refs, trace)?;
    Ok((
        refs.operand(&boundary.operation, explicit_count + index, &capture.parent_source)?,
        refs.wire(&capture.child_placeholder)?,
        capture.parent_source.clone(),
    ))
}

fn trace_boolean_metadata(
    refs: &CertificateRefBuilder<'_>,
    sequential: &CoreNodeRef,
    operand_base: usize,
    sequential_sources: [(&ValueHandle, &ValueHandle); 4],
    active_gate_count: &GatherConstructionTrace,
    metadata: &LayerMetadataConstructionTrace,
) -> Result<BooleanLayerMetadataLayout, DiamondCompileError> {
    let layer_index =
        trace_evaluate_int(refs, &active_gate_count.index, None, &active_gate_count.index)?;
    let selected = trace_dynamic_get(refs, active_gate_count)?;
    let scalar = LayerScalarMetadataRef {
        source_input_name: "circuit-active-gate-count".to_owned(),
        root_input: refs.wire(sequential_sources[0].0)?,
        sequential_operand: refs.operand(sequential, operand_base, sequential_sources[0].0)?,
        body_source: refs.wire(sequential_sources[0].1)?,
        layer_index,
        selected,
    };
    let family = |source_name: &str,
                  operand_index: usize,
                  root: &ValueHandle,
                  body_source: &ValueHandle,
                  flattened: &LoopConstructionTrace<mxx_dsl::EvaluateIntConstructionTrace>,
                  gathered: &LoopConstructionTrace<GatherConstructionTrace>|
     -> Result<LayerFamilyMetadataRef, DiamondCompileError> {
        let flattened_index = &flattened.scope.body;
        Ok(LayerFamilyMetadataRef {
            source_input_name: source_name.to_owned(),
            root_input: refs.wire(root)?,
            sequential_operand: refs.operand(sequential, operand_index, root)?,
            body_source: refs.wire(body_source)?,
            flattened_indices: trace_parallel_loop(refs, flattened)?,
            flattened_index: trace_evaluate_int(
                refs,
                &flattened_index.expression,
                Some(&flattened_index.zero),
                &flattened_index.output,
            )?,
            gathered: trace_parallel_family_get(refs, gathered)?,
        })
    };
    Ok(BooleanLayerMetadataLayout {
        active_gate_count: scalar,
        opcode: family(
            "circuit-gate-kind",
            operand_base + 1,
            sequential_sources[1].0,
            sequential_sources[1].1,
            &metadata.flattened_indices,
            &metadata.gate_kinds,
        )?,
        left_source: family(
            "circuit-left-source",
            operand_base + 2,
            sequential_sources[2].0,
            sequential_sources[2].1,
            &metadata.flattened_indices,
            &metadata.left_sources,
        )?,
        right_source: family(
            "circuit-right-source",
            operand_base + 3,
            sequential_sources[3].0,
            sequential_sources[3].1,
            &metadata.flattened_indices,
            &metadata.right_sources,
        )?,
    })
}

fn trace_six_way_select<const N: usize>(
    refs: &CertificateRefBuilder<'_>,
    trace: &mxx_gadgets::circuit::boolean_dsl::SelectConstructionTrace<N>,
) -> Result<SixWaySelectRef, DiamondCompileError> {
    if N != 6 {
        return Err(certificate_error("six-way selection trace has the wrong arity"));
    }
    let output = refs.wire(&trace.output)?;
    let operation = output.node.clone();
    let branches = trace
        .branches
        .iter()
        .enumerate()
        .map(|(index, branch)| refs.operand(&operation, index + 1, branch))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| certificate_error("six-way selection trace has the wrong arity"))?;
    Ok(SixWaySelectRef {
        selector: refs.operand(&operation, 0, &trace.selector)?,
        branches,
        operation,
        output,
    })
}

fn trace_two_way_select<const N: usize>(
    refs: &CertificateRefBuilder<'_>,
    trace: &mxx_gadgets::circuit::boolean_dsl::SelectConstructionTrace<N>,
) -> Result<TwoWaySelectRef, DiamondCompileError> {
    if N != 2 {
        return Err(certificate_error("two-way selection trace has the wrong arity"));
    }
    let output = refs.wire(&trace.output)?;
    let operation = output.node.clone();
    let branches = trace
        .branches
        .iter()
        .enumerate()
        .map(|(index, branch)| refs.operand(&operation, index + 1, branch))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| certificate_error("two-way selection trace has the wrong arity"))?;
    Ok(TwoWaySelectRef {
        selector: refs.operand(&operation, 0, &trace.selector)?,
        branches,
        operation,
        output,
    })
}

fn trace_parallel_select(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<mxx_dsl::SelectConstructionTrace>,
) -> Result<(ParallelLoopRef, DynamicSelectParts), DiamondCompileError> {
    let parallel_loop = trace_parallel_loop(refs, trace)?;
    let selected = trace_select(refs, &trace.scope.body)?;
    let [selector, branches @ ..] = selected.inputs.as_slice() else {
        return Err(certificate_error("selection trace has no selector"));
    };
    Ok((
        parallel_loop,
        DynamicSelectParts {
            operation: selected.operation,
            selector: selector.clone(),
            branches: branches.to_vec(),
            output: selected.outputs[0].clone(),
        },
    ))
}

struct DynamicSelectParts {
    operation: CoreNodeRef,
    selector: CoreOperandRef,
    branches: Vec<CoreOperandRef>,
    output: CoreWireRef,
}

fn trace_parallel_six_way_select(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<mxx_dsl::SelectConstructionTrace>,
) -> Result<ParallelSixWaySelectRef, DiamondCompileError> {
    let (parallel_loop, select) = trace_parallel_select(refs, trace)?;
    let branches: [CoreOperandRef; 6] = select
        .branches
        .try_into()
        .map_err(|_| certificate_error("parallel six-way selection has the wrong arity"))?;
    let body_branches = branches.each_ref().map(|branch| branch.wire.clone());
    let branch_families = parallel_loop.arguments[1..].to_vec().try_into().map_err(|_| {
        certificate_error("parallel six-way selection boundary has the wrong arity")
    })?;
    Ok(ParallelSixWaySelectRef {
        parallel_loop: parallel_loop.operation.clone(),
        body_scope: parallel_loop.body_scope.clone(),
        selector_family: parallel_loop.arguments[0].clone(),
        branch_families,
        body_selector: select.selector.wire.clone(),
        body_branches,
        select: SixWaySelectRef {
            operation: select.operation,
            selector: select.selector,
            branches,
            output: select.output.clone(),
        },
        body_output: select.output,
        output_family: parallel_loop.outputs[0].clone(),
    })
}

fn trace_parallel_two_way_select(
    refs: &CertificateRefBuilder<'_>,
    trace: &LoopConstructionTrace<mxx_dsl::SelectConstructionTrace>,
) -> Result<ParallelTwoWaySelectRef, DiamondCompileError> {
    let (parallel_loop, select) = trace_parallel_select(refs, trace)?;
    let branches: [CoreOperandRef; 2] = select
        .branches
        .try_into()
        .map_err(|_| certificate_error("parallel two-way selection has the wrong arity"))?;
    let body_branches = branches.each_ref().map(|branch| branch.wire.clone());
    let branch_families = parallel_loop.arguments[1..].to_vec().try_into().map_err(|_| {
        certificate_error("parallel two-way selection boundary has the wrong arity")
    })?;
    Ok(ParallelTwoWaySelectRef {
        parallel_loop: parallel_loop.operation.clone(),
        body_scope: parallel_loop.body_scope.clone(),
        selector_family: parallel_loop.arguments[0].clone(),
        branch_families,
        body_selector: select.selector.wire.clone(),
        body_branches,
        select: TwoWaySelectRef {
            operation: select.operation,
            selector: select.selector,
            branches,
            output: select.output.clone(),
        },
        body_output: select.output,
        output_family: parallel_loop.outputs[0].clone(),
    })
}

fn trace_local_decomposition(
    refs: &CertificateRefBuilder<'_>,
    trace: &DecompositionConstructionTrace,
) -> Result<LocalGadgetDecompositionRef, DiamondCompileError> {
    let decomposition = refs.wire(&trace.decomposition)?;
    let operation = decomposition.node.clone();
    Ok(LocalGadgetDecompositionRef {
        decomposition_node: operation.clone(),
        right_public_key: refs.operand(&operation, 0, &trace.input)?,
        base: operation.parameter(mxx_correctness::CoreNodeParameter::GadgetDecomposeBase),
        digit_count: operation
            .parameter(mxx_correctness::CoreNodeParameter::GadgetDecomposeDigitCount),
        decomposition,
        materialized: refs.wire(&trace.materialized)?,
    })
}

fn trace_public_key_boolean(
    refs: &CertificateRefBuilder<'_>,
    trace: &PublicKeyBooleanConstructionTrace,
    selected_output: &DynamicFamilyGetConstructionTrace,
) -> Result<
    (PublicKeyBooleanLoopLayout, EncryptPublicKeyRhsDecomposition, LocalBooleanGateLayout),
    DiamondCompileError,
> {
    let layers = &trace.layers;
    let boundary = trace_loop_boundary(refs, &layers.layer_scan)?;
    let body = &layers.layer_scan.scope.body;
    let gate_loop = &body.gate_slots;
    let gate = &gate_loop.scope.body;
    let gate_one_parent = gate_loop
        .scope
        .captures
        .iter()
        .find(|capture| capture.child_placeholder == gate.gate.one)
        .ok_or_else(|| certificate_error("public-key one capture is missing"))?
        .parent_source
        .clone();
    let (one_public_key, body_one_public_key, _) =
        trace_capture_operand(refs, &layers.layer_scan, 5, &gate_one_parent)?;
    let metadata = trace_boolean_metadata(
        refs,
        &boundary.operation,
        1,
        [
            (&layers.active_gate_counts, &body.body_active_gate_counts),
            (&layers.gate_kinds, &body.body_gate_kinds),
            (&layers.left_sources, &body.body_left_sources),
            (&layers.right_sources, &body.body_right_sources),
        ],
        &body.active_gate_count,
        &body.metadata,
    )?;
    let loop_layout = PublicKeyBooleanLoopLayout {
        layer_scan: boundary.operation.clone(),
        body_scope: boundary.body_scope,
        initial_public_keys: refs.operand(&boundary.operation, 0, &layers.initial_state)?,
        active_gate_counts: refs.operand(&boundary.operation, 1, &layers.active_gate_counts)?,
        gate_kinds: refs.operand(&boundary.operation, 2, &layers.gate_kinds)?,
        left_sources: refs.operand(&boundary.operation, 3, &layers.left_sources)?,
        right_sources: refs.operand(&boundary.operation, 4, &layers.right_sources)?,
        one_public_key,
        final_public_keys: boundary.outputs[0].clone(),
        body_initial_public_keys: refs.wire(&body.body_state)?,
        body_active_gate_counts: refs.wire(&body.body_active_gate_counts)?,
        body_gate_kinds: refs.wire(&body.body_gate_kinds)?,
        body_left_sources: refs.wire(&body.body_left_sources)?,
        body_right_sources: refs.wire(&body.body_right_sources)?,
        body_one_public_key,
        body_final_public_keys: refs.wire(&body.body_output)?,
        metadata,
        selected_output: trace_dynamic_get_trace(refs, selected_output)?,
    };
    let parent_loop = trace_parallel_loop(refs, gate_loop)?;
    let (_, body_one, _) = trace_capture_operand(refs, gate_loop, 3, &gate.gate.one)?;
    let (active_gate_count, body_active_gate_count, _) =
        trace_capture_operand(refs, gate_loop, 3, &gate.active_gate_count)?;
    let zero = trace_matrix_binary(refs, &gate.gate.one, &gate.gate.one, &gate.gate.zero)?;
    let not = trace_matrix_binary(refs, &gate.gate.one, &gate.gate.left, &gate.gate.not)?;
    let product = trace_matrix_binary(
        refs,
        &gate.gate.left,
        &gate.gate.right_decomposition_materialized,
        &gate.gate.product,
    )?;
    let sum = trace_matrix_binary(refs, &gate.gate.left, &gate.gate.right, &gate.gate.sum)?;
    let two_product = trace_matrix_binary(
        refs,
        &gate.gate.product,
        &gate.gate.two_scalar,
        &gate.gate.two_product,
    )?;
    let xor = trace_matrix_binary(refs, &gate.gate.sum, &gate.gate.two_product, &gate.gate.xor)?;
    let gate_layout = LocalBooleanGateLayout {
        body_scope: parent_loop.body_scope.clone(),
        opcode_family: parent_loop.arguments[0].clone(),
        left_family: parent_loop.arguments[1].clone(),
        right_family: parent_loop.arguments[2].clone(),
        one_public_key: trace_capture_operand(refs, gate_loop, 3, &gate.gate.one)?.0,
        active_gate_count,
        left_selection: trace_parallel_family_get(refs, &body.left_values)?,
        body_opcode: refs.wire(&gate.opcode)?,
        body_left: refs.wire(&gate.left)?,
        body_right: refs.wire(&gate.right)?,
        body_one_public_key: body_one,
        body_active_gate_count,
        zero,
        one: refs.wire(&gate.gate.one)?,
        copy: refs.wire(&gate.gate.left)?,
        not,
        product: product.clone(),
        sum,
        two_product,
        xor,
        candidate_select: trace_six_way_select(refs, &gate.candidate_select)?,
        active_select: trace_two_way_select(refs, &gate.active_select)?,
        parent_loop,
    };
    let decomp_trace = DecompositionConstructionTrace {
        input: gate.gate.right.clone(),
        decomposition: gate.gate.right_decomposition.clone(),
        materialized: gate.gate.right_decomposition_materialized.clone(),
    };
    let decomposition = EncryptPublicKeyRhsDecomposition {
        right_selection: trace_parallel_family_get(refs, &body.right_values)?,
        enclosing_parallel_loop: gate_layout.parent_loop.operation.clone(),
        body_scope: gate_layout.body_scope.clone(),
        right_public_key_family: gate_layout.right_family.clone(),
        body_right_public_key: gate_layout.body_right.clone(),
        local: trace_local_decomposition(refs, &decomp_trace)?,
        multiplication_consumer: product.right,
    };
    Ok((loop_layout, decomposition, gate_layout))
}

fn trace_sequential_one<T>(
    refs: &CertificateRefBuilder<'_>,
    layers: &LoopConstructionTrace<T>,
    repetition: &LoopConstructionTrace<ValueHandle>,
) -> Result<(CoreOperandRef, CoreWireRef), DiamondCompileError> {
    let body_one = repetition
        .scope
        .captures
        .first()
        .ok_or_else(|| certificate_error("encoding one repetition capture is missing"))?
        .parent_source
        .clone();
    let (outer, inner, _) = trace_capture_operand(refs, layers, 7, &body_one)?;
    Ok((outer, inner))
}

fn trace_parallel_consumer(operation: &ParallelMatrixBinaryRef) -> ParallelDecompositionConsumer {
    ParallelDecompositionConsumer {
        consumer_loop: operation.parallel_loop.clone(),
        decomposition_family: operation.right_family.clone(),
        body_scope: operation.body_scope.clone(),
        body_decomposition: operation.body_right.clone(),
        multiplication_consumer: operation.operation.right.clone(),
    }
}

struct EncodingComponentInputs<'a> {
    state_input: &'a ValueHandle,
    state_output: &'a ValueHandle,
    left: &'a LoopConstructionTrace<GatherConstructionTrace>,
    right: &'a LoopConstructionTrace<GatherConstructionTrace>,
    one_repetition: &'a LoopConstructionTrace<ValueHandle>,
    one_family: &'a ValueHandle,
    copy_family: &'a ValueHandle,
    zero: &'a LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    not: &'a LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    product: FamilyProductRef,
    sum: &'a LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    two_product: &'a LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    xor: &'a LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    candidate: &'a LoopConstructionTrace<mxx_dsl::SelectConstructionTrace>,
    active_mask: &'a LoopConstructionTrace<ValueHandle>,
    active: &'a LoopConstructionTrace<mxx_dsl::SelectConstructionTrace>,
}

fn trace_encoding_component(
    refs: &CertificateRefBuilder<'_>,
    inputs: EncodingComponentInputs<'_>,
) -> Result<FamilyBooleanGateLayout, DiamondCompileError> {
    let candidate_select = trace_parallel_six_way_select(refs, inputs.candidate)?;
    let active_select = trace_parallel_two_way_select(refs, inputs.active)?;
    Ok(FamilyBooleanGateLayout {
        state_input: refs.wire(inputs.state_input)?,
        state_output: refs.wire(inputs.state_output)?,
        left_selection: trace_parallel_family_get(refs, inputs.left)?,
        right_selection: trace_parallel_family_get(refs, inputs.right)?,
        opcode_family: candidate_select.selector_family.wire.clone(),
        active_family: active_select.selector_family.wire.clone(),
        zero: trace_parallel_matrix_binary(refs, inputs.zero)?,
        one_repetition: trace_parallel_loop(refs, inputs.one_repetition)?,
        one_family: refs.wire(inputs.one_family)?,
        copy_family: refs.wire(inputs.copy_family)?,
        not: trace_parallel_matrix_binary(refs, inputs.not)?,
        product: inputs.product,
        sum: trace_parallel_matrix_binary(refs, inputs.sum)?,
        two_product: trace_parallel_matrix_binary(refs, inputs.two_product)?,
        xor: trace_parallel_matrix_binary(refs, inputs.xor)?,
        candidate_select,
        active_mask: trace_parallel_loop(refs, inputs.active_mask)?,
        active_select,
    })
}

fn trace_encoding_boolean(
    refs: &CertificateRefBuilder<'_>,
    trace: &EncodingBooleanConstructionTrace,
    selected_output: &DynamicFamilyGetConstructionTrace,
) -> Result<
    (
        EncodingBooleanLoopLayout,
        DecryptEncodingRhsDecomposition,
        FamilyBooleanGateLayout,
        FamilyBooleanGateLayout,
        FamilyBooleanGateLayout,
    ),
    DiamondCompileError,
> {
    let boundary = trace_loop_boundary(refs, &trace.layers)?;
    let body = &trace.layers.scope.body;
    let (one_vector, body_one_vector) =
        trace_sequential_one(refs, &trace.layers, &body.one_repetition.vectors)?;
    let (one_public_key, body_one_public_key) =
        trace_sequential_one(refs, &trace.layers, &body.one_repetition.public_keys)?;
    let (one_plaintext, body_one_plaintext) =
        trace_sequential_one(refs, &trace.layers, &body.one_repetition.plaintexts)?;
    let metadata = trace_boolean_metadata(
        refs,
        &boundary.operation,
        3,
        [
            (&trace.active_gate_counts, &body.body_active_gate_counts),
            (&trace.gate_kinds, &body.body_gate_kinds),
            (&trace.left_sources, &body.body_left_sources),
            (&trace.right_sources, &body.body_right_sources),
        ],
        &body.active_gate_count,
        &body.metadata,
    )?;
    let loop_layout = EncodingBooleanLoopLayout {
        layer_scan: boundary.operation.clone(),
        body_scope: boundary.body_scope,
        initial_vectors: refs.operand(&boundary.operation, 0, &trace.initial_vectors)?,
        initial_public_keys: refs.operand(&boundary.operation, 1, &trace.initial_public_keys)?,
        initial_plaintexts: refs.operand(&boundary.operation, 2, &trace.initial_plaintexts)?,
        active_gate_counts: refs.operand(&boundary.operation, 3, &trace.active_gate_counts)?,
        gate_kinds: refs.operand(&boundary.operation, 4, &trace.gate_kinds)?,
        left_sources: refs.operand(&boundary.operation, 5, &trace.left_sources)?,
        right_sources: refs.operand(&boundary.operation, 6, &trace.right_sources)?,
        one_vector,
        one_public_key,
        one_plaintext,
        final_vectors: boundary.outputs[0].clone(),
        final_public_keys: boundary.outputs[1].clone(),
        final_plaintexts: boundary.outputs[2].clone(),
        body_initial_vectors: refs.wire(&body.body_vectors)?,
        body_initial_public_keys: refs.wire(&body.body_public_keys)?,
        body_initial_plaintexts: refs.wire(&body.body_plaintexts)?,
        body_active_gate_counts: refs.wire(&body.body_active_gate_counts)?,
        body_gate_kinds: refs.wire(&body.body_gate_kinds)?,
        body_left_sources: refs.wire(&body.body_left_sources)?,
        body_right_sources: refs.wire(&body.body_right_sources)?,
        body_one_vector,
        body_one_public_key,
        body_one_plaintext,
        body_final_vectors: refs.wire(&body.output_vectors)?,
        body_final_public_keys: refs.wire(&body.output_public_keys)?,
        body_final_plaintexts: refs.wire(&body.output_plaintexts)?,
        metadata,
        selected_vector: trace_dynamic_get_trace(refs, selected_output)?,
    };

    let pk_product = trace_parallel_matrix_binary(refs, &body.multiplication.public_keys)?;
    let vector_left = trace_parallel_matrix_binary(
        refs,
        &body.multiplication.left_vectors_times_right_decompositions,
    )?;
    let vector_right = trace_parallel_matrix_binary(
        refs,
        &body.multiplication.right_vectors_times_left_plaintexts,
    )?;
    let vector_sum = trace_parallel_matrix_binary(refs, &body.multiplication.vectors)?;
    let plaintext_product = trace_parallel_matrix_binary(refs, &body.multiplication.plaintexts)?;
    let vectors = trace_encoding_component(
        refs,
        EncodingComponentInputs {
            state_input: &body.body_vectors,
            state_output: &body.output_vectors,
            left: &body.left_gather.vectors,
            right: &body.right_gather.vectors,
            one_repetition: &body.one_repetition.vectors,
            one_family: &body.one_vectors,
            copy_family: &body.left_vectors,
            zero: &body.zero_operations.vectors,
            not: &body.not_operations.vectors,
            product: FamilyProductRef::EncodingVector {
                left_times_right_decomposition: vector_left.clone(),
                right_times_left_plaintext: vector_right,
                sum: vector_sum,
            },
            sum: &body.sum_operations.vectors,
            two_product: &body.two_product_operations.vectors,
            xor: &body.xor_operations.vectors,
            candidate: &body.candidate_selection.vectors,
            active_mask: &body.active_mask_loop,
            active: &body.active_selection.vectors,
        },
    )?;
    let public_keys = trace_encoding_component(
        refs,
        EncodingComponentInputs {
            state_input: &body.body_public_keys,
            state_output: &body.output_public_keys,
            left: &body.left_gather.public_keys,
            right: &body.right_gather.public_keys,
            one_repetition: &body.one_repetition.public_keys,
            one_family: &body.one_public_keys,
            copy_family: &body.left_public_keys,
            zero: &body.zero_operations.public_keys,
            not: &body.not_operations.public_keys,
            product: FamilyProductRef::Direct(pk_product.clone()),
            sum: &body.sum_operations.public_keys,
            two_product: &body.two_product_operations.public_keys,
            xor: &body.xor_operations.public_keys,
            candidate: &body.candidate_selection.public_keys,
            active_mask: &body.active_mask_loop,
            active: &body.active_selection.public_keys,
        },
    )?;
    let plaintexts = trace_encoding_component(
        refs,
        EncodingComponentInputs {
            state_input: &body.body_plaintexts,
            state_output: &body.output_plaintexts,
            left: &body.left_gather.plaintexts,
            right: &body.right_gather.plaintexts,
            one_repetition: &body.one_repetition.plaintexts,
            one_family: &body.one_plaintexts,
            copy_family: &body.left_plaintexts,
            zero: &body.zero_operations.plaintexts,
            not: &body.not_operations.plaintexts,
            product: FamilyProductRef::Direct(plaintext_product),
            sum: &body.sum_operations.plaintexts,
            two_product: &body.two_product_operations.plaintexts,
            xor: &body.xor_operations.plaintexts,
            candidate: &body.candidate_selection.plaintexts,
            active_mask: &body.active_mask_loop,
            active: &body.active_selection.plaintexts,
        },
    )?;

    let decomposition_loop =
        trace_parallel_loop(refs, &body.multiplication.right_public_key_decompositions)?;
    let decomp_body = &body.multiplication.right_public_key_decompositions.scope.body;
    let decomposition = DecryptEncodingRhsDecomposition {
        right_selection: trace_parallel_family_get(refs, &body.right_gather.public_keys)?,
        decomposition_loop: decomposition_loop.operation.clone(),
        body_scope: decomposition_loop.body_scope.clone(),
        right_public_key_family: decomposition_loop.arguments[0].clone(),
        body_right_public_key: refs.wire(&decomp_body.input)?,
        local: trace_local_decomposition(refs, decomp_body)?,
        body_output: refs.wire(&decomp_body.materialized)?,
        decomposition_family: decomposition_loop.outputs[0].clone(),
        public_key_consumer: trace_parallel_consumer(&pk_product),
        vector_consumer: trace_parallel_consumer(&vector_left),
    };
    Ok((loop_layout, decomposition, vectors, public_keys, plaintexts))
}

fn trace_decoder(
    refs: &CertificateRefBuilder<'_>,
    trace: &DecoderConstructionTrace,
) -> Result<DecoderLayout, DiamondCompileError> {
    let one_vector = trace_ordered_matrix_binary(refs, &trace.one_vector)?;
    let k_vector = trace_ordered_matrix_binary(refs, &trace.k_vector)?;
    let decoder_vector = trace_ordered_matrix_binary(refs, &trace.decoder_vector)?;
    let one_minus_circuit = trace_ordered_matrix_binary(refs, &trace.one_minus_circuit)?;
    let projected_difference = trace_ordered_matrix_binary(refs, &trace.projected_difference)?;
    let k_plus_projection = trace_ordered_matrix_binary(refs, &trace.k_plus_projection)?;
    let residual = trace_ordered_matrix_binary(refs, &trace.residual)?;
    Ok(DecoderLayout {
        one_preimage: one_vector.right.wire.clone(),
        k_preimage: k_vector.right.wire.clone(),
        decoder_preimage: decoder_vector.right.wire.clone(),
        r_decomposed: projected_difference.right.wire.clone(),
        selected_circuit_vector: one_minus_circuit.right.wire.clone(),
        one_vector,
        k_vector,
        decoder_vector,
        one_minus_circuit,
        projected_difference,
        k_plus_projection,
        residual,
        extract_coefficient: trace_unary_node(refs, &trace.extract_coefficient)?,
        threshold: trace_evaluate_int(refs, &trace.threshold, None, &trace.threshold)?,
        lower_compare: trace_binary_node(refs, &trace.lower_compare)?,
        upper_scale: trace_binary_node(refs, &trace.upper_scale)?,
        upper_compare: trace_binary_node(refs, &trace.upper_compare)?,
        lower_to_int: trace_unary_node(refs, &trace.lower_to_int)?,
        upper_to_int: trace_unary_node(refs, &trace.upper_to_int)?,
        comparison_sum: trace_binary_node(refs, &trace.comparison_sum)?,
        equals_two: trace_binary_node(refs, &trace.equals_two)?,
        decoded: refs.wire(&trace.decoded)?,
    })
}

fn build_diamond_certificate_from_traces(
    encrypt_id: &StageId,
    decrypt_id: &StageId,
    encryption: &DiamondEncryptionBuild,
    decryption: &DiamondDecryptionBuild,
) -> Result<DiamondCertificate, DiamondCompileError> {
    let encrypt = CertificateRefBuilder {
        stage: encrypt_id.clone(),
        graph: &encryption.graph.graph.graph,
        freeze_map: &encryption.freeze_map,
    };
    let decrypt = CertificateRefBuilder {
        stage: decrypt_id.clone(),
        graph: &decryption.graph.graph.graph,
        freeze_map: &decryption.freeze_map,
    };
    let workflow = trace_workflow(&encrypt, &decrypt, &encryption.trace, &decryption.trace)?;
    let (encryption_loop, encrypt_decomposition, encryption_gate) = trace_public_key_boolean(
        &encrypt,
        &encryption.trace.boolean_layers,
        &encryption.trace.selected_circuit_output,
    )?;
    let (
        decryption_loop,
        decrypt_decomposition,
        decryption_vectors,
        decryption_public_keys,
        decryption_plaintexts,
    ) = trace_encoding_boolean(
        &decrypt,
        &decryption.trace.boolean_layers,
        &decryption.trace.selected_circuit_output,
    )?;
    Ok(DiamondCertificate {
        message: Box::new(trace_message_construction(&encrypt, &encryption.trace.message)?),
        public_key_sampling: Box::new(trace_public_key_sampling(
            &encrypt,
            &workflow,
            &encryption.trace.public_key_sampling,
        )?),
        encryption_initial_public_keys: Box::new(trace_encryption_initial_public_keys(
            &encrypt,
            &encryption.trace.initial_public_keys,
        )?),
        input_preprocessing: trace_input_preprocessing(
            &encrypt,
            &workflow,
            &encryption.trace.preprocessing,
        )?,
        artifact_preprocessing: Box::new(trace_artifact_preprocessing(
            &encrypt,
            &workflow,
            &encryption.trace.artifact_preprocessing,
        )?),
        input_injection: trace_input_injection(&decrypt, &decryption.trace.input_injection)?,
        decryption_initial_encodings: Box::new(trace_decryption_initial_encodings(
            &decrypt,
            &workflow,
            &decryption.trace.initial_encodings,
        )?),
        boolean_layers: BooleanLayersLayout {
            public_keys_artifact: trace_artifact(&workflow, DiamondArtifactNames::PUBLIC_KEYS)?
                .clone(),
            encryption: encryption_loop,
            decryption: decryption_loop,
            encrypt_public_key_rhs_decomposition: encrypt_decomposition,
            decrypt_encoding_rhs_decomposition: decrypt_decomposition,
            encryption_gate: Box::new(encryption_gate),
            decryption_vectors: Box::new(decryption_vectors),
            decryption_public_keys: Box::new(decryption_public_keys),
            decryption_plaintexts: Box::new(decryption_plaintexts),
        },
        decoder: trace_decoder(&decrypt, &decryption.trace.decoder)?,
        workflow,
    })
}

impl WitnessEncryptionProtocol for DiamondWeCompiler {
    type Error = DiamondCompileError;

    fn protocol_decl(&self) -> Result<WitnessEncryptionProtocolDecl, Self::Error> {
        DiamondWeCompiler::protocol_decl(self)
    }
}

impl WitnessEncryptionProtocol for DiamondWeProtocolFamily {
    type Error = DiamondCompileError;

    fn protocol_decl(&self) -> Result<WitnessEncryptionProtocolDecl, Self::Error> {
        DiamondWeProtocolFamily::protocol_decl(self)
    }
}

fn decode_boolean_interval(
    noisy_plaintext: Mat,
    modulus: IntExpr,
) -> (Bool, DecoderTailConstructionTrace) {
    let extract_inputs = vec![noisy_plaintext.value_handle().clone()];
    let coefficient = noisy_plaintext.extract_coefficient(0);
    let extract_coefficient = OrderedOperationConstructionTrace {
        inputs: extract_inputs,
        output: coefficient.value_handle().clone(),
    };
    let quarter = Int::evaluate(IntExpr::RoundDiv(
        Box::new(IntExpr::Sub(Box::new(modulus), Box::new(IntExpr::constant(2)))),
        Box::new(IntExpr::constant(4)),
    ));
    let threshold = quarter.value_handle().clone();
    let three = Int::constant(3);
    let upper_scale_inputs = vec![quarter.value_handle().clone(), three.value_handle().clone()];
    let upper = quarter.clone().mul(three);
    let upper_scale = OrderedOperationConstructionTrace {
        inputs: upper_scale_inputs,
        output: upper.value_handle().clone(),
    };
    let lower_compare_inputs =
        vec![quarter.value_handle().clone(), coefficient.value_handle().clone()];
    let lower_ok = quarter.less_equal(coefficient.clone());
    let lower_compare = OrderedOperationConstructionTrace {
        inputs: lower_compare_inputs,
        output: lower_ok.value_handle().clone(),
    };
    let upper_compare_inputs =
        vec![coefficient.value_handle().clone(), upper.value_handle().clone()];
    let upper_ok = coefficient.less_equal(upper);
    let upper_compare = OrderedOperationConstructionTrace {
        inputs: upper_compare_inputs,
        output: upper_ok.value_handle().clone(),
    };
    let lower_to_int_inputs = vec![lower_ok.value_handle().clone()];
    let lower_int = lower_ok.to_int();
    let lower_to_int = OrderedOperationConstructionTrace {
        inputs: lower_to_int_inputs,
        output: lower_int.value_handle().clone(),
    };
    let upper_to_int_inputs = vec![upper_ok.value_handle().clone()];
    let upper_int = upper_ok.to_int();
    let upper_to_int = OrderedOperationConstructionTrace {
        inputs: upper_to_int_inputs,
        output: upper_int.value_handle().clone(),
    };
    let comparison_sum_inputs =
        vec![lower_int.value_handle().clone(), upper_int.value_handle().clone()];
    let comparison = lower_int.add(upper_int);
    let comparison_sum = OrderedOperationConstructionTrace {
        inputs: comparison_sum_inputs,
        output: comparison.value_handle().clone(),
    };
    let two = Int::constant(2);
    let equals_two_inputs = vec![comparison.value_handle().clone(), two.value_handle().clone()];
    let decoded = comparison.equal(two);
    let equals_two = OrderedOperationConstructionTrace {
        inputs: equals_two_inputs,
        output: decoded.value_handle().clone(),
    };
    (
        decoded,
        DecoderTailConstructionTrace {
            extract_coefficient,
            threshold,
            lower_compare,
            upper_scale,
            upper_compare,
            lower_to_int,
            upper_to_int,
            comparison_sum,
            equals_two,
        },
    )
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

    fn protocol_and_certificate() -> (WitnessEncryptionProtocolDecl, Box<DiamondCertificate>) {
        let mut declaration = compiler().protocol_decl().expect("Diamond protocol declaration");
        let SemanticCertificate::Diamond(certificate) =
            std::mem::take(&mut declaration.protocol.semantic_certificate)
        else {
            panic!("Diamond protocol must carry a Diamond certificate");
        };
        (declaration, certificate)
    }

    #[test]
    fn certificate_rejects_wrong_operand_direction() {
        let (declaration, mut certificate) = protocol_and_certificate();
        certificate.message.select.inputs.swap(1, 2);
        assert!(certificate.validate_references(declaration.protocol()).is_err());
    }

    #[test]
    fn certificate_rejects_wrong_loop_wiring() {
        let (declaration, mut certificate) = protocol_and_certificate();
        certificate
            .encryption_initial_public_keys
            .public_candidates
            .parallel_loop
            .body_inputs
            .swap(0, 1);
        assert!(certificate.validate_references(declaration.protocol()).is_err());
    }

    #[test]
    fn certificate_rejects_wrong_artifact_provenance() {
        let (declaration, mut certificate) = protocol_and_certificate();
        certificate.artifact_preprocessing.one_preimage_artifact.producer_output =
            certificate.artifact_preprocessing.k_preimage_artifact.producer_output.clone();
        assert!(certificate.validate_references(declaration.protocol()).is_err());
    }

    #[test]
    fn certificate_rejects_wrong_decoder_wiring() {
        let (declaration, mut certificate) = protocol_and_certificate();
        certificate.decoder.one_preimage = certificate.decoder.k_preimage.clone();
        assert!(certificate.validate_references(declaration.protocol()).is_err());
    }

    #[test]
    fn certificate_rejects_index_formula_output_outside_its_body() {
        let (declaration, mut certificate) = protocol_and_certificate();
        certificate.input_preprocessing.transition_source_indices.body_output =
            certificate.input_preprocessing.transition_source_indices.parallel_loop.outputs[0]
                .clone();
        assert!(certificate.validate_references(declaration.protocol()).is_err());
    }

    #[test]
    fn certificate_rejects_wrong_decoder_threshold_expression() {
        let (declaration, mut certificate) = protocol_and_certificate();
        certificate.decoder.threshold.expression = IntExpr::constant(0);
        assert!(certificate.validate_references(declaration.protocol()).is_err());
    }

    #[test]
    fn construction_traces_resolve_exact_stage_nodes() {
        let family = DiamondWeProtocolFamily::new(b"trace-resolution".to_vec());
        let encryption = family.build_encryption().expect("encryption graph");
        for handle in [
            &encryption.trace.message.select.output,
            &encryption.trace.public_key_sampling.packed,
            &encryption.trace.initial_public_keys.circuit_inputs.outputs[0],
            &encryption
                .trace
                .preprocessing
                .transition_targets
                .scope
                .body
                .selector_construction
                .bit_scan
                .scope
                .body
                .special
                .outputs[0],
            &encryption.trace.artifact_preprocessing.one_preimage.sample.outputs[0],
            &encryption.trace.artifact_preprocessing.witness_targets.outputs[0],
            &encryption.trace.artifact_preprocessing.decoder_target.target.outputs[0],
        ] {
            encryption
                .freeze_map
                .resolve_unique(handle)
                .expect("encryption construction handle resolves exactly");
        }

        let production = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
        let decryption = family.build_decryption(production).expect("decryption graph");
        for handle in [
            &decryption.trace.initial_encodings.witness_digits.outputs[0],
            &decryption.trace.initial_encodings.witness_vectors.outputs[0],
            &decryption.trace.initial_encodings.active_witness_selection.vectors.outputs[0],
            &decryption.trace.initial_encodings.selected_instance.public_keys.outputs[0],
            &decryption.trace.initial_encodings.circuit_inputs.plaintexts.outputs[0],
            &decryption.trace.boolean_layers.layers.outputs[0],
            &decryption.trace.decoder.decoded,
        ] {
            decryption
                .freeze_map
                .resolve_unique(handle)
                .expect("decryption construction handle resolves exactly");
        }
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
        assert_eq!(declaration.protocol().correctness.requires.len(), 3);
        assert_eq!(declaration.protocol().params.len(), 15);
        assert!(declaration.protocol().correctness.protocol_inputs.iter().any(
            |(name, destinations)| name.0 == BOOLEAN_WITNESS_INPUT &&
                destinations ==
                    &vec![(
                        StageId("decrypt".to_owned()),
                        StageInputName(BOOLEAN_WITNESS_INPUT.to_owned()),
                    )]
        ));
        assert!(
            declaration.protocol().correctness.protocol_inputs.iter().any(
                |(name, destinations)| name.0 == "circuit-gate-kind" && destinations.len() == 2
            )
        );
        let emitted = mxx_correctness::emit_protocol_for(
            "diamond-we-family",
            declaration.protocol(),
            "MxxWe",
        )
        .unwrap();
        assert!(emitted.statement.contains("structure DiamondWeFamilyCorrectStatement"));
        assert!(emitted.statement.contains("accepts_valid_parameters :"));
        assert!(!emitted.proof_scaffold.contains("fun _ => false"));
        assert!(
            emitted
                .ir
                .contains(".roundDivide (.parameter \"diamond_modulus\") (.constant (2 : Int))")
        );
        assert!(!emitted.ir.contains(
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
        )
        .unwrap();
        let second = mxx_correctness::emit_protocol_for(
            "diamond-we-family",
            second.protocol_decl().unwrap().protocol(),
            "MxxWe",
        )
        .unwrap();
        let direct =
            mxx_correctness::emit_protocol_for("diamond-we-family", direct.protocol(), "MxxWe")
                .unwrap();
        assert_eq!(first.protocol_hash, second.protocol_hash);
        assert_eq!(first.protocol_hash, direct.protocol_hash);
    }
}
