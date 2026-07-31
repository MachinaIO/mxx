//! Graph-IR construction for the Diamond input-injection portion of
//! Diamond witness encryption.
//!
//! The builders in this module reproduce the level/state transition formulas
//! used by `DiamondInjector::preprocess` and `DiamondInjector::online_eval`.

use mxx_bgg::{
    BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, CircuitCompileError,
    PolyCircuitCompiler,
};
use mxx_gadgets::{Poly, circuit::PolyCircuit};
use mxx_ir_core::{
    Graph, GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, OutputFamilyError, ParamEnv, Port,
    SubgraphBuildError, TrapdoorFamilyWire, WireRef,
    artifact::{ArtifactConfidentiality, ProductionId},
    expr::{ExprError, RealExpr},
    graph::{CompileParameter, CompileParameterKind},
    node::{
        ConcatAxis, ConstantMatrix, HashVariant, IndexRange, IntBinaryOp, IntCompareOp,
        LoopInputMode, MatrixBinaryOp, NodeKind, SampleRange,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use thiserror::Error;

const PREFIX_SIZE: usize = 2;
const FINAL_CHECKPOINT_FAMILY: &str = "diamond_final_checkpoint_family";
const SECRET_SIZE: usize = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInjectorGraphConfig {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub state_columns: IntExpr,
    pub concrete_state_columns: usize,
    pub preimage_chunk_columns: usize,
    pub input_count: usize,
    pub base: usize,
    pub batch_bits: usize,
    pub trapdoor_sigma: RealExpr,
    pub gadget_base: IntExpr,
    pub gadget_digit_count: IntExpr,
    pub error_sigma: RealExpr,
}

#[derive(Debug, Error)]
pub enum DiamondInjectorGraphError {
    #[error("Diamond input count, base, batch width, and state columns must be positive")]
    NonPositiveParameter,
    #[error("Diamond base must be at least 2^batch_bits")]
    BaseTooSmall,
    #[error("Diamond state count overflow")]
    StateCountOverflow,
    #[error(transparent)]
    OutputFamily(#[from] OutputFamilyError),
    #[error(transparent)]
    Circuit(#[from] CircuitCompileError),
    #[error(transparent)]
    Expression(#[from] ExprError),
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
    #[error("Diamond WE witness and instance sizes do not match the circuit")]
    CircuitArity,
    #[error("Diamond WE witness size does not match the injector bit capacity")]
    WitnessSize,
    #[error("Diamond WE currently requires exactly one circuit output")]
    CircuitOutput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondWEGraphConfig {
    pub injector: DiamondInjectorGraphConfig,
    pub witness_size: usize,
    pub instance_size: usize,
    pub bgg_columns: IntExpr,
    pub concrete_bgg_columns: usize,
    pub gadget_base: IntExpr,
    pub bgg_tag: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondWEArtifactNames {
    pub injector: DiamondInjectorArtifactNames,
    pub public_keys: String,
    pub r: String,
    pub r_decomposed: String,
    pub one_preimage: String,
    pub witness_preimages: String,
    pub k_preimage: String,
    pub decoder_preimage: String,
}

impl Default for DiamondWEArtifactNames {
    fn default() -> Self {
        Self {
            injector: DiamondInjectorArtifactNames::default(),
            public_keys: "we_public_keys".to_owned(),
            r: "we_r".to_owned(),
            r_decomposed: "we_r_decomposed".to_owned(),
            one_preimage: "we_one_preimage".to_owned(),
            witness_preimages: "we_witness_preimages".to_owned(),
            k_preimage: "we_k_preimage".to_owned(),
            decoder_preimage: "we_decoder_preimage".to_owned(),
        }
    }
}

impl DiamondWEArtifactNames {
    pub fn one_preimage_chunk(&self, chunk: usize) -> String {
        format!("{}_chunk_{chunk}", self.one_preimage)
    }

    pub fn witness_preimages_chunk(&self, chunk: usize) -> String {
        format!("{}_chunk_{chunk}", self.witness_preimages)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInjectorArtifactNames {
    pub initial_state: String,
    pub transition_prefix: String,
}

impl Default for DiamondInjectorArtifactNames {
    fn default() -> Self {
        Self {
            initial_state: "diamond_initial_state".to_owned(),
            transition_prefix: "diamond_transition".to_owned(),
        }
    }
}

impl DiamondInjectorArtifactNames {
    pub fn transition_chunk(&self, level: usize, state: usize, chunk: usize) -> String {
        format!("{}_level_{level}_state_{state}_chunk_{chunk}", self.transition_prefix)
    }

    pub fn final_state(&self, state: usize) -> String {
        format!("diamond_final_state_{state}")
    }
}

impl DiamondInjectorGraphConfig {
    pub fn validate(&self) -> Result<(), DiamondInjectorGraphError> {
        if self.input_count == 0 ||
            self.base == 0 ||
            self.batch_bits == 0 ||
            self.concrete_state_columns == 0 ||
            self.preimage_chunk_columns == 0
        {
            return Err(DiamondInjectorGraphError::NonPositiveParameter);
        }
        let required_base = 1usize
            .checked_shl(
                self.batch_bits.try_into().map_err(|_| DiamondInjectorGraphError::BaseTooSmall)?,
            )
            .ok_or(DiamondInjectorGraphError::BaseTooSmall)?;
        if self.base < required_base {
            return Err(DiamondInjectorGraphError::BaseTooSmall);
        }
        self.state_count(self.input_count)?;
        Ok(())
    }

    pub fn state_count(&self, level: usize) -> Result<usize, DiamondInjectorGraphError> {
        level
            .checked_mul(self.batch_bits)
            .and_then(|count| count.checked_add(1))
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)
    }

    fn first_new_state(&self, level: usize) -> Result<usize, DiamondInjectorGraphError> {
        level
            .checked_sub(1)
            .and_then(|level| level.checked_mul(self.batch_bits))
            .and_then(|offset| offset.checked_add(1))
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)
    }

    fn new_bit(
        &self,
        level: usize,
        state: usize,
    ) -> Result<Option<usize>, DiamondInjectorGraphError> {
        let first = self.first_new_state(level)?;
        let end = first
            .checked_add(self.batch_bits)
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
        Ok(if (first..end).contains(&state) { Some(state - first) } else { None })
    }

    fn scalar_type(&self) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        }
    }

    fn state_type(&self) -> MatrixType {
        MatrixType { columns: self.state_columns.clone(), ..self.scalar_type() }
    }

    fn public_type(&self) -> MatrixType {
        MatrixType {
            rows: IntExpr::constant(PREFIX_SIZE * SECRET_SIZE),
            columns: self.state_columns.clone(),
            ..self.scalar_type()
        }
    }

    fn selector_type(&self) -> MatrixType {
        MatrixType {
            rows: IntExpr::constant(PREFIX_SIZE * SECRET_SIZE),
            columns: IntExpr::constant(PREFIX_SIZE * SECRET_SIZE),
            ..self.scalar_type()
        }
    }

    fn transition_chunk_count(&self) -> usize {
        self.concrete_state_columns.div_ceil(self.preimage_chunk_columns)
    }

    fn transition_chunk_bounds(
        &self,
        chunk: usize,
    ) -> Result<IndexRange, DiamondInjectorGraphError> {
        let start = chunk
            .checked_mul(self.preimage_chunk_columns)
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
        let end = start
            .checked_add(self.preimage_chunk_columns)
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?
            .min(self.concrete_state_columns);
        if start >= end {
            return Err(DiamondInjectorGraphError::StateCountOverflow);
        }
        Ok(IndexRange { start, end })
    }

    fn transition_chunk_type(&self, bounds: IndexRange) -> MatrixType {
        MatrixType {
            rows: self.state_columns.clone(),
            columns: IntExpr::constant(bounds.end - bounds.start),
            ..self.scalar_type()
        }
    }
}

impl DiamondWEGraphConfig {
    fn projection_chunk_count(&self) -> usize {
        self.concrete_bgg_columns.div_ceil(self.injector.preimage_chunk_columns)
    }

    fn projection_chunk_bounds(
        &self,
        chunk: usize,
    ) -> Result<IndexRange, DiamondInjectorGraphError> {
        let start = chunk
            .checked_mul(self.injector.preimage_chunk_columns)
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
        let end = start
            .checked_add(self.injector.preimage_chunk_columns)
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?
            .min(self.concrete_bgg_columns);
        if start >= end {
            return Err(DiamondInjectorGraphError::StateCountOverflow);
        }
        Ok(IndexRange { start, end })
    }

    fn projection_chunk_type(&self, bounds: IndexRange) -> MatrixType {
        MatrixType {
            rows: self.injector.state_columns.clone(),
            columns: IntExpr::constant(bounds.end - bounds.start),
            ..self.injector.scalar_type()
        }
    }
}

/// Builds the input-injector preprocessing graph. The `k` input is the 1x1
/// plaintext matrix embedded in the empty-prefix state.
pub fn build_preprocessing_graph(
    config: &DiamondInjectorGraphConfig,
    names: &DiamondInjectorArtifactNames,
) -> Result<Graph, DiamondInjectorGraphError> {
    config.validate()?;
    let mut builder = GraphBuilder::new("diamond-we-injector-preprocess", Vec::new());
    let scalar_type = config.scalar_type();
    let state_type = config.state_type();
    let public_type = config.public_type();
    let selector_type = config.selector_type();
    let k = builder.input("k", scalar_type.clone());

    let mut previous_checkpoints = sample_checkpoint_family(&mut builder, config, 0)?;
    let initial_checkpoint =
        builder.trapdoor_family_get_static(&previous_checkpoints, IntExpr::constant(0));

    let secret_epsilon = ternary(&mut builder, scalar_type.clone());
    let initial_selector = builder.concat(
        ConcatAxis::Columns,
        &[secret_epsilon, k],
        MatrixType { columns: IntExpr::constant(PREFIX_SIZE), ..scalar_type.clone() },
    );
    let initial_product = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &initial_selector,
        &initial_checkpoint.public,
        state_type.clone(),
    );
    let initial_error = builder.gaussian_sample(state_type.clone(), config.error_sigma.clone());
    let initial_state = builder.matrix_binary(
        MatrixBinaryOp::Add,
        &initial_product,
        &initial_error,
        state_type.clone(),
    );
    builder.output(names.initial_state.clone(), &initial_state, ArtifactConfidentiality::Public);

    for level in 1..=config.input_count {
        let state_count = config.state_count(level)?;
        let current_checkpoints = sample_checkpoint_family(&mut builder, config, level)?;
        let first_new = config.first_new_state(level)?;
        let masks = (0..config.base)
            .map(|_| ternary(&mut builder, scalar_type.clone()))
            .collect::<Vec<_>>();
        for state in 0..state_count {
            let destination_checkpoint =
                builder.trapdoor_family_get_static(&current_checkpoints, IntExpr::constant(state));
            let selectors = masks
                .iter()
                .enumerate()
                .map(|(digit, mask)| {
                    if let Some(bit) = config.new_bit(level, state)? {
                        Ok(special_selector(
                            &mut builder,
                            mask,
                            (digit >> bit) & 1,
                            &scalar_type,
                            &selector_type,
                        ))
                    } else if state == 0 {
                        Ok(diagonal_selector(
                            &mut builder,
                            mask,
                            true,
                            &scalar_type,
                            &selector_type,
                        ))
                    } else {
                        Ok(diagonal_selector(
                            &mut builder,
                            mask,
                            false,
                            &scalar_type,
                            &selector_type,
                        ))
                    }
                })
                .collect::<Result<Vec<_>, DiamondInjectorGraphError>>()?;
            let selectors = builder.family_pack(&selectors)?;
            let source_state = if state >= first_new { 0 } else { state };
            for chunk in 0..config.transition_chunk_count() {
                let bounds = config.transition_chunk_bounds(chunk)?;
                let target_type = MatrixType {
                    columns: IntExpr::constant(bounds.end - bounds.start),
                    ..public_type.clone()
                };
                let transition_type = config.transition_chunk_type(bounds);
                let destination_chunk = builder.slice(
                    &destination_checkpoint.public,
                    None,
                    Some(bounds),
                    target_type.clone(),
                );
                let mut body = GraphBuilder::new(
                    format!("diamond-transition-level-{level}-state-{state}-chunk-{chunk}"),
                    vec![CompileParameter {
                        name: "digit".to_owned(),
                        kind: CompileParameterKind::Integer,
                    }],
                );
                let selector = body.input("selector", selector_type.clone());
                let destination = body.input("destination", target_type.clone());
                let source = body.trapdoor_input(
                    "source",
                    public_type.clone(),
                    config.trapdoor_sigma.clone(),
                    config.gadget_base.clone(),
                    config.gadget_digit_count.clone(),
                );
                let target_product = body.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    &selector,
                    &destination,
                    target_type.clone(),
                );
                let error = body.gaussian_sample(target_type.clone(), config.error_sigma.clone());
                let target =
                    body.matrix_binary(MatrixBinaryOp::Add, &target_product, &error, target_type);
                let preimage = body.preimage_sample(&source, &target, transition_type.clone());
                body.value_output_wire("preimage", preimage.wire);
                let source_checkpoint = builder.trapdoor_family_get_static(
                    &previous_checkpoints,
                    IntExpr::constant(source_state),
                );
                let family = builder
                    .parallel_loop(
                        body.finish(),
                        IntExpr::constant(config.base),
                        "digit",
                        Vec::new(),
                        vec![selectors.wire, destination_chunk.wire, source_checkpoint.wire],
                        vec![
                            LoopInputMode::Zip,
                            LoopInputMode::Broadcast,
                            LoopInputMode::Broadcast,
                        ],
                        std::slice::from_ref(&transition_type),
                    )?
                    .into_iter()
                    .next()
                    .expect("one declared transition loop output");
                builder.output_family_wire(
                    names.transition_chunk(level, state, chunk),
                    &family,
                    ArtifactConfidentiality::Public,
                );
            }
        }
        previous_checkpoints = current_checkpoints;
    }
    builder.value_output_wire(FINAL_CHECKPOINT_FAMILY, previous_checkpoints.wire);
    Ok(builder.finish())
}

fn sample_checkpoint_family(
    builder: &mut GraphBuilder,
    config: &DiamondInjectorGraphConfig,
    level: usize,
) -> Result<TrapdoorFamilyWire, DiamondInjectorGraphError> {
    let count = config.state_count(level)?;
    let mut body = GraphBuilder::new(
        format!("diamond-checkpoint-level-{level}"),
        vec![CompileParameter { name: "state".to_owned(), kind: CompileParameterKind::Integer }],
    );
    let checkpoint = body.trapdoor_sample(
        config.public_type(),
        config.trapdoor_sigma.clone(),
        config.gadget_base.clone(),
        config.gadget_digit_count.clone(),
    );
    body.value_output_wire("checkpoint", checkpoint.wire);
    Ok(builder.parallel_trapdoor_loop(
        body.finish(),
        IntExpr::constant(count),
        "state",
        Vec::new(),
        Vec::new(),
        Vec::new(),
        config.public_type(),
        config.trapdoor_sigma.clone(),
        config.gadget_base.clone(),
        config.gadget_digit_count.clone(),
    )?)
}

fn sample_single_preimage_family(
    builder: &mut GraphBuilder,
    graph_name: String,
    trapdoor: &mxx_ir_core::TrapdoorWire,
    target: &MatrixWire,
    preimage_type: MatrixType,
) -> Result<MatrixFamilyWire, DiamondInjectorGraphError> {
    let mut body = GraphBuilder::new(
        graph_name,
        vec![CompileParameter {
            name: "chunk_instance".to_owned(),
            kind: CompileParameterKind::Integer,
        }],
    );
    let trapdoor_input = body.trapdoor_input(
        "trapdoor",
        trapdoor.public.matrix_type.clone(),
        trapdoor.sigma.clone(),
        trapdoor.gadget_base.clone(),
        trapdoor.digit_count.clone(),
    );
    let target_input = body.input("target", target.matrix_type.clone());
    let preimage = body.preimage_sample(&trapdoor_input, &target_input, preimage_type.clone());
    body.value_output_wire("preimage", preimage.wire);
    Ok(builder
        .parallel_loop(
            body.finish(),
            IntExpr::constant(1),
            "chunk_instance",
            Vec::new(),
            vec![trapdoor.wire, target.wire],
            vec![LoopInputMode::Broadcast, LoopInputMode::Broadcast],
            std::slice::from_ref(&preimage_type),
        )?
        .into_iter()
        .next()
        .expect("one declared single-preimage loop output"))
}

/// Builds Diamond WE encryption/key-generation, including input-injector
/// preprocessing, BGG+ public-key circuit evaluation, and all final projection
/// preimages. The graph takes `k`, `hash_key`, and one bool input per public
/// instance bit.
pub fn build_keygen_graph<P: Poly>(
    config: &DiamondWEGraphConfig,
    names: &DiamondWEArtifactNames,
    circuit: &PolyCircuit<P>,
) -> Result<Graph, DiamondInjectorGraphError> {
    config.injector.validate()?;
    if config.witness_size != config.injector.input_count.saturating_mul(config.injector.batch_bits)
    {
        return Err(DiamondInjectorGraphError::WitnessSize);
    }
    if circuit.num_input() != config.witness_size.saturating_add(config.instance_size) {
        return Err(DiamondInjectorGraphError::CircuitArity);
    }
    if circuit.num_output() != 1 {
        return Err(DiamondInjectorGraphError::CircuitOutput);
    }
    if config.concrete_bgg_columns == 0 {
        return Err(DiamondInjectorGraphError::NonPositiveParameter);
    }
    let modulus = config.injector.modulus.evaluate(&ParamEnv::default())?;
    if modulus <= BigInt::from(0) {
        return Err(DiamondInjectorGraphError::NonPositiveParameter);
    }

    let graph = build_preprocessing_graph(&config.injector, &names.injector)?;
    let final_checkpoint_output = graph
        .outputs
        .get(FINAL_CHECKPOINT_FAMILY)
        .copied()
        .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
    let final_checkpoints = TrapdoorFamilyWire {
        wire: final_checkpoint_output,
        matrix_type: config.injector.public_type(),
        count: IntExpr::constant(config.injector.state_count(config.injector.input_count)?),
        sigma: config.injector.trapdoor_sigma.clone(),
        gadget_base: config.injector.gadget_base.clone(),
        digit_count: config.injector.gadget_digit_count.clone(),
    };
    let mut builder = GraphBuilder::from_graph(graph);
    builder.remove_output(FINAL_CHECKPOINT_FAMILY);
    let scalar_type = config.injector.scalar_type();
    let public_key_type = MatrixType { columns: config.bgg_columns.clone(), ..scalar_type.clone() };
    let full_public_key_type = MatrixType {
        columns: IntExpr::constant(
            config
                .concrete_bgg_columns
                .checked_mul(config.witness_size.saturating_add(1))
                .ok_or(DiamondInjectorGraphError::StateCountOverflow)?,
        ),
        ..scalar_type.clone()
    };
    let hash_key = builder.bytes_input("hash_key", 32);
    let mut witness_tag = config.bgg_tag.clone();
    witness_tag.extend_from_slice(b":witness_public_keys");
    let full_keys = builder.hash_sample(
        hash_key,
        full_public_key_type,
        HashVariant::Plain,
        witness_tag,
        Vec::new(),
        None,
        None,
    );
    let mut public_keys = Vec::with_capacity(config.witness_size + 1);
    for index in 0..=config.witness_size {
        let start = index
            .checked_mul(config.concrete_bgg_columns)
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
        public_keys.push(
            builder.slice(
                &full_keys,
                None,
                Some(IndexRange {
                    start,
                    end: start
                        .checked_add(config.concrete_bgg_columns)
                        .ok_or(DiamondInjectorGraphError::StateCountOverflow)?,
                }),
                public_key_type.clone(),
            ),
        );
    }
    let public_key_family = builder.family_pack(&public_keys)?;
    builder.output_family_wire(
        names.public_keys.clone(),
        &public_key_family,
        ArtifactConfidentiality::Public,
    );
    let one_public_key =
        BggPublicKeyWire { matrix: public_keys[0].clone(), reveal_plaintext: true };
    let zero_public_key = BggPublicKeyWire {
        matrix: builder.matrix_scale(&public_keys[0], IntExpr::constant(0)),
        reveal_plaintext: true,
    };
    let mut circuit_inputs = public_keys[1..]
        .iter()
        .cloned()
        .map(|matrix| BggPublicKeyWire { matrix, reveal_plaintext: true })
        .collect::<Vec<_>>();
    for index in 0..config.instance_size {
        let bit = builder.boolean_input(format!("instance_{index}"));
        let index_wire = builder.bool_to_int(bit);
        let selected = builder
            .select(index_wire, &[zero_public_key.matrix.clone(), one_public_key.matrix.clone()]);
        circuit_inputs.push(BggPublicKeyWire { matrix: selected, reveal_plaintext: true });
    }
    let compiler = PolyCircuitCompiler {
        public_key: BggPublicKeyCompiler {
            base: config.gadget_base.clone(),
            decomposed_type: MatrixType {
                rows: config.bgg_columns.clone(),
                columns: config.bgg_columns.clone(),
                ..scalar_type.clone()
            },
        },
    };
    let out_public_key = compiler
        .compile_public_keys(&mut builder, circuit, one_public_key.clone(), circuit_inputs)?
        .into_iter()
        .next()
        .ok_or(DiamondInjectorGraphError::CircuitOutput)?;

    let gadget = builder.constant_matrix(
        public_key_type.clone(),
        ConstantMatrix::Gadget { base: config.gadget_base.clone(), small: false },
    );
    let zero_row = builder.constant_matrix(public_key_type.clone(), ConstantMatrix::Zero);
    let one_top = builder.matrix_binary(
        MatrixBinaryOp::Subtract,
        &one_public_key.matrix,
        &gadget,
        public_key_type.clone(),
    );
    let zero_checkpoint =
        builder.trapdoor_family_get_static(&final_checkpoints, IntExpr::constant(0));
    let negative_gadget = builder.matrix_negate(&gadget);
    for chunk in 0..config.projection_chunk_count() {
        let bounds = config.projection_chunk_bounds(chunk)?;
        let chunk_public_type = MatrixType {
            columns: IntExpr::constant(bounds.end - bounds.start),
            ..public_key_type.clone()
        };
        let chunk_target_type = MatrixType {
            rows: IntExpr::constant(PREFIX_SIZE),
            columns: IntExpr::constant(bounds.end - bounds.start),
            ..scalar_type.clone()
        };
        let chunk_preimage_type = config.projection_chunk_type(bounds);
        let one_top_chunk = builder.slice(&one_top, None, Some(bounds), chunk_public_type.clone());
        let zero_chunk = builder.slice(&zero_row, None, Some(bounds), chunk_public_type.clone());
        let one_target =
            stack_two_rows(&mut builder, &one_top_chunk, &zero_chunk, chunk_target_type.clone());
        let one_preimage = sample_single_preimage_family(
            &mut builder,
            format!("diamond-we-one-preimage-chunk-{chunk}"),
            &zero_checkpoint,
            &one_target,
            chunk_preimage_type.clone(),
        )?;
        builder.output_family_wire(
            names.one_preimage_chunk(chunk),
            &one_preimage,
            ArtifactConfidentiality::Public,
        );

        let mut body = GraphBuilder::new(
            format!("diamond-we-witness-preimage-chunk-{chunk}"),
            vec![CompileParameter { name: "bit".to_owned(), kind: CompileParameterKind::Integer }],
        );
        let public_key = body.input("public_key", public_key_type.clone());
        let trapdoor = body.trapdoor_input(
            "trapdoor",
            config.injector.public_type(),
            config.injector.trapdoor_sigma.clone(),
            config.injector.gadget_base.clone(),
            config.injector.gadget_digit_count.clone(),
        );
        let negative_gadget_input = body.input("negative_gadget", public_key_type.clone());
        let public_key = body.slice(&public_key, None, Some(bounds), chunk_public_type.clone());
        let negative_gadget_input =
            body.slice(&negative_gadget_input, None, Some(bounds), chunk_public_type.clone());
        let target =
            stack_two_rows(&mut body, &public_key, &negative_gadget_input, chunk_target_type);
        let preimage = body.preimage_sample(&trapdoor, &target, chunk_preimage_type.clone());
        body.value_output_wire("preimage", preimage.wire);
        let witness_preimages = builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(config.witness_size),
                "bit",
                Vec::new(),
                vec![public_key_family.wire, final_checkpoints.wire, negative_gadget.wire],
                vec![
                    LoopInputMode::ZipOffset { offset: 1 },
                    LoopInputMode::ZipOffset { offset: 1 },
                    LoopInputMode::Broadcast,
                ],
                std::slice::from_ref(&chunk_preimage_type),
            )?
            .into_iter()
            .next()
            .expect("one declared witness-preimage loop output");
        builder.output_family_wire(
            names.witness_preimages_chunk(chunk),
            &witness_preimages,
            ArtifactConfidentiality::Public,
        );
    }

    let mut k_tag = config.bgg_tag.clone();
    k_tag.extend_from_slice(b":k_public_key");
    let k_public_key = builder.hash_sample(
        hash_key,
        public_key_type.clone(),
        HashVariant::Plain,
        k_tag,
        Vec::new(),
        None,
        None,
    );
    let first_column = IndexRange { start: 0, end: 1 };
    let k_public_key_first =
        builder.slice(&k_public_key, None, Some(first_column), scalar_type.clone());
    let selector_unit = builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Identity);
    let selector = builder.matrix_scale(&selector_unit, IntExpr::constant(&modulus / 2));
    let scalar_target_type =
        MatrixType { rows: IntExpr::constant(PREFIX_SIZE), ..scalar_type.clone() };
    let scalar_preimage_type =
        MatrixType { rows: config.injector.state_columns.clone(), ..scalar_type.clone() };
    let k_target =
        stack_two_rows(&mut builder, &k_public_key_first, &selector, scalar_target_type.clone());
    let k_preimage =
        builder.preimage_sample(&zero_checkpoint, &k_target, scalar_preimage_type.clone());
    builder.output(names.k_preimage.clone(), &k_preimage, ArtifactConfidentiality::Public);

    let mut r_tag = config.bgg_tag.clone();
    r_tag.extend_from_slice(b":r");
    let r = builder.hash_sample(
        hash_key,
        public_key_type.clone(),
        HashVariant::Plain,
        r_tag,
        Vec::new(),
        None,
        None,
    );
    builder.output(names.r.clone(), &r, ArtifactConfidentiality::Public);
    let r_first = builder.slice(&r, None, Some(first_column), scalar_type.clone());
    let r_decomposed = builder.gadget_decompose(
        &r_first,
        config.gadget_base.clone(),
        MatrixType {
            rows: config.bgg_columns.clone(),
            columns: IntExpr::constant(1),
            ..scalar_type.clone()
        },
    );
    builder.output(names.r_decomposed.clone(), &r_decomposed, ArtifactConfidentiality::Public);
    let one_minus_out = builder.matrix_binary(
        MatrixBinaryOp::Subtract,
        &one_public_key.matrix,
        &out_public_key.matrix,
        public_key_type.clone(),
    );
    let dec_projection = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &one_minus_out,
        &r_decomposed,
        scalar_type.clone(),
    );
    let dec_public_key = builder.matrix_binary(
        MatrixBinaryOp::Add,
        &k_public_key_first,
        &dec_projection,
        scalar_type.clone(),
    );
    let zero_scalar = builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Zero);
    let decoder_target =
        stack_two_rows(&mut builder, &dec_public_key, &zero_scalar, scalar_target_type);
    let decoder_preimage =
        builder.preimage_sample(&zero_checkpoint, &decoder_target, scalar_preimage_type);
    builder.output(
        names.decoder_preimage.clone(),
        &decoder_preimage,
        ArtifactConfidentiality::Public,
    );
    Ok(builder.finish())
}

/// Builds the online injector graph. Each transition family is loaded lazily
/// and exactly one digit branch per state/level is selected.
pub fn build_evaluation_graph(
    config: &DiamondInjectorGraphConfig,
    names: &DiamondInjectorArtifactNames,
    production_id: ProductionId,
) -> Result<Graph, DiamondInjectorGraphError> {
    config.validate()?;
    let mut builder = GraphBuilder::new("diamond-we-injector-evaluate", Vec::new());
    let state_type = config.state_type();
    let initial_state = builder.artifact_input(
        "initial_state_artifact",
        state_type.clone(),
        production_id.clone(),
        names.initial_state.clone(),
        ArtifactConfidentiality::Public,
    );
    let mut states = vec![builder.family_pack(std::slice::from_ref(&initial_state))?];
    for level in 1..=config.input_count {
        let digit = builder.integer_input(format!("digit_{level}"));
        let state_count = config.state_count(level)?;
        let first_new = config.first_new_state(level)?;
        let mut next = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let source = if state >= first_new { &states[0] } else { &states[state] };
            let mut selected_transitions = Vec::with_capacity(config.transition_chunk_count());
            let mut chunk_types = Vec::with_capacity(config.transition_chunk_count());
            for chunk in 0..config.transition_chunk_count() {
                let bounds = config.transition_chunk_bounds(chunk)?;
                let transition_type = config.transition_chunk_type(bounds);
                let family = builder.artifact_family_input(
                    format!("transition_{level}_{state}_chunk_{chunk}"),
                    transition_type,
                    production_id.clone(),
                    names.transition_chunk(level, state, chunk),
                    IntExpr::constant(config.base),
                    ArtifactConfidentiality::Public,
                );
                selected_transitions.push(builder.family_get_dynamic(&family, digit));
                chunk_types.push(MatrixType {
                    columns: IntExpr::constant(bounds.end - bounds.start),
                    ..config.scalar_type()
                });
            }
            let mut body = GraphBuilder::new(
                format!("diamond-evaluate-level-{level}-state-{state}"),
                vec![CompileParameter {
                    name: "state_instance".to_owned(),
                    kind: CompileParameterKind::Integer,
                }],
            );
            let source_input = body.input("source", state_type.clone());
            let mut chunks = Vec::with_capacity(selected_transitions.len());
            for (chunk, (transition, state_chunk_type)) in
                selected_transitions.iter().zip(&chunk_types).enumerate()
            {
                let transition_input =
                    body.input(format!("transition_chunk_{chunk}"), transition.matrix_type.clone());
                chunks.push(body.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    &source_input,
                    &transition_input,
                    state_chunk_type.clone(),
                ));
            }
            let state_output = body.concat(ConcatAxis::Columns, &chunks, state_type.clone());
            body.value_output_wire("state", state_output.wire);
            let mut args = Vec::with_capacity(selected_transitions.len() + 1);
            args.push(source.wire);
            args.extend(selected_transitions.iter().map(|transition| transition.wire));
            let mut input_modes = Vec::with_capacity(args.len());
            input_modes.push(LoopInputMode::Zip);
            input_modes.extend((0..selected_transitions.len()).map(|_| LoopInputMode::Broadcast));
            let state_family = builder
                .parallel_loop(
                    body.finish(),
                    IntExpr::constant(1),
                    "state_instance",
                    Vec::new(),
                    args,
                    input_modes,
                    std::slice::from_ref(&state_type),
                )?
                .into_iter()
                .next()
                .expect("one declared state loop output");
            next.push(state_family);
        }
        states = next;
    }
    for (state, family) in states.iter().enumerate() {
        builder.output_family_wire(
            names.final_state(state),
            family,
            ArtifactConfidentiality::Public,
        );
    }
    Ok(builder.finish())
}

/// Builds the complete Diamond WE decryption/evaluation graph over artifacts
/// produced by [`build_keygen_graph`].
pub fn build_we_evaluation_graph<P: Poly>(
    config: &DiamondWEGraphConfig,
    names: &DiamondWEArtifactNames,
    production_id: ProductionId,
    circuit: &PolyCircuit<P>,
) -> Result<Graph, DiamondInjectorGraphError> {
    if circuit.num_input() != config.witness_size.saturating_add(config.instance_size) {
        return Err(DiamondInjectorGraphError::CircuitArity);
    }
    if circuit.num_output() != 1 {
        return Err(DiamondInjectorGraphError::CircuitOutput);
    }
    let graph = build_evaluation_graph(&config.injector, &names.injector, production_id.clone())?;
    let final_count = config.injector.state_count(config.injector.input_count)?;
    let final_state_outputs = (0..final_count)
        .map(|state| {
            graph
                .outputs
                .get(&names.injector.final_state(state))
                .copied()
                .ok_or(DiamondInjectorGraphError::StateCountOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let digit_wires = (1..=config.injector.input_count)
        .map(|level| {
            let name = format!("digit_{level}");
            graph
                .nodes
                .iter()
                .find_map(|node| match &node.kind {
                    NodeKind::Input { name: node_name, .. } if node_name == &name => {
                        Some(WireRef { node: node.id, port: Port(0) })
                    }
                    _ => None,
                })
                .ok_or(DiamondInjectorGraphError::StateCountOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut builder = GraphBuilder::from_graph(graph);
    let final_state_families = final_state_outputs
        .into_iter()
        .enumerate()
        .map(|(state, wire)| {
            builder.remove_output(&names.injector.final_state(state));
            MatrixFamilyWire {
                wire,
                matrix_type: config.injector.state_type(),
                count: IntExpr::constant(1),
            }
        })
        .collect::<Vec<_>>();
    let root_state = builder.family_get_static(&final_state_families[0], IntExpr::constant(0));

    let scalar_type = config.injector.scalar_type();
    let public_key_type = MatrixType { columns: config.bgg_columns.clone(), ..scalar_type.clone() };
    let scalar_preimage_type =
        MatrixType { rows: config.injector.state_columns.clone(), ..scalar_type.clone() };
    let mut one_preimages = Vec::with_capacity(config.projection_chunk_count());
    let mut witness_preimages = Vec::with_capacity(config.projection_chunk_count());
    for chunk in 0..config.projection_chunk_count() {
        let bounds = config.projection_chunk_bounds(chunk)?;
        let preimage_type = config.projection_chunk_type(bounds);
        let one_family = builder.artifact_family_input(
            format!("one_preimage_chunk_{chunk}_artifact"),
            preimage_type.clone(),
            production_id.clone(),
            names.one_preimage_chunk(chunk),
            IntExpr::constant(1),
            ArtifactConfidentiality::Public,
        );
        one_preimages.push(builder.family_get_static(&one_family, IntExpr::constant(0)));
        witness_preimages.push(builder.artifact_family_input(
            format!("witness_preimage_chunk_{chunk}_artifacts"),
            preimage_type,
            production_id.clone(),
            names.witness_preimages_chunk(chunk),
            IntExpr::constant(config.witness_size),
            ArtifactConfidentiality::Public,
        ));
    }
    let k_preimage = builder.artifact_input(
        "k_preimage_artifact",
        scalar_preimage_type.clone(),
        production_id.clone(),
        names.k_preimage.clone(),
        ArtifactConfidentiality::Public,
    );
    let decoder_preimage = builder.artifact_input(
        "decoder_preimage_artifact",
        scalar_preimage_type,
        production_id.clone(),
        names.decoder_preimage.clone(),
        ArtifactConfidentiality::Public,
    );

    let public_keys = builder.artifact_family_input(
        "public_key_artifacts",
        public_key_type.clone(),
        production_id.clone(),
        names.public_keys.clone(),
        IntExpr::constant(config.witness_size.saturating_add(1)),
        ArtifactConfidentiality::Public,
    );
    let one_public_key = builder.family_get_static(&public_keys, IntExpr::constant(0));
    let one_vector_chunks = one_preimages
        .iter()
        .enumerate()
        .map(|(chunk, preimage)| {
            let bounds = config.projection_chunk_bounds(chunk)?;
            Ok(builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &root_state,
                preimage,
                MatrixType {
                    columns: IntExpr::constant(bounds.end - bounds.start),
                    ..scalar_type.clone()
                },
            ))
        })
        .collect::<Result<Vec<_>, DiamondInjectorGraphError>>()?;
    let one_vector =
        builder.concat(ConcatAxis::Columns, &one_vector_chunks, public_key_type.clone());
    let one_plaintext = builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Identity);
    let one_encoding = BggEncodingWire {
        vector: one_vector,
        pubkey: BggPublicKeyWire { matrix: one_public_key, reveal_plaintext: true },
        plaintext: Some(one_plaintext.clone()),
    };
    let zero_encoding = zero_encoding(&mut builder, &one_encoding);

    let mut circuit_inputs = Vec::with_capacity(config.witness_size + config.instance_size);
    for bit in 0..config.witness_size {
        let digit = bit / config.injector.batch_bits;
        let bit_in_digit = bit % config.injector.batch_bits;
        let state = 1 + bit;
        let state = builder.family_get_static(&final_state_families[state], IntExpr::constant(0));
        let vector_chunks = witness_preimages
            .iter()
            .enumerate()
            .map(|(chunk, family)| {
                let bounds = config.projection_chunk_bounds(chunk)?;
                let witness_preimage = builder.family_get_static(family, IntExpr::constant(bit));
                Ok(builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    &state,
                    &witness_preimage,
                    MatrixType {
                        columns: IntExpr::constant(bounds.end - bounds.start),
                        ..scalar_type.clone()
                    },
                ))
            })
            .collect::<Result<Vec<_>, DiamondInjectorGraphError>>()?;
        let vector = builder.concat(ConcatAxis::Columns, &vector_chunks, public_key_type.clone());
        let bit_wire = builder.bit_extract(digit_wires[digit], IntExpr::constant(bit_in_digit));
        let bit_index = builder.bool_to_int(bit_wire);
        let zero_plaintext = builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Zero);
        let bit_plaintext = builder.select(bit_index, &[zero_plaintext, one_plaintext.clone()]);
        let public_key = builder.family_get_static(&public_keys, IntExpr::constant(bit + 1));
        circuit_inputs.push(BggEncodingWire {
            vector,
            pubkey: BggPublicKeyWire { matrix: public_key, reveal_plaintext: true },
            plaintext: Some(bit_plaintext),
        });
    }
    for index in 0..config.instance_size {
        let bit = builder.boolean_input(format!("instance_{index}"));
        let bit_index = builder.bool_to_int(bit);
        circuit_inputs.push(select_encoding(
            &mut builder,
            bit_index,
            &zero_encoding,
            &one_encoding,
        ));
    }
    let compiler = PolyCircuitCompiler {
        public_key: BggPublicKeyCompiler {
            base: config.gadget_base.clone(),
            decomposed_type: MatrixType {
                rows: config.bgg_columns.clone(),
                columns: config.bgg_columns.clone(),
                ..scalar_type.clone()
            },
        },
    };
    let out_encoding = compiler
        .compile_encodings(&mut builder, circuit, one_encoding.clone(), circuit_inputs)?
        .into_iter()
        .next()
        .ok_or(DiamondInjectorGraphError::CircuitOutput)?;

    let k_vector = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &root_state,
        &k_preimage,
        scalar_type.clone(),
    );
    let decoder = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &root_state,
        &decoder_preimage,
        scalar_type.clone(),
    );
    let r_decomposed = builder.artifact_input(
        "r_decomposed_artifact",
        MatrixType {
            rows: config.bgg_columns.clone(),
            columns: IntExpr::constant(1),
            ..scalar_type.clone()
        },
        production_id,
        names.r_decomposed.clone(),
        ArtifactConfidentiality::Public,
    );
    let difference = builder.matrix_binary(
        MatrixBinaryOp::Subtract,
        &one_encoding.vector,
        &out_encoding.vector,
        public_key_type.clone(),
    );
    let full_projection = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &difference,
        &r_decomposed,
        scalar_type.clone(),
    );
    let encoded = builder.matrix_binary(
        MatrixBinaryOp::Add,
        &k_vector,
        &full_projection,
        scalar_type.clone(),
    );
    let noisy_plaintext =
        builder.matrix_binary(MatrixBinaryOp::Subtract, &decoder, &encoded, scalar_type);
    let modulus = config.injector.modulus.evaluate(&ParamEnv::default())?;
    let decoded = legacy_boolean_decode(&mut builder, &noisy_plaintext, &modulus);
    builder.value_output_wire("message", decoded);
    Ok(builder.finish())
}

/// Reproduces the legacy Diamond WE boolean decoder exactly.
///
/// The concrete implementation classifies the raw constant coefficient as
/// true precisely on the closed interval
/// `[floor(q / 4), 3 * floor(q / 4)]`. `ExtractCoefficient` exposes a centered
/// representative, so the same interval is the union of the positive branch
/// `coefficient >= floor(q / 4)` and the negative branch
/// `coefficient <= 3 * floor(q / 4) - q`.
pub(super) fn legacy_boolean_decode(
    builder: &mut GraphBuilder,
    noisy_plaintext: &MatrixWire,
    modulus: &BigInt,
) -> WireRef {
    debug_assert!(modulus > &BigInt::from(0));
    let coefficient = builder.extract_coefficient(noisy_plaintext, IntExpr::constant(0));
    let quarter = modulus / 4;
    let upper_centered = &quarter * 3 - modulus;
    let quarter = builder.constant_int(quarter);
    let upper_centered = builder.constant_int(upper_centered);
    let positive = builder.int_compare(IntCompareOp::LessEqual, quarter, coefficient);
    let negative = builder.int_compare(IntCompareOp::LessEqual, coefficient, upper_centered);
    let positive = builder.bool_to_int(positive);
    let negative = builder.bool_to_int(negative);
    let branch_count = builder.int_binary(IntBinaryOp::Add, positive, negative);
    let zero = builder.constant_int(0);
    builder.int_compare(IntCompareOp::Less, zero, branch_count)
}

fn zero_encoding(builder: &mut GraphBuilder, one: &BggEncodingWire) -> BggEncodingWire {
    BggEncodingWire {
        vector: builder.matrix_scale(&one.vector, IntExpr::constant(0)),
        pubkey: BggPublicKeyWire {
            matrix: builder.matrix_scale(&one.pubkey.matrix, IntExpr::constant(0)),
            reveal_plaintext: one.pubkey.reveal_plaintext,
        },
        plaintext: one
            .plaintext
            .as_ref()
            .map(|plaintext| builder.matrix_scale(plaintext, IntExpr::constant(0))),
    }
}

fn select_encoding(
    builder: &mut GraphBuilder,
    index: WireRef,
    zero: &BggEncodingWire,
    one: &BggEncodingWire,
) -> BggEncodingWire {
    BggEncodingWire {
        vector: builder.select(index, &[zero.vector.clone(), one.vector.clone()]),
        pubkey: BggPublicKeyWire {
            matrix: builder.select(index, &[zero.pubkey.matrix.clone(), one.pubkey.matrix.clone()]),
            reveal_plaintext: zero.pubkey.reveal_plaintext && one.pubkey.reveal_plaintext,
        },
        plaintext: match (&zero.plaintext, &one.plaintext) {
            (Some(zero), Some(one)) => Some(builder.select(index, &[zero.clone(), one.clone()])),
            _ => None,
        },
    }
}

/// Represents a two-row block matrix as `e_0 * top + e_1 * bottom`.
/// This is value-identical to row concatenation, while keeping the two blocks
/// visible to symbolic distribution after a preimage rewrite.
fn stack_two_rows(
    builder: &mut GraphBuilder,
    top: &MatrixWire,
    bottom: &MatrixWire,
    output_type: MatrixType,
) -> MatrixWire {
    let embedding_type = MatrixType {
        rows: output_type.rows.clone(),
        columns: IntExpr::constant(1),
        ..output_type.clone()
    };
    let top_embedding = builder.constant_matrix(
        embedding_type.clone(),
        ConstantMatrix::UnitColumn { index: IntExpr::constant(0) },
    );
    let bottom_embedding = builder.constant_matrix(
        embedding_type,
        ConstantMatrix::UnitColumn { index: IntExpr::constant(1) },
    );
    let top =
        builder.matrix_binary(MatrixBinaryOp::Multiply, &top_embedding, top, output_type.clone());
    let bottom = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &bottom_embedding,
        bottom,
        output_type.clone(),
    );
    builder.matrix_binary(MatrixBinaryOp::Add, &top, &bottom, output_type)
}

fn ternary(builder: &mut GraphBuilder, matrix_type: MatrixType) -> MatrixWire {
    builder.uniform_sample(
        matrix_type,
        SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
    )
}

fn diagonal_selector(
    builder: &mut GraphBuilder,
    mask: &MatrixWire,
    keep_k: bool,
    scalar_type: &MatrixType,
    selector_type: &MatrixType,
) -> MatrixWire {
    let bottom = if keep_k {
        builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Identity)
    } else {
        mask.clone()
    };
    builder.concat(ConcatAxis::Diagonal, &[mask.clone(), bottom], selector_type.clone())
}

fn special_selector(
    builder: &mut GraphBuilder,
    mask: &MatrixWire,
    bit: usize,
    scalar_type: &MatrixType,
    selector_type: &MatrixType,
) -> MatrixWire {
    let bit_mask = builder.matrix_scale(mask, IntExpr::constant(bit));
    let top = builder.concat(
        ConcatAxis::Columns,
        &[mask.clone(), bit_mask],
        MatrixType { columns: IntExpr::constant(PREFIX_SIZE), ..scalar_type.clone() },
    );
    let zero = builder.constant_matrix(
        MatrixType { columns: IntExpr::constant(PREFIX_SIZE), ..scalar_type.clone() },
        ConstantMatrix::Zero,
    );
    builder.concat(ConcatAxis::Rows, &[top, zero], selector_type.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        ParamEnv,
        artifact::{SpecHash, export_validated_manifest},
        node::NodeKind,
        types::WireId,
        validate,
    };
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    fn config() -> DiamondInjectorGraphConfig {
        DiamondInjectorGraphConfig {
            modulus: IntExpr::constant(257),
            ring_dimension: IntExpr::constant(8),
            state_columns: IntExpr::constant(18),
            concrete_state_columns: 18,
            preimage_chunk_columns: 5,
            input_count: 2,
            base: 2,
            batch_bits: 1,
            trapdoor_sigma: RealExpr::FromInt(IntExpr::constant(4)),
            gadget_base: IntExpr::constant(2),
            gadget_digit_count: IntExpr::constant(7),
            error_sigma: RealExpr::FromInt(IntExpr::constant(1)),
        }
    }

    #[test]
    fn row_stack_has_the_declared_core_type() {
        let scalar = config().scalar_type();
        let row = MatrixType { columns: IntExpr::constant(3), ..scalar };
        let stacked_type = MatrixType { rows: IntExpr::constant(2), ..row.clone() };
        let mut builder = GraphBuilder::new("row-stack", Vec::new());
        let top = builder.input("top", row.clone());
        let bottom = builder.input("bottom", row);
        let stacked = stack_two_rows(&mut builder, &top, &bottom, stacked_type);
        builder.output("stacked", &stacked, ArtifactConfidentiality::Public);
        let graph = builder.finish();
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        let wire = WireId { instantiation_path: Vec::new(), wire: stacked.wire };
        assert_eq!(validated.wires[&wire].matrix_type().expect("matrix").rows, 2);
    }

    #[test]
    fn preprocessing_graph_elaborates_all_level_state_digit_transitions() {
        let graph = build_preprocessing_graph(&config(), &DiamondInjectorArtifactNames::default())
            .expect("graph");
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        assert!(validated.outputs.contains_key("diamond_initial_state"));
        assert!(validated.outputs.contains_key("diamond_transition_level_2_state_2_chunk_3"));
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::ParallelLoop(_)))
                .count(),
            23,
            "three checkpoint-family loops plus twenty transition-chunk loops"
        );
        assert!(
            graph.nodes.iter().all(|node| !matches!(node.kind, NodeKind::PreimageSample { .. })),
            "transition preimages must be sampled inside bounded loop bodies"
        );
        assert!(
            graph.nodes.iter().all(|node| !matches!(node.kind, NodeKind::TrapdoorSample { .. })),
            "checkpoint trapdoors must be sampled inside bounded loop bodies"
        );
    }

    #[test]
    fn evaluation_graph_uses_one_lazy_family_get_per_level_state_chunk() {
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let graph = build_evaluation_graph(
            &config(),
            &DiamondInjectorArtifactNames::default(),
            production_id,
        )
        .expect("graph");
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::FamilyGetDynamic))
                .count(),
            20
        );
    }

    #[test]
    fn keygen_graph_elaborates_injector_circuit_and_final_projections() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(2).to_vec()[0];
        circuit.output([input]);
        let graph = build_keygen_graph(
            &DiamondWEGraphConfig {
                injector: config(),
                witness_size: 2,
                instance_size: 0,
                bgg_columns: IntExpr::constant(9),
                concrete_bgg_columns: 9,
                gadget_base: IntExpr::constant(2),
                bgg_tag: b"test".to_vec(),
            },
            &DiamondWEArtifactNames::default(),
            &circuit,
        )
        .expect("keygen graph");
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        assert!(validated.outputs.contains_key("we_one_preimage_chunk_0"));
        assert!(validated.outputs.contains_key("we_one_preimage_chunk_1"));
        assert!(validated.outputs.contains_key("we_witness_preimages_chunk_0"));
        assert!(validated.outputs.contains_key("we_witness_preimages_chunk_1"));
        assert!(validated.outputs.contains_key("we_k_preimage"));
        assert!(validated.outputs.contains_key("we_decoder_preimage"));
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::PreimageSample { .. }))
                .count(),
            2,
            "only the one-column k and decoder preimages remain root nodes"
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::ParallelLoop(_)))
                .count(),
            27,
            "three checkpoint loops, twenty transition chunk loops, and two one/witness chunk-loop pairs"
        );
        let r_decomposed = validated.outputs["we_r_decomposed"];
        let r_decomposed = WireId { instantiation_path: Vec::new(), wire: r_decomposed };
        assert_eq!(
            validated.wires[&r_decomposed].matrix_type().expect("r decomposition matrix").columns,
            1,
            "decryption stores only the one decomposition column it consumes"
        );
        assert!(
            graph.nodes.iter().any(|node| {
                matches!(
                    &node.kind,
                    NodeKind::MatrixScale { scalar }
                        if scalar == &IntExpr::constant(128)
                )
            }),
            "the q/2 selector must use floor(257 / 2), matching the concrete implementation"
        );
    }

    #[test]
    fn we_evaluation_graph_validates_against_keygen_artifact_wiring() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(2).to_vec()[0];
        circuit.output([input]);
        let config = DiamondWEGraphConfig {
            injector: config(),
            witness_size: 2,
            instance_size: 0,
            bgg_columns: IntExpr::constant(9),
            concrete_bgg_columns: 9,
            gadget_base: IntExpr::constant(2),
            bgg_tag: b"test".to_vec(),
        };
        let names = DiamondWEArtifactNames::default();
        let production_id = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] };
        let keygen = build_keygen_graph(&config, &names, &circuit).expect("keygen graph");
        let keygen = validate(&keygen, &ParamEnv::default()).expect("keygen validation");
        let manifest = export_validated_manifest(production_id.clone(), &keygen)
            .expect("keygen artifact manifest");
        let evaluation =
            build_we_evaluation_graph(&config, &names, production_id.clone(), &circuit)
                .expect("evaluation graph");
        let validated = mxx_ir_core::validate_with_manifests(
            &evaluation,
            &ParamEnv::default(),
            &std::collections::BTreeMap::from([(production_id, manifest)]),
        )
        .expect("validation");
        assert!(validated.outputs.contains_key("message"));
        assert!(
            evaluation
                .nodes
                .iter()
                .any(|node| matches!(node.kind, NodeKind::Input { artifact: Some(_), .. }))
        );
    }
}
