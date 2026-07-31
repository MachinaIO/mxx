//! Graph-IR construction for the Diamond input-injection portion of
//! Diamond witness encryption.
//!
//! The builders in this module reproduce the level/state transition formulas
//! used by `DiamondInjector::preprocess` and `DiamondInjector::online_eval`.

use mxx_bgg::{
    BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, CircuitCompileError, GraphBuilder,
    MatrixWire, OutputFamilyError, PolyCircuitCompiler, TrapdoorWire,
};
use mxx_gadgets::{Poly, circuit::PolyCircuit};
use mxx_ir_core::{
    Graph, IntExpr, Port, WireRef,
    artifact::ProductionId,
    expr::RealExpr,
    node::{
        ConcatAxis, ConstantMatrix, HashVariant, IndexRange, MatrixBinaryOp, NodeKind, SampleRange,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use thiserror::Error;

const PREFIX_SIZE: usize = 2;
const SECRET_SIZE: usize = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInjectorGraphConfig {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub state_columns: IntExpr,
    pub concrete_state_columns: usize,
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
    pub fn transition(&self, level: usize, state: usize) -> String {
        format!("{}_level_{level}_state_{state}", self.transition_prefix)
    }
}

impl DiamondInjectorGraphConfig {
    pub fn validate(&self) -> Result<(), DiamondInjectorGraphError> {
        if self.input_count == 0 ||
            self.base == 0 ||
            self.batch_bits == 0 ||
            self.concrete_state_columns == 0
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

    fn transition_type(&self) -> MatrixType {
        MatrixType {
            rows: self.state_columns.clone(),
            columns: self.state_columns.clone(),
            ..self.scalar_type()
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
    let transition_type = config.transition_type();
    let k = builder.input("k", scalar_type.clone());

    let mut checkpoints = Vec::<Vec<TrapdoorWire>>::with_capacity(config.input_count + 1);
    for level in 0..=config.input_count {
        let count = config.state_count(level)?;
        checkpoints.push(
            (0..count)
                .map(|_| {
                    builder.trapdoor_sample(
                        public_type.clone(),
                        config.trapdoor_sigma.clone(),
                        config.gadget_base.clone(),
                        config.gadget_digit_count.clone(),
                    )
                })
                .collect(),
        );
    }

    let secret_epsilon = ternary(&mut builder, scalar_type.clone());
    let initial_selector = builder.concat(
        ConcatAxis::Columns,
        &[secret_epsilon, k],
        MatrixType { columns: IntExpr::constant(PREFIX_SIZE), ..scalar_type.clone() },
    );
    let initial_product = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &initial_selector,
        &checkpoints[0][0].public,
        state_type.clone(),
    );
    let initial_error = builder.gaussian_sample(state_type.clone(), config.error_sigma.clone());
    let initial_state = builder.matrix_binary(
        MatrixBinaryOp::Add,
        &initial_product,
        &initial_error,
        state_type.clone(),
    );
    builder.output(names.initial_state.clone(), &initial_state);

    for level in 1..=config.input_count {
        let state_count = config.state_count(level)?;
        let first_new = config.first_new_state(level)?;
        let masks = (0..config.base)
            .map(|_| ternary(&mut builder, scalar_type.clone()))
            .collect::<Vec<_>>();
        for state in 0..state_count {
            let mut family = Vec::with_capacity(config.base);
            for (digit, mask) in masks.iter().enumerate() {
                let selector = if let Some(bit) = config.new_bit(level, state)? {
                    special_selector(
                        &mut builder,
                        mask,
                        (digit >> bit) & 1,
                        &scalar_type,
                        &selector_type,
                    )
                } else if state == 0 {
                    diagonal_selector(&mut builder, mask, true, &scalar_type, &selector_type)
                } else {
                    diagonal_selector(&mut builder, mask, false, &scalar_type, &selector_type)
                };
                let target_product = builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    &selector,
                    &checkpoints[level][state].public,
                    public_type.clone(),
                );
                let error =
                    builder.gaussian_sample(public_type.clone(), config.error_sigma.clone());
                let target = builder.matrix_binary(
                    MatrixBinaryOp::Add,
                    &target_product,
                    &error,
                    public_type.clone(),
                );
                let source_state = if state >= first_new { 0 } else { state };
                family.push(builder.preimage_sample(
                    &checkpoints[level - 1][source_state],
                    &target,
                    transition_type.clone(),
                ));
            }
            builder.output_family(names.transition(level, state), &family)?;
        }
    }
    Ok(builder.finish())
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

    let graph = build_preprocessing_graph(&config.injector, &names.injector)?;
    let checkpoints = trapdoor_checkpoints(&graph, &config.injector)?;
    let mut builder = GraphBuilder::from_graph(graph);
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
    builder.output_family(names.public_keys.clone(), &public_keys)?;
    let one_public_key = BggPublicKeyWire { matrix: public_keys[0].clone() };
    let zero_public_key =
        BggPublicKeyWire { matrix: builder.matrix_scale(&public_keys[0], IntExpr::constant(0)) };
    let mut circuit_inputs = public_keys[1..]
        .iter()
        .cloned()
        .map(|matrix| BggPublicKeyWire { matrix })
        .collect::<Vec<_>>();
    for index in 0..config.instance_size {
        let bit = builder.boolean_input(format!("instance_{index}"));
        let index_wire = builder.bool_to_int(bit);
        let selected = builder
            .select(index_wire, &[zero_public_key.matrix.clone(), one_public_key.matrix.clone()]);
        circuit_inputs.push(BggPublicKeyWire { matrix: selected });
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
    let projection_target_type = MatrixType {
        rows: IntExpr::constant(PREFIX_SIZE),
        columns: config.bgg_columns.clone(),
        ..scalar_type.clone()
    };
    let projection_preimage_type = MatrixType {
        rows: config.injector.state_columns.clone(),
        columns: config.bgg_columns.clone(),
        ..scalar_type.clone()
    };

    let one_top = builder.matrix_binary(
        MatrixBinaryOp::Subtract,
        &one_public_key.matrix,
        &gadget,
        public_key_type.clone(),
    );
    let one_target =
        stack_two_rows(&mut builder, &one_top, &zero_row, projection_target_type.clone());
    let one_preimage = builder.preimage_sample(
        &checkpoints[config.injector.input_count][0],
        &one_target,
        projection_preimage_type.clone(),
    );
    builder.output(names.one_preimage.clone(), &one_preimage);

    let mut witness_preimages = Vec::with_capacity(config.witness_size);
    let negative_gadget = builder.matrix_negate(&gadget);
    for (bit, public_key) in public_keys[1..].iter().enumerate() {
        let target = stack_two_rows(
            &mut builder,
            public_key,
            &negative_gadget,
            projection_target_type.clone(),
        );
        let digit = bit / config.injector.batch_bits;
        let bit_in_digit = bit % config.injector.batch_bits;
        let state = 1usize
            .checked_add(
                digit
                    .checked_mul(config.injector.batch_bits)
                    .and_then(|value| value.checked_add(bit_in_digit))
                    .ok_or(DiamondInjectorGraphError::StateCountOverflow)?,
            )
            .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
        witness_preimages.push(builder.preimage_sample(
            &checkpoints[config.injector.input_count][state],
            &target,
            projection_preimage_type.clone(),
        ));
    }
    builder.output_family(names.witness_preimages.clone(), &witness_preimages)?;

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
    let selector_unit = builder.constant_matrix(
        public_key_type.clone(),
        ConstantMatrix::UnitRow { index: IntExpr::constant(0) },
    );
    let selector = builder.matrix_scale(
        &selector_unit,
        IntExpr::RoundDiv(
            Box::new(config.injector.modulus.clone()),
            Box::new(IntExpr::constant(2)),
        ),
    );
    let k_target =
        stack_two_rows(&mut builder, &k_public_key, &selector, projection_target_type.clone());
    let k_preimage = builder.preimage_sample(
        &checkpoints[config.injector.input_count][0],
        &k_target,
        projection_preimage_type.clone(),
    );
    builder.output(names.k_preimage.clone(), &k_preimage);

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
    builder.output(names.r.clone(), &r);
    let r_decomposed = builder.gadget_decompose(
        &r,
        config.gadget_base.clone(),
        MatrixType {
            rows: config.bgg_columns.clone(),
            columns: config.bgg_columns.clone(),
            ..scalar_type.clone()
        },
    );
    builder.output(names.r_decomposed.clone(), &r_decomposed);
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
        public_key_type.clone(),
    );
    let dec_public_key = builder.matrix_binary(
        MatrixBinaryOp::Add,
        &k_public_key,
        &dec_projection,
        public_key_type.clone(),
    );
    let decoder_target =
        stack_two_rows(&mut builder, &dec_public_key, &zero_row, projection_target_type);
    let decoder_preimage = builder.preimage_sample(
        &checkpoints[config.injector.input_count][0],
        &decoder_target,
        projection_preimage_type,
    );
    builder.output(names.decoder_preimage.clone(), &decoder_preimage);
    Ok(builder.finish())
}

fn trapdoor_checkpoints(
    graph: &Graph,
    config: &DiamondInjectorGraphConfig,
) -> Result<Vec<Vec<TrapdoorWire>>, DiamondInjectorGraphError> {
    let mut samples = graph.nodes.iter().filter_map(|node| match &node.kind {
        NodeKind::TrapdoorSample { matrix_type, sigma, gadget_base, digit_count } => {
            Some(TrapdoorWire {
                wire: WireRef { node: node.id, port: Port(1) },
                public: MatrixWire {
                    wire: WireRef { node: node.id, port: Port(0) },
                    matrix_type: matrix_type.clone(),
                },
                sigma: sigma.clone(),
                gadget_base: gadget_base.clone(),
                digit_count: digit_count.clone(),
            })
        }
        _ => None,
    });
    let mut checkpoints = Vec::with_capacity(config.input_count + 1);
    for level in 0..=config.input_count {
        checkpoints.push(
            (0..config.state_count(level)?)
                .map(|_| samples.next().ok_or(DiamondInjectorGraphError::StateCountOverflow))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if samples.next().is_some() {
        return Err(DiamondInjectorGraphError::StateCountOverflow);
    }
    Ok(checkpoints)
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
    let transition_type = config.transition_type();
    let mut states = vec![builder.artifact_input(
        "initial_state_artifact",
        state_type.clone(),
        production_id.clone(),
        names.initial_state.clone(),
    )];
    for level in 1..=config.input_count {
        let digit = builder.integer_input(format!("digit_{level}"));
        let state_count = config.state_count(level)?;
        let first_new = config.first_new_state(level)?;
        let mut next = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let family = builder.artifact_family_input(
                format!("transition_{level}_{state}"),
                transition_type.clone(),
                production_id.clone(),
                names.transition(level, state),
                IntExpr::constant(config.base),
                config.base,
            );
            let selected = builder.select(digit, &family);
            let source = if state >= first_new { &states[0] } else { &states[state] };
            next.push(builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                source,
                &selected,
                state_type.clone(),
            ));
        }
        states = next;
    }
    builder.output_family("final_states", &states)?;
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
    let output = graph
        .outputs
        .get("final_states")
        .copied()
        .ok_or(DiamondInjectorGraphError::StateCountOverflow)?;
    let final_states = (0..final_count)
        .map(|port| MatrixWire {
            wire: WireRef { node: output.node, port: Port(port as u32) },
            matrix_type: config.injector.state_type(),
        })
        .collect::<Vec<_>>();
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
    builder.remove_output("final_states");

    let scalar_type = config.injector.scalar_type();
    let public_key_type = MatrixType { columns: config.bgg_columns.clone(), ..scalar_type.clone() };
    let preimage_type = MatrixType {
        rows: config.injector.state_columns.clone(),
        columns: config.bgg_columns.clone(),
        ..scalar_type.clone()
    };
    let one_preimage = builder.artifact_input(
        "one_preimage_artifact",
        preimage_type.clone(),
        production_id.clone(),
        names.one_preimage.clone(),
    );
    let witness_preimages = builder.artifact_family_input(
        "witness_preimage_artifacts",
        preimage_type.clone(),
        production_id.clone(),
        names.witness_preimages.clone(),
        IntExpr::constant(config.witness_size),
        config.witness_size,
    );
    let k_preimage = builder.artifact_input(
        "k_preimage_artifact",
        preimage_type.clone(),
        production_id.clone(),
        names.k_preimage.clone(),
    );
    let decoder_preimage = builder.artifact_input(
        "decoder_preimage_artifact",
        preimage_type.clone(),
        production_id.clone(),
        names.decoder_preimage.clone(),
    );

    let public_keys = builder.artifact_family_input(
        "public_key_artifacts",
        public_key_type.clone(),
        production_id.clone(),
        names.public_keys.clone(),
        IntExpr::constant(config.witness_size.saturating_add(1)),
        config.witness_size.saturating_add(1),
    );
    let one_vector = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        &final_states[0],
        &one_preimage,
        public_key_type.clone(),
    );
    let one_plaintext = builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Identity);
    let one_encoding = BggEncodingWire {
        vector: one_vector,
        pubkey: BggPublicKeyWire { matrix: public_keys[0].clone() },
        plaintext: Some(one_plaintext.clone()),
    };
    let zero_encoding = zero_encoding(&mut builder, &one_encoding);

    let mut circuit_inputs = Vec::with_capacity(config.witness_size + config.instance_size);
    for bit in 0..config.witness_size {
        let digit = bit / config.injector.batch_bits;
        let bit_in_digit = bit % config.injector.batch_bits;
        let state = 1 + bit;
        let vector = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &final_states[state],
            &witness_preimages[bit],
            public_key_type.clone(),
        );
        let bit_wire = builder.bit_extract(digit_wires[digit], IntExpr::constant(bit_in_digit));
        let bit_index = builder.bool_to_int(bit_wire);
        let zero_plaintext = builder.constant_matrix(scalar_type.clone(), ConstantMatrix::Zero);
        let bit_plaintext = builder.select(bit_index, &[zero_plaintext, one_plaintext.clone()]);
        circuit_inputs.push(BggEncodingWire {
            vector,
            pubkey: BggPublicKeyWire { matrix: public_keys[bit + 1].clone() },
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

    let root_state = &final_states[0];
    let k_full_vector = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        root_state,
        &k_preimage,
        public_key_type.clone(),
    );
    let decoder_full = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        root_state,
        &decoder_preimage,
        public_key_type.clone(),
    );
    let k_vector = builder.slice(
        &k_full_vector,
        None,
        Some(IndexRange { start: 0, end: 1 }),
        scalar_type.clone(),
    );
    let decoder = builder.slice(
        &decoder_full,
        None,
        Some(IndexRange { start: 0, end: 1 }),
        scalar_type.clone(),
    );
    let r_decomposed = builder.artifact_input(
        "r_decomposed_artifact",
        MatrixType {
            rows: config.bgg_columns.clone(),
            columns: config.bgg_columns.clone(),
            ..scalar_type.clone()
        },
        production_id,
        names.r_decomposed.clone(),
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
        public_key_type,
    );
    let projection = builder.slice(
        &full_projection,
        None,
        Some(IndexRange { start: 0, end: 1 }),
        scalar_type.clone(),
    );
    let encoded =
        builder.matrix_binary(MatrixBinaryOp::Add, &k_vector, &projection, scalar_type.clone());
    let noisy_plaintext =
        builder.matrix_binary(MatrixBinaryOp::Subtract, &decoder, &encoded, scalar_type);
    let decoded = builder.threshold_decode(
        &noisy_plaintext,
        IntExpr::constant(2),
        IntExpr::constant(1),
        true,
    );
    builder.output_wire("message", decoded);
    Ok(builder.finish())
}

fn zero_encoding(builder: &mut GraphBuilder, one: &BggEncodingWire) -> BggEncodingWire {
    BggEncodingWire {
        vector: builder.matrix_scale(&one.vector, IntExpr::constant(0)),
        pubkey: BggPublicKeyWire {
            matrix: builder.matrix_scale(&one.pubkey.matrix, IntExpr::constant(0)),
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
        artifact::{ExportArtifact, SpecHash, export_manifest},
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
        builder.output("stacked", &stacked);
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
        assert!(validated.outputs.contains_key("diamond_transition_level_2_state_2"));
    }

    #[test]
    fn evaluation_graph_uses_one_select_per_level_state() {
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let graph = build_evaluation_graph(
            &config(),
            &DiamondInjectorArtifactNames::default(),
            production_id,
        )
        .expect("graph");
        assert_eq!(
            graph.nodes.iter().filter(|node| matches!(node.kind, NodeKind::Select { .. })).count(),
            5
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
        assert!(validated.outputs.contains_key("we_one_preimage"));
        assert!(validated.outputs.contains_key("we_witness_preimages"));
        assert!(validated.outputs.contains_key("we_k_preimage"));
        assert!(validated.outputs.contains_key("we_decoder_preimage"));
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
        let artifacts = keygen
            .outputs
            .iter()
            .map(|(name, wire)| {
                let node = keygen.source.node(wire.node).expect("output node");
                let family = matches!(node.kind, NodeKind::Output { .. })
                    .then(|| {
                        node.args
                            .iter()
                            .enumerate()
                            .map(|(port, _)| WireId {
                                instantiation_path: Vec::new(),
                                wire: WireRef { node: wire.node, port: Port(port as u32) },
                            })
                            .collect::<Vec<_>>()
                    })
                    .filter(|family| family.len() > 1);
                let id = WireId { instantiation_path: Vec::new(), wire: *wire };
                let wire_type = keygen.wires[&id].matrix_type().expect("matrix output").clone();
                (
                    name.clone(),
                    ExportArtifact {
                        wire: id,
                        wire_type,
                        family,
                        content_hash: None,
                        layout: None,
                    },
                )
            })
            .collect();
        let manifest = export_manifest(production_id.clone(), &artifacts);
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
