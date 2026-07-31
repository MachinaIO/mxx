use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, RealExpr, TrapdoorFamilyWire,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix, IndexRange, LoopInputMode, MatrixBinaryOp, SampleRange},
    types::MatrixType,
};
use num_bigint::BigInt;
use thiserror::Error;

pub const DIAMOND_INITIAL_STATE: &str = "diamond_input_initial_state";
pub const DIAMOND_FINAL_PUBLIC: &str = "diamond_input_final_public";
pub const DIAMOND_FINAL_TRAPDOORS: &str = "diamond_input_final_trapdoors";

const DIAMOND_STATE_ROWS: usize = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputInjectionCompiler {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub input_count: usize,
    pub base: usize,
    pub batch_bits: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub chunk_columns: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputInjectionArtifacts {
    production_id: ProductionId,
    retained_input_digits: Option<Vec<u32>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputInjectionPreprocessingWires {
    pub initial_state: MatrixWire,
    pub final_public: MatrixFamilyWire,
    pub final_trapdoors: TrapdoorFamilyWire,
    transitions: Vec<Vec<Vec<MatrixFamilyWire>>>,
    level_public: Vec<MatrixFamilyWire>,
    secret_epsilon: MatrixWire,
    digit_secret_masks: Vec<Vec<MatrixWire>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputInjectionWires {
    pub states: Vec<MatrixWire>,
    pub final_public: MatrixFamilyWire,
    pub final_trapdoors: TrapdoorFamilyWire,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum DiamondInputInjectionError {
    #[error("Diamond input injection requires a positive base")]
    ZeroBase,
    #[error("Diamond input injection requires a positive batch-bit count")]
    ZeroBatchBits,
    #[error("Diamond input injection batch-bit count exceeds u32")]
    BatchBitsTooLarge,
    #[error("Diamond input injection base must be at least 2^batch_bits")]
    BaseTooSmall,
    #[error("Diamond input injection requires a positive gadget digit count")]
    ZeroDigitCount,
    #[error("Diamond input injection requires a positive transition chunk width")]
    ZeroChunkColumns,
    #[error("Diamond input injection expected {expected} digits but received {actual}")]
    DigitCountMismatch { expected: usize, actual: usize },
    #[error("Diamond input digit {value} at position {position} is outside base {base}")]
    DigitOutOfRange { position: usize, value: u32, base: usize },
    #[error("the artifact production retains a different input-digit sequence")]
    RetainedDigitMismatch,
    #[error(transparent)]
    Subgraph(#[from] mxx_ir_core::SubgraphBuildError),
    #[error(transparent)]
    Family(#[from] mxx_ir_core::OutputFamilyError),
}

impl DiamondInputInjectionArtifacts {
    pub fn new(
        production_id: ProductionId,
        compiler: &DiamondInputInjectionCompiler,
        retained_input_digits: Option<&[u32]>,
    ) -> Result<Self, DiamondInputInjectionError> {
        if let Some(digits) = retained_input_digits {
            compiler.validate_digits(digits)?;
        }
        Ok(Self {
            production_id,
            retained_input_digits: retained_input_digits.map(<[u32]>::to_vec),
        })
    }

    pub fn production_id(&self) -> &ProductionId {
        &self.production_id
    }
}

impl DiamondInputInjectionCompiler {
    pub fn validate_layout(&self) -> Result<(), DiamondInputInjectionError> {
        if self.base == 0 {
            return Err(DiamondInputInjectionError::ZeroBase);
        }
        if self.batch_bits == 0 {
            return Err(DiamondInputInjectionError::ZeroBatchBits);
        }
        if self.batch_bits > u32::BITS as usize {
            return Err(DiamondInputInjectionError::BatchBitsTooLarge);
        }
        let minimum_base = 1usize
            .checked_shl(self.batch_bits as u32)
            .ok_or(DiamondInputInjectionError::BaseTooSmall)?;
        if self.base < minimum_base {
            return Err(DiamondInputInjectionError::BaseTooSmall);
        }
        if self.digit_count == 0 {
            return Err(DiamondInputInjectionError::ZeroDigitCount);
        }
        if self.chunk_columns == 0 {
            return Err(DiamondInputInjectionError::ZeroChunkColumns);
        }
        Ok(())
    }

    pub fn state_count_at_level(&self, level: usize) -> usize {
        1usize
            .checked_add(
                level
                    .checked_mul(self.batch_bits)
                    .expect("Diamond input-injection state count overflow"),
            )
            .expect("Diamond input-injection state count overflow")
    }

    pub fn bit_state_index(&self, input: usize, bit: usize) -> usize {
        assert!(bit < self.batch_bits, "Diamond input-injection bit index out of range");
        1usize
            .checked_add(
                input
                    .checked_mul(self.batch_bits)
                    .expect("Diamond input-injection bit-state index overflow"),
            )
            .and_then(|index| index.checked_add(bit))
            .expect("Diamond input-injection bit-state index overflow")
    }

    pub fn build_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        plaintext: &MatrixWire,
    ) -> Result<DiamondInputInjectionPreprocessingWires, DiamondInputInjectionError> {
        self.validate_layout()?;
        assert_eq!(
            plaintext.matrix_type,
            self.scalar_type(),
            "Diamond input-injection plaintext must be a 1x1 matrix"
        );

        let mut level_trapdoors = Vec::with_capacity(self.input_count + 1);
        let mut level_public = Vec::with_capacity(self.input_count + 1);
        for level in 0..=self.input_count {
            let trapdoors = self.build_level_trapdoors(builder, level)?;
            let public = self.build_level_public(builder, level, &trapdoors)?;
            level_trapdoors.push(trapdoors);
            level_public.push(public);
        }

        let secret_epsilon = builder.uniform_sample(
            self.scalar_type(),
            SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
        );
        let initial_selector = builder.concat(
            ConcatAxis::Columns,
            &[secret_epsilon.clone(), plaintext.clone()],
            self.matrix_type(1, DIAMOND_STATE_ROWS),
        );
        let b0 = builder.family_get_static(&level_public[0], IntExpr::constant(0));
        let initial = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &initial_selector,
            &b0,
            self.state_type(),
        );
        let initial_state = self.add_error(builder, initial);

        let mut transitions = Vec::with_capacity(self.input_count);
        let mut digit_secret_masks = Vec::with_capacity(self.input_count);
        for level in 1..=self.input_count {
            let mut level_transitions = Vec::with_capacity(self.base);
            let mut level_masks = Vec::with_capacity(self.base);
            for digit in 0..self.base {
                let mask = builder.uniform_sample(
                    self.scalar_type(),
                    SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                );
                let chunks = self.build_transition_chunks(
                    builder,
                    level,
                    digit,
                    &mask,
                    &level_trapdoors[level - 1],
                    &level_public[level],
                )?;
                level_masks.push(mask);
                level_transitions.push(chunks);
            }
            digit_secret_masks.push(level_masks);
            transitions.push(level_transitions);
        }

        Ok(DiamondInputInjectionPreprocessingWires {
            initial_state,
            final_public: level_public[self.input_count].clone(),
            final_trapdoors: level_trapdoors[self.input_count].clone(),
            transitions,
            level_public,
            secret_epsilon,
            digit_secret_masks,
        })
    }

    pub fn export_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        wires: &DiamondInputInjectionPreprocessingWires,
        retained_input_digits: Option<&[u32]>,
    ) -> Result<(), DiamondInputInjectionError> {
        if let Some(digits) = retained_input_digits {
            self.validate_digits(digits)?;
        }
        builder.output(
            DIAMOND_INITIAL_STATE,
            &wires.initial_state,
            ArtifactConfidentiality::Public,
        );
        builder.output_family_wire(
            DIAMOND_FINAL_PUBLIC,
            &wires.final_public,
            ArtifactConfidentiality::Public,
        );
        builder.output_wire(
            DIAMOND_FINAL_TRAPDOORS,
            wires.final_trapdoors.wire,
            ArtifactConfidentiality::Private,
        );
        for level in 1..=self.input_count {
            for digit in 0..self.base {
                let retained =
                    retained_input_digits.is_none_or(|digits| digits[level - 1] as usize == digit);
                if !retained {
                    continue;
                }
                for (chunk, family) in wires.transitions[level - 1][digit].iter().enumerate() {
                    builder.output_family_wire(
                        self.transition_name(level, digit, chunk),
                        family,
                        ArtifactConfidentiality::Public,
                    );
                }
            }
        }
        Ok(())
    }

    pub fn build_online(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &DiamondInputInjectionArtifacts,
        input_digits: &[u32],
    ) -> Result<DiamondInputInjectionWires, DiamondInputInjectionError> {
        self.validate_layout()?;
        self.validate_digits(input_digits)?;
        if artifacts
            .retained_input_digits
            .as_deref()
            .is_some_and(|retained| retained != input_digits)
        {
            return Err(DiamondInputInjectionError::RetainedDigitMismatch);
        }
        let initial_state = builder.artifact_input(
            "diamond_input_initial_state_input",
            self.state_type(),
            artifacts.production_id.clone(),
            DIAMOND_INITIAL_STATE,
            ArtifactConfidentiality::Public,
        );
        let final_public = builder.artifact_family_input(
            "diamond_input_final_public_input",
            self.state_public_type(),
            artifacts.production_id.clone(),
            DIAMOND_FINAL_PUBLIC,
            IntExpr::constant(self.state_count_at_level(self.input_count)),
            ArtifactConfidentiality::Public,
        );
        let final_trapdoors = builder.artifact_trapdoor_family_input(
            "diamond_input_final_trapdoors_input",
            self.state_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
            artifacts.production_id.clone(),
            DIAMOND_FINAL_TRAPDOORS,
            IntExpr::constant(self.state_count_at_level(self.input_count)),
            ArtifactConfidentiality::Private,
        );

        let mut states = vec![initial_state];
        for (input, digit) in input_digits.iter().copied().enumerate() {
            let level = input + 1;
            let previous = states;
            let previous_zero = previous[0].clone();
            let families = self
                .chunks()
                .into_iter()
                .enumerate()
                .map(|(chunk, columns)| {
                    builder.artifact_family_input(
                        format!("diamond_transition_l{level}_d{digit}_c{chunk}_input"),
                        self.matrix_type(self.state_columns(), columns.end - columns.start),
                        artifacts.production_id.clone(),
                        self.transition_name(level, digit as usize, chunk),
                        IntExpr::constant(self.state_count_at_level(level)),
                        ArtifactConfidentiality::Public,
                    )
                })
                .collect::<Vec<_>>();
            states = (0..self.state_count_at_level(level))
                .map(|state| {
                    let lhs = if self.new_bit_index(level, state).is_some() {
                        &previous_zero
                    } else {
                        &previous[state]
                    };
                    let products = families
                        .iter()
                        .map(|family| {
                            let transition =
                                builder.family_get_static(family, IntExpr::constant(state));
                            builder.matrix_binary(
                                MatrixBinaryOp::Multiply,
                                lhs,
                                &transition,
                                self.matrix_type_expr(1, transition.matrix_type.columns.clone()),
                            )
                        })
                        .collect::<Vec<_>>();
                    if products.len() == 1 {
                        products[0].clone()
                    } else {
                        builder.concat(ConcatAxis::Columns, &products, self.state_type())
                    }
                })
                .collect();
        }
        Ok(DiamondInputInjectionWires { states, final_public, final_trapdoors })
    }

    fn build_level_trapdoors(
        &self,
        builder: &mut GraphBuilder,
        level: usize,
    ) -> Result<TrapdoorFamilyWire, DiamondInputInjectionError> {
        let mut body = GraphBuilder::new(format!("diamond-input-b-level-{level}"), Vec::new());
        let trapdoor = body.trapdoor_sample(
            self.state_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        body.value_output_wire("0_trapdoor", trapdoor.wire);
        Ok(builder.parallel_trapdoor_loop(
            body.finish(),
            IntExpr::constant(self.state_count_at_level(level)),
            "state",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            self.state_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        )?)
    }

    fn build_level_public(
        &self,
        builder: &mut GraphBuilder,
        level: usize,
        trapdoors: &TrapdoorFamilyWire,
    ) -> Result<MatrixFamilyWire, DiamondInputInjectionError> {
        let mut body =
            GraphBuilder::new(format!("diamond-input-b-public-level-{level}"), Vec::new());
        let trapdoor = body.trapdoor_input(
            "0_trapdoor",
            self.state_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        body.value_output_wire("0_public", trapdoor.public.wire);
        Ok(builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(self.state_count_at_level(level)),
                "state",
                Vec::new(),
                vec![trapdoors.wire],
                vec![LoopInputMode::Zip],
                &[self.state_public_type()],
            )?
            .remove(0))
    }

    fn build_transition_chunks(
        &self,
        builder: &mut GraphBuilder,
        level: usize,
        digit: usize,
        secret_mask: &MatrixWire,
        previous_trapdoors: &TrapdoorFamilyWire,
        target_public: &MatrixFamilyWire,
    ) -> Result<Vec<MatrixFamilyWire>, DiamondInputInjectionError> {
        let source_trapdoors = (0..self.state_count_at_level(level))
            .map(|state| {
                builder.trapdoor_family_get_static(
                    previous_trapdoors,
                    IntExpr::constant(self.transition_source_state(level, state)),
                )
            })
            .collect::<Vec<_>>();
        let source_trapdoors = builder.trapdoor_family_pack(&source_trapdoors)?;
        let selectors = (0..self.state_count_at_level(level))
            .map(|state| self.transition_selector(builder, level, digit, state, secret_mask))
            .collect::<Vec<_>>();
        let selectors = builder.family_pack(&selectors)?;
        self.chunks()
            .into_iter()
            .enumerate()
            .map(|(chunk, columns)| {
                self.build_transition_chunk_family(
                    builder,
                    level,
                    digit,
                    chunk,
                    columns,
                    &source_trapdoors,
                    target_public,
                    &selectors,
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn build_transition_chunk_family(
        &self,
        builder: &mut GraphBuilder,
        level: usize,
        digit: usize,
        chunk: usize,
        columns: IndexRange,
        source_trapdoors: &TrapdoorFamilyWire,
        target_public: &MatrixFamilyWire,
        selectors: &MatrixFamilyWire,
    ) -> Result<MatrixFamilyWire, DiamondInputInjectionError> {
        let width = columns.end - columns.start;
        let mut body = GraphBuilder::new(
            format!("diamond-input-transition-l{level}-d{digit}-c{chunk}"),
            Vec::new(),
        );
        let source = body.trapdoor_input(
            "0_source",
            self.state_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let body_target_public = body.input("1_target", self.state_public_type());
        let selector =
            body.input("2_selector", self.matrix_type(DIAMOND_STATE_ROWS, DIAMOND_STATE_ROWS));
        let public_chunk = body.slice(
            &body_target_public,
            None,
            Some(columns),
            self.matrix_type(DIAMOND_STATE_ROWS, width),
        );
        let target = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &selector,
            &public_chunk,
            self.matrix_type(DIAMOND_STATE_ROWS, width),
        );
        let target = self.add_error(&mut body, target);
        let preimage =
            body.preimage_sample(&source, &target, self.matrix_type(self.state_columns(), width));
        body.value_output_wire("0_preimage", preimage.wire);
        Ok(builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(self.state_count_at_level(level)),
                "state",
                Vec::new(),
                vec![source_trapdoors.wire, target_public.wire, selectors.wire],
                vec![LoopInputMode::Zip; 3],
                &[preimage.matrix_type],
            )?
            .remove(0))
    }

    fn transition_selector(
        &self,
        builder: &mut GraphBuilder,
        level: usize,
        digit: usize,
        state: usize,
        secret_mask: &MatrixWire,
    ) -> MatrixWire {
        let zero = builder.constant_matrix(self.scalar_type(), ConstantMatrix::Zero);
        if let Some(bit) = self.new_bit_index(level, state) {
            let bit_mask = builder.matrix_scale(secret_mask, IntExpr::constant((digit >> bit) & 1));
            let top = builder.concat(
                ConcatAxis::Columns,
                &[secret_mask.clone(), bit_mask],
                self.matrix_type(1, DIAMOND_STATE_ROWS),
            );
            let bottom = builder.concat(
                ConcatAxis::Columns,
                &[zero.clone(), zero],
                self.matrix_type(1, DIAMOND_STATE_ROWS),
            );
            return builder.concat(
                ConcatAxis::Rows,
                &[top, bottom],
                self.matrix_type(DIAMOND_STATE_ROWS, DIAMOND_STATE_ROWS),
            );
        }
        let lower_right = if state == 0 {
            builder.constant_matrix(self.scalar_type(), ConstantMatrix::Identity)
        } else {
            secret_mask.clone()
        };
        let top = builder.concat(
            ConcatAxis::Columns,
            &[secret_mask.clone(), zero.clone()],
            self.matrix_type(1, DIAMOND_STATE_ROWS),
        );
        let bottom = builder.concat(
            ConcatAxis::Columns,
            &[zero, lower_right],
            self.matrix_type(1, DIAMOND_STATE_ROWS),
        );
        builder.concat(
            ConcatAxis::Rows,
            &[top, bottom],
            self.matrix_type(DIAMOND_STATE_ROWS, DIAMOND_STATE_ROWS),
        )
    }

    fn validate_digits(&self, digits: &[u32]) -> Result<(), DiamondInputInjectionError> {
        if digits.len() != self.input_count {
            return Err(DiamondInputInjectionError::DigitCountMismatch {
                expected: self.input_count,
                actual: digits.len(),
            });
        }
        for (position, value) in digits.iter().copied().enumerate() {
            if value as usize >= self.base {
                return Err(DiamondInputInjectionError::DigitOutOfRange {
                    position,
                    value,
                    base: self.base,
                });
            }
        }
        Ok(())
    }

    fn new_bit_index(&self, level: usize, state: usize) -> Option<usize> {
        debug_assert!(level > 0);
        let first = 1 + (level - 1) * self.batch_bits;
        if (first..first + self.batch_bits).contains(&state) { Some(state - first) } else { None }
    }

    fn transition_source_state(&self, level: usize, state: usize) -> usize {
        if self.new_bit_index(level, state).is_some() { 0 } else { state }
    }

    fn add_error(&self, builder: &mut GraphBuilder, value: MatrixWire) -> MatrixWire {
        let error = builder.gaussian_sample(value.matrix_type.clone(), self.error_sigma.clone());
        builder.matrix_binary(MatrixBinaryOp::Add, &value, &error, value.matrix_type.clone())
    }

    fn chunks(&self) -> Vec<IndexRange> {
        (0..self.state_columns())
            .step_by(self.chunk_columns)
            .map(|start| IndexRange {
                start,
                end: (start + self.chunk_columns).min(self.state_columns()),
            })
            .collect()
    }

    fn transition_name(&self, level: usize, digit: usize, chunk: usize) -> String {
        format!("diamond_input_transition_level_{level}_digit_{digit}_chunk_{chunk}")
    }

    fn state_columns(&self) -> usize {
        DIAMOND_STATE_ROWS
            .checked_mul(self.digit_count + 2)
            .expect("Diamond input-injection public column count overflow")
    }

    fn scalar_type(&self) -> MatrixType {
        self.matrix_type(1, 1)
    }

    fn state_type(&self) -> MatrixType {
        self.matrix_type(1, self.state_columns())
    }

    fn state_public_type(&self) -> MatrixType {
        self.matrix_type(DIAMOND_STATE_ROWS, self.state_columns())
    }

    fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
        self.matrix_type_expr(rows, IntExpr::constant(columns))
    }

    fn matrix_type_expr(&self, rows: usize, columns: IntExpr) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(rows),
            columns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ParamEnv, artifact::SpecHash, validate, validate_with_manifests};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::{CpuDcrtBackend, cpu_backend},
        execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    fn output_matrix(
        result: &mxx_runtime::ExecutionResult<CpuDcrtBackend>,
        name: &str,
    ) -> DCRTPolyMatrix {
        let RuntimeValue::Matrix(value) = &result.outputs[name] else {
            panic!("{name} must be a matrix");
        };
        value.as_ref().clone()
    }

    #[test]
    #[serial_test::serial]
    fn preprocessing_and_online_graph_preserve_diamond_state_relations() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let compiler = DiamondInputInjectionCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            input_count: 2,
            base: 4,
            batch_bits: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            error_sigma: RealExpr::from_f64_exact(0.0).expect("finite sigma"),
            chunk_columns: 3,
        };
        let input_digits = [3, 1];
        let plaintext_value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_usize_to_constant(&parameters, 7)],
        );

        let mut producer = GraphBuilder::new("diamond-input-preprocessing-test", Vec::new());
        let plaintext = producer.input("plaintext", compiler.scalar_type());
        let wires =
            compiler.build_preprocessing(&mut producer, &plaintext).expect("preprocessing graph");
        producer.value_output_wire("inspect_secret_epsilon", wires.secret_epsilon.wire);
        for (level, digit) in input_digits.iter().copied().enumerate() {
            producer.value_output_wire(
                format!("inspect_mask_{}", level + 1),
                wires.digit_secret_masks[level][digit as usize].wire,
            );
        }
        for state in 0..compiler.state_count_at_level(compiler.input_count) {
            let public = producer.family_get_static(
                &wires.level_public[compiler.input_count],
                IntExpr::constant(state),
            );
            producer.value_output_wire(format!("inspect_final_public_{state}"), public.wire);
        }
        compiler
            .export_preprocessing(&mut producer, &wires, Some(&input_digits))
            .expect("preprocessing exports");
        let producer = validate(&producer.finish(), &ParamEnv::default()).expect("producer graph");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([(
                "plaintext".to_owned(),
                RuntimeValue::matrix(plaintext_value.clone()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("preprocessing execution");
        let production_id = produced.production_id.clone().expect("artifact production");
        let manifest = store.manifest(&production_id).expect("producer manifest").clone();
        assert_eq!(
            manifest.artifacts[DIAMOND_FINAL_PUBLIC].confidentiality,
            ArtifactConfidentiality::Public
        );
        assert_eq!(
            manifest.artifacts[DIAMOND_FINAL_TRAPDOORS].confidentiality,
            ArtifactConfidentiality::Private
        );
        assert!(manifest.artifacts[DIAMOND_FINAL_TRAPDOORS].content_hash.is_none());

        let artifacts = DiamondInputInjectionArtifacts::new(
            production_id.clone(),
            &compiler,
            Some(&input_digits),
        )
        .expect("artifact descriptor");
        let mut consumer = GraphBuilder::new("diamond-input-online-test", Vec::new());
        let online =
            compiler.build_online(&mut consumer, &artifacts, &input_digits).expect("online graph");
        for (state, value) in online.states.iter().enumerate() {
            consumer.value_output_wire(format!("state_{state}"), value.wire);
        }
        let imported_public =
            consumer.family_get_static(&online.final_public, IntExpr::constant(0));
        let imported_trapdoor =
            consumer.trapdoor_family_get_static(&online.final_trapdoors, IntExpr::constant(0));
        consumer.value_output_wire("inspect_imported_public", imported_public.wire);
        consumer
            .value_output_wire("inspect_imported_trapdoor_public", imported_trapdoor.public.wire);
        let consumer = validate_with_manifests(
            &consumer.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([(production_id, manifest)]),
        )
        .expect("consumer graph");
        let consumed =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("online execution");
        assert_eq!(
            output_matrix(&consumed, "inspect_imported_public"),
            output_matrix(&consumed, "inspect_imported_trapdoor_public")
        );

        let mut accumulated_secret = output_matrix(&produced, "inspect_secret_epsilon");
        for level in 1..=compiler.input_count {
            accumulated_secret =
                accumulated_secret * output_matrix(&produced, &format!("inspect_mask_{level}"));
        }
        let secret = accumulated_secret.entry(0, 0);
        for state in 0..compiler.state_count_at_level(compiler.input_count) {
            let second = if state == 0 {
                plaintext_value.entry(0, 0)
            } else {
                let bit_index = state - 1;
                let input = bit_index / compiler.batch_bits;
                let bit = bit_index % compiler.batch_bits;
                secret.clone() *
                    DCRTPoly::from_usize_to_constant(
                        &parameters,
                        ((input_digits[input] as usize) >> bit) & 1,
                    )
            };
            let selector =
                DCRTPolyMatrix::from_poly_vec_row(&parameters, vec![secret.clone(), second]);
            let expected =
                selector * output_matrix(&produced, &format!("inspect_final_public_{state}"));
            assert_eq!(output_matrix(&consumed, &format!("state_{state}")), expected);
        }
        assert!(
            compiler
                .chunks()
                .last()
                .is_some_and(|chunk| { chunk.end - chunk.start < compiler.chunk_columns })
        );
    }

    #[test]
    fn retained_artifacts_reject_a_different_digit_sequence() {
        let compiler = DiamondInputInjectionCompiler {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            input_count: 1,
            base: 2,
            batch_bits: 1,
            digit_count: 2,
            gadget_base: IntExpr::constant(4),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            error_sigma: RealExpr::from_f64_exact(0.0).expect("finite sigma"),
            chunk_columns: 2,
        };
        let artifacts = DiamondInputInjectionArtifacts::new(
            ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] },
            &compiler,
            Some(&[0]),
        )
        .expect("artifact descriptor");
        let mut builder = GraphBuilder::new("diamond-input-retained-mismatch", Vec::new());
        assert_eq!(
            compiler.build_online(&mut builder, &artifacts, &[1]),
            Err(DiamondInputInjectionError::RetainedDigitMismatch)
        );
    }
}
