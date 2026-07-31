use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, RealExpr, SubgraphBuildError,
    TrapdoorWire, ValueFamilyWire,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix, IndexRange, LoopInputMode, MatrixBinaryOp},
    types::MatrixType,
};
use thiserror::Error;

pub const MASKED_DECODER_PREIMAGES: &str = "masked_decoder_preimages";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaskedHighBitDecoderCompiler {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_size: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
    pub trapdoor_sigma: RealExpr,
    pub coefficient_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaskedHighBitDecoderArtifacts {
    pub production_id: ProductionId,
    pub slot_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaskedHighBitDecoderPreprocessingWires {
    pub preimages: MatrixFamilyWire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaskedHighBitDecoderOutputs {
    /// One family per decoded coefficient. Each family is indexed by decoder slot.
    pub coefficients: Vec<ValueFamilyWire>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum MaskedHighBitDecoderError {
    #[error("masked decoder dimensions and coefficient count must be nonzero")]
    EmptyLayout,
    #[error("masked decoder input families must have the artifact slot count")]
    FamilyCountMismatch,
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
}

impl MaskedHighBitDecoderCompiler {
    pub fn validate_layout(&self) -> Result<(), MaskedHighBitDecoderError> {
        if self.secret_size == 0 || self.digit_count == 0 || self.coefficient_count == 0 {
            return Err(MaskedHighBitDecoderError::EmptyLayout);
        }
        Ok(())
    }

    pub fn public_key_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.public_key_columns())
    }

    pub fn decoder_public_type(&self) -> MatrixType {
        self.matrix_type(self.decoder_rows(), self.decoder_columns())
    }

    pub fn decoder_state_type(&self) -> MatrixType {
        self.matrix_type(1, self.decoder_columns())
    }

    pub fn encoding_vector_type(&self) -> MatrixType {
        self.matrix_type(1, self.public_key_columns())
    }

    pub fn scalar_type(&self) -> MatrixType {
        self.matrix_type(1, 1)
    }

    pub fn build_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        decoder_trapdoor: &TrapdoorWire,
        public_keys: &MatrixFamilyWire,
        slot_count: usize,
    ) -> Result<MaskedHighBitDecoderPreprocessingWires, MaskedHighBitDecoderError> {
        self.validate_layout()?;
        if public_keys.count != IntExpr::constant(slot_count) {
            return Err(MaskedHighBitDecoderError::FamilyCountMismatch);
        }

        let mut body = GraphBuilder::new(
            format!("bgg-masked-decoder-preprocess-d{}-k{}", self.secret_size, self.digit_count),
            Vec::new(),
        );
        let body_public_key = body.input("0_public_key", self.public_key_type());
        let body_trapdoor = body.trapdoor_input(
            "1_decoder_trapdoor",
            self.decoder_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let identity = body.constant_matrix(
            self.matrix_type(self.secret_size, self.secret_size),
            ConstantMatrix::Identity,
        );
        let selector = body.slice(
            &identity,
            None,
            Some(IndexRange { start: 0, end: 1 }),
            self.matrix_type(self.secret_size, 1),
        );
        let decomposed_selector = body.gadget_decompose_with_layout(
            &selector,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(self.public_key_columns(), 1),
        );
        let top = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &body_public_key,
            &decomposed_selector,
            self.matrix_type(self.secret_size, 1),
        );
        let bottom =
            body.constant_matrix(self.matrix_type(self.secret_size, 1), ConstantMatrix::Zero);
        let target =
            body.concat(ConcatAxis::Rows, &[top, bottom], self.matrix_type(self.decoder_rows(), 1));
        let preimage = body.preimage_sample(
            &body_trapdoor,
            &target,
            self.matrix_type(self.decoder_columns(), 1),
        );
        body.value_output_wire("0_preimage", preimage.wire);

        let [preimages] = builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(slot_count),
                "slot",
                Vec::new(),
                vec![public_keys.wire, decoder_trapdoor.wire],
                vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
                &[self.matrix_type(self.decoder_columns(), 1)],
            )?
            .try_into()
            .expect("one preprocessing output was declared");
        Ok(MaskedHighBitDecoderPreprocessingWires { preimages })
    }

    pub fn export_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        wires: &MaskedHighBitDecoderPreprocessingWires,
    ) {
        builder.output_family_wire(
            MASKED_DECODER_PREIMAGES,
            &wires.preimages,
            ArtifactConfidentiality::Public,
        );
    }

    pub fn import_preimages(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &MaskedHighBitDecoderArtifacts,
    ) -> Result<MatrixFamilyWire, MaskedHighBitDecoderError> {
        self.validate_layout()?;
        Ok(builder.artifact_family_input(
            "masked_decoder_preimages_input",
            self.matrix_type(self.decoder_columns(), 1),
            artifacts.production_id.clone(),
            MASKED_DECODER_PREIMAGES,
            IntExpr::constant(artifacts.slot_count),
            ArtifactConfidentiality::Public,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_online(
        &self,
        builder: &mut GraphBuilder,
        decoder_state: &MatrixWire,
        preimages: &MatrixFamilyWire,
        secret_dependent_vectors: &MatrixFamilyWire,
        public_bottom_plaintexts: &MatrixFamilyWire,
        plaintext_modulus: IntExpr,
        output_bool: bool,
        slot_count: usize,
    ) -> Result<MaskedHighBitDecoderOutputs, MaskedHighBitDecoderError> {
        self.validate_layout()?;
        let expected_count = IntExpr::constant(slot_count);
        if preimages.count != expected_count ||
            secret_dependent_vectors.count != expected_count ||
            public_bottom_plaintexts.count != expected_count
        {
            return Err(MaskedHighBitDecoderError::FamilyCountMismatch);
        }

        let mut body = GraphBuilder::new(
            format!(
                "bgg-masked-decoder-online-d{}-k{}-coeff{}-bool{}",
                self.secret_size, self.digit_count, self.coefficient_count, output_bool
            ),
            Vec::new(),
        );
        let body_preimage = body.input("0_preimage", self.matrix_type(self.decoder_columns(), 1));
        let body_vector = body.input("1_secret_dependent", self.encoding_vector_type());
        let body_public_bottom = body.input("2_public_bottom", self.scalar_type());
        let body_decoder_state = body.input("3_decoder_state", self.decoder_state_type());

        let identity = body.constant_matrix(
            self.matrix_type(self.secret_size, self.secret_size),
            ConstantMatrix::Identity,
        );
        let selector = body.slice(
            &identity,
            None,
            Some(IndexRange { start: 0, end: 1 }),
            self.matrix_type(self.secret_size, 1),
        );
        let decomposed_selector = body.gadget_decompose_with_layout(
            &selector,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(self.public_key_columns(), 1),
        );
        let projected_decoder = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &body_decoder_state,
            &body_preimage,
            self.scalar_type(),
        );
        let projected_encoding = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &body_vector,
            &decomposed_selector,
            self.scalar_type(),
        );
        let difference = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &projected_decoder,
            &projected_encoding,
            self.scalar_type(),
        );
        let noisy_plaintext = body.matrix_binary(
            MatrixBinaryOp::Add,
            &difference,
            &body_public_bottom,
            self.scalar_type(),
        );
        let first = body.threshold_decode(
            &noisy_plaintext,
            plaintext_modulus,
            IntExpr::constant(self.coefficient_count),
            output_bool,
        );
        for coefficient in 0..self.coefficient_count {
            body.value_output_wire(
                format!("{coefficient:08}_coefficient"),
                mxx_ir_core::WireRef {
                    node: first.node,
                    port: mxx_ir_core::Port(coefficient as u32),
                },
            );
        }
        let coefficients = builder.parallel_value_loop(
            body.finish(),
            expected_count,
            "slot",
            Vec::new(),
            vec![
                preimages.wire,
                secret_dependent_vectors.wire,
                public_bottom_plaintexts.wire,
                decoder_state.wire,
            ],
            vec![
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Broadcast,
            ],
            self.coefficient_count,
        )?;
        Ok(MaskedHighBitDecoderOutputs { coefficients })
    }

    fn public_key_columns(&self) -> usize {
        self.secret_size * self.digit_count
    }

    fn decoder_rows(&self) -> usize {
        self.secret_size * 2
    }

    fn decoder_columns(&self) -> usize {
        self.decoder_rows() * (self.digit_count + 2)
    }

    fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ParamEnv, validate};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    fn compiler() -> MaskedHighBitDecoderCompiler {
        MaskedHighBitDecoderCompiler {
            modulus: IntExpr::constant(BigInt::from(65537)),
            ring_dimension: IntExpr::constant(8),
            secret_size: 2,
            digit_count: 4,
            gadget_base: IntExpr::constant(4),
            trapdoor_sigma: RealExpr::from_f64_exact(5.0).expect("finite sigma"),
            coefficient_count: 8,
        }
    }

    #[test]
    fn preprocessing_and_online_graphs_validate() {
        let compiler = compiler();
        let mut preprocessing = GraphBuilder::new("decoder-preprocessing", Vec::new());
        let trapdoor = preprocessing.trapdoor_sample(
            compiler.decoder_public_type(),
            compiler.trapdoor_sigma.clone(),
            compiler.gadget_base.clone(),
            IntExpr::constant(compiler.digit_count),
        );
        let public_keys = preprocessing.family_input(
            "public_keys",
            compiler.public_key_type(),
            IntExpr::constant(3),
        );
        let wires =
            compiler.build_preprocessing(&mut preprocessing, &trapdoor, &public_keys, 3).unwrap();
        compiler.export_preprocessing(&mut preprocessing, &wires);
        validate(&preprocessing.finish(), &ParamEnv::default()).unwrap();

        let mut online = GraphBuilder::new("decoder-online", Vec::new());
        let decoder_state = online.input("decoder_state", compiler.decoder_state_type());
        let preimages = online.family_input(
            "preimages",
            compiler.matrix_type(compiler.decoder_columns(), 1),
            IntExpr::constant(3),
        );
        let vectors =
            online.family_input("vectors", compiler.encoding_vector_type(), IntExpr::constant(3));
        let public_bottom =
            online.family_input("public_bottom", compiler.scalar_type(), IntExpr::constant(3));
        let outputs = compiler
            .build_online(
                &mut online,
                &decoder_state,
                &preimages,
                &vectors,
                &public_bottom,
                IntExpr::constant(2),
                true,
                3,
            )
            .unwrap();
        assert_eq!(outputs.coefficients.len(), 8);
        for (index, family) in outputs.coefficients.iter().enumerate() {
            online.value_output_wire(format!("decoded_{index}"), family.wire);
        }
        validate(&online.finish(), &ParamEnv::default()).unwrap();
    }

    #[test]
    fn online_execution_matches_explicit_threshold_decode_oracle() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let q = parameters.modulus().as_ref().clone();
        let digit_count = parameters.modulus_digits();
        let compiler = MaskedHighBitDecoderCompiler {
            modulus: IntExpr::constant(BigInt::from(q.clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            digit_count,
            gadget_base: IntExpr::constant(1u64 << parameters.base_bits()),
            trapdoor_sigma: RealExpr::from_f64_exact(5.0).unwrap(),
            coefficient_count: parameters.ring_dimension() as usize,
        };
        let slot_count = 3usize;
        let plaintext_modulus = BigUint::from(5u8);
        let mut builder = GraphBuilder::new("decoder-online-runtime", Vec::new());
        let decoder_state = builder.input("decoder_state", compiler.decoder_state_type());
        let preimages = builder.family_input(
            "preimages",
            compiler.matrix_type(compiler.decoder_columns(), 1),
            IntExpr::constant(slot_count),
        );
        let vectors = builder.family_input(
            "vectors",
            compiler.encoding_vector_type(),
            IntExpr::constant(slot_count),
        );
        let public_bottom = builder.family_input(
            "public_bottom",
            compiler.scalar_type(),
            IntExpr::constant(slot_count),
        );
        let outputs = compiler
            .build_online(
                &mut builder,
                &decoder_state,
                &preimages,
                &vectors,
                &public_bottom,
                IntExpr::constant(BigInt::from(plaintext_modulus.clone())),
                false,
                slot_count,
            )
            .unwrap();
        for (coefficient, family) in outputs.coefficients.iter().enumerate() {
            for slot in 0..slot_count {
                let value = builder.value_family_get_static(family, IntExpr::constant(slot));
                builder.value_output_wire(format!("slot_{slot}_coefficient_{coefficient}"), value);
            }
        }
        let validated = validate(&builder.finish(), &ParamEnv::default()).unwrap();

        let decoder_coefficients = (0..compiler.coefficient_count)
            .map(|coefficient| BigUint::from(coefficient + 7))
            .collect::<Vec<_>>();
        let decoder_polynomial = DCRTPoly::from_biguints(&parameters, &decoder_coefficients);
        let decoder_state_value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            std::iter::once(decoder_polynomial)
                .chain((1..compiler.decoder_columns()).map(|_| DCRTPoly::const_zero(&parameters)))
                .collect(),
        );
        let preimage_value = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            std::iter::once(vec![DCRTPoly::const_one(&parameters)])
                .chain(
                    (1..compiler.decoder_columns())
                        .map(|_| vec![DCRTPoly::const_zero(&parameters)]),
                )
                .collect(),
        );
        let mut vector_values = Vec::with_capacity(slot_count);
        let mut public_bottom_values = Vec::with_capacity(slot_count);
        let mut expected_values = Vec::with_capacity(slot_count);
        for slot in 0..slot_count {
            let secret_coefficients = (0..compiler.coefficient_count)
                .map(|coefficient| BigUint::from(coefficient + slot + 3))
                .collect::<Vec<_>>();
            let secret_polynomial = DCRTPoly::from_biguints(&parameters, &secret_coefficients);
            vector_values.push(DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                std::iter::once(secret_polynomial)
                    .chain(
                        (1..compiler.public_key_columns())
                            .map(|_| DCRTPoly::const_zero(&parameters)),
                    )
                    .collect(),
            ));

            let mut public_bottom_coefficients = Vec::with_capacity(compiler.coefficient_count);
            let mut slot_expected = Vec::with_capacity(compiler.coefficient_count);
            for coefficient in 0..compiler.coefficient_count {
                let target_plaintext = BigUint::from((slot + coefficient) % 5);
                let target = (&q * target_plaintext + &plaintext_modulus / BigUint::from(2u8)) /
                    &plaintext_modulus;
                let residual =
                    &decoder_coefficients[coefficient] - &secret_coefficients[coefficient];
                let public_bottom = (&target + &q - &residual) % &q;
                let noisy = (&decoder_coefficients[coefficient] + &q -
                    &secret_coefficients[coefficient] +
                    &public_bottom) %
                    &q;
                let decoded = (&plaintext_modulus * noisy + &q / BigUint::from(2u8)) / &q %
                    &plaintext_modulus;
                public_bottom_coefficients.push(public_bottom);
                slot_expected.push(BigInt::from(decoded));
            }
            public_bottom_values.push(DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_biguints(&parameters, &public_bottom_coefficients)],
            ));
            expected_values.push(slot_expected);
        }
        let repeated_family = |value: &DCRTPolyMatrix| {
            RuntimeValue::IndexedFamily(
                (0..slot_count).map(|_| RuntimeValue::matrix(value.clone())).collect(),
            )
        };
        let inputs = BTreeMap::from([
            ("decoder_state".to_owned(), RuntimeValue::matrix(decoder_state_value)),
            ("preimages".to_owned(), repeated_family(&preimage_value)),
            (
                "vectors".to_owned(),
                RuntimeValue::IndexedFamily(
                    vector_values.into_iter().map(RuntimeValue::matrix).collect(),
                ),
            ),
            (
                "public_bottom".to_owned(),
                RuntimeValue::IndexedFamily(
                    public_bottom_values.into_iter().map(RuntimeValue::matrix).collect(),
                ),
            ),
        ]);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh).unwrap();
        for slot in 0..slot_count {
            for coefficient in 0..compiler.coefficient_count {
                let expected = &expected_values[slot][coefficient];
                let RuntimeValue::Int(actual) =
                    &result.outputs[&format!("slot_{slot}_coefficient_{coefficient}")]
                else {
                    panic!("decoded output must be an integer")
                };
                assert_eq!(actual, expected);
            }
        }
    }
}
