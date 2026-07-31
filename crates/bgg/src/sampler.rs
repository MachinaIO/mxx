use crate::{
    BggEncodingWire, BggPolyEncodingWire, BggPublicKeyWire, NaiveBggEncodingVecWire,
    NaiveBggPublicKeyVecWire,
};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, RealExpr, SubgraphBuildError, WireRef,
    node::{
        ConcatAxis, ConstantMatrix, HashVariant, IndexRange, LoopInputMode, MatrixBinaryOp,
        SampleRange,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use thiserror::Error;

/// Explicit BGG+ matrix layout used by the Graph IR samplers.
///
/// `digit_count` is deliberately explicit: the sampler graph must not infer a
/// backend gadget layout from the modulus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggSamplerLayout {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_dimension: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
}

impl BggSamplerLayout {
    pub fn scalar_type(&self) -> MatrixType {
        self.matrix_type(1, 1)
    }

    pub fn secret_type(&self) -> MatrixType {
        self.matrix_type(1, self.secret_dimension)
    }

    pub fn public_key_type(&self) -> MatrixType {
        self.matrix_type(self.secret_dimension, self.public_key_columns())
    }

    pub fn vector_type(&self) -> MatrixType {
        self.matrix_type(1, self.public_key_columns())
    }

    pub fn slot_secret_type(&self) -> MatrixType {
        self.matrix_type(self.secret_dimension, self.secret_dimension)
    }

    pub fn public_key_columns(&self) -> usize {
        self.secret_dimension
            .checked_mul(self.digit_count)
            .expect("BGG+ public-key column count overflow")
    }

    fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn packed_public_key_type(&self, count: usize) -> MatrixType {
        self.matrix_type(
            self.secret_dimension,
            self.public_key_columns()
                .checked_mul(count)
                .expect("packed BGG+ public-key column count overflow"),
        )
    }

    fn packed_vector_type(&self, count: usize) -> MatrixType {
        self.matrix_type(
            1,
            self.public_key_columns()
                .checked_mul(count)
                .expect("packed BGG+ vector column count overflow"),
        )
    }

    fn plaintext_row_type(&self, count: usize) -> MatrixType {
        self.matrix_type(1, count)
    }
}

#[derive(Clone, Debug)]
pub struct BggPublicKeySampler {
    pub layout: BggSamplerLayout,
}

#[derive(Clone, Debug)]
pub struct BggEncodingSampler {
    pub layout: BggSamplerLayout,
    pub gaussian_sigma: Option<RealExpr>,
}

#[derive(Clone, Debug)]
pub struct BggPolyEncodingSampler {
    pub layout: BggSamplerLayout,
    pub gaussian_sigma: Option<RealExpr>,
}

#[derive(Clone, Debug)]
pub struct NaiveBggPublicKeyVecSampler {
    pub layout: BggSamplerLayout,
    pub slot_count: IntExpr,
}

#[derive(Clone, Debug)]
pub struct NaiveBggEncodingVecSampler {
    pub scalar: BggEncodingSampler,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggPolyEncodingSample {
    pub encodings: Vec<BggPolyEncodingWire>,
    pub slot_secret_matrices: MatrixFamilyWire,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BggSampleError {
    #[error("BGG+ sampling requires public_keys.len() == plaintexts.len() + 1")]
    InputCountMismatch,
    #[error("BGG+ sampler received a secret vector with the wrong matrix type")]
    SecretTypeMismatch,
    #[error("BGG+ sampler received a public key with the wrong matrix type")]
    PublicKeyTypeMismatch,
    #[error("BGG+ sampler received a plaintext with the wrong matrix type")]
    PlaintextTypeMismatch,
    #[error("BGG+ polynomial sampler families must have matching slot counts")]
    SlotCountMismatch,
    #[error("BGG+ polynomial sampler received a slot-secret family with the wrong matrix type")]
    SlotSecretTypeMismatch,
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
}

impl BggPublicKeySampler {
    /// Reproduces the legacy public-key sampler as one hash-sampled packed
    /// matrix followed by column slices. The first key always reveals the
    /// constant-one plaintext.
    pub fn sample(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Vec<BggPublicKeyWire> {
        let count = reveal_plaintexts.len() + 1;
        let all = builder.hash_sample(
            hash_key,
            self.layout.packed_public_key_type(count),
            HashVariant::Plain,
            tag.to_vec(),
            Vec::new(),
            None,
            None,
        );
        let columns = self.layout.public_key_columns();
        (0..count)
            .map(|index| BggPublicKeyWire {
                matrix: builder.slice(
                    &all,
                    None,
                    Some(IndexRange { start: columns * index, end: columns * (index + 1) }),
                    self.layout.public_key_type(),
                ),
                reveal_plaintext: index == 0 || reveal_plaintexts[index - 1],
            })
            .collect()
    }
}

impl BggEncodingSampler {
    /// Emits the exact BGG+ relation used by the former concrete sampler:
    ///
    /// `s * [A_0 | ... | A_t] - [1 | x_1 | ... | x_t] tensor (s * G) + e`.
    pub fn sample(
        &self,
        builder: &mut GraphBuilder,
        secret: &MatrixWire,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[MatrixWire],
    ) -> Result<Vec<BggEncodingWire>, BggSampleError> {
        validate_inputs(&self.layout, secret, public_keys, plaintexts)?;
        let count = public_keys.len();
        let all_public_keys = builder.concat(
            ConcatAxis::Columns,
            &public_keys.iter().map(|key| key.matrix.clone()).collect::<Vec<_>>(),
            self.layout.packed_public_key_type(count),
        );
        let one = builder.constant_matrix(self.layout.scalar_type(), ConstantMatrix::Identity);
        let mut extended_plaintexts = Vec::with_capacity(count);
        extended_plaintexts.push(one);
        extended_plaintexts.extend_from_slice(plaintexts);
        let encoded_plaintexts = builder.concat(
            ConcatAxis::Columns,
            &extended_plaintexts,
            self.layout.plaintext_row_type(count),
        );
        let all_vector = sample_packed_encoding(
            builder,
            &self.layout,
            self.gaussian_sigma.as_ref(),
            secret,
            &all_public_keys,
            &encoded_plaintexts,
            count,
        );
        Ok(slice_encodings(
            builder,
            &self.layout,
            &all_vector,
            public_keys,
            extended_plaintexts.into_iter(),
        ))
    }
}

impl BggPolyEncodingSampler {
    /// Samples all slot-local BGG+ vectors in one bounded parallel-loop graph.
    ///
    /// A supplied slot-secret family is consumed with `Zip`. Otherwise each
    /// loop instance samples one ternary secret matrix and returns the family
    /// so callers can persist or reuse it.
    pub fn sample(
        &self,
        builder: &mut GraphBuilder,
        secret: &MatrixWire,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[MatrixFamilyWire],
        slot_count: IntExpr,
        slot_secret_matrices: Option<&MatrixFamilyWire>,
    ) -> Result<BggPolyEncodingSample, BggSampleError> {
        validate_poly_inputs(
            &self.layout,
            secret,
            public_keys,
            plaintexts,
            &slot_count,
            slot_secret_matrices,
        )?;
        let count = public_keys.len();
        let all_public_keys = builder.concat(
            ConcatAxis::Columns,
            &public_keys.iter().map(|key| key.matrix.clone()).collect::<Vec<_>>(),
            self.layout.packed_public_key_type(count),
        );

        let mut body = GraphBuilder::new(
            format!(
                "bgg-poly-sample-{}-{}",
                count,
                if slot_secret_matrices.is_some() { "supplied-secret" } else { "fresh-secret" }
            ),
            Vec::new(),
        );
        let body_secret = body.input("00000000000000000000_secret", self.layout.secret_type());
        let body_public_keys = body
            .input("00000000000000000001_public_keys", self.layout.packed_public_key_type(count));
        let mut args = vec![secret.wire, all_public_keys.wire];
        let mut input_modes = vec![LoopInputMode::Broadcast, LoopInputMode::Broadcast];

        let body_slot_secret = if let Some(slot_secrets) = slot_secret_matrices {
            args.push(slot_secrets.wire);
            input_modes.push(LoopInputMode::Zip);
            body.input("00000000000000000002_slot_secret", self.layout.slot_secret_type())
        } else {
            body.uniform_sample(
                self.layout.slot_secret_type(),
                SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
            )
        };
        let transformed_secret = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &body_secret,
            &body_slot_secret,
            self.layout.secret_type(),
        );

        let one = body.constant_matrix(self.layout.scalar_type(), ConstantMatrix::Identity);
        let mut extended_plaintexts = Vec::with_capacity(count);
        extended_plaintexts.push(one.clone());
        for (index, plaintexts) in plaintexts.iter().enumerate() {
            args.push(plaintexts.wire);
            input_modes.push(LoopInputMode::Zip);
            extended_plaintexts.push(
                body.input(format!("{:020}_plaintext", index + 3), self.layout.scalar_type()),
            );
        }
        let encoded_plaintexts = body.concat(
            ConcatAxis::Columns,
            &extended_plaintexts,
            self.layout.plaintext_row_type(count),
        );
        let all_vector = sample_packed_encoding(
            &mut body,
            &self.layout,
            self.gaussian_sigma.as_ref(),
            &transformed_secret,
            &body_public_keys,
            &encoded_plaintexts,
            count,
        );

        let columns = self.layout.public_key_columns();
        let mut output_types = Vec::with_capacity(count + 2);
        for index in 0..count {
            let vector = body.slice(
                &all_vector,
                None,
                Some(IndexRange { start: columns * index, end: columns * (index + 1) }),
                self.layout.vector_type(),
            );
            body.value_output_wire(format!("{index:020}_vector"), vector.wire);
            output_types.push(self.layout.vector_type());
        }
        let fresh_secret_output = slot_secret_matrices.is_none();
        if fresh_secret_output {
            body.value_output_wire(format!("{count:020}_slot_secret"), body_slot_secret.wire);
            output_types.push(self.layout.slot_secret_type());
        }
        let constant_plaintext_output = public_keys[0].reveal_plaintext;
        if constant_plaintext_output {
            let output_index = count + usize::from(fresh_secret_output);
            body.value_output_wire(format!("{output_index:020}_constant_plaintext"), one.wire);
            output_types.push(self.layout.scalar_type());
        }

        let mut outputs = builder.parallel_loop(
            body.finish(),
            slot_count,
            "slot",
            Vec::new(),
            args,
            input_modes,
            &output_types,
        )?;
        let vector_families = outputs.drain(..count).collect::<Vec<_>>();
        let slot_secret_matrices = if fresh_secret_output {
            outputs.remove(0)
        } else {
            slot_secret_matrices.expect("validated supplied slot-secret family").clone()
        };
        let constant_plaintexts = constant_plaintext_output.then(|| outputs.remove(0));

        let encodings = vector_families
            .into_iter()
            .enumerate()
            .map(|(index, vectors)| BggPolyEncodingWire {
                vectors,
                pubkey: public_keys[index].clone(),
                plaintexts: if !public_keys[index].reveal_plaintext {
                    None
                } else if index == 0 {
                    constant_plaintexts.clone()
                } else {
                    Some(plaintexts[index - 1].clone())
                },
            })
            .collect();
        Ok(BggPolyEncodingSample { encodings, slot_secret_matrices })
    }
}

impl NaiveBggPublicKeyVecSampler {
    /// Reproduces the legacy per-output, per-slot hash namespace without
    /// unrolling the slot family into the parent graph.
    pub fn sample(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, BggSampleError> {
        let mut outputs = Vec::with_capacity(reveal_plaintexts.len() + 1);
        for output_index in 0..=reveal_plaintexts.len() {
            let reveal_plaintext = output_index == 0 || reveal_plaintexts[output_index - 1];
            let packed_count = if output_index == 0 { 1 } else { 2 };
            let mut body = GraphBuilder::new(
                format!("naive-bgg-public-key-sample-{output_index}"),
                Vec::new(),
            );
            let body_key = body.bytes_input("0_hash_key", 32);
            let mut tag_prefix = tag.to_vec();
            tag_prefix.extend_from_slice(&(output_index as u64).to_le_bytes());
            let all = body.hash_sample_with_encoded_tags(
                body_key,
                self.layout.packed_public_key_type(packed_count),
                HashVariant::Plain,
                tag_prefix,
                Vec::new(),
                Vec::new(),
                vec![IntExpr::Var("slot".to_owned())],
                None,
                None,
            );
            let matrix = if output_index == 0 {
                all
            } else {
                let columns = self.layout.public_key_columns();
                body.slice(
                    &all,
                    None,
                    Some(IndexRange { start: columns, end: columns * 2 }),
                    self.layout.public_key_type(),
                )
            };
            body.value_output_wire("0_matrix", matrix.wire);
            let mut families = builder.nonempty_parallel_loop(
                body.finish(),
                self.slot_count.clone(),
                "slot",
                Vec::new(),
                vec![hash_key],
                vec![LoopInputMode::Broadcast],
                &[self.layout.public_key_type()],
            )?;
            outputs
                .push(NaiveBggPublicKeyVecWire { matrices: families.remove(0), reveal_plaintext });
        }
        Ok(outputs)
    }
}

impl NaiveBggEncodingVecSampler {
    /// Samples each logical output in its own slot loop, preserving the
    /// legacy sampler's independent packed Gaussian draw per output and slot.
    pub fn sample(
        &self,
        builder: &mut GraphBuilder,
        secret: &MatrixWire,
        public_keys: &[NaiveBggPublicKeyVecWire],
        plaintexts: &[MatrixFamilyWire],
    ) -> Result<Vec<NaiveBggEncodingVecWire>, BggSampleError> {
        if public_keys.len() != plaintexts.len() + 1 {
            return Err(BggSampleError::InputCountMismatch);
        }
        if secret.matrix_type != self.scalar.layout.secret_type() {
            return Err(BggSampleError::SecretTypeMismatch);
        }
        let slot_count = public_keys[0].matrices.count.clone();
        for public_key in public_keys {
            if public_key.matrices.count != slot_count {
                return Err(BggSampleError::SlotCountMismatch);
            }
            if public_key.matrices.matrix_type != self.scalar.layout.public_key_type() {
                return Err(BggSampleError::PublicKeyTypeMismatch);
            }
        }
        for plaintext in plaintexts {
            if plaintext.count != slot_count {
                return Err(BggSampleError::SlotCountMismatch);
            }
            if plaintext.matrix_type != self.scalar.layout.scalar_type() {
                return Err(BggSampleError::PlaintextTypeMismatch);
            }
        }

        let mut outputs = Vec::with_capacity(public_keys.len());
        for output_index in 0..public_keys.len() {
            let mut body = GraphBuilder::new(
                format!(
                    "naive-bgg-encoding-sample-{output_index}-{}",
                    if public_keys[output_index].reveal_plaintext { "revealed" } else { "hidden" }
                ),
                Vec::new(),
            );
            let body_secret = body.input("0_secret", self.scalar.layout.secret_type());
            let one_public_key = BggPublicKeyWire {
                matrix: body.input("1_one_public_key", self.scalar.layout.public_key_type()),
                reveal_plaintext: public_keys[0].reveal_plaintext,
            };
            let mut args = vec![secret.wire, public_keys[0].matrices.wire];
            let mut modes = vec![LoopInputMode::Broadcast, LoopInputMode::Zip];
            let selected = if output_index == 0 {
                self.scalar.sample(&mut body, &body_secret, &[one_public_key], &[])?.remove(0)
            } else {
                let public_key = BggPublicKeyWire {
                    matrix: body.input("2_public_key", self.scalar.layout.public_key_type()),
                    reveal_plaintext: public_keys[output_index].reveal_plaintext,
                };
                let plaintext = body.input("3_plaintext", self.scalar.layout.scalar_type());
                args.extend([
                    public_keys[output_index].matrices.wire,
                    plaintexts[output_index - 1].wire,
                ]);
                modes.extend([LoopInputMode::Zip, LoopInputMode::Zip]);
                self.scalar
                    .sample(&mut body, &body_secret, &[one_public_key, public_key], &[plaintext])?
                    .remove(1)
            };
            body.value_output_wire("0_vector", selected.vector.wire);
            let mut output_types = vec![self.scalar.layout.vector_type()];
            let constant_plaintext_output = output_index == 0 && selected.plaintext.is_some();
            if constant_plaintext_output {
                body.value_output_wire(
                    "1_constant_plaintext",
                    selected.plaintext.as_ref().expect("checked constant plaintext").wire,
                );
                output_types.push(self.scalar.layout.scalar_type());
            }
            let mut families = builder.nonempty_parallel_loop(
                body.finish(),
                slot_count.clone(),
                "slot",
                Vec::new(),
                args,
                modes,
                &output_types,
            )?;
            let vectors = families.remove(0);
            let plaintexts = if constant_plaintext_output {
                Some(families.remove(0))
            } else if output_index > 0 && public_keys[output_index].reveal_plaintext {
                Some(plaintexts[output_index - 1].clone())
            } else {
                None
            };
            outputs.push(NaiveBggEncodingVecWire {
                vectors,
                pubkeys: public_keys[output_index].matrices.clone(),
                pubkey_reveal_plaintext: public_keys[output_index].reveal_plaintext,
                plaintexts,
            });
        }
        Ok(outputs)
    }
}

fn sample_packed_encoding(
    builder: &mut GraphBuilder,
    layout: &BggSamplerLayout,
    gaussian_sigma: Option<&RealExpr>,
    secret: &MatrixWire,
    all_public_keys: &MatrixWire,
    encoded_plaintexts: &MatrixWire,
    count: usize,
) -> MatrixWire {
    let packed_type = layout.packed_vector_type(count);
    let first = builder.matrix_binary(
        MatrixBinaryOp::Multiply,
        secret,
        all_public_keys,
        packed_type.clone(),
    );
    let gadget = builder.constant_matrix(
        layout.public_key_type(),
        ConstantMatrix::Gadget { base: layout.gadget_base.clone(), small: false },
    );
    let secret_gadget =
        builder.matrix_binary(MatrixBinaryOp::Multiply, secret, &gadget, layout.vector_type());
    let second = builder.tensor(encoded_plaintexts, &secret_gadget, packed_type.clone());
    let difference =
        builder.matrix_binary(MatrixBinaryOp::Subtract, &first, &second, packed_type.clone());
    let error = match gaussian_sigma {
        Some(sigma) => builder.gaussian_sample(packed_type.clone(), sigma.clone()),
        None => builder.constant_matrix(packed_type.clone(), ConstantMatrix::Zero),
    };
    builder.matrix_binary(MatrixBinaryOp::Add, &difference, &error, packed_type)
}

fn slice_encodings(
    builder: &mut GraphBuilder,
    layout: &BggSamplerLayout,
    all_vector: &MatrixWire,
    public_keys: &[BggPublicKeyWire],
    plaintexts: impl Iterator<Item = MatrixWire>,
) -> Vec<BggEncodingWire> {
    let columns = layout.public_key_columns();
    public_keys
        .iter()
        .zip(plaintexts)
        .enumerate()
        .map(|(index, (pubkey, plaintext))| BggEncodingWire {
            vector: builder.slice(
                all_vector,
                None,
                Some(IndexRange { start: columns * index, end: columns * (index + 1) }),
                layout.vector_type(),
            ),
            pubkey: pubkey.clone(),
            plaintext: pubkey.reveal_plaintext.then_some(plaintext),
        })
        .collect()
}

fn validate_inputs(
    layout: &BggSamplerLayout,
    secret: &MatrixWire,
    public_keys: &[BggPublicKeyWire],
    plaintexts: &[MatrixWire],
) -> Result<(), BggSampleError> {
    if public_keys.len() != plaintexts.len() + 1 {
        return Err(BggSampleError::InputCountMismatch);
    }
    if secret.matrix_type != layout.secret_type() {
        return Err(BggSampleError::SecretTypeMismatch);
    }
    if public_keys.iter().any(|key| key.matrix.matrix_type != layout.public_key_type()) {
        return Err(BggSampleError::PublicKeyTypeMismatch);
    }
    if plaintexts.iter().any(|plaintext| plaintext.matrix_type != layout.scalar_type()) {
        return Err(BggSampleError::PlaintextTypeMismatch);
    }
    Ok(())
}

fn validate_poly_inputs(
    layout: &BggSamplerLayout,
    secret: &MatrixWire,
    public_keys: &[BggPublicKeyWire],
    plaintexts: &[MatrixFamilyWire],
    slot_count: &IntExpr,
    slot_secret_matrices: Option<&MatrixFamilyWire>,
) -> Result<(), BggSampleError> {
    if public_keys.len() != plaintexts.len() + 1 {
        return Err(BggSampleError::InputCountMismatch);
    }
    if secret.matrix_type != layout.secret_type() {
        return Err(BggSampleError::SecretTypeMismatch);
    }
    if public_keys.iter().any(|key| key.matrix.matrix_type != layout.public_key_type()) {
        return Err(BggSampleError::PublicKeyTypeMismatch);
    }
    if plaintexts.iter().any(|plaintext| plaintext.matrix_type != layout.scalar_type()) {
        return Err(BggSampleError::PlaintextTypeMismatch);
    }
    if plaintexts.iter().any(|plaintext| &plaintext.count != slot_count) {
        return Err(BggSampleError::SlotCountMismatch);
    }
    if let Some(slot_secrets) = slot_secret_matrices {
        if slot_secrets.count != *slot_count {
            return Err(BggSampleError::SlotCountMismatch);
        }
        if slot_secrets.matrix_type != layout.slot_secret_type() {
            return Err(BggSampleError::SlotSecretTypeMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use keccak_asm::Keccak256;
    use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality, validate};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::cpu_backend,
        execute,
        transcript::{SamplingMode, TranscriptRecorder},
    };
    use num_bigint::{BigInt, Sign};
    use std::collections::BTreeMap;

    fn layout(parameters: &DCRTPolyParams, secret_dimension: usize) -> BggSamplerLayout {
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus().into();
        BggSamplerLayout {
            modulus: IntExpr::constant(BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_dimension,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        }
    }

    fn scalar(parameters: &DCRTPolyParams, rotation: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            vec![vec![DCRTPoly::const_rotate_poly(parameters, rotation)]],
        )
    }

    fn secret(parameters: &DCRTPolyParams, dimension: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(
            parameters,
            (0..dimension)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        parameters,
                        index % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn public_key_and_encoding_graphs_match_the_legacy_matrix_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = layout(&parameters, 2);
        let key = [23u8; 32];
        let tag = b"bgg-ir-sampler";
        let mut builder = GraphBuilder::new("bgg-sampler-relation", Vec::new());
        let key_wire = builder.bytes_input("key", 32);
        let secret_wire = builder.input("secret", layout.secret_type());
        let plaintext_wires = [
            builder.input("plaintext_0", layout.scalar_type()),
            builder.input("plaintext_1", layout.scalar_type()),
        ];
        let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
            &mut builder,
            key_wire,
            tag,
            &[false, true],
        );
        let encodings = BggEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(&mut builder, &secret_wire, &public_keys, &plaintext_wires)
            .expect("compatible sampler inputs");
        for (index, public_key) in public_keys.iter().enumerate() {
            builder.output(
                format!("public_key_{index}"),
                &public_key.matrix,
                ArtifactConfidentiality::Public,
            );
            builder.output(
                format!("vector_{index}"),
                &encodings[index].vector,
                ArtifactConfidentiality::Public,
            );
        }
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("valid BGG+ sampler graph");

        let secret_value = secret(&parameters, layout.secret_dimension);
        let plaintext_values = [scalar(&parameters, 2), scalar(&parameters, 3)];
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &validated,
            &mut backend,
            BTreeMap::from([
                ("key".to_owned(), RuntimeValue::Bytes(key.to_vec())),
                ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
                ("plaintext_0".to_owned(), RuntimeValue::matrix(plaintext_values[0].clone())),
                ("plaintext_1".to_owned(), RuntimeValue::matrix(plaintext_values[1].clone())),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("BGG+ sampler execution");

        let packed = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &parameters,
            key,
            tag,
            layout.secret_dimension,
            layout.public_key_columns() * public_keys.len(),
            DistType::FinRingDist,
        );
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        let encoded_plaintexts = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![
                DCRTPoly::const_one(&parameters),
                plaintext_values[0].entry(0, 0),
                plaintext_values[1].entry(0, 0),
            ],
        );
        let all_vectors = secret_value.clone() * packed.clone() -
            encoded_plaintexts.tensor(&(secret_value.clone() * gadget));
        for index in 0..public_keys.len() {
            let expected_public_key = packed.slice_columns(
                layout.public_key_columns() * index,
                layout.public_key_columns() * (index + 1),
            );
            let RuntimeValue::Matrix(actual_public_key) =
                &result.outputs[&format!("public_key_{index}")]
            else {
                panic!("public-key output must be a matrix");
            };
            assert_eq!(actual_public_key.as_ref(), &expected_public_key);

            let expected_vector = all_vectors.slice_columns(
                layout.public_key_columns() * index,
                layout.public_key_columns() * (index + 1),
            );
            let RuntimeValue::Matrix(actual_vector) = &result.outputs[&format!("vector_{index}")]
            else {
                panic!("encoding output must be a matrix");
            };
            assert_eq!(actual_vector.as_ref(), &expected_vector);
        }
        assert!(encodings[0].plaintext.is_some());
        assert!(encodings[1].plaintext.is_none());
        assert!(encodings[2].plaintext.is_some());
    }

    #[test]
    fn polynomial_sampler_uses_one_fresh_ternary_secret_per_slot() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = layout(&parameters, 2);
        let slot_count = 3usize;
        let mut builder = GraphBuilder::new("bgg-poly-sampler-relation", Vec::new());
        let secret_wire = builder.input("secret", layout.secret_type());
        let public_key_wires = [
            BggPublicKeyWire {
                matrix: builder.input("public_key_0", layout.public_key_type()),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: builder.input("public_key_1", layout.public_key_type()),
                reveal_plaintext: true,
            },
        ];
        let plaintext_slots = (0..slot_count)
            .map(|slot| builder.input(format!("plaintext_{slot}"), layout.scalar_type()))
            .collect::<Vec<_>>();
        let plaintext_family =
            builder.family_pack(&plaintext_slots).expect("homogeneous plaintext family");
        let sample = BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                &mut builder,
                &secret_wire,
                &public_key_wires,
                std::slice::from_ref(&plaintext_family),
                IntExpr::constant(slot_count),
                None,
            )
            .expect("compatible polynomial sampler inputs");
        for slot in 0..slot_count {
            let vector =
                builder.family_get_static(&sample.encodings[1].vectors, IntExpr::constant(slot));
            let slot_secret =
                builder.family_get_static(&sample.slot_secret_matrices, IntExpr::constant(slot));
            builder.output(format!("vector_{slot}"), &vector, ArtifactConfidentiality::Public);
            builder.output(
                format!("slot_secret_{slot}"),
                &slot_secret,
                ArtifactConfidentiality::Private,
            );
        }
        let validated = validate(&builder.finish(), &ParamEnv::default())
            .expect("valid polynomial BGG+ sampler graph");

        let secret_value = secret(&parameters, layout.secret_dimension);
        let public_key_values = [
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &parameters,
                [31u8; 32],
                b"poly-public-key-0",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &parameters,
                [31u8; 32],
                b"poly-public-key-1",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
        ];
        let plaintext_values =
            (0..slot_count).map(|slot| scalar(&parameters, slot + 1)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::from([
            ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
            ("public_key_0".to_owned(), RuntimeValue::matrix(public_key_values[0].clone())),
            ("public_key_1".to_owned(), RuntimeValue::matrix(public_key_values[1].clone())),
        ]);
        for (slot, plaintext) in plaintext_values.iter().enumerate() {
            inputs.insert(format!("plaintext_{slot}"), RuntimeValue::matrix(plaintext.clone()));
        }
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("polynomial BGG+ sampler execution");
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        for slot in 0..slot_count {
            let RuntimeValue::Matrix(slot_secret) = &result.outputs[&format!("slot_secret_{slot}")]
            else {
                panic!("slot secret must be a matrix");
            };
            let RuntimeValue::Matrix(vector) = &result.outputs[&format!("vector_{slot}")] else {
                panic!("slot vector must be a matrix");
            };
            let transformed_secret = secret_value.clone() * slot_secret.as_ref();
            let expected = transformed_secret.clone() * public_key_values[1].clone() -
                plaintext_values[slot].clone().tensor(&(transformed_secret * gadget.clone()));
            assert_eq!(vector.as_ref(), &expected);
        }
    }

    fn supplied_secret_graph(
        layout: &BggSamplerLayout,
        sigma: Option<RealExpr>,
    ) -> mxx_ir_core::ValidatedGraph {
        let slot_count = 2usize;
        let mut builder = GraphBuilder::new("bgg-poly-supplied-secret", Vec::new());
        let secret_wire = builder.input("secret", layout.secret_type());
        let public_keys = [
            BggPublicKeyWire {
                matrix: builder.input("public_key_0", layout.public_key_type()),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: builder.input("public_key_1", layout.public_key_type()),
                reveal_plaintext: false,
            },
        ];
        let plaintexts = (0..slot_count)
            .map(|slot| builder.input(format!("plaintext_{slot}"), layout.scalar_type()))
            .collect::<Vec<_>>();
        let plaintexts = builder.family_pack(&plaintexts).expect("plaintext family");
        let slot_secrets = (0..slot_count)
            .map(|slot| builder.input(format!("slot_secret_{slot}"), layout.slot_secret_type()))
            .collect::<Vec<_>>();
        let slot_secrets = builder.family_pack(&slot_secrets).expect("slot-secret family");
        let sample = BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: sigma }
            .sample(
                &mut builder,
                &secret_wire,
                &public_keys,
                &[plaintexts],
                IntExpr::constant(slot_count),
                Some(&slot_secrets),
            )
            .expect("compatible supplied-secret graph");
        assert!(
            sample.encodings[1].plaintexts.is_none(),
            "hidden public-key metadata must suppress plaintext metadata"
        );
        assert_eq!(sample.slot_secret_matrices, slot_secrets);
        for slot in 0..slot_count {
            let vector =
                builder.family_get_static(&sample.encodings[1].vectors, IntExpr::constant(slot));
            builder.output(format!("vector_{slot}"), &vector, ArtifactConfidentiality::Public);
        }
        validate(&builder.finish(), &ParamEnv::default()).expect("valid supplied-secret graph")
    }

    #[test]
    fn supplied_slot_secret_preserves_zip_order_and_zero_sigma_matches_no_error() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = layout(&parameters, 2);
        let no_error = supplied_secret_graph(&layout, None);
        let zero_error = supplied_secret_graph(
            &layout,
            Some(RealExpr::from_f64_exact(0.0).expect("finite zero")),
        );
        let secret_value = secret(&parameters, layout.secret_dimension);
        let slot_secret_values = [
            DCRTPolyMatrix::identity(&parameters, layout.secret_dimension, None),
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    vec![
                        DCRTPoly::const_rotate_poly(&parameters, 1),
                        DCRTPoly::const_rotate_poly(&parameters, 2),
                    ],
                    vec![
                        DCRTPoly::const_rotate_poly(&parameters, 3),
                        DCRTPoly::const_rotate_poly(&parameters, 4),
                    ],
                ],
            ),
        ];
        let public_key_values = [
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &parameters,
                [41u8; 32],
                b"supplied-public-key-0",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &parameters,
                [41u8; 32],
                b"supplied-public-key-1",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
        ];
        let plaintexts = [scalar(&parameters, 3), scalar(&parameters, 5)];
        let inputs = BTreeMap::from([
            ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
            ("public_key_0".to_owned(), RuntimeValue::matrix(public_key_values[0].clone())),
            ("public_key_1".to_owned(), RuntimeValue::matrix(public_key_values[1].clone())),
            ("plaintext_0".to_owned(), RuntimeValue::matrix(plaintexts[0].clone())),
            ("plaintext_1".to_owned(), RuntimeValue::matrix(plaintexts[1].clone())),
            ("slot_secret_0".to_owned(), RuntimeValue::matrix(slot_secret_values[0].clone())),
            ("slot_secret_1".to_owned(), RuntimeValue::matrix(slot_secret_values[1].clone())),
        ]);
        let execute_graph = |graph: &mxx_ir_core::ValidatedGraph| {
            let mut backend = cpu_backend([parameters.clone()]);
            let mut store = MemoryArtifactStore::default();
            execute(graph, &mut backend, inputs.clone(), &mut store, SamplingMode::Fresh)
                .expect("supplied-secret execution")
        };
        let none_result = execute_graph(&no_error);
        let zero_result = execute_graph(&zero_error);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        for slot in 0..2 {
            let RuntimeValue::Matrix(none_vector) = &none_result.outputs[&format!("vector_{slot}")]
            else {
                panic!("none vector output");
            };
            let RuntimeValue::Matrix(zero_vector) = &zero_result.outputs[&format!("vector_{slot}")]
            else {
                panic!("zero vector output");
            };
            assert_eq!(none_vector, zero_vector);
            let transformed_secret = secret_value.clone() * slot_secret_values[slot].clone();
            let expected = transformed_secret.clone() * public_key_values[1].clone() -
                plaintexts[slot].clone().tensor(&(transformed_secret * gadget.clone()));
            assert_eq!(none_vector.as_ref(), &expected);
        }
    }

    #[test]
    fn naive_vector_samplers_preserve_nonempty_slots_tags_and_encoding_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = layout(&parameters, 2);
        let key = [47u8; 32];
        let tag = b"naive-bgg-ir";
        let slot_count = 2usize;
        let mut builder = GraphBuilder::new("naive-bgg-samplers", Vec::new());
        let key_wire = builder.bytes_input("key", key.len());
        let secret_wire = builder.input("secret", layout.secret_type());
        let public_keys = NaiveBggPublicKeyVecSampler {
            layout: layout.clone(),
            slot_count: IntExpr::constant(slot_count),
        }
        .sample(&mut builder, key_wire, tag, &[true])
        .expect("nonempty public-key families");
        let plaintext_wires = (0..slot_count)
            .map(|slot| builder.input(format!("plaintext_{slot}"), layout.scalar_type()))
            .collect::<Vec<_>>();
        let plaintexts = builder.family_pack(&plaintext_wires).expect("plaintext family");
        let encodings = NaiveBggEncodingVecSampler {
            scalar: BggEncodingSampler { layout: layout.clone(), gaussian_sigma: None },
        }
        .sample(&mut builder, &secret_wire, &public_keys, std::slice::from_ref(&plaintexts))
        .expect("compatible naive sampler inputs");
        for output in 0..public_keys.len() {
            for slot in 0..slot_count {
                let public_key = builder
                    .family_get_static(&public_keys[output].matrices, IntExpr::constant(slot));
                let vector =
                    builder.family_get_static(&encodings[output].vectors, IntExpr::constant(slot));
                builder.output(
                    format!("public_key_{output}_{slot}"),
                    &public_key,
                    ArtifactConfidentiality::Public,
                );
                builder.output(
                    format!("vector_{output}_{slot}"),
                    &vector,
                    ArtifactConfidentiality::Public,
                );
            }
        }
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("valid naive sampler graph");

        let secret_value = secret(&parameters, layout.secret_dimension);
        let plaintext_values = [scalar(&parameters, 2), scalar(&parameters, 5)];
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &validated,
            &mut backend,
            BTreeMap::from([
                ("key".to_owned(), RuntimeValue::Bytes(key.to_vec())),
                ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
                ("plaintext_0".to_owned(), RuntimeValue::matrix(plaintext_values[0].clone())),
                ("plaintext_1".to_owned(), RuntimeValue::matrix(plaintext_values[1].clone())),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("naive sampler execution");

        let hash_sampler = DCRTPolyHashSampler::<Keccak256>::new();
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        for output in 0..public_keys.len() {
            for slot in 0..slot_count {
                let mut slot_tag = tag.to_vec();
                slot_tag.extend_from_slice(&(output as u64).to_le_bytes());
                slot_tag.extend_from_slice(&(slot as u64).to_le_bytes());
                let packed_count = if output == 0 { 1 } else { 2 };
                let packed = hash_sampler.sample_hash(
                    &parameters,
                    key,
                    &slot_tag,
                    layout.secret_dimension,
                    layout.public_key_columns() * packed_count,
                    DistType::FinRingDist,
                );
                let expected_public_key = if output == 0 {
                    packed
                } else {
                    packed
                        .slice_columns(layout.public_key_columns(), layout.public_key_columns() * 2)
                };
                let RuntimeValue::Matrix(actual_public_key) =
                    &result.outputs[&format!("public_key_{output}_{slot}")]
                else {
                    panic!("naive public-key output")
                };
                assert_eq!(actual_public_key.as_ref(), &expected_public_key);

                let plaintext = if output == 0 {
                    DCRTPolyMatrix::identity(&parameters, 1, None)
                } else {
                    plaintext_values[slot].clone()
                };
                let expected_vector = secret_value.clone() * expected_public_key -
                    plaintext.tensor(&(secret_value.clone() * gadget.clone()));
                let RuntimeValue::Matrix(actual_vector) =
                    &result.outputs[&format!("vector_{output}_{slot}")]
                else {
                    panic!("naive encoding output")
                };
                assert_eq!(actual_vector.as_ref(), &expected_vector);
            }
        }
        assert!(public_keys.iter().all(|public_key| public_key.reveal_plaintext));
        assert!(encodings.iter().all(|encoding| encoding.plaintexts.is_some()));

        let mut parameterized = GraphBuilder::new(
            "naive-bgg-nonempty",
            vec![mxx_ir_core::graph::CompileParameter {
                name: "slots".to_owned(),
                kind: mxx_ir_core::graph::CompileParameterKind::Integer,
            }],
        );
        let key_wire = parameterized.bytes_input("key", key.len());
        NaiveBggPublicKeyVecSampler {
            layout: layout.clone(),
            slot_count: IntExpr::Var("slots".to_owned()),
        }
        .sample(&mut parameterized, key_wire, tag, &[])
        .expect("parameterized sampler graph");
        let graph = parameterized.finish();
        let zero_env = ParamEnv {
            integers: BTreeMap::from([("slots".to_owned(), BigInt::from(0))]),
            ..Default::default()
        };
        assert!(
            validate(&graph, &zero_env)
                .expect_err("zero-slot naive sampler must be rejected")
                .to_string()
                .contains("loop count must be at least 1")
        );
        let two_env = ParamEnv {
            integers: BTreeMap::from([("slots".to_owned(), BigInt::from(2))]),
            ..Default::default()
        };
        validate(&graph, &two_env).expect("positive parameterized slot count");

        let mut external = GraphBuilder::new(
            "naive-bgg-external-nonempty",
            vec![mxx_ir_core::graph::CompileParameter {
                name: "slots".to_owned(),
                kind: mxx_ir_core::graph::CompileParameterKind::Integer,
            }],
        );
        let count = IntExpr::Var("slots".to_owned());
        let secret = external.input("secret", layout.secret_type());
        let public_key = NaiveBggPublicKeyVecWire {
            matrices: external.family_input("public_keys", layout.public_key_type(), count.clone()),
            reveal_plaintext: true,
        };
        NaiveBggEncodingVecSampler { scalar: BggEncodingSampler { layout, gaussian_sigma: None } }
            .sample(&mut external, &secret, &[public_key], &[])
            .expect("external parameterized families");
        assert!(
            validate(&external.finish(), &zero_env)
                .expect_err("zero-slot external encoding family must be rejected")
                .to_string()
                .contains("loop count must be at least 1")
        );
    }

    #[test]
    fn fresh_slot_secret_draws_replay_at_the_same_loop_sites() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = layout(&parameters, 2);
        let slot_count = 2usize;
        let mut builder = GraphBuilder::new("bgg-poly-replay", Vec::new());
        let secret_wire = builder.input("secret", layout.secret_type());
        let public_keys = [BggPublicKeyWire {
            matrix: builder.input("public_key", layout.public_key_type()),
            reveal_plaintext: true,
        }];
        let sample = BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                &mut builder,
                &secret_wire,
                &public_keys,
                &[],
                IntExpr::constant(slot_count),
                None,
            )
            .expect("fresh-secret graph");
        for slot in 0..slot_count {
            let vector =
                builder.family_get_static(&sample.encodings[0].vectors, IntExpr::constant(slot));
            let slot_secret =
                builder.family_get_static(&sample.slot_secret_matrices, IntExpr::constant(slot));
            builder.output(format!("vector_{slot}"), &vector, ArtifactConfidentiality::Public);
            builder.output(
                format!("slot_secret_{slot}"),
                &slot_secret,
                ArtifactConfidentiality::Private,
            );
        }
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("valid replay graph");
        let inputs = BTreeMap::from([
            (
                "secret".to_owned(),
                RuntimeValue::matrix(secret(&parameters, layout.secret_dimension)),
            ),
            (
                "public_key".to_owned(),
                RuntimeValue::matrix(DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                    &parameters,
                    [43u8; 32],
                    b"replay-public-key",
                    layout.secret_dimension,
                    layout.public_key_columns(),
                    DistType::FinRingDist,
                )),
            ),
        ]);
        let mut recorder = TranscriptRecorder::default();
        let mut record_backend = cpu_backend([parameters.clone()]);
        let mut record_store = MemoryArtifactStore::default();
        let recorded = execute(
            &validated,
            &mut record_backend,
            inputs.clone(),
            &mut record_store,
            SamplingMode::Record(&mut recorder),
        )
        .expect("record execution");
        assert_eq!(recorder.iter().count(), slot_count);
        let replayer = recorder.into_replayer();
        let mut replay_backend = cpu_backend([parameters]);
        let mut replay_store = MemoryArtifactStore::default();
        let replayed = execute(
            &validated,
            &mut replay_backend,
            inputs,
            &mut replay_store,
            SamplingMode::Replay(&replayer),
        )
        .expect("replay execution");
        for name in recorded.outputs.keys() {
            let RuntimeValue::Matrix(recorded_value) = &recorded.outputs[name] else {
                panic!("{name} recorded output");
            };
            let RuntimeValue::Matrix(replayed_value) = &replayed.outputs[name] else {
                panic!("{name} replayed output");
            };
            assert_eq!(recorded_value, replayed_value);
        }
    }

    #[test]
    fn runtime_rejects_a_sampler_gadget_layout_that_the_backend_cannot_honor() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut layout = layout(&parameters, 2);
        layout.gadget_base = IntExpr::constant(1u64 << (parameters.base_bits() + 1));
        let mut builder = GraphBuilder::new("bgg-invalid-gadget-layout", Vec::new());
        let secret_wire = builder.input("secret", layout.secret_type());
        let public_key = BggPublicKeyWire {
            matrix: builder.input("public_key", layout.public_key_type()),
            reveal_plaintext: true,
        };
        let encodings = BggEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(&mut builder, &secret_wire, &[public_key], &[])
            .expect("statically shaped graph");
        builder.output("vector", &encodings[0].vector, ArtifactConfidentiality::Public);
        let validated = validate(&builder.finish(), &ParamEnv::default())
            .expect("backend layout is checked at execution");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let error = match execute(
            &validated,
            &mut backend,
            BTreeMap::from([
                (
                    "secret".to_owned(),
                    RuntimeValue::matrix(secret(&parameters, layout.secret_dimension)),
                ),
                (
                    "public_key".to_owned(),
                    RuntimeValue::matrix(DCRTPolyMatrix::zero(
                        &parameters,
                        layout.secret_dimension,
                        layout.public_key_columns(),
                    )),
                ),
            ]),
            &mut store,
            SamplingMode::Fresh,
        ) {
            Ok(_) => panic!("mismatched backend gadget layout must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not match backend base"));
    }
}
