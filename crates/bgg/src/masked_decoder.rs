//! Masked high-bit decoding expressed with declarative matrix families.

use mxx_dsl::{Bool, DslContext, DslError, Family, Int, Mat, Ring, Trapdoor};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, IndexRange},
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

#[derive(Clone)]
pub struct MaskedHighBitDecoderPreprocessingWires {
    pub preimages: Family<Mat>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaskedHighBitDecoderArtifacts {
    pub production_id: ProductionId,
    pub slot_count: usize,
}

pub enum MaskedHighBitDecoderOutputs {
    Integers(Vec<Family<Int>>),
    Booleans(Vec<Family<Bool>>),
}

#[derive(Debug, Error)]
pub enum MaskedHighBitDecoderError {
    #[error("masked decoder dimensions and coefficient count must be nonzero")]
    EmptyLayout,
    #[error("masked decoder input families must have the configured slot count")]
    FamilyCountMismatch,
    #[error(transparent)]
    Dsl(#[from] DslError),
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
        decoder_trapdoor: Trapdoor,
        public_keys: Family<Mat>,
        slot_count: usize,
    ) -> Result<MaskedHighBitDecoderPreprocessingWires, MaskedHighBitDecoderError> {
        self.validate_layout()?;
        if public_keys.count() != &IntExpr::constant(slot_count) ||
            public_keys.element_type() != &self.public_key_type()
        {
            return Err(MaskedHighBitDecoderError::FamilyCountMismatch);
        }
        let ring = Ring::new(self.modulus.clone(), self.ring_dimension.clone());
        let (secret_size, digit_count, decoder_columns) =
            (self.secret_size, self.digit_count, self.decoder_columns());
        let gadget_base = self.gadget_base.clone();
        let preimages = public_keys.parallel_map(move |_, public_key| {
            let selector = ring
                .identity(secret_size)
                .slice(None, Some(IndexRange { start: 0.into(), end: 1.into() }));
            let top = public_key * selector.decompose(gadget_base.clone(), digit_count).as_mat();
            let target = Mat::concat(ConcatAxis::Rows, vec![top, ring.zero((secret_size, 1))]);
            decoder_trapdoor.sample_preimage(target, (decoder_columns, 1)).as_mat()
        })?;
        Ok(MaskedHighBitDecoderPreprocessingWires { preimages })
    }

    pub fn export_preprocessing(
        &self,
        context: DslContext,
        wires: MaskedHighBitDecoderPreprocessingWires,
    ) -> Result<DslContext, MaskedHighBitDecoderError> {
        Ok(context.public_family_output(MASKED_DECODER_PREIMAGES, wires.preimages)?)
    }

    pub fn import_preimages(
        &self,
        artifacts: &MaskedHighBitDecoderArtifacts,
    ) -> Result<Family<Mat>, MaskedHighBitDecoderError> {
        self.validate_layout()?;
        Ok(Ring::new(self.modulus.clone(), self.ring_dimension.clone()).family_artifact_input(
            artifacts.production_id.clone(),
            MASKED_DECODER_PREIMAGES,
            artifacts.slot_count,
            (self.decoder_columns(), 1),
            ArtifactConfidentiality::Public,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_online(
        &self,
        decoder_state: Mat,
        preimages: Family<Mat>,
        vectors: Family<Mat>,
        bottoms: Family<Mat>,
        plaintext_modulus: IntExpr,
        output_bool: bool,
        slot_count: usize,
    ) -> Result<MaskedHighBitDecoderOutputs, MaskedHighBitDecoderError> {
        self.validate_layout()?;
        let expected = IntExpr::constant(slot_count);
        if preimages.count() != &expected ||
            vectors.count() != &expected ||
            bottoms.count() != &expected
        {
            return Err(MaskedHighBitDecoderError::FamilyCountMismatch);
        }
        let ring = Ring::new(self.modulus.clone(), self.ring_dimension.clone());
        let (secret_size, digit_count) = (self.secret_size, self.digit_count);
        let gadget_base = self.gadget_base.clone();
        let noisy = preimages.parallel_zip3_values(
            vectors,
            bottoms,
            move |_, preimage, vector, bottom| {
                let selector = ring
                    .identity(secret_size)
                    .slice(None, Some(IndexRange { start: 0.into(), end: 1.into() }));
                decoder_state.clone() * preimage -
                    vector * selector.decompose(gadget_base.clone(), digit_count).as_mat() +
                    bottom
            },
        )?;
        if output_bool {
            Ok(MaskedHighBitDecoderOutputs::Booleans(
                noisy.parallel_threshold_decode_bools(plaintext_modulus, self.coefficient_count)?,
            ))
        } else {
            Ok(MaskedHighBitDecoderOutputs::Integers(
                noisy.parallel_threshold_decode_ints(plaintext_modulus, self.coefficient_count)?,
            ))
        }
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
            rows: rows.into(),
            columns: columns.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::execute_graph;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{RuntimeValue, backend::poly::CpuDcrtBackend};
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    fn compiler() -> MaskedHighBitDecoderCompiler {
        MaskedHighBitDecoderCompiler {
            modulus: 65_537.into(),
            ring_dimension: 8.into(),
            secret_size: 2,
            digit_count: 4,
            gadget_base: 4.into(),
            trapdoor_sigma: RealExpr::from_integer(5),
            coefficient_count: 4,
        }
    }

    #[test]
    fn preprocessing_and_online_graphs_validate() {
        let compiler = compiler();
        let ring = Ring::new(compiler.modulus.clone(), compiler.ring_dimension.clone());
        let trapdoor = ring.sample_trapdoor(
            compiler.decoder_rows(),
            compiler.trapdoor_sigma.clone(),
            compiler.gadget_base.clone(),
            compiler.digit_count,
        );
        let public_keys = ring.input_family(
            "public-keys",
            3,
            (compiler.secret_size, compiler.public_key_columns()),
        );
        let preprocessing =
            compiler.build_preprocessing(trapdoor, public_keys, 3).expect("preprocessing");
        let preprocessing = compiler
            .export_preprocessing(DslContext::new("decoder-preprocessing"), preprocessing)
            .expect("export")
            .build()
            .expect("build preprocessing");
        preprocessing.validate(&ParamEnv::default()).expect("validate preprocessing");
        preprocessing.elaborate(&ParamEnv::default()).expect("elaborate preprocessing");

        let state = ring.input("state", (1, compiler.decoder_columns()));
        let preimages = ring.input_family("preimages", 3, (compiler.decoder_columns(), 1));
        let vectors = ring.input_family("vectors", 3, (1, compiler.public_key_columns()));
        let bottoms = ring.input_family("bottoms", 3, (1, 1));
        let MaskedHighBitDecoderOutputs::Integers(outputs) = compiler
            .build_online(state, preimages, vectors, bottoms, 2.into(), false, 3)
            .expect("online")
        else {
            panic!("integer outputs")
        };
        let context = outputs
            .into_iter()
            .enumerate()
            .try_fold(DslContext::new("decoder-online"), |context, (index, output)| {
                context.int_family_output(format!("coefficient-{index}"), output)
            })
            .expect("outputs");
        let online = context.build().expect("build online");
        online.validate(&ParamEnv::default()).expect("validate online");
    }

    #[test]
    fn online_runtime_matches_explicit_threshold_decode_oracle() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let q = parameters.modulus().as_ref().clone();
        let digit_count = parameters.modulus_digits();
        let compiler = MaskedHighBitDecoderCompiler {
            modulus: IntExpr::constant(BigInt::from(q.clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            digit_count,
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            trapdoor_sigma: RealExpr::from_integer(5),
            coefficient_count: parameters.ring_dimension() as usize,
        };
        let slot_count = 3usize;
        let plaintext_modulus = BigUint::from(5u8);
        let ring = Ring::new(compiler.modulus.clone(), compiler.ring_dimension.clone());
        let preimages = Family::pack(
            (0..slot_count)
                .map(|slot| ring.input(format!("preimage-{slot}"), (compiler.decoder_columns(), 1)))
                .collect(),
        )
        .unwrap();
        let vectors = Family::pack(
            (0..slot_count)
                .map(|slot| {
                    ring.input(format!("vector-{slot}"), (1, compiler.public_key_columns()))
                })
                .collect(),
        )
        .unwrap();
        let bottoms = Family::pack(
            (0..slot_count).map(|slot| ring.input(format!("bottom-{slot}"), (1, 1))).collect(),
        )
        .unwrap();
        let MaskedHighBitDecoderOutputs::Integers(outputs) = compiler
            .build_online(
                ring.input("decoder-state", (1, compiler.decoder_columns())),
                preimages,
                vectors,
                bottoms,
                IntExpr::constant(BigInt::from(plaintext_modulus.clone())),
                false,
                slot_count,
            )
            .unwrap()
        else {
            panic!("integer outputs")
        };
        let mut context = DslContext::new("decoder-online-runtime");
        for (coefficient, output) in outputs.into_iter().enumerate() {
            context =
                context.int_family_output(format!("coefficient-{coefficient}"), output).unwrap();
        }
        let graph = context.build().unwrap();

        let decoder_coefficients = (0..compiler.coefficient_count)
            .map(|coefficient| BigUint::from(coefficient + 7))
            .collect::<Vec<_>>();
        let decoder_state = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            std::iter::once(DCRTPoly::from_biguints(&parameters, &decoder_coefficients))
                .chain((1..compiler.decoder_columns()).map(|_| DCRTPoly::const_zero(&parameters)))
                .collect(),
        );
        let preimage = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            std::iter::once(vec![DCRTPoly::const_one(&parameters)])
                .chain(
                    (1..compiler.decoder_columns())
                        .map(|_| vec![DCRTPoly::const_zero(&parameters)]),
                )
                .collect(),
        );
        let mut inputs =
            BTreeMap::from([("decoder-state".to_owned(), RuntimeValue::matrix(decoder_state))]);
        let mut expected = vec![Vec::with_capacity(compiler.coefficient_count); slot_count];
        for slot in 0..slot_count {
            let secret_coefficients = (0..compiler.coefficient_count)
                .map(|coefficient| BigUint::from(coefficient + slot + 3))
                .collect::<Vec<_>>();
            let vector = DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                std::iter::once(DCRTPoly::from_biguints(&parameters, &secret_coefficients))
                    .chain(
                        (1..compiler.public_key_columns())
                            .map(|_| DCRTPoly::const_zero(&parameters)),
                    )
                    .collect(),
            );
            let mut bottom_coefficients = Vec::with_capacity(compiler.coefficient_count);
            for coefficient in 0..compiler.coefficient_count {
                let target_plaintext = BigUint::from((slot + coefficient) % 5);
                let target = (&q * target_plaintext + &plaintext_modulus / BigUint::from(2u8)) /
                    &plaintext_modulus;
                let residual =
                    &decoder_coefficients[coefficient] - &secret_coefficients[coefficient];
                let bottom = (&target + &q - &residual) % &q;
                let noisy = (&decoder_coefficients[coefficient] + &q -
                    &secret_coefficients[coefficient] +
                    &bottom) %
                    &q;
                let decoded = (&plaintext_modulus * noisy + &q / BigUint::from(2u8)) / &q %
                    &plaintext_modulus;
                bottom_coefficients.push(bottom);
                expected[slot].push(BigInt::from(decoded));
            }
            inputs.insert(format!("preimage-{slot}"), RuntimeValue::matrix(preimage.clone()));
            inputs.insert(format!("vector-{slot}"), RuntimeValue::matrix(vector));
            inputs.insert(
                format!("bottom-{slot}"),
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_biguints(&parameters, &bottom_coefficients)],
                )),
            );
        }
        let result = execute_graph(graph, parameters, inputs);
        for coefficient in 0..compiler.coefficient_count {
            let RuntimeValue::<CpuDcrtBackend>::IndexedFamily(values) =
                &result.outputs[&format!("coefficient-{coefficient}")]
            else {
                panic!("decoded coefficient must be an integer family")
            };
            for slot in 0..slot_count {
                let RuntimeValue::Int(actual) = &values[slot] else {
                    panic!("decoded family member must be an integer")
                };
                assert_eq!(actual, &expected[slot][coefficient]);
            }
        }
    }
}
