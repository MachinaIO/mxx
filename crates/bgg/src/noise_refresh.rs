//! Naive BGG+ noise refresh expressed entirely with declarative DSL values.

use crate::{
    BggPublicKeyCompiler, BggPublicKeyWire, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
    NaiveBggVecCompiler, NaiveVecCompileError,
};
use mxx_dsl::{Bytes, DslContext, DslError, Family, HashTag, Mat, Parallel, Ring, Trapdoor};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix, IndexRange},
    types::MatrixType,
};
use thiserror::Error;

pub const NOISE_REFRESH_A_PRIME: &str = "noise_refresh_a_prime";
pub const NOISE_REFRESH_DECODER_PREIMAGES: &str = "noise_refresh_decoder_preimages";

#[derive(Clone)]
pub struct NaiveBggNoiseRefreshCompiler {
    pub public_key: BggPublicKeyCompiler,
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_size: usize,
    pub slot_count: usize,
    pub digit_count: usize,
    pub crt_scale_factors: Vec<IntExpr>,
    pub crt_plaintext_moduli: Vec<IntExpr>,
    pub reconstruction_coefficients: Vec<IntExpr>,
    pub decoder_public_columns: usize,
    pub decoder_trapdoor_sigma: RealExpr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NaiveBggNoiseRefreshArtifacts {
    pub production_id: ProductionId,
}

#[derive(Clone)]
pub struct NaiveBggNoiseRefreshPreprocessingWires {
    pub a_prime: NaiveBggPublicKeyVecWire,
    pub decoder_public_keys: Family<Mat>,
    pub decoder_preimages: Family<Mat>,
}

#[derive(Clone)]
pub struct NaiveBggNoiseRefreshArtifactWires {
    pub a_prime: NaiveBggPublicKeyVecWire,
    pub decoder_preimages: Family<Mat>,
}

#[derive(Debug, Error)]
pub enum NaiveBggNoiseRefreshError {
    #[error("noise-refresh dimensions and CRT layout must be nonzero")]
    EmptyLayout,
    #[error("noise refresh requires exactly one logical slot per ring coefficient")]
    SlotRingMismatch,
    #[error("noise-refresh CRT parameter vectors must have the same length")]
    CrtLayoutMismatch,
    #[error("noise-refresh family count or matrix type is incompatible with the layout")]
    FamilyLayoutMismatch,
    #[error("decoded noise-refresh material has the wrong output count")]
    DecodedMaterialCount,
    #[error(transparent)]
    NaiveVec(#[from] NaiveVecCompileError),
    #[error(transparent)]
    Dsl(#[from] DslError),
}

impl NaiveBggNoiseRefreshCompiler {
    pub fn validate_layout(&self) -> Result<(), NaiveBggNoiseRefreshError> {
        if self.secret_size == 0 ||
            self.slot_count == 0 ||
            self.digit_count == 0 ||
            self.decoder_public_columns == 0 ||
            self.crt_plaintext_moduli.is_empty()
        {
            return Err(NaiveBggNoiseRefreshError::EmptyLayout);
        }
        if self.ring_dimension != IntExpr::constant(self.slot_count) {
            return Err(NaiveBggNoiseRefreshError::SlotRingMismatch);
        }
        if self.crt_plaintext_moduli.len() != self.crt_scale_factors.len() ||
            self.crt_plaintext_moduli.len() != self.reconstruction_coefficients.len()
        {
            return Err(NaiveBggNoiseRefreshError::CrtLayoutMismatch);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_preprocessing(
        &self,
        hash_key: Bytes,
        refresh_id: &[u8],
        one: &NaiveBggPublicKeyVecWire,
        refreshed_input: &NaiveBggPublicKeyVecWire,
        decoded_material: &[NaiveBggPublicKeyVecWire],
        decoder_trapdoor: Trapdoor,
    ) -> Result<NaiveBggNoiseRefreshPreprocessingWires, NaiveBggNoiseRefreshError> {
        self.validate_public_bundle(one)?;
        self.validate_public_bundle(refreshed_input)?;
        let refresh_terms = self.public_refresh_terms(decoded_material)?;
        let a_prime = self.sample_a_prime(hash_key, refresh_id)?;
        let decoder_by_crt =
            self.preprocess_decoder_keys(one, refreshed_input, &a_prime, &refresh_terms)?;
        let decoder_public_keys = self.flatten_crt_families(&decoder_by_crt)?;
        let decoder_preimages = decoder_public_keys.clone().parallel_map(move |_, target| {
            decoder_trapdoor
                .sample_preimage(target, (self.decoder_public_columns, self.public_key_columns()))
                .as_mat()
        })?;
        Ok(NaiveBggNoiseRefreshPreprocessingWires {
            a_prime,
            decoder_public_keys,
            decoder_preimages,
        })
    }

    pub fn export_preprocessing(
        &self,
        context: DslContext,
        wires: NaiveBggNoiseRefreshPreprocessingWires,
    ) -> Result<DslContext, NaiveBggNoiseRefreshError> {
        Ok(context
            .public_family_output(NOISE_REFRESH_A_PRIME, wires.a_prime.matrices)?
            .public_family_output(NOISE_REFRESH_DECODER_PREIMAGES, wires.decoder_preimages)?)
    }

    pub fn import_artifacts(
        &self,
        artifacts: &NaiveBggNoiseRefreshArtifacts,
    ) -> Result<NaiveBggNoiseRefreshArtifactWires, NaiveBggNoiseRefreshError> {
        self.validate_layout()?;
        let ring = self.ring();
        Ok(NaiveBggNoiseRefreshArtifactWires {
            a_prime: NaiveBggPublicKeyVecWire {
                matrices: ring.family_artifact_input(
                    artifacts.production_id.clone(),
                    NOISE_REFRESH_A_PRIME,
                    self.slot_count,
                    (self.secret_size, self.public_key_columns()),
                    ArtifactConfidentiality::Public,
                ),
                reveal_plaintext: true,
            },
            decoder_preimages: ring.family_artifact_input(
                artifacts.production_id.clone(),
                NOISE_REFRESH_DECODER_PREIMAGES,
                self.flat_decoder_count(),
                (self.decoder_public_columns, self.public_key_columns()),
                ArtifactConfidentiality::Public,
            ),
        })
    }

    pub fn project_decoder_preimages(
        &self,
        decoder_state: Mat,
        preimages: Family<Mat>,
    ) -> Result<Family<Mat>, NaiveBggNoiseRefreshError> {
        if decoder_state.matrix_type() != &self.decoder_state_type() ||
            preimages.element_type() != &self.decoder_preimage_type() ||
            preimages.count() != &IntExpr::constant(self.flat_decoder_count())
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        Ok(preimages.parallel_map(move |_, preimage| decoder_state.clone() * preimage)?)
    }

    pub fn build_online(
        &self,
        one: &NaiveBggEncodingVecWire,
        refreshed_input: &NaiveBggEncodingVecWire,
        decoded_material: &[NaiveBggEncodingVecWire],
        artifacts: &NaiveBggNoiseRefreshArtifactWires,
        projected_decoders: Family<Mat>,
    ) -> Result<NaiveBggEncodingVecWire, NaiveBggNoiseRefreshError> {
        self.validate_encoding_bundle(one)?;
        self.validate_encoding_bundle(refreshed_input)?;
        self.validate_public_bundle(&artifacts.a_prime)?;
        if projected_decoders.element_type() != &self.vector_type() ||
            projected_decoders.count() != &IntExpr::constant(self.flat_decoder_count())
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        let refresh_terms = self.encoding_refresh_terms(decoded_material)?;
        let decoders = self.split_flat_family_by_crt(projected_decoders)?;
        let levels = self.online_level_vectors(
            one,
            refreshed_input,
            &artifacts.a_prime,
            &refresh_terms,
            &decoders,
        )?;
        let vectors = self.recompose_levels(&levels)?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys: artifacts.a_prime.matrices.clone(),
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        })
    }

    fn sample_a_prime(
        &self,
        hash_key: Bytes,
        refresh_id: &[u8],
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveBggNoiseRefreshError> {
        let mut prefix = refresh_id.to_vec();
        prefix.extend_from_slice(b":a_prime:");
        let ring = self.ring();
        let secret_size = self.secret_size;
        let columns = self.public_key_columns();
        let matrices = Parallel::range(self.slot_count).map(move |slot| {
            let mut tag = HashTag::from(prefix.clone());
            tag.push(slot);
            ring.hash_matrix(hash_key.clone(), tag, (secret_size, columns))
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: true })
    }

    fn public_refresh_terms(
        &self,
        decoded: &[NaiveBggPublicKeyVecWire],
    ) -> Result<Vec<Family<Mat>>, NaiveBggNoiseRefreshError> {
        self.validate_decoded_count(decoded.len())?;
        let compiler = NaiveBggVecCompiler { public_key: self.public_key.clone() };
        let target = self.ring().constant(
            (self.secret_size, 1),
            ConstantMatrix::UnitColumn { index: IntExpr::constant(self.secret_size - 1) },
        );
        let collapsed = decoded
            .iter()
            .map(|value| {
                let projected = compiler.matrix_mul_public_keys(value, &target)?;
                self.collapse_slots(projected.matrices)
            })
            .collect::<Result<Vec<_>, NaiveBggNoiseRefreshError>>()?;
        self.assemble_refresh_terms(&collapsed)
    }

    fn encoding_refresh_terms(
        &self,
        decoded: &[NaiveBggEncodingVecWire],
    ) -> Result<Vec<Family<Mat>>, NaiveBggNoiseRefreshError> {
        self.validate_decoded_count(decoded.len())?;
        let compiler = NaiveBggVecCompiler { public_key: self.public_key.clone() };
        let target = self.ring().constant(
            (self.secret_size, 1),
            ConstantMatrix::UnitColumn { index: IntExpr::constant(self.secret_size - 1) },
        );
        let collapsed = decoded
            .iter()
            .map(|value| {
                let projected = compiler.matrix_mul_encodings(value, &target)?;
                self.collapse_slots(projected.vectors)
            })
            .collect::<Result<Vec<_>, NaiveBggNoiseRefreshError>>()?;
        self.assemble_refresh_terms(&collapsed)
    }

    fn collapse_slots(&self, family: Family<Mat>) -> Result<Mat, NaiveBggNoiseRefreshError> {
        if family.count() != &IntExpr::constant(self.slot_count) {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        Ok((0..self.slot_count)
            .map(|slot| {
                family.get_static(slot) *
                    self.ring().constant(
                        (1, 1),
                        ConstantMatrix::Rotation { exponent: IntExpr::constant(slot) },
                    )
            })
            .reduce(|sum, term| sum + term)
            .expect("nonzero slot count"))
    }

    fn assemble_refresh_terms(
        &self,
        collapsed: &[Mat],
    ) -> Result<Vec<Family<Mat>>, NaiveBggNoiseRefreshError> {
        (0..self.crt_depth())
            .map(|crt| {
                Family::pack(
                    (0..self.slot_count)
                        .map(|slot| {
                            (0..self.digit_count)
                                .map(|digit| {
                                    let index = slot * self.crt_depth() * self.digit_count +
                                        crt * self.digit_count +
                                        digit;
                                    self.embed_digit(collapsed[index].clone(), digit)
                                })
                                .reduce(|sum, term| sum + term)
                                .expect("nonzero digit count")
                        })
                        .collect(),
                )
                .map_err(Into::into)
            })
            .collect()
    }

    fn embed_digit(&self, projected: Mat, digit: usize) -> Mat {
        let zero = self
            .ring()
            .zero((projected.matrix_type().rows.clone(), projected.matrix_type().columns.clone()));
        Mat::concat(
            ConcatAxis::Columns,
            (0..self.public_key_columns())
                .map(|column| {
                    if column % self.digit_count == digit {
                        projected.clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect(),
        )
    }

    fn preprocess_decoder_keys(
        &self,
        one: &NaiveBggPublicKeyVecWire,
        refreshed: &NaiveBggPublicKeyVecWire,
        a_prime: &NaiveBggPublicKeyVecWire,
        refresh_terms: &[Family<Mat>],
    ) -> Result<Vec<Family<Mat>>, NaiveBggNoiseRefreshError> {
        (0..self.crt_depth())
            .map(|crt| {
                let scale = self.crt_scale_factors[crt].clone();
                let gadget = self.ring().gadget(
                    self.secret_size,
                    self.public_key.base.clone(),
                    self.digit_count,
                );
                Family::pack(
                    (0..self.slot_count)
                        .map(|slot| {
                            let one_term = self.public_key.matrix_mul(
                                &BggPublicKeyWire {
                                    matrix: one.matrices.get_static(slot),
                                    reveal_plaintext: true,
                                },
                                &(a_prime.matrices.get_static(slot) *
                                    self.ring().polynomial([scale.clone()])),
                            );
                            let input_term = self.public_key.matrix_mul(
                                &BggPublicKeyWire {
                                    matrix: refreshed.matrices.get_static(slot),
                                    reveal_plaintext: true,
                                },
                                &(gadget.clone() * self.ring().polynomial([scale.clone()])),
                            );
                            input_term.matrix + refresh_terms[crt].get_static(slot) -
                                one_term.matrix
                        })
                        .collect(),
                )
                .map_err(Into::into)
            })
            .collect()
    }

    fn online_level_vectors(
        &self,
        one: &NaiveBggEncodingVecWire,
        refreshed: &NaiveBggEncodingVecWire,
        a_prime: &NaiveBggPublicKeyVecWire,
        refresh_terms: &[Family<Mat>],
        decoders: &[Family<Mat>],
    ) -> Result<Vec<Family<Mat>>, NaiveBggNoiseRefreshError> {
        (0..self.crt_depth())
            .map(|crt| {
                let scale = self.crt_scale_factors[crt].clone();
                let gadget = self.ring().gadget(
                    self.secret_size,
                    self.public_key.base.clone(),
                    self.digit_count,
                );
                Family::pack(
                    (0..self.slot_count)
                        .map(|slot| {
                            let a_decomposed = (a_prime.matrices.get_static(slot) *
                                self.ring().polynomial([scale.clone()]))
                            .decompose(self.public_key.base.clone(), self.digit_count)
                            .as_mat();
                            let gadget_decomposed = (gadget.clone() *
                                self.ring().polynomial([scale.clone()]))
                            .decompose(self.public_key.base.clone(), self.digit_count)
                            .as_mat();
                            refreshed.vectors.get_static(slot) * gadget_decomposed +
                                refresh_terms[crt].get_static(slot) -
                                one.vectors.get_static(slot) * a_decomposed -
                                decoders[crt].get_static(slot)
                        })
                        .collect(),
                )
                .map_err(Into::into)
            })
            .collect()
    }

    fn recompose_levels(
        &self,
        levels: &[Family<Mat>],
    ) -> Result<Family<Mat>, NaiveBggNoiseRefreshError> {
        let wide_levels = levels
            .iter()
            .map(|level| {
                Mat::concat(
                    ConcatAxis::Columns,
                    (0..self.slot_count).map(|slot| level.get_static(slot)).collect(),
                )
            })
            .collect();
        let wide = Mat::crt_recompose(
            wide_levels,
            self.crt_plaintext_moduli.clone(),
            self.reconstruction_coefficients.clone(),
        );
        Ok(Family::pack(
            (0..self.slot_count)
                .map(|slot| {
                    wide.clone()
                        .slice(
                            None,
                            Some(IndexRange {
                                start: IntExpr::constant(slot * self.public_key_columns()),
                                end: IntExpr::constant((slot + 1) * self.public_key_columns()),
                            }),
                        )
                        .reshape(1, self.public_key_columns())
                })
                .collect(),
        )?)
    }

    fn flatten_crt_families(
        &self,
        families: &[Family<Mat>],
    ) -> Result<Family<Mat>, NaiveBggNoiseRefreshError> {
        Ok(Family::pack(
            (0..self.slot_count)
                .flat_map(|slot| families.iter().map(move |family| family.get_static(slot)))
                .collect(),
        )?)
    }

    fn split_flat_family_by_crt(
        &self,
        family: Family<Mat>,
    ) -> Result<Vec<Family<Mat>>, NaiveBggNoiseRefreshError> {
        (0..self.crt_depth())
            .map(|crt| {
                Family::pack(
                    (0..self.slot_count)
                        .map(|slot| family.get_static(slot * self.crt_depth() + crt))
                        .collect(),
                )
                .map_err(Into::into)
            })
            .collect()
    }

    fn validate_public_bundle(
        &self,
        value: &NaiveBggPublicKeyVecWire,
    ) -> Result<(), NaiveBggNoiseRefreshError> {
        self.validate_layout()?;
        if value.matrices.element_type() != &self.public_key_type() ||
            value.matrices.count() != &IntExpr::constant(self.slot_count)
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        Ok(())
    }

    fn validate_encoding_bundle(
        &self,
        value: &NaiveBggEncodingVecWire,
    ) -> Result<(), NaiveBggNoiseRefreshError> {
        if value.vectors.element_type() != &self.vector_type() ||
            value.pubkeys.element_type() != &self.public_key_type() ||
            value.vectors.count() != &IntExpr::constant(self.slot_count) ||
            value.pubkeys.count() != value.vectors.count() ||
            value.plaintexts.as_ref().is_some_and(|plaintexts| {
                plaintexts.element_type() != &self.scalar_type() ||
                    plaintexts.count() != value.vectors.count()
            })
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        Ok(())
    }

    fn validate_decoded_count(&self, count: usize) -> Result<(), NaiveBggNoiseRefreshError> {
        if count != self.slot_count * self.crt_depth() * self.digit_count {
            return Err(NaiveBggNoiseRefreshError::DecodedMaterialCount);
        }
        Ok(())
    }

    fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }
    fn public_key_columns(&self) -> usize {
        self.secret_size * self.digit_count
    }
    fn crt_depth(&self) -> usize {
        self.crt_plaintext_moduli.len()
    }
    fn flat_decoder_count(&self) -> usize {
        self.slot_count * self.crt_depth()
    }
    fn public_key_type(&self) -> MatrixType {
        self.ring().matrix_type((self.secret_size, self.public_key_columns()))
    }
    fn vector_type(&self) -> MatrixType {
        self.ring().matrix_type((1, self.public_key_columns()))
    }
    fn scalar_type(&self) -> MatrixType {
        self.ring().matrix_type((1, 1))
    }
    fn decoder_state_type(&self) -> MatrixType {
        self.ring().matrix_type((1, self.decoder_public_columns))
    }
    fn decoder_preimage_type(&self) -> MatrixType {
        self.ring().matrix_type((self.decoder_public_columns, self.public_key_columns()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    #[test]
    fn preprocessing_and_online_refresh_build_valid_graphs() {
        let ring = Ring::new(65_537, 2);
        let compiler = NaiveBggNoiseRefreshCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 9.into(),
            },
            modulus: 65_537.into(),
            ring_dimension: 2.into(),
            secret_size: 2,
            slot_count: 2,
            digit_count: 9,
            crt_scale_factors: vec![3.into()],
            crt_plaintext_moduli: vec![5.into()],
            reconstruction_coefficients: vec![1.into()],
            decoder_public_columns: 22,
            decoder_trapdoor_sigma: RealExpr::from_integer(5),
        };
        let public = |name: &str| NaiveBggPublicKeyVecWire {
            matrices: ring.input_family(name, 2, (2, 18)),
            reveal_plaintext: true,
        };
        let one = public("one");
        let refreshed = public("refreshed");
        let decoded = (0..18).map(|index| public(&format!("decoded-{index}"))).collect::<Vec<_>>();
        let trapdoor = ring.sample_trapdoor(2, 5, 4, 9);
        assert_eq!(one.matrices.element_type(), &compiler.public_key_type());
        assert_eq!(one.matrices.count(), &IntExpr::constant(compiler.slot_count));
        let preprocessing = compiler
            .build_preprocessing(
                ring.bytes_input("hash-key", 32),
                b"test",
                &one,
                &refreshed,
                &decoded,
                trapdoor,
            )
            .expect("preprocessing");
        let graph = compiler
            .export_preprocessing(DslContext::new("noise-refresh"), preprocessing.clone())
            .expect("outputs")
            .build()
            .expect("graph");
        graph.validate(&ParamEnv::default()).expect("valid graph");
        graph.elaborate(&ParamEnv::default()).expect("symbolic preprocessing graph");

        let artifacts = NaiveBggNoiseRefreshArtifactWires {
            a_prime: preprocessing.a_prime,
            decoder_preimages: preprocessing.decoder_preimages,
        };
        let projected = compiler
            .project_decoder_preimages(
                ring.input("decoder-state", (1, compiler.decoder_public_columns)),
                artifacts.decoder_preimages.clone(),
            )
            .expect("project decoders");
        let encoding = |name: &str| NaiveBggEncodingVecWire {
            vectors: ring.input_family(format!("{name}-vectors"), 2, (1, 18)),
            pubkeys: ring.input_family(format!("{name}-keys"), 2, (2, 18)),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(ring.input_family(format!("{name}-plaintexts"), 2, (1, 1))),
        };
        let online_one = encoding("online-one");
        let online_refreshed = encoding("online-refreshed");
        let online_decoded =
            (0..18).map(|index| encoding(&format!("online-decoded-{index}"))).collect::<Vec<_>>();
        let refreshed = compiler
            .build_online(&online_one, &online_refreshed, &online_decoded, &artifacts, projected)
            .expect("online refresh");
        let online_graph = DslContext::new("noise-refresh-online")
            .family_output("vectors", refreshed.vectors)
            .expect("vectors")
            .family_output("public-keys", refreshed.pubkeys)
            .expect("public keys")
            .build()
            .expect("online graph");
        online_graph.validate(&ParamEnv::default()).expect("valid online graph");
        online_graph.elaborate(&ParamEnv::default()).expect("symbolic online graph");
    }

    #[test]
    fn online_runtime_matches_explicit_zero_refresh_oracle() {
        let parameters = DCRTPolyParams::new(4, 2, 10, 5);
        let q = parameters.modulus().as_ref().clone();
        let (plaintext_moduli, _, depth) = parameters.to_crt();
        let reconstruction_coefficients = parameters.reconst_coeffs();
        let digit_count = parameters.modulus_digits();
        let ring = Ring::new(BigInt::from(q.clone()), parameters.ring_dimension() as usize);
        let compiler = NaiveBggNoiseRefreshCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
            modulus: IntExpr::constant(BigInt::from(q.clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            slot_count: parameters.ring_dimension() as usize,
            digit_count,
            crt_scale_factors: plaintext_moduli
                .iter()
                .map(|modulus| IntExpr::constant(BigInt::from(&q / BigUint::from(*modulus))))
                .collect(),
            crt_plaintext_moduli: plaintext_moduli
                .iter()
                .map(|modulus| IntExpr::constant(*modulus))
                .collect(),
            reconstruction_coefficients: reconstruction_coefficients
                .iter()
                .cloned()
                .map(IntExpr::constant)
                .collect(),
            decoder_public_columns: digit_count + 2,
            decoder_trapdoor_sigma: RealExpr::from_integer(5),
        };
        assert_eq!(depth, compiler.crt_depth());

        let encoding = |prefix: &str| NaiveBggEncodingVecWire {
            vectors: ring.input_family(
                format!("{prefix}-vectors"),
                compiler.slot_count,
                (1, compiler.public_key_columns()),
            ),
            pubkeys: ring.input_family(
                format!("{prefix}-keys"),
                compiler.slot_count,
                (compiler.secret_size, compiler.public_key_columns()),
            ),
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        };
        let one = encoding("one");
        let refreshed = encoding("refreshed");
        let decoded = encoding("decoded");
        let decoded_material =
            vec![decoded; compiler.slot_count * compiler.crt_depth() * compiler.digit_count];
        let artifacts = NaiveBggNoiseRefreshArtifactWires {
            a_prime: NaiveBggPublicKeyVecWire {
                matrices: ring.input_family(
                    "a-prime",
                    compiler.slot_count,
                    (compiler.secret_size, compiler.public_key_columns()),
                ),
                reveal_plaintext: true,
            },
            decoder_preimages: ring.input_family(
                "unused-preimages",
                compiler.flat_decoder_count(),
                (compiler.decoder_public_columns, compiler.public_key_columns()),
            ),
        };
        let projected = ring.input_family(
            "projected-decoders",
            compiler.flat_decoder_count(),
            (1, compiler.public_key_columns()),
        );
        let output = compiler
            .build_online(&one, &refreshed, &decoded_material, &artifacts, projected)
            .unwrap();
        let mut context = DslContext::new("noise-refresh-online-runtime");
        for slot in 0..compiler.slot_count {
            context =
                context.output(format!("slot-{slot}"), output.vectors.get_static(slot)).unwrap();
        }
        let graph = context.build().unwrap();

        let zero_vector = DCRTPolyMatrix::zero(&parameters, 1, compiler.public_key_columns());
        let zero_public =
            DCRTPolyMatrix::zero(&parameters, compiler.secret_size, compiler.public_key_columns());
        let family = |value: &DCRTPolyMatrix, count: usize| {
            RuntimeValue::IndexedFamily(
                (0..count).map(|_| RuntimeValue::matrix(value.clone())).collect(),
            )
        };
        let slot_values = (0..compiler.slot_count)
            .map(|slot| {
                DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    (0..compiler.public_key_columns())
                        .map(|column| {
                            DCRTPoly::from_biguints(
                                &parameters,
                                &(0..parameters.ring_dimension() as usize)
                                    .map(|coefficient| {
                                        BigUint::from(1 + slot + column + coefficient)
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let projected_values = slot_values
            .iter()
            .flat_map(|source| {
                plaintext_moduli.iter().map(|plaintext_modulus| {
                    let scale = &q / BigUint::from(*plaintext_modulus);
                    source.clone() * DCRTPoly::from_biguint_to_constant(&parameters, &q - scale)
                })
            })
            .collect::<Vec<_>>();
        let result = execute_graph(
            graph,
            parameters.clone(),
            BTreeMap::from([
                ("one-vectors".to_owned(), family(&zero_vector, compiler.slot_count)),
                ("one-keys".to_owned(), family(&zero_public, compiler.slot_count)),
                ("refreshed-vectors".to_owned(), family(&zero_vector, compiler.slot_count)),
                ("refreshed-keys".to_owned(), family(&zero_public, compiler.slot_count)),
                ("decoded-vectors".to_owned(), family(&zero_vector, compiler.slot_count)),
                ("decoded-keys".to_owned(), family(&zero_public, compiler.slot_count)),
                ("a-prime".to_owned(), family(&zero_public, compiler.slot_count)),
                (
                    "projected-decoders".to_owned(),
                    RuntimeValue::IndexedFamily(
                        projected_values.into_iter().map(RuntimeValue::matrix).collect(),
                    ),
                ),
            ]),
        );
        let reconstruction_sum = reconstruction_coefficients.iter().sum::<BigUint>() % &q;
        for (slot, source) in slot_values.iter().enumerate() {
            let expected = source.clone() *
                DCRTPoly::from_biguint_to_constant(&parameters, reconstruction_sum.clone());
            assert_eq!(matrix_output(&result, &format!("slot-{slot}")), &expected);
        }
    }
}
