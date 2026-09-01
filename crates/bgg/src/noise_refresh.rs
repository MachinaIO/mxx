//! Naive BGG+ noise refresh expressed entirely with declarative DSL values.

use crate::{
    BggPublicKeyCompiler, BggPublicKeyWire, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
    NaiveBggVecCompiler, NaiveVecCompileError,
};
use mxx_dsl::{
    Bytes, DslContext, DslError, Family, HashTag, Mat, Parallel, Preimage, Ring, Trapdoor,
};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix},
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
    /// Zero rows appended below each decoder target. Diamond applications use
    /// one row so the target matches the two-row final injector basis; callers
    /// with a scalar decoder trapdoor use zero.
    pub decoder_zero_rows: usize,
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
    pub decoder_preimages: Family<Preimage>,
}

#[derive(Clone)]
pub struct NaiveBggNoiseRefreshArtifactWires {
    pub a_prime: NaiveBggPublicKeyVecWire,
    pub decoder_preimages: Family<Preimage>,
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
        let decoder_columns = self.decoder_public_columns;
        let public_columns = self.public_key_columns();
        let zero_rows = self.decoder_zero_rows;
        let ring = self.ring();
        let decoder_preimages =
            decoder_public_keys.clone().parallel_map_values(move |_, target| {
                let target = if zero_rows == 0 {
                    target
                } else {
                    Mat::concat(
                        ConcatAxis::Rows,
                        vec![target, ring.zero((zero_rows, public_columns))],
                    )
                };
                decoder_trapdoor.sample_preimage(target, (decoder_columns, public_columns))
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
            .public_preimage_family_output(
                NOISE_REFRESH_DECODER_PREIMAGES,
                wires.decoder_preimages,
            )?)
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
            decoder_preimages: ring.preimage_family_artifact_input(
                artifacts.production_id.clone(),
                NOISE_REFRESH_DECODER_PREIMAGES,
                vec![IntExpr::constant(self.flat_decoder_count())],
                (self.decoder_public_columns, self.public_key_columns()),
                ArtifactConfidentiality::Public,
            ),
        })
    }

    pub fn project_decoder_preimages(
        &self,
        decoder_state: Mat,
        preimages: Family<Preimage>,
    ) -> Result<Family<Mat>, NaiveBggNoiseRefreshError> {
        if decoder_state.matrix_type() != &self.decoder_state_type() ||
            preimages.element_type() != &self.decoder_preimage_type() ||
            preimages.count() != &IntExpr::constant(self.flat_decoder_count())
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        // Applying each decoder preimage is the explicit projection
        // D K_j: the state row is multiplied by the supplied K_j on the
        // right, so the preimage relation is consumed rather than inferred.
        Ok(Parallel::range(self.flat_decoder_count()).map_values(move |index| {
            decoder_state.clone().apply_preimage(preimages.get(vec![index.as_int()]))
        })?)
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
        // The unit column is an arbitrary projection target used to collapse
        // decoded public matrices; its decomposition is an action target, not
        // a claim that the column is a canonical gadget encoding.
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
        // Apply the same target projection to decoded encoding vectors before
        // assembling the CRT refresh terms, preserving the vector carrier.
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
        // A slot family is collapsed as sum_i C_i R_i, where R_i is the
        // rotation monomial for slot i; each term retains its original row
        // carrier while moving it into the common coefficient position.
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
                            // This public decoder target combines the scaled
                            // refreshed-input term, the CRT refresh term, and
                            // the scaled one term as input + refresh - one.
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
                            .decompose(self.public_key.base.clone(), self.digit_count);
                            let gadget_decomposed = (self.ring().polynomial([scale.clone()]) *
                                gadget.clone())
                            .decompose(self.public_key.base.clone(), self.digit_count);
                            // The online level computes refreshed*K_g + R -
                            // one*K_a - decoder, with each decomposition
                            // consumed on the right in its existing carrier.
                            refreshed.vectors.get_static(slot).mul_decomposed(gadget_decomposed) +
                                refresh_terms[crt].get_static(slot) -
                                one.vectors.get_static(slot).mul_decomposed(a_decomposed) -
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
        Ok(Family::pack(
            (0..self.slot_count)
                .map(|slot| {
                    // CRT recomposition is the coefficient-wise linear
                    // combination of all plaintext-modulus levels.
                    Mat::crt_recompose(
                        levels.iter().map(|level| level.get_static(slot)).collect(),
                        self.crt_plaintext_moduli.clone(),
                        self.reconstruction_coefficients.clone(),
                    )
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
            decoder_zero_rows: 0,
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
        let modulus = compiler.modulus.clone();
        let artifacts = NaiveBggNoiseRefreshArtifactWires {
            a_prime: NaiveBggPublicKeyVecWire {
                matrices: ring.input_family(
                    "a-prime",
                    compiler.slot_count,
                    (compiler.secret_size, compiler.public_key_columns()),
                ),
                reveal_plaintext: true,
            },
            decoder_preimages: ring
                .input_family(
                    "unused-preimages",
                    compiler.flat_decoder_count(),
                    (compiler.decoder_public_columns, compiler.public_key_columns()),
                )
                .parallel_map_values(move |_, preimage| {
                    preimage.decompose(modulus.clone(), 1).into_preimage_relation()
                })
                .unwrap(),
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
            RuntimeValue::Family((0..count).map(|_| RuntimeValue::matrix(value.clone())).collect())
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
                    RuntimeValue::Family(
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
