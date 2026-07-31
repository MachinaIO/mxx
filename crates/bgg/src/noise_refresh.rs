use crate::{
    AdvancedGateLowering, BggPublicKeyCompiler, CircuitCompileError, NaiveBggEncodingVecWire,
    NaiveBggPublicKeyVecWire, NaiveBggVecCompiler, NaiveVecCompileError, PolyCircuitCompiler,
};
use mxx_gadgets::{Poly, circuit::PolyCircuit};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, OutputFamilyError, RealExpr,
    SubgraphBuildError, TrapdoorWire, WireRef,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix, HashVariant, LoopInputMode, MatrixBinaryOp},
    types::MatrixType,
};
use thiserror::Error;

pub const NOISE_REFRESH_A_PRIME: &str = "noise_refresh_a_prime";
pub const NOISE_REFRESH_DECODER_PREIMAGES: &str = "noise_refresh_decoder_preimages";

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NaiveBggNoiseRefreshPreprocessingWires {
    pub a_prime: NaiveBggPublicKeyVecWire,
    /// Slot-major, then CRT-level order.
    pub decoder_public_keys: MatrixFamilyWire,
    /// Slot-major, then CRT-level order.
    pub decoder_preimages: MatrixFamilyWire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NaiveBggNoiseRefreshArtifactWires {
    pub a_prime: NaiveBggPublicKeyVecWire,
    /// Slot-major, then CRT-level order.
    pub decoder_preimages: MatrixFamilyWire,
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
    Circuit(#[from] CircuitCompileError),
    #[error(transparent)]
    NaiveVec(#[from] NaiveVecCompileError),
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
    #[error(transparent)]
    OutputFamily(#[from] OutputFamilyError),
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

    pub fn compile_public_material<P, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: NaiveBggPublicKeyVecWire,
        inputs: impl IntoIterator<Item = NaiveBggPublicKeyVecWire>,
        lowering: &mut L,
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, NaiveBggNoiseRefreshError>
    where
        P: Poly,
        L: AdvancedGateLowering<P, NaiveBggPublicKeyVecWire>,
    {
        self.validate_layout()?;
        Ok(PolyCircuitCompiler { public_key: self.public_key.clone() }
            .compile_naive_public_keys_with_lowering(builder, circuit, one, inputs, lowering)?)
    }

    pub fn compile_encoding_material<P, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: NaiveBggEncodingVecWire,
        inputs: impl IntoIterator<Item = NaiveBggEncodingVecWire>,
        lowering: &mut L,
    ) -> Result<Vec<NaiveBggEncodingVecWire>, NaiveBggNoiseRefreshError>
    where
        P: Poly,
        L: AdvancedGateLowering<P, NaiveBggEncodingVecWire>,
    {
        self.validate_layout()?;
        Ok(PolyCircuitCompiler { public_key: self.public_key.clone() }
            .compile_naive_encodings_with_lowering(builder, circuit, one, inputs, lowering)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        refresh_id: &[u8],
        one: &NaiveBggPublicKeyVecWire,
        refreshed_input: &NaiveBggPublicKeyVecWire,
        decoded_material: &[NaiveBggPublicKeyVecWire],
        decoder_trapdoor: &TrapdoorWire,
    ) -> Result<NaiveBggNoiseRefreshPreprocessingWires, NaiveBggNoiseRefreshError> {
        self.validate_public_bundle(one)?;
        self.validate_public_bundle(refreshed_input)?;
        self.validate_decoder_trapdoor(decoder_trapdoor)?;
        let refresh_terms = self.public_refresh_terms(builder, decoded_material)?;
        let a_prime = self.sample_a_prime(builder, hash_key, refresh_id)?;
        let decoder_by_crt =
            self.preprocess_decoder_keys(builder, one, refreshed_input, &a_prime, &refresh_terms)?;
        let decoder_public_keys = self.flatten_crt_families(builder, &decoder_by_crt)?;
        let decoder_preimages =
            self.sample_decoder_preimages(builder, decoder_trapdoor, &decoder_public_keys)?;
        Ok(NaiveBggNoiseRefreshPreprocessingWires {
            a_prime,
            decoder_public_keys,
            decoder_preimages,
        })
    }

    pub fn export_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        wires: &NaiveBggNoiseRefreshPreprocessingWires,
    ) {
        builder.output_family_wire(
            NOISE_REFRESH_A_PRIME,
            &wires.a_prime.matrices,
            ArtifactConfidentiality::Public,
        );
        builder.output_family_wire(
            NOISE_REFRESH_DECODER_PREIMAGES,
            &wires.decoder_preimages,
            ArtifactConfidentiality::Public,
        );
    }

    pub fn import_artifacts(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &NaiveBggNoiseRefreshArtifacts,
    ) -> Result<NaiveBggNoiseRefreshArtifactWires, NaiveBggNoiseRefreshError> {
        self.validate_layout()?;
        Ok(NaiveBggNoiseRefreshArtifactWires {
            a_prime: NaiveBggPublicKeyVecWire {
                matrices: builder.artifact_family_input(
                    "noise_refresh_a_prime_input",
                    self.public_key_type(),
                    artifacts.production_id.clone(),
                    NOISE_REFRESH_A_PRIME,
                    IntExpr::constant(self.slot_count),
                    ArtifactConfidentiality::Public,
                ),
                reveal_plaintext: true,
            },
            decoder_preimages: builder.artifact_family_input(
                "noise_refresh_decoder_preimages_input",
                self.decoder_preimage_type(),
                artifacts.production_id.clone(),
                NOISE_REFRESH_DECODER_PREIMAGES,
                IntExpr::constant(self.flat_decoder_count()),
                ArtifactConfidentiality::Public,
            ),
        })
    }

    pub fn project_decoder_preimages(
        &self,
        builder: &mut GraphBuilder,
        decoder_state: &MatrixWire,
        preimages: &MatrixFamilyWire,
    ) -> Result<MatrixFamilyWire, NaiveBggNoiseRefreshError> {
        self.validate_layout()?;
        if decoder_state.matrix_type != self.decoder_state_type() ||
            preimages.matrix_type != self.decoder_preimage_type() ||
            preimages.count != IntExpr::constant(self.flat_decoder_count())
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        let mut body = GraphBuilder::new("noise-refresh-project-decoders", Vec::new());
        let body_preimage = body.input("0_preimage", self.decoder_preimage_type());
        let body_state = body.input("1_decoder_state", self.decoder_state_type());
        let output = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &body_state,
            &body_preimage,
            self.vector_type(),
        );
        body.value_output_wire("0_decoder", output.wire);
        let [family] = builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(self.flat_decoder_count()),
                "decoder",
                Vec::new(),
                vec![preimages.wire, decoder_state.wire],
                vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
                &[self.vector_type()],
            )?
            .try_into()
            .expect("one projected decoder output was declared");
        Ok(family)
    }

    pub fn build_online(
        &self,
        builder: &mut GraphBuilder,
        one: &NaiveBggEncodingVecWire,
        refreshed_input: &NaiveBggEncodingVecWire,
        decoded_material: &[NaiveBggEncodingVecWire],
        artifacts: &NaiveBggNoiseRefreshArtifactWires,
        projected_decoders: &MatrixFamilyWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveBggNoiseRefreshError> {
        self.validate_encoding_bundle(one)?;
        self.validate_encoding_bundle(refreshed_input)?;
        self.validate_public_bundle(&artifacts.a_prime)?;
        if projected_decoders.matrix_type != self.vector_type() ||
            projected_decoders.count != IntExpr::constant(self.flat_decoder_count())
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        let refresh_terms = self.encoding_refresh_terms(builder, decoded_material)?;
        let decoder_by_crt = self.split_flat_family_by_crt(builder, projected_decoders)?;
        let level_vectors = self.online_level_vectors(
            builder,
            one,
            refreshed_input,
            &artifacts.a_prime,
            &refresh_terms,
            &decoder_by_crt,
        )?;
        let vectors = self.recompose_levels(builder, &level_vectors)?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys: artifacts.a_prime.matrices.clone(),
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        })
    }

    fn sample_a_prime(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        refresh_id: &[u8],
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveBggNoiseRefreshError> {
        let mut body = GraphBuilder::new("noise-refresh-a-prime", Vec::new());
        let body_hash_key = body.bytes_input("0_hash_key", 32);
        let mut tag = Vec::with_capacity(refresh_id.len() + b":a_prime:".len());
        tag.extend_from_slice(refresh_id);
        tag.extend_from_slice(b":a_prime:");
        let output = body.hash_sample_with_encoded_tags(
            body_hash_key,
            self.public_key_type(),
            HashVariant::Plain,
            tag,
            Vec::new(),
            Vec::new(),
            vec![IntExpr::Var("slot".to_owned())],
            None,
            None,
        );
        body.value_output_wire("0_a_prime", output.wire);
        let [matrices] = builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(self.slot_count),
                "slot",
                Vec::new(),
                vec![hash_key],
                vec![LoopInputMode::Broadcast],
                &[self.public_key_type()],
            )?
            .try_into()
            .expect("one A-prime output was declared");
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: true })
    }

    fn public_refresh_terms(
        &self,
        builder: &mut GraphBuilder,
        decoded: &[NaiveBggPublicKeyVecWire],
    ) -> Result<Vec<MatrixFamilyWire>, NaiveBggNoiseRefreshError> {
        self.validate_decoded_count(decoded.len())?;
        let vec_compiler = NaiveBggVecCompiler { public_key: self.public_key.clone() };
        let target = builder.constant_matrix(
            self.matrix_type(self.secret_size, 1),
            ConstantMatrix::UnitColumn { index: IntExpr::constant(self.secret_size - 1) },
        );
        let projected = decoded
            .iter()
            .map(|value| vec_compiler.matrix_mul_public_keys(builder, value, &target))
            .collect::<Result<Vec<_>, _>>()?;
        let collapsed = projected
            .iter()
            .map(|value| self.collapse_slots(builder, &value.matrices))
            .collect::<Result<Vec<_>, _>>()?;
        self.assemble_refresh_terms(builder, &collapsed)
    }

    fn encoding_refresh_terms(
        &self,
        builder: &mut GraphBuilder,
        decoded: &[NaiveBggEncodingVecWire],
    ) -> Result<Vec<MatrixFamilyWire>, NaiveBggNoiseRefreshError> {
        self.validate_decoded_count(decoded.len())?;
        let vec_compiler = NaiveBggVecCompiler { public_key: self.public_key.clone() };
        let target = builder.constant_matrix(
            self.matrix_type(self.secret_size, 1),
            ConstantMatrix::UnitColumn { index: IntExpr::constant(self.secret_size - 1) },
        );
        let projected = decoded
            .iter()
            .map(|value| vec_compiler.matrix_mul_encodings(builder, value, &target))
            .collect::<Result<Vec<_>, _>>()?;
        let collapsed = projected
            .iter()
            .map(|value| self.collapse_slots(builder, &value.vectors))
            .collect::<Result<Vec<_>, _>>()?;
        self.assemble_refresh_terms(builder, &collapsed)
    }

    fn collapse_slots(
        &self,
        builder: &mut GraphBuilder,
        family: &MatrixFamilyWire,
    ) -> Result<MatrixWire, NaiveBggNoiseRefreshError> {
        if family.count != IntExpr::constant(self.slot_count) {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        let mut terms = Vec::with_capacity(self.slot_count);
        for slot in 0..self.slot_count {
            let value = builder.family_get_static(family, IntExpr::constant(slot));
            let rotation = builder.constant_matrix(
                self.scalar_type(),
                ConstantMatrix::Rotation { exponent: IntExpr::constant(slot) },
            );
            terms.push(builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &value,
                &rotation,
                value.matrix_type.clone(),
            ));
        }
        Ok(sum_matrices(builder, terms))
    }

    fn assemble_refresh_terms(
        &self,
        builder: &mut GraphBuilder,
        collapsed: &[MatrixWire],
    ) -> Result<Vec<MatrixFamilyWire>, NaiveBggNoiseRefreshError> {
        let mut by_crt = Vec::with_capacity(self.crt_depth());
        for crt in 0..self.crt_depth() {
            let mut by_slot = Vec::with_capacity(self.slot_count);
            for slot in 0..self.slot_count {
                let digit_terms = (0..self.digit_count)
                    .map(|digit| {
                        let index = slot * self.crt_depth() * self.digit_count +
                            crt * self.digit_count +
                            digit;
                        self.embed_digit(builder, &collapsed[index], digit)
                    })
                    .collect::<Vec<_>>();
                by_slot.push(sum_matrices(builder, digit_terms));
            }
            by_crt.push(builder.family_pack(&by_slot)?);
        }
        Ok(by_crt)
    }

    fn embed_digit(
        &self,
        builder: &mut GraphBuilder,
        projected: &MatrixWire,
        digit: usize,
    ) -> MatrixWire {
        let zero = builder.constant_matrix(
            MatrixType { columns: IntExpr::constant(1), ..projected.matrix_type.clone() },
            ConstantMatrix::Zero,
        );
        let columns =
            (0..self.public_key_columns())
                .map(|column| {
                    if column % self.digit_count == digit {
                        projected.clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect::<Vec<_>>();
        builder.concat(
            ConcatAxis::Columns,
            &columns,
            MatrixType {
                columns: IntExpr::constant(self.public_key_columns()),
                ..projected.matrix_type.clone()
            },
        )
    }

    fn preprocess_decoder_keys(
        &self,
        builder: &mut GraphBuilder,
        one: &NaiveBggPublicKeyVecWire,
        refreshed_input: &NaiveBggPublicKeyVecWire,
        a_prime: &NaiveBggPublicKeyVecWire,
        refresh_terms: &[MatrixFamilyWire],
    ) -> Result<Vec<MatrixFamilyWire>, NaiveBggNoiseRefreshError> {
        let mut outputs = Vec::with_capacity(self.crt_depth());
        for crt in 0..self.crt_depth() {
            let mut body =
                GraphBuilder::new(format!("noise-refresh-preprocess-crt-{crt}"), Vec::new());
            let body_one = body.input("0_one", self.public_key_type());
            let body_input = body.input("1_refreshed_input", self.public_key_type());
            let body_a_prime = body.input("2_a_prime", self.public_key_type());
            let body_refresh = body.input("3_refresh", self.public_key_type());
            let scaled_a_prime =
                body.matrix_scale(&body_a_prime, self.crt_scale_factors[crt].clone());
            let one_term = self.public_key.matrix_mul(
                &mut body,
                &crate::BggPublicKeyWire { matrix: body_one, reveal_plaintext: true },
                &scaled_a_prime,
            );
            let gadget = body.constant_matrix(
                self.public_key_type(),
                ConstantMatrix::Gadget { base: self.public_key.base.clone(), small: false },
            );
            let scaled_gadget = body.matrix_scale(&gadget, self.crt_scale_factors[crt].clone());
            let input_term = self.public_key.matrix_mul(
                &mut body,
                &crate::BggPublicKeyWire { matrix: body_input, reveal_plaintext: true },
                &scaled_gadget,
            );
            let with_refresh = body.matrix_binary(
                MatrixBinaryOp::Add,
                &input_term.matrix,
                &body_refresh,
                self.public_key_type(),
            );
            let output = body.matrix_binary(
                MatrixBinaryOp::Subtract,
                &with_refresh,
                &one_term.matrix,
                self.public_key_type(),
            );
            body.value_output_wire("0_decoder_public_key", output.wire);
            let [family] = builder
                .parallel_loop(
                    body.finish(),
                    IntExpr::constant(self.slot_count),
                    "slot",
                    Vec::new(),
                    vec![
                        one.matrices.wire,
                        refreshed_input.matrices.wire,
                        a_prime.matrices.wire,
                        refresh_terms[crt].wire,
                    ],
                    vec![LoopInputMode::Zip; 4],
                    &[self.public_key_type()],
                )?
                .try_into()
                .expect("one decoder public-key output was declared");
            outputs.push(family);
        }
        Ok(outputs)
    }

    fn sample_decoder_preimages(
        &self,
        builder: &mut GraphBuilder,
        decoder_trapdoor: &TrapdoorWire,
        decoder_public_keys: &MatrixFamilyWire,
    ) -> Result<MatrixFamilyWire, NaiveBggNoiseRefreshError> {
        let mut body = GraphBuilder::new("noise-refresh-decoder-preimages", Vec::new());
        let body_target = body.input("0_target", self.public_key_type());
        let body_trapdoor = body.trapdoor_input(
            "1_trapdoor",
            self.decoder_public_type(),
            self.decoder_trapdoor_sigma.clone(),
            self.public_key.base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let preimage =
            body.preimage_sample(&body_trapdoor, &body_target, self.decoder_preimage_type());
        body.value_output_wire("0_preimage", preimage.wire);
        let [family] = builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(self.flat_decoder_count()),
                "decoder",
                Vec::new(),
                vec![decoder_public_keys.wire, decoder_trapdoor.wire],
                vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
                &[self.decoder_preimage_type()],
            )?
            .try_into()
            .expect("one decoder preimage output was declared");
        Ok(family)
    }

    fn online_level_vectors(
        &self,
        builder: &mut GraphBuilder,
        one: &NaiveBggEncodingVecWire,
        refreshed_input: &NaiveBggEncodingVecWire,
        a_prime: &NaiveBggPublicKeyVecWire,
        refresh_terms: &[MatrixFamilyWire],
        decoders: &[MatrixFamilyWire],
    ) -> Result<Vec<MatrixFamilyWire>, NaiveBggNoiseRefreshError> {
        let mut outputs = Vec::with_capacity(self.crt_depth());
        for crt in 0..self.crt_depth() {
            let mut body = GraphBuilder::new(format!("noise-refresh-online-crt-{crt}"), Vec::new());
            let body_one = body.input("0_one", self.vector_type());
            let body_input = body.input("1_refreshed_input", self.vector_type());
            let body_a_prime = body.input("2_a_prime", self.public_key_type());
            let body_refresh = body.input("3_refresh", self.vector_type());
            let body_decoder = body.input("4_decoder", self.vector_type());
            let scaled_a_prime =
                body.matrix_scale(&body_a_prime, self.crt_scale_factors[crt].clone());
            let a_prime_decomposed = body.gadget_decompose(
                &scaled_a_prime,
                self.public_key.base.clone(),
                self.decomposed_type(self.public_key_columns()),
            );
            let one_term = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &body_one,
                &a_prime_decomposed,
                self.vector_type(),
            );
            let gadget = body.constant_matrix(
                self.public_key_type(),
                ConstantMatrix::Gadget { base: self.public_key.base.clone(), small: false },
            );
            let scaled_gadget = body.matrix_scale(&gadget, self.crt_scale_factors[crt].clone());
            let gadget_decomposed = body.gadget_decompose(
                &scaled_gadget,
                self.public_key.base.clone(),
                self.decomposed_type(self.public_key_columns()),
            );
            let input_term = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &body_input,
                &gadget_decomposed,
                self.vector_type(),
            );
            let with_refresh = body.matrix_binary(
                MatrixBinaryOp::Add,
                &input_term,
                &body_refresh,
                self.vector_type(),
            );
            let without_one = body.matrix_binary(
                MatrixBinaryOp::Subtract,
                &with_refresh,
                &one_term,
                self.vector_type(),
            );
            let output = body.matrix_binary(
                MatrixBinaryOp::Subtract,
                &without_one,
                &body_decoder,
                self.vector_type(),
            );
            body.value_output_wire("0_level", output.wire);
            let [family] = builder
                .parallel_loop(
                    body.finish(),
                    IntExpr::constant(self.slot_count),
                    "slot",
                    Vec::new(),
                    vec![
                        one.vectors.wire,
                        refreshed_input.vectors.wire,
                        a_prime.matrices.wire,
                        refresh_terms[crt].wire,
                        decoders[crt].wire,
                    ],
                    vec![LoopInputMode::Zip; 5],
                    &[self.vector_type()],
                )?
                .try_into()
                .expect("one online level output was declared");
            outputs.push(family);
        }
        Ok(outputs)
    }

    fn recompose_levels(
        &self,
        builder: &mut GraphBuilder,
        levels: &[MatrixFamilyWire],
    ) -> Result<MatrixFamilyWire, NaiveBggNoiseRefreshError> {
        let wide_type = self.matrix_type(1, self.slot_count * self.public_key_columns());
        let wide_levels = levels
            .iter()
            .map(|level| {
                let slots = (0..self.slot_count)
                    .map(|slot| builder.family_get_static(level, IntExpr::constant(slot)))
                    .collect::<Vec<_>>();
                builder.concat(ConcatAxis::Columns, &slots, wide_type.clone())
            })
            .collect::<Vec<_>>();
        let wide_output = builder.crt_recompose(
            &wide_levels,
            self.crt_plaintext_moduli.clone(),
            self.reconstruction_coefficients.clone(),
        );
        let slots = (0..self.slot_count)
            .map(|slot| {
                builder.slice(
                    &wide_output,
                    None,
                    Some(mxx_ir_core::node::IndexRange {
                        start: slot * self.public_key_columns(),
                        end: (slot + 1) * self.public_key_columns(),
                    }),
                    self.vector_type(),
                )
            })
            .collect::<Vec<_>>();
        Ok(builder.family_pack(&slots)?)
    }

    fn flatten_crt_families(
        &self,
        builder: &mut GraphBuilder,
        families: &[MatrixFamilyWire],
    ) -> Result<MatrixFamilyWire, NaiveBggNoiseRefreshError> {
        let mut values = Vec::with_capacity(self.flat_decoder_count());
        for slot in 0..self.slot_count {
            for family in families {
                values.push(builder.family_get_static(family, IntExpr::constant(slot)));
            }
        }
        Ok(builder.family_pack(&values)?)
    }

    fn split_flat_family_by_crt(
        &self,
        builder: &mut GraphBuilder,
        family: &MatrixFamilyWire,
    ) -> Result<Vec<MatrixFamilyWire>, NaiveBggNoiseRefreshError> {
        (0..self.crt_depth())
            .map(|crt| {
                let values = (0..self.slot_count)
                    .map(|slot| {
                        builder.family_get_static(
                            family,
                            IntExpr::constant(slot * self.crt_depth() + crt),
                        )
                    })
                    .collect::<Vec<_>>();
                Ok(builder.family_pack(&values)?)
            })
            .collect()
    }

    fn validate_public_bundle(
        &self,
        value: &NaiveBggPublicKeyVecWire,
    ) -> Result<(), NaiveBggNoiseRefreshError> {
        self.validate_layout()?;
        if value.matrices.matrix_type != self.public_key_type() ||
            value.matrices.count != IntExpr::constant(self.slot_count)
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        Ok(())
    }

    fn validate_encoding_bundle(
        &self,
        value: &NaiveBggEncodingVecWire,
    ) -> Result<(), NaiveBggNoiseRefreshError> {
        if value.vectors.matrix_type != self.vector_type() ||
            value.pubkeys.matrix_type != self.public_key_type() ||
            value.vectors.count != IntExpr::constant(self.slot_count) ||
            value.pubkeys.count != IntExpr::constant(self.slot_count) ||
            value.plaintexts.as_ref().is_some_and(|plaintexts| {
                plaintexts.matrix_type != self.scalar_type() ||
                    plaintexts.count != IntExpr::constant(self.slot_count)
            })
        {
            return Err(NaiveBggNoiseRefreshError::FamilyLayoutMismatch);
        }
        Ok(())
    }

    fn validate_decoder_trapdoor(
        &self,
        value: &TrapdoorWire,
    ) -> Result<(), NaiveBggNoiseRefreshError> {
        if value.public.matrix_type != self.decoder_public_type() ||
            value.sigma != self.decoder_trapdoor_sigma ||
            value.gadget_base != self.public_key.base ||
            value.digit_count != IntExpr::constant(self.digit_count)
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
        self.matrix_type(self.secret_size, self.public_key_columns())
    }

    fn vector_type(&self) -> MatrixType {
        self.matrix_type(1, self.public_key_columns())
    }

    fn scalar_type(&self) -> MatrixType {
        self.matrix_type(1, 1)
    }

    fn decoder_public_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.decoder_public_columns)
    }

    fn decoder_state_type(&self) -> MatrixType {
        self.matrix_type(1, self.decoder_public_columns)
    }

    fn decoder_preimage_type(&self) -> MatrixType {
        self.matrix_type(self.decoder_public_columns, self.public_key_columns())
    }

    fn decomposed_type(&self, columns: usize) -> MatrixType {
        self.matrix_type(self.public_key_columns(), columns)
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

fn sum_matrices(builder: &mut GraphBuilder, values: Vec<MatrixWire>) -> MatrixWire {
    values
        .into_iter()
        .reduce(|left, right| {
            builder.matrix_binary(MatrixBinaryOp::Add, &left, &right, left.matrix_type.clone())
        })
        .expect("validated noise-refresh layouts always contain at least one term")
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

    fn compiler() -> NaiveBggNoiseRefreshCompiler {
        let modulus = IntExpr::constant(BigInt::from(65537));
        let ring_dimension = IntExpr::constant(2);
        let public_key = BggPublicKeyCompiler {
            base: IntExpr::constant(4),
            decomposed_type: MatrixType {
                modulus: modulus.clone(),
                ring_dimension: ring_dimension.clone(),
                rows: IntExpr::constant(18),
                columns: IntExpr::constant(18),
            },
        };
        NaiveBggNoiseRefreshCompiler {
            public_key,
            modulus,
            ring_dimension,
            secret_size: 2,
            slot_count: 2,
            digit_count: 9,
            crt_scale_factors: vec![IntExpr::constant(7), IntExpr::constant(11)],
            crt_plaintext_moduli: vec![IntExpr::constant(17), IntExpr::constant(19)],
            reconstruction_coefficients: vec![IntExpr::constant(3), IntExpr::constant(5)],
            decoder_public_columns: 22,
            decoder_trapdoor_sigma: RealExpr::from_f64_exact(5.0).expect("finite sigma"),
        }
    }

    fn public_bundle(
        builder: &mut GraphBuilder,
        compiler: &NaiveBggNoiseRefreshCompiler,
        name: &str,
    ) -> NaiveBggPublicKeyVecWire {
        NaiveBggPublicKeyVecWire {
            matrices: builder.family_input(
                name,
                compiler.public_key_type(),
                IntExpr::constant(compiler.slot_count),
            ),
            reveal_plaintext: true,
        }
    }

    fn encoding_bundle(
        builder: &mut GraphBuilder,
        compiler: &NaiveBggNoiseRefreshCompiler,
        name: &str,
    ) -> NaiveBggEncodingVecWire {
        NaiveBggEncodingVecWire {
            vectors: builder.family_input(
                format!("{name}_vectors"),
                compiler.vector_type(),
                IntExpr::constant(compiler.slot_count),
            ),
            pubkeys: builder.family_input(
                format!("{name}_public_keys"),
                compiler.public_key_type(),
                IntExpr::constant(compiler.slot_count),
            ),
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        }
    }

    #[test]
    fn preprocessing_and_online_graphs_validate() {
        let compiler = compiler();
        let decoded_count = compiler.slot_count * compiler.crt_depth() * compiler.digit_count;

        let mut preprocessing = GraphBuilder::new("noise-refresh-preprocessing", Vec::new());
        let hash_key = preprocessing.bytes_input("hash_key", 32);
        let one = public_bundle(&mut preprocessing, &compiler, "one");
        let refreshed = public_bundle(&mut preprocessing, &compiler, "refreshed");
        let decoded = (0..decoded_count)
            .map(|index| public_bundle(&mut preprocessing, &compiler, &format!("decoded_{index}")))
            .collect::<Vec<_>>();
        let decoder_trapdoor = preprocessing.trapdoor_sample(
            compiler.decoder_public_type(),
            compiler.decoder_trapdoor_sigma.clone(),
            compiler.public_key.base.clone(),
            IntExpr::constant(compiler.digit_count),
        );
        let wires = compiler
            .build_preprocessing(
                &mut preprocessing,
                hash_key,
                b"test-refresh",
                &one,
                &refreshed,
                &decoded,
                &decoder_trapdoor,
            )
            .expect("preprocessing graph");
        compiler.export_preprocessing(&mut preprocessing, &wires);
        validate(&preprocessing.finish(), &ParamEnv::default()).expect("valid preprocessing");

        let mut online = GraphBuilder::new("noise-refresh-online", Vec::new());
        let one = encoding_bundle(&mut online, &compiler, "one");
        let refreshed = encoding_bundle(&mut online, &compiler, "refreshed");
        let decoded = (0..decoded_count)
            .map(|index| encoding_bundle(&mut online, &compiler, &format!("decoded_{index}")))
            .collect::<Vec<_>>();
        let artifacts = NaiveBggNoiseRefreshArtifactWires {
            a_prime: public_bundle(&mut online, &compiler, "a_prime"),
            decoder_preimages: online.family_input(
                "decoder_preimages",
                compiler.decoder_preimage_type(),
                IntExpr::constant(compiler.flat_decoder_count()),
            ),
        };
        let decoder_state = online.input("decoder_state", compiler.decoder_state_type());
        let projected = compiler
            .project_decoder_preimages(&mut online, &decoder_state, &artifacts.decoder_preimages)
            .expect("project decoders");
        let output = compiler
            .build_online(&mut online, &one, &refreshed, &decoded, &artifacts, &projected)
            .expect("online graph");
        online.output_family_wire(
            "refreshed_output",
            &output.vectors,
            ArtifactConfidentiality::Public,
        );
        validate(&online.finish(), &ParamEnv::default()).expect("valid online graph");
    }

    #[test]
    fn online_execution_matches_explicit_zero_refresh_oracle() {
        let parameters = DCRTPolyParams::new(4, 2, 10, 5);
        let q = parameters.modulus().as_ref().clone();
        let (plaintext_moduli, _, depth) = parameters.to_crt();
        let reconstruction_coefficients = parameters.reconst_coeffs();
        let digit_count = parameters.modulus_digits();
        let matrix_type = |rows, columns| MatrixType {
            modulus: IntExpr::constant(BigInt::from(q.clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        };
        let compiler = NaiveBggNoiseRefreshCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(1u64 << parameters.base_bits()),
                decomposed_type: matrix_type(digit_count, digit_count),
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
                .map(|coefficient| IntExpr::constant(coefficient.clone()))
                .collect(),
            decoder_public_columns: digit_count + 2,
            decoder_trapdoor_sigma: RealExpr::from_f64_exact(5.0).unwrap(),
        };
        assert_eq!(depth, compiler.crt_depth());

        let mut builder = GraphBuilder::new("noise-refresh-online-runtime", Vec::new());
        let one = encoding_bundle(&mut builder, &compiler, "one");
        let refreshed = encoding_bundle(&mut builder, &compiler, "refreshed");
        let decoded = encoding_bundle(&mut builder, &compiler, "decoded");
        let decoded_material =
            vec![decoded; compiler.slot_count * compiler.crt_depth() * compiler.digit_count];
        let artifacts = NaiveBggNoiseRefreshArtifactWires {
            a_prime: public_bundle(&mut builder, &compiler, "a_prime"),
            decoder_preimages: builder.family_input(
                "unused_preimages",
                compiler.decoder_preimage_type(),
                IntExpr::constant(compiler.flat_decoder_count()),
            ),
        };
        let projected = builder.family_input(
            "projected_decoders",
            compiler.vector_type(),
            IntExpr::constant(compiler.flat_decoder_count()),
        );
        let output = compiler
            .build_online(&mut builder, &one, &refreshed, &decoded_material, &artifacts, &projected)
            .unwrap();
        for slot in 0..compiler.slot_count {
            let value = builder.family_get_static(&output.vectors, IntExpr::constant(slot));
            builder.output(format!("slot_{slot}"), &value, ArtifactConfidentiality::Public);
        }
        let validated = validate(&builder.finish(), &ParamEnv::default()).unwrap();

        let zero_vector = DCRTPolyMatrix::zero(&parameters, 1, digit_count);
        let zero_public = DCRTPolyMatrix::zero(&parameters, 1, digit_count);
        let zero_preimage =
            DCRTPolyMatrix::zero(&parameters, compiler.decoder_public_columns, digit_count);
        let slot_values = (0..compiler.slot_count)
            .map(|slot| {
                DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    (0..digit_count)
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
        let family = |value: &DCRTPolyMatrix, count: usize| {
            RuntimeValue::IndexedFamily(
                (0..count).map(|_| RuntimeValue::matrix(value.clone())).collect(),
            )
        };
        let projected_values = slot_values
            .iter()
            .flat_map(|source| {
                plaintext_moduli.iter().map(|plaintext_modulus| {
                    let scale = &q / BigUint::from(*plaintext_modulus);
                    source.clone() * DCRTPoly::from_biguint_to_constant(&parameters, &q - scale)
                })
            })
            .collect::<Vec<_>>();
        let inputs = BTreeMap::from([
            ("one_vectors".to_owned(), family(&zero_vector, compiler.slot_count)),
            ("one_public_keys".to_owned(), family(&zero_public, compiler.slot_count)),
            ("refreshed_vectors".to_owned(), family(&zero_vector, compiler.slot_count)),
            ("refreshed_public_keys".to_owned(), family(&zero_public, compiler.slot_count)),
            ("decoded_vectors".to_owned(), family(&zero_vector, compiler.slot_count)),
            ("decoded_public_keys".to_owned(), family(&zero_public, compiler.slot_count)),
            ("a_prime".to_owned(), family(&zero_public, compiler.slot_count)),
            (
                "projected_decoders".to_owned(),
                RuntimeValue::IndexedFamily(
                    projected_values.into_iter().map(RuntimeValue::matrix).collect(),
                ),
            ),
            ("unused_preimages".to_owned(), family(&zero_preimage, compiler.flat_decoder_count())),
        ]);
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result =
            execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh).unwrap();
        let reconstruction_sum = reconstruction_coefficients.iter().sum::<BigUint>() % &q;
        for (slot, source) in slot_values.iter().enumerate() {
            let expected = source.clone() *
                DCRTPoly::from_biguint_to_constant(&parameters, reconstruction_sum.clone());
            let RuntimeValue::Matrix(actual) = &result.outputs[&format!("slot_{slot}")] else {
                panic!("slot output must be a matrix")
            };
            assert_eq!(actual.as_ref(), &expected);
        }
    }
}
