//! One exact Diamond iO PRF round over declarative BGG+ families.
//!
//! The surrounding compiler supplies Ring-GSW Goldreich outputs and the
//! branch-specific decoded refresh material. This module owns the Diamond
//! equations that connect those generic gadgets: branch masking, common-key
//! rebasing through the final injector state, and noise refresh.

use super::{
    DiamondIoArtifactNames, DiamondIoConfig, DiamondIoConfigError, DiamondIoFunction,
    final_circuit::{build_final_function_decrypt_circuit, build_final_mask_decrypt_circuit},
};
use mxx_bgg::{
    BggPublicKeyCompiler, BggPublicKeySampler, BggPublicKeyWire, BggSampleError, BggSamplerLayout,
    CircuitCompileError, LweLookupCompileError, MaskedHighBitDecoderCompiler,
    MaskedHighBitDecoderError, MaskedHighBitDecoderOutputs, NaiveBggEncodingVecWire,
    NaiveBggNoiseRefreshArtifactWires, NaiveBggNoiseRefreshCompiler, NaiveBggNoiseRefreshError,
    NaiveBggPublicKeyVecWire, NaiveBggVecCompiler, NaiveEncodingSlotOperations,
    NaiveLweLookupEncodingLowering, NaiveLweLookupPreprocessingEntry,
    NaiveLweLookupPreprocessingLowering, NaivePublicKeySlotOperations, NaiveVecCompileError,
    PolyCircuitCompiler, bind_naive_lwe_lookup_invocations,
};
use mxx_dsl::{
    BoundedMetadata, BuiltGraph, Bytes, DslContext, DslError, Family, HashTag, Int, Mat, Parallel,
    Ring, Trapdoor, VirtualMat,
};
use mxx_gadgets::{
    Poly,
    circuit::PolyCircuit,
    circuit_gadgets::{arith::NestedRnsPoly, fhe::ring_gsw_nested_rns::NestedRnsRingGswContext},
    input_injector::{DiamondInputConfigError, DiamondInputInjector, DiamondInputPreprocessError},
    noise_refresh::material::build_noise_refresh_material_circuit,
};
use mxx_ir_core::{
    artifact::{ArtifactConfidentiality, ProductionId},
    node::ConcatAxis,
};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{PolyParams, dcrt::poly::DCRTPoly},
};
use std::sync::Arc;
use thiserror::Error;

pub const HASH_KEY_INPUT: &str = "diamond-io/hash-key";
pub const NOISE_REFRESH_HASH_KEY_INPUT: &str = "diamond-io/noise-refresh-hash-key";
pub const PRIVATE_K_INPUT: &str = "diamond-io/private-k";
pub const NATIVE_SEED_INPUT_PREFIX: &str = "diamond-io/native-seed";
pub const PUBLIC_INPUT_DIGIT_PREFIX: &str = "diamond-io/input-digit";
pub const OUTPUT_PREFIX: &str = "diamond-io/output";

pub fn output_name(index: usize) -> String {
    format!("{OUTPUT_PREFIX}-{index}")
}

pub trait DiamondIoPoly: Poly {
    type Matrix: PolyMatrix<P = Self>;
}

impl DiamondIoPoly for DCRTPoly {
    type Matrix = DCRTPolyMatrix;
}

pub struct DiamondIoPreprocessingGraph {
    pub graph: BuiltGraph,
}

pub struct DiamondIoEvaluationGraph {
    pub graph: BuiltGraph,
}

#[derive(Clone)]
pub struct DiamondIoCompiler<P: DiamondIoPoly + 'static> {
    pub config: DiamondIoConfig,
    pub ring_gsw: Arc<NestedRnsRingGswContext<P>>,
}

#[derive(Debug, Error)]
pub enum DiamondIoCompileError {
    #[error(transparent)]
    Config(#[from] DiamondIoConfigError),
    #[error(transparent)]
    InputConfig(#[from] DiamondInputConfigError),
    #[error(transparent)]
    Input(#[from] DiamondInputPreprocessError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    Sample(#[from] BggSampleError),
    #[error(transparent)]
    Circuit(#[from] CircuitCompileError),
    #[error(transparent)]
    Round(#[from] DiamondIoRoundError),
    #[error(transparent)]
    Decoder(#[from] MaskedHighBitDecoderError),
    #[error(transparent)]
    Naive(#[from] NaiveVecCompileError),
    #[error(transparent)]
    Lookup(#[from] LweLookupCompileError),
    #[error(transparent)]
    NativeSeed(#[from] super::DiamondIoNativeSeedError),
    #[error("Diamond iO circuit output layout is inconsistent with its Ring-GSW context")]
    CircuitOutputLayout,
}

struct LookupPreprocessing {
    hash_key: Bytes,
    trapdoors: Vec<Trapdoor>,
    entries: Vec<NaiveLweLookupPreprocessingEntry>,
    next_circuit: usize,
}

struct LookupEvaluation {
    production: ProductionId,
    c_b_by_slot: Family<Mat>,
    next_circuit: usize,
}

impl<P: DiamondIoPoly + 'static> DiamondIoCompiler<P> {
    fn compile_public_circuit(
        &self,
        compiler: &PolyCircuitCompiler,
        circuit: &PolyCircuit<P>,
        one: NaiveBggPublicKeyVecWire,
        inputs: impl IntoIterator<Item = NaiveBggPublicKeyVecWire>,
        lookup: &mut LookupPreprocessing,
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, DiamondIoCompileError> {
        let prefix = vec![lookup.next_circuit];
        lookup.next_circuit += 1;
        let mut lowering = NaiveLweLookupPreprocessingLowering::new(
            self.ring_gsw.params.clone(),
            lookup.hash_key.clone(),
            lookup.trapdoors.clone(),
            self.config.gadget_base.clone().into(),
            self.config.digit_count.into(),
            prefix,
        )?;
        let mut slots = NaivePublicKeySlotOperations;
        let outputs = compiler.compile_naive_public_keys_with_lowerings(
            circuit,
            one,
            inputs,
            &mut lowering,
            &mut slots,
        )?;
        lookup.entries.extend(lowering.into_entries());
        Ok(outputs)
    }

    fn compile_encoding_circuit(
        &self,
        compiler: &PolyCircuitCompiler,
        circuit: &PolyCircuit<P>,
        one: NaiveBggEncodingVecWire,
        inputs: impl IntoIterator<Item = NaiveBggEncodingVecWire>,
        lookup: &mut LookupEvaluation,
    ) -> Result<Vec<NaiveBggEncodingVecWire>, DiamondIoCompileError> {
        let prefix = [lookup.next_circuit];
        lookup.next_circuit += 1;
        let public_key_type = one.pubkeys.element_type().clone();
        let invocations = bind_naive_lwe_lookup_invocations(
            &self.ring_gsw.params,
            circuit,
            lookup.production.clone(),
            public_key_type,
            self.config.gadget_base.clone().into(),
            self.config.digit_count.into(),
            self.config.ring_dimension,
            &prefix,
        )?;
        let mut lowering =
            NaiveLweLookupEncodingLowering::new(invocations, lookup.c_b_by_slot.clone())?;
        let mut slots = NaiveEncodingSlotOperations;
        Ok(compiler.compile_naive_encodings_with_lowerings(
            circuit,
            one,
            inputs,
            &mut lowering,
            &mut slots,
        )?)
    }

    pub fn new(
        config: DiamondIoConfig,
        ring_gsw: Arc<NestedRnsRingGswContext<P>>,
    ) -> Result<Self, DiamondIoCompileError> {
        let compiler = Self { config, ring_gsw };
        compiler.validate_ring_gsw_layout()?;
        Ok(compiler)
    }

    pub fn build_preprocessing(
        &self,
        function: &DiamondIoFunction,
    ) -> Result<DiamondIoPreprocessingGraph, DiamondIoCompileError> {
        self.validate(function)?;
        let ring = self.ring();
        let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
        let noise_refresh_hash_key = ring.bytes_input(NOISE_REFRESH_HASH_KEY_INPUT, 32);
        let private_k = ring.input(PRIVATE_K_INPUT, (1, 1));
        let private_k = private_k.clone().assume(VirtualMat::bounded(
            "diamond-io/private-k",
            private_k.matrix_type().clone(),
            BoundedMetadata::conservative(1),
        ))?;
        let injector = DiamondInputInjector::new(self.config.input_config())?;
        let injection = injector.preprocess(private_k)?;
        let lookup_trapdoor = ring.sample_trapdoor(
            1,
            self.config.trapdoor_sigma.clone(),
            self.config.gadget_base.clone(),
            self.config.digit_count,
        );
        let lookup_base = lookup_trapdoor.public_matrix();
        let lookup_base_keys = NaiveBggPublicKeyVecWire {
            matrices: Parallel::range(self.config.ring_dimension)
                .map(move |_| lookup_base.clone())?,
            reveal_plaintext: false,
        };
        let lookup_base_projection = self.projection_preimages(
            injection.final_trapdoors[0].clone(),
            &lookup_base_keys,
            false,
            false,
        )?;
        let mut lookups = LookupPreprocessing {
            hash_key: hash_key.clone(),
            trapdoors: vec![lookup_trapdoor; self.config.ring_dimension],
            entries: Vec::new(),
            next_circuit: 0,
        };
        let (one, k, input_bits) = self.sample_public_keys(hash_key.clone())?;
        let selectors = self.digit_public_keys(&input_bits)?;
        let wire_count = self.ring_gsw.flattened_ciphertext_input_count();
        let max_p_modulus = self
            .ring_gsw
            .nested_rns
            .p_moduli
            .iter()
            .copied()
            .max()
            .ok_or(DiamondIoConfigError::RingGswLayout)?;
        let seed_inputs = super::declare_native_seed_inputs(
            &ring,
            NATIVE_SEED_INPUT_PREFIX,
            self.config.seed_bits,
            wire_count,
            self.config.ring_dimension,
            self.config.native_ring_gsw_ciphertext_error_norm(max_p_modulus)?,
        )?;
        let arithmetic = self.arithmetic();
        let mut seed = seed_inputs
            .iter()
            .flat_map(|seed| seed.scalar_families.iter())
            .map(|scalars| arithmetic.large_scalar_mul_public_key_families(&one, scalars.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let circuit = PolyCircuitCompiler { public_key: self.public_key_compiler() };
        let round = DiamondIoRoundCompiler { config: self.config.clone() };
        let mut rounds = Vec::with_capacity(self.config.round_count());
        for round_index in 0..self.config.round_count() {
            let branch_outputs = self.compile_round_public_branches(
                &circuit,
                &one,
                &seed,
                round_index,
                wire_count,
                &mut lookups,
            )?;
            let decoded = self.compile_public_refresh_material(
                &circuit,
                &one,
                &k,
                &branch_outputs,
                round_index,
                &mut lookups,
            )?;
            let preprocessed = round.preprocess_round(
                round_index,
                noise_refresh_hash_key.clone(),
                &one,
                &selectors[round_index],
                &branch_outputs,
                &decoded,
                injection.final_trapdoors[0].clone(),
            )?;
            seed = preprocessed
                .refresh
                .first()
                .ok_or(DiamondIoCompileError::CircuitOutputLayout)?
                .iter()
                .map(|refresh| refresh.a_prime.clone())
                .collect();
            rounds.push(preprocessed);
        }
        let final_outputs =
            self.compile_final_public_outputs(&circuit, &one, &k, &seed, function, &mut lookups)?;
        let decoder = self.decoder();
        let decoder_preimages = final_outputs
            .iter()
            .map(|output| {
                let combined = arithmetic.add_public_keys(
                    &output.function_secret_dependent,
                    &output.mask_secret_dependent,
                )?;
                Ok(decoder
                    .build_preprocessing(
                        injection.final_trapdoors[0].clone(),
                        combined.matrices,
                        self.config.ring_dimension,
                    )?
                    .preimages)
            })
            .collect::<Result<Vec<_>, DiamondIoCompileError>>()?;

        let mut context = DslContext::new("diamond-io-preprocessing")
            .public_output(DiamondIoArtifactNames::INJECTOR_INITIAL_STATE, injection.p)?
            .public_bytes_output(DiamondIoArtifactNames::HASH_KEY, hash_key)?;
        for (level, branches) in injection.transitions.into_iter().enumerate() {
            for (digit, states) in branches.into_iter().enumerate() {
                for (state, transition) in states.into_iter().enumerate() {
                    context = context.public_output(
                        DiamondIoArtifactNames::injector_transition(level + 1, digit, state),
                        transition,
                    )?;
                }
            }
        }
        context = self.export_public_key(
            context,
            DiamondIoArtifactNames::SCALAR_ONE_PUBLIC_KEY,
            one.clone(),
        )?;
        context = self.export_public_key(
            context,
            DiamondIoArtifactNames::SCALAR_K_PUBLIC_KEY,
            k.clone(),
        )?;
        for (bit, key) in input_bits.iter().cloned().enumerate() {
            context = self.export_public_key(
                context,
                DiamondIoArtifactNames::scalar_input_public_key(bit),
                key,
            )?;
        }
        context = self.export_projection_artifacts(
            context,
            &injection.final_trapdoors,
            &one,
            &k,
            &input_bits,
        )?;
        context = context.public_family_output(
            DiamondIoArtifactNames::LOOKUP_BASE_PROJECTION,
            lookup_base_projection,
        )?;
        for (seed_index, inputs) in seed_inputs.into_iter().enumerate() {
            for (wire, family) in inputs.scalar_families.into_iter().enumerate() {
                context = context.public_family_output(
                    DiamondIoArtifactNames::native_seed_bindings(seed_index, wire),
                    family,
                )?;
            }
        }
        for (round_index, round) in rounds.into_iter().enumerate() {
            for (wire, common) in round.common_public_keys.into_iter().enumerate() {
                context = context.public_family_output(
                    DiamondIoArtifactNames::round_common_public_key(round_index, wire),
                    common.matrices,
                )?;
                context = context.public_family_output(
                    DiamondIoArtifactNames::round_refresh_a_prime(round_index, wire),
                    round.refresh[0][wire].a_prime.matrices.clone(),
                )?;
                for branch in 0..self.config.branch_count()? {
                    context = context.public_family_output(
                        DiamondIoArtifactNames::round_rebase_preimages(round_index, branch, wire),
                        round.branch_rebase_preimages[branch][wire].clone(),
                    )?;
                    context = context.public_family_output(
                        DiamondIoArtifactNames::round_refresh_decoder_preimages(
                            round_index,
                            branch,
                            wire,
                        ),
                        round.refresh[branch][wire].decoder_preimages.clone(),
                    )?;
                }
            }
        }
        for (output, (wires, preimages)) in
            final_outputs.into_iter().zip(decoder_preimages).enumerate()
        {
            context = context
                .public_family_output(
                    DiamondIoArtifactNames::final_function_secret_dependent_public_key(output),
                    wires.function_secret_dependent.matrices,
                )?
                .public_family_output(
                    DiamondIoArtifactNames::final_function_public_bottom(output),
                    wires.function_bottom.matrices,
                )?
                .public_family_output(
                    DiamondIoArtifactNames::final_mask_secret_dependent_public_key(output),
                    wires.mask_secret_dependent.matrices,
                )?
                .public_family_output(
                    DiamondIoArtifactNames::final_mask_public_bottom(output),
                    wires.mask_bottom.matrices,
                )?
                .public_family_output(
                    DiamondIoArtifactNames::final_decoder_preimages(output),
                    preimages,
                )?;
        }
        for entry in lookups.entries {
            context = entry.export(context)?;
        }
        Ok(DiamondIoPreprocessingGraph { graph: context.build()? })
    }

    pub fn build_evaluation(
        &self,
        function: &DiamondIoFunction,
        production: ProductionId,
    ) -> Result<DiamondIoEvaluationGraph, DiamondIoCompileError> {
        self.validate(function)?;
        let ring = self.ring();
        let hash_key = ring.bytes_artifact_input(
            production.clone(),
            DiamondIoArtifactNames::HASH_KEY,
            32,
            ArtifactConfidentiality::Public,
        );
        let injector = DiamondInputInjector::new(self.config.input_config())?;
        let initial = ring.artifact_input(
            production.clone(),
            DiamondIoArtifactNames::INJECTOR_INITIAL_STATE,
            (1, self.config.input_config().state_columns()?),
            ArtifactConfidentiality::Public,
        );
        let digits = (0..self.config.input_count)
            .map(|round| {
                ring.input(format!("{PUBLIC_INPUT_DIGIT_PREFIX}-{round}"), (1, 1))
                    .extract_coefficient(0)
            })
            .collect::<Vec<_>>();
        let transitions = self.import_transitions(production.clone());
        let states = injector.evaluate(initial, &digits, &transitions)?.states;
        let lookup_base_preimages = ring.family_artifact_input(
            production.clone(),
            DiamondIoArtifactNames::LOOKUP_BASE_PROJECTION,
            self.config.ring_dimension,
            (self.config.input_config().state_columns()?, self.config.digit_count + 2),
            ArtifactConfidentiality::Public,
        );
        let root_state = states[0].clone();
        let c_b_by_slot =
            lookup_base_preimages.parallel_map(move |_, preimage| root_state.clone() * preimage)?;
        let mut lookups =
            LookupEvaluation { production: production.clone(), c_b_by_slot, next_circuit: 0 };
        let one_public = self.import_public_key(
            production.clone(),
            DiamondIoArtifactNames::SCALAR_ONE_PUBLIC_KEY,
            true,
        );
        let k_public = self.import_public_key(
            production.clone(),
            DiamondIoArtifactNames::SCALAR_K_PUBLIC_KEY,
            false,
        );
        let one = self.project_encoding(
            &states[0],
            production.clone(),
            DiamondIoArtifactNames::ONE_PROJECTION,
            one_public,
            Some(ring.identity(1)),
        )?;
        let k = self.project_encoding(
            &states[0],
            production.clone(),
            DiamondIoArtifactNames::K_PROJECTION,
            k_public,
            None,
        )?;
        let mut input_bits = Vec::with_capacity(self.config.input_bits()?);
        for bit in 0..self.config.input_bits()? {
            let digit = bit / self.config.batch_bits;
            let bit_in_digit = bit % self.config.batch_bits;
            let state = self.config.input_config().bit_state_index(digit, bit_in_digit)?;
            let plaintext = digits[digit]
                .clone()
                .bit(bit_in_digit)
                .to_int()
                .select(vec![ring.zero((1, 1)), ring.identity(1)])?;
            input_bits.push(self.project_encoding(
                &states[state],
                production.clone(),
                DiamondIoArtifactNames::input_projection(bit),
                self.import_public_key(
                    production.clone(),
                    DiamondIoArtifactNames::scalar_input_public_key(bit),
                    true,
                ),
                Some(plaintext),
            )?);
        }
        let selectors = self.digit_encodings(&input_bits)?;
        let wire_count = self.ring_gsw.flattened_ciphertext_input_count();
        let arithmetic = self.arithmetic();
        let mut seed = (0..self.config.seed_bits)
            .flat_map(|seed| (0..wire_count).map(move |wire| (seed, wire)))
            .map(|(seed, wire)| {
                let scalars = ring.family_artifact_input(
                    production.clone(),
                    DiamondIoArtifactNames::native_seed_bindings(seed, wire),
                    self.config.ring_dimension,
                    (1, 1),
                    ArtifactConfidentiality::Public,
                );
                arithmetic.large_scalar_mul_encoding_families(&one, scalars)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let circuit = PolyCircuitCompiler { public_key: self.public_key_compiler() };
        let round_compiler = DiamondIoRoundCompiler { config: self.config.clone() };
        for round in 0..self.config.round_count() {
            let branches = self.compile_round_encoding_branches(
                &circuit,
                &one,
                &seed,
                round,
                wire_count,
                &mut lookups,
            )?;
            let decoded = self.compile_encoding_refresh_material(
                &circuit,
                &one,
                &k,
                &branches,
                round,
                &mut lookups,
            )?;
            let artifacts = self.import_round_artifacts(
                production.clone(),
                round,
                self.config.seed_bits * wire_count,
            );
            let evaluated = round_compiler.evaluate_round(
                round,
                hash_key.clone(),
                digits[round].clone(),
                states[0].clone(),
                &one,
                &selectors[round],
                &branches,
                &decoded,
                &artifacts,
            )?;
            if evaluated.rebased.len() != self.config.seed_bits * wire_count {
                return Err(DiamondIoCompileError::CircuitOutputLayout);
            }
            seed = evaluated.next_seed;
        }
        let final_outputs =
            self.compile_final_encoding_outputs(&circuit, &one, &k, &seed, function, &mut lookups)?;
        let decoder = self.decoder();
        let mut context = DslContext::new("diamond-io-evaluation");
        for (output, wires) in final_outputs.into_iter().enumerate() {
            let combined = arithmetic
                .add_encodings(&wires.function_secret_dependent, &wires.mask_secret_dependent)?;
            let function_bottom = wires
                .function_bottom
                .plaintexts
                .ok_or(DiamondIoCompileError::CircuitOutputLayout)?;
            let mask_bottom =
                wires.mask_bottom.plaintexts.ok_or(DiamondIoCompileError::CircuitOutputLayout)?;
            let bottoms =
                function_bottom.parallel_zip(mask_bottom, |_, left, right| left + right)?;
            let preimages = ring.family_artifact_input(
                production.clone(),
                DiamondIoArtifactNames::final_decoder_preimages(output),
                self.config.ring_dimension,
                (self.config.input_config().state_columns()?, 1),
                ArtifactConfidentiality::Public,
            );
            let MaskedHighBitDecoderOutputs::Booleans(decoded) = decoder.build_online(
                states[0].clone(),
                preimages,
                combined.vectors,
                bottoms,
                2.into(),
                true,
                self.config.ring_dimension,
            )?
            else {
                return Err(DiamondIoCompileError::CircuitOutputLayout);
            };
            context = context.bool_family_output(output_name(output), decoded[0].clone())?;
        }
        Ok(DiamondIoEvaluationGraph { graph: context.build()? })
    }

    fn validate(&self, function: &DiamondIoFunction) -> Result<(), DiamondIoCompileError> {
        self.config.validate(function)?;
        self.validate_ring_gsw_layout()?;
        Ok(())
    }

    fn validate_ring_gsw_layout(&self) -> Result<(), DiamondIoCompileError> {
        let parameters = &self.ring_gsw.params;
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        let (_, _, full_crt_depth) = parameters.to_crt();
        let gadget_base =
            1u64.checked_shl(parameters.base_bits()).ok_or(DiamondIoConfigError::RingGswLayout)?;
        if self.config.modulus != num_bigint::BigInt::from(modulus.as_ref().clone()) ||
            self.config.ring_dimension != parameters.ring_dimension() as usize ||
            self.config.digit_count != parameters.modulus_digits() ||
            self.config.gadget_base != num_bigint::BigInt::from(gadget_base) ||
            self.ring_gsw.num_slots != self.config.ring_dimension ||
            self.ring_gsw.level_offset != 0 ||
            self.ring_gsw.active_levels != full_crt_depth ||
            self.config.ring_gsw_width != self.ring_gsw.width()
        {
            return Err(DiamondIoConfigError::RingGswLayout.into());
        }
        Ok(())
    }

    fn ring(&self) -> Ring {
        Ring::new(self.config.modulus.clone(), self.config.ring_dimension)
    }

    fn sampler_layout(&self) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: self.config.modulus.clone().into(),
            ring_dimension: self.config.ring_dimension.into(),
            secret_dimension: 1,
            digit_count: self.config.digit_count,
            gadget_base: self.config.gadget_base.clone().into(),
        }
    }

    fn public_key_compiler(&self) -> BggPublicKeyCompiler {
        BggPublicKeyCompiler {
            ring: self.ring(),
            base: self.config.gadget_base.clone().into(),
            digit_count: self.config.digit_count.into(),
        }
    }

    fn arithmetic(&self) -> NaiveBggVecCompiler {
        NaiveBggVecCompiler { public_key: self.public_key_compiler() }
    }

    fn decoder(&self) -> MaskedHighBitDecoderCompiler {
        MaskedHighBitDecoderCompiler {
            modulus: self.config.modulus.clone().into(),
            ring_dimension: self.config.ring_dimension.into(),
            secret_size: 1,
            digit_count: self.config.digit_count,
            gadget_base: self.config.gadget_base.clone().into(),
            trapdoor_sigma: self.config.trapdoor_sigma.clone(),
            coefficient_count: self.config.ring_dimension,
        }
    }

    fn sample_public_keys(
        &self,
        hash_key: Bytes,
    ) -> Result<
        (NaiveBggPublicKeyVecWire, NaiveBggPublicKeyVecWire, Vec<NaiveBggPublicKeyVecWire>),
        DiamondIoCompileError,
    > {
        let input_bits = self.config.input_bits()?;
        let mut reveal = Vec::with_capacity(input_bits + 1);
        reveal.push(false);
        reveal.extend(std::iter::repeat_n(true, input_bits));
        let mut scalar = BggPublicKeySampler { layout: self.sampler_layout() }.sample(
            hash_key,
            HashTag::from(self.config.bgg_tag.clone()),
            &reveal,
        );
        let one = scalar.remove(0);
        let k = scalar.remove(0);
        let duplicate = |key: BggPublicKeyWire| -> Result<_, DslError> {
            Ok(NaiveBggPublicKeyVecWire {
                matrices: Parallel::range(self.config.ring_dimension)
                    .map(move |_| key.matrix.clone())?,
                reveal_plaintext: key.reveal_plaintext,
            })
        };
        Ok((
            duplicate(one)?,
            duplicate(k)?,
            scalar.into_iter().map(duplicate).collect::<Result<_, _>>()?,
        ))
    }

    fn digit_public_keys(
        &self,
        bits: &[NaiveBggPublicKeyVecWire],
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, DiamondIoCompileError> {
        bits.chunks_exact(self.config.batch_bits)
            .map(|chunk| {
                let mut sum = self.small_scalar_public(&chunk[0], 1)?;
                for (bit, key) in chunk.iter().enumerate().skip(1) {
                    sum = self
                        .arithmetic()
                        .add_public_keys(&sum, &self.small_scalar_public(key, 1usize << bit)?)?;
                }
                Ok(sum)
            })
            .collect()
    }

    fn digit_encodings(
        &self,
        bits: &[NaiveBggEncodingVecWire],
    ) -> Result<Vec<NaiveBggEncodingVecWire>, DiamondIoCompileError> {
        bits.chunks_exact(self.config.batch_bits)
            .map(|chunk| {
                let mut sum = self.small_scalar_encoding(&chunk[0], 1)?;
                for (bit, key) in chunk.iter().enumerate().skip(1) {
                    sum = self
                        .arithmetic()
                        .add_encodings(&sum, &self.small_scalar_encoding(key, 1usize << bit)?)?;
                }
                Ok(sum)
            })
            .collect()
    }

    fn small_scalar_public(
        &self,
        input: &NaiveBggPublicKeyVecWire,
        scalar: usize,
    ) -> Result<NaiveBggPublicKeyVecWire, DiamondIoCompileError> {
        let compiler = self.public_key_compiler();
        let scalar = self.ring().polynomial([scalar.into()]);
        let reveal = input.reveal_plaintext;
        Ok(NaiveBggPublicKeyVecWire {
            matrices: input.matrices.clone().parallel_map(move |_, matrix| {
                compiler
                    .small_scalar_mul(
                        &BggPublicKeyWire { matrix, reveal_plaintext: reveal },
                        &scalar,
                    )
                    .matrix
            })?,
            reveal_plaintext: reveal,
        })
    }

    fn small_scalar_encoding(
        &self,
        input: &NaiveBggEncodingVecWire,
        scalar: usize,
    ) -> Result<NaiveBggEncodingVecWire, DiamondIoCompileError> {
        Ok(self
            .arithmetic()
            .small_scalar_mul_encodings(input, &self.ring().polynomial([scalar.into()]))?)
    }

    fn compile_round_public_branches(
        &self,
        compiler: &PolyCircuitCompiler,
        one: &NaiveBggPublicKeyVecWire,
        seed: &[NaiveBggPublicKeyVecWire],
        round: usize,
        wire_count: usize,
        lookups: &mut LookupPreprocessing,
    ) -> Result<Vec<Vec<NaiveBggPublicKeyVecWire>>, DiamondIoCompileError> {
        let conceptual = self.config.branch_count()? * self.config.seed_bits;
        (0..self.config.branch_count()?)
            .map(|branch| {
                let circuit = super::circuits::build_goldreich_full_domain_range_circuit(
                    self.ring_gsw.clone(),
                    self.config.seed_bits,
                    conceptual,
                    branch * self.config.seed_bits,
                    self.config.seed_bits,
                    super::circuits::goldreich_round_seed(
                        self.config.goldreich_graph_seed,
                        b"seed-refresh",
                        round,
                        None,
                    ),
                );
                let outputs = self.compile_public_circuit(
                    compiler,
                    &circuit,
                    one.clone(),
                    seed.to_vec(),
                    lookups,
                )?;
                if outputs.len() != self.config.seed_bits * wire_count {
                    return Err(DiamondIoCompileError::CircuitOutputLayout);
                }
                Ok(outputs)
            })
            .collect()
    }

    fn compile_round_encoding_branches(
        &self,
        compiler: &PolyCircuitCompiler,
        one: &NaiveBggEncodingVecWire,
        seed: &[NaiveBggEncodingVecWire],
        round: usize,
        wire_count: usize,
        lookups: &mut LookupEvaluation,
    ) -> Result<Vec<Vec<NaiveBggEncodingVecWire>>, DiamondIoCompileError> {
        let conceptual = self.config.branch_count()? * self.config.seed_bits;
        (0..self.config.branch_count()?)
            .map(|branch| {
                let circuit = super::circuits::build_goldreich_full_domain_range_circuit(
                    self.ring_gsw.clone(),
                    self.config.seed_bits,
                    conceptual,
                    branch * self.config.seed_bits,
                    self.config.seed_bits,
                    super::circuits::goldreich_round_seed(
                        self.config.goldreich_graph_seed,
                        b"seed-refresh",
                        round,
                        None,
                    ),
                );
                let outputs = self.compile_encoding_circuit(
                    compiler,
                    &circuit,
                    one.clone(),
                    seed.to_vec(),
                    lookups,
                )?;
                if outputs.len() != self.config.seed_bits * wire_count {
                    return Err(DiamondIoCompileError::CircuitOutputLayout);
                }
                Ok(outputs)
            })
            .collect()
    }

    fn material_circuit(
        &self,
        round: usize,
        branch: usize,
    ) -> mxx_gadgets::circuit::PolyCircuit<P> {
        build_noise_refresh_material_circuit::<P, NestedRnsPoly<P>, P::Matrix>(
            self.ring_gsw.clone(),
            self.config.seed_bits,
            self.config.noise_refresh_v_bits,
            super::circuits::goldreich_round_seed(
                self.config.goldreich_graph_seed,
                b"noise-refresh",
                round,
                Some(branch),
            ),
            self.config.noise_refresh_cbd_n,
            self.config.ring_dimension,
        )
    }

    fn compile_public_refresh_material(
        &self,
        compiler: &PolyCircuitCompiler,
        one: &NaiveBggPublicKeyVecWire,
        k: &NaiveBggPublicKeyVecWire,
        branches: &[Vec<NaiveBggPublicKeyVecWire>],
        round: usize,
        lookups: &mut LookupPreprocessing,
    ) -> Result<Vec<Vec<Vec<NaiveBggPublicKeyVecWire>>>, DiamondIoCompileError> {
        branches
            .iter()
            .enumerate()
            .map(|(branch, seed)| {
                let mut inputs = seed.clone();
                inputs.push(k.clone());
                let material_circuit = self.material_circuit(round, branch);
                let material = self.compile_public_circuit(
                    compiler,
                    &material_circuit,
                    one.clone(),
                    inputs,
                    lookups,
                )?;
                Ok((0..seed.len()).map(|_| material.clone()).collect())
            })
            .collect()
    }

    fn compile_encoding_refresh_material(
        &self,
        compiler: &PolyCircuitCompiler,
        one: &NaiveBggEncodingVecWire,
        k: &NaiveBggEncodingVecWire,
        branches: &[Vec<NaiveBggEncodingVecWire>],
        round: usize,
        lookups: &mut LookupEvaluation,
    ) -> Result<Vec<Vec<Vec<NaiveBggEncodingVecWire>>>, DiamondIoCompileError> {
        branches
            .iter()
            .enumerate()
            .map(|(branch, seed)| {
                let mut inputs = seed.clone();
                inputs.push(k.clone());
                let material_circuit = self.material_circuit(round, branch);
                let material = self.compile_encoding_circuit(
                    compiler,
                    &material_circuit,
                    one.clone(),
                    inputs,
                    lookups,
                )?;
                Ok((0..seed.len()).map(|_| material.clone()).collect())
            })
            .collect()
    }

    fn compile_final_public_outputs(
        &self,
        compiler: &PolyCircuitCompiler,
        one: &NaiveBggPublicKeyVecWire,
        k: &NaiveBggPublicKeyVecWire,
        seed: &[NaiveBggPublicKeyVecWire],
        function: &DiamondIoFunction,
        lookups: &mut LookupPreprocessing,
    ) -> Result<Vec<FinalWires<NaiveBggPublicKeyVecWire>>, DiamondIoCompileError> {
        let wire_count = self.ring_gsw.flattened_ciphertext_input_count();
        let [_, _, conceptual] = self.config.goldreich_stream_sizes(function)?;
        let mask_bits = function.output_bits() *
            self.config.ring_dimension *
            self.config.prf_mask_output_coeff_bits;
        let prg = super::circuits::build_goldreich_full_domain_range_circuit(
            self.ring_gsw.clone(),
            self.config.seed_bits,
            conceptual,
            0,
            conceptual,
            super::circuits::goldreich_round_seed(
                self.config.goldreich_graph_seed,
                b"final-function-mask",
                self.config.round_count(),
                None,
            ),
        );
        let outputs =
            self.compile_public_circuit(compiler, &prg, one.clone(), seed.to_vec(), lookups)?;
        let mask_wire_len = mask_bits * wire_count;
        if outputs.len() != conceptual * wire_count || mask_wire_len > outputs.len() {
            return Err(DiamondIoCompileError::CircuitOutputLayout);
        }
        let (mask_outputs, function_outputs) = outputs.split_at(mask_wire_len);
        let mask_per_output =
            self.config.ring_dimension * self.config.prf_mask_output_coeff_bits * wire_count;
        (0..function.output_bits())
            .map(|output| {
                let mut mask_inputs = vec![k.clone()];
                mask_inputs.extend_from_slice(
                    &mask_outputs[output * mask_per_output..(output + 1) * mask_per_output],
                );
                let mask_circuit = build_final_mask_decrypt_circuit::<P, P::Matrix>(
                    self.ring_gsw.clone(),
                    self.config.prf_mask_output_coeff_bits,
                );
                let mask = self.compile_public_circuit(
                    compiler,
                    &mask_circuit,
                    one.clone(),
                    mask_inputs,
                    lookups,
                )?;
                let mut function_inputs = vec![k.clone()];
                function_inputs.extend_from_slice(
                    &function_outputs[output * wire_count..(output + 1) * wire_count],
                );
                let function_circuit =
                    build_final_function_decrypt_circuit::<P, P::Matrix>(self.ring_gsw.clone());
                let function = self.compile_public_circuit(
                    compiler,
                    &function_circuit,
                    one.clone(),
                    function_inputs,
                    lookups,
                )?;
                if mask.len() != 2 || function.len() != 2 {
                    return Err(DiamondIoCompileError::CircuitOutputLayout);
                }
                Ok(FinalWires {
                    function_secret_dependent: function[0].clone(),
                    function_bottom: function[1].clone(),
                    mask_secret_dependent: mask[0].clone(),
                    mask_bottom: mask[1].clone(),
                })
            })
            .collect()
    }

    fn compile_final_encoding_outputs(
        &self,
        compiler: &PolyCircuitCompiler,
        one: &NaiveBggEncodingVecWire,
        k: &NaiveBggEncodingVecWire,
        seed: &[NaiveBggEncodingVecWire],
        function: &DiamondIoFunction,
        lookups: &mut LookupEvaluation,
    ) -> Result<Vec<FinalWires<NaiveBggEncodingVecWire>>, DiamondIoCompileError> {
        let wire_count = self.ring_gsw.flattened_ciphertext_input_count();
        let [_, _, conceptual] = self.config.goldreich_stream_sizes(function)?;
        let mask_bits = function.output_bits() *
            self.config.ring_dimension *
            self.config.prf_mask_output_coeff_bits;
        let prg = super::circuits::build_goldreich_full_domain_range_circuit(
            self.ring_gsw.clone(),
            self.config.seed_bits,
            conceptual,
            0,
            conceptual,
            super::circuits::goldreich_round_seed(
                self.config.goldreich_graph_seed,
                b"final-function-mask",
                self.config.round_count(),
                None,
            ),
        );
        let outputs =
            self.compile_encoding_circuit(compiler, &prg, one.clone(), seed.to_vec(), lookups)?;
        let mask_wire_len = mask_bits * wire_count;
        if outputs.len() != conceptual * wire_count || mask_wire_len > outputs.len() {
            return Err(DiamondIoCompileError::CircuitOutputLayout);
        }
        let (mask_outputs, function_outputs) = outputs.split_at(mask_wire_len);
        let mask_per_output =
            self.config.ring_dimension * self.config.prf_mask_output_coeff_bits * wire_count;
        (0..function.output_bits())
            .map(|output| {
                let mut mask_inputs = vec![k.clone()];
                mask_inputs.extend_from_slice(
                    &mask_outputs[output * mask_per_output..(output + 1) * mask_per_output],
                );
                let mask_circuit = build_final_mask_decrypt_circuit::<P, P::Matrix>(
                    self.ring_gsw.clone(),
                    self.config.prf_mask_output_coeff_bits,
                );
                let mask = self.compile_encoding_circuit(
                    compiler,
                    &mask_circuit,
                    one.clone(),
                    mask_inputs,
                    lookups,
                )?;
                let mut function_inputs = vec![k.clone()];
                function_inputs.extend_from_slice(
                    &function_outputs[output * wire_count..(output + 1) * wire_count],
                );
                let function_circuit =
                    build_final_function_decrypt_circuit::<P, P::Matrix>(self.ring_gsw.clone());
                let function = self.compile_encoding_circuit(
                    compiler,
                    &function_circuit,
                    one.clone(),
                    function_inputs,
                    lookups,
                )?;
                if mask.len() != 2 || function.len() != 2 {
                    return Err(DiamondIoCompileError::CircuitOutputLayout);
                }
                Ok(FinalWires {
                    function_secret_dependent: function[0].clone(),
                    function_bottom: function[1].clone(),
                    mask_secret_dependent: mask[0].clone(),
                    mask_bottom: mask[1].clone(),
                })
            })
            .collect()
    }

    fn export_public_key(
        &self,
        context: DslContext,
        name: impl Into<String>,
        key: NaiveBggPublicKeyVecWire,
    ) -> Result<DslContext, DslError> {
        context.public_family_output(name, key.matrices)
    }

    fn projection_preimages(
        &self,
        trapdoor: Trapdoor,
        key: &NaiveBggPublicKeyVecWire,
        top_plaintext: bool,
        bottom_plaintext: bool,
    ) -> Result<Family<Mat>, DslError> {
        let ring = self.ring();
        let gadget = ring.gadget(1, self.config.gadget_base.clone(), self.config.digit_count);
        let state_columns = self.config.input_config().state_columns().expect("validated layout");
        let columns = self.config.public_key_columns();
        key.matrices.clone().parallel_map(move |_, public_key| {
            let top = if top_plaintext { public_key - gadget.clone() } else { public_key };
            let bottom = if bottom_plaintext { -gadget.clone() } else { ring.zero((1, columns)) };
            trapdoor
                .sample_preimage(
                    Mat::concat(ConcatAxis::Rows, vec![top, bottom]),
                    (state_columns, columns),
                )
                .as_mat()
        })
    }

    fn export_projection_artifacts(
        &self,
        mut context: DslContext,
        trapdoors: &[Trapdoor],
        one: &NaiveBggPublicKeyVecWire,
        k: &NaiveBggPublicKeyVecWire,
        input_bits: &[NaiveBggPublicKeyVecWire],
    ) -> Result<DslContext, DiamondIoCompileError> {
        context = context.public_family_output(
            DiamondIoArtifactNames::ONE_PROJECTION,
            self.projection_preimages(trapdoors[0].clone(), one, true, false)?,
        )?;
        context = context.public_family_output(
            DiamondIoArtifactNames::K_PROJECTION,
            self.projection_preimages(trapdoors[0].clone(), k, false, true)?,
        )?;
        for (bit, key) in input_bits.iter().enumerate() {
            let digit = bit / self.config.batch_bits;
            let bit_in_digit = bit % self.config.batch_bits;
            let state = self.config.input_config().bit_state_index(digit, bit_in_digit)?;
            context = context.public_family_output(
                DiamondIoArtifactNames::input_projection(bit),
                self.projection_preimages(trapdoors[state].clone(), key, false, true)?,
            )?;
        }
        Ok(context)
    }

    fn import_public_key(
        &self,
        production: ProductionId,
        name: impl Into<String>,
        reveal_plaintext: bool,
    ) -> NaiveBggPublicKeyVecWire {
        NaiveBggPublicKeyVecWire {
            matrices: self.ring().family_artifact_input(
                production,
                name,
                self.config.ring_dimension,
                (1, self.config.public_key_columns()),
                ArtifactConfidentiality::Public,
            ),
            reveal_plaintext,
        }
    }

    fn project_encoding(
        &self,
        state: &Mat,
        production: ProductionId,
        projection: impl Into<String>,
        public_key: NaiveBggPublicKeyVecWire,
        plaintext: Option<Mat>,
    ) -> Result<NaiveBggEncodingVecWire, DslError> {
        let preimages = self.ring().family_artifact_input(
            production,
            projection,
            self.config.ring_dimension,
            (
                self.config.input_config().state_columns().expect("validated layout"),
                self.config.public_key_columns(),
            ),
            ArtifactConfidentiality::Public,
        );
        let state = state.clone();
        let vectors = preimages.parallel_map(move |_, preimage| state.clone() * preimage)?;
        let plaintexts = plaintext
            .map(|value| Parallel::range(self.config.ring_dimension).map(move |_| value.clone()))
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys: public_key.matrices,
            pubkey_reveal_plaintext: public_key.reveal_plaintext,
            plaintexts,
        })
    }

    fn import_transitions(&self, production: ProductionId) -> Vec<Vec<Vec<Mat>>> {
        (1..=self.config.input_count)
            .map(|level| {
                let states = self
                    .config
                    .input_config()
                    .state_count_at_level(level)
                    .expect("validated layout");
                (0..self.config.digit_base)
                    .map(|digit| {
                        (0..states)
                            .map(|state| {
                                self.ring().artifact_input(
                                    production.clone(),
                                    DiamondIoArtifactNames::injector_transition(
                                        level, digit, state,
                                    ),
                                    (
                                        self.config.input_config().state_columns().expect("layout"),
                                        self.config.input_config().state_columns().expect("layout"),
                                    ),
                                    ArtifactConfidentiality::Public,
                                )
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn import_round_artifacts(
        &self,
        production: ProductionId,
        round: usize,
        wire_count: usize,
    ) -> DiamondIoRoundPreprocessing {
        let ring = self.ring();
        let common_public_keys = (0..wire_count)
            .map(|wire| NaiveBggPublicKeyVecWire {
                matrices: ring.family_artifact_input(
                    production.clone(),
                    DiamondIoArtifactNames::round_common_public_key(round, wire),
                    self.config.ring_dimension,
                    (1, self.config.public_key_columns()),
                    ArtifactConfidentiality::Public,
                ),
                reveal_plaintext: true,
            })
            .collect::<Vec<_>>();
        let branch_rebase_preimages = (0..self.config.digit_base)
            .map(|branch| {
                (0..wire_count)
                    .map(|wire| {
                        ring.family_artifact_input(
                            production.clone(),
                            DiamondIoArtifactNames::round_rebase_preimages(round, branch, wire),
                            self.config.ring_dimension,
                            (
                                self.config.input_config().state_columns().expect("layout"),
                                self.config.public_key_columns(),
                            ),
                            ArtifactConfidentiality::Public,
                        )
                    })
                    .collect()
            })
            .collect();
        let refresh = (0..self.config.digit_base)
            .map(|branch| {
                (0..wire_count)
                    .map(|wire| NaiveBggNoiseRefreshArtifactWires {
                        a_prime: NaiveBggPublicKeyVecWire {
                            matrices: ring.family_artifact_input(
                                production.clone(),
                                DiamondIoArtifactNames::round_refresh_a_prime(round, wire),
                                self.config.ring_dimension,
                                (1, self.config.public_key_columns()),
                                ArtifactConfidentiality::Public,
                            ),
                            reveal_plaintext: true,
                        },
                        decoder_preimages: ring.family_artifact_input(
                            production.clone(),
                            DiamondIoArtifactNames::round_refresh_decoder_preimages(
                                round, branch, wire,
                            ),
                            self.config.ring_dimension *
                                self.config.refresh_crt_plaintext_moduli.len(),
                            (
                                self.config.input_config().state_columns().expect("layout"),
                                self.config.refresh_decoder_public_columns,
                            ),
                            ArtifactConfidentiality::Public,
                        ),
                    })
                    .collect()
            })
            .collect();
        DiamondIoRoundPreprocessing { common_public_keys, branch_rebase_preimages, refresh }
    }
}

#[derive(Clone)]
struct FinalWires<T> {
    function_secret_dependent: T,
    function_bottom: T,
    mask_secret_dependent: T,
    mask_bottom: T,
}

#[derive(Clone)]
pub struct DiamondIoRoundPreprocessing {
    pub common_public_keys: Vec<NaiveBggPublicKeyVecWire>,
    /// Indexed as `[branch][wire]`; each family contains one final-state
    /// preimage per slot.
    pub branch_rebase_preimages: Vec<Vec<Family<Mat>>>,
    /// Indexed as `[branch][wire]`.
    pub refresh: Vec<Vec<NaiveBggNoiseRefreshArtifactWires>>,
}

#[derive(Clone)]
pub struct DiamondIoRoundEvaluation {
    pub rebased: Vec<NaiveBggEncodingVecWire>,
    pub next_seed: Vec<NaiveBggEncodingVecWire>,
}

#[derive(Debug, Error)]
pub enum DiamondIoRoundError {
    #[error("Diamond PRF round input families have inconsistent branch or wire counts")]
    Layout,
    #[error(transparent)]
    Config(#[from] DiamondInputConfigError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    Naive(#[from] NaiveVecCompileError),
    #[error(transparent)]
    Refresh(#[from] NaiveBggNoiseRefreshError),
}

#[derive(Clone)]
pub struct DiamondIoRoundCompiler {
    pub config: DiamondIoConfig,
}

impl DiamondIoRoundCompiler {
    /// Preprocesses all digit branches. For every branch output `Y_b`, this
    /// samples a preimage that makes the selected online value equal to the
    /// same common public key:
    ///
    /// `state * R_b + (Y_b + (selector - b) H_b) = A_common`.
    ///
    /// The common-key value is then refreshed with the existing generic BGG+
    /// noise-refresh compiler. No iO-specific refresh arithmetic is copied.
    #[allow(clippy::too_many_arguments)]
    pub fn preprocess_round(
        &self,
        round: usize,
        hash_key: Bytes,
        one: &NaiveBggPublicKeyVecWire,
        selector: &NaiveBggPublicKeyVecWire,
        branch_outputs: &[Vec<NaiveBggPublicKeyVecWire>],
        decoded_refresh_material: &[Vec<Vec<NaiveBggPublicKeyVecWire>>],
        final_trapdoor: Trapdoor,
    ) -> Result<DiamondIoRoundPreprocessing, DiamondIoRoundError> {
        self.validate_branch_layout(branch_outputs, decoded_refresh_material)?;
        let branch_count = self.branch_count();
        let wire_count = branch_outputs[0].len();
        let ring = self.ring();
        let arithmetic = self.arithmetic();
        let common_public_keys = (0..wire_count)
            .map(|wire| self.common_public_key(hash_key.clone(), round, wire))
            .collect::<Result<Vec<_>, _>>()?;
        let mut branch_rebase_preimages = Vec::with_capacity(branch_count);
        let mut refresh = Vec::with_capacity(branch_count);
        for branch in 0..branch_count {
            let branch_sub = arithmetic
                .sub_public_keys(selector, &self.small_scalar_mul_public(one, branch)?)?;
            let mut branch_preimages = Vec::with_capacity(wire_count);
            let mut branch_refresh = Vec::with_capacity(wire_count);
            for wire in 0..wire_count {
                let mask = self.branch_mask(hash_key.clone(), round, branch, wire);
                let masked = arithmetic.add_public_keys(
                    &branch_outputs[branch][wire],
                    &arithmetic.matrix_mul_public_keys(&branch_sub, &mask)?,
                )?;
                let common = &common_public_keys[wire];
                let public_columns = self.public_columns();
                let targets = common.matrices.clone().parallel_zip(masked.matrices, {
                    let ring = ring.clone();
                    move |_, common, masked| {
                        Mat::concat(
                            ConcatAxis::Rows,
                            vec![common - masked, ring.zero((1, public_columns))],
                        )
                    }
                })?;
                let state_columns = self.config.input_config().state_columns()?;
                let public_columns = self.public_columns();
                let trapdoor = final_trapdoor.clone();
                let preimages = targets.parallel_map(move |_, target| {
                    trapdoor.sample_preimage(target, (state_columns, public_columns)).as_mat()
                })?;
                branch_preimages.push(preimages);

                let refresh_compiler = self.refresh_compiler();
                let refresh_id = self.refresh_id(round, wire);
                let wires = refresh_compiler.build_preprocessing(
                    hash_key.clone(),
                    &refresh_id,
                    one,
                    common,
                    &decoded_refresh_material[branch][wire],
                    final_trapdoor.clone(),
                )?;
                branch_refresh.push(NaiveBggNoiseRefreshArtifactWires {
                    a_prime: wires.a_prime,
                    decoder_preimages: wires.decoder_preimages,
                });
            }
            branch_rebase_preimages.push(branch_preimages);
            refresh.push(branch_refresh);
        }
        Ok(DiamondIoRoundPreprocessing { common_public_keys, branch_rebase_preimages, refresh })
    }

    /// Applies one public-input-selected PRF branch and the matching refresh.
    /// All branch values may coexist in the graph, but `selected_branch`
    /// chooses exactly one branch at every rebase and refresh boundary. A
    /// runtime can later replace this with branch-range subgraph calls without
    /// changing the symbolic equations or artifact schema.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_round(
        &self,
        round: usize,
        hash_key: Bytes,
        selected_branch: Int,
        final_state: Mat,
        one: &NaiveBggEncodingVecWire,
        selector: &NaiveBggEncodingVecWire,
        branch_outputs: &[Vec<NaiveBggEncodingVecWire>],
        decoded_refresh_material: &[Vec<Vec<NaiveBggEncodingVecWire>>],
        preprocessing: &DiamondIoRoundPreprocessing,
    ) -> Result<DiamondIoRoundEvaluation, DiamondIoRoundError> {
        self.validate_branch_layout(branch_outputs, decoded_refresh_material)?;
        let branch_count = self.branch_count();
        let wire_count = branch_outputs[0].len();
        if preprocessing.common_public_keys.len() != wire_count ||
            preprocessing.branch_rebase_preimages.len() != branch_count ||
            preprocessing.refresh.len() != branch_count
        {
            return Err(DiamondIoRoundError::Layout);
        }
        let arithmetic = self.arithmetic();
        let mut rebased = vec![Vec::with_capacity(wire_count); branch_count];
        for branch in 0..branch_count {
            let branch_sub = arithmetic.sub_encodings(
                selector,
                &arithmetic
                    .small_scalar_mul_encodings(one, &self.ring().polynomial([branch.into()]))?,
            )?;
            for wire in 0..wire_count {
                let mask = self.branch_mask(hash_key.clone(), round, branch, wire);
                let masked = arithmetic.add_encodings(
                    &branch_outputs[branch][wire],
                    &arithmetic.matrix_mul_encodings(&branch_sub, &mask)?,
                )?;
                let preimages = preprocessing.branch_rebase_preimages[branch][wire].clone();
                let projected = preimages.parallel_map({
                    let final_state = final_state.clone();
                    move |_, preimage| final_state.clone() * preimage
                })?;
                let vectors =
                    projected.parallel_zip(masked.vectors, |_, left, right| left + right)?;
                rebased[branch].push(NaiveBggEncodingVecWire {
                    vectors,
                    pubkeys: preprocessing.common_public_keys[wire].matrices.clone(),
                    pubkey_reveal_plaintext: true,
                    plaintexts: None,
                });
            }
        }

        let mut next = Vec::with_capacity(wire_count);
        let mut selected_rebased_outputs = Vec::with_capacity(wire_count);
        for wire in 0..wire_count {
            let selected_rebased = self.select_encoding(
                selected_branch.clone(),
                (0..branch_count).map(|branch| rebased[branch][wire].clone()).collect(),
            )?;
            let selected_material = (0..decoded_refresh_material[0][wire].len())
                .map(|material| {
                    self.select_encoding(
                        selected_branch.clone(),
                        (0..branch_count)
                            .map(|branch| decoded_refresh_material[branch][wire][material].clone())
                            .collect(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let selected_refresh = self.select_refresh(
                selected_branch.clone(),
                (0..branch_count)
                    .map(|branch| preprocessing.refresh[branch][wire].clone())
                    .collect(),
            )?;
            let projected_decoders = self.refresh_compiler().project_decoder_preimages(
                final_state.clone(),
                selected_refresh.decoder_preimages.clone(),
            )?;
            selected_rebased_outputs.push(selected_rebased.clone());
            next.push(self.refresh_compiler().build_online(
                one,
                &selected_rebased,
                &selected_material,
                &selected_refresh,
                projected_decoders,
            )?);
        }
        Ok(DiamondIoRoundEvaluation { rebased: selected_rebased_outputs, next_seed: next })
    }

    fn validate_branch_layout<W>(
        &self,
        branch_outputs: &[Vec<W>],
        decoded: &[Vec<Vec<W>>],
    ) -> Result<(), DiamondIoRoundError> {
        let branch_count = self.branch_count();
        let Some(wire_count) = branch_outputs.first().map(Vec::len) else {
            return Err(DiamondIoRoundError::Layout);
        };
        if wire_count == 0 ||
            branch_outputs.len() != branch_count ||
            decoded.len() != branch_count ||
            branch_outputs.iter().any(|branch| branch.len() != wire_count) ||
            decoded.iter().any(|branch| {
                branch.len() != wire_count ||
                    branch.iter().any(|material| {
                        material.len() !=
                            self.config.ring_dimension *
                                self.config.refresh_crt_plaintext_moduli.len() *
                                self.config.digit_count
                    })
            })
        {
            return Err(DiamondIoRoundError::Layout);
        }
        Ok(())
    }

    fn select_encoding(
        &self,
        selected: Int,
        branches: Vec<NaiveBggEncodingVecWire>,
    ) -> Result<NaiveBggEncodingVecWire, DiamondIoRoundError> {
        let slot_count = self.config.ring_dimension;
        let vectors = Family::pack(
            (0..slot_count)
                .map(|slot| {
                    selected.clone().select(
                        branches.iter().map(|branch| branch.vectors.get_static(slot)).collect(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let pubkeys = Family::pack(
            (0..slot_count)
                .map(|slot| {
                    selected.clone().select(
                        branches.iter().map(|branch| branch.pubkeys.get_static(slot)).collect(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: branches.iter().all(|branch| branch.pubkey_reveal_plaintext),
            plaintexts: None,
        })
    }

    fn select_refresh(
        &self,
        selected: Int,
        branches: Vec<NaiveBggNoiseRefreshArtifactWires>,
    ) -> Result<NaiveBggNoiseRefreshArtifactWires, DiamondIoRoundError> {
        let slot_count = self.config.ring_dimension;
        let a_prime = NaiveBggPublicKeyVecWire {
            matrices: Family::pack(
                (0..slot_count)
                    .map(|slot| {
                        selected.clone().select(
                            branches
                                .iter()
                                .map(|branch| branch.a_prime.matrices.get_static(slot))
                                .collect(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
            reveal_plaintext: true,
        };
        let count = slot_count * self.config.refresh_crt_plaintext_moduli.len();
        let decoder_preimages = Family::pack(
            (0..count)
                .map(|index| {
                    selected.clone().select(
                        branches
                            .iter()
                            .map(|branch| branch.decoder_preimages.get_static(index))
                            .collect(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        Ok(NaiveBggNoiseRefreshArtifactWires { a_prime, decoder_preimages })
    }

    fn common_public_key(
        &self,
        hash_key: Bytes,
        round: usize,
        wire: usize,
    ) -> Result<NaiveBggPublicKeyVecWire, DslError> {
        let ring = self.ring();
        let columns = self.public_columns();
        let matrices = Parallel::range(self.config.ring_dimension).map(move |slot| {
            let mut hash_tag = tag(b"DiamondIoPrfCommonRebase", &[round, wire]);
            hash_tag.push(slot);
            ring.hash_matrix(hash_key.clone(), hash_tag, (1, columns))
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: true })
    }

    fn branch_mask(&self, hash_key: Bytes, round: usize, branch: usize, wire: usize) -> Mat {
        self.ring().hash_matrix(
            hash_key,
            tag(b"DiamondIoPrfBranchMask", &[round, branch, wire]),
            (1, self.public_columns()),
        )
    }

    fn small_scalar_mul_public(
        &self,
        input: &NaiveBggPublicKeyVecWire,
        scalar: usize,
    ) -> Result<NaiveBggPublicKeyVecWire, DslError> {
        let compiler = self.public_key_compiler();
        let scalar = self.ring().polynomial([scalar.into()]);
        let reveal = input.reveal_plaintext;
        let matrices = input.matrices.clone().parallel_map(move |_, matrix| {
            compiler
                .small_scalar_mul(&BggPublicKeyWire { matrix, reveal_plaintext: reveal }, &scalar)
                .matrix
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: reveal })
    }

    fn refresh_compiler(&self) -> NaiveBggNoiseRefreshCompiler {
        NaiveBggNoiseRefreshCompiler {
            public_key: self.public_key_compiler(),
            modulus: self.config.modulus.clone().into(),
            ring_dimension: self.config.ring_dimension.into(),
            secret_size: 1,
            slot_count: self.config.ring_dimension,
            digit_count: self.config.digit_count,
            crt_scale_factors: self.config.refresh_crt_scale_factors.clone(),
            crt_plaintext_moduli: self.config.refresh_crt_plaintext_moduli.clone(),
            reconstruction_coefficients: self.config.refresh_reconstruction_coefficients.clone(),
            decoder_public_columns: self.config.refresh_decoder_public_columns,
            decoder_zero_rows: 1,
            decoder_trapdoor_sigma: self.config.trapdoor_sigma.clone(),
        }
    }

    fn arithmetic(&self) -> NaiveBggVecCompiler {
        NaiveBggVecCompiler { public_key: self.public_key_compiler() }
    }

    fn public_key_compiler(&self) -> BggPublicKeyCompiler {
        BggPublicKeyCompiler {
            ring: self.ring(),
            base: self.config.gadget_base_expr(),
            digit_count: self.config.digit_count_expr(),
        }
    }

    fn ring(&self) -> Ring {
        Ring::new(self.config.modulus.clone(), self.config.ring_dimension)
    }

    fn branch_count(&self) -> usize {
        1usize << self.config.batch_bits
    }

    fn public_columns(&self) -> usize {
        self.config.public_key_columns()
    }

    fn refresh_id(&self, round: usize, wire: usize) -> Vec<u8> {
        let mut id = b"DiamondIoPrfRefresh".to_vec();
        id.extend_from_slice(&round.to_le_bytes());
        id.extend_from_slice(&wire.to_le_bytes());
        id
    }
}

fn tag(domain: &[u8], indices: &[usize]) -> HashTag {
    let mut bytes = domain.to_vec();
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    HashTag::from(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::DiamondIoFunction;
    use mxx_dsl::DslContext;
    use mxx_gadgets::{circuit::PolyCircuit, circuit_gadgets::arith::NestedRnsPolyContext};
    use mxx_ir_core::{ParamEnv, RealExpr};
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

    fn top_level_compiler() -> (DiamondIoCompiler<DCRTPoly>, DiamondIoFunction) {
        let parameters = DCRTPolyParams::new(2, 1, 10, 5);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &parameters,
            5,
            2,
            16,
            false,
            Some(1),
        ));
        let ring_gsw = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            2,
            nested_rns,
            Some(1),
            Some(0),
        ));
        let modulus: Arc<BigUint> = parameters.modulus();
        let (crt_moduli, _, _) = parameters.to_crt();
        let function = DiamondIoFunction::GoldreichPrf { output_bits: 1 };
        let mut config = DiamondIoConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count: parameters.modulus_digits(),
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-top-level".to_vec(),
            seed_bits: 5,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: [2; 32],
            goldreich_graph_seed: [3; 32],
            ring_gsw_width: ring_gsw.width(),
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
            refresh_crt_scale_factors: crt_moduli
                .iter()
                .map(|modulus_i| BigInt::from(modulus.as_ref() / *modulus_i).into())
                .collect(),
            refresh_crt_plaintext_moduli: crt_moduli
                .iter()
                .map(|modulus| BigInt::from(*modulus).into())
                .collect(),
            refresh_reconstruction_coefficients: parameters
                .reconst_coeffs()
                .into_iter()
                .map(BigInt::from)
                .map(Into::into)
                .collect(),
            refresh_decoder_public_columns: 2 * (parameters.modulus_digits() + 2),
        };
        config.seed_bits = config.minimum_goldreich_seed_bits(&function).unwrap();
        (DiamondIoCompiler::new(config, ring_gsw).unwrap(), function)
    }

    #[test]
    fn compiler_rejects_ring_parameters_that_disagree_with_the_native_context() {
        let (compiler, _) = top_level_compiler();
        let context = compiler.ring_gsw.clone();
        let base = compiler.config;

        let mut wrong_modulus = base.clone();
        wrong_modulus.modulus += 1;
        let mut wrong_digit_count = base.clone();
        wrong_digit_count.digit_count += 1;
        let mut wrong_gadget_base = base.clone();
        wrong_gadget_base.gadget_base += 1;

        for config in [wrong_modulus, wrong_digit_count, wrong_gadget_base] {
            assert!(matches!(
                DiamondIoCompiler::new(config, context.clone()),
                Err(DiamondIoCompileError::Config(DiamondIoConfigError::RingGswLayout))
            ));
        }

        let parameters = DCRTPolyParams::new(2, 2, 10, 5);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &parameters,
            5,
            2,
            16,
            false,
            Some(2),
        ));
        let partial = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            2,
            nested_rns.clone(),
            Some(1),
            Some(0),
        ));
        let offset = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            2,
            nested_rns.clone(),
            Some(1),
            Some(1),
        ));
        let wrong_slots = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            1,
            nested_rns,
            Some(2),
            Some(0),
        ));
        let modulus: Arc<BigUint> = parameters.modulus();
        for context in [partial, offset, wrong_slots] {
            let mut config = base.clone();
            config.modulus = BigInt::from(modulus.as_ref().clone());
            config.ring_dimension = parameters.ring_dimension() as usize;
            config.digit_count = parameters.modulus_digits();
            config.gadget_base = BigInt::from(1u64 << parameters.base_bits());
            config.ring_gsw_width = context.width();
            assert!(matches!(
                DiamondIoCompiler::new(config, context),
                Err(DiamondIoCompileError::Config(DiamondIoConfigError::RingGswLayout))
            ));
        }
    }

    fn symbolic_manifest(
        production: ProductionId,
        elaborated: &mxx_ir_symbolic::ElaboratedGraph,
    ) -> mxx_ir_symbolic::manifest::Manifest {
        let artifacts = elaborated
            .outputs
            .iter()
            .map(|(name, reference)| {
                let wire = elaborated.wire(reference).unwrap();
                (
                    name.clone(),
                    mxx_ir_symbolic::manifest::ExportArtifact {
                        wire_type: wire.wire_type.clone(),
                        expression: wire.expression,
                        family: wire.family.clone(),
                        content_hash: None,
                        layout: None,
                    },
                )
            })
            .collect();
        mxx_ir_symbolic::manifest::export_manifest(
            production,
            &artifacts,
            &elaborated.atoms,
            &elaborated.expressions,
            &elaborated.preimage_relations,
            elaborated.assumption_digest,
        )
        .unwrap()
    }

    #[test]
    #[ignore = "expands the full nested-RNS Diamond iO graph"]
    fn top_level_producer_and_consumer_are_manifest_linked() {
        use mxx_ir_core::artifact::{SpecHash, export_validated_manifest};
        use std::collections::BTreeSet;

        let (compiler, function) = top_level_compiler();
        let producer = compiler.build_preprocessing(&function).unwrap();
        let bindings = ParamEnv::default();
        let validated_producer = producer.graph.validate(&bindings).unwrap();
        let expected = DiamondIoArtifactNames::all_public_names(
            &compiler.config,
            &function,
            compiler.ring_gsw.flattened_ciphertext_input_count(),
        )
        .unwrap()
        .into_iter()
        .collect::<BTreeSet<_>>();
        let actual = validated_producer.source.outputs().keys().cloned().collect::<BTreeSet<_>>();
        assert!(expected.is_subset(&actual));
        let lookup_artifacts = actual.difference(&expected).collect::<Vec<_>>();
        assert!(!lookup_artifacts.is_empty());
        assert!(lookup_artifacts.iter().all(|name| name.starts_with("lwe_lookup_")));
        let production = ProductionId { spec_hash: SpecHash([4; 32]), execution_nonce: [5; 32] };
        let artifact_manifest =
            export_validated_manifest(production.clone(), &validated_producer).unwrap();
        let elaborated_producer = producer.graph.elaborate(&bindings).unwrap();
        let symbolic_manifest = symbolic_manifest(production.clone(), &elaborated_producer);
        let consumer = compiler.build_evaluation(&function, production.clone()).unwrap();
        let manifests = BTreeMap::from([(production, artifact_manifest)]);
        consumer.graph.validate_with_manifests(&bindings, &manifests).unwrap();
        consumer
            .graph
            .elaborate_with_manifests(&bindings, &manifests, &[symbolic_manifest])
            .unwrap();
    }

    fn config() -> DiamondIoConfig {
        let mut config = DiamondIoConfig {
            modulus: 257.into(),
            ring_dimension: 2,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: 4.into(),
            digit_count: 2,
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-round-test".to_vec(),
            seed_bits: 5,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: [2; 32],
            goldreich_graph_seed: [3; 32],
            ring_gsw_width: 4,
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
            refresh_crt_scale_factors: vec![1.into()],
            refresh_crt_plaintext_moduli: vec![2.into()],
            refresh_reconstruction_coefficients: vec![1.into()],
            refresh_decoder_public_columns: 8,
        };
        let function = DiamondIoFunction::GoldreichPrf { output_bits: 1 };
        config.seed_bits = config.minimum_goldreich_seed_bits(&function).unwrap();
        config
    }

    #[test]
    fn one_full_branch_rebase_and_refresh_round_validates_and_elaborates() {
        let config = config();
        config.validate(&DiamondIoFunction::GoldreichPrf { output_bits: 1 }).unwrap();
        let compiler = DiamondIoRoundCompiler { config: config.clone() };
        let ring = compiler.ring();
        let public = |name: &str| NaiveBggPublicKeyVecWire {
            matrices: ring.input_family(name, 2, (1, 2)),
            reveal_plaintext: true,
        };
        let one_public = public("one-public");
        let selector_public = public("selector-public");
        let branch_public = vec![vec![public("branch-0")], vec![public("branch-1")]];
        let decoded_public = (0..2)
            .map(|branch| {
                vec![
                    (0..4)
                        .map(|material| public(&format!("decoded-public-{branch}-{material}")))
                        .collect(),
                ]
            })
            .collect::<Vec<Vec<Vec<_>>>>();
        let hash_key = ring.bytes_input("hash-key", 32);
        let final_trapdoor = ring.sample_trapdoor(2, 4, 4, 2);
        let preprocessing = compiler
            .preprocess_round(
                0,
                hash_key.clone(),
                &one_public,
                &selector_public,
                &branch_public,
                &decoded_public,
                final_trapdoor,
            )
            .unwrap();

        let encoding = |name: &str, keys: Family<Mat>| NaiveBggEncodingVecWire {
            vectors: ring.input_family(format!("{name}-vectors"), 2, (1, 2)),
            pubkeys: keys,
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        };
        let one = encoding("one", one_public.matrices);
        let selector = encoding("selector", selector_public.matrices);
        let branch_outputs = (0..2)
            .map(|branch| {
                vec![encoding(
                    &format!("branch-{branch}"),
                    branch_public[branch][0].matrices.clone(),
                )]
            })
            .collect::<Vec<_>>();
        let decoded = (0..2)
            .map(|branch| {
                vec![
                    (0..4)
                        .map(|material| {
                            encoding(
                                &format!("decoded-{branch}-{material}"),
                                decoded_public[branch][0][material].matrices.clone(),
                            )
                        })
                        .collect(),
                ]
            })
            .collect::<Vec<Vec<Vec<_>>>>();
        let selected = ring.input("selected-branch", (1, 1)).extract_coefficient(0);
        let evaluation = compiler
            .evaluate_round(
                0,
                hash_key,
                selected,
                ring.input("final-state", (1, 8)),
                &one,
                &selector,
                &branch_outputs,
                &decoded,
                &preprocessing,
            )
            .unwrap();
        let graph = DslContext::new("diamond-io-one-prf-round")
            .family_output("rebased", evaluation.rebased[0].vectors.clone())
            .unwrap()
            .family_output("next-seed", evaluation.next_seed[0].vectors.clone())
            .unwrap()
            .family_output("next-public-key", evaluation.next_seed[0].pubkeys.clone())
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        graph.elaborate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn batched_round_selects_every_branch_and_matches_direct_refresh_runtime() {
        let parameters = DCRTPolyParams::new(2, 1, 10, 5);
        let modulus: std::sync::Arc<BigUint> = parameters.modulus();
        let digit_count = parameters.modulus_digits();
        let state_columns = 2 * (digit_count + 2);
        let config = DiamondIoConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: 2,
            input_count: 1,
            digit_base: 4,
            batch_bits: 2,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count,
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-round-runtime".to_vec(),
            seed_bits: 5,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: [2; 32],
            goldreich_graph_seed: [3; 32],
            ring_gsw_width: 4,
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
            refresh_crt_scale_factors: vec![BigInt::from(modulus.as_ref() / 2u8).into()],
            refresh_crt_plaintext_moduli: vec![2.into()],
            refresh_reconstruction_coefficients: vec![1.into()],
            refresh_decoder_public_columns: state_columns,
        };
        let material_count =
            config.ring_dimension * config.refresh_crt_plaintext_moduli.len() * digit_count;

        for selected_branch in 0..4usize {
            let compiler = DiamondIoRoundCompiler { config: config.clone() };
            let ring = compiler.ring();
            let public = |name: &str| NaiveBggPublicKeyVecWire {
                matrices: ring.input_family(name, 2, (1, digit_count)),
                reveal_plaintext: true,
            };
            let one_public = public("one-public");
            let selector_public = public("selector-public");
            let branch_public = (0..4)
                .map(|branch| vec![public(&format!("branch-public-{branch}"))])
                .collect::<Vec<_>>();
            let decoded_public = (0..4)
                .map(|branch| {
                    vec![
                        (0..material_count)
                            .map(|material| public(&format!("decoded-public-{branch}-{material}")))
                            .collect(),
                    ]
                })
                .collect::<Vec<Vec<Vec<_>>>>();
            let hash_key = ring.bytes_input("hash-key", 32);
            let preprocessing = compiler
                .preprocess_round(
                    0,
                    hash_key.clone(),
                    &one_public,
                    &selector_public,
                    &branch_public,
                    &decoded_public,
                    ring.sample_trapdoor(2, 4, config.gadget_base_expr(), digit_count),
                )
                .unwrap();
            let encoding = |name: &str, keys: Family<Mat>| NaiveBggEncodingVecWire {
                vectors: ring.input_family(format!("{name}-vectors"), 2, (1, digit_count)),
                pubkeys: keys,
                pubkey_reveal_plaintext: true,
                plaintexts: None,
            };
            let one = encoding("one", one_public.matrices);
            let selector = encoding("selector", selector_public.matrices);
            let branch_outputs = (0..4)
                .map(|branch| {
                    vec![encoding(
                        &format!("branch-{branch}"),
                        branch_public[branch][0].matrices.clone(),
                    )]
                })
                .collect::<Vec<_>>();
            let decoded = (0..4)
                .map(|branch| {
                    vec![
                        (0..material_count)
                            .map(|material| {
                                encoding(
                                    &format!("decoded-{branch}-{material}"),
                                    decoded_public[branch][0][material].matrices.clone(),
                                )
                            })
                            .collect(),
                    ]
                })
                .collect::<Vec<Vec<Vec<_>>>>();
            let selected = ring.input("selected-branch", (1, 1)).extract_coefficient(0);
            let final_state = ring.input("final-state", (1, state_columns));
            let evaluation = compiler
                .evaluate_round(
                    0,
                    hash_key,
                    selected,
                    final_state.clone(),
                    &one,
                    &selector,
                    &branch_outputs,
                    &decoded,
                    &preprocessing,
                )
                .unwrap();
            let oracle_artifacts = preprocessing.refresh[selected_branch][0].clone();
            let oracle_decoders = compiler
                .refresh_compiler()
                .project_decoder_preimages(final_state, oracle_artifacts.decoder_preimages.clone())
                .unwrap();
            let oracle = compiler
                .refresh_compiler()
                .build_online(
                    &one,
                    &branch_outputs[selected_branch][0],
                    &decoded[selected_branch][0],
                    &oracle_artifacts,
                    oracle_decoders,
                )
                .unwrap();
            let graph = DslContext::new(format!("diamond-round-runtime-{selected_branch}"))
                .family_output("rebased", evaluation.rebased[0].vectors.clone())
                .unwrap()
                .family_output(
                    "selected-branch-vector",
                    branch_outputs[selected_branch][0].vectors.clone(),
                )
                .unwrap()
                .family_output("next", evaluation.next_seed[0].vectors.clone())
                .unwrap()
                .family_output("oracle", oracle.vectors)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();

            let matrix = |rows: usize, columns: usize, value: usize| {
                DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    (0..rows)
                        .map(|_| {
                            (0..columns)
                                .map(|_| DCRTPoly::from_usize_to_constant(&parameters, value))
                                .collect()
                        })
                        .collect(),
                )
            };
            let family = |rows: usize, columns: usize, value: usize| {
                RuntimeValue::IndexedFamily(
                    (0..2).map(|_| RuntimeValue::matrix(matrix(rows, columns, value))).collect(),
                )
            };
            let mut inputs = BTreeMap::from([
                ("hash-key".to_owned(), RuntimeValue::Bytes(vec![7; 32])),
                ("one-public".to_owned(), family(1, digit_count, 0)),
                ("selector-public".to_owned(), family(1, digit_count, 0)),
                ("one-vectors".to_owned(), family(1, digit_count, 1)),
                ("selector-vectors".to_owned(), family(1, digit_count, selected_branch)),
                ("selected-branch".to_owned(), RuntimeValue::matrix(matrix(1, 1, selected_branch))),
                ("final-state".to_owned(), RuntimeValue::matrix(matrix(1, state_columns, 0))),
            ]);
            for branch in 0..4 {
                inputs.insert(format!("branch-public-{branch}"), family(1, digit_count, 0));
                inputs.insert(
                    format!("branch-{branch}-vectors"),
                    family(1, digit_count, 10 + branch),
                );
                for material in 0..material_count {
                    inputs.insert(
                        format!("decoded-public-{branch}-{material}"),
                        family(1, digit_count, 0),
                    );
                    inputs.insert(
                        format!("decoded-{branch}-{material}-vectors"),
                        family(1, digit_count, 0),
                    );
                }
            }
            let result = execute(
                &graph,
                &mut cpu_backend([parameters.clone()]),
                inputs,
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let family_output = |name: &str| match &result.outputs[name] {
                RuntimeValue::IndexedFamily(values) => values
                    .iter()
                    .map(|value| match value {
                        RuntimeValue::Matrix(matrix) => matrix.as_ref().clone(),
                        _ => panic!("{name} family member must be a matrix"),
                    })
                    .collect::<Vec<_>>(),
                _ => panic!("{name} must be a family"),
            };
            assert_eq!(family_output("rebased"), family_output("selected-branch-vector"));
            assert_eq!(family_output("next"), family_output("oracle"));
        }
    }

    #[test]
    fn nonzero_mask_rebase_maps_the_related_final_state_to_the_common_key() {
        let parameters = DCRTPolyParams::new(2, 1, 10, 5);
        let modulus: std::sync::Arc<BigUint> = parameters.modulus();
        let digit_count = parameters.modulus_digits();
        let state_columns = 2 * (digit_count + 2);
        let config = DiamondIoConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: 2,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count,
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-rebase-sign".to_vec(),
            seed_bits: 5,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: [2; 32],
            goldreich_graph_seed: [3; 32],
            ring_gsw_width: 4,
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
            refresh_crt_scale_factors: vec![BigInt::from(modulus.as_ref() / 2u8).into()],
            refresh_crt_plaintext_moduli: vec![2.into()],
            refresh_reconstruction_coefficients: vec![1.into()],
            refresh_decoder_public_columns: state_columns,
        };
        let compiler = DiamondIoRoundCompiler { config: config.clone() };
        let ring = compiler.ring();
        let public = |name: &str| NaiveBggPublicKeyVecWire {
            matrices: ring.input_family(name, 2, (1, digit_count)),
            reveal_plaintext: true,
        };
        let one_public = public("one-public");
        let selector_public = public("selector-public");
        let branch_public = vec![vec![public("branch-public-0")], vec![public("branch-public-1")]];
        let decoded_public = (0..2)
            .map(|branch| {
                vec![
                    (0..2 * digit_count)
                        .map(|material| public(&format!("decoded-public-{branch}-{material}")))
                        .collect(),
                ]
            })
            .collect::<Vec<Vec<Vec<_>>>>();
        let hash_key = ring.bytes_input("hash-key", 32);
        let trapdoor = ring.sample_trapdoor(2, 4, config.gadget_base_expr(), digit_count);
        let final_state = ring.input("state-secret", (1, 2)) * trapdoor.public_matrix();
        let preprocessing = compiler
            .preprocess_round(
                0,
                hash_key.clone(),
                &one_public,
                &selector_public,
                &branch_public,
                &decoded_public,
                trapdoor,
            )
            .unwrap();
        let encoding = |name: &str, keys: Family<Mat>| NaiveBggEncodingVecWire {
            vectors: ring.input_family(format!("{name}-vectors"), 2, (1, digit_count)),
            pubkeys: keys,
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        };
        let one = encoding("one", one_public.matrices);
        let selector = encoding("selector", selector_public.matrices);
        let branch_outputs = (0..2)
            .map(|branch| {
                vec![encoding(
                    &format!("branch-{branch}"),
                    branch_public[branch][0].matrices.clone(),
                )]
            })
            .collect::<Vec<_>>();
        let decoded = (0..2)
            .map(|branch| {
                vec![
                    (0..2 * digit_count)
                        .map(|material| {
                            encoding(
                                &format!("decoded-{branch}-{material}"),
                                decoded_public[branch][0][material].matrices.clone(),
                            )
                        })
                        .collect(),
                ]
            })
            .collect::<Vec<Vec<Vec<_>>>>();
        let evaluation = compiler
            .evaluate_round(
                0,
                hash_key,
                ring.input("selected", (1, 1)).extract_coefficient(0),
                final_state,
                &one,
                &selector,
                &branch_outputs,
                &decoded,
                &preprocessing,
            )
            .unwrap();
        let mut context = DslContext::new("diamond-rebase-sign-runtime");
        for slot in 0..2 {
            context = context
                .output(format!("rebased-{slot}"), evaluation.rebased[0].vectors.get_static(slot))
                .unwrap()
                .output(
                    format!("common-{slot}"),
                    preprocessing.common_public_keys[0].matrices.get_static(slot),
                )
                .unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let matrix = |rows: usize, columns: usize, value: usize| {
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                (0..rows)
                    .map(|_| {
                        (0..columns)
                            .map(|_| DCRTPoly::from_usize_to_constant(&parameters, value))
                            .collect()
                    })
                    .collect(),
            )
        };
        let family = |value: usize| {
            RuntimeValue::IndexedFamily(
                (0..2).map(|_| RuntimeValue::matrix(matrix(1, digit_count, value))).collect(),
            )
        };
        let mut inputs = BTreeMap::from([
            ("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x5a; 32])),
            ("one-public".to_owned(), family(1)),
            ("selector-public".to_owned(), family(3)),
            ("one-vectors".to_owned(), family(1)),
            ("selector-vectors".to_owned(), family(3)),
            ("branch-public-0".to_owned(), family(5)),
            ("branch-public-1".to_owned(), family(7)),
            ("branch-0-vectors".to_owned(), family(5)),
            ("branch-1-vectors".to_owned(), family(7)),
            ("selected".to_owned(), RuntimeValue::matrix(matrix(1, 1, 1))),
            (
                "state-secret".to_owned(),
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    vec![vec![
                        DCRTPoly::from_usize_to_constant(&parameters, 1),
                        DCRTPoly::from_usize_to_constant(&parameters, 0),
                    ]],
                )),
            ),
        ]);
        for branch in 0..2 {
            for material in 0..2 * digit_count {
                inputs.insert(format!("decoded-public-{branch}-{material}"), family(0));
                inputs.insert(format!("decoded-{branch}-{material}-vectors"), family(0));
            }
        }
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let matrix_output = |name: &str| match &result.outputs[name] {
            RuntimeValue::Matrix(matrix) => matrix.as_ref().clone(),
            _ => panic!("{name} must be a matrix output"),
        };
        for slot in 0..2 {
            assert_eq!(
                matrix_output(&format!("rebased-{slot}")),
                matrix_output(&format!("common-{slot}"))
            );
        }
    }
}
