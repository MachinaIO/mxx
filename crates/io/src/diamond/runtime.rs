//! Diamond iO runtime and transient native Ring-GSW seed material.

use super::{
    DiamondIoFunction,
    graph::{
        DiamondIoCompileError, DiamondIoCompiler, DiamondIoPoly, PUBLIC_INPUT_DIGIT_PREFIX,
        output_name,
    },
};
use crate::Obfuscation;
use mxx_dsl::Ring;
use mxx_gadgets::circuit_gadgets::fhe::ring_gsw_nested_rns::{
    NativeRingGswCiphertext, NestedRnsRingGswContext, declare_native_ring_gsw_dsl_inputs,
    encrypt_plaintext_bit_with_sampler, native_ring_gsw_scalar_bindings,
    sample_public_key_with_samplers,
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ProductionId, production_id},
    encoding::spec_hash,
};
use mxx_primitives::{
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue, SessionStore, backend::poly::PolyBackend, execute,
    execute_in_session_with_config, transcript::SamplingMode,
};
use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

/// Private setup result. `secret_key` and `seed_bits` are consumed while the
/// obfuscation graph is executed and must not be retained in the public handle.
pub struct DiamondIoNativeSeedSetup<P> {
    secret_key: P,
    seed_bits: Vec<bool>,
    ciphertexts: Vec<NativeRingGswCiphertext<P>>,
    public_key_error_sigma: f64,
}

impl<P> DiamondIoNativeSeedSetup<P> {
    /// Plaintext seed used by correctness tests that inject this exact setup.
    pub fn seed_bits(&self) -> &[bool] {
        &self.seed_bits
    }

    fn matches_error_sigma(&self, expected: f64) -> bool {
        self.public_key_error_sigma.to_bits() == expected.to_bits()
    }
}

#[derive(Debug, Error)]
pub enum DiamondIoNativeSeedError {
    #[error("Diamond iO requires at least one private seed bit")]
    EmptySeed,
    #[error("native Ring-GSW public-key error sigma must be finite and strictly positive")]
    InvalidErrorSigma,
    #[error("native Ring-GSW binding count differs between seed ciphertexts")]
    BindingLayout,
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
}

/// Compact public result of Diamond preprocessing. Native Ring-GSW keys and
/// ciphertexts are transient graph inputs and are deliberately not retained.
#[derive(Clone, Debug)]
pub struct DiamondIoObfuscation {
    pub function: DiamondIoFunction,
    pub production: ProductionId,
}

pub struct DiamondIoRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    M::P: DiamondIoPoly<Matrix = M> + 'static,
{
    pub compiler: DiamondIoCompiler<M::P>,
    pub parameters: <M::P as Poly>::Params,
    pub backend: PolyBackend<M, U, H, T>,
    pub store: S,
    pub execution_config: ExecutionConfig,
}

#[derive(Debug, Error)]
pub enum DiamondIoRuntimeError {
    #[error(transparent)]
    Compile(#[from] DiamondIoCompileError),
    #[error(transparent)]
    NativeSeed(#[from] DiamondIoNativeSeedError),
    #[error("Diamond iO runtime parameters do not match its compiler configuration")]
    ParameterMismatch,
    #[error("native Ring-GSW seed noise does not match the compiler configuration")]
    NativeSeedNoiseMismatch,
    #[error("Diamond iO expression evaluation failed: {0}")]
    Expression(String),
    #[error("Diamond iO graph validation failed: {0}")]
    Validation(String),
    #[error("Diamond iO artifact store failed: {0}")]
    Store(String),
    #[error("Diamond iO graph execution failed: {0}")]
    Execution(String),
    #[error("Diamond iO input has the wrong number of bits")]
    InputLength,
    #[error("Diamond iO preprocessing graph does not match the public handle")]
    ProductionGraphMismatch,
    #[error("Diamond iO evaluation returned an invalid or inconsistent Boolean output")]
    OutputLayout,
}

pub fn sample_native_seed<P, M, HS, US>(
    parameters: &P::Params,
    context: &Arc<NestedRnsRingGswContext<P>>,
    seed_bits: usize,
    hash_key: [u8; 32],
    public_key_error_sigma: f64,
) -> Result<DiamondIoNativeSeedSetup<P>, DiamondIoNativeSeedError>
where
    P: Poly + Send + Sync + 'static,
    M: PolyMatrix<P = P>,
    HS: PolyHashSampler<[u8; 32], M = M>,
    US: PolyUniformSampler<M = M> + Sync,
{
    if seed_bits == 0 {
        return Err(DiamondIoNativeSeedError::EmptySeed);
    }
    if !public_key_error_sigma.is_finite() || public_key_error_sigma <= 0.0 {
        return Err(DiamondIoNativeSeedError::InvalidErrorSigma);
    }
    let secret_key = US::new().sample_poly(parameters, &DistType::TernaryDist);
    let public_key = sample_public_key_with_samplers::<P, M, HS, US, _>(
        parameters,
        context.width(),
        &secret_key,
        hash_key,
        b"diamond-io-ring-gsw-public-key",
        Some(public_key_error_sigma),
    );
    let bits = (0..seed_bits).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let ciphertexts = bits
        .par_iter()
        .map(|bit| {
            encrypt_plaintext_bit_with_sampler::<P, M, US>(
                parameters,
                context.nested_rns.as_ref(),
                &public_key,
                *bit,
            )
        })
        .collect();
    Ok(DiamondIoNativeSeedSetup {
        secret_key,
        seed_bits: bits,
        ciphertexts,
        public_key_error_sigma,
    })
}

/// Returns the exact input names and scalar families used to compile every
/// encrypted seed bit. Public-key preprocessing and encoding evaluation call
/// this same function, preventing drift in ciphertext wire order.
pub fn declare_native_seed_inputs(
    ring: &Ring,
    prefix: &str,
    seed_bits: usize,
    wire_count: usize,
    slot_count: usize,
    ciphertext_error_norm: mxx_ir_core::RealExpr,
) -> Result<
    Vec<mxx_gadgets::circuit_gadgets::fhe::ring_gsw_nested_rns::NativeRingGswDslInputs>,
    DiamondIoNativeSeedError,
> {
    (0..seed_bits)
        .map(|seed| {
            Ok(declare_native_ring_gsw_dsl_inputs(
                ring,
                &format!("{prefix}-seed-{seed}"),
                wire_count,
                slot_count,
                ciphertext_error_norm.clone(),
            )?)
        })
        .collect()
}

/// Converts sampled ciphertexts into the runtime values expected by
/// [`declare_native_seed_inputs`].
pub fn native_seed_bindings<P, M>(
    parameters: &P::Params,
    context: &Arc<NestedRnsRingGswContext<P>>,
    prefix: &str,
    ciphertexts: &[NativeRingGswCiphertext<P>],
) -> Result<BTreeMap<String, Vec<M>>, DiamondIoNativeSeedError>
where
    P: Poly + Send + Sync + 'static,
    M: PolyMatrix<P = P>,
{
    let per_seed = ciphertexts
        .par_iter()
        .map(|ciphertext| {
            native_ring_gsw_scalar_bindings::<P, M>(
                parameters,
                context.nested_rns.as_ref(),
                ciphertext,
                context.level_offset,
                Some(context.active_levels),
            )
        })
        .collect::<Vec<_>>();
    let Some(wire_count) = per_seed.first().map(Vec::len) else {
        return Err(DiamondIoNativeSeedError::EmptySeed);
    };
    if per_seed.iter().any(|seed| seed.len() != wire_count) {
        return Err(DiamondIoNativeSeedError::BindingLayout);
    }
    Ok(per_seed
        .into_iter()
        .enumerate()
        .flat_map(|(seed, wires)| {
            wires
                .into_iter()
                .enumerate()
                .map(move |(wire, values)| (format!("{prefix}-seed-{seed}-{wire}"), values))
        })
        .collect())
}

impl<M, U, H, T, S> DiamondIoRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    M::P: DiamondIoPoly<Matrix = M> + Send + Sync + 'static,
    U: PolyUniformSampler<M = M> + Sync,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    pub fn new(
        compiler: DiamondIoCompiler<M::P>,
        parameters: <M::P as Poly>::Params,
        store: S,
    ) -> Result<Self, DiamondIoRuntimeError> {
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        if compiler.config.modulus != num_bigint::BigInt::from(modulus.as_ref().clone()) ||
            compiler.config.ring_dimension != parameters.ring_dimension() as usize ||
            compiler.config.digit_count != parameters.modulus_digits() ||
            compiler.config.gadget_base !=
                num_bigint::BigInt::from(1u64 << parameters.base_bits())
        {
            return Err(DiamondIoRuntimeError::ParameterMismatch);
        }
        Ok(Self {
            compiler,
            backend: PolyBackend::new_for_execution([parameters.clone()]),
            parameters,
            store,
            execution_config: ExecutionConfig::default(),
        })
    }

    pub fn with_execution_config(mut self, execution_config: ExecutionConfig) -> Self {
        self.execution_config = execution_config;
        self
    }

    pub fn obfuscate_with_hash_key(
        &mut self,
        function: &DiamondIoFunction,
        hash_key: [u8; 32],
    ) -> Result<DiamondIoObfuscation, DiamondIoRuntimeError> {
        let bindings = ParamEnv::default();
        let public_key_error_sigma = self
            .compiler
            .config
            .ring_gsw_public_key_error_sigma
            .as_ref()
            .ok_or(DiamondIoRuntimeError::NativeSeedNoiseMismatch)?
            .evaluate_f64(&bindings)
            .map_err(|error| DiamondIoRuntimeError::Expression(error.to_string()))?;
        let native = sample_native_seed::<M::P, M, H, U>(
            &self.parameters,
            &self.compiler.ring_gsw,
            self.compiler.config.seed_bits,
            hash_key,
            public_key_error_sigma,
        )?;
        self.obfuscate_with_native_seed(function, hash_key, native)
    }

    /// Executes preprocessing with explicitly sampled native seed material.
    /// Secret values are transient inputs and are not retained by the public
    /// obfuscation handle.
    pub fn obfuscate_with_native_seed(
        &mut self,
        function: &DiamondIoFunction,
        hash_key: [u8; 32],
        native: DiamondIoNativeSeedSetup<M::P>,
    ) -> Result<DiamondIoObfuscation, DiamondIoRuntimeError> {
        let configured_sigma = self
            .compiler
            .config
            .ring_gsw_public_key_error_sigma
            .as_ref()
            .ok_or(DiamondIoRuntimeError::NativeSeedNoiseMismatch)?
            .evaluate_f64(&ParamEnv::default())
            .map_err(|error| DiamondIoRuntimeError::Expression(error.to_string()))?;
        if !native.matches_error_sigma(configured_sigma) {
            return Err(DiamondIoRuntimeError::NativeSeedNoiseMismatch);
        }
        let built = self.compiler.build_preprocessing(function)?.graph;
        let bindings = ParamEnv::default();
        let validated = built
            .validate(&bindings)
            .map_err(|error| DiamondIoRuntimeError::Validation(error.to_string()))?;
        let production = production_id(
            spec_hash(&validated.source, &validated.bindings)
                .map_err(|error| DiamondIoRuntimeError::Validation(error.to_string()))?,
            hash_key,
        );
        let seed_bindings = native_seed_bindings::<M::P, M>(
            &self.parameters,
            &self.compiler.ring_gsw,
            super::graph::NATIVE_SEED_INPUT_PREFIX,
            &native.ciphertexts,
        )?;
        let mut inputs = BTreeMap::from([
            (super::graph::HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.to_vec())),
            (
                super::graph::NOISE_REFRESH_HASH_KEY_INPUT.to_owned(),
                RuntimeValue::Bytes(self.compiler.config.noise_refresh_hash_key.to_vec()),
            ),
            (
                super::graph::PRIVATE_K_INPUT.to_owned(),
                RuntimeValue::matrix(M::from_poly_vec(
                    &self.parameters,
                    vec![vec![native.secret_key]],
                )),
            ),
        ]);
        inputs.extend(seed_bindings.into_iter().map(|(name, matrices)| {
            (
                name,
                RuntimeValue::IndexedFamily(
                    matrices.into_iter().map(RuntimeValue::matrix).collect(),
                ),
            )
        }));
        execute_in_session_with_config(
            &validated,
            &mut self.backend,
            inputs,
            &mut self.store,
            hash_key,
            self.execution_config,
        )
        .map_err(|error| DiamondIoRuntimeError::Execution(error.to_string()))?;
        Ok(DiamondIoObfuscation { function: function.clone(), production })
    }

    pub fn evaluate_bits(
        &mut self,
        obfuscation: &DiamondIoObfuscation,
        input: &[bool],
    ) -> Result<Vec<bool>, DiamondIoRuntimeError> {
        if input.len() != self.compiler.config.input_bits().map_err(DiamondIoCompileError::from)? {
            return Err(DiamondIoRuntimeError::InputLength);
        }
        let preprocessing = self.compiler.build_preprocessing(&obfuscation.function)?.graph;
        let validated_preprocessing = preprocessing
            .validate(&ParamEnv::default())
            .map_err(|error| DiamondIoRuntimeError::Validation(error.to_string()))?;
        let preprocessing_hash =
            spec_hash(&validated_preprocessing.source, &validated_preprocessing.bindings)
                .map_err(|error| DiamondIoRuntimeError::Validation(error.to_string()))?;
        if preprocessing_hash != obfuscation.production.spec_hash {
            return Err(DiamondIoRuntimeError::ProductionGraphMismatch);
        }
        let manifest = self
            .store
            .load_manifest(&obfuscation.production)
            .map_err(|error| DiamondIoRuntimeError::Store(error.to_string()))?;
        let manifests = BTreeMap::from([(obfuscation.production.clone(), manifest)]);
        let evaluation = self
            .compiler
            .build_evaluation(&obfuscation.function, obfuscation.production.clone())?
            .graph;
        let validated = evaluation
            .validate_with_manifests(&ParamEnv::default(), &manifests)
            .map_err(|error| DiamondIoRuntimeError::Validation(error.to_string()))?;
        let mut inputs = BTreeMap::new();
        for (digit, value) in input
            .chunks_exact(self.compiler.config.batch_bits)
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .fold(0usize, |value, (bit, set)| value | (usize::from(*set) << bit))
            })
            .enumerate()
        {
            inputs.insert(
                format!("{PUBLIC_INPUT_DIGIT_PREFIX}-{digit}"),
                RuntimeValue::matrix(self.scalar_matrix(value)),
            );
        }
        let result =
            execute(&validated, &mut self.backend, inputs, &mut self.store, SamplingMode::Fresh)
                .map_err(|error| DiamondIoRuntimeError::Execution(error.to_string()))?;
        (0..obfuscation.function.output_bits())
            .map(|output| {
                let Some(RuntimeValue::IndexedFamily(slots)) =
                    result.outputs.get(&output_name(output))
                else {
                    return Err(DiamondIoRuntimeError::OutputLayout);
                };
                let mut values = slots.iter().map(|slot| match slot {
                    RuntimeValue::Bool(value) => Ok(*value),
                    _ => Err(DiamondIoRuntimeError::OutputLayout),
                });
                let first = values.next().ok_or(DiamondIoRuntimeError::OutputLayout)??;
                if values.any(|value| value.is_err() || value.is_ok_and(|value| value != first)) {
                    return Err(DiamondIoRuntimeError::OutputLayout);
                }
                Ok(first)
            })
            .collect()
    }

    fn scalar_matrix(&self, value: usize) -> M {
        M::from_poly_vec(
            &self.parameters,
            vec![vec![M::P::from_usize_to_constant(&self.parameters, value)]],
        )
    }
}

impl<M, U, H, T, S> Obfuscation for DiamondIoRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    M::P: DiamondIoPoly<Matrix = M> + Send + Sync + 'static,
    U: PolyUniformSampler<M = M> + Sync,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    type Function = DiamondIoFunction;
    type Obfuscation = DiamondIoObfuscation;
    type Input = Vec<bool>;
    type Output = Vec<bool>;
    type Error = DiamondIoRuntimeError;

    fn obfuscate(&mut self, function: &Self::Function) -> Result<Self::Obfuscation, Self::Error> {
        self.obfuscate_with_hash_key(function, rand::random())
    }

    fn evaluate(
        &mut self,
        obfuscation: &Self::Obfuscation,
        input: &Self::Input,
    ) -> Result<Self::Output, Self::Error> {
        self.evaluate_bits(obfuscation, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keccak_asm::Keccak256;
    use mxx_gadgets::{
        circuit::PolyCircuit,
        circuit_gadgets::{
            arith::NestedRnsPolyContext,
            fhe::ring_gsw_nested_rns::NestedRnsRingGswContext,
            fhe_prg::goldreich::{
                GoldreichFullDomainRangeGenerator, GoldreichGraphGeneration,
                evaluate_goldreich_bits,
            },
        },
    };
    use mxx_ir_core::RealExpr;
    use mxx_primitives::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };
    use mxx_runtime::artifact::MemoryArtifactStore;
    use num_bigint::BigInt;

    type TestRuntime = DiamondIoRuntime<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        MemoryArtifactStore,
    >;

    #[test]
    fn generic_seed_setup_and_bindings_share_one_wire_layout() {
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
        let context = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            2,
            nested_rns,
            Some(1),
            Some(0),
        ));
        assert!(matches!(
            sample_native_seed::<
                DCRTPoly,
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyUniformSampler,
            >(&parameters, &context, 3, [7; 32], 0.0),
            Err(DiamondIoNativeSeedError::InvalidErrorSigma)
        ));
        let setup = sample_native_seed::<
            DCRTPoly,
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyUniformSampler,
        >(&parameters, &context, 3, [7; 32], 1.0)
        .unwrap();
        assert!(setup.matches_error_sigma(1.0));
        assert!(!setup.matches_error_sigma(2.0));
        let bindings = native_seed_bindings::<DCRTPoly, DCRTPolyMatrix>(
            &parameters,
            &context,
            "diamond-native",
            &setup.ciphertexts,
        )
        .unwrap();
        let wire_count = bindings.len() / setup.seed_bits.len();
        assert!(wire_count > 0);
        assert_eq!(bindings.len(), setup.seed_bits.len() * wire_count);
        assert!(bindings.values().all(|slots| slots.len() == parameters.ring_dimension() as usize));

        let modulus: Arc<num_bigint::BigUint> = parameters.modulus();
        let ring = Ring::new(
            num_bigint::BigInt::from(modulus.as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let declarations = declare_native_seed_inputs(
            &ring,
            "diamond-native",
            setup.seed_bits.len(),
            wire_count,
            parameters.ring_dimension() as usize,
            RealExpr::from_integer(1),
        )
        .unwrap();
        assert_eq!(declarations.len(), setup.seed_bits.len());
        assert!(
            declarations
                .iter()
                .flat_map(|declaration| declaration.input_names.iter())
                .all(|name| bindings.contains_key(name))
        );
    }

    #[test]
    fn native_seed_bottom_row_exposes_encryption_noise_to_the_simulator() {
        let ring = Ring::new(257, 8);
        let declarations = declare_native_seed_inputs(
            &ring,
            "diamond-native-noise",
            1,
            4,
            2,
            RealExpr::from_integer(7),
        )
        .unwrap();
        let graph = mxx_dsl::DslContext::new("diamond-native-noise-assumption")
            .public_family_output("bottom", declarations[0].scalar_families[2].clone())
            .unwrap()
            .build()
            .unwrap();
        let elaborated = graph.elaborate(&ParamEnv::default()).unwrap();
        let report = mxx_noise_simulator::simulate(&elaborated).unwrap();
        let output = &report.outputs["bottom"];
        assert!(output.has_signal);
        assert_eq!(
            output.noise.as_ref().expect("bottom-row error must remain visible").bound,
            bigdecimal::BigDecimal::from(7)
        );
    }

    #[test]
    #[ignore = "expands and executes the full nested-RNS Diamond iO graph"]
    fn obfuscation_runtime_matches_the_plaintext_goldreich_path_with_nonzero_noise() {
        let parameters = DCRTPolyParams::new(2, 1, 10, 5);
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus();
        let mut native_circuit = PolyCircuit::<DCRTPoly>::new();
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut native_circuit,
            &parameters,
            5,
            2,
            16,
            false,
            Some(1),
        ));
        let ring_gsw = Arc::new(NestedRnsRingGswContext::from_arith_context(
            &mut native_circuit,
            &parameters,
            2,
            nested_rns,
            Some(1),
            Some(0),
        ));
        let function = DiamondIoFunction::GoldreichPrf { output_bits: 1 };
        let mut config = super::super::DiamondIoConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count: parameters.modulus_digits(),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-io-runtime-test".to_vec(),
            seed_bits: 5,
            prf_mask_output_coeff_bits: 1,
            noise_refresh_v_bits: 1,
            noise_refresh_cbd_n: 2,
            noise_refresh_hash_key: rand::random(),
            goldreich_graph_seed: rand::random(),
            ring_gsw_width: ring_gsw.width(),
            ring_gsw_public_key_error_sigma: Some(RealExpr::from_f64_exact(0.125).unwrap()),
            refresh_crt_scale_factors: vec![BigInt::from(modulus.as_ref() / 2u8).into()],
            refresh_crt_plaintext_moduli: vec![2.into()],
            refresh_reconstruction_coefficients: vec![1.into()],
            refresh_decoder_public_columns: 2 * (parameters.modulus_digits() + 2),
        };
        config.seed_bits = config.minimum_goldreich_seed_bits(&function).unwrap();
        let compiler = DiamondIoCompiler::new(config.clone(), ring_gsw.clone()).unwrap();
        let mut runtime =
            TestRuntime::new(compiler, parameters.clone(), MemoryArtifactStore::default()).unwrap();
        let hash_key = rand::random();
        let native = sample_native_seed::<
            DCRTPoly,
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyUniformSampler,
        >(&parameters, &ring_gsw, config.seed_bits, hash_key, 0.125)
        .unwrap();
        let mut plaintext_seed = native.seed_bits.clone();
        let input = [rand::random::<bool>()];
        let obfuscation = runtime.obfuscate_with_native_seed(&function, hash_key, native).unwrap();

        let round_output_count = config.branch_count().unwrap() * config.seed_bits;
        let mut round_generator = GoldreichFullDomainRangeGenerator::new(
            config.seed_bits,
            round_output_count,
            super::super::circuits::goldreich_round_seed(
                config.goldreich_graph_seed,
                b"seed-refresh",
                0,
                None,
            ),
            GoldreichGraphGeneration::default(),
        );
        let round_graph = round_generator.next_range(0, round_output_count);
        let round_bits = evaluate_goldreich_bits(&round_graph, &plaintext_seed);
        let branch = usize::from(input[0]);
        plaintext_seed =
            round_bits[branch * config.seed_bits..(branch + 1) * config.seed_bits].to_vec();

        let [_, _, final_output_count] = config.goldreich_stream_sizes(&function).unwrap();
        let mut final_generator = GoldreichFullDomainRangeGenerator::new(
            config.seed_bits,
            final_output_count,
            super::super::circuits::goldreich_round_seed(
                config.goldreich_graph_seed,
                b"final-function-mask",
                config.round_count(),
                None,
            ),
            GoldreichGraphGeneration::default(),
        );
        let final_graph = final_generator.next_range(0, final_output_count);
        let final_bits = evaluate_goldreich_bits(&final_graph, &plaintext_seed);
        let mask_bits = config.ring_dimension * config.prf_mask_output_coeff_bits;
        let expected = vec![final_bits[mask_bits]];

        assert_eq!(runtime.evaluate_bits(&obfuscation, &input).unwrap(), expected);
    }
}
