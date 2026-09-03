#![cfg(any())]
// Diamond iO is temporarily disabled together with `mxx_io::diamond`.

use keccak_asm::Keccak256;
use mxx_gadgets::{
    circuit::PolyCircuit,
    circuit_gadgets::{
        arith::NestedRnsPolyContext,
        fhe::ring_gsw_nested_rns::NestedRnsRingGswContext,
        fhe_prg::goldreich::{
            GoldreichFullDomainRangeGenerator, GoldreichGraphGeneration, evaluate_goldreich_bits,
        },
    },
};
use mxx_io::diamond::{
    DiamondIoCompiler, DiamondIoConfig, DiamondIoFunction, DiamondIoParameterSearch,
    DiamondIoRuntime, goldreich_round_seed, sample_native_seed,
};
use mxx_ir_core::RealExpr;
use mxx_primitives::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
    },
    sampler::{
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::gpu::GpuDCRTPolyTrapdoorSampler,
    },
};
use mxx_runtime::artifact::MemoryArtifactStore;
use num_bigint::BigInt;
use std::{env, sync::Arc};

type GpuDiamondIoRuntime = DiamondIoRuntime<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
    MemoryArtifactStore,
>;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "full GPU Diamond iO parameter search and round trip is intentionally explicit"]
fn test_gpu_diamond_io_parameter_search_and_round_trip() {
    let function = DiamondIoFunction::GoldreichPrf { output_bits: 1 };
    let template = DiamondIoConfig {
        modulus: BigInt::from(257),
        ring_dimension: 8,
        input_count: 1,
        digit_base: 2,
        batch_bits: 1,
        gadget_base: BigInt::from(2),
        digit_count: 9,
        trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
        error_sigma: RealExpr::from_integer(1),
        preimage_max_coefficient_bound: 1_000_000.into(),
        bgg_tag: b"diamond-io-gpu-integration".to_vec(),
        seed_bits: 5,
        prf_mask_output_coeff_bits: 1,
        noise_refresh_v_bits: 1,
        noise_refresh_cbd_n: 2,
        noise_refresh_hash_key: [41; 32],
        goldreich_graph_seed: [43; 32],
        ring_gsw_width: 1,
        ring_gsw_public_key_error_sigma: Some(RealExpr::from_integer(1)),
        refresh_crt_scale_factors: vec![1.into()],
        refresh_crt_plaintext_moduli: vec![2.into()],
        refresh_reconstruction_coefficients: vec![1.into()],
        refresh_decoder_public_columns: 2,
    };
    let search = DiamondIoParameterSearch {
        template,
        min_crt_depth: 1,
        initial_max_crt_depth: 2,
        min_log_ring_dimension: env_usize("MXX_DIAMOND_IO_TEST_MIN_LOG_RING_DIM", 3),
        max_log_ring_dimension: env_usize("MXX_DIAMOND_IO_TEST_MAX_LOG_RING_DIM", 8),
        crt_modulus_bits: 20,
        gadget_base_bits: 4,
        security_bits: env_usize("MXX_DIAMOND_IO_TEST_SECURITY_BITS", 1),
        error_sigma: 1.0,
        native_ring_gsw_error_sigma: 1.0,
        nested_p_moduli_bits: 5,
        nested_max_unreduced_muls: 2,
        nested_scale: 16,
    };
    let selected = search.search(&function).expect("Diamond iO GPU parameter search");
    let (moduli, _, depth) = selected.parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(
        selected.parameters.ring_dimension(),
        moduli,
        selected.parameters.base_bits(),
    );
    let mut native_circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let nested_rns = Arc::new(NestedRnsPolyContext::setup(
        &mut native_circuit,
        &gpu_parameters,
        search.nested_p_moduli_bits,
        search.nested_max_unreduced_muls,
        search.nested_scale,
        false,
        Some(depth),
    ));
    let ring_gsw = Arc::new(NestedRnsRingGswContext::from_arith_context(
        &mut native_circuit,
        &gpu_parameters,
        gpu_parameters.ring_dimension() as usize,
        nested_rns,
        Some(depth),
        Some(0),
    ));
    let compiler = DiamondIoCompiler::new(selected.compiler.config.clone(), ring_gsw).unwrap();
    let mut runtime =
        GpuDiamondIoRuntime::new(compiler, gpu_parameters, MemoryArtifactStore::default()).unwrap();
    let native = sample_native_seed::<
        GpuDCRTPoly,
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyUniformSampler,
    >(
        &runtime.parameters,
        &runtime.compiler.ring_gsw,
        runtime.compiler.config.seed_bits,
        [47; 32],
        1.0,
    )
    .unwrap();
    let mut plaintext_seed = native.seed_bits().to_vec();
    let obfuscation = runtime.obfuscate_with_native_seed(&function, [47; 32], native).unwrap();
    let input = [true];
    let output = runtime.evaluate_bits(&obfuscation, &input).unwrap();
    let config = &runtime.compiler.config;
    let round_output_count = config.branch_count().unwrap() * config.seed_bits;
    let mut round_generator = GoldreichFullDomainRangeGenerator::new(
        config.seed_bits,
        round_output_count,
        goldreich_round_seed(config.goldreich_graph_seed, b"seed-refresh", 0, None),
        GoldreichGraphGeneration::default(),
    );
    let round_bits = evaluate_goldreich_bits(
        &round_generator.next_range(0, round_output_count),
        &plaintext_seed,
    );
    let branch = usize::from(input[0]);
    plaintext_seed =
        round_bits[branch * config.seed_bits..(branch + 1) * config.seed_bits].to_vec();
    let [_, _, final_output_count] = config.goldreich_stream_sizes(&function).unwrap();
    let mut final_generator = GoldreichFullDomainRangeGenerator::new(
        config.seed_bits,
        final_output_count,
        goldreich_round_seed(
            config.goldreich_graph_seed,
            b"final-function-mask",
            config.round_count(),
            None,
        ),
        GoldreichGraphGeneration::default(),
    );
    let final_bits = evaluate_goldreich_bits(
        &final_generator.next_range(0, final_output_count),
        &plaintext_seed,
    );
    let mask_bits = config.ring_dimension * config.prf_mask_output_coeff_bits;
    assert_eq!(output, vec![final_bits[mask_bits]]);
    gpu_device_sync();
}
