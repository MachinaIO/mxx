#![cfg(any())]
// AKY24 iO is temporarily disabled together with `mxx_io::aky24`.

use keccak_asm::Keccak256;
use mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::{
    GoldreichGraph, GoldreichGraphGeneration, evaluate_goldreich_bits,
};
use mxx_io::aky24::{Aky24GoldreichPrf, Aky24IoConfig, Aky24IoParameterSearch, Aky24IoRuntime};
use mxx_ir_core::RealExpr;
use mxx_primitives::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, gpu_device_sync},
    },
    sampler::{
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::gpu::GpuDCRTPolyTrapdoorSampler,
    },
};
use mxx_runtime::artifact::MemoryArtifactStore;
use num_bigint::BigInt;

type GpuAky24Runtime = Aky24IoRuntime<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
    MemoryArtifactStore,
>;

#[test]
#[ignore = "full GPU AKY24 iO parameter search and round trip is intentionally explicit"]
fn test_gpu_aky24_io_parameter_search_and_round_trip() {
    let function = Aky24GoldreichPrf { output_bits: 1, graph_seed: [29; 32] };
    let template = Aky24IoConfig {
        modulus: BigInt::from(257),
        ring_dimension: 8,
        input_size: 5,
        gadget_base: BigInt::from(2),
        digit_count: 9,
        modulus_split: BigInt::from(1),
        trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
        secret_sigma: RealExpr::from_integer(2),
        b_error_sigma: RealExpr::from_integer(1),
        fhe_error_sigma: RealExpr::from_integer(1),
        attribute_error_sigma: RealExpr::from_integer(1),
        preimage_max_coefficient_bound: BigInt::from(1u64 << 20),
        security_parameter_bits: 8,
        cascade_randomness_bits: 8,
        gaussian_sample_bits: 8,
        uniform_statistical_bits: 8,
        function: function.clone(),
    };
    let selected = Aky24IoParameterSearch {
        template,
        min_crt_depth: 1,
        initial_max_crt_depth: 2,
        min_log_ring_dimension: 3,
        max_log_ring_dimension: 16,
        crt_modulus_bits: 50,
        security_bits: 128,
    }
    .search()
    .expect("AKY24 GPU parameter search");
    let (moduli, _, _) = selected.parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(
        selected.parameters.ring_dimension(),
        moduli,
        selected.parameters.base_bits(),
    );
    let mut runtime =
        GpuAky24Runtime::new(selected.compiler, gpu_parameters, MemoryArtifactStore::default())
            .unwrap();
    let obfuscation = runtime.obfuscate_with_nonce(&function, [31; 32]).unwrap();
    let input = [false, true, false, true, true];
    let output = runtime.evaluate_bits(&obfuscation, &input).unwrap();
    let graph = GoldreichGraph::generate(
        input.len(),
        function.output_bits,
        function.graph_seed,
        GoldreichGraphGeneration::default(),
    );
    assert_eq!(output, evaluate_goldreich_bits(&graph, &input));
    gpu_device_sync();
}
