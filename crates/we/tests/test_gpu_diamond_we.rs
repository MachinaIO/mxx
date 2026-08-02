#![cfg(feature = "gpu")]

use keccak_asm::Keccak256;
use mxx_gadgets::circuit::PolyCircuit;
use mxx_ir_core::RealExpr;
use mxx_primitives::{
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            poly::DCRTPoly,
        },
    },
    sampler::{
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::gpu::GpuDCRTPolyTrapdoorSampler,
    },
};
use mxx_runtime::{ExecutionConfig, artifact::MemoryArtifactStore};
use mxx_we::diamond::{
    DiamondParameterSearch, DiamondWeCompiler, DiamondWeConfig, DiamondWeRuntime,
};
use num_bigint::BigInt;
use std::{env, num::NonZeroUsize};

type GpuDiamondRuntime = DiamondWeRuntime<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
    MemoryArtifactStore,
>;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn witness_circuit<P: Poly>(witness_size: usize) -> PolyCircuit<P> {
    let mut circuit = PolyCircuit::new();
    let witnesses = circuit.input(witness_size).gate_ids();
    let output = witnesses
        .into_iter()
        .reduce(|left, right| circuit.and_gate(left, right).as_single_wire())
        .expect("the GPU Diamond test uses a nonempty witness");
    circuit.output([output]);
    circuit
}

#[test]
#[ignore = "full GPU Diamond WE parameter search and round trip is intentionally explicit"]
fn test_gpu_diamond_we_error_search_and_round_trip() {
    let security_bits = env_usize("MXX_DIAMOND_TEST_SECURITY_BITS", 1);
    let witness_size = env_usize("MXX_DIAMOND_TEST_WITNESS_SIZE", 1);
    let min_log_ring_dimension = env_usize("MXX_DIAMOND_TEST_MIN_LOG_RING_DIM", 3);
    let max_log_ring_dimension = env_usize("MXX_DIAMOND_TEST_MAX_LOG_RING_DIM", 8);
    let input_count = witness_size;
    let cpu_circuit = witness_circuit::<DCRTPoly>(witness_size);
    let search = DiamondParameterSearch {
        min_crt_depth: 1,
        initial_max_crt_depth: 2,
        min_log_ring_dimension,
        max_log_ring_dimension,
        crt_modulus_bits: 20,
        gadget_base_bits: 4,
        security_bits,
        input_count,
        digit_base: 2,
        batch_bits: 1,
        trapdoor_sigma: 4.578,
        error_sigma: 1.0,
        bgg_tag: b"diamond-gpu-integration".to_vec(),
    };
    let selected = search.search(&cpu_circuit, &[]).expect("GPU Diamond parameter search");
    eprintln!(
        "selected Diamond GPU test parameters: ring_dimension={}, crt_depth={}, modulus_bits={}, security_bits={}",
        selected.ring_dimension,
        selected.crt_depth,
        selected.modulus.bits(),
        selected.achieved_security_bits,
    );
    let (moduli, _, _) = selected.parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(
        selected.parameters.ring_dimension(),
        moduli,
        selected.parameters.base_bits(),
    );
    let compiler = DiamondWeCompiler::new(DiamondWeConfig {
        modulus: BigInt::from(selected.modulus),
        ring_dimension: selected.ring_dimension as usize,
        input_count,
        digit_base: 2,
        batch_bits: 1,
        gadget_base: BigInt::from(1u64 << selected.parameters.base_bits()),
        digit_count: selected.parameters.modulus_digits(),
        trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
        error_sigma: RealExpr::from_f64_exact(1.0).unwrap(),
        bgg_tag: b"diamond-gpu-integration".to_vec(),
    })
    .unwrap();
    let parallelism = NonZeroUsize::new(env_usize("MXX_DIAMOND_TEST_PARALLELISM", 2)).unwrap();
    let mut runtime =
        GpuDiamondRuntime::new(compiler, gpu_parameters, MemoryArtifactStore::default())
            .unwrap()
            .with_execution_config(ExecutionConfig { max_parallel_instances: parallelism });
    let witness = vec![true; witness_size];
    let gpu_circuit = witness_circuit::<GpuDCRTPoly>(witness_size);
    let ciphertext = runtime
        .encrypt_with_hash_key(true, gpu_circuit, &[], [0x53; 32])
        .expect("GPU Diamond encryption");
    assert!(runtime.decrypt(&ciphertext, &witness).expect("GPU Diamond decryption"));
    gpu_device_sync();
}
