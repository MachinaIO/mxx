#[cfg(feature = "gpu")]
use keccak_asm::Keccak256;
#[cfg(feature = "gpu")]
use mxx::{
    agr16::{
        encoding::Agr16Encoding,
        public_key::Agr16PublicKey,
        sampler::{AGR16EncodingSampler, AGR16PublicKeySampler},
    },
    circuit::{PolyCircuit, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
        },
    },
    sampler::{
        DistType, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
    },
};
#[cfg(feature = "gpu")]
use std::{hint::black_box, time::Instant};
#[cfg(feature = "gpu")]
use tracing::info;

#[cfg(feature = "gpu")]
struct NoopAgr16PkPlt;

#[cfg(feature = "gpu")]
impl PltEvaluator<Agr16PublicKey<GpuDCRTPolyMatrix>> for NoopAgr16PkPlt {
    fn public_lookup(
        &self,
        _params: &<Agr16PublicKey<GpuDCRTPolyMatrix> as mxx::circuit::evaluable::Evaluable>::Params,
        _plt: &PublicLut<
            <Agr16PublicKey<GpuDCRTPolyMatrix> as mxx::circuit::evaluable::Evaluable>::P,
        >,
        _one: &Agr16PublicKey<GpuDCRTPolyMatrix>,
        _input: &Agr16PublicKey<GpuDCRTPolyMatrix>,
        _gate_id: GateId,
        _lut_id: usize,
    ) -> Agr16PublicKey<GpuDCRTPolyMatrix> {
        panic!("NoopAgr16PkPlt should not be called in this benchmark");
    }
}

#[cfg(feature = "gpu")]
struct NoopAgr16EncPlt;

#[cfg(feature = "gpu")]
impl PltEvaluator<Agr16Encoding<GpuDCRTPolyMatrix>> for NoopAgr16EncPlt {
    fn public_lookup(
        &self,
        _params: &<Agr16Encoding<GpuDCRTPolyMatrix> as mxx::circuit::evaluable::Evaluable>::Params,
        _plt: &PublicLut<
            <Agr16Encoding<GpuDCRTPolyMatrix> as mxx::circuit::evaluable::Evaluable>::P,
        >,
        _one: &Agr16Encoding<GpuDCRTPolyMatrix>,
        _input: &Agr16Encoding<GpuDCRTPolyMatrix>,
        _gate_id: GateId,
        _lut_id: usize,
    ) -> Agr16Encoding<GpuDCRTPolyMatrix> {
        panic!("NoopAgr16EncPlt should not be called in this benchmark");
    }
}

#[cfg(feature = "gpu")]
fn sample_fixture_with_aux_depth(
    input_size: usize,
    auxiliary_depth: usize,
    params: &GpuDCRTPolyParams,
) -> (
    Vec<Agr16PublicKey<GpuDCRTPolyMatrix>>,
    Vec<Agr16Encoding<GpuDCRTPolyMatrix>>,
    Vec<GpuDCRTPoly>,
    GpuDCRTPoly,
) {
    let key: [u8; 32] = rand::random();
    let tag: u64 = rand::random();
    let tag_bytes = tag.to_le_bytes();

    let pubkey_sampler =
        AGR16PublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(key, auxiliary_depth);
    let reveal_plaintexts = vec![true; input_size];
    let pubkeys = pubkey_sampler.sample(params, &tag_bytes, &reveal_plaintexts);

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secret = uniform_sampler.sample_poly(params, &DistType::TernaryDist);
    let secrets = vec![secret.clone()];
    let plaintexts = (0..input_size)
        .map(|_| uniform_sampler.sample_poly(params, &DistType::FinRingDist))
        .collect::<Vec<_>>();
    let encoding_sampler =
        AGR16EncodingSampler::<GpuDCRTPolyUniformSampler>::new(params, &secrets, None);
    let encodings = encoding_sampler.sample(params, &pubkeys, &plaintexts);

    (pubkeys, encodings, plaintexts, secret)
}

#[cfg(feature = "gpu")]
fn scalar_matrix(params: &GpuDCRTPolyParams, value: GpuDCRTPoly) -> GpuDCRTPolyMatrix {
    GpuDCRTPolyMatrix::from_poly_vec_row(params, vec![value])
}

#[cfg(feature = "gpu")]
fn assert_primary_auxiliary_invariants(
    encoding: &Agr16Encoding<GpuDCRTPolyMatrix>,
    secret: &GpuDCRTPoly,
) {
    assert!(
        !encoding.pubkey.c_times_s_pubkeys.is_empty() && !encoding.c_times_s_encodings.is_empty(),
        "AGR16 encoding must keep at least one recursive c_times_s level"
    );
    let expected_c_times_s = (encoding.pubkey.c_times_s_pubkeys[0].clone() * secret) +
        (encoding.vector.clone() * secret);
    assert_eq!(encoding.c_times_s_encodings[0], expected_c_times_s);
}

#[cfg(feature = "gpu")]
fn assert_eval_output_matches_equation_5_1(
    params: &GpuDCRTPolyParams,
    secret: &GpuDCRTPoly,
    pk_out: &Agr16PublicKey<GpuDCRTPolyMatrix>,
    enc_out: &Agr16Encoding<GpuDCRTPolyMatrix>,
    expected_plain: GpuDCRTPoly,
) {
    assert_eq!(enc_out.pubkey, *pk_out);
    let expected_ct = (scalar_matrix(params, secret.clone()) * pk_out.matrix.clone()) +
        scalar_matrix(params, expected_plain.clone());
    assert_eq!(enc_out.vector, expected_ct);
    assert_primary_auxiliary_invariants(enc_out, secret);
    assert_eq!(enc_out.plaintext, Some(expected_plain));
}

#[cfg(feature = "gpu")]
fn bench_agr16_complete_binary_tree_depth_env_probe_gpu() {
    gpu_device_sync();
    let _ = tracing_subscriber::fmt::try_init();

    let crt_bits = 52usize;
    let crt_depth = 9usize;
    let ring_dim = 1u32 << 14;
    let base_bits = (crt_bits / 2) as u32;
    let cpu_params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, base_bits);
    let (moduli, _, _) = cpu_params.to_crt();
    let params = GpuDCRTPolyParams::new(ring_dim, moduli, base_bits);

    let depth = std::env::var("MXX_AGR16_TREE_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let auxiliary_depth = std::env::var("MXX_AGR16_AUX_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(depth + 1);
    assert!(depth > 0, "MXX_AGR16_TREE_DEPTH must be positive");
    assert!(
        auxiliary_depth >= depth + 1,
        "MXX_AGR16_AUX_DEPTH must satisfy auxiliary_depth >= depth + 1"
    );

    let leaf_count = 1usize << depth;
    let (pubkeys, encodings, plaintexts, secret) =
        sample_fixture_with_aux_depth(leaf_count, auxiliary_depth, &params);

    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(leaf_count);
    let mut level = inputs.clone();
    while level.len() > 1 {
        level = level.chunks_exact(2).map(|pair| circuit.mul_gate(pair[0], pair[1])).collect();
    }
    circuit.output(vec![level[0]]);

    gpu_device_sync();
    let start = Instant::now();
    let pk_outputs = circuit.eval(
        &params,
        pubkeys[0].clone(),
        pubkeys.iter().skip(1).cloned().collect(),
        None::<&NoopAgr16PkPlt>,
    );
    let enc_outputs = circuit.eval(
        &params,
        encodings[0].clone(),
        encodings.iter().skip(1).cloned().collect(),
        None::<&NoopAgr16EncPlt>,
    );
    gpu_device_sync();
    let elapsed = start.elapsed();
    black_box((&pk_outputs, &enc_outputs));

    let expected_plain = plaintexts.iter().cloned().reduce(|acc, next| acc * next).unwrap();
    assert_eval_output_matches_equation_5_1(
        &params,
        &secret,
        &pk_outputs[0],
        &enc_outputs[0],
        expected_plain,
    );

    info!(
        "GPU AGR16 complete binary-tree env-probe benchmark: depth={}, aux_depth={}, params=(ring_dim={}, crt_depth={}, crt_bits={}, base_bits={}), elapsed={:?}",
        depth, auxiliary_depth, ring_dim, crt_depth, crt_bits, base_bits, elapsed
    );
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU AGR16 benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_agr16_complete_binary_tree_depth_env_probe_gpu();
}
