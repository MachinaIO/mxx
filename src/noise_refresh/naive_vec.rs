use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, PublicKeyAuxBenchEstimator,
        benchmark_gate_operation, estimate_public_key_circuit_bench_with_aux,
    },
    bgg::{
        encoding::BggEncoding,
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        public_key::BggPublicKey,
    },
    circuit::{BatchedWire, PolyCircuit, evaluable::Evaluable},
    decoder::{
        bench::{
            bit_decomposed_refresh_material_counts, bit_decomposed_refresh_material_summary,
            goldreich_cbd_error_prg_summary,
        },
        mask_circuit::build_one_ciphertext_bit_decrypt_circuit,
        masked_high_bit::decode_centered_masked_matrix,
    },
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPolyContext},
        fhe::{
            ring_gsw::{RingGswCiphertext, RingGswContext},
            ring_gsw_nested_rns::{
                ciphertext_inputs_from_native, encrypt_plaintext_bit, sample_public_key,
            },
        },
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    noise_refresh::{
        NoiseRefresher,
        circuit_decrypt::build_refreshed_wire_digit_all_crt_decrypt,
        circuit_merge::build_refreshed_wire_digit_all_crt_merge,
        circuit_prg::{
            build_goldreich_encrypted_seed_material_ranges,
            build_representative_goldreich_mask_material_circuit,
        },
    },
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyHashSampler},
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{marker::PhantomData, sync::Arc};
use tracing::{debug, info};

const NATIVE_BENCH_ITERATIONS: usize = 1;

pub fn debug_sample_prg_public_key_wires<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    secret_size: usize,
    label: &[u8],
    round_idx: usize,
    output_offset: usize,
    wire_count: usize,
    num_slots: usize,
) -> Vec<NaiveBGGPublicKeyVec<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let cols = secret_size * params.modulus_digits();
    (0..wire_count)
        .into_par_iter()
        .map(|wire_idx| {
            let hash_sampler = HS::new();
            let keys = (0..num_slots)
                .map(|slot_idx| {
                    let mut tag =
                        Vec::with_capacity(label.len() + 3 * std::mem::size_of::<usize>() + 1);
                    tag.extend_from_slice(label);
                    tag.extend_from_slice(&round_idx.to_le_bytes());
                    tag.extend_from_slice(&output_offset.to_le_bytes());
                    tag.extend_from_slice(&wire_idx.to_le_bytes());
                    tag.extend_from_slice(&slot_idx.to_le_bytes());
                    BggPublicKey::new(
                        hash_sampler.sample_hash(
                            params,
                            hash_key,
                            &tag,
                            secret_size,
                            cols,
                            DistType::FinRingDist,
                        ),
                        true,
                    )
                })
                .collect::<Vec<_>>();
            NaiveBGGPublicKeyVec::new(params, keys)
        })
        .collect()
}

pub fn debug_sample_prg_plaintext_wires<P>(
    params: &P::Params,
    native_params: &DCRTPolyParams,
    ring_gsw_context: &NestedRnsPolyContext,
    ring_gsw_width: usize,
    native_fhe_decryption_key: &DCRTPoly,
    hash_key: [u8; 32],
    label: &[u8],
    wire_count: usize,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<crate::circuit::evaluable::PolyVec<P>>
where
    P: Poly,
{
    let public_key = sample_public_key(
        native_params,
        ring_gsw_width,
        native_fhe_decryption_key,
        hash_key,
        label,
        None,
    );
    (0..wire_count)
        .flat_map(|_| {
            let bit = rand::random::<bool>();
            let ciphertext =
                encrypt_plaintext_bit(native_params, ring_gsw_context, &public_key, bit);
            ciphertext_inputs_from_native::<P>(
                params,
                ring_gsw_context,
                &ciphertext,
                level_offset,
                enable_levels,
            )
        })
        .take(wire_count)
        .collect()
}

pub fn debug_sample_prg_encoding_wires<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    secret: &[M::P],
    plaintexts: &[crate::circuit::evaluable::PolyVec<M::P>],
    label: &[u8],
    round_idx: usize,
    output_offset: usize,
    num_slots: usize,
) -> Vec<NaiveBGGEncodingVec<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let public_keys = debug_sample_prg_public_key_wires::<M, HS>(
        params,
        hash_key,
        secret.len(),
        label,
        round_idx,
        output_offset,
        plaintexts.len(),
        num_slots,
    );
    let secret_vec = M::from_poly_vec_row(params, secret.to_vec());
    let secret_gadget = secret_vec.clone() * M::gadget_matrix(params, secret.len());
    let secret_vec_bytes = Arc::<[u8]>::from(secret_vec.into_compact_bytes());
    let secret_gadget_bytes = Arc::<[u8]>::from(secret_gadget.into_compact_bytes());
    plaintexts
        .par_iter()
        .zip(public_keys.into_par_iter())
        .map(|(plaintext, public_key_vec)| {
            let secret_vec = M::from_compact_bytes(params, secret_vec_bytes.as_ref());
            let secret_gadget = M::from_compact_bytes(params, secret_gadget_bytes.as_ref());
            assert_eq!(
                plaintext.len(),
                num_slots,
                "debug PRG plaintext slot count must match public-key slot count"
            );
            let encodings = (0..num_slots)
                .map(|slot_idx| {
                    let public_key = public_key_vec.key(slot_idx);
                    let slot_plaintext = plaintext.as_slice()[slot_idx].clone();
                    let encoded_plaintext =
                        M::from_poly_vec_row(params, vec![slot_plaintext.clone()]);
                    let vector = secret_vec.clone() * public_key.matrix.clone() -
                        encoded_plaintext.tensor(&secret_gadget);
                    BggEncoding::new(vector, public_key, Some(slot_plaintext))
                })
                .collect::<Vec<_>>();
            NaiveBGGEncodingVec::new(params, encodings)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NoiseRefreshBenchEstimateParts {
    pub(crate) material: CircuitBenchSummary,
    pub(crate) per_refresh: CircuitBenchSummary,
    pub(crate) total: CircuitBenchSummary,
}

fn scaled_estimate_summary(unit: CircuitBenchEstimate, task_count: usize) -> CircuitBenchSummary {
    let total_time = unit.total_time.clone() * BigUint::from(task_count);
    let max_parallelism = unit.max_parallelism.clone() * BigUint::from(task_count);
    let summary = CircuitBenchSummary::from_nanos(total_time, unit.latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(unit.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn estimate_summary(unit: CircuitBenchEstimate) -> CircuitBenchSummary {
    scaled_estimate_summary(unit, 1)
}

fn refresh_slot_id(refresh_id: &[u8], label: &[u8], idx: usize) -> Vec<u8> {
    let mut id = Vec::with_capacity(refresh_id.len() + label.len() + std::mem::size_of::<usize>());
    id.extend_from_slice(refresh_id);
    id.extend_from_slice(label);
    id.extend_from_slice(&idx.to_le_bytes());
    id
}

fn debug_sample_prg_output_plaintexts<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    label: &[u8],
    wire_count: usize,
    slot_count: usize,
) -> Vec<M>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    (0..wire_count)
        .into_par_iter()
        .map(|wire_idx| {
            let hash_sampler = HS::new();
            let mut tag = Vec::with_capacity(label.len() + std::mem::size_of::<usize>());
            tag.extend_from_slice(label);
            tag.extend_from_slice(&wire_idx.to_le_bytes());
            hash_sampler.sample_hash(params, hash_key, &tag, 1, slot_count, DistType::BitDist)
        })
        .collect()
}

fn debug_sample_prg_output_public_wires<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    label: &[u8],
    wire_count: usize,
    slot_count: usize,
    one: &NaiveBGGPublicKeyVec<M>,
) -> Vec<NaiveBGGPublicKeyVec<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    debug_sample_prg_output_plaintexts::<M, HS>(params, hash_key, label, wire_count, slot_count)
        .into_par_iter()
        .map(|plaintexts| {
            NaiveBGGPublicKeyVec::new(
                params,
                (0..slot_count)
                    .map(|slot_idx| {
                        one.key(slot_idx).large_scalar_mul(
                            params,
                            &plaintexts.entry(0, slot_idx).coeffs_biguints(),
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}

fn debug_sample_prg_output_encoding_wires<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    label: &[u8],
    wire_count: usize,
    slot_count: usize,
    one: &NaiveBGGEncodingVec<M>,
) -> Vec<NaiveBGGEncodingVec<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    debug_sample_prg_output_plaintexts::<M, HS>(params, hash_key, label, wire_count, slot_count)
        .into_par_iter()
        .map(|plaintexts| {
            NaiveBGGEncodingVec::new(
                params,
                (0..slot_count)
                    .map(|slot_idx| {
                        one.encoding(slot_idx).large_scalar_mul(
                            params,
                            &plaintexts.entry(0, slot_idx).coeffs_biguints(),
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}

fn scaled_summary(summary: CircuitBenchSummary, task_count: usize) -> CircuitBenchSummary {
    let total_time = summary.total_time.clone() * BigUint::from(task_count);
    let max_parallelism = summary.max_parallelism.clone() * BigUint::from(task_count);
    let scaled = CircuitBenchSummary::from_nanos(total_time, summary.latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

fn expand_batched_wires(wires: &[BatchedWire]) -> Vec<BatchedWire> {
    wires.iter().flat_map(|wire| (0..wire.len()).map(|idx| wire.at(idx))).collect()
}

fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time.clone()).sum::<BigUint>();
    let latency = parts.iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism =
        parts.iter().map(|part| part.max_parallelism.clone()).max().unwrap_or_default();
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.iter().map(|part| part.peak_vram).max().unwrap_or(0))
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn measured_crt_recompose_unit_summary<M>(params: &<M::P as Poly>::Params) -> CircuitBenchSummary
where
    M: PolyMatrix,
{
    let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
    let q: Arc<BigUint> = params.modulus().into();
    let log_base_q = params.modulus_digits();
    let ring_dim = params.ring_dimension() as usize;
    let crt_values = q_moduli
        .iter()
        .take(crt_depth)
        .enumerate()
        .map(|(crt_idx, &q_i)| {
            let scale = q.as_ref() / BigUint::from(q_i);
            let row = (0..log_base_q)
                .map(|digit_idx| {
                    let coeffs = (0..ring_dim)
                        .map(|coeff_idx| {
                            let base = BigUint::from((1 + crt_idx + digit_idx + coeff_idx) as u64);
                            (base * &scale) % q.as_ref()
                        })
                        .collect::<Vec<_>>();
                    M::P::from_biguints(params, &coeffs)
                })
                .collect::<Vec<_>>();
            M::from_poly_vec_row(params, row)
        })
        .collect::<Vec<_>>();
    let bench = benchmark_gate_operation(NATIVE_BENCH_ITERATIONS, || {
        crt_recompose_rows::<M>(params, &crt_values, 1)
    });
    let summary = CircuitBenchSummary::new(bench.time, bench.time, 1u32);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(bench.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn measured_a_prime_hash_sampling_unit_summary<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    secret_size: usize,
) -> CircuitBenchSummary
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let log_base_q = params.modulus_digits();
    let bench = benchmark_gate_operation(NATIVE_BENCH_ITERATIONS, || {
        let matrix = HS::new().sample_hash(
            params,
            hash_key,
            b"noise-refresh-bench-a-prime",
            secret_size,
            secret_size
                .checked_mul(log_base_q)
                .expect("noise-refresh a_prime benchmark column count overflow"),
            DistType::FinRingDist,
        );
        matrix.into_compact_bytes()
    });
    let summary = CircuitBenchSummary::new(bench.time, bench.time, 1u32);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(bench.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

/// Naive-vector implementation of one-wire noise refresh.
///
/// The type is intentionally specialized to `NaiveBGGPublicKeyVec` and `NaiveBGGEncodingVec`.
/// Those containers expose independent scalar BGG objects per logical slot, which lets this
/// implementation collapse decoded slotwise refresh material by direct matrix arithmetic instead
/// of forcing the generic `Evaluable` API to model every native CRT recomposition step.
#[derive(Clone)]
pub struct NoiseRefresherNaiveVec<M, A, HS>
where
    M: PolyMatrix,
    A: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
{
    ring_gsw: Arc<RingGswContext<M::P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    hash_key: [u8; 32],
    debug_reuse_single_material: bool,
    material_circuit: Arc<PolyCircuit<M::P>>,
    _hash_sampler: PhantomData<HS>,
}

impl<M, A, HS> NoiseRefresherNaiveVec<M, A, HS>
where
    M: PolyMatrix,
    M::P: 'static,
    A: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
{
    pub fn new(
        ring_gsw: Arc<RingGswContext<M::P, A>>,
        seed_bits: usize,
        v_bits: usize,
        graph_seed: [u8; 32],
        cbd_n: usize,
        hash_key: [u8; 32],
    ) -> Self {
        assert!(seed_bits > 0, "seed_bits must be positive");
        assert!(v_bits > 0, "v_bits must be positive");
        assert!(cbd_n > 0, "cbd_n must be positive");
        let debug_reuse_single_material = cfg!(test);
        let num_slots = ring_gsw.params.ring_dimension() as usize;
        let material_circuit = Arc::new(build_noise_refresh_material_circuit::<M::P, A, M>(
            ring_gsw.clone(),
            seed_bits,
            v_bits,
            graph_seed,
            cbd_n,
            num_slots,
            debug_reuse_single_material,
        ));
        Self {
            ring_gsw,
            seed_bits,
            v_bits,
            graph_seed,
            cbd_n,
            hash_key,
            debug_reuse_single_material,
            material_circuit,
            _hash_sampler: PhantomData,
        }
    }

    pub fn with_debug_reuse_single_material(mut self, enabled: bool) -> Self {
        self.debug_reuse_single_material = cfg!(test) && enabled;
        let num_slots = self.ring_gsw.params.ring_dimension() as usize;
        self.material_circuit = Arc::new(build_noise_refresh_material_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            self.seed_bits,
            self.v_bits,
            self.graph_seed,
            self.cbd_n,
            num_slots,
            self.debug_reuse_single_material,
        ));
        self
    }

    fn params(&self) -> &<M::P as Poly>::Params {
        &self.ring_gsw.params
    }

    /// Estimates the preprocessing side of the current naive-vector implementation.
    ///
    /// This models the same work as `preprocess`: evaluate the PRG/decrypt/merge material circuit
    /// under public-key evaluators, transform each decoded material output into a one-column
    /// contribution, collapse the per-slot matrices, and combine those columns with the
    /// `NaiveBGGPublicKeyVec` one and refreshed-input public keys.  The returned value is only the
    /// total preprocessing summary; stage estimates are emitted with `debug!` for inspection.
    ///
    /// `BenchEstimator` does not expose an arbitrary `matrix_mul(rhs_matrix)` primitive, so this
    /// model uses scalar BGG `estimate_large_scalar_mul(&[1])` as the cost proxy for each public
    /// matrix multiplication.  Both operations are dominated by gadget decomposition of the
    /// right-hand target.
    pub fn estimate_preprocess_bench<PKBE>(
        &self,
        public_key_estimator: &PKBE,
    ) -> CircuitBenchSummary
    where
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + PublicKeyAuxBenchEstimator<M::P> + Sync,
        HS: PolyHashSampler<[u8; 32], M = M>,
    {
        self.estimate_preprocess_bench_parts(public_key_estimator).total
    }

    pub(crate) fn estimate_preprocess_bench_parts<PKBE>(
        &self,
        public_key_estimator: &PKBE,
    ) -> NoiseRefreshBenchEstimateParts
    where
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + PublicKeyAuxBenchEstimator<M::P> + Sync,
        HS: PolyHashSampler<[u8; 32], M = M>,
    {
        let num_slots = self.params().ring_dimension() as usize;
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        // The naive-vector refresh paths used by AKY24 and DiamondIO refresh scalar BGG wires:
        // `one.key(0).matrix.row_size()` and `refreshed_input.key(0).matrix.row_size()` are one in
        // those call sites. The estimator has no concrete `one` vector, so it measures the same
        // scalar `A'` hash shape directly.
        let secret_size = 1usize;
        let combine_task_count =
            num_slots.checked_mul(crt_depth).expect("noise-refresh combine task count overflow");

        let scalar_target = [BigUint::from(1u32)];
        let pk_matrix_mul = public_key_estimator.estimate_large_scalar_mul(&scalar_target);
        let pk_add = public_key_estimator.estimate_add();
        let pk_sub = public_key_estimator.estimate_sub();
        let material_counts = bit_decomposed_refresh_material_counts(
            num_slots,
            log_base_q,
            crt_depth,
            self.v_bits,
            self.debug_reuse_single_material,
        );
        let mask_prg_circuit =
            build_representative_goldreich_mask_material_circuit::<M::P, A>(self.ring_gsw.clone());
        let decrypt_contribution_circuit = build_one_ciphertext_bit_decrypt_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            BigUint::from(2u64),
        );
        let mask_prg_unit = estimate_public_key_circuit_bench_with_aux::<
            NaiveBGGPublicKeyVec<M>,
            PKBE,
        >(public_key_estimator, self.params(), &mask_prg_circuit);
        let error_prg_unit = goldreich_cbd_error_prg_summary(
            mask_prg_unit.clone(),
            pk_add.clone(),
            pk_sub.clone(),
            self.cbd_n,
        );
        let decrypt_contribution_unit =
            estimate_public_key_circuit_bench_with_aux::<NaiveBGGPublicKeyVec<M>, PKBE>(
                public_key_estimator,
                self.params(),
                &decrypt_contribution_circuit,
            );
        let public_material_summary = bit_decomposed_refresh_material_summary(
            error_prg_unit.clone(),
            mask_prg_unit.clone(),
            decrypt_contribution_unit.clone(),
            pk_add.clone(),
            material_counts,
            self.v_bits,
        );
        let a_prime_sampling_unit = measured_a_prime_hash_sampling_unit_summary::<M, HS>(
            self.params(),
            self.hash_key,
            secret_size,
        );
        let a_prime_sampling_stage = scaled_summary(a_prime_sampling_unit.clone(), num_slots);
        let collapse_add_count = log_base_q
            .checked_mul(num_slots.saturating_sub(1))
            .expect("noise-refresh collapse add count overflow");
        let preprocess_combine_unit = sequential_summaries(&[
            // Compute the public-key one-term for the naive slot vector:
            // `one.keys[slot_idx].matrix_mul((q / q_i) * A')`.
            estimate_summary(pk_matrix_mul.clone()),
            // Compute the refreshed-input public-key term for the naive slot vector:
            // `refreshed_input.keys[slot_idx].matrix_mul((q / q_i) * G)`.
            estimate_summary(pk_matrix_mul.clone()),
            // Apply the one-column target to each of the `log_base_q` decoded material columns.
            scaled_estimate_summary(pk_matrix_mul.clone(), log_base_q),
            // For each decoded material column, `collapse_slot_matrices` sums `num_slots`
            // rotated slot matrices into one polynomial column; `concat_owned_columns` then only
            // lays those columns side by side and is not modeled as arithmetic.
            scaled_estimate_summary(pk_add.clone(), collapse_add_count),
            // Add the refreshed-input term and the decoded refresh term.
            estimate_summary(pk_add.clone()),
            // Subtract the one-term so the public refresh matrix has the intended sign.
            estimate_summary(pk_sub.clone()),
        ]);

        let preprocess_combine_stage = sequential_summaries(&[
            a_prime_sampling_stage.clone(),
            scaled_summary(preprocess_combine_unit.clone(), combine_task_count),
        ]);
        let preprocess_summary = sequential_summaries(&[
            public_material_summary.clone(),
            preprocess_combine_stage.clone(),
        ]);

        debug!(
            v_bits = self.v_bits,
            num_slots,
            crt_depth,
            log_base_q,
            combine_task_count,
            ?material_counts,
            ?a_prime_sampling_unit,
            ?a_prime_sampling_stage,
            ?error_prg_unit,
            ?mask_prg_unit,
            ?decrypt_contribution_unit,
            ?public_material_summary,
            ?preprocess_combine_unit,
            ?preprocess_combine_stage,
            ?preprocess_summary,
            "estimated naive-vector noise-refresh preprocess benchmark"
        );

        NoiseRefreshBenchEstimateParts {
            material: public_material_summary,
            per_refresh: preprocess_combine_stage,
            total: preprocess_summary,
        }
    }

    /// Estimates the online-evaluation side of the current naive-vector implementation.
    ///
    /// This models the same work as `online_eval`: evaluate the PRG/decrypt/merge material circuit
    /// under encoding evaluators, combine each `(slot_idx, crt_idx)` decoded material vector with
    /// the `NaiveBGGEncodingVec` one, refreshed input, and decoder terms, then round and CRT
    /// recompose one final row per slot.  The returned value is only the total online summary;
    /// stage estimates are emitted with `debug!` for inspection.
    pub fn estimate_online_eval_bench<EncBE>(
        &self,
        encoding_estimator: &EncBE,
    ) -> CircuitBenchSummary
    where
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        HS: PolyHashSampler<[u8; 32], M = M>,
    {
        self.estimate_online_eval_bench_parts(encoding_estimator).total
    }

    pub(crate) fn estimate_online_eval_bench_parts<EncBE>(
        &self,
        encoding_estimator: &EncBE,
    ) -> NoiseRefreshBenchEstimateParts
    where
        EncBE: BenchEstimator<NaiveBGGEncodingVec<M>> + Sync,
        HS: PolyHashSampler<[u8; 32], M = M>,
    {
        let num_slots = self.params().ring_dimension() as usize;
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        // See the preprocessing estimator above: online refresh recomputes the same scalar `A'`
        // matrices from `(refresh_id, slot_idx)` and therefore pays the same hash-sampling unit.
        let secret_size = 1usize;
        let combine_task_count =
            num_slots.checked_mul(crt_depth).expect("noise-refresh combine task count overflow");

        let scalar_target = [BigUint::from(1u32)];
        let enc_matrix_mul = encoding_estimator.estimate_large_scalar_mul(&scalar_target);
        let enc_add = encoding_estimator.estimate_add();
        let enc_sub = encoding_estimator.estimate_sub();
        let material_counts = bit_decomposed_refresh_material_counts(
            num_slots,
            log_base_q,
            crt_depth,
            self.v_bits,
            self.debug_reuse_single_material,
        );
        let mask_prg_circuit =
            build_representative_goldreich_mask_material_circuit::<M::P, A>(self.ring_gsw.clone());
        let decrypt_contribution_circuit = build_one_ciphertext_bit_decrypt_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            BigUint::from(2u64),
        );
        let mask_prg_unit = encoding_estimator.estimate_circuit_bench(&mask_prg_circuit);
        let error_prg_unit = goldreich_cbd_error_prg_summary(
            mask_prg_unit.clone(),
            enc_add.clone(),
            enc_sub.clone(),
            self.cbd_n,
        );
        let decrypt_contribution_unit =
            encoding_estimator.estimate_circuit_bench(&decrypt_contribution_circuit);
        let online_material_summary = bit_decomposed_refresh_material_summary(
            error_prg_unit.clone(),
            mask_prg_unit.clone(),
            decrypt_contribution_unit.clone(),
            enc_add.clone(),
            material_counts,
            self.v_bits,
        );
        let a_prime_sampling_unit = measured_a_prime_hash_sampling_unit_summary::<M, HS>(
            self.params(),
            self.hash_key,
            secret_size,
        );
        let a_prime_sampling_stage = scaled_summary(a_prime_sampling_unit.clone(), num_slots);
        let collapse_add_count = log_base_q
            .checked_mul(num_slots.saturating_sub(1))
            .expect("noise-refresh collapse add count overflow");
        let online_combine_unit = sequential_summaries(&[
            // Compute the encoding one-term for the naive slot vector:
            // `one.encodings[slot_idx].matrix_mul((q / q_i) * A')`.
            estimate_summary(enc_matrix_mul.clone()),
            // Compute the refreshed-input encoding term for the naive slot vector:
            // `refreshed_input.encodings[slot_idx].matrix_mul((q / q_i) * G)`.
            estimate_summary(enc_matrix_mul.clone()),
            // Apply the one-column target to each of the `log_base_q` decoded material columns.
            scaled_estimate_summary(enc_matrix_mul.clone(), log_base_q),
            // For each decoded material column, `collapse_slot_matrices` sums `num_slots`
            // rotated slot vectors into one polynomial column; `concat_owned_columns` then only
            // lays those columns side by side and is not modeled as arithmetic.
            scaled_estimate_summary(enc_add.clone(), collapse_add_count),
            // Add the refreshed-input term and the decoded refresh term.
            estimate_summary(enc_add.clone()),
            // Subtract the one-term to leave the desired positive `A'` contribution after
            // decoding.
            estimate_summary(enc_sub.clone()),
            // Subtract the caller-provided decoder for this `(slot_idx, crt_idx)` level.
            estimate_summary(enc_sub.clone()),
        ]);

        // Measure the native `crt_recompose_rows` work for one final slot row directly.  This is
        // not a BGG encoding gate: it performs coefficient rounding, reconstruction-coefficient
        // multiplication, and CRT-level accumulation on raw `M` matrices.
        let online_crt_recompose_unit = measured_crt_recompose_unit_summary::<M>(self.params());

        let online_combine_stage = sequential_summaries(&[
            a_prime_sampling_stage.clone(),
            scaled_summary(online_combine_unit.clone(), combine_task_count),
        ]);
        let online_crt_stage = scaled_summary(online_crt_recompose_unit.clone(), num_slots);
        let online_per_refresh =
            sequential_summaries(&[online_combine_stage.clone(), online_crt_stage.clone()]);
        let online_summary =
            sequential_summaries(&[online_material_summary.clone(), online_per_refresh.clone()]);

        debug!(
            v_bits = self.v_bits,
            num_slots,
            crt_depth,
            log_base_q,
            combine_task_count,
            crt_recompose_task_count = num_slots,
            ?material_counts,
            ?a_prime_sampling_unit,
            ?a_prime_sampling_stage,
            ?error_prg_unit,
            ?mask_prg_unit,
            ?decrypt_contribution_unit,
            ?online_material_summary,
            ?online_combine_unit,
            ?online_combine_stage,
            ?online_crt_recompose_unit,
            ?online_crt_stage,
            ?online_summary,
            "estimated naive-vector noise-refresh online benchmark"
        );

        NoiseRefreshBenchEstimateParts {
            material: online_material_summary,
            per_refresh: online_per_refresh,
            total: online_summary,
        }
    }
}

impl<M, A, HS> NoiseRefresherNaiveVec<M, A, HS>
where
    M: PolyMatrix + 'static,
    M::P: 'static,
    A: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub fn preprocess_many<PE, ST>(
        &self,
        refresh_ids: &[Vec<u8>],
        one: &NaiveBGGPublicKeyVec<M>,
        refreshed_inputs: &[NaiveBGGPublicKeyVec<M>],
        enc_seeds: &[NaiveBGGPublicKeyVec<M>],
        decryption_key: &NaiveBGGPublicKeyVec<M>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Vec<(NaiveBGGPublicKeyVec<M>, Vec<NaiveBGGPublicKeyVec<M>>)>
    where
        PE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    {
        info!(
            target: "mxx::func_enc::aky24",
            refresh_id_count = refresh_ids.len(),
            refreshed_input_count = refreshed_inputs.len(),
            seed_wire_count = enc_seeds.len(),
            "naive-vector noise-refresh preprocess_many entered"
        );
        assert_eq!(
            refresh_ids.len(),
            refreshed_inputs.len(),
            "refresh id count must match refreshed input count"
        );
        let num_slots = refreshed_inputs
            .first()
            .expect("preprocess_many requires at least one refreshed input")
            .num_slots();
        assert_eq!(
            num_slots,
            self.params().ring_dimension() as usize,
            "Naive noise refresh currently expects one logical slot per ring coefficient"
        );
        debug!(
            target: "mxx::func_enc::aky24",
            "naive-vector noise-refresh preprocess_many checking refreshed input slots"
        );
        refreshed_inputs.par_iter().for_each(|input| one.assert_compatible(input));
        one.assert_compatible(decryption_key);
        debug!(
            target: "mxx::func_enc::aky24",
            "naive-vector noise-refresh preprocess_many checking encrypted seed slots"
        );
        enc_seeds.par_iter().for_each(|seed| one.assert_compatible(seed));

        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        let secret_size = one.key(0).matrix.row_size();
        if self.debug_reuse_single_material {
            info!(
                target: "mxx::func_enc::aky24",
                num_slots,
                crt_depth,
                log_base_q,
                "naive-vector noise-refresh preprocess_many sampling debug PRG-output material"
            );
            let debug_decode_circuit = build_noise_refresh_debug_material_decode_circuit::<
                M::P,
                A,
                M,
            >(self.ring_gsw.clone(), self.v_bits);
            let material_input_count = debug_decode_circuit.num_input() - 1;
            assert_eq!(
                material_input_count % 2,
                0,
                "debug noise-refresh material input count must split into error and mask halves"
            );
            let per_material_input_count = material_input_count / 2;
            let error_wire = debug_sample_prg_output_public_wires::<M, HS>(
                self.params(),
                self.hash_key,
                b"noise-refresh-debug-error-material",
                1,
                num_slots,
                one,
            )
            .into_iter()
            .next()
            .expect("debug noise-refresh must sample one reusable error wire");
            let mask_wire = debug_sample_prg_output_public_wires::<M, HS>(
                self.params(),
                self.hash_key,
                b"noise-refresh-debug-mask-material",
                1,
                num_slots,
                one,
            )
            .into_iter()
            .next()
            .expect("debug noise-refresh must sample one reusable mask wire");
            let mut inputs = Vec::with_capacity(material_input_count + 1);
            inputs.extend(std::iter::repeat_n(error_wire, per_material_input_count));
            inputs.extend(std::iter::repeat_n(mask_wire, per_material_input_count));
            inputs.push(decryption_key.clone());
            let decoded = debug_decode_circuit.eval(
                self.params(),
                one.clone(),
                inputs,
                Some(plt_evaluator),
                Some(
                    slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
                ),
                None,
            );
            let decoded_row_size = decoded
                .first()
                .and_then(|value| (value.num_slots() > 0).then(|| value.key(0).matrix.row_size()))
                .expect("debug noise-refresh decoded public material must be nonempty");
            let unit_column_target =
                M::unit_column_vector(self.params(), decoded_row_size, decoded_row_size - 1);
            let decoded = decoded
                .into_iter()
                .map(|value| value.matrix_mul(self.params(), &unit_column_target))
                .collect::<Vec<_>>();
            let decoded_refresh_terms = decoded_refresh_terms_public(
                self.params(),
                &decoded,
                num_slots,
                crt_depth,
                log_base_q,
                secret_size,
            );
            return refresh_ids
                .par_iter()
                .zip(refreshed_inputs.par_iter())
                .map(|(refresh_id, refreshed_input)| {
                    self.preprocess_from_decoded(
                        refresh_id,
                        one,
                        refreshed_input,
                        &decoded_refresh_terms,
                    )
                })
                .collect();
        }

        debug!(
            target: "mxx::func_enc::aky24",
            seed_wire_count = enc_seeds.len(),
            refreshed_input_count = refreshed_inputs.len(),
            "naive-vector noise-refresh preprocess_many cloning encrypted seed inputs"
        );
        let material_seed_bits = if self.debug_reuse_single_material { 5 } else { self.seed_bits };
        let seed_wire_count = enc_seeds
            .len()
            .checked_div(self.seed_bits)
            .expect("noise-refresh seed bit count must be positive");
        assert_eq!(
            seed_wire_count * self.seed_bits,
            enc_seeds.len(),
            "noise-refresh encrypted seed wire count must be divisible by seed_bits"
        );
        let material_seed_wire_count = material_seed_bits
            .checked_mul(seed_wire_count)
            .expect("noise-refresh material seed wire count overflow");
        info!(
            target: "mxx::func_enc::aky24",
            num_slots,
            seed_bits = self.seed_bits,
            material_seed_bits,
            v_bits = self.v_bits,
            cbd_n = self.cbd_n,
            "naive-vector noise-refresh preprocess_many preparing decoded material"
        );
        let started = std::time::Instant::now();
        let mut inputs =
            enc_seeds[..material_seed_wire_count].par_iter().cloned().collect::<Vec<_>>();
        inputs.push(decryption_key.clone());
        info!(
            target: "mxx::func_enc::aky24",
            input_count = inputs.len(),
            output_count = self.material_circuit.num_output(),
            "naive-vector noise-refresh preprocess_many evaluating cached material circuit once"
        );
        let decoded = self.material_circuit.eval(
            self.params(),
            one.clone(),
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>),
            None,
        );
        debug!(
            target: "mxx::func_enc::aky24",
            decoded_count = decoded.len(),
            refreshed_input_count = refreshed_inputs.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "naive-vector noise-refresh preprocess_many material circuit evaluated"
        );
        if let Some(first_decoded) = decoded.first() {
            if first_decoded.num_slots() > 0 {
                let first_key = first_decoded.key(0);
                info!(
                    target: "mxx::func_enc::aky24",
                    decoded_slots = first_decoded.num_slots(),
                    decoded_rows = first_key.matrix.row_size(),
                    decoded_cols = first_key.matrix.col_size(),
                    "naive-vector noise-refresh preprocess_many first decoded material shape"
                );
            }
        }
        let decoded_row_size = decoded
            .first()
            .and_then(|value| (value.num_slots() > 0).then(|| value.key(0).matrix.row_size()))
            .expect("noise-refresh decoded public material must be nonempty");
        let unit_column_target =
            M::unit_column_vector(self.params(), decoded_row_size, decoded_row_size - 1);
        let decoded = decoded
            .into_iter()
            .map(|value| value.matrix_mul(self.params(), &unit_column_target))
            .collect::<Vec<_>>();
        let decoded_refresh_terms = decoded_refresh_terms_public(
            self.params(),
            &decoded,
            num_slots,
            crt_depth,
            log_base_q,
            secret_size,
        );
        refresh_ids
            .par_iter()
            .zip(refreshed_inputs.par_iter())
            .map(|(refresh_id, refreshed_input)| {
                self.preprocess_from_decoded(
                    refresh_id,
                    one,
                    refreshed_input,
                    &decoded_refresh_terms,
                )
            })
            .collect()
    }

    pub fn online_eval_many<PE, ST>(
        &self,
        refresh_ids: &[Vec<u8>],
        one: &NaiveBGGEncodingVec<M>,
        refreshed_inputs: &[NaiveBGGEncodingVec<M>],
        enc_seeds: &[NaiveBGGEncodingVec<M>],
        decryption_key: &NaiveBGGEncodingVec<M>,
        decoder_sets: &[Vec<M>],
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Vec<NaiveBGGEncodingVec<M>>
    where
        PE: PltEvaluator<NaiveBGGEncodingVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
    {
        info!(
            target: "mxx::func_enc::aky24",
            refresh_id_count = refresh_ids.len(),
            refreshed_input_count = refreshed_inputs.len(),
            seed_wire_count = enc_seeds.len(),
            decoder_set_count = decoder_sets.len(),
            "naive-vector noise-refresh online_eval_many entered"
        );
        assert_eq!(
            refresh_ids.len(),
            refreshed_inputs.len(),
            "refresh id count must match refreshed input count"
        );
        assert_eq!(
            refresh_ids.len(),
            decoder_sets.len(),
            "refresh id count must match decoder set count"
        );
        let num_slots = refreshed_inputs
            .first()
            .expect("online_eval_many requires at least one refreshed input")
            .num_slots();
        assert_eq!(
            num_slots,
            self.params().ring_dimension() as usize,
            "Naive noise refresh currently expects one logical slot per ring coefficient"
        );
        debug!(
            target: "mxx::func_enc::aky24",
            "naive-vector noise-refresh online_eval_many checking refreshed input slots"
        );
        refreshed_inputs.par_iter().for_each(|input| one.assert_compatible(input));
        one.assert_compatible(decryption_key);
        debug!(
            target: "mxx::func_enc::aky24",
            "naive-vector noise-refresh online_eval_many checking encrypted seed slots"
        );
        enc_seeds.par_iter().for_each(|seed| one.assert_compatible(seed));

        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        let secret_size = one.encoding(0).pubkey.matrix.row_size();
        if self.debug_reuse_single_material {
            info!(
                target: "mxx::func_enc::aky24",
                num_slots,
                crt_depth,
                log_base_q,
                "naive-vector noise-refresh online_eval_many sampling debug PRG-output material"
            );
            let debug_decode_circuit = build_noise_refresh_debug_material_decode_circuit::<
                M::P,
                A,
                M,
            >(self.ring_gsw.clone(), self.v_bits);
            let material_input_count = debug_decode_circuit.num_input() - 1;
            assert_eq!(
                material_input_count % 2,
                0,
                "debug noise-refresh material input count must split into error and mask halves"
            );
            let per_material_input_count = material_input_count / 2;
            let error_wire = debug_sample_prg_output_encoding_wires::<M, HS>(
                self.params(),
                self.hash_key,
                b"noise-refresh-debug-error-material",
                1,
                num_slots,
                one,
            )
            .into_iter()
            .next()
            .expect("debug noise-refresh must sample one reusable error wire");
            let mask_wire = debug_sample_prg_output_encoding_wires::<M, HS>(
                self.params(),
                self.hash_key,
                b"noise-refresh-debug-mask-material",
                1,
                num_slots,
                one,
            )
            .into_iter()
            .next()
            .expect("debug noise-refresh must sample one reusable mask wire");
            let mut inputs = Vec::with_capacity(material_input_count + 1);
            inputs.extend(std::iter::repeat_n(error_wire, per_material_input_count));
            inputs.extend(std::iter::repeat_n(mask_wire, per_material_input_count));
            inputs.push(decryption_key.clone());
            let decoded = debug_decode_circuit.eval(
                self.params(),
                one.clone(),
                inputs,
                Some(plt_evaluator),
                Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
                None,
            );
            let decoded_row_size = decoded
                .first()
                .and_then(|value| {
                    (value.num_slots() > 0).then(|| value.encoding(0).pubkey.matrix.row_size())
                })
                .expect("debug noise-refresh decoded encoding material must be nonempty");
            let unit_column_target =
                M::unit_column_vector(self.params(), decoded_row_size, decoded_row_size - 1);
            let decoded = decoded
                .into_iter()
                .map(|value| value.matrix_mul(self.params(), &unit_column_target))
                .collect::<Vec<_>>();
            let decoded_refresh_terms = decoded_refresh_terms_encoding(
                self.params(),
                &decoded,
                num_slots,
                crt_depth,
                log_base_q,
                secret_size,
            );
            return refresh_ids
                .par_iter()
                .zip(refreshed_inputs.par_iter())
                .zip(decoder_sets.par_iter())
                .map(|((refresh_id, refreshed_input), decoders)| {
                    self.online_from_decoded(
                        refresh_id,
                        one,
                        refreshed_input,
                        &decoded_refresh_terms,
                        decoders,
                    )
                })
                .collect();
        }

        debug!(
            target: "mxx::func_enc::aky24",
            seed_wire_count = enc_seeds.len(),
            refreshed_input_count = refreshed_inputs.len(),
            "naive-vector noise-refresh online_eval_many cloning encrypted seed inputs"
        );
        let material_seed_bits = if self.debug_reuse_single_material { 5 } else { self.seed_bits };
        let seed_wire_count = enc_seeds
            .len()
            .checked_div(self.seed_bits)
            .expect("noise-refresh seed bit count must be positive");
        assert_eq!(
            seed_wire_count * self.seed_bits,
            enc_seeds.len(),
            "noise-refresh encrypted seed wire count must be divisible by seed_bits"
        );
        let material_seed_wire_count = material_seed_bits
            .checked_mul(seed_wire_count)
            .expect("noise-refresh material seed wire count overflow");
        info!(
            target: "mxx::func_enc::aky24",
            num_slots,
            seed_bits = self.seed_bits,
            material_seed_bits,
            v_bits = self.v_bits,
            cbd_n = self.cbd_n,
            "naive-vector noise-refresh online_eval_many preparing decoded material"
        );
        let started = std::time::Instant::now();
        let mut inputs =
            enc_seeds[..material_seed_wire_count].par_iter().cloned().collect::<Vec<_>>();
        inputs.push(decryption_key.clone());
        info!(
            target: "mxx::func_enc::aky24",
            input_count = inputs.len(),
            output_count = self.material_circuit.num_output(),
            "naive-vector noise-refresh online_eval_many evaluating cached material circuit once"
        );
        let decoded = self.material_circuit.eval(
            self.params(),
            one.clone(),
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
            None,
        );
        debug!(
            target: "mxx::func_enc::aky24",
            decoded_count = decoded.len(),
            refreshed_input_count = refreshed_inputs.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "naive-vector noise-refresh online_eval_many material circuit evaluated"
        );
        if let Some(first_decoded) = decoded.first() {
            if first_decoded.num_slots() > 0 {
                let first_encoding = first_decoded.encoding(0);
                info!(
                    target: "mxx::func_enc::aky24",
                    decoded_slots = first_decoded.num_slots(),
                    decoded_rows = first_encoding.vector.row_size(),
                    decoded_cols = first_encoding.vector.col_size(),
                    "naive-vector noise-refresh online_eval_many first decoded material shape"
                );
            }
        }
        let decoded_row_size = decoded
            .first()
            .and_then(|value| {
                (value.num_slots() > 0).then(|| value.encoding(0).pubkey.matrix.row_size())
            })
            .expect("noise-refresh decoded encoding material must be nonempty");
        let unit_column_target =
            M::unit_column_vector(self.params(), decoded_row_size, decoded_row_size - 1);
        let decoded = decoded
            .into_iter()
            .map(|value| value.matrix_mul(self.params(), &unit_column_target))
            .collect::<Vec<_>>();
        let decoded_refresh_terms = decoded_refresh_terms_encoding(
            self.params(),
            &decoded,
            num_slots,
            crt_depth,
            log_base_q,
            secret_size,
        );
        refresh_ids
            .par_iter()
            .zip(refreshed_inputs.par_iter())
            .zip(decoder_sets.par_iter())
            .map(|((refresh_id, refreshed_input), decoders)| {
                self.online_from_decoded(
                    refresh_id,
                    one,
                    refreshed_input,
                    &decoded_refresh_terms,
                    decoders,
                )
            })
            .collect()
    }

    /// Computes many online refreshes while materializing decoder matrices in
    /// bounded chunks.
    ///
    /// This has the same semantics as `online_eval_many`, but `decoder_factory`
    /// is called only for the currently processed chunk. It is useful for
    /// DiamondIO, where decoder matrices are read from disk and multiplied by
    /// the final Diamond state; building all decoder sets for all refreshed
    /// wires at once can exhaust GPU memory even though each individual matrix
    /// is tiny.
    #[allow(clippy::too_many_arguments)]
    pub fn online_eval_many_with_decoder_factory<PE, ST, DF>(
        &self,
        refresh_ids: &[Vec<u8>],
        one: &NaiveBGGEncodingVec<M>,
        refreshed_inputs: &[NaiveBGGEncodingVec<M>],
        enc_seeds: &[NaiveBGGEncodingVec<M>],
        decryption_key: &NaiveBGGEncodingVec<M>,
        decoder_factory: DF,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Vec<NaiveBGGEncodingVec<M>>
    where
        PE: PltEvaluator<NaiveBGGEncodingVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
        DF: Fn(usize) -> Vec<M> + Sync,
    {
        info!(
            target: "mxx::func_enc::aky24",
            refresh_id_count = refresh_ids.len(),
            refreshed_input_count = refreshed_inputs.len(),
            seed_wire_count = enc_seeds.len(),
            decoder_chunk_size = crate::env::noise_refresh_decoder_chunk_size(),
            "naive-vector noise-refresh online_eval_many_with_decoder_factory entered"
        );
        assert_eq!(
            refresh_ids.len(),
            refreshed_inputs.len(),
            "refresh id count must match refreshed input count"
        );
        let num_slots = refreshed_inputs
            .first()
            .expect("online_eval_many_with_decoder_factory requires at least one refreshed input")
            .num_slots();
        assert_eq!(
            num_slots,
            self.params().ring_dimension() as usize,
            "Naive noise refresh currently expects one logical slot per ring coefficient"
        );
        refreshed_inputs.par_iter().for_each(|input| one.assert_compatible(input));
        one.assert_compatible(decryption_key);
        enc_seeds.par_iter().for_each(|seed| one.assert_compatible(seed));

        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        let secret_size = one.encoding(0).pubkey.matrix.row_size();
        let decoded_refresh_terms = if self.debug_reuse_single_material {
            info!(
                target: "mxx::func_enc::aky24",
                num_slots,
                crt_depth,
                log_base_q,
                "naive-vector noise-refresh chunked online eval sampling debug PRG-output material"
            );
            let debug_decode_circuit = build_noise_refresh_debug_material_decode_circuit::<
                M::P,
                A,
                M,
            >(self.ring_gsw.clone(), self.v_bits);
            let material_input_count = debug_decode_circuit.num_input() - 1;
            assert_eq!(
                material_input_count % 2,
                0,
                "debug noise-refresh material input count must split into error and mask halves"
            );
            let per_material_input_count = material_input_count / 2;
            let error_wire = debug_sample_prg_output_encoding_wires::<M, HS>(
                self.params(),
                self.hash_key,
                b"noise-refresh-debug-error-material",
                1,
                num_slots,
                one,
            )
            .into_iter()
            .next()
            .expect("debug noise-refresh must sample one reusable error wire");
            let mask_wire = debug_sample_prg_output_encoding_wires::<M, HS>(
                self.params(),
                self.hash_key,
                b"noise-refresh-debug-mask-material",
                1,
                num_slots,
                one,
            )
            .into_iter()
            .next()
            .expect("debug noise-refresh must sample one reusable mask wire");
            let mut inputs = Vec::with_capacity(material_input_count + 1);
            inputs.extend(std::iter::repeat_n(error_wire, per_material_input_count));
            inputs.extend(std::iter::repeat_n(mask_wire, per_material_input_count));
            inputs.push(decryption_key.clone());
            let decoded = debug_decode_circuit.eval(
                self.params(),
                one.clone(),
                inputs,
                Some(plt_evaluator),
                Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
                None,
            );
            let decoded_row_size = decoded
                .first()
                .and_then(|value| {
                    (value.num_slots() > 0).then(|| value.encoding(0).pubkey.matrix.row_size())
                })
                .expect("debug noise-refresh decoded encoding material must be nonempty");
            let unit_column_target =
                M::unit_column_vector(self.params(), decoded_row_size, decoded_row_size - 1);
            let decoded = decoded
                .into_iter()
                .map(|value| value.matrix_mul(self.params(), &unit_column_target))
                .collect::<Vec<_>>();
            decoded_refresh_terms_encoding(
                self.params(),
                &decoded,
                num_slots,
                crt_depth,
                log_base_q,
                secret_size,
            )
        } else {
            let material_seed_bits = self.seed_bits;
            let seed_wire_count = enc_seeds
                .len()
                .checked_div(self.seed_bits)
                .expect("noise-refresh seed bit count must be positive");
            assert_eq!(
                seed_wire_count * self.seed_bits,
                enc_seeds.len(),
                "noise-refresh encrypted seed wire count must be divisible by seed_bits"
            );
            let material_seed_wire_count = material_seed_bits
                .checked_mul(seed_wire_count)
                .expect("noise-refresh material seed wire count overflow");
            let mut inputs =
                enc_seeds[..material_seed_wire_count].par_iter().cloned().collect::<Vec<_>>();
            inputs.push(decryption_key.clone());
            let decoded = self.material_circuit.eval(
                self.params(),
                one.clone(),
                inputs,
                Some(plt_evaluator),
                Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
                None,
            );
            let decoded_row_size = decoded
                .first()
                .and_then(|value| {
                    (value.num_slots() > 0).then(|| value.encoding(0).pubkey.matrix.row_size())
                })
                .expect("noise-refresh decoded encoding material must be nonempty");
            let unit_column_target =
                M::unit_column_vector(self.params(), decoded_row_size, decoded_row_size - 1);
            let decoded = decoded
                .into_iter()
                .map(|value| value.matrix_mul(self.params(), &unit_column_target))
                .collect::<Vec<_>>();
            decoded_refresh_terms_encoding(
                self.params(),
                &decoded,
                num_slots,
                crt_depth,
                log_base_q,
                secret_size,
            )
        };

        let chunk_size = crate::env::noise_refresh_decoder_chunk_size();
        let mut outputs = Vec::with_capacity(refresh_ids.len());
        for chunk_start in (0..refresh_ids.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(refresh_ids.len());
            let decoder_started = std::time::Instant::now();
            let decoder_sets = (chunk_start..chunk_end).map(&decoder_factory).collect::<Vec<_>>();
            debug!(
                target: "mxx::func_enc::aky24",
                chunk_start,
                chunk_end,
                decoder_set_count = decoder_sets.len(),
                decoder_count = decoder_sets.iter().map(Vec::len).sum::<usize>(),
                elapsed_ms = decoder_started.elapsed().as_millis(),
                "naive-vector noise-refresh chunked online eval materialized decoder chunk"
            );
            let mut chunk_outputs = refresh_ids[chunk_start..chunk_end]
                .par_iter()
                .zip(refreshed_inputs[chunk_start..chunk_end].par_iter())
                .zip(decoder_sets.par_iter())
                .map(|((refresh_id, refreshed_input), decoders)| {
                    self.online_from_decoded(
                        refresh_id,
                        one,
                        refreshed_input,
                        &decoded_refresh_terms,
                        decoders,
                    )
                })
                .collect::<Vec<_>>();
            outputs.append(&mut chunk_outputs);
        }
        outputs
    }

    fn preprocess_from_decoded(
        &self,
        refresh_id: &[u8],
        one: &NaiveBGGPublicKeyVec<M>,
        refreshed_input: &NaiveBGGPublicKeyVec<M>,
        decoded_refresh_terms: &[M],
    ) -> (NaiveBGGPublicKeyVec<M>, Vec<NaiveBGGPublicKeyVec<M>>) {
        let num_slots = refreshed_input.num_slots();
        let secret_size = one.key(0).matrix.row_size();
        let hash_sampler = HS::new();
        let a_prime = NaiveBGGPublicKeyVec::new(
            self.params(),
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let slot_refresh_id = refresh_slot_id(refresh_id, b":a_prime:", slot_idx);
                    BggPublicKey::new(
                        hash_sampler.sample_hash(
                            self.params(),
                            self.hash_key,
                            &slot_refresh_id,
                            secret_size,
                            secret_size * self.params().modulus_digits(),
                            DistType::FinRingDist,
                        ),
                        true,
                    )
                })
                .collect::<Vec<_>>(),
        );
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        assert_eq!(decoded_refresh_terms.len(), num_slots * crt_depth);
        let refresh_keys = (0..crt_depth)
            .into_par_iter()
            .map(|crt_idx| {
                NaiveBGGPublicKeyVec::new(
                    self.params(),
                    (0..num_slots)
                        .map(|slot_idx| {
                            let (q_over_qi, _reconst_coeff) = self.params().to_crt_coeffs(crt_idx);
                            let one_key = one.key(slot_idx);
                            let a_prime_key = a_prime.key(slot_idx);
                            let refreshed_input_key = refreshed_input.key(slot_idx);
                            // The one-term is evaluated with the natural `+(q/q_i) * A'` target,
                            // then subtracted from the accumulated refresh matrix below. In the
                            // online path this subtraction makes `online_terms - decoder` leave a
                            // positive `+(q/q_i) * A'`.
                            let one_term = one_key
                                .matrix_mul(
                                    self.params(),
                                    &(a_prime_key.matrix.clone() *
                                        constant_poly::<M>(self.params(), q_over_qi.clone())),
                                )
                                .matrix
                                .clone();
                            // The refreshed input target is positive `+(q/q_i) * G`, so online
                            // subtraction of the decoder cancels the public `sA_x` part and
                            // leaves `-xG`.
                            let input_term = refreshed_input_key
                                .matrix_mul(
                                    self.params(),
                                    &(M::gadget_matrix(self.params(), secret_size) *
                                        constant_poly::<M>(self.params(), q_over_qi)),
                                )
                                .matrix
                                .clone();
                            let refresh_term =
                                decoded_refresh_terms[slot_idx * crt_depth + crt_idx].clone();
                            BggPublicKey::new(input_term + &refresh_term - &one_term, true)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        (a_prime, refresh_keys)
    }

    fn online_from_decoded(
        &self,
        refresh_id: &[u8],
        one: &NaiveBGGEncodingVec<M>,
        refreshed_input: &NaiveBGGEncodingVec<M>,
        decoded_refresh_terms: &[M],
        decoders: &[M],
    ) -> NaiveBGGEncodingVec<M> {
        let num_slots = refreshed_input.num_slots();
        let secret_size = one.encoding(0).pubkey.matrix.row_size();
        let hash_sampler = HS::new();
        let a_prime = NaiveBGGPublicKeyVec::new(
            self.params(),
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let slot_refresh_id = refresh_slot_id(refresh_id, b":a_prime:", slot_idx);
                    BggPublicKey::new(
                        hash_sampler.sample_hash(
                            self.params(),
                            self.hash_key,
                            &slot_refresh_id,
                            secret_size,
                            secret_size * self.params().modulus_digits(),
                            DistType::FinRingDist,
                        ),
                        true,
                    )
                })
                .collect::<Vec<_>>(),
        );
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        assert_eq!(decoded_refresh_terms.len(), num_slots * crt_depth);
        assert_eq!(
            decoders.len(),
            num_slots * crt_depth,
            "decoder count must equal num_slots * crt_depth"
        );
        let crt_level_vectors = (0..num_slots * crt_depth)
            .into_par_iter()
            .map(|flat_idx| {
                let slot_idx = flat_idx / crt_depth;
                let crt_idx = flat_idx % crt_depth;
                let (q_over_qi, _reconst_coeff) = self.params().to_crt_coeffs(crt_idx);
                // This mirrors preprocessing: evaluate the natural `+(q/q_i) * A'` target and
                // subtract the resulting one-term from the online accumulation. Since BGG
                // encodings have the form `sA - plaintext * target + error`, the final
                // `online_terms - decoder` leaves `+(q/q_i) * A'`.
                let a_prime_key = a_prime.key(slot_idx);
                let one_term = one
                    .encoding(slot_idx)
                    .matrix_mul(
                        self.params(),
                        &(a_prime_key.matrix.clone() *
                            constant_poly::<M>(self.params(), q_over_qi.clone())),
                    )
                    .vector
                    .clone();
                // The positive gadget target leaves the desired `-(q/q_i) * xG` after decoder
                // subtraction.  Together with the `+A'` term and the decoded `+new_error`, the
                // rounded CRT recomposition recovers `A' - xG + new_error`.
                let input_term = refreshed_input
                    .encoding(slot_idx)
                    .matrix_mul(
                        self.params(),
                        &(M::gadget_matrix(self.params(), secret_size) *
                            constant_poly::<M>(self.params(), q_over_qi)),
                    )
                    .vector
                    .clone();
                let refresh_term = decoded_refresh_terms[slot_idx * crt_depth + crt_idx].clone();
                input_term + &refresh_term - &one_term - &decoders[flat_idx]
            })
            .collect::<Vec<_>>();
        let refreshed: M = crt_recompose_rows(self.params(), &crt_level_vectors, num_slots);
        NaiveBGGEncodingVec::new(
            self.params(),
            (0..num_slots)
                .map(|slot_idx| {
                    BggEncoding::new(
                        refreshed.slice_rows(slot_idx, slot_idx + 1),
                        a_prime.key(slot_idx),
                        None,
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<M, A, HS> NoiseRefresher<NaiveBGGPublicKeyVec<M>, NaiveBGGEncodingVec<M>, M>
    for NoiseRefresherNaiveVec<M, A, HS>
where
    M: PolyMatrix + 'static,
    M::P: 'static,
    A: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn preprocess<PE, ST>(
        &self,
        refresh_id: &[u8],
        one: &NaiveBGGPublicKeyVec<M>,
        refreshed_input: &NaiveBGGPublicKeyVec<M>,
        enc_seeds: &[NaiveBGGPublicKeyVec<M>],
        decryption_key: &NaiveBGGPublicKeyVec<M>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> (NaiveBGGPublicKeyVec<M>, Vec<NaiveBGGPublicKeyVec<M>>)
    where
        PE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    {
        self.preprocess_many(
            &[refresh_id.to_vec()],
            one,
            std::slice::from_ref(refreshed_input),
            enc_seeds,
            decryption_key,
            plt_evaluator,
            slot_transfer_evaluator,
        )
        .into_iter()
        .next()
        .expect("single preprocess call must produce one result")
    }

    fn online_eval<PE, ST>(
        &self,
        refresh_id: &[u8],
        one: &NaiveBGGEncodingVec<M>,
        refreshed_input: &NaiveBGGEncodingVec<M>,
        enc_seeds: &[NaiveBGGEncodingVec<M>],
        decryption_key: &NaiveBGGEncodingVec<M>,
        decoders: &[M],
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> NaiveBGGEncodingVec<M>
    where
        PE: PltEvaluator<NaiveBGGEncodingVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
    {
        self.online_eval_many(
            &[refresh_id.to_vec()],
            one,
            std::slice::from_ref(refreshed_input),
            enc_seeds,
            decryption_key,
            &[decoders.to_vec()],
            plt_evaluator,
            slot_transfer_evaluator,
        )
        .into_iter()
        .next()
        .expect("single online_eval call must produce one result")
    }
}

/// Builds the circuit evaluated by both the preprocessing and online paths.
///
/// Input order is all encrypted seed ciphertexts followed by one decryption-key wire. Output order
/// is `slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx`; every output is a
/// decoded `error + mask` polynomial wire for one reference slot, CRT level, and gadget digit.
pub(crate) fn build_noise_refresh_material_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    num_slots: usize,
    debug_reuse_single_material: bool,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(num_slots > 0, "num_slots must be positive");
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    assert_eq!(num_slots, ring_dim, "num_slots must match ring_dim");

    let mut circuit = ring_gsw.fresh_circuit();
    let material_seed_bits = if debug_reuse_single_material { 5 } else { seed_bits };
    assert!(
        seed_bits >= material_seed_bits,
        "noise-refresh material seed bits must not exceed configured seed bits"
    );
    let seed_inputs = (0..material_seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let seed_wires =
        seed_inputs.iter().flat_map(RingGswCiphertext::sub_circuit_wires).collect::<Vec<_>>();
    let ciphertext_template = {
        let mut template_circuit = ring_gsw.fresh_circuit();
        RingGswCiphertext::input(ring_gsw.clone(), None, &mut template_circuit)
    };
    let ciphertext_wire_count =
        ciphertext_template.sub_circuit_wires().iter().map(|wire| wire.len()).sum::<usize>();
    let decrypt_sub_id = circuit.register_sub_circuit(
        build_refreshed_wire_digit_all_crt_decrypt::<P, A, M>(ring_gsw.clone(), v_bits),
    );
    let merge_sub_id = circuit
        .register_sub_circuit(build_refreshed_wire_digit_all_crt_merge::<P>(&ring_gsw.params));
    let mut outputs = Vec::new();
    let log_base_q = ring_gsw.params.modulus_digits();
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let debug_material = if debug_reuse_single_material {
        let prg_sub_id =
            circuit.register_sub_circuit(build_goldreich_encrypted_seed_material_ranges::<P, A>(
                ring_gsw.clone(),
                material_seed_bits,
                v_bits,
                graph_seed,
                cbd_n,
                0,
                0,
                1,
                &[(0, 1)],
                false,
            ));
        let prg_outputs =
            expand_batched_wires(&circuit.call_sub_circuit(prg_sub_id, seed_wires.clone()));
        let logical_outputs = prg_outputs
            .chunks_exact(ciphertext_wire_count)
            .map(|chunk| RingGswCiphertext::from_sub_circuit_outputs(&ciphertext_template, chunk))
            .collect::<Vec<_>>();
        assert_eq!(
            logical_outputs.len(),
            2,
            "debug noise-refresh material must contain one error and one mask ciphertext"
        );
        Some((logical_outputs[0].clone(), logical_outputs[1].clone()))
    } else {
        None
    };
    if let Some((debug_error, debug_mask)) = debug_material.as_ref() {
        let mut decrypt_inputs = Vec::new();
        decrypt_inputs.push(BatchedWire::single(decryption_key));
        for _ in 0..ring_dim {
            decrypt_inputs.extend(debug_error.sub_circuit_wires());
        }
        for _ in 0..crt_depth * mask_q_chunk_len {
            decrypt_inputs.extend(debug_mask.sub_circuit_wires());
        }
        let decrypt_outputs = circuit.call_sub_circuit(decrypt_sub_id, decrypt_inputs);
        let merge_outputs = circuit.call_sub_circuit(merge_sub_id, decrypt_outputs);
        assert_eq!(merge_outputs.len(), crt_depth);
        for _slot_idx in 0..num_slots {
            for crt_output in merge_outputs.iter().take(crt_depth) {
                for _digit_idx in 0..log_base_q {
                    outputs.push(crt_output.clone());
                }
            }
        }
        circuit.output(outputs);
        return circuit;
    }

    for slot_idx in 0..num_slots {
        let mut by_crt_digit =
            vec![vec![BatchedWire::single(decryption_key); log_base_q]; crt_depth];
        for digit_idx in 0..log_base_q {
            let material;
            let error_start = digit_idx * ring_dim;
            let mask_ranges = (0..crt_depth)
                .map(|crt_idx| {
                    (
                        crt_idx * log_base_q * mask_q_chunk_len + digit_idx * mask_q_chunk_len,
                        mask_q_chunk_len,
                    )
                })
                .collect::<Vec<_>>();
            let prg_sub_id = circuit.register_sub_circuit(
                build_goldreich_encrypted_seed_material_ranges::<P, A>(
                    ring_gsw.clone(),
                    material_seed_bits,
                    v_bits,
                    graph_seed,
                    cbd_n,
                    slot_idx,
                    error_start,
                    ring_dim,
                    &mask_ranges,
                    true,
                ),
            );
            let prg_outputs =
                expand_batched_wires(&circuit.call_sub_circuit(prg_sub_id, seed_wires.clone()));
            material = prg_outputs
                .chunks_exact(ciphertext_wire_count)
                .map(|chunk| {
                    RingGswCiphertext::from_sub_circuit_outputs(&ciphertext_template, chunk)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                material.len(),
                ring_dim + crt_depth * mask_q_chunk_len,
                "noise-refresh ranged material output size mismatch"
            );
            let (errors, masks) = material.split_at(ring_dim);

            let mut decrypt_inputs = Vec::new();
            decrypt_inputs.push(BatchedWire::single(decryption_key));
            decrypt_inputs.extend(errors.iter().flat_map(RingGswCiphertext::sub_circuit_wires));
            decrypt_inputs.extend(masks.iter().flat_map(RingGswCiphertext::sub_circuit_wires));

            let decrypt_outputs = circuit.call_sub_circuit(decrypt_sub_id, decrypt_inputs);
            let merge_outputs = circuit.call_sub_circuit(merge_sub_id, decrypt_outputs);
            assert_eq!(merge_outputs.len(), crt_depth);
            for (crt_idx, output) in merge_outputs.into_iter().enumerate() {
                by_crt_digit[crt_idx][digit_idx] = output;
            }
        }
        outputs.extend(by_crt_digit.into_iter().flatten());
    }
    circuit.output(outputs);
    circuit
}

fn build_noise_refresh_debug_material_decode_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    v_bits: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    let mut circuit = ring_gsw.fresh_circuit();
    let debug_error = RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit);
    let debug_mask = RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit);
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let decrypt_sub_id = circuit.register_sub_circuit(
        build_refreshed_wire_digit_all_crt_decrypt::<P, A, M>(ring_gsw.clone(), v_bits),
    );
    let merge_sub_id = circuit
        .register_sub_circuit(build_refreshed_wire_digit_all_crt_merge::<P>(&ring_gsw.params));
    let log_base_q = ring_gsw.params.modulus_digits();
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let mut decrypt_inputs = Vec::new();
    decrypt_inputs.push(BatchedWire::single(decryption_key));
    for _ in 0..ring_dim {
        decrypt_inputs.extend(debug_error.sub_circuit_wires());
    }
    for _ in 0..crt_depth * mask_q_chunk_len {
        decrypt_inputs.extend(debug_mask.sub_circuit_wires());
    }
    let decrypt_outputs = circuit.call_sub_circuit(decrypt_sub_id, decrypt_inputs);
    let merge_outputs = circuit.call_sub_circuit(merge_sub_id, decrypt_outputs);
    assert_eq!(merge_outputs.len(), crt_depth);
    let mut outputs = Vec::new();
    for _slot_idx in 0..ring_dim {
        for crt_output in merge_outputs.iter().take(crt_depth) {
            for _digit_idx in 0..log_base_q {
                outputs.push(crt_output.clone());
            }
        }
    }
    circuit.output(outputs);
    circuit
}

fn collapse_slot_matrices<M, F>(
    params: &<M::P as Poly>::Params,
    num_slots: usize,
    matrix_at_slot: F,
) -> M
where
    M: PolyMatrix,
    F: Fn(usize) -> M,
{
    let ring_dim = params.ring_dimension() as usize;
    assert_eq!(num_slots, ring_dim);
    (0..ring_dim)
        .map(|slot_idx| matrix_at_slot(slot_idx) * M::P::const_rotate_poly(params, slot_idx))
        .reduce(|acc, term| acc + &term)
        .expect("ring_dim must be positive")
}

fn decoded_refresh_terms_public<M>(
    params: &<M::P as Poly>::Params,
    decoded: &[NaiveBGGPublicKeyVec<M>],
    num_slots: usize,
    crt_depth: usize,
    log_base_q: usize,
    secret_size: usize,
) -> Vec<M>
where
    M: PolyMatrix,
{
    assert_eq!(
        decoded.len(),
        num_slots * crt_depth * log_base_q,
        "decoded public material count must equal num_slots * crt_depth * log_base_q",
    );
    (0..num_slots * crt_depth)
        .into_par_iter()
        .map(|flat_idx| {
            let slot_idx = flat_idx / crt_depth;
            let crt_idx = flat_idx % crt_depth;
            let digit_terms = (0..log_base_q)
                .map(|digit_idx| {
                    let decoded_idx =
                        slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx;
                    let projected_digit =
                        collapse_slot_matrices(params, decoded[decoded_idx].num_slots(), |idx| {
                            decoded[decoded_idx].key(idx).matrix.clone()
                        });
                    embed_projected_digit_matrix(
                        params,
                        projected_digit,
                        secret_size,
                        log_base_q,
                        digit_idx,
                    )
                })
                .collect::<Vec<_>>();
            sum_owned_matrices(digit_terms)
        })
        .collect()
}

fn decoded_refresh_terms_encoding<M>(
    params: &<M::P as Poly>::Params,
    decoded: &[NaiveBGGEncodingVec<M>],
    num_slots: usize,
    crt_depth: usize,
    log_base_q: usize,
    secret_size: usize,
) -> Vec<M>
where
    M: PolyMatrix,
{
    assert_eq!(
        decoded.len(),
        num_slots * crt_depth * log_base_q,
        "decoded encoding material count must equal num_slots * crt_depth * log_base_q",
    );
    (0..num_slots * crt_depth)
        .into_par_iter()
        .map(|flat_idx| {
            let slot_idx = flat_idx / crt_depth;
            let crt_idx = flat_idx % crt_depth;
            let digit_terms = (0..log_base_q)
                .map(|digit_idx| {
                    let decoded_idx =
                        slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx;
                    let projected_digit =
                        collapse_slot_matrices(params, decoded[decoded_idx].num_slots(), |idx| {
                            decoded[decoded_idx].encoding(idx).vector.clone()
                        });
                    embed_projected_digit_matrix(
                        params,
                        projected_digit,
                        secret_size,
                        log_base_q,
                        digit_idx,
                    )
                })
                .collect::<Vec<_>>();
            sum_owned_matrices(digit_terms)
        })
        .collect()
}

fn crt_recompose_rows<M>(params: &<M::P as Poly>::Params, crt_values: &[M], num_slots: usize) -> M
where
    M: PolyMatrix,
{
    let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
    assert_eq!(crt_values.len(), num_slots * crt_depth);
    let output_cols = crt_values
        .first()
        .expect("CRT recomposition requires at least one level vector")
        .col_size();
    debug_assert!(
        crt_values.iter().all(|value| value.row_size() == 1 && value.col_size() == output_cols),
        "CRT recomposition level vectors must be one-row matrices with a consistent column count"
    );
    let reconst_coeffs = params.reconst_coeffs();
    let rows = (0..num_slots)
        .map(|slot_idx| {
            let mut row = M::zero(params, 1, output_cols);
            for crt_idx in 0..crt_depth {
                let level = &crt_values[slot_idx * crt_depth + crt_idx];
                let q_i_big = BigUint::from(q_moduli[crt_idx]);
                let rounded = decode_centered_masked_matrix(params, level, &q_i_big);
                row.add_in_place(
                    &(rounded * constant_poly::<M>(params, reconst_coeffs[crt_idx].clone())),
                );
            }
            row
        })
        .collect::<Vec<_>>();
    let mut rows = rows;
    let first = rows.remove(0);
    first.concat_rows_owned(rows)
}

fn concat_owned_columns<M: PolyMatrix>(mut matrices: Vec<M>) -> M {
    let first = matrices.remove(0);
    first.concat_columns_owned(matrices)
}

fn sum_owned_matrices<M: PolyMatrix>(matrices: Vec<M>) -> M {
    matrices
        .into_iter()
        .reduce(|acc, matrix| acc + &matrix)
        .expect("at least one matrix is required")
}

fn embed_projected_digit_matrix<M>(
    params: &<M::P as Poly>::Params,
    projected_digit: M,
    secret_size: usize,
    log_base_q: usize,
    digit_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    assert_eq!(
        projected_digit.col_size(),
        1,
        "projected decoded noise-refresh digit must have one column"
    );
    assert!(digit_idx < log_base_q, "digit index must be in range");
    let row_size = projected_digit.row_size();
    let zero_column = M::zero(params, row_size, 1);
    let columns = (0..secret_size * log_base_q)
        .map(|col_idx| {
            if col_idx % log_base_q == digit_idx {
                projected_digit.clone()
            } else {
                zero_column.clone()
            }
        })
        .collect::<Vec<_>>();
    concat_owned_columns(columns)
}

fn constant_poly<M>(params: &<M::P as Poly>::Params, value: BigUint) -> M::P
where
    M: PolyMatrix,
{
    M::P::from_biguint_to_constant(params, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_traits::One;

    #[test]
    fn crt_recompose_rows_rounds_scaled_level_vectors() {
        let params = DCRTPolyParams::new(2, 2, 10, 5);
        let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
        let q = params.modulus().as_ref().clone();
        let num_slots = 2;
        let log_base_q = params.modulus_digits();
        let expected = (0..num_slots)
            .map(|row| {
                (0..log_base_q)
                    .map(|col| BigUint::from((3 + row * 4 + col * 2) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut crt_values = Vec::new();
        for row_values in expected.iter().take(num_slots) {
            for &q_i in q_moduli.iter().take(crt_depth) {
                let scale = &q / BigUint::from(q_i);
                let entries = row_values
                    .iter()
                    .take(log_base_q)
                    .map(|value| {
                        let scaled = (value * &scale + BigUint::one()) % &q;
                        DCRTPoly::from_biguint_to_constant(&params, scaled)
                    })
                    .collect::<Vec<_>>();
                crt_values.push(DCRTPolyMatrix::from_poly_vec_row(&params, entries));
            }
        }

        let recomposed = crt_recompose_rows::<DCRTPolyMatrix>(&params, &crt_values, num_slots);

        assert_eq!(recomposed.size(), (num_slots, log_base_q));
        for row in 0..num_slots {
            for col in 0..log_base_q {
                assert_eq!(
                    recomposed.entry(row, col).coeffs_biguints()[0],
                    expected[row][col],
                    "recomposed value mismatch at row {row}, column {col}"
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    mod gpu_bench_tests {
        use super::*;
        use crate::{
            bench_estimator::{
                BggEncodingBenchEstimator, BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
                naive_vec::NaiveBGGVecBenchEstimator,
            },
            bgg::public_key::BggPublicKey,
            circuit::{PolyCircuit, gate::GateId},
            element::PolyElem,
            gadgets::{
                arith::nested_rns::{NestedRnsPoly, NestedRnsPolyContext},
                fhe::ring_gsw::RingGswContext,
            },
            lookup::{PublicLut, lwe::LWEBGGPubKeyPltEvaluator},
            matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
            poly::{
                Poly,
                dcrt::{
                    gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
                    params::DCRTPolyParams,
                },
            },
            sampler::{
                PolyTrapdoorSampler,
                gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
                trapdoor::GpuDCRTPolyTrapdoorSampler,
            },
            slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
        };
        use keccak_asm::Keccak256;
        use std::sync::Arc;
        use tempfile::tempdir;

        fn gpu_test_params() -> GpuDCRTPolyParams {
            let cpu_params = DCRTPolyParams::new(2, 1, 12, 6);
            let (q_moduli, _crt_bits, _crt_depth) = cpu_params.to_crt();
            GpuDCRTPolyParams::new(cpu_params.ring_dimension(), q_moduli, cpu_params.base_bits())
        }

        fn gpu_noise_refresher() -> NoiseRefresherNaiveVec<
            GpuDCRTPolyMatrix,
            NestedRnsPoly<GpuDCRTPoly>,
            GpuDCRTPolyHashSampler<Keccak256>,
        > {
            let params = gpu_test_params();
            let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
            let active_levels = 1;
            let nested_rns = Arc::new(NestedRnsPolyContext::setup(
                &mut circuit,
                &params,
                6,
                1,
                1 << 8,
                false,
                Some(active_levels),
            ));
            let ring_gsw = Arc::new(RingGswContext::from_arith_context(
                &mut circuit,
                &params,
                params.ring_dimension() as usize,
                nested_rns,
                Some(active_levels),
                None,
            ));
            NoiseRefresherNaiveVec::new(ring_gsw, 6, 1, [0x42; 32], 1, [0x24; 32])
        }

        fn public_key_vec_bench_estimator(
            params: &GpuDCRTPolyParams,
            num_slots: usize,
        ) -> NaiveBGGVecBenchEstimator<
            BggPublicKeyBenchEstimator<
                GpuDCRTPolyMatrix,
                LWEBGGPubKeyPltEvaluator<
                    GpuDCRTPolyMatrix,
                    GpuDCRTPolyHashSampler<Keccak256>,
                    GpuDCRTPolyTrapdoorSampler,
                >,
                BggPublicKeySTEvaluator<
                    GpuDCRTPolyMatrix,
                    GpuDCRTPolyUniformSampler,
                    GpuDCRTPolyHashSampler<Keccak256>,
                    GpuDCRTPolyTrapdoorSampler,
                >,
            >,
        > {
            let key_matrix = GpuDCRTPolyMatrix::gadget_matrix(params, 1);
            let key_a = BggPublicKey::new(key_matrix.clone(), true);
            let key_b = BggPublicKey::new(key_matrix, true);
            let small_scalar = [3u32];
            let large_scalar = [BigUint::from(5u32)];
            let public_lut = PublicLut::new(
                params,
                2,
                |params: &GpuDCRTPolyParams, input| {
                    Some((input, <GpuDCRTPoly as Poly>::Elem::constant(&params.modulus(), input)))
                },
                Some((1, <GpuDCRTPoly as Poly>::Elem::constant(&params.modulus(), 1))),
            );
            let slot_transfer_src_slots = [(0u32, None)];
            let samples = BggPublicKeyBenchSamples {
                params,
                add_lhs: &key_a,
                add_rhs: &key_b,
                sub_lhs: &key_a,
                sub_rhs: &key_b,
                mul_lhs: &key_a,
                mul_rhs: &key_b,
                small_scalar_input: &key_a,
                small_scalar: &small_scalar,
                large_scalar_input: &key_a,
                large_scalar: &large_scalar,
                public_lut_one: &key_a,
                public_lut_input: &key_b,
                public_lut: &public_lut,
                public_lut_gate_id: GateId(17),
                public_lut_id: 7,
                slot_transfer_input: &key_a,
                slot_transfer_src_slots: &slot_transfer_src_slots,
                slot_transfer_gate_id: GateId(19),
            };
            let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, 4.578);
            let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, 1);
            let eval_dir = tempdir().expect("temporary benchmark evaluator directory");
            let plt_eval = LWEBGGPubKeyPltEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(
                [0x31; 32],
                trapdoor_sampler,
                Arc::new(pub_matrix),
                Arc::new(trapdoor),
                eval_dir.path().to_path_buf(),
            );
            let estimator_trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, 4.578);
            let (estimator_trapdoor, estimator_pub_matrix) =
                estimator_trapdoor_sampler.trapdoor(params, 1);
            let plt_estimator = LWEBGGPubKeyPltEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(
                [0x31; 32],
                estimator_trapdoor_sampler,
                Arc::new(estimator_pub_matrix),
                Arc::new(estimator_trapdoor),
                eval_dir.path().to_path_buf(),
            );
            let st_eval = BggPublicKeySTEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyUniformSampler,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(
                [0x32; 32], 1, num_slots, 4.578, 0.0, eval_dir.path().to_path_buf()
            );
            let st_estimator =
                BggPublicKeySTEvaluator::<
                    GpuDCRTPolyMatrix,
                    GpuDCRTPolyUniformSampler,
                    GpuDCRTPolyHashSampler<Keccak256>,
                    GpuDCRTPolyTrapdoorSampler,
                >::new(
                    [0x32; 32], 1, num_slots, 4.578, 0.0, eval_dir.path().to_path_buf()
                );
            let scalar_estimator = BggPublicKeyBenchEstimator::benchmark(
                &samples,
                &plt_eval,
                &st_eval,
                plt_estimator,
                st_estimator,
                1,
            );
            NaiveBGGVecBenchEstimator::new(scalar_estimator, num_slots)
        }

        fn encoding_vec_bench_estimator(
            params: &GpuDCRTPolyParams,
            num_slots: usize,
        ) -> NaiveBGGVecBenchEstimator<BggEncodingBenchEstimator<GpuDCRTPolyMatrix>> {
            let scalar_estimator = BggEncodingBenchEstimator::benchmark(params, 1);
            NaiveBGGVecBenchEstimator::new(scalar_estimator, num_slots)
        }

        #[test]
        fn noise_refresh_naive_vec_bench_preprcess() {
            let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
            let refresher = gpu_noise_refresher();
            let num_slots = refresher.params().ring_dimension() as usize;
            let material_estimator = public_key_vec_bench_estimator(refresher.params(), num_slots);

            let summary = refresher.estimate_preprocess_bench(&material_estimator);
            tracing::info!(
                ?summary,
                total_time_nanos = %summary.total_time,
                latency = summary.latency,
                max_parallelism = %summary.max_parallelism,
                "naive-vector noise-refresh preprocess benchmark summary"
            );

            assert!(summary.total_time > BigUint::from(0u32));
            assert!(summary.latency.is_finite());
            assert!(summary.latency > 0.0);
            assert!(summary.max_parallelism > BigUint::from(0u32));
        }

        #[test]
        fn noise_refresh_naive_vec_bench_online_eval() {
            let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
            let refresher = gpu_noise_refresher();
            let num_slots = refresher.params().ring_dimension() as usize;
            let material_estimator = encoding_vec_bench_estimator(refresher.params(), num_slots);

            let summary = refresher.estimate_online_eval_bench(&material_estimator);
            tracing::info!(
                ?summary,
                total_time_nanos = %summary.total_time,
                latency = summary.latency,
                max_parallelism = %summary.max_parallelism,
                "naive-vector noise-refresh online benchmark summary"
            );

            assert!(summary.total_time > BigUint::from(0u32));
            assert!(summary.latency.is_finite());
            assert!(summary.latency > 0.0);
            assert!(summary.max_parallelism > BigUint::from(0u32));

            let crt_unit =
                measured_crt_recompose_unit_summary::<GpuDCRTPolyMatrix>(refresher.params());
            assert!(crt_unit.latency.is_finite());
            assert!(crt_unit.max_parallelism > BigUint::from(0u32));
        }
    }
}
