use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, benchmark_gate_operation,
    },
    bgg::{
        encoding::BggEncoding,
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        public_key::BggPublicKey,
    },
    circuit::{BatchedWire, PolyCircuit, evaluable::Evaluable},
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    noise_refresh::{
        NoiseRefresher,
        circuit_decrypt::build_refreshed_wire_digit_all_crt_decrypt,
        circuit_merge::build_refreshed_wire_digit_all_crt_merge,
        circuit_prg::{
            build_goldreich_encrypted_seeds_with_output, goldreich_noise_refresh_output_sizes,
        },
    },
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{marker::PhantomData, sync::Arc};
use tracing::debug;

const NATIVE_BENCH_ITERATIONS: usize = 1;

fn scaled_estimate_summary(unit: CircuitBenchEstimate, task_count: usize) -> CircuitBenchSummary {
    let total_time = unit.total_time * task_count as f64;
    let max_parallelism = unit
        .max_parallelism
        .checked_mul(task_count as u128)
        .expect("noise-refresh benchmark parallelism overflowed while scaling a stage");
    let summary = CircuitBenchSummary::new(total_time, unit.latency, max_parallelism);
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

fn scaled_summary(summary: CircuitBenchSummary, task_count: usize) -> CircuitBenchSummary {
    let total_time = summary.total_time * task_count as f64;
    let max_parallelism = summary
        .max_parallelism
        .checked_mul(task_count as u128)
        .expect("noise-refresh benchmark parallelism overflowed while scaling a summary");
    let scaled = CircuitBenchSummary::new(total_time, summary.latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time).sum::<f64>();
    let latency = parts.iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism = parts.iter().map(|part| part.max_parallelism).max().unwrap_or(0);
    let summary = CircuitBenchSummary::new(total_time, latency, max_parallelism);
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
    let summary = CircuitBenchSummary::new(bench.time, bench.time, 1);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(bench.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn expand_batched_wires(wires: &[BatchedWire]) -> Vec<BatchedWire> {
    wires.iter().flat_map(|wire| (0..wire.len()).map(|idx| wire.at(idx))).collect()
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
        Self {
            ring_gsw,
            seed_bits,
            v_bits,
            graph_seed,
            cbd_n,
            hash_key,
            _hash_sampler: PhantomData,
        }
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
        PKBE: BenchEstimator<NaiveBGGPublicKeyVec<M>> + Sync,
    {
        let num_slots = self.params().ring_dimension() as usize;
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        let combine_task_count =
            num_slots.checked_mul(crt_depth).expect("noise-refresh combine task count overflow");

        let material_circuit = build_noise_refresh_material_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            self.seed_bits,
            self.v_bits,
            self.graph_seed,
            self.cbd_n,
            num_slots,
        );
        let public_material_summary =
            public_key_estimator.estimate_circuit_bench(&material_circuit);

        let scalar_target = [BigUint::from(1u32)];
        let pk_matrix_mul = public_key_estimator.estimate_large_scalar_mul(&scalar_target);
        let pk_add = public_key_estimator.estimate_add();
        let pk_sub = public_key_estimator.estimate_sub();
        let collapse_add_count = log_base_q
            .checked_mul(num_slots.saturating_sub(1))
            .expect("noise-refresh collapse add count overflow");
        let preprocess_combine_unit = sequential_summaries(&[
            // Compute the public-key one-term for the naive slot vector:
            // `one.keys[slot_idx].matrix_mul((q / q_i) * A')`.
            estimate_summary(pk_matrix_mul),
            // Compute the refreshed-input public-key term for the naive slot vector:
            // `refreshed_input.keys[slot_idx].matrix_mul((q / q_i) * G)`.
            estimate_summary(pk_matrix_mul),
            // Apply the one-column target to each of the `log_base_q` decoded material columns.
            scaled_estimate_summary(pk_matrix_mul, log_base_q),
            // For each decoded material column, `collapse_slot_matrices` sums `num_slots`
            // rotated slot matrices into one polynomial column; `concat_owned_columns` then only
            // lays those columns side by side and is not modeled as arithmetic.
            scaled_estimate_summary(pk_add, collapse_add_count),
            // Add the refreshed-input term and the decoded refresh term.
            estimate_summary(pk_add),
            // Subtract the one-term so the public refresh matrix has the intended sign.
            estimate_summary(pk_sub),
        ]);

        let preprocess_combine_stage = scaled_summary(preprocess_combine_unit, combine_task_count);
        let preprocess_summary =
            sequential_summaries(&[public_material_summary, preprocess_combine_stage]);

        debug!(
            v_bits = self.v_bits,
            num_slots,
            crt_depth,
            log_base_q,
            combine_task_count,
            ?public_material_summary,
            ?preprocess_combine_unit,
            ?preprocess_combine_stage,
            ?preprocess_summary,
            "estimated naive-vector noise-refresh preprocess benchmark"
        );

        preprocess_summary
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
    {
        let num_slots = self.params().ring_dimension() as usize;
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        let combine_task_count =
            num_slots.checked_mul(crt_depth).expect("noise-refresh combine task count overflow");

        let material_circuit = build_noise_refresh_material_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            self.seed_bits,
            self.v_bits,
            self.graph_seed,
            self.cbd_n,
            num_slots,
        );
        let online_material_summary = encoding_estimator.estimate_circuit_bench(&material_circuit);

        let scalar_target = [BigUint::from(1u32)];
        let enc_matrix_mul = encoding_estimator.estimate_large_scalar_mul(&scalar_target);
        let enc_add = encoding_estimator.estimate_add();
        let enc_sub = encoding_estimator.estimate_sub();
        let collapse_add_count = log_base_q
            .checked_mul(num_slots.saturating_sub(1))
            .expect("noise-refresh collapse add count overflow");
        let online_combine_unit = sequential_summaries(&[
            // Compute the encoding one-term for the naive slot vector:
            // `one.encodings[slot_idx].matrix_mul((q / q_i) * A')`.
            estimate_summary(enc_matrix_mul),
            // Compute the refreshed-input encoding term for the naive slot vector:
            // `refreshed_input.encodings[slot_idx].matrix_mul((q / q_i) * G)`.
            estimate_summary(enc_matrix_mul),
            // Apply the one-column target to each of the `log_base_q` decoded material columns.
            scaled_estimate_summary(enc_matrix_mul, log_base_q),
            // For each decoded material column, `collapse_slot_matrices` sums `num_slots`
            // rotated slot vectors into one polynomial column; `concat_owned_columns` then only
            // lays those columns side by side and is not modeled as arithmetic.
            scaled_estimate_summary(enc_add, collapse_add_count),
            // Add the refreshed-input term and the decoded refresh term.
            estimate_summary(enc_add),
            // Subtract the one-term to leave the desired positive `A'` contribution after
            // decoding.
            estimate_summary(enc_sub),
            // Subtract the caller-provided decoder for this `(slot_idx, crt_idx)` level.
            estimate_summary(enc_sub),
        ]);

        // Measure the native `crt_recompose_rows` work for one final slot row directly.  This is
        // not a BGG encoding gate: it performs coefficient rounding, reconstruction-coefficient
        // multiplication, and CRT-level accumulation on raw `M` matrices.
        let online_crt_recompose_unit = measured_crt_recompose_unit_summary::<M>(self.params());

        let online_combine_stage = scaled_summary(online_combine_unit, combine_task_count);
        let online_crt_stage = scaled_summary(online_crt_recompose_unit, num_slots);
        let online_summary = sequential_summaries(&[
            online_material_summary,
            online_combine_stage,
            online_crt_stage,
        ]);

        debug!(
            v_bits = self.v_bits,
            num_slots,
            crt_depth,
            log_base_q,
            combine_task_count,
            crt_recompose_task_count = num_slots,
            ?online_material_summary,
            ?online_combine_unit,
            ?online_combine_stage,
            ?online_crt_recompose_unit,
            ?online_crt_stage,
            ?online_summary,
            "estimated naive-vector noise-refresh online benchmark"
        );

        online_summary
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
        let num_slots = refreshed_input.num_slots();
        assert_eq!(
            num_slots,
            self.params().ring_dimension() as usize,
            "Naive noise refresh currently expects one logical slot per ring coefficient"
        );
        one.assert_compatible(refreshed_input);
        one.assert_compatible(decryption_key);
        for seed in enc_seeds {
            one.assert_compatible(seed);
        }

        let circuit = build_noise_refresh_material_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            self.seed_bits,
            self.v_bits,
            self.graph_seed,
            self.cbd_n,
            num_slots,
        );
        assert_eq!(
            enc_seeds.len() + 1,
            circuit.num_input(),
            "encrypted seed wire count must match the flattened noise-refresh circuit inputs"
        );
        let mut inputs = enc_seeds.to_vec();
        inputs.push(decryption_key.clone());
        let decoded = circuit.eval(
            self.params(),
            one.clone(),
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>),
            None,
        );
        let unit_column_target = M::identity(self.params(), 1, None);
        let decoded = decoded
            .into_par_iter()
            .map(|value| value.matrix_mul(self.params(), &unit_column_target))
            .collect::<Vec<_>>();
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
                            1,
                            self.params().modulus_digits(),
                            DistType::FinRingDist,
                        ),
                        true,
                    )
                })
                .collect::<Vec<_>>(),
        );
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        assert_eq!(decoded.len(), num_slots * crt_depth * log_base_q);
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
                                    &(M::gadget_matrix(self.params(), 1) *
                                        constant_poly::<M>(self.params(), q_over_qi)),
                                )
                                .matrix
                                .clone();
                            let columns = (0..log_base_q)
                                .map(|digit_idx| {
                                    let idx = slot_idx * crt_depth * log_base_q +
                                        crt_idx * log_base_q +
                                        digit_idx;
                                    collapse_slot_matrices(
                                        self.params(),
                                        decoded[idx].num_slots(),
                                        |slot_idx| decoded[idx].key(slot_idx).matrix.clone(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let refresh_term = concat_owned_columns(columns);
                            BggPublicKey::new(input_term + &refresh_term - &one_term, true)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        (a_prime, refresh_keys)
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
        let num_slots = refreshed_input.num_slots();
        assert_eq!(
            num_slots,
            self.params().ring_dimension() as usize,
            "Naive noise refresh currently expects one logical slot per ring coefficient"
        );
        one.assert_compatible(refreshed_input);
        one.assert_compatible(decryption_key);
        for seed in enc_seeds {
            one.assert_compatible(seed);
        }

        let circuit = build_noise_refresh_material_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            self.seed_bits,
            self.v_bits,
            self.graph_seed,
            self.cbd_n,
            num_slots,
        );
        assert_eq!(
            enc_seeds.len() + 1,
            circuit.num_input(),
            "encrypted seed wire count must match the flattened noise-refresh circuit inputs"
        );
        let mut inputs = enc_seeds.to_vec();
        inputs.push(decryption_key.clone());
        let decoded = circuit.eval(
            self.params(),
            one.clone(),
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
            None,
        );
        let unit_column_target = M::identity(self.params(), 1, None);
        let decoded = decoded
            .into_par_iter()
            .map(|value| value.matrix_mul(self.params(), &unit_column_target))
            .collect::<Vec<_>>();
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
                            1,
                            self.params().modulus_digits(),
                            DistType::FinRingDist,
                        ),
                        true,
                    )
                })
                .collect::<Vec<_>>(),
        );
        let (_q_moduli, _crt_bits, crt_depth) = self.params().to_crt();
        let log_base_q = self.params().modulus_digits();
        assert_eq!(decoded.len(), num_slots * crt_depth * log_base_q);
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
                        &(M::gadget_matrix(self.params(), 1) *
                            constant_poly::<M>(self.params(), q_over_qi)),
                    )
                    .vector
                    .clone();
                let columns = (0..log_base_q)
                    .map(|digit_idx| {
                        let idx =
                            slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx;
                        collapse_slot_matrices(
                            self.params(),
                            decoded[idx].num_slots(),
                            |slot_idx| decoded[idx].encoding(slot_idx).vector.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                let refresh_term = concat_owned_columns(columns);
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
    let seed_inputs = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
    let seed_wires =
        seed_inputs.iter().flat_map(RingGswCiphertext::sub_circuit_wires).collect::<Vec<_>>();
    let mut shared_inputs = seed_wires.clone();
    shared_inputs.push(BatchedWire::single(decryption_key));
    let mut outputs = Vec::new();
    for slot_idx in 0..num_slots {
        let slot_sub_id =
            circuit.register_sub_circuit(build_noise_refresh_slot_material_circuit::<P, A, M>(
                ring_gsw.clone(),
                seed_bits,
                v_bits,
                graph_seed,
                cbd_n,
                slot_idx,
            ));
        outputs.extend(circuit.call_sub_circuit(slot_sub_id, shared_inputs.clone()));
    }
    circuit.output(outputs);
    circuit
}

fn build_noise_refresh_slot_material_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    slot_idx: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    assert!(slot_idx < ring_dim, "slot_idx must be in range");
    let log_base_q = ring_gsw.params.modulus_digits();
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let output_sizes =
        goldreich_noise_refresh_output_sizes(ring_dim, log_base_q, crt_depth, v_bits);
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask chunk length overflow");

    let mut circuit = ring_gsw.fresh_circuit();
    let seed_inputs = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let decryption_key = circuit.input(1).at(0).as_single_wire();
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
    let seed_wires =
        seed_inputs.iter().flat_map(RingGswCiphertext::sub_circuit_wires).collect::<Vec<_>>();
    let prg_sub_id =
        circuit.register_sub_circuit(build_goldreich_encrypted_seeds_with_output::<P, A>(
            ring_gsw.clone(),
            seed_bits,
            v_bits,
            graph_seed,
            cbd_n,
            slot_idx,
        ));
    let prg_outputs =
        expand_batched_wires(&circuit.call_sub_circuit(prg_sub_id, seed_wires.clone()));
    let logical_outputs = prg_outputs
        .chunks_exact(ciphertext_wire_count)
        .map(|chunk| RingGswCiphertext::from_sub_circuit_outputs(&ciphertext_template, chunk))
        .collect::<Vec<_>>();
    assert_eq!(logical_outputs.len(), output_sizes.total);
    let (errors, masks) = logical_outputs.split_at(output_sizes.cbd_values);
    let mut by_crt_digit = vec![vec![BatchedWire::single(decryption_key); log_base_q]; crt_depth];

    for digit_idx in 0..log_base_q {
        let mut decrypt_inputs = Vec::new();
        decrypt_inputs.push(BatchedWire::single(decryption_key));

        let error_start = digit_idx * ring_dim;
        decrypt_inputs.extend(
            errors[error_start..error_start + ring_dim]
                .iter()
                .flat_map(RingGswCiphertext::sub_circuit_wires),
        );

        for crt_idx in 0..crt_depth {
            let mask_start = crt_idx * log_base_q * mask_q_chunk_len + digit_idx * mask_q_chunk_len;
            decrypt_inputs.extend(
                masks[mask_start..mask_start + mask_q_chunk_len]
                    .iter()
                    .flat_map(RingGswCiphertext::sub_circuit_wires),
            );
        }

        let decrypt_outputs = circuit.call_sub_circuit(decrypt_sub_id, decrypt_inputs);
        let merge_outputs = circuit.call_sub_circuit(merge_sub_id, decrypt_outputs);
        assert_eq!(merge_outputs.len(), crt_depth);
        for (crt_idx, output) in merge_outputs.into_iter().enumerate() {
            by_crt_digit[crt_idx][digit_idx] = output;
        }
    }

    circuit.output(by_crt_digit.into_iter().flatten());
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

fn crt_recompose_rows<M>(params: &<M::P as Poly>::Params, crt_values: &[M], num_slots: usize) -> M
where
    M: PolyMatrix,
{
    let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
    assert_eq!(crt_values.len(), num_slots * crt_depth);
    let q: Arc<BigUint> = params.modulus().into();
    let half_q = q.as_ref() / BigUint::from(2u64);
    let reconst_coeffs = params.reconst_coeffs();
    let rows = (0..num_slots)
        .map(|slot_idx| {
            let mut row = M::zero(params, 1, params.modulus_digits());
            for crt_idx in 0..crt_depth {
                let level = &crt_values[slot_idx * crt_depth + crt_idx];
                let (level_rows, level_cols) = level.size();
                let q_i_big = BigUint::from(q_moduli[crt_idx]);
                let mut rounded = M::zero(params, level_rows, level_cols);
                for level_row in 0..level_rows {
                    for level_col in 0..level_cols {
                        let rounded_coeffs = level
                            .entry(level_row, level_col)
                            .coeffs_biguints()
                            .into_iter()
                            .map(|coeff| ((&q_i_big * coeff + &half_q) / q.as_ref()) % &q_i_big)
                            .collect::<Vec<_>>();
                        rounded.set_entry(
                            level_row,
                            level_col,
                            M::P::from_biguints(params, &rounded_coeffs),
                        );
                    }
                }
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
            NoiseRefresherNaiveVec::new(ring_gsw, 5, 1, [0x42; 32], 1, [0x24; 32])
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
                total_time = summary.total_time,
                latency = summary.latency,
                max_parallelism = summary.max_parallelism,
                "naive-vector noise-refresh preprocess benchmark summary"
            );

            assert!(summary.total_time.is_finite());
            assert!(summary.total_time > 0.0);
            assert!(summary.latency.is_finite());
            assert!(summary.latency > 0.0);
            assert!(summary.max_parallelism > 0);
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
                total_time = summary.total_time,
                latency = summary.latency,
                max_parallelism = summary.max_parallelism,
                "naive-vector noise-refresh online benchmark summary"
            );

            assert!(summary.total_time.is_finite());
            assert!(summary.total_time > 0.0);
            assert!(summary.latency.is_finite());
            assert!(summary.latency > 0.0);
            assert!(summary.max_parallelism > 0);

            let crt_unit =
                measured_crt_recompose_unit_summary::<GpuDCRTPolyMatrix>(refresher.params());
            assert!(crt_unit.total_time.is_finite());
            assert!(crt_unit.latency.is_finite());
            assert!(crt_unit.max_parallelism > 0);
        }
    }
}
