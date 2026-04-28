use crate::{
    bgg::naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
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
use num_traits::Zero;
use rayon::prelude::*;
use std::{marker::PhantomData, sync::Arc};

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

    fn num_slots_from_public_key(input: &NaiveBGGPublicKeyVec<M>) -> usize {
        input.num_slots()
    }

    fn num_slots_from_encoding(input: &NaiveBGGEncodingVec<M>) -> usize {
        input.num_slots()
    }

    fn build_eval_circuit(&self, num_slots: usize) -> PolyCircuit<M::P> {
        build_noise_refresh_material_circuit::<M::P, A, M>(
            self.ring_gsw.clone(),
            self.seed_bits,
            self.v_bits,
            self.graph_seed,
            self.cbd_n,
            num_slots,
        )
    }

    fn sample_a_prime(&self, refresh_id: &[u8]) -> M
    where
        HS: PolyHashSampler<[u8; 32], M = M>,
    {
        HS::new().sample_hash(
            self.params(),
            self.hash_key,
            refresh_id,
            1,
            self.params().modulus_digits(),
            DistType::FinRingDist,
        )
    }

    fn scaled_gadget_target(&self, crt_idx: usize) -> M {
        let (q_over_qi, _reconst_coeff) = self.params().to_crt_coeffs(crt_idx);
        M::gadget_matrix(self.params(), 1) * constant_poly::<M>(self.params(), q_over_qi)
    }

    fn neg_scaled_a_prime(&self, a_prime: &M, crt_idx: usize) -> M {
        let (q_over_qi, _reconst_coeff) = self.params().to_crt_coeffs(crt_idx);
        let modulus: Arc<BigUint> = self.params().modulus().into();
        let scalar = if (&q_over_qi % modulus.as_ref()).is_zero() {
            BigUint::zero()
        } else {
            modulus.as_ref() - (&q_over_qi % modulus.as_ref())
        };
        a_prime.clone() * constant_poly::<M>(self.params(), scalar)
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
    ) -> (M, Vec<M>)
    where
        PE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    {
        assert_eq!(enc_seeds.len(), self.seed_bits, "encrypted seed count mismatch");
        let num_slots = Self::num_slots_from_public_key(refreshed_input);
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

        let circuit = self.build_eval_circuit(num_slots);
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
        let decoded = apply_unit_column_to_public_keys(self.params(), decoded);
        let a_prime = self.sample_a_prime(refresh_id);
        let refresh_matrices = combine_public_refresh_matrices(
            self,
            &a_prime,
            one,
            refreshed_input,
            &decoded,
            num_slots,
        );
        (a_prime, refresh_matrices)
    }

    fn online_eval<PE, ST>(
        &self,
        refresh_id: &[u8],
        one: &NaiveBGGEncodingVec<M>,
        refreshed_input: &NaiveBGGEncodingVec<M>,
        enc_seeds: &[NaiveBGGEncodingVec<M>],
        decryption_key: &NaiveBGGEncodingVec<M>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> M
    where
        PE: PltEvaluator<NaiveBGGEncodingVec<M>>,
        ST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
    {
        assert_eq!(enc_seeds.len(), self.seed_bits, "encrypted seed count mismatch");
        let num_slots = Self::num_slots_from_encoding(refreshed_input);
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

        let circuit = self.build_eval_circuit(num_slots);
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
        let decoded = apply_unit_column_to_encodings(self.params(), decoded);
        let a_prime = self.sample_a_prime(refresh_id);
        let crt_level_vectors = combine_encoding_refresh_vectors(
            self,
            &a_prime,
            one,
            refreshed_input,
            &decoded,
            num_slots,
        );
        crt_recompose_rows(self.params(), &crt_level_vectors, num_slots)
    }
}

/// Builds the circuit evaluated by both the preprocessing and online paths.
///
/// Input order is all encrypted seed ciphertexts followed by one decryption-key wire. Output order
/// is `slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx`; every output is a
/// decoded `error + mask` polynomial wire for one reference slot, CRT level, and gadget digit.
fn build_noise_refresh_material_circuit<P, A, M>(
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
    let ciphertext_wire_count = ciphertext_template.sub_circuit_wires().len();
    let decrypt_sub_id = circuit.register_sub_circuit(
        build_refreshed_wire_digit_all_crt_decrypt::<P, A, M>(ring_gsw.clone(), v_bits),
    );
    let merge_sub_id = circuit
        .register_sub_circuit(build_refreshed_wire_digit_all_crt_merge::<P>(&ring_gsw.params));
    let seed_wires =
        seed_inputs.iter().flat_map(RingGswCiphertext::sub_circuit_wires).collect::<Vec<_>>();
    let mut outputs = Vec::with_capacity(num_slots * crt_depth * log_base_q);

    for slot_idx in 0..num_slots {
        let prg_sub_id =
            circuit.register_sub_circuit(build_goldreich_encrypted_seeds_with_output::<P, A>(
                ring_gsw.clone(),
                seed_bits,
                v_bits,
                graph_seed,
                cbd_n,
                slot_idx,
            ));
        let prg_outputs = circuit.call_sub_circuit(prg_sub_id, seed_wires.clone());
        let logical_outputs = prg_outputs
            .chunks_exact(ciphertext_wire_count)
            .map(|chunk| RingGswCiphertext::from_sub_circuit_outputs(&ciphertext_template, chunk))
            .collect::<Vec<_>>();
        assert_eq!(logical_outputs.len(), output_sizes.total);
        let (errors, masks) = logical_outputs.split_at(output_sizes.cbd_values);
        let mut by_crt_digit =
            vec![vec![BatchedWire::single(decryption_key); log_base_q]; crt_depth];

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
                let mask_start =
                    crt_idx * log_base_q * mask_q_chunk_len + digit_idx * mask_q_chunk_len;
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
        outputs.extend(by_crt_digit.into_iter().flatten());
    }

    circuit.output(outputs);
    circuit
}

fn apply_unit_column_to_public_keys<M>(
    params: &<M::P as Poly>::Params,
    decoded: Vec<NaiveBGGPublicKeyVec<M>>,
) -> Vec<NaiveBGGPublicKeyVec<M>>
where
    M: PolyMatrix,
{
    let target = M::identity(params, 1, None);
    decoded.into_par_iter().map(|value| value.matrix_mul(params, &target)).collect()
}

fn apply_unit_column_to_encodings<M>(
    params: &<M::P as Poly>::Params,
    decoded: Vec<NaiveBGGEncodingVec<M>>,
) -> Vec<NaiveBGGEncodingVec<M>>
where
    M: PolyMatrix,
{
    let target = M::identity(params, 1, None);
    decoded.into_par_iter().map(|value| value.matrix_mul(params, &target)).collect()
}

fn combine_public_refresh_matrices<M, A, HS>(
    refresher: &NoiseRefresherNaiveVec<M, A, HS>,
    a_prime: &M,
    one: &NaiveBGGPublicKeyVec<M>,
    refreshed_input: &NaiveBGGPublicKeyVec<M>,
    decoded: &[NaiveBGGPublicKeyVec<M>],
    num_slots: usize,
) -> Vec<M>
where
    M: PolyMatrix,
    M::P: 'static,
    A: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    HS: Send + Sync,
{
    let (_q_moduli, _crt_bits, crt_depth) = refresher.params().to_crt();
    let log_base_q = refresher.params().modulus_digits();
    assert_eq!(decoded.len(), num_slots * crt_depth * log_base_q);

    (0..num_slots * crt_depth)
        .into_par_iter()
        .map(|flat_idx| {
            let slot_idx = flat_idx / crt_depth;
            let crt_idx = flat_idx % crt_depth;
            let one_term = one
                .matrix_mul(refresher.params(), &refresher.neg_scaled_a_prime(a_prime, crt_idx))
                .keys[slot_idx]
                .matrix
                .clone();
            let input_term = refreshed_input
                .matrix_mul(refresher.params(), &refresher.scaled_gadget_target(crt_idx))
                .keys[slot_idx]
                .matrix
                .clone();
            let refresh_term =
                decoded_digit_columns(refresher.params(), decoded, slot_idx, crt_idx);
            one_term + &input_term + &refresh_term
        })
        .collect()
}

fn combine_encoding_refresh_vectors<M, A, HS>(
    refresher: &NoiseRefresherNaiveVec<M, A, HS>,
    a_prime: &M,
    one: &NaiveBGGEncodingVec<M>,
    refreshed_input: &NaiveBGGEncodingVec<M>,
    decoded: &[NaiveBGGEncodingVec<M>],
    num_slots: usize,
) -> Vec<M>
where
    M: PolyMatrix,
    M::P: 'static,
    A: DecomposeArithmeticGadget<M::P> + ModularArithmeticPlanner<M::P>,
    HS: Send + Sync,
{
    let (_q_moduli, _crt_bits, crt_depth) = refresher.params().to_crt();
    let log_base_q = refresher.params().modulus_digits();
    assert_eq!(decoded.len(), num_slots * crt_depth * log_base_q);

    (0..num_slots * crt_depth)
        .into_par_iter()
        .map(|flat_idx| {
            let slot_idx = flat_idx / crt_depth;
            let crt_idx = flat_idx % crt_depth;
            let one_term = one
                .matrix_mul(refresher.params(), &refresher.neg_scaled_a_prime(a_prime, crt_idx))
                .encodings[slot_idx]
                .vector
                .clone();
            let input_term = refreshed_input
                .matrix_mul(refresher.params(), &refresher.scaled_gadget_target(crt_idx))
                .encodings[slot_idx]
                .vector
                .clone();
            let refresh_term =
                decoded_encoding_digit_columns(refresher.params(), decoded, slot_idx, crt_idx);
            one_term + &input_term + &refresh_term
        })
        .collect()
}

fn decoded_digit_columns<M>(
    params: &<M::P as Poly>::Params,
    decoded: &[NaiveBGGPublicKeyVec<M>],
    slot_idx: usize,
    crt_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    let (_q_moduli, _crt_bits, crt_depth) = params.to_crt();
    let log_base_q = params.modulus_digits();
    let columns = (0..log_base_q)
        .map(|digit_idx| {
            let idx = slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx;
            collapse_public_key_slots(params, &decoded[idx])
        })
        .collect::<Vec<_>>();
    concat_owned_columns(columns)
}

fn decoded_encoding_digit_columns<M>(
    params: &<M::P as Poly>::Params,
    decoded: &[NaiveBGGEncodingVec<M>],
    slot_idx: usize,
    crt_idx: usize,
) -> M
where
    M: PolyMatrix,
{
    let (_q_moduli, _crt_bits, crt_depth) = params.to_crt();
    let log_base_q = params.modulus_digits();
    let columns = (0..log_base_q)
        .map(|digit_idx| {
            let idx = slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx;
            collapse_encoding_slots(params, &decoded[idx])
        })
        .collect::<Vec<_>>();
    concat_owned_columns(columns)
}

fn collapse_public_key_slots<M>(
    params: &<M::P as Poly>::Params,
    input: &NaiveBGGPublicKeyVec<M>,
) -> M
where
    M: PolyMatrix,
{
    let ring_dim = params.ring_dimension() as usize;
    assert_eq!(input.num_slots(), ring_dim);
    (0..ring_dim)
        .map(|slot_idx| input.keys[slot_idx].matrix.clone() * monomial_poly::<M>(params, slot_idx))
        .reduce(|acc, term| acc + &term)
        .expect("ring_dim must be positive")
}

fn collapse_encoding_slots<M>(params: &<M::P as Poly>::Params, input: &NaiveBGGEncodingVec<M>) -> M
where
    M: PolyMatrix,
{
    let ring_dim = params.ring_dimension() as usize;
    assert_eq!(input.num_slots(), ring_dim);
    (0..ring_dim)
        .map(|slot_idx| {
            input.encodings[slot_idx].vector.clone() * monomial_poly::<M>(params, slot_idx)
        })
        .reduce(|acc, term| acc + &term)
        .expect("ring_dim must be positive")
}

fn crt_recompose_rows<M>(params: &<M::P as Poly>::Params, crt_values: &[M], num_slots: usize) -> M
where
    M: PolyMatrix,
{
    let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
    assert_eq!(crt_values.len(), num_slots * crt_depth);
    let reconst_coeffs = params.reconst_coeffs();
    let rows = (0..num_slots)
        .map(|slot_idx| {
            let mut row = M::zero(params, 1, params.modulus_digits());
            for crt_idx in 0..crt_depth {
                let level = &crt_values[slot_idx * crt_depth + crt_idx];
                let rounded = round_scaled_matrix_to_crt(params, level, q_moduli[crt_idx]);
                row.add_in_place(
                    &(rounded * constant_poly::<M>(params, reconst_coeffs[crt_idx].clone())),
                );
            }
            row
        })
        .collect::<Vec<_>>();
    concat_owned_rows(rows)
}

fn round_scaled_matrix_to_crt<M>(params: &<M::P as Poly>::Params, matrix: &M, q_i: u64) -> M
where
    M: PolyMatrix,
{
    let (rows, cols) = matrix.size();
    let q: Arc<BigUint> = params.modulus().into();
    let half_q = q.as_ref() / BigUint::from(2u64);
    let q_i_big = BigUint::from(q_i);
    let mut out = M::zero(params, rows, cols);
    for row in 0..rows {
        for col in 0..cols {
            let coeffs = matrix.entry(row, col).coeffs_biguints();
            let rounded = coeffs
                .into_iter()
                .map(|coeff| ((&q_i_big * coeff + &half_q) / q.as_ref()) % &q_i_big)
                .collect::<Vec<_>>();
            out.set_entry(row, col, M::P::from_biguints(params, &rounded));
        }
    }
    out
}

fn concat_owned_columns<M: PolyMatrix>(mut matrices: Vec<M>) -> M {
    let first = matrices.remove(0);
    first.concat_columns_owned(matrices)
}

fn concat_owned_rows<M: PolyMatrix>(mut matrices: Vec<M>) -> M {
    let first = matrices.remove(0);
    first.concat_rows_owned(matrices)
}

fn constant_poly<M>(params: &<M::P as Poly>::Params, value: BigUint) -> M::P
where
    M: PolyMatrix,
{
    M::P::from_biguint_to_constant(params, value)
}

fn monomial_poly<M>(params: &<M::P as Poly>::Params, exponent: usize) -> M::P
where
    M: PolyMatrix,
{
    M::P::const_rotate_poly(params, exponent)
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
}
