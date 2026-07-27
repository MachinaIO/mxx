use std::{cell::RefCell, marker::PhantomData, sync::Arc};

use num_bigint::BigUint;
use rayon::prelude::*;

use crate::{
    bgg::{
        encoding::BggEncoding,
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
    },
    decoder::{
        Decoder,
        artifact::{DecoderArtifactSink, DecoderArtifactSource},
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};

/// Decode one coefficient that should contain `(q / plaintext_modulus) * value`
/// plus a centered mask.
pub fn decode_centered_masked_integer_coeff(
    coeff: BigUint,
    q_modulus: &BigUint,
    plaintext_modulus: &BigUint,
) -> BigUint {
    assert!(*plaintext_modulus > BigUint::from(1u32), "plaintext modulus must be at least two");
    let half_q = q_modulus / 2u32;
    ((plaintext_modulus * coeff + half_q) / q_modulus) % plaintext_modulus
}

/// Decode one coefficient that should contain `q/2 * bit` plus a centered mask.
pub fn decode_centered_masked_boolean_coeff(coeff: BigUint, q_modulus: &BigUint) -> bool {
    decode_centered_masked_integer_coeff(coeff, q_modulus, &BigUint::from(2u32)) ==
        BigUint::from(1u32)
}

/// Decode every coefficient in a matrix to `plaintext_modulus` by centered
/// nearest-integer rounding modulo the full ciphertext modulus.
pub fn decode_centered_masked_matrix<M>(
    params: &<M::P as Poly>::Params,
    noisy_plaintext: &M,
    plaintext_modulus: &BigUint,
) -> M
where
    M: PolyMatrix,
{
    let q: Arc<BigUint> = params.modulus().into();
    let entries = (0..noisy_plaintext.row_size())
        .flat_map(|row_idx| (0..noisy_plaintext.col_size()).map(move |col_idx| (row_idx, col_idx)))
        .collect::<Vec<_>>();
    let rounded_entries = entries
        .into_par_iter()
        .map(|(row_idx, col_idx)| {
            let rounded_coeffs = noisy_plaintext
                .entry(row_idx, col_idx)
                .coeffs_biguints()
                .into_iter()
                .map(|coeff| {
                    decode_centered_masked_integer_coeff(coeff, q.as_ref(), plaintext_modulus)
                })
                .collect::<Vec<_>>();
            (row_idx, col_idx, M::P::from_biguints(params, &rounded_coeffs))
        })
        .collect::<Vec<_>>();
    let mut rounded = M::zero(params, noisy_plaintext.row_size(), noisy_plaintext.col_size());
    for (row_idx, col_idx, poly) in rounded_entries {
        rounded.set_entry(row_idx, col_idx, poly);
    }
    rounded
}

/// Public-key-side inputs for a masked high-bit decoder.
#[derive(Debug, Clone)]
pub struct MaskedHighBitPreprocessInput<M: PolyMatrix> {
    /// One secret-dependent public-key vector per decoded output.
    pub secret_dependent_outputs: Vec<NaiveBGGPublicKeyVec<M>>,
}

/// Summary produced after decoder preimage artifacts are persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskedHighBitPreprocessOutput {
    /// Total number of per-slot decoder preimages written by preprocessing.
    ///
    /// Each slot of each logical decoder output gets its own target and
    /// preimage artifact, so this is the sum of `slot_counts`.
    pub decoder_count: usize,
    /// Number of per-slot decoder artifacts associated with each logical
    /// decoded output, in output order.
    pub slot_counts: Vec<usize>,
}

/// Encoding-side output pair for one masked high-bit value.
#[derive(Debug, Clone)]
pub struct MaskedHighBitEvaluatedOutput<M: PolyMatrix> {
    pub secret_dependent: NaiveBGGEncodingVec<M>,
    pub public_bottom: NaiveBGGEncodingVec<M>,
}

/// Encoding-side inputs for online masked high-bit decoding.
#[derive(Debug, Clone)]
pub struct MaskedHighBitOnlineInput<M: PolyMatrix> {
    /// Final Diamond/BGG decoder state used to project persisted preimages.
    pub decoder_state: M,
    /// One `(secret-dependent, public-bottom)` pair per decoded output.
    pub outputs: Vec<MaskedHighBitEvaluatedOutput<M>>,
    /// Plaintext modulus for each decoded output. `2` gives boolean high-bit
    /// decoding; CRT noise-refresh can use the matching `q_i`.
    pub plaintext_moduli: Vec<BigUint>,
}

/// Online decode result, including projected decoder outputs for diagnostics.
#[derive(Debug, Clone)]
pub struct MaskedHighBitDecodeOutput<M: PolyMatrix> {
    /// Rounded decoded coefficients for each logical output, row-major over
    /// matrix entries and then coefficient order inside each polynomial.
    pub decoded: Vec<Vec<BigUint>>,
    pub decoder_outputs: Vec<M>,
    pub slot_counts: Vec<usize>,
}

/// Shared decoder for values represented as a high bit plus a centered mask.
///
/// The protocol supplies the trapdoor-preimage sampler as a closure. This keeps
/// graph seed derivation, hash-domain separation, and trapdoor ownership in the
/// calling protocol while centralizing target layout, artifact persistence, and
/// online cancellation.
pub struct MaskedHighBitDecoder<M, A, PS, ID>
where
    M: PolyMatrix,
    A: DecoderArtifactSink + DecoderArtifactSource,
    PS: Fn(usize, &M) -> M,
    ID: Fn(usize) -> String,
{
    params: <M::P as Poly>::Params,
    secret_size: usize,
    artifacts: RefCell<A>,
    preimage_sampler: PS,
    artifact_id: ID,
    _m: PhantomData<M>,
}

impl<M, A, PS, ID> MaskedHighBitDecoder<M, A, PS, ID>
where
    M: PolyMatrix,
    A: DecoderArtifactSink + DecoderArtifactSource,
    PS: Fn(usize, &M) -> M,
    ID: Fn(usize) -> String,
{
    pub fn new(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        artifacts: A,
        preimage_sampler: PS,
        artifact_id: ID,
    ) -> Self {
        Self {
            params: params.clone(),
            secret_size,
            artifacts: RefCell::new(artifacts),
            preimage_sampler,
            artifact_id,
            _m: PhantomData,
        }
    }

    fn identity_selector(&self) -> M {
        M::identity(&self.params, self.secret_size, None).slice_columns(0, 1)
    }

    pub fn projected_public_key_target(&self, public_key_matrix: &M) -> M {
        let identity_selector = self.identity_selector();
        self.projected_public_key_target_with_selector(public_key_matrix, &identity_selector)
    }

    pub fn projected_public_key_target_with_selector(
        &self,
        public_key_matrix: &M,
        identity_selector: &M,
    ) -> M {
        let top = public_key_matrix.mul_decompose(identity_selector);
        let bottom = M::zero(&self.params, self.secret_size, top.col_size());
        top.concat_rows(&[&bottom])
    }

    pub fn preprocess_target(&self, decoder_idx: usize, target: &M) {
        let preimage = (self.preimage_sampler)(decoder_idx, target);
        self.artifacts.borrow_mut().write_matrix(&(self.artifact_id)(decoder_idx), &preimage);
    }

    pub fn preprocess_public_key_matrix(&self, decoder_idx: usize, public_key_matrix: &M) {
        let target = self.projected_public_key_target(public_key_matrix);
        self.preprocess_target(decoder_idx, &target);
    }

    pub fn preprocess_public_key_matrix_with_selector(
        &self,
        decoder_idx: usize,
        public_key_matrix: &M,
        identity_selector: &M,
    ) {
        let target =
            self.projected_public_key_target_with_selector(public_key_matrix, identity_selector);
        self.preprocess_target(decoder_idx, &target);
    }

    fn projected_decoder_output(&self, decoder_state: &M, decoder_idx: usize) -> M {
        let preimage = self
            .artifacts
            .borrow()
            .read_matrix::<M>(&self.params, &(self.artifact_id)(decoder_idx));
        decoder_state.clone() * &preimage
    }

    /// Decode from artifact offset zero.
    ///
    /// `retain_decoder_outputs` is intended for test diagnostics that need to
    /// inspect `decoder_state * preimage` directly. Production callers should
    /// leave it `false` so projected decoder outputs are streamed and dropped.
    pub fn online_decode_with_decoder_output_retention(
        &self,
        input: MaskedHighBitOnlineInput<M>,
        retain_decoder_outputs: bool,
    ) -> MaskedHighBitDecodeOutput<M> {
        self.online_decode_with_offset_and_decoder_output_retention(
            input,
            0,
            retain_decoder_outputs,
        )
    }

    /// Decode a contiguous range of logical outputs whose preimages start at
    /// `initial_decoder_offset`.
    ///
    /// The online cancellation for each slot computes
    /// `decoder_state * preimage - secret_dependent + public_bottom`, then
    /// rounds that centered masked plaintext modulo the requested plaintext
    /// modulus. Offsets let callers process a large decoder one shard at a
    /// time while still addressing the globally persisted decoder artifacts.
    ///
    /// When `retain_decoder_outputs` is true, all projected decoder outputs are
    /// materialized once and returned in `MaskedHighBitDecodeOutput`; this is
    /// useful for relation checks in tests. When false, each projected output
    /// is loaded, used, and dropped immediately to reduce peak memory.
    pub fn online_decode_with_offset_and_decoder_output_retention(
        &self,
        input: MaskedHighBitOnlineInput<M>,
        initial_decoder_offset: usize,
        retain_decoder_outputs: bool,
    ) -> MaskedHighBitDecodeOutput<M> {
        let identity_selector = self.identity_selector();
        let slot_counts = input
            .outputs
            .iter()
            .map(|output| output.secret_dependent.num_slots())
            .collect::<Vec<_>>();
        let decoder_count = slot_counts.iter().sum::<usize>();
        // Diagnostic retention lets tests compare every projected decoder
        // output against an expected secret/public-key product. Production
        // paths keep this empty and recompute one projection per slot below.
        let decoder_outputs = if retain_decoder_outputs {
            (0..decoder_count)
                .map(|decoder_idx| {
                    self.projected_decoder_output(
                        &input.decoder_state,
                        initial_decoder_offset + decoder_idx,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        assert_eq!(
            input.outputs.len(),
            input.plaintext_moduli.len(),
            "masked decoder output count must match plaintext modulus count"
        );
        let decoded = input
            .outputs
            .iter()
            .zip(input.plaintext_moduli.iter())
            .zip(slot_counts.iter().scan(0usize, |offset, slots| {
                let current = *offset;
                *offset += *slots;
                Some(current)
            }))
            .map(|((output, plaintext_modulus), decoder_offset)| {
                let mut rounded = Vec::new();
                assert_eq!(
                    output.secret_dependent.num_slots(),
                    output.public_bottom.num_slots(),
                    "masked decoder secret-dependent and public-bottom slot counts must match"
                );
                for slot_idx in 0..output.secret_dependent.num_slots() {
                    let evaluated = output.secret_dependent.encoding(slot_idx);
                    let public_bottom_encoding = output.public_bottom.encoding(slot_idx);
                    let public_bottom = public_bottom_encoding
                        .plaintext
                        .as_ref()
                        .expect("masked high-bit public-bottom output must reveal plaintext");
                    let decoder_idx = decoder_offset + slot_idx;
                    let projected_decoder_output = if retain_decoder_outputs {
                        decoder_outputs[decoder_idx].clone()
                    } else {
                        self.projected_decoder_output(
                            &input.decoder_state,
                            initial_decoder_offset + decoder_idx,
                        )
                    };
                    let noisy_plaintext = projected_decoder_output -
                        &evaluated.vector.mul_decompose(&identity_selector) +
                        &M::from_poly_vec_row(&self.params, vec![public_bottom.clone()]);
                    let rounded_matrix = decode_centered_masked_matrix(
                        &self.params,
                        &noisy_plaintext,
                        plaintext_modulus,
                    );
                    for row_idx in 0..rounded_matrix.row_size() {
                        for col_idx in 0..rounded_matrix.col_size() {
                            rounded
                                .extend(rounded_matrix.entry(row_idx, col_idx).coeffs_biguints());
                        }
                    }
                }
                rounded
            })
            .collect();
        MaskedHighBitDecodeOutput { decoded, decoder_outputs, slot_counts }
    }

    pub fn online_decode_first_coeff_with_offset(
        &self,
        input: MaskedHighBitOnlineInput<M>,
        initial_decoder_offset: usize,
    ) -> (Vec<BigUint>, Vec<usize>) {
        let slot_counts = input
            .outputs
            .iter()
            .map(|output| output.secret_dependent.num_slots())
            .collect::<Vec<_>>();
        assert_eq!(
            input.outputs.len(),
            input.plaintext_moduli.len(),
            "masked decoder output count must match plaintext modulus count"
        );
        let decoded = input
            .outputs
            .iter()
            .zip(input.plaintext_moduli.iter())
            .zip(slot_counts.iter().scan(0usize, |offset, slots| {
                let current = *offset;
                *offset += *slots;
                Some(current)
            }))
            .map(|((output, plaintext_modulus), decoder_offset)| {
                assert_eq!(
                    output.secret_dependent.num_slots(),
                    output.public_bottom.num_slots(),
                    "masked decoder secret-dependent and public-bottom slot counts must match"
                );
                self.online_decode_coeffs_from_slots(
                    &input.decoder_state,
                    plaintext_modulus,
                    initial_decoder_offset + decoder_offset,
                    output.secret_dependent.num_slots(),
                    |slot_idx| {
                        (
                            output.secret_dependent.encoding(slot_idx),
                            output.public_bottom.encoding(slot_idx),
                        )
                    },
                )
                .into_iter()
                .next()
                .expect("masked decoder output must decode at least one coefficient")
            })
            .collect::<Vec<_>>();
        (decoded, slot_counts)
    }

    /// Stream masked slot encodings and decode every coefficient in every slot.
    ///
    /// This path is used by DiamondIO's final PRF mask output when the caller
    /// wants to avoid materializing a full `NaiveBGGEncodingVec` for one logical
    /// output. It still consumes every slot and every polynomial coefficient, so
    /// masking all coefficients has the same semantics as the ordinary
    /// `online_decode_range` path; callers may choose to keep only the first
    /// decoded coefficient afterward when the public API returns one scalar bit.
    pub fn online_decode_coeffs_from_slots<F>(
        &self,
        decoder_state: &M,
        plaintext_modulus: &BigUint,
        initial_decoder_offset: usize,
        slot_count: usize,
        slot_encoding: F,
    ) -> Vec<BigUint>
    where
        F: FnMut(usize) -> (BggEncoding<M>, BggEncoding<M>),
    {
        let identity_selector = self.identity_selector();
        let mut slot_encoding = slot_encoding;
        let mut decoded = Vec::new();
        for slot_idx in 0..slot_count {
            let (evaluated, public_bottom_encoding) = slot_encoding(slot_idx);
            let public_bottom = public_bottom_encoding
                .plaintext
                .as_ref()
                .expect("masked high-bit public-bottom output must reveal plaintext");
            let projected_decoder_output =
                self.projected_decoder_output(decoder_state, initial_decoder_offset + slot_idx);
            let noisy_plaintext = projected_decoder_output -
                &evaluated.vector.mul_decompose(&identity_selector) +
                &M::from_poly_vec_row(&self.params, vec![public_bottom.clone()]);
            let rounded_matrix =
                decode_centered_masked_matrix(&self.params, &noisy_plaintext, plaintext_modulus);
            for row_idx in 0..rounded_matrix.row_size() {
                for col_idx in 0..rounded_matrix.col_size() {
                    decoded.extend(rounded_matrix.entry(row_idx, col_idx).coeffs_biguints());
                }
            }
        }
        decoded
    }
}

impl<M, A, PS, ID> Decoder for MaskedHighBitDecoder<M, A, PS, ID>
where
    M: PolyMatrix,
    A: DecoderArtifactSink + DecoderArtifactSource,
    PS: Fn(usize, &M) -> M,
    ID: Fn(usize) -> String,
{
    type PreprocessInput = MaskedHighBitPreprocessInput<M>;
    type PreprocessOutput = MaskedHighBitPreprocessOutput;
    type OnlineInput = MaskedHighBitOnlineInput<M>;
    type Output = MaskedHighBitDecodeOutput<M>;

    fn preprocess(&self, input: Self::PreprocessInput) -> Self::PreprocessOutput {
        let mut decoder_count = 0usize;
        let mut slot_counts = Vec::with_capacity(input.secret_dependent_outputs.len());
        for secret_dependent in input.secret_dependent_outputs {
            slot_counts.push(secret_dependent.num_slots());
            for slot_idx in 0..secret_dependent.num_slots() {
                let public_key = secret_dependent.key(slot_idx);
                self.preprocess_public_key_matrix(decoder_count, &public_key.matrix);
                decoder_count += 1;
            }
        }
        MaskedHighBitPreprocessOutput { decoder_count, slot_counts }
    }

    fn online_decode_range(
        &self,
        input: Self::OnlineInput,
        initial_decoder_offset: usize,
    ) -> Self::Output {
        self.online_decode_with_offset_and_decoder_output_retention(
            input,
            initial_decoder_offset,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_centered_masked_boolean_coeff, decode_centered_masked_integer_coeff,
        decode_centered_masked_matrix,
    };
    use crate::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_bigint::BigUint;

    #[test]
    fn centered_masked_boolean_coeff_decode_matches_rounding_regions() {
        let q = BigUint::from(1_048_583u64);
        let half_q = &q / 2u32;
        let safe_radius = &q / 4u32;
        assert!(!decode_centered_masked_boolean_coeff(BigUint::ZERO, &q));
        assert!(!decode_centered_masked_boolean_coeff(&safe_radius - 1u32, &q));
        assert!(decode_centered_masked_boolean_coeff(half_q.clone(), &q));
        assert!(decode_centered_masked_boolean_coeff(&half_q + &safe_radius - 1u32, &q));
    }

    #[test]
    fn centered_masked_integer_coeff_decode_supports_arbitrary_modulus() {
        let q = BigUint::from(1_048_583u64);
        let t = BigUint::from(17u32);
        let value = BigUint::from(9u32);
        let encoded = (&q * &value) / &t;
        assert_eq!(decode_centered_masked_integer_coeff(encoded, &q, &t), value);
    }

    #[test]
    fn centered_masked_matrix_decode_preserves_shape() {
        let params = DCRTPolyParams::new(4, 2, 10, 5);
        let q = params.modulus().as_ref().clone();
        let t = BigUint::from(7u32);
        let values = [BigUint::from(1u32), BigUint::from(5u32)];
        let encoded = values
            .iter()
            .map(|value| DCRTPoly::from_biguint_to_constant(&params, (&q * value) / &t))
            .collect::<Vec<_>>();
        let matrix = DCRTPolyMatrix::from_poly_vec_row(&params, encoded);
        let decoded: DCRTPolyMatrix = decode_centered_masked_matrix(&params, &matrix, &t);
        assert_eq!(decoded.size(), (1, 2));
        assert_eq!(decoded.entry(0, 0).coeffs_biguints()[0], values[0]);
        assert_eq!(decoded.entry(0, 1).coeffs_biguints()[0], values[1]);
    }
}
