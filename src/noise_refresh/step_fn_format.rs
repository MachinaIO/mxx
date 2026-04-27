//! Formatting and evaluation helpers for noise-refresh PRG material.
//!
//! This module intentionally treats PRG material as ordinary circuit inputs once it reaches the
//! formatting phase.  The decrypt/combine circuits below do not care whether the supplied
//! `enc_next_seed`, error, and mask ciphertexts were produced by the real Goldreich PRG, by a
//! benchmark shortcut, or by a caller-provided fixture.  They only require the same layout and
//! enough ciphertexts to run the Ring-GSW decrypt and CRT combination logic.

use crate::{
    circuit::{BatchedWire, PolyCircuit, evaluable::Evaluable, gate::GateId},
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::BooleanCiphertext,
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::{ops::Range, sync::Arc};

use super::step_fn_prg::{NoiseRefreshPrgParameters, build_goldreich_encrypted_seeds_with_output};

/// Flattens logical Boolean ciphertexts into the wire list expected by a subcircuit call.
///
/// This is the only place in the formatter where logical ciphertext grouping is intentionally
/// erased.  All layout checks happen before or after this flattening step.
fn ciphertext_wires<P, C>(ciphertexts: &[C]) -> Vec<BatchedWire>
where
    P: Poly,
    C: BooleanCiphertext<P>,
{
    // Subcircuit calls operate on flattened wire lists.  This helper erases the logical ciphertext
    // grouping when feeding a child circuit.
    ciphertexts.iter().flat_map(|ciphertext| ciphertext.sub_circuit_wires()).collect()
}

/// Reconstructs logical ciphertext wrappers from a flat subcircuit output stream.
///
/// `templates` define how many wires each logical ciphertext consumes and how to wrap those wires
/// back into the concrete ciphertext type.  `next_output_start` is advanced monotonically so the
/// caller can recover `enc_next_seed`, then errors, then masks from one flat output list.
fn recover_ciphertexts_from_outputs<P, C>(
    templates: &[C],
    outputs: &[BatchedWire],
    next_output_start: &mut usize,
) -> Vec<C>
where
    P: Poly,
    C: BooleanCiphertext<P>,
{
    // A subcircuit call returns only a flat output list.  The templates describe how many wires
    // belong to each logical ciphertext and how to rebuild that ciphertext wrapper around the newly
    // produced output gates.
    templates
        .iter()
        .map(|template| {
            let output_gate_count =
                template.sub_circuit_wires().into_iter().map(|wire| wire.len()).sum::<usize>();
            let next_output_end = *next_output_start + output_gate_count;
            assert!(
                next_output_end <= outputs.len(),
                "subcircuit output ended before all ciphertext outputs were recovered"
            );
            let output = C::from_sub_circuit_outputs(
                template,
                &outputs[*next_output_start..next_output_end],
            );
            *next_output_start = next_output_end;
            output
        })
        .collect()
}

/// Adds a nonempty list of scalar wires with the circuit's ordinary addition gate.
fn sum_gate_ids<P: Poly>(circuit: &mut PolyCircuit<P>, values: &[GateId]) -> GateId {
    let (first, rest) = values.split_first().expect("at least one gate is required");
    rest.iter().fold(*first, |acc, value| circuit.add_gate(acc, *value).as_single_wire())
}

/// Decrypts one coefficient-level error polynomial.
///
/// The input layout is exactly `ring_dim` coefficient ciphertexts for one gadget digit.  A single
/// `decrypt_batch` call decrypts all coefficients and places the recovered coefficient values in
/// the corresponding slots of one output wire.  There is no second slot-chunk axis here: this wire
/// is the error polynomial that will be added to one BGG+ encoding column.
fn decrypt_error_coefficients_as_polynomial<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_coefficients: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_modulus: BigUint,
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
    let first_ctx: Arc<RingGswContext<P, A>> = encrypted_coefficients
        .first()
        .expect("at least one encrypted coefficient is required")
        .ctx
        .clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    assert_eq!(
        encrypted_coefficients.len(),
        ring_dim,
        "encrypted coefficient count {} must match ring_dim {}",
        encrypted_coefficients.len(),
        ring_dim
    );

    let ciphertexts = encrypted_coefficients.iter().collect::<Vec<_>>();
    RingGswCiphertext::decrypt_batch::<M>(
        ciphertexts.as_slice(),
        decryption_key,
        plaintext_modulus,
        circuit,
    )
}

/// Decrypts one bit-decomposed polynomial.
///
/// The input layout is `ring_dim * bit_size` ciphertexts ordered by coefficient, then bit index.
/// For each bit index, one `decrypt_batch` call decrypts the `ring_dim` coefficient ciphertexts and
/// places those coefficient values in one slotwise polynomial wire.  The bit-polynomial wires are
/// then summed.  This matches the refresh layout: one mask polynomial is needed per gadget digit
/// and CRT level, not one extra polynomial per output slot.
///
/// The plaintext moduli are supplied by the caller instead of recomputed here.  The formatter has
/// one fixed CRT modulus for the current subcircuit, so callers should derive this slice once from
/// the cached `params.to_crt()` data for that CRT level and reuse it across all mask polynomial
/// decryptions in the subcircuit.
fn decrypt_bit_decomposed_polynomial<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_moduli: &[BigUint],
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let bit_size = plaintext_moduli.len();
    assert!(bit_size > 0, "bit_size must be positive");
    assert!(
        plaintext_moduli.iter().all(|modulus| !modulus.is_zero()),
        "all bit plaintext moduli must be positive"
    );
    let first_ctx: Arc<RingGswContext<P, A>> =
        encrypted_bits.first().expect("at least one encrypted bit is required").ctx.clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    let expected_len = ring_dim.checked_mul(bit_size).expect("Ring-GSW bit chunk length overflow");
    assert_eq!(
        encrypted_bits.len(),
        expected_len,
        "encrypted bit chunk must equal ring_dim * bit_size"
    );
    let bit_terms = (0..bit_size)
        .map(|bit_idx| {
            let ciphertexts = (0..ring_dim)
                .map(|coeff_idx| &encrypted_bits[coeff_idx * bit_size + bit_idx])
                .collect::<Vec<_>>();
            RingGswCiphertext::decrypt_batch::<M>(
                ciphertexts.as_slice(),
                decryption_key,
                plaintext_moduli[bit_idx].clone(),
                circuit,
            )
        })
        .collect::<Vec<_>>();
    sum_gate_ids(circuit, &bit_terms)
}

/// Builds one CRT-specific formatter subcircuit for a single output chunk.
///
/// Inputs are one flattened encrypted next-seed wire, all error ciphertexts for that chunk, and
/// only the mask ciphertexts for the selected CRT level.  Outputs are:
/// 1. the encrypted next-seed wire scaled by `q / q_i`, and
/// 2. `log_base_q` decoded refresh wires, each equal to decrypted error plus decrypted mask.
fn build_enc_next_seed_chunk_crt_formatter_subcircuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    v_bits: usize,
    plaintext_modulus: BigUint,
    full_q: &BigUint,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(v_bits > 0, "v_bits must be positive");
    assert!(!plaintext_modulus.is_zero(), "CRT plaintext modulus must be positive");

    let mut circuit = ring_gsw.fresh_circuit();
    let enc_next_seed_wire = circuit.input(1).as_single_wire();
    let decryption_key = GateId(0);
    let ring_dim = ring_gsw.ring_dim();
    let log_base_q = ring_gsw.log_base_q();
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let mask_plaintext_moduli = plaintext_moduli_from_crt_modulus(&plaintext_modulus, v_bits);

    let errors = (0..log_base_q * ring_dim)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let masks = (0..log_base_q * mask_q_chunk_len)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    let q_over_qi = full_q / &plaintext_modulus;
    let enc_next_seed_output = circuit.large_scalar_mul(enc_next_seed_wire, &[q_over_qi]);
    let mut decoded_refresh_outputs = Vec::with_capacity(log_base_q);
    for digit_idx in 0..log_base_q {
        let error_start = digit_idx * ring_dim;
        let digit_errors = &errors[error_start..error_start + ring_dim];
        let decrypted_error = decrypt_error_coefficients_as_polynomial::<P, A, M>(
            &mut circuit,
            digit_errors,
            decryption_key,
            plaintext_modulus.clone(),
        );

        let mask_start = digit_idx * mask_q_chunk_len;
        let digit_masks = &masks[mask_start..mask_start + mask_q_chunk_len];
        let decrypted_mask = decrypt_bit_decomposed_polynomial::<P, A, M>(
            &mut circuit,
            digit_masks,
            decryption_key,
            &mask_plaintext_moduli,
        );

        decoded_refresh_outputs
            .push(circuit.add_gate(decrypted_error, decrypted_mask).as_single_wire());
    }

    let mut outputs = Vec::with_capacity(1 + decoded_refresh_outputs.len());
    outputs.push(enc_next_seed_output);
    outputs.extend(decoded_refresh_outputs.iter().copied().map(BatchedWire::single));
    circuit.output(outputs);
    circuit
}

/// Builds one all-CRT formatter subcircuit for a single output chunk.
///
/// `PolyParams::to_crt()` owns the cached CRT modulus list for the polynomial parameters.  This
/// helper reads that list once, registers one CRT-specific chunk formatter per `q_i`, calls each
/// formatter with the shared encrypted next-seed wire and the CRT-specific mask slice, and exposes
/// the concatenated child outputs as its own outputs.
fn build_enc_next_seed_chunk_all_crt_formatter<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    v_bits: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(v_bits > 0, "v_bits must be positive");
    let (q_moduli, _crt_bits, q_moduli_depth) = ring_gsw.params.to_crt();
    assert_eq!(
        q_moduli_depth,
        ring_gsw.crt_depth(),
        "Ring-GSW CRT depth must match params.to_crt()"
    );
    let full_q: Arc<BigUint> = ring_gsw.params.modulus().into();

    let mut circuit = ring_gsw.fresh_circuit();
    let enc_next_seed_wire = BatchedWire::single(circuit.input(1).as_single_wire());
    let ring_dim = ring_gsw.ring_dim();
    let log_base_q = ring_gsw.log_base_q();
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let errors = (0..log_base_q * ring_dim)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let masks = (0..q_moduli_depth * log_base_q * mask_q_chunk_len)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    let formatter_subcircuit_ids = q_moduli
        .iter()
        .map(|&q_i| {
            let formatter = build_enc_next_seed_chunk_crt_formatter_subcircuit::<P, A, M>(
                ring_gsw.clone(),
                v_bits,
                BigUint::from(q_i),
                full_q.as_ref(),
            );
            circuit.register_sub_circuit(formatter)
        })
        .collect::<Vec<_>>();

    let error_inputs = ciphertext_wires::<P, RingGswCiphertext<P, A>>(&errors);
    let mut outputs =
        Vec::with_capacity(q_moduli_depth.checked_mul(1 + log_base_q).expect("output overflow"));
    for crt_idx in 0..q_moduli_depth {
        let mask_start = crt_idx
            .checked_mul(log_base_q)
            .and_then(|value| value.checked_mul(mask_q_chunk_len))
            .expect("CRT mask slice start overflow");
        let mask_end = mask_start
            .checked_add(log_base_q.checked_mul(mask_q_chunk_len).expect("CRT mask slice overflow"))
            .expect("CRT mask slice end overflow");
        let mut formatter_inputs = Vec::with_capacity(
            1 + error_inputs.len() +
                log_base_q.checked_mul(mask_q_chunk_len).expect("CRT mask input count overflow"),
        );
        formatter_inputs.push(enc_next_seed_wire);
        formatter_inputs.extend(error_inputs.iter().copied());
        formatter_inputs
            .extend(ciphertext_wires::<P, RingGswCiphertext<P, A>>(&masks[mask_start..mask_end]));
        outputs
            .extend(circuit.call_sub_circuit(formatter_subcircuit_ids[crt_idx], formatter_inputs));
    }
    circuit.output(outputs);
    circuit
}

/// Derives mask bit plaintext moduli once for one CRT level.
///
/// The caller supplies the cached CRT modulus `q_i`; bit `j` uses `q_i / 2^j` with one-based bit
/// numbering.  Keeping this derivation outside the coefficient loops avoids rebuilding the same
/// `BigUint` values for every mask polynomial in a formatter subcircuit.
fn plaintext_moduli_from_crt_modulus(crt_modulus: &BigUint, bit_size: usize) -> Vec<BigUint> {
    assert!(bit_size > 0, "bit_size must be positive");
    assert!(!crt_modulus.is_zero(), "CRT modulus must be positive");
    (1..=bit_size)
        .map(|bit_idx| {
            let modulus = crt_modulus >> bit_idx;
            assert!(
                !modulus.is_zero(),
                "CRT modulus / 2^{bit_idx} must be positive for Ring-GSW decrypt"
            );
            modulus
        })
        .collect()
}

/*
/// All shape parameters needed to interpret PRG material in the formatter phase.
///
/// The layout intentionally separates `expanded_seed_bits` from `ciphertext_wire_count`: a selected
/// logical next-seed bit expands to one output chunk per encrypted-bit wire.  Error and mask sizes
/// are then derived from those output chunks, not specified independently.
#[derive(Debug, Clone)]
struct NextSeedEncodingLayout {
    seed_bits: usize,
    input_bits_per_step: usize,
    ciphertext_wire_count: usize,
    ring_dim: usize,
    log_base_q: usize,
    crt_depth: usize,
    v_bits: usize,
    expanded_seed_bits: usize,
    errors_per_seed: usize,
    mask_q_chunk_len: usize,
    masks_per_seed: usize,
}

/// Selected PRG material range after applying an optional output step index.
///
/// `output_selection` is a range over conceptual encrypted next-seed logical outputs.  The
/// formatter consumes `output_chunk_count = output_selection.len() * ciphertext_wire_count` flat
/// next-seed wires.  When `output_selection` is `None`, the full step material is selected.
#[derive(Debug, Clone)]
struct SelectedPrgMaterial {
    output_chunk_count: usize,
    output_selection: Option<Range<usize>>,
}

impl NextSeedEncodingLayout {
    /// Creates the full protocol layout from Ring-GSW parameters and step-function dimensions.
    fn new<P, A>(
        ring_gsw: &RingGswContext<P, A>,
        seed_bits: usize,
        input_bits_per_step: usize,
        v_bits: usize,
    ) -> Self
    where
        P: Poly,
        A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    {
        let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
        let ring_dim = ring_gsw.ring_dim();
        let log_base_q = ring_gsw.log_base_q();
        let crt_depth = ring_gsw.crt_depth();
        let expanded_seed_bits = seed_bits
            .checked_mul(input_bits_per_step)
            .expect("seed_bits * input_bits_per_step overflow");
        let errors_per_seed = ciphertext_wire_count
            .checked_mul(log_base_q)
            .and_then(|value| value.checked_mul(ring_dim))
            .expect("errors per seed overflow");
        let base_mask_group_count =
            ciphertext_wire_count.checked_mul(log_base_q).expect("base mask group count overflow");
        let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
        let mask_group_len =
            mask_q_chunk_len.checked_mul(crt_depth).expect("mask group length overflow");
        let masks_per_seed =
            base_mask_group_count.checked_mul(mask_group_len).expect("masks per seed overflow");

        Self {
            seed_bits,
            input_bits_per_step,
            ciphertext_wire_count,
            ring_dim,
            log_base_q,
            crt_depth,
            v_bits,
            expanded_seed_bits,
            errors_per_seed,
            mask_q_chunk_len,
            masks_per_seed,
        }
    }

    /// Creates a formatter-only layout for already supplied PRG material.
    ///
    /// This constructor is used by benchmark or fixture circuits that bypass Goldreich generation.
    /// `output_chunk_count` is the number of flattened encrypted next-seed wires that the caller
    /// will provide directly, so it is not required to be a full `seed_bits *
    /// ciphertext_wire_count` step.
    fn for_output_chunks<P, A>(
        ring_gsw: &RingGswContext<P, A>,
        output_chunk_count: usize,
        v_bits: usize,
    ) -> Self
    where
        P: Poly,
        A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    {
        assert!(output_chunk_count > 0, "output_chunk_count must be positive");
        let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
        let seed_bits = output_chunk_count.div_ceil(ciphertext_wire_count);
        let input_bits_per_step = 1usize;
        let mut layout = Self::new(ring_gsw, seed_bits, input_bits_per_step, v_bits);
        layout.expanded_seed_bits = seed_bits;
        layout
    }

    /// Converts an optional step index into the exact PRG material range needed by that step.
    ///
    /// A selected step `t` maps to logical encrypted next-seed outputs
    /// `[t * seed_bits, (t + 1) * seed_bits)`.  Error and mask ranges are derived from that range
    /// by the PRG phase, which prevents callers from selecting inconsistent material.
    fn selected_material(&self, output_step_idx: Option<usize>) -> SelectedPrgMaterial {
        let output_selection = if let Some(output_step_idx) = output_step_idx {
            assert!(
                output_step_idx < self.input_bits_per_step,
                "output step index must be less than input_bits_per_step"
            );
            let start = output_step_idx
                .checked_mul(self.seed_bits)
                .expect("selected next-seed step start overflow");
            let end =
                start.checked_add(self.seed_bits).expect("selected next-seed step end overflow");
            start..end
        } else {
            0..self.expanded_seed_bits
        };
        let selected_output_chunk_count = output_selection
            .len()
            .checked_mul(self.ciphertext_wire_count)
            .expect("selected next-seed wire count overflow");
        SelectedPrgMaterial {
            output_chunk_count: selected_output_chunk_count,
            output_selection: output_step_idx.map(|_| output_selection),
        }
    }

    /// Full number of error ciphertexts for the unselected step function.
    fn full_error_count(&self) -> usize {
        self.expanded_seed_bits
            .checked_mul(self.ciphertext_wire_count)
            .and_then(|value| value.checked_mul(self.log_base_q))
            .and_then(|value| value.checked_mul(self.ring_dim))
            .expect("Goldreich error ciphertext count overflow")
    }

    /// Full number of mask ciphertexts for the unselected step function.
    fn full_mask_count(&self) -> usize {
        self.expanded_seed_bits
            .checked_mul(self.ciphertext_wire_count)
            .and_then(|value| value.checked_mul(self.ring_dim))
            .and_then(|value| value.checked_mul(self.v_bits))
            .and_then(|value| value.checked_mul(self.log_base_q))
            .and_then(|value| value.checked_mul(self.crt_depth))
            .expect("Goldreich mask ciphertext count overflow")
    }

    /// Number of error ciphertexts needed for one flattened encrypted next-seed wire.
    fn errors_per_output(&self) -> usize {
        self.log_base_q.checked_mul(self.ring_dim).expect("errors per output overflow")
    }

    /// Number of mask ciphertexts needed for one flattened encrypted next-seed wire.
    fn masks_per_output(&self) -> usize {
        self.log_base_q
            .checked_mul(self.crt_depth)
            .and_then(|value| value.checked_mul(self.mask_q_chunk_len))
            .expect("masks per output overflow")
    }

    /// Offset of the mask subrange for `(digit_idx, crt_idx)` inside one output chunk.
    fn mask_crt_slice_start(&self, digit_idx: usize, crt_idx: usize) -> usize {
        digit_idx * self.crt_depth * self.mask_q_chunk_len + crt_idx * self.mask_q_chunk_len
    }
}

/// Computes the plaintext moduli used when decrypting bit chunks.
///
/// Bit indices are one-based for the modulus formula: bit `j` uses `scale / 2^j`, so the first bit
/// is decrypted modulo `scale / 2`, the second modulo `scale / 4`, and so on.  Each modulus must be
/// positive because Ring-GSW decryption scales by the inverse of the plaintext modulus.
pub fn ring_gsw_noise_refresh_plaintext_moduli(bit_size: usize, scale: &BigUint) -> Vec<BigUint> {
    assert!(bit_size > 0, "bit_size must be positive");
    assert!(!scale.is_zero(), "scale must be positive");
    (1..=bit_size)
        .map(|bit_idx| {
            let divisor = BigUint::one() << bit_idx;
            let modulus = scale / divisor;
            assert!(
                !modulus.is_zero(),
                "scale / 2^{bit_idx} must be positive for Ring-GSW decrypt"
            );
            modulus
        })
        .collect()
}

/// Decrypts one or more Ring-GSW bit chunks and outputs one polynomial wire per chunk.
///
/// The input length must be a multiple of `ring_dim * bit_size`.  Each chunk is interpreted as
/// `ring_dim` coefficient groups, and each coefficient group contains `bit_size` encrypted bits.
/// For every coefficient group, the circuit decrypts all bits under their bit-dependent plaintext
/// moduli, sums those decrypted bit values, and then sums the coefficient-group results directly.
/// It does not apply an additional coefficient-dependent monomial shift; `decrypt_batch` already
/// performs the slot-reduction step that determines where batched decryptions are stored.
///
/// This helper appends the decrypted chunk wires as outputs of the supplied circuit and returns the
/// corresponding `GateId`s for callers that want to compose more logic before exposing their own
/// final outputs.
pub fn decrypt_ring_gsw_bit_chunks<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    bit_size: usize,
    scale: &BigUint,
) -> Vec<GateId>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(bit_size > 0, "bit_size must be positive");
    assert!(!encrypted_bits.is_empty(), "at least one encrypted bit chunk is required");
    let first_ctx: Arc<RingGswContext<P, A>> =
        encrypted_bits.first().expect("validated non-empty encrypted bits").ctx.clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    let chunk_len = ring_dim.checked_mul(bit_size).expect("Ring-GSW bit chunk length overflow");
    assert_eq!(
        encrypted_bits.len() % chunk_len,
        0,
        "encrypted bit input length {} must be a multiple of ring_dim * bit_size {}",
        encrypted_bits.len(),
        chunk_len
    );
    for (idx, ciphertext) in encrypted_bits.iter().enumerate() {
        assert!(
            Arc::ptr_eq(&ciphertext.ctx, &first_ctx),
            "encrypted bit ciphertext {idx} must share the first Ring-GSW context"
        );
    }

    let outputs = encrypted_bits
        .chunks_exact(chunk_len)
        .map(|chunk| {
            decrypt_bit_decomposed_polynomial::<P, A, M>(
                circuit,
                chunk,
                decryption_key,
                bit_size,
                scale,
            )
        })
        .collect::<Vec<_>>();
    circuit.output(outputs.iter().copied());
    outputs
}

fn append_next_seed_encoding_formatting<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<RingGswContext<P, A>>,
    layout: &NextSeedEncodingLayout,
    enc_next_seed_wires: &[GateId],
    errors: &[RingGswCiphertext<P, A>],
    masks: &[RingGswCiphertext<P, A>],
) -> Vec<BatchedWire>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    // This function is the core formatting phase.  It does not know whether its inputs came from
    // Goldreich or from a benchmark shortcut; it only enforces the layout and runs the decrypt plus
    // CRT-level composition logic.
    assert!(
        layout.ring_dim <= ring_gsw.num_slots,
        "next-seed encoding slot assignment requires ring_dim {} <= Ring-GSW num_slots {}",
        layout.ring_dim,
        ring_gsw.num_slots
    );
    let (q_moduli, _crt_bits, q_moduli_depth) = ring_gsw.params.to_crt();
    assert_eq!(q_moduli_depth, layout.crt_depth, "Ring-GSW CRT depth must match params.to_crt()");
    assert_eq!(q_moduli.len(), layout.crt_depth, "one CRT modulus is required for each CRT level");
    let full_q: Arc<BigUint> = ring_gsw.params.modulus().into();
    let selected_output_chunk_count = enc_next_seed_wires.len();

    let formatter_subcircuit_ids = q_moduli
        .iter()
        .map(|&q_i| {
            let formatter = build_enc_next_seed_chunk_crt_formatter_subcircuit::<P, A, M>(
                ring_gsw.clone(),
                layout.v_bits,
                BigUint::from(q_i),
                full_q.as_ref(),
            );
            circuit.register_sub_circuit(formatter)
        })
        .collect::<Vec<_>>();

    let errors_per_output = layout.errors_per_output();
    let masks_per_output = layout.masks_per_output();
    assert_eq!(errors.len(), selected_output_chunk_count * errors_per_output);
    assert_eq!(masks.len(), selected_output_chunk_count * masks_per_output);

    let mut enc_next_seed_outputs = Vec::with_capacity(
        selected_output_chunk_count
            .checked_mul(layout.crt_depth)
            .expect("next-seed formatter output count overflow"),
    );
    let mut decoded_refresh_encodings = Vec::with_capacity(
        selected_output_chunk_count
            .checked_mul(layout.crt_depth)
            .and_then(|value| value.checked_mul(layout.log_base_q))
            .expect("decoded refresh formatter output count overflow"),
    );
    for output_chunk_idx in 0..selected_output_chunk_count {
        let output_errors = &errors
            [output_chunk_idx * errors_per_output..(output_chunk_idx + 1) * errors_per_output];
        let output_masks =
            &masks[output_chunk_idx * masks_per_output..(output_chunk_idx + 1) * masks_per_output];
        for crt_idx in 0..layout.crt_depth {
            let mut formatter_inputs = Vec::new();
            formatter_inputs.push(BatchedWire::single(enc_next_seed_wires[output_chunk_idx]));
            formatter_inputs.extend(ciphertext_wires::<P, RingGswCiphertext<P, A>>(output_errors));
            for digit_idx in 0..layout.log_base_q {
                let start = layout.mask_crt_slice_start(digit_idx, crt_idx);
                formatter_inputs.extend(ciphertext_wires::<P, RingGswCiphertext<P, A>>(
                    &output_masks[start..start + layout.mask_q_chunk_len],
                ));
            }
            let formatter_outputs =
                circuit.call_sub_circuit(formatter_subcircuit_ids[crt_idx], formatter_inputs);
            assert_eq!(
                formatter_outputs.len(),
                1 + layout.log_base_q,
                "formatter subcircuit must output one next-seed wire and log_base_q refresh wires"
            );
            enc_next_seed_outputs.push(formatter_outputs[0]);
            decoded_refresh_encodings.extend(formatter_outputs[1..].iter().copied());
        }
    }
    assert_eq!(
        enc_next_seed_outputs.len(),
        selected_output_chunk_count * layout.crt_depth,
        "formatter next-seed output count mismatch"
    );
    assert_eq!(
        decoded_refresh_encodings.len(),
        selected_output_chunk_count * layout.crt_depth * layout.log_base_q,
        "formatter decoded refresh output count mismatch"
    );

    let mut circuit_outputs = Vec::new();
    circuit_outputs.extend(enc_next_seed_outputs);
    circuit_outputs.extend(decoded_refresh_encodings);
    circuit_outputs
}

/// Builds only the decrypt-and-combine phase for already available PRG material.
///
/// The circuit inputs are independent of how the PRG material was produced:
/// - `output_chunk_count` scalar wires for flattened encrypted next-seed encodings,
/// - `output_chunk_count * log_base_q * ring_dim` error ciphertexts, and
/// - `output_chunk_count * log_base_q * crt_depth * ring_dim * v_bits` mask ciphertexts.
///
/// This is the phase to use for benchmark shortcuts that reuse or stub PRG outputs.  As long as
/// the caller supplies enough ciphertexts in the expected layout, the circuit runs the same
/// Ring-GSW decrypt and `decrypted_error + decrypted_mask` composition as the full step function.
pub fn build_next_seed_encoding_formatter_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    output_chunk_count: usize,
    v_bits: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(output_chunk_count > 0, "output_chunk_count must be positive");
    assert!(v_bits > 0, "v_bits must be positive");
    let mut circuit = ring_gsw.fresh_circuit();
    let layout =
        NextSeedEncodingLayout::for_output_chunks(ring_gsw.as_ref(), output_chunk_count, v_bits);
    let enc_next_seed_wires =
        (0..output_chunk_count).map(|_| circuit.input(1).as_single_wire()).collect::<Vec<_>>();
    let errors = (0..output_chunk_count * layout.errors_per_output())
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let masks = (0..output_chunk_count * layout.masks_per_output())
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let circuit_outputs = append_next_seed_encoding_formatting::<P, A, M>(
        &mut circuit,
        ring_gsw,
        &layout,
        &enc_next_seed_wires,
        &errors,
        &masks,
    );
    circuit.output(circuit_outputs);
    circuit
}

/// Builds the circuit that outputs the next seed and its decoded refresh encodings.
///
/// The returned circuit has one class of inputs:
/// - `seed_bits` encrypted seed ciphertext inputs.
///
/// Ring-GSW decryption inside this circuit uses the circuit constant-one wire as the decryption-key
/// encoding.  Evaluators already receive that value through `PolyCircuit::eval`'s explicit `one`
/// argument, so the constructed circuit does not allocate a separate decryption-key input.
///
/// Internally the circuit calls the Goldreich encrypted-seed expansion subcircuit.  The child
/// subcircuit outputs `enc_next_seed`, `errors`, and `masks` as flattened ciphertext wires; this
/// builder reconstructs the logical ciphertext wrappers from templates so it can decrypt the
/// `errors` and `masks` with the Ring-GSW gadget decryption API.
///
/// Final output order:
/// 1. the CRT-scaled encrypted `enc_next_seed` encoding wires,
/// 2. one decoded refresh encoding wire per element, computed as `decrypted_error[element] +
///    decrypted_mask[element]`.
///
/// Final output size:
/// - `enc_next_seed` contributes `seed_bits * input_bits_per_step` encrypted bit ciphertexts.  Each
///   encrypted bit is represented by `ciphertext_wire_count` circuit wires.  Each such wire is then
///   multiplied by the CRT-specific scalar `q/q_i` using `large_scalar_mul`, so the flat next-seed
///   output prefix contains `seed_bits * input_bits_per_step * ciphertext_wire_count * crt_depth`
///   wires.
/// - The decoded refresh suffix is already collapsed to slotwise polynomial wires.  For every
///   expanded seed bit there are `ciphertext_wire_count * log_base_q` base refresh groups, and each
///   group is decoded once for each CRT level.  Therefore the suffix contains `seed_bits *
///   input_bits_per_step * ciphertext_wire_count * log_base_q * crt_depth` wires.
/// - Each decoded refresh wire consumes exactly `ring_dim` coefficient ciphertexts for the error
///   polynomial and `ring_dim * v_bits` mask-bit ciphertexts for the same CRT level.  A single
///   `decrypt_batch` call per error digit, and one per mask bit digit, places coefficient values in
///   the slots of the returned polynomial wire.
/// - Errors and masks are not emitted separately.  The final decoded suffix is the element-wise sum
///   `decrypted_errors[i] + decrypted_masks[i]`, so its length is the shared decoded length above,
///   not twice that length.
pub fn build_next_seed_encodings_circuit<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    build_next_seed_encodings_circuit_with_output_step_idx::<P, A, M>(
        ring_gsw,
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
        None,
    )
}

pub fn build_next_seed_encodings_circuit_with_output_step_idx<P, A, M>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    output_step_idx: Option<usize>,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(seed_bits >= 5, "Goldreich PRG requires at least five encrypted seed bits");
    assert!(input_bits_per_step > 0, "input_bits_per_step must be positive");
    assert!(v_bits > 0, "v_bits must be positive");

    // The outer circuit owns only the encrypted seed ciphertext inputs.  The Goldreich expansion
    // itself is registered below as a child subcircuit and called with these seed inputs.  Gadget
    // decryption uses the circuit's constant-one wire as the encoded decryption key, matching the
    // evaluator API where `one` is passed separately from ordinary circuit inputs.
    let mut circuit = ring_gsw.fresh_circuit();
    let encrypted_seeds = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let layout =
        NextSeedEncodingLayout::new(ring_gsw.as_ref(), seed_bits, input_bits_per_step, v_bits);
    let selected_material = layout.selected_material(output_step_idx);

    // Build a private Goldreich expansion circuit and keep its logical output templates before
    // registering it.  Once the subcircuit is called, only flat wires are available, so templates
    // are needed to recover `RingGswCiphertext` handles around those output gates.
    let goldreich_subcircuit =
        build_goldreich_encrypted_seeds_with_output::<P, RingGswCiphertext<P, A>>(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            selected_material.output_selection.clone(),
        );
    let error_templates = goldreich_subcircuit.output.errors.clone();
    let mask_templates = goldreich_subcircuit.output.masks.clone();
    assert_eq!(
        goldreich_subcircuit.output.enc_next_seed.len() * layout.ciphertext_wire_count,
        selected_material.output_chunk_count,
        "selected enc_next_seed ciphertexts must flatten to the selected output chunk count"
    );

    assert!(
        layout.ring_dim <= ring_gsw.num_slots,
        "next-seed encoding slot assignment requires ring_dim {} <= Ring-GSW num_slots {}",
        layout.ring_dim,
        ring_gsw.num_slots
    );
    let (q_moduli, _crt_bits, q_moduli_depth) = ring_gsw.params.to_crt();
    assert_eq!(q_moduli_depth, layout.crt_depth, "Ring-GSW CRT depth must match params.to_crt()");
    assert_eq!(q_moduli.len(), layout.crt_depth, "one CRT modulus is required for each CRT level");
    let selected_output_chunk_count = selected_material.output_chunk_count;

    let goldreich_subcircuit_id = circuit.register_sub_circuit(goldreich_subcircuit.circuit);
    let goldreich_outputs = circuit.call_sub_circuit(
        goldreich_subcircuit_id,
        ciphertext_wires::<P, RingGswCiphertext<P, A>>(&encrypted_seeds),
    );

    let mut next_output_start = selected_output_chunk_count;
    let enc_next_seed_wires = goldreich_outputs[..selected_output_chunk_count]
        .iter()
        .map(|wire| wire.as_single_wire())
        .collect::<Vec<_>>();
    let errors = recover_ciphertexts_from_outputs::<P, RingGswCiphertext<P, A>>(
        &error_templates,
        &goldreich_outputs,
        &mut next_output_start,
    );
    let masks = recover_ciphertexts_from_outputs::<P, RingGswCiphertext<P, A>>(
        &mask_templates,
        &goldreich_outputs,
        &mut next_output_start,
    );
    assert_eq!(next_output_start, goldreich_outputs.len());

    let expected_error_count = selected_material
        .output_selection
        .as_ref()
        .map(|selection| selection.len() * layout.errors_per_seed)
        .unwrap_or_else(|| layout.full_error_count());
    assert_eq!(error_templates.len(), expected_error_count);
    let expected_mask_count = selected_material
        .output_selection
        .as_ref()
        .map(|selection| selection.len() * layout.masks_per_seed)
        .unwrap_or_else(|| layout.full_mask_count());
    assert_eq!(mask_templates.len(), expected_mask_count);

    let circuit_outputs = append_next_seed_encoding_formatting::<P, A, M>(
        &mut circuit,
        ring_gsw,
        &layout,
        &enc_next_seed_wires,
        &errors,
        &masks,
    );
    circuit.output(circuit_outputs);
    circuit
}

#[cfg(test)]
mod circuit_tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext},
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    use super::super::step_fn_prg::{
        GoldreichNoiseRefreshOutputSizes, goldreich_noise_refresh_output_size,
        goldreich_noise_refresh_output_sizes,
    };

    #[test]
    fn goldreich_output_size_matches_formula() {
        assert_eq!(
            goldreich_noise_refresh_output_size(7, 11, 3, 8, 5, 2, 13),
            11 * (7 + 3 * 7 * (5 * 8 + 8 * 13 * 5 * 2))
        );
    }

    #[test]
    fn goldreich_output_sizes_split_cbd_and_uniform_segments() {
        assert_eq!(
            goldreich_noise_refresh_output_sizes(7, 11, 3, 8, 5, 2, 13),
            GoldreichNoiseRefreshOutputSizes {
                next_seed_bits: 11 * 7,
                mask_bits: 11 * 3 * 7 * 8 * 13 * 5 * 2,
                cbd_values: 11 * 3 * 7 * 5 * 8,
                total: 11 * (7 + 3 * 7 * (5 * 8 + 8 * 13 * 5 * 2)),
            }
        );
    }

    #[test]
    fn plaintext_moduli_use_one_based_bit_index() {
        assert_eq!(
            ring_gsw_noise_refresh_plaintext_moduli(3, &BigUint::from(64u64)),
            vec![BigUint::from(32u64), BigUint::from(16u64), BigUint::from(8u64)]
        );
    }

    #[test]
    fn goldreich_step_selection_derives_matching_error_and_mask_ranges() {
        let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
        let ring_dim = 2u32;
        let num_slots = 2usize;
        let active_levels = 1usize;
        let crt_bits = 10usize;
        let base_bits = 10u32;
        let p_moduli_bits = 5usize;
        let params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, base_bits);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &params,
            p_moduli_bits,
            2,
            1u64 << 5,
            false,
            Some(active_levels),
        ));
        let ring_gsw =
            Arc::new(RingGswContext::<DCRTPoly, NestedRnsPoly<DCRTPoly>>::from_arith_context(
                &mut setup_circuit,
                &params,
                num_slots,
                nested_rns,
                Some(active_levels),
                None,
            ));
        let seed_bits = 7usize;
        let input_bits_per_step = 3usize;
        let v_bits = 1usize;
        let selected_step_idx = 2usize;
        let layout =
            NextSeedEncodingLayout::new(ring_gsw.as_ref(), seed_bits, input_bits_per_step, v_bits);
        let selected = layout.selected_material(Some(selected_step_idx));
        let selection = selected.output_selection.expect("step selection must create PRG range");
        assert_eq!(selection, selected_step_idx * seed_bits..(selected_step_idx + 1) * seed_bits);
        assert_eq!(selected.output_chunk_count, seed_bits * layout.ciphertext_wire_count);

        let expected_error_count = selection
            .len()
            .checked_mul(layout.ciphertext_wire_count)
            .and_then(|value| value.checked_mul(layout.log_base_q))
            .and_then(|value| value.checked_mul(layout.ring_dim))
            .unwrap();
        let expected_mask_count = selection
            .len()
            .checked_mul(layout.ciphertext_wire_count)
            .and_then(|value| value.checked_mul(layout.ring_dim))
            .and_then(|value| value.checked_mul(v_bits))
            .and_then(|value| value.checked_mul(layout.log_base_q))
            .and_then(|value| value.checked_mul(layout.crt_depth))
            .unwrap();
        assert_eq!(expected_error_count, selection.len() * layout.errors_per_seed);
        assert_eq!(expected_mask_count, selection.len() * layout.masks_per_seed);
    }

    #[test]
    fn enc_next_seed_chunk_crt_formatter_outputs_one_unit() {
        let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
        let ring_dim = 2u32;
        let num_slots = 2usize;
        let active_levels = 1usize;
        let crt_bits = 10usize;
        let base_bits = 10u32;
        let p_moduli_bits = 5usize;
        let params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, base_bits);
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &params,
            p_moduli_bits,
            2,
            1u64 << 5,
            false,
            Some(active_levels),
        ));
        let ring_gsw = Arc::new(RingGswContext::from_arith_context(
            &mut setup_circuit,
            &params,
            num_slots,
            nested_rns,
            Some(active_levels),
            None,
        ));
        let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
        let full_q = params.modulus();
        let circuit = build_enc_next_seed_chunk_crt_formatter_subcircuit::<
            DCRTPoly,
            NestedRnsPoly<DCRTPoly>,
            DCRTPolyMatrix,
        >(ring_gsw.clone(), 1, BigUint::from(q_moduli[0]), &full_q);

        assert_eq!(circuit.num_output(), 1 + ring_gsw.log_base_q());

        let output_chunk_count = 3usize;
        let formatter_circuit = build_next_seed_encoding_formatter_circuit::<
            DCRTPoly,
            NestedRnsPoly<DCRTPoly>,
            DCRTPolyMatrix,
        >(ring_gsw.clone(), output_chunk_count, 1);
        assert_eq!(
            formatter_circuit.num_output(),
            output_chunk_count * ring_gsw.crt_depth() * (1 + ring_gsw.log_base_q())
        );
    }
}
fn combine_next_seed_encoding_chunks<M, E>(
    params: &<M::P as Poly>::Params,
    enc_next_seed_chunk: &[E],
    errors_plus_masks_chunk: &[E],
    unit_column_target: &M,
) -> Vec<E>
where
    M: PolyMatrix,
    E: Evaluable<P = M::P, Params = <M::P as Poly>::Params>,
{
    // Circuit evaluation yields two groups per output chunk: CRT-scaled encrypted next-seed terms
    // and decoded refresh wires.  The decoded refresh wires are one-column objects, so they are
    // first multiplied by `G^-1(u_1)` and concatenated to match the `log_base_q` next-seed columns.
    let log_base_q = params.modulus_digits();
    let crt_depth = enc_next_seed_chunk.len();
    assert_ne!(crt_depth, 0, "crt_depth must be positive");
    assert_eq!(
        errors_plus_masks_chunk.len(),
        log_base_q * crt_depth,
        "errors_plus_masks chunk length must be log_base_q * crt_depth"
    );

    let crt_terms = enc_next_seed_chunk
        .par_iter()
        .enumerate()
        .map(|(crt_idx, enc_next_seed_term)| {
            let refresh_columns = errors_plus_masks_chunk
                [crt_idx * log_base_q..(crt_idx + 1) * log_base_q]
                .par_iter()
                .map(|refresh_encoding| refresh_encoding.matrix_mul(params, unit_column_target))
                .collect::<Vec<_>>();
            let (first_refresh, rest_refresh) =
                refresh_columns.split_first().expect("log_base_q must be positive");
            let refresh_term = first_refresh.concat_columns(rest_refresh);
            enc_next_seed_term.clone() + &refresh_term
        })
        .collect::<Vec<_>>();

    crt_terms
}

/// Evaluates the next-seed-encoding circuit over an arbitrary `Evaluable` representation.
///
/// The input `enc_seeds` must have exactly `ciphertext_wire_count * seed_bits` entries, one for
/// each flattened encrypted seed ciphertext wire.  The constant-one representation is passed
/// separately as `one`, matching the `PolyCircuit::eval` API.  The `plt_evaluator` and
/// `slot_transfer_evaluator` are also explicit required arguments and are forwarded to
/// `PolyCircuit::eval` as `Some(...)`, so the caller always controls the evaluator set used by
/// circuit evaluation.  The next-seed-encoding circuit also uses the constant-one wire as the
/// encoded Ring-GSW decryption key, so it is not included in the ordinary circuit input vector.
///
/// The final compression keeps CRT levels separate.  For each output chunk and each CRT level,
/// it adds the circuit-produced `q/q_i`-scaled next-seed term to the decoded
/// error-plus-mask term for the same CRT modulus.  The decoded refresh columns all use the
/// same multiplication by `G^-1(u_1)`; the resulting one-column values are concatenated
/// into the `log_base_q` refresh columns before being added to the corresponding next-seed
/// term.
pub fn fhe_prf_evaluable_for_step_fn<P, A, M, E, PE, ST>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    one: &E,
    enc_seeds: &[E],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    parallel_gates: Option<usize>,
) -> Vec<E>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
    E: Evaluable<P = P, Params = <P as Poly>::Params>,
    PE: PltEvaluator<E>,
    ST: SlotTransferEvaluator<E>,
{
    fhe_prf_evaluable_for_step_fn_with_output_step_idx::<P, A, M, E, PE, ST>(
        ring_gsw,
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
        one,
        enc_seeds,
        plt_evaluator,
        slot_transfer_evaluator,
        parallel_gates,
        None,
    )
}

pub fn fhe_prf_evaluable_for_step_fn_with_output_step_idx<P, A, M, E, PE, ST>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    one: &E,
    enc_seeds: &[E],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    parallel_gates: Option<usize>,
    output_step_idx: Option<usize>,
) -> Vec<E>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
    E: Evaluable<P = P, Params = <P as Poly>::Params>,
    PE: PltEvaluator<E>,
    ST: SlotTransferEvaluator<E>,
{
    let secret_size = 1usize;
    let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
    let expected_enc_seeds = ciphertext_wire_count
        .checked_mul(seed_bits)
        .expect("noise-refresh FHE PRF input count overflow");
    assert_eq!(
        enc_seeds.len(),
        expected_enc_seeds,
        "fhe_prf_evaluable_for_step_fn expects one value per encrypted seed ciphertext wire"
    );

    let circuit = build_next_seed_encodings_circuit_with_output_step_idx::<P, A, M>(
        ring_gsw.clone(),
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
        output_step_idx,
    );

    let circuit_outputs = circuit.eval(
        &ring_gsw.params,
        one.clone(),
        enc_seeds.to_vec(),
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
        parallel_gates,
    );

    let expanded_seed_bits = seed_bits
        .checked_mul(input_bits_per_step)
        .expect("seed_bits * input_bits_per_step overflow");
    let full_output_chunk_count = expanded_seed_bits
        .checked_mul(ciphertext_wire_count)
        .expect("next-seed output chunk count overflow");
    let output_chunk_count = if let Some(output_step_idx) = output_step_idx {
        assert!(
            output_step_idx < input_bits_per_step,
            "output step index must be less than input_bits_per_step"
        );
        seed_bits
            .checked_mul(ciphertext_wire_count)
            .expect("selected next-seed step wire count overflow")
    } else {
        full_output_chunk_count
    };
    let log_base_q = ring_gsw.log_base_q();
    let crt_depth = ring_gsw.crt_depth();
    let enc_next_seed_output_len = output_chunk_count
        .checked_mul(crt_depth)
        .expect("CRT-scaled next-seed output count overflow");
    let errors_plus_masks_output_len = output_chunk_count
        .checked_mul(log_base_q)
        .and_then(|value| value.checked_mul(crt_depth))
        .expect("decoded refresh output count overflow");
    assert_eq!(
        circuit_outputs.len(),
        enc_next_seed_output_len + errors_plus_masks_output_len,
        "next-seed-encoding circuit output count must match enc_next_seed plus decoded refresh suffix"
    );
    let (enc_next_seed_outputs, errors_plus_masks_outputs) =
        circuit_outputs.split_at(enc_next_seed_output_len);

    let params = &ring_gsw.params;
    let unit_column_target =
        M::scaled_unit_column_vector(params, secret_size, 0, M::P::const_one(params));
    let (q_moduli, _crt_bits, q_moduli_depth) = params.to_crt();
    assert_eq!(q_moduli_depth, crt_depth, "Ring-GSW CRT depth must match params.to_crt()");
    assert_eq!(q_moduli.len(), crt_depth, "one CRT modulus is required for each CRT level");

    enc_next_seed_outputs
        .par_chunks_exact(crt_depth)
        .zip(errors_plus_masks_outputs.par_chunks_exact(log_base_q * crt_depth))
        .flat_map(|(enc_next_seed_chunk, errors_plus_masks_chunk)| {
            combine_next_seed_encoding_chunks::<M, E>(
                params,
                enc_next_seed_chunk,
                errors_plus_masks_chunk,
                &unit_column_target,
            )
        })
        .collect()
}

#[cfg(test)]
mod fhe_prf_tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use crate::{
        __PAIR, __TestState,
        bgg::{
            poly_encoding::BggPolyEncoding,
            sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
        },
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext},
        lookup::{
            lwe::{LWEBGGPolyEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
            poly::PolyPltEvaluator,
            poly_vec::PolyVecPltEvaluator,
        },
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::dcrt::gpu::{
            GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        },
        sampler::{
            PolyTrapdoorSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::{
            PolyVecSlotTransferEvaluator, bgg_poly_encoding::BggPolyEncodingSTEvaluator,
            bgg_pubkey::BggPublicKeySTEvaluator,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use crate::{
        bgg::public_key::BggPublicKey,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    #[cfg(feature = "gpu")]
    use keccak_asm::Keccak256;
    #[cfg(feature = "gpu")]
    use std::{path::Path, sync::Arc, time::Instant};
    #[cfg(feature = "gpu")]
    use tracing::info;

    #[cfg(feature = "gpu")]
    fn log_relation_step(test_start: Instant, step_start: &mut Instant, label: &str) {
        let now = Instant::now();
        info!(
            target: "noise_refresh::fhe_prf_relation",
            step = label,
            step_elapsed_ms = now.duration_since(*step_start).as_millis(),
            total_elapsed_ms = now.duration_since(test_start).as_millis(),
            "relation test step completed"
        );
        *step_start = now;
    }

    fn constant_matrix(
        params: &DCRTPolyParams,
        rows: usize,
        cols: usize,
        offset: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            params,
            (0..rows)
                .map(|row| {
                    (0..cols)
                        .map(|col| {
                            DCRTPoly::from_usize_to_constant(params, offset + row * cols + col + 1)
                        })
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    fn combine_next_seed_encoding_chunks_outputs_log_base_q_columns() {
        let params = DCRTPolyParams::default();
        let secret_size = 1usize;
        let log_base_q = params.modulus_digits();
        let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
        let gadget_cols = secret_size * log_base_q;
        let enc_next_seed_chunk =
            BggPublicKey::new(constant_matrix(&params, secret_size, gadget_cols, 0), true);
        let errors_plus_masks_chunk = (0..log_base_q * crt_depth)
            .map(|idx| {
                BggPublicKey::new(
                    constant_matrix(&params, secret_size, gadget_cols, 100 + idx * 100),
                    true,
                )
            })
            .collect::<Vec<_>>();
        let unit_column_target = DCRTPolyMatrix::scaled_unit_column_vector(
            &params,
            secret_size,
            0,
            DCRTPoly::const_one(&params),
        );
        let full_q: Arc<num_bigint::BigUint> = params.modulus().into();
        let enc_next_seed_terms = q_moduli
            .par_iter()
            .map(|&q_i| {
                let q_over_qi = full_q.as_ref() / num_bigint::BigUint::from(q_i);
                enc_next_seed_chunk.large_scalar_mul(&params, &[q_over_qi])
            })
            .collect::<Vec<_>>();

        let combined = combine_next_seed_encoding_chunks::<DCRTPolyMatrix, BggPublicKey<_>>(
            &params,
            enc_next_seed_terms.as_slice(),
            errors_plus_masks_chunk.as_slice(),
            &unit_column_target,
        );

        let expected_terms = enc_next_seed_terms
            .par_iter()
            .enumerate()
            .map(|(crt_idx, enc_next_seed_term)| {
                let refresh_columns = errors_plus_masks_chunk
                    [crt_idx * log_base_q..(crt_idx + 1) * log_base_q]
                    .par_iter()
                    .map(|refresh_encoding| {
                        refresh_encoding.matrix_mul(&params, &unit_column_target)
                    })
                    .collect::<Vec<_>>();
                let (first_refresh, rest_refresh) =
                    refresh_columns.split_first().expect("log_base_q must be positive");
                let refresh_term = first_refresh.concat_columns(rest_refresh);
                enc_next_seed_term.clone() + &refresh_term
            })
            .collect::<Vec<_>>();
        assert_eq!(combined.len(), crt_depth);
        assert!(combined.iter().all(|term| term.matrix.col_size() == log_base_q));
        assert_eq!(combined, expected_terms);
    }

    #[cfg(feature = "gpu")]
    type TestArithmetic = NestedRnsPoly<GpuDCRTPoly>;
    #[cfg(feature = "gpu")]
    type TestRingGswContext =
        crate::gadgets::fhe::ring_gsw_nested_rns::NestedRnsRingGswContext<GpuDCRTPoly>;

    #[cfg(feature = "gpu")]
    fn create_test_context(
        circuit: &mut PolyCircuit<GpuDCRTPoly>,
        gpu_ids: Vec<i32>,
    ) -> (GpuDCRTPolyParams, Arc<TestRingGswContext>) {
        let ring_dim = 2u32;
        let num_slots = 2usize;
        let active_levels = 1usize;
        let crt_bits = 10usize;
        let base_bits = 10u32;
        let p_moduli_bits = 5usize;
        let scale = 1u64 << 5;
        let cpu_params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, base_bits);
        let (moduli, _crt_bits, _crt_depth) = cpu_params.to_crt();
        let params = GpuDCRTPolyParams::new_with_gpu(
            ring_dim,
            moduli,
            base_bits,
            gpu_ids,
            Some(num_slots as u32),
        );
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            p_moduli_bits,
            2,
            scale,
            false,
            Some(active_levels),
        ));
        let ring_gsw = Arc::new(RingGswContext::from_arith_context(
            circuit,
            &params,
            num_slots,
            nested_rns,
            Some(active_levels),
            None,
        ));
        (params, ring_gsw)
    }

    #[cfg(feature = "gpu")]
    fn constant_plaintext_row<P: Poly>(
        params: &P::Params,
        values: impl IntoIterator<Item = usize>,
    ) -> Vec<Arc<[u8]>> {
        values
            .into_iter()
            .map(|value| {
                Arc::<[u8]>::from(P::from_usize_to_constant(params, value).to_compact_bytes())
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn round_qi_scaled_poly<P: Poly>(params: &P::Params, value: &P, q_i: u64) -> P {
        let q: Arc<num_bigint::BigUint> = params.modulus().into();
        let q_i = num_bigint::BigUint::from(q_i);
        let half_q = q.as_ref() / num_bigint::BigUint::from(2u32);
        let rounded_coeffs = value
            .coeffs_biguints()
            .into_iter()
            .map(|coeff| ((coeff * &q_i) + &half_q) / q.as_ref())
            .collect::<Vec<_>>();
        P::from_biguints(params, &rounded_coeffs)
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    #[sequential_test::sequential]
    #[ignore = "expensive Nested RNS BGG relation test; run explicitly when needed"]
    async fn fhe_prf_evaluable_for_step_fn_bgg_poly_encoding_matches_public_key_relation() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
        let test_start = Instant::now();
        let mut step_start = test_start;
        info!(target: "noise_refresh::fhe_prf_relation", "relation test started");
        let _storage_lock = storage_test_lock().await;
        log_relation_step(test_start, &mut step_start, "acquired storage lock");

        gpu_device_sync();
        let gpu_ids = detected_gpu_device_ids();
        if gpu_ids.is_empty() {
            info!(
                target: "noise_refresh::fhe_prf_relation",
                "skipping GpuDCRTPolyMatrix relation test because no GPU devices were detected"
            );
            return;
        }
        info!(
            target: "noise_refresh::fhe_prf_relation",
            gpu_device_count = gpu_ids.len(),
            ?gpu_ids,
            "detected GPU devices"
        );
        let mut setup_circuit = PolyCircuit::new();
        let (params, ring_gsw) = create_test_context(&mut setup_circuit, gpu_ids);
        let seed_bits = 13usize;
        let input_bits_per_step = 1usize;
        let v_bits = 1usize;
        let cbd_n = 2usize;
        let output_step_idx = 0usize;
        let graph_seed = [0x5au8; 32];
        let hash_key = [0x91u8; 32];
        let secret_size = 1usize;
        let num_slots = ring_gsw.ring_dim();
        let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
        let enc_seed_count = seed_bits * ciphertext_wire_count;
        let reveal_plaintexts = vec![true; enc_seed_count];
        info!(
            target: "noise_refresh::fhe_prf_relation",
            seed_bits,
            input_bits_per_step,
            v_bits,
            cbd_n,
            output_step_idx,
            ring_dim = num_slots,
            ciphertext_wire_count,
            enc_seed_count,
            log_base_q = params.modulus_digits(),
            crt_depth = ring_gsw.crt_depth(),
            "relation test parameters"
        );
        log_relation_step(test_start, &mut step_start, "created Nested RNS Ring-GSW context");

        let dir_path = "test_data/noise_refresh_fhe_prf_poly_encoding_relation";
        let dir = Path::new(dir_path);
        if dir.exists() {
            std::fs::remove_dir_all(dir).unwrap();
        }
        std::fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
        log_relation_step(test_start, &mut step_start, "initialized storage");

        let pubkey_sampler =
            BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(hash_key, secret_size);
        let public_keys =
            pubkey_sampler.sample(&params, b"noise_refresh_fhe_prf", &reveal_plaintexts);
        log_relation_step(test_start, &mut step_start, "sampled BGG public keys");
        let pubkey_st_evaluator =
            BggPublicKeySTEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyUniformSampler,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(hash_key, secret_size, num_slots, 4.578, 0.0, dir.to_path_buf());
        let lwe_trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, 4.578);
        let (lwe_trapdoor, lwe_pub_matrix) = lwe_trapdoor_sampler.trapdoor(&params, secret_size);
        let pubkey_plt_evaluator = LWEBGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(
            hash_key,
            lwe_trapdoor_sampler,
            Arc::new(lwe_pub_matrix.clone()),
            Arc::new(lwe_trapdoor),
            dir.to_path_buf(),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            "evaluating output-limited circuit over BGGPublicKey"
        );
        let pubkey_outputs = fhe_prf_evaluable_for_step_fn_with_output_step_idx::<
            GpuDCRTPoly,
            TestArithmetic,
            GpuDCRTPolyMatrix,
            BggPublicKey<GpuDCRTPolyMatrix>,
            _,
            _,
        >(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            &public_keys[0],
            &public_keys[1..],
            &pubkey_plt_evaluator,
            &pubkey_st_evaluator,
            Some(1),
            Some(output_step_idx),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_count = pubkey_outputs.len(),
            first_col_size = pubkey_outputs.first().map(|pk| pk.matrix.col_size()).unwrap_or(0),
            "BGGPublicKey outputs"
        );
        log_relation_step(test_start, &mut step_start, "evaluated BGGPublicKey outputs");

        info!(
            target: "noise_refresh::fhe_prf_relation",
            "sampling PublicLUT and slot-transfer auxiliary matrices"
        );
        pubkey_plt_evaluator.sample_aux_matrices(&params);
        pubkey_st_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        log_relation_step(
            test_start,
            &mut step_start,
            "sampled and flushed PublicLUT and slot-transfer aux matrices",
        );
        let slot_secret_mats = pubkey_st_evaluator
            .load_slot_secret_mats_checkpoint(&params)
            .expect("slot secret matrix checkpoints should exist after sample_aux_matrices");
        log_relation_step(test_start, &mut step_start, "loaded slot secret matrices");

        let plaintext_rows = (0..enc_seed_count)
            .map(|input_idx| {
                constant_plaintext_row::<GpuDCRTPoly>(
                    &params,
                    (0..num_slots).map(|slot| (input_idx + slot) % 2),
                )
            })
            .collect::<Vec<_>>();
        log_relation_step(test_start, &mut step_start, "built plaintext rows");
        let secrets = vec![GpuDCRTPoly::const_one(&params)];
        let poly_encoding_sampler =
            BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
        let input_encodings = poly_encoding_sampler.sample(
            &params,
            &public_keys,
            &plaintext_rows,
            Some(&slot_secret_mats),
        );
        log_relation_step(test_start, &mut step_start, "sampled zero-error BGGPolyEncoding inputs");
        let b0_matrix = pubkey_st_evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
        let base_secret_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
        let c_b0 = base_secret_vec.clone() * &b0_matrix;
        let c_lwe_pub_matrix_compact_bytes_by_slot = LWEBGGPolyEncodingPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
        >::build_c_b_compact_bytes_by_slot::<
            GpuDCRTPolyUniformSampler,
        >(
            &params,
            &base_secret_vec,
            &lwe_pub_matrix,
            &slot_secret_mats,
            None,
        );
        let poly_encoding_plt_evaluator = LWEBGGPolyEncodingPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
        >::new(
            hash_key,
            dir.to_path_buf(),
            c_lwe_pub_matrix_compact_bytes_by_slot,
        );
        let poly_encoding_st_evaluator = BggPolyEncodingSTEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
        >::new(
            hash_key,
            dir.to_path_buf(),
            pubkey_st_evaluator.checkpoint_prefix(&params),
            c_b0.to_compact_bytes(),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            "evaluating output-limited circuit over BGGPolyEncoding"
        );
        let poly_encoding_outputs = fhe_prf_evaluable_for_step_fn_with_output_step_idx::<
            GpuDCRTPoly,
            TestArithmetic,
            GpuDCRTPolyMatrix,
            BggPolyEncoding<GpuDCRTPolyMatrix>,
            _,
            _,
        >(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            &input_encodings[0],
            &input_encodings[1..],
            &poly_encoding_plt_evaluator,
            &poly_encoding_st_evaluator,
            Some(1),
            Some(output_step_idx),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_count = poly_encoding_outputs.len(),
            first_col_size = poly_encoding_outputs
                .first()
                .map(|enc| enc.pubkey.matrix.col_size())
                .unwrap_or(0),
            "BGGPolyEncoding outputs"
        );
        log_relation_step(test_start, &mut step_start, "evaluated BGGPolyEncoding outputs");
        assert_eq!(poly_encoding_outputs.len(), pubkey_outputs.len());

        info!(
            target: "noise_refresh::fhe_prf_relation",
            "building output-limited PolyVec reference circuit"
        );
        let reference_circuit = build_next_seed_encodings_circuit_with_output_step_idx::<
            GpuDCRTPoly,
            TestArithmetic,
            GpuDCRTPolyMatrix,
        >(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            Some(output_step_idx),
        );
        log_relation_step(test_start, &mut step_start, "built PolyVec reference circuit");
        let polyvec_inputs = plaintext_rows
            .iter()
            .map(|row| {
                PolyVec::new(
                    row.iter()
                        .map(|bytes| GpuDCRTPoly::from_compact_bytes(&params, bytes.as_ref()))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let polyvec_one = PolyVec::new(vec![GpuDCRTPoly::const_one(&params); num_slots]);
        let polyvec_plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        info!(target: "noise_refresh::fhe_prf_relation", "evaluating PolyVec reference outputs");
        let reference_outputs = reference_circuit.eval(
            &params,
            polyvec_one,
            polyvec_inputs,
            Some(&polyvec_plt_evaluator),
            Some(&PolyVecSlotTransferEvaluator::new()),
            Some(1),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_count = reference_outputs.len(),
            "PolyVec reference outputs"
        );
        log_relation_step(test_start, &mut step_start, "evaluated PolyVec reference outputs");

        let output_chunk_count = seed_bits * ciphertext_wire_count;
        let log_base_q = params.modulus_digits();
        let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
        let enc_next_seed_output_len = output_chunk_count * crt_depth;
        let (reference_enc_next_seed, reference_refresh) =
            reference_outputs.split_at(enc_next_seed_output_len);
        let gadget_row = GpuDCRTPolyMatrix::gadget_matrix(&params, secret_size).get_row(0);
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_chunk_count,
            log_base_q,
            crt_depth,
            num_slots,
            "checking public-key/poly-encoding relation"
        );

        for output_idx in 0..output_chunk_count {
            for crt_idx in 0..crt_depth {
                let flat_idx = output_idx * crt_depth + crt_idx;
                let pubkey_output = &pubkey_outputs[flat_idx];
                let encoding_output = &poly_encoding_outputs[flat_idx];
                assert_eq!(encoding_output.pubkey, *pubkey_output);
                for slot_idx in 0..num_slots {
                    let slot_secret = GpuDCRTPolyMatrix::from_compact_bytes(
                        &params,
                        slot_secret_mats[slot_idx].as_ref(),
                    );
                    let lhs = slot_secret.clone() * pubkey_output.matrix.clone() -
                        encoding_output.vector(slot_idx);
                    let enc_next_seed_plain =
                        reference_enc_next_seed[flat_idx].as_slice()[slot_idx].clone();
                    let expected_cols = (0..log_base_q)
                        .map(|digit_idx| {
                            let refresh_idx = output_idx * log_base_q * crt_depth +
                                crt_idx * log_base_q +
                                digit_idx;
                            let refresh_plain =
                                reference_refresh[refresh_idx].as_slice()[slot_idx].clone();
                            let column_plain = (enc_next_seed_plain.clone() *
                                &gadget_row[digit_idx]) +
                                refresh_plain;
                            slot_secret.entry(0, 0) * column_plain
                        })
                        .collect::<Vec<_>>();
                    let expected = GpuDCRTPolyMatrix::from_poly_vec_row(&params, expected_cols);
                    assert_eq!(
                        lhs, expected,
                        "output {output_idx}, crt {crt_idx}, slot {slot_idx}"
                    );

                    let q_i = q_moduli[crt_idx];
                    let rounded_lhs_cols = lhs
                        .get_row(0)
                        .into_iter()
                        .map(|poly| round_qi_scaled_poly(&params, &poly, q_i))
                        .collect::<Vec<_>>();
                    let rounded_expected_cols = (0..log_base_q)
                        .map(|digit_idx| {
                            let refresh_idx = output_idx * log_base_q * crt_depth +
                                crt_idx * log_base_q +
                                digit_idx;
                            let enc_next_seed_plain = round_qi_scaled_poly(
                                &params,
                                &reference_enc_next_seed[flat_idx].as_slice()[slot_idx],
                                q_i,
                            );
                            let error_plain = round_qi_scaled_poly(
                                &params,
                                &reference_refresh[refresh_idx].as_slice()[slot_idx],
                                q_i,
                            );
                            let column_plain =
                                (enc_next_seed_plain * &gadget_row[digit_idx]) + error_plain;
                            slot_secret.entry(0, 0) * column_plain
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        GpuDCRTPolyMatrix::from_poly_vec_row(&params, rounded_lhs_cols),
                        GpuDCRTPolyMatrix::from_poly_vec_row(&params, rounded_expected_cols),
                        "rounded output {output_idx}, crt {crt_idx}, slot {slot_idx}"
                    );
                }
            }
        }
        log_relation_step(test_start, &mut step_start, "checked public-key/poly-encoding relation");
    }
}
*/
