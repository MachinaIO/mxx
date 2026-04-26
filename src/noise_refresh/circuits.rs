//! Circuit builders used by the noise-refresh protocol.
//!
//! The functions in this module construct `PolyCircuit`s only.  Public parameters such as the
//! Goldreich graph seed, the Ring-GSW context, the seed length, and the mask bit-size are fixed at
//! construction time.  Secret values, namely encrypted seed bits and the decryption key wire, are
//! represented as circuit inputs inside the returned circuit.

use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticContext, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::{
            BooleanCiphertext, GoldreichFheCbdPrg, GoldreichFhePrg, GoldreichGraph,
        },
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use digest::Digest;
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::{marker::PhantomData, sync::Arc};

pub trait NoiseRefreshPrgParameters {
    /// Ring dimension `n` used by Ring-GSW ciphertexts.
    fn ring_dim(&self) -> usize;

    /// Number of gadget/base digits `m_g = log_base(q)` used in the Ring-GSW decomposition.
    fn log_base_q(&self) -> usize;

    /// CRT depth `d`, i.e. the number of CRT moduli used by the polynomial modulus.
    fn crt_depth(&self) -> usize;

    /// Number of scalar circuit wires needed to represent one encrypted Boolean ciphertext.
    ///
    /// The noise-refresh PRG expands one logical seed bit into enough mask/error material for every
    /// wire of that encrypted bit.  This count is therefore a property of the Boolean ciphertext
    /// representation, not a property of the seed itself.
    fn encrypted_bit_wire_count(&self) -> usize;
}

impl<P, A> NoiseRefreshPrgParameters for RingGswContext<P, A>
where
    P: Poly,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    fn ring_dim(&self) -> usize {
        self.params.ring_dimension() as usize
    }

    fn log_base_q(&self) -> usize {
        self.params.modulus_digits()
    }

    fn crt_depth(&self) -> usize {
        let (_moduli, _crt_bits, crt_depth) = self.params.to_crt();
        crt_depth
    }

    fn encrypted_bit_wire_count(&self) -> usize {
        2usize
            .checked_mul(self.width())
            .and_then(|value| value.checked_mul(self.active_levels))
            .and_then(|value| value.checked_mul(self.arith_ctx.q_level_row_width()))
            .expect("encrypted Boolean ciphertext wire count overflow")
    }
}

struct GoldreichNoiseRefreshOutput<P: Poly, C: BooleanCiphertext<P>> {
    // The first `seed_bits` uniform PRG outputs.  They are kept only while composing the
    // next-seed-encoding circuit, where they become the first outputs of the outer circuit.
    next_seed: Vec<C>,
    // Uniform PRG outputs after `next_seed`.  These are later chunked as
    // `seed_bits * ciphertext_wire_count * log_base_q` groups, each containing one CRT-depth
    // sequence of `ring_dim * v_bits` encrypted bits.
    masks: Vec<C>,
    // CBD PRF outputs.  Each output is an encrypted integer coefficient, not an encrypted bit.
    // The layout is grouped by seed bit, then ciphertext wire, then gadget digit, then ring
    // coefficient.
    errors: Vec<C>,
    _poly: PhantomData<P>,
}

struct GoldreichNoiseRefreshCircuit<P: Poly, C: BooleanCiphertext<P>> {
    circuit: PolyCircuit<P>,
    output: GoldreichNoiseRefreshOutput<P, C>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldreichNoiseRefreshOutputSizes {
    /// Number of encrypted Boolean ciphertexts used for the next seed.
    pub next_seed_bits: usize,
    /// Number of encrypted Boolean ciphertexts used as mask bits.
    pub mask_bits: usize,
    /// Number of encrypted CBD integer ciphertexts used as error coefficients.
    pub cbd_values: usize,
    /// Total number of logical ciphertext outputs before flattening each ciphertext into wires.
    pub total: usize,
}

/// Computes the logical output sizes of the Goldreich-based noise-refresh expansion.
///
/// The seed contains `seed_bits = s` encrypted Boolean ciphertexts.  Each encrypted Boolean
/// ciphertext is internally represented by `ciphertext_wire_count` scalar circuit wires, so the
/// protocol expands material for `s * ciphertext_wire_count` wire positions.
/// `input_bits_per_step` repeats the existing output layout that many times; it does not add more
/// seed inputs, it only asks the PRG to produce more refresh material per seed expansion step.
///
/// Output layout:
/// - `next_seed_bits = s * input_bits_per_step`
/// - `cbd_values = s * input_bits_per_step * ciphertext_wire_count * ring_dim * log_base_q *
///   ring_dim`
/// - `mask_bits = s * input_bits_per_step * ciphertext_wire_count * ring_dim * v_bits * log_base_q
///   * crt_depth * ring_dim`
///
/// The returned sizes count logical ciphertext objects, not flattened `GateId`s.  Flattening is
/// delegated to `BooleanCiphertext::sub_circuit_wires`, because the exact wire representation is
/// ciphertext-implementation dependent.
pub fn goldreich_noise_refresh_output_sizes(
    seed_bits: usize,
    input_bits_per_step: usize,
    ciphertext_wire_count: usize,
    ring_dim: usize,
    log_base_q: usize,
    crt_depth: usize,
    v_bits: usize,
) -> GoldreichNoiseRefreshOutputSizes {
    assert!(seed_bits > 0, "seed bit length must be positive");
    assert!(input_bits_per_step > 0, "input_bits_per_step must be positive");
    assert!(ciphertext_wire_count > 0, "ciphertext_wire_count must be positive");
    assert!(ring_dim > 0, "ring_dim must be positive");
    assert!(log_base_q > 0, "log_base_q must be positive");
    assert!(crt_depth > 0, "crt_depth must be positive");
    let expanded_seed_bits = seed_bits
        .checked_mul(input_bits_per_step)
        .expect("Goldreich noise-refresh seed_bits * input_bits_per_step overflow");
    let encrypted_seed_wire_count = ciphertext_wire_count
        .checked_mul(expanded_seed_bits)
        .expect("Goldreich noise-refresh ciphertext_wire_count * expanded_seed_bits overflow");
    let next_seed_bits = expanded_seed_bits;
    let cbd_values = encrypted_seed_wire_count
        .checked_mul(ring_dim)
        .and_then(|value| value.checked_mul(log_base_q))
        .and_then(|value| value.checked_mul(ring_dim))
        .expect("Goldreich noise-refresh CBD output size overflow");
    let mask_bits = encrypted_seed_wire_count
        .checked_mul(ring_dim)
        .and_then(|value| value.checked_mul(v_bits))
        .and_then(|value| value.checked_mul(log_base_q))
        .and_then(|value| value.checked_mul(crt_depth))
        .and_then(|value| value.checked_mul(ring_dim))
        .expect("Goldreich noise-refresh uniform output size overflow");
    let uniform_bits = next_seed_bits
        .checked_add(mask_bits)
        .expect("Goldreich noise-refresh uniform output size overflow");
    let total = uniform_bits
        .checked_add(cbd_values)
        .expect("Goldreich noise-refresh total output size overflow");
    GoldreichNoiseRefreshOutputSizes { next_seed_bits, mask_bits, cbd_values, total }
}

pub fn goldreich_noise_refresh_output_size(
    seed_bits: usize,
    input_bits_per_step: usize,
    ciphertext_wire_count: usize,
    ring_dim: usize,
    log_base_q: usize,
    crt_depth: usize,
    v_bits: usize,
) -> usize {
    goldreich_noise_refresh_output_sizes(
        seed_bits,
        input_bits_per_step,
        ciphertext_wire_count,
        ring_dim,
        log_base_q,
        crt_depth,
        v_bits,
    )
    .total
}

/// Builds the Goldreich encrypted-seed expansion circuit.
///
/// The returned circuit creates `seed_bits` encrypted Boolean ciphertext inputs internally.  It
/// then evaluates two related but graph-separated PRG components on those inputs:
///
/// 1. A uniform Goldreich PRG with output size `seed_bits * input_bits_per_step + mask_bits`. The
///    first `seed_bits * input_bits_per_step` outputs are the encrypted next seed material; the
///    rest are mask bits.
/// 2. A CBD PRF with output size `cbd_values`. These outputs are encrypted integer error
///    coefficients rather than encrypted bits.
///
/// The circuit output order is:
/// `next_seed`, then `errors`, then `masks`, with each logical ciphertext flattened into its
/// `sub_circuit_wires()` representation.
pub fn evaluate_goldreich_encrypted_seeds<P, C>(
    ring_gsw: Arc<C::Context>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
    C::Context: NoiseRefreshPrgParameters,
{
    build_goldreich_encrypted_seeds_with_output::<P, C>(
        ring_gsw,
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
    )
    .circuit
}

fn build_goldreich_encrypted_seeds_with_output<P, C>(
    ring_gsw: Arc<C::Context>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
) -> GoldreichNoiseRefreshCircuit<P, C>
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
    C::Context: NoiseRefreshPrgParameters,
{
    // This is a standalone subcircuit: all encrypted seed ciphertexts are declared as inputs of the
    // newly-created circuit rather than being passed in from an outer builder.
    let mut circuit = PolyCircuit::new();
    let input_size = seed_bits;
    assert!(input_size >= 5, "Goldreich PRG requires at least five encrypted seed bits");

    // Create one encrypted Boolean ciphertext input for each seed bit.  `C` abstracts over the
    // concrete FHE ciphertext representation, so this works for Ring-GSW ciphertexts and any future
    // Boolean ciphertext type that implements the same interface.
    let encrypted_seeds = (0..seed_bits)
        .map(|_| C::sub_circuit_input(ring_gsw.clone(), &mut circuit))
        .collect::<Vec<_>>();

    // Compute output sizes from the ciphertext context.  The important point is that
    // `ciphertext_wire_count` is not an extra protocol parameter: it is derived from the Boolean
    // ciphertext representation used by this context.
    let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
    let output_sizes = goldreich_noise_refresh_output_sizes(
        seed_bits,
        input_bits_per_step,
        ciphertext_wire_count,
        ring_gsw.ring_dim(),
        ring_gsw.log_base_q(),
        ring_gsw.crt_depth(),
        v_bits,
    );

    // The uniform PRG produces both next-seed bits and mask bits.  The CBD values are intentionally
    // excluded from this graph so that they can be generated with a separate graph family.
    let uniform_prg = GoldreichFhePrg::setup(
        &mut circuit,
        ring_gsw.clone(),
        input_size,
        output_sizes.next_seed_bits + output_sizes.mask_bits,
        graph_seed,
    );
    let uniform_outputs = uniform_prg.evaluate_uniform(&encrypted_seeds, &mut circuit);

    // The CBD PRF must not reuse the same public Goldreich graph as the uniform PRG for the same
    // input seed.  `setup_cbd_prf_distinct_from_uniform` domain-separates the graph seed and checks
    // graph structure collisions before returning.
    let cbd_prf = setup_cbd_prf_distinct_from_uniform(
        &mut circuit,
        ring_gsw,
        input_size,
        output_sizes.cbd_values,
        graph_seed,
        cbd_n,
        &uniform_prg,
    );
    let errors = cbd_prf.evaluate_cbd_prf(&encrypted_seeds, &mut circuit);

    // The uniform segment is split by protocol meaning: the first `s * input_bits_per_step`
    // ciphertexts become the next seed material, and all remaining uniform ciphertexts are masks
    // consumed by the next-seed-encoding circuit.
    let (next_seed, masks) = uniform_outputs.split_at(output_sizes.next_seed_bits);
    let next_seed = next_seed.to_vec();
    let masks = masks.to_vec();
    debug_assert_eq!(next_seed.len() + errors.len() + masks.len(), output_sizes.total);

    // Outer circuits see only flattened wires.  The logical grouping is preserved privately through
    // `GoldreichNoiseRefreshOutput`, but the actual `PolyCircuit` outputs are scalar/batched wires
    // in the exact order below.
    circuit.output(
        next_seed
            .iter()
            .chain(errors.iter())
            .chain(masks.iter())
            .flat_map(|output| output.sub_circuit_wires()),
    );
    let _ = (uniform_prg, cbd_prf);
    let output = GoldreichNoiseRefreshOutput { next_seed, masks, errors, _poly: PhantomData };
    GoldreichNoiseRefreshCircuit { circuit, output }
}

fn setup_cbd_prf_distinct_from_uniform<P, C>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<C::Context>,
    input_size: usize,
    output_size: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    uniform_prg: &GoldreichFhePrg<P, C>,
) -> GoldreichFheCbdPrg<P, C>
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
{
    assert!(output_size > 0, "Goldreich CBD output size must be positive");
    for counter in 0u64.. {
        // The counter gives a deterministic retry path if the derived CBD graph happens to collide
        // with the uniform graph.  In normal parameter ranges this should be rare, but the explicit
        // retry makes the non-overlap invariant part of setup rather than an assumption.
        let cbd_seed = derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshCBD/v1", counter);
        let cbd_prf = GoldreichFheCbdPrg::setup(
            circuit,
            ring_gsw.clone(),
            input_size,
            output_size,
            cbd_seed,
            cbd_n,
        );
        if cbd_prf
            .uniform_graphs()
            .iter()
            .all(|graph| !same_goldreich_graph_structure(graph, uniform_prg.graph()))
        {
            return cbd_prf;
        }
    }
    unreachable!("u64 counter space must contain a CBD graph distinct from the uniform graph")
}

fn derive_noise_refresh_graph_seed(base_seed: [u8; 32], label: &[u8], counter: u64) -> [u8; 32] {
    // Domain separation keeps the CBD graph stream independent from the uniform graph stream while
    // still making the construction deterministic from the public `graph_seed`.
    let mut hasher = Keccak256::new();
    hasher.update(label);
    hasher.update(base_seed);
    hasher.update(counter.to_le_bytes());
    let digest = hasher.finalize();
    let mut derived = [0u8; 32];
    derived.copy_from_slice(digest.as_ref());
    derived
}

fn same_goldreich_graph_structure(lhs: &GoldreichGraph, rhs: &GoldreichGraph) -> bool {
    lhs.input_size == rhs.input_size && lhs.edges == rhs.edges && lhs.generation == rhs.generation
}

fn ciphertext_wires<P, C>(ciphertexts: &[C]) -> Vec<BatchedWire>
where
    P: Poly,
    C: BooleanCiphertext<P>,
{
    // Subcircuit calls operate on flattened wire lists.  This helper erases the logical ciphertext
    // grouping when feeding a child circuit.
    ciphertexts.iter().flat_map(|ciphertext| ciphertext.sub_circuit_wires()).collect()
}

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

fn monomial_scalar(ring_dim: usize, coeff_idx: usize) -> Vec<u32> {
    assert!(coeff_idx < ring_dim, "monomial coefficient index must be in range");
    // Multiplication by this sparse scalar maps a constant decrypted coefficient into the
    // `X^coeff_idx` monomial slot, so summing these terms reconstructs a polynomial from its
    // coefficient-wise decryptions.
    let mut scalar = vec![0u32; ring_dim];
    scalar[coeff_idx] = 1;
    scalar
}

fn sum_gate_ids<P: Poly>(circuit: &mut PolyCircuit<P>, values: &[GateId]) -> GateId {
    let (first, rest) = values.split_first().expect("at least one gate is required");
    rest.iter().fold(*first, |acc, value| circuit.add_gate(acc, *value).as_single_wire())
}

fn decrypt_ring_gsw_coeff_slot_batch_sum<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_coeff_chunks_by_slot: &[&[RingGswCiphertext<P, A>]],
    decryption_key: GateId,
    plaintext_modulus: BigUint,
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
    // This helper handles one logical error chunk across all output slots.  Each output slot owns
    // `ring_dim` encrypted coefficient ciphertexts.  For a fixed coefficient index, `decrypt_batch`
    // decrypts the ciphertext from every output slot at once and returns one slotwise wire whose
    // slot `i` contains the coefficient for output slot `i`.  Multiplying by `X^k` and summing over
    // coefficient indices reconstructs all `ring_dim` output polynomials inside one wire.
    let first_ctx: Arc<RingGswContext<P, A>> = encrypted_coeff_chunks_by_slot
        .first()
        .and_then(|chunk| chunk.first())
        .expect("at least one encrypted coefficient is required")
        .ctx
        .clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    assert_eq!(
        encrypted_coeff_chunks_by_slot.len(),
        ring_dim,
        "encrypted coefficient slot batch length {} must match ring_dim {}",
        encrypted_coeff_chunks_by_slot.len(),
        ring_dim
    );
    for (slot_idx, encrypted_coeffs) in encrypted_coeff_chunks_by_slot.iter().enumerate() {
        assert_eq!(
            encrypted_coeffs.len(),
            ring_dim,
            "encrypted coefficient chunk for slot {slot_idx} must have ring_dim entries"
        );
        for (coeff_idx, ciphertext) in encrypted_coeffs.iter().enumerate() {
            assert!(
                Arc::ptr_eq(&ciphertext.ctx, &first_ctx),
                "encrypted coefficient ({slot_idx}, {coeff_idx}) must share the first Ring-GSW context"
            );
        }
    }

    let terms = (0..ring_dim)
        .map(|coeff_idx| {
            let ciphertexts = encrypted_coeff_chunks_by_slot
                .iter()
                .map(|chunk| &chunk[coeff_idx])
                .collect::<Vec<_>>();
            let decrypted_coeff_batch = RingGswCiphertext::decrypt_batch::<M>(
                ciphertexts.as_slice(),
                decryption_key,
                plaintext_modulus.clone(),
                circuit,
            );
            let scalar = monomial_scalar(ring_dim, coeff_idx);
            circuit.small_scalar_mul(decrypted_coeff_batch, &scalar).as_single_wire()
        })
        .collect::<Vec<_>>();
    sum_gate_ids(circuit, &terms)
}

fn decrypt_ring_gsw_bit_chunk_sum<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    bit_size: usize,
    scale: &BigUint,
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    decrypt_ring_gsw_bit_slot_batch_sum::<P, A, M>(
        circuit,
        &[encrypted_bits],
        decryption_key,
        bit_size,
        scale,
    )
}

fn decrypt_ring_gsw_bit_slot_batch_sum<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bit_chunks_by_slot: &[&[RingGswCiphertext<P, A>]],
    decryption_key: GateId,
    bit_size: usize,
    scale: &BigUint,
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let plaintext_moduli = ring_gsw_noise_refresh_plaintext_moduli(bit_size, scale);
    let first_ctx: Arc<RingGswContext<P, A>> = encrypted_bit_chunks_by_slot
        .first()
        .and_then(|chunk| chunk.first())
        .expect("at least one encrypted bit is required")
        .ctx
        .clone();
    let ring_dim = first_ctx.params.ring_dimension() as usize;
    let expected_len = ring_dim.checked_mul(bit_size).expect("Ring-GSW bit chunk length overflow");
    assert!(
        encrypted_bit_chunks_by_slot.len() <= ring_dim,
        "encrypted bit slot batch length {} must be at most ring_dim {}",
        encrypted_bit_chunks_by_slot.len(),
        ring_dim
    );
    for (slot_idx, encrypted_bits) in encrypted_bit_chunks_by_slot.iter().enumerate() {
        assert_eq!(
            encrypted_bits.len(),
            expected_len,
            "encrypted bit chunk for slot {slot_idx} must equal ring_dim * bit_size"
        );
        for (idx, ciphertext) in encrypted_bits.iter().enumerate() {
            assert!(
                Arc::ptr_eq(&ciphertext.ctx, &first_ctx),
                "encrypted bit ciphertext ({slot_idx}, {idx}) must share the first Ring-GSW context"
            );
        }
    }

    let coeff_terms = (0..ring_dim)
        .map(|coeff_idx| {
            // For a fixed coefficient, decrypt every bit under its bit-dependent plaintext modulus
            // `scale / 2^j`, sum those decrypted bit contributions, and then move the coefficient
            // aggregate into the polynomial slot for this coefficient.
            let bit_terms = (0..bit_size)
                .map(|bit_idx| {
                    let ciphertexts = encrypted_bit_chunks_by_slot
                        .iter()
                        .map(|chunk| &chunk[coeff_idx * bit_size + bit_idx])
                        .collect::<Vec<_>>();
                    RingGswCiphertext::decrypt_batch::<M>(
                        ciphertexts.as_slice(),
                        decryption_key,
                        plaintext_moduli[bit_idx].clone(),
                        circuit,
                    )
                })
                .collect::<Vec<_>>();
            let coeff_sum = sum_gate_ids(circuit, &bit_terms);
            let scalar = monomial_scalar(ring_dim, coeff_idx);
            circuit.small_scalar_mul(coeff_sum, &scalar).as_single_wire()
        })
        .collect::<Vec<_>>();
    sum_gate_ids(circuit, &coeff_terms)
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
/// For every coefficient `k`, the circuit decrypts all bits, sums the decrypted bit values,
/// multiplies the sum by `X^k` using `small_scalar_mul`, and finally sums all coefficient terms.
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
            decrypt_ring_gsw_bit_chunk_sum::<P, A, M>(
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

/// Builds the circuit that outputs the next seed and its decoded refresh encodings.
///
/// The returned circuit has two classes of inputs:
/// - `seed_bits` encrypted seed ciphertext inputs.
/// - one decryption-key wire input.
///
/// Internally the circuit calls the Goldreich encrypted-seed expansion subcircuit.  The child
/// subcircuit outputs `next_seed`, `errors`, and `masks` as flattened ciphertext wires; this
/// builder reconstructs the logical ciphertext wrappers from templates so it can decrypt the
/// `errors` and `masks` with the Ring-GSW gadget decryption API.
///
/// Final output order:
/// 1. the encrypted `next_seed` ciphertext wires,
/// 2. one decoded refresh encoding wire per element, computed as `decrypted_error[element] +
///    decrypted_mask[element]`.
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
    assert!(seed_bits >= 5, "Goldreich PRG requires at least five encrypted seed bits");
    assert!(input_bits_per_step > 0, "input_bits_per_step must be positive");
    assert!(v_bits > 0, "v_bits must be positive");

    // The outer circuit owns the actual public API inputs: encrypted seed ciphertexts followed by a
    // decryption-key wire.  The Goldreich expansion itself is registered below as a child
    // subcircuit and called with these seed inputs.
    let mut circuit = PolyCircuit::new();
    let encrypted_seeds = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let decryption_key = circuit.input(1).as_single_wire();

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
        );
    let next_seed_templates = goldreich_subcircuit.output.next_seed.clone();
    let error_templates = goldreich_subcircuit.output.errors.clone();
    let mask_templates = goldreich_subcircuit.output.masks.clone();
    let goldreich_subcircuit_id = circuit.register_sub_circuit(goldreich_subcircuit.circuit);
    let goldreich_outputs = circuit.call_sub_circuit(
        goldreich_subcircuit_id,
        ciphertext_wires::<P, RingGswCiphertext<P, A>>(&encrypted_seeds),
    );

    // Recover the three logical segments in the same order used by the Goldreich subcircuit output:
    // `next_seed`, then `errors`, then `masks`.
    let mut next_output_start = 0usize;
    let next_seed = recover_ciphertexts_from_outputs::<P, RingGswCiphertext<P, A>>(
        &next_seed_templates,
        &goldreich_outputs,
        &mut next_output_start,
    );
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
    assert_eq!(
        next_output_start,
        goldreich_outputs.len(),
        "all Goldreich subcircuit outputs must be consumed"
    );

    let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
    let ring_dim = ring_gsw.ring_dim();
    assert!(
        ring_dim <= ring_gsw.num_slots,
        "next-seed encoding slot assignment requires ring_dim {} <= Ring-GSW num_slots {}",
        ring_dim,
        ring_gsw.num_slots
    );
    let log_base_q = ring_gsw.log_base_q();
    let crt_depth = ring_gsw.crt_depth();
    let (q_moduli, _crt_bits, q_moduli_depth) = ring_gsw.params.to_crt();
    assert_eq!(q_moduli_depth, crt_depth, "Ring-GSW CRT depth must match params.to_crt()");
    assert_eq!(q_moduli.len(), crt_depth, "one CRT modulus is required for each CRT level");

    // Error layout:
    //
    //   seed_bits * input_bits_per_step
    //     * ciphertext_wire_count
    //     * log_base_q
    //     * ring_dim extra output groups
    //     * ring_dim
    //
    // The extra `ring_dim` factor is interpreted as `ring_dim` slot chunks.  The chunks are decoded
    // together with `decrypt_batch`, so one output wire carries all `ring_dim` slot results instead
    // of exposing one separate wire per slot.
    let expanded_seed_bits = seed_bits
        .checked_mul(input_bits_per_step)
        .expect("seed_bits * input_bits_per_step overflow");
    let expected_error_count = expanded_seed_bits
        .checked_mul(ciphertext_wire_count)
        .and_then(|value| value.checked_mul(ring_dim))
        .and_then(|value| value.checked_mul(log_base_q))
        .and_then(|value| value.checked_mul(ring_dim))
        .expect("Goldreich error ciphertext count overflow");
    assert_eq!(
        errors.len(),
        expected_error_count,
        "Goldreich error ciphertext count must match s * ciphertext_wire_count * n * m_g"
    );
    let base_error_chunks_per_seed =
        ciphertext_wire_count.checked_mul(log_base_q).expect("error chunks per seed overflow");
    let error_slot_chunk_len =
        base_error_chunks_per_seed.checked_mul(ring_dim).expect("error slot chunk overflow");
    let errors_per_seed =
        error_slot_chunk_len.checked_mul(ring_dim).expect("errors per seed overflow");
    let mut decrypted_errors = Vec::with_capacity(
        expanded_seed_bits
            .checked_mul(crt_depth)
            .and_then(|value| value.checked_mul(base_error_chunks_per_seed))
            .expect("decrypted error output count overflow"),
    );
    for seed_errors in errors.chunks_exact(errors_per_seed) {
        let slot_error_chunks = seed_errors.chunks_exact(error_slot_chunk_len).collect::<Vec<_>>();
        for base_chunk_idx in 0..base_error_chunks_per_seed {
            let error_chunks_by_slot = slot_error_chunks
                .iter()
                .map(|slot_errors| {
                    let start = base_chunk_idx * ring_dim;
                    &slot_errors[start..start + ring_dim]
                })
                .collect::<Vec<_>>();
            for &q_j in &q_moduli {
                // Each coefficient position is decrypted across all slot chunks in one
                // `decrypt_batch` call, then the coefficient terms are recombined into one
                // slotwise polynomial wire.
                decrypted_errors.push(decrypt_ring_gsw_coeff_slot_batch_sum::<P, A, M>(
                    &mut circuit,
                    error_chunks_by_slot.as_slice(),
                    decryption_key,
                    BigUint::from(q_j),
                ));
            }
        }
    }

    // Mask layout:
    //
    //   seed_bits * input_bits_per_step
    //     * ciphertext_wire_count
    //     * log_base_q
    //     * ring_dim extra output groups
    //     * crt_depth
    //     * ring_dim
    //     * v_bits
    //
    // Unlike errors, masks already contain a distinct `ring_dim * v_bits` ciphertext block for
    // every CRT level.  The extra `ring_dim` factor is again interpreted as slot chunks and decoded
    // together with `decrypt_batch`.
    let expected_mask_count = expanded_seed_bits
        .checked_mul(ciphertext_wire_count)
        .and_then(|value| value.checked_mul(ring_dim))
        .and_then(|value| value.checked_mul(v_bits))
        .and_then(|value| value.checked_mul(log_base_q))
        .and_then(|value| value.checked_mul(crt_depth))
        .and_then(|value| value.checked_mul(ring_dim))
        .expect("Goldreich mask ciphertext count overflow");
    assert_eq!(
        masks.len(),
        expected_mask_count,
        "Goldreich mask ciphertext count must match s * ciphertext_wire_count * n * v * m_g * d"
    );
    let full_q: Arc<BigUint> = ring_gsw.params.modulus().into();
    let base_mask_group_count =
        ciphertext_wire_count.checked_mul(log_base_q).expect("base mask group count overflow");
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    let mask_group_len =
        mask_q_chunk_len.checked_mul(crt_depth).expect("mask group length overflow");
    let mask_slot_chunk_len =
        base_mask_group_count.checked_mul(mask_group_len).expect("mask slot chunk length overflow");
    let masks_per_seed =
        mask_slot_chunk_len.checked_mul(ring_dim).expect("masks per seed overflow");
    let mut decrypted_masks = Vec::with_capacity(
        expanded_seed_bits
            .checked_mul(base_mask_group_count)
            .and_then(|value| value.checked_mul(crt_depth))
            .expect("decrypted mask output count overflow"),
    );
    for seed_masks in masks.chunks_exact(masks_per_seed) {
        let slot_mask_chunks = seed_masks.chunks_exact(mask_slot_chunk_len).collect::<Vec<_>>();
        for base_group_idx in 0..base_mask_group_count {
            for crt_idx in 0..crt_depth {
                let mask_q_chunks_by_slot = slot_mask_chunks
                    .iter()
                    .map(|slot_masks| {
                        let start = base_group_idx * mask_group_len + crt_idx * mask_q_chunk_len;
                        &slot_masks[start..start + mask_q_chunk_len]
                    })
                    .collect::<Vec<_>>();
                // The bit-wise and coefficient-wise collapse is performed across all slot chunks,
                // producing one slotwise polynomial wire for this CRT-specific mask group.
                decrypted_masks.push(decrypt_ring_gsw_bit_slot_batch_sum::<P, A, M>(
                    &mut circuit,
                    mask_q_chunks_by_slot.as_slice(),
                    decryption_key,
                    v_bits,
                    full_q.as_ref(),
                ));
            }
        }
    }

    assert_eq!(
        decrypted_errors.len(),
        decrypted_masks.len(),
        "decoded error and mask vectors must have the same length before element-wise addition"
    );
    let decoded_refresh_encodings = decrypted_errors
        .iter()
        .zip(decrypted_masks.iter())
        .map(|(&error, &mask)| circuit.add_gate(error, mask).as_single_wire())
        .collect::<Vec<_>>();

    // The public output includes the next seed first, followed by the decoded per-element refresh
    // material.  The next seed stays encrypted.  Errors and masks are not exposed separately; each
    // output refresh encoding is the element-wise sum of the corresponding decoded error and mask.
    let mut circuit_outputs = ciphertext_wires::<P, RingGswCiphertext<P, A>>(&next_seed);
    circuit_outputs.extend(decoded_refresh_encodings.iter().copied().map(BatchedWire::single));
    circuit.output(circuit_outputs);
    circuit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldreich_output_size_matches_formula() {
        assert_eq!(
            goldreich_noise_refresh_output_size(7, 11, 3, 8, 5, 2, 13),
            11 * (7 + 3 * 7 * (8 * 5 * 8 + 8 * 13 * 5 * 2 * 8))
        );
    }

    #[test]
    fn goldreich_output_sizes_split_cbd_and_uniform_segments() {
        assert_eq!(
            goldreich_noise_refresh_output_sizes(7, 11, 3, 8, 5, 2, 13),
            GoldreichNoiseRefreshOutputSizes {
                next_seed_bits: 11 * 7,
                mask_bits: 11 * 3 * 7 * 8 * 13 * 5 * 2 * 8,
                cbd_values: 11 * 3 * 7 * 8 * 5 * 8,
                total: 11 * (7 + 3 * 7 * (8 * 5 * 8 + 8 * 13 * 5 * 2 * 8)),
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
}
