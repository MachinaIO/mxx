//! Goldreich PRG material generation for the noise-refresh protocol.
//!
//! This module owns only the first phase of the step function: expanding encrypted seed bits into
//! encrypted PRG material.  It does not decrypt or combine that material.  The formatting phase in
//! `step_fn_format` can consume the material produced here, but it can also consume benchmark or
//! fixture material with the same layout.
//!
//! The generated logical output order is always `enc_next_seed`, then CBD `errors`, then `masks`.
//! Each logical ciphertext is flattened only at the `PolyCircuit` boundary via
//! `BooleanCiphertext::sub_circuit_wires`.

use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticContext, ModularArithmeticPlanner},
        fhe::ring_gsw::RingGswContext,
        fhe_prg::goldreich::{
            BooleanCiphertext, GoldreichFheCbdPrg, GoldreichFhePrg, GoldreichGraph,
        },
    },
    poly::{Poly, PolyParams},
};
use digest::Digest;
use keccak_asm::Keccak256;
use std::{marker::PhantomData, ops::Range, sync::Arc};

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

pub trait NoiseRefreshCircuitFactory<P: Poly>: NoiseRefreshPrgParameters {
    /// Create a new standalone circuit that shares the helper registries owned by this context.
    ///
    /// Ring-GSW and its arithmetic gadgets pre-register helper subcircuits during setup.  A
    /// noise-refresh builder still returns a freshly constructed `PolyCircuit`, but that circuit
    /// must share those registries so later Ring-GSW arithmetic can call the pre-registered helper
    /// subcircuits by id.
    fn fresh_noise_refresh_circuit(&self) -> PolyCircuit<P>;
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

impl<P, A> NoiseRefreshCircuitFactory<P> for RingGswContext<P, A>
where
    P: Poly,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    fn fresh_noise_refresh_circuit(&self) -> PolyCircuit<P> {
        self.fresh_circuit()
    }
}

/// Logical PRG material recovered while constructing the Goldreich subcircuit.
///
/// The fields mirror the circuit output order but keep ciphertext wrappers around each logical
/// output.  Callers use these wrappers as templates when a parent circuit calls the subcircuit and
/// receives only flat output wires.
pub(crate) struct GoldreichNoiseRefreshOutput<P: Poly, C: BooleanCiphertext<P>> {
    /// Encrypted next-seed uniform outputs.
    ///
    /// These are still encrypted Boolean ciphertexts.  The name intentionally uses `enc_` to avoid
    /// confusing them with plaintext next-seed bits.
    pub(crate) enc_next_seed: Vec<C>,
    /// Uniform mask bits after `enc_next_seed`.
    ///
    /// The formatter later groups them by selected output chunk, gadget digit, output slot, CRT
    /// level, coefficient, and bit position.
    pub(crate) masks: Vec<C>,
    /// CBD PRF outputs.
    ///
    /// Each output is an encrypted integer coefficient, not an encrypted bit.  The formatter
    /// treats these as already coefficient-level Ring-GSW ciphertexts.
    pub(crate) errors: Vec<C>,
    _poly: PhantomData<P>,
}

/// A Goldreich material subcircuit together with logical output templates.
///
/// `circuit` is the actual standalone `PolyCircuit`; `output` is kept only while composing a parent
/// circuit so the parent can reconstruct ciphertext wrappers around the subcircuit outputs.
pub(crate) struct GoldreichNoiseRefreshCircuit<P: Poly, C: BooleanCiphertext<P>> {
    pub(crate) circuit: PolyCircuit<P>,
    pub(crate) output: GoldreichNoiseRefreshOutput<P, C>,
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
/// - `cbd_values = s * input_bits_per_step * ciphertext_wire_count * log_base_q * ring_dim`
/// - `mask_bits = s * input_bits_per_step * ciphertext_wire_count * ring_dim * v_bits * log_base_q
///   * crt_depth`
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
        .checked_mul(log_base_q)
        .and_then(|value| value.checked_mul(ring_dim))
        .expect("Goldreich noise-refresh CBD output size overflow");
    let mask_bits = encrypted_seed_wire_count
        .checked_mul(ring_dim)
        .and_then(|value| value.checked_mul(v_bits))
        .and_then(|value| value.checked_mul(log_base_q))
        .and_then(|value| value.checked_mul(crt_depth))
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
/// `enc_next_seed`, then `errors`, then `masks`, with each logical ciphertext flattened into its
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
    C::Context: NoiseRefreshCircuitFactory<P>,
{
    build_goldreich_encrypted_seeds_with_output::<P, C>(
        ring_gsw,
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
        None,
    )
    .circuit
}

pub(crate) fn build_goldreich_encrypted_seeds_with_output<P, C>(
    ring_gsw: Arc<C::Context>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    output_selection: Option<Range<usize>>,
) -> GoldreichNoiseRefreshCircuit<P, C>
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
    C::Context: NoiseRefreshCircuitFactory<P>,
{
    // `output_selection` is a range over conceptual encrypted next-seed logical outputs.  It is the
    // only caller-controlled range.  Matching CBD error and mask ranges are derived below from this
    // next-seed range so the caller cannot accidentally request mismatched PRG material.
    // This is a standalone subcircuit: all encrypted seed ciphertexts are declared as inputs of the
    // newly-created circuit rather than being passed in from an outer builder.
    let mut circuit = ring_gsw.fresh_noise_refresh_circuit();
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
    let output_selection = output_selection.unwrap_or(0..output_sizes.next_seed_bits);
    let enc_next_seed_count = output_selection.len();
    assert!(enc_next_seed_count > 0);
    assert!(
        output_selection.start <= output_sizes.next_seed_bits,
        "selected enc_next_seed range must start inside the conceptual next-seed output"
    );
    assert!(
        output_selection.end <= output_sizes.next_seed_bits,
        "selected enc_next_seed range must be contained in the conceptual next-seed output"
    );
    let errors_per_enc_next_seed = ciphertext_wire_count
        .checked_mul(ring_gsw.log_base_q())
        .and_then(|value| value.checked_mul(ring_gsw.ring_dim()))
        .expect("errors per enc_next_seed output overflow");
    let masks_per_enc_next_seed = ciphertext_wire_count
        .checked_mul(ring_gsw.ring_dim())
        .and_then(|value| value.checked_mul(v_bits))
        .and_then(|value| value.checked_mul(ring_gsw.log_base_q()))
        .and_then(|value| value.checked_mul(ring_gsw.crt_depth()))
        .expect("masks per enc_next_seed output overflow");
    let cbd_output_start = output_selection
        .start
        .checked_mul(errors_per_enc_next_seed)
        .expect("selected CBD output range start overflow");
    let cbd_output_count = enc_next_seed_count
        .checked_mul(errors_per_enc_next_seed)
        .expect("selected CBD output range length overflow");
    let mask_output_start = output_selection
        .start
        .checked_mul(masks_per_enc_next_seed)
        .expect("selected mask output range start overflow");
    let mask_output_count = enc_next_seed_count
        .checked_mul(masks_per_enc_next_seed)
        .expect("selected mask output range length overflow");

    // The uniform PRG conceptually produces `enc_next_seed || masks`, but the circuit evaluates
    // only the intervals implied by the selected next-seed logical outputs.  Callers do not choose
    // error or mask ranges directly: each selected next-seed bit expands to all ciphertext wires
    // needed by that bit's encoding, and those wires determine the matching CBD and mask ranges.
    // Range setup preserves the full-output indexing, so the mask interval uses different graph
    // edges from the next-seed interval even when both intervals have the same width.
    let uniform_output_size = output_sizes
        .next_seed_bits
        .checked_add(output_sizes.mask_bits)
        .expect("uniform Goldreich output size overflow");
    let enc_next_seed_prg = GoldreichFhePrg::setup_range(
        &mut circuit,
        ring_gsw.clone(),
        input_size,
        uniform_output_size,
        output_selection.start,
        enc_next_seed_count,
        graph_seed,
    );
    let enc_next_seed = enc_next_seed_prg.evaluate_uniform(&encrypted_seeds, &mut circuit);
    let mask_prg = GoldreichFhePrg::setup_range(
        &mut circuit,
        ring_gsw.clone(),
        input_size,
        uniform_output_size,
        output_sizes.next_seed_bits + mask_output_start,
        mask_output_count,
        graph_seed,
    );
    let masks = mask_prg.evaluate_uniform(&encrypted_seeds, &mut circuit);

    // The CBD PRF must not reuse the same public Goldreich graph as the uniform PRG for the same
    // input seed.  `setup_cbd_prf_distinct_from_uniform` domain-separates the graph seed and checks
    // graph structure collisions before returning.
    let cbd_prf = setup_cbd_prf_distinct_from_uniform(
        &mut circuit,
        ring_gsw,
        input_size,
        output_sizes.cbd_values,
        cbd_output_start,
        cbd_output_count,
        graph_seed,
        cbd_n,
        &[enc_next_seed_prg.graph(), mask_prg.graph()],
    );
    let errors = cbd_prf.evaluate_cbd_prf(&encrypted_seeds, &mut circuit);
    debug_assert_eq!(
        enc_next_seed.len() + errors.len() + masks.len(),
        enc_next_seed_count + cbd_output_count + mask_output_count
    );

    // Outer circuits see only flattened wires.  The logical grouping is preserved privately through
    // `GoldreichNoiseRefreshOutput`, but the actual `PolyCircuit` outputs are scalar/batched wires
    // in the exact order below.
    circuit.output(
        enc_next_seed
            .iter()
            .chain(errors.iter())
            .chain(masks.iter())
            .flat_map(|output| output.sub_circuit_wires()),
    );
    let _ = (enc_next_seed_prg, mask_prg, cbd_prf);
    let output = GoldreichNoiseRefreshOutput { enc_next_seed, masks, errors, _poly: PhantomData };
    GoldreichNoiseRefreshCircuit { circuit, output }
}

fn setup_cbd_prf_distinct_from_uniform<P, C>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<C::Context>,
    input_size: usize,
    conceptual_output_size: usize,
    range_start: usize,
    range_len: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    uniform_graphs: &[&GoldreichGraph],
) -> GoldreichFheCbdPrg<P, C>
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
{
    // The CBD PRF internally uses several Goldreich graphs.  They must not accidentally reuse the
    // uniform next-seed or mask graph for the same seed inputs, because that would couple the error
    // distribution to uniform material.  We domain-separate the seed and retry if this local
    // collision check fails.
    assert!(range_len > 0, "Goldreich CBD output range length must be positive");
    for counter in 0u64.. {
        // The counter gives a deterministic retry path if the derived CBD graph happens to collide
        // with the uniform graph.  In normal parameter ranges this should be rare, but the explicit
        // retry makes the non-overlap invariant part of setup rather than an assumption.
        let cbd_seed = derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshCBD/v1", counter);
        let cbd_prf = GoldreichFheCbdPrg::setup_range(
            circuit,
            ring_gsw.clone(),
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            cbd_seed,
            cbd_n,
        );
        if cbd_prf.uniform_graphs().iter().all(|graph| {
            uniform_graphs
                .iter()
                .all(|uniform_graph| !same_goldreich_graph_structure(graph, uniform_graph))
        }) {
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
    // Graph seeds are intentionally ignored here: two different seeds that produce the same edge
    // structure are still a collision for circuit semantics.
    lhs.input_size == rhs.input_size && lhs.edges == rhs.edges && lhs.generation == rhs.generation
}
