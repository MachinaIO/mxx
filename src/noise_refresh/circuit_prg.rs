//! Goldreich PRG material generation for the noise-refresh protocol.
//!
//! This module owns only the first phase of the step function: expanding encrypted seed bits into
//! encrypted PRG material.  It does not decrypt or combine that material.  The formatting phase in
//! `circuit_format` can consume the material produced here, but it can also consume benchmark or
//! fixture material with the same layout.
//!
//! The generated logical output order is always CBD `errors`, then `masks`.  Each logical
//! ciphertext is flattened only at the `PolyCircuit` boundary via
//! `RingGswCiphertext::sub_circuit_wires`.

use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::{GoldreichFheCbdPrg, GoldreichFhePrg},
    },
    poly::{Poly, PolyParams},
};
use digest::Digest;
use keccak_asm::Keccak256;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldreichNoiseRefreshOutputSizes {
    /// Number of encrypted Boolean ciphertexts used as mask bits.
    pub mask_bits: usize,
    /// Number of encrypted CBD integer ciphertexts used as error coefficients.
    pub cbd_values: usize,
    /// Total number of logical ciphertext outputs before flattening each ciphertext into wires.
    pub total: usize,
}

/// Computes the logical output sizes of one Goldreich-based noise-refresh wire expansion.
///
/// The noise-refresh PRG no longer generates encrypted next-seed material.  It only generates the
/// refresh material needed to update one existing encoding wire across all CRT levels:
/// `log_base_q * ring_dim` coefficient-level CBD errors and `crt_depth * log_base_q * ring_dim *
/// v_bits` mask bits.
///
/// Output layout:
/// - `cbd_values = log_base_q * ring_dim`
/// - `mask_bits = crt_depth * log_base_q * ring_dim * v_bits`
///
/// The returned sizes count logical Ring-GSW ciphertext objects, not flattened `GateId`s.
pub fn goldreich_noise_refresh_output_sizes(
    ring_dim: usize,
    log_base_q: usize,
    crt_depth: usize,
    v_bits: usize,
) -> GoldreichNoiseRefreshOutputSizes {
    assert!(ring_dim > 0, "ring_dim must be positive");
    assert!(log_base_q > 0, "log_base_q must be positive");
    assert!(crt_depth > 0, "crt_depth must be positive");
    assert!(v_bits > 0, "v_bits must be positive");
    let cbd_values =
        log_base_q.checked_mul(ring_dim).expect("Goldreich noise-refresh CBD output size overflow");
    let mask_bits = log_base_q
        .checked_mul(ring_dim)
        .and_then(|value| value.checked_mul(v_bits))
        .and_then(|value| value.checked_mul(crt_depth))
        .expect("Goldreich noise-refresh uniform output size overflow");
    let total = mask_bits
        .checked_add(cbd_values)
        .expect("Goldreich noise-refresh total output size overflow");
    GoldreichNoiseRefreshOutputSizes { mask_bits, cbd_values, total }
}

pub fn goldreich_noise_refresh_output_size(
    ring_dim: usize,
    log_base_q: usize,
    crt_depth: usize,
    v_bits: usize,
) -> usize {
    goldreich_noise_refresh_output_sizes(ring_dim, log_base_q, crt_depth, v_bits).total
}

/// Builds the Goldreich encrypted-seed expansion circuit.
///
/// The returned circuit creates `seed_bits` encrypted Boolean ciphertext inputs internally.  It
/// then evaluates two domain-separated PRG components on those inputs:
///
/// 1. A uniform Goldreich PRG with output size `mask_bits`.
/// 2. A CBD PRF with output size `cbd_values`. These outputs are encrypted integer error
///    coefficients rather than encrypted bits.
///
/// The circuit output order is `errors`, then `masks`, with each logical ciphertext flattened into
/// its `sub_circuit_wires()` representation.
pub fn evaluate_goldreich_encrypted_seeds<P, A>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    output_scope_idx: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    build_goldreich_encrypted_seeds_with_output::<P, A>(
        ring_gsw,
        seed_bits,
        v_bits,
        graph_seed,
        cbd_n,
        output_scope_idx,
    )
}

pub(crate) fn build_goldreich_encrypted_seeds_with_output<P, A>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    output_scope_idx: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    // `output_scope_idx` identifies the logical refresh wire being generated.  It is deliberately
    // not a mutable "consume" counter: independent builders can derive the same scope id from their
    // own work item and construct PRG material in parallel without sharing state.
    //
    // TODO(noise-refresh): implement graph-collision detection for scoped Goldreich material.  For
    // now, scopes are domain-separated by hash inputs, but we do not exhaustively compare generated
    // graph structures across `(domain, output_scope_idx)` pairs because that check is expensive
    // for benchmark-sized circuits.
    // This is a standalone subcircuit: all encrypted seed ciphertexts are declared as inputs of the
    // newly-created circuit rather than being passed in from an outer builder.
    let mut circuit = ring_gsw.fresh_circuit();
    let input_size = seed_bits;
    assert!(input_size >= 5, "Goldreich PRG requires at least five encrypted seed bits");

    // Create one encrypted Ring-GSW Boolean ciphertext input for each seed bit.
    let encrypted_seeds = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    // Compute the one-wire refresh material size from the Ring-GSW parameters.  The encrypted seed
    // input count controls the Goldreich PRG input size only; it no longer scales the amount of
    // material produced by this noise-refresh builder.
    let output_sizes = goldreich_noise_refresh_output_sizes(
        ring_gsw.params.ring_dimension() as usize,
        ring_gsw.params.modulus_digits(),
        ring_gsw.params.to_crt().2,
        v_bits,
    );
    let scope_counter =
        u64::try_from(output_scope_idx).expect("noise-refresh output scope index exceeds u64");
    let mask_seed =
        derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshMask/v1", scope_counter);
    let mask_prg = GoldreichFhePrg::setup_range(
        &mut circuit,
        ring_gsw.clone(),
        input_size,
        output_sizes.mask_bits,
        0,
        output_sizes.mask_bits,
        mask_seed,
    );
    let masks = mask_prg.evaluate_uniform(&encrypted_seeds, &mut circuit);

    let cbd_seed =
        derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshCBD/v1", scope_counter);
    let cbd_prf = GoldreichFheCbdPrg::setup_range(
        &mut circuit,
        ring_gsw,
        input_size,
        output_sizes.cbd_values,
        0,
        output_sizes.cbd_values,
        cbd_seed,
        cbd_n,
    );
    let errors = cbd_prf.evaluate_cbd_prf(&encrypted_seeds, &mut circuit);
    debug_assert_eq!(errors.len() + masks.len(), output_sizes.cbd_values + output_sizes.mask_bits);

    circuit.output(errors.iter().chain(masks.iter()).flat_map(|output| output.sub_circuit_wires()));
    let _ = (mask_prg, cbd_prf);
    circuit
}

pub(crate) fn derive_noise_refresh_graph_seed(
    base_seed: [u8; 32],
    label: &[u8],
    counter: u64,
) -> [u8; 32] {
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
