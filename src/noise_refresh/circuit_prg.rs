//! Goldreich PRG material generation for the noise-refresh protocol.
//!
//! This module owns only the first phase of the step function: expanding encrypted seed bits into
//! encrypted PRG material.  It does not decrypt or combine that material.  The decrypt phase in
//! `circuit_decrypt` can consume the material produced here, and `circuit_merge` can also consume
//! benchmark or fixture material that has already been decoded into the same slotwise layout.
//!
//! The generated logical output order is always CBD `errors`, then `masks`.  Each logical
//! ciphertext is flattened only at the `PolyCircuit` boundary via
//! `RingGswCiphertext::sub_circuit_wires`.

use crate::{
    circuit::PolyCircuit,
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::{GoldreichFheCbdPrg, evaluate_goldreich_uniform_range},
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

#[derive(Debug, Clone)]
pub(crate) struct GoldreichNoiseRefreshMaterial<P, A>
where
    P: Poly,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    pub errors: Vec<RingGswCiphertext<P, A>>,
    pub masks: Vec<RingGswCiphertext<P, A>>,
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

pub(crate) fn build_goldreich_encrypted_seed_material_ranges<P, A>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    output_scope_idx: usize,
    error_range_start: usize,
    error_range_len: usize,
    mask_ranges: &[(usize, usize)],
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    let mut circuit = ring_gsw.fresh_circuit();
    let input_size = seed_bits;
    assert!(input_size >= 5, "Goldreich PRG requires at least five encrypted seed bits");
    assert!(error_range_len > 0, "noise-refresh error range length must be positive");
    assert!(!mask_ranges.is_empty(), "noise-refresh mask ranges must be nonempty");

    let encrypted_seeds = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();

    let scope_counter =
        u64::try_from(output_scope_idx).expect("noise-refresh output scope index exceeds u64");
    let material = evaluate_goldreich_noise_refresh_material_ranges(
        &mut circuit,
        ring_gsw,
        &encrypted_seeds,
        v_bits,
        graph_seed,
        cbd_n,
        scope_counter,
        error_range_start,
        error_range_len,
        mask_ranges,
    );

    circuit.output(
        material
            .errors
            .iter()
            .chain(material.masks.iter())
            .flat_map(|output| output.sub_circuit_wires()),
    );
    circuit
}

pub(crate) fn build_goldreich_encrypted_seed_error_material_range<P, A>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    output_scope_idx: usize,
    error_range_start: usize,
    error_range_len: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    assert!(error_range_len > 0, "noise-refresh error range length must be positive");
    let mut circuit = ring_gsw.fresh_circuit();
    assert!(seed_bits >= 5, "Goldreich PRG requires at least five encrypted seed bits");
    let encrypted_seeds = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let output_sizes = goldreich_noise_refresh_output_sizes(
        ring_gsw.params.ring_dimension() as usize,
        ring_gsw.params.modulus_digits(),
        ring_gsw.params.to_crt().2,
        1,
    );
    let error_range_end =
        error_range_start.checked_add(error_range_len).expect("error range end overflow");
    assert!(
        error_range_end <= output_sizes.cbd_values,
        "noise-refresh error range exceeds conceptual CBD output size"
    );
    let scope_counter =
        u64::try_from(output_scope_idx).expect("noise-refresh output scope index exceeds u64");
    let cbd_seed =
        derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshCBD/v1", scope_counter);
    let cbd_prf = GoldreichFheCbdPrg::setup_range(
        &mut circuit,
        ring_gsw,
        seed_bits,
        output_sizes.cbd_values,
        error_range_start,
        error_range_len,
        cbd_seed,
        cbd_n,
    );
    let errors = cbd_prf.evaluate_cbd_prf(&encrypted_seeds, &mut circuit);
    circuit.output(errors.iter().flat_map(|output| output.sub_circuit_wires()));
    circuit
}

pub(crate) fn build_goldreich_encrypted_seed_mask_material_ranges<P, A>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    output_scope_idx: usize,
    mask_ranges: &[(usize, usize)],
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    assert!(!mask_ranges.is_empty(), "noise-refresh mask ranges must be nonempty");
    let mut circuit = ring_gsw.fresh_circuit();
    assert!(seed_bits >= 5, "Goldreich PRG requires at least five encrypted seed bits");
    let encrypted_seeds = (0..seed_bits)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let output_sizes = goldreich_noise_refresh_output_sizes(
        ring_gsw.params.ring_dimension() as usize,
        ring_gsw.params.modulus_digits(),
        ring_gsw.params.to_crt().2,
        v_bits,
    );
    for &(mask_start, mask_len) in mask_ranges {
        assert!(mask_len > 0, "mask range length must be positive");
        let mask_end = mask_start.checked_add(mask_len).expect("mask range end overflow");
        assert!(
            mask_end <= output_sizes.mask_bits,
            "noise-refresh mask range exceeds conceptual mask output size"
        );
    }
    let scope_counter =
        u64::try_from(output_scope_idx).expect("noise-refresh output scope index exceeds u64");
    let mask_seed =
        derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshMask/v1", scope_counter);
    let masks = mask_ranges
        .iter()
        .flat_map(|&(mask_start, mask_len)| {
            evaluate_goldreich_uniform_range(
                &mut circuit,
                ring_gsw.clone(),
                &encrypted_seeds,
                output_sizes.mask_bits,
                mask_start,
                mask_len,
                mask_seed,
            )
        })
        .collect::<Vec<_>>();
    circuit.output(masks.iter().flat_map(|output| output.sub_circuit_wires()));
    circuit
}

pub(crate) fn evaluate_goldreich_noise_refresh_material_ranges<P, A>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<RingGswContext<P, A>>,
    encrypted_seeds: &[RingGswCiphertext<P, A>],
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    scope_counter: u64,
    error_range_start: usize,
    error_range_len: usize,
    mask_ranges: &[(usize, usize)],
) -> GoldreichNoiseRefreshMaterial<P, A>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    let input_size = encrypted_seeds.len();
    assert!(input_size >= 5, "Goldreich PRG requires at least five encrypted seed bits");
    assert!(v_bits > 0, "v_bits must be positive");
    assert!(cbd_n > 0, "cbd_n must be positive");
    assert!(error_range_len > 0, "error range length must be positive");
    assert!(!mask_ranges.is_empty(), "mask ranges must be nonempty");
    let output_sizes = goldreich_noise_refresh_output_sizes(
        ring_gsw.params.ring_dimension() as usize,
        ring_gsw.params.modulus_digits(),
        ring_gsw.params.to_crt().2,
        v_bits,
    );
    let error_range_end =
        error_range_start.checked_add(error_range_len).expect("error range end overflow");
    assert!(
        error_range_end <= output_sizes.cbd_values,
        "noise-refresh error range exceeds conceptual CBD output size"
    );
    for &(mask_start, mask_len) in mask_ranges {
        assert!(mask_len > 0, "mask range length must be positive");
        let mask_end = mask_start.checked_add(mask_len).expect("mask range end overflow");
        assert!(
            mask_end <= output_sizes.mask_bits,
            "noise-refresh mask range exceeds conceptual mask output size"
        );
    }

    let cbd_seed =
        derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshCBD/v1", scope_counter);
    let cbd_prf = GoldreichFheCbdPrg::setup_range(
        circuit,
        ring_gsw.clone(),
        input_size,
        output_sizes.cbd_values,
        error_range_start,
        error_range_len,
        cbd_seed,
        cbd_n,
    );
    let errors = cbd_prf.evaluate_cbd_prf(encrypted_seeds, circuit);

    let mask_seed =
        derive_noise_refresh_graph_seed(graph_seed, b"NoiseRefreshMask/v1", scope_counter);
    let masks = mask_ranges
        .iter()
        .flat_map(|&(mask_start, mask_len)| {
            evaluate_goldreich_uniform_range(
                circuit,
                ring_gsw.clone(),
                encrypted_seeds,
                output_sizes.mask_bits,
                mask_start,
                mask_len,
                mask_seed,
            )
        })
        .collect::<Vec<_>>();
    debug_assert_eq!(
        errors.len() + masks.len(),
        error_range_len + mask_ranges.iter().map(|(_, len)| *len).sum::<usize>()
    );

    GoldreichNoiseRefreshMaterial { errors, masks }
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
