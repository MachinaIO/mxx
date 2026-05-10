//! Benchmark-estimation helpers for masked decoder work.

use num_bigint::BigUint;
use num_traits::Zero;

use crate::bench_estimator::{
    CircuitBenchEstimate, CircuitBenchSummary, scale_independent_summary,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitDecomposedRefreshMaterialCounts {
    pub(crate) error_prg_output_count: usize,
    pub(crate) mask_prg_output_count: usize,
    pub(crate) error_decrypt_contribution_count: usize,
    pub(crate) mask_decrypt_contribution_count: usize,
    pub(crate) mask_polynomial_count: usize,
    pub(crate) merge_add_count: usize,
}

impl BitDecomposedRefreshMaterialCounts {
    pub(crate) fn total_prg_output_count(&self) -> usize {
        self.error_prg_output_count
            .checked_add(self.mask_prg_output_count)
            .expect("bit-decomposed refresh PRG output count overflow")
    }

    pub(crate) fn total_decrypt_contribution_count(&self) -> usize {
        self.error_decrypt_contribution_count
            .checked_add(self.mask_decrypt_contribution_count)
            .expect("bit-decomposed refresh decrypt contribution count overflow")
    }
}

fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    let summary = CircuitBenchSummary::from_nanos(
        estimate.total_time,
        estimate.latency,
        estimate.max_parallelism,
    );
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(estimate.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time.clone()).sum::<BigUint>();
    let latency = parts.iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism =
        parts.iter().map(|part| part.max_parallelism.clone()).max().unwrap_or_else(BigUint::zero);
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

fn parallel_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts.iter().map(|part| part.total_time.clone()).sum::<BigUint>();
    let latency = parts.iter().map(|part| part.latency).fold(0.0f64, f64::max);
    let max_parallelism = parts.iter().map(|part| part.max_parallelism.clone()).sum::<BigUint>();
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.iter().map(|part| part.peak_vram).sum::<usize>())
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

/// Computes the independent PRG/decrypt/reduction work counts for a
/// bit-decomposed noise-refresh material circuit.
///
/// Non-debug material expands one PRG stream per `(slot, digit)`. Each stream
/// produces `ring_dim` CBD error ciphertexts and `crt_depth * ring_dim *
/// mask_bits` mask ciphertexts. Error ciphertexts are decrypted once per CRT
/// level, while each bit-decomposed mask polynomial decrypts `ring_dim *
/// mask_bits` ciphertext-bit contributions.
pub(crate) fn bit_decomposed_refresh_material_counts(
    ring_dim: usize,
    modulus_digits: usize,
    crt_depth: usize,
    mask_bits: usize,
    reuse_single_material: bool,
) -> BitDecomposedRefreshMaterialCounts {
    assert!(ring_dim > 0, "ring_dim must be positive");
    assert!(modulus_digits > 0, "modulus_digits must be positive");
    assert!(crt_depth > 0, "crt_depth must be positive");
    assert!(mask_bits > 0, "mask_bits must be positive");

    if reuse_single_material {
        return BitDecomposedRefreshMaterialCounts {
            error_prg_output_count: 1,
            mask_prg_output_count: 1,
            error_decrypt_contribution_count: ring_dim
                .checked_mul(crt_depth)
                .expect("debug refresh error decrypt contribution count overflow"),
            mask_decrypt_contribution_count: crt_depth
                .checked_mul(ring_dim)
                .and_then(|count| count.checked_mul(mask_bits))
                .expect("debug refresh mask decrypt contribution count overflow"),
            mask_polynomial_count: crt_depth,
            merge_add_count: crt_depth,
        };
    }

    let slot_digit_count =
        ring_dim.checked_mul(modulus_digits).expect("refresh slot-digit count overflow");
    let mask_polynomial_count =
        slot_digit_count.checked_mul(crt_depth).expect("refresh mask polynomial count overflow");
    let error_prg_output_count =
        slot_digit_count.checked_mul(ring_dim).expect("refresh error PRG output count overflow");
    let mask_prg_output_count = mask_polynomial_count
        .checked_mul(ring_dim)
        .and_then(|count| count.checked_mul(mask_bits))
        .expect("refresh mask PRG output count overflow");
    let error_decrypt_contribution_count = mask_polynomial_count
        .checked_mul(ring_dim)
        .expect("refresh error decrypt contribution count overflow");
    let mask_decrypt_contribution_count = mask_prg_output_count;

    BitDecomposedRefreshMaterialCounts {
        error_prg_output_count,
        mask_prg_output_count,
        error_decrypt_contribution_count,
        mask_decrypt_contribution_count,
        mask_polynomial_count,
        merge_add_count: mask_polynomial_count,
    }
}

/// Estimates a noise-refresh material circuit from representative units without
/// materializing the full circuit.
pub(crate) fn bit_decomposed_refresh_material_summary(
    error_prg_unit: CircuitBenchSummary,
    mask_prg_unit: CircuitBenchSummary,
    decrypt_contribution_unit: CircuitBenchSummary,
    add_unit: CircuitBenchEstimate,
    counts: BitDecomposedRefreshMaterialCounts,
    mask_bits: usize,
) -> CircuitBenchSummary {
    let error_prg = scale_independent_summary(error_prg_unit, counts.error_prg_output_count);
    let mask_prg = scale_independent_summary(mask_prg_unit, counts.mask_prg_output_count);
    let decrypt_contributions = scale_independent_summary(
        decrypt_contribution_unit,
        counts.total_decrypt_contribution_count(),
    );
    let mask_reduction = bit_decomposed_polynomial_mask_reduction_summary(
        add_unit.clone(),
        Some(add_unit.clone()),
        Some(add_unit.clone()),
        mask_bits,
        counts.mask_polynomial_count,
    );
    let merge = scale_independent_summary(estimate_summary(add_unit), counts.merge_add_count);

    sequential_summaries(&[error_prg, mask_prg, decrypt_contributions, mask_reduction, merge])
}

/// Estimate the representative CBD error PRG unit from one measured uniform
/// Goldreich output plus the pairwise additions used to combine CBD terms.
///
/// The concrete CBD representative circuit evaluates `cbd_n` positive and
/// `cbd_n` negative Goldreich outputs, reduces each side with a balanced add
/// tree, then subtracts the negative sum. This helper keeps that dependency
/// structure without materializing the full CBD circuit in benchmark
/// estimation.
pub(crate) fn goldreich_cbd_error_prg_summary(
    uniform_output_unit: CircuitBenchSummary,
    add_unit: CircuitBenchEstimate,
    sub_unit: CircuitBenchEstimate,
    cbd_n: usize,
) -> CircuitBenchSummary {
    assert!(cbd_n > 0, "CBD representative PRG estimate requires cbd_n > 0");
    let output_count = cbd_n.checked_mul(2).expect("CBD PRG output count overflow");
    let prg_outputs = scale_independent_summary(uniform_output_unit, output_count);
    let positive_reduce = pairwise_reduction_summary(add_unit, cbd_n);
    let negative_reduce = positive_reduce.clone();
    let reductions = parallel_summaries(&[positive_reduce, negative_reduce]);
    let subtract = estimate_summary(sub_unit);
    sequential_summaries(&[prg_outputs, reductions, subtract])
}

/// Number of independent Ring-GSW decrypt contributions in a bit-decomposed
/// polynomial mask.
///
/// A polynomial mask has `ring_dim` coefficients, each coefficient is
/// represented by `mask_bits` encrypted bits, and callers may have multiple
/// independent logical outputs.
pub(crate) fn bit_decomposed_polynomial_mask_decrypt_contribution_count(
    ring_dim: usize,
    mask_bits: usize,
    output_count: usize,
) -> usize {
    assert!(mask_bits > 0, "mask_bits must be positive");
    ring_dim
        .checked_mul(mask_bits)
        .and_then(|count| count.checked_mul(output_count))
        .expect("bit-decomposed polynomial mask decrypt contribution count overflow")
}

/// Scale a representative one-ciphertext-bit decrypt contribution to a full
/// bit-decomposed polynomial mask decrypt workload.
pub(crate) fn scale_bit_decomposed_polynomial_mask_decrypt_contributions(
    unit: CircuitBenchSummary,
    ring_dim: usize,
    mask_bits: usize,
    output_count: usize,
) -> (CircuitBenchSummary, usize) {
    let contribution_count = bit_decomposed_polynomial_mask_decrypt_contribution_count(
        ring_dim,
        mask_bits,
        output_count,
    );
    (scale_independent_summary(unit, contribution_count), contribution_count)
}

/// Number of additions needed to reduce `mask_bits` decrypted bit terms.
pub(crate) fn bit_decomposed_mask_reduce_add_count(mask_bits: usize) -> usize {
    mask_bits.saturating_sub(1)
}

/// Pairwise-addition tree depth needed to reduce `input_count` values to one.
pub(crate) fn pairwise_reduction_depth(input_count: usize) -> usize {
    if input_count <= 1 {
        return 0;
    }
    usize::BITS as usize - (input_count - 1).leading_zeros() as usize
}

fn pairwise_reduction_summary(
    add_unit: CircuitBenchEstimate,
    input_count: usize,
) -> CircuitBenchSummary {
    let add_count = input_count.saturating_sub(1);
    if add_count == 0 {
        return CircuitBenchSummary::from_nanos(BigUint::from(0u8), 0.0, BigUint::from(0u8));
    }
    let depth = pairwise_reduction_depth(input_count);
    let first_layer_width = input_count / 2;
    let total_time = add_unit.total_time.clone() * BigUint::from(add_count);
    let latency = add_unit.latency * depth as f64;
    let max_parallelism = add_unit.max_parallelism.clone() * BigUint::from(first_layer_width);
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(add_unit.peak_vram.saturating_mul(first_layer_width))
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

/// Estimate the arithmetic that recombines bit-decomposed polynomial-mask
/// decrypt terms after each one-bit Ring-GSW decrypt contribution has run.
///
/// The secret-dependent and public-bottom branches each reduce `mask_bits`
/// terms with a balanced tree, so their latency is `ceil(log2(mask_bits))`
/// add gates rather than `mask_bits - 1` sequential gates. `combine_unit`
/// models an optional branch-combine add, and `center_unit` models the
/// optional centering add/sub applied after reduction.
pub(crate) fn bit_decomposed_polynomial_mask_reduction_summary(
    add_unit: CircuitBenchEstimate,
    combine_unit: Option<CircuitBenchEstimate>,
    center_unit: Option<CircuitBenchEstimate>,
    mask_bits: usize,
    polynomial_count: usize,
) -> CircuitBenchSummary {
    assert!(mask_bits > 0, "mask_bits must be positive");
    let branch_reduce = pairwise_reduction_summary(add_unit, mask_bits);
    // Ring-GSW decrypt returns split branches. The secret-dependent branch and public-bottom
    // branch reduce the same `mask_bits` terms independently, so their total work and peak
    // parallelism add, while their contribution to critical-path latency is the maximum of the two
    // identical balanced reduction trees.
    let mut per_polynomial_parts =
        vec![parallel_summaries(&[branch_reduce.clone(), branch_reduce])];
    if let Some(combine_unit) = combine_unit {
        per_polynomial_parts.push(estimate_summary(combine_unit));
    }
    if let Some(center_unit) = center_unit {
        per_polynomial_parts.push(estimate_summary(center_unit));
    }
    scale_independent_summary(sequential_summaries(&per_polynomial_parts), polynomial_count)
}
