use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::Zero;

use crate::{
    bench_estimator::{CircuitBenchEstimate, CircuitBenchSummary, scale_independent_summary},
    circuit::PolyCircuit,
    decoder::mask_circuit::append_one_ciphertext_bit_decrypt,
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPoly},
        fhe::{ring_gsw::RingGswCiphertext, ring_gsw_nested_rns::NestedRnsRingGswContext},
        fhe_prg::goldreich::{GoldreichEdge, GoldreichFhePrg, GoldreichGraph},
    },
    matrix::PolyMatrix,
    poly::Poly,
};

pub(crate) fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
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

pub(crate) fn scale_estimate(estimate: CircuitBenchEstimate, count: usize) -> CircuitBenchSummary {
    scale_summary(estimate_summary(estimate), count)
}

pub(crate) fn scale_estimate_biguint(
    estimate: CircuitBenchEstimate,
    count: &BigUint,
) -> CircuitBenchSummary {
    scale_summary_biguint(estimate_summary(estimate), count)
}

pub(crate) fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    scale_independent_summary(summary, count)
}

pub(crate) fn scale_summary_biguint(
    summary: CircuitBenchSummary,
    count: &BigUint,
) -> CircuitBenchSummary {
    let scaled = CircuitBenchSummary::from_nanos(
        summary.total_time.clone() * count,
        summary.latency,
        summary.max_parallelism.clone() * count,
    );
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

pub(crate) fn repeat_sequential_summary(
    summary: CircuitBenchSummary,
    count: usize,
) -> CircuitBenchSummary {
    let total_time = summary.total_time * BigUint::from(count);
    let latency = summary.latency * count as f64;
    let repeated = CircuitBenchSummary::from_nanos(total_time, latency, summary.max_parallelism);
    #[cfg(feature = "gpu")]
    {
        repeated.with_peak_vram(summary.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        repeated
    }
}

pub(crate) fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
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

pub(crate) fn parallel_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
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

pub(crate) fn matrix_compact_bytes_for_shape(
    nrow: usize,
    ncol: usize,
    ring_dim: usize,
    modulus_bits: u16,
) -> usize {
    let coeff_count =
        nrow.checked_mul(ncol).and_then(|count| count.checked_mul(ring_dim)).unwrap_or(usize::MAX);
    let payload = coeff_count.checked_mul(modulus_bits as usize).unwrap_or(usize::MAX).div_ceil(8);
    // Compact-byte size estimate used without materializing the underlying matrix payload.
    payload + 128
}

pub(crate) fn representative_goldreich_prg_one_output_circuit<P>(
    mut circuit: PolyCircuit<P>,
    ring_gsw_context: Arc<NestedRnsRingGswContext<P>>,
    representative_seed_bits: usize,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
{
    assert!(
        representative_seed_bits >= 5,
        "representative Goldreich PRG circuit requires at least five seed bits"
    );
    let seed_ciphertexts = (0..representative_seed_bits)
        .map(|_| {
            RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let graph = GoldreichGraph::from_edges(
        representative_seed_bits,
        vec![GoldreichEdge::new([0, 1, 2], [3, 4])],
        Default::default(),
    );
    let goldreich = GoldreichFhePrg::from_public_graph(&mut circuit, ring_gsw_context, graph);
    let outputs = goldreich.evaluate_uniform(&seed_ciphertexts, &mut circuit);
    circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
    circuit
}

pub(crate) fn prf_mask_decrypt_one_ciphertext_bit_circuit<P, M>(
    mut circuit: PolyCircuit<P>,
    ring_gsw_context: Arc<NestedRnsRingGswContext<P>>,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    NestedRnsPoly<P>: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let decrypted = append_one_ciphertext_bit_decrypt::<P, NestedRnsPoly<P>, M>(
        &mut circuit,
        ring_gsw_context,
        BigUint::from(2u64),
    );
    circuit.output(vec![decrypted.secret_dependent, decrypted.public_bottom]);
    circuit
}
