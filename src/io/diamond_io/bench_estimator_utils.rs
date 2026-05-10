use num_bigint::BigUint;
use num_traits::Zero;

use crate::bench_estimator::{
    CircuitBenchEstimate, CircuitBenchSummary, scale_independent_summary,
};

pub(super) fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
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

pub(super) fn scale_estimate(estimate: CircuitBenchEstimate, count: usize) -> CircuitBenchSummary {
    scale_summary(estimate_summary(estimate), count)
}

pub(super) fn scale_estimate_biguint(
    estimate: CircuitBenchEstimate,
    count: &BigUint,
) -> CircuitBenchSummary {
    scale_summary_biguint(estimate_summary(estimate), count)
}

pub(super) fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    scale_independent_summary(summary, count)
}

pub(super) fn scale_summary_biguint(
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

pub(super) fn repeat_sequential_summary(
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

pub(super) fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
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

pub(super) fn parallel_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
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
