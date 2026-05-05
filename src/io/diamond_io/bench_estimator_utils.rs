use crate::bench_estimator::{CircuitBenchEstimate, CircuitBenchSummary};

pub(super) fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    let summary =
        CircuitBenchSummary::new(estimate.total_time, estimate.latency, estimate.max_parallelism);
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

pub(super) fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    let total_time = summary.total_time * count as f64;
    let max_parallelism = summary
        .max_parallelism
        .checked_mul(count as u128)
        .expect("DiamondIO benchmark parallelism overflow while scaling summary");
    let scaled = CircuitBenchSummary::new(total_time, summary.latency, max_parallelism);
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
    let total_time = summary.total_time * count as f64;
    let latency = summary.latency * count as f64;
    let repeated = CircuitBenchSummary::new(total_time, latency, summary.max_parallelism);
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
    let total_time = parts.iter().map(|part| part.total_time).sum::<f64>();
    let latency = parts.iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism = parts.iter().map(|part| part.max_parallelism).max().unwrap_or(0);
    let summary = CircuitBenchSummary::new(total_time, latency, max_parallelism);
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
    let total_time = parts.iter().map(|part| part.total_time).sum::<f64>();
    let latency = parts.iter().map(|part| part.latency).fold(0.0f64, f64::max);
    let max_parallelism = parts
        .iter()
        .map(|part| part.max_parallelism)
        .try_fold(0u128, |acc, value| acc.checked_add(value))
        .expect("DiamondIO benchmark parallelism overflow while parallelizing summaries");
    let summary = CircuitBenchSummary::new(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.iter().map(|part| part.peak_vram).sum::<usize>())
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}
