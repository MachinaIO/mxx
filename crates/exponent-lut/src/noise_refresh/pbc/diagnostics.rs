//! Aggregation for the opt-in PBC parameter diagnostic.
//!
//! This module records construction and layout costs only.  It deliberately
//! contains no ciphertext or operational-noise model.

use serde::{Deserialize, Serialize};

use std::time::Instant;

use super::{
    PbcError, PbcLayoutSeed, PbcParameters, PbcPublicLayout, PbcRootSeed, derive_attempt_seed,
    schedule::{ValidatedSupport, deterministic_matching, schedule_from_owners},
};

/// Stable names used by [`PbcDiagnosticReport::performance_values`].
pub const PERFORMANCE_CATEGORIES: [&str; 7] = [
    "pbc_layout_hashing",
    "pbc_matching",
    "pbc_real_selector_packages",
    "pbc_dummy_selector_packages",
    "pbc_bucket_cell_evaluations",
    "pbc_bucket_reductions",
    "pbc_padding_overhead",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Measurements from one attempted PBC layout generation.
pub struct PbcDiagnosticSample {
    /// Number of seed attempts performed.
    pub attempts: u32,
    /// Accepted attempt, if any.
    pub accepted_attempt: Option<u32>,
    /// Accepted rectangular bucket width, if any.
    pub bucket_width: Option<usize>,
    /// Count of width-limit failures.
    pub bucket_width_failures: u32,
    /// Count of matching failures.
    pub no_perfect_schedule_failures: u32,
    /// Layout construction time in nanoseconds.
    pub layout_time_nanos: u128,
    /// Matching time in nanoseconds.
    pub matching_time_nanos: u128,
}

impl PbcDiagnosticSample {
    /// Returns whether this sample produced an accepted layout.
    pub fn accepted(&self) -> bool {
        self.accepted_attempt.is_some()
    }
}

/// Measures one complete v1 key-layout generation attempt sequence.
///
/// The support is supplied together with the root seed so this API models the
/// supported honest key-generation order: the sparse support is fixed before
/// layout randomness is derived, and failed layout seeds are retried without
/// changing the support.  Callers cannot provide a pre-built public layout or
/// invoke the exact scheduler directly.  This keeps the diagnostic from
/// turning a long-lived public layout into an API for scheduling arbitrary
/// later supports.
///
/// The returned sample contains only aggregate timing, retry, and rectangular
/// layout information.  It never contains the support, selected slots, or the
/// private schedule.  Structural errors are returned immediately.  Exhausting
/// the declared retry budget is represented by an accepted-less sample so a
/// caller can include the failure in an aggregate success-rate report.
pub fn measure_key_layout(
    parameters: &PbcParameters,
    root_seed: PbcRootSeed,
    support: &[usize],
) -> Result<PbcDiagnosticSample, PbcError> {
    parameters.validate()?;
    let support = ValidatedSupport::new(parameters, support)?;

    let mut layout_time_nanos = 0_u128;
    let mut matching_time_nanos = 0_u128;
    let mut bucket_width_failures = 0_u32;
    let mut no_perfect_schedule_failures = 0_u32;

    for attempt in 0..parameters.max_seed_attempts {
        let seed: PbcLayoutSeed = derive_attempt_seed(root_seed, attempt);
        let layout_start = Instant::now();
        let layout = match PbcPublicLayout::build(parameters, seed, attempt) {
            Ok(layout) => layout,
            Err(PbcError::BucketWidthExceeded) => {
                layout_time_nanos =
                    layout_time_nanos.saturating_add(layout_start.elapsed().as_nanos());
                bucket_width_failures += 1;
                continue;
            }
            Err(error) => return Err(error),
        };
        layout_time_nanos = layout_time_nanos.saturating_add(layout_start.elapsed().as_nanos());

        let matching_start = Instant::now();
        let matching_result = deterministic_matching(&layout, &support);
        matching_time_nanos =
            matching_time_nanos.saturating_add(matching_start.elapsed().as_nanos());
        match matching_result {
            Ok(owners) => {
                // Schedule construction and invariant validation intentionally
                // happen after the matching timer has stopped.
                schedule_from_owners(&layout, &support, owners)?;
                return Ok(PbcDiagnosticSample {
                    attempts: attempt + 1,
                    accepted_attempt: Some(attempt),
                    bucket_width: Some(layout.bucket_width),
                    bucket_width_failures,
                    no_perfect_schedule_failures,
                    layout_time_nanos,
                    matching_time_nanos,
                });
            }
            Err(PbcError::NoPerfectSchedule) => {
                no_perfect_schedule_failures += 1;
            }
            Err(error) => return Err(error),
        }
    }

    Ok(PbcDiagnosticSample {
        attempts: parameters.max_seed_attempts,
        accepted_attempt: None,
        bucket_width: None,
        bucket_width_failures,
        no_perfect_schedule_failures,
        layout_time_nanos,
        matching_time_nanos,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Aggregated layout-generation and cost measurements.
pub struct PbcDiagnosticReport {
    /// Names of reported performance categories.
    pub performance_categories: Vec<String>,
    /// Stable category-keyed values. Timing values are nanoseconds; count
    /// values use the natural units named by the corresponding field below.
    pub performance_values: std::collections::BTreeMap<String, u128>,
    /// Parameter set used by the samples.
    pub parameters: PbcParameters,
    /// Number of recorded trials.
    pub trials: u64,
    /// Number of accepted trials.
    pub accepted_trials: u64,
    /// Number of first-attempt successes.
    pub first_attempt_successes: u64,
    /// Cumulative successes indexed by attempt number.
    pub cumulative_success_by_attempt: Vec<u64>,
    /// Aggregate width-limit failures.
    pub bucket_width_failures: u64,
    /// Aggregate matching failures.
    pub no_perfect_schedule_failures: u64,
    /// Minimum accepted bucket width.
    pub bucket_width_min: Option<usize>,
    /// Median accepted bucket width.
    pub bucket_width_p50: Option<usize>,
    /// 95th-percentile accepted bucket width.
    pub bucket_width_p95: Option<usize>,
    /// Maximum accepted bucket width.
    pub bucket_width_max: Option<usize>,
    /// Number of real cells.
    pub real_cells: u128,
    /// Minimum padded storage size.
    pub padded_storage_min: Option<u128>,
    /// Median padded storage size.
    pub padded_storage_p50: Option<u128>,
    /// 95th-percentile padded storage size.
    pub padded_storage_p95: Option<u128>,
    /// Maximum padded storage size.
    pub padded_storage_max: Option<u128>,
    /// Median padded-to-real storage ratio.
    pub padded_to_real_ratio_p50: Option<f64>,
    /// Aggregate layout time in nanoseconds.
    pub layout_time_nanos: u128,
    /// Aggregate matching time in nanoseconds.
    pub matching_time_nanos: u128,
    /// Number of real selector packages.
    pub selector_real_packages: u128,
    /// Number of dummy selector packages.
    pub selector_dummy_packages: u128,
    /// Total selector packages.
    pub selector_total_packages: u128,
    /// Number of bucket-cell evaluations.
    pub bucket_cell_evaluations: u128,
    /// Number of bucket reductions.
    pub bucket_reductions: u128,
    /// Padding overhead in storage units.
    pub padding_overhead: u128,
}

#[derive(Clone, Debug)]
/// Accumulates opt-in PBC construction diagnostics without ciphertext data.
pub struct PbcDiagnosticAggregator {
    parameters: PbcParameters,
    samples: Vec<PbcDiagnosticSample>,
}

impl PbcDiagnosticAggregator {
    /// Starts an empty report for `parameters`.
    pub fn new(parameters: PbcParameters) -> Self {
        Self { parameters, samples: Vec::new() }
    }

    /// Adds one layout-generation sample to the aggregate.
    pub fn record(&mut self, sample: PbcDiagnosticSample) {
        self.samples.push(sample);
    }

    /// Borrows all recorded samples.
    pub fn samples(&self) -> &[PbcDiagnosticSample] {
        &self.samples
    }

    /// Computes the aggregate report, including percentile storage estimates.
    pub fn finish(&self) -> PbcDiagnosticReport {
        let accepted: Vec<&PbcDiagnosticSample> =
            self.samples.iter().filter(|sample| sample.accepted()).collect();
        let mut widths: Vec<usize> =
            accepted.iter().filter_map(|sample| sample.bucket_width).collect();
        widths.sort_unstable();
        let padded: Vec<u128> = widths
            .iter()
            .map(|&width| self.parameters.bucket_count as u128 * width as u128)
            .collect();
        let accepted_trials = accepted.len() as u64;
        let mut cumulative_success_by_attempt =
            vec![0_u64; self.parameters.max_seed_attempts as usize];
        for sample in &accepted {
            if let Some(attempt) = sample.accepted_attempt {
                for count in cumulative_success_by_attempt.iter_mut().skip(attempt as usize) {
                    *count += 1;
                }
            }
        }
        let real_cells_per_layout = (self.parameters.hash_count as u128)
            .saturating_mul(self.parameters.universe_size as u128);
        let real_cells = real_cells_per_layout.saturating_mul(accepted_trials as u128);
        let selector_dummy_packages =
            (self.parameters.bucket_count as u128).saturating_mul(accepted_trials as u128);
        let selector_real_packages = real_cells;
        let bucket_cell_evaluations: u128 = padded.iter().sum();
        let logical_nonpadding = selector_real_packages.saturating_add(selector_dummy_packages);
        let padding_overhead = bucket_cell_evaluations.saturating_sub(logical_nonpadding);
        let layout_time_nanos: u128 = accepted.iter().map(|sample| sample.layout_time_nanos).sum();
        let matching_time_nanos: u128 =
            accepted.iter().map(|sample| sample.matching_time_nanos).sum();
        let bucket_reductions = self.parameters.bucket_count as u128 * accepted_trials as u128;
        let selector_total_packages =
            selector_real_packages.saturating_add(selector_dummy_packages);
        let mut performance_values = std::collections::BTreeMap::new();
        performance_values.insert("pbc_layout_hashing".to_owned(), layout_time_nanos);
        performance_values.insert("pbc_matching".to_owned(), matching_time_nanos);
        performance_values.insert("pbc_real_selector_packages".to_owned(), selector_real_packages);
        performance_values
            .insert("pbc_dummy_selector_packages".to_owned(), selector_dummy_packages);
        performance_values
            .insert("pbc_bucket_cell_evaluations".to_owned(), bucket_cell_evaluations);
        performance_values.insert("pbc_bucket_reductions".to_owned(), bucket_reductions);
        performance_values.insert("pbc_padding_overhead".to_owned(), padding_overhead);
        PbcDiagnosticReport {
            performance_categories: PERFORMANCE_CATEGORIES
                .iter()
                .map(|category| (*category).to_owned())
                .collect(),
            performance_values,
            parameters: self.parameters.clone(),
            trials: self.samples.len() as u64,
            accepted_trials,
            first_attempt_successes: accepted
                .iter()
                .filter(|sample| sample.accepted_attempt == Some(0))
                .count() as u64,
            cumulative_success_by_attempt,
            bucket_width_failures: self
                .samples
                .iter()
                .map(|sample| sample.bucket_width_failures as u64)
                .sum(),
            no_perfect_schedule_failures: self
                .samples
                .iter()
                .map(|sample| sample.no_perfect_schedule_failures as u64)
                .sum(),
            bucket_width_min: widths.first().copied(),
            bucket_width_p50: percentile(&widths, 50),
            bucket_width_p95: percentile(&widths, 95),
            bucket_width_max: widths.last().copied(),
            real_cells,
            padded_storage_min: padded.first().copied(),
            padded_storage_p50: percentile(&padded, 50),
            padded_storage_p95: percentile(&padded, 95),
            padded_storage_max: padded.last().copied(),
            padded_to_real_ratio_p50: percentile(&padded, 50)
                .map(|value| value as f64 / real_cells_per_layout.max(1) as f64),
            layout_time_nanos,
            matching_time_nanos,
            selector_real_packages,
            selector_dummy_packages,
            selector_total_packages,
            bucket_cell_evaluations,
            bucket_reductions,
            padding_overhead,
        }
    }
}

fn percentile<T: Copy>(values: &[T], percentile: usize) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    // Nearest-rank percentile: p95 of two observations is the larger one.
    let index = ((values.len() * percentile).saturating_add(99) / 100)
        .saturating_sub(1)
        .min(values.len() - 1);
    Some(values[index])
}

impl PbcDiagnosticReport {
    /// Serializes the aggregate report as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Formats the aggregate report as human-readable text.
    pub fn to_text(&self) -> String {
        let optional =
            |value: Option<usize>| value.map_or_else(|| "n/a".to_owned(), |v| v.to_string());
        format!(
            concat!(
                "PBC diagnostic\n",
                "parameters: nu={} h={} c={} k={} max_seed_attempts={} profile={:?}\n",
                "trials: {} accepted={} first_attempt_successes={} cumulative_success_by_attempt={:?}\n",
                "failures: bucket_width={} no_perfect_schedule={}\n",
                "bucket_width: min={} p50={} p95={} max={} padded_storage: min={} p50={} p95={} max={} ratio_p50={:?}\n",
                "cells (accepted-trial totals): real={} selector_real={} selector_dummy={} selector_total={} bucket_cell_evaluations={} bucket_reductions={} padding_overhead={}\n",
                "time_nanos: layout={} matching={}\n",
                "performance_categories: {}\n",
                "performance_values: {:?}"
            ),
            self.parameters.universe_size,
            self.parameters.support_weight,
            self.parameters.hash_count,
            self.parameters.bucket_count,
            self.parameters.max_seed_attempts,
            self.parameters.profile,
            self.trials,
            self.accepted_trials,
            self.first_attempt_successes,
            self.cumulative_success_by_attempt,
            self.bucket_width_failures,
            self.no_perfect_schedule_failures,
            optional(self.bucket_width_min),
            optional(self.bucket_width_p50),
            optional(self.bucket_width_p95),
            optional(self.bucket_width_max),
            self.padded_storage_min.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_storage_p50.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_storage_p95.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_storage_max.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_to_real_ratio_p50,
            self.real_cells,
            self.selector_real_packages,
            self.selector_dummy_packages,
            self.selector_total_packages,
            self.bucket_cell_evaluations,
            self.bucket_reductions,
            self.padding_overhead,
            self.layout_time_nanos,
            self.matching_time_nanos,
            PERFORMANCE_CATEGORIES.join(","),
            self.performance_values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(attempt: Option<u32>, width: Option<usize>) -> PbcDiagnosticSample {
        PbcDiagnosticSample {
            attempts: 3,
            accepted_attempt: attempt,
            bucket_width: width,
            bucket_width_failures: 1,
            no_perfect_schedule_failures: 1,
            layout_time_nanos: 7,
            matching_time_nanos: 11,
        }
    }

    #[test]
    fn aggregation_reports_percentiles_and_overhead() {
        let parameters = PbcParameters::custom(10, 2, 3, 4, 3, None);
        let mut aggregate = PbcDiagnosticAggregator::new(parameters);
        aggregate.record(sample(Some(0), Some(4)));
        aggregate.record(sample(Some(2), Some(6)));
        aggregate.record(sample(None, None));
        let report = aggregate.finish();
        assert_eq!(report.trials, 3);
        assert_eq!(report.accepted_trials, 2);
        assert_eq!(report.first_attempt_successes, 1);
        assert_eq!(report.cumulative_success_by_attempt, vec![1, 1, 2]);
        assert_eq!(report.bucket_width_min, Some(4));
        assert_eq!(report.bucket_width_p50, Some(4));
        assert_eq!(report.bucket_width_p95, Some(6));
        assert_eq!(report.bucket_width_max, Some(6));
        assert_eq!(report.real_cells, report.accepted_trials as u128 * 3 * 10);
        assert_eq!(report.padded_storage_min, Some(16));
        assert_eq!(report.padded_storage_max, Some(24));
        assert_eq!(report.bucket_cell_evaluations, 40);
        assert_eq!(report.bucket_reductions, 8);
        assert_eq!(report.padding_overhead, 0);
        assert_eq!(report.bucket_width_failures, 3);
        assert_eq!(report.no_perfect_schedule_failures, 3);
        assert_eq!(report.performance_values["pbc_layout_hashing"], 14);
        assert_eq!(report.performance_values["pbc_matching"], 22);
        assert_eq!(report.performance_values["pbc_real_selector_packages"], 60);
        assert_eq!(report.performance_values["pbc_dummy_selector_packages"], 8);
    }

    #[test]
    fn text_and_json_use_stable_category_names() {
        let report =
            PbcDiagnosticAggregator::new(PbcParameters::custom(4, 1, 2, 2, 1, None)).finish();
        let text = report.to_text();
        for category in PERFORMANCE_CATEGORIES {
            assert!(text.contains(category));
            assert!(report.to_json().unwrap().contains(category));
        }
        assert!(report.to_json().unwrap().contains("cumulative_success_by_attempt"));
    }

    #[test]
    fn key_layout_measurement_is_deterministic_and_keeps_support_private() {
        let parameters = PbcParameters::custom(10, 2, 3, 4, 3, None);
        let first = measure_key_layout(&parameters, PbcRootSeed([23; 32]), &[1, 7]).unwrap();
        let second = measure_key_layout(&parameters, PbcRootSeed([23; 32]), &[1, 7]).unwrap();
        assert_eq!(first.attempts, second.attempts);
        assert_eq!(first.accepted_attempt, second.accepted_attempt);
        assert_eq!(first.bucket_width, second.bucket_width);
        assert_eq!(first.bucket_width_failures, second.bucket_width_failures);
        assert_eq!(first.no_perfect_schedule_failures, second.no_perfect_schedule_failures);
        assert!(format!("{first:?}").contains("accepted_attempt"));
        assert!(!format!("{first:?}").contains("1, 7"));
    }

    #[test]
    fn key_layout_measurement_rejects_invalid_support_before_reporting() {
        let parameters = PbcParameters::custom(10, 2, 3, 4, 3, None);
        assert!(matches!(
            measure_key_layout(&parameters, PbcRootSeed([24; 32]), &[1]),
            Err(PbcError::SupportSize)
        ));
        assert!(matches!(
            measure_key_layout(&parameters, PbcRootSeed([24; 32]), &[1, 1]),
            Err(PbcError::InvalidSupport)
        ));
    }
}
