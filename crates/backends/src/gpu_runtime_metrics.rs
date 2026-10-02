//! Measured direct-Graph candidates and the selected physical plan report.
//! Resource feasibility is established by actual allocation and Graph build;
//! timing values rank only those feasible candidates.

use crate::gpu_execution_plan::GpuExecutionSiteKey;
use mxx_ir_core::visualize::NodeCost;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuStageReport {
    pub wave_instances: usize,
    pub columns_per_job: Vec<usize>,
    pub predicted_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuWarmupReport {
    pub predicted_seconds: f64,
    pub limiting_stage: Option<GpuExecutionSiteKey>,
    pub stages: Vec<GpuStageReport>,
    pub reason: String,
    /// Each graph node's predicted seconds per execute, measured only when
    /// `GpuRuntimeOptions::profile_nodes` is set; empty otherwise, and in a
    /// report saved before node costs existed.
    #[serde(default)]
    pub node_costs: Vec<NodeCost>,
    /// Predicted seconds per execute including artifact reads and writes,
    /// measured by `plan_with_store` for a host or file store, or for any
    /// store when `GpuRuntimeOptions::io_trial_waves` is set: the
    /// I/O trial's wall time, with each root wave group and host-driven loop
    /// extrapolated from its last measured wave or iteration. `None` without
    /// a trial or for a plan without artifact I/O.
    #[serde(default)]
    pub io_predicted_seconds: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct GpuMeasuredCostCache {
    points: BTreeMap<(usize, usize), f64>,
}

impl GpuMeasuredCostCache {
    pub fn insert(&mut self, wave_instances: usize, columns_per_job: usize, seconds: f64) {
        self.points.insert((wave_instances, columns_per_job), seconds);
    }

    pub fn exact(&self, wave_instances: usize, columns_per_job: usize) -> Option<f64> {
        self.points.get(&(wave_instances, columns_per_job)).copied()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::GpuWarmupReport;

    #[test]
    fn a_report_saved_without_node_costs_still_loads() {
        let report: GpuWarmupReport = serde_json::from_str(
            r#"{"predicted_seconds":1.5,"limiting_stage":null,"stages":[],"reason":"saved"}"#,
        )
        .unwrap();
        assert_eq!(report.predicted_seconds, 1.5);
        assert!(report.node_costs.is_empty());
    }
}
