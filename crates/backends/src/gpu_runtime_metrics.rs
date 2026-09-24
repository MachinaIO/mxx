//! Measured direct-Graph candidates and the selected physical plan report.
//! Resource feasibility is established by actual allocation and Graph build;
//! timing values rank only those feasible candidates.

use crate::gpu_execution_plan::GpuExecutionSiteKey;
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
