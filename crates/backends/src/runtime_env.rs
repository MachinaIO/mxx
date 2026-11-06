//! GPU runtime settings read once during preparation; see `crate::gpu_runtime` for the table of
//! options and their environment variables.

use num_bigint::BigInt;
use std::{collections::BTreeMap, env, num::NonZeroUsize, ops::RangeInclusive};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuRuntimeOptions {
    pub max_parallel_instances: NonZeroUsize,
    /// Optional upper bound for the descending column-tile search. Unset
    /// starts at the widest concrete matrix, including intermediates.
    pub max_columns_per_job: Option<NonZeroUsize>,
    pub measurement_warmups: usize,
    pub measurement_iterations: NonZeroUsize,
    pub release_fence_interval: Option<NonZeroUsize>,
    /// Caller-guaranteed closed ranges for named Int inputs, including every
    /// member of an Int family. The plan rejects unknown or noninteger names.
    pub integer_input_ranges: BTreeMap<String, RangeInclusive<BigInt>>,
    /// Caller-guaranteed maximum canonical payload bytes for Int/TypedBlob artifacts.
    /// A key with no index covers every member of that family. This avoids enumerating
    /// metadata for huge families. Imports reject payloads exceeding their capacity.
    pub artifact_payload_size_bounds: BTreeMap<crate::artifact::ArtifactKey, usize>,
    /// Subgraphs the planner executes with their registered native kernel
    /// instead of their body; empty runs every subgraph from its body.
    pub subgraph_kernels: Vec<crate::gpu_subgraph_kernel::GpuSubgraphKernel>,
    /// After selecting a candidate, time every graph node's operations
    /// separately and report each node's predicted share of the plan's time.
    pub profile_nodes: bool,
    /// Enable an artifact I/O trial even for a GPU-resident store. Trials run
    /// one cold and one steady wave or host-loop iteration (or the only one
    /// for a count-one group).
    /// Host and file stores always run the trial. Trial exports are discarded.
    pub io_trial: bool,
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum GpuRuntimeConfigError {
    #[error("invalid GPU setting: {0}")]
    Invalid(String),
}

fn positive_usize(name: &str, default: usize) -> Result<usize, GpuRuntimeConfigError> {
    let Some(value) = env::var_os(name) else {
        return Ok(default);
    };
    let value = value
        .into_string()
        .map_err(|_| GpuRuntimeConfigError::Invalid(format!("{name} is not valid UTF-8")))?;
    let parsed = value.parse::<usize>().map_err(|_| {
        GpuRuntimeConfigError::Invalid(format!(
            "{name} must be a positive unsigned integer, got {value:?}"
        ))
    })?;
    if parsed == 0 {
        return Err(GpuRuntimeConfigError::Invalid(format!("{name} must be positive")));
    }
    Ok(parsed)
}

fn nonnegative_usize(name: &str, default: usize) -> Result<usize, GpuRuntimeConfigError> {
    let Some(value) = env::var_os(name) else {
        return Ok(default);
    };
    let value = value
        .into_string()
        .map_err(|_| GpuRuntimeConfigError::Invalid(format!("{name} is not valid UTF-8")))?;
    value.parse::<usize>().map_err(|_| {
        GpuRuntimeConfigError::Invalid(format!(
            "{name} must be a non-negative unsigned integer, got {value:?}"
        ))
    })
}

fn flag(name: &str) -> Result<bool, GpuRuntimeConfigError> {
    match env::var(name).as_deref() {
        Err(env::VarError::NotPresent) | Ok("0" | "false") => Ok(false),
        Ok("1" | "true") => Ok(true),
        _ => Err(GpuRuntimeConfigError::Invalid(format!("{name} must be 0, 1, false, or true"))),
    }
}

impl GpuRuntimeOptions {
    /// Read all runtime settings exactly once.  Callers should retain this
    /// value in the runtime/plan and use it as the frozen preparation policy.
    pub fn from_env() -> Result<Self, GpuRuntimeConfigError> {
        let max_parallel_instances =
            NonZeroUsize::new(positive_usize("MXX_GPU_MAX_PARALLEL_INSTANCES", 1)?)
                .expect("positive_usize rejects zero");
        let max_columns_per_job = match env::var_os("MXX_GPU_MAX_COLUMNS_PER_JOB") {
            None => None,
            Some(_) => Some(
                NonZeroUsize::new(positive_usize("MXX_GPU_MAX_COLUMNS_PER_JOB", 1)?)
                    .expect("positive_usize rejects zero"),
            ),
        };
        let measurement_warmups = nonnegative_usize("MXX_GPU_MEASUREMENT_WARMUPS", 0)?;
        let measurement_iterations =
            NonZeroUsize::new(positive_usize("MXX_GPU_MEASUREMENT_ITERATIONS", 1)?)
                .expect("positive_usize rejects zero");
        let release_fence_interval = match env::var_os("MXX_GPU_RELEASE_FENCE_INTERVAL") {
            None => None,
            Some(_) => Some(
                NonZeroUsize::new(positive_usize("MXX_GPU_RELEASE_FENCE_INTERVAL", 1)?)
                    .expect("positive_usize rejects zero"),
            ),
        };
        let profile_nodes = flag("MXX_GPU_PROFILE_NODES")?;
        let io_trial = flag("MXX_GPU_IO_TRIAL")?;
        Ok(Self {
            max_parallel_instances,
            max_columns_per_job,
            measurement_warmups,
            measurement_iterations,
            release_fence_interval,
            integer_input_ranges: BTreeMap::new(),
            artifact_payload_size_bounds: BTreeMap::new(),
            subgraph_kernels: Vec::new(),
            profile_nodes,
            io_trial,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::GpuRuntimeOptions;
    use std::num::NonZeroUsize;

    #[test]
    #[serial_test::serial]
    fn runtime_defaults_match_the_preparation_contract() {
        for name in [
            "MXX_GPU_MAX_PARALLEL_INSTANCES",
            "MXX_GPU_MAX_COLUMNS_PER_JOB",
            "MXX_GPU_MEASUREMENT_WARMUPS",
            "MXX_GPU_MEASUREMENT_ITERATIONS",
            "MXX_GPU_RELEASE_FENCE_INTERVAL",
            "MXX_GPU_PROFILE_NODES",
            "MXX_GPU_IO_TRIAL",
        ] {
            unsafe { std::env::remove_var(name) };
        }
        let options = GpuRuntimeOptions::from_env().unwrap();
        assert_eq!(options.max_parallel_instances.get(), 1);
        assert_eq!(options.max_columns_per_job, None);
        assert_eq!(options.measurement_warmups, 0);
        assert_eq!(options.measurement_iterations.get(), 1);
        assert_eq!(options.release_fence_interval, None);
        assert!(!options.profile_nodes);
        assert!(!options.io_trial);
    }

    #[test]
    #[serial_test::serial]
    fn zero_warmups_are_valid_but_other_counts_are_positive() {
        unsafe {
            std::env::set_var("MXX_GPU_MAX_PARALLEL_INSTANCES", "3");
            std::env::set_var("MXX_GPU_MAX_COLUMNS_PER_JOB", "5");
            std::env::set_var("MXX_GPU_MEASUREMENT_WARMUPS", "0");
            std::env::set_var("MXX_GPU_MEASUREMENT_ITERATIONS", "5");
            std::env::set_var("MXX_GPU_RELEASE_FENCE_INTERVAL", "7");
            std::env::set_var("MXX_GPU_PROFILE_NODES", "1");
            std::env::set_var("MXX_GPU_IO_TRIAL", "1");
        }
        let options = GpuRuntimeOptions::from_env().unwrap();
        assert_eq!(options.max_parallel_instances.get(), 3);
        assert_eq!(options.max_columns_per_job.map(NonZeroUsize::get), Some(5));
        assert_eq!(options.measurement_warmups, 0);
        assert_eq!(options.measurement_iterations.get(), 5);
        assert_eq!(options.release_fence_interval.map(NonZeroUsize::get), Some(7));
        assert!(options.profile_nodes);
        assert!(options.io_trial);
        unsafe {
            std::env::remove_var("MXX_GPU_MAX_PARALLEL_INSTANCES");
            std::env::remove_var("MXX_GPU_MAX_COLUMNS_PER_JOB");
            std::env::remove_var("MXX_GPU_MEASUREMENT_WARMUPS");
            std::env::remove_var("MXX_GPU_MEASUREMENT_ITERATIONS");
            std::env::remove_var("MXX_GPU_RELEASE_FENCE_INTERVAL");
            std::env::remove_var("MXX_GPU_PROFILE_NODES");
            std::env::remove_var("MXX_GPU_IO_TRIAL");
        }
    }

    #[test]
    #[serial_test::serial]
    fn column_search_cap_rejects_invalid_values() {
        for value in ["0", "-1", "abc", "184467440737095516160"] {
            unsafe { std::env::set_var("MXX_GPU_MAX_COLUMNS_PER_JOB", value) };
            let result = GpuRuntimeOptions::from_env();
            unsafe { std::env::remove_var("MXX_GPU_MAX_COLUMNS_PER_JOB") };
            assert!(result.unwrap_err().to_string().contains("MXX_GPU_MAX_COLUMNS_PER_JOB"));
        }
    }
}
