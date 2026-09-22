//! Environment settings owned by the GPU runtime.
//!
//! Primitive settings such as the VRAM percentage and the preimage retry cap
//! remain owned by `mxx-primitives`.  This module only parses the runtime
//! settings, so a runtime can read them once and carry the resulting values
//! through preparation without consulting the process environment again.

use std::{env, num::NonZeroUsize};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuRuntimeOptions {
    pub max_parallel_instances: NonZeroUsize,
    pub measurement_warmups: usize,
    pub measurement_iterations: NonZeroUsize,
    pub release_fence_interval: Option<NonZeroUsize>,
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

impl GpuRuntimeOptions {
    /// Read all runtime settings exactly once.  Callers should retain this
    /// value in the runtime/plan and use it as the frozen preparation policy.
    pub fn from_env() -> Result<Self, GpuRuntimeConfigError> {
        let max_parallel_instances =
            NonZeroUsize::new(positive_usize("MXX_GPU_MAX_PARALLEL_INSTANCES", 64)?)
                .expect("positive_usize rejects zero");
        let measurement_warmups = nonnegative_usize("MXX_GPU_MEASUREMENT_WARMUPS", 1)?;
        let measurement_iterations =
            NonZeroUsize::new(positive_usize("MXX_GPU_MEASUREMENT_ITERATIONS", 2)?)
                .expect("positive_usize rejects zero");
        let release_fence_interval = match env::var_os("MXX_GPU_RELEASE_FENCE_INTERVAL") {
            None => None,
            Some(_) => Some(
                NonZeroUsize::new(positive_usize("MXX_GPU_RELEASE_FENCE_INTERVAL", 1)?)
                    .expect("positive_usize rejects zero"),
            ),
        };
        Ok(Self {
            max_parallel_instances,
            measurement_warmups,
            measurement_iterations,
            release_fence_interval,
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
            "MXX_GPU_MEASUREMENT_WARMUPS",
            "MXX_GPU_MEASUREMENT_ITERATIONS",
            "MXX_GPU_RELEASE_FENCE_INTERVAL",
        ] {
            unsafe { std::env::remove_var(name) };
        }
        let options = GpuRuntimeOptions::from_env().unwrap();
        assert_eq!(options.max_parallel_instances.get(), 64);
        assert_eq!(options.measurement_warmups, 1);
        assert_eq!(options.measurement_iterations.get(), 2);
        assert_eq!(options.release_fence_interval, None);
    }

    #[test]
    #[serial_test::serial]
    fn zero_warmups_are_valid_but_other_counts_are_positive() {
        unsafe {
            std::env::set_var("MXX_GPU_MAX_PARALLEL_INSTANCES", "3");
            std::env::set_var("MXX_GPU_MEASUREMENT_WARMUPS", "0");
            std::env::set_var("MXX_GPU_MEASUREMENT_ITERATIONS", "5");
            std::env::set_var("MXX_GPU_RELEASE_FENCE_INTERVAL", "7");
        }
        let options = GpuRuntimeOptions::from_env().unwrap();
        assert_eq!(options.max_parallel_instances.get(), 3);
        assert_eq!(options.measurement_warmups, 0);
        assert_eq!(options.measurement_iterations.get(), 5);
        assert_eq!(options.release_fence_interval.map(NonZeroUsize::get), Some(7));
        unsafe {
            std::env::remove_var("MXX_GPU_MAX_PARALLEL_INSTANCES");
            std::env::remove_var("MXX_GPU_MEASUREMENT_WARMUPS");
            std::env::remove_var("MXX_GPU_MEASUREMENT_ITERATIONS");
            std::env::remove_var("MXX_GPU_RELEASE_FENCE_INTERVAL");
        }
    }
}
