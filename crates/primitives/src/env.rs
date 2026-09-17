//! Environment-variable helpers owned by the primitive layer.

/// `MXX_CUDA_STREAM_POOL_SIZE`: number of reusable compute streams owned by
/// each GPU context and device. Default: 32.
pub fn cuda_stream_pool_size() -> usize {
    std::env::var("MXX_CUDA_STREAM_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(32)
}

/// `BLOCK_SIZE`: generic processing block size used in utilities (default: 100).
pub fn block_size() -> usize {
    std::env::var("BLOCK_SIZE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(100)
}

fn positive_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a positive unsigned integer, got {value:?}"))
            .and_then(|parsed| {
                if parsed == 0 { Err(format!("{name} must be positive")) } else { Ok(parsed) }
            }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid UTF-8")),
    }
}

/// `MXX_GPU_VRAM_PERCENT`: percentage of each GPU's total VRAM available to
/// one GPU operation. Default: 80.
pub fn gpu_vram_percent() -> Result<u32, String> {
    let percent = positive_usize("MXX_GPU_VRAM_PERCENT", 80)?;
    if percent <= 100 {
        Ok(percent as u32)
    } else {
        Err(format!("MXX_GPU_VRAM_PERCENT must be between 1 and 100, got {percent}"))
    }
}

/// Maximum number of sampler attempts for each target-column tile.
///
/// This is intentionally fail-closed: malformed or zero values are errors,
/// rather than silently selecting an unbounded retry policy.
pub fn gpu_preimage_max_tile_attempts() -> Result<usize, String> {
    positive_usize("MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS", 64)
}

/// Repetitions of paired compact-operation library measurements. Each sample
/// includes submission and result completion; setup and validation are separate.
#[cfg(all(test, feature = "gpu"))]
pub(crate) fn compact_operation_test_repeats() -> usize {
    positive_usize("MXX_PRIMITIVE_TEST_REPEATS", 3).unwrap()
}

#[cfg(test)]
pub(crate) fn modulus_conversion_test_parameters() -> (u32, usize, usize, u32) {
    // Small unit-test defaults; overrides permit the same production path to
    // exercise larger coefficient vectors and CRT bases without source edits.
    let dimension = positive_usize("MXX_PRIMITIVE_TEST_RING_DIMENSION", 32).unwrap();
    let depth = positive_usize("MXX_PRIMITIVE_TEST_CRT_DEPTH", 4).unwrap();
    let bits = positive_usize("MXX_PRIMITIVE_TEST_CRT_BITS", 30).unwrap();
    let base_bits = positive_usize("MXX_PRIMITIVE_TEST_BASE_BITS", 2).unwrap();
    assert!(dimension >= 16 && dimension.is_power_of_two());
    assert!(depth >= 4);
    (u32::try_from(dimension).unwrap(), depth, bits, u32::try_from(base_bits).unwrap())
}

#[cfg(test)]
mod tests {
    use super::{gpu_vram_percent, positive_usize};

    #[test]
    #[serial_test::serial]
    fn positive_parser_uses_default_only_when_unset() {
        let name = "MXX_TEST_POSITIVE_PARSER";
        unsafe { std::env::remove_var(name) };
        assert_eq!(positive_usize(name, 64).unwrap(), 64);
        unsafe { std::env::set_var(name, "7") };
        assert_eq!(positive_usize(name, 64).unwrap(), 7);
        unsafe { std::env::set_var(name, "0") };
        assert!(positive_usize(name, 64).is_err());
        unsafe { std::env::set_var(name, "-1") };
        assert!(positive_usize(name, 64).is_err());
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    #[serial_test::serial]
    fn gpu_vram_percent_accepts_only_one_through_one_hundred() {
        let name = "MXX_GPU_VRAM_PERCENT";
        unsafe { std::env::remove_var(name) };
        assert_eq!(gpu_vram_percent().unwrap(), 80);
        for value in ["1", "37", "100"] {
            unsafe { std::env::set_var(name, value) };
            assert_eq!(gpu_vram_percent().unwrap(), value.parse().unwrap());
        }
        for value in ["0", "101", "-1", "invalid"] {
            unsafe { std::env::set_var(name, value) };
            assert!(gpu_vram_percent().is_err());
        }
        unsafe { std::env::remove_var(name) };
    }
}

/// Internal re-execution marker for the GPU retirement failure unit test. A
/// quarantined execution intentionally survives until its isolated process exits.
#[cfg(all(test, feature = "gpu"))]
pub(crate) const GPU_RETIREMENT_TEST_CHILD: &str = "MXX_PRIMITIVE_GPU_RETIREMENT_TEST_CHILD";

/// Isolates the prepared-occupancy test's process-wide CUDA pool counters while
/// leaving unrelated tests concurrent in the parent test process.
#[cfg(all(test, feature = "gpu"))]
pub(crate) const GPU_PREPARED_OCCUPANCY_TEST_CHILD: &str =
    "MXX_PRIMITIVE_GPU_PREPARED_OCCUPANCY_TEST_CHILD";

/// Isolates unit tests that assert exact process-global live-context counts or
/// require exactly one live context. The `#[sequential]` and
/// `serial(gpu_context)` guard groups do not exclude each other, so the checks
/// run in a child process while unrelated tests stay concurrent in the parent.
#[cfg(all(test, feature = "gpu"))]
pub(crate) const GPU_CONTEXT_COUNT_TEST_CHILD: &str = "MXX_PRIMITIVE_GPU_CONTEXT_COUNT_TEST_CHILD";
