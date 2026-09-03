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
