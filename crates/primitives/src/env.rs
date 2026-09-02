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

fn optional_positive_usize(name: &str) -> Result<Option<usize>, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a positive unsigned integer, got {value:?}"))
            .and_then(|parsed| {
                if parsed == 0 { Err(format!("{name} must be positive")) } else { Ok(Some(parsed)) }
            }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid UTF-8")),
    }
}

/// Maximum bytes reserved for one compact small-matrix operation.
pub fn gpu_small_matrix_residency_bytes() -> Result<usize, String> {
    positive_usize("MXX_GPU_SMALL_MATRIX_RESIDENCY_BYTES", 1 << 30)
}

/// Fixed reserve for CUDA allocator fragmentation, opaque event resources,
/// and pinned sampler-control storage. This is subtracted from the configured
/// residency budget before admitting any queried owners or workspaces.
pub fn gpu_small_matrix_allocator_headroom_bytes() -> Result<usize, String> {
    positive_usize("MXX_GPU_SMALL_MATRIX_ALLOCATOR_HEADROOM_BYTES", 64 << 20)
}

/// Explicit debug/benchmark override for the automatic small-RHS tile
/// scheduler. `None` means that production code should choose a safe tile
/// from the residency budget and operation shape.
pub fn mul_small_rhs_tile_columns() -> Result<Option<usize>, String> {
    optional_positive_usize("MXX_MUL_SMALL_RHS_TILE_COLUMNS")
}

/// Explicit debug/benchmark override for the automatic small-RHS wave
/// scheduler. `None` means that production code should choose a safe wave
/// from the residency budget and operation shape.
pub fn mul_small_rhs_k_tile() -> Result<Option<usize>, String> {
    optional_positive_usize("MXX_MUL_SMALL_RHS_K_TILE")
}

/// Explicit debug/benchmark override for the automatic small-RHS wave
/// scheduler. `None` means that production code should choose a safe wave
/// from the residency budget and operation shape.
pub fn mul_small_rhs_limb_wave() -> Result<Option<usize>, String> {
    optional_positive_usize("MXX_MUL_SMALL_RHS_LIMB_WAVE")
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
    use super::positive_usize;

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
}
