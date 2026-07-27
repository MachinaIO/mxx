//! Environment-variable helpers owned by the primitive layer.

/// `MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH`: number of RHS columns processed
/// together by `mul_decompose` and `mul_decompose_small`. Default: 1.
pub fn mul_decompose_column_chunk_width() -> usize {
    std::env::var("MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1)
}

/// `BLOCK_SIZE`: generic processing block size used in utilities (default: 100).
pub fn block_size() -> usize {
    std::env::var("BLOCK_SIZE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(100)
}
