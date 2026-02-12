/// Environment variable helpers for runtime configuration.

/// `MXX_CIRCUIT_PARALLEL_GATES`: max number of gates processed in parallel per level.
/// If unset or invalid, the evaluator uses full parallelism.
pub fn circuit_parallel_gates() -> Option<usize> {
    std::env::var("MXX_CIRCUIT_PARALLEL_GATES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// `LUT_PREIMAGE_CHUNK_SIZE`: number of LUT/gate preimages per batch (default: 80).
pub fn lut_preimage_chunk_size() -> usize {
    std::env::var("LUT_PREIMAGE_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50)
}

/// `BLOCK_SIZE`: generic processing block size used in utilities (default: 100).
pub fn block_size() -> usize {
    std::env::var("BLOCK_SIZE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(100)
}

/// `LUT_BYTES_LIMIT`: max size of batched lookup tables in bytes (unset = no limit).
pub fn lut_bytes_limit() -> Option<usize> {
    std::env::var("LUT_BYTES_LIMIT").ok().and_then(|s| s.parse::<usize>().ok())
}

const DEFAULT_WEE25_TOPJ_BATCH: usize = 200;
const DEFAULT_WEE25_COMMIT_CACHE_PERSIST_BATCH: usize = 300;

/// `WEE25_TOPJ_PARALLEL_BATCH`: block batch size for top_j generation (default: 200).
pub fn wee25_topj_parallel_batch() -> usize {
    std::env::var("WEE25_TOPJ_PARALLEL_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_WEE25_TOPJ_BATCH)
}

/// `WEE25_COMMIT_CACHE_PERSIST_BATCH`: number of commit-cache nodes buffered before persisting
/// (default: 1000).
pub fn wee25_commit_cache_persist_batch() -> usize {
    std::env::var("WEE25_COMMIT_CACHE_PERSIST_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_WEE25_COMMIT_CACHE_PERSIST_BATCH)
}
