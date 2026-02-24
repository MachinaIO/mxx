#[cfg(feature = "gpu")]
fn default_gpu_parallelism() -> usize {
    crate::poly::dcrt::gpu::detected_gpu_device_count().max(1)
}

/// Environment variable helpers for runtime configuration.

/// `MXX_CIRCUIT_PARALLEL_GATES`: max number of gates processed in parallel per level.
/// GPU feature enabled: default is detected GPU device count and must be <= detected GPU count.
/// Non-GPU: if unset/invalid, evaluator uses full parallelism.
pub fn circuit_parallel_gates() -> Option<usize> {
    let parsed = std::env::var("MXX_CIRCUIT_PARALLEL_GATES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        let device_count = default_gpu_parallelism();
        let value = parsed.unwrap_or(device_count);
        assert!(
            value <= device_count,
            "MXX_CIRCUIT_PARALLEL_GATES must be <= available GPU devices: requested={}, devices={}",
            value,
            device_count
        );
        Some(value)
    }
    #[cfg(not(feature = "gpu"))]
    {
        parsed
    }
}

/// `LUT_PREIMAGE_CHUNK_SIZE`: number of LUT/gate preimages per batch.
/// Default: GPU feature enabled => detected GPU device count, otherwise 30.
pub fn lut_preimage_chunk_size() -> usize {
    let parsed = std::env::var("LUT_PREIMAGE_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        parsed.unwrap_or_else(default_gpu_parallelism)
    }
    #[cfg(not(feature = "gpu"))]
    {
        parsed.unwrap_or(30)
    }
}

/// `GGH15_GATE_PARALLELISM`: max number of gate preimages to process in parallel.
/// Default: GPU feature enabled => detected GPU device count, otherwise 30.
pub fn ggh15_gate_parallelism() -> usize {
    let parsed = std::env::var("GGH15_GATE_PARALLELISM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        parsed.unwrap_or_else(default_gpu_parallelism)
    }
    #[cfg(not(feature = "gpu"))]
    {
        parsed.unwrap_or(30)
    }
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
