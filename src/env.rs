#[cfg(feature = "gpu")]
fn default_gpu_parallelism() -> usize {
    crate::poly::dcrt::gpu::detected_gpu_device_count().max(1)
}

fn validate_positive_parallelism(name: &str, value: usize) -> usize {
    assert!(value > 0, "{name} must be > 0: requested={value}");
    value
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
// This helper is kept for callers that want an explicit GPU-device cap, but
// `resolve_circuit_parallel_gates` intentionally does not call it.
fn validate_gpu_parallelism(name: &str, value: usize) -> usize {
    let device_count = default_gpu_parallelism();
    assert!(
        value <= device_count,
        "{name} must be <= available GPU devices: requested={}, devices={}",
        value,
        device_count
    );
    value
}

/// Environment variable helpers for runtime configuration.

/// `MXX_CIRCUIT_PARALLEL_GATES`: max number of gates processed in parallel per level.
/// GPU feature enabled: default is detected GPU device count and must be <= detected GPU count.
/// Non-GPU: if unset/invalid, evaluator uses full parallelism.
pub fn circuit_parallel_gates() -> Option<usize> {
    resolve_circuit_parallel_gates(None)
}

/// Resolve the effective circuit gate parallelism from an explicit override or the environment.
///
/// When `override_parallelism` is `None`, this preserves the existing
/// `MXX_CIRCUIT_PARALLEL_GATES` behavior.
pub fn resolve_circuit_parallel_gates(override_parallelism: Option<usize>) -> Option<usize> {
    let override_parallelism = override_parallelism
        .map(|value| validate_positive_parallelism("circuit gate parallelism", value));
    let parsed = std::env::var("MXX_CIRCUIT_PARALLEL_GATES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        let value =
            override_parallelism.unwrap_or_else(|| parsed.unwrap_or_else(default_gpu_parallelism));
        Some(value)
    }
    #[cfg(not(feature = "gpu"))]
    {
        override_parallelism.or(parsed)
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

/// `BGG_POLY_ENCODING_SLOT_PARALLELISM`: max number of BGG poly-encoding slots to process in
/// parallel in slot-wise arithmetic / evaluable operations.
/// Default: GPU feature enabled => detected GPU device count, otherwise 30.
/// GPU feature enabled: logical slot parallelism may exceed the detected device count; the GPU
/// poly-encoding path assigns logical slots to detected devices in a round-robin pattern.
pub fn bgg_poly_encoding_slot_parallelism() -> usize {
    let parsed = std::env::var("BGG_POLY_ENCODING_SLOT_PARALLELISM")
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

/// `SLOT_TRANSFER_SLOT_PARALLELISM`: max number of slot auxiliary samples processed in parallel.
/// Default: GPU feature enabled => detected GPU device count, otherwise 30.
pub fn slot_transfer_slot_parallelism() -> usize {
    let parsed = std::env::var("SLOT_TRANSFER_SLOT_PARALLELISM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        let device_count = default_gpu_parallelism();
        parsed.unwrap_or(device_count).min(device_count).max(1)
    }
    #[cfg(not(feature = "gpu"))]
    {
        parsed.unwrap_or(30)
    }
}

/// `AUX_SAMPLING_CHUNK_WIDTH`: column chunk width for chunked auxiliary-sampling decomposition /
/// hash-window assembly in the public-lookup and slot-transfer paths.
/// Default: 30.
pub fn aux_sampling_chunk_width() -> usize {
    std::env::var("AUX_SAMPLING_CHUNK_WIDTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(30)
}

/// `BLOCK_SIZE`: generic processing block size used in utilities (default: 100).
pub fn block_size() -> usize {
    std::env::var("BLOCK_SIZE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(100)
}

/// `LUT_BYTES_LIMIT`: max size of batched lookup tables in bytes (unset = no limit).
pub fn lut_bytes_limit() -> Option<usize> {
    std::env::var("LUT_BYTES_LIMIT").ok().and_then(|s| s.parse::<usize>().ok())
}

/// `LUT_INDEX_SYNC_EVERY`: sync `lookup_tables.index` after this many append operations.
/// Default: 100.
pub fn lut_index_sync_every() -> usize {
    std::env::var("LUT_INDEX_SYNC_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|value| validate_positive_parallelism("LUT_INDEX_SYNC_EVERY", value))
        .unwrap_or(100)
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

#[cfg(test)]
mod tests {
    use super::resolve_circuit_parallel_gates;

    #[test]
    fn resolve_circuit_parallel_gates_preserves_large_override() {
        assert_eq!(resolve_circuit_parallel_gates(Some(usize::MAX)), Some(usize::MAX));
    }
}
