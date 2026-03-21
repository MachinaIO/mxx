#[cfg(feature = "gpu")]
fn default_gpu_parallelism() -> usize {
    crate::poly::dcrt::gpu::detected_gpu_device_count().max(1)
}

fn validate_positive_parallelism(name: &str, value: usize) -> usize {
    assert!(value > 0, "{name} must be > 0: requested={value}");
    value
}

#[cfg(feature = "gpu")]
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
    let override_parallelism = override_parallelism.map(|value| {
        let value = validate_positive_parallelism("circuit gate parallelism", value);
        #[cfg(feature = "gpu")]
        let value = validate_gpu_parallelism("circuit gate parallelism", value);
        value
    });
    let parsed = std::env::var("MXX_CIRCUIT_PARALLEL_GATES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        let value = override_parallelism.unwrap_or_else(|| {
            let parsed = parsed.unwrap_or_else(default_gpu_parallelism);
            validate_gpu_parallelism("MXX_CIRCUIT_PARALLEL_GATES", parsed)
        });
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
pub fn bgg_poly_encoding_slot_parallelism() -> usize {
    let parsed = std::env::var("BGG_POLY_ENCODING_SLOT_PARALLELISM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0);
    #[cfg(feature = "gpu")]
    {
        let device_count = default_gpu_parallelism();
        let value = parsed.unwrap_or(device_count);
        assert!(
            value <= device_count,
            "BGG_POLY_ENCODING_SLOT_PARALLELISM must be <= available GPU devices: requested={}, devices={}",
            value,
            device_count
        );
        value
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
