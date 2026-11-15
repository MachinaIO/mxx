//! Environment variables of the core.

/// The number of iterations of an independent loop that host execution runs concurrently, in one
/// batch, with rayon (`MXX_CORE_HOST_LOOP_BATCH`, default 64). Each running iteration holds its own
/// values, so this bounds the peak memory of a loop.
pub fn host_loop_batch() -> usize {
    std::env::var("MXX_CORE_HOST_LOOP_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64)
}
