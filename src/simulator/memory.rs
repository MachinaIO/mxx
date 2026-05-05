//! Memory diagnostics for heavy simulations and benchmark-estimation flows.

use tracing::info;

/// Logs the current process memory usage when the platform provides it.
pub(crate) fn log_process_memory(label: &str) {
    if let Some(usage) = memory_stats::memory_stats() {
        info!(
            physical_mem_bytes = usage.physical_mem,
            virtual_mem_bytes = usage.virtual_mem,
            "{label}"
        );
    }
}
