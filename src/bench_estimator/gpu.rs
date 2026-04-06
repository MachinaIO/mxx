use std::{
    ffi::CStr,
    os::raw::{c_char, c_int},
    time::Duration,
};

use super::{BenchOperationMeasurement, measure_bench_operation, measure_samples_with_interval};

unsafe extern "C" {
    fn gpu_device_count(out_count: *mut c_int) -> c_int;
    fn gpu_device_mem_info(device: c_int, out_free: *mut usize, out_total: *mut usize) -> c_int;
    fn gpu_last_error() -> *const c_char;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuVramUsage {
    pub device_id: i32,
    pub free_bytes: usize,
    pub total_bytes: usize,
}

impl GpuVramUsage {
    pub fn used_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuVramSample {
    pub elapsed: Duration,
    pub usage_by_device: Vec<GpuVramUsage>,
}

fn last_gpu_error_string() -> String {
    unsafe {
        let ptr = gpu_last_error();
        if ptr.is_null() {
            return "unknown GPU error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn sample_gpu_vram_usage(elapsed: Duration) -> Result<GpuVramSample, String> {
    let mut count: c_int = 0;
    let status = unsafe { gpu_device_count(&mut count) };
    if status != 0 {
        return Err(last_gpu_error_string());
    }
    if count < 0 {
        return Err("invalid GPU device count".to_string());
    }

    let mut usage_by_device = Vec::with_capacity(count as usize);
    for device in 0..count {
        let mut free_bytes = 0usize;
        let mut total_bytes = 0usize;
        let status = unsafe { gpu_device_mem_info(device, &mut free_bytes, &mut total_bytes) };
        if status != 0 {
            return Err(last_gpu_error_string());
        }
        usage_by_device.push(GpuVramUsage { device_id: device, free_bytes, total_bytes });
    }

    Ok(GpuVramSample { elapsed, usage_by_device })
}

pub fn measure_gpu_vram_usage<R, F>(
    interval: Duration,
    op: F,
) -> Result<(R, Vec<GpuVramSample>), String>
where
    F: FnOnce() -> R,
{
    measure_samples_with_interval(interval, sample_gpu_vram_usage, op)
}

fn total_used_vram(sample: &GpuVramSample) -> usize {
    sample
        .usage_by_device
        .iter()
        .try_fold(0usize, |total, usage| total.checked_add(usage.used_bytes()))
        .expect("total GPU VRAM usage overflowed usize")
}

fn peak_total_used_vram(samples: &[GpuVramSample]) -> usize {
    samples.iter().map(total_used_vram).max().unwrap_or(0)
}

pub(crate) fn benchmark_gate_operation<R, F>(
    iterations: usize,
    mut op: F,
) -> BenchOperationMeasurement
where
    F: FnMut() -> R,
{
    let (time, samples) = measure_gpu_vram_usage(Duration::from_millis(1), move || {
        measure_bench_operation(iterations, || op())
    })
    .expect("GPU VRAM sampling failed while benchmarking a gate operation");
    BenchOperationMeasurement { time, peak_vram: peak_total_used_vram(&samples) }
}
