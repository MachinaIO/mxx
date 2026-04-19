use super::*;
use crate::poly::dcrt::gpu::detected_gpu_device_ids;

pub(crate) fn public_lookup_gpu_device_ids() -> Vec<i32> {
    let device_ids = detected_gpu_device_ids();
    assert!(!device_ids.is_empty(), "at least one GPU device is required for GGH15 public lookup");
    device_ids
}

#[cfg(test)]
pub(crate) fn public_lookup_round_robin_device_slot(
    logical_idx: usize,
    device_count: usize,
) -> usize {
    assert!(device_count > 0, "round-robin GPU device selection requires at least one device");
    logical_idx % device_count
}

pub(super) fn preload_c_b0_by_params<M>(
    params: &<M::P as Poly>::Params,
    c_b0_compact_bytes: &[u8],
) -> Vec<(<<M as PolyMatrix>::P as Poly>::Params, Arc<M>)>
where
    M: PolyMatrix,
{
    let mut c_b0_by_params = Vec::new();

    // Keep preload device set aligned with eval dispatch: eval path uses detected GPUs,
    // while params may still carry a narrower set (e.g. single-GPU default).
    let device_ids = public_lookup_gpu_device_ids();
    for device_id in device_ids {
        let local_params = params.params_for_device(device_id);
        let local_c_b0 = Arc::new(M::from_compact_bytes(&local_params, c_b0_compact_bytes));
        c_b0_by_params.push((local_params, local_c_b0));
    }

    c_b0_by_params
}
