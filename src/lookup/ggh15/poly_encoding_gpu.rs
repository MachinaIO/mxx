use super::*;
use crate::poly::{PolyParams, dcrt::gpu::detected_gpu_device_ids};

pub(super) fn preload_c_b0_by_slot_by_params_gpu<M>(
    params: &<M::P as Poly>::Params,
    c_b0_compact_bytes_by_slot: &[Vec<u8>],
) -> Vec<(<<M as PolyMatrix>::P as Poly>::Params, Vec<Arc<M>>)>
where
    M: PolyMatrix,
{
    let mut c_b0_by_slot_by_params = Vec::new();

    let device_ids = detected_gpu_device_ids();
    for device_id in device_ids {
        let local_params = params.params_for_device(device_id);
        let local_c_b0_by_slot = c_b0_compact_bytes_by_slot
            .iter()
            .map(|bytes| Arc::new(M::from_compact_bytes(&local_params, bytes)))
            .collect::<Vec<_>>();
        c_b0_by_slot_by_params.push((local_params, local_c_b0_by_slot));
    }

    c_b0_by_slot_by_params
}
