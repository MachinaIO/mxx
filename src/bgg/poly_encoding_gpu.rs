use crate::poly::PolyParams;
use rayon::prelude::*;

pub(super) fn slot_device_ids<P: PolyParams>(params: &P, slot_parallelism: usize) -> Vec<i32> {
    let device_ids = params.device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for BGG poly-encoding slot processing"
    );
    assert!(
        slot_parallelism <= device_ids.len(),
        "BGG poly-encoding slot parallelism must be <= params GPU count: requested={}, params_gpus={}",
        slot_parallelism,
        device_ids.len()
    );
    device_ids.into_iter().take(slot_parallelism).collect()
}

pub(super) fn effective_slot_parallelism_gpu<P: PolyParams>(
    params: &P,
    num_slots: usize,
    requested: usize,
) -> usize {
    requested.min(num_slots).min(params.device_ids().len().max(1))
}

pub(super) fn map_slots_with_params_gpu<P, T, F>(
    params: &P,
    num_slots: usize,
    slot_parallelism: usize,
    f: F,
) -> Vec<T>
where
    P: PolyParams,
    T: Send,
    F: Fn(usize, &P) -> T + Send + Sync,
{
    let device_ids = slot_device_ids(params, slot_parallelism);
    let mut outputs = Vec::with_capacity(num_slots);
    for slot_start in (0..num_slots).step_by(device_ids.len()) {
        let chunk_len = (slot_start + device_ids.len()).min(num_slots) - slot_start;
        let chunk_device_ids = &device_ids[..chunk_len];
        let mut chunk_outputs = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let local_params = params.params_for_device(chunk_device_ids[offset]);
                f(slot_start + offset, &local_params)
            })
            .collect::<Vec<_>>();
        outputs.append(&mut chunk_outputs);
    }
    outputs
}
