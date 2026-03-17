use crate::poly::dcrt::gpu::detected_gpu_device_ids;

fn effective_slot_parallelism_gpu_with_requested(num_slots: usize, requested: usize) -> usize {
    requested.min(num_slots).min(detected_gpu_device_ids().len().max(1))
}

pub(super) fn effective_slot_parallelism_gpu(num_slots: usize) -> usize {
    let requested = crate::env::bgg_poly_encoding_slot_parallelism().max(1);
    effective_slot_parallelism_gpu_with_requested(num_slots, requested)
}

#[cfg(test)]
mod tests {
    use super::effective_slot_parallelism_gpu_with_requested;
    use crate::{__PAIR, __TestState, poly::dcrt::gpu::detected_gpu_device_ids};

    #[sequential_test::sequential]
    #[test]
    fn test_effective_slot_parallelism_gpu_uses_detected_gpu_ids() {
        let detected_count = detected_gpu_device_ids().len().max(1);
        let num_slots = detected_count + 3;
        let actual = effective_slot_parallelism_gpu_with_requested(num_slots, detected_count);
        let expected = detected_count.min(num_slots);
        assert_eq!(actual, expected);
    }
}
