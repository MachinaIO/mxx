use crate::poly::{PolyParams, dcrt::gpu::detected_gpu_device_ids};
use rayon::prelude::*;

pub(super) fn slot_device_ids(slot_parallelism: usize) -> Vec<i32> {
    let device_ids = detected_gpu_device_ids();
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

pub(super) fn effective_slot_parallelism_gpu(num_slots: usize, requested: usize) -> usize {
    requested.min(num_slots).min(detected_gpu_device_ids().len().max(1))
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
    let device_ids = slot_device_ids(slot_parallelism);
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

#[cfg(test)]
mod tests {
    use super::{effective_slot_parallelism_gpu, slot_device_ids};
    use crate::{
        __PAIR, __TestState,
        poly::{
            PolyParams,
            dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
        },
    };

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct FakeParams {
        inner: DCRTPolyParams,
        fake_gpu_ids: Vec<i32>,
    }

    impl PolyParams for FakeParams {
        type Modulus = <DCRTPolyParams as PolyParams>::Modulus;

        fn modulus(&self) -> Self::Modulus {
            self.inner.modulus()
        }

        fn base_bits(&self) -> u32 {
            self.inner.base_bits()
        }

        fn modulus_bits(&self) -> usize {
            self.inner.modulus_bits()
        }

        fn modulus_digits(&self) -> usize {
            self.inner.modulus_digits()
        }

        fn ring_dimension(&self) -> u32 {
            self.inner.ring_dimension()
        }

        fn to_crt(&self) -> (Vec<u64>, usize, usize) {
            self.inner.to_crt()
        }

        fn device_ids(&self) -> Vec<i32> {
            self.fake_gpu_ids.clone()
        }

        fn params_for_device(&self, _device_id: i32) -> Self {
            self.clone()
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_slot_device_ids_uses_detected_gpu_ids() {
        let detected_gpu_ids = detected_gpu_device_ids();

        if detected_gpu_ids.is_empty() {
            let panic = std::panic::catch_unwind(|| slot_device_ids(1))
                .expect_err("without detected GPUs the helper should reject slot processing");
            let panic_msg = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .expect("panic payload should be a string");
            assert!(panic_msg.contains("at least one GPU device is required"));
            return;
        }

        let slot_parallelism = detected_gpu_ids.len().min(2);
        assert_eq!(
            slot_device_ids(slot_parallelism),
            detected_gpu_ids[..slot_parallelism].to_vec()
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_effective_slot_parallelism_gpu_uses_detected_gpu_ids() {
        let detected_count = detected_gpu_device_ids().len().max(1);
        let params = FakeParams {
            inner: DCRTPolyParams::default(),
            fake_gpu_ids: if detected_count > 1 { vec![901] } else { vec![901, 902] },
        };
        let requested = detected_count;
        let num_slots = detected_count + 3;
        let actual = effective_slot_parallelism_gpu(num_slots, requested);
        assert_eq!(actual, requested.min(num_slots));
        if detected_count > 1 {
            assert_ne!(actual, params.device_ids().len().min(requested).min(num_slots));
        }
    }
}
