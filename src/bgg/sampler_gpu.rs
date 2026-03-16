use crate::poly::PolyParams;

pub(super) fn effective_slot_parallelism_gpu<P: PolyParams>(params: &P, num_slots: usize) -> usize {
    let requested = crate::env::bgg_poly_encoding_slot_parallelism().max(1);
    requested.min(num_slots).min(params.device_ids().len().max(1))
}
