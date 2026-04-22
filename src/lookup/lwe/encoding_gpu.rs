use super::*;
use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    lookup::{
        PublicLut,
        lwe::{derive_k_low_chunk, k_high_chunk_count, read_k_high_chunk},
    },
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
};
use rayon::prelude::*;
use std::{path::Path, sync::Arc, time::Instant};
use tracing::{debug, info};

pub(crate) fn public_lookup_gpu_device_ids() -> Vec<i32> {
    let device_ids = detected_gpu_device_ids();
    if device_ids.is_empty() { vec![0] } else { device_ids }
}

#[allow(dead_code)]
pub(crate) fn public_lookup_round_robin_device_slot(
    logical_idx: usize,
    device_count: usize,
) -> usize {
    assert!(device_count > 0, "device_count must be positive");
    logical_idx % device_count
}

struct LwePublicLookupSharedByDevice<M: PolyMatrix> {
    device_id: i32,
    params: <<M as PolyMatrix>::P as Poly>::Params,
    c_b: M,
}

#[derive(Clone, Copy, Debug)]
enum ContributionFamily {
    KHigh,
    KLow,
}

#[derive(Clone, Copy, Debug)]
struct LweContributionTask {
    chunk_idx: usize,
    family: ContributionFamily,
}

struct LoadedLweContributionTask<M: PolyMatrix> {
    task: LweContributionTask,
    device_slot: usize,
    rhs_chunk: Option<M>,
}

struct ComputedLweContributionTask {
    task: LweContributionTask,
    bytes: Arc<[u8]>,
}

impl<M> Clone for LoadedLweContributionTask<M>
where
    M: PolyMatrix + Clone,
{
    fn clone(&self) -> Self {
        Self { task: self.task, device_slot: self.device_slot, rhs_chunk: self.rhs_chunk.clone() }
    }
}

fn prepare_shared_by_device<M>(
    params: &<M::P as Poly>::Params,
    c_b_compact_bytes: &[u8],
) -> Vec<LwePublicLookupSharedByDevice<M>>
where
    M: PolyMatrix + Send + Sync + 'static,
{
    public_lookup_gpu_device_ids()
        .into_par_iter()
        .map(|device_id| {
            let local_params = if params.device_ids().first().copied() == Some(device_id) {
                params.clone()
            } else {
                params.params_for_device(device_id)
            };
            let c_b = M::from_compact_bytes(&local_params, c_b_compact_bytes);
            LwePublicLookupSharedByDevice { device_id, params: local_params, c_b }
        })
        .collect()
}

fn load_input_vector_by_device<M>(
    shared: &[LwePublicLookupSharedByDevice<M>],
    input_vector: &M,
) -> Vec<M>
where
    M: PolyMatrix + Send + Sync + 'static,
{
    let input_vector_bytes = Arc::<[u8]>::from(input_vector.to_compact_bytes());
    shared
        .par_iter()
        .map(|shared_dev| M::from_compact_bytes(&shared_dev.params, input_vector_bytes.as_ref()))
        .collect()
}

fn load_wave<M>(
    shared: &[LwePublicLookupSharedByDevice<M>],
    tasks: &[LweContributionTask],
    dir: &Path,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
) -> Vec<LoadedLweContributionTask<M>>
where
    M: PolyMatrix + Send + Sync + 'static,
{
    tasks
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(device_slot, task)| {
            let rhs_chunk = match task.family {
                ContributionFamily::KHigh => Some(read_k_high_chunk::<M>(
                    &shared[device_slot].params,
                    dir,
                    gate_id,
                    lut_id,
                    row_size,
                    lut_entry_idx,
                    task.chunk_idx,
                )),
                ContributionFamily::KLow => None,
            };
            LoadedLweContributionTask { task, device_slot, rhs_chunk }
        })
        .collect()
}

fn compute_wave<M, SH>(
    hash_key: [u8; 32],
    shared: &[LwePublicLookupSharedByDevice<M>],
    input_vector_by_device: &[M],
    loaded_wave: Vec<LoadedLweContributionTask<M>>,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
) -> Vec<ComputedLweContributionTask>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    loaded_wave
        .into_par_iter()
        .map(|loaded| {
            let shared_dev = &shared[loaded.device_slot];
            let rhs_chunk = match loaded.task.family {
                ContributionFamily::KHigh => {
                    loaded.rhs_chunk.expect("k_high contribution must load a checkpoint rhs chunk")
                }
                ContributionFamily::KLow => derive_k_low_chunk::<M, SH>(
                    &shared_dev.params,
                    row_size,
                    hash_key,
                    gate_id,
                    lut_id,
                    lut_entry_idx,
                    loaded.task.chunk_idx,
                ),
            };
            let lhs = match loaded.task.family {
                ContributionFamily::KHigh => &shared_dev.c_b,
                ContributionFamily::KLow => &input_vector_by_device[loaded.device_slot],
            };
            let output = lhs * &rhs_chunk;
            ComputedLweContributionTask {
                task: loaded.task,
                bytes: Arc::<[u8]>::from(output.into_compact_bytes()),
            }
        })
        .collect()
}

fn reduce_contribution_bytes_by_chunk<M>(
    reduction_params: &<M::P as Poly>::Params,
    contribution_bytes_by_chunk: Vec<Vec<Arc<[u8]>>>,
) -> M
where
    M: PolyMatrix + Send + Sync + 'static,
{
    let contribution_count = contribution_bytes_by_chunk
        .first()
        .map(Vec::len)
        .expect("LWE GPU public lookup requires at least one output chunk");
    assert!(contribution_count > 0, "LWE GPU public lookup requires at least one contribution");

    let full_contributions = (0..contribution_count)
        .into_par_iter()
        .map(|contribution_idx| {
            let mut chunk_iter = contribution_bytes_by_chunk.iter().map(|contribution_bytes| {
                M::from_compact_bytes(
                    reduction_params,
                    contribution_bytes[contribution_idx].as_ref(),
                )
            });
            let first_chunk = chunk_iter.next().expect("validated non-empty contribution chunks");
            first_chunk.concat_columns_owned(chunk_iter.collect())
        })
        .collect::<Vec<_>>();

    let mut full_contributions_iter = full_contributions.into_iter();
    let mut accum = full_contributions_iter.next().expect("validated non-empty contribution list");
    for contribution in full_contributions_iter {
        accum.add_in_place(&contribution);
    }
    accum
}

pub(crate) fn public_lookup_slot_gpu<M, SH>(
    params: &<BggEncoding<M> as Evaluable>::Params,
    plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
    dir_path: &Path,
    hash_key: [u8; 32],
    row_size: usize,
    c_b_compact_bytes: &[u8],
    input_vector: &M,
    input_plaintext: &M::P,
    gate_id: GateId,
    lut_id: usize,
) -> BggEncoding<M>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let start = Instant::now();
    let z_u64 = input_plaintext.const_coeff_u64();
    let (k, y_k) = plt
        .get(params, z_u64)
        .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z_u64));
    let lut_entry_idx = usize::try_from(k).expect("LUT row index must fit in usize");
    let y_k_poly = M::P::from_elem_to_constant(params, &y_k);
    let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, hash_key, gate_id);
    let pubkey = BggPublicKey::new(a_lt, true);

    let shared = prepare_shared_by_device::<M>(params, c_b_compact_bytes);
    let input_vector_by_device = load_input_vector_by_device(&shared, input_vector);
    let output_chunk_count = k_high_chunk_count::<M>(params, row_size);
    let wave_width = shared.len().max(1);
    let mut tasks = Vec::with_capacity(output_chunk_count * 2);
    for chunk_idx in 0..output_chunk_count {
        tasks.push(LweContributionTask { chunk_idx, family: ContributionFamily::KHigh });
        tasks.push(LweContributionTask { chunk_idx, family: ContributionFamily::KLow });
    }
    info!(
        "LWE GPU public lookup start: gate_id={}, lut_id={}, lut_entry_idx={}, chunks={}, devices={}, tasks={}",
        gate_id,
        lut_id,
        lut_entry_idx,
        output_chunk_count,
        shared.len(),
        tasks.len()
    );
    for (slot_idx, shared_dev) in shared.iter().enumerate() {
        debug!(
            "LWE GPU public lookup device slot prepared: slot={}, device_id={}",
            slot_idx, shared_dev.device_id
        );
    }

    let mut contribution_bytes_by_chunk = vec![Vec::<Arc<[u8]>>::new(); output_chunk_count];
    let mut cursor = 0usize;
    let mut current_loaded = load_wave::<M>(
        &shared,
        &tasks[cursor..(cursor + wave_width).min(tasks.len())],
        dir_path,
        gate_id,
        lut_id,
        row_size,
        lut_entry_idx,
    );
    cursor += current_loaded.len();
    let mut completed_task_count = 0usize;

    while !current_loaded.is_empty() {
        let current_loaded_owned = current_loaded;
        let has_next = cursor < tasks.len();
        let (computed_wave, next_loaded) = if has_next {
            let next_end = (cursor + wave_width).min(tasks.len());
            let next_tasks = tasks[cursor..next_end].to_vec();
            let (computed_wave, next_loaded) = rayon::join(
                || {
                    compute_wave::<M, SH>(
                        hash_key,
                        &shared,
                        &input_vector_by_device,
                        current_loaded_owned,
                        gate_id,
                        lut_id,
                        row_size,
                        lut_entry_idx,
                    )
                },
                || {
                    load_wave::<M>(
                        &shared,
                        &next_tasks,
                        dir_path,
                        gate_id,
                        lut_id,
                        row_size,
                        lut_entry_idx,
                    )
                },
            );
            cursor = next_end;
            (computed_wave, next_loaded)
        } else {
            (
                compute_wave::<M, SH>(
                    hash_key,
                    &shared,
                    &input_vector_by_device,
                    current_loaded_owned,
                    gate_id,
                    lut_id,
                    row_size,
                    lut_entry_idx,
                ),
                Vec::new(),
            )
        };

        for computed in computed_wave {
            contribution_bytes_by_chunk[computed.task.chunk_idx].push(computed.bytes);
            completed_task_count += 1;
        }
        info!(
            "LWE GPU public lookup progress: {}/{} tasks ({:.1}%)",
            completed_task_count,
            tasks.len(),
            100.0 * (completed_task_count as f64) / (tasks.len() as f64)
        );
        current_loaded = next_loaded;
    }

    let vector =
        reduce_contribution_bytes_by_chunk::<M>(&shared[0].params, contribution_bytes_by_chunk);

    info!(
        "LWE GPU public lookup finished: gate_id={}, lut_id={}, lut_entry_idx={}, elapsed={:?}",
        gate_id,
        lut_id,
        lut_entry_idx,
        start.elapsed()
    );
    BggEncoding::new(vector, pubkey, Some(y_k_poly))
}
