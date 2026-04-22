use super::*;
use crate::{
    bench_estimator::{PublicLutSampleAuxBenchEstimator, SampleAuxBenchEstimate},
    bgg::public_key::BggPublicKey,
    element::PolyElem,
    lookup::lwe::{
        column_chunk_bounds, column_chunk_id_prefix, derive_k_low_chunk, k_high_chunk_count,
        k_high_row_checkpoint_prefix,
    },
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
    sampler::trapdoor::GpuPreimageRequest,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use tracing::{debug, info};

enum DeviceReplica<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> DeviceReplica<'_, T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<T> Deref for DeviceReplica<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

struct GpuLweDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: <<M as PolyMatrix>::P as Poly>::Params,
    trapdoor: DeviceReplica<'a, T>,
    pub_matrix: DeviceReplica<'a, M>,
    gadget: M,
}

#[derive(Clone)]
struct GpuLweChunkTask<P: Poly> {
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    x_k_usize: usize,
    chunk_idx: usize,
    input_pubkey_bytes: Arc<[u8]>,
    output_pubkey_bytes: Arc<[u8]>,
    y_elem: P::Elem,
}

struct LoadedLweChunkTask<M>
where
    M: PolyMatrix,
{
    device_slot: usize,
    task: GpuLweChunkTask<M::P>,
    a_z: M,
    a_lt: M,
    y_poly: M::P,
}

struct ComputedLweChunkTask<M>
where
    M: PolyMatrix,
{
    task: GpuLweChunkTask<M::P>,
    preimage_chunk: M,
}

fn source_device_first(mut device_ids: Vec<i32>, source_device_id: i32) -> Vec<i32> {
    if let Some(source_pos) = device_ids.iter().position(|&device_id| device_id == source_device_id)
    {
        device_ids.swap(0, source_pos);
    }
    device_ids
}

fn prepare_gpu_device_shared<'a, M, SH, ST>(
    evaluator: &'a LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    params: &<M::P as Poly>::Params,
    row_size: usize,
) -> Vec<GpuLweDeviceShared<'a, M, ST::Trapdoor>>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    let source_device_id = params
        .device_ids()
        .into_iter()
        .next()
        .unwrap_or_else(|| detected_gpu_device_ids().into_iter().next().unwrap_or(0));
    let detected = detected_gpu_device_ids();
    let device_ids = if detected.is_empty() {
        params.device_ids()
    } else {
        source_device_first(detected, source_device_id)
    };
    let needs_cross_device_copy = device_ids.iter().any(|&device_id| device_id != source_device_id);
    let trapdoor_bytes =
        needs_cross_device_copy.then(|| ST::trapdoor_to_bytes(evaluator.trapdoor.as_ref()));
    let pub_matrix_bytes =
        needs_cross_device_copy.then(|| evaluator.pub_matrix.as_ref().to_compact_bytes());

    device_ids
        .into_iter()
        .map(|device_id| {
            let local_params = if device_id == source_device_id {
                params.clone()
            } else {
                params.params_for_device(device_id)
            };
            let (local_trapdoor, local_pub_matrix) = if device_id == source_device_id {
                (
                    DeviceReplica::Borrowed(evaluator.trapdoor.as_ref()),
                    DeviceReplica::Borrowed(evaluator.pub_matrix.as_ref()),
                )
            } else {
                let trapdoor = ST::trapdoor_from_bytes(
                    &local_params,
                    trapdoor_bytes
                        .as_ref()
                        .expect("cross-device trapdoor bytes must exist for LWE GPU sampling"),
                )
                .expect("failed to deserialize LWE trapdoor replica");
                let pub_matrix = M::from_compact_bytes(
                    &local_params,
                    pub_matrix_bytes
                        .as_ref()
                        .expect("cross-device pub_matrix bytes must exist for LWE GPU sampling"),
                );
                (DeviceReplica::Owned(trapdoor), DeviceReplica::Owned(pub_matrix))
            };
            let gadget = M::gadget_matrix(&local_params, row_size);
            GpuLweDeviceShared {
                params: local_params,
                trapdoor: local_trapdoor,
                pub_matrix: local_pub_matrix,
                gadget,
            }
        })
        .collect()
}

fn load_wave<M, ST>(
    shared: &[GpuLweDeviceShared<'_, M, ST::Trapdoor>],
    tasks: &[GpuLweChunkTask<M::P>],
) -> Vec<LoadedLweChunkTask<M>>
where
    M: PolyMatrix + Send + Sync + 'static,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    tasks
        .iter()
        .cloned()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(device_slot, task)| {
            let shared_dev = &shared[device_slot];
            let a_z = M::from_compact_bytes(&shared_dev.params, task.input_pubkey_bytes.as_ref());
            let a_lt = M::from_compact_bytes(&shared_dev.params, task.output_pubkey_bytes.as_ref());
            let y_poly = M::P::from_elem_to_constant(&shared_dev.params, &task.y_elem);
            LoadedLweChunkTask { device_slot, task, a_z, a_lt, y_poly }
        })
        .collect()
}

fn build_adjusted_target_chunk<M, SH, ST>(
    evaluator: &LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    shared_dev: &GpuLweDeviceShared<'_, M, ST::Trapdoor>,
    loaded: &LoadedLweChunkTask<M>,
    row_size: usize,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    let x_poly = M::P::from_usize_to_constant(&shared_dev.params, loaded.task.x_k_usize);
    let ext_matrix = loaded.a_z.clone() - &(shared_dev.gadget.clone() * x_poly);
    let total_cols = row_size * shared_dev.params.modulus_digits();
    let (col_start, col_len) = column_chunk_bounds(total_cols, loaded.task.chunk_idx);
    let gadget_chunk = shared_dev.gadget.slice_columns(col_start, col_start + col_len);
    let target_chunk = loaded.a_lt.slice_columns(col_start, col_start + col_len) -
        &(gadget_chunk * loaded.y_poly.clone());
    let k_low_chunk = derive_k_low_chunk::<M, SH>(
        &shared_dev.params,
        row_size,
        evaluator.hash_key,
        loaded.task.gate_id,
        loaded.task.lut_id,
        loaded.task.lut_entry_idx,
        loaded.task.chunk_idx,
    );
    target_chunk - &(ext_matrix * &k_low_chunk)
}

fn compute_wave<M, SH, ST>(
    evaluator: &LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    shared: &[GpuLweDeviceShared<'_, M, ST::Trapdoor>],
    loaded_wave: Vec<LoadedLweChunkTask<M>>,
    row_size: usize,
) -> Vec<ComputedLweChunkTask<M>>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    if loaded_wave.is_empty() {
        return Vec::new();
    }

    let requests = loaded_wave
        .iter()
        .enumerate()
        .map(|(request_idx, loaded)| {
            let shared_dev = &shared[loaded.device_slot];
            GpuPreimageRequest {
                entry_idx: request_idx,
                params: &shared_dev.params,
                trapdoor: shared_dev.trapdoor.as_ref(),
                public_matrix: shared_dev.pub_matrix.as_ref(),
                target: build_adjusted_target_chunk(evaluator, shared_dev, loaded, row_size),
            }
        })
        .collect::<Vec<_>>();

    evaluator
        .trap_sampler
        .preimage_batched_sharded(requests)
        .into_iter()
        .map(|(request_idx, preimage_chunk)| ComputedLweChunkTask {
            task: loaded_wave[request_idx].task.clone(),
            preimage_chunk,
        })
        .collect()
}

fn computed_wave_to_jobs<M>(computed_wave: Vec<ComputedLweChunkTask<M>>) -> Vec<CompactBytesJob>
where
    M: PolyMatrix + Send + Sync + 'static,
{
    computed_wave
        .into_par_iter()
        .map(|computed| {
            CompactBytesJob::new(
                column_chunk_id_prefix(
                    &k_high_row_checkpoint_prefix(
                        computed.task.gate_id,
                        computed.task.lut_id,
                        computed.task.lut_entry_idx,
                    ),
                    computed.task.chunk_idx,
                ),
                vec![(0usize, computed.preimage_chunk)],
            )
        })
        .collect()
}

fn store_chunk_tasks_gpu<M, SH, ST>(
    evaluator: &LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    params: &<M::P as Poly>::Params,
    row_size: usize,
    tasks: Vec<GpuLweChunkTask<M::P>>,
    progress_label: &str,
) where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
    <M::P as Poly>::Elem: Send + Sync,
{
    if tasks.is_empty() {
        return;
    }

    let shared = prepare_gpu_device_shared::<M, SH, ST>(evaluator, params, row_size);
    let wave_width = shared.len().max(1);
    let total_tasks = tasks.len();
    let completed_tasks = AtomicUsize::new(0);
    let mut cursor = 0usize;
    let mut current_loaded =
        load_wave::<M, ST>(&shared, &tasks[cursor..(cursor + wave_width).min(total_tasks)]);
    cursor += current_loaded.len();
    let mut pending_store_jobs: Option<Vec<CompactBytesJob>> = None;

    while !current_loaded.is_empty() {
        let current_task_count = current_loaded.len();
        let has_next = cursor < total_tasks;
        let current_loaded_owned = current_loaded;

        let (computed_wave, next_loaded) = if let Some(previous_jobs) = pending_store_jobs.take() {
            if has_next {
                let next_end = (cursor + wave_width).min(total_tasks);
                let next_tasks = tasks[cursor..next_end].to_vec();
                let ((computed_wave, next_loaded), ()) = rayon::join(
                    || {
                        rayon::join(
                            || {
                                compute_wave::<M, SH, ST>(
                                    evaluator,
                                    &shared,
                                    current_loaded_owned,
                                    row_size,
                                )
                            },
                            || load_wave::<M, ST>(&shared, &next_tasks),
                        )
                    },
                    || previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store),
                );
                cursor = next_end;
                (computed_wave, next_loaded)
            } else {
                let (computed_wave, ()) = rayon::join(
                    || {
                        compute_wave::<M, SH, ST>(
                            evaluator,
                            &shared,
                            current_loaded_owned,
                            row_size,
                        )
                    },
                    || previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store),
                );
                (computed_wave, Vec::new())
            }
        } else if has_next {
            let next_end = (cursor + wave_width).min(total_tasks);
            let next_tasks = tasks[cursor..next_end].to_vec();
            let (computed_wave, next_loaded) = rayon::join(
                || compute_wave::<M, SH, ST>(evaluator, &shared, current_loaded_owned, row_size),
                || load_wave::<M, ST>(&shared, &next_tasks),
            );
            cursor = next_end;
            (computed_wave, next_loaded)
        } else {
            (
                compute_wave::<M, SH, ST>(evaluator, &shared, current_loaded_owned, row_size),
                Vec::new(),
            )
        };

        let jobs = computed_wave_to_jobs(computed_wave);
        debug!(
            "{} wave computed: wave_tasks={}, compact_bytes_total={}",
            progress_label,
            current_task_count,
            compact_bytes_job_total(&jobs)
        );
        pending_store_jobs = Some(jobs);
        let done =
            completed_tasks.fetch_add(current_task_count, Ordering::Relaxed) + current_task_count;
        info!(
            "{} progress: {}/{} chunk tasks ({:.1}%)",
            progress_label,
            done,
            total_tasks,
            100.0 * (done as f64) / (total_tasks as f64)
        );
        current_loaded = next_loaded;
    }

    if let Some(previous_jobs) = pending_store_jobs.take() {
        previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
    }
}

pub(super) fn sample_aux_matrices_gpu<M, SH, ST>(
    evaluator: &LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    params: &<M::P as Poly>::Params,
    gate_entries: Vec<(GateId, GateState<M>)>,
    lut_entries: HashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
) where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
    <M::P as Poly>::Elem: Send + Sync,
{
    let start = Instant::now();
    let gate_count = gate_entries.len();
    let row_size = evaluator.pub_matrix.row_size();
    let chunk_count = k_high_chunk_count::<M>(params, row_size);
    let checkpoint_index = load_checkpoint_index(&evaluator.dir_path);
    let checkpoint_part_index_cache = build_part_index_cache(checkpoint_index.as_ref());

    let total_rows: usize = gate_entries
        .iter()
        .map(|(_, state)| {
            lut_entries
                .get(&state.lut_id)
                .unwrap_or_else(|| panic!("missing LUT state for lut_id {}", state.lut_id))
                .len()
        })
        .sum();

    let task_builds = gate_entries
        .into_par_iter()
        .map(|(gate_id, gate_state)| {
            let plt = lut_entries
                .get(&gate_state.lut_id)
                .unwrap_or_else(|| panic!("missing LUT state for lut_id {}", gate_state.lut_id));
            let input_pubkey_bytes = Arc::<[u8]>::from(gate_state.input_pubkey_bytes);
            let output_pubkey_bytes = Arc::<[u8]>::from(gate_state.output_pubkey_bytes);
            let entry_task_builds = plt
                .entries(params)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(x_k, (k, y_k))| {
                    let x_k_usize = usize::try_from(x_k).expect("LUT input must fit in usize");
                    let lut_entry_idx =
                        usize::try_from(k).expect("LUT row index must fit in usize");
                    if k_high_row_checkpoint_complete(
                        checkpoint_index.as_ref(),
                        checkpoint_part_index_cache.as_ref(),
                        gate_id,
                        gate_state.lut_id,
                        row_size,
                        params.modulus_digits(),
                        lut_entry_idx,
                    ) {
                        return (Vec::new(), 1usize);
                    }

                    let entry_tasks = (0..chunk_count)
                        .map(|chunk_idx| GpuLweChunkTask {
                            gate_id,
                            lut_id: gate_state.lut_id,
                            lut_entry_idx,
                            x_k_usize,
                            chunk_idx,
                            input_pubkey_bytes: Arc::clone(&input_pubkey_bytes),
                            output_pubkey_bytes: Arc::clone(&output_pubkey_bytes),
                            y_elem: y_k.clone(),
                        })
                        .collect::<Vec<_>>();
                    (entry_tasks, 0usize)
                })
                .collect::<Vec<_>>();
            let gate_resumed_rows: usize =
                entry_task_builds.iter().map(|(_, resumed)| *resumed).sum();
            let gate_tasks = entry_task_builds
                .into_iter()
                .flat_map(|(entry_tasks, _)| entry_tasks)
                .collect::<Vec<_>>();
            (gate_tasks, gate_resumed_rows)
        })
        .collect::<Vec<_>>();
    let resumed_rows: usize = task_builds.iter().map(|(_, resumed)| *resumed).sum();
    let tasks = task_builds.into_iter().flat_map(|(gate_tasks, _)| gate_tasks).collect::<Vec<_>>();

    let pending_rows = tasks.len() / chunk_count.max(1);
    if tasks.is_empty() {
        info!(
            "No pending LWE GPU auxiliary chunks to sample (rows_total={}, rows_resumed={}, elapsed={:?})",
            total_rows,
            resumed_rows,
            start.elapsed()
        );
        return;
    }

    info!(
        "LWE GPU sample_aux_matrices start: gates={}, rows_total={}, rows_pending={}, rows_resumed={}, chunk_count={}, chunk_tasks={}",
        gate_count,
        total_rows,
        pending_rows,
        resumed_rows,
        chunk_count,
        tasks.len()
    );
    store_chunk_tasks_gpu::<M, SH, ST>(
        evaluator,
        params,
        row_size,
        tasks,
        "LWE GPU sample_aux_matrices",
    );

    info!(
        "LWE GPU sample_aux_matrices finished: chunk_tasks={}, rows_total={}, rows_resumed={}, elapsed={:?}",
        pending_rows * chunk_count,
        total_rows,
        resumed_rows,
        start.elapsed()
    );
}

impl<M, SH, ST> PublicLutSampleAuxBenchEstimator<M> for LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
    <M::P as Poly>::Elem: Send + Sync,
{
    type Params = <M::P as Poly>::Params;

    fn estimate_public_lut_sample_aux_matrices(
        &self,
        params: &Self::Params,
        total_lut_entries: usize,
        total_lut_gates: usize,
    ) -> SampleAuxBenchEstimate {
        let total_preimages = total_lut_entries
            .checked_mul(total_lut_gates)
            .expect("total sampled LWE preimages overflowed usize");
        if total_preimages == 0 {
            return SampleAuxBenchEstimate::default();
        }

        let row_size = self.pub_matrix.row_size();
        let chunk_count = k_high_chunk_count::<M>(params, row_size);
        let shared = prepare_gpu_device_shared(self, params, row_size);
        let shared_dev = shared
            .first()
            .expect("LWE GPU benchmark estimator requires at least one prepared device");
        let input_pubkey = SH::new().sample_hash(
            params,
            self.hash_key,
            b"lwe_bench_input_pubkey",
            row_size,
            row_size * params.modulus_digits(),
            crate::sampler::DistType::FinRingDist,
        );
        let output_pubkey = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, GateId(0));
        let input_pubkey_bytes = Arc::<[u8]>::from(input_pubkey.clone().into_compact_bytes());
        let output_pubkey_bytes = Arc::<[u8]>::from(output_pubkey.clone().into_compact_bytes());
        let y_elem = <M::P as Poly>::Elem::constant(&params.modulus(), 1u64);

        let start = Instant::now();
        let loaded = LoadedLweChunkTask {
            device_slot: 0,
            task: GpuLweChunkTask {
                gate_id: GateId(0),
                lut_id: 0,
                lut_entry_idx: 0,
                x_k_usize: 0,
                chunk_idx: 0,
                input_pubkey_bytes: Arc::clone(&input_pubkey_bytes),
                output_pubkey_bytes: Arc::clone(&output_pubkey_bytes),
                y_elem: y_elem.clone(),
            },
            a_z: M::from_compact_bytes(&shared_dev.params, input_pubkey_bytes.as_ref()),
            a_lt: M::from_compact_bytes(&shared_dev.params, output_pubkey_bytes.as_ref()),
            y_poly: M::P::from_elem_to_constant(&shared_dev.params, &y_elem),
        };
        let target = build_adjusted_target_chunk(self, shared_dev, &loaded, row_size);
        let chunk = self.trap_sampler.preimage(
            &shared_dev.params,
            shared_dev.trapdoor.as_ref(),
            shared_dev.pub_matrix.as_ref(),
            &target,
        );
        let compact_len = chunk.into_compact_bytes().len();
        let elapsed = start.elapsed().as_secs_f64();
        let total_chunk_count = BigUint::from(total_preimages) * BigUint::from(chunk_count);
        SampleAuxBenchEstimate::from_chunk_big_count(elapsed, total_chunk_count, compact_len)
    }

    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        params: &Self::Params,
        plt: &PublicLut<M::P>,
        used_inputs: &[u64],
        lut_id: usize,
        gate_id: GateId,
        _error_sigma: f64,
    ) {
        let row_size = self.pub_matrix.row_size();
        let chunk_count = k_high_chunk_count::<M>(params, row_size);
        let checkpoint_index = load_checkpoint_index(&self.dir_path);
        let checkpoint_part_index_cache = build_part_index_cache(checkpoint_index.as_ref());
        let input_pubkey = SH::new().sample_hash(
            params,
            self.hash_key,
            b"lwe_bench_input_pubkey",
            row_size,
            row_size * params.modulus_digits(),
            crate::sampler::DistType::FinRingDist,
        );
        let output_pubkey = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id);
        let input_pubkey_bytes = Arc::<[u8]>::from(input_pubkey.into_compact_bytes());
        let output_pubkey_bytes = Arc::<[u8]>::from(output_pubkey.into_compact_bytes());

        let mut lut_rows = used_inputs
            .iter()
            .map(|&x| {
                let (row_idx, y_elem) = plt.get(params, x).unwrap_or_else(|| {
                    panic!("synthetic poly-bench checkpoint input {x} missing from LUT")
                });
                (
                    usize::try_from(x).expect("LUT input must fit usize"),
                    usize::try_from(row_idx).expect("LUT row index must fit usize"),
                    y_elem.clone(),
                )
            })
            .collect::<Vec<_>>();
        lut_rows.sort_unstable_by_key(|(_, row_idx, _)| *row_idx);
        lut_rows.dedup_by(|(_, left_idx, _), (_, right_idx, _)| left_idx == right_idx);

        let task_builds = lut_rows
            .into_par_iter()
            .map(|(x_k_usize, lut_entry_idx, y_elem)| {
                if k_high_row_checkpoint_complete(
                    checkpoint_index.as_ref(),
                    checkpoint_part_index_cache.as_ref(),
                    gate_id,
                    lut_id,
                    row_size,
                    params.modulus_digits(),
                    lut_entry_idx,
                ) {
                    return (Vec::new(), 1usize);
                }

                let entry_tasks = (0..chunk_count)
                    .map(|chunk_idx| GpuLweChunkTask {
                        gate_id,
                        lut_id,
                        lut_entry_idx,
                        x_k_usize,
                        chunk_idx,
                        input_pubkey_bytes: Arc::clone(&input_pubkey_bytes),
                        output_pubkey_bytes: Arc::clone(&output_pubkey_bytes),
                        y_elem: y_elem.clone(),
                    })
                    .collect::<Vec<_>>();
                (entry_tasks, 0usize)
            })
            .collect::<Vec<_>>();
        let resumed_rows: usize = task_builds.iter().map(|(_, resumed)| *resumed).sum();
        let tasks =
            task_builds.into_iter().flat_map(|(entry_tasks, _)| entry_tasks).collect::<Vec<_>>();
        let pending_rows = tasks.len() / chunk_count.max(1);
        if tasks.is_empty() {
            info!(
                "No pending LWE GPU dummy poly-bench chunks to sample (rows_total={}, rows_resumed={})",
                resumed_rows, resumed_rows
            );
            return;
        }

        info!(
            "LWE GPU dummy poly-bench sample_aux start: rows_total={}, rows_pending={}, rows_resumed={}, chunk_count={}",
            pending_rows + resumed_rows,
            pending_rows,
            resumed_rows,
            chunk_count
        );
        store_chunk_tasks_gpu::<M, SH, ST>(
            self,
            params,
            row_size,
            tasks,
            "LWE GPU dummy poly-bench sample_aux",
        );
        info!(
            "LWE GPU dummy poly-bench sample_aux finished: rows_total={}, rows_resumed={}",
            pending_rows + resumed_rows,
            resumed_rows
        );
    }
}
