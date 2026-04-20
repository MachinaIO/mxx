use super::*;
use crate::poly::{PolyParams, dcrt::gpu::detected_gpu_device_ids};
use rayon::prelude::*;
use std::{collections::HashMap, sync::Arc, time::Instant};

pub(super) struct GpuSlotTransferSharedByDevice<M: PolyMatrix> {
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
    pub c_b0: M,
    pub out_pubkey_matrix: M,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SlotTransferSlotBenchMeasurement {
    pub latency: f64,
    pub max_parallelism: u128,
    pub total_time: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SlotTransferStageBenchMeasurement {
    latency: f64,
    max_parallelism: u128,
    total_time: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SlotTransferTaskGroupBenchStats {
    task_count: usize,
    max_load_ms: f64,
    max_compute_ms: f64,
    max_store_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum SlotTransferStage1Task {
    B0Chunk { chunk_idx: usize },
    GateChunk { chunk_idx: usize },
    InputDirect,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum SlotTransferStage1BenchGroup {
    B0,
    Gate,
    InputDirect,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SlotTransferStage2Task {
    chunk_idx: usize,
}

struct LoadedSlotTransferStage1Task<M: PolyMatrix> {
    device_slot: usize,
    task: SlotTransferStage1Task,
    rhs_chunk: M,
    load_ms: f64,
}

struct ComputedSlotTransferStage1Task {
    task: SlotTransferStage1Task,
    bytes: Vec<u8>,
    load_ms: f64,
    compute_ms: f64,
}

struct LoadedSlotTransferStage2Task<M: PolyMatrix> {
    task: SlotTransferStage2Task,
    lhs: M,
    rhs_chunk: M,
    load_ms: f64,
}

struct ComputedSlotTransferStage2Task {
    task: SlotTransferStage2Task,
    bytes: Vec<u8>,
    load_ms: f64,
    compute_ms: f64,
}

pub(super) fn effective_gpu_slot_parallelism(slot_parallelism: usize) -> usize {
    let device_ids = detected_gpu_device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for BggPolyEncoding slot transfer"
    );
    slot_parallelism.min(device_ids.len()).max(1)
}

fn slot_device_ids(slot_parallelism: usize) -> Vec<i32> {
    let clamped_parallelism = effective_gpu_slot_parallelism(slot_parallelism);
    detected_gpu_device_ids().into_iter().take(clamped_parallelism).collect()
}

fn prepare_slot_transfer_shared_by_device<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    params: &<M::P as Poly>::Params,
    gate_id: GateId,
    secret_size: usize,
    slot_parallelism: usize,
) -> Vec<GpuSlotTransferSharedByDevice<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    slot_device_ids(slot_parallelism)
        .into_par_iter()
        .map(|device_id| {
            let local_params = params.params_for_device(device_id);
            let out_pubkey =
                evaluator.output_pubkey_for_params(&local_params, secret_size, gate_id);
            let c_b0 = M::from_compact_bytes(&local_params, &evaluator.c_b0_bytes);
            GpuSlotTransferSharedByDevice {
                params: local_params,
                c_b0,
                out_pubkey_matrix: out_pubkey.matrix,
            }
        })
        .collect()
}

fn assign_tasks_with_affinity<T, K, F>(
    tasks: Vec<T>,
    device_count: usize,
    affinity_key: F,
) -> Vec<Vec<T>>
where
    K: Eq + std::hash::Hash + Copy,
    F: Fn(&T) -> K,
{
    let mut per_device = (0..device_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut device_loads = vec![0usize; device_count];
    let mut preferred_device_by_key = HashMap::<K, usize>::new();

    for task in tasks {
        let key = affinity_key(&task);
        let min_load = device_loads.iter().copied().min().expect("device_count must be positive");
        let preferred = preferred_device_by_key.get(&key).copied();
        let chosen_device = preferred
            .filter(|&device_slot| device_loads[device_slot] == min_load)
            .unwrap_or_else(|| {
                device_loads
                    .iter()
                    .position(|&load| load == min_load)
                    .expect("at least one device must have the minimum load")
            });
        device_loads[chosen_device] += 1;
        preferred_device_by_key.entry(key).or_insert(chosen_device);
        per_device[chosen_device].push(task);
    }

    per_device
}

fn stage1_task_lhs_family(task: &SlotTransferStage1Task) -> usize {
    match task {
        SlotTransferStage1Task::InputDirect => 1,
        SlotTransferStage1Task::B0Chunk { .. } | SlotTransferStage1Task::GateChunk { .. } => 0,
    }
}

fn stage1_task_bench_group(task: &SlotTransferStage1Task) -> SlotTransferStage1BenchGroup {
    match task {
        SlotTransferStage1Task::B0Chunk { .. } => SlotTransferStage1BenchGroup::B0,
        SlotTransferStage1Task::GateChunk { .. } => SlotTransferStage1BenchGroup::Gate,
        SlotTransferStage1Task::InputDirect => SlotTransferStage1BenchGroup::InputDirect,
    }
}

fn maybe_elapsed_ms(started: Option<Instant>) -> f64 {
    started.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0)
}

fn update_group_store_timing<K>(
    group_stats: &mut HashMap<K, SlotTransferTaskGroupBenchStats>,
    group: K,
    load_ms: f64,
    compute_ms: f64,
    store_ms: f64,
) where
    K: Eq + std::hash::Hash + Copy,
{
    let stats = group_stats.entry(group).or_default();
    stats.max_load_ms = stats.max_load_ms.max(load_ms);
    stats.max_compute_ms = stats.max_compute_ms.max(compute_ms);
    stats.max_store_ms = stats.max_store_ms.max(store_ms);
}

fn finalize_stage_bench<K>(
    group_stats: &HashMap<K, SlotTransferTaskGroupBenchStats>,
    stage_parallelism: usize,
) -> SlotTransferStageBenchMeasurement
where
    K: Eq + std::hash::Hash,
{
    let latency = group_stats.values().fold(0.0_f64, |max_latency, stats| {
        max_latency.max(stats.max_load_ms.max(stats.max_compute_ms).max(stats.max_store_ms))
    });
    let total_time = group_stats.values().fold(0.0_f64, |sum, stats| {
        let group_latency = stats.max_load_ms.max(stats.max_compute_ms).max(stats.max_store_ms);
        sum + group_latency * stats.task_count as f64
    });
    SlotTransferStageBenchMeasurement {
        latency,
        max_parallelism: stage_parallelism as u128,
        total_time,
    }
}

fn add_stage_bench_to_slot_measurement(
    slot_bench_measurement: &mut SlotTransferSlotBenchMeasurement,
    stage_bench: SlotTransferStageBenchMeasurement,
) {
    slot_bench_measurement.latency += stage_bench.latency;
    slot_bench_measurement.max_parallelism =
        slot_bench_measurement.max_parallelism.max(stage_bench.max_parallelism);
    slot_bench_measurement.total_time += stage_bench.total_time;
}

fn add_scalar_stage_to_slot_measurement(
    slot_bench_measurement: &mut SlotTransferSlotBenchMeasurement,
    stage_ms: f64,
) {
    slot_bench_measurement.latency += stage_ms;
    slot_bench_measurement.max_parallelism = slot_bench_measurement.max_parallelism.max(1);
    slot_bench_measurement.total_time += stage_ms;
}

fn build_stage1_tasks(
    b0_chunk_count: usize,
    gate_chunk_count: usize,
) -> Vec<SlotTransferStage1Task> {
    let mut stage1_tasks = Vec::new();
    stage1_tasks
        .extend((0..b0_chunk_count).map(|chunk_idx| SlotTransferStage1Task::B0Chunk { chunk_idx }));
    stage1_tasks.extend(
        (0..gate_chunk_count).map(|chunk_idx| SlotTransferStage1Task::GateChunk { chunk_idx }),
    );
    stage1_tasks.push(SlotTransferStage1Task::InputDirect);
    stage1_tasks
}

fn load_stage1_rhs_chunk<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    shared: &GpuSlotTransferSharedByDevice<M>,
    dir: &std::path::Path,
    src_slot: usize,
    dst_slot: usize,
    gate_id: GateId,
    slot_preimage_b0_total_cols: usize,
    gate_total_cols: usize,
    task: SlotTransferStage1Task,
) -> M
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    match task {
        SlotTransferStage1Task::B0Chunk { chunk_idx } => read_matrix_column_chunk(
            &shared.params,
            dir,
            &evaluator.slot_preimage_b0_id_prefix(src_slot),
            slot_preimage_b0_total_cols,
            chunk_idx,
        ),
        SlotTransferStage1Task::GateChunk { chunk_idx } => read_matrix_column_chunk(
            &shared.params,
            dir,
            &evaluator.gate_preimage_id_prefix(gate_id, dst_slot),
            gate_total_cols,
            chunk_idx,
        ),
        SlotTransferStage1Task::InputDirect => evaluator
            .slot_a_for_params(&shared.params, shared.out_pubkey_matrix.row_size(), dst_slot)
            .decompose(),
    }
}

fn concat_chunk_bytes_on_base<M, B>(params: &<M::P as Poly>::Params, chunk_bytes: Vec<B>) -> M
where
    M: PolyMatrix + Send,
    B: AsRef<[u8]> + Send,
{
    if chunk_bytes.len() == 1 {
        return M::from_compact_bytes(params, chunk_bytes[0].as_ref());
    }
    let mut chunk_iter = chunk_bytes
        .into_par_iter()
        .map(|bytes| M::from_compact_bytes(params, bytes.as_ref()))
        .collect::<Vec<_>>()
        .into_iter();
    let first = chunk_iter.next().expect("slot-transfer chunk byte list must be non-empty");
    first.concat_columns_owned(chunk_iter.collect())
}

fn reduce_slot_transfer_matrix_bytes<M>(
    params: &<M::P as Poly>::Params,
    input_direct_bytes: Arc<[u8]>,
    c_transfer_chunk_bytes: Vec<Arc<[u8]>>,
    c_gate_bytes: Arc<[u8]>,
    output_plaintext: &M::P,
    scalar_poly: Option<&M::P>,
) -> M
where
    M: PolyMatrix + Send,
{
    let mut accum = M::from_compact_bytes(params, input_direct_bytes.as_ref());
    let c_transfer = concat_chunk_bytes_on_base::<M, Arc<[u8]>>(params, c_transfer_chunk_bytes);
    let transfer_term = c_transfer * output_plaintext.clone();
    let ((), c_gate) = rayon::join(
        || accum.add_in_place(&transfer_term),
        || M::from_compact_bytes(params, c_gate_bytes.as_ref()),
    );
    if let Some(scalar_poly) = scalar_poly {
        accum = accum * scalar_poly;
    }
    accum.add_in_place(&c_gate);
    accum
}

fn output_slot_for_params_gpu<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    params: &<M::P as Poly>::Params,
    input: &BggPolyEncoding<M>,
    plaintext_bytes_by_slot: &[Arc<[u8]>],
    src_slots: &[(u32, Option<u32>)],
    gate_id: GateId,
    dst_slot: usize,
    shared_by_device: &[GpuSlotTransferSharedByDevice<M>],
    slot_bench_output: Option<&mut SlotTransferSlotBenchMeasurement>,
) -> (Arc<[u8]>, Arc<[u8]>)
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let collect_bench = slot_bench_output.is_some();
    let mut slot_bench_measurement = SlotTransferSlotBenchMeasurement::default();
    let (src_slot_u32, scalar) = src_slots[dst_slot];
    let src_slot = usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
    assert!(
        src_slot < input.num_slots(),
        "source slot index {} out of range for input slot count {}",
        src_slot,
        input.num_slots()
    );

    let input_vector_bytes = &input.vector_bytes[src_slot];
    let secret_size = shared_by_device[0].out_pubkey_matrix.row_size();
    let src_plaintext =
        M::P::from_compact_bytes(params, plaintext_bytes_by_slot[src_slot].as_ref());
    let constant_term = src_plaintext
        .coeffs_biguints()
        .into_iter()
        .next()
        .expect("plaintext polynomial must contain a constant coefficient");
    drop(src_plaintext);
    let mut output_plaintext = M::P::from_biguint_to_constant(params, constant_term);
    let dir = evaluator.dir_path.as_path();
    let slot_preimage_b0_total_cols = trapdoor_public_column_count::<M>(params, secret_size * 2);
    let slot_preimage_b1_total_cols = secret_size * params.modulus_digits();
    let gate_total_cols = secret_size * params.modulus_digits();
    let stage1_started = Instant::now();
    let b0_chunk_count = column_chunk_count(slot_preimage_b0_total_cols);
    let b1_chunk_count = column_chunk_count(slot_preimage_b1_total_cols);
    let gate_chunk_count = column_chunk_count(gate_total_cols);
    let mut stage1_tasks = Vec::new();
    stage1_tasks
        .extend((0..b0_chunk_count).map(|chunk_idx| SlotTransferStage1Task::B0Chunk { chunk_idx }));
    stage1_tasks.extend(
        (0..gate_chunk_count).map(|chunk_idx| SlotTransferStage1Task::GateChunk { chunk_idx }),
    );
    stage1_tasks.push(SlotTransferStage1Task::InputDirect);
    let mut stage1_group_stats = if collect_bench {
        let mut stats =
            HashMap::<SlotTransferStage1BenchGroup, SlotTransferTaskGroupBenchStats>::new();
        for task in &stage1_tasks {
            stats.entry(stage1_task_bench_group(task)).or_default().task_count += 1;
        }
        stats
    } else {
        HashMap::new()
    };
    let stage1_batches =
        assign_tasks_with_affinity(stage1_tasks, shared_by_device.len(), stage1_task_lhs_family);
    let stage1_wave_count = stage1_batches.iter().map(Vec::len).max().unwrap_or(0);
    let input_vector_by_device = shared_by_device
        .par_iter()
        .map(|shared| M::from_compact_bytes(&shared.params, input_vector_bytes.as_ref()))
        .collect::<Vec<_>>();
    let load_stage1_wave = |wave_idx: usize| {
        let loaded_wave = stage1_batches
            .par_iter()
            .zip(shared_by_device.par_iter())
            .enumerate()
            .filter_map(|(device_slot, (batch, shared))| {
                let task = batch.get(wave_idx).copied()?;
                let load_started = collect_bench.then(Instant::now);
                let rhs_chunk = load_stage1_rhs_chunk::<M, HS>(
                    evaluator,
                    shared,
                    dir,
                    src_slot,
                    dst_slot,
                    gate_id,
                    slot_preimage_b0_total_cols,
                    gate_total_cols,
                    task,
                );
                Some(LoadedSlotTransferStage1Task {
                    device_slot,
                    task,
                    rhs_chunk,
                    load_ms: maybe_elapsed_ms(load_started),
                })
            })
            .collect::<Vec<_>>();
        (!loaded_wave.is_empty()).then_some(loaded_wave)
    };
    let compute_stage1_wave = |loaded_wave: Vec<LoadedSlotTransferStage1Task<M>>| {
        loaded_wave
            .into_par_iter()
            .map(|loaded_task| {
                let compute_started = collect_bench.then(Instant::now);
                let lhs = match loaded_task.task {
                    SlotTransferStage1Task::InputDirect => {
                        &input_vector_by_device[loaded_task.device_slot]
                    }
                    SlotTransferStage1Task::B0Chunk { .. } |
                    SlotTransferStage1Task::GateChunk { .. } => {
                        &shared_by_device[loaded_task.device_slot].c_b0
                    }
                };
                let output = lhs * &loaded_task.rhs_chunk;
                ComputedSlotTransferStage1Task {
                    task: loaded_task.task,
                    bytes: output.into_compact_bytes(),
                    load_ms: loaded_task.load_ms,
                    compute_ms: maybe_elapsed_ms(compute_started),
                }
            })
            .collect::<Vec<_>>()
    };
    let mut b0_chunk_bytes = (0..b0_chunk_count).map(|_| None).collect::<Vec<_>>();
    let mut gate_chunk_bytes = (0..gate_chunk_count).map(|_| None).collect::<Vec<_>>();
    let mut input_direct_bytes = None;
    let mut store_stage1_wave = |outputs: Vec<ComputedSlotTransferStage1Task>| {
        for output in outputs {
            let store_started = collect_bench.then(Instant::now);
            match output.task {
                SlotTransferStage1Task::B0Chunk { chunk_idx } => {
                    b0_chunk_bytes[chunk_idx] = Some(output.bytes);
                }
                SlotTransferStage1Task::GateChunk { chunk_idx } => {
                    gate_chunk_bytes[chunk_idx] = Some(output.bytes);
                }
                SlotTransferStage1Task::InputDirect => {
                    input_direct_bytes = Some(output.bytes);
                }
            }
            if collect_bench {
                update_group_store_timing(
                    &mut stage1_group_stats,
                    stage1_task_bench_group(&output.task),
                    output.load_ms,
                    output.compute_ms,
                    maybe_elapsed_ms(store_started),
                );
            }
        }
    };
    let mut next_stage1_wave_idx = 1usize;
    let mut current_stage1_wave = if stage1_wave_count == 0 { None } else { load_stage1_wave(0) };
    let mut previous_stage1_outputs = None;
    while let Some(loaded_stage1_wave) = current_stage1_wave.take() {
        let should_load_next = next_stage1_wave_idx < stage1_wave_count;
        let previous_outputs_to_store = previous_stage1_outputs.take();
        let ((computed_outputs, next_loaded_wave), ()) = rayon::join(
            || {
                rayon::join(
                    || compute_stage1_wave(loaded_stage1_wave),
                    || if should_load_next { load_stage1_wave(next_stage1_wave_idx) } else { None },
                )
            },
            || {
                if let Some(outputs) = previous_outputs_to_store {
                    store_stage1_wave(outputs);
                }
            },
        );
        previous_stage1_outputs = Some(computed_outputs);
        current_stage1_wave = next_loaded_wave;
        if should_load_next {
            next_stage1_wave_idx += 1;
        }
    }
    if let Some(outputs) = previous_stage1_outputs.take() {
        store_stage1_wave(outputs);
    }
    let b0_chunk_bytes = b0_chunk_bytes
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, maybe_bytes)| {
            maybe_bytes.unwrap_or_else(|| {
                panic!("missing slot-transfer b0 chunk bytes for chunk {chunk_idx}")
            })
        })
        .collect::<Vec<_>>();
    let gate_chunk_bytes = gate_chunk_bytes
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, maybe_bytes)| {
            maybe_bytes.unwrap_or_else(|| {
                panic!("missing slot-transfer gate chunk bytes for chunk {chunk_idx}")
            })
        })
        .collect::<Vec<_>>();
    let c_b0_slot_preimage_b0: M = concat_chunk_bytes_on_base(params, b0_chunk_bytes);
    let c_b0_slot_preimage_b0_bytes = Arc::<[u8]>::from(c_b0_slot_preimage_b0.into_compact_bytes());
    let c_gate_bytes = Arc::<[u8]>::from(
        concat_chunk_bytes_on_base::<M, Vec<u8>>(params, gate_chunk_bytes).into_compact_bytes(),
    );
    let input_direct_bytes =
        Arc::<[u8]>::from(input_direct_bytes.expect("missing slot-transfer input-direct bytes"));
    if collect_bench {
        let stage1_bench =
            finalize_stage_bench(&stage1_group_stats, stage1_batches.iter().map(Vec::len).sum());
        add_stage_bench_to_slot_measurement(&mut slot_bench_measurement, stage1_bench);
    }
    let _stage1_ms = stage1_started.elapsed().as_secs_f64() * 1000.0;

    let stage2_started = Instant::now();
    let stage2_tasks = (0..b1_chunk_count)
        .map(|chunk_idx| SlotTransferStage2Task { chunk_idx })
        .collect::<Vec<_>>();
    let mut stage2_group_stats = if collect_bench {
        let mut stats = HashMap::<usize, SlotTransferTaskGroupBenchStats>::new();
        let entry = stats.entry(0).or_default();
        entry.task_count = stage2_tasks.len();
        stats
    } else {
        HashMap::new()
    };
    let stage2_batches =
        assign_tasks_with_affinity(stage2_tasks, shared_by_device.len(), |_| 0usize);
    let stage2_wave_count = stage2_batches.iter().map(Vec::len).max().unwrap_or(0);
    let mut stage1_mid_cache_by_device =
        (0..shared_by_device.len()).map(|_| None).collect::<Vec<_>>();
    let load_stage2_wave = |wave_idx: usize, stage1_mid_cache_by_device: &mut Vec<Option<M>>| {
        let loaded_wave = stage2_batches
            .iter()
            .zip(shared_by_device.iter())
            .zip(stage1_mid_cache_by_device.iter_mut())
            .filter_map(|((batch, shared), stage1_mid_cache)| {
                let task = batch.get(wave_idx).copied()?;
                let load_started = collect_bench.then(Instant::now);
                let lhs = stage1_mid_cache
                    .get_or_insert_with(|| {
                        M::from_compact_bytes(&shared.params, c_b0_slot_preimage_b0_bytes.as_ref())
                    })
                    .clone();
                let rhs_chunk = read_matrix_column_chunk::<M>(
                    &shared.params,
                    dir,
                    &evaluator.slot_preimage_b1_id_prefix(dst_slot),
                    slot_preimage_b1_total_cols,
                    task.chunk_idx,
                );
                Some(LoadedSlotTransferStage2Task {
                    task,
                    lhs,
                    rhs_chunk,
                    load_ms: maybe_elapsed_ms(load_started),
                })
            })
            .collect::<Vec<_>>();
        (!loaded_wave.is_empty()).then_some(loaded_wave)
    };
    let compute_stage2_wave = |loaded_wave: Vec<LoadedSlotTransferStage2Task<M>>| {
        loaded_wave
            .into_par_iter()
            .map(|loaded_task| {
                let compute_started = collect_bench.then(Instant::now);
                let output = &loaded_task.lhs * &loaded_task.rhs_chunk;
                ComputedSlotTransferStage2Task {
                    task: loaded_task.task,
                    bytes: output.into_compact_bytes(),
                    load_ms: loaded_task.load_ms,
                    compute_ms: maybe_elapsed_ms(compute_started),
                }
            })
            .collect::<Vec<_>>()
    };
    let mut c_transfer_chunk_bytes = (0..b1_chunk_count).map(|_| None).collect::<Vec<_>>();
    let mut store_stage2_wave = |outputs: Vec<ComputedSlotTransferStage2Task>| {
        for output in outputs {
            let store_started = collect_bench.then(Instant::now);
            c_transfer_chunk_bytes[output.task.chunk_idx] = Some(Arc::<[u8]>::from(output.bytes));
            if collect_bench {
                update_group_store_timing(
                    &mut stage2_group_stats,
                    0usize,
                    output.load_ms,
                    output.compute_ms,
                    maybe_elapsed_ms(store_started),
                );
            }
        }
    };
    let mut next_stage2_wave_idx = 1usize;
    let mut current_stage2_wave = if stage2_wave_count == 0 {
        None
    } else {
        load_stage2_wave(0, &mut stage1_mid_cache_by_device)
    };
    let mut previous_stage2_outputs = None;
    while let Some(loaded_stage2_wave) = current_stage2_wave.take() {
        let should_load_next = next_stage2_wave_idx < stage2_wave_count;
        let previous_outputs_to_store = previous_stage2_outputs.take();
        let ((computed_outputs, next_loaded_wave), ()) = rayon::join(
            || {
                rayon::join(
                    || compute_stage2_wave(loaded_stage2_wave),
                    || {
                        if should_load_next {
                            load_stage2_wave(next_stage2_wave_idx, &mut stage1_mid_cache_by_device)
                        } else {
                            None
                        }
                    },
                )
            },
            || {
                if let Some(outputs) = previous_outputs_to_store {
                    store_stage2_wave(outputs);
                }
            },
        );
        previous_stage2_outputs = Some(computed_outputs);
        current_stage2_wave = next_loaded_wave;
        if should_load_next {
            next_stage2_wave_idx += 1;
        }
    }
    if let Some(outputs) = previous_stage2_outputs.take() {
        store_stage2_wave(outputs);
    }
    let c_transfer_chunk_bytes = c_transfer_chunk_bytes
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, maybe_bytes)| {
            maybe_bytes.unwrap_or_else(|| {
                panic!("missing slot-transfer transfer chunk bytes for chunk {chunk_idx}")
            })
        })
        .collect::<Vec<_>>();
    if collect_bench {
        let stage2_bench =
            finalize_stage_bench(&stage2_group_stats, stage2_batches.iter().map(Vec::len).sum());
        add_stage_bench_to_slot_measurement(&mut slot_bench_measurement, stage2_bench);
    }
    let _stage2_ms = stage2_started.elapsed().as_secs_f64() * 1000.0;

    let stage3_started = Instant::now();
    let scalar_poly = scalar.map(|scalar| M::P::from_usize_to_constant(params, scalar as usize));
    let out_vector = reduce_slot_transfer_matrix_bytes::<M>(
        params,
        input_direct_bytes,
        c_transfer_chunk_bytes,
        c_gate_bytes,
        &output_plaintext,
        scalar_poly.as_ref(),
    );
    if let Some(scalar_poly) = scalar_poly.as_ref() {
        output_plaintext = output_plaintext * scalar_poly;
    }
    let output_vector_bytes = Arc::<[u8]>::from(out_vector.into_compact_bytes());
    let output_plaintext_bytes = Arc::<[u8]>::from(output_plaintext.to_compact_bytes());
    if collect_bench {
        add_scalar_stage_to_slot_measurement(
            &mut slot_bench_measurement,
            stage3_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if let Some(slot_bench_output) = slot_bench_output {
        *slot_bench_output = slot_bench_measurement;
    }

    (output_vector_bytes, output_plaintext_bytes)
}

pub(super) fn evaluate_slot_transfer_slots_gpu<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    params: &<M::P as Poly>::Params,
    input: &BggPolyEncoding<M>,
    plaintext_bytes_by_slot: &[Arc<[u8]>],
    src_slots: &[(u32, Option<u32>)],
    gate_id: GateId,
    out_pubkey: &BggPublicKey<M>,
    configured_parallelism: usize,
) -> (Vec<Arc<[u8]>>, Vec<Arc<[u8]>>)
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let slot_count = src_slots.len();
    let shared_by_device = prepare_slot_transfer_shared_by_device::<M, HS>(
        evaluator,
        params,
        gate_id,
        out_pubkey.matrix.row_size(),
        configured_parallelism,
    );
    let mut output_vector_bytes = Vec::with_capacity(slot_count);
    let mut output_plaintext_bytes = Vec::with_capacity(slot_count);

    for dst_slot in 0..slot_count {
        let (vector_bytes, plaintext_bytes) = output_slot_for_params_gpu::<M, HS>(
            evaluator,
            params,
            input,
            plaintext_bytes_by_slot,
            src_slots,
            gate_id,
            dst_slot,
            &shared_by_device,
            None,
        );
        output_vector_bytes.push(vector_bytes);
        output_plaintext_bytes.push(plaintext_bytes);
    }

    (output_vector_bytes, output_plaintext_bytes)
}

fn benchmark_output_slot_for_params_gpu<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    params: &<M::P as Poly>::Params,
    input: &BggPolyEncoding<M>,
    plaintext_bytes_by_slot: &[Arc<[u8]>],
    src_slots: &[(u32, Option<u32>)],
    gate_id: GateId,
    dst_slot: usize,
    shared: &GpuSlotTransferSharedByDevice<M>,
) -> SlotTransferSlotBenchMeasurement
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let mut slot_bench_measurement = SlotTransferSlotBenchMeasurement::default();
    let (src_slot_u32, scalar) = src_slots[dst_slot];
    let src_slot = usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
    assert!(
        src_slot < input.num_slots(),
        "source slot index {} out of range for input slot count {}",
        src_slot,
        input.num_slots()
    );

    let input_vector_bytes = &input.vector_bytes[src_slot];
    let secret_size = shared.out_pubkey_matrix.row_size();
    let src_plaintext =
        M::P::from_compact_bytes(params, plaintext_bytes_by_slot[src_slot].as_ref());
    let constant_term = src_plaintext
        .coeffs_biguints()
        .into_iter()
        .next()
        .expect("plaintext polynomial must contain a constant coefficient");
    let mut output_plaintext = M::P::from_biguint_to_constant(params, constant_term);
    let dir = evaluator.dir_path.as_path();
    let slot_preimage_b0_total_cols = trapdoor_public_column_count::<M>(params, secret_size * 2);
    let slot_preimage_b1_total_cols = secret_size * params.modulus_digits();
    let gate_total_cols = secret_size * params.modulus_digits();
    let b0_chunk_count = column_chunk_count(slot_preimage_b0_total_cols);
    let b1_chunk_count = column_chunk_count(slot_preimage_b1_total_cols);
    let gate_chunk_count = column_chunk_count(gate_total_cols);

    let stage1_tasks = build_stage1_tasks(b0_chunk_count, gate_chunk_count);
    let mut stage1_group_stats =
        HashMap::<SlotTransferStage1BenchGroup, SlotTransferTaskGroupBenchStats>::new();
    let mut stage1_representative_tasks = Vec::new();
    for task in stage1_tasks {
        let entry = stage1_group_stats.entry(stage1_task_bench_group(&task)).or_default();
        if entry.task_count == 0 {
            stage1_representative_tasks.push(task);
        }
        entry.task_count += 1;
    }
    let input_vector = M::from_compact_bytes(&shared.params, input_vector_bytes.as_ref());
    let mut b0_chunk_bytes = (0..b0_chunk_count).map(|_| None).collect::<Vec<_>>();
    let mut gate_chunk_bytes = (0..gate_chunk_count).map(|_| None).collect::<Vec<_>>();
    let mut input_direct_bytes = None;
    for task in stage1_representative_tasks {
        let load_started = Instant::now();
        let rhs_chunk = load_stage1_rhs_chunk::<M, HS>(
            evaluator,
            shared,
            dir,
            src_slot,
            dst_slot,
            gate_id,
            slot_preimage_b0_total_cols,
            gate_total_cols,
            task,
        );
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let compute_started = Instant::now();
        let lhs = match task {
            SlotTransferStage1Task::InputDirect => &input_vector,
            SlotTransferStage1Task::B0Chunk { .. } | SlotTransferStage1Task::GateChunk { .. } => {
                &shared.c_b0
            }
        };
        let output = lhs * &rhs_chunk;
        let compute_ms = compute_started.elapsed().as_secs_f64() * 1000.0;
        let store_started = Instant::now();
        let output_bytes = output.into_compact_bytes();
        match task {
            SlotTransferStage1Task::B0Chunk { chunk_idx } => {
                let bytes = Some(output_bytes);
                for chunk_bytes in &mut b0_chunk_bytes {
                    *chunk_bytes = bytes.clone();
                }
                debug_assert!(chunk_idx < b0_chunk_count);
            }
            SlotTransferStage1Task::GateChunk { chunk_idx } => {
                let bytes = Some(output_bytes);
                for chunk_bytes in &mut gate_chunk_bytes {
                    *chunk_bytes = bytes.clone();
                }
                debug_assert!(chunk_idx < gate_chunk_count);
            }
            SlotTransferStage1Task::InputDirect => {
                input_direct_bytes = Some(output_bytes);
            }
        }
        update_group_store_timing(
            &mut stage1_group_stats,
            stage1_task_bench_group(&task),
            load_ms,
            compute_ms,
            store_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let c_b0_slot_preimage_b0: M = concat_chunk_bytes_on_base(
        params,
        b0_chunk_bytes
            .into_iter()
            .enumerate()
            .map(|(chunk_idx, maybe_bytes)| {
                maybe_bytes.unwrap_or_else(|| {
                    panic!("missing slot-transfer b0 chunk bytes for chunk {chunk_idx}")
                })
            })
            .collect(),
    );
    let c_b0_slot_preimage_b0_bytes = Arc::<[u8]>::from(c_b0_slot_preimage_b0.into_compact_bytes());
    let c_gate_bytes = Arc::<[u8]>::from(
        concat_chunk_bytes_on_base::<M, Vec<u8>>(
            params,
            gate_chunk_bytes
                .into_iter()
                .enumerate()
                .map(|(chunk_idx, maybe_bytes)| {
                    maybe_bytes.unwrap_or_else(|| {
                        panic!("missing slot-transfer gate chunk bytes for chunk {chunk_idx}")
                    })
                })
                .collect(),
        )
        .into_compact_bytes(),
    );
    let input_direct_bytes =
        Arc::<[u8]>::from(input_direct_bytes.expect("missing slot-transfer input-direct bytes"));
    let stage1_bench = finalize_stage_bench(
        &stage1_group_stats,
        stage1_group_stats.values().map(|stats| stats.task_count).sum(),
    );
    add_stage_bench_to_slot_measurement(&mut slot_bench_measurement, stage1_bench);

    let stage2_tasks = (0..b1_chunk_count)
        .map(|chunk_idx| SlotTransferStage2Task { chunk_idx })
        .collect::<Vec<_>>();
    let mut stage2_group_stats = HashMap::<usize, SlotTransferTaskGroupBenchStats>::new();
    stage2_group_stats.entry(0usize).or_default().task_count = stage2_tasks.len();
    let mut c_transfer_chunk_bytes = (0..b1_chunk_count).map(|_| None).collect::<Vec<_>>();
    for task in stage2_tasks.into_iter().take(1) {
        let load_started = Instant::now();
        let lhs = M::from_compact_bytes(&shared.params, c_b0_slot_preimage_b0_bytes.as_ref());
        let rhs_chunk = read_matrix_column_chunk::<M>(
            &shared.params,
            dir,
            &evaluator.slot_preimage_b1_id_prefix(dst_slot),
            slot_preimage_b1_total_cols,
            task.chunk_idx,
        );
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let compute_started = Instant::now();
        let output = &lhs * &rhs_chunk;
        let compute_ms = compute_started.elapsed().as_secs_f64() * 1000.0;
        let store_started = Instant::now();
        let output_bytes = output.into_compact_bytes();
        let bytes = Some(Arc::<[u8]>::from(output_bytes));
        for chunk_bytes in &mut c_transfer_chunk_bytes {
            *chunk_bytes = bytes.clone();
        }
        update_group_store_timing(
            &mut stage2_group_stats,
            0usize,
            load_ms,
            compute_ms,
            store_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let c_transfer_chunk_bytes = c_transfer_chunk_bytes
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, maybe_bytes)| {
            maybe_bytes.unwrap_or_else(|| {
                panic!("missing slot-transfer transfer chunk bytes for chunk {chunk_idx}")
            })
        })
        .collect::<Vec<_>>();
    let stage2_bench = finalize_stage_bench(
        &stage2_group_stats,
        stage2_group_stats.values().map(|stats| stats.task_count).sum(),
    );
    add_stage_bench_to_slot_measurement(&mut slot_bench_measurement, stage2_bench);

    let stage3_started = Instant::now();
    let scalar_poly = scalar.map(|scalar| M::P::from_usize_to_constant(params, scalar as usize));
    let out_vector = reduce_slot_transfer_matrix_bytes::<M>(
        params,
        input_direct_bytes,
        c_transfer_chunk_bytes,
        c_gate_bytes,
        &output_plaintext,
        scalar_poly.as_ref(),
    );
    if let Some(scalar_poly) = scalar_poly.as_ref() {
        output_plaintext = output_plaintext * scalar_poly;
    }
    let _output_vector_bytes = Arc::<[u8]>::from(out_vector.into_compact_bytes());
    let _output_plaintext_bytes = Arc::<[u8]>::from(output_plaintext.to_compact_bytes());
    add_scalar_stage_to_slot_measurement(
        &mut slot_bench_measurement,
        stage3_started.elapsed().as_secs_f64() * 1000.0,
    );

    slot_bench_measurement
}

pub(super) fn benchmark_slot_transfer_chunk_gpu<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    samples: &crate::bench_estimator::BggPolyEncodingBenchSamples<'_, M>,
    iterations: usize,
) -> crate::bench_estimator::PolyEncodingChunkBenchMeasurement
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let params = samples.params;
    let input = samples.slot_transfer_input;
    let plaintext_bytes_by_slot = input
        .plaintext_bytes
        .as_ref()
        .expect("BggPolyEncoding slot transfer benchmark requires plaintext_bytes");
    let secret_size = input.pubkey.matrix.row_size();
    let out_pubkey =
        evaluator.output_pubkey_for_params(params, secret_size, samples.slot_transfer_gate_id);
    let configured_parallelism = effective_gpu_slot_parallelism(
        crate::env::slot_transfer_slot_parallelism()
            .max(1)
            .min(samples.slot_transfer_src_slots.len().max(1)),
    );
    let shared_by_device = prepare_slot_transfer_shared_by_device::<M, HS>(
        evaluator,
        params,
        samples.slot_transfer_gate_id,
        out_pubkey.matrix.row_size(),
        configured_parallelism,
    );
    let iterations = iterations.max(1);
    let mut latency_sum = 0.0;
    let mut total_time_sum = 0.0;
    let mut max_parallelism = 0u128;

    for _ in 0..iterations {
        let slot_bench = benchmark_output_slot_for_params_gpu::<M, HS>(
            evaluator,
            params,
            input,
            plaintext_bytes_by_slot,
            samples.slot_transfer_src_slots,
            samples.slot_transfer_gate_id,
            0,
            &shared_by_device[0],
        );
        latency_sum += slot_bench.latency;
        total_time_sum += slot_bench.total_time;
        max_parallelism = max_parallelism.max(slot_bench.max_parallelism);
    }

    crate::bench_estimator::PolyEncodingChunkBenchMeasurement {
        latency: latency_sum / iterations as f64,
        max_parallelism,
        total_time: total_time_sum / iterations as f64,
        peak_vram: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::slot_device_ids;
    use crate::{
        __PAIR, __TestState,
        bgg::{
            poly_encoding::BggPolyEncoding,
            public_key::BggPublicKey,
            sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
        },
        circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
        },
        sampler::{
            DistType, PolyHashSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::{BggPolyEncodingSTEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path, sync::Arc};

    const SIGMA: f64 = 4.578;

    struct EnvVarGuard {
        key: &'static str,
        old_value: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old_value = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, old_value }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.old_value {
                unsafe { std::env::set_var(self.key, value) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    struct DummyGpuPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<GpuDCRTPolyMatrix>> for DummyGpuPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<GpuDCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPublicKey<GpuDCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPublicKey<GpuDCRTPolyMatrix>,
            _input: &BggPublicKey<GpuDCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPublicKey<GpuDCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer GPU tests")
        }
    }

    struct DummyGpuPolyEncodingPltEvaluator;

    impl PltEvaluator<BggPolyEncoding<GpuDCRTPolyMatrix>> for DummyGpuPolyEncodingPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPolyEncoding<GpuDCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPolyEncoding<GpuDCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPolyEncoding<GpuDCRTPolyMatrix>,
            _input: &BggPolyEncoding<GpuDCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPolyEncoding<GpuDCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer GPU tests")
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_slot_transfer_bgg_poly_encoding_gpu_uses_detected_gpu_ids() {
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

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_slot_transfer_bgg_poly_encoding_gpu_runs_on_gpu_repeatedly() {
        let detected_gpu_ids = detected_gpu_device_ids();
        assert!(
            !detected_gpu_ids.is_empty(),
            "at least one GPU device is required for BggPolyEncoding slot-transfer GPU tests"
        );

        let _storage_lock = storage_test_lock().await;
        let _parallelism_guard = EnvVarGuard::set(
            "SLOT_TRANSFER_SLOT_PARALLELISM",
            &(detected_gpu_ids.len() + 2).to_string(),
        );

        for iter in 0..5usize {
            gpu_device_sync();
            let params = GpuDCRTPolyParams::new(4, vec![131041, 131009], 1);
            let hash_key = [0x41u8; 32];
            let secret_size = 2usize;
            let num_slots = 3usize;
            let dir_path = format!("test_data/test_slot_transfer_bgg_poly_encoding_gpu_{iter}");
            let dir = Path::new(&dir_path);
            if dir.exists() {
                fs::remove_dir_all(dir).unwrap();
            }
            fs::create_dir_all(dir).unwrap();
            init_storage_system(dir.to_path_buf());

            let tag: u64 = (iter as u64) + 11;
            let bgg_pubkey_sampler =
                BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(
                    hash_key,
                    secret_size,
                );
            let public_keys = bgg_pubkey_sampler.sample(&params, &tag.to_le_bytes(), &[true]);

            let secrets = (0..secret_size)
                .map(|idx| GpuDCRTPoly::from_usize_to_constant(&params, idx + 2))
                .collect::<Vec<_>>();
            let plaintext_rows = vec![vec![
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, 5).to_compact_bytes(),
                ),
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, 9).to_compact_bytes(),
                ),
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, 4).to_compact_bytes(),
                ),
            ]];
            let one_pubkey = public_keys[0].clone();
            let input_pubkey = public_keys[1].clone();
            let pubkey_evaluator =
                BggPublicKeySTEvaluator::<
                    GpuDCRTPolyMatrix,
                    GpuDCRTPolyUniformSampler,
                    GpuDCRTPolyHashSampler<Keccak256>,
                    GpuDCRTPolyTrapdoorSampler,
                >::new(
                    hash_key, secret_size, num_slots, SIGMA, 0.0, dir.to_path_buf()
                );

            let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];
            let mut circuit = PolyCircuit::new();
            let inputs = circuit.input(1);
            let transferred = circuit.slot_transfer_gate(inputs.at(0), &src_slots);
            let transferred_gate = transferred.as_single_wire();
            circuit.output(vec![transferred]);

            let result_pubkey = circuit.eval(
                &params,
                one_pubkey,
                vec![input_pubkey.clone()],
                None::<&DummyGpuPubKeyPltEvaluator>,
                Some(&pubkey_evaluator),
                None,
            );
            assert_eq!(result_pubkey.len(), 1);

            pubkey_evaluator.sample_aux_matrices(&params);
            wait_for_all_writes(dir.to_path_buf()).await.unwrap();
            let slot_secret_mats =
                pubkey_evaluator.load_slot_secret_mats_checkpoint(&params).expect(
                    "gpu slot secret matrix checkpoints should exist after sample_aux_matrices",
                );

            let encoding_sampler =
                BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
            let encodings = encoding_sampler.sample(
                &params,
                &public_keys,
                &plaintext_rows,
                Some(&slot_secret_mats),
            );
            let one = encodings[0].clone();
            let input = encodings[1].clone();

            let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
            let b0_matrix = pubkey_evaluator
                .load_b0_matrix_checkpoint(&params)
                .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
            let c_b0 = s_vec.clone() * &b0_matrix;
            let evaluator = BggPolyEncodingSTEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
            >::new(
                hash_key,
                dir.to_path_buf(),
                pubkey_evaluator.checkpoint_prefix(&params),
                c_b0.to_compact_bytes(),
            );

            let result = circuit.eval(
                &params,
                one,
                vec![input.clone()],
                None::<&DummyGpuPolyEncodingPltEvaluator>,
                Some(&evaluator),
                Some(1),
            );
            assert_eq!(result.len(), 1);
            let output = &result[0];
            assert_eq!(output.pubkey, result_pubkey[0]);

            let a_out = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_gate_a_out_{}", transferred_gate),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            );
            let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, secret_size);

            for (dst_slot, (src_slot_u32, scalar)) in src_slots.into_iter().enumerate() {
                let src_slot = src_slot_u32 as usize;
                let s_dst = GpuDCRTPolyMatrix::from_compact_bytes(
                    &params,
                    slot_secret_mats[dst_slot].as_ref(),
                );
                let source_plaintext = input
                    .plaintext(src_slot)
                    .expect("input encoding should reveal plaintext constants");
                let constant_term = source_plaintext
                    .coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("plaintext must have a constant coefficient");
                let mut scaled_plaintext =
                    GpuDCRTPoly::from_biguint_to_constant(&params, constant_term);
                if let Some(scalar) = scalar {
                    let scalar_poly = GpuDCRTPoly::from_usize_to_constant(&params, scalar as usize);
                    scaled_plaintext = scaled_plaintext * scalar_poly;
                }
                let expected_vector = (s_vec.clone() * &s_dst) *
                    &(a_out.clone() - (gadget_matrix.clone() * &scaled_plaintext));
                assert_eq!(output.vector(dst_slot), expected_vector);

                let output_plaintext = output
                    .plaintext(dst_slot)
                    .expect("slot-transfer output should reveal plaintext constants");
                assert_eq!(output_plaintext, scaled_plaintext);
            }

            gpu_device_sync();
        }
    }
}
