use super::*;
use crate::{
    bgg::public_key::BggPublicKey,
    lookup::ggh15::{
        pubkey::{
            column_chunk_bounds, column_chunk_count, read_matrix_column_chunk,
            trapdoor_public_column_count,
        },
        public_lookup_gpu_device_ids,
    },
    poly::PolyParams,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::Mul,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use tracing::debug;

use crate::lookup::ggh15::encoding::GGH15PublicLookupSharedState;

pub(super) struct GpuPublicLookupSharedByDevice<M: PolyMatrix> {
    pub device_id: i32,
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
    pub shared: GGH15PublicLookupSharedState<M>,
}

struct LoadedPublicLookupSlot<M: PolyMatrix> {
    slot_idx: usize,
    slot_started: Instant,
    load_ms: f64,
    c_b0_by_device: Vec<M>,
    input_vector_by_device: Vec<M>,
    x_by_device: Vec<M::P>,
}

struct ComputedPublicLookupSlot<M: PolyMatrix> {
    slot_idx: usize,
    slot_started: Instant,
    load_ms: f64,
    plaintext_bytes: Arc<[u8]>,
    output_vector: M,
    stage1_ms: f64,
    stage2_ms: f64,
    compute_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct PublicLookupStageBenchMeasurement {
    latency: f64,
    max_parallelism: u128,
    total_time: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PublicLookupSlotBenchMeasurement {
    pub latency: f64,
    pub max_parallelism: u128,
    pub total_time: f64,
}

const LHS_CB0: usize = 0;
const LHS_INPUT_VECTOR: usize = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PublicLookupStage1Family {
    Gy,
    V,
    Vx,
    Gate1,
    Randomized,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PublicLookupStage1Key {
    family: PublicLookupStage1Family,
    inner_chunk_idx: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublicLookupStage1Task {
    // Computes the chunk-local identity term c_b0 * I_k. This task already produces a
    // final addend for output chunk k, so it does not need any stage-2 rhs expansion.
    DirectIdentity { output_chunk_idx: usize },
    // Computes a reusable left-hand intermediate for family j. Stage 2 will later multiply
    // this value by the family-specific rhs chunk for each output chunk k.
    Intermediate { key: PublicLookupStage1Key },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PublicLookupStage2Task {
    key: PublicLookupStage1Key,
    output_chunk_idx: usize,
}

struct PublicLookupStage2Contribution {
    device_slot: usize,
    family: PublicLookupStage1Family,
    output_chunk_idx: usize,
    bytes: Arc<[u8]>,
    key: PublicLookupStage1Key,
    load_ms: f64,
    compute_ms: f64,
}

struct LoadedPublicLookupStage1Task<M: PolyMatrix> {
    device_slot: usize,
    task: PublicLookupStage1Task,
    rhs_chunk: M,
    load_ms: f64,
}

struct ComputedPublicLookupStage1Task {
    device_slot: usize,
    task: PublicLookupStage1Task,
    bytes: Arc<[u8]>,
    load_ms: f64,
    compute_ms: f64,
}

struct LoadedPublicLookupStage2Task<M: PolyMatrix> {
    device_slot: usize,
    task: PublicLookupStage2Task,
    stage1_mid: M,
    rhs_chunk: M,
    load_ms: f64,
}

struct PublicLookupStage2WaveDeviceContext<'a, M: PolyMatrix> {
    device_slot: usize,
    stage1_cache: &'a mut HashMap<PublicLookupStage1Key, M>,
    gy_rhs_cache: &'a mut HashMap<usize, M>,
    v_rhs_cache: &'a mut HashMap<usize, M>,
    vx_rhs_cache: &'a mut HashMap<usize, M>,
    lut_rhs_cache: &'a mut HashMap<usize, M>,
    batch: &'a Vec<PublicLookupStage2Task>,
    shared_dev: &'a GpuPublicLookupSharedByDevice<M>,
    metadata: &'a (M, String, String),
    x: &'a M::P,
}

#[derive(Clone, Copy, Debug, Default)]
struct PublicLookupTaskGroupBenchStats {
    task_count: usize,
    max_load_ms: f64,
    max_compute_ms: f64,
    max_store_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PublicLookupStage1BenchGroup {
    DirectIdentity,
    Intermediate(PublicLookupStage1Family),
}

pub(super) fn effective_gpu_slot_parallelism(requested: usize) -> usize {
    requested.min(public_lookup_gpu_device_ids().len().max(1)).max(1)
}

pub(super) fn prepare_public_lookup_shared_by_device<M>(
    params: &<M::P as Poly>::Params,
    shared: &GGH15PublicLookupSharedState<M>,
    slot_parallelism: usize,
) -> Vec<GpuPublicLookupSharedByDevice<M>>
where
    M: PolyMatrix,
{
    let gadget_matrix_bytes = Arc::<[u8]>::from(shared.gadget_matrix.as_ref().to_compact_bytes());
    let out_pubkey_matrix_bytes = Arc::<[u8]>::from(shared.out_pubkey.matrix.to_compact_bytes());
    let effective_parallelism = effective_gpu_slot_parallelism(slot_parallelism);

    public_lookup_gpu_device_ids()
        .into_iter()
        .take(effective_parallelism)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|device_id| {
            let local_params = params.params_for_device(device_id);
            GpuPublicLookupSharedByDevice {
                device_id,
                params: local_params.clone(),
                shared: GGH15PublicLookupSharedState {
                    d: shared.d,
                    m_g: shared.m_g,
                    gadget_matrix: Arc::new(M::from_compact_bytes(
                        &local_params,
                        gadget_matrix_bytes.as_ref(),
                    )),
                    preimage_gate1_id_prefix: shared.preimage_gate1_id_prefix.clone(),
                    preimage_gate2_identity_id_prefix: shared
                        .preimage_gate2_identity_id_prefix
                        .clone(),
                    preimage_gate2_gy_id_prefix: shared.preimage_gate2_gy_id_prefix.clone(),
                    preimage_gate2_v_id_prefix: shared.preimage_gate2_v_id_prefix.clone(),
                    preimage_gate2_vx_id_prefix: shared.preimage_gate2_vx_id_prefix.clone(),
                    out_pubkey: BggPublicKey {
                        matrix: M::from_compact_bytes(
                            &local_params,
                            out_pubkey_matrix_bytes.as_ref(),
                        ),
                        reveal_plaintext: shared.out_pubkey.reveal_plaintext,
                    },
                },
            }
        })
        .collect()
}

// Greedily assigns tasks to devices while balancing load first and affinity second.
// Each task is mapped to an affinity key K. We always place the next task on a device
// with the current minimum task count. If the key has already been seen and its previous
// device is still tied for the minimum load, we reuse that device to keep related tasks
// together. Otherwise, we pick the first minimum-load device and record it as the new
// preferred device for that key. This keeps all devices busy while still biasing equal-key
// tasks toward the same device when that does not reduce parallelism.
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

fn stage1_task_lhs_family(task: &PublicLookupStage1Task) -> usize {
    match task {
        PublicLookupStage1Task::DirectIdentity { .. } => LHS_CB0,
        PublicLookupStage1Task::Intermediate { key } => {
            if key.family == PublicLookupStage1Family::Randomized {
                LHS_INPUT_VECTOR
            } else {
                LHS_CB0
            }
        }
    }
}

fn stage1_task_bench_group(task: &PublicLookupStage1Task) -> PublicLookupStage1BenchGroup {
    match task {
        PublicLookupStage1Task::DirectIdentity { .. } => {
            PublicLookupStage1BenchGroup::DirectIdentity
        }
        PublicLookupStage1Task::Intermediate { key } => {
            PublicLookupStage1BenchGroup::Intermediate(key.family)
        }
    }
}

fn maybe_elapsed_ms(started: Option<Instant>) -> f64 {
    started.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0)
}

fn update_group_store_timing<K>(
    group_stats: &mut HashMap<K, PublicLookupTaskGroupBenchStats>,
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
    group_stats: &HashMap<K, PublicLookupTaskGroupBenchStats>,
    stage_parallelism: usize,
) -> PublicLookupStageBenchMeasurement
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
    PublicLookupStageBenchMeasurement {
        latency,
        max_parallelism: stage_parallelism as u128,
        total_time,
    }
}

fn load_public_lookup_slot<M>(
    slot_idx: usize,
    slot_count: usize,
    gate_id: GateId,
    lut_id: usize,
    completed_slots: &AtomicUsize,
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    c_b0_compact_bytes: &Arc<[u8]>,
    input_vector_compact_bytes: &Arc<[u8]>,
    x_compact_bytes: &Arc<[u8]>,
) -> LoadedPublicLookupSlot<M>
where
    M: PolyMatrix + Send + Sync,
    M::P: Send + Sync,
{
    let slot_started = Instant::now();
    let load_started = Instant::now();
    let loaded_by_device = shared_by_device
        .par_iter()
        .map(|shared_dev| {
            (
                M::from_compact_bytes(&shared_dev.params, c_b0_compact_bytes.as_ref()),
                M::from_compact_bytes(&shared_dev.params, input_vector_compact_bytes.as_ref()),
                M::P::from_compact_bytes(&shared_dev.params, x_compact_bytes.as_ref()),
            )
        })
        .collect::<Vec<_>>();
    let mut c_b0_by_device = Vec::with_capacity(loaded_by_device.len());
    let mut input_vector_by_device = Vec::with_capacity(loaded_by_device.len());
    let mut x_by_device = Vec::with_capacity(loaded_by_device.len());
    for (c_b0, input_vector, x) in loaded_by_device {
        c_b0_by_device.push(c_b0);
        input_vector_by_device.push(input_vector);
        x_by_device.push(x);
    }
    let loaded_slot = LoadedPublicLookupSlot {
        slot_idx,
        slot_started,
        load_ms: load_started.elapsed().as_secs_f64() * 1000.0,
        c_b0_by_device,
        input_vector_by_device,
        x_by_device,
    };
    debug!(
        "GGH15 BGG poly-encoding gpu slot loaded: gate_id={}, lut_id={}, slot={}, completed_before={}/{}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        slot_idx,
        completed_slots.load(Ordering::Relaxed),
        slot_count,
        loaded_slot.load_ms
    );
    loaded_slot
}

fn compute_public_lookup_slot<M, HS>(
    slot: LoadedPublicLookupSlot<M>,
    gate_id: GateId,
    lut_id: usize,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    shared_by_device: &[GpuPublicLookupSharedByDevice<M>],
    slot_bench_output: Option<&mut PublicLookupSlotBenchMeasurement>,
) -> ComputedPublicLookupSlot<M>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let stage_started = Instant::now();
    let collect_bench = slot_bench_output.is_some();
    let mut slot_bench_measurement = PublicLookupSlotBenchMeasurement::default();
    let stage0_started = Instant::now();
    let base_params = &shared_by_device[0].params;
    let x = &slot.x_by_device[0];
    let x_u64 = x.const_coeff_u64();
    let (_, y) = plt
        .get(base_params, x_u64)
        .unwrap_or_else(|| panic!("{:?} not found in LUT for gate {}", x_u64, gate_id));
    let plaintext_bytes =
        Arc::<[u8]>::from(M::P::from_elem_to_constant(base_params, &y).to_compact_bytes());
    let gate1_total_cols =
        trapdoor_public_column_count::<M>(base_params, shared_by_device[0].shared.d);
    let output_chunk_count = column_chunk_count(shared_by_device[0].shared.m_g);
    let gate1_chunk_count = column_chunk_count(gate1_total_cols);
    let stage0_ms = stage0_started.elapsed().as_secs_f64() * 1000.0;

    // Stage 0: fix the slot-local LUT output and matrix shapes that every later stage reuses.
    let stage1_started = Instant::now();
    let mut stage1_tasks = Vec::new();
    stage1_tasks.extend(
        (0..output_chunk_count)
            .map(|output_chunk_idx| PublicLookupStage1Task::DirectIdentity { output_chunk_idx }),
    );
    for family in [
        PublicLookupStage1Family::Gy,
        PublicLookupStage1Family::V,
        PublicLookupStage1Family::Vx,
        PublicLookupStage1Family::Gate1,
        PublicLookupStage1Family::Randomized,
    ] {
        let chunk_count = if family == PublicLookupStage1Family::Gate1 {
            gate1_chunk_count
        } else {
            output_chunk_count
        };
        stage1_tasks.extend((0..chunk_count).map(|inner_chunk_idx| {
            PublicLookupStage1Task::Intermediate {
                key: PublicLookupStage1Key { family, inner_chunk_idx },
            }
        }));
    }
    let mut stage1_group_stats = if collect_bench {
        let mut stats =
            HashMap::<PublicLookupStage1BenchGroup, PublicLookupTaskGroupBenchStats>::new();
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
    let u_g_id = format!("ggh15_lut_u_g_matrix_{}", gate_id);
    let load_stage1_wave = |wave_idx: usize| {
        let loaded_wave = stage1_batches
            .par_iter()
            .zip(shared_by_device.par_iter())
            .enumerate()
            .filter_map(|(device_slot, (batch, shared_dev))| {
                let task = batch.get(wave_idx).copied()?;
                let task_label = match task {
                    PublicLookupStage1Task::DirectIdentity { output_chunk_idx } => {
                        format!("direct_identity(output_chunk={output_chunk_idx})")
                    }
                    PublicLookupStage1Task::Intermediate { key } => format!(
                        "intermediate(family={:?}, inner_chunk={})",
                        key.family, key.inner_chunk_idx
                    ),
                };
                let load_started = collect_bench.then(Instant::now);
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage1 load start: gate_id={}, lut_id={}, slot={}, wave={}, device_slot={}, device_id={}, task={}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    wave_idx,
                    device_slot,
                    shared_dev.device_id,
                    task_label
                );
                let rhs_chunk = match task {
                    PublicLookupStage1Task::DirectIdentity { output_chunk_idx } => {
                        read_matrix_column_chunk(
                            &shared_dev.params,
                            dir,
                            &shared_dev.shared.preimage_gate2_identity_id_prefix,
                            shared_dev.shared.m_g,
                            output_chunk_idx,
                            "preimage_gate2_identity",
                        )
                    }
                    PublicLookupStage1Task::Intermediate { key } => match key.family {
                        PublicLookupStage1Family::Gy => read_matrix_column_chunk(
                            &shared_dev.params,
                            dir,
                            &shared_dev.shared.preimage_gate2_gy_id_prefix,
                            shared_dev.shared.m_g,
                            key.inner_chunk_idx,
                            "preimage_gate2_gy",
                        ),
                        PublicLookupStage1Family::V => read_matrix_column_chunk(
                            &shared_dev.params,
                            dir,
                            &shared_dev.shared.preimage_gate2_v_id_prefix,
                            shared_dev.shared.m_g,
                            key.inner_chunk_idx,
                            "preimage_gate2_v",
                        ),
                        PublicLookupStage1Family::Vx => read_matrix_column_chunk(
                            &shared_dev.params,
                            dir,
                            &shared_dev.shared.preimage_gate2_vx_id_prefix,
                            shared_dev.shared.m_g,
                            key.inner_chunk_idx,
                            "preimage_gate2_vx",
                        ),
                        PublicLookupStage1Family::Gate1 => read_matrix_column_chunk(
                            &shared_dev.params,
                            dir,
                            &shared_dev.shared.preimage_gate1_id_prefix,
                            gate1_total_cols,
                            key.inner_chunk_idx,
                            "preimage_gate1",
                        ),
                        PublicLookupStage1Family::Randomized => {
                            let (inner_start, inner_len) =
                                column_chunk_bounds(shared_dev.shared.m_g, key.inner_chunk_idx);
                            HS::new().sample_hash_decomposed_columns(
                                &shared_dev.params,
                                hash_key,
                                u_g_id.clone(),
                                shared_dev.shared.d,
                                shared_dev.shared.m_g,
                                inner_start,
                                inner_len,
                                DistType::FinRingDist,
                            )
                        }
                    },
                };
                let load_ms = maybe_elapsed_ms(load_started);
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage1 load complete: gate_id={}, lut_id={}, slot={}, wave={}, device_slot={}, device_id={}, task={}, rhs_rows={}, rhs_cols={}, load_ms={:.3}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    wave_idx,
                    device_slot,
                    shared_dev.device_id,
                    task_label,
                    rhs_chunk.row_size(),
                    rhs_chunk.col_size(),
                    load_ms
                );
                Some(LoadedPublicLookupStage1Task {
                    device_slot,
                    task,
                    rhs_chunk,
                    load_ms,
                })
            })
            .collect::<Vec<_>>();
        (!loaded_wave.is_empty()).then_some(loaded_wave)
    };
    let compute_stage1_wave = |loaded_wave: Vec<LoadedPublicLookupStage1Task<M>>| {
        loaded_wave
            .into_par_iter()
            .map(|loaded_task| {
                let task_label = match loaded_task.task {
                    PublicLookupStage1Task::DirectIdentity { output_chunk_idx } => {
                        format!("direct_identity(output_chunk={output_chunk_idx})")
                    }
                    PublicLookupStage1Task::Intermediate { key } => format!(
                        "intermediate(family={:?}, inner_chunk={})",
                        key.family, key.inner_chunk_idx
                    ),
                };
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage1 compute start: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    loaded_task.device_slot,
                    shared_by_device[loaded_task.device_slot].device_id,
                    task_label
                );
                let compute_started = collect_bench.then(Instant::now);
                let lhs = match loaded_task.task {
                    PublicLookupStage1Task::DirectIdentity { .. } => {
                        &slot.c_b0_by_device[loaded_task.device_slot]
                    }
                    PublicLookupStage1Task::Intermediate { key } => {
                        if key.family == PublicLookupStage1Family::Randomized {
                            &slot.input_vector_by_device[loaded_task.device_slot]
                        } else {
                            &slot.c_b0_by_device[loaded_task.device_slot]
                        }
                    }
                };
                let output = lhs * &loaded_task.rhs_chunk;
                let output_bytes = output.into_compact_bytes();
                let compute_ms = maybe_elapsed_ms(compute_started);
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage1 compute complete: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}, output_bytes={}, compute_ms={:.3}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    loaded_task.device_slot,
                    shared_by_device[loaded_task.device_slot].device_id,
                    task_label,
                    output_bytes.len(),
                    compute_ms
                );
                ComputedPublicLookupStage1Task {
                    device_slot: loaded_task.device_slot,
                    task: loaded_task.task,
                    bytes: Arc::<[u8]>::from(output_bytes),
                    load_ms: loaded_task.load_ms,
                    compute_ms,
                }
            })
            .collect::<Vec<_>>()
    };
    let mut store_stage1_wave =
        |device_outputs: Vec<ComputedPublicLookupStage1Task>,
         direct_identity_by_chunk: &mut Vec<Option<Arc<[u8]>>>,
         stage1_results: &mut HashMap<PublicLookupStage1Key, Arc<[u8]>>| {
            for output in device_outputs {
                let store_started = collect_bench.then(Instant::now);
                let task = output.task;
                let task_label = match task {
                    PublicLookupStage1Task::DirectIdentity { output_chunk_idx } => {
                        format!("direct_identity(output_chunk={output_chunk_idx})")
                    }
                    PublicLookupStage1Task::Intermediate { key } => format!(
                        "intermediate(family={:?}, inner_chunk={})",
                        key.family, key.inner_chunk_idx
                    ),
                };
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage1 store start: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    output.device_slot,
                    shared_by_device[output.device_slot].device_id,
                    task_label
                );
                let bytes = output.bytes;
                match task {
                    PublicLookupStage1Task::DirectIdentity { output_chunk_idx } => {
                        direct_identity_by_chunk[output_chunk_idx] = Some(bytes);
                    }
                    PublicLookupStage1Task::Intermediate { key } => {
                        stage1_results.insert(key, bytes);
                    }
                }
                let store_ms = maybe_elapsed_ms(store_started);
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage1 store complete: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}, store_ms={:.3}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    output.device_slot,
                    shared_by_device[output.device_slot].device_id,
                    task_label,
                    store_ms
                );
                if collect_bench {
                    update_group_store_timing(
                        &mut stage1_group_stats,
                        stage1_task_bench_group(&task),
                        output.load_ms,
                        output.compute_ms,
                        store_ms,
                    );
                }
            }
        };
    // Stage 1: pipeline loading the next lhs/rhs wave, computing the current GPU wave,
    // and storing the previous wave back into the slot-local stage1 caches.
    let mut direct_identity_by_chunk = vec![None; output_chunk_count];
    let mut stage1_results = HashMap::new();
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
                    || {
                        if should_load_next { load_stage1_wave(next_stage1_wave_idx) } else { None }
                    },
                )
            },
            || {
                if let Some(device_outputs) = previous_outputs_to_store {
                    store_stage1_wave(
                        device_outputs,
                        &mut direct_identity_by_chunk,
                        &mut stage1_results,
                    );
                }
            },
        );
        previous_stage1_outputs = Some(computed_outputs);
        current_stage1_wave = next_loaded_wave;
        if should_load_next {
            next_stage1_wave_idx += 1;
        }
    }
    if let Some(device_outputs) = previous_stage1_outputs.take() {
        store_stage1_wave(device_outputs, &mut direct_identity_by_chunk, &mut stage1_results);
    }
    for output_chunk_idx in 0..output_chunk_count {
        assert!(
            direct_identity_by_chunk[output_chunk_idx].is_some(),
            "missing stage1 direct identity for output chunk {}",
            output_chunk_idx
        );
    }
    let stage1_ms = stage1_started.elapsed().as_secs_f64() * 1000.0;
    if collect_bench {
        let stage1_bench =
            finalize_stage_bench(&stage1_group_stats, stage1_batches.iter().map(Vec::len).sum());
        slot_bench_measurement.latency += stage1_bench.latency;
        slot_bench_measurement.max_parallelism =
            slot_bench_measurement.max_parallelism.max(stage1_bench.max_parallelism);
        slot_bench_measurement.total_time += stage1_bench.total_time;
    }

    let stage2_started = Instant::now();
    let mut stage2_tasks = Vec::new();
    for family in [
        PublicLookupStage1Family::Gy,
        PublicLookupStage1Family::V,
        PublicLookupStage1Family::Vx,
        PublicLookupStage1Family::Gate1,
        PublicLookupStage1Family::Randomized,
    ] {
        let inner_chunk_count = if family == PublicLookupStage1Family::Gate1 {
            gate1_chunk_count
        } else {
            output_chunk_count
        };
        for inner_chunk_idx in 0..inner_chunk_count {
            let key = PublicLookupStage1Key { family, inner_chunk_idx };
            stage2_tasks.extend(
                (0..output_chunk_count)
                    .map(|output_chunk_idx| PublicLookupStage2Task { key, output_chunk_idx }),
            );
        }
    }
    let mut stage2_group_stats = if collect_bench {
        let mut stats = HashMap::<PublicLookupStage1Key, PublicLookupTaskGroupBenchStats>::new();
        for task in &stage2_tasks {
            stats.entry(task.key).or_default().task_count += 1;
        }
        stats
    } else {
        HashMap::new()
    };
    let stage2_batches =
        assign_tasks_with_affinity(stage2_tasks, shared_by_device.len(), |task| task.key);
    let stage2_wave_count = stage2_batches.iter().map(Vec::len).max().unwrap_or(0);
    let stage2_metadata = slot
        .x_by_device
        .par_iter()
        .zip(shared_by_device.par_iter())
        .map(|(device_x, shared_dev)| {
            let device_x_u64 = device_x.const_coeff_u64();
            let (lut_row_idx, y_elem) =
                plt.get(&shared_dev.params, device_x_u64).unwrap_or_else(|| {
                    panic!("{:?} not found in LUT for gate {}", device_x_u64, gate_id)
                });
            let y_poly = M::P::from_elem_to_constant(&shared_dev.params, &y_elem);
            let gy = shared_dev.shared.gadget_matrix.as_ref().clone() * y_poly;
            let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_{}", lut_id);
            let lut_aux_row_id = format!(
                "{lut_aux_prefix}_idx{}",
                usize::try_from(lut_row_idx).expect("LUT row index must fit in usize")
            );
            let v_tag = format!(
                "ggh15_lut_v_idx_{}_{}",
                lut_id,
                usize::try_from(lut_row_idx).expect("LUT row index must fit in usize")
            );
            (gy, lut_aux_row_id, v_tag)
        })
        .collect::<Vec<_>>();
    let mut stage1_cache_by_device = (0..shared_by_device.len())
        .map(|_| HashMap::<PublicLookupStage1Key, M>::new())
        .collect::<Vec<_>>();
    let mut gy_rhs_cache_by_device =
        (0..shared_by_device.len()).map(|_| HashMap::<usize, M>::new()).collect::<Vec<_>>();
    let mut v_rhs_cache_by_device =
        (0..shared_by_device.len()).map(|_| HashMap::<usize, M>::new()).collect::<Vec<_>>();
    let mut vx_rhs_cache_by_device =
        (0..shared_by_device.len()).map(|_| HashMap::<usize, M>::new()).collect::<Vec<_>>();
    let mut lut_rhs_cache_by_device =
        (0..shared_by_device.len()).map(|_| HashMap::<usize, M>::new()).collect::<Vec<_>>();
    let load_stage2_wave =
        |wave_idx: usize,
         stage1_cache_by_device: &mut Vec<HashMap<PublicLookupStage1Key, M>>,
         gy_rhs_cache_by_device: &mut Vec<HashMap<usize, M>>,
         v_rhs_cache_by_device: &mut Vec<HashMap<usize, M>>,
         vx_rhs_cache_by_device: &mut Vec<HashMap<usize, M>>,
         lut_rhs_cache_by_device: &mut Vec<HashMap<usize, M>>| {
            let mut stage1_cache_iter = stage1_cache_by_device.iter_mut();
            let mut gy_rhs_cache_iter = gy_rhs_cache_by_device.iter_mut();
            let mut v_rhs_cache_iter = v_rhs_cache_by_device.iter_mut();
            let mut vx_rhs_cache_iter = vx_rhs_cache_by_device.iter_mut();
            let mut lut_rhs_cache_iter = lut_rhs_cache_by_device.iter_mut();

            let per_device_contexts = stage2_batches
                .iter()
                .zip(shared_by_device.iter())
                .zip(stage2_metadata.iter())
                .zip(slot.x_by_device.iter())
                .enumerate()
                .map(|(device_slot, (((batch, shared_dev), metadata), x))| {
                    PublicLookupStage2WaveDeviceContext {
                        device_slot,
                        stage1_cache: stage1_cache_iter
                            .next()
                            .expect("missing stage1 cache for device"),
                        gy_rhs_cache: gy_rhs_cache_iter
                            .next()
                            .expect("missing gy rhs cache for device"),
                        v_rhs_cache: v_rhs_cache_iter
                            .next()
                            .expect("missing v rhs cache for device"),
                        vx_rhs_cache: vx_rhs_cache_iter
                            .next()
                            .expect("missing vx rhs cache for device"),
                        lut_rhs_cache: lut_rhs_cache_iter
                            .next()
                            .expect("missing lut rhs cache for device"),
                        batch,
                        shared_dev,
                        metadata,
                        x,
                    }
                })
                .collect::<Vec<_>>();
            let loaded_wave = per_device_contexts
                .into_par_iter()
                .filter_map(|ctx| {
                    let task = ctx.batch.get(wave_idx).copied()?;
                    let task_label = format!(
                        "family={:?}, inner_chunk={}, output_chunk={}",
                        task.key.family, task.key.inner_chunk_idx, task.output_chunk_idx
                    );
                    let load_started = collect_bench.then(Instant::now);
                    debug!(
                        "GGH15 BGG poly-encoding gpu public-lookup stage2 load start: gate_id={}, lut_id={}, slot={}, wave={}, device_slot={}, device_id={}, task={}",
                        gate_id,
                        lut_id,
                        slot.slot_idx,
                        wave_idx,
                        ctx.device_slot,
                        ctx.shared_dev.device_id,
                        task_label
                    );
                    let stage1_mid = ctx
                        .stage1_cache
                        .entry(task.key)
                        .or_insert_with(|| {
                            M::from_compact_bytes(
                                &ctx.shared_dev.params,
                                stage1_results
                                    .get(&task.key)
                                    .unwrap_or_else(|| {
                                        panic!("missing stage1 result for {:?}", task.key)
                                    })
                                    .as_ref(),
                            )
                        })
                        .clone();
                    let rhs_full = match task.key.family {
                        PublicLookupStage1Family::Gy => {
                            ctx.gy_rhs_cache.entry(task.output_chunk_idx).or_insert_with(|| {
                                let (col_start, col_len) = column_chunk_bounds(
                                    ctx.shared_dev.shared.m_g,
                                    task.output_chunk_idx,
                                );
                                ctx.metadata
                                    .0
                                    .slice_columns(col_start, col_start + col_len)
                                    .decompose()
                            })
                        }
                        PublicLookupStage1Family::V | PublicLookupStage1Family::Randomized => {
                            ctx.v_rhs_cache.entry(task.output_chunk_idx).or_insert_with(|| {
                                let (col_start, col_len) = column_chunk_bounds(
                                    ctx.shared_dev.shared.m_g,
                                    task.output_chunk_idx,
                                );
                                HS::new().sample_hash_decomposed_columns(
                                    &ctx.shared_dev.params,
                                    hash_key,
                                    ctx.metadata.2.clone(),
                                    ctx.shared_dev.shared.d,
                                    ctx.shared_dev.shared.m_g,
                                    col_start,
                                    col_len,
                                    DistType::FinRingDist,
                                )
                            })
                        }
                        PublicLookupStage1Family::Vx => {
                            ctx.vx_rhs_cache.entry(task.output_chunk_idx).or_insert_with(|| {
                                let v_rhs = ctx
                                    .v_rhs_cache
                                    .entry(task.output_chunk_idx)
                                    .or_insert_with(|| {
                                        let (col_start, col_len) = column_chunk_bounds(
                                            ctx.shared_dev.shared.m_g,
                                            task.output_chunk_idx,
                                        );
                                        HS::new().sample_hash_decomposed_columns(
                                            &ctx.shared_dev.params,
                                            hash_key,
                                            ctx.metadata.2.clone(),
                                            ctx.shared_dev.shared.d,
                                            ctx.shared_dev.shared.m_g,
                                            col_start,
                                            col_len,
                                            DistType::FinRingDist,
                                        )
                                    })
                                    .clone();
                                v_rhs * ctx.x.clone()
                            })
                        }
                        PublicLookupStage1Family::Gate1 => {
                            ctx.lut_rhs_cache.entry(task.output_chunk_idx).or_insert_with(|| {
                                read_matrix_column_chunk(
                                    &ctx.shared_dev.params,
                                    dir,
                                    &ctx.metadata.1,
                                    ctx.shared_dev.shared.m_g,
                                    task.output_chunk_idx,
                                    "preimage_lut",
                                )
                            })
                        }
                    };
                    let (inner_start, inner_len) = match task.key.family {
                        PublicLookupStage1Family::Gate1 => {
                            column_chunk_bounds(gate1_total_cols, task.key.inner_chunk_idx)
                        }
                        PublicLookupStage1Family::Gy |
                        PublicLookupStage1Family::V |
                        PublicLookupStage1Family::Vx |
                        PublicLookupStage1Family::Randomized => {
                            column_chunk_bounds(ctx.shared_dev.shared.m_g, task.key.inner_chunk_idx)
                        }
                    };
                    let rhs_chunk = rhs_full.slice(
                        inner_start,
                        inner_start + inner_len,
                        0,
                        rhs_full.col_size(),
                    );
                    let load_ms = maybe_elapsed_ms(load_started);
                    debug!(
                        "GGH15 BGG poly-encoding gpu public-lookup stage2 load complete: gate_id={}, lut_id={}, slot={}, wave={}, device_slot={}, device_id={}, task={}, stage1_mid_rows={}, stage1_mid_cols={}, rhs_rows={}, rhs_cols={}, load_ms={:.3}",
                        gate_id,
                        lut_id,
                        slot.slot_idx,
                        wave_idx,
                        ctx.device_slot,
                        ctx.shared_dev.device_id,
                        task_label,
                        stage1_mid.row_size(),
                        stage1_mid.col_size(),
                        rhs_chunk.row_size(),
                        rhs_chunk.col_size(),
                        load_ms
                    );
                    Some(LoadedPublicLookupStage2Task {
                        device_slot: ctx.device_slot,
                        task,
                        stage1_mid,
                        rhs_chunk,
                        load_ms,
                    })
                })
                .collect::<Vec<_>>();
            (!loaded_wave.is_empty()).then_some(loaded_wave)
        };
    let compute_stage2_wave = |loaded_wave: Vec<LoadedPublicLookupStage2Task<M>>| {
        loaded_wave
            .into_par_iter()
            .map(|loaded_task| {
                let task_label = format!(
                    "family={:?}, inner_chunk={}, output_chunk={}",
                    loaded_task.task.key.family,
                    loaded_task.task.key.inner_chunk_idx,
                    loaded_task.task.output_chunk_idx
                );
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage2 compute start: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    loaded_task.device_slot,
                    shared_by_device[loaded_task.device_slot].device_id,
                    task_label
                );
                let compute_started = collect_bench.then(Instant::now);
                let contribution = &loaded_task.stage1_mid * &loaded_task.rhs_chunk;
                let contribution_bytes = contribution.into_compact_bytes();
                let compute_ms = maybe_elapsed_ms(compute_started);
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage2 compute complete: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}, output_bytes={}, compute_ms={:.3}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    loaded_task.device_slot,
                    shared_by_device[loaded_task.device_slot].device_id,
                    task_label,
                    contribution_bytes.len(),
                    compute_ms
                );
                PublicLookupStage2Contribution {
                    device_slot: loaded_task.device_slot,
                    family: loaded_task.task.key.family,
                    output_chunk_idx: loaded_task.task.output_chunk_idx,
                    key: loaded_task.task.key,
                    bytes: Arc::<[u8]>::from(contribution_bytes),
                    load_ms: loaded_task.load_ms,
                    compute_ms,
                }
            })
            .collect::<Vec<_>>()
    };
    let mut store_stage2_wave =
        |contributions: Vec<PublicLookupStage2Contribution>,
         reduced_chunks: &mut Vec<Option<M>>| {
            for contribution in contributions {
                let store_started = collect_bench.then(Instant::now);
                let task_label = format!(
                    "family={:?}, inner_chunk={}, output_chunk={}",
                    contribution.key.family,
                    contribution.key.inner_chunk_idx,
                    contribution.output_chunk_idx
                );
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage2 store start: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    contribution.device_slot,
                    shared_by_device[contribution.device_slot].device_id,
                    task_label
                );
                let contribution_matrix =
                    M::from_compact_bytes(&shared_by_device[0].params, contribution.bytes.as_ref());
                let accum =
                    reduced_chunks[contribution.output_chunk_idx].get_or_insert_with(|| {
                        M::zero(
                            &shared_by_device[0].params,
                            contribution_matrix.row_size(),
                            contribution_matrix.col_size(),
                        )
                    });
                match contribution.family {
                    PublicLookupStage1Family::Gate1 => {
                        let neg_contribution = M::zero(
                            &shared_by_device[0].params,
                            contribution_matrix.row_size(),
                            contribution_matrix.col_size(),
                        ) - contribution_matrix;
                        accum.add_in_place(&neg_contribution);
                    }
                    PublicLookupStage1Family::Gy |
                    PublicLookupStage1Family::V |
                    PublicLookupStage1Family::Vx |
                    PublicLookupStage1Family::Randomized => {
                        accum.add_in_place(&contribution_matrix);
                    }
                }
                let store_ms = maybe_elapsed_ms(store_started);
                debug!(
                    "GGH15 BGG poly-encoding gpu public-lookup stage2 store complete: gate_id={}, lut_id={}, slot={}, device_slot={}, device_id={}, task={}, accum_rows={}, accum_cols={}, store_ms={:.3}",
                    gate_id,
                    lut_id,
                    slot.slot_idx,
                    contribution.device_slot,
                    shared_by_device[contribution.device_slot].device_id,
                    task_label,
                    accum.row_size(),
                    accum.col_size(),
                    store_ms
                );
                if collect_bench {
                    update_group_store_timing(
                        &mut stage2_group_stats,
                        contribution.key,
                        contribution.load_ms,
                        contribution.compute_ms,
                        store_ms,
                    );
                }
            }
        };
    // Stage 2: pipeline loading the next rhs wave, computing the current GPU wave, and
    // storing the previous wave into the signed output-chunk accumulators.
    let mut reduced_chunks = (0..output_chunk_count)
        .map(|output_chunk_idx| {
            direct_identity_by_chunk[output_chunk_idx]
                .as_ref()
                .map(|bytes| M::from_compact_bytes(&shared_by_device[0].params, bytes.as_ref()))
        })
        .collect::<Vec<_>>();
    let mut next_stage2_wave_idx = 1usize;
    let mut current_stage2_wave = if stage2_wave_count == 0 {
        None
    } else {
        load_stage2_wave(
            0,
            &mut stage1_cache_by_device,
            &mut gy_rhs_cache_by_device,
            &mut v_rhs_cache_by_device,
            &mut vx_rhs_cache_by_device,
            &mut lut_rhs_cache_by_device,
        )
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
                            load_stage2_wave(
                                next_stage2_wave_idx,
                                &mut stage1_cache_by_device,
                                &mut gy_rhs_cache_by_device,
                                &mut v_rhs_cache_by_device,
                                &mut vx_rhs_cache_by_device,
                                &mut lut_rhs_cache_by_device,
                            )
                        } else {
                            None
                        }
                    },
                )
            },
            || {
                if let Some(contributions) = previous_outputs_to_store {
                    store_stage2_wave(contributions, &mut reduced_chunks);
                }
            },
        );
        previous_stage2_outputs = Some(computed_outputs);
        current_stage2_wave = next_loaded_wave;
        if should_load_next {
            next_stage2_wave_idx += 1;
        }
    }
    if let Some(contributions) = previous_stage2_outputs.take() {
        store_stage2_wave(contributions, &mut reduced_chunks);
    }
    // Stage 3: after the staged signed reduction, each reduced chunk already equals the
    // final public-lookup block for this slot.
    let stage3_started = Instant::now();
    let reduced_chunks = reduced_chunks
        .into_iter()
        .enumerate()
        .map(|(output_chunk_idx, accum)| {
            accum.unwrap_or_else(|| panic!("missing reduced output chunk {output_chunk_idx}"))
        })
        .collect::<Vec<_>>();
    let stage2_ms = stage2_started.elapsed().as_secs_f64() * 1000.0;
    if collect_bench {
        let stage2_bench =
            finalize_stage_bench(&stage2_group_stats, stage2_batches.iter().map(Vec::len).sum());
        slot_bench_measurement.latency += stage2_bench.latency;
        slot_bench_measurement.max_parallelism =
            slot_bench_measurement.max_parallelism.max(stage2_bench.max_parallelism);
        slot_bench_measurement.total_time += stage2_bench.total_time;
    }
    let stage3_ms = stage3_started.elapsed().as_secs_f64() * 1000.0;
    if collect_bench {
        slot_bench_measurement.latency += stage3_ms;
        slot_bench_measurement.max_parallelism = slot_bench_measurement.max_parallelism.max(1);
        slot_bench_measurement.total_time += stage3_ms;
    }

    // Stage 4: concatenate the reduced chunks into the final device-local output vector.
    let stage4_started = Instant::now();
    let output_vector = if reduced_chunks.len() == 1 {
        reduced_chunks
            .into_iter()
            .next()
            .expect("public-lookup output chunk list must be non-empty")
    } else {
        let mut iter = reduced_chunks.into_iter();
        let first = iter.next().expect("public-lookup output chunk list must be non-empty");
        first.concat_columns_owned(iter.collect())
    };
    let stage4_ms = stage4_started.elapsed().as_secs_f64() * 1000.0;
    if collect_bench {
        slot_bench_measurement.latency += stage4_ms;
        slot_bench_measurement.max_parallelism = slot_bench_measurement.max_parallelism.max(1);
        slot_bench_measurement.total_time += stage4_ms;
    }
    if collect_bench {
        slot_bench_measurement.latency += stage0_ms;
        slot_bench_measurement.max_parallelism = slot_bench_measurement.max_parallelism.max(1);
        slot_bench_measurement.total_time += stage0_ms;
    }
    if let Some(slot_bench_out) = slot_bench_output {
        *slot_bench_out = slot_bench_measurement;
    }

    ComputedPublicLookupSlot {
        slot_idx: slot.slot_idx,
        slot_started: slot.slot_started,
        load_ms: slot.load_ms,
        plaintext_bytes,
        output_vector,
        stage1_ms,
        stage2_ms,
        compute_ms: stage_started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn store_public_lookup_slot<M>(
    computed_slot: ComputedPublicLookupSlot<M>,
    slot_count: usize,
    gate_id: GateId,
    lut_id: usize,
    completed_slots: &AtomicUsize,
) -> (Arc<[u8]>, Arc<[u8]>)
where
    M: PolyMatrix,
{
    let serialize_started = Instant::now();
    let vector_bytes = Arc::<[u8]>::from(computed_slot.output_vector.into_compact_bytes());
    let serialize_ms = serialize_started.elapsed().as_secs_f64() * 1000.0;
    let completed_slot_count = completed_slots.fetch_add(1, Ordering::Relaxed) + 1;
    debug_slot_stage_timings(
        "GGH15 BGG poly-encoding gpu slot",
        gate_id,
        lut_id,
        computed_slot.slot_idx,
        completed_slot_count,
        slot_count,
        computed_slot.load_ms,
        computed_slot.stage1_ms + computed_slot.stage2_ms,
        serialize_ms,
        computed_slot.slot_started.elapsed().as_secs_f64() * 1000.0,
        None,
    );
    debug!(
        "GGH15 BGG poly-encoding gpu slot multistage evaluation finished: gate_id={}, lut_id={}, slot={}, stage1_ms={:.3}, stage2_ms={:.3}, compute_ms={:.3}",
        gate_id,
        lut_id,
        computed_slot.slot_idx + 1,
        computed_slot.stage1_ms,
        computed_slot.stage2_ms,
        computed_slot.compute_ms
    );
    (vector_bytes, computed_slot.plaintext_bytes)
}

pub(super) fn benchmark_public_lookup_chunk_gpu<M, HS>(
    evaluator: &super::GGH15BGGPolyEncodingPltEvaluator<M, HS>,
    samples: &crate::bench_estimator::BggPolyEncodingBenchSamples<'_, M>,
    iterations: usize,
) -> crate::bench_estimator::PolyEncodingChunkBenchMeasurement
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let plaintext_compact_bytes_by_slot =
        samples.public_lut_input.plaintext_bytes.as_ref().expect(
            "the BGG poly encoding should reveal plaintexts for public-lookup benchmarking",
        );
    let dir = Path::new(&evaluator.dir_path);
    let d = samples.public_lut_input.pubkey.matrix.row_size();
    let shared = build_public_lookup_shared_state::<M, HS>(
        samples.params,
        dir,
        &evaluator.checkpoint_prefix,
        evaluator.hash_key,
        samples.public_lut_gate_id,
        d,
    );
    let configured_parallelism =
        effective_gpu_slot_parallelism(crate::env::bgg_poly_encoding_slot_parallelism().max(1));
    let shared_by_device = prepare_public_lookup_shared_by_device::<M>(
        samples.params,
        &shared,
        configured_parallelism,
    );
    let iterations = iterations.max(1);
    let mut latency_sum = 0.0;
    let mut total_time_sum = 0.0;
    let mut max_parallelism = 0;

    for _ in 0..iterations {
        let completed_slots = AtomicUsize::new(0);
        let loaded_slot = load_public_lookup_slot::<M>(
            0,
            1,
            samples.public_lut_gate_id,
            samples.public_lut_id,
            &completed_slots,
            &shared_by_device,
            &evaluator.c_b0_compact_bytes_by_slot[0],
            &samples.public_lut_input.vector_bytes[0],
            &plaintext_compact_bytes_by_slot[0],
        );
        let mut slot_bench = PublicLookupSlotBenchMeasurement::default();
        let _computed_slot = compute_public_lookup_slot::<M, HS>(
            loaded_slot,
            samples.public_lut_gate_id,
            samples.public_lut_id,
            samples.public_lut,
            dir,
            &evaluator.checkpoint_prefix,
            evaluator.hash_key,
            &shared_by_device,
            Some(&mut slot_bench),
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

pub(super) fn evaluate_public_lookup_slots_gpu<M, HS>(
    params: &<M::P as Poly>::Params,
    plt: &PublicLut<M::P>,
    dir: &Path,
    checkpoint_prefix: &str,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    input: &BggPolyEncoding<M>,
    plaintext_compact_bytes_by_slot: &[Arc<[u8]>],
    c_b0_compact_bytes_by_slot: &[Arc<[u8]>],
    shared: &GGH15PublicLookupSharedState<M>,
    configured_parallelism: usize,
) -> (Vec<Arc<[u8]>>, Vec<Arc<[u8]>>)
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let evaluate_started = Instant::now();
    debug!(
        "GGH15 BGG poly-encoding gpu slot evaluation started: gate_id={}, lut_id={}, slot_count={}, slot_parallelism={}",
        gate_id,
        lut_id,
        input.num_slots(),
        configured_parallelism
    );
    let prepare_started = Instant::now();
    let shared_by_device =
        prepare_public_lookup_shared_by_device::<M>(params, shared, configured_parallelism);
    let prepared_device_ids =
        shared_by_device.iter().map(|entry| entry.device_id).collect::<Vec<_>>();
    debug!(
        "Prepared GGH15 BGG poly-encoding gpu shared state: gate_id={}, lut_id={}, device_count={}, device_ids={:?}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        shared_by_device.len(),
        prepared_device_ids,
        prepare_started.elapsed().as_secs_f64() * 1000.0
    );
    let slot_count = input.num_slots();
    let mut output_vector_bytes = Vec::with_capacity(slot_count);
    let mut output_plaintext_bytes = Vec::with_capacity(slot_count);
    let completed_slots = AtomicUsize::new(0);

    // TODO: Parallelize across slots with multiple devices.
    for slot_idx in 0..slot_count {
        let loaded_slot = load_public_lookup_slot(
            slot_idx,
            slot_count,
            gate_id,
            lut_id,
            &completed_slots,
            &shared_by_device,
            &c_b0_compact_bytes_by_slot[slot_idx],
            &input.vector_bytes[slot_idx],
            &plaintext_compact_bytes_by_slot[slot_idx],
        );
        let computed_slot = compute_public_lookup_slot::<M, HS>(
            loaded_slot,
            gate_id,
            lut_id,
            plt,
            dir,
            checkpoint_prefix,
            hash_key,
            &shared_by_device,
            None,
        );
        let (vector_bytes, plaintext_bytes) =
            store_public_lookup_slot(computed_slot, slot_count, gate_id, lut_id, &completed_slots);
        output_vector_bytes.push(vector_bytes);
        output_plaintext_bytes.push(plaintext_bytes);
    }
    debug!(
        "GGH15 BGG poly-encoding gpu slot evaluation finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        slot_count,
        evaluate_started.elapsed().as_secs_f64() * 1000.0
    );
    (output_vector_bytes, output_plaintext_bytes)
}

#[cfg(test)]
mod tests {
    use super::effective_gpu_slot_parallelism;
    use crate::{
        __PAIR, __TestState,
        lookup::ggh15::{public_lookup_gpu_device_ids, public_lookup_round_robin_device_slot},
        poly::dcrt::gpu::detected_gpu_device_ids,
    };

    #[sequential_test::sequential]
    #[test]
    fn test_ggh15_slot_device_ids_uses_detected_gpu_ids() {
        let detected_gpu_ids = detected_gpu_device_ids();

        if detected_gpu_ids.is_empty() {
            let panic = std::panic::catch_unwind(|| {
                public_lookup_gpu_device_ids()
                    .into_iter()
                    .take(effective_gpu_slot_parallelism(1))
                    .collect::<Vec<_>>()
            })
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
        let effective_parallelism = effective_gpu_slot_parallelism(slot_parallelism);
        assert_eq!(
            public_lookup_gpu_device_ids()
                .into_iter()
                .take(effective_parallelism)
                .collect::<Vec<_>>(),
            detected_gpu_ids[..slot_parallelism].to_vec()
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ggh15_effective_gpu_slot_parallelism_clamps_to_detected_gpus() {
        let detected_count = detected_gpu_device_ids().len().max(1);
        assert_eq!(effective_gpu_slot_parallelism(detected_count + 5), detected_count);
    }

    #[test]
    fn test_ggh15_round_robin_slot_device_slot_balances_logical_slots() {
        let device_count = 3;
        let logical_slots = 11;
        let mut counts = vec![0usize; device_count];
        for slot_idx in 0..logical_slots {
            counts[public_lookup_round_robin_device_slot(slot_idx, device_count)] += 1;
        }

        let min_count = *counts.iter().min().expect("counts must be non-empty");
        let max_count = *counts.iter().max().expect("counts must be non-empty");
        assert!(max_count - min_count <= 1, "counts={counts:?}");
    }
}
