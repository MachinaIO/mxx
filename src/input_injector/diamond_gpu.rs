use super::DiamondInjector;
use crate::{
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::column_chunk_count,
};
use rayon::prelude::*;
use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};
use tracing::{debug, info};

struct GpuDiamondKStageShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <M::P as Poly>::Params,
    source_b: M,
    source_trapdoor: T,
    target_b: M,
}

struct LoadedDiamondPreprocessChunk<M>
where
    M: PolyMatrix,
{
    device_slot: usize,
    task: DiamondPreprocessChunkTask,
    ext_matrix: M,
    target: M,
    load_s: f64,
}

struct ComputedDiamondPreprocessChunk<M>
where
    M: PolyMatrix,
{
    task: DiamondPreprocessChunkTask,
    output: M,
    load_s: f64,
    compute_s: f64,
}

#[derive(Clone, Copy, Debug)]
enum DiamondPreprocessChunkTask {
    K { level: usize, digit_value: usize, state_idx: usize, chunk_idx: usize },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum DiamondEvalFamily {
    State(usize),
}

#[derive(Clone, Copy, Debug)]
struct DiamondEvalTask {
    family: DiamondEvalFamily,
    chunk_idx: usize,
}

struct DiamondEvalFamilySpec {
    lhs_bytes: Arc<[u8]>,
    rhs_chunk_bytes: Vec<Arc<[u8]>>,
}

struct GpuDiamondEvalShared<M>
where
    M: PolyMatrix,
{
    device_id: i32,
    params: <M::P as Poly>::Params,
    lhs_by_family: HashMap<DiamondEvalFamily, M>,
}

struct LoadedDiamondEvalTask<M>
where
    M: PolyMatrix,
{
    device_slot: usize,
    task: DiamondEvalTask,
    rhs_chunk: M,
    load_s: f64,
}

struct ComputedDiamondEvalTask<M>
where
    M: PolyMatrix,
{
    task: DiamondEvalTask,
    output: M,
    load_s: f64,
    compute_s: f64,
}

fn maybe_elapsed_s(started: Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

fn preprocess_progress_label(completed_tasks: usize, total_tasks: usize) -> String {
    let percent =
        if total_tasks == 0 { 100usize } else { completed_tasks.saturating_mul(100) / total_tasks };
    format!("{}/{} ({} %)", completed_tasks, total_tasks, percent)
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
    let first = chunk_iter.next().expect("diamond injector chunk byte list must be non-empty");
    first.concat_columns_owned(chunk_iter.collect())
}

fn round_robin_batches<T: Copy>(tasks: &[T], device_count: usize) -> Vec<Vec<T>> {
    let mut batches = (0..device_count.max(1)).map(|_| Vec::new()).collect::<Vec<_>>();
    for (task_idx, task) in tasks.iter().copied().enumerate() {
        batches[task_idx % device_count.max(1)].push(task);
    }
    batches
}

impl<M, US, HS, TS> DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn effective_gpu_device_ids(&self) -> Vec<i32> {
        if !self.gpu_device_ids.is_empty() {
            return self.gpu_device_ids.clone();
        }
        let device_ids = self.params.device_ids();
        if device_ids.is_empty() { vec![0] } else { device_ids }
    }

    fn prepare_gpu_k_stage_shared(
        &self,
        source_b_bytes: &[u8],
        source_trapdoor_bytes: &[u8],
        target_b_bytes: &[u8],
    ) -> Vec<GpuDiamondKStageShared<M, TS::Trapdoor>> {
        self.effective_gpu_device_ids()
            .into_par_iter()
            .map(|device_id| {
                let local_params = self.params.params_for_device(device_id);
                let source_trapdoor = TS::trapdoor_from_bytes(&local_params, source_trapdoor_bytes)
                    .unwrap_or_else(|| {
                        panic!(
                            "DiamondInjector failed to decode K-stage trapdoor checkpoint on device {}",
                            device_id
                        )
                    });
                GpuDiamondKStageShared {
                    device_id,
                    params: local_params.clone(),
                    source_b: M::from_compact_bytes(&local_params, source_b_bytes),
                    source_trapdoor,
                    target_b: M::from_compact_bytes(&local_params, target_b_bytes),
                }
            })
            .collect()
    }

    fn prepare_gpu_eval_shared(
        &self,
        family_specs: &HashMap<DiamondEvalFamily, DiamondEvalFamilySpec>,
    ) -> Vec<GpuDiamondEvalShared<M>> {
        self.effective_gpu_device_ids()
            .into_par_iter()
            .map(|device_id| {
                let local_params = self.params.params_for_device(device_id);
                let lhs_by_family = family_specs
                    .iter()
                    .map(|(family, spec)| {
                        (*family, M::from_compact_bytes(&local_params, spec.lhs_bytes.as_ref()))
                    })
                    .collect::<HashMap<_, _>>();
                GpuDiamondEvalShared { device_id, params: local_params, lhs_by_family }
            })
            .collect()
    }

    fn preprocess_chunk_id(&self, task: &DiamondPreprocessChunkTask) -> String {
        match task {
            DiamondPreprocessChunkTask::K { level, digit_value, state_idx, chunk_idx } => {
                self.chunk_id(&self.k_id(*level, *digit_value, *state_idx), *chunk_idx)
            }
        }
    }

    fn read_chunk_bytes(&self, dir_path: &Path, id: &str, total_cols: usize) -> Vec<Arc<[u8]>> {
        (0..column_chunk_count(total_cols))
            .map(|chunk_idx| {
                Arc::<[u8]>::from(self.read_matrix_bytes(dir_path, &self.chunk_id(id, chunk_idx)))
            })
            .collect()
    }

    fn store_preprocess_wave(
        &self,
        dir_path: &Path,
        outputs: Vec<ComputedDiamondPreprocessChunk<M>>,
    ) -> usize {
        outputs
            .into_par_iter()
            .map(|output| {
                let store_started = Instant::now();
                let chunk_id = self.preprocess_chunk_id(&output.task);
                let bytes = output.output.into_compact_bytes();
                self.write_matrix_bytes(dir_path, &chunk_id, &bytes);
                debug!(
                    ?output.task,
                    chunk_id,
                    load_s = output.load_s,
                    compute_s = output.compute_s,
                    store_s = maybe_elapsed_s(store_started),
                    "diamond injector gpu preprocess: stored chunk"
                );
                1usize
            })
            .sum::<usize>()
    }

    fn preprocess_k_stage_gpu(
        &self,
        dir_path: &Path,
        hash_key: [u8; 32],
        level: usize,
        tasks: Vec<DiamondPreprocessChunkTask>,
        source_b_bytes: &[u8],
        source_trapdoor_bytes: &[u8],
        target_b_bytes: &[u8],
        secret_mask_bytes: &HashMap<usize, Arc<[u8]>>,
    ) -> usize {
        if tasks.is_empty() {
            return 0;
        }

        let stage_started = Instant::now();
        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let shared_by_device =
            self.prepare_gpu_k_stage_shared(source_b_bytes, source_trapdoor_bytes, target_b_bytes);
        let device_count = shared_by_device.len().max(1);
        let task_batches = round_robin_batches(&tasks, device_count);
        let wave_count = task_batches.iter().map(Vec::len).max().unwrap_or(0);

        info!(
            level,
            device_count,
            total_tasks = tasks.len(),
            progress = preprocess_progress_label(0, tasks.len()),
            "diamond injector gpu preprocess: K stage starting"
        );

        let load_wave = |wave_idx: usize| {
            let loaded_wave = task_batches
                .par_iter()
                .zip(shared_by_device.par_iter())
                .enumerate()
                .filter_map(|(device_slot, (batch, shared))| {
                    let task = batch.get(wave_idx).copied()?;
                    let load_started = Instant::now();
                    let DiamondPreprocessChunkTask::K {
                        level: task_level,
                        digit_value,
                        state_idx,
                        chunk_idx,
                    } = task;
                    debug_assert_eq!(task_level, level);
                    let secret_mask = M::from_compact_bytes(
                        &shared.params,
                        secret_mask_bytes
                            .get(&digit_value)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing K-stage secret checkpoint for level {}, digit {}",
                                    level, digit_value
                                )
                            })
                            .as_ref(),
                    );
                    let ext_matrix = if self.new_bit_idx_for_state(level, state_idx).is_some() {
                        self.sample_w_block_with_params(&shared.params, hash_key, 0, level - 1)
                    } else {
                        self.sample_w_block_with_params(
                            &shared.params,
                            hash_key,
                            state_idx,
                            level - 1,
                        )
                    };
                    let target = self.build_k_target_chunk_with_params(
                        &shared.params,
                        hash_key,
                        level,
                        digit_value,
                        state_idx,
                        &secret_mask,
                        &shared.target_b,
                        chunk_idx,
                    );
                    drop(secret_mask);
                    debug!(
                        device_id = shared.device_id,
                        level,
                        digit_value,
                        state_idx,
                        chunk_idx,
                        load_s = maybe_elapsed_s(load_started),
                        "diamond injector gpu preprocess: loaded K-stage chunk"
                    );
                    Some(LoadedDiamondPreprocessChunk {
                        device_slot,
                        task,
                        ext_matrix,
                        target,
                        load_s: maybe_elapsed_s(load_started),
                    })
                })
                .collect::<Vec<_>>();
            (!loaded_wave.is_empty()).then_some(loaded_wave)
        };

        let compute_wave = |loaded_wave: Vec<LoadedDiamondPreprocessChunk<M>>| {
            loaded_wave
                .into_par_iter()
                .map(|loaded| {
                    let shared = &shared_by_device[loaded.device_slot];
                    let compute_started = Instant::now();
                    let output = trap_sampler.preimage_extend(
                        &shared.params,
                        &shared.source_trapdoor,
                        &shared.source_b,
                        &loaded.ext_matrix,
                        &loaded.target,
                    );
                    ComputedDiamondPreprocessChunk {
                        task: loaded.task,
                        output,
                        load_s: loaded.load_s,
                        compute_s: maybe_elapsed_s(compute_started),
                    }
                })
                .collect::<Vec<_>>()
        };

        let mut completed_tasks = 0usize;
        let mut next_wave_idx = 1usize;
        let mut current_wave = if wave_count == 0 { None } else { load_wave(0) };
        let mut previous_outputs = None;
        while let Some(loaded_wave) = current_wave.take() {
            let should_load_next = next_wave_idx < wave_count;
            let previous_outputs_to_store = previous_outputs.take();
            let wave_started = Instant::now();
            let ((computed_outputs, next_loaded_wave), stored_now) = rayon::join(
                || {
                    rayon::join(
                        || compute_wave(loaded_wave),
                        || if should_load_next { load_wave(next_wave_idx) } else { None },
                    )
                },
                || {
                    previous_outputs_to_store
                        .map(|outputs| self.store_preprocess_wave(dir_path, outputs))
                        .unwrap_or(0)
                },
            );
            completed_tasks += stored_now;
            info!(
                level,
                wave = next_wave_idx,
                wave_count,
                completed_tasks,
                total_tasks = tasks.len(),
                progress = preprocess_progress_label(completed_tasks, tasks.len()),
                elapsed_s = maybe_elapsed_s(wave_started),
                "diamond injector gpu preprocess: K stage wave completed"
            );
            previous_outputs = Some(computed_outputs);
            current_wave = next_loaded_wave;
            next_wave_idx += 1;
        }
        if let Some(outputs) = previous_outputs {
            completed_tasks += self.store_preprocess_wave(dir_path, outputs);
        }

        info!(
            level,
            completed_tasks,
            total_tasks = tasks.len(),
            progress = preprocess_progress_label(completed_tasks, tasks.len()),
            elapsed_s = maybe_elapsed_s(stage_started),
            "diamond injector gpu preprocess: K stage finished"
        );
        completed_tasks
    }

    fn gpu_left_mul_families(
        &self,
        family_specs: HashMap<DiamondEvalFamily, DiamondEvalFamilySpec>,
    ) -> HashMap<DiamondEvalFamily, M> {
        if family_specs.is_empty() {
            return HashMap::new();
        }

        let shared_by_device = self.prepare_gpu_eval_shared(&family_specs);
        let device_count = shared_by_device.len().max(1);
        let tasks = family_specs
            .iter()
            .flat_map(|(family, spec)| {
                (0..spec.rhs_chunk_bytes.len())
                    .map(move |chunk_idx| DiamondEvalTask { family: *family, chunk_idx })
            })
            .collect::<Vec<_>>();
        let task_batches = round_robin_batches(&tasks, device_count);
        let wave_count = task_batches.iter().map(Vec::len).max().unwrap_or(0);

        let load_wave = |wave_idx: usize| {
            let loaded_wave = task_batches
                .par_iter()
                .zip(shared_by_device.par_iter())
                .enumerate()
                .filter_map(|(device_slot, (batch, shared))| {
                    let task = batch.get(wave_idx).copied()?;
                    let load_started = Instant::now();
                    let rhs_bytes = family_specs
                        .get(&task.family)
                        .unwrap_or_else(|| {
                            panic!("missing rhs bytes for diamond eval family {:?}", task.family)
                        })
                        .rhs_chunk_bytes[task.chunk_idx]
                        .clone();
                    let rhs_chunk = M::from_compact_bytes(&shared.params, rhs_bytes.as_ref());
                    debug!(
                        device_id = shared.device_id,
                        family = ?task.family,
                        chunk_idx = task.chunk_idx,
                        load_s = maybe_elapsed_s(load_started),
                        "diamond injector gpu online_eval: loaded rhs chunk"
                    );
                    Some(LoadedDiamondEvalTask {
                        device_slot,
                        task,
                        rhs_chunk,
                        load_s: maybe_elapsed_s(load_started),
                    })
                })
                .collect::<Vec<_>>();
            (!loaded_wave.is_empty()).then_some(loaded_wave)
        };

        let compute_wave = |loaded_wave: Vec<LoadedDiamondEvalTask<M>>| {
            loaded_wave
                .into_par_iter()
                .map(|loaded| {
                    let shared = &shared_by_device[loaded.device_slot];
                    let compute_started = Instant::now();
                    let lhs = shared.lhs_by_family.get(&loaded.task.family).unwrap_or_else(|| {
                        panic!("missing lhs for diamond eval family {:?}", loaded.task.family)
                    });
                    let output = lhs.clone() * &loaded.rhs_chunk;
                    ComputedDiamondEvalTask {
                        task: loaded.task,
                        output,
                        load_s: loaded.load_s,
                        compute_s: maybe_elapsed_s(compute_started),
                    }
                })
                .collect::<Vec<_>>()
        };

        let mut output_buffers = family_specs
            .iter()
            .map(|(family, spec)| (*family, vec![None; spec.rhs_chunk_bytes.len()]))
            .collect::<HashMap<_, _>>();

        let mut store_wave = |outputs: Vec<ComputedDiamondEvalTask<M>>| {
            for output in outputs {
                let store_started = Instant::now();
                let bytes = output.output.into_compact_bytes();
                output_buffers.get_mut(&output.task.family).unwrap_or_else(|| {
                    panic!("missing output buffer for diamond eval family {:?}", output.task.family)
                })[output.task.chunk_idx] = Some(bytes);
                debug!(
                    family = ?output.task.family,
                    chunk_idx = output.task.chunk_idx,
                    load_s = output.load_s,
                    compute_s = output.compute_s,
                    store_s = maybe_elapsed_s(store_started),
                    "diamond injector gpu online_eval: stored chunk"
                );
            }
        };

        let mut next_wave_idx = 1usize;
        let mut current_wave = if wave_count == 0 { None } else { load_wave(0) };
        let mut previous_outputs = None;
        while let Some(loaded_wave) = current_wave.take() {
            let should_load_next = next_wave_idx < wave_count;
            let previous_outputs_to_store = previous_outputs.take();
            let ((computed_outputs, next_loaded_wave), ()) = rayon::join(
                || {
                    rayon::join(
                        || compute_wave(loaded_wave),
                        || if should_load_next { load_wave(next_wave_idx) } else { None },
                    )
                },
                || {
                    if let Some(outputs) = previous_outputs_to_store {
                        store_wave(outputs);
                    }
                },
            );
            info!(
                wave = next_wave_idx,
                wave_count, "diamond injector gpu online_eval: wave completed"
            );
            previous_outputs = Some(computed_outputs);
            current_wave = next_loaded_wave;
            next_wave_idx += 1;
        }
        if let Some(outputs) = previous_outputs {
            store_wave(outputs);
        }

        family_specs
            .into_iter()
            .map(|(family, spec)| {
                let assembled = concat_chunk_bytes_on_base::<M, Vec<u8>>(
                    &self.params,
                    output_buffers
                        .remove(&family)
                        .unwrap_or_else(|| {
                            panic!("missing assembled output buffer for {:?}", family)
                        })
                        .into_iter()
                        .enumerate()
                        .map(|(chunk_idx, bytes)| {
                            bytes.unwrap_or_else(|| {
                                panic!(
                                    "missing chunk {} for diamond eval family {:?}",
                                    chunk_idx, family
                                )
                            })
                        })
                        .collect::<Vec<_>>(),
                );
                debug_assert_eq!(
                    assembled.col_size(),
                    spec.rhs_chunk_bytes.len().min(1) * 0 + assembled.col_size()
                );
                (family, assembled)
            })
            .collect()
    }

    pub(super) fn preprocess_gpu(&self, dir_path: &Path, hash_key: [u8; 32], k: &M::P) {
        self.ensure_dir(dir_path);
        self.write_metadata(
            dir_path,
            &super::DiamondInjectorMetadata { input_count: self.input_count, base: self.base },
        );
        self.write_bytes(dir_path, self.k_plaintext_id(), &k.to_compact_bytes());

        let preprocess_started = Instant::now();
        let state_cols = self.state_col_size(&self.params);
        info!("diamond injector gpu preprocess: starting");

        // Persist the empty-prefix seed once. This seed is not a trapdoor
        // preimage; it is the initial encoding that online evaluation uses as
        // p_{epsilon,0} before any digit transition is applied.
        let secret_epsilon_bytes =
            self.load_or_sample_secret_epsilon_bytes(dir_path, self.secret_epsilon_id());
        if !self.matrix_exists(dir_path, self.p_epsilon_id()) {
            let (b0_bytes, _) = self.load_or_sample_b_checkpoint_bytes(dir_path, 0);
            let b0_matrix = M::from_compact_bytes(&self.params, &b0_bytes);
            let secret_epsilon = M::from_compact_bytes(&self.params, &secret_epsilon_bytes);
            let p_epsilon = self.build_initial_encoding(hash_key, &b0_matrix, &secret_epsilon, k);
            self.write_matrix(dir_path, self.p_epsilon_id(), &p_epsilon);
            drop(p_epsilon);
            drop(secret_epsilon);
            drop(b0_matrix);
        }

        let mut total_tasks = 0usize;
        let mut completed_tasks = 0usize;

        // Process K_{i,b,j} level by level so each stage only keeps B_{i-1},
        // B_i, the trapdoor of B_{i-1}, and the current level's secret masks
        // resident at once.
        for level in 1..=self.input_count {
            let secret_mask_bytes = (0..self.base)
                .map(|digit_value| {
                    (
                        digit_value,
                        Arc::<[u8]>::from(self.load_or_sample_digit_secret_mask_bytes(
                            dir_path,
                            &self.digit_secret_id(level, digit_value),
                        )),
                    )
                })
                .collect::<HashMap<_, _>>();
            let mut stage_tasks = Vec::new();
            for digit_value in 0..self.base {
                for state_idx in 0..self.expanded_state_count_after_level(level) {
                    for chunk_idx in 0..column_chunk_count(state_cols) {
                        let task = DiamondPreprocessChunkTask::K {
                            level,
                            digit_value,
                            state_idx,
                            chunk_idx,
                        };
                        if !self.matrix_exists(dir_path, &self.preprocess_chunk_id(&task)) {
                            stage_tasks.push(task);
                        }
                    }
                }
            }
            total_tasks += stage_tasks.len();
            let (source_b_bytes, source_trapdoor_bytes) =
                self.load_or_sample_b_checkpoint_bytes(dir_path, level - 1);
            let (target_b_bytes, _) = self.load_or_sample_b_checkpoint_bytes(dir_path, level);
            if stage_tasks.is_empty() {
                info!(
                    level,
                    progress = preprocess_progress_label(0, 0),
                    "diamond injector gpu preprocess: K stage already checkpointed"
                );
            } else {
                completed_tasks += self.preprocess_k_stage_gpu(
                    dir_path,
                    hash_key,
                    level,
                    stage_tasks,
                    &source_b_bytes,
                    &source_trapdoor_bytes,
                    &target_b_bytes,
                    &secret_mask_bytes,
                );
            }
        }

        if total_tasks == 0 {
            info!(
                elapsed_s = preprocess_started.elapsed().as_secs_f64(),
                "diamond injector gpu preprocess: all checkpoints already exist"
            );
            return;
        }

        info!(
            completed_tasks,
            total_tasks,
            elapsed_s = preprocess_started.elapsed().as_secs_f64(),
            "diamond injector gpu preprocess: finished"
        );
    }

    pub(super) fn online_eval_gpu(
        &self,
        dir_path: &Path,
        preprocess_out: &super::DiamondInjectorPreprocessOut<M, TS::Trapdoor>,
        input_digits: &[u32],
    ) -> Vec<M> {
        self.validate_digits(input_digits);
        assert_eq!(
            self.read_bytes(dir_path, self.preprocess_hash_key_id()).as_slice(),
            &preprocess_out.hash_key,
            "DiamondInjector gpu online_eval preprocess hash key mismatch"
        );
        let metadata = self.read_metadata(dir_path);
        assert_eq!(
            metadata.input_count, self.input_count,
            "DiamondInjector metadata input count mismatch"
        );
        assert_eq!(metadata.base, self.base, "DiamondInjector metadata base mismatch");

        let online_started = Instant::now();
        let state_cols = self.state_col_size(&self.params);
        // Start from the persisted empty-prefix seed.
        let mut states = vec![self.read_matrix(dir_path, self.p_epsilon_id())];

        for (digit_idx, digit_value) in input_digits.iter().copied().enumerate() {
            let level = digit_idx + 1;
            let prev_states = std::mem::take(&mut states);
            let prev_p0_bytes = Arc::<[u8]>::from(prev_states[0].to_compact_bytes());
            // Schedule one matrix-product family per active branch. The helper
            // below splits the right-hand-side chunks across devices and then
            // reassembles each branch on CPU from compact bytes.
            let family_specs = (0..self.expanded_state_count_after_level(level))
                .map(|state_idx| {
                    let lhs_bytes = if self.new_bit_idx_for_state(level, state_idx).is_some() {
                        prev_p0_bytes.clone()
                    } else {
                        Arc::<[u8]>::from(prev_states[state_idx].to_compact_bytes())
                    };
                    let rhs_id = self.k_id(level, digit_value as usize, state_idx);
                    (
                        DiamondEvalFamily::State(state_idx),
                        DiamondEvalFamilySpec {
                            lhs_bytes,
                            rhs_chunk_bytes: self.read_chunk_bytes(dir_path, &rhs_id, state_cols),
                        },
                    )
                })
                .collect::<HashMap<_, _>>();
            let outputs = self.gpu_left_mul_families(family_specs);
            states = (0..self.expanded_state_count_after_level(level))
                .map(|state_idx| {
                    outputs
                        .get(&DiamondEvalFamily::State(state_idx))
                        .unwrap_or_else(|| {
                            panic!(
                                "missing gpu online_eval state output for level {}, state {}",
                                level, state_idx
                            )
                        })
                        .clone()
                })
                .collect::<Vec<_>>();
        }

        info!(
            elapsed_s = online_started.elapsed().as_secs_f64(),
            "diamond injector gpu online_eval: finished"
        );
        states
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DiamondInjector, InputInjector};
    use crate::{
        __PAIR, __TestState,
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
    };
    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    type TestInjector = DiamondInjector<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
    >;

    #[sequential_test::sequential]
    #[test]
    fn test_gpu_diamond_injector_online_eval_returns_exact_bgg_relations() {
        type TestPoly = <GpuDCRTPolyMatrix as PolyMatrix>::P;

        let _ = tracing_subscriber::fmt::try_init();
        gpu_device_sync();

        let gpu_ids = detected_gpu_device_ids();
        assert!(
            !gpu_ids.is_empty(),
            "at least one GPU device is required for input_injector GPU tests"
        );

        let cpu_params = DCRTPolyParams::default();
        let (moduli, _, _) = cpu_params.to_crt();
        let params = GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli,
            cpu_params.base_bits(),
            vec![gpu_ids[0]],
            Some(1),
        );

        let input_count = 3;
        let base = 4;
        let batch_bits = 2;
        let dir = tempdir().expect("temporary directory should be created");

        let injector = TestInjector::new(params.clone(), input_count, base, 4.578, 0.0)
            .with_gpu_device_ids(gpu_ids.clone());

        let k = TestPoly::from_usize_to_constant(&params, 3);

        let preprocess_out = injector.preprocess(dir.path(), &k);

        let digits = vec![2u32, 1u32, 3u32];
        let states = injector.online_eval(dir.path(), &preprocess_out, &digits);
        assert_eq!(states.len(), 1 + input_count * batch_bits);

        let mut secret_matrix = injector.read_matrix(dir.path(), injector.secret_epsilon_id());
        assert_eq!(secret_matrix.size(), (1, super::super::DIAMOND_SECRET_SIZE));
        for (digit_idx, digit_value) in digits.iter().copied().enumerate() {
            let secret_mask = injector.read_matrix(
                dir.path(),
                &injector.digit_secret_id(digit_idx + 1, digit_value as usize),
            );
            assert_eq!(
                secret_mask.size(),
                (super::super::DIAMOND_SECRET_SIZE, super::super::DIAMOND_SECRET_SIZE)
            );
            secret_matrix = secret_matrix * secret_mask;
        }
        let base_public_matrix =
            preprocess_out.final_pub_matrix.concat_columns(&[&injector
                .sample_w_block_with_params(&params, preprocess_out.hash_key, 0, input_count)]);
        let base_selector = GpuDCRTPolyMatrix::from_poly_vec_row(
            &params,
            vec![secret_matrix.entry(0, 0), k.clone()],
        );
        assert_eq!(states[0], base_selector * base_public_matrix);

        for digit_idx in 0..input_count {
            for bit_idx in 0..batch_bits {
                let state_idx = injector.bit_state_idx(digit_idx, bit_idx);
                let bit_value = ((digits[digit_idx] as usize) >> bit_idx) & 1;
                let bit_plaintext = TestPoly::from_usize_to_constant(&params, bit_value);
                let bit_public_matrix =
                    preprocess_out.final_pub_matrix.concat_columns(&[&injector
                        .sample_w_block_with_params(
                            &params,
                            preprocess_out.hash_key,
                            state_idx,
                            input_count,
                        )]);
                let bit_selector = GpuDCRTPolyMatrix::from_poly_vec_row(
                    &params,
                    vec![secret_matrix.entry(0, 0), secret_matrix.entry(0, 0) * &bit_plaintext],
                );
                assert_eq!(states[state_idx], bit_selector * bit_public_matrix);
            }
        }
    }
}
