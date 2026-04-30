use super::DiamondInjector;
use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::column_chunk_count,
};
use rayon::prelude::*;
use std::{collections::HashMap, sync::Arc, time::Instant};
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

struct GpuDiamondOutputStageShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <M::P as Poly>::Params,
    source_b: M,
    source_trapdoor: T,
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
    L { input_idx: usize, chunk_idx: usize },
    M { decoder_idx: usize, chunk_idx: usize },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum DiamondEvalFamily {
    State(usize),
    One,
    Input(usize),
    Decoder(usize),
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

    fn prepare_gpu_output_stage_shared(
        &self,
        source_b_bytes: &[u8],
        source_trapdoor_bytes: &[u8],
    ) -> Vec<GpuDiamondOutputStageShared<M, TS::Trapdoor>> {
        self.effective_gpu_device_ids()
            .into_par_iter()
            .map(|device_id| {
                let local_params = self.params.params_for_device(device_id);
                let source_trapdoor = TS::trapdoor_from_bytes(&local_params, source_trapdoor_bytes)
                    .unwrap_or_else(|| {
                        panic!(
                            "DiamondInjector failed to decode output-stage trapdoor checkpoint on device {}",
                            device_id
                        )
                    });
                GpuDiamondOutputStageShared {
                    device_id,
                    params: local_params.clone(),
                    source_b: M::from_compact_bytes(&local_params, source_b_bytes),
                    source_trapdoor,
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
            DiamondPreprocessChunkTask::L { input_idx, chunk_idx } => {
                self.chunk_id(&self.l_id(*input_idx), *chunk_idx)
            }
            DiamondPreprocessChunkTask::M { decoder_idx, chunk_idx } => {
                self.chunk_id(&self.m_id(*decoder_idx), *chunk_idx)
            }
        }
    }

    fn read_chunk_bytes(&self, id: &str, total_cols: usize) -> Vec<Arc<[u8]>> {
        (0..column_chunk_count(total_cols))
            .map(|chunk_idx| {
                Arc::<[u8]>::from(self.read_matrix_bytes(&self.chunk_id(id, chunk_idx)))
            })
            .collect()
    }

    fn store_preprocess_wave(&self, outputs: Vec<ComputedDiamondPreprocessChunk<M>>) -> usize {
        outputs
            .into_par_iter()
            .map(|output| {
                let store_started = Instant::now();
                let chunk_id = self.preprocess_chunk_id(&output.task);
                let bytes = output.output.into_compact_bytes();
                self.write_matrix_bytes(&chunk_id, &bytes);
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
                    let (digit_value, state_idx, chunk_idx) = match task {
                        DiamondPreprocessChunkTask::K {
                            level: task_level,
                            digit_value,
                            state_idx,
                            chunk_idx,
                        } => {
                            debug_assert_eq!(task_level, level);
                            (digit_value, state_idx, chunk_idx)
                        }
                        _ => panic!("non-K task scheduled in K stage: {:?}", task),
                    };
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
                        self.sample_w_block_with_params(&shared.params, 0, level - 1)
                    } else {
                        self.sample_w_block_with_params(&shared.params, state_idx, level - 1)
                    };
                    let target = self.build_k_target_chunk_with_params(
                        &shared.params,
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
                        .map(|outputs| self.store_preprocess_wave(outputs))
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
            completed_tasks += self.store_preprocess_wave(outputs);
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

    fn preprocess_l_stage_gpu(
        &self,
        tasks: Vec<DiamondPreprocessChunkTask>,
        source_b_bytes: &[u8],
        source_trapdoor_bytes: &[u8],
        one_bytes: Arc<[u8]>,
        input_pubkey_bytes: &[Arc<[u8]>],
    ) -> usize {
        if tasks.is_empty() {
            return 0;
        }

        let stage_started = Instant::now();
        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let shared_by_device =
            self.prepare_gpu_output_stage_shared(source_b_bytes, source_trapdoor_bytes);
        let device_count = shared_by_device.len().max(1);
        let task_batches = round_robin_batches(&tasks, device_count);
        let wave_count = task_batches.iter().map(Vec::len).max().unwrap_or(0);

        info!(
            device_count,
            total_tasks = tasks.len(),
            progress = preprocess_progress_label(0, tasks.len()),
            "diamond injector gpu preprocess: L stage starting"
        );

        let load_wave = |wave_idx: usize| {
            let loaded_wave = task_batches
                .par_iter()
                .zip(shared_by_device.par_iter())
                .enumerate()
                .filter_map(|(device_slot, (batch, shared))| {
                    let task = batch.get(wave_idx).copied()?;
                    let load_started = Instant::now();
                    let (input_idx, chunk_idx) = match task {
                        DiamondPreprocessChunkTask::L { input_idx, chunk_idx } => {
                            (input_idx, chunk_idx)
                        }
                        _ => panic!("non-L task scheduled in L stage: {:?}", task),
                    };
                    let ext_matrix = self.sample_w_block_with_params(
                        &shared.params,
                        input_idx,
                        self.input_count,
                    );
                    let gadget = M::gadget_matrix(&shared.params, super::DIAMOND_SECRET_SIZE);
                    let target = if input_idx == 0 {
                        let one_pubkey = BggPublicKey::new(
                            M::from_compact_bytes(&shared.params, one_bytes.as_ref()),
                            false,
                        );
                        let target = self.build_one_target_chunk_with_params(
                            &shared.params,
                            &one_pubkey,
                            &gadget,
                            chunk_idx,
                        );
                        drop(one_pubkey);
                        target
                    } else {
                        let pubkey = BggPublicKey::new(
                            M::from_compact_bytes(
                                &shared.params,
                                input_pubkey_bytes[input_idx - 1].as_ref(),
                            ),
                            false,
                        );
                        let target = self.build_input_target_chunk_with_params(
                            &shared.params,
                            &pubkey,
                            &gadget,
                            chunk_idx,
                        );
                        drop(pubkey);
                        target
                    };
                    drop(gadget);
                    debug!(
                        device_id = shared.device_id,
                        input_idx,
                        chunk_idx,
                        load_s = maybe_elapsed_s(load_started),
                        "diamond injector gpu preprocess: loaded L-stage chunk"
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
                        .map(|outputs| self.store_preprocess_wave(outputs))
                        .unwrap_or(0)
                },
            );
            completed_tasks += stored_now;
            info!(
                wave = next_wave_idx,
                wave_count,
                completed_tasks,
                total_tasks = tasks.len(),
                progress = preprocess_progress_label(completed_tasks, tasks.len()),
                elapsed_s = maybe_elapsed_s(wave_started),
                "diamond injector gpu preprocess: L stage wave completed"
            );
            previous_outputs = Some(computed_outputs);
            current_wave = next_loaded_wave;
            next_wave_idx += 1;
        }
        if let Some(outputs) = previous_outputs {
            completed_tasks += self.store_preprocess_wave(outputs);
        }

        info!(
            completed_tasks,
            total_tasks = tasks.len(),
            progress = preprocess_progress_label(completed_tasks, tasks.len()),
            elapsed_s = maybe_elapsed_s(stage_started),
            "diamond injector gpu preprocess: L stage finished"
        );
        completed_tasks
    }

    fn preprocess_m_stage_gpu(
        &self,
        tasks: Vec<DiamondPreprocessChunkTask>,
        source_b_bytes: &[u8],
        source_trapdoor_bytes: &[u8],
        decoder_pubkey_bytes: &[Arc<[u8]>],
    ) -> usize {
        if tasks.is_empty() {
            return 0;
        }

        let stage_started = Instant::now();
        let trap_sampler = TS::new(&self.params, self.trapdoor_sigma);
        let shared_by_device =
            self.prepare_gpu_output_stage_shared(source_b_bytes, source_trapdoor_bytes);
        let device_count = shared_by_device.len().max(1);
        let task_batches = round_robin_batches(&tasks, device_count);
        let wave_count = task_batches.iter().map(Vec::len).max().unwrap_or(0);

        info!(
            device_count,
            total_tasks = tasks.len(),
            progress = preprocess_progress_label(0, tasks.len()),
            "diamond injector gpu preprocess: M stage starting"
        );

        let load_wave = |wave_idx: usize| {
            let loaded_wave = task_batches
                .par_iter()
                .zip(shared_by_device.par_iter())
                .enumerate()
                .filter_map(|(device_slot, (batch, shared))| {
                    let task = batch.get(wave_idx).copied()?;
                    let load_started = Instant::now();
                    let (decoder_idx, chunk_idx) = match task {
                        DiamondPreprocessChunkTask::M { decoder_idx, chunk_idx } => {
                            (decoder_idx, chunk_idx)
                        }
                        _ => panic!("non-M task scheduled in M stage: {:?}", task),
                    };
                    let ext_matrix =
                        self.sample_w_block_with_params(&shared.params, 0, self.input_count);
                    let decoder_pubkey = BggPublicKey::new(
                        M::from_compact_bytes(
                            &shared.params,
                            decoder_pubkey_bytes[decoder_idx].as_ref(),
                        ),
                        false,
                    );
                    let target = self.build_decoder_target_chunk_with_params(
                        &shared.params,
                        &decoder_pubkey,
                        chunk_idx,
                    );
                    drop(decoder_pubkey);
                    debug!(
                        device_id = shared.device_id,
                        decoder_idx,
                        chunk_idx,
                        load_s = maybe_elapsed_s(load_started),
                        "diamond injector gpu preprocess: loaded M-stage chunk"
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
                        .map(|outputs| self.store_preprocess_wave(outputs))
                        .unwrap_or(0)
                },
            );
            completed_tasks += stored_now;
            info!(
                wave = next_wave_idx,
                wave_count,
                completed_tasks,
                total_tasks = tasks.len(),
                progress = preprocess_progress_label(completed_tasks, tasks.len()),
                elapsed_s = maybe_elapsed_s(wave_started),
                "diamond injector gpu preprocess: M stage wave completed"
            );
            previous_outputs = Some(computed_outputs);
            current_wave = next_loaded_wave;
            next_wave_idx += 1;
        }
        if let Some(outputs) = previous_outputs {
            completed_tasks += self.store_preprocess_wave(outputs);
        }

        info!(
            completed_tasks,
            total_tasks = tasks.len(),
            progress = preprocess_progress_label(completed_tasks, tasks.len()),
            elapsed_s = maybe_elapsed_s(stage_started),
            "diamond injector gpu preprocess: M stage finished"
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

    pub(super) fn preprocess_gpu(
        &self,
        one: &BggPublicKey<M>,
        input_digits: &[BggPublicKey<M>],
        decoders: &[BggPublicKey<M>],
    ) {
        self.validate_lengths(input_digits, decoders);
        self.ensure_dir();
        self.write_metadata(&super::DiamondInjectorMetadata {
            input_count: self.input_count,
            base: self.base,
            decoder_count: self.decoder_count,
        });

        let preprocess_started = Instant::now();
        let state_cols = self.state_col_size(&self.params);
        let output_cols = self.gadget_col_size(&self.params);
        let input_bit_count = self.input_bit_count();
        info!("diamond injector gpu preprocess: starting");

        // Persist the empty-prefix seed once. This seed is not a trapdoor
        // preimage; it is the initial encoding that online evaluation uses as
        // p_{epsilon,0} before any digit transition is applied.
        let secret_epsilon_bytes = self.load_or_sample_secret_mask_bytes(self.secret_epsilon_id());
        if !self.matrix_exists(self.p_epsilon_id()) {
            let (b0_bytes, _) = self.load_or_sample_b_checkpoint_bytes(0);
            let b0_matrix = M::from_compact_bytes(&self.params, &b0_bytes);
            let secret_epsilon = M::from_compact_bytes(&self.params, &secret_epsilon_bytes);
            let p_epsilon = self.build_initial_encoding(&b0_matrix, &secret_epsilon);
            self.write_matrix(self.p_epsilon_id(), &p_epsilon);
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
                        Arc::<[u8]>::from(self.load_or_sample_secret_mask_bytes(
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
                        if !self.matrix_exists(&self.preprocess_chunk_id(&task)) {
                            stage_tasks.push(task);
                        }
                    }
                }
            }
            total_tasks += stage_tasks.len();
            if stage_tasks.is_empty() {
                info!(
                    level,
                    progress = preprocess_progress_label(0, 0),
                    "diamond injector gpu preprocess: K stage already checkpointed"
                );
                continue;
            }

            let (source_b_bytes, source_trapdoor_bytes) =
                self.load_or_sample_b_checkpoint_bytes(level - 1);
            let (target_b_bytes, _) = self.load_or_sample_b_checkpoint_bytes(level);
            completed_tasks += self.preprocess_k_stage_gpu(
                level,
                stage_tasks,
                &source_b_bytes,
                &source_trapdoor_bytes,
                &target_b_bytes,
                &secret_mask_bytes,
            );
        }

        // Process every L_j chunk using only the final source trapdoor. The
        // one/bit input public keys stay on CPU as compact bytes and are loaded per
        // task immediately before building the target chunk.
        let mut l_tasks = Vec::new();
        for input_idx in 0..=input_bit_count {
            for chunk_idx in 0..column_chunk_count(output_cols) {
                let task = DiamondPreprocessChunkTask::L { input_idx, chunk_idx };
                if !self.matrix_exists(&self.preprocess_chunk_id(&task)) {
                    l_tasks.push(task);
                }
            }
        }
        total_tasks += l_tasks.len();
        if l_tasks.is_empty() {
            info!(
                progress = preprocess_progress_label(0, 0),
                "diamond injector gpu preprocess: L stage already checkpointed"
            );
        } else {
            let (source_b_bytes, source_trapdoor_bytes) =
                self.load_or_sample_b_checkpoint_bytes(self.input_count);
            let one_bytes = Arc::<[u8]>::from(one.matrix.to_compact_bytes());
            let input_pubkey_bytes = input_digits
                .iter()
                .map(|pubkey| Arc::<[u8]>::from(pubkey.matrix.to_compact_bytes()))
                .collect::<Vec<_>>();
            completed_tasks += self.preprocess_l_stage_gpu(
                l_tasks,
                &source_b_bytes,
                &source_trapdoor_bytes,
                one_bytes,
                &input_pubkey_bytes,
            );
        }

        // Process every decoder chunk against the same final source trapdoor.
        // Decoder public keys are kept as CPU-side compact bytes and materialized
        // on a device only for the task that needs them.
        let mut m_tasks = Vec::new();
        for decoder_idx in 0..self.decoder_count {
            for chunk_idx in 0..column_chunk_count(output_cols) {
                let task = DiamondPreprocessChunkTask::M { decoder_idx, chunk_idx };
                if !self.matrix_exists(&self.preprocess_chunk_id(&task)) {
                    m_tasks.push(task);
                }
            }
        }
        total_tasks += m_tasks.len();
        if m_tasks.is_empty() {
            info!(
                progress = preprocess_progress_label(0, 0),
                "diamond injector gpu preprocess: M stage already checkpointed"
            );
        } else {
            let (source_b_bytes, source_trapdoor_bytes) =
                self.load_or_sample_b_checkpoint_bytes(self.input_count);
            let decoder_pubkey_bytes = decoders
                .iter()
                .map(|pubkey| Arc::<[u8]>::from(pubkey.matrix.to_compact_bytes()))
                .collect::<Vec<_>>();
            completed_tasks += self.preprocess_m_stage_gpu(
                m_tasks,
                &source_b_bytes,
                &source_trapdoor_bytes,
                &decoder_pubkey_bytes,
            );
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
        input_digits: &[u32],
        one: &BggPublicKey<M>,
        input_digit_pubkeys: &[BggPublicKey<M>],
        decoders: &[BggPublicKey<M>],
    ) -> (BggEncoding<M>, Vec<BggEncoding<M>>, Vec<BggEncoding<M>>) {
        self.validate_lengths(input_digit_pubkeys, decoders);
        self.validate_digits(input_digits);
        let metadata = self.read_metadata();
        assert_eq!(
            metadata.input_count, self.input_count,
            "DiamondInjector metadata input count mismatch"
        );
        assert_eq!(metadata.base, self.base, "DiamondInjector metadata base mismatch");
        assert_eq!(
            metadata.decoder_count, self.decoder_count,
            "DiamondInjector metadata decoder count mismatch"
        );

        let online_started = Instant::now();
        let state_cols = self.state_col_size(&self.params);
        let output_cols = self.gadget_col_size(&self.params);
        // Start from the persisted empty-prefix seed.
        let mut states = vec![self.read_matrix(self.p_epsilon_id())];

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
                            rhs_chunk_bytes: self.read_chunk_bytes(&rhs_id, state_cols),
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

        let p0_bytes = Arc::<[u8]>::from(states[0].to_compact_bytes());
        let mut family_specs = HashMap::new();
        // Turn the surviving base branch into the encoding of one.
        family_specs.insert(
            DiamondEvalFamily::One,
            DiamondEvalFamilySpec {
                lhs_bytes: p0_bytes.clone(),
                rhs_chunk_bytes: self.read_chunk_bytes(&self.l_id(0), output_cols),
            },
        );
        // Turn each bit-specific branch into the encoding for the chosen bit.
        for bit_output_idx in 0..self.input_bit_count() {
            let digit_idx = bit_output_idx / self.batch_bits();
            let bit_idx = bit_output_idx % self.batch_bits();
            let state_idx = self.bit_state_idx(digit_idx, bit_idx);
            family_specs.insert(
                DiamondEvalFamily::Input(bit_output_idx),
                DiamondEvalFamilySpec {
                    lhs_bytes: Arc::<[u8]>::from(states[state_idx].to_compact_bytes()),
                    rhs_chunk_bytes: self.read_chunk_bytes(&self.l_id(state_idx), output_cols),
                },
            );
        }
        // Turn the surviving base branch into every decoder output.
        for decoder_idx in 0..self.decoder_count {
            family_specs.insert(
                DiamondEvalFamily::Decoder(decoder_idx),
                DiamondEvalFamilySpec {
                    lhs_bytes: p0_bytes.clone(),
                    rhs_chunk_bytes: self.read_chunk_bytes(&self.m_id(decoder_idx), output_cols),
                },
            );
        }
        let outputs = self.gpu_left_mul_families(family_specs);

        let one_output = self.build_output_encoding(
            outputs
                .get(&DiamondEvalFamily::One)
                .unwrap_or_else(|| panic!("missing gpu online_eval one output"))
                .clone(),
            one.clone(),
            Some(M::P::const_one(&self.params)),
        );
        let digit_outputs = input_digits
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(digit_idx, digit_value)| {
                (0..self.batch_bits())
                    .map(move |bit_idx| (digit_idx, digit_value as usize, bit_idx))
            })
            .map(|(digit_idx, digit_value, bit_idx)| {
                let bit_output_idx = self.bit_pubkey_idx(digit_idx, bit_idx);
                self.build_output_encoding(
                    outputs
                        .get(&DiamondEvalFamily::Input(bit_output_idx))
                        .unwrap_or_else(|| {
                            panic!("missing gpu online_eval input output {}", bit_output_idx)
                        })
                        .clone(),
                    input_digit_pubkeys[bit_output_idx].clone(),
                    Some(M::P::from_usize_to_constant(
                        &self.params,
                        self.digit_bit_value(digit_value, bit_idx),
                    )),
                )
            })
            .collect::<Vec<_>>();
        let decoder_outputs = decoders
            .iter()
            .cloned()
            .enumerate()
            .map(|(decoder_idx, pubkey)| {
                self.build_output_encoding(
                    outputs
                        .get(&DiamondEvalFamily::Decoder(decoder_idx))
                        .unwrap_or_else(|| {
                            panic!("missing gpu online_eval decoder output {}", decoder_idx)
                        })
                        .clone(),
                    pubkey,
                    Some(M::P::const_zero(&self.params)),
                )
            })
            .collect::<Vec<_>>();

        info!(
            elapsed_s = online_started.elapsed().as_secs_f64(),
            "diamond injector gpu online_eval: finished"
        );
        (one_output, digit_outputs, decoder_outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DiamondInjector, InputInjector};
    use crate::{
        __PAIR, __TestState,
        bgg::public_key::BggPublicKey,
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{
            DistType, PolyHashSampler,
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

    fn sample_pubkey(
        params: &GpuDCRTPolyParams,
        hash_key: [u8; 32],
        tag: &str,
    ) -> BggPublicKey<GpuDCRTPolyMatrix> {
        let matrix = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            params,
            hash_key,
            tag,
            super::super::DIAMOND_SECRET_SIZE,
            super::super::DIAMOND_SECRET_SIZE * params.modulus_digits(),
            DistType::FinRingDist,
        );
        BggPublicKey::new(matrix, true)
    }

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

        let hash_key = [11u8; 32];
        let input_count = 3;
        let base = 4;
        let batch_bits = 2;
        let decoder_count = 2;
        let dir = tempdir().expect("temporary directory should be created");

        let injector = TestInjector::new(
            params.clone(),
            hash_key,
            input_count,
            base,
            decoder_count,
            4.578,
            0.0,
            dir.path().to_path_buf(),
        )
        .with_gpu_device_ids(gpu_ids.clone());

        let one_pubkey = sample_pubkey(&params, hash_key, "diamond_gpu_one_pubkey");
        let input_pubkeys = (0..input_count * batch_bits)
            .map(|bit_idx| {
                sample_pubkey(&params, hash_key, &format!("diamond_gpu_input_pubkey_{bit_idx}"))
            })
            .collect::<Vec<_>>();
        let decoder_pubkeys = (0..decoder_count)
            .map(|decoder_idx| {
                sample_pubkey(
                    &params,
                    hash_key,
                    &format!("diamond_gpu_decoder_pubkey_{decoder_idx}"),
                )
            })
            .collect::<Vec<_>>();

        injector.preprocess(&one_pubkey, &input_pubkeys, &decoder_pubkeys);

        let digits = vec![2u32, 1u32, 3u32];
        let (one_output, digit_outputs, decoder_outputs) =
            injector.online_eval(&digits, &one_pubkey, &input_pubkeys, &decoder_pubkeys);
        assert_eq!(digit_outputs.len(), input_count * batch_bits);
        assert_eq!(decoder_outputs.len(), decoder_count);

        let mut secret_matrix = injector.read_matrix(injector.secret_epsilon_id());
        for (digit_idx, digit_value) in digits.iter().copied().enumerate() {
            secret_matrix = secret_matrix *
                injector
                    .read_matrix(&injector.digit_secret_id(digit_idx + 1, digit_value as usize));
        }
        let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, super::super::DIAMOND_SECRET_SIZE);
        let secret_times_gadget = secret_matrix.clone() * &gadget;

        assert_eq!(one_output.vector, secret_matrix.clone() * (&one_pubkey.matrix - &gadget));
        assert_eq!(one_output.plaintext, Some(TestPoly::const_one(&params)));

        for digit_idx in 0..input_count {
            for bit_idx in 0..batch_bits {
                let output_idx = digit_idx * batch_bits + bit_idx;
                let output = &digit_outputs[output_idx];
                let bit_value = ((digits[digit_idx] as usize) >> bit_idx) & 1;
                let plaintext = TestPoly::from_usize_to_constant(&params, bit_value);
                let expected = secret_matrix.clone() * &input_pubkeys[output_idx].matrix -
                    (secret_times_gadget.clone() * plaintext.clone());
                assert_eq!(output.vector, expected);
                assert_eq!(output.plaintext, Some(plaintext));
            }
        }

        for (decoder_idx, output) in decoder_outputs.iter().enumerate() {
            assert_eq!(output.vector, secret_matrix.clone() * &decoder_pubkeys[decoder_idx].matrix);
            assert_eq!(output.plaintext, Some(TestPoly::const_zero(&params)));
        }
    }
}
