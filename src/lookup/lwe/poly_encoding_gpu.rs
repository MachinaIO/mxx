use super::*;
use crate::{
    bench_estimator::{PolyEncodingChunkBenchMeasurement, benchmark_gate_operation},
    lookup::lwe::{
        derive_k_low_chunk, encoding::public_lookup_slot_gpu, k_high_chunk_count,
        public_lookup_gpu_device_ids, read_k_high_chunk,
    },
};
use rayon::prelude::*;
use std::{path::Path, sync::Arc, time::Instant};
use tracing::{debug, info};

#[derive(Clone, Copy, Debug)]
enum BenchContributionFamily {
    KHigh,
    KLow,
}

#[derive(Clone, Copy, Debug)]
struct BenchContributionTask {
    chunk_idx: usize,
    family: BenchContributionFamily,
}

struct BenchSharedByDevice<M: PolyMatrix> {
    params: <<M as PolyMatrix>::P as Poly>::Params,
    c_b: M,
}

#[derive(Clone)]
struct BenchLoadedContributionTask<M: PolyMatrix> {
    task: BenchContributionTask,
    device_slot: usize,
    rhs_chunk: Option<M>,
}

struct BenchComputedContributionTask {
    bytes: Arc<[u8]>,
}

fn prepare_bench_shared_by_device<M>(
    params: &<M::P as Poly>::Params,
    c_b_compact_bytes: &[u8],
) -> Vec<BenchSharedByDevice<M>>
where
    M: PolyMatrix + Send + Sync + 'static,
{
    let local_params = params.clone();
    let c_b = M::from_compact_bytes(&local_params, c_b_compact_bytes);
    vec![BenchSharedByDevice { params: local_params, c_b }]
}

fn load_bench_input_vector_by_device<M>(
    shared: &[BenchSharedByDevice<M>],
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

fn load_bench_wave<M>(
    shared: &[BenchSharedByDevice<M>],
    tasks: &[BenchContributionTask],
    dir: &Path,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
) -> Vec<BenchLoadedContributionTask<M>>
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
                BenchContributionFamily::KHigh => Some(read_k_high_chunk::<M>(
                    &shared[device_slot].params,
                    dir,
                    gate_id,
                    lut_id,
                    row_size,
                    lut_entry_idx,
                    task.chunk_idx,
                )),
                BenchContributionFamily::KLow => None,
            };
            BenchLoadedContributionTask { task, device_slot, rhs_chunk }
        })
        .collect()
}

fn compute_bench_wave<M, SH>(
    hash_key: [u8; 32],
    shared: &[BenchSharedByDevice<M>],
    input_vector_by_device: &[M],
    loaded_wave: Vec<BenchLoadedContributionTask<M>>,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    lut_entry_idx: usize,
) -> Vec<BenchComputedContributionTask>
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
                BenchContributionFamily::KHigh => loaded
                    .rhs_chunk
                    .expect("k_high benchmark contribution must load a checkpoint rhs chunk"),
                BenchContributionFamily::KLow => derive_k_low_chunk::<M, SH>(
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
                BenchContributionFamily::KHigh => &shared_dev.c_b,
                BenchContributionFamily::KLow => &input_vector_by_device[loaded.device_slot],
            };
            assert_eq!(
                lhs.col_size(),
                rhs_chunk.row_size(),
                "LWE GPU bench contribution dimension mismatch: family={:?}, chunk_idx={}, gate_id={}, lut_id={}, lut_entry_idx={}, lhs={}x{}, rhs={}x{}",
                loaded.task.family,
                loaded.task.chunk_idx,
                gate_id,
                lut_id,
                lut_entry_idx,
                lhs.row_size(),
                lhs.col_size(),
                rhs_chunk.row_size(),
                rhs_chunk.col_size()
            );
            let output = lhs * &rhs_chunk;
            BenchComputedContributionTask { bytes: Arc::<[u8]>::from(output.into_compact_bytes()) }
        })
        .collect()
}

fn reduce_bench_contribution_bytes_by_chunk<M>(
    reduction_params: &<M::P as Poly>::Params,
    contribution_bytes_by_chunk: Vec<Vec<Arc<[u8]>>>,
) -> M
where
    M: PolyMatrix + Send + Sync + 'static,
{
    let contribution_count = contribution_bytes_by_chunk
        .first()
        .map(Vec::len)
        .expect("LWE GPU public-lookup benchmark requires at least one output chunk");
    assert!(
        contribution_count > 0,
        "LWE GPU public-lookup benchmark requires at least one contribution"
    );
    for (chunk_idx, contribution_bytes) in contribution_bytes_by_chunk.iter().enumerate() {
        assert_eq!(
            contribution_bytes.len(),
            contribution_count,
            "chunk {} contribution count mismatch",
            chunk_idx
        );
    }

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

pub(super) fn benchmark_public_lookup_slot_gpu<M, SH>(
    evaluator: &LWEBGGPolyEncodingPltEvaluator<M, SH>,
    samples: &crate::bench_estimator::BggPolyEncodingBenchSamples<'_, M>,
    iterations: usize,
) -> PolyEncodingChunkBenchMeasurement
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static + Send + Sync,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let plaintext_compact_bytes_by_slot =
        samples.public_lut_input.plaintext_bytes.as_ref().expect(
            "the BGG poly encoding should reveal plaintexts for public-lookup benchmarking",
        );
    let row_size = samples.public_lut_input.pubkey.matrix.row_size();
    let input_vector =
        M::from_compact_bytes(samples.params, samples.public_lut_input.vector_bytes[0].as_ref());
    let input_plaintext =
        M::P::from_compact_bytes(samples.params, plaintext_compact_bytes_by_slot[0].as_ref());
    let z_u64 = input_plaintext.const_coeff_u64();
    let (k, _) = samples
        .public_lut
        .get(samples.params, z_u64)
        .unwrap_or_else(|| panic!("{:?} is not exist in public lookup f", z_u64));
    let lut_entry_idx = usize::try_from(k).expect("LUT row index must fit in usize");
    let shared = prepare_bench_shared_by_device::<M>(
        samples.params,
        evaluator.c_b_compact_bytes_by_slot[0].as_ref(),
    );
    let shared_dev = shared
        .first()
        .expect("LWE GPU public-lookup benchmark requires at least one prepared device");
    let input_vector_by_device = load_bench_input_vector_by_device(&shared[..1], &input_vector);
    let output_chunk_count = k_high_chunk_count::<M>(samples.params, row_size);

    let k_high_task =
        BenchContributionTask { chunk_idx: 0, family: BenchContributionFamily::KHigh };
    let k_low_task = BenchContributionTask { chunk_idx: 0, family: BenchContributionFamily::KLow };

    let k_high_load_bench = benchmark_gate_operation(iterations, || {
        load_bench_wave::<M>(
            &shared[..1],
            &[k_high_task],
            &evaluator.dir_path,
            samples.public_lut_gate_id,
            samples.public_lut_id,
            row_size,
            lut_entry_idx,
        )
    });
    let k_low_load_bench = benchmark_gate_operation(iterations, || {
        load_bench_wave::<M>(
            &shared[..1],
            &[k_low_task],
            &evaluator.dir_path,
            samples.public_lut_gate_id,
            samples.public_lut_id,
            row_size,
            lut_entry_idx,
        )
    });

    let loaded_k_high = load_bench_wave::<M>(
        &shared[..1],
        &[k_high_task],
        &evaluator.dir_path,
        samples.public_lut_gate_id,
        samples.public_lut_id,
        row_size,
        lut_entry_idx,
    );
    let loaded_k_low = load_bench_wave::<M>(
        &shared[..1],
        &[k_low_task],
        &evaluator.dir_path,
        samples.public_lut_gate_id,
        samples.public_lut_id,
        row_size,
        lut_entry_idx,
    );
    let k_high_compute_bench = benchmark_gate_operation(iterations, || {
        compute_bench_wave::<M, SH>(
            evaluator.hash_key,
            &shared[..1],
            &input_vector_by_device,
            loaded_k_high.clone(),
            samples.public_lut_gate_id,
            samples.public_lut_id,
            row_size,
            lut_entry_idx,
        )
    });
    let k_low_compute_bench = benchmark_gate_operation(iterations, || {
        compute_bench_wave::<M, SH>(
            evaluator.hash_key,
            &shared[..1],
            &input_vector_by_device,
            loaded_k_low.clone(),
            samples.public_lut_gate_id,
            samples.public_lut_id,
            row_size,
            lut_entry_idx,
        )
    });

    let contribution_bytes_by_chunk = (0..output_chunk_count)
        .map(|chunk_idx| {
            let k_high_loaded = load_bench_wave::<M>(
                &shared[..1],
                &[BenchContributionTask { chunk_idx, family: BenchContributionFamily::KHigh }],
                &evaluator.dir_path,
                samples.public_lut_gate_id,
                samples.public_lut_id,
                row_size,
                lut_entry_idx,
            );
            let k_low_loaded = load_bench_wave::<M>(
                &shared[..1],
                &[BenchContributionTask { chunk_idx, family: BenchContributionFamily::KLow }],
                &evaluator.dir_path,
                samples.public_lut_gate_id,
                samples.public_lut_id,
                row_size,
                lut_entry_idx,
            );
            let k_high_bytes = compute_bench_wave::<M, SH>(
                evaluator.hash_key,
                &shared[..1],
                &input_vector_by_device,
                k_high_loaded,
                samples.public_lut_gate_id,
                samples.public_lut_id,
                row_size,
                lut_entry_idx,
            )
            .into_iter()
            .next()
            .expect("benchmark must compute a representative k_high task")
            .bytes;
            let k_low_bytes = compute_bench_wave::<M, SH>(
                evaluator.hash_key,
                &shared[..1],
                &input_vector_by_device,
                k_low_loaded,
                samples.public_lut_gate_id,
                samples.public_lut_id,
                row_size,
                lut_entry_idx,
            )
            .into_iter()
            .next()
            .expect("benchmark must compute a representative k_low task")
            .bytes;
            vec![k_high_bytes, k_low_bytes]
        })
        .collect::<Vec<_>>();
    let stage2_bench = benchmark_gate_operation(iterations, || {
        reduce_bench_contribution_bytes_by_chunk::<M>(
            &shared_dev.params,
            contribution_bytes_by_chunk.clone(),
        )
    });

    let stage1_load_latency = k_high_load_bench.time.max(k_low_load_bench.time);
    let stage1_compute_latency = k_high_compute_bench.time.max(k_low_compute_bench.time);
    let latency = stage1_load_latency.max(stage1_compute_latency) + stage2_bench.time;
    let peak_vram = k_high_load_bench
        .peak_vram
        .max(k_low_load_bench.peak_vram)
        .max(k_high_compute_bench.peak_vram)
        .max(k_low_compute_bench.peak_vram)
        .max(stage2_bench.peak_vram);

    let max_parallelism = u128::try_from(
        output_chunk_count.checked_mul(2).expect("stage-1 task count overflowed usize"),
    )
    .expect("stage-1 task count overflowed u128");
    PolyEncodingChunkBenchMeasurement::new(
        latency * max_parallelism as f64,
        latency,
        max_parallelism,
        peak_vram,
    )
}

pub(super) fn public_lookup_poly_gpu<M, SH>(
    evaluator: &LWEBGGPolyEncodingPltEvaluator<M, SH>,
    params: &<BggPolyEncoding<M> as Evaluable>::Params,
    plt: &PublicLut<<BggPolyEncoding<M> as Evaluable>::P>,
    input: &BggPolyEncoding<M>,
    plaintext_compact_bytes_by_slot: &[Arc<[u8]>],
    gate_id: GateId,
    lut_id: usize,
) -> (Vec<Arc<[u8]>>, Vec<Arc<[u8]>>)
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static + Send + Sync,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    let start = Instant::now();
    let num_slots = input.num_slots();
    let device_count = public_lookup_gpu_device_ids().len().max(1);
    let row_size = input.pubkey.matrix.row_size();
    info!(
        "LWE GPU poly-encoding public lookup start: gate_id={}, lut_id={}, slot_count={}, devices={}",
        gate_id, lut_id, num_slots, device_count
    );

    let mut output_vector_bytes = Vec::with_capacity(num_slots);
    let mut output_plaintext_bytes = Vec::with_capacity(num_slots);

    // TODO: Parallelize this outer slot loop across multiple devices once we have a slot-level
    // scheduler that can coordinate safely with the per-slot multi-device GPU work already used
    // by `public_lookup_slot_gpu` without issuing concurrent compute requests to the same device.
    for slot in 0..num_slots {
        let slot_started = Instant::now();
        let c_b_bytes = evaluator.c_b_compact_bytes_by_slot[slot].as_ref();
        let input_vector = input.vector_for_params(params, slot);
        let input_plaintext =
            M::P::from_compact_bytes(params, plaintext_compact_bytes_by_slot[slot].as_ref());
        let slot_output = public_lookup_slot_gpu::<M, SH>(
            params,
            plt,
            &evaluator.dir_path,
            evaluator.hash_key,
            row_size,
            c_b_bytes,
            &input_vector,
            &input_plaintext,
            gate_id,
            lut_id,
            None,
        );
        output_vector_bytes.push(Arc::<[u8]>::from(slot_output.vector.into_compact_bytes()));
        output_plaintext_bytes.push(Arc::<[u8]>::from(
            slot_output
                .plaintext
                .expect("poly public lookup should reveal plaintexts")
                .to_compact_bytes(),
        ));
        debug!(
            "LWE GPU poly-encoding slot finished: gate_id={}, lut_id={}, slot={}/{}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            slot + 1,
            num_slots,
            slot_started.elapsed().as_secs_f64() * 1000.0
        );
    }

    info!(
        "LWE GPU poly-encoding public lookup finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
        gate_id,
        lut_id,
        num_slots,
        start.elapsed().as_secs_f64() * 1000.0
    );
    (output_vector_bytes, output_plaintext_bytes)
}
