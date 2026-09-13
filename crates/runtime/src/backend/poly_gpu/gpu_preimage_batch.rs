//! Allocating Preimage waves with independent bounded retries.

use super::*;
use crate::backend::PreimageRequest;
use mxx_primitives::sampler::{PolyTrapdoorSampler, trapdoor::gpu::GpuPreimageAttempt};

impl GpuDcrtBackend {
    pub(super) fn execute_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<GpuFleetMatrix, GpuFleetTrapdoor>>,
    ) -> Result<Vec<GpuFleetSmallMatrix>, PolyBackendError> {
        let attempts = mxx_primitives::env::gpu_preimage_max_tile_attempts()
            .map_err(PolyBackendError::GpuSubmission)?;
        let mut replicas = HashMap::new();
        for request in &requests {
            if request.target.col_size() != 0 && !replicas.contains_key(&request.public.id) {
                replicas.insert(request.public.id, self.full_matrix_replicas(&request.public)?);
            }
        }
        self.restart_runtime_pilot_after_fixed_inputs()
            .map_err(PolyBackendError::GpuCalibration)?;
        let replicas = Arc::new(replicas);
        let schedules = requests
            .iter()
            .map(|request| {
                let columns = request.target.col_size();
                GpuColumnSchedule::new(
                    columns,
                    (0..self.devices.len()).map(|device| self.active_role_width(device)).collect(),
                    self.fresh_column_intervals(columns)?,
                )
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let requests = Arc::new(requests);
        let mut waves = schedules.iter().map(GpuColumnSchedule::waves).collect::<Vec<_>>();
        let mut outputs = (0..requests.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        loop {
            let mut device_jobs = vec![Vec::new(); self.devices.len()];
            for (instance, waves) in waves.iter_mut().enumerate() {
                for job in waves.next().into_iter().flatten() {
                    device_jobs[job.device].push((instance, job));
                }
            }
            if device_jobs.iter().all(Vec::is_empty) {
                break;
            }
            let requests = requests.clone();
            let replicas = replicas.clone();
            let produced = self.enqueue.map(&mut self.devices, move |device, (device_id, backend)| {
                let mut groups = std::collections::BTreeMap::new();
                for &(instance, job) in &device_jobs[device] {
                    let request = &requests[instance];
                    let public = &replicas[&request.public.id][device];
                    groups.entry((public.params().context_identity(), public.level(), public.size(),
                        request.sigma.to_bits(), job.end - job.start))
                        .or_insert_with(Vec::new).push((instance, job));
                }
                let mut produced = Vec::new();
                for jobs in groups.values() {
                    let first = &requests[jobs[0].0];
                    let params = replicas[&first.public.id][device].params().clone();
                    let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, first.sigma);
                    let mut tiles = Vec::with_capacity(jobs.len());
                    let mut destinations = Vec::with_capacity(jobs.len());
                    for &(instance, job) in jobs {
                        let request = &requests[instance];
                        let mut target = match request.target.column_range(job.start, job.end) {
                            PolyMatrixColumnData::Resident { value, start, end } =>
                                Self::matrix_piece_on_device(backend, &value, start, end)?,
                            PolyMatrixColumnData::CpuStaging { bytes, start, end, .. } =>
                                GpuDCRTPolyMatrix::from_cpu_staging_columns(&params, &bytes, start, end),
                        };
                        if !target.is_ntt() { target.ntt_all_in_place(); }
                        tiles.push(target);
                        destinations.push(GpuDCRTPolyTrapdoorSampler::preimage_destination(
                            &params, request.matrix_type.rows, job.end - job.start,
                            request.max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?)
                            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?);
                    }
                    let mut accepted = vec![false; jobs.len()];
                    for attempt in 0..attempts {
                        let mut indices = Vec::new();
                        let mut pending = Vec::new();
                        for (index, destination) in destinations.iter_mut().enumerate() {
                            if accepted[index] { continue; }
                            let (instance, job) = jobs[index];
                            let request = &requests[instance];
                            let trapdoor = request.trapdoor.values.iter()
                                .find(|trapdoor| trapdoor.r.params() == &params)
                                .ok_or_else(|| PolyBackendError::GpuSubmission("preimage trapdoor context is missing".into()))?;
                            indices.push(index);
                            pending.push(GpuPreimageAttempt {
                                trapdoor, public: &replicas[&request.public.id][device], target: &tiles[index], destination,
                                column_start: 0,
                                global_column_start: preimage_seed_column_start(request.target.global_column_start(), job.start)?,
                                attempt, seed: request.randomness_seed,
                            });
                        }
                        let flags = sampler.preimage_attempt_batch(&params, pending, &mut ())
                            .map_err(PolyBackendError::GpuSubmission)?;
                        for (index, flag) in indices.into_iter().zip(flags) { accepted[index] = flag; }
                        if accepted.iter().all(|flag| *flag) { break; }
                    }
                    if let Some(index) = accepted.iter().position(|flag| !flag) {
                        let (instance, job) = jobs[index];
                        return Err(PolyBackendError::GpuSubmission(format!(
                            "preimage instance {instance} columns {}..{} exhausted {attempts} attempts", job.start, job.end)));
                    }
                    produced.extend(jobs.iter().zip(destinations).map(|((instance, job), value)|
                        (*instance, GpuColumnShard { device_id: *device_id, global_column_start: job.start, value })));
                }
                Ok::<_, PolyBackendError>(produced)
            }).map_err(PolyBackendError::from)?;
            for (instance, shard) in produced.into_iter().flatten() {
                outputs[instance].push(shard);
            }
        }
        Ok(outputs
            .into_par_iter()
            .zip(requests.par_iter())
            .map(|(mut shards, request)| {
                shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
                GpuFleetSmallMatrix::new(
                    request.matrix_type.rows,
                    request.target.col_size(),
                    shards,
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::{
        poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
        sampler::PolyUniformSampler,
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_allocating_preimage_batch_handles_staged_targets_and_tails() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3)
            .max(2);
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, 5.0);
        let (trapdoor, public) = sampler.trapdoor(&params, 1);
        let public = Arc::new(GpuFleetMatrix::from_matrix(public));
        let trapdoor = Arc::new(GpuFleetTrapdoor { values: Arc::new(vec![trapdoor]) });
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: public.columns,
            columns,
        };
        let targets = (0..3)
            .map(|_| {
                GpuDCRTPolyUniformSampler::new().sample_uniform(
                    &params,
                    1,
                    columns,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let requests: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(index, target)| {
                let source: Arc<dyn PolyMatrixColumnSource<GpuFleetMatrix>> = if index == 1 {
                    Arc::new(PreimageTarget::staged(
                        &params,
                        1,
                        columns,
                        Arc::new(target.clone().into_cpu_staging_bytes()),
                    ))
                } else {
                    Arc::new(PreimageTarget {
                        rows: 1,
                        columns,
                        data: PolyMatrixColumnData::Resident {
                            value: Arc::new(GpuFleetMatrix::from_matrix(target.clone())),
                            start: 0,
                            end: columns,
                        },
                    })
                };
                PreimageRequest {
                    matrix_type: ty.clone(),
                    sigma: 5.0,
                    gadget_base: BigInt::from(16),
                    digit_count: params.modulus_digits(),
                    max_coefficient_bound: &ty.modulus / 2,
                    trapdoor: trapdoor.clone(),
                    public: public.clone(),
                    target: source,
                    randomness_seed: rand::random(),
                }
            })
            .collect();
        let mut backend = super::super::super::gpu_backend_on([params.clone()], [device]);
        backend.set_column_widths_for_operation(
            [213; 32],
            GpuColumnWidths { gpu0: Some(columns - 1), nonzero: None },
        );
        backend.select_gpu_operation([213; 32]).unwrap();
        let mut rejected = requests.clone();
        rejected[0].max_coefficient_bound = BigInt::from(0);
        rejected[0].target = Arc::new(PreimageTarget {
            rows: 1,
            columns,
            data: PolyMatrixColumnData::Resident {
                value: Arc::new(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::unit_row_vector(
                    &params, columns, 0,
                ))),
                start: 0,
                end: columns,
            },
        });
        let failure = backend.sample_preimage_batch(rejected).unwrap_err().to_string();
        assert!(failure.contains("preimage instance 0 columns"), "{failure}");
        assert!(failure.contains("exhausted"), "{failure}");
        // The failed worker returned its device state. Real outputs and pending
        // native readers from that batch must not obstruct the next submission.
        let outputs = backend.sample_preimage_batch(requests).unwrap();
        assert_eq!(outputs.len(), targets.len());
        for (output, target) in outputs.iter().zip(&targets) {
            assert_eq!(output.size(), (public.columns, columns));
            assert_eq!(output.shards.len(), 2);
            for shard in output.shards.iter() {
                let actual = public.shards[0].value.multiply_small_rhs(&shard.value).unwrap();
                assert_eq!(
                    actual,
                    target.slice_columns(
                        shard.global_column_start,
                        shard.global_column_start + shard.value.size().1
                    )
                );
            }
        }
    }
}
