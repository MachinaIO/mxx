//! Prepared Preimage candidates, sharing phase claims and independent retries.

use super::*;
use mxx_primitives::sampler::trapdoor::gpu::{
    GpuPreimageAttempt, GpuPreimageBatchPhase, GpuPreimageBatchResources,
};

pub(super) struct PreparedPreimageJob<'a> {
    pub(super) output: GpuSmallMatrix,
    pub(super) destination_column: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) payload: &'a PreimagePayload,
    pub(super) public: &'a GpuDCRTPolyMatrix,
}

struct PhaseClaims<'a> {
    broker: &'a PreparedClaimBroker,
    claims: &'a [Vec<GpuTracedClaim>; 16],
    parameters: &'a GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
}

impl GpuPreimageBatchResources for PhaseClaims<'_> {
    fn run<T>(
        &mut self,
        phase: GpuPreimageBatchPhase,
        jobs: usize,
        operation: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let claims = &self.claims[phase as usize];
        if jobs == 1 {
            return self.broker.hold_traced(claims, operation);
        }
        let workspace_kind = match phase {
            GpuPreimageBatchPhase::Cutoff => Some(GpuPreparedSlotKind::CompactWorkspace),
            GpuPreimageBatchPhase::SampleP1 | GpuPreimageBatchPhase::Gadget => {
                Some(GpuPreparedSlotKind::SamplerWorkspace)
            }
            _ => None,
        };
        if let Some(workspace_kind) = workspace_kind {
            // Native sampling/cutoff acquires all staging before event claims.
            let expanded = [true, false]
                .into_iter()
                .flat_map(|workspace| {
                    (0..jobs).flat_map(move |_| {
                        claims
                            .iter()
                            .copied()
                            .filter(move |claim| (claim.kind() == workspace_kind) == workspace)
                    })
                })
                .collect::<Vec<_>>();
            return self.broker.hold_traced(&expanded, operation);
        }
        let claims = (0..jobs)
            .flat_map(|_| claims.iter().copied())
            .chain(if phase == GpuPreimageBatchPhase::Residual {
                PreimageClaimPlan::residual_batch_metadata(
                    self.parameters,
                    self.rows,
                    self.columns,
                    jobs,
                )?
            } else {
                Vec::new()
            })
            .collect::<Vec<_>>();
        self.broker.hold_traced(&claims, operation)
    }
}

impl PreparedOperation {
    /// Jobs share the operation class, parameter context and actual column
    /// width. Owners, global offsets, seeds and acceptance remain per job.
    pub(super) fn run_preimage_batch(
        &self,
        mut jobs: Vec<PreparedPreimageJob<'_>>,
        broker: &PreparedClaimBroker,
    ) -> Result<Vec<GpuSmallMatrix>, String> {
        let Self::Preimage { ty, bound, sigma_bits, public_rows, plan, .. } = self else {
            unreachable!("Preimage batch operation")
        };
        let Some(first) = jobs.first() else { return Ok(Vec::new()) };
        let parameters = first.output.params().clone();
        let width = first.end - first.start;
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&parameters, f64::from_bits(*sigma_bits));
        let attempt_claims =
            plan.attempt_claims(&parameters, *public_rows, width, ty.rows, bound)?;
        let tile_claims = plan.tile_claims(&parameters, *public_rows, width)?;
        let mut inputs = Vec::with_capacity(jobs.len());
        for job in &mut jobs {
            let trapdoor = job
                .payload
                .trapdoors
                .iter()
                .find(|trapdoor| trapdoor.r.params() == &parameters)
                .ok_or("preimage trapdoor is not resident in this context")?;
            let tile = broker
                .hold_traced(&tile_claims, || {
                    materialize_preimage_tile(
                        &parameters,
                        &job.payload.target,
                        *public_rows,
                        job.start,
                        job.end,
                    )
                })
                .map_err(|error| format!("preimage target: {error}"))?;
            if job.destination_column == 0 {
                broker
                    .hold_traced(&plan.destination, || {
                        job.output.prepare_preimage_hard_cutoff();
                        Ok(())
                    })
                    .map_err(|error| format!("preimage destination: {error}"))?;
            }
            inputs.push((trapdoor, job.public, tile));
        }
        let mut phases = PhaseClaims {
            broker,
            claims: &attempt_claims,
            parameters: &parameters,
            rows: *public_rows,
            columns: width,
        };
        let mut accepted = vec![false; jobs.len()];
        for attempt in 0..plan.attempts {
            let mut indices = Vec::new();
            let pending = jobs
                .iter_mut()
                .zip(&inputs)
                .enumerate()
                .filter_map(|(index, (job, (trapdoor, public, tile)))| {
                    if accepted[index] {
                        return None;
                    }
                    indices.push(index);
                    Some(GpuPreimageAttempt {
                        trapdoor,
                        public,
                        target: tile,
                        destination: &mut job.output,
                        column_start: job.destination_column,
                        global_column_start: job.payload.target_global_column_start + job.start,
                        attempt,
                        seed: job.payload.seed,
                    })
                })
                .collect();
            let flags = sampler
                .preimage_attempt_batch(&parameters, pending, &mut phases)
                .map_err(|error| format!("preimage attempt {attempt}: {error}"))?;
            for (index, flag) in indices.into_iter().zip(flags) {
                accepted[index] = flag;
            }
            if accepted.iter().all(|accepted| *accepted) {
                return Ok(jobs.into_iter().map(|job| job.output).collect());
            }
        }
        let index = accepted.iter().position(|accepted| !accepted).unwrap();
        Err(format!(
            "preimage columns {}..{} exhausted {} bounded attempts",
            jobs[index].start, jobs[index].end, plan.attempts
        ))
    }
}

impl GpuDcrtBackend {
    pub(in super::super) fn execute_prepared_preimage_batch(
        &mut self,
        requests: Vec<crate::backend::PreimageRequest<GpuFleetMatrix, GpuFleetTrapdoor>>,
    ) -> Result<Vec<GpuFleetSmallMatrix>, PolyBackendError> {
        use crate::gpu_measurement::{GpuAdmittedMeasurement, GpuColumnMeasurement};
        if self.prepared_invocations.len() < requests.len() ||
            self.prepared_invocations.iter().take(requests.len()).any(|invocation| {
                !matches!(invocation.operation, PreparedOperation::Preimage { .. })
            })
        {
            return Err(PolyBackendError::GpuSubmission(
                "preimage batch has no matching admitted invocations".into(),
            ));
        }
        let mut plans = Vec::with_capacity(requests.len());
        let mut invocations = Vec::with_capacity(requests.len());
        for (invocation, request) in self.prepared_invocations.drain(..requests.len()).zip(requests)
        {
            let payload = PreimagePayload {
                trapdoors: request.trapdoor.values.clone(),
                public: request.public.as_ref().clone(),
                target: request.target.column_range(0, request.target.col_size()),
                target_global_column_start: request.target.global_column_start(),
                seed: request.randomness_seed,
            };
            plans.push(invocation.plan);
            invocations.push((
                invocation.operation,
                invocation.template.intervals.clone(),
                payload,
                invocation.prepared,
            ));
        }
        let brokers = invocations
            .iter()
            .map(|(_, intervals, _, _)| {
                intervals
                    .iter()
                    .map(|range| self.claim_broker(&range.parameters))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if let Some(sink) = self.admitted_measurement_sink.as_mut() {
            let summaries = plans.iter().map(GpuColumnMemoryPlan::summary).collect::<Vec<_>>();
            let mut owners = HashMap::new();
            let layouts = invocations
                .iter()
                .map(|(operation, intervals, payload, _)| {
                    let next = owners.len();
                    let owner = *owners.entry(payload.public.id).or_insert(next);
                    let shapes = intervals
                        .iter()
                        .map(|range| {
                            (range.parameters.context_identity(), range.level, range.interval)
                        })
                        .collect::<Vec<_>>();
                    (operation.kind_name(), owner, payload.public.size(), shapes)
                })
                .collect::<Vec<_>>();
            let scenario = mxx_ir_core::encoding::hash_canonical(&(
                "prepared preimage sibling batch",
                self.active_operation,
                layouts,
                &summaries,
            ))
            .expect("concrete batch layout");
            sink(GpuAdmittedMeasurement::Invocation {
                operation: self.active_operation,
                plans: summaries,
                host_import: false,
                scenario,
            });
        }
        let mut measurement = self.admitted_measurement_sink.as_mut().map(|sink| {
            GpuColumnMeasurement::new(
                invocations
                    .iter()
                    .flat_map(|(_, intervals, _, _)| intervals.iter())
                    .map(|range| (range.interval.device, range.parameters.clone()))
                    .collect(),
                self.prepared_ledger.as_ref().unwrap().prepared_inventory().fold(
                    std::collections::BTreeMap::<usize, Vec<Arc<GpuPreparedStorage>>>::new(),
                    |mut stores, (device, storage)| {
                        stores.entry(device).or_default().push(storage);
                        stores
                    },
                ),
                sink,
                None,
                false,
            )
        });
        let invocations = Arc::new(invocations);
        let initialize = invocations.clone();
        let run = invocations.clone();
        let outputs = GpuColumnMemoryPlan::execute(
            plans,
            &mut self.enqueue,
            &mut self.devices,
            measurement.as_mut(),
            move |instance, device, _, fixed, physical| {
                if !fixed.is_empty() || !physical.is_empty() {
                    return Err(GpuAdmissionError::InvalidPlan(
                        "unexpected physical Preimage output".into(),
                    ));
                }
                let (operation, intervals, _, _) = &initialize[instance];
                let rows = operation
                    .output_rows::<GpuFleetMatrix>(None, &[])
                    .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
                intervals
                    .iter()
                    .filter(|range| range.interval.device == device)
                    .map(|range| {
                        let PreparedMatrixValue::Compact(value) = operation
                            .initialize_output(
                                &range.parameters,
                                range.level,
                                range.evaluation,
                                rows,
                                range.interval.end - range.interval.start,
                            )
                            .map_err(GpuAdmissionError::NativeReservation)?
                        else {
                            unreachable!("Preimage output")
                        };
                        Ok(Some(GpuColumnShard {
                            device_id: range.parameters.device_ids()[0],
                            global_column_start: range.interval.start,
                            value,
                        }))
                    })
                    .collect::<Result<Vec<_>, GpuAdmissionError>>()
            },
            |_, _, reservations| Ok(reservations.iter().map(|r| r.requests().to_vec()).collect()),
            move |jobs, _, outputs, leases| {
                let mut groups = std::collections::BTreeMap::new();
                for &(instance, job) in jobs {
                    if !leases[instance].as_ref().unwrap().scratch.is_empty() {
                        return Err(GpuAdmissionError::InvalidPlan(
                            "unexpected physical Preimage scratch".into(),
                        ));
                    }
                    let (operation, intervals, _, _) = &run[instance];
                    let PreparedOperation::Preimage {
                        ty,
                        bound,
                        sigma_bits,
                        gadget_base,
                        digit_count,
                        public_rows,
                        ..
                    } = operation
                    else {
                        unreachable!("Preimage invocation")
                    };
                    let range = &intervals[job.source_interval];
                    groups
                        .entry((
                            range.parameters.context_identity(),
                            range.level,
                            ty.rows,
                            bound.clone(),
                            *sigma_bits,
                            gadget_base.clone(),
                            *digit_count,
                            *public_rows,
                            job.end - job.start,
                        ))
                        .or_insert_with(Vec::new)
                        .push((instance, job));
                }
                for jobs in groups.values() {
                    let (first, first_job) = jobs[0];
                    let mut metadata = Vec::with_capacity(jobs.len());
                    let pending = jobs
                        .iter()
                        .map(|&(instance, job)| {
                            let (operation, intervals, payload, prepared) = &run[instance];
                            let range = &intervals[job.source_interval];
                            let output = outputs[instance].as_mut().unwrap()[range.destination]
                                .take()
                                .expect("retained Preimage output");
                            metadata.push((
                                instance,
                                range.destination,
                                output.device_id,
                                output.global_column_start,
                            ));
                            PreparedPreimageJob {
                                output: output.value,
                                destination_column: job.start - range.interval.start,
                                start: job.start,
                                end: job.end,
                                payload,
                                public: &operation
                                    .source(
                                        prepared,
                                        &payload.public,
                                        range.left_source.expect("admitted Preimage public matrix"),
                                        range.left_prepared,
                                    )
                                    .value,
                            }
                        })
                        .collect();
                    let values = run[first]
                        .0
                        .run_preimage_batch(pending, &brokers[first][first_job.source_interval])
                        .map_err(GpuAdmissionError::NativeReservation)?;
                    for ((instance, destination, device_id, global_column_start), value) in
                        metadata.into_iter().zip(values)
                    {
                        outputs[instance].as_mut().unwrap()[destination] =
                            Some(GpuColumnShard { device_id, global_column_start, value });
                    }
                }
                Ok(())
            },
        )
        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let mut shards = (0..invocations.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (_, instances) in outputs {
            for (instance, output) in instances.into_iter().enumerate() {
                shards[instance].extend(output.into_iter().flatten().map(|shard| shard.unwrap()));
            }
        }
        shards
            .into_par_iter()
            .zip(invocations.par_iter())
            .map(|(mut shards, (operation, _, _, _))| {
                shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
                Ok(GpuFleetSmallMatrix::new(
                    operation.output_rows::<GpuFleetMatrix>(None, &[])?,
                    operation.output_columns::<GpuFleetMatrix>(None),
                    shards,
                ))
            })
            .collect()
    }
}
