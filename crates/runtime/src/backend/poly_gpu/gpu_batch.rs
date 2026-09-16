//! Submit sibling column jobs through native batches, with retained prepared
//! destinations or allocating destinations selected by the execution mode.

use super::*;
use crate::gpu_measurement::{GpuAdmittedMeasurement, GpuColumnMeasurement};
use mxx_primitives::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrixColumnView;

impl GpuDcrtBackend {
    pub(in super::super) fn execute_matrix_batch(
        &mut self,
        inputs: &[Vec<&GpuFleetMatrix>],
        expected: Vec<PreparedOperation>,
    ) -> Result<Vec<GpuFleetMatrix>, PolyBackendError> {
        if !self.prepared_required {
            if !matches!(
                expected.first(),
                Some(PreparedOperation::Multiply { .. } | PreparedOperation::Accumulate { .. })
            ) {
                self.restart_runtime_pilot_after_matrix_inputs(
                    &inputs.iter().flat_map(|inputs| inputs.iter().copied()).collect::<Vec<_>>(),
                )?;
            }
            // Explicit calibration still uses its ordinary primitive runner.
            // The selected widths subsequently bound every sibling's real work.
            if let Some(input) = inputs.first() {
                if let PreparedOperation::Accumulate { products, bias, rows } = &expected[0] {
                    if self.runtime_pilot_is_pending() {
                        let request = MatrixMulAccumulateRequest {
                            products: products
                                .iter()
                                .enumerate()
                                .map(|(index, (coefficient, _))| {
                                    (
                                        coefficient.clone(),
                                        Arc::new(input[2 * index].clone()),
                                        Arc::new(input[2 * index + 1].clone()),
                                    )
                                })
                                .collect(),
                            bias: bias.then(|| Arc::new((**input.last().unwrap()).clone())),
                        };
                        let columns = input[usize::from(!products[0].1)].columns;
                        let runner = self.accumulate_column_runner(&request, *rows, columns)?;
                        self.calibrate_column_operation(columns, runner)?;
                    }
                } else if let PreparedOperation::Multiply { scales_left } = expected[0] {
                    if self.runtime_pilot_is_pending() {
                        let (scalable, fixed) =
                            if scales_left { (input[0], input[1]) } else { (input[1], input[0]) };
                        let runner = self.multiply_column_runner(scalable, fixed, scales_left)?;
                        self.calibrate_column_operation(scalable.columns, Arc::new(runner))?;
                    }
                } else if matches!(
                    expected[0],
                    PreparedOperation::Add | PreparedOperation::Subtract
                ) {
                    let subtract = expected[0] == PreparedOperation::Subtract;
                    let runner = Self::binary_column_runner(
                        input[0],
                        input[1],
                        move |backend, left, right| {
                            if subtract {
                                backend.sub(left, right)
                            } else {
                                backend.add(left, right)
                            }
                        },
                    );
                    self.calibrate_column_operation(input[0].columns, Arc::new(runner))?;
                } else {
                    let operation = match &expected[0] {
                        PreparedOperation::Negate => GpuUnaryColumnOperation::Negate,
                        PreparedOperation::Scale(scalar) => {
                            GpuUnaryColumnOperation::Scale(scalar.clone())
                        }
                        PreparedOperation::Automorphism(index) => {
                            GpuUnaryColumnOperation::Automorphism(*index)
                        }
                        _ => unreachable!("allocating unary batch"),
                    };
                    let runner = Self::unary_column_runner(input[0], operation);
                    self.calibrate_column_operation(input[0].columns, Arc::new(runner))?;
                }
            }
            let inputs = Arc::new(
                inputs
                    .iter()
                    .zip(&expected)
                    .map(|(input, operation)| {
                        let mut values =
                            input.iter().map(|value| (*value).clone()).collect::<Vec<_>>();
                        if matches!(operation, PreparedOperation::Multiply { scales_left: false }) {
                            values.swap(0, 1);
                        }
                        if let PreparedOperation::Accumulate { products, .. } = operation {
                            for (index, (_, scales_left)) in products.iter().enumerate() {
                                if !scales_left {
                                    values.swap(2 * index, 2 * index + 1);
                                }
                            }
                        }
                        values
                    })
                    .collect::<Vec<_>>(),
            );
            let operations = Arc::new(expected);
            // Fixed operands are shared across all sibling jobs on each
            // destination GPU. Keep their native owners alive through the last
            // column wave instead of copying them per instance or range.
            let replicas = Arc::new(
                self.devices
                    .par_iter_mut()
                    .map(|(_, backend)| {
                        let mut replicas = HashMap::new();
                        for (input, operation) in inputs.iter().zip(operations.iter()) {
                            let count = match operation {
                                PreparedOperation::Multiply { .. } => 1,
                                PreparedOperation::Accumulate { products, .. } => products.len(),
                                _ => 0,
                            };
                            for index in (0..count).map(|index| 2 * index + 1) {
                                if input[0].columns == 0 || replicas.contains_key(&input[index].id)
                                {
                                    continue;
                                }
                                let mut fixed = Self::matrix_operand_on_device(
                                    backend,
                                    &input[index],
                                    0,
                                    input[index].columns,
                                )?;
                                if !fixed.is_ntt() {
                                    let mut normalized = fixed.as_ref().clone();
                                    normalized.ntt_all_in_place();
                                    fixed = GpuMatrixOperand::Materialized(normalized);
                                }
                                replicas.insert(input[index].id, fixed);
                            }
                        }
                        Ok::<_, PolyBackendError>(replicas)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            if matches!(
                operations.first(),
                Some(PreparedOperation::Multiply { .. } | PreparedOperation::Accumulate { .. })
            ) {
                self.restart_runtime_pilot_after_fixed_inputs()
                    .map_err(PolyBackendError::GpuCalibration)?;
            }
            let schedules = inputs
                .iter()
                .zip(operations.iter())
                .map(|(input, operation)| {
                    let owner = &input[0];
                    let mut boundaries = input
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| match operation {
                            PreparedOperation::Multiply { .. } => *index == 0,
                            PreparedOperation::Accumulate { .. } => *index % 2 == 0,
                            _ => true,
                        })
                        .map(|(_, value)| value)
                        .flat_map(|value| {
                            value.shards.iter().map(|shard| shard.global_column_start)
                        })
                        .collect::<Vec<_>>();
                    boundaries.par_sort_unstable();
                    boundaries.dedup();
                    let schedule =
                        self.owned_column_schedule(owner.columns, &owner.shards, |value| {
                            value.size().1
                        })?;
                    let intervals =
                        schedule
                            .intervals()
                            .iter()
                            .flat_map(|interval| {
                                let cuts =
                                    std::iter::once(interval.start)
                                        .chain(boundaries.iter().copied().filter(|cut| {
                                            interval.start < *cut && *cut < interval.end
                                        }))
                                        .chain(std::iter::once(interval.end))
                                        .collect::<Vec<_>>();
                                cuts.windows(2)
                                    .map(|cut| GpuColumnInterval {
                                        device: interval.device,
                                        start: cut[0],
                                        end: cut[1],
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                    GpuColumnSchedule::new(owner.columns, schedule.widths().to_vec(), intervals)
                        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut waves = schedules.iter().map(GpuColumnSchedule::waves).collect::<Vec<_>>();
            let mut outputs = (0..inputs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
            loop {
                let mut groups = vec![Vec::new(); self.devices.len()];
                for (instance, waves) in waves.iter_mut().enumerate() {
                    for job in waves.next().into_iter().flatten() {
                        groups[job.device].push((instance, job));
                    }
                }
                if groups.iter().all(Vec::is_empty) {
                    break;
                }
                let sources = inputs.clone();
                let operations = operations.clone();
                let replicas = replicas.clone();
                let result =
                    self.enqueue
                        .map(&mut self.devices, move |device, (device_id, backend)| {
                            let mut classes = std::collections::BTreeMap::new();
                            for &(instance, job) in &groups[device] {
                                let source = &sources[instance][0];
                                let shard =
                                    &source.shards[source.shards.partition_point(|shard| {
                                        shard.global_column_start <= job.start
                                    }) - 1];
                                let value = &shard.value;
                                classes
                                    .entry((
                                        value.params().context_identity(),
                                        value.level(),
                                        value.is_ntt(),
                                        value.size().0,
                                        job.end - job.start,
                                        operations[instance].kind_name(),
                                        sources[instance].get(1).map(GpuFleetMatrix::size),
                                        if let PreparedOperation::Accumulate {
                                            products,
                                            rows,
                                            ..
                                        } = &operations[instance]
                                        {
                                            (products.len(), *rows)
                                        } else {
                                            (1, value.size().0)
                                        },
                                        matches!(
                                            operations[instance],
                                            PreparedOperation::Multiply { scales_left: true }
                                        ),
                                    ))
                                    .or_insert_with(Vec::new)
                                    .push((instance, job));
                            }
                            // This cache lives for one submitted device wave. It
                            // shares actual source owners across operation classes
                            // without retaining ranges from earlier waves.
                            let mut ranges = HashMap::new();
                            let mut prepare =
                                |source: &GpuFleetMatrix,
                                 start: usize,
                                 end: usize,
                                 evaluation: bool| {
                                    let key = (source.id, start, end, evaluation);
                                    if let Some(range) = ranges.get(&key) {
                                        return Ok::<_, PolyBackendError>(Clone::clone(range));
                                    }
                                    let index = source.shards.partition_point(|shard| {
                                        shard.global_column_start <= start
                                    }) - 1;
                                    let shard = &source.shards[index];
                                    let range = if backend
                                        .matrix_is_on_active_placement(&shard.value) &&
                                        shard.value.is_ntt() == evaluation
                                    {
                                        (
                                            Arc::new(GpuMatrixOperand::Resident {
                                                shards: source.shards.clone(),
                                                index,
                                            }),
                                            start - shard.global_column_start..
                                                end - shard.global_column_start,
                                        )
                                    } else {
                                        let mut value = Self::matrix_piece_on_device(
                                            backend, source, start, end,
                                        )?;
                                        if value.is_ntt() != evaluation {
                                            if evaluation {
                                                value.ntt_all_in_place();
                                            } else {
                                                value.intt_all_in_place();
                                            }
                                        }
                                        (
                                            Arc::new(GpuMatrixOperand::Materialized(value)),
                                            0..end - start,
                                        )
                                    };
                                    ranges.insert(key, range.clone());
                                    Ok(range)
                                };
                            let mut outputs = Vec::new();
                            for (class, jobs) in &classes {
                                let evaluation = class.2 ||
                                    matches!(
                                        operations[jobs[0].0],
                                        PreparedOperation::Multiply { .. } |
                                            PreparedOperation::Accumulate { .. }
                                    );
                                let pieces = jobs
                                    .iter()
                                    .map(|(instance, job)| {
                                        prepare(
                                            &sources[*instance][0],
                                            job.start,
                                            job.end,
                                            evaluation,
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let views = pieces
                                    .iter()
                                    .map(|(piece, columns)| piece.column_view(columns.clone()))
                                    .collect::<Result<Vec<_>, _>>()
                                    .map_err(PolyBackendError::GpuSubmission)?;
                                let mut right = Vec::new();
                                if matches!(
                                    operations[jobs[0].0],
                                    PreparedOperation::Add | PreparedOperation::Subtract
                                ) {
                                    for (instance, job) in jobs {
                                        right.push(prepare(
                                            &sources[*instance][1],
                                            job.start,
                                            job.end,
                                            evaluation,
                                        )?);
                                    }
                                }
                                let mut extra = Vec::new();
                                if let PreparedOperation::Accumulate { .. } = &operations[jobs[0].0]
                                {
                                    for (instance, job) in jobs {
                                        extra.push(
                                            sources[*instance]
                                                .iter()
                                                .skip(2)
                                                .step_by(2)
                                                .map(|source| {
                                                    prepare(source, job.start, job.end, true)
                                                })
                                                .collect::<Result<Vec<_>, _>>()?,
                                        );
                                    }
                                }
                                let values = match &operations[jobs[0].0] {
                                    PreparedOperation::Accumulate { .. } => {
                                        let sums = jobs
                                            .iter()
                                            .zip(&views)
                                            .zip(&extra)
                                            .map(|(((instance, _), first), extra)| {
                                                let PreparedOperation::Accumulate {
                                                    products,
                                                    bias,
                                                    ..
                                                } = &operations[*instance]
                                                else {
                                                    unreachable!()
                                                };
                                                let terms = products
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(index, (coefficient, scales_left))| {
                                                        let scalable = if index == 0 {
                                                            *first
                                                        } else {
                                                            extra[index - 1].0.column_view(
                                                                extra[index - 1].1.clone(),
                                                            )?
                                                        };
                                                        let fixed = &replicas[device]
                                                            [&sources[*instance][2 * index + 1].id];
                                                        let fixed =
                                                            fixed.column_view(0..fixed.size().1)?;
                                                        Ok(if *scales_left {
                                                            (coefficient.clone(), scalable, fixed)
                                                        } else {
                                                            (coefficient.clone(), fixed, scalable)
                                                        })
                                                    })
                                                    .collect::<Result<Vec<_>, String>>()?;
                                                let bias = if *bias {
                                                    let (value, columns) = extra.last().unwrap();
                                                    Some(value.column_view(columns.clone())?)
                                                } else {
                                                    None
                                                };
                                                Ok((terms, bias, None))
                                            })
                                            .collect::<Result<Vec<_>, String>>()
                                            .map_err(PolyBackendError::GpuSubmission)?;
                                        GpuDCRTPolyMatrixColumnView::multiply_accumulate_batch(sums)
                                    }
                                    PreparedOperation::Multiply { scales_left } => {
                                        let pairs = views
                                            .into_iter()
                                            .zip(jobs)
                                            .map(|(view, (instance, _))| {
                                                let fixed =
                                                    &replicas[device][&sources[*instance][1].id];
                                                let fixed = fixed.column_view(0..fixed.size().1)?;
                                                Ok(if *scales_left {
                                                    (view, fixed, None)
                                                } else {
                                                    (fixed, view, None)
                                                })
                                            })
                                            .collect::<Result<Vec<_>, String>>()
                                            .map_err(PolyBackendError::GpuSubmission)?;
                                        GpuDCRTPolyMatrixColumnView::multiply_batch(pairs)
                                    }
                                    PreparedOperation::Add | PreparedOperation::Subtract => {
                                        let pairs = views
                                            .into_iter()
                                            .zip(&right)
                                            .map(|(left, (right, columns))| {
                                                Ok((
                                                    left,
                                                    right.column_view(columns.clone())?,
                                                    None,
                                                ))
                                            })
                                            .collect::<Result<Vec<_>, String>>()
                                            .map_err(PolyBackendError::GpuSubmission)?;
                                        GpuDCRTPolyMatrixColumnView::binary_batch(
                                            pairs,
                                            operations[jobs[0].0] == PreparedOperation::Subtract,
                                        )
                                    }
                                    PreparedOperation::Negate => {
                                        GpuDCRTPolyMatrixColumnView::negate_batch(
                                            views.into_iter().map(|view| (view, None)),
                                        )
                                    }
                                    PreparedOperation::Scale(_) => {
                                        GpuDCRTPolyMatrixColumnView::scale_integer_batch(
                                            views.into_iter().zip(jobs).map(
                                                |(view, (instance, _))| {
                                                    let PreparedOperation::Scale(scalar) =
                                                        &operations[*instance]
                                                    else {
                                                        unreachable!()
                                                    };
                                                    (view, scalar, None)
                                                },
                                            ),
                                        )
                                    }
                                    PreparedOperation::Automorphism(_) => {
                                        GpuDCRTPolyMatrixColumnView::ring_automorphism_batch(
                                            views.into_iter().zip(jobs).map(
                                                |(view, (instance, _))| {
                                                    let PreparedOperation::Automorphism(index) =
                                                        &operations[*instance]
                                                    else {
                                                        unreachable!()
                                                    };
                                                    (view, *index, None)
                                                },
                                            ),
                                        )
                                    }
                                    _ => unreachable!("allocating unary batch"),
                                }
                                .map_err(PolyBackendError::GpuSubmission)?;
                                outputs.extend(jobs.iter().zip(values).map(
                                    |((instance, job), value)| {
                                        (
                                            *instance,
                                            GpuColumnShard {
                                                device_id: *device_id,
                                                global_column_start: job.start,
                                                value,
                                            },
                                        )
                                    },
                                ));
                            }
                            Ok::<_, PolyBackendError>(outputs)
                        })
                        .map_err(PolyBackendError::from)?;
                for (instance, shard) in result.into_iter().flatten() {
                    outputs[instance].push(shard);
                }
            }
            return Ok(outputs
                .into_par_iter()
                .zip(inputs.par_iter())
                .zip(operations.par_iter())
                .map(|((mut shards, input), operation)| {
                    shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
                    let rows = if matches!(operation, PreparedOperation::Multiply { .. }) &&
                        input[1].size() != (1, 1)
                    {
                        input[1].rows
                    } else {
                        input[0].rows
                    };
                    let rows = if let PreparedOperation::Accumulate { rows, .. } = operation {
                        *rows
                    } else {
                        rows
                    };
                    GpuFleetMatrix::new(rows, input[0].columns, shards)
                })
                .collect());
        }
        if self.prepared_invocations.len() < inputs.len() ||
            self.prepared_invocations.iter().zip(inputs).zip(&expected).any(
                |((invocation, input), operation)| {
                    &invocation.operation != operation ||
                        !invocation.matches_caller_operands(input, None)
                },
            )
        {
            return Err(PolyBackendError::GpuSubmission(
                "matrix batch has no matching admitted invocations".into(),
            ));
        }
        let mut plans = Vec::with_capacity(inputs.len());
        let mut invocations = Vec::with_capacity(inputs.len());
        for invocation in self.prepared_invocations.drain(..inputs.len()) {
            let CompiledMatrixInvocation {
                operation, left, right, template, prepared, plan, ..
            } = invocation;
            plans.push(plan);
            invocations.push((
                operation,
                left.unwrap(),
                right,
                template.intervals.clone(),
                prepared,
            ));
        }
        if let Some(sink) = self.admitted_measurement_sink.as_mut() {
            // Preserve aliases across all siblings without including allocation
            // IDs: a broadcast owner differs from independent equal-shaped data.
            let mut owners = HashMap::new();
            let layouts = invocations
                .iter()
                .map(|(operation, left, right, intervals, _)| {
                    let inputs = std::iter::once(left)
                        .chain(right)
                        .map(|matrix| {
                            let next = owners.len();
                            let owner = *owners.entry(matrix.id).or_insert(next);
                            let shards = matrix
                                .shards
                                .iter()
                                .map(|shard| {
                                    (
                                        shard.device_id,
                                        shard.global_column_start,
                                        shard.value.size(),
                                        shard.value.params().context_identity(),
                                        shard.value.level(),
                                        shard.value.is_ntt(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            (owner, matrix.size(), shards)
                        })
                        .collect::<Vec<_>>();
                    let mappings = intervals
                        .iter()
                        .map(|range| {
                            (
                                range.interval,
                                range.destination,
                                range.parameters.context_identity(),
                                range.level,
                                range.evaluation,
                                range.left_source,
                                &range.right_source,
                            )
                        })
                        .collect::<Vec<_>>();
                    (operation.kind_name(), inputs, mappings)
                })
                .collect::<Vec<_>>();
            let summaries = plans.iter().map(GpuColumnMemoryPlan::summary).collect::<Vec<_>>();
            let scenario = mxx_ir_core::encoding::hash_canonical(&(
                "compiled prepared sibling batch",
                self.active_operation,
                layouts,
                &summaries,
            ))
            .expect("concrete batch layout is serializable");
            sink(GpuAdmittedMeasurement::Invocation {
                operation: self.active_operation,
                plans: summaries,
                host_import: false,
                scenario,
            });
        }
        let brokers = invocations
            .iter()
            .map(|(_, _, _, intervals, _)| {
                intervals
                    .iter()
                    .map(|range| self.claim_broker(&range.parameters))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut measurement = self.admitted_measurement_sink.as_mut().map(|sink| {
            GpuColumnMeasurement::new(
                invocations
                    .iter()
                    .flat_map(|(_, _, _, intervals, _)| intervals.iter())
                    .map(|range| (range.interval.device, range.parameters.clone()))
                    .collect(),
                self.prepared_ledger.as_ref().unwrap().prepared_inventory().fold(
                    std::collections::BTreeMap::<usize, Vec<Arc<GpuPreparedStorage>>>::new(),
                    |mut stores, (device, storage)| {
                        stores.entry(device).or_default().push(storage.clone());
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
                        "unexpected physical batch output demand".into(),
                    ));
                }
                let (operation, input, right, intervals, _) = &initialize[instance];
                let rows = operation
                    .output_rows(Some(input), right)
                    .map_err(|e| GpuAdmissionError::InvalidPlan(e.to_string()))?;
                intervals
                    .iter()
                    .filter(|range| range.interval.device == device)
                    .map(|range| {
                        Ok(Some(GpuColumnShard {
                            device_id: range.parameters.device_ids()[0],
                            global_column_start: range.interval.start,
                            value: {
                                let PreparedMatrixValue::Matrix(value) = operation
                                    .initialize_output(
                                        &range.parameters,
                                        range.level,
                                        range.evaluation,
                                        rows,
                                        range.interval.end - range.interval.start,
                                    )
                                    .map_err(GpuAdmissionError::NativeReservation)?
                                else {
                                    unreachable!("ordinary batch output")
                                };
                                value
                            },
                        }))
                    })
                    .collect::<Result<Vec<_>, GpuAdmissionError>>()
            },
            |_, _, reservations| Ok(reservations.iter().map(|r| r.requests().to_vec()).collect()),
            move |jobs, _, outputs, leases| {
                // Different contexts, formats or tails are separate native
                // batches. Equal shapes still share one launch even when their
                // owners have different pitches or are Broadcast aliases.
                let mut groups = std::collections::BTreeMap::new();
                for &(instance, job) in jobs {
                    if !leases[instance].as_ref().unwrap().scratch.is_empty() {
                        return Err(GpuAdmissionError::InvalidPlan(
                            "unexpected physical batch scratch".into(),
                        ));
                    }
                    let (operation, input, right, intervals, _) = &run[instance];
                    let rows = operation
                        .output_rows(Some(input), right)
                        .map_err(|e| GpuAdmissionError::InvalidPlan(e.to_string()))?;
                    let range = &intervals[job.source_interval];
                    groups
                        .entry((
                            operation.kind_name(),
                            if let PreparedOperation::Accumulate { products, .. } = operation {
                                products.len()
                            } else {
                                1
                            },
                            range.parameters.context_identity(),
                            range.level,
                            range.evaluation,
                            rows,
                            input.rows,
                            right.first().map(GpuFleetMatrix::size),
                            job.end - job.start,
                        ))
                        .or_insert_with(Vec::new)
                        .push((instance, job));
                }
                for jobs in groups.values() {
                    let (first, first_job) = jobs[0];
                    let (expected, _, right, intervals, _) = &run[first];
                    let range = &intervals[first_job.source_interval];
                    let first_output =
                        outputs[first].as_ref().unwrap()[range.destination].as_ref().unwrap();
                    let claims = expected
                        .batch_workspaces(
                            &range.parameters,
                            range.level,
                            first_output.value.size(),
                            jobs.len(),
                            right.first().is_some_and(|matrix| matrix.size() == (1, 1)),
                        )
                        .map_err(|error| GpuAdmissionError::NativeReservation(error.to_string()))?
                        .into_iter()
                        .map(GpuTracedClaim::workspace)
                        .collect::<Vec<_>>();
                    let destinations = jobs
                        .iter()
                        .map(|&(instance, job)| {
                            let (operation, input, _, intervals, prepared) = &run[instance];
                            let range = &intervals[job.source_interval];
                            let source = operation.source(
                                prepared,
                                input,
                                range.left_source.unwrap(),
                                range.left_prepared,
                            );
                            let view = source
                                .value
                                .column_view(
                                    job.start - source.global_column_start..
                                        job.end - source.global_column_start,
                                )
                                .map_err(GpuAdmissionError::NativeReservation)?;
                            let output = outputs[instance].as_mut().unwrap()[range.destination]
                                .take()
                                .unwrap();
                            Ok((
                                view,
                                Some((
                                    output.value,
                                    0..operation
                                        .output_rows(Some(input), &run[instance].2)
                                        .expect("validated output shape"),
                                    job.start - range.interval.start..
                                        job.end - range.interval.start,
                                )),
                            ))
                        })
                        .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
                    let results = brokers[first][first_job.source_interval]
                        .hold(&claims, false, || {
                            if *expected == PreparedOperation::Negate {
                                GpuDCRTPolyMatrixColumnView::negate_batch(destinations)
                            } else if matches!(expected, PreparedOperation::Scale(_)) {
                                GpuDCRTPolyMatrixColumnView::scale_integer_batch(
                                    jobs.iter().zip(destinations).map(
                                        |(&(instance, _), (input, output))| {
                                            let PreparedOperation::Scale(scalar) = &run[instance].0
                                            else {
                                                unreachable!("scale batch")
                                            };
                                            (input, scalar, output)
                                        },
                                    ),
                                )
                            } else if matches!(expected, PreparedOperation::Automorphism(_)) {
                                GpuDCRTPolyMatrixColumnView::ring_automorphism_batch(
                                    jobs.iter().zip(destinations).map(
                                        |(&(instance, _), (input, output))| {
                                            let PreparedOperation::Automorphism(index) =
                                                &run[instance].0
                                            else {
                                                unreachable!("automorphism batch")
                                            };
                                            (input, *index, output)
                                        },
                                    ),
                                )
                            } else if matches!(expected, PreparedOperation::Accumulate { .. }) {
                                let sums = jobs
                                    .iter()
                                    .zip(destinations)
                                    .map(|(&(instance, job), (primary, destination))| {
                                        let (operation, _, right, intervals, prepared) =
                                            &run[instance];
                                        let PreparedOperation::Accumulate {
                                            products, bias, ..
                                        } = operation
                                        else {
                                            unreachable!("accumulate batch")
                                        };
                                        let range = &intervals[job.source_interval];
                                        let source =
                                            |index: usize, fixed: bool| -> Result<_, String> {
                                                let value = operation.source(
                                                    prepared,
                                                    &right[index],
                                                    range.right_source[index],
                                                    range.right_prepared[index],
                                                );
                                                let columns = if fixed {
                                                    0..right[index].columns
                                                } else {
                                                    job.start..job.end
                                                };
                                                value.value.column_view(
                                                    columns.start - value.global_column_start..
                                                        columns.end - value.global_column_start,
                                                )
                                            };
                                        let products = products
                                            .iter()
                                            .enumerate()
                                            .map(|(index, (coefficient, scales_left))| {
                                                let scalable = if index == 0 {
                                                    primary
                                                } else {
                                                    source(2 * index - 1, false)?
                                                };
                                                let fixed = source(2 * index, true)?;
                                                let (left, right) = if *scales_left {
                                                    (scalable, fixed)
                                                } else {
                                                    (fixed, scalable)
                                                };
                                                Ok((coefficient.clone(), left, right))
                                            })
                                            .collect::<Result<Vec<_>, String>>()?;
                                        let bias = if *bias {
                                            Some(source(right.len() - 1, false)?)
                                        } else {
                                            None
                                        };
                                        Ok((products, bias, destination))
                                    })
                                    .collect::<Result<Vec<_>, String>>()?;
                                GpuDCRTPolyMatrixColumnView::multiply_accumulate_batch(sums)
                            } else {
                                let pairs = jobs
                                    .iter()
                                    .zip(destinations)
                                    .map(|(&(instance, job), (left, output))| {
                                        let (operation, _, right, intervals, prepared) =
                                            &run[instance];
                                        let range = &intervals[job.source_interval];
                                        let source = operation.source(
                                            prepared,
                                            &right[0],
                                            range.right_source[0],
                                            range.right_prepared[0],
                                        );
                                        let columns = operation
                                            .other_columns(0, &right[0], job.start, job.end);
                                        let right = source.value.column_view(
                                            columns.start - source.global_column_start..
                                                columns.end - source.global_column_start,
                                        )?;
                                        Ok(
                                            if matches!(
                                                operation,
                                                PreparedOperation::Multiply { scales_left: false }
                                            ) {
                                                (right, left, output)
                                            } else {
                                                (left, right, output)
                                            },
                                        )
                                    })
                                    .collect::<Result<Vec<_>, String>>()?;
                                if matches!(expected, PreparedOperation::Multiply { .. }) {
                                    GpuDCRTPolyMatrixColumnView::multiply_batch(pairs)
                                } else {
                                    GpuDCRTPolyMatrixColumnView::binary_batch(
                                        pairs,
                                        *expected == PreparedOperation::Subtract,
                                    )
                                }
                            }
                        })
                        .map_err(GpuAdmissionError::NativeReservation)?;
                    for (&(instance, job), value) in jobs.iter().zip(results) {
                        let range = &run[instance].3[job.source_interval];
                        outputs[instance].as_mut().unwrap()[range.destination] =
                            Some(GpuColumnShard {
                                device_id: range.parameters.device_ids()[0],
                                global_column_start: range.interval.start,
                                value,
                            });
                    }
                }
                Ok(())
            },
        )
        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
        let mut shards = (0..inputs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (_, outputs) in outputs {
            for (instance, outputs) in outputs.into_iter().enumerate() {
                shards[instance].extend(outputs.into_iter().flatten().flatten());
            }
        }
        shards
            .into_par_iter()
            .zip(invocations.par_iter())
            .map(|(mut shards, (operation, input, right, _, _))| {
                shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
                Ok(GpuFleetMatrix::new(
                    operation.output_rows(Some(input), right)?,
                    operation.output_columns(Some(input)),
                    shards,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::{
        poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
        sampler::{PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_allocating_batches_preserve_parameters_and_column_tails() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(3);
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
        let mut backend = super::super::super::super::gpu_backend_on([params.clone()], [device]);
        backend.set_column_widths_for_operation(
            [211; 32],
            GpuColumnWidths { gpu0: Some(columns - 1), nonzero: None },
        );
        backend.select_gpu_operation([211; 32]).unwrap();
        let inputs = (0..3)
            .map(|index| {
                let value = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    2 + index % 2,
                    columns,
                    DistType::FinRingDist,
                );
                Arc::new(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                    &params, &value,
                )))
            })
            .collect::<Vec<_>>();
        backend.set_column_widths_for_operation(
            [212; 32],
            GpuColumnWidths { gpu0: Some(columns / 2), nonzero: None },
        );
        backend.select_gpu_operation([212; 32]).unwrap();
        let right = inputs
            .iter()
            .map(|input| backend.negate(input).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        backend.select_gpu_operation([211; 32]).unwrap();
        let mut fixed = (0..2)
            .map(|index| {
                let value = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    4,
                    2 + index,
                    DistType::FinRingDist,
                );
                Arc::new(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                    &params, &value,
                )))
            })
            .collect::<Vec<_>>();
        fixed.push(fixed[0].clone());
        let scalar = Arc::new(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
            &params,
            &DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist),
        )));
        let scalars = [-(BigInt::from(1) << 90usize) - 7, BigInt::from(0), BigInt::from(3)];
        let indices = [1, 3, 2 * n as usize - 1];
        for operation in 0..8 {
            let expected = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| match operation {
                    0 => backend.negate(input),
                    1 => backend.scale_integer(input, &scalars[index]),
                    2 => backend.ring_automorphism(input, indices[index]),
                    3 => backend.add(input, &right[index]),
                    4 => backend.sub(input, &right[index]),
                    5 => backend.multiply(&fixed[index], input),
                    6 => backend.multiply(&scalar, input),
                    _ => backend.multiply(input, &scalar),
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let actual = match operation {
                0 => backend.negate_batch(inputs.clone()),
                1 => backend.scale_integer_batch(
                    inputs.iter().cloned().zip(scalars.iter().cloned()).collect(),
                ),
                2 => backend.ring_automorphism_batch(inputs.iter().cloned().zip(indices).collect()),
                3 => backend.add_batch(inputs.iter().cloned().zip(right.iter().cloned()).collect()),
                4 => backend.sub_batch(inputs.iter().cloned().zip(right.iter().cloned()).collect()),
                5 => backend
                    .multiply_batch(fixed.iter().cloned().zip(inputs.iter().cloned()).collect()),
                6 => backend.multiply_batch(
                    inputs.iter().map(|input| (scalar.clone(), input.clone())).collect(),
                ),
                _ => backend.multiply_batch(
                    inputs.iter().map(|input| (input.clone(), scalar.clone())).collect(),
                ),
            }
            .unwrap();
            assert_eq!(actual.len(), inputs.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.size(), expected.size());
                if operation < 3 || operation >= 5 {
                    assert_eq!(actual.shards.len(), 2, "full-width job and one-column tail");
                } else {
                    assert_eq!(
                        actual.shards.len(),
                        right[0].shards.len(),
                        "intersect the RHS boundaries"
                    );
                }
                assert_eq!(actual.shards.len(), expected.shards.len());
                for (actual, expected) in actual.shards.iter().zip(expected.shards.iter()) {
                    assert_eq!(actual.global_column_start, expected.global_column_start);
                    assert_eq!(actual.device_id, expected.device_id);
                    assert_eq!(actual.value.to_cpu_matrix(), expected.value.to_cpu_matrix());
                }
            }
        }
        let bias = Arc::new(GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
            &params,
            &DCRTPolyUniformSampler::new().sample_uniform(&cpu, 4, columns, DistType::FinRingDist),
        )));
        let sum_inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                if index == 2 {
                    let mut value = (*input.shards[0].value).clone();
                    value.intt_all_in_place();
                    Arc::new(GpuFleetMatrix::from_matrix(value))
                } else {
                    input.clone()
                }
            })
            .collect::<Vec<_>>();
        let sums = sum_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| MatrixMulAccumulateRequest {
                products: [1 + index as i32, -3, 5, 2, -1]
                    .into_iter()
                    .map(|coefficient| {
                        (BigInt::from(coefficient), fixed[index].clone(), input.clone())
                    })
                    .collect(),
                bias: (index % 2 == 0).then(|| bias.clone()),
            })
            .collect::<Vec<_>>();
        // The scalar reference requires evaluation inputs. The batch receives
        // the mathematically identical coefficient-form owner above, testing
        // its required production normalization against that trusted reference.
        let expected = sums
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, mut request)| {
                for (_, _, right) in &mut request.products {
                    *right = inputs[index].clone();
                }
                backend.matrix_mul_accumulate(request)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let actual = backend.matrix_mul_accumulate_batch(sums).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.size(), (4, columns));
            assert_eq!(actual.shards.len(), 2);
            assert_eq!(actual.shards.len(), expected.shards.len());
            for (actual, expected) in actual.shards.iter().zip(expected.shards.iter()) {
                assert_eq!(actual.global_column_start, expected.global_column_start);
                assert_eq!(actual.value.to_cpu_matrix(), expected.value.to_cpu_matrix());
            }
        }
        assert!(backend.negate_batch(Vec::new()).unwrap().is_empty());
    }
}
