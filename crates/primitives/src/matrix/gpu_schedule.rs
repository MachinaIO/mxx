//! Fixed stream dependencies shared by every destination-aware primitive.

use super::*;
use crate::poly::dcrt::gpu::{
    GpuDCRTPolyParams, GpuPreparedPlanRef as NativePlan,
    GpuPreparedScheduleOpaque as ScheduleOpaque, gpu_prepared_schedule_begin,
    gpu_prepared_schedule_bind_dependencies, gpu_prepared_schedule_create,
    gpu_prepared_schedule_destroy, gpu_prepared_schedule_end, gpu_prepared_schedule_provision,
    gpu_prepared_schedule_stream_context, gpu_prepared_schedule_stream_count,
};

/// Borrowed only during preparation. The schedule snapshots native launch
/// streams and retains their matrix/context owners, not borrowed plan pointers.
pub enum GpuPreparedSchedulePlan<'a> {
    Arithmetic(&'a GpuPreparedArithmeticCommand),
    Accumulate(&'a GpuPreparedAccumulateCommand),
    Transform(&'a GpuPreparedTransform, &'a Arc<GpuDCRTPolyMatrix>),
    Modulus(&'a GpuPreparedModulusCommand, &'a Arc<GpuDCRTPolyMatrix>),
    InputCopy(&'a GpuPreparedInputCopy),
    Transpose(&'a GpuPreparedTranspose),
    CenteredRebase(&'a GpuPreparedCenteredRebase),
    GadgetDecompose(&'a GpuPreparedGadgetDecompose),
    Sampling(&'a GpuPreparedSampling),
    HashSample(&'a GpuPreparedHashSample),
    SmallRhs(&'a GpuPreparedSmallRhs),
    CrtRecompose(&'a GpuPreparedCrtRecompose),
    Readback(&'a GpuPreparedConstCoeffReadback),
    Reconstruction(&'a GpuPreparedRnsReconstruction),
    Upload(&'a GpuPreparedRnsUpload),
    CompactDecompose(&'a GpuPreparedCompactDecompose),
    Threshold(&'a GpuPreparedThreshold),
    ScalarPack(&'a GpuPreparedScalarPack),
    ScalarOp(&'a GpuPreparedScalarOp),
    ScalarMatrixSelect(&'a GpuPreparedScalarMatrixSelect),
    ScalarUpload(&'a GpuPreparedScalarBuffer),
    PreimageCutoff(&'a super::super::GpuPreparedPreimageCutoff),
    PreimagePhases(&'a super::super::GpuPreparedPreimagePhases),
}

pub struct GpuPreparedSchedule {
    raw: NonNull<ScheduleOpaque>,
    // Drop native events before the execution/matrix owners that supply streams.
    owners: Vec<Arc<GpuDCRTPolyMatrix>>,
    dependencies: Vec<Arc<GpuPreparedSchedule>>,
    provisioned: bool,
    bound: bool,
}

unsafe impl Send for GpuPreparedSchedule {}
unsafe impl Sync for GpuPreparedSchedule {}

impl GpuPreparedSchedule {
    /// Capture the real stream(s), including nested transforms and all serde
    /// limbs. This does not allocate CUDA resources or submit GPU work.
    pub fn new(
        plans: &[GpuPreparedSchedulePlan<'_>],
        dependencies: &[Arc<Self>],
    ) -> Result<Self, String> {
        let mut native = Vec::new();
        let mut owners = Vec::new();
        for plan in plans {
            match plan {
                GpuPreparedSchedulePlan::Arithmetic(p) => {
                    native.push(NativePlan { kind: 0, plan: p.plan.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::Accumulate(p) => {
                    for command in &p.commands {
                        native.push(NativePlan { kind: 0, plan: command.plan.raw.as_ptr().cast() });
                        owners.push(Arc::clone(&command.output));
                    }
                }
                GpuPreparedSchedulePlan::Transform(p, owner) => {
                    if p.matrix != owner.raw {
                        return Err("schedule transform owner mismatch".into());
                    }
                    native.push(NativePlan { kind: 1, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(owner));
                }
                GpuPreparedSchedulePlan::Modulus(p, owner) => {
                    if p.target != owner.raw {
                        return Err("schedule modulus owner mismatch".into());
                    }
                    native.push(NativePlan { kind: 2, plan: p.plan.raw.as_ptr().cast() });
                    owners.push(Arc::clone(owner));
                }
                GpuPreparedSchedulePlan::InputCopy(p) => {
                    native.push(NativePlan { kind: 3, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::Transpose(p) => {
                    native.push(NativePlan { kind: 4, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::CenteredRebase(p) => {
                    native.push(NativePlan { kind: 5, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::GadgetDecompose(p) => {
                    native.push(NativePlan { kind: 6, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::Sampling(p) => {
                    native.push(NativePlan { kind: 7, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::HashSample(p) => {
                    native.push(NativePlan { kind: 7, plan: p.sampler.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.sampler.output));
                }
                GpuPreparedSchedulePlan::SmallRhs(p) => {
                    native.push(NativePlan { kind: 8, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::CrtRecompose(p) => {
                    native.push(NativePlan { kind: 9, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::Readback(p) => {
                    native.push(NativePlan { kind: 10, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.source));
                }
                GpuPreparedSchedulePlan::Reconstruction(p) => {
                    native.push(NativePlan { kind: 10, plan: p.readback.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.readback.source));
                }
                GpuPreparedSchedulePlan::Upload(p) => {
                    native.push(NativePlan { kind: 11, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.target));
                }
                GpuPreparedSchedulePlan::CompactDecompose(p) => {
                    let (kind, plan) = p.schedule_record();
                    native.push(NativePlan { kind, plan });
                    owners.push(Arc::clone(p.source_owner()));
                }
                GpuPreparedSchedulePlan::Threshold(p) => {
                    native.push(NativePlan { kind: 13, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.source));
                    owners.push(Arc::clone(p.output().anchor()));
                }
                GpuPreparedSchedulePlan::ScalarPack(p) => {
                    native.push(NativePlan { kind: 14, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                }
                GpuPreparedSchedulePlan::ScalarOp(p) => {
                    native.push(NativePlan { kind: 15, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(p.output.anchor()));
                }
                GpuPreparedSchedulePlan::ScalarMatrixSelect(p) => {
                    native.push(NativePlan { kind: 17, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(&p.output));
                    owners.extend(p.sources().iter().cloned());
                }
                GpuPreparedSchedulePlan::ScalarUpload(p) => {
                    native.push(NativePlan { kind: 16, plan: p.raw.as_ptr().cast() });
                    owners.push(Arc::clone(p.anchor()));
                }
                GpuPreparedSchedulePlan::PreimageCutoff(p) => {
                    let (kind, plan) = p.schedule_record();
                    native.push(NativePlan { kind, plan });
                    owners.extend(p.sources().iter().cloned());
                }
                GpuPreparedSchedulePlan::PreimagePhases(p) => {
                    let (kind, plan) = p.schedule_record();
                    native.push(NativePlan { kind, plan });
                    owners.extend(p.owners().iter().cloned());
                }
            }
        }
        let dependencies_raw =
            dependencies.iter().map(|p| p.raw.as_ptr().cast_const()).collect::<Vec<_>>();
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_prepared_schedule_create(
                native.as_ptr(),
                native.len(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len(),
                &mut raw,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or("native schedule missing")?,
            owners,
            dependencies: dependencies.to_vec(),
            provisioned: false,
            bound: !dependencies.is_empty(),
        })
    }

    pub fn stream_count(&self) -> usize {
        unsafe { gpu_prepared_schedule_stream_count(self.raw.as_ptr()) }
    }

    /// Resolve the actual launch-context owner once, before event provisioning.
    pub fn stream_parameters(&self) -> Result<Vec<GpuDCRTPolyParams>, String> {
        (0..self.stream_count())
            .map(|index| {
                let context =
                    unsafe { gpu_prepared_schedule_stream_context(self.raw.as_ptr(), index) }
                        as usize;
                self.owners
                    .iter()
                    .find(|owner| owner.params().context_identity() == context)
                    .map(|owner| owner.params().clone())
                    .ok_or_else(|| "prepared schedule stream has no retained context owner".into())
            })
            .collect()
    }

    /// Consume exactly `stream_count()` CompletionEvent claims under the
    /// caller's detached dispatch permit, before publishing this executable.
    pub fn provision(&mut self) -> Result<(), String> {
        if self.provisioned {
            return Err("prepared schedule is already provisioned".into());
        }
        let status = unsafe { gpu_prepared_schedule_provision(self.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        self.provisioned = true;
        Ok(())
    }

    /// One-shot final setup step. Root commands explicitly bind an empty slice.
    pub fn bind_dependencies(&mut self, dependencies: &[Arc<Self>]) -> Result<(), String> {
        let pointers = dependencies.iter().map(|p| p.raw.as_ptr().cast_const()).collect::<Vec<_>>();
        let status = unsafe {
            gpu_prepared_schedule_bind_dependencies(
                self.raw.as_ptr(),
                pointers.as_ptr(),
                pointers.len(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        self.dependencies = dependencies.to_vec();
        self.bound = true;
        Ok(())
    }

    pub fn begin(&self) -> Result<(), String> {
        assert!(self.provisioned && self.bound, "unfinished schedule cannot execute");
        let status = unsafe { gpu_prepared_schedule_begin(self.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    pub fn end(&self) -> Result<(), String> {
        assert!(self.provisioned && self.bound, "unfinished schedule cannot execute");
        let status = unsafe { gpu_prepared_schedule_end(self.raw.as_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    pub fn is_ready(&self) -> Result<bool, String> {
        let mut ready = false;
        let status = unsafe {
            crate::poly::dcrt::gpu::gpu_prepared_schedule_is_ready(self.raw.as_ptr(), &mut ready)
        };
        if status != 0 { Err(last_error_string()) } else { Ok(ready) }
    }
}

impl Drop for GpuPreparedSchedule {
    fn drop(&mut self) {
        unsafe { gpu_prepared_schedule_destroy(self.raw.as_ptr()) };
        self.dependencies.clear();
        self.owners.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        sampler::gpu::sample_seeded_gadget_source_columns,
    };
    use rand::Rng;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_schedule_independent_streams_join() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let left_seed = crate::poly::dcrt::gpu::GpuRngSeed::from_bytes(rand::rng().random());
        let right_seed = crate::poly::dcrt::gpu::GpuRngSeed::from_bytes(rand::rng().random());
        let expected = sample_seeded_gadget_source_columns(&params, 1, 1, 0, 1, left_seed) +
            sample_seeded_gadget_source_columns(&params, 1, 1, 0, 1, right_seed);
        let output = || {
            Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                &params,
                1,
                1,
                params.crt_depth() - 1,
                false,
                None,
            ))
        };
        let a = output();
        let b = output();
        let joined = output();
        let left_plan = GpuPreparedSampling::bind(
            Arc::clone(&a),
            GpuMatrixSampleDist::Uniform,
            0.0,
            u64::MAX,
            1,
            0,
            None,
        )
        .unwrap();
        let right_plan = GpuPreparedSampling::bind(
            Arc::clone(&b),
            GpuMatrixSampleDist::Uniform,
            0.0,
            u64::MAX,
            1,
            0,
            None,
        )
        .unwrap();
        let join_plan = GpuPreparedArithmetic::bind(
            GpuPreparedArithmeticKind::Add,
            Arc::clone(&a),
            Some(Arc::clone(&b)),
            Arc::clone(&joined),
        )
        .unwrap();
        let both = GpuPreparedSchedule::new(
            &[
                GpuPreparedSchedulePlan::Sampling(&left_plan),
                GpuPreparedSchedulePlan::Sampling(&right_plan),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(both.stream_count(), 2, "independent branches must use distinct native streams");
        let schedule = |plan| {
            let mut schedule =
                GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Sampling(plan)], &[]).unwrap();
            schedule.provision().unwrap();
            schedule.bind_dependencies(&[]).unwrap();
            Arc::new(schedule)
        };
        let left_schedule = schedule(&left_plan);
        let right_schedule = schedule(&right_plan);
        let mut join_schedule =
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Arithmetic(&join_plan)], &[])
                .unwrap();
        join_schedule.provision().unwrap();
        join_schedule
            .bind_dependencies(&[Arc::clone(&left_schedule), Arc::clone(&right_schedule)])
            .unwrap();
        assert!(join_schedule.bind_dependencies(&[]).is_err());
        expected.wait_until_ready();
        unsafe extern "C" {
            fn gpu_test_matrix_stream_gate(
                matrix: *mut crate::poly::dcrt::gpu::GpuMatrixOpaque,
            ) -> *mut std::ffi::c_void;
            fn gpu_test_release_stream_gate(gate: *mut std::ffi::c_void);
        }
        struct Gate(*mut std::ffi::c_void);
        impl Drop for Gate {
            fn drop(&mut self) {
                unsafe { gpu_test_release_stream_gate(self.0) };
            }
        }
        let gate = Gate(unsafe { gpu_test_matrix_stream_gate(a.raw) });
        assert!(!gate.0.is_null());
        left_schedule.begin().unwrap();
        left_plan.submit(left_seed).unwrap();
        left_schedule.end().unwrap();
        right_schedule.begin().unwrap();
        right_plan.submit(right_seed).unwrap();
        right_schedule.end().unwrap();
        join_schedule.begin().unwrap();
        join_plan.submit().unwrap();
        join_schedule.end().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !b.is_ready().unwrap() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let independent_ready = b.is_ready().unwrap();
        let join_pending = !joined.is_ready().unwrap();
        drop(gate);
        assert!(independent_ready, "independent stream was serialized behind blocked branch");
        assert!(join_pending, "join bypassed its blocked predecessor");
        joined.wait_until_ready();
        #[cfg(feature = "gpu-instrumentation")]
        {
            crate::poly::dcrt::gpu::gpu_test_reset_work_counters();
            crate::poly::dcrt::gpu::gpu_test_set_work_gate(true);
        }
        // Identical production submission, with no allocation, validation,
        // event construction, or stream selection inside this loop.
        for _ in 0..300 {
            left_schedule.begin().unwrap();
            left_plan.submit(left_seed).unwrap();
            left_schedule.end().unwrap();
            right_schedule.begin().unwrap();
            right_plan.submit(right_seed).unwrap();
            right_schedule.end().unwrap();
            join_schedule.begin().unwrap();
            join_plan.submit().unwrap();
            join_schedule.end().unwrap();
        }
        #[cfg(feature = "gpu-instrumentation")]
        {
            crate::poly::dcrt::gpu::gpu_test_set_work_gate(false);
            let (events, streams, validations, allocations, kernels, measurements) =
                crate::poly::dcrt::gpu::gpu_test_work_counters();
            assert_eq!((events, streams, validations, allocations, measurements), (0, 0, 0, 0, 0));
            assert!(kernels >= 600, "instrumented replay must launch both sampler branches");
        }
        assert_eq!(joined.to_cpu_matrix(), expected.to_cpu_matrix());
    }
}
