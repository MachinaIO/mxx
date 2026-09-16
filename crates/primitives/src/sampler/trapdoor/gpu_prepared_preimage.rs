//! Prepared preimage sequence. Allocation and binding happen only in `bind`.

use super::{GpuDCRTTrapdoor, preimage_c, preimage_seed, preimage_smoothing_parameter};
use crate::{
    matrix::{
        PolyMatrix, SmallMatrixError, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixSampleDist, GpuPreparedArithmetic,
            GpuPreparedArithmeticCommand, GpuPreparedArithmeticKind, GpuPreparedInputCopy,
            GpuPreparedPreimageCutoff, GpuPreparedPreimagePhases, GpuPreparedRange,
            GpuPreparedSampling, GpuPreparedSchedule, GpuPreparedSchedulePlan, GpuPreparedSlotKind,
            GpuPreparedTransform, GpuPreparedTranspose, GpuPreparedView, GpuSmallMatrix,
            GpuTracedClaim,
        },
    },
    poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
    sampler::bounds::default_preimage_cutoff,
};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum GpuPreparedPreimageError {
    #[error("{0}")]
    Sampling(#[from] SmallMatrixError),
    #[error("prepared preimage GPU failure: {0}")]
    Gpu(String),
}

pub struct GpuPreparedPreimageSampler {
    inputs: [GpuPreparedInputCopy; 2],
    transposes: [Arc<GpuPreparedTranspose>; 2],
    grams: [GpuPreparedArithmeticCommand; 3],
    gram_transforms: [(GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>); 3],
    p2: Arc<GpuPreparedSampling>,
    product: [GpuPreparedArithmeticCommand; 2],
    product_transform: (GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>),
    phases: GpuPreparedPreimagePhases,
    assemble: [GpuPreparedArithmeticCommand; 2],
    residual: [GpuPreparedArithmeticCommand; 2],
    residual_transform: (GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>),
    correction: [GpuPreparedArithmeticCommand; 2],
    publish: [GpuPreparedArithmeticCommand; 2],
    candidate_transform: (GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>),
    attempts: usize,
    column_start: usize,
    columns: usize,
    submitted: bool,
    // All masked sampler plans above drop before their acceptance owner.
    cutoff: GpuPreparedPreimageCutoff,
}

impl GpuPreparedPreimageSampler {
    fn matrix_layout(params: &GpuDCRTPolyParams, d: usize, columns: usize) -> [GpuTracedClaim; 16] {
        let k = params.modulus_digits();
        [
            (d, d * k),
            (d, d * k),
            (d * k, d),
            (d * k, d),
            (d, d),
            (d, d),
            (d, d),
            (d * k, columns),
            (2 * d, columns),
            (2 * d, columns),
            (d * (2 + k), columns),
            (d, columns),
            (d, columns),
            (d * k, columns),
            (2 * d, columns),
            (d * (2 + k), columns),
        ]
        .map(|(rows, columns)| GpuTracedClaim::matrix(rows, columns, params.crt_depth() - 1, true))
    }

    pub fn allocation_claims(
        params: &GpuDCRTPolyParams,
        rows: usize,
        output: &GpuSmallMatrix,
    ) -> Result<Vec<GpuTracedClaim>, String> {
        let mut claims = Self::matrix_layout(params, rows, output.size().1).to_vec();
        claims.extend(
            GpuPreparedPreimagePhases::allocation_layout(params, rows, output.size().1)?
                .into_iter()
                .filter(|layout| {
                    layout.bytes != 0 || layout.kind == GpuPreparedSlotKind::CompletionEvent
                })
                .map(GpuTracedClaim::workspace),
        );
        claims.extend(
            GpuPreparedPreimageCutoff::allocation_layout(output)?
                .into_iter()
                .map(GpuTracedClaim::workspace),
        );
        Ok(claims)
    }

    pub fn bind(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public: Arc<GpuDCRTPolyMatrix>,
        target: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuSmallMatrix>,
        sigma: f64,
        column_start: usize,
    ) -> Result<Self, String> {
        let d = public.row_size();
        let columns = target.col_size();
        let k = params.modulus_digits();
        let base = 1u32.checked_shl(params.base_bits()).ok_or("preimage base overflow")?;
        let minimum = default_preimage_cutoff(params.ring_dimension(), d, k, base, sigma)
            .ok_or("invalid prepared preimage parameters")?;
        if params.dropped_moduli() != 0 ||
            d == 0 ||
            columns == 0 ||
            target.row_size() != d ||
            public.col_size() != d * (2 + k) ||
            trapdoor.r.size() != (d, d * k) ||
            trapdoor.e.size() != (d, d * k) ||
            output.size() != (d * (2 + k), columns) ||
            output.max_coefficient_bound() < &minimum ||
            public.params() != params ||
            target.params() != params ||
            output.params() != params ||
            trapdoor.r.params() != params ||
            trapdoor.e.params() != params ||
            !public.is_ntt() ||
            !target.is_ntt()
        {
            return Err("prepared preimage input/output contract mismatch".into());
        }
        let attempts = crate::env::gpu_preimage_max_tile_attempts()?;
        let c = preimage_c(base, sigma);
        let smoothing =
            preimage_smoothing_parameter(base, sigma, d, params.ring_dimension() as usize, k);
        let [
            r,
            e,
            rt,
            et,
            gram_a,
            gram_b,
            gram_d,
            p2_owner,
            product_owner,
            p1_owner,
            perturbation,
            public_product,
            residual_owner,
            z,
            correction_owner,
            candidate,
        ] = Self::matrix_layout(params, d, columns).map(|claim| {
            Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                params,
                claim.rows(),
                claim.columns(),
                claim.level().expect("matrix layout level"),
                claim.is_evaluation().expect("matrix layout format"),
                None,
            ))
        });
        let coefficient = |owner: &Arc<GpuDCRTPolyMatrix>| {
            GpuDCRTPolyMatrix::prepared_shape(
                Arc::clone(owner),
                owner.row_size(),
                owner.col_size(),
                owner.level(),
                false,
            )
        };
        let arithmetic = |kind,
                          lhs: &Arc<GpuDCRTPolyMatrix>,
                          rhs: Option<&Arc<GpuDCRTPolyMatrix>>,
                          out: &Arc<GpuDCRTPolyMatrix>| {
            GpuPreparedArithmetic::bind(kind, Arc::clone(lhs), rhs.cloned(), Arc::clone(out))
        };
        let ranged = |kind,
                      lhs: &Arc<GpuDCRTPolyMatrix>,
                      rhs: Option<&Arc<GpuDCRTPolyMatrix>>,
                      out: &Arc<GpuDCRTPolyMatrix>,
                      left_rows,
                      right_rows,
                      output_rows| {
            GpuPreparedArithmetic::bind_with_view(
                kind,
                Arc::clone(lhs),
                rhs.cloned(),
                Arc::clone(out),
                Some(GpuPreparedView {
                    left: GpuPreparedRange { rows: left_rows, columns: 0..lhs.col_size() },
                    right: GpuPreparedRange {
                        rows: right_rows,
                        columns: 0..rhs.map_or(lhs.col_size(), |value| value.col_size()),
                    },
                    output: GpuPreparedRange { rows: output_rows, columns: 0..out.col_size() },
                }),
                0,
            )
        };
        let inputs = [
            GpuPreparedInputCopy::bind(Arc::clone(&r), Arc::clone(&r), None)?,
            GpuPreparedInputCopy::bind(Arc::clone(&e), Arc::clone(&e), None)?,
        ];
        let transposes = [
            GpuPreparedTranspose::bind(Arc::clone(&r), Arc::clone(&rt), None)?,
            GpuPreparedTranspose::bind(Arc::clone(&e), Arc::clone(&et), None)?,
        ];
        let gram_owners = [gram_a, gram_b, gram_d];
        let grams = [
            arithmetic(GpuPreparedArithmeticKind::Multiply, &r, Some(&rt), &gram_owners[0])?,
            arithmetic(GpuPreparedArithmeticKind::Multiply, &r, Some(&et), &gram_owners[1])?,
            arithmetic(GpuPreparedArithmeticKind::Multiply, &e, Some(&et), &gram_owners[2])?,
        ];
        let gram_coeff = [
            coefficient(&gram_owners[0])?,
            coefficient(&gram_owners[1])?,
            coefficient(&gram_owners[2])?,
        ];
        let gram_transforms = [
            (GpuPreparedTransform::new_inverse(&gram_coeff[0])?, Arc::clone(&gram_coeff[0])),
            (GpuPreparedTransform::new_inverse(&gram_coeff[1])?, Arc::clone(&gram_coeff[1])),
            (GpuPreparedTransform::new_inverse(&gram_coeff[2])?, Arc::clone(&gram_coeff[2])),
        ];
        let p2 = GpuPreparedSampling::bind(
            Arc::clone(&p2_owner),
            GpuMatrixSampleDist::Gauss,
            (smoothing * smoothing - c * c).sqrt(),
            u64::MAX,
            columns,
            0,
            None,
        )?;
        let product = [
            ranged(
                GpuPreparedArithmeticKind::Multiply,
                &r,
                Some(&p2_owner),
                &product_owner,
                0..d,
                0..d * k,
                0..d,
            )?,
            ranged(
                GpuPreparedArithmeticKind::Multiply,
                &e,
                Some(&p2_owner),
                &product_owner,
                0..d,
                0..d * k,
                d..2 * d,
            )?,
        ];
        let product_coeff = coefficient(&product_owner)?;
        let product_transform =
            (GpuPreparedTransform::new_inverse(&product_coeff)?, Arc::clone(&product_coeff));
        let assemble = [
            ranged(
                GpuPreparedArithmeticKind::Copy,
                &p1_owner,
                None,
                &perturbation,
                0..2 * d,
                0..2 * d,
                0..2 * d,
            )?,
            ranged(
                GpuPreparedArithmeticKind::Copy,
                &p2_owner,
                None,
                &perturbation,
                0..d * k,
                0..d * k,
                2 * d..d * (2 + k),
            )?,
        ];
        let residual = [
            arithmetic(
                GpuPreparedArithmeticKind::Multiply,
                &public,
                Some(&perturbation),
                &public_product,
            )?,
            arithmetic(
                GpuPreparedArithmeticKind::Subtract,
                &target,
                Some(&public_product),
                &residual_owner,
            )?,
        ];
        let residual_coeff = coefficient(&residual_owner)?;
        let residual_transform =
            (GpuPreparedTransform::new_inverse(&residual_coeff)?, Arc::clone(&residual_coeff));
        let mut phases = GpuPreparedPreimagePhases::bind(
            [
                Arc::clone(&gram_coeff[0]),
                Arc::clone(&gram_coeff[1]),
                Arc::clone(&gram_coeff[2]),
                product_coeff,
                Arc::clone(&p1_owner),
                residual_coeff,
                Arc::clone(&z),
            ],
            params.base_bits(),
            c,
            smoothing,
            sigma,
        )?;
        let correction = [
            ranged(
                GpuPreparedArithmeticKind::Multiply,
                &r,
                Some(&z),
                &correction_owner,
                0..d,
                0..d * k,
                0..d,
            )?,
            ranged(
                GpuPreparedArithmeticKind::Multiply,
                &e,
                Some(&z),
                &correction_owner,
                0..d,
                0..d * k,
                d..2 * d,
            )?,
        ];
        let publish = [
            ranged(
                GpuPreparedArithmeticKind::Add,
                &p1_owner,
                Some(&correction_owner),
                &candidate,
                0..2 * d,
                0..2 * d,
                0..2 * d,
            )?,
            ranged(
                GpuPreparedArithmeticKind::Add,
                &p2_owner,
                Some(&z),
                &candidate,
                0..d * k,
                0..d * k,
                2 * d..d * (2 + k),
            )?,
        ];
        let candidate_coeff = coefficient(&candidate)?;
        let candidate_transform =
            (GpuPreparedTransform::new_inverse(&candidate_coeff)?, Arc::clone(&candidate_coeff));
        let cutoff = GpuPreparedPreimageCutoff::bind(vec![(output, candidate_coeff, 0, 0)])?;
        // This struct drops the phase/sampler owners before cutoff; submission
        // is exclusive through &mut self and all readers precede each update.
        unsafe {
            phases.bind_acceptance(&cutoff, 0)?;
            p2.bind_preimage_acceptance(&cutoff, 0)?;
        }
        Ok(Self {
            inputs,
            transposes,
            grams,
            gram_transforms,
            p2,
            product,
            product_transform,
            phases,
            assemble,
            residual,
            residual_transform,
            correction,
            publish,
            candidate_transform,
            attempts,
            column_start,
            columns,
            submitted: false,
            cutoff,
        })
    }

    pub fn submit(&mut self, trapdoor: &GpuDCRTTrapdoor, seed: [u8; 32]) -> Result<(), String> {
        self.cutoff.begin()?;
        self.submitted = true;
        self.inputs[0].submit_borrowed(&trapdoor.r)?;
        self.inputs[1].submit_borrowed(&trapdoor.e)?;
        for command in &self.transposes {
            command.submit()?;
        }
        for command in &self.grams {
            command.submit()?;
        }
        for (transform, owner) in &self.gram_transforms {
            transform.submit_shared(owner)?;
        }
        self.phases.refresh_covariance()?;
        for attempt in 0..self.attempts {
            let candidate = preimage_seed(seed, b"candidate", self.column_start, attempt);
            let perturb = preimage_seed(candidate.to_bytes(), b"perturb", 0, 0);
            self.p2.submit(preimage_seed(perturb.to_bytes(), b"p2", 0, 0))?;
            for command in &self.product {
                command.submit()?;
            }
            self.product_transform.0.submit_shared(&self.product_transform.1)?;
            self.phases.sample_p1(preimage_seed(perturb.to_bytes(), b"p1", 0, 0))?;
            for command in &self.assemble {
                command.submit()?;
            }
            for command in &self.residual {
                command.submit()?;
            }
            self.residual_transform.0.submit_shared(&self.residual_transform.1)?;
            self.phases.sample_gadget(preimage_seed(candidate.to_bytes(), b"z", 0, 0))?;
            for command in &self.correction {
                command.submit()?;
            }
            for command in &self.publish {
                command.submit()?;
            }
            self.candidate_transform.0.submit_shared(&self.candidate_transform.1)?;
            self.cutoff.submit()?;
        }
        Ok(())
    }

    pub fn wait(&mut self) -> Result<(), GpuPreparedPreimageError> {
        if !self.submitted {
            return Ok(());
        }
        if self.cutoff.wait().map_err(GpuPreparedPreimageError::Gpu)?[0] == 0 {
            return Err(SmallMatrixError::AttemptExhausted {
                column_start: self.column_start,
                column_count: self.columns,
                attempts: self.attempts,
            }
            .into());
        }
        Ok(())
    }

    pub fn is_ready(&mut self) -> Result<bool, GpuPreparedPreimageError> {
        if !self.submitted {
            return Ok(true);
        }
        let Some(status) = self.cutoff.poll().map_err(GpuPreparedPreimageError::Gpu)? else {
            return Ok(false);
        };
        if status[0] == 0 {
            return Err(SmallMatrixError::AttemptExhausted {
                column_start: self.column_start,
                column_count: self.columns,
                attempts: self.attempts,
            }
            .into());
        }
        Ok(true)
    }

    pub fn schedule(&self) -> Result<GpuPreparedSchedule, String> {
        let mut plans = Vec::new();
        plans.extend(self.inputs.iter().map(GpuPreparedSchedulePlan::InputCopy));
        plans.extend(self.transposes.iter().map(|plan| GpuPreparedSchedulePlan::Transpose(plan)));
        for commands in [
            &self.grams[..],
            &self.product,
            &self.assemble,
            &self.residual,
            &self.correction,
            &self.publish,
        ] {
            plans.extend(commands.iter().map(GpuPreparedSchedulePlan::Arithmetic));
        }
        for (transform, owner) in self.gram_transforms.iter().chain([
            &self.product_transform,
            &self.residual_transform,
            &self.candidate_transform,
        ]) {
            plans.push(GpuPreparedSchedulePlan::Transform(transform, owner));
        }
        plans.push(GpuPreparedSchedulePlan::Sampling(&self.p2));
        plans.push(GpuPreparedSchedulePlan::PreimagePhases(&self.phases));
        plans.push(GpuPreparedSchedulePlan::PreimageCutoff(&self.cutoff));
        GpuPreparedSchedule::new(&plans, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::ResidentPolyMatrixColumnSource,
        poly::dcrt::params::DCRTPolyParams,
        sampler::{
            DistType, PolyTrapdoorSampler, PolyUniformSampler,
            gpu::GpuDCRTPolyUniformSampler,
            trapdoor::gpu::{GpuDCRTPolyTrapdoorSampler, gpu_params_from_cpu},
        },
    };

    #[test]
    #[serial_test::serial]
    fn test_gpu_prepared_preimage_matches_existing_sampler_and_changed_trapdoor() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let d = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(1);
        let params = gpu_params_from_cpu(&DCRTPolyParams::new(n, 2, 17, 2, None, None));
        let sigma = 3.2;
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, sigma);
        let bound =
            default_preimage_cutoff(n, d, params.modulus_digits(), 1 << params.base_bits(), sigma)
                .unwrap();
        let (trapdoor, public) = sampler.trapdoor(&params, d);
        let public = Arc::new(public);
        let target = Arc::new(GpuDCRTPolyUniformSampler::new().sample_uniform(
            &params,
            d,
            1,
            DistType::FinRingDist,
        ));
        let output = Arc::new(
            GpuSmallMatrix::new_empty(&params, d * (2 + params.modulus_digits()), 1, bound.clone())
                .unwrap(),
        );
        let mut plan = GpuPreparedPreimageSampler::bind(
            &params,
            &trapdoor,
            Arc::clone(&public),
            Arc::clone(&target),
            Arc::clone(&output),
            sigma,
            0,
        )
        .unwrap();
        let seed = rand::random();
        let target_source = ResidentPolyMatrixColumnSource::new(target.as_ref().clone());
        let expected = sampler
            .preimage(&params, &trapdoor, public.as_ref(), &target_source, bound.clone(), seed)
            .unwrap();
        #[cfg(feature = "gpu-instrumentation")]
        {
            crate::poly::dcrt::gpu::gpu_test_reset_work_counters();
            crate::poly::dcrt::gpu::gpu_test_set_work_gate(true);
        }
        let submitted = plan.submit(&trapdoor, seed);
        #[cfg(feature = "gpu-instrumentation")]
        {
            crate::poly::dcrt::gpu::gpu_test_set_work_gate(false);
            let (events, streams, validations, allocations, launches, measurements) =
                crate::poly::dcrt::gpu::gpu_test_work_counters();
            assert_eq!((events, streams, validations, allocations, measurements), (0, 0, 0, 0, 0));
            assert!(launches > 0);
        }
        submitted.unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !plan.is_ready().unwrap() {
            assert!(std::time::Instant::now() < deadline, "prepared preimage completion timed out");
            std::thread::yield_now();
        }
        plan.wait().unwrap();
        assert_eq!(
            output.to_canonical_coefficients().unwrap(),
            expected.to_canonical_coefficients().unwrap()
        );

        let (next_trapdoor, next_public) = sampler.trapdoor(&params, d);
        let copy =
            GpuPreparedInputCopy::bind(Arc::clone(&public), Arc::new(next_public.clone()), None)
                .unwrap();
        copy.submit_borrowed(&next_public).unwrap();
        let next_seed = rand::random();
        let expected = sampler
            .preimage(&params, &next_trapdoor, &next_public, &target_source, bound, next_seed)
            .unwrap();
        plan.submit(&next_trapdoor, next_seed).unwrap();
        plan.wait().unwrap();
        assert_eq!(
            output.to_canonical_coefficients().unwrap(),
            expected.to_canonical_coefficients().unwrap()
        );
    }
}
