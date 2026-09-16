//! Fixed trapdoor owners and sampler/arithmetic commands.

use super::GpuDCRTTrapdoor;
use crate::{
    matrix::{
        PolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixRangeConstant, GpuMatrixSampleDist, GpuPreparedArithmetic,
            GpuPreparedArithmeticCommand, GpuPreparedArithmeticKind, GpuPreparedRange,
            GpuPreparedSampling, GpuPreparedSchedule, GpuPreparedSchedulePlan,
            GpuPreparedTransform, GpuPreparedTranspose, GpuPreparedView, GpuTracedClaim,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, GpuRngSeed},
    },
};
use std::sync::{Arc, Mutex};

pub struct GpuPreparedTrapdoorSampler {
    samplers: [Arc<GpuPreparedSampling>; 3],
    transposes: [Arc<GpuPreparedTranspose>; 2],
    grams: [GpuPreparedArithmeticCommand; 3],
    gram_transforms: [(GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>); 3],
    public_commands: Box<[GpuPreparedArithmeticCommand]>,
    trapdoor: Arc<GpuDCRTTrapdoor>,
    public: Arc<GpuDCRTPolyMatrix>,
}

impl GpuPreparedTrapdoorSampler {
    pub fn allocation_claims(params: &GpuDCRTPolyParams, d: usize) -> [GpuTracedClaim; 13] {
        let k = params.modulus_digits();
        [
            (d, d * k),
            (d, d * k),
            (d, d),
            (d * k, d),
            (d * k, d),
            (d, d),
            (d, d),
            (d, d),
            (d, d * k),
            (d, d * k),
            (d, d * k),
            (d, d * k),
            (d, d),
        ]
        .map(|(rows, columns)| GpuTracedClaim::matrix(rows, columns, params.crt_depth() - 1, true))
    }
    pub fn bind(
        params: &GpuDCRTPolyParams,
        public: Arc<GpuDCRTPolyMatrix>,
        sigma: f64,
    ) -> Result<Self, String> {
        let d = public.row_size();
        let k = params.modulus_digits();
        if params.dropped_moduli() != 0 ||
            d == 0 ||
            public.col_size() != d * (2 + k) ||
            public.params() != params ||
            !public.is_ntt() ||
            !sigma.is_finite() ||
            sigma <= 0.0
        {
            return Err("prepared trapdoor contract mismatch".into());
        }
        let [
            r,
            e,
            abar,
            rt,
            et,
            gram_a,
            gram_b,
            gram_d,
            product,
            sum,
            tail,
            mut gadget,
            mut identity,
        ] = Self::allocation_claims(params, d).map(|claim| {
            Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                params,
                claim.rows(),
                claim.columns(),
                claim.level().expect("matrix layout level"),
                claim.is_evaluation().expect("matrix layout format"),
                None,
            ))
        });
        Arc::get_mut(&mut gadget).expect("fresh gadget owner").fill_constant_columns(
            0..d,
            0..d * k,
            0,
            GpuMatrixRangeConstant::Gadget { small: false, digit_count: Some(k) },
        )?;
        Arc::get_mut(&mut identity).expect("fresh identity owner").fill_constant_columns(
            0..d,
            0..d,
            0,
            GpuMatrixRangeConstant::Identity,
        )?;
        let header = |owner: &Arc<GpuDCRTPolyMatrix>, evaluation| {
            GpuDCRTPolyMatrix::prepared_shape(
                Arc::clone(owner),
                owner.row_size(),
                owner.col_size(),
                owner.level(),
                evaluation,
            )
        };
        let arithmetic = |kind,
                          lhs: &Arc<GpuDCRTPolyMatrix>,
                          rhs: Option<&Arc<GpuDCRTPolyMatrix>>,
                          output: &Arc<GpuDCRTPolyMatrix>| {
            GpuPreparedArithmetic::bind(kind, Arc::clone(lhs), rhs.cloned(), Arc::clone(output))
        };
        let samplers = [
            GpuPreparedSampling::bind(
                Arc::clone(&r),
                GpuMatrixSampleDist::Gauss,
                sigma,
                u64::MAX,
                d * k,
                0,
                None,
            )?,
            GpuPreparedSampling::bind(
                Arc::clone(&e),
                GpuMatrixSampleDist::Gauss,
                sigma,
                u64::MAX,
                d * k,
                0,
                None,
            )?,
            GpuPreparedSampling::bind(
                Arc::clone(&abar),
                GpuMatrixSampleDist::Uniform,
                0.0,
                0,
                d,
                0,
                None,
            )?,
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
        let coefficients = [
            header(&gram_owners[0], false)?,
            header(&gram_owners[1], false)?,
            header(&gram_owners[2], false)?,
        ];
        let gram_transforms = [
            (GpuPreparedTransform::new_inverse(&coefficients[0])?, Arc::clone(&coefficients[0])),
            (GpuPreparedTransform::new_inverse(&coefficients[1])?, Arc::clone(&coefficients[1])),
            (GpuPreparedTransform::new_inverse(&coefficients[2])?, Arc::clone(&coefficients[2])),
        ];
        let own_header = |owner: &Arc<GpuDCRTPolyMatrix>, evaluation| {
            Arc::try_unwrap(header(owner, evaluation)?)
                .map_err(|_| "fresh prepared trapdoor header is unexpectedly shared".to_owned())
        };
        let trapdoor = Arc::new(GpuDCRTTrapdoor {
            r: own_header(&r, true)?,
            e: own_header(&e, true)?,
            a_mat_coeff: own_header(&gram_owners[0], false)?,
            b_mat_coeff: own_header(&gram_owners[1], false)?,
            d_mat_coeff: own_header(&gram_owners[2], false)?,
            p1_covariance_cache: Arc::new(Mutex::new(None)),
        });
        let mut public_commands = vec![
            arithmetic(GpuPreparedArithmeticKind::Multiply, &abar, Some(&r), &product)?,
            arithmetic(GpuPreparedArithmeticKind::Add, &product, Some(&e), &sum)?,
            arithmetic(GpuPreparedArithmeticKind::Subtract, &gadget, Some(&sum), &tail)?,
        ];
        for (source, start) in [(abar, 0), (identity, d), (tail, 2 * d)] {
            public_commands.push(GpuPreparedArithmetic::bind_with_view(
                GpuPreparedArithmeticKind::Copy,
                Arc::clone(&source),
                None,
                Arc::clone(&public),
                Some(GpuPreparedView {
                    left: GpuPreparedRange { rows: 0..d, columns: 0..source.col_size() },
                    right: GpuPreparedRange { rows: 0..d, columns: 0..source.col_size() },
                    output: GpuPreparedRange {
                        rows: 0..d,
                        columns: start..start + source.col_size(),
                    },
                }),
                0,
            )?);
        }
        Ok(Self {
            samplers,
            transposes,
            grams,
            gram_transforms,
            public_commands: public_commands.into_boxed_slice(),
            trapdoor,
            public,
        })
    }

    pub fn submit(&mut self, seeds: [GpuRngSeed; 3]) -> Result<(), String> {
        for (sampler, seed) in self.samplers.iter().zip(seeds) {
            sampler.submit(seed)?;
        }
        for command in &self.transposes {
            command.submit()?;
        }
        for command in &self.grams {
            command.submit()?;
        }
        for (command, owner) in &self.gram_transforms {
            command.submit_shared(owner)?;
        }
        for command in &self.public_commands {
            command.submit()?;
        }
        Ok(())
    }

    pub fn trapdoor(&self) -> &Arc<GpuDCRTTrapdoor> {
        &self.trapdoor
    }
    pub fn public(&self) -> &Arc<GpuDCRTPolyMatrix> {
        &self.public
    }

    pub fn schedule(&self) -> Result<GpuPreparedSchedule, String> {
        let mut plans = Vec::new();
        plans.extend(self.samplers.iter().map(|plan| GpuPreparedSchedulePlan::Sampling(plan)));
        plans.extend(self.transposes.iter().map(|plan| GpuPreparedSchedulePlan::Transpose(plan)));
        plans.extend(
            self.grams
                .iter()
                .chain(self.public_commands.iter())
                .map(GpuPreparedSchedulePlan::Arithmetic),
        );
        for (transform, owner) in &self.gram_transforms {
            plans.push(GpuPreparedSchedulePlan::Transform(transform, owner));
        }
        GpuPreparedSchedule::new(&plans, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly::dcrt::params::DCRTPolyParams, sampler::trapdoor::gpu::gpu_params_from_cpu};

    #[test]
    #[serial_test::serial]
    fn test_gpu_prepared_trapdoor_preserves_gadget_identity_on_reuse() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let d = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(1);
        let params = gpu_params_from_cpu(&DCRTPolyParams::new(n, 2, 17, 2, None, None));
        let public = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
            &params,
            d,
            d * (2 + params.modulus_digits()),
            params.crt_depth() - 1,
            true,
            None,
        ));
        let mut plan = GpuPreparedTrapdoorSampler::bind(&params, Arc::clone(&public), 3.2).unwrap();
        let identity = GpuDCRTPolyMatrix::identity(&params, d * params.modulus_digits(), None);
        let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, d, None);
        for _ in 0..2 {
            let seeds = std::array::from_fn(|_| GpuRngSeed::from_bytes(rand::random()));
            #[cfg(feature = "gpu-instrumentation")]
            {
                crate::poly::dcrt::gpu::gpu_test_reset_work_counters();
                crate::poly::dcrt::gpu::gpu_test_set_work_gate(true);
            }
            let submitted = plan.submit(seeds);
            #[cfg(feature = "gpu-instrumentation")]
            {
                crate::poly::dcrt::gpu::gpu_test_set_work_gate(false);
                let (events, streams, validations, allocations, launches, measurements) =
                    crate::poly::dcrt::gpu::gpu_test_work_counters();
                assert_eq!(
                    (events, streams, validations, allocations, measurements),
                    (0, 0, 0, 0, 0)
                );
                assert!(launches > 0);
            }
            submitted.unwrap();
            let trapdoor = plan.trapdoor();
            let secret = trapdoor.r.concat_rows(&[&trapdoor.e, &identity]);
            assert_eq!(&*public * &secret, gadget);
        }
    }
}
