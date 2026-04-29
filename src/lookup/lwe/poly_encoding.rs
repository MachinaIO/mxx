#[cfg(feature = "gpu")]
#[path = "poly_encoding_gpu.rs"]
mod gpu;

use crate::{
    bench_estimator::{PolyEncodingChunkBenchMeasurement, PolyEncodingPublicLutBenchEstimator},
    bgg::{poly_encoding::BggPolyEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{DistType, PolyHashSampler, PolyUniformSampler},
};
use std::{marker::PhantomData, ops::Mul, path::PathBuf, sync::Arc, time::Instant};
use tracing::debug;

#[cfg(not(feature = "gpu"))]
use crate::bench_estimator::benchmark_gate_operation;
#[cfg(not(feature = "gpu"))]
use rayon::prelude::*;
#[cfg(not(feature = "gpu"))]
use std::path::Path;

use super::derive_a_lt_matrix;
#[cfg(not(feature = "gpu"))]
use super::{
    encoding::{public_lookup_output_chunk_cpu, public_lookup_slot_cpu},
    k_high_chunk_count,
};

#[derive(Debug, Clone)]
pub struct LWEBGGPolyEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub c_b_compact_bytes_by_slot: Vec<Arc<[u8]>>,
    _marker: PhantomData<SH>,
}

impl<M, SH> LWEBGGPolyEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn build_c_b_compact_bytes_by_slot<US>(
        params: &<M::P as Poly>::Params,
        secret_vec: &M,
        b_matrix: &M,
        slot_secret_mats: &[Vec<u8>],
        gauss_sigma: Option<f64>,
    ) -> Vec<Arc<[u8]>>
    where
        US: PolyUniformSampler<M = M>,
        for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
    {
        let error_sampler = match gauss_sigma {
            Some(sigma) if sigma == 0.0 => None,
            None => None,
            Some(sigma) => {
                assert!(sigma > 0.0, "gauss_sigma must be positive when it is non-zero");
                Some((US::new(), sigma))
            }
        };

        slot_secret_mats
            .iter()
            .map(|slot_secret_mat_bytes| {
                let slot_secret_mat = M::from_compact_bytes(params, slot_secret_mat_bytes);
                let transformed_secret_vec = secret_vec.clone() * &slot_secret_mat;
                let mut c_b = transformed_secret_vec * b_matrix;
                if let Some((error_sampler, sigma)) = &error_sampler {
                    let error = error_sampler.sample_uniform(
                        params,
                        c_b.row_size(),
                        c_b.col_size(),
                        DistType::GaussDist { sigma: *sigma },
                    );
                    c_b = c_b + error;
                }
                Arc::<[u8]>::from(c_b.into_compact_bytes())
            })
            .collect()
    }

    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        c_b_compact_bytes_by_slot: Vec<Arc<[u8]>>,
    ) -> Self {
        Self { hash_key, dir_path, c_b_compact_bytes_by_slot, _marker: PhantomData }
    }
}

impl<M, SH> PltEvaluator<BggPolyEncoding<M>> for LWEBGGPolyEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<BggPolyEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPolyEncoding<M> as Evaluable>::P>,
        _: &BggPolyEncoding<M>,
        input: &BggPolyEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPolyEncoding<M> {
        let public_lookup_started = Instant::now();
        let num_slots = input.num_slots();
        debug!(
            "LWE BGG poly-encoding public lookup started: gate_id={}, lut_id={}, slot_count={}",
            gate_id, lut_id, num_slots
        );

        let plaintext_compact_bytes_by_slot = input
            .plaintext_bytes
            .as_ref()
            .expect("the BGG poly encoding should reveal plaintexts for public lookup");
        assert_eq!(
            plaintext_compact_bytes_by_slot.len(),
            num_slots,
            "BGG poly encoding public lookup requires plaintext_bytes.len() == num_slots"
        );
        assert_eq!(
            self.c_b_compact_bytes_by_slot.len(),
            num_slots,
            "slot-wise c_b compact-bytes cache must match the BGG poly encoding slot count"
        );

        let row_size = input.pubkey.matrix.row_size();
        let out_pubkey = BggPublicKey::new(
            derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id),
            true,
        );

        #[cfg(feature = "gpu")]
        {
            let (output_vector_bytes, output_plaintext_bytes) = gpu::public_lookup_poly_gpu::<M, SH>(
                self,
                params,
                plt,
                input,
                plaintext_compact_bytes_by_slot,
                gate_id,
                lut_id,
            );
            let out = BggPolyEncoding::new(
                params.clone(),
                output_vector_bytes,
                out_pubkey,
                Some(output_plaintext_bytes),
            );
            debug!(
                "LWE BGG poly-encoding public lookup finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
                gate_id,
                lut_id,
                num_slots,
                public_lookup_started.elapsed().as_secs_f64() * 1000.0
            );
            return out;
        }

        #[cfg(not(feature = "gpu"))]
        {
            let output_by_slot = (0..num_slots)
                .into_par_iter()
                .map(|slot| {
                    let c_b = M::from_compact_bytes(
                        params,
                        self.c_b_compact_bytes_by_slot[slot].as_ref(),
                    );
                    let input_vector = input.vector_for_params(params, slot);
                    let x = M::P::from_compact_bytes(
                        params,
                        plaintext_compact_bytes_by_slot[slot].as_ref(),
                    );
                    let slot_output = public_lookup_slot_cpu::<M, SH>(
                        params,
                        plt,
                        &self.dir_path,
                        self.hash_key,
                        row_size,
                        &c_b,
                        &input_vector,
                        &x,
                        gate_id,
                        lut_id,
                        None,
                    );
                    (
                        Arc::<[u8]>::from(slot_output.vector.into_compact_bytes()),
                        Arc::<[u8]>::from(
                            slot_output
                                .plaintext
                                .expect("poly public lookup should reveal plaintexts")
                                .to_compact_bytes(),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            let (output_vector_bytes, output_plaintext_bytes): (Vec<_>, Vec<_>) =
                output_by_slot.into_iter().unzip();
            let out = BggPolyEncoding::new(
                params.clone(),
                output_vector_bytes,
                out_pubkey,
                Some(output_plaintext_bytes),
            );
            debug!(
                "LWE BGG poly-encoding public lookup finished: gate_id={}, lut_id={}, slot_count={}, elapsed_ms={:.3}",
                gate_id,
                lut_id,
                num_slots,
                public_lookup_started.elapsed().as_secs_f64() * 1000.0
            );
            out
        }
    }
}

impl<M, SH> PolyEncodingPublicLutBenchEstimator<M> for LWEBGGPolyEncodingPltEvaluator<M, SH>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static + Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn benchmark_public_lookup_chunk(
        &self,
        samples: &crate::bench_estimator::BggPolyEncodingBenchSamples<'_, M>,
        iterations: usize,
    ) -> PolyEncodingChunkBenchMeasurement {
        #[cfg(feature = "gpu")]
        {
            return gpu::benchmark_public_lookup_slot_gpu::<M, SH>(self, samples, iterations);
        }

        #[cfg(not(feature = "gpu"))]
        {
            let plaintext_compact_bytes_by_slot =
                samples.public_lut_input.plaintext_bytes.as_ref().expect(
                    "the BGG poly encoding should reveal plaintexts for public-lookup benchmarking",
                );
            let row_size = samples.public_lut_input.pubkey.matrix.row_size();
            let c_b =
                M::from_compact_bytes(samples.params, self.c_b_compact_bytes_by_slot[0].as_ref());
            let input_vector = M::from_compact_bytes(
                samples.params,
                samples.public_lut_input.vector_bytes[0].as_ref(),
            );
            let x = M::P::from_compact_bytes(
                samples.params,
                plaintext_compact_bytes_by_slot[0].as_ref(),
            );
            let chunk_count = k_high_chunk_count::<M>(samples.params, row_size);
            let dir = Path::new(&self.dir_path);
            let bench = benchmark_gate_operation(iterations, || {
                public_lookup_output_chunk_cpu::<M, SH>(
                    samples.params,
                    samples.public_lut,
                    dir,
                    self.hash_key,
                    row_size,
                    &c_b,
                    &input_vector,
                    &x,
                    samples.public_lut_gate_id,
                    samples.public_lut_id,
                    0,
                )
                .into_compact_bytes()
            });
            let latency = bench.time;
            let max_parallelism = chunk_count as u128;
            PolyEncodingChunkBenchMeasurement::new(
                latency * max_parallelism as f64,
                latency,
                max_parallelism,
                bench.peak_vram,
            )
        }
    }
}
