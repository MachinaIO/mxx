#[cfg(feature = "gpu")]
#[path = "poly_encoding_gpu.rs"]
mod gpu;

use crate::{
    bgg::poly_encoding::BggPolyEncoding,
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::PolyHashSampler,
};
use std::{
    marker::PhantomData,
    ops::Mul,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::debug;

use super::encoding::{apply_public_lookup_to_slot, build_public_lookup_shared_state};

#[derive(Debug, Clone)]
pub struct GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub checkpoint_prefix: String,
    c_b0_compact_bytes_by_slot: Vec<Arc<[u8]>>,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        _params: &<M::P as Poly>::Params,
        secret_vec: M,
        b0_matrix: M,
        slot_secret_mats: Vec<M>,
    ) -> Self {
        let c_b0_compact_bytes_by_slot = slot_secret_mats
            .into_iter()
            .map(|slot_secret_mat| {
                let transformed_secret_vec = secret_vec.clone() * &slot_secret_mat;
                let c_b0 = transformed_secret_vec * &b0_matrix;
                Arc::<[u8]>::from(c_b0.into_compact_bytes())
            })
            .collect::<Vec<_>>();

        Self { hash_key, dir_path, checkpoint_prefix, c_b0_compact_bytes_by_slot, _hs: PhantomData }
    }
}

impl<M, HS> PltEvaluator<BggPolyEncoding<M>> for GGH15BGGPolyEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
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
            "GGH15BGGPolyEncodingPltEvaluator::public_lookup start: gate_id={}, lut_id={}, slots={}",
            gate_id, lut_id, num_slots
        );

        let plaintexts = input
            .plaintexts
            .as_ref()
            .expect("the BGG poly encoding should reveal plaintexts for public lookup");
        assert_eq!(
            plaintexts.len(),
            num_slots,
            "BGG poly encoding public lookup requires plaintexts.len() == num_slots"
        );
        assert_eq!(
            self.c_b0_compact_bytes_by_slot.len(),
            num_slots,
            "slot-wise c_b0 compact-bytes cache must match the BGG poly encoding slot count"
        );

        let plaintext_compact_bytes_by_slot = plaintexts
            .iter()
            .map(|plaintext| Arc::<[u8]>::from(plaintext.to_compact_bytes()))
            .collect::<Vec<_>>();
        let dir = Path::new(&self.dir_path);
        let d = input.pubkey.matrix.row_size();
        let checkpoint_prefix = &self.checkpoint_prefix;
        let shared = build_public_lookup_shared_state::<M, HS>(
            params,
            dir,
            checkpoint_prefix,
            self.hash_key,
            gate_id,
            d,
        );
        let out_pubkey = shared.out_pubkey.clone();
        let configured_parallelism =
            crate::env::bgg_poly_encoding_slot_parallelism().max(1).min(num_slots.max(1));

        #[cfg(feature = "gpu")]
        {
            let (output_vector_bytes, output_plaintext_bytes) =
                gpu::evaluate_public_lookup_slots_gpu::<M, HS>(
                    params,
                    plt,
                    dir,
                    checkpoint_prefix,
                    self.hash_key,
                    gate_id,
                    lut_id,
                    input,
                    &plaintext_compact_bytes_by_slot,
                    &self.c_b0_compact_bytes_by_slot,
                    &shared,
                    configured_parallelism,
                );
            let output_plaintexts = output_plaintext_bytes
                .iter()
                .map(|bytes| M::P::from_compact_bytes(params, bytes.as_ref()))
                .collect::<Vec<_>>();
            let out = BggPolyEncoding::from_vector_bytes(
                params.clone(),
                output_vector_bytes,
                out_pubkey,
                Some(output_plaintexts),
            );
            debug!(
                "GGH15BGGPolyEncodingPltEvaluator::public_lookup end: gate_id={}, lut_id={}, elapsed_ms={:.3}",
                gate_id,
                lut_id,
                public_lookup_started.elapsed().as_secs_f64() * 1000.0
            );
            return out;
        }

        #[cfg(not(feature = "gpu"))]
        {
            let mut output_vector_bytes = Vec::with_capacity(num_slots);
            let mut output_plaintext_bytes = Vec::with_capacity(num_slots);

            for slot_start in (0..num_slots).step_by(configured_parallelism) {
                let chunk_len = (slot_start + configured_parallelism).min(num_slots) - slot_start;
                let c_b0_chunk =
                    &self.c_b0_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
                let input_vector_chunk = &input.vector_bytes[slot_start..slot_start + chunk_len];
                let x_chunk = &plaintext_compact_bytes_by_slot[slot_start..slot_start + chunk_len];
                let mut chunk_outputs = (0..chunk_len)
                    .into_par_iter()
                    .map(|offset| {
                        let c_b0 = M::from_compact_bytes(params, c_b0_chunk[offset].as_ref());
                        let input_vector =
                            M::from_compact_bytes(params, input_vector_chunk[offset].as_ref());
                        let x = M::P::from_compact_bytes(params, x_chunk[offset].as_ref());
                        let slot_output = apply_public_lookup_to_slot::<M, HS>(
                            params,
                            plt,
                            dir,
                            checkpoint_prefix,
                            self.hash_key,
                            &c_b0,
                            &shared,
                            &input_vector,
                            &x,
                            gate_id,
                            lut_id,
                        );
                        drop(c_b0);
                        drop(input_vector);
                        drop(x);
                        (
                            Arc::<[u8]>::from(slot_output.vector.into_compact_bytes()),
                            Arc::<[u8]>::from(slot_output.plaintext.to_compact_bytes()),
                        )
                    })
                    .collect::<Vec<_>>();
                for (vector_bytes, plaintext_bytes) in chunk_outputs.drain(..) {
                    output_vector_bytes.push(vector_bytes);
                    output_plaintext_bytes.push(plaintext_bytes);
                }
            }

            let output_plaintexts = output_plaintext_bytes
                .iter()
                .map(|bytes| M::P::from_compact_bytes(params, bytes.as_ref()))
                .collect::<Vec<_>>();
            let out = BggPolyEncoding::from_vector_bytes(
                params.clone(),
                output_vector_bytes,
                out_pubkey,
                Some(output_plaintexts),
            );

            debug!(
                "GGH15BGGPolyEncodingPltEvaluator::public_lookup end: gate_id={}, lut_id={}, elapsed_ms={:.3}",
                gate_id,
                lut_id,
                public_lookup_started.elapsed().as_secs_f64() * 1000.0
            );
            out
        }
    }
}
