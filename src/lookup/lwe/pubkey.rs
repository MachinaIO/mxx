use crate::{
    bgg::public_key::BggPublicKey,
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler},
    storage::write::{BatchLookupBuffer, add_lookup_buffer, get_lookup_buffer},
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData, path::PathBuf, sync::Arc};
use tracing::info;

use super::{derive_a_lt_matrix, derive_k_low, k_high_checkpoint_prefix};

#[derive(Debug)]
struct GateState<M>
where
    M: PolyMatrix,
{
    lut_id: usize,
    input_pubkey_bytes: Vec<u8>,
    output_pubkey_bytes: Vec<u8>,
    _m: PhantomData<M>,
}

#[derive(Debug)]
pub struct LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trap_sampler: ST,
    pub pub_matrix: Arc<M>,
    pub trapdoor: Arc<ST::Trapdoor>,
    pub dir_path: PathBuf,
    lut_state: DashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
    gate_state: DashMap<GateId, GateState<M>>,
    _sh: PhantomData<SH>,
    _st: PhantomData<ST>,
}

impl<M, SH, ST> PltEvaluator<BggPublicKey<M>> for LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        _: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let row_size = input.matrix.row_size();
        let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id);
        self.lut_state.entry(lut_id).or_insert_with(|| plt.clone());
        self.gate_state.insert(
            gate_id,
            GateState {
                lut_id,
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                output_pubkey_bytes: a_lt.to_compact_bytes(),
                _m: PhantomData,
            },
        );
        BggPublicKey { matrix: a_lt, reveal_plaintext: true }
    }
}

impl<M, SH, ST> LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(
        hash_key: [u8; 32],
        trap_sampler: ST,
        pub_matrix: Arc<M>,
        trapdoor: Arc<ST::Trapdoor>,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            trap_sampler,
            pub_matrix,
            trapdoor,
            dir_path,
            lut_state: DashMap::new(),
            gate_state: DashMap::new(),
            _sh: PhantomData,
            _st: PhantomData,
        }
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        info!("Sampling LWE LUT auxiliary matrices");

        let lut_ids: Vec<usize> = self.lut_state.iter().map(|entry| *entry.key()).collect();
        let mut lut_entries: HashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>> =
            HashMap::with_capacity(lut_ids.len());
        for lut_id in lut_ids {
            if let Some((_, plt)) = self.lut_state.remove(&lut_id) {
                lut_entries.insert(lut_id, plt);
            }
        }

        let gate_ids: Vec<GateId> = self.gate_state.iter().map(|entry| *entry.key()).collect();
        let total_gates = gate_ids.len();
        let mut gate_entries = Vec::with_capacity(gate_ids.len());
        for gate_id in gate_ids {
            if let Some((_, state)) = self.gate_state.remove(&gate_id) {
                gate_entries.push((gate_id, state));
            }
        }

        if gate_entries.is_empty() {
            info!("No LWE gate auxiliary matrices to sample");
            return;
        }

        for (gate_id, gate_state) in gate_entries {
            let plt = lut_entries.get(&gate_state.lut_id).unwrap_or_else(|| {
                panic!(
                    "LUT state for lut_id {} not found while sampling gate {}",
                    gate_state.lut_id, gate_id
                )
            });
            let input_pubkey = M::from_compact_bytes(params, &gate_state.input_pubkey_bytes);
            let output_pubkey = M::from_compact_bytes(params, &gate_state.output_pubkey_bytes);
            let buffer = sample_k_high_buffer::<M, SH, ST, _>(
                plt,
                params,
                self.hash_key,
                &self.trap_sampler,
                &self.pub_matrix,
                &self.trapdoor,
                &input_pubkey,
                &output_pubkey,
                gate_id,
                gate_state.lut_id,
            );
            add_lookup_buffer(buffer);
        }

        info!("Sampled {} LWE gate auxiliary matrices", total_gates);
    }
}

fn sample_k_high_buffer<M, SH, ST, P>(
    plt: &PublicLut<P>,
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    trap_sampler: &ST,
    pub_matrix: &M,
    trapdoor: &ST::Trapdoor,
    a_z: &M,
    a_lt: &M,
    gate_id: GateId,
    lut_id: usize,
) -> BatchLookupBuffer
where
    P: Poly + 'static,
    M: PolyMatrix<P = P> + Send + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    let row_size = pub_matrix.row_size();
    let gadget = M::gadget_matrix(params, row_size);
    let chunk_size = crate::env::lut_preimage_chunk_size().max(1);
    info!(
        "start collecting LWE k_high matrices for gate {} (entries={}, parallel_batch={})",
        gate_id,
        plt.len(),
        chunk_size
    );
    let mut k_high_by_entry = Vec::with_capacity(plt.len());
    let mut batch = Vec::with_capacity(chunk_size);
    for (x_k, (k, y_k)) in plt.entries(params) {
        batch.push((x_k, k, y_k));
        if batch.len() >= chunk_size {
            let chunk = std::mem::take(&mut batch);
            let mut partial = chunk
                .into_par_iter()
                .map(|(x_k, k, y_k)| {
                    let x_k_usize = usize::try_from(x_k).expect("LUT input must fit in usize");
                    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
                    let x_poly = P::from_usize_to_constant(params, x_k_usize);
                    let y_poly = P::from_elem_to_constant(params, &y_k);
                    let ext_matrix = a_z.clone() - &(gadget.clone() * x_poly);
                    let target = a_lt.clone() - &(gadget.clone() * y_poly);
                    let k_low =
                        derive_k_low::<M, SH>(params, row_size, hash_key, gate_id, lut_id, k_usize);
                    let adjusted_target = target - &(ext_matrix * &k_low);
                    (k_usize, trap_sampler.preimage(params, trapdoor, pub_matrix, &adjusted_target))
                })
                .collect::<Vec<_>>();
            k_high_by_entry.append(&mut partial);
            batch = Vec::with_capacity(chunk_size);
        }
    }
    if !batch.is_empty() {
        let mut partial = batch
            .into_par_iter()
            .map(|(x_k, k, y_k)| {
                let x_k_usize = usize::try_from(x_k).expect("LUT input must fit in usize");
                let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
                let x_poly = P::from_usize_to_constant(params, x_k_usize);
                let y_poly = P::from_elem_to_constant(params, &y_k);
                let ext_matrix = a_z.clone() - &(gadget.clone() * x_poly);
                let target = a_lt.clone() - &(gadget.clone() * y_poly);
                let k_low =
                    derive_k_low::<M, SH>(params, row_size, hash_key, gate_id, lut_id, k_usize);
                let adjusted_target = target - &(ext_matrix * &k_low);
                (k_usize, trap_sampler.preimage(params, trapdoor, pub_matrix, &adjusted_target))
            })
            .collect::<Vec<_>>();
        k_high_by_entry.append(&mut partial);
    }
    info!("finish collecting LWE k_high matrices for gate {}", gate_id);
    get_lookup_buffer(k_high_by_entry, &k_high_checkpoint_prefix(gate_id, lut_id))
}
