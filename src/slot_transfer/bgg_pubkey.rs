use crate::{
    bgg::public_key::BggPublicKey,
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::SlotTransferEvaluator,
    storage::{
        read::read_bytes_from_multi_batch,
        write::{
            add_lookup_buffer, get_lookup_buffer, get_lookup_buffer_bytes, init_storage_system,
        },
    },
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{marker::PhantomData, path::PathBuf};
use tracing::info;

pub(crate) struct SlotAuxSample<M: PolyMatrix> {
    pub(crate) slot_idx: usize,
    pub(crate) slot_a: M,
    pub(crate) slot_a_bytes: Vec<u8>,
    pub(crate) preimage_b0: M,
    pub(crate) preimage_b1: M,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPublicKeySTGateState {
    pub input_pubkey_bytes: Vec<u8>,
    pub src_slots: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub secret_size: usize,
    pub num_slots: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub dir_path: PathBuf,
    gate_states: DashMap<GateId, BggPublicKeySTGateState>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        hash_key: [u8; 32],
        secret_size: usize,
        num_slots: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            secret_size,
            num_slots,
            trapdoor_sigma,
            error_sigma,
            dir_path,
            gate_states: DashMap::new(),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    pub fn gate_state(&self, gate_id: GateId) -> Option<BggPublicKeySTGateState> {
        self.gate_states.get(&gate_id).map(|state| state.value().clone())
    }

    pub(crate) fn sample_error_matrix(
        &self,
        params: &<M::P as Poly>::Params,
        nrow: usize,
        ncol: usize,
    ) -> M {
        assert!(self.error_sigma >= 0.0, "error_sigma {} must be nonnegative", self.error_sigma);
        if self.error_sigma == 0.0 {
            M::zero(params, nrow, ncol)
        } else {
            US::new().sample_uniform(
                params,
                nrow,
                ncol,
                DistType::GaussDist { sigma: self.error_sigma },
            )
        }
    }

    fn aux_checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        let (_, crt_bits, crt_depth) = params.to_crt();
        format!(
            "slot_transfer_aux_d{}_slots{}_crtbits{}_crtdepth{}_ring{}_base{}_sigma{:.6}_err{:.6}_key{}",
            self.secret_size,
            self.num_slots,
            crt_bits,
            crt_depth,
            params.ring_dimension(),
            params.base_bits(),
            self.trapdoor_sigma,
            self.error_sigma,
            self.hash_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        )
    }

    fn b0_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b0", self.aux_checkpoint_prefix(params))
    }

    fn b0_trapdoor_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b0_trapdoor", self.aux_checkpoint_prefix(params))
    }

    fn b1_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b1", self.aux_checkpoint_prefix(params))
    }

    fn b1_trapdoor_id_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        format!("{}_b1_trapdoor", self.aux_checkpoint_prefix(params))
    }

    fn slot_a_id_prefix(&self, params: &<M::P as Poly>::Params, slot_idx: usize) -> String {
        format!("{}_slot_a_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    fn slot_preimage_b0_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        slot_idx: usize,
    ) -> String {
        format!("{}_slot_preimage_b0_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    fn slot_preimage_b1_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        slot_idx: usize,
    ) -> String {
        format!("{}_slot_preimage_b1_{}", self.aux_checkpoint_prefix(params), slot_idx)
    }

    fn gate_preimage_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        dst_slot: usize,
    ) -> String {
        format!("{}_gate_preimage_{}_dst{}", self.aux_checkpoint_prefix(params), gate_id, dst_slot)
    }

    fn store_matrix_checkpoint(matrix: M, id_prefix: &str) {
        add_lookup_buffer(get_lookup_buffer(vec![(0usize, matrix)], id_prefix));
    }

    fn store_bytes_checkpoint(bytes: Vec<u8>, id_prefix: &str) {
        add_lookup_buffer(get_lookup_buffer_bytes(vec![(0usize, bytes)], id_prefix));
    }

    fn load_matrix_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        id_prefix: &str,
    ) -> Option<M> {
        let bytes = read_bytes_from_multi_batch(self.dir_path.as_path(), id_prefix, 0)?;
        Some(M::from_compact_bytes(params, &bytes))
    }

    pub fn checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        self.aux_checkpoint_prefix(params)
    }

    pub fn load_b0_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        self.load_matrix_checkpoint(params, &self.b0_id_prefix(params))
    }

    pub fn load_b1_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        self.load_matrix_checkpoint(params, &self.b1_id_prefix(params))
    }

    #[cfg(not(feature = "gpu"))]
    fn sample_slot_batch_cpu(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
        identity: &M,
        gadget_matrix: &M,
        slot_secret_mats: &[Vec<u8>],
        batch_slot_indices: &[usize],
    ) -> Vec<SlotAuxSample<M>> {
        let m_g = self.secret_size * params.modulus_digits();
        let b1_size = self.secret_size * 2;
        batch_slot_indices
            .par_iter()
            .copied()
            .map(|slot_idx| {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let slot_secret_mat =
                    M::from_compact_bytes(params, slot_secret_mats[slot_idx].as_ref());
                assert_eq!(
                    slot_secret_mat.row_size(),
                    self.secret_size,
                    "slot secret matrix {} row size {} must match secret_size {}",
                    slot_idx,
                    slot_secret_mat.row_size(),
                    self.secret_size
                );
                assert_eq!(
                    slot_secret_mat.col_size(),
                    self.secret_size,
                    "slot secret matrix {} col size {} must match secret_size {}",
                    slot_idx,
                    slot_secret_mat.col_size(),
                    self.secret_size
                );
                let a_i = HS::new().sample_hash(
                    params,
                    self.hash_key,
                    format!("slot_transfer_slot_a_{}", slot_idx),
                    self.secret_size,
                    m_g,
                    DistType::FinRingDist,
                );
                let s_concat_identity = slot_secret_mat.clone().concat_columns(&[identity]);
                let mut target_b0 = s_concat_identity * b1_matrix;
                target_b0.add_in_place(&self.sample_error_matrix(
                    params,
                    self.secret_size,
                    b1_matrix.col_size(),
                ));
                let preimage_b0 = trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target_b0);

                let neg_slot_secret_gadget = -(slot_secret_mat * gadget_matrix);
                let mut target_b1 = a_i.clone().concat_rows(&[&neg_slot_secret_gadget]);
                target_b1.add_in_place(&self.sample_error_matrix(params, b1_size, m_g));
                let preimage_b1 = trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &target_b1);
                let slot_a_bytes = a_i.to_compact_bytes();
                SlotAuxSample { slot_idx, slot_a: a_i, slot_a_bytes, preimage_b0, preimage_b1 }
            })
            .collect()
    }

    #[cfg(not(feature = "gpu"))]
    fn sample_gate_batch_cpu(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        slot_secret_mats: &[Vec<u8>],
        slot_a_bytes_by_slot: &[Vec<u8>],
        gate_id: GateId,
        state: &BggPublicKeySTGateState,
        slot_chunk: &[(usize, u32)],
    ) -> Vec<(usize, M)> {
        let m_g = self.secret_size * params.modulus_digits();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let a_in = M::from_compact_bytes(params, &state.input_pubkey_bytes);
        let a_out = HS::new().sample_hash(
            params,
            self.hash_key,
            format!("slot_transfer_gate_a_out_{}", gate_id),
            self.secret_size,
            m_g,
            DistType::FinRingDist,
        );
        slot_chunk
            .par_iter()
            .copied()
            .map(|(dst_slot, src_slot_u32)| {
                let src_slot =
                    usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
                assert!(
                    src_slot < self.num_slots,
                    "source slot index {} out of range for num_slots {}",
                    src_slot,
                    self.num_slots
                );
                let s_j = M::from_compact_bytes(params, slot_secret_mats[dst_slot].as_ref());
                let s_i = M::from_compact_bytes(params, slot_secret_mats[src_slot].as_ref());
                let a_j = M::from_compact_bytes(params, slot_a_bytes_by_slot[dst_slot].as_ref());
                let lhs = s_j * &a_out;
                let rhs = (s_i * &a_in) * a_j.decompose();
                let mut target = lhs - rhs;
                target.add_in_place(&self.sample_error_matrix(params, self.secret_size, m_g));
                let preimage = trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &target);
                (dst_slot, preimage)
            })
            .collect()
    }

    pub fn sample_aux_matrices(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        slot_secret_mats: Vec<Vec<u8>>,
    ) {
        assert_eq!(
            self.secret_size,
            b0_matrix.row_size(),
            "b0 matrix rows {} must match evaluator secret_size {}",
            b0_matrix.row_size(),
            self.secret_size
        );
        assert_eq!(
            slot_secret_mats.len(),
            self.num_slots,
            "slot_secret_mats length {} must match evaluator num_slots {}",
            slot_secret_mats.len(),
            self.num_slots
        );

        info!(
            "Sampling slot-transfer auxiliary matrices (secret_size={}, num_slots={})",
            self.secret_size, self.num_slots
        );
        init_storage_system(self.dir_path.clone());

        Self::store_matrix_checkpoint(b0_matrix.clone(), &self.b0_id_prefix(params));
        Self::store_bytes_checkpoint(
            TS::trapdoor_to_bytes(b0_trapdoor),
            &self.b0_trapdoor_id_prefix(params),
        );

        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let b1_size = self.secret_size.checked_mul(2).expect("slot-transfer b1 size overflow");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        Self::store_matrix_checkpoint(b1_matrix.clone(), &self.b1_id_prefix(params));
        Self::store_bytes_checkpoint(
            TS::trapdoor_to_bytes(&b1_trapdoor),
            &self.b1_trapdoor_id_prefix(params),
        );

        let slot_parallelism =
            crate::env::slot_transfer_slot_parallelism().max(1).min(self.num_slots.max(1));
        info!(
            "Slot-transfer slot auxiliary parallelism: SLOT_TRANSFER_SLOT_PARALLELISM={}",
            slot_parallelism
        );
        let mut slot_a_bytes_by_slot = vec![Vec::new(); self.num_slots];
        #[cfg(feature = "gpu")]
        let gpu_slot_shared = self.prepare_gpu_device_shared(
            params,
            b0_matrix,
            b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            slot_parallelism,
        );
        #[cfg(feature = "gpu")]
        let gpu_slot_secret_mats =
            self.copy_compact_matrices_to_gpu_devices(&slot_secret_mats, &gpu_slot_shared);
        #[cfg(not(feature = "gpu"))]
        let identity = M::identity(params, self.secret_size, None);
        #[cfg(not(feature = "gpu"))]
        let gadget_matrix = M::gadget_matrix(params, self.secret_size);

        for batch in (0..self.num_slots).collect::<Vec<_>>().chunks(slot_parallelism.max(1)) {
            #[cfg(feature = "gpu")]
            let sampled =
                self.sample_slot_batch_gpu(&gpu_slot_shared, &gpu_slot_secret_mats, batch);
            #[cfg(not(feature = "gpu"))]
            let sampled = self.sample_slot_batch_cpu(
                params,
                b0_matrix,
                b0_trapdoor,
                &b1_matrix,
                &b1_trapdoor,
                &identity,
                &gadget_matrix,
                &slot_secret_mats,
                batch,
            );
            let sampled_slot_jobs = sampled
                .into_par_iter()
                .map(|sample| {
                    let slot_idx = sample.slot_idx;
                    let slot_a_id_prefix = self.slot_a_id_prefix(params, slot_idx);
                    let slot_preimage_b0_id_prefix =
                        self.slot_preimage_b0_id_prefix(params, slot_idx);
                    let slot_preimage_b1_id_prefix =
                        self.slot_preimage_b1_id_prefix(params, slot_idx);
                    (
                        slot_idx,
                        sample.slot_a_bytes,
                        vec![
                            (slot_a_id_prefix, sample.slot_a),
                            (slot_preimage_b0_id_prefix, sample.preimage_b0),
                            (slot_preimage_b1_id_prefix, sample.preimage_b1),
                        ],
                    )
                })
                .collect::<Vec<_>>();
            let sampled_slot_bytes = sampled_slot_jobs
                .par_iter()
                .map(|(slot_idx, slot_a_bytes, _)| (*slot_idx, slot_a_bytes.clone()))
                .collect::<Vec<_>>();
            for (slot_idx, slot_a_bytes) in sampled_slot_bytes {
                slot_a_bytes_by_slot[slot_idx] = slot_a_bytes;
            }
            let mut slot_job_iter =
                sampled_slot_jobs.into_iter().flat_map(|(_, _, jobs)| jobs.into_iter());
            loop {
                let slot_job_chunk =
                    slot_job_iter.by_ref().take(slot_parallelism.max(1)).collect::<Vec<_>>();
                if slot_job_chunk.is_empty() {
                    break;
                }
                slot_job_chunk.into_par_iter().for_each(|(id_prefix, matrix)| {
                    Self::store_matrix_checkpoint(matrix, &id_prefix)
                });
            }
        }

        let gate_ids: Vec<GateId> = self.gate_states.iter().map(|entry| *entry.key()).collect();
        let gate_entries = gate_ids
            .into_par_iter()
            .filter_map(|gate_id| {
                self.gate_states.remove(&gate_id).map(|(_, state)| (gate_id, state))
            })
            .collect::<Vec<_>>();

        if gate_entries.is_empty() {
            info!("No slot-transfer gate auxiliary matrices to sample");
            return;
        }

        info!(
            "Slot-transfer gate auxiliary parallelism cap: SLOT_TRANSFER_SLOT_PARALLELISM={}",
            slot_parallelism
        );
        #[cfg(feature = "gpu")]
        let gpu_gate_shared = self.prepare_gpu_device_shared(
            params,
            b0_matrix,
            b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            slot_parallelism,
        );

        for (gate_id, state) in &gate_entries {
            let gate_slot_entries = state.src_slots.iter().copied().enumerate().collect::<Vec<_>>();
            let gate_parallelism = slot_parallelism.min(gate_slot_entries.len().max(1));
            info!("Slot-transfer gate {} effective parallelism: {}", gate_id, gate_parallelism);
            for sampled_chunk in gate_slot_entries.chunks(gate_parallelism) {
                #[cfg(feature = "gpu")]
                let selected_slot_indices = {
                    let mut indices = sampled_chunk
                        .iter()
                        .flat_map(|(dst_slot, src_slot_u32)| {
                            let src_slot = usize::try_from(*src_slot_u32)
                                .expect("source slot index must fit in usize");
                            [*dst_slot, src_slot]
                        })
                        .collect::<Vec<_>>();
                    indices.sort_unstable();
                    indices.dedup();
                    indices
                };
                #[cfg(feature = "gpu")]
                let selected_slot_secret_mats = selected_slot_indices
                    .iter()
                    .map(|slot_idx| slot_secret_mats[*slot_idx].clone())
                    .collect::<Vec<_>>();
                #[cfg(feature = "gpu")]
                let gpu_gate_slot_secret_mats = self.copy_compact_matrices_to_gpu_devices(
                    &selected_slot_secret_mats,
                    &gpu_gate_shared,
                );
                #[cfg(feature = "gpu")]
                let selected_slot_a_mats = selected_slot_indices
                    .iter()
                    .map(|slot_idx| slot_a_bytes_by_slot[*slot_idx].clone())
                    .collect::<Vec<_>>();
                #[cfg(feature = "gpu")]
                let gpu_gate_slot_a_mats = self
                    .copy_compact_matrices_to_gpu_devices(&selected_slot_a_mats, &gpu_gate_shared);
                #[cfg(feature = "gpu")]
                let sampled_chunk = self.sample_gate_batch_gpu(
                    &gpu_gate_shared,
                    &gpu_gate_slot_secret_mats,
                    &gpu_gate_slot_a_mats,
                    &selected_slot_indices,
                    *gate_id,
                    state,
                    sampled_chunk,
                );
                #[cfg(not(feature = "gpu"))]
                let sampled_chunk = self.sample_gate_batch_cpu(
                    params,
                    b0_matrix,
                    b0_trapdoor,
                    &slot_secret_mats,
                    &slot_a_bytes_by_slot,
                    *gate_id,
                    state,
                    sampled_chunk,
                );
                let gate_jobs = sampled_chunk
                    .into_par_iter()
                    .map(|(dst_slot, preimage)| {
                        (self.gate_preimage_id_prefix(params, *gate_id, dst_slot), preimage)
                    })
                    .collect::<Vec<_>>();
                gate_jobs.into_par_iter().for_each(|(id_prefix, preimage)| {
                    Self::store_matrix_checkpoint(preimage, &id_prefix);
                });
            }
        }
    }
}

impl<M, US, HS, TS> SlotTransferEvaluator<BggPublicKey<M>>
    for BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn slot_transfer(
        &self,
        params: &<M::P as Poly>::Params,
        input: &BggPublicKey<M>,
        src_slots: &[u32],
        gate_id: GateId,
    ) -> BggPublicKey<M> {
        assert_eq!(
            src_slots.len(),
            self.num_slots,
            "source slot count {} does not match evaluator num_slots {}",
            src_slots.len(),
            self.num_slots
        );
        assert_eq!(
            input.matrix.row_size(),
            self.secret_size,
            "input pubkey rows {} must match evaluator secret_size {}",
            input.matrix.row_size(),
            self.secret_size
        );
        assert_eq!(
            input.matrix.col_size(),
            self.secret_size * params.modulus_digits(),
            "input pubkey columns {} must equal secret_size * modulus_digits {}",
            input.matrix.col_size(),
            self.secret_size * params.modulus_digits()
        );

        self.gate_states.insert(
            gate_id,
            BggPublicKeySTGateState {
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                src_slots: src_slots.to_vec(),
            },
        );

        let hash_sampler = HS::new();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("slot_transfer_gate_a_out_{}", gate_id),
            self.secret_size,
            self.secret_size * params.modulus_digits(),
            DistType::FinRingDist,
        );
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[cfg(test)]
mod tests {
    use super::BggPublicKeySTEvaluator;
    use crate::{
        __PAIR, __TestState,
        bgg::public_key::BggPublicKey,
        circuit::PolyCircuit,
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{
            DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::{
            read::read_matrix_from_multi_batch,
            write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        },
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

    const SIGMA: f64 = 4.578;

    struct DummyPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::Params,
            _plt: &PublicLut<
                <BggPublicKey<DCRTPolyMatrix> as crate::circuit::evaluable::Evaluable>::P,
            >,
            _one: &BggPublicKey<DCRTPolyMatrix>,
            _input: &BggPublicKey<DCRTPolyMatrix>,
            _gate_id: crate::circuit::gate::GateId,
            _lut_id: usize,
        ) -> BggPublicKey<DCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer tests")
        }
    }

    #[test]
    fn bgg_public_key_st_evaluator_records_gate_state_and_hashes_output_matrix() {
        let params = DCRTPolyParams::default();
        let hash_key = [0x42u8; 32];
        let m_g = 2 * params.modulus_digits();
        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_test_input".to_string(),
                2,
                m_g,
                DistType::FinRingDist,
            ),
            false,
        );
        let one = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_test_one".to_string(),
                2,
                m_g,
                DistType::FinRingDist,
            ),
            true,
        );

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[1, 0]);
        circuit.output(vec![transferred]);

        let evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(
                hash_key, 2, 2, SIGMA, 0.0, "test_data/test_slot_transfer_gate_state".into()
            );
        let result = circuit.eval(
            &params,
            one,
            vec![input_pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            None,
        );

        assert_eq!(result.len(), 1);
        let expected_matrix = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred),
            input_pubkey.matrix.row_size(),
            input_pubkey.matrix.row_size() * params.modulus_digits(),
            DistType::FinRingDist,
        );
        assert_eq!(result[0], BggPublicKey::new(expected_matrix, true));

        let stored = evaluator.gate_state(transferred).expect("missing stored gate state");
        assert_eq!(stored.input_pubkey_bytes, input_pubkey.matrix.to_compact_bytes());
        assert_eq!(stored.src_slots, vec![1, 0]);
    }

    #[test]
    #[should_panic(expected = "source slot count 1 does not match evaluator num_slots 2")]
    fn bgg_public_key_st_evaluator_rejects_unexpected_slot_count() {
        let params = DCRTPolyParams::default();
        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                [0x24u8; 32],
                "slot_transfer_bad_slots_input".to_string(),
                2,
                2 * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );
        let evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(
                [0x24u8; 32], 2, 2, SIGMA, 0.0, "test_data/test_slot_transfer_bad_slots".into()
            );

        let _ = <BggPublicKeySTEvaluator<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        > as crate::slot_transfer::SlotTransferEvaluator<BggPublicKey<DCRTPolyMatrix>>>::slot_transfer(
            &evaluator,
            &params,
            &input_pubkey,
            &[0],
            crate::circuit::gate::GateId(9),
        );
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn bgg_public_key_st_evaluator_samples_and_persists_aux_matrices() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let hash_key = [0x31u8; 32];
        let secret_size = 2usize;
        let num_slots = 2usize;
        let dir_path = "test_data/test_slot_transfer_aux";
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());

        let input_pubkey = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_aux_input".to_string(),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );
        let one = BggPublicKey::new(
            DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_aux_one".to_string(),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            ),
            true,
        );

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[1, 0]);
        circuit.output(vec![transferred]);

        let evaluator =
            BggPublicKeySTEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(hash_key, secret_size, num_slots, SIGMA, 0.0, dir.to_path_buf());
        let result = circuit.eval(
            &params,
            one,
            vec![input_pubkey.clone()],
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            None,
        );
        assert_eq!(result.len(), 1);

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, secret_size);

        let slot_secret_mats = (0..num_slots)
            .map(|_| {
                DCRTPolyUniformSampler::new()
                    .sample_uniform(&params, secret_size, secret_size, DistType::TernaryDist)
                    .into_compact_bytes()
            })
            .collect::<Vec<_>>();

        evaluator.sample_aux_matrices(&params, &b0_matrix, &b0_trapdoor, slot_secret_mats.clone());
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();

        let checkpoint_prefix = evaluator.checkpoint_prefix(&params);
        let b1_matrix = evaluator
            .load_b1_matrix_checkpoint(&params)
            .expect("b1 matrix checkpoint should exist after sample_aux_matrices");
        let loaded_b0 = evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
        assert_eq!(loaded_b0, b0_matrix);

        let identity = DCRTPolyMatrix::identity(&params, secret_size, None);
        let gadget_matrix = DCRTPolyMatrix::gadget_matrix(&params, secret_size);
        let m_g = secret_size * params.modulus_digits();

        for slot_idx in 0..num_slots {
            let s_i =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[slot_idx].as_ref());
            let a_i = read_matrix_from_multi_batch::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_slot_a_{slot_idx}"),
                0,
            )
            .expect("slot A_i checkpoint should exist");
            let expected_a_i = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_slot_a_{}", slot_idx),
                secret_size,
                m_g,
                DistType::FinRingDist,
            );
            assert_eq!(a_i, expected_a_i);

            let slot_preimage_b0 = read_matrix_from_multi_batch::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_slot_preimage_b0_{slot_idx}"),
                0,
            )
            .expect("slot preimage in b0 basis should exist");
            let slot_preimage_b1 = read_matrix_from_multi_batch::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_slot_preimage_b1_{slot_idx}"),
                0,
            )
            .expect("slot preimage in b1 basis should exist");

            let expected_target_b0 = s_i.clone().concat_columns(&[&identity]) * &b1_matrix;
            let neg_slot_secret_gadget = -(s_i.clone() * &gadget_matrix);
            let expected_target_b1 = a_i.clone().concat_rows(&[&neg_slot_secret_gadget]);
            assert_eq!(b0_matrix.clone() * &slot_preimage_b0, expected_target_b0);
            assert_eq!(b1_matrix.clone() * &slot_preimage_b1, expected_target_b1);
        }

        let a_out = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &params,
            hash_key,
            format!("slot_transfer_gate_a_out_{}", transferred),
            secret_size,
            m_g,
            DistType::FinRingDist,
        );

        for (dst_slot, src_slot) in [1usize, 0usize].into_iter().enumerate() {
            let s_j =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[dst_slot].as_ref());
            let s_i =
                DCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[src_slot].as_ref());
            let a_j = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_slot_a_{}", dst_slot),
                secret_size,
                m_g,
                DistType::FinRingDist,
            );
            let gate_preimage = read_matrix_from_multi_batch::<DCRTPolyMatrix>(
                &params,
                dir,
                &format!("{checkpoint_prefix}_gate_preimage_{}_dst{}", transferred, dst_slot),
                0,
            )
            .expect("gate preimage checkpoint should exist");

            let expected_target = s_j * &a_out - (s_i * &input_pubkey.matrix) * a_j.decompose();
            assert_eq!(b0_matrix.clone() * &gate_preimage, expected_target);
        }
    }
}
