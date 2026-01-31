use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{PolyCircuit, PolyGateType, evaluable::Evaluable, gate::GateId},
    commit::wee25::Wee25Commit,
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::write::{add_lookup_buffer, get_lookup_buffer},
    utils::mod_inverse_mod_q,
};
use dashmap::DashMap;
use std::{collections::HashMap, marker::PhantomData};

#[derive(Debug, Clone)]
struct GateState<M: PolyMatrix> {
    gate_id: GateId,
    lut_id: usize,
    input_pubkey: BggPublicKey<M>,
}

#[derive(Debug, Clone)]
pub struct CommitBGGPubKeyPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub wee25_commit: Wee25Commit<M>,
    pub lut_gate_start_ids: HashMap<GateId, usize>,
    pub lut_vector_len: usize,
    pub b_1: M,
    gate_states: DashMap<GateId, GateState<M>>,
    luts: DashMap<usize, PublicLut<M::P>>,
    _hs: PhantomData<HS>,
}

impl<M, HS> CommitBGGPubKeyPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub fn setup<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        trapdoor_sigma: f64,
        tree_base: usize,
        hash_key: [u8; 32],
        circuit: &PolyCircuit<M::P>,
    ) -> Self {
        let wee25_commit =
            Wee25Commit::<M>::setup::<US, TS>(params, secret_size, trapdoor_sigma, tree_base);
        let (lut_gate_start_ids, lut_vector_len) = map_lut_gate_start_ids(circuit);
        let hash_sampler = HS::new();
        let b_1 = hash_sampler.sample_hash(
            params,
            hash_key,
            b"COMMIT_LUT_B1".to_vec(),
            secret_size,
            wee25_commit.m_b,
            DistType::FinRingDist,
        );
        Self {
            hash_key,
            wee25_commit,
            lut_gate_start_ids,
            lut_vector_len,
            b_1,
            gate_states: DashMap::new(),
            luts: DashMap::new(),
            _hs: PhantomData,
        }
    }

    pub fn commits_all_lut_matrices(&self, params: &<M::P as Poly>::Params) {
        let lut_ids: Vec<usize> = self.luts.iter().map(|entry| *entry.key()).collect();
        let mut luts = HashMap::new();
        for lut_id in lut_ids {
            if let Some((_, plt)) = self.luts.remove(&lut_id) {
                luts.insert(lut_id, plt);
            }
        }
        let gate_ids: Vec<GateId> = self.gate_states.iter().map(|entry| *entry.key()).collect();
        let mut gate_states = Vec::with_capacity(gate_ids.len());
        for gate_id in gate_ids {
            if let Some((_, state)) = self.gate_states.remove(&gate_id) {
                gate_states.push((state.gate_id, state.lut_id, state.input_pubkey));
            }
        }
        let committed_matrix = build_committed_matrix::<M, HS>(
            params,
            &luts,
            &gate_states,
            self.hash_key,
            &self.wee25_commit,
            &self.lut_gate_start_ids,
            self.lut_vector_len,
            &self.b_1,
        );
        let commit = self.wee25_commit.commit(params, &committed_matrix);
        add_lookup_buffer(get_lookup_buffer(vec![(0, commit)], &format!("commit")));
    }
}

impl<M, HS> PltEvaluator<BggPublicKey<M>> for CommitBGGPubKeyPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<M::P as Poly>::Params,
        plt: &PublicLut<M::P>,
        _: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        self.luts.entry(lut_id).or_insert_with(|| plt.clone());
        let row_size = input.matrix.row_size();
        self.gate_states
            .insert(gate_id, GateState { gate_id, lut_id, input_pubkey: input.clone() });
        let a_out = derive_a_out_matrix::<M, HS>(params, row_size, self.hash_key, gate_id);
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

// #[derive(Debug, Clone)]
// pub struct CommitBGGEncodingPltEvaluator<M, HS>
// where
//     M: PolyMatrix,
//     M::P: 'static,
//     HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
// {
//     pub hash_key: [u8; 32],
//     pub wee25_commit: Wee25Commit<M>,
//     pub lut_gate_start_ids: HashMap<GateId, usize>,
//     pub lut_vector_len: usize,
//     pub b_1: M,
//     pub c_0: M,
//     pub c_1: M,
//     pub committed_matrix: M,
//     _hs: PhantomData<HS>,
// }

// impl<M, HS> CommitBGGEncodingPltEvaluator<M, HS>
// where
//     M: PolyMatrix,
//     HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
// {
//     pub fn setup<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
//         params: &<M::P as Poly>::Params,
//         trapdoor_sigma: f64,
//         error_sigma: f64,
//         tree_base: usize,
//         hash_key: [u8; 32],
//         circuit: &PolyCircuit<M::P>,
//         luts: &HashMap<usize, PublicLut<M::P>>,
//         gate_states: &[(GateId, usize, BggPublicKey<M>)],
//         secret: &M,
//     ) -> Self {
//         let secret_size = secret.col_size();
//         let wee25_commit =
//             Wee25Commit::<M>::setup::<US, TS>(params, secret_size, trapdoor_sigma, tree_base);
//         let (lut_gate_start_ids, lut_vector_len) = map_lut_gate_start_ids(circuit);
//         let error_sampler = US::new();
//         let b_1 = {
//             let hash_sampler = HS::new();
//             hash_sampler.sample_hash(
//                 params,
//                 hash_key,
//                 b"COMMIT_LUT_B1".to_vec(),
//                 secret_size,
//                 wee25_commit.m_b,
//                 DistType::FinRingDist,
//             )
//         };
//         let committed_matrix = build_committed_matrix::<M, HS>(
//             params,
//             luts,
//             gate_states,
//             hash_key,
//             &wee25_commit,
//             &lut_gate_start_ids,
//             lut_vector_len,
//             &b_1,
//         );
//         let commit = wee25_commit.commit(params, &committed_matrix);
//         let c_0 = {
//             let error = error_sampler.sample_uniform(
//                 params,
//                 secret.row_size(),
//                 commit.col_size(),
//                 DistType::GaussDist { sigma: error_sigma },
//             );
//             secret.clone() * (commit + &b_1) + error
//         };
//         let c_1 = {
//             let error = error_sampler.sample_uniform(
//                 params,
//                 secret.row_size(),
//                 wee25_commit.m_b,
//                 DistType::GaussDist { sigma: error_sigma },
//             );
//             secret.clone() * wee25_commit.b.clone() + error
//         };
//         Self {
//             hash_key,
//             wee25_commit,
//             lut_gate_start_ids,
//             lut_vector_len,
//             b_1,
//             c_0,
//             c_1,
//             committed_matrix,
//             _hs: PhantomData,
//         }
//     }
// }

// impl<M, HS> PltEvaluator<BggEncoding<M>> for CommitBGGEncodingPltEvaluator<M, HS>
// where
//     M: PolyMatrix,
//     HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
// {
//     fn public_lookup(
//         &self,
//         params: &<M::P as Poly>::Params,
//         plt: &PublicLut<M::P>,
//         _: &BggEncoding<M>,
//         input: &BggEncoding<M>,
//         gate_id: GateId,
//         lut_id: usize,
//     ) -> BggEncoding<M> {
//         let x = input
//             .plaintext
//             .as_ref()
//             .expect("the BGG encoding should reveal plaintext for public lookup");
//         let (k, y) = plt.get(params, x).unwrap_or_else(|| {
//             panic!("{:?} not found in LUT for gate {}", x.to_const_int(), gate_id)
//         });
//         let lut_vector_idx = lut_gate_index(&self.lut_gate_start_ids, gate_id, k);
//         let col_range = Some(lut_vector_idx..lut_vector_idx + 1);
//         let verifier = self.wee25_commit.verifier(self.lut_vector_len, col_range);
//         let secret_size = input.pubkey.matrix.row_size();
//         let m_g = self.wee25_commit.m_g;
//         let m_b = self.wee25_commit.m_b;
//         let r_g_i =
//             derive_r_g_i_matrix::<M, HS>(params, secret_size, m_b, self.hash_key, gate_id, k);
//         let reconst_coeffs = params.reconst_coeffs();
//         let idx_inverse = mod_inverse_mod_q::<M::P>(k as u64, params, &reconst_coeffs).unwrap();
//         let canceler = (self.b_1.clone() * verifier + &r_g_i) *
//             M::P::from_biguint_to_constant(params, idx_inverse);
//         // let a_out = derive_a_out_matrix::<M, HS>(params, secret_size, self.hash_key, gate_id);
//         // let gadget = M::gadget_matrix(params, secret_size);
//         // let padded = (a_out.clone() - gadget.clone() * y).concat_columns(&[&M::zero(
//         //     params,
//         //     secret_size,
//         //     m_b - m_g,
//         // )]);
//         // let msg = padded + &r_g_i - input.pubkey.matrix.clone() * canceler.decompose();
//         // let opening =
//         // let opening = self.wee25_commit.open(params, msg, col_range)
//         // let row_size = input.pubkey.matrix.row_size();
//         // let a_out = derive_a_out_matrix::<M, HS>(params, row_size, self.hash_key, gate_id);
//         // let gadget = M::gadget_matrix(params, row_size);
//         // let c_b0_times_plaintext = if let Some(plaintext) = &input.plaintext {
//         //     self.c_b0.clone() * plaintext.clone()
//         // } else {
//         //     M::zero(params, row_size, self.c_b0.col_size())
//         // };
//         // BggEncoding {
//         //     pubkey: BggPublicKey { matrix: a_out - c_b0_times_plaintext, reveal_plaintext:
// true         // },     plaintext: input.plaintext.clone(),
//         //     vector: M::zero(params, row_size, 1),
//         // }
//         ()
//     }
// }

fn build_committed_matrix<M, HS>(
    params: &<M::P as Poly>::Params,
    luts: &HashMap<usize, PublicLut<M::P>>,
    gate_states: &[(GateId, usize, BggPublicKey<M>)],
    hash_key: [u8; 32],
    wee25_commit: &Wee25Commit<M>,
    lut_gate_start_ids: &HashMap<GateId, usize>,
    lut_vector_len: usize,
    b_1: &M,
) -> M
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let secret_size = gate_states[0].2.matrix.row_size();
    let mut committed_matrices = vec![];
    let reconst_coeffs = params.reconst_coeffs();
    let gadget = M::gadget_matrix(params, secret_size);
    let verifier = wee25_commit.verifier(lut_vector_len, None);
    let m_g = wee25_commit.m_g;
    let m_b = wee25_commit.m_b;
    for (gate_id, lut_id, input_pubkey) in gate_states.iter() {
        let a_out = derive_a_out_matrix::<M, HS>(params, secret_size, hash_key, *gate_id);
        let lut = luts.get(lut_id).unwrap();
        let start_idx = lut_gate_start_ids[gate_id];
        for (_, (idx, y_poly)) in lut.entries(params) {
            let r_g_i =
                derive_r_g_i_matrix::<M, HS>(params, secret_size, m_b, hash_key, *gate_id, idx);
            let idx_inverse =
                mod_inverse_mod_q::<M::P>(idx as u64, params, &reconst_coeffs).unwrap();
            let verifier =
                verifier.slice_columns(m_b * (start_idx + idx), m_b * (start_idx + idx + 1));
            let canceler = (b_1.clone() * verifier + &r_g_i) *
                M::P::from_biguint_to_constant(params, idx_inverse);
            let padded = (a_out.clone() - gadget.clone() * y_poly).concat_columns(&[&M::zero(
                params,
                secret_size,
                m_b - m_g,
            )]);
            let msg = padded + &r_g_i - input_pubkey.matrix.clone() * canceler.decompose();
            committed_matrices.push(msg);
        }
    }
    committed_matrices[0].concat_columns(&committed_matrices[1..].iter().collect::<Vec<_>>())
}

fn map_lut_gate_start_ids<P: Poly>(circuit: &PolyCircuit<P>) -> (HashMap<GateId, usize>, usize) {
    let mut start_ids = HashMap::new();
    let mut cursor = 0usize;
    for (gate_id, gate) in circuit.gates_in_id_order() {
        if let PolyGateType::PubLut { lut_id } = gate.gate_type {
            start_ids.insert(*gate_id, cursor);
            cursor += circuit.lookup_len(lut_id);
        }
    }
    (start_ids, cursor)
}

fn lut_gate_index(
    start_ids: &HashMap<GateId, usize>,
    gate_id: GateId,
    lut_input_index: usize,
) -> usize {
    let start = start_ids
        .get(&gate_id)
        .unwrap_or_else(|| panic!("missing LUT gate start id for gate {gate_id}"));
    start + lut_input_index
}

fn derive_a_out_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    id: GateId,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m = row_size * params.modulus_digits();
    let hash_sampler = SH::new();
    let tag = format!("A_OUT_{id}");
    hash_sampler.sample_hash(params, hash_key, tag.into_bytes(), row_size, m, DistType::FinRingDist)
}

fn derive_r_g_i_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    m_b: usize,
    hash_key: [u8; 32],
    id: GateId,
    index: usize,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let hash_sampler = SH::new();
    let tag = format!("R_{id}_{index}");
    hash_sampler.sample_hash(
        params,
        hash_key,
        tag.into_bytes(),
        row_size,
        m_b,
        DistType::FinRingDist,
    )
}
