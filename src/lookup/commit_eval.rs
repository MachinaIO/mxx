use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{PolyCircuit, gate::GateId},
    commit::wee25::{MsgMatrixStream, Wee25Commit, Wee25PublicParams},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::read_matrix_from_multi_batch,
        write::{add_lookup_buffer, get_lookup_buffer},
    },
    utils::mod_inverse_mod_q,
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData, path::PathBuf, sync::Arc};

#[derive(Debug, Clone)]
struct GateState<M: PolyMatrix> {
    gate_id: GateId,
    lut_id: usize,
    one_pubkey: BggPublicKey<M>,
    input_pubkey: BggPublicKey<M>,
}

#[derive(Debug, Clone)]
struct GateStateCollector<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    hash_key: [u8; 32],
    gate_states: DashMap<GateId, GateState<M>>,
    luts: DashMap<usize, PublicLut<M::P>>,
    _hs: PhantomData<HS>,
}

impl<M, HS> GateStateCollector<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn new(hash_key: [u8; 32]) -> Self {
        Self { hash_key, gate_states: DashMap::new(), luts: DashMap::new(), _hs: PhantomData }
    }
}

impl<M, HS> PltEvaluator<BggPublicKey<M>> for GateStateCollector<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<M::P as Poly>::Params,
        plt: &PublicLut<M::P>,
        one: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        self.luts.entry(lut_id).or_insert_with(|| plt.clone());
        let row_size = input.matrix.row_size();
        self.gate_states.insert(
            gate_id,
            GateState { gate_id, lut_id, one_pubkey: one.clone(), input_pubkey: input.clone() },
        );
        let a_out = derive_a_out_matrix::<M, HS>(params, row_size, self.hash_key, gate_id);
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[derive(Debug, Clone)]
pub struct CommitBGGPubKeyPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trapdoor_sigma: f64,
    pub wee25_commit: Wee25Commit<M>,
    pub wee25_public_params: Wee25PublicParams<M>,
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
    pub fn setup<US: PolyUniformSampler<M = M> + Send + Sync, TS: PolyTrapdoorSampler<M = M>>(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        trapdoor_sigma: f64,
        tree_base: usize,
        hash_key: [u8; 32],
    ) -> Self {
        tracing::debug!("CommitBGGPubKeyPltEvaluator::setup start");
        let wee25_commit = Wee25Commit::<M>::new(params, secret_size, tree_base);
        let wee25_public_params =
            wee25_commit.sample_public_params::<US, TS>(params, trapdoor_sigma);
        tracing::debug!("Wee25PublicParams setup done");
        let hash_sampler = HS::new();
        let b_1 = hash_sampler.sample_hash(
            params,
            hash_key,
            b"COMMIT_LUT_B1".to_vec(),
            secret_size,
            wee25_commit.m_b,
            DistType::FinRingDist,
        );
        let result = Self {
            hash_key,
            trapdoor_sigma,
            wee25_commit,
            wee25_public_params,
            b_1,
            gate_states: DashMap::new(),
            luts: DashMap::new(),
            _hs: PhantomData,
        };
        tracing::debug!("CommitBGGPubKeyPltEvaluator::setup done");
        result
    }

    pub fn from_public_params(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        trapdoor_sigma: f64,
        tree_base: usize,
        hash_key: [u8; 32],
        wee25_public_params: Wee25PublicParams<M>,
    ) -> Self {
        let wee25_commit = Wee25Commit::<M>::new(params, secret_size, tree_base);
        debug_assert_eq!(
            wee25_public_params.b.col_size(),
            wee25_commit.m_b,
            "wee25 public params column size mismatch"
        );
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
            trapdoor_sigma,
            wee25_commit,
            wee25_public_params,
            b_1,
            gate_states: DashMap::new(),
            luts: DashMap::new(),
            _hs: PhantomData,
        }
    }

    pub fn commit_all_lut_matrices<TS: PolyTrapdoorSampler<M = M> + Send + Sync>(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
    ) {
        tracing::debug!("commit_all_lut_matrices start");
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
                gate_states.push((
                    state.gate_id,
                    state.lut_id,
                    state.one_pubkey,
                    state.input_pubkey,
                ));
            }
        }
        tracing::debug!("commit_all_lut_matrices build msg_stream start");
        let (msg_stream, _, _, _) = build_msg_stream_for_lut_eval::<M, HS>(
            params,
            &self.wee25_commit,
            &self.wee25_public_params,
            &self.b_1,
            self.hash_key,
            &luts,
            &gate_states,
        );
        tracing::debug!("commit_all_lut_matrices build msg_stream done");
        self.wee25_public_params.write_to_storage(Wee25PublicParams::<M>::default_storage_prefix());
        let commit = self.wee25_commit.commit(params, &msg_stream, &self.wee25_public_params);
        let target = commit + &self.b_1;
        let trapdoor_sampler = TS::new(params, self.trapdoor_sigma);
        let preimage = trapdoor_sampler.preimage(params, b0_trapdoor, b0_matrix, &target);
        add_lookup_buffer(get_lookup_buffer(vec![(0, preimage)], &format!("preimage_of_commit")));
        tracing::debug!("commit_all_lut_matrices done");
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
        one: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        tracing::debug!(
            "CommitBGGPubKeyPltEvaluator::public_lookup gate_id={gate_id} lut_id={lut_id}"
        );
        self.luts.entry(lut_id).or_insert_with(|| plt.clone());
        let row_size = input.matrix.row_size();
        self.gate_states.insert(
            gate_id,
            GateState { gate_id, lut_id, one_pubkey: one.clone(), input_pubkey: input.clone() },
        );
        let a_out = derive_a_out_matrix::<M, HS>(params, row_size, self.hash_key, gate_id);
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[derive(Debug, Clone)]
pub struct CommitBGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    M::P: 'static,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub wee25_commit: Wee25Commit<M>,
    pub wee25_public_params: Wee25PublicParams<M>,
    pub hash_key: [u8; 32],
    pub b_1: M,
    pub c_b: M,
    pub c_commit: M,
    pub luts: HashMap<usize, PublicLut<M::P>>,
    pub gate_states: Vec<(GateId, usize, BggPublicKey<M>, BggPublicKey<M>)>,
    pub lut_gate_start_ids: HashMap<GateId, usize>,
    pub reconst_coeffs: Vec<num_bigint::BigUint>,
    _hs: PhantomData<HS>,
}

impl<M, HS> CommitBGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub fn setup(
        params: &<M::P as Poly>::Params,
        tree_base: usize,
        hash_key: [u8; 32],
        circuit: &PolyCircuit<M::P>,
        one_pubkey: &BggPublicKey<M>,
        input_pubkeys: &[BggPublicKey<M>],
        c_b0: &M,
        c_b: &M,
        dir_path: &PathBuf,
    ) -> Self {
        tracing::debug!("CommitBGGEncodingPltEvaluator::setup start");
        let secret_size = one_pubkey.matrix.row_size();
        let dir = std::path::Path::new(dir_path);
        let wee25_commit = Wee25Commit::<M>::new(params, secret_size, tree_base);
        let wee25_public_params = Wee25PublicParams::<M>::read_from_storage(
            params,
            dir,
            Wee25PublicParams::<M>::default_storage_prefix(),
        )
        .unwrap_or_else(|| panic!("wee25 public params not found"));
        let hash_sampler = HS::new();
        let b_1 = hash_sampler.sample_hash(
            params,
            hash_key,
            b"COMMIT_LUT_B1".to_vec(),
            secret_size,
            wee25_commit.m_b,
            DistType::FinRingDist,
        );

        let gate_state_collector = GateStateCollector::<M, HS>::new(hash_key);
        // setup pubkeys for all LUT gates
        let _ = circuit.eval(params, one_pubkey, input_pubkeys, Some(&gate_state_collector));
        let preimage =
            read_matrix_from_multi_batch::<M>(params, dir, &format!("preimage_of_commit"), 0)
                .unwrap_or_else(|| panic!("preimage_of_commit not found"));
        let c_commit = c_b0.clone() * preimage;
        let luts = gate_state_collector
            .luts
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect::<HashMap<_, _>>();
        let gate_states = gate_state_collector
            .gate_states
            .iter()
            .map(|entry| {
                let state = entry.value();
                (state.gate_id, state.lut_id, state.one_pubkey.clone(), state.input_pubkey.clone())
            })
            .collect::<Vec<_>>();
        let (lut_gate_start_ids, _lut_vector_len, _) = build_lut_gate_layout(&luts, &gate_states);
        let result = Self {
            wee25_commit,
            wee25_public_params,
            hash_key,
            b_1,
            c_b: c_b.clone(),
            c_commit,
            luts,
            gate_states,
            lut_gate_start_ids,
            reconst_coeffs: params.reconst_coeffs(),
            _hs: PhantomData,
        };
        tracing::debug!("CommitBGGEncodingPltEvaluator::setup done");
        result
    }
}

impl<M, HS> PltEvaluator<BggEncoding<M>> for CommitBGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<M::P as Poly>::Params,
        plt: &PublicLut<M::P>,
        one: &BggEncoding<M>,
        input: &BggEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggEncoding<M> {
        tracing::debug!(
            "CommitBGGEncodingPltEvaluator::public_lookup gate_id={gate_id} lut_id={lut_id}"
        );
        let x = input
            .plaintext
            .as_ref()
            .expect("the BGG encoding should reveal plaintext for public lookup");
        let (k, y) = plt.get(params, x).unwrap_or_else(|| {
            panic!("{:?} not found in LUT for gate {}", x.to_const_int(), gate_id)
        });
        let lut_vector_idx = lut_gate_index(&self.lut_gate_start_ids, gate_id, k);
        let (msg_stream, padded_len, _, _) = build_msg_stream_for_lut_eval::<M, HS>(
            params,
            &self.wee25_commit,
            &self.wee25_public_params,
            &self.b_1,
            self.hash_key,
            &self.luts,
            &self.gate_states,
        );
        let opening = self.wee25_commit.open(
            params,
            &msg_stream,
            Some(lut_vector_idx..(lut_vector_idx + 1)),
            &self.wee25_public_params,
        );
        let verifier = self.wee25_commit.verifier(
            padded_len,
            Some(lut_vector_idx..(lut_vector_idx + 1)),
            &self.wee25_public_params,
        );
        let secret_size = input.pubkey.matrix.row_size();
        let m_b = self.wee25_commit.m_b;
        let r_g_i =
            derive_r_g_i_matrix::<M, HS>(params, secret_size, m_b, self.hash_key, gate_id, k);
        let canceler = derive_canceler_matrix::<M>(
            params,
            &self.b_1,
            &verifier,
            &r_g_i,
            k,
            self.reconst_coeffs.as_slice(),
        );
        let c_lut = self.c_commit.clone() * verifier + self.c_b.clone() * opening;
        let c_x = (input.vector.clone() + &one.vector) * canceler.decompose();
        let c_out = (c_lut + c_x).slice_columns(0, self.wee25_commit.m_g);
        let a_out = derive_a_out_matrix::<M, HS>(
            params,
            input.pubkey.matrix.row_size(),
            self.hash_key,
            gate_id,
        );
        BggEncoding {
            pubkey: BggPublicKey { matrix: a_out, reveal_plaintext: true },
            vector: c_out,
            plaintext: Some(y),
        }
    }
}

impl<M, HS> CommitBGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
}

fn build_msg_stream_for_lut_eval<'a, M, HS>(
    params: &'a <M::P as Poly>::Params,
    wee25_commit: &'a Wee25Commit<M>,
    wee25_public_params: &'a Wee25PublicParams<M>,
    b_1: &'a M,
    hash_key: [u8; 32],
    luts: &'a HashMap<usize, PublicLut<M::P>>,
    gate_states: &'a [(GateId, usize, BggPublicKey<M>, BggPublicKey<M>)],
) -> (MsgMatrixStream<'a, M>, usize, HashMap<GateId, usize>, usize)
where
    M: PolyMatrix + 'a,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    tracing::debug!("build_msg_stream_for_lut_eval start");
    if gate_states.is_empty() {
        panic!("no LUT gates found for commit evaluator");
    }
    let secret_size = gate_states[0].2.matrix.row_size();
    let reconst_coeffs = params.reconst_coeffs();
    let gadget = M::gadget_matrix(params, secret_size);
    let m_g = wee25_commit.m_g;
    let m_b = wee25_commit.m_b;
    let b_1 = b_1.clone();
    let tree_base = wee25_commit.tree_base;
    let lut_layout = build_lut_layout_for_eval::<M>(luts, gate_states, tree_base);
    let padded_len = lut_layout.padded_len;
    let lut_vector_len = lut_layout.lut_vector_len;
    let lut_gate_start_ids = lut_layout.lut_gate_start_ids;
    let gate_ranges = lut_layout.gate_ranges;
    tracing::debug!(
        "build_msg_stream_for_lut_eval padded_len={padded_len} lut_vector_len={lut_vector_len}"
    );
    let reconst_coeffs = Arc::new(reconst_coeffs);
    let msg_stream = MsgMatrixStream::new(padded_len, move |range| {
        let range_start = range.start;
        let range_end = range.end;
        let verifier_range =
            wee25_commit.verifier(padded_len, Some(range_start..range_end), wee25_public_params);
        (range_start..range_end)
            .into_par_iter()
            .map(|global_idx| {
                if global_idx >= lut_vector_len {
                    return M::zero(params, secret_size, m_b);
                }
                let (start_idx, _end_idx, gate_id, lut_id, one_pubkey, input_pubkey) = gate_ranges
                    .iter()
                    .find(|(start, end, _, _, _, _)| global_idx >= *start && global_idx < *end)
                    .unwrap_or_else(|| panic!("missing LUT range for index {global_idx}"));
                let start_idx = *start_idx;
                let gate_id = *gate_id;
                let lut_id = *lut_id;
                let lut_input_index = global_idx - start_idx;
                let lut = luts.get(&lut_id).unwrap();
                let input = M::P::from_usize_to_constant(params, lut_input_index);
                let (idx, y_poly) = lut
                    .get(params, &input)
                    .unwrap_or_else(|| panic!("LUT entry {} missing", lut_input_index));
                let verifier_idx = start_idx + idx;
                let verifier_slice = if verifier_idx >= range_start && verifier_idx < range_end {
                    let local_idx = verifier_idx - range_start;
                    verifier_range.slice_columns(m_b * local_idx, m_b * (local_idx + 1))
                } else {
                    wee25_commit.verifier(
                        padded_len,
                        Some(verifier_idx..(verifier_idx + 1)),
                        wee25_public_params,
                    )
                };
                let a_out = derive_a_out_matrix::<M, HS>(params, secret_size, hash_key, gate_id);
                let r_g_i =
                    derive_r_g_i_matrix::<M, HS>(params, secret_size, m_b, hash_key, gate_id, idx);
                let canceler = derive_canceler_matrix::<M>(
                    params,
                    &b_1,
                    &verifier_slice,
                    &r_g_i,
                    idx,
                    reconst_coeffs.as_slice(),
                );
                let padded = (a_out.clone() - gadget.clone() * y_poly).concat_columns(&[&M::zero(
                    params,
                    secret_size,
                    m_b - m_g,
                )]);
                let pubkey_sum = input_pubkey.matrix.clone() + &one_pubkey.matrix;
                padded + &r_g_i - pubkey_sum * canceler.decompose()
            })
            .collect::<Vec<_>>()
    });
    tracing::debug!("build_msg_stream_for_lut_eval done");
    (msg_stream, padded_len, lut_gate_start_ids, lut_vector_len)
}

fn build_lut_gate_layout<M: PolyMatrix>(
    luts: &HashMap<usize, PublicLut<M::P>>,
    gate_states: &[(GateId, usize, BggPublicKey<M>, BggPublicKey<M>)],
) -> (
    HashMap<GateId, usize>,
    usize,
    Vec<(usize, usize, GateId, usize, BggPublicKey<M>, BggPublicKey<M>)>,
) {
    let mut sorted = gate_states.to_vec();
    sorted.sort_by_key(|(gate_id, _, _, _)| *gate_id);
    let mut start_ids = HashMap::new();
    let mut gate_ranges = Vec::with_capacity(sorted.len());
    let mut cursor = 0usize;
    for (gate_id, lut_id, one_pubkey, input_pubkey) in sorted.into_iter() {
        let lut = luts.get(&lut_id).unwrap_or_else(|| panic!("missing LUT for lut_id={lut_id}"));
        let start_idx = cursor;
        let end_idx = start_idx + lut.len();
        start_ids.insert(gate_id, start_idx);
        gate_ranges.push((start_idx, end_idx, gate_id, lut_id, one_pubkey, input_pubkey));
        cursor = end_idx;
    }
    (start_ids, cursor, gate_ranges)
}

#[derive(Debug)]
struct LutLayout<M: PolyMatrix> {
    lut_gate_start_ids: HashMap<GateId, usize>,
    lut_vector_len: usize,
    padded_len: usize,
    gate_ranges: Vec<(usize, usize, GateId, usize, BggPublicKey<M>, BggPublicKey<M>)>,
}

fn build_lut_layout_for_eval<M: PolyMatrix>(
    luts: &HashMap<usize, PublicLut<M::P>>,
    gate_states: &[(GateId, usize, BggPublicKey<M>, BggPublicKey<M>)],
    tree_base: usize,
) -> LutLayout<M> {
    let (lut_gate_start_ids, lut_vector_len, mut gate_ranges) =
        build_lut_gate_layout::<M>(luts, gate_states);
    gate_ranges.sort_by_key(|(start_idx, _, _, _, _, _)| *start_idx);
    let padded_len = compute_padded_len(tree_base, lut_vector_len);
    LutLayout { lut_gate_start_ids, lut_vector_len, padded_len, gate_ranges }
}

pub(crate) fn compute_padded_len(tree_base: usize, lut_vector_len: usize) -> usize {
    let mut padded_len = tree_base;
    while padded_len < lut_vector_len {
        padded_len *= tree_base;
    }
    padded_len
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

fn derive_canceler_matrix<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    b_1: &M,
    verifier_slice: &M,
    r_g_i: &M,
    idx: usize,
    reconst_coeffs: &[num_bigint::BigUint],
) -> M {
    let idx_inverse = mod_inverse_mod_q::<M::P>(idx as u64 + 1, params, reconst_coeffs).unwrap();
    (b_1.clone() * verifier_slice + r_g_i) * M::P::from_biguint_to_constant(params, idx_inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};
    use tracing::info;

    fn setup_lsb_constant_binary_plt(t_n: usize, params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::<DCRTPoly>::new_from_usize_range(
            params,
            t_n,
            |params, k| (k, DCRTPoly::from_usize_to_lsb(params, k)),
            None,
        )
    }

    const SIGMA: f64 = 4.578;

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_commit_plt_eval_single_input() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let plt = setup_lsb_constant_binary_plt(16, &params);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output]);

        let d = 1;
        let input_size = 1;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let secrets =
            uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
        let rand_int = (rand::random::<u64>() % 16) as usize;
        let plaintexts = vec![DCRTPoly::from_usize_to_constant(&params, rand_int); input_size];

        let reveal_plaintexts = vec![true; input_size];
        let bgg_encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b0_trapdoor, b0) = trapdoor_sampler.trapdoor(&params, d);
        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let c_b0 = s_vec.clone() * &b0;

        let dir_path = "test_data/test_commit_plt_eval_single_input";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let tree_base = 2;
        info!("plt pubkey evaluator setup start");
        let plt_pubkey_evaluator =
            CommitBGGPubKeyPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup::<
                DCRTPolyUniformSampler,
                DCRTPolyTrapdoorSampler,
            >(&params, d, SIGMA, tree_base, key);
        info!("plt pubkey evaluator setup done");

        info!("circuit eval pubkey start");
        let result_pubkey = circuit.eval(
            &params,
            &enc_one.pubkey,
            std::slice::from_ref(&enc1.pubkey),
            Some(&plt_pubkey_evaluator),
        );
        info!("circuit eval pubkey done");
        info!("commit_all_lut_matrices start");
        plt_pubkey_evaluator.commit_all_lut_matrices::<DCRTPolyTrapdoorSampler>(
            &params,
            &b0,
            &b0_trapdoor,
        );
        info!("commit_all_lut_matrices done");
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);
        let result_pubkey = &result_pubkey[0];

        let c_b = s_vec.clone() * plt_pubkey_evaluator.wee25_public_params.b.clone();
        info!("plt encoding evaluator setup start");
        let plt_encoding_evaluator =
            CommitBGGEncodingPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup(
                &params,
                tree_base,
                key,
                &circuit,
                &enc_one.pubkey,
                std::slice::from_ref(&enc1.pubkey),
                &c_b0,
                &c_b,
                &dir.to_path_buf(),
            );
        info!("plt encoding evaluator setup done");

        info!("circuit eval encoding start");
        let result_encoding = circuit.eval(
            &params,
            &enc_one,
            std::slice::from_ref(&enc1),
            Some(&plt_encoding_evaluator),
        );
        info!("circuit eval encoding done");
        assert_eq!(result_encoding.len(), 1);
        let result_encoding = &result_encoding[0];
        assert_eq!(result_encoding.pubkey, result_pubkey.clone());

        let expected_plaintext = plt.get(&params, &plaintexts[0]).unwrap().1;
        assert_eq!(result_encoding.plaintext.clone().unwrap(), expected_plaintext.clone());

        let expected_vector = s_vec.clone() *
            (result_encoding.pubkey.matrix.clone() -
                (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
        assert_eq!(result_encoding.vector, expected_vector);
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_commit_plt_eval_multi_inputs() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let plt = setup_lsb_constant_binary_plt(16, &params);

        let mut circuit = PolyCircuit::new();
        let input_size = 5;
        let inputs = circuit.input(input_size);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let outputs = inputs
            .iter()
            .map(|&input| circuit.public_lookup_gate(input, plt_id))
            .collect::<Vec<_>>();
        circuit.output(outputs);

        let d = 1;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let secrets =
            uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
        let rand_ints =
            (0..input_size).map(|_| (rand::random::<u64>() % 16) as usize).collect::<Vec<_>>();
        let plaintexts = rand_ints
            .iter()
            .map(|&rand_int| DCRTPoly::from_usize_to_constant(&params, rand_int))
            .collect::<Vec<_>>();

        let reveal_plaintexts = vec![true; input_size];
        let bgg_encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_one = encodings[0].clone();
        let input_pubkeys = pubkeys[1..].to_vec();
        let input_encodings = encodings[1..].to_vec();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b0_trapdoor, b0) = trapdoor_sampler.trapdoor(&params, d);
        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let c_b0 = s_vec.clone() * &b0;

        let dir_path = "test_data/test_commit_plt_eval_multi_inputs";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let tree_base = 2;
        info!("plt pubkey evaluator setup start");
        let plt_pubkey_evaluator =
            CommitBGGPubKeyPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup::<
                DCRTPolyUniformSampler,
                DCRTPolyTrapdoorSampler,
            >(&params, d, SIGMA, tree_base, key);
        info!("plt pubkey evaluator setup done");

        info!("circuit eval pubkey start");
        let result_pubkey =
            circuit.eval(&params, &enc_one.pubkey, &input_pubkeys, Some(&plt_pubkey_evaluator));
        info!("circuit eval pubkey done");
        info!("commit_all_lut_matrices start");
        plt_pubkey_evaluator.commit_all_lut_matrices::<DCRTPolyTrapdoorSampler>(
            &params,
            &b0,
            &b0_trapdoor,
        );
        info!("commit_all_lut_matrices done");
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), input_size);

        let c_b = s_vec.clone() * plt_pubkey_evaluator.wee25_public_params.b.clone();
        info!("plt encoding evaluator setup start");
        let plt_encoding_evaluator =
            CommitBGGEncodingPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup(
                &params,
                tree_base,
                key,
                &circuit,
                &enc_one.pubkey,
                &input_pubkeys,
                &c_b0,
                &c_b,
                &dir.to_path_buf(),
            );
        info!("plt encoding evaluator setup done");

        info!("circuit eval encoding start");
        let result_encoding =
            circuit.eval(&params, &enc_one, &input_encodings, Some(&plt_encoding_evaluator));
        info!("circuit eval encoding done");
        assert_eq!(result_encoding.len(), input_size);

        for (idx, encoding) in result_encoding.iter().enumerate() {
            let expected_plaintext = plt.get(&params, &plaintexts[idx]).unwrap().1;
            assert_eq!(encoding.plaintext.clone().unwrap(), expected_plaintext.clone());

            let expected_vector = s_vec.clone() *
                (encoding.pubkey.matrix.clone() -
                    (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
            assert_eq!(encoding.vector, expected_vector);
        }
    }
}
