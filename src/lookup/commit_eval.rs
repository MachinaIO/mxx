use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{PolyCircuit, PolyGateType, gate::GateId},
    commit::wee25::{MsgMatrixStream, Wee25Commit},
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
use std::{collections::HashMap, marker::PhantomData, path::PathBuf};

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
    pub trapdoor_sigma: f64,
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
        tracing::debug!("CommitBGGPubKeyPltEvaluator::setup start");
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
        let result = Self {
            hash_key,
            trapdoor_sigma,
            wee25_commit,
            lut_gate_start_ids,
            lut_vector_len,
            b_1,
            gate_states: DashMap::new(),
            luts: DashMap::new(),
            _hs: PhantomData,
        };
        tracing::debug!("CommitBGGPubKeyPltEvaluator::setup done");
        result
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
                gate_states.push((state.gate_id, state.lut_id, state.input_pubkey));
            }
        }
        tracing::debug!("commit_all_lut_matrices build msg_stream start");
        let (msg_stream, _verifier, _) =
            self.build_msg_stream_verifier_and_cancelers(params, luts, gate_states);
        tracing::debug!("commit_all_lut_matrices build msg_stream done");
        let commit = self.wee25_commit.commit(params, &msg_stream);
        let target = commit + &self.b_1;
        let trapdoor_sampler = TS::new(params, self.trapdoor_sigma);
        let preimage = trapdoor_sampler.preimage(params, b0_trapdoor, b0_matrix, &target);
        add_lookup_buffer(get_lookup_buffer(vec![(0, preimage)], &format!("preimage_of_commit")));
        tracing::debug!("commit_all_lut_matrices done");
    }

    fn build_msg_stream_verifier_and_cancelers<'a>(
        &'a self,
        params: &'a <M::P as Poly>::Params,
        luts: HashMap<usize, PublicLut<M::P>>,
        gate_states: Vec<(GateId, usize, BggPublicKey<M>)>,
    ) -> (MsgMatrixStream<'a, M>, M, Vec<M>)
    where
        M: 'a,
    {
        tracing::debug!("build_msg_stream_verifier_and_cancelers start");
        let secret_size = gate_states[0].2.matrix.row_size();
        let reconst_coeffs = params.reconst_coeffs();
        let gadget = M::gadget_matrix(params, secret_size);
        let m_g = self.wee25_commit.m_g;
        let m_b = self.wee25_commit.m_b;
        let b_1 = self.b_1.clone();
        let hash_key = self.hash_key;
        let tree_base = self.wee25_commit.tree_base;
        let mut padded_len = tree_base;
        while padded_len < self.lut_vector_len {
            padded_len *= tree_base;
        }
        let verifier = self.wee25_commit.verifier(padded_len, None);
        let verifier_for_stream = verifier.clone();
        let mut gate_ranges = gate_states
            .par_iter()
            .map(|(gate_id, lut_id, input_pubkey)| {
                let lut = luts.get(lut_id).unwrap();
                let start_idx = self.lut_gate_start_ids[gate_id];
                let end_idx = start_idx + lut.len();
                (start_idx, end_idx, *gate_id, *lut_id, input_pubkey.clone())
            })
            .collect::<Vec<_>>();
        gate_ranges.sort_by_key(|(start_idx, _, _, _, _)| *start_idx);
        tracing::debug!("build_msg_stream_verifier_and_cancelers cancelers start");
        let cancelers = (0..self.lut_vector_len)
            .into_par_iter()
            .map(|global_idx| {
                let (start_idx, _end_idx, gate_id, lut_id, _input_pubkey) = gate_ranges
                    .iter()
                    .find(|(start, end, _, _, _)| global_idx >= *start && global_idx < *end)
                    .unwrap_or_else(|| panic!("missing LUT range for index {global_idx}"));
                let start_idx = *start_idx;
                let gate_id = *gate_id;
                let lut_id = *lut_id;
                let lut_input_index = global_idx - start_idx;
                let lut = luts.get(&lut_id).unwrap();
                let input = M::P::from_usize_to_constant(params, lut_input_index);
                let (idx, _y_poly) = lut
                    .get(params, &input)
                    .unwrap_or_else(|| panic!("LUT entry {} missing", lut_input_index));
                let r_g_i =
                    derive_r_g_i_matrix::<M, HS>(params, secret_size, m_b, hash_key, gate_id, idx);
                let idx_inverse =
                    mod_inverse_mod_q::<M::P>(idx as u64 + 1, params, &reconst_coeffs).unwrap();
                let verifier = verifier_for_stream
                    .slice_columns(m_b * (start_idx + idx), m_b * (start_idx + idx + 1));
                (b_1.clone() * verifier + &r_g_i) *
                    M::P::from_biguint_to_constant(params, idx_inverse)
            })
            .collect::<Vec<_>>();
        tracing::debug!("build_msg_stream_verifier_and_cancelers cancelers done");
        let cancelers_for_stream = cancelers.clone();
        let msg_stream = MsgMatrixStream::new(padded_len, move |range| {
            (range.start..range.end)
                .into_par_iter()
                .map(|global_idx| {
                    if global_idx >= self.lut_vector_len {
                        return M::zero(params, secret_size, m_b);
                    }
                    let (start_idx, _end_idx, gate_id, lut_id, input_pubkey) = gate_ranges
                        .iter()
                        .find(|(start, end, _, _, _)| global_idx >= *start && global_idx < *end)
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
                    let a_out =
                        derive_a_out_matrix::<M, HS>(params, secret_size, hash_key, gate_id);
                    let r_g_i = derive_r_g_i_matrix::<M, HS>(
                        params,
                        secret_size,
                        m_b,
                        hash_key,
                        gate_id,
                        idx,
                    );
                    let canceler = cancelers_for_stream[global_idx].clone();
                    let padded = (a_out.clone() - gadget.clone() * y_poly)
                        .concat_columns(&[&M::zero(params, secret_size, m_b - m_g)]);
                    padded + &r_g_i - input_pubkey.matrix.clone() * canceler.decompose()
                })
                .collect::<Vec<_>>()
        });
        tracing::debug!("build_msg_stream_verifier_and_cancelers done");
        (msg_stream, verifier, cancelers)
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
        tracing::debug!(
            "CommitBGGPubKeyPltEvaluator::public_lookup gate_id={gate_id} lut_id={lut_id}"
        );
        self.luts.entry(lut_id).or_insert_with(|| plt.clone());
        let row_size = input.matrix.row_size();
        self.gate_states
            .insert(gate_id, GateState { gate_id, lut_id, input_pubkey: input.clone() });
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
    pub commit_pubkey_evaluator: CommitBGGPubKeyPltEvaluator<M, HS>,
    pub c_b: M,
    pub c_commit: M,
    pub lut_helpers: Vec<(M, M, M)>,
    _hs: PhantomData<HS>,
}

impl<M, HS> CommitBGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    pub fn setup<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
        params: &<M::P as Poly>::Params,
        trapdoor_sigma: f64,
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
        let commit_pubkey_evaluator = CommitBGGPubKeyPltEvaluator::<M, HS>::setup::<US, TS>(
            params,
            secret_size,
            trapdoor_sigma,
            tree_base,
            hash_key,
            circuit,
        );
        // setup pubkeys for all LUT gates
        let _ = circuit.eval(params, one_pubkey, input_pubkeys, Some(&commit_pubkey_evaluator));
        let dir = std::path::Path::new(dir_path);
        let preimage =
            read_matrix_from_multi_batch::<M>(params, dir, &format!("preimage_of_commit"), 0)
                .unwrap_or_else(|| panic!("preimage_of_commit not found"));
        let c_commit = c_b0.clone() * preimage;
        let luts = commit_pubkey_evaluator
            .luts
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect::<HashMap<_, _>>();
        let gate_states = commit_pubkey_evaluator
            .gate_states
            .iter()
            .map(|entry| {
                let state = entry.value();
                (state.gate_id, state.lut_id, state.input_pubkey.clone())
            })
            .collect::<Vec<_>>();
        let m_b = commit_pubkey_evaluator.wee25_commit.m_b;
        let (verifier, cancelers, opening_all) = {
            let (msg_stream, verifier, cancelers) = commit_pubkey_evaluator
                .build_msg_stream_verifier_and_cancelers(params, luts, gate_states);
            tracing::debug!("CommitBGGEncodingPltEvaluator::setup opening start");
            let opening_all = commit_pubkey_evaluator.wee25_commit.open(params, &msg_stream, None);
            tracing::debug!("CommitBGGEncodingPltEvaluator::setup opening done");
            (verifier, cancelers, opening_all)
        };
        let lut_helpers = (0..commit_pubkey_evaluator.lut_vector_len)
            .into_par_iter()
            .map(|idx| {
                let verifier_slice = verifier.slice_columns(m_b * idx, m_b * (idx + 1));
                let opening = opening_all.slice_columns(m_b * idx, m_b * (idx + 1));
                (verifier_slice, opening, cancelers[idx].clone())
            })
            .collect::<Vec<_>>();
        let result = Self {
            commit_pubkey_evaluator,
            c_b: c_b.clone(),
            c_commit,
            lut_helpers,
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
        let lut_vector_idx =
            lut_gate_index(&self.commit_pubkey_evaluator.lut_gate_start_ids, gate_id, k);
        let (verifier, opening, canceler) = &self.lut_helpers[lut_vector_idx];
        let c_lut = self.c_commit.clone() * verifier + self.c_b.clone() * opening;
        let c_x = (input.vector.clone() + &one.vector) * canceler.decompose();
        let c_out = c_lut + c_x;
        let a_out = derive_a_out_matrix::<M, HS>(
            params,
            input.pubkey.matrix.row_size(),
            self.commit_pubkey_evaluator.hash_key,
            gate_id,
        );
        BggEncoding {
            pubkey: BggPublicKey { matrix: a_out, reveal_plaintext: true },
            vector: c_out,
            plaintext: Some(y),
        }
    }
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
            >(&params, d, SIGMA, tree_base, key, &circuit);
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

        let c_b = s_vec.clone() * plt_pubkey_evaluator.wee25_commit.b.clone();
        info!("plt encoding evaluator setup start");
        let plt_encoding_evaluator = CommitBGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::setup::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
            &params,
            SIGMA,
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
        assert_eq!(result_encoding.plaintext.clone().unwrap(), expected_plaintext);
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
            >(&params, d, SIGMA, tree_base, key, &circuit);
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

        let c_b = s_vec.clone() * plt_pubkey_evaluator.wee25_commit.b.clone();
        info!("plt encoding evaluator setup start");
        let plt_encoding_evaluator = CommitBGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::setup::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
            &params,
            SIGMA,
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
            assert_eq!(encoding.plaintext.clone().unwrap(), expected_plaintext);
        }
    }
}
