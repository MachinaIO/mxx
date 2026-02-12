use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::read_matrix_from_multi_batch,
        write::{add_lookup_buffer, get_lookup_buffer},
    },
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};
use tracing::{debug, info};

struct GateState<M>
where
    M: PolyMatrix,
{
    lut_id: usize,
    s_g_bytes: Vec<u8>,
    input_pubkey_bytes: Vec<u8>,
    input_pubkey_reveal_plaintext: bool,
    output_pubkey_bytes: Vec<u8>,
    output_pubkey_reveal_plaintext: bool,
    _m: PhantomData<M>,
}

pub struct GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub d_matrix: M,
    pub b0_matrix: Arc<M>,
    pub b0_trapdoor: Arc<TS::Trapdoor>,
    pub dir_path: PathBuf,
    pub insert_1_to_s: bool,
    one_pubkey: OnceLock<Arc<BggPublicKey<M>>>,
    lut_state: DashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
    gate_state: DashMap<GateId, GateState<M>>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(
        hash_key: [u8; 32],
        trapdoor_sigma: f64,
        error_sigma: f64,
        params: &<M::P as Poly>::Params,
        b0_matrix: Arc<M>,
        b0_trapdoor: Arc<TS::Trapdoor>,
        dir_path: PathBuf,
        insert_1_to_s: bool,
    ) -> Self {
        let d = b0_matrix.row_size();
        let m = d * params.modulus_digits();
        debug_assert!(!insert_1_to_s || d > 1, "cannot insert 1 into s when d = 1");
        let hash_sampler = HS::new();
        let d_matrix = hash_sampler.sample_hash(
            &params,
            hash_key,
            b"ggh15_d_matrix".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        );

        Self {
            hash_key,
            trapdoor_sigma,
            error_sigma,
            d_matrix,
            b0_matrix,
            b0_trapdoor,
            dir_path,
            insert_1_to_s,
            one_pubkey: OnceLock::new(),
            lut_state: DashMap::new(),
            gate_state: DashMap::new(),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    fn set_one_pubkey(&self, one: &BggPublicKey<M>) {
        let _ = self.one_pubkey.get_or_init(|| Arc::new(one.clone()));
    }

    fn get_one_pubkey(&self) -> Arc<BggPublicKey<M>> {
        self.one_pubkey.get().unwrap_or_else(|| panic!("one_pubkey is not set")).clone()
    }

    fn sample_lut_preimages(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        b_trapdoor_lut: &TS::Trapdoor,
        b_matrix_lut: &M,
        batch: &[(usize, M::P)],
        part_idx: usize,
    ) {
        let d = self.b0_matrix.row_size();
        let m = d * params.modulus_digits();
        let hash_sampler = HS::new();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let hash_key = self.hash_key;
        let d_matrix = self.d_matrix.clone();
        let k_l_preimages: Vec<(usize, M)> = batch
            .par_iter()
            .map(|(idx, y_poly)| {
                let gadget_matrix = M::gadget_matrix(params, d);
                let r_idx = hash_sampler.sample_hash(
                    params,
                    hash_key,
                    format!("ggh15_r_{}_idx_{}", lut_id, idx),
                    d,
                    m,
                    DistType::FinRingDist,
                );
                let idx_poly = M::P::from_usize_to_constant(params, *idx);

                let target_top = -gadget_matrix.clone() * y_poly.clone();
                let target_middle = r_idx.clone() * idx_poly + &d_matrix;
                let target = target_top.concat_rows(&[&target_middle, &-r_idx]);
                (*idx, trap_sampler.preimage(params, b_trapdoor_lut, b_matrix_lut, &target))
            })
            .collect();
        let kl_id = if part_idx == 0 {
            format!("ggh15_lut_{}", lut_id)
        } else {
            format!("ggh15_lut_{}_part{}", lut_id, part_idx)
        };
        add_lookup_buffer(get_lookup_buffer(k_l_preimages, &kl_id));
    }

    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs_f64();
        if secs >= 1.0 { format!("{secs:.3}s") } else { format!("{:.1}ms", secs * 1000.0) }
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        info!("Sampling LUT and gate auxiliary matrices");
        let start = Instant::now();
        let chunk_size = crate::env::lut_preimage_chunk_size();

        let lut_ids: Vec<usize> = self.lut_state.iter().map(|entry| *entry.key()).collect();
        let mut lut_entries = Vec::with_capacity(lut_ids.len());
        for lut_id in lut_ids {
            if let Some((_, plt)) = self.lut_state.remove(&lut_id) {
                lut_entries.push((lut_id, plt));
            }
        }

        let total_lut_rows: usize = lut_entries.iter().map(|(_, plt)| plt.len()).sum();
        debug!(
            "LUT sampling start: lut_count={}, total_rows={}, chunk_size={}",
            lut_entries.len(),
            total_lut_rows,
            chunk_size
        );

        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let d = self.b0_matrix.row_size();
        let mut b_matrix_entries: Vec<(usize, M)> = Vec::with_capacity(lut_entries.len());
        let mut b_matrix_map: HashMap<usize, Arc<M>> = HashMap::with_capacity(lut_entries.len());
        let mut processed_lut_rows = 0usize;

        for (lut_id, plt) in lut_entries {
            let lut_start = Instant::now();
            let (b_trapdoor_lut, b_matrix_lut) = trap_sampler.trapdoor(params, 3 * d);
            info!(
                "Sampled BGG LUT trapdoor and matrix for LUT id {} (matrix size: {}x{})",
                lut_id,
                b_matrix_lut.row_size(),
                b_matrix_lut.col_size()
            );
            let mut part_idx = 0usize;
            let mut batch: Vec<(usize, M::P)> = Vec::with_capacity(chunk_size);
            for (_, (idx, y_poly)) in plt.entries(params) {
                batch.push((idx, y_poly));
                if batch.len() >= chunk_size {
                    self.sample_lut_preimages(
                        params,
                        lut_id,
                        &b_trapdoor_lut,
                        &b_matrix_lut,
                        &batch,
                        part_idx,
                    );
                    processed_lut_rows = processed_lut_rows.saturating_add(batch.len());
                    let pct = if total_lut_rows == 0 {
                        100.0
                    } else {
                        (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                    };
                    debug!(
                        "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                        processed_lut_rows,
                        total_lut_rows,
                        Self::format_duration(start.elapsed())
                    );
                    part_idx += 1;
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                self.sample_lut_preimages(
                    params,
                    lut_id,
                    &b_trapdoor_lut,
                    &b_matrix_lut,
                    &batch,
                    part_idx,
                );
                processed_lut_rows = processed_lut_rows.saturating_add(batch.len());
                let pct = if total_lut_rows == 0 {
                    100.0
                } else {
                    (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                };
                debug!(
                    "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                    processed_lut_rows,
                    total_lut_rows,
                    Self::format_duration(start.elapsed())
                );
            }
            drop(b_trapdoor_lut);
            b_matrix_entries.push((lut_id, b_matrix_lut.clone()));
            b_matrix_map.insert(lut_id, Arc::new(b_matrix_lut));
            debug!("LUT {} complete in {}", lut_id, Self::format_duration(lut_start.elapsed()));
        }

        if !b_matrix_entries.is_empty() {
            add_lookup_buffer(get_lookup_buffer(b_matrix_entries, "ggh15_b_matrix_lut"));
        }

        let gate_ids: Vec<GateId> = self.gate_state.iter().map(|entry| *entry.key()).collect();
        let mut gate_entries = Vec::with_capacity(gate_ids.len());
        for gate_id in gate_ids {
            if let Some((_, state)) = self.gate_state.remove(&gate_id) {
                gate_entries.push((gate_id, state));
            }
        }

        let total_gate_count = gate_entries.len();
        debug!("Gate sampling start: total_gates={}, chunk_size={}", total_gate_count, chunk_size);

        if gate_entries.is_empty() {
            info!("No gate auxiliary matrices to sample");
            return;
        }

        let mut gates_by_lut: HashMap<usize, Vec<(GateId, GateState<M>)>> = HashMap::new();
        for (gate_id, state) in gate_entries {
            gates_by_lut.entry(state.lut_id).or_default().push((gate_id, state));
        }

        let b0_trapdoor = self.b0_trapdoor.clone();
        let b0_matrix = self.b0_matrix.clone();
        let error_sigma = self.error_sigma;
        let trapdoor_sigma = self.trapdoor_sigma;
        let mut total_gates = 0usize;

        for (lut_id, mut gates) in gates_by_lut {
            let lut_gate_start = Instant::now();
            let b_matrix_lut = b_matrix_map[&lut_id].clone();
            let d = b_matrix_lut.row_size() / 3;
            let b_matrix_lut_1 = Arc::new(b_matrix_lut.slice_rows(0, d));
            let b_matrix_lut_2 = Arc::new(b_matrix_lut.slice_rows(d, 2 * d));
            let b_matrix_lut_3 = Arc::new(b_matrix_lut.slice_rows(2 * d, 3 * d));

            while !gates.is_empty() {
                let take = gates.len().min(chunk_size);
                let current: Vec<(GateId, GateState<M>)> = gates.drain(..take).collect();
                total_gates += current.len();
                let b_matrix_lut_1 = b_matrix_lut_1.clone();
                let b_matrix_lut_2 = b_matrix_lut_2.clone();
                let b_matrix_lut_3 = b_matrix_lut_3.clone();
                let results: Vec<(GateId, M, M, M, M, M)> = current
                    .into_par_iter()
                    .map(|(gate_id, state)| {
                        let uniform_sampler = US::new();
                        let trap_sampler = TS::new(params, trapdoor_sigma);
                        let one_pubkey = self.get_one_pubkey();
                        let s_g = M::from_compact_bytes(params, &state.s_g_bytes);
                        let input_pubkey = BggPublicKey {
                            matrix: M::from_compact_bytes(params, &state.input_pubkey_bytes),
                            reveal_plaintext: state.input_pubkey_reveal_plaintext,
                        };
                        let output_pubkey = BggPublicKey {
                            matrix: M::from_compact_bytes(params, &state.output_pubkey_bytes),
                            reveal_plaintext: state.output_pubkey_reveal_plaintext,
                        };
                        let input_matrix = input_pubkey.matrix;
                        let a_out = output_pubkey.matrix;

                        let c_matrix_1 = {
                            let error = uniform_sampler.sample_uniform(
                                params,
                                d,
                                b_matrix_lut_2.col_size(),
                                DistType::GaussDist { sigma: error_sigma },
                            );
                            s_g.clone() * b_matrix_lut_2.as_ref() + error
                        };
                        let c_matrix_2 = {
                            let error = uniform_sampler.sample_uniform(
                                params,
                                d,
                                b_matrix_lut_3.col_size(),
                                DistType::GaussDist { sigma: error_sigma },
                            );
                            s_g.clone() * b_matrix_lut_3.as_ref() + error
                        };
                        let k_to_ggh_target = {
                            let one_matrix = &one_pubkey.matrix;
                            let one_muled = one_matrix.mul_decompose(b_matrix_lut_1.as_ref()) +
                                one_matrix.mul_decompose(&c_matrix_1);
                            let input_muled = input_matrix.mul_decompose(&c_matrix_2);
                            one_muled + input_muled
                        };
                        let k_to_ggh = trap_sampler.preimage(
                            params,
                            &b0_trapdoor,
                            &b0_matrix,
                            &k_to_ggh_target,
                        );
                        (gate_id, c_matrix_1, c_matrix_2, k_to_ggh_target, k_to_ggh, a_out)
                    })
                    .collect();

                for (gate_id, c1, c2, target, k_to_ggh, a_out) in results {
                    let gate_aux_entries =
                        vec![(0, c1), (1, c2), (2, target), (3, k_to_ggh), (4, a_out)];
                    let gate_aux_id = format!("ggh15_gate_aux_{}", gate_id);
                    add_lookup_buffer(get_lookup_buffer(gate_aux_entries, &gate_aux_id));
                }
                let pct = if total_gate_count == 0 {
                    100.0
                } else {
                    (total_gates as f64) * 100.0 / (total_gate_count as f64)
                };
                debug!(
                    "Gates processed: {}/{} ({pct:.1}%), elapsed={}",
                    total_gates,
                    total_gate_count,
                    Self::format_duration(start.elapsed())
                );
            }
            debug!(
                "Gate group for LUT {} complete in {}",
                lut_id,
                Self::format_duration(lut_gate_start.elapsed())
            );
        }

        info!(
            "Sampled LUT and gate auxiliary matrices in {} ({} gates)",
            Self::format_duration(start.elapsed()),
            total_gates
        );
    }
}

impl<M, US, HS, TS> PltEvaluator<BggPublicKey<M>> for GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        one: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size();
        let uniform_sampler = US::new();
        debug!("Starting public lookup for gate {}", gate_id);
        self.set_one_pubkey(one);
        self.lut_state.entry(lut_id).or_insert_with(|| plt.clone());

        let s_g = if self.insert_1_to_s {
            let s_g_bar =
                uniform_sampler.sample_uniform(params, d - 1, d - 1, DistType::TernaryDist);
            s_g_bar.concat_diag(&[&M::identity(params, 1, None)])
        } else {
            uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist)
        };
        let a_out = {
            let error = uniform_sampler.sample_uniform(
                params,
                d,
                self.d_matrix.col_size(),
                DistType::GaussDist { sigma: self.error_sigma },
            );
            s_g.clone() * &self.d_matrix + error
        };
        let output_pubkey = BggPublicKey { matrix: a_out, reveal_plaintext: true };
        self.gate_state.insert(
            gate_id,
            GateState {
                lut_id,
                s_g_bytes: s_g.to_compact_bytes(),
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                // input_pubkey_bytes: Vec::new(),
                input_pubkey_reveal_plaintext: input.reveal_plaintext,
                output_pubkey_bytes: output_pubkey.matrix.to_compact_bytes(),
                // output_pubkey_bytes: Vec::new(),
                output_pubkey_reveal_plaintext: output_pubkey.reveal_plaintext,
                _m: PhantomData,
            },
        );
        debug!("Public lookup for gate {} recorded", gate_id);
        output_pubkey
    }
}

#[derive(Debug, Clone)]
pub struct GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub d_matrix: M,
    pub c_b0: M,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        params: &<M::P as Poly>::Params,
        dir_path: PathBuf,
        secret_size: usize,
        c_b0: M,
    ) -> Self {
        let d = secret_size;
        let m = d * params.modulus_digits();
        let hash_sampler = HS::new();
        let d_matrix = hash_sampler.sample_hash(
            &params,
            hash_key,
            b"ggh15_d_matrix".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        );
        Self { hash_key, dir_path, d_matrix, c_b0, _hs: PhantomData }
    }
}

impl<M, HS> PltEvaluator<BggEncoding<M>> for GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        one: &BggEncoding<M>,
        input: &BggEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggEncoding<M> {
        let x = input
            .plaintext
            .as_ref()
            .expect("the BGG encoding should reveal plaintext for public lookup");
        let (k, y) = plt.get(params, x).unwrap_or_else(|| {
            panic!("{:?} not found in LUT for gate {}", x.to_const_int(), gate_id)
        });

        let dir = std::path::Path::new(&self.dir_path);
        let b_matrix_lut =
            read_matrix_from_multi_batch::<M>(params, dir, "ggh15_b_matrix_lut", lut_id)
                .unwrap_or_else(|| panic!("b_matrix_lut for lut {} not found", lut_id));
        debug_assert_eq!(b_matrix_lut.row_size() % 3, 0);
        let d = b_matrix_lut.row_size() / 3;
        let b_matrix_lut_1 = b_matrix_lut.slice_rows(0, d);
        let gate_aux_id = format!("ggh15_gate_aux_{}", gate_id);
        let c_matrix_1 = read_matrix_from_multi_batch::<M>(params, dir, &gate_aux_id, 0)
            .unwrap_or_else(|| panic!("c_matrix_1 for gate {} not found", gate_id));
        let c_matrix_2 = read_matrix_from_multi_batch::<M>(params, dir, &gate_aux_id, 1)
            .unwrap_or_else(|| panic!("c_matrix_2 for gate {} not found", gate_id));
        let k_to_ggh = read_matrix_from_multi_batch::<M>(params, dir, &gate_aux_id, 3)
            .unwrap_or_else(|| panic!("k_to_ggh for gate {} not found", gate_id));
        let k_lut =
            read_matrix_from_multi_batch::<M>(params, dir, &format!("ggh15_lut_{}", lut_id), k)
                .unwrap_or_else(|| panic!("k_lut (index {}) for lut {} not found", k, lut_id));
        let a_out = read_matrix_from_multi_batch::<M>(params, dir, &gate_aux_id, 4)
            .unwrap_or_else(|| panic!("a_out for gate {} not found", gate_id));

        let d_to_ggh = self.c_b0.clone() * k_to_ggh;
        let one_vector = &one.vector;
        let term_const =
            one_vector.mul_decompose(&b_matrix_lut_1) + one_vector.mul_decompose(&c_matrix_1);
        let term_input = input.vector.mul_decompose(&c_matrix_2);
        let p_g = d_to_ggh - &(term_const + term_input);
        let c_out = p_g * k_lut;
        let output_pubkey = BggPublicKey { matrix: a_out, reveal_plaintext: true };
        BggEncoding::new(c_out, output_pubkey, Some(y))
    }
}

#[cfg(test)]
mod test {
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
    use std::{fs, path::Path, sync::Arc};

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
    async fn test_ggh15_plt_eval_single_input() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_constant_binary_plt(16, &params);

        // Create a simple circuit with the lookup table
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output]);

        let d = 2;
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

        // Storage directory
        let dir_path = "test_data/test_ggh15_plt_eval_single_input";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let error_sigma = 0.0;
        let insert_1_to_s = false;
        let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(
            key,
            SIGMA,
            error_sigma,
            &params,
            Arc::new(b0),
            Arc::new(b0_trapdoor),
            dir_path.into(),
            insert_1_to_s,
        );

        let result_pubkey = circuit.eval(
            &params,
            &enc_one.pubkey,
            std::slice::from_ref(&enc1.pubkey),
            Some(&plt_pubkey_evaluator),
        );
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);
        let result_pubkey = &result_pubkey[0];

        let plt_encoding_evaluator = GGH15BGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(key, &params, dir_path.into(), d, c_b0);

        let result_encoding = circuit.eval(
            &params,
            &enc_one,
            std::slice::from_ref(&enc1),
            Some(&plt_encoding_evaluator),
        );
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
    async fn test_ggh15_plt_eval_multi_inputs() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_constant_binary_plt(16, &params);

        // Create a simple circuit with the lookup table
        let mut circuit = PolyCircuit::new();
        let input_size = 5;
        let inputs = circuit.input(input_size);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let outputs = inputs
            .iter()
            .map(|&input| circuit.public_lookup_gate(input, plt_id))
            .collect::<Vec<_>>();
        circuit.output(outputs);

        let d = 2;
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

        // Storage directory
        let dir_path = "test_data/test_ggh15_plt_eval_multi_inputs";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let error_sigma = 0.0;
        let insert_1_to_s = false;
        let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(
            key,
            SIGMA,
            error_sigma,
            &params,
            Arc::new(b0),
            Arc::new(b0_trapdoor),
            dir_path.into(),
            insert_1_to_s,
        );

        let result_pubkey =
            circuit.eval(&params, &enc_one.pubkey, &input_pubkeys, Some(&plt_pubkey_evaluator));
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), input_size);

        let plt_encoding_evaluator = GGH15BGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(key, &params, dir_path.into(), d, c_b0);

        let result_encoding =
            circuit.eval(&params, &enc_one, &input_encodings, Some(&plt_encoding_evaluator));
        assert_eq!(result_encoding.len(), input_size);

        for i in 0..input_size {
            let result_encoding_i = &result_encoding[i];
            assert_eq!(result_encoding_i.pubkey, result_pubkey[i].clone());

            let expected_plaintext = plt.get(&params, &plaintexts[i]).unwrap().1;
            assert_eq!(result_encoding_i.plaintext.clone().unwrap(), expected_plaintext.clone());

            let expected_vector = s_vec.clone() *
                (result_encoding_i.pubkey.matrix.clone() -
                    (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
            assert_eq!(result_encoding_i.vector, expected_vector);
        }
    }
}
