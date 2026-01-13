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
use rayon::prelude::*;
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::info;

#[derive(Debug)]
pub struct GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
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
    matrix_per_lut: Arc<Mutex<HashMap<usize, Arc<M>>>>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M>,
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
            matrix_per_lut: Arc::new(Mutex::new(HashMap::new())),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    fn sample_lut_preimages(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        lut_id: usize,
        b_trapdoor_lut: &Arc<TS::Trapdoor>,
        b_matrix_lut: &Arc<M>,
    ) {
        let d = self.b0_matrix.row_size();
        let m = d * params.modulus_digits();
        let hash_sampler = HS::new();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let hash_key = self.hash_key;
        let d_matrix = self.d_matrix.clone();
        info!("Preparing LUT for LUT id {}", lut_id);
        let start = Instant::now();
        let chunk_size = std::env::var("LUT_PREIMAGE_CHUNK_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(80);
        let mut part_idx = 0usize;
        let mut total_matrices = 0usize;
        let mut batch: Vec<(usize, M::P)> = Vec::with_capacity(chunk_size);

        let flush_batch = |batch: Vec<(usize, M::P)>, part_idx: usize| {
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
                    (*idx, trap_sampler.preimage(params, &b_trapdoor_lut, &b_matrix_lut, &target))
                })
                .collect();
            let kl_id = if part_idx == 0 {
                format!("ggh15_lut_{}", lut_id)
            } else {
                format!("ggh15_lut_{}_part{}", lut_id, part_idx)
            };
            add_lookup_buffer(get_lookup_buffer(k_l_preimages, &kl_id));
        };

        for (_, (idx, y_poly)) in plt.entries(params) {
            batch.push((idx, y_poly));
            if batch.len() >= chunk_size {
                let current = std::mem::replace(&mut batch, Vec::with_capacity(chunk_size));
                total_matrices += current.len();
                flush_batch(current, part_idx);
                part_idx += 1;
            }
        }
        if !batch.is_empty() {
            total_matrices += batch.len();
            flush_batch(batch, part_idx);
            part_idx += 1;
        }
        info!(
            "Prepared LUT for LUT id {} in {:?} ({} matrices, {} parts)",
            lut_id,
            start.elapsed(),
            total_matrices,
            part_idx
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
        one: BggPublicKey<M>,
        input: BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size();
        let uniform_sampler = US::new();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        info!("Starting public lookup for gate {}", gate_id);
        let (b_matrix_lut, b_trapdoor_lut) = {
            let cached = {
                let guard = self.matrix_per_lut.lock().unwrap();
                guard.get(&lut_id).cloned()
            };
            if let Some(matrix) = cached {
                (matrix, None)
            } else {
                let (b_trapdoor_lut, b_matrix_lut) = trap_sampler.trapdoor(params, 3 * d);
                info!(
                    "Sampled BGG LUT trapdoor and matrix for LUT id {} (matrix size: {}x{})",
                    lut_id,
                    b_matrix_lut.row_size(),
                    b_matrix_lut.col_size()
                );
                let b_trapdoor_lut = Arc::new(b_trapdoor_lut);
                let b_matrix_lut = Arc::new(b_matrix_lut);
                let mut guard = self.matrix_per_lut.lock().unwrap();
                if let Some(matrix) = guard.get(&lut_id) {
                    (matrix.clone(), None)
                } else {
                    guard.insert(lut_id, b_matrix_lut.clone());
                    (b_matrix_lut, Some(b_trapdoor_lut))
                }
            }
        };
        if let Some(b_trapdoor_lut) = b_trapdoor_lut {
            self.sample_lut_preimages(params, plt, lut_id, &b_trapdoor_lut, &b_matrix_lut);
        }

        let s_g = if self.insert_1_to_s {
            let s_g_bar =
                uniform_sampler.sample_uniform(params, d - 1, d - 1, DistType::TernaryDist);
            s_g_bar.concat_diag(&[&M::identity(params, 1, None)])
        } else {
            uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist)
        };
        let b_matrix_lut_1 = b_matrix_lut.slice_rows(0, d);
        let b_matrix_lut_2 = b_matrix_lut.slice_rows(d, 2 * d);
        let b_matrix_lut_3 = b_matrix_lut.slice_rows(2 * d, 3 * d);

        let c_matrix_1 = {
            let error = uniform_sampler.sample_uniform(
                params,
                d,
                b_matrix_lut_2.col_size(),
                DistType::GaussDist { sigma: self.error_sigma },
            );
            s_g.clone() * &b_matrix_lut_2 + error
        };
        let c_matrix_2 = {
            let error = uniform_sampler.sample_uniform(
                params,
                d,
                b_matrix_lut_3.col_size(),
                DistType::GaussDist { sigma: self.error_sigma },
            );
            s_g.clone() * &b_matrix_lut_3 + error
        };

        let k_to_ggh_target = {
            let one_muled =
                one.matrix.clone() * (b_matrix_lut_1.decompose() + c_matrix_1.decompose());
            let input_muled = input.matrix.clone() * c_matrix_2.decompose();
            one_muled + input_muled
        };
        let k_to_ggh =
            trap_sampler.preimage(params, &self.b0_trapdoor, &self.b0_matrix, &k_to_ggh_target);
        let a_out = {
            let error = uniform_sampler.sample_uniform(
                params,
                d,
                self.d_matrix.col_size(),
                DistType::GaussDist { sigma: self.error_sigma },
            );
            s_g * &self.d_matrix + error
        };

        let gate_bundle_id = format!("ggh15_gate_bundle_{}", gate_id);
        add_lookup_buffer(get_lookup_buffer(
            vec![
                (0, b_matrix_lut_1),
                (1, c_matrix_1),
                (2, c_matrix_2),
                (3, k_to_ggh),
                (4, a_out.clone()),
            ],
            &gate_bundle_id,
        ));

        BggPublicKey { matrix: a_out, reveal_plaintext: true }
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
        one: BggEncoding<M>,
        input: BggEncoding<M>,
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
        let gate_bundle_id = format!("ggh15_gate_bundle_{}", gate_id);
        let b_matrix_lut_1 = read_matrix_from_multi_batch::<M>(params, dir, &gate_bundle_id, 0)
            .unwrap_or_else(|| panic!("b_matrix_lut_1 for gate {} not found", gate_id));
        let c_matrix_1 = read_matrix_from_multi_batch::<M>(params, dir, &gate_bundle_id, 1)
            .unwrap_or_else(|| panic!("c_matrix_1 for gate {} not found", gate_id));
        let c_matrix_2 = read_matrix_from_multi_batch::<M>(params, dir, &gate_bundle_id, 2)
            .unwrap_or_else(|| panic!("c_matrix_2 for gate {} not found", gate_id));
        let k_to_ggh = read_matrix_from_multi_batch::<M>(params, dir, &gate_bundle_id, 3)
            .unwrap_or_else(|| panic!("k_to_ggh for gate {} not found", gate_id));
        let k_lut =
            read_matrix_from_multi_batch::<M>(params, dir, &format!("ggh15_lut_{}", lut_id), k)
                .unwrap_or_else(|| panic!("k_lut (index {}) for lut {} not found", k, lut_id));
        let a_out = read_matrix_from_multi_batch::<M>(params, dir, &gate_bundle_id, 4)
            .unwrap_or_else(|| panic!("a_out for gate {} not found", gate_id));

        let d_to_ggh = self.c_b0.clone() * k_to_ggh;
        let term_const = one.vector.clone() * (b_matrix_lut_1.decompose() + c_matrix_1.decompose());
        let term_input = input.vector.clone() * c_matrix_2.decompose();
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
            Some(plt_pubkey_evaluator),
        );
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
            Some(plt_encoding_evaluator),
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
            circuit.eval(&params, &enc_one.pubkey, &input_pubkeys, Some(plt_pubkey_evaluator));
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), input_size);

        let plt_encoding_evaluator = GGH15BGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(key, &params, dir_path.into(), d, c_b0);

        let result_encoding =
            circuit.eval(&params, &enc_one, &input_encodings, Some(plt_encoding_evaluator));
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
