use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    gadgets,
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::read_matrix_from_multi_batch,
        write::{BatchLookupBuffer, add_lookup_buffer, get_lookup_buffer},
    },
};
use rayon::prelude::*;
use std::{
    collections::HashSet,
    marker::PhantomData,
    path::PathBuf,
    sync::{Arc, Mutex},
};

fn enqueue_single_matrix<M>(matrix: M, id_prefix: &str)
where
    M: PolyMatrix + Send + 'static,
{
    let buffer: BatchLookupBuffer = get_lookup_buffer(vec![(0usize, matrix)], id_prefix);
    add_lookup_buffer(buffer);
}

#[derive(Debug)]
pub struct Ggh15BggPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub b0_matrix: Arc<M>,
    pub b0_trapdoor: Arc<TS::Trapdoor>,
    pub b1_matrix: Arc<M>,              // B in the spec
    pub b1_trapdoor: Arc<TS::Trapdoor>, // trapdoor of B
    pub pubkey_1: BggPublicKey<M>,
    pub a_x_prime: Arc<M>,
    pub a_l_prime: Arc<M>,
    pub a_s_prime: Arc<M>,
    pub a_out_prime: Arc<M>,
    pub dir_path: PathBuf,
    pub insert_1_to_s: bool,
    lut_ready: Arc<Mutex<HashSet<usize>>>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> Ggh15BggPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        hash_key: [u8; 32],
        trapdoor_sigma: f64,
        error_sigma: f64,
        params: &<M::P as Poly>::Params,
        b0_matrix: Arc<M>,
        b0_trapdoor: Arc<TS::Trapdoor>,
        pubkey_1: BggPublicKey<M>,
        dir_path: PathBuf,
        insert_1_to_s: bool,
    ) -> Self {
        let trap_sampler = TS::new(params, trapdoor_sigma);
        let d = b0_matrix.row_size();
        debug_assert!(!insert_1_to_s || d > 1, "cannot insert 1 into s when d = 1");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, 3 * d);
        let m = d * params.modulus_digits();
        let hash_sampler = HS::new();
        let a_x_prime = Arc::new(hash_sampler.sample_hash(
            params,
            hash_key,
            b"ggh15_a_x_prime".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        ));
        let a_l_prime = Arc::new(hash_sampler.sample_hash(
            params,
            hash_key,
            b"ggh15_a_l_prime".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        ));
        let a_s_prime = Arc::new(hash_sampler.sample_hash(
            params,
            hash_key,
            b"ggh15_a_s_prime".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        ));
        let a_out_prime = Arc::new(a_l_prime.as_ref().clone() * a_s_prime.decompose());

        Self {
            hash_key,
            trapdoor_sigma,
            error_sigma,
            b0_matrix,
            b0_trapdoor,
            b1_matrix: Arc::new(b1_matrix),
            b1_trapdoor: Arc::new(b1_trapdoor),
            pubkey_1,
            a_x_prime,
            a_l_prime,
            a_s_prime,
            a_out_prime,
            dir_path,
            insert_1_to_s,
            lut_ready: Arc::new(Mutex::new(HashSet::new())),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }
}

impl<M, US, HS, TS> PltEvaluator<BggPublicKey<M>> for Ggh15BggPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        input: BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size();
        let m = d * params.modulus_digits();
        let hash_sampler = HS::new();

        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b_g_trapdoor, b_g) = trap_sampler.trapdoor(params, 2 * d);
        let b_g1 = b_g.slice_rows(0, d);
        let b_g2 = b_g.slice_rows(d, 2 * d);
        let a1_ginv = self.pubkey_1.matrix.clone() * b_g1.decompose();
        let ax_ginv = input.matrix.clone() * b_g2.decompose();
        let target_to_ggh = a1_ginv + ax_ginv;
        let k_to_ggh =
            trap_sampler.preimage(params, &self.b0_trapdoor, &self.b0_matrix, &target_to_ggh);

        let uniform_sampler = US::new();
        let s_g = if self.insert_1_to_s {
            let s_g_bar =
                uniform_sampler.sample_uniform(params, d - 1, d - 1, DistType::TernaryDist);
            s_g_bar.concat_diag(&[&M::identity(params, 1, None)])
        } else {
            uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist)
        };
        let k_g_target = {
            let tensored_s = s_g.concat_diag(&[&s_g]);
            let j = M::identity(params, d, None).concat_rows(&[&M::zero(params, d, d)]);
            let secret_extend = tensored_s.concat_columns(&[&j]);
            let error = uniform_sampler.sample_uniform(
                params,
                2 * d,
                self.b1_matrix.col_size(),
                DistType::GaussDist { sigma: self.error_sigma },
            );
            secret_extend * self.b1_matrix.as_ref() + error
        };
        let k_g = trap_sampler.preimage(params, &b_g_trapdoor, &b_g, &k_g_target);

        let gadget_matrix = M::gadget_matrix(params, d);
        let target_to_bgg = {
            let top_left = self.a_x_prime.concat_columns(&[&M::zero(params, d, m)]) -
                M::zero(params, d, m).concat_columns(&[&gadget_matrix]);
            let top_right = self.a_s_prime.concat_columns(&[&M::zero(params, d, m)]);
            let top = top_left.concat_columns(&[&top_right]);
            let bottom = M::zero(params, d, m).concat_rows(&[&-gadget_matrix]);
            top.concat_rows(&[&bottom])
        };
        let k_to_bgg =
            trap_sampler.preimage(params, &self.b1_trapdoor, &self.b1_matrix, &target_to_bgg);

        let mut need_lut = false;
        {
            let mut guard = self.lut_ready.lock().unwrap();
            if !guard.contains(&lut_id) {
                guard.insert(lut_id);
                need_lut = true;
            }
        }
        if need_lut {
            let k_l_preimages: Vec<(usize, M)> = plt
                .f
                .par_iter()
                .map(|(_, (idx, y_elem))| {
                    let y_poly = M::P::from_elem_to_constant(params, y_elem);
                    let gadget_matrix = M::gadget_matrix(params, d);
                    let r_idx = hash_sampler.sample_hash(
                        params,
                        self.hash_key,
                        format!("ggh15_r_{}_idx_{}", lut_id, idx),
                        d,
                        m,
                        DistType::FinRingDist,
                    );
                    let idx_poly = M::P::from_usize_to_constant(params, *idx);

                    let target_top = self.a_l_prime.as_ref().clone() -
                        gadget_matrix.clone() * y_poly -
                        self.a_x_prime.as_ref().clone() * &r_idx.decompose() +
                        r_idx * idx_poly;
                    let target = target_top.concat_rows(&[&M::zero(params, 2 * d, m)]);
                    (
                        *idx,
                        trap_sampler.preimage(params, &self.b1_trapdoor, &self.b1_matrix, &target),
                    )
                })
                .collect();
            let kl_id = format!("ggh15_lut_{}", lut_id);
            add_lookup_buffer(get_lookup_buffer(k_l_preimages, &kl_id));
        }

        let target_dec_top = -self.a_out_prime.as_ref().clone();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_a_out_gate_{}", gate_id).into_bytes(),
            d,
            m,
            DistType::FinRingDist,
        );
        let target_dec = target_dec_top.concat_rows(&[&M::zero(params, d, m), &a_out]);
        let k_dec = trap_sampler.preimage(params, &self.b1_trapdoor, &self.b1_matrix, &target_dec);

        enqueue_single_matrix(b_g, &format!("ggh15_bg_gate_{}", gate_id));
        enqueue_single_matrix(k_to_ggh, &format!("ggh15_kggh_gate_{}", gate_id));
        enqueue_single_matrix(k_g, &format!("ggh15_kg_gate_{}", gate_id));
        enqueue_single_matrix(k_to_bgg, &format!("ggh15_kbgg_gate_{}", gate_id));
        enqueue_single_matrix(k_dec, &format!("ggh15_kdec_gate {}", gate_id));

        // Output public key for this LUT gate.
        BggPublicKey { matrix: a_out, reveal_plaintext: true }
    }
}

#[derive(Debug, Clone)]
pub struct Ggh15BggEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub const_one: BggEncoding<M>,
    pub a_x_prime: M,
    pub a_l_prime: M,
    pub a_s_prime: M,
    pub c_b0: M,
    _hs: PhantomData<HS>,
}

impl<M, HS> Ggh15BggEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        params: &<M::P as Poly>::Params,
        dir_path: PathBuf,
        const_one: BggEncoding<M>,
        c_b0: M,
    ) -> Self {
        let d = const_one.pubkey.matrix.row_size();
        let m = d * params.modulus_digits();
        let hash_sampler = HS::new();
        let a_x_prime = hash_sampler.sample_hash(
            params,
            hash_key,
            b"ggh15_a_x_prime".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        );
        let a_l_prime = hash_sampler.sample_hash(
            params,
            hash_key,
            b"ggh15_a_l_prime".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        );
        let a_s_prime = hash_sampler.sample_hash(
            params,
            hash_key,
            b"ggh15_a_s_prime".to_vec(),
            d,
            m,
            DistType::FinRingDist,
        );
        Self {
            hash_key,
            dir_path,
            const_one,
            a_x_prime,
            a_l_prime,
            a_s_prime,
            c_b0,
            _hs: PhantomData,
        }
    }
}

impl<M, HS> PltEvaluator<BggEncoding<M>> for Ggh15BggEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        input: BggEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggEncoding<M> {
        let x = input
            .plaintext
            .as_ref()
            .expect("the BGG encoding should reveal plaintext for public lookup");
        let x_coeff = x
            .coeffs()
            .first()
            .cloned()
            .expect("plaintext polynomial must contain at least one coefficient");
        let (k, y_coeff) = plt.get(params, &x_coeff).unwrap_or_else(|| {
            panic!("{:?} not found in LUT for gate {}", x.to_const_int(), gate_id)
        });

        let d = input.pubkey.matrix.row_size();
        let log_base_q = params.modulus_digits();
        let m = d * log_base_q;

        let hash_sampler = HS::new();

        let dir = std::path::Path::new(&self.dir_path);
        let b_g = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_bg_gate_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("b_g for gate {} not found", gate_id));
        let k_to_ggh = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_kggh_gate_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("k_to_ggh for gate {} not found", gate_id));
        let k_g = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_kg_gate_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("k_g for gate {} not found", gate_id));
        let k_to_bgg = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_kbgg_gate_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("k_to_bgg for gate {} not found", gate_id));
        let k_lut =
            read_matrix_from_multi_batch::<M>(params, dir, &format!("ggh15_lut_{}", lut_id), k)
                .unwrap_or_else(|| panic!("k_lut (index {}) for lut {} not found", k, lut_id));
        let k_dec = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_kdec_gate {}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("decoder for gate {} not found", gate_id));

        let d_to_ggh = self.c_b0.clone() * k_to_ggh;
        let b_g1 = b_g.slice_rows(0, d);
        let b_g2 = b_g.slice_rows(d, 2 * d);
        let term_const = self.const_one.vector.clone() * b_g1.decompose();
        let term_input = input.vector.clone() * b_g2.decompose();
        let p_g = d_to_ggh - &(term_const + term_input);

        let p_g_prime = p_g * k_g;

        let c_prime = p_g_prime.clone() * k_to_bgg;
        let c_x_prime = c_prime.slice_columns(0, m);
        let c_s_prime = c_prime.slice_columns(m, 2 * m);

        let d_lut = p_g_prime.clone() * k_lut;
        let r_k = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_r_{}_idx_{}", lut_id, k),
            d,
            m,
            DistType::FinRingDist,
        );
        let c_lut_prime = d_lut + c_x_prime * r_k.decompose();

        let d_dec = p_g_prime * k_dec;
        let y = M::P::from_elem_to_constant(params, &y_coeff);
        let c_dec_prime = c_lut_prime * self.a_s_prime.decompose() + c_s_prime * &y;
        let c_out = d_dec + c_dec_prime;

        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_a_out_gate_{}", gate_id).into_bytes(),
            d,
            m,
            DistType::FinRingDist,
        );
        let output_pubkey = BggPublicKey { matrix: a_out, reveal_plaintext: true };
        BggEncoding::new(c_out, output_pubkey, Some(y))
    }
}
