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
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{debug, info};

struct GateState<M>
where
    M: PolyMatrix,
{
    lut_id: usize,
    input_pubkey_bytes: Vec<u8>,
    output_pubkey_bytes: Vec<u8>,
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
    pub b0_matrix: Arc<M>,
    pub b0_trapdoor: Arc<TS::Trapdoor>,
    pub dir_path: PathBuf,
    pub insert_1_to_s: bool,
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
        b0_matrix: Arc<M>,
        b0_trapdoor: Arc<TS::Trapdoor>,
        dir_path: PathBuf,
        insert_1_to_s: bool,
    ) -> Self {
        let d = b0_matrix.row_size();
        debug_assert!(!insert_1_to_s || d > 1, "cannot insert 1 into s when d = 1");

        Self {
            hash_key,
            trapdoor_sigma,
            error_sigma,
            b0_matrix,
            b0_trapdoor,
            dir_path,
            insert_1_to_s,
            lut_state: DashMap::new(),
            gate_state: DashMap::new(),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    fn sample_lut_preimages(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        w_matrix: &M,
        batch: &[(usize, M::P)],
        part_idx: usize,
    ) {
        debug!(
            "Sampling LUT preimages: lut_id={}, part_idx={}, batch_size={}",
            lut_id,
            part_idx,
            batch.len()
        );
        let d = self.b0_matrix.row_size();
        let m = d * params.modulus_digits();
        let k = params.modulus_digits();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let uniform_sampler = US::new();
        let gadget_matrix = M::gadget_matrix(params, d);
        let block0_end = m;
        let block1_end = block0_end + m;
        let block2_end = block1_end + (m * k);
        let block3_end = block2_end + (m * k);
        debug_assert_eq!(
            w_matrix.col_size(),
            block3_end,
            "w_matrix columns must match [I | Gy | G^-1(v_idx) | G^-1(v_idx*idx)] layout"
        );
        let w_block_identity = w_matrix.slice_columns(0, block0_end);
        let w_block_gy = w_matrix.slice_columns(block0_end, block1_end);
        let w_block_v = w_matrix.slice_columns(block1_end, block2_end);
        let w_block_v_idx = w_matrix.slice_columns(block2_end, block3_end);
        let (vs, preimages): (Vec<(usize, M)>, Vec<(usize, M)>) = batch
            .par_iter()
            .map(|(idx, y_poly)| {
                let v_idx = uniform_sampler.sample_uniform(
                    params,
                    m,
                    m,
                    DistType::GaussDist { sigma: self.trapdoor_sigma },
                );
                debug!(
                    "Sampled v_idx for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let idx_poly = M::P::from_usize_to_constant(params, *idx);
                let gy = gadget_matrix.clone() * y_poly;
                let v_idx_scaled = v_idx.clone() * idx_poly;
                // Compute w_matrix * t_idx without materializing t_idx:
                // t_idx = [I; G^-1(G*y); G^-1(v_idx); G^-1(v_idx*idx)].
                let mut w_t_idx = w_block_identity.clone();
                w_t_idx = w_t_idx + w_block_gy.mul_decompose(&gy);
                w_t_idx = w_t_idx + w_block_v.mul_decompose(&v_idx);
                w_t_idx = w_t_idx + w_block_v_idx.mul_decompose(&v_idx_scaled);
                let target = M::zero(params, d, m).concat_rows(&[&w_t_idx]);
                debug!(
                    "Constructed target for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let k_l_preimage = trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &target);
                debug!(
                    "Sampled LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                ((*idx, v_idx), (*idx, k_l_preimage))
            })
            .collect();
        let vs_id = if part_idx == 0 {
            format!("ggh15_lut_v_idx_{}", lut_id)
        } else {
            format!("ggh15_lut_v_idx_{}_part{}", lut_id, part_idx)
        };
        let preimages_idx = if part_idx == 0 {
            format!("ggh15_lut_preimage_{}", lut_id)
        } else {
            format!("ggh15_lut_preimage_{}_part{}", lut_id, part_idx)
        };
        add_lookup_buffer(get_lookup_buffer(vs, &vs_id));
        add_lookup_buffer(get_lookup_buffer(preimages, &preimages_idx));
    }

    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs_f64();
        if secs >= 1.0 { format!("{secs:.3}s") } else { format!("{:.1}ms", secs * 1000.0) }
    }

    fn derive_w_matrix(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let d = self.b0_matrix.row_size();
        let m_g = d * params.modulus_digits();
        HS::new().sample_hash(
            params,
            self.hash_key,
            format!("ggh15_lut_w_matrix_{}", lut_id),
            d,
            2 * m_g + 2 * m_g * params.modulus_digits(),
            DistType::FinRingDist,
        )
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
        let d = self.b0_matrix.row_size();
        info!("Sampling auxiliary matrices with d = {}", d);
        let m_g = d * params.modulus_digits();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, 2 * d);
        let mut processed_lut_rows = 0usize;

        for (lut_id, plt) in lut_entries {
            let lut_start = Instant::now();
            let w_matrix = self.derive_w_matrix(params, lut_id);

            let mut part_idx = 0usize;
            let mut batch: Vec<(usize, M::P)> = Vec::with_capacity(chunk_size);
            for (_, (idx, y_poly)) in plt.entries(params) {
                batch.push((idx, y_poly));
                if batch.len() >= chunk_size {
                    self.sample_lut_preimages(
                        params,
                        lut_id,
                        &b1_trapdoor,
                        &b1_matrix,
                        &w_matrix,
                        &batch,
                        part_idx,
                    );
                    debug!(
                        "Sampled LUT preimages: lut_id={}, part_idx={}, batch_size={}",
                        lut_id,
                        part_idx,
                        batch.len()
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
                    &b1_trapdoor,
                    &b1_matrix,
                    &w_matrix,
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
            debug!("LUT {} complete in {}", lut_id, Self::format_duration(lut_start.elapsed()));
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

            while !gates.is_empty() {
                let take = gates.len().min(chunk_size);
                let current: Vec<(GateId, GateState<M>)> = gates.drain(..take).collect();
                total_gates += current.len();
                current.into_par_iter().for_each(|(gate_id, state)| {
                        let uniform_sampler = US::new();
                        let trap_sampler = TS::new(params, trapdoor_sigma);
                        let hash_sampler = HS::new();
                        let s_g = if self.insert_1_to_s {
                            let s_g_bar = uniform_sampler.sample_uniform(
                                params,
                                d - 1,
                                d - 1,
                                DistType::TernaryDist,
                            );
                            s_g_bar.concat_diag(&[&M::identity(params, 1, None)])
                        } else {
                            uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist)
                        };
                        let s_g_concat = M::identity(params, d, None).concat_columns(&[&s_g]);
                        let gate_target1 = {
                            let error = uniform_sampler.sample_uniform(
                                params,
                                d,
                                b1_matrix.col_size(),
                                DistType::GaussDist { sigma: error_sigma },
                            );
                            s_g_concat.clone() * &b1_matrix + error
                        };
                        let preimage_gate1 =
                            trap_sampler.preimage(params, &b0_trapdoor, &b0_matrix, &gate_target1);
                        drop(gate_target1);
                        drop(s_g_concat);
                        drop(s_g);
                        debug!("Sampled gate preimage 1: gate_id={}, lut_id={}", gate_id, lut_id);
                        let preimage_gate1_id = format!("ggh15_preimage_gate1_{}", gate_id);
                        add_lookup_buffer(get_lookup_buffer(
                            vec![(0, preimage_gate1)],
                            &preimage_gate1_id,
                        ));

                        let input_matrix = M::from_compact_bytes(params, &state.input_pubkey_bytes);
                        let out_matrix = M::from_compact_bytes(params, &state.output_pubkey_bytes);
                        let u_g_matrix = hash_sampler.sample_hash(
                            params,
                            self.hash_key,
                            format!("ggh15_lut_u_g_matrix_{}", gate_id),
                            d,
                            m_g,
                            DistType::FinRingDist,
                        );
                        let u_g_times_gadget = u_g_matrix.clone() * &M::gadget_matrix(params, m_g);
                        drop(u_g_matrix);
                        let u_g_times_gadget_decompose = u_g_times_gadget.decompose();

                        let w_matrix = self.derive_w_matrix(params, lut_id);
                        let k = params.modulus_digits();
                        let block0_end = m_g;
                        let block1_end = block0_end + m_g;
                        let block2_end = block1_end + (m_g * k);
                        let block3_end = block2_end + (m_g * k);
                        debug_assert_eq!(
                            w_matrix.col_size(),
                            block3_end,
                            "w_matrix columns must match [I | Gy | G^-1(v_idx) | G^-1(v_idx*idx)] layout"
                        );
                        let w_block_identity = w_matrix.slice_columns(0, block0_end);
                        let w_block_gy = w_matrix.slice_columns(block0_end, block1_end);
                        let w_block_v = w_matrix.slice_columns(block1_end, block2_end);
                        let w_block_vx = w_matrix.slice_columns(block2_end, block3_end);
                        drop(w_matrix);

                        let target_gate2_identity = out_matrix.concat_rows(&[&w_block_identity]);
                        drop(out_matrix);
                        drop(w_block_identity);
                        let preimage_gate2_identity =
                            trap_sampler.preimage(params, &b1_trapdoor, &b1_matrix, &target_gate2_identity);
                        drop(target_gate2_identity);
                        add_lookup_buffer(get_lookup_buffer(
                            vec![(0, preimage_gate2_identity)],
                            &format!("ggh15_preimage_gate2_identity_{}", gate_id),
                        ));

                        let target_high_gy = -M::gadget_matrix(params, d);
                        let target_gate2_gy = target_high_gy.concat_rows(&[&w_block_gy]);
                        drop(target_high_gy);
                        drop(w_block_gy);
                        let preimage_gate2_gy =
                            trap_sampler.preimage(params, &b1_trapdoor, &b1_matrix, &target_gate2_gy);
                        drop(target_gate2_gy);
                        add_lookup_buffer(get_lookup_buffer(
                            vec![(0, preimage_gate2_gy)],
                            &format!("ggh15_preimage_gate2_gy_{}", gate_id),
                        ));

                        let target_high_v = -(input_matrix * &u_g_times_gadget_decompose);
                        drop(u_g_times_gadget_decompose);
                        let target_gate2_v = target_high_v.concat_rows(&[&w_block_v]);
                        drop(target_high_v);
                        drop(w_block_v);
                        let preimage_gate2_v =
                            trap_sampler.preimage(params, &b1_trapdoor, &b1_matrix, &target_gate2_v);
                        drop(target_gate2_v);
                        add_lookup_buffer(get_lookup_buffer(
                            vec![(0, preimage_gate2_v)],
                            &format!("ggh15_preimage_gate2_v_{}", gate_id),
                        ));

                        let target_gate2_vx = u_g_times_gadget.concat_rows(&[&w_block_vx]);
                        drop(u_g_times_gadget);
                        drop(w_block_vx);
                        let preimage_gate2_vx =
                            trap_sampler.preimage(params, &b1_trapdoor, &b1_matrix, &target_gate2_vx);
                        drop(target_gate2_vx);
                        add_lookup_buffer(get_lookup_buffer(
                            vec![(0, preimage_gate2_vx)],
                            &format!("ggh15_preimage_gate2_vx_{}", gate_id),
                        ));
                    });
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
        _: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size();
        debug!("Starting public lookup for gate {}", gate_id);
        self.lut_state.entry(lut_id).or_insert_with(|| plt.clone());

        let hash_sampler = HS::new();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_gate_a_out_{}", gate_id),
            d,
            d * params.modulus_digits(),
            DistType::FinRingDist,
        );
        let output_pubkey = BggPublicKey { matrix: a_out, reveal_plaintext: true };
        self.gate_state.insert(
            gate_id,
            GateState {
                lut_id,
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                output_pubkey_bytes: output_pubkey.matrix.to_compact_bytes(),
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
    pub c_b0: M,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, c_b0: M) -> Self {
        Self { hash_key, dir_path, c_b0, _hs: PhantomData }
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
        _: &BggEncoding<M>,
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
        let preimage_gate1 = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_preimage_gate1_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate1 for gate {} not found", gate_id));
        let preimage_gate2_identity = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_preimage_gate2_identity_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_identity for gate {} not found", gate_id));
        let preimage_gate2_gy = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_preimage_gate2_gy_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_gy for gate {} not found", gate_id));
        let preimage_gate2_v = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_preimage_gate2_v_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_v for gate {} not found", gate_id));
        let preimage_gate2_vx = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_preimage_gate2_vx_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_vx for gate {} not found", gate_id));
        let preimage_lut = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_lut_preimage_{}", lut_id),
            k,
        )
        .unwrap_or_else(|| panic!("preimage_lut (index {}) for lut {} not found", k, lut_id));
        let v_idx = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("ggh15_lut_v_idx_{}", lut_id),
            k,
        )
        .unwrap_or_else(|| panic!("v_idx (index {}) for lut {} not found", k, lut_id));

        let hash_sampler = HS::new();
        let d = input.pubkey.matrix.row_size();
        let m_g = d * params.modulus_digits();
        let u_g = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", gate_id),
            d,
            m_g,
            DistType::FinRingDist,
        );
        let mk = m_g * params.modulus_digits();
        debug_assert_eq!(
            preimage_gate2_identity.col_size(),
            m_g,
            "preimage_gate2_identity must have m_g columns"
        );
        debug_assert_eq!(
            preimage_gate2_gy.col_size(),
            m_g,
            "preimage_gate2_gy must have m_g columns"
        );
        debug_assert_eq!(
            preimage_gate2_v.col_size(),
            mk,
            "preimage_gate2_v must have m_g*k columns"
        );
        debug_assert_eq!(
            preimage_gate2_vx.col_size(),
            mk,
            "preimage_gate2_vx must have m_g*k columns"
        );

        let gy = M::gadget_matrix(params, d) * y.clone();
        let v_idx_scaled = v_idx.clone() * x;
        let sg_times_b1 = self.c_b0.clone() * preimage_gate1;
        let c_pre2_identity = sg_times_b1.clone() * preimage_gate2_identity;
        let c_pre2_gy = (sg_times_b1.clone() * preimage_gate2_gy).mul_decompose(&gy);
        let c_pre2_v = (sg_times_b1.clone() * preimage_gate2_v).mul_decompose(&v_idx);
        let c_pre2_vx = (sg_times_b1.clone() * preimage_gate2_vx).mul_decompose(&v_idx_scaled);
        let c_const = c_pre2_identity + c_pre2_gy + c_pre2_v + c_pre2_vx - sg_times_b1 * preimage_lut;
        let c_x_randomized = input.vector.clone() *
            (u_g * M::gadget_matrix(params, m_g)).decompose() *
            v_idx.decompose();
        let c_out = c_const + c_x_randomized;
        let out_pubkey = BggPublicKey {
            matrix: hash_sampler.sample_hash(
                params,
                self.hash_key,
                format!("ggh15_gate_a_out_{}", gate_id),
                d,
                m_g,
                DistType::FinRingDist,
            ),
            reveal_plaintext: true,
        };
        BggEncoding::new(c_out, out_pubkey, Some(y))
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
        >::new(key, dir_path.into(), c_b0);

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
        >::new(key, dir_path.into(), c_b0);

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
