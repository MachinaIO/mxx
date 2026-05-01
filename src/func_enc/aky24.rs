use std::{marker::PhantomData, sync::Arc, time::Instant};

use digest::Digest;
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use tracing::{debug, info};

use crate::{
    bgg::naive_vec::{
        NaiveBGGEncodingVec, NaiveBGGEncodingVecSampler, NaiveBGGPublicKeyVec,
        NaiveBGGPublicKeyVecSampler,
    },
    circuit::{Evaluable, PolyCircuit, evaluable::PolyVec, gate::GateId},
    func_enc::FuncEnc,
    gadgets::{
        arith::{
            DecomposeArithmeticGadget, ModularArithmeticPlanner, NestedRnsPoly,
            NestedRnsPolyContext,
        },
        fhe::{
            ring_gsw::RingGswCiphertext,
            ring_gsw_nested_rns::{
                NestedRnsRingGswContext, ciphertext_inputs_from_native, encrypt_plaintext_bit,
                sample_public_key,
            },
        },
        fhe_prg::goldreich::GoldreichFhePrg,
    },
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    noise_refresh::{
        NoiseRefresher, NoiseRefresherNaiveVec,
        circuit_decrypt::{decrypt_bit_decomposed_scalar, mask_plaintext_moduli_from_full_modulus},
    },
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::SlotTransferEvaluator,
};

#[cfg(test)]
use crate::gadgets::fhe::ring_gsw_nested_rns::{active_q_modulus, decrypt_ciphertext};

#[derive(Debug, Clone)]
pub struct Aky24Params<M, TD>
where
    M: PolyMatrix,
{
    pub poly_params: <M::P as Poly>::Params,
    pub native_poly_params: DCRTPolyParams,
    pub ring_gsw_context: Arc<NestedRnsPolyContext>,
    pub ring_gsw_width: usize,
    pub ring_gsw_level_offset: usize,
    pub ring_gsw_enable_levels: Option<usize>,
    pub ring_gsw_public_key_error_sigma: Option<f64>,
    pub bgg_tag: Vec<u8>,
    pub trapdoor_sigma: f64,
    pub encoding_error_sigma: Option<f64>,
    pub b_error_sigma: Option<f64>,
    pub b_matrix: M,
    pub b_trapdoor: TD,
    pub prf_seed_bits: usize,
    pub prf_mask_output_coeff_bits: usize,
    pub goldreich_graph_seed: [u8; 32],
    pub noise_refresh_v_bits: usize,
    pub noise_refresh_cbd_n: usize,
    pub noise_refresh_hash_key: [u8; 32],
    _m: PhantomData<M>,
}

impl<M, TD> Aky24Params<M, TD>
where
    M: PolyMatrix,
{
    pub fn new(
        poly_params: <M::P as Poly>::Params,
        native_poly_params: DCRTPolyParams,
        ring_gsw_context: Arc<NestedRnsPolyContext>,
        ring_gsw_width: usize,
        ring_gsw_level_offset: usize,
        ring_gsw_enable_levels: Option<usize>,
        ring_gsw_public_key_error_sigma: Option<f64>,
        bgg_tag: Vec<u8>,
        trapdoor_sigma: f64,
        encoding_error_sigma: Option<f64>,
        b_matrix: M,
        b_trapdoor: TD,
        prf_seed_bits: usize,
        prf_mask_output_coeff_bits: usize,
        goldreich_graph_seed: [u8; 32],
        noise_refresh_v_bits: usize,
        noise_refresh_cbd_n: usize,
        noise_refresh_hash_key: [u8; 32],
    ) -> Self {
        assert!(noise_refresh_v_bits > 0, "noise_refresh_v_bits must be positive");
        assert!(noise_refresh_cbd_n > 0, "noise_refresh_cbd_n must be positive");
        assert!(prf_seed_bits > 0, "prf_seed_bits must be positive");
        assert!(prf_mask_output_coeff_bits > 0, "prf_mask_output_coeff_bits must be positive");
        Self {
            poly_params,
            native_poly_params,
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            ring_gsw_public_key_error_sigma,
            bgg_tag,
            trapdoor_sigma,
            encoding_error_sigma,
            b_error_sigma: encoding_error_sigma,
            b_matrix,
            b_trapdoor,
            prf_seed_bits,
            prf_mask_output_coeff_bits,
            goldreich_graph_seed,
            noise_refresh_v_bits,
            noise_refresh_cbd_n,
            noise_refresh_hash_key,
            _m: PhantomData,
        }
    }

    pub fn q(&self) -> Arc<BigUint> {
        self.poly_params.modulus().into()
    }

    pub fn n(&self) -> usize {
        self.poly_params.ring_dimension() as usize
    }

    pub fn prf_seed_bits(&self) -> usize {
        self.prf_seed_bits
    }

    pub fn prf_mask_output_coeff_bits(&self) -> usize {
        self.prf_mask_output_coeff_bits
    }

    pub fn secret_size(&self) -> usize {
        2
    }

    pub fn gadget_columns(&self) -> usize {
        self.secret_size() * self.poly_params.modulus_digits()
    }
}

#[derive(Debug, Clone)]
pub struct Aky24EncKey<M: PolyMatrix> {
    pub b_matrix: M,
    pub bgg_hash_key: [u8; 32],
}

pub struct Aky24MasterKey<M, TS>
where
    M: PolyMatrix,
    TS: PolyTrapdoorSampler<M = M>,
{
    pub b_matrix: M,
    pub b_trapdoor: TS::Trapdoor,
    pub bgg_hash_key: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct Aky24Ciphertext<M: PolyMatrix> {
    pub c_b: M,
    pub encodings: Vec<NaiveBGGEncodingVec<M>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aky24Func {
    DebugIdentity,
}

impl Aky24Func {
    pub fn output_size(&self) -> usize {
        match self {
            Self::DebugIdentity => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Aky24FuncKey<M: PolyMatrix> {
    pub func: Aky24Func,
    pub preimage: M,
    pub public_prf_seed: [bool; 128],
    pub prf_refresh_preimages: Vec<M>,
}

#[derive(Debug, Clone)]
pub enum Aky24Output {
    DebugIdentity { decrypted: bool },
}

pub struct Aky24FuncEnc<
    M,
    SH,
    US,
    TS,
    PKPE = NoCircuitEvaluator,
    PKST = NoCircuitEvaluator,
    ENCPE = NoCircuitEvaluator,
    ENCST = NoCircuitEvaluator,
> where
    M: PolyMatrix,
{
    pub pk_lookup_evaluator: Option<PKPE>,
    pub pk_slot_transfer_evaluator: Option<PKST>,
    pub enc_lookup_evaluator: Option<ENCPE>,
    pub enc_slot_transfer_evaluator: Option<ENCST>,
    _m: PhantomData<(M, SH, US, TS)>,
}

impl<M, SH, US, TS, PKPE, PKST, ENCPE, ENCST> Aky24FuncEnc<M, SH, US, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
{
    pub fn new(
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
        enc_lookup_evaluator: Option<ENCPE>,
        enc_slot_transfer_evaluator: Option<ENCST>,
    ) -> Self {
        Self {
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            enc_lookup_evaluator,
            enc_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }
}

impl<M, SH, US, TS, PKPE, PKST, ENCPE, ENCST> Aky24FuncEnc<M, SH, US, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    US: PolyUniformSampler<M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PKPE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
    PKST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    ENCPE: PltEvaluator<NaiveBGGEncodingVec<M>>,
    ENCST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
    TS::Trapdoor: Clone,
{
    fn keygen_prf_mask_target(
        &self,
        params: &Aky24Params<M, TS::Trapdoor>,
        msk: &Aky24MasterKey<M, TS>,
        trapdoor_sampler: &TS,
        public_prf_seed: [bool; 128],
        one: NaiveBGGPublicKeyVec<M>,
        decryption_key: NaiveBGGPublicKeyVec<M>,
        mut seed_wires: Vec<NaiveBGGPublicKeyVec<M>>,
        wire_count: usize,
    ) -> (M, Vec<M>) {
        let prf_seed_bits = params.prf_seed_bits();
        let pk_lookup_evaluator = self
            .pk_lookup_evaluator
            .as_ref()
            .expect("AKY24 PRF mask keygen requires a public-key lookup evaluator");
        let pk_slot_transfer_evaluator = self
            .pk_slot_transfer_evaluator
            .as_ref()
            .expect("AKY24 PRF mask keygen requires a public-key slot-transfer evaluator");
        let mut refresh_context_circuit = PolyCircuit::new();
        let noise_refresher = NoiseRefresherNaiveVec::<M, NestedRnsPoly<M::P>, SH>::new(
            build_ring_gsw_context(params, &mut refresh_context_circuit),
            prf_seed_bits,
            params.noise_refresh_v_bits,
            params.goldreich_graph_seed,
            params.noise_refresh_cbd_n,
            params.noise_refresh_hash_key,
        );
        let mut refresh_preimages = Vec::new();
        info!(
            seed_bits = prf_seed_bits,
            wire_count, "AKY24 PRF mask keygen starting PRG/noise-refresh rounds"
        );
        for (round_idx, selected_bit) in public_prf_seed.iter().copied().enumerate() {
            debug!(
                round_idx,
                selected_bit,
                seed_wire_count = seed_wires.len(),
                "AKY24 PRF mask keygen building PRG circuit"
            );
            let started = Instant::now();
            let prg_circuit = build_goldreich_prg_circuit(params, round_idx, 2 * prf_seed_bits);
            debug!(
                round_idx,
                elapsed_ms = started.elapsed().as_millis(),
                "AKY24 PRF mask keygen built PRG circuit"
            );
            debug!(
                round_idx,
                selected_bit,
                seed_wire_count = seed_wires.len(),
                "AKY24 PRF mask keygen evaluating PRG circuit"
            );
            let started = Instant::now();
            let prg_outputs = prg_circuit.eval(
                &params.poly_params,
                one.clone(),
                seed_wires.clone(),
                self.pk_lookup_evaluator.as_ref(),
                self.pk_slot_transfer_evaluator.as_ref().map(|evaluator| {
                    evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>
                }),
                None,
            );
            debug!(
                round_idx,
                prg_output_count = prg_outputs.len(),
                elapsed_ms = started.elapsed().as_millis(),
                "AKY24 PRF mask keygen evaluated PRG circuit"
            );
            let half_start = if selected_bit { prf_seed_bits * wire_count } else { 0 };
            let selected_wires = &prg_outputs[half_start..half_start + prf_seed_bits * wire_count];
            let mut next_seed_wires = Vec::with_capacity(prf_seed_bits * wire_count);
            for (wire_idx, refreshed_input) in selected_wires.iter().enumerate() {
                let refresh_id =
                    aky24_refresh_id(&public_prf_seed, round_idx, selected_bit, wire_idx);
                debug!(
                    round_idx,
                    wire_idx,
                    selected_wire_count = selected_wires.len(),
                    "AKY24 PRF mask keygen noise-refresh preprocess started"
                );
                let started = Instant::now();
                let (a_prime, refresh_keys) = noise_refresher.preprocess(
                    &refresh_id,
                    &one,
                    refreshed_input,
                    &seed_wires,
                    &decryption_key,
                    pk_lookup_evaluator,
                    pk_slot_transfer_evaluator,
                );
                debug!(
                    round_idx,
                    wire_idx,
                    a_prime_slots = a_prime.num_slots(),
                    refresh_key_count = refresh_keys.len(),
                    elapsed_ms = started.elapsed().as_millis(),
                    "AKY24 PRF mask keygen noise-refresh preprocess finished"
                );
                // `preprocess` returns one slotwise refresh target per CRT level. Store matching
                // preimages in the decoder order expected by `online_eval`:
                // `slot_idx * crt_depth + refresh_crt_idx`.
                for slot_idx in 0..a_prime.num_slots() {
                    for (refresh_crt_idx, refresh_crt_key) in refresh_keys.iter().enumerate() {
                        let refresh_key_slot = refresh_crt_key.key(slot_idx);
                        debug!(
                            round_idx,
                            wire_idx,
                            slot_idx,
                            refresh_crt_idx,
                            target_rows = refresh_key_slot.matrix.row_size(),
                            target_cols = refresh_key_slot.matrix.col_size(),
                            "AKY24 PRF mask keygen refresh preimage sampling started"
                        );
                        let started = Instant::now();
                        let preimage = trapdoor_sampler.preimage(
                            &params.poly_params,
                            &msk.b_trapdoor,
                            &msk.b_matrix,
                            &refresh_key_slot.matrix,
                        );
                        debug!(
                            round_idx,
                            wire_idx,
                            slot_idx,
                            refresh_crt_idx,
                            elapsed_ms = started.elapsed().as_millis(),
                            "AKY24 PRF mask keygen refresh preimage sampling finished"
                        );
                        refresh_preimages.push(preimage);
                    }
                }
                next_seed_wires.push(a_prime);
            }
            seed_wires = next_seed_wires;
            info!(round_idx, "AKY24 PRF mask keygen finished refresh round");
        }

        let mask_output_bits = params.prf_mask_output_coeff_bits();
        let mask_prg_circuit =
            build_goldreich_prg_circuit(params, public_prf_seed.len(), mask_output_bits);
        debug!(
            mask_output_bits,
            seed_wire_count = seed_wires.len(),
            "AKY24 PRF mask keygen evaluating final mask expansion PRG circuit"
        );
        let started = Instant::now();
        let mask_output_wires = mask_prg_circuit.eval(
            &params.poly_params,
            one.clone(),
            seed_wires,
            self.pk_lookup_evaluator.as_ref(),
            self.pk_slot_transfer_evaluator
                .as_ref()
                .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>),
            None,
        );
        debug!(
            mask_output_wire_count = mask_output_wires.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "AKY24 PRF mask keygen evaluated final mask expansion PRG circuit"
        );
        assert_eq!(
            mask_output_wires.len(),
            mask_output_bits * wire_count,
            "AKY24 final mask PRG output must be mask_output_bits Ring-GSW ciphertexts"
        );

        // The PRF refresh rounds and final expansion are slotwise, but the current AKY24 output
        // decoder only reads the constant/output slot. Project the final mask decrypt to slot 0
        // before circuit evaluation so we do not build or evaluate unused mask outputs for the
        // other slots.
        let one_slot_one = NaiveBGGPublicKeyVec::new(&params.poly_params, vec![one.key(0)]);
        let mut mask_inputs = Vec::with_capacity(1 + mask_output_wires.len());
        mask_inputs
            .push(NaiveBGGPublicKeyVec::new(&params.poly_params, vec![decryption_key.key(0)]));
        mask_inputs.extend(
            mask_output_wires
                .into_iter()
                .map(|wire| NaiveBGGPublicKeyVec::new(&params.poly_params, vec![wire.key(0)])),
        );
        let mask_circuit = build_prf_mask_circuit(params);
        debug!(
            input_count = mask_inputs.len(),
            "AKY24 PRF mask keygen evaluating final scalar mask decrypt circuit"
        );
        let started = Instant::now();
        let evaluated = mask_circuit.eval(
            &params.poly_params,
            one_slot_one,
            mask_inputs,
            self.pk_lookup_evaluator.as_ref(),
            self.pk_slot_transfer_evaluator
                .as_ref()
                .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>),
            None,
        );
        debug!(
            evaluated_count = evaluated.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "AKY24 PRF mask keygen evaluated final scalar mask decrypt circuit"
        );
        let target = evaluated
            .first()
            .map(|keys| keys.key(0))
            .expect("AKY24 PRF mask final decrypt must produce one output")
            .matrix
            .clone();
        let selector = M::unit_column_vector(&params.poly_params, params.secret_size(), 1);
        let mask_target = target.mul_decompose(&selector);
        info!(refresh_preimages = refresh_preimages.len(), "AKY24 PRF mask keygen finished");
        (mask_target, refresh_preimages)
    }

    fn dec_prf_mask_encoding(
        &self,
        params: &Aky24Params<M, TS::Trapdoor>,
        ct: &Aky24Ciphertext<M>,
        fsk: &Aky24FuncKey<M>,
        one: &NaiveBGGEncodingVec<M>,
        decryption_key: &NaiveBGGEncodingVec<M>,
        mut seed_wires: Vec<NaiveBGGEncodingVec<M>>,
        wire_count: usize,
    ) -> M {
        let prf_seed_bits = params.prf_seed_bits();
        let enc_lookup_evaluator = self
            .enc_lookup_evaluator
            .as_ref()
            .expect("AKY24 PRF mask dec requires an encoding lookup evaluator");
        let enc_slot_transfer_evaluator = self
            .enc_slot_transfer_evaluator
            .as_ref()
            .expect("AKY24 PRF mask dec requires an encoding slot-transfer evaluator");
        let mut refresh_context_circuit = PolyCircuit::new();
        let noise_refresher = NoiseRefresherNaiveVec::<M, NestedRnsPoly<M::P>, SH>::new(
            build_ring_gsw_context(params, &mut refresh_context_circuit),
            prf_seed_bits,
            params.noise_refresh_v_bits,
            params.goldreich_graph_seed,
            params.noise_refresh_cbd_n,
            params.noise_refresh_hash_key,
        );
        let (_, _crt_bits, crt_depth) = params.poly_params.to_crt();
        let decoder_count = prf_seed_bits
            .checked_mul(wire_count)
            .and_then(|count| count.checked_mul(one.num_slots()))
            .and_then(|count| count.checked_mul(crt_depth))
            .and_then(|count| count.checked_mul(128))
            .expect("AKY24 PRF mask decoder count overflow");
        assert_eq!(
            fsk.prf_refresh_preimages.len(),
            decoder_count,
            "AKY24 PRF mask refresh preimage count mismatch"
        );
        let mut preimage_cursor = 0;
        for (round_idx, selected_bit) in fsk.public_prf_seed.iter().copied().enumerate() {
            let prg_circuit = build_goldreich_prg_circuit(params, round_idx, 2 * prf_seed_bits);
            let prg_outputs = prg_circuit.eval(
                &params.poly_params,
                one.clone(),
                seed_wires.clone(),
                self.enc_lookup_evaluator.as_ref(),
                self.enc_slot_transfer_evaluator.as_ref().map(|evaluator| {
                    evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>
                }),
                None,
            );
            let half_start = if selected_bit { prf_seed_bits * wire_count } else { 0 };
            let selected_wires = &prg_outputs[half_start..half_start + prf_seed_bits * wire_count];
            let mut next_seed_wires = Vec::with_capacity(prf_seed_bits * wire_count);
            for (wire_idx, refreshed_input) in selected_wires.iter().enumerate() {
                let refresh_id =
                    aky24_refresh_id(&fsk.public_prf_seed, round_idx, selected_bit, wire_idx);
                let per_wire_decoder_count = one.num_slots() * crt_depth;
                let decoders = fsk.prf_refresh_preimages
                    [preimage_cursor..preimage_cursor + per_wire_decoder_count]
                    .iter()
                    .map(|preimage| ct.c_b.clone() * preimage)
                    .collect::<Vec<_>>();
                preimage_cursor += per_wire_decoder_count;
                next_seed_wires.push(noise_refresher.online_eval(
                    &refresh_id,
                    one,
                    refreshed_input,
                    &seed_wires,
                    decryption_key,
                    &decoders,
                    enc_lookup_evaluator,
                    enc_slot_transfer_evaluator,
                ));
            }
            seed_wires = next_seed_wires;
            info!(round_idx, "AKY24 PRF mask dec finished refresh round");
        }

        let mask_output_bits = params.prf_mask_output_coeff_bits();
        let mask_prg_circuit =
            build_goldreich_prg_circuit(params, fsk.public_prf_seed.len(), mask_output_bits);
        let mask_output_wires = mask_prg_circuit.eval(
            &params.poly_params,
            one.clone(),
            seed_wires,
            self.enc_lookup_evaluator.as_ref(),
            self.enc_slot_transfer_evaluator
                .as_ref()
                .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
            None,
        );
        assert_eq!(
            mask_output_wires.len(),
            mask_output_bits * wire_count,
            "AKY24 final mask PRG output must be mask_output_bits Ring-GSW ciphertexts"
        );

        // Match keygen: after slotwise refresh and final expansion, the scalar PRF mask is needed
        // only for the decoded output slot, so evaluate the final mask decrypt over slot 0 only.
        let one_slot_one = NaiveBGGEncodingVec::new(&params.poly_params, vec![one.encoding(0)]);
        let mut mask_inputs = Vec::with_capacity(1 + mask_output_wires.len());
        mask_inputs
            .push(NaiveBGGEncodingVec::new(&params.poly_params, vec![decryption_key.encoding(0)]));
        mask_inputs.extend(
            mask_output_wires
                .into_iter()
                .map(|wire| NaiveBGGEncodingVec::new(&params.poly_params, vec![wire.encoding(0)])),
        );
        let mask_circuit = build_prf_mask_circuit(params);
        let evaluated = mask_circuit.eval(
            &params.poly_params,
            one_slot_one,
            mask_inputs,
            self.enc_lookup_evaluator.as_ref(),
            self.enc_slot_transfer_evaluator
                .as_ref()
                .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
            None,
        );
        let evaluated_encoding = evaluated
            .first()
            .map(|encoding_vec| encoding_vec.encoding(0))
            .expect("AKY24 PRF mask final decrypt must produce one output encoding");
        let selector = M::unit_column_vector(&params.poly_params, params.secret_size(), 1);
        let mask_message = evaluated_encoding.vector.mul_decompose(&selector);
        info!("AKY24 PRF mask dec finished");
        mask_message
    }
}

impl<M, SH, US, TS, PKPE, PKST, ENCPE, ENCST> FuncEnc
    for Aky24FuncEnc<M, SH, US, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    US: PolyUniformSampler<M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PKPE: PltEvaluator<NaiveBGGPublicKeyVec<M>>,
    PKST: SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>,
    ENCPE: PltEvaluator<NaiveBGGEncodingVec<M>>,
    ENCST: SlotTransferEvaluator<NaiveBGGEncodingVec<M>>,
    TS::Trapdoor: Clone,
{
    type Params = Aky24Params<M, TS::Trapdoor>;
    type EncKey = Aky24EncKey<M>;
    type MasterKey = Aky24MasterKey<M, TS>;
    type Msg = bool;
    type Ciphertext = Aky24Ciphertext<M>;
    type Func = Aky24Func;
    type FuncKey = Aky24FuncKey<M>;
    type Output = Aky24Output;

    fn setup(&self, params: &Self::Params) -> (Self::EncKey, Self::MasterKey) {
        info!(
            n = params.n(),
            gadget_columns = params.gadget_columns(),
            trapdoor_sigma = params.trapdoor_sigma,
            "AKY24 setup started"
        );
        debug!("AKY24 setup loading B trapdoor matrix from params");
        let b_matrix = params.b_matrix.clone();
        let b_trapdoor = params.b_trapdoor.clone();

        let bgg_hash_key = rand::random();

        let enc_key = Aky24EncKey { b_matrix: b_matrix.clone(), bgg_hash_key };
        let master_key = Aky24MasterKey { b_matrix, b_trapdoor, bgg_hash_key };
        info!("AKY24 setup finished");
        (enc_key, master_key)
    }

    fn enc(
        &self,
        params: &Self::Params,
        enc_key: &Self::EncKey,
        msg: &Self::Msg,
    ) -> Self::Ciphertext {
        info!(msg = *msg, "AKY24 enc started");
        let uniform_sampler = US::new();
        debug!("AKY24 enc sampling secret");
        let secret = uniform_sampler.sample_poly(&params.poly_params, &DistType::TernaryDist);
        let minus_one =
            M::P::const_zero(&params.poly_params) - M::P::const_one(&params.poly_params);
        let secret_vec =
            M::from_poly_vec_row(&params.poly_params, vec![secret.clone(), minus_one.clone()]);
        debug!(
            b_cols = enc_key.b_matrix.col_size(),
            b_error_sigma = ?params.b_error_sigma,
            "AKY24 enc computing c_b"
        );
        let b_error = match params.b_error_sigma {
            Some(sigma) => uniform_sampler.sample_uniform(
                &params.poly_params,
                1,
                enc_key.b_matrix.col_size(),
                DistType::GaussDist { sigma },
            ),
            None => M::zero(&params.poly_params, 1, enc_key.b_matrix.col_size()),
        };
        let c_b = secret_vec * &enc_key.b_matrix + b_error;

        let fhe_decryption_key_poly =
            uniform_sampler.sample_poly(&params.poly_params, &DistType::TernaryDist);
        debug!("AKY24 enc sampled FHE decryption key");
        let native_fhe_decryption_key = DCRTPoly::from_biguints(
            &params.native_poly_params,
            &fhe_decryption_key_poly.coeffs_biguints(),
        );
        info!(
            ring_gsw_width = params.ring_gsw_width,
            public_key_error_sigma = ?params.ring_gsw_public_key_error_sigma,
            "AKY24 enc sampling native Ring-GSW public key"
        );
        let ring_gsw_public_key = sample_public_key(
            &params.native_poly_params,
            params.ring_gsw_width,
            &native_fhe_decryption_key,
            enc_key.bgg_hash_key,
            b"aky24_fe_ring_gsw_public_key",
            params.ring_gsw_public_key_error_sigma,
        );
        info!("AKY24 enc encrypting plaintext bit with native Ring-GSW");
        let native_message_ciphertext = encrypt_plaintext_bit(
            &params.native_poly_params,
            params.ring_gsw_context.as_ref(),
            &ring_gsw_public_key,
            *msg,
        );
        #[cfg(test)]
        {
            let native_decrypted = decrypt_ciphertext(
                &params.native_poly_params,
                params.ring_gsw_context.as_ref(),
                &native_message_ciphertext,
                &native_fhe_decryption_key,
                2,
            );
            let native_q = active_q_modulus(params.ring_gsw_context.as_ref());
            let native_quarter_q = &native_q / 4u32;
            let native_three_quarter_q = &native_quarter_q * 3u32;
            let native_decrypted_coeffs = native_decrypted.coeffs_biguints();
            let native_decoded_coeffs = native_decrypted
                .coeffs_biguints()
                .into_iter()
                .map(|coeff| coeff > native_quarter_q && coeff < native_three_quarter_q)
                .collect::<Vec<_>>();
            info!(
                native_q = %native_q,
                ?native_decrypted_coeffs,
                ?native_decoded_coeffs,
                "AKY24 enc native Ring-GSW decrypt check before BGG encoding"
            );
        }
        info!("AKY24 enc converting native Ring-GSW ciphertext into circuit inputs");
        let message_ciphertext_inputs = ciphertext_inputs_from_native::<M::P>(
            &params.poly_params,
            params.ring_gsw_context.as_ref(),
            &native_message_ciphertext,
            params.ring_gsw_level_offset,
            params.ring_gsw_enable_levels,
        );
        let num_slots = message_ciphertext_inputs
            .first()
            .map(PolyVec::len)
            .expect("Ring-GSW ciphertext input conversion must produce at least one input");
        info!(
            num_slots,
            plaintext_input_count = 1 + message_ciphertext_inputs.len(),
            "AKY24 enc converted Ring-GSW ciphertext inputs"
        );
        let prf_seed_bits = params.prf_seed_bits();
        let mut plaintext_inputs =
            Vec::with_capacity(1 + message_ciphertext_inputs.len() * (1 + prf_seed_bits));
        plaintext_inputs.push(PolyVec::new(vec![fhe_decryption_key_poly; num_slots]));
        plaintext_inputs.extend(message_ciphertext_inputs);
        info!(prf_seed_bits, "AKY24 enc sampling private PRF seed ciphertexts");
        for seed_bit_idx in 0..prf_seed_bits {
            let seed_bit = rand::random::<bool>();
            debug!(seed_bit_idx, seed_bit, "AKY24 enc encrypting private PRF seed bit");
            let native_seed_ciphertext = encrypt_plaintext_bit(
                &params.native_poly_params,
                params.ring_gsw_context.as_ref(),
                &ring_gsw_public_key,
                seed_bit,
            );
            plaintext_inputs.extend(ciphertext_inputs_from_native::<M::P>(
                &params.poly_params,
                params.ring_gsw_context.as_ref(),
                &native_seed_ciphertext,
                params.ring_gsw_level_offset,
                params.ring_gsw_enable_levels,
            ));
        }
        info!("AKY24 enc sampling BGG public keys");
        let bgg_public_keys = sample_bgg_public_keys::<M, SH>(params, enc_key.bgg_hash_key);

        info!(
            plaintext_input_count = plaintext_inputs.len(),
            num_slots, "AKY24 enc sampling BGG encodings"
        );
        let encoding_sampler = NaiveBGGEncodingVecSampler::<US>::new(
            &params.poly_params,
            &[secret, minus_one],
            params.encoding_error_sigma,
            num_slots,
        );
        let encodings =
            encoding_sampler.sample(&params.poly_params, &bgg_public_keys, &plaintext_inputs);

        info!(encoding_count = encodings.len(), "AKY24 enc finished");
        Aky24Ciphertext { c_b, encodings }
    }

    fn keygen(
        &self,
        params: &Self::Params,
        msk: &Self::MasterKey,
        func: &Self::Func,
    ) -> Self::FuncKey {
        info!(?func, "AKY24 keygen started");
        info!("AKY24 keygen building function circuit");
        let circuit = build_func_circuit(params, func);
        info!("AKY24 keygen sampling BGG public keys");
        let bgg_public_keys = sample_bgg_public_keys::<M, SH>(params, msk.bgg_hash_key);
        let one = bgg_public_keys[0].clone();
        let decryption_key = bgg_public_keys[1].clone();
        let prf_seed_bits = params.prf_seed_bits();
        let wire_count = public_key_wire_count(&bgg_public_keys, prf_seed_bits);
        let message_start = 2;
        let seed_start = message_start + wire_count;
        let message_wires = bgg_public_keys[message_start..seed_start].to_vec();
        let seed_wires =
            bgg_public_keys[seed_start..seed_start + prf_seed_bits * wire_count].to_vec();
        let function_decryption_key =
            NaiveBGGPublicKeyVec::new(&params.poly_params, vec![decryption_key.key(0)]);
        let pk_slot_transfer_evaluator = self
            .pk_slot_transfer_evaluator
            .as_ref()
            .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>);
        let mut function_inputs = Vec::with_capacity(1 + message_wires.len());
        function_inputs.push(function_decryption_key);
        function_inputs.extend(message_wires);
        info!(
            input_count = function_inputs.len(),
            "AKY24 keygen evaluating function circuit over BGG public keys"
        );
        let evaluated_public_keys = circuit.eval(
            &params.poly_params,
            one.clone(),
            function_inputs,
            self.pk_lookup_evaluator.as_ref(),
            pk_slot_transfer_evaluator,
            None,
        );
        let evaluated_target = evaluated_public_keys
            .first()
            .map(|keys| keys.key(0))
            .expect("AKY24 keygen circuit evaluation must produce an evaluated target")
            .matrix
            .clone();
        info!(
            evaluated_rows = evaluated_target.row_size(),
            evaluated_cols = evaluated_target.col_size(),
            "AKY24 keygen circuit evaluation finished"
        );
        let selector = M::unit_column_vector(&params.poly_params, params.secret_size(), 1);
        let preimage_target = evaluated_target.mul_decompose(&selector);
        debug!(
            target_rows = preimage_target.row_size(),
            target_cols = preimage_target.col_size(),
            "AKY24 keygen applied G^-1((0,1)^T) to evaluated public key"
        );
        let public_prf_seed = std::array::from_fn(|_| rand::random::<bool>());
        let trapdoor_sampler = TS::new(&params.poly_params, params.trapdoor_sigma);
        let (mask_preimage_target, prf_refresh_preimages) = self.keygen_prf_mask_target(
            params,
            msk,
            &trapdoor_sampler,
            public_prf_seed,
            one,
            decryption_key,
            seed_wires,
            wire_count,
        );
        let combined_preimage_target = preimage_target + mask_preimage_target;
        info!("AKY24 keygen sampling trapdoor preimage for function output plus PRF mask");
        let preimage = trapdoor_sampler.preimage(
            &params.poly_params,
            &msk.b_trapdoor,
            &msk.b_matrix,
            &combined_preimage_target,
        );
        info!(?func, prf_refresh_preimages = prf_refresh_preimages.len(), "AKY24 keygen finished");
        Aky24FuncKey { func: *func, preimage, public_prf_seed, prf_refresh_preimages }
    }

    fn dec(
        &self,
        params: &Self::Params,
        ct: &Self::Ciphertext,
        fsk: &Self::FuncKey,
    ) -> Self::Output {
        info!(func = ?fsk.func, encoding_count = ct.encodings.len(), "AKY24 dec started");
        let one = ct.encodings.first().expect("AKY24 ciphertext must include a one encoding");
        let decryption_key =
            ct.encodings.get(1).expect("AKY24 ciphertext must include a decryption-key encoding");
        let prf_seed_bits = params.prf_seed_bits();
        let wire_count = encoding_wire_count(&ct.encodings, prf_seed_bits);
        let message_start = 2;
        let seed_start = message_start + wire_count;
        let message_wires = ct.encodings[message_start..seed_start].to_vec();
        let seed_wires = ct.encodings[seed_start..seed_start + prf_seed_bits * wire_count].to_vec();
        let function_decryption_key =
            NaiveBGGEncodingVec::new(&params.poly_params, vec![decryption_key.encoding(0)]);
        let mut inputs = Vec::with_capacity(1 + message_wires.len());
        inputs.push(function_decryption_key);
        inputs.extend(message_wires);
        info!("AKY24 dec building function circuit");
        let circuit = build_func_circuit(params, &fsk.func);
        info!(
            input_count = inputs.len(),
            "AKY24 dec evaluating function circuit over BGG encodings"
        );
        let evaluated_encodings = circuit.eval(
            &params.poly_params,
            one.clone(),
            inputs,
            self.enc_lookup_evaluator.as_ref(),
            self.enc_slot_transfer_evaluator
                .as_ref()
                .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<NaiveBGGEncodingVec<M>>),
            None,
        );

        match fsk.func {
            Aky24Func::DebugIdentity => {
                info!("AKY24 dec circuit evaluation finished; decoding DebugIdentity output");
                let evaluated_encoding = evaluated_encodings
                    .first()
                    .map(|encoding_vec| encoding_vec.encoding(0))
                    .expect("AKY24 DebugIdentity evaluation must produce one output encoding");
                let selector = M::unit_column_vector(&params.poly_params, params.secret_size(), 1);
                let evaluated_message = evaluated_encoding.vector.mul_decompose(&selector);
                let mask_message = self.dec_prf_mask_encoding(
                    params,
                    ct,
                    fsk,
                    one,
                    decryption_key,
                    seed_wires,
                    wire_count,
                );
                let combined_message = evaluated_message + mask_message;
                let noisy_plaintext = combined_message - &(ct.c_b.clone() * &fsk.preimage);
                assert_eq!(
                    noisy_plaintext.size(),
                    (1, 1),
                    "AKY24 DebugIdentity decoding expects a 1x1 noisy plaintext matrix"
                );
                let q: Arc<BigUint> = params.poly_params.modulus().into();
                let q_ref = q.as_ref();
                let quarter_q = q_ref / 4u32;
                let three_quarter_q = (&quarter_q) * 3u32;
                let decoded_coeffs = noisy_plaintext
                    .entry(0, 0)
                    .coeffs_biguints()
                    .into_iter()
                    .map(|coeff| coeff > quarter_q && coeff < three_quarter_q)
                    .collect::<Vec<_>>();
                let decoded_coeff_idx = 0;
                let decrypted = decoded_coeffs[decoded_coeff_idx];
                info!(decrypted, decoded_coeff_idx, ?decoded_coeffs, "AKY24 dec finished");
                Aky24Output::DebugIdentity { decrypted }
            }
        }
    }
}

fn build_func_circuit<M, TD>(params: &Aky24Params<M, TD>, func: &Aky24Func) -> PolyCircuit<M::P>
where
    M: PolyMatrix,
    M::P: 'static,
{
    info!(?func, "AKY24 build_func_circuit started");
    let mut circuit = PolyCircuit::new();
    match func {
        Aky24Func::DebugIdentity => {
            debug!("AKY24 build_func_circuit setting up nested-RNS context");
            let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
                &mut circuit,
                &params.poly_params,
                params.ring_gsw_context.p_moduli_bits,
                params.ring_gsw_context.max_unreduced_muls,
                params.ring_gsw_context.scale,
                false,
                params.ring_gsw_enable_levels,
            ));
            debug!("AKY24 build_func_circuit setting up Ring-GSW context");
            let ring_gsw_context = Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
                &mut circuit,
                &params.poly_params,
                params.n(),
                nested_rns_context,
                params.ring_gsw_enable_levels,
                Some(params.ring_gsw_level_offset),
            ));
            let fhe_decryption_key = circuit.input(1).at(0).as_single_wire();
            let ciphertext =
                RingGswCiphertext::input(ring_gsw_context, Some(BigUint::from(1u64)), &mut circuit);
            let decrypted =
                ciphertext.decrypt::<M>(fhe_decryption_key, BigUint::from(2u64), &mut circuit);
            circuit.output(vec![decrypted]);
        }
    }
    info!(?func, "AKY24 build_func_circuit finished");
    circuit
}

fn build_ring_gsw_context<M, TD>(
    params: &Aky24Params<M, TD>,
    circuit: &mut PolyCircuit<M::P>,
) -> Arc<NestedRnsRingGswContext<M::P>>
where
    M: PolyMatrix,
    M::P: 'static,
{
    let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
        circuit,
        &params.poly_params,
        params.ring_gsw_context.p_moduli_bits,
        params.ring_gsw_context.max_unreduced_muls,
        params.ring_gsw_context.scale,
        false,
        params.ring_gsw_enable_levels,
    ));
    Arc::new(NestedRnsRingGswContext::<M::P>::from_arith_context(
        circuit,
        &params.poly_params,
        params.n(),
        nested_rns_context,
        params.ring_gsw_enable_levels,
        Some(params.ring_gsw_level_offset),
    ))
}

fn build_goldreich_prg_circuit<M, TD>(
    params: &Aky24Params<M, TD>,
    round_idx: usize,
    output_bits: usize,
) -> PolyCircuit<M::P>
where
    M: PolyMatrix,
    M::P: 'static,
{
    let prf_seed_bits = params.prf_seed_bits();
    assert!(
        prf_seed_bits >= 5,
        "AKY24 Goldreich PRF seed bit length must be at least 5 for the Goldreich graph"
    );
    assert!(output_bits > 0, "AKY24 Goldreich PRF output bit length must be positive");
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_ring_gsw_context(params, &mut circuit);
    let seed_ciphertexts = (0..prf_seed_bits)
        .map(|_| {
            RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let goldreich = GoldreichFhePrg::setup(
        &mut circuit,
        ring_gsw_context,
        prf_seed_bits,
        output_bits,
        derive_goldreich_graph_seed(params.goldreich_graph_seed, round_idx),
    );
    let outputs = goldreich.evaluate_uniform(&seed_ciphertexts, &mut circuit);
    circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
    circuit
}

fn build_prf_mask_circuit<M, TD>(params: &Aky24Params<M, TD>) -> PolyCircuit<M::P>
where
    M: PolyMatrix,
    M::P: 'static,
{
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_ring_gsw_context(params, &mut circuit);
    let fhe_decryption_key = circuit.input(1).at(0).as_single_wire();
    let q: Arc<BigUint> = params.poly_params.modulus().into();
    let mask_output_bits = params.prf_mask_output_coeff_bits();
    let plaintext_moduli = mask_plaintext_moduli_from_full_modulus(q.as_ref(), mask_output_bits);
    let encrypted_bits = (0..mask_output_bits)
        .map(|_| {
            RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let decrypted = decrypt_bit_decomposed_scalar::<M::P, NestedRnsPoly<M::P>, M>(
        &mut circuit,
        &encrypted_bits,
        fhe_decryption_key,
        &plaintext_moduli,
    );
    circuit.output(vec![decrypted]);
    circuit
}

fn derive_goldreich_graph_seed(base_seed: [u8; 32], round_idx: usize) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(b"AKY24GoldreichPrfGraph/v1");
    hasher.update(base_seed);
    hasher.update(round_idx.to_le_bytes());
    hasher.finalize().into()
}

fn aky24_refresh_id(
    public_seed: &[bool; 128],
    round_idx: usize,
    selected_bit: bool,
    wire_idx: usize,
) -> Vec<u8> {
    let mut id = Vec::with_capacity(16 + 16 + public_seed.len());
    id.extend_from_slice(b"AKY24PrfRefresh/v1");
    id.extend(public_seed.iter().map(|bit| u8::from(*bit)));
    id.extend_from_slice(&round_idx.to_le_bytes());
    id.push(u8::from(selected_bit));
    id.extend_from_slice(&wire_idx.to_le_bytes());
    id
}

fn encoding_wire_count<M>(encodings: &[NaiveBGGEncodingVec<M>], seed_bits: usize) -> usize
where
    M: PolyMatrix,
{
    assert!(
        encodings.len() >= 2,
        "AKY24 ciphertext must contain one encoding and decryption-key encoding"
    );
    let payload_count = encodings.len() - 2;
    assert!(
        payload_count.is_multiple_of(seed_bits + 1),
        "AKY24 ciphertext payload count must be one message ciphertext plus seed ciphertexts"
    );
    payload_count / (seed_bits + 1)
}

fn public_key_wire_count<M>(keys: &[NaiveBGGPublicKeyVec<M>], seed_bits: usize) -> usize
where
    M: PolyMatrix,
{
    assert!(keys.len() >= 2, "AKY24 public keys must contain one and decryption-key keys");
    let payload_count = keys.len() - 2;
    assert!(
        payload_count.is_multiple_of(seed_bits + 1),
        "AKY24 public-key payload count must be one message ciphertext plus seed ciphertexts"
    );
    payload_count / (seed_bits + 1)
}

pub struct NoCircuitEvaluator;

impl<E: Evaluable> PltEvaluator<E> for NoCircuitEvaluator {
    fn public_lookup(
        &self,
        _params: &E::Params,
        _plt: &PublicLut<E::P>,
        _one: &E,
        _input: &E,
        _gate_id: GateId,
        _lut_id: usize,
    ) -> E {
        panic!("AKY24 DebugIdentity circuit does not support public lookup gates")
    }
}

impl<E: Evaluable> SlotTransferEvaluator<E> for NoCircuitEvaluator {
    fn slot_transfer(
        &self,
        _params: &E::Params,
        _input: &E,
        _src_slots: &[(u32, Option<u32>)],
        _gate_id: GateId,
    ) -> E {
        panic!("AKY24 DebugIdentity circuit does not support slot-transfer gates")
    }
}

fn sample_bgg_public_keys<M, SH>(
    params: &Aky24Params<M, impl Send + Sync>,
    bgg_hash_key: [u8; 32],
) -> Vec<NaiveBGGPublicKeyVec<M>>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    info!("AKY24 sample_bgg_public_keys started");
    let native_uniform_sampler = crate::sampler::uniform::DCRTPolyUniformSampler::new();
    let shape_fhe_decryption_key =
        native_uniform_sampler.sample_poly(&params.native_poly_params, &DistType::TernaryDist);
    debug!("AKY24 sample_bgg_public_keys sampling shape native Ring-GSW public key");
    let shape_ring_gsw_public_key = sample_public_key(
        &params.native_poly_params,
        params.ring_gsw_width,
        &shape_fhe_decryption_key,
        bgg_hash_key,
        b"aky24_fe_shape_ring_gsw_public_key",
        None,
    );
    debug!("AKY24 sample_bgg_public_keys encrypting shape Ring-GSW ciphertext");
    let shape_native_ciphertext = encrypt_plaintext_bit(
        &params.native_poly_params,
        params.ring_gsw_context.as_ref(),
        &shape_ring_gsw_public_key,
        true,
    );
    debug!("AKY24 sample_bgg_public_keys converting shape ciphertext inputs");
    let shape_ciphertext_inputs = ciphertext_inputs_from_native::<M::P>(
        &params.poly_params,
        params.ring_gsw_context.as_ref(),
        &shape_native_ciphertext,
        params.ring_gsw_level_offset,
        params.ring_gsw_enable_levels,
    );
    let num_slots = shape_ciphertext_inputs
        .first()
        .map(PolyVec::len)
        .expect("Ring-GSW ciphertext input conversion must produce at least one input");
    let ciphertext_input_count = shape_ciphertext_inputs.len();
    let ring_gsw_ciphertext_count = 1 + params.prf_seed_bits();
    info!(
        num_slots,
        ciphertext_input_count,
        ring_gsw_ciphertext_count,
        "AKY24 sample_bgg_public_keys inferred BGG vector shape"
    );
    let reveal_plaintexts = std::iter::once(false)
        .chain(std::iter::repeat_n(true, ciphertext_input_count * ring_gsw_ciphertext_count))
        .collect::<Vec<_>>();
    let public_key_sampler = NaiveBGGPublicKeyVecSampler::<[u8; 32], SH>::new(
        bgg_hash_key,
        params.secret_size(),
        num_slots,
    );
    let public_keys =
        public_key_sampler.sample(&params.poly_params, &params.bgg_tag, &reveal_plaintexts);
    info!(public_key_count = public_keys.len(), "AKY24 sample_bgg_public_keys finished");
    public_keys
}

pub fn build_ring_gsw_decrypt_circuit<P, A, M>(
    ciphertext: &RingGswCiphertext<P, A>,
) -> PolyCircuit<P>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    let mut circuit = PolyCircuit::new();
    let secret_key = circuit.input(1).at(0).as_single_wire();
    let decrypted = ciphertext.decrypt::<M>(secret_key, BigUint::from(2u64), &mut circuit);
    circuit.output(vec![decrypted]);
    circuit
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        gadgets::arith::ModularArithmeticContext,
        lookup::lwe::{
            LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
            NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator,
        },
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
        sampler::{
            PolyTrapdoorSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::NaiveBGGVecSlotTransferEvaluator,
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};
    use tracing_subscriber::prelude::*;

    type TestFuncEnc = Aky24FuncEnc<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyTrapdoorSampler,
        NaiveLWEBGGPublicKeyVecPltEvaluator<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >,
        NaiveBGGVecSlotTransferEvaluator,
        NaiveLWEBGGEncodingVecPltEvaluator<GpuDCRTPolyMatrix, GpuDCRTPolyHashSampler<Keccak256>>,
        NaiveBGGVecSlotTransferEvaluator,
    >;

    fn gpu_params_from_cpu(params: &DCRTPolyParams) -> GpuDCRTPolyParams {
        let (moduli, _, _) = params.to_crt();
        GpuDCRTPolyParams::new(params.ring_dimension(), moduli, params.base_bits())
    }

    fn prepare_clean_storage(dir_path: &str) {
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_aky24_debug_identity_decrypts_random_bit() {
        let log_filter = tracing_subscriber::filter::Targets::new()
            .with_target("mxx::func_enc::aky24", tracing::Level::DEBUG)
            .with_target("mxx::func_enc::aky24::tests", tracing::Level::DEBUG);
        let _ = tracing_subscriber::registry()
            .with(log_filter)
            .with(tracing_subscriber::fmt::layer())
            .try_init();
        info!("AKY24 GPU DebugIdentity roundtrip test started");
        let _storage_lock = storage_test_lock().await;
        let ring_dim = 2;
        let active_levels = 1;
        let crt_bits = 10;
        let base_bits = (crt_bits / 2) as u32;
        let p_moduli_bits = 5;
        let max_unreduced_muls = 2;
        let nested_rns_scale = 1 << 8;
        info!(
            ring_dim,
            active_levels,
            crt_bits,
            base_bits,
            p_moduli_bits,
            max_unreduced_muls,
            nested_rns_scale,
            "AKY24 GPU test constructing native and GPU params"
        );
        let native_poly_params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, base_bits);
        let poly_params = gpu_params_from_cpu(&native_poly_params);
        let q: Arc<BigUint> = poly_params.modulus().into();
        let prf_mask_output_bound = q.as_ref() / 4u32;
        let prf_mask_output_coeff_bits = prf_mask_output_bound.bits().saturating_sub(1) as usize;
        assert!(
            prf_mask_output_coeff_bits > 0,
            "AKY24 test requires a positive PRF mask output coefficient bit size"
        );
        let max_prf_mask = (BigUint::from(1u64) << prf_mask_output_coeff_bits) - 1u32;
        assert!(
            max_prf_mask <= prf_mask_output_bound,
            "AKY24 test PRF mask range must stay below floor(q/4)"
        );
        let prf_seed_bits = prf_mask_output_coeff_bits.max(5);
        info!(
            prf_seed_bits,
            prf_mask_output_coeff_bits,
            ?prf_mask_output_bound,
            "AKY24 GPU test configured PRF mask output coefficient range"
        );
        let mut setup_circuit = PolyCircuit::<GpuDCRTPoly>::new();
        info!("AKY24 GPU test setting up nested-RNS context");
        let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &poly_params,
            p_moduli_bits,
            max_unreduced_muls,
            nested_rns_scale,
            false,
            Some(active_levels),
        ));
        let ring_gsw_level_offset = 0;
        let ring_gsw_enable_levels = Some(active_levels);
        info!("AKY24 GPU test computing Ring-GSW width");
        let ring_gsw_width = 2 *
            <NestedRnsPolyContext as ModularArithmeticContext<GpuDCRTPoly>>::gadget_len(
                ring_gsw_context.as_ref(),
                ring_gsw_enable_levels,
                Some(ring_gsw_level_offset),
            );
        info!("AKY24 GPU test sampling setup B trapdoor material");
        let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&poly_params, 4.578);
        let (b_trapdoor, b_matrix) = trapdoor_sampler.trapdoor(&poly_params, 2);
        info!(ring_gsw_width, "AKY24 GPU test building AKY24 params");
        let params = Aky24Params::<GpuDCRTPolyMatrix, _>::new(
            poly_params.clone(),
            native_poly_params,
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            Some(0.0),
            b"aky24_test".to_vec(),
            4.578,
            None,
            b_matrix,
            b_trapdoor,
            prf_seed_bits,
            prf_mask_output_coeff_bits,
            [0x42; 32],
            1,
            1,
            [0x24; 32],
        );
        let mut scheme = TestFuncEnc::new(None, None, None, None);
        info!("AKY24 GPU test running setup");
        let (enc_key, master_key) = scheme.setup(&params);
        let dir_path = "test_data/test_aky24_debug_identity_decrypts_random_bit";
        info!(dir_path, "AKY24 GPU test preparing lookup storage");
        prepare_clean_storage(dir_path);
        info!("AKY24 GPU test installing public-key lookup evaluator");
        scheme.pk_lookup_evaluator =
            Some(NaiveLWEBGGPublicKeyVecPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
                _,
            >::new(
                enc_key.bgg_hash_key,
                trapdoor_sampler,
                Arc::new(master_key.b_matrix.clone()),
                Arc::new(master_key.b_trapdoor.clone()),
                dir_path.into(),
            )));
        scheme.pk_slot_transfer_evaluator = Some(NaiveBGGVecSlotTransferEvaluator::new());
        scheme.enc_slot_transfer_evaluator = Some(NaiveBGGVecSlotTransferEvaluator::new());
        let msg = rand::random::<bool>();
        info!(msg, "AKY24 GPU test encrypting random bit");
        let ciphertext = scheme.enc(&params, &enc_key, &msg);
        info!("AKY24 GPU test installing encoding lookup evaluator");
        scheme.enc_lookup_evaluator =
            Some(NaiveLWEBGGEncodingVecPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
            >::new(
                enc_key.bgg_hash_key,
                dir_path.into(),
                ciphertext.c_b.clone(),
            )));
        info!("AKY24 GPU test running keygen");
        let func_key = scheme.keygen(&params, &master_key, &Aky24Func::DebugIdentity);
        info!("AKY24 GPU test sampling lookup auxiliary matrices");
        scheme
            .pk_lookup_evaluator
            .as_ref()
            .expect("AKY24 test must install a public-key lookup evaluator")
            .sample_aux_matrices(&params.poly_params);
        info!("AKY24 GPU test waiting for lookup auxiliary matrix writes");
        wait_for_all_writes(Path::new(dir_path).to_path_buf()).await.unwrap();
        info!("AKY24 GPU test running dec");
        let output = scheme.dec(&params, &ciphertext, &func_key);

        let Aky24Output::DebugIdentity { decrypted } = output else {
            panic!("AKY24 DebugIdentity test must return DebugIdentity output");
        };
        info!(msg, decrypted, "AKY24 GPU DebugIdentity roundtrip test finished");
        assert_eq!(decrypted, msg);
    }
}
