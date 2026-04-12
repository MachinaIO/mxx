use crate::{
    circuit::{PolyCircuit, SubCircuitDiskStorage, evaluable::PolyVec, gate::GateId},
    gadgets::{
        arith::{
            NestedRnsPoly, NestedRnsPolyContext, nested_rns_gadget_decomposed,
            nested_rns_gadget_vector,
        },
        conv_mul::negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit,
        ntt::encode_nested_rns_poly_vec_with_offset,
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyHashSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        uniform::DCRTPolyUniformSampler,
    },
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm},
};
use bigdecimal::BigDecimal;
use dashmap::DashMap;
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};
use tracing::debug;

const MUL_TOP_LEVEL_BATCH_COLUMNS: usize = 8;
const MUL_TOP_LEVEL_SUPER_BATCH_GROUPS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RingGswAddEntryPlanKey {
    pre_full_reduce: bool,
    reduce_levels: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RingGswSubEntryPlanKey {
    pre_full_reduce: bool,
    reduce_levels: Vec<bool>,
    trace_multipliers: Vec<BigUint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RingGswEntryOutputMetadata {
    max_plaintexts: Vec<BigUint>,
    p_max_traces: Vec<BigUint>,
}

fn validate_num_slots<P: Poly>(params: &P::Params, num_slots: usize) {
    assert!(num_slots > 0, "num_slots must be positive");
    assert!(
        num_slots <= params.ring_dimension() as usize,
        "num_slots {} exceeds ring dimension {}",
        num_slots,
        params.ring_dimension()
    );
}

fn reduce_nested_rns_terms_pairwise<P, F>(
    mut current_layer: Vec<NestedRnsPoly<P>>,
    circuit: &mut PolyCircuit<P>,
    mut combine: F,
) -> NestedRnsPoly<P>
where
    P: Poly,
    F: FnMut(&NestedRnsPoly<P>, &NestedRnsPoly<P>, &mut PolyCircuit<P>) -> NestedRnsPoly<P>,
{
    assert!(
        !current_layer.is_empty(),
        "pairwise reduction requires at least one NestedRnsPoly term"
    );
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity((current_layer.len() + 1) / 2);
        let mut iter = current_layer.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next_layer.push(combine(&left, &right, circuit));
            } else {
                next_layer.push(left);
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("pairwise reduction must leave one term")
}

fn flatten_nested_rns_entries<P: Poly>(entries: &[NestedRnsPoly<P>]) -> Vec<GateId> {
    entries
        .par_iter()
        .flat_map_iter(|entry| entry.inner.iter().flat_map(|level| level.iter().copied()))
        .collect()
}

fn flatten_nested_rns_q_level_rows_for_gadget_terms<P: Poly>(
    entries: &[NestedRnsPoly<P>],
    gadget_len: usize,
    chunk_width: usize,
) -> Vec<GateId> {
    assert!(
        !entries.is_empty(),
        "q-level row flattening requires at least one NestedRnsPoly entry"
    );
    assert_eq!(
        entries.len() % gadget_len,
        0,
        "Ring-GSW row width {} must be a multiple of gadget_len {}",
        entries.len(),
        gadget_len
    );
    entries
        .par_iter()
        .enumerate()
        .flat_map_iter(|(entry_offset, entry)| {
            let sparse_q_idx = (entry_offset % gadget_len) / chunk_width;
            entry.inner[sparse_q_idx].iter().copied()
        })
        .collect()
}

fn nested_rns_from_flat_outputs<P: Poly>(
    template: &NestedRnsPoly<P>,
    outputs: &[GateId],
    max_plaintexts: Vec<BigUint>,
    p_max_traces: Vec<BigUint>,
) -> NestedRnsPoly<P> {
    let levels = template.active_q_moduli().len();
    let p_moduli_depth = template.ctx.p_moduli.len();
    assert_eq!(
        outputs.len(),
        levels * p_moduli_depth,
        "flattened Ring-GSW nested-RNS output size must match active_levels * p_moduli_depth"
    );
    NestedRnsPoly::new(
        template.ctx.clone(),
        outputs.chunks(p_moduli_depth).map(|row| row.to_vec()).collect::<Vec<_>>(),
        Some(template.level_offset),
        template.enable_levels,
        max_plaintexts,
    )
    .with_p_max_traces(p_max_traces)
}

fn sparse_nested_rns_output_template<P: Poly>(
    ctx: Arc<NestedRnsPolyContext>,
    active_levels: usize,
    level_offset: usize,
    circuit: &mut PolyCircuit<P>,
) -> NestedRnsPoly<P> {
    let zero_gate = circuit.const_zero_gate();
    let inner = vec![vec![zero_gate; ctx.p_moduli.len()]; active_levels];
    NestedRnsPoly::new(
        ctx,
        inner,
        Some(level_offset),
        Some(active_levels),
        vec![BigUint::ZERO; active_levels],
    )
    .with_p_max_traces(vec![BigUint::ZERO; active_levels])
}

fn raw_decomposed_term_bound(template_ctx: &NestedRnsPolyContext, term_idx: usize) -> BigUint {
    if term_idx < template_ctx.p_moduli.len() {
        BigUint::from(template_ctx.p_moduli[term_idx] - 1)
    } else {
        BigUint::from(
            u64::try_from(template_ctx.p_moduli.len()).expect("p_moduli length must fit in u64"),
        )
    }
}

fn raw_decomposed_entry_metadata(
    template_ctx: &NestedRnsPolyContext,
    active_levels: usize,
    sparse_q_idx: usize,
    term_idx: usize,
) -> (Vec<BigUint>, Vec<BigUint>) {
    let term_bound = raw_decomposed_term_bound(template_ctx, term_idx);
    let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
    let mut p_max_traces = vec![BigUint::ZERO; active_levels];
    max_plaintexts[sparse_q_idx] = term_bound.clone();
    p_max_traces[sparse_q_idx] = term_bound;
    (max_plaintexts, p_max_traces)
}

fn add_nested_rns_metadata(
    ctx: &NestedRnsPolyContext,
    active_levels: usize,
    level_offset: usize,
    left_max_plaintexts: &[BigUint],
    left_p_max_traces: &[BigUint],
    right_max_plaintexts: &[BigUint],
    right_p_max_traces: &[BigUint],
) -> (Vec<BigUint>, Vec<BigUint>) {
    assert_eq!(
        left_max_plaintexts.len(),
        active_levels,
        "left metadata depth must match active levels"
    );
    assert_eq!(left_p_max_traces.len(), active_levels, "left trace depth must match active levels");
    assert_eq!(
        right_max_plaintexts.len(),
        active_levels,
        "right metadata depth must match active levels"
    );
    assert_eq!(
        right_p_max_traces.len(),
        active_levels,
        "right trace depth must match active levels"
    );
    let reduced_trace = ctx.reduced_p_max_trace();
    let mut left_bounds = left_max_plaintexts.to_vec();
    let mut right_bounds = right_max_plaintexts.to_vec();
    let mut left_traces = left_p_max_traces.to_vec();
    let mut right_traces = right_p_max_traces.to_vec();

    let predicted_bounds = left_bounds
        .par_iter()
        .zip(right_bounds.par_iter())
        .map(|(left_bound, right_bound)| left_bound + right_bound)
        .collect::<Vec<_>>();
    if predicted_bounds.iter().any(|bound| bound >= ctx.p_full_ref()) {
        let reduced_bounds =
            ctx.full_reduce_max_plaintexts[level_offset..level_offset + active_levels].to_vec();
        left_bounds = reduced_bounds.clone();
        right_bounds = reduced_bounds;
        left_traces = vec![reduced_trace.clone(); active_levels];
        right_traces = vec![reduced_trace.clone(); active_levels];
    }

    let trace_needs_reduce = left_traces
        .par_iter()
        .zip(right_traces.par_iter())
        .map(|(left_trace, right_trace)| {
            left_trace + right_trace >= *ctx.lut_mod_p_max_map_size_ref()
        })
        .collect::<Vec<_>>();
    left_traces = left_traces
        .into_par_iter()
        .zip(trace_needs_reduce.par_iter())
        .map(|(trace, should_reduce)| if *should_reduce { reduced_trace.clone() } else { trace })
        .collect();
    right_traces = right_traces
        .into_par_iter()
        .zip(trace_needs_reduce.par_iter())
        .map(|(trace, should_reduce)| if *should_reduce { reduced_trace.clone() } else { trace })
        .collect();

    let final_bounds = left_bounds
        .par_iter()
        .zip(right_bounds.par_iter())
        .map(|(left_bound, right_bound)| left_bound + right_bound)
        .collect::<Vec<_>>();
    assert!(
        final_bounds.iter().all(|bound| bound < ctx.p_full_ref()),
        "metadata-only add output exceeds p_full"
    );
    let final_traces = left_traces
        .par_iter()
        .zip(right_traces.par_iter())
        .map(|(left_trace, right_trace)| left_trace + right_trace)
        .collect::<Vec<_>>();
    assert!(
        final_traces.iter().all(|trace| trace < ctx.lut_mod_p_max_map_size_ref()),
        "metadata-only add output exceeds lut_mod_p_max_map_size"
    );
    (final_bounds, final_traces)
}

fn raw_decomposed_conv_output_metadata(
    ctx: &NestedRnsPolyContext,
    active_levels: usize,
    level_offset: usize,
    sparse_q_idx: usize,
    term_idx: usize,
    lhs_q_level_bound: &BigUint,
    num_slots: usize,
) -> (Vec<BigUint>, Vec<BigUint>) {
    assert!(
        sparse_q_idx < active_levels,
        "sparse_q_idx {} exceeds active levels {}",
        sparse_q_idx,
        active_levels
    );
    assert!(num_slots > 0, "num_slots must be positive");

    let term_bound = raw_decomposed_term_bound(ctx, term_idx);
    let mut term_max_plaintexts = vec![BigUint::ZERO; active_levels];
    let mut term_p_max_traces = vec![BigUint::ZERO; active_levels];
    term_max_plaintexts[sparse_q_idx] = lhs_q_level_bound * &term_bound;
    term_p_max_traces[sparse_q_idx] = ctx.reduced_p_max_trace();

    let mut current_layer = vec![(term_max_plaintexts, term_p_max_traces); num_slots];
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity(current_layer.len().div_ceil(2));
        let mut iter = current_layer.into_iter();
        while let Some((left_bounds, left_traces)) = iter.next() {
            if let Some((right_bounds, right_traces)) = iter.next() {
                next_layer.push(add_nested_rns_metadata(
                    ctx,
                    active_levels,
                    level_offset,
                    &left_bounds,
                    &left_traces,
                    &right_bounds,
                    &right_traces,
                ));
            } else {
                next_layer.push((left_bounds, left_traces));
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("raw decomposed convolution metadata requires at least one term")
}

pub type NativeRingGswCiphertext = [Vec<DCRTPoly>; 2];

fn active_q_modulus(ctx: &NestedRnsPolyContext) -> BigUint {
    BigUint::from(*ctx.q_moduli().first().expect("Ring-GSW helpers require one active q modulus"))
}

fn ring_gsw_randomizer_norm_ctx<P: Poly>(
    params: &P::Params,
    width: usize,
    max_decomposition_value: u64,
) -> Arc<SimulatorContext> {
    let ring_dim_sqrt = BigDecimal::from(params.ring_dimension() as u64)
        .sqrt()
        .expect("sqrt(ring_dimension) failed");
    let base = BigDecimal::from(max_decomposition_value) + BigDecimal::from(1u64);
    Arc::new(SimulatorContext::new(ring_dim_sqrt, base, width, 1, 1))
}

fn native_gadget_matrix(params: &DCRTPolyParams, ctx: &NestedRnsPolyContext) -> DCRTPolyMatrix {
    let gadget_row = nested_rns_gadget_vector::<DCRTPoly, DCRTPolyMatrix>(params, ctx, None, None)
        .get_row(0)
        .into_par_iter()
        .map(|poly| {
            DCRTPoly::from_biguint_to_constant(
                params,
                poly.coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("nested-RNS gadget row entry must contain a constant coefficient"),
            )
        })
        .collect::<Vec<_>>();
    let gadget_len = gadget_row.len();
    let zero = DCRTPoly::const_zero(params);

    let mut top = gadget_row.clone();
    top.extend((0..gadget_len).map(|_| zero.clone()));

    let mut bottom = vec![zero.clone(); gadget_len];
    bottom.extend(gadget_row);

    DCRTPolyMatrix::from_poly_vec(params, vec![top, bottom])
}

fn native_gadget_decompose_window(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    input_poly: &DCRTPoly,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> Vec<DCRTPoly> {
    let decomposed = nested_rns_gadget_decomposed::<DCRTPoly, DCRTPolyMatrix>(
        params,
        ctx,
        &DCRTPolyMatrix::from_poly_vec(params, vec![vec![input_poly.clone()]]),
        enable_levels,
        level_offset,
    );
    assert_eq!(
        decomposed.col_size(),
        1,
        "nested-RNS gadget decomposition for a single polynomial must yield one column"
    );
    (0..decomposed.row_size())
        .into_par_iter()
        .map(|row_idx| decomposed.entry(row_idx, 0))
        .collect::<Vec<_>>()
}

fn native_gadget_decompose(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    input_poly: &DCRTPoly,
) -> Vec<DCRTPoly> {
    native_gadget_decompose_window(params, ctx, input_poly, None, None)
}

pub fn sample_secret_key(params: &DCRTPolyParams) -> DCRTPoly {
    let uniform_sampler = DCRTPolyUniformSampler::new();
    uniform_sampler.sample_poly(params, &DistType::TernaryDist)
}

pub fn sample_public_key<B: AsRef<[u8]>>(
    params: &DCRTPolyParams,
    width: usize,
    secret_key: &DCRTPoly,
    hash_key: [u8; 32],
    tag: B,
    error: Option<&[DCRTPoly]>,
) -> NativeRingGswCiphertext {
    assert!(width > 0, "Ring-GSW public-key width must be positive");
    if let Some(error) = error {
        assert_eq!(
            error.len(),
            width,
            "Ring-GSW public-key error length {} must match width {}",
            error.len(),
            width
        );
    }
    let hash_sampler = DCRTPolyHashSampler::<Keccak256>::new();
    let a =
        hash_sampler.sample_hash(params, hash_key, tag, 1, width, DistType::FinRingDist).get_row(0);
    let b = a
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let base = -(secret_key.clone() * entry);
            match error {
                Some(error) => base + error[idx].clone(),
                None => base,
            }
        })
        .collect::<Vec<DCRTPoly>>();
    [a, b]
}

pub fn encrypt_plaintext_bit<B: AsRef<[u8]>>(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    public_key: &NativeRingGswCiphertext,
    plaintext: u64,
    randomizer_key: [u8; 32],
    randomizer_tag: B,
) -> NativeRingGswCiphertext {
    let width = public_key[0].len();
    assert_eq!(public_key[1].len(), width, "Ring-GSW public key rows must have the same width");
    let hash_sampler = DCRTPolyHashSampler::<Keccak256>::new();
    let randomizer = hash_sampler.sample_hash(
        params,
        randomizer_key,
        randomizer_tag,
        width,
        width,
        DistType::BitDist,
    );
    let public_matrix =
        DCRTPolyMatrix::from_poly_vec(params, vec![public_key[0].clone(), public_key[1].clone()]);
    let gadget_matrix = native_gadget_matrix(params, ctx);
    let plaintext_poly = DCRTPoly::from_biguint_to_constant(params, BigUint::from(plaintext));
    let ciphertext = (public_matrix * randomizer) + (gadget_matrix * plaintext_poly);
    [ciphertext.get_row(0), ciphertext.get_row(1)]
}

pub fn ciphertext_inputs_from_native<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<PolyVec<P>> {
    let mut inputs = Vec::new();
    for row in ciphertext {
        for poly in row {
            inputs.extend(encode_nested_rns_poly_vec_with_offset::<P>(
                params,
                ctx,
                &poly.coeffs_biguints(),
                level_offset,
                enable_levels,
            ));
        }
    }
    inputs
}

fn ciphertext_poly_from_output(params: &DCRTPolyParams, output: &PolyVec<DCRTPoly>) -> DCRTPoly {
    DCRTPoly::from_biguints(
        params,
        &output
            .as_slice()
            .iter()
            .map(|slot_poly| {
                slot_poly
                    .coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("output slot polynomial must contain a constant coefficient")
            })
            .collect::<Vec<_>>(),
    )
}

fn ciphertext_row_from_outputs(
    params: &DCRTPolyParams,
    outputs: &[PolyVec<DCRTPoly>],
) -> Vec<DCRTPoly> {
    outputs.par_iter().map(|output| ciphertext_poly_from_output(params, output)).collect()
}

pub fn ciphertext_from_outputs(
    params: &DCRTPolyParams,
    outputs: &[PolyVec<DCRTPoly>],
    width: usize,
) -> NativeRingGswCiphertext {
    assert_eq!(
        outputs.len(),
        2 * width,
        "Ring-GSW output must contain one reconstructed polynomial per ciphertext entry"
    );
    let (row0, row1) = rayon::join(
        || ciphertext_row_from_outputs(params, &outputs[..width]),
        || ciphertext_row_from_outputs(params, &outputs[width..]),
    );
    [row0, row1]
}

pub fn decrypt_ciphertext(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    secret_key: &DCRTPoly,
    plaintext_modulus: u64,
) -> DCRTPoly {
    let q = active_q_modulus(ctx);
    let scaled = &q / BigUint::from(plaintext_modulus);
    let zero_poly = DCRTPoly::const_zero(params);
    let scaled_poly = DCRTPoly::from_biguint_to_constant(params, scaled);
    let mut g_inverse = native_gadget_decompose(params, ctx, &zero_poly);
    g_inverse.extend(native_gadget_decompose(params, ctx, &scaled_poly));
    let products = ciphertext[0]
        .par_iter()
        .zip(ciphertext[1].par_iter())
        .zip(g_inverse.par_iter())
        .map(|((top, bottom), g_inv)| ((top.clone() * secret_key) + bottom) * g_inv)
        .collect::<Vec<_>>();
    let mut iter = products.into_iter();
    let mut acc = iter.next().expect("Ring-GSW decryption requires at least one ciphertext column");
    for term in iter {
        acc += term;
    }
    acc
}

#[derive(Debug, Clone)]
pub struct RingGswContext<P: Poly> {
    pub params: P::Params,
    pub num_slots: usize,
    pub nested_rns: Arc<NestedRnsPolyContext>,
    pub level_offset: usize,
    pub active_levels: usize,
    pub randomizer_norm_ctx: Arc<SimulatorContext>,
    add_entry_cache: DashMap<RingGswAddEntryPlanKey, usize>,
    sub_entry_cache: DashMap<RingGswSubEntryPlanKey, usize>,
    pub mul_super_batch_columns: usize,
    pub mul_super_batch_subcircuit_id: usize,
    pub mul_super_batch_tail_columns: usize,
    pub mul_super_batch_tail_subcircuit_id: usize,
    pub mul_output_max_plaintexts: Vec<BigUint>,
    pub mul_output_p_max_traces: Vec<BigUint>,
}

impl<P: Poly> RingGswContext<P> {
    pub fn width(&self) -> usize {
        2 * self.active_levels * (self.nested_rns.p_moduli.len() + 1)
    }

    pub fn gadget_len(&self) -> usize {
        self.active_levels * (self.nested_rns.p_moduli.len() + 1)
    }

    pub fn fresh_randomizer_norm(&self) -> PolyMatrixNorm {
        PolyMatrixNorm::new(
            self.randomizer_norm_ctx.clone(),
            self.width(),
            self.width(),
            BigDecimal::from(1u64),
            None,
        )
    }

    pub fn decomposed_randomizer_norm(&self) -> PolyMatrixNorm {
        let max_p_modulus = *self
            .nested_rns
            .p_moduli
            .iter()
            .max()
            .expect("RingGswContext requires at least one p modulus");
        PolyMatrixNorm::new(
            self.randomizer_norm_ctx.clone(),
            self.width(),
            self.width(),
            BigDecimal::from(max_p_modulus),
            None,
        )
    }
}

impl<P: Poly + 'static> RingGswContext<P> {
    fn shared_helper_storage(
        storage: Option<SubCircuitDiskStorage>,
        prefix: &str,
    ) -> SubCircuitDiskStorage {
        storage.unwrap_or_else(|| SubCircuitDiskStorage::temporary(prefix))
    }

    fn helper_circuit_with_storage(storage: &SubCircuitDiskStorage) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        circuit.use_subcircuit_disk_storage(storage.clone());
        circuit
    }

    fn entry_input_from_template(
        template: &NestedRnsPoly<P>,
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
    ) -> NestedRnsPoly<P> {
        NestedRnsPoly::input_with_metadata(
            ctx,
            template.enable_levels,
            Some(template.level_offset),
            template.max_plaintexts.clone(),
            template.p_max_traces.clone(),
            circuit,
        )
    }

    fn entry_binary_subcircuit<F>(
        source_circuit: &PolyCircuit<P>,
        lhs: &NestedRnsPoly<P>,
        rhs: &NestedRnsPoly<P>,
        sub_circuit_storage: SubCircuitDiskStorage,
        combine: F,
    ) -> (PolyCircuit<P>, RingGswEntryOutputMetadata)
    where
        F: Fn(&NestedRnsPoly<P>, &NestedRnsPoly<P>, &mut PolyCircuit<P>) -> NestedRnsPoly<P> + Copy,
    {
        let mut helper_circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let helper_ctx =
            Arc::new(lhs.ctx.register_shared_subcircuits_in(source_circuit, &mut helper_circuit));
        let lhs_entry =
            Self::entry_input_from_template(lhs, helper_ctx.clone(), &mut helper_circuit);
        let rhs_entry = Self::entry_input_from_template(rhs, helper_ctx, &mut helper_circuit);
        let output = combine(&lhs_entry, &rhs_entry, &mut helper_circuit);
        let metadata = RingGswCiphertext::entry_output_metadata(&output);
        helper_circuit.output(flatten_nested_rns_entries(std::slice::from_ref(&output)));
        (helper_circuit, metadata)
    }

    fn add_entry_subcircuit(
        source_circuit: &PolyCircuit<P>,
        lhs: &NestedRnsPoly<P>,
        rhs: &NestedRnsPoly<P>,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> (PolyCircuit<P>, RingGswEntryOutputMetadata) {
        Self::entry_binary_subcircuit(
            source_circuit,
            lhs,
            rhs,
            sub_circuit_storage,
            |left, right, circuit| left.add(right, circuit),
        )
    }

    fn sub_entry_subcircuit(
        source_circuit: &PolyCircuit<P>,
        lhs: &NestedRnsPoly<P>,
        rhs: &NestedRnsPoly<P>,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> (PolyCircuit<P>, RingGswEntryOutputMetadata) {
        Self::entry_binary_subcircuit(
            source_circuit,
            lhs,
            rhs,
            sub_circuit_storage,
            |left, right, circuit| left.sub(right, circuit),
        )
    }

    fn mul_raw_decomposed_q_level_entry_subcircuit(
        source_circuit: &PolyCircuit<P>,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        _level_offset: usize,
        sparse_q_idx: usize,
        term_helper: Arc<PolyCircuit<P>>,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> PolyCircuit<P> {
        let mut circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let ctx =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let lhs_row = circuit.input(ctx.p_moduli.len());
        let term_gate = circuit.input(1)[0];
        let term_row = vec![term_gate; ctx.p_moduli.len()];
        let term_helper_id = circuit.register_shared_sub_circuit(term_helper);
        let mut inputs = lhs_row;
        inputs.extend(term_row);
        let product_row = circuit.call_sub_circuit(term_helper_id, &inputs);
        let zero_gate = circuit.const_zero_gate();
        let mut outputs = Vec::with_capacity(active_levels * ctx.p_moduli.len());
        for q_idx in 0..active_levels {
            if q_idx == sparse_q_idx {
                outputs.extend(product_row.iter().copied());
            } else {
                outputs.extend(std::iter::repeat_n(zero_gate, ctx.p_moduli.len()));
            }
        }
        circuit.output(outputs);
        circuit
    }

    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        p_moduli_bits: usize,
        max_unreduced_muls: usize,
        scale: u64,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> Self {
        validate_num_slots::<P>(params, num_slots);
        let level_offset = level_offset.unwrap_or(0);
        let (_, _, max_q_moduli_depth) = params.to_crt();
        let active_levels = enable_levels.unwrap_or_else(|| {
            max_q_moduli_depth
                .checked_sub(level_offset)
                .expect("level_offset must not exceed q_moduli_depth")
        });
        assert!(active_levels > 0, "RingGswContext requires at least one active q level");
        let nested_rns_depth = level_offset + active_levels;
        assert!(
            nested_rns_depth <= max_q_moduli_depth,
            "active Ring-GSW q-window exceeds NestedRnsPolyContext depth"
        );
        let setup_start = Instant::now();
        let nested_rns_start = Instant::now();
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            params,
            p_moduli_bits,
            max_unreduced_muls,
            scale,
            false,
            Some(nested_rns_depth),
        ));
        debug!(
            "RingGswContext::setup nested_rns context ready: active_levels={}, width_hint={}, elapsed_ms={}",
            active_levels,
            2 * active_levels * (nested_rns.p_moduli.len() + 1),
            nested_rns_start.elapsed().as_millis()
        );
        let width = 2 * active_levels * (nested_rns.p_moduli.len() + 1);
        let max_p_modulus = *nested_rns
            .p_moduli
            .iter()
            .max()
            .expect("RingGswContext requires at least one p modulus");
        let randomizer_norm_ctx = ring_gsw_randomizer_norm_ctx::<P>(params, width, max_p_modulus);
        let helper_storage =
            Self::shared_helper_storage(circuit.cloned_subcircuit_disk_storage(), "ring-gsw");
        let helper_build_start = Instant::now();
        let (mul_column_subcircuit, mul_output_template) = Self::mul_column_subcircuit(
            circuit,
            params,
            num_slots,
            nested_rns.as_ref(),
            active_levels,
            level_offset,
            width,
            helper_storage.clone(),
        );
        let mul_column_subcircuit = Arc::new(mul_column_subcircuit);
        debug!(
            "RingGswContext::setup mul helper subcircuit built: width={}, elapsed_ms={}",
            width,
            helper_build_start.elapsed().as_millis()
        );
        let mul_batch_columns = width.min(MUL_TOP_LEVEL_BATCH_COLUMNS);
        let batch_helper_start = Instant::now();
        let mul_batch_subcircuit = Arc::new(Self::mul_columns_batch_subcircuit(
            circuit,
            nested_rns.as_ref(),
            active_levels,
            level_offset,
            width,
            mul_batch_columns,
            Arc::clone(&mul_column_subcircuit),
            helper_storage.clone(),
        ));
        let mul_batch_tail_columns = width % mul_batch_columns;
        let mul_batch_tail_subcircuit = if mul_batch_tail_columns > 0 {
            Some(Arc::new(Self::mul_columns_batch_subcircuit(
                circuit,
                nested_rns.as_ref(),
                active_levels,
                level_offset,
                width,
                mul_batch_tail_columns,
                Arc::clone(&mul_column_subcircuit),
                helper_storage.clone(),
            )))
        } else {
            None
        };
        debug!(
            "RingGswContext::setup mul batch helpers built: batch_columns={}, tail_columns={}, elapsed_ms={}",
            mul_batch_columns,
            mul_batch_tail_columns,
            batch_helper_start.elapsed().as_millis()
        );
        let mul_super_batch_columns =
            (mul_batch_columns * MUL_TOP_LEVEL_SUPER_BATCH_GROUPS).min(width);
        let super_batch_helper_start = Instant::now();
        let mul_super_batch_subcircuit_id =
            circuit.register_sub_circuit(Self::mul_super_batch_subcircuit(
                circuit,
                nested_rns.as_ref(),
                active_levels,
                level_offset,
                width,
                mul_super_batch_columns,
                mul_batch_columns,
                Arc::clone(&mul_batch_subcircuit),
                mul_batch_tail_subcircuit.clone(),
                helper_storage.clone(),
            ));
        let mul_super_batch_tail_columns = width % mul_super_batch_columns;
        let mul_super_batch_tail_subcircuit_id = if mul_super_batch_tail_columns > 0 {
            circuit.register_sub_circuit(Self::mul_super_batch_subcircuit(
                circuit,
                nested_rns.as_ref(),
                active_levels,
                level_offset,
                width,
                mul_super_batch_tail_columns,
                mul_batch_columns,
                Arc::clone(&mul_batch_subcircuit),
                mul_batch_tail_subcircuit.clone(),
                helper_storage.clone(),
            ))
        } else {
            mul_super_batch_subcircuit_id
        };
        debug!(
            "RingGswContext::setup mul super-batch helpers built: super_batch_columns={}, tail_columns={}, elapsed_ms={}",
            mul_super_batch_columns,
            mul_super_batch_tail_columns,
            super_batch_helper_start.elapsed().as_millis()
        );
        let ctx = Arc::new(Self {
            params: params.clone(),
            num_slots,
            nested_rns,
            level_offset,
            active_levels,
            randomizer_norm_ctx,
            add_entry_cache: DashMap::new(),
            sub_entry_cache: DashMap::new(),
            mul_super_batch_columns,
            mul_super_batch_subcircuit_id,
            mul_super_batch_tail_columns,
            mul_super_batch_tail_subcircuit_id,
            mul_output_max_plaintexts: mul_output_template.max_plaintexts,
            mul_output_p_max_traces: mul_output_template.p_max_traces,
        });
        debug!(
            "RingGswContext::setup completed: width={}, wrapper_prebuild_elapsed_ms={}, total_elapsed_ms={}",
            width,
            0,
            setup_start.elapsed().as_millis()
        );
        Arc::try_unwrap(ctx).expect("RingGswContext setup must not retain temporary Arc clones")
    }

    fn mul_subcircuit(
        source_circuit: &PolyCircuit<P>,
        params: &P::Params,
        num_slots: usize,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> (PolyCircuit<P>, NestedRnsPoly<P>) {
        let start = Instant::now();
        let mut circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let nested_rns =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let (normalized_max_plaintexts, normalized_p_max_traces) =
            nested_rns.full_reduce_output_metadata(Some(active_levels), Some(level_offset));
        let chunk_width = template_ctx.p_moduli.len() + 1;
        let gadget_len = active_levels * chunk_width;
        assert_eq!(
            width,
            2 * gadget_len,
            "Ring-GSW mul subcircuit width {} must equal 2 * gadget_len {}",
            width,
            gadget_len
        );
        let column_helper_start = Instant::now();
        let (mul_column_subcircuit, mul_output_template) = Self::mul_column_subcircuit(
            source_circuit,
            params,
            num_slots,
            template_ctx,
            active_levels,
            level_offset,
            width,
            sub_circuit_storage.clone(),
        );
        let mul_column_subcircuit_id = circuit.register_sub_circuit(mul_column_subcircuit);
        debug!(
            "RingGswContext::mul_subcircuit column helper registered: width={}, elapsed_ms={}",
            width,
            column_helper_start.elapsed().as_millis()
        );

        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_subcircuit inputs allocated: width={}, elapsed_ms={}",
            width,
            input_start.elapsed().as_millis()
        );

        let template_entry =
            lhs_row0.first().expect("RingGswContext::mul_subcircuit requires positive width");
        let levels = template_entry.active_q_moduli().len();
        let p_moduli_depth = template_entry.ctx.p_moduli.len();
        let entry_size = levels * p_moduli_depth;

        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswContext::mul_subcircuit lhs inputs flattened: width={}, input_len={}, elapsed_ms={}",
            width,
            lhs_inputs.len(),
            lhs_inputs_start.elapsed().as_millis()
        );

        let mut row0 = Vec::with_capacity(width);
        let mut row1 = Vec::with_capacity(width);
        let column_loop_start = Instant::now();
        for col_idx in 0..width {
            let mut rhs_suffix = Vec::with_capacity(2 * entry_size);
            rhs_suffix.extend(rhs_row0[col_idx].inner.iter().flatten().copied());
            rhs_suffix.extend(rhs_row1[col_idx].inner.iter().flatten().copied());
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                mul_column_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            assert_eq!(
                outputs.len(),
                2 * entry_size,
                "Ring-GSW column mul subcircuit output size must match two ciphertext entries"
            );
            row0.push(nested_rns_from_flat_outputs(
                template_entry,
                &outputs[..entry_size],
                mul_output_template.max_plaintexts.clone(),
                mul_output_template.p_max_traces.clone(),
            ));
            row1.push(nested_rns_from_flat_outputs(
                template_entry,
                &outputs[entry_size..],
                mul_output_template.max_plaintexts.clone(),
                mul_output_template.p_max_traces.clone(),
            ));
        }
        debug!(
            "RingGswContext::mul_subcircuit column loop finished: width={}, elapsed_ms={}",
            width,
            column_loop_start.elapsed().as_millis()
        );

        let (row0_outputs, row1_outputs) =
            rayon::join(|| flatten_nested_rns_entries(&row0), || flatten_nested_rns_entries(&row1));
        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        circuit.output(outputs);
        debug!(
            "RingGswContext::mul_subcircuit finished: width={}, entry_size={}, total_elapsed_ms={}",
            width,
            entry_size,
            start.elapsed().as_millis()
        );
        (circuit, mul_output_template)
    }

    fn mul_columns_batch_subcircuit(
        source_circuit: &PolyCircuit<P>,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        batch_columns: usize,
        mul_column_subcircuit: Arc<PolyCircuit<P>>,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> PolyCircuit<P> {
        assert!(batch_columns > 0, "batch_columns must be positive");
        assert!(
            batch_columns <= width,
            "batch_columns {} must not exceed width {}",
            batch_columns,
            width
        );
        let start = Instant::now();
        let mut circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let nested_rns =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let (normalized_max_plaintexts, normalized_p_max_traces) =
            nested_rns.full_reduce_output_metadata(Some(active_levels), Some(level_offset));

        let column_helper_start = Instant::now();
        let mul_column_subcircuit_id = circuit.register_shared_sub_circuit(mul_column_subcircuit);
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit column helper registered: width={}, batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            column_helper_start.elapsed().as_millis()
        );

        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row0 = (0..batch_columns)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row1 = (0..batch_columns)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit inputs allocated: width={}, batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            input_start.elapsed().as_millis()
        );

        let template_entry = lhs_row0
            .first()
            .expect("RingGswContext::mul_columns_batch_subcircuit requires positive width");
        let levels = template_entry.active_q_moduli().len();
        let p_moduli_depth = template_entry.ctx.p_moduli.len();
        let entry_size = levels * p_moduli_depth;

        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit lhs inputs flattened: width={}, input_len={}, elapsed_ms={}",
            width,
            lhs_inputs.len(),
            lhs_inputs_start.elapsed().as_millis()
        );

        let mut row0_outputs = Vec::with_capacity(batch_columns * entry_size);
        let mut row1_outputs = Vec::with_capacity(batch_columns * entry_size);
        let batch_loop_start = Instant::now();
        for col_idx in 0..batch_columns {
            let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
                || flatten_nested_rns_entries(&rhs_row0[col_idx..col_idx + 1]),
                || flatten_nested_rns_entries(&rhs_row1[col_idx..col_idx + 1]),
            );
            let mut rhs_suffix = rhs_row0_inputs;
            rhs_suffix.extend(rhs_row1_inputs);
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                mul_column_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            assert_eq!(
                outputs.len(),
                2 * entry_size,
                "Ring-GSW batch mul column output size must match two ciphertext entries"
            );
            row0_outputs.extend_from_slice(&outputs[..entry_size]);
            row1_outputs.extend_from_slice(&outputs[entry_size..]);
        }
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit batch loop finished: width={}, batch_columns={}, elapsed_ms={}",
            width,
            batch_columns,
            batch_loop_start.elapsed().as_millis()
        );

        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        circuit.output(outputs);
        debug!(
            "RingGswContext::mul_columns_batch_subcircuit finished: width={}, batch_columns={}, entry_size={}, total_elapsed_ms={}",
            width,
            batch_columns,
            entry_size,
            start.elapsed().as_millis()
        );
        circuit
    }

    fn mul_super_batch_subcircuit(
        source_circuit: &PolyCircuit<P>,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        super_batch_columns: usize,
        batch_columns: usize,
        batch_subcircuit: Arc<PolyCircuit<P>>,
        batch_tail_subcircuit: Option<Arc<PolyCircuit<P>>>,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> PolyCircuit<P> {
        assert!(super_batch_columns > 0, "super_batch_columns must be positive");
        assert!(
            super_batch_columns <= width,
            "super_batch_columns {} must not exceed width {}",
            super_batch_columns,
            width
        );
        assert!(
            batch_columns > 0 && batch_columns <= super_batch_columns,
            "batch_columns {} must be in 1..={} for super-batch helper",
            batch_columns,
            super_batch_columns
        );
        let start = Instant::now();
        let mut circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let nested_rns =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let (normalized_max_plaintexts, normalized_p_max_traces) =
            nested_rns.full_reduce_output_metadata(Some(active_levels), Some(level_offset));

        let batch_helper_start = Instant::now();
        let batch_subcircuit_id = circuit.register_shared_sub_circuit(batch_subcircuit);
        let batch_tail_columns = super_batch_columns % batch_columns;
        let batch_tail_subcircuit_id = if batch_tail_columns > 0 {
            Some(
                circuit.register_shared_sub_circuit(
                    batch_tail_subcircuit
                        .expect("super-batch tail helper must exist for non-zero tail columns"),
                ),
            )
        } else {
            None
        };
        debug!(
            "RingGswContext::mul_super_batch_subcircuit batch helper(s) registered: width={}, super_batch_columns={}, batch_columns={}, tail_columns={}, elapsed_ms={}",
            width,
            super_batch_columns,
            batch_columns,
            batch_tail_columns,
            batch_helper_start.elapsed().as_millis()
        );

        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row0 = (0..super_batch_columns)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_row1 = (0..super_batch_columns)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_super_batch_subcircuit inputs allocated: width={}, super_batch_columns={}, elapsed_ms={}",
            width,
            super_batch_columns,
            input_start.elapsed().as_millis()
        );

        let template_entry = lhs_row0
            .first()
            .expect("RingGswContext::mul_super_batch_subcircuit requires positive width");
        let levels = template_entry.active_q_moduli().len();
        let p_moduli_depth = template_entry.ctx.p_moduli.len();
        let entry_size = levels * p_moduli_depth;

        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswContext::mul_super_batch_subcircuit lhs inputs flattened: width={}, input_len={}, elapsed_ms={}",
            width,
            lhs_inputs.len(),
            lhs_inputs_start.elapsed().as_millis()
        );

        let mut row0_outputs = Vec::with_capacity(super_batch_columns * entry_size);
        let mut row1_outputs = Vec::with_capacity(super_batch_columns * entry_size);
        let super_batch_loop_start = Instant::now();
        for col_start in (0..super_batch_columns).step_by(batch_columns) {
            let col_end = (col_start + batch_columns).min(super_batch_columns);
            let actual_batch_columns = col_end - col_start;
            let current_batch_subcircuit_id = if actual_batch_columns == batch_columns {
                batch_subcircuit_id
            } else if actual_batch_columns == batch_tail_columns {
                batch_tail_subcircuit_id
                    .expect("super-batch tail helper must exist for non-zero tail columns")
            } else {
                unreachable!(
                    "unexpected Ring-GSW super-batch width {}; configured batch={}, tail={}",
                    actual_batch_columns, batch_columns, batch_tail_columns
                );
            };
            let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
                || flatten_nested_rns_entries(&rhs_row0[col_start..col_end]),
                || flatten_nested_rns_entries(&rhs_row1[col_start..col_end]),
            );
            let mut rhs_suffix = rhs_row0_inputs;
            rhs_suffix.extend(rhs_row1_inputs);
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                current_batch_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            debug_assert_eq!(outputs.len(), 2 * actual_batch_columns * entry_size);
            let (row0_batch_outputs, row1_batch_outputs) =
                outputs.split_at(actual_batch_columns * entry_size);
            row0_outputs.extend_from_slice(row0_batch_outputs);
            row1_outputs.extend_from_slice(row1_batch_outputs);
        }
        debug!(
            "RingGswContext::mul_super_batch_subcircuit loop finished: width={}, super_batch_columns={}, batch_columns={}, elapsed_ms={}",
            width,
            super_batch_columns,
            batch_columns,
            super_batch_loop_start.elapsed().as_millis()
        );

        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        circuit.output(outputs);
        debug!(
            "RingGswContext::mul_super_batch_subcircuit finished: width={}, super_batch_columns={}, batch_columns={}, entry_size={}, total_elapsed_ms={}",
            width,
            super_batch_columns,
            batch_columns,
            entry_size,
            start.elapsed().as_millis()
        );
        circuit
    }

    fn mul_q_level_row_with_raw_terms_subcircuit(
        source_circuit: &PolyCircuit<P>,
        num_slots: usize,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> (PolyCircuit<P>, NestedRnsPoly<P>) {
        let mut helper_circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let helper_ctx = Arc::new(
            template_ctx.register_shared_subcircuits_in(source_circuit, &mut helper_circuit),
        );
        let (helper_max_plaintexts, _helper_p_max_traces) =
            helper_ctx.full_reduce_output_metadata(Some(active_levels), Some(level_offset));
        let chunk_width = template_ctx.p_moduli.len() + 1;
        let gadget_len = active_levels * chunk_width;
        assert_eq!(
            width,
            2 * gadget_len,
            "Ring-GSW q-level row helper width {} must equal 2 * gadget_len {}",
            width,
            gadget_len
        );
        let product_template = sparse_nested_rns_output_template(
            helper_ctx.clone(),
            active_levels,
            level_offset,
            &mut helper_circuit,
        );
        let term_helper =
            Arc::new(negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit::<P>(
                source_circuit,
                template_ctx,
                1,
                num_slots,
            ));
        let product_subcircuits = (0..active_levels)
            .into_par_iter()
            .map(|sparse_q_idx| {
                Self::mul_raw_decomposed_q_level_entry_subcircuit(
                    source_circuit,
                    template_ctx,
                    active_levels,
                    level_offset,
                    sparse_q_idx,
                    Arc::clone(&term_helper),
                    sub_circuit_storage.clone(),
                )
            })
            .collect::<Vec<_>>();
        let product_subcircuit_ids = product_subcircuits
            .into_iter()
            .map(|subcircuit| helper_circuit.register_sub_circuit(subcircuit))
            .collect::<Vec<_>>();
        let product_output_metadata = (0..gadget_len)
            .into_par_iter()
            .map(|entry_idx| {
                let sparse_q_idx = entry_idx / chunk_width;
                let term_idx = entry_idx % chunk_width;
                raw_decomposed_conv_output_metadata(
                    helper_ctx.as_ref(),
                    active_levels,
                    level_offset,
                    sparse_q_idx,
                    term_idx,
                    &helper_max_plaintexts[sparse_q_idx],
                    num_slots,
                )
            })
            .collect::<Vec<_>>();
        let row_q_levels =
            (0..width).map(|_| helper_circuit.input(helper_ctx.p_moduli.len())).collect::<Vec<_>>();
        let column_terms = (0..width).map(|_| helper_circuit.input(1)[0]).collect::<Vec<_>>();
        let products = row_q_levels
            .iter()
            .zip(column_terms.iter())
            .enumerate()
            .map(|(col_idx, (lhs_row, &term_gate))| {
                let sparse_q_idx = (col_idx % gadget_len) / chunk_width;
                let mut inputs = lhs_row.clone();
                inputs.push(term_gate);
                let outputs =
                    helper_circuit.call_sub_circuit(product_subcircuit_ids[sparse_q_idx], &inputs);
                let (output_max_plaintexts, output_p_max_traces) =
                    &product_output_metadata[col_idx % gadget_len];
                nested_rns_from_flat_outputs(
                    &product_template,
                    &outputs,
                    output_max_plaintexts.clone(),
                    output_p_max_traces.clone(),
                )
            })
            .collect::<Vec<_>>();
        let result = reduce_nested_rns_terms_pairwise(
            products,
            &mut helper_circuit,
            |lhs, rhs, helper_circuit| lhs.add(rhs, helper_circuit),
        );
        helper_circuit.output(result.inner.iter().flatten().copied().collect());
        (helper_circuit, result)
    }

    fn mul_q_level_rows_with_raw_terms_batch_subcircuit(
        source_circuit: &PolyCircuit<P>,
        num_slots: usize,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        batch_columns: usize,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> (PolyCircuit<P>, NestedRnsPoly<P>) {
        assert!(batch_columns > 0, "batch_columns must be positive");
        let mut helper_circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let chunk_width = template_ctx.p_moduli.len() + 1;
        let gadget_len = active_levels * chunk_width;
        assert_eq!(
            width,
            2 * gadget_len,
            "Ring-GSW q-level batch helper width {} must equal 2 * gadget_len {}",
            width,
            gadget_len
        );
        let helper_build_start = Instant::now();
        let (row_subcircuit, row_output_template) = Self::mul_q_level_row_with_raw_terms_subcircuit(
            source_circuit,
            num_slots,
            template_ctx,
            active_levels,
            level_offset,
            width,
            sub_circuit_storage.clone(),
        );
        let row_subcircuit_id = helper_circuit.register_sub_circuit(row_subcircuit);
        debug!(
            "RingGswContext::mul_q_level_rows_with_raw_terms_batch_subcircuit row helper registered: batch_columns={}, active_levels={}, elapsed_ms={}",
            batch_columns,
            active_levels,
            helper_build_start.elapsed().as_millis()
        );
        let helper_ctx = Arc::new(
            template_ctx.register_shared_subcircuits_in(source_circuit, &mut helper_circuit),
        );
        let row0_q_levels =
            (0..width).map(|_| helper_circuit.input(helper_ctx.p_moduli.len())).collect::<Vec<_>>();
        let row1_q_levels =
            (0..width).map(|_| helper_circuit.input(helper_ctx.p_moduli.len())).collect::<Vec<_>>();
        let batch_column_terms = (0..batch_columns)
            .map(|_| (0..width).map(|_| helper_circuit.input(1)[0]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        debug!(
            "RingGswContext::mul_q_level_rows_with_raw_terms_batch_subcircuit inputs allocated: batch_columns={}, width={}, elapsed_ms={}",
            batch_columns,
            width,
            helper_build_start.elapsed().as_millis()
        );
        let entry_size = active_levels * helper_ctx.p_moduli.len();
        let mut row0_outputs = Vec::with_capacity(batch_columns * entry_size);
        let mut row1_outputs = Vec::with_capacity(batch_columns * entry_size);
        for column_terms in batch_column_terms.iter() {
            let mut row0_inputs =
                Vec::with_capacity(width * helper_ctx.p_moduli.len() + column_terms.len());
            row0_inputs.extend(row0_q_levels.iter().flatten().copied());
            row0_inputs.extend(column_terms.iter().copied());
            row0_outputs.extend(helper_circuit.call_sub_circuit(row_subcircuit_id, &row0_inputs));

            let mut row1_inputs =
                Vec::with_capacity(width * helper_ctx.p_moduli.len() + column_terms.len());
            row1_inputs.extend(row1_q_levels.iter().flatten().copied());
            row1_inputs.extend(column_terms.iter().copied());
            row1_outputs.extend(helper_circuit.call_sub_circuit(row_subcircuit_id, &row1_inputs));
        }
        let mut outputs = row0_outputs;
        outputs.extend(row1_outputs);
        helper_circuit.output(outputs);
        debug!(
            "RingGswContext::mul_q_level_rows_with_raw_terms_batch_subcircuit finished: batch_columns={}, width={}, total_elapsed_ms={}",
            batch_columns,
            width,
            helper_build_start.elapsed().as_millis()
        );
        (helper_circuit, row_output_template)
    }

    fn mul_column_subcircuit(
        source_circuit: &PolyCircuit<P>,
        _params: &P::Params,
        num_slots: usize,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
        sub_circuit_storage: SubCircuitDiskStorage,
    ) -> (PolyCircuit<P>, NestedRnsPoly<P>) {
        let start = Instant::now();
        let mut circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
        let nested_rns =
            Arc::new(template_ctx.register_shared_subcircuits_in(source_circuit, &mut circuit));
        let chunk_width = template_ctx.p_moduli.len() + 1;
        let gadget_len = active_levels * chunk_width;
        assert_eq!(
            width,
            2 * gadget_len,
            "Ring-GSW mul helper width {} must equal 2 * gadget_len {}",
            width,
            gadget_len
        );
        let (normalized_max_plaintexts, normalized_p_max_traces) =
            nested_rns.full_reduce_output_metadata(Some(active_levels), Some(level_offset));
        let dot_helper_start = Instant::now();
        let (local_dot_row_with_column_subcircuit_id, local_dot_output_template) = {
            let mut helper_circuit = Self::helper_circuit_with_storage(&sub_circuit_storage);
            let helper_ctx = Arc::new(
                template_ctx.register_shared_subcircuits_in(source_circuit, &mut helper_circuit),
            );
            let (helper_max_plaintexts, _helper_p_max_traces) =
                helper_ctx.full_reduce_output_metadata(Some(active_levels), Some(level_offset));
            let product_template = sparse_nested_rns_output_template(
                helper_ctx.clone(),
                active_levels,
                level_offset,
                &mut helper_circuit,
            );
            let term_helper =
                Arc::new(negacyclic_conv_mul_right_decomposed_term_many_shared_subcircuit::<P>(
                    source_circuit,
                    template_ctx,
                    1,
                    num_slots,
                ));
            let product_subcircuits_start = Instant::now();
            let product_subcircuits = (0..active_levels)
                .into_par_iter()
                .map(|sparse_q_idx| {
                    Self::mul_raw_decomposed_q_level_entry_subcircuit(
                        source_circuit,
                        template_ctx,
                        active_levels,
                        level_offset,
                        sparse_q_idx,
                        Arc::clone(&term_helper),
                        sub_circuit_storage.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let product_subcircuit_ids = product_subcircuits
                .into_iter()
                .map(|subcircuit| helper_circuit.register_sub_circuit(subcircuit))
                .collect::<Vec<_>>();
            debug!(
                "RingGswContext::mul_column_subcircuit q-level product helpers built/registered: active_levels={}, elapsed_ms={}",
                active_levels,
                product_subcircuits_start.elapsed().as_millis()
            );
            let product_template_start = Instant::now();
            let product_output_metadata = (0..gadget_len)
                .into_par_iter()
                .map(|entry_idx| {
                    let sparse_q_idx = entry_idx / chunk_width;
                    let term_idx = entry_idx % chunk_width;
                    raw_decomposed_conv_output_metadata(
                        helper_ctx.as_ref(),
                        active_levels,
                        level_offset,
                        sparse_q_idx,
                        term_idx,
                        &helper_max_plaintexts[sparse_q_idx],
                        num_slots,
                    )
                })
                .collect::<Vec<_>>();
            debug!(
                "RingGswContext::mul_column_subcircuit product metadata built in parallel: gadget_len={}, elapsed_ms={}",
                gadget_len,
                product_template_start.elapsed().as_millis()
            );
            let row_q_levels = (0..width)
                .map(|_| helper_circuit.input(helper_ctx.p_moduli.len()))
                .collect::<Vec<_>>();
            let column_terms = (0..width).map(|_| helper_circuit.input(1)[0]).collect::<Vec<_>>();
            let products = row_q_levels
                .iter()
                .zip(column_terms.iter())
                .enumerate()
                .map(|(col_idx, (lhs_row, &term_gate))| {
                    let sparse_q_idx = (col_idx % gadget_len) / chunk_width;
                    let mut inputs = lhs_row.clone();
                    inputs.push(term_gate);
                    let outputs = helper_circuit
                        .call_sub_circuit(product_subcircuit_ids[sparse_q_idx], &inputs);
                    let (output_max_plaintexts, output_p_max_traces) =
                        &product_output_metadata[col_idx % gadget_len];
                    nested_rns_from_flat_outputs(
                        &product_template,
                        &outputs,
                        output_max_plaintexts.clone(),
                        output_p_max_traces.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let result = reduce_nested_rns_terms_pairwise(
                products,
                &mut helper_circuit,
                |lhs, rhs, helper_circuit| lhs.add(rhs, helper_circuit),
            );
            let result_template = result.clone();
            helper_circuit.output(result.inner.into_iter().flatten().collect());
            (circuit.register_sub_circuit(helper_circuit), result_template)
        };
        debug!(
            "RingGswContext::mul_column_subcircuit dot helper ready: width={}, elapsed_ms={}",
            width,
            dot_helper_start.elapsed().as_millis()
        );
        let input_start = Instant::now();
        let lhs_row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let lhs_row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input_with_metadata(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    normalized_max_plaintexts.clone(),
                    normalized_p_max_traces.clone(),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let rhs_top = NestedRnsPoly::input_with_metadata(
            nested_rns.clone(),
            Some(active_levels),
            Some(level_offset),
            normalized_max_plaintexts.clone(),
            normalized_p_max_traces.clone(),
            &mut circuit,
        );
        let rhs_bottom = NestedRnsPoly::input_with_metadata(
            nested_rns.clone(),
            Some(active_levels),
            Some(level_offset),
            normalized_max_plaintexts,
            normalized_p_max_traces,
            &mut circuit,
        );
        debug!(
            "RingGswContext::mul_column_subcircuit inputs allocated: width={}, elapsed_ms={}",
            width,
            input_start.elapsed().as_millis()
        );
        let g_inverse_start = Instant::now();
        let mut g_inverse_terms = Vec::with_capacity(width);
        for q_idx in 0..active_levels {
            let (ys, w) = rhs_top.decomposition_terms_for_level(q_idx, &mut circuit);
            g_inverse_terms.extend(ys);
            g_inverse_terms.push(w);
        }
        for q_idx in 0..active_levels {
            let (ys, w) = rhs_bottom.decomposition_terms_for_level(q_idx, &mut circuit);
            g_inverse_terms.extend(ys);
            g_inverse_terms.push(w);
        }
        debug!(
            "RingGswContext::mul_column_subcircuit raw decomposition terms built: width={}, elapsed_ms={}",
            width,
            g_inverse_start.elapsed().as_millis()
        );
        let dot_products_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_q_level_rows_for_gadget_terms(&lhs_row0, gadget_len, chunk_width),
            || flatten_nested_rns_q_level_rows_for_gadget_terms(&lhs_row1, gadget_len, chunk_width),
        );
        let row0: NestedRnsPoly<P> = {
            let template = &lhs_row0[0];
            let mut inputs = lhs_row0_inputs;
            inputs.extend(g_inverse_terms.clone());
            let outputs =
                circuit.call_sub_circuit(local_dot_row_with_column_subcircuit_id, &inputs);
            nested_rns_from_flat_outputs(
                template,
                &outputs,
                local_dot_output_template.max_plaintexts.clone(),
                local_dot_output_template.p_max_traces.clone(),
            )
        };
        let row1: NestedRnsPoly<P> = {
            let template = &lhs_row1[0];
            let mut inputs = lhs_row1_inputs;
            inputs.extend(g_inverse_terms);
            let outputs =
                circuit.call_sub_circuit(local_dot_row_with_column_subcircuit_id, &inputs);
            nested_rns_from_flat_outputs(
                template,
                &outputs,
                local_dot_output_template.max_plaintexts.clone(),
                local_dot_output_template.p_max_traces.clone(),
            )
        };
        circuit.output(flatten_nested_rns_entries(&[row0, row1]));
        debug!(
            "RingGswContext::mul_column_subcircuit finished: width={}, dot_products_elapsed_ms={}, total_elapsed_ms={}",
            width,
            dot_products_start.elapsed().as_millis(),
            start.elapsed().as_millis()
        );
        (circuit, local_dot_output_template)
    }
}

#[derive(Debug, Clone)]
pub struct RingGswCiphertext<P: Poly> {
    pub ctx: Arc<RingGswContext<P>>,
    pub rows: [Vec<NestedRnsPoly<P>>; 2],
    pub randomizer_norm: PolyMatrixNorm,
    pub max_plaintext: BigUint,
}

impl<P: Poly + 'static> RingGswCiphertext<P> {
    fn entry_output_metadata(entry: &NestedRnsPoly<P>) -> RingGswEntryOutputMetadata {
        RingGswEntryOutputMetadata {
            max_plaintexts: entry.max_plaintexts.clone(),
            p_max_traces: entry.p_max_traces.clone(),
        }
    }

    fn active_q_moduli(entry: &NestedRnsPoly<P>) -> &[u64] {
        let levels = entry.max_plaintexts.len();
        &entry.ctx.q_moduli()[entry.level_offset..entry.level_offset + levels]
    }

    fn full_reduce_entry_output_metadata(entry: &NestedRnsPoly<P>) -> RingGswEntryOutputMetadata {
        let levels = entry.max_plaintexts.len();
        let (max_plaintexts, p_max_traces) =
            entry.ctx.full_reduce_output_metadata(Some(levels), Some(entry.level_offset));
        RingGswEntryOutputMetadata { max_plaintexts, p_max_traces }
    }

    fn lazy_reduce_entry_output_metadata(
        ctx: &NestedRnsPolyContext,
        metadata: &RingGswEntryOutputMetadata,
        reduce_levels: &[bool],
    ) -> RingGswEntryOutputMetadata {
        debug_assert_eq!(metadata.p_max_traces.len(), reduce_levels.len());
        RingGswEntryOutputMetadata {
            max_plaintexts: metadata.max_plaintexts.clone(),
            p_max_traces: metadata
                .p_max_traces
                .iter()
                .zip(reduce_levels.iter())
                .map(
                    |(trace, reduce)| {
                        if *reduce { ctx.reduced_p_max_trace() } else { trace.clone() }
                    },
                )
                .collect(),
        }
    }

    fn map_binary_row_entries<T, F>(
        lhs_row: &[NestedRnsPoly<P>],
        rhs_row: &[NestedRnsPoly<P>],
        f: F,
    ) -> Vec<T>
    where
        T: Send,
        F: Fn(&NestedRnsPoly<P>, &NestedRnsPoly<P>) -> T + Sync + Send,
    {
        debug_assert_eq!(lhs_row.len(), rhs_row.len());
        lhs_row.par_iter().zip(rhs_row.par_iter()).map(|(lhs, rhs)| f(lhs, rhs)).collect()
    }

    fn compute_add_entry_plan_and_output(
        left: &NestedRnsPoly<P>,
        right: &NestedRnsPoly<P>,
    ) -> (RingGswAddEntryPlanKey, RingGswEntryOutputMetadata) {
        debug_assert_eq!(left.max_plaintexts.len(), right.max_plaintexts.len());
        let q_moduli = Self::active_q_moduli(left);
        let p_full = left.ctx.p_full_ref().clone();
        let pre_full_reduce = q_moduli
            .par_iter()
            .enumerate()
            .any(|(q_idx, _)| &left.max_plaintexts[q_idx] + &right.max_plaintexts[q_idx] > p_full);
        let left_before_reduce = if pre_full_reduce {
            Self::full_reduce_entry_output_metadata(left)
        } else {
            Self::entry_output_metadata(left)
        };
        let right_before_reduce = if pre_full_reduce {
            Self::full_reduce_entry_output_metadata(right)
        } else {
            Self::entry_output_metadata(right)
        };
        let reduce_levels = left_before_reduce
            .p_max_traces
            .par_iter()
            .zip(right_before_reduce.p_max_traces.par_iter())
            .map(|(lhs_trace, rhs_trace)| {
                lhs_trace + rhs_trace >= *left.ctx.lut_mod_p_max_map_size_ref()
            })
            .collect::<Vec<_>>();
        let left_after_reduce = Self::lazy_reduce_entry_output_metadata(
            left.ctx.as_ref(),
            &left_before_reduce,
            &reduce_levels,
        );
        let right_after_reduce = Self::lazy_reduce_entry_output_metadata(
            right.ctx.as_ref(),
            &right_before_reduce,
            &reduce_levels,
        );
        let output_metadata = RingGswEntryOutputMetadata {
            max_plaintexts: left_after_reduce
                .max_plaintexts
                .par_iter()
                .zip(right_after_reduce.max_plaintexts.par_iter())
                .map(|(lhs_bound, rhs_bound)| lhs_bound + rhs_bound)
                .collect(),
            p_max_traces: left_after_reduce
                .p_max_traces
                .par_iter()
                .zip(right_after_reduce.p_max_traces.par_iter())
                .map(|(lhs_trace, rhs_trace)| lhs_trace + rhs_trace)
                .collect(),
        };
        (RingGswAddEntryPlanKey { pre_full_reduce, reduce_levels }, output_metadata)
    }

    fn compute_sub_entry_plan_and_output(
        left: &NestedRnsPoly<P>,
        right: &NestedRnsPoly<P>,
    ) -> (RingGswSubEntryPlanKey, RingGswEntryOutputMetadata) {
        debug_assert_eq!(left.max_plaintexts.len(), right.max_plaintexts.len());
        let q_moduli = Self::active_q_moduli(left);
        let p_full = left.ctx.p_full_ref().clone();
        let pre_full_reduce = q_moduli
            .par_iter()
            .enumerate()
            .any(|(q_idx, &q_i)| &left.max_plaintexts[q_idx] + BigUint::from(q_i - 1) > p_full);
        let left_before_reduce = if pre_full_reduce {
            Self::full_reduce_entry_output_metadata(left)
        } else {
            Self::entry_output_metadata(left)
        };
        let right_before_reduce = if pre_full_reduce {
            Self::full_reduce_entry_output_metadata(right)
        } else {
            Self::entry_output_metadata(right)
        };
        let p_max_minus_one = left.ctx.reduced_p_max_trace();
        let p_max = &p_max_minus_one + BigUint::from(1u64);
        let trace_multiplier = |trace: &BigUint| (trace + &p_max_minus_one) / &p_max;
        let predicted_traces = left_before_reduce
            .p_max_traces
            .par_iter()
            .zip(right_before_reduce.p_max_traces.par_iter())
            .map(|(lhs_trace, rhs_trace)| lhs_trace + trace_multiplier(rhs_trace) * &p_max)
            .collect::<Vec<_>>();
        let reduce_levels = predicted_traces
            .par_iter()
            .map(|trace| trace >= left.ctx.lut_mod_p_max_map_size_ref())
            .collect::<Vec<_>>();
        let left_after_reduce = Self::lazy_reduce_entry_output_metadata(
            left.ctx.as_ref(),
            &left_before_reduce,
            &reduce_levels,
        );
        let right_after_reduce = Self::lazy_reduce_entry_output_metadata(
            right.ctx.as_ref(),
            &right_before_reduce,
            &reduce_levels,
        );
        let trace_multipliers =
            right_after_reduce.p_max_traces.par_iter().map(trace_multiplier).collect::<Vec<_>>();
        let output_metadata = RingGswEntryOutputMetadata {
            max_plaintexts: left_after_reduce
                .max_plaintexts
                .par_iter()
                .zip(q_moduli.par_iter())
                .map(|(lhs_bound, &q_i)| lhs_bound + BigUint::from(q_i - 1))
                .collect(),
            p_max_traces: left_after_reduce
                .p_max_traces
                .par_iter()
                .zip(trace_multipliers.par_iter())
                .map(|(lhs_trace, multiplier)| lhs_trace + multiplier * &p_max)
                .collect(),
        };
        (
            RingGswSubEntryPlanKey { pre_full_reduce, reduce_levels, trace_multipliers },
            output_metadata,
        )
    }

    fn ensure_add_entry_subcircuit(
        &self,
        left: &NestedRnsPoly<P>,
        right: &NestedRnsPoly<P>,
        cache_key: &RingGswAddEntryPlanKey,
        circuit: &mut PolyCircuit<P>,
    ) -> usize {
        if let Some(existing) = self.ctx.add_entry_cache.get(cache_key) {
            return *existing.value();
        }
        let helper_storage = RingGswContext::<P>::shared_helper_storage(
            circuit.cloned_subcircuit_disk_storage(),
            "ring-gsw-add-entry",
        );
        let (subcircuit, _output_metadata) =
            RingGswContext::add_entry_subcircuit(circuit, left, right, helper_storage);
        let subcircuit_id = circuit.register_sub_circuit(subcircuit);
        self.ctx.add_entry_cache.insert(cache_key.clone(), subcircuit_id);
        subcircuit_id
    }

    fn ensure_sub_entry_subcircuit(
        &self,
        left: &NestedRnsPoly<P>,
        right: &NestedRnsPoly<P>,
        cache_key: &RingGswSubEntryPlanKey,
        circuit: &mut PolyCircuit<P>,
    ) -> usize {
        if let Some(existing) = self.ctx.sub_entry_cache.get(cache_key) {
            return *existing.value();
        }
        let helper_storage = RingGswContext::<P>::shared_helper_storage(
            circuit.cloned_subcircuit_disk_storage(),
            "ring-gsw-sub-entry",
        );
        let (subcircuit, _output_metadata) =
            RingGswContext::sub_entry_subcircuit(circuit, left, right, helper_storage);
        let subcircuit_id = circuit.register_sub_circuit(subcircuit);
        self.ctx.sub_entry_cache.insert(cache_key.clone(), subcircuit_id);
        subcircuit_id
    }

    fn call_entry_subcircuit(
        &self,
        left: &NestedRnsPoly<P>,
        right: &NestedRnsPoly<P>,
        subcircuit_id: usize,
        output_metadata: &RingGswEntryOutputMetadata,
        circuit: &mut PolyCircuit<P>,
    ) -> NestedRnsPoly<P> {
        let mut inputs = flatten_nested_rns_entries(std::slice::from_ref(left));
        inputs.extend(flatten_nested_rns_entries(std::slice::from_ref(right)));
        let outputs = circuit.call_sub_circuit(subcircuit_id, &inputs);
        nested_rns_from_flat_outputs(
            left,
            &outputs,
            output_metadata.max_plaintexts.clone(),
            output_metadata.p_max_traces.clone(),
        )
    }

    fn normalize_mul_entry(
        entry: &NestedRnsPoly<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> NestedRnsPoly<P> {
        let levels = entry.active_q_moduli().len();
        let (reduced_max_plaintexts, reduced_p_max_traces) =
            entry.ctx.full_reduce_output_metadata(Some(levels), Some(entry.level_offset));
        let needs_full_reduce = entry
            .max_plaintexts
            .iter()
            .zip(reduced_max_plaintexts.iter())
            .any(|(current, reduced)| current > reduced);
        let needs_trace_reduce = entry
            .p_max_traces
            .iter()
            .zip(reduced_p_max_traces.iter())
            .any(|(current, reduced)| current > reduced);
        if needs_full_reduce {
            entry.full_reduce(circuit)
        } else if needs_trace_reduce {
            entry.prepare_for_reconstruct(circuit)
        } else {
            entry.clone()
        }
    }

    fn normalize_mul_row(
        row: &[NestedRnsPoly<P>],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<NestedRnsPoly<P>> {
        row.iter().map(|entry| Self::normalize_mul_entry(entry, circuit)).collect()
    }

    pub fn new(
        ctx: Arc<RingGswContext<P>>,
        rows: [Vec<NestedRnsPoly<P>>; 2],
        randomizer_norm: PolyMatrixNorm,
        max_plaintext: BigUint,
    ) -> Self {
        let ciphertext = Self { ctx, rows, randomizer_norm, max_plaintext };
        ciphertext.assert_consistent();
        ciphertext
    }

    pub fn input(
        ctx: Arc<RingGswContext<P>>,
        max_plaintext: Option<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let [row0, row1] = Self::input_rows(
            ctx.nested_rns.clone(),
            ctx.width(),
            ctx.active_levels,
            ctx.level_offset,
            circuit,
        );
        let randomizer_norm = ctx.fresh_randomizer_norm();
        Self::new(
            ctx,
            [row0, row1],
            randomizer_norm,
            max_plaintext.unwrap_or_else(|| BigUint::from(1u64)),
        )
    }

    pub fn width(&self) -> usize {
        self.rows[0].len()
    }

    pub fn gadget_len(&self) -> usize {
        self.ctx.gadget_len()
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let (row0_plan, row1_plan) = rayon::join(
            || {
                Self::map_binary_row_entries(
                    &self.rows[0],
                    &other.rows[0],
                    Self::compute_add_entry_plan_and_output,
                )
            },
            || {
                Self::map_binary_row_entries(
                    &self.rows[1],
                    &other.rows[1],
                    Self::compute_add_entry_plan_and_output,
                )
            },
        );
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .zip(row0_plan.iter())
            .map(|((left, right), (cache_key, output_metadata))| {
                let subcircuit_id =
                    self.ensure_add_entry_subcircuit(left, right, cache_key, circuit);
                self.call_entry_subcircuit(left, right, subcircuit_id, output_metadata, circuit)
            })
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .zip(row1_plan.iter())
            .map(|((left, right), (cache_key, output_metadata))| {
                let subcircuit_id =
                    self.ensure_add_entry_subcircuit(left, right, cache_key, circuit);
                self.call_entry_subcircuit(left, right, subcircuit_id, output_metadata, circuit)
            })
            .collect::<Vec<_>>();
        let randomizer_norm = &self.randomizer_norm + &other.randomizer_norm;
        let max_plaintext = &self.max_plaintext + &other.max_plaintext;
        Self::new(self.ctx.clone(), [row0, row1], randomizer_norm, max_plaintext)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let (row0_plan, row1_plan) = rayon::join(
            || {
                Self::map_binary_row_entries(
                    &self.rows[0],
                    &other.rows[0],
                    Self::compute_sub_entry_plan_and_output,
                )
            },
            || {
                Self::map_binary_row_entries(
                    &self.rows[1],
                    &other.rows[1],
                    Self::compute_sub_entry_plan_and_output,
                )
            },
        );
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .zip(row0_plan.iter())
            .map(|((left, right), (cache_key, output_metadata))| {
                let subcircuit_id =
                    self.ensure_sub_entry_subcircuit(left, right, cache_key, circuit);
                self.call_entry_subcircuit(left, right, subcircuit_id, output_metadata, circuit)
            })
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .zip(row1_plan.iter())
            .map(|((left, right), (cache_key, output_metadata))| {
                let subcircuit_id =
                    self.ensure_sub_entry_subcircuit(left, right, cache_key, circuit);
                self.call_entry_subcircuit(left, right, subcircuit_id, output_metadata, circuit)
            })
            .collect::<Vec<_>>();
        let randomizer_norm = &self.randomizer_norm + &other.randomizer_norm;
        let max_plaintext = &self.max_plaintext + &other.max_plaintext;
        Self::new(self.ctx.clone(), [row0, row1], randomizer_norm, max_plaintext)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let start = Instant::now();
        let width = self.width();
        let helper_start = Instant::now();
        let template_entry =
            self.rows[0].first().expect("RingGswCiphertext must contain at least one column");
        let levels = template_entry.active_q_moduli().len();
        let p_moduli_depth = template_entry.ctx.p_moduli.len();
        let entry_size = levels * p_moduli_depth;
        let lhs_row0 = Self::normalize_mul_row(&self.rows[0], circuit);
        let lhs_row1 = Self::normalize_mul_row(&self.rows[1], circuit);
        let rhs_row0 = Self::normalize_mul_row(&other.rows[0], circuit);
        let rhs_row1 = Self::normalize_mul_row(&other.rows[1], circuit);
        debug!(
            "RingGswCiphertext::mul wrapper helper ready: width={}, elapsed_ms={}",
            width,
            helper_start.elapsed().as_millis()
        );
        let inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        let lhs_input_set_id = circuit.intern_input_set(&lhs_inputs);
        debug!(
            "RingGswCiphertext::mul wrapper inputs flattened: width={}, elapsed_ms={}",
            width,
            inputs_start.elapsed().as_millis()
        );

        let mul_start = Instant::now();
        let mut row0 = Vec::with_capacity(width);
        let mut row1 = Vec::with_capacity(width);
        let super_batch_columns = self.ctx.mul_super_batch_columns;
        for col_start in (0..width).step_by(super_batch_columns) {
            let col_end = (col_start + super_batch_columns).min(width);
            let actual_super_batch_columns = col_end - col_start;
            let current_super_batch_subcircuit_id = if actual_super_batch_columns ==
                self.ctx.mul_super_batch_columns
            {
                self.ctx.mul_super_batch_subcircuit_id
            } else if actual_super_batch_columns == self.ctx.mul_super_batch_tail_columns {
                self.ctx.mul_super_batch_tail_subcircuit_id
            } else {
                unreachable!(
                    "unexpected Ring-GSW mul super-batch width {}; configured super-batch={}, tail={}",
                    actual_super_batch_columns,
                    self.ctx.mul_super_batch_columns,
                    self.ctx.mul_super_batch_tail_columns
                );
            };
            let (rhs_row0_inputs, rhs_row1_inputs) = rayon::join(
                || flatten_nested_rns_entries(&rhs_row0[col_start..col_end]),
                || flatten_nested_rns_entries(&rhs_row1[col_start..col_end]),
            );
            let mut rhs_suffix = rhs_row0_inputs;
            rhs_suffix.extend(rhs_row1_inputs);
            let outputs = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
                current_super_batch_subcircuit_id,
                lhs_input_set_id,
                &rhs_suffix,
                &[],
            );
            debug_assert_eq!(outputs.len(), 2 * actual_super_batch_columns * entry_size);
            let (row0_batch_outputs, row1_batch_outputs) =
                outputs.split_at(actual_super_batch_columns * entry_size);
            row0.extend((0..actual_super_batch_columns).map(|batch_idx| {
                let start = batch_idx * entry_size;
                let end = start + entry_size;
                nested_rns_from_flat_outputs(
                    template_entry,
                    &row0_batch_outputs[start..end],
                    self.ctx.mul_output_max_plaintexts.clone(),
                    self.ctx.mul_output_p_max_traces.clone(),
                )
            }));
            row1.extend((0..actual_super_batch_columns).map(|batch_idx| {
                let start = batch_idx * entry_size;
                let end = start + entry_size;
                nested_rns_from_flat_outputs(
                    template_entry,
                    &row1_batch_outputs[start..end],
                    self.ctx.mul_output_max_plaintexts.clone(),
                    self.ctx.mul_output_p_max_traces.clone(),
                )
            }));
        }
        debug!(
            "RingGswCiphertext::mul subcircuit call finished: width={}, elapsed_ms={}",
            width,
            mul_start.elapsed().as_millis()
        );

        let randomizer_norm = (&self.randomizer_norm * self.ctx.decomposed_randomizer_norm()) +
            &other.randomizer_norm;
        let max_plaintext = &self.max_plaintext * &other.max_plaintext;
        let result = Self::new(self.ctx.clone(), [row0, row1], randomizer_norm, max_plaintext);
        debug!(
            "RingGswCiphertext::mul finished: width={}, entry_size={}, total_elapsed_ms={}",
            width,
            entry_size,
            start.elapsed().as_millis()
        );
        result
    }

    pub fn and(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::and requires lhs.max_plaintext == 1"
        );
        assert_eq!(
            other.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::and requires rhs.max_plaintext == 1"
        );
        self.mul(other, circuit)
    }

    pub fn xor(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            self.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::xor requires lhs.max_plaintext == 1"
        );
        assert_eq!(
            other.max_plaintext,
            BigUint::from(1u64),
            "RingGswCiphertext::xor requires rhs.max_plaintext == 1"
        );
        self.assert_compatible(other);
        let sum = self.add(other, circuit);
        let product = self.mul(other, circuit);
        let sum_minus_product = sum.sub(&product, circuit);
        let result = sum_minus_product.sub(&product, circuit);
        Self::new(result.ctx.clone(), result.rows, result.randomizer_norm, BigUint::from(1u64))
    }

    pub fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Vec<GateId> {
        let mut outputs = Vec::with_capacity(2 * self.width());
        for row in &self.rows {
            for entry in row {
                outputs.push(entry.reconstruct(circuit));
            }
        }
        outputs
    }

    fn input_rows(
        nested_rns: Arc<NestedRnsPolyContext>,
        width: usize,
        active_levels: usize,
        level_offset: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> [Vec<NestedRnsPoly<P>>; 2] {
        let row0 = (0..width)
            .map(|_| {
                NestedRnsPoly::input(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    circuit,
                )
            })
            .collect::<Vec<_>>();
        let row1 = (0..width)
            .map(|_| {
                NestedRnsPoly::input(
                    nested_rns.clone(),
                    Some(active_levels),
                    Some(level_offset),
                    circuit,
                )
            })
            .collect::<Vec<_>>();
        [row0, row1]
    }

    fn collapse_slots_to_single_poly(
        wire: GateId,
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> GateId {
        let mut collapsed_terms = (0..num_slots)
            .map(|slot| {
                let transferred = circuit.slot_transfer_gate(wire, &[(slot as u32, None)]);
                if slot == 0 { transferred } else { circuit.rotate_gate(transferred, slot as u64) }
            })
            .collect::<Vec<_>>();
        let mut collapsed =
            collapsed_terms.drain(..1).next().expect("slot-collapsing requires at least one slot");
        for term in collapsed_terms {
            collapsed = circuit.add_gate(collapsed, term);
        }
        collapsed
    }

    fn assert_consistent(&self) {
        let width = self.rows[0].len();
        assert!(width > 0, "RingGswCiphertext width must be positive");
        assert_eq!(self.rows[1].len(), width, "RingGswCiphertext rows must have matching widths");
        assert_eq!(
            width,
            self.ctx.width(),
            "RingGswCiphertext width {} must equal context width {}",
            width,
            self.ctx.width()
        );
        assert_eq!(
            self.randomizer_norm.nrow, width,
            "RingGswCiphertext randomizer trace rows {} must match width {}",
            self.randomizer_norm.nrow, width
        );
        assert_eq!(
            self.randomizer_norm.ncol, width,
            "RingGswCiphertext randomizer trace cols {} must match width {}",
            self.randomizer_norm.ncol, width
        );
        assert_eq!(
            self.randomizer_norm.ctx(),
            self.ctx.randomizer_norm_ctx.as_ref(),
            "RingGswCiphertext randomizer trace context must match the RingGswContext norm context"
        );

        for row in &self.rows {
            for entry in row {
                assert!(
                    Arc::ptr_eq(&entry.ctx, &self.ctx.nested_rns),
                    "RingGswCiphertext entries must share the RingGswContext NestedRnsPolyContext"
                );
                assert_eq!(
                    entry.level_offset, self.ctx.level_offset,
                    "RingGswCiphertext entries must share the RingGswContext q-level offset"
                );
                assert_eq!(
                    entry.enable_levels,
                    Some(self.ctx.active_levels),
                    "RingGswCiphertext entries must share the RingGswContext active-level configuration"
                );
                assert_eq!(
                    entry.active_q_moduli().len(),
                    self.ctx.active_levels,
                    "RingGswCiphertext entries must share the RingGswContext active q-window depth"
                );
            }
        }
    }

    fn assert_compatible(&self, other: &Self) {
        self.assert_consistent();
        other.assert_consistent();
        assert!(
            Arc::ptr_eq(&self.ctx, &other.ctx),
            "RingGswCiphertext operands must share the same RingGswContext"
        );
    }
}

impl<P: Poly + 'static> RingGswCiphertext<P> {
    pub fn estimate_decryption_error_norm(&self, error_sigma: f64) -> BigDecimal {
        self.assert_consistent();
        assert!(error_sigma.is_finite(), "error_sigma must be finite");
        assert!(error_sigma >= 0.0, "error_sigma must be non-negative");
        assert_eq!(
            self.width() % 2,
            0,
            "RingGswCiphertext width {} must be even to split into top/bottom halves",
            self.width()
        );
        let sigma = BigDecimal::from_f64(error_sigma)
            .expect("finite error_sigma must convert to BigDecimal");
        let (_top, bottom_half_randomizer) = self.randomizer_norm.split_rows(self.width() / 2);
        let max_p_modulus = *self
            .ctx
            .nested_rns
            .p_moduli
            .iter()
            .max()
            .expect("RingGswCiphertext requires at least one p modulus");
        let p_max_matrix = PolyMatrixNorm::new(
            self.ctx.randomizer_norm_ctx.clone(),
            bottom_half_randomizer.ncol,
            1,
            BigDecimal::from(max_p_modulus),
            None,
        );
        let public_key_error = PolyMatrixNorm::sample_gauss(
            self.ctx.randomizer_norm_ctx.clone(),
            1,
            bottom_half_randomizer.nrow,
            sigma,
        );
        let final_error = public_key_error * (bottom_half_randomizer * p_max_matrix);
        final_error.poly_norm.norm
    }

    pub fn decrypt<M>(
        &self,
        wire_secret_key: GateId,
        plaintext_modulus: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> GateId
    where
        M: PolyMatrix<P = P>,
    {
        self.assert_consistent();
        assert!(!plaintext_modulus.is_zero(), "plaintext_modulus must be positive");
        let gadget_len = self.gadget_len();
        assert_eq!(
            self.width(),
            2 * gadget_len,
            "RingGswCiphertext width {} must equal 2 * gadget_len {}",
            self.width(),
            gadget_len
        );

        let gadget_constants = nested_rns_gadget_vector::<P, M>(
            &self.ctx.params,
            self.ctx.nested_rns.as_ref(),
            Some(self.ctx.active_levels),
            Some(self.ctx.level_offset),
        )
        .get_row(0)
        .into_par_iter()
        .map(|entry| entry.coeffs_biguints()[0].clone())
        .collect::<Vec<_>>();
        assert_eq!(
            gadget_constants.len(),
            gadget_len,
            "Ring-GSW decrypt gadget vector length {} must match gadget_len {}",
            gadget_constants.len(),
            gadget_len
        );

        let active_q_moduli = self.rows[0][0].active_q_moduli();
        let scaled = active_q_moduli
            .iter()
            .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i)) /
            &plaintext_modulus;
        let scaled_poly = P::from_biguint_to_constant(&self.ctx.params, scaled);
        let scaled_g_inverse_matrix = nested_rns_gadget_decomposed::<P, M>(
            &self.ctx.params,
            self.ctx.nested_rns.as_ref(),
            &M::from_poly_vec_column(&self.ctx.params, vec![scaled_poly]),
            Some(self.ctx.active_levels),
            Some(self.ctx.level_offset),
        );
        let (scaled_rows, scaled_cols) = scaled_g_inverse_matrix.size();
        assert_eq!(
            scaled_cols, 1,
            "scaled gadget decomposition column count {} must equal 1",
            scaled_cols
        );
        let scaled_g_inverse = (0..scaled_rows)
            .map(|row_idx| {
                let coeff = scaled_g_inverse_matrix.entry(row_idx, 0).coeffs_biguints()[0].clone();
                active_q_moduli
                    .iter()
                    .map(|&q_i| {
                        (&coeff % BigUint::from(q_i))
                            .to_u64()
                            .expect("scaled gadget decomposition residue must fit in u64")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let top_entry =
            self.decrypt_linear_combination_row(&self.rows[0], &scaled_g_inverse, circuit);
        let bottom_entry =
            self.decrypt_linear_combination_row(&self.rows[1], &scaled_g_inverse, circuit);
        let prepared_top = top_entry.prepare_for_reconstruct(circuit);
        let p_depth = prepared_top.ctx.p_moduli.len();
        let mut weighted_top_terms = Vec::with_capacity(gadget_len);
        for q_idx in 0..prepared_top.active_q_moduli().len() {
            let level_base = q_idx * (p_depth + 1);
            let (ys, w) = prepared_top.decomposition_terms_for_level(q_idx, circuit);
            for p_idx in 0..p_depth {
                let collapsed =
                    Self::collapse_slots_to_single_poly(ys[p_idx], self.ctx.num_slots, circuit);
                let top_times_secret = circuit.mul_gate(collapsed, wire_secret_key);
                let gadget_scalar = &gadget_constants[level_base + p_idx];
                if gadget_scalar.is_zero() {
                    continue;
                }
                weighted_top_terms.push(
                    circuit.large_scalar_mul(top_times_secret, std::slice::from_ref(gadget_scalar)),
                );
            }
            let collapsed_w = Self::collapse_slots_to_single_poly(w, self.ctx.num_slots, circuit);
            let w_times_secret = circuit.mul_gate(collapsed_w, wire_secret_key);
            let gadget_scalar = &gadget_constants[level_base + p_depth];
            if gadget_scalar.is_zero() {
                continue;
            }
            weighted_top_terms.push(
                circuit.large_scalar_mul(w_times_secret, std::slice::from_ref(gadget_scalar)),
            );
        }
        let mut sum = circuit.large_scalar_mul(wire_secret_key, &[BigUint::ZERO]);
        for term in weighted_top_terms {
            sum = circuit.add_gate(sum, term);
        }
        let reconstructed_bottom = Self::collapse_slots_to_single_poly(
            bottom_entry.reconstruct(circuit),
            self.ctx.num_slots,
            circuit,
        );
        circuit.add_gate(sum, reconstructed_bottom)
    }

    fn decrypt_linear_combination_row(
        &self,
        row: &[NestedRnsPoly<P>],
        scaled_g_inverse: &[Vec<u64>],
        circuit: &mut PolyCircuit<P>,
    ) -> NestedRnsPoly<P> {
        assert_eq!(
            scaled_g_inverse.len(),
            self.gadget_len(),
            "scaled gadget decomposition length {} must match gadget_len {}",
            scaled_g_inverse.len(),
            self.gadget_len()
        );
        let zero_towers = vec![0u64; self.ctx.active_levels];
        let mut terms = scaled_g_inverse
            .iter()
            .enumerate()
            .filter(|(_idx, tower_constants)| tower_constants.iter().any(|&value| value != 0))
            .map(|(idx, tower_constants)| {
                row[self.gadget_len() + idx].const_mul(tower_constants, circuit)
            })
            .collect::<Vec<_>>();
        if terms.is_empty() {
            return row[0].const_mul(&zero_towers, circuit);
        }
        reduce_nested_rns_terms_pairwise(terms.split_off(0), circuit, |left, right, circuit| {
            left.add(right, circuit)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::arith::DEFAULT_MAX_UNREDUCED_MULS,
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use bigdecimal::BigDecimal;
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use rand::Rng;
    use std::sync::Arc;
    use tempfile::tempdir;

    const BASE_BITS: u32 = 6;
    const CRT_BITS: usize = 18;
    const ACTIVE_LEVELS: usize = 1;
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 4;
    fn create_test_context_with(
        circuit: &mut PolyCircuit<DCRTPoly>,
        ring_dim: u32,
        num_slots: usize,
        active_levels: usize,
        crt_bits: usize,
        p_moduli_bits: usize,
        max_unused_muls: usize,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, BASE_BITS);
        let ctx = Arc::new(RingGswContext::setup(
            circuit,
            &params,
            num_slots,
            p_moduli_bits,
            max_unused_muls,
            SCALE,
            Some(active_levels),
            None,
        ));
        (params, ctx)
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        create_test_context_with(
            circuit,
            NUM_SLOTS as u32,
            NUM_SLOTS,
            ACTIVE_LEVELS,
            CRT_BITS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
        )
    }

    fn sample_binary_input_pair() -> (u64, u64) {
        let mut rng = rand::rng();
        (rng.random_range(0..2u64), rng.random_range(0..2u64))
    }

    fn sample_hash_key() -> [u8; 32] {
        let mut rng = rand::rng();
        let mut key = [0u8; 32];
        rng.fill(&mut key);
        key
    }

    fn eval_outputs<P: Poly + 'static>(
        params: &P::Params,
        num_slots: usize,
        circuit: &PolyCircuit<P>,
        inputs: Vec<PolyVec<P>>,
    ) -> Vec<PolyVec<P>> {
        eval_outputs_with_parallel_gates(params, num_slots, circuit, inputs, None)
    }

    fn eval_outputs_with_parallel_gates<P: Poly + 'static>(
        params: &P::Params,
        num_slots: usize,
        circuit: &PolyCircuit<P>,
        inputs: Vec<PolyVec<P>>,
        parallel_gates: Option<usize>,
    ) -> Vec<PolyVec<P>> {
        let one = PolyVec::new(vec![P::const_one(params); num_slots]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            parallel_gates,
        )
    }

    fn rounded_coeffs<P: Poly>(
        decrypted: &P,
        plaintext_modulus: u64,
        q_modulus: &BigUint,
    ) -> Vec<u64> {
        let half_q = q_modulus / BigUint::from(2u64);
        decrypted
            .coeffs_biguints()
            .into_iter()
            .map(|slot| {
                let rounded = (BigUint::from(plaintext_modulus) * slot + &half_q) / q_modulus;
                rounded.to_u64().expect("rounded plaintext slot must fit in u64")
            })
            .collect()
    }

    fn expected_coeffs(expected: u64) -> Vec<u64> {
        let mut coeffs = vec![0u64; NUM_SLOTS];
        coeffs[0] = expected;
        coeffs
    }

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        include!("ring_gsw_gpu_tests.rs");
    }

    #[test]
    fn test_ring_gsw_input_randomizer_norm_starts_with_width_by_width_unit_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let input = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);

        assert_eq!(input.randomizer_norm, ctx.fresh_randomizer_norm());
        assert_eq!(input.randomizer_norm.nrow, ctx.width());
        assert_eq!(input.randomizer_norm.ncol, ctx.width());
        assert_eq!(input.randomizer_norm.poly_norm.norm, BigDecimal::from(1u64));
        assert_eq!(input.max_plaintext, BigUint::from(1u64));
    }

    #[test]
    fn test_ring_gsw_input_max_plaintext_accepts_explicit_override() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let input = RingGswCiphertext::input(ctx, Some(BigUint::from(7u64)), &mut circuit);

        assert_eq!(input.max_plaintext, BigUint::from(7u64));
    }

    #[test]
    fn test_ring_gsw_add_randomizer_norm_and_sub_trace_sum_plaintext_bounds() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(3u64)), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(5u64)), &mut circuit);

        let expected = &lhs.randomizer_norm + &rhs.randomizer_norm;
        let sum = lhs.add(&rhs, &mut circuit);
        let difference = lhs.sub(&rhs, &mut circuit);

        assert_eq!(sum.randomizer_norm, expected);
        assert_eq!(sum.max_plaintext, BigUint::from(8u64));
        assert_eq!(difference.randomizer_norm, expected);
        assert_eq!(difference.max_plaintext, BigUint::from(8u64));
    }

    #[test]
    fn test_ring_gsw_mul_randomizer_norm_traces_plaintext_product() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(3u64)), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(5u64)), &mut circuit);

        let expected =
            (&lhs.randomizer_norm * ctx.decomposed_randomizer_norm()) + &rhs.randomizer_norm;
        let product = lhs.mul(&rhs, &mut circuit);

        assert_eq!(product.randomizer_norm, expected);
        assert_eq!(product.max_plaintext, BigUint::from(15u64));
    }

    #[test]
    fn test_ring_gsw_xor_keeps_boolean_plaintext_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, None, &mut circuit);

        let result = lhs.xor(&rhs, &mut circuit);

        assert_eq!(result.max_plaintext, BigUint::from(1u64));
    }

    #[test]
    #[should_panic(expected = "RingGswCiphertext::and requires lhs.max_plaintext == 1")]
    fn test_ring_gsw_and_rejects_non_boolean_plaintext_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), Some(BigUint::from(2u64)), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, None, &mut circuit);

        let _ = lhs.and(&rhs, &mut circuit);
    }

    #[test]
    #[should_panic(expected = "RingGswCiphertext::xor requires rhs.max_plaintext == 1")]
    fn test_ring_gsw_xor_rejects_non_boolean_plaintext_bound() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, Some(BigUint::from(2u64)), &mut circuit);

        let _ = lhs.xor(&rhs, &mut circuit);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_encrypt_roundtrip_matches_circuit_and_native_decrypt_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let ciphertext_input = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1)[0];
        let decrypted_input = ciphertext_input.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(2u64),
            &mut circuit,
        );
        circuit.output(vec![decrypted_input]);
        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            ctx.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        for (plaintext, tag) in [(0u64, b"roundtrip_zero".as_slice()), (1u64, b"roundtrip_one")] {
            let ciphertext = encrypt_plaintext_bit(
                &params,
                ctx.nested_rns.as_ref(),
                &public_key,
                plaintext,
                randomizer_hash_key,
                tag,
            );
            let expected = expected_coeffs(plaintext);
            let decrypted =
                decrypt_ciphertext(&params, ctx.nested_rns.as_ref(), &ciphertext, &secret_key, 2);
            assert_eq!(
                rounded_coeffs(&decrypted, 2, &q_modulus),
                expected,
                "native Ring-GSW encrypt/decrypt should recover the plaintext exactly when e = 0"
            );
            let circuit_inputs = [
                ciphertext_inputs_from_native(
                    &params,
                    ctx.nested_rns.as_ref(),
                    &ciphertext,
                    0,
                    Some(ctx.active_levels),
                ),
                vec![PolyVec::new(vec![secret_key.clone()])],
            ]
            .concat();
            let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, circuit_inputs);
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].len(), 1);
            assert_eq!(
                rounded_coeffs(&outputs[0].as_slice()[0], 2, &q_modulus),
                expected_coeffs(plaintext),
                "in-circuit Ring-GSW decrypt should recover the plaintext exactly when e = 0"
            );
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_add_circuit_decrypts_to_expected_integer_sum_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1)[0];
        let plaintext_modulus = 3u64;
        let sum = lhs.add(&rhs, &mut circuit);
        let decrypted_sum = sum.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_sum]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let expected = (x1 + x2) % plaintext_modulus;
        let lhs_tag = format!("add_circuit_lhs_{x1}_{x2}");
        let rhs_tag = format!("add_circuit_rhs_{x1}_{x2}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW addition should decrypt in-circuit to the plaintext-modulus sum for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_sub_circuit_decrypts_to_expected_integer_difference_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1)[0];
        let plaintext_modulus = 3u64;
        let difference = lhs.sub(&rhs, &mut circuit);
        let decrypted_difference = difference.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_difference]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let expected = (x1 + plaintext_modulus - x2) % plaintext_modulus;
        let lhs_tag = format!("sub_circuit_lhs_{x1}_{x2}");
        let rhs_tag = format!("sub_circuit_rhs_{x1}_{x2}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW subtraction should decrypt in-circuit to the plaintext-modulus difference for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_mul_circuit_decrypts_to_expected_integer_product_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1)[0];
        let plaintext_modulus = 2u64;
        let product = lhs.mul(&rhs, &mut circuit);
        let decrypted_product = product.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_product]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let expected = (x1 * x2) % plaintext_modulus;
        let lhs_tag = format!("mul_circuit_lhs_{x1}_{x2}");
        let rhs_tag = format!("mul_circuit_rhs_{x1}_{x2}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW multiplication should decrypt in-circuit to the plaintext-modulus product for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_chained_mul_circuit_decrypts_to_expected_integer_product_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs1 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs2 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let wire_secret_key = circuit.input(1)[0];
        let plaintext_modulus = 2u64;
        let product = lhs.mul(&rhs1, &mut circuit);
        let chained_product = product.mul(&rhs2, &mut circuit);
        let decrypted_product = chained_product.decrypt::<DCRTPolyMatrix>(
            wire_secret_key,
            BigUint::from(plaintext_modulus),
            &mut circuit,
        );
        circuit.output(vec![decrypted_product]);

        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            lhs.width(),
            &secret_key,
            public_key_hash_key,
            b"ring_gsw_public_key",
            None,
        );
        let q_modulus = BigUint::from(ctx.nested_rns.q_moduli()[0]);

        let (x1, x2) = sample_binary_input_pair();
        let x3 = rand::rng().random_range(0..2u64);
        let expected = (x1 * x2 * x3) % plaintext_modulus;
        let lhs_tag = format!("chain_mul_circuit_lhs_{x1}_{x2}_{x3}");
        let rhs1_tag = format!("chain_mul_circuit_rhs1_{x1}_{x2}_{x3}");
        let rhs2_tag = format!("chain_mul_circuit_rhs2_{x1}_{x2}_{x3}");
        let lhs_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x1,
            randomizer_hash_key,
            lhs_tag.as_bytes(),
        );
        let rhs1_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x2,
            randomizer_hash_key,
            rhs1_tag.as_bytes(),
        );
        let rhs2_native = encrypt_plaintext_bit(
            &params,
            ctx.nested_rns.as_ref(),
            &public_key,
            x3,
            randomizer_hash_key,
            rhs2_tag.as_bytes(),
        );

        let inputs = [
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &lhs_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs1_native,
                0,
                Some(ctx.active_levels),
            ),
            ciphertext_inputs_from_native(
                &params,
                ctx.nested_rns.as_ref(),
                &rhs2_native,
                0,
                Some(ctx.active_levels),
            ),
            vec![PolyVec::new(vec![secret_key.clone()])],
        ]
        .concat();
        let outputs = eval_outputs(&params, NUM_SLOTS, &circuit, inputs);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(
            rounded_coeffs(&outputs[0].as_slice()[0], plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Chained Ring-GSW multiplication should decrypt in-circuit to the plaintext-modulus product for sampled x1={x1}, x2={x2}, x3={x3}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    #[ignore = "expensive circuit-structure reporting test; run with --ignored --nocapture"]
    fn test_ring_gsw_mul_large_circuit_metrics() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
        let crt_bits = 24usize;
        let crt_depth = 1usize;
        let ring_dim = 1u32 << 16;
        let num_slots = 1usize << 16;
        let p_moduli_bits = 7;
        let max_unused_muls = 4;

        let mul1_disk_dir = tempdir().expect("create temp dir for disk-backed sub-circuits");
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.enable_subcircuits_in_disk(mul1_disk_dir.path());
        let (_params, ctx) = create_test_context_with(
            &mut circuit,
            ring_dim,
            num_slots,
            crt_depth,
            crt_bits,
            p_moduli_bits,
            max_unused_muls,
        );
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let product = lhs.mul(&rhs, &mut circuit);
        let outputs = product.reconstruct(&mut circuit);
        circuit.output(outputs);

        println!(
            "mul 1 ring_gsw_mul metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        let mul1_depth = circuit.non_free_depth();
        println!("mul 1 non-free depth end {}", mul1_depth);
        println!("mul 1 gate counts {:?}", circuit.count_gates_by_type_vec());

        let mul2_disk_dir = tempdir().expect("create temp dir for disk-backed sub-circuits");
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.enable_subcircuits_in_disk(mul2_disk_dir.path());
        let (_params, ctx) = create_test_context_with(
            &mut circuit,
            ring_dim,
            num_slots,
            crt_depth,
            crt_bits,
            p_moduli_bits,
            max_unused_muls,
        );
        let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs1 = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
        let rhs2 = RingGswCiphertext::input(ctx, None, &mut circuit);
        let product1 = lhs.mul(&rhs1, &mut circuit);
        let product2 = product1.mul(&rhs2, &mut circuit);
        let outputs = product2.reconstruct(&mut circuit);
        circuit.output(outputs);

        println!(
            "mul 2 ring_gsw_mul metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        let mul2_depth = circuit.non_free_depth();
        println!("mul 2 non-free depth end {}", mul2_depth);
        println!("mul 2 gate counts {:?}", circuit.count_gates_by_type_vec());

        println!("mul 2 vs mul 1 depth increase: {}", mul2_depth - mul1_depth);
    }
}
