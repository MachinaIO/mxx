use crate::{
    circuit::{PolyCircuit, evaluable::PolyVec, gate::GateId},
    gadgets::{
        arith::{
            NestedRnsPoly, NestedRnsPolyContext, nested_rns_gadget_decomposed,
            nested_rns_gadget_vector,
        },
        conv_mul::negacyclic_conv_mul_right_sparse,
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
};
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};
use tracing::debug;

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
        .iter()
        .flat_map(|entry| entry.inner.iter().flat_map(|level| level.iter().copied()))
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

pub type NativeRingGswCiphertext = [Vec<DCRTPoly>; 2];

fn active_q_modulus(ctx: &NestedRnsPolyContext) -> BigUint {
    BigUint::from(*ctx.q_moduli().first().expect("Ring-GSW helpers require one active q modulus"))
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

fn native_gadget_decompose(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    input_poly: &DCRTPoly,
) -> Vec<DCRTPoly> {
    let decomposed = nested_rns_gadget_decomposed::<DCRTPoly, DCRTPolyMatrix>(
        params,
        ctx,
        &DCRTPolyMatrix::from_poly_vec(params, vec![vec![input_poly.clone()]]),
        None,
        None,
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
        .iter()
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

pub fn ciphertext_inputs_from_native(
    params: &DCRTPolyParams,
    ctx: &NestedRnsPolyContext,
    ciphertext: &NativeRingGswCiphertext,
    level_offset: usize,
    enable_levels: Option<usize>,
) -> Vec<PolyVec<DCRTPoly>> {
    let mut inputs = Vec::new();
    for row in ciphertext {
        for poly in row {
            inputs.extend(encode_nested_rns_poly_vec_with_offset::<DCRTPoly>(
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
    let row0 = (0..width)
        .map(|idx| ciphertext_poly_from_output(params, &outputs[idx]))
        .collect::<Vec<_>>();
    let row1 = (0..width)
        .map(|idx| ciphertext_poly_from_output(params, &outputs[width + idx]))
        .collect::<Vec<_>>();
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
        .iter()
        .zip(ciphertext[1].iter())
        .zip(g_inverse.iter())
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
    pub mul_column_subcircuit_id: usize,
}

impl<P: Poly> RingGswContext<P> {
    pub fn width(&self) -> usize {
        2 * self.active_levels * (self.nested_rns.p_moduli.len() + 1)
    }

    pub fn gadget_len(&self) -> usize {
        self.active_levels * (self.nested_rns.p_moduli.len() + 1)
    }
}

impl<P: Poly + 'static> RingGswContext<P> {
    fn mul_sparse_entry_subcircuit(
        params: &P::Params,
        num_slots: usize,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        sparse_q_idx: usize,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let ctx = Arc::new(template_ctx.register_subcircuits_in(&mut circuit));
        let lhs = NestedRnsPoly::input(
            ctx.clone(),
            Some(active_levels),
            Some(level_offset),
            &mut circuit,
        );
        let rhs = NestedRnsPoly::input(ctx, Some(active_levels), Some(level_offset), &mut circuit);
        let product = negacyclic_conv_mul_right_sparse(
            params,
            &mut circuit,
            &lhs,
            &rhs,
            sparse_q_idx,
            num_slots,
        );
        circuit.output(product.inner.into_iter().flatten().collect());
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
        let helper_build_start = Instant::now();
        let mul_column_subcircuit = Self::mul_column_subcircuit(
            params,
            num_slots,
            nested_rns.as_ref(),
            active_levels,
            level_offset,
            width,
        );
        debug!(
            "RingGswContext::setup helper subcircuit built: width={}, elapsed_ms={}",
            width,
            helper_build_start.elapsed().as_millis()
        );
        let mul_column_subcircuit_id = circuit.register_sub_circuit(mul_column_subcircuit);
        debug!(
            "RingGswContext::setup completed: width={}, total_elapsed_ms={}",
            width,
            setup_start.elapsed().as_millis()
        );
        Self {
            params: params.clone(),
            num_slots,
            nested_rns,
            level_offset,
            active_levels,
            mul_column_subcircuit_id,
        }
    }

    fn mul_column_subcircuit(
        params: &P::Params,
        num_slots: usize,
        template_ctx: &NestedRnsPolyContext,
        active_levels: usize,
        level_offset: usize,
        width: usize,
    ) -> PolyCircuit<P> {
        let start = Instant::now();
        let mut circuit = PolyCircuit::<P>::new();
        let nested_rns = Arc::new(template_ctx.register_subcircuits_in(&mut circuit));
        let dot_helper_start = Instant::now();
        let local_dot_row_with_column_subcircuit_id = {
            let mut helper_circuit = PolyCircuit::<P>::new();
            let helper_ctx = Arc::new(template_ctx.register_subcircuits_in(&mut helper_circuit));
            let chunk_width = helper_ctx.p_moduli.len() + 1;
            let gadget_len = active_levels * chunk_width;
            assert_eq!(
                width,
                2 * gadget_len,
                "Ring-GSW mul helper width {} must equal 2 * gadget_len {}",
                width,
                gadget_len
            );
            let product_subcircuits_start = Instant::now();
            let product_subcircuits = (0..width)
                .into_par_iter()
                .map(|col_idx| {
                    let sparse_q_idx = (col_idx % gadget_len) / chunk_width;
                    Self::mul_sparse_entry_subcircuit(
                        params,
                        num_slots,
                        template_ctx,
                        active_levels,
                        level_offset,
                        sparse_q_idx,
                    )
                })
                .collect::<Vec<_>>();
            debug!(
                "RingGswContext::mul_column_subcircuit product helpers built in parallel: width={}, elapsed_ms={}",
                width,
                product_subcircuits_start.elapsed().as_millis()
            );
            let row = (0..width)
                .map(|_| {
                    NestedRnsPoly::input(
                        helper_ctx.clone(),
                        Some(active_levels),
                        Some(level_offset),
                        &mut helper_circuit,
                    )
                })
                .collect::<Vec<_>>();
            let column = (0..width)
                .map(|_| {
                    NestedRnsPoly::input(
                        helper_ctx.clone(),
                        Some(active_levels),
                        Some(level_offset),
                        &mut helper_circuit,
                    )
                })
                .collect::<Vec<_>>();
            let product_subcircuit_ids = product_subcircuits
                .into_iter()
                .map(|subcircuit| helper_circuit.register_sub_circuit(subcircuit))
                .collect::<Vec<_>>();
            let levels = row[0].active_q_moduli().len();
            let max_plaintexts = vec![row[0].ctx.max_plaintext_below_p_full(); levels];
            let p_max_traces = vec![row[0].ctx.max_p_max_trace_below_lut_map_size(); levels];
            let products = row
                .iter()
                .zip(column.iter())
                .enumerate()
                .map(|(col_idx, (lhs, rhs))| {
                    let mut inputs = flatten_nested_rns_entries(std::slice::from_ref(lhs));
                    inputs.extend(flatten_nested_rns_entries(std::slice::from_ref(rhs)));
                    let outputs =
                        helper_circuit.call_sub_circuit(product_subcircuit_ids[col_idx], &inputs);
                    nested_rns_from_flat_outputs(
                        lhs,
                        &outputs,
                        max_plaintexts.clone(),
                        p_max_traces.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let result = reduce_nested_rns_terms_pairwise(
                products,
                &mut helper_circuit,
                |lhs, rhs, helper_circuit| lhs.add(rhs, helper_circuit),
            );
            helper_circuit.output(result.inner.into_iter().flatten().collect());
            circuit.register_sub_circuit(helper_circuit)
        };
        debug!(
            "RingGswContext::mul_column_subcircuit dot helper ready: width={}, elapsed_ms={}",
            width,
            dot_helper_start.elapsed().as_millis()
        );
        let input_start = Instant::now();
        let [lhs_row0, lhs_row1] = RingGswCiphertext::<P>::input_rows(
            nested_rns.clone(),
            width,
            active_levels,
            level_offset,
            &mut circuit,
        );
        let rhs_top = NestedRnsPoly::input(
            nested_rns.clone(),
            Some(active_levels),
            Some(level_offset),
            &mut circuit,
        );
        let rhs_bottom =
            NestedRnsPoly::input(nested_rns, Some(active_levels), Some(level_offset), &mut circuit);
        debug!(
            "RingGswContext::mul_column_subcircuit inputs allocated: width={}, elapsed_ms={}",
            width,
            input_start.elapsed().as_millis()
        );
        let g_inverse_start = Instant::now();
        let mut g_inverse_col = rhs_top.gadget_decompose(&mut circuit);
        g_inverse_col.extend(rhs_bottom.gadget_decompose(&mut circuit));
        debug!(
            "RingGswContext::mul_column_subcircuit g_inverse built: width={}, elapsed_ms={}",
            width,
            g_inverse_start.elapsed().as_millis()
        );
        let dot_products_start = Instant::now();
        let g_inverse_inputs = flatten_nested_rns_entries(&g_inverse_col);
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&lhs_row0),
            || flatten_nested_rns_entries(&lhs_row1),
        );
        let row0: NestedRnsPoly<P> = {
            let template = &lhs_row0[0];
            let mut inputs = lhs_row0_inputs;
            inputs.extend(g_inverse_inputs.clone());
            let outputs =
                circuit.call_sub_circuit(local_dot_row_with_column_subcircuit_id, &inputs);
            let levels = template.active_q_moduli().len();
            let max_plaintexts = vec![template.ctx.max_plaintext_below_p_full(); levels];
            let p_max_traces = vec![template.ctx.max_p_max_trace_below_lut_map_size(); levels];
            nested_rns_from_flat_outputs(template, &outputs, max_plaintexts, p_max_traces)
        };
        let row1: NestedRnsPoly<P> = {
            let template = &lhs_row1[0];
            let mut inputs = lhs_row1_inputs;
            inputs.extend(g_inverse_inputs);
            let outputs =
                circuit.call_sub_circuit(local_dot_row_with_column_subcircuit_id, &inputs);
            let levels = template.active_q_moduli().len();
            let max_plaintexts = vec![template.ctx.max_plaintext_below_p_full(); levels];
            let p_max_traces = vec![template.ctx.max_p_max_trace_below_lut_map_size(); levels];
            nested_rns_from_flat_outputs(template, &outputs, max_plaintexts, p_max_traces)
        };
        circuit.output(flatten_nested_rns_entries(&[row0, row1]));
        debug!(
            "RingGswContext::mul_column_subcircuit finished: width={}, dot_products_elapsed_ms={}, total_elapsed_ms={}",
            width,
            dot_products_start.elapsed().as_millis(),
            start.elapsed().as_millis()
        );
        circuit
    }
}

#[derive(Debug, Clone)]
pub struct RingGswCiphertext<P: Poly> {
    pub ctx: Arc<RingGswContext<P>>,
    pub rows: [Vec<NestedRnsPoly<P>>; 2],
}

impl<P: Poly + 'static> RingGswCiphertext<P> {
    pub fn new(ctx: Arc<RingGswContext<P>>, rows: [Vec<NestedRnsPoly<P>>; 2]) -> Self {
        let ciphertext = Self { ctx, rows };
        ciphertext.assert_consistent();
        ciphertext
    }

    pub fn input(ctx: Arc<RingGswContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let [row0, row1] = Self::input_rows(
            ctx.nested_rns.clone(),
            ctx.width(),
            ctx.active_levels,
            ctx.level_offset,
            circuit,
        );
        Self::new(ctx, [row0, row1])
    }

    pub fn width(&self) -> usize {
        self.rows[0].len()
    }

    pub fn gadget_len(&self) -> usize {
        self.ctx.gadget_len()
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .map(|(lhs, rhs)| lhs.add(rhs, circuit))
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .map(|(lhs, rhs)| lhs.add(rhs, circuit))
            .collect::<Vec<_>>();
        Self::new(self.ctx.clone(), [row0, row1])
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let row0 = self.rows[0]
            .iter()
            .zip(other.rows[0].iter())
            .map(|(lhs, rhs)| lhs.sub(rhs, circuit))
            .collect::<Vec<_>>();
        let row1 = self.rows[1]
            .iter()
            .zip(other.rows[1].iter())
            .map(|(lhs, rhs)| lhs.sub(rhs, circuit))
            .collect::<Vec<_>>();
        Self::new(self.ctx.clone(), [row0, row1])
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let start = Instant::now();
        let width = self.width();
        let template_entry =
            self.rows[0].first().expect("RingGswCiphertext must contain at least one column");
        let levels = template_entry.active_q_moduli().len();
        let p_moduli_depth = template_entry.ctx.p_moduli.len();
        let entry_size = levels * p_moduli_depth;
        let lhs_inputs_start = Instant::now();
        let (lhs_row0_inputs, lhs_row1_inputs) = rayon::join(
            || flatten_nested_rns_entries(&self.rows[0]),
            || flatten_nested_rns_entries(&self.rows[1]),
        );
        let mut lhs_inputs = lhs_row0_inputs;
        lhs_inputs.extend(lhs_row1_inputs);
        debug!(
            "RingGswCiphertext::mul lhs inputs flattened: width={}, elapsed_ms={}",
            width,
            lhs_inputs_start.elapsed().as_millis()
        );
        let rhs_inputs_start = Instant::now();
        let rhs_column_inputs = (0..width)
            .into_par_iter()
            .map(|col_idx| {
                let mut inputs =
                    flatten_nested_rns_entries(std::slice::from_ref(&other.rows[0][col_idx]));
                inputs.extend(flatten_nested_rns_entries(std::slice::from_ref(
                    &other.rows[1][col_idx],
                )));
                inputs
            })
            .collect::<Vec<_>>();
        debug!(
            "RingGswCiphertext::mul rhs column inputs flattened in parallel: width={}, elapsed_ms={}",
            width,
            rhs_inputs_start.elapsed().as_millis()
        );
        let max_plaintexts = vec![template_entry.ctx.max_plaintext_below_p_full(); levels];
        let p_max_traces = vec![template_entry.ctx.max_p_max_trace_below_lut_map_size(); levels];
        let mut row0 = Vec::with_capacity(width);
        let mut row1 = Vec::with_capacity(width);

        let column_loop_start = Instant::now();
        for column_inputs in rhs_column_inputs {
            let mut inputs = lhs_inputs.clone();
            inputs.extend(column_inputs);
            let outputs = circuit.call_sub_circuit(self.ctx.mul_column_subcircuit_id, &inputs);
            assert_eq!(
                outputs.len(),
                2 * entry_size,
                "Ring-GSW mul-column subcircuit output size must match two ciphertext entries"
            );
            row0.push(
                NestedRnsPoly::new(
                    template_entry.ctx.clone(),
                    outputs[..entry_size]
                        .chunks(p_moduli_depth)
                        .map(|row| row.to_vec())
                        .collect::<Vec<_>>(),
                    Some(template_entry.level_offset),
                    template_entry.enable_levels,
                    max_plaintexts.clone(),
                )
                .with_p_max_traces(p_max_traces.clone()),
            );
            row1.push(
                NestedRnsPoly::new(
                    template_entry.ctx.clone(),
                    outputs[entry_size..]
                        .chunks(p_moduli_depth)
                        .map(|row| row.to_vec())
                        .collect::<Vec<_>>(),
                    Some(template_entry.level_offset),
                    template_entry.enable_levels,
                    max_plaintexts.clone(),
                )
                .with_p_max_traces(p_max_traces.clone()),
            );
        }
        debug!(
            "RingGswCiphertext::mul column loop finished: width={}, elapsed_ms={}",
            width,
            column_loop_start.elapsed().as_millis()
        );

        let result = Self::new(self.ctx.clone(), [row0, row1]);
        debug!(
            "RingGswCiphertext::mul finished: width={}, entry_size={}, total_elapsed_ms={}",
            width,
            entry_size,
            start.elapsed().as_millis()
        );
        result
    }

    pub fn and(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.mul(other, circuit)
    }

    pub fn xor(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible(other);
        let sum = self.add(other, circuit);
        let product = self.mul(other, circuit);
        let sum_minus_product = sum.sub(&product, circuit);
        sum_minus_product.sub(&product, circuit)
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
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use rand::Rng;
    use std::sync::Arc;

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
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, BASE_BITS);
        let ctx = Arc::new(RingGswContext::setup(
            circuit,
            &params,
            num_slots,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            Some(active_levels),
            None,
        ));
        (params, ctx)
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        create_test_context_with(circuit, NUM_SLOTS as u32, NUM_SLOTS, ACTIVE_LEVELS, CRT_BITS)
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

    fn eval_outputs(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let one = PolyVec::new(vec![DCRTPoly::const_one(params); NUM_SLOTS]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        )
    }

    fn rounded_coeffs(
        decrypted: &DCRTPoly,
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

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_native_encrypt_decrypt_roundtrip_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
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
            let decrypted =
                decrypt_ciphertext(&params, ctx.nested_rns.as_ref(), &ciphertext, &secret_key, 2);
            assert_eq!(
                rounded_coeffs(&decrypted, 2, &q_modulus),
                {
                    let mut coeffs = vec![0u64; NUM_SLOTS];
                    coeffs[0] = plaintext;
                    coeffs
                },
                "native Ring-GSW encrypt/decrypt should recover the plaintext exactly when e = 0"
            );
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_add_decrypts_to_expected_integer_sum_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let sum = lhs.add(&rhs, &mut circuit);
        let reconstructed_sum = sum.reconstruct(&mut circuit);
        circuit.output(reconstructed_sum);
        let plaintext_modulus = 3u64;

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
        let lhs_tag = format!("add_lhs_{x1}_{x2}");
        let rhs_tag = format!("add_rhs_{x1}_{x2}");
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
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let decrypted = decrypt_ciphertext(
            &params,
            ctx.nested_rns.as_ref(),
            &reconstructed,
            &secret_key,
            plaintext_modulus,
        );
        assert_eq!(
            rounded_coeffs(&decrypted, plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW addition should decrypt to the plaintext-modulus sum for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_sub_decrypts_to_expected_integer_difference_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let difference = lhs.sub(&rhs, &mut circuit);
        let reconstructed_difference = difference.reconstruct(&mut circuit);
        circuit.output(reconstructed_difference);
        let plaintext_modulus = 3u64;

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
        let lhs_tag = format!("sub_lhs_{x1}_{x2}");
        let rhs_tag = format!("sub_rhs_{x1}_{x2}");
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
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let decrypted = decrypt_ciphertext(
            &params,
            ctx.nested_rns.as_ref(),
            &reconstructed,
            &secret_key,
            plaintext_modulus,
        );
        assert_eq!(
            rounded_coeffs(&decrypted, plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW subtraction should decrypt to the plaintext-modulus difference for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_mul_decrypts_to_expected_integer_product_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let product = lhs.mul(&rhs, &mut circuit);
        let reconstructed_product = product.reconstruct(&mut circuit);
        circuit.output(reconstructed_product);
        let plaintext_modulus = 2u64;

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
        let lhs_tag = format!("mul_lhs_{x1}_{x2}");
        let rhs_tag = format!("mul_rhs_{x1}_{x2}");
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
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let decrypted = decrypt_ciphertext(
            &params,
            ctx.nested_rns.as_ref(),
            &reconstructed,
            &secret_key,
            plaintext_modulus,
        );
        assert_eq!(
            rounded_coeffs(&decrypted, plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW multiplication should decrypt to the plaintext-modulus product for sampled x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_xor_decrypts_to_expected_boolean_result_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let xor = lhs.xor(&rhs, &mut circuit);
        let reconstructed_xor = xor.reconstruct(&mut circuit);
        circuit.output(reconstructed_xor);

        let plaintext_modulus = 3u64;

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
        let expected = x1 ^ x2;
        let lhs_tag = format!("xor_lhs_{x1}_{x2}");
        let rhs_tag = format!("xor_rhs_{x1}_{x2}");
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
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let decrypted = decrypt_ciphertext(
            &params,
            ctx.nested_rns.as_ref(),
            &reconstructed,
            &secret_key,
            plaintext_modulus,
        );
        assert_eq!(
            rounded_coeffs(&decrypted, plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW xor should decrypt to the Boolean xor of the sampled input bits for x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_ring_gsw_and_decrypts_to_expected_boolean_result_without_error() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let lhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let and = lhs.and(&rhs, &mut circuit);
        let reconstructed_and = and.reconstruct(&mut circuit);
        circuit.output(reconstructed_and);

        let plaintext_modulus = 2u64;

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
        let expected = x1 & x2;
        let lhs_tag = format!("and_lhs_{x1}_{x2}");
        let rhs_tag = format!("and_rhs_{x1}_{x2}");
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
        ]
        .concat();
        let outputs = eval_outputs(&params, &circuit, inputs);
        let reconstructed = ciphertext_from_outputs(&params, &outputs, lhs.width());
        let decrypted = decrypt_ciphertext(
            &params,
            ctx.nested_rns.as_ref(),
            &reconstructed,
            &secret_key,
            plaintext_modulus,
        );
        assert_eq!(
            rounded_coeffs(&decrypted, plaintext_modulus, &q_modulus),
            expected_coeffs(expected),
            "Ring-GSW and should decrypt to the Boolean and of the sampled input bits for x1={x1}, x2={x2}"
        );
    }

    #[sequential_test::sequential]
    #[test]
    // #[ignore = "expensive circuit-structure reporting test; run with --ignored --nocapture"]
    fn test_ring_gsw_mul_large_circuit_metrics() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
        let crt_bits = 24usize;
        let crt_depth = 1usize;
        let ring_dim = 1u32 << 10;
        let num_slots = 1usize << 10;

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_params, ctx) =
            create_test_context_with(&mut circuit, ring_dim, num_slots, crt_depth, crt_bits);
        let lhs = RingGswCiphertext::input(ctx.clone(), &mut circuit);
        let rhs = RingGswCiphertext::input(ctx, &mut circuit);
        let product = lhs.mul(&rhs, &mut circuit);
        let outputs = product.reconstruct(&mut circuit);
        circuit.output(outputs);

        println!(
            "ring_gsw_mul metrics: crt_bits={crt_bits}, crt_depth={crt_depth}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        println!("non-free depth {}", circuit.non_free_depth());
        println!("gate counts {:?}", circuit.count_gates_by_type_vec());
    }
}
