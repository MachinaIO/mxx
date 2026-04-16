use super::*;
use crate::{
    matrix::PolyMatrix,
    utils::{pow_biguint_usize, round_div},
};
use num_traits::One;

/// Lower bound on the CRT product required to tolerate the configured unreduced multiplication
/// budget.
///
/// `sample_crt_primes` uses exactly this bound when choosing the synthetic p-moduli for a
/// nested-RNS context, so exposing it here keeps the budget calculation shared between setup code
/// and tests.
pub(crate) fn sample_crt_primes_mul_budget_bound(
    sum_p_moduli: u64,
    modulus_count: usize,
    q_max: u64,
) -> BigUint {
    let modulus_count =
        u64::try_from(modulus_count).expect("p_moduli length must fit in u64 for bound tracking");
    BigUint::from(sum_p_moduli + modulus_count) * BigUint::from(q_max) / BigUint::from(2u64)
}

/// Euclidean gcd helper used while enforcing pairwise-coprime synthetic p-moduli.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Return the first pairwise-coprime CRT moduli that are large enough to sustain the configured
/// multiplication budget.
///
/// The selection is deterministic. `NestedRnsPolyContext::setup` relies on that determinism so
/// identical parameter sets always produce the same helper LUT layout and subcircuit bindings.
pub(crate) fn sample_crt_primes(
    max_bit_width: usize,
    q_max: u64,
    max_unreduced_muls: usize,
) -> Vec<u64> {
    let lower = 3u64;
    let upper = 1u64 << max_bit_width;
    let mut results: Vec<u64> = Vec::new();
    let mut sum = 0u64;
    let mut prod = BigUint::one();
    let mut prod_reached = false;

    for candidate in lower..upper {
        if results.iter().all(|&chosen| gcd_u64(candidate, chosen) == 1) {
            results.push(candidate);
            sum += candidate;
            prod *= BigUint::from(candidate);
        }
        let mul_budget_bound = sample_crt_primes_mul_budget_bound(sum, results.len(), q_max);
        if pow_biguint_usize(&mul_budget_bound, max_unreduced_muls) < prod {
            prod_reached = true;
            break;
        }
    }

    if !prod_reached {
        panic!(
            "failed to find enough pairwise coprime integers with bit width {max_bit_width} to satisfy q_max {q_max} and max_unreduced_muls {max_unreduced_muls}"
        );
    }

    results
}

/// Resolve the q-window and synthetic p-moduli used by the pure encoding helpers.
///
/// This mirrors the context setup path without requiring a prebuilt `NestedRnsPolyContext`, which
/// is why the standalone encoding APIs can reconstruct the exact same layout as the circuit
/// helpers.
fn resolve_nested_rns_encoding_layout<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    q_level: Option<usize>,
) -> (Vec<u64>, usize, Vec<u64>) {
    let (q_moduli, _, q_moduli_depth) = params.to_crt();
    let active_q_level = q_level.unwrap_or(q_moduli_depth);
    let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
    let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max, max_unreduced_muls);
    (q_moduli, active_q_level, p_moduli)
}

/// Parallel map helper for producing `output_count` nested-RNS outputs with access to `params`.
///
/// Encoding helpers call this instead of open-coding the CPU/GPU split each time. The behavior is
/// intentionally identical across backends; only the scheduling policy differs.
pub(crate) fn map_nested_rns_outputs_with_params<P, T, F>(
    params: &P::Params,
    output_count: usize,
    f: F,
) -> Vec<T>
where
    P: Poly,
    T: Send,
    F: Fn(usize, &P::Params) -> T + Send + Sync,
{
    if output_count == 0 {
        return Vec::new();
    }

    #[cfg(feature = "gpu")]
    {
        return gpu::map_nested_rns_outputs_with_params_gpu::<P, T, F>(params, output_count, f);
    }

    #[cfg(not(feature = "gpu"))]
    {
        (0..output_count).into_par_iter().map(|idx| f(idx, params)).collect()
    }
}

/// Backend-aware parallel map helper for plain values.
pub(crate) fn map_nested_rns_values<T, F>(count: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if count == 0 {
        return Vec::new();
    }

    #[cfg(feature = "gpu")]
    {
        return (0..count).map(f).collect();
    }

    #[cfg(not(feature = "gpu"))]
    {
        (0..count).into_par_iter().map(f).collect()
    }
}

/// Resolve the active q-window for encoding- and gadget-related helpers.
///
/// These callers all operate on the same conceptual slice: `level_offset` chooses the first
/// active q-level, and `enable_levels` chooses how many consecutive q-levels participate. Central
///izing the validation here guarantees every helper encodes the same window shape.
pub(crate) fn resolve_nested_rns_active_window(
    ctx: &NestedRnsPolyContext,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> (usize, Vec<u64>) {
    let level_offset = level_offset.unwrap_or(0);
    let active_levels = enable_levels.unwrap_or_else(|| {
        ctx.q_moduli_depth
            .checked_sub(level_offset)
            .expect("level_offset must not exceed q_moduli_depth")
    });
    let active_q_moduli =
        ctx.q_moduli.iter().skip(level_offset).take(active_levels).copied().collect::<Vec<_>>();
    (level_offset, active_q_moduli)
}

/// Compute the CRT reconstruction coefficients for the currently active q-level window.
///
/// Matrix/gadget encoders use these coefficients to rebuild one sparse q-level into the ambient
/// modulus exactly the same way `NestedRnsPoly::reconstruct` does at the circuit level.
fn nested_rns_level_reconstruction_coeffs(active_q_moduli: &[u64]) -> Vec<BigUint> {
    let active_modulus =
        active_q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    active_q_moduli
        .iter()
        .map(|&q_i| {
            let q_i_big = BigUint::from(q_i);
            let q_over_qi = &active_modulus / &q_i_big;
            let q_over_qi_mod =
                (&q_over_qi % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let inv = mod_inverse(q_over_qi_mod, q_i).expect("CRT moduli must be coprime");
            (&q_over_qi * BigUint::from(inv)) % &active_modulus
        })
        .collect::<Vec<_>>()
}

/// Subtract `rhs` from `lhs` modulo `modulus` while staying in canonical nonnegative residues.
///
/// The encoding helpers mirror the circuit's nonnegative-residue convention, so they cannot rely on
/// signed intermediate values here.
fn nested_rns_sub_mod(lhs: BigUint, rhs: &BigUint, modulus: &BigUint) -> BigUint {
    let rhs = rhs % modulus;
    if lhs >= rhs { lhs - rhs } else { lhs + modulus - rhs }
}

/// Decode the decomposition helper subcircuit in plain integer space.
///
/// This is used only by CPU-side matrix/gadget encoders. It mirrors the circuit path:
/// each `x_i` is converted into `y_i = x_i * (p/p_i)^(-1) mod p_i`, then the rounded sum
/// `w = round(sum_i round(y_i * scale / p_i) / scale)` is produced.
pub(crate) fn nested_rns_decomposition_terms_from_row(
    ctx: &NestedRnsPolyContext,
    row: &[u64],
) -> (Vec<BigUint>, BigUint) {
    let mut ys = Vec::with_capacity(ctx.p_moduli.len());
    let mut real_sum = 0u64;
    for (p_idx, &p_i) in ctx.p_moduli.iter().enumerate() {
        let x_i = row[p_idx] % p_i;
        let p_over_pi_mod_pi = (&ctx.p_over_pis[p_idx] % BigUint::from(p_i))
            .to_u64()
            .expect("CRT residue must fit in u64");
        let p_over_pi_inv = mod_inverse(p_over_pi_mod_pi, p_i).expect("CRT moduli must be coprime");
        let y_i = ((x_i as u128 * p_over_pi_inv as u128) % p_i as u128) as u64;
        real_sum += round_div(y_i * ctx.scale, p_i);
        ys.push(BigUint::from(y_i));
    }
    let w = BigUint::from(round_div(real_sum, ctx.scale));
    (ys, w)
}

/// Re-encode one sparse q-level row back into the ambient ring modulus.
///
/// The gadget/vector helpers build values in p-space first because the decomposition logic is
/// defined there. This helper performs the exact same reconstruction that
/// `NestedRnsPoly::reconstruct` would do for a single active q-level, but entirely on plain
/// integers so matrix generators can produce expected constants without building a circuit first.
pub(crate) fn nested_rns_sparse_level_slot_value<P: Poly>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    reconst_coeff: &BigUint,
    row: &[u64],
) -> BigUint {
    let modulus: std::sync::Arc<BigUint> = params.modulus().into();
    let modulus = modulus.as_ref();
    let (ys, w) = nested_rns_decomposition_terms_from_row(ctx, row);
    let sum_without_reduce = ys
        .iter()
        .zip(ctx.p_over_pis.iter())
        .fold(BigUint::ZERO, |acc, (y_i, p_hat_i)| (acc + y_i * p_hat_i) % modulus);
    let pv = (&w * &ctx.p_full) % modulus;
    let sum_q_k = nested_rns_sub_mod(sum_without_reduce, &pv, modulus);
    (sum_q_k * reconst_coeff) % modulus
}

pub fn nested_rns_gadget_vector<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> M
where
    P: Poly,
    M: PolyMatrix<P = P>,
{
    let (level_offset, active_q_moduli) =
        resolve_nested_rns_active_window(ctx, enable_levels, level_offset);
    let chunk_width = ctx.p_moduli.len() + 1;
    let reconst_coeffs = nested_rns_level_reconstruction_coeffs(&active_q_moduli);
    let coeff_count = params.ring_dimension() as usize;
    let mut gadget_row = Vec::with_capacity(active_q_moduli.len() * chunk_width);
    for (q_idx, level_values) in
        ctx.gadget_values[level_offset..level_offset + active_q_moduli.len()].iter().enumerate()
    {
        for residue in level_values {
            let row = ctx
                .p_moduli
                .iter()
                .map(|&p_i| (residue % BigUint::from(p_i)).to_u64().expect("row residue must fit"))
                .collect::<Vec<_>>();
            let coeff =
                nested_rns_sparse_level_slot_value::<P>(params, ctx, &reconst_coeffs[q_idx], &row);
            gadget_row.push(P::from_biguints(params, &vec![coeff; coeff_count]));
        }
    }
    M::from_poly_vec_row(params, gadget_row)
}

pub fn nested_rns_gadget_decomposed<P, M>(
    params: &P::Params,
    ctx: &NestedRnsPolyContext,
    value: &M,
    enable_levels: Option<usize>,
    level_offset: Option<usize>,
) -> M
where
    P: Poly,
    M: PolyMatrix<P = P>,
{
    let (_, active_q_moduli) = resolve_nested_rns_active_window(ctx, enable_levels, level_offset);
    let chunk_width = ctx.p_moduli.len() + 1;
    let gadget_len = active_q_moduli.len() * chunk_width;
    let (row_size, col_size) = value.size();
    let reconst_coeffs = nested_rns_level_reconstruction_coeffs(&active_q_moduli);

    let entry_polys = map_nested_rns_values(row_size * col_size, |entry_idx| {
        let row_idx = entry_idx / col_size;
        let col_idx = entry_idx % col_size;
        let coeffs = value.entry(row_idx, col_idx).coeffs_biguints();
        let mut output_coeffs = vec![vec![BigUint::ZERO; coeffs.len()]; gadget_len];
        for (q_idx, &q_i) in active_q_moduli.iter().enumerate() {
            let q_i_big = BigUint::from(q_i);
            for (coeff_idx, coeff) in coeffs.iter().enumerate() {
                let input_residue =
                    (coeff % &q_i_big).to_u64().expect("q-level residue must fit in u64");
                let input_row =
                    ctx.p_moduli.iter().map(|&p_i| input_residue % p_i).collect::<Vec<_>>();
                let (ys, w) = nested_rns_decomposition_terms_from_row(ctx, &input_row);
                for digit_idx in 0..chunk_width {
                    let scalar = if digit_idx < ctx.p_moduli.len() {
                        ys[digit_idx].to_u64().expect("decomposition digit must fit in u64")
                    } else {
                        w.to_u64().expect("rounding digit must fit in u64")
                    };
                    let encoded_row =
                        ctx.p_moduli.iter().map(|&p_i| scalar % p_i).collect::<Vec<_>>();
                    let target_row = q_idx * chunk_width + digit_idx;
                    output_coeffs[target_row][coeff_idx] = nested_rns_sparse_level_slot_value::<P>(
                        params,
                        ctx,
                        &reconst_coeffs[q_idx],
                        &encoded_row,
                    );
                }
            }
        }
        let output_polys = output_coeffs
            .into_iter()
            .map(|coeffs| P::from_biguints(params, &coeffs))
            .collect::<Vec<_>>();
        (row_idx, col_idx, output_polys)
    });

    let mut decomposed = (0..row_size * gadget_len)
        .map(|_| vec![P::const_zero(params); col_size])
        .collect::<Vec<_>>();
    for (row_idx, col_idx, output_polys) in entry_polys {
        for (target_row, poly) in output_polys.into_iter().enumerate() {
            decomposed[row_idx * gadget_len + target_row][col_idx] = poly;
        }
    }

    M::from_poly_vec(params, decomposed)
}

pub fn encode_nested_rns_poly_compact_bytes<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level: Option<usize>,
) -> Vec<Vec<u8>> {
    encode_nested_rns_poly_compact_bytes_with_offset::<P>(
        p_moduli_bits,
        max_unreduced_muls,
        params,
        input,
        0,
        q_level,
    )
}

pub fn encode_nested_rns_poly_compact_bytes_with_offset<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level_offset: usize,
    q_level: Option<usize>,
) -> Vec<Vec<u8>> {
    let (q_moduli, active_q_level, p_moduli) =
        resolve_nested_rns_encoding_layout::<P>(p_moduli_bits, max_unreduced_muls, params, q_level);
    let input_mod_q = q_moduli
        .iter()
        .skip(q_level_offset)
        .take(active_q_level)
        .map(|&q_i| input % BigUint::from(q_i))
        .collect::<Vec<_>>();
    let p_moduli_depth = p_moduli.len();
    let output_count = active_q_level * p_moduli_depth;

    map_nested_rns_outputs_with_params::<P, _, _>(params, output_count, |idx, local_params| {
        let q_idx = idx / p_moduli_depth;
        let p_idx = idx % p_moduli_depth;
        let residue = &input_mod_q[q_idx] % p_moduli[p_idx];
        P::from_biguint_to_constant(local_params, residue).to_compact_bytes()
    })
}

pub fn encode_nested_rns_poly<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level: Option<usize>,
) -> Vec<P> {
    encode_nested_rns_poly_with_offset::<P>(
        p_moduli_bits,
        max_unreduced_muls,
        params,
        input,
        0,
        q_level,
    )
}

pub fn encode_nested_rns_poly_with_offset<P: Poly>(
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    params: &P::Params,
    input: &BigUint,
    q_level_offset: usize,
    q_level: Option<usize>,
) -> Vec<P> {
    let (q_moduli, active_q_level, p_moduli) =
        resolve_nested_rns_encoding_layout::<P>(p_moduli_bits, max_unreduced_muls, params, q_level);
    let p_moduli_depth = p_moduli.len();
    let mut polys = vec![Vec::with_capacity(p_moduli_depth); active_q_level];
    for (q_idx, &q_i) in q_moduli.iter().skip(q_level_offset).take(active_q_level).enumerate() {
        let input_qi = input % BigUint::from(q_i);
        for &p_i in &p_moduli {
            polys[q_idx].push(P::from_biguint_to_constant(params, &input_qi % p_i));
        }
    }
    polys.into_iter().flatten().collect::<Vec<_>>()
}
