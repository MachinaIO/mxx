//! Dynamic evaluation primitives shared by the public-key and encoding
//! lowerings, so both build the same public actions.
//!
//! A Dynamic LUT `L` on an encoding `c` of `X^u` is
//! `sum_{k in H_L} SMul_{D_{k,L}}(Aut_k(c))`, an encoding of `X^{L(u)}` (or
//! `L(u)` for a coefficient table) whose key is computed the same way from
//! the input key:
//!
//! - `Aut_1` is the identity. For `k != 1`, `Aut_k` first routes the encoding and its key through
//!   `sigma_k`, so the payload and the mask move to `s_k = (sigma_k(s_bar), 1)`. The product with
//!   the switching package of `k` (the gadget ciphertext `C_k` with `s_k C_k = s G -
//!   sigma_k(e_{h,k^-1})`, see [`automorphism_ciphertext`]) switches the payload back to `s`: `r_k
//!   = s_k T_k - sigma_k(mu) s G + e_r`. The helper `h_k = s H_k - s_k G + e_h` then switches the
//!   mask: `Aut_k(c) = h_k Dec_G(T_k) + r_k` under the key `A_k = H_k Dec_G(T_k)`.
//! - `SMul_D(c) = c V_D` with `V_D = Dec_G(D G)`, applied to the key as `A V_D`. With `G = I_2 (x)
//!   g`, `V_D` is block diagonal with the `D x D` block `Dec_g(D g)` ([`scalar_action`]); its error
//!   gain does not grow with `D`, unlike the product `D c`.
//!
//! `D_{k,L} = W^-1 sum_j X^{L(j)} sigma_k(X^-j)` (or `L(j)` in place of
//! `X^{L(j)}`) is [`super::EncodingLowering::lut_coefficient`], so the table
//! values enter only these public actions, never encryption.
use super::*;

/// `Select(selector, values)`: the value at position `selector`.
pub(crate) fn select(values: &[usize], selector: IntExpr) -> IntExpr {
    IntExpr::Select {
        selector: Box::new(selector),
        branches: values.iter().map(|value| IntExpr::constant(*value)).collect(),
    }
}

/// The automorphisms of `table` at `level`: its input layout's indices in
/// ascending order (the identity first), each in the setup group `group`
/// (ascending). A Dynamic LUT needs its values, which encryption never
/// reads.
pub(crate) fn lut_automorphisms(
    parameters: &RnsBggParameters,
    group: &[usize],
    level: usize,
    table: &crate::program::LutTable,
) -> Result<Vec<usize>, RnsBggError> {
    parameters.dynamic_route(level)?;
    if table.values().is_none() {
        return Err(RnsBggError::Binding("a Dynamic LUT needs its table values"));
    }
    let indices = table
        .input_layout()
        .automorphism_indices(parameters.level(level)?.ring_dimension() as usize)
        .map_err(|_| RnsBggError::Lut)?;
    if indices.first() != Some(&1) || indices.iter().any(|k| group.binary_search(k).is_err()) {
        return Err(RnsBggError::Binding("a Dynamic LUT uses an automorphism outside H"));
    }
    Ok(indices)
}

/// The switching ciphertext `C_k = sigma_k(H_{k^-1}) - [0; sigma_k(h_{k^-1})]`
/// of automorphism `k`, from the public `H_{k^-1}` and the helper
/// `h_{k^-1} = s H_{k^-1} - s_{k^-1} G + e`. Since the second entry of
/// `s_k` is one, `s_k C_k = sigma_k(s H_{k^-1}) - sigma_k(h_{k^-1}) = s G -
/// sigma_k(e)`. It is computed by every reader; no stage stores it.
pub(crate) fn automorphism_ciphertext(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    hash_key: mxx_dsl::Bytes,
    k: IntExpr,
    inverse: IntExpr,
    inverse_helper: RnsBggRow,
) -> Result<RnsBggMatrix, RnsBggError> {
    let routed = map_matrix(setup.helper_key(parameters, hash_key, inverse)?, |entry| {
        entry.ring_automorphism(k.clone())
    });
    let helper = map_row(inverse_helper, |entry| entry.ring_automorphism(k.clone()));
    Ok((routed.0, (routed.1.0 - helper.0, routed.1.1 - helper.1)))
}

/// The `D x D` block `Dec_g(D g)` of the public scalar action
/// `V_D = Dec_G(D G)` at `level`, for a public coefficient `D` in its ring.
pub(crate) fn scalar_action(
    parameters: &RnsBggParameters,
    level: usize,
    coefficient: Mat,
) -> Result<Preimage, RnsBggError> {
    let current = parameters.level(level)?;
    let base = 1u64
        .checked_shl(current.base_bits())
        .ok_or(RnsBggError::Parameters("scalar action base overflow"))?;
    Ok((coefficient * parameters.public_gadget(level)?).decompose(base, current.modulus_digits()))
}

/// `row V_D` for a two-block row: each block times the `D x D` block.
pub(crate) fn scalar_action_row(row: RnsBggRow, action: &Preimage) -> RnsBggRow {
    map_row(row, |entry| entry.mul_small_rhs(action.clone()))
}

/// `A V_D` for a `2 x 2` block key.
pub(crate) fn scalar_action_matrix(matrix: RnsBggMatrix, action: &Preimage) -> RnsBggMatrix {
    map_matrix(matrix, |entry| entry.mul_small_rhs(action.clone()))
}
