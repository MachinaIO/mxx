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
//!
//! With a LUT radix `B` ([`RnsBggDynamicSetup::with_lut_radix`]), a branch
//! `k = g^e` is switched once per nonzero radix-`B` digit of `e`: the branch
//! `parent(k)` without the top digit is switched by `g^(a_top B^top)`, so the
//! branches form a tree of depth at most `log_B |H|` whose `|H| - 1` switches
//! share their intermediate results ([`radix_plan`]). Each switch acts on an
//! encoding whose payload is a monomial, so the payload term of `C_k` keeps
//! its `l1` norm one; the scalar actions, whose coefficients carry `W^-1`,
//! are applied last, per branch. A LUT's `branches` public action holds the
//! product keys `T` depth by depth, in the order of [`RadixDepth::branches`].
//!
//! With compact encodings ([`RnsBggDynamicSetup::with_compact_encodings`]),
//! the fresh encodings are stored in two columns over `pq` and expanded where
//! they are read ([`expand_compact_row`]); their keys are the expansions of
//! `2 x 2` keys over `R_pq` ([`expand_compact_matrix`]).
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

/// The branches of one depth of a LUT's radix plan (see the module
/// documentation).
pub(crate) struct RadixDepth {
    /// The branches switched at this depth: the parents of the next depth
    /// first, then the others, each ascending.
    pub(crate) branches: Vec<usize>,
    /// The automorphism switching each branch from its parent.
    pub(crate) switches: Vec<usize>,
    /// The position of each branch's parent among the previous depth's
    /// branches; zero at the first depth, whose parent is the input.
    pub(crate) parents: Vec<usize>,
    /// The number of leading branches that are parents of the next depth.
    pub(crate) internal: usize,
}

/// The radix plan of the nonidentity branches `switched` of a LUT, depth by
/// depth. Every parent of a branch is itself a branch (or the identity): the
/// branches of a LUT form a subgroup of the cyclic `H`, which removing a top
/// radix digit keeps.
pub(crate) fn radix_plan(
    setup: &RnsBggDynamicSetup,
    switched: &[usize],
) -> Result<Vec<RadixDepth>, RnsBggError> {
    let mut steps = BTreeMap::new();
    for k in switched {
        steps.insert(*k, setup.radix_step(*k)?);
    }
    let mut depths = BTreeMap::<usize, usize>::new();
    let depth_of = |k: usize| -> Result<usize, RnsBggError> {
        let mut depth = 0;
        let mut current = k;
        while current != 1 {
            current = steps
                .get(&current)
                .ok_or(RnsBggError::Parameters("a radix parent is not a LUT branch"))?
                .0;
            depth += 1;
        }
        Ok(depth)
    };
    for k in switched {
        depths.insert(*k, depth_of(*k)?);
    }
    let parents =
        steps.values().map(|(parent, _)| *parent).collect::<std::collections::BTreeSet<_>>();
    let max_depth = depths.values().copied().max().unwrap_or(0);
    let mut plan: Vec<RadixDepth> = Vec::with_capacity(max_depth);
    for depth in 1..=max_depth {
        let (internal, leaves): (Vec<usize>, Vec<usize>) = depths
            .iter()
            .filter(|(_, d)| **d == depth)
            .map(|(k, _)| *k)
            .partition(|k| parents.contains(k));
        let branches = internal.iter().chain(&leaves).copied().collect::<Vec<_>>();
        let previous = plan.last().map(|depth| depth.branches.as_slice()).unwrap_or(&[]);
        let parent_positions = branches
            .iter()
            .map(|k| match steps[k].0 {
                1 => Ok(0),
                parent => previous
                    .iter()
                    .position(|candidate| *candidate == parent)
                    .ok_or(RnsBggError::Parameters("a radix parent is not a LUT branch")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        plan.push(RadixDepth {
            switches: branches.iter().map(|k| steps[k].1).collect(),
            parents: parent_positions,
            internal: internal.len(),
            branches,
        });
    }
    Ok(plan)
}

/// `ModSwitch_(pq -> q)(entry G~)` at `level`: the `1 x 1` `entry` over the
/// compact ring times the lift `G~` of the `1 x D` `gadget` (its centered
/// integers re-encoded in `R_pq`), rounded to the level ring. For a compact
/// encoding `c = s (A - p X I_2) + e`, the expansion of each component is the
/// component of `s (A' - X G) + e'` (see [`compact_expansion_bound`]).
pub(crate) fn expand_compact_entry(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    level: usize,
    gadget: &Mat,
    entry: Mat,
) -> Result<Mat, RnsBggError> {
    let ring = setup
        .compact_ring(parameters, level)?
        .ok_or(RnsBggError::Parameters("the setup's encodings are not compact"))?;
    Ok((entry * gadget.clone().centered_rebase(&ring)).modulus_switch(&parameters.ring(level)?))
}

/// The expansion of a compact row (two `1 x 1` entries over `R_pq`) to a row of
/// two `1 x D` blocks at `level` under the public gadget.
pub(crate) fn expand_compact_row(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    level: usize,
    row: RnsBggRow,
) -> Result<RnsBggRow, RnsBggError> {
    expand_compact_row_with(parameters, setup, level, &parameters.public_gadget(level)?, row)
}

/// [`expand_compact_row`] under `gadget` (the state gadget for a private input).
pub(crate) fn expand_compact_row_with(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    level: usize,
    gadget: &Mat,
    row: RnsBggRow,
) -> Result<RnsBggRow, RnsBggError> {
    Ok((
        expand_compact_entry(parameters, setup, level, gadget, row.0)?,
        expand_compact_entry(parameters, setup, level, gadget, row.1)?,
    ))
}

/// A stored compact companion or helper row (two `1 x 1` entries over `pq`)
/// expanded at the setup ring under the public gadget, as every reader of the
/// setup's packages expands it.
pub fn expand_stored_row(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    row: RnsBggRow,
) -> Result<RnsBggRow, RnsBggError> {
    expand_compact_row(parameters, setup, 0, row)
}

/// The expansion `A' = ModSwitch_(pq -> q)(A~ G~)` of a compact `2 x 2` key
/// to a `2 x 2` block key of `1 x D` blocks at `level` under `gadget`.
pub(crate) fn expand_compact_matrix(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    level: usize,
    gadget: &Mat,
    matrix: RnsBggMatrix,
) -> Result<RnsBggMatrix, RnsBggError> {
    Ok((
        expand_compact_row_with(parameters, setup, level, gadget, matrix.0)?,
        expand_compact_row_with(parameters, setup, level, gadget, matrix.1)?,
    ))
}

/// The compact encoding `s A~ - p X + e` over the compact ring `ring`: the
/// secret row `secret` and the payload row `payload` (both already over
/// `R_pq`), the `2 x 2` key `matrix`, and a fresh `1 x 2` error under
/// `contract`. For a scalar payload `x`, `X = x s` gives `s (A~ - p x I_2) + e`.
pub(crate) fn compact_encoding_row(
    ring: &Ring,
    matrix: RnsBggMatrix,
    secret: &RnsBggRow,
    payload: RnsBggRow,
    p: &num_bigint::BigUint,
    contract: &super::RnsBggGaussian,
) -> Result<RnsBggRow, RnsBggError> {
    let scale = ring.polynomial([num_bigint::BigInt::from(p.clone()).into()]);
    let error = contract.sample_row(ring, 1)?;
    Ok((
        secret.0.clone() * matrix.0.0 + secret.1.clone() * matrix.1.0 - scale.clone() * payload.0 +
            error.0,
        secret.0.clone() * matrix.0.1 + secret.1.clone() * matrix.1.1 - scale * payload.1 + error.1,
    ))
}

/// The coefficient bound of an expanded compact encoding's error, for a fresh
/// error of coefficient bound `cutoff`:
/// `ceil(cutoff q / p) + ceil((n + 2) / 2)`. The error `e G~ / p` has
/// coefficients at most `cutoff max |g~| / p <= cutoff q / p`; the rounding of
/// the key `s (A~ G~ / p - A')` at most `(||s_bar||_1 + 1) / 2 <= (n + 1) / 2`
/// for the ternary secret; and the rounding of the switch at most `1/2`.
pub fn compact_expansion_bound(
    parameters: &RnsBggParameters,
    extension: &[u64],
    cutoff: &num_bigint::BigUint,
) -> Result<num_bigint::BigUint, RnsBggError> {
    let level = parameters.level(0)?;
    let q = level.modulus().as_ref().clone();
    let p = extension
        .iter()
        .map(|prime| num_bigint::BigUint::from(*prime))
        .product::<num_bigint::BigUint>();
    if p <= num_bigint::BigUint::from(1u8) {
        return Err(RnsBggError::Parameters("empty compact extension"));
    }
    let n = num_bigint::BigUint::from(level.ring_dimension());
    Ok((cutoff * &q + &p - 1u8) / &p + (n + 3u8) / 2u8)
}

/// The product `p` of the compact extension of `setup`, if any.
pub(crate) fn compact_modulus(setup: &RnsBggDynamicSetup) -> Option<num_bigint::BigUint> {
    setup
        .compact_extension()
        .map(|extension| extension.iter().map(|prime| num_bigint::BigUint::from(*prime)).product())
}
