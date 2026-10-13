//! LUTs on public payloads, shared by the public-key and encoding lowerings
//! of both RHS modes.
//!
//! A public encoding `c_x = s A_x - x s g + e` (one secret for the payload
//! and the mask) cannot use a LUT switching ciphertext, which is under the
//! payload secret of private encodings. Its branch `k` is instead switched
//! by the LUT helpers alone, which every encryption already publishes:
//!
//! - Route: `sigma_k(c_x)` is under `sigma_k(s)` for both the mask and the payload `sigma_k(x)`.
//! - Payload switch ([`public_lut_switch`]): the helper of `k^-1`, routed by `sigma_k`, is the
//!   mixed encoding `c_{k,1} = sigma_k(s) A_{k,1} - beta s gt + e'` of one, with mask `sigma_k(s)`
//!   and payload secret `s`. With the scalar action `V` of the branch coefficient `D_k` (`beta gt V
//!   = beta D_k gt`) and `M = A_{k,1} V`, the public product `sigma_k(c_x) G^-1(M) + sigma_k(x)
//!   (c_{k,1} V)` is `sigma_k(s) K - beta D_k sigma_k(x) s gt + err` with `K = sigma_k(A_x)
//!   G^-1(M)`: the `sigma_k(s) sigma_k(x) M` terms cancel, the input error is decomposed once, and
//!   the helper error is multiplied only by the signed monomial `sigma_k(x)` and the bounded digits
//!   `V`, never by `D_k`.
//! - Mask switch ([`public_lut_mask_switch`]): the helper `h_k = s A'_k - beta sigma_k(s) gt + e_h`
//!   of `k`, times `Dec(K)`, removes `sigma_k(s) K`, as for a private LUT.
//!
//! Static helpers (`h_sigma = s A_sigma - p_l sigma(s) gt_(l+1) + e`) carry the
//! level's scale `beta = p_l` and the next level's gadget `gt_(l+1)`; their
//! mask switch leaves `sigma_k(s) (K - p_l ModDown(K))`, of size `p_l`, which
//! the one ModDown of the branch sum to `l + 1` removes. That ModDown maps
//! `p_l gt_(l+1)` to `public_gadget(l + 1)`, so the result is a public
//! encoding of the LUT value at `l + 1` in either output representation.
//! Dynamic helpers have `beta = 1` and the public gadget, so no ModDown is
//! needed (see [`crate::rns_bgg::dynamic`]).
//!
//! A Static layout of stride `s > 0` has an automorphism set `S` that is not
//! closed under inverses modulo `2n` (only modulo `2n / 2^s`), so the helper
//! of `k^-1` may not exist. The bridge `c_{k,1} V_k` is then built from
//! helpers of `S` alone ([`static_public_lut`]): for a shortest word `k^-1 =
//! j_1 ... j_h` over `S` ([`inverse_words`]) and `r_i = (j_1 ... j_i)^-1`,
//! `C_1 = sigma_(r_1)(h_(j_1)) V_k` has mask `sigma_(r_1)(s)` and payload
//! `D_k` under `s`, and each further helper `sigma_(r_i)(h_(j_i))` is a mask
//! switch from `sigma_(r_(i-1))(s)` to `sigma_(r_i)(s)` ([`public_lut_mask_switch`]),
//! ending at `sigma_k(s)`. Only public keys are decomposed, so every hop adds
//! its helper error and mask remainder once, unamplified; `V_k` acts on the
//! first helper only. A word of length one is the direct switch.
use super::*;
use mxx_dsl::{DslError, Int};
use std::collections::BTreeMap;

/// For every `k` of `automorphisms` (a set of automorphism indices of ring
/// dimension `n` containing the identity), a shortest word `j_1 ... j_h` of
/// positions in `automorphisms` whose indices multiply to `k^-1` modulo
/// `2n`; the identity's word is empty. The words exist because each `k` has
/// finite order; a breadth-first search over products of the nonidentity
/// indices finds the shortest.
pub(crate) fn inverse_words(
    automorphisms: &[usize],
    ring_dimension: usize,
) -> Result<Vec<Vec<usize>>, RnsBggError> {
    let modulus = 2 * ring_dimension;
    let mut words = BTreeMap::from([(1usize, Vec::new())]);
    let mut frontier = vec![1usize];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for element in frontier {
            for (position, j) in automorphisms.iter().enumerate().filter(|(_, j)| **j != 1) {
                let product = (element as u128 * *j as u128 % modulus as u128) as usize;
                if !words.contains_key(&product) {
                    let mut word = words[&element].clone();
                    word.push(position);
                    words.insert(product, word);
                    next.push(product);
                }
            }
        }
        frontier = next;
    }
    automorphisms
        .iter()
        .map(|k| {
            let inverse =
                super::setup::automorphism_inverse(*k, ring_dimension).ok_or(RnsBggError::Lut)?;
            words.get(&inverse).cloned().ok_or(RnsBggError::Lut)
        })
        .collect()
}

/// A loop-carried sum held as zero or one value (an optional row or key),
/// plus a term present exactly when the sum is.
fn add_optional<T>(
    sum: Vec<T>,
    term: Option<T>,
    add: impl Fn(T, T) -> T,
) -> Result<Vec<T>, DslError> {
    match (sum.into_iter().next(), term) {
        (Some(sum), Some(term)) => Ok(vec![add(sum, term)]),
        (None, None) => Ok(Vec::new()),
        _ => Err(DslError::Schema),
    }
}

/// `key Dec`, for an action given by its stacked column decompositions.
fn key_by_columns<D: RightAction>(key: RnsBggMatrix, columns: (D, D)) -> RnsBggMatrix {
    let mut rows = multiply_rows_by_columns(vec![key.0, key.1], columns).into_iter();
    let top = rows.next().expect("two rows");
    (top, rows.next().expect("two rows"))
}

/// The payload switch of one branch with the coefficient action `action`
/// fused (none: the identity action): `routed` is `sigma_k(c_x)`'s row (if
/// any) and key, `payload` the routed public payload `sigma_k(x)` (needed
/// with a row), and `helper` the routed helper of `k^-1`, its row (needed
/// with a row) and key `A_{k,1}`. Returns the switched row and key `K` at
/// `level`, under the mask `sigma_k(s)` and the payload secret `s`.
pub(crate) fn public_lut_switch<D: RightAction>(
    parameters: &RnsBggParameters,
    level: usize,
    routed: (Option<RnsBggRow>, RnsBggMatrix),
    payload: Option<Mat>,
    helper: (Option<RnsBggRow>, RnsBggMatrix),
    action: Option<&D>,
) -> Result<(Option<RnsBggRow>, RnsBggMatrix), RnsBggError> {
    let (helper_row, helper_key) = helper;
    let (helper_row, target) = match action {
        Some(action) => (
            helper_row.map(|row| super::dynamic::scalar_action_row(row, action)),
            super::dynamic::scalar_action_matrix(helper_key, action),
        ),
        None => (helper_row, helper_key),
    };
    let columns = compact_decompose_columns(parameters, level, level, target)?;
    let (row, key) = routed;
    let Some(row) = row else {
        return Ok((None, key_by_columns(key, columns)));
    };
    let (row, key) = multiply_row_and_public_by_columns(row, key, columns);
    let helper_row =
        helper_row.ok_or(RnsBggError::Binding("a public LUT row switch needs its helper row"))?;
    let payload =
        payload.ok_or(RnsBggError::Binding("a public LUT row switch needs its payload"))?;
    let helper_term = (helper_row.0 * payload.clone(), helper_row.1 * payload);
    Ok((Some(add_encoding_rows(row, helper_term)), key))
}

/// The mask switch of one branch: the switched row (if any) and key `K` at
/// `level`, the helper `h_k` of the branch (its row, needed with a row) and
/// its public matrix (if the output key is wanted). `K` is decomposed at
/// `decomposition_level` (Static: `level + 1` after a ModDown of `K`, as the
/// private LUT does; Dynamic: `level`). Returns the row and key under `s`.
pub(crate) fn public_lut_mask_switch(
    parameters: &RnsBggParameters,
    level: usize,
    decomposition_level: usize,
    switched: (Option<RnsBggRow>, RnsBggMatrix),
    helper: (Option<RnsBggRow>, Option<RnsBggMatrix>),
) -> Result<(Option<RnsBggRow>, Option<RnsBggMatrix>), RnsBggError> {
    let (row, key) = switched;
    let key = if decomposition_level == level {
        key
    } else {
        mod_down_public(key, &parameters.ring(decomposition_level)?)
    };
    let columns = compact_decompose_columns(parameters, level, decomposition_level, key)?;
    let (helper_row, helper_key) = helper;
    let helper_row = match row {
        Some(_) => Some(
            helper_row
                .ok_or(RnsBggError::Binding("a public LUT mask switch needs its helper row"))?,
        ),
        None => None,
    };
    let mut rows = helper_row.into_iter().collect::<Vec<_>>();
    if let Some((top, bottom)) = helper_key.clone() {
        rows.extend([top, bottom]);
    }
    if rows.is_empty() {
        return Ok((None, None));
    }
    let mut products = multiply_rows_by_columns(rows, columns).into_iter();
    let row = row.map(|row| add_encoding_rows(row, products.next().expect("helper row")));
    let key = helper_key.map(|_| {
        let top = products.next().expect("two rows");
        (top, products.next().expect("two rows"))
    });
    Ok((row, key))
}

/// The Static scalar action `V = Dec_(l+1)(D g_(l+1))` of a branch
/// coefficient `D` in the ring of `level + 1`, centered-rebased to `level`:
/// `p_l gt_(l+1) V = p_l D gt_(l+1)` modulo `q_l`.
pub(crate) fn static_public_lut_action(
    parameters: &RnsBggParameters,
    level: usize,
    coefficient: Mat,
) -> Result<SmallMatrix, RnsBggError> {
    let next_level = level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
    parameters.gadget_inverse(
        next_level,
        level,
        coefficient * parameters.public_gadget(next_level)?,
    )
}

/// The Static identity branch of a public LUT: `c_x G_pub^-1(T)` for the
/// block-diagonal target `T = I_2 (x) (p_l gt_(l+1) V_1)`, already under `s`,
/// so neither helper is needed. `input` is the row (if any) and key at
/// `level`; `action` is [`static_public_lut_action`] of `D_1`.
pub(crate) fn static_public_lut_identity(
    parameters: &RnsBggParameters,
    level: usize,
    input: (Option<RnsBggRow>, RnsBggMatrix),
    action: &SmallMatrix,
) -> Result<(Option<RnsBggRow>, RnsBggMatrix), RnsBggError> {
    let ring = parameters.ring(level)?;
    let scale = parameters.scalar(level, parameters.message_scale(level)?)?;
    let target = action.clone().right_multiply(scale * parameters.state_gadget(level)?);
    let zero = ring.zero((1, parameters.component_columns(level)?));
    let target = ((target.clone(), zero.clone()), (zero, target));
    let columns = compact_decompose_columns(parameters, level, level, target)?;
    let (row, key) = input;
    Ok(match row {
        Some(row) => {
            let (row, key) = multiply_row_and_public_by_columns(row, key, columns);
            (Some(row), key)
        }
        None => (None, key_by_columns(key, columns)),
    })
}

/// The Static LUT of a public payload at `level` (see the module
/// documentation), before its ModDown: `input` is the row (if any) and key,
/// `payload` the input's public payload in the ring of `level` (needed with a
/// row), and `helpers` the encryption's LUT helpers of `table`'s layout at
/// `level` (needed with a row), member `i` for the `i`-th automorphism. The
/// public matrices `A_sigma` are sampled from `hash_key`. The branches are
/// accumulated sequentially, as for a private LUT, so one is resident at a
/// time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn static_public_lut(
    parameters: &RnsBggParameters,
    hash_key: Bytes,
    level: usize,
    table: &crate::program::LutTable,
    values: super::dynamic::LutEntries<'_>,
    input: (Option<RnsBggRow>, RnsBggMatrix),
    payload: Option<Mat>,
    helpers: Option<&(Family<Mat>, Family<Mat>)>,
) -> Result<(Option<RnsBggRow>, RnsBggMatrix), RnsBggError> {
    use super::dynamic::select;
    let next_level = level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
    let ring_dimension = parameters.level(level)?.ring_dimension() as usize;
    let sigmas =
        table.input_layout().automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
    if sigmas.first() != Some(&1) {
        return Err(RnsBggError::Lut);
    }
    let words = inverse_words(&sigmas, ring_dimension)?;
    let action = |k: IntExpr| {
        let coefficient = EncodingLowering::lut_coefficient(
            parameters,
            next_level,
            table.input_layout(),
            table.output_representation(),
            k,
            values.clone(),
        )?;
        static_public_lut_action(parameters, level, coefficient)
    };
    let initial = static_public_lut_identity(
        parameters,
        level,
        input.clone(),
        &action(IntExpr::constant(1))?,
    )?;
    if sigmas.len() == 1 {
        return Ok(initial);
    }
    let public = |sigma: IntExpr| {
        super::public_key::lut_public_matrix(parameters, hash_key.clone(), level, Int::from(sigma))
    };
    let helper_row = |position: IntExpr| {
        helpers.map(|(top, bottom)| {
            let position = Int::evaluate(position);
            (top.at(position.clone()), bottom.at(position))
        })
    };
    let schema = |_| DslError::Schema;
    let (row, key) = input;
    // The nonidentity branches grouped by the length of their inverse word,
    // so that one loop body per length builds every bridge of the group.
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (position, word) in words.iter().enumerate().skip(1) {
        groups.entry(word.len()).or_default().push(position);
    }
    // The loop state holds the row sum as a vector of zero or one row.
    let mut sum = (initial.0.into_iter().collect::<Vec<_>>(), initial.1);
    for (length, positions) in groups {
        // Per hop `i`: the helper position of `j_i` and `r_i = (j_1 ... j_i)^-1`.
        let hops = (0..length)
            .map(|hop| {
                let members =
                    positions.iter().map(|branch| words[*branch][hop]).collect::<Vec<_>>();
                let routes = positions
                    .iter()
                    .map(|branch| {
                        let prefix = words[*branch][..=hop].iter().fold(1u128, |product, member| {
                            product * sigmas[*member] as u128 % (2 * ring_dimension) as u128
                        }) as usize;
                        super::setup::automorphism_inverse(prefix, ring_dimension)
                            .ok_or(RnsBggError::Lut)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((members, routes))
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        let count = IntExpr::constant(positions.len());
        sum = mxx_dsl::iterate(count, sum, |index, (row_sum, key_sum)| {
            let index = index.expression()?;
            let position = select(&positions, index.clone());
            let k = select(&sigmas, position.clone());
            let route = |r: &IntExpr| {
                let r = r.clone();
                move |entry: Mat| entry.ring_automorphism(r.clone())
            };
            // The routed helper of `j_i` with its public matrix.
            let hop_helper = |hop: usize| -> Result<(Option<RnsBggRow>, RnsBggMatrix), DslError> {
                let (members, routes) = &hops[hop];
                let member = select(members, index.clone());
                let r = select(routes, index.clone());
                let j = select(&sigmas, member.clone());
                Ok((
                    helper_row(member).map(|row| map_row(row, route(&r))),
                    map_matrix(public(j).map_err(schema)?, route(&r)),
                ))
            };
            // `C_1 = sigma_(r_1)(h_(j_1)) V_k`, then one mask switch per
            // further hop, ending under the mask `sigma_k(s)`.
            let action = action(k.clone()).map_err(schema)?;
            let (first_row, first_key) = hop_helper(0)?;
            let mut bridge = (
                first_row.map(|row| super::dynamic::scalar_action_row(row, &action)),
                super::dynamic::scalar_action_matrix(first_key, &action),
            );
            for hop in 1..length {
                let (helper_row, helper_key) = hop_helper(hop)?;
                let (bridge_row, bridge_key) = public_lut_mask_switch(
                    parameters,
                    level,
                    next_level,
                    bridge,
                    (helper_row, Some(helper_key)),
                )
                .map_err(schema)?;
                bridge = (bridge_row, bridge_key.ok_or(DslError::Schema)?);
            }
            let route_k = route(&k);
            let switched = public_lut_switch(
                parameters,
                level,
                (row.clone().map(|row| map_row(row, &route_k)), map_matrix(key.clone(), &route_k)),
                payload.clone().map(&route_k),
                bridge,
                None::<&SmallMatrix>,
            )
            .map_err(schema)?;
            let (branch_row, branch_key) = public_lut_mask_switch(
                parameters,
                level,
                next_level,
                switched,
                (helper_row(position), Some(public(k).map_err(schema)?)),
            )
            .map_err(schema)?;
            Ok((
                add_optional(row_sum, branch_row, add_encoding_rows)?,
                add_public_matrices(key_sum, branch_key.ok_or(DslError::Schema)?),
            ))
        })
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
    }
    Ok((sum.0.into_iter().next(), sum.1))
}

/// The Dynamic LUT of a public payload at `level` (see the module
/// documentation and [`super::dynamic::radix_plan`]): `input` is the row (if
/// any) and key, `payload` the input's public payload in the ring of `level`
/// (needed with a row), `helpers` the imported automorphism helpers (needed
/// with a row), and `key` whether the output key is computed. The identity
/// branch is the scalar action of `D_1`. Per depth of the radix plan, the
/// parents of the next depth are switched from their parents without a
/// coefficient and kept with their keys, each contributing the scalar action
/// of its coefficient; every other branch is switched from its parent with
/// its coefficient fused. A direct plan is one depth without parents.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dynamic_public_lut(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    hash_key: Bytes,
    level: usize,
    table: &crate::program::LutTable,
    values: super::dynamic::LutEntries<'_>,
    input: (Option<RnsBggRow>, RnsBggMatrix),
    payload: Option<Mat>,
    helpers: Option<&Family<RnsBggRow>>,
    key: bool,
) -> Result<(Option<RnsBggRow>, Option<RnsBggMatrix>), RnsBggError> {
    use super::dynamic::{
        lut_automorphisms, radix_plan, scalar_action, scalar_action_matrix, scalar_action_row,
        select,
    };
    let automorphisms = lut_automorphisms(parameters, setup.automorphisms(), level, table)?;
    let action = |k: IntExpr| {
        scalar_action(
            parameters,
            level,
            EncodingLowering::lut_coefficient(
                parameters,
                level,
                table.input_layout(),
                table.output_representation(),
                k,
                values.clone(),
            )?,
        )
    };
    let identity = action(IntExpr::constant(1))?;
    let (row, input_key) = input;
    let mut sum = (
        row.clone().map(|row| scalar_action_row(row, &identity)).into_iter().collect::<Vec<_>>(),
        key.then(|| scalar_action_matrix(input_key.clone(), &identity))
            .into_iter()
            .collect::<Vec<_>>(),
    );
    let finish = |(rows, keys): (Vec<RnsBggRow>, Vec<RnsBggMatrix>)| {
        (rows.into_iter().next(), keys.into_iter().next())
    };
    let switched = automorphisms[1..].to_vec();
    if switched.is_empty() {
        return Ok(finish(sum));
    }
    let schema = |_| DslError::Schema;
    let graph = |error: DslError| RnsBggError::Graph(error.to_string());
    let members = |ks: &[usize], member: &dyn Fn(usize) -> Result<usize, RnsBggError>| {
        ks.iter().map(|k| member(*k)).collect::<Result<Vec<_>, _>>()
    };
    let helper_row = |member: IntExpr| helpers.map(|helpers| helpers.at(Int::evaluate(member)));
    // Switch `parent` by the `index`-th switch of a depth to its branch, with
    // the branch coefficient fused if `fused`, and its key if `with_key`.
    let switch = |parent: (Option<RnsBggRow>, RnsBggMatrix),
                  switches: &[usize],
                  branches: &[usize],
                  index: IntExpr,
                  fused: bool,
                  with_key: bool|
     -> Result<(Option<RnsBggRow>, Option<RnsBggMatrix>), RnsBggError> {
        let inverses = members(switches, &|k| setup.automorphism_inverse(k))?;
        let s = select(switches, index.clone());
        let inverse = select(&inverses, index.clone());
        let inverse_member =
            select(&members(&inverses, &|k| setup.helper_member(k))?, index.clone());
        let member = select(&members(switches, &|k| setup.helper_member(k))?, index.clone());
        let k = select(branches, index);
        let route = |entry: Mat| entry.ring_automorphism(s.clone());
        let action = fused.then(|| action(k.clone())).transpose()?;
        let switched = public_lut_switch(
            parameters,
            level,
            (parent.0.map(|row| map_row(row, route)), map_matrix(parent.1, route)),
            payload.clone().map(|payload| payload.ring_automorphism(k)),
            (
                helper_row(inverse_member).map(|row| map_row(row, route)),
                map_matrix(setup.helper_key(parameters, hash_key.clone(), inverse)?, route),
            ),
            action.as_ref(),
        )?;
        let helper_key =
            with_key.then(|| setup.helper_key(parameters, hash_key.clone(), s)).transpose()?;
        public_lut_mask_switch(parameters, level, level, switched, (helper_row(member), helper_key))
    };
    let plan = radix_plan(setup, &switched)?;
    // The switched parents of the current depth: their rows (zero or one
    // each) and keys.
    let mut parents: Option<(Family<Vec<RnsBggRow>>, Family<RnsBggMatrix>)> = None;
    for depth in &plan {
        let parent = |branch: IntExpr| match &parents {
            None => (row.clone(), input_key.clone()),
            Some((rows, keys)) => {
                let position = Int::evaluate(select(&depth.parents, branch));
                (rows.at(position.clone()).into_iter().next(), keys.at(position))
            }
        };
        let switched_parents = (depth.internal > 0)
            .then(|| {
                let pairs = mxx_dsl::parallel(depth.internal, |position| {
                    let index = position.expression()?;
                    let (row, key) = switch(
                        parent(index.clone()),
                        &depth.switches,
                        &depth.branches,
                        index,
                        false,
                        true,
                    )
                    .map_err(schema)?;
                    Ok((row.into_iter().collect::<Vec<_>>(), key.ok_or(DslError::Schema)?))
                })
                .map_err(graph)?;
                Ok::<_, RnsBggError>((
                    pairs.field(|(row, _)| row).map_err(graph)?,
                    pairs.field(|(_, key)| key).map_err(graph)?,
                ))
            })
            .transpose()?;
        if let Some((rows, keys)) = &switched_parents {
            sum = mxx_dsl::iterate(
                IntExpr::constant(depth.internal),
                sum,
                |index, (row_sum, key_sum)| {
                    let action =
                        action(select(&depth.branches, index.expression()?)).map_err(schema)?;
                    let row = rows
                        .at(index.clone())
                        .into_iter()
                        .next()
                        .map(|row| scalar_action_row(row, &action));
                    let branch_key = key.then(|| scalar_action_matrix(keys.at(index), &action));
                    Ok((
                        add_optional(row_sum, row, add_encoding_rows)?,
                        add_optional(key_sum, branch_key, add_public_matrices)?,
                    ))
                },
            )
            .map_err(graph)?;
        }
        let leaves = depth.branches.len() - depth.internal;
        if leaves > 0 {
            sum = mxx_dsl::iterate(IntExpr::constant(leaves), sum, |index, (row_sum, key_sum)| {
                let branch =
                    (IntExpr::constant(depth.internal) + index.expression()?).canonicalize();
                let (row, branch_key) = switch(
                    parent(branch.clone()),
                    &depth.switches,
                    &depth.branches,
                    branch,
                    true,
                    key,
                )
                .map_err(schema)?;
                Ok((
                    add_optional(row_sum, row, add_encoding_rows)?,
                    add_optional(key_sum, branch_key, add_public_matrices)?,
                ))
            })
            .map_err(graph)?;
        }
        parents = switched_parents;
    }
    Ok(finish(sum))
}
