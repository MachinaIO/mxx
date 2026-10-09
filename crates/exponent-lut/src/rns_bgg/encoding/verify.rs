//! Output verification and decoding.
use super::*;

/// Verify one output returned by [`crate::api::online_eval`].
///
/// The output is a paired RNS-BGG carrier. Its public-key ports, encoded-row
/// ports, and (for public outputs) public payload are consumed from the
/// ordinary runtime result owner. The residual graph and its final decode are
/// concrete RNS-BGG application operations; no generic runtime decoder or
/// verification artifact is exposed. `payload` and `mask` remain
/// caller-owned, and neither is passed to online evaluation. Dynamic
/// parameters take no mask: their single secret `payload` is also the mask.
pub fn verify_output<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    expected: &P,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
    execution: &mut E,
    store: &mut S,
) -> Result<(), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let (wire, output_level, decoding) = output_wire_level(program, parameters, output_index)?;
    verify_output_at_level(
        program,
        result,
        output_index,
        wire,
        output_level,
        decoding,
        expected,
        parameters,
        payload,
        mask,
        execution,
        store,
    )
}

/// The wire of output `output_index`, the level of its finalized carrier (a
/// coefficient output is moved down once more by `finish_outputs`), and, for
/// a coefficient output, its decryption modulus `t` and scale `k` (see
/// [`RnsBggParameters::coefficient_output_decoding`]). A member of a
/// `parallel` family is described by its body's output wire, a member of an
/// input family by the family's own wire.
fn output_wire_level(
    program: &ExponentLutProgram,
    parameters: &RnsBggParameters,
    output_index: usize,
) -> Result<(WireId, usize, Option<(u64, BigUint)>), ApiError> {
    let wire = match program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("verification output index is out of range"))?
    {
        Operand::Scalar(wire) => *wire,
        Operand::Member { family, .. } => program
            .family_body_outputs()
            .into_iter()
            .find_map(|(output, inner)| (output == *family).then_some(inner))
            .unwrap_or(*family),
    };
    let workload = RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?;
    let source_level = workload.level(wire).map_err(ApiError::scheme)?;
    let coefficient_output = program
        .wire_is_coefficient_output(wire)
        .map_err(|_| ApiError::input("verification output metadata is invalid"))?;
    let output_level = source_level
        .checked_add(usize::from(coefficient_output))
        .ok_or_else(|| ApiError::input("verification output level overflow"))?;
    let decoding = coefficient_output
        .then(|| parameters.coefficient_output_decoding(program, wire, output_level))
        .transpose()
        .map_err(ApiError::scheme)?
        .map(|(t, k, _)| (t, k));
    Ok((wire, output_level, decoding))
}

/// Decrypt private output `output_index` of [`crate::api::online_eval`] with
/// the caller's mask (for Dynamic parameters, the payload secret, which is
/// also the mask) and return its plaintext coefficients at `positions`:
/// modulo two for an exponent output, modulo its decryption modulus `t` for
/// a coefficient output.
///
/// A private output is a terminal-level carrier: the terminal state gadget
/// is the identity, its message scale is `Delta` (`floor(q / 2)` for an
/// exponent output, the scale `k` of [`RnsBggParameters::coefficient_output_scale`]
/// for a coefficient output), and the second component of the payload
/// secret is one. Its second row component is therefore
/// `s * A_01 + A_11 - Delta * mu + e` for the mask `s = (s_0, 1)` and the
/// paired public key `A`, so the phase `row_1 - (s_0 * A_01 + A_11)` is
/// `-Delta * mu + e`. Threshold decoding modulo two of the phase returns
/// every coefficient of an exponent output's `mu` modulo two whenever the
/// error is below `q / 4`; threshold decoding modulo `t` of the negated phase
/// `k mu - e` returns `mu mod t` whenever `2 t |e| + 2 r W < q` (`t k = q - r`,
/// `|mu| <= W`). The payload secret is not needed. The graph is
/// the one decryption step of the workflow: its inputs are the online
/// evaluation outputs `output-{output_index}.1`, `.3`, and `.5`, named
/// `decode-key-01`, `decode-key-11`, and `decode-row-1`, and the mask, and
/// its outputs are `decoded-{position}`.
#[allow(clippy::too_many_arguments)]
pub fn decode_output<E, S>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
    positions: &[usize],
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
) -> Result<Vec<BigInt>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let (wire, output_level, decoding) = output_wire_level(program, parameters, output_index)?;
    if output_level + 1 != parameters.levels().len() {
        return Err(ApiError::input("only a terminal-level output can be decoded"));
    }
    if !matches!(
        program.wire_knowledge(wire).map_err(ApiError::scheme)?,
        AttributeKnowledge::PrivatePayload
    ) {
        return Err(ApiError::input("only a private output is decoded"));
    }
    let ring = parameters.ring(output_level).map_err(ApiError::scheme)?;
    let dimension = parameters.level(output_level).map_err(ApiError::scheme)?.ring_dimension();
    let length = positions.iter().max().map_or(0, |position| position + 1);
    if length > dimension as usize {
        return Err(ApiError::input("decoded position is outside the ring"));
    }
    let mut runtime = BTreeMap::new();
    for (port, input) in [(1, "decode-key-01"), (3, "decode-key-11"), (5, "decode-row-1")] {
        let value = result
            .outputs
            .get(&format!("output-{output_index}.{port}"))
            .ok_or_else(|| ApiError::input("decoded output port is missing"))?;
        runtime.insert(input.to_owned(), value.clone());
    }
    let mask = &stage_mask(parameters, payload, mask)?;
    let mask_row = reduce_secret(parameters, &mask.row, output_level).map_err(ApiError::scheme)?;
    let phase = ring.input("decode-row-1", (1, 1)) -
        (mask_row.0 * ring.input("decode-key-01", (1, 1)) +
            mask_row.1 * ring.input("decode-key-11", (1, 1)));
    let mut graph = DslContext::new("rns-bgg-output-decode");
    let decoded = match &decoding {
        Some((t, _)) => (-phase).threshold_decode_ints(*t, length),
        None => phase.threshold_decode_ints(2u8, length),
    };
    for position in positions {
        graph = graph
            .output(format!("decoded-{position}"), decoded[*position].clone())
            .map_err(ApiError::scheme)?;
    }
    let graph = graph.build().map_err(ApiError::scheme)?;
    if let Some(binding) = &mask.binding {
        binding.bind(&mut runtime);
    }
    let runtime = declared_runtime_inputs(&graph, runtime);
    let outputs = crate::api::execute_graph(
        StageGraph::new(graph, runtime, BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        execution,
        store,
        before_execute,
    )?;
    positions
        .iter()
        .map(|position| {
            let value = outputs
                .outputs
                .get(&format!("decoded-{position}"))
                .ok_or_else(|| ApiError::input("decoded coefficient is missing"))?;
            let [value] = <[BigInt; 1]>::try_from(execution.download_integers(value)?)
                .map_err(|_| ApiError::input("decoded coefficient is not one integer"))?;
            Ok(value)
        })
        .collect()
}

/// Check public output `output_index` of [`crate::api::online_eval`]
/// against the plaintext `expected` with the caller's mask (for Dynamic
/// parameters, the payload secret), and return the number of residual
/// coefficients that do not decode to zero.
///
/// A public output is a terminal-level carrier whose payload is the mask
/// `s = (s_0, 1)`: its row components are `c_j = s_0 A_0j + A_1j - mu s_j G
/// + e_j` for the paired public key `A`, its public plaintext `mu`, and the
/// native gadget `G` of the level. Unlike a private output it has no
/// coefficient scaled by `floor(q / 2)`, so it is not decrypted; instead the
/// residual `c_j - (s_0 A_0j + A_1j) + expected s_j G` is `e_j` exactly when
/// `expected = mu`, and every coefficient of every column of both components
/// threshold-decodes to zero modulo two whenever the error is below `q / 4`.
/// The graph is the one check of the workflow: its inputs are the online
/// evaluation outputs `output-{output_index}.0` to `.5`, named
/// `decode-key-{port}` and `decode-row-{component}`, the mask, and the
/// plaintext `decode-expected.coefficient` and `decode-expected.exponent`
/// (a signed monomial `c X^k`); its output `decoded-residual`
/// is the sum of the decoded coefficients.
#[allow(clippy::too_many_arguments)]
pub fn decode_public_output<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
    expected: &P,
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
) -> Result<BigInt, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let (wire, output_level, _) = output_wire_level(program, parameters, output_index)?;
    if output_level + 1 != parameters.levels().len() {
        return Err(ApiError::input("only a terminal-level output can be decoded"));
    }
    if !matches!(
        program.wire_knowledge(wire).map_err(ApiError::scheme)?,
        AttributeKnowledge::PublicPayload
    ) {
        return Err(ApiError::input("only a public output is checked against its plaintext"));
    }
    let ring = parameters.ring(output_level).map_err(ApiError::scheme)?;
    let level = parameters.level(output_level).map_err(ApiError::scheme)?;
    let dimension = level.ring_dimension() as usize;
    let columns = parameters
        .key_component_columns(output_level, AttributeKnowledge::PublicPayload)
        .map_err(ApiError::scheme)?;
    let mut runtime = BTreeMap::new();
    for port in 0..6 {
        let value = result
            .outputs
            .get(&format!("output-{output_index}.{port}"))
            .ok_or_else(|| ApiError::input("decoded output port is missing"))?;
        let name = if port < 4 {
            format!("decode-key-{port}")
        } else {
            format!("decode-row-{}", port - 4)
        };
        runtime.insert(name, value.clone());
    }
    let expected = match super::plaintext_inputs(
        &ring,
        "decode-expected",
        &InputBinding::Scalar(expected.clone()),
        &mut runtime,
    )
    .map_err(ApiError::input)?
    {
        InputBinding::Scalar(expected) => expected,
        InputBinding::Family(_) => unreachable!("a scalar plaintext binds a scalar"),
    };
    let mask = &stage_mask(parameters, payload, mask)?;
    let mask_row = reduce_secret(parameters, &mask.row, output_level).map_err(ApiError::scheme)?;
    let mask_row = [mask_row.0, mask_row.1];
    let key = |port: usize| ring.input(format!("decode-key-{port}"), (1, columns));
    let scaled = expected * parameters.public_gadget(output_level).map_err(ApiError::scheme)?;
    let mut total = Int::constant(0);
    for (component, mask_component) in mask_row.iter().enumerate() {
        let residual = ring.input(format!("decode-row-{component}"), (1, columns)) -
            (mask_row[0].clone() * key(component) + mask_row[1].clone() * key(2 + component)) +
            mask_component.clone() * scaled.clone();
        for column in 0..columns {
            let entry = residual.clone().slice(
                None,
                Some(mxx_ir_core::node::IndexRange {
                    start: IntExpr::constant(column),
                    end: IntExpr::constant(column + 1),
                }),
            );
            for decoded in entry.threshold_decode_ints(2u8, dimension) {
                total = total.add(decoded);
            }
        }
    }
    let graph = DslContext::new("rns-bgg-public-output-decode")
        .output("decoded-residual", total)
        .map_err(ApiError::scheme)?
        .build()
        .map_err(ApiError::scheme)?;
    if let Some(binding) = &mask.binding {
        binding.bind(&mut runtime);
    }
    let runtime = declared_runtime_inputs(&graph, runtime);
    let outputs = crate::api::execute_graph(
        StageGraph::new(graph, runtime, BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        execution,
        store,
        before_execute,
    )?;
    let value = outputs
        .outputs
        .get("decoded-residual")
        .ok_or_else(|| ApiError::input("decoded residual is missing"))?;
    let [value] = <[BigInt; 1]>::try_from(execution.download_integers(value)?)
        .map_err(|_| ApiError::input("decoded residual is not one integer"))?;
    Ok(value)
}

/// The error `e' = y - (s A' - Delta mu G_Q)` of the refreshed row `y`
/// (output `refresh` of an [`crate::api::online_eval`] with
/// `declaration`'s refresh of `source`) for the target plaintext `expected`
/// (a signed monomial `c X^k`), under the caller's mask (for Dynamic
/// parameters, the payload secret): per gadget column of `G_Q`, the `n`
/// coefficients centered modulo `Q`. `A'` is sampled from `hash_key`, and
/// `Delta` and `G_Q` are those of the target's level. The graph is the one
/// check of the workflow: its inputs are the row (`refresh-row`), the mask,
/// the hash key and the plaintext `refresh-expected.coefficient` and
/// `refresh-expected.exponent`; its output `refresh-error` is the row error.
#[allow(clippy::too_many_arguments)]
pub fn refresh_error<E, S, P>(
    declaration: &crate::refresh::RefreshDeclaration,
    source: &ExponentLutProgram,
    result: &StageOutputs,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
    expected: &P,
    hash_key: &[u8],
    execution: &mut E,
    store: &mut S,
) -> Result<Vec<Vec<BigInt>>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let level = parameters.refresh_level(declaration, source).map_err(ApiError::scheme)?;
    let ring = parameters.ring(level).map_err(ApiError::scheme)?;
    let columns = parameters.level(level).map_err(ApiError::scheme)?.modulus_digits();
    let mut runtime = BTreeMap::new();
    runtime.insert(
        "refresh-row".to_owned(),
        result
            .outputs
            .get("refresh")
            .ok_or_else(|| ApiError::input("the refreshed row is missing"))?
            .clone(),
    );
    runtime.insert(super::HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.into()));
    let expected = match super::plaintext_inputs(
        &ring,
        "refresh-expected",
        &InputBinding::Scalar(expected.clone()),
        &mut runtime,
    )
    .map_err(ApiError::input)?
    {
        InputBinding::Scalar(expected) => expected,
        InputBinding::Family(_) => unreachable!("a scalar plaintext binds a scalar"),
    };
    let mask = &stage_mask(parameters, payload, mask)?;
    let s = reduce_secret(parameters, &mask.row, level).map_err(ApiError::scheme)?;
    let a_prime = parameters
        .refresh_matrix(level, ring.bytes_input(super::HASH_KEY_INPUT, hash_key.len()), declaration)
        .map_err(ApiError::scheme)?;
    let a_row = |row: usize| {
        a_prime.clone().slice(
            Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(row),
                end: IntExpr::constant(row + 1),
            }),
            None,
        )
    };
    let delta = parameters
        .scalar(level, parameters.message_scale(level).map_err(ApiError::scheme)?)
        .map_err(ApiError::scheme)?;
    let error = ring.input("refresh-row", (1, columns)) -
        (s.0 * a_row(0) + s.1 * a_row(1) -
            delta * expected * parameters.public_gadget(level).map_err(ApiError::scheme)?);
    let graph = DslContext::new("rns-bgg-refresh-error")
        .output("refresh-error", error)
        .map_err(ApiError::scheme)?
        .build()
        .map_err(ApiError::scheme)?;
    if let Some(binding) = &mask.binding {
        binding.bind(&mut runtime);
    }
    let runtime = declared_runtime_inputs(&graph, runtime);
    let outputs = crate::api::execute_graph(
        StageGraph::new(graph, runtime, BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        execution,
        store,
        |_| Ok(()),
    )?;
    let error = execution.download_matrix(
        outputs
            .outputs
            .get("refresh-error")
            .ok_or_else(|| ApiError::input("the refresh error is missing"))?,
    )?;
    let modulus =
        BigInt::from(parameters.level(level).map_err(ApiError::scheme)?.modulus().as_ref().clone());
    Ok((0..columns)
        .map(|column| {
            error
                .entry(0, column)
                .coeffs_biguints()
                .into_iter()
                .map(|value| {
                    let value = BigInt::from(value);
                    if &value * 2u8 > modulus { value - &modulus } else { value }
                })
                .collect()
        })
        .collect())
}

/// The error rows `e'_j = y_j - (sigma A'_(j) - Delta_0 r_j x G^st_0)`
/// (`j = 0, 1`) of the refresh to the top level `top`: `y_j` recombines the
/// output `refresh` of every part `(chain, j)` (`results[chain][j]`, the
/// residue `y_j mod q`) modulo `Q_0`; `sigma = (sigma_0, 1)` with the mask
/// child's bit polynomial `sigma`, `x = expected`, `r` the payload secret,
/// and `Delta_0`, `G^st_0` those of the reference chain `reference`
/// ([`RnsBggParameters::top_refresh_chain`] with `None`). `A'_(j)` is
/// sampled under the top refresh's matrix namespace of `hash_key`. Per
/// column, the `n` coefficients centered modulo `Q_0`.
#[allow(clippy::too_many_arguments)]
pub fn top_refresh_error<E, S, P>(
    top: &crate::refresh::top::TopRefreshDeclaration,
    results: &[[&StageOutputs; 2]],
    reference: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    sigma: &P,
    expected: &P,
    hash_key: &[u8],
    execution: &mut E,
    store: &mut S,
) -> Result<[Vec<Vec<BigInt>>; 2], ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    use crate::rns_bgg::refresh::{
        child_hash_key, hash_key_input_name, top_matrix, top_refresh_combine,
    };
    let declaration = top.reference();
    let columns = declaration.columns();
    if results.len() != top.chains() {
        return Err(ApiError::input("one pair of part results per chain"));
    }
    let ring = reference.ring(0).map_err(ApiError::scheme)?;
    let mut runtime = BTreeMap::new();
    let namespace = top.matrix_namespace();
    let key_name = hash_key_input_name(Some(&namespace));
    let key = child_hash_key(hash_key, &namespace);
    runtime.insert(key_name.clone(), RuntimeValue::Bytes(key.as_slice().into()));
    let key = ring.bytes_input(key_name, key.len());
    let mut plaintext = |name: &str, value: &P| match super::plaintext_inputs(
        &ring,
        name,
        &InputBinding::Scalar(value.clone()),
        &mut runtime,
    ) {
        Ok(InputBinding::Scalar(value)) => Ok(value),
        Ok(InputBinding::Family(_)) => unreachable!("a scalar plaintext binds a scalar"),
        Err(error) => Err(ApiError::input(error)),
    };
    let sigma = plaintext("top-refresh-sigma", sigma)?;
    let expected = plaintext("top-refresh-expected", expected)?;
    let r = reduce_secret(reference, &payload.row, 0).map_err(ApiError::scheme)?;
    let delta = reference
        .scalar(0, reference.message_scale(0).map_err(ApiError::scheme)?)
        .map_err(ApiError::scheme)?;
    let gadget = reference.state_gadget(0).map_err(ApiError::scheme)?;
    let mut graph = DslContext::new("rns-bgg-top-refresh-error");
    for component in 0..2 {
        let part =
            top.part(0, component).ok_or_else(|| ApiError::input("the top refresh has no part"))?;
        let residues = results
            .iter()
            .enumerate()
            .map(|(chain, result)| {
                let name = format!("residue-{chain}-{component}");
                let value = result[component]
                    .outputs
                    .get("refresh")
                    .ok_or_else(|| ApiError::input("a part's residue is missing"))?
                    .clone();
                runtime.insert(name.clone(), value);
                let prime = top
                    .output_prime(chain)
                    .ok_or_else(|| ApiError::input("the top refresh has no such chain"))?;
                Ok(mxx_dsl::Ring::from_crt_moduli(
                    vec![IntExpr::from(prime)],
                    declaration.ring_dimension,
                )
                .input(name, (1, columns)))
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        let refreshed = top_refresh_combine(declaration, residues).map_err(ApiError::scheme)?;
        let a_prime = top_matrix(declaration, key.clone(), part);
        let a_row = |row: usize| {
            a_prime.clone().slice(
                Some(mxx_ir_core::node::IndexRange {
                    start: IntExpr::constant(row),
                    end: IntExpr::constant(row + 1),
                }),
                None,
            )
        };
        let payload_secret = if component == 0 { r.0.clone() } else { r.1.clone() };
        let error = refreshed -
            (sigma.clone() * a_row(0) + a_row(1) -
                delta.clone() * payload_secret * expected.clone() * gadget.clone());
        graph = graph
            .output(format!("top-refresh-error-{component}"), error)
            .map_err(ApiError::scheme)?;
    }
    let graph = graph.build().map_err(ApiError::scheme)?;
    if let Some(binding) = &payload.binding {
        binding.bind(&mut runtime);
    }
    let runtime = declared_runtime_inputs(&graph, runtime);
    let outputs = crate::api::execute_graph(
        StageGraph::new(graph, runtime, BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        execution,
        store,
        |_| Ok(()),
    )?;
    let modulus =
        BigInt::from(reference.level(0).map_err(ApiError::scheme)?.modulus().as_ref().clone());
    let mut errors = [Vec::new(), Vec::new()];
    for (component, rows) in errors.iter_mut().enumerate() {
        let error = execution.download_matrix(
            outputs
                .outputs
                .get(&format!("top-refresh-error-{component}"))
                .ok_or_else(|| ApiError::input("the top refresh error is missing"))?,
        )?;
        *rows = (0..columns)
            .map(|column| {
                error
                    .entry(0, column)
                    .coeffs_biguints()
                    .into_iter()
                    .map(|value| {
                        let value = BigInt::from(value);
                        if &value * 2u8 > modulus { value - &modulus } else { value }
                    })
                    .collect()
            })
            .collect();
    }
    Ok(errors)
}

fn verify_output_at_level<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    wire: WireId,
    output_level: usize,
    decoding: Option<(u64, BigUint)>,
    expected: &P,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
    execution: &mut E,
    store: &mut S,
) -> Result<(), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let mask = &stage_mask(parameters, payload, mask)?;
    // Only the terminal relation (identity state gadget, scale floor(q / 2)
    // or the coefficient scale k) separates plaintexts by more than the
    // decoding threshold; an intermediate carrier's gadget powers can be
    // smaller than its noise, so it cannot be verified by decoding.
    if output_level + 1 != parameters.levels().len() {
        return Err(ApiError::input(format!(
            "verification output is at level {output_level}, not the terminal level {}",
            parameters.levels().len() - 1
        )));
    }
    let output_ring = parameters.ring(output_level).map_err(ApiError::scheme)?;
    let knowledge = program.wire_knowledge(wire).map_err(ApiError::scheme)?;
    let private_output = matches!(knowledge, AttributeKnowledge::PrivatePayload);
    let columns =
        parameters.key_component_columns(output_level, knowledge).map_err(ApiError::scheme)?;
    let expected_ports = if private_output { 6 } else { 7 };
    let prefix = format!("output-{output_index}.");
    let mut ports = BTreeMap::new();
    for name in result.outputs.keys().filter(|name| name.starts_with(&prefix)) {
        let port = name
            .strip_prefix(&prefix)
            .and_then(|port| port.parse::<usize>().ok())
            .ok_or_else(|| ApiError::input("verification output port is malformed"))?;
        if port >= expected_ports || ports.insert(port, name.clone()).is_some() {
            return Err(ApiError::input("verification output port is out of range"));
        }
    }
    if ports.len() != expected_ports || ports.keys().copied().ne(0..expected_ports) {
        return Err(ApiError::input(format!(
            "verification output has {} ports, expected {expected_ports}",
            ports.len()
        )));
    }

    let mut runtime_inputs = BTreeMap::new();
    for (port, name) in ports {
        let value = result
            .outputs
            .get(&name)
            .ok_or_else(|| ApiError::input("verification output port is missing"))?;
        if !matches!(value, RuntimeValue::Matrix(_)) {
            return Err(ApiError::input("verification output port is not a matrix"));
        }
        let input_name = match port {
            0..=3 => format!("verification-key-{port}"),
            4..=5 => format!("verification-row-{}", port - 4),
            6 => "verification-payload".to_owned(),
            _ => unreachable!("port range checked above"),
        };
        runtime_inputs.insert(input_name, value.clone());
    }

    let key_ports = (0..4)
        .map(|port| output_ring.input(format!("verification-key-{port}"), (1, columns)))
        .collect::<Vec<_>>();
    let key = RnsBggPublicKey::new(
        wire,
        (
            (key_ports[0].clone(), key_ports[1].clone()),
            (key_ports[2].clone(), key_ports[3].clone()),
        ),
    );
    let row = (
        output_ring.input("verification-row-0", (1, columns)),
        output_ring.input("verification-row-1", (1, columns)),
    );
    let public_payload =
        if private_output { None } else { Some(output_ring.input("verification-payload", (1, 1))) };

    // At source level l, the LUT switching target contributes
    // Delta_l * Delta_(l+1) * mu * r * G_(l+2). The one production
    // ModDown removes Delta_l. The finalized carrier therefore obeys the
    // ordinary level-(l+1) relation, with its own scale and gadget width.
    // Verify that target-ring relation directly; extending a finalized
    // carrier cannot recover its pre-ModDown representative.
    let expected = plaintext_constant(&output_ring, expected);
    let public_payload_residual =
        public_payload.as_ref().map(|payload| payload.clone() - expected.clone());
    let scale = match &decoding {
        Some((_, k)) => k.clone(),
        None => parameters.message_scale(output_level).map_err(ApiError::scheme)?,
    };
    let plaintext_modulus = BigInt::from(decoding.as_ref().map_or(2, |(t, _)| *t));
    let encoding = RnsBggEncoding::new(key, row, public_payload);
    let residual = encoding
        .verification_residual(parameters, output_level, payload, mask, expected, &scale)
        .map_err(ApiError::scheme)?;
    // The backend decoder consumes one polynomial (entry 0,0). Check every
    // gadget column, including the high powers which distinguish plaintexts.
    let mut graph = DslContext::new("rns-bgg-output-verification");
    let mut residual_names = Vec::with_capacity(2 * columns);
    for (component, row) in [residual.0, residual.1].into_iter().enumerate() {
        for column in 0..columns {
            let name = format!("verification-residual.{component}.{column}");
            let value = row.clone().slice(
                None,
                Some(mxx_ir_core::node::IndexRange {
                    start: IntExpr::constant(column),
                    end: IntExpr::constant(column + 1),
                }),
            );
            graph = graph.output(name.clone(), value).map_err(ApiError::scheme)?;
            residual_names.push(name);
        }
    }
    if let Some(payload_residual) = public_payload_residual {
        graph = graph
            .output("verification-payload-residual", payload_residual)
            .map_err(ApiError::scheme)?;
    }
    let graph = graph.build().map_err(ApiError::scheme)?;
    let mut runtime_inputs = runtime_inputs;
    for binding in [&payload.binding, &mask.binding].into_iter().flatten() {
        binding.bind(&mut runtime_inputs);
    }
    let runtime_inputs = declared_runtime_inputs(&graph, runtime_inputs);
    let verification = crate::api::execute_graph(
        StageGraph::new(graph, runtime_inputs, BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        execution,
        store,
        |_| Ok(()),
    )?;
    let output_parameters = parameters.level(output_level).map_err(ApiError::scheme)?;
    let modulus = BigInt::from(output_parameters.modulus().as_ref().clone());
    let download = |name: &str| -> Result<DCRTPolyMatrix, ApiError> {
        let value = verification
            .outputs
            .get(name)
            .ok_or_else(|| ApiError::input("output verification residual is missing"))?;
        execution.download_matrix(value)
    };
    for name in residual_names {
        let residual = download(&name)?;
        let decoded = threshold_decode_coefficients(
            residual.entry(0, 0).coeffs_biguints(),
            &modulus,
            &plaintext_modulus,
            output_parameters.ring_dimension() as usize,
        );
        if decoded.iter().any(|value| value != &BigInt::from(0u8)) {
            return Err(ApiError::input("verification residual is nonzero"));
        }
    }
    if !private_output {
        let residual = download("verification-payload-residual")?;
        if residual.entry(0, 0).coeffs_biguints().iter().any(|value| !value.is_zero()) {
            return Err(ApiError::input("public payload does not match expected plaintext"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use mxx_backends::{
        MemoryArtifactStore, backend::poly::cpu_backend, poly::dcrt::poly::DCRTPoly,
    };

    /// A finalized output below the terminal level is refused: its gadget
    /// powers need not exceed its noise.
    #[test]
    fn intermediate_coefficient_verification_is_refused() {
        let high = DCRTPolyParams::new(16, 3, 17, 1, None, None);
        let primes = high.to_crt().0;
        let middle =
            high.select_modulus(&(BigUint::from(primes[0]) * BigUint::from(primes[1]))).unwrap();
        let low = high.select_modulus(&BigUint::from(primes[0])).unwrap();
        let parameters =
            RnsBggParameters::new(vec![high.clone(), middle.clone(), low], 1, 1).unwrap();
        check_finalized_coefficient_verification(&parameters, &middle, false);
    }

    /// The terminal level has an identity state gadget, so only its public
    /// message scale separates a correct output from a wrong one.
    #[test]
    fn terminal_coefficient_verification_rejects_wrong_plaintext() {
        let high = DCRTPolyParams::new(16, 2, 17, 1, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low.clone()], 1, 1).unwrap();
        check_finalized_coefficient_verification(&parameters, &low, true);
    }

    fn check_finalized_coefficient_verification(
        parameters: &RnsBggParameters,
        output_level: &DCRTPolyParams,
        terminal: bool,
    ) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let lut = builder
            .register_lut(
                LutTable::unary(
                    2,
                    ExpLayout { stride_log2: 0, width_log2: 1 },
                    2,
                    vec![0, 1],
                    crate::program::LutOutputRepresentation::Coefficient,
                )
                .unwrap(),
            )
            .unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash(rand::random()),
            execution_nonce: rand::random(),
        };
        let mut lowerer = EncodingLowering {
            parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            workload: Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap()),
            program: program.clone(),
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: None,
        };
        let source_ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret {
            row: (source_ring.identity(1), source_ring.identity(1)),
            binding: None,
        };
        let mask = RnsBggMaskSecret {
            row: (source_ring.zero((1, 1)), source_ring.identity(1)),
            binding: None,
        };
        // The raw carrier of a coefficient LUT at level 0: `Delta_0 v G_1`
        // against the full level-1 public gadget, which the one ModDown
        // turns into `v G_1` and the scale-up action into `k v`.
        let columns = parameters
            .lut_output_columns(1, crate::program::LutOutputRepresentation::Coefficient)
            .unwrap();
        let zero = source_ring.zero((1, columns));
        let key =
            RnsBggPublicKey::new(output, ((zero.clone(), zero.clone()), (zero.clone(), zero)));
        let target_gadget =
            parameters.public_gadget(1).unwrap().centered_rebase(&parameters.ring(0).unwrap());
        let raw_signal =
            target_gadget * parameters.scalar(0, parameters.scale(0).unwrap().clone()).unwrap();
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
            parameters.levels().iter().cloned(),
        ));
        let mut store = MemoryArtifactStore::default();
        for bit in [0usize, 1] {
            let row = -raw_signal.clone() * source_ring.polynomial([IntExpr::constant(bit)]);
            let raw = RnsBggEncoding::new(key.clone(), (row.clone(), row), None);
            let finalized = lowerer.finish_outputs(vec![raw]).unwrap().pop().unwrap();
            let graph = DslContext::new("coefficient-finalization-verification")
                .output("output-0", finalized)
                .unwrap()
                .build()
                .unwrap();
            let result = execute_graph(
                StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let verified = verify_output(
                &program,
                &result,
                0,
                &DCRTPoly::from_usize_to_constant(output_level, bit),
                parameters,
                &payload,
                Some(&mask),
                &mut execution,
                &mut store,
            );
            if !terminal {
                assert!(verified.is_err(), "an intermediate output must not be verified");
                continue;
            }
            verified.unwrap();
            assert!(
                verify_output(
                    &program,
                    &result,
                    0,
                    &DCRTPoly::from_usize_to_constant(output_level, 1 - bit),
                    parameters,
                    &payload,
                    Some(&mask),
                    &mut execution,
                    &mut store
                )
                .is_err(),
                "wrong coefficient plaintext must be rejected"
            );
        }
    }

    #[test]
    fn public_payload_verification_rejects_corruption() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(input)]).unwrap());
        let ring = parameters.ring(0).unwrap();
        let columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let zero = ring.zero((1, columns));
        let key = RnsBggPublicKey::new(
            input,
            ((zero.clone(), zero.clone()), (zero.clone(), zero.clone())),
        );
        let expected_poly = DCRTPoly::from_usize_to_constant(&level, 1);
        let expected = plaintext_constant(&ring, &expected_poly);
        let mask = RnsBggMaskSecret { row: (ring.zero((1, 1)), ring.identity(1)), binding: None };
        let delta = parameters.scalar(0, parameters.scale(0).unwrap().clone()).unwrap();
        let deterministic =
            mask.row.1.clone() * expected.clone() * parameters.public_gadget(0).unwrap() * delta;
        let row = (ring.zero((1, columns)), -deterministic);
        let payload =
            RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)), binding: None };

        for (name, public_payload, expected_result) in [
            ("public-payload-valid", expected.clone(), Ok(())),
            ("public-payload-corrupt", expected + ring.identity(1), Err(())),
        ] {
            let graph = DslContext::new(name)
                .output(
                    "output-0",
                    RnsBggEncoding::new(key.clone(), row.clone(), Some(public_payload)),
                )
                .unwrap()
                .build()
                .unwrap();
            let mut execution =
                mxx_backends::authority::CpuExecution::new(cpu_backend([level.clone()]));
            let mut store = MemoryArtifactStore::default();
            let result = execute_graph(
                StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let verified = verify_output(
                &program,
                &result,
                0,
                &expected_poly,
                &parameters,
                &payload,
                Some(&mask),
                &mut execution,
                &mut store,
            )
            .map(|_| ());
            assert_eq!(verified.is_ok(), expected_result.is_ok(), "{name}");
        }
    }

    #[test]
    fn verification_residual_uses_the_paired_public_key() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let a = ring.input("verification-paired-key-a", (1, 1));
        let b = ring.input("verification-paired-key-b", (1, 1));
        let key_a =
            RnsBggPublicKey::new(WireId(0), ((a.clone(), a.clone()), (a.clone(), a.clone())));
        let key_b =
            RnsBggPublicKey::new(WireId(0), ((b.clone(), b.clone()), (b.clone(), b.clone())));
        let row = (
            ring.input("verification-paired-row-0", (1, 1)),
            ring.input("verification-paired-row-1", (1, 1)),
        );
        let payload =
            RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)), binding: None };
        let mask = RnsBggMaskSecret { row: (ring.identity(1), ring.identity(1)), binding: None };
        let expected = ring.polynomial([IntExpr::constant(BigInt::from(1u8))]);
        let residual_a = RnsBggEncoding::new(key_a, row.clone(), None)
            .verification_residual(
                &parameters,
                0,
                &payload,
                &mask,
                expected.clone(),
                &parameters.message_scale(0).unwrap(),
            )
            .unwrap();
        let residual_b = RnsBggEncoding::new(key_b, row, None)
            .verification_residual(
                &parameters,
                0,
                &payload,
                &mask,
                expected,
                &parameters.message_scale(0).unwrap(),
            )
            .unwrap();
        assert_ne!(residual_a.flatten(), residual_b.flatten());
    }

    /// Two levels of two and one 17-bit primes, so the switching target of a
    /// LUT at level 0 is wider for a coefficient table (`modulus_digits(1)`)
    /// than for an exponent one (`component_columns(1)`, one column at the
    /// terminal level). Ring dimension and sigma are overridable by
    /// `MXX_EXPONENT_LUT_COEFFICIENT_TEST_{RING_DIMENSION,SIGMA}`.
    fn unequal_width_parameters(mode: RhsMode) -> RnsBggParameters {
        let read = |name: &str, default: &str| {
            std::env::var(format!("MXX_EXPONENT_LUT_COEFFICIENT_TEST_{name}"))
                .unwrap_or_else(|_| default.to_owned())
        };
        let ring_dimension = read("RING_DIMENSION", "16").parse().unwrap();
        let sigma = read("SIGMA", "1").parse::<bigdecimal::BigDecimal>().unwrap();
        let high = DCRTPolyParams::new(ring_dimension, 2, 17, 6, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        RnsBggParameters::new(vec![high, low], sigma, 1).unwrap().with_rhs_mode(mode)
    }

    /// Preprocess, encrypt and evaluate `program` with the private scalar
    /// input bound to `input`, under `parameters`' RHS mode, and return the
    /// online result with the payload secret and the stage mask. The online
    /// evaluation receives the Program without its LUT values.
    fn run_without_rhs(
        parameters: &RnsBggParameters,
        program: &ExponentLutProgram,
        lhs: &crate::api::PlaintextInputs<WireId, DCRTPoly>,
        execution: &mut mxx_backends::authority::CpuExecution,
        store: &mut MemoryArtifactStore,
    ) -> (StageOutputs, RnsBggPayloadSecret, Option<RnsBggMaskSecret>) {
        let hash_key: [u8; 32] = rand::random();
        let context: [u8; 32] = rand::random();
        let dynamic = parameters.rhs_mode() == RhsMode::Dynamic;
        let sample_payload = |execution: &mut mxx_backends::authority::CpuExecution,
                              store: &mut MemoryArtifactStore| {
            RnsEncryptBuild::new(parameters, context)
                .sample_payload_secret(execution, store, |_| Ok(()))
                .unwrap()
        };
        let static_payload = (!dynamic).then(|| sample_payload(execution, store));
        let (preprocess, online_eval) = crate::api::preprocess(
            program,
            &ParamEnv::default(),
            &crate::api::PlaintextInputs::<RhsId, DCRTPoly>::new(),
            static_payload.as_ref(),
            &hash_key,
            crate::rns_bgg::RnsPreprocessBuild::new(parameters, context),
            execution,
            store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let payload = static_payload.unwrap_or_else(|| sample_payload(execution, store));
        let mask = (!dynamic).then(|| {
            RnsEncryptBuild::new(parameters, rand::random())
                .sample_mask_secret(execution, store, |_| Ok(()))
                .unwrap()
        });
        let encrypted = crate::api::encrypt(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            lhs,
            &preprocess,
            &payload,
            mask.as_ref(),
            &hash_key,
            RnsEncryptBuild::new(parameters, context),
            execution,
            store,
            |build| {
                if dynamic {
                    build.bind_rhs_inputs(
                        &ParamEnv::default(),
                        &crate::api::PlaintextInputs::<RhsId, DCRTPoly>::new(),
                        rand::random(),
                    )?;
                }
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
        let result = crate::api::online_eval(
            &program.without_lut_values(),
            &ParamEnv::default(),
            &online_eval,
            &encrypted,
            &hash_key,
            RnsOnlineEvalBuild::new(parameters, context),
            execution,
            store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        (result, payload, mask)
    }

    /// A coefficient LUT at level 0 of a chain whose adjacent widths differ:
    /// online evaluation finalizes it once to the terminal level 1, scaling
    /// its `v G_1` payload by `G^-1(k)` to `k v`. Its residual
    /// `row - s A + r k v` is within the noise evaluator's bound for that
    /// output, verification accepts `v` and rejects a wrong plaintext, and
    /// decryption returns `v mod t`, in Static and Dynamic mode, for both
    /// table entries (the larger one is the bound `W`) and both signs of the
    /// input monomial.
    #[test]
    fn coefficient_lut_finalizes_scales_and_decodes() {
        for (mode, bit, negative) in
            [RhsMode::Static, RhsMode::Dynamic].into_iter().flat_map(|mode| {
                [(0, false), (1, false), (0, true), (1, true)].map(|(b, n)| (mode, b, n))
            })
        {
            let parameters = unequal_width_parameters(mode);
            assert_ne!(
                parameters
                    .lut_output_columns(1, crate::program::LutOutputRepresentation::Coefficient)
                    .unwrap(),
                parameters.component_columns(1).unwrap(),
                "the test needs unequal adjacent widths"
            );
            let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
            let values = vec![3, 5];
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let table = builder
                .register_lut(
                    LutTable::unary(
                        2,
                        layout,
                        8,
                        values.clone(),
                        crate::program::LutOutputRepresentation::Coefficient,
                    )
                    .unwrap(),
                )
                .unwrap();
            let output = builder.lut(Operand::Scalar(input), table).unwrap();
            let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
            assert_eq!(program.wire_coefficient_bound(output).unwrap(), Some(5));
            let (t, k, _) = parameters.coefficient_output_decoding(&program, output, 1).unwrap();
            assert_eq!(t, 12);

            let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
            let bindings = crate::rns_bgg::RnsBggNoiseBindings::from_inputs(
                &program,
                &[crate::rns_bgg::RnsBggNoiseValue::sampled_encoding(
                    &parameters,
                    b"coefficient-lut-input",
                    &[BigInt::from(1u8)],
                )],
                &[],
            )
            .unwrap();
            let report = crate::api::noise_simulate(
                &program,
                crate::rns_bgg::RnsNoiseBuild::new(&parameters, &program, &workload, &bindings)
                    .unwrap(),
                |_, outputs| Ok(outputs),
            )
            .unwrap();
            let bound = report.outputs[0].bound.error.clone();

            let level = parameters.levels().first().unwrap();
            let monomial = DCRTPoly::const_rotate_poly(level, bit);
            let lhs = BTreeMap::from([(
                input,
                InputBinding::Scalar(if negative { -monomial } else { monomial }),
            )]);
            let value = if negative { -(values[bit] as i64) } else { values[bit] as i64 };
            let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
                parameters.levels().iter().cloned(),
            ));
            let mut store = MemoryArtifactStore::default();
            let (result, payload, mask) =
                run_without_rhs(&parameters, &program, &lhs, &mut execution, &mut store);

            let finalized = 1;
            let ring = parameters.ring(finalized).unwrap();
            let mut runtime = BTreeMap::new();
            for port in 0..6 {
                runtime.insert(
                    format!("port-{port}"),
                    result.outputs[&format!("output-0.{port}")].clone(),
                );
            }
            let port = |index: usize| ring.input(format!("port-{index}"), (1, 1));
            let stage_mask = stage_mask(&parameters, &payload, mask.as_ref()).unwrap();
            let s = reduce_secret(&parameters, &stage_mask.row, finalized).unwrap();
            let r = reduce_secret(&parameters, &payload.row, finalized).unwrap();
            let message = ring.polynomial([IntExpr::constant(BigInt::from(k.clone()) * value)]);
            let residual = |component: usize, r_k: Mat| {
                port(4 + component) -
                    (s.0.clone() * port(component) + s.1.clone() * port(2 + component)) +
                    r_k * message.clone()
            };
            let graph = DslContext::new("coefficient-lut-residual")
                .output("residual-0", residual(0, r.0.clone()))
                .unwrap()
                .output("residual-1", residual(1, r.1.clone()))
                .unwrap()
                .build()
                .unwrap();
            for binding in [&payload.binding, &stage_mask.binding].into_iter().flatten() {
                binding.bind(&mut runtime);
            }
            let runtime = declared_runtime_inputs(&graph, runtime);
            let residuals = execute_graph(
                StageGraph::new(graph, runtime, BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let modulus =
                BigInt::from(parameters.level(finalized).unwrap().modulus().as_ref().clone());
            for name in ["residual-0", "residual-1"] {
                let matrix =
                    <mxx_backends::authority::CpuExecution as crate::api::StageExecution<
                        MemoryArtifactStore,
                    >>::download_matrix(&execution, &residuals.outputs[name])
                    .unwrap();
                for coefficient in matrix.entry(0, 0).coeffs_biguints() {
                    let value = BigInt::from(coefficient);
                    let centered = if &value * 2u8 > modulus { value - &modulus } else { value };
                    assert!(
                        centered.magnitude() <= &bound,
                        "{mode:?}: residual {centered} exceeds the simulated bound {bound}"
                    );
                }
            }
            let verify = |value: i64, execution: &mut _, store: &mut _| {
                let plaintext = DCRTPoly::from_usize_to_constant(
                    &parameters.levels()[1],
                    value.unsigned_abs() as usize,
                );
                verify_output(
                    &program,
                    &result,
                    0,
                    &if value < 0 { -plaintext } else { plaintext },
                    &parameters,
                    &payload,
                    mask.as_ref(),
                    execution,
                    store,
                )
            };
            verify(value, &mut execution, &mut store).unwrap();
            assert!(verify(-value, &mut execution, &mut store).is_err());
            let decoded = decode_output(
                &program,
                &result,
                0,
                &parameters,
                &payload,
                mask.as_ref(),
                &[0],
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(decoded, vec![BigInt::from(value.rem_euclid(t as i64))], "{mode:?}");
        }
    }

    /// `X^k` in the negacyclic ring of `level`.
    fn monomial(level: &DCRTPolyParams, k: usize) -> DCRTPoly {
        let n = level.ring_dimension() as usize;
        let value = DCRTPoly::const_rotate_poly(level, k % n);
        if (k / n) % 2 == 1 { -value } else { value }
    }

    /// The private member family `X^(e_o)`, `o < exponents.len()`, and its
    /// plaintext binding.
    fn monomial_family(
        builder: &mut ExponentLutProgramBuilder,
        level: &DCRTPolyParams,
        exponents: &[usize],
    ) -> (WireId, crate::api::PlaintextInputs<WireId, DCRTPoly>) {
        let input = builder
            .input_family(IntExpr::constant(exponents.len()), AttributeKnowledge::PrivatePayload)
            .unwrap();
        let members = exponents.iter().map(|e| monomial(level, *e)).collect();
        (input, BTreeMap::from([(input, InputBinding::Family(members))]))
    }

    /// A family body multiplies member `X^(e_o)` by `c = 2 - X^2` shifted to
    /// `X^o`, and the members are summed: verification accepts
    /// `sum_o (2 - X^2) X^(e_o + o)`, including a negacyclic wrap, and
    /// rejects its successor.
    #[test]
    fn small_constant_multiplies_exponent_members_in_place() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let exponents = [1, 5, 14];
        let mut builder = ExponentLutProgramBuilder::new();
        let (input, lhs) = monomial_family(&mut builder, &level, &exponents);
        let placed = builder
            .parallel(
                crate::program::FamilyRange::full(IntExpr::constant(exponents.len())).unwrap(),
                |b, o| {
                    b.mul_small_constant(
                        Operand::Member { family: input, index: o.clone() },
                        vec![2, 0, -1],
                        o,
                    )
                    .map(Operand::Scalar)
                },
            )
            .unwrap();
        let output = builder.sum(placed).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let expected =
            exponents.iter().enumerate().fold(DCRTPoly::const_zero(&level), |sum, (o, e)| {
                let two = DCRTPoly::from_usize_to_constant(&level, 2);
                sum + two * monomial(&level, e + o) - monomial(&level, e + o + 2)
            });
        let mut execution =
            mxx_backends::authority::CpuExecution::new(cpu_backend([level.clone()]));
        let mut store = MemoryArtifactStore::default();
        let (result, payload, mask) =
            run_without_rhs(&parameters, &program, &lhs, &mut execution, &mut store);
        let verify = |expected: &DCRTPoly, execution: &mut _, store: &mut _| {
            verify_output(
                &program,
                &result,
                0,
                expected,
                &parameters,
                &payload,
                mask.as_ref(),
                execution,
                store,
            )
        };
        verify(&expected, &mut execution, &mut store).unwrap();
        let wrong = expected + DCRTPoly::const_one(&level);
        assert!(verify(&wrong, &mut execution, &mut store).is_err());
    }

    /// Coefficient LUT samples `T[e_o]` placed at `X^o` by a monomial small
    /// constant and summed: the finalized output decrypts to `T[e_o]` at
    /// coefficient `o`, under both RHS modes.
    #[test]
    fn small_constant_places_coefficient_samples() {
        for mode in [RhsMode::Static, RhsMode::Dynamic] {
            let parameters = unequal_width_parameters(mode);
            let level = parameters.levels()[0].clone();
            let exponents = [3, 9, 12];
            let values = (0..16).map(|e| e % 7).collect::<Vec<_>>();
            let mut builder = ExponentLutProgramBuilder::new();
            let (input, lhs) = monomial_family(&mut builder, &level, &exponents);
            let table = builder
                .register_lut(
                    LutTable::unary(
                        16,
                        ExpLayout { stride_log2: 0, width_log2: 4 },
                        7,
                        values.clone(),
                        crate::program::LutOutputRepresentation::Coefficient,
                    )
                    .unwrap(),
                )
                .unwrap();
            let placed = builder
                .parallel(
                    crate::program::FamilyRange::full(IntExpr::constant(exponents.len())).unwrap(),
                    |b, o| {
                        let sample =
                            b.lut(Operand::Member { family: input, index: o.clone() }, table)?;
                        b.mul_small_constant(Operand::Scalar(sample), vec![1], o)
                            .map(Operand::Scalar)
                    },
                )
                .unwrap();
            let output = builder.sum(placed).unwrap();
            let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
            let terminal = parameters.levels()[1].clone();
            let mut coefficients = vec![0; 16];
            for (o, e) in exponents.iter().enumerate() {
                coefficients[o] = values[*e] as u32;
            }
            let expected = DCRTPoly::from_u32s(&terminal, &coefficients);
            let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
                parameters.levels().iter().cloned(),
            ));
            let mut store = MemoryArtifactStore::default();
            let (result, payload, mask) =
                run_without_rhs(&parameters, &program, &lhs, &mut execution, &mut store);
            verify_output(
                &program,
                &result,
                0,
                &expected,
                &parameters,
                &payload,
                mask.as_ref(),
                &mut execution,
                &mut store,
            )
            .unwrap_or_else(|error| panic!("{mode:?}: {error:?}"));
            let decoded = decode_output(
                &program,
                &result,
                0,
                &parameters,
                &payload,
                mask.as_ref(),
                &[0, 1, 2, 3],
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let mut expected =
                exponents.iter().map(|e| BigInt::from(values[*e])).collect::<Vec<_>>();
            expected.push(BigInt::from(0u8));
            assert_eq!(decoded, expected, "{mode:?}");
        }
    }

    /// The simulated noise of `input * c` against that of `input`: a monomial
    /// leaves the bound unchanged and `3 - 2 X^2` multiplies it by five.
    #[test]
    fn small_constant_noise_is_exact_for_monomials() {
        let parameters = unequal_width_parameters(RhsMode::Static);
        let bound = |coefficients: Option<Vec<i64>>| {
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let output = match coefficients {
                Some(coefficients) => builder
                    .mul_small_constant(Operand::Scalar(input), coefficients, IntExpr::constant(3))
                    .unwrap(),
                None => input,
            };
            let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
            let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
            let bindings = crate::rns_bgg::RnsBggNoiseBindings::from_inputs(
                &program,
                &[crate::rns_bgg::RnsBggNoiseValue::sampled_encoding(
                    &parameters,
                    b"small-constant-input",
                    &[BigInt::from(1u8)],
                )],
                &[],
            )
            .unwrap();
            crate::api::noise_simulate(
                &program,
                crate::rns_bgg::RnsNoiseBuild::new(&parameters, &program, &workload, &bindings)
                    .unwrap(),
                |_, outputs| Ok(outputs),
            )
            .unwrap()
            .outputs[0]
                .bound
                .clone()
        };
        let identity = bound(None);
        assert_eq!(bound(Some(vec![0, -1])), identity);
        let scaled = bound(Some(vec![3, 0, -2]));
        assert_eq!(scaled.error, &identity.error * 5u8);
        assert_eq!(scaled.payload_l1, &identity.payload_l1 * 5u8);
    }
}
