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
    let (wire, output_level) = output_wire_level(program, output_index)?;
    verify_output_at_level(
        program,
        result,
        output_index,
        wire,
        output_level,
        expected,
        parameters,
        payload,
        mask,
        execution,
        store,
    )
}

/// The wire of scalar output `output_index` and the level of its finalized
/// carrier: a coefficient output is moved down once more by `finish_outputs`.
fn output_wire_level(
    program: &ExponentLutProgram,
    output_index: usize,
) -> Result<(WireId, usize), ApiError> {
    let wire = match program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("verification output index is out of range"))?
    {
        Operand::Scalar(wire) => *wire,
        Operand::Member { .. } => {
            return Err(ApiError::input("verification output must be scalar"));
        }
    };
    let workload = RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?;
    let source_level = workload.level(wire).map_err(ApiError::scheme)?;
    let coefficient_output = program
        .wire_is_coefficient_output(wire)
        .map_err(|_| ApiError::input("verification output metadata is invalid"))?;
    let output_level = source_level
        .checked_add(usize::from(coefficient_output))
        .ok_or_else(|| ApiError::input("verification output level overflow"))?;
    Ok((wire, output_level))
}

/// Decrypt private output `output_index` of [`crate::api::online_eval`] with
/// the caller's mask (for Dynamic parameters, the payload secret, which is
/// also the mask) and return its plaintext coefficients at `positions`,
/// modulo two.
///
/// A private output is a terminal-level carrier: the terminal state gadget
/// is the identity, its message scale is `floor(q / 2)`, and the second
/// component of the payload secret is one. Its second row component is
/// therefore `s * A_01 + A_11 - floor(q / 2) * mu + e` for the mask
/// `s = (s_0, 1)` and the paired public key `A`, so the phase
/// `row_1 - (s_0 * A_01 + A_11)` is `-floor(q / 2) * mu + e`. Threshold
/// decoding modulo two returns every coefficient of `mu` modulo two whenever
/// the error is below `q / 4`; the payload secret is not needed. The graph is
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
    let (wire, output_level) = output_wire_level(program, output_index)?;
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
    let decoded = phase.threshold_decode_ints(2u8, length);
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
    let (wire, output_level) = output_wire_level(program, output_index)?;
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

/// Verify one output after a declared RNS-BGG refresh.
///
/// A refresh changes the carrier modulus and public key, so the ordinary
/// Program workload level is no longer the level of the returned carrier.
/// The declaration supplies that target level while the rest of the
/// verification remains the same application-facing operation as
/// [`verify_output`].
pub fn verify_refreshed_output<E, S, P>(
    program: &ExponentLutProgram,
    declaration: &RefreshDeclaration,
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
    let wire = declaration.source_output();
    let output = program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("verification output index is out of range"))?;
    if !matches!(output, Operand::Scalar(actual) if *actual == wire) {
        return Err(ApiError::input("refreshed verification output does not match declaration"));
    }
    verify_output_at_level(
        program,
        result,
        output_index,
        wire,
        declaration.target_level(),
        expected,
        parameters,
        payload,
        mask,
        execution,
        store,
    )
}

fn verify_output_at_level<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    wire: WireId,
    output_level: usize,
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
    // Only the terminal relation (identity state gadget, public scale
    // floor(q / 2)) separates plaintexts by more than the q / 4 decoding
    // threshold; an intermediate carrier's gadget powers can be smaller than
    // its noise, so it cannot be verified by decoding.
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
    let encoding = RnsBggEncoding::new(key, row, public_payload);
    let residual = encoding
        .verification_residual(parameters, output_level, payload, mask, expected)
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
            &BigInt::from(2u8),
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
            refresh_prf_namespace: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
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
        let columns = parameters.component_columns(1).unwrap();
        assert_ne!(columns, parameters.component_columns(0).unwrap());
        let zero = source_ring.zero((1, columns));
        let key =
            RnsBggPublicKey::new(output, ((zero.clone(), zero.clone()), (zero.clone(), zero)));
        let target_gadget =
            parameters.state_gadget(1).unwrap().centered_rebase(&parameters.ring(0).unwrap());
        let scale = parameters.message_scale(0).unwrap() * parameters.message_scale(1).unwrap();
        let raw_signal = target_gadget * parameters.scalar(0, scale).unwrap();
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
            .verification_residual(&parameters, 0, &payload, &mask, expected.clone())
            .unwrap();
        let residual_b = RnsBggEncoding::new(key_b, row, None)
            .verification_residual(&parameters, 0, &payload, &mask, expected)
            .unwrap();
        assert_ne!(residual_a.flatten(), residual_b.flatten());
    }
}
