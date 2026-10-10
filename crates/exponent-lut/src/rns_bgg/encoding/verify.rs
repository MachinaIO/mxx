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
/// `output_use` is the use the Program was preprocessed with; it selects the
/// carrier (see [`RnsBggParameters::output_carrier`]) whose relation is
/// checked.
#[allow(clippy::too_many_arguments)]
pub fn verify_output<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    expected: &P,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
    output_use: RnsOutputUse,
    execution: &mut E,
    store: &mut S,
) -> Result<(), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let (wire, carrier) = output_wire_carrier(program, parameters, output_index, output_use)?;
    verify_output_carrier(
        program,
        result,
        output_index,
        wire,
        &carrier,
        expected,
        parameters,
        payload,
        mask,
        execution,
        store,
    )
}

/// The wire of output `output_index` and its resolved carrier under
/// `output_use`. A member of a `parallel` family is described by its body's
/// output wire, a member of an input family by the family's own wire.
fn output_wire_carrier(
    program: &ExponentLutProgram,
    parameters: &RnsBggParameters,
    output_index: usize,
    output_use: RnsOutputUse,
) -> Result<(WireId, RnsOutputCarrier), ApiError> {
    let wire = match program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("verification output index is out of range"))?
    {
        Operand::Scalar(wire) => *wire,
        Operand::Member { family, .. } | Operand::Selected { family, .. } => program
            .family_body_outputs()
            .into_iter()
            .find_map(|(output, inner)| (output == *family).then_some(inner))
            .unwrap_or(*family),
    };
    let workload = RnsBggDerivedWorkload::from_program_with_parameters(parameters, program)
        .map_err(ApiError::scheme)?;
    let level = workload.level(wire).map_err(ApiError::scheme)?;
    let carrier =
        parameters.output_carrier(program, wire, level, output_use).map_err(ApiError::scheme)?;
    Ok((wire, carrier))
}

/// Decrypt private output `output_index` of [`crate::api::online_eval`] with
/// the caller's mask (for Dynamic parameters, the payload secret, which is
/// also the mask) and return its plaintext coefficients at `positions`:
/// modulo two for an exponent output, modulo its decryption modulus `t` for
/// a coefficient output.
///
/// A decodable private output carrier (see
/// [`RnsBggParameters::output_carrier`] under `output_use`) is a scalar
/// column per component: its message scale is `Delta` (`floor(q / 2)` for an
/// exponent output, the scale `k` of
/// [`RnsBggParameters::coefficient_output_scale`] for a coefficient output),
/// and the second component of the payload secret is one. Its second row component is therefore
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
    output_use: RnsOutputUse,
    positions: &[usize],
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
) -> Result<Vec<BigInt>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let (wire, carrier) = output_wire_carrier(program, parameters, output_index, output_use)?;
    let Some(plaintext_modulus) = carrier.plaintext_modulus else {
        return Err(ApiError::input("the output carrier is not directly decodable"));
    };
    let output_level = carrier.level;
    if parameters.rhs_mode() == RhsMode::Static && output_level + 1 != parameters.levels().len() {
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
    // A coefficient carrier is decoded from `k mu - e`; an exponent carrier's
    // phase `-floor(q/2) mu + e` decodes modulo two either way.
    let decoded = if program.wire_is_coefficient_output(wire).map_err(ApiError::scheme)? {
        (-phase).threshold_decode_ints(plaintext_modulus, length)
    } else {
        phase.threshold_decode_ints(plaintext_modulus, length)
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
    let (wire, carrier) =
        output_wire_carrier(program, parameters, output_index, RnsOutputUse::DirectDecode)?;
    let output_level = carrier.level;
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

#[allow(clippy::too_many_arguments)]
fn verify_output_carrier<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    wire: WireId,
    carrier: &RnsOutputCarrier,
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
    let output_level = carrier.level;
    let knowledge = program.wire_knowledge(wire).map_err(ApiError::scheme)?;
    let private_output = matches!(knowledge, AttributeKnowledge::PrivatePayload);
    // A carrier separates plaintexts by more than the decoding threshold when
    // it is decodable (a scalar column with scale `floor(q / 2)` or `k`) or a
    // full gadget of scale one, whose last gadget column separates them by
    // about `q / 2`. A Static carrier below the terminal level has a scaled
    // state gadget whose powers can be smaller than its noise.
    let full_gadget = carrier.gadget != RnsOutputGadget::Scalar && carrier.scale.is_one();
    if private_output && carrier.plaintext_modulus.is_none() && !full_gadget {
        return Err(ApiError::input(format!(
            "verification output at level {output_level} is neither decodable nor a full gadget"
        )));
    }
    // Static carriers are verified at the terminal level only, where its
    // identity state gadget and public scale apply; a Dynamic chain keeps a
    // gadget successor, and its decodable or full-gadget carriers below the
    // terminal level have scale one or an explicit decoding scale.
    if (!private_output || parameters.rhs_mode() == RhsMode::Static) &&
        output_level + 1 != parameters.levels().len()
    {
        return Err(ApiError::input(format!(
            "verification output is at level {output_level}, not the terminal level {}",
            parameters.levels().len() - 1
        )));
    }
    let output_ring = parameters.ring(output_level).map_err(ApiError::scheme)?;
    let columns = if private_output {
        carrier.columns
    } else {
        parameters.key_component_columns(output_level, knowledge).map_err(ApiError::scheme)?
    };
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

    // The carrier obeys the relation of its resolved level, scale, and
    // gadget: verify that relation directly; extending a normalized carrier
    // cannot recover its pre-ModDown representative.
    let expected = plaintext_constant(&output_ring, expected);
    let public_payload_residual =
        public_payload.as_ref().map(|payload| payload.clone() - expected.clone());
    let plaintext_modulus = BigInt::from(carrier.plaintext_modulus.unwrap_or(2));
    let encoding = RnsBggEncoding::new(key, row, public_payload);
    let residual = encoding
        .verification_residual(parameters, carrier, payload, mask, expected)
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
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            output_use: RnsOutputUse::default(),
            dynamic: None,
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
                RnsOutputUse::DirectDecode,
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
                    RnsOutputUse::DirectDecode,
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
                RnsOutputUse::DirectDecode,
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
        let carrier = RnsOutputCarrier {
            level: 0,
            columns: 1,
            scale: parameters.message_scale(0).unwrap(),
            gadget: RnsOutputGadget::Scalar,
            action: None,
            plaintext_modulus: Some(2),
        };
        let residual_a = RnsBggEncoding::new(key_a, row.clone(), None)
            .verification_residual(&parameters, &carrier, &payload, &mask, expected.clone())
            .unwrap();
        let residual_b = RnsBggEncoding::new(key_b, row, None)
            .verification_residual(&parameters, &carrier, &payload, &mask, expected)
            .unwrap();
        assert_ne!(residual_a.flatten(), residual_b.flatten());
    }

    /// Two levels of two and one 17-bit primes, so the switching target of a
    /// LUT at level 0 is wider for a coefficient table (`modulus_digits(1)`)
    /// than for an exponent one (`component_columns(1)`, one column at the
    /// terminal level). Ring dimension and sigma are overridable by
    /// `MXX_EXPONENT_LUT_COEFFICIENT_TEST_{RING_DIMENSION,SIGMA}`.
    fn unequal_width_parameters() -> RnsBggParameters {
        let read = |name: &str, default: &str| {
            std::env::var(format!("MXX_EXPONENT_LUT_COEFFICIENT_TEST_{name}"))
                .unwrap_or_else(|_| default.to_owned())
        };
        let ring_dimension = read("RING_DIMENSION", "16").parse().unwrap();
        let sigma = read("SIGMA", "1").parse::<bigdecimal::BigDecimal>().unwrap();
        let high = DCRTPolyParams::new(ring_dimension, 2, 17, 6, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        RnsBggParameters::new(vec![high, low], sigma, 1).unwrap()
    }

    /// Preprocess, encrypt and evaluate `program` with the private scalar
    /// input bound to `input` under Static parameters, and return the online
    /// result with the payload secret and the stage mask. The online
    /// evaluation receives the Program without its LUT values.
    fn run_without_rhs(
        parameters: &RnsBggParameters,
        program: &ExponentLutProgram,
        lhs: &crate::api::PlaintextInputs<WireId, DCRTPoly>,
        execution: &mut mxx_backends::authority::CpuExecution,
        store: &mut MemoryArtifactStore,
    ) -> (StageOutputs, RnsBggPayloadSecret, RnsBggMaskSecret) {
        let hash_key: [u8; 32] = rand::random();
        let context: [u8; 32] = rand::random();
        let payload = RnsEncryptBuild::new(parameters, context)
            .sample_payload_secret(execution, store, |_| Ok(()))
            .unwrap();
        let (preprocess, online_eval) = crate::api::preprocess(
            program,
            &ParamEnv::default(),
            &crate::api::PlaintextInputs::<RhsId, DCRTPoly>::new(),
            Some(&payload),
            &hash_key,
            crate::rns_bgg::RnsPreprocessBuild::new(parameters, context),
            execution,
            store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let mask = RnsEncryptBuild::new(parameters, rand::random())
            .sample_mask_secret(execution, store, |_| Ok(()))
            .unwrap();
        let encrypted = crate::api::encrypt(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            lhs,
            Some(&preprocess),
            &payload,
            Some(&mask),
            &hash_key,
            RnsEncryptBuild::new(parameters, context),
            execution,
            store,
            |_| Ok(()),
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
    /// decryption returns `v mod t`, in Static mode (the Dynamic coefficient
    /// output, on an equal-basis chain, is checked with the Dynamic stages),
    /// for both table entries (the larger one is the bound `W`) and both
    /// signs of the input monomial.
    #[test]
    fn coefficient_lut_finalizes_scales_and_decodes() {
        let mode = RhsMode::Static;
        for (bit, negative) in [(0, false), (1, false), (0, true), (1, true)] {
            let parameters = unequal_width_parameters();
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
            let stage_mask = stage_mask(&parameters, &payload, Some(&mask)).unwrap();
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
                    Some(&mask),
                    RnsOutputUse::DirectDecode,
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
                Some(&mask),
                RnsOutputUse::DirectDecode,
                &[0],
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(decoded, vec![BigInt::from(value.rem_euclid(t as i64))], "{mode:?}");
        }
    }
}
