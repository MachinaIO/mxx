//! Noise traces, Program audits, parameter planning and search, and the
//! parameter manifest.
use super::*;

/// Noise trace with finalized declared outputs separated from raw wires.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RnsBggNoiseTrace {
    /// Accepted model, when an audit completed.
    pub acceptance: Option<NoiseModelKind>,
    /// Joint physical event count.
    pub joint_event_count: BigUint,
    /// Tail threshold for average-case acceptance.
    pub z_squared: Option<NonnegativeRational>,
    /// Dependency graph for shared sources.
    pub dependencies: crate::noise_utils::NoiseDependencyGraph,
    /// Raw symbolic wire values.
    pub wires: BTreeMap<WireId, RnsBggNoiseValue>,
    /// Physical event budget for each dependency node. A family body and LUT
    /// branch set are represented once in the symbolic trace, so their active
    /// multiplicities are expanded before this map is recorded. Decoder
    /// transcript nodes store their already-expanded physical event count.
    pub node_event_counts: BTreeMap<usize, BigUint>,
    /// Finalized declared outputs in declaration order.
    pub outputs: Vec<RnsBggNoiseValue>,
    /// The noise of the composed decoder, if any.
    pub decoder: Option<super::RnsBggDecoderNoise>,
}

/// Trace one Program using one workload and one evaluator path.
pub fn trace_program(
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
) -> Result<RnsBggNoiseTrace, ProgramLowerError<RnsBggError>> {
    let workload = RnsBggDerivedWorkload::from_program_with_parameters(parameters, program)
        .map_err(ProgramLowerError::Backend)?;
    let mut environment = Environment::new();
    for (id, declaration) in program.inputs() {
        let value = bindings.inputs.get(id).ok_or(ProgramLowerError::Backend(
            RnsBggError::Binding("missing input noise contract"),
        ))?;
        let lowered = match value.clone() {
            InputBinding::Scalar(value) => InputBinding::Scalar((*id, value)),
            InputBinding::Family(family) => InputBinding::Family((*id, family)),
        };
        environment.bind_wire(*id, lowered).map_err(ProgramLowerError::Program)?;
        if matches!(declaration.shape(), InputBinding::Family(_)) &&
            !matches!(value, InputBinding::Family(_))
        {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::ExpectedFamilyWire(*id),
            ));
        }
    }
    for (id, declaration) in program.rhs_inputs() {
        let value = bindings.rhs.get(id).ok_or(ProgramLowerError::Backend(
            RnsBggError::Binding("missing RHS noise contract"),
        ))?;
        let lowered = match value {
            InputBinding::Scalar(_) => InputBinding::Scalar(RhsRef { id: *id, index: None }),
            InputBinding::Family(family) => InputBinding::Family((*id, family.range.clone())),
        };
        environment.bind_rhs(*id, lowered).map_err(ProgramLowerError::Program)?;
        if matches!(declaration, InputBinding::Family(_)) &&
            !matches!(value, InputBinding::Family(_))
        {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::ExpectedFamilyRhs(*id),
            ));
        }
    }
    let mut evaluator = RnsBggNoiseEvaluator::new(parameters, program, &workload, bindings);
    for (wire, binding) in bindings.inputs.iter() {
        let value = match binding {
            InputBinding::Scalar(value) => value.clone(),
            InputBinding::Family(family) => family.element.clone(),
        };
        evaluator.trace.wires.insert(*wire, value);
        let node = crate::noise_utils::NoiseNodeId(evaluator.trace.dependencies.nodes.len());
        evaluator.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Binding(
            crate::noise_utils::NoiseBinding::Input(*wire),
        ));
        evaluator.trace.dependencies.wires.insert(*wire, node);
        let multiplicity = match binding {
            InputBinding::Scalar(_) => BigUint::one(),
            InputBinding::Family(family) => {
                family.validate_support().map_err(|error| ProgramLowerError::Backend(error))?
            }
        };
        let coordinates = evaluator.event_coordinates(*wire).map_err(ProgramLowerError::Backend)?;
        evaluator.trace.node_event_counts.insert(node.0, multiplicity * coordinates);
    }
    let mut families = RnsBggNoiseFamilies::default();
    let outputs =
        crate::program::lower_program(program, &mut evaluator, &mut families, &mut environment)?;
    evaluator.trace.outputs =
        evaluator.finish_outputs(outputs).map_err(ProgramLowerError::Backend)?;
    Ok(evaluator.trace)
}

/// Public parameter-search request. LUT widths are read from the Program.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggPlanRequest {
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Fresh encoding sigma.
    #[serde(with = "crate::noise_utils::bigdecimal_string")]
    pub encoding_error_sigma: bigdecimal::BigDecimal,
    /// Target CRT prime bits.
    pub crt_bits: usize,
    /// Gadget decomposition base bits, at most 31.
    pub base_bits: u32,
    /// Terminal CRT depth.
    pub terminal_depth: usize,
    /// Maximum candidate CRT depth.
    pub max_crt_depth: usize,
    /// Declared terminal value bound.
    pub terminal_value_bound: BigUint,
    /// Logical operations per physical drop.
    pub operations_per_drop: usize,
    /// CRT primes removed by each physical drop.
    pub primes_per_drop: usize,
    /// RHS timing mode of the planned parameters.
    #[serde(default)]
    pub rhs_mode: RhsMode,
}

/// Serialized parameter-search evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggParameterManifest {
    /// Original request.
    pub request: RnsBggPlanRequest,
    /// Frozen Program identity.
    pub program: crate::program::ExponentLutProgramId,
    /// Commitment to public noise contracts.
    pub noise_contract_commitment: [u8; 32],
    /// Commitment to the selected parameter bases and request.
    pub parameter_commitment: [u8; 32],
    /// Actual selected CRT prime bases.
    pub levels: Vec<Vec<u64>>,
    /// Base widths for each level.
    pub base_bits: Vec<u32>,
    /// Accepted noise trace.
    pub trace: RnsBggNoiseTrace,
    /// Average-case policy used when this trace was accepted.
    pub average: Option<crate::decoder::simulation::AverageCaseConfig>,
    /// Estimated persistent artifact bytes.
    pub estimated_artifact_bytes: BigUint,
    /// Commitment to the complete accepted trace, including dependencies.
    pub trace_commitment: [u8; 32],
}

fn parameter_artifact_bytes(
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
) -> Result<BigUint, RnsBggError> {
    let polynomial_bytes = |level: usize| -> Result<BigUint, RnsBggError> {
        let level = parameters.level(level)?;
        Ok(BigUint::from(level.ring_dimension()) * level.modulus().bits().div_ceil(8))
    };
    let mut bytes = (0..parameters.levels().len())
        .map(|level| Ok(polynomial_bytes(level)? * 16u8))
        .sum::<Result<BigUint, RnsBggError>>()?;
    if parameters.rhs_mode() == RhsMode::Static {
        return Ok(bytes);
    }
    // Dynamic encryption publishes, per RHS member and LUT branch consumed at
    // level i, the full ciphertext (2 m_(i+1) polynomials at q_(i+1)) and its
    // N_i = m_i m_(i+1) digit companions of M_i polynomials at q_i (spec
    // Eq. 066).
    let workload = RnsBggDerivedWorkload::from_program(program)?;
    // `output_columns` is the ciphertext's per-component width: the RHS
    // state width, or a LUT's switching-target width.
    let material =
        |level: usize, output_columns: usize, count: usize| -> Result<BigUint, RnsBggError> {
            let m = parameters.component_columns(level)? * 2;
            let m_next = output_columns * 2;
            let public_columns = parameters.level(level)?.modulus_digits() * 2;
            let ciphertext = polynomial_bytes(level + 1)? * m_next * 2u8;
            let companions = polynomial_bytes(level)? * m * m_next * public_columns;
            Ok((ciphertext + companions) * count)
        };
    for (rhs, shape) in program.rhs_inputs() {
        let count = match shape {
            InputBinding::Scalar(()) => 1,
            InputBinding::Family(count) => count
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|count| count.to_usize())
                .ok_or(RnsBggError::Binding("RHS family count is not concrete"))?,
        };
        let level = workload.rhs_level(*rhs)?;
        bytes += material(level, parameters.component_columns(level + 1)?, count)?;
    }
    let tables = program.encryption_inputs();
    for (level, lut) in workload.luts() {
        let table = tables.luts().get(lut.0 as usize).ok_or(RnsBggError::Lut)?;
        let columns = parameters.lut_output_columns(*level + 1, table.output_representation())?;
        bytes += material(*level, columns, table.input_width())?;
    }
    Ok(bytes)
}

fn noise_contract_commitment(
    program: &ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    terminal_value_bounds: &[BigUint],
    average: Option<&crate::decoder::simulation::AverageCaseConfig>,
) -> [u8; 32] {
    stable_source_id(&(
        program.id(),
        &bindings.inputs,
        &bindings.rhs,
        terminal_value_bounds,
        average,
    ))
}

fn parameter_commitment(
    request: &RnsBggPlanRequest,
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    noise_contract_commitment: [u8; 32],
) -> [u8; 32] {
    stable_source_id(&(
        request,
        program.id(),
        parameters.levels().iter().map(|level| level.to_crt().0).collect::<Vec<_>>(),
        parameters.levels().iter().map(PolyParams::base_bits).collect::<Vec<_>>(),
        noise_contract_commitment,
    ))
}

fn trace_commitment(trace: &RnsBggNoiseTrace) -> [u8; 32] {
    stable_source_id(&(
        &trace.acceptance,
        &trace.z_squared,
        &trace.dependencies,
        &trace.wires,
        &trace.node_event_counts,
        &trace.outputs,
        &trace.joint_event_count,
        &trace.decoder,
    ))
}

fn average_event_budget(trace: &RnsBggNoiseTrace) -> Result<BigUint, RnsBggError> {
    let events = trace.node_event_counts.values().sum::<BigUint>();
    Ok(events.max(BigUint::one()))
}

impl RnsBggParameters {
    fn construct_program_parameters(
        request: &RnsBggPlanRequest,
        program: &ExponentLutProgram,
    ) -> Result<Self, RnsBggError> {
        if request.operations_per_drop == 0 ||
            request.primes_per_drop == 0 ||
            request.terminal_depth == 0 ||
            request.max_crt_depth < request.terminal_depth
        {
            return Err(RnsBggError::Parameters("invalid public parameter request"));
        }
        let workload = RnsBggDerivedWorkload::from_program(program)?;
        let logical_count = workload.levels.values().copied().max().unwrap_or(0);
        let pool = DCRTPolyParams::try_new(
            request.ring_dimension,
            request.max_crt_depth,
            request.crt_bits,
            request.base_bits,
            None,
            None,
        )
        .map_err(|_| RnsBggError::Parameters("unsupported prime pool"))?;
        let primes = pool.to_crt().0;
        let mut levels = Vec::with_capacity(logical_count + 1);
        for index in 0..=logical_count {
            let depth = super::super::planned_crt_depth(
                index,
                request.operations_per_drop,
                request.primes_per_drop,
                request.max_crt_depth,
                request.terminal_depth,
            )?;
            let modulus = primes[..depth].iter().copied().map(BigUint::from).product();
            levels.push(
                pool.select_modulus(&modulus)
                    .ok_or(RnsBggError::Parameters("invalid prime basis"))?,
            );
        }
        Ok(RnsBggParameters::new(
            levels,
            request.encoding_error_sigma.clone(),
            request.operations_per_drop,
        )?
        .with_rhs_mode(request.rhs_mode))
    }

    /// Evaluate the selected Program noise contract and terminal envelopes.
    pub fn audit_program(
        &self,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
        terminal_value_bounds: &[BigUint],
        average: Option<&crate::decoder::simulation::AverageCaseConfig>,
    ) -> Result<RnsBggNoiseTrace, RnsBggError> {
        if terminal_value_bounds.len() != program.outputs().len() {
            return Err(RnsBggError::Parameters("terminal bounds must match declared outputs"));
        }
        let started = std::time::Instant::now();
        let mut trace = trace_program(self, program, bindings).map_err(|error| match error {
            ProgramLowerError::Backend(error) => error,
            ProgramLowerError::Program(error) => RnsBggError::Graph(error.to_string()),
            ProgramLowerError::Dsl(error) => RnsBggError::Graph(error.to_string()),
            ProgramLowerError::UnsupportedStructure { reason } => RnsBggError::Graph(reason),
        })?;
        tracing::info!(
            wires = trace.wires.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "noise trace finished"
        );
        let workload = RnsBggDerivedWorkload::from_program_with_parameters(self, program)?;
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "noise audit workload derived"
        );
        // Average-case acceptance is the caller's explicit choice. For
        // Dynamic parameters it is a heuristic: the digit companions' moments
        // assume the uncorrelated-error model (H1/H2) of the Static terms,
        // without RHS independence (H4); the deterministic bound remains the
        // conservative check.
        let average = average.filter(|config| config.allow_average_acceptance);
        let z_squared = average
            .map(|config| {
                let events = average_event_budget(&trace)?;
                let z_squared = config
                    .z_squared_for_events(&events)
                    .map_err(|_| RnsBggError::Parameters("invalid average-case event count"))?;
                let correction = config
                    .tail_correction_bits
                    .checked_mul(2)
                    .ok_or(RnsBggError::Parameters("invalid average-case tail correction"))?;
                Ok::<_, RnsBggError>(z_squared.scaled(&(BigUint::one() << correction)))
            })
            .transpose()?;

        if let Some(z_squared) = z_squared.as_ref() {
            let mut values = BTreeMap::<usize, (WireId, &RnsBggNoiseValue)>::new();
            for (wire, value) in &trace.wires {
                let node =
                    trace.dependencies.wires.get(wire).ok_or_else(|| {
                        RnsBggError::Graph("noise dependency node is missing".into())
                    })?;
                values.entry(node.0).or_insert((*wire, value));
            }
            for (wire, value) in values.values() {
                let moments = value
                    .moments
                    .as_ref()
                    .ok_or(RnsBggError::Parameters("average-case input moment model is missing"))?;
                let level = workload.level(*wire)?;
                let modulus_limit = NonnegativeRational::new(
                    self.level(level)?.modulus().as_ref().clone(),
                    4u8.into(),
                )
                .map_err(|_| RnsBggError::Parameters("invalid average-case rational budget"))?;
                // Only the q/4 threshold applies. ModDown divides the noise
                // and the modulus by the same drop, so a wire over this limit
                // also fails the terminal decode; noise above half the drop
                // only remains as tracked noise at the next level.
                if !moments.fits(&modulus_limit, z_squared) {
                    tracing::info!(
                        wire = wire.0,
                        level,
                        levels = self.levels.len(),
                        variance_log2 = moments.variance.numerator.bits() as i64 -
                            moments.variance.denominator.bits() as i64,
                        modulus_bits = self.level(level)?.modulus().bits(),
                        scale_bits = self.scale(level)?.bits(),
                        z_squared_log2 = variance_log2(z_squared),
                        top_modulus_bits = self.levels[0].modulus().bits(),
                        first_scale_bits = self.scale(0)?.bits(),
                        contributions = %value.describe_contributions(),
                        "average-case evaluation noise exceeds the level budget"
                    );
                    return Err(RnsBggError::Budget("evaluation noise"));
                }
            }
        }

        for ((index, output), terminal_bound) in
            program.outputs().iter().enumerate().zip(terminal_value_bounds)
        {
            let wire = match output {
                crate::program::Operand::Scalar(id) |
                crate::program::Operand::Member { family: id, .. } => *id,
            };
            let value = trace
                .outputs
                .get(index)
                .ok_or(RnsBggError::Graph("missing audited output".into()))?;
            if let Some(z_squared) = z_squared.as_ref() {
                let moments = value
                    .moments
                    .as_ref()
                    .ok_or(RnsBggError::Parameters("average-case input moment model is missing"))?;
                let level = workload.level(wire)?;
                let coefficient = program
                    .wire_is_coefficient_output(wire)
                    .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?;
                let modulus = if coefficient {
                    self.levels
                        .get(level + 1)
                        .or_else(|| self.levels.get(level))
                        .ok_or(RnsBggError::Parameters("level outside modulus chain"))?
                        .modulus()
                } else {
                    self.level(level)?.modulus()
                };
                let payload = terminal_bound * 2u8;
                if modulus.as_ref() <= &payload {
                    return Err(RnsBggError::Budget("terminal range"));
                }
                let modulus_limit = if coefficient {
                    self.coefficient_output_budget(program, wire, level)?
                } else {
                    NonnegativeRational::new(modulus.as_ref().clone(), 4u8.into()).map_err(
                        |_| RnsBggError::Parameters("invalid average-case rational budget"),
                    )?
                };
                if !moments.fits(&modulus_limit, z_squared) {
                    tracing::info!(
                        wire = wire.0,
                        level,
                        variance_log2 = variance_log2(&moments.variance),
                        modulus_bits = modulus.bits(),
                        contributions = %value.describe_contributions(),
                        "average-case terminal noise exceeds the output budget"
                    );
                    return Err(RnsBggError::Budget("terminal range"));
                }
            } else {
                let level = workload.level(wire)?;
                let fits = if program
                    .wire_is_coefficient_output(wire)
                    .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?
                {
                    let budget = self.coefficient_output_budget(program, wire, level)?;
                    &value.bound.error * &budget.denominator < budget.numerator
                } else {
                    self.noise_fits(level, &value.bound)?
                };
                if !fits {
                    return Err(RnsBggError::Budget(
                        "evaluation noise does not fit the selected modulus",
                    ));
                }
            }
            if &value.bound.payload_l1 > terminal_bound {
                return Err(RnsBggError::Budget("terminal payload exceeds declared bound"));
            }
        }
        trace.acceptance = Some(if z_squared.is_some() {
            NoiseModelKind::AverageCase
        } else {
            NoiseModelKind::WorstCase
        });
        trace.z_squared = z_squared;
        Ok(trace)
    }

    /// Construct a deterministic chain from the Program's actual logical levels.
    pub fn plan_program(
        request: &RnsBggPlanRequest,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
    ) -> Result<(Self, RnsBggParameterManifest), RnsBggError> {
        let parameters = Self::construct_program_parameters(request, program)?;
        let trace = parameters.audit_program(
            program,
            bindings,
            &vec![request.terminal_value_bound.clone(); program.outputs().len()],
            None,
        )?;
        let terminal_value_bounds =
            vec![request.terminal_value_bound.clone(); program.outputs().len()];
        let noise_contract_commitment =
            noise_contract_commitment(program, bindings, &terminal_value_bounds, None);
        let parameter_commitment =
            parameter_commitment(request, &parameters, program, noise_contract_commitment);
        let estimated_artifact_bytes = parameter_artifact_bytes(&parameters, program)?;
        let trace_digest = trace_commitment(&trace);
        let manifest = RnsBggParameterManifest {
            request: request.clone(),
            program: program.id(),
            noise_contract_commitment,
            parameter_commitment,
            levels: parameters.levels().iter().map(|level| level.to_crt().0).collect(),
            base_bits: parameters.levels().iter().map(PolyParams::base_bits).collect(),
            trace,
            average: None,
            estimated_artifact_bytes,
            trace_commitment: trace_digest,
        };
        Ok((parameters, manifest))
    }

    /// Search candidates, preserving the exact evaluator/audit path.
    pub fn search_program(
        request: &RnsBggPlanRequest,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
        terminal_value_bounds: &[BigUint],
        average: Option<&crate::decoder::simulation::AverageCaseConfig>,
    ) -> Result<(Self, RnsBggParameterManifest), RnsBggError> {
        if terminal_value_bounds.is_empty() {
            return Err(RnsBggError::Parameters("terminal bounds must not be empty"));
        }
        let trial_bound = terminal_value_bounds
            .iter()
            .max()
            .cloned()
            .ok_or(RnsBggError::Parameters("terminal bounds must not be empty"))?;
        let mut last_error = RnsBggError::Budget("no CRT candidate satisfies the noise contract");
        for depth in request.terminal_depth..=request.max_crt_depth {
            let mut candidate = request.clone();
            candidate.max_crt_depth = depth;
            candidate.terminal_value_bound = trial_bound.clone();
            let Ok(parameters) = Self::construct_program_parameters(&candidate, program) else {
                continue;
            };
            match parameters.audit_program(program, bindings, terminal_value_bounds, average) {
                Ok(trace) => {
                    let noise_contract_commitment = noise_contract_commitment(
                        program,
                        bindings,
                        terminal_value_bounds,
                        average,
                    );
                    let manifest = RnsBggParameterManifest {
                        request: candidate.clone(),
                        program: program.id(),
                        noise_contract_commitment,
                        parameter_commitment: parameter_commitment(
                            &candidate,
                            &parameters,
                            program,
                            noise_contract_commitment,
                        ),
                        levels: parameters.levels().iter().map(|level| level.to_crt().0).collect(),
                        base_bits: parameters.levels().iter().map(PolyParams::base_bits).collect(),
                        trace_commitment: trace_commitment(&trace),
                        trace,
                        average: average.cloned(),
                        estimated_artifact_bytes: parameter_artifact_bytes(&parameters, program)?,
                    };
                    return Ok((parameters, manifest));
                }
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }
}

impl RnsBggParameterManifest {
    /// Rehydrate and validate a manifest against the frozen Program and
    /// contracts.  No serialized level or commitment is trusted on its own.
    pub fn validate(
        &self,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
        terminal_value_bounds: &[BigUint],
    ) -> Result<RnsBggParameters, RnsBggError> {
        if self.program != program.id() ||
            self.levels.is_empty() ||
            self.levels.len() != self.base_bits.len()
        {
            return Err(RnsBggError::Binding("parameter manifest does not match the Program"));
        }
        let commitment = noise_contract_commitment(
            program,
            bindings,
            terminal_value_bounds,
            self.average.as_ref(),
        );
        if commitment != self.noise_contract_commitment {
            return Err(RnsBggError::Binding("noise contract commitment mismatch"));
        }
        let pool = DCRTPolyParams::try_new(
            self.request.ring_dimension,
            self.request.max_crt_depth,
            self.request.crt_bits,
            self.request.base_bits,
            None,
            None,
        )
        .map_err(|_| RnsBggError::Parameters("unsupported manifest prime pool"))?;
        let mut levels = Vec::with_capacity(self.levels.len());
        for (primes, base_bits) in self.levels.iter().zip(&self.base_bits) {
            if primes.is_empty() || primes.iter().any(|prime| *prime < 3) {
                return Err(RnsBggError::Binding("invalid manifest CRT basis"));
            }
            let modulus = primes.iter().copied().map(BigUint::from).product();
            let level = pool
                .select_modulus(&modulus)
                .ok_or(RnsBggError::Binding("manifest CRT basis is not in the prime pool"))?;
            if level.base_bits() != *base_bits {
                return Err(RnsBggError::Binding("manifest base width mismatch"));
            }
            levels.push(level);
        }
        let parameters = RnsBggParameters::new(
            levels,
            self.request.encoding_error_sigma.clone(),
            self.request.operations_per_drop,
        )?
        .with_rhs_mode(self.request.rhs_mode);
        let expected_parameter_commitment = parameter_commitment(
            &self.request,
            &parameters,
            program,
            self.noise_contract_commitment,
        );
        if expected_parameter_commitment != self.parameter_commitment {
            return Err(RnsBggError::Binding("parameter commitment mismatch"));
        }
        if parameter_artifact_bytes(&parameters, program)? != self.estimated_artifact_bytes {
            return Err(RnsBggError::Binding("artifact size estimate mismatch"));
        }
        let trace = parameters.audit_program(
            program,
            bindings,
            terminal_value_bounds,
            self.average.as_ref(),
        )?;
        if trace_commitment(&trace) != self.trace_commitment {
            return Err(RnsBggError::Binding("manifest noise trace mismatch"));
        }
        Ok(parameters)
    }
}

#[cfg(test)]
mod test {
    use super::{super::test::nested_alignment_parameters, *};

    fn identity_program() -> (ExponentLutProgram, RnsBggParameters) {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(input)]).unwrap();
        (program, nested_alignment_parameters())
    }

    #[test]
    fn average_case_can_accept_a_candidate_rejected_by_worst_case() {
        let (program, parameters) = identity_program();
        let mut input =
            RnsBggNoiseValue::sampled_encoding(&parameters, b"average-input", &[1.into()]);
        input.bound.error = parameters.level(0).unwrap().modulus().as_ref().clone() * 2u8;
        let bindings = RnsBggNoiseBindings::from_inputs(&program, &[input], &[]).unwrap();
        let terminal = [BigUint::one()];
        assert!(parameters.audit_program(&program, &bindings, &terminal, None).is_err());
        let average = crate::decoder::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        let trace =
            parameters.audit_program(&program, &bindings, &terminal, Some(&average)).unwrap();
        assert_eq!(trace.acceptance, Some(NoiseModelKind::AverageCase));
    }

    /// Dynamic parameters may be accepted by the average case when the
    /// caller asks for it, as Static ones can.
    #[test]
    fn dynamic_audit_accepts_the_average_case_on_request() {
        let (program, parameters) = identity_program();
        let parameters = parameters.with_rhs_mode(RhsMode::Dynamic);
        let mut input =
            RnsBggNoiseValue::sampled_encoding(&parameters, b"average-input", &[1.into()]);
        input.bound.error = parameters.level(0).unwrap().modulus().as_ref().clone() * 2u8;
        let bindings = RnsBggNoiseBindings::from_inputs(&program, &[input], &[]).unwrap();
        let terminal = [BigUint::one()];
        assert!(parameters.audit_program(&program, &bindings, &terminal, None).is_err());
        let average = crate::decoder::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        let trace =
            parameters.audit_program(&program, &bindings, &terminal, Some(&average)).unwrap();
        assert_eq!(trace.acceptance, Some(NoiseModelKind::AverageCase));
    }

    #[test]
    fn average_case_rejects_missing_input_moments() {
        let (program, parameters) = identity_program();
        let bindings = RnsBggNoiseBindings::from_inputs(
            &program,
            &[RnsBggNoiseValue::public(BigUint::one(), &[1.into()])],
            &[],
        )
        .unwrap();
        let average = crate::decoder::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        assert!(matches!(
            parameters.audit_program(&program, &bindings, &[BigUint::one()], Some(&average)),
            Err(RnsBggError::Parameters("average-case input moment model is missing"))
        ));
    }

    #[test]
    fn average_event_budget_deduplicates_nodes_and_applies_tail_correction() {
        let parameters = nested_alignment_parameters();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let program = builder
            .finish(vec![
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(input),
            ])
            .unwrap();
        let input = RnsBggNoiseValue::sampled_encoding(&parameters, b"budget-input", &[1.into()]);
        let bindings = RnsBggNoiseBindings::from_inputs(&program, &[input], &[]).unwrap();
        let trace = trace_program(&parameters, &program, &bindings).unwrap();
        let events = average_event_budget(&trace).unwrap();
        let expected = BigUint::from(parameters.level(0).unwrap().ring_dimension()) *
            2u8 *
            BigUint::from(parameters.component_columns(0).unwrap());
        assert_eq!(events, expected);
        let base = crate::decoder::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        }
        .z_squared_for_events(&events)
        .unwrap();
        let corrected = crate::decoder::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 2,
            allow_average_acceptance: true,
        }
        .z_squared_for_events(&events)
        .unwrap()
        .scaled(&(BigUint::one() << 4));
        assert!(corrected.numerator > base.numerator);
    }
}
