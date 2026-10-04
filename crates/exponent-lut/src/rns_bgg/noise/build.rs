//! The noise stage adapter.
use super::*;

impl<'a> RnsNoiseBuild<'a> {
    /// Construct a noise stage from the already validated workload.
    pub fn new(
        parameters: &'a RnsBggParameters,
        program: &'a ExponentLutProgram,
        workload: &'a RnsBggDerivedWorkload,
        bindings: &'a RnsBggNoiseBindings,
    ) -> Result<Self, RnsBggError> {
        let mut environment = Environment::new();
        for (id, declaration) in program.inputs() {
            let value = bindings
                .inputs
                .get(id)
                .ok_or(RnsBggError::Binding("missing input noise contract"))?
                .clone();
            environment
                .bind_wire(
                    *id,
                    match (declaration.shape(), value) {
                        (InputBinding::Scalar(()), InputBinding::Scalar(value)) => {
                            InputBinding::Scalar((*id, value))
                        }
                        (InputBinding::Family(_), InputBinding::Family(value)) => {
                            InputBinding::Family((*id, value))
                        }
                        _ => return Err(RnsBggError::Binding("input binding shape mismatch")),
                    },
                )
                .map_err(|_| RnsBggError::Binding("invalid input binding"))?;
        }
        for (id, declaration) in program.rhs_inputs() {
            let value = bindings
                .rhs
                .get(id)
                .ok_or(RnsBggError::Binding("missing RHS noise contract"))?
                .clone();
            environment
                .bind_rhs(
                    *id,
                    match (declaration, value) {
                        (InputBinding::Scalar(()), InputBinding::Scalar(_)) => {
                            InputBinding::Scalar(RhsRef { id: *id, index: None })
                        }
                        (InputBinding::Family(_), InputBinding::Family(family)) => {
                            InputBinding::Family((*id, family.range.clone()))
                        }
                        _ => return Err(RnsBggError::Binding("RHS binding shape mismatch")),
                    },
                )
                .map_err(|_| RnsBggError::Binding("invalid RHS binding"))?;
        }
        let mut evaluator = RnsBggNoiseEvaluator::new(parameters, program, workload, bindings);
        for (wire, binding) in &bindings.inputs {
            let value = match binding {
                InputBinding::Scalar(value) => value.clone(),
                InputBinding::Family(family) => family.element.clone(),
            };
            evaluator.trace.wires.insert(*wire, value);
            let node = crate::noise_utils::NoiseNodeId(evaluator.trace.dependencies.nodes.len());
            evaluator.trace.dependencies.nodes.push(
                crate::noise_utils::NoiseDependencyNode::Binding(
                    crate::noise_utils::NoiseBinding::Input(*wire),
                ),
            );
            evaluator.trace.dependencies.wires.insert(*wire, node);
            let multiplicity = match binding {
                InputBinding::Scalar(_) => BigUint::one(),
                InputBinding::Family(family) => family.validate_support()?,
            };
            let coordinates = evaluator.event_coordinates(*wire)?;
            evaluator.trace.node_event_counts.insert(node.0, multiplicity * coordinates);
        }
        Ok(Self {
            evaluator: Some(evaluator),
            families: RnsBggNoiseFamilies::default(),
            environment,
            refresh_prf: None,
            pending_refresh: None,
        })
    }

    /// Configure this builder for the prepared PRF Program belonging to one
    /// refresh CRT channel.  This binding is intentionally separate from the
    /// source Program validation: the prepared declaration is the authority
    /// for the PRF output wires and labels.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<(), RnsBggError> {
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        evaluator.parameters.validate_refresh_prf_schedule(declaration, slot)?;
        declaration.validate_prf_program(evaluator.program.id())?;
        if self.refresh_prf.is_some() {
            return Err(RnsBggError::Binding("refresh PRF is already bound"));
        }
        self.refresh_prf =
            Some((slot, *declaration.commitment(), evaluator.parameters.parameter_identity()));
        Ok(())
    }

    /// Lower the prepared PRF Program once and return its outputs as
    /// `(wire, (wire, value))` pairs in declared output order.  No ordinary
    /// output finalization is applied to these wires.
    fn lower_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
    ) -> Result<Vec<(WireId, (WireId, RnsBggNoiseValue))>, RnsBggError> {
        let (slot, commitment, parameter_identity) =
            self.refresh_prf.ok_or(RnsBggError::Binding("refresh PRF is not bound"))?;
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        if parameter_identity != evaluator.parameters.parameter_identity() {
            return Err(RnsBggError::Binding("refresh PRF parameter identity changed"));
        }
        if commitment != *declaration.commitment() || slot >= declaration.slots().len() {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        let mut families = RnsBggNoiseFamilies::default();
        let outputs = crate::program::lower_program(
            evaluator.program,
            evaluator,
            &mut families,
            &mut self.environment,
        )
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        Ok(outputs.into_iter().map(|(wire, value)| (wire, (wire, value))).collect())
    }

    /// Apply one declared refresh using one separately configured PRF builder
    /// per CRT slot. The common refresh trait is invoked exactly once; the
    /// channel Programs are lowered once each by its callback.
    pub fn apply_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        inputs: BTreeMap<usize, (WireId, RnsBggNoiseValue)>,
        mut channels: BTreeMap<usize, RnsNoiseBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() {
            return Err(RnsBggError::Binding("refresh output was already selected"));
        }
        declaration.validate_prf_slots(&channels)?;
        let mut channel_traces = Vec::with_capacity(channels.len());
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        let source_level = evaluator.workload.level(declaration.source_output())?;
        evaluator.parameters.validate_refresh_level_contract(declaration, source_level)?;
        for (slot, channel) in &channels {
            let expected_child_parameters =
                evaluator.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if channel.evaluator.as_ref().is_none_or(|child| {
                child.parameters.parameter_identity() != expected_child_parameters
            }) || channel.refresh_prf.map(|(_, _, identity)| identity) !=
                Some(expected_child_parameters)
            {
                return Err(RnsBggError::Binding("refresh PRF child parameter identity mismatch"));
            }
        }
        let output = {
            let mut refresh = super::super::RnsBggRefresh::new(evaluator, inputs);
            crate::api::apply_refresh(declaration, &mut refresh, |callback_declaration, _| {
                callback_declaration.collect_prf_outputs(|slot| {
                    let mut channel = channels
                        .remove(&slot)
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                    let outputs = channel.lower_refresh_prf(callback_declaration)?;
                    channel_traces.push(
                        channel
                            .evaluator
                            .as_ref()
                            .ok_or(RnsBggError::Binding("noise evaluator consumed"))?
                            .trace
                            .clone(),
                    );
                    Ok(outputs)
                })
            })
            .map_err(|error| match error {
                crate::api::RefreshApplicationError::Refresh(error) => error,
                crate::api::RefreshApplicationError::OutputMismatch => {
                    RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
                }
            })?
        };
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        for channel_trace in &channel_traces {
            evaluator.merge_refresh_trace(channel_trace);
        }
        self.pending_refresh = Some((output_index, output));
        Ok(())
    }

    /// Compose refresh noise on this builder using the existing evaluator.
    /// The same method is used by simulation and by checked resumption paths;
    /// refresh metadata is retained in the evaluator and never wrapped in a
    /// second carrier.
    pub fn with_refresh_noise_parameters(
        &mut self,
        parameters: RefreshNoiseParameters,
    ) -> Result<(), RnsBggError> {
        self.evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("noise evaluator consumed"))?
            .with_refresh_noise_parameters(parameters)
    }
}

impl<'a> crate::api::NoiseBuild for RnsNoiseBuild<'a> {
    type Evaluator = RnsBggNoiseEvaluator<'a>;
    type Families = RnsBggNoiseFamilies;
    type Report = RnsBggNoiseTrace;
    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                (WireId, RnsBggNoiseValue),
                RhsRef<IntExpr>,
                (WireId, RnsBggNoiseFamily),
                (RhsId, FamilyRange),
            >,
        ),
        crate::api::ApiError,
    > {
        Ok((
            self.evaluator
                .as_mut()
                .ok_or_else(|| crate::api::ApiError::input("noise evaluator consumed"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
    fn build_report(
        mut self,
        outputs: Vec<(WireId, RnsBggNoiseValue)>,
    ) -> Result<Self::Report, crate::api::ApiError> {
        let mut evaluator = self
            .evaluator
            .take()
            .ok_or_else(|| crate::api::ApiError::input("noise evaluator consumed"))?;
        let finalized = evaluator
            .finish_outputs(outputs)
            .map_err(|error| crate::api::ApiError::scheme(error))?;
        let mut trace = evaluator.trace;
        let mut finalized = finalized;
        if let Some((index, refreshed)) = self.pending_refresh {
            if index >= finalized.len() {
                return Err(crate::api::ApiError::input("refresh output index is out of range"));
            }
            finalized[index] = refreshed.1;
        }
        trace.outputs = finalized;
        Ok(trace)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn refresh_prf_binding_checks_the_schedule_before_the_program() {
        let (parameters, source, declaration) =
            super::super::super::tests::refresh_prf_child_fixture(RhsMode::Static);
        let workload = RnsBggDerivedWorkload::from_program(&source).unwrap();
        let bindings = RnsBggNoiseBindings::from_inputs(
            &source,
            &[RnsBggNoiseValue::sampled_encoding(&parameters, b"source", &[BigInt::from(1u8)])],
            &[],
        )
        .unwrap();
        let mut build = RnsNoiseBuild::new(&parameters, &source, &workload, &bindings).unwrap();
        assert!(matches!(
            build.bind_refresh_prf(&declaration, 1),
            Err(RnsBggError::Binding("refresh PRF parameter schedule is not [q, ..., q, p]"))
        ));
        assert!(matches!(
            build.bind_refresh_prf(&declaration, 0),
            Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))
        ));
    }
}
