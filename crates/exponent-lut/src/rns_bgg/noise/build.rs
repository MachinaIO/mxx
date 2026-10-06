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
            pending_decoder: None,
        })
    }

    /// Configure this builder to lower `declaration`'s smudge Program: its
    /// smudge LUT members are finalized and transformed in their root bodies,
    /// as the encoding does.
    pub fn bind_decoder_smudge(
        &mut self,
        declaration: &DecoderDeclaration,
    ) -> Result<(), RnsBggError> {
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        declaration.validate_smudge_program(evaluator.program.id())?;
        if evaluator.decoder_smudge.is_some() {
            return Err(RnsBggError::Binding("decoder smudge is already bound"));
        }
        evaluator.decoder_smudge = Some(Arc::new(declaration.clone()));
        Ok(())
    }

    /// Compose the noise of `declaration`'s decoder of output `output_index`,
    /// whose lowered source value is `source`, with the bound smudge child
    /// `smudge` (on [`RnsBggParameters::decoder_smudge_parameters`]) and the
    /// unmask trapdoor sigma `trapdoor_sigma`. The report is returned in
    /// the trace as [`RnsBggNoiseTrace::decoder`].
    pub fn apply_decoder(
        &mut self,
        output_index: usize,
        declaration: &DecoderDeclaration,
        source: (WireId, RnsBggNoiseValue),
        mut smudge: RnsNoiseBuild<'a>,
        trapdoor_sigma: f64,
    ) -> Result<(), RnsBggError> {
        if self.pending_decoder.is_some() {
            return Err(RnsBggError::Binding("decoder output was already selected"));
        }
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        let child =
            smudge.evaluator.as_ref().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        evaluator.parameters.validate_decoder_smudge_parameters(
            child.parameters,
            declaration,
            evaluator.program,
        )?;
        if child.decoder_smudge.as_ref().map(|bound| *bound.commitment()) !=
            Some(*declaration.commitment()) ||
            evaluator.program.outputs().get(output_index) !=
                Some(&crate::program::Operand::Scalar(declaration.source_output()))
        {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        evaluator.unmask_trapdoor_sigma = Some(trapdoor_sigma);
        let mut decoder = super::super::RnsBggDecoder { lowerer: evaluator, source };
        let report = crate::api::apply_decoder(declaration, &mut decoder, |_| {
            let child = smudge
                .evaluator
                .as_mut()
                .ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
            let outputs = crate::program::lower_program(
                child.program,
                child,
                &mut smudge.families,
                &mut smudge.environment,
            )
            .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            Ok(outputs.into_iter().map(|(_, value)| value).collect())
        })?;
        let child_trace = smudge
            .evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("noise evaluator consumed"))?
            .trace
            .clone();
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        evaluator.merge_decoder_trace(&child_trace);
        self.pending_decoder = Some(report);
        Ok(())
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
        trace.outputs = finalized;
        trace.decoder = self.pending_decoder;
        Ok(trace)
    }
}
