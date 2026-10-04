//! The preprocessing stage adapter.
use super::*;

/// Classify a lowered public-key carrier for the preprocess artifact graph.
///
/// Raw input keys are sampled only from the hash key, so they are cacheable
/// regardless of the payload knowledge attached to the input declaration.
/// Derived public-payload wires remain hash-key/public-only, while private
/// payload wires (and refresh replacements) may depend on transferred secret
/// or private-plaintext material.
fn preprocess_output_availability(
    program: &ExponentLutProgram,
    wire: WireId,
    refresh_wire: Option<WireId>,
) -> Result<ArtifactAvailability, RnsBggError> {
    if refresh_wire == Some(wire) {
        return Ok(ArtifactAvailability::Transferred);
    }
    if program.inputs().contains_key(&wire) ||
        program
            .wire_knowledge(wire)
            .map_err(|_| RnsBggError::Binding("output wire knowledge"))? ==
            AttributeKnowledge::PublicPayload
    {
        Ok(ArtifactAvailability::Cached)
    } else {
        Ok(ArtifactAvailability::Transferred)
    }
}

fn deduplicate_outputs(outputs: Vec<RnsBggPublicKey>) -> Vec<RnsBggPublicKey> {
    let mut seen = BTreeSet::new();
    outputs.into_iter().filter(|output| seen.insert(output.wire)).collect()
}

impl<'a> RnsPreprocessBuild<'a> {
    /// Create an unbound preprocessing adapter.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            evaluator: None,
            families: super::encoding::OnlineEvalFamilies { parameters },
            environment: Environment::new(),
            runtime: BTreeMap::new(),
            refresh_declaration: None,
            pending_refresh: None,
            refresh_prf_binding: None,
            refresh_prf_children: BTreeMap::new(),
        }
    }

    /// Bind RHS plaintexts for a prepared PRF child without exposing the
    /// generic stage trait at application call sites. Dynamic children bind
    /// no RHS plaintexts and no secret.
    pub fn bind_rhs_inputs<P: Poly>(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        payload: Option<&RnsBggPayloadSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        <Self as PreprocessBuild<P>>::bind_inputs(self, program, parameters, rhs, payload, hash_key)
    }

    /// Bind the common refresh declaration before the preprocessing graph is
    /// built.  The declaration is copied as immutable metadata only; sampled
    /// matrices remain graph/artifact values owned by this stage.
    pub fn bind_refresh_declaration(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    ) -> Result<(), ApiError> {
        let evaluator =
            self.evaluator.as_ref().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        evaluator.validate_refresh_declaration(declaration).map_err(ApiError::scheme)?;
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }

    /// Bind this builder to one channel of the declaration's prepared PRF.
    ///
    /// A channel builder is intentionally separate from the source-stage
    /// declaration binding: it owns the prepared PRF Program and is lowered
    /// only when [`Self::apply_refresh`] dispatches the common refresh
    /// operation.  This check therefore validates the PRF output schema and
    /// coefficient lineage, without accepting a source carrier as a
    /// substitute for the prepared evaluator.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
        slot: usize,
    ) -> Result<(), ApiError> {
        self.parameters
            .validate_refresh_prf_schedule(declaration, slot)
            .map_err(ApiError::scheme)?;
        let evaluator =
            self.evaluator.as_ref().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        let expected_program = evaluator.program.id();
        declaration
            .validate_prf_program(expected_program)
            .and_then(|()| declaration.validate_prf_outputs(&evaluator.program))
            .map_err(|error| ApiError::scheme(RnsBggError::Refresh(error)))?;
        if self.refresh_prf_binding.is_some() {
            return Err(ApiError::input("refresh PRF is already bound"));
        }
        if let Some(evaluator) = self.evaluator.as_mut() {
            evaluator.refresh_prf_namespace = Some((*declaration.commitment(), slot));
        }
        self.refresh_prf_binding = Some((
            *declaration.commitment(),
            slot,
            expected_program,
            self.parameters.parameter_identity(),
        ));
        Ok(())
    }

    /// Lower the prepared PRF Program once and return its outputs as
    /// `(wire, value)` pairs in declared output order.
    fn lower_refresh_prf(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    ) -> Result<Vec<(WireId, RnsBggPublicKey)>, RnsBggError> {
        let (commitment, slot, program, parameter_identity) =
            self.refresh_prf_binding.ok_or(RnsBggError::Binding("refresh PRF is not bound"))?;
        if parameter_identity != self.parameters.parameter_identity() {
            return Err(RnsBggError::Binding("refresh PRF parameter identity changed"));
        }
        if commitment != *declaration.commitment() ||
            slot >= declaration.slots().len() ||
            program != declaration.prepared().program().id()
        {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("refresh PRF builder was not bound"))?;
        let program = evaluator.program.clone();
        let outputs = lower_program(&program, evaluator, &mut self.families, &mut self.environment)
            .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        Ok(outputs.into_iter().map(|value| (value.wire, value)).collect())
    }

    /// Apply one declared channel-aware refresh through the common lifecycle.
    /// The refreshed carrier replaces the selected preprocess output exactly
    /// once before graph construction.
    pub fn apply_refresh(
        &mut self,
        output_index: usize,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
        inputs: BTreeMap<usize, RnsBggPublicKey>,
        mut channels: BTreeMap<usize, RnsPreprocessBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() || output_index >= declaration.output_schema().len() {
            return Err(RnsBggError::Binding(
                "refresh output was already selected or is out of range",
            ));
        }
        let program = self
            .evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("preprocess builder was not bound"))?
            .program
            .as_ref();
        validate_refresh_output_target(program, output_index)?;
        if self
            .refresh_declaration
            .as_ref()
            .is_some_and(|bound| bound.commitment() != declaration.commitment())
        {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        declaration.validate_prf_slots(&channels).map_err(RnsBggError::Refresh)?;
        for (slot, child) in &channels {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if child.parameters.parameter_identity() != expected_child_parameters ||
                child.refresh_prf_binding.map(|(_, _, _, identity)| identity) !=
                    Some(expected_child_parameters)
            {
                return Err(RnsBggError::Binding("refresh PRF child parameter identity mismatch"));
            }
        }
        let (output, child_material) = {
            let evaluator = self
                .evaluator
                .as_mut()
                .ok_or(RnsBggError::Binding("preprocess builder was not bound"))?;
            let mut refresh = RnsBggRefresh::new(evaluator, inputs);
            let mut child_material = Vec::with_capacity(channels.len());
            let output = apply_refresh(declaration, &mut refresh, |callback_declaration, _| {
                let per_label = callback_declaration.collect_prf_outputs(|slot| {
                    channels
                        .get_mut(&slot)
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?
                        .lower_refresh_prf(callback_declaration)
                })?;
                for (slot, mut child) in std::mem::take(&mut channels) {
                    let evaluator = child
                        .evaluator
                        .take()
                        .ok_or(RnsBggError::Binding("refresh PRF builder was not bound"))?;
                    child_material.push((slot, evaluator, std::mem::take(&mut child.runtime)));
                }
                Ok(per_label)
            })
            .map_err(|error| match error {
                RefreshApplicationError::Refresh(error) => error,
                RefreshApplicationError::OutputMismatch => {
                    RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
                }
            })?;
            (output, child_material)
        };
        for (slot, child, runtime) in child_material {
            merge_runtime(&mut self.runtime, runtime)?;
            if self.refresh_prf_children.insert(slot, child).is_some() {
                return Err(RnsBggError::Binding("refresh PRF slot was already retained"));
            }
        }
        self.pending_refresh = Some((output_index, output));
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }
}

/// The stored part of a family of RHS ciphertexts: each member's bottom row.
fn merge_runtime(
    destination: &mut BTreeMap<String, RuntimeValue>,
    source: BTreeMap<String, RuntimeValue>,
) -> Result<(), RnsBggError> {
    for (name, value) in source {
        match destination.get(&name) {
            Some(RuntimeValue::Bytes(existing)) => match &value {
                RuntimeValue::Bytes(candidate) if existing == candidate => continue,
                RuntimeValue::Bytes(_) => {
                    return Err(RnsBggError::Binding("refresh child runtime input conflicts"));
                }
                _ => return Err(RnsBggError::Binding("refresh child runtime input schema")),
            },
            // A secret input name is unique to one sampled secret.
            Some(RuntimeValue::Matrix(_)) if matches!(value, RuntimeValue::Matrix(_)) => {}
            Some(_) => return Err(RnsBggError::Binding("refresh child runtime input schema")),
            None => {
                destination.insert(name, value);
            }
        }
    }
    Ok(())
}

impl<'a, P: Poly> PreprocessBuild<P> for RnsPreprocessBuild<'a> {
    type Evaluator = PublicKeyLowering<'a>;
    type Families = super::encoding::OnlineEvalFamilies<'a>;
    type PayloadSecret = RnsBggPayloadSecret;
    type EncryptDescriptor = RnsBggPreprocessEncryptDescriptor;
    type OnlineEvalDescriptor = RnsBggPreprocessOnlineEvalDescriptor;
    type Encrypt = RnsEncryptBuild<'a>;
    type OnlineEval = RnsOnlineEvalBuild<'a>;
    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                RnsBggPublicKey,
                RhsRef<Int>,
                PreprocessWireFamily,
                Family<RhsRef<Int>>,
            >,
        ),
        ApiError,
    > {
        Ok((
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_inputs first"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
    fn bind_inputs(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        payload: Option<&Self::PayloadSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        match (dynamic, payload) {
            (false, Some(_)) | (true, None) => {}
            (false, None) => {
                return Err(ApiError::input("Static preprocessing requires the payload secret"));
            }
            (true, Some(_)) => {
                return Err(ApiError::input("Dynamic preprocessing takes no secret"));
            }
        }
        if dynamic && !rhs.is_empty() {
            return Err(ApiError::input(
                "Dynamic preprocessing takes no RHS plaintexts; encryption binds them",
            ));
        }
        if !dynamic &&
            (rhs.len() != program.rhs_inputs().len() ||
                rhs.keys().any(|id| !program.rhs_inputs().contains_key(id)))
        {
            return Err(ApiError::input("RHS bindings do not exactly match Program inputs"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let bytes = ring.bytes_input(HASH_KEY_INPUT, hash_key.len());
        let program = Arc::new(program.clone());
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(&program).map_err(ApiError::scheme)?);
        let mut evaluator = PublicKeyLowering::new(
            self.parameters,
            program.clone(),
            workload,
            payload.cloned(),
            bytes,
            self.context,
        );
        for (wire, declaration) in program.inputs() {
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(
                    evaluator.sample_input_public_key(*wire).map_err(ApiError::scheme)?,
                ),
                // Each member's key is sampled where it is read.
                InputBinding::Family(count) => InputBinding::Family(PreprocessWireFamily::Input {
                    wire: *wire,
                    level: evaluator.workload.level(*wire).map_err(ApiError::scheme)?,
                    knowledge: evaluator
                        .program
                        .wire_knowledge(*wire)
                        .map_err(|_| ApiError::input("input wire knowledge"))?,
                    hash_key: evaluator.hash_key.clone(),
                    count: count.clone(),
                }),
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|e| ApiError::input(e.to_string()))?;
        }
        let prefix = super::plaintext_input_prefix();
        for (id, declaration) in program.rhs_inputs() {
            if dynamic {
                // Only references: every key derives from public companion
                // keys, and encryption binds the plaintexts.
                evaluator.workload.rhs_level(*id).map_err(ApiError::scheme)?;
                let binding = match declaration {
                    InputBinding::Scalar(()) => {
                        InputBinding::Scalar(RhsRef { id: *id, index: None::<Int> })
                    }
                    InputBinding::Family(count) => {
                        let count = count
                            .evaluate(parameters)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or_else(|| ApiError::input("RHS family count is not concrete"))?;
                        InputBinding::Family(
                            mxx_dsl::parallel(IntExpr::constant(count), |index| {
                                Ok(RhsRef { id: *id, index: Some(index) })
                            })
                            .map_err(ApiError::scheme)?,
                        )
                    }
                };
                self.environment
                    .bind_rhs(*id, binding)
                    .map_err(|e| ApiError::input(e.to_string()))?;
                continue;
            }
            let value = rhs.get(id).ok_or_else(|| ApiError::input("missing RHS plaintext"))?;
            let plaintext = super::plaintext_inputs(
                &ring,
                &format!("{prefix}/rhs/{id:?}"),
                value,
                &mut self.runtime,
            )
            .map_err(ApiError::input)?;
            match (declaration, plaintext) {
                (InputBinding::Scalar(()), InputBinding::Scalar(plaintext)) => {
                    evaluator.rhs.insert(*id, InputBinding::Scalar(plaintext));
                    let reference = RhsRef { id: *id, index: None::<Int> };
                    let material =
                        evaluator.prepare_rhs_material(&reference).map_err(ApiError::scheme)?;
                    evaluator.rhs_material.insert(*id, InputBinding::Scalar(material));
                    self.environment
                        .bind_rhs(*id, InputBinding::Scalar(reference))
                        .map_err(|e| ApiError::input(e.to_string()))?;
                }
                (InputBinding::Family(count), InputBinding::Family(plaintexts)) => {
                    let expected = count
                        .clone()
                        .evaluate(parameters)
                        .ok()
                        .and_then(|value| value.to_usize())
                        .ok_or_else(|| ApiError::input("RHS family count is not concrete"))?;
                    let members = plaintexts.count().clone();
                    if members != IntExpr::constant(expected) {
                        return Err(ApiError::input("RHS family count does not match Program"));
                    }
                    evaluator.rhs.insert(*id, InputBinding::Family(plaintexts));
                    let prepared = mxx_dsl::parallel(members.clone(), |index| {
                        evaluator
                            .prepare_rhs_material(&RhsRef { id: *id, index: Some(index) })
                            .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    evaluator.rhs_material.insert(*id, InputBinding::Family(prepared));
                    let references = mxx_dsl::parallel(members, |index| {
                        Ok(RhsRef { id: *id, index: Some(index) })
                    })
                    .map_err(ApiError::scheme)?;
                    self.environment
                        .bind_rhs(*id, InputBinding::Family(references))
                        .map_err(|e| ApiError::input(e.to_string()))?;
                }
                _ => return Err(ApiError::input("RHS plaintext shape does not match Program")),
            }
        }
        // Plaintext RHS graph handles stay for lowering: a loop that reads an
        // RHS family member prepares its compact ciphertext there.
        self.runtime.insert(HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.into()));
        if let Some(binding) = payload.and_then(|payload| payload.binding.as_ref()) {
            binding.bind(&mut self.runtime);
        }
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.evaluator = Some(evaluator);
        Ok(())
    }
    fn artifact_identities(&self) -> ([u8; 32], [u8; 32]) {
        let Some(evaluator) = self.evaluator.as_ref() else {
            return ([0; 32], [0; 32]);
        };
        let mut digest = Sha256::new();
        digest.update(evaluator.program.id().as_bytes());
        digest.update(self.parameters.parameter_identity());
        digest.update(self.hash_key_commitment);
        if let Some(declaration) = &self.refresh_declaration {
            digest.update(declaration.commitment());
        }
        let encryption = digest.clone().finalize().into();
        digest.update(b"online-eval");
        let online_eval = digest.finalize().into();
        (encryption, online_eval)
    }
    fn finalize_descriptors(
        production: &ProductionId,
        (encrypt, online_eval): (Self::EncryptDescriptor, Self::OnlineEvalDescriptor),
    ) -> Result<(Self::EncryptDescriptor, Self::OnlineEvalDescriptor), ApiError> {
        let encrypt = encrypt.with_source(production.clone()).map_err(ApiError::scheme)?;
        Ok((encrypt, online_eval))
    }

    fn build_graph(
        mut self,
        outputs: Vec<RnsBggPublicKey>,
    ) -> Result<(StageGraph, (Self::EncryptDescriptor, Self::OnlineEvalDescriptor)), ApiError> {
        let mut evaluator =
            self.evaluator.ok_or_else(|| ApiError::input("preprocess builder was not bound"))?;
        let mut outputs = outputs;
        let refresh_output_wire = self.pending_refresh.as_ref().map(|(_, output)| output.wire);
        if let Some((index, output)) = self.pending_refresh.take() {
            let target = outputs.get_mut(index).ok_or_else(|| {
                ApiError::input("refresh output index is outside lowered outputs")
            })?;
            *target = output;
        }
        let outputs =
            deduplicate_outputs(evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?);
        let mut graph = DslContext::new("rns-bgg-preprocess");

        // Input public keys are sampled from the hash key alone, so no stage
        // stores them: encryption and online evaluation derive them where
        // they need them. A raw input that is also a Program output is
        // skipped here for the same reason.
        for output in outputs {
            if evaluator.program.inputs().contains_key(&output.wire) {
                continue;
            }
            let name = format!("public-key/{:?}", output.wire);
            graph = match preprocess_output_availability(
                &evaluator.program,
                output.wire,
                refresh_output_wire,
            )
            .map_err(ApiError::scheme)?
            {
                ArtifactAvailability::Cached => graph.cached_output(name, output),
                ArtifactAvailability::Transferred => graph.transferred_output(name, output),
            }
            .map_err(ApiError::scheme)?;
        }
        // Prepared PRF children use their own source-q key schedule.  Publish
        // their input keys under the declaration/slot namespace so encryption
        // cannot resolve a same-named outer-program key or another slot's key.
        if let Some(declaration) = self.refresh_declaration.as_ref() {
            let commitment = *declaration.commitment();
            for (slot, child) in &self.refresh_prf_children {
                for (wire, input) in child.program.inputs() {
                    let names = (0..4)
                        .map(|port| {
                            refresh_prf_input_key_artifact_name(commitment, *slot, *wire, port)
                        })
                        .collect::<Vec<_>>();
                    match input.shape() {
                        InputBinding::Scalar(()) => {
                            let key =
                                child.sample_input_public_key(*wire).map_err(ApiError::scheme)?;
                            for (name, entry) in names.into_iter().zip(matrix_entries(&key.matrix))
                            {
                                graph =
                                    graph.cached_output(name, entry).map_err(ApiError::scheme)?;
                            }
                        }
                        InputBinding::Family(count) => {
                            let key_parts = (0..4)
                                .map(|port| {
                                    mxx_dsl::parallel(count.clone(), |index| {
                                        let key = child
                                            .sample_input_public_key_member(*wire, Some(index))
                                            .map_err(|_| DslError::Schema)?;
                                        Ok(matrix_entries(&key.matrix)[port].clone())
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()
                                .map_err(ApiError::scheme)?;
                            for (name, entry) in names.into_iter().zip(key_parts) {
                                graph =
                                    graph.cached_output(name, entry).map_err(ApiError::scheme)?;
                            }
                        }
                    }
                }
                for (rhs, prepared) in &child.rhs_material {
                    match prepared {
                        InputBinding::Scalar(value) => {
                            graph = graph
                                .transferred_output(
                                    refresh_prf_artifact_name(
                                        commitment,
                                        *slot,
                                        &format!("rhs/{rhs:?}"),
                                    ),
                                    value.1.clone(),
                                )
                                .map_err(ApiError::scheme)?;
                        }
                        InputBinding::Family(value) => {
                            graph = graph
                                .transferred_output(
                                    refresh_prf_artifact_name(
                                        commitment,
                                        *slot,
                                        &format!("rhs/{rhs:?}/family"),
                                    ),
                                    child
                                        .published_rhs_family(rhs, value)?
                                        .field(|matrix| matrix.1)
                                        .map_err(ApiError::scheme)?,
                                )
                                .map_err(ApiError::scheme)?;
                        }
                    }
                }
                for ((evaluation_level, decomposition_level, lut, layout), material) in
                    &child.lut_material
                {
                    let role = format!(
                        "lut-switching/{evaluation_level}/{decomposition_level}/{lut:?}/{layout:?}"
                    );
                    graph = graph
                        .transferred_output(
                            refresh_prf_artifact_name(commitment, *slot, &role),
                            material.clone(),
                        )
                        .map_err(ApiError::scheme)?;
                }
            }
        }
        // Only the bottom row of each RHS ciphertext is stored; its top row
        // is sampled from the hash key (`rhs_top`).
        for (rhs, prepared) in &evaluator.rhs_material {
            match prepared {
                InputBinding::Scalar(value) => {
                    graph = graph
                        .transferred_output(format!("rhs/{rhs:?}"), value.1.clone())
                        .map_err(ApiError::scheme)?;
                }
                InputBinding::Family(value) => {
                    graph = graph
                        .transferred_output(
                            format!("rhs/{rhs:?}/family"),
                            evaluator
                                .published_rhs_family(rhs, value)?
                                .field(|matrix| matrix.1)
                                .map_err(ApiError::scheme)?,
                        )
                        .map_err(ApiError::scheme)?;
                }
            }
        }
        for ((evaluation_level, decomposition_level, lut, layout), material) in
            &evaluator.lut_material
        {
            graph = graph
                .transferred_output(
                    format!(
                        "lut-switching/{evaluation_level}/{decomposition_level}/{lut:?}/{layout:?}"
                    ),
                    material.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        for ((commitment, slot), sum) in &evaluator.refresh_sums {
            for (index, entry) in matrix_entries(sum).into_iter().enumerate() {
                graph = graph
                    .transferred_output(
                        refresh_artifact_name(*commitment, *slot, &format!("a-sum.{index}")),
                        entry,
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
        for ((commitment, slot), star) in &evaluator.refresh_stars {
            for (index, entry) in matrix_entries(star).into_iter().enumerate() {
                graph = graph
                    .cached_output(
                        refresh_artifact_name(*commitment, *slot, &format!("a-star.{index}")),
                        entry,
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
        for ((commitment, slot), action) in &evaluator.refresh_targets {
            graph = graph
                .transferred_output(
                    refresh_artifact_name(*commitment, *slot, "target"),
                    action.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        for ((commitment, slot, label), action) in &evaluator.refresh_label_targets {
            graph = graph
                .transferred_output(
                    refresh_label_artifact_name(*commitment, *slot, *label),
                    action.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        // Dynamic refresh keys are public; encryption forms the secret-
        // dependent actions and A_sum from them.
        for ((commitment, slot, label), key) in &evaluator.refresh_keys {
            graph = graph
                .cached_output(refresh_key_artifact_name(*commitment, *slot, *label), key.clone())
                .map_err(ApiError::scheme)?;
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let input_levels = evaluator
            .program
            .inputs()
            .keys()
            .map(|wire| Ok((*wire, evaluator.workload.level(*wire)?)))
            .collect::<Result<BTreeMap<_, _>, RnsBggError>>()
            .map_err(ApiError::scheme)?;
        let mut binding = RnsBggArtifactBinding::new(
            self.context,
            evaluator.program.id(),
            self.parameters.parameter_identity(),
            self.hash_key_commitment,
            Vec::<ProductionId>::new(),
            self.parameters.rhs_mode(),
        )
        .map_err(ApiError::scheme)?
        .with_input_levels(input_levels);
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            let rhs = evaluator
                .program
                .rhs_inputs()
                .iter()
                .map(|(id, shape)| Ok((*id, (shape.clone(), evaluator.workload.rhs_level(*id)?))))
                .collect::<Result<BTreeMap<_, _>, RnsBggError>>()
                .map_err(ApiError::scheme)?;
            binding = binding.with_dynamic_schema(rhs, evaluator.workload.luts().clone());
        }
        if let Some(declaration) = &self.refresh_declaration {
            binding = binding.with_refresh_declaration(declaration);
        }
        let encrypt_binding = RnsBggPreprocessEncryptDescriptor::from_binding(binding.clone());
        let online_eval_binding = RnsBggPreprocessOnlineEvalDescriptor::from_binding(binding);
        let runtime = super::encoding::declared_runtime_inputs(&graph, self.runtime);
        Ok((
            StageGraph::new(graph, runtime, BTreeMap::<ProductionId, Manifest>::new()),
            (encrypt_binding, online_eval_binding),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use mxx_ir_core::ParamEnv;

    #[test]
    fn preprocess_classifies_raw_and_public_outputs_as_cached() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let public_output = builder
            .add(crate::program::Operand::Scalar(public), crate::program::Operand::Scalar(public))
            .unwrap();
        let private_output = builder
            .mul_public(
                crate::program::Operand::Scalar(public),
                crate::program::Operand::Scalar(private),
            )
            .unwrap();
        let rhs = builder.rhs_input().unwrap();
        let private_rhs_output = builder
            .mul_private(
                crate::program::Operand::Scalar(private_output),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let program = builder
            .finish(vec![
                crate::program::Operand::Scalar(private),
                crate::program::Operand::Scalar(public_output),
                crate::program::Operand::Scalar(private_rhs_output),
            ])
            .unwrap();

        assert_eq!(
            preprocess_output_availability(&program, private, None).unwrap(),
            ArtifactAvailability::Cached
        );
        assert_eq!(
            preprocess_output_availability(&program, public_output, None).unwrap(),
            ArtifactAvailability::Cached
        );
        assert_eq!(
            preprocess_output_availability(&program, private_output, None).unwrap(),
            ArtifactAvailability::Transferred
        );
        assert_eq!(
            preprocess_output_availability(&program, private_rhs_output, None).unwrap(),
            ArtifactAvailability::Transferred
        );
        assert_eq!(
            preprocess_output_availability(&program, public_output, Some(public_output)).unwrap(),
            ArtifactAvailability::Transferred
        );
    }

    #[test]
    fn repeated_program_outputs_keep_the_first_public_key_carrier() {
        let ring = crate::test_support::explicit_ring(&[17], 4);
        let matrix = || {
            let one = ring.identity(1);
            ((one.clone(), one.clone()), (one.clone(), one))
        };
        let outputs = vec![
            RnsBggPublicKey::new(WireId(4), matrix()),
            RnsBggPublicKey::new(WireId(4), matrix()),
            RnsBggPublicKey::new(WireId(7), matrix()),
            RnsBggPublicKey::new(WireId(4), matrix()),
        ];
        let outputs = deduplicate_outputs(outputs);
        assert_eq!(
            outputs.iter().map(|output| output.wire).collect::<Vec<_>>(),
            [WireId(4), WireId(7)]
        );
    }

    #[test]
    fn online_evaluation_identity_extends_the_encryption_identity_by_its_domain() {
        let (parameters, source, _) =
            super::super::tests::refresh_prf_child_fixture(RhsMode::Dynamic);
        let mut build = RnsPreprocessBuild::new(&parameters, [1; 32]);
        build
            .bind_rhs_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &source,
                &mxx_ir_core::ParamEnv::default(),
                &BTreeMap::new(),
                None,
                &[2; 32],
            )
            .unwrap();
        let (encryption, online_eval) = <RnsPreprocessBuild as PreprocessBuild<
            mxx_backends::poly::dcrt::poly::DCRTPoly,
        >>::artifact_identities(&build);
        let mut digest = Sha256::new();
        digest.update(source.id().as_bytes());
        digest.update(parameters.parameter_identity());
        digest.update(build.hash_key_commitment);
        assert_eq!(encryption, <[u8; 32]>::from(digest.clone().finalize()));
        digest.update(b"online-eval");
        assert_eq!(online_eval, <[u8; 32]>::from(digest.finalize()));
    }

    #[test]
    fn refresh_prf_binding_reports_a_wrong_program_as_a_refresh_error() {
        let (parameters, source, declaration) =
            super::super::tests::refresh_prf_child_fixture(RhsMode::Dynamic);
        let mut build = RnsPreprocessBuild::new(&parameters, [1; 32]);
        build
            .bind_rhs_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &source,
                &mxx_ir_core::ParamEnv::default(),
                &BTreeMap::new(),
                None,
                &[2; 32],
            )
            .unwrap();
        let Err(ApiError::Scheme { source }) = build.bind_refresh_prf(&declaration, 0) else {
            panic!("a wrong PRF Program must be a scheme error");
        };
        assert!(matches!(
            source.downcast_ref::<RnsBggError>(),
            Some(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))
        ));
    }

    #[test]
    fn bind_inputs_canonicalizes_later_rhs_levels_from_natural_input_ring() {
        use mxx_backends::poly::dcrt::poly::DCRTPoly;

        let source = DCRTPolyParams::new(16, 3, 17, 4, None, None);
        let moduli = source.to_crt().0;
        let middle = source
            .select_modulus(
                &moduli[..2]
                    .iter()
                    .fold(BigUint::one(), |product, prime| product * BigUint::from(*prime)),
            )
            .unwrap();
        let terminal = source.select_modulus(&BigUint::from(moduli[0])).unwrap();
        let parameters =
            RnsBggParameters::new(vec![source.clone(), middle, terminal], 1, 1).unwrap();

        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let first_rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(first_rhs),
            )
            .unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(second_rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();
        let expected_workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
        assert_eq!(expected_workload.rhs_level(first_rhs).unwrap(), 0);
        assert_eq!(expected_workload.rhs_level(second_rhs).unwrap(), 1);

        let ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret {
            row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)),
            binding: None,
        };
        // Plaintexts are signed monomials, here constants.
        let value = |constant| DCRTPoly::from_u32s(&source, &[constant]);
        let mut rhs = BTreeMap::new();
        rhs.insert(first_rhs, InputBinding::Scalar(value(3)));
        rhs.insert(second_rhs, InputBinding::Scalar(value(5)));

        let mut preprocess = RnsPreprocessBuild::new(&parameters, [0x52; 32]);
        <RnsPreprocessBuild<'_> as PreprocessBuild<DCRTPoly>>::bind_inputs(
            &mut preprocess,
            &program,
            &ParamEnv::default(),
            &rhs,
            Some(&payload),
            b"later-rhs-levels",
        )
        .unwrap();

        let evaluator = preprocess.evaluator.as_ref().unwrap();
        let second_material = match evaluator.rhs_material.get(&second_rhs).unwrap() {
            InputBinding::Scalar(value) => value,
            InputBinding::Family(_) => panic!("scalar RHS has family material"),
        };
        let expected_schema =
            parameters.ring(2).unwrap().matrix_type((1, parameters.component_columns(2).unwrap()));
        for entry in matrix_entries(second_material) {
            assert_eq!(entry.matrix_type(), &expected_schema);
        }
    }
}
