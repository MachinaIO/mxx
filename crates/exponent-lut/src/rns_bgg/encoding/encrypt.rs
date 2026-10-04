//! The encryption stage adapter.
use super::*;

fn validate_refresh_prf_input_key_artifacts(
    manifests: &BTreeMap<ProductionId, Manifest>,
    production: &ProductionId,
    parameters: &RnsBggParameters,
    inputs: &crate::program::EncryptionInputs,
    commitment: [u8; 32],
    slot: usize,
) -> Result<(), RnsBggError> {
    let manifest = manifests
        .get(production)
        .ok_or(RnsBggError::Binding("refresh PRF preprocess manifest is missing"))?;
    let ring = resolve_ring(parameters.ring(0)?.as_ref())?;
    for (wire, declaration) in inputs.inputs() {
        let columns = parameters.key_component_columns(0, declaration.knowledge())?;
        let artifact_type =
            ArtifactType::Matrix(ConcreteMatrixType { ring: ring.clone(), rows: 1, columns });
        let family_count = match declaration.shape() {
            InputBinding::Scalar(()) => None,
            InputBinding::Family(count) => Some(
                count
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(RnsBggError::Binding("refresh PRF family count is not concrete"))?,
            ),
        };
        for port in 0..4 {
            let name = refresh_prf_input_key_artifact_name(commitment, slot, *wire, port);
            let artifact = manifest
                .artifacts
                .get(&name)
                .ok_or(RnsBggError::Binding("refresh PRF input-key artifact is missing"))?;
            if artifact.artifact_type != artifact_type || artifact.family_count != family_count {
                return Err(RnsBggError::Binding("refresh PRF input-key artifact schema mismatch"));
            }
        }
    }
    Ok(())
}

impl<'a> RnsEncryptBuild<'a> {
    /// Creates an unbound encryption builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            hash_key: Vec::new(),
            inputs: None,
            projection_identity: None,
            payload_secret: None,
            mask_secret: None,
            plaintexts: BTreeMap::new(),
            preprocess: None,
            binding: None,
            refresh_declaration: None,
            manifests: BTreeMap::new(),
            runtime: BTreeMap::new(),
            refresh_prf_children: BTreeMap::new(),
            refresh_prf_binding: None,
            dynamic_schema: None,
            dynamic_rhs: None,
        }
    }

    /// Bind public LHS plaintexts and secrets for a prepared PRF child.
    /// This application-facing method keeps the generic stage trait out of
    /// refresh child setup code. Dynamic children take no mask.
    pub fn bind_lhs_inputs<P: Poly>(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &RnsBggPayloadSecret,
        mask: Option<&RnsBggMaskSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        <Self as EncryptBuild<P>>::bind_inputs(
            self, inputs, parameters, lhs, payload, mask, hash_key,
        )
    }

    fn bind_preprocess_artifacts_inner(
        &mut self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ) -> Result<(), ApiError> {
        let binding = preprocess.descriptor().binding().clone();
        if binding.context() != &self.context ||
            (self.refresh_prf_binding.is_none() &&
                binding.parameters() != &self.parameters.parameter_identity()) ||
            binding.rhs_mode() != self.parameters.rhs_mode() ||
            binding.hash_key_commitment() != &self.hash_key_commitment
        {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess binding does not match encryption",
            ));
        }
        let production = preprocess.production().clone();
        if binding.sources().len() != 1 || !binding.sources().contains(&production) {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess production is not a declared source",
            ));
        }
        self.preprocess = Some(production.clone());
        self.manifests = BTreeMap::from([(production.clone(), preprocess.manifest().clone())]);
        if self.inputs.is_none() {
            return Err(ApiError::input("encryption inputs are not bound"));
        }
        // A PRF child's schema is its own Program's, bound with it.
        if self.parameters.rhs_mode() == RhsMode::Dynamic && self.refresh_prf_binding.is_none() {
            self.dynamic_schema =
                Some((binding.dynamic_rhs().clone(), binding.dynamic_luts().clone()));
        }
        self.binding = Some(binding);
        Ok(())
    }

    /// Bind the preprocess artifacts used by a prepared PRF child.
    pub fn bind_preprocess_artifacts(
        &mut self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ) -> Result<(), ApiError> {
        self.bind_preprocess_artifacts_inner(preprocess)
    }

    /// Bind this encryption child to one slot of the prepared PRF Program.
    /// The ordinary encryption checkpoint remains sourced from the outer
    /// Program; this explicit binding records the prepared identity and the
    /// slot projection used when its D0/initial carriers are published.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
        program: &ExponentLutProgram,
    ) -> Result<(), RnsBggError> {
        self.parameters.validate_refresh_prf_schedule(declaration, slot)?;
        declaration.validate_prf_program(program.id())?;
        let inputs = self
            .inputs
            .as_ref()
            .ok_or(RnsBggError::Binding("refresh PRF child inputs are not bound"))?;
        if inputs.inputs() != program.inputs() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if self.refresh_prf_binding.is_some() {
            return Err(RnsBggError::Binding("refresh PRF child is already bound"));
        }
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            let workload = RnsBggDerivedWorkload::from_program(program)?;
            let rhs = program
                .rhs_inputs()
                .iter()
                .map(|(id, shape)| Ok((*id, (shape.clone(), workload.rhs_level(*id)?))))
                .collect::<Result<BTreeMap<_, _>, RnsBggError>>()?;
            self.dynamic_schema = Some((rhs, workload.luts().clone()));
        }
        self.refresh_prf_binding = Some((
            *declaration.commitment(),
            slot,
            program.id(),
            self.parameters.parameter_identity(),
        ));
        Ok(())
    }

    /// Attach already-bound encryption children for the prepared PRF.
    ///
    /// Child outputs are emitted into the outer graph under a commitment and
    /// slot namespace.  No child checkpoint or second execution is created.
    pub fn bind_refresh_prf_children(
        &mut self,
        declaration: &RefreshDeclaration,
        children: BTreeMap<usize, RnsEncryptBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        declaration.validate_prf_slots(&children)?;
        if children.keys().any(|slot| self.refresh_prf_children.contains_key(slot)) {
            return Err(RnsBggError::Binding("refresh PRF child slot is already bound"));
        }
        let expected_program = declaration.prepared().program().id();
        for (slot, child) in &children {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            let binding = child
                .binding
                .as_ref()
                .ok_or(RnsBggError::Binding("refresh PRF child artifacts are not bound"))?;
            if binding.context() != &self.context ||
                binding.parameters() != &self.parameters.parameter_identity() ||
                binding.hash_key_commitment() != &self.hash_key_commitment ||
                binding.program() != declaration.source_program() ||
                child.inputs.is_none() ||
                child.payload_secret.is_none() ||
                child.mask_secret.is_none() ||
                child.preprocess.is_none() ||
                child.parameters.parameter_identity() != expected_child_parameters ||
                child.refresh_prf_binding.map(|(commitment, child_slot, program, identity)| {
                    commitment == *declaration.commitment() &&
                        child_slot == *slot &&
                        program == expected_program &&
                        identity == expected_child_parameters
                }) != Some(true)
            {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            validate_refresh_prf_input_key_artifacts(
                &child.manifests,
                child
                    .preprocess
                    .as_ref()
                    .ok_or(RnsBggError::Binding("refresh PRF child preprocess is not bound"))?,
                &child.parameters,
                child
                    .inputs
                    .as_ref()
                    .ok_or(RnsBggError::Binding("refresh PRF child inputs are not bound"))?,
                *declaration.commitment(),
                *slot,
            )?;
            if child
                .refresh_declaration
                .as_ref()
                .is_some_and(|bound| bound.commitment() != declaration.commitment())
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
            if declaration.slots().get(*slot).is_none() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
            // A Dynamic child encrypts under the parent's single secret, in
            // the parent's batch.
            let secret = |build: &RnsEncryptBuild<'a>| {
                build
                    .payload_secret
                    .as_ref()
                    .and_then(|secret| secret.binding.as_ref())
                    .map(|binding| binding.name.clone())
            };
            let batch = |build: &RnsEncryptBuild<'a>| build.dynamic_rhs.as_ref().map(|(id, _)| *id);
            if self.parameters.rhs_mode() == RhsMode::Dynamic &&
                (secret(child) != secret(self) ||
                    batch(child).is_none() ||
                    batch(child) != batch(self))
            {
                return Err(RnsBggError::Binding(
                    "Dynamic refresh children must share the parent's secret and batch",
                ));
            }
        }
        self.refresh_prf_children =
            children.into_iter().map(|(slot, child)| (slot, Box::new(child))).collect();
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }

    fn append_refresh_prf_child(
        child: &RnsEncryptBuild<'a>,
        declaration: &RefreshDeclaration,
        slot: usize,
        mut graph: DslContext,
        hash_key: &Bytes,
    ) -> Result<DslContext, ApiError> {
        let inputs = child
            .inputs
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child inputs are not bound"))?;
        let production = child
            .preprocess
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child preprocess is not bound"))?;
        let payload = child
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child payload is not bound"))?;
        let mask = child
            .mask_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child mask is not bound"))?;
        let commitment = *declaration.commitment();
        for (name, family) in
            child.lut_helper_families(mask, hash_key, inputs).map_err(ApiError::scheme)?
        {
            graph = graph
                .transferred_output(refresh_prf_artifact_name(commitment, slot, &name), family)
                .map_err(ApiError::scheme)?;
        }
        let workload = RnsBggDerivedWorkload::from_program(declaration.prepared().program())
            .map_err(ApiError::scheme)?;
        for (wire, declaration_input) in inputs.inputs() {
            // Each input is encoded at its logical level in the PRF Program.
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let ring = child.parameters.ring(level).map_err(ApiError::scheme)?;
            let columns = child
                .parameters
                .key_component_columns(level, declaration_input.knowledge())
                .map_err(ApiError::scheme)?;
            let key = RnsBggPublicKey {
                wire: *wire,
                matrix: (
                    (
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 0),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 1),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 2),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 3),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                ),
            };
            let value = child
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("refresh PRF child plaintext is not bound"))?;
            let encode = |key: RnsBggPublicKey, plaintext: Mat| {
                let plaintext = checked_modulus_convert(plaintext, &ring)?;
                if matches!(declaration_input.knowledge(), AttributeKnowledge::PublicPayload) {
                    child.encode_public(&key, level, mask, plaintext)
                } else {
                    child.encode_private(&key, level, payload, mask, plaintext)
                }
            };
            graph = match value {
                InputBinding::Scalar(value) => graph
                    .transferred_output(
                        refresh_prf_artifact_name(commitment, slot, &format!("input-{wire:?}")),
                        encode(key, value.clone()).map_err(ApiError::scheme)?,
                    )
                    .map_err(ApiError::scheme)?,
                InputBinding::Family(plaintexts) => {
                    let key_parts = (0..4)
                        .map(|port| {
                            ring.family_artifact_input(
                                production.clone(),
                                refresh_prf_input_key_artifact_name(commitment, slot, *wire, port),
                                plaintexts.count().clone(),
                                (1, columns),
                                ArtifactAvailability::Cached,
                            )
                        })
                        .collect::<Vec<_>>();
                    let plaintext_ring = child.parameters.ring(0).map_err(ApiError::scheme)?;
                    let family = mxx_dsl::parallel(plaintexts.count().clone(), |index| {
                        let plaintext = plaintexts.member(&plaintext_ring, index.clone());
                        let member_key = RnsBggPublicKey::new(
                            *wire,
                            (
                                (key_parts[0].at(index.clone()), key_parts[1].at(index.clone())),
                                (key_parts[2].at(index.clone()), key_parts[3].at(index.clone())),
                            ),
                        );
                        encode(member_key, plaintext).map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    graph
                        .transferred_output(
                            refresh_prf_artifact_name(commitment, slot, &format!("input-{wire:?}")),
                            family,
                        )
                        .map_err(ApiError::scheme)?
                }
            };
        }
        if child.parameters.rhs_mode() == RhsMode::Dynamic {
            graph = child.dynamic_material(
                graph,
                hash_key,
                declaration.prepared().program().id(),
                Some((commitment, slot)),
                |role| refresh_prf_artifact_name(commitment, slot, &role),
            )?;
        }
        Ok(graph)
    }

    /// Samples a payload secret on `execution`; it stays resident there.
    /// `before_execute` observes the sampling graph, whose output `secret`
    /// later stages read through [`RnsBggPayloadSecret::input_name`].
    pub fn sample_payload_secret<E, S>(
        &self,
        execution: &mut E,
        store: &mut S,
        before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
    ) -> Result<RnsBggPayloadSecret, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let (row, binding) = sample_secret_row(self.parameters, execution, store, before_execute)?;
        Ok(RnsBggPayloadSecret { row, binding: Some(binding) })
    }

    /// Samples a mask secret on `execution`; it stays resident there.
    /// Dynamic parameters reject it: their single payload secret is also the
    /// mask.
    /// `before_execute` observes the sampling graph, whose output `secret`
    /// later stages read through [`RnsBggMaskSecret::input_name`].
    pub fn sample_mask_secret<E, S>(
        &self,
        execution: &mut E,
        store: &mut S,
        before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
    ) -> Result<RnsBggMaskSecret, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            return Err(ApiError::input("Dynamic encryption samples no separate mask"));
        }
        let (row, binding) = sample_secret_row(self.parameters, execution, store, before_execute)?;
        Ok(RnsBggMaskSecret { row, binding: Some(binding) })
    }

    /// Binds the immutable refresh declaration used to publish decoder and
    /// per-label helper artifacts during encryption.
    pub fn bind_refresh_declaration(
        &mut self,
        declaration: &RefreshDeclaration,
    ) -> Result<(), RnsBggError> {
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }

    fn encode_private(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_private_inner(self.parameters, key, payload, mask, plaintext, level)
    }

    fn encode_public(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_public_inner(self.parameters, key, mask, plaintext, level)
    }

    /// The LUT helpers, one family per level and LUT input layout, whose
    /// member `i` belongs to the layout's `i`-th automorphism: its exports
    /// stream from a slot per lane rather than a slot per helper.
    fn lut_helper_families(
        &self,
        mask: &RnsBggMaskSecret,
        hash_key: &Bytes,
        inputs: &crate::program::EncryptionInputs,
    ) -> Result<Vec<(String, Family<RnsBggRow>)>, RnsBggError> {
        let mut families = Vec::new();
        for level in 0..self.parameters.levels().len().saturating_sub(1) {
            let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
            let layouts = inputs
                .luts()
                .iter()
                .map(|table| (format!("{:?}", table.input_layout()), table.input_layout()))
                .collect::<BTreeMap<_, _>>();
            for (name, layout) in layouts {
                let sigmas =
                    layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
                let family = mxx_dsl::parallel(sigmas.len(), |index| {
                    let sigma = IntExpr::Select {
                        selector: Box::new(index.expression()?),
                        branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
                    };
                    let public_data = super::public_key::lut_public_matrix(
                        self.parameters,
                        hash_key.clone(),
                        level,
                        Int::from(sigma.clone()),
                    )
                    .map_err(|_| DslError::Schema)?;
                    self.encrypt_lut_helper(mask, public_data, level, sigma)
                        .map_err(|_| DslError::Schema)
                })
                .map_err(graph)?;
                families.push((format!("lut-helper/{level}/{name}"), family));
            }
        }
        Ok(families)
    }

    /// Encrypt one preprocess-provided public mask-alignment matrix with the
    /// bound mask.  The `(level, sigma)` key is fixed by the Program and is
    /// not inferred from a LUT width or a family scope.
    fn encrypt_lut_helper(
        &self,
        mask: &RnsBggMaskSecret,
        public_data: RnsBggMatrix,
        level: usize,
        sigma: IntExpr,
    ) -> Result<RnsBggRow, RnsBggError> {
        let mask = reduce_secret(self.parameters, &mask.row, level)?;
        let scale = self.parameters.scalar(level, self.parameters.message_scale(level)?)?;
        let gadget = self.parameters.state_gadget(level)?;
        let columns = self.parameters.component_columns(level)?;
        let correction = (
            scale.clone() * mask.0.clone().ring_automorphism(sigma.clone()) * gadget.clone(),
            scale * mask.1.clone().ring_automorphism(sigma) * gadget,
        );
        let error = fresh_error(self.parameters, level, columns)?;
        Ok((
            mask.0.clone() * public_data.0.0 + mask.1.clone() * public_data.1.0 - correction.0 +
                error.0,
            mask.0 * public_data.0.1 + mask.1 * public_data.1.1 - correction.1 + error.1,
        ))
    }
}

impl<'a, P: Poly> EncryptBuild<P> for RnsEncryptBuild<'a> {
    type PayloadSecret = RnsBggPayloadSecret;
    type MaskSecret = RnsBggMaskSecret;
    type PreprocessDescriptor = RnsBggPreprocessEncryptDescriptor;
    type Descriptor = RnsBggEncryptionDescriptor;

    fn bind_inputs(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &Self::PayloadSecret,
        mask: Option<&Self::MaskSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let mask = stage_mask(self.parameters, payload, mask)?;
        if lhs.len() != inputs.inputs().len() ||
            lhs.keys().any(|wire| !inputs.inputs().contains_key(wire))
        {
            return Err(ApiError::input("encryption inputs do not exactly match Program inputs"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let prefix = super::plaintext_input_prefix();
        for (wire, declaration) in inputs.inputs() {
            let value = lhs.get(wire).ok_or_else(|| ApiError::input("missing encryption input"))?;
            match (declaration.shape(), value) {
                (InputBinding::Scalar(()), InputBinding::Scalar(_)) => {}
                (InputBinding::Family(count), InputBinding::Family(values)) => {
                    let expected = count
                        .evaluate(parameters)
                        .ok()
                        .and_then(|value| value.to_usize())
                        .ok_or_else(|| {
                            ApiError::input("encryption input family count is not concrete")
                        })?;
                    if values.len() != expected {
                        return Err(ApiError::input(
                            "encryption input family count does not match Program",
                        ));
                    }
                }
                _ => return Err(ApiError::input("encryption input shape does not match Program")),
            }
            let plaintext = super::plaintext_inputs(
                &ring,
                &format!("{prefix}/input/{wire:?}"),
                value,
                &mut self.runtime,
            )
            .map_err(ApiError::input)?;
            self.plaintexts.insert(*wire, plaintext);
        }
        self.inputs = Some(inputs.clone());
        self.projection_identity = Some(
            encryption_projection_identity(inputs, parameters, self.parameters)
                .map_err(ApiError::scheme)?,
        );
        for binding in [&payload.binding, &mask.binding].into_iter().flatten() {
            binding.bind(&mut self.runtime);
        }
        self.payload_secret = Some(payload.clone());
        self.mask_secret = Some(mask);
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.hash_key = hash_key.to_vec();
        Ok(())
    }

    fn bind_artifacts(
        &mut self,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
    ) -> Result<(), ApiError> {
        self.bind_preprocess_artifacts_inner(preprocess)
    }

    fn artifact_identity(&self) -> [u8; 32] {
        let mut digest = sha2::Sha256::new();
        digest.update(b"mxx-rns-bgg/encryption/v1");
        digest.update(self.context);
        digest.update(self.parameters.parameter_identity());
        digest.update(self.hash_key_commitment);
        if let Some(projection) = self.projection_identity {
            digest.update(projection);
        }
        if let Some(production) = &self.preprocess {
            digest.update(production.spec_hash.0);
            digest.update(production.execution_nonce);
        }
        if let Some(inputs) = &self.inputs {
            for wire in inputs.inputs().keys() {
                digest.update(format!("public-key/{wire:?}").as_bytes());
            }
        }
        if let Some(binding) = &self.binding {
            for source in binding.sources() {
                digest.update(source.spec_hash.0);
                digest.update(source.execution_nonce);
            }
        }
        // A Dynamic batch selects its own sampling session.
        if let Some((batch, _)) = &self.dynamic_rhs {
            digest.update(b"batch");
            digest.update(batch);
        }
        digest.finalize().into()
    }

    fn build_graph(mut self) -> Result<(StageGraph, Self::Descriptor), ApiError> {
        let inputs = self
            .inputs
            .clone()
            .ok_or_else(|| ApiError::input("encryption inputs are not bound"))?;
        if self.preprocess.is_none() {
            return Err(ApiError::input("preprocess artifacts are not bound"));
        }
        let payload = self
            .payload_secret
            .clone()
            .ok_or_else(|| ApiError::input("payload secret is not bound"))?;
        let mask =
            self.mask_secret.clone().ok_or_else(|| ApiError::input("mask secret is not bound"))?;
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let hash_key = ring.bytes_input(super::HASH_KEY_INPUT, self.hash_key.len());
        let mut graph = DslContext::new("rns-bgg-encrypt");
        if let Some(declaration) = self.refresh_declaration.as_ref() {
            for (slot, child) in &self.refresh_prf_children {
                graph =
                    Self::append_refresh_prf_child(child, declaration, *slot, graph, &hash_key)?;
                merge_runtime_inputs(&mut self.runtime, &child.runtime)?;
                self.manifests.extend(child.manifests.clone());
            }
        }
        for (name, family) in
            self.lut_helper_families(&mask, &hash_key, &inputs).map_err(ApiError::scheme)?
        {
            graph = graph.transferred_output(name, family).map_err(ApiError::scheme)?;
        }
        // Each input is encoded at its logical level under a public key
        // sampled from the hash key; only the row and, for a public input, the
        // payload are stored, and every later stage samples the key again.
        let binding =
            self.binding.as_ref().ok_or_else(|| ApiError::input("preprocess is not bound"))?;
        for (wire, declaration) in inputs.inputs() {
            let level = binding.input_levels().get(wire).copied().unwrap_or(0);
            let input_ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            // Plaintexts are bound in the natural input ring, level zero.
            let plaintext_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
            let knowledge = declaration.knowledge();
            let public = matches!(knowledge, AttributeKnowledge::PublicPayload);
            let value = self
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("missing encryption plaintext"))?;
            let encode = |member: Option<IntExpr>, plaintext: Mat| {
                let key = super::public_key::sampled_key(
                    self.parameters,
                    level,
                    hash_key.clone(),
                    *wire,
                    knowledge,
                    member,
                )?;
                let plaintext = checked_modulus_convert(plaintext, &input_ring)?;
                if public {
                    self.encode_public(&key, level, &mask, plaintext)
                } else {
                    self.encode_private(&key, level, &payload, &mask, plaintext)
                }
            };
            let name = format!("input-{wire:?}");
            graph = match (value, public) {
                (InputBinding::Scalar(value), true) => {
                    let encoding = encode(None, value.clone()).map_err(ApiError::scheme)?;
                    let payload = encoding
                        .public_payload
                        .ok_or_else(|| ApiError::input("public encoding has no payload"))?;
                    graph.transferred_output(name, (encoding.row.0, encoding.row.1, payload))
                }
                (InputBinding::Scalar(value), false) => {
                    let encoding = encode(None, value.clone()).map_err(ApiError::scheme)?;
                    graph.transferred_output(name, encoding.row)
                }
                (InputBinding::Family(plaintexts), true) => {
                    let stored = mxx_dsl::parallel(plaintexts.count().clone(), |index| {
                        let plaintext = plaintexts.member(&plaintext_ring, index.clone());
                        let encoding = encode(Some(index.expression()?), plaintext)
                            .map_err(|_| DslError::Schema)?;
                        let payload = encoding.public_payload.ok_or(DslError::Schema)?;
                        Ok((encoding.row.0, encoding.row.1, payload))
                    })
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(name, stored)
                }
                (InputBinding::Family(plaintexts), false) => {
                    let stored = mxx_dsl::parallel(plaintexts.count().clone(), |index| {
                        let plaintext = plaintexts.member(&plaintext_ring, index.clone());
                        encode(Some(index.expression()?), plaintext)
                            .map(|encoding| encoding.row)
                            .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(name, stored)
                }
            }
            .map_err(ApiError::scheme)?;
        }
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        if dynamic {
            graph =
                self.dynamic_material(graph, &hash_key, binding.program(), None, |role| role)?;
        }
        if let Some(declaration) = self.refresh_declaration.clone() {
            if dynamic {
                graph =
                    self.dynamic_refresh(graph, &declaration, &mask).map_err(ApiError::scheme)?;
            } else {
                for slot in declaration.slots() {
                    let sum = self
                        .preprocess_refresh_matrix(&declaration, slot.slot, "a-sum")
                        .map_err(ApiError::scheme)?;
                    let decoder = self
                        .refresh_decoder(&declaration, &mask, slot.slot, sum)
                        .map_err(ApiError::scheme)?;
                    graph = graph
                        .transferred_output(
                            refresh_artifact_name(*declaration.commitment(), slot.slot, "decoder"),
                            decoder.clone(),
                        )
                        .map_err(ApiError::scheme)?;
                }
            }
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        if super::graph_declares_input(&graph, super::HASH_KEY_INPUT) {
            self.runtime.insert(
                super::HASH_KEY_INPUT.to_owned(),
                RuntimeValue::Bytes(self.hash_key.as_slice().into()),
            );
        }
        let mut descriptor =
            self.binding.ok_or_else(|| ApiError::input("artifact binding is not available"))?;
        if let Some(declaration) = &self.refresh_declaration {
            descriptor = descriptor.with_refresh_declaration(declaration);
        }
        let runtime = declared_runtime_inputs(&graph, self.runtime);
        Ok((
            StageGraph::new(graph, runtime, self.manifests),
            RnsBggEncryptionDescriptor::from_binding(descriptor),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn encryption_lut_helper_produces_one_compact_member() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let one = ring.polynomial([IntExpr::constant(BigInt::from(1u8))]);
        let public_data = ((one.clone(), one.clone()), (one.clone(), one.clone()));
        let builder = RnsEncryptBuild::new(&parameters, [0; 32]);
        let mask = RnsBggMaskSecret {
            row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)),
            binding: None,
        };
        let helper =
            builder.encrypt_lut_helper(&mask, public_data, 0, IntExpr::constant(1)).unwrap();
        assert_eq!(helper.0.matrix_type().columns, IntExpr::constant(1));
    }

    #[test]
    fn encryption_projection_identity_is_sensitive_to_lut_semantics() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let mut first = crate::program::ExponentLutProgramBuilder::new();
        let input = first.input(AttributeKnowledge::PrivatePayload).unwrap();
        first
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![0, 1],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let first = first.finish(vec![Operand::Scalar(input)]).unwrap();
        let mut second = crate::program::ExponentLutProgramBuilder::new();
        let input = second.input(AttributeKnowledge::PrivatePayload).unwrap();
        second
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![1, 0],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let second = second.finish(vec![Operand::Scalar(input)]).unwrap();
        let env = mxx_ir_core::ParamEnv::default();
        assert_ne!(
            encryption_projection_identity(&first.encryption_inputs(), &env, &parameters).unwrap(),
            encryption_projection_identity(&second.encryption_inputs(), &env, &parameters).unwrap()
        );
    }
}
