//! The encryption stage adapter.
use super::*;

fn validate_decoder_smudge_input_key_artifacts(
    manifests: &BTreeMap<ProductionId, Manifest>,
    production: &ProductionId,
    parameters: &RnsBggParameters,
    inputs: &crate::program::EncryptionInputs,
    commitment: [u8; 32],
) -> Result<(), RnsBggError> {
    let manifest = manifests
        .get(production)
        .ok_or(RnsBggError::Binding("decoder smudge preprocess manifest is missing"))?;
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
                    .ok_or(RnsBggError::Binding("decoder smudge family count is not concrete"))?,
            ),
        };
        for port in 0..4 {
            let name = decoder_smudge_input_key_artifact_name(commitment, *wire, port);
            let artifact = manifest
                .artifacts
                .get(&name)
                .ok_or(RnsBggError::Binding("decoder smudge input-key artifact is missing"))?;
            if artifact.artifact_type != artifact_type || artifact.family_count != family_count {
                return Err(RnsBggError::Binding(
                    "decoder smudge input-key artifact schema mismatch",
                ));
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
            decoder_declaration: None,
            manifests: BTreeMap::new(),
            runtime: BTreeMap::new(),
            decoder_smudge_child: None,
            decoder_smudge: None,
            dynamic_schema: None,
            dynamic_rhs: None,
            prelude: None,
            hash_namespace: None,
            refresh: None,
        }
    }

    /// Make this builder a refresh child that samples its public matrices
    /// from the stage hash key derived under `namespace`.
    pub fn with_hash_namespace(mut self, namespace: [u8; 32]) -> Self {
        self.hash_namespace = Some(namespace);
        self
    }

    /// Attach `declaration`'s noise refresh children, each bound to its
    /// inputs (the PRF label and `mono(0)`), the secrets and the source's
    /// preprocess artifacts: `fresh` the fresh-noise child, `smudges` the
    /// decoders' smudge children in decoder order. Their encodings, the one
    /// encoding and the unmask hint are published into this encryption.
    pub fn bind_refresh(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        fresh: RnsEncryptBuild<'a>,
        smudges: Vec<RnsEncryptBuild<'a>>,
    ) -> Result<(), ApiError> {
        self.bind_refresh_children(declaration, None, fresh, smudges)
    }

    /// Attach the top-refresh part `declaration`'s children as
    /// [`Self::bind_refresh`] does, with its mask child `mask` (bound to the
    /// shared mask keys' inputs under [`crate::rns_bgg::refresh_mask_namespace`]).
    pub fn bind_top_refresh(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        mask: RnsEncryptBuild<'a>,
        fresh: RnsEncryptBuild<'a>,
        smudges: Vec<RnsEncryptBuild<'a>>,
    ) -> Result<(), ApiError> {
        self.bind_refresh_children(declaration, Some(mask), fresh, smudges)
    }

    fn bind_refresh_children(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        mask: Option<RnsEncryptBuild<'a>>,
        fresh: RnsEncryptBuild<'a>,
        smudges: Vec<RnsEncryptBuild<'a>>,
    ) -> Result<(), ApiError> {
        let mismatch = || ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch));
        if self.decoder_smudge_child.is_some() || self.refresh.is_some() {
            return Err(ApiError::input("a decoder or refresh was already bound"));
        }
        if self.parameters.rhs_mode() != RhsMode::Static ||
            smudges.len() != declaration.decoders().len()
        {
            return Err(mismatch());
        }
        let bound = |child: &RnsEncryptBuild<'a>, namespace: [u8; 32]| {
            child.binding.as_ref().is_some_and(|binding| {
                binding.context() == &self.context &&
                    binding.hash_key_commitment() == &self.hash_key_commitment &&
                    binding.program() == declaration.source_program() &&
                    binding.decoder_commitment() == Some(declaration.spec())
            }) && child.inputs.is_some() &&
                child.payload_secret.is_some() &&
                child.mask_secret.is_some() &&
                child.hash_namespace == Some(namespace)
        };
        if !bound(&fresh, crate::rns_bgg::refresh::refresh_fresh_namespace(declaration)) ||
            fresh.inputs.as_ref().map(|inputs| inputs.inputs()) !=
                Some(declaration.fresh().program().inputs())
        {
            return Err(mismatch());
        }
        for (child, decoder) in smudges.iter().zip(declaration.decoders()) {
            if !bound(child, *decoder.commitment()) ||
                child.decoder_smudge.as_ref().map(|bound| *bound.commitment()) !=
                    Some(*decoder.commitment())
            {
                return Err(mismatch());
            }
        }
        match (declaration.top(), &mask) {
            (None, None) => {}
            (Some(part), Some(child))
                if bound(child, crate::rns_bgg::refresh::refresh_mask_namespace(declaration)) &&
                    child.inputs.as_ref().map(|inputs| inputs.inputs()) ==
                        Some(part.mask().program().inputs()) => {}
            _ => return Err(mismatch()),
        }
        self.refresh = Some((declaration.clone(), mask.map(Box::new), Box::new(fresh), smudges));
        Ok(())
    }

    /// Bind public LHS plaintexts and secrets for a decoder smudge child.
    /// This application-facing method keeps the generic stage trait out of
    /// decoder child setup code. Dynamic children take no mask.
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
        // A child encrypts on its own chain, so only the root checks the
        // parameter identity.
        let child = self.decoder_smudge.is_some() || self.hash_namespace.is_some();
        if binding.context() != &self.context ||
            (!child && binding.parameters() != &self.parameters.parameter_identity()) ||
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
        // A decoder smudge child's schema is its own Program's, bound with it.
        if self.parameters.rhs_mode() == RhsMode::Dynamic && self.decoder_smudge.is_none() {
            self.dynamic_schema =
                Some((binding.dynamic_rhs().clone(), binding.dynamic_luts().clone()));
        }
        self.binding = Some(binding);
        Ok(())
    }

    /// Bind the preprocess artifacts used by a decoder smudge child.
    pub fn bind_preprocess_artifacts(
        &mut self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ) -> Result<(), ApiError> {
        self.bind_preprocess_artifacts_inner(preprocess)
    }

    /// Bind this encryption child, on
    /// [`RnsBggParameters::decoder_smudge_parameters`], to `declaration`'s smudge
    /// Program, after its LHS inputs and the source's preprocess artifacts
    /// are bound. Its encodings are published into the parent's graph under
    /// the decoder namespace; it is never executed on its own.
    pub fn bind_decoder_smudge(
        &mut self,
        declaration: &DecoderDeclaration,
    ) -> Result<(), RnsBggError> {
        let program = declaration.smudge().program();
        let inputs = self
            .inputs
            .as_ref()
            .ok_or(RnsBggError::Binding("decoder smudge child inputs are not bound"))?;
        if inputs.inputs() != program.inputs() {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        if self.decoder_smudge.is_some() {
            return Err(RnsBggError::Binding("decoder smudge child is already bound"));
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
        self.decoder_smudge = Some(declaration.clone());
        Ok(())
    }

    /// Attach the bound smudge child of `declaration`'s decoder: its encodings
    /// and the unmask hint are published into this encryption.
    pub fn bind_decoder(
        &mut self,
        declaration: &DecoderDeclaration,
        child: RnsEncryptBuild<'a>,
    ) -> Result<(), ApiError> {
        self.attach_decoder(declaration, child).map_err(ApiError::scheme)
    }

    fn attach_decoder(
        &mut self,
        declaration: &DecoderDeclaration,
        child: RnsEncryptBuild<'a>,
    ) -> Result<(), RnsBggError> {
        if self.decoder_smudge_child.is_some() {
            return Err(RnsBggError::Binding("a decoder is already bound"));
        }
        let binding = child
            .binding
            .as_ref()
            .ok_or(RnsBggError::Binding("decoder smudge child artifacts are not bound"))?;
        if binding.context() != &self.context ||
            binding.parameters() != &self.parameters.parameter_identity() ||
            binding.hash_key_commitment() != &self.hash_key_commitment ||
            binding.program() != declaration.source_program() ||
            binding.decoder_commitment() != Some(declaration.commitment()) ||
            child.inputs.is_none() ||
            child.payload_secret.is_none() ||
            child.mask_secret.is_none() ||
            child.decoder_smudge.as_ref().map(|bound| *bound.commitment()) !=
                Some(*declaration.commitment())
        {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        validate_decoder_smudge_input_key_artifacts(
            &child.manifests,
            child
                .preprocess
                .as_ref()
                .ok_or(RnsBggError::Binding("decoder smudge child preprocess is not bound"))?,
            child.parameters,
            child
                .inputs
                .as_ref()
                .ok_or(RnsBggError::Binding("decoder smudge child inputs are not bound"))?,
            *declaration.commitment(),
        )?;
        // A Dynamic child encrypts under the parent's single secret, in the
        // parent's batch.
        let secret = |build: &RnsEncryptBuild<'a>| {
            build
                .payload_secret
                .as_ref()
                .and_then(|secret| secret.binding.as_ref())
                .map(|binding| binding.name.clone())
        };
        let batch = |build: &RnsEncryptBuild<'a>| build.dynamic_rhs.as_ref().map(|(id, _)| *id);
        if self.parameters.rhs_mode() == RhsMode::Dynamic &&
            (secret(&child) != secret(self) ||
                batch(&child).is_none() ||
                batch(&child) != batch(self))
        {
            return Err(RnsBggError::Binding(
                "a Dynamic decoder smudge child must share the parent's secret and batch",
            ));
        }
        self.decoder_smudge_child = Some(Box::new(child));
        self.decoder_declaration = Some(declaration.clone());
        Ok(())
    }

    /// Publish the encodings of a child Program's inputs (and its LUT
    /// helpers) under the names `name(role)`, with the input keys preprocessing
    /// published as `input_key(wire, port)`; `hash_key` is the child's
    /// sampling key and `program` its Program.
    fn append_child(
        child: &RnsEncryptBuild<'a>,
        program: &crate::program::ExponentLutProgram,
        name: impl Fn(&str) -> String,
        input_key: impl Fn(WireId, usize) -> String,
        mut graph: DslContext,
        hash_key: &Bytes,
    ) -> Result<DslContext, ApiError> {
        let inputs = child
            .inputs
            .as_ref()
            .ok_or_else(|| ApiError::input("decoder smudge child inputs are not bound"))?;
        let production = child
            .preprocess
            .as_ref()
            .ok_or_else(|| ApiError::input("decoder smudge child preprocess is not bound"))?;
        let payload = child
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("decoder smudge child payload is not bound"))?;
        let mask = child
            .mask_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("decoder smudge child mask secret is not bound"))?;
        for (role, family) in
            child.lut_helper_families(mask, hash_key, inputs).map_err(ApiError::scheme)?
        {
            graph = graph.transferred_output(name(&role), family).map_err(ApiError::scheme)?;
        }
        let workload = RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?;
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
                            input_key(*wire, 0),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            input_key(*wire, 1),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            input_key(*wire, 2),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            input_key(*wire, 3),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                ),
            };
            let value = child
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("decoder smudge child plaintext is not bound"))?;
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
                        name(&format!("input-{wire:?}")),
                        encode(key, value.clone()).map_err(ApiError::scheme)?,
                    )
                    .map_err(ApiError::scheme)?,
                InputBinding::Family(plaintexts) => {
                    let key_parts = (0..4)
                        .map(|port| {
                            ring.family_artifact_input(
                                production.clone(),
                                input_key(*wire, port),
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
                        .transferred_output(name(&format!("input-{wire:?}")), family)
                        .map_err(ApiError::scheme)?
                }
            };
        }
        if child.parameters.rhs_mode() == RhsMode::Dynamic {
            let decoder = child
                .decoder_smudge
                .as_ref()
                .ok_or_else(|| ApiError::input("a Dynamic child must be a decoder smudge"))?;
            graph = child.dynamic_material(
                graph,
                hash_key,
                program.id(),
                Some(*decoder.commitment()),
                |role| name(&role),
            )?;
        }
        Ok(graph)
    }

    /// Publish `declaration`'s refresh: every child's encodings under its
    /// namespace and hash key, the one encoding `c_1 = s A_1 - s (I_2 ⊗ G_Q) +
    /// e_1` (a public encoding of `1` at the decoding ring) and the unmask
    /// hint `b = s B + e_b`. `hash_key` is the stage hash key.
    #[allow(clippy::too_many_arguments)]
    fn append_refresh(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        mask_child: Option<&RnsEncryptBuild<'a>>,
        fresh: &RnsEncryptBuild<'a>,
        smudges: &[RnsEncryptBuild<'a>],
        mut graph: DslContext,
        hash_key: &Bytes,
        mask: &RnsBggMaskSecret,
    ) -> Result<DslContext, ApiError> {
        use crate::rns_bgg::refresh::{
            hash_key_input_name, refresh_artifact_name, refresh_fresh_artifact_name,
            refresh_fresh_input_key_artifact_name, refresh_mask_artifact_name,
            refresh_mask_input_key_artifact_name,
        };
        let spec = *declaration.spec();
        let mut child_graph = |graph: DslContext,
                               child: &RnsEncryptBuild<'a>,
                               program: &crate::program::ExponentLutProgram,
                               name: &dyn Fn(&str) -> String,
                               input_key: &dyn Fn(WireId, usize) -> String|
         -> Result<DslContext, ApiError> {
            let input = hash_key_input_name(child.hash_namespace.as_ref());
            let child_key = self
                .parameters
                .ring(0)
                .map_err(ApiError::scheme)?
                .bytes_input(input.clone(), child.hash_key.len());
            let graph = Self::append_child(child, program, name, input_key, graph, &child_key)?;
            merge_runtime_inputs(&mut self.runtime, &child.runtime)?;
            self.runtime.insert(input, RuntimeValue::Bytes(child.hash_key.as_slice().into()));
            self.manifests.extend(child.manifests.clone());
            Ok(graph)
        };
        graph = child_graph(
            graph,
            fresh,
            declaration.fresh().program(),
            &|role| refresh_fresh_artifact_name(spec, role),
            &|wire, port| refresh_fresh_input_key_artifact_name(spec, wire, port),
        )?;
        if let (Some(child), Some(part)) = (mask_child, declaration.top()) {
            graph = child_graph(
                graph,
                child,
                part.mask().program(),
                &|role| refresh_mask_artifact_name(spec, role),
                &|wire, port| refresh_mask_input_key_artifact_name(spec, wire, port),
            )?;
        }
        for (child, decoder) in smudges.iter().zip(declaration.decoders()) {
            let commitment = *decoder.commitment();
            graph = child_graph(
                graph,
                child,
                decoder.smudge().program(),
                &|role| decoder_smudge_artifact_name(commitment, role),
                &|wire, port| decoder_smudge_input_key_artifact_name(commitment, wire, port),
            )?;
        }
        // The decoding ring is the children's last level, with the CRT basis
        // of the target's level (checked at preprocessing).
        let parameters = fresh.parameters;
        let decoding = parameters.levels().len() - 1;
        let one_key = parameters
            .refresh_one_key(decoding, hash_key.clone(), declaration)
            .map_err(ApiError::scheme)?;
        let one = encode_public_inner(
            parameters,
            &RnsBggPublicKey::new(declaration.target_output(), one_key),
            mask,
            parameters.scalar(decoding, 1).map_err(ApiError::scheme)?,
            decoding,
        )
        .map_err(ApiError::scheme)?;
        let hint =
            self.unmask_hint(refresh_artifact_name(spec, "unmask-public"), parameters, decoding)?;
        graph
            .transferred_output(refresh_artifact_name(spec, "one-encoding"), one.row)
            .map_err(ApiError::scheme)?
            .transferred_output(refresh_artifact_name(spec, "unmask-hint"), hint)
            .map_err(ApiError::scheme)
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
        let prefix = super::plaintext_input_prefix(self.hash_namespace.as_ref(), "input");
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
        // A refresh child samples from its derived key; every binding still
        // names the stage key.
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.hash_key = match &self.hash_namespace {
            Some(namespace) => crate::rns_bgg::refresh::child_hash_key(hash_key, namespace),
            None => hash_key.to_vec(),
        };
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

    fn prelude_graph(&mut self) -> Result<Option<StageGraph>, ApiError> {
        if self.parameters.rhs_mode() != RhsMode::Dynamic {
            return Ok(None);
        }
        let graph = DslContext::new("rns-bgg-encrypt-prelude");
        let (mut graph, mut exported) = self.dynamic_family_ciphertexts(graph, |role| role)?;
        let mut runtime = self.runtime.clone();
        if let (Some(declaration), Some(child)) =
            (self.decoder_declaration.as_ref(), self.decoder_smudge_child.as_ref())
        {
            let commitment = *declaration.commitment();
            let (child_graph, child_exported) = child
                .dynamic_family_ciphertexts(graph, |role| {
                    decoder_smudge_artifact_name(commitment, &role)
                })?;
            graph = child_graph;
            exported |= child_exported;
            merge_runtime_inputs(&mut runtime, &child.runtime)?;
        }
        if !exported {
            return Ok(None);
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let runtime = declared_runtime_inputs(&graph, runtime);
        Ok(Some(StageGraph::new(graph, runtime, BTreeMap::new())))
    }

    fn bind_prelude(
        &mut self,
        production: ProductionId,
        manifest: Manifest,
    ) -> Result<(), ApiError> {
        if let Some(child) = self.decoder_smudge_child.as_mut() {
            child.prelude = Some((production.clone(), manifest.clone()));
        }
        self.prelude = Some((production, manifest));
        Ok(())
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
        let hash_key_input =
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref());
        let hash_key = ring.bytes_input(hash_key_input.clone(), self.hash_key.len());
        let mut graph = DslContext::new("rns-bgg-encrypt");
        if let (Some(declaration), Some(child)) =
            (self.decoder_declaration.as_ref(), self.decoder_smudge_child.as_ref())
        {
            let commitment = *declaration.commitment();
            graph = Self::append_child(
                child,
                declaration.smudge().program(),
                |role| decoder_smudge_artifact_name(commitment, role),
                |wire, port| decoder_smudge_input_key_artifact_name(commitment, wire, port),
                graph,
                &hash_key,
            )?;
            // The child's last level is the decoding level `L`, with its CRT
            // basis (checked when the child is bound).
            let decoding = child.parameters.levels().len() - 1;
            let hint = self.unmask_hint(
                decoder_artifact_name(commitment, "unmask-public"),
                child.parameters,
                decoding,
            )?;
            graph = graph
                .transferred_output(
                    decoder_artifact_name(*declaration.commitment(), "unmask-hint"),
                    hint,
                )
                .map_err(ApiError::scheme)?;
            merge_runtime_inputs(&mut self.runtime, &child.runtime)?;
            self.manifests.extend(child.manifests.clone());
        }
        if let Some((declaration, mask_child, fresh, smudges)) = self.refresh.take() {
            graph = self.append_refresh(
                &declaration,
                mask_child.as_deref(),
                &fresh,
                &smudges,
                graph,
                &hash_key,
                &mask,
            )?;
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
        let graph = graph.build().map_err(ApiError::scheme)?;
        if super::graph_declares_input(&graph, &hash_key_input) {
            self.runtime
                .insert(hash_key_input, RuntimeValue::Bytes(self.hash_key.as_slice().into()));
        }
        let mut descriptor =
            self.binding.ok_or_else(|| ApiError::input("artifact binding is not available"))?;
        if let Some((production, manifest)) = self.prelude.clone() {
            self.manifests.insert(production.clone(), manifest.clone());
            descriptor = descriptor.with_ciphertext_prelude(production, manifest);
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
