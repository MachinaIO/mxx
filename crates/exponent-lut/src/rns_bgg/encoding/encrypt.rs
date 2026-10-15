//! The encryption stage adapter.
use super::*;

fn validate_decoder_smudge_input_key_artifacts(
    manifests: &BTreeMap<ProductionId, Manifest>,
    production: &ProductionId,
    parameters: &RnsBggParameters,
    inputs: &crate::program::EncryptionInputs,
    key: &str,
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
            let name = decoder_smudge_input_key_artifact_name(&key, *wire, port);
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
            dynamic_setup: None,
            dynamic_rhs: None,
            prelude: None,
            hash_namespace: None,
            refresh: None,
            inputs_by_shape: false,
            root_rows: None,
        }
    }

    /// Bind an *imported* encryption of `inputs` on a rooted chain (see
    /// [`crate::rns_bgg::RnsBggRoot`]): instead of encoding plaintexts under secrets, the stage
    /// publishes, under the artifact names an encryption uses, `Gamma` of the root row of every
    /// input (its tag in `root_inputs`, read from [`super::root_row_input_name`]) to the input's
    /// level and form, and of the root helper row of every LUT branch automorphism (read from
    /// [`super::root_helper_input_name`]) to the state form of every non-terminal level. A public
    /// input's payload is its plaintext in `lhs`, which holds the public inputs only. The rows
    /// are the encodings of a mask secret no stage knows, so the stage binds no secret.
    pub fn bind_root_rows<P: Poly>(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        root_inputs: BTreeMap<WireId, String>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if self.parameters.root().is_none() || self.parameters.rhs_mode() != RhsMode::Static {
            return Err(ApiError::input("an imported encryption needs a rooted Static chain"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let prefix = super::plaintext_input_prefix(self.hash_namespace.as_ref(), "input");
        for (wire, declaration) in inputs.inputs() {
            if !root_inputs.contains_key(wire) {
                return Err(ApiError::input("an imported input has no root tag"));
            }
            let public = declaration.knowledge() == AttributeKnowledge::PublicPayload;
            match (public, lhs.get(wire)) {
                (true, Some(value)) => {
                    let name = format!("{prefix}/input/{wire:?}");
                    let plaintext = super::plaintext_inputs(&ring, &name, value, &mut self.runtime)
                        .map_err(ApiError::input)?;
                    self.plaintexts.insert(*wire, plaintext);
                }
                (false, None) => {}
                _ => return Err(ApiError::input("lhs must hold exactly the public inputs")),
            }
        }
        self.inputs = Some(inputs.clone());
        self.projection_identity = Some(
            encryption_projection_identity(inputs, parameters, self.parameters)
                .map_err(ApiError::scheme)?,
        );
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.hash_key = hash_key.to_vec();
        self.root_rows = Some(root_inputs);
        Ok(())
    }

    /// Run an imported encryption (see [`Self::bind_root_rows`]) of `inputs` against the
    /// preprocessing `preprocess`, reading the root rows `rows` (graph input values by
    /// [`super::root_row_input_name`] and [`super::root_helper_input_name`]): the stage an online
    /// evaluation binds as its encryption. `before_execute` observes its validated graph.
    #[allow(clippy::too_many_arguments)]
    pub fn import<E, S, P>(
        mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        root_inputs: BTreeMap<WireId, String>,
        rows: BTreeMap<String, RuntimeValue>,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
        hash_key: &[u8],
        execution: &mut E,
        store: &mut S,
        before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
    ) -> Result<StageArtifacts<RnsBggEncryptionDescriptor>, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
        P: Poly,
    {
        self.bind_root_rows(inputs, parameters, lhs, root_inputs, hash_key)?;
        self.bind_preprocess_artifacts_inner(Some(preprocess))?;
        self.runtime.extend(rows);
        // The rows are runtime inputs, so an import is never reused: every run is its own
        // production, whatever rows an earlier import of the same Program published.
        let mut digest = sha2::Sha256::new();
        digest.update(<Self as EncryptBuild<P>>::artifact_identity(&self));
        digest.update(b"import");
        digest.update(rand::random::<[u8; 32]>());
        let identity: [u8; 32] = digest.finalize().into();
        let (graph, descriptor) = <Self as EncryptBuild<P>>::build_graph(self)?;
        crate::api::execute_artifacts(
            graph,
            parameters,
            descriptor,
            identity,
            execution,
            store,
            before_execute,
        )
    }

    /// The imported encryption's published rows (see [`Self::bind_root_rows`]).
    fn imported_rows(
        &self,
        mut graph: DslContext,
        inputs: &crate::program::EncryptionInputs,
        root_inputs: &BTreeMap<WireId, String>,
    ) -> Result<DslContext, ApiError> {
        let root = self
            .parameters
            .root()
            .ok_or_else(|| ApiError::input("an imported encryption needs a rooted chain"))?;
        let root_ring = root.ring();
        let columns = root.columns();
        let retarget = |level: usize, form: crate::rns_bgg::RnsBggForm, row: RnsBggRow| {
            self.parameters.retarget_row(level, form, row).map_err(|_| DslError::Schema)
        };
        for level in 0..self.parameters.levels().len().saturating_sub(1) {
            let ring_dimension =
                self.parameters.level(level).map_err(ApiError::scheme)?.ring_dimension() as usize;
            let tables = inputs
                .luts()
                .iter()
                .map(|table| (table.helper_family_key(), table))
                .collect::<BTreeMap<_, _>>();
            for (key, table) in tables {
                let count = table
                    .branch_automorphisms(ring_dimension)
                    .map_err(|_| ApiError::scheme(RnsBggError::Lut))?
                    .len();
                let ports = [0, 1].map(|port| {
                    root_ring.input_family(
                        super::root_helper_input_name(&key, port),
                        count,
                        (1, columns),
                    )
                });
                let family = mxx_dsl::parallel(count, |index| {
                    retarget(
                        level,
                        crate::rns_bgg::RnsBggForm::State,
                        (ports[0].at(index.clone()), ports[1].at(index)),
                    )
                })
                .map_err(ApiError::scheme)?;
                graph = graph
                    .transferred_output(format!("lut-helper/{level}/{key}"), family)
                    .map_err(ApiError::scheme)?;
            }
        }
        let binding =
            self.binding.as_ref().ok_or_else(|| ApiError::input("preprocess is not bound"))?;
        for (wire, declaration) in inputs.inputs() {
            let level = binding.input_levels().get(wire).copied().unwrap_or(0);
            let tag = &root_inputs[wire];
            let public = declaration.knowledge() == AttributeKnowledge::PublicPayload;
            let form = if public {
                crate::rns_bgg::RnsBggForm::Public
            } else {
                crate::rns_bgg::RnsBggForm::State
            };
            let input_ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            let plaintext_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
            let name = format!("input-{wire:?}");
            let port = |port: usize| super::root_row_input_name(tag, port);
            graph = match declaration.shape() {
                InputBinding::Scalar(()) => {
                    let row = retarget(
                        level,
                        form,
                        (
                            root_ring.input(port(0), (1, columns)),
                            root_ring.input(port(1), (1, columns)),
                        ),
                    )
                    .map_err(ApiError::scheme)?;
                    if public {
                        let Some(InputBinding::Scalar(plaintext)) = self.plaintexts.get(wire)
                        else {
                            return Err(ApiError::input("missing public plaintext"));
                        };
                        let payload = checked_modulus_convert(plaintext.clone(), &input_ring)
                            .map_err(ApiError::scheme)?;
                        graph.transferred_output(name, (row.0, row.1, payload))
                    } else {
                        graph.transferred_output(name, row)
                    }
                }
                InputBinding::Family(count) => {
                    let ports = [0, 1].map(|index| {
                        root_ring.input_family(port(index), count.clone(), (1, columns))
                    });
                    if public {
                        let Some(InputBinding::Family(plaintexts)) = self.plaintexts.get(wire)
                        else {
                            return Err(ApiError::input("missing public plaintexts"));
                        };
                        let stored = mxx_dsl::parallel(count.clone(), |index| {
                            let row = retarget(
                                level,
                                form,
                                (ports[0].at(index.clone()), ports[1].at(index.clone())),
                            )?;
                            let payload = checked_modulus_convert(
                                plaintexts.member(&plaintext_ring, index),
                                &input_ring,
                            )
                            .map_err(|_| DslError::Schema)?;
                            Ok((row.0, row.1, payload))
                        })
                        .map_err(ApiError::scheme)?;
                        graph.transferred_output(name, stored)
                    } else {
                        let stored = mxx_dsl::parallel(count.clone(), |index| {
                            retarget(level, form, (ports[0].at(index.clone()), ports[1].at(index)))
                        })
                        .map_err(ApiError::scheme)?;
                        graph.transferred_output(name, stored)
                    }
                }
            }
            .map_err(ApiError::scheme)?;
        }
        Ok(graph)
    }

    /// This builder binding the inputs' plaintexts by their declared shapes only, whatever the
    /// stage ([`crate::api::encrypt`]) is given, as [`Self::bind_lhs_inputs`] with `None` does:
    /// the stage graph is recorded and exported but cannot be executed.
    pub fn with_inputs_by_shape(mut self) -> Self {
        self.inputs_by_shape = true;
        self
    }

    /// Make this builder a refresh child that samples its public matrices
    /// from the stage hash key derived under `namespace`.
    pub fn with_hash_namespace(mut self, namespace: [u8; 32]) -> Self {
        self.hash_namespace = Some(namespace);
        self
    }

    /// Attach the children of part `part` (`(chain, component)`) of `declaration`'s refresh,
    /// each bound to its inputs (the PRF label and `mono(0)`), the secrets and the source's
    /// preprocess artifacts: `fresh` the fresh-noise child (under
    /// [`RefreshDeclaration::fresh_namespace`]) and `smudge` the decoder's smudge child. Their
    /// encodings, the one encoding and the unmask hint are published into this encryption.
    pub fn bind_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        part: (usize, usize),
        fresh: RnsEncryptBuild<'a>,
        smudge: RnsEncryptBuild<'a>,
    ) -> Result<(), ApiError> {
        let declaration =
            crate::rns_bgg::refresh::refresh_part(declaration, part).map_err(ApiError::scheme)?;
        let mismatch = || ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch));
        if self.decoder_smudge_child.is_some() || self.refresh.is_some() {
            return Err(ApiError::input("a decoder or refresh was already bound"));
        }
        if self.parameters.rhs_mode() != RhsMode::Static {
            return Err(mismatch());
        }
        let bound = |child: &RnsEncryptBuild<'a>, namespace: [u8; 32]| {
            child.binding.as_ref().is_some_and(|binding| {
                binding.context() == &self.context &&
                    binding.hash_key_commitment() == &self.hash_key_commitment &&
                    binding.program() == Some(declaration.source_program()) &&
                    binding.decoder_commitment() == Some(declaration.spec())
            }) && child.inputs.is_some() &&
                child.payload_secret.is_some() &&
                child.mask_secret.is_some() &&
                child.hash_namespace == Some(namespace)
        };
        let decoder = declaration.decoder();
        if !bound(&fresh, declaration.fresh_namespace()) ||
            fresh.inputs.as_ref().map(|inputs| inputs.inputs()) !=
                Some(declaration.fresh().program().inputs()) ||
            !bound(&smudge, *decoder.commitment()) ||
            smudge.decoder_smudge.as_ref().map(|bound| *bound.commitment()) !=
                Some(*decoder.commitment())
        {
            return Err(mismatch());
        }
        self.refresh = Some((declaration.clone(), Box::new(fresh), Box::new(smudge)));
        Ok(())
    }

    /// Bind public LHS plaintexts `lhs` and secrets for a decoder smudge or
    /// refresh child. This application-facing method keeps the generic stage
    /// trait out of child setup code. Dynamic children take no mask.
    ///
    /// `None` binds the plaintexts by the inputs' declared shapes only (see
    /// [`super::super::RnsPreprocessBuild::bind_rhs_inputs`]): the graph is
    /// the one of any plaintexts of those shapes, which can be validated and
    /// exported but not executed.
    pub fn bind_lhs_inputs<P: Poly>(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: Option<&PlaintextInputs<WireId, P>>,
        payload: &RnsBggPayloadSecret,
        mask: Option<&RnsBggMaskSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        self.bind_encryption_inputs(inputs, parameters, lhs, payload, mask, hash_key)
    }

    fn bind_preprocess_artifacts_inner(
        &mut self,
        preprocess: Option<&StageArtifacts<RnsBggPreprocessEncryptDescriptor>>,
    ) -> Result<(), ApiError> {
        if self.inputs.is_none() {
            return Err(ApiError::input("encryption inputs are not bound"));
        }
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        let Some(preprocess) = preprocess else {
            // A Dynamic encryption belongs to its setup, not to a Program.
            if !dynamic {
                return Err(ApiError::input("Static encryption needs its preprocess artifacts"));
            }
            let setup = self.bound_dynamic_setup()?;
            setup
                .validate_public_seed(&self.context, &self.hash_key_commitment)
                .map_err(ApiError::scheme)?;
            self.binding = Some(
                RnsBggArtifactBinding::new(
                    self.context,
                    None,
                    self.parameters.parameter_identity(),
                    self.hash_key_commitment,
                    Vec::new(),
                    RhsMode::Dynamic,
                )
                .map_err(ApiError::scheme)?
                .with_dynamic_setup(setup.identity(), None),
            );
            return Ok(());
        };
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
        // A Dynamic encryption reads a preprocessing production only for an
        // applied decoder, whose unmask matrix that production samples.
        if dynamic && self.decoder_smudge.is_none() && binding.decoder_commitment().is_none() {
            return Err(ApiError::compatibility(
                "Dynamic encryption binds no Program preprocessing",
            ));
        }
        self.preprocess = Some(production.clone());
        self.manifests = BTreeMap::from([(production.clone(), preprocess.manifest().clone())]);
        self.binding = Some(binding);
        Ok(())
    }

    /// Bind the preprocess artifacts used by a decoder smudge child.
    pub fn bind_preprocess_artifacts(
        &mut self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ) -> Result<(), ApiError> {
        self.bind_preprocess_artifacts_inner(Some(preprocess))
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
        // A Dynamic child encrypts the packages of its own setup.
        if self.parameters.rhs_mode() == RhsMode::Dynamic && self.dynamic_setup.is_none() {
            return Err(RnsBggError::Binding("a Dynamic decoder smudge child needs its setup"));
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
            binding.program() != Some(declaration.source_program()) ||
            binding.decoder_commitment() != Some(declaration.commitment()) ||
            child.inputs.is_none() ||
            child.payload_secret.is_none() ||
            child.mask_secret.is_none() ||
            child.decoder_smudge.as_ref().map(|bound| *bound.commitment()) !=
                Some(*declaration.commitment())
        {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        // A Static child's input keys are published by preprocessing; a
        // Dynamic child's are its setup's.
        if self.parameters.rhs_mode() == RhsMode::Static {
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
                &declaration.artifact_key(),
            )?;
        }
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
        if child.parameters.rhs_mode() == RhsMode::Dynamic {
            return child.dynamic_material(graph, hash_key, |role| name(&role));
        }
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
        for (role, family) in child
            .lut_helper_families(
                mask,
                hash_key,
                child.parameters.root_hash_key_input().as_ref(),
                inputs,
            )
            .map_err(ApiError::scheme)?
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
        Ok(graph)
    }

    /// Publish the refresh part `declaration`: every child's encodings under its
    /// namespace and hash key, the one encoding `c_1 = s A_1 - s (I_2 ⊗ G_Q) +
    /// e_1` (a public encoding of `1` at the decoding ring) and the unmask
    /// hint `b = s B + e_b`. `hash_key` is the stage hash key.
    #[allow(clippy::too_many_arguments)]
    fn append_refresh(
        &mut self,
        declaration: &RefreshPartDeclaration,
        fresh: &RnsEncryptBuild<'a>,
        smudge: &RnsEncryptBuild<'a>,
        mut graph: DslContext,
        hash_key: &Bytes,
        mask: &RnsBggMaskSecret,
    ) -> Result<DslContext, ApiError> {
        use crate::rns_bgg::refresh::{
            hash_key_input_name, refresh_artifact_name, refresh_fresh_artifact_name,
            refresh_fresh_input_key_artifact_name,
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
        let decoder = declaration.decoder();
        let key = decoder.artifact_key();
        graph = child_graph(
            graph,
            smudge,
            decoder.smudge().program(),
            &|role| decoder_smudge_artifact_name(&key, role),
            &|wire, port| decoder_smudge_input_key_artifact_name(&key, wire, port),
        )?;
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
            &parameters.encoding_contract(),
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
        encode_private_inner(
            self.parameters,
            key,
            payload,
            mask,
            plaintext,
            level,
            &self.parameters.encoding_contract(),
        )
    }

    fn encode_public(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_public_inner(
            self.parameters,
            key,
            mask,
            plaintext,
            level,
            &self.parameters.encoding_contract(),
        )
    }

    /// The LUT helpers, one family per level and LUT input layout, whose
    /// member `i` belongs to the layout's `i`-th automorphism: its exports
    /// stream from a slot per lane rather than a slot per helper.
    fn lut_helper_families(
        &self,
        mask: &RnsBggMaskSecret,
        hash_key: &Bytes,
        root_hash_key: Option<&Bytes>,
        inputs: &crate::program::EncryptionInputs,
    ) -> Result<Vec<(String, Family<RnsBggRow>)>, RnsBggError> {
        let mut families = Vec::new();
        for level in 0..self.parameters.levels().len().saturating_sub(1) {
            let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
            let tables = inputs
                .luts()
                .iter()
                .map(|table| (table.helper_family_key(), table))
                .collect::<BTreeMap<_, _>>();
            for (name, table) in tables {
                let sigmas =
                    table.branch_automorphisms(ring_dimension).map_err(|_| RnsBggError::Lut)?;
                let family = mxx_dsl::parallel(sigmas.len(), |index| {
                    let sigma = IntExpr::Select {
                        selector: Box::new(index.expression()?),
                        branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
                    };
                    let public_data = super::public_key::lut_public_matrix(
                        self.parameters,
                        hash_key.clone(),
                        root_hash_key.cloned(),
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

impl RnsEncryptBuild<'_> {
    /// Bind the encryption inputs: the plaintexts `lhs`, or, for `None`,
    /// their declared shapes only (see [`Self::bind_lhs_inputs`]).
    fn bind_encryption_inputs<P: Poly>(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: Option<&PlaintextInputs<WireId, P>>,
        payload: &RnsBggPayloadSecret,
        mask: Option<&RnsBggMaskSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let mask = stage_mask(self.parameters, payload, mask)?;
        // A Dynamic encryption's inputs are its setup's input coordinates
        // (see [`RnsBggDynamicSetup::encryption_inputs`]).
        if self.parameters.rhs_mode() == RhsMode::Dynamic &&
            self.bound_dynamic_setup()?.encryption_inputs() != *inputs
        {
            return Err(ApiError::input("Dynamic encryption inputs are not the setup's"));
        }
        if let Some(lhs) = lhs &&
            (lhs.len() != inputs.inputs().len() ||
                lhs.keys().any(|wire| !inputs.inputs().contains_key(wire)))
        {
            return Err(ApiError::input("encryption inputs do not exactly match Program inputs"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let prefix = super::plaintext_input_prefix(self.hash_namespace.as_ref(), "input");
        for (wire, declaration) in inputs.inputs() {
            let concrete =
                |count: &IntExpr| {
                    count.evaluate(parameters).ok().and_then(|value| value.to_usize()).ok_or_else(
                        || ApiError::input("encryption input family count is not concrete"),
                    )
                };
            let name = format!("{prefix}/input/{wire:?}");
            let plaintext = match lhs {
                Some(lhs) => {
                    let value =
                        lhs.get(wire).ok_or_else(|| ApiError::input("missing encryption input"))?;
                    match (declaration.shape(), value) {
                        (InputBinding::Scalar(()), InputBinding::Scalar(_)) => {}
                        (InputBinding::Family(count), InputBinding::Family(values)) => {
                            if values.len() != concrete(count)? {
                                return Err(ApiError::input(
                                    "encryption input family count does not match Program",
                                ));
                            }
                        }
                        _ => {
                            return Err(ApiError::input(
                                "encryption input shape does not match Program",
                            ));
                        }
                    }
                    super::plaintext_inputs(&ring, &name, value, &mut self.runtime)
                }
                None => {
                    let shape = match declaration.shape() {
                        InputBinding::Scalar(()) => InputBinding::Scalar(()),
                        InputBinding::Family(count) => InputBinding::Family(concrete(count)?),
                    };
                    super::plaintext_handles(&ring, &name, shape)
                }
            }
            .map_err(ApiError::input)?;
            self.plaintexts.insert(*wire, plaintext);
        }
        self.inputs = Some(inputs.clone());
        // The projection a Static encryption is bound to includes the LUT
        // values; a Dynamic one is its setup, which names no table.
        self.projection_identity = Some(match &self.dynamic_setup {
            Some(setup) => setup.identity(),
            None => encryption_projection_identity(inputs, parameters, self.parameters)
                .map_err(ApiError::scheme)?,
        });
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
        let lhs = (!self.inputs_by_shape).then_some(lhs);
        self.bind_encryption_inputs(inputs, parameters, lhs, payload, mask, hash_key)
    }

    fn bind_artifacts(
        &mut self,
        preprocess: Option<&StageArtifacts<Self::PreprocessDescriptor>>,
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
        // A Dynamic session selects its own sampling session, and a bound
        // prelude production is part of the material it encrypts.
        if let Some((session, _)) = &self.dynamic_rhs {
            digest.update(b"session");
            digest.update(session);
        }
        if let Some((production, _)) = &self.prelude {
            digest.update(b"prelude");
            digest.update(production.spec_hash.0);
            digest.update(production.execution_nonce);
        }
        digest.finalize().into()
    }

    fn prelude_graph(&mut self) -> Result<Option<StageGraph>, ApiError> {
        if self.parameters.rhs_mode() != RhsMode::Dynamic {
            return Ok(None);
        }
        // A bound prelude production (an enclosing scheme's) exports the
        // bottom rows of every coordinate; otherwise the encryption's own
        // prelude samples them from the bound plaintexts.
        let plaintexts = self
            .bound_dynamic_rhs()?
            .values()
            .any(|value| matches!(value, DynamicRhsValue::Plaintext(_)));
        if self.prelude.is_some() {
            if plaintexts {
                return Err(ApiError::input(
                    "RHS plaintexts need the encryption's own prelude production",
                ));
            }
            return Ok(None);
        }
        if self.bound_dynamic_rhs()?.values().any(|value| matches!(value, DynamicRhsValue::Prelude))
        {
            return Err(ApiError::input(
                "an RHS without a plaintext needs a bound prelude production",
            ));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let hash_key = ring.bytes_input(super::HASH_KEY_INPUT, self.hash_key.len());
        let graph = DslContext::new("rns-bgg-encrypt-prelude");
        let (mut graph, mut exported) = self.dynamic_rhs_bottoms(graph, &hash_key, |role| role)?;
        let mut runtime = self.runtime.clone();
        if let (Some(declaration), Some(child)) =
            (self.decoder_declaration.as_ref(), self.decoder_smudge_child.as_ref())
        {
            let key = declaration.artifact_key();
            let (child_graph, child_exported) =
                child.dynamic_rhs_bottoms(graph, &hash_key, |role| {
                    decoder_smudge_artifact_name(&key, &role)
                })?;
            graph = child_graph;
            exported |= child_exported;
            merge_runtime_inputs(&mut runtime, &child.runtime)?;
        }
        if !exported {
            return Ok(None);
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        if super::graph_declares_input(&graph, super::HASH_KEY_INPUT) {
            runtime.insert(
                super::HASH_KEY_INPUT.to_owned(),
                RuntimeValue::Bytes(self.hash_key.as_slice().into()),
            );
        }
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
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        if self.binding.is_none() || (!dynamic && self.preprocess.is_none()) {
            return Err(ApiError::input("preprocess artifacts are not bound"));
        }
        if let Some(root_inputs) = self.root_rows.take() {
            if self.decoder_smudge_child.is_some() || self.refresh.is_some() {
                return Err(ApiError::input("an imported encryption binds no decoder or refresh"));
            }
            let graph = self
                .imported_rows(DslContext::new("rns-bgg-encrypt"), &inputs, &root_inputs)?
                .build()
                .map_err(ApiError::scheme)?;
            if let Some(root) = self.parameters.root() {
                self.runtime.insert(
                    root.hash_key_input_name(),
                    RuntimeValue::Bytes(root.hash_key().as_slice().into()),
                );
            }
            let descriptor =
                self.binding.ok_or_else(|| ApiError::input("artifact binding is not available"))?;
            let runtime = declared_runtime_inputs(&graph, self.runtime);
            return Ok((
                StageGraph::new(graph, runtime, self.manifests),
                RnsBggEncryptionDescriptor::from_binding(descriptor),
            ));
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
            let key = declaration.artifact_key();
            graph = Self::append_child(
                child,
                declaration.smudge().program(),
                |role| decoder_smudge_artifact_name(&key, role),
                |wire, port| decoder_smudge_input_key_artifact_name(&key, wire, port),
                graph,
                &hash_key,
            )?;
            // The child's last level is the decoding level `L`, with its CRT
            // basis (checked when the child is bound).
            let decoding = child.parameters.levels().len() - 1;
            let hint = self.unmask_hint(
                decoder_artifact_name(&key, "unmask-public"),
                child.parameters,
                decoding,
            )?;
            graph = graph
                .transferred_output(
                    decoder_artifact_name(&declaration.artifact_key(), "unmask-hint"),
                    hint,
                )
                .map_err(ApiError::scheme)?;
            merge_runtime_inputs(&mut self.runtime, &child.runtime)?;
            self.manifests.extend(child.manifests.clone());
        }
        if let Some((declaration, fresh, smudge)) = self.refresh.take() {
            graph = self.append_refresh(&declaration, &fresh, &smudge, graph, &hash_key, &mask)?;
        }
        if dynamic {
            graph = self.dynamic_material(graph, &hash_key, |role| role)?;
        } else {
            for (name, family) in self
                .lut_helper_families(
                    &mask,
                    &hash_key,
                    self.parameters.root_hash_key_input().as_ref(),
                    &inputs,
                )
                .map_err(ApiError::scheme)?
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
                let encode = |member: Option<Int>, plaintext: Mat| {
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
                            let encoding = encode(Some(index.clone()), plaintext)
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
                            encode(Some(index.clone()), plaintext)
                                .map(|encoding| encoding.row)
                                .map_err(|_| DslError::Schema)
                        })
                        .map_err(ApiError::scheme)?;
                        graph.transferred_output(name, stored)
                    }
                }
                .map_err(ApiError::scheme)?;
            }
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        if super::graph_declares_input(&graph, &hash_key_input) {
            self.runtime
                .insert(hash_key_input, RuntimeValue::Bytes(self.hash_key.as_slice().into()));
        }
        if let Some(root) = self.parameters.root() {
            self.runtime.insert(
                root.hash_key_input_name(),
                RuntimeValue::Bytes(root.hash_key().as_slice().into()),
            );
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

    /// Rows encrypted at the root modulus and imported through `Gamma` (see
    /// [`crate::rns_bgg::RnsBggRoot`]) evaluate as encryptions do: a private LUT followed by a
    /// public product, noise-free, ending at the terminal level, on a chain whose level 0 is the
    /// root (every `Gamma` is the identity) and on a chain of fewer primes (every `Gamma`
    /// ModDowns).
    #[test]
    fn imported_root_rows_evaluate_as_encryptions() {
        use crate::{
            program::{
                ExpLayout, ExponentLutProgramBuilder, LutOutputRepresentation, LutTable, Operand,
            },
            rns_bgg::{RnsBggRoot, RnsOutputUse, RnsPreprocessBuild, verify_output},
        };
        use mxx_backends::{
            artifact::MemoryArtifactStore,
            authority::CpuExecution,
            backend::poly::cpu_backend,
            poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        };
        let n = 16u32;
        let primes = crate::test_support::primes(n, 4, 17);
        let pool = DCRTPolyParams::try_new(n, 4, 17, 4, Some(primes.clone()), None).unwrap();
        let select = |count: usize| {
            pool.select_modulus(
                &primes[..count].iter().map(|prime| BigUint::from(*prime)).product(),
            )
            .unwrap()
        };
        let root = RnsBggRoot::new(pool.clone(), rand::random());
        let root_parameters = RnsBggParameters::new(vec![pool.clone()], 0, 1).unwrap();
        let layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        let table = vec![2usize, 0, 3, 1];
        let mut builder = ExponentLutProgramBuilder::new();
        let x = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let alpha = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let lut = builder
            .register_lut(
                LutTable::unary(
                    4,
                    layout,
                    4,
                    table.clone(),
                    LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let looked = builder.lut(Operand::Scalar(x), lut).unwrap();
        let product = builder.mul_public(Operand::Scalar(alpha), Operand::Scalar(looked)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(product)]).unwrap();
        let tags = BTreeMap::from([(x, "x".to_owned()), (alpha, "alpha".to_owned())]);
        // `mono(e) = X^(2e)` in the negacyclic ring of dimension 16.
        let monomial = |parameters: &DCRTPolyParams, e: usize| {
            let rotation = 2 * (e % 16);
            let value = DCRTPoly::const_rotate_poly(parameters, rotation % 16);
            if rotation >= 16 { -value } else { value }
        };
        for depths in [vec![4, 4], vec![3, 2]] {
            let parameters =
                RnsBggParameters::new(depths.iter().map(|d| select(*d)).collect(), 0, 1)
                    .unwrap()
                    .with_root(root.clone())
                    .unwrap();
            let mut execution = CpuExecution::new(cpu_backend(
                parameters.levels().iter().cloned().chain(std::iter::once(pool.clone())),
            ));
            let mut store = MemoryArtifactStore::default();
            let (hash_key, context) = (rand::random::<[u8; 32]>(), rand::random::<[u8; 32]>());
            let payload = RnsEncryptBuild::new(&root_parameters, context)
                .sample_payload_secret(&mut execution, &mut store, |_| Ok(()))
                .unwrap();
            let mask = RnsEncryptBuild::new(&root_parameters, rand::random())
                .sample_mask_secret(&mut execution, &mut store, |_| Ok(()))
                .unwrap();
            let (preprocess, online_eval) = crate::api::preprocess(
                &program,
                &ParamEnv::default(),
                &PlaintextInputs::<crate::program::RhsId, DCRTPoly>::new(),
                Some(&payload),
                &hash_key,
                RnsPreprocessBuild::new(&parameters, context).with_root_inputs(tags.clone()),
                &mut execution,
                &mut store,
                |_, outputs| Ok(outputs),
                |_| Ok(()),
            )
            .unwrap();
            // The root rows of the inputs and of every helper, noise-free, at the root.
            let (x_value, alpha_value) =
                (rand::random::<u8>() as usize % 4, rand::random::<u8>() as usize % 8);
            let ring = root.ring();
            let mono = |e: usize| ring.identity(1).multiply_monomial(Int::constant(2 * e as i64));
            let zero = || (ring.zero((1, root.columns())), ring.zero((1, root.columns())));
            let (s, t) = (mask.row().clone(), payload.row().clone());
            let root_hash_key = root.hash_key_input();
            let input = |tag: &str, target: (Mat, Mat)| {
                let key = root.key(root_hash_key.clone(), RnsBggRoot::input_tag(tag, None));
                root.encode(&key, &s, target, zero()).unwrap()
            };
            let x_row = input("x", (mono(x_value) * t.0.clone(), mono(x_value) * t.1.clone()));
            let alpha_row =
                input("alpha", (mono(alpha_value) * s.0.clone(), mono(alpha_value) * s.1.clone()));
            let mut graph = DslContext::new("root-rows");
            for (tag, row) in [("x", x_row), ("alpha", alpha_row)] {
                graph = graph
                    .output(root_row_input_name(tag, 0), row.0)
                    .unwrap()
                    .output(root_row_input_name(tag, 1), row.1)
                    .unwrap();
            }
            for table in program.encryption_inputs().luts() {
                let sigmas = table.branch_automorphisms(n as usize).unwrap();
                for port in 0..2 {
                    let family = mxx_dsl::parallel(sigmas.len(), |index| {
                        let sigma = IntExpr::Select {
                            selector: Box::new(index.expression()?),
                            branches: sigmas
                                .iter()
                                .map(|sigma| IntExpr::constant(*sigma))
                                .collect(),
                        };
                        let key = root.key(
                            root_hash_key.clone(),
                            RnsBggRoot::helper_tag(Int::from(sigma.clone())),
                        );
                        let target = (s.0.clone().ring_automorphism(sigma), ring.identity(1));
                        let row =
                            root.encode(&key, &s, target, zero()).map_err(|_| DslError::Schema)?;
                        Ok(if port == 0 { row.0 } else { row.1 })
                    })
                    .unwrap();
                    graph = graph
                        .output(root_helper_input_name(&table.helper_family_key(), port), family)
                        .unwrap();
                }
            }
            let mut runtime = BTreeMap::from([(
                root.hash_key_input_name(),
                RuntimeValue::Bytes(root.hash_key().as_slice().into()),
            )]);
            payload.bind_runtime(&mut runtime);
            mask.bind_runtime(&mut runtime);
            let graph = graph.build().unwrap();
            let runtime = declared_runtime_inputs(&graph, runtime);
            let rows = crate::api::execute_graph(
                StageGraph::new(graph, runtime, BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap()
            .outputs;
            let lhs = BTreeMap::from([(
                alpha,
                InputBinding::Scalar(monomial(&parameters.levels()[0], alpha_value)),
            )]);
            let encrypted = RnsEncryptBuild::new(&parameters, context)
                .import(
                    &program.encryption_inputs(),
                    &ParamEnv::default(),
                    &lhs,
                    tags.clone(),
                    rows,
                    &preprocess,
                    &hash_key,
                    &mut execution,
                    &mut store,
                    |_| Ok(()),
                )
                .unwrap();
            let result = crate::api::online_eval(
                &program,
                &ParamEnv::default(),
                &online_eval,
                &encrypted,
                &hash_key,
                RnsOnlineEvalBuild::new(&parameters, context).with_root_inputs(tags.clone()),
                &mut execution,
                &mut store,
                |_, outputs| Ok(outputs),
                |_| Ok(()),
            )
            .unwrap();
            let terminal = parameters.levels().last().unwrap();
            let expected = monomial(terminal, alpha_value + table[x_value]);
            let verify = |value: &DCRTPoly, execution: &mut CpuExecution, store: &mut _| {
                verify_output(
                    &program,
                    &result,
                    0,
                    value,
                    &parameters,
                    &payload,
                    Some(&mask),
                    RnsOutputUse::DirectDecode,
                    execution,
                    store,
                )
            };
            verify(&expected, &mut execution, &mut store)
                .unwrap_or_else(|error| panic!("depths {depths:?}: {error:?}"));
            let wrong = monomial(terminal, alpha_value + table[x_value] + 1);
            assert!(verify(&wrong, &mut execution, &mut store).is_err());
        }
    }

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
