//! The preprocessing stage adapter.
use super::*;

/// Classify a lowered public-key carrier for the preprocess artifact graph.
///
/// Raw input keys are sampled only from the hash key, so they are cacheable
/// regardless of the payload knowledge attached to the input declaration.
/// Derived public-payload wires remain hash-key/public-only, while private
/// payload wires may depend on transferred secret or private-plaintext
/// material.
fn preprocess_output_availability(
    program: &ExponentLutProgram,
    wire: WireId,
) -> Result<ArtifactAvailability, RnsBggError> {
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
            artifact_prefix: None,
            rhs_values: None,
            parameters,
            context,
            hash_key_commitment: [0; 32],
            evaluator: None,
            families: super::encoding::OnlineEvalFamilies::new(parameters),
            environment: Environment::new(),
            runtime: BTreeMap::new(),
            output_use: RnsOutputUse::default(),
            dynamic: None,
            data: crate::program::ProgramData::new(),
            graph_data: None,
            hash_namespace: None,
            inputs_by_shape: false,
            root_inputs: BTreeMap::new(),
        }
    }

    /// Namespace this instance's inputs and artifacts without changing its sampling seed.
    /// Configure before binding; choose distinct, nonempty prefixes for Programs in one context.
    pub fn with_artifact_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.artifact_prefix = Some(prefix.into());
        self
    }

    /// Bind Static RHS plaintext expressions from the caller's graph at level zero.
    /// Families keep their symbolic counts and are read inside graph loops. This adds
    /// no plaintext inputs and does not execute encryption or online evaluation.
    pub fn bind_rhs_values(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        rhs: BTreeMap<RhsId, InputBinding<Mat, super::RhsPlaintexts>>,
        payload: Option<&RnsBggPayloadSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if self.evaluator.is_some() {
            return Err(ApiError::input("preprocessing is already bound"));
        }
        if self.parameters.rhs_mode() != RhsMode::Static {
            return Err(ApiError::input("graph RHS plaintexts require Static mode"));
        }
        if rhs.len() != program.rhs_inputs().len() ||
            rhs.keys().any(|id| !program.rhs_inputs().contains_key(id))
        {
            return Err(ApiError::input("RHS values do not exactly match Program inputs"));
        }
        self.rhs_values = Some(rhs);
        self.bind_program_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
            program, parameters, None, payload, hash_key,
        )
    }

    /// This builder reading every input of `root_inputs` from the root object of its tag on
    /// a rooted chain (see [`crate::rns_bgg::RnsBggRoot`]): its key is the re-targeted root
    /// key of the tag. Online evaluation must be given the same tags.
    pub fn with_root_inputs(mut self, root_inputs: BTreeMap<WireId, String>) -> Self {
        self.root_inputs = root_inputs;
        self
    }

    /// The key matrices of a family exported by `lower_deferred_exporting` that was lowered as
    /// one graph family (a materialized Program family), as a family of their own: a
    /// projection of the lowered family, so the caller can publish it without reading its
    /// members. `None` for an exported input family.
    pub fn exported_keys(
        family: &super::lowering::PreprocessWireFamily,
    ) -> Result<Option<Family<RnsBggMatrix>>, ApiError> {
        match family {
            super::lowering::PreprocessWireFamily::Stored(family) => Ok(Some(
                family
                    .field(|key| key.matrix)
                    .map_err(|error| ApiError::input(error.to_string()))?,
            )),
            super::lowering::PreprocessWireFamily::Input { .. } => Ok(None),
        }
    }

    /// This builder lowering the member reads of each input family of `bases` at an index that
    /// depends on the loop index, in an eager parallel family, as gathers by outer index
    /// families ([`crate::program::FamilyExecution::gathers_input_reads`]), so that the members
    /// of an input family bound to artifacts are read ahead of the lanes that consume them. A
    /// gathered read of family `wire` reads member `base + index` of it, for a family of every
    /// instance's members side by side (base `0` otherwise): the member its root key or row
    /// source receives already holds the base. Such a family is read only by gathers.
    pub fn with_gather_bases(mut self, bases: BTreeMap<WireId, Int>) -> Self {
        self.families.gather_bases = bases;
        self
    }

    /// Bind the variables (`IntExpr::Var`) the Program's integer expressions read: a Program
    /// lowered in the body of an outer graph loop reads that loop's index as a variable, so one
    /// Program serves every iteration.
    pub fn with_variables(mut self, variables: BTreeMap<String, IntExpr>) -> Self {
        self.families.variables = variables;
        self
    }

    /// Make this builder sample its public matrices from the stage hash key
    /// derived under `namespace`, so that Programs of one shape preprocessed
    /// next to each other never share a hash-derived matrix.
    pub fn with_hash_namespace(mut self, namespace: [u8; 32]) -> Self {
        self.hash_namespace = Some(namespace);
        self
    }

    /// This builder for a Program with integer data ([`crate::program::ProgramData`]), which
    /// preprocessing reads as graph inputs and commits to.
    pub fn with_data(mut self, data: crate::program::ProgramData) -> Self {
        self.data = data;
        self
    }

    /// Bind the Program's integer data families to graph values of the composing graph (for
    /// example a member's row selected from a larger family by a loop index), before the RHS
    /// binding. They replace [`Self::with_data`]: preprocessing declares no data input and
    /// commits to no concrete entries, so an online evaluation binds its data the same way.
    /// `data` must cover exactly the declared families with their declared counts.
    pub fn bind_data_values(
        &mut self,
        program: &ExponentLutProgram,
        data: BTreeMap<crate::program::DataId, (Family<Int>, Int)>,
    ) -> Result<(), ApiError> {
        if self.evaluator.is_some() {
            return Err(ApiError::input("bind graph data before the RHS"));
        }
        let (data, offsets) = crate::program::validate_data_values(program, data)
            .map_err(|error| ApiError::input(error.to_string()))?;
        self.graph_data = Some(data);
        self.families.data_offsets = offsets;
        Ok(())
    }

    /// This builder for compiling a Program against the Dynamic `setup`,
    /// with its inputs and RHS mapped to setup coordinates by `binding`.
    /// The binding records the setup and the mapping, so online evaluation
    /// accepts only ciphertexts of the same setup.
    pub fn with_dynamic_setup(
        mut self,
        setup: &RnsBggDynamicSetup,
        binding: RnsBggDynamicBinding,
    ) -> Result<Self, RnsBggError> {
        setup.validate_parameters(self.parameters)?;
        if setup.context() != &self.context {
            return Err(RnsBggError::Binding("the Dynamic setup has another context"));
        }
        self.dynamic = Some((Arc::new(setup.clone()), binding));
        Ok(self)
    }

    /// This builder binding the Program's RHS plaintexts and data by their declared shapes only,
    /// whatever the stage ([`crate::api::preprocess`]) is given, as [`Self::bind_rhs_inputs`]
    /// with `None` does: the stage graph is recorded and exported but cannot be executed.
    /// Family cardinalities stay arbitrary-precision; no host-addressable selector vector is
    /// needed to declare or lower the graph.
    pub fn with_inputs_by_shape(mut self) -> Self {
        self.inputs_by_shape = true;
        self
    }

    /// This builder with the output use `output_use` (see [`RnsOutputUse`]).
    /// The preprocessing binding records it, and online evaluation and noise
    /// simulation must select the same use.
    pub fn with_output_use(mut self, output_use: RnsOutputUse) -> Self {
        self.output_use = output_use;
        self
    }

    /// Bind RHS plaintexts `rhs` and the builder's Program data (see
    /// [`Self::with_data`]) without exposing the generic stage trait at
    /// application call sites. Dynamic Programs bind no RHS plaintexts and no
    /// secret.
    ///
    /// `None` binds the RHS plaintexts and the data by the Program's declared
    /// shapes only: the stage graph is the one of any plaintexts and data of
    /// those shapes (graph inputs of the same names and counts), so it can be
    /// validated, recorded and exported without materializing one selector
    /// plaintext and table entry per key member, but not executed (the inputs
    /// have no values).
    pub fn bind_rhs_inputs<P: Poly>(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        rhs: Option<&PlaintextInputs<RhsId, P>>,
        payload: Option<&RnsBggPayloadSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        self.bind_program_inputs(program, parameters, rhs, payload, hash_key)
    }
}

impl<'a> RnsPreprocessBuild<'a> {
    /// Bind `program`'s inputs: its RHS plaintexts are `values` and its data
    /// the builder's (see [`Self::with_data`]), or, for `None`, both are bound
    /// by their declared shapes only (see [`Self::bind_rhs_inputs`]).
    fn bind_program_inputs<P: Poly>(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        values: Option<&PlaintextInputs<RhsId, P>>,
        payload: Option<&RnsBggPayloadSecret>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if self.evaluator.is_some() || self.artifact_prefix.as_deref() == Some("") {
            return Err(ApiError::input(
                "preprocessing needs a fresh builder and a nonempty namespace",
            ));
        }
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        // A Static Program without RHS inputs (only public inputs, as a label-hash part) may
        // omit the payload secret; Static material that reads it still requires it.
        match (dynamic, payload) {
            (false, Some(_)) | (true, None) => {}
            (false, None) if program.rhs_inputs().is_empty() => {}
            (false, None) => {
                return Err(ApiError::input("Static preprocessing requires the payload secret"));
            }
            (true, Some(_)) => {
                return Err(ApiError::input("Dynamic preprocessing takes no secret"));
            }
        }
        if let Some(rhs) = values {
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
        }
        if dynamic {
            let (setup, binding) = self.dynamic.as_ref().ok_or_else(|| {
                ApiError::input("Dynamic preprocessing needs its setup (with_dynamic_setup)")
            })?;
            setup
                .validate_public_seed(&self.context, &hash_key_commitment(hash_key))
                .map_err(ApiError::scheme)?;
            binding.validate(setup, program).map_err(ApiError::scheme)?;
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let hash_key_input = instance_name(
            self.artifact_prefix.as_deref(),
            &crate::rns_bgg::hash_key_input_name(self.hash_namespace.as_ref()),
        );
        let sampling_key = match &self.hash_namespace {
            Some(namespace) => crate::rns_bgg::child_hash_key(hash_key, namespace),
            None => hash_key.to_vec(),
        };
        let bytes = ring.bytes_input(hash_key_input.clone(), sampling_key.len());
        let program = Arc::new(program.clone());
        let workload = Arc::new(
            RnsBggDerivedWorkload::from_program_with_parameters(self.parameters, &program)
                .map_err(ApiError::scheme)?,
        );
        let mut evaluator = PublicKeyLowering::new(
            self.parameters,
            program.clone(),
            workload,
            payload.cloned(),
            bytes,
        );
        evaluator.output_use = self.output_use;
        evaluator.dynamic = self.dynamic.clone();
        evaluator.root_inputs = self.root_inputs.clone();
        evaluator.root_hash_key = self.parameters.root_hash_key_input();
        // Every data family is a graph input named under the builder's
        // namespace, with the builder's values unless bound by shape only.
        let name = |family| {
            instance_name(
                self.artifact_prefix.as_deref(),
                &crate::rns_bgg::data_input_name(self.hash_namespace.as_ref(), family),
            )
        };
        let graph_data = self.graph_data.is_some();
        let data = match &self.graph_data {
            Some(data) => data.clone(),
            None => crate::program::data_inputs(&program, name),
        };
        if (values.is_some() || self.rhs_values.is_some()) && !self.inputs_by_shape && !graph_data {
            self.data.validate(&program).map_err(|error| ApiError::input(error.to_string()))?;
            for family in program.data_families().keys() {
                let values = self.data.values(*family).expect("validated program data");
                self.runtime.insert(
                    name(*family),
                    RuntimeValue::integer_values(
                        values.iter().map(|value| (*value).into()).collect(),
                    ),
                );
            }
            evaluator.data_values = Some(self.data.clone());
        }
        self.families.data = data.clone();
        evaluator.data = data;
        // Static LUT switching material is prepared at root scope, so that
        // a LUT in a family body only reads it in every member. A LUT on a
        // public payload has none.
        if !dynamic {
            let luts = program.encryption_inputs().luts().to_vec();
            for (input, lut) in program.family_body_luts() {
                if program
                    .wire_knowledge(input)
                    .map_err(|error| ApiError::input(error.to_string()))? ==
                    crate::program::AttributeKnowledge::PublicPayload
                {
                    continue;
                }
                let level = evaluator.workload.level(input).map_err(ApiError::scheme)?;
                let table =
                    luts.get(lut.0 as usize).ok_or_else(|| ApiError::scheme(RnsBggError::Lut))?;
                evaluator
                    .prepare_lut_switching_ciphertexts(level, lut, table)
                    .map_err(ApiError::scheme)?;
            }
        }
        for (wire, declaration) in program.inputs() {
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(
                    evaluator.sample_input_public_key(*wire).map_err(ApiError::scheme)?,
                ),
                // Each member's key is sampled where it is read.
                InputBinding::Family(count) => InputBinding::Family(PreprocessWireFamily::Input {
                    wire: *wire,
                    key: evaluator
                        .input_key_source(*wire, declaration.knowledge())
                        .map_err(ApiError::scheme)?,
                    hash_key: evaluator.hash_key.clone(),
                    count: count.clone(),
                }),
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|e| ApiError::input(e.to_string()))?;
        }
        let prefix = instance_name(
            self.artifact_prefix.as_deref(),
            &super::plaintext_input_prefix(self.hash_namespace.as_ref(), "rhs"),
        );
        for (id, declaration) in program.rhs_inputs() {
            if dynamic {
                // Only references: every key derives from setup matrices, and
                // encryption binds the plaintexts.
                let binding = match declaration {
                    InputBinding::Scalar(()) => {
                        InputBinding::Scalar(RhsRef { id: *id, index: None::<Int> })
                    }
                    InputBinding::Family(count) => {
                        let count = count
                            .evaluate(parameters)
                            .ok()
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
            let name = format!("{prefix}/rhs/{id:?}");
            let plaintext = if let Some(values) = &self.rhs_values {
                let schema = ring.matrix_type((1, 1));
                match values.get(id).ok_or_else(|| ApiError::input("missing RHS value"))? {
                    InputBinding::Scalar(value) if value.matrix_type() == &schema => {
                        InputBinding::Scalar(value.clone())
                    }
                    InputBinding::Family(super::RhsPlaintexts::Values(values))
                        if values.schema().element == mxx_dsl::MatType(schema) =>
                    {
                        InputBinding::Family(super::PlaintextFamily::Graph(
                            super::RhsPlaintexts::Values(values.clone()),
                        ))
                    }
                    InputBinding::Family(lifted @ super::RhsPlaintexts::Lifted { .. }) => {
                        InputBinding::Family(super::PlaintextFamily::Graph(lifted.clone()))
                    }
                    _ => return Err(ApiError::input("RHS values must be level-zero polynomials")),
                }
            } else {
                match values {
                    Some(rhs) => super::plaintext_inputs(
                        &ring,
                        &name,
                        rhs.get(id).ok_or_else(|| ApiError::input("missing RHS plaintext"))?,
                        &mut self.runtime,
                    ),
                    None => {
                        let shape = match declaration {
                            InputBinding::Scalar(()) => InputBinding::Scalar(()),
                            InputBinding::Family(count) => InputBinding::Family(
                                count.evaluate(parameters).ok().map(IntExpr::constant).ok_or_else(
                                    || ApiError::input("RHS family count is not concrete"),
                                )?,
                            ),
                        };
                        super::plaintext_handles(&ring, &name, shape)
                    }
                }
                .map_err(ApiError::input)?
            };
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
                        .ok_or_else(|| ApiError::input("RHS family count is not concrete"))?;
                    let members = plaintexts.count().clone();
                    if members.evaluate(parameters).ok().as_ref() != Some(&expected) {
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
        self.runtime.insert(hash_key_input, RuntimeValue::Bytes(sampling_key.as_slice().into()));
        if let Some(root) = self.parameters.root() {
            self.runtime.insert(
                root.hash_key_input_name(),
                RuntimeValue::Bytes(root.hash_key().as_slice().into()),
            );
        }
        if let Some(binding) = payload.and_then(|payload| payload.binding.as_ref()) {
            binding.bind(&mut self.runtime);
        }
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.evaluator = Some(evaluator);
        Ok(())
    }
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
        let rhs = (!self.inputs_by_shape).then_some(rhs);
        self.bind_program_inputs(program, parameters, rhs, payload, hash_key)
    }
    fn artifact_identities(&self) -> ([u8; 32], [u8; 32]) {
        let Some(evaluator) = self.evaluator.as_ref() else {
            return ([0; 32], [0; 32]);
        };
        let mut digest = Sha256::new();
        digest.update(evaluator.program.id().as_bytes());
        digest.update(self.parameters.parameter_identity());
        digest.update(self.hash_key_commitment);
        // A Program preprocessed under a hash namespace samples its public matrices from the key
        // derived under it, so two namespaces are two stages (none keeps the identity unchanged).
        if let Some(namespace) = &self.hash_namespace {
            digest.update(b"hash-namespace");
            digest.update(namespace);
        }
        if let Some(prefix) = &self.artifact_prefix {
            digest.update(b"artifact-prefix");
            digest.update(prefix.as_bytes());
        }
        // A compiled Program is bound to its output use and, for Dynamic
        // parameters, to its setup and its mapping to the setup coordinates.
        // The default direct decoding of a Static Program keeps its identity.
        if self.output_use == RnsOutputUse::Gadget {
            digest.update(b"gadget-output");
        }
        if let Some((setup, binding)) = &self.dynamic {
            digest.update(setup.identity());
            digest.update(serde_json::to_vec(binding).expect("setup mapping encoding"));
        }
        if !evaluator.program.data_families().is_empty() && self.graph_data.is_none() {
            digest.update(self.data.commitment());
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

    fn emit(
        self,
        mut graph: DslContext,
        outputs: Vec<RnsBggPublicKey>,
    ) -> Result<GraphEmission<(Self::EncryptDescriptor, Self::OnlineEvalDescriptor)>, ApiError>
    {
        let name = |role: &str| instance_name(self.artifact_prefix.as_deref(), role);
        let mut evaluator =
            self.evaluator.ok_or_else(|| ApiError::input("preprocess builder was not bound"))?;
        // Every declared output has a carrier, including one a composing graph read as deferred
        // column blocks (`ProgramBuild::lower_deferred`) and so does not pass in `outputs`.
        let carriers = evaluator
            .program
            .outputs()
            .iter()
            .map(|output| {
                let wire = match output {
                    Operand::Scalar(wire) |
                    Operand::Member { family: wire, .. } |
                    Operand::Selected { family: wire, .. } => *wire,
                };
                let level = evaluator.workload.level(wire)?;
                self.parameters.output_carrier(
                    &evaluator.program,
                    wire,
                    level,
                    evaluator.output_use,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::scheme)?;
        let outputs =
            deduplicate_outputs(evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?);

        // Input public keys are sampled from the hash key alone, so no stage
        // stores them: encryption and online evaluation derive them where
        // they need them. A raw input that is also a Program output is
        // skipped here for the same reason.
        for output in outputs {
            if evaluator.program.inputs().contains_key(&output.wire) {
                continue;
            }
            let name = name(&format!("public-key/{:?}", output.wire));
            graph = match preprocess_output_availability(&evaluator.program, output.wire)
                .map_err(ApiError::scheme)?
            {
                ArtifactAvailability::Cached => graph.cached_output(name, output),
                ArtifactAvailability::Transferred => graph.transferred_output(name, output),
            }
            .map_err(ApiError::scheme)?;
        }
        // Only the bottom row of each RHS ciphertext is stored; its top row
        // is sampled from the hash key (`rhs_top`).
        for (rhs, prepared) in &evaluator.rhs_material {
            match prepared {
                InputBinding::Scalar(value) => {
                    graph = graph
                        .transferred_output(name(&format!("rhs/{rhs:?}")), value.1.clone())
                        .map_err(ApiError::scheme)?;
                }
                InputBinding::Family(value) => {
                    graph = graph
                        .transferred_output(
                            name(&format!("rhs/{rhs:?}/family")),
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
                    name(&format!(
                        "lut-switching/{evaluation_level}/{decomposition_level}/{lut:?}/{layout:?}"
                    )),
                    material.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        graph = publish_public_actions(graph, &evaluator.public_actions, |role| name(&role))?;
        let input_levels = evaluator
            .program
            .inputs()
            .keys()
            .map(|wire| Ok((*wire, evaluator.workload.level(*wire)?)))
            .collect::<Result<BTreeMap<_, _>, RnsBggError>>()
            .map_err(ApiError::scheme)?;
        let mut binding = RnsBggArtifactBinding::new(
            self.context,
            Some(evaluator.program.id()),
            self.parameters.parameter_identity(),
            self.hash_key_commitment,
            Vec::<ProductionId>::new(),
            self.parameters.rhs_mode(),
        )
        .map_err(ApiError::scheme)?
        .with_artifact_prefix(self.artifact_prefix.clone())
        .with_input_levels(input_levels)
        .with_outputs(evaluator.output_use, carriers);
        if !evaluator.program.data_families().is_empty() && self.graph_data.is_none() {
            binding = binding.with_data(self.data.commitment());
        }
        if let Some((setup, mapping)) = &evaluator.dynamic {
            binding = binding.with_dynamic_setup(setup.identity(), Some(mapping.clone()));
        }
        let encrypt_binding = RnsBggPreprocessEncryptDescriptor::from_binding(binding.clone());
        let online_eval_binding = RnsBggPreprocessOnlineEvalDescriptor::from_binding(binding);
        Ok(GraphEmission {
            graph,
            runtime_inputs: self.runtime,
            manifests: BTreeMap::new(),
            descriptor: (encrypt_binding, online_eval_binding),
        })
    }
}

impl ProgramBuild for RnsPreprocessBuild<'_> {
    type Wire = RnsBggPublicKey;
    type WireFamily = super::lowering::PreprocessWireFamily;

    fn lower(&mut self, program: &ExponentLutProgram) -> Result<Vec<Self::Wire>, ApiError> {
        let (evaluator, families, environment) = self.bound_lowering(program)?;
        lower_program(program, evaluator, families, environment).map_err(ApiError::lowering)
    }

    fn lower_deferred(
        &mut self,
        program: &ExponentLutProgram,
        deferred: &std::collections::BTreeSet<usize>,
    ) -> Result<Vec<crate::program::LoweredOutput<Self::Wire, Self::WireFamily>>, ApiError> {
        let (evaluator, families, environment) = self.bound_lowering(program)?;
        crate::program::lower_program_deferred(program, evaluator, families, environment, deferred)
            .map_err(ApiError::lowering)
    }

    fn deferred_block(
        &mut self,
        program: &ExponentLutProgram,
        columns: &crate::program::DeferredColumns<Self::Wire, Self::WireFamily>,
        block: &IntExpr,
    ) -> Result<Self::Wire, ApiError> {
        let (evaluator, families, environment) = self.bound_lowering(program)?;
        columns.block(program, evaluator, families, environment, block).map_err(ApiError::lowering)
    }
    fn lower_deferred_exporting(
        &mut self,
        program: &ExponentLutProgram,
        deferred: &std::collections::BTreeSet<usize>,
        exports: &std::collections::BTreeSet<crate::program::WireId>,
    ) -> Result<
        (
            Vec<crate::program::LoweredOutput<Self::Wire, Self::WireFamily>>,
            std::collections::BTreeMap<crate::program::WireId, Self::WireFamily>,
        ),
        ApiError,
    > {
        let (evaluator, families, environment) = self.bound_lowering(program)?;
        crate::program::lower_program_deferred_exporting(
            program,
            evaluator,
            families,
            environment,
            deferred,
            exports,
        )
        .map_err(ApiError::lowering)
    }

    fn family_member(
        &mut self,
        program: &ExponentLutProgram,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<Self::Wire, ApiError> {
        let (evaluator, families, _environment) = self.bound_lowering(program)?;
        crate::program::FamilyExecution::wire_at(families, evaluator, family, index)
            .map_err(ApiError::lowering)
    }
}

impl<'a> RnsPreprocessBuild<'a> {
    /// The bound lowering state of `program`: its evaluator, families and environment.
    #[allow(clippy::type_complexity)]
    fn bound_lowering(
        &mut self,
        program: &ExponentLutProgram,
    ) -> Result<
        (
            &mut PublicKeyLowering<'a>,
            &mut super::encoding::OnlineEvalFamilies<'a>,
            &mut Environment<
                RnsBggPublicKey,
                RhsRef<Int>,
                super::lowering::PreprocessWireFamily,
                Family<RhsRef<Int>>,
            >,
        ),
        ApiError,
    > {
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("preprocessing is not bound"))?;
        if evaluator.program.id() != program.id() {
            return Err(ApiError::input("lowered Program differs from the bound preprocessing"));
        }
        Ok((evaluator, &mut self.families, &mut self.environment))
    }
}

impl RnsPreprocessBuild<'_> {
    /// Publish this instance into `graph`, leaving it open for further composition.
    /// Finalize both returned descriptors against the eventual shared production.
    pub fn emit(
        self,
        graph: DslContext,
        outputs: Vec<RnsBggPublicKey>,
    ) -> Result<
        GraphEmission<(RnsBggPreprocessEncryptDescriptor, RnsBggPreprocessOnlineEvalDescriptor)>,
        ApiError,
    > {
        <Self as PreprocessBuild<mxx_backends::poly::dcrt::poly::DCRTPoly>>::emit(
            self, graph, outputs,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::api::StageArtifacts;

    use mxx_ir_core::ParamEnv;

    /// A small CPU-only chain; its dimension and CRT prime size are test-overridable.
    fn emission_parameters() -> RnsBggParameters {
        let read = |name: &str, default| {
            std::env::var(format!("MXX_EMISSION_TEST_{name}"))
                .map_or(default, |value| value.parse().unwrap())
        };
        let level = DCRTPolyParams::new(
            read("DIMENSION", 16) as u32,
            2,
            read("PRIME_BITS", 30),
            4,
            None,
            None,
        );
        RnsBggParameters::new(vec![level; 2], read("SIGMA", 1) as u64, 1).unwrap()
    }

    /// Two copies of one Program retain separate hash, plaintext, data and artifact names.
    #[test]
    fn test_emission_namespaces_keep_distinct_program_inputs_and_artifacts() {
        use mxx_backends::poly::dcrt::poly::DCRTPoly;
        let parameters = emission_parameters();
        let ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret::from_row(&parameters, ring.identity(1)).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let data = builder.data_family(1usize, Some(2)).unwrap();
        let output = builder
            .mul_private(
                Operand::Scalar(input),
                Operand::Selected { family: rhs, data, position: IntExpr::constant(0) },
            )
            .unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let make = |prefix: &str, choice: usize| {
            let values = crate::program::ProgramData::new().with(data, vec![choice]);
            let mut build = RnsPreprocessBuild::new(&parameters, rand::random())
                .with_artifact_prefix(prefix)
                .with_data(values);
            let constant = |value| DCRTPoly::from_u32s(parameters.level(0).unwrap(), &[value]);
            build
                .bind_rhs_inputs(
                    &program,
                    &ParamEnv::default(),
                    Some(&BTreeMap::from([(
                        rhs,
                        InputBinding::Family(vec![constant(0), constant(1)]),
                    )])),
                    Some(&payload),
                    &rand::random::<[u8; 32]>(),
                )
                .unwrap();
            build
        };
        let mut first = make("first", 0);
        let outputs = crate::api::lower(&program, &mut first).unwrap();
        let first = first.emit(DslContext::new("two-programs"), outputs).unwrap();
        let mut second = make("second", 1);
        let outputs = crate::api::lower(&program, &mut second).unwrap();
        let second = second.emit(first.graph, outputs).unwrap();
        let mut runtime = first.runtime_inputs;
        assert!(second.runtime_inputs.keys().all(|name| !runtime.contains_key(name)));
        runtime.extend(second.runtime_inputs);
        let graph = second.graph.build().unwrap();
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        for prefix in ["first", "second"] {
            assert!(graph_declares_input(&graph, &format!("{prefix}/rns-bgg/hash-key")));
            assert!(graph_declares_input(&graph, &format!("{prefix}/exponent-lut/data.0")));
            assert!(
                runtime
                    .keys()
                    .any(|name| name.starts_with(&format!("{prefix}/")) && name.contains("/rhs/"))
            );
            assert!(
                validated
                    .source
                    .outputs()
                    .keys()
                    .any(|name| name.starts_with(&format!("{prefix}/public-key/")))
            );
            assert!(
                validated
                    .source
                    .outputs()
                    .keys()
                    .any(|name| name.starts_with(&format!("{prefix}/rhs/")))
            );
        }
        assert_eq!(second.descriptor.1.artifact_prefix(), Some("second"));
    }

    /// Compare the legacy stages with two single graphs, using identical hash seeds and
    /// secrets. Compare each key to its preprocessing and decode the terminal private product.
    /// With zero sampling error all preprocessing keys must also agree across graphs;
    /// otherwise only the public product is independent of fresh RHS sampling errors.
    #[test]
    fn test_emission_single_graph_matches_legacy_keys_and_decoding() {
        use mxx_backends::{
            MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
            poly::dcrt::poly::DCRTPoly,
        };
        let parameters = emission_parameters();
        let ring = parameters.ring(0).unwrap();
        let secret = || {
            ring.polynomial(
                (0..parameters.level(0).unwrap().ring_dimension())
                    .map(|_| IntExpr::constant(i32::from(rand::random::<u8>() % 3) - 1))
                    .collect::<Vec<_>>(),
            )
        };
        let payload = RnsBggPayloadSecret::from_row(&parameters, secret()).unwrap();
        let mask = crate::rns_bgg::RnsBggMaskSecret::from_row(&parameters, secret()).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let output = builder.mul_public(Operand::Scalar(public), Operand::Scalar(private)).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let data = builder.data_family(1usize, Some(2)).unwrap();
        let rhs_output = builder
            .mul_private(
                Operand::Scalar(private),
                Operand::Selected { family: rhs, data, position: IntExpr::constant(0) },
            )
            .unwrap();
        let program =
            builder.finish(vec![Operand::Scalar(output), Operand::Scalar(rhs_output)]).unwrap();
        let values = crate::program::ProgramData::new().with(data, vec![1]);
        let context = rand::random();
        let hash_key: [u8; 32] = rand::random();
        let bit = u32::from(rand::random::<bool>());
        let plaintext = ring.polynomial([IntExpr::constant(bit)]);
        let lhs = BTreeMap::from([
            (
                private,
                InputBinding::Scalar(DCRTPoly::from_u32s(parameters.level(0).unwrap(), &[bit])),
            ),
            (public, InputBinding::Scalar(DCRTPoly::from_u32s(parameters.level(0).unwrap(), &[1]))),
        ]);
        let mut execution = CpuExecution::new(cpu_backend(parameters.levels().iter().cloned()));
        let mut store = MemoryArtifactStore::default();
        let (legacy_encrypt, legacy_online) = crate::api::preprocess(
            &program,
            &ParamEnv::default(),
            &BTreeMap::from([(
                rhs,
                InputBinding::Family(vec![
                    DCRTPoly::from_u32s(parameters.level(0).unwrap(), &[0]),
                    DCRTPoly::from_u32s(parameters.level(0).unwrap(), &[1]),
                ]),
            )]),
            Some(&payload),
            &hash_key,
            RnsPreprocessBuild::new(&parameters, context).with_data(values.clone()),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let encrypted = crate::api::encrypt(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            &lhs,
            Some(&legacy_encrypt),
            &payload,
            Some(&mask),
            &hash_key,
            RnsEncryptBuild::new(&parameters, context),
            &mut execution,
            &mut store,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let legacy = crate::api::online_eval(
            &program,
            &ParamEnv::default(),
            &legacy_online,
            &encrypted,
            &hash_key,
            RnsOnlineEvalBuild::new(&parameters, context).with_data(values.clone()),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();

        let mut preprocessing = RnsPreprocessBuild::new(&parameters, context)
            .with_artifact_prefix("instance")
            .with_data(values.clone());
        preprocessing
            .bind_rhs_values(
                &program,
                &ParamEnv::default(),
                BTreeMap::from([(
                    rhs,
                    InputBinding::Family(crate::rns_bgg::RhsPlaintexts::Values(
                        mxx_dsl::parallel(2usize, |index| {
                            Ok(ring.polynomial([IntExpr::Rem(
                                Box::new(index.expression()?),
                                Box::new(IntExpr::constant(2)),
                            )]))
                        })
                        .unwrap(),
                    )),
                )]),
                Some(&payload),
                &hash_key,
            )
            .unwrap();
        let keys = crate::api::lower(&program, &mut preprocessing).unwrap();
        let source = preprocessing.evaluator.as_ref().unwrap();
        let root = crate::rns_bgg::RnsBggRoot::new(parameters.level(0).unwrap().clone(), hash_key);
        let private_key = source.sample_input_public_key(private).unwrap();
        let public_key = source.sample_input_public_key(public).unwrap();
        let error = || parameters.encoding_contract().sample_row(&ring, root.columns()).unwrap();
        let private_row = root
            .encode(
                &private_key.matrix,
                mask.row(),
                (payload.row().0.clone() * plaintext.clone(), plaintext),
                error(),
            )
            .unwrap();
        let public_row =
            root.encode(&public_key.matrix, mask.row(), mask.row().clone(), error()).unwrap();
        let graph = DslContext::new("composed-preprocess")
            .transferred_output("initial-private", private_row)
            .unwrap()
            .transferred_output("initial-public", public_row)
            .unwrap();
        let (graph, descriptor) = preprocessing.emit(graph, keys).unwrap().build().unwrap();
        let composed = crate::api::execute_artifacts(
            graph,
            &ParamEnv::default(),
            descriptor,
            rand::random(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let online = StageArtifacts::new(
            composed.production().clone(),
            composed.manifest().clone(),
            composed.descriptor().1.clone(),
            composed.session().clone(),
        );
        let row = |name: &str| {
            (0..2)
                .map(|port| {
                    ring.artifact_input(
                        composed.production().clone(),
                        format!("{name}.{port}"),
                        (1, root.columns()),
                        ArtifactAvailability::Transferred,
                    )
                })
                .collect::<Vec<_>>()
        };
        let private_row = row("initial-private");
        let public_row = row("initial-public");
        let mut evaluation = RnsOnlineEvalBuild::new(&parameters, context).with_data(values);
        evaluation
            .bind_rows(
                &program,
                &ParamEnv::default(),
                &online,
                BTreeMap::from([
                    (
                        private,
                        InputBinding::Scalar((private_row[0].clone(), private_row[1].clone())),
                    ),
                    (public, InputBinding::Scalar((public_row[0].clone(), public_row[1].clone()))),
                ]),
                BTreeMap::from([(public, InputBinding::Scalar(ring.identity(1)))]),
                &hash_key,
            )
            .unwrap();
        let outputs = crate::api::lower(&program, &mut evaluation).unwrap();
        let mut emission = evaluation.emit(DslContext::new("evaluate-like"), outputs).unwrap();
        assert_eq!(emission.manifests.len(), 1, "no encryption production is read");
        emission
            .manifests
            .insert(legacy_online.production().clone(), legacy_online.manifest().clone());
        for (index, wire) in [output, rhs_output].into_iter().enumerate() {
            for port in 0..4 {
                let role = format!("public-key/{wire:?}.{port}");
                let entry = &legacy_online.manifest().artifacts[&role];
                let mxx_ir_core::artifact::ArtifactType::Matrix(matrix) = &entry.artifact_type
                else {
                    panic!("public key matrix");
                };
                for (kind, stage) in [("legacy", &legacy_online), ("emitted", &online)] {
                    let key = ring.artifact_input(
                        stage.production().clone(),
                        stage.descriptor().artifact_name(&role),
                        (matrix.rows, matrix.columns),
                        entry.availability,
                    );
                    emission.graph =
                        emission.graph.output(format!("{kind}-key-{index}-{port}"), key).unwrap();
                }
            }
        }
        let (graph, ()) = emission.build().unwrap();
        let mut emitted = crate::api::execute_graph(
            graph,
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        emitted.outputs = emitted
            .outputs
            .into_iter()
            .map(|(name, value)| {
                (name.strip_prefix("instance/").unwrap_or(&name).to_owned(), value)
            })
            .collect();
        for index in 0..2 {
            for port in 0..4 {
                let name = format!("output-{index}.{port}");
                assert_eq!(
                    crate::test_support::full(&legacy.outputs[&name]),
                    crate::test_support::full(
                        &emitted.outputs[&format!("legacy-key-{index}-{port}")]
                    )
                );
                assert_eq!(
                    crate::test_support::full(&emitted.outputs[&name]),
                    crate::test_support::full(
                        &emitted.outputs[&format!("emitted-key-{index}-{port}")]
                    )
                );
                if index == 0 || parameters.encoding_error_bound() == &BigUint::from(0u8) {
                    assert_eq!(
                        crate::test_support::full(&legacy.outputs[&name]),
                        crate::test_support::full(&emitted.outputs[&name])
                    );
                }
            }
            for result in [&legacy, &emitted] {
                if index == 0 {
                    continue;
                }
                let decoded = crate::rns_bgg::decode_output(
                    &program,
                    result,
                    index,
                    &parameters,
                    &payload,
                    Some(&mask),
                    RnsOutputUse::default(),
                    &[0],
                    &mut execution,
                    &mut store,
                    |_| Ok(()),
                )
                .unwrap();
                assert_eq!(decoded, vec![BigInt::from(bit)]);
            }
        }
    }

    /// One graph preprocesses a Program for two instances in a parallel loop, each with its
    /// own graph-valued data (selecting another RHS member) and its own hash domain, and
    /// publishes every artifact role as one family of both instances; an evaluation of a
    /// runtime-selected instance binds the same graph data and reads that instance's material.
    /// The decoded output is the instance's selected RHS times the input bit.
    #[test]
    fn test_graph_data_and_hash_domain_select_one_instance_of_a_loop() {
        use mxx_backends::{
            MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
        };
        use mxx_dsl::{Leaf, Mat};
        let base = emission_parameters();
        let ring = base.ring(0).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let data = builder.data_family(1usize, Some(2)).unwrap();
        let output = builder
            .mul_private(
                Operand::Scalar(private),
                Operand::Selected { family: rhs, data, position: IntExpr::constant(0) },
            )
            .unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let context: [u8; 32] = rand::random();
        let hash_key: [u8; 32] = rand::random();
        let bit = u32::from(rand::random::<bool>());
        let secret = |ring: &mxx_dsl::Ring| ring.uniform_interval((1, 1), -1, 1);
        let (payload_row, mask_row) = (secret(&ring), secret(&ring));
        // Instance `m` selects RHS member `m`, whose plaintext is `m + 1`.
        let selections = mxx_dsl::int_family_input("selections", 2usize);
        let (mut descriptor, mut names) = (None, Vec::new());
        let mut runtime = BTreeMap::new();
        let instances = mxx_dsl::parallel(2usize, |member| {
            let parameters = base.clone().with_hash_domain(vec![member.clone()]);
            let payload = RnsBggPayloadSecret::from_row(&parameters, payload_row.clone()).unwrap();
            let mask =
                crate::rns_bgg::RnsBggMaskSecret::from_row(&parameters, mask_row.clone()).unwrap();
            let mut preprocessing =
                RnsPreprocessBuild::new(&parameters, context).with_artifact_prefix("instances");
            preprocessing
                .bind_data_values(
                    &program,
                    BTreeMap::from([(
                        data,
                        (
                            mxx_dsl::parallel(1usize, |_| Ok(selections.at(member.clone())))?,
                            Int::constant(0),
                        ),
                    )]),
                )
                .unwrap();
            let plaintexts = mxx_dsl::parallel(2usize, |index| {
                Ok(ring.polynomial([IntExpr::Add(
                    Box::new(index.expression()?),
                    Box::new(IntExpr::constant(1)),
                )]))
            })?;
            preprocessing
                .bind_rhs_values(
                    &program,
                    &ParamEnv::default(),
                    BTreeMap::from([(
                        rhs,
                        InputBinding::Family(crate::rns_bgg::RhsPlaintexts::Values(plaintexts)),
                    )]),
                    Some(&payload),
                    &hash_key,
                )
                .unwrap();
            let keys = crate::api::lower(&program, &mut preprocessing).unwrap();
            let key =
                preprocessing.evaluator.as_ref().unwrap().sample_input_public_key(private).unwrap();
            let root =
                crate::rns_bgg::RnsBggRoot::new(parameters.level(0).unwrap().clone(), hash_key);
            let plaintext = ring.polynomial([IntExpr::constant(bit)]);
            let row = root
                .encode(
                    &key.matrix,
                    mask.row(),
                    (payload.row().0.clone() * plaintext.clone(), plaintext),
                    (ring.zero((1, root.columns())), ring.zero((1, root.columns()))),
                )
                .unwrap();
            let emission = preprocessing.emit(DslContext::new("instance"), keys).unwrap();
            descriptor = Some(emission.descriptor.1);
            runtime.extend(emission.runtime_inputs);
            let mut leaves = emission
                .graph
                .into_outputs()
                .into_iter()
                .filter(|(name, _, _)| {
                    name.starts_with("instances/rhs/") || name.starts_with("instances/public-key/")
                })
                .map(|(name, leaf, _)| (name, leaf))
                .collect::<Vec<_>>();
            leaves.push(("row.0".to_owned(), Leaf::of(row.0).unwrap()));
            leaves.push(("row.1".to_owned(), Leaf::of(row.1).unwrap()));
            let values: Vec<Leaf>;
            (names, values) = leaves.into_iter().unzip();
            Ok(values)
        })
        .unwrap();
        let mut graph = DslContext::new("instances");
        for (position, name) in names.iter().enumerate() {
            let family = instances.field(|leaves| leaves[position].clone()).unwrap();
            let count = match family.at(0).wire_type() {
                mxx_ir_core::types::WireType::IndexedFamily { count, .. } => Some(count.clone()),
                _ => None,
            };
            let published = match count {
                None => Leaf::of(family).unwrap(),
                Some(count) => Leaf::of(
                    mxx_dsl::parallel(IntExpr::constant(2) * count.clone(), |index| {
                        let per = Int::evaluate(count.clone());
                        Ok(family.at(index.clone().div(per.clone())).at(index.rem(per)).unwrap())
                    })
                    .unwrap(),
                )
                .unwrap(),
            };
            graph = graph.transferred_output(name, published).unwrap();
        }
        runtime.insert(
            "selections".to_owned(),
            RuntimeValue::integer_values(vec![0.into(), 1.into()]),
        );
        let graph = graph.build().unwrap();
        let mut execution = CpuExecution::new(cpu_backend(base.levels().iter().cloned()));
        let mut store = MemoryArtifactStore::default();
        let production = crate::api::execute_artifacts(
            StageGraph::new(graph, runtime, BTreeMap::new()),
            &ParamEnv::default(),
            (),
            rand::random(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let preprocessed = StageArtifacts::new(
            production.production().clone(),
            production.manifest().clone(),
            descriptor.unwrap(),
            production.session().clone(),
        );
        let instance = rand::random::<bool>() as usize;
        let context_graph = DslContext::new("evaluate-instance");
        let member = context_graph.input::<Int>("instance", mxx_dsl::IntType).unwrap();
        let parameters = base.clone().with_hash_domain(vec![member.clone()]);
        let selected = |name: &str| -> Mat {
            ring.family_artifact_input(
                production.production().clone(),
                name,
                2usize,
                (1, base.component_columns(0).unwrap()),
                ArtifactAvailability::Transferred,
            )
            .at(member.clone())
        };
        let mut evaluation = RnsOnlineEvalBuild::new(&parameters, context)
            .with_material_member(member.clone(), 2usize);
        evaluation
            .bind_data_values(
                &program,
                BTreeMap::from([(
                    data,
                    (
                        mxx_dsl::parallel(1usize, |_| {
                            Ok(mxx_dsl::int_family_input("selections", 2usize).at(member.clone()))
                        })
                        .unwrap(),
                        Int::constant(0),
                    ),
                )]),
            )
            .unwrap();
        evaluation
            .bind_rows(
                &program,
                &ParamEnv::default(),
                &preprocessed,
                BTreeMap::from([(
                    private,
                    InputBinding::Scalar((selected("row.0"), selected("row.1"))),
                )]),
                BTreeMap::new(),
                &hash_key,
            )
            .unwrap();
        let outputs = crate::api::lower(&program, &mut evaluation).unwrap();
        let mut emission = evaluation.emit(context_graph, outputs).unwrap();
        // The instance's published output key and the other instance's.
        for port in 0..4 {
            let role = format!("instances/public-key/{output:?}.{port}");
            let entry = &production.manifest().artifacts[&role];
            assert_eq!(entry.family_count, Some(2), "one key per instance");
            let mxx_ir_core::artifact::ArtifactType::Matrix(matrix) = &entry.artifact_type else {
                panic!("public key matrices")
            };
            let keys = ring.family_artifact_input(
                production.production().clone(),
                role,
                2usize,
                (matrix.rows, matrix.columns),
                entry.availability,
            );
            emission.graph = emission
                .graph
                .output(format!("published-{port}"), keys.at(member.clone()))
                .unwrap()
                .output(format!("other-{port}"), keys.at(Int::constant(1).sub(member.clone())))
                .unwrap();
        }
        let (mut graph, ()) = emission.build().unwrap();
        graph.runtime_inputs.insert("instance".to_owned(), RuntimeValue::Int(instance.into()));
        graph.runtime_inputs.insert(
            "selections".to_owned(),
            RuntimeValue::integer_values(vec![0.into(), 1.into()]),
        );
        graph.manifests.insert(production.production().clone(), production.manifest().clone());
        let result = crate::api::execute_graph(
            graph,
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let full = |name: &str| crate::test_support::full(&result.outputs[name]);
        for port in 0..4 {
            let evaluated = full(&format!("instances/output-0.{port}"));
            assert_eq!(evaluated, full(&format!("published-{port}")));
            if port >= 2 {
                // The bottom key rows of the two instances' private products differ by their
                // RHS ciphertexts, which differ in plaintext and hash domain.
                assert_ne!(evaluated, full(&format!("other-{port}")));
            }
        }
    }

    /// Shape-only Static preprocessing preserves a selector count larger than `usize` through
    /// plaintext handles, reference/material families and the actual Program lowerer.
    #[test]
    fn oversized_rhs_shape_lowers_without_a_runtime_vector() {
        use mxx_backends::poly::dcrt::poly::DCRTPoly;
        let parameters =
            RnsBggParameters::new(vec![DCRTPolyParams::new(16, 2, 17, 4, None, None); 2], 1, 1)
                .unwrap();
        let count = IntExpr::constant(usize::MAX as u128 * 130);
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(count.clone()).unwrap();
        let products = builder
            .parallel(
                crate::program::FamilyRange::full(count.clone()).unwrap(),
                |builder, index| {
                    builder
                        .mul_private(Operand::Scalar(input), Operand::Member { family: rhs, index })
                        .map(Operand::Scalar)
                },
            )
            .unwrap();
        let program = builder
            .finish(vec![Operand::Member { family: products, index: IntExpr::constant(0) }])
            .unwrap();
        let payload = crate::rns_bgg::RnsBggPayloadSecret {
            row: (parameters.ring(0).unwrap().identity(1), parameters.ring(0).unwrap().identity(1)),
            binding: None,
        };
        let mut child = RnsPreprocessBuild::new(&parameters, [1; 32]).with_inputs_by_shape();
        child
            .bind_rhs_inputs::<DCRTPoly>(
                &program,
                &ParamEnv::default(),
                None,
                Some(&payload),
                &[2; 32],
            )
            .unwrap();
        let evaluator = child.evaluator.as_ref().unwrap();
        let InputBinding::Family(plaintexts) = &evaluator.rhs[&rhs] else {
            panic!("selector plaintext family")
        };
        assert_eq!(plaintexts.count(), &count);
        let (evaluator, families, environment) =
            <RnsPreprocessBuild as PreprocessBuild<DCRTPoly>>::lowering_state(&mut child).unwrap();
        lower_program(&program, evaluator, families, environment).unwrap();

        let ring = parameters.ring(0).unwrap();
        let plaintexts = mxx_dsl::parallel(count.clone(), |_| Ok(ring.identity(1))).unwrap();
        let mut composed = RnsPreprocessBuild::new(&parameters, rand::random())
            .with_artifact_prefix("symbolic-rhs");
        composed
            .bind_rhs_values(
                &program,
                &ParamEnv::default(),
                BTreeMap::from([(
                    rhs,
                    InputBinding::Family(crate::rns_bgg::RhsPlaintexts::Values(plaintexts)),
                )]),
                Some(&payload),
                &rand::random::<[u8; 32]>(),
            )
            .unwrap();
        let outputs = crate::api::lower(&program, &mut composed).unwrap();
        let emission = composed.emit(DslContext::new("symbolic-rhs"), outputs).unwrap();
        assert!(!emission.runtime_inputs.keys().any(|name| name.contains("/rhs/")));
        let (graph, _) = emission.build().unwrap();
        graph.graph.validate(&ParamEnv::default()).unwrap();
    }

    /// Static preprocessing takes no payload secret for a Program without RHS inputs, and
    /// requires it for one with RHS inputs.
    #[test]
    fn static_preprocessing_needs_the_payload_secret_only_for_rhs() {
        let parameters = emission_parameters();
        let public = |rhs: bool| {
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
            let output = if rhs {
                let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
                let rhs = builder.rhs_input().unwrap();
                builder.mul_private(Operand::Scalar(private), Operand::Scalar(rhs)).unwrap()
            } else {
                builder.mul_public(Operand::Scalar(input), Operand::Scalar(input)).unwrap()
            };
            builder.finish(vec![Operand::Scalar(output)]).unwrap()
        };
        let bind = |program: &ExponentLutProgram, rhs| {
            RnsPreprocessBuild::new(&parameters, rand::random()).bind_rhs_values(
                program,
                &ParamEnv::default(),
                rhs,
                None,
                &rand::random::<[u8; 32]>(),
            )
        };
        bind(&public(false), BTreeMap::new()).unwrap();
        let program = public(true);
        let rhs = *program.rhs_inputs().keys().next().unwrap();
        let ring = parameters.ring(0).unwrap();
        let error = bind(&program, BTreeMap::from([(rhs, InputBinding::Scalar(ring.identity(1)))]))
            .unwrap_err();
        assert!(error.to_string().contains("requires the payload secret"), "{error}");
    }

    /// Graph secrets reject matrices of the wrong width and require no resident binding.
    #[test]
    fn test_emission_graph_secrets_validate_schema() {
        let parameters = emission_parameters();
        let ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret::from_row(&parameters, ring.identity(1)).unwrap();
        let mask =
            crate::rns_bgg::RnsBggMaskSecret::from_row(&parameters, ring.identity(1)).unwrap();
        assert!(payload.input_name().is_none());
        assert!(mask.input_name().is_none());
        assert!(RnsBggPayloadSecret::from_row(&parameters, ring.zero((1, 2))).is_err());
        assert!(
            crate::rns_bgg::RnsBggMaskSecret::from_row(&parameters, ring.zero((2, 1))).is_err()
        );
    }

    /// Graph input families bind symbolically, and public payload coverage is exact.
    #[test]
    fn test_emission_graph_row_families_validate_shape() {
        use mxx_backends::{
            MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
            poly::dcrt::poly::DCRTPoly,
        };
        let parameters = emission_parameters();
        let ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret::from_row(&parameters, ring.identity(1)).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input =
            builder.input_family(IntExpr::constant(3), AttributeKnowledge::PublicPayload).unwrap();
        let output = builder.sum(input).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let context = rand::random();
        let hash_key: [u8; 32] = rand::random();
        let mut execution = CpuExecution::new(cpu_backend(parameters.levels().iter().cloned()));
        let mut store = MemoryArtifactStore::default();
        let (_, preprocessing) = crate::api::preprocess(
            &program,
            &ParamEnv::default(),
            &PlaintextInputs::<RhsId, DCRTPoly>::new(),
            Some(&payload),
            &hash_key,
            RnsPreprocessBuild::new(&parameters, context).with_artifact_prefix("family"),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |_| Ok(()),
        )
        .unwrap();
        let rows = |count| {
            mxx_dsl::parallel(count, |_| {
                Ok((
                    ring.zero((
                        1,
                        parameters
                            .key_component_columns(0, AttributeKnowledge::PublicPayload)
                            .unwrap(),
                    )),
                    ring.zero((
                        1,
                        parameters
                            .key_component_columns(0, AttributeKnowledge::PublicPayload)
                            .unwrap(),
                    )),
                ))
            })
            .unwrap()
        };
        let plaintexts = mxx_dsl::parallel(3usize, |_| Ok(ring.identity(1))).unwrap();
        let mut evaluation = RnsOnlineEvalBuild::new(&parameters, context);
        assert!(
            evaluation
                .bind_rows(
                    &program,
                    &ParamEnv::default(),
                    &preprocessing,
                    BTreeMap::from([(
                        input,
                        InputBinding::Family(crate::program::GraphFamily::Values(rows(3usize)))
                    )]),
                    BTreeMap::new(),
                    &hash_key
                )
                .is_err()
        );
        let mut wrong = RnsOnlineEvalBuild::new(&parameters, context);
        assert!(
            wrong
                .bind_rows(
                    &program,
                    &ParamEnv::default(),
                    &preprocessing,
                    BTreeMap::from([(
                        input,
                        InputBinding::Family(crate::program::GraphFamily::Values(rows(2usize)))
                    )]),
                    BTreeMap::from([(
                        input,
                        InputBinding::Family(crate::program::GraphFamily::Values(
                            plaintexts.clone()
                        ))
                    )]),
                    &hash_key
                )
                .is_err()
        );
        evaluation
            .bind_rows(
                &program,
                &ParamEnv::default(),
                &preprocessing,
                BTreeMap::from([(
                    input,
                    InputBinding::Family(crate::program::GraphFamily::Values(rows(3usize))),
                )]),
                BTreeMap::from([(
                    input,
                    InputBinding::Family(crate::program::GraphFamily::Values(plaintexts)),
                )]),
                &hash_key,
            )
            .unwrap();
        let outputs = crate::api::lower(&program, &mut evaluation).unwrap();
        let (graph, ()) = evaluation
            .emit(DslContext::new("family-evaluation"), outputs)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(graph.manifests.len(), 1);
        graph.graph.validate_with_manifests(&ParamEnv::default(), &graph.manifests).unwrap();
    }

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
            preprocess_output_availability(&program, private).unwrap(),
            ArtifactAvailability::Cached
        );
        assert_eq!(
            preprocess_output_availability(&program, public_output).unwrap(),
            ArtifactAvailability::Cached
        );
        assert_eq!(
            preprocess_output_availability(&program, private_output).unwrap(),
            ArtifactAvailability::Transferred
        );
        assert_eq!(
            preprocess_output_availability(&program, private_rhs_output).unwrap(),
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
        let parameters =
            RnsBggParameters::new(vec![DCRTPolyParams::new(16, 2, 17, 4, None, None); 2], 1, 1)
                .unwrap()
                .with_rhs_mode(RhsMode::Dynamic);
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let source = builder.finish(vec![Operand::Scalar(input)]).unwrap();
        let (setup, binding) = RnsBggDynamicSetup::for_program(
            &parameters,
            [1; 32],
            &[2; 32],
            &source,
            ExpLayout { stride_log2: 0, width_log2: 0 },
            RnsBggDynamicDistributions::uniform(parameters.encoding_contract()),
        )
        .unwrap();
        let mut build = RnsPreprocessBuild::new(&parameters, [1; 32])
            .with_dynamic_setup(&setup, binding.clone())
            .unwrap();
        build
            .bind_rhs_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &source,
                &mxx_ir_core::ParamEnv::default(),
                Some(&BTreeMap::new()),
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
        digest.update(setup.identity());
        digest.update(serde_json::to_vec(&binding).unwrap());
        assert_eq!(encryption, <[u8; 32]>::from(digest.clone().finalize()));
        digest.update(b"online-eval");
        assert_eq!(online_eval, <[u8; 32]>::from(digest.finalize()));
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
    /// The deferred blocks of a row `Sum` over a family of a family, read in a loop over the
    /// blocks, equal the columns of the eagerly lowered row: the member of a lazy family reading
    /// another lazy family's members, and the block sums, compute the same keys.
    #[test]
    fn test_deferred_row_blocks_match_the_row_columns() {
        use crate::program::{FamilyRange, LoweredOutput};
        use mxx_backends::{
            MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
        };
        let parameters = emission_parameters();
        let (columns, per) = (2usize, 3usize);
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let inputs = builder
            .input_family(IntExpr::constant(columns * per), AttributeKnowledge::PublicPayload)
            .unwrap();
        let range = || FamilyRange::full(IntExpr::constant(columns * per)).unwrap();
        let doubled = builder
            .parallel(range(), |builder, index| {
                builder
                    .add(
                        Operand::Member { family: inputs, index: index.clone() },
                        Operand::Member { family: inputs, index },
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let placed = builder
            .parallel(range(), |builder, index| {
                let shift = IntExpr::Rem(Box::new(index.clone()), Box::new(IntExpr::constant(per)));
                builder
                    .mul_small_constant(
                        Operand::Member { family: doubled, index },
                        vec![1, -1],
                        shift,
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let row = builder.sum_columns(placed, columns).unwrap();
        let program = builder.finish(vec![Operand::Scalar(row)]).unwrap();
        let (context, hash_key): ([u8; 32], [u8; 32]) = (rand::random(), rand::random());
        let ring = parameters.ring(0).unwrap();
        let payload =
            RnsBggPayloadSecret::from_row(&parameters, ring.uniform_interval((1, 1), -1, 1))
                .unwrap();
        let build = || {
            let mut build = RnsPreprocessBuild::new(&parameters, context);
            build
                .bind_rhs_values(
                    &program,
                    &ParamEnv::default(),
                    BTreeMap::new(),
                    Some(&payload),
                    &hash_key,
                )
                .unwrap();
            build
        };
        let mut eager = build();
        let row = crate::api::lower(&program, &mut eager).unwrap().remove(0);
        let mut deferred = build();
        let lowered =
            deferred.lower_deferred(&program, &std::collections::BTreeSet::from([0])).unwrap();
        let LoweredOutput::Columns(row_columns) = &lowered[0] else {
            panic!("the row is deferred");
        };
        assert_eq!(row_columns.columns(), columns);
        let blocks = mxx_dsl::parallel(columns, |block| {
            Ok(deferred
                .deferred_block(&program, row_columns, &block.expression()?)
                .map_err(|_| mxx_dsl::DslError::Schema)?
                .matrix)
        })
        .unwrap();
        let ports = |matrix: &RnsBggMatrix| {
            [matrix.0.0.clone(), matrix.0.1.clone(), matrix.1.0.clone(), matrix.1.1.clone()]
        };
        let width = parameters.component_columns(0).unwrap();
        let mut emission = eager.emit(DslContext::new("deferred-row"), Vec::new()).unwrap();
        let deferred = deferred.emit(DslContext::new("deferred-blocks"), Vec::new()).unwrap();
        emission.runtime_inputs.extend(deferred.runtime_inputs);
        for block in 0..columns {
            let member = blocks.at(Int::constant(block));
            for (port, (eager, deferred)) in
                ports(&row.matrix).into_iter().zip(ports(&member)).enumerate()
            {
                let range = mxx_ir_core::node::IndexRange {
                    start: IntExpr::constant(block * width),
                    end: IntExpr::constant((block + 1) * width),
                };
                emission.graph = emission
                    .graph
                    .output(format!("row-{block}-{port}"), eager.slice(None, Some(range)))
                    .unwrap()
                    .output(format!("block-{block}-{port}"), deferred)
                    .unwrap();
            }
        }
        let (graph, _) = emission.build().unwrap();
        let mut execution = CpuExecution::new(cpu_backend(parameters.levels().to_vec()));
        let outputs = crate::api::execute_graph(
            graph,
            &ParamEnv::default(),
            &mut execution,
            &mut MemoryArtifactStore::default(),
            |_| Ok(()),
        )
        .unwrap()
        .outputs;
        for block in 0..columns {
            for port in 0..4 {
                assert_eq!(
                    crate::test_support::full(&outputs[&format!("row-{block}-{port}")]),
                    crate::test_support::full(&outputs[&format!("block-{block}-{port}")]),
                    "block {block}, port {port}"
                );
            }
        }
    }
}

/// Publish the public actions of Dynamic root sites (see
/// [`super::public_action_role`]) under the artifact names `name` gives. Leaf row families
/// are direct member-loop outputs, exported per lane without a root repacking consumer.
fn publish_public_actions(
    mut graph: DslContext,
    actions: &BTreeMap<WireId, (RnsBggMatrix, Vec<super::BranchKeys>)>,
    name: impl Fn(String) -> String,
) -> Result<DslContext, ApiError> {
    for (wire, (key, branches)) in actions {
        let role = super::public_action_role(*wire, "key");
        for (port, entry) in matrix_entries(key).into_iter().enumerate() {
            graph = graph
                .transferred_output(name(format!("{role}.{port}")), entry)
                .map_err(ApiError::scheme)?;
        }
        for keys in branches {
            for (port, rows) in keys.rows.iter().enumerate() {
                graph = graph
                    .transferred_output(
                        name(super::branch_keys_role(*wire, keys.group, port)),
                        rows.clone(),
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
    }
    Ok(graph)
}
