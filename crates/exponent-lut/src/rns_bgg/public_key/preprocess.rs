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
            parameters,
            context,
            hash_key_commitment: [0; 32],
            evaluator: None,
            families: super::encoding::OnlineEvalFamilies::new(parameters),
            environment: Environment::new(),
            runtime: BTreeMap::new(),
            decoder_declaration: None,
            decoder_smudge_child: None,
            unmask_bound: None,
            output_use: RnsOutputUse::default(),
            dynamic: None,
            data: crate::program::ProgramData::new(),
            hash_namespace: None,
            refresh: None,
            refresh_matrix_key: None,
        }
    }

    /// Make this builder a refresh child that samples its public matrices
    /// from the stage hash key derived under `namespace`.
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

    /// This builder with the output use `output_use` (see [`RnsOutputUse`]).
    /// The preprocessing binding records it, and online evaluation and noise
    /// simulation must select the same use.
    pub fn with_output_use(mut self, output_use: RnsOutputUse) -> Self {
        self.output_use = output_use;
        self
    }

    /// Bind RHS plaintexts for a decoder smudge child without exposing the
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

    /// Bind this builder, on [`RnsBggParameters::decoder_smudge_parameters`],
    /// to lower `declaration`'s smudge Program: its smudge LUT members are
    /// finalized and transformed in their root bodies.
    pub fn bind_decoder_smudge(
        &mut self,
        declaration: &crate::decoder::decoding::DecoderDeclaration,
    ) -> Result<(), ApiError> {
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        declaration
            .validate_smudge_program(evaluator.program.id())
            .map_err(|error| ApiError::scheme(RnsBggError::Decoder(error)))?;
        if evaluator.decoder_smudge.is_some() {
            return Err(ApiError::input("decoder smudge is already bound"));
        }
        evaluator.decoder_smudge = Some(Arc::new(declaration.clone()));
        Ok(())
    }

    /// Prepare the unmasking of `declaration`'s decoder of output
    /// `output_index`, whose lowered source key is `source`: lower the bound
    /// smudge child `smudge` once, form the summed component-1 key `A` of the
    /// source column selection and the smudge sum, sample the public `B`
    /// with its trapdoor (sigma `trapdoor_sigma`), and sample the preimage
    /// `K` of `A`. The graph publishes `K`, `B`, and the smudge child's
    /// public material under the decoder namespace, and the trapdoor never
    /// leaves it; the source output's public key is published unchanged.
    pub fn apply_decoder(
        &mut self,
        output_index: usize,
        declaration: &crate::decoder::decoding::DecoderDeclaration,
        source: RnsBggPublicKey,
        mut smudge: RnsPreprocessBuild<'a>,
        trapdoor_sigma: &bigdecimal::BigDecimal,
    ) -> Result<(), ApiError> {
        if self.decoder_declaration.is_some() {
            return Err(ApiError::input("decoder was already applied"));
        }
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        self.parameters
            .validate_decoder_smudge_parameters(smudge.parameters, declaration, &evaluator.program)
            .map_err(ApiError::scheme)?;
        if smudge
            .evaluator
            .as_ref()
            .and_then(|child| child.decoder_smudge.as_ref())
            .map(|bound| *bound.commitment()) !=
            Some(*declaration.commitment()) ||
            evaluator.program.outputs().get(output_index) !=
                Some(&Operand::Scalar(declaration.source_output())) ||
            source.wire != declaration.source_output()
        {
            return Err(ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch)));
        }
        let mut decoder = RnsBggDecoder { lowerer: evaluator, source };
        let key = apply_decoder(declaration, &mut decoder, |_| {
            let child = smudge
                .evaluator
                .as_mut()
                .ok_or(RnsBggError::Binding("decoder smudge builder was not bound"))?;
            let program = child.program.clone();
            lower_program(&program, child, &mut smudge.families, &mut smudge.environment)
                .map_err(|error| RnsBggError::Graph(error.to_string()))
        })
        .map_err(ApiError::scheme)?;
        let (trapdoor, bound) = sample_unmask_trapdoor(
            self.parameters,
            declaration,
            &evaluator.program,
            trapdoor_sigma,
        )?;
        let preimage = crate::decoder::unmask::preimage(&trapdoor, key);
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        evaluator.unmask = Some((declaration.artifact_key(), preimage, trapdoor.public_matrix()));
        merge_runtime(&mut self.runtime, std::mem::take(&mut smudge.runtime))
            .map_err(ApiError::scheme)?;
        self.decoder_smudge_child = Some(
            smudge
                .evaluator
                .take()
                .ok_or_else(|| ApiError::input("decoder smudge was not bound"))?,
        );
        self.unmask_bound = Some(bound);
        self.decoder_declaration = Some(declaration.clone());
        Ok(())
    }

    /// Prepare `declaration`'s noise refresh of output `target_index`, whose
    /// lowered public key is `target`: lower the bound fresh-noise child
    /// `fresh` and the decoders' smudge children `smudges` (in decoder order)
    /// once, compute every decoded key `K_i = A_1 X_i - K_x Y_i + K_f Z_i` with
    /// the same homomorphic steps as the online rows (see
    /// [`crate::refresh`]), add each decoder's smudge key, and sample one
    /// preimage per decoder under a single unmask trapdoor (sigma
    /// `trapdoor_sigma`). The graph publishes the children's public material,
    /// `B` and the preimages; the trapdoor never leaves it.
    pub fn apply_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        target: RnsBggPublicKey,
        fresh: RnsPreprocessBuild<'a>,
        smudges: Vec<RnsPreprocessBuild<'a>>,
        trapdoor_sigma: &bigdecimal::BigDecimal,
    ) -> Result<(), ApiError> {
        self.prepare_refresh(
            target_index,
            declaration,
            target,
            None,
            fresh,
            smudges,
            trapdoor_sigma,
        )
    }

    /// Prepare the top-refresh part `declaration` (see
    /// [`crate::refresh::top`]) of output `target_index` as
    /// [`Self::apply_refresh`] does, with its mask child `mask`: the one key
    /// is the mask carrier's key next to the public one key's second block,
    /// the target key is the part's component, and only the output prime is
    /// decoded. `A'` is sampled under the key set by
    /// [`Self::with_refresh_matrix_key`].
    #[allow(clippy::too_many_arguments)]
    pub fn apply_top_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        target: RnsBggPublicKey,
        mask: RnsPreprocessBuild<'a>,
        fresh: RnsPreprocessBuild<'a>,
        smudges: Vec<RnsPreprocessBuild<'a>>,
        trapdoor_sigma: &bigdecimal::BigDecimal,
    ) -> Result<(), ApiError> {
        self.prepare_refresh(
            target_index,
            declaration,
            target,
            Some(mask),
            fresh,
            smudges,
            trapdoor_sigma,
        )
    }

    /// Sample a top refresh's `A'` from the stage key `hash_key` derived
    /// under `namespace` ([`crate::refresh::top::TopRefreshDeclaration::matrix_namespace`]),
    /// the same for every chain.
    pub fn with_refresh_matrix_key(mut self, namespace: [u8; 32], hash_key: &[u8]) -> Self {
        let name = crate::rns_bgg::refresh::hash_key_input_name(Some(&namespace));
        let key = crate::rns_bgg::refresh::child_hash_key(hash_key, &namespace);
        self.runtime.insert(name.clone(), RuntimeValue::Bytes(key.as_slice().into()));
        self.refresh_matrix_key = Some((name, key.len()));
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        target: RnsBggPublicKey,
        mut mask: Option<RnsPreprocessBuild<'a>>,
        mut fresh: RnsPreprocessBuild<'a>,
        mut smudges: Vec<RnsPreprocessBuild<'a>>,
        trapdoor_sigma: &bigdecimal::BigDecimal,
    ) -> Result<(), ApiError> {
        use crate::rns_bgg::refresh::{
            refresh_column, refresh_combination, refresh_fresh_namespace, refresh_mask_namespace,
        };
        if self.decoder_declaration.is_some() || self.refresh.is_some() {
            return Err(ApiError::input("a decoder or refresh was already applied"));
        }
        let mismatch = || ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch));
        let child_program = |child: &RnsPreprocessBuild<'a>| {
            child.evaluator.as_ref().map(|evaluator| evaluator.program.id())
        };
        let evaluator =
            self.evaluator.as_ref().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        let smudge_parameters = smudges.iter().map(|child| child.parameters).collect::<Vec<_>>();
        let level = self
            .parameters
            .validate_refresh_children(
                declaration,
                &evaluator.program,
                mask.as_ref().map(|child| child.parameters),
                fresh.parameters,
                &smudge_parameters,
            )
            .map_err(ApiError::scheme)?;
        if let (Some(part), Some(child)) = (declaration.top(), mask.as_ref()) &&
            (child_program(child) != Some(part.mask().id()) ||
                child.hash_namespace != Some(refresh_mask_namespace(declaration)) ||
                child.evaluator.as_ref().is_some_and(|child| child.decoder_smudge.is_some()))
        {
            return Err(mismatch());
        }
        if evaluator.program.outputs().get(target_index) !=
            Some(&Operand::Scalar(declaration.target_output())) ||
            target.wire != declaration.target_output() ||
            child_program(&fresh) != Some(declaration.fresh().id()) ||
            fresh.hash_namespace != Some(refresh_fresh_namespace(declaration)) ||
            fresh.evaluator.as_ref().is_some_and(|child| child.decoder_smudge.is_some())
        {
            return Err(mismatch());
        }
        for (child, decoder) in smudges.iter().zip(declaration.decoders()) {
            if child.hash_namespace != Some(*decoder.commitment()) ||
                child
                    .evaluator
                    .as_ref()
                    .and_then(|child| child.decoder_smudge.as_ref())
                    .map(|bound| *bound.commitment()) !=
                    Some(*decoder.commitment())
            {
                return Err(mismatch());
            }
        }
        let ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
        let lower = |child: &mut RnsPreprocessBuild<'a>| {
            let lowerer = child
                .evaluator
                .as_mut()
                .ok_or(RnsBggError::Binding("refresh child builder was not bound"))?;
            let program = lowerer.program.clone();
            lower_program(&program, lowerer, &mut child.families, &mut child.environment)
                .map_err(|error| RnsBggError::Graph(error.to_string()))
        };
        // The fresh carriers' keys, moved down once to the decoding ring
        // (component 1: their payload secret component is one).
        let fresh_keys = lower(&mut fresh)
            .map_err(ApiError::scheme)?
            .into_iter()
            .map(|key| super::mod_down_public(key.matrix, &ring))
            .collect::<Vec<_>>();
        let hash_key = evaluator.hash_key.clone();
        let matrix_key = match (declaration.top(), &self.refresh_matrix_key) {
            (None, _) => hash_key.clone(),
            (Some(_), Some((name, length))) => ring.bytes_input(name.clone(), *length),
            (Some(_), None) => {
                return Err(ApiError::input("a top refresh needs with_refresh_matrix_key"));
            }
        };
        let a_prime = self
            .parameters
            .refresh_matrix(level, matrix_key, declaration)
            .map_err(ApiError::scheme)?;
        let one = self
            .parameters
            .refresh_one_key(level, hash_key, declaration)
            .map_err(ApiError::scheme)?;
        // A top-refresh part's one key: the mask carrier's key (component 1,
        // moved down once like the fresh carriers) for the first block.
        let one = match mask.as_mut() {
            None => one,
            Some(child) => {
                let key = lower(child)
                    .map_err(ApiError::scheme)?
                    .into_iter()
                    .next()
                    .ok_or_else(mismatch)?;
                let key = super::mod_down_public(key.matrix, &ring);
                ((key.0.1, one.0.1), (key.1.1, one.1.1))
            }
        };
        let component = declaration.component();
        let target_top = if component == 0 { &target.matrix.0.0 } else { &target.matrix.0.1 };
        let target_bottom = if component == 0 { &target.matrix.1.0 } else { &target.matrix.1.1 };
        let fresh_top = fresh_keys.iter().map(|key| key.0.1.clone()).collect::<Vec<_>>();
        let fresh_bottom = fresh_keys.iter().map(|key| key.1.1.clone()).collect::<Vec<_>>();
        let (trapdoor, bound) = sample_unmask_trapdoor(
            self.parameters,
            &declaration.decoders()[0],
            &evaluator.program,
            trapdoor_sigma,
        )?;
        let mut preimages = Vec::with_capacity(declaration.decoders().len());
        let columns = declaration.columns();
        for (position, prime) in declaration.decoded_primes().into_iter().enumerate() {
            let actions = self
                .parameters
                .refresh_actions(level, prime, a_prime.clone(), declaration)
                .map_err(ApiError::scheme)?;
            let top = refresh_combination(
                (one.0.0.clone(), one.0.1.clone()),
                target_top.clone(),
                &fresh_top,
                &actions,
            );
            let bottom = refresh_combination(
                (one.1.0.clone(), one.1.1.clone()),
                target_bottom.clone(),
                &fresh_bottom,
                &actions,
            );
            for column in 0..columns {
                let smudge = lower(&mut smudges[position * columns + column])
                    .map_err(ApiError::scheme)?
                    .into_iter()
                    .next()
                    .ok_or_else(mismatch)?;
                let key_top = refresh_column(top.clone(), column) +
                    checked_modulus_convert(smudge.matrix.0.1, &ring)
                        .map_err(ApiError::scheme)?;
                let key_bottom = refresh_column(bottom.clone(), column) +
                    checked_modulus_convert(smudge.matrix.1.1, &ring)
                        .map_err(ApiError::scheme)?;
                let key =
                    Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![key_top, key_bottom]);
                preimages.push(crate::decoder::unmask::preimage(&trapdoor, key));
            }
        }
        merge_runtime(&mut self.runtime, std::mem::take(&mut fresh.runtime))
            .map_err(ApiError::scheme)?;
        if let Some(child) = mask.as_mut() {
            merge_runtime(&mut self.runtime, std::mem::take(&mut child.runtime))
                .map_err(ApiError::scheme)?;
        }
        for child in &mut smudges {
            merge_runtime(&mut self.runtime, std::mem::take(&mut child.runtime))
                .map_err(ApiError::scheme)?;
        }
        let take = |mut child: RnsPreprocessBuild<'a>| {
            child.evaluator.take().ok_or_else(|| ApiError::input("refresh child was not bound"))
        };
        self.refresh = Some(PreparedRefresh {
            declaration: declaration.clone(),
            mask: mask.map(take).transpose()?,
            fresh: take(fresh)?,
            smudges: smudges.into_iter().map(take).collect::<Result<Vec<_>, _>>()?,
            preimages,
            public: trapdoor.public_matrix(),
            bound,
        });
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
                    return Err(RnsBggError::Binding("decoder child runtime input conflicts"));
                }
                _ => return Err(RnsBggError::Binding("decoder child runtime input schema")),
            },
            // A secret input name is unique to one sampled secret.
            Some(RuntimeValue::Matrix(_)) if matches!(value, RuntimeValue::Matrix(_)) => {}
            Some(_) => return Err(RnsBggError::Binding("decoder child runtime input schema")),
            None => {
                destination.insert(name, value);
            }
        }
    }
    Ok(())
}

/// Publish a child Program's input keys (cached) and Static material
/// (transferred) under the names `name(role)` and `input_key(wire, port)`.
fn publish_child(
    mut graph: DslContext,
    child: &PublicKeyLowering<'_>,
    name: impl Fn(&str) -> String,
    input_key: impl Fn(WireId, usize) -> String,
) -> Result<DslContext, ApiError> {
    for (wire, input) in child.program.inputs() {
        let names = (0..4).map(|port| input_key(*wire, port)).collect::<Vec<_>>();
        match input.shape() {
            InputBinding::Scalar(()) => {
                let key = child.sample_input_public_key(*wire).map_err(ApiError::scheme)?;
                for (name, entry) in names.into_iter().zip(matrix_entries(&key.matrix)) {
                    graph = graph.cached_output(name, entry).map_err(ApiError::scheme)?;
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
                    graph = graph.cached_output(name, entry).map_err(ApiError::scheme)?;
                }
            }
        }
    }
    for (rhs, prepared) in &child.rhs_material {
        graph = match prepared {
            InputBinding::Scalar(value) => {
                graph.transferred_output(name(&format!("rhs/{rhs:?}")), value.1.clone())
            }
            InputBinding::Family(value) => graph.transferred_output(
                name(&format!("rhs/{rhs:?}/family")),
                child
                    .published_rhs_family(rhs, value)?
                    .field(|matrix| matrix.1)
                    .map_err(ApiError::scheme)?,
            ),
        }
        .map_err(ApiError::scheme)?;
    }
    graph = publish_public_actions(graph, &child.public_actions, |role| name(&role))?;
    for ((evaluation_level, decomposition_level, lut, layout), material) in &child.lut_material {
        let role =
            format!("lut-switching/{evaluation_level}/{decomposition_level}/{lut:?}/{layout:?}");
        graph =
            graph.transferred_output(name(&role), material.clone()).map_err(ApiError::scheme)?;
    }
    Ok(graph)
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
        let hash_key_input =
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref());
        let sampling_key = match &self.hash_namespace {
            Some(namespace) => crate::rns_bgg::refresh::child_hash_key(hash_key, namespace),
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
        self.data.validate(&program).map_err(|error| ApiError::input(error.to_string()))?;
        let data = crate::program::data_inputs(&program);
        self.families.data = data.clone();
        evaluator.data = data;
        for (family, _) in program.data_families() {
            let values = self.data.values(*family).expect("validated program data");
            self.runtime.insert(
                crate::program::ProgramData::input_name(*family),
                RuntimeValue::integer_values(values.iter().map(|value| (*value).into()).collect()),
            );
        }
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
        let prefix = super::plaintext_input_prefix(self.hash_namespace.as_ref(), "rhs");
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
        self.runtime.insert(hash_key_input, RuntimeValue::Bytes(sampling_key.as_slice().into()));
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
        if let Some(declaration) = &self.decoder_declaration {
            digest.update(declaration.commitment());
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
        if !evaluator.program.data_families().is_empty() {
            digest.update(self.data.commitment());
        }
        if let Some(refresh) = &self.refresh {
            digest.update(refresh.declaration.spec());
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
        self,
        outputs: Vec<RnsBggPublicKey>,
    ) -> Result<(StageGraph, (Self::EncryptDescriptor, Self::OnlineEvalDescriptor)), ApiError> {
        let mut evaluator =
            self.evaluator.ok_or_else(|| ApiError::input("preprocess builder was not bound"))?;
        let carriers = outputs
            .iter()
            .map(|output| {
                let level = evaluator.workload.level(output.wire)?;
                self.parameters.output_carrier(
                    &evaluator.program,
                    output.wire,
                    level,
                    evaluator.output_use,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::scheme)?;
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
            graph = match preprocess_output_availability(&evaluator.program, output.wire)
                .map_err(ApiError::scheme)?
            {
                ArtifactAvailability::Cached => graph.cached_output(name, output),
                ArtifactAvailability::Transferred => graph.transferred_output(name, output),
            }
            .map_err(ApiError::scheme)?;
        }
        // The smudge child uses its own source-ring key schedule. Publish its
        // input keys and Static material under the decoder namespace so
        // encryption cannot resolve a same-named outer-program artifact.
        if let (Some(declaration), Some(child)) =
            (self.decoder_declaration.as_ref(), self.decoder_smudge_child.as_ref())
        {
            let key = declaration.artifact_key();
            graph = publish_child(
                graph,
                child,
                |role| decoder_smudge_artifact_name(&key, role),
                |wire, port| decoder_smudge_input_key_artifact_name(&key, wire, port),
            )?;
        }
        if let Some(refresh) = &self.refresh {
            let spec = *refresh.declaration.spec();
            graph = publish_child(
                graph,
                &refresh.fresh,
                |role| crate::rns_bgg::refresh::refresh_fresh_artifact_name(spec, role),
                |wire, port| {
                    crate::rns_bgg::refresh::refresh_fresh_input_key_artifact_name(spec, wire, port)
                },
            )?;
            if let Some(mask) = &refresh.mask {
                graph = publish_child(
                    graph,
                    mask,
                    |role| crate::rns_bgg::refresh::refresh_mask_artifact_name(spec, role),
                    |wire, port| {
                        crate::rns_bgg::refresh::refresh_mask_input_key_artifact_name(
                            spec, wire, port,
                        )
                    },
                )?;
            }
            for ((child, decoder), preimage) in
                refresh.smudges.iter().zip(refresh.declaration.decoders()).zip(&refresh.preimages)
            {
                let key = decoder.artifact_key();
                graph = publish_child(
                    graph,
                    child,
                    |role| decoder_smudge_artifact_name(&key, role),
                    |wire, port| decoder_smudge_input_key_artifact_name(&key, wire, port),
                )?
                .transferred_output(
                    decoder_artifact_name(&key, "unmask-preimage"),
                    preimage.clone(),
                )
                .map_err(ApiError::scheme)?;
            }
            graph = graph
                .transferred_output(
                    crate::rns_bgg::refresh::refresh_artifact_name(spec, "unmask-public"),
                    refresh.public.clone(),
                )
                .map_err(ApiError::scheme)?;
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
        graph = publish_public_actions(graph, &evaluator.public_actions, |role| role)?;
        // The unmasking of the decoder: `K` for online evaluation and `B` for
        // encryption. The trapdoor itself stays in the setup production.
        if let Some((key, preimage, public)) = &evaluator.unmask {
            graph = graph
                .transferred_output(decoder_artifact_name(key, "unmask-preimage"), preimage.clone())
                .map_err(ApiError::scheme)?
                .transferred_output(decoder_artifact_name(key, "unmask-public"), public.clone())
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
            Some(evaluator.program.id()),
            self.parameters.parameter_identity(),
            self.hash_key_commitment,
            Vec::<ProductionId>::new(),
            self.parameters.rhs_mode(),
        )
        .map_err(ApiError::scheme)?
        .with_input_levels(input_levels)
        .with_outputs(evaluator.output_use, carriers);
        if !evaluator.program.data_families().is_empty() {
            binding = binding.with_data(self.data.commitment());
        }
        if let Some((setup, mapping)) = &evaluator.dynamic {
            binding = binding.with_dynamic_setup(setup.identity(), Some(mapping.clone()));
        }
        if let (Some(declaration), Some(bound)) = (&self.decoder_declaration, &self.unmask_bound) {
            binding = binding.with_decoder(*declaration.commitment(), bound.clone());
        }
        if let Some(refresh) = &self.refresh {
            binding = binding.with_decoder(*refresh.declaration.spec(), refresh.bound.clone());
        }
        let encrypt_binding = RnsBggPreprocessEncryptDescriptor::from_binding(binding.clone());
        let online_eval_binding = RnsBggPreprocessOnlineEvalDescriptor::from_binding(binding);
        let runtime = super::encoding::declared_runtime_inputs(&graph, self.runtime);
        Ok((
            StageGraph::new(graph, runtime, BTreeMap::new()),
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
        digest.update(setup.identity());
        digest.update(serde_json::to_vec(&binding).unwrap());
        assert_eq!(encryption, <[u8; 32]>::from(digest.clone().finalize()));
        digest.update(b"online-eval");
        assert_eq!(online_eval, <[u8; 32]>::from(digest.finalize()));
    }

    /// Static preprocessing of a decoder smudge child prepares every selector
    /// member inside the root body that reads it at its own index and exports
    /// the family from that loop: no selector family is read from the
    /// material prepared at binding, so none stays resident.
    #[test]
    fn decoder_smudge_selectors_stream_from_their_root_loops() {
        use mxx_backends::poly::dcrt::poly::DCRTPoly;
        let (pool, parameters) = crate::rns_bgg::tests::decoder_test_chain();
        let level = parameters.level(0).unwrap().clone();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let source = builder.finish(vec![Operand::Scalar(input)]).unwrap();
        let (declaration, keys) =
            crate::decoder::decoding::fixture::declaration(&source, input, 8, 1);
        let smudge_parameters =
            parameters.decoder_smudge_parameters(&declaration, &source, &pool, 1).unwrap();
        let selectors =
            declaration.smudge().prf().selector_plaintexts::<DCRTPoly>(&level, &keys).unwrap();
        let payload = crate::rns_bgg::RnsBggPayloadSecret {
            row: (
                smudge_parameters.ring(0).unwrap().identity(1),
                smudge_parameters.ring(0).unwrap().identity(1),
            ),
            binding: None,
        };
        let mut child = RnsPreprocessBuild::new(&smudge_parameters, [1; 32]);
        child
            .bind_rhs_inputs(
                declaration.smudge().program(),
                &mxx_ir_core::ParamEnv::default(),
                &selectors,
                Some(&payload),
                &[2; 32],
            )
            .unwrap();
        child.bind_decoder_smudge(&declaration).unwrap();
        let (evaluator, families, environment) =
            <RnsPreprocessBuild as PreprocessBuild<DCRTPoly>>::lowering_state(&mut child).unwrap();
        let program = evaluator.program.clone();
        lower_program(&program, evaluator, families, environment).unwrap();
        let evaluator = child.evaluator.as_ref().unwrap();
        assert!(evaluator.prepared_rhs_reads.is_empty(), "{:?}", evaluator.prepared_rhs_reads);
        assert_eq!(evaluator.streamed_rhs.len(), program.rhs_inputs().len());
        assert!(evaluator.streamed_rhs.values().all(Option::is_some));
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

/// Publish the public actions of Dynamic root sites (see
/// [`super::public_action_role`]) under the artifact names `name` gives.
fn publish_public_actions(
    mut graph: DslContext,
    actions: &BTreeMap<WireId, (RnsBggMatrix, Option<[Mat; 2]>)>,
    name: impl Fn(String) -> String,
) -> Result<DslContext, ApiError> {
    for (wire, (key, branches)) in actions {
        let role = super::public_action_role(*wire, "key");
        for (port, entry) in matrix_entries(key).into_iter().enumerate() {
            graph = graph
                .transferred_output(name(format!("{role}.{port}")), entry)
                .map_err(ApiError::scheme)?;
        }
        let role = super::public_action_role(*wire, "branches");
        for (port, row) in branches.iter().flatten().enumerate() {
            graph = graph
                .transferred_output(name(format!("{role}.{port}")), row.clone())
                .map_err(ApiError::scheme)?;
        }
    }
    Ok(graph)
}
