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
            decoder_declaration: None,
            decoder_smudge_child: None,
            unmask_bound: None,
            output_use: RnsOutputUse::default(),
            dynamic: None,
            data: crate::program::ProgramData::new(),
            graph_data: None,
            hash_namespace: None,
            refresh: None,
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
    /// [`Self::with_data`]) for a decoder smudge or refresh child without
    /// exposing the generic stage trait at application call sites. Dynamic
    /// children bind no RHS plaintexts and no secret.
    ///
    /// `None` binds the RHS plaintexts and the data by the Program's declared
    /// shapes only: the stage graph is the one of any plaintexts and data of
    /// those shapes (graph inputs of the same names and counts), so it can be
    /// validated, recorded and exported without materializing one selector
    /// plaintext and table entry per key member, but not executed (the inputs
    /// have no values). The data are then not checked against the key set,
    /// and the stage exports a child's data without entries (see
    /// `RnsBggArtifactBinding::child_data`).
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

    /// Bind this builder, on [`RnsBggParameters::decoder_smudge_parameters`],
    /// to lower `declaration`'s smudge Program: its smudge LUT members are
    /// finalized and transformed in their root bodies. The bound coordinate
    /// tables must be those of the declaration's smudge key set.
    pub fn bind_decoder_smudge(
        &mut self,
        declaration: &crate::decoder::decoding::DecoderDeclaration,
    ) -> Result<(), ApiError> {
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        declaration
            .validate_smudge_program(evaluator.program.id())
            .map_err(|error| ApiError::scheme(RnsBggError::Decoder(error)))?;
        // Data bound by shape only have no entries to check.
        if let Some(data) = &evaluator.data_values {
            declaration
                .smudge()
                .prf()
                .check_tables(data)
                .map_err(|_| ApiError::input("the smudge tables are not those of its key set"))?;
        }
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
        self.check_child_namespace(&smudge)?;
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

    /// Prepare part `part` (`(chain, component)`) of `declaration`'s refresh of output
    /// `target_index`, whose lowered public key is `target`: lower the bound fresh-noise child
    /// `fresh` and decoder's smudge child `smudge` once, compute the decoded key
    /// `K = A_1 X - K_x Y + K_f Z` with the same homomorphic steps as the online rows (see
    /// [`crate::refresh`]), its one key the key `K_g` of the caller's encoding of `sigma_0 G`
    /// next to the public one key's second block and its target key the part's component, add
    /// the smudge key, and sample the preimage under an unmask trapdoor (sigma
    /// `trapdoor_sigma`). The graph publishes the children's public material, `B` and the
    /// preimage; the trapdoor never leaves it.
    ///
    /// `sigma` is the caller's encoding of `sigma_0 G` of the part's chain, which both parts of
    /// the chain read (see [Caller inputs](crate::refresh#caller-inputs)): the preprocessing
    /// artifacts of its
    /// Program, whose first output is the encoding, preprocessed with the output use
    /// [`RnsOutputUse::Gadget`] on [`RnsBggParameters::refresh_child_parameters`] (or any chain
    /// whose output carrier is a carrier of the decoding ring); the graph reads its key
    /// `public-key/<wire>` (component 1). `A'` is the graph input
    /// [`crate::rns_bgg::REFRESH_MATRIX_INPUT`] (see [`Self::with_refresh_matrix`]).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        part: (usize, usize),
        target: RnsBggPublicKey,
        sigma: (
            &crate::api::StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
            &ExponentLutProgram,
        ),
        fresh: RnsPreprocessBuild<'a>,
        smudge: RnsPreprocessBuild<'a>,
        trapdoor_sigma: &bigdecimal::BigDecimal,
    ) -> Result<(), ApiError> {
        let declaration =
            crate::rns_bgg::refresh::refresh_part(declaration, part).map_err(ApiError::scheme)?;
        self.prepare_refresh(
            target_index,
            declaration,
            target,
            sigma,
            fresh,
            smudge,
            trapdoor_sigma,
        )
    }

    /// The value of the caller's `A' = [A'_(0) | A'_(1)]` of a refresh, the graph input
    /// [`crate::rns_bgg::REFRESH_MATRIX_INPUT`] (`2 x 2 w_0` over the reference's level-0 ring,
    /// see [Caller inputs](crate::refresh#caller-inputs)). A stage built without it can be
    /// recorded but not executed.
    pub fn with_refresh_matrix(mut self, a_prime: RuntimeValue) -> Self {
        self.runtime.insert(crate::rns_bgg::REFRESH_MATRIX_INPUT.to_owned(), a_prime);
        self
    }

    fn prepare_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshPartDeclaration,
        target: RnsBggPublicKey,
        (sigma, sigma_program): (
            &crate::api::StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
            &ExponentLutProgram,
        ),
        mut fresh: RnsPreprocessBuild<'a>,
        mut smudge: RnsPreprocessBuild<'a>,
        trapdoor_sigma: &bigdecimal::BigDecimal,
    ) -> Result<(), ApiError> {
        use crate::rns_bgg::refresh::refresh_combination;
        if self.decoder_declaration.is_some() || self.refresh.is_some() {
            return Err(ApiError::input("a decoder or refresh was already applied"));
        }
        self.check_child_namespace(&fresh)?;
        self.check_child_namespace(&smudge)?;
        let mismatch = || ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch));
        let child_program = |child: &RnsPreprocessBuild<'a>| {
            child.evaluator.as_ref().map(|evaluator| evaluator.program.id())
        };
        let evaluator =
            self.evaluator.as_ref().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        let level = self
            .parameters
            .validate_refresh_children(
                declaration,
                &evaluator.program,
                fresh.parameters,
                smudge.parameters,
            )
            .map_err(ApiError::scheme)?;
        // The caller's encoding of `sigma_0 G`: the preprocessing of its Program, whose first
        // output is a gadget carrier.
        let sigma_binding = sigma.descriptor().binding();
        let Some(Operand::Scalar(sigma_wire)) = sigma_program.outputs().first().cloned() else {
            return Err(ApiError::input("the sigma Program's first output is not a scalar"));
        };
        if sigma_binding.program() != Some(sigma_program.id()) ||
            sigma_binding.output_use() != RnsOutputUse::Gadget
        {
            return Err(ApiError::input(
                "the sigma artifacts are not a gadget-output preprocessing of the sigma Program",
            ));
        }
        if evaluator.program.outputs().get(target_index) !=
            Some(&Operand::Scalar(declaration.target_output())) ||
            target.wire != declaration.target_output() ||
            child_program(&fresh) != Some(declaration.fresh().id()) ||
            fresh.hash_namespace != Some(declaration.fresh_namespace()) ||
            fresh.evaluator.as_ref().is_some_and(|child| child.decoder_smudge.is_some())
        {
            return Err(mismatch());
        }
        // The children's coordinate tables must be those of the committed key
        // sets (the smudge children checked theirs when bound to their
        // decoders); data bound by shape only have no entries to check.
        let check = |prf: &crate::program_gadgets::prf::SparseLwrPrf,
                     child: &RnsPreprocessBuild<'a>| {
            match child.evaluator.as_ref().map(|child| child.data_values.as_ref()) {
                Some(None) => Ok(()),
                data => {
                    prf.check_tables(data.flatten().unwrap_or(&crate::program::ProgramData::new()))
                }
            }
        };
        if check(declaration.fresh().sampler().prf(), &fresh).is_err() {
            return Err(ApiError::input("the refresh tables are not those of its key sets"));
        }
        let decoder = declaration.decoder();
        if smudge.hash_namespace != Some(*decoder.commitment()) ||
            smudge
                .evaluator
                .as_ref()
                .and_then(|child| child.decoder_smudge.as_ref())
                .map(|bound| *bound.commitment()) !=
                Some(*decoder.commitment())
        {
            return Err(mismatch());
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
        // The fresh carrier row's key, moved down once to the decoding ring
        // (component 1: its payload secret component is one).
        let fresh_key = lower(&mut fresh)
            .map_err(ApiError::scheme)?
            .into_iter()
            .next()
            .map(|key| super::mod_down_public(key.matrix, &ring))
            .ok_or_else(mismatch)?;
        let hash_key = evaluator.hash_key.clone();
        let a_prime =
            self.parameters.refresh_matrix(level, declaration).map_err(ApiError::scheme)?;
        let one = self
            .parameters
            .refresh_one_key(level, hash_key, declaration)
            .map_err(ApiError::scheme)?;
        // The one key: the key `K_g` of the caller's encoding of `sigma_0 G` (component 1, a
        // carrier of the decoding ring with its `m` columns) for the first block.
        let one = {
            let columns = self.parameters.level(level).map_err(ApiError::scheme)?.modulus_digits();
            let availability = preprocess_output_availability(sigma_program, sigma_wire)
                .map_err(ApiError::scheme)?;
            let port = |port: usize| {
                ring.artifact_input(
                    sigma.production().clone(),
                    sigma.descriptor().artifact_name(&format!("public-key/{sigma_wire:?}.{port}")),
                    (1, columns),
                    availability,
                )
            };
            ((port(1), one.0.1), (port(3), one.1.1))
        };
        let component = declaration.component();
        let target_top = if component == 0 { &target.matrix.0.0 } else { &target.matrix.0.1 };
        let target_bottom = if component == 0 { &target.matrix.1.0 } else { &target.matrix.1.1 };
        let columns = declaration.columns();
        let (fresh_top, fresh_bottom) = (fresh_key.0.1, fresh_key.1.1);
        let (trapdoor, bound) =
            sample_unmask_trapdoor(self.parameters, decoder, &evaluator.program, trapdoor_sigma)?;
        // One preimage of the whole decoded row: column `j` of the key is column `j` of the
        // decoded key plus column `j`'s smudge key (the smudge child sums each column into its
        // column).
        let actions = self
            .parameters
            .refresh_actions(level, a_prime, declaration.output_prime(), declaration.reference())
            .map_err(ApiError::scheme)?;
        let fresh_term = |row| {
            crate::rns_bgg::refresh::row_fresh_term(row, columns, &actions)
                .map_err(ApiError::scheme)
        };
        let top = refresh_combination(
            (one.0.0, one.0.1),
            target_top.clone(),
            fresh_term(fresh_top)?,
            &actions,
        )
        .map_err(ApiError::scheme)?;
        let bottom = refresh_combination(
            (one.1.0, one.1.1),
            target_bottom.clone(),
            fresh_term(fresh_bottom)?,
            &actions,
        )
        .map_err(ApiError::scheme)?;
        let smudge_key = lower(&mut smudge)
            .map_err(ApiError::scheme)?
            .into_iter()
            .next()
            .ok_or_else(mismatch)?;
        let key_top = top +
            checked_modulus_convert(smudge_key.matrix.0.1, &ring).map_err(ApiError::scheme)?;
        let key_bottom = bottom +
            checked_modulus_convert(smudge_key.matrix.1.1, &ring).map_err(ApiError::scheme)?;
        let key = Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![key_top, key_bottom]);
        let preimage = crate::decoder::unmask::preimage(&trapdoor, key);
        for child in [&mut fresh, &mut smudge] {
            merge_runtime(&mut self.runtime, std::mem::take(&mut child.runtime))
                .map_err(ApiError::scheme)?;
        }
        let take = |mut child: RnsPreprocessBuild<'a>| {
            child.evaluator.take().ok_or_else(|| ApiError::input("refresh child was not bound"))
        };
        self.refresh = Some(PreparedRefresh {
            declaration: declaration.clone(),
            sigma: (sigma.production().clone(), sigma.manifest().clone()),
            fresh: take(fresh)?,
            smudge: take(smudge)?,
            preimage,
            public: trapdoor.public_matrix(),
            bound,
        });
        Ok(())
    }
}

impl RnsPreprocessBuild<'_> {
    /// The entries of every integer data family of a decoder's or refresh's
    /// child Program this stage exports, by artifact name (`data/<id>` under
    /// the child's artifact names); `None` for a family bound by shape only.
    /// The stage's own Program data are not exported: online evaluation is
    /// given them, and the binding commits to them.
    /// Children must bind inputs in the parent's instance namespace before composition.
    fn check_child_namespace(&self, child: &Self) -> Result<(), ApiError> {
        if self.artifact_prefix != child.artifact_prefix {
            return Err(ApiError::input("preprocess child must use its parent's artifact prefix"));
        }
        Ok(())
    }

    /// Data published by composed children, including the enclosing instance namespace.
    fn published_child_data(&self) -> BTreeMap<String, Option<Vec<usize>>> {
        let mut published = BTreeMap::new();
        let mut add = |child: &PublicKeyLowering<'_>, name: &dyn Fn(&str) -> String| {
            for id in child.data.keys() {
                published.insert(
                    instance_name(self.artifact_prefix.as_deref(), &name(&format!("data/{id:?}"))),
                    child.data_values.as_ref().and_then(|data| data.values(*id)).map(<[_]>::to_vec),
                );
            }
        };
        if let (Some(declaration), Some(child)) =
            (&self.decoder_declaration, &self.decoder_smudge_child)
        {
            let key = declaration.artifact_key();
            add(child, &|role| decoder_smudge_artifact_name(&key, role));
        }
        if let Some(refresh) = &self.refresh {
            let spec = *refresh.declaration.spec();
            add(&refresh.fresh, &|role| {
                crate::rns_bgg::refresh::refresh_fresh_artifact_name(spec, role)
            });
            let key = refresh.declaration.decoder().artifact_key();
            add(&refresh.smudge, &|role| decoder_smudge_artifact_name(&key, role));
        }
        published
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
    for (id, data) in &child.data {
        graph = graph
            .transferred_output(name(&format!("data/{id:?}")), data.clone())
            .map_err(ApiError::scheme)?;
    }
    Ok(graph)
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
        match (dynamic, payload) {
            (false, Some(_)) | (true, None) => {}
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
            &crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
        );
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
        if let Some(declaration) = &self.decoder_declaration {
            digest.update(declaration.commitment());
        }
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
        if let Some(refresh) = &self.refresh {
            digest.update(refresh.declaration.spec());
        }
        // The children's data, by their artifact names (none for a stage
        // without them, whose identity is unchanged).
        let child_data = self.published_child_data();
        if !child_data.is_empty() {
            digest.update(b"child-data");
            digest.update(
                serde_json::to_vec(&child_data).expect("child data have a canonical encoding"),
            );
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
        let child_data = self.published_child_data();
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
                |role| name(&decoder_smudge_artifact_name(&key, role)),
                |wire, port| name(&decoder_smudge_input_key_artifact_name(&key, wire, port)),
            )?;
        }
        if let Some(refresh) = &self.refresh {
            let spec = *refresh.declaration.spec();
            graph = publish_child(
                graph,
                &refresh.fresh,
                |role| name(&crate::rns_bgg::refresh::refresh_fresh_artifact_name(spec, role)),
                |wire, port| {
                    name(&crate::rns_bgg::refresh::refresh_fresh_input_key_artifact_name(
                        spec, wire, port,
                    ))
                },
            )?;
            let key = refresh.declaration.decoder().artifact_key();
            graph = publish_child(
                graph,
                &refresh.smudge,
                |role| name(&decoder_smudge_artifact_name(&key, role)),
                |wire, port| name(&decoder_smudge_input_key_artifact_name(&key, wire, port)),
            )?
            .transferred_output(
                name(&decoder_artifact_name(&key, "unmask-preimage")),
                refresh.preimage.clone(),
            )
            .map_err(ApiError::scheme)?;
            graph = graph
                .transferred_output(
                    name(&crate::rns_bgg::refresh::refresh_artifact_name(spec, "unmask-public")),
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
        // The unmasking of the decoder: `K` for online evaluation and `B` for
        // encryption. The trapdoor itself stays in the setup production.
        if let Some((key, preimage, public)) = &evaluator.unmask {
            graph = graph
                .transferred_output(
                    name(&decoder_artifact_name(key, "unmask-preimage")),
                    preimage.clone(),
                )
                .map_err(ApiError::scheme)?
                .transferred_output(
                    name(&decoder_artifact_name(key, "unmask-public")),
                    public.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
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
        .with_outputs(evaluator.output_use, carriers)
        .with_child_data(child_data);
        if !evaluator.program.data_families().is_empty() && self.graph_data.is_none() {
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
        // A refresh reads the key of the caller's encoding of `sigma_0 G` from its production.
        let manifests = self.refresh.map(|refresh| refresh.sigma).into_iter().collect();
        Ok(GraphEmission {
            graph,
            runtime_inputs: self.runtime,
            manifests,
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
        let graph = DslContext::new("obfuscate-like")
            .transferred_output("initial-private", private_row)
            .unwrap()
            .transferred_output("initial-public", public_row)
            .unwrap();
        let (graph, descriptor) = preprocessing.emit(graph, keys).unwrap().build().unwrap();
        let obfuscation = crate::api::execute_artifacts(
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
            obfuscation.production().clone(),
            obfuscation.manifest().clone(),
            obfuscation.descriptor().1.clone(),
            obfuscation.session().clone(),
        );
        let row = |name: &str| {
            (0..2)
                .map(|port| {
                    ring.artifact_input(
                        obfuscation.production().clone(),
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
        let (selectors, data) =
            declaration.smudge().prf().key_inputs::<DCRTPoly>(&level, &keys).unwrap();
        let payload = crate::rns_bgg::RnsBggPayloadSecret {
            row: (
                smudge_parameters.ring(0).unwrap().identity(1),
                smudge_parameters.ring(0).unwrap().identity(1),
            ),
            binding: None,
        };
        let mut child = RnsPreprocessBuild::new(&smudge_parameters, [1; 32]).with_data(data);
        child
            .bind_rhs_inputs(
                declaration.smudge().program(),
                &mxx_ir_core::ParamEnv::default(),
                Some(&selectors),
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
