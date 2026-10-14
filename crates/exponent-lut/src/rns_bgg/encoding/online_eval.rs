//! The online evaluation stage adapter and its decoding.
use super::*;

/// The family of input encodings whose fields are the imported families:
/// four public-key parts, two row parts, and an optional payload.
fn encoding_family(
    wire: WireId,
    key_parts: &[Family<Mat>],
    rows: &[Family<Mat>],
    payload: Option<&Family<Mat>>,
    count: IntExpr,
) -> Result<Family<RnsBggEncoding>, ApiError> {
    let ([k0, k1, k2, k3], [r0, r1]) = (key_parts, rows) else {
        return Err(ApiError::input("an input encoding family has the wrong field count"));
    };
    let first = |family: &Family<Mat>| family.at(Int::constant(0));
    let element = RnsBggEncoding::new(
        RnsBggPublicKey::new(wire, ((first(k0), first(k1)), (first(k2), first(k3)))),
        (first(r0), first(r1)),
        payload.map(first),
    );
    let fields = key_parts
        .iter()
        .chain(rows)
        .chain(payload)
        .map(|family| family.value_handle().clone())
        .collect::<Vec<_>>();
    family_of_fields(&element, count, &fields).map_err(ApiError::scheme)
}

/// The public payloads of a PRF input family on `ring`: member `i < v` is
/// the signed monomial `X^(a_i stride)` of label value `a_i` in `layout` (its
/// rotation reduced modulo `2n`, negated past `X^n`) and member `v` is
/// `mono(0) = 1`, as [`crate::program_gadgets::prf::SparseLwrPrf::plaintext_inputs`]
/// encodes them.
fn label_payloads(
    ring: &mxx_dsl::Ring,
    values: &[usize],
    layout: ExpLayout,
    count: &IntExpr,
) -> Result<Family<Mat>, ApiError> {
    let dimension = ring.ring_dimension() as usize;
    let (coefficients, exponents): (Vec<_>, Vec<_>) = values
        .iter()
        .copied()
        .chain([0])
        .map(|value| {
            let rotation = (value * layout.stride()) % (2 * dimension);
            let sign = if rotation >= dimension { -1 } else { 1 };
            (IntExpr::constant(sign), IntExpr::constant(rotation % dimension))
        })
        .unzip();
    if IntExpr::constant(coefficients.len()) != *count {
        return Err(ApiError::input("the refresh label does not match the PRF input family"));
    }
    mxx_dsl::parallel(count.clone(), |index| {
        let select = |branches: &Vec<IntExpr>| {
            Ok::<_, DslError>(Int::from(IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: branches.clone(),
            }))
        };
        Ok(crate::rns_bgg::signed_monomial(ring, select(&coefficients)?, select(&exponents)?))
    })
    .map_err(ApiError::scheme)
}

/// Named hash-key graph inputs and their key bytes.
type HashKeyInputs = Vec<(String, Vec<u8>)>;

impl<'a> RnsOnlineEvalBuild<'a> {
    /// Creates an unbound online evaluation builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            hash_key: Vec::new(),
            evaluator: None,
            families: OnlineEvalFamilies::new(parameters),
            environment: Environment::new(),
            pending_decoder: None,
            unmask_preimage_bound: None,
            output_use: RnsOutputUse::default(),
            output_carriers: Vec::new(),
            dynamic: None,
            data: crate::program::ProgramData::new(),
            hash_namespace: None,
            pending_refresh: None,
            inputs_by_shape: false,
            runtime: BTreeMap::new(),
        }
    }

    /// Bind every integer data family of the child Program `program` to the
    /// entries preprocessing exported under `name("data/<id>")` (recorded in
    /// its binding `preprocess`), as a resident graph input of the same name:
    /// a protocol links the input to that artifact. The coordinate tables of
    /// `prf` must be those of its committed key set. A family preprocessing
    /// bound by shape only is bound with its declared count and no value.
    /// Returns the data inputs, which the child's lowering reads.
    fn bind_child_data(
        &mut self,
        program: &ExponentLutProgram,
        preprocess: &RnsBggArtifactBinding,
        prf: &crate::program_gadgets::prf::SparseLwrPrf,
        name: impl Fn(&str) -> String,
    ) -> Result<BTreeMap<crate::program::DataId, Family<Int>>, ApiError> {
        let mut bound = crate::program::ProgramData::new();
        let mut shape_only = false;
        let mut inputs = BTreeMap::new();
        for (id, declaration) in program.data_families() {
            let name = name(&format!("data/{id:?}"));
            let entries = preprocess
                .child_data(&name)
                .ok_or_else(|| ApiError::compatibility("preprocessing exported no such data"))?;
            match entries {
                Some(entries) => {
                    self.runtime.insert(
                        name.clone(),
                        RuntimeValue::integer_values(
                            entries.iter().map(|entry| BigInt::from(*entry)).collect(),
                        ),
                    );
                    bound = bound.with(*id, entries.to_vec());
                }
                None => shape_only = true,
            }
            inputs.insert(*id, mxx_dsl::int_family_input(name, declaration.count()));
        }
        // Data bound by shape only have no entries to check.
        if !shape_only {
            bound.validate(program).map_err(|_| {
                ApiError::compatibility("the exported data do not match the Program")
            })?;
            prf.check_tables(&bound).map_err(|_| {
                ApiError::compatibility("the exported data are not those of the key set")
            })?;
        }
        Ok(inputs)
    }

    /// The value of the caller's `A'` of a refresh (see
    /// `RnsPreprocessBuild::with_refresh_matrix`).
    pub fn with_refresh_matrix(mut self, a_prime: RuntimeValue) -> Self {
        self.runtime.insert(crate::rns_bgg::REFRESH_MATRIX_INPUT.to_owned(), a_prime);
        self
    }

    /// This builder binding the Program's integer data by their declared shapes only (graph
    /// inputs without values, unchecked): with a preprocessing bound by shape (see
    /// `RnsPreprocessBuild::with_inputs_by_shape`), the stage graph is recorded and exported
    /// but cannot be executed.
    pub fn with_inputs_by_shape(mut self) -> Self {
        self.inputs_by_shape = true;
        self
    }

    /// This builder for a Program with integer data ([`crate::program::ProgramData`]): the
    /// data its preprocessing was given.
    pub fn with_data(mut self, data: crate::program::ProgramData) -> Self {
        self.data = data;
        self
    }

    /// This builder for a Program bound to the Dynamic `setup` through
    /// `binding`, which must be the setup and mapping it was preprocessed
    /// with; the encryption must belong to the same setup.
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

    /// Check the Dynamic compatibility of `preprocess` (the compiled
    /// Program) and `encryption` (the ciphertext): both belong to the bound
    /// setup, the Program was compiled with the bound mapping, and the
    /// ciphertext names no Program unless it carries a decoder, whose
    /// unmask hint the Program's preprocessing fixes.
    fn validate_dynamic_stages(
        &self,
        program: &ExponentLutProgram,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
    ) -> Result<(Arc<RnsBggDynamicSetup>, RnsBggDynamicBinding), ApiError> {
        let (setup, binding) = self.dynamic.clone().ok_or_else(|| {
            ApiError::input("Dynamic evaluation needs its setup (with_dynamic_setup)")
        })?;
        binding.validate(&setup, program).map_err(ApiError::scheme)?;
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        if preprocess_binding.dynamic_setup() != Some(&setup.identity()) ||
            encryption_binding.dynamic_setup() != Some(&setup.identity())
        {
            return Err(ApiError::compatibility("the stages belong to another Dynamic setup"));
        }
        if preprocess_binding.dynamic_binding() != Some(&binding) {
            return Err(ApiError::compatibility(
                "the Program was compiled with another setup mapping",
            ));
        }
        let decoded = preprocess_binding.decoder_commitment().is_some();
        let expected_sources: &[ProductionId] =
            if decoded { std::slice::from_ref(preprocess.production()) } else { &[] };
        if encryption_binding.sources() != expected_sources ||
            (!decoded && encryption_binding.program().is_some())
        {
            return Err(ApiError::compatibility(
                "a Dynamic ciphertext names a Program it was not encrypted for",
            ));
        }
        Ok((setup, binding))
    }

    /// This builder with the output use `output_use`, which must be the one
    /// the Program was preprocessed with.
    pub fn with_output_use(mut self, output_use: RnsOutputUse) -> Self {
        self.output_use = output_use;
        self
    }

    /// Bind this builder, on [`RnsBggParameters::decoder_smudge_parameters`],
    /// to `declaration`'s smudge Program: its inputs and material are imported
    /// from the source's preprocess and encryption productions under the
    /// decoder namespace, and its smudge LUT members are finalized and
    /// transformed in their root bodies.
    pub fn bind_decoder_smudge(
        &mut self,
        declaration: &DecoderDeclaration,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if declaration.target() != crate::decoder::decoding::DecoderTarget::Output {
            return Err(ApiError::input(
                "a refresh decoder binds through bind_refresh_smudge_child",
            ));
        }
        let commitment = *declaration.commitment();
        let key = declaration.artifact_key();
        self.bind_child(
            declaration.smudge().program(),
            declaration.smudge().prf(),
            commitment,
            Some(declaration.clone()),
            decoder_artifact_name(&key, "smudge"),
            None,
            preprocess,
            encryption,
            hash_key,
        )
    }

    /// Bind this builder to the fresh-noise child of part `part` (`(chain, component)`) of
    /// `declaration`'s refresh; it must carry [`RefreshDeclaration::fresh_namespace`] (see
    /// [`Self::with_hash_namespace`]). Its inputs and material are imported under the child's
    /// namespace; the public payload of its PRF label is the label hashed from `label_tag` (see
    /// [`RefreshDeclaration::label`]), not the encrypted one.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_refresh_fresh_child(
        &mut self,
        declaration: &RefreshDeclaration,
        part: (usize, usize),
        label_tag: &[u8],
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let declaration =
            crate::rns_bgg::refresh::refresh_part(declaration, part).map_err(ApiError::scheme)?;
        let spec = *declaration.spec();
        self.bind_refresh_child(
            declaration,
            (declaration.fresh_namespace(), declaration.fresh().program()),
            declaration.fresh().sampler().prf(),
            None,
            crate::rns_bgg::refresh::refresh_artifact_name(spec, "fresh"),
            label_tag,
            (preprocess, encryption),
            hash_key,
        )
    }

    /// Bind this builder to the smudge child of the decoder of part `part` of `declaration`'s
    /// refresh (see [`Self::bind_refresh_fresh_child`]); it must carry the decoder's commitment
    /// ([`RefreshDeclaration::decoder`]) as its hash namespace.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_refresh_smudge_child(
        &mut self,
        declaration: &RefreshDeclaration,
        part: (usize, usize),
        label_tag: &[u8],
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let declaration =
            crate::rns_bgg::refresh::refresh_part(declaration, part).map_err(ApiError::scheme)?;
        let decoder = declaration.decoder();
        self.bind_refresh_child(
            declaration,
            (*decoder.commitment(), decoder.smudge().program()),
            decoder.smudge().prf(),
            Some(decoder.clone()),
            decoder_artifact_name(&decoder.artifact_key(), "smudge"),
            label_tag,
            (preprocess, encryption),
            hash_key,
        )
    }

    /// Bind this builder to the refresh child `program` of the part `declaration`, under the
    /// hash namespace `namespace` and the artifact prefix `prefix`, with its PRF `prf`'s label
    /// from `label_tag`; `decoder_smudge` for the decoder's smudge child.
    #[allow(clippy::too_many_arguments)]
    fn bind_refresh_child(
        &mut self,
        declaration: &RefreshPartDeclaration,
        (namespace, program): ([u8; 32], &ExponentLutProgram),
        prf: &crate::program_gadgets::prf::SparseLwrPrf,
        decoder_smudge: Option<DecoderDeclaration>,
        prefix: String,
        label_tag: &[u8],
        (preprocess, encryption): (
            &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
            &StageArtifacts<RnsBggEncryptionDescriptor>,
        ),
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let key: [u8; 32] =
            hash_key.try_into().map_err(|_| ApiError::input("the hash key is not 32 bytes"))?;
        let label =
            declaration.label(key, label_tag).map_err(|_| ApiError::input("refresh label"))?;
        if self.hash_namespace != Some(namespace) {
            return Err(ApiError::input("the refresh child's hash namespace does not match"));
        }
        let label = Some((prf.input_vector(), label, prf.profile().layout));
        self.bind_child(
            program,
            prf,
            *declaration.spec(),
            decoder_smudge,
            prefix,
            label,
            preprocess,
            encryption,
            hash_key,
        )
    }

    /// Make this builder a refresh child that samples its public matrices
    /// from the stage hash key derived under `namespace`.
    pub fn with_hash_namespace(mut self, namespace: [u8; 32]) -> Self {
        self.hash_namespace = Some(namespace);
        self
    }

    /// Bind this child builder to `prepared`, whose stage artifacts carry the
    /// decoder or refresh commitment `commitment`: its inputs are imported
    /// under `name("input-<wire>.<port>")` and its RHS material under
    /// `name(role)`; a `decoder_smudge` child transforms its smudge members;
    /// `label`, the PRF's public input family with its label values and
    /// exponent layout, replaces that family's encrypted public payload;
    /// the integer data of `prepared` (the coordinate tables of `prf`, the
    /// child's PRF) are bound from the entries preprocessing exported under
    /// `name("data/<id>")`.
    #[allow(clippy::too_many_arguments)]
    fn bind_child(
        &mut self,
        prepared: &ExponentLutProgram,
        prf: &crate::program_gadgets::prf::SparseLwrPrf,
        commitment: [u8; 32],
        decoder_smudge: Option<DecoderDeclaration>,
        prefix: String,
        label: Option<(WireId, Vec<usize>, ExpLayout)>,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let name = |role: &str| format!("{prefix}/{role}");
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        let expected_hash_key = hash_key_commitment(hash_key);
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.hash_key_commitment() != &expected_hash_key ||
                binding.decoder_commitment() != Some(&commitment)
            {
                return Err(ApiError::compatibility(
                    "decoder smudge artifacts do not match the source declaration",
                ));
            }
        }
        // A Dynamic ciphertext names no Program.
        if decoder_smudge.as_ref().is_some_and(|decoder| {
            [preprocess_binding, encryption_binding].iter().any(|binding| {
                binding.program().is_some_and(|program| program != decoder.source_program())
            })
        }) {
            return Err(ApiError::compatibility(
                "decoder smudge artifacts do not match the source declaration",
            ));
        }
        if !preprocess_binding.sources().is_empty() ||
            encryption_binding.sources().len() != 1 ||
            !encryption_binding.sources().contains(preprocess.production())
        {
            return Err(ApiError::compatibility(
                "RNS-BGG decoder encryption source is not the preprocess production",
            ));
        }
        let preprocess_production = preprocess.production().clone();
        let encryption_production = encryption.production().clone();
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let prelude = encryption_binding.ciphertext_prelude().map(|(production, manifest)| {
            manifests.insert(production.clone(), manifest.clone());
            production.clone()
        });
        let workload = Arc::new(
            RnsBggDerivedWorkload::from_program_with_parameters(self.parameters, prepared)
                .map_err(ApiError::scheme)?,
        );
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = match &self.hash_namespace {
            Some(namespace) => crate::rns_bgg::refresh::child_hash_key(hash_key, namespace),
            None => hash_key.to_vec(),
        };
        // The child's encryption publishes each input at its logical level
        // with its public key, under the child's namespace.
        self.environment = Environment::new();
        let hash_key = self.parameters.ring(0).map_err(ApiError::scheme)?.bytes_input(
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            self.hash_key.len(),
        );
        // A Dynamic child's inputs are encodings of its setup's coordinates
        // published under the child's namespace; a Static child's keys are
        // published by the encryption as well.
        let dynamic = self.dynamic.clone();
        if dynamic.is_some() {
            self.bind_encrypted_inputs(
                prepared,
                &workload,
                &encryption_production,
                &hash_key,
                dynamic.as_ref(),
                label.as_ref(),
                |role| name(&role),
            )?;
        } else {
            for (wire, input) in prepared.inputs() {
                let public = matches!(input.knowledge(), AttributeKnowledge::PublicPayload);
                let level = workload.level(*wire).map_err(ApiError::scheme)?;
                let ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
                let columns = self
                    .parameters
                    .key_component_columns(level, input.knowledge())
                    .map_err(ApiError::scheme)?;
                let port = |port: usize, columns: usize| {
                    ring.artifact_input(
                        encryption_production.clone(),
                        name(&format!("input-{wire:?}.{port}")),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    )
                };
                let family_port = |port: usize, count: &IntExpr, columns: usize| {
                    ring.family_artifact_input(
                        encryption_production.clone(),
                        name(&format!("input-{wire:?}.{port}")),
                        count.clone(),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    )
                };
                let binding = match input.shape() {
                    InputBinding::Scalar(()) => InputBinding::Scalar(RnsBggEncoding::new(
                        RnsBggPublicKey::new(
                            *wire,
                            (
                                (port(0, columns), port(1, columns)),
                                (port(2, columns), port(3, columns)),
                            ),
                        ),
                        (port(4, columns), port(5, columns)),
                        public.then(|| port(6, 1)),
                    )),
                    InputBinding::Family(count) => {
                        let key_parts = (0..4)
                            .map(|index| family_port(index, count, columns))
                            .collect::<Vec<_>>();
                        let rows = (4..6)
                            .map(|index| family_port(index, count, columns))
                            .collect::<Vec<_>>();
                        let payload = match &label {
                            Some((label_wire, values, layout)) if label_wire == wire => {
                                Some(label_payloads(&ring, values, *layout, count)?)
                            }
                            _ => public.then(|| family_port(6, count, 1)),
                        };
                        InputBinding::Family(OnlineEvalWireFamily::Stored(encoding_family(
                            *wire,
                            &key_parts,
                            &rows,
                            payload.as_ref(),
                            count.clone(),
                        )?))
                    }
                };
                self.environment
                    .bind_wire(*wire, binding)
                    .map_err(|error| ApiError::input(error.to_string()))?;
            }
        }
        let data = self.bind_child_data(prepared, preprocess_binding, prf, name)?;
        self.families.data = data.clone();
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            prepared,
            &workload,
            &preprocess_production,
            prelude.as_ref(),
            dynamic.as_ref().map(|(setup, binding)| (setup.as_ref(), binding)),
            &hash_key,
            |role: String| name(&role),
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let mut evaluator = EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(prepared.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            decoder_smudge: decoder_smudge.map(Arc::new),
            artifact_prefix: Some(prefix),
            current_site: None,
            hash_key,
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            data,
            output_use: RnsOutputUse::default(),
            dynamic: dynamic.map(|(setup, binding)| DynamicEvaluation {
                setup,
                binding,
                automorphisms: None,
            }),
        };
        if evaluator.dynamic.is_some() {
            evaluator.import_dynamic_packages().map_err(ApiError::scheme)?;
        } else {
            evaluator.import_body_lut_material().map_err(ApiError::scheme)?;
        }
        self.evaluator = Some(evaluator);
        Ok(())
    }

    /// Apply `declaration`'s decoder to output `output_index` when the
    /// graph is built: the bound smudge child `smudge` is lowered once, and that
    /// output becomes the plaintext `mu mod q_(L+1)` (see
    /// [`crate::decoder`]) instead of an encoding.
    pub fn apply_decoder(
        &mut self,
        output_index: usize,
        declaration: &DecoderDeclaration,
        smudge: RnsOnlineEvalBuild<'a>,
    ) -> Result<(), ApiError> {
        if self.pending_decoder.is_some() {
            return Err(ApiError::input("decoder was already applied"));
        }
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
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
                Some(&Operand::Scalar(declaration.source_output()))
        {
            return Err(ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch)));
        }
        self.pending_decoder = Some((output_index, declaration.clone(), Box::new(smudge)));
        Ok(())
    }

    /// The plaintext `round((p/q)(b K - c))` of the pending decoder, for the
    /// summed decode row `c` of the source and the lowered smudge.
    fn decode_output(
        &mut self,
        output_index: usize,
        declaration: &DecoderDeclaration,
        mut smudge: RnsOnlineEvalBuild<'a>,
        outputs: &[RnsBggEncoding],
    ) -> Result<Mat, RnsBggError> {
        let source = outputs
            .get(output_index)
            .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?
            .clone();
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        let level = evaluator.workload.level(source.wire())?;
        let mut decoder = RnsBggDecoder { lowerer: evaluator, source };
        let encoding = apply_decoder(declaration, &mut decoder, |_| {
            let (child, families, environment) = smudge
                .lowering_state_inner()
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            let program = child.program.clone();
            crate::program::lower_program(&program, child, families, environment)
                .map_err(|error| RnsBggError::Graph(error.to_string()))
        })?;
        let child_manifests =
            smudge.evaluator.as_ref().map(|child| child.manifests.clone()).unwrap_or_default();
        self.runtime.append(&mut smudge.runtime);
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        evaluator.manifests.extend(child_manifests);
        let key = declaration.artifact_key();
        let ring = self.parameters.ring(level)?;
        let columns = mxx_ir_core::types::TrapdoorInstantiation::Computational
            .columns(
                crate::decoder::unmask::UNMASK_ROWS,
                self.parameters.level(level)?.modulus_digits(),
            )
            .ok_or(RnsBggError::Parameters("the unmask trapdoor width overflows"))?;
        let bound = self
            .unmask_preimage_bound
            .clone()
            .ok_or(RnsBggError::Binding("the unmask preimage bound is not bound"))?;
        let preimage = ring.preimage_artifact_input(
            evaluator.preprocess.clone(),
            decoder_artifact_name(&key, "unmask-preimage"),
            (columns, 1),
            BigInt::from(bound),
            ArtifactAvailability::Transferred,
        );
        let hint = ring.artifact_input(
            evaluator.encryption.clone(),
            decoder_artifact_name(&key, "unmask-hint"),
            (1, columns),
            ArtifactAvailability::Transferred,
        );
        // The halved smudge weights leave `2^-1` in every coefficient (see
        // `crate::decoder::decoding`); removing it centers the smudge.
        let modulus: BigUint = self.parameters.level(level)?.modulus().as_ref().clone();
        let half = IntExpr::constant((modulus + 1u8) / 2u8);
        let decoded = crate::decoder::unmask::unmask(hint, preimage, encoding) -
            ring.all_coefficients((1, 1), half);
        let plaintext = self.parameters.decoder_plaintext_ring(declaration, &evaluator.program)?;
        Ok(decoded.modulus_switch(&plaintext))
    }

    /// Bind every input of `program` from the encryption production
    /// `encryption`, which stores each input's row (and public payload); its
    /// public key is derived from the hash key here, or where a family
    /// member is read. A Static input is stored at its logical level under
    /// its wire; a Dynamic one at the setup ring under its setup coordinate.
    /// `label`, a public input family with its label values and exponent
    /// layout, replaces that family's encrypted public payload (see
    /// [`Self::bind_refresh_fresh_child`]). `name` maps a role to its artifact name.
    fn bind_encrypted_inputs(
        &mut self,
        program: &ExponentLutProgram,
        workload: &RnsBggDerivedWorkload,
        encryption: &ProductionId,
        hash_key: &Bytes,
        dynamic: Option<&(Arc<RnsBggDynamicSetup>, RnsBggDynamicBinding)>,
        label: Option<&(WireId, Vec<usize>, ExpLayout)>,
        name: impl Fn(String) -> String,
    ) -> Result<(), ApiError> {
        for (wire, declaration) in program.inputs() {
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let knowledge = declaration.knowledge();
            let (key, role, ring_level) = match dynamic {
                Some((setup, binding)) => {
                    // The encoding at the setup ring is valid at every
                    // logical level of an equal-basis chain below its last.
                    self.parameters.dynamic_route(level).map_err(ApiError::scheme)?;
                    let coordinate = binding.input(*wire).map_err(ApiError::scheme)?;
                    (
                        InputKeySource::Dynamic { setup: setup.clone(), coordinate },
                        super::dynamic::dynamic_input_role(coordinate),
                        0,
                    )
                }
                None => {
                    (InputKeySource::Static { level, knowledge }, format!("input-{wire:?}"), level)
                }
            };
            let ring = self.parameters.ring(ring_level).map_err(ApiError::scheme)?;
            let columns = self
                .parameters
                .key_component_columns(ring_level, knowledge)
                .map_err(ApiError::scheme)?;
            let public = matches!(knowledge, AttributeKnowledge::PublicPayload);
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => {
                    let port = |port: usize, columns: usize| {
                        ring.artifact_input(
                            encryption.clone(),
                            name(format!("{role}.{port}")),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    };
                    let key = key
                        .key(self.parameters, hash_key.clone(), *wire, None)
                        .map_err(ApiError::scheme)?;
                    InputBinding::Scalar(RnsBggEncoding::new(
                        key,
                        (port(0, columns), port(1, columns)),
                        public.then(|| port(2, 1)),
                    ))
                }
                InputBinding::Family(count) => {
                    let port = |port: usize, columns: usize| {
                        ring.family_artifact_input(
                            encryption.clone(),
                            name(format!("{role}.{port}")),
                            count.clone(),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    };
                    let row_ports = [port(0, columns), port(1, columns)];
                    let element: RnsBggRow =
                        (row_ports[0].at(Int::constant(0)), row_ports[1].at(Int::constant(0)));
                    let fields = row_ports
                        .iter()
                        .map(|family| family.value_handle().clone())
                        .collect::<Vec<_>>();
                    InputBinding::Family(OnlineEvalWireFamily::Input {
                        wire: *wire,
                        key,
                        hash_key: hash_key.clone(),
                        rows: family_of_fields(&element, count.clone(), &fields)
                            .map_err(ApiError::scheme)?,
                        payloads: match label {
                            Some((label_wire, values, layout)) if label_wire == wire => {
                                Some(label_payloads(&ring, values, *layout, count)?)
                            }
                            _ => public.then(|| port(2, 1)),
                        },
                    })
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        Ok(())
    }

    /// Apply part `part` (`(chain, component)`) of `declaration`'s refresh to output
    /// `target_index` when the graph is built: the bound children (`fresh` and the decoder's
    /// `smudge`) are lowered once, and the graph gains the output `refresh`, the CRT term of the
    /// part's residue `y_j mod q` of its output prime `q`, a `1 x w_0` row in the ring of the
    /// product `P` of the refreshed primes (the residue itself when `P = q`); the terms of every
    /// chain sum to `y_j mod P` (see [`crate::refresh`]).
    ///
    /// `sigma` is the row `g` of the caller's encoding of `sigma_0 G` whose key the part's
    /// preprocessing read (see [Caller inputs](crate::refresh#caller-inputs)): component 1 of
    /// that Program's first output, `1 x m` in the decoding ring (port
    /// [`crate::rns_bgg::SIGMA_ROW_OUTPUT`] of its online evaluation). The graph reads it as a
    /// plain input; without a value the stage can be recorded but not executed. `A'` is the
    /// graph input [`crate::rns_bgg::REFRESH_MATRIX_INPUT`] (see [`Self::with_refresh_matrix`]).
    pub fn apply_refresh(
        &mut self,
        target_index: usize,
        declaration: &RefreshDeclaration,
        part: (usize, usize),
        sigma: Option<RuntimeValue>,
        fresh: RnsOnlineEvalBuild<'a>,
        smudge: RnsOnlineEvalBuild<'a>,
    ) -> Result<(), ApiError> {
        let declaration =
            crate::rns_bgg::refresh::refresh_part(declaration, part).map_err(ApiError::scheme)?;
        if self.pending_decoder.is_some() || self.pending_refresh.is_some() {
            return Err(ApiError::input("a decoder or refresh was already applied"));
        }
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
        self.parameters
            .validate_refresh_children(
                declaration,
                &evaluator.program,
                fresh.parameters,
                smudge.parameters,
            )
            .map_err(ApiError::scheme)?;
        let child_program = |child: &RnsOnlineEvalBuild<'a>| {
            child.evaluator.as_ref().map(|child| child.program.id())
        };
        let mismatch = || ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch));
        let decoder = declaration.decoder();
        if evaluator.program.outputs().get(target_index) !=
            Some(&Operand::Scalar(declaration.target_output())) ||
            child_program(&fresh) != Some(declaration.fresh().id()) ||
            fresh.hash_namespace != Some(declaration.fresh_namespace()) ||
            smudge.hash_namespace != Some(*decoder.commitment()) ||
            smudge
                .evaluator
                .as_ref()
                .and_then(|child| child.decoder_smudge.as_ref())
                .map(|bound| *bound.commitment()) !=
                Some(*decoder.commitment())
        {
            return Err(mismatch());
        }
        if let Some(sigma) = sigma {
            self.runtime.insert(
                crate::rns_bgg::refresh::refresh_sigma_input_name(*declaration.spec()),
                sigma,
            );
        }
        self.pending_refresh =
            Some((target_index, declaration.clone(), Box::new(fresh), Box::new(smudge)));
        Ok(())
    }

    /// The refreshed row of the pending refresh for the lowered Program
    /// `outputs`, and the hash-key inputs of its children.
    fn refresh_row(
        &mut self,
        target_index: usize,
        declaration: &RefreshPartDeclaration,
        mut fresh: RnsOnlineEvalBuild<'a>,
        mut smudge: RnsOnlineEvalBuild<'a>,
        outputs: &[RnsBggEncoding],
    ) -> Result<(Mat, HashKeyInputs), RnsBggError> {
        use crate::rns_bgg::refresh::{
            hash_key_input_name, refresh_artifact_name, refresh_combination,
            refresh_sigma_input_name,
        };
        let target = outputs
            .get(target_index)
            .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?
            .clone();
        let lower = |child: &mut RnsOnlineEvalBuild<'a>| {
            let (lowerer, families, environment) = child
                .lowering_state_inner()
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            let program = lowerer.program.clone();
            let outputs = crate::program::lower_program(&program, lowerer, families, environment)
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            let manifests = lowerer.manifests.clone();
            Ok::<_, RnsBggError>((outputs, manifests))
        };
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        let level = evaluator.workload.level(target.wire())?;
        let ring = self.parameters.ring(level)?;
        let (fresh_outputs, mut manifests) = lower(&mut fresh)?;
        // The fresh carrier row's component-1 row, moved down once to the
        // decoding ring and never finalized for decryption.
        let fresh_row = fresh_outputs
            .into_iter()
            .next()
            .map(|encoding| (mod_down_encoding(encoding.row, &ring).1, declaration.columns()))
            .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?;
        let spec = *declaration.spec();
        let columns = self.parameters.level(level)?.modulus_digits();
        let encryption = evaluator.encryption.clone();
        let preprocess = evaluator.preprocess.clone();
        let one = (
            ring.artifact_input(
                encryption.clone(),
                refresh_artifact_name(spec, "one-encoding.0"),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
            ring.artifact_input(
                encryption.clone(),
                refresh_artifact_name(spec, "one-encoding.1"),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
        );
        let hint_columns = crate::decoder::unmask::UNMASK_ROWS * (columns + 2);
        let hint = ring.artifact_input(
            encryption,
            refresh_artifact_name(spec, "unmask-hint"),
            (1, hint_columns),
            ArtifactAvailability::Transferred,
        );
        let bound = self
            .unmask_preimage_bound
            .clone()
            .ok_or(RnsBggError::Binding("the unmask preimage bound is not bound"))?;
        let a_prime = self.parameters.refresh_matrix(level, declaration)?;
        // The one encoding: the caller's row `g` of the encoding of `sigma_0 G` (a carrier of
        // the decoding ring) for the first block.
        let one = (ring.input(refresh_sigma_input_name(spec), (1, columns)), one.1);
        let target_row =
            if declaration.component() == 0 { target.row.0.clone() } else { target.row.1.clone() };
        let output_columns = declaration.columns();
        // The halved smudge weights leave `2^-1` in every coefficient (see
        // `crate::decoder::decoding`); removing it centers the smudge.
        let modulus: BigUint = self.parameters.level(level)?.modulus().as_ref().clone();
        // The columns concatenate one scalar constant, keeping the graph's
        // node numbering, which the Lean proofs of the export name.
        let half = ring.all_coefficients((1, 1), IntExpr::constant((modulus + 1u8) / 2u8));
        let half = match output_columns {
            1 => half,
            columns => Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![half; columns]),
        };
        // The output prime's row is decoded at once: its smudge child sums each column's
        // members into its column, and the preimage has one column per decoded column.
        let decoder = declaration.decoder();
        let actions = self.parameters.refresh_actions(level, a_prime, declaration)?;
        let decoded = refresh_combination(one, target_row, fresh_row, &actions)?;
        let (smudge_outputs, child_manifests) = lower(&mut smudge)?;
        manifests.extend(child_manifests);
        let smudge_output = smudge_outputs
            .into_iter()
            .next()
            .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?;
        let encoding = decoded + checked_modulus_convert(smudge_output.row.1, &ring)?;
        let preimage = ring.preimage_artifact_input(
            preprocess,
            decoder_artifact_name(&decoder.artifact_key(), "unmask-preimage"),
            (hint_columns, output_columns),
            BigInt::from(bound),
            ArtifactAvailability::Transferred,
        );
        let unmasked = crate::decoder::unmask::unmask(hint, preimage, encoding) - half;
        // The part rounds `Q_q y + M + N` to its residue `y mod q` and outputs its CRT term
        // modulo the product `P` of the refreshed primes; the chains' terms sum to `y mod P`.
        let refreshed = crate::rns_bgg::refresh::top_residue_term(
            declaration,
            unmasked.modulus_switch(&Ring::from_crt_moduli(
                vec![IntExpr::from(declaration.output_modulus())],
                self.parameters.level(level)?.ring_dimension(),
            )),
        )?;
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        evaluator.manifests.extend(manifests);
        let hash_keys = [&fresh, &smudge]
            .into_iter()
            .map(|child| {
                (hash_key_input_name(child.hash_namespace.as_ref()), child.hash_key.clone())
            })
            .collect::<Vec<_>>();
        for child in [&mut fresh, &mut smudge] {
            self.runtime.append(&mut child.runtime);
        }
        Ok((refreshed, hash_keys))
    }

    fn lowering_state_inner(
        &mut self,
    ) -> Result<
        (
            &mut EncodingLowering<'a>,
            &mut OnlineEvalFamilies<'a>,
            &mut Environment<
                RnsBggEncoding,
                (RnsBggMatrix, RhsRef<Int>),
                OnlineEvalWireFamily,
                OnlineEvalRhsFamily,
            >,
        ),
        ApiError,
    > {
        Ok((
            self.evaluator
                .as_mut()
                .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
}

impl<'a> OnlineEvalBuild for RnsOnlineEvalBuild<'a> {
    type Evaluator = EncodingLowering<'a>;
    type Families = OnlineEvalFamilies<'a>;
    type PreprocessDescriptor = RnsBggPreprocessOnlineEvalDescriptor;
    type EncryptDescriptor = RnsBggEncryptionDescriptor;

    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                RnsBggEncoding,
                (RnsBggMatrix, RhsRef<Int>),
                OnlineEvalWireFamily,
                OnlineEvalRhsFamily,
            >,
        ),
        ApiError,
    > {
        self.lowering_state_inner()
    }

    fn bind_artifacts(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
        encryption: &StageArtifacts<Self::EncryptDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        for declaration in program.inputs().values() {
            if let InputBinding::Family(count) = declaration.shape() {
                let count =
                    count.evaluate(parameters).ok().and_then(|value| value.to_usize()).ok_or_else(
                        || ApiError::input("online evaluation input family count is not concrete"),
                    )?;
                if count == 0 {
                    return Err(ApiError::input("online evaluation input family count is empty"));
                }
            }
        }
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        let expected_parameters = self.parameters.parameter_identity();
        let preprocess_production = preprocess.production().clone();
        let encryption_production = encryption.production().clone();
        let expected_hash_key = hash_key_commitment(hash_key);
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.parameters() != &expected_parameters ||
                binding.rhs_mode() != self.parameters.rhs_mode() ||
                binding.hash_key_commitment() != &expected_hash_key
            {
                return Err(ApiError::compatibility(
                    "RNS-BGG artifact binding does not match the online evaluation",
                ));
            }
        }
        if preprocess_binding.program() != Some(program.id()) {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess binding does not match the online evaluation Program",
            ));
        }
        if !self.inputs_by_shape {
            self.data.validate(program).map_err(|error| ApiError::input(error.to_string()))?;
        }
        let expected_data = (!program.data_families().is_empty()).then(|| self.data.commitment());
        if preprocess_binding.data() != expected_data.as_ref() {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess binding has other Program data than the online evaluation",
            ));
        }
        // The data are graph inputs whose values this builder was given.
        let data = crate::program::data_inputs(program, |family| {
            crate::rns_bgg::data_input_name(self.hash_namespace.as_ref(), family)
        });
        for family in program.data_families().keys().filter(|_| !self.inputs_by_shape) {
            let values = self.data.values(*family).expect("validated program data");
            self.runtime.insert(
                crate::rns_bgg::data_input_name(self.hash_namespace.as_ref(), *family),
                RuntimeValue::integer_values(values.iter().map(|value| (*value).into()).collect()),
            );
        }
        self.families.data = data.clone();
        if !preprocess_binding.sources().is_empty() {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess root unexpectedly declares sources",
            ));
        }
        // A Static ciphertext is encrypted for one compiled Program; a
        // Dynamic one for the setup the compiled Program belongs to.
        let dynamic = match self.parameters.rhs_mode() {
            RhsMode::Static => {
                if encryption_binding.program() != Some(program.id()) ||
                    encryption_binding.sources().len() != 1 ||
                    !encryption_binding.sources().contains(&preprocess_production)
                {
                    return Err(ApiError::compatibility(
                        "RNS-BGG encryption source is not the preprocess production",
                    ));
                }
                None
            }
            RhsMode::Dynamic => {
                Some(self.validate_dynamic_stages(program, preprocess, encryption)?)
            }
        };
        if preprocess_binding.decoder_commitment() != encryption_binding.decoder_commitment() {
            return Err(ApiError::compatibility(
                "RNS-BGG decoder declaration commitment differs between stages",
            ));
        }
        if preprocess_binding.output_use() != self.output_use {
            return Err(ApiError::compatibility(
                "RNS-BGG output use differs from the preprocessed Program",
            ));
        }
        self.unmask_preimage_bound = preprocess_binding.unmask_preimage_bound().cloned();
        self.output_carriers = preprocess_binding.output_carriers().to_vec();
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let prelude = encryption_binding.ciphertext_prelude().map(|(production, manifest)| {
            manifests.insert(production.clone(), manifest.clone());
            production.clone()
        });
        let workload = Arc::new(
            RnsBggDerivedWorkload::from_program_with_parameters(self.parameters, program)
                .map_err(ApiError::scheme)?,
        );
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        let hash_key = self.parameters.ring(0).map_err(ApiError::scheme)?.bytes_input(
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            self.hash_key.len(),
        );
        self.bind_encrypted_inputs(
            program,
            &workload,
            &encryption_production,
            &hash_key,
            dynamic.as_ref(),
            None,
            |role| role,
        )?;
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            program,
            &workload,
            &preprocess_production,
            prelude.as_ref(),
            dynamic.as_ref().map(|(setup, binding)| (setup.as_ref(), binding)),
            &hash_key,
            |role| role,
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let mut evaluator = EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            decoder_smudge: None,
            artifact_prefix: None,
            current_site: None,
            hash_key,
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            data,
            output_use: self.output_use,
            dynamic: dynamic.map(|(setup, binding)| DynamicEvaluation {
                setup,
                binding,
                automorphisms: None,
            }),
        };
        if evaluator.dynamic.is_some() {
            evaluator.import_dynamic_packages().map_err(ApiError::scheme)?;
        } else {
            evaluator.import_body_lut_material().map_err(ApiError::scheme)?;
        }
        self.evaluator = Some(evaluator);
        Ok(())
    }

    fn build_graph(mut self, outputs: Vec<RnsBggEncoding>) -> Result<StageGraph, ApiError> {
        let decoded = match self.pending_decoder.take() {
            Some((index, declaration, smudge)) => Some((
                index,
                self.decode_output(index, &declaration, *smudge, &outputs)
                    .map_err(ApiError::scheme)?,
            )),
            None => None,
        };
        let refreshed = match self.pending_refresh.take() {
            Some((index, declaration, fresh, smudge)) => Some(
                self.refresh_row(index, &declaration, *fresh, *smudge, &outputs)
                    .map_err(ApiError::scheme)?,
            ),
            None => None,
        };
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
        let carriers = outputs
            .iter()
            .map(|output| {
                let level = evaluator.workload.level(output.wire())?;
                self.parameters.output_carrier(
                    &evaluator.program,
                    output.wire(),
                    level,
                    evaluator.output_use,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::scheme)?;
        if carriers != self.output_carriers {
            return Err(ApiError::compatibility(
                "online evaluation output carriers differ from the preprocessed ones",
            ));
        }
        let outputs = evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-online-eval");
        let mut hash_keys = vec![(
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            self.hash_key.clone(),
        )];
        if let Some((row, child_keys)) = refreshed {
            graph = graph.output("refresh", row).map_err(ApiError::scheme)?;
            hash_keys.extend(child_keys);
        }
        for (index, output) in outputs.into_iter().enumerate() {
            // A decoded output is its plaintext modulo `q_(L+1)`, not an
            // encoding.
            graph = match &decoded {
                Some((decoded, plaintext)) if *decoded == index => {
                    graph.output(format!("output-{index}"), plaintext.clone())
                }
                _ => graph.output(format!("output-{index}"), output),
            }
            .map_err(ApiError::scheme)?;
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let mut runtime = super::declared_runtime_inputs(&graph, std::mem::take(&mut self.runtime));
        for (name, key) in hash_keys {
            if super::graph_declares_input(&graph, &name) {
                runtime.insert(name, RuntimeValue::Bytes(key.as_slice().into()));
            }
        }
        Ok(StageGraph::new(graph, runtime, evaluator.manifests.clone()))
    }
}
