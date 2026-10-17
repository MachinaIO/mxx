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
    /// Select this instance's namespace before binding, which checks the preprocess descriptor.
    pub fn with_artifact_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.artifact_prefix = Some(prefix.into());
        self
    }

    /// Bind root-ring rows and public plaintext expressions, without an encryption stage.
    ///
    /// `rows` covers exactly the Program inputs; `public_payloads` covers exactly its
    /// public inputs. Each family retains the Program's symbolic cardinality. Rooted
    /// chains use the tags supplied to `with_root_inputs` and retarget each row at its
    /// consuming level. Static RHS material is read from `preprocess`'s namespace.
    /// Dynamic ciphertext packages still require `bind_artifacts`.
    pub fn bind_rows(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &ParamEnv,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        rows: BTreeMap<WireId, InputBinding<RnsBggRow, crate::program::GraphFamily<RnsBggRow>>>,
        public_payloads: BTreeMap<WireId, InputBinding<Mat, crate::program::GraphFamily<Mat>>>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if rows.len() != program.inputs().len() ||
            rows.keys().any(|wire| !program.inputs().contains_key(wire)) ||
            public_payloads.len() !=
                program
                    .inputs()
                    .values()
                    .filter(|input| input.knowledge() == AttributeKnowledge::PublicPayload)
                    .count() ||
            public_payloads.keys().any(|wire| {
                program
                    .inputs()
                    .get(wire)
                    .is_none_or(|input| input.knowledge() != AttributeKnowledge::PublicPayload)
            })
        {
            return Err(ApiError::input(
                "graph rows or public payloads do not match Program inputs",
            ));
        }
        self.bind_sources(program, parameters, preprocess, None, hash_key)?;
        self.evaluator.as_mut().expect("bound sources").root_lut_helpers = Some(BTreeMap::new());
        let evaluator = self.evaluator.as_ref().expect("bound sources");
        for (wire, input) in program.inputs() {
            let level = evaluator.workload.level(*wire).map_err(ApiError::scheme)?;
            let key = InputKeySource::Static {
                level,
                knowledge: input.knowledge(),
                root: self
                    .root_inputs
                    .get(wire)
                    .map(|tag| {
                        self.parameters
                            .root_hash_key_input()
                            .map(|hash_key| (tag.clone(), hash_key))
                            .ok_or_else(|| ApiError::input("a root input needs a rooted chain"))
                    })
                    .transpose()?,
            };
            let payload = public_payloads.get(wire);
            let encoding = |row, payload, index| {
                graph_input_encoding(
                    self.parameters,
                    *wire,
                    level,
                    input.knowledge(),
                    &key,
                    evaluator.hash_key.clone(),
                    row,
                    payload,
                    index,
                )
            };
            let binding = match (input.shape(), &rows[wire], payload) {
                (InputBinding::Scalar(()), InputBinding::Scalar(row), payload) => {
                    let payload = match payload {
                        Some(InputBinding::Scalar(value)) => Some(value.clone()),
                        None => None,
                        _ => return Err(ApiError::input("public payload shape differs from input")),
                    };
                    InputBinding::Scalar(
                        encoding(row.clone(), payload, None).map_err(ApiError::scheme)?,
                    )
                }
                (InputBinding::Family(count), InputBinding::Family(rows), payload) => {
                    if rows.count() != count || payload.is_some_and(|payload|
                        !matches!(payload, InputBinding::Family(values) if values.count() == count))
                    {
                        return Err(ApiError::input("graph row or public payload family count differs"));
                    }
                    // Each member is encoded where the Program reads it.
                    InputBinding::Family(OnlineEvalWireFamily::Graph {
                        wire: *wire,
                        level,
                        knowledge: input.knowledge(),
                        key: key.clone(),
                        hash_key: evaluator.hash_key.clone(),
                        rows: rows.clone(),
                        payloads: match payload {
                            Some(InputBinding::Family(values)) => Some(values.clone()),
                            _ => None,
                        },
                    })
                }
                _ => return Err(ApiError::input("graph row shape differs from Program input")),
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        Ok(())
    }

    /// Bind root-ring LUT helper rows by `LutTable::helper_family_key`, after `bind_rows`: the
    /// helper of branch `i` of a key's layout is member `positions[i]` of its family `rows`, read
    /// and re-targeted where the branch is evaluated, at the LUT's consuming level.
    pub fn bind_lut_helper_rows(
        &mut self,
        helpers: BTreeMap<String, (Family<RnsBggRow>, Vec<IntExpr>)>,
    ) -> Result<(), ApiError> {
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_rows first"))?;
        let target = evaluator
            .root_lut_helpers
            .as_mut()
            .ok_or_else(|| ApiError::input("LUT graph helpers require bind_rows"))?;
        *target = helpers;
        Ok(())
    }

    /// Bind the graph-produced hint used when a decoder is applied to graph rows.
    pub fn bind_decoder_hint(&mut self, hint: Mat) -> Result<(), ApiError> {
        let evaluator =
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_rows first"))?;
        if evaluator.root_lut_helpers.is_none() {
            return Err(ApiError::input("a graph decoder hint requires bind_rows"));
        }
        evaluator.decoder_hint = Some(hint);
        Ok(())
    }

    /// Bind the Program's integer data families to graph values, before the rows, as their
    /// preprocessing did (`RnsPreprocessBuild::bind_data_values`). They replace
    /// [`Self::with_data`] and its resident inputs.
    pub fn bind_data_values(
        &mut self,
        program: &ExponentLutProgram,
        data: BTreeMap<crate::program::DataId, (Family<Int>, Int)>,
    ) -> Result<(), ApiError> {
        if self.evaluator.is_some() {
            return Err(ApiError::input("bind graph data before the rows"));
        }
        let (data, offsets) = crate::program::validate_data_values(program, data)
            .map_err(|error| ApiError::input(error.to_string()))?;
        self.graph_data = Some(data);
        self.families.data_offsets = offsets;
        Ok(())
    }

    /// Read member `member` of every Static preprocessing artifact, before the rows. A graph
    /// that preprocesses one Program for `members` instances (in a parallel loop, under a hash
    /// domain per instance, see `RnsBggParameters::with_hash_domain`) publishes each artifact
    /// role as one family of the instances' values: a matrix role as `members` matrices, a
    /// family role of `count` matrices as the `members count` matrices of the instances side
    /// by side. `member` may be a runtime integer.
    pub fn with_material_member(mut self, member: Int, members: impl Into<IntExpr>) -> Self {
        self.material_member = Some((member, members.into()));
        self
    }

    /// The resident inputs of this evaluation: its hash keys and given data. A composing
    /// caller that lowers the Program without [`Self::emit`] binds them in its graph.
    pub fn runtime_inputs(&self) -> BTreeMap<String, RuntimeValue> {
        let mut runtime = self.runtime.clone();
        runtime.insert(
            instance_name(
                self.artifact_prefix.as_deref(),
                &crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            ),
            RuntimeValue::Bytes(self.hash_key.as_slice().into()),
        );
        if let Some(root) = self.parameters.root() {
            runtime.insert(
                root.hash_key_input_name(),
                RuntimeValue::Bytes(root.hash_key().as_slice().into()),
            );
        }
        runtime
    }

    /// Emit evaluation into `graph`, leaving freezing and execution to its caller.
    pub fn emit(
        self,
        graph: DslContext,
        outputs: Vec<RnsBggEncoding>,
    ) -> Result<GraphEmission<()>, ApiError> {
        <Self as OnlineEvalBuild>::emit(self, graph, outputs)
    }

    /// Creates an unbound online evaluation builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            artifact_prefix: None,
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
            graph_data: None,
            material_member: None,
            hash_namespace: None,
            pending_refresh: None,
            inputs_by_shape: false,
            root_inputs: BTreeMap::new(),
            extra_outputs: Vec::new(),
            runtime: BTreeMap::new(),
        }
    }

    /// Add the graph output `name` holding `value`, a value (a matrix or a family of matrices)
    /// a composing caller computed from the lowered Program outputs it was given (see
    /// [`crate::api::online_eval`]): the stage returns it beside the Program outputs, which stay
    /// as preprocessing resolved them.
    pub fn add_output<V: mxx_dsl::GraphValue + 'static>(
        &mut self,
        name: impl Into<String>,
        value: V,
    ) {
        let name = name.into();
        self.extra_outputs.push(Box::new(move |graph: DslContext| graph.output(name, value)));
    }

    /// Bind the value of the graph input `name` that a composing caller's computation reads.
    pub fn bind_runtime_input(&mut self, name: impl Into<String>, value: RuntimeValue) {
        self.runtime.insert(name.into(), value);
    }

    /// This builder reading the inputs of `root_inputs` under the root keys of their tags, as
    /// their preprocessing did (see `RnsPreprocessBuild::with_root_inputs`).
    pub fn with_root_inputs(mut self, root_inputs: BTreeMap<WireId, String>) -> Self {
        self.root_inputs = root_inputs;
        self
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
    /// but cannot be executed. Input and data cardinalities need not fit a host index: graph
    /// validation uses integer expressions rather than allocating their members.
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
        let encryption_prefix = encryption.descriptor().artifact_name(&prefix);
        let prefix = preprocess.descriptor().artifact_name(&prefix);
        let name = |role: &str| format!("{prefix}/{role}");
        let ciphertext_name = |role: &str| format!("{encryption_prefix}/{role}");
        let preprocess_binding = preprocess.descriptor().binding();
        self.artifact_prefix = preprocess.descriptor().artifact_prefix().map(str::to_owned);
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
            instance_name(
                self.artifact_prefix.as_deref(),
                &crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            ),
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
                |role| ciphertext_name(&role),
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
                        ciphertext_name(&format!("input-{wire:?}.{port}")),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    )
                };
                let family_port = |port: usize, count: &IntExpr, columns: usize| {
                    ring.family_artifact_input(
                        encryption_production.clone(),
                        ciphertext_name(&format!("input-{wire:?}.{port}")),
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
            None,
            |role: String| {
                if dynamic.is_some() { ciphertext_name(&role) } else { name(&role) }
            },
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let mut evaluator = EncodingLowering {
            encryption_artifact_prefix: Some(encryption_prefix),
            root_lut_helpers: None,
            decoder_hint: None,
            material_member: None,
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
            root_hash_key: self.parameters.root_hash_key_input(),
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
            evaluator.preprocess_artifact_name(decoder_artifact_name(&key, "unmask-preimage")),
            (columns, 1),
            BigInt::from(bound),
            ArtifactAvailability::Transferred,
        );
        let hint = if let Some(hint) = &evaluator.decoder_hint {
            hint.clone()
        } else if evaluator.root_lut_helpers.is_some() {
            return Err(RnsBggError::Binding("graph row decoding requires a graph hint"));
        } else {
            ring.artifact_input(
                evaluator.encryption.clone(),
                evaluator.encryption_artifact_name(decoder_artifact_name(&key, "unmask-hint")),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        };
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
                    let root = match self.root_inputs.get(wire) {
                        Some(tag) => Some((
                            tag.clone(),
                            self.parameters.root_hash_key_input().ok_or_else(|| {
                                ApiError::input("a root input needs a rooted chain")
                            })?,
                        )),
                        None => None,
                    };
                    (
                        InputKeySource::Static { level, knowledge, root },
                        format!("input-{wire:?}"),
                        level,
                    )
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
    /// `sigma` is the row `g` of the caller's encoding of `sigma_0 G` of the part's chain (shared
    /// by its two parts) whose key the part's preprocessing read (see [Caller
    /// inputs](crate::refresh#caller-inputs)): component 1 of that Program's first output, `1 x
    /// m` in the decoding ring (port [`crate::rns_bgg::SIGMA_ROW_OUTPUT`] of its online
    /// evaluation). The graph reads it as a plain input; without a value the stage can be
    /// recorded but not executed. `A'` is the graph input
    /// [`crate::rns_bgg::REFRESH_MATRIX_INPUT`] (see [`Self::with_refresh_matrix`]).
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
                evaluator.encryption_artifact_name(refresh_artifact_name(spec, "one-encoding.0")),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
            ring.artifact_input(
                encryption.clone(),
                evaluator.encryption_artifact_name(refresh_artifact_name(spec, "one-encoding.1")),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
        );
        let hint_columns = crate::decoder::unmask::UNMASK_ROWS * (columns + 2);
        let hint = ring.artifact_input(
            encryption,
            evaluator.encryption_artifact_name(refresh_artifact_name(spec, "unmask-hint")),
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
        let actions = self.parameters.refresh_actions(
            level,
            a_prime,
            declaration.output_prime(),
            declaration.reference(),
        )?;
        let (fresh_row, blocks) = fresh_row;
        let fresh_term = crate::rns_bgg::refresh::row_fresh_term(fresh_row, blocks, &actions)?;
        let decoded = refresh_combination(one, target_row, fresh_term, &actions)?;
        let (smudge_outputs, child_manifests) = lower(&mut smudge)?;
        manifests.extend(child_manifests);
        let smudge_output = smudge_outputs
            .into_iter()
            .next()
            .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?;
        let encoding = decoded + checked_modulus_convert(smudge_output.row.1, &ring)?;
        let preimage = ring.preimage_artifact_input(
            preprocess,
            evaluator.preprocess_artifact_name(decoder_artifact_name(
                &decoder.artifact_key(),
                "unmask-preimage",
            )),
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
                (
                    instance_name(
                        child.artifact_prefix.as_deref(),
                        &hash_key_input_name(child.hash_namespace.as_ref()),
                    ),
                    child.hash_key.clone(),
                )
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
        if let Some(evaluator) = self.evaluator.as_mut() {
            if evaluator.root_lut_helpers.is_some() {
                evaluator.import_body_lut_material().map_err(ApiError::scheme)?;
            }
        }
        Ok((
            self.evaluator
                .as_mut()
                .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
}

/// Retarget a caller's root row and derive its key using the preprocessing key schedule.
#[allow(clippy::too_many_arguments)]
pub(super) fn graph_input_encoding(
    parameters: &RnsBggParameters,
    wire: WireId,
    level: usize,
    knowledge: AttributeKnowledge,
    key: &InputKeySource,
    hash_key: Bytes,
    row: RnsBggRow,
    payload: Option<Mat>,
    index: Option<Int>,
) -> Result<RnsBggEncoding, RnsBggError> {
    let form = match knowledge {
        AttributeKnowledge::PublicPayload => crate::rns_bgg::RnsBggForm::Public,
        AttributeKnowledge::PrivatePayload => crate::rns_bgg::RnsBggForm::State,
    };
    let (ring, columns) = match parameters.root() {
        Some(root) => (root.ring(), root.columns()),
        None => (parameters.ring(level)?, parameters.key_component_columns(level, knowledge)?),
    };
    if row.0.matrix_type() != &ring.matrix_type((1, columns)) ||
        row.1.matrix_type() != &ring.matrix_type((1, columns))
    {
        return Err(RnsBggError::Binding("graph input row has the wrong ring or width"));
    }
    let row =
        if parameters.root().is_some() { parameters.retarget_row(level, form, row)? } else { row };
    let payload = payload
        .map(|value| {
            if value.matrix_type().rows != IntExpr::constant(1) ||
                value.matrix_type().columns != IntExpr::constant(1)
            {
                return Err(RnsBggError::Binding("public graph payload must be a polynomial"));
            }
            checked_modulus_convert(value, &parameters.ring(level)?)
        })
        .transpose()?;
    Ok(RnsBggEncoding::new(key.key(parameters, hash_key, wire, index)?, row, payload))
}

impl ProgramBuild for RnsOnlineEvalBuild<'_> {
    type Wire = RnsBggEncoding;
    type WireFamily = OnlineEvalWireFamily;

    fn lower(&mut self, program: &ExponentLutProgram) -> Result<Vec<Self::Wire>, ApiError> {
        let (evaluator, families, environment) = self.lowering_state_inner()?;
        if evaluator.program.id() != program.id() {
            return Err(ApiError::input("lowered Program differs from the bound evaluation"));
        }
        crate::program::lower_program(program, evaluator, families, environment)
            .map_err(ApiError::lowering)
    }

    fn lower_deferred(
        &mut self,
        program: &ExponentLutProgram,
        deferred: &std::collections::BTreeSet<usize>,
    ) -> Result<Vec<crate::program::LoweredOutput<Self::Wire, Self::WireFamily>>, ApiError> {
        let (evaluator, families, environment) = self.lowering_state_inner()?;
        if evaluator.program.id() != program.id() {
            return Err(ApiError::input("lowered Program differs from the bound evaluation"));
        }
        crate::program::lower_program_deferred(program, evaluator, families, environment, deferred)
            .map_err(ApiError::lowering)
    }

    fn deferred_block(
        &mut self,
        program: &ExponentLutProgram,
        columns: &crate::program::DeferredColumns<Self::Wire, Self::WireFamily>,
        block: &IntExpr,
    ) -> Result<Self::Wire, ApiError> {
        let (evaluator, families, environment) = self.lowering_state_inner()?;
        if evaluator.program.id() != program.id() {
            return Err(ApiError::input("lowered Program differs from the bound evaluation"));
        }
        columns.block(program, evaluator, families, environment, block).map_err(ApiError::lowering)
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
        self.bind_sources(program, parameters, preprocess, Some(encryption), hash_key)
    }

    fn emit(
        mut self,
        mut graph: DslContext,
        outputs: Vec<RnsBggEncoding>,
    ) -> Result<GraphEmission<()>, ApiError> {
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
        let name = |role: &str| instance_name(self.artifact_prefix.as_deref(), role);
        let mut hash_keys = vec![(
            name(&crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref())),
            self.hash_key.clone(),
        )];
        if let Some(root) = self.parameters.root() {
            hash_keys.push((root.hash_key_input_name(), root.hash_key().to_vec()));
        }
        if let Some((row, child_keys)) = refreshed {
            graph = graph.output(name("refresh"), row).map_err(ApiError::scheme)?;
            hash_keys.extend(child_keys);
        }
        for output in std::mem::take(&mut self.extra_outputs) {
            graph = output(graph).map_err(ApiError::scheme)?;
        }
        for (index, output) in outputs.into_iter().enumerate() {
            // A decoded output is its plaintext modulo `q_(L+1)`, not an
            // encoding.
            graph = match &decoded {
                Some((decoded, plaintext)) if *decoded == index => {
                    graph.output(name(&format!("output-{index}")), plaintext.clone())
                }
                _ => graph.output(name(&format!("output-{index}")), output),
            }
            .map_err(ApiError::scheme)?;
        }
        for (name, key) in hash_keys {
            self.runtime.insert(name, RuntimeValue::Bytes(key.as_slice().into()));
        }
        Ok(GraphEmission {
            graph,
            runtime_inputs: self.runtime,
            manifests: evaluator.manifests.clone(),
            descriptor: (),
        })
    }
}

impl RnsOnlineEvalBuild<'_> {
    /// Bind preprocessing and optional ciphertext provenance before selecting row sources.
    fn bind_sources(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: Option<&StageArtifacts<RnsBggEncryptionDescriptor>>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if self.evaluator.is_some() {
            return Err(ApiError::input("online evaluation needs a fresh builder"));
        }
        for declaration in program.inputs().values() {
            if let InputBinding::Family(count) = declaration.shape() {
                let count = count.evaluate(parameters).map_err(|_| {
                    ApiError::input("online evaluation input family count is not concrete")
                })?;
                if count <= BigInt::from(0) {
                    return Err(ApiError::input("online evaluation input family count is empty"));
                }
            }
        }
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.map(|stage| stage.descriptor().binding());
        let prefix = preprocess.descriptor().artifact_prefix().map(str::to_owned);
        if self.artifact_prefix.is_some() && self.artifact_prefix != prefix {
            return Err(ApiError::compatibility(
                "online artifact prefix differs from preprocessing",
            ));
        }
        self.artifact_prefix = prefix;
        let expected_parameters = self.parameters.parameter_identity();
        let preprocess_production = preprocess.production().clone();
        let encryption_production = encryption
            .map_or_else(|| preprocess_production.clone(), |stage| stage.production().clone());
        let expected_hash_key = hash_key_commitment(hash_key);
        for binding in std::iter::once(preprocess_binding).chain(encryption_binding) {
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
        let graph_data = self.graph_data.is_some();
        if !self.inputs_by_shape && !graph_data {
            self.data.validate(program).map_err(|error| ApiError::input(error.to_string()))?;
        }
        let expected_data =
            (!program.data_families().is_empty() && !graph_data).then(|| self.data.commitment());
        if preprocess_binding.data() != expected_data.as_ref() {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess binding has other Program data than the online evaluation",
            ));
        }
        // The data are graph values, or graph inputs whose values this builder was given.
        let data = match &self.graph_data {
            Some(data) => data.clone(),
            None => crate::program::data_inputs(program, |family| {
                instance_name(
                    self.artifact_prefix.as_deref(),
                    &crate::rns_bgg::data_input_name(self.hash_namespace.as_ref(), family),
                )
            }),
        };
        for family in
            program.data_families().keys().filter(|_| !self.inputs_by_shape && !graph_data)
        {
            let values = self.data.values(*family).expect("validated program data");
            self.runtime.insert(
                instance_name(
                    self.artifact_prefix.as_deref(),
                    &crate::rns_bgg::data_input_name(self.hash_namespace.as_ref(), *family),
                ),
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
                if let Some(encryption_binding) = encryption_binding {
                    if encryption_binding.program() != Some(program.id()) ||
                        encryption_binding.sources().len() != 1 ||
                        !encryption_binding.sources().contains(&preprocess_production)
                    {
                        return Err(ApiError::compatibility(
                            "RNS-BGG encryption source is not the preprocess production",
                        ));
                    }
                }
                None
            }
            RhsMode::Dynamic => Some(self.validate_dynamic_stages(
                program,
                preprocess,
                encryption.ok_or_else(|| {
                    ApiError::input("Dynamic evaluation requires ciphertext packages")
                })?,
            )?),
        };
        if encryption_binding.is_some_and(|binding| {
            preprocess_binding.decoder_commitment() != binding.decoder_commitment()
        }) {
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
        if let Some(encryption) = encryption {
            manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        }
        let prelude = encryption_binding.and_then(|binding| binding.ciphertext_prelude()).map(
            |(production, manifest)| {
                manifests.insert(production.clone(), manifest.clone());
                production.clone()
            },
        );
        let workload = Arc::new(
            RnsBggDerivedWorkload::from_program_with_parameters(self.parameters, program)
                .map_err(ApiError::scheme)?,
        );
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        let hash_key = self.parameters.ring(0).map_err(ApiError::scheme)?.bytes_input(
            instance_name(
                self.artifact_prefix.as_deref(),
                &crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            ),
            self.hash_key.len(),
        );
        if encryption.is_some() {
            self.bind_encrypted_inputs(
                program,
                &workload,
                &encryption_production,
                &hash_key,
                dynamic.as_ref(),
                None,
                |role| encryption.expect("encryption branch").descriptor().artifact_name(&role),
            )?;
        }
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            program,
            &workload,
            &preprocess_production,
            prelude.as_ref(),
            dynamic.as_ref().map(|(setup, binding)| (setup.as_ref(), binding)),
            &hash_key,
            self.material_member.as_ref(),
            |role| match (dynamic.as_ref(), encryption) {
                (Some(_), Some(stage)) => stage.descriptor().artifact_name(&role),
                _ => instance_name(self.artifact_prefix.as_deref(), &role),
            },
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let mut evaluator = EncodingLowering {
            encryption_artifact_prefix: encryption
                .and_then(|stage| stage.descriptor().artifact_prefix())
                .map(str::to_owned),
            root_lut_helpers: encryption.is_none().then(BTreeMap::new),
            decoder_hint: None,
            material_member: self.material_member.clone(),
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            decoder_smudge: None,
            artifact_prefix: self.artifact_prefix.clone(),
            current_site: None,
            hash_key,
            root_hash_key: self.parameters.root_hash_key_input(),
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
        } else if encryption.is_some() {
            evaluator.import_body_lut_material().map_err(ApiError::scheme)?;
        }
        self.evaluator = Some(evaluator);
        Ok(())
    }
}
