//! The online evaluation stage adapter.
use super::*;

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
                &crate::rns_bgg::hash_key_input_name(self.hash_namespace.as_ref()),
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
            output_use: RnsOutputUse::default(),
            output_carriers: Vec::new(),
            dynamic: None,
            data: crate::program::ProgramData::new(),
            graph_data: None,
            material_member: None,
            hash_namespace: None,
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

    /// The rows, key matrices and public payloads of a family exported by
    /// `lower_deferred_exporting` that was lowered as one graph family (a materialized Program
    /// family), as families of their own: projections of the lowered family, so the caller can
    /// publish them without reading their members. `None` for an exported input family; the
    /// payloads are `None` for a private family.
    #[allow(clippy::type_complexity)]
    pub fn exported_encodings(
        family: &OnlineEvalWireFamily,
    ) -> Result<Option<(Family<RnsBggRow>, Family<RnsBggMatrix>, Option<Family<Mat>>)>, ApiError>
    {
        let OnlineEvalWireFamily::Stored(family) = family else { return Ok(None) };
        let error = |error: mxx_dsl::DslError| ApiError::input(error.to_string());
        let public = mxx_dsl::GraphValue::schema(family).element.public_payload.is_some();
        Ok(Some((
            family.field(|encoding| encoding.row).map_err(error)?,
            family.field(|encoding| encoding.public_key.matrix).map_err(error)?,
            public
                .then(|| family.field(|encoding| encoding.public_payload.expect("a public family")))
                .transpose()
                .map_err(error)?,
        )))
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

    /// This builder reading the inputs of `root_inputs` under the root keys of their tags, as
    /// their preprocessing did (see `RnsPreprocessBuild::with_root_inputs`).
    pub fn with_root_inputs(mut self, root_inputs: BTreeMap<WireId, String>) -> Self {
        self.root_inputs = root_inputs;
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
    /// ciphertext names no Program.
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
        if !encryption_binding.sources().is_empty() || encryption_binding.program().is_some() {
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

    /// Make this builder sample its public matrices from the stage hash key
    /// derived under `namespace`, as its preprocessing and encryption did.
    pub fn with_hash_namespace(mut self, namespace: [u8; 32]) -> Self {
        self.hash_namespace = Some(namespace);
        self
    }

    /// Bind every input of `program` from the encryption production
    /// `encryption`, which stores each input's row (and public payload); its
    /// public key is derived from the hash key here, or where a family
    /// member is read. A Static input is stored at its logical level under
    /// its wire; a Dynamic one at the setup ring under its setup coordinate.
    /// `name` maps a role to its artifact name.
    fn bind_encrypted_inputs(
        &mut self,
        program: &ExponentLutProgram,
        workload: &RnsBggDerivedWorkload,
        encryption: &ProductionId,
        hash_key: &Bytes,
        dynamic: Option<&(Arc<RnsBggDynamicSetup>, RnsBggDynamicBinding)>,
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
                        payloads: public.then(|| port(2, 1)),
                    })
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        Ok(())
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
        let (evaluator, families, environment) = self.lowering_state_inner()?;
        if evaluator.program.id() != program.id() {
            return Err(ApiError::input("lowered Program differs from the bound evaluation"));
        }
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
        let (evaluator, families, _environment) = self.lowering_state_inner()?;
        if evaluator.program.id() != program.id() {
            return Err(ApiError::input("lowered Program differs from the bound evaluation"));
        }
        crate::program::FamilyExecution::wire_at(families, evaluator, family, index)
            .map_err(ApiError::lowering)
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
            name(&crate::rns_bgg::hash_key_input_name(self.hash_namespace.as_ref())),
            self.hash_key.clone(),
        )];
        if let Some(root) = self.parameters.root() {
            hash_keys.push((root.hash_key_input_name(), root.hash_key().to_vec()));
        }
        for output in std::mem::take(&mut self.extra_outputs) {
            graph = output(graph).map_err(ApiError::scheme)?;
        }
        for (index, output) in outputs.into_iter().enumerate() {
            graph =
                graph.output(name(&format!("output-{index}")), output).map_err(ApiError::scheme)?;
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
        if preprocess_binding.output_use() != self.output_use {
            return Err(ApiError::compatibility(
                "RNS-BGG output use differs from the preprocessed Program",
            ));
        }
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
        // A namespaced Program samples from its derived key, as its preprocessing and
        // encryption did; every binding still names the stage key.
        self.hash_key = match &self.hash_namespace {
            Some(namespace) => crate::rns_bgg::child_hash_key(hash_key, namespace),
            None => hash_key.to_vec(),
        };
        let hash_key = self.parameters.ring(0).map_err(ApiError::scheme)?.bytes_input(
            instance_name(
                self.artifact_prefix.as_deref(),
                &crate::rns_bgg::hash_key_input_name(self.hash_namespace.as_ref()),
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
            material_member: self.material_member.clone(),
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
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
