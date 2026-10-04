//! The online evaluation stage adapter, refresh wiring, and refresh
//! checkpoint/resume.
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

impl<'a> RnsOnlineEvalBuild<'a> {
    /// Creates an unbound online evaluation builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            hash_key: Vec::new(),
            evaluator: None,
            families: OnlineEvalFamilies { parameters },
            environment: Environment::new(),
            pending_refresh: None,
            refreshed_input: None,
            refresh_prf: None,
            refreshed_public_key: None,
        }
    }

    /// Bind a slot-scoped online evaluation child to the prepared PRF Program.
    ///
    /// Ordinary artifact binding retains strict source-Program equality. This
    /// explicit child path validates the combined source checkpoints and then
    /// imports only the slot-scoped PRF input ports.
    pub fn bind_refresh_artifacts(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let prepared = declaration.prepared().program();
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        let expected_hash_key = hash_key_commitment(hash_key);
        if slot >= declaration.slots().len() || declaration.output_schema().is_empty() {
            return Err(ApiError::input("invalid refresh PRF slot or schema"));
        }
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.program() != declaration.source_program() ||
                binding.hash_key_commitment() != &expected_hash_key ||
                binding.refresh_commitment() != Some(declaration.commitment())
            {
                return Err(ApiError::compatibility(
                    "refresh PRF artifacts do not match source declaration",
                ));
            }
        }
        if !preprocess_binding.sources().is_empty() {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh preprocess root unexpectedly declares sources",
            ));
        }
        if encryption_binding.sources().len() != 1 ||
            !encryption_binding.sources().contains(preprocess.production())
        {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh encryption source is not preprocess production",
            ));
        }
        let preprocess_production = preprocess.production().clone();
        let encryption_production = encryption.production().clone();
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(prepared).map_err(ApiError::scheme)?);
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        // The prepared PRF encryption stage publishes each input at its
        // logical level (`append_refresh_prf_child`), with its public key; the
        // refresh route applies the per-slot p-level action to the outputs.
        let commitment = *declaration.commitment();
        self.environment = Environment::new();
        for (wire, input) in prepared.inputs() {
            let public = matches!(input.knowledge(), AttributeKnowledge::PublicPayload);
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            let columns = self
                .parameters
                .key_component_columns(level, input.knowledge())
                .map_err(ApiError::scheme)?;
            let scalar = || {
                RnsBggEncoding::new(
                    RnsBggPublicKey::new(
                        *wire,
                        (
                            (
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.0"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.1"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                            ),
                            (
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.2"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.3"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                            ),
                        ),
                    ),
                    (
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.4"),
                            ),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        ),
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.5"),
                            ),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        ),
                    ),
                    public.then(|| {
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.6"),
                            ),
                            (1, 1),
                            ArtifactAvailability::Transferred,
                        )
                    }),
                )
            };
            let binding = match input.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(scalar()),
                InputBinding::Family(count) => {
                    let key_parts = (0..4)
                        .map(|port| {
                            ring.family_artifact_input(
                                encryption_production.clone(),
                                refresh_prf_artifact_name(
                                    commitment,
                                    slot,
                                    &format!("input-{wire:?}.{port}"),
                                ),
                                count.clone(),
                                (1, columns),
                                ArtifactAvailability::Transferred,
                            )
                        })
                        .collect::<Vec<_>>();
                    let rows = (0..2)
                        .map(|port| {
                            ring.family_artifact_input(
                                encryption_production.clone(),
                                refresh_prf_artifact_name(
                                    commitment,
                                    slot,
                                    &format!("input-{wire:?}.{}", port + 4),
                                ),
                                count.clone(),
                                (1, columns),
                                ArtifactAvailability::Transferred,
                            )
                        })
                        .collect::<Vec<_>>();
                    let payload = public.then(|| {
                        ring.family_artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.6"),
                            ),
                            count.clone(),
                            (1, 1),
                            ArtifactAvailability::Transferred,
                        )
                    });
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
        let commitment = *declaration.commitment();
        let hash_key = self
            .parameters
            .ring(0)
            .map_err(ApiError::scheme)?
            .bytes_input(super::HASH_KEY_INPUT, 32);
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            prepared,
            &workload,
            &preprocess_production,
            &encryption_production,
            &hash_key,
            |role: String| refresh_prf_artifact_name(commitment, slot, &role),
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(prepared.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            refresh_prf_namespace: Some((*declaration.commitment(), slot)),
            current_site: None,
            hash_key,
            context: self.context,
            rhs_companions: BTreeMap::new(),
        });
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_rhs_companions()
            .map_err(ApiError::scheme)?;
        Ok(())
    }

    /// Bind one child to the structural PRF program frozen in `declaration`.
    ///
    /// The PRF is a separate Program from the outer computation.  Checking
    /// that identity here prevents a child lowerer for an unrelated source
    /// graph from being accepted merely because its carrier has the right
    /// row shape.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<(), RnsBggError> {
        self.parameters.validate_refresh_prf_schedule(declaration, slot)?;
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("PRF artifacts are not bound"))?;
        declaration.validate_prf_program(evaluator.program.id())?;
        declaration.validate_prf_outputs(&evaluator.program)?;
        if declaration.prepared().setup_identity().program != evaluator.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        self.refresh_prf =
            Some((slot, evaluator.program.id(), self.parameters.parameter_identity()));
        Ok(())
    }

    /// Bind a checkpointed refresh as the sole input of an identity
    /// continuation. Only lazy artifact handles are created. The checkpoint's
    /// row and matching target public-key ports are imported from its
    /// production; caller code cannot substitute an unrelated carrier.
    fn bind_refreshed_input(
        &mut self,
        program: &ExponentLutProgram,
        declaration: &RefreshDeclaration,
        artifacts: &StageArtifacts<RnsBggRefreshCheckpointDescriptor>,
    ) -> Result<(), ApiError> {
        let (&input_wire, declaration_input) = program
            .inputs()
            .iter()
            .next()
            .ok_or_else(|| ApiError::input("resumption Program has no input"))?;
        let descriptor = artifacts.descriptor().binding();
        if descriptor.context() != &self.context ||
            descriptor.program() != declaration.source_program() ||
            descriptor.refresh_commitment() != Some(declaration.commitment())
        {
            return Err(ApiError::compatibility(
                "refreshed input checkpoint does not match its declaration",
            ));
        }
        if declaration.output_schema().is_empty() {
            return Err(ApiError::input("refreshed input schema is empty"));
        }
        // A resumed Program is evaluated against the target chain, whose
        // level zero is the refresh output level.  The old source Program's
        // parameter digest intentionally does not identify this builder.
        let resumed_level = self.parameters.level(0).map_err(ApiError::scheme)?;
        let resumed_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        if program.inputs().len() != 1 ||
            !program.inputs().contains_key(&input_wire) ||
            program.outputs() != [Operand::Scalar(input_wire)]
        {
            return Err(ApiError::compatibility(
                "only an identity continuation can resume from a refresh checkpoint without stage artifacts",
            ));
        }
        if !matches!(declaration_input.shape(), InputBinding::Scalar(())) {
            return Err(ApiError::input("refreshed input must be scalar"));
        }
        let knowledge = program.wire_knowledge(input_wire).map_err(ApiError::scheme)?;
        let key_columns =
            self.parameters.key_component_columns(0, knowledge).map_err(ApiError::scheme)?;
        let row_columns = key_columns;
        let expected_port = resumed_ring.matrix_type((1, key_columns));
        let schema_ports = declaration
            .output_schema()
            .iter()
            .filter_map(|wire| match wire {
                WireType::Matrix(matrix) => Some(matrix),
                _ => None,
            })
            .collect::<Vec<_>>();
        let target_modulus =
            declaration.slots().iter().try_fold(BigUint::from(1u8), |product, slot| {
                slot.q_t
                    .to_biguint()
                    .map(|modulus| product * modulus)
                    .ok_or(ApiError::compatibility("refresh target modulus is invalid"))
            })?;
        if schema_ports.len() != 4 ||
            schema_ports.iter().any(|port| **port != expected_port) ||
            target_modulus != *resumed_level.modulus().as_ref()
        {
            return Err(ApiError::compatibility(
                "resumption builder must use the target modulus chain at level zero",
            ));
        }
        let target_ring = resumed_ring;
        let production = artifacts.production().clone();
        let manifest = artifacts.manifest();
        let checkpoint_name =
            |role: &str| refresh_checkpoint_artifact_name(*declaration.commitment(), role);
        let resumed_concrete_ring = resolve_ring(target_ring.as_ref()).map_err(ApiError::scheme)?;
        let matrix_type = |columns| {
            ArtifactType::Matrix(ConcreteMatrixType {
                ring: resumed_concrete_ring.clone(),
                rows: 1,
                columns,
            })
        };
        let require_artifact = |name: &str, artifact_type: ArtifactType| {
            let artifact = manifest.artifacts.get(name).ok_or_else(|| {
                ApiError::compatibility("refresh checkpoint schema is missing a required port")
            })?;
            if artifact.artifact_type != artifact_type || artifact.family_count.is_some() {
                return Err(ApiError::compatibility(
                    "refresh checkpoint artifact schema does not match the declaration",
                ));
            }
            Ok(())
        };
        require_artifact(&checkpoint_name("row.0"), matrix_type(row_columns))?;
        require_artifact(&checkpoint_name("row.1"), matrix_type(row_columns))?;
        for port in 0..4 {
            require_artifact(
                &checkpoint_name(&format!("target.{port}")),
                matrix_type(key_columns),
            )?;
        }
        let row = (
            target_ring.artifact_input(
                production.clone(),
                checkpoint_name("row.0"),
                (1, row_columns),
                ArtifactAvailability::Transferred,
            ),
            target_ring.artifact_input(
                production.clone(),
                checkpoint_name("row.1"),
                (1, row_columns),
                ArtifactAvailability::Transferred,
            ),
        );
        let public_key = RnsBggPublicKey::new(
            input_wire,
            (
                (
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.0"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.1"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                ),
                (
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.2"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.3"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                ),
            ),
        );
        if public_key.schema().wire_types() != declaration.output_schema() {
            return Err(ApiError::input("refresh public-key schema does not match declaration"));
        }
        let public_payload = if matches!(knowledge, AttributeKnowledge::PublicPayload) {
            require_artifact(&checkpoint_name("row.public"), matrix_type(1))?;
            Some(target_ring.artifact_input(
                production.clone(),
                checkpoint_name("row.public"),
                (1, 1),
                ArtifactAvailability::Transferred,
            ))
        } else {
            None
        };
        // Initialize the lowerer here rather than requiring an ordinary
        // bind_artifacts call.  The checkpoint's production is the only
        // runtime input needed by the identity continuation; source
        // dependency productions remain available for a later gated Program.
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?);
        let mut environment = Environment::new();
        environment
            .bind_wire(
                input_wire,
                InputBinding::Scalar(RnsBggEncoding::new(
                    public_key.clone(),
                    row.clone(),
                    public_payload.clone(),
                )),
            )
            .map_err(|error| ApiError::input(error.to_string()))?;
        let mut sources = descriptor.sources().to_vec();
        sources.sort();
        sources.dedup();
        let preprocess = sources.first().cloned().unwrap_or_else(|| production.clone());
        let hash_key = self
            .parameters
            .ring(0)
            .map_err(ApiError::scheme)?
            .bytes_input(super::HASH_KEY_INPUT, 32);
        let encryption = sources.get(1).cloned().unwrap_or_else(|| preprocess.clone());
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            program,
            &workload,
            &preprocess,
            &encryption,
            &hash_key,
            |role| role,
        )? {
            environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let manifests = BTreeMap::from([(production.clone(), artifacts.manifest().clone())]);
        if !manifests.contains_key(&production) {
            return Err(ApiError::compatibility(
                "refresh checkpoint production is not in its manifest map",
            ));
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess,
            encryption,
            manifests,
            refresh_prf_namespace: None,
            current_site: None,
            hash_key,
            context: self.context,
            rhs_companions: BTreeMap::new(),
        });
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_rhs_companions()
            .map_err(ApiError::scheme)?;
        self.environment = environment;
        self.refreshed_public_key = Some(public_key.clone());
        self.refreshed_input =
            Some((declaration.clone(), RnsBggEncoding::new(public_key, row, public_payload)));
        Ok(())
    }

    /// Returns the lazily imported target public key for a resumed refresh.
    pub fn refreshed_public_key(&self) -> Option<&RnsBggPublicKey> {
        self.refreshed_public_key.as_ref()
    }

    /// Returns the lazily imported refreshed encoding for resumption.
    pub fn refreshed_encoding(&self) -> Option<&RnsBggEncoding> {
        self.refreshed_input.as_ref().map(|(_, encoding)| encoding)
    }

    /// Bind one declared refresh and replace the selected output at graph
    /// finalization.  The prepared PRF children are lowered internally when
    /// the graph is finalized; callers provide no carrier map or evaluator
    /// closure.
    pub fn apply_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        prf_children: BTreeMap<usize, RnsOnlineEvalBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() {
            return Err(RnsBggError::Binding("refresh was already bound"));
        }
        declaration.validate_prf_slots(&prf_children)?;
        for (slot, child) in &prf_children {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if child.refresh_prf.map(|(bound_slot, program, identity)| {
                bound_slot == *slot &&
                    program == declaration.prepared().program().id() &&
                    identity == expected_child_parameters
            }) != Some(true) ||
                child.parameters.parameter_identity() != expected_child_parameters
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
        }
        self.pending_refresh = Some((output_index, declaration.clone(), prf_children));
        Ok(())
    }

    fn lower_refresh_children(
        declaration: &RefreshDeclaration,
        children: &mut BTreeMap<usize, RnsOnlineEvalBuild<'a>>,
    ) -> Result<BTreeMap<PrfLabel, BTreeMap<usize, RnsBggEncoding>>, RnsBggError> {
        declaration.collect_prf_outputs(|slot| {
            let child =
                children.get_mut(&slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let (evaluator, families, environment) = child
                .lowering_state_inner()
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            let values = crate::program::lower_program(
                declaration.prepared().program(),
                evaluator,
                families,
                environment,
            )
            .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            Ok(values.into_iter().map(|value| (value.wire(), value)).collect())
        })
    }

    /// Finalize the pending refresh against lowered outer outputs and return
    /// the resulting symbolic carrier before graph finalization.
    fn finalize_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        mut children: BTreeMap<usize, RnsOnlineEvalBuild<'a>>,
        outputs: &mut [RnsBggEncoding],
    ) -> Result<RnsBggEncoding, RnsBggError> {
        declaration.validate_prf_slots(&children)?;
        for (slot, child) in &children {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if child.refresh_prf.map(|(bound_slot, program, identity)| {
                bound_slot == *slot &&
                    program == declaration.prepared().program().id() &&
                    identity == expected_child_parameters
            }) != Some(true) ||
                child.parameters.parameter_identity() != expected_child_parameters
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
        }
        let source = outputs
            .get(output_index)
            .ok_or(RnsBggError::Refresh(RefreshError::TargetMismatch))?
            .clone();
        let inputs = self.refresh_inputs_from_output(declaration, &source)?;
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        let mut refresh = RnsBggRefresh::new(evaluator, inputs);
        let refreshed = apply_refresh(declaration, &mut refresh, |callback_declaration, _| {
            Self::lower_refresh_children(callback_declaration, &mut children)
        })
        .map_err(|error| match error {
            RefreshApplicationError::Refresh(error) => error,
            RefreshApplicationError::OutputMismatch => {
                RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
            }
        })?;
        let refreshed_target = evaluator.refresh_public_target(declaration, source.wire())?;
        for child in children.values() {
            if let Some(child_evaluator) = child.evaluator.as_ref() {
                evaluator.manifests.extend(child_evaluator.manifests.clone());
            }
        }
        outputs[output_index] = refreshed.clone();
        self.refreshed_public_key = Some(refreshed_target);
        Ok(refreshed)
    }

    fn refresh_inputs_from_output(
        &self,
        declaration: &RefreshDeclaration,
        source: &RnsBggEncoding,
    ) -> Result<BTreeMap<usize, RnsBggEncoding>, RnsBggError> {
        if source.wire() != declaration.source_output() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        let level = evaluator.workload.level(source.wire())?;
        let ring = self.parameters.ring(level)?;
        let expected_public =
            matches!(self.program_knowledge(source.wire())?, AttributeKnowledge::PublicPayload);
        let key_knowledge = if expected_public {
            AttributeKnowledge::PublicPayload
        } else {
            AttributeKnowledge::PrivatePayload
        };
        let key_columns = self.parameters.key_component_columns(level, key_knowledge)?;
        let row_columns = key_columns;
        let expected_row = ring.matrix_type((1, row_columns));
        let expected_key = ring.matrix_type((1, key_columns));
        if source.row.0.matrix_type() != &expected_row ||
            source.row.1.matrix_type() != &expected_row
        {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if source.public_key.wire != source.wire() ||
            source.public_key.matrix.0.0.matrix_type() != &expected_key ||
            source.public_key.matrix.0.1.matrix_type() != &expected_key ||
            source.public_key.matrix.1.0.matrix_type() != &expected_key ||
            source.public_key.matrix.1.1.matrix_type() != &expected_key
        {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if expected_public != source.public_payload.is_some() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if let Some(payload) = &source.public_payload {
            if payload.matrix_type() != &ring.matrix_type((1, 1)) {
                return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
            }
        }
        declaration
            .slots()
            .iter()
            .map(|slot| {
                // Each slot receives the same raw q-level carrier.  The
                // refresh route owns the single q→p ModDown; reducing here
                // would silently remove that protocol stage.
                Ok((
                    slot.slot,
                    RnsBggEncoding::new(
                        source.public_key.clone(),
                        (source.row.0.clone(), source.row.1.clone()),
                        source.public_payload.clone(),
                    ),
                ))
            })
            .collect()
    }

    fn program_knowledge(&self, wire: WireId) -> Result<AttributeKnowledge, RnsBggError> {
        self.evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?
            .program
            .wire_knowledge(wire)
            .map_err(|_| RnsBggError::Binding("wire metadata"))
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
        if let Some((declaration, input)) = self.refreshed_input.clone() {
            if declaration.output_schema().is_empty() {
                return Err(ApiError::input("refreshed input declaration schema is empty"));
            }
            let evaluator = self
                .evaluator
                .as_ref()
                .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
            let input_declaration = evaluator
                .program
                .inputs()
                .get(&input.wire())
                .ok_or_else(|| ApiError::input("refreshed wire is not a Program input"))?;
            if !matches!(input_declaration.shape(), InputBinding::Scalar(())) {
                return Err(ApiError::input("refreshed input must be scalar"));
            }
            self.environment.wires.insert(input.wire(), InputBinding::Scalar(input));
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

/// Checkpoint a refresh output that has already been returned by
/// [`crate::api::online_eval`].
///
/// This operation only imports the returned output ports into a small
/// artifact graph. It does not lower or execute the refresh again. The
/// checkpoint stores the row tuple, an optional public payload, and the
/// matching target public key as separate outputs so that a resumed carrier
/// cannot accidentally treat the whole paired encoding as its row.
pub fn checkpoint_refreshed_output<E, S>(
    program: &ExponentLutProgram,
    declaration: &RefreshDeclaration,
    result: &StageOutputs,
    output_index: usize,
    preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
    encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
    parameters: &RnsBggParameters,
    hash_key: &[u8],
    execution: &mut E,
    store: &mut S,
) -> Result<StageArtifacts<RnsBggRefreshCheckpointDescriptor>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let (graph, descriptor) = build_refresh_checkpoint_graph(
        program,
        declaration,
        result,
        output_index,
        preprocess,
        encryption,
        parameters,
        hash_key,
    )?;
    execute_artifacts(
        graph,
        &ParamEnv::default(),
        descriptor,
        domain_identity(*declaration.commitment(), b"refresh-checkpoint"),
        execution,
        store,
        |_| Ok(()),
    )
}

/// Resume a refresh through the built-in single-input identity continuation.
///
/// The continuation is deliberately outside the original estimated graph:
/// it is created and executed only after the caller has checkpointed the
/// already-returned refresh output. No refresh callback or intermediate
/// carrier is accepted by this API.
pub fn resume_refreshed_output<E, S>(
    declaration: &RefreshDeclaration,
    checkpoint: &StageArtifacts<RnsBggRefreshCheckpointDescriptor>,
    parameters: &RnsBggParameters,
    execution: &mut E,
    store: &mut S,
) -> Result<StageOutputs, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let binding = checkpoint.descriptor().binding();
    let manifest = checkpoint.manifest();
    let public_payload = manifest
        .artifacts
        .contains_key(&refresh_checkpoint_artifact_name(*declaration.commitment(), "row.public"));
    let knowledge = if public_payload {
        AttributeKnowledge::PublicPayload
    } else {
        AttributeKnowledge::PrivatePayload
    };
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(knowledge).map_err(ApiError::scheme)?;
    let program = builder.finish(vec![Operand::Scalar(input)]).map_err(ApiError::scheme)?;

    let mut online_eval = RnsOnlineEvalBuild::new(parameters, *binding.context());
    online_eval.bind_refreshed_input(&program, declaration, checkpoint)?;
    let outputs = {
        let (evaluator, families, environment) = online_eval.lowering_state_inner()?;
        crate::program::lower_program(&program, evaluator, families, environment)
            .map_err(ApiError::lowering)?
    };
    let graph = online_eval.build_graph(outputs)?;
    execute_graph(graph, &ParamEnv::default(), execution, store, |_| Ok(()))
}

fn build_refresh_checkpoint_graph(
    program: &ExponentLutProgram,
    declaration: &RefreshDeclaration,
    result: &StageOutputs,
    output_index: usize,
    preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
    encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
    parameters: &RnsBggParameters,
    hash_key: &[u8],
) -> Result<(StageGraph, RnsBggRefreshCheckpointDescriptor), ApiError> {
    let output = program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("refresh checkpoint output index is out of range"))?;
    let source_wire = match output {
        Operand::Scalar(wire) => *wire,
        Operand::Member { .. } => {
            return Err(ApiError::input("refresh checkpoint output must be scalar"));
        }
    };
    if source_wire != declaration.source_output() {
        return Err(ApiError::input("refresh checkpoint output does not match declaration"));
    }
    if declaration.output_schema().len() != 4 {
        return Err(ApiError::input("refresh checkpoint target schema must contain four ports"));
    }
    let source_knowledge = program.wire_knowledge(source_wire).map_err(ApiError::scheme)?;
    let expected_ports =
        if matches!(source_knowledge, AttributeKnowledge::PrivatePayload) { 6 } else { 7 };
    let prefix = format!("output-{output_index}.");
    let mut ports = BTreeMap::new();
    for name in result.outputs.keys().filter(|name| name.starts_with(&prefix)) {
        let port = name
            .strip_prefix(&prefix)
            .and_then(|port| port.parse::<usize>().ok())
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is malformed"))?;
        if port >= expected_ports || ports.insert(port, name.clone()).is_some() {
            return Err(ApiError::input("refresh checkpoint output port is out of range"));
        }
    }
    if ports.len() != expected_ports || ports.keys().copied().ne(0..expected_ports) {
        return Err(ApiError::input("refresh checkpoint output port set is incomplete"));
    }

    let target_level = declaration.target_level();
    let ring = parameters.ring(target_level).map_err(ApiError::scheme)?;
    let columns = parameters
        .key_component_columns(target_level, source_knowledge)
        .map_err(ApiError::scheme)?;
    let expected_port = ring.matrix_type((1, columns));
    let mut runtime_inputs = BTreeMap::new();
    let mut input = |port: usize, name: String| -> Result<Mat, ApiError> {
        let output_name = ports
            .get(&port)
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is missing"))?;
        let value = result
            .outputs
            .get(output_name)
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is missing"))?
            .clone();
        if !matches!(value, RuntimeValue::Matrix(_)) {
            return Err(ApiError::input("refresh checkpoint output is not a matrix"));
        }
        runtime_inputs.insert(name.clone(), value);
        Ok(ring.input(name, (1, columns)))
    };
    let key = (
        (
            input(0, "refresh-checkpoint-key-0".to_owned())?,
            input(1, "refresh-checkpoint-key-1".to_owned())?,
        ),
        (
            input(2, "refresh-checkpoint-key-2".to_owned())?,
            input(3, "refresh-checkpoint-key-3".to_owned())?,
        ),
    );
    let row = (
        input(4, "refresh-checkpoint-row-0".to_owned())?,
        input(5, "refresh-checkpoint-row-1".to_owned())?,
    );
    let schema_ports = declaration
        .output_schema()
        .iter()
        .filter_map(|port| match port {
            WireType::Matrix(matrix) => Some(matrix),
            _ => None,
        })
        .collect::<Vec<_>>();
    if schema_ports.len() != 4 || schema_ports.iter().any(|port| **port != expected_port) {
        return Err(ApiError::input("refresh checkpoint target schema mismatch"));
    }
    if key.0.0.matrix_type() != &expected_port ||
        key.0.1.matrix_type() != &expected_port ||
        key.1.0.matrix_type() != &expected_port ||
        key.1.1.matrix_type() != &expected_port ||
        row.0.matrix_type() != &expected_port ||
        row.1.matrix_type() != &expected_port
    {
        return Err(ApiError::input("refresh checkpoint carrier schema mismatch"));
    }
    let public_payload = if matches!(source_knowledge, AttributeKnowledge::PublicPayload) {
        let output_name = ports
            .get(&6)
            .ok_or_else(|| ApiError::input("refresh checkpoint public payload is missing"))?;
        let value = result
            .outputs
            .get(output_name)
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is missing"))?
            .clone();
        if !matches!(value, RuntimeValue::Matrix(_)) {
            return Err(ApiError::input("refresh checkpoint public payload is not a matrix"));
        }
        let name = "refresh-checkpoint-row-public".to_owned();
        runtime_inputs.insert(name.clone(), value);
        Some(ring.input(name, (1, 1)))
    } else {
        None
    };

    let preprocess_binding = preprocess.descriptor().binding();
    let encryption_binding = encryption.descriptor().binding();
    let expected_parameters = parameters.parameter_identity();
    let expected_hash_key = hash_key_commitment(hash_key);
    for binding in [preprocess_binding, encryption_binding] {
        if binding.context() != encryption_binding.context() ||
            binding.parameters() != &expected_parameters ||
            binding.program() != declaration.source_program() ||
            binding.hash_key_commitment() != &expected_hash_key ||
            binding.refresh_commitment() != Some(declaration.commitment())
        {
            return Err(ApiError::compatibility("refresh checkpoint source binding mismatch"));
        }
    }
    if !preprocess_binding.sources().is_empty() ||
        encryption_binding.sources().len() != 1 ||
        !encryption_binding.sources().contains(preprocess.production())
    {
        return Err(ApiError::compatibility("refresh checkpoint source artifacts are invalid"));
    }
    let descriptor = RnsBggArtifactBinding::new(
        *encryption_binding.context(),
        declaration.source_program(),
        expected_parameters,
        expected_hash_key,
        Vec::new(),
        parameters.rhs_mode(),
    )
    .map_err(ApiError::scheme)?
    .with_refresh_declaration(declaration);
    let checkpoint_name =
        |role: &str| refresh_checkpoint_artifact_name(*declaration.commitment(), role);
    let mut graph = DslContext::new("rns-bgg-refresh-checkpoint")
        .transferred_output(checkpoint_name("row"), row)
        .map_err(ApiError::scheme)?;
    if let Some(payload) = public_payload {
        graph = graph
            .transferred_output(checkpoint_name("row.public"), payload)
            .map_err(ApiError::scheme)?;
    }
    graph = graph.transferred_output(checkpoint_name("target"), key).map_err(ApiError::scheme)?;
    let mut manifests = BTreeMap::new();
    manifests.insert(preprocess.production().clone(), preprocess.manifest().clone());
    manifests.insert(encryption.production().clone(), encryption.manifest().clone());
    for source in [preprocess.production(), encryption.production()] {
        if !manifests.contains_key(source) {
            return Err(ApiError::compatibility("refresh checkpoint source manifest is missing"));
        }
    }
    Ok((
        StageGraph::new(graph.build().map_err(ApiError::scheme)?, runtime_inputs, manifests),
        RnsBggRefreshCheckpointDescriptor::from_binding(descriptor),
    ))
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
                binding.program() != program.id() ||
                binding.hash_key_commitment() != &expected_hash_key
            {
                return Err(ApiError::compatibility(
                    "RNS-BGG artifact binding does not match the online evaluation Program",
                ));
            }
        }
        if !preprocess_binding.sources().is_empty() {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess root unexpectedly declares sources",
            ));
        }
        if encryption_binding.sources().len() != 1 ||
            !encryption_binding.sources().contains(&preprocess_production)
        {
            return Err(ApiError::compatibility(
                "RNS-BGG encryption source is not the preprocess production",
            ));
        }
        if preprocess_binding.refresh_commitment() != encryption_binding.refresh_commitment() {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh declaration commitment differs between stages",
            ));
        }
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?);
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        // Encryption stores each input's row (and public payload) at its
        // logical level; its public key is sampled from the hash key here, or
        // where a family member is read.
        let hash_key = self
            .parameters
            .ring(0)
            .map_err(ApiError::scheme)?
            .bytes_input(super::HASH_KEY_INPUT, 32);
        for (wire, declaration) in program.inputs() {
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            let knowledge = declaration.knowledge();
            let columns = self
                .parameters
                .key_component_columns(level, knowledge)
                .map_err(ApiError::scheme)?;
            let public = matches!(knowledge, AttributeKnowledge::PublicPayload);
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => {
                    let port = |port: usize, columns: usize| {
                        ring.artifact_input(
                            encryption_production.clone(),
                            format!("input-{wire:?}.{port}"),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    };
                    let key = super::public_key::sampled_key(
                        self.parameters,
                        level,
                        hash_key.clone(),
                        *wire,
                        knowledge,
                        None,
                    )
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
                            encryption_production.clone(),
                            format!("input-{wire:?}.{port}"),
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
                        level,
                        knowledge,
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
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            program,
            &workload,
            &preprocess_production,
            &encryption_production,
            &hash_key,
            |role| role,
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            refresh_prf_namespace: None,
            current_site: None,
            hash_key,
            context: self.context,
            rhs_companions: BTreeMap::new(),
        });
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_rhs_companions()
            .map_err(ApiError::scheme)?;
        Ok(())
    }

    fn build_graph(mut self, outputs: Vec<RnsBggEncoding>) -> Result<StageGraph, ApiError> {
        let mut outputs = outputs;
        let pending = self.pending_refresh.take();
        if let Some((index, declaration, mut children)) = pending {
            self.finalize_refresh(index, &declaration, std::mem::take(&mut children), &mut outputs)
                .map_err(ApiError::scheme)?;
        }
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
        let outputs = evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-online-eval");
        for (index, output) in outputs.into_iter().enumerate() {
            graph = graph.output(format!("output-{index}"), output).map_err(ApiError::scheme)?;
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let mut runtime = BTreeMap::new();
        if super::graph_declares_input(&graph, super::HASH_KEY_INPUT) {
            runtime.insert(
                super::HASH_KEY_INPUT.to_owned(),
                RuntimeValue::Bytes(self.hash_key.as_slice().into()),
            );
        }
        Ok(StageGraph::new(graph, runtime, evaluator.manifests.clone()))
    }
}
