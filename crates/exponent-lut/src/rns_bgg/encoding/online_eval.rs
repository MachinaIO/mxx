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
            pending_decoder: None,
            unmask_preimage_bound: None,
        }
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
        let prepared = declaration.smudge().program();
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        let expected_hash_key = hash_key_commitment(hash_key);
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.program() != declaration.source_program() ||
                binding.hash_key_commitment() != &expected_hash_key ||
                binding.decoder_commitment() != Some(declaration.commitment())
            {
                return Err(ApiError::compatibility(
                    "decoder smudge artifacts do not match the source declaration",
                ));
            }
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
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(prepared).map_err(ApiError::scheme)?);
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        // The smudge child's encryption publishes each input at its logical
        // level with its public key, under the decoder namespace.
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
                                    decoder_smudge_artifact_name(
                                        commitment,
                                        &format!("input-{wire:?}.0"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    decoder_smudge_artifact_name(
                                        commitment,
                                        &format!("input-{wire:?}.1"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                            ),
                            (
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    decoder_smudge_artifact_name(
                                        commitment,
                                        &format!("input-{wire:?}.2"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    decoder_smudge_artifact_name(
                                        commitment,
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
                            decoder_smudge_artifact_name(commitment, &format!("input-{wire:?}.4")),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        ),
                        ring.artifact_input(
                            encryption_production.clone(),
                            decoder_smudge_artifact_name(commitment, &format!("input-{wire:?}.5")),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        ),
                    ),
                    public.then(|| {
                        ring.artifact_input(
                            encryption_production.clone(),
                            decoder_smudge_artifact_name(commitment, &format!("input-{wire:?}.6")),
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
                                decoder_smudge_artifact_name(
                                    commitment,
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
                                decoder_smudge_artifact_name(
                                    commitment,
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
                            decoder_smudge_artifact_name(commitment, &format!("input-{wire:?}.6")),
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
            prelude.as_ref(),
            &hash_key,
            |role: String| decoder_smudge_artifact_name(commitment, &role),
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
            decoder_smudge: Some(Arc::new(declaration.clone())),
            current_site: None,
            hash_key,
            context: self.context,
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
        });
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_rhs_companions()
            .map_err(ApiError::scheme)?;
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_body_lut_material()
            .map_err(ApiError::scheme)?;
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
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        evaluator.manifests.extend(child_manifests);
        let commitment = *declaration.commitment();
        let ring = self.parameters.ring(level)?;
        let columns = crate::decoder::unmask::UNMASK_ROWS *
            (self.parameters.level(level)?.modulus_digits() + 2);
        let bound = self
            .unmask_preimage_bound
            .clone()
            .ok_or(RnsBggError::Binding("the unmask preimage bound is not bound"))?;
        let preimage = ring.preimage_artifact_input(
            evaluator.preprocess.clone(),
            decoder_artifact_name(commitment, "unmask-preimage"),
            (columns, 1),
            BigInt::from(bound),
            ArtifactAvailability::Transferred,
        );
        let hint = ring.artifact_input(
            evaluator.encryption.clone(),
            decoder_artifact_name(commitment, "unmask-hint"),
            (1, columns),
            ArtifactAvailability::Transferred,
        );
        // The halved smudge weights leave `2^-1` in every coefficient (see
        // `crate::decoder::decoding`); removing it centers the smudge.
        let modulus: BigUint = self.parameters.level(level)?.modulus().as_ref().clone();
        let half = IntExpr::constant((modulus + 1u8) / 2u8);
        let dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let decoded = crate::decoder::unmask::unmask(hint, preimage, encoding) -
            ring.polynomial(std::iter::repeat_n(half, dimension));
        Ok(decoded.modulus_switch(&self.parameters.ring(level + 1)?))
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
        if preprocess_binding.decoder_commitment() != encryption_binding.decoder_commitment() {
            return Err(ApiError::compatibility(
                "RNS-BGG decoder declaration commitment differs between stages",
            ));
        }
        self.unmask_preimage_bound = preprocess_binding.unmask_preimage_bound().cloned();
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let prelude = encryption_binding.ciphertext_prelude().map(|(production, manifest)| {
            manifests.insert(production.clone(), manifest.clone());
            production.clone()
        });
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
            prelude.as_ref(),
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
            decoder_smudge: None,
            current_site: None,
            hash_key,
            context: self.context,
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
        });
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_rhs_companions()
            .map_err(ApiError::scheme)?;
        self.evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?
            .import_body_lut_material()
            .map_err(ApiError::scheme)?;
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
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
        let outputs = evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-online-eval");
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
