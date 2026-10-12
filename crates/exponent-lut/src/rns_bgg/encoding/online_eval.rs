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
            families: OnlineEvalFamilies { parameters },
            environment: Environment::new(),
            pending_decoder: None,
            unmask_preimage_bound: None,
            hash_namespace: None,
            pending_refresh: None,
            refresh_matrix_key: None,
        }
    }

    /// Sample a top refresh's `A'` from the stage key `hash_key` derived
    /// under `namespace` (see `RnsPreprocessBuild::with_refresh_matrix_key`).
    pub fn with_refresh_matrix_key(mut self, namespace: [u8; 32], hash_key: &[u8]) -> Self {
        self.refresh_matrix_key =
            Some((namespace, crate::rns_bgg::refresh::child_hash_key(hash_key, &namespace)));
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
            return Err(ApiError::input("a refresh decoder binds through bind_refresh_child"));
        }
        let commitment = *declaration.commitment();
        let key = declaration.artifact_key();
        self.bind_child(
            declaration.smudge().program(),
            commitment,
            Some(declaration.clone()),
            decoder_artifact_name(&key, "smudge"),
            None,
            preprocess,
            encryption,
            hash_key,
        )
    }

    /// Bind this refresh child to its Program: the fresh-noise child of
    /// `declaration` for `decoder = None`, else that decoder's smudge
    /// Program. The builder must carry the child's hash namespace (see
    /// [`Self::with_hash_namespace`]). Its inputs and material are imported
    /// under the child's namespace; the public payload of its PRF label is
    /// the label hashed from `label_tag` (see
    /// [`crate::refresh::RefreshDeclaration::label`]), not the encrypted one.
    pub fn bind_refresh_child(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        decoder: Option<&DecoderDeclaration>,
        label_tag: &[u8],
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        self.bind_refresh_child_inner(
            declaration,
            false,
            decoder,
            label_tag,
            preprocess,
            encryption,
            hash_key,
        )
    }

    /// Bind this builder to the mask child of the top-refresh part
    /// `declaration` (see [`Self::bind_refresh_child`]); it must carry
    /// [`crate::rns_bgg::refresh_mask_namespace`].
    pub fn bind_refresh_mask_child(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        label_tag: &[u8],
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        self.bind_refresh_child_inner(
            declaration,
            true,
            None,
            label_tag,
            preprocess,
            encryption,
            hash_key,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_refresh_child_inner(
        &mut self,
        declaration: &crate::refresh::RefreshDeclaration,
        mask: bool,
        decoder: Option<&DecoderDeclaration>,
        label_tag: &[u8],
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let key: [u8; 32] =
            hash_key.try_into().map_err(|_| ApiError::input("the hash key is not 32 bytes"))?;
        let label =
            declaration.label(key, label_tag).map_err(|_| ApiError::input("refresh label"))?;
        let spec = *declaration.spec();
        let (namespace, program, prf) = match decoder {
            None if mask => {
                let part = declaration
                    .top()
                    .ok_or_else(|| ApiError::input("the refresh has no mask child"))?;
                (
                    crate::rns_bgg::refresh::refresh_mask_namespace(declaration),
                    part.mask().program(),
                    part.mask().sampler().prf(),
                )
            }
            None => (
                crate::rns_bgg::refresh::refresh_fresh_namespace(declaration),
                declaration.fresh().program(),
                declaration.fresh().sampler().prf(),
            ),
            Some(decoder) => {
                if !declaration
                    .decoders()
                    .iter()
                    .any(|candidate| candidate.commitment() == decoder.commitment())
                {
                    return Err(ApiError::scheme(RnsBggError::Decoder(
                        DecoderError::TargetMismatch,
                    )));
                }
                (*decoder.commitment(), decoder.smudge().program(), decoder.smudge().prf())
            }
        };
        if self.hash_namespace != Some(namespace) {
            return Err(ApiError::input("the refresh child's hash namespace does not match"));
        }
        let label = Some((prf.input_vector(), label, prf.profile().layout));
        match decoder {
            None => self.bind_child(
                program,
                spec,
                None,
                crate::rns_bgg::refresh::refresh_artifact_name(
                    spec,
                    if mask { "mask" } else { "fresh" },
                ),
                label,
                preprocess,
                encryption,
                hash_key,
            ),
            Some(decoder) => {
                let key = decoder.artifact_key();
                self.bind_child(
                    program,
                    spec,
                    Some(decoder.clone()),
                    decoder_artifact_name(&key, "smudge"),
                    label,
                    preprocess,
                    encryption,
                    hash_key,
                )
            }
        }
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
    /// exponent layout, replaces that family's encrypted public payload.
    #[allow(clippy::too_many_arguments)]
    fn bind_child(
        &mut self,
        prepared: &ExponentLutProgram,
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
        if decoder_smudge
            .as_ref()
            .is_some_and(|decoder| preprocess_binding.program() != decoder.source_program())
        {
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
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(prepared).map_err(ApiError::scheme)?);
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = match &self.hash_namespace {
            Some(namespace) => crate::rns_bgg::refresh::child_hash_key(hash_key, namespace),
            None => hash_key.to_vec(),
        };
        // The child's encryption publishes each input at its logical level
        // with its public key, under the child's namespace.
        self.environment = Environment::new();
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
                    let key_parts =
                        (0..4).map(|index| family_port(index, count, columns)).collect::<Vec<_>>();
                    let rows =
                        (4..6).map(|index| family_port(index, count, columns)).collect::<Vec<_>>();
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
        let hash_key = self.parameters.ring(0).map_err(ApiError::scheme)?.bytes_input(
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            self.hash_key.len(),
        );
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            prepared,
            &workload,
            &preprocess_production,
            &encryption_production,
            prelude.as_ref(),
            &hash_key,
            |role: String| name(&role),
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
            decoder_smudge: decoder_smudge.map(Arc::new),
            artifact_prefix: Some(prefix),
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
        let key = declaration.artifact_key();
        let ring = self.parameters.ring(level)?;
        let columns = crate::decoder::unmask::UNMASK_ROWS *
            (self.parameters.level(level)?.modulus_digits() + 2);
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
        let dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let decoded = crate::decoder::unmask::unmask(hint, preimage, encoding) -
            ring.polynomial(std::iter::repeat_n(half, dimension));
        Ok(decoded.modulus_switch(&self.parameters.ring(level + 1)?))
    }

    /// Apply `declaration`'s noise refresh to output `target_index` when the
    /// graph is built: the bound children (`fresh`, and `smudges` in decoder
    /// order) are lowered once, and the graph gains the output `refresh`, the
    /// refreshed row `y = s A' - Delta x G_Q + e'` (see [`crate::refresh`]).
    pub fn apply_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        fresh: RnsOnlineEvalBuild<'a>,
        smudges: Vec<RnsOnlineEvalBuild<'a>>,
    ) -> Result<(), ApiError> {
        self.pend_refresh(target_index, declaration, None, fresh, smudges)
    }

    /// Apply the top-refresh part `declaration` as [`Self::apply_refresh`]
    /// does, with its bound mask child `mask`: the graph output `refresh` is
    /// the part's residue `y_j mod q` of its output prime `q`, a `1 x w_0`
    /// row in the ring of `q` (see [`crate::refresh::top`]).
    pub fn apply_top_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        mask: RnsOnlineEvalBuild<'a>,
        fresh: RnsOnlineEvalBuild<'a>,
        smudges: Vec<RnsOnlineEvalBuild<'a>>,
    ) -> Result<(), ApiError> {
        self.pend_refresh(target_index, declaration, Some(mask), fresh, smudges)
    }

    fn pend_refresh(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        mask: Option<RnsOnlineEvalBuild<'a>>,
        fresh: RnsOnlineEvalBuild<'a>,
        smudges: Vec<RnsOnlineEvalBuild<'a>>,
    ) -> Result<(), ApiError> {
        if self.pending_decoder.is_some() || self.pending_refresh.is_some() {
            return Err(ApiError::input("a decoder or refresh was already applied"));
        }
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
        let smudge_parameters = smudges.iter().map(|child| child.parameters).collect::<Vec<_>>();
        self.parameters
            .validate_refresh_children(
                declaration,
                &evaluator.program,
                mask.as_ref().map(|child| child.parameters),
                fresh.parameters,
                &smudge_parameters,
            )
            .map_err(ApiError::scheme)?;
        let child_program = |child: &RnsOnlineEvalBuild<'a>| {
            child.evaluator.as_ref().map(|child| child.program.id())
        };
        let mismatch = || ApiError::scheme(RnsBggError::Decoder(DecoderError::TargetMismatch));
        if evaluator.program.outputs().get(target_index) !=
            Some(&Operand::Scalar(declaration.target_output())) ||
            child_program(&fresh) != Some(declaration.fresh().id()) ||
            fresh.hash_namespace !=
                Some(crate::rns_bgg::refresh::refresh_fresh_namespace(declaration))
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
        if let (Some(part), Some(child)) = (declaration.top(), mask.as_ref()) &&
            (child_program(child) != Some(part.mask().id()) ||
                child.hash_namespace !=
                    Some(crate::rns_bgg::refresh::refresh_mask_namespace(declaration)))
        {
            return Err(mismatch());
        }
        if declaration.top().is_some() && self.refresh_matrix_key.is_none() {
            return Err(ApiError::input("a top refresh needs with_refresh_matrix_key"));
        }
        self.pending_refresh =
            Some((target_index, declaration.clone(), mask.map(Box::new), Box::new(fresh), smudges));
        Ok(())
    }

    /// The refreshed row of the pending refresh for the lowered Program
    /// `outputs`, and the hash-key inputs of its children.
    fn refresh_row(
        &mut self,
        target_index: usize,
        declaration: &crate::refresh::RefreshDeclaration,
        mut mask: Option<RnsOnlineEvalBuild<'a>>,
        mut fresh: RnsOnlineEvalBuild<'a>,
        mut smudges: Vec<RnsOnlineEvalBuild<'a>>,
        outputs: &[RnsBggEncoding],
    ) -> Result<(Mat, HashKeyInputs), RnsBggError> {
        use crate::rns_bgg::refresh::{
            hash_key_input_name, refresh_artifact_name, refresh_column, refresh_combination,
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
        // The fresh carriers' component-1 rows, moved down once to the
        // decoding ring and never finalized for decryption.
        let fresh_rows = fresh_outputs
            .into_iter()
            .map(|encoding| mod_down_encoding(encoding.row, &ring).1)
            .collect::<Vec<_>>();
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
        let matrix_key = match (declaration.top(), &self.refresh_matrix_key) {
            (None, _) => evaluator.hash_key.clone(),
            (Some(_), Some((namespace, key))) => {
                ring.bytes_input(hash_key_input_name(Some(namespace)), key.len())
            }
            (Some(_), None) => {
                return Err(RnsBggError::Binding("a top refresh needs its matrix key"));
            }
        };
        let a_prime = self.parameters.refresh_matrix(level, matrix_key, declaration)?;
        // A top-refresh part's one encoding: the mask carrier's component-1
        // row (moved down once) for the first block.
        let one = match mask.as_mut() {
            None => one,
            Some(child) => {
                let (outputs, child_manifests) = lower(child)?;
                manifests.extend(child_manifests);
                let carrier = outputs
                    .into_iter()
                    .next()
                    .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?;
                (mod_down_encoding(carrier.row, &ring).1, one.1)
            }
        };
        let target_row =
            if declaration.component() == 0 { target.row.0.clone() } else { target.row.1.clone() };
        let output_columns = declaration.columns();
        // The halved smudge weights leave `2^-1` in every coefficient (see
        // `crate::decoder::decoding`); removing it centers the smudge.
        let modulus: BigUint = self.parameters.level(level)?.modulus().as_ref().clone();
        let dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let half = ring
            .polynomial(std::iter::repeat_n(IntExpr::constant((modulus + 1u8) / 2u8), dimension));
        let mut unmasked = Vec::with_capacity(declaration.primes());
        for (position, prime) in declaration.decoded_primes().into_iter().enumerate() {
            let actions =
                self.parameters.refresh_actions(level, prime, a_prime.clone(), declaration)?;
            let decoded =
                refresh_combination(one.clone(), target_row.clone(), &fresh_rows, &actions);
            let mut row = Vec::with_capacity(output_columns);
            for column in 0..output_columns {
                let decoder = declaration
                    .decoder(prime, column)
                    .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?;
                let (smudge, child_manifests) =
                    lower(&mut smudges[position * output_columns + column])?;
                manifests.extend(child_manifests);
                let smudge = smudge
                    .into_iter()
                    .next()
                    .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?;
                let encoding = refresh_column(decoded.clone(), column) +
                    checked_modulus_convert(smudge.row.1, &ring)?;
                let preimage = ring.preimage_artifact_input(
                    preprocess.clone(),
                    decoder_artifact_name(&decoder.artifact_key(), "unmask-preimage"),
                    (hint_columns, 1),
                    BigInt::from(bound.clone()),
                    ArtifactAvailability::Transferred,
                );
                row.push(
                    crate::decoder::unmask::unmask(hint.clone(), preimage, encoding) - half.clone(),
                );
            }
            unmasked.push(Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, row));
        }
        let refreshed = match declaration.top() {
            // `crt_recompose` rounds every `Q_i y + M + N` to `y mod q_i` and
            // recombines the residues modulo `Q`, column by column.
            None => {
                let (primes, coefficients) = self.parameters.refresh_crt(level)?;
                Mat::crt_recompose(unmasked, primes, coefficients, &ring)
            }
            // A top-refresh part rounds `Q_q y + M + N` to its residue
            // `y mod q`; the chains' residues are recombined modulo `Q_0`.
            Some(part) => unmasked
                .into_iter()
                .next()
                .ok_or(RnsBggError::Decoder(DecoderError::TargetMismatch))?
                .modulus_switch(&Ring::from_crt_moduli(
                    vec![IntExpr::from(part.output_modulus())],
                    self.parameters.level(level)?.ring_dimension(),
                )),
        };
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online evaluation artifacts are not bound"))?;
        evaluator.manifests.extend(manifests);
        let mut hash_keys = mask
            .iter()
            .chain(std::iter::once(&fresh))
            .chain(&smudges)
            .map(|child| {
                (hash_key_input_name(child.hash_namespace.as_ref()), child.hash_key.clone())
            })
            .collect::<Vec<_>>();
        if let Some((namespace, key)) = &self.refresh_matrix_key {
            hash_keys.push((hash_key_input_name(Some(namespace)), key.clone()));
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
        let hash_key = self.parameters.ring(0).map_err(ApiError::scheme)?.bytes_input(
            crate::rns_bgg::refresh::hash_key_input_name(self.hash_namespace.as_ref()),
            self.hash_key.len(),
        );
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
            artifact_prefix: None,
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
        let refreshed = match self.pending_refresh.take() {
            Some((index, declaration, mask, fresh, smudges)) => Some(
                self.refresh_row(
                    index,
                    &declaration,
                    mask.map(|mask| *mask),
                    *fresh,
                    smudges,
                    &outputs,
                )
                .map_err(ApiError::scheme)?,
            ),
            None => None,
        };
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online evaluation artifacts are not bound"))?;
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
        let mut runtime = BTreeMap::new();
        for (name, key) in hash_keys {
            if super::graph_declares_input(&graph, &name) {
                runtime.insert(name, RuntimeValue::Bytes(key.as_slice().into()));
            }
        }
        Ok(StageGraph::new(graph, runtime, evaluator.manifests.clone()))
    }
}
