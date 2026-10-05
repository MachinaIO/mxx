//! Public-key lowering of Program operations.
use super::*;

/// A family bound for preprocessing.
///
/// A stored family holds complete public keys. An input family holds only what
/// its keys are sampled from: a member read samples that member's key from the
/// hash key in place (`sampled_key`), as online evaluation does, so no stage
/// materializes the keys of a whole input family.
#[derive(Clone)]
pub enum PreprocessWireFamily {
    Stored(Family<RnsBggPublicKey>),
    Input {
        wire: WireId,
        level: usize,
        knowledge: AttributeKnowledge,
        hash_key: Bytes,
        count: IntExpr,
    },
}

impl PreprocessWireFamily {
    /// Read one member: an input member's key is sampled from the hash key.
    fn member(
        &self,
        parameters: &RnsBggParameters,
        index: IntExpr,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        match self {
            Self::Stored(family) => Ok(family.at(Int::evaluate(index))),
            Self::Input { wire, level, knowledge, hash_key, .. } => {
                sampled_key(parameters, *level, hash_key.clone(), *wire, *knowledge, Some(index))
            }
        }
    }

    fn count(&self) -> IntExpr {
        match self {
            Self::Stored(family) => family.count().clone(),
            Self::Input { count, .. } => count.clone(),
        }
    }
}

/// Preprocessing reads families through the same execution as online
/// evaluation, which samples an input member's public key where it is read.
impl<'a, 'b> crate::program::FamilyExecution<PublicKeyLowering<'a>>
    for super::encoding::OnlineEvalFamilies<'b>
{
    type WireFamily = PreprocessWireFamily;
    type RhsFamily = Family<RhsRef<Int>>;

    fn wire_at(
        &mut self,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<RnsBggPublicKey, crate::program::ProgramLowerError<RnsBggError>> {
        family
            .member(self.parameters, index.clone())
            .map_err(crate::program::ProgramLowerError::Backend)
    }

    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<RhsRef<Int>, crate::program::ProgramLowerError<RnsBggError>> {
        // A member reference names its member by the index expression itself,
        // so a read can tell the loop member it belongs to.
        Ok(RhsRef {
            id: mxx_dsl::GraphValue::schema(family).element.id,
            index: Some(Int::evaluate(index.clone())),
        })
    }

    fn parallel(
        &mut self,
        evaluator: &mut PublicKeyLowering<'a>,
        range: &crate::program::FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut PublicKeyLowering<'a>,
            IntExpr,
        )
            -> Result<RnsBggPublicKey, crate::program::ProgramLowerError<RnsBggError>>,
    ) -> Result<Self::WireFamily, crate::program::ProgramLowerError<RnsBggError>> {
        crate::program::graph_parallel(self, evaluator, range, body)
            .map(PreprocessWireFamily::Stored)
    }

    fn sum(
        &mut self,
        evaluator: &mut PublicKeyLowering<'a>,
        family: &Self::WireFamily,
    ) -> Result<RnsBggPublicKey, crate::program::ProgramLowerError<RnsBggError>> {
        use crate::program::ProgramLowerError;
        let parameters = self.parameters;
        crate::program::graph_sum(evaluator, family.count(), |index| {
            let index = index.expression().map_err(ProgramLowerError::Dsl)?;
            family.member(parameters, index).map_err(ProgramLowerError::Backend)
        })
    }
}

impl<'a> PublicKeyLowering<'a> {
    /// Construct a public-key lowerer for one immutable Program/workload pair.
    pub(crate) fn new(
        parameters: &'a RnsBggParameters,
        program: Arc<ExponentLutProgram>,
        workload: Arc<RnsBggDerivedWorkload>,
        payload_secret: Option<RnsBggPayloadSecret>,
        hash_key: Bytes,
        context: [u8; 32],
    ) -> Self {
        Self {
            parameters,
            program,
            workload,
            payload_secret,
            hash_key,
            context,
            refresh_mask: None,
            rhs: BTreeMap::new(),
            rhs_material: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            refresh_decoder: None,
            current_site: None,
            parallel_bodies: Vec::new(),
            finished_bodies: Vec::new(),
            streamed_rhs: BTreeMap::new(),
            prepared_rhs_reads: std::collections::BTreeSet::new(),
        }
    }

    /// The material of RHS family member `rhs` when the root-level loop that
    /// reads it prepares it: the member is that loop's own member, and the
    /// loop runs once per family member. The body prepares it once and the
    /// loop returns it for export, so no stage keeps the family's materials
    /// resident. A family read any other way uses the materials prepared at
    /// binding, and may not also be read this way.
    fn streamed_rhs_member(
        &mut self,
        rhs: &RhsRef<Int>,
    ) -> Result<Option<RnsBggMatrix>, RnsBggError> {
        let Some(InputBinding::Family(plaintexts)) = self.rhs.get(&rhs.id) else {
            return Ok(None);
        };
        let members = plaintexts.count().clone();
        let index = rhs
            .index
            .as_ref()
            .map(Int::expression)
            .transpose()
            .map_err(|_| RnsBggError::Binding("RHS member index is not an index expression"))?
            .map(|index| index.canonicalize());
        let own_member = match (self.parallel_bodies.as_slice(), &index) {
            ([(count, member, _)], Some(index)) => {
                let empty = mxx_ir_core::ParamEnv::default();
                member == index &&
                    count
                        .evaluate(&empty)
                        .is_ok_and(|count| members.evaluate(&empty) == Ok(count))
            }
            _ => false,
        };
        if !own_member || self.prepared_rhs_reads.contains(&rhs.id) {
            if self.streamed_rhs.contains_key(&rhs.id) {
                return Err(RnsBggError::Binding(
                    "an RHS family its loop prepares is also read elsewhere",
                ));
            }
            self.prepared_rhs_reads.insert(rhs.id);
            return Ok(None);
        }
        if matches!(self.streamed_rhs.get(&rhs.id), Some(Some(_))) {
            return Err(RnsBggError::Binding("an RHS family is read by two loops"));
        }
        if let Some((_, material)) =
            self.parallel_bodies[0].2.iter().find(|(prepared, _)| *prepared == rhs.id)
        {
            return Ok(Some(material.clone()));
        }
        let material = self.prepare_rhs_material(rhs)?;
        self.parallel_bodies[0].2.push((rhs.id, material.clone()));
        self.streamed_rhs.insert(rhs.id, None);
        Ok(Some(material))
    }

    /// The materials an RHS family publishes: those its loop prepared, or
    /// else those `prepared` at binding.
    pub(super) fn published_rhs_family(
        &self,
        rhs: &RhsId,
        prepared: &Family<RnsBggMatrix>,
    ) -> Result<Family<RnsBggMatrix>, ApiError> {
        match self.streamed_rhs.get(rhs) {
            Some(Some(streamed)) => Ok(streamed.clone()),
            Some(None) => Err(ApiError::input("an RHS family loop was not committed")),
            None => Ok(prepared.clone()),
        }
    }
    fn site(&self) -> Result<&LoweringSite, RnsBggError> {
        self.current_site.as_ref().ok_or(RnsBggError::Binding("missing lowering site"))
    }
    fn output(&self) -> Result<WireId, RnsBggError> {
        Ok(self.site()?.output)
    }

    /// The Static payload secret, which only Static material reads.
    pub(super) fn payload(&self) -> Result<&RnsBggPayloadSecret, RnsBggError> {
        self.payload_secret
            .as_ref()
            .ok_or(RnsBggError::Binding("Static preprocessing requires the payload secret"))
    }

    /// The pre-rounding Dynamic product key `T = sum_a B_{a,j} Dec(A)_{:,a}`
    /// of state key `key` at `level`, with companion keys from `material`'s
    /// tag (see [`super::companion_columns`]). A LUT branch passes its
    /// automorphism `sigma`, which maps each stored key `B^0` to
    /// `sigma(B^0)`, the key of the automorphed companion encoding.
    /// `columns` is the ciphertext's per-component output width.
    fn companion_product_key(
        &self,
        level: usize,
        key: &RnsBggMatrix,
        columns: usize,
        material: String,
        member: Option<IntExpr>,
        sigma: Option<IntExpr>,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let tag = super::companion_tag(
            self.parameters,
            &self.context,
            self.program.id(),
            self.refresh_mask.as_ref().map(|declaration| *declaration.commitment()),
            material,
            member,
            level,
        );
        let rows = [&key.0, &key.1].map(super::join_row);
        let key_entry = |_, r: usize, a| Ok(super::row_entry(&rows[r], a));
        let companion_key = |_, a, j| {
            let key =
                super::companion_key(self.parameters, level, self.hash_key.clone(), &tag, a, j)?;
            Ok(match &sigma {
                Some(sigma) => map_matrix(key, |entry| entry.ring_automorphism(sigma.clone())),
                None => key,
            })
        };
        let rows = super::companion_columns(
            self.parameters,
            level,
            1,
            columns,
            &key_entry,
            &companion_key,
            None,
        )?;
        Ok((super::split_row(&rows[0], columns), super::split_row(&rows[1], columns)))
    }

    /// Sample an input key with shape selected from Program knowledge.
    pub(crate) fn sample_input_public_key(
        &self,
        wire: WireId,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        self.sample_input_public_key_member(wire, None)
    }

    pub(super) fn sample_input_public_key_member(
        &self,
        wire: WireId,
        member: Option<Int>,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        let knowledge = self
            .program
            .wire_knowledge(wire)
            .map_err(|_| RnsBggError::Binding("wire knowledge"))?;
        let member =
            member.as_ref().map(Int::expression).transpose().map_err(|_| {
                RnsBggError::Binding("input member index is not an index expression")
            })?;
        sampled_key(
            self.parameters,
            self.workload.level(wire)?,
            self.hash_key.clone(),
            wire,
            knowledge,
            member,
        )
    }
    /// Produce the compact setup-fixed RHS ciphertext.  Only this source is
    /// retained; its signed decomposition is deliberately deferred to the
    /// consuming operation.
    pub(super) fn prepare_rhs_material(
        &mut self,
        rhs: &RhsRef<Int>,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let level = self.workload.rhs_level(rhs.id)?;
        // Bind the sampled RHS to the caller's plaintext through the same
        // payload-constrained ciphertext relation used by private input
        // encryption. The plaintext is a graph handle; it is never copied
        // into the artifact identity or the online evaluation lowerer.
        let plaintext = match self.rhs.get(&rhs.id) {
            Some(InputBinding::Scalar(value)) => value.clone(),
            Some(InputBinding::Family(plaintexts)) => plaintexts.member(
                &self.parameters.ring(0)?,
                rhs.index.clone().ok_or(RnsBggError::Binding("family RHS index"))?,
            ),
            None => return Err(RnsBggError::Binding("RHS plaintext was not bound")),
        };
        // The member enters the tag as an index expression, as it does where
        // online evaluation reads the member.
        let member = rhs
            .index
            .as_ref()
            .map(Int::expression)
            .transpose()
            .map_err(|_| RnsBggError::Binding("RHS member index is not an index expression"))?;
        rhs_ciphertext(self.parameters, level, &self.payload()?.row, plaintext, || {
            rhs_top(
                self.parameters,
                self.program.id(),
                self.hash_key.clone(),
                rhs.id,
                level,
                member,
            )
        })
    }
    /// Materialize setup-fixed compact `B_sigma` switching ciphertexts for a
    /// complete LUT/layout identity evaluated at `level`. Gadget
    /// decomposition is deferred until the branch is consumed. The family is
    /// built in the current scope and cached; a LUT inside a family body
    /// reuses material prepared at root scope.
    pub(crate) fn prepare_lut_switching_ciphertexts(
        &mut self,
        level: usize,
        lut_id: LutId,
        table: &LutTable,
    ) -> Result<Family<RnsBggMatrix>, RnsBggError> {
        let next_level =
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let target_ring = self.parameters.ring(next_level)?;
        let layout = table.input_layout();
        layout
            .validate(self.parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let key = (level, next_level, lut_id, layout);
        if let Some(switching_ciphertexts) = self.lut_material.get(&key) {
            return Ok(switching_ciphertexts.clone());
        }
        let count = IntExpr::constant(table.values().len());
        let sigmas = layout
            .automorphism_indices(self.parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let payload = &self.payload()?.row;
        let columns =
            self.parameters.lut_output_columns(next_level, table.output_representation())?;
        let switching_ciphertexts = mxx_dsl::parallel(count, |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let mut tag = mxx_dsl::HashTag::from(
                format!("rns-bgg/lut-b/{:?}/{level}/{next_level}", lut_id).into_bytes(),
            );
            tag.push("sigma");
            tag.push(sigma.clone());
            tag.push("branch");
            tag.push(index.expression()?);
            lut_switching_ciphertext(self.parameters, level, table, payload, sigma, || {
                Ok(matrix_from_hash(&target_ring, self.hash_key.clone(), tag, columns).0)
            })
            .map_err(|_| DslError::Schema)
        })
        .map_err(graph)?;
        self.lut_material.insert(key, switching_ciphertexts.clone());
        Ok(switching_ciphertexts)
    }

    /// ModDown declared coefficient outputs once after graph aggregation.
    pub(crate) fn finish_outputs(
        &mut self,
        outputs: Vec<RnsBggPublicKey>,
    ) -> Result<Vec<RnsBggPublicKey>, RnsBggError> {
        outputs
            .into_iter()
            .map(|output| {
                if self
                    .program
                    .wire_is_coefficient_output(output.wire)
                    .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
                {
                    let level = self.workload.level(output.wire)?;
                    let target = level
                        .checked_add(1)
                        .ok_or(RnsBggError::Parameters("coefficient output level overflow"))?;
                    let q_ring = self.parameters.ring(target)?;
                    Ok(RnsBggPublicKey::new(
                        output.wire,
                        super::mod_down_public(output.matrix, &q_ring),
                    ))
                } else {
                    Ok(output)
                }
            })
            .collect()
    }
}

impl ExponentLUTProgramLowering for PublicKeyLowering<'_> {
    type Wire = RnsBggPublicKey;
    type Rhs = RhsRef<Int>;
    type Error = RnsBggError;
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        if self.workload.level(lhs.wire)? != self.workload.level(rhs.wire)? {
            return Err(RnsBggError::Binding("Add level mismatch"));
        }
        Ok(RnsBggPublicKey::new(
            self.output()?,
            super::add_public_matrices(lhs.matrix.clone(), rhs.matrix.clone()),
        ))
    }
    fn mul_public(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let lhs_knowledge = self
            .program
            .wire_knowledge(lhs.wire)
            .map_err(|_| RnsBggError::Binding("MulPublic LHS metadata"))?;
        let rhs_knowledge = self
            .program
            .wire_knowledge(rhs.wire)
            .map_err(|_| RnsBggError::Binding("MulPublic RHS metadata"))?;
        if lhs_knowledge != AttributeKnowledge::PublicPayload {
            return Err(RnsBggError::Binding("MulPublic LHS must be public"));
        }
        if self
            .program
            .wire_is_coefficient_output(lhs.wire)
            .map_err(|_| RnsBggError::Binding("MulPublic LHS metadata"))? ||
            self.program
                .wire_is_coefficient_output(rhs.wire)
                .map_err(|_| RnsBggError::Binding("MulPublic RHS metadata"))?
        {
            return Err(RnsBggError::Binding("MulPublic requires exponent operands"));
        }
        let lhs_level = self.workload.level(lhs.wire)?;
        let rhs_level = self.workload.level(rhs.wire)?;
        let target_level =
            self.parameters.mul_public_target_level(lhs_level, rhs_level, rhs_knowledge)?;

        // The public LHS is aligned to the RHS-owned target level. A public
        // RHS is aligned as well when public/public selects the lower
        // modulus; equal physical levels are a graph no-op.
        let lhs_matrix = apply_public_alignment_matrix(
            self.parameters,
            lhs_level,
            target_level,
            lhs.matrix.clone(),
        )?;
        let rhs_matrix = if rhs_knowledge == AttributeKnowledge::PublicPayload {
            apply_public_alignment_matrix(
                self.parameters,
                rhs_level,
                target_level,
                rhs.matrix.clone(),
            )?
        } else {
            rhs.matrix.clone()
        };
        let rhs_decomposition = decompose_matrix(self.parameters, target_level, rhs_matrix)?;
        Ok(RnsBggPublicKey::new(
            self.output()?,
            super::multiply_public_action(lhs_matrix, rhs_decomposition),
        ))
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.wire)
            .map_err(|_| RnsBggError::Binding("MulPrivate input metadata"))? ==
            AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Binding("RNS-BGG MulPrivate requires a private input"));
        }
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            // The key is fixed from public companion keys alone; the RHS
            // ciphertext and its digits exist only after encryption.
            let level = self.workload.level(input.wire)?;
            let rhs_level = self.workload.rhs_level(rhs.id)?;
            if rhs_level != level {
                return Err(RnsBggError::RhsLevelMismatch {
                    rhs: rhs.id,
                    expected: level,
                    actual: rhs_level,
                });
            }
            let member =
                rhs.index.as_ref().map(Int::expression).transpose().map_err(|_| {
                    RnsBggError::Binding("RHS member index is not an index expression")
                })?;
            let key = self.companion_product_key(
                level,
                &input.matrix,
                self.parameters.component_columns(level + 1)?,
                format!("rhs/{:?}", rhs.id),
                member,
                None,
            )?;
            let target_ring = self.parameters.ring(self.workload.level(self.output()?)?)?;
            return Ok(RnsBggPublicKey::new(
                self.output()?,
                super::mod_down_public(key, &target_ring),
            ));
        }
        let streamed = self.streamed_rhs_member(rhs)?;
        let compact = match (streamed, self.rhs_material.get(&rhs.id)) {
            (Some(material), _) => material,
            (None, Some(InputBinding::Scalar(value))) => value.clone(),
            (None, Some(InputBinding::Family(values))) => {
                values.at(rhs.index.clone().ok_or(RnsBggError::Binding("family RHS index"))?)
            }
            (None, None) => return Err(RnsBggError::Binding("RHS material was not prepared")),
        };
        let evaluation_level = self.workload.level(input.wire)?;
        let rhs_level = self.workload.rhs_level(rhs.id)?;
        if rhs_level != evaluation_level {
            return Err(RnsBggError::RhsLevelMismatch {
                rhs: rhs.id,
                expected: evaluation_level,
                actual: rhs_level,
            });
        }
        let decomposition_level = evaluation_level
            .checked_add(1)
            .ok_or(RnsBggError::Parameters("RHS target level overflow"))?;
        let target_ring = self.parameters.ring(decomposition_level)?;
        let compact = checked_modulus_convert_matrix(compact, &target_ring)?;
        let compact = decompose_matrix(self.parameters, decomposition_level, compact)?;
        let evaluation_ring = self.parameters.ring(evaluation_level)?;
        // Only the bounded decomposition digits cross back into the
        // evaluation ring; the compact RHS artifact itself remains at its
        // native next-level schema.
        let action = super::centered_rebase_compact(compact, &evaluation_ring);
        let target_level = self.workload.level(self.output()?)?;
        let target_ring = self.parameters.ring(target_level)?;
        Ok(RnsBggPublicKey::new(
            self.output()?,
            super::mod_down_public(
                super::multiply_public_action(input.matrix.clone(), action),
                &target_ring,
            ),
        ))
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.wire)
            .map_err(|_| RnsBggError::Binding("LUT input metadata"))? ==
            AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Binding("RNS-BGG LUT requires a private input"));
        }
        let lut_id = self.program.lut_id_at(self.site()?.output).map_err(|_| RnsBggError::Lut)?;
        let source_level = self.workload.level(input.wire)?;
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        // Dynamic switching ciphertexts are generated by encryption; the key
        // uses their public companion keys only.
        let switching_ciphertexts = (!dynamic)
            .then(|| self.prepare_lut_switching_ciphertexts(source_level, lut_id, table))
            .transpose()?;
        let next_level =
            source_level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let source_ring = self.parameters.ring(source_level)?;
        let target_ring = self.parameters.ring(next_level)?;
        let layout = table.input_layout();
        let sigmas = layout
            .automorphism_indices(self.parameters.level(source_level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        // Accumulate the branches sequentially: modular addition is exact and
        // associative, so the sum and its noise bound equal the balanced tree,
        // while only one branch is resident instead of the whole branch family
        // and every tree level.
        let columns =
            self.parameters.lut_output_columns(next_level, table.output_representation())?;
        let zero = source_ring.zero((1, columns));
        let initial = ((zero.clone(), zero.clone()), (zero.clone(), zero));
        let result =
            mxx_dsl::iterate(IntExpr::constant(table.values().len()), initial, |index, sum| {
                let sigma = IntExpr::Select {
                    selector: Box::new(index.expression()?),
                    branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
                };
                let routed = super::map_matrix(input.matrix.clone(), |entry| {
                    entry.ring_automorphism(sigma.clone())
                });
                let switched = match &switching_ciphertexts {
                    Some(switching_ciphertexts) => {
                        let switching_decomposition = super::centered_rebase_compact(
                            decompose_matrix(
                                self.parameters,
                                next_level,
                                switching_ciphertexts.at(index.clone()),
                            )
                            .map_err(|_| DslError::Schema)?,
                            &source_ring,
                        );
                        super::multiply_public_action(routed, switching_decomposition)
                    }
                    None => self
                        .companion_product_key(
                            source_level,
                            &routed,
                            columns,
                            format!("lut/{lut_id:?}"),
                            Some(index.expression()?),
                            Some(sigma.clone()),
                        )
                        .map_err(|_| DslError::Schema)?,
                };
                let switched_public_key = super::mod_down_public(switched, &target_ring);
                // A_sigma is sampled in its own branch, so one is resident.
                let data = lut_public_matrix(
                    self.parameters,
                    self.hash_key.clone(),
                    source_level,
                    Int::from(sigma),
                )
                .map_err(|_| DslError::Schema)?;
                let branch = if switching_ciphertexts.is_some() {
                    let switched_key_decomposition = super::centered_rebase_compact(
                        decompose_matrix(self.parameters, next_level, switched_public_key)
                            .map_err(|_| DslError::Schema)?,
                        &source_ring,
                    );
                    public_lut_branch(data, switched_key_decomposition)
                } else {
                    // Dynamic: the rounded key is decomposed one column at a
                    // time.
                    let rows = super::column_action(
                        self.parameters,
                        source_level,
                        next_level,
                        &switched_public_key,
                        columns,
                        vec![super::join_row(&data.0), super::join_row(&data.1)],
                    )
                    .map_err(|_| DslError::Schema)?;
                    (super::split_row(&rows[0], columns), super::split_row(&rows[1], columns))
                };
                Ok(super::add_public_matrices(sum, branch))
            })
            .map_err(graph)?;
        let output = self.output()?;
        if let Some(declaration) = self.refresh_mask.clone() &&
            let Some(family) = declaration.mask().family_of_lut(lut_id)
        {
            // The refresh mask member is finalized once and transformed in
            // its root body, exactly as the encoding does.
            let finalized = next_level;
            let member =
                self.site()?.indices.first().cloned().ok_or(RnsBggError::Binding(
                    "a refresh mask LUT must be a root family member",
                ))?;
            let (action, coefficient) = super::refresh_member_action(
                self.parameters,
                &declaration,
                family,
                member,
                finalized,
            )?;
            let finalized = super::mod_down_public(result, &self.parameters.ring(finalized)?);
            return Ok(RnsBggPublicKey::new(
                output,
                map_matrix(finalized, |entry| super::act_on_entry(entry, &action, &coefficient)),
            ));
        }
        if !self
            .program
            .wire_is_coefficient_output(output)
            .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
        {
            let target = self.workload.level(output)?;
            let ring = self.parameters.ring(target)?;
            return Ok(RnsBggPublicKey::new(output, super::mod_down_public(result, &ring)));
        }
        Ok(RnsBggPublicKey::new(output, result))
    }
}

impl ProgramLoweringContext for PublicKeyLowering<'_> {
    fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl GraphParallelLowering for PublicKeyLowering<'_> {
    /// The RHS family members the body prepared (`streamed_rhs_member`).
    type SideOutput = Vec<RnsBggMatrix>;

    fn begin_graph_parallel(&mut self, count: &IntExpr, index: &IntExpr) {
        self.parallel_bodies.push((count.clone(), index.canonicalize(), Vec::new()));
    }

    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
        let (_, _, prepared) = self
            .parallel_bodies
            .pop()
            .ok_or(RnsBggError::Binding("a parallel body finished before it began"))?;
        let (families, materials): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        self.finished_bodies.push(families);
        Ok(materials)
    }

    fn commit_graph_parallel(&mut self, side: Family<Self::SideOutput>) -> Result<(), Self::Error> {
        let families = self
            .finished_bodies
            .pop()
            .ok_or(RnsBggError::Binding("a parallel loop committed before its body finished"))?;
        for (position, rhs) in families.into_iter().enumerate() {
            let family = side
                .field(|materials| materials.into_iter().nth(position).expect("side output field"))
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            self.streamed_rhs.insert(rhs, Some(family));
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        api::execute_graph,
        program::{GraphFamilyExecution, ProgramValidationError},
    };
    use mxx_backends::{MemoryArtifactStore, backend::poly::cpu_backend};
    use mxx_ir_core::ParamEnv;

    #[test]
    fn cloned_program_lowers_equal_lut_registrations_by_instruction_identity() {
        let high = DCRTPolyParams::new(16, 2, 17, 2, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let layout = ExpLayout { stride_log2: 1, width_log2: 1 };
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = LutTable::unary(
            2,
            layout,
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(layout),
        )
        .unwrap();
        let first = builder.register_lut(table.clone()).unwrap();
        let second = builder.register_lut(table).unwrap();
        // Reverse registration order so neither pointer identity nor
        // invocation count can stand in for the ID.
        let output_second = builder.lut(crate::program::Operand::Scalar(input), second).unwrap();
        let output_first = builder.lut(crate::program::Operand::Scalar(input), first).unwrap();
        let program = builder
            .finish(vec![
                crate::program::Operand::Scalar(output_second),
                crate::program::Operand::Scalar(output_first),
            ])
            .unwrap();
        let owned_program = Arc::new(program.clone());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let ring = parameters.ring(0).unwrap();
        let hash_key = ring.bytes_input("equal-luts-hash-key", 32);
        let mut public = PublicKeyLowering::new(
            &parameters,
            owned_program.clone(),
            workload.clone(),
            Some(RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)), binding: None }),
            hash_key.clone(),
            [0; 32],
        );
        let input_key = public.sample_input_public_key(input).unwrap();
        let mut public_environment = Environment::new();
        public_environment.bind_wire(input, InputBinding::Scalar(input_key.clone())).unwrap();
        let public_outputs = lower_program(
            &program,
            &mut public,
            &mut GraphFamilyExecution,
            &mut public_environment,
        )
        .unwrap();
        let ids = public.lut_material.keys().map(|key| key.2).collect::<Vec<_>>();
        assert_eq!(ids, [first, second]);

        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash(rand::random()),
            execution_nonce: rand::random(),
        };
        let mut online_eval = EncodingLowering {
            parameters: &parameters,
            program: owned_program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            refresh_mask: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            current_site: None,
            hash_key,
        };
        let row = ring.input("equal-luts-input", (1, parameters.component_columns(0).unwrap()));
        let mut online_eval_environment = Environment::new();
        online_eval_environment
            .bind_wire(
                input,
                InputBinding::Scalar(RnsBggEncoding::new(input_key, (row.clone(), row), None)),
            )
            .unwrap();
        let online_eval_outputs = lower_program(
            &program,
            &mut online_eval,
            &mut GraphFamilyExecution,
            &mut online_eval_environment,
        )
        .unwrap();
        let expected =
            parameters.ring(1).unwrap().matrix_type((1, parameters.component_columns(1).unwrap()));
        for (key, encoding) in public_outputs.iter().zip(&online_eval_outputs) {
            assert!(encoding.public_key.schema() == key.schema());
            for entry in [
                &encoding.row.0,
                &encoding.row.1,
                &encoding.public_key.matrix.0.0,
                &encoding.public_key.matrix.0.1,
                &encoding.public_key.matrix.1.0,
                &encoding.public_key.matrix.1.1,
            ] {
                assert_eq!(entry.matrix_type(), &expected);
            }
        }
        // Both LUTs read the helpers of level zero; a graph with both
        // outputs declares each helper input once.
        let mut context = mxx_dsl::DslContext::new("equal-luts-online-eval");
        for (index, encoding) in online_eval_outputs.iter().enumerate() {
            context = context.output(format!("row-{index}"), encoding.row.1.clone()).unwrap();
        }
        context.build().unwrap();
    }

    #[test]
    fn sampled_input_key_width_follows_payload_knowledge() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let program =
            Arc::new(builder.finish(vec![crate::program::Operand::Scalar(private)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            Some(RnsBggPayloadSecret {
                row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)),
                binding: None,
            }),
            ring.bytes_input("sampled-key-schema", 32),
            [0; 32],
        );
        let private_key = lowering.sample_input_public_key(private).unwrap();
        let public_key = lowering.sample_input_public_key(public).unwrap();
        assert_eq!(
            private_key.matrix.0.0.matrix_type(),
            &ring.matrix_type((
                1,
                parameters.key_component_columns(0, AttributeKnowledge::PrivatePayload).unwrap()
            ))
        );
        assert_eq!(
            public_key.matrix.0.0.matrix_type(),
            &ring.matrix_type((
                1,
                parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap()
            ))
        );
    }

    fn input_key_matrix(ring: &mxx_dsl::Ring, prefix: &str, columns: usize) -> RnsBggMatrix {
        let entry = |suffix: &str| ring.input(format!("{prefix}-{suffix}"), (1, columns));
        ((entry("00"), entry("01")), (entry("10"), entry("11")))
    }

    #[test]
    fn mul_public_aligns_public_lhs_to_private_rhs_level() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let private_rhs = builder
            .mul_private(
                crate::program::Operand::Scalar(private_input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let output = builder
            .mul_public(
                crate::program::Operand::Scalar(public_input),
                crate::program::Operand::Scalar(private_rhs),
            )
            .unwrap();
        // The public input is also an output, which keeps it at level zero,
        // so the product must align it to the private operand's level.
        let program = Arc::new(
            builder
                .finish(vec![
                    crate::program::Operand::Scalar(output),
                    crate::program::Operand::Scalar(public_input),
                ])
                .unwrap(),
        );
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        assert_eq!(workload.level(private_rhs).unwrap(), 1);
        assert_eq!(workload.level(public_input).unwrap(), 0);
        assert_eq!(
            parameters.mul_public_target_level(0, 1, AttributeKnowledge::PrivatePayload).unwrap(),
            1
        );
        let low_ring = parameters.ring(1).unwrap();
        let high_ring = parameters.ring(0).unwrap();
        let mut lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            Some(RnsBggPayloadSecret {
                row: (high_ring.uniform_interval((1, 1), -1, 1), high_ring.identity(1)),
                binding: None,
            }),
            high_ring.bytes_input("mul-public-private-level", 32),
            [0; 32],
        );
        lowering.current_site =
            Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() });
        let public_key = RnsBggPublicKey::new(
            public_input,
            input_key_matrix(
                &high_ring,
                "mul-public-public-lhs",
                parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap(),
            ),
        );
        let private_key = RnsBggPublicKey::new(
            private_rhs,
            input_key_matrix(
                &low_ring,
                "mul-public-private-rhs",
                parameters.key_component_columns(1, AttributeKnowledge::PrivatePayload).unwrap(),
            ),
        );
        let result = <PublicKeyLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowering,
            &public_key,
            &private_key,
        )
        .unwrap();
        assert_eq!(
            result.matrix.0.0.matrix_type(),
            &low_ring.matrix_type((
                1,
                parameters.key_component_columns(1, AttributeKnowledge::PrivatePayload).unwrap(),
            ))
        );
    }

    #[test]
    fn mul_public_numeric_relation_matches_independent_action_and_rejects_wrong_key() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public_rhs_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_output = builder
            .mul_public(
                crate::program::Operand::Scalar(public_input),
                crate::program::Operand::Scalar(private_input),
            )
            .unwrap();
        let public_output = builder
            .mul_public(
                crate::program::Operand::Scalar(public_input),
                crate::program::Operand::Scalar(public_rhs_input),
            )
            .unwrap();
        let program = Arc::new(
            builder
                .finish(vec![
                    crate::program::Operand::Scalar(private_output),
                    crate::program::Operand::Scalar(public_output),
                ])
                .unwrap(),
        );
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let rhs_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PrivatePayload).unwrap();
        let lhs_entry =
            ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(0) });
        let rhs_entry = ring
            .constant((1, rhs_columns), ConstantMatrix::UnitRow { index: IntExpr::constant(0) });
        let rhs_entry_alt = rhs_entry.clone() * IntExpr::constant(2);
        let public_rhs_entry =
            ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(1) });
        let lhs_matrix = ((lhs_entry.clone(), lhs_entry.clone()), (lhs_entry.clone(), lhs_entry));
        let rhs_matrix =
            ((rhs_entry.clone(), rhs_entry_alt.clone()), (rhs_entry_alt.clone(), rhs_entry));
        let public_rhs_matrix = (
            (public_rhs_entry.clone(), public_rhs_entry.clone()),
            (public_rhs_entry.clone(), public_rhs_entry),
        );
        let mut lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            Some(RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)), binding: None }),
            ring.bytes_input("mul-public-numeric-key", 32),
            [0; 32],
        );
        lowering.current_site =
            Some(LoweringSite { output: private_output, indices: Vec::new(), ranges: Vec::new() });
        let actual_private = <PublicKeyLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowering,
            &RnsBggPublicKey::new(public_input, lhs_matrix.clone()),
            &RnsBggPublicKey::new(private_input, rhs_matrix.clone()),
        )
        .unwrap();
        lowering.current_site =
            Some(LoweringSite { output: public_output, indices: Vec::new(), ranges: Vec::new() });
        let actual_public = <PublicKeyLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowering,
            &RnsBggPublicKey::new(public_input, lhs_matrix.clone()),
            &RnsBggPublicKey::new(public_rhs_input, public_rhs_matrix.clone()),
        )
        .unwrap();
        let base = 1u64 << level.base_bits();
        let columns = level.modulus_digits();
        let direct_decompose = |matrix: RnsBggMatrix| {
            (
                (matrix.0.0.decompose(base, columns), matrix.0.1.decompose(base, columns)),
                (matrix.1.0.decompose(base, columns), matrix.1.1.decompose(base, columns)),
            )
        };
        let direct_action = |lhs: RnsBggMatrix, rhs: RnsBggDigits| {
            (
                (
                    lhs.0.0.clone().mul_small_rhs(rhs.0.0.clone()) +
                        lhs.0.1.clone().mul_small_rhs(rhs.1.0.clone()),
                    lhs.0.0.mul_small_rhs(rhs.0.1.clone()) + lhs.0.1.mul_small_rhs(rhs.1.1.clone()),
                ),
                (
                    lhs.1.0.clone().mul_small_rhs(rhs.0.0) + lhs.1.1.clone().mul_small_rhs(rhs.1.0),
                    lhs.1.0.mul_small_rhs(rhs.0.1) + lhs.1.1.mul_small_rhs(rhs.1.1),
                ),
            )
        };
        let expected_private =
            direct_action(lhs_matrix.clone(), direct_decompose(rhs_matrix.clone()));
        let expected_public =
            direct_action(lhs_matrix.clone(), direct_decompose(public_rhs_matrix));
        let wrong_rhs = (
            (rhs_entry_alt.clone(), rhs_entry_alt.clone()),
            (rhs_entry_alt.clone(), rhs_entry_alt),
        );
        let wrong_key = direct_action(lhs_matrix, direct_decompose(wrong_rhs));
        let mut graph = DslContext::new("mul-public-numeric-relation");
        for (index, (actual_entry, expected_entry)) in matrix_entries(&actual_private.matrix)
            .into_iter()
            .zip(matrix_entries(&expected_private).into_iter())
            .enumerate()
        {
            graph = graph
                .output(format!("private-difference-{index}"), actual_entry - expected_entry)
                .unwrap();
        }
        for (index, (actual_entry, expected_entry)) in matrix_entries(&actual_public.matrix)
            .into_iter()
            .zip(matrix_entries(&expected_public).into_iter())
            .enumerate()
        {
            graph = graph
                .output(format!("public-difference-{index}"), actual_entry - expected_entry)
                .unwrap();
        }
        graph = graph.output("wrong-key", actual_private.matrix.0.0 - wrong_key.0.0).unwrap();
        let graph = graph.build().unwrap();
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend([level]));
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            StageGraph::new(graph, BTreeMap::new(), BTreeMap::<ProductionId, Manifest>::new()),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        for index in 0..4 {
            let matrix =
                crate::test_support::full(&result.outputs[&format!("private-difference-{index}")]);
            assert!(matrix.entry(0, 0).coeffs_biguints().iter().all(|value| value.is_zero()));
            let matrix =
                crate::test_support::full(&result.outputs[&format!("public-difference-{index}")]);
            assert!(matrix.entry(0, 0).coeffs_biguints().iter().all(|value| value.is_zero()));
        }
        let wrong = crate::test_support::full(&result.outputs["wrong-key"]);
        assert!(wrong.entry(0, 0).coeffs_biguints().iter().any(|value| !value.is_zero()));
    }

    #[test]
    fn mul_public_rejects_a_private_lhs_even_with_a_public_rhs() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_lhs_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let public_rhs_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let private_lhs = builder
            .mul_private(
                crate::program::Operand::Scalar(public_lhs_input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let result = builder.mul_public(
            crate::program::Operand::Scalar(private_lhs),
            crate::program::Operand::Scalar(public_rhs_input),
        );
        assert!(matches!(result, Err(ProgramValidationError::InvalidStructure(message))
            if message == "MulPublic requires a public exponent LHS"));
    }

    #[test]
    fn rhs_preparation_uses_next_level_payload_ciphertext_and_compact_schema() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let output = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let program =
            Arc::new(builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        assert_eq!(workload.rhs_level(rhs).unwrap(), 0);

        let ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret {
            row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)),
            binding: None,
        };
        let mut lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            Some(payload),
            ring.bytes_input("rhs-schema-key", 32),
            [0; 32],
        );
        lowering.rhs.insert(
            rhs,
            InputBinding::Scalar(ring.polynomial([IntExpr::constant(BigInt::from(3u8))])),
        );
        let compact =
            lowering.prepare_rhs_material(&RhsRef { id: rhs, index: None::<Int> }).unwrap();
        let expected_compact_schema =
            parameters.ring(1).unwrap().matrix_type((1, parameters.component_columns(1).unwrap()));
        for entry in matrix_entries(&compact) {
            assert_eq!(
                entry.matrix_type(),
                &expected_compact_schema,
                "prepared RHS is stored as a compact next-level matrix"
            );
        }
        let action = super::centered_rebase_compact(
            decompose_matrix(&parameters, 1, compact.clone()).unwrap(),
            &ring,
        );
        let expected_columns = parameters.level(1).unwrap().modulus_digits();
        let expected_action_schema =
            ring.matrix_type((expected_columns, parameters.component_columns(1).unwrap()));
        for entry in [&action.0.0, &action.0.1, &action.1.0, &action.1.1] {
            assert_eq!(
                entry.matrix_type(),
                &expected_action_schema,
                "RHS action uses the next-level signed decomposition, centered in the input ring"
            );
        }
        let graph = DslContext::new("rhs-next-level-payload-action")
            .transferred_output("rhs", action)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn rhs_family_members_reuse_the_next_level_action_schema() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low], 1, 1).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let products = builder
            .parallel(
                crate::program::FamilyRange::full(IntExpr::constant(2)).unwrap(),
                |builder, index| {
                    builder
                        .mul_private(
                            crate::program::Operand::Scalar(input),
                            crate::program::Operand::Member { family: rhs, index },
                        )
                        .map(crate::program::Operand::Scalar)
                },
            )
            .unwrap();
        let output = builder.sum(products).unwrap();
        let program =
            Arc::new(builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret {
            row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)),
            binding: None,
        };
        let mut lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            Some(payload),
            ring.bytes_input("rhs-family-schema-key", 32),
            [0; 32],
        );
        let constant =
            |value: u32| mxx_backends::poly::dcrt::poly::DCRTPoly::from_u32s(&high, &[value]);
        let plaintexts = super::super::plaintext_inputs(
            &ring,
            "rhs-family-schema-plaintext",
            &InputBinding::Family(vec![constant(2), constant(5)]),
            &mut BTreeMap::new(),
        )
        .unwrap();
        lowering.rhs.insert(rhs, plaintexts);
        let first = lowering
            .prepare_rhs_material(&RhsRef { id: rhs, index: Some(Int::constant(0usize)) })
            .unwrap();
        let second = lowering
            .prepare_rhs_material(&RhsRef { id: rhs, index: Some(Int::constant(1usize)) })
            .unwrap();
        assert_eq!(first.0.0.matrix_type(), second.0.0.matrix_type());
        assert_ne!(first.flatten(), second.flatten());
    }
}
