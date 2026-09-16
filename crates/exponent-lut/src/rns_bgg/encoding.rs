//! RNS-BGG secret sampling and online encoding operations.
use super::*;

impl<R: GraphValue> GraphValue for RnsBggEncoding<R> {
    type Schema = RnsBggEncoding<R::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.row.flatten()
    }

    fn schema(&self) -> Self::Schema {
        RnsBggEncoding {
            row: self.row.schema(),
            level: self.level,
            key_id: self.key_id,
            context: self.context,
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            row: R::from_values(&schema.row, values)?,
            level: schema.level,
            key_id: schema.key_id,
            context: schema.context,
        })
    }
}

impl<R: GraphValueSchema> GraphValueSchema for RnsBggEncoding<R> {
    type Value = RnsBggEncoding<R::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggEncoding {
            row: self.row.placeholders_from(next),
            level: self.level,
            key_id: self.key_id,
            context: self.context,
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.row.wire_types()
    }
}

impl RnsBggScheme {
    /// Online evaluation is exactly one block row product and one ModDown.
    pub fn evaluate_mul_private(
        &self,
        input: &RnsBggEncoding,
        prepared: &RnsBggPreparedPrivateMul,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        use mxx_primitives::poly::PolyParams;
        self.check_encoding(input, &prepared.input)?;
        let next = num_bigint::BigInt::from(
            self.parameters.level(prepared.output.level)?.modulus().as_ref().clone(),
        );
        Ok(RnsBggEncoding {
            row: (input.row.clone() * prepared.rhs.lifted_ciphertext.clone())
                .block_mod_switch(next),
            level: prepared.output.level,
            key_id: prepared.output.id,
            context: self.context,
        })
    }
    pub fn evaluate_mul_private_family(
        &self,
        input: &RnsBggEncodingFamily,
        prepared: &RnsBggPreparedPrivateMul<
            RnsBggPublicKeyFamily,
            ProgramValue<RnsBggMatrix, Family<RnsBggMatrix>>,
            ProgramValue<RnsBggDigits, Family<RnsBggDigits>>,
        >,
    ) -> Result<RnsBggEncodingFamily, RnsBggError> {
        self.check_encoding_family(input, &prepared.input)?;
        let rows = mxx_dsl::parallel(input.range.count().clone(), |index| {
            let element = RnsBggEncoding {
                row: input.rows.at(&index),
                level: input.level,
                key_id: input.key_id,
                context: input.context,
            };
            let prepared = RnsBggPreparedPrivateMul {
                input: RnsBggPublicKey {
                    matrix: prepared.input.matrices.at(&index),
                    level: prepared.input.level,
                    id: prepared.input.id,
                    context: prepared.input.context,
                },
                output: RnsBggPublicKey {
                    matrix: prepared.output.matrices.at(&index),
                    level: prepared.output.level,
                    id: prepared.output.id,
                    context: prepared.output.context,
                },
                rhs: RnsBggPrivateRhs {
                    ciphertext: prepared.rhs.ciphertext.at(&index),
                    lifted_ciphertext: prepared.rhs.lifted_ciphertext.at(&index),
                    level: prepared.rhs.level,
                    id: prepared.rhs.id,
                    rhs_id: prepared.rhs.rhs_id,
                    context: prepared.rhs.context,
                },
            };
            self.evaluate_mul_private(&element, &prepared)
                .map(|value| value.row)
                .map_err(|_| DslError::Schema)
        })
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        Ok(RnsBggEncodingFamily {
            rows: ProgramValue::Family(rows),
            level: prepared.output.level,
            key_id: prepared.output.id,
            context: self.context,
            range: input.range.clone(),
        })
    }
    /// Add preserves e_lhs + e_rhs; public multiplication preserves nu*e_lhs.
    /// Both operations retain the source level and its payload scale.
    pub fn evaluate_linear(
        &self,
        lhs: &RnsBggEncoding,
        rhs: &RnsBggEncoding,
        prepared: &RnsBggPreparedLinear,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        self.check_encoding(lhs, &prepared.lhs)?;
        self.check_encoding(rhs, &prepared.rhs)?;
        let row = match &prepared.operation {
            LinearOperation::Add => lhs.row.clone() + rhs.row.clone(),
            LinearOperation::MulPublic(multiplier) => multiplier.clone() * lhs.row.clone(),
        };
        Ok(RnsBggEncoding {
            row,
            level: prepared.output.level,
            key_id: prepared.output.id,
            context: self.context,
        })
    }
    pub(super) fn small_error(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Mat, RnsBggError> {
        let ring = self.ring(level)?;
        if self.encoding_errors_noiseless {
            return Ok(ring.zero((rows, columns)));
        }
        Ok(ring.gaussian(
            (rows, columns),
            self.parameters.encoding_error_sigma_expr(),
            self.parameters.encoding_error_bound.clone(),
        ))
    }
    pub fn sample_payload_secret(&self) -> Result<RnsBggPayloadSecret, RnsBggError> {
        let ring = self.ring(0)?;
        Ok(RnsBggPayloadSecret {
            row: RnsBggRow::new([[ring.uniform_interval((1, 1), -1, 1), ring.identity(1)]])
                .map_err(|e| RnsBggError::Graph(e.to_string()))?,
            context: self.context,
        })
    }
    pub fn sample_mask_secret(&self) -> Result<RnsBggMaskSecret, RnsBggError> {
        let ring = self.ring(0)?;
        Ok(RnsBggMaskSecret {
            row: RnsBggRow::new([[ring.uniform_interval((1, 1), -1, 1), ring.identity(1)]])
                .map_err(|e| RnsBggError::Graph(e.to_string()))?,
            context: self.context,
        })
    }
    #[cfg(test)]
    pub(crate) fn test_secrets(&self) -> Result<TestSecrets, RnsBggError> {
        Ok(TestSecrets { mask: self.sample_mask_secret()?, payload: self.sample_payload_secret()? })
    }
    pub(super) fn small_error_row(
        &self,
        level: usize,
        columns: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        RnsBggRow::new([[
            self.small_error(level, 1, columns)?,
            self.small_error(level, 1, columns)?,
        ]])
        .map_err(|e| RnsBggError::Graph(e.to_string()))
    }
    pub fn payload_at(
        &self,
        secret: &RnsBggPayloadSecret,
        level: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        if secret.context != self.context {
            return Err(RnsBggError::Context);
        }
        let modulus = BigInt::from(self.parameters.level(level)?.modulus().as_ref().clone());
        Ok(secret.row.clone().reduce_modulus(modulus))
    }
    pub fn mask_at(
        &self,
        secret: &RnsBggMaskSecret,
        level: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        if secret.context != self.context {
            return Err(RnsBggError::Context);
        }
        let modulus = BigInt::from(self.parameters.level(level)?.modulus().as_ref().clone());
        Ok(secret.row.clone().reduce_modulus(modulus))
    }
    /// Online input preparation needs only s and the preprocessing contribution.
    /// Fresh encoding error is sampled here exactly once with the canonical
    /// Gaussian cutoff derived from the configured sigma.
    pub fn encode(
        &self,
        secret: &RnsBggMaskSecret,
        key: &RnsBggPublicKey,
        material: &RnsBggInputMaterial,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        self.check_key(key)?;
        if material.context != self.context ||
            material.level != key.level ||
            material.key_id != key.id
        {
            return Err(RnsBggError::Context);
        }
        let s = self.mask_at(secret, key.level)?;
        let row = s * key.matrix.clone() +
            material.row.clone() +
            self.small_error_row(key.level, self.parameters.component_columns(key.level)?)?;
        Ok(RnsBggEncoding { row, level: key.level, key_id: key.id, context: self.context })
    }
    pub(crate) fn check_encoding(
        &self,
        value: &RnsBggEncoding,
        key: &RnsBggPublicKey,
    ) -> Result<(), RnsBggError> {
        self.check_key(key)?;
        if value.context != self.context ||
            value.level != key.level ||
            value.key_id != key.id ||
            value.row.matrix_type() !=
                &self
                    .ring(value.level)?
                    .matrix_type((1, 2 * self.parameters.component_columns(value.level)?))
        {
            return Err(RnsBggError::Context);
        }
        Ok(())
    }
    pub fn prepare_online_masks(
        &self,
        secret: &RnsBggMaskSecret,
        bank: &RnsBggMaskBank,
    ) -> Result<RnsBggOnlineMasks, RnsBggError> {
        if bank.context != self.context {
            return Err(RnsBggError::Context);
        }
        let level = bank.level;
        let s = self.mask_at(secret, level)?;
        let p = self.scalar(level, self.parameters.scale(level)?.clone())?;
        let gadget = self.state_gadget(level)?;
        let columns = self.parameters.component_columns(level)?;
        let public =
            Family::pack(bank.branches.iter().map(|branch| branch.public.clone()).collect())
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let rows = mxx_dsl::parallel(bank.branches.len(), |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: bank
                    .branches
                    .iter()
                    .map(|branch| IntExpr::constant(branch.sigma))
                    .collect(),
            };
            Ok(s.clone() * public.at(&index) -
                (p.clone() * s.clone().ring_automorphism(sigma))
                    .map(|entry| entry * gadget.clone()) +
                self.small_error_row(level, columns).map_err(|_| DslError::Schema)?)
        })
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let rows = bank
            .branches
            .iter()
            .enumerate()
            .map(|(index, branch)| (branch.sigma, rows.at(index)))
            .collect();
        Ok(RnsBggOnlineMasks { id: bank.id, level, context: self.context, rows: Arc::new(rows) })
    }
    pub fn evaluate_lut(
        &self,
        input: &RnsBggEncoding,
        prepared: &RnsBggPreparedLut,
        masks: &RnsBggOnlineMasks,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        let raw = self.evaluate_lut_raw(
            input,
            &prepared.input,
            &prepared.branches,
            prepared.binding.mask_bank,
            masks,
        )?;
        let next = BigInt::from(self.parameters.level(input.level + 1)?.modulus().as_ref().clone());
        Ok(RnsBggEncoding {
            row: raw.block_mod_switch(next),
            level: prepared.output.level,
            key_id: prepared.output.id,
            context: self.context,
        })
    }
    /// z remains at the input base so terminal outputs can share one rounding.
    pub(super) fn evaluate_lut_raw(
        &self,
        input: &RnsBggEncoding,
        expected: &RnsBggPublicKey,
        branches: &[PreparedBranch],
        bank_id: [u8; 32],
        masks: &RnsBggOnlineMasks,
    ) -> Result<RnsBggRow, RnsBggError> {
        self.check_encoding(input, expected)?;
        if masks.id != bank_id || masks.context != self.context || masks.level != input.level {
            return Err(RnsBggError::Context);
        }
        let inputs = branches
            .iter()
            .map(|branch| {
                Ok((
                    masks.rows.get(&branch.sigma).ok_or(RnsBggError::Context)?.clone(),
                    branch.lifted_intermediate.clone(),
                    branch.lifted_ciphertext.clone(),
                ))
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        let inputs = Family::pack(inputs).map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let terms = mxx_dsl::parallel(branches.len(), |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: branches.iter().map(|branch| IntExpr::constant(branch.sigma)).collect(),
            };
            let (mask, intermediate, ciphertext) = inputs.at(&index);
            Ok(mask * intermediate + input.row.clone().ring_automorphism(sigma) * ciphertext)
        })
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        sum(terms)
    }

    pub(super) fn check_encoding_family(
        &self,
        value: &RnsBggEncodingFamily,
        key: &RnsBggPublicKeyFamily,
    ) -> Result<(), RnsBggError> {
        self.check_public_family(key)?;
        if value.context != key.context ||
            value.level != key.level ||
            value.key_id != key.id ||
            value.range != key.range ||
            matches!(&value.rows, ProgramValue::Family(values) if values.count() != key.range.count()) ||
            value.rows.element_schema().matrix_type() !=
                self.ring(value.level)?
                    .matrix_type((1, 2 * self.parameters.component_columns(value.level)?))
        {
            return Err(RnsBggError::Context);
        }
        Ok(())
    }
    pub fn evaluate_lut_family(
        &self,
        input: &RnsBggEncodingFamily,
        prepared: &RnsBggPreparedLutFamily,
        masks: &RnsBggOnlineMasks,
    ) -> Result<RnsBggEncodingFamily, RnsBggError> {
        self.check_encoding_family(input, &prepared.input)?;
        let rows = mxx_dsl::parallel(input.range.count().clone(), |index| {
            let scalar = RnsBggEncoding {
                row: input.rows.at(&index),
                level: input.level,
                key_id: input.key_id,
                context: input.context,
            };
            self.evaluate_lut(&scalar, &prepared.prepared.at(&index), masks)
                .map(|value| value.row)
                .map_err(|_| DslError::Schema)
        })?;
        Ok(RnsBggEncodingFamily {
            rows: ProgramValue::Family(rows),
            level: prepared.output.level,
            key_id: prepared.output.id,
            context: self.context,
            range: input.range.clone(),
        })
    }
    pub fn evaluate_linear_family(
        &self,
        lhs: &RnsBggEncodingFamily,
        rhs: &RnsBggEncodingFamily,
        prepared: &RnsBggPreparedLinear<RnsBggPublicKeyFamily, ProgramValue<Mat, Family<Mat>>>,
    ) -> Result<RnsBggEncodingFamily, RnsBggError> {
        self.check_encoding_family(lhs, &prepared.lhs)?;
        self.check_encoding_family(rhs, &prepared.rhs)?;
        let rows = mxx_dsl::parallel(lhs.range.count().clone(), |index| {
            let left = lhs.rows.at(&index);
            Ok(match &prepared.operation {
                LinearOperation::MulPublic(multiplier) => multiplier.at(&index) * left,
                LinearOperation::Add => left + rhs.rows.at(&index),
            })
        })
        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggEncodingFamily {
            rows: ProgramValue::Family(rows),
            level: lhs.level,
            key_id: prepared.output.id,
            context: self.context,
            range: lhs.range.clone(),
        })
    }
    pub fn evaluate_reduce_family(
        &self,
        value: &RnsBggEncodingFamily,
        prepared: &RnsBggPreparedReduction,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        self.check_encoding_family(value, &prepared.input)?;
        let row = match &value.rows {
            ProgramValue::Scalar(row) => {
                self.ring(value.level)?.polynomial([value.range.count().clone()]) * row.clone()
            }
            ProgramValue::Family(rows) => sum(rows.clone())?,
        };
        Ok(RnsBggEncoding {
            row,
            level: value.level,
            key_id: prepared.output.id,
            context: self.context,
        })
    }
    pub fn evaluate_terminal<P>(
        &self,
        input: &RnsBggEncoding,
        prepared: &RnsBggPreparedTerminal<RnsBggPublicKey, RnsBggMatrix, P>,
        masks: &RnsBggOnlineMasks,
    ) -> Result<RnsBggTerminalAccumulator<RnsBggRow>, RnsBggError> {
        Ok(RnsBggTerminalAccumulator {
            value: self.evaluate_lut_raw(
                input,
                &prepared.input,
                &prepared.branches,
                prepared.binding.mask_bank,
                masks,
            )?,
            level: input.level,
            context: self.context,
            stages: vec![prepared.id],
        })
    }
    pub fn finalize_terminal_encoding(
        &self,
        raw: RnsBggTerminalAccumulator<RnsBggRow>,
        public: &RnsBggTerminalPublicKey,
    ) -> Result<RnsBggTerminalEncoding, RnsBggError> {
        // These comparisons run while constructing the graph, not on the device.
        if raw.context != self.context ||
            public.context != self.context ||
            public.level != raw.level + 1 ||
            raw.stages != public.stages
        {
            return Err(RnsBggError::Context);
        }
        let next = BigInt::from(self.parameters.level(public.level)?.modulus().as_ref().clone());
        Ok(RnsBggTerminalEncoding { row: raw.value.block_mod_switch(next), level: public.level })
    }
    /// Shared public ciphertext lifts and online masks are constructed outside
    /// the indexed body. Aggregate mode reduces raw rows before finalization.
    pub fn evaluate_terminal_family(
        &self,
        input: &RnsBggEncodingFamily,
        prepared: &RnsBggPreparedTerminalFamily,
        masks: &RnsBggOnlineMasks,
    ) -> Result<Family<RnsBggRow>, RnsBggError> {
        self.check_encoding_family(input, &prepared.input)?;
        let next = BigInt::from(self.parameters.level(input.level + 1)?.modulus().as_ref().clone());
        let input = input.clone().into_program()?;
        let raw = crate::readout::evaluate_terminal_value(
            self,
            &input,
            &ProgramValue::Family(prepared.cache.clone()),
            masks,
            prepared.mode == RnsBggTerminalFamilyMode::Sum,
            |raw| Ok(if prepared.finalize { raw.value.block_mod_switch(next) } else { raw.value }),
        )
        .map_err(|error| match error {
            crate::readout::TerminalError::Scheme(error) => error,
            error => RnsBggError::Graph(error.to_string()),
        })?;
        match raw {
            ProgramValue::Scalar(value) => Family::pack(vec![value]).map_err(RnsBggError::from),
            ProgramValue::Family(values) => Ok(values),
        }
    }

    pub fn sample_private_rhs(
        &self,
        secret: &RnsBggPayloadSecret,
        level: usize,
        rhs: RnsBggRhsPlaintext,
        hash_key: mxx_dsl::Bytes,
        tag: impl Into<mxx_dsl::HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggPrivateRhs, RnsBggError> {
        let (prefix, target) = self.private_rhs_basis(secret, level)?;
        let ciphertext =
            self.encrypt_private_rhs(level, rhs.payload, &prefix, &target, hash_key, tag)?;
        Ok(RnsBggPrivateRhs {
            ciphertext,
            lifted_ciphertext: (),
            level,
            context: self.context,
            rhs_id: rhs.id,
            id,
        })
    }
    // These values are independent of the selector. Capture them once in a
    // generated family instead of rebuilding the gadget and secret per member.
    fn private_rhs_basis(
        &self,
        secret: &RnsBggPayloadSecret,
        level: usize,
    ) -> Result<(Mat, RnsBggRow), RnsBggError> {
        let next = level + 1;
        let r = self.payload_at(secret, next)?;
        let prefix = -r.entry(0, 0).clone();
        let scaled = self.scalar(next, self.parameters.scale(next)?.clone())? * r;
        let gadget = self.state_gadget(next)?;
        Ok((prefix, scaled.map(|entry| entry * gadget.clone())))
    }
    fn encrypt_private_rhs(
        &self,
        level: usize,
        payload: Mat,
        prefix: &Mat,
        target: &RnsBggRow,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let next = level + 1;
        let ring = self.ring(next)?;
        let columns = self.parameters.component_columns(next)?;
        let mut tag = tag.into();
        tag.push("rns_bgg-private-rhs");
        tag.push(mxx_ir_core::IntExpr::constant(level));
        let top = RnsBggRow::hash(&ring, hash_key, tag, columns);
        // Ring scalars commute: payload * (scale * r * gadget) is the
        // original target, with fresh independent errors for every member.
        let error = self.small_error_row(next, columns)?;
        let bottom = RnsBggRow::new([std::array::from_fn(|column| {
            Mat::multi_row_gemm_accumulate(
                vec![
                    (1usize, prefix.clone(), top.entry(0, column).clone()),
                    (1usize, payload.clone(), target.entry(0, column).clone()),
                ],
                Some(error.entry(0, column).clone()),
            )
        })])?;
        Ok(RnsBggMatrix::from_rows(top, bottom))
    }
    /// One sampled ciphertext per member, with a single structural graph body.
    pub fn sample_private_rhs_family(
        &self,
        secret: &RnsBggPayloadSecret,
        level: usize,
        rhs: &RnsBggRhsPlaintextFamily,
        hash_key: mxx_dsl::Bytes,
        tag: impl Into<mxx_dsl::HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggPrivateRhs<RnsBggRhsFamily>, RnsBggError> {
        let (prefix, target) = self.private_rhs_basis(secret, level)?;
        let mut root = tag.into();
        root.push("rns_bgg-private-rhs-family");
        root.push(mxx_ir_core::IntExpr::constant(level));
        let body = mxx_dsl::Subgraph::define(
            format!("rns-bgg-selector-{id:02x?}"),
            mxx_dsl::Int::constant(0).schema(),
            |index: Int| {
                let mut tag = root.clone();
                tag.push(index.clone());
                self.encrypt_private_rhs(
                    level,
                    rhs.payloads.at(&index),
                    &prefix,
                    &target,
                    hash_key.clone(),
                    tag,
                )
                .map_err(|_| DslError::Schema)
            },
        )
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        Ok(RnsBggPrivateRhs {
            ciphertext: RnsBggRhsFamily::Generated { body, count: rhs.payloads.count().clone() },
            lifted_ciphertext: (),
            rhs_id: rhs.id,
            level,
            context: self.context,
            id,
        })
    }
    /// `id` is a setup artifact identity, unique to this generated helper
    /// material (including its mask bank). Persist it with the material and
    /// reuse it on import; equal table/RHS identities alone are insufficient.
    /// This caller contract avoids hashing ciphertext coefficients online.
    pub fn sample_lut_material(
        &self,
        secret: &RnsBggPayloadSecret,
        table: Option<&LutTable>,
        rhs: Option<RnsBggRhsPlaintext>,
        bank: &RnsBggMaskBank,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggLutMaterial, RnsBggError> {
        if bank.context != self.context || secret.context != self.context {
            return Err(RnsBggError::Context);
        }
        let level = bank.level;
        let next = level + 1;
        let ring = self.ring(next)?;
        if table.is_some_and(|table| table.rhs_width().is_some() || rhs.is_some()) {
            return Err(RnsBggError::Lut);
        }
        if rhs.as_ref().is_some_and(|rhs| rhs.payload.matrix_type() != &ring.matrix_type((1, 1))) {
            return Err(RnsBggError::Context);
        }
        let width = table.map_or(1, |table| table.values().len());
        let layout = ExpLayout { stride_log2: 0, width_log2: width.trailing_zeros() };
        let sigmas = layout
            .automorphism_indices(self.parameters.level(next)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let r = self.payload_at(secret, next)?;
        let prefix = r.entry(0, 0).clone();
        let root = tag.into();
        let mut branches = Vec::with_capacity(width);
        for sigma in sigmas {
            let mask = bank
                .branches
                .iter()
                .find(|branch| branch.sigma == sigma)
                .ok_or(RnsBggError::Lut)?;
            let coefficient = match table {
                Some(table) => crate::utils::lut_coefficient(&ring, &layout, sigma, table.values())
                    .map_err(|_| RnsBggError::Lut)?,
                None => ring.identity(1),
            };
            let multiplier = rhs.as_ref().map_or_else(
                || ring.identity(1),
                |rhs| rhs.payload.clone().ring_automorphism(sigma),
            );
            let target = self.scalar(next, self.parameters.scale(next)?.clone())? *
                coefficient *
                multiplier *
                r.clone();
            let gadget = self.state_gadget(next)?;
            let target = target.map(|entry| entry * gadget.clone());
            let mut tag = root.clone();
            tag.push("rns_bgg-rhs");
            tag.push(mxx_ir_core::IntExpr::constant(level));
            tag.push(mxx_ir_core::IntExpr::constant(sigma));
            let columns = target.entry(0, 0).matrix_type().columns.clone();
            let columns = usize::try_from(
                columns.evaluate(&ParamEnv::default()).expect("concrete target width"),
            )
            .expect("target width fits usize");
            let top = RnsBggRow::hash(&ring, hash_key.clone(), tag, columns);
            let last = -prefix.clone().ring_automorphism(sigma) * top.clone() +
                target +
                self.small_error_row(next, columns)?;
            branches.push((mask.clone(), RnsBggMatrix::from_rows(top, last)));
        }
        Ok(RnsBggLutMaterial {
            mask_bank: bank.id,
            id,
            level,
            context: self.context,
            table: table.map(LutTable::commitment),
            rhs_id: rhs.as_ref().map(|rhs| rhs.id),
            branches,
        })
    }
    /// A refresh preserves the native monomial and changes its terminal scale.
    /// The payload secret enters fresh targets only during preprocessing.
    /// Encode a CRT channel's data target with spacing H. The resumable state
    /// scale p_ref and gadget are supplied by this scheme's output level.
    pub fn sample_refresh_data_material(
        &self,
        secret: &RnsBggPayloadSecret,
        layout: crate::layout::ExpLayout,
        scale: &num_bigint::BigUint,
        bank: &RnsBggMaskBank,
        hash_key: mxx_dsl::Bytes,
        tag: impl Into<mxx_dsl::HashTag>,
        id: [u8; 32],
    ) -> Result<ProgramValue<RnsBggTerminalMaterial, Family<RnsBggTerminalMaterial>>, RnsBggError>
    {
        let level = bank.level + 1;
        let ring = self.ring(level)?;
        // For the identity row table, sum_i sigma(X^(-s*i))*X^(s*i)
        // is width in the identity branch and zero in every other branch.
        // Keep every fresh ciphertext/error; only avoid the dense O(N^2) table.
        debug_assert_eq!(layout.width(), bank.branches.len());
        let columns = self.parameters.component_columns(level)?;
        let gadget = self.state_gadget(level)?;
        let payload = self.payload_at(secret, level)?.map(|entry| entry * gadget.clone());
        let transform = RnsBggMatrix::from_rows(
            self.scalar(level, -BigInt::from(scale * self.parameters.scale(level)?))? * payload,
            RnsBggRow::new([[ring.zero((1, columns)), ring.zero((1, columns))]])?,
        );
        self.sample_terminal_material(secret, None, None, Some(&transform), bank, hash_key, tag, id)
            .map(ProgramValue::Scalar)
    }
    /// `id` identifies the complete generated family and its mask bank.
    /// Preserve it across artifact storage/import; allocate a new identity
    /// when regenerating the material, even for the same table and RHS.
    pub fn sample_lut_material_family(
        &self,
        secret: &RnsBggPayloadSecret,
        table: Option<&LutTable>,
        rhs: Option<&RnsBggRhsPlaintextFamily>,
        bank: &RnsBggMaskBank,
        count: IntExpr,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggLutMaterialFamily, RnsBggError> {
        if table.is_some_and(|table| table.rhs_width().is_some() || rhs.is_some()) {
            return Err(RnsBggError::Lut);
        }
        let rhs_type = self.ring(bank.level + 1)?.matrix_type((1, 1));
        if rhs.is_some_and(|rhs| {
            rhs.payloads.count() != &count || rhs.payloads.element_type() != &rhs_type
        }) {
            return Err(RnsBggError::Context);
        }
        let width = table.map_or(1, |table| table.values().len());
        let layout =
            crate::layout::ExpLayout { stride_log2: 0, width_log2: width.trailing_zeros() };
        let masks = layout
            .automorphism_indices(self.parameters.level(bank.level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?
            .into_iter()
            .map(|sigma| {
                bank.branches
                    .iter()
                    .find(|mask| mask.sigma == sigma)
                    .cloned()
                    .ok_or(RnsBggError::Lut)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let root = tag.into();
        let ciphertexts = mxx_dsl::parallel(count, |index| {
            let mut tag = root;
            tag.push(index.clone());
            let material = self
                .sample_lut_material(
                    secret,
                    table,
                    rhs.map(|rhs| RnsBggRhsPlaintext {
                        payload: rhs.payloads.at(&index),
                        id: rhs.id,
                    }),
                    bank,
                    hash_key,
                    tag,
                    id,
                )
                .map_err(|_| DslError::Schema)?;
            Ok(RnsBggBranches(
                material.branches.into_iter().map(|(_, ciphertext)| ciphertext).collect(),
            ))
        })
        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggLutMaterialFamily {
            binding: RnsBggLutBinding {
                mask_bank: bank.id,
                material_id: id,
                level: bank.level,
                context: self.context,
                table: table.map(LutTable::commitment),
                rhs: rhs.map(|rhs| rhs.id),
            },
            ciphertexts,
            masks,
        })
    }
    /// Preprocess a fresh terminal target without the online secret s.
    /// Inputs must be one signed monomial in the declared native domain.
    /// Each member covers its complete native subring domain.
    pub fn sample_readout_material(
        &self,
        secret: &RnsBggPayloadSecret,
        readout: &ValueReadout,
        bank: &RnsBggMaskBank,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        id: [u8; 32],
        weight: Option<RnsBggReadoutWeights>,
        mode: Option<RnsBggTerminalFamilyMode>,
    ) -> Result<RnsBggReadoutMaterial, RnsBggError> {
        let n = self.parameters.level(bank.level)?.ring_dimension() as usize;
        if readout.layout.width() != n / readout.layout.stride() {
            return Err(RnsBggError::Parameters("native readout requires full subring width"));
        }
        readout.validate(&self.ring(bank.level)?).map_err(|_| RnsBggError::Lut)?;
        let weight =
            weight.unwrap_or_else(|| RnsBggReadoutWeights::Shared([vec![1u8.into()], vec![]]));
        let base_weight = match &weight {
            RnsBggReadoutWeights::Shared(value) => value.clone(),
            RnsBggReadoutWeights::Indexed { .. } => [vec![1u8.into()], vec![]],
        };
        let modulus =
            BigInt::from(self.parameters.level(bank.level + 1)?.modulus().as_ref().clone());
        let (table, digit_bound) = readout
            .weighted_centered_table(&base_weight, &modulus)
            .map_err(|_| RnsBggError::Lut)?;
        let target_bound = match &weight {
            RnsBggReadoutWeights::Shared(weight) => {
                weight.iter().flatten().map(|v| v.magnitude()).max().cloned().unwrap_or_default() *
                    &digit_bound
            }
            RnsBggReadoutWeights::Indexed { coefficient_bound, .. } => {
                &digit_bound * coefficient_bound
            }
        };
        let terminal = match &weight {
            RnsBggReadoutWeights::Shared(_) => {
                ProgramValue::Scalar(self.sample_terminal_material(
                    secret,
                    Some(&table),
                    None,
                    None,
                    bank,
                    hash_key,
                    tag,
                    id,
                )?)
            }
            RnsBggReadoutWeights::Indexed { rows, .. } => {
                let ring = self.ring(bank.level + 1)?;
                let transforms = mxx_dsl::parallel(rows.count().clone(), |index| {
                    let row = rows.at(index);
                    let columns = row.entry(0, 0).matrix_type().columns.clone();
                    Ok(RnsBggMatrix::from_rows(
                        row,
                        RnsBggRow::new([[
                            ring.zero((1, columns.clone())),
                            ring.zero((1, columns)),
                        ]])?,
                    ))
                })
                .map_err(|e| RnsBggError::Graph(e.to_string()))?;
                self.sample_terminal_material_family(
                    secret,
                    &table,
                    &transforms,
                    bank,
                    hash_key,
                    tag,
                    id,
                )?
            }
        };
        Ok(RnsBggReadoutMaterial {
            mode: mode.unwrap_or(RnsBggTerminalFamilyMode::Independent),
            identity: readout.identity(),
            weight,
            target_bound,
            terminal,
        })
    }
    /// `table=None` is the identity row [X^v, 0] on the mask bank domain.
    /// Fresh target is -D_k (or -sigma_k(rhs)*D_k). In particular, neither
    /// p_next nor the payload secret r multiplies the row table (PDF 10.1).
    /// The caller guarantees the combined exponent is in the table domain;
    /// encrypted zero selectors produce a zero row but retain their errors.
    /// `output_transform` routes row targets at level+1; it can depend on a
    /// structural family index. The caller binds it in `id` and supplies the
    /// transformed target bound to noise planning. It never scales fresh error.
    pub fn sample_terminal_material(
        &self,
        secret: &RnsBggPayloadSecret,
        table: Option<&RowLutTable<2>>,
        rhs: Option<RnsBggRhsPlaintext>,
        output_transform: Option<&RnsBggMatrix>,
        bank: &RnsBggMaskBank,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggTerminalMaterial, RnsBggError> {
        if bank.context != self.context {
            return Err(RnsBggError::Context);
        }
        let next = bank.level + 1;
        let ring = self.ring(next)?;
        let n = self.parameters.level(next)?.ring_dimension() as usize;
        if table.is_some_and(|table| table.values.iter().flatten().any(|p| p.len() > n)) {
            return Err(RnsBggError::Lut);
        }
        if rhs.as_ref().is_some_and(|rhs| rhs.payload.matrix_type() != &ring.matrix_type((1, 1))) {
            return Err(RnsBggError::Context);
        }
        let sigmas = match table {
            Some(table) => table.layout.automorphism_indices(n).map_err(|_| RnsBggError::Lut)?,
            None => bank.branches.iter().map(|branch| branch.sigma).collect(),
        };
        let masks =
            bank.branches.iter().map(|branch| (branch.sigma, branch)).collect::<BTreeMap<_, _>>();
        let r = self.payload_at(secret, next)?;
        let modulus = BigInt::from(self.parameters.level(next)?.modulus().as_ref().clone());
        let inverse =
            modular_inverse(&BigInt::from(sigmas.len()), &modulus).ok_or(RnsBggError::Lut)?;
        // Scalar digit rows are a single polynomial evaluated at X^(-sigma).
        // Compile it once instead of emitting width^2 monomial products.
        let scalar_rows = table
            .filter(|table| table.values.iter().flatten().all(|p| p.len() <= 1))
            .map(|table| {
                RnsBggRow::new([std::array::from_fn(|column| {
                    let mut coefficients =
                        vec![BigInt::from(0u8); table.layout.stride() * (table.width() - 1) + 1];
                    for (index, row) in table.values.iter().enumerate() {
                        if let Some(value) = row[column].first() {
                            coefficients[index * table.layout.stride()] = value.clone();
                        }
                    }
                    ring.polynomial(coefficients.into_iter().map(IntExpr::constant))
                })])
            })
            .transpose()?;
        let values = table
            .filter(|_| scalar_rows.is_none())
            .into_iter()
            .flat_map(|table| table.values.iter())
            .map(|row| {
                RnsBggRow::new([row.each_ref().map(|p| {
                    if p.is_empty() {
                        ring.zero((1, 1))
                    } else {
                        ring.polynomial(p.iter().cloned().map(IntExpr::constant))
                    }
                })])
                .map_err(|e| RnsBggError::Graph(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let normalized_rows = table
            .filter(|_| scalar_rows.is_none())
            .map(|table| {
                sigmas
                    .iter()
                    .map(|sigma| {
                        let sum = table
                            .sieve_sum(&ring, *sigma, &values)
                            .map_err(|_| RnsBggError::Lut)?;
                        Ok(self.scalar(next, inverse.clone())? * sum)
                    })
                    .collect::<Result<Vec<_>, RnsBggError>>()
                    .and_then(|rows| {
                        Family::pack(rows).map_err(|error| RnsBggError::Graph(error.to_string()))
                    })
            })
            .transpose()?;
        let root = tag.into();
        let mut branches = Vec::with_capacity(sigmas.len());
        let mut lifted_ciphertexts = Vec::with_capacity(sigmas.len());
        let batch = crate::utils::preprocess_batch_size()?;
        for start in (0..sigmas.len()).step_by(batch) {
            let len = batch.min(sigmas.len() - start);
            let sampled = mxx_dsl::parallel(len, |local| {
                let index = local.add(Int::constant(start));
                let sigma = IntExpr::Select {
                    selector: Box::new(index.expression()?),
                    branches: sigmas.iter().copied().map(IntExpr::constant).collect(),
                };
                let normalized = if let Some(rows) = &scalar_rows {
                    ring.polynomial([IntExpr::constant(inverse.clone())]) *
                        rows.clone().ring_automorphism(IntExpr::constant(2 * n) - sigma.clone())
                } else if let Some(rows) = &normalized_rows {
                    rows.at(&index)
                } else {
                    let identity = IntExpr::Select {
                        selector: Box::new(index.expression()?),
                        branches: sigmas
                            .iter()
                            .map(|sigma| IntExpr::constant(u8::from(*sigma == 1)))
                            .collect(),
                    };
                    RnsBggRow::new([[ring.polynomial([identity]), ring.zero((1, 1))]])?
                };
                let multiplier = rhs.as_ref().map_or_else(
                    || ring.identity(1),
                    |rhs| rhs.payload.clone().ring_automorphism(sigma.clone()),
                );
                let target = -multiplier * normalized;
                // Routing multiplies plaintext before independent fresh branch error.
                let target = match output_transform {
                    Some(transform) => target * transform.clone(),
                    None => target,
                };
                let mut tag = root.clone();
                tag.push("rns_bgg-terminal-rhs");
                tag.push(IntExpr::constant(bank.level));
                tag.push(sigma.clone());
                let columns = usize::try_from(
                    target
                        .entry(0, 0)
                        .matrix_type()
                        .columns
                        .evaluate(&ParamEnv::default())
                        .expect("concrete target width"),
                )
                .expect("target width fits usize");
                let top = RnsBggRow::hash(&ring, hash_key.clone(), tag, columns);
                let last = -r.entry(0, 0).clone().ring_automorphism(sigma) * top.clone() +
                    target +
                    self.small_error_row(next, columns).map_err(|_| DslError::Schema)?;
                let ciphertext = RnsBggMatrix::from_rows(top, last);
                let lifted = self
                    .prepare_digits(bank.level, ciphertext.clone())
                    .map_err(|_| DslError::Schema)?;
                Ok((ciphertext, lifted))
            })?;
            let ciphertexts = sampled.field(|(ciphertext, _)| ciphertext)?;
            let digits = sampled.field(|(_, digits)| digits)?;
            for local in 0..len {
                let sigma = sigmas[start + local];
                let mask = *masks.get(&sigma).ok_or(RnsBggError::Lut)?;
                branches.push((mask.clone(), ciphertexts.at(local)));
                lifted_ciphertexts.push(digits.at(local));
            }
        }
        Ok(RnsBggTerminalMaterial {
            lifted_ciphertexts,
            material: RnsBggLutMaterial {
                mask_bank: bank.id,
                id,
                level: bank.level,
                context: self.context,
                table: table.map(|table| table.id),
                rhs_id: rhs.map(|rhs| rhs.id),
                branches,
            },
        })
    }
    /// Fresh per-member row targets. Transforms are public level+1 matrices;
    /// their family order and contents are bound by the caller's material ID.
    /// Payload-secret-dependent ciphertexts are generated during preprocessing.
    pub fn sample_terminal_material_family(
        &self,
        secret: &crate::rns_bgg::RnsBggPayloadSecret,
        table: &RowLutTable<2>,
        transforms: &Family<RnsBggMatrix>,
        bank: &RnsBggMaskBank,
        hash_key: mxx_dsl::Bytes,
        tag: impl Into<mxx_dsl::HashTag>,
        id: [u8; 32],
    ) -> Result<ProgramValue<RnsBggTerminalMaterial, Family<RnsBggTerminalMaterial>>, RnsBggError>
    {
        let root = tag.into();
        let materials = mxx_dsl::parallel(transforms.count().clone(), |index| {
            let transform = transforms.at(&index);
            let mut tag = root;
            tag.push(index);
            self.sample_terminal_material(
                secret,
                Some(table),
                None,
                Some(&transform),
                bank,
                hash_key,
                tag,
                id,
            )
            .map_err(|_| DslError::Schema)
        })
        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(ProgramValue::Family(materials))
    }
}

impl crate::scheme::PreparedProgram<RnsBggScheme> {
    /// Sample s-dependent rows once per bank, outside every indexed gate
    /// body. Only public masks live in this cache, including after import.
    pub fn prepare_online_helpers(
        &self,
        secret: &crate::rns_bgg::RnsBggMaskSecret,
    ) -> Result<
        BTreeMap<
            ProgramGateInstance,
            ProgramValue<RnsBggOnlineLutHelpers, (Family<RnsBggOnlineLutHelpers>, FamilyRange)>,
        >,
        RnsBggError,
    > {
        let mut banks = BTreeMap::<
            [u8; 32],
            (usize, [u8; 32], BTreeMap<usize, crate::rns_bgg::MaskBranch>),
        >::new();
        let mut bindings = BTreeMap::new();
        // The public mask fields are captures in the cache family, so selecting
        // a member preserves their original handles; no mask is sampled per member.
        let family_caches =
            self.family_caches.values().map(|cache| cache.at(0)).collect::<Vec<_>>();
        for (gate, value) in [&self.cache, &self.shared_cache]
            .into_iter()
            .chain(family_caches.iter())
            .flat_map(|cache| cache.gates.iter())
        {
            let (bank, binding) = match value {
                PreparedGate::Lut(prepared) => (prepared.mask_bank(), &prepared.binding),
                _ => continue,
            };
            let target =
                banks.entry(bank.id).or_insert_with(|| (bank.level, bank.context, BTreeMap::new()));
            if target.0 != bank.level || target.1 != bank.context {
                return Err(RnsBggError::Context);
            }
            // Gate tables can use different subsets of one setup bank.
            for branch in bank.branches {
                target.2.entry(branch.sigma).or_insert(branch);
            }
            bindings.insert(*gate, binding.clone());
        }
        let masks = banks
            .into_iter()
            .map(|(id, (level, context, branches))| {
                let bank = crate::rns_bgg::RnsBggMaskBank {
                    id,
                    level,
                    context,
                    branches: branches.into_values().collect(),
                };
                Ok((id, self.scheme.prepare_online_masks(secret, &bank)?))
            })
            .collect::<Result<BTreeMap<_, _>, RnsBggError>>()?;
        Ok(bindings
            .into_iter()
            .map(|(gate, binding)| {
                let helpers = masks[&binding.mask_bank].clone();
                (gate, ProgramValue::Scalar(RnsBggOnlineLutHelpers { binding, masks: helpers }))
            })
            .collect())
    }
}

impl ExponentProgramLowering for RnsBggEncodingEvaluator<'_> {
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        _: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        let Some(PreparedGate::MulPrivate(prepared)) = self.cache.gates.get(&gate) else {
            return Err(RnsBggError::Context);
        };
        if prepared.rhs.rhs_id != rhs.id() {
            return Err(RnsBggError::Context);
        }
        Ok((self.scheme.evaluate_mul_private(&input.0, prepared)?, None))
    }
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        self.evaluate_linear(private, public, gate)
    }
    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        self.evaluate_linear(lhs, rhs, gate)
    }
    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: LutId,
        input: &Self::Wire,
        helpers: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        let Some(PreparedGate::Lut(prepared)) = self.cache.gates.get(&gate) else {
            return Err(RnsBggError::Context);
        };
        let binding = &prepared.binding;
        let helpers = helpers.ok_or(RnsBggError::Context)?;
        if helpers.binding != *binding {
            return Err(RnsBggError::Context);
        }
        check_lut_binding(self.scheme, program, lut, input.0.level, binding)?;
        Ok((
            self.scheme.evaluate_lut(&input.0, prepared, &helpers.masks)?,
            self.cache.plaintexts.get(&gate).cloned(),
        ))
    }
}
impl RnsBggEncodingEvaluator<'_> {
    fn evaluate_linear(
        &mut self,
        private: &(RnsBggEncoding, Option<Mat>),
        public: &(RnsBggEncoding, Option<Mat>),
        gate: ProgramGateInstance,
    ) -> Result<(RnsBggEncoding, Option<Mat>), RnsBggError> {
        let Some(PreparedGate::Linear(prepared)) = self.cache.gates.get(&gate) else {
            return Err(RnsBggError::Context);
        };
        Ok((
            self.scheme.evaluate_linear(&private.0, &public.0, prepared)?,
            self.cache.plaintexts.get(&gate).cloned(),
        ))
    }
}

#[cfg(test)]
pub(crate) mod private_mul_tests {
    use super::*;
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::{
        element::PolyElem,
        matrix::PolyMatrix,
        poly::{Poly, PolyParams},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_encoding_graph_uses_gaussian_sampler_contract() {
        let n = 16;
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(4u8),
            crt_bits: 32,
            terminal_depth: 1,
            max_crt_depth: 2,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), [7u8; 32]);
        let error = scheme.small_error(0, 1, 1).unwrap();
        let graph = DslContext::new("rns-gaussian-contract")
            .output("error", error)
            .unwrap()
            .build()
            .unwrap()
            .graph;
        let samplers = graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                    Some((sigma, max_coefficient_bound))
                }
                NodeKind::UniformIntervalSample { .. } => {
                    panic!("encoding error still uses CBD sampling")
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(samplers.len(), 1);
        assert_eq!(samplers[0].0, &parameters.encoding_error_sigma_expr());
        assert_eq!(samplers[0].1, &IntExpr::from(parameters.encoding_error_bound.clone()));
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn test_rns_bgg_noiseless_encoding_only_replaces_gaussian_error() {
        let n = 16;
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(4u8),
            crt_bits: 32,
            terminal_depth: 1,
            max_crt_depth: 2,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        })
        .unwrap();
        let regular = RnsBggScheme::new(parameters.clone(), [7u8; 32]);
        let noiseless = RnsBggScheme::new_noiseless_for_testing(parameters.clone(), [7u8; 32]);
        assert_eq!(regular.parameter_identity(), noiseless.parameter_identity());
        for (scheme, expected_gaussian) in [(&regular, true), (&noiseless, false)] {
            let graph = DslContext::new(if expected_gaussian {
                "rns-gaussian-contract"
            } else {
                "rns-noiseless-contract"
            })
            .output("error", scheme.small_error(0, 1, 1).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .graph;
            let gaussian_count = graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
                .count();
            assert_eq!(gaussian_count, usize::from(expected_gaussian));
            let zero_count = graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| {
                    matches!(
                        node.kind(),
                        NodeKind::ConstantMatrix {
                            value: mxx_ir_core::node::ConstantMatrix::Zero,
                            ..
                        }
                    )
                })
                .count();
            assert_eq!(zero_count, usize::from(!expected_gaussian));
        }
    }

    #[test]
    fn test_rns_bgg_scalar_prepared_caches_cross_parallel_scope() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 10,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secrets = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let key =
            scheme.sample_input_key(0, seed.clone(), b"input".as_slice(), rand::random()).unwrap();
        let input = scheme
            .encode(
                &secrets.mask,
                &key,
                &scheme.prepare_input(&secrets.payload, &key, high.polynomial([1.into()])).unwrap(),
            )
            .unwrap();
        let rhs = scheme
            .sample_private_rhs(
                &secrets.payload,
                0,
                RnsBggRhsPlaintext {
                    payload: scheme.ring(1).unwrap().polynomial([1.into()]),
                    id: rand::random(),
                },
                seed,
                b"shared-rhs".as_slice(),
                rand::random(),
            )
            .unwrap();
        // The shared ciphertext and its lift precede both structural scopes.
        let rhs = scheme.prepare_private_rhs(&rhs).unwrap();
        use crate::program::{
            AttributeKnowledge, ExponentLutProgramBuilder, FamilyProgram, FamilyRhsSource,
            FamilyWireSource, Gate, ProgramStep, ProgramWireType, RhsId,
        };
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 0 }, n as usize).unwrap();
        let root = builder
            .input(1, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let rhs_family = builder.rhs_family(1).unwrap();
        let rhs_input = builder.rhs_input(rhs_family, 1, 1).unwrap();
        let local = WireId::from_index(0);
        let product = WireId::from_index(1);
        let output = WireId::from_index(2);
        let local_rhs = RhsId::from_index(0);
        let family_output = builder
            .family(
                2,
                FamilyProgram::new(
                    BTreeMap::from([(
                        local,
                        FamilyWireSource::Broadcast(builder.input_wire(root).unwrap()),
                    )]),
                    BTreeMap::from([(local_rhs, FamilyRhsSource::Broadcast(rhs_input))]),
                    vec![
                        Gate::MulPrivate { lhs: local, rhs: local_rhs, output: product },
                        Gate::Add { lhs: product, rhs: product, output },
                    ],
                    output,
                )
                .unwrap(),
            )
            .unwrap();
        builder.output(family_output).unwrap();
        let program = builder.build().unwrap();
        let ProgramStep::Family { id, body, .. } = &program.steps()[0] else {
            panic!("family body")
        };
        let private_gate = ProgramGateInstance::FamilyGate { step: 0, gate: 0, family: *id };
        let linear_gate = ProgramGateInstance::FamilyGate { step: 0, gate: 1, family: *id };
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(*id, FamilyRange::full(2).unwrap());
        let caches = crate::scheme::prepare_family(
            &scheme,
            &program,
            0,
            2.into(),
            &ranges,
            |_| {
                let (wires, rhs_bindings) = body.bind_inputs(
                    |_| Ok((key.clone(), None)),
                    |_| Err(DslError::Schema),
                    |_| Ok(RnsBggPublicRhs::Fixed(rhs.clone())),
                    |_| Err(DslError::Schema),
                )?;
                Ok((
                    wires,
                    rhs_bindings,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    RnsBggProgramCache::default(),
                ))
            },
            |value, mut cache| {
                let (public, _) = value;
                let Some(PreparedGate::MulPrivate(private)) = cache.gates.remove(&private_gate)
                else {
                    return Err(DslError::Schema)
                };
                let Some(PreparedGate::Linear(linear)) = cache.gates.remove(&linear_gate) else {
                    return Err(DslError::Schema)
                };
                // Only loop-varying fields leave this scope. The fixed RHS and
                // its high lift remain captured once outside both loops.
                Ok((public, private.output, linear))
            },
        )
        .unwrap();
        let schema = caches.schema().element;
        assert_eq!(schema.2.lhs.id, schema.1.id);
        assert_eq!(schema.0.level, 1);
        assert_eq!(schema.0.context, scheme.context);
        let outputs = crate::scheme::evaluate_family(
            &scheme,
            &program,
            0,
            2.into(),
            &ranges,
            |index| {
                let (_, private_output, linear) = caches.at(index);
                let private = RnsBggPreparedPrivateMul {
                    input: key.clone(),
                    output: private_output,
                    rhs: rhs.clone(),
                };
                let (wires, rhs_bindings) = body.bind_inputs(
                    |_| Ok((input.clone(), None)),
                    |_| Err(DslError::Schema),
                    |_| Ok(RnsBggPublicRhs::Identity(rhs.rhs_id)),
                    |_| Err(DslError::Schema),
                )?;
                Ok((
                    wires,
                    rhs_bindings,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    RnsBggProgramCache {
                        gates: BTreeMap::from([
                            (private_gate, PreparedGate::MulPrivate(private)),
                            (linear_gate, PreparedGate::Linear(linear)),
                        ]),
                        ..RnsBggProgramCache::default()
                    },
                    RnsBggProgramCache::default(),
                ))
            },
            |value| {
                let (encoding, _) = value;
                Ok(encoding)
            },
        )
        .unwrap();
        let direct = scheme.prepare_mul_private(&key, &rhs, schema.1.id).unwrap();
        let expected_product = scheme.evaluate_mul_private(&input, &direct).unwrap();
        let expected_linear =
            scheme.prepare_add(direct.output_key(), direct.output_key(), schema.0.id).unwrap();
        let expected =
            scheme.evaluate_linear(&expected_product, &expected_product, &expected_linear).unwrap();
        let mut graph = DslContext::new("rns_bgg-scalar-cache-parallel");
        for member in 0..2 {
            let actual = outputs.at(member);
            let (public, _, _) = caches.at(member);
            graph = graph
                .output(format!("encoding-{member}"), actual.row - expected.row.clone())
                .unwrap()
                .output(
                    format!("public-{member}"),
                    public.matrix - expected_linear.output_key().matrix.clone(),
                )
                .unwrap();
        }
        let graph = graph.build().unwrap();
        let count_lifts = |scope: &mxx_ir_core::graph::GraphScope| {
            scope
                .nodes()
                .iter()
                .filter(|node| {
                    matches!(node.kind(), NodeKind::CenteredExtend { .. }) &&
                        node.output_types()
                            .iter()
                            .any(|ty| matches!(ty, WireType::Preimage { .. }))
                })
                .count()
        };
        assert_eq!(count_lifts(graph.graph.root_scope()), 4);
        assert_eq!(
            graph.graph.scopes().values().map(count_lifts).sum::<usize>(),
            4,
            "one shared two-by-two bounded lift, with no digit conversions inside either family body"
        );
        let count_switches = |scope: &mxx_ir_core::graph::GraphScope| {
            scope
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::BlockModSwitch { .. }))
                .count()
        };
        assert_eq!(
            graph.graph.scopes().values().map(count_switches).sum::<usize>() -
                count_switches(graph.graph.root_scope()),
            6,
            "four public and two online polynomial switches belong to the family bodies"
        );
        let results = execute(
            &graph.validate(&ParamEnv::default()).unwrap(),
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial difference") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in polynomial.coeffs() {
                    assert_eq!(coefficient.value(), &BigUint::from(0u8));
                }
            }
        }
    }

    #[test]
    fn test_rns_bgg_direct_private_mul_relation_rounding_and_graph() {
        check_private_mul(
            |parameters| cpu_backend(parameters.levels().to_vec()),
            |matrix, q, bound| {
                for polynomial in matrix
                    .block_entries(0..matrix.row_size(), 0..matrix.col_size())
                    .into_iter()
                    .flatten()
                {
                    for coefficient in polynomial.coeffs() {
                        let absolute = coefficient.value().clone().min(q - coefficient.value());
                        assert!(&absolute <= bound, "direct error {absolute} exceeds {bound}");
                    }
                }
            },
        );
    }

    pub(crate) fn check_private_mul<B: mxx_runtime::Backend>(
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &BigUint, &BigUint),
    ) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let high_parameters = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let primes = high_parameters.to_crt().0;
        let levels = [3, 2, 1]
            .into_iter()
            .map(|depth| DCRTPolyParams::new(n, depth, 17, 4, Some(primes[..depth].to_vec()), None))
            .collect();
        let parameters = RnsBggParameters::new(levels, 2, 1).unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secrets = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let q_high = BigInt::from(parameters.level(0).unwrap().modulus().as_ref().clone());
        let q_low = BigInt::from(parameters.level(1).unwrap().modulus().as_ref().clone());
        let p = scheme.scalar(0, parameters.scale(0).unwrap().clone()).unwrap();
        let next_scale = scheme.scalar(1, parameters.scale(1).unwrap().clone()).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let key =
            scheme.sample_input_key(0, seed.clone(), b"input".as_slice(), rand::random()).unwrap();
        // Non-monomial inputs exercise convolution and payload norms above one.
        let mu = high.uniform_interval((1, 1), -1, 1) + high.uniform_interval((1, 1), -1, 1);
        let nu = low.uniform_interval((1, 1), -1, 1) + low.uniform_interval((1, 1), -1, 1);
        let input = scheme
            .encode(
                &secrets.mask,
                &key,
                &scheme.prepare_input(&secrets.payload, &key, mu.clone()).unwrap(),
            )
            .unwrap();
        let rhs = scheme
            .sample_private_rhs(
                &secrets.payload,
                0,
                RnsBggRhsPlaintext { payload: nu.clone(), id: rand::random() },
                seed.clone(),
                b"rhs".as_slice(),
                rand::random(),
            )
            .unwrap();
        let rhs = scheme.prepare_private_rhs(&rhs).unwrap();
        let prepared = scheme.prepare_mul_private(&key, &rhs, rand::random()).unwrap();
        let output = scheme.evaluate_mul_private(&input, &prepared).unwrap();

        let graph = DslContext::new("rns_bgg-direct-private-shape")
            .public_output("a", prepared.output.matrix.clone())
            .unwrap()
            .output("c", output.row.clone())
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        // One 2x2 public block ModDown and one 1x2 encoding block ModDown.
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::BlockModSwitch { .. }))
                .count(),
            6
        );
        assert!(!nodes.iter().any(|node| matches!(
            node.kind(),
            NodeKind::RingAutomorphism { .. } | NodeKind::Concat { .. } | NodeKind::Slice { .. }
        )));
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)).count(),
            12
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
                .count(),
            4
        );
        assert_eq!(output.level, 1);

        let s = scheme.mask_at(&secrets.mask, 0).unwrap();
        let r = scheme.payload_at(&secrets.payload, 0).unwrap();
        let r_low = scheme.payload_at(&secrets.payload, 1).unwrap();
        let input_gadget = scheme.state_gadget(0).unwrap();
        let output_gadget = scheme.state_gadget(1).unwrap();
        let r = r.map(|entry| entry * input_gadget.clone());
        let payload_row = r_low.clone().map(|entry| entry * output_gadget.clone());
        let error_in =
            input.row.clone() - s.clone() * key.matrix.clone() + p.clone() * mu.clone() * r;
        let error_rhs = r_low.clone() * rhs.ciphertext.clone() -
            next_scale.clone() * nu.clone() * payload_row.clone();
        let error_out = output.row.clone() -
            scheme.mask_at(&secrets.mask, 1).unwrap() * prepared.output.matrix.clone() +
            next_scale.clone() *
                mu.clone().reduce_modulus(q_low.clone()) *
                nu.clone() *
                payload_row.clone();
        // rho(z) = z - p*lift(MD(z)), using the executed output and public key.
        let public_remainder = key.matrix.clone() * prepared.rhs.lifted_ciphertext.clone() -
            p.clone() * prepared.output.matrix.clone().centered_extend(q_high.clone());
        let encoding_remainder = input.row.clone() * prepared.rhs.lifted_ciphertext.clone() -
            p.clone() * output.row.centered_extend(q_high.clone());
        let signed_terms = error_in * prepared.rhs.lifted_ciphertext -
            p.clone() * mu.clone() * error_rhs.centered_extend(q_high.clone()) +
            s * public_remainder -
            encoding_remainder;
        let relation = p * error_out.clone().centered_extend(q_high) - signed_terms;
        let count = n as usize;
        let range = crate::program::FamilyRange::full(count).unwrap();
        let public_family = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(count, |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: 0,
            id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let encoding_family = RnsBggEncodingFamily {
            rows: ProgramValue::Family(
                mxx_dsl::parallel(count, |_| Ok(input.row.clone())).unwrap(),
            ),
            level: 0,
            key_id: key.id,
            context: scheme.context,
            range,
        };
        let family_rhs = scheme
            .sample_private_rhs_family(
                &secrets.payload,
                0,
                &RnsBggRhsPlaintextFamily {
                    payloads: mxx_dsl::parallel(count, |_| Ok(nu.clone())).unwrap(),
                    id: rand::random(),
                },
                seed,
                b"family-rhs".as_slice(),
                rand::random(),
            )
            .unwrap();
        let family_rhs =
            scheme.prepare_private_rhs_family(&family_rhs, &public_family.range).unwrap();
        let family_prepared = scheme
            .prepare_mul_private_family(
                &public_family,
                &ProgramValue::Family(family_rhs),
                rand::random(),
            )
            .unwrap();
        let family_output =
            scheme.evaluate_mul_private_family(&encoding_family, &family_prepared).unwrap();
        let shape = DslContext::new("rns_bgg-direct-family-shape")
            .public_output("a", family_prepared.output.matrices.clone())
            .unwrap()
            .output("c", family_output.rows.clone())
            .unwrap()
            .build()
            .unwrap();
        let switches = shape
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::BlockModSwitch { .. }))
            .count();
        assert_eq!(switches, 6, "family size must not expand the operation body");
        let s_low = scheme.mask_at(&secrets.mask, 1).unwrap();
        let payload_term = next_scale * mu.reduce_modulus(q_low.clone()) * nu * payload_row;
        let family_error = mxx_dsl::parallel(count, |index| {
            Ok(family_output.rows.at(&index) -
                s_low.clone() * family_prepared.output.matrices.at(&index) +
                payload_term.clone())
        })
        .unwrap();
        let mut verifier = DslContext::new("rns_bgg-direct-private-relation")
            .output("relation", relation)
            .unwrap()
            .output("error", error_out)
            .unwrap();
        // Observation endpoints only: the production operation has one family
        // body, checked above, and every member's actual error is inspected.
        for member in 0..count {
            verifier =
                verifier.output(format!("family-error-{member}"), family_error.at(member)).unwrap();
        }
        let graph = verifier.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut make_backend(&parameters),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let input_bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: BigUint::from(2 * n),
        };
        let bound = parameters
            .private_mul_noise_bound(0, &input_bound, &1u8.into(), BigUint::from(2 * n).pow(2))
            .unwrap();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial output") };
            if name.starts_with("relation") {
                check_matrix(&matrix, parameters.level(0).unwrap().modulus().as_ref(), &0u8.into());
            } else {
                check_matrix(
                    &matrix,
                    parameters.level(1).unwrap().modulus().as_ref(),
                    &bound.error,
                );
            }
        }
    }
}

#[cfg(test)]
pub(super) mod prf_tests {
    use super::*;
    use crate::{
        layout::ExpLayout,
        noise_refresh::prf::{
            PreparedSparseLwrPrf, PrfSelectionInputs, SparseLwrPrfProfile, SparseLwrPrfProgram,
        },
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::{
        element::PolyElem,
        poly::{Poly, PolyParams},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_selector_batches_share_fresh_ciphertexts_across_outputs() {
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: 8,
            widths: vec![1, 1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.sample_payload_secret().unwrap();
        let high = scheme.ring(0).unwrap();
        let ring = scheme.ring(1).unwrap();
        let rhs = scheme
            .sample_private_rhs_family(
                &secret,
                0,
                &RnsBggRhsPlaintextFamily {
                    payloads: Family::pack(
                        (0..5)
                            .map(|index| ring.polynomial([usize::from(index == 2).into()]))
                            .collect(),
                    )
                    .unwrap(),
                    id: rand::random(),
                },
                high.bytes_input("hash-key", 32),
                b"shared-selector".as_slice(),
                rand::random(),
            )
            .unwrap();
        let weights = Family::pack(
            (0..2)
                .flat_map(|_| (0..5).map(|index| ring.polynomial([(index + 1).into()])))
                .collect(),
        )
        .unwrap();
        let combined = scheme
            .combine_fixed_rhs(
                &rhs,
                &weights,
                &FamilyRange::bounded(0, 5, 5).unwrap(),
                rand::random(),
            )
            .unwrap();
        let RnsBggRhsFamily::Values(values) = combined.ciphertext else {
            panic!("combined family")
        };
        let difference = values.at(0) - values.at(1);
        let columns = parameters.component_columns(1).unwrap();
        let mut graph = DslContext::new("shared-selector-errors")
            .output("zero", ring.zero((1, columns)))
            .unwrap();
        for row in 0..2 {
            for column in 0..2 {
                graph = graph
                    .output(
                        format!("difference-{row}-{column}"),
                        difference.entry(row, column).clone(),
                    )
                    .unwrap();
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes((0..32).map(|_| rand::random()).collect()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(zero) = &result.outputs["zero"] else { panic!("zero matrix") };
        for row in 0..2 {
            for column in 0..2 {
                let RuntimeValue::Matrix(value) =
                    &result.outputs[&format!("difference-{row}-{column}")]
                else {
                    panic!("matrix")
                };
                assert_eq!(value, zero, "both outputs must use the same fresh selector errors");
            }
        }
    }

    #[test]
    fn test_rns_bgg_composed_prf_two_steps_active_range_and_signed_labels() {
        check_prf_selection(
            |parameters| cpu_backend(parameters.levels().to_vec()),
            |matrix, q, bound| {
                for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                    for coefficient in polynomial.coeffs() {
                        assert!(coefficient.value().clone().min(q - coefficient.value()) <= *bound);
                    }
                }
            },
        );
    }

    pub(crate) fn check_prf_selection<B: mxx_runtime::Backend>(
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &BigUint, &BigUint),
    ) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1, 1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secrets = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let key = scheme
            .sample_input_key(0, seed.clone(), b"initial".as_slice(), rand::random())
            .unwrap();
        let layout = ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() };
        let program =
            SparseLwrPrfProgram::new(SparseLwrPrfProfile::new(layout, 1).unwrap(), 4, 2).unwrap();
        let mut blocks = Vec::new();
        let mut expected = [0usize; 3];
        for level in 0..2 {
            let ring = scheme.ring(level + 1).unwrap();
            let selected = 1 + usize::from(rand::random::<bool>());
            let selectors = scheme
                .sample_private_rhs_family(
                    &secrets.payload,
                    level,
                    &RnsBggRhsPlaintextFamily {
                        // Inactive slots are deliberately nonzero.
                        payloads: Family::pack(
                            (0..4)
                                .map(|cell| {
                                    ring.polynomial([usize::from(
                                        cell == 0 || cell == 3 || cell == selected,
                                    )
                                    .into()])
                                })
                                .collect(),
                        )
                        .unwrap(),
                        id: rand::random(),
                    },
                    seed.clone(),
                    format!("selectors-{level}").into_bytes(),
                    rand::random(),
                )
                .unwrap();
            let mut labels = Vec::new();
            for (label, exponent) in expected.iter_mut().enumerate() {
                for cell in 1..3 {
                    let value = (label + 1) * (level + 1) + cell * (n as usize - 1);
                    labels.push(layout.mono(&ring, &BigInt::from(value)).unwrap());
                    if cell == selected {
                        *exponent += value;
                    }
                }
            }
            blocks.push(PrfSelectionInputs {
                selectors,
                labels: Family::pack(labels).unwrap(),
                active: FamilyRange::bounded(1, 2, 4).unwrap(),
                id: rand::random(),
            });
        }
        let (public, prepared) = PreparedSparseLwrPrf::new(
            &scheme,
            program,
            RnsBggPublicProgramValue::Scalar((key.clone(), None)),
            &blocks,
        )
        .unwrap();
        let initial = scheme
            .encode(
                &secrets.mask,
                &key,
                &scheme.prepare_input(&secrets.payload, &key, high.polynomial([1.into()])).unwrap(),
            )
            .unwrap();
        let output = prepared
            .evaluate(&scheme, RnsBggEncodingProgramValue::Scalar((initial, None)))
            .unwrap();
        let RnsBggPublicProgramValue::Family(key) = public else { panic!("batched key") };
        let RnsBggEncodingProgramValue::Family(encoding) = output else {
            panic!("batched encoding")
        };
        let shape = DslContext::new("rns_bgg-composed-prf-shape")
            .public_output("a", key.field(|value| value.0.matrix).unwrap())
            .unwrap()
            .output("c", encoding.field(|value| value.0.row).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            shape.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::BlockModSwitch { .. }))
                .count(),
            12,
            "two steps, each with four public and two online polynomial ModDowns"
        );
        assert!(!nodes.iter().any(|node| matches!(
            node.kind(),
            NodeKind::RingAutomorphism { .. } | NodeKind::Concat { .. } | NodeKind::Slice { .. }
        )));
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)).count(),
            24
        );
        let low = scheme.ring(2).unwrap();
        let mut graph = DslContext::new("rns_bgg-composed-prf-relations");
        for (label, exponent) in expected.into_iter().enumerate() {
            let error = encoding.at(label).0.row -
                scheme.mask_at(&secrets.mask, 2).unwrap() * key.at(label).0.matrix +
                scheme.scalar(2, parameters.scale(2).unwrap().clone()).unwrap() *
                    layout.mono(&low, &BigInt::from(exponent)).unwrap() *
                    scheme.payload_at(&secrets.payload, 2).unwrap();
            graph = graph.output(format!("error-{label}"), error).unwrap();
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut make_backend(&parameters),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let mut bound = RnsBggNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() };
        for level in 0..2 {
            bound =
                parameters.private_mul_noise_bound(level, &bound, &2u8.into(), 1u8.into()).unwrap();
        }
        for value in result.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial error") };
            check_matrix(matrix, parameters.level(2).unwrap().modulus().as_ref(), &bound.error);
        }
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    use crate::rns_bgg::{RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest};
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{matrix::dcrt_poly::DCRTPolyMatrix, poly::PolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_linear_operations_preserve_exact_error_and_level() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let request = RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan(&request).unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let ring = scheme.ring(0).unwrap();
        let hash_key = ring.bytes_input("hash-key", 32);
        let lhs_key = scheme
            .sample_input_key(0, hash_key.clone(), b"lhs".as_slice(), rand::random())
            .unwrap();
        let rhs_key = scheme
            .sample_input_key(0, hash_key.clone(), b"rhs".as_slice(), rand::random())
            .unwrap();
        let lower_key = scheme
            .sample_input_key(1, hash_key.clone(), b"lower".as_slice(), rand::random())
            .unwrap();
        // General polynomials exercise linear payload semantics, including
        // signs and negacyclic convolution, instead of only exponent values.
        let mu = ring.uniform_interval((1, 1), -1, 1) + ring.uniform_interval((1, 1), -1, 1);
        let nu = ring.uniform_interval((1, 1), -1, 1) + ring.uniform_interval((1, 1), -1, 1);
        let lhs = scheme
            .encode(
                &secret.mask,
                &lhs_key,
                &scheme.prepare_input(&secret.payload, &lhs_key, mu.clone()).unwrap(),
            )
            .unwrap();
        let rhs = scheme
            .encode(
                &secret.mask,
                &rhs_key,
                &scheme.prepare_input(&secret.payload, &rhs_key, nu.clone()).unwrap(),
            )
            .unwrap();
        let scale = scheme.scalar(0, parameters.scale(0).unwrap().clone()).unwrap();
        let q = BigInt::from(parameters.level(0).unwrap().modulus().as_ref().clone());
        let s = secret.mask.row.clone().reduce_modulus(q.clone());
        let gadget = scheme.state_gadget(0).unwrap();
        let r = secret.payload.row.clone().reduce_modulus(q).map(|entry| entry * gadget.clone());
        let lhs_error = lhs.row.clone() - s.clone() * lhs_key.matrix.clone() +
            scale.clone() * mu.clone() * r.clone();
        let rhs_error = rhs.row.clone() - s.clone() * rhs_key.matrix.clone() +
            scale.clone() * nu.clone() * r.clone();
        let add = scheme.prepare_add(&lhs_key, &rhs_key, rand::random()).unwrap();
        let added = scheme.evaluate_linear(&lhs, &rhs, &add).unwrap();
        let mul =
            scheme.prepare_mul_public(&lhs_key, &rhs_key, nu.clone(), rand::random()).unwrap();
        let multiplied = scheme.evaluate_linear(&lhs, &rhs, &mul).unwrap();
        assert_eq!(added.level(), 0);
        assert_eq!(multiplied.level(), 0);
        assert!(scheme.prepare_add(&lhs_key, &lower_key, rand::random()).is_err());
        assert!(
            scheme.prepare_mul_public(&lhs_key, &lower_key, nu.clone(), rand::random()).is_err()
        );
        assert!(
            scheme
                .prepare_mul_public(&lhs_key, &rhs_key, ring.identity(2), rand::random())
                .is_err()
        );
        assert!(scheme.evaluate_linear(&rhs, &lhs, &add).is_err());
        let other = RnsBggScheme::new(parameters.clone(), rand::random());
        assert!(other.evaluate_linear(&lhs, &rhs, &mul).is_err());
        let graph = DslContext::new("rns_bgg-linear-error-relations")
            .output(
                "add",
                added.row - s.clone() * add.output_key().matrix.clone() +
                    scale.clone() * (mu.clone() + nu.clone()) * r.clone() -
                    lhs_error.clone() -
                    rhs_error,
            )
            .unwrap()
            .output(
                "mul-public",
                multiplied.row - s * mul.output_key().matrix.clone() + scale * mu * nu.clone() * r -
                    nu * lhs_error,
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let zero = DCRTPolyMatrix::zero(
            parameters.level(0).unwrap(),
            1,
            parameters.component_columns(0).unwrap(),
        );
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("matrix residual") };
            assert_eq!(matrix.as_ref(), &zero);
        }
        let input_bound = RnsBggNoiseBound { error: 2u8.into(), payload_l1: 3u8.into() };
        let sum_bound = input_bound.add(&input_bound);
        assert_eq!(sum_bound.error, BigUint::from(4u8));
        assert_eq!(sum_bound.payload_l1, BigUint::from(6u8));
        let product_bound = input_bound.mul_public(&BigUint::from(5u8));
        assert_eq!(product_bound.error, BigUint::from(10u8));
        assert_eq!(product_bound.payload_l1, BigUint::from(15u8));
    }
}

#[cfg(test)]
pub(crate) mod encoding_tests {
    use super::*;
    use crate::rns_bgg::{RnsBggNoiseBound, RnsBggPlanRequest};
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{element::PolyElem, poly::Poly};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, executor::execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigUint;
    use num_traits::One;
    use std::collections::BTreeMap;

    /// Golden round-trip fixtures persist their online preparation separately
    /// from the public cache, so comparison uses the same sampled helper error.
    pub(crate) fn export_online_masks(
        masks: &RnsBggOnlineMasks,
        mut graph: DslContext,
        prefix: &str,
    ) -> DslContext {
        for (sigma, row) in masks.rows.iter() {
            for column in 0..2 {
                graph = graph
                    .private_output(
                        format!("{prefix}-{sigma}-{column}"),
                        row.entry(0, column).clone(),
                    )
                    .unwrap();
            }
        }
        graph
    }
    pub(crate) fn import_online_masks(
        scheme: &RnsBggScheme,
        template: &RnsBggOnlineMasks,
        production: &mxx_ir_core::artifact::ProductionId,
        prefix: &str,
    ) -> RnsBggOnlineMasks {
        let ring = scheme.ring(template.level).unwrap();
        let rows = template
            .rows
            .keys()
            .map(|sigma| {
                (
                    *sigma,
                    RnsBggRow::new([std::array::from_fn(|column| {
                        ring.artifact_input(
                            production.clone(),
                            format!("{prefix}-{sigma}-{column}"),
                            (1, scheme.parameters.component_columns(template.level).unwrap()),
                            mxx_ir_core::artifact::ArtifactConfidentiality::Private,
                        )
                    })])
                    .unwrap(),
                )
            })
            .collect();
        RnsBggOnlineMasks { rows: Arc::new(rows), ..template.clone() }
    }

    #[test]
    fn test_rns_bgg_preprocessing_is_reused_with_independent_online_secrets() {
        check_preprocessing_phases(
            |parameters| cpu_backend(parameters.levels().iter().cloned()),
            |matrix, name, q, bound| {
                for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                    for coefficient in polynomial.coeffs() {
                        let absolute = coefficient.value().clone().min(q - coefficient.value());
                        assert!(&absolute <= bound, "{name}: {absolute}");
                    }
                }
            },
        );
    }

    pub(crate) fn check_preprocessing_phases<B: mxx_runtime::Backend>(
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &str, &BigUint, &BigUint),
    ) {
        use crate::{
            program::FamilyRange,
            rns_bgg::{RnsBggEncodingFamily, RnsBggPublicKeyFamily, RnsBggRhsPlaintextFamily},
        };
        use mxx_ir_core::artifact::ArtifactConfidentiality;
        use mxx_runtime::artifact::{ArtifactStore, FileArtifactStore};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 8,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let payload_secret = scheme.sample_payload_secret().unwrap();
        let key =
            scheme.sample_input_key(0, seed.clone(), b"input".as_slice(), rand::random()).unwrap();
        let mu = high.polynomial([2.into(), (-3).into()]);
        let material = scheme.prepare_input(&payload_secret, &key, mu.clone()).unwrap();
        let mut producer = DslContext::new("rns_bgg-input-preprocessing");
        let (graph, input_descriptor) =
            scheme.export_input_material(&key, &material, producer, "input").unwrap();
        producer = graph;
        let input_descriptor =
            serde_json::from_slice(&serde_json::to_vec(&input_descriptor).unwrap()).unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (1 as usize).trailing_zeros(),
                },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let lut_material = scheme
            .sample_lut_material(
                &payload_secret,
                None,
                None,
                &bank,
                seed.clone(),
                b"lut".as_slice(),
                rand::random(),
            )
            .unwrap();
        let prepared_lut = scheme.prepare_lut(&key, &lut_material, rand::random()).unwrap();
        let (graph, lut_descriptor) =
            scheme.export_prepared_lut(&prepared_lut, producer, "lut").unwrap();
        producer = graph;
        let range = FamilyRange::bounded(0, 2, 2).unwrap();
        let public_family = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(2, |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: 0,
            id: key.id,
            context: key.context,
            range: range.clone(),
        };
        let low = scheme.ring(1).unwrap();
        let rhs = RnsBggRhsPlaintextFamily {
            payloads: Family::pack(vec![low.identity(1), low.zero((1, 1))]).unwrap(),
            id: rand::random(),
        };
        let family_material = scheme
            .sample_lut_material_family(
                &payload_secret,
                None,
                Some(&rhs),
                &bank,
                2.into(),
                seed.clone(),
                b"family".as_slice(),
                rand::random(),
            )
            .unwrap();
        let prepared_family =
            scheme.prepare_lut_family(&public_family, &family_material, rand::random()).unwrap();
        let (graph, family_descriptor) =
            scheme.export_prepared_lut_family(&prepared_family, producer, "family").unwrap();
        producer = graph;
        // A public production has no d rows: s has not even been sampled.
        for column in 0..2 {
            producer = producer
                // Test-only oracle data, not a public preprocessing artifact.
                .private_output(
                    format!("oracle-r-{column}"),
                    payload_secret.row.entry(0, column).clone(),
                )
                .unwrap();
        }
        let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
        std::fs::create_dir_all("test_data/rns_bgg_input_phases").unwrap();
        let directory = tempfile::tempdir_in("test_data/rns_bgg_input_phases").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = make_backend(&parameters);
        let produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let manifest = store.load_manifest(&production).unwrap();
        for (name, artifact) in &manifest.artifacts {
            if name.starts_with("lut.") ||
                name.starts_with("family.") ||
                name.starts_with("input.key.")
            {
                assert!(!name.contains(".d."), "online helper in public preprocessing: {name}");
                assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Public);
            } else if name.starts_with("input.payload.") || name.starts_with("oracle-r-") {
                assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Private);
            }
        }
        let private_row = |prefix: &str| {
            RnsBggRow::new([std::array::from_fn(|column| {
                high.artifact_input(
                    production.clone(),
                    format!("{prefix}-{column}"),
                    (1, 1),
                    ArtifactConfidentiality::Private,
                )
            })])
            .unwrap()
        };
        let (key, imported) =
            scheme.import_input_material(&input_descriptor, production.clone(), key.id, 0).unwrap();
        let mut wrong_key = key.id;
        wrong_key[0] ^= 1;
        assert!(
            scheme
                .import_input_material(&input_descriptor, production.clone(), wrong_key, 0)
                .is_err()
        );
        assert!(
            scheme.import_input_material(&input_descriptor, production.clone(), key.id, 1).is_err()
        );
        let prepared_lut = scheme
            .import_prepared_lut(
                &lut_descriptor,
                production.clone(),
                &lut_material.binding(),
                key.id,
                prepared_lut.output.id,
            )
            .unwrap();
        let prepared_family = scheme
            .import_prepared_lut_family(
                &family_descriptor,
                production.clone(),
                family_material.binding(),
                key.id,
                prepared_family.output.id,
                &range,
            )
            .unwrap();
        let low_q = parameters.level(1).unwrap().modulus();
        let low_modulus = BigInt::from(low_q.as_ref().clone());

        let lut_bound = parameters
            .lut_noise_bound(
                0,
                &RnsBggNoiseBound {
                    error: parameters.encoding_error_bound.clone(),
                    payload_l1: 5u8.into(),
                },
                1,
                &1u8.into(),
                5u8.into(),
            )
            .unwrap();
        assert!(parameters.noise_fits(1, &lut_bound).unwrap());
        let oracle_r = private_row("oracle-r");
        let q = parameters.level(0).unwrap().modulus();

        let mut online = DslContext::new("rns_bgg-independent-online-secrets");
        for index in 0..2 {
            // s is sampled only after the preprocessing production was closed.
            let s = scheme.sample_mask_secret().unwrap();
            let encoding = scheme.encode(&s, &key, &imported).unwrap();
            let online_masks = scheme.prepare_online_masks(&s, &prepared_lut.mask_bank()).unwrap();
            let selected = scheme.evaluate_lut(&encoding, &prepared_lut, &online_masks).unwrap();
            let mut wrong_masks = online_masks.clone();
            wrong_masks.id[0] ^= 1;
            assert!(scheme.evaluate_lut(&encoding, &prepared_lut, &wrong_masks).is_err());
            let input_family = RnsBggEncodingFamily {
                rows: ProgramValue::Family(
                    mxx_dsl::parallel(2, |_| Ok(encoding.row.clone())).unwrap(),
                ),
                level: 0,
                key_id: key.id,
                context: key.context,
                range: range.clone(),
            };
            let selected_family =
                scheme.evaluate_lut_family(&input_family, &prepared_family, &online_masks).unwrap();
            let low_s = s.row.clone().reduce_modulus(low_modulus.clone());
            let low_gadget = scheme.state_gadget(1).unwrap();
            let low_r = oracle_r
                .clone()
                .reduce_modulus(low_modulus.clone())
                .map(|entry| entry * low_gadget.clone());
            let low_mu = mu.clone().reduce_modulus(low_modulus.clone());
            let low_error = |row: RnsBggRow, public: RnsBggMatrix, payload: Mat| {
                row - low_s.clone() * public +
                    scheme.scalar(1, parameters.scale(1).unwrap().clone()).unwrap() *
                        payload *
                        low_r.clone()
            };
            online = online
                .output(
                    format!("lut-error-{index}"),
                    low_error(selected.row, prepared_lut.output.matrix.clone(), low_mu.clone()),
                )
                .unwrap();
            for member in 0..2 {
                let payload = if member == 0 { low_mu.clone() } else { low.zero((1, 1)) };
                online = online
                    .output(
                        format!("family-error-{index}-{member}"),
                        low_error(
                            selected_family.rows.at(member),
                            prepared_family.output.matrices.at(member),
                            payload,
                        ),
                    )
                    .unwrap();
            }
            let error = encoding.row - s.row * key.matrix.clone() +
                scheme.scalar(0, parameters.scale(0).unwrap().clone()).unwrap() *
                    mu.clone() *
                    oracle_r.clone().map(|entry| entry * scheme.state_gadget(0).unwrap());
            online = online.output(format!("error-{index}"), error).unwrap();
        }
        let mut wrong_key = key.clone();
        wrong_key.id[0] ^= 1;
        assert!(
            scheme.encode(&scheme.sample_mask_secret().unwrap(), &wrong_key, &imported).is_err()
        );
        let online = online
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]),
            )
            .unwrap();
        let results =
            execute(&online, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for (name, output) in results.outputs {
            let RuntimeValue::Matrix(matrix) = output else { panic!("polynomial") };
            let (modulus, bound) = if name.starts_with("error-") {
                (q.as_ref(), parameters.encoding_error_bound.clone())
            } else {
                (low_q.as_ref(), lut_bound.error.clone())
            };
            check_matrix(&matrix, &name, modulus, &bound);
        }
    }

    #[test]
    fn test_rns_bgg_mul_private_preserves_coefficient_selectors_and_alignment() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let request = RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan(&request).unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let hash_key = high.bytes_input("hash-key", 32);
        let key = scheme
            .sample_input_key(0, hash_key.clone(), b"input".as_slice(), rand::random())
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (1 as usize).trailing_zeros(),
                },
                hash_key.clone(),
                b"identity".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_bank = scheme.prepare_online_masks(&secret.mask, &bank).unwrap();
        let low_key = scheme
            .sample_input_key(1, hash_key.clone(), b"low-input".as_slice(), rand::random())
            .unwrap();
        let low_payload = scheme.ring(1).unwrap().uniform_interval((1, 1), -1, 1);
        let low_input = scheme
            .encode(
                &secret.mask,
                &low_key,
                &scheme.prepare_input(&secret.payload, &low_key, low_payload.clone()).unwrap(),
            )
            .unwrap();
        let payload = high.uniform_interval((1, 1), -1, 1);
        let input = scheme
            .encode(
                &secret.mask,
                &key,
                &scheme.prepare_input(&secret.payload, &key, payload.clone()).unwrap(),
            )
            .unwrap();
        let q = parameters.level(1).unwrap().modulus();
        let s = scheme.mask_at(&secret.mask, 1).unwrap();
        let r = scheme.payload_at(&secret.payload, 1).unwrap();

        let mut graph = DslContext::new("rns_bgg-identity-selectors");
        for (case, selector) in [Some(0u8), Some(1), None].into_iter().enumerate() {
            let material = scheme
                .sample_lut_material(
                    &secret.payload,
                    None,
                    selector.map(|bit| RnsBggRhsPlaintext {
                        payload: scheme.scalar(1, bit).unwrap(),
                        id: rand::random(),
                    }),
                    &bank,
                    hash_key.clone(),
                    format!("identity-{case}").into_bytes(),
                    rand::random(),
                )
                .unwrap();
            assert_eq!(material.table_commitment(), None);
            assert_eq!(material.has_rhs(), selector.is_some());
            let prepared = scheme.prepare_lut(&key, &material, rand::random()).unwrap();
            let selected = scheme.evaluate_lut(&input, &prepared, &online_bank).unwrap();
            assert_eq!(selected.level(), 1);
            let joined =
                scheme.prepare_add(prepared.output_key(), &low_key, rand::random()).unwrap();
            let output = scheme.evaluate_linear(&selected, &low_input, &joined).unwrap();
            // A zero coefficient selector annihilates the payload. It is not
            // an exponent-zero encoding, which would preserve the payload.
            let expected = payload.clone().reduce_modulus(BigInt::from(q.as_ref().clone())) *
                scheme.scalar(1, selector.unwrap_or(1)).unwrap() +
                low_payload.clone();
            let physical_error =
                output.row - s.clone() * joined.output_key().matrix.clone() + expected * r.clone();
            graph = graph.output(format!("error-{case}"), physical_error).unwrap();
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let input_bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: n.into(),
        };
        let selected_bound =
            parameters.lut_noise_bound(0, &input_bound, 1, &1u8.into(), n.into()).unwrap();
        assert_eq!(selected_bound.payload_l1, BigUint::from(n));
        let bound = selected_bound.add(&input_bound);
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("error row") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in polynomial.coeffs() {
                    let magnitude =
                        coefficient.value().clone().min(q.as_ref() - coefficient.value());
                    assert!(magnitude <= bound.error);
                }
            }
        }
    }

    #[test]
    fn test_rns_bgg_private_product_and_two_luts_all_binary_inputs() {
        use mxx_primitives::matrix::PolyMatrix;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(32);
        let pool = DCRTPolyParams::new(n, 4, 17, 4, None, None);
        let primes = pool.to_crt().0;
        let parameters = RnsBggParameters::new(
            [4, 3, 2, 1]
                .into_iter()
                .map(|depth| {
                    DCRTPolyParams::new(n, depth, 17, 4, Some(primes[..depth].to_vec()), None)
                })
                .collect(),
            2,
            1,
        )
        .unwrap();
        assert_eq!(parameters.scale(3).unwrap(), &BigUint::one());
        assert!(parameters.scale(1).unwrap() > &BigUint::one());
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let hash_key = scheme.ring(0).unwrap().bytes_input("hash-key", 32);
        let input_key = scheme
            .sample_input_key(0, hash_key.clone(), b"inputs".as_slice(), rand::random())
            .unwrap();
        let mask0 = scheme
            .sample_mask_bank(
                1,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (4 as usize).trailing_zeros(),
                },
                hash_key.clone(),
                b"mask0".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_mask0 = scheme.prepare_online_masks(&secret.mask, &mask0).unwrap();
        let mask1 = scheme
            .sample_mask_bank(
                2,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (2 as usize).trailing_zeros(),
                },
                hash_key.clone(),
                b"mask1".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_mask1 = scheme.prepare_online_masks(&secret.mask, &mask1).unwrap();
        let table = LutTable::unary(4, 2, vec![1, 1, 1, 0]).unwrap();
        let flip = LutTable::unary(2, 2, vec![1, 0]).unwrap();
        let second_material = scheme
            .sample_lut_material(
                &secret.payload,
                Some(&flip),
                None,
                &mask1,
                hash_key.clone(),
                b"second".as_slice(),
                rand::random(),
            )
            .unwrap();
        let material = scheme
            .sample_lut_material(
                &secret.payload,
                Some(&table),
                None,
                &mask0,
                hash_key.clone(),
                b"shared-first-lut".as_slice(),
                rand::random(),
            )
            .unwrap();
        let mut graph = DslContext::new("rns_bgg-two-level-binary-lut");
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        for y in 0..2 {
            let rhs = layout.mono(&scheme.ring(1).unwrap(), &BigInt::from(2 * y)).unwrap();
            let rhs = scheme
                .sample_private_rhs(
                    &secret.payload,
                    0,
                    RnsBggRhsPlaintext { payload: rhs, id: rand::random() },
                    hash_key.clone(),
                    format!("rhs-{y}").into_bytes(),
                    rand::random(),
                )
                .unwrap();
            let rhs = scheme.prepare_private_rhs(&rhs).unwrap();
            let product_prepared =
                scheme.prepare_mul_private(&input_key, &rhs, rand::random()).unwrap();
            let first_prepared = scheme
                .prepare_lut(product_prepared.output_key(), &material, rand::random())
                .unwrap();
            let second_prepared = scheme
                .prepare_lut(first_prepared.output_key(), &second_material, rand::random())
                .unwrap();
            for x in 0..2 {
                let mu = layout.mono(&scheme.ring(0).unwrap(), &BigInt::from(x)).unwrap();
                let input = scheme
                    .encode(
                        &secret.mask,
                        &input_key,
                        &scheme.prepare_input(&secret.payload, &input_key, mu).unwrap(),
                    )
                    .unwrap();
                let product = scheme.evaluate_mul_private(&input, &product_prepared).unwrap();
                let first = scheme.evaluate_lut(&product, &first_prepared, &online_mask0).unwrap();
                let second = scheme.evaluate_lut(&first, &second_prepared, &online_mask1).unwrap();
                let value = flip.values()[table.values()[x + 2 * y]];
                let expected_mu =
                    layout.mono(&scheme.ring(3).unwrap(), &BigInt::from(value)).unwrap();
                let s = scheme.mask_at(&secret.mask, 3).unwrap();
                let r = scheme.payload_at(&secret.payload, 3).unwrap();
                let error =
                    second.row - s * second_prepared.output_key().matrix.clone() + expected_mu * r;

                graph = graph.output(format!("error-{x}-{y}"), error).unwrap();
                assert!(scheme.evaluate_lut(&input, &second_prepared, &online_mask1).is_err());
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".to_owned(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let mut bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: BigUint::one(),
        };
        bound = parameters.private_mul_noise_bound(0, &bound, &1u8.into(), BigUint::one()).unwrap();
        bound = parameters.lut_noise_bound(1, &bound, 4, &1u8.into(), BigUint::one()).unwrap();
        bound = parameters.lut_noise_bound(2, &bound, 2, &1u8.into(), BigUint::one()).unwrap();
        let q = parameters.level(3).unwrap().modulus();
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else {
                panic!("matrix error output");
            };
            for polynomial in matrix
                .block_entries(0..matrix.row_size(), 0..matrix.col_size())
                .into_iter()
                .flatten()
            {
                for coefficient in polynomial.coeffs() {
                    let magnitude =
                        coefficient.value().clone().min(q.as_ref() - coefficient.value());
                    assert!(
                        magnitude <= bound.error,
                        "observed {magnitude}, bound {}",
                        bound.error
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod common_prf_tests {
    use super::{ProgramFamilyBindings, RnsBggEncoding, RnsBggPrivateRhs, RnsBggPublicRhs};
    use crate::{
        layout::ExpLayout,
        noise_refresh::{
            pbc::{PbcActiveCellIndex, PbcPublicLayout},
            prf::*,
        },
        program::FamilyRange,
    };
    use std::collections::BTreeMap;
    #[test]
    fn test_sparse_lwr_common_interpreter_accepts_rns_bgg_two_block_inputs() {
        check_rns_bgg_two_block_inputs(0);
    }

    #[test]
    fn test_sparse_lwr_common_interpreter_accepts_strided_rns_bgg_two_block_inputs() {
        check_rns_bgg_two_block_inputs(1);
    }

    fn check_rns_bgg_two_block_inputs(stride_log2: u32) {
        use crate::{
            noise_refresh::pbc::{PbcCell, PbcParameters, PbcRootSeed, derive_attempt_seed},
            rns_bgg::{
                RnsBggEncodingProgramValue, RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest,
                RnsBggPublicProgramValue, RnsBggRhsPlaintextFamily, RnsBggScheme,
            },
        };
        use mxx_dsl::{DslContext, Family};
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::{
            element::PolyElem,
            matrix::PolyMatrix,
            poly::{Poly, PolyParams},
        };
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
            transcript::SamplingMode,
        };
        use num_bigint::{BigInt, BigUint};
        let n: u32 = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let exponent_layout =
            ExpLayout { stride_log2, width_log2: n.trailing_zeros() - stride_log2 };
        let native_width = exponent_layout.width();
        let pbc = PbcPublicLayout::build(
            &PbcParameters::custom(2, 1, 2, 2, 1, None),
            derive_attempt_seed(PbcRootSeed(rand::random()), 0),
            0,
        )
        .unwrap();
        let active = PbcActiveCellIndex::build(&pbc).unwrap();
        let program = SparseLwrPrfProgram::new(
            SparseLwrPrfProfile::new(exponent_layout, native_width / 2).unwrap(),
            pbc.bucket_width,
            2,
        )
        .unwrap();
        // This fixture exercises ordinary Program Family selection, including
        // its separate public multiplication and reduction. Native PRFs use
        // the fixed-product Program instead.
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        builder.set_layout(exponent_layout, n as usize).unwrap();
        let family_input = builder
            .input(
                native_width,
                crate::program::ProgramWireType {
                    stride_log2: exponent_layout.stride_log2,
                    knowledge: crate::program::AttributeKnowledge::Private,
                },
            )
            .unwrap();
        let mut family_output = builder.input_wire(family_input).unwrap();
        let mut selector_families = Vec::new();
        let mut wire_families = Vec::new();
        for _ in 0..2 {
            let selectors = builder.rhs_family(native_width).unwrap();
            let values = builder.wire_family(native_width).unwrap();
            family_output =
                builder.select(family_output, selectors, values, pbc.bucket_width).unwrap();
            selector_families.push(selectors);
            wire_families.push(values);
        }
        builder.output(family_output).unwrap();
        let family_program = builder.build().unwrap();

        let widths = active.bucket_active_widths().collect::<Vec<_>>();
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1, 1, native_width],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 24,
            terminal_value_bound: (native_width - 1).into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let initial_key = scheme
            .sample_input_key(0, seed.clone(), b"initial".as_slice(), rand::random())
            .unwrap();
        let topology = crate::noise_refresh::simulation::RefreshTranscriptTopology::new(
            &scheme,
            &(initial_key.clone(), None),
            &ParamEnv::default(),
            vec![1; 3],
            1,
        )
        .unwrap();
        let columns = 2 * parameters.component_columns(0).unwrap();
        assert_eq!(topology.component_columns, columns);
        assert_eq!(topology.joint_event_count().unwrap(), 4 * columns * n as usize);
        let labels =
            crate::noise_refresh::refresh::RefreshPrfLabelIndex::new(rand::random(), &topology)
                .unwrap();
        assert_eq!(labels.len(), topology.joint_event_count().unwrap());
        for flat in 0..labels.len() {
            assert_eq!(labels.index_of(labels.label(flat).unwrap()), Some(flat));
        }

        let initial = scheme
            .encode(
                &secret.mask,
                &initial_key,
                &scheme
                    .prepare_input(
                        &secret.payload,
                        &initial_key,
                        program.initial_attribute(&high).unwrap(),
                    )
                    .unwrap(),
            )
            .unwrap();
        let entries = active.iter().collect::<Vec<_>>();
        let label = |bucket: usize, slot: usize| match pbc.cells[bucket][slot] {
            PbcCell::Real { coordinate, .. } => native_width - 1 - coordinate,
            PbcCell::Dummy => 0,
            PbcCell::Padding => panic!("active index must exclude padding"),
        };
        let mut public_members = Vec::new();
        let mut online_members = Vec::new();
        for block in 0..2 {
            let ring = scheme.ring(block + 1).unwrap();
            let key = scheme
                .sample_input_key(
                    block + 1,
                    seed.clone(),
                    format!("labels-{block}").into_bytes(),
                    rand::random(),
                )
                .unwrap();
            let block_entries = active.bucket_iter(block).unwrap().collect::<Vec<_>>();
            let plaintexts = Family::pack(
                block_entries
                    .iter()
                    .map(|(bucket, slot, _)| {
                        exponent_layout.mono(&ring, &BigInt::from(label(*bucket, *slot))).unwrap()
                    })
                    .collect(),
            )
            .unwrap();
            let rows = mxx_dsl::parallel(block_entries.len(), |index| {
                Ok(scheme
                    .encode(
                        &secret.mask,
                        &key,
                        &scheme.prepare_input(&secret.payload, &key, plaintexts.at(index)).unwrap(),
                    )
                    .unwrap()
                    .row)
            })
            .unwrap();
            let range =
                FamilyRange::bounded(block_entries[0].2, block_entries.len(), pbc.bucket_width)
                    .unwrap();
            public_members.push((
                Family::pack(
                    (0..block_entries.len())
                        .map(|index| (key.clone(), Some(plaintexts.at(index))))
                        .collect(),
                )
                .unwrap(),
                range.clone(),
            ));
            online_members.push((
                Family::pack(
                    (0..block_entries.len())
                        .map(|index| {
                            (
                                RnsBggEncoding {
                                    row: rows.at(index),
                                    level: block + 1,
                                    key_id: key.id,
                                    context: scheme.context,
                                },
                                Some(plaintexts.at(index)),
                            )
                        })
                        .collect(),
                )
                .unwrap(),
                range,
            ));
        }

        let mut bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: 1u8.into(),
        };
        for (block, count) in widths.iter().enumerate() {
            let branch =
                parameters.private_mul_noise_bound(block, &bound, &1u8.into(), 1u8.into()).unwrap();
            // One-hot is a payload contract. Every active selector ciphertext
            // still contributes its inherited and fresh error.
            bound = RnsBggNoiseBound { error: branch.error * count, payload_l1: 1u8.into() };
        }
        assert!(parameters.noise_fits(2, &bound).unwrap());
        let rounding_values = program.readout.canonical_coefficients().unwrap();
        let terminal_bank = scheme
            .sample_mask_bank(
                2,
                exponent_layout,
                seed.clone(),
                b"terminal-bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_terminal_bank =
            scheme.prepare_online_masks(&secret.mask, &terminal_bank).unwrap();
        let readout_material = scheme
            .sample_readout_material(
                &secret.payload,
                &program.readout,
                &terminal_bank,
                seed.clone(),
                b"rounding-table".as_slice(),
                rand::random(),
                None,
                None,
            )
            .unwrap();
        let terminal_bound = parameters
            .terminal_noise_bound(
                2,
                [(&bound, native_width, &1u8.into())],
                readout_material.target_bound().clone(),
            )
            .unwrap();
        assert!(parameters.terminal_noise_fits(3, &terminal_bound).unwrap());
        let terminal_q = parameters.level(3).unwrap().modulus();
        let terminal_s =
            secret.mask.row.clone().reduce_modulus(BigInt::from(terminal_q.as_ref().clone()));
        let low = scheme.ring(2).unwrap();
        let q = parameters.level(2).unwrap().modulus();
        let s = secret.mask.row.clone().reduce_modulus(BigInt::from(q.as_ref().clone()));
        let gadget = scheme.state_gadget(2).unwrap();
        let r = secret
            .payload
            .row
            .clone()
            .reduce_modulus(BigInt::from(q.as_ref().clone()))
            .map(|entry| entry * gadget.clone());
        let mut graph = DslContext::new("generic-sparse-lwr-rns_bgg");
        let mut seen_halves = [false; 2];
        for choice in 0..4 {
            let selected = (0..2)
                .map(|block| {
                    let positions = active.bucket_iter(block).unwrap().collect::<Vec<_>>();
                    if choice & (1 << block) == 0 {
                        positions[0]
                    } else {
                        *positions.last().unwrap()
                    }
                })
                .collect::<Vec<_>>();
            let mut public_blocks = Vec::new();
            let mut online_blocks = Vec::new();
            for block in 0..2 {
                let ring = scheme.ring(block + 1).unwrap();
                let rhs_id = rand::random();
                let selectors = RnsBggRhsPlaintextFamily {
                    payloads: Family::pack(
                        entries
                            .iter()
                            .map(|entry| {
                                ring.polynomial([usize::from(selected.contains(entry)).into()])
                            })
                            .collect(),
                    )
                    .unwrap(),
                    id: rhs_id,
                };
                let material = scheme
                    .sample_private_rhs_family(
                        &secret.payload,
                        block,
                        &selectors,
                        seed.clone(),
                        format!("selectors-{choice}-{block}").into_bytes(),
                        rand::random(),
                    )
                    .unwrap();
                let positions = active.bucket_iter(block).unwrap().collect::<Vec<_>>();
                let range = FamilyRange::bounded(positions[0].2, positions.len(), pbc.bucket_width)
                    .unwrap();
                let material = scheme.prepare_private_rhs_family(&material, &range).unwrap();
                let rhs_values = Family::pack(
                    (0..positions.len())
                        .map(|local| {
                            RnsBggPublicRhs::Fixed(RnsBggPrivateRhs {
                                ciphertext: material.ciphertext.at(local),
                                lifted_ciphertext: material.lifted_ciphertext.at(local),
                                level: material.level,
                                context: material.context,
                                id: material.id,
                                rhs_id: material.rhs_id,
                            })
                        })
                        .collect(),
                )
                .unwrap();
                public_blocks.push((selector_families[block], (rhs_values, range.clone())));
                online_blocks.push((
                    selector_families[block],
                    (
                        Family::pack(vec![RnsBggPublicRhs::Identity(rhs_id); positions.len()])
                            .unwrap(),
                        range,
                    ),
                ));
            }
            let public_inputs = BTreeMap::from([(family_input, (initial_key.clone(), None))]);
            let online_inputs = BTreeMap::from([(family_input, (initial.clone(), None))]);
            let public_rhs = public_blocks.into_iter().collect::<BTreeMap<_, _>>();
            let online_rhs = online_blocks.into_iter().collect::<BTreeMap<_, _>>();
            let public_values = wire_families
                .iter()
                .copied()
                .zip(public_members.iter().cloned())
                .collect::<BTreeMap<_, _>>();
            let online_values = wire_families
                .iter()
                .copied()
                .zip(online_members.iter().cloned())
                .collect::<BTreeMap<_, _>>();
            let mut ranges = ProgramFamilyBindings::new();
            for (block, (start, count)) in
                program.active_block_ranges(&pbc).unwrap().1.into_iter().enumerate()
            {
                let crate::program::ProgramStep::Family { id, .. } = &family_program.steps()[block]
                else {
                    unreachable!()
                };
                ranges.bind(*id, FamilyRange::bounded(start, count, pbc.bucket_width).unwrap());
            }
            let prepared = crate::scheme::PreparedProgram::new(
                scheme.clone(),
                family_program.clone(),
                &crate::program::ProgramBindings::new()
                    .inputs(public_inputs.into_iter())
                    .rhs_family_values(public_rhs.into_iter())
                    .wire_families(public_values.into_iter()),
                ranges,
                None,
            )
            .unwrap();
            let evaluated = prepared
                .evaluate(
                    &crate::program::ProgramBindings::new()
                        .inputs(online_inputs.into_iter())
                        .rhs_family_values(online_rhs.into_iter())
                        .wire_families(online_values.into_iter()),
                )
                .unwrap();
            let output_key =
                RnsBggPublicProgramValue::Scalar(prepared.public_values()[&family_output].clone());
            let output = RnsBggEncodingProgramValue::Scalar(evaluated[&family_output].clone());
            let readout_artifacts =
                program.prepare_readout(&scheme, &output_key, &readout_material).unwrap();
            let readout_public =
                program.project_readout(&scheme, &output_key, &readout_artifacts, &()).unwrap();
            let rounded = program
                .evaluate_readout(&scheme, &output, &readout_artifacts, &online_terminal_bank)
                .unwrap();
            let exponent =
                selected.iter().map(|(bucket, slot, _)| label(*bucket, *slot)).sum::<usize>();
            let exponent = exponent % (2 * rounding_values.len());
            seen_halves[exponent / rounding_values.len()] = true;
            let expected = if exponent < rounding_values.len() {
                rounding_values[exponent].clone()
            } else {
                BigInt::from(1u8) - &rounding_values[exponent - rounding_values.len()]
            };
            let expected_row = crate::rns_bgg::RnsBggRow::new([[
                scheme.scalar(3, expected).unwrap(),
                scheme.scalar(3, 0).unwrap(),
            ]])
            .unwrap();
            // The affine half is inside the fresh target, not applied to the error.
            let terminal_error = rounded.values.scalar().unwrap().clone() -
                terminal_s.clone() * readout_public.values.scalar().unwrap().clone() -
                expected_row;
            graph = graph.output(format!("terminal-error-{choice}"), terminal_error).unwrap();
            let RnsBggPublicProgramValue::Scalar((output_key, _)) = output_key else {
                panic!("scalar key")
            };
            let RnsBggEncodingProgramValue::Scalar((output, _)) = output else {
                panic!("scalar encoding")
            };
            let ideal = exponent_layout.mono(&low, &BigInt::from(exponent)).unwrap();
            let error = output.row - s.clone() * output_key.matrix +
                scheme.scalar(2, parameters.scale(2).unwrap().clone()).unwrap() *
                    ideal *
                    r.clone();
            graph = graph.output(format!("error-{choice}"), error).unwrap();
        }
        assert_eq!(seen_halves, [true, true], "PRF cases must exercise both native halves");
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().iter().cloned()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (name, value) in results.outputs {
            let (q, limit) = if name.starts_with("terminal-error-") {
                (terminal_q.as_ref(), &terminal_bound.error)
            } else {
                (q.as_ref(), &bound.error)
            };
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            for polynomial in matrix
                .block_entries(0..matrix.row_size(), 0..matrix.col_size())
                .into_iter()
                .flatten()
            {
                for coefficient in polynomial.coeffs() {
                    let absolute: BigUint =
                        coefficient.value().clone().min(q - coefficient.value());
                    assert!(&absolute <= limit, "{name}: {absolute} > {}", limit);
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod family_tests {
    use super::*;
    use crate::rns_bgg::{RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest};
    use mxx_dsl::DslContext;
    use mxx_primitives::{element::PolyElem, poly::Poly};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    pub(crate) fn composed_rhs_fixture()
    -> (mxx_ir_core::ValidatedGraph, RnsBggParameters, RnsBggNoiseBound) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1, 4],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 6,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let hash_key = scheme.ring(0).unwrap().bytes_input("hash-key", 32);
        let key = scheme
            .sample_input_key(0, hash_key.clone(), b"composed-input".as_slice(), rand::random())
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                1,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (4 as usize).trailing_zeros(),
                },
                hash_key.clone(),
                b"composed-mask".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_bank = scheme.prepare_online_masks(&secret.mask, &bank).unwrap();
        // The active slots are 1 and 2. Nonzero selectors outside that range
        // must not enter the RHS composition. Both active ciphertext errors do.
        let selectors = RnsBggRhsPlaintextFamily {
            payloads: Family::pack(
                [1u8, 0, 1, 1].into_iter().map(|bit| scheme.scalar(1, bit).unwrap()).collect(),
            )
            .unwrap(),
            id: rand::random(),
        };
        let table = LutTable::unary(4, 2, vec![0, 1, 0, 1]).unwrap();
        let rhs_bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 },
                hash_key.clone(),
                b"composed-rhs-bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let material = scheme
            .sample_lut_material_family(
                &secret.payload,
                None,
                Some(&selectors),
                &rhs_bank,
                4.into(),
                hash_key.clone(),
                b"composed-slots".as_slice(),
                rand::random(),
            )
            .unwrap();
        let lut_material = scheme
            .sample_lut_material(
                &secret.payload,
                Some(&table),
                None,
                &bank,
                hash_key,
                b"shared-composed-lut".as_slice(),
                rand::random(),
            )
            .unwrap();
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let layout = crate::layout::ExpLayout { stride_log2: 0, width_log2: 2 };
        let q = parameters.level(2).unwrap().modulus();
        let modulus = BigInt::from(q.as_ref().clone());

        let mut graph = DslContext::new("rns_bgg-composed-rhs-wrap");
        for label in 0..2 {
            let labels = Family::pack(vec![
                layout.mono(&scheme.ring(1).unwrap(), &1.into()).unwrap(),
                layout.mono(&scheme.ring(1).unwrap(), &BigInt::from(label)).unwrap(),
            ])
            .unwrap();
            let mut composed =
                scheme.compose_lut_material(&material, &labels, &range, rand::random()).unwrap();
            // The table-free identity branch is exactly a fixed RHS GSW.
            // Preserve the existing slot composition; apply it as a separate
            // private product before consuming the shared unary table.
            assert_eq!(composed.branches.len(), 1);
            let composed = RnsBggPrivateRhs {
                ciphertext: composed.branches.remove(0).1,
                lifted_ciphertext: (),
                level: composed.level,
                context: composed.context,
                id: composed.id,
                rhs_id: composed.rhs_id.unwrap(),
            };
            let composed = scheme.prepare_private_rhs(&composed).unwrap();
            let product = scheme.prepare_mul_private(&key, &composed, rand::random()).unwrap();
            let prepared =
                scheme.prepare_lut(product.output_key(), &lut_material, rand::random()).unwrap();
            for input in 0..2 {
                let payload = layout.mono(&scheme.ring(0).unwrap(), &BigInt::from(input)).unwrap();
                let encoding = scheme
                    .encode(
                        &secret.mask,
                        &key,
                        &scheme.prepare_input(&secret.payload, &key, payload).unwrap(),
                    )
                    .unwrap();
                let multiplied = scheme.evaluate_mul_private(&encoding, &product).unwrap();
                let output = scheme.evaluate_lut(&multiplied, &prepared, &online_bank).unwrap();
                let expected = layout
                    .mono(&scheme.ring(2).unwrap(), &BigInt::from((input + label) % 2))
                    .unwrap();
                let error = output.row -
                    secret.mask.row.clone().reduce_modulus(modulus.clone()) *
                        prepared.output_key().matrix.clone() +
                    expected * secret.payload.row.clone().reduce_modulus(modulus.clone());
                graph = graph.output(format!("error-{label}-{input}"), error).unwrap();
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let (noise_material, noise_rhs) =
            crate::rns_bgg::noise::RnsBggGateNoiseMaterial::from_one_hot_labels(
                0,
                None,
                &[
                    crate::rns_bgg::noise::RnsBggNoiseAction::new(
                        crate::rns_bgg::noise::stable_source_id(&("encoding-one-hot", 0usize)),
                        crate::noise_utils::RightActionNoiseGain::unit(),
                    ),
                    crate::rns_bgg::noise::RnsBggNoiseAction::new(
                        crate::rns_bgg::noise::stable_source_id(&("encoding-one-hot", 1usize)),
                        crate::noise_utils::RightActionNoiseGain::unit(),
                    ),
                ],
            )
            .unwrap();
        let product_bound = parameters
            .private_mul_noise_bound(
                0,
                &RnsBggNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
                &noise_material.fresh_error_gain.l1,
                noise_rhs.payload_l1,
            )
            .unwrap();
        let bound = parameters
            .lut_noise_bound(1, &product_bound, 4, &1u8.into(), product_bound.payload_l1.clone())
            .unwrap();
        (graph, parameters, bound)
    }

    #[test]
    fn test_rns_bgg_composed_rhs_wraps_selected_public_label() {
        let (graph, parameters, bound) = composed_rhs_fixture();
        let q = parameters.level(2).unwrap().modulus();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().iter().cloned()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            for entry in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in entry.coeffs() {
                    assert!(
                        coefficient.value().clone().min(q.as_ref() - coefficient.value()) <=
                            bound.error
                    );
                }
            }
        }
    }

    pub(crate) fn structural_selection_fixture()
    -> (mxx_ir_core::ValidatedGraph, RnsBggParameters, RnsBggNoiseBound) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            terminal_value_bound: 3u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let hash_key = scheme.ring(0).unwrap().bytes_input("hash-key", 32);
        let key = scheme
            .sample_input_key(0, hash_key.clone(), b"input".as_slice(), rand::random())
            .unwrap();
        let payload = scheme.ring(0).unwrap().polynomial([0.into(), 1.into()]);
        let input = scheme
            .encode(
                &secret.mask,
                &key,
                &scheme.prepare_input(&secret.payload, &key, payload.clone()).unwrap(),
            )
            .unwrap();
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let input_key = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(2, |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: 0,
            id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let input = RnsBggEncodingFamily {
            rows: ProgramValue::Family(mxx_dsl::parallel(2, |_| Ok(input.row.clone())).unwrap()),
            level: 0,
            key_id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        // The last selector is outside the active range and must not affect
        // either output payload or the active-member aggregation.
        let rhs = RnsBggRhsPlaintextFamily {
            payloads: Family::pack(
                [0u8, 1, 0, 1].into_iter().map(|bit| scheme.scalar(1, bit).unwrap()).collect(),
            )
            .unwrap(),
            id: rand::random(),
        };
        let material = scheme
            .sample_private_rhs_family(
                &secret.payload,
                0,
                &rhs,
                hash_key.clone(),
                b"selectors".as_slice(),
                rand::random(),
            )
            .unwrap();
        let material = scheme.prepare_private_rhs_family(&material, &range).unwrap();
        let prepared = scheme
            .prepare_mul_private_family(
                &input_key,
                &ProgramValue::Family(material.clone()),
                rand::random(),
            )
            .unwrap();
        let selected = scheme.evaluate_mul_private_family(&input, &prepared).unwrap();
        let public_key = scheme
            .sample_input_key(1, hash_key.clone(), b"public".as_slice(), rand::random())
            .unwrap();
        let public_family = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(2, |_| Ok(public_key.matrix.clone())).unwrap(),
            ),
            level: 1,
            id: public_key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let weights =
            Family::pack(vec![scheme.scalar(1, 2u8).unwrap(), scheme.scalar(1, 3u8).unwrap()])
                .unwrap();
        let public_rows = mxx_dsl::parallel(2, |index| {
            scheme
                .encode(
                    &secret.mask,
                    &public_key,
                    &scheme
                        .prepare_input(&secret.payload, &public_key, weights.at(&index))
                        .unwrap(),
                )
                .map(|value| value.row)
                .map_err(|_| DslError::Schema)
        })
        .unwrap();
        let public_encodings = RnsBggEncodingFamily {
            rows: ProgramValue::Family(public_rows),
            level: 1,
            key_id: public_key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let weighted_prepared = scheme
            .prepare_linear_family(
                prepared.output_key(),
                &public_family,
                Some(ProgramValue::Family(weights.clone())),
                rand::random(),
            )
            .unwrap();
        let weighted = scheme
            .evaluate_linear_family(&selected, &public_encodings, &weighted_prepared)
            .unwrap();
        let reduction =
            scheme.prepare_reduce_family(weighted_prepared.output_key(), rand::random()).unwrap();
        let output = scheme.evaluate_reduce_family(&weighted, &reduction).unwrap();
        use crate::{
            program::{
                AttributeKnowledge, ExponentLutProgramBuilder, ProgramBindings,
                ProgramFamilyBindings, ProgramFamilyId, ProgramGateInstance, ProgramWireType,
            },
            scheme::PreparedProgram,
        };
        let mut builder = ExponentLutProgramBuilder::new();
        builder
            .set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize)
            .unwrap();
        let input_id = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let rhs_family = builder.rhs_family(2).unwrap();
        let member_family = builder.wire_family(2).unwrap();
        let output_id = builder
            .select(builder.input_wire(input_id).unwrap(), rhs_family, member_family, 4)
            .unwrap();
        let repeated_output_id = builder
            .select(builder.input_wire(input_id).unwrap(), rhs_family, member_family, 4)
            .unwrap();
        builder.output(output_id).unwrap();
        builder.output(repeated_output_id).unwrap();
        let program = builder.build().unwrap();
        let public_inputs = BTreeMap::from([(input_id, (key.clone(), None))]);
        let public_members = BTreeMap::from([(
            member_family,
            (
                Family::pack(
                    (0..2).map(|index| (public_key.clone(), Some(weights.at(index)))).collect(),
                )
                .unwrap(),
                range.clone(),
            ),
        )]);
        let rhs_values = Family::pack(
            (0..2)
                .map(|index| {
                    RnsBggPublicRhs::Fixed(RnsBggPrivateRhs {
                        ciphertext: material.ciphertext.at(index),
                        lifted_ciphertext: material.lifted_ciphertext.at(index),
                        level: material.level,
                        context: material.context,
                        id: material.id,
                        rhs_id: material.rhs_id,
                    })
                })
                .collect(),
        )
        .unwrap();
        let rhs_bindings = BTreeMap::from([(rhs_family, (rhs_values, range.clone()))]);
        let online_rhs_bindings = BTreeMap::from([(
            rhs_family,
            (Family::pack(vec![RnsBggPublicRhs::Identity(rhs.id); 2]).unwrap(), range.clone()),
        )]);
        let at = ProgramGateInstance::FamilyGate {
            step: 0,
            gate: 0,
            family: match &program.steps()[0] {
                crate::program::ProgramStep::Family { id, .. } => *id,
                _ => unreachable!(),
            },
        };
        let repeated_at = ProgramGateInstance::FamilyGate {
            step: 1,
            gate: 0,
            family: match &program.steps()[1] {
                crate::program::ProgramStep::Family { id, .. } => *id,
                _ => unreachable!(),
            },
        };
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), range.clone());
        ranges.bind(ProgramFamilyId::from_index(1), range.clone());
        let program = PreparedProgram::new(
            scheme.clone(),
            program,
            &ProgramBindings::new()
                .inputs(public_inputs.iter().map(|(&id, value)| (id, value.clone())))
                .rhs_family_values(rhs_bindings.iter().map(|(&id, value)| (id, value.clone())))
                .wire_families(public_members.iter().map(|(&id, value)| (id, value.clone()))),
            ranges,
            None,
        )
        .unwrap();
        for gate in [at, repeated_at] {
            let ProgramGateInstance::FamilyGate { family, .. } = gate else { unreachable!() };
            for index in [0, 1] {
                let cache = program.family_caches[&family].at(index);
                let PreparedGate::MulPrivate(prepared) = &cache.gates[&gate] else {
                    panic!("prepared scalar private multiplication inside family")
                };
                assert_eq!(prepared.input.matrix.flatten(), key.matrix.flatten());
            }
        }
        for (step, output) in [(0, output_id), (1, repeated_output_id)] {
            let reduction = &program.cache().gates[&ProgramGateInstance::FamilyReduction { step }];
            let output_key = &program.public_values()[&output].0;
            assert_eq!(reduction.flatten(), output_key.matrix.flatten());
            let schema =
                serde_json::from_slice(&serde_json::to_vec(&reduction.schema()).unwrap()).unwrap();
            let restored: PreparedGate =
                PreparedGate::from_values(&schema, &reduction.flatten()).unwrap();
            assert!(restored.schema() == reduction.schema());
            assert_eq!(restored.flatten(), output_key.matrix.flatten());
        }
        let online_inputs = BTreeMap::from([(
            input_id,
            (
                RnsBggEncoding {
                    row: input.rows.at(0),
                    level: 0,
                    key_id: key.id,
                    context: scheme.context,
                },
                None,
            ),
        )]);
        let online_members = BTreeMap::from([(
            member_family,
            (
                Family::pack(
                    (0..2)
                        .map(|index| {
                            (
                                RnsBggEncoding {
                                    row: public_encodings.rows.at(index),
                                    level: 1,
                                    key_id: public_key.id,
                                    context: scheme.context,
                                },
                                Some(weights.at(index)),
                            )
                        })
                        .collect(),
                )
                .unwrap(),
                range.clone(),
            ),
        )]);
        let program_values = program
            .evaluate(
                &ProgramBindings::new()
                    .inputs(online_inputs.iter().map(|(&id, value)| (id, value.clone())))
                    .rhs_family_values(
                        online_rhs_bindings.iter().map(|(&id, value)| (id, value.clone())),
                    )
                    .wire_families(online_members.iter().map(|(&id, value)| (id, value.clone()))),
            )
            .unwrap();
        let (program_output, _) = &program_values[&output_id];
        let (program_key, _) = &program.public_values()[&output_id];
        let (repeated_output, _) = &program_values[&repeated_output_id];
        let repeated_difference = repeated_output.row.clone() - program_output.row.clone();
        let program_difference = program_output.row.clone() - output.row.clone();
        let public_difference = program_key.matrix.clone() - reduction.output_key().matrix.clone();
        let q = parameters.level(1).unwrap().modulus();
        let modulus = BigInt::from(q.as_ref().clone());
        let expected = scheme.scalar(1, 2u8).unwrap() * payload.reduce_modulus(modulus.clone());
        let physical_error = output.row -
            secret.mask.row.clone().reduce_modulus(modulus.clone()) *
                reduction.output_key().matrix.clone() +
            expected * secret.payload.row.clone().reduce_modulus(modulus);

        let graph = DslContext::new("rns_bgg-structural-selection")
            .output("repeated-difference", repeated_difference)
            .unwrap()
            .output("program-difference", program_difference)
            .unwrap()
            .output("public-difference", public_difference)
            .unwrap()
            .output("error", physical_error)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let bound = parameters
            .private_mul_noise_bound(
                0,
                &RnsBggNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
                &1u8.into(),
                1u8.into(),
            )
            .unwrap();
        // Errors are weighted by both public coefficients (2 + 3), even though
        // the second slot's selected plaintext is zero.
        let bound = bound.mul_public(&BigUint::from(5u8));
        let mut wrong = input.clone();
        wrong.range = FamilyRange::bounded(0, 2, 4).unwrap();
        assert!(scheme.evaluate_mul_private_family(&wrong, &prepared).is_err());
        (graph, parameters, bound)
    }
    #[test]
    fn test_rns_bgg_structural_selection_respects_active_range() {
        let (graph, parameters, bound) = structural_selection_fixture();
        let q = parameters.level(1).unwrap().modulus();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (name, value) in &results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("error polynomial") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in polynomial.coeffs() {
                    if name.contains("difference") {
                        assert_eq!(coefficient.value(), &BigUint::from(0u8));
                    }
                    assert!(
                        coefficient.value().clone().min(q.as_ref() - coefficient.value()) <=
                            bound.error
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod program_tests {
    use super::*;
    use crate::{
        layout::ExpLayout,
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, LutTable, ProgramBindings,
            ProgramWireType,
        },
        rns_bgg::{RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest, RnsBggRhsPlaintext},
        scheme::PreparedProgram,
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        element::PolyElem,
        poly::{Poly, PolyParams},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};

    #[test]
    fn test_rns_bgg_scheme_program_all_gates_and_rhs_binding() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1, 4],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 6,
            terminal_value_bound: 4u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.test_secrets().unwrap();
        let hash_key = scheme.ring(0).unwrap().bytes_input("hash-key", 32);
        let input_key = scheme
            .sample_input_key(0, hash_key.clone(), b"input".as_slice(), rand::random())
            .unwrap();
        let weight_key = scheme
            .sample_input_key(2, hash_key.clone(), b"weight".as_slice(), rand::random())
            .unwrap();
        let weight = scheme.scalar(2, 2u8).unwrap();
        let weight_encoding = scheme
            .encode(
                &secret.mask,
                &weight_key,
                &scheme.prepare_input(&secret.payload, &weight_key, weight.clone()).unwrap(),
            )
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                1,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (4 as usize).trailing_zeros(),
                },
                hash_key.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_bank = scheme.prepare_online_masks(&secret.mask, &bank).unwrap();
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let table = LutTable::binary(2, 2, 2, vec![1, 1, 1, 0]).unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(layout, n as usize).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let weight_input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
            .unwrap();
        let rhs_family = builder.rhs_family(2).unwrap();
        let rhs_input = builder.rhs_input(rhs_family, 2, 2).unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let mapped = builder.lut(builder.input_wire(input).unwrap(), lut, Some(rhs_input)).unwrap();
        let doubled = builder.add(mapped, mapped).unwrap();
        let output =
            builder.mul_public(doubled, builder.input_wire(weight_input).unwrap()).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let gate = ProgramGateInstance::Gate { step: 1 };
        let crate::program::Gate::Lut { lut_id: lut, .. } = program.gates()[1] else {
            panic!("unary LUT")
        };
        let table = program.lut(lut).unwrap();
        let material = scheme
            .sample_lut_material(
                &secret.payload,
                Some(table),
                None,
                &bank,
                hash_key.clone(),
                b"shared-unary-lut".as_slice(),
                rand::random(),
            )
            .unwrap();
        assert!(
            scheme
                .sample_lut_material(
                    &secret.payload,
                    Some(table),
                    Some(RnsBggRhsPlaintext {
                        payload: scheme.scalar(2, 1u8).unwrap(),
                        id: rand::random()
                    }),
                    &bank,
                    hash_key.clone(),
                    b"rejected-fused-lut".as_slice(),
                    rand::random()
                )
                .is_err()
        );
        let mut rejected = material.clone();
        rejected.rhs_id = Some(rand::random());
        let intermediate_key = scheme
            .sample_input_key(1, hash_key.clone(), b"rejected-input".as_slice(), rand::random())
            .unwrap();
        assert!(scheme.prepare_lut(&intermediate_key, &rejected, rand::random()).is_err());
        let online_helpers = BTreeMap::from([(
            gate,
            ProgramValue::Scalar(RnsBggOnlineLutHelpers {
                binding: material.binding(),
                masks: online_bank.clone(),
            }),
        )]);
        let helpers = BTreeMap::from([(gate, ProgramValue::Scalar(material))]);
        let q = parameters.level(2).unwrap().modulus();
        let modulus = BigInt::from(q.as_ref().clone());
        let s = secret.mask.row.clone().reduce_modulus(modulus.clone());
        let r = secret.payload.row.clone().reduce_modulus(modulus);
        let t_inverse =
            mxx_primitives::utils::mod_inverse_biguints(&2u8.into(), q.as_ref()).unwrap();
        let mut graph = DslContext::new("rns_bgg-generic-program");
        for y in 0..2 {
            let rhs_id = rand::random();
            let rhs_material = scheme
                .sample_private_rhs(
                    &secret.payload,
                    0,
                    RnsBggRhsPlaintext {
                        payload: layout
                            .mono(&scheme.ring(1).unwrap(), &BigInt::from(2 * y))
                            .unwrap(),
                        id: rhs_id,
                    },
                    hash_key.clone(),
                    format!("rhs-{y}").into_bytes(),
                    rand::random(),
                )
                .unwrap();
            let rhs_material = scheme.prepare_private_rhs(&rhs_material).unwrap();
            let rhs = BTreeMap::from([(rhs_input, RnsBggPublicRhs::Identity(rhs_id))]);
            let public_rhs = BTreeMap::from([(rhs_input, RnsBggPublicRhs::Fixed(rhs_material))]);
            let public_inputs = BTreeMap::from([
                (input, (input_key.clone(), None)),
                (weight_input, (weight_key.clone(), Some(weight.clone()))),
            ]);
            let prepared = PreparedProgram::new(
                scheme.clone(),
                program.clone(),
                &ProgramBindings::new()
                    .inputs(public_inputs.iter().map(|(&id, value)| (id, value.clone())))
                    .rhs_values(public_rhs.iter().map(|(&id, value)| (id, value.clone())))
                    .gate_helpers(helpers.iter().map(|(&id, value)| (id, value.clone()))),
                ProgramFamilyBindings::new(),
                None,
            )
            .unwrap();
            let (output_value, _) = &prepared.public_values()[&output];
            let output_key = &output_value;
            for x in 0..2 {
                let encoding = scheme
                    .encode(
                        &secret.mask,
                        &input_key,
                        &scheme
                            .prepare_input(
                                &secret.payload,
                                &input_key,
                                layout.mono(&scheme.ring(0).unwrap(), &BigInt::from(x)).unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap();
                let inputs = BTreeMap::from([
                    (input, (encoding, None)),
                    (weight_input, (weight_encoding.clone(), Some(weight.clone()))),
                ]);
                let values = prepared
                    .evaluate(
                        &ProgramBindings::new()
                            .inputs(inputs.iter().map(|(&id, value)| (id, value.clone())))
                            .rhs_values(rhs.iter().map(|(&id, value)| (id, value.clone())))
                            .gate_helpers(
                                online_helpers.iter().map(|(&id, value)| (id, value.clone())),
                            ),
                    )
                    .unwrap();
                let (actual, actual_plaintext) = &values[&output];
                assert!(actual_plaintext.is_none());
                let mut wrong_helpers = online_helpers.clone();
                let ProgramValue::Scalar(wrong_helper) = wrong_helpers.get_mut(&gate).unwrap()
                else {
                    unreachable!()
                };
                wrong_helper.binding.material_id[0] ^= 1;
                assert!(
                    prepared
                        .evaluate(
                            &ProgramBindings::new()
                                .inputs(inputs.iter().map(|(&id, value)| (id, value.clone())))
                                .rhs_values(rhs.iter().map(|(&id, value)| (id, value.clone())))
                                .gate_helpers(
                                    wrong_helpers.iter().map(|(&id, value)| (id, value.clone())),
                                )
                        )
                        .is_err(),
                    "a helper from another artifact must not match by table/RHS alone"
                );
                let wrong_rhs =
                    BTreeMap::from([(rhs_input, RnsBggPublicRhs::Identity(rand::random()))]);
                assert!(
                    prepared
                        .evaluate(
                            &ProgramBindings::new()
                                .inputs(inputs.iter().map(|(&id, value)| (id, value.clone())))
                                .rhs_values(
                                    wrong_rhs.iter().map(|(&id, value)| (id, value.clone()))
                                )
                                .gate_helpers(
                                    online_helpers.iter().map(|(&id, value)| (id, value.clone())),
                                )
                        )
                        .is_err()
                );
                let expected = scheme.scalar(2, 4u8).unwrap() *
                    layout
                        .mono(&scheme.ring(2).unwrap(), &BigInt::from(table.values()[x + 2 * y]))
                        .unwrap();
                let error = actual.row.clone() - s.clone() * output_key.matrix.clone() +
                    expected * r.clone();
                graph = graph
                    .output(
                        format!("error-{x}-{y}"),
                        scheme.scalar(2, t_inverse.clone()).unwrap() * error,
                    )
                    .unwrap();
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().to_vec()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let product_bound = parameters
            .private_mul_noise_bound(
                0,
                &RnsBggNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
                &1u8.into(),
                1u8.into(),
            )
            .unwrap();
        let lut_bound =
            parameters.lut_noise_bound(1, &product_bound, 4, &1u8.into(), 1u8.into()).unwrap();
        let bound = lut_bound.add(&lut_bound).mul_public(&BigUint::from(2u8));
        for value in results.outputs.values() {
            let RuntimeValue::Matrix(matrix) = value else { panic!("error matrix") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in polynomial.coeffs() {
                    assert!(
                        coefficient.value().clone().min(q.as_ref() - coefficient.value()) <=
                            bound.error
                    );
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod terminal_tests {
    use super::*;
    use crate::rns_bgg::{
        RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest, noise::RnsBggTerminalNoise,
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, ValidatedGraph};
    use mxx_primitives::{element::PolyElem, poly::Poly};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    pub(crate) fn terminal_fixture() -> (ValidatedGraph, RnsBggParameters, RnsBggTerminalNoise) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let weight = BigInt::from(1u8) << 60usize;
        let table_a = RowLutTable::new(
            vec![
                [vec![], vec![]],
                [vec![], vec![0.into(), 0.into(), weight.clone()]],
                [vec![0.into(), -weight.clone()], vec![]],
                [vec![3.into()], vec![weight.clone(), (-1).into()]],
            ],
            None,
        )
        .unwrap();
        let table_b = RowLutTable::new(
            vec![
                [vec![], vec![]],
                [vec![0.into(), 0.into(), 0.into(), &weight + 1u8], vec![]],
                [vec![], vec![]],
                [vec![0.into(), 0.into(), 0.into(), &weight + 1u8], vec![]],
            ],
            None,
        )
        .unwrap();
        let value_bound = table_a.coefficient_bound() + table_b.coefficient_bound();
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![4],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 3,
            max_crt_depth: 12,
            terminal_value_bound: value_bound.clone(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let secret = scheme.test_secrets().unwrap();
        let key_a = scheme
            .sample_input_key(0, seed.clone(), b"input-a".as_slice(), rand::random())
            .unwrap();
        let key_b = scheme
            .sample_input_key(0, seed.clone(), b"input-b".as_slice(), rand::random())
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (4 as usize).trailing_zeros(),
                },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_bank = scheme.prepare_online_masks(&secret.mask, &bank).unwrap();
        let material_a = scheme
            .sample_terminal_material(
                &secret.payload,
                Some(&table_a),
                None,
                None,
                &bank,
                seed.clone(),
                b"target-a".as_slice(),
                rand::random(),
            )
            .unwrap();
        let material_b = scheme
            .sample_terminal_material(
                &secret.payload,
                Some(&table_b),
                Some(RnsBggRhsPlaintext {
                    payload: low.polynomial([0.into(), 1.into()]),
                    id: rand::random(),
                }),
                None,
                &bank,
                seed,
                b"target-b".as_slice(),
                rand::random(),
            )
            .unwrap();
        use crate::readout::{PreparedTerminal, TerminalError};
        let public_a = (key_a.clone(), None);
        let public_b = (key_b.clone(), None);
        let prepared =
            PreparedTerminal::new(&scheme, &[(&public_a, &material_a), (&public_b, &material_b)])
                .unwrap();
        for (term, material) in prepared.prepared_terms().iter().zip([&material_a, &material_b]) {
            for (branch, lifted) in term.branches.iter().zip(&material.lifted_ciphertexts) {
                assert_eq!(
                    branch.lifted_ciphertext.flatten(),
                    lifted.flatten(),
                    "terminal preparation reuses the sampler's cached high-base lift"
                );
            }
        }
        let public = prepared.public();
        assert_eq!(public.level(), 1);
        let bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: 1u8.into(),
        };
        let gain = BigUint::from(1u8);
        let noise = parameters
            .terminal_noise_bound(0, [(&bound, 4, &gain), (&bound, 4, &gain)], value_bound)
            .unwrap();
        assert_eq!(noise.branches, 8);
        assert!(parameters.terminal_noise_fits(1, &noise).unwrap());
        let unweighted = parameters
            .terminal_noise_bound(0, [(&bound, 4, &gain), (&bound, 4, &gain)], 1u8.into())
            .unwrap();
        assert_eq!(noise.error, unweighted.error, "weight changes range, not fresh error");
        let single = parameters.terminal_noise_bound(0, [(&bound, 4, &gain)], 1u8.into()).unwrap();
        let ordinary = parameters.lut_noise_bound(0, &bound, 4, &gain, 1u8.into()).unwrap();
        assert_eq!(single.error, ordinary.error);
        assert!(noise.error <= single.error * 2u8);
        let q = parameters.level(1).unwrap().modulus();

        let s = secret.mask.row.clone().reduce_modulus(BigInt::from(q.as_ref().clone()));
        let mut graph = DslContext::new("rns_bgg-weighted-terminal-batch");
        for x in 0..4 {
            for y in 0..3 {
                let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
                let a = scheme
                    .encode(
                        &secret.mask,
                        &key_a,
                        &scheme
                            .prepare_input(
                                &secret.payload,
                                &key_a,
                                layout.mono(&high, &x.into()).unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap();
                let b = scheme
                    .encode(
                        &secret.mask,
                        &key_b,
                        &scheme
                            .prepare_input(
                                &secret.payload,
                                &key_b,
                                layout.mono(&high, &y.into()).unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap();
                let a = (a, None);
                let b = (b, None);
                let actual = prepared.evaluate(&[(&a, &online_bank), (&b, &online_bank)]).unwrap();
                assert_eq!(actual.level(), 1);
                // Expected values use the public table entries directly, not
                // the automorphism decomposition used to generate ciphertexts.
                let row = |values: &[Vec<BigInt>; 2]| {
                    RnsBggRow::new([values.each_ref().map(|p| {
                        if p.is_empty() {
                            low.zero((1, 1))
                        } else {
                            low.polynomial(p.iter().cloned().map(IntExpr::constant))
                        }
                    })])
                    .unwrap()
                };
                let expected = row(&table_a.values[x]) + row(&table_b.values[y + 1]);
                let error = actual.row().clone() - s.clone() * public.matrix().clone() - expected;
                graph = graph.output(format!("error-{x}-{y}"), error).unwrap();
            }
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        fn switch_count(
            graph: &mxx_ir_core::Graph,
            scope: &mxx_ir_core::graph::FrozenGraphScopeId,
        ) -> usize {
            graph
                .scope(scope)
                .unwrap()
                .nodes()
                .iter()
                .enumerate()
                .map(|(index, node)| {
                    use mxx_ir_core::node::NodeKind;
                    let own = usize::from(matches!(node.kind(), NodeKind::BlockModSwitch { .. }));
                    let Some(child) =
                        graph.child_scope_id(scope, mxx_ir_core::types::NodeId(index as u64))
                    else {
                        return own;
                    };
                    let count = match node.kind() {
                        NodeKind::ParallelLoop(spec) => {
                            spec.count.evaluate(&ParamEnv::default()).unwrap()
                        }
                        NodeKind::SequentialLoop(spec) => {
                            spec.count.evaluate(&ParamEnv::default()).unwrap()
                        }
                        _ => 1.into(),
                    };
                    own + usize::try_from(count).unwrap() * switch_count(graph, &child)
                })
                .sum()
        }
        let switches = switch_count(&graph.source, &mxx_ir_core::graph::FrozenGraphScopeId::Root);
        assert_eq!(
            switches,
            2 * 4 * 4 + 4 + 12 * 2,
            "internal branch switches plus one aggregate public matrix and one aggregate row per case"
        );
        assert!(matches!(PreparedTerminal::new(&scheme, &[]), Err(TerminalError::Empty)));
        assert!(matches!(prepared.evaluate(&[]), Err(TerminalError::InputCount)));
        let prepared_a = scheme.prepare_terminal(&key_a, &material_a).unwrap();
        let prepared_b = scheme.prepare_terminal(&key_b, &material_b).unwrap();
        let wrong_input = scheme
            .encode(
                &secret.mask,
                &key_a,
                &scheme.prepare_input(&secret.payload, &key_a, high.identity(1)).unwrap(),
            )
            .unwrap();
        assert!(scheme.evaluate_terminal(&wrong_input, &prepared_b, &online_bank).is_err());
        let incomplete = scheme.evaluate_terminal(&wrong_input, &prepared_a, &online_bank).unwrap();
        assert!(scheme.finalize_terminal_encoding(incomplete, &public).is_err());
        let mut wrong = prepared_b.public_accumulator();
        wrong.context[0] ^= 1;
        assert!(prepared_a.public_accumulator().combine(wrong).is_err());
        assert!(RowLutTable::<2>::new(vec![], None).is_err());
        (graph, parameters, noise)
    }

    pub(crate) fn routed_terminal_fixture()
    -> (ValidatedGraph, RnsBggParameters, RnsBggTerminalNoise) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(8);
        let weight = BigInt::from(1u8) << 40usize;
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 3,
            max_crt_depth: 12,
            terminal_value_bound: weight.magnitude().clone(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secrets = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let bank = scheme
            .sample_mask_bank(0, layout, seed.clone(), b"routed-bank".as_slice(), rand::random())
            .unwrap();
        let masks = scheme.prepare_online_masks(&secrets.mask, &bank).unwrap();
        let key = scheme
            .sample_input_key(0, seed.clone(), b"routed-key".as_slice(), rand::random())
            .unwrap();
        let input = scheme
            .encode(
                &secrets.mask,
                &key,
                &scheme
                    .prepare_input(&secrets.payload, &key, high.polynomial([0.into(), 1.into()]))
                    .unwrap(),
            )
            .unwrap();
        let table =
            RowLutTable::new(vec![[vec![], vec![]], [vec![1.into()], vec![]]], None).unwrap();
        // Distinct destinations and signs exercise coefficient routing inside
        // one structural body, with the ordinary unweighted error bound.
        let destinations = Family::pack(vec![
            RnsBggRow::new([[low.polynomial([weight.clone().into()]), low.zero((1, 1))]]).unwrap(),
            RnsBggRow::new([[
                low.zero((1, 1)),
                low.polynomial([0.into(), (-weight.clone()).into()]),
            ]])
            .unwrap(),
        ])
        .unwrap();
        let q = BigInt::from(parameters.level(1).unwrap().modulus().as_ref().clone());
        let mask_secret = secrets.mask.row.clone().reduce_modulus(q.clone());

        let errors = mxx_dsl::parallel(2, |index| {
            let destination = destinations.at(&index);
            let transform = RnsBggMatrix::from_rows(
                destination.clone(),
                RnsBggRow::new([[low.zero((1, 1)), low.zero((1, 1))]])?,
            );
            let mut tag = HashTag::from(b"routed-terminal".as_slice());
            tag.push(index);
            let material = scheme
                .sample_terminal_material(
                    &secrets.payload,
                    Some(&table),
                    None,
                    Some(&transform),
                    &bank,
                    seed,
                    tag,
                    rand::random(),
                )
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            let prepared =
                scheme.prepare_terminal(&key, &material).map_err(|_| mxx_dsl::DslError::Schema)?;
            let public = scheme
                .finalize_terminal_public(prepared.public_accumulator())
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            let raw = scheme
                .evaluate_terminal(&input, &prepared, &masks)
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            let output = scheme
                .finalize_terminal_encoding(raw, &public)
                .map_err(|_| mxx_dsl::DslError::Schema)?;
            Ok(output.row().clone() - mask_secret * public.matrix - destination)
        })
        .unwrap();
        let graph = DslContext::new("rns_bgg-indexed-terminal-routing")
            .output("route-0", errors.at(0))
            .unwrap()
            .output("route-1", errors.at(1))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input_bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: 1u8.into(),
        };
        let gain = 1u8.into();
        let noise = parameters
            .terminal_noise_bound(0, [(&input_bound, 2, &gain)], weight.magnitude().clone())
            .unwrap();
        (graph, parameters, noise)
    }

    #[test]
    fn test_rns_bgg_terminal_noise_groups_match_expanded_members() {
        let (_, parameters, _) = routed_terminal_fixture();
        let left = RnsBggNoiseBound { error: 3u8.into(), payload_l1: 1u8.into() };
        let right = RnsBggNoiseBound { error: 7u8.into(), payload_l1: 3u8.into() };
        let left_gain = 1u8.into();
        let right_gain = 5u8.into();
        for (left_members, right_members) in [(1, 1), (3, 5), (9, 2)] {
            let grouped = parameters
                .terminal_noise_bound(
                    0,
                    [
                        (&left, 2 * left_members, &left_gain),
                        (&right, 4 * right_members, &right_gain),
                    ],
                    31u8.into(),
                )
                .unwrap();
            let expanded = parameters
                .terminal_noise_bound(
                    0,
                    std::iter::repeat_n((&left, 2, &left_gain), left_members)
                        .chain(std::iter::repeat_n((&right, 4, &right_gain), right_members)),
                    31u8.into(),
                )
                .unwrap();
            assert_eq!(grouped.branches, 2 * left_members + 4 * right_members);
            assert_eq!(grouped.branches, expanded.branches);
            assert_eq!(grouped.error, expanded.error);
            assert_eq!(grouped.coefficient_bound, expanded.coefficient_bound);
        }
    }

    #[test]
    fn test_rns_bgg_terminal_indexed_routing_keeps_fresh_error_unscaled() {
        let (graph, parameters, noise) = routed_terminal_fixture();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().iter().cloned()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let q = parameters.level(1).unwrap().modulus();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in polynomial.coeffs() {
                    let absolute =
                        coefficient.value().clone().min(q.as_ref() - coefficient.value());
                    assert!(absolute <= noise.error, "{name}: {absolute} > {}", noise.error);
                }
            }
        }
    }

    #[test]
    fn test_rns_bgg_terminal_rows_large_weights_and_raw_batch_relations() {
        let (graph, parameters, noise) = terminal_fixture();
        let results = execute(
            &graph,
            &mut cpu_backend(parameters.levels().iter().cloned()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let q = parameters.level(1).unwrap().modulus();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                for coefficient in polynomial.coeffs() {
                    let absolute =
                        coefficient.value().clone().min(q.as_ref() - coefficient.value());
                    assert!(absolute <= noise.error, "{name}: {absolute} > {}", noise.error);
                }
            }
        }
    }
}
