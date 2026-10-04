//! Public-key refresh arithmetic and its `NoiseRefresh` implementation.
use super::*;

fn decompose_refresh_matrix(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    decomposition_level: usize,
    matrix: RnsBggMatrix,
) -> Result<RnsBggDecomposition, RnsBggError> {
    let decomposition = parameters.level(decomposition_level)?;
    let base = 1u64
        .checked_shl(decomposition.base_bits())
        .ok_or(RnsBggError::Parameters("refresh decomposition base overflow"))?;
    let decomposition_count = decomposition.modulus_digits();
    let value = (
        (
            matrix.0.0.decompose(base, decomposition_count),
            matrix.0.1.decompose(base, decomposition_count),
        ),
        (
            matrix.1.0.decompose(base, decomposition_count),
            matrix.1.1.decompose(base, decomposition_count),
        ),
    );
    let evaluation_ring = parameters.ring(evaluation_level)?;
    Ok(super::centered_rebase_compact(value, &evaluation_ring))
}

/// A refresh action `[top; desired - r0 top + eta]` at the slot's `p` level
/// under payload row `payload = (r0, 1)`.
pub(in crate::rns_bgg) fn refresh_payload_action(
    parameters: &RnsBggParameters,
    decomposition_level: usize,
    payload: &RnsBggRow,
    desired: RnsBggRow,
    top: RnsBggRow,
    target_columns: usize,
) -> Result<RnsBggMatrix, RnsBggError> {
    let p_ring = parameters.ring(decomposition_level)?;
    let r0 = checked_modulus_convert(payload.0.clone(), &p_ring)?;
    super::ciphertext_matrix(parameters, decomposition_level, target_columns, r0, top, desired)
}

/// The hash-sampled top row of a setup-fixed (Static) refresh action.
fn refresh_action_top(
    parameters: &RnsBggParameters,
    decomposition_level: usize,
    hash_key: Bytes,
    tag: impl Into<mxx_dsl::HashTag>,
    target_columns: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let p_ring = parameters.ring(decomposition_level)?;
    let top_matrix = matrix_from_hash(&p_ring, hash_key, tag, target_columns);
    Ok(top_matrix.0)
}

/// The negated positional target `-W_label` of one refresh label action at
/// the slot's `p` level. The encoding carries the private payload with a
/// minus sign, so applying the action to the PRF value contributes
/// `+z W_label`; a fresh-error label is additionally weighted by `H = p/q_t`.
pub(in crate::rns_bgg) fn refresh_label_action_target(
    parameters: &RnsBggParameters,
    declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    slot: usize,
    label: RefreshPrfLabel,
    evaluation_level: usize,
    decomposition_level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let h =
        matches!(label, RefreshPrfLabel::FreshError { .. }).then(|| &metadata.p / &metadata.q_t);
    let target_q = refresh_label_target_with_columns(
        parameters,
        evaluation_level,
        parameters.component_columns(declaration.target_level())?,
        declaration.digits().base,
        label,
        h,
    )?;
    let target_p =
        checked_modulus_convert_matrix(target_q, &parameters.ring(decomposition_level)?)?;
    Ok(map_matrix(target_p, |entry| -entry).0)
}

/// The data action target `H lift_p(Delta_* t H_*)` of one refresh slot, the
/// target-level payload row under payload row `payload` lifted to the slot's
/// `p` level and scaled by `H = p/q_t`.
pub(in crate::rns_bgg) fn refresh_data_action_target(
    parameters: &RnsBggParameters,
    declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    slot: usize,
    payload: &RnsBggRow,
    decomposition_level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let target_level = declaration.target_level();
    let p_ring = parameters.ring(decomposition_level)?;
    let h_p = p_ring.polynomial([IntExpr::constant(&metadata.p / &metadata.q_t)]);
    let output_ring = parameters.ring(target_level)?;
    let output_gadget = parameters.state_gadget(target_level)?;
    let output_scale = parameters.scalar(target_level, parameters.message_scale(target_level)?)?;
    let output_payload = checked_modulus_convert_row(payload.clone(), &output_ring)?;
    let desired_output = (
        output_scale.clone() * output_payload.0 * output_gadget.clone(),
        output_scale * output_payload.1 * output_gadget,
    );
    Ok(lift_output_target_to_p(desired_output, &p_ring, h_p))
}

fn lift_output_target_to_p(value: RnsBggRow, p_ring: &Ring, h_p: Mat) -> RnsBggRow {
    let h_p = h_p.centered_rebase(p_ring);
    super::map_row(value, |entry| h_p.clone() * entry.centered_rebase(p_ring))
}

fn refresh_output_wire(
    program: &ExponentLutProgram,
    output_index: usize,
) -> Result<WireId, RnsBggError> {
    let output = program
        .outputs()
        .get(output_index)
        .ok_or(RnsBggError::Binding("refresh output is outside Program outputs"))?;
    Ok(match output {
        crate::program::Operand::Scalar(wire) |
        crate::program::Operand::Member { family: wire, .. } => *wire,
    })
}

pub(super) fn validate_refresh_output_target(
    program: &ExponentLutProgram,
    output_index: usize,
) -> Result<(), RnsBggError> {
    let wire = refresh_output_wire(program, output_index)?;
    let occurrences = program
        .outputs()
        .iter()
        .filter(|output| match output {
            crate::program::Operand::Scalar(candidate) |
            crate::program::Operand::Member { family: candidate, .. } => *candidate == wire,
        })
        .count();
    if occurrences > 1 {
        return Err(RnsBggError::Binding(
            "cannot refresh a duplicated Program output wire because public-key artifacts are deduplicated",
        ));
    }
    if program.inputs().contains_key(&wire) {
        return Err(RnsBggError::Binding(
            "cannot refresh a raw Program input output because its public-key artifact is cached",
        ));
    }
    Ok(())
}

impl PublicKeyLowering<'_> {
    pub(super) fn validate_refresh_declaration(
        &self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    ) -> Result<(), RnsBggError> {
        if declaration.source_program() != self.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        self.program
            .wire_is_coefficient_output(declaration.source_output())
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidSourceOutput))
            .and_then(|is_coefficient| {
                if is_coefficient {
                    Err(RnsBggError::Refresh(RefreshError::InvalidSourceOutput))
                } else {
                    Ok(())
                }
            })?;
        // Resolve both logical levels before graph construction.  This checks
        // the target ring and ensures the source output has a valid RNS
        // representation; the declaration itself owns CRT divisibility and
        // the exact sparse-PRF schema.
        let source_level = self.workload.level(declaration.source_output())?;
        self.parameters.validate_refresh_level_contract(declaration, source_level)?;
        self.parameters.level(declaration.target_level())?;
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        Ok(())
    }
}

#[cfg(test)]
fn scale_public_matrix(matrix: RnsBggMatrix, scalar: &BigInt) -> RnsBggMatrix {
    map_matrix(matrix, |entry| entry * scalar.clone())
}

#[cfg(test)]
fn mod_down_lift_public(
    matrix: RnsBggMatrix,
    channel_ring: &Ring,
    target_ring: &Ring,
) -> RnsBggMatrix {
    map_matrix(matrix, |entry| entry.block_mod_switch(channel_ring, 1).centered_rebase(target_ring))
}

/// Project one refresh anchor into its plaintext channel and then preserve
/// that centered representative in the combined CRT ring.
///
/// This is deliberately not [`mod_down_lift_public`]: the public refresh
/// carrier is the channel's `A_star`, so it must be reduced modulo `q_t`
/// without dividing by `q_t`.  The decoder target is formed separately from
/// `A_sum - H*A_star` by the encoding stage.
fn reduce_lift_public_anchor(
    matrix: RnsBggMatrix,
    channel_ring: &Ring,
    target_ring: &Ring,
) -> Result<RnsBggMatrix, RnsBggError> {
    let reduced = checked_modulus_convert_matrix(matrix, channel_ring)?;
    Ok(map_matrix(reduced, |entry| entry.centered_rebase(target_ring)))
}

fn matrix_schema_matches(lhs: &RnsBggMatrix, rhs: &RnsBggMatrix) -> bool {
    lhs.0.0.matrix_type() == rhs.0.0.matrix_type() &&
        lhs.0.1.matrix_type() == rhs.0.1.matrix_type() &&
        lhs.1.0.matrix_type() == rhs.1.0.matrix_type() &&
        lhs.1.1.matrix_type() == rhs.1.1.matrix_type()
}

#[cfg(test)]
fn refresh_label_target(
    parameters: &RnsBggParameters,
    level: usize,
    base: usize,
    label: RefreshPrfLabel,
    fresh_h: Option<BigInt>,
) -> Result<RnsBggMatrix, RnsBggError> {
    refresh_label_target_with_columns(
        parameters,
        level,
        parameters.component_columns(level)?,
        base,
        label,
        fresh_h,
    )
}

fn refresh_label_target_with_columns(
    parameters: &RnsBggParameters,
    level: usize,
    columns: usize,
    base: usize,
    label: RefreshPrfLabel,
    fresh_h: Option<BigInt>,
) -> Result<RnsBggMatrix, RnsBggError> {
    let ring = parameters.ring(level)?;
    let (component, coefficient, digit, fresh, h) = match label {
        RefreshPrfLabel::Mask { component, coefficient, digit, .. } => {
            (component, coefficient, digit, false, BigInt::from(1u8))
        }
        RefreshPrfLabel::FreshError { component, coefficient, digit, .. } => {
            (component, coefficient, digit, true, fresh_h.unwrap_or_else(|| BigInt::from(1u8)))
        }
    };
    let component_limit = columns
        .checked_mul(2)
        .ok_or(RnsBggError::Parameters("refresh component index overflow"))?;
    if component >= component_limit {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let block = component / columns;
    let column = component % columns;
    let power = ring.constant(
        (1, 1),
        ConstantMatrix::PowerOfBase {
            base: IntExpr::constant(base),
            exponent: IntExpr::constant(digit),
        },
    );
    let rotation = ring
        .constant((1, 1), ConstantMatrix::Rotation { exponent: IntExpr::constant(coefficient) });
    let mut weight = power * rotation;
    if fresh {
        weight = weight * ring.polynomial([IntExpr::constant(h)]);
    }
    let unit =
        ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(column) });
    let row = weight * unit;
    let first = ring.polynomial([IntExpr::constant(1usize - block)]);
    let second = ring.polynomial([IntExpr::constant(block)]);
    let row = (first * row.clone(), second * row);
    let zero = ring.zero((1, columns));
    Ok(((row.0.clone(), row.1.clone()), (zero.clone(), zero)))
}

fn validate_refresh_prf_wire(actual: WireId, expected: WireId) -> Result<(), RnsBggError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))
    }
}

fn validate_refresh_source(
    parameters: &RnsBggParameters,
    anchor: &RnsBggPublicKey,
    expected_wire: WireId,
    slot: &crate::noise_refresh::refresh::RefreshSlot,
) -> Result<usize, RnsBggError> {
    if anchor.wire != expected_wire {
        return Err(RnsBggError::Refresh(RefreshError::AnchorMismatch));
    }
    let modulus =
        slot.q.to_biguint().ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    let level = refresh_modulus_level(parameters, &modulus, "refresh source modulus")?;
    let expected = parameters.ring(level)?.matrix_type((1, parameters.component_columns(level)?));
    if [
        anchor.matrix.0.0.matrix_type(),
        anchor.matrix.0.1.matrix_type(),
        anchor.matrix.1.0.matrix_type(),
        anchor.matrix.1.1.matrix_type(),
    ]
    .iter()
    .any(|actual| **actual != expected)
    {
        return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
    }
    Ok(level)
}

fn refresh_output_anchor(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    target_columns: usize,
    hash_key: Bytes,
    commitment: [u8; 32],
    slot: usize,
    wire: WireId,
) -> Result<RnsBggPublicKey, RnsBggError> {
    let ring = parameters.ring(evaluation_level)?;
    let tag = mxx_dsl::HashTag::from(
        format!("rns-bgg/refresh-anchor/{commitment:?}/{slot}").into_bytes(),
    );
    Ok(RnsBggPublicKey::new(wire, matrix_from_hash(&ring, hash_key, tag, target_columns)))
}

fn refresh_modulus_level(
    parameters: &RnsBggParameters,
    modulus: &BigUint,
    context: &'static str,
) -> Result<usize, RnsBggError> {
    parameters
        .levels()
        .iter()
        .position(|level| level.modulus().as_ref() == modulus)
        .ok_or(RnsBggError::Binding(context))
}

fn validate_refresh_qp_levels(
    parameters: &RnsBggParameters,
    slot: &crate::noise_refresh::refresh::RefreshSlot,
    evaluation_level: usize,
) -> Result<usize, RnsBggError> {
    let decomposition_level = evaluation_level
        .checked_add(1)
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    let decomposition = parameters.level(decomposition_level).map_err(|_| {
        RnsBggError::Binding("refresh p modulus is not the immediate q state level")
    })?;
    let p_modulus =
        slot.p.to_biguint().ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    if decomposition.modulus().as_ref() != &p_modulus {
        return Err(RnsBggError::Binding("refresh p modulus is not the immediate q state level"));
    }
    let q_modulus =
        slot.q.to_biguint().ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    if &q_modulus % &p_modulus != BigUint::zero() {
        return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
    }
    Ok(decomposition_level)
}

fn declared_refresh_target_modulus(
    slots: &[crate::noise_refresh::refresh::RefreshSlot],
) -> Result<BigUint, RnsBggError> {
    slots.iter().try_fold(BigUint::one(), |product, slot| {
        slot.q_t
            .to_biguint()
            .map(|modulus| product * modulus)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))
    })
}

fn validate_refresh_target_modulus(
    slots: &[crate::noise_refresh::refresh::RefreshSlot],
    target: &BigUint,
) -> Result<(), RnsBggError> {
    if declared_refresh_target_modulus(slots)? != *target {
        return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
    }
    Ok(())
}

impl PublicKeyLowering<'_> {
    fn finish_channel_refresh(
        &mut self,
        inputs: &BTreeMap<usize, RnsBggPublicKey>,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
        outputs: &BTreeMap<PrfLabel, BTreeMap<usize, RnsBggPublicKey>>,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        self.validate_refresh_declaration(declaration)?;
        let source_wire = declaration.source_output();
        let slot_count = declaration.slots().len();
        if inputs.len() != slot_count || inputs.keys().copied().ne(0..slot_count) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let target_columns = self.parameters.component_columns(declaration.target_level())?;
        let mut anchor_levels = Vec::with_capacity(slot_count);
        let mut decomposition_levels = Vec::with_capacity(slot_count);
        let mut output_anchors = Vec::with_capacity(slot_count);
        for slot in 0..slot_count {
            let ain = inputs.get(&slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let evaluation_level = validate_refresh_source(
                self.parameters,
                ain,
                source_wire,
                declaration
                    .slots()
                    .get(slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?,
            )?;
            let decomposition_level = validate_refresh_qp_levels(
                self.parameters,
                declaration
                    .slots()
                    .get(slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?,
                evaluation_level,
            )?;
            anchor_levels.push(evaluation_level);
            decomposition_levels.push(decomposition_level);
            output_anchors.push(refresh_output_anchor(
                self.parameters,
                evaluation_level,
                target_columns,
                self.hash_key.clone(),
                *declaration.commitment(),
                slot,
                source_wire,
            )?);
        }

        let target_parameters = self.parameters.level(declaration.target_level())?;
        let target_ring = self.parameters.ring(declaration.target_level())?;
        let expected_port = target_ring.matrix_type((1, target_columns));
        let schema_ports = declaration
            .output_schema()
            .iter()
            .filter_map(|wire| match wire {
                WireType::Matrix(matrix) => Some(matrix),
                _ => None,
            })
            .collect::<Vec<_>>();
        if schema_ports.len() != 4 || schema_ports.iter().any(|port| **port != expected_port) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        validate_refresh_target_modulus(declaration.slots(), target_parameters.modulus().as_ref())?;

        // A_sum is built independently in each intermediate ring.  Mask
        // labels may contribute only to their owning slot; fresh labels must
        // be present in every slot and therefore preserve their shared source.
        // Dynamic actions depend on the secret, so encryption builds A_sum
        // from the source and label keys recorded here.
        let dynamic = self.parameters.rhs_mode() == RhsMode::Dynamic;
        let mut sums = vec![None::<RnsBggMatrix>; slot_count];
        let expected_wires = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(_, wire)| wire)
            .collect::<Vec<_>>();
        for (index, expected_label) in declaration.prf_labels().iter().enumerate() {
            let expected_wire = expected_wires
                .get(index)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let label = declaration
                .labels()
                .label(index)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let channels = outputs
                .get(expected_label)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let valid_digit = match label {
                RefreshPrfLabel::Mask { slot, digit, .. } => {
                    slot < declaration.digits().mask_counts.len() &&
                        digit < declaration.digits().mask_counts[slot]
                }
                RefreshPrfLabel::FreshError { digit, .. } => {
                    digit < declaration.digits().fresh_count
                }
            };
            if !valid_digit {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
            let expected_slots = label.slots(slot_count).map_err(RnsBggError::Refresh)?;
            if channels.len() != expected_slots.len() ||
                channels.keys().copied().ne(expected_slots.iter().copied())
            {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            for slot in expected_slots {
                let value = channels
                    .get(&slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
                let ain =
                    inputs.get(&slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                validate_refresh_prf_wire(value.wire, *expected_wire)?;
                if !matrix_schema_matches(&value.matrix, &ain.matrix) {
                    return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                }
                let slot_metadata = &declaration.slots()[slot];
                let modulus = slot_metadata
                    .q
                    .to_biguint()
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
                let expected_modulus = BigInt::from(modulus);
                if super::concrete_ring(&ain.matrix.0.0)?.modulus() != expected_modulus ||
                    super::concrete_ring(&value.matrix.0.0)?.modulus() != expected_modulus
                {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                let evaluation_level = anchor_levels[slot];
                let decomposition_level = decomposition_levels[slot];
                if dynamic {
                    // Encryption forms this label's action, and its share
                    // of `A_sum`, from the child-PRF output key.
                    self.refresh_keys.insert(
                        (*declaration.commitment(), slot, Some(index)),
                        value.matrix.clone(),
                    );
                    continue;
                }
                let target_columns =
                    self.parameters.component_columns(declaration.target_level())?;
                let top = refresh_action_top(
                    self.parameters,
                    decomposition_level,
                    self.hash_key.clone(),
                    mxx_dsl::HashTag::from(
                        format!(
                            "rns-bgg/refresh-label/{:?}/{slot}/{index}",
                            declaration.commitment()
                        )
                        .into_bytes(),
                    ),
                    target_columns,
                )?;
                let label_action = refresh_payload_action(
                    self.parameters,
                    decomposition_level,
                    &self.payload()?.row,
                    refresh_label_action_target(
                        self.parameters,
                        declaration,
                        slot,
                        label,
                        evaluation_level,
                        decomposition_level,
                    )?,
                    top,
                    target_columns,
                )?;
                let label_action_decomposition = decompose_refresh_matrix(
                    self.parameters,
                    evaluation_level,
                    decomposition_level,
                    label_action.clone(),
                )?;
                let contribution =
                    super::multiply_public_action(value.matrix.clone(), label_action_decomposition);
                self.refresh_label_targets
                    .insert((*declaration.commitment(), slot, index), label_action);
                sums[slot] = Some(match sums[slot].take() {
                    None => contribution,
                    Some(sum) => add_public_matrices(sum, contribution),
                });
            }
        }

        let target_modulus = target_parameters.modulus().as_ref().clone();
        let mut values = Vec::with_capacity(slot_count);
        let mut coefficients = Vec::with_capacity(slot_count);
        for slot in 0..slot_count {
            let ain = inputs.get(&slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let aout = output_anchors
                .get(slot)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let slot_modulus = declaration.slots()[slot]
                .q
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let evaluation_level = anchor_levels[slot];
            let decomposition_level = decomposition_levels[slot];
            let p_ring = self.parameters.ring(decomposition_level)?;
            let output_ring = self.parameters.ring(declaration.target_level())?;
            if &slot_modulus % &target_modulus != BigUint::zero() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            // The refresh carrier is the per-slot A_star anchor.  Reduce it
            // modulo q_t without scaling, then centered-extend into Q* before
            // CRT recomposition.  Decoder targets remain A_sum-H*A_star and
            // are assembled by the encoding stage from the raw artifacts.
            let aout_p = checked_modulus_convert_matrix(aout.matrix.clone(), &p_ring)?;
            let target_at_output = reduce_lift_public_anchor(
                aout_p.clone(),
                &self.parameters.ring_for_modulus(&declaration.slots()[slot].q_t)?,
                &output_ring,
            )?;
            if values.first().is_some_and(|first| !matrix_schema_matches(first, &target_at_output))
            {
                return Err(RnsBggError::Binding("refresh CRT channel schemas differ"));
            }
            if dynamic {
                // Encryption forms the data action and `A_sum` from the
                // source key.
                self.refresh_keys
                    .insert((*declaration.commitment(), slot, None), ain.matrix.clone());
            } else {
                let a_sum = sums[slot]
                    .take()
                    .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
                let target_columns =
                    self.parameters.component_columns(declaration.target_level())?;
                let top = refresh_action_top(
                    self.parameters,
                    decomposition_level,
                    self.hash_key.clone(),
                    mxx_dsl::HashTag::from(
                        format!("rns-bgg/refresh-data/{:?}/{slot}", declaration.commitment())
                            .into_bytes(),
                    ),
                    target_columns,
                )?;
                let payload = &self.payload()?.row;
                let action = refresh_payload_action(
                    self.parameters,
                    decomposition_level,
                    payload,
                    refresh_data_action_target(
                        self.parameters,
                        declaration,
                        slot,
                        payload,
                        decomposition_level,
                    )?,
                    top,
                    target_columns,
                )?;
                let action_decomposition = decompose_refresh_matrix(
                    self.parameters,
                    evaluation_level,
                    decomposition_level,
                    action.clone(),
                )?;
                let data_public =
                    super::multiply_public_action(ain.matrix.clone(), action_decomposition);
                let a_sum = mod_down_public(add_public_matrices(a_sum, data_public), &p_ring);
                self.refresh_sums.insert((*declaration.commitment(), slot), a_sum);
                self.refresh_targets.insert((*declaration.commitment(), slot), action);
            }
            self.refresh_stars.insert((*declaration.commitment(), slot), aout_p);
            values.push(target_at_output);
            coefficients.push(self.parameters.scalar(
                declaration.target_level(),
                BigInt::from(declaration.crt_weight(slot).map_err(RnsBggError::Refresh)?),
            )?);
        }
        let values = Family::pack(values).map_err(graph)?;
        let coefficients = Family::pack(coefficients).map_err(graph)?;
        let output =
            RnsBggPublicKey::new(source_wire, super::recompose_public(values, coefficients)?);
        if output.schema().wire_types() != declaration.output_schema() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        Ok(output)
    }
}

/// Public-key refresh dispatch through the common PRF callback contract.
impl NoiseRefresh for RnsBggRefresh<'_, PublicKeyLowering<'_>> {
    type PrfOutput = BTreeMap<usize, RnsBggPublicKey>;
    type Output = RnsBggPublicKey;
    type Error = RnsBggError;
    fn finish_noise_refresh(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
        outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
    ) -> Result<Self::Output, Self::Error> {
        let inputs = self.inputs().clone();
        self.lowerer().finish_channel_refresh(&inputs, declaration, &outputs)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use mxx_ir_core::ParamEnv;

    #[test]
    fn refresh_rejects_a_raw_input_program_output() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let raw = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let derived = builder
            .add(crate::program::Operand::Scalar(raw), crate::program::Operand::Scalar(raw))
            .unwrap();
        let program = builder
            .finish(vec![
                crate::program::Operand::Scalar(raw),
                crate::program::Operand::Scalar(derived),
            ])
            .unwrap();

        assert!(matches!(
            validate_refresh_output_target(&program, 0),
            Err(RnsBggError::Binding(
                "cannot refresh a raw Program input output because its public-key artifact is cached"
            ))
        ));
        assert!(validate_refresh_output_target(&program, 1).is_ok());
    }

    #[test]
    fn refresh_rejects_a_later_duplicate_program_output_wire() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let output = builder
            .add(crate::program::Operand::Scalar(input), crate::program::Operand::Scalar(input))
            .unwrap();
        let program = builder
            .finish(vec![
                crate::program::Operand::Scalar(output),
                crate::program::Operand::Scalar(output),
            ])
            .unwrap();

        assert!(matches!(
            validate_refresh_output_target(&program, 1),
            Err(RnsBggError::Binding(
                "cannot refresh a duplicated Program output wire because public-key artifacts are deduplicated"
            ))
        ));
    }

    #[test]
    fn refresh_label_target_keeps_component_coefficient_and_digit_coordinates() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let first = refresh_label_target(
            &parameters,
            0,
            3,
            RefreshPrfLabel::Mask {
                refresh_id: [1; 32],
                slot: 0,
                component: 0,
                coefficient: 0,
                digit: 0,
            },
            None,
        )
        .unwrap();
        let second = refresh_label_target(
            &parameters,
            0,
            3,
            RefreshPrfLabel::FreshError {
                refresh_id: [1; 32],
                component: 1,
                coefficient: 1,
                digit: 2,
            },
            Some(BigInt::from(5u8)),
        )
        .unwrap();
        assert_ne!(first.0.0.flatten(), second.0.0.flatten());
        assert_ne!(first.0.1.flatten(), second.0.1.flatten());
    }

    #[test]
    fn refresh_rejects_a_prf_wire_from_the_wrong_ordered_output() {
        assert!(validate_refresh_prf_wire(WireId(9), WireId(10)).is_err());
        assert!(validate_refresh_prf_wire(WireId(10), WireId(10)).is_ok());
    }

    #[test]
    fn refresh_prf_artifacts_are_slot_scoped() {
        let commitment = [0xabu8; 32];
        assert_eq!(
            refresh_prf_artifact_name(commitment, 0, "rhs/RhsId(1)"),
            "refresh/abababababababababababababababababababababababababababababababab/slot/0/prf/rhs/RhsId(1)"
        );
        assert_ne!(
            refresh_prf_artifact_name(commitment, 0, "rhs/RhsId(1)"),
            refresh_prf_artifact_name(commitment, 1, "rhs/RhsId(1)")
        );
        assert_eq!(
            refresh_label_artifact_name(commitment, 0, 7),
            refresh_artifact_name(commitment, 0, "target/7")
        );
        assert_eq!(
            refresh_checkpoint_artifact_name(commitment, "target"),
            "refresh/abababababababababababababababababababababababababababababababab/target"
        );
    }

    #[test]
    fn refresh_target_modulus_is_the_product_of_all_plaintext_slots() {
        let slots = vec![
            crate::noise_refresh::refresh::RefreshSlot {
                slot: 0,
                q: BigInt::from(15u8),
                p: BigInt::from(5u8),
                q_t: BigInt::from(3u8),
            },
            crate::noise_refresh::refresh::RefreshSlot {
                slot: 1,
                q: BigInt::from(35u8),
                p: BigInt::from(7u8),
                q_t: BigInt::from(5u8),
            },
        ];
        assert_eq!(declared_refresh_target_modulus(&slots).unwrap(), BigUint::from(15u8));
        let mismatched = vec![
            slots[0].clone(),
            crate::noise_refresh::refresh::RefreshSlot {
                q_t: BigInt::from(7u8),
                ..slots[1].clone()
            },
        ];
        assert!(validate_refresh_target_modulus(&mismatched, &BigUint::from(15u8)).is_err());
    }

    #[test]
    fn refresh_two_prime_channels_mod_down_then_lift_to_combined_ring() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let source = explicit_ring(&primes, 4);
        let combined = explicit_ring(&primes[..2], 4);
        let value = source.input("refresh-source", (1, 1));
        let matrix = ((value.clone(), value.clone()), (value.clone(), value));
        let lifted = mod_down_lift_public(matrix, &explicit_ring(&primes[..1], 4), &combined);
        for row in [lifted.0, lifted.1] {
            for entry in [row.0, row.1] {
                assert_eq!(modulus_of(entry.matrix_type()), BigInt::from(primes[0]) * primes[1]);
            }
        }
        assert_eq!(
            crate::noise_refresh::refresh::refresh_crt_weight(
                &[
                    crate::noise_refresh::refresh::RefreshSlot {
                        slot: 0,
                        q: 105.into(),
                        p: 105.into(),
                        q_t: 3.into(),
                    },
                    crate::noise_refresh::refresh::RefreshSlot {
                        slot: 1,
                        q: 105.into(),
                        p: 105.into(),
                        q_t: 5.into(),
                    },
                ],
                0
            )
            .unwrap(),
            BigUint::from(10u8)
        );
    }

    #[test]
    fn refresh_anchor_reduction_is_not_moddown_and_decoder_target_is_distinct() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let source = explicit_ring(&primes, 4);
        let combined = explicit_ring(&primes[..2], 4);
        let anchor_entry = source.input("refresh-anchor", (1, 1));
        let anchor = (
            (anchor_entry.clone(), anchor_entry.clone()),
            (anchor_entry.clone(), anchor_entry.clone()),
        );
        let a_sum_entry = source.input("refresh-a-sum", (1, 1));
        let a_sum =
            ((a_sum_entry, source.zero((1, 1))), (source.zero((1, 1)), source.zero((1, 1))));

        // Both channels are present in the CRT declaration.  The anchor
        // carrier uses plaintext reduction for each q_t, while the old
        // ModDown path divides by q_t and therefore produces a different
        // graph even when the final ring is the same Q* ring.
        for channel in [&primes[..1], &primes[1..2]] {
            let channel = explicit_ring(channel, 4);
            let reduced = reduce_lift_public_anchor(anchor.clone(), &channel, &combined).unwrap();
            let divided = mod_down_lift_public(anchor.clone(), &channel, &combined);
            assert_ne!(reduced.flatten(), divided.flatten());
            assert_eq!(modulus_of(reduced.0.0.matrix_type()), BigInt::from(primes[0]) * primes[1]);
        }

        // H is intentionally non-unit (p/q_t = a*b*c/b = a*c).  The decoder
        // target includes the A_sum contribution and remains a separate
        // artifact from the public A_star carrier.
        let channel = explicit_ring(&primes[1..2], 4);
        let h = BigInt::from(primes[0]) * primes[2];
        let decoder_target = mod_down_lift_public(
            add_public_matrices(a_sum, scale_public_matrix(anchor.clone(), &-h)),
            &channel,
            &combined,
        );
        let public_anchor = reduce_lift_public_anchor(anchor, &channel, &combined).unwrap();
        assert_ne!(decoder_target.flatten(), public_anchor.flatten());
    }

    #[test]
    fn refresh_nonzero_anchor_data_action_is_part_of_published_a_sum() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let anchor_entry = ring.input("refresh-q-anchor", (1, 1));
        let anchor = (
            (anchor_entry.clone(), anchor_entry.clone()),
            (anchor_entry.clone(), anchor_entry.clone()),
        );
        let action = decompose_matrix(
            &parameters,
            0,
            scale_public_matrix(anchor.clone(), &BigInt::from(-3i8)),
        )
        .unwrap();
        let data_public = multiply_public_action(anchor, action);
        let zero = ring.zero((1, 1));
        let empty_sum = ((zero.clone(), zero.clone()), (zero.clone(), zero));
        let published_sum = add_public_matrices(empty_sum.clone(), data_public.clone());

        assert_ne!(data_public.flatten(), empty_sum.flatten());
        assert_ne!(published_sum.flatten(), empty_sum.flatten());
        assert_eq!(published_sum.0.0.matrix_type(), data_public.0.0.matrix_type());
    }

    #[test]
    fn refresh_data_terminal_zero_noise_oracle_keeps_ain_and_aout_distinct() {
        let q = 35i64;
        let p = 5i64;
        let h = 5i64;
        let r0 = 2i64;
        let r1 = 1i64;
        let mu = 3i64;
        let ain = 11i64;
        let aout = 17i64;
        assert!(q > p && p > 1 && mu != 0 && r0 != 0 && ain != aout);

        let top = [4i64, 2i64];
        let target = [1i64, 3i64];
        let bottom = [h * target[0] - r0 * top[0], h * target[1] - r0 * top[1]];
        for component in 0..2 {
            let payload_relation = (r0 * top[component] + r1 * bottom[component]).rem_euclid(p);
            assert_eq!(payload_relation, (h * target[component]).rem_euclid(p));
        }
        let data = bottom.map(|value| (ain * value).rem_euclid(q));
        let anchor = aout.rem_euclid(p);
        assert_ne!(data[0], data[1]);
        assert_ne!(anchor, ain.rem_euclid(p));
        assert_eq!(bottom[0].rem_euclid(p), 2);
        assert_eq!(bottom[1].rem_euclid(p), 1);
    }

    #[test]
    fn refresh_output_target_lifts_qstar_into_containing_p() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let qstar = explicit_ring(&primes[..2], 4);
        let p_ring = explicit_ring(&primes, 4);
        let value = qstar.input("refresh-qstar-target", (1, 1));
        let row = (value.clone(), value);
        let lifted = lift_output_target_to_p(
            row.clone(),
            &p_ring,
            qstar.polynomial([IntExpr::constant(BigInt::from(3u8))]),
        );
        let reduced = checked_modulus_convert_row(row, &p_ring).unwrap();
        let reduced = super::map_row(reduced, |entry| {
            qstar.polynomial([IntExpr::constant(BigInt::from(3u8))]) * entry
        });
        assert_eq!(
            modulus_of(lifted.0.matrix_type()),
            BigInt::from(primes[0]) * primes[1] * primes[2]
        );
        assert_ne!(lifted.flatten(), reduced.flatten());
        let graph = DslContext::new("refresh-qstar-to-p")
            .transferred_output("lifted", lifted)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn refresh_target_width_is_used_across_q_anchor_p_action_and_a_star() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let evaluation_level = 0;
        let decomposition_level = 1;
        let target_level = 1;
        let evaluation_columns = parameters.component_columns(evaluation_level).unwrap();
        let target_columns = parameters.component_columns(target_level).unwrap();
        assert_ne!(evaluation_columns, target_columns);
        let hash_key =
            parameters.ring(evaluation_level).unwrap().bytes_input("refresh-width-key", 32);

        let anchor = refresh_output_anchor(
            &parameters,
            evaluation_level,
            target_columns,
            hash_key.clone(),
            [3u8; 32],
            0,
            WireId(4),
        )
        .unwrap();
        assert_eq!(
            anchor.matrix.0.0.matrix_type(),
            &parameters.ring(evaluation_level).unwrap().matrix_type((1, target_columns))
        );

        let payload = RnsBggPayloadSecret {
            row: (
                parameters.ring(evaluation_level).unwrap().uniform_interval((1, 1), -1, 1),
                parameters.ring(evaluation_level).unwrap().identity(1),
            ),
            binding: None,
        };
        let p_ring = parameters.ring(decomposition_level).unwrap();
        let desired = (p_ring.zero((1, target_columns)), p_ring.zero((1, target_columns)));
        let action = refresh_payload_action(
            &parameters,
            decomposition_level,
            &payload.row,
            desired,
            refresh_action_top(
                &parameters,
                decomposition_level,
                hash_key,
                mxx_dsl::HashTag::from(b"refresh-width-test".to_vec()),
                target_columns,
            )
            .unwrap(),
            target_columns,
        )
        .unwrap();
        assert_eq!(
            crate::test_support::modulus_of(action.0.0.matrix_type()),
            BigInt::from(parameters.level(decomposition_level).unwrap().modulus().as_ref().clone())
        );
        assert_eq!(action.0.0.matrix_type().columns, IntExpr::constant(target_columns));

        let graph = DslContext::new("refresh-target-width")
            .transferred_output("anchor", anchor.matrix.clone())
            .unwrap()
            .transferred_output("action", action.clone())
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();

        let a_star = checked_modulus_convert_matrix(anchor.matrix, &p_ring).unwrap();
        assert_eq!(a_star.0.0.matrix_type(), &p_ring.matrix_type((1, target_columns)));
    }
}
