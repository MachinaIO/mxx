//! Encoding refresh arithmetic, affine offset routing, and its
//! `NoiseRefresh` implementation.
use super::*;

impl EncodingLowering<'_> {
    fn refresh_action(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
        role: &str,
    ) -> Result<RnsBggDecomposition, RnsBggError> {
        self.refresh_action_for_label(declaration, None, slot, role)
    }

    fn refresh_action_for_label(
        &self,
        declaration: &RefreshDeclaration,
        label: Option<usize>,
        slot: usize,
        role: &str,
    ) -> Result<RnsBggDecomposition, RnsBggError> {
        let q_level = refresh_slot_q_level(self.parameters, declaration, slot)?;
        let p_level = refresh_slot_level(self.parameters, declaration, slot)?;
        let ring = self.parameters.ring(p_level)?;
        // The preprocess stage stores the compact p-ring action matrix.  It
        // is decomposed only at this consuming q-level; no decomposition is
        // persisted in the stage boundary.
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let prefix = match label {
            Some(label) => refresh_label_artifact_name(*declaration.commitment(), slot, label),
            None => refresh_artifact_name(*declaration.commitment(), slot, role),
        };
        let matrix = |port: usize| {
            ring.artifact_input(
                self.ciphertext_production().clone(),
                format!("{prefix}.{port}"),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        };
        let matrix = ((matrix(0), matrix(1)), (matrix(2), matrix(3)));
        compact_decompose_matrix(self.parameters, q_level, p_level, matrix)
    }

    pub(super) fn refresh_decoder(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        let ring = self.parameters.ring(refresh_slot_level(self.parameters, declaration, slot)?)?;
        // The decoder is a target-matrix artifact.  Keep the slot p-level for
        // its ring and decomposition contract, but use the declared target
        // level for the matrix width.
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let prefix = refresh_artifact_name(*declaration.commitment(), slot, "decoder");
        Ok((
            ring.artifact_input(
                self.encryption.clone(),
                format!("{prefix}.0"),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
            ring.artifact_input(
                self.encryption.clone(),
                format!("{prefix}.1"),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
        ))
    }

    fn refresh_crt_weight(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<Mat, RnsBggError> {
        let weight = declaration.crt_weight(slot)?;
        Ok(self
            .parameters
            .ring(declaration.target_level())?
            .polynomial([IntExpr::constant(BigInt::from(weight))]))
    }

    /// Reconstruct the public refresh target from the public per-slot p-ring
    /// anchor emitted by preprocessing.  The anchor is reduced to `q_t`, centered
    /// extended to `Q*`, and then CRT recomposed.  The decoder's
    /// `A_sum-H*A_star` term is deliberately kept separate in the encryption
    /// graph; it is not part of the public target key.
    pub(super) fn refresh_public_target(
        &self,
        declaration: &RefreshDeclaration,
        wire: WireId,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        let level = declaration.target_level();
        let target_ring = self.parameters.ring(level)?;
        let production = self.preprocess.clone();
        let commitment = *declaration.commitment();
        let mut values = Vec::with_capacity(declaration.slots().len());
        let mut coefficients = Vec::with_capacity(declaration.slots().len());
        for slot in declaration.slots() {
            let slot_modulus = slot
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let source_level = self
                .parameters
                .levels()
                .iter()
                .position(|candidate| candidate.modulus().as_ref() == &slot_modulus)
                .ok_or(RnsBggError::Binding("refresh slot modulus is not configured"))?;
            // A-star is a public target-matrix artifact.  Its p-ring is tied
            // to the slot's source modulus, while its matrix width is the
            // declared refresh target width.
            let target_columns = self.parameters.component_columns(level)?;
            let ring = self.parameters.ring(source_level)?;
            let matrix = |role: &str| -> RnsBggMatrix {
                (
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.0", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.1", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.2", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.3", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                )
            };
            let a_star = matrix("a-star");
            let target = reduce_anchor_channel(
                a_star,
                &self.parameters.ring_for_modulus(&slot.q_t)?,
                &target_ring,
            )?;
            values.push(target);
            coefficients.push(self.refresh_crt_weight(declaration, slot.slot)?);
        }
        let values = Family::pack(values).map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let coefficients =
            Family::pack(coefficients).map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let matrix = recompose_public(values, coefficients)?;
        Ok(RnsBggPublicKey::new(wire, matrix))
    }
}

impl EncodingLowering<'_> {
    fn validate_refresh_declaration(
        &self,
        declaration: &RefreshDeclaration,
    ) -> Result<(), RnsBggError> {
        if declaration.source_program() != self.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        self.program
            .wire_is_coefficient_output(declaration.source_output())
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidSourceOutput))
            .and_then(|coefficient| {
                if coefficient {
                    Err(RnsBggError::Refresh(RefreshError::InvalidSourceOutput))
                } else {
                    Ok(())
                }
            })?;
        self.workload.level(declaration.source_output())?;
        self.parameters.level(declaration.target_level())?;
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        Ok(())
    }
}

impl<'a> RnsEncryptBuild<'a> {
    /// The public refresh matrix `role` of `slot` (`a-sum` or `a-star`)
    /// published by preprocessing, at the slot's `p` ring.
    pub(super) fn preprocess_refresh_matrix(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
        role: &str,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let metadata = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let p_ring = self.parameters.ring_for_modulus(&metadata.p)?;
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let production =
            self.preprocess.clone().ok_or(RnsBggError::Binding("missing preprocess"))?;
        let availability = match role {
            "a-sum" => ArtifactAvailability::Transferred,
            "a-star" => ArtifactAvailability::Cached,
            _ => return Err(RnsBggError::Binding("unknown refresh decoder matrix")),
        };
        let name = refresh_artifact_name(*declaration.commitment(), slot, role);
        let port = |port: usize| {
            p_ring.artifact_input(
                production.clone(),
                format!("{name}.{port}"),
                (1, columns),
                availability,
            )
        };
        Ok(((port(0), port(1)), (port(2), port(3))))
    }

    /// The refresh decoder `s (A_sum - H A_star) - o + e` of `slot`, for the
    /// slot's action sum `sum`.
    pub(super) fn refresh_decoder(
        &self,
        declaration: &RefreshDeclaration,
        mask: &RnsBggMaskSecret,
        slot: usize,
        sum: RnsBggMatrix,
    ) -> Result<RnsBggRow, RnsBggError> {
        let p_level = refresh_slot_level(self.parameters, declaration, slot)?;
        let metadata = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let p_ring = self.parameters.ring_for_modulus(&metadata.p)?;
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let anchor = self.preprocess_refresh_matrix(declaration, slot, "a-star")?;
        let h = p_ring.polynomial([IntExpr::constant(&metadata.p / &metadata.q_t)]);
        let delta = (
            (sum.0.0 - h.clone() * anchor.0.0, sum.0.1 - h.clone() * anchor.0.1),
            (sum.1.0 - h.clone() * anchor.1.0, sum.1.1 - h * anchor.1.1),
        );
        let secret = checked_modulus_convert_row(mask.row.clone(), &p_ring)?;
        let affine_offset = refresh_affine_offset_row(self.parameters, declaration, slot, p_level)?;
        let decoder = (
            secret.0.clone() * delta.0.0 + secret.1.clone() * delta.1.0 - affine_offset.0,
            secret.0 * delta.0.1 + secret.1 * delta.1.1 - affine_offset.1,
        );
        let error = encoding_error_row(self.parameters, &p_ring, columns)?;
        Ok((decoder.0 + error.0, decoder.1 + error.1))
    }

    /// Dynamic refresh helpers under the single secret, for every slot: the
    /// data and label actions, sampled in this encryption, and the decoder
    /// of their key sum `A_sum = MS_(q->p)(A_0 D_0 + sum_l A_l D_l)`, formed
    /// from the public source and label keys published by preprocessing.
    pub(super) fn dynamic_refresh(
        &self,
        mut graph: DslContext,
        declaration: &RefreshDeclaration,
        mask: &RnsBggMaskSecret,
    ) -> Result<DslContext, RnsBggError> {
        let payload = &self.payload_secret.as_ref().ok_or(RnsBggError::Binding("secret"))?.row;
        let production =
            self.preprocess.clone().ok_or(RnsBggError::Binding("missing preprocess"))?;
        let commitment = *declaration.commitment();
        let target_columns = self.parameters.component_columns(declaration.target_level())?;
        for slot in 0..declaration.slots().len() {
            let q_level = refresh_slot_q_level(self.parameters, declaration, slot)?;
            let p_level = refresh_slot_level(self.parameters, declaration, slot)?;
            let p_ring = self.parameters.ring(p_level)?;
            let q_ring = self.parameters.ring(q_level)?;
            let key_columns = self.parameters.component_columns(q_level)?;
            let key = |label: Option<usize>| -> RnsBggMatrix {
                let name = refresh_key_artifact_name(commitment, slot, label);
                let port = |port: usize| {
                    q_ring.artifact_input(
                        production.clone(),
                        format!("{name}.{port}"),
                        (1, key_columns),
                        ArtifactAvailability::Cached,
                    )
                };
                ((port(0), port(1)), (port(2), port(3)))
            };
            let applied = |action: &RnsBggMatrix, key: RnsBggMatrix| {
                compact_decompose_matrix(self.parameters, q_level, p_level, action.clone())
                    .map(|decomposition| multiply_public_action(key, decomposition))
            };
            let action = |target: RnsBggRow| {
                super::public_key::refresh_payload_action(
                    self.parameters,
                    p_level,
                    payload,
                    target,
                    uniform_row(&p_ring, target_columns),
                    target_columns,
                )
            };
            let data = action(super::public_key::refresh_data_action_target(
                self.parameters,
                declaration,
                slot,
                payload,
                p_level,
            )?)?;
            let mut sum = applied(&data, key(None))?;
            graph = graph
                .transferred_output(refresh_artifact_name(commitment, slot, "target"), data)?;
            for index in 0..declaration.labels().len() {
                let label = declaration
                    .labels()
                    .label(index)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                if !label
                    .slots(declaration.slots().len())
                    .map_err(RnsBggError::Refresh)?
                    .contains(&slot)
                {
                    continue;
                }
                let label_action = action(super::public_key::refresh_label_action_target(
                    self.parameters,
                    declaration,
                    slot,
                    label,
                    q_level,
                    p_level,
                )?)?;
                sum = add_public_matrices(sum, applied(&label_action, key(Some(index)))?);
                graph = graph.transferred_output(
                    refresh_label_artifact_name(commitment, slot, index),
                    label_action,
                )?;
            }
            let decoder =
                self.refresh_decoder(declaration, mask, slot, mod_down_public(sum, &p_ring))?;
            graph = graph
                .transferred_output(refresh_artifact_name(commitment, slot, "decoder"), decoder)?;
        }
        Ok(graph)
    }
}

fn refresh_slot_level(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<usize, RnsBggError> {
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let modulus = metadata
        .p
        .to_biguint()
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    parameters
        .levels()
        .iter()
        .position(|level| level.modulus().as_ref() == &modulus)
        .ok_or(RnsBggError::Binding("refresh slot modulus is not configured"))
}

fn refresh_slot_q_level(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<usize, RnsBggError> {
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let modulus = metadata
        .q
        .to_biguint()
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    parameters
        .levels()
        .iter()
        .position(|level| level.modulus().as_ref() == &modulus)
        .ok_or(RnsBggError::Binding("refresh slot q modulus is not configured"))
}

fn refresh_affine_coefficients(
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<(BigInt, BigInt), RnsBggError> {
    let digits = declaration.digits();
    let mask_digits =
        *digits.mask_counts.get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let base = BigUint::from(digits.base);
    let offset = |count, mask| {
        canonical_digit_offset(&base, count, mask)
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidLayout))
    };
    let mask_offset = offset(mask_digits, true)?;
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let h = metadata.p.clone() / metadata.q_t.clone();
    let fresh_offset = h * offset(digits.fresh_count, false)?;
    Ok((mask_offset, fresh_offset))
}

fn refresh_affine_route(
    ring: &mxx_dsl::Ring,
    columns: usize,
    component: usize,
    coefficient: usize,
    scalar: BigInt,
) -> Result<RnsBggRow, RnsBggError> {
    let component_limit = columns
        .checked_mul(2)
        .ok_or(RnsBggError::Parameters("refresh component index overflow"))?;
    if component >= component_limit {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let block = component / columns;
    let column = component % columns;
    let rotation = ring
        .constant((1, 1), ConstantMatrix::Rotation { exponent: IntExpr::constant(coefficient) });
    let unit =
        ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(column) });
    let row = ring.polynomial([IntExpr::constant(scalar)]) * rotation * unit;
    let first = ring.polynomial([IntExpr::constant(1usize - block)]);
    let second = ring.polynomial([IntExpr::constant(block)]);
    let zero = ring.zero((1, columns));
    Ok((first.clone() * row.clone() + second.clone() * zero.clone(), second * row + first * zero))
}

fn refresh_affine_offset_row(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    slot: usize,
    level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let ring = parameters.ring(level)?;
    let columns = parameters.component_columns(declaration.target_level())?;
    let (mask_offset, fresh_offset) = refresh_affine_coefficients(declaration, slot)?;
    let mut coordinates = BTreeSet::new();
    for index in 0..declaration.labels().len() {
        let label = declaration
            .labels()
            .label(index)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let (component, coefficient) = match label {
            RefreshPrfLabel::Mask { component, coefficient, .. } |
            RefreshPrfLabel::FreshError { component, coefficient, .. } => (component, coefficient),
        };
        coordinates.insert((component, coefficient));
    }
    if coordinates.is_empty() {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let zero = ring.zero((1, columns));
    let mut offset = (zero.clone(), zero);
    for (component, coefficient) in coordinates {
        let mask =
            refresh_affine_route(&ring, columns, component, coefficient, mask_offset.clone())?;
        let fresh =
            refresh_affine_route(&ring, columns, component, coefficient, fresh_offset.clone())?;
        offset = add_encoding_rows(offset, add_encoding_rows(mask, fresh));
    }
    Ok(offset)
}

/// Encoding-side implementation of the common one-shot refresh contract.
impl NoiseRefresh for RnsBggRefresh<'_, EncodingLowering<'_>> {
    type PrfOutput = BTreeMap<usize, RnsBggEncoding>;
    type Output = RnsBggEncoding;
    type Error = RnsBggError;
    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
    ) -> Result<Self::Output, Self::Error> {
        let inputs = self.inputs().clone();
        let slots = declaration.slots();
        let digits = declaration.digits();
        let labels = declaration.prf_labels();
        let label_count = declaration.labels().len();
        let target_level = declaration.target_level();
        if inputs.len() != slots.len() || inputs.keys().copied().ne(0..slots.len()) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let source =
            inputs.get(&0).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?.clone();
        let lowerer = self.lowerer();
        for input in inputs.values() {
            if input.wire() != declaration.source_output() {
                return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
            }
        }
        lowerer.validate_refresh_declaration(declaration)?;
        let target_parameters = lowerer.parameters.level(target_level)?;
        let target_ring = lowerer.parameters.ring(target_level)?;
        let target_columns = lowerer.parameters.component_columns(target_level)?;
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
        let mut target_modulus = BigUint::from(1u8);
        for slot in slots {
            target_modulus *= slot
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        }
        if target_modulus != *target_parameters.modulus().as_ref() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        if digits.mask_counts.len() != slots.len() || digits.fresh_count == 0 {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }

        let mut channels = Vec::with_capacity(slots.len());
        for slot in slots {
            channels.push((slot.slot, None::<RnsBggRow>));
        }
        let ordered_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(label, wire)| (label.clone(), wire))
            .collect::<Vec<_>>();
        for index in 0..label_count {
            let label = declaration
                .labels()
                .label(index)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let prf_label =
                labels.get(index).ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let expected_wire = ordered_outputs
                .get(index)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?
                .1;
            let channel_outputs = outputs
                .get(prf_label)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let expected = match label {
                RefreshPrfLabel::Mask { slot, .. } => vec![slot],
                RefreshPrfLabel::FreshError { .. } => slots.iter().map(|slot| slot.slot).collect(),
            };
            if channel_outputs.keys().copied().collect::<Vec<_>>() != expected {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            for slot in expected {
                let value = channel_outputs
                    .get(&slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
                if value.wire() != expected_wire {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                if source.public_payload.is_some() != value.public_payload.is_some() {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                if let (Some(source_payload), Some(payload)) =
                    (&source.public_payload, &value.public_payload)
                {
                    if source_payload.matrix_type() != payload.matrix_type() {
                        return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                    }
                }
                let terminal =
                    lowerer.refresh_action_for_label(declaration, Some(index), slot, "target")?;
                // The preprocessing action already includes the positional
                // digit weight (and H for shared fresh error). Applying it to
                // the row here would weight every PRF digit twice.
                let routed = multiply_encoding_action(value.row.clone(), terminal);
                let (_, accumulator) = channels
                    .get_mut(slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                *accumulator = Some(match accumulator.take() {
                    Some(previous) => add_encoding_rows(previous, routed),
                    None => routed,
                });
            }
        }
        let mut channel_rows = Vec::with_capacity(channels.len());
        let mut channel_weights = Vec::with_capacity(channels.len());
        for (slot, row) in channels {
            let mut row = row.ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let target = lowerer.refresh_action(declaration, slot, "target")?;
            let input =
                inputs.get(&slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            row = add_encoding_rows(multiply_encoding_action(input.row.clone(), target), row);
            // Terminal actions aggregate in the source (q) ring.  Collapse
            // that aggregate once to the slot's producer modulus before the
            // decoder, which is emitted in the p-level ring.
            let slot_metadata = declaration
                .slots()
                .get(slot)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let producer_level = refresh_slot_level(lowerer.parameters, declaration, slot)?;
            row = mod_down_encoding(row, &lowerer.parameters.ring(producer_level)?);
            let decoder = lowerer.refresh_decoder(declaration, slot)?;
            row = add_encoding_rows(row, (-decoder.0, -decoder.1));
            row = mod_down_lift_encoding(
                row,
                &lowerer.parameters.ring_for_modulus(&slot_metadata.q_t)?,
                &target_ring,
            );
            channel_rows.push(row);
            channel_weights.push(lowerer.refresh_crt_weight(declaration, slot)?);
        }
        let values = Family::pack(channel_rows).map_err(graph)?;
        let coefficients = Family::pack(channel_weights).map_err(graph)?;
        let result = recompose_encoding(values, coefficients)?;
        if result.0.matrix_type() != &expected_port || result.1.matrix_type() != &expected_port {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        let public_payload = match source.public_payload.clone() {
            Some(payload) => Some(checked_modulus_convert(payload, &target_ring)?),
            None => None,
        };
        let refreshed_public_key = lowerer.refresh_public_target(declaration, source.wire())?;
        Ok(RnsBggEncoding::new(refreshed_public_key, result, public_payload))
    }
}

fn mod_down_lift_encoding(row: RnsBggRow, channel_ring: &Ring, target_ring: &Ring) -> RnsBggRow {
    map_row(row, |entry| entry.block_mod_switch(channel_ring, 1).centered_rebase(target_ring))
}

fn reduce_anchor_channel(
    a_star: RnsBggMatrix,
    channel_ring: &Ring,
    target_ring: &Ring,
) -> Result<RnsBggMatrix, RnsBggError> {
    let reduced = checked_modulus_convert_matrix(a_star, channel_ring)?;
    Ok(map_matrix(reduced, |entry| entry.centered_rebase(target_ring)))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn refresh_two_prime_encoding_channels_mod_down_then_lift_to_combined_ring() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let source = explicit_ring(&primes, 4);
        let producer = explicit_ring(&[primes[0], primes[2]], 4);
        let combined_modulus = BigInt::from(primes[0]) * primes[1];
        let value = source.input("refresh-source", (1, 1));
        let row = (value.clone(), value);
        let producer_row = mod_down_encoding(row, &producer);
        assert_eq!(modulus_of(producer_row.0.matrix_type()), BigInt::from(primes[0]) * primes[2]);
        let raw_reduction = checked_modulus_convert(producer_row.0.clone(), &producer).unwrap();
        assert_eq!(producer_row.0.flatten(), raw_reduction.flatten());
        let lifted = mod_down_lift_encoding(
            producer_row,
            &explicit_ring(&primes[..1], 4),
            &explicit_ring(&primes[..2], 4),
        );
        assert_eq!(modulus_of(lifted.0.matrix_type()), combined_modulus);
        assert_eq!(modulus_of(lifted.1.matrix_type()), combined_modulus);
    }

    #[test]
    fn refresh_target_uses_only_anchor_and_lifts_from_qt_to_combined_ring() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let ring = explicit_ring(&primes, 4);
        let channel = explicit_ring(&primes[1..2], 4);
        let combined = explicit_ring(&primes[..2], 4);
        let zero = ring.zero((1, 1));
        let zero_matrix = ((zero.clone(), zero.clone()), (zero.clone(), zero.clone()));
        let anchor =
            ((ring.input("refresh-a-star", (1, 1)), zero.clone()), (zero.clone(), zero.clone()));
        let zero_target = reduce_anchor_channel(zero_matrix, &channel, &combined).unwrap();
        let anchored_target = reduce_anchor_channel(anchor, &channel, &combined).unwrap();
        assert_ne!(zero_target.flatten(), anchored_target.flatten());
        assert_eq!(
            modulus_of(anchored_target.0.0.matrix_type()),
            BigInt::from(primes[0]) * primes[1]
        );
    }

    #[test]
    fn refresh_decoder_uses_p_ring_target_without_an_extra_moddown() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let p_modulus = BigInt::from(primes[0]) * primes[2];
        let p_ring = explicit_ring(&[primes[0], primes[2]], 4);
        let sum = p_ring.input("refresh-a-sum-p", (1, 1));
        let zero = p_ring.zero((1, 1));
        let a_sum = ((sum.clone(), zero.clone()), (zero.clone(), zero.clone()));
        let a_star =
            ((p_ring.input("refresh-a-star-p", (1, 1)), zero.clone()), (zero.clone(), zero));
        let h = p_ring.polynomial([IntExpr::constant(BigInt::from(21u8))]);
        let p_target = (
            (a_sum.0.0 - h.clone() * a_star.0.0, a_sum.0.1 - h.clone() * a_star.0.1),
            (a_sum.1.0 - h.clone() * a_star.1.0, a_sum.1.1 - h * a_star.1.1),
        );
        assert_eq!(modulus_of(p_target.0.0.matrix_type()), p_modulus);
    }

    #[test]
    fn refresh_affine_offsets_cover_components_coefficients_and_b2_d1() {
        // b = 2: a mask offset is -(S_d + b^d/2), a fresh one -(b/2) S_d.
        let base = BigUint::from(2u8);
        assert_eq!(canonical_digit_offset(&base, 1, true).unwrap(), BigInt::from(-2));
        assert_eq!(canonical_digit_offset(&base, 2, true).unwrap(), BigInt::from(-5));
        assert_eq!(canonical_digit_offset(&base, 1, false).unwrap(), BigInt::from(-1));
        assert_eq!(canonical_digit_offset(&base, 2, false).unwrap(), BigInt::from(-3));
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let ring = explicit_ring(&[primes[0], primes[2]], 4);
        let columns = 2;
        let mask = refresh_affine_route(&ring, columns, 0, 0, BigInt::from(-2i8)).unwrap();
        let fresh = refresh_affine_route(&ring, columns, 3, 1, BigInt::from(-5i8)).unwrap();
        let offset = add_encoding_rows(mask, fresh);
        assert_eq!(modulus_of(offset.0.matrix_type()), BigInt::from(primes[0]) * primes[2]);
        assert_ne!(offset.flatten(), ring.zero((1, columns)).flatten());
        let secret0 = ring.input("refresh-secret-0", (1, columns));
        let secret1 = ring.input("refresh-secret-1", (1, columns));
        let decoder0 = secret0.clone() - offset.0.clone();
        let decoder1 = secret1.clone() - offset.0.clone();
        assert_ne!(decoder0.flatten(), decoder1.flatten());
    }
}
