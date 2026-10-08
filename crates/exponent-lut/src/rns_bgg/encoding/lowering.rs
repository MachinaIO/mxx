//! Encoding lowering of Program operations and the online evaluation family
//! execution.
use super::*;

/// A family bound for online evaluation.
///
/// A stored family holds complete members. An input family holds only what
/// encryption stores, each member's row and, for a public input, its payload;
/// a member read samples the member's public key from the hash key in place
/// (`sampled_key`), so no stage materializes the keys of a whole family.
#[derive(Clone)]
pub enum OnlineEvalWireFamily {
    Stored(Family<RnsBggEncoding>),
    Input {
        wire: WireId,
        level: usize,
        knowledge: AttributeKnowledge,
        hash_key: Bytes,
        rows: Family<RnsBggRow>,
        payloads: Option<Family<Mat>>,
    },
}

/// A family RHS bound for online evaluation: the stored bottom row of each
/// member's compact ciphertext. A Static member read samples its top row
/// from the hash key in place (`rhs_top`); Dynamic encryption stores the top
/// rows too, since they are sampled per encryption.
#[derive(Clone)]
pub struct OnlineEvalRhsFamily {
    rhs: RhsId,
    program: crate::program::ExponentLutProgramId,
    level: usize,
    hash_key: Bytes,
    tops: Option<Family<RnsBggRow>>,
    bottoms: Family<RnsBggRow>,
}

/// Family execution over graph families, which derives the hash-sampled parts
/// of input and RHS family members where they are read. Online evaluation and
/// preprocessing both use it.
pub struct OnlineEvalFamilies<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
}

impl OnlineEvalFamilies<'_> {
    fn member(
        &self,
        family: &OnlineEvalWireFamily,
        index: IntExpr,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        match family {
            OnlineEvalWireFamily::Stored(family) => Ok(family.at(Int::evaluate(index))),
            OnlineEvalWireFamily::Input { wire, level, knowledge, hash_key, rows, payloads } => {
                let key = super::public_key::sampled_key(
                    self.parameters,
                    *level,
                    hash_key.clone(),
                    *wire,
                    *knowledge,
                    Some(index.clone()),
                )?;
                let index = Int::evaluate(index);
                let payload = payloads.as_ref().map(|payloads| payloads.at(index.clone()));
                Ok(RnsBggEncoding::new(key, rows.at(index), payload))
            }
        }
    }

    fn count(family: &OnlineEvalWireFamily) -> IntExpr {
        match family {
            OnlineEvalWireFamily::Stored(family) => family.count().clone(),
            OnlineEvalWireFamily::Input { rows, .. } => rows.count().clone(),
        }
    }
}

impl<'a, 'b> crate::program::FamilyExecution<EncodingLowering<'a>> for OnlineEvalFamilies<'b> {
    type WireFamily = OnlineEvalWireFamily;
    type RhsFamily = OnlineEvalRhsFamily;

    fn wire_at(
        &mut self,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<RnsBggEncoding, crate::program::ProgramLowerError<RnsBggError>> {
        self.member(family, index.clone()).map_err(crate::program::ProgramLowerError::Backend)
    }

    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<(RnsBggMatrix, RhsRef<Int>), crate::program::ProgramLowerError<RnsBggError>> {
        let member = Int::evaluate(index.clone());
        let top = match &family.tops {
            Some(tops) => tops.at(member.clone()),
            None => super::public_key::rhs_top(
                self.parameters,
                family.program,
                family.hash_key.clone(),
                family.rhs,
                family.level,
                Some(index.clone()),
            )
            .map_err(crate::program::ProgramLowerError::Backend)?,
        };
        Ok((
            (top, family.bottoms.at(member.clone())),
            RhsRef { id: family.rhs, index: Some(member) },
        ))
    }

    fn parallel(
        &mut self,
        evaluator: &mut EncodingLowering<'a>,
        range: &crate::program::FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut EncodingLowering<'a>,
            IntExpr,
        )
            -> Result<RnsBggEncoding, crate::program::ProgramLowerError<RnsBggError>>,
    ) -> Result<Self::WireFamily, crate::program::ProgramLowerError<RnsBggError>> {
        crate::program::graph_parallel(self, evaluator, range, body)
            .map(OnlineEvalWireFamily::Stored)
    }

    fn sum(
        &mut self,
        evaluator: &mut EncodingLowering<'a>,
        family: &Self::WireFamily,
    ) -> Result<RnsBggEncoding, crate::program::ProgramLowerError<RnsBggError>> {
        use crate::program::ProgramLowerError;
        crate::program::graph_sum(evaluator, Self::count(family), |index| {
            let index = index.expression().map_err(ProgramLowerError::Dsl)?;
            self.member(family, index).map_err(ProgramLowerError::Backend)
        })
    }
}

/// Import every RHS of `program`, for binding before the Program is
/// lowered, each paired with its reference. Static preprocessing publishes
/// the bottom row of each compact RHS at the level after its consumer's, and
/// each member's top row is sampled where the member is read; Dynamic
/// encryption publishes both rows. A family RHS is one family per row port.
/// `name` maps a role to its artifact name.
pub(super) fn import_rhs_bindings(
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    workload: &RnsBggDerivedWorkload,
    preprocess: &ProductionId,
    encryption: &ProductionId,
    prelude: Option<&ProductionId>,
    hash_key: &Bytes,
    name: impl Fn(String) -> String,
) -> Result<Vec<(RhsId, InputBinding<(RnsBggMatrix, RhsRef<Int>), OnlineEvalRhsFamily>)>, ApiError>
{
    let dynamic = parameters.rhs_mode() == RhsMode::Dynamic;
    let mut bindings = Vec::with_capacity(program.rhs_inputs().len());
    for (rhs, shape) in program.rhs_inputs() {
        let consumer_level = workload.rhs_level(*rhs).map_err(ApiError::scheme)?;
        let level = consumer_level.checked_add(1).ok_or_else(|| {
            ApiError::scheme(RnsBggError::Parameters("RHS target level overflow"))
        })?;
        let ring = parameters.ring(level).map_err(ApiError::scheme)?;
        let columns = parameters.component_columns(level).map_err(ApiError::scheme)?;
        let binding = match shape {
            InputBinding::Scalar(()) => {
                let port = |port: usize| {
                    let (production, role) = if dynamic {
                        (encryption, format!("rhs/{rhs:?}/ciphertext.{port}"))
                    } else {
                        (preprocess, format!("rhs/{rhs:?}.{}", port - 2))
                    };
                    ring.artifact_input(
                        production.clone(),
                        name(role),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    )
                };
                let top = if dynamic {
                    (port(0), port(1))
                } else {
                    super::public_key::rhs_top(
                        parameters,
                        program.id(),
                        hash_key.clone(),
                        *rhs,
                        consumer_level,
                        None,
                    )
                    .map_err(ApiError::scheme)?
                };
                InputBinding::Scalar(((top, (port(2), port(3))), RhsRef { id: *rhs, index: None }))
            }
            InputBinding::Family(count) => {
                // Dynamic family ciphertexts are the encryption prelude's.
                let production = if dynamic {
                    prelude.ok_or_else(|| {
                        ApiError::compatibility(
                            "Dynamic RHS family ciphertexts need the encryption prelude",
                        )
                    })?
                } else {
                    preprocess
                };
                let row = |ports: [usize; 2]| {
                    let families = ports.map(|port| {
                        let role = if dynamic {
                            format!("rhs/{rhs:?}/family/ciphertext.{port}")
                        } else {
                            format!("rhs/{rhs:?}/family.{}", port - 2)
                        };
                        ring.family_artifact_input(
                            production.clone(),
                            name(role),
                            count.clone(),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    });
                    let element: RnsBggRow =
                        (families[0].at(Int::constant(0)), families[1].at(Int::constant(0)));
                    let fields = families
                        .iter()
                        .map(|family| family.value_handle().clone())
                        .collect::<Vec<_>>();
                    family_of_fields(&element, count.clone(), &fields).map_err(ApiError::scheme)
                };
                InputBinding::Family(OnlineEvalRhsFamily {
                    rhs: *rhs,
                    program: program.id(),
                    level: consumer_level,
                    hash_key: hash_key.clone(),
                    tops: dynamic.then(|| row([0, 1])).transpose()?,
                    bottoms: row([2, 3])?,
                })
            }
        };
        bindings.push((*rhs, binding));
    }
    Ok(bindings)
}

impl EncodingLowering<'_> {
    fn preprocess_artifact_name(&self, role: String) -> String {
        self.encryption_artifact_name(role)
    }

    pub(super) fn encryption_artifact_name(&self, role: String) -> String {
        self.artifact_prefix.as_ref().map_or(role.clone(), |prefix| format!("{prefix}/{role}"))
    }

    /// The production of the RHS and LUT-switching ciphertexts: preprocessing
    /// for Static, encryption for Dynamic (whose RHS family ciphertexts are
    /// in the encryption prelude instead).
    pub(super) fn ciphertext_production(&self) -> &ProductionId {
        match self.parameters.rhs_mode() {
            RhsMode::Static => &self.preprocess,
            RhsMode::Dynamic => &self.encryption,
        }
    }

    /// Compute the native Walsh/automorphism branch polynomial for one unary
    /// LUT branch.  The coefficient depends on the complete table, not only
    /// its width; public-key preparation uses this same helper.
    pub(crate) fn lut_coefficient(
        parameters: &RnsBggParameters,
        level: usize,
        input_layout: ExpLayout,
        representation: crate::program::LutOutputRepresentation,
        sigma: impl Into<IntExpr>,
        values: &[usize],
    ) -> Result<Mat, RnsBggError> {
        let width = values.len();
        if width == 0 || !width.is_power_of_two() {
            return Err(RnsBggError::Lut);
        }
        let ring = parameters.ring(level)?;
        input_layout
            .validate(parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        if input_layout.width() != width {
            return Err(RnsBggError::Lut);
        }
        let modulus = parameters.level(level)?.modulus().as_ref().clone();
        let inverse =
            mxx_backends::utils::mod_inverse_biguints(&(BigUint::from(width) % &modulus), &modulus)
                .ok_or(RnsBggError::Lut)?;
        let sigma = sigma.into();
        let mut terms = values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, output)| {
                let left = match representation {
                    crate::program::LutOutputRepresentation::Coefficient => {
                        ring.polynomial([IntExpr::constant(BigInt::from(output))])
                    }
                    crate::program::LutOutputRepresentation::Exponent(layout) => {
                        layout.mono(&ring, &BigInt::from(output)).map_err(|_| RnsBggError::Lut)?
                    }
                };
                let right = input_layout
                    .mono(&ring, &-BigInt::from(index))
                    .map_err(|_| RnsBggError::Lut)?
                    .ring_automorphism(sigma.clone());
                Ok(left * right)
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        while terms.len() > 1 {
            let mut next = Vec::with_capacity(terms.len().div_ceil(2));
            let mut iterator = terms.into_iter();
            while let Some(left) = iterator.next() {
                next.push(match iterator.next() {
                    Some(right) => left + right,
                    None => left,
                });
            }
            terms = next;
        }
        Ok(ring.polynomial([IntExpr::constant(BigInt::from(inverse))]) * terms.pop().unwrap())
    }

    /// Evaluate a public LUT payload with the same Walsh/automorphism
    /// coefficients used by the online evaluation encoding.  The payload is public, so
    /// the complete evaluation can remain a graph expression; private and
    /// unknown payloads never enter this path. It is the one step of the
    /// online evaluation that reads the table's values.
    fn lut_public_payload(
        &self,
        level: usize,
        payload: Mat,
        table: &LutTable,
    ) -> Result<Mat, RnsBggError> {
        let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let input_layout = table.input_layout();
        input_layout.validate(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let output_level = if table.output_representation().is_coefficient() {
            level
        } else {
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?
        };
        let output_ring = self.parameters.ring(output_level)?;
        let payload = checked_modulus_convert(payload, &output_ring)?;
        let sigmas =
            input_layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        // The output payload is computed in the clear, from the values.
        let values = table
            .values()
            .ok_or(RnsBggError::Binding("the values of a LUT on a public payload are withheld"))?;
        // One loop body over the automorphism indices, as preprocessing builds
        // its switching ciphertexts, so the graph does not grow with the
        // square of the table width.
        let terms = mxx_dsl::parallel(IntExpr::constant(sigmas.len()), |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let coefficient = Self::lut_coefficient(
                self.parameters,
                output_level,
                input_layout,
                table.output_representation(),
                sigma.clone(),
                values,
            )
            .map_err(|_| DslError::Schema)?;
            Ok(coefficient * payload.clone().ring_automorphism(sigma))
        })
        .map_err(graph)?;
        crate::utils::balanced_sum_family(terms, |lhs, rhs| lhs + rhs).map_err(graph)
    }

    fn site(&self) -> Result<&LoweringSite, RnsBggError> {
        self.current_site.as_ref().ok_or(RnsBggError::Binding("missing lowering site"))
    }

    /// The encryption's LUT helpers of `level` for `table`'s input layout,
    /// member `i` for its `i`-th automorphism.
    fn lut_helpers(
        &self,
        level: usize,
        table: &LutTable,
        count: usize,
    ) -> Result<(Family<Mat>, Family<Mat>), RnsBggError> {
        let ring = self.parameters.ring(level)?;
        let columns = self.parameters.component_columns(level)?;
        let layout = table.input_layout();
        let member = |port: usize| {
            ring.family_artifact_input(
                self.encryption.clone(),
                self.encryption_artifact_name(format!("lut-helper/{level}/{layout:?}.{port}")),
                count,
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        };
        Ok((member(0), member(1)))
    }

    fn lut_public_data(&self, level: usize, sigma: Int) -> Result<RnsBggMatrix, RnsBggError> {
        super::public_key::lut_public_matrix(self.parameters, self.hash_key.clone(), level, sigma)
    }

    fn rhs_decomposition(
        &self,
        rhs: &RnsBggMatrix,
        evaluation_level: usize,
    ) -> Result<(SmallMatrix, SmallMatrix), RnsBggError> {
        // Decompose immediately before the RHS action consumes the compact
        // artifact.  The resulting bounded digits are transported to the
        // evaluation ring, but never persisted at the stage boundary.
        let decomposition_level = evaluation_level
            .checked_add(1)
            .ok_or(RnsBggError::Parameters("RHS target level overflow"))?;
        compact_decompose_columns(
            self.parameters,
            evaluation_level,
            decomposition_level,
            rhs.clone(),
        )
    }

    /// The material of LUT `lut_id` evaluated at `level`: the one imported
    /// where the Program was bound for a LUT in a family body, otherwise
    /// imported here.
    fn lut_material(
        &self,
        level: usize,
        lut_id: crate::program::LutId,
        table: &LutTable,
    ) -> Result<super::LutMaterial, RnsBggError> {
        if let Some(material) = self.lut_material.get(&(level, lut_id)) {
            return Ok(material.clone());
        }
        let (switching, sigmas) = self.lut_switching(level, lut_id, table)?;
        let helper = self.lut_helpers(level, table, sigmas.len())?;
        let columns = self.parameters.lut_output_columns(
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?,
            table.output_representation(),
        )?;
        // Dynamic: the switching ciphertexts' digit companions, one block
        // per branch.
        let companions = (self.parameters.rhs_mode() == RhsMode::Dynamic)
            .then(|| {
                self.companion_family(
                    level,
                    columns,
                    lut_switching_role(level, lut_id, table.input_layout(), "/digit-encoding"),
                    IntExpr::constant(sigmas.len()),
                )
            })
            .transpose()?;
        Ok((switching, sigmas, helper, companions))
    }

    /// Import, at root scope, the material of every LUT in a family body
    /// (see [`Self::lut_material`]).
    pub(super) fn import_body_lut_material(&mut self) -> Result<(), RnsBggError> {
        let luts = self.program.encryption_inputs().luts().to_vec();
        for (input, lut_id) in self.program.family_body_luts() {
            let level = self.workload.level(input)?;
            let table = luts.get(lut_id.0 as usize).ok_or(RnsBggError::Lut)?;
            let material = self.lut_material(level, lut_id, table)?;
            self.lut_material.insert((level, lut_id), material);
        }
        Ok(())
    }

    /// The imported compact `B_sigma` switching ciphertexts of one LUT
    /// evaluated at `evaluation_level`, one family per matrix port, with its
    /// automorphism branch indices.
    fn lut_switching(
        &self,
        evaluation_level: usize,
        lut_id: crate::program::LutId,
        table: &LutTable,
    ) -> Result<(Vec<Family<Mat>>, Vec<usize>), RnsBggError> {
        let count = IntExpr::constant(table.input_width());
        let decomposition_level =
            evaluation_level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let decomposition_ring = self.parameters.ring(decomposition_level)?;
        let input_layout = table.input_layout();
        let ring_dimension = self.parameters.level(evaluation_level)?.ring_dimension() as usize;
        input_layout.validate(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let target_columns = self
            .parameters
            .lut_output_columns(decomposition_level, table.output_representation())?;
        let sigma_indices =
            input_layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let families = (0..4)
            .map(|port| {
                decomposition_ring.family_artifact_input(
                    self.ciphertext_production().clone(),
                    self.preprocess_artifact_name(lut_switching_role(
                        evaluation_level,
                        lut_id,
                        input_layout,
                        &format!(".{port}"),
                    )),
                    count.clone(),
                    (1, target_columns),
                    ArtifactAvailability::Transferred,
                )
            })
            .collect::<Vec<_>>();
        Ok((families, sigma_indices))
    }

    /// For a LUT branch with automorphism `sigma` and switching ciphertext
    /// `b_sigma`: the routed input row switched by `b_sigma`, and the
    /// decomposition of the switched public key. The switching decomposition acts on the
    /// routed row and public key in one product each.
    fn lut_branch_action(
        &self,
        input: &RnsBggEncoding,
        b_sigma: RnsBggMatrix,
        sigma: IntExpr,
    ) -> Result<(RnsBggRow, (SmallMatrix, SmallMatrix)), DslError> {
        let evaluation_level = self.workload.level(input.wire()).map_err(|_| DslError::Schema)?;
        let decomposition_level = evaluation_level.checked_add(1).ok_or(DslError::Schema)?;
        let target_ring =
            self.parameters.ring(decomposition_level).map_err(|_| DslError::Schema)?;
        let switching_decomposition = compact_decompose_columns(
            self.parameters,
            evaluation_level,
            decomposition_level,
            b_sigma.clone(),
        )
        .map_err(|_| DslError::Schema)?;
        let routed = map_matrix(input.public_key.matrix.clone(), |entry| {
            entry.ring_automorphism(sigma.clone())
        });
        let routed_row = (
            input.row.0.clone().ring_automorphism(sigma.clone()),
            input.row.1.clone().ring_automorphism(sigma.clone()),
        );
        let (switched_row, switched_matrix) =
            multiply_row_and_public_by_columns(routed_row, routed, switching_decomposition);
        let switched_public_key = mod_down_public(switched_matrix, &target_ring);
        let switched_key_decomposition = compact_decompose_columns(
            self.parameters,
            evaluation_level,
            decomposition_level,
            switched_public_key,
        )
        .map_err(|_| DslError::Schema)?;
        Ok((switched_row, switched_key_decomposition))
    }

    /// Finalize declared coefficient outputs once after graph aggregation:
    /// ModDown to the next level, then the scale-up action `G^-1(k)` (see
    /// [`RnsBggParameters::coefficient_output_action`]).
    pub(super) fn finish_outputs(
        &mut self,
        outputs: Vec<RnsBggEncoding>,
    ) -> Result<Vec<RnsBggEncoding>, RnsBggError> {
        outputs
            .into_iter()
            .map(|output| {
                if self
                    .program
                    .wire_is_coefficient_output(output.wire())
                    .map_err(|_| RnsBggError::Binding("invalid output wire metadata"))?
                {
                    let level = self.workload.level(output.wire())?;
                    let target = level
                        .checked_add(1)
                        .ok_or(RnsBggError::Parameters("coefficient output level overflow"))?;
                    let ring = self.parameters.ring(target)?;
                    let (_, k, _) = self.parameters.coefficient_output_decoding(
                        &self.program,
                        output.wire(),
                        target,
                    )?;
                    let action = self.parameters.coefficient_output_action(target, &k)?;
                    let act = |entry: Mat| action.clone().right_multiply(entry);
                    let public_key = RnsBggPublicKey::new(
                        output.public_key.wire,
                        map_matrix(mod_down_public(output.public_key.matrix, &ring), act),
                    );
                    Ok(RnsBggEncoding::new(
                        public_key,
                        map_row(mod_down_encoding(output.row, &ring), act),
                        output
                            .public_payload
                            .map(|payload| checked_modulus_convert(payload, &ring))
                            .transpose()?,
                    ))
                } else {
                    Ok(output)
                }
            })
            .collect()
    }

    fn checked_payload(
        &self,
        wire: WireId,
        payload: Option<Mat>,
    ) -> Result<Option<Mat>, RnsBggError> {
        let public = matches!(
            self.program.wire_knowledge(wire).map_err(|_| RnsBggError::Binding("wire metadata"))?,
            AttributeKnowledge::PublicPayload
        );
        if public != payload.is_some() {
            return Err(RnsBggError::Binding("public payload does not match Program metadata"));
        }
        Ok(payload)
    }
}

impl ExponentLUTProgramLowering for EncodingLowering<'_> {
    type Wire = RnsBggEncoding;
    /// The RHS ciphertext with its reference, which names a Dynamic RHS's
    /// companion keys and encodings.
    type Rhs = (RnsBggMatrix, RhsRef<Int>);
    type Error = RnsBggError;
    /// The key, the rows and a public payload are each multiplied by the
    /// plaintext `c` (see the public-key lowering).
    fn mul_small_constant(
        &mut self,
        input: &Self::Wire,
        coefficients: &[i64],
        shift: &IntExpr,
    ) -> Result<Self::Wire, Self::Error> {
        let ring = self.parameters.ring(self.workload.level(input.wire())?)?;
        let multiply =
            |entry: Mat| super::mul_small_constant_entry(&ring, entry, coefficients, shift);
        let output = self.site()?.output;
        let public_key =
            RnsBggPublicKey::new(output, map_matrix(input.public_key.matrix.clone(), multiply));
        let public_payload = input.public_payload.clone().map(multiply);
        let public_payload = self.checked_payload(output, public_payload)?;
        Ok(RnsBggEncoding::new(
            public_key,
            (multiply(input.row.0.clone()), multiply(input.row.1.clone())),
            public_payload,
        ))
    }
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        if self.workload.level(lhs.wire())? != self.workload.level(rhs.wire())? {
            return Err(RnsBggError::Binding("Add key level mismatch"));
        }
        let public_payload = match (&lhs.public_payload, &rhs.public_payload) {
            (Some(lhs), Some(rhs)) => Some(lhs.clone() + rhs.clone()),
            (None, None) => None,
            _ => return Err(RnsBggError::Binding("mixed public/private Add")),
        };
        let output = self.site()?.output;
        let public_key = RnsBggPublicKey::new(
            output,
            add_public_matrices(lhs.public_key.matrix.clone(), rhs.public_key.matrix.clone()),
        );
        let public_payload = self.checked_payload(output, public_payload)?;
        Ok(RnsBggEncoding::new(
            public_key,
            add_encoding_rows(lhs.row.clone(), rhs.row.clone()),
            public_payload,
        ))
    }
    fn mul_public(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let lhs_knowledge = self
            .program
            .wire_knowledge(lhs.wire())
            .map_err(|_| RnsBggError::Binding("MulPublic LHS metadata"))?;
        let rhs_knowledge = self
            .program
            .wire_knowledge(rhs.wire())
            .map_err(|_| RnsBggError::Binding("MulPublic RHS metadata"))?;
        if lhs_knowledge != AttributeKnowledge::PublicPayload {
            return Err(RnsBggError::Binding("MulPublic LHS must be public"));
        }
        let lhs_level = self.workload.level(lhs.wire())?;
        let rhs_level = self.workload.level(rhs.wire())?;
        let target_level =
            self.parameters.mul_public_target_level(lhs_level, rhs_level, rhs_knowledge)?;
        let target_ring = self.parameters.ring(target_level)?;

        // The public LHS is aligned to the private RHS-owned level. The RHS
        // remains untouched when private; public/public additionally aligns
        // its public carrier to the selected lower modulus.
        let lhs_matrix = apply_public_alignment_matrix(
            self.parameters,
            lhs_level,
            target_level,
            lhs.public_key.matrix.clone(),
        )?;
        let lhs_row =
            apply_public_alignment_row(self.parameters, lhs_level, target_level, lhs.row.clone())?;
        let rhs_matrix = if rhs_knowledge == AttributeKnowledge::PublicPayload {
            apply_public_alignment_matrix(
                self.parameters,
                rhs_level,
                target_level,
                rhs.public_key.matrix.clone(),
            )?
        } else {
            rhs.public_key.matrix.clone()
        };
        let rhs_row = if rhs_knowledge == AttributeKnowledge::PublicPayload {
            apply_public_alignment_row(self.parameters, rhs_level, target_level, rhs.row.clone())?
        } else {
            rhs.row.clone()
        };
        let lhs_payload = checked_modulus_convert(
            lhs.public_payload.clone().ok_or(RnsBggError::Binding("public LHS payload missing"))?,
            &target_ring,
        )?;
        let rhs_payload = rhs
            .public_payload
            .clone()
            .map(|payload| checked_modulus_convert(payload, &target_ring))
            .transpose()?;
        // One decomposition of the RHS key acts on the LHS row and key.
        let columns =
            compact_decompose_columns(self.parameters, target_level, target_level, rhs_matrix)?;
        let (acted_row, acted_matrix) =
            multiply_row_and_public_by_columns(lhs_row, lhs_matrix, columns);
        let row = add_encoding_rows(
            acted_row,
            (rhs_row.0.clone() * lhs_payload.clone(), rhs_row.1.clone() * lhs_payload.clone()),
        );
        let output = self.site()?.output;
        let public_key = RnsBggPublicKey::new(output, acted_matrix);
        let output_payload = rhs_payload.map(|rhs_payload| lhs_payload * rhs_payload);
        Ok(RnsBggEncoding::new(public_key, row, self.checked_payload(output, output_payload)?))
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.wire())
            .map_err(|_| RnsBggError::Binding("MulPrivate input metadata"))? ==
            AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Binding("RNS-BGG MulPrivate requires a private input"));
        }
        let evaluation_level = self.workload.level(input.wire())?;
        let (ciphertext, reference) = rhs;
        let output = self.site()?.output;
        let target_level = self.workload.level(output)?;
        let target_ring = self.parameters.ring(target_level)?;
        let (row, matrix) = match self.parameters.rhs_mode() {
            RhsMode::Static => multiply_row_and_public_by_columns(
                input.row.clone(),
                input.public_key.matrix.clone(),
                self.rhs_decomposition(ciphertext, evaluation_level)?,
            ),
            // Dynamic: `c Dec(C)` plus the companion row, with the key
            // fixed in preprocessing from public companion keys (Eq. 049).
            RhsMode::Dynamic => {
                let companions = self
                    .rhs_companions
                    .get(&reference.id)
                    .ok_or(RnsBggError::Binding("Dynamic RHS companions are not imported"))?;
                let member = reference
                    .index
                    .as_ref()
                    .map(Int::expression)
                    .transpose()
                    .map_err(|_| RnsBggError::Binding("RHS member index"))?;
                let columns = self.parameters.component_columns(evaluation_level + 1)?;
                let rows = self.companion_rows(
                    evaluation_level,
                    &input.public_key.matrix,
                    columns,
                    format!("rhs/{:?}", reference.id),
                    member,
                    None,
                    companions,
                )?;
                let product = super::column_action(
                    self.parameters,
                    evaluation_level,
                    evaluation_level + 1,
                    ciphertext,
                    columns,
                    vec![super::join_row(&input.row)],
                )?;
                let split = |row: &Mat| super::split_row(row, columns);
                (split(&(product[0].clone() + rows[2].clone())), (split(&rows[0]), split(&rows[1])))
            }
        };
        let public_key = RnsBggPublicKey::new(output, mod_down_public(matrix, &target_ring));
        Ok(RnsBggEncoding::new(
            public_key,
            mod_down_encoding(row, &target_ring),
            self.checked_payload(output, None)?,
        ))
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.wire())
            .map_err(|_| RnsBggError::Binding("LUT input metadata"))? ==
            AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Binding("RNS-BGG LUT requires a private input"));
        }
        let lut_id = self.program.lut_id_at(self.site()?.output).map_err(|_| RnsBggError::Lut)?;
        let level = self.workload.level(input.wire())?;
        let (switching, sigmas, helper, companions) = self.lut_material(level, lut_id, table)?;
        let output = self.site()?.output;
        let columns = self.parameters.lut_output_columns(
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?,
            table.output_representation(),
        )?;
        // Accumulate the branches sequentially, as preprocessing does: the sum
        // and its noise bound equal the balanced tree, while only one branch,
        // including its sampled A_sigma, is resident at a time.
        let evaluation_ring = self.parameters.ring(level)?;
        let zero = evaluation_ring.zero((1, columns));
        let initial =
            ((zero.clone(), zero.clone()), ((zero.clone(), zero.clone()), (zero.clone(), zero)));
        let b_at = |index: Int| {
            (
                (switching[0].at(index.clone()), switching[1].at(index.clone())),
                (switching[2].at(index.clone()), switching[3].at(index)),
            )
        };
        let this = &*self;
        let branch = |index: Int, (row_sum, key_sum), b_sigma| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let (switched_row, switched_key_decomposition) =
                this.lut_branch_action(input, b_sigma, sigma.clone())?;
            let mask = (helper.0.at(index.clone()), helper.1.at(index));
            let public_data =
                this.lut_public_data(level, Int::from(sigma)).map_err(|_| DslError::Schema)?;
            let (masked, switched_key) =
                multiply_row_and_public_by_columns(mask, public_data, switched_key_decomposition);
            let row = add_encoding_rows(row_sum, add_encoding_rows(masked, switched_row));
            let public_key = add_public_matrices(key_sum, switched_key);
            Ok::<_, DslError>((row, public_key))
        };
        // Each iteration first reads the next branch's switching ciphertext
        // and carries it to the next iteration, so that load overlaps this
        // branch's work; the last branch runs after the loop.
        let last = table.input_width().checked_sub(1).ok_or(RnsBggError::Lut)?;
        let (row, public_key) = if let Some(companions) = companions {
            // Dynamic: one companion loop first builds every branch's key
            // rows `T_s` and companion row `E_s` (block `s` of each row), so
            // the stored companions are read in one loop nest; the branch loop
            // then adds `sigma(c) Dec(C_sigma) + E_s` and acts with the
            // rounded `T_s` on the helper and `A_sigma`, one column at a
            // time.
            let rows = self.companion_rows(
                level,
                &input.public_key.matrix,
                columns,
                format!("lut/{lut_id:?}"),
                None,
                Some(&sigmas),
                &companions,
            )?;
            let width = 2 * columns;
            let next = level + 1;
            let target_ring = self.parameters.ring(next)?;
            let split = |row: &Mat| super::split_row(row, columns);
            let schema = |_| DslError::Schema;
            let count = sigmas.len();
            let ((row, key), _) = mxx_dsl::iterate(
                IntExpr::constant(count),
                (initial, b_at(Int::constant(0))),
                |index, ((row_sum, key_sum), b_sigma)| {
                    // Read the next branch's switching ciphertext (the first
                    // again after the last) while this branch runs.
                    let b_next = b_at(Int::evaluate(
                        IntExpr::Rem(
                            Box::new(index.expression()? + IntExpr::constant(1)),
                            Box::new(IntExpr::constant(count)),
                        )
                        .canonicalize(),
                    ));
                    let start = (index.expression()? * IntExpr::constant(width)).canonicalize();
                    let block = |row: &Mat| {
                        split(&row.clone().slice(
                            None,
                            Some(mxx_ir_core::node::IndexRange {
                                start: start.clone(),
                                end: (start.clone() + IntExpr::constant(width)).canonicalize(),
                            }),
                        ))
                    };
                    let sigma = IntExpr::Select {
                        selector: Box::new(index.expression()?),
                        branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
                    };
                    let routed_row =
                        map_row(input.row.clone(), |entry| entry.ring_automorphism(sigma.clone()));
                    let product = super::column_action(
                        this.parameters,
                        level,
                        next,
                        &b_sigma,
                        columns,
                        vec![super::join_row(&routed_row)],
                    )
                    .map_err(schema)?;
                    let switched_row = add_encoding_rows(split(&product[0]), block(&rows[2]));
                    let switched_key =
                        mod_down_public((block(&rows[0]), block(&rows[1])), &target_ring);
                    let mask = (helper.0.at(index.clone()), helper.1.at(index));
                    let data = this.lut_public_data(level, Int::from(sigma)).map_err(schema)?;
                    let actions = super::column_action(
                        this.parameters,
                        level,
                        next,
                        &switched_key,
                        columns,
                        vec![
                            super::join_row(&mask),
                            super::join_row(&data.0),
                            super::join_row(&data.1),
                        ],
                    )
                    .map_err(schema)?;
                    let row = add_encoding_rows(
                        row_sum,
                        add_encoding_rows(split(&actions[0]), switched_row),
                    );
                    let key =
                        add_public_matrices(key_sum, (split(&actions[1]), split(&actions[2])));
                    Ok(((row, key), b_next))
                },
            )
            .map_err(graph)?;
            (row, key)
        } else {
            let (sums, b_last) = mxx_dsl::iterate(
                IntExpr::constant(last),
                (initial, b_at(Int::constant(0))),
                |index, (sums, b_sigma)| {
                    let next = b_at(index.clone().add(1usize));
                    Ok((branch(index, sums, b_sigma)?, next))
                },
            )
            .map_err(graph)?;
            branch(Int::constant(last), sums, b_last).map_err(graph)?
        };
        if let Some(declaration) = self.decoder_smudge.clone() &&
            declaration.smudge().lut() == lut_id
        {
            return self.smudge_member(&declaration, level, output, row, public_key);
        }
        let mut public_key = RnsBggPublicKey::new(output, public_key);
        let mut row = row;
        if !self
            .program
            .wire_is_coefficient_output(output)
            .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
        {
            let target = self.workload.level(output)?;
            let ring = self.parameters.ring(target)?;
            public_key.matrix = mod_down_public(public_key.matrix, &ring);
            row = mod_down_encoding(row, &ring);
        }
        let public_payload = if matches!(
            self.program
                .wire_knowledge(output)
                .map_err(|_| RnsBggError::Binding("wire metadata"))?,
            AttributeKnowledge::PublicPayload
        ) {
            let payload = input
                .public_payload
                .clone()
                .ok_or(RnsBggError::Binding("public LUT input payload missing"))?;
            let payload = self.lut_public_payload(level, payload, table)?;
            let target_level = self.workload.level(output)?;
            Some(checked_modulus_convert(payload, &self.parameters.ring(target_level)?)?)
        } else {
            None
        };
        Ok(RnsBggEncoding::new(public_key, row, self.checked_payload(output, public_payload)?))
    }
}

impl EncodingLowering<'_> {
    /// Finalize the raw (scale-1) coefficient LUT carrier of one decoder smudge member at
    /// `level` once, to `level + 1`, and apply its member transform (see
    /// [`super::smudge_member_action`]). The smudge `Sum` then adds the
    /// transformed members inside the root loop.
    fn smudge_member(
        &self,
        declaration: &DecoderDeclaration,
        level: usize,
        output: WireId,
        row: RnsBggRow,
        public_key: RnsBggMatrix,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        let finalized =
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let ring = self.parameters.ring(finalized)?;
        let member = self
            .site()?
            .indices
            .first()
            .cloned()
            .ok_or(RnsBggError::Binding("a decoder smudge LUT must be a root family member"))?;
        let (action, coefficient) =
            super::smudge_member_action(self.parameters, declaration, member, finalized)?;
        let act = |entry| super::act_on_entry(entry, &action, &coefficient);
        Ok(RnsBggEncoding::new(
            RnsBggPublicKey::new(output, map_matrix(mod_down_public(public_key, &ring), act)),
            map_row(mod_down_encoding(row, &ring), act),
            None,
        ))
    }
}

impl ProgramLoweringContext for EncodingLowering<'_> {
    fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl GraphParallelLowering for EncodingLowering<'_> {
    type SideOutput = Vec<Mat>;
    fn begin_graph_parallel(&mut self, _: &IntExpr, _: &IntExpr) {}
    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
        Ok(Vec::new())
    }
    fn commit_graph_parallel(&mut self, _: Family<Self::SideOutput>) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use mxx_backends::{MemoryArtifactStore, backend::poly::cpu_backend};

    #[test]
    fn lut_coefficients_evaluate_values_with_distinct_input_and_output_layouts() {
        let level = DCRTPolyParams::new(16, 1, 17, 2, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let input_layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        let output_layout = ExpLayout { stride_log2: 2, width_log2: 2 };
        let mut graph = DslContext::new("lut-layout-arithmetic");
        let values = [0usize, 1, 3, 2];
        for (role, representation) in [
            crate::program::LutOutputRepresentation::Coefficient,
            crate::program::LutOutputRepresentation::Exponent(output_layout),
        ]
        .into_iter()
        .enumerate()
        {
            for (input, expected) in values.iter().copied().enumerate() {
                let monomial = input_layout.mono(&ring, &BigInt::from(input)).unwrap();
                let mut actual = ring.zero((1, 1));
                for sigma in input_layout.automorphism_indices(16).unwrap() {
                    let coefficient = EncodingLowering::lut_coefficient(
                        &parameters,
                        0,
                        input_layout,
                        representation,
                        IntExpr::constant(sigma),
                        &values,
                    )
                    .unwrap();
                    actual = actual + coefficient * monomial.clone().ring_automorphism(sigma);
                }
                let expected = match representation {
                    crate::program::LutOutputRepresentation::Coefficient => {
                        ring.polynomial([IntExpr::constant(expected)])
                    }
                    crate::program::LutOutputRepresentation::Exponent(layout) => {
                        layout.mono(&ring, &BigInt::from(expected)).unwrap()
                    }
                };
                graph =
                    graph.output(format!("difference-{role}-{input}"), actual - expected).unwrap();
            }
        }
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend([level]));
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            StageGraph::new(graph.build().unwrap(), BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
            let matrix = crate::test_support::full(&result.outputs[&name]);
            assert!(
                matrix
                    .entry(0, 0)
                    .coeffs_biguints()
                    .iter()
                    .all(|value| *value == BigUint::from(0u8)),
                "{name}"
            );
        }
    }

    #[test]
    fn lut_coefficients_depend_on_values_not_only_width() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let constant = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 0],
        )
        .unwrap();
        let identity = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 1],
        )
        .unwrap();
        assert_ne!(constant.flatten(), identity.flatten());
        let declared_layout = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Exponent(ExpLayout {
                stride_log2: 1,
                width_log2: 1,
            }),
            IntExpr::constant(1),
            &[0, 1],
        )
        .unwrap();
        assert_ne!(identity.flatten(), declared_layout.flatten());
    }

    #[test]
    fn lowering_lut_path_consumes_table_coefficients() {
        fn lower(
            values: Vec<usize>,
            representation: crate::program::LutOutputRepresentation,
        ) -> RnsBggEncoding {
            let table = LutTable::unary(
                2,
                ExpLayout { stride_log2: 0, width_log2: 1 },
                2,
                values,
                representation,
            )
            .unwrap();
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let lut = builder.register_lut(table.clone()).unwrap();
            let output = builder.lut(Operand::Scalar(input), lut).unwrap();
            let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
            let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
            let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
            let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
            let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
            let production = ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([7; 32]),
                execution_nonce: [9; 32],
            };
            let mut lowerer = EncodingLowering {
                parameters: &parameters,
                hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
                program,
                workload,
                preprocess: production.clone(),
                encryption: production,
                manifests: BTreeMap::new(),
                decoder_smudge: None,
                artifact_prefix: None,
                context: [0; 32],
                rhs_companions: BTreeMap::new(),
                lut_material: BTreeMap::new(),
                current_site: Some(LoweringSite {
                    output,
                    indices: Vec::new(),
                    ranges: Vec::new(),
                }),
            };
            let ring = parameters.ring(0).unwrap();
            let columns = parameters.component_columns(0).unwrap();
            let value = ring.input("lut-lowering-input", (1, columns));
            let key = RnsBggPublicKey::new(
                input,
                ((value.clone(), value.clone()), (value.clone(), value.clone())),
            );
            let input = RnsBggEncoding::new(key, (value.clone(), value), None);
            <EncodingLowering<'_> as ExponentLUTProgramLowering>::lut(&mut lowerer, &input, &table)
                .unwrap()
        }

        let constant = lower(vec![0, 0], crate::program::LutOutputRepresentation::Coefficient);
        let identity = lower(vec![0, 1], crate::program::LutOutputRepresentation::Coefficient);
        assert_ne!(constant.flatten(), identity.flatten());
        let exponent = lower(
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(ExpLayout {
                stride_log2: 0,
                width_log2: 1,
            }),
        );
        assert_ne!(identity.flatten(), exponent.flatten());
    }

    #[test]
    fn rhs_imports_use_next_level_compact_schema_for_scalar_and_family() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let source_columns = parameters.component_columns(0).unwrap();
        let compact_columns = parameters.component_columns(1).unwrap();
        assert_ne!(source_columns, compact_columns);
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([29; 32]),
            execution_nonce: [31; 32],
        };
        let compact_type = parameters.ring(1).unwrap().matrix_type((1, compact_columns));
        let decomposed_type = parameters
            .ring(0)
            .unwrap()
            .matrix_type((2 * parameters.level(1).unwrap().modulus_digits(), compact_columns));
        let decomposed_bound = IntExpr::RoundDiv(
            Box::new(IntExpr::constant(
                BigInt::from(1u8) << parameters.level(1).unwrap().base_bits(),
            )),
            Box::new(IntExpr::constant(BigInt::from(2u8))),
        )
        .canonicalize();
        let lowerer = |program: &Arc<ExponentLutProgram>, output| EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program: Arc::clone(program),
            workload: Arc::new(RnsBggDerivedWorkload::from_program(program).unwrap()),
            preprocess: production.clone(),
            encryption: production.clone(),
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let assert_compact = |matrix: &RnsBggMatrix| {
            for entry in [&matrix.0.0, &matrix.0.1, &matrix.1.0, &matrix.1.1] {
                assert_eq!(entry.matrix_type(), &compact_type);
            }
        };

        let mut scalar_builder = crate::program::ExponentLutProgramBuilder::new();
        let scalar_input = scalar_builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let scalar_rhs = scalar_builder.rhs_input().unwrap();
        let scalar_output = scalar_builder
            .mul_private(Operand::Scalar(scalar_input), Operand::Scalar(scalar_rhs))
            .unwrap();
        let scalar_program =
            Arc::new(scalar_builder.finish(vec![Operand::Scalar(scalar_output)]).unwrap());
        let scalar_lowerer = lowerer(&scalar_program, scalar_output);
        let hash_key = scalar_lowerer.hash_key.clone();
        let bindings = import_rhs_bindings(
            &parameters,
            &scalar_program,
            &scalar_lowerer.workload,
            &production,
            &production,
            None,
            &hash_key,
            |role| role,
        )
        .unwrap();
        let [(id, InputBinding::Scalar((scalar_compact, _)))] = bindings.as_slice() else {
            panic!("one scalar RHS binding expected");
        };
        assert_eq!(*id, scalar_rhs);
        assert_compact(scalar_compact);
        let scalar_action = scalar_lowerer.rhs_decomposition(scalar_compact, 0).unwrap();
        assert_eq!(scalar_action.0.matrix_type(), &decomposed_type);
        assert_eq!(scalar_action.0.max_coefficient_bound(), &decomposed_bound);
        let scalar_name = format!("rhs/{scalar_rhs:?}");
        let scalar_graph = DslContext::new("rhs-scalar-consumer-schema")
            .transferred_output(scalar_name.clone(), scalar_action)
            .unwrap()
            .build()
            .unwrap();
        assert!(scalar_graph.graph.outputs().contains_key(&format!("{scalar_name}.0")));

        let mut family_builder = crate::program::ExponentLutProgramBuilder::new();
        let family_input = family_builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let family_rhs = family_builder.rhs_family(IntExpr::constant(2)).unwrap();
        let family_products = family_builder
            .parallel(
                crate::program::FamilyRange::full(IntExpr::constant(2)).unwrap(),
                |builder, index| {
                    builder
                        .mul_private(
                            Operand::Scalar(family_input),
                            Operand::Member { family: family_rhs, index },
                        )
                        .map(Operand::Scalar)
                },
            )
            .unwrap();
        let family_output = family_builder.sum(family_products).unwrap();
        let family_program =
            Arc::new(family_builder.finish(vec![Operand::Scalar(family_output)]).unwrap());
        let family_lowerer = lowerer(&family_program, family_output);
        let bindings = import_rhs_bindings(
            &parameters,
            &family_program,
            &family_lowerer.workload,
            &production,
            &production,
            None,
            &hash_key,
            |role| role,
        )
        .unwrap();
        let [(id, InputBinding::Family(family))] = bindings.as_slice() else {
            panic!("one family RHS binding expected");
        };
        assert_eq!(*id, family_rhs);
        // Only the bottom rows are imported, as the port families themselves:
        // no loop reads them.
        assert_eq!(family.bottoms.flatten().len(), 2);
        assert!(family.bottoms.flatten().iter().all(|value| matches!(
            value.node().kind(),
            mxx_ir_core::node::NodeKind::Input { artifact: Some(_), .. }
        )));
        let mut families = OnlineEvalFamilies { parameters: &parameters };
        let mut member_at = |index: IntExpr| {
            <OnlineEvalFamilies<'_> as crate::program::FamilyExecution<EncodingLowering<'_>>>::rhs_at(
                &mut families,
                family,
                &index,
            )
            .unwrap()
            .0
        };
        let member = member_at(IntExpr::constant(1));
        assert_compact(&member);
        let family_action = family_lowerer.rhs_decomposition(&member, 0).unwrap();
        assert_eq!(family_action.0.matrix_type(), &decomposed_type);
        assert_eq!(family_action.0.max_coefficient_bound(), &decomposed_bound);
        let family_name = format!("rhs/{family_rhs:?}/family");
        let family_members = mxx_dsl::parallel(IntExpr::constant(2), |index| {
            family_lowerer
                .rhs_decomposition(&member_at(index.expression()?), 0)
                .map_err(|_| DslError::Schema)
        })
        .unwrap();
        let family_graph = DslContext::new("rhs-family-consumer-schema")
            .transferred_output(family_name.clone(), family_members)
            .unwrap()
            .build()
            .unwrap();
        assert!(family_graph.graph.outputs().contains_key(&format!("{family_name}.0")));
    }

    #[test]
    fn mul_public_uses_only_second_payload_and_propagates_public_product() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_wire = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let output = builder
            .mul_public(Operand::Scalar(public_wire), Operand::Scalar(private_wire))
            .unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([19; 32]),
            execution_nonce: [23; 32],
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: production.clone(),
            encryption: production.clone(),
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let ring = parameters.ring(0).unwrap();
        let private_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PrivatePayload).unwrap();
        let public_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let private_row = ring.input("mul-public-private-row", (1, private_columns));
        let public_row = ring.input("mul-public-public-row", (1, public_columns));
        let public_payload = ring.input("mul-public-payload", (1, 1));
        let private_key = RnsBggPublicKey::new(
            private_wire,
            (
                (private_row.clone(), private_row.clone()),
                (private_row.clone(), private_row.clone()),
            ),
        );
        let public_key = RnsBggPublicKey::new(
            public_wire,
            ((public_row.clone(), public_row.clone()), (public_row.clone(), public_row.clone())),
        );
        let private =
            RnsBggEncoding::new(private_key, (private_row.clone(), private_row.clone()), None);
        let public = RnsBggEncoding::new(
            public_key.clone(),
            (public_row.clone(), public_row.clone()),
            Some(public_payload.clone()),
        );
        let result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &public,
            &private,
        )
        .unwrap();
        assert!(result.public_payload.is_none());
        let public_with_other_payload = RnsBggEncoding::new(
            public_key.clone(),
            (public.row.0.clone(), public.row.1.clone()),
            Some(ring.input("mul-public-other-payload", (1, 1))),
        );
        let other_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &public_with_other_payload,
            &private,
        )
        .unwrap();
        assert_ne!(result.row.flatten(), other_result.row.flatten());

        let mut public_builder = crate::program::ExponentLutProgramBuilder::new();
        let public_lhs_wire = public_builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let public_rhs_wire = public_builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let public_output = public_builder
            .mul_public(Operand::Scalar(public_lhs_wire), Operand::Scalar(public_rhs_wire))
            .unwrap();
        let public_program =
            Arc::new(public_builder.finish(vec![Operand::Scalar(public_output)]).unwrap());
        let public_workload =
            Arc::new(RnsBggDerivedWorkload::from_program(&public_program).unwrap());
        let mut public_lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program: public_program,
            workload: public_workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: Some(LoweringSite {
                output: public_output,
                indices: Vec::new(),
                ranges: Vec::new(),
            }),
        };
        let lhs_payload = ring.input("mul-public-lhs-payload", (1, 1));
        let rhs_payload = public_payload.clone();
        let public_lhs = RnsBggEncoding::new(
            RnsBggPublicKey::new(
                public_lhs_wire,
                (
                    (public_row.clone(), public_row.clone()),
                    (public_row.clone(), public_row.clone()),
                ),
            ),
            (public_row.clone(), public_row.clone()),
            Some(lhs_payload.clone()),
        );
        let public_rhs = RnsBggEncoding::new(
            RnsBggPublicKey::new(
                public_rhs_wire,
                (
                    (public_row.clone(), public_row.clone()),
                    (public_row.clone(), public_row.clone()),
                ),
            ),
            (public_row.clone(), public_row),
            Some(rhs_payload.clone()),
        );
        let public_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut public_lowerer,
            &public_lhs,
            &public_rhs,
        )
        .unwrap();
        let output_payload = public_result.public_payload.expect("public product payload");
        assert_eq!(output_payload.matrix_type(), &ring.matrix_type((1, 1)));
    }

    #[test]
    fn mul_public_aligns_public_carrier_before_private_same_level_action() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let private_level_one =
            builder.mul_private(Operand::Scalar(private_input), Operand::Scalar(rhs)).unwrap();
        let output = builder
            .mul_public(Operand::Scalar(public_input), Operand::Scalar(private_level_one))
            .unwrap();
        // Also an output, the public input stays at level zero and is aligned.
        let program = Arc::new(
            builder.finish(vec![Operand::Scalar(output), Operand::Scalar(public_input)]).unwrap(),
        );
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([41; 32]),
            execution_nonce: [43; 32],
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let low_ring = parameters.ring(1).unwrap();
        let high_ring = parameters.ring(0).unwrap();
        let private_columns =
            parameters.key_component_columns(1, AttributeKnowledge::PrivatePayload).unwrap();
        let public_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let private_entry = low_ring.input("mul-public-private-level-one", (1, private_columns));
        let public_entry = high_ring.input("mul-public-public-level-zero", (1, public_columns));
        let private_key = RnsBggPublicKey::new(
            private_level_one,
            (
                (private_entry.clone(), private_entry.clone()),
                (private_entry.clone(), private_entry.clone()),
            ),
        );
        let public_key = RnsBggPublicKey::new(
            public_input,
            (
                (public_entry.clone(), public_entry.clone()),
                (public_entry.clone(), public_entry.clone()),
            ),
        );
        let private =
            RnsBggEncoding::new(private_key, (private_entry.clone(), private_entry.clone()), None);
        let public_payload = high_ring.input("mul-public-level-zero-payload", (1, 1));
        let public = RnsBggEncoding::new(
            public_key,
            (public_entry.clone(), public_entry),
            Some(public_payload),
        );
        let result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &public,
            &private,
        )
        .unwrap();
        let expected = low_ring.matrix_type((1, private_columns));
        assert_eq!(result.row.0.matrix_type(), &expected);
        assert_eq!(result.row.1.matrix_type(), &expected);
        for entry in [
            &result.public_key.matrix.0.0,
            &result.public_key.matrix.0.1,
            &result.public_key.matrix.1.0,
            &result.public_key.matrix.1.1,
        ] {
            assert_eq!(entry.matrix_type(), &expected);
        }
        assert!(result.public_payload.is_none());
    }

    #[test]
    fn mul_public_encoding_rows_match_independent_residual_oracle() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_lhs_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_rhs_wire = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public_rhs_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_output = builder
            .mul_public(Operand::Scalar(public_lhs_wire), Operand::Scalar(private_rhs_wire))
            .unwrap();
        let public_output = builder
            .mul_public(Operand::Scalar(public_lhs_wire), Operand::Scalar(public_rhs_wire))
            .unwrap();
        let program = Arc::new(
            builder
                .finish(vec![Operand::Scalar(private_output), Operand::Scalar(public_output)])
                .unwrap(),
        );
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let public_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let private_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PrivatePayload).unwrap();
        let entry = |columns, index| {
            ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(index) })
        };
        let lhs_entry = entry(public_columns, 0);
        let private_entry = entry(private_columns, 0);
        let private_entry_alt = private_entry.clone() * IntExpr::constant(2);
        let public_rhs_entry = entry(public_columns, 1);
        let lhs_key = ((lhs_entry.clone(), lhs_entry.clone()), (lhs_entry.clone(), lhs_entry));
        let private_rhs_key = (
            (private_entry.clone(), private_entry_alt.clone()),
            (private_entry_alt.clone(), private_entry),
        );
        let public_rhs_key = (
            (public_rhs_entry.clone(), public_rhs_entry.clone()),
            (public_rhs_entry.clone(), public_rhs_entry),
        );
        let mask = (ring.identity(1), ring.zero((1, 1)));
        let payload = (ring.identity(1), ring.zero((1, 1)));
        let lhs_plaintext = ring.polynomial([IntExpr::constant(2)]);
        let private_plaintext = ring.polynomial([IntExpr::constant(3)]);
        let public_rhs_plaintext = ring.polynomial([IntExpr::constant(5)]);
        let public_gadget = parameters.public_gadget(0).unwrap();
        let state_gadget = parameters.state_gadget(0).unwrap();
        let delta = parameters.scalar(0, parameters.message_scale(0).unwrap()).unwrap();
        let mask_times_key = |key: &RnsBggMatrix| {
            (
                mask.0.clone() * key.0.0.clone() + mask.1.clone() * key.1.0.clone(),
                mask.0.clone() * key.0.1.clone() + mask.1.clone() * key.1.1.clone(),
            )
        };
        let public_signal =
            (mask.0.clone() * public_gadget.clone(), mask.1.clone() * public_gadget.clone());
        let private_signal = (
            payload.0.clone() * state_gadget.clone() * delta.clone() * private_plaintext.clone(),
            payload.1.clone() * state_gadget * delta * private_plaintext,
        );
        let lhs_row = {
            let carrier = mask_times_key(&lhs_key);
            (
                carrier.0 - public_signal.0.clone() * lhs_plaintext.clone(),
                carrier.1 - public_signal.1.clone() * lhs_plaintext.clone(),
            )
        };
        let private_rhs_row = {
            let carrier = mask_times_key(&private_rhs_key);
            (carrier.0 - private_signal.0.clone(), carrier.1 - private_signal.1.clone())
        };
        let public_rhs_row = {
            let carrier = mask_times_key(&public_rhs_key);
            (
                carrier.0 - public_signal.0.clone() * public_rhs_plaintext.clone(),
                carrier.1 - public_signal.1.clone() * public_rhs_plaintext.clone(),
            )
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([127; 32]),
                execution_nonce: [129; 32],
            },
            encryption: ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([127; 32]),
                execution_nonce: [129; 32],
            },
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: Some(LoweringSite {
                output: private_output,
                indices: Vec::new(),
                ranges: Vec::new(),
            }),
        };
        let private_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(public_lhs_wire, lhs_key.clone()),
                lhs_row.clone(),
                Some(lhs_plaintext.clone()),
            ),
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(private_rhs_wire, private_rhs_key.clone()),
                private_rhs_row.clone(),
                None,
            ),
        )
        .unwrap();
        lowerer.current_site =
            Some(LoweringSite { output: public_output, indices: Vec::new(), ranges: Vec::new() });
        let public_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(public_lhs_wire, lhs_key.clone()),
                lhs_row,
                Some(lhs_plaintext.clone()),
            ),
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(public_rhs_wire, public_rhs_key.clone()),
                public_rhs_row,
                Some(public_rhs_plaintext.clone()),
            ),
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
        let expected_private_key =
            direct_action(lhs_key.clone(), direct_decompose(private_rhs_key.clone()));
        let expected_public_key =
            direct_action(lhs_key.clone(), direct_decompose(public_rhs_key.clone()));
        let direct_row_action = |row: RnsBggRow, key: RnsBggMatrix| {
            let decomposition = direct_decompose(key);
            (
                row.0.clone().mul_small_rhs(decomposition.0.0) +
                    row.1.clone().mul_small_rhs(decomposition.1.0),
                row.0.mul_small_rhs(decomposition.0.1) + row.1.mul_small_rhs(decomposition.1.1),
            )
        };
        let mask_of = |key: &RnsBggMatrix| mask_times_key(key);
        let public_action_for_private =
            direct_row_action(public_signal.clone(), private_rhs_key.clone());
        let private_rhs_carrier = mask_of(&private_rhs_key);
        let private_expected_signal = (
            public_action_for_private.0 * lhs_plaintext.clone() +
                private_rhs_carrier.0 * -lhs_plaintext.clone() +
                private_signal.0.clone() * lhs_plaintext.clone(),
            public_action_for_private.1 * lhs_plaintext.clone() +
                private_rhs_carrier.1 * -lhs_plaintext.clone() +
                private_signal.1.clone() * lhs_plaintext.clone(),
        );
        let private_residual = (
            private_result.row.0.clone() -
                mask.0.clone() * expected_private_key.0.0 -
                mask.1.clone() * expected_private_key.1.0 +
                private_expected_signal.0.clone(),
            private_result.row.1.clone() -
                mask.0.clone() * expected_private_key.0.1 -
                mask.1.clone() * expected_private_key.1.1 +
                private_expected_signal.1,
        );
        let public_action = direct_row_action(public_signal.clone(), public_rhs_key.clone());
        let public_rhs_carrier = mask_of(&public_rhs_key);
        let public_expected_signal = (
            public_action.0 * lhs_plaintext.clone() - public_rhs_carrier.0 * lhs_plaintext.clone() +
                public_signal.0.clone() * lhs_plaintext * public_rhs_plaintext.clone(),
            public_action.1 * ring.polynomial([IntExpr::constant(2)]) -
                public_rhs_carrier.1 * ring.polynomial([IntExpr::constant(2)]) +
                public_signal.1 * ring.polynomial([IntExpr::constant(2)]) * public_rhs_plaintext,
        );
        let public_residual = (
            public_result.row.0.clone() -
                mask.0.clone() * expected_public_key.0.0 -
                mask.1.clone() * expected_public_key.1.0 +
                public_expected_signal.0,
            public_result.row.1.clone() -
                mask.0.clone() * expected_public_key.0.1 -
                mask.1.clone() * expected_public_key.1.1 +
                public_expected_signal.1,
        );
        let wrong_rhs_key = (
            (private_entry_alt.clone(), private_entry_alt.clone()),
            (private_entry_alt.clone(), private_entry_alt),
        );
        let wrong_key = direct_action(lhs_key, direct_decompose(wrong_rhs_key));
        let wrong_residual = private_result.row.0 -
            mask.0.clone() * wrong_key.0.0 -
            mask.1.clone() * wrong_key.1.0 +
            private_expected_signal.0.clone();
        let graph = DslContext::new("mul-public-encoding-residual-oracle")
            .output("private-residual-0", private_residual.0)
            .unwrap()
            .output("private-residual-1", private_residual.1)
            .unwrap()
            .output("public-residual-0", public_residual.0)
            .unwrap()
            .output("public-residual-1", public_residual.1)
            .unwrap()
            .output("wrong-residual", wrong_residual)
            .unwrap()
            .build()
            .unwrap();
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend([level]));
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        for name in
            ["private-residual-0", "private-residual-1", "public-residual-0", "public-residual-1"]
        {
            let matrix = crate::test_support::full(&result.outputs[name]);
            assert!(
                matrix.entry(0, 0).coeffs_biguints().iter().all(|value| value.is_zero()),
                "{name} residual is nonzero: {:?}",
                matrix.entry(0, 0).coeffs_biguints()
            );
        }
        let wrong = crate::test_support::full(&result.outputs["wrong-residual"]);
        assert!(wrong.entry(0, 0).coeffs_biguints().iter().any(|value| !value.is_zero()));
    }

    #[test]
    fn coefficient_finish_outputs_finalizes_the_paired_key_and_row_once() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = LutTable::unary(
            2,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Coefficient,
        )
        .unwrap();
        let lut = builder.register_lut(table).unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([47; 32]),
            execution_nonce: [53; 32],
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            decoder_smudge: None,
            artifact_prefix: None,
            context: [0; 32],
            rhs_companions: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        assert!(lowerer.program.wire_is_coefficient_output(output).unwrap());
        let source_ring = parameters.ring(0).unwrap();
        let columns = parameters.component_columns(0).unwrap();
        let value = source_ring.input("coefficient-finish", (1, columns));
        let key = RnsBggPublicKey::new(
            output,
            ((value.clone(), value.clone()), (value.clone(), value.clone())),
        );
        let encoding = RnsBggEncoding::new(key, (value.clone(), value), None);
        let finalized = lowerer.finish_outputs(vec![encoding]).unwrap().pop().unwrap();
        use crate::test_support::modulus_of;
        let target_modulus = BigInt::from(parameters.level(1).unwrap().modulus().as_ref().clone());
        assert_eq!(modulus_of(finalized.row.0.matrix_type()), target_modulus);
        assert_eq!(modulus_of(finalized.row.1.matrix_type()), target_modulus);
        for entry in [
            &finalized.public_key.matrix.0.0,
            &finalized.public_key.matrix.0.1,
            &finalized.public_key.matrix.1.0,
            &finalized.public_key.matrix.1.1,
        ] {
            assert_eq!(modulus_of(entry.matrix_type()), target_modulus);
        }
    }

    #[test]
    fn public_lut_input_is_rejected_before_payload_propagation() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let table = LutTable::unary(
            2,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Coefficient,
        )
        .unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&program).is_err());
    }
}
