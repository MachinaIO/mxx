//! Public-key preparation and preprocessing for RNS-BGG.
#![warn(missing_docs)]

use super::*;
use crate::{
    api::{
        ApiError, PlaintextInputs, PreprocessBuild, RefreshApplicationError, StageGraph,
        apply_refresh,
    },
    noise_refresh::{
        prf::PrfLabel,
        refresh::{NoiseRefresh, RefreshError, RefreshPrfLabel},
    },
    program::{
        AttributeKnowledge, Environment, ExpLayout, ExponentLUTProgramLowering, ExponentLutProgram,
        GraphFamilyExecution, GraphParallelLowering, InputBinding, LoweringSite, LutId, LutTable,
        ProgramLoweringContext, RhsId, RhsRef, WireId, lower_program,
    },
};
use mxx_dsl::{Bytes, DslContext, DslError, Family, GraphValue, GraphValueSchema, Int, Mat};
use mxx_ir_core::{
    IntExpr, ValueHandle,
    artifact::{ArtifactCatalog, ProductionId},
    node::ConstantMatrix,
    types::WireType,
};
use mxx_primitives::poly::Poly;
use mxx_runtime::{Backend, RuntimeValue};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

/// Public-key carrier paired with its validated Program wire.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggPublicKey<M = RnsBggMatrix> {
    pub(crate) wire: WireId,
    pub(crate) matrix: M,
}

impl<M> RnsBggPublicKey<M> {
    /// Construct a carrier for `wire` with a matrix matching that wire's
    /// validated public-key schema.
    ///
    /// Callers are responsible for preserving the Program wire identity and
    /// matrix schema; stage builders validate both when the carrier is used.
    pub fn new(wire: WireId, matrix: M) -> Self {
        Self { wire, matrix }
    }
    /// Program wire represented by this key.
    pub fn wire(&self) -> WireId {
        self.wire
    }
    /// Matrix carried by this key.
    pub fn matrix(&self) -> &M {
        &self.matrix
    }
}

impl<M: GraphValue> GraphValue for RnsBggPublicKey<M> {
    type Schema = RnsBggPublicKey<M::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.matrix.flatten()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggPublicKey::new(self.wire, self.matrix.schema())
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        Ok(Self::new(schema.wire, M::from_values(&schema.matrix, values)?))
    }
}
impl<M: GraphValueSchema> GraphValueSchema for RnsBggPublicKey<M> {
    type Value = RnsBggPublicKey<M::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPublicKey::new(self.wire, self.matrix.placeholders_from(next))
    }
    fn wire_types(&self) -> Vec<WireType> {
        self.matrix.wire_types()
    }
}

fn graph(error: impl std::fmt::Display) -> RnsBggError {
    RnsBggError::Graph(error.to_string())
}

fn matrix_entries(matrix: &RnsBggMatrix) -> [Mat; 4] {
    [matrix.0.0.clone(), matrix.0.1.clone(), matrix.1.0.clone(), matrix.1.1.clone()]
}

fn public_lut_branch(
    public_data: RnsBggMatrix,
    switched_key_decomposition: RnsBggDecomposition,
) -> RnsBggMatrix {
    multiply_public_action(public_data, switched_key_decomposition)
}

fn matrix_from_hash(
    ring: &mxx_dsl::Ring,
    key: Bytes,
    tag: impl Into<mxx_dsl::HashTag>,
    columns: usize,
) -> RnsBggMatrix {
    let tag = tag.into();
    (
        (
            ring.hash_matrix(key.clone(), tag.clone(), (1, columns)),
            ring.hash_matrix(key.clone(), tag.clone(), (1, columns)),
        ),
        (
            ring.hash_matrix(key.clone(), tag.clone(), (1, columns)),
            ring.hash_matrix(key, tag, (1, columns)),
        ),
    )
}

fn input_hash_tag(wire: WireId, level: usize, member: Option<Int>) -> mxx_dsl::HashTag {
    let mut tag = mxx_dsl::HashTag::from(format!("rns-bgg/input/{wire:?}/{level}").into_bytes());
    if let Some(member) = member {
        tag.push("member");
        tag.push(member);
    }
    tag
}

fn decompose_matrix(
    parameters: &RnsBggParameters,
    level: usize,
    matrix: RnsBggMatrix,
) -> Result<RnsBggDecomposition, RnsBggError> {
    let p = parameters.level(level)?;
    let base = 1u64
        .checked_shl(p.base_bits())
        .ok_or(RnsBggError::Parameters("decomposition base overflow"))?;
    let decomposition_count = p.modulus_digits();
    Ok((
        (
            matrix.0.0.decompose(base, decomposition_count),
            matrix.0.1.decompose(base, decomposition_count),
        ),
        (
            matrix.1.0.decompose(base, decomposition_count),
            matrix.1.1.decompose(base, decomposition_count),
        ),
    ))
}

fn centered_extend_decomposition(
    value: RnsBggDecomposition,
    modulus: BigInt,
) -> RnsBggDecomposition {
    (
        (value.0.0.centered_extend(modulus.clone()), value.0.1.centered_extend(modulus.clone())),
        (value.1.0.centered_extend(modulus.clone()), value.1.1.centered_extend(modulus)),
    )
}

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
    let evaluation_modulus =
        BigInt::from(parameters.level(evaluation_level)?.modulus().as_ref().clone());
    Ok(centered_extend_decomposition(value, evaluation_modulus))
}

fn refresh_payload_action(
    parameters: &RnsBggParameters,
    decomposition_level: usize,
    payload: &RnsBggPayloadSecret,
    desired: RnsBggRow,
    hash_key: Bytes,
    tag: impl Into<mxx_dsl::HashTag>,
    target_columns: usize,
) -> Result<RnsBggMatrix, RnsBggError> {
    let p_modulus = BigInt::from(parameters.level(decomposition_level)?.modulus().as_ref().clone());
    let p_ring = parameters.ring(decomposition_level)?;
    let top_matrix = matrix_from_hash(&p_ring, hash_key, tag, target_columns);
    let r0 = checked_modulus_convert(payload.row.0.clone(), p_modulus.clone())?;
    let eta =
        crate::rns_bgg::encoding::fresh_error(parameters, decomposition_level, target_columns)?;
    let bottom = (
        desired.0 - r0.clone() * top_matrix.0.0.clone() + eta.0,
        desired.1 - r0 * top_matrix.0.1.clone() + eta.1,
    );
    Ok(((top_matrix.0.0, top_matrix.0.1), bottom))
}

fn lift_output_target_to_p(value: RnsBggRow, p_modulus: BigInt, h_p: Mat) -> RnsBggRow {
    let h_p = h_p.centered_extend(p_modulus.clone());
    super::map_row(value, |entry| h_p.clone() * entry.centered_extend(p_modulus.clone()))
}

fn lut_payload_target(scale: Mat, coefficient: Mat, payload: RnsBggRow, gadget: Mat) -> RnsBggRow {
    (
        scale.clone() * coefficient.clone() * payload.0 * gadget.clone(),
        scale * coefficient * payload.1 * gadget,
    )
}

fn sampled_key(
    parameters: &RnsBggParameters,
    level: usize,
    hash_key: Bytes,
    wire: WireId,
    knowledge: AttributeKnowledge,
    member: Option<Int>,
) -> Result<RnsBggPublicKey, RnsBggError> {
    let ring = parameters.ring(level)?;
    let columns = parameters.key_component_columns(level, knowledge)?;
    Ok(RnsBggPublicKey::new(
        wire,
        matrix_from_hash(&ring, hash_key, input_hash_tag(wire, level, member), columns),
    ))
}

/// Structural public-key lowerer. All prepared values are compact graph
/// handles; the maps do not own expanded matrices or concrete family members.
pub struct PublicKeyLowering<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
    pub(crate) program: Arc<ExponentLutProgram>,
    pub(crate) workload: Arc<RnsBggDerivedWorkload>,
    pub(crate) payload_secret: RnsBggPayloadSecret,
    pub(crate) hash_key: Bytes,
    pub(crate) rhs: BTreeMap<RhsId, InputBinding<Mat, Family<Mat>>>,
    /// Setup-fixed RHS ciphertexts.  These are compact matrices; their signed
    /// decompositions are constructed at the consuming level only.
    pub(crate) rhs_material: BTreeMap<RhsId, InputBinding<RnsBggMatrix, Family<RnsBggMatrix>>>,
    /// Compact LUT switching ciphertexts, keyed by the complete table/layout
    /// identity.  The family member is the concrete automorphism branch.
    pub(crate) lut_material:
        BTreeMap<(usize, usize, crate::program::LutId, ExpLayout), Family<RnsBggMatrix>>,
    /// Deterministic public `A_sigma` matrices, shared by level and
    /// automorphism independently of LUT, layout, wire, or scope.
    pub(crate) lut_public_data: BTreeMap<(usize, usize), RnsBggMatrix>,
    /// Refresh targets and compact actions, keyed by declaration commitment
    /// and CRT slot.  Keeping the commitment in the key prevents material
    /// from one refresh declaration being reused by another invocation.
    pub(crate) refresh_sums: BTreeMap<([u8; 32], usize), RnsBggMatrix>,
    pub(crate) refresh_stars: BTreeMap<([u8; 32], usize), RnsBggMatrix>,
    pub(crate) refresh_targets: BTreeMap<([u8; 32], usize), RnsBggMatrix>,
    pub(crate) refresh_label_targets: BTreeMap<([u8; 32], usize, usize), RnsBggMatrix>,
    pub(crate) current_site: Option<LoweringSite>,
}

impl<'a> PublicKeyLowering<'a> {
    /// Construct a public-key lowerer for one immutable Program/workload pair.
    pub(crate) fn new(
        parameters: &'a RnsBggParameters,
        program: Arc<ExponentLutProgram>,
        workload: Arc<RnsBggDerivedWorkload>,
        payload_secret: RnsBggPayloadSecret,
        hash_key: Bytes,
    ) -> Self {
        Self {
            parameters,
            program,
            workload,
            payload_secret,
            hash_key,
            rhs: BTreeMap::new(),
            rhs_material: BTreeMap::new(),
            lut_material: BTreeMap::new(),
            lut_public_data: BTreeMap::new(),
            refresh_sums: BTreeMap::new(),
            refresh_stars: BTreeMap::new(),
            refresh_targets: BTreeMap::new(),
            refresh_label_targets: BTreeMap::new(),
            current_site: None,
        }
    }
    fn site(&self) -> Result<&LoweringSite, RnsBggError> {
        self.current_site.as_ref().ok_or(RnsBggError::Binding("missing lowering site"))
    }
    fn output(&self) -> Result<WireId, RnsBggError> {
        Ok(self.site()?.output)
    }

    /// Sample an input key with shape selected from Program knowledge.
    pub(crate) fn sample_input_public_key(
        &self,
        wire: WireId,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        self.sample_input_public_key_member(wire, None)
    }

    fn sample_input_public_key_member(
        &self,
        wire: WireId,
        member: Option<Int>,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        let knowledge = self
            .program
            .wire_knowledge(wire)
            .map_err(|_| RnsBggError::Binding("wire knowledge"))?;
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
    fn prepare_rhs_material(&mut self, rhs: &RhsRef<Int>) -> Result<RnsBggMatrix, RnsBggError> {
        let level = self.workload.rhs_level(rhs.id)?;
        let next_level =
            level.checked_add(1).ok_or(RnsBggError::Parameters("RHS target level overflow"))?;
        let next_modulus =
            BigInt::from(self.parameters.level(next_level)?.modulus().as_ref().clone());
        let next_ring = self.parameters.ring(next_level)?;
        let target_columns = self.parameters.component_columns(next_level)?;
        // Bind the sampled RHS to the caller's plaintext through the same
        // payload-constrained ciphertext relation used by private input
        // encryption. The plaintext is a graph handle; it is never copied
        // into the artifact identity or the online lowerer.
        let plaintext = match self.rhs.get(&rhs.id) {
            Some(InputBinding::Scalar(value)) => value.clone(),
            Some(InputBinding::Family(values)) => {
                values.at(rhs.index.clone().ok_or(RnsBggError::Binding("family RHS index"))?)
            }
            None => return Err(RnsBggError::Binding("RHS plaintext was not bound")),
        };
        // Plaintext handles are canonicalized in the caller's natural input
        // ring (currently level zero).  Do not require them to already carry
        // the RHS consumer's logical level: the setup-fixed ciphertext is
        // intentionally sampled at q_(level + 1), and this conversion is the
        // boundary where the plaintext is represented in that ring.
        let nu = checked_modulus_convert(plaintext, next_modulus.clone())?;
        let payload = checked_modulus_convert_row(self.payload_secret.row.clone(), next_modulus)?;
        let gadget = self.parameters.state_gadget(next_level)?;
        let scale =
            self.parameters.scalar(next_level, self.parameters.scale(next_level)?.clone())?;
        let target = (payload.0.clone() * gadget.clone(), payload.1.clone() * gadget);
        let desired = (target.0 * scale.clone() * nu.clone(), target.1 * scale * nu);
        let tag = mxx_dsl::HashTag::from(
            format!("rns-bgg/rhs/{:?}/{level}/{next_level}", rhs.id).into_bytes(),
        );
        let top_matrix = matrix_from_hash(&next_ring, self.hash_key.clone(), tag, target_columns);
        let error =
            crate::rns_bgg::encoding::fresh_error(self.parameters, next_level, target_columns)?;
        let bottom = (
            desired.0 - payload.0.clone() * top_matrix.0.0.clone() + error.0,
            desired.1 - payload.0 * top_matrix.0.1.clone() + error.1,
        );
        Ok(((top_matrix.0.0, top_matrix.0.1), bottom))
    }
    /// Eagerly materialize the deterministic public `A_sigma` superset for
    /// the complete input/LUT projection.  No Program body or workload is
    /// consulted here, so encryption can derive exactly the same key set.
    pub(crate) fn prepare_lut_public_data_projection(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
    ) -> Result<(), RnsBggError> {
        for (level, sigma) in canonical_lut_helper_keys(self.parameters, inputs)? {
            if self.lut_public_data.contains_key(&(level, sigma)) {
                continue;
            }
            let ring = self.parameters.ring(level)?;
            let columns = self.parameters.component_columns(level)?;
            let data = matrix_from_hash(
                &ring,
                self.hash_key.clone(),
                mxx_dsl::HashTag::from(format!("rns-bgg/lut-a/{level}/sigma/{sigma}").into_bytes()),
                columns,
            );
            self.lut_public_data.insert((level, sigma), data);
        }
        Ok(())
    }

    /// Resolve `A_sigma` for one LUT layout from the eagerly prepared store.
    /// A missing key is an invalid preprocessing projection, never a reason
    /// for body lowering to synthesize a new public matrix.
    fn lut_public_data_for(
        &self,
        level: usize,
        table: &LutTable,
    ) -> Result<Family<RnsBggMatrix>, RnsBggError> {
        let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let layout = table.input_layout();
        layout.validate(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let sigmas = layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        Family::pack(
            sigmas
                .iter()
                .map(|sigma| {
                    self.lut_public_data
                        .get(&(level, *sigma))
                        .cloned()
                        .ok_or(RnsBggError::Binding("missing eagerly prepared LUT A matrix"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(graph)
    }

    /// Materialize setup-fixed compact `B_sigma` switching ciphertexts for a
    /// complete LUT/layout identity.  Gadget decomposition is deferred until
    /// the branch is consumed.
    pub(crate) fn prepare_lut_switching_ciphertexts(
        &mut self,
        input: &RnsBggPublicKey,
        lut_id: LutId,
        table: &LutTable,
    ) -> Result<Family<RnsBggMatrix>, RnsBggError> {
        let level = self.workload.level(input.wire)?;
        let next_level =
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let target_modulus =
            BigInt::from(self.parameters.level(next_level)?.modulus().as_ref().clone());
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
        let switching_ciphertexts = mxx_dsl::parallel(count, |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let coefficient = super::EncodingLowering::lut_coefficient(
                self.parameters,
                next_level,
                layout,
                table.output_representation(),
                sigma.clone(),
                table.values(),
            )
            .map_err(|_| DslError::Schema)?;
            // B_sigma is sampled in the next ring and constrained by the
            // payload secret. Its signed switching decomposition has
            // each compact digit centered-extended to the source ring; the
            // full B matrix is never extended as a dense matrix.
            let payload = checked_modulus_convert_row(
                self.payload_secret.row.clone(),
                target_modulus.clone(),
            )
            .map_err(|_| DslError::Schema)?;
            let gadget = self.parameters.state_gadget(next_level).map_err(|_| DslError::Schema)?;
            let scale = self
                .parameters
                .scalar(
                    next_level,
                    self.parameters.scale(next_level).map_err(|_| DslError::Schema)?.clone(),
                )
                .map_err(|_| DslError::Schema)?;
            let target = lut_payload_target(scale, coefficient, payload.clone(), gadget);
            let ring_next = self.parameters.ring(next_level).map_err(|_| DslError::Schema)?;
            let columns =
                self.parameters.component_columns(next_level).map_err(|_| DslError::Schema)?;
            let mut tag = mxx_dsl::HashTag::from(
                format!("rns-bgg/lut-b/{:?}/{level}/{next_level}", lut_id).into_bytes(),
            );
            tag.push("sigma");
            tag.push(sigma.clone());
            tag.push("branch");
            tag.push(index.expression()?);
            let top_matrix = matrix_from_hash(&ring_next, self.hash_key.clone(), tag, columns);
            let top = (top_matrix.0.0, top_matrix.0.1);
            let prefix = payload.0.ring_automorphism(sigma.clone());
            let error = crate::rns_bgg::encoding::fresh_error(self.parameters, next_level, columns)
                .map_err(|_| DslError::Schema)?;
            let b_sigma = (
                top.clone(),
                (
                    -prefix.clone() * top.0 + target.0 + error.0,
                    -prefix * top.1 + target.1 + error.1,
                ),
            );
            Ok(b_sigma)
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
                    let q = BigInt::from(self.parameters.level(target)?.modulus().as_ref().clone());
                    Ok(RnsBggPublicKey::new(output.wire, super::mod_down_public(output.matrix, q)))
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
        let compact = match self.rhs_material.get(&rhs.id) {
            Some(InputBinding::Scalar(value)) => value.clone(),
            Some(InputBinding::Family(values)) => {
                values.at(rhs.index.clone().ok_or(RnsBggError::Binding("family RHS index"))?)
            }
            None => return Err(RnsBggError::Binding("RHS material was not prepared")),
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
        let target_modulus =
            BigInt::from(self.parameters.level(decomposition_level)?.modulus().as_ref().clone());
        let compact = checked_modulus_convert_matrix(compact, target_modulus)?;
        let compact = decompose_matrix(self.parameters, decomposition_level, compact)?;
        let evaluation_modulus =
            BigInt::from(self.parameters.level(evaluation_level)?.modulus().as_ref().clone());
        // Only the bounded decomposition digits cross back into the
        // evaluation ring; the compact RHS artifact itself remains at its
        // native next-level schema.
        let action = centered_extend_decomposition(compact, evaluation_modulus);
        let target_level = self.workload.level(self.output()?)?;
        let target_modulus =
            BigInt::from(self.parameters.level(target_level)?.modulus().as_ref().clone());
        Ok(RnsBggPublicKey::new(
            self.output()?,
            super::mod_down_public(
                super::multiply_public_action(input.matrix.clone(), action),
                target_modulus,
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
        let data = self.lut_public_data_for(source_level, table)?;
        let switching_ciphertexts = self.prepare_lut_switching_ciphertexts(input, lut_id, table)?;
        let next_level =
            source_level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let source_modulus =
            BigInt::from(self.parameters.level(source_level)?.modulus().as_ref().clone());
        let target_modulus =
            BigInt::from(self.parameters.level(next_level)?.modulus().as_ref().clone());
        let layout = table.input_layout();
        let sigmas = layout
            .automorphism_indices(self.parameters.level(source_level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let branch_decompositions =
            mxx_dsl::parallel(IntExpr::constant(table.values().len()), |index| {
                let sigma = IntExpr::Select {
                    selector: Box::new(index.expression()?),
                    branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
                };
                let switching_decomposition = centered_extend_decomposition(
                    decompose_matrix(
                        self.parameters,
                        next_level,
                        switching_ciphertexts.at(index.clone()),
                    )
                    .map_err(|_| DslError::Schema)?,
                    source_modulus.clone(),
                );
                let routed = super::map_matrix(input.matrix.clone(), |entry| {
                    entry.ring_automorphism(sigma.clone())
                });
                let switched_public_key = super::mod_down_public(
                    super::multiply_public_action(routed, switching_decomposition),
                    target_modulus.clone(),
                );
                let switched_key_decomposition = centered_extend_decomposition(
                    decompose_matrix(self.parameters, next_level, switched_public_key)
                        .map_err(|_| DslError::Schema)?,
                    source_modulus.clone(),
                );
                Ok(public_lut_branch(data.at(index), switched_key_decomposition))
            })
            .map_err(graph)?;
        let output = self.output()?;
        let result =
            crate::utils::balanced_sum_family(branch_decompositions, super::add_public_matrices)
                .map_err(graph)?;
        if !self
            .program
            .wire_is_coefficient_output(output)
            .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
        {
            let target = self.workload.level(output)?;
            let modulus = BigInt::from(self.parameters.level(target)?.modulus().as_ref().clone());
            return Ok(RnsBggPublicKey::new(output, super::mod_down_public(result, modulus)));
        }
        Ok(RnsBggPublicKey::new(output, result))
    }
}

impl PublicKeyLowering<'_> {
    fn validate_refresh_declaration(
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
impl ProgramLoweringContext for PublicKeyLowering<'_> {
    fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl GraphParallelLowering for PublicKeyLowering<'_> {
    type SideOutput = ();

    fn begin_graph_parallel(&mut self) {}

    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
        Ok(())
    }

    fn commit_graph_parallel(&mut self) -> Result<(), Self::Error> {
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
    channel_modulus: BigInt,
    target_modulus: BigInt,
) -> RnsBggMatrix {
    map_matrix(matrix, |entry| {
        entry.block_mod_switch(channel_modulus.clone(), 1).centered_extend(target_modulus.clone())
    })
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
    channel_modulus: BigInt,
    target_modulus: BigInt,
) -> Result<RnsBggMatrix, RnsBggError> {
    let reduced = checked_modulus_convert_matrix(matrix, channel_modulus)?;
    Ok(map_matrix(reduced, |entry| entry.centered_extend(target_modulus.clone())))
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

fn expected_refresh_slots(
    label: &RefreshPrfLabel,
    slot_count: usize,
) -> Result<Vec<usize>, RnsBggError> {
    match label {
        RefreshPrfLabel::Mask { slot, .. } => {
            if *slot >= slot_count {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
            Ok(vec![*slot])
        }
        RefreshPrfLabel::FreshError { .. } => {
            if slot_count == 0 {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
            Ok((0..slot_count).collect())
        }
    }
}

fn refresh_crt_weight(
    slots: &[crate::noise_refresh::refresh::RefreshSlot],
    slot: usize,
) -> Result<BigInt, RnsBggError> {
    let total = slots.iter().try_fold(BigUint::one(), |total, slot| {
        let modulus = slot
            .q_t
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        Ok::<_, RnsBggError>(total * modulus)
    })?;
    let qt = slots
        .get(slot)
        .and_then(|entry| entry.q_t.to_biguint())
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    let base = &total / &qt;
    let inverse = mxx_primitives::utils::mod_inverse_biguints(&(&base % &qt), &qt)
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    Ok(BigInt::from(base * inverse))
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
            let fresh = match label {
                RefreshPrfLabel::Mask { slot, digit, .. } => {
                    if slot >= declaration.digits().mask_counts.len() ||
                        digit >= declaration.digits().mask_counts[slot]
                    {
                        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
                    }
                    false
                }
                RefreshPrfLabel::FreshError { digit, .. } => {
                    if digit >= declaration.digits().fresh_count {
                        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
                    }
                    true
                }
            };
            let expected_slots = expected_refresh_slots(&label, slot_count)?;
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
                let expected_modulus = IntExpr::constant(BigInt::from(modulus));
                if ain.matrix.0.0.matrix_type().modulus != expected_modulus ||
                    value.matrix.0.0.matrix_type().modulus != expected_modulus
                {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                let h = fresh.then(|| &slot_metadata.p / &slot_metadata.q_t);
                let evaluation_level = anchor_levels[slot];
                let decomposition_level = decomposition_levels[slot];
                let target_columns =
                    self.parameters.component_columns(declaration.target_level())?;
                let target_q = refresh_label_target_with_columns(
                    self.parameters,
                    evaluation_level,
                    target_columns,
                    declaration.digits().base,
                    label,
                    h.clone(),
                )?;
                let p_modulus = BigInt::from(
                    self.parameters.level(decomposition_level)?.modulus().as_ref().clone(),
                );
                // Encoding carries the private payload with a minus sign.
                // The preprocessing action therefore stores -W_label so
                // that applying it to the PRF value contributes +z*W_label.
                let target_p = checked_modulus_convert_matrix(target_q, p_modulus.clone())
                    .map(|target| map_matrix(target, |entry| -entry))?;
                let label_action = refresh_payload_action(
                    self.parameters,
                    decomposition_level,
                    &self.payload_secret,
                    (target_p.0.0, target_p.0.1),
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
            let a_sum =
                sums[slot].take().ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let h = &declaration.slots()[slot].p / &declaration.slots()[slot].q_t;
            let slot_modulus = declaration.slots()[slot]
                .q
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let evaluation_level = anchor_levels[slot];
            let decomposition_level = decomposition_levels[slot];
            let target_level = declaration.target_level();
            let target_columns = self.parameters.component_columns(target_level)?;
            let p_modulus = BigInt::from(
                self.parameters.level(decomposition_level)?.modulus().as_ref().clone(),
            );
            let p_ring = self.parameters.ring(decomposition_level)?;
            let h_p = p_ring.polynomial([IntExpr::constant(h.clone())]);
            let output_modulus =
                BigInt::from(self.parameters.level(target_level)?.modulus().as_ref().clone());
            let output_gadget = self.parameters.state_gadget(target_level)?;
            let output_scale = self
                .parameters
                .scalar(target_level, self.parameters.scale(target_level)?.clone())?;
            let output_payload =
                checked_modulus_convert_row(self.payload_secret.row.clone(), output_modulus)?;
            let desired_output = (
                output_scale.clone() * output_payload.0 * output_gadget.clone(),
                output_scale * output_payload.1 * output_gadget,
            );
            let desired_data = lift_output_target_to_p(desired_output, p_modulus.clone(), h_p);
            let action = refresh_payload_action(
                self.parameters,
                decomposition_level,
                &self.payload_secret,
                desired_data,
                self.hash_key.clone(),
                mxx_dsl::HashTag::from(
                    format!("rns-bgg/refresh-data/{:?}/{slot}", declaration.commitment())
                        .into_bytes(),
                ),
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
            if &slot_modulus % &target_modulus != BigUint::zero() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            // The refresh carrier is the per-slot A_star anchor.  Reduce it
            // modulo q_t without scaling, then centered-extend into Q* before
            // CRT recomposition.  Decoder targets remain A_sum-H*A_star and
            // are assembled by the encoding stage from the raw artifacts.
            let aout_p = checked_modulus_convert_matrix(aout.matrix.clone(), p_modulus.clone())?;
            let target_at_output = reduce_lift_public_anchor(
                aout_p.clone(),
                BigInt::from(
                    declaration.slots()[slot]
                        .q_t
                        .to_biguint()
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?,
                ),
                BigInt::from(target_modulus.clone()),
            )?;
            if values.first().is_some_and(|first| !matrix_schema_matches(first, &target_at_output))
            {
                return Err(RnsBggError::Binding("refresh CRT channel schemas differ"));
            }
            let a_sum = add_public_matrices(a_sum, data_public);
            let a_sum = mod_down_public(a_sum, p_modulus.clone());
            self.refresh_sums.insert((*declaration.commitment(), slot), a_sum);
            self.refresh_stars.insert((*declaration.commitment(), slot), aout_p);
            self.refresh_targets.insert((*declaration.commitment(), slot), action);
            values.push(target_at_output);
            coefficients.push(self.parameters.scalar(
                declaration.target_level(),
                refresh_crt_weight(declaration.slots(), slot)?,
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

/// Preprocessing stage adapter using the common stage lifecycle.
pub struct RnsPreprocessBuild<'a, B: Backend> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    hash_key_commitment: [u8; 32],
    evaluator: Option<PublicKeyLowering<'a>>,
    families: GraphFamilyExecution,
    environment:
        Environment<RnsBggPublicKey, RhsRef<Int>, Family<RnsBggPublicKey>, Family<RhsRef<Int>>>,
    runtime: BTreeMap<String, RuntimeValue<B>>,
    refresh_declaration: Option<crate::noise_refresh::refresh::RefreshDeclaration>,
    pending_refresh: Option<(usize, RnsBggPublicKey)>,
    refresh_prf_binding: Option<([u8; 32], usize, ExponentLutProgramId, [u8; 32])>,
    refresh_prf_children: BTreeMap<usize, PublicKeyLowering<'a>>,
}
impl<'a, B: Backend> RnsPreprocessBuild<'a, B> {
    /// Create an unbound preprocessing adapter.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            evaluator: None,
            families: GraphFamilyExecution,
            environment: Environment::new(),
            runtime: BTreeMap::new(),
            refresh_declaration: None,
            pending_refresh: None,
            refresh_prf_binding: None,
            refresh_prf_children: BTreeMap::new(),
        }
    }

    /// Bind RHS plaintexts for a prepared PRF child without exposing the
    /// generic stage trait at application call sites.
    pub fn bind_rhs_inputs<P: Poly>(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        payload: &RnsBggPayloadSecret,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        <Self as PreprocessBuild<B, P>>::bind_inputs(
            self, program, parameters, rhs, payload, hash_key,
        )
    }

    /// Bind the common refresh declaration before the preprocessing graph is
    /// built.  The declaration is copied as immutable metadata only; sampled
    /// matrices remain graph/artifact values owned by this stage.
    pub fn bind_refresh_declaration(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    ) -> Result<(), ApiError> {
        let evaluator =
            self.evaluator.as_ref().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        evaluator.validate_refresh_declaration(declaration).map_err(ApiError::scheme)?;
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }

    /// Bind this builder to one channel of the declaration's prepared PRF.
    ///
    /// A channel builder is intentionally separate from the source-stage
    /// declaration binding: it owns the prepared PRF Program and is lowered
    /// only when [`Self::apply_refresh`] dispatches the common refresh
    /// operation.  This check therefore validates the PRF output schema and
    /// coefficient lineage, without accepting a source carrier as a
    /// substitute for the prepared evaluator.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
        slot: usize,
    ) -> Result<(), ApiError> {
        self.parameters
            .validate_refresh_prf_schedule(declaration, slot)
            .map_err(ApiError::scheme)?;
        if slot >= declaration.slots().len() {
            return Err(ApiError::input("refresh PRF slot is outside declaration"));
        }
        let source_q = declaration
            .slots()
            .get(slot)
            .and_then(|slot| slot.q.to_biguint())
            .ok_or_else(|| ApiError::input("refresh source modulus is invalid"))?;
        if self
            .parameters
            .level(0)
            .map_err(|error| ApiError::input(error.to_string()))?
            .modulus()
            .as_ref() !=
            &source_q
        {
            return Err(ApiError::input("refresh PRF child is not configured at source q"));
        }
        let evaluator =
            self.evaluator.as_ref().ok_or_else(|| ApiError::input("bind_inputs first"))?;
        let expected_program = declaration.prepared().program().id();
        if evaluator.program.id() != expected_program {
            return Err(ApiError::input("refresh PRF Program does not match declaration"));
        }
        let outputs = declaration.prepared().declaration().ordered_outputs().collect::<Vec<_>>();
        if outputs.is_empty() {
            return Err(ApiError::input("refresh declaration has no prepared PRF outputs"));
        }
        for (_, wire) in outputs {
            let coefficient = evaluator
                .program
                .wire_is_coefficient_output(wire)
                .map_err(|error| ApiError::input(error.to_string()))?;
            if !coefficient {
                return Err(ApiError::input("refresh PRF output is not a coefficient output"));
            }
            if !evaluator.program.outputs().iter().any(|operand| {
                matches!(operand, crate::program::Operand::Scalar(output) if *output == wire)
            }) {
                return Err(ApiError::input("refresh PRF output is not a Program output"));
            }
        }
        if self.refresh_prf_binding.is_some() {
            return Err(ApiError::input("refresh PRF is already bound"));
        }
        self.refresh_prf_binding = Some((
            *declaration.commitment(),
            slot,
            expected_program,
            self.parameters.parameter_identity(),
        ));
        Ok(())
    }

    fn lower_refresh_prf(
        &mut self,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
    ) -> Result<BTreeMap<PrfLabel, RnsBggPublicKey>, RnsBggError> {
        let (commitment, slot, program, parameter_identity) =
            self.refresh_prf_binding.ok_or(RnsBggError::Binding("refresh PRF is not bound"))?;
        if parameter_identity != self.parameters.parameter_identity() {
            return Err(RnsBggError::Binding("refresh PRF parameter identity changed"));
        }
        if commitment != *declaration.commitment() ||
            slot >= declaration.slots().len() ||
            program != declaration.prepared().program().id()
        {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("refresh PRF builder was not bound"))?;
        let program = evaluator.program.clone();
        let outputs = lower_program(&program, evaluator, &mut self.families, &mut self.environment)
            .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let expected_wires = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(_, wire)| wire)
            .collect::<Vec<_>>();
        if outputs.len() != expected_wires.len() {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        declaration
            .prf_labels()
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, label)| {
                let expected_wire = expected_wires[index];
                let value = outputs[index].clone();
                validate_refresh_prf_wire(value.wire, expected_wire)?;
                Ok((label, value))
            })
            .collect()
    }

    /// Apply one declared channel-aware refresh through the common lifecycle.
    /// The refreshed carrier replaces the selected preprocess output exactly
    /// once before graph construction.
    pub fn apply_refresh(
        &mut self,
        output_index: usize,
        declaration: &crate::noise_refresh::refresh::RefreshDeclaration,
        inputs: BTreeMap<usize, RnsBggPublicKey>,
        mut channels: BTreeMap<usize, RnsPreprocessBuild<'a, B>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() || output_index >= declaration.output_schema().len() {
            return Err(RnsBggError::Binding(
                "refresh output was already selected or is out of range",
            ));
        }
        if self
            .refresh_declaration
            .as_ref()
            .is_some_and(|bound| bound.commitment() != declaration.commitment())
        {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        if channels.len() != declaration.slots().len() ||
            channels.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        for (slot, child) in &channels {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if child.parameters.parameter_identity() != expected_child_parameters ||
                child.refresh_prf_binding.map(|(_, _, _, identity)| identity) !=
                    Some(expected_child_parameters)
            {
                return Err(RnsBggError::Binding("refresh PRF child parameter identity mismatch"));
            }
        }
        let (output, child_material) = {
            let evaluator = self
                .evaluator
                .as_mut()
                .ok_or(RnsBggError::Binding("preprocess builder was not bound"))?;
            let mut refresh = RnsBggRefresh::new(evaluator, inputs);
            let mut child_material = Vec::with_capacity(channels.len());
            let output =
                apply_refresh(declaration, &mut refresh, |callback_declaration, labels| {
                    let mut per_label =
                        BTreeMap::<PrfLabel, BTreeMap<usize, RnsBggPublicKey>>::new();
                    let label_positions = labels
                        .iter()
                        .enumerate()
                        .map(|(index, label)| (label.clone(), index))
                        .collect::<BTreeMap<_, _>>();
                    for slot in 0..callback_declaration.slots().len() {
                        let child = channels
                            .get_mut(&slot)
                            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                        let outputs = child.lower_refresh_prf(callback_declaration)?;
                        for (label, value) in outputs {
                            let Some(&label_index) = label_positions.get(&label) else {
                                per_label.entry(label).or_default().insert(slot, value);
                                continue;
                            };
                            let refresh_label = declaration
                                .labels()
                                .label(label_index)
                                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                            let expected_slots = expected_refresh_slots(
                                &refresh_label,
                                callback_declaration.slots().len(),
                            )?;
                            if !expected_slots.contains(&slot) {
                                continue;
                            }
                            if per_label.entry(label).or_default().insert(slot, value).is_some() {
                                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                            }
                        }
                    }
                    for (slot, mut child) in std::mem::take(&mut channels) {
                        let evaluator = child
                            .evaluator
                            .take()
                            .ok_or(RnsBggError::Binding("refresh PRF builder was not bound"))?;
                        child_material.push((slot, evaluator, std::mem::take(&mut child.runtime)));
                    }
                    Ok(per_label)
                })
                .map_err(|error| match error {
                    RefreshApplicationError::Refresh(error) => error,
                    RefreshApplicationError::OutputMismatch => {
                        RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
                    }
                })?;
            (output, child_material)
        };
        for (slot, child, runtime) in child_material {
            merge_runtime(&mut self.runtime, runtime)?;
            if self.refresh_prf_children.insert(slot, child).is_some() {
                return Err(RnsBggError::Binding("refresh PRF slot was already retained"));
            }
        }
        self.pending_refresh = Some((output_index, output));
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }
}

fn merge_runtime<B: Backend>(
    destination: &mut BTreeMap<String, RuntimeValue<B>>,
    source: BTreeMap<String, RuntimeValue<B>>,
) -> Result<(), RnsBggError> {
    for (name, value) in source {
        match destination.get(&name) {
            Some(RuntimeValue::Bytes(existing)) => match &value {
                RuntimeValue::Bytes(candidate) if existing == candidate => continue,
                RuntimeValue::Bytes(_) => {
                    return Err(RnsBggError::Binding("refresh child runtime input conflicts"));
                }
                _ => return Err(RnsBggError::Binding("refresh child runtime input schema")),
            },
            Some(_) => return Err(RnsBggError::Binding("refresh child runtime input schema")),
            None => {
                destination.insert(name, value);
            }
        }
    }
    Ok(())
}

impl<'a, B: Backend, P: Poly> PreprocessBuild<B, P> for RnsPreprocessBuild<'a, B> {
    type Evaluator = PublicKeyLowering<'a>;
    type Families = GraphFamilyExecution;
    type PayloadSecret = RnsBggPayloadSecret;
    type EncryptDescriptor = RnsBggPreprocessEncryptDescriptor;
    type OnlineDescriptor = RnsBggPreprocessOnlineDescriptor;
    type Encrypt = RnsEncryptBuild<'a, B>;
    type Online = RnsOnlineBuild<'a>;
    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                RnsBggPublicKey,
                RhsRef<Int>,
                Family<RnsBggPublicKey>,
                Family<RhsRef<Int>>,
            >,
        ),
        ApiError,
    > {
        Ok((
            self.evaluator.as_mut().ok_or_else(|| ApiError::input("bind_inputs first"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
    fn bind_inputs(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        rhs: &PlaintextInputs<RhsId, P>,
        payload: &Self::PayloadSecret,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if rhs.len() != program.rhs_inputs().len() ||
            rhs.keys().any(|id| !program.rhs_inputs().contains_key(id))
        {
            return Err(ApiError::input("RHS bindings do not exactly match Program inputs"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let bytes = ring.bytes_input("rns-bgg/hash-key", hash_key.len());
        let program = Arc::new(program.clone());
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(&program).map_err(ApiError::scheme)?);
        let mut evaluator = PublicKeyLowering::new(
            self.parameters,
            program.clone(),
            workload,
            payload.clone(),
            bytes,
        );
        // A_sigma is setup-fixed and projection-owned.  Prepare the complete
        // superset before walking the body so unused LUTs and LUTs in later
        // logical levels are still available to the independent encrypt
        // stage.  B_sigma remains body/site material and is added by `lut`.
        evaluator
            .prepare_lut_public_data_projection(&program.encryption_inputs())
            .map_err(ApiError::scheme)?;
        for (wire, declaration) in program.inputs() {
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(
                    evaluator.sample_input_public_key(*wire).map_err(ApiError::scheme)?,
                ),
                InputBinding::Family(count) => {
                    let family = mxx_dsl::parallel(count.clone(), |index| {
                        evaluator
                            .sample_input_public_key_member(*wire, Some(index))
                            .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    InputBinding::Family(family)
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|e| ApiError::input(e.to_string()))?;
        }
        for (id, declaration) in program.rhs_inputs() {
            let value = rhs.get(id).ok_or_else(|| ApiError::input("missing RHS plaintext"))?;
            match (declaration, value) {
                (InputBinding::Scalar(()), InputBinding::Scalar(poly)) => {
                    let plaintext = ring.polynomial(
                        poly.coeffs_biguints()
                            .into_iter()
                            .map(|v| IntExpr::constant(BigInt::from(v))),
                    );
                    evaluator.rhs.insert(*id, InputBinding::Scalar(plaintext));
                    let reference = RhsRef { id: *id, index: None::<Int> };
                    let material =
                        evaluator.prepare_rhs_material(&reference).map_err(ApiError::scheme)?;
                    evaluator.rhs_material.insert(*id, InputBinding::Scalar(material));
                    self.environment
                        .bind_rhs(*id, InputBinding::Scalar(reference))
                        .map_err(|e| ApiError::input(e.to_string()))?;
                }
                (InputBinding::Family(count), InputBinding::Family(values)) => {
                    let expected = count
                        .clone()
                        .evaluate(parameters)
                        .ok()
                        .and_then(|value| value.to_usize())
                        .ok_or_else(|| ApiError::input("RHS family count is not concrete"))?;
                    if expected != values.len() {
                        return Err(ApiError::input("RHS family count does not match Program"));
                    }
                    let plaintexts = Family::pack(
                        values
                            .iter()
                            .map(|value| {
                                ring.polynomial(
                                    value
                                        .coeffs_biguints()
                                        .into_iter()
                                        .map(|v| IntExpr::constant(BigInt::from(v))),
                                )
                            })
                            .collect(),
                    )
                    .map_err(|error| ApiError::input(error.to_string()))?;
                    evaluator.rhs.insert(*id, InputBinding::Family(plaintexts));
                    let prepared = mxx_dsl::parallel(IntExpr::constant(values.len()), |index| {
                        evaluator
                            .prepare_rhs_material(&RhsRef { id: *id, index: Some(index) })
                            .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    evaluator.rhs_material.insert(*id, InputBinding::Family(prepared));
                    let references = mxx_dsl::parallel(IntExpr::constant(values.len()), |index| {
                        Ok(RhsRef { id: *id, index: Some(index) })
                    })
                    .map_err(ApiError::scheme)?;
                    self.environment
                        .bind_rhs(*id, InputBinding::Family(references))
                        .map_err(|e| ApiError::input(e.to_string()))?;
                }
                _ => return Err(ApiError::input("RHS plaintext shape does not match Program")),
            }
        }
        // Plaintext RHS graph handles are needed only while preparing compact
        // setup-fixed ciphertexts.  Never retain them for online lowering.
        evaluator.rhs.clear();
        self.runtime.insert("rns-bgg/hash-key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec()));
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.evaluator = Some(evaluator);
        Ok(())
    }
    fn artifact_identities(&self) -> ([u8; 32], [u8; 32]) {
        let Some(evaluator) = self.evaluator.as_ref() else {
            return ([0; 32], [0; 32]);
        };
        let mut digest = Sha256::new();
        digest.update(evaluator.program.id().as_bytes());
        digest.update(self.parameters.parameter_identity());
        digest.update(self.hash_key_commitment);
        if let Some(declaration) = &self.refresh_declaration {
            digest.update(declaration.commitment());
        }
        let encryption = digest.clone().finalize().into();
        digest.update(b"online");
        let online = digest.finalize().into();
        (encryption, online)
    }
    fn finalize_descriptors(
        production: &ProductionId,
        (encrypt, online): (Self::EncryptDescriptor, Self::OnlineDescriptor),
    ) -> Result<(Self::EncryptDescriptor, Self::OnlineDescriptor), ApiError> {
        let encrypt = encrypt.with_source(production.clone()).map_err(ApiError::scheme)?;
        Ok((encrypt, online))
    }

    fn build_graph(
        mut self,
        outputs: Vec<RnsBggPublicKey>,
    ) -> Result<(StageGraph<B>, (Self::EncryptDescriptor, Self::OnlineDescriptor)), ApiError> {
        let mut evaluator =
            self.evaluator.ok_or_else(|| ApiError::input("preprocess builder was not bound"))?;
        let mut outputs = outputs;
        if let Some((index, output)) = self.pending_refresh.take() {
            let target = outputs.get_mut(index).ok_or_else(|| {
                ApiError::input("refresh output index is outside lowered outputs")
            })?;
            *target = output;
        }
        let outputs = evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-preprocess");
        for output in outputs {
            graph = graph
                .artifact_output(format!("public-key/{:?}", output.wire), output)
                .map_err(ApiError::scheme)?;
        }
        // Prepared PRF children use their own source-q key schedule.  Publish
        // their input keys under the declaration/slot namespace so encryption
        // cannot resolve a same-named outer-program key or another slot's key.
        if let Some(declaration) = self.refresh_declaration.as_ref() {
            let commitment = *declaration.commitment();
            for (slot, child) in &self.refresh_prf_children {
                for (wire, input) in child.program.inputs() {
                    let names = (0..4)
                        .map(|port| {
                            refresh_prf_input_key_artifact_name(commitment, *slot, *wire, port)
                        })
                        .collect::<Vec<_>>();
                    match input.shape() {
                        InputBinding::Scalar(()) => {
                            let key =
                                child.sample_input_public_key(*wire).map_err(ApiError::scheme)?;
                            for (name, entry) in names.into_iter().zip(matrix_entries(&key.matrix))
                            {
                                graph =
                                    graph.artifact_output(name, entry).map_err(ApiError::scheme)?;
                            }
                        }
                        InputBinding::Family(count) => {
                            let key_parts = (0..4)
                                .map(|port| {
                                    mxx_dsl::parallel(count.clone(), |index| {
                                        let key = child
                                            .sample_input_public_key_member(*wire, Some(index))
                                            .map_err(|_| DslError::Schema)?;
                                        Ok(matrix_entries(&key.matrix)[port].clone())
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()
                                .map_err(ApiError::scheme)?;
                            for (name, entry) in names.into_iter().zip(key_parts) {
                                graph =
                                    graph.artifact_output(name, entry).map_err(ApiError::scheme)?;
                            }
                        }
                    }
                }
                for (rhs, prepared) in &child.rhs_material {
                    match prepared {
                        InputBinding::Scalar(value) => {
                            graph = graph
                                .artifact_output(
                                    refresh_prf_artifact_name(
                                        commitment,
                                        *slot,
                                        &format!("rhs/{rhs:?}"),
                                    ),
                                    value.clone(),
                                )
                                .map_err(ApiError::scheme)?;
                        }
                        InputBinding::Family(value) => {
                            graph = graph
                                .artifact_output(
                                    refresh_prf_artifact_name(
                                        commitment,
                                        *slot,
                                        &format!("rhs/{rhs:?}/family"),
                                    ),
                                    value.clone(),
                                )
                                .map_err(ApiError::scheme)?;
                        }
                    }
                }
                for ((evaluation_level, decomposition_level, lut, layout), material) in
                    &child.lut_material
                {
                    let role = format!(
                        "lut-switching/{evaluation_level}/{decomposition_level}/{lut:?}/{layout:?}"
                    );
                    graph = graph
                        .artifact_output(
                            refresh_prf_artifact_name(commitment, *slot, &role),
                            material.clone(),
                        )
                        .map_err(ApiError::scheme)?;
                }
                for ((evaluation_level, sigma), matrix) in &child.lut_public_data {
                    for (port, entry) in matrix_entries(matrix).into_iter().enumerate() {
                        graph = graph
                            .artifact_output(
                                refresh_prf_artifact_name(
                                    commitment,
                                    *slot,
                                    &format!("lut-public/{evaluation_level}/{sigma}.{port}"),
                                ),
                                entry,
                            )
                            .map_err(ApiError::scheme)?;
                    }
                }
            }
        }
        for (rhs, prepared) in &evaluator.rhs_material {
            match prepared {
                InputBinding::Scalar(value) => {
                    graph = graph
                        .artifact_output(format!("rhs/{rhs:?}"), value.clone())
                        .map_err(ApiError::scheme)?;
                }
                InputBinding::Family(value) => {
                    graph = graph
                        .artifact_output(format!("rhs/{rhs:?}/family"), value.clone())
                        .map_err(ApiError::scheme)?;
                }
            }
        }
        for ((evaluation_level, decomposition_level, lut, layout), material) in
            &evaluator.lut_material
        {
            graph = graph
                .artifact_output(
                    format!(
                        "lut-switching/{evaluation_level}/{decomposition_level}/{lut:?}/{layout:?}"
                    ),
                    material.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        // A_sigma is deterministic public mask-alignment data.  Publish one
        // matrix per (evaluation level, sigma), independently of LUT,
        // layout, wire, and family site.  Encryption and online evaluation
        // resolve these canonical keys instead of receiving a synthetic
        // helper-origin family.
        for ((evaluation_level, sigma), matrix) in &evaluator.lut_public_data {
            for (port, entry) in matrix_entries(matrix).into_iter().enumerate() {
                graph = graph
                    .artifact_output(format!("lut-public/{evaluation_level}/{sigma}.{port}"), entry)
                    .map_err(ApiError::scheme)?;
            }
        }
        for ((commitment, slot), sum) in &evaluator.refresh_sums {
            for (index, entry) in matrix_entries(sum).into_iter().enumerate() {
                graph = graph
                    .artifact_output(
                        refresh_artifact_name(*commitment, *slot, &format!("a-sum.{index}")),
                        entry,
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
        for ((commitment, slot), star) in &evaluator.refresh_stars {
            for (index, entry) in matrix_entries(star).into_iter().enumerate() {
                graph = graph
                    .artifact_output(
                        refresh_artifact_name(*commitment, *slot, &format!("a-star.{index}")),
                        entry,
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
        for ((commitment, slot), action) in &evaluator.refresh_targets {
            graph = graph
                .artifact_output(
                    refresh_artifact_name(*commitment, *slot, "target"),
                    action.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        for ((commitment, slot, label), action) in &evaluator.refresh_label_targets {
            graph = graph
                .artifact_output(
                    refresh_label_artifact_name(*commitment, *slot, *label),
                    action.clone(),
                )
                .map_err(ApiError::scheme)?;
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let mut binding = RnsBggArtifactBinding::new(
            self.context,
            evaluator.program.id(),
            self.parameters.parameter_identity(),
            self.hash_key_commitment,
            Vec::<ProductionId>::new(),
        )
        .map_err(ApiError::scheme)?;
        if let Some(declaration) = &self.refresh_declaration {
            binding = binding.with_refresh_declaration(declaration);
        }
        let encrypt_binding = RnsBggPreprocessEncryptDescriptor::from_binding(binding.clone());
        let online_binding = RnsBggPreprocessOnlineDescriptor::from_binding(binding);
        Ok((
            StageGraph::new(graph, self.runtime, ArtifactCatalog::default()),
            (encrypt_binding, online_binding),
        ))
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
    use crate::{api::execute_graph, program::ProgramValidationError};
    use mxx_runtime::{
        ExecutionConfig, MemoryArtifactStore, RuntimeValue, backend::poly::cpu_backend,
    };

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
        let mut public = PublicKeyLowering::new(
            &parameters,
            owned_program.clone(),
            workload.clone(),
            RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)) },
            ring.bytes_input("equal-luts-hash-key", 32),
        );
        public.prepare_lut_public_data_projection(&program.encryption_inputs()).unwrap();
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
        let mut online = EncodingLowering {
            parameters: &parameters,
            program: owned_program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            catalog: mxx_ir_core::artifact::ArtifactCatalog::default(),
            refresh_prf_namespace: None,
            current_site: None,
        };
        let row = ring.input("equal-luts-input", (1, parameters.component_columns(0).unwrap()));
        let mut online_environment = Environment::new();
        online_environment
            .bind_wire(
                input,
                InputBinding::Scalar(RnsBggEncoding::new(input_key, (row.clone(), row), None)),
            )
            .unwrap();
        let online_outputs = lower_program(
            &program,
            &mut online,
            &mut GraphFamilyExecution,
            &mut online_environment,
        )
        .unwrap();
        let expected =
            parameters.ring(1).unwrap().matrix_type((1, parameters.component_columns(1).unwrap()));
        for (key, encoding) in public_outputs.iter().zip(&online_outputs) {
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
    }
    use mxx_ir_core::ParamEnv;

    #[test]
    fn public_key_carrier_is_wire_and_matrix() {
        let ring = mxx_dsl::Ring::new(BigInt::from(17u8), 4);
        let key = RnsBggPublicKey::<Mat>::new(
            WireId(3),
            ring.polynomial([IntExpr::constant(BigInt::from(0u8))]),
        );
        assert_eq!(key.wire(), WireId(3));
        assert_eq!(key.matrix().matrix_type(), &ring.matrix_type((1, 1)));
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
            RnsBggPayloadSecret { row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)) },
            ring.bytes_input("sampled-key-schema", 32),
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
        let program =
            Arc::new(builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap());
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
            RnsBggPayloadSecret {
                row: (high_ring.uniform_interval((1, 1), -1, 1), high_ring.identity(1)),
            },
            high_ring.bytes_input("mul-public-private-level", 32),
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
            RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)) },
            ring.bytes_input("mul-public-numeric-key", 32),
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
        let direct_action = |lhs: RnsBggMatrix, rhs: RnsBggDecomposition| {
            (
                (
                    rhs.0.0.clone().mul_small_rhs(lhs.0.0.clone()) +
                        rhs.1.0.clone().mul_small_rhs(lhs.0.1.clone()),
                    rhs.0.1.clone().mul_small_rhs(lhs.0.0) + rhs.1.1.clone().mul_small_rhs(lhs.0.1),
                ),
                (
                    rhs.0.0.mul_small_rhs(lhs.1.0.clone()) + rhs.1.0.mul_small_rhs(lhs.1.1.clone()),
                    rhs.0.1.mul_small_rhs(lhs.1.0) + rhs.1.1.mul_small_rhs(lhs.1.1),
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
        let mut backend = cpu_backend([level]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_graph(
            StageGraph::new(graph, BTreeMap::new(), ArtifactCatalog::default()),
            &ParamEnv::default(),
            &mut backend,
            &mut store,
            ExecutionConfig::default(),
            |_| Ok(()),
        )
        .unwrap();
        for index in 0..4 {
            let RuntimeValue::Matrix(matrix) = result
                .materialize_output(
                    &format!("private-difference-{index}"),
                    &mut backend,
                    &mut store,
                )
                .unwrap()
            else {
                panic!("private MulPublic difference is not a matrix");
            };
            assert!(matrix.entry(0, 0).coeffs_biguints().iter().all(|value| value.is_zero()));
            let RuntimeValue::Matrix(matrix) = result
                .materialize_output(&format!("public-difference-{index}"), &mut backend, &mut store)
                .unwrap()
            else {
                panic!("public MulPublic difference is not a matrix");
            };
            assert!(matrix.entry(0, 0).coeffs_biguints().iter().all(|value| value.is_zero()));
        }
        let RuntimeValue::Matrix(wrong) =
            result.materialize_output("wrong-key", &mut backend, &mut store).unwrap()
        else {
            panic!("wrong key difference is not a matrix");
        };
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
    fn refresh_routes_masks_to_owner_and_fresh_error_to_every_slot() {
        let refresh_id = [0xabu8; 32];
        let mask =
            RefreshPrfLabel::Mask { refresh_id, slot: 1, component: 0, coefficient: 0, digit: 0 };
        let fresh =
            RefreshPrfLabel::FreshError { refresh_id, component: 0, coefficient: 0, digit: 0 };
        assert_eq!(expected_refresh_slots(&mask, 3).unwrap(), vec![1]);
        assert_eq!(expected_refresh_slots(&fresh, 3).unwrap(), vec![0, 1, 2]);
        assert!(expected_refresh_slots(&mask, 1).is_err());
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
        let source = mxx_dsl::Ring::new(BigInt::from(105u8), 4);
        let value = source.input("refresh-source", (1, 1));
        let matrix = ((value.clone(), value.clone()), (value.clone(), value));
        let lifted = mod_down_lift_public(matrix, BigInt::from(3u8), BigInt::from(15u8));
        for row in [lifted.0, lifted.1] {
            for entry in [row.0, row.1] {
                assert_eq!(entry.matrix_type().modulus, IntExpr::constant(BigInt::from(15u8)));
            }
        }
        assert_eq!(
            refresh_crt_weight(
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
            BigInt::from(10u8)
        );
    }

    #[test]
    fn refresh_anchor_reduction_is_not_moddown_and_decoder_target_is_distinct() {
        let source = mxx_dsl::Ring::new(BigInt::from(105u8), 4);
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
        for channel_modulus in [BigInt::from(3u8), BigInt::from(5u8)] {
            let reduced = reduce_lift_public_anchor(
                anchor.clone(),
                channel_modulus.clone(),
                BigInt::from(15u8),
            )
            .unwrap();
            let divided = mod_down_lift_public(anchor.clone(), channel_modulus, BigInt::from(15u8));
            assert_ne!(reduced.flatten(), divided.flatten());
            assert_eq!(reduced.0.0.matrix_type().modulus, IntExpr::constant(BigInt::from(15u8)));
        }

        // H is intentionally non-unit (p/q_t = 105/5 = 21).  The decoder
        // target includes the A_sum contribution and remains a separate
        // artifact from the public A_star carrier.
        let decoder_target = mod_down_lift_public(
            add_public_matrices(a_sum, scale_public_matrix(anchor.clone(), &BigInt::from(-21i8))),
            BigInt::from(5u8),
            BigInt::from(15u8),
        );
        let public_anchor =
            reduce_lift_public_anchor(anchor, BigInt::from(5u8), BigInt::from(15u8)).unwrap();
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
        let qstar = mxx_dsl::Ring::new(BigInt::from(15u8), 4);
        let value = qstar.input("refresh-qstar-target", (1, 1));
        let row = (value.clone(), value);
        let lifted = lift_output_target_to_p(
            row.clone(),
            BigInt::from(105u8),
            qstar.polynomial([IntExpr::constant(BigInt::from(3u8))]),
        );
        let reduced = checked_modulus_convert_row(row, BigInt::from(105u8)).unwrap();
        let reduced = super::map_row(reduced, |entry| {
            qstar.polynomial([IntExpr::constant(BigInt::from(3u8))]) * entry
        });
        assert_eq!(lifted.0.matrix_type().modulus, IntExpr::constant(BigInt::from(105u8)));
        assert_ne!(lifted.flatten(), reduced.flatten());
        let graph = DslContext::new("refresh-qstar-to-p")
            .artifact_output("lifted", lifted)
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
        };
        let p_ring = parameters.ring(decomposition_level).unwrap();
        let desired = (p_ring.zero((1, target_columns)), p_ring.zero((1, target_columns)));
        let action = refresh_payload_action(
            &parameters,
            decomposition_level,
            &payload,
            desired,
            hash_key,
            mxx_dsl::HashTag::from(b"refresh-width-test".to_vec()),
            target_columns,
        )
        .unwrap();
        assert_eq!(
            action.0.0.matrix_type().modulus,
            IntExpr::constant(BigInt::from(
                parameters.level(decomposition_level).unwrap().modulus().as_ref().clone()
            ))
        );
        assert_eq!(action.0.0.matrix_type().columns, IntExpr::constant(target_columns));

        let graph = DslContext::new("refresh-target-width")
            .artifact_output("anchor", anchor.matrix.clone())
            .unwrap()
            .artifact_output("action", action.clone())
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();

        let a_star = checked_modulus_convert_matrix(
            anchor.matrix,
            BigInt::from(parameters.level(decomposition_level).unwrap().modulus().as_ref().clone()),
        )
        .unwrap();
        assert_eq!(a_star.0.0.matrix_type(), &p_ring.matrix_type((1, target_columns)));
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
        let payload =
            RnsBggPayloadSecret { row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)) };
        let mut lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            payload,
            ring.bytes_input("rhs-schema-key", 32),
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
        let action = centered_extend_decomposition(
            decompose_matrix(&parameters, 1, compact.clone()).unwrap(),
            BigInt::from(parameters.level(0).unwrap().modulus().as_ref().clone()),
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
            .artifact_output("rhs", action)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn bind_inputs_canonicalizes_later_rhs_levels_from_natural_input_ring() {
        use mxx_primitives::poly::dcrt::poly::DCRTPoly;
        use mxx_runtime::backend::poly::CpuDcrtBackend;

        let source = DCRTPolyParams::new(16, 3, 17, 4, None, None);
        let moduli = source.to_crt().0;
        let middle = source
            .select_modulus(
                &moduli[..2]
                    .iter()
                    .fold(BigUint::one(), |product, prime| product * BigUint::from(*prime)),
            )
            .unwrap();
        let terminal = source.select_modulus(&BigUint::from(moduli[0])).unwrap();
        let parameters =
            RnsBggParameters::new(vec![source.clone(), middle, terminal], 1, 1).unwrap();

        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let first_rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(first_rhs),
            )
            .unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(second_rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();
        let expected_workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
        assert_eq!(expected_workload.rhs_level(first_rhs).unwrap(), 0);
        assert_eq!(expected_workload.rhs_level(second_rhs).unwrap(), 1);

        let ring = parameters.ring(0).unwrap();
        let payload =
            RnsBggPayloadSecret { row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)) };
        let value = |constant| {
            DCRTPoly::from_u32s(&source, &vec![constant; source.ring_dimension() as usize])
        };
        let mut rhs = BTreeMap::new();
        rhs.insert(first_rhs, InputBinding::Scalar(value(3)));
        rhs.insert(second_rhs, InputBinding::Scalar(value(5)));

        let mut preprocess = RnsPreprocessBuild::<CpuDcrtBackend>::new(&parameters, [0x52; 32]);
        <RnsPreprocessBuild<'_, CpuDcrtBackend> as PreprocessBuild<
            CpuDcrtBackend,
            DCRTPoly,
        >>::bind_inputs(
            &mut preprocess,
            &program,
            &ParamEnv::default(),
            &rhs,
            &payload,
            b"later-rhs-levels",
        )
        .unwrap();

        let evaluator = preprocess.evaluator.as_ref().unwrap();
        let second_material = match evaluator.rhs_material.get(&second_rhs).unwrap() {
            InputBinding::Scalar(value) => value,
            InputBinding::Family(_) => panic!("scalar RHS has family material"),
        };
        let expected_schema =
            parameters.ring(2).unwrap().matrix_type((1, parameters.component_columns(2).unwrap()));
        for entry in matrix_entries(second_material) {
            assert_eq!(entry.matrix_type(), &expected_schema);
        }
    }

    #[test]
    fn rhs_payload_ciphertext_has_zero_noise_private_product_residual() {
        // This is the scalar oracle for the graph construction above.  The
        // top sample U cancels from the payload product; the remaining term
        // is the desired next-level payload target (up to the configured
        // fresh error).
        let p = 11i64;
        let q = 121i64;
        let nu = 3i64;
        let delta_next = 5i64;
        let gadget_next = 7i64;
        let r0 = 2i64;
        let r1 = 1i64;
        let u = 4i64;
        let desired = nu * delta_next * gadget_next;
        let bottom = desired - r0 * u;
        let payload_product = (r0 * u + r1 * bottom).rem_euclid(p);
        assert_eq!(payload_product, desired.rem_euclid(p));

        let private_input = 9i64;
        let product = (private_input * payload_product).rem_euclid(q);
        assert_eq!(product.rem_euclid(p), (private_input * desired).rem_euclid(p));
        assert_ne!(desired.rem_euclid(p), 0);
        assert!(q > p && p > 1);
    }

    #[test]
    fn rhs_family_members_reuse_the_next_level_action_schema() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
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
        let payload =
            RnsBggPayloadSecret { row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)) };
        let mut lowering = PublicKeyLowering::new(
            &parameters,
            program,
            workload,
            payload,
            ring.bytes_input("rhs-family-schema-key", 32),
        );
        lowering.rhs.insert(
            rhs,
            InputBinding::Family(
                Family::pack(vec![
                    ring.polynomial([IntExpr::constant(BigInt::from(2u8))]),
                    ring.polynomial([IntExpr::constant(BigInt::from(5u8))]),
                ])
                .unwrap(),
            ),
        );
        let first = lowering
            .prepare_rhs_material(&RhsRef { id: rhs, index: Some(Int::constant(0usize)) })
            .unwrap();
        let second = lowering
            .prepare_rhs_material(&RhsRef { id: rhs, index: Some(Int::constant(1usize)) })
            .unwrap();
        assert_eq!(first.0.0.matrix_type(), second.0.0.matrix_type());
        assert_ne!(first.flatten(), second.flatten());
    }

    #[test]
    fn equal_width_lut_tables_produce_distinct_public_branch_coefficients() {
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
    }

    #[test]
    fn public_lut_branch_is_helper_action_only() {
        let ring = mxx_dsl::Ring::new(BigInt::from(17u8), 4);
        let data = (
            (ring.input("lut-data-00", (1, 1)), ring.input("lut-data-01", (1, 1))),
            (ring.input("lut-data-10", (1, 1)), ring.input("lut-data-11", (1, 1))),
        );
        let switching_decomposition = (
            (
                ring.preimage_input("lut-switching-decomposition-00", (1, 1), 17),
                ring.preimage_input("lut-switching-decomposition-01", (1, 1), 17),
            ),
            (
                ring.preimage_input("lut-switching-decomposition-10", (1, 1), 17),
                ring.preimage_input("lut-switching-decomposition-11", (1, 1), 17),
            ),
        );
        let branch = public_lut_branch(data.clone(), switching_decomposition.clone());
        let expected = multiply_public_action(data, switching_decomposition);
        assert_eq!(branch.0.0.matrix_type(), expected.0.0.matrix_type());
        let graph = DslContext::new("public-lut-branch")
            .artifact_output("branch", branch)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn lut_payload_target_uses_distinct_r0_and_r1_components() {
        let ring = mxx_dsl::Ring::new(BigInt::from(257u16), 4);
        let r0 = ring.input("lut-r0", (1, 1));
        let r1 = ring.identity(1);
        let one = ring.identity(1);
        let target = lut_payload_target(one.clone(), one, (r0, r1), ring.identity(1));
        assert_ne!(target.0.flatten(), target.1.flatten());
    }

    #[test]
    fn switching_decomposition_is_centered_extended_into_the_source_ring() {
        let next = mxx_dsl::Ring::new(BigInt::from(17u8), 4);
        let compact = next.preimage_input("lut-switching-decomposition-next", (1, 1), 17);
        let decomposition = ((compact.clone(), compact.clone()), (compact.clone(), compact));
        let extended = centered_extend_decomposition(decomposition, BigInt::from(257u16));
        assert_eq!(extended.0.0.matrix_type().modulus, IntExpr::constant(BigInt::from(257u16)));
        assert_eq!(extended.1.1.matrix_type().modulus, IntExpr::constant(BigInt::from(257u16)));
    }
}
