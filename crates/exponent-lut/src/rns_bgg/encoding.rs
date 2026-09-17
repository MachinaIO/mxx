//! Encoding, input encryption, and online lowering for RNS-BGG.
//!
//! Secret-dependent sampling belongs to [`RnsEncryptBuild`].  The online
//! lowerer only resolves lazy, compact artifact handles and evaluates the
//! graph operations described by the common Program walker.
#![warn(missing_docs)]

use super::*;
use crate::{
    api::{
        ApiError, EncryptBuild, OnlineBuild, PlaintextInputs, RefreshApplicationError,
        StageArtifacts, StageGraph, apply_refresh,
    },
    noise_refresh::{
        prf::PrfLabel,
        refresh::{NoiseRefresh, RefreshDeclaration, RefreshError, RefreshPrfLabel},
    },
    program::{
        AttributeKnowledge, Environment, ExpLayout, ExponentLUTProgramLowering, ExponentLutProgram,
        FamilyScope, GraphFamilyExecution, GraphParallelLowering, InputBinding, LoweringSite,
        LutTable, Operand, ProgramLoweringContext, RhsRef, WireId,
    },
};
use mxx_dsl::{DslContext, DslError, Family, GraphValue, GraphValueSchema, Int, Mat};
use mxx_ir_core::{
    IntExpr, ParamEnv, ValueHandle,
    artifact::{ArtifactCatalog, ArtifactType, ProductionId},
    node::ConstantMatrix,
    types::{ConcreteMatrixType, WireType},
};
use mxx_primitives::poly::Poly;
use mxx_runtime::{Backend, RuntimeValue};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Encoded row paired with its Program wire and, for public wires, plaintext.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggEncoding<R = RnsBggRow, A = Mat> {
    pub(crate) wire: WireId,
    pub(crate) row: R,
    pub(crate) public_payload: Option<A>,
}

impl<R, A> RnsBggEncoding<R, A> {
    /// Construct an encoding carrier for `wire`.
    ///
    /// Callers are responsible for preserving the Program wire identity,
    /// row schema, and public-payload presence implied by that wire's
    /// validated knowledge; stage builders validate these invariants when
    /// the carrier is consumed.
    pub fn new(wire: WireId, row: R, public_payload: Option<A>) -> Self {
        Self { wire, row, public_payload }
    }
    /// Program wire represented by this encoding.
    pub fn wire(&self) -> WireId {
        self.wire
    }
    /// Encoded row.
    pub fn row(&self) -> &R {
        &self.row
    }
    /// Paired public payload, if present.
    pub fn public_payload(&self) -> Option<&A> {
        self.public_payload.as_ref()
    }
}

impl RnsBggEncoding<RnsBggRow, Mat> {
    /// Returns the encryption residual against the deterministic BGG relation.
    /// Fresh Gaussian error is therefore the returned value; no plaintext is
    /// reconstructed and public payload pairing remains metadata-driven.
    pub fn verification_residual(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
        public_key: &RnsBggPublicKey,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        expected: Mat,
    ) -> Result<RnsBggRow, RnsBggError> {
        if self.wire != public_key.wire {
            return Err(RnsBggError::Binding("verification residual wire mismatch"));
        }
        let ring = parameters.ring(level)?;
        if expected.matrix_type() != &ring.matrix_type((1, 1)) {
            return Err(RnsBggError::Binding("verification residual plaintext ring"));
        }
        let mask = reduce_secret(parameters, &mask.row, level)?;
        let mask_matrix = (
            mask.0.clone() * public_key.matrix.0.0.clone() +
                mask.1.clone() * public_key.matrix.1.0.clone(),
            mask.0.clone() * public_key.matrix.0.1.clone() +
                mask.1.clone() * public_key.matrix.1.1.clone(),
        );
        let delta = parameters.scalar(level, parameters.scale(level)?.clone())?;
        let deterministic = if self.public_payload.is_some() {
            let gadget = parameters.public_gadget(level)?;
            let scale = expected.clone() * gadget * delta;
            (mask.0.clone() * scale.clone(), mask.1.clone() * scale)
        } else {
            let payload = reduce_secret(parameters, &payload.row, level)?;
            let gadget = parameters.state_gadget(level)?;
            let target = (payload.0 * gadget.clone(), payload.1 * gadget);
            let scale = delta * expected;
            (target.0 * scale.clone(), target.1 * scale)
        };
        Ok((
            self.row.0.clone() - mask_matrix.0 + deterministic.0,
            self.row.1.clone() - mask_matrix.1 + deterministic.1,
        ))
    }
}

impl<R: GraphValue, A: GraphValue> GraphValue for RnsBggEncoding<R, A> {
    type Schema = RnsBggEncoding<R::Schema, A::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.row.flatten();
        if let Some(payload) = &self.public_payload {
            values.extend(payload.flatten());
        }
        values
    }
    fn schema(&self) -> Self::Schema {
        RnsBggEncoding {
            wire: self.wire,
            row: self.row.schema(),
            public_payload: self.public_payload.as_ref().map(GraphValue::schema),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let row_len = schema.row.wire_types().len();
        if values.len() < row_len {
            return Err(DslError::Schema);
        }
        let row = R::from_values(&schema.row, &values[..row_len])?;
        let public_payload = match &schema.public_payload {
            Some(payload) => Some(A::from_values(payload, &values[row_len..])?),
            None if values.len() == row_len => None,
            None => return Err(DslError::Schema),
        };
        Ok(Self { wire: schema.wire, row, public_payload })
    }
}

impl<R: GraphValueSchema, A: GraphValueSchema> GraphValueSchema for RnsBggEncoding<R, A> {
    type Value = RnsBggEncoding<R::Value, A::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggEncoding {
            wire: self.wire,
            row: self.row.placeholders_from(next),
            public_payload: self.public_payload.as_ref().map(|p| p.placeholders_from(next)),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        let mut values = self.row.wire_types();
        if let Some(payload) = &self.public_payload {
            values.extend(payload.wire_types());
        }
        values
    }
}

/// Secret payload row used for private input and RHS encryption.
#[derive(Clone)]
pub struct RnsBggPayloadSecret {
    pub(crate) row: RnsBggRow,
}

/// Secret mask row used for input and helper encryption.
#[derive(Clone)]
pub struct RnsBggMaskSecret {
    pub(crate) row: RnsBggRow,
}

fn graph<E: std::fmt::Display>(error: E) -> RnsBggError {
    RnsBggError::Graph(error.to_string())
}

fn merge_runtime_inputs<B: Backend>(
    destination: &mut BTreeMap<String, RuntimeValue<B>>,
    source: &BTreeMap<String, RuntimeValue<B>>,
) -> Result<(), ApiError> {
    for (name, value) in source {
        match destination.get(name) {
            Some(RuntimeValue::Bytes(existing)) => match value {
                RuntimeValue::Bytes(candidate) if existing == candidate => {}
                RuntimeValue::Bytes(_) => {
                    return Err(ApiError::input("refresh child runtime input conflicts"));
                }
                _ => return Err(ApiError::input("refresh child runtime input schema")),
            },
            Some(_) => return Err(ApiError::input("refresh child runtime input schema")),
            None => {
                destination.insert(name.clone(), value.clone());
            }
        }
    }
    Ok(())
}

fn secret_row(parameters: &RnsBggParameters) -> Result<RnsBggRow, RnsBggError> {
    let ring = parameters.ring(0)?;
    Ok((ring.uniform_interval((1, 1), -1, 1), ring.identity(1)))
}

fn reduce_secret(
    parameters: &RnsBggParameters,
    row: &RnsBggRow,
    level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let modulus = BigInt::from(parameters.level(level)?.modulus().as_ref().clone());
    checked_modulus_convert_row(row.clone(), modulus)
}

pub(crate) fn fresh_error(
    parameters: &RnsBggParameters,
    level: usize,
    columns: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let ring = parameters.ring(level)?;
    let sigma = crate::utils::decimal_to_real_expr(parameters.encoding_error_sigma())
        .map_err(|_| RnsBggError::Parameters("invalid encoding error sigma"))?;
    Ok((
        ring.gaussian((1, columns), sigma.clone(), parameters.encoding_error_bound().clone()),
        ring.gaussian((1, columns), sigma, parameters.encoding_error_bound().clone()),
    ))
}

fn encode_private_inner(
    parameters: &RnsBggParameters,
    key: &RnsBggPublicKey,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    plaintext: Mat,
    level: usize,
) -> Result<RnsBggEncoding, RnsBggError> {
    let ring = parameters.ring(level)?;
    if *plaintext.matrix_type() != ring.matrix_type((1, 1)) {
        return Err(RnsBggError::Binding("private plaintext ring"));
    }
    let s = reduce_secret(parameters, &mask.row, level)?;
    let r = reduce_secret(parameters, &payload.row, level)?;
    let gadget = parameters.state_gadget(level)?;
    let target = (r.0.clone() * gadget.clone(), r.1.clone() * gadget.clone());
    let delta = parameters.scalar(level, parameters.scale(level)?.clone())?;
    let matrix = key.matrix.clone();
    let mask_matrix =
        (s.0.clone() * matrix.0.0 + s.1.clone() * matrix.1.0, s.0 * matrix.0.1 + s.1 * matrix.1.1);
    let scale = delta * plaintext;
    let error = fresh_error(parameters, level, parameters.component_columns(level)?)?;
    let row = (
        mask_matrix.0 - target.0 * scale.clone() + error.0,
        mask_matrix.1 - target.1 * scale + error.1,
    );
    Ok(RnsBggEncoding::new(key.wire, row, None))
}

fn encode_public_inner(
    parameters: &RnsBggParameters,
    key: &RnsBggPublicKey,
    mask: &RnsBggMaskSecret,
    plaintext: Mat,
    level: usize,
) -> Result<RnsBggEncoding, RnsBggError> {
    let ring = parameters.ring(level)?;
    if *plaintext.matrix_type() != ring.matrix_type((1, 1)) {
        return Err(RnsBggError::Binding("public plaintext ring"));
    }
    let s = reduce_secret(parameters, &mask.row, level)?;
    let gadget = parameters.public_gadget(level)?;
    let scale = parameters.scalar(level, parameters.scale(level)?.clone())?;
    let matrix = key.matrix.clone();
    let mask_matrix = (
        s.0.clone() * matrix.0.0 + s.1.clone() * matrix.1.0,
        s.0.clone() * matrix.0.1 + s.1.clone() * matrix.1.1,
    );
    let public_scale = plaintext.clone() * gadget * scale;
    let error = fresh_error(parameters, level, parameters.level(level)?.modulus_digits())?;
    let row = (
        mask_matrix.0 - s.0 * public_scale.clone() + error.0,
        mask_matrix.1 - s.1 * public_scale + error.1,
    );
    Ok(RnsBggEncoding::new(key.wire, row, Some(plaintext)))
}

/// Online lowering state.  Its artifact fields are metadata and lazy handles,
/// never resident matrix payloads.
pub struct EncodingLowering<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
    pub(crate) program: Arc<ExponentLutProgram>,
    pub(crate) workload: Arc<RnsBggDerivedWorkload>,
    pub(crate) preprocess: ProductionId,
    pub(crate) encryption: ProductionId,
    pub(crate) catalog: ArtifactCatalog,
    pub(crate) current_site: Option<LoweringSite>,
}

impl EncodingLowering<'_> {
    /// Compute the native Walsh/automorphism branch polynomial for one unary
    /// LUT branch.  The coefficient depends on the complete table, not only
    /// its width; public-key preparation uses this same helper.
    pub(crate) fn lut_coefficient(
        parameters: &RnsBggParameters,
        level: usize,
        representation: crate::program::LutOutputRepresentation,
        sigma: impl Into<IntExpr>,
        values: &[usize],
    ) -> Result<Mat, RnsBggError> {
        let width = values.len();
        if width == 0 || !width.is_power_of_two() {
            return Err(RnsBggError::Lut);
        }
        let layout = match representation {
            crate::program::LutOutputRepresentation::Coefficient => {
                ExpLayout { stride_log2: 0, width_log2: 0 }
            }
            crate::program::LutOutputRepresentation::Exponent(layout) => layout,
        };
        let ring = parameters.ring(level)?;
        let modulus = parameters.level(level)?.modulus().as_ref().clone();
        let inverse = mxx_primitives::utils::mod_inverse_biguints(
            &(BigUint::from(width) % &modulus),
            &modulus,
        )
        .ok_or(RnsBggError::Lut)?;
        let sigma = sigma.into();
        let mut terms = values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, output)| {
                let left =
                    layout.mono(&ring, &BigInt::from(output)).map_err(|_| RnsBggError::Lut)?;
                let right = layout
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

    fn site(&self) -> Result<&LoweringSite, RnsBggError> {
        self.current_site.as_ref().ok_or(RnsBggError::Binding("missing lowering site"))
    }

    fn scope_extent(site: &LoweringSite) -> Result<Option<(IntExpr, IntExpr)>, RnsBggError> {
        let scope = site
            .family_scope()
            .map_err(|_| RnsBggError::Binding("invalid lowering family scope"))?;
        if scope.ranges().is_empty() {
            return Ok(None);
        }
        let mut extent = IntExpr::constant(1);
        for range in scope.ranges() {
            extent = IntExpr::Mul(Box::new(extent), Box::new(range.count().clone())).canonicalize();
        }
        Ok(Some((
            extent,
            scope
                .flattened_index()
                .map_err(|_| RnsBggError::Binding("invalid lowering family index"))?,
        )))
    }

    fn rhs_decomposition(&self, rhs: &RhsRef<Int>) -> Result<RnsBggDecomposition, RnsBggError> {
        let level = self.workload.rhs_level(rhs.id)?;
        let decomposition_level =
            level.checked_add(1).ok_or(RnsBggError::Parameters("RHS target level overflow"))?;
        // RHS preprocessing samples and decomposes at the next modulus-chain
        // level, then centered-extends the compact digits into the source
        // evaluation ring.  Keep those concerns separate here: the imported
        // matrices live in `level`, while their compact width and coefficient
        // bound come from `decomposition_level`.
        if let Some(index) = rhs.index.clone() {
            let count = match self.program.rhs_inputs().get(&rhs.id) {
                Some(InputBinding::Family(count)) => count.clone(),
                _ => return Err(RnsBggError::Binding("family RHS index")),
            };
            let ring = self.parameters.ring(level)?;
            let columns = self.parameters.component_columns(decomposition_level)?;
            let bound =
                BigInt::from(1u8) << self.parameters.level(decomposition_level)?.base_bits();
            let preimage_family = |port: usize| {
                ring.preimage_family_artifact_input(
                    self.preprocess.clone(),
                    format!("rhs/{:?}/family.{port}", rhs.id),
                    count.clone(),
                    (1, columns),
                    bound.clone(),
                )
            };
            let first = preimage_family(0);
            let second = preimage_family(1);
            let third = preimage_family(2);
            let fourth = preimage_family(3);
            return Ok((
                (first.at(index.clone()), second.at(index.clone())),
                (third.at(index.clone()), fourth.at(index)),
            ));
        }
        self.decomposition_at(
            &self.preprocess,
            format!("rhs/{:?}", rhs.id),
            level,
            decomposition_level,
        )
    }

    fn mul_public_action(&self) -> Result<RnsBggDecomposition, RnsBggError> {
        let site = self.site()?;
        let level = self.workload.level(site.output)?;
        if let Some((extent, index)) = Self::scope_extent(site)? {
            let ring = self.parameters.ring(level)?;
            let columns = self.parameters.component_columns(level)?;
            let bound = BigInt::from(1u8) << self.parameters.level(level)?.base_bits();
            let families = (0..4)
                .map(|port| {
                    ring.preimage_family_artifact_input(
                        self.preprocess.clone(),
                        format!("public-action/{:?}.{port}", site.output),
                        extent.clone(),
                        (1, columns),
                        bound.clone(),
                    )
                    .at(index.clone())
                })
                .collect::<Vec<_>>();
            return Ok((
                (families[0].clone(), families[1].clone()),
                (families[2].clone(), families[3].clone()),
            ));
        }
        self.decomposition(&self.preprocess, format!("public-action/{:?}", site.output), level)
    }
    fn lut_actions(
        &self,
        count: IntExpr,
        evaluation_level: usize,
    ) -> Result<Family<(RnsBggDecomposition, RnsBggDecomposition)>, RnsBggError> {
        let site = self.site()?;
        let decomposition_level =
            evaluation_level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let evaluation_ring = self.parameters.ring(evaluation_level)?;
        let decomposition_parameters = self.parameters.level(decomposition_level)?;
        // The producer decomposes B_sigma and U_sigma at the next level, then
        // centered-extends each compact digit back into the input evaluation
        // ring.  Thus the imported preimages use the evaluation modulus, the
        // next-level radix/digit bound, and the next-level output width.
        let target_columns = self.parameters.component_columns(decomposition_level)?;
        let bound = BigInt::from(1u8) << decomposition_parameters.base_bits();
        let (family_count, scope_index) = match Self::scope_extent(site)? {
            Some((extent, index)) => (
                IntExpr::Mul(Box::new(extent), Box::new(count.clone())).canonicalize(),
                Some(index),
            ),
            None => (count.clone(), None),
        };
        let families = (0..8)
            .map(|port| {
                evaluation_ring.preimage_family_artifact_input(
                    self.preprocess.clone(),
                    format!("lut-action/{:?}.{port}", site.output),
                    family_count.clone(),
                    (1, target_columns),
                    bound.clone(),
                )
            })
            .collect::<Vec<_>>();
        let branch_count = count.clone();
        mxx_dsl::parallel(branch_count.clone(), |index| {
            let index = match scope_index.clone() {
                Some(scope_index) => Int::evaluate(scope_index)
                    .mul(Int::evaluate(branch_count.clone()))
                    .add(index.clone()),
                None => index.clone(),
            };
            Ok((
                (
                    (families[0].at(index.clone()), families[1].at(index.clone())),
                    (families[2].at(index.clone()), families[3].at(index.clone())),
                ),
                (
                    (families[4].at(index.clone()), families[5].at(index.clone())),
                    (families[6].at(index.clone()), families[7].at(index)),
                ),
            ))
        })
        .map_err(graph)
    }
    fn lut_helper(&self, count: IntExpr, origin: WireId) -> Result<RnsBggLutHelper, RnsBggError> {
        let level = self.workload.level(origin)?;
        let ring = self.parameters.ring(level)?;
        let columns = self.parameters.component_columns(level)?;
        let scope = self
            .site()?
            .family_scope()
            .map_err(|_| RnsBggError::Binding("invalid LUT helper scope"))?;
        let (family_count, scope_index) = match Self::scope_extent(self.site()?)? {
            Some((extent, index)) => (
                IntExpr::Mul(Box::new(extent), Box::new(count.clone())).canonicalize(),
                Some(index),
            ),
            None => (count.clone(), None),
        };
        let first = ring.family_artifact_input(
            self.encryption.clone(),
            format!("{}.0", lut_helper_artifact_name(origin, &scope)),
            family_count.clone(),
            (1, columns),
        );
        let second = ring.family_artifact_input(
            self.encryption.clone(),
            format!("{}.1", lut_helper_artifact_name(origin, &scope)),
            family_count.clone(),
            (1, columns),
        );
        let branch_count = count.clone();
        mxx_dsl::parallel(branch_count.clone(), |index| {
            let index = match scope_index.clone() {
                Some(scope_index) => Int::evaluate(scope_index)
                    .mul(Int::evaluate(branch_count.clone()))
                    .add(index.clone()),
                None => index.clone(),
            };
            Ok((first.at(index.clone()), second.at(index)))
        })
        .map_err(graph)
    }

    fn decomposition(
        &self,
        production: &ProductionId,
        prefix: String,
        level: usize,
    ) -> Result<RnsBggDecomposition, RnsBggError> {
        self.decomposition_at(production, prefix, level, level)
    }

    fn decomposition_at(
        &self,
        production: &ProductionId,
        prefix: String,
        evaluation_level: usize,
        decomposition_level: usize,
    ) -> Result<RnsBggDecomposition, RnsBggError> {
        let ring = self.parameters.ring(evaluation_level)?;
        let columns = self.parameters.component_columns(decomposition_level)?;
        let params = self.parameters.level(decomposition_level)?;
        let bound = BigInt::from(1u8) << params.base_bits();
        let preimage = |port: usize| {
            ring.preimage_artifact_input(
                production.clone(),
                format!("{prefix}.{port}"),
                (1, columns),
                bound.clone(),
            )
        };
        Ok(((preimage(0), preimage(1)), (preimage(2), preimage(3))))
    }

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
        let metadata = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let ring = mxx_dsl::Ring::new(
            metadata.q.clone(),
            self.parameters.level(q_level)?.ring_dimension() as usize,
        );
        // Actions are evaluated against raw q carriers, but their compact
        // decomposition uses the producer's p-level radix/digit count and
        // the target output's column width.
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let bound = BigInt::from(1u8) << self.parameters.level(p_level)?.base_bits();
        let prefix = match label {
            Some(label) => refresh_label_artifact_name(*declaration.commitment(), slot, label),
            None => refresh_artifact_name(*declaration.commitment(), slot, role),
        };
        let preimage = |port: usize| {
            ring.preimage_artifact_input(
                self.preprocess.clone(),
                format!("{prefix}.{port}"),
                (1, columns),
                bound.clone(),
            )
        };
        Ok(((preimage(0), preimage(1)), (preimage(2), preimage(3))))
    }

    fn refresh_decoder(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        let level = refresh_slot_level(self.parameters, declaration, slot)?;
        let metadata = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let ring = mxx_dsl::Ring::new(
            metadata.p.clone(),
            self.parameters.level(level)?.ring_dimension() as usize,
        );
        // The decoder is a target-matrix artifact.  Keep the slot p-level for
        // its ring and decomposition contract, but use the declared target
        // level for the matrix width.
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let prefix = refresh_artifact_name(*declaration.commitment(), slot, "decoder");
        Ok((
            ring.artifact_input(self.encryption.clone(), format!("{prefix}.0"), (1, columns)),
            ring.artifact_input(self.encryption.clone(), format!("{prefix}.1"), (1, columns)),
        ))
    }

    fn refresh_crt_weight(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<Mat, RnsBggError> {
        let mut product = BigUint::from(1u8);
        for value in declaration.slots() {
            product *= value
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        }
        let selected = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?
            .q_t
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        let base = &product / &selected;
        let inverse = mxx_primitives::utils::mod_inverse_biguints(&(&base % &selected), &selected)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        let target_parameters = self.parameters.level(declaration.target_level())?;
        let ring_dimension = target_parameters.ring_dimension();
        Ok(mxx_dsl::Ring::new(
            target_parameters.modulus().as_ref().clone(),
            ring_dimension as usize,
        )
        .polynomial([IntExpr::constant(BigInt::from(base * inverse))]))
    }

    /// Reconstruct the public refresh target from the public per-slot p-ring
    /// anchor emitted by preprocessing.  The anchor is reduced to `q_t`, centered
    /// extended to `Q*`, and then CRT recomposed.  The decoder's
    /// `A_sum-H*A_star` term is deliberately kept separate in the encryption
    /// graph; it is not part of the public target key.
    fn refresh_public_target(
        &self,
        declaration: &RefreshDeclaration,
        wire: WireId,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        let level = declaration.target_level();
        let target_modulus = self.parameters.level(level)?.modulus().as_ref().clone();
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
            let ring = mxx_dsl::Ring::new(
                slot.p.clone(),
                self.parameters.level(source_level)?.ring_dimension() as usize,
            );
            let matrix = |role: &str| -> RnsBggMatrix {
                (
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.0", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.1", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.2", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.3", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                        ),
                    ),
                )
            };
            let a_star = matrix("a-star");
            let target = reduce_anchor_channel(
                a_star,
                BigInt::from(
                    slot.q_t
                        .to_biguint()
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?,
                ),
                BigInt::from(target_modulus.clone()),
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
    fn finish_outputs(
        &mut self,
        outputs: Vec<RnsBggEncoding>,
    ) -> Result<Vec<RnsBggEncoding>, RnsBggError> {
        outputs
            .into_iter()
            .map(|output| {
                if self
                    .program
                    .wire_is_coefficient_output(output.wire)
                    .map_err(|_| RnsBggError::Binding("invalid output wire metadata"))?
                {
                    let level = self.workload.level(output.wire)?;
                    let target = level
                        .checked_add(1)
                        .ok_or(RnsBggError::Parameters("coefficient output level overflow"))?;
                    let modulus =
                        BigInt::from(self.parameters.level(target)?.modulus().as_ref().clone());
                    Ok(RnsBggEncoding { row: mod_down_encoding(output.row, modulus), ..output })
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
    type Rhs = RhsRef<Int>;
    type Error = RnsBggError;
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        let public_payload = match (&lhs.public_payload, &rhs.public_payload) {
            (Some(lhs), Some(rhs)) => Some(lhs.clone() + rhs.clone()),
            (None, None) => None,
            _ => return Err(RnsBggError::Binding("mixed public/private Add")),
        };
        let output = self.site()?.output;
        let public_payload = self.checked_payload(output, public_payload)?;
        Ok(RnsBggEncoding::new(
            output,
            add_encoding_rows(lhs.row.clone(), rhs.row.clone()),
            public_payload,
        ))
    }
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let private_knowledge = self
            .program
            .wire_knowledge(private.wire)
            .map_err(|_| RnsBggError::Binding("private MulPublic metadata"))?;
        let public_knowledge = self
            .program
            .wire_knowledge(public.wire)
            .map_err(|_| RnsBggError::Binding("public MulPublic metadata"))?;
        if private_knowledge != AttributeKnowledge::PrivatePayload {
            return Err(RnsBggError::Binding("MulPublic first operand must be private"));
        }
        if public_knowledge != AttributeKnowledge::PublicPayload {
            return Err(RnsBggError::Binding("MulPublic second operand must be public"));
        }
        let payload =
            public.public_payload.clone().ok_or(RnsBggError::Binding("public payload missing"))?;
        let action = self.mul_public_action()?;
        let row = add_encoding_rows(
            multiply_encoding_action(public.row.clone(), action),
            (private.row.0.clone() * payload.clone(), private.row.1.clone() * payload.clone()),
        );
        let output = self.site()?.output;
        Ok(RnsBggEncoding::new(output, row, self.checked_payload(output, None)?))
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error> {
        let action = self.rhs_decomposition(rhs)?;
        let output = self.site()?.output;
        let target_level = self.workload.level(output)?;
        let target_modulus =
            BigInt::from(self.parameters.level(target_level)?.modulus().as_ref().clone());
        Ok(RnsBggEncoding::new(
            output,
            mod_down_encoding(multiply_encoding_action(input.row.clone(), action), target_modulus),
            self.checked_payload(output, None)?,
        ))
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        let count = IntExpr::constant(table.values().len());
        let level = self.workload.level(input.wire)?;
        let actions = self.lut_actions(count.clone(), level)?;
        // The workload records helper provenance on the LUT output wire.  A
        // first LUT therefore has no entry for its input wire yet; subsequent
        // LUTs inherit the output's original helper origin.
        let origin = self.workload.lut_helper_origin(self.site()?.output)?;
        let helper = self.lut_helper(count.clone(), origin)?;
        let output = self.site()?.output;
        let layout =
            ExpLayout { stride_log2: 0, width_log2: table.values().len().trailing_zeros() };
        let sigmas = layout
            .automorphism_indices(self.parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let terms = mxx_dsl::parallel(count, |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let (d_sigma, e_sigma) = actions.at(index.clone());
            let mask = helper.at(index);
            let routed_input = (
                input.row.0.clone().ring_automorphism(sigma.clone()),
                input.row.1.clone().ring_automorphism(sigma),
            );
            Ok(add_encoding_rows(
                multiply_encoding_action(mask, e_sigma),
                multiply_encoding_action(routed_input, d_sigma),
            ))
        })
        .map_err(graph)?;
        let mut row =
            crate::utils::balanced_sum_family(terms, |lhs, rhs| add_encoding_rows(lhs, rhs))
                .map_err(graph)?;
        if !self
            .program
            .wire_is_coefficient_output(output)
            .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
        {
            let target = self.workload.level(output)?;
            let modulus = BigInt::from(self.parameters.level(target)?.modulus().as_ref().clone());
            row = mod_down_encoding(row, modulus);
        }
        Ok(RnsBggEncoding::new(output, row, self.checked_payload(output, None)?))
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

impl ProgramLoweringContext for EncodingLowering<'_> {
    fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl GraphParallelLowering for EncodingLowering<'_> {
    type SideOutput = ();
    fn begin_graph_parallel(&mut self) {}
    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
        Ok(())
    }
    fn commit_graph_parallel(
        &mut self,
        outputs: Family<Self::SideOutput>,
    ) -> Result<(), Self::Error> {
        if !outputs.schema().wire_types().is_empty() {
            return Err(RnsBggError::Binding("encoding parallel side output is not empty"));
        }
        Ok(())
    }
}

/// Encryption builder; secrets and plaintexts are stage-local.
pub struct RnsEncryptBuild<'a, B: Backend> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    inputs: Option<crate::program::EncryptionInputs>,
    payload_secret: Option<RnsBggPayloadSecret>,
    mask_secret: Option<RnsBggMaskSecret>,
    plaintexts: BTreeMap<WireId, InputBinding<Mat, Family<Mat>>>,
    preprocess: Option<ProductionId>,
    binding: Option<RnsBggArtifactBinding>,
    refresh_declaration: Option<RefreshDeclaration>,
    catalog: ArtifactCatalog,
    runtime: BTreeMap<String, RuntimeValue<B>>,
    refresh_prf_children: BTreeMap<usize, Box<RnsEncryptBuild<'a, B>>>,
    refresh_prf_binding: Option<([u8; 32], usize, crate::program::ExponentLutProgramId)>,
}

impl<'a, B: Backend> RnsEncryptBuild<'a, B> {
    /// Creates an unbound encryption builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            inputs: None,
            payload_secret: None,
            mask_secret: None,
            plaintexts: BTreeMap::new(),
            preprocess: None,
            binding: None,
            refresh_declaration: None,
            catalog: ArtifactCatalog::default(),
            runtime: BTreeMap::new(),
            refresh_prf_children: BTreeMap::new(),
            refresh_prf_binding: None,
        }
    }

    /// Bind this encryption child to one slot of the prepared PRF Program.
    /// The ordinary encryption checkpoint remains sourced from the outer
    /// Program; this explicit binding records the prepared identity and the
    /// slot projection used when its D0/initial carriers are published.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
        program: &ExponentLutProgram,
    ) -> Result<(), RnsBggError> {
        if slot >= declaration.slots().len() ||
            program.id() != declaration.prepared().program().id()
        {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let inputs = self
            .inputs
            .as_ref()
            .ok_or(RnsBggError::Binding("refresh PRF child inputs are not bound"))?;
        if inputs.inputs() != program.inputs() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if self.refresh_prf_binding.is_some() {
            return Err(RnsBggError::Binding("refresh PRF child is already bound"));
        }
        self.refresh_prf_binding = Some((*declaration.commitment(), slot, program.id()));
        Ok(())
    }

    /// Attach already-bound encryption children for the prepared PRF.
    ///
    /// Child outputs are emitted into the outer graph under a commitment and
    /// slot namespace.  No child checkpoint or second execution is created.
    pub fn bind_refresh_prf_children(
        &mut self,
        declaration: &RefreshDeclaration,
        children: BTreeMap<usize, RnsEncryptBuild<'a, B>>,
    ) -> Result<(), RnsBggError> {
        if children.len() != declaration.slots().len() ||
            children.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        if children.keys().any(|slot| self.refresh_prf_children.contains_key(slot)) {
            return Err(RnsBggError::Binding("refresh PRF child slot is already bound"));
        }
        let expected_program = declaration.prepared().program().id();
        for (slot, child) in &children {
            let binding = child
                .binding
                .as_ref()
                .ok_or(RnsBggError::Binding("refresh PRF child artifacts are not bound"))?;
            if binding.context() != &self.context ||
                binding.parameters() != &self.parameters.parameter_identity() ||
                binding.program() != declaration.source_program() ||
                child.inputs.is_none() ||
                child.payload_secret.is_none() ||
                child.mask_secret.is_none() ||
                child.preprocess.is_none() ||
                child.refresh_prf_binding.map(|(commitment, child_slot, program)| {
                    commitment == *declaration.commitment() &&
                        child_slot == *slot &&
                        program == expected_program
                }) != Some(true)
            {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            if child
                .refresh_declaration
                .as_ref()
                .is_some_and(|bound| bound.commitment() != declaration.commitment())
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
            if declaration.slots().get(*slot).is_none() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
        }
        self.refresh_prf_children =
            children.into_iter().map(|(slot, child)| (slot, Box::new(child))).collect();
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }

    fn append_refresh_prf_child(
        child: &RnsEncryptBuild<'a, B>,
        declaration: &RefreshDeclaration,
        slot: usize,
        mut graph: DslContext,
    ) -> Result<DslContext, ApiError> {
        let inputs = child
            .inputs
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child inputs are not bound"))?;
        let production = child
            .preprocess
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child preprocess is not bound"))?;
        let payload = child
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child payload is not bound"))?;
        let mask = child
            .mask_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child mask is not bound"))?;
        let ring = child.parameters.ring(0).map_err(ApiError::scheme)?;
        let columns = child.parameters.component_columns(0).map_err(ApiError::scheme)?;
        let commitment = *declaration.commitment();
        for (wire, declaration_input) in inputs.inputs() {
            let key = RnsBggPublicKey {
                wire: *wire,
                matrix: (
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.0"),
                            (1, columns),
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.1"),
                            (1, columns),
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.2"),
                            (1, columns),
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.3"),
                            (1, columns),
                        ),
                    ),
                ),
            };
            let value = child
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("refresh PRF child plaintext is not bound"))?;
            let encode = |key: RnsBggPublicKey, plaintext: Mat| {
                if matches!(declaration_input.knowledge(), AttributeKnowledge::PublicPayload) {
                    child.encode_public(&key, 0, mask, plaintext)
                } else {
                    child.encode_private(&key, 0, payload, mask, plaintext)
                }
            };
            graph = match value {
                InputBinding::Scalar(value) => graph
                    .artifact_output(
                        refresh_prf_artifact_name(commitment, slot, &format!("input-{wire:?}")),
                        encode(key, value.clone()).map_err(ApiError::scheme)?,
                    )
                    .map_err(ApiError::scheme)?,
                InputBinding::Family(values) => {
                    let family = mxx_dsl::parallel(values.count().clone(), |index| {
                        let member_key = key.clone();
                        encode(member_key, values.at(index)).map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    let rows = family.field(|value| value.row.clone()).map_err(ApiError::scheme)?;
                    let payloads = if matches!(
                        declaration_input.knowledge(),
                        AttributeKnowledge::PublicPayload
                    ) {
                        Some(
                            family
                                .field(|value| {
                                    value.public_payload.clone().expect("public payload metadata")
                                })
                                .map_err(ApiError::scheme)?,
                        )
                    } else {
                        None
                    };
                    graph
                        .artifact_output(
                            refresh_prf_artifact_name(commitment, slot, &format!("input-{wire:?}")),
                            RnsBggEncoding::new(*wire, rows, payloads),
                        )
                        .map_err(ApiError::scheme)?
                }
            };
        }
        Ok(graph)
    }

    /// Samples a payload secret.
    pub fn sample_payload_secret(&self) -> Result<RnsBggPayloadSecret, RnsBggError> {
        Ok(RnsBggPayloadSecret { row: secret_row(self.parameters)? })
    }
    /// Samples a mask secret.
    pub fn sample_mask_secret(&self) -> Result<RnsBggMaskSecret, RnsBggError> {
        Ok(RnsBggMaskSecret { row: secret_row(self.parameters)? })
    }

    /// Binds the immutable refresh declaration used to publish decoder and
    /// per-label helper artifacts during encryption.
    pub fn bind_refresh_declaration(
        &mut self,
        declaration: &RefreshDeclaration,
    ) -> Result<(), RnsBggError> {
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }
    fn encode_private(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_private_inner(self.parameters, key, payload, mask, plaintext, level)
    }
    fn encode_public(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_public_inner(self.parameters, key, mask, plaintext, level)
    }
    /// Encrypt one preprocess-provided public LUT family with the bound mask.
    fn encrypt_lut_helper(
        &self,
        mask: &RnsBggMaskSecret,
        public_data: &RnsBggLutPublicData,
        level: usize,
    ) -> Result<RnsBggLutHelper, RnsBggError> {
        let mask = reduce_secret(self.parameters, &mask.row, level)?;
        let count = public_data
            .count()
            .evaluate(&ParamEnv::default())
            .map_err(|_| RnsBggError::Lut)?
            .to_usize()
            .ok_or(RnsBggError::Lut)?;
        let layout = ExpLayout { stride_log2: 0, width_log2: count.trailing_zeros() };
        let sigmas = layout
            .automorphism_indices(self.parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let scale = self.parameters.scalar(level, self.parameters.scale(level)?.clone())?;
        let gadget = self.parameters.state_gadget(level)?;
        let columns = self.parameters.component_columns(level)?;
        mxx_dsl::parallel(public_data.count().clone(), |index| {
            let matrix = public_data.at(index.clone());
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let correction = (
                scale.clone() * mask.0.clone().ring_automorphism(sigma.clone()) * gadget.clone(),
                scale.clone() * mask.1.clone().ring_automorphism(sigma) * gadget.clone(),
            );
            let error =
                fresh_error(self.parameters, level, columns).map_err(|_| DslError::Schema)?;
            Ok((
                mask.0.clone() * matrix.0.0 + mask.1.clone() * matrix.1.0 - correction.0 + error.0,
                mask.0.clone() * matrix.0.1 + mask.1.clone() * matrix.1.1 - correction.1 + error.1,
            ))
        })
        .map_err(graph)
    }

    fn refresh_decoder(
        &self,
        declaration: &RefreshDeclaration,
        mask: &RnsBggMaskSecret,
        slot: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        let p_level = refresh_slot_level(self.parameters, declaration, slot)?;
        let metadata = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let p_ring = mxx_dsl::Ring::new(
            metadata.p.clone(),
            self.parameters.level(p_level)?.ring_dimension() as usize,
        );
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let commitment = *declaration.commitment();
        let production =
            self.preprocess.clone().ok_or(RnsBggError::Binding("missing preprocess"))?;
        let matrix = |role: &str| -> Result<RnsBggMatrix, RnsBggError> {
            Ok((
                (
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.0", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                    ),
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.1", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                    ),
                ),
                (
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.2", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                    ),
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.3", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                    ),
                ),
            ))
        };
        let sum = matrix("a-sum")?;
        let anchor = matrix("a-star")?;
        let h = p_ring.polynomial([IntExpr::constant(&metadata.p / &metadata.q_t)]);
        let delta = (
            (sum.0.0 - h.clone() * anchor.0.0, sum.0.1 - h.clone() * anchor.0.1),
            (sum.1.0 - h.clone() * anchor.1.0, sum.1.1 - h * anchor.1.1),
        );
        let modulus = BigInt::from(metadata.p.clone());
        let secret = checked_modulus_convert_row(mask.row.clone(), modulus)?;
        let affine_offset = refresh_affine_offset_row(self.parameters, declaration, slot, p_level)?;
        let decoder = (
            secret.0.clone() * delta.0.0 + secret.1.clone() * delta.1.0 - affine_offset.0,
            secret.0 * delta.0.1 + secret.1 * delta.1.1 - affine_offset.1,
        );
        let sigma = crate::utils::decimal_to_real_expr(self.parameters.encoding_error_sigma())
            .map_err(|_| RnsBggError::Parameters("invalid encoding error sigma"))?;
        let error = (
            p_ring.gaussian(
                (1, columns),
                sigma.clone(),
                self.parameters.encoding_error_bound().clone(),
            ),
            p_ring.gaussian((1, columns), sigma, self.parameters.encoding_error_bound().clone()),
        );
        Ok((decoder.0 + error.0, decoder.1 + error.1))
    }
}

fn polynomial_input<P: Poly>(ring: &mxx_dsl::Ring, value: &P) -> Mat {
    ring.polynomial(
        value
            .coeffs_biguints()
            .into_iter()
            .map(|coefficient| IntExpr::constant(BigInt::from(coefficient))),
    )
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

fn refresh_digit_sum(base: usize, digits: usize) -> BigInt {
    let base = BigInt::from(base);
    let mut power = BigInt::from(1u8);
    let mut sum = BigInt::from(0u8);
    for _ in 0..digits {
        sum += power.clone();
        power *= &base;
    }
    sum
}

fn refresh_affine_coefficients(
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<(BigInt, BigInt), RnsBggError> {
    let digits = declaration.digits();
    let mask_digits =
        *digits.mask_counts.get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let base = BigInt::from(digits.base);
    let mask_support = base.pow(mask_digits as u32);
    let mask_offset = -refresh_digit_sum(digits.base, mask_digits) - &mask_support / 2u8;
    let fresh_sum = refresh_digit_sum(digits.base, digits.fresh_count);
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let h = metadata.p.clone() / metadata.q_t.clone();
    let fresh_offset = h * (-(base / 2u8) * fresh_sum);
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

impl<'a, B: Backend, P: Poly> EncryptBuild<B, P> for RnsEncryptBuild<'a, B> {
    type PayloadSecret = RnsBggPayloadSecret;
    type MaskSecret = RnsBggMaskSecret;
    type PreprocessDescriptor = RnsBggArtifactBinding;
    type Descriptor = RnsBggArtifactBinding;

    fn bind_inputs(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &Self::PayloadSecret,
        mask: &Self::MaskSecret,
    ) -> Result<(), ApiError> {
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        for wire in inputs.inputs().keys() {
            let value = lhs.get(wire).ok_or_else(|| ApiError::input("missing encryption input"))?;
            let converted = match value {
                InputBinding::Scalar(value) => InputBinding::Scalar(polynomial_input(&ring, value)),
                InputBinding::Family(values) => InputBinding::Family(
                    Family::pack(
                        values.iter().map(|value| polynomial_input(&ring, value)).collect(),
                    )
                    .map_err(|error| ApiError::input(error.to_string()))?,
                ),
            };
            self.plaintexts.insert(*wire, converted);
        }
        self.inputs = Some(inputs.clone());
        self.payload_secret = Some(payload.clone());
        self.mask_secret = Some(mask.clone());
        Ok(())
    }

    fn bind_artifacts(
        &mut self,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
    ) -> Result<(), ApiError> {
        let binding = preprocess.checkpoint().descriptor().clone();
        if binding.context() != &self.context ||
            binding.parameters() != &self.parameters.parameter_identity()
        {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess binding does not match encryption",
            ));
        }
        let production = preprocess.checkpoint().production().clone();
        if binding.sources().len() != 1 || !binding.sources().contains(&production) {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess production is not a declared source",
            ));
        }
        self.preprocess = Some(production);
        self.binding = Some(binding);
        self.catalog = preprocess.checkpoint().catalog().clone();
        Ok(())
    }

    fn artifact_identity(&self) -> [u8; 32] {
        let mut digest = sha2::Sha256::new();
        digest.update(b"mxx-rns-bgg/encryption/v1");
        digest.update(self.context);
        digest.update(self.parameters.parameter_identity());
        digest.finalize().into()
    }

    fn build_graph(mut self) -> Result<(StageGraph<B>, Self::Descriptor), ApiError> {
        let inputs = self
            .inputs
            .clone()
            .ok_or_else(|| ApiError::input("encryption inputs are not bound"))?;
        let production = self
            .preprocess
            .clone()
            .ok_or_else(|| ApiError::input("preprocess artifacts are not bound"))?;
        let payload = self
            .payload_secret
            .clone()
            .ok_or_else(|| ApiError::input("payload secret is not bound"))?;
        let mask =
            self.mask_secret.clone().ok_or_else(|| ApiError::input("mask secret is not bound"))?;
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-encrypt");
        if let Some(declaration) = self.refresh_declaration.as_ref() {
            for (slot, child) in &self.refresh_prf_children {
                graph = Self::append_refresh_prf_child(child, declaration, *slot, graph)?;
                merge_runtime_inputs(&mut self.runtime, &child.runtime)?;
                self.catalog.merge(child.catalog.clone()).map_err(ApiError::scheme)?;
            }
        }
        if let Some(binding) = self.binding.as_ref() {
            let root_scope = FamilyScope::new(Vec::new(), Vec::new())
                .map_err(|_| ApiError::input("invalid root LUT helper scope"))?;
            for (origin, level, count) in binding.lut_helpers() {
                let helper_ring = self.parameters.ring(*level).map_err(ApiError::scheme)?;
                let columns =
                    self.parameters.component_columns(*level).map_err(ApiError::scheme)?;
                let family_count = IntExpr::constant(*count);
                let ports = (0..4)
                    .map(|port| {
                        helper_ring.family_artifact_input(
                            production.clone(),
                            format!("{}.{port}", lut_helper_artifact_name(*origin, &root_scope)),
                            family_count.clone(),
                            (1, columns),
                        )
                    })
                    .collect::<Vec<_>>();
                let public_data = mxx_dsl::parallel(family_count, |index| {
                    Ok((
                        (ports[0].at(index.clone()), ports[1].at(index.clone())),
                        (ports[2].at(index.clone()), ports[3].at(index)),
                    ))
                })
                .map_err(ApiError::scheme)?;
                let helper = self
                    .encrypt_lut_helper(&mask, &public_data, *level)
                    .map_err(ApiError::scheme)?;
                graph = graph
                    .artifact_output(lut_helper_artifact_name(*origin, &root_scope), helper)
                    .map_err(ApiError::scheme)?;
            }
        }
        for (wire, declaration) in inputs.inputs() {
            let columns = self.parameters.component_columns(0).map_err(ApiError::scheme)?;
            let scalar_key = || RnsBggPublicKey {
                wire: *wire,
                matrix: (
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.0"),
                            (1, columns),
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.1"),
                            (1, columns),
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.2"),
                            (1, columns),
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("public-key/{wire:?}.3"),
                            (1, columns),
                        ),
                    ),
                ),
            };
            let value = self
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("missing encryption plaintext"))?;
            let encode = |key: RnsBggPublicKey, plaintext: Mat| {
                if matches!(declaration.knowledge(), AttributeKnowledge::PublicPayload) {
                    self.encode_public(&key, 0, &mask, plaintext)
                } else {
                    self.encode_private(&key, 0, &payload, &mask, plaintext)
                }
            };
            graph = match value {
                InputBinding::Scalar(value) => graph
                    .artifact_output(
                        format!("input-{wire:?}"),
                        encode(scalar_key(), value.clone()).map_err(ApiError::scheme)?,
                    )
                    .map_err(ApiError::scheme)?,
                InputBinding::Family(values) => {
                    let family_count = values.count().clone();
                    let key_parts = (0..4)
                        .map(|port| {
                            ring.family_artifact_input(
                                production.clone(),
                                format!("public-key/{wire:?}.{port}"),
                                family_count.clone(),
                                (1, columns),
                            )
                        })
                        .collect::<Vec<_>>();
                    let encodings = mxx_dsl::parallel(values.count().clone(), |member| {
                        let key = RnsBggPublicKey {
                            wire: *wire,
                            matrix: (
                                (key_parts[0].at(member.clone()), key_parts[1].at(member.clone())),
                                (key_parts[2].at(member.clone()), key_parts[3].at(member.clone())),
                            ),
                        };
                        encode(key, values.at(member)).map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    let rows =
                        encodings.field(|value| value.row.clone()).map_err(ApiError::scheme)?;
                    let output =
                        if matches!(declaration.knowledge(), AttributeKnowledge::PublicPayload) {
                            let payloads = encodings
                                .field(|value| {
                                    value.public_payload.clone().expect("public payload metadata")
                                })
                                .map_err(ApiError::scheme)?;
                            RnsBggEncoding::new(*wire, rows, Some(payloads))
                        } else {
                            RnsBggEncoding::new(*wire, rows, None)
                        };
                    graph
                        .artifact_output(format!("input-{wire:?}"), output)
                        .map_err(ApiError::scheme)?
                }
            };
        }
        if let Some(declaration) = self.refresh_declaration.clone() {
            for slot in declaration.slots() {
                let decoder = self
                    .refresh_decoder(&declaration, &mask, slot.slot)
                    .map_err(ApiError::scheme)?;
                graph = graph
                    .artifact_output(
                        refresh_artifact_name(*declaration.commitment(), slot.slot, "decoder"),
                        decoder.clone(),
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let mut descriptor =
            self.binding.ok_or_else(|| ApiError::input("artifact binding is not available"))?;
        if let Some(declaration) = &self.refresh_declaration {
            descriptor = descriptor.with_refresh_declaration(declaration);
        }
        Ok((StageGraph::new(graph, self.runtime, self.catalog), descriptor))
    }
}

/// Online builder with no backend, secrets, or plaintexts.
pub struct RnsOnlineBuild<'a> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    evaluator: Option<EncodingLowering<'a>>,
    families: GraphFamilyExecution,
    environment:
        Environment<RnsBggEncoding, RhsRef<Int>, Family<RnsBggEncoding>, Family<RhsRef<Int>>>,
    pending_refresh: Option<(usize, RefreshDeclaration, BTreeMap<usize, RnsOnlineBuild<'a>>)>,
    refreshed_input: Option<(RefreshDeclaration, RnsBggEncoding)>,
    refresh_prf: Option<(usize, crate::program::ExponentLutProgramId)>,
    resumption_source: Option<ProductionId>,
    refreshed_public_key: Option<RnsBggPublicKey>,
}

impl<'a> RnsOnlineBuild<'a> {
    /// Creates an unbound online builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            evaluator: None,
            families: GraphFamilyExecution,
            environment: Environment::new(),
            pending_refresh: None,
            refreshed_input: None,
            refresh_prf: None,
            resumption_source: None,
            refreshed_public_key: None,
        }
    }

    /// Bind a slot-scoped online child to the prepared PRF Program.
    ///
    /// Ordinary artifact binding retains strict source-Program equality. This
    /// explicit child path validates the combined source checkpoints and then
    /// imports only the slot-scoped PRF input ports.
    pub fn bind_refresh_artifacts(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
        preprocess: &StageArtifacts<RnsBggArtifactBinding>,
        encryption: &StageArtifacts<RnsBggArtifactBinding>,
    ) -> Result<(), ApiError> {
        let prepared = declaration.prepared().program();
        let expected_parameters = self.parameters.parameter_identity();
        let preprocess_binding = preprocess.checkpoint().descriptor();
        let encryption_binding = encryption.checkpoint().descriptor();
        if slot >= declaration.slots().len() || declaration.output_schema().is_empty() {
            return Err(ApiError::input("invalid refresh PRF slot or schema"));
        }
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.parameters() != &expected_parameters ||
                binding.program() != declaration.source_program() ||
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
            !encryption_binding.sources().contains(preprocess.checkpoint().production())
        {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh encryption source is not preprocess production",
            ));
        }
        let preprocess_production = preprocess.checkpoint().production().clone();
        let encryption_production = encryption.checkpoint().production().clone();
        let mut catalog = preprocess.checkpoint().catalog().clone();
        catalog.merge(encryption.checkpoint().catalog().clone()).map_err(ApiError::scheme)?;
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(prepared).map_err(ApiError::scheme)?);
        // The prepared PRF encryption stage publishes its coefficient rows at
        // the raw source (q) level (`append_refresh_prf_child` uses level 0).
        // Keep this child projection at that same level; the refresh route
        // applies the per-slot p-level action and performs q→p explicitly.
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let columns = self.parameters.component_columns(0).map_err(ApiError::scheme)?;
        let commitment = *declaration.commitment();
        self.environment = Environment::new();
        for (wire, input) in prepared.inputs() {
            let public = matches!(input.knowledge(), AttributeKnowledge::PublicPayload);
            let scalar = || {
                RnsBggEncoding::new(
                    *wire,
                    (
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.0"),
                            ),
                            (1, columns),
                        ),
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.1"),
                            ),
                            (1, columns),
                        ),
                    ),
                    public.then(|| {
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.public"),
                            ),
                            (1, 1),
                        )
                    }),
                )
            };
            let binding = match input.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(scalar()),
                InputBinding::Family(count) => {
                    let rows = (0..2)
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
                            )
                        })
                        .collect::<Vec<_>>();
                    let payload = public.then(|| {
                        ring.family_artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.public"),
                            ),
                            count.clone(),
                            (1, 1),
                        )
                    });
                    InputBinding::Family(
                        mxx_dsl::parallel(count.clone(), |index| {
                            Ok(RnsBggEncoding::new(
                                *wire,
                                (rows[0].at(index.clone()), rows[1].at(index.clone())),
                                payload.as_ref().map(|value| value.at(index)),
                            ))
                        })
                        .map_err(ApiError::scheme)?,
                    )
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        for (rhs, input) in prepared.rhs_inputs() {
            let binding = match input {
                InputBinding::Scalar(()) => InputBinding::Scalar(RhsRef { id: *rhs, index: None }),
                InputBinding::Family(count) => InputBinding::Family(
                    mxx_dsl::parallel(count.clone(), |index| {
                        Ok(RhsRef { id: *rhs, index: Some(index) })
                    })
                    .map_err(ApiError::scheme)?,
                ),
            };
            self.environment
                .bind_rhs(*rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(prepared.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            catalog,
            current_site: None,
        });
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
        if slot >= declaration.slots().len() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("PRF artifacts are not bound"))?;
        if declaration.prepared().program().id() != evaluator.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let declared_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(_, wire)| wire)
            .collect::<Vec<_>>();
        let program_outputs = evaluator
            .program
            .outputs()
            .iter()
            .filter_map(|output| match output {
                Operand::Scalar(wire) => Some(*wire),
                Operand::Member { .. } => None,
            })
            .collect::<Vec<_>>();
        if declared_outputs.iter().any(|wire| !program_outputs.contains(wire)) {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        if declared_outputs.iter().any(|wire| {
            evaluator
                .program
                .wire_is_coefficient_output(*wire)
                .map(|coefficient| !coefficient)
                .unwrap_or(true)
        }) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidSourceOutput));
        }
        let slot_modulus = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?
            .p
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        if !self.parameters.levels().iter().any(|level| level.modulus().as_ref() == &slot_modulus) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        if declaration.prepared().setup_identity().program != evaluator.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        self.refresh_prf = Some((slot, evaluator.program.id()));
        Ok(())
    }

    /// Bind a checkpointed refresh as a checked input of a new Program.
    ///
    /// Only lazy artifact handles are created. The checkpoint's row and
    /// matching target public-key ports are imported from its production;
    /// caller code cannot substitute an unrelated carrier.
    pub fn bind_refreshed_input(
        &mut self,
        program: &ExponentLutProgram,
        input_wire: WireId,
        declaration: &RefreshDeclaration,
        artifacts: &StageArtifacts<RnsBggArtifactBinding>,
    ) -> Result<(), ApiError> {
        let checkpoint = artifacts.checkpoint();
        let descriptor = checkpoint.descriptor();
        if descriptor.context() != &self.context ||
            descriptor.program() != declaration.source_program() ||
            descriptor.refresh_commitment() != Some(declaration.commitment())
        {
            return Err(ApiError::compatibility(
                "refreshed input checkpoint does not match its declaration",
            ));
        }
        if descriptor.sources().is_empty() {
            return Err(ApiError::compatibility(
                "refreshed input checkpoint has no dependency sources",
            ));
        }
        if input_wire == declaration.source_output() {
            return Err(ApiError::input("resumption input must use the new Program wire"));
        }
        if declaration.output_schema().is_empty() {
            return Err(ApiError::input("refreshed input schema is empty"));
        }
        // A resumed Program is evaluated against the target chain, whose
        // level zero is the refresh output level.  The old source Program's
        // parameter digest intentionally does not identify this builder.
        let resumed_level = self.parameters.level(0).map_err(ApiError::scheme)?;
        let resumed_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let resumed_columns = self.parameters.component_columns(0).map_err(ApiError::scheme)?;
        let expected_port = resumed_ring.matrix_type((1, resumed_columns));
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
        if program.inputs().len() != 1 ||
            !program.inputs().contains_key(&input_wire) ||
            program.outputs() != [Operand::Scalar(input_wire)]
        {
            return Err(ApiError::compatibility(
                "only an identity continuation can resume from a refresh checkpoint without stage artifacts",
            ));
        }
        let declaration_input = program
            .inputs()
            .get(&input_wire)
            .ok_or_else(|| ApiError::input("resumption wire is not a Program input"))?;
        if !matches!(declaration_input.shape(), InputBinding::Scalar(())) {
            return Err(ApiError::input("refreshed input must be scalar"));
        }
        let knowledge = program.wire_knowledge(input_wire).map_err(ApiError::scheme)?;
        let target_ring = resumed_ring;
        let columns = resumed_columns;
        let production = checkpoint.production().clone();
        let manifest = checkpoint
            .catalog()
            .manifests()
            .get(&production)
            .ok_or_else(|| ApiError::compatibility("refresh checkpoint manifest is missing"))?;
        for source in descriptor.sources() {
            if !checkpoint.catalog().manifests().contains_key(source) {
                return Err(ApiError::compatibility(
                    "refresh checkpoint is missing a declared dependency source",
                ));
            }
        }
        let checkpoint_name =
            |role: &str| refresh_checkpoint_artifact_name(*declaration.commitment(), role);
        let matrix_type = |columns| {
            ArtifactType::Matrix(ConcreteMatrixType {
                modulus: BigInt::from(resumed_level.modulus().as_ref().clone()),
                ring_dimension: resumed_level.ring_dimension() as usize,
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
        require_artifact(&checkpoint_name("row.0"), matrix_type(columns))?;
        require_artifact(&checkpoint_name("row.1"), matrix_type(columns))?;
        for port in 0..4 {
            require_artifact(&checkpoint_name(&format!("target.{port}")), matrix_type(columns))?;
        }
        let row = (
            target_ring.artifact_input(production.clone(), checkpoint_name("row.0"), (1, columns)),
            target_ring.artifact_input(production.clone(), checkpoint_name("row.1"), (1, columns)),
        );
        let public_key = RnsBggPublicKey::new(
            input_wire,
            (
                (
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.0"),
                        (1, columns),
                    ),
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.1"),
                        (1, columns),
                    ),
                ),
                (
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.2"),
                        (1, columns),
                    ),
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.3"),
                        (1, columns),
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
                    input_wire,
                    row.clone(),
                    public_payload.clone(),
                )),
            )
            .map_err(|error| ApiError::input(error.to_string()))?;
        for (rhs, rhs_declaration) in program.rhs_inputs() {
            let binding = match rhs_declaration {
                InputBinding::Scalar(()) => InputBinding::Scalar(RhsRef { id: *rhs, index: None }),
                InputBinding::Family(count) => InputBinding::Family(
                    mxx_dsl::parallel(count.clone(), |index| {
                        Ok(RhsRef { id: *rhs, index: Some(index) })
                    })
                    .map_err(ApiError::scheme)?,
                ),
            };
            environment
                .bind_rhs(*rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let mut sources = descriptor.sources().to_vec();
        sources.sort();
        sources.dedup();
        let preprocess = sources.first().cloned().unwrap_or_else(|| production.clone());
        let encryption = sources.get(1).cloned().unwrap_or_else(|| preprocess.clone());
        let catalog = checkpoint.catalog().clone();
        if !catalog.manifests().contains_key(&production) {
            return Err(ApiError::compatibility(
                "refresh checkpoint production is not in its catalog",
            ));
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess,
            encryption,
            catalog,
            current_site: None,
        });
        self.environment = environment;
        self.refreshed_public_key = Some(public_key);
        self.resumption_source = Some(checkpoint.production().clone());
        self.refreshed_input =
            Some((declaration.clone(), RnsBggEncoding::new(input_wire, row, public_payload)));
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
        prf_children: BTreeMap<usize, RnsOnlineBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() {
            return Err(RnsBggError::Binding("refresh was already bound"));
        }
        if prf_children.len() != declaration.slots().len() ||
            prf_children.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        for (slot, child) in &prf_children {
            if child.refresh_prf.map(|(bound_slot, program)| {
                bound_slot == *slot && program == declaration.prepared().program().id()
            }) != Some(true)
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
        }
        self.pending_refresh = Some((output_index, declaration.clone(), prf_children));
        Ok(())
    }

    fn lower_refresh_children(
        declaration: &RefreshDeclaration,
        children: &mut BTreeMap<usize, RnsOnlineBuild<'a>>,
    ) -> Result<BTreeMap<PrfLabel, BTreeMap<usize, RnsBggEncoding>>, RnsBggError> {
        let labels = declaration.prf_labels();
        let ordered_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(label, wire)| (label.clone(), wire))
            .collect::<Vec<_>>();
        if ordered_outputs.len() != labels.len() ||
            ordered_outputs.iter().map(|(label, _)| label).ne(labels.iter())
        {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let mut outputs = BTreeMap::<PrfLabel, BTreeMap<usize, RnsBggEncoding>>::new();
        for (slot, child) in children.iter_mut() {
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
            if values.len() != ordered_outputs.len() {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            for (index, ((label, wire), value)) in ordered_outputs.iter().zip(values).enumerate() {
                let refresh_label = declaration
                    .labels()
                    .label(index)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                if matches!(refresh_label, RefreshPrfLabel::Mask { slot: owner, .. } if owner != *slot)
                {
                    continue;
                }
                if value.wire != *wire {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                outputs.entry(label.clone()).or_default().insert(*slot, value);
            }
        }
        for label in labels {
            let channels =
                outputs.get(label).ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let index = declaration
                .prf_labels()
                .iter()
                .position(|expected| expected == label)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let refresh_label = declaration
                .labels()
                .label(index)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let expected_slots = match refresh_label {
                RefreshPrfLabel::Mask { slot, .. } => vec![slot],
                RefreshPrfLabel::FreshError { .. } => (0..children.len()).collect(),
            };
            if channels.len() != expected_slots.len() ||
                channels.keys().copied().ne(expected_slots.iter().copied())
            {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
        }
        Ok(outputs)
    }

    /// Compose a refresh against already-lowered outer outputs and return the
    /// actual symbolic carrier before graph finalization.
    pub fn compose_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        mut children: BTreeMap<usize, RnsOnlineBuild<'a>>,
        outputs: &mut [RnsBggEncoding],
    ) -> Result<RnsBggEncoding, RnsBggError> {
        if children.len() != declaration.slots().len() ||
            children.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        for (slot, child) in &children {
            if child.refresh_prf.map(|(bound_slot, program)| {
                bound_slot == *slot && program == declaration.prepared().program().id()
            }) != Some(true)
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
            .ok_or(RnsBggError::Binding("online artifacts are not bound"))?;
        let mut refresh = RnsBggRefresh::new(evaluator, inputs, declaration)?;
        let refreshed = apply_refresh(declaration, &mut refresh, |callback_declaration, labels| {
            if callback_declaration.prf_labels() != labels {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            Self::lower_refresh_children(callback_declaration, &mut children)
        })
        .map_err(|error| match error {
            RefreshApplicationError::Refresh(error) => error,
            RefreshApplicationError::OutputMismatch => {
                RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
            }
        })?;
        let refreshed_target = evaluator.refresh_public_target(declaration, source.wire)?;
        for child in children.values() {
            if let Some(child_evaluator) = child.evaluator.as_ref() {
                evaluator
                    .catalog
                    .merge(child_evaluator.catalog.clone())
                    .map_err(|_| RnsBggError::Binding("refresh child catalog conflict"))?;
            }
        }
        outputs[output_index] = refreshed.clone();
        self.refreshed_public_key = Some(refreshed_target);
        Ok(refreshed)
    }

    /// Build a checkpointable graph for a composed refresh result.  The graph
    /// publishes the refreshed row and its matching target public key under
    /// one declaration namespace and returns the existing artifact binding.
    pub fn build_refresh_checkpoint_graph<B: Backend>(
        &mut self,
        declaration: &RefreshDeclaration,
        refreshed: RnsBggEncoding,
        target: RnsBggPublicKey,
    ) -> Result<(StageGraph<B>, RnsBggArtifactBinding), ApiError> {
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or_else(|| ApiError::input("online artifacts are not bound"))?;
        if refreshed.wire != declaration.source_output() ||
            target.wire != declaration.source_output()
        {
            return Err(ApiError::input("refresh checkpoint carrier wire mismatch"));
        }
        let target_ring =
            self.parameters.ring(declaration.target_level()).map_err(ApiError::scheme)?;
        let target_columns = self
            .parameters
            .component_columns(declaration.target_level())
            .map_err(ApiError::scheme)?;
        let expected_row = target_ring.matrix_type((1, target_columns));
        if refreshed.row.0.matrix_type() != &expected_row ||
            refreshed.row.1.matrix_type() != &expected_row
        {
            return Err(ApiError::input("refresh checkpoint row schema mismatch"));
        }
        let public = evaluator.program.wire_knowledge(refreshed.wire).map_err(ApiError::scheme)?;
        if matches!(public, AttributeKnowledge::PublicPayload) != refreshed.public_payload.is_some()
        {
            return Err(ApiError::input("refresh checkpoint payload metadata mismatch"));
        }
        if let Some(payload) = &refreshed.public_payload {
            if payload.matrix_type() != &target_ring.matrix_type((1, 1)) {
                return Err(ApiError::input("refresh checkpoint payload schema mismatch"));
            }
        }
        if target.schema().wire_types() != declaration.output_schema() {
            return Err(ApiError::input("refresh checkpoint target schema mismatch"));
        }
        let mut sources = vec![evaluator.preprocess.clone(), evaluator.encryption.clone()];
        sources.sort();
        sources.dedup();
        let descriptor = RnsBggArtifactBinding::new(
            self.context,
            evaluator.program.id(),
            self.parameters.parameter_identity(),
            sources,
        )
        .map_err(ApiError::scheme)?
        .with_refresh_declaration(declaration);
        let mut graph = DslContext::new("rns-bgg-refresh-checkpoint");
        graph = graph
            .artifact_output(
                refresh_checkpoint_artifact_name(*declaration.commitment(), "row"),
                refreshed,
            )
            .map_err(ApiError::scheme)?;
        graph = graph
            .artifact_output(
                refresh_checkpoint_artifact_name(*declaration.commitment(), "target"),
                target,
            )
            .map_err(ApiError::scheme)?;
        Ok((
            StageGraph::new(
                graph.build().map_err(ApiError::scheme)?,
                BTreeMap::new(),
                evaluator.catalog.clone(),
            ),
            descriptor,
        ))
    }

    fn refresh_inputs_from_output(
        &self,
        declaration: &RefreshDeclaration,
        source: &RnsBggEncoding,
    ) -> Result<BTreeMap<usize, RnsBggEncoding>, RnsBggError> {
        if source.wire != declaration.source_output() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("online artifacts are not bound"))?;
        let level = evaluator.workload.level(source.wire)?;
        let ring = self.parameters.ring(level)?;
        let columns = self.parameters.component_columns(level)?;
        let expected_row = ring.matrix_type((1, columns));
        if source.row.0.matrix_type() != &expected_row ||
            source.row.1.matrix_type() != &expected_row
        {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let expected_public =
            matches!(self.program_knowledge(source.wire)?, AttributeKnowledge::PublicPayload);
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
                        source.wire,
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
            .ok_or(RnsBggError::Binding("online artifacts are not bound"))?
            .program
            .wire_knowledge(wire)
            .map_err(|_| RnsBggError::Binding("wire metadata"))
    }

    fn lowering_state_inner(
        &mut self,
    ) -> Result<
        (
            &mut EncodingLowering<'a>,
            &mut GraphFamilyExecution,
            &mut Environment<
                RnsBggEncoding,
                RhsRef<Int>,
                Family<RnsBggEncoding>,
                Family<RhsRef<Int>>,
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
                .ok_or_else(|| ApiError::input("online artifacts are not bound"))?;
            let input_declaration = evaluator
                .program
                .inputs()
                .get(&input.wire)
                .ok_or_else(|| ApiError::input("refreshed wire is not a Program input"))?;
            if !matches!(input_declaration.shape(), InputBinding::Scalar(())) {
                return Err(ApiError::input("refreshed input must be scalar"));
            }
            self.environment.wires.insert(input.wire, InputBinding::Scalar(input));
        }
        Ok((
            self.evaluator
                .as_mut()
                .ok_or_else(|| ApiError::input("online artifacts are not bound"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
}

impl<'a, B: Backend> OnlineBuild<B> for RnsOnlineBuild<'a> {
    type Evaluator = EncodingLowering<'a>;
    type Families = GraphFamilyExecution;
    type PreprocessDescriptor = RnsBggArtifactBinding;
    type EncryptDescriptor = RnsBggArtifactBinding;

    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                RnsBggEncoding,
                RhsRef<Int>,
                Family<RnsBggEncoding>,
                Family<RhsRef<Int>>,
            >,
        ),
        ApiError,
    > {
        self.lowering_state_inner()
    }

    fn bind_artifacts(
        &mut self,
        program: &ExponentLutProgram,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
        encryption: &StageArtifacts<Self::EncryptDescriptor>,
    ) -> Result<(), ApiError> {
        let preprocess_binding = preprocess.checkpoint().descriptor();
        let encryption_binding = encryption.checkpoint().descriptor();
        let expected_parameters = self.parameters.parameter_identity();
        let preprocess_production = preprocess.checkpoint().production().clone();
        let encryption_production = encryption.checkpoint().production().clone();
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.parameters() != &expected_parameters ||
                binding.program() != program.id()
            {
                return Err(ApiError::compatibility(
                    "RNS-BGG artifact binding does not match online Program",
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
        let mut catalog = preprocess.checkpoint().catalog().clone();
        catalog.merge(encryption.checkpoint().catalog().clone()).map_err(ApiError::scheme)?;
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?);
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        for (wire, declaration) in program.inputs() {
            let columns = self.parameters.component_columns(0).map_err(ApiError::scheme)?;
            let public = matches!(declaration.knowledge(), AttributeKnowledge::PublicPayload);
            let bind_scalar = |suffix: &str| {
                let row = (
                    ring.artifact_input(
                        encryption_production.clone(),
                        format!("input-{wire:?}.{suffix}0"),
                        (1, columns),
                    ),
                    ring.artifact_input(
                        encryption_production.clone(),
                        format!("input-{wire:?}.{suffix}1"),
                        (1, columns),
                    ),
                );
                let payload = public.then(|| {
                    ring.artifact_input(
                        encryption_production.clone(),
                        format!("input-{wire:?}.{suffix}public"),
                        (1, 1),
                    )
                });
                RnsBggEncoding::new(*wire, row, payload)
            };
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(bind_scalar("")),
                InputBinding::Family(count) => {
                    let rows = (0..2)
                        .map(|port| {
                            ring.family_artifact_input(
                                encryption_production.clone(),
                                format!("input-{wire:?}.{port}"),
                                count.clone(),
                                (1, columns),
                            )
                        })
                        .collect::<Vec<_>>();
                    let payload = public.then(|| {
                        ring.family_artifact_input(
                            encryption_production.clone(),
                            format!("input-{wire:?}.public"),
                            count.clone(),
                            (1, 1),
                        )
                    });
                    let family = mxx_dsl::parallel(count.clone(), |index| {
                        Ok(RnsBggEncoding::new(
                            *wire,
                            (rows[0].at(index.clone()), rows[1].at(index.clone())),
                            payload.as_ref().map(|payload| payload.at(index)),
                        ))
                    })
                    .map_err(ApiError::scheme)?;
                    InputBinding::Family(family)
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        for (rhs, declaration) in program.rhs_inputs() {
            match declaration {
                InputBinding::Scalar(()) => self
                    .environment
                    .bind_rhs(*rhs, InputBinding::Scalar(RhsRef { id: *rhs, index: None }))
                    .map_err(|error| ApiError::input(error.to_string()))?,
                InputBinding::Family(count) => {
                    let family = mxx_dsl::parallel(count.clone(), |index| {
                        Ok(RhsRef { id: *rhs, index: Some(index) })
                    })
                    .map_err(ApiError::scheme)?;
                    self.environment
                        .bind_rhs(*rhs, InputBinding::Family(family))
                        .map_err(|error| ApiError::input(error.to_string()))?;
                }
            }
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            catalog,
            current_site: None,
        });
        Ok(())
    }

    fn build_graph(mut self, outputs: Vec<RnsBggEncoding>) -> Result<StageGraph<B>, ApiError> {
        let mut outputs = outputs;
        let pending = self.pending_refresh.take();
        if let Some((index, declaration, mut children)) = pending {
            self.compose_refresh(index, &declaration, std::mem::take(&mut children), &mut outputs)
                .map_err(ApiError::scheme)?;
        }
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online artifacts are not bound"))?;
        let outputs = evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-online");
        for (index, output) in outputs.into_iter().enumerate() {
            graph = graph.output(format!("output-{index}"), output).map_err(ApiError::scheme)?;
        }
        Ok(StageGraph::new(
            graph.build().map_err(ApiError::scheme)?,
            BTreeMap::new(),
            evaluator.catalog.clone(),
        ))
    }
}

/// Encoding-side implementation of the common one-shot refresh contract.
impl NoiseRefresh for RnsBggRefresh<'_, EncodingLowering<'_>> {
    type PrfOutput = BTreeMap<usize, RnsBggEncoding>;
    type Output = RnsBggEncoding;
    type Error = RnsBggError;
    fn required_prf_labels(&self) -> Result<Vec<PrfLabel>, Self::Error> {
        Ok(self.declaration().prf_labels().to_vec())
    }
    fn finish_noise_refresh(
        &mut self,
        outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
    ) -> Result<Self::Output, Self::Error> {
        let labels = self.required_prf_labels()?;
        if outputs.len() != labels.len() || labels.iter().any(|label| !outputs.contains_key(label))
        {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let declaration = self.declaration().clone();
        let inputs = self.inputs().clone();
        let slots = self.slots().to_vec();
        let digits = self.digits().clone();
        let label_count = self.labels().len();
        let target_level = self.target_level();
        let source =
            inputs.get(&0).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?.clone();
        let lowerer = self.lowerer();
        for input in inputs.values() {
            if input.wire != declaration.source_output() {
                return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
            }
        }
        lowerer.validate_refresh_declaration(&declaration)?;
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
        for slot in &slots {
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
        for slot in &slots {
            channels.push((slot.slot, None::<RnsBggRow>));
        }
        let ordered_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(label, wire)| (label.clone(), wire))
            .collect::<Vec<_>>();
        if ordered_outputs.len() != labels.len() ||
            ordered_outputs.iter().map(|(label, _)| label).ne(labels.iter())
        {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
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
                if value.wire != expected_wire {
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
                    lowerer.refresh_action_for_label(&declaration, Some(index), slot, "target")?;
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
            let target = lowerer.refresh_action(&declaration, slot, "target")?;
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
            let producer_modulus = slot_metadata
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            row = mod_down_encoding(row, BigInt::from(producer_modulus));
            let decoder = lowerer.refresh_decoder(&declaration, slot)?;
            row = add_encoding_rows(row, (-decoder.0, -decoder.1));
            let slot_modulus = slot_metadata.q_t.clone();
            let slot_modulus = slot_modulus
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            row = mod_down_lift_encoding(
                row,
                BigInt::from(slot_modulus),
                BigInt::from(target_modulus.clone()),
            );
            channel_rows.push(row);
            channel_weights.push(lowerer.refresh_crt_weight(&declaration, slot)?);
        }
        let values = Family::pack(channel_rows).map_err(graph)?;
        let coefficients = Family::pack(channel_weights).map_err(graph)?;
        let result = recompose_encoding(values, coefficients)?;
        if result.0.matrix_type() != &expected_port || result.1.matrix_type() != &expected_port {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        let public_payload = match source.public_payload.clone() {
            Some(payload) => Some(checked_modulus_convert(
                payload,
                BigInt::from(target_parameters.modulus().as_ref().clone()),
            )?),
            None => None,
        };
        Ok(RnsBggEncoding::new(source.wire, result, public_payload))
    }
}

fn mod_down_lift_encoding(
    row: RnsBggRow,
    channel_modulus: BigInt,
    target_modulus: BigInt,
) -> RnsBggRow {
    map_row(row, |entry| {
        entry.block_mod_switch(channel_modulus.clone(), 1).centered_extend(target_modulus.clone())
    })
}

fn reduce_anchor_channel(
    a_star: RnsBggMatrix,
    channel_modulus: BigInt,
    target_modulus: BigInt,
) -> Result<RnsBggMatrix, RnsBggError> {
    let reduced = checked_modulus_convert_matrix(a_star, channel_modulus)?;
    Ok(map_matrix(reduced, |entry| entry.centered_extend(target_modulus.clone())))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn encryption_lut_helper_produces_one_lazy_family_member() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let one = ring.polynomial([IntExpr::constant(BigInt::from(1u8))]);
        let public_data =
            Family::pack(vec![(((one.clone(), one.clone()), (one.clone(), one.clone())))]).unwrap();
        let builder = RnsEncryptBuild::<mxx_runtime::backend::poly::CpuDcrtBackend>::new(
            &parameters,
            [0; 32],
        );
        let mask = builder.sample_mask_secret().unwrap();
        let helper = builder.encrypt_lut_helper(&mask, &public_data, 0).unwrap();
        assert_eq!(helper.count(), &IntExpr::constant(1));
    }

    #[test]
    fn lut_coefficients_depend_on_values_not_only_width() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let constant = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 0],
        )
        .unwrap();
        let identity = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 1],
        )
        .unwrap();
        assert_ne!(constant.flatten(), identity.flatten());
        let declared_layout = EncodingLowering::lut_coefficient(
            &parameters,
            0,
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
            let table = LutTable::unary(2, 2, values, representation).unwrap();
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
                program,
                workload,
                preprocess: production.clone(),
                encryption: production,
                catalog: ArtifactCatalog::default(),
                current_site: Some(LoweringSite {
                    output,
                    indices: Vec::new(),
                    ranges: Vec::new(),
                }),
            };
            let ring = parameters.ring(0).unwrap();
            let columns = parameters.component_columns(0).unwrap();
            let value = ring.input("lut-lowering-input", (1, columns));
            let input = RnsBggEncoding::new(input, (value.clone(), value), None);
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
    fn lut_artifacts_use_next_level_columns_and_scoped_flattening() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = LutTable::unary(
            2,
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(ExpLayout {
                stride_log2: 0,
                width_log2: 1,
            }),
        )
        .unwrap();
        let lut = builder.register_lut(table).unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([11; 32]),
            execution_nonce: [13; 32],
        };
        let lowerer = EncodingLowering {
            parameters: &parameters,
            program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            catalog: ArtifactCatalog::default(),
            current_site: Some(LoweringSite {
                output,
                indices: vec![IntExpr::constant(3)],
                ranges: vec![
                    crate::program::FamilyRange::bounded(
                        IntExpr::constant(2),
                        IntExpr::constant(4),
                        IntExpr::constant(8),
                    )
                    .unwrap(),
                ],
            }),
        };

        let branch_count = IntExpr::constant(2);
        let actions = lowerer.lut_actions(branch_count.clone(), 0).unwrap();
        assert_eq!(actions.count(), &branch_count);
        let action = actions.at(Int::constant(0));
        assert_eq!(
            action.0.0.0.matrix_type().columns,
            IntExpr::constant(parameters.component_columns(1).unwrap())
        );
        assert_eq!(
            action.0.0.0.matrix_type().modulus,
            IntExpr::constant(parameters.level(0).unwrap().modulus().as_ref().clone())
        );

        let unscoped = EncodingLowering {
            parameters: lowerer.parameters,
            program: lowerer.program.clone(),
            workload: lowerer.workload.clone(),
            preprocess: lowerer.preprocess.clone(),
            encryption: lowerer.encryption.clone(),
            catalog: ArtifactCatalog::default(),
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let unscoped_action =
            unscoped.lut_actions(IntExpr::constant(2), 0).unwrap().at(Int::constant(0));
        assert_ne!(action.flatten(), unscoped_action.flatten());
        let scoped = lowerer.site().unwrap().family_scope().unwrap();
        let scoped_name = lut_helper_artifact_name(input, &scoped);
        let root_name =
            lut_helper_artifact_name(input, &FamilyScope::new(Vec::new(), Vec::new()).unwrap());
        let alternate_scope = FamilyScope::new(
            vec![
                crate::program::FamilyRange::bounded(
                    IntExpr::constant(3),
                    IntExpr::constant(4),
                    IntExpr::constant(8),
                )
                .unwrap(),
            ],
            vec![IntExpr::constant(4)],
        )
        .unwrap();
        let alternate_name = lut_helper_artifact_name(input, &alternate_scope);
        assert_ne!(scoped_name, root_name);
        assert_ne!(scoped_name, alternate_name);
        assert!(scoped_name.contains("scope/start=Const(2),count=Const(4)"));

        let helper = lowerer.lut_helper(branch_count, input).unwrap();
        assert_eq!(helper.count(), &IntExpr::constant(2));
        assert_eq!(
            helper.at(Int::constant(3)).0.matrix_type().columns,
            parameters.component_columns(0).unwrap().into()
        );
        let unscoped_helper = unscoped.lut_helper(IntExpr::constant(2), input).unwrap();
        assert_ne!(
            helper.at(Int::constant(0)).flatten(),
            unscoped_helper.at(Int::constant(0)).flatten()
        );
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

        let mut scalar_builder = crate::program::ExponentLutProgramBuilder::new();
        let scalar_input = scalar_builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let scalar_rhs = scalar_builder.rhs_input().unwrap();
        let scalar_output = scalar_builder
            .mul_private(Operand::Scalar(scalar_input), Operand::Scalar(scalar_rhs))
            .unwrap();
        let scalar_program =
            Arc::new(scalar_builder.finish(vec![Operand::Scalar(scalar_output)]).unwrap());
        let scalar_workload =
            Arc::new(RnsBggDerivedWorkload::from_program(&scalar_program).unwrap());
        let scalar_lowerer = EncodingLowering {
            parameters: &parameters,
            program: scalar_program,
            workload: scalar_workload,
            preprocess: production.clone(),
            encryption: production.clone(),
            catalog: ArtifactCatalog::default(),
            current_site: Some(LoweringSite {
                output: scalar_output,
                indices: Vec::new(),
                ranges: Vec::new(),
            }),
        };
        let scalar_action = scalar_lowerer
            .rhs_decomposition(&RhsRef { id: scalar_rhs, index: None::<Int> })
            .unwrap();
        let source_ring = parameters.ring(0).unwrap();
        assert_eq!(scalar_action.0.0.matrix_type(), &source_ring.matrix_type((1, compact_columns)));
        assert_eq!(
            scalar_action.0.0.max_coefficient_bound(),
            &IntExpr::constant(BigInt::from(1u8) << parameters.level(1).unwrap().base_bits())
        );
        let scalar_name = format!("rhs/{scalar_rhs:?}");
        let scalar_graph = DslContext::new("rhs-scalar-consumer-schema")
            .artifact_output(scalar_name.clone(), scalar_action)
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
        let family_workload =
            Arc::new(RnsBggDerivedWorkload::from_program(&family_program).unwrap());
        let family_lowerer = EncodingLowering {
            parameters: &parameters,
            program: family_program,
            workload: family_workload,
            preprocess: production.clone(),
            encryption: production,
            catalog: ArtifactCatalog::default(),
            current_site: Some(LoweringSite {
                output: family_output,
                indices: vec![IntExpr::constant(1)],
                ranges: vec![crate::program::FamilyRange::full(IntExpr::constant(2)).unwrap()],
            }),
        };
        let family_action = family_lowerer
            .rhs_decomposition(&RhsRef { id: family_rhs, index: Some(Int::constant(1)) })
            .unwrap();
        assert_eq!(family_action.0.0.matrix_type(), &source_ring.matrix_type((1, compact_columns)));
        assert_eq!(
            family_action.0.0.max_coefficient_bound(),
            &IntExpr::constant(BigInt::from(1u8) << parameters.level(1).unwrap().base_bits())
        );
        let family_name = format!("rhs/{family_rhs:?}/family");
        let family_members = mxx_dsl::parallel(IntExpr::constant(2), |index| {
            let member = family_lowerer
                .rhs_decomposition(&RhsRef { id: family_rhs, index: Some(index) })
                .map_err(|_| DslError::Schema)?;
            Ok(member)
        })
        .unwrap();
        let family_graph = DslContext::new("rhs-family-consumer-schema")
            .artifact_output(family_name.clone(), family_members)
            .unwrap()
            .build()
            .unwrap();
        assert!(family_graph.graph.outputs().contains_key(&format!("{family_name}.0")));
    }

    #[test]
    fn mul_public_uses_only_second_payload_and_rejects_public_first_operand() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let private_wire = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let output = builder
            .mul_public(Operand::Scalar(private_wire), Operand::Scalar(public_wire))
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
            program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            catalog: ArtifactCatalog::default(),
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let ring = parameters.ring(0).unwrap();
        let columns = parameters.component_columns(0).unwrap();
        let private_row = ring.input("mul-public-private-row", (1, columns));
        let public_row = ring.input("mul-public-public-row", (1, columns));
        let public_payload = ring.input("mul-public-payload", (1, 1));
        let private = RnsBggEncoding::new(private_wire, (private_row.clone(), private_row), None);
        let public = RnsBggEncoding::new(
            public_wire,
            (public_row.clone(), public_row),
            Some(public_payload),
        );
        let result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &private,
            &public,
        )
        .unwrap();
        assert!(result.public_payload.is_none());
        let public_with_other_payload = RnsBggEncoding::new(
            public_wire,
            (public.row.0.clone(), public.row.1.clone()),
            Some(ring.input("mul-public-other-payload", (1, 1))),
        );
        let other_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &private,
            &public_with_other_payload,
        )
        .unwrap();
        assert_ne!(result.row.flatten(), other_result.row.flatten());

        let public_as_first =
            RnsBggEncoding::new(public_wire, (private.row.0.clone(), private.row.1.clone()), None);
        assert!(matches!(
            <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
                &mut lowerer,
                &public_as_first,
                &public,
            ),
            Err(RnsBggError::Binding("MulPublic first operand must be private"))
        ));
    }

    #[test]
    fn refresh_two_prime_encoding_channels_mod_down_then_lift_to_combined_ring() {
        let source = mxx_dsl::Ring::new(BigInt::from(105u8), 4);
        let value = source.input("refresh-source", (1, 1));
        let row = (value.clone(), value);
        let producer_row = mod_down_encoding(row, BigInt::from(21u8));
        assert_eq!(producer_row.0.matrix_type().modulus, IntExpr::constant(BigInt::from(21u8)));
        let raw_reduction =
            checked_modulus_convert(producer_row.0.clone(), BigInt::from(21u8)).unwrap();
        assert_eq!(producer_row.0.flatten(), raw_reduction.flatten());
        let lifted = mod_down_lift_encoding(producer_row, BigInt::from(5u8), BigInt::from(15u8));
        assert_eq!(lifted.0.matrix_type().modulus, IntExpr::constant(BigInt::from(15u8)));
        assert_eq!(lifted.1.matrix_type().modulus, IntExpr::constant(BigInt::from(15u8)));
    }

    #[test]
    fn refresh_target_uses_only_anchor_and_lifts_from_qt_to_combined_ring() {
        let ring = mxx_dsl::Ring::new(BigInt::from(105u8), 4);
        let zero = ring.zero((1, 1));
        let zero_matrix = ((zero.clone(), zero.clone()), (zero.clone(), zero.clone()));
        let anchor =
            ((ring.input("refresh-a-star", (1, 1)), zero.clone()), (zero.clone(), zero.clone()));
        let zero_target =
            reduce_anchor_channel(zero_matrix, BigInt::from(5u8), BigInt::from(15u8)).unwrap();
        let anchored_target =
            reduce_anchor_channel(anchor, BigInt::from(5u8), BigInt::from(15u8)).unwrap();
        assert_ne!(zero_target.flatten(), anchored_target.flatten());
        assert_eq!(
            anchored_target.0.0.matrix_type().modulus,
            IntExpr::constant(BigInt::from(15u8))
        );
    }

    #[test]
    fn refresh_decoder_uses_p_ring_target_without_an_extra_moddown() {
        let p_ring = mxx_dsl::Ring::new(BigInt::from(21u8), 4);
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
        assert_eq!(p_target.0.0.matrix_type().modulus, IntExpr::constant(BigInt::from(21u8)));
    }

    #[test]
    fn refresh_affine_offsets_cover_components_coefficients_and_b2_d1() {
        assert_eq!(refresh_digit_sum(2, 1), BigInt::from(1u8));
        assert_eq!(refresh_digit_sum(2, 2), BigInt::from(3u8));
        let ring = mxx_dsl::Ring::new(BigInt::from(21u8), 4);
        let columns = 2;
        let mask = refresh_affine_route(&ring, columns, 0, 0, BigInt::from(-2i8)).unwrap();
        let fresh = refresh_affine_route(&ring, columns, 3, 1, BigInt::from(-5i8)).unwrap();
        let offset = add_encoding_rows(mask, fresh);
        assert_eq!(offset.0.matrix_type().modulus, IntExpr::constant(BigInt::from(21u8)));
        assert_ne!(offset.flatten(), ring.zero((1, columns)).flatten());
        let secret0 = ring.input("refresh-secret-0", (1, columns));
        let secret1 = ring.input("refresh-secret-1", (1, columns));
        let decoder0 = secret0.clone() - offset.0.clone();
        let decoder1 = secret1.clone() - offset.0.clone();
        assert_ne!(decoder0.flatten(), decoder1.flatten());
    }
}
