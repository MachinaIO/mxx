//! Fixed linear-transform encodings used by Tall slot operations.

use crate::tall_encoding::{TallCompileError, anchor_matrix_rows, rotate_family, same_matrix_type};
use mxx_dsl::{Bytes, DslContext, Family, HashTag, Mat, Ring};
use mxx_gadgets::{
    Poly,
    circuit::{GateParamSource, PolyCircuit, PolyGateType, SlotTransferSpec, SubCircuitParamValue},
};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
};
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};

/// Stable identity of one directly provisioned rotation pair.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TallRotationEncodingKey {
    /// Number of slots affected by the rotation.
    pub num_slots: u32,
    /// Nonzero offset normalized modulo `num_slots`.
    pub offset: u32,
}

impl TallRotationEncodingKey {
    /// Normalizes a nonempty rotation request. Identity rotations return `None`.
    pub fn normalize(num_slots: u32, offset: u32) -> Result<Option<Self>, TallCompileError> {
        if num_slots == 0 {
            return Err(TallCompileError::InvalidRotationLayout);
        }
        let offset = offset % num_slots;
        Ok((offset != 0).then_some(Self { num_slots, offset }))
    }
}

/// Deterministic artifact names for one rotation pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallRotationEncodingArtifactNames {
    /// Forward public matrix artifact.
    pub a_forward: String,
    /// Backward public matrix artifact.
    pub a_backward: String,
}

impl TallRotationEncodingArtifactNames {
    /// Builds names scoped by the complete normalized rotation identity.
    pub fn for_key(key: TallRotationEncodingKey) -> Self {
        let prefix = format!("bgg_tall_rotation_n{}_r{}", key.num_slots, key.offset);
        Self {
            a_forward: format!("{prefix}_a_forward"),
            a_backward: format!("{prefix}_a_backward"),
        }
    }
}

/// Runtime artifact descriptor for public rotation matrices of one slot count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallRotationEncodingArtifacts {
    /// Production containing the public matrices.
    pub production_id: ProductionId,
    /// Exact slot count served by the producing compiler.
    pub slot_count: u32,
}

/// Public matrices for one fixed linear Tall transform `L * X * R`.
#[derive(Clone)]
pub struct TallLinearTransformPublicWires {
    /// Public matrix applied on the left.
    pub left_matrix: Mat,
    /// Public matrix applied on the right.
    pub right_matrix: Mat,
}

/// Online fixed-matrix encodings for one linear Tall transform `L * X * R`.
#[derive(Clone)]
pub struct TallLinearTransformEncodingWires {
    /// Public matrix applied on the left.
    pub left_matrix: Mat,
    /// Public matrix applied on the right.
    pub right_matrix: Mat,
    /// Secret-dependent encoding rows for the left matrix.
    pub left_rows: Family<Mat>,
    /// Secret-dependent encoding rows for the right matrix.
    pub right_rows: Family<Mat>,
}

/// Public preprocessing wires for every offset handled by one compiler instance.
#[derive(Clone, Default)]
pub struct TallRotationEncodingPreprocessingWires {
    /// Directly provisioned nonidentity pairs.
    pub rotations: BTreeMap<TallRotationEncodingKey, TallLinearTransformPublicWires>,
}

pub const TALL_ANCHOR_REDUCE_MATRIX_ARTIFACT: &str = "bgg_tall_anchor_reduce_matrix";

/// Compiler for tall rotation encoding preprocessing and artifact import.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallRotationEncodingCompiler {
    /// Ciphertext modulus.
    pub modulus: IntExpr,
    /// Polynomial ring dimension.
    pub ring_dimension: IntExpr,
    /// Secret row width.
    pub secret_size: usize,
    /// Exact number of slots served by this compiler.
    pub slot_count: usize,
    /// Gadget radix.
    pub gadget_base: IntExpr,
    /// Number of gadget digits.
    pub digit_count: usize,
    /// Gaussian width for tall rotation encoding errors.
    pub error_sigma: RealExpr,
    /// Explicit hard coefficient cutoff for tall rotation encoding errors.
    pub error_max_coefficient_bound: IntExpr,
}

/// Finds every normalized nonidentity rotation pair required by a circuit.
///
/// Registered and summed sub-circuits are traversed with their concrete
/// parameter bindings, so preprocessing can provision exactly one artifact
/// pair per `(num_slots, normalized_offset)` identity.
pub fn required_tall_rotation_encodings<P: Poly>(
    circuit: &PolyCircuit<P>,
) -> Result<BTreeSet<TallRotationEncodingKey>, TallCompileError> {
    let mut required = BTreeSet::new();
    collect_tall_rotation_encodings(circuit, &[], &mut required)?;
    Ok(required)
}

/// Returns the graph's sole anchor-reduction specification, if present.
pub fn required_tall_anchor_reduce_encoding<P: Poly>(
    circuit: &PolyCircuit<P>,
) -> Result<Option<(u32, Vec<BigUint>)>, TallCompileError> {
    let mut required = None;
    collect_tall_anchor_reduce_encoding(circuit, &[], &mut required)?;
    Ok(required)
}

fn collect_tall_anchor_reduce_encoding<P: Poly>(
    circuit: &PolyCircuit<P>,
    bindings: &[SubCircuitParamValue],
    required: &mut Option<(u32, Vec<BigUint>)>,
) -> Result<(), TallCompileError> {
    let mut visited_calls = BTreeSet::new();
    let mut visited_summed_calls = BTreeSet::new();
    for (_, gate) in circuit.gates_in_id_order() {
        match &gate.gate_type {
            PolyGateType::SlotTransfer { src_slots } => {
                let spec = match src_slots {
                    GateParamSource::Const(spec) => spec,
                    GateParamSource::Param(parameter) => match bindings.get(*parameter) {
                        Some(SubCircuitParamValue::SlotTransfer(spec)) => spec,
                        _ => continue,
                    },
                };
                if let SlotTransferSpec::AnchorReduce { num_blocks, lane_scalars } = spec {
                    let candidate = (*num_blocks, lane_scalars.clone());
                    if required.as_ref().is_some_and(|current| current != &candidate) {
                        return Err(TallCompileError::ConflictingAnchorReduceEncoding);
                    }
                    *required = Some(candidate);
                }
            }
            PolyGateType::SubCircuitOutput { call_id, .. } if visited_calls.insert(*call_id) => {
                let info = circuit.sub_circuit_call_info(*call_id);
                collect_tall_anchor_reduce_encoding(
                    circuit.registered_sub_circuit_ref(info.sub_circuit_id).as_ref(),
                    info.param_bindings.as_ref(),
                    required,
                )?;
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. }
                if visited_summed_calls.insert(*summed_call_id) =>
            {
                let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                for call_bindings in &info.param_bindings {
                    collect_tall_anchor_reduce_encoding(
                        child.as_ref(),
                        call_bindings.as_ref(),
                        required,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_tall_rotation_encodings<P: Poly>(
    circuit: &PolyCircuit<P>,
    bindings: &[SubCircuitParamValue],
    required: &mut BTreeSet<TallRotationEncodingKey>,
) -> Result<(), TallCompileError> {
    let mut visited_calls = BTreeSet::new();
    let mut visited_summed_calls = BTreeSet::new();
    for (_, gate) in circuit.gates_in_id_order() {
        match &gate.gate_type {
            PolyGateType::SlotTransfer { src_slots } => {
                let spec = match src_slots {
                    GateParamSource::Const(spec) => spec,
                    GateParamSource::Param(parameter) => match bindings.get(*parameter) {
                        Some(SubCircuitParamValue::SlotTransfer(spec)) => spec,
                        _ => {
                            return Err(TallCompileError::InvalidRotationParameter {
                                gate: gate.gate_id.index(),
                            });
                        }
                    },
                };
                if let SlotTransferSpec::Rotation { diagonal, num_slots } = spec &&
                    let Some(key) = TallRotationEncodingKey::normalize(*num_slots, *diagonal)?
                {
                    required.insert(key);
                }
            }
            PolyGateType::SubCircuitOutput { call_id, .. } if visited_calls.insert(*call_id) => {
                let info = circuit.sub_circuit_call_info(*call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                collect_tall_rotation_encodings(
                    child.as_ref(),
                    info.param_bindings.as_ref(),
                    required,
                )?;
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. }
                if visited_summed_calls.insert(*summed_call_id) =>
            {
                let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                for call_bindings in &info.param_bindings {
                    collect_tall_rotation_encodings(
                        child.as_ref(),
                        call_bindings.as_ref(),
                        required,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

impl TallRotationEncodingCompiler {
    /// Computes the one `S * G` family shared by every rotation offset and direction.
    pub fn secret_gadget_rows(
        &self,
        secret_rows: Family<Mat>,
    ) -> Result<Family<Mat>, TallCompileError> {
        self.validate_secret_rows(&secret_rows)?;
        let gadget =
            self.ring().gadget(self.secret_size, self.gadget_base.clone(), self.digit_count);
        Ok(secret_rows.parallel_map(move |_, secret| secret * gadget.clone())?)
    }

    pub fn preprocess_anchor_reduce(
        &self,
        hash_key: Bytes,
        num_blocks: u32,
        lane_scalars: Vec<BigUint>,
    ) -> Result<TallLinearTransformPublicWires, TallCompileError> {
        self.validate_anchor_reduce(num_blocks, &lane_scalars)?;
        let matrix = self.ring().hash_matrix(
            hash_key,
            HashTag::from(b"bgg_tall_anchor_reduce_matrix".as_slice()),
            (self.secret_size, self.gadget_columns()),
        );
        Ok(TallLinearTransformPublicWires { left_matrix: matrix.clone(), right_matrix: matrix })
    }

    pub fn encode_anchor_reduce(
        &self,
        public: &TallLinearTransformPublicWires,
        num_blocks: u32,
        lane_scalars: &[BigUint],
        secret_rows: Family<Mat>,
        secret_gadget_rows: Family<Mat>,
    ) -> Result<TallLinearTransformEncodingWires, TallCompileError> {
        self.validate_anchor_reduce(num_blocks, lane_scalars)?;
        self.validate_secret_rows(&secret_rows)?;
        if secret_gadget_rows.count() != &IntExpr::constant(self.slot_count) {
            return Err(TallCompileError::InvalidAnchorReduceLayout);
        }
        let transformed = anchor_matrix_rows(
            &secret_gadget_rows,
            usize::try_from(num_blocks).map_err(|_| TallCompileError::InvalidAnchorReduceLayout)?,
            lane_scalars,
            &self.ring(),
        )?;
        let ring = self.ring();
        let matrix = public.left_matrix.clone();
        let sigma = self.error_sigma.clone();
        let bound = self.error_max_coefficient_bound.clone();
        let columns = self.gadget_columns();
        let rows = secret_rows.parallel_zip(transformed, move |_, secret, shifted| {
            secret * matrix.clone() - shifted +
                ring.gaussian((1, columns), sigma.clone(), bound.clone())
        })?;
        Ok(TallLinearTransformEncodingWires {
            left_matrix: public.left_matrix.clone(),
            right_matrix: public.right_matrix.clone(),
            left_rows: rows.clone(),
            right_rows: rows,
        })
    }

    pub fn export_anchor_reduce_preprocessing(
        &self,
        context: DslContext,
        public: TallLinearTransformPublicWires,
    ) -> Result<DslContext, TallCompileError> {
        Ok(context.public_output(TALL_ANCHOR_REDUCE_MATRIX_ARTIFACT, public.left_matrix)?)
    }

    pub fn import_anchor_reduce_artifact(
        &self,
        artifacts: &TallRotationEncodingArtifacts,
        num_blocks: u32,
        lane_scalars: Vec<BigUint>,
    ) -> Result<TallLinearTransformPublicWires, TallCompileError> {
        self.validate_anchor_reduce(num_blocks, &lane_scalars)?;
        let matrix = self.ring().artifact_input(
            artifacts.production_id.clone(),
            TALL_ANCHOR_REDUCE_MATRIX_ARTIFACT,
            (self.secret_size, self.gadget_columns()),
            ArtifactConfidentiality::Public,
        );
        Ok(TallLinearTransformPublicWires { left_matrix: matrix.clone(), right_matrix: matrix })
    }

    /// Builds the public matrices for all directly requested nonidentity rotation pairs.
    pub fn preprocess(
        &self,
        hash_key: Bytes,
        offsets: &[u32],
    ) -> Result<TallRotationEncodingPreprocessingWires, TallCompileError> {
        self.validate_public_layout()?;
        let num_slots =
            u32::try_from(self.slot_count).map_err(|_| TallCompileError::InvalidRotationLayout)?;
        let mut keys = BTreeSet::new();
        for offset in offsets {
            if let Some(key) = TallRotationEncodingKey::normalize(num_slots, *offset)? &&
                !keys.insert(key)
            {
                return Err(TallCompileError::DuplicateRotation {
                    num_slots: key.num_slots,
                    offset: key.offset,
                });
            }
        }
        let mut rotations = BTreeMap::new();
        for key in keys {
            let a_forward = self.tall_rotation_public_matrix(&hash_key, key, false);
            let a_backward = self.tall_rotation_public_matrix(&hash_key, key, true);
            rotations.insert(
                key,
                TallLinearTransformPublicWires { left_matrix: a_forward, right_matrix: a_backward },
            );
        }
        Ok(TallRotationEncodingPreprocessingWires { rotations })
    }

    /// Constructs the secret-dependent rotation rows inside the online graph.
    pub fn encode_rotation(
        &self,
        key: TallRotationEncodingKey,
        public: &TallLinearTransformPublicWires,
        secret_rows: Family<Mat>,
    ) -> Result<TallLinearTransformEncodingWires, TallCompileError> {
        self.validate_online_layout(key, public, &secret_rows)?;
        let ring = self.ring();
        let gadget = ring.gadget(self.secret_size, self.gadget_base.clone(), self.digit_count);
        let forward_offset =
            usize::try_from(key.offset).map_err(|_| TallCompileError::InvalidRotationLayout)?;
        let backward_offset = usize::try_from((key.num_slots - key.offset) % key.num_slots)
            .map_err(|_| TallCompileError::InvalidRotationLayout)?;
        let shifted_forward = rotate_family(&secret_rows, forward_offset, self.slot_count)?;
        let shifted_backward = rotate_family(&secret_rows, backward_offset, self.slot_count)?;
        let c_forward = secret_rows.clone().parallel_zip(shifted_forward, {
            let ring = ring.clone();
            let a_forward = public.left_matrix.clone();
            let gadget = gadget.clone();
            let sigma = self.error_sigma.clone();
            let columns = self.gadget_columns();
            move |_, current, shifted| {
                current * a_forward.clone() - shifted * gadget.clone() +
                    ring.gaussian(
                        (1, columns),
                        sigma.clone(),
                        self.error_max_coefficient_bound.clone(),
                    )
            }
        })?;
        let c_backward = secret_rows.parallel_zip(shifted_backward, {
            let ring = ring.clone();
            let a_backward = public.right_matrix.clone();
            let gadget = gadget.clone();
            let sigma = self.error_sigma.clone();
            let columns = self.gadget_columns();
            move |_, current, shifted| {
                current * a_backward.clone() - shifted * gadget.clone() +
                    ring.gaussian(
                        (1, columns),
                        sigma.clone(),
                        self.error_max_coefficient_bound.clone(),
                    )
            }
        })?;
        Ok(TallLinearTransformEncodingWires {
            left_matrix: public.left_matrix.clone(),
            right_matrix: public.right_matrix.clone(),
            left_rows: c_forward,
            right_rows: c_backward,
        })
    }

    /// Exports every public rotation matrix.
    pub fn export_preprocessing(
        &self,
        mut context: DslContext,
        wires: TallRotationEncodingPreprocessingWires,
    ) -> Result<DslContext, TallCompileError> {
        for (key, rotation) in wires.rotations {
            let names = TallRotationEncodingArtifactNames::for_key(key);
            context = context
                .public_output(names.a_forward, rotation.left_matrix)?
                .public_output(names.a_backward, rotation.right_matrix)?;
        }
        Ok(context)
    }

    /// Imports one public rotation pair for online encoding.
    pub fn import_artifacts(
        &self,
        artifacts: &TallRotationEncodingArtifacts,
        offset: u32,
    ) -> Result<Option<(TallRotationEncodingKey, TallLinearTransformPublicWires)>, TallCompileError>
    {
        let expected_slots =
            u32::try_from(self.slot_count).map_err(|_| TallCompileError::InvalidRotationLayout)?;
        if artifacts.slot_count != expected_slots {
            return Err(TallCompileError::InvalidRotationLayout);
        }
        let Some(key) = TallRotationEncodingKey::normalize(expected_slots, offset)? else {
            return Ok(None);
        };
        let names = TallRotationEncodingArtifactNames::for_key(key);
        let ring = self.ring();
        Ok(Some((
            key,
            TallLinearTransformPublicWires {
                left_matrix: ring.artifact_input(
                    artifacts.production_id.clone(),
                    names.a_forward,
                    (self.secret_size, self.gadget_columns()),
                    ArtifactConfidentiality::Public,
                ),
                right_matrix: ring.artifact_input(
                    artifacts.production_id.clone(),
                    names.a_backward,
                    (self.secret_size, self.gadget_columns()),
                    ArtifactConfidentiality::Public,
                ),
            },
        )))
    }

    fn validate_public_layout(&self) -> Result<(), TallCompileError> {
        if self.secret_size == 0 || self.slot_count == 0 || self.digit_count == 0 {
            return Err(TallCompileError::InvalidLayout);
        }
        Ok(())
    }

    fn validate_anchor_reduce(
        &self,
        num_blocks: u32,
        lane_scalars: &[BigUint],
    ) -> Result<(), TallCompileError> {
        self.validate_public_layout()?;
        let blocks =
            usize::try_from(num_blocks).map_err(|_| TallCompileError::InvalidAnchorReduceLayout)?;
        if blocks == 0 ||
            lane_scalars.is_empty() ||
            blocks.checked_mul(lane_scalars.len()) != Some(self.slot_count)
        {
            return Err(TallCompileError::InvalidAnchorReduceLayout);
        }
        Ok(())
    }

    fn validate_online_layout(
        &self,
        key: TallRotationEncodingKey,
        public: &TallLinearTransformPublicWires,
        secret_rows: &Family<Mat>,
    ) -> Result<(), TallCompileError> {
        self.validate_secret_rows(secret_rows)?;
        if key.num_slots !=
            u32::try_from(self.slot_count)
                .map_err(|_| TallCompileError::InvalidRotationLayout)? ||
            !same_matrix_type(
                public.left_matrix.matrix_type(),
                &self.ring().matrix_type((self.secret_size, self.gadget_columns())),
            ) ||
            !same_matrix_type(
                public.right_matrix.matrix_type(),
                &self.ring().matrix_type((self.secret_size, self.gadget_columns())),
            )
        {
            return Err(TallCompileError::InvalidLayout);
        }
        Ok(())
    }

    fn validate_secret_rows(&self, secret_rows: &Family<Mat>) -> Result<(), TallCompileError> {
        self.validate_public_layout()?;
        if secret_rows.count() != &IntExpr::constant(self.slot_count) ||
            !same_matrix_type(
                secret_rows.element_type(),
                &self.ring().matrix_type((1, self.secret_size)),
            )
        {
            return Err(TallCompileError::InvalidLayout);
        }
        Ok(())
    }

    pub(crate) fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }

    fn gadget_columns(&self) -> usize {
        self.secret_size
            .checked_mul(self.digit_count)
            .expect("rotation gadget column count overflow")
    }

    fn tall_rotation_public_matrix(
        &self,
        hash_key: &Bytes,
        key: TallRotationEncodingKey,
        inverse: bool,
    ) -> Mat {
        self.ring().hash_matrix(
            hash_key.clone(),
            tall_rotation_public_key_tag(key, inverse),
            (self.secret_size, self.gadget_columns()),
        )
    }
}

pub(crate) fn tall_rotation_public_key_tag(key: TallRotationEncodingKey, inverse: bool) -> HashTag {
    HashTag::from(
        format!(
            "bgg_tall_rotation_n{}_r{}_{}",
            key.num_slots,
            key.offset,
            if inverse { "backward" } else { "forward" }
        )
        .into_bytes(),
    )
}
