//! Tall rotation encoding preprocessing and artifact definitions.

use crate::tall_encoding::{TallCompileError, rotate_family, same_matrix_type};
use mxx_dsl::{Bytes, DslContext, Family, HashTag, Mat, Ring};
use mxx_gadgets::{
    Poly,
    circuit::{GateParamSource, PolyCircuit, PolyGateType, SlotTransferSpec, SubCircuitParamValue},
};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
};
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

/// Direction in which a provisioned cyclic rotation pair is applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TallRotationDirection {
    /// Applies `P_r`, so output row `j` receives input row `j-r`.
    Forward,
    /// Applies `P_r^-1`, so output row `j` receives input row `j+r`.
    Backward,
}

/// Deterministic artifact names for one rotation pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallRotationEncodingArtifactNames {
    /// Forward public matrix artifact.
    pub a_forward: String,
    /// Backward public matrix artifact.
    pub a_backward: String,
    /// Forward permutation-encoding family artifact.
    pub c_forward: String,
    /// Backward permutation-encoding family artifact.
    pub c_backward: String,
}

impl TallRotationEncodingArtifactNames {
    /// Builds names scoped by the complete normalized rotation identity.
    pub fn for_key(key: TallRotationEncodingKey) -> Self {
        let prefix = format!("bgg_tall_rotation_n{}_r{}", key.num_slots, key.offset);
        Self {
            a_forward: format!("{prefix}_a_forward"),
            a_backward: format!("{prefix}_a_backward"),
            c_forward: format!("{prefix}_c_forward"),
            c_backward: format!("{prefix}_c_backward"),
        }
    }
}

/// Runtime artifact descriptor for rotation pairs of one slot count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallRotationEncodingArtifacts {
    /// Production containing the public matrices and encoding families.
    pub production_id: ProductionId,
    /// Exact slot count served by the producing compiler.
    pub slot_count: u32,
}

/// Four graph wires forming one directly provisioned rotation pair.
#[derive(Clone)]
pub struct TallRotationEncodingWires {
    /// Pair identity.
    pub key: TallRotationEncodingKey,
    /// Public matrix for `P_r`.
    pub a_forward: Mat,
    /// Public matrix for `P_r^-1`.
    pub a_backward: Mat,
    /// Encoding rows for `P_r`.
    pub c_forward: Family<Mat>,
    /// Encoding rows for `P_r^-1`.
    pub c_backward: Family<Mat>,
}

/// Preprocessing wires for every offset handled by one compiler instance.
#[derive(Clone, Default)]
pub struct TallRotationEncodingPreprocessingWires {
    /// Directly provisioned nonidentity pairs.
    pub rotations: BTreeMap<TallRotationEncodingKey, TallRotationEncodingWires>,
}

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
    /// Builds all directly requested nonidentity rotation pairs.
    pub fn preprocess(
        &self,
        hash_key: Bytes,
        secret: Mat,
        slot_secrets: Family<Mat>,
        offsets: &[u32],
    ) -> Result<TallRotationEncodingPreprocessingWires, TallCompileError> {
        self.validate_layout(&secret, &slot_secrets)?;
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
        let ring = self.ring();
        let transformed = slot_secrets.parallel_map({
            let secret = secret.clone();
            move |_, transform| secret.clone() * transform
        })?;
        let gadget = ring.gadget(self.secret_size, self.gadget_base.clone(), self.digit_count);
        let mut rotations = BTreeMap::new();
        for key in keys {
            let inverse_offset = (key.num_slots - key.offset) % key.num_slots;
            let a_forward = self.tall_rotation_public_matrix(&hash_key, key, false);
            let a_backward = self.tall_rotation_public_matrix(&hash_key, key, true);
            let shifted_forward = rotate_family(
                &transformed,
                usize::try_from(key.offset).map_err(|_| TallCompileError::InvalidRotationLayout)?,
                self.slot_count,
            )?;
            let shifted_backward = rotate_family(
                &transformed,
                usize::try_from(inverse_offset)
                    .map_err(|_| TallCompileError::InvalidRotationLayout)?,
                self.slot_count,
            )?;
            let c_forward = transformed.clone().parallel_zip(shifted_forward, {
                let ring = ring.clone();
                let a_forward = a_forward.clone();
                let gadget = gadget.clone();
                let sigma = self.error_sigma.clone();
                let columns = self.gadget_columns();
                move |_, current, shifted| {
                    current * a_forward.clone() - shifted * gadget.clone() +
                        ring.gaussian((1, columns), sigma.clone())
                }
            })?;
            let c_backward = transformed.clone().parallel_zip(shifted_backward, {
                let ring = ring.clone();
                let a_backward = a_backward.clone();
                let gadget = gadget.clone();
                let sigma = self.error_sigma.clone();
                let columns = self.gadget_columns();
                move |_, current, shifted| {
                    current * a_backward.clone() - shifted * gadget.clone() +
                        ring.gaussian((1, columns), sigma.clone())
                }
            })?;
            rotations.insert(
                key,
                TallRotationEncodingWires { key, a_forward, a_backward, c_forward, c_backward },
            );
        }
        Ok(TallRotationEncodingPreprocessingWires { rotations })
    }

    /// Exports every public rotation matrix and row family.
    pub fn export_preprocessing(
        &self,
        mut context: DslContext,
        wires: TallRotationEncodingPreprocessingWires,
    ) -> Result<DslContext, TallCompileError> {
        for (key, rotation) in wires.rotations {
            let names = TallRotationEncodingArtifactNames::for_key(key);
            context = context
                .public_output(names.a_forward, rotation.a_forward)?
                .public_output(names.a_backward, rotation.a_backward)?
                .public_family_output(names.c_forward, rotation.c_forward)?
                .public_family_output(names.c_backward, rotation.c_backward)?;
        }
        Ok(context)
    }

    /// Imports one directly provisioned rotation pair.
    pub fn import_artifacts(
        &self,
        artifacts: &TallRotationEncodingArtifacts,
        offset: u32,
    ) -> Result<Option<TallRotationEncodingWires>, TallCompileError> {
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
        Ok(Some(TallRotationEncodingWires {
            key,
            a_forward: ring.artifact_input(
                artifacts.production_id.clone(),
                names.a_forward,
                (self.secret_size, self.gadget_columns()),
                ArtifactConfidentiality::Public,
            ),
            a_backward: ring.artifact_input(
                artifacts.production_id.clone(),
                names.a_backward,
                (self.secret_size, self.gadget_columns()),
                ArtifactConfidentiality::Public,
            ),
            c_forward: ring.family_artifact_input(
                artifacts.production_id.clone(),
                names.c_forward,
                self.slot_count,
                (1, self.gadget_columns()),
                ArtifactConfidentiality::Public,
            ),
            c_backward: ring.family_artifact_input(
                artifacts.production_id.clone(),
                names.c_backward,
                self.slot_count,
                (1, self.gadget_columns()),
                ArtifactConfidentiality::Public,
            ),
        }))
    }

    fn validate_layout(
        &self,
        secret: &Mat,
        slot_secrets: &Family<Mat>,
    ) -> Result<(), TallCompileError> {
        if self.secret_size == 0 ||
            self.slot_count == 0 ||
            self.digit_count == 0 ||
            slot_secrets.count() != &IntExpr::constant(self.slot_count) ||
            !same_matrix_type(
                secret.matrix_type(),
                &self.ring().matrix_type((1, self.secret_size)),
            ) ||
            !same_matrix_type(
                slot_secrets.element_type(),
                &self.ring().matrix_type((self.secret_size, self.secret_size)),
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
