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
        GraphParallelLowering, InputBinding, LoweringSite, LutId, LutTable, ProgramLoweringContext,
        RhsId, RhsRef, WireId, lower_program,
    },
};
use mxx_backends::{RuntimeValue, poly::Poly};
use mxx_dsl::{Bytes, DslContext, DslError, Family, GraphValue, GraphValueSchema, Int, Mat};
use mxx_ir_core::{
    IntExpr, ValueHandle,
    artifact::{ArtifactAvailability, Manifest, ProductionId},
    node::ConstantMatrix,
    types::WireType,
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

mod lowering;
mod preprocess;
mod refresh;
mod sampling;

pub(super) use self::{lowering::*, refresh::*, sampling::*};

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

/// Structural public-key lowerer. All prepared values are compact graph
/// handles; the maps do not own expanded matrices or concrete family members.
pub struct PublicKeyLowering<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
    pub(crate) program: Arc<ExponentLutProgram>,
    pub(crate) workload: Arc<RnsBggDerivedWorkload>,
    /// The Static payload secret; Dynamic preprocessing has none.
    pub(crate) payload_secret: Option<RnsBggPayloadSecret>,
    pub(crate) hash_key: Bytes,
    /// Setup context and, for a prepared PRF child, its refresh declaration
    /// and slot: the public namespace of Dynamic digit-companion keys.
    pub(crate) context: [u8; 32],
    pub(crate) refresh_prf_namespace: Option<([u8; 32], usize)>,
    /// RHS plaintexts, read from graph inputs; a family is read member by
    /// member.
    pub(crate) rhs: BTreeMap<RhsId, InputBinding<Mat, super::PlaintextFamily>>,
    /// Setup-fixed RHS ciphertexts.  These are compact matrices; their signed
    /// decompositions are constructed at the consuming level only.
    pub(crate) rhs_material: BTreeMap<RhsId, InputBinding<RnsBggMatrix, Family<RnsBggMatrix>>>,
    /// Compact LUT switching ciphertexts, keyed by the complete table/layout
    /// identity.  The family member is the concrete automorphism branch.
    pub(crate) lut_material:
        BTreeMap<(usize, usize, crate::program::LutId, ExpLayout), Family<RnsBggMatrix>>,
    /// Refresh targets and compact actions, keyed by declaration commitment
    /// and CRT slot.  Keeping the commitment in the key prevents material
    /// from one refresh declaration being reused by another invocation.
    pub(crate) refresh_sums: BTreeMap<([u8; 32], usize), RnsBggMatrix>,
    pub(crate) refresh_stars: BTreeMap<([u8; 32], usize), RnsBggMatrix>,
    pub(crate) refresh_targets: BTreeMap<([u8; 32], usize), RnsBggMatrix>,
    pub(crate) refresh_label_targets: BTreeMap<([u8; 32], usize, usize), RnsBggMatrix>,
    /// Dynamic refresh keys encryption consumes to form `A_sum`, keyed by
    /// declaration commitment, slot, and label: the source key (`None`) and
    /// each label's child-PRF output key.
    pub(crate) refresh_keys: BTreeMap<([u8; 32], usize, Option<usize>), RnsBggMatrix>,
    pub(crate) current_site: Option<LoweringSite>,
    /// Open parallel bodies, outermost first: each one's instance count and
    /// member index, and the RHS family members it prepared.
    parallel_bodies: Vec<(IntExpr, IntExpr, Vec<(RhsId, RnsBggMatrix)>)>,
    /// The RHS families of each finished body, awaiting its loop's commit.
    finished_bodies: Vec<Vec<RhsId>>,
    /// RHS families whose members the loop reading them prepares (see
    /// `streamed_rhs_member`), with the family that loop returned.
    streamed_rhs: BTreeMap<RhsId, Option<Family<RnsBggMatrix>>>,
    /// RHS families read from the materials prepared at binding.
    prepared_rhs_reads: std::collections::BTreeSet<RhsId>,
}

/// Preprocessing stage adapter using the common stage lifecycle.
pub struct RnsPreprocessBuild<'a> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    hash_key_commitment: [u8; 32],
    evaluator: Option<PublicKeyLowering<'a>>,
    families: super::encoding::OnlineEvalFamilies<'a>,
    environment:
        Environment<RnsBggPublicKey, RhsRef<Int>, PreprocessWireFamily, Family<RhsRef<Int>>>,
    runtime: BTreeMap<String, RuntimeValue>,
    refresh_declaration: Option<crate::noise_refresh::refresh::RefreshDeclaration>,
    pending_refresh: Option<(usize, RnsBggPublicKey)>,
    refresh_prf_binding: Option<([u8; 32], usize, ExponentLutProgramId, [u8; 32])>,
    refresh_prf_children: BTreeMap<usize, PublicKeyLowering<'a>>,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn public_key_carrier_is_wire_and_matrix() {
        let ring = crate::test_support::explicit_ring(&[17], 4);
        let key = RnsBggPublicKey::<Mat>::new(
            WireId(3),
            ring.polynomial([IntExpr::constant(BigInt::from(0u8))]),
        );
        assert_eq!(key.wire(), WireId(3));
        assert_eq!(key.matrix().matrix_type(), &ring.matrix_type((1, 1)));
    }
}
