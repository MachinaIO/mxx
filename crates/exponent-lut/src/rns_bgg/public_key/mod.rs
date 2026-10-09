//! Public-key preparation and preprocessing for RNS-BGG.
#![warn(missing_docs)]

use super::*;
use crate::{
    api::{ApiError, PlaintextInputs, PreprocessBuild, StageGraph, apply_decoder},
    decoder::decoding::{Decoder, DecoderError},
    program::{
        AttributeKnowledge, Environment, ExpLayout, ExponentLUTProgramLowering, ExponentLutProgram,
        GraphParallelLowering, InputBinding, LoweringSite, LutId, LutTable, Operand,
        ProgramLoweringContext, RhsId, RhsRef, WireId, lower_program,
    },
};
use mxx_backends::{RuntimeValue, poly::Poly};
use mxx_dsl::{
    Bytes, DslContext, DslError, Family, GraphValue, GraphValueSchema, Int, Mat, Preimage, Trapdoor,
};
use mxx_ir_core::{
    IntExpr, ValueHandle,
    artifact::{ArtifactAvailability, ProductionId},
    types::WireType,
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

mod decoder;
mod lowering;
mod preprocess;
mod sampling;

use self::decoder::sample_unmask_trapdoor;
pub(super) use self::{lowering::*, sampling::*};
#[cfg(test)]
use mxx_ir_core::node::ConstantMatrix;

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
    /// Setup context and, for a decoder smudge child, its decoder declaration:
    /// the public namespace of Dynamic digit-companion keys.
    pub(crate) context: [u8; 32],
    pub(crate) decoder_smudge: Option<Arc<DecoderDeclaration>>,
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
    /// The unmasking of the composed decoder: its declaration commitment, the
    /// preimage `K` of the summed key, and the public `B` it is a preimage
    /// under.
    pub(crate) unmask: Option<([u8; 32], Preimage, Mat)>,
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
    decoder_declaration: Option<crate::decoder::decoding::DecoderDeclaration>,
    /// The lowered smudge Program child of the applied decoder, whose public
    /// material this stage publishes under the decoder namespace.
    decoder_smudge_child: Option<PublicKeyLowering<'a>>,
    /// The preimage coefficient bound of the applied decoder's unmask key.
    unmask_bound: Option<BigUint>,
    /// For a refresh child: the namespace its hash key is derived under (see
    /// `rns_bgg::refresh::child_hash_key`).
    hash_namespace: Option<[u8; 32]>,
    /// The applied noise refresh.
    refresh: Option<PreparedRefresh<'a>>,
    /// The graph input name and length of a top refresh's `A'` key.
    refresh_matrix_key: Option<(String, usize)>,
}

/// A noise refresh prepared by preprocessing: its lowered children, the
/// public `B` of the shared unmask trapdoor, one preimage per decoder (in
/// decoder order) and their coefficient bound.
struct PreparedRefresh<'a> {
    declaration: crate::refresh::RefreshDeclaration,
    mask: Option<PublicKeyLowering<'a>>,
    fresh: PublicKeyLowering<'a>,
    smudges: Vec<PublicKeyLowering<'a>>,
    preimages: Vec<Preimage>,
    public: Mat,
    bound: BigUint,
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
