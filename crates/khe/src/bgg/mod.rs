//! BGG+ constructions expressed directly with the declarative graph DSL.
//!
//! - `public_key` and `encoding` define BGG+ public keys and encodings: their wires, schemas, and
//!   samplers.
//! - `circuit`: `PolyCircuitCompiler` lowers a polynomial circuit to public-key or encoding graphs,
//!   in naive and Tall variants. Encoding compilation takes a decomposition provider called with
//!   each gate instance; it returns the preprocessed preimage for each multiplication, so the
//!   online encoding graph never builds public-key matrices or gadget decompositions. The producer
//!   must bind each cached decomposition to the right gate; shapes and bounds alone do not
//!   establish that binding.
//! - `boolean`: BGG+ evaluation of dynamic Boolean circuit families.
//! - `lwe_lookup`: LWE-based public lookup tables with preprocessing artifacts.
//! - `naive_vec`, `slot_operation`: per-slot vectors, slot transfer, and rotation.
//! - `tall_encoding`, `tall_rotation_encoding`: Tall encodings with one row per slot and their
//!   linear-transform preprocessing.

pub mod boolean;
pub mod circuit;
pub mod encoding;
pub mod lwe_lookup;
pub mod naive_vec;
pub mod public_key;
pub mod slot_operation;
pub mod tall_encoding;
pub mod tall_rotation_encoding;

pub(crate) fn static_ring_modulus(ring: &mxx_ir_core::RingRef) -> Option<num_bigint::BigUint> {
    use mxx_ir_core::{IntExpr, RingExpr};
    use num_bigint::BigUint;
    let RingExpr::Explicit { crt_moduli, .. } = ring.expression() else { return None };
    crt_moduli.iter().try_fold(BigUint::from(1u8), |product, prime| {
        let IntExpr::Const(prime) = prime else { return None };
        Some(product * prime.to_biguint()?)
    })
}

#[cfg(test)]
pub(crate) mod test_utils;

pub use boolean::{
    BggEncodingFamily, BggPublicKeyFamily, CircuitEncoding, CircuitEncodingType,
    DynamicBooleanBggError, evaluate_boolean_encoding_layers, evaluate_boolean_public_key_layers,
};
pub use circuit::{
    CircuitCompileError, NaiveEncodingSlotOperations, NaivePublicKeySlotOperations, NoPublicLookup,
    NoSlotOperations, PolyCircuitCompiler,
};
pub use encoding::{
    BggEncodingCompiler, BggEncodingSampler, BggEncodingType, BggEncodingWire, BggSampleError,
    BggSamplerLayout, EncodingCompileError,
};
pub use lwe_lookup::{
    LweLookupArtifactNames, LweLookupArtifactWires, LweLookupArtifacts, LweLookupCompileError,
    LweLookupCompiler, LweLookupEncodingLowering, LweLookupIdentity, LweLookupInvocation,
    LweLookupPreprocessingEntry, LweLookupPreprocessingLowering, LweLookupPreprocessingWires,
    LweLookupPublicKeyLowering, LweLookupTable, LweLookupTallEncodingLowering,
    NaiveLweLookupEncodingLowering, NaiveLweLookupInvocation, NaiveLweLookupPreprocessingEntry,
    NaiveLweLookupPreprocessingLowering, NaiveLweLookupPublicKeyLowering,
    bind_lwe_lookup_invocations, bind_naive_lwe_lookup_invocations, collect_lwe_lookup_identities,
    collect_lwe_lookup_identities_with_prefix,
};
pub use naive_vec::{
    NaiveBggEncodingVecSampler, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecSampler,
    NaiveBggPublicKeyVecWire, NaiveBggVecCompiler, NaiveVecCompileError,
};
pub use public_key::{
    BggPublicKeyCompiler, BggPublicKeySampler, BggPublicKeyType, BggPublicKeyWire,
};
pub use slot_operation::{
    BggSlotTransferArtifactCompiler, BggSlotTransferArtifactError, BggSlotTransferBaseArtifacts,
    BggSlotTransferBaseWires, BggSlotTransferGateArtifacts, BggSlotTransferGateRequest,
    BggSlotTransferGateWires, BggSlotTransferPublicKeyLowering, BggSlotTransferPublicSlotWires,
    BggSlotTransferSlotArtifacts, BggSlotTransferSlotWires, BggTallSlotLowering,
    BggTallSlotPublicKeyLowering, NaiveBggSlotTransferCompiler, SlotFamilyCompileError,
};
pub use tall_encoding::{
    BggTallEncodingCompiler, BggTallEncodingSample, BggTallEncodingSampler, BggTallEncodingWire,
    BggTallPlaintext, TallCompileError,
};
pub use tall_rotation_encoding::{
    TALL_ANCHOR_REDUCE_MATRIX_ARTIFACT, TallLinearTransformEncodingWires,
    TallLinearTransformPublicWires, TallRotationEncodingArtifactNames,
    TallRotationEncodingArtifacts, TallRotationEncodingCompiler, TallRotationEncodingKey,
    TallRotationEncodingPreprocessingWires, required_tall_anchor_reduce_encoding,
    required_tall_rotation_encodings,
};
