//! BGG+ constructions expressed directly with the declarative graph DSL.

pub mod attribute_encoding;
pub mod boolean;
pub mod circuit;
pub mod encoding;
pub mod lwe_lookup;
pub mod masked_decoder;
pub mod naive_vec;
pub mod noise_refresh;
pub mod public_key;
pub mod slot_operation;
pub mod tall_encoding;
pub mod tall_operational;
pub mod tall_rotation_encoding;
pub mod wee25_commitment;
pub mod wee25_opening;
pub mod wee25_public_parameters;

#[cfg(test)]
mod test_utils;

pub use attribute_encoding::{
    AttributeEncodingCompiler, AttributeEncodingWire, AttributeEvaluationError,
    AttributeMatrixEvaluation,
};
pub use boolean::{
    BggEncodingFamily, BggPublicKeyFamily, DynamicBooleanBggError,
    evaluate_boolean_encoding_layers, evaluate_boolean_public_key_layers,
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
pub use masked_decoder::{
    MASKED_DECODER_PREIMAGES, MaskedHighBitDecoderArtifacts, MaskedHighBitDecoderCompiler,
    MaskedHighBitDecoderError, MaskedHighBitDecoderOutputs, MaskedHighBitDecoderPreprocessingWires,
};
pub use naive_vec::{
    NaiveBggEncodingVecSampler, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecSampler,
    NaiveBggPublicKeyVecWire, NaiveBggVecCompiler, NaiveVecCompileError,
};
pub use noise_refresh::{
    NOISE_REFRESH_A_PRIME, NOISE_REFRESH_DECODER_PREIMAGES, NaiveBggNoiseRefreshArtifactWires,
    NaiveBggNoiseRefreshArtifacts, NaiveBggNoiseRefreshCompiler, NaiveBggNoiseRefreshError,
    NaiveBggNoiseRefreshPreprocessingWires,
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
pub use tall_operational::{
    TallNestedRnsDescriptor, TallOperationalError, TallOperationalEstimate, TallOperationalInputs,
    estimate_tall_nested_rns,
};
pub use tall_rotation_encoding::{
    TallRotationDirection, TallRotationEncodingArtifactNames, TallRotationEncodingArtifacts,
    TallRotationEncodingCompiler, TallRotationEncodingKey, TallRotationEncodingPreprocessingWires,
    TallRotationEncodingWires, required_tall_rotation_encodings,
};
pub use wee25_commitment::{
    Wee25CommitmentCompiler, Wee25CommitmentError, Wee25CommitmentTreeWire,
};
pub use wee25_opening::{
    WEE25_COMMITMENT, WEE25_COMMITMENT_NODES, WEE25_PUBLIC_B, WEE25_T_BOTTOM, WEE25_T_TOP,
    Wee25CommitmentArtifacts, Wee25PublicParameterArtifacts, Wee25PublicParameterWires,
    Wee25VerificationWire,
};
pub use wee25_public_parameters::{
    WEE25_PUBLIC_B_TRAPDOOR, Wee25PublicParameterCompiler, Wee25PublicParameterPreprocessingWires,
};

// The WEE25 commitment-backed lookup evaluator remains intentionally absent.
