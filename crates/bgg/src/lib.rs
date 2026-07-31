//! Graph-IR compilers for BGG+ wire bundles and polynomial circuits.

pub mod circuit;
pub mod digits_to_int;
pub mod encoding;
pub mod input_injection;
pub mod lwe_lookup;
pub mod masked_decoder;
pub mod naive_vec;
pub mod noise_refresh;
pub mod poly_encoding;
pub mod public_key;
pub mod sampler;
pub mod slot_transfer;
pub mod slot_transfer_artifact;
pub mod slot_transfer_poly_encoding;
pub mod slot_transfer_public_key;
pub mod wee25_commitment;
pub mod wee25_opening;
pub mod wee25_public_parameters;

pub use circuit::{
    AdvancedGateLowering, CircuitCompileError, CompositeAdvancedGateLowering, PolyCircuitCompiler,
};
pub use digits_to_int::{BggDigitsToIntCompiler, DigitsToIntCompileError};
pub use encoding::{BggEncodingCompiler, BggEncodingWire};
pub use input_injection::{
    DIAMOND_FINAL_PUBLIC, DIAMOND_FINAL_TRAPDOORS, DIAMOND_INITIAL_STATE,
    DiamondInputInjectionArtifacts, DiamondInputInjectionCompiler, DiamondInputInjectionError,
    DiamondInputInjectionPreprocessingWires, DiamondInputInjectionWires,
};
pub use lwe_lookup::{
    LweLookupArtifactNames, LweLookupArtifactWires, LweLookupArtifacts, LweLookupCompileError,
    LweLookupCompiler, LweLookupEncodingLowering, LweLookupIdentity, LweLookupInvocation,
    LweLookupPolyEncodingLowering, LweLookupPreprocessingWire, LweLookupPublicKeyLowering,
    LweLookupTable, NaiveLweLookupEncodingLowering, NaiveLweLookupInvocation,
    NaiveLweLookupPublicKeyLowering,
};
pub use masked_decoder::{
    MASKED_DECODER_PREIMAGES, MaskedHighBitDecoderArtifacts, MaskedHighBitDecoderCompiler,
    MaskedHighBitDecoderError, MaskedHighBitDecoderOutputs, MaskedHighBitDecoderPreprocessingWires,
};
pub use naive_vec::{
    NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire, NaiveBggVecCompiler, NaiveVecCompileError,
};
pub use noise_refresh::{
    NOISE_REFRESH_A_PRIME, NOISE_REFRESH_DECODER_PREIMAGES, NaiveBggNoiseRefreshArtifactWires,
    NaiveBggNoiseRefreshArtifacts, NaiveBggNoiseRefreshCompiler, NaiveBggNoiseRefreshError,
    NaiveBggNoiseRefreshPreprocessingWires,
};
pub use poly_encoding::{BggPolyEncodingCompiler, BggPolyEncodingWire, PolyEncodingCompileError};
pub use public_key::{BggPublicKeyCompiler, BggPublicKeyWire};
pub use sampler::{
    BggEncodingSampler, BggPolyEncodingSample, BggPolyEncodingSampler, BggPublicKeySampler,
    BggSampleError, BggSamplerLayout, NaiveBggEncodingVecSampler, NaiveBggPublicKeyVecSampler,
};
pub use slot_transfer::{NaiveBggSlotTransferCompiler, SlotFamilyCompileError};
pub use slot_transfer_artifact::{
    BggSlotTransferArtifactCompiler, BggSlotTransferBaseArtifacts, BggSlotTransferBaseWires,
    BggSlotTransferGateArtifacts, BggSlotTransferGateWires, BggSlotTransferPublicSlotWires,
    BggSlotTransferSlotArtifacts, BggSlotTransferSlotWires, SlotTransferArtifactError,
};
pub use slot_transfer_poly_encoding::BggPolySlotTransferLowering;
pub use slot_transfer_public_key::{BggSlotTransferGateRequest, BggSlotTransferPublicKeyLowering};
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
