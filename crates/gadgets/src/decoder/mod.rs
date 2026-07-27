//! Shared masked-decoder building blocks.
//!
//! The decoder module contains protocol-independent pieces used by
//! noise-refresh, AKY24, and DiamondIO masked decode paths. Protocols still own
//! seed evolution and selected-branch PRG logic; this module owns mask decrypt
//! circuits, artifact persistence interfaces, mask-bit simulation helpers, and
//! benchmark scaling utilities.

pub mod artifact;
pub mod bench;
pub mod mask_circuit;
pub mod masked_high_bit;
pub mod prg;
pub mod simulation;

/// Offline/online split for protocol decoders.
///
/// Implementations should build and persist the decoder material in
/// `preprocess`, then consume the corresponding online encoding and stored
/// material in `online_decode`.
pub trait Decoder {
    type PreprocessInput;
    type PreprocessOutput;
    type OnlineInput;
    type Output;

    fn preprocess(&self, input: Self::PreprocessInput) -> Self::PreprocessOutput;

    fn online_decode(&self, input: Self::OnlineInput) -> Self::Output {
        self.online_decode_range(input, 0)
    }

    fn online_decode_range(
        &self,
        input: Self::OnlineInput,
        initial_decoder_offset: usize,
    ) -> Self::Output;
}
