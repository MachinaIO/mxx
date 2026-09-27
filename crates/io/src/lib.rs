//! The indistinguishability-obfuscation interface ([`Obfuscation`]). Diamond iO and AKY24 iO were
//! removed during the DSL migration; their latest implementations remain on the `main` branch at
//! `d5d6fba26f1d20f11d4648a3fd1c9b35241ff4a9` (`src/io/diamond_io.rs` and `src/io/aky24_io.rs`).

/// Common interface for indistinguishability obfuscation schemes.
pub trait Obfuscation {
    /// User-facing function descriptor accepted by the obfuscator.
    type Function;
    /// Persistable obfuscation object produced by preprocessing the function.
    type Obfuscation;
    /// Plain input type accepted by online evaluation.
    type Input;
    /// Plain output type returned by online evaluation.
    type Output;
    /// Scheme-specific preprocessing or evaluation error.
    type Error;

    /// Obfuscate `func` into an in-memory application value.
    fn obfuscate(&mut self, function: &Self::Function) -> Result<Self::Obfuscation, Self::Error>;

    /// Evaluate `obfuscation` on `input`.
    fn evaluate(
        &mut self,
        obfuscation: &Self::Obfuscation,
        input: &Self::Input,
    ) -> Result<Self::Output, Self::Error>;
}
