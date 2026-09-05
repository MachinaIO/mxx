// AKY24 iO is temporarily disabled until its full cascade receives end-to-end validation.
// pub mod aky24;
// Application modules are disabled until their protocol-specific hard-bound
// recurrences or certified correctness checkers are available.

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
