/// Interface for indistinguishability obfuscation schemes.
pub trait IndisObf {
    /// Function type accepted by the obfuscator.
    type Func;
    /// Obfuscated representation produced by the obfuscator.
    type Obfuscation;
    /// Input type accepted by evaluation.
    type Input;
    /// Output type returned by evaluation.
    type Output;

    /// Obfuscate `func`.
    fn obfuscation(&self, func: Self::Func) -> Self::Obfuscation;

    /// Evaluate `obf` on `input`.
    fn eval(&self, obf: &Self::Obfuscation, input: Self::Input) -> Self::Output;
}
