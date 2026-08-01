// AKY24 iO and Diamond iO are disabled pending separate application cutovers to
// the declarative DSL and current IR/runtime APIs.
// pub mod aky24_io;
// pub mod diamond_io;
// mod graph;
// pub(crate) mod utils;

/// Common interface for indistinguishability obfuscation schemes.
pub trait Obfuscation {
    /// User-facing function descriptor accepted by the obfuscator.
    type FuncType;
    /// Persistable obfuscation object produced by preprocessing the function.
    type Obf;
    /// Plain input type accepted by online evaluation.
    type Input;
    /// Plain output type returned by online evaluation.
    type Output;

    /// Obfuscate `func` into an in-memory application value.
    fn obfuscation(&self, func: Self::FuncType) -> Self::Obf;

    /// Evaluate `obf` on `input`.
    fn eval(&self, obf: &Self::Obf, input: Self::Input) -> Self::Output;
}
