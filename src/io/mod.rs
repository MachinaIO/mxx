pub mod aky24_io;
pub mod diamond_io;
pub(crate) mod utils;

use std::path::Path;

pub use aky24_io::{Aky24IO, Aky24IOFuncType};
pub use diamond_io::{DiamondIO, DiamondIOFuncType, DiamondIOObf};

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

    /// Obfuscate `func`, storing any large preprocessing artifacts under `dir_path`.
    fn obfuscation(&self, dir_path: &Path, func: Self::FuncType) -> Self::Obf;

    /// Evaluate `obf` on `input`, reading persisted artifacts from `dir_path`.
    fn eval(&self, dir_path: &Path, obf: &Self::Obf, input: Self::Input) -> Self::Output;
}
