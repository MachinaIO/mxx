//! The implementation-agnostic syntax of public-key homomorphic encryption and its binding to DSL
//! graphs.
//!
//! The syntax is that of the literature: `KeyGen` returns a public key, an evaluation key, and a
//! secret key; `Enc` encrypts a message under the public key; `Eval` evaluates a circuit on
//! ciphertexts with the evaluation key; `Dec` decrypts with the secret key. A scheme evaluates the
//! circuits up to its [`Depth`], a finite bound for a leveled scheme such as BGV or
//! [`Depth::Unbounded`] for a bootstrapped one such as TFHE. Correctness is stated for one
//! circuit: decrypting its evaluation on fresh encryptions gives its plaintext value, either in the
//! worst case (every execution) or on average (failure with at most a declared probability). Both
//! statements and IND-CPA security are defined once, in Lean, by `MxxFhe`
//! (`crates/fhe/lean/spec`); [`lean::export`] writes the graphs of an [`Fhe`] scheme and a circuit
//! and the claims of a [`Statement`], `CorrectnessClaim` and `BoundedSecurityClaim`, that apply
//! those definitions to them.
//!
//! A circuit is a [`Circuit`] of two-input gates, the same for every scheme. A scheme gives each
//! algorithm as a function on DSL values ([`Fhe`]) and its gate twice, on messages
//! ([`Fhe::plain_gate`]) and on ciphertexts ([`Fhe::eval_gate`]); [`evaluate`] and [`plain_eval`]
//! run a circuit with them. The type of a ciphertext may depend on the gates that returned it, such
//! as the level of BGV, and the ciphertext value carries it, so no algorithm takes it as an
//! argument.
//!
//! [`lean::export`] builds one graph per algorithm with the input and output names of [`roles`]:
//!
//! | Graph | Inputs | Outputs |
//! | --- | --- | --- |
//! | `KeyGen` | none | `public_key`, `evaluation_key`, `secret_key` |
//! | `Enc` | `message`, `public_key` | `ciphertext` |
//! | `Eval` of the circuit | `ciphertext_0`, ..., `evaluation_key` | `ciphertext` |
//! | `Dec` of each ciphertext type | `ciphertext`, `secret_key` | `message` |
//! | `plain_eval` of the circuit | `message_0`, ... | `message` |
//!
//! The security statement names only its allowed assumptions
//! ([`mxx_ir_core::security::AssumptionId`]) and a cost bound: it asks for reductions whose
//! simulators are IR programs of bounded cost. A proof supplies them as [`Reduction`]s, two graphs
//! per assumption:
//!
//! | Graph | Inputs | Outputs |
//! | --- | --- | --- |
//! | setup | the challenge components of the assumption | `public_key`, `evaluation_key`, any state |
//! | challenge | `message`, any setup outputs by name | `ciphertext` of the type `Enc` returns |

mod circuit;
mod graphs;
pub mod lean;
mod scheme;
mod statement;

pub use circuit::{Circuit, Depth};
pub use graphs::{BindingError, graphs};
pub use scheme::{Fhe, evaluate, plain_eval};
pub use statement::{Correctness, Reduction, SecurityStatement, Statement};

/// The input and output names of the algorithm graphs.
pub mod roles {
    pub const PUBLIC_KEY: &str = "public_key";
    pub const EVALUATION_KEY: &str = "evaluation_key";
    pub const SECRET_KEY: &str = "secret_key";
    pub const MESSAGE: &str = "message";
    pub const CIPHERTEXT: &str = "ciphertext";

    /// The `index`-th ciphertext input of a gate.
    pub fn ciphertext(index: usize) -> String {
        format!("ciphertext_{index}")
    }

    /// The `index`-th message input of a gate's plaintext function.
    pub fn message(index: usize) -> String {
        format!("message_{index}")
    }
}
