//! The syntax of a public-key homomorphic encryption scheme and `Eval` of a circuit.

use super::{Circuit, Depth};
use mxx_dsl::GraphValue;
use mxx_ir_core::protocol::InputValueContract;

/// A public-key homomorphic encryption scheme built in the DSL.
///
/// Each algorithm builds its output from DSL values. The type of a ciphertext may depend on the
/// gates that returned it, such as the level of BGV, and the ciphertext value carries it.
/// [`Fhe::eval_gate`] is only applied within the scheme's [`Depth`], and may panic beyond it.
pub trait Fhe {
    /// A message: the value on a circuit's wire.
    type Message: GraphValue;
    type PublicKey: GraphValue;
    type EvaluationKey: GraphValue;
    type SecretKey: GraphValue;
    type Ciphertext: GraphValue;

    /// The depth of the circuits `Eval` runs.
    fn depth(&self) -> Depth;
    /// The message space: the type of a message and the messages correctness covers.
    fn message_space(&self) -> (<Self::Message as GraphValue>::Schema, InputValueContract);
    /// `KeyGen`.
    fn key_generation(&self) -> (Self::PublicKey, Self::EvaluationKey, Self::SecretKey);
    /// `Enc`.
    fn encrypt(&self, public_key: &Self::PublicKey, message: &Self::Message) -> Self::Ciphertext;
    /// The gate on messages, built in the DSL without samplers.
    fn plain_gate(&self, x: &Self::Message, y: &Self::Message) -> Self::Message;
    /// The gate on ciphertexts: one gate of `Eval`.
    fn eval_gate(
        &self,
        evaluation_key: &Self::EvaluationKey,
        x: &Self::Ciphertext,
        y: &Self::Ciphertext,
    ) -> Self::Ciphertext;
    /// `Dec`.
    fn decrypt(&self, secret_key: &Self::SecretKey, ciphertext: &Self::Ciphertext)
    -> Self::Message;
}

/// `Eval` of `circuit` on ciphertexts of its inputs, gate by gate.
pub fn evaluate<S: Fhe>(
    scheme: &S,
    evaluation_key: &S::EvaluationKey,
    circuit: &Circuit,
    inputs: &[S::Ciphertext],
) -> S::Ciphertext {
    circuit.run(inputs, |x, y| scheme.eval_gate(evaluation_key, x, y))
}

/// The plaintext value of `circuit` on `inputs`.
pub fn plain_eval<S: Fhe>(scheme: &S, circuit: &Circuit, inputs: &[S::Message]) -> S::Message {
    circuit.run(inputs, |x, y| scheme.plain_gate(x, y))
}
