//! The binding of TFHE to the public-key homomorphic encryption syntax ([`crate::spec`]).
//!
//! | Syntax | TFHE |
//! | --- | --- |
//! | messages | one bit, the integer `0` or `1` |
//! | depth | [`Depth::Unbounded`] |
//! | gate | NAND on bits |
//! | `KeyGen` | [`TfheParams::keygen`]: the public key is the ring-LWE sample and the key-switch key, the evaluation key the bootstrapping and key-switch keys, the secret key the LWE secret |
//! | `Enc` | [`TfheParams::encrypt`]: ring-LWE encryption under the public key, sample extraction, and key switching |
//! | gate on ciphertexts | [`TfheParams::nand`]: a bootstrapped NAND |
//! | `Dec` | [`TfheParams::decrypt`] |
//!
//! Every gate bootstraps, so the circuits have no depth limit and a ciphertext has one type
//! whatever gates returned it; only its declared noise bound differs.
use crate::{
    BootstrappingKey, KeySwitchKey, LweCiphertext, TfheParams,
    spec::{Correctness, Depth, Fhe, Statement},
};
use mxx_dsl::{Family, Int, IntType, Mat};
use mxx_ir_core::protocol::InputValueContract;

/// TFHE at `params`.
pub struct TfheScheme<'a> {
    pub params: &'a TfheParams,
}

/// The bound `2^-k` on the failure probability of one fresh NAND gate that this crate proves.
pub const NAND_FAILURE_PROBABILITY_LOG2: u32 = 100;

impl TfheScheme<'_> {
    /// The statement this crate proves for the scheme: one NAND gate on fresh ciphertexts
    /// decrypts correctly except with probability at most `2^-NAND_FAILURE_PROBABILITY_LOG2`.
    /// No security statement is claimed yet.
    pub fn statement(&self) -> Statement {
        Statement {
            correctness: Correctness::AverageCase {
                failure_probability_log2: NAND_FAILURE_PROBABILITY_LOG2,
            },
            security: None,
        }
    }
}

impl Fhe for TfheScheme<'_> {
    type Message = Int;
    type PublicKey = (Mat, Mat, KeySwitchKey);
    type EvaluationKey = (BootstrappingKey, KeySwitchKey);
    type SecretKey = Family<Int>;
    type Ciphertext = LweCiphertext;

    fn depth(&self) -> Depth {
        Depth::Unbounded
    }

    fn message_space(&self) -> (IntType, InputValueContract) {
        (IntType, InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() })
    }

    fn key_generation(
        &self,
    ) -> ((Mat, Mat, KeySwitchKey), (BootstrappingKey, KeySwitchKey), Family<Int>) {
        let keys = self.params.keygen().unwrap();
        (
            (keys.public_a, keys.public_b, keys.key_switch_key.clone()),
            (keys.bootstrapping_key, keys.key_switch_key),
            keys.lwe_secret,
        )
    }

    fn encrypt(&self, public_key: &(Mat, Mat, KeySwitchKey), message: &Int) -> LweCiphertext {
        let (a, b, key_switch_key) = public_key;
        self.params.encrypt(a, b, key_switch_key, message).unwrap()
    }

    fn plain_gate(&self, x: &Int, y: &Int) -> Int {
        Int::constant(1).sub(x.clone().mul(y.clone()))
    }

    fn eval_gate(
        &self,
        evaluation_key: &(BootstrappingKey, KeySwitchKey),
        x: &LweCiphertext,
        y: &LweCiphertext,
    ) -> LweCiphertext {
        let (bootstrapping_key, key_switch_key) = evaluation_key;
        self.params.nand(x, y, bootstrapping_key, key_switch_key).unwrap()
    }

    fn decrypt(&self, secret_key: &Family<Int>, ciphertext: &LweCiphertext) -> Int {
        self.params.decrypt(secret_key, ciphertext).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::TfheScheme;
    use crate::{spec, utils};

    #[test]
    fn circuit_with_unread_input_exports() {
        let tfhe = utils::tfhe_params();
        let scheme = TfheScheme { params: &tfhe };
        // The output is input 0; no gate reads input 1, so neither graph reads it.
        let circuit = spec::Circuit::new(2, vec![], 0).unwrap();
        let directory = tempfile::tempdir().unwrap();
        spec::lean::export(&scheme, &circuit, &scheme.statement(), &[], directory.path()).unwrap();
    }
}
