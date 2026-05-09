pub mod diamond_we;

use crate::{circuit::PolyCircuit, poly::Poly};

pub use diamond_we::{DiamondWE, DiamondWECiphertext};

/// Common interface for witness encryption schemes.
pub trait WitnessEnc<P: Poly> {
    type Msg;
    type Inst;
    type Wtns;
    type Ciphertext;

    fn enc(
        &self,
        msg: &Self::Msg,
        circuit: PolyCircuit<P>,
        instance: &Self::Inst,
    ) -> Self::Ciphertext;

    fn dec(&self, ct: &Self::Ciphertext, witness: &Self::Wtns) -> Self::Msg;
}
