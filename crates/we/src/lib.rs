//! Witness-encryption applications built from the declarative graph DSL.

pub mod diamond;

use mxx_gadgets::circuit::PolyCircuit;
use mxx_primitives::poly::Poly;

/// Common interface for witness encryption schemes.
pub trait WitnessEnc<P: Poly> {
    type Msg;
    type Inst;
    type Wtns;
    type Ciphertext;
    type Error;

    fn enc(
        &mut self,
        msg: &Self::Msg,
        circuit: PolyCircuit<P>,
        instance: &Self::Inst,
    ) -> Result<Self::Ciphertext, Self::Error>;

    fn dec(
        &mut self,
        ct: &Self::Ciphertext,
        witness: &Self::Wtns,
    ) -> Result<Self::Msg, Self::Error>;
}
