// Diamond WE is disabled pending a separate application cutover to the
// declarative DSL and current IR/runtime APIs.
// pub mod diamond_we;

use mxx_gadgets::circuit::PolyCircuit;
use mxx_primitives::poly::Poly;

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
