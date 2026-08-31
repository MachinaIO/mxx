//! Witness-encryption protocols expressed as validated declarative graphs.

pub mod diamond;

use mxx_gadgets::circuit::{BooleanCircuitData, BooleanCircuitShape};

pub trait WitnessEncryptionRuntime {
    type Ciphertext;
    type Message;
    type Error;

    fn shape(&self) -> &BooleanCircuitShape;

    fn encrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        message: &Self::Message,
    ) -> Result<Self::Ciphertext, Self::Error>;

    fn decrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        witness: &[bool],
        ciphertext: &Self::Ciphertext,
    ) -> Result<Self::Message, Self::Error>;
}
