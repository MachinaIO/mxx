use crate::{
    circuit::{PolyCircuit, gate::GateId},
    poly::Poly,
};

pub fn secret_inner_product<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    public_vec: &[GateId],
    secret_vec: &[GateId],
) -> GateId {
    assert_eq!(public_vec.len(), secret_vec.len(), "vector lengths must match");
    if public_vec.is_empty() {
        return circuit.const_zero_gate().as_single_wire();
    }

    // Multiply with public input on the left to keep BGG encoding semantics.
    let mut acc = circuit.mul_gate(public_vec[0], secret_vec[0]);
    for (&public_id, &secret_id) in public_vec.iter().zip(secret_vec.iter()).skip(1) {
        let prod = circuit.mul_gate(public_id, secret_id);
        acc = circuit.add_gate(acc, prod);
    }
    acc.as_single_wire()
}
